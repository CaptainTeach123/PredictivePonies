"""Walk-forward comparison of FormFav model variants against the market.

    python scripts/formfav_experiments.py [--store data/raw/formfav] [--test-share 0.4]

For every test day (the newest ``--test-share`` of race days) each variant is
refitted on all races from earlier days only, then scored on that day: no
fitted number ever sees a race from its own day or later [A-08].

Variants are conditional logits (model.py) that differ only in their inputs.
"Residual" variants include the market's log probability as an input, so the
form features only have to explain what the price misses (Benter 1994).

Prints summary numbers only (safe for public CI logs); never race data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ponies import config  # noqa: E402
from ponies.formfav import FF_FEATURE_NAMES, load_store  # noqa: E402
from ponies.model import ConditionalLogit, _race_codes, race_softmax, win_targets  # noqa: E402

EXTRA = ["place_rate_distance", "place_rate_track", "place_rate_condition", "track_distance_win_rate",
         "peak_rating_gap", "best_class_won_rel", "in_optimal_range", "runs_this_prep",
         "best_recent_finish", "avg_finish_last6", "decorator_sentiment", "leader_style"]
HISTORY = ["prev_margin_log", "prev_market_logp", "prev_won", "days_since_prev_log", "has_prev"]
MARKET_EXTRA = ["price_move", "late_move", "mk_logp_sq", "book_spread"]


def log_probs(dec: pd.Series, race_id: pd.Series) -> pd.Series:
    inv = 1 / dec
    return np.log(inv / inv.groupby(race_id).transform("sum"))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["won"].notna()].copy()
    winners = df.groupby("race_id")["won"].transform("sum")
    df = df[(winners == 1) & (df.groupby("race_id")["race_id"].transform("size") >= 2)].copy()
    df["dec"] = df["morning_line"].astype(float) + 1
    ok = df.groupby("race_id")["dec"].transform(lambda s: s.notna().all())
    df = df[ok].copy()
    df["mk_logp"] = log_probs(df["dec"], df["race_id"])
    df["mk_logp_sq"] = df["mk_logp"] ** 2
    df["best"] = df["market_best"].fillna(df["dec"])
    df["book_spread"] = np.log(df["best"] / df["dec"])
    df["late_move"] = np.log(df["market_t30"] / df["dec"])   # > 0: firmed in the last 30 min
    sp = df["final_odds"].astype(float) + 1
    sp_ok = sp.notna().groupby(df["race_id"]).transform("all")
    df["sp_logp"] = log_probs(sp.where(sp_ok), df["race_id"]).where(sp_ok)
    df["prev_margin_log"] = np.log1p(df["prev_margin"])
    df["days_since_prev_log"] = np.log1p(df["days_since_prev"])
    df["has_prev"] = df["prev_market_logp"].notna().astype(float)
    return df.reset_index(drop=True)


def decorator_columns(df: pd.DataFrame, top: int = 15) -> list[str]:
    counts = df["decorators"].fillna("").str.split(",").explode().value_counts()
    names = [n for n in counts.index if n][:top]
    for n in names:
        df[f"deco_{n}"] = df["decorators"].fillna("").str.split(",").map(lambda xs, n=n: float(n in xs))
    return [f"deco_{n}" for n in names]


def race_losses(p: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    codes, n = _race_codes(df["race_id"])
    y = win_targets(df["won"].to_numpy(), codes, n)
    return np.bincount(codes, weights=-y * np.log(np.clip(p, 1e-15, 1)), minlength=n)


def walk_forward(df: pd.DataFrame, features: list[str], test_days: list, l2: float | None,
                 min_train_races: int = 300) -> np.ndarray:
    """Out-of-sample probability for every runner on a test day (NaN elsewhere)."""
    out = np.full(len(df), np.nan)
    for d in test_days:
        train, test = df[df["date"] < d], df[df["date"] == d]
        if train["race_id"].nunique() < min_train_races or test.empty:
            continue
        m = ConditionalLogit(features, l2=l2, unpenalized=("mk_logp",)).fit(train)
        out[test.index] = m.predict_proba(test)
    return out


def forward_select(df: pd.DataFrame, pool: list[str], days: list, max_features: int) -> list[str]:
    """Add one input at a time (to the market log prob) while walk-forward log
    loss on ``days`` improves by more than 0.001 per race."""
    def loss(feats):
        p = walk_forward(df, ["mk_logp"] + feats, days, None)
        ok = ~np.isnan(p)
        return float(race_losses(p[ok], df[ok]).mean())
    chosen, best = [], loss([])
    while len(chosen) < max_features:
        trials = {f: loss(chosen + [f]) for f in pool if f not in chosen}
        f, val = min(trials.items(), key=lambda kv: kv[1])
        if val > best - 0.001:
            break
        chosen.append(f)
        best = val
    return chosen


def bets(p: np.ndarray, price: np.ndarray, won: np.ndarray, thr: float) -> dict:
    sel = p * price - 1 > thr
    n = int(sel.sum())
    if not n:
        return {"bets": 0}
    ret = np.where(won[sel] == 1, price[sel] - 1, -1.0)
    return {"bets": n, "wins": int((won[sel] == 1).sum()), "roi": round(float(ret.mean()), 4),
            "roi_se": round(float(ret.std(ddof=1) / math.sqrt(n)), 4) if n > 1 else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path(config.RAW_DIR / "formfav"))
    ap.add_argument("--test-share", type=float, default=0.4)
    ap.add_argument("--max-select", type=int, default=6)
    a = ap.parse_args()

    df = prepare(load_store(a.store))
    days = sorted(df["date"].unique())
    if len(days) < 5:
        print("too few race days")
        return 0
    deco = decorator_columns(df)
    test_days = days[int(len(days) * (1 - a.test_share)):]
    print(json.dumps({"races": int(df["race_id"].nunique()), "runners": len(df), "days": len(days),
                      "test_days": len(test_days),
                      "coverage": {c: round(float(df[c].notna().mean()), 3) for c in
                                   ("market_best", "market_t30", "prev_market_logp", "sp_logp")}}))

    form = FF_FEATURE_NAMES
    pool = MARKET_EXTRA + form + EXTRA + deco + HISTORY
    variants = {
        "market_calibrated": (["mk_logp"], None),
        "form_only": (form, None),
        "residual_form": (["mk_logp"] + form, None),
    }
    for l2 in (0.03, 0.1, 0.3, 1.0):
        variants[f"residual_form_l2_{l2}"] = (["mk_logp"] + form, l2)
        variants[f"residual_all_l2_{l2}"] = (["mk_logp"] + pool, l2)
    # greedy forward selection: chosen on the first half of the test days,
    # judged only on the second half (reported separately below)
    half = len(test_days) // 2
    chosen = forward_select(df, pool, test_days[:half], max_features=a.max_select)
    print(json.dumps({"selected_on_first_half": chosen}))
    probs = {name: walk_forward(df, f, test_days, l2) for name, (f, l2) in variants.items()}
    have = ~np.isnan(probs["market_calibrated"])
    S = df[have]
    codes, k = _race_codes(S["race_id"])
    p_mk = race_softmax(S["mk_logp"].to_numpy(), codes, k)
    base = race_losses(p_mk, S)
    won = S["won"].to_numpy()
    report = {"test_races": int(k), "test_from": str(S["date"].min().date()) if k else None,
              "uniform_log_loss": round(float(np.mean(np.log(S.groupby("race_id").size()))), 4),
              "market_log_loss": round(float(base.mean()), 4)}
    sp_ok = S["sp_logp"].notna().to_numpy()
    if sp_ok.all():
        report["sp_log_loss"] = round(float(race_losses(race_softmax(S["sp_logp"].to_numpy(), codes, k), S).mean()), 4)
    else:
        report["sp_races_missing"] = int(S.loc[~sp_ok, "race_id"].nunique())
    report["market_bets_best_price"] = bets(p_mk, S["best"].to_numpy(), won, 0.0)
    rows = {}
    for name, p in probs.items():
        p = p[have]
        ll = race_losses(p, S)
        d = base - ll
        top = pd.Series(p).groupby(codes).transform("max").to_numpy() == p
        rows[name] = {
            "log_loss": round(float(ll.mean()), 4),
            "gain_vs_market": round(float(d.mean()), 4),
            "se": round(float(d.std(ddof=1) / math.sqrt(len(d))), 4),
            "top_pick_win": round(float(won[top].sum() / k), 4),
            "bets_best_price": {str(t): bets(p, S["best"].to_numpy(), won, t) for t in (0.0, 0.1, 0.25)},
            "bets_median_price": bets(p, S["dec"].to_numpy(), won, 0.1),
        }
    report["market_top_pick_win"] = round(float(won[pd.Series(p_mk).groupby(codes).transform("max").to_numpy() == p_mk].sum() / k), 4)
    print(json.dumps(report))
    for name, r in rows.items():  # one line per variant
        print(json.dumps({"variant": name, **r}))
    # honest check of the selected set: second half only
    late = test_days[half:]
    L = df[df["date"].isin(late)]
    pm = walk_forward(df, ["mk_logp"], late, None)
    ps = walk_forward(df, ["mk_logp"] + chosen, late, None)
    ok = ~np.isnan(pm)
    L, pm, ps = df[ok], pm[ok], ps[ok]
    c2, k2 = _race_codes(L["race_id"])
    raw = race_losses(race_softmax(L["mk_logp"].to_numpy(), c2, k2), L)
    for name, p in (("market_calibrated", pm), ("selected", ps)):
        d = raw - race_losses(p, L)
        print(json.dumps({"second_half": name, "races": int(k2), "gain_vs_market": round(float(d.mean()), 4),
                          "se": round(float(d.std(ddof=1) / math.sqrt(len(d))), 4),
                          "bets_best_price_0.1": bets(p, L["best"].to_numpy(), L["won"].to_numpy(), 0.1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

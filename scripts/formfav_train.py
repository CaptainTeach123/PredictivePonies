"""Train and evaluate the win model on collected FormFav races.

    python scripts/formfav_train.py [--store data/raw/formfav] [--out model.json]

Chronological thirds by race date [A-08][A-37]:
  A (oldest 60%)  fit the conditional logit on FormFav card features
  B (next 20%)    fit the Benter blend: softmax(c1 * model log-prob + c2 * log market prob)
  C (newest 20%)  score everything; nothing was fitted on C

Markets compared: the pre-race bookmaker price [A-42] and the starting price.
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
from ponies import config, odds  # noqa: E402
from ponies.formfav import FF_FEATURE_NAMES, load_store  # noqa: E402
from ponies.model import ConditionalLogit, _race_codes, race_softmax, win_targets  # noqa: E402


def market_logp(df: pd.DataFrame, col: str) -> np.ndarray:
    """log implied probability (normalised per race); NaN where odds missing."""
    dec = df[col].map(odds.to_decimal).to_numpy(dtype=float)
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby("race_id").indices.items():
        out[idx] = np.log(odds.implied_probabilities(dec[idx]))
    return out


def race_losses(p: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Per-race log loss -sum(y log p)."""
    codes, n = _race_codes(df["race_id"])
    y = win_targets(df["won"].to_numpy(), codes, n)
    return np.bincount(codes, weights=-y * np.log(np.clip(p, 1e-15, 1)), minlength=n)


def compare(name: str, mine: np.ndarray, base: np.ndarray) -> dict:
    d = base - mine  # > 0: we beat the baseline
    return {"vs": name, "gain_per_race": round(float(d.mean()), 4),
            "se": round(float(d.std(ddof=1) / math.sqrt(len(d))), 4), "races": int(len(d))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path(config.RAW_DIR / "formfav"))
    ap.add_argument("--out", type=Path, default=None, help="write the fitted model (weights only) here")
    ap.add_argument("--predict", type=Path, default=None,
                    help="write predictions for upcoming races (licensed data: keep private)")
    a = ap.parse_args()

    df = load_store(a.store)
    if df.empty:
        print("store is empty")
        return 1
    upcoming = df[df["won"].isna() & (df["date"] >= pd.Timestamp.now().normalize() - pd.Timedelta(days=1))].copy()
    df = df[df["won"].notna()]
    winners = df.groupby("race_id")["won"].transform("sum")
    df = df[(winners == 1) & (df.groupby("race_id")["race_id"].transform("size") >= 2)].copy()
    df["mk_logp"] = market_logp(df, "morning_line")   # pre-race bookmaker close
    df["sp_logp"] = market_logp(df, "final_odds")     # starting price
    complete = df.groupby("race_id")["mk_logp"].transform(lambda s: s.notna().all())
    df = df[complete].reset_index(drop=True)
    races = df.groupby("race_id")["date"].first().sort_values()
    n = len(races)
    print(json.dumps({"usable_races": n, "runners": int(len(df)),
                      "dates": [str(races.min().date()), str(races.max().date())] if n else None}))
    if n < 150:
        print("too few complete races with a market to train and test yet")
        return 0

    cut_b, cut_c = races.index[int(n * 0.6)], races.index[int(n * 0.8)]
    order = {rid: i for i, rid in enumerate(races.index)}
    pos = df["race_id"].map(order)
    A, B, C = df[pos < order[cut_b]], df[(pos >= order[cut_b]) & (pos < order[cut_c])], df[pos >= order[cut_c]]

    model = ConditionalLogit(FF_FEATURE_NAMES).fit(A)

    def blend_frame(part):
        return pd.DataFrame({"race_id": part["race_id"].to_numpy(), "won": part["won"].to_numpy(),
                             "model_logp": np.log(np.clip(model.predict_proba(part), 1e-12, 1)),
                             "market_logp": part["mk_logp"].to_numpy()})

    blend = ConditionalLogit(["model_logp", "market_logp"])
    blend.fit(blend_frame(B))
    # unscaled weights: score = c1 * model_logp + c2 * market_logp (+ constant)
    c = (blend.weights / blend.stds).tolist()

    Cb = blend_frame(C)
    codes, k = _race_codes(C["race_id"])
    p_model = model.predict_proba(C)
    p_blend = blend.predict_proba(Cb)
    p_mk = race_softmax(C["mk_logp"].to_numpy(), codes, k)
    ll = {name: race_losses(p, C) for name, p in
          (("model", p_model), ("blend", p_blend), ("market_pre_race", p_mk))}
    sp_ok = C.groupby("race_id")["sp_logp"].transform(lambda s: s.notna().all()).to_numpy()
    Csp = C[sp_ok]
    if len(Csp):
        c2, k2 = _race_codes(Csp["race_id"])
        ll_sp = race_losses(race_softmax(Csp["sp_logp"].to_numpy(), c2, k2), Csp)
        ll_bl_sp = race_losses(p_blend[sp_ok], Csp)

    # value bets on C: blend edge vs the pre-race price, settled at that price [A-13][A-14]
    dec = C["morning_line"].map(odds.to_decimal).to_numpy(dtype=float)
    edge = p_blend * dec - 1
    won = C["won"].to_numpy()
    bets = edge > config.VALUE_EDGE_THRESHOLD
    profit = float(np.sum(np.where(won[bets] == 1, dec[bets] - 1, -1.0))) if bets.any() else 0.0

    report = {
        "train_races_A": int(A["race_id"].nunique()), "blend_races_B": int(B["race_id"].nunique()),
        "test_races_C": int(k), "test_from": str(C["date"].min().date()),
        "log_loss_C": {k_: round(float(v.mean()), 4) for k_, v in ll.items()},
        "uniform_log_loss_C": round(float(np.mean(np.log(C.groupby("race_id").size()))), 4),
        "blend_weights": {"model_logp": round(c[0], 3), "market_logp": round(c[1], 3)},
        "blend_vs_pre_race_market": compare("market_pre_race", ll["blend"], ll["market_pre_race"]),
        "model_vs_pre_race_market": compare("market_pre_race", ll["model"], ll["market_pre_race"]),
        "top_pick_win_rate_C": {
            "model": round(float(won[pd.Series(p_model).groupby(codes).transform("max").to_numpy() == p_model].sum() / k), 4),
            "market": round(float(won[pd.Series(p_mk).groupby(codes).transform("max").to_numpy() == p_mk].sum() / k), 4),
        },
        "value_bets_C": {"bets": int(bets.sum()), "wins": int((won[bets] == 1).sum()),
                         "profit_units": round(profit, 2),
                         "roi": round(profit / bets.sum(), 4) if bets.any() else None},
    }
    if len(Csp):
        report["vs_starting_price"] = {"sp_log_loss": round(float(ll_sp.mean()), 4),
                                       **compare("starting_price", ll_bl_sp, ll_sp)}
    print(json.dumps(report, indent=1))
    final = ConditionalLogit(FF_FEATURE_NAMES).fit(df)
    if a.out:
        a.out.write_text(json.dumps({"model": final.to_dict(), "blend": c}, indent=1))
    if a.predict:
        g = report["blend_vs_pre_race_market"]
        value_ok = g["gain_per_race"] > 2 * g["se"]  # [A-35]
        cards = predict_upcoming(final, c, upcoming, value_ok)
        a.predict.parent.mkdir(parents=True, exist_ok=True)
        a.predict.write_text(json.dumps({"value_flags_enabled": value_ok, "evaluation": report, "races": cards}))
        print(json.dumps({"upcoming_races_predicted": len(cards), "value_flags_enabled": value_ok,
                          "value_flags": sum(r["value"] for c_ in cards for r in c_["runners"])}))
    return 0


def predict_upcoming(model: ConditionalLogit, c: list, up: pd.DataFrame, value_ok: bool) -> list[dict]:
    """Blended win probabilities for upcoming races with a current price for every runner."""
    if up.empty:
        return []
    up = up.copy()
    up["mk_logp"] = market_logp(up, "morning_line")
    up = up[up.groupby("race_id")["mk_logp"].transform(lambda s: s.notna().all())].reset_index(drop=True)
    if up.empty:
        return []
    codes, k = _race_codes(up["race_id"])
    p_model = model.predict_proba(up)
    score = c[0] * np.log(np.clip(p_model, 1e-12, 1)) + c[1] * up["mk_logp"].to_numpy()
    p_blend = race_softmax(score, codes, k)
    p_mk = race_softmax(up["mk_logp"].to_numpy(), codes, k)
    dec = up["morning_line"].map(odds.to_decimal).to_numpy(dtype=float)
    up["p_model"], up["p_blend"], up["p_market"], up["price"] = p_model, p_blend, p_mk, dec
    out = []
    for rid, g in up.groupby("race_id", sort=False):
        g = g.sort_values("p_blend", ascending=False)
        out.append({"race_id": rid, "date": str(g["date"].iloc[0].date()), "track": g["track"].iloc[0],
                    "race": int(g["race_number"].iloc[0]), "runners": [{
                        "number": r.program, "horse": r.horse_name, "price": round(r.price, 2),
                        "win_blend": round(r.p_blend, 4), "win_model": round(r.p_model, 4),
                        "win_market": round(r.p_market, 4), "edge": round(r.p_blend * r.price - 1, 4),
                        "value": bool(value_ok and r.p_blend * r.price - 1 > config.VALUE_EDGE_THRESHOLD),
                    } for r in g.itertuples()]})
    return out


if __name__ == "__main__":
    sys.exit(main())

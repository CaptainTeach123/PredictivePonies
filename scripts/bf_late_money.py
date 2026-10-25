"""Experiment "late_money": late money / price momentum on the Betfair Exchange
(AU WIN markets from data/raw/betfair/split/au_win.parquet).

    python scripts/bf_late_money.py [--end 2018-06-01] [--burn-in 6]

For every runner that traded 30 minutes before the start (ltp_t30 present,
~39% of AU runners, ~79% of markets have at least one such runner, ~18% have
all runners traded):

    move = log(ltp_t30 / ltp_pre)      (> 0 = price shortened = "firmed")

Sections (all numbers printed as JSON lines, summary numbers only):
  A. calibration of ltp_pre for runners / markets WITH vs WITHOUT t30 trading
  B. calibration of ltp_pre and ROI by move bucket (deciles, fixed cuts, and
     fixed cuts x price band); commission 0 / 5% / 8% on net winnings [A-54];
     halves of the period; ROI at Betfair SP and at one tick below ltp_pre as
     realistic-execution checks
  C. walk-forward ConditionalLogit on [log Shin prob at ltp_pre, move,
     move*|move|] with the market term unpenalised, monthly expanding-window
     refits, per-race log-loss gain vs the market, and bets where
     model_prob * ltp_pre > 1 + k. Three universes: markets where every runner
     traded at t30; markets with any t30 trading, missing move imputed with the
     race median; the same with missing move = 0. Bets are also shown with the
     market restricted to a coherent book (sum of 1/ltp_pre in [0.98, 1.05]),
     because an under-round book means the last traded prices were stale.
  D. Benjamini-Hochberg over every bet cell tested, the both-halves > 2 SE
     rule, and an opposite-half development check (pick the best cell in one
     half, evaluate it in the other).

Nothing here is a guaranteed available price: ltp_pre is the last traded
price before the off (one-minute snapshots on the Basic plan), so ROI at BSP
and at one tick worse are the realistic lines.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ponies import odds  # noqa: E402
from ponies.betfair import net_return  # noqa: E402
from ponies.model import ConditionalLogit  # noqa: E402

COMMISSIONS = (0.0, 0.05, 0.08)
PRICE_BANDS = [1.0, 2.0, 3.0, 4.5, 6.5, 10.0, 15.0, 25.0, 50.0, 1e9]
COARSE_BANDS = [1.0, 4.0, 10.0, 1e9]
FIXED_CUTS = [-np.inf, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, np.inf]
K_VALUES = (0.0, 0.05, 0.10, 0.20)
FEATURES = ["log_p", "move", "move_sq"]
MIN_BETS = 200

# Betfair price ladder: (lower bound inclusive, upper bound exclusive, increment)
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1001.0, 10.0)]


def out(obj) -> None:
    print(json.dumps(obj, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    raise TypeError(str(type(o)))


def r4(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) else round(x, 4)


def net_returns(price: np.ndarray, won: np.ndarray, commission: float) -> np.ndarray:
    """Vectorised ponies.betfair.net_return [A-54]: commission on net winnings only."""
    price = np.asarray(price, dtype=float)
    return np.where(won, (price - 1.0) * (1.0 - commission), -1.0)


def tick_down(price: np.ndarray) -> np.ndarray:
    """One Betfair tick below ``price`` (the price you get if the last trade is one
    tick better than what is actually on offer)."""
    p = np.asarray(price, dtype=float)
    inc = np.full_like(p, 10.0)
    q = p - 1e-9  # increment of the band just below the price
    for lo, hi, step in TICKS:
        inc = np.where((q >= lo) & (q < hi), step, inc)
    return np.maximum(1.01, np.round(p - inc, 2))


def bands(series: pd.Series, edges: list[float]) -> pd.Series:
    labels = [f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+" for lo, hi in zip(edges, edges[1:])]
    return pd.cut(series, edges, right=False, labels=labels)


# --------------------------------------------------------------------------- data

def load(path: Path, end: str | None) -> pd.DataFrame:
    cols = ["market_id", "market_name", "start", "date", "field_size", "selection_id",
            "ltp_pre", "ltp_t30", "bsp", "won", "market_type"]
    df = pd.read_parquet(path, columns=cols)
    df = df.drop_duplicates(["market_id", "selection_id"])
    df = df[(df["market_type"] == "WIN") & df["won"].notna()]
    if end:
        df = df[df["date"] < pd.Timestamp(end)]
    g = df.groupby("market_id")
    one_winner = g["won"].transform("sum") == 1          # drops dead heats (0.2% of markets)
    df["_ok"] = df["ltp_pre"].notna() & (df["ltp_pre"] > 1)
    all_priced = df.groupby("market_id")["_ok"].transform("min").astype(bool)
    df = df[one_winner & all_priced].drop(columns=["_ok", "market_type"]).copy()
    df = df.sort_values(["start", "market_id", "ltp_pre"]).reset_index(drop=True)

    # market probabilities at ltp_pre: Shin [A-51] (proportional makes no difference here)
    lp = df["ltp_pre"].to_numpy()
    p_shin = np.empty(len(df))
    for _, idx in df.groupby("market_id", sort=False).indices.items():
        p_shin[idx] = odds.shin_probabilities(lp[idx])
    df["p_shin"] = p_shin
    df["log_p"] = np.log(df["p_shin"])
    df["book"] = (1 / df["ltp_pre"]).groupby(df["market_id"]).transform("sum")
    df["harness"] = df["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b")
    df["has_t30"] = df["ltp_t30"].notna() & (df["ltp_t30"] > 1)
    df["move"] = np.where(df["has_t30"], np.log(df["ltp_t30"] / df["ltp_pre"]), np.nan)
    df["move_sq"] = df["move"] * df["move"].abs()
    n_t30 = df.groupby("market_id")["has_t30"].transform("sum")
    df["t30_cover"] = np.select([n_t30 == 0, n_t30 == df["field_size"]], ["none", "all"], "partial")
    df["race_id"] = df["market_id"]
    df["month"] = df["date"].dt.to_period("M")
    df["won_b"] = df["won"].to_numpy() == 1
    return df


def mid_day(df: pd.DataFrame) -> pd.Timestamp:
    days = np.sort(df["date"].unique())
    return pd.Timestamp(days[len(days) // 2])


# --------------------------------------------------------------------------- cells

def bet_cell(g: pd.DataFrame, mid: pd.Timestamp, price_col: str = "ltp_pre") -> dict:
    """Back 1 unit on every row of ``g`` at ``price_col``. Reports ROI at each
    commission, the per-bet SE (at 5%), both halves (ROI and SE at 5%), ROI at
    BSP and at one tick below ltp_pre (5%), strike vs implied."""
    n = len(g)
    res = {"bets": int(n)}
    if n == 0:
        return res
    won = g["won_b"].to_numpy()
    price = g[price_col].to_numpy(dtype=float)
    for c in COMMISSIONS:
        res[f"roi_c{int(c * 100)}"] = r4(net_returns(price, won, c).mean())
    r5 = net_returns(price, won, 0.05)
    res["se"] = r4(r5.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    first = (g["date"] < mid).to_numpy()
    halves, halves_se = [], []
    for mask in (first, ~first):
        h = r5[mask]
        halves.append(r4(h.mean()) if len(h) else None)
        halves_se.append(r4(h.std(ddof=1) / math.sqrt(len(h))) if len(h) > 1 else None)
    res["halves_roi_c5"] = halves
    res["halves_se_c5"] = halves_se
    res["halves_bets"] = [int(first.sum()), int((~first).sum())]
    res["positive_both_halves"] = bool(all(h is not None and s is not None and h > 2 * s
                                           for h, s in zip(halves, halves_se)))
    res["strike"] = r4(won.mean())
    res["implied_shin"] = r4(g["p_shin"].mean())
    res["strike_over_implied"] = r4(won.mean() / g["p_shin"].mean())
    res["median_price"] = r4(np.median(price))
    # realistic execution: one tick worse than the last traded price, and the Betfair SP
    res["roi_c5_tick_down"] = r4(net_returns(tick_down(price), won, 0.05).mean())
    has_bsp = g["bsp"].notna().to_numpy() & (g["bsp"].to_numpy(dtype=float) > 1)
    if has_bsp.sum() > 1:
        res["roi_c5_bsp"] = r4(net_returns(g["bsp"].to_numpy(dtype=float)[has_bsp], won[has_bsp], 0.05).mean())
        res["bsp_bets"] = int(has_bsp.sum())
    return res


def z_of(cell: dict) -> float:
    if cell.get("bets", 0) < MIN_BETS or not cell.get("se"):
        return float("nan")
    return cell["roi_c5"] / cell["se"]


def half_z(cell: dict, h: int) -> float:
    r, s = cell["halves_roi_c5"][h], cell["halves_se_c5"][h]
    if r is None or s is None or cell["halves_bets"][h] < MIN_BETS:
        return float("nan")
    return r / s


def market_calibration(g: pd.DataFrame) -> dict:
    won = g["won_b"].to_numpy()
    ll = float(-np.log(g.loc[won, "p_shin"].clip(1e-12)).mean())
    fields = g.groupby("market_id")["field_size"].first()
    return {"markets": int(g["market_id"].nunique()), "runners": int(len(g)),
            "book_pct_median": r4(g.groupby("market_id")["book"].first().median()),
            "market_log_loss_shin": r4(ll),
            "info_gain_vs_uniform": r4(float(np.log(fields).mean()) - ll),  # comparable across field sizes
            "mean_field": r4(g.groupby("market_id")["field_size"].first().mean()),
            "harness_share": r4(g.groupby("market_id")["harness"].first().mean())}


# --------------------------------------------------------------------------- walk-forward model

def walk_forward(uni: pd.DataFrame, burn_in: int) -> tuple[pd.DataFrame, list[dict]]:
    """Monthly expanding-window refits: month m is predicted by a model fitted on
    every row dated strictly before the first day of m."""
    months = sorted(uni["month"].unique())
    preds = np.full(len(uni), np.nan)
    month_arr = uni["month"].to_numpy()
    coefs = []
    for i, m in enumerate(months):
        if i < burn_in:
            continue
        train = uni[month_arr < m]
        test_idx = np.flatnonzero(month_arr == m)
        if train["race_id"].nunique() < 500 or len(test_idx) == 0:
            continue
        model = ConditionalLogit(FEATURES, unpenalized=("log_p",)).fit(train)
        preds[test_idx] = model.predict_proba(uni.iloc[test_idx])
        raw = model.weights / model.stds
        coefs.append({"month": str(m), "train_races": int(train["race_id"].nunique()),
                      "coef_log_p": float(raw[0]), "coef_move": float(raw[1]), "coef_move_sq": float(raw[2]),
                      "iters": len(model.history)})
    uni = uni.assign(p_model=preds)
    return uni[uni["p_model"].notna()].copy(), coefs


def log_loss_gain(test: pd.DataFrame, mid: pd.Timestamp) -> dict:
    w = test[test["won_b"]]
    gain = (-np.log(w["p_shin"].clip(1e-12)) + np.log(w["p_model"].clip(1e-12))).to_numpy()
    first = (w["date"] < mid).to_numpy()
    res = {"races": int(len(w)), "ll_market_shin": r4(-np.log(w["p_shin"].clip(1e-12)).mean()),
           "ll_model": r4(-np.log(w["p_model"].clip(1e-12)).mean()),
           "gain_per_race": r4(gain.mean()), "gain_se": r4(gain.std(ddof=1) / math.sqrt(len(gain))),
           "gain_halves": [r4(gain[first].mean()), r4(gain[~first].mean())],
           "gain_halves_se": [r4(gain[first].std(ddof=1) / math.sqrt(max(first.sum(), 2))),
                              r4(gain[~first].std(ddof=1) / math.sqrt(max((~first).sum(), 2)))]}
    res["positive_2se"] = bool(gain.mean() > 2 * gain.std(ddof=1) / math.sqrt(len(gain)))
    return res


# --------------------------------------------------------------------------- multiple testing

def benjamini_hochberg(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Boolean mask of rejected hypotheses at FDR q (one-sided p-values; NaN = not tested)."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    rej = np.zeros(len(p), dtype=bool)
    if m == 0:
        return rej
    order = np.argsort(p[ok])
    sorted_p = p[ok][order]
    thresh = q * (np.arange(1, m + 1) / m)
    below = np.flatnonzero(sorted_p <= thresh)
    if len(below):
        k = below.max()
        idx = np.flatnonzero(ok)[order[: k + 1]]
        rej[idx] = True
    return rej


def one_sided_p(z: float) -> float:
    if math.isnan(z):
        return float("nan")
    return 0.5 * math.erfc(z / math.sqrt(2))


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=Path("data/raw/betfair/split/au_win.parquet"))
    ap.add_argument("--end", default=None, help="development: use only dates before this")
    ap.add_argument("--burn-in", type=int, default=6, help="months before the first walk-forward prediction")
    a = ap.parse_args()
    t0 = time.time()

    df = load(a.path, a.end)
    mid = mid_day(df)
    days = np.sort(df["date"].unique())
    cells: list[tuple[str, dict]] = []  # every bet rule tested, for BH

    out({"section": "data", "markets": int(df["market_id"].nunique()), "runners": int(len(df)),
         "from": str(pd.Timestamp(days[0]).date()), "to": str(pd.Timestamp(days[-1]).date()),
         "mid_day_for_halves": str(mid.date()),
         "runners_with_t30": r4(df["has_t30"].mean()),
         "markets_t30_cover": df.groupby("market_id")["t30_cover"].first().value_counts(normalize=True).round(4).to_dict(),
         "harness_runner_share": r4(df["harness"].mean()),
         "t30_cover_by_year": df.groupby(df["date"].dt.year)["has_t30"].mean().round(3).to_dict(),
         "move_summary_t30_runners": df.loc[df["has_t30"], "move"].describe()[["mean", "std", "25%", "50%", "75%"]].round(4).to_dict(),
         "load_seconds": round(time.time() - t0, 1)})

    # ---------------- A. with vs without t30 trading --------------------------------------------
    out({"section": "A_market_level_calibration_by_t30_cover",
         "note": "markets grouped by how many runners traded 30 min out; t30 coverage rises from 29% (2016) to 52% (2020) and is higher for short prices, so compare within band",
         **{cov: market_calibration(g) for cov, g in df.groupby("t30_cover")}})
    out({"section": "A_market_level_calibration_by_t30_cover_and_code",
         "note": "untraded-at-t30 markets are mostly harness, which has lower log loss anyway: compare within code",
         **{f"{'harness' if h else 'gallops'}:{cov}": market_calibration(g)
            for (cov, h), g in df.groupby(["t30_cover", "harness"])}})
    df["band"] = bands(df["ltp_pre"], PRICE_BANDS)
    rows = []
    for (band, has), g in df.groupby(["band", "has_t30"], observed=True):
        c = bet_cell(g, mid)
        rows.append({"band": str(band), "runner_traded_t30": bool(has), **c})
        cells.append((f"A:band={band}:t30={bool(has)}", c))
    out({"section": "A_runner_level_calibration_by_band_and_t30", "rows": rows})
    # same, inside markets where some but not all runners traded (controls for market liquidity)
    rows = []
    part = df[df["t30_cover"] == "partial"]
    for (band, has), g in part.groupby([bands(part["ltp_pre"], COARSE_BANDS), "has_t30"], observed=True):
        rows.append({"band": str(band), "runner_traded_t30": bool(has),
                     **{k: v for k, v in bet_cell(g, mid).items()
                        if k in ("bets", "strike", "implied_shin", "strike_over_implied", "roi_c5", "se", "halves_roi_c5")}})
    out({"section": "A_within_partial_markets_by_coarse_band_and_t30", "rows": rows})

    # ---------------- B. move buckets --------------------------------------------------------------
    t30 = df[df["has_t30"]].copy()
    out({"section": "B_note", "conditional_on": "runners with a trade 30 min before the start (ltp_t30 present)",
         "runners": int(len(t30)), "share_of_all_runners": r4(len(t30) / len(df)),
         "markets_represented": int(t30["market_id"].nunique())})
    # deciles (boundaries from the whole t30 sample: descriptive, not a fitted rule)
    t30["decile"] = pd.qcut(t30["move"], 10, labels=False, duplicates="drop")
    rows = []
    for d, g in t30.groupby("decile"):
        c = bet_cell(g, mid)
        rows.append({"decile": int(d), "move_range": [r4(g["move"].min()), r4(g["move"].max())],
                     "move_mean": r4(g["move"].mean()), **c})
        cells.append((f"B:decile={d}", c))
    out({"section": "B_calibration_and_roi_by_move_decile", "rows": rows})
    t30["cut"] = pd.cut(t30["move"], FIXED_CUTS, right=False)
    rows = []
    for cut, g in t30.groupby("cut", observed=True):
        c = bet_cell(g, mid)
        rows.append({"move_cut": str(cut), "move_mean": r4(g["move"].mean()), **c})
        cells.append((f"B:cut={cut}", c))
    out({"section": "B_calibration_and_roi_by_fixed_move_cut", "rows": rows})
    rows = []
    t30["cband"] = bands(t30["ltp_pre"], COARSE_BANDS)
    for (cband, cut), g in t30.groupby(["cband", "cut"], observed=True):
        c = bet_cell(g, mid)
        rows.append({"price_band": str(cband), "move_cut": str(cut), **{k: c.get(k) for k in (
            "bets", "strike", "implied_shin", "strike_over_implied", "roi_c0", "roi_c5", "roi_c8", "se",
            "halves_roi_c5", "halves_se_c5", "positive_both_halves", "roi_c5_bsp", "roi_c5_tick_down")}})
        cells.append((f"B:band={cband}:cut={cut}", c))
    out({"section": "B_roi_by_price_band_x_fixed_move_cut", "rows": rows})
    rows = []
    for (h, cut), g in t30.groupby(["harness", "cut"], observed=True):
        c = bet_cell(g, mid)
        code = "harness" if h else "gallops"
        rows.append({"code": code, "move_cut": str(cut), **{k: c.get(k) for k in (
            "bets", "strike", "implied_shin", "strike_over_implied", "roi_c0", "roi_c5", "roi_c8", "se",
            "halves_roi_c5", "halves_se_c5", "positive_both_halves", "roi_c5_bsp", "roi_c5_tick_down")}})
        cells.append((f"B:code={code}:cut={cut}", c))
    out({"section": "B_roi_by_code_x_fixed_move_cut", "rows": rows})
    # how does the market move overall? book at t30 vs pre for fully traded markets
    full = df[df["t30_cover"] == "all"]
    book30 = (1 / full["ltp_t30"]).groupby(full["market_id"]).sum()
    out({"section": "B_fully_traded_markets", "markets": int(len(book30)),
         "book_pct_t30_median": r4(book30.median()), "book_pct_pre_median": r4(full.groupby("market_id")["book"].first().median()),
         "corr_move_vs_log_p_pre": r4(np.corrcoef(full["move"], full["log_p"])[0, 1])})

    # ---------------- C. walk-forward conditional logit -------------------------------------------
    any_t30 = df[df["t30_cover"] != "none"]
    zero_imp = any_t30.assign(move=any_t30["move"].fillna(0.0), move_sq=any_t30["move_sq"].fillna(0.0))
    variants = {
        "all_t30_markets": df[df["t30_cover"] == "all"],      # move observed for every runner
        "any_t30_markets": any_t30,                           # missing move -> race-median imputation [A-07]
        "any_t30_markets_zero_impute": zero_imp,              # missing move -> 0 ("no information")
    }
    model_results = {}
    for name, uni in variants.items():
        t1 = time.time()
        test, coefs = walk_forward(uni.reset_index(drop=True), a.burn_in)
        if test.empty:
            out({"section": "C_model", "variant": name, "error": "no predictions"})
            continue
        tmid = mid_day(test)
        ll = log_loss_gain(test, tmid)
        co = pd.DataFrame(coefs)
        res = {"section": "C_model", "variant": name, "features": FEATURES, "refits": len(coefs),
               "predicted_from": str(test["date"].min().date()), "predicted_to": str(test["date"].max().date()),
               "test_markets": int(test["market_id"].nunique()), "test_runners": int(len(test)),
               "mid_day_for_halves": str(tmid.date()),
               "coef_median_raw": {k: r4(co[k].median()) for k in ("coef_log_p", "coef_move", "coef_move_sq")},
               "coef_range_raw": {k: [r4(co[k].min()), r4(co[k].max())] for k in ("coef_log_p", "coef_move", "coef_move_sq")},
               "coef_first_last": {k: [r4(co[k].iloc[0]), r4(co[k].iloc[-1])] for k in ("coef_move", "coef_move_sq")},
               "log_loss": ll, "seconds": round(time.time() - t1, 1)}
        out(res)
        model_results[name] = {"test": test, "ll": ll}
        # bets on runners whose move is observed, where model prob * ltp_pre > 1 + k
        obs = test[test["has_t30"]].copy()
        obs["edge"] = obs["p_model"] * obs["ltp_pre"]
        rows = []
        for k in K_VALUES:
            g = obs[obs["edge"] > 1 + k]
            c = bet_cell(g, tmid)
            rows.append({"k": k, "share_of_runners": r4(len(g) / max(len(obs), 1)),
                         "mean_move": r4(g["move"].mean()) if len(g) else None,
                         "share_firmed": r4((g["move"] > 0).mean()) if len(g) else None,
                         # stale-price diagnostics: an under-round book means ltp_pre was not a live price
                         "median_book": r4(g["book"].median()) if len(g) else None,
                         "share_book_below_0.95": r4((g["book"] < 0.95).mean()) if len(g) else None,
                         "median_ltp_pre_over_bsp": r4((g["ltp_pre"] / g["bsp"]).median()) if len(g) else None, **c})
            cells.append((f"C:{name}:k={k}", c))
        out({"section": "C_model_bets", "variant": name,
             "note": "bets only on runners with observed move; price = ltp_pre, plus BSP and one-tick-down lines", "rows": rows})
        # the same rule restricted to coherent markets (book of 1/ltp_pre within [0.98, 1.05]),
        # i.e. where the last traded prices were plausibly still available
        coh = obs[(obs["book"] >= 0.98) & (obs["book"] <= 1.05)]
        rows = []
        for k in (0.0, 0.05):
            g = coh[coh["edge"] > 1 + k]
            c = bet_cell(g, tmid)
            rows.append({"k": k, "share_of_runners": r4(len(g) / max(len(obs), 1)), **c})
            cells.append((f"C:{name}:coherent_book:k={k}", c))
        out({"section": "C_model_bets_coherent_book_only", "variant": name,
             "coherent_share_of_runners": r4(len(coh) / max(len(obs), 1)), "rows": rows})
        # the same bets at Betfair SP as a *decision* price too (bet placed as SP bet)
        rows = []
        for k in (0.0, 0.10):
            g = obs[(obs["edge"] > 1 + k) & obs["bsp"].notna() & (obs["bsp"] > 1)]
            c = bet_cell(g, tmid, price_col="bsp")
            rows.append({"k": k, **{kk: c.get(kk) for kk in ("bets", "roi_c0", "roi_c5", "roi_c8", "se", "halves_roi_c5", "positive_both_halves")}})
            cells.append((f"C:{name}:bsp_bet:k={k}", c))
        out({"section": "C_model_bets_settled_at_bsp", "variant": name, "rows": rows})
        # calibration of the model's edge: does edge predict excess return at all?
        obs["edge_bin"] = pd.cut(obs["edge"], [0, 0.9, 1.0, 1.05, 1.1, 1.2, 10], right=False)
        rows = []
        for b, g in obs.groupby("edge_bin", observed=True):
            c = bet_cell(g, tmid)
            rows.append({"edge_bin": str(b), **{kk: c.get(kk) for kk in ("bets", "strike", "implied_shin", "roi_c0", "roi_c5", "se", "halves_roi_c5")}})
        out({"section": "C_model_edge_calibration", "variant": name, "rows": rows})

    # ---------------- D. multiple testing, both halves, opposite-half development ------------------
    names = [n for n, _ in cells]
    zs = np.array([z_of(c) for _, c in cells])
    ps = np.array([one_sided_p(z) for z in zs])
    rej = benjamini_hochberg(ps, 0.05)
    tested = int((~np.isnan(ps)).sum())
    both = [n for n, c in cells if c.get("positive_both_halves") and c.get("bets", 0) >= MIN_BETS]
    out({"section": "D_multiple_testing", "bet_cells_tested_with_min_bets": tested, "bet_cells_total": len(cells),
         "min_bets": MIN_BETS, "fdr_q": 0.05,
         "bh_survivors_roi_c5": [names[i] for i in np.flatnonzero(rej)],
         "positive_both_halves_2se_c5": both,
         "best_cells_by_z": sorted([{"cell": n, "z": r4(z), "roi_c5": c["roi_c5"], "bets": c["bets"],
                                     "halves_roi_c5": c["halves_roi_c5"]}
                                    for n, c, z in zip(names, [c for _, c in cells], zs) if not math.isnan(z)],
                                   key=lambda d: -d["z"])[:8]})
    # opposite-half development: choose the best cell in one half, evaluate in the other
    for dev in (0, 1):
        hz = np.array([half_z(c, dev) for _, c in cells])
        if np.all(np.isnan(hz)):
            continue
        i = int(np.nanargmax(hz))
        c = cells[i][1]
        other = 1 - dev
        out({"section": "D_opposite_half_development", "development_half": dev + 1, "chosen_cell": names[i],
             "dev_roi_c5": c["halves_roi_c5"][dev], "dev_z": r4(hz[i]), "dev_bets": c["halves_bets"][dev],
             "holdout_roi_c5": c["halves_roi_c5"][other], "holdout_z": r4(half_z(c, other)),
             "holdout_bets": c["halves_bets"][other],
             "holds": bool(not math.isnan(half_z(c, other)) and half_z(c, other) > 2)})
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

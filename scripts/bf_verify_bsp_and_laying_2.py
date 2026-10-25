"""Adversarial verification (leakage / data-quality lens) of the "bsp_and_laying" claim:

    AU gallops WIN runners with ltp_pre in [4.5, 6.5): backing 1 unit at Betfair SP
    returns +0.36% (+/- 0.07%) of stake more than backing the same runner at the
    last traded price, 5% commission on both; halves +0.38% / +0.35%, 63,158 bets;
    a BH survivor among 122 paired cells. Relative finding: backing at SP loses -3.9%.

    python scripts/bf_verify_bsp_and_laying_2.py [--end 2018-06-01]

(The harness asked for ..._0.py; that file already exists in git for the $10-15 cell,
so this is a new file.) Reuses load() / back_returns() / tick_down() / tick_up() /
benjamini_hochberg() from scripts/bf_bsp_and_laying.py (imported by path, not edited)
so the reproduction is exact. Nothing is fitted. What is checked:

  0. data quality on the gallops WIN table: duplicates, dead heats, rows vs field_size,
     one WIN market per event+start, prices on the Betfair ladder, BSP coverage.
  1. reproduction of the cell: c0/c5/c8, iid SE, date-clustered SE, halves at the
     median day, by year, odd/even ISO weeks.
  2. price discreteness: the band holds the ladder step at 6.0 (tick 0.1 -> 0.2).
     Per-tick decomposition 4.5 .. 6.4: runners, winners, mean log(bsp/ltp) all /
     winners, paired gain, SE, and each tick's share of the band's total gain;
     sub-bands [4.5,6.0) and [6.0,6.5), the band minus tick 6.0, shifted windows,
     with halves and BH over that fixed family; the same at every ladder step
     (2,3,4,6,10,20,30,50,100: at / one tick below / one tick above, with the
     "uniform fair price" prediction (w_up - w_down) / (4 t) for the step tick).
  3. staleness / liquidity proxies (one-minute snapshots): runner traded 30 min out
     or not; ltp_pre == ltp_t30 (no print change in 30 min); market book at ltp_pre
     (< 0.95 = stale after a late scratching); bsp book; metro vs country; rank.
  4. survivorship: markets dropped because some runner never traded pre-off, and
     dead-heat markets: the paired gain on their band runners that do have prices.
  5. attainability: SP vs the last print, vs one tick worse, vs the average.
  6. winners vs all runners: is the gain just strike x E[bsp - ltp]?

Prints JSON lines with summary numbers only; never prints horse names.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

_spec = importlib.util.spec_from_file_location("bspl", HERE / "bf_bsp_and_laying.py")
bspl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bspl)

C5 = 0.05
COMMS = (0.0, 0.05, 0.08)
LO, HI = 4.5, 6.5
LADDER_STEPS = (2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 30.0, 50.0, 100.0)
METRO = ("flemington", "caulfield", "moonee valley", "sandown", "randwick", "rosehill", "warwick farm",
         "canterbury", "eagle farm", "doomben", "morphettville", "ascot", "belmont")
# fixed before looking at outcomes: the claim's band, its two sides of the ladder step, and shifted windows
VARIANTS = [("[4.5,6.5) claim", 4.5, 6.5), ("[4.5,6.0) below the step", 4.5, 6.0), ("[6.0,6.5) at/above the step", 6.0, 6.5),
            ("[4.0,6.0)", 4.0, 6.0), ("[5.0,6.0)", 5.0, 6.0), ("[5.5,6.5)", 5.5, 6.5), ("[4.5,7.0)", 4.5, 7.0),
            ("[6.0,8.0)", 6.0, 8.0), ("[3.0,4.5)", 3.0, 4.5), ("[6.5,10)", 6.5, 10.0)]


def out(obj) -> None:
    print(json.dumps(obj, default=bspl._json_default), flush=True)


def r4(x):
    return bspl.r4(x)


def r5(x):
    x = float(x)
    return None if math.isnan(x) else round(x, 5)


def mean_se(x: np.ndarray) -> tuple[float, float]:
    return bspl._mean_se(np.asarray(x, dtype=float))


def cluster_se(d: np.ndarray, groups: np.ndarray) -> float:
    """SE of the mean with arbitrary within-cluster correlation (cluster = race day)."""
    n = len(d)
    if n < 2:
        return float("nan")
    e = d - d.mean()
    s = pd.Series(e).groupby(groups).sum().to_numpy()
    return float(math.sqrt((s ** 2).sum()) / n)


def paired(df: pd.DataFrame, idx: np.ndarray, digits: int = 4) -> dict:
    """Paired (bsp return - ltp_pre return) on the runners idx that have both prices."""
    sub = df.iloc[idx]
    sub = sub[sub["bsp_ok"].to_numpy()]
    n = len(sub)
    if n == 0:
        return {"bets": 0}
    lp = sub["ltp_pre"].to_numpy(dtype=float)
    bsp = sub["bsp"].to_numpy(dtype=float)
    won = sub["won_b"].to_numpy()
    first = sub["first"].to_numpy()
    res = {"bets": n, "n_won": int(won.sum()), "strike": r4(won.mean()), "implied_ltp": r4((1 / lp).mean()),
           "implied_bsp": r4((1 / bsp).mean()), "median_ltp": r4(np.median(lp))}
    for c in COMMS:
        d = bspl.back_returns(bsp, won, c) - bspl.back_returns(lp, won, c)
        res[f"diff_c{int(c * 100)}"] = round(float(d.mean()), digits)
    d5 = bspl.back_returns(bsp, won, C5) - bspl.back_returns(lp, won, C5)
    m, se = mean_se(d5)
    res["se"] = round(se, digits) if not math.isnan(se) else None
    res["z"] = r4(m / se) if se and se > 0 else None
    se_cl = cluster_se(d5, sub["date"].to_numpy())
    res["se_cl_date"] = round(se_cl, digits) if not math.isnan(se_cl) else None
    res["z_cl_date"] = r4(m / se_cl) if se_cl and se_cl > 0 else None
    h = []
    for mask in (first, ~first):
        hm, hs = mean_se(d5[mask])
        h.append([int(mask.sum()), int(won[mask].sum()), round(hm, digits) if not math.isnan(hm) else None,
                  round(hs, digits) if not math.isnan(hs) else None])
    res["halves_[n,won,diff_c5,se]"] = h
    res["both_halves_over_2se"] = bool(all(v[2] is not None and v[3] and v[2] > 2 * v[3] for v in h))
    lr = np.log(bsp / lp)
    res["mean_log_ratio_all"] = r4(lr.mean())
    res["mean_ratio_minus1_all"] = r4((bsp / lp - 1).mean())
    res["mean_log_ratio_winners"] = r4(lr[won].mean()) if won.any() else None
    res["mean_ratio_minus1_winners"] = r4((bsp[won] / lp[won] - 1).mean()) if won.any() else None
    res["gain_if_won_indep_of_ratio_c5"] = r4(won.mean() * (bsp - lp).mean() * (1 - C5))
    res["back_ltp_c5"] = r4(bspl.back_returns(lp, won, C5).mean())
    res["back_bsp_c5"] = r4(bspl.back_returns(bsp, won, C5).mean())
    res["back_ltp_tick_down_c5"] = r4(bspl.back_returns(bspl.tick_down(lp), won, C5).mean())
    res["share_sp_better_1tick"] = r4((bsp >= bspl.tick_up(lp) - 1e-9).mean())
    res["share_sp_worse_1tick"] = r4((bsp <= bspl.tick_down(lp) + 1e-9).mean())
    return res


def compact(p: dict) -> list:
    """[bets, n_won, diff_c0, diff_c5, diff_c8, se, z, z_cl_date, half1, half2, both_halves, mean_log_ratio_all, mean_log_ratio_winners]"""
    if p.get("bets", 0) == 0:
        return [0]
    h = p["halves_[n,won,diff_c5,se]"]
    return [p["bets"], p["n_won"], p["diff_c0"], p["diff_c5"], p["diff_c8"], p["se"], p["z"], p["z_cl_date"], h[0][2], h[1][2],
            p["both_halves_over_2se"], p["mean_log_ratio_all"], p["mean_log_ratio_winners"]]


def data_quality(raw_path: Path, g: pd.DataFrame) -> dict:
    cols = ["market_id", "event_id", "venue", "market_name", "start", "field_size", "selection_id", "ltp_pre", "bsp",
            "won", "n_winners", "market_type", "dead_heat_share"]
    w = pd.read_parquet(raw_path, columns=cols)
    n_raw = len(w)
    w = w.drop_duplicates(["market_id", "selection_id"])
    w = w[w["market_type"] == "WIN"]
    harness = w["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b")
    w = w[~harness]
    mk = w.drop_duplicates("market_id")
    rows = w.groupby("market_id").size()
    fs = w.groupby("market_id")["field_size"].first()
    ws = w.groupby("market_id")["won"].sum()
    res = {"rows_raw_all_codes": int(n_raw), "rows_after_dedupe_all_codes": int(len(pd.read_parquet(raw_path, columns=["market_id", "selection_id"]).drop_duplicates())),
           "gallops_rows": int(len(w)), "gallops_markets": int(len(mk)),
           "n_winners_values": {str(k): int(v) for k, v in mk["n_winners"].value_counts().items()},
           "won_sum_per_market": {str(k): int(v) for k, v in ws.value_counts().items()},
           "markets_unsettled_won_nan": int(w.groupby("market_id")["won"].apply(lambda s: s.isna().any()).sum()),
           "markets_rows_ne_field_size": int((rows != fs).sum()),
           "win_markets_per_event_start": {str(k): int(v) for k, v in mk.groupby(["event_id", "start"])["market_id"].nunique().value_counts().items()},
           "win_markets_per_venue_start": {str(k): int(v) for k, v in mk.groupby(["venue", "start"])["market_id"].nunique().value_counts().items()},
           "selection_repeated_within_market_after_dedupe": 0}
    lp = w["ltp_pre"].dropna().to_numpy(dtype=float)
    inc = bspl._increment(lp - 1e-9)
    res["ltp_pre_off_ladder_share"] = r4((np.abs(np.round(lp / inc) * inc - lp) > 1e-6).mean())
    b = w["bsp"].dropna().to_numpy(dtype=float)
    incb = bspl._increment(b - 1e-9)
    res["bsp_off_ladder_share"] = r4((np.abs(np.round(b / incb) * incb - b) > 1e-6).mean())
    res["bsp_equals_ltp_share"] = r4((w["bsp"] == w["ltp_pre"]).mean())
    res["bsp_missing_share_gallops"] = r4(w["bsp"].isna().mean())
    res["ltp_pre_missing_share_gallops"] = r4(w["ltp_pre"].isna().mean())
    bb = (1.0 / g["bsp"]).groupby(g["market_id"]).sum()
    lb = (1.0 / g["ltp_pre"]).groupby(g["market_id"]).sum()
    res["bsp_book_q05_50_95"] = [r4(v) for v in bb.quantile([0.05, 0.5, 0.95])]
    res["ltp_book_q05_50_95"] = [r4(v) for v in lb.quantile([0.05, 0.5, 0.95])]
    res["markets_ltp_book_below_0.95"] = int((lb < 0.95).sum())
    res["markets_ltp_book_above_1.05"] = int((lb > 1.05).sum())
    return res


def tick_profile(g: pd.DataFrame, lo: float, hi: float, min_n: int = 200) -> dict:
    lp = g["ltp_pre"].round(2).to_numpy()
    ok = g["bsp_ok"].to_numpy()
    ticks = sorted(v for v in pd.unique(lp[(lp >= lo) & (lp < hi) & ok]) if (lp == v).sum() >= min_n)
    rows = {}
    for t in ticks:
        idx = np.flatnonzero((lp == t) & ok)
        p = paired(g, idx)
        rows[f"{t:g}"] = [p["bets"], p["n_won"], p["strike"], p["implied_ltp"], p["mean_log_ratio_all"], p["mean_log_ratio_winners"],
                          p["diff_c5"], p["se"], r4(p["diff_c5"] * p["bets"]), p["halves_[n,won,diff_c5,se]"][0][2], p["halves_[n,won,diff_c5,se]"][1][2]]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=HERE.parent / "data/raw/betfair/split/au_win.parquet")
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    t0 = time.time()
    df, info, mid = bspl.load(a.path, a.end)
    g = df[~df["harness"]].reset_index(drop=True)
    out({"section": "data", "markets_gallops": int(g["market_id"].nunique()), "runners_gallops": int(len(g)),
         "mid_date_for_halves": str(mid.date()), "end": a.end, "bsp_coverage": r4(g["bsp_ok"].mean()),
         "t30_coverage": r4(g["has_t30"].mean()), "from": info["from"], "to": info["to"]})
    out({"section": "0_data_quality", **data_quality(a.path, g)})

    lp = g["ltp_pre"].to_numpy(dtype=float)
    lpr = g["ltp_pre"].round(2).to_numpy()
    band = (lp >= LO) & (lp < HI)
    idx_band = np.flatnonzero(band)
    head = paired(g, idx_band)
    out({"section": "1_reproduce", "cell": f"gallops ltp_pre in [{LO:g},{HI:g}) paired SP minus LTP, 1 unit each", **head})
    years = g["date"].dt.year.to_numpy()
    week = g["date"].dt.isocalendar().week.to_numpy().astype(int)
    out({"section": "1_by_year", "columns": "compact = [bets, n_won, diff_c0, diff_c5, diff_c8, se, z, z_cl_date, half1, half2, both_halves, mean_log_ratio_all, mean_log_ratio_winners]",
         "rows": {str(y): compact(paired(g, np.flatnonzero(band & (years == y)))) for y in sorted(np.unique(years))},
         "odd_weeks": compact(paired(g, np.flatnonzero(band & (week % 2 == 1)))),
         "even_weeks": compact(paired(g, np.flatnonzero(band & (week % 2 == 0))))})

    # ---- 2. price discreteness: per tick, the ladder step at 6.0
    prof = tick_profile(g, 4.0, 8.0)
    out({"section": "2_by_tick", "columns": "[bets, n_won, strike, implied_ltp, mean_log(bsp/ltp) all, mean_log winners, paired diff c5, se, bets*diff (gain units), half1, half2]",
         "rows": prof})
    band_total = sum(v[8] for k, v in prof.items() if LO <= float(k) < HI)
    contrib = {k: r4(v[8] / band_total) for k, v in prof.items() if LO <= float(k) < HI}
    step_ticks = [k for k in contrib if float(k) >= 6.0]
    out({"section": "2_band_gain_share_by_tick", "band_total_gain_units_c5": r4(band_total), "band_bets": int(len(idx_band)),
         "share_by_tick": contrib, "share_of_gain_from_ticks_6.0_6.2_6.4": r4(sum(contrib[k] for k in step_ticks)),
         "share_of_bets_at_ticks_6.0_6.2_6.4": r4(sum(prof[k][0] for k in step_ticks) / len(idx_band)),
         "share_of_gain_from_tick_6.0_alone": contrib.get("6"), "share_of_bets_at_tick_6.0": r4(prof["6"][0] / len(idx_band)) if "6" in prof else None,
         "runners_per_0.1_of_price_5.5-5.9": r4(np.mean([prof[k][0] for k in ("5.5", "5.6", "5.7", "5.8", "5.9") if k in prof])),
         "runners_at_6.0 (tick covers ~0.15 of price)": prof.get("6", [None])[0],
         "runners_per_0.1_of_price_6.2-6.4": r4(np.mean([prof[k][0] / 2 for k in ("6.2", "6.4") if k in prof]))})
    # the same at every ladder step: is the spike generic?
    steps = {}
    okb = g["bsp_ok"].to_numpy()
    for t in LADDER_STEPS:
        below_t = bspl.tick_down(np.array([t]))[0]
        above_t = bspl.tick_up(np.array([t]))[0]
        at = np.flatnonzero((lpr == t) & okb)
        below = np.flatnonzero((lpr == round(below_t, 2)) & okb)
        above = np.flatnonzero((lpr == round(above_t, 2)) & okb)
        p_at, p_b, p_a = paired(g, at), paired(g, below), paired(g, above)
        w_down, w_up = t - below_t, above_t - t
        steps[f"{t:g}"] = {"pure_ladder_prediction_mean_ratio_minus1": r4((w_up - w_down) / (4 * t)),
                           "mean_ratio_minus1_all": [p_b.get("mean_ratio_minus1_all"), p_at.get("mean_ratio_minus1_all"), p_a.get("mean_ratio_minus1_all")],
                           "excess_at_step_over_neighbours": r4(p_at.get("mean_ratio_minus1_all", np.nan) - 0.5 * (p_b.get("mean_ratio_minus1_all", np.nan) + p_a.get("mean_ratio_minus1_all", np.nan))),
                           "bets_[below,at,above]": [p_b.get("bets"), p_at.get("bets"), p_a.get("bets")],
                           "paired_c5_[below,at,above]": [p_b.get("diff_c5"), p_at.get("diff_c5"), p_a.get("diff_c5")],
                           "se_[below,at,above]": [p_b.get("se"), p_at.get("se"), p_a.get("se")]}
    out({"section": "2_ladder_steps", "what": "last print exactly at a ladder step (tick width doubles+) vs one tick below / above; "
                                               "pure_ladder_prediction = (w_up - w_down) / (4 t) for a fair price uniform across the print's cell",
         "rows": steps})
    # band variants (fixed list) with BH
    variants = {name: (lp >= lo) & (lp < hi) for name, lo, hi in VARIANTS}
    variants["[4.5,6.5) minus tick 6.0"] = band & (lpr != 6.0)
    variants["[4.5,6.5) minus ticks 6.0 6.2 6.4"] = band & (lpr < 6.0)
    cells = {k: paired(g, np.flatnonzero(m)) for k, m in variants.items()}
    pvals = {k: bspl.one_sided_p(v["z"]) if v.get("z") is not None else float("nan") for k, v in cells.items()}
    bh = bspl.benjamini_hochberg(list(pvals.values()), 0.05)
    out({"section": "2_band_variants", "cells_tested": len(cells), "bh_rejections": int(bh.sum()),
         "columns": "compact + [bh_reject]", "rows": {k: compact(v) + [bool(b)] for (k, v), b in zip(cells.items(), bh)}})
    fs = g["field_size"].to_numpy()
    out({"section": "2_field_12plus_split", "rows": {
        "[4.5,6.5) field 12+": compact(paired(g, np.flatnonzero(band & (fs >= 12)))),
        "[4.5,6.0) field 12+": compact(paired(g, np.flatnonzero((lp >= 4.5) & (lp < 6.0) & (fs >= 12)))),
        "[6.0,6.5) field 12+": compact(paired(g, np.flatnonzero((lp >= 6.0) & (lp < 6.5) & (fs >= 12)))),
        "[4.5,6.5) field 2-11": compact(paired(g, np.flatnonzero(band & (fs < 12)))),
        "[4.5,6.0) field 2-11": compact(paired(g, np.flatnonzero((lp >= 4.5) & (lp < 6.0) & (fs < 12)))),
        "[6.0,6.5) field 2-11": compact(paired(g, np.flatnonzero((lp >= 6.0) & (lp < 6.5) & (fs < 12))))}})
    # by year for the two sides of the step
    out({"section": "2_sides_by_year", "rows": {
        str(y): {"[4.5,6.0)": compact(paired(g, np.flatnonzero((lp >= 4.5) & (lp < 6.0) & (years == y)))),
                 "[6.0,6.5)": compact(paired(g, np.flatnonzero((lp >= 6.0) & (lp < 6.5) & (years == y))))} for y in sorted(np.unique(years))}})

    # ---- 3. staleness / liquidity proxies
    has = g["has_t30"].to_numpy()
    same = has & (g["ltp_t30"].round(2).to_numpy() == lpr)
    liq = g.groupby("market_id")["has_t30"].transform("mean").to_numpy()
    vmap = pd.read_parquet(a.path, columns=["market_id", "venue"]).drop_duplicates("market_id").set_index("market_id")["venue"]
    venue = g["market_id"].map(vmap).fillna("").str.lower()
    metro = venue.apply(lambda v: any(m in v for m in METRO)).to_numpy()
    rank = g.groupby("market_id").cumcount().to_numpy() + 1
    lb = (1.0 / g["ltp_pre"]).groupby(g["market_id"]).transform("sum").to_numpy()
    bb = (1.0 / g["bsp"].fillna(np.inf)).groupby(g["market_id"]).transform("sum").to_numpy()
    proxies = {
        "runner traded 30 min out (has_t30)": band & has,
        "runner did NOT trade 30 min out": band & ~has,
        "has_t30 and ltp_pre == ltp_t30 (no print change in 30 min: stale-print candidate)": band & same,
        "has_t30 and ltp_pre != ltp_t30": band & has & ~same,
        "market liquidity: share of runners with t30 >= 0.5": band & (liq >= 0.5),
        "market liquidity: share with t30 < 0.5": band & (liq < 0.5),
        "market liquidity: no runner with t30": band & (liq == 0),
        "metro venue": band & metro,
        "non-metro venue": band & ~metro,
        "rank at ltp_pre 1": band & (rank == 1),
        "rank 2-3": band & (rank >= 2) & (rank <= 3),
        "rank 4+": band & (rank >= 4),
        "ltp book < 0.95 (stale after a late scratching?)": band & (lb < 0.95),
        "ltp book in [0.95,0.98)": band & (lb >= 0.95) & (lb < 0.98),
        "ltp book in [0.98,1.03]": band & (lb >= 0.98) & (lb <= 1.03),
        "ltp book > 1.03": band & (lb > 1.03),
        "bsp book in [0.98,1.03]": band & (bb >= 0.98) & (bb <= 1.03),
        "bsp book outside [0.98,1.03]": band & ((bb < 0.98) | (bb > 1.03)),
        "ltp book in [0.98,1.03] AND [4.5,6.0)": (lp >= 4.5) & (lp < 6.0) & (lb >= 0.98) & (lb <= 1.03),
        "ltp book in [0.98,1.03] AND [6.0,6.5)": (lp >= 6.0) & (lp < 6.5) & (lb >= 0.98) & (lb <= 1.03),
    }
    out({"section": "3_staleness_liquidity_proxies", "columns": "compact (see 1_by_year)",
         "rows": {k: compact(paired(g, np.flatnonzero(m))) for k, m in proxies.items()}})
    out({"section": "3b_step_vs_rest_by_staleness", "rows": {
        "tick 6.0, ltp_pre==ltp_t30": compact(paired(g, np.flatnonzero(same & (lpr == 6.0)))),
        "tick 6.0, ltp_pre!=ltp_t30 (has t30)": compact(paired(g, np.flatnonzero(has & ~same & (lpr == 6.0)))),
        "tick 6.0, no t30": compact(paired(g, np.flatnonzero(~has & (lpr == 6.0)))),
        "[4.5,6.0), ltp_pre==ltp_t30": compact(paired(g, np.flatnonzero(same & (lp >= 4.5) & (lp < 6.0)))),
        "[4.5,6.0), ltp_pre!=ltp_t30 (has t30)": compact(paired(g, np.flatnonzero(has & ~same & (lp >= 4.5) & (lp < 6.0)))),
        "[4.5,6.0), no t30": compact(paired(g, np.flatnonzero(~has & (lp >= 4.5) & (lp < 6.0))))}})

    # ---- 3c. clean-reference subsets: fresh print (runner traded 30 min out) AND no stale-scratching market
    # (ltp book in [0.98,1.03]) AND, where stated, not the ladder-step tick 6.0. CONDITIONAL on t30 trading.
    book_ok = (lb >= 0.98) & (lb <= 1.03)
    lo_side_m, hi_side_m = (lp >= 4.5) & (lp < 6.0), (lp >= 6.0) & (lp < 6.5)
    clean = {
        "band, book ok, has_t30": band & book_ok & has,
        "band, book ok, has_t30, print changed since t30": band & book_ok & has & ~same,
        "band, book ok, has_t30, minus tick 6.0": band & book_ok & has & (lpr != 6.0),
        "band, book ok, has_t30, print changed, minus tick 6.0": band & book_ok & has & ~same & (lpr != 6.0),
        "band, book ok, NO t30": band & book_ok & ~has,
        "[4.5,6.0), book ok, has_t30": lo_side_m & book_ok & has,
        "[4.5,6.0), book ok, NO t30": lo_side_m & book_ok & ~has,
        "[6.0,6.5), book ok, has_t30": hi_side_m & book_ok & has,
        "[6.0,6.5), book ok, NO t30": hi_side_m & book_ok & ~has,
        "tick 6.0, book ok, has_t30": (lpr == 6.0) & book_ok & has,
        "ticks 6.2-6.4, book ok, has_t30": (lpr > 6.0) & (lpr < 6.5) & book_ok & has,
        "band, book ok, metro": band & book_ok & metro,
        "band, book ok, metro, has_t30": band & book_ok & metro & has,
        "band, book ok, non-metro, has_t30": band & book_ok & ~metro & has,
        "band, book ok, market t30 share >= 0.5, has_t30": band & book_ok & (liq >= 0.5) & has,
        "band, book ok, field 12+, has_t30": band & book_ok & (fs >= 12) & has,
        "band, book ok, field 12+, NO t30": band & book_ok & (fs >= 12) & ~has,
    }
    out({"section": "3c_clean_reference_subsets", "columns": "compact (see 1_by_year)", "band_share_has_t30": r4(has[band].mean()),
         "band_share_book_ok": r4(book_ok[band].mean()),
         "has_t30_share_by_year": {str(y): r4(has[band & (years == y)].mean()) for y in sorted(np.unique(years))},
         "rows": {k: compact(paired(g, np.flatnonzero(m))) for k, m in clean.items()}})
    # mean log(bsp/ltp) for all runners by staleness proxy: is the SP drift itself concentrated in stale prints?
    out({"section": "3d_mean_log_ratio_all_runners_by_proxy", "rows": {
        k: {"n": int(m.sum()), "mean_log_ratio_all": r4(np.log(g["bsp"].to_numpy()[m] / lp[m]).mean()),
            "mean_log_ratio_winners": r4(np.log(g["bsp"].to_numpy()[m & g["won_b"].to_numpy()] / lp[m & g["won_b"].to_numpy()]).mean())}
        for k, m in {"band has_t30": band & has, "band no t30": band & ~has, "band has_t30 & book ok": band & has & book_ok,
                     "band no t30 & book ok": band & ~has & book_ok}.items()}})

    # ---- 4. survivorship: markets the original dropped
    cols = ["market_id", "market_name", "date", "selection_id", "ltp_pre", "bsp", "won", "market_type", "dead_heat_share"]
    w = pd.read_parquet(a.path, columns=cols).drop_duplicates(["market_id", "selection_id"])
    w = w[(w["market_type"] == "WIN") & w["won"].notna()]
    if a.end:
        w = w[w["date"] < pd.Timestamp(a.end)]
    w = w[~w["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b")]
    used = set(g["market_id"].unique())
    dropped = w[~w["market_id"].isin(used)].copy()
    n_win = dropped.groupby("market_id")["won"].transform("sum")
    priced = dropped["ltp_pre"].notna() & (dropped["ltp_pre"] > 1)
    dh = dropped[(n_win > 1) & priced & dropped["bsp"].notna() & (dropped["bsp"] > 1)]
    unpriced_mkts = dropped[(n_win == 1) & priced & dropped["bsp"].notna() & (dropped["bsp"] > 1)]
    rows = {}
    for name, sub in (("markets with a never-traded runner (band runners that did trade)", unpriced_mkts), ("dead-heat markets (band runners, dead-heat share applied)", dh)):
        s = sub[(sub["ltp_pre"] >= LO) & (sub["ltp_pre"] < HI)]
        if len(s) == 0:
            rows[name] = {"bets": 0}
            continue
        lpx, bx = s["ltp_pre"].to_numpy(dtype=float), s["bsp"].to_numpy(dtype=float)
        share = s["dead_heat_share"].fillna(0).to_numpy(dtype=float)
        wonx = s["won"].to_numpy() == 1
        # dead heat: stake is split, the winning part pays price x share, the rest is lost
        rb = np.where(wonx, (bx * share - 1.0) * np.where(bx * share > 1, 1 - C5, 1.0), -1.0)
        rl = np.where(wonx, (lpx * share - 1.0) * np.where(lpx * share > 1, 1 - C5, 1.0), -1.0)
        m, se = mean_se(rb - rl)
        rows[name] = {"bets": int(len(s)), "markets": int(s["market_id"].nunique()), "n_won": int(wonx.sum()), "diff_c5": r4(m), "se": r4(se),
                      "mean_ratio_minus1_all": r4((bx / lpx - 1).mean())}
    # pooled: claim cell + the never-traded-runner markets
    s = unpriced_mkts[(unpriced_mkts["ltp_pre"] >= LO) & (unpriced_mkts["ltp_pre"] < HI)]
    sub = g.iloc[idx_band]
    sub = sub[sub["bsp_ok"]]
    d_main = bspl.back_returns(sub["bsp"].to_numpy(dtype=float), sub["won_b"].to_numpy(), C5) - bspl.back_returns(sub["ltp_pre"].to_numpy(dtype=float), sub["won_b"].to_numpy(), C5)
    d_extra = bspl.back_returns(s["bsp"].to_numpy(dtype=float), s["won"].to_numpy() == 1, C5) - bspl.back_returns(s["ltp_pre"].to_numpy(dtype=float), s["won"].to_numpy() == 1, C5)
    m, se = mean_se(np.concatenate([d_main, d_extra]))
    rows["claim cell + never-traded-runner markets pooled"] = {"bets": int(len(d_main) + len(d_extra)), "diff_c5": r4(m), "se": r4(se)}
    out({"section": "4_survivorship_dropped_markets", "dropped_markets": int(dropped["market_id"].nunique()),
         "dropped_dead_heat_markets": int(dropped.loc[n_win > 1, "market_id"].nunique()), "rows": rows})

    # ---- 5. attainability: reference price
    att = {}
    for name, m in {"[4.5,6.5)": band, "[4.5,6.0)": (lp >= 4.5) & (lp < 6.0), "[6.0,6.5)": (lp >= 6.0) & (lp < 6.5)}.items():
        s = g.iloc[np.flatnonzero(m)]
        s = s[s["bsp_ok"]]
        l_, b_, w_ = s["ltp_pre"].to_numpy(dtype=float), s["bsp"].to_numpy(dtype=float), s["won_b"].to_numpy()
        rb = bspl.back_returns(b_, w_, C5)
        rl = bspl.back_returns(l_, w_, C5)
        rt = bspl.back_returns(bspl.tick_down(l_), w_, C5)
        ru = bspl.back_returns(bspl.tick_up(l_), w_, C5)
        rows = {}
        for ref, rr in (("vs last print", rl), ("vs one tick worse", rt), ("vs one tick better", ru), ("vs average of last and one tick worse", 0.5 * (rl + rt))):
            mm, ss = mean_se(rb - rr)
            rows[ref] = [int(len(s)), r4(mm), r4(ss), r4(mm / ss) if ss else None]
        att[name] = rows
    out({"section": "5_attainability_reference_price", "columns": "[bets, SP gain c5, se, z]", "rows": att})

    # ---- 6. winners vs all
    lo_side, hi_side = paired(g, np.flatnonzero((lp >= 4.5) & (lp < 6.0))), paired(g, np.flatnonzero((lp >= 6.0) & (lp < 6.5)))
    out({"section": "6_winners_vs_all", "rows": {
        name: {"mean_ratio_minus1_all": p["mean_ratio_minus1_all"], "mean_ratio_minus1_winners": p["mean_ratio_minus1_winners"],
               "observed_gain_c5": p["diff_c5"], "gain_if_won_indep_of_ratio_c5": p["gain_if_won_indep_of_ratio_c5"],
               "strike_vs_implied_ltp_vs_implied_bsp": [p["strike"], p["implied_ltp"], p["implied_bsp"]],
               "share_sp_better_1tick": p["share_sp_better_1tick"], "share_sp_worse_1tick": p["share_sp_worse_1tick"]}
        for name, p in (("[4.5,6.5)", head), ("[4.5,6.0)", lo_side), ("[6.0,6.5)", hi_side))},
        "note": "if observed ~ independent gain, SP carries no information about the result and the cell is the band's mean bsp/ltp excess"})
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

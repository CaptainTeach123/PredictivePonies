"""Adversarial verification of the "bsp_and_laying" claim (leakage / data-quality lens):

    Gallops runners with ltp_pre in [10, 15): backing at Betfair SP returns +0.88% (+/- 0.13%)
    more per unit than backing at the last traded price (5% commission on both).

    python scripts/bf_verify_bsp_and_laying_0.py [--end 2018-06-01]

Reuses load()/Rets() from scripts/bf_bsp_and_laying.py (imported, not edited) so the
reproduction is exact, then attacks the cell:

  0. data quality: duplicates (same race under two market_ids), dead heats, rows vs
     field_size, ltp_pre / bsp on the tick ladder, BSP book per market.
  1. reproduction of the headline cell (c0 / c5 / c8, SE, halves).
  2. decomposition by ltp_pre TICK (6.0 .. 30 and the ladder changes 30/50/100):
     mean log(bsp/ltp) for all runners and for winners, the paired gain per tick and
     each tick's share of the band's total gain. Spikes at the ladder changes
     (10, 20, 30, 50, 100: the tick widens there) are a mechanical artefact of
     "last traded price" being a discrete print, not an available price.
  3. the same band with the 10.0 tick removed, its lower / upper halves, and shifted
     bands with the same width (fixed list; BH over this family is reported).
  4. time: by year, by half, monthly slope.
  5. staleness / liquidity proxies: runner traded 30 min out or not; ltp_pre equal to
     ltp_t30 (no price change in the last 30 min: a stale-print candidate); market
     liquidity (share of runners with a t30 price); metro vs country venues; rank.
  6. attainability: SP vs the last price, vs one tick worse, and vs the average of the
     two (the last print is on either side of the spread about half the time).
  7. winners vs all runners: is the paired gain just strike x E[bsp - ltp]?

Prints JSON lines with summary numbers only (never horse names).
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
LADDER_CHANGES = (10.0, 20.0, 30.0, 50.0, 100.0)
METRO = ("flemington", "caulfield", "moonee valley", "sandown", "randwick", "rosehill", "warwick farm",
         "canterbury", "eagle farm", "doomben", "morphettville", "ascot", "belmont")
SHIFTED_BANDS = [(10.0, 15.0), (8.0, 13.0), (9.0, 14.0), (11.0, 16.0), (12.0, 17.0), (12.0, 18.0), (10.0, 12.5),
                 (12.5, 15.0), (10.5, 15.0), (11.0, 15.0), (10.0, 20.0), (12.5, 20.0), (6.5, 10.0), (15.0, 25.0)]


def out(obj) -> None:
    print(json.dumps(obj, default=bspl._json_default), flush=True)


def r4(x):
    return bspl.r4(x)


def mean_se(x: np.ndarray) -> tuple[float, float]:
    return bspl._mean_se(np.asarray(x, dtype=float))


def paired(df: pd.DataFrame, idx: np.ndarray, digits: int = 4) -> dict:
    """Paired (bsp return - ltp_pre return) on runners idx with both prices.
    Also the mean log ratio for all runners and for winners."""
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
           "median_ltp": r4(np.median(lp))}
    for c in COMMS:
        d = bspl.back_returns(bsp, won, c) - bspl.back_returns(lp, won, c)
        res[f"diff_c{int(c * 100)}"] = round(float(d.mean()), digits)
    d5 = bspl.back_returns(bsp, won, C5) - bspl.back_returns(lp, won, C5)
    m, se = mean_se(d5)
    res["se"] = round(se, digits) if not math.isnan(se) else None
    res["z"] = r4(m / se) if se and se > 0 else None
    h = []
    for mask in (first, ~first):
        hm, hs = mean_se(d5[mask])
        h.append([int(mask.sum()), round(hm, digits) if not math.isnan(hm) else None,
                  round(hs, digits) if not math.isnan(hs) else None])
    res["halves_[n,diff_c5,se]"] = h
    res["both_halves_over_2se"] = bool(all(v[1] is not None and v[2] and v[1] > 2 * v[2] for v in h))
    lr = np.log(bsp / lp)
    res["mean_log_ratio_all"] = r4(lr.mean())
    res["mean_ratio_minus1_all"] = r4((bsp / lp - 1).mean())
    res["mean_log_ratio_winners"] = r4(lr[won].mean()) if won.any() else None
    res["mean_ratio_minus1_winners"] = r4((bsp[won] / lp[won] - 1).mean()) if won.any() else None
    # if won were independent of the ratio, the paired gain would be strike x E[(bsp-ltp)] x (1-c)
    res["gain_if_won_indep_of_ratio_c5"] = r4(won.mean() * (bsp - lp).mean() * (1 - C5))
    res["back_ltp_c5"] = r4(bspl.back_returns(lp, won, C5).mean())
    res["back_bsp_c5"] = r4(bspl.back_returns(bsp, won, C5).mean())
    res["back_ltp_tick_down_c5"] = r4(bspl.back_returns(bspl.tick_down(lp), won, C5).mean())
    return res


def compact(p: dict) -> list:
    """[bets, n_won, diff_c0, diff_c5, diff_c8, se, z, half1, half2, both_halves, mean_log_ratio_all, mean_log_ratio_winners]"""
    if p.get("bets", 0) == 0:
        return [0]
    h = p["halves_[n,diff_c5,se]"]
    return [p["bets"], p["n_won"], p["diff_c0"], p["diff_c5"], p["diff_c8"], p["se"], p["z"], h[0][1], h[1][1],
            p["both_halves_over_2se"], p["mean_log_ratio_all"], p["mean_log_ratio_winners"]]


def data_quality(raw_path: Path, g: pd.DataFrame) -> dict:
    cols = ["market_id", "event_id", "venue", "market_name", "start", "field_size", "selection_id", "ltp_pre", "bsp",
            "won", "n_winners", "market_type"]
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
    res = {"rows_raw_all_codes": int(n_raw), "gallops_rows": int(len(w)), "gallops_markets": int(len(mk)),
           "n_winners_values": {str(k): int(v) for k, v in mk["n_winners"].value_counts().items()},
           "won_sum_per_market": {str(k): int(v) for k, v in ws.value_counts().items()},
           "markets_rows_ne_field_size": int((rows != fs).sum()),
           "win_markets_per_event_start": {str(k): int(v) for k, v in mk.groupby(["event_id", "start"])["market_id"].nunique().value_counts().items()},
           "win_markets_per_venue_start": {str(k): int(v) for k, v in mk.groupby(["venue", "start"])["market_id"].nunique().value_counts().items()}}
    lp = w["ltp_pre"].dropna().to_numpy(dtype=float)
    inc = bspl._increment(lp - 1e-9)
    res["ltp_pre_off_ladder_share"] = r4((np.abs(np.round(lp / inc) * inc - lp) > 1e-6).mean())
    b = w["bsp"].dropna().to_numpy(dtype=float)
    incb = bspl._increment(b - 1e-9)
    res["bsp_off_ladder_share"] = r4((np.abs(np.round(b / incb) * incb - b) > 1e-6).mean())
    res["bsp_equals_ltp_share"] = r4((w["bsp"] == w["ltp_pre"]).mean())
    # BSP book per market (used markets only)
    bb = (1.0 / g["bsp"]).groupby(g["market_id"]).sum()
    lb = (1.0 / g["ltp_pre"]).groupby(g["market_id"]).sum()
    res["bsp_book_q05_50_95"] = [r4(v) for v in bb.quantile([0.05, 0.5, 0.95])]
    res["ltp_book_q05_50_95"] = [r4(v) for v in lb.quantile([0.05, 0.5, 0.95])]
    res["markets_bsp_book_outside_0.97_1.05"] = int(((bb < 0.97) | (bb > 1.05)).sum())
    res["markets_ltp_book_outside_0.97_1.05"] = int(((lb < 0.97) | (lb > 1.05)).sum())
    return res


def tick_profile(g: pd.DataFrame, lo: float, hi: float) -> dict:
    lp = g["ltp_pre"].round(2).to_numpy()
    ok = g["bsp_ok"].to_numpy()
    ticks = sorted(v for v in pd.unique(lp[(lp >= lo) & (lp < hi) & ok]) if (lp == v).sum() >= 200)
    rows = {}
    for t in ticks:
        idx = np.flatnonzero((lp == t) & ok)
        p = paired(g, idx)
        rows[f"{t:g}"] = [p["bets"], p["n_won"], p["mean_log_ratio_all"], p["mean_log_ratio_winners"], p["diff_c5"], p["se"],
                          r4(p["diff_c5"] * p["bets"])]
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
         "t30_coverage": r4(g["has_t30"].mean())})
    out({"section": "0_data_quality", **data_quality(a.path, g)})

    lp = g["ltp_pre"].to_numpy(dtype=float)
    lpr = g["ltp_pre"].round(2).to_numpy()
    band = (lp >= 10) & (lp < 15)
    idx_band = np.flatnonzero(band)
    head = paired(g, idx_band)
    out({"section": "1_reproduce", "cell": "gallops ltp_pre in [10,15) paired SP minus LTP", **head})

    # ---- 2. per-tick decomposition
    prof = tick_profile(g, 6.0, 30.0)
    out({"section": "2_by_tick", "columns": "[bets, n_won, mean_log(bsp/ltp) all, mean_log winners, paired diff c5, se, bets*diff (contribution)]",
         "rows": prof})
    band_total = sum(v[6] for k, v in prof.items() if 10 <= float(k) < 15)
    contrib = {k: r4(v[6] / band_total) for k, v in prof.items() if 10 <= float(k) < 15}
    out({"section": "2_band_contribution_share_by_tick", "band_total_gain_units": r4(band_total), "share": contrib})
    spikes = {}
    for lc in LADDER_CHANGES:
        at = np.flatnonzero((lpr == lc) & g["bsp_ok"].to_numpy())
        below = np.flatnonzero((lpr < lc) & (lpr >= bspl.tick_down(np.array([lc]))[0] - 1e-9) & g["bsp_ok"].to_numpy())
        above = np.flatnonzero((lpr > lc) & (lpr <= bspl.tick_up(np.array([lc]))[0] + 1e-9) & g["bsp_ok"].to_numpy())
        spikes[f"{lc:g}"] = {"at": compact(paired(g, at)), "tick_below": compact(paired(g, below)), "tick_above": compact(paired(g, above))}
    out({"section": "2_ladder_change_spikes", "columns": "compact = [bets, n_won, diff_c0, diff_c5, diff_c8, se, z, half1, half2, both_halves, mean_log_ratio_all, mean_log_ratio_winners]",
         "rows": spikes})

    # ---- 3. band variants and shifted bands
    variants = {"[10,15)": band, "[10,15) minus tick 10.0": band & (lpr != 10.0), "[10,15) minus ticks 10.0 and 10.5": band & (lpr > 10.5)}
    for lo, hi in SHIFTED_BANDS:
        variants[f"[{lo:g},{hi:g})"] = (lp >= lo) & (lp < hi)
    cells = {}
    for name, m in variants.items():
        cells[name] = paired(g, np.flatnonzero(m))
    pvals = {k: bspl.one_sided_p(v["z"]) if v.get("z") is not None else float("nan") for k, v in cells.items()}
    bh = bspl.benjamini_hochberg(list(pvals.values()), 0.05)
    out({"section": "3_band_variants", "cells_tested": len(cells), "bh_rejections": int(bh.sum()),
         "rows": {k: compact(v) + [bool(b)] for (k, v), b in zip(cells.items(), bh)}})

    # ---- 4. time
    years = g["date"].dt.year.to_numpy()
    by_year = {str(y): compact(paired(g, np.flatnonzero(band & (years == y)))) for y in sorted(np.unique(years))}
    out({"section": "4_by_year", "rows": by_year})
    sub = g.iloc[idx_band]
    sub = sub[sub["bsp_ok"]]
    d5 = bspl.back_returns(sub["bsp"].to_numpy(dtype=float), sub["won_b"].to_numpy(), C5) - bspl.back_returns(sub["ltp_pre"].to_numpy(dtype=float), sub["won_b"].to_numpy(), C5)
    mo = sub["month"].to_numpy()
    mdf = pd.DataFrame({"m": mo, "d": d5}).groupby("m")["d"].agg(["mean", "size"])
    x = np.arange(len(mdf), dtype=float)
    wts = mdf["size"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    W = np.sqrt(wts)
    beta, *_ = np.linalg.lstsq(X * W[:, None], mdf["mean"].to_numpy() * W, rcond=None)
    resid = mdf["mean"].to_numpy() - X @ beta
    s2 = float((wts * resid ** 2).sum() / max(len(x) - 2, 1))
    cov = s2 * np.linalg.inv((X * wts[:, None]).T @ X)
    last12 = np.flatnonzero(np.isin(mo, mdf.index[-12:]))
    m12, s12 = mean_se(d5[last12])
    out({"section": "4_monthly_trend", "months": int(len(mdf)), "slope_per_month": r4(beta[1]), "slope_se": r4(math.sqrt(cov[1, 1])),
         "fitted_first_month": r4(beta[0]), "fitted_last_month": r4(beta[0] + beta[1] * x[-1]),
         "last_12_months_diff_c5": r4(m12), "last_12_months_se": r4(s12), "last_12_months_bets": int(len(last12))})

    # ---- 5. staleness / liquidity proxies
    has = g["has_t30"].to_numpy()
    same = has & (g["ltp_t30"].round(2).to_numpy() == lpr)
    liq = g.groupby("market_id")["has_t30"].transform("mean").to_numpy()
    venue = g["venue"].fillna("").str.lower()
    metro = venue.apply(lambda v: any(m in v for m in METRO)).to_numpy()
    rank = g.groupby("market_id").cumcount().to_numpy() + 1  # df sorted by ltp_pre within market in load()
    fs = g["field_size"].to_numpy()
    lb = (1.0 / g["ltp_pre"]).groupby(g["market_id"]).transform("sum").to_numpy()
    bb = (1.0 / g["bsp"].fillna(np.inf)).groupby(g["market_id"]).transform("sum").to_numpy()
    proxies = {
        "runner traded 30 min out (has_t30)": band & has,
        "runner did NOT trade 30 min out": band & ~has,
        "has_t30 and ltp_pre == ltp_t30 (no print change in 30 min)": band & same,
        "has_t30 and ltp_pre != ltp_t30": band & has & ~same,
        "market liquidity: share with t30 >= 0.5": band & (liq >= 0.5),
        "market liquidity: share with t30 < 0.5": band & (liq < 0.5),
        "market liquidity: no runner with t30": band & (liq == 0),
        "metro venue": band & metro,
        "non-metro venue": band & ~metro,
        "rank at ltp_pre 1-3": band & (rank <= 3),
        "rank 4-6": band & (rank >= 4) & (rank <= 6),
        "rank 7+": band & (rank >= 7),
        "field 2-8": band & (fs <= 8),
        "field 9-11": band & (fs >= 9) & (fs <= 11),
        "field 12+": band & (fs >= 12),
        "ltp book in [0.98,1.03]": band & (lb >= 0.98) & (lb <= 1.03),
        "ltp book outside [0.98,1.03]": band & ((lb < 0.98) | (lb > 1.03)),
        "bsp book in [0.98,1.03]": band & (bb >= 0.98) & (bb <= 1.03),
        "bsp book outside [0.98,1.03]": band & ((bb < 0.98) | (bb > 1.03)),
    }
    out({"section": "5_staleness_liquidity_proxies", "columns": "compact (see section 2)",
         "rows": {k: compact(paired(g, np.flatnonzero(m))) for k, m in proxies.items()}})
    # the stale-print proxy by tick 10.0 vs the rest
    out({"section": "5b_tick10_vs_rest_by_staleness", "rows": {
        "tick 10.0, ltp_pre==ltp_t30": compact(paired(g, np.flatnonzero(same & (lpr == 10.0)))),
        "tick 10.0, ltp_pre!=ltp_t30 (has t30)": compact(paired(g, np.flatnonzero(has & ~same & (lpr == 10.0)))),
        "tick 10.0, no t30": compact(paired(g, np.flatnonzero(~has & (lpr == 10.0)))),
        "ticks 10.5-14.5, ltp_pre==ltp_t30": compact(paired(g, np.flatnonzero(same & band & (lpr != 10.0)))),
        "ticks 10.5-14.5, ltp_pre!=ltp_t30 (has t30)": compact(paired(g, np.flatnonzero(has & ~same & band & (lpr != 10.0)))),
        "ticks 10.5-14.5, no t30": compact(paired(g, np.flatnonzero(~has & band & (lpr != 10.0))))}})

    # ---- 6. attainability: SP vs last price / one tick worse / the average of the two
    att = {}
    for name, m in {"[10,15)": band, "[10,15) minus tick 10.0": band & (lpr != 10.0), "[12.5,15)": (lp >= 12.5) & (lp < 15),
                    "[10,12.5)": (lp >= 10) & (lp < 12.5)}.items():
        s = g.iloc[np.flatnonzero(m)]
        s = s[s["bsp_ok"]]
        l_, b_, w_ = s["ltp_pre"].to_numpy(dtype=float), s["bsp"].to_numpy(dtype=float), s["won_b"].to_numpy()
        rb = bspl.back_returns(b_, w_, C5)
        rl = bspl.back_returns(l_, w_, C5)
        rt = bspl.back_returns(bspl.tick_down(l_), w_, C5)
        rows = {}
        for ref, rr in (("vs last price", rl), ("vs one tick worse", rt), ("vs average of the two", 0.5 * (rl + rt))):
            mm, ss = mean_se(rb - rr)
            rows[ref] = [int(len(s)), r4(mm), r4(ss), r4(mm / ss) if ss else None]
        att[name] = rows
    out({"section": "6_attainability_reference_price", "columns": "[bets, SP gain c5, se, z]", "rows": att})

    # ---- 7. winners vs all runners
    out({"section": "7_winners_vs_all", "band": "[10,15)",
         "mean_ratio_minus1_all": head["mean_ratio_minus1_all"], "mean_ratio_minus1_winners": head["mean_ratio_minus1_winners"],
         "observed_gain_c5": head["diff_c5"], "gain_if_won_indep_of_ratio_c5": head["gain_if_won_indep_of_ratio_c5"],
         "note": "if the two gains agree, SP carries no information about the result: the cell is the band's mean bsp/ltp excess"})
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

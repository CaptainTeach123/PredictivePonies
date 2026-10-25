"""Adversarial verification (statistics + execution lens) of the "bsp_and_laying" claim:

    AU gallops WIN runners with ltp_pre in [10, 15): backing 1 unit at Betfair SP
    returns more than backing the same runner at the last traded price:
    paired difference +0.88% +/- 0.13% after 5% commission on both
    (n 77,012; halves +1.26% / +0.53%).  Relative finding; backing at SP itself
    loses -2.6%.

    python scripts/bf_verify_bsp_and_laying_1.py [--end 2018-06-01] [--boot 400] [--splits 200]

Same data filters as scripts/bf_bsp_and_laying.py (dedupe, settled WIN, exactly
one winner, every runner traded pre-off, gallops = market name without
Pace/Trot).  Nothing is fitted.  What is re-run / added:

  1. the cell itself: paired bsp-return minus ltp_pre-return at commission
     0 / 5 / 8 / 10%, per-bet SE, halves at the median day (as the original);
  2. cluster-robust SEs by market, by meeting (event_id) and by race day, a
     race-day cluster bootstrap, and a month-as-unit t-test (36 months);
  3. the split: calendar years, half-years, odd/even ISO weeks, and 200 random
     day-level half splits (how often the both-halves > 2 SE rule holds);
  4. the band: 1-point sub-bands, shifted/widened windows, inclusive upper
     edge, and by field size (the original 2-5 cell is -4.7%);
  5. influence: share of the paired gain from winners whose SP was > 20% /
     > 50% above the last price (stale-quote suspects); winsorised ratio;
  6. mechanism: strike rate by bsp/ltp_pre move bucket vs implied at ltp_pre
     and at bsp (is the SP drift informative, or free?);
  7. execution: liquidity proxies (market book at ltp_pre near 100%; runner
     traded 30 min out), matched one tick worse than the last trade, the
     "limit at last price with SP fallback" execution (= worse of the two),
     SP-better / SP-worse shares on winners only, mean(bsp - ltp | won);
  8. multiple testing: the cell's p-value from the clustered SE against a
     Bonferroni over the 556 cells of the original script.

Prints JSON lines with summary numbers only; never prints horse names.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

PATH = HERE.parent / "data" / "raw" / "betfair" / "split" / "au_win.parquet"
COMMISSIONS = (0.0, 0.05, 0.08, 0.10)
C_TEST = 0.05
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1001.0, 10.0)]
FIELD_EDGES = [2, 6, 9, 12, 99]
FIELD_LABELS = ["2-5", "6-8", "9-11", "12+"]
N_CELLS_ORIGINAL = 556


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    raise TypeError(str(type(o)))


def out(obj) -> None:
    print(json.dumps(obj, default=_json_default), flush=True)


def r4(x):
    x = float(x)
    return None if math.isnan(x) else round(x, 5)


def back_returns(price, won, c):
    price = np.asarray(price, dtype=float)
    return np.where(won, (price - 1.0) * (1.0 - c), -1.0)


def _increment(q):
    inc = np.full_like(q, 10.0)
    for lo, hi, step in TICKS:
        inc = np.where((q >= lo) & (q < hi), step, inc)
    return inc


def tick_down(price):
    p = np.asarray(price, dtype=float)
    return np.maximum(1.01, np.round(p - _increment(p - 1e-9), 2))


def tick_up(price):
    p = np.asarray(price, dtype=float)
    return np.minimum(1000.0, np.round(p + _increment(p + 1e-9), 2))


def one_sided_p(z):
    if z is None or math.isnan(z):
        return float("nan")
    return 0.5 * math.erfc(z / math.sqrt(2))


def mean_se(x):
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else float("nan")), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(n))


def cluster_se(ret, cl):
    """SE of the mean with observations clustered by ``cl``."""
    n = len(ret)
    if n < 2:
        return float("nan")
    d = ret - ret.mean()
    s = pd.Series(d).groupby(np.asarray(cl)).sum().to_numpy()
    g = len(s)
    if g < 2:
        return float("nan")
    return float(math.sqrt((s ** 2).sum() * g / (g - 1)) / n)


def cluster_boot(ret, cl, n_boot, seed=0):
    """Percentile interval for the mean, resampling clusters."""
    df = pd.DataFrame({"r": ret, "c": np.asarray(cl)})
    agg = df.groupby("c")["r"].agg(["sum", "count"])
    s, k = agg["sum"].to_numpy(), agg["count"].to_numpy()
    g = len(s)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, g, g)
        means[b] = s[idx].sum() / k[idx].sum()
    return {"boot_se": r4(means.std(ddof=1)), "ci95": [r4(np.quantile(means, 0.025)), r4(np.quantile(means, 0.975))],
            "share_boot_below_0": r4((means <= 0).mean()), "clusters": int(g)}


def basic(ret, won=None):
    m, se = mean_se(ret)
    res = {"bets": int(len(ret)), "roi": r4(m), "se": r4(se), "z": r4(m / se) if se and se > 0 else None}
    if won is not None:
        res["n_won"] = int(won.sum())
    return res


def paired(sub, c=C_TEST):
    """bsp return minus ltp_pre return, 1 unit backed on each."""
    won = sub["won_b"].to_numpy()
    return back_returns(sub["bsp"].to_numpy(), won, c) - back_returns(sub["ltp_pre"].to_numpy(), won, c)


def full_cell(sub, mid, c=C_TEST, boot=0):
    d = paired(sub, c)
    won = sub["won_b"].to_numpy()
    res = basic(d, won)
    res["se_cl_market"] = r4(cluster_se(d, sub["market_id"].to_numpy()))
    res["se_cl_event"] = r4(cluster_se(d, sub["event_id"].to_numpy()))
    res["se_cl_date"] = r4(cluster_se(d, sub["date"].to_numpy()))
    first = (sub["date"] < mid).to_numpy()
    halves = []
    for mask in (first, ~first):
        h = basic(d[mask], won[mask])
        h["se_cl_date"] = r4(cluster_se(d[mask], sub["date"].to_numpy()[mask]))
        halves.append(h)
    res["halves"] = halves
    res["both_halves_gt_2se"] = bool(all(h["roi"] is not None and h["se"] and h["roi"] > 2 * h["se"] for h in halves))
    res["both_halves_gt_2se_cl_date"] = bool(all(h["roi"] is not None and h["se_cl_date"] and h["roi"] > 2 * h["se_cl_date"] for h in halves))
    if boot:
        res["day_cluster_bootstrap"] = cluster_boot(d, sub["date"].to_numpy(), boot)
    return res


def load(end):
    cols = ["market_id", "event_id", "market_name", "date", "field_size", "selection_id", "ltp_pre", "ltp_t30",
            "bsp", "won", "market_type"]
    df = pd.read_parquet(PATH, columns=cols)
    n_raw = len(df)
    df = df.drop_duplicates(["market_id", "selection_id"])
    df = df[(df["market_type"] == "WIN") & df["won"].notna()]
    if end:
        df = df[df["date"] < pd.Timestamp(end)]
    one_winner = df.groupby("market_id")["won"].transform("sum") == 1
    ok = df["ltp_pre"].notna() & (df["ltp_pre"] > 1)
    all_priced = ok.groupby(df["market_id"]).transform("min").astype(bool)
    df = df[one_winner & all_priced].drop(columns=["market_type"]).copy()
    df["harness"] = df["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b")
    df = df[~df["harness"]].copy()  # gallops only
    df["won_b"] = df["won"].to_numpy() == 1
    df["bsp_ok"] = df["bsp"].notna() & (df["bsp"] > 1)
    df["has_t30"] = df["ltp_t30"].notna() & (df["ltp_t30"] > 1)
    df["fgroup"] = pd.cut(df["field_size"], FIELD_EDGES, right=False, labels=FIELD_LABELS).astype(str)
    df["book"] = (1.0 / df["ltp_pre"]).groupby(df["market_id"]).transform("sum")
    df["book_bsp"] = (1.0 / df["bsp"].where(df["bsp_ok"])).groupby(df["market_id"]).transform("sum")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["half_year"] = df["date"].dt.year.astype(str) + "H" + ((df["date"].dt.month > 6).astype(int) + 1).astype(str)
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
    days = sorted(df["date"].unique())
    mid = pd.Timestamp(days[len(days) // 2])
    info = {"rows_raw": int(n_raw), "gallops_markets": int(df["market_id"].nunique()), "gallops_runners": int(len(df)),
            "from": str(pd.Timestamp(days[0]).date()), "to": str(pd.Timestamp(days[-1]).date()),
            "mid_date_for_halves": str(mid.date()), "bsp_runner_coverage": r4(df["bsp_ok"].mean())}
    return df, info, mid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--splits", type=int, default=200)
    a = ap.parse_args()
    t0 = time.time()
    df, info, mid = load(a.end)
    out({"section": "data", **info, "end": a.end})

    lp = df["ltp_pre"].to_numpy()
    band = (lp >= 10.0) & (lp < 15.0) & df["bsp_ok"].to_numpy()
    g = df[band].copy()
    won = g["won_b"].to_numpy()
    ratio = g["bsp"].to_numpy() / g["ltp_pre"].to_numpy()
    g["ratio"] = ratio

    # ---- 1. reproduction -----------------------------------------------------------------------
    rep = {f"paired_c{int(c * 100)}": basic(paired(g, c)) for c in COMMISSIONS}
    rep["back_ltp_pre"] = {f"c{int(c * 100)}": r4(back_returns(g["ltp_pre"], won, c).mean()) for c in COMMISSIONS}
    rep["back_bsp"] = {f"c{int(c * 100)}": r4(back_returns(g["bsp"], won, c).mean()) for c in COMMISSIONS}
    rep["back_ltp_tick_down"] = {f"c{int(c * 100)}": r4(back_returns(tick_down(g["ltp_pre"]), won, c).mean()) for c in COMMISSIONS}
    rep["strike"] = r4(won.mean())
    rep["implied_ltp"] = r4((1.0 / g["ltp_pre"]).mean())
    rep["implied_bsp"] = r4((1.0 / g["bsp"]).mean())
    rep["mean_bsp_minus_ltp_given_won"] = r4((g["bsp"] - g["ltp_pre"])[won].mean())
    rep["median_ratio_all"] = r4(np.median(ratio))
    rep["mean_ratio_all"] = r4(ratio.mean())
    rep["mean_inv_ratio_all"] = r4((1.0 / ratio).mean())
    rep["median_ratio_won"] = r4(np.median(ratio[won]))
    rep["median_ratio_lost"] = r4(np.median(ratio[~won]))
    rep["mean_log_ratio_won"] = r4(np.log(ratio[won]).mean())
    rep["mean_log_ratio_lost"] = r4(np.log(ratio[~won]).mean())
    out({"section": "1_reproduction_gallops_ltp_10_15", **rep,
         "note": "paired = back 1 at bsp minus back 1 at ltp_pre on the same runner, commission on net winnings of each"})

    # ---- 2. clustered SEs, bootstrap, month-as-unit ------------------------------------------------
    fc = full_cell(g, mid, C_TEST, boot=a.boot)
    d5 = paired(g, C_TEST)
    mon = pd.DataFrame({"d": d5, "m": g["month"].to_numpy()}).groupby("m")["d"].agg(["mean", "count"])
    mm, ms = mean_se(mon["mean"].to_numpy())
    wmean = float((mon["mean"] * mon["count"]).sum() / mon["count"].sum())
    out({"section": "2_clustered_se_c5", **fc,
         "months_as_units": {"months": int(len(mon)), "unweighted_mean": r4(mm), "se_across_months": r4(ms),
                             "t": r4(mm / ms) if ms else None, "share_months_positive": r4((mon["mean"] > 0).mean()),
                             "weighted_mean": r4(wmean)},
         "p_one_sided_cl_date": one_sided_p(fc["roi"] / fc["se_cl_date"]) if fc["se_cl_date"] else None,
         "bonferroni_threshold_556_cells": r4(0.05 / N_CELLS_ORIGINAL)})

    # ---- 3. alternative splits -----------------------------------------------------------------
    by_year = {}
    for y, gy in g.groupby("year"):
        dy = paired(gy, C_TEST)
        b = basic(dy, gy["won_b"].to_numpy())
        b["se_cl_date"] = r4(cluster_se(dy, gy["date"].to_numpy()))
        by_year[str(y)] = b
    by_hy = {}
    for hy, gy in g.groupby("half_year"):
        dy = paired(gy, C_TEST)
        b = basic(dy, gy["won_b"].to_numpy())
        b["se_cl_date"] = r4(cluster_se(dy, gy["date"].to_numpy()))
        by_hy[hy] = b
    odd = (g["iso_week"] % 2 == 1).to_numpy()
    by_week = {"odd_weeks": basic(d5[odd], won[odd]), "even_weeks": basic(d5[~odd], won[~odd])}
    # random day-level half splits
    days = g["date"].unique()
    rng = np.random.default_rng(42)
    day_sum = pd.DataFrame({"d": d5, "day": g["date"].to_numpy()}).groupby("day")["d"].agg(["sum", "count", lambda x: (x ** 2).sum()])
    day_sum.columns = ["s", "n", "ss"]
    both = 0
    min_half_z = []
    for _ in range(a.splits):
        pick = rng.permutation(len(day_sum)) < len(day_sum) // 2
        zs = []
        for mask in (pick, ~pick):
            s, n, ss = day_sum["s"].to_numpy()[mask].sum(), day_sum["n"].to_numpy()[mask].sum(), day_sum["ss"].to_numpy()[mask].sum()
            m = s / n
            var = (ss - n * m * m) / (n - 1)
            zs.append(m / math.sqrt(var / n))
        min_half_z.append(min(zs))
        both += all(z > 2 for z in zs)
    out({"section": "3_alternative_splits_c5", "by_year": by_year, "by_half_year": by_hy, "odd_even_iso_weeks": by_week,
         "random_day_half_splits": {"splits": a.splits, "share_both_halves_gt_2se": r4(both / a.splits),
                                    "min_half_z_q10_q50_q90": [r4(v) for v in np.quantile(min_half_z, [0.1, 0.5, 0.9])]}})

    # ---- 4. band sensitivity --------------------------------------------------------------------
    allg = df[df["bsp_ok"]].copy()
    lp_all = allg["ltp_pre"].to_numpy()
    windows = {"[10,11)": (10, 11), "[11,12)": (11, 12), "[12,13)": (12, 13), "[13,14)": (13, 14), "[14,15)": (14, 15),
               "[8,12)": (8, 12), "[9,14)": (9, 14), "[11,16)": (11, 16), "[12,18)": (12, 18), "[10,15]": (10, 15.0001),
               "[6.5,10)": (6.5, 10), "[15,25)": (15, 25), "[4.5,25)": (4.5, 25), "[6.5,15)": (6.5, 15), "[10,20)": (10, 20)}
    bands = {}
    for name, (lo, hi) in windows.items():
        sel = (lp_all >= lo) & (lp_all < hi)
        sub = allg[sel]
        dsub = paired(sub, C_TEST)
        b = basic(dsub, sub["won_b"].to_numpy())
        b["se_cl_date"] = r4(cluster_se(dsub, sub["date"].to_numpy()))
        first = (sub["date"] < mid).to_numpy()
        b["halves"] = [r4(dsub[first].mean()), r4(dsub[~first].mean())]
        b["halves_se"] = [r4(mean_se(dsub[first])[1]), r4(mean_se(dsub[~first])[1])]
        bands[name] = b
    by_field = {}
    for f, gf in g.groupby("fgroup"):
        dfld = paired(gf, C_TEST)
        b = basic(dfld, gf["won_b"].to_numpy())
        b["se_cl_date"] = r4(cluster_se(dfld, gf["date"].to_numpy()))
        b["median_field"] = int(gf["field_size"].median())
        by_field[f] = b
    out({"section": "4_band_sensitivity_c5", "windows": bands, "ltp_10_15_by_field_size": by_field})

    # ---- 5. influence of large SP drifts ---------------------------------------------------------
    tot = d5.sum()
    infl = {}
    for thr in (1.1, 1.2, 1.5, 2.0):
        big = won & (ratio > thr)
        infl[f"winners_ratio_gt_{thr:g}"] = {"n": int(big.sum()), "share_of_total_paired_gain": r4(d5[big].sum() / tot) if tot else None,
                                            "roi_excluding_them": r4(d5[~big].mean())}
    for thr in (0.9, 0.8):
        small = won & (ratio < thr)
        infl[f"winners_ratio_lt_{thr:g}"] = {"n": int(small.sum()), "contribution_to_roi": r4(d5[small].sum() / len(d5))}
    # winsorise the ratio: cap bsp at ltp*cap and floor at ltp/cap
    wins = {}
    for cap in (1.25, 1.5, 2.0):
        bsp_w = np.clip(g["bsp"].to_numpy(), g["ltp_pre"].to_numpy() / cap, g["ltp_pre"].to_numpy() * cap)
        dw = back_returns(bsp_w, won, C_TEST) - back_returns(g["ltp_pre"].to_numpy(), won, C_TEST)
        wins[f"cap_{cap:g}"] = basic(dw)
    # drop the k most favourable winners
    order = np.argsort(-d5)
    drop_k = {str(k): r4(np.delete(d5, order[:k]).mean()) for k in (10, 50, 200, 500)}
    out({"section": "5_influence_c5", "total_paired_gain_units": r4(tot), "n_won": int(won.sum()), **infl,
         "winsorised_ratio": wins, "roi_after_dropping_k_best_winners": drop_k,
         "ratio_quantiles_winners": [r4(v) for v in np.quantile(ratio[won], [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])],
         "ratio_quantiles_losers": [r4(v) for v in np.quantile(ratio[~won], [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])]})

    # ---- 6. mechanism: is the SP drift informative? -------------------------------------------------
    cuts = [0, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 99]
    labels = ["<0.85", "0.85-0.95", "0.95-1.0", "1.0-1.05", "1.05-1.15", "1.15-1.3", ">=1.3"]
    g["rb"] = pd.cut(g["ratio"], cuts, right=False, labels=labels).astype(str)
    mech = {}
    for lab in labels:
        s = g[g["rb"] == lab]
        if len(s) == 0:
            continue
        w = s["won_b"].to_numpy()
        n = len(s)
        strike = w.mean()
        il, ib = (1.0 / s["ltp_pre"]).mean(), (1.0 / s["bsp"]).mean()
        mech[lab] = {"n": n, "share": r4(n / len(g)), "strike": r4(strike), "implied_ltp": r4(il), "implied_bsp": r4(ib),
                     "strike_se": r4(math.sqrt(strike * (1 - strike) / n)),
                     "paired_c5": r4(paired(s, C_TEST).mean()), "contribution_to_roi": r4(paired(s, C_TEST).sum() / len(g)),
                     "back_ltp_c0": r4(back_returns(s["ltp_pre"], w, 0.0).mean()), "back_bsp_c0": r4(back_returns(s["bsp"], w, 0.0).mean())}
    # log-loss of 1/ltp vs 1/bsp (normalised within the band only as a crude check) -- use raw implied probs
    pl, pb = 1.0 / g["ltp_pre"].to_numpy(), 1.0 / g["bsp"].to_numpy()
    ll_l = -(won * np.log(pl) + (~won) * np.log(1 - pl)).mean()
    ll_b = -(won * np.log(pb) + (~won) * np.log(1 - pb)).mean()
    out({"section": "6_mechanism_strike_by_bsp_over_ltp_bucket", "buckets": mech,
         "runner_logloss_at_1_over_ltp": r4(ll_l), "runner_logloss_at_1_over_bsp": r4(ll_b),
         "note": "if strike tracks implied_bsp inside each bucket the SP move carries information (SP gain is a Jensen artefact of a "
                 "stale quote); if strike tracks implied_ltp the move is noise and SP really is the better price"})

    # ---- 7. execution / liquidity ------------------------------------------------------------------
    ex = {}
    book = g["book"].to_numpy()
    for name, mask in {"book_ltp_0.98_1.03": (book >= 0.98) & (book <= 1.03), "book_ltp_0.95_1.05": (book >= 0.95) & (book <= 1.05),
                       "book_ltp_below_0.95": book < 0.95, "book_ltp_above_1.05": book > 1.05,
                       "runner_traded_t30": g["has_t30"].to_numpy(), "runner_not_traded_t30": ~g["has_t30"].to_numpy(),
                       "field_lt_9": (g["field_size"] < 9).to_numpy(), "field_ge_9": (g["field_size"] >= 9).to_numpy()}.items():
        s = g[mask]
        if len(s) == 0:
            continue
        ds = paired(s, C_TEST)
        b = basic(ds, s["won_b"].to_numpy())
        b["se_cl_date"] = r4(cluster_se(ds, s["date"].to_numpy()))
        b["share_of_band"] = r4(len(s) / len(g))
        first = (s["date"] < mid).to_numpy()
        b["halves"] = [r4(ds[first].mean()), r4(ds[~first].mean())]
        b["back_bsp_c5"] = r4(back_returns(s["bsp"], s["won_b"].to_numpy(), C_TEST).mean())
        b["back_ltp_c5"] = r4(back_returns(s["ltp_pre"], s["won_b"].to_numpy(), C_TEST).mean())
        ex[name] = b
    bsp_arr, lp_arr = g["bsp"].to_numpy(), g["ltp_pre"].to_numpy()
    worse = np.minimum(bsp_arr, lp_arr)
    r_worse = back_returns(worse, won, C_TEST)
    r_bsp = back_returns(bsp_arr, won, C_TEST)
    r_ltp = back_returns(lp_arr, won, C_TEST)
    r_tick = back_returns(tick_down(lp_arr), won, C_TEST)
    execution = {
        "sp_vs_ltp_tick_down": basic(r_bsp - r_tick),
        "sp_vs_limit_at_ltp_with_sp_fallback(=worse of the two)": basic(r_bsp - r_worse),
        "ltp_vs_worse_of_two": basic(r_ltp - r_worse),
        "share_sp_better_1tick_all": r4((bsp_arr >= tick_up(lp_arr) - 1e-9).mean()),
        "share_sp_worse_1tick_all": r4((bsp_arr <= tick_down(lp_arr) + 1e-9).mean()),
        "share_sp_better_1tick_winners": r4((bsp_arr >= tick_up(lp_arr) - 1e-9)[won].mean()),
        "share_sp_worse_1tick_winners": r4((bsp_arr <= tick_down(lp_arr) + 1e-9)[won].mean()),
        "median_bsp_minus_ltp_winners": r4(np.median((bsp_arr - lp_arr)[won])),
        "book_ltp_quantiles_band": [r4(v) for v in np.quantile(book, [0.1, 0.5, 0.9])],
        "book_bsp_quantiles_band": [r4(v) for v in np.nanquantile(g["book_bsp"].to_numpy(), [0.1, 0.5, 0.9])]}
    out({"section": "7_execution_liquidity_c5", "subsets": ex, **execution,
         "note": "a backer with a live screen sees the best available back price at the off, which captures part of any last-minute "
                 "move that ltp_pre (<= 1 min stale snapshot) misses; the 'limit at ltp with SP fallback' row is the pessimistic "
                 "execution of trying to take the last price"})

    # ---- 8. multiple testing summary -----------------------------------------------------------------
    z_cl = fc["roi"] / fc["se_cl_date"] if fc["se_cl_date"] else float("nan")
    out({"section": "8_multiple_testing", "z_iid": fc["z"], "z_cl_date": r4(z_cl), "p_cl_date": one_sided_p(z_cl),
         "bonferroni_556": bool(one_sided_p(z_cl) < 0.05 / N_CELLS_ORIGINAL),
         "note": "the original BH family (122 paired cells) is a subset of the 556 cells the script touched; Bonferroni over all "
                 "556 is the harshest correction"})
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Adversarial verification (statistics + execution lens) of the "bsp_and_laying" claim:

    AU gallops WIN runners with ltp_pre in [4.5, 6.5): backing 1 unit at Betfair SP
    returns +0.36% of stake more than backing the same runner at the last traded
    price (5% commission on both; n 63,158; SE 0.07%; halves +0.38% / +0.35%).
    A RELATIVE execution finding: backing at SP itself loses -3.9%.

    python scripts/bf_verify_bsp_and_laying_1_ltp45.py [--end 2018-06-01] [--boot 400] [--splits 300]

Same filters as scripts/bf_bsp_and_laying.py (dedupe, settled WIN, exactly one
winner, every runner traded pre-off, gallops = market name without Pace/Trot).
Nothing is fitted.  Re-run / added:

  1. the cell at commission 0 / 5 / 6 / 8 / 10%, per-bet SE, halves at the
     median day (as the original), strike vs implied at both prices;
  2. cluster-robust SEs by market, meeting (event_id), race day and
     venue-day, a race-day cluster bootstrap, month-as-unit t-test and sign test;
  3. the split: calendar years, half-years, thirds, odd/even ISO weeks, and
     random day-level half splits (how often the both-halves > 2 SE rule holds);
  4. the band: 0.5-point sub-bands (the tick changes from 0.1 to 0.2 at 6.0),
     shifted / widened windows, inclusive upper edge, by field size;
  5. influence: share of the paired gain from winners whose SP was far above
     the last price; winsorised ratio; dropping the k best winners;
  6. mechanism: strike by bsp/ltp_pre move bucket vs implied at each price
     (is a late drift informative, or is SP simply the longer price?);
  7. execution: LTP-book subsets (stale-quote suspects), runner traded 30 min
     out, ltp_pre == ltp_t30 (no price change in 30 min: a stale-quote proxy),
     one tick worse than the last trade, worse-of-the-two execution;
  8. external replication of the same cell, never run by the original: NZ, GB
     and IE gallops WIN markets and AU harness, same filters;
  9. multiple testing: the clustered p-value against Bonferroni over the 556
     cells the original touched.

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

SPLIT = HERE.parent / "data" / "raw" / "betfair" / "split"
COMMISSIONS = (0.0, 0.05, 0.06, 0.08, 0.10)
C_TEST = 0.05
LO, HI = 4.5, 6.5
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1001.0, 10.0)]
FIELD_EDGES = [2, 6, 9, 12, 99]
FIELD_LABELS = ["2-5", "6-8", "9-11", "12+"]
N_CELLS_ORIGINAL = 556
N_CELLS_FAMILY = 122


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
    """bsp return minus ltp_pre return, 1 unit backed on each (same runner)."""
    won = sub["won_b"].to_numpy()
    return back_returns(sub["bsp"].to_numpy(), won, c) - back_returns(sub["ltp_pre"].to_numpy(), won, c)


def sub_cell(sub, mid=None, c=C_TEST):
    d = paired(sub, c)
    won = sub["won_b"].to_numpy()
    b = basic(d, won)
    b["se_cl_date"] = r4(cluster_se(d, sub["date"].to_numpy()))
    if mid is not None and len(sub):
        first = (sub["date"] < mid).to_numpy()
        b["halves"] = [r4(d[first].mean()) if first.any() else None, r4(d[~first].mean()) if (~first).any() else None]
        b["halves_se"] = [r4(mean_se(d[first])[1]), r4(mean_se(d[~first])[1])]
    return b


def load(path, end, gallops=True):
    cols = ["market_id", "event_id", "venue", "market_name", "date", "field_size", "selection_id", "ltp_pre", "ltp_t30",
            "bsp", "won", "market_type"]
    df = pd.read_parquet(path, columns=cols)
    n_raw = len(df)
    df = df.drop_duplicates(["market_id", "selection_id"])
    df = df[(df["market_type"] == "WIN") & df["won"].notna()]
    if end:
        df = df[df["date"] < pd.Timestamp(end)]
    one_winner = df.groupby("market_id")["won"].transform("sum") == 1
    ok = df["ltp_pre"].notna() & (df["ltp_pre"] > 1)
    all_priced = ok.groupby(df["market_id"]).transform("min").astype(bool)
    df = df[one_winner & all_priced].drop(columns=["market_type"]).copy()
    harness = df["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b")
    df = df[~harness].copy() if gallops else df[harness].copy()
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
    df["venue_day"] = df["venue"].fillna("?") + "|" + df["date"].astype(str)
    days = sorted(df["date"].unique())
    mid = pd.Timestamp(days[len(days) // 2]) if days else pd.NaT
    info = {"rows_raw": int(n_raw), "markets": int(df["market_id"].nunique()), "runners": int(len(df)),
            "from": str(pd.Timestamp(days[0]).date()) if days else None, "to": str(pd.Timestamp(days[-1]).date()) if days else None,
            "mid_date_for_halves": str(mid.date()) if days else None, "bsp_runner_coverage": r4(df["bsp_ok"].mean()) if len(df) else None}
    return df, info, mid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--splits", type=int, default=300)
    a = ap.parse_args()
    t0 = time.time()
    df, info, mid = load(SPLIT / "au_win.parquet", a.end, gallops=True)
    out({"section": "data", "code": "AU gallops", **info, "end": a.end})

    lp = df["ltp_pre"].to_numpy()
    band = (lp >= LO) & (lp < HI) & df["bsp_ok"].to_numpy()
    g = df[band].copy()
    won = g["won_b"].to_numpy()
    ratio = g["bsp"].to_numpy() / g["ltp_pre"].to_numpy()
    g["ratio"] = ratio
    lp_g, bsp_g = g["ltp_pre"].to_numpy(), g["bsp"].to_numpy()

    # ---- 1. reproduction -----------------------------------------------------------------------
    rep = {f"paired_c{int(round(c * 100))}": basic(paired(g, c)) for c in COMMISSIONS}
    rep["back_ltp_pre"] = {f"c{int(round(c * 100))}": r4(back_returns(lp_g, won, c).mean()) for c in COMMISSIONS}
    rep["back_bsp"] = {f"c{int(round(c * 100))}": r4(back_returns(bsp_g, won, c).mean()) for c in COMMISSIONS}
    rep["back_ltp_tick_down"] = {f"c{int(round(c * 100))}": r4(back_returns(tick_down(lp_g), won, c).mean()) for c in COMMISSIONS}
    first = (g["date"] < mid).to_numpy()
    d5 = paired(g, C_TEST)
    rep["halves_c5"] = [basic(d5[first], won[first]), basic(d5[~first], won[~first])]
    rep["strike"] = r4(won.mean())
    rep["implied_ltp"] = r4((1.0 / lp_g).mean())
    rep["implied_bsp"] = r4((1.0 / bsp_g).mean())
    rep["mean_bsp_minus_ltp_all"] = r4((bsp_g - lp_g).mean())
    rep["mean_bsp_minus_ltp_given_won"] = r4((bsp_g - lp_g)[won].mean())
    rep["mean_bsp_minus_ltp_given_lost"] = r4((bsp_g - lp_g)[~won].mean())
    rep["median_ratio_all"] = r4(np.median(ratio))
    rep["mean_ratio_all"] = r4(ratio.mean())
    rep["mean_log_ratio_won"] = r4(np.log(ratio[won]).mean())
    rep["mean_log_ratio_lost"] = r4(np.log(ratio[~won]).mean())
    rep["gain_if_won_independent_of_move_c5"] = r4(won.mean() * (bsp_g - lp_g).mean() * (1 - C_TEST))
    rep["share_sp_better_1tick"] = r4((bsp_g >= tick_up(lp_g) - 1e-9).mean())
    rep["share_sp_worse_1tick"] = r4((bsp_g <= tick_down(lp_g) + 1e-9).mean())
    rep["share_sp_better_1tick_winners"] = r4((bsp_g >= tick_up(lp_g) - 1e-9)[won].mean())
    rep["share_sp_worse_1tick_winners"] = r4((bsp_g <= tick_down(lp_g) + 1e-9)[won].mean())
    out({"section": "1_reproduction_gallops_ltp_4.5_6.5", **rep,
         "note": "paired = back 1 at bsp minus back 1 at ltp_pre on the same runner; nonzero only on winners"})

    # ---- 2. clustered SEs, bootstrap, month-as-unit ------------------------------------------------
    cl = {"se_iid": r4(mean_se(d5)[1]),
          "se_cl_market": r4(cluster_se(d5, g["market_id"].to_numpy())),
          "se_cl_event": r4(cluster_se(d5, g["event_id"].to_numpy())),
          "se_cl_venue_day": r4(cluster_se(d5, g["venue_day"].to_numpy())),
          "se_cl_date": r4(cluster_se(d5, g["date"].to_numpy()))}
    m5 = float(d5.mean())
    cl["z_cl_date"] = r4(m5 / cl["se_cl_date"])
    halves = []
    for mask in (first, ~first):
        h = basic(d5[mask], won[mask])
        h["se_cl_date"] = r4(cluster_se(d5[mask], g["date"].to_numpy()[mask]))
        h["z_cl_date"] = r4(h["roi"] / h["se_cl_date"]) if h["se_cl_date"] else None
        halves.append(h)
    cl["halves"] = halves
    cl["both_halves_gt_2se_iid"] = bool(all(h["roi"] > 2 * h["se"] for h in halves))
    cl["both_halves_gt_2se_cl_date"] = bool(all(h["roi"] > 2 * h["se_cl_date"] for h in halves))
    cl["day_cluster_bootstrap"] = cluster_boot(d5, g["date"].to_numpy(), a.boot)
    mon = pd.DataFrame({"d": d5, "m": g["month"].to_numpy()}).groupby("m")["d"].agg(["mean", "count"])
    mm, ms = mean_se(mon["mean"].to_numpy())
    k_pos = int((mon["mean"] > 0).sum())
    n_m = int(len(mon))
    # exact binomial one-sided p for the sign test
    p_sign = sum(math.comb(n_m, k) for k in range(k_pos, n_m + 1)) / 2 ** n_m
    cl["months_as_units"] = {"months": n_m, "unweighted_mean": r4(mm), "se_across_months": r4(ms), "t": r4(mm / ms) if ms else None,
                             "months_positive": k_pos, "share_months_positive": r4(k_pos / n_m), "sign_test_p_one_sided": r4(p_sign)}
    out({"section": "2_clustered_se_c5", "roi": r4(m5), **cl,
         "p_one_sided_cl_date": one_sided_p(m5 / cl["se_cl_date"]),
         "note": "the paired difference is nonzero only on winners and a market has one winner, so market clusters hold "
                 "at most one nonzero term each; day / venue-day clusters are the relevant ones"})

    # ---- 3. alternative splits -----------------------------------------------------------------
    by_year = {str(y): sub_cell(gy) for y, gy in g.groupby("year")}
    by_hy = {hy: sub_cell(gy) for hy, gy in g.groupby("half_year")}
    days_sorted = np.sort(g["date"].unique())
    t1, t2 = days_sorted[len(days_sorted) // 3], days_sorted[2 * len(days_sorted) // 3]
    thirds = {"T1": sub_cell(g[g["date"] < t1]), "T2": sub_cell(g[(g["date"] >= t1) & (g["date"] < t2)]), "T3": sub_cell(g[g["date"] >= t2])}
    odd = (g["iso_week"] % 2 == 1).to_numpy()
    by_week = {"odd_weeks": basic(d5[odd], won[odd]), "even_weeks": basic(d5[~odd], won[~odd])}
    rng = np.random.default_rng(42)
    day_sum = pd.DataFrame({"d": d5, "day": g["date"].to_numpy()}).groupby("day")["d"].agg(["sum", "count", lambda x: (x ** 2).sum()])
    day_sum.columns = ["s", "n", "ss"]
    S, N, SS = day_sum["s"].to_numpy(), day_sum["n"].to_numpy(), day_sum["ss"].to_numpy()
    both = 0
    min_half_z = []
    for _ in range(a.splits):
        pick = rng.permutation(len(day_sum)) < len(day_sum) // 2
        zs = []
        for mask in (pick, ~pick):
            s, n, ss = S[mask].sum(), N[mask].sum(), SS[mask].sum()
            m = s / n
            var = (ss - n * m * m) / (n - 1)
            zs.append(m / math.sqrt(var / n))
        min_half_z.append(min(zs))
        both += all(z > 2 for z in zs)
    hy_keys = sorted(by_hy)
    t = np.arange(len(hy_keys), dtype=float)
    y = np.array([by_hy[k]["roi"] for k in hy_keys])
    w = np.array([1.0 / by_hy[k]["se_cl_date"] ** 2 for k in hy_keys])
    tm, ym = (w * t).sum() / w.sum(), (w * y).sum() / w.sum()
    slope = (w * (t - tm) * (y - ym)).sum() / (w * (t - tm) ** 2).sum()
    slope_se = math.sqrt(1.0 / (w * (t - tm) ** 2).sum())
    last2 = g[g["year"] >= 2019]
    trend = {"half_year_wls_slope_per_half_year": r4(slope), "slope_se": r4(slope_se), "slope_z": r4(slope / slope_se),
             "pooled_2019_2020": sub_cell(last2), "pooled_2016_2018": sub_cell(g[g["year"] <= 2018])}
    out({"section": "3_alternative_splits_c5", "by_year": by_year, "by_half_year": by_hy, "thirds": thirds,
         "odd_even_iso_weeks": by_week, "trend": trend,
         "random_day_half_splits": {"splits": a.splits, "share_both_halves_gt_2se": r4(both / a.splits),
                                    "min_half_z_q10_q50_q90": [r4(v) for v in np.quantile(min_half_z, [0.1, 0.5, 0.9])]}})

    # ---- 4. band sensitivity --------------------------------------------------------------------
    allg = df[df["bsp_ok"]].copy()
    lp_all = allg["ltp_pre"].to_numpy()
    windows = {"[4.5,5)": (4.5, 5), "[5,5.5)": (5, 5.5), "[5.5,6)": (5.5, 6), "[6,6.5)": (6, 6.5),
               "[4.5,6)_tick0.1": (4.5, 6), "[6,6.5)_tick0.2": (6, 6.5),
               "[4,6)": (4, 6), "[5,7)": (5, 7), "[3.5,5.5)": (3.5, 5.5), "[5.5,7.5)": (5.5, 7.5), "[4.5,6.5]": (4.5, 6.5001),
               "[3,4.5)": (3, 4.5), "[6.5,10)": (6.5, 10), "[3,10)": (3, 10), "[4.5,15)": (4.5, 15)}
    bands = {name: sub_cell(allg[(lp_all >= lo) & (lp_all < hi)], mid) for name, (lo, hi) in windows.items()}
    by_field = {}
    for f, gf in g.groupby("fgroup"):
        b = sub_cell(gf, mid)
        b["median_field"] = int(gf["field_size"].median())
        by_field[f] = b
    # exact tick points: is the gain concentrated at particular last prices?
    by_tick = {}
    for p, gp in g.groupby(np.round(lp_g, 2)):
        if len(gp) >= 500:
            by_tick[f"{p:g}"] = [int(len(gp)), int(gp["won_b"].sum()), r4(paired(gp, C_TEST).mean()), r4(mean_se(paired(gp, C_TEST))[1])]
    out({"section": "4_band_sensitivity_c5", "windows": bands, "ltp_4.5_6.5_by_field_size": by_field,
         "by_exact_last_price [n, n_won, paired_c5, se]": by_tick})

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
    wins = {}
    for cap in (1.1, 1.25, 1.5, 2.0):
        bsp_w = np.clip(bsp_g, lp_g / cap, lp_g * cap)
        dw = back_returns(bsp_w, won, C_TEST) - back_returns(lp_g, won, C_TEST)
        wins[f"cap_{cap:g}"] = basic(dw)
    order = np.argsort(-d5)
    drop_k = {str(k): r4(np.delete(d5, order[:k]).mean()) for k in (10, 50, 200, 500, 1000)}
    out({"section": "5_influence_c5", "total_paired_gain_units": r4(tot), "n_won": int(won.sum()), **infl,
         "winsorised_ratio": wins, "roi_after_dropping_k_best_winners": drop_k,
         "ratio_quantiles_winners": [r4(v) for v in np.quantile(ratio[won], [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])],
         "ratio_quantiles_losers": [r4(v) for v in np.quantile(ratio[~won], [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])]})

    # ---- 6. mechanism: is the SP move informative? ---------------------------------------------------
    cuts = [0, 0.85, 0.95, 0.99, 1.01, 1.05, 1.15, 1.3, 99]
    labels = ["<0.85", "0.85-0.95", "0.95-0.99", "0.99-1.01", "1.01-1.05", "1.05-1.15", "1.15-1.3", ">=1.3"]
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
        ps = paired(s, C_TEST)
        mech[lab] = {"n": n, "share": r4(n / len(g)), "strike": r4(strike), "strike_se": r4(math.sqrt(strike * (1 - strike) / n)),
                     "implied_ltp": r4(il), "implied_bsp": r4(ib), "paired_c5": r4(ps.mean()), "contribution_to_roi": r4(ps.sum() / len(g)),
                     "back_ltp_c0": r4(back_returns(s["ltp_pre"], w, 0.0).mean()), "back_bsp_c0": r4(back_returns(s["bsp"], w, 0.0).mean())}
    pl, pb = 1.0 / lp_g, 1.0 / bsp_g
    ll_l = -(won * np.log(pl) + (~won) * np.log(1 - pl)).mean()
    ll_b = -(won * np.log(pb) + (~won) * np.log(1 - pb)).mean()
    out({"section": "6_mechanism_strike_by_bsp_over_ltp_bucket", "buckets": mech,
         "runner_logloss_at_1_over_ltp": r4(ll_l), "runner_logloss_at_1_over_bsp": r4(ll_b),
         "note": "if strike tracks implied_bsp inside a bucket the SP move carried information; if it tracks implied_ltp the move "
                 "was noise and SP was simply the better (or worse) price"})

    # ---- 7. execution / liquidity / stale quotes ----------------------------------------------------
    ex = {}
    book = g["book"].to_numpy()
    has = g["has_t30"].to_numpy()
    same = has & (np.abs(g["ltp_t30"].to_numpy() - lp_g) < 1e-9)
    subsets = {"book_ltp_0.98_1.03": (book >= 0.98) & (book <= 1.03), "book_ltp_0.95_1.05": (book >= 0.95) & (book <= 1.05),
               "book_ltp_below_0.98": book < 0.98, "book_ltp_below_0.95": book < 0.95, "book_ltp_above_1.03": book > 1.03,
               "runner_traded_t30": has, "runner_not_traded_t30": ~has,
               "t30_and_ltp_pre_equals_ltp_t30 (no price change in 30 min: stale suspect)": same,
               "t30_and_price_changed_since_t30": has & ~same,
               "field_lt_9": (g["field_size"] < 9).to_numpy(), "field_ge_9": (g["field_size"] >= 9).to_numpy(),
               "field_ge_12": (g["field_size"] >= 12).to_numpy()}
    for name, mask in subsets.items():
        s = g[mask]
        if len(s) == 0:
            continue
        b = sub_cell(s, mid)
        b["share_of_band"] = r4(len(s) / len(g))
        b["back_bsp_c5"] = r4(back_returns(s["bsp"], s["won_b"].to_numpy(), C_TEST).mean())
        b["back_ltp_c5"] = r4(back_returns(s["ltp_pre"], s["won_b"].to_numpy(), C_TEST).mean())
        b["strike"] = r4(s["won_b"].mean())
        b["implied_ltp"] = r4((1.0 / s["ltp_pre"]).mean())
        b["implied_bsp"] = r4((1.0 / s["bsp"]).mean())
        ex[name] = b
    worse = np.minimum(bsp_g, lp_g)
    r_worse = back_returns(worse, won, C_TEST)
    r_bsp = back_returns(bsp_g, won, C_TEST)
    r_ltp = back_returns(lp_g, won, C_TEST)
    r_tick = back_returns(tick_down(lp_g), won, C_TEST)
    r_tick_up = back_returns(tick_up(lp_g), won, C_TEST)
    execution = {
        "sp_vs_ltp_tick_down": basic(r_bsp - r_tick),
        "sp_vs_ltp_tick_up (if the backer could always get one tick better than the last trade)": basic(r_bsp - r_tick_up),
        "sp_vs_limit_at_ltp_with_sp_fallback(=worse of the two)": basic(r_bsp - r_worse),
        "ltp_vs_worse_of_two": basic(r_ltp - r_worse),
        "paired_gain_in_ticks_per_winner": r4(((bsp_g - lp_g)[won] / _increment(lp_g[won])).mean()),
        "median_bsp_minus_ltp_winners": r4(np.median((bsp_g - lp_g)[won])),
        "book_ltp_quantiles_band": [r4(v) for v in np.quantile(book, [0.05, 0.1, 0.5, 0.9, 0.95])],
        "book_bsp_quantiles_band": [r4(v) for v in np.nanquantile(g["book_bsp"].to_numpy(), [0.05, 0.1, 0.5, 0.9, 0.95])],
        "share_bsp_off_tick_ladder": r4((np.abs(bsp_g - np.round(bsp_g / _increment(bsp_g - 1e-9)) * _increment(bsp_g - 1e-9)) > 1e-6).mean())}
    out({"section": "7_execution_liquidity_c5", "subsets": ex, **execution,
         "note": "ltp_pre is a <= 1 min stale snapshot of the last trade, not an available price; the tick-down row is the "
                 "pessimistic execution of taking the last price, the tick-up row the optimistic one"})

    # ---- 8. external replication (same cell, same filters) ------------------------------------------
    repl = {}
    for code, path, gallops in (("AU harness", SPLIT / "au_win.parquet", False), ("NZ gallops", SPLIT / "nz_win.parquet", True),
                                ("GB gallops", SPLIT / "gb_win.parquet", True), ("IE gallops", SPLIT / "ie_win.parquet", True)):
        if not path.exists():
            continue
        d2, i2, mid2 = load(path, a.end, gallops=gallops)
        lp2 = d2["ltp_pre"].to_numpy()
        s = d2[(lp2 >= LO) & (lp2 < HI) & d2["bsp_ok"].to_numpy()]
        if len(s) < 100:
            repl[code] = {"bets": int(len(s))}
            continue
        w2 = s["won_b"].to_numpy()
        b = sub_cell(s, mid2)
        b["z_cl_date"] = r4(b["roi"] / b["se_cl_date"]) if b["se_cl_date"] else None
        b["markets_total"] = i2["markets"]
        b["bsp_coverage"] = i2["bsp_runner_coverage"]
        b["strike"] = r4(w2.mean())
        b["implied_ltp"] = r4((1.0 / s["ltp_pre"]).mean())
        b["implied_bsp"] = r4((1.0 / s["bsp"]).mean())
        b["back_ltp_c5"] = r4(back_returns(s["ltp_pre"], w2, C_TEST).mean())
        b["back_bsp_c5"] = r4(back_returns(s["bsp"], w2, C_TEST).mean())
        b["paired_c0"] = r4(paired(s, 0.0).mean())
        b["share_sp_better_1tick"] = r4((s["bsp"].to_numpy() >= tick_up(s["ltp_pre"].to_numpy()) - 1e-9).mean())
        b["share_sp_worse_1tick"] = r4((s["bsp"].to_numpy() <= tick_down(s["ltp_pre"].to_numpy()) + 1e-9).mean())
        b["book_ltp_median"] = r4(s["book"].median())
        b["field_ge_12"] = sub_cell(s[s["field_size"] >= 12])
        repl[code] = b
        del d2
    out({"section": "8_external_replication_ltp_4.5_6.5_c5", "cells": repl,
         "note": "same band, same filters, other jurisdictions; the original ran AU only"})

    # ---- 9. multiple testing summary -----------------------------------------------------------------
    z_cl = m5 / cl["se_cl_date"]
    p_cl = one_sided_p(z_cl)
    out({"section": "9_multiple_testing", "z_iid": r4(m5 / cl["se_iid"]), "z_cl_date": r4(z_cl), "p_cl_date": p_cl,
         "bonferroni_122_family": bool(p_cl < 0.05 / N_CELLS_FAMILY), "bonferroni_556_all_cells": bool(p_cl < 0.05 / N_CELLS_ORIGINAL),
         "bonferroni_threshold_556": r4(0.05 / N_CELLS_ORIGINAL)})
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

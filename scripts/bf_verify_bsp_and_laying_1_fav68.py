"""Adversarial verification (statistics + execution lens) of the "bsp_and_laying" claim:

    AU gallops WIN, favourite (rank 1 at ltp_pre) in a 6-8 runner field: backing 1 unit
    at the last traded price returns more than backing the same runner at Betfair SP:
    paired SP-minus-LTP = -0.84% +/- 0.10% after 5% commission (n 17,979; z -8.5;
    halves -0.82% / -0.85%).  Relative, conditional on being matched at the last price;
    one tick worse than the last price already flips it.

    python scripts/bf_verify_bsp_and_laying_1_fav68.py [--end 2018-06-01] [--boot 400] [--splits 200]

(The harness asked for scripts/bf_verify_bsp_and_laying_1.py; that file already exists
for the $10-15 cell, so this one carries a suffix.  Nothing existing is edited.)

Same data filters as scripts/bf_bsp_and_laying.py (dedupe, settled WIN, exactly one
winner, every runner traded pre-off, gallops = market name without Pace/Trot), re-derived
here with event_id / venue kept for clustering, and checked against the original's counts.
Nothing is fitted.  Sections (JSON lines, summary numbers only, never horse names):

  1. the cell: paired bsp-return minus ltp_pre-return at commission 0/5/6/8/10%, per-bet
     SE, halves at the median day (as the original), z;
  2. cluster-robust SEs by market (1 favourite per market, so identical), by meeting
     (event_id), by race day, by venue, a race-day cluster bootstrap and a month-as-unit
     t-test;
  3. splits: calendar years, half-years, odd/even ISO weeks, exact field size 6/7/8, random
     day-level half splits (how often the both-halves > 2 SE rule holds in this direction);
  4. mechanism: bsp/ltp_pre on all favourites vs winners vs losers (is SP shorter for every
     6-8 favourite, or only for the winners?); strike vs implied at ltp and at bsp; the
     timing component (ratio independent of winning) vs the informational remainder;
  5. specificity: favourites by field group and by price sub-band, non-favourites in 6-8
     fields at the same prices, second favourites, every 6-8 runner under $3.5;
  6. execution: one and two ticks worse than the last price, the mixture break-even share
     matched at the last price, the "limit at last price, SP fallback" execution (filled only
     when the price drifted), and the t30 -> pre -> SP path (conditional on a t30 trade);
  7. influence: winsorised ratio, |log ratio| > 0.3 excluded, share of the paired total from
     the 1% most extreme winners;
  8. the sibling cells the claim also names ($2-3 all runners, favourites 2-5, $50+ all
     runners): cluster-robust SEs by market and by day, trimmed versions;
  9. multiple testing over the original's 122 paired cells, two-sided (Bonferroni and BH),
     from its cells CSV when present.
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
sys.path.insert(0, str(HERE))
import bf_bsp_and_laying as bspl  # noqa: E402  the original's helpers (tick ladder, bands, load logic)

COMMS = (0.0, 0.05, 0.06, 0.08, 0.10)
C = 0.05
SCRATCH = Path("/tmp/claude-0/-home-user-PredictivePonies/43d94981-d22b-59d1-a1cc-00a2ff7cb49d/scratchpad")


def out(obj) -> None:
    print(json.dumps(obj, default=bspl._json_default), flush=True)


def r4(x):
    return bspl.r4(x)


# ----------------------------------------------------------------------------- data (same filters as the original)


def load(path: Path, end: str | None) -> tuple[pd.DataFrame, pd.Timestamp, dict]:
    cols = ["market_id", "event_id", "venue", "market_name", "date", "field_size", "selection_id", "ltp_pre",
            "ltp_t30", "bsp", "won", "market_type"]
    df = pd.read_parquet(path, columns=cols)
    df = df.drop_duplicates(["market_id", "selection_id"])
    df = df[(df["market_type"] == "WIN") & df["won"].notna()]
    if end:
        df = df[df["date"] < pd.Timestamp(end)]
    one_winner = df.groupby("market_id")["won"].transform("sum") == 1
    ok = df["ltp_pre"].notna() & (df["ltp_pre"] > 1)
    all_priced = ok.groupby(df["market_id"]).transform("min").astype(bool)
    df = df[one_winner & all_priced].drop(columns=["market_type"]).copy()
    df = df.sort_values(["date", "market_id", "ltp_pre", "selection_id"]).reset_index(drop=True)
    df["rank"] = df.groupby("market_id").cumcount() + 1
    df["fav"] = df["rank"] == 1
    df["won_b"] = df["won"].to_numpy() == 1
    df["harness"] = df["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b")
    df["band"] = pd.cut(df["ltp_pre"], bspl.PRICE_BANDS, right=False, labels=bspl.BAND_LABELS).astype(str)
    df["fgroup"] = pd.cut(df["field_size"], bspl.FIELD_EDGES, right=False, labels=bspl.FIELD_LABELS).astype(str)
    df["month"] = df["date"].dt.to_period("M")
    df["bsp_ok"] = df["bsp"].notna() & (df["bsp"] > 1)
    df["has_t30"] = df["ltp_t30"].notna() & (df["ltp_t30"] > 1)
    days = sorted(df["date"].unique())
    mid = pd.Timestamp(days[len(days) // 2])
    df["first"] = df["date"] < mid
    info = {"markets_used": int(df["market_id"].nunique()), "runners_used": int(len(df)),
            "markets_gallops": int(df.loc[~df["harness"], "market_id"].nunique()),
            "mid_date_for_halves": str(mid.date()), "from": str(pd.Timestamp(days[0]).date()),
            "to": str(pd.Timestamp(days[-1]).date())}
    return df, mid, info


# ----------------------------------------------------------------------------- statistics


def back(price, won, c):
    return bspl.back_returns(price, won, c)


def mean_se(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else float("nan")), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(n))


def cluster_se(x: np.ndarray, groups) -> tuple[float, int]:
    """SE of the mean with arbitrary within-cluster correlation (Liang-Zeger with one regressor)."""
    n = len(x)
    if n < 2:
        return float("nan"), n
    e = x - x.mean()
    s = pd.Series(e).groupby(np.asarray(groups)).sum()
    g = len(s)
    se = math.sqrt(float((s.to_numpy() ** 2).sum())) / n * math.sqrt(g / max(g - 1, 1))
    return se, g


def cluster_boot(x: np.ndarray, groups, reps: int, seed: int = 0) -> float:
    """SE of the mean from resampling clusters (days) with replacement."""
    rng = np.random.default_rng(seed)
    s = pd.DataFrame({"x": x, "g": np.asarray(groups)}).groupby("g")["x"].agg(["sum", "count"])
    sums, cnts = s["sum"].to_numpy(), s["count"].to_numpy()
    g = len(sums)
    means = []
    for _ in range(reps):
        idx = rng.integers(0, g, g)
        means.append(sums[idx].sum() / cnts[idx].sum())
    return float(np.std(means, ddof=1))


def paired_stats(sub: pd.DataFrame, mid: pd.Timestamp | None = None, label: str = "", full: bool = True) -> dict:
    """The paired SP-minus-LTP cell on ``sub`` (rows with bsp), in the original's layout plus clustering."""
    sub = sub[sub["bsp_ok"]]
    n = len(sub)
    res = {"cell": label, "bets": int(n)}
    if n == 0:
        return res
    won = sub["won_b"].to_numpy()
    lp = sub["ltp_pre"].to_numpy(dtype=float)
    sp = sub["bsp"].to_numpy(dtype=float)
    res["n_won"] = int(won.sum())
    res["markets"] = int(sub["market_id"].nunique())
    res["strike"] = r4(won.mean())
    res["implied_ltp"] = r4((1 / lp).mean())
    res["implied_bsp"] = r4((1 / sp).mean())
    res["median_ltp"] = r4(np.median(lp))
    res["median_bsp"] = r4(np.median(sp))
    d = {c: back(sp, won, c) - back(lp, won, c) for c in COMMS}
    res["diff_by_commission"] = {f"c{int(c * 100)}": r4(d[c].mean()) for c in COMMS}
    m, se = mean_se(d[C])
    res["diff_c5"], res["se_iid"] = r4(m), r4(se)
    res["z_iid"] = r4(m / se) if se and se > 0 else None
    for c in (0.0, 0.05, 0.08):
        res[f"back_ltp_c{int(c * 100)}"] = r4(back(lp, won, c).mean())
        res[f"back_bsp_c{int(c * 100)}"] = r4(back(sp, won, c).mean())
    res["back_ltp_tick_down_c5"] = r4(back(bspl.tick_down(lp), won, C).mean())
    if full:
        for key, col in (("market", "market_id"), ("meeting", "event_id"), ("day", "date"), ("venue", "venue"), ("month", "month")):
            cse, g = cluster_se(d[C], sub[col].to_numpy())
            res[f"se_cluster_{key}"] = r4(cse)
            res[f"n_clusters_{key}"] = int(g)
            res[f"z_cluster_{key}"] = r4(m / cse) if cse and cse > 0 else None
    if mid is not None:
        first = sub["date"].to_numpy() < np.datetime64(mid)
        halves = []
        for mk in (first, ~first):
            hm, hs = mean_se(d[C][mk])
            halves.append([int(mk.sum()), int(won[mk].sum()), r4(hm), r4(hs), r4(hm / hs) if hs and hs > 0 else None])
        res["halves_[n,won,diff_c5,se,z]"] = halves
        res["both_halves_below_minus_2se"] = bool(all(h[2] is not None and h[3] and h[2] < -2 * h[3] for h in halves))
    return res


def ratio_stats(sub: pd.DataFrame) -> dict:
    sub = sub[sub["bsp_ok"]]
    lp = sub["ltp_pre"].to_numpy(dtype=float)
    sp = sub["bsp"].to_numpy(dtype=float)
    won = sub["won_b"].to_numpy()
    ratio = sp / lp
    lr = np.log(ratio)
    better = sp >= bspl.tick_up(lp) - 1e-9
    worse = sp <= bspl.tick_down(lp) + 1e-9

    def part(mask):
        if mask.sum() == 0:
            return None
        return {"n": int(mask.sum()), "mean_ratio_minus_1": r4((ratio[mask] - 1).mean()),
                "mean_log_ratio": r4(lr[mask].mean()), "median_ratio": r4(np.median(ratio[mask])),
                "q10_q90_ratio": [r4(v) for v in np.quantile(ratio[mask], [0.1, 0.9])],
                "share_sp_better_1tick": r4(better[mask].mean()), "share_sp_worse_1tick": r4(worse[mask].mean()),
                "share_sp_equal": r4((~better & ~worse)[mask].mean()),
                "mean_bsp_minus_ltp": r4((sp - lp)[mask].mean())}

    return {"all": part(np.ones(len(sub), dtype=bool)), "winners": part(won), "losers": part(~won)}


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=HERE.parent / "data/raw/betfair/split/au_win.parquet")
    ap.add_argument("--end", default=None)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--cells-csv", type=Path, default=SCRATCH / "bspl_full_cells.csv")
    a = ap.parse_args()
    t0 = time.time()
    df, mid, info = load(a.path, a.end)
    g = df[~df["harness"]].reset_index(drop=True)
    out({"section": "data", **info, "end": a.end, "gallops_runners": int(len(g)),
         "gallops_bsp_coverage": r4(g["bsp_ok"].mean()), "gallops_t30_coverage": r4(g["has_t30"].mean()),
         "original_counts_expected": {"markets_gallops": 63506, "runners_gallops_paired_all": 618780, "fav_6_8": 17979}})

    fav = g[g["fav"]]
    cell = fav[fav["fgroup"] == "6-8"]
    one_per_market = int(cell["market_id"].nunique()) == len(cell)

    # 1 + 2: the cell, with clustering ------------------------------------------------------------------
    rep = paired_stats(cell, mid, "gallops favourite field 6-8")
    rep["one_favourite_per_market"] = one_per_market
    d5 = (back(cell["bsp"].to_numpy(float), cell["won_b"].to_numpy(), C)
          - back(cell["ltp_pre"].to_numpy(float), cell["won_b"].to_numpy(), C))
    rep["se_day_cluster_bootstrap"] = r4(cluster_boot(d5, cell["date"].to_numpy(), a.boot))
    # month-as-unit t-test (unweighted mean of monthly means)
    mm = pd.Series(d5).groupby(cell["month"].to_numpy()).agg(["mean", "count"])
    t_m = mm["mean"].mean() / (mm["mean"].std(ddof=1) / math.sqrt(len(mm)))
    rep["month_as_unit"] = {"months": int(len(mm)), "mean_of_monthly_means": r4(mm["mean"].mean()),
                            "t": r4(t_m), "share_months_negative": r4((mm["mean"] < 0).mean())}
    out({"section": "1_2_reproduce_and_cluster", **rep,
         "note": "diff = return at bsp minus return at ltp_pre, 1 unit back, commission on net winnings; negative = last price better"})

    # 3: splits ----------------------------------------------------------------------------------------------
    splits = {}
    cell_y = cell.assign(year=cell["date"].dt.year, half=cell["date"].dt.year.astype(str) + "H" + ((cell["date"].dt.month > 6) + 1).astype(str),
                         oddweek=cell["date"].dt.isocalendar().week.astype(int) % 2)
    for key in ("year", "half", "oddweek", "field_size"):
        splits[key] = {}
        for val, s in cell_y.groupby(key):
            r = paired_stats(s, None, str(val), full=False)
            splits[key][str(val)] = [r["bets"], r["n_won"], r["diff_c5"], r["se_iid"], r["z_iid"]]
    # random day-level half splits
    rng = np.random.default_rng(1)
    days = cell["date"].unique()
    di = pd.Index(days).get_indexer(cell["date"].to_numpy())
    both = 0
    zs = []
    for _ in range(a.splits):
        lab = rng.random(len(days)) < 0.5
        h = lab[di]
        ok = True
        for mk in (h, ~h):
            m, s = mean_se(d5[mk])
            zs.append(m / s)
            ok &= m < -2 * s
        both += ok
    splits["random_day_half_splits"] = {"splits": a.splits, "share_both_halves_below_minus_2se": r4(both / a.splits),
                                        "half_z_q05_q50_q95": [r4(v) for v in np.quantile(zs, [0.05, 0.5, 0.95])]}
    out({"section": "3_splits", "columns": "[bets, n_won, diff_c5, se_iid, z]", **splits})

    # 4: mechanism ---------------------------------------------------------------------------------------------
    rs = ratio_stats(cell)
    c_ = cell[cell["bsp_ok"]]
    won = c_["won_b"].to_numpy()
    lp = c_["ltp_pre"].to_numpy(float)
    sp = c_["bsp"].to_numpy(float)
    p_win = won.mean()
    timing = p_win * (sp - lp).mean() * (1 - C)  # the paired diff if the SP move were independent of winning
    actual = (won * (sp - lp) * (1 - C)).mean()
    strike_vs = {"strike": r4(p_win), "implied_ltp": r4((1 / lp).mean()), "implied_bsp": r4((1 / sp).mean()),
                 "back_c0_ltp": r4(back(lp, won, 0).mean()), "back_c0_bsp": r4(back(sp, won, 0).mean())}
    # by SP move bucket: strike vs implied at ltp and at bsp
    lr = np.log(sp / lp)
    cuts = [-np.inf, -0.1, -0.03, -1e-9, 1e-9, 0.03, 0.1, np.inf]
    labs = ["firm>10%", "firm 3-10%", "firm <3%", "unchanged", "drift <3%", "drift 3-10%", "drift>10%"]
    mv = pd.cut(lr, cuts, labels=labs).astype(str)
    by_move = {}
    for lab in labs:
        m = mv == lab
        if m.sum():
            by_move[lab] = {"n": int(m.sum()), "strike": r4(won[m].mean()), "implied_ltp": r4((1 / lp[m]).mean()),
                            "implied_bsp": r4((1 / sp[m]).mean()), "share": r4(m.mean())}
    out({"section": "4_mechanism", "ratio_bsp_over_ltp": rs, **strike_vs,
         "paired_diff_c5_actual": r4(actual),
         "timing_component_c5 (P(win) x mean(bsp-ltp) over ALL favourites x 0.95)": r4(timing),
         "informational_remainder_c5": r4(actual - timing),
         "by_sp_move_bucket": by_move})

    # 5: specificity ---------------------------------------------------------------------------------------------
    spec = {}
    for f, s in fav.groupby("fgroup"):
        r = paired_stats(s, mid, f, full=False)
        spec[f"favourite field {f}"] = [r["bets"], r["n_won"], r["median_ltp"], r["diff_c5"], r["se_iid"], r["z_iid"],
                                        r["halves_[n,won,diff_c5,se,z]"][0][2], r["halves_[n,won,diff_c5,se,z]"][1][2]]
    sub_bands = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 1e9]
    pb = pd.cut(cell["ltp_pre"], sub_bands, right=False).astype(str)
    for b, s in cell.groupby(pb):
        r = paired_stats(s, mid, b, full=False)
        spec[f"fav 6-8 ltp {b}"] = [r["bets"], r["n_won"], r["median_ltp"], r["diff_c5"], r["se_iid"], r["z_iid"],
                                    r["halves_[n,won,diff_c5,se,z]"][0][2], r["halves_[n,won,diff_c5,se,z]"][1][2]]
    g68 = g[g["fgroup"] == "6-8"]
    others = {"non-fav 6-8 ltp [2,3.5)": g68[(~g68["fav"]) & (g68["ltp_pre"] >= 2) & (g68["ltp_pre"] < 3.5)],
              "fav 6-8 ltp [2,3.5)": g68[(g68["fav"]) & (g68["ltp_pre"] >= 2) & (g68["ltp_pre"] < 3.5)],
              "rank 2 in 6-8": g68[g68["rank"] == 2],
              "rank 3+ in 6-8": g68[g68["rank"] >= 3],
              "all 6-8 ltp < 3.5": g68[g68["ltp_pre"] < 3.5],
              "all gallops ltp [2,3)": g[g["band"] == "2-3"],
              "fav any field ltp [2,3)": fav[fav["band"] == "2-3"],
              "non-fav any field ltp [2,3)": g[(~g["fav"]) & (g["band"] == "2-3")],
              "fav any field ltp [3,4.5)": fav[fav["band"] == "3-4.5"],
              "non-fav any field ltp [3,4.5)": g[(~g["fav"]) & (g["band"] == "3-4.5")]}
    for k, s in others.items():
        r = paired_stats(s, mid, k, full=True)
        spec[k] = [r["bets"], r["n_won"], r["median_ltp"], r["diff_c5"], r["se_iid"], r["z_iid"],
                   r["halves_[n,won,diff_c5,se,z]"][0][2], r["halves_[n,won,diff_c5,se,z]"][1][2],
                   {"se_cluster_market": r["se_cluster_market"], "se_cluster_day": r["se_cluster_day"]}]
    out({"section": "5_specificity", "columns": "[bets, n_won, median_ltp, diff_c5, se_iid, z, half1_c5, half2_c5, (cluster SEs)]", **spec})

    # 6: execution -----------------------------------------------------------------------------------------------
    ex = {}
    r_ltp = back(lp, won, C)
    r_t1 = back(bspl.tick_down(lp), won, C)
    r_t2 = back(bspl.tick_down(bspl.tick_down(lp)), won, C)
    r_sp = back(sp, won, C)
    ex["back_c5"] = {"at_ltp": r4(r_ltp.mean()), "one_tick_worse": r4(r_t1.mean()), "two_ticks_worse": r4(r_t2.mean()), "at_bsp": r4(r_sp.mean())}
    ex["diff_vs_bsp_c5"] = {"ltp": r4((r_sp - r_ltp).mean()), "one_tick_worse": r4((r_sp - r_t1).mean()), "two_ticks_worse": r4((r_sp - r_t2).mean())}
    g_ltp, g_t1, g_t2 = (r_ltp - r_sp).mean(), (r_t1 - r_sp).mean(), (r_t2 - r_sp).mean()
    ex["breakeven_share_matched_at_ltp"] = {"rest_one_tick_worse": r4(-g_t1 / (g_ltp - g_t1)) if g_t1 < 0 else 0.0,
                                            "rest_two_ticks_worse": r4(-g_t2 / (g_ltp - g_t2)) if g_t2 < 0 else 0.0}
    # limit order at the last price, SP fallback: matched only when the price did not firm (bsp >= ltp)
    filled = sp >= lp
    r_limit = np.where(filled, r_ltp, r_sp)
    ex["limit_at_ltp_sp_fallback"] = {"fill_share": r4(filled.mean()), "fill_share_winners": r4(filled[won].mean()),
                                      "return_c5": r4(r_limit.mean()), "vs_bsp_c5": r4((r_limit - r_sp).mean()),
                                      "se_iid": r4(mean_se(r_limit - r_sp)[1])}
    # the optimistic mirror: better of the two prices (never available to a real backer)
    ex["better_of_two_prices_c5 (not bettable)"] = r4(back(np.maximum(lp, sp), won, C).mean())
    # t30 -> pre -> SP path, conditional on a t30 trade
    t = c_[c_["has_t30"]]
    if len(t):
        w, l30, lpp, spp = t["won_b"].to_numpy(), t["ltp_t30"].to_numpy(float), t["ltp_pre"].to_numpy(float), t["bsp"].to_numpy(float)
        p30, ppre, psp = back(l30, w, C), back(lpp, w, C), back(spp, w, C)
        ex["t30_path (CONDITIONAL on t30 trade)"] = {
            "n": int(len(t)), "share_of_cell": r4(len(t) / len(c_)), "n_won": int(w.sum()),
            "mean_log(pre/t30)": {"all": r4(np.log(lpp / l30).mean()), "winners": r4(np.log(lpp / l30)[w].mean()), "losers": r4(np.log(lpp / l30)[~w].mean())},
            "mean_log(bsp/pre)": {"all": r4(np.log(spp / lpp).mean()), "winners": r4(np.log(spp / lpp)[w].mean()), "losers": r4(np.log(spp / lpp)[~w].mean())},
            "back_c5": {"at_t30": r4(p30.mean()), "at_pre": r4(ppre.mean()), "at_bsp": r4(psp.mean())},
            "paired_pre_minus_t30_c5": [r4(v) for v in mean_se(ppre - p30)],
            "paired_bsp_minus_pre_c5": [r4(v) for v in mean_se(psp - ppre)],
            "paired_bsp_minus_pre_c5_no_t30": [r4(v) for v in mean_se((r_sp - r_ltp)[~c_["has_t30"].to_numpy()])]}
    out({"section": "6_execution", **ex})

    # 7: influence -------------------------------------------------------------------------------------------------
    inf = {}
    ratio = sp / lp
    for lo, hi in ((0.8, 1.25), (0.9, 1.1)):
        spw = lp * np.clip(ratio, lo, hi)
        inf[f"winsorised ratio [{lo},{hi}]"] = [r4(v) for v in mean_se(back(spw, won, C) - r_ltp)]
    keep = np.abs(np.log(ratio)) <= 0.3
    inf["exclude |log ratio| > 0.3"] = {"n": int(keep.sum()), "diff_se": [r4(v) for v in mean_se((r_sp - r_ltp)[keep])]}
    dw = (r_sp - r_ltp)[won]
    order = np.argsort(dw)
    k = max(1, int(0.01 * len(dw)))
    inf["share_of_total_from_1pct_most_negative_winners"] = r4(dw[order[:k]].sum() / dw.sum()) if dw.sum() != 0 else None
    inf["share_of_total_from_1pct_most_positive_winners"] = r4(dw[order[-k:]].sum() / dw.sum()) if dw.sum() != 0 else None
    inf["diff_c5_excluding_1pct_each_tail_of_winners"] = r4(np.concatenate([dw[order[k:-k]], np.zeros(int((~won).sum()))]).mean())
    inf["ratio_extremes"] = {"min": r4(ratio.min()), "max": r4(ratio.max()), "share_ratio_lt_0.7": r4((ratio < 0.7).mean()), "share_ratio_gt_1.4": r4((ratio > 1.4).mean())}
    inf["sd_of_diff"] = r4(np.std(r_sp - r_ltp, ddof=1))
    out({"section": "7_influence", **inf})

    # 8: sibling cells --------------------------------------------------------------------------------------------
    sib = {}
    sibs = {"gallops ltp [2,3) all runners": g[g["band"] == "2-3"],
            "gallops favourite field 2-5": fav[fav["fgroup"] == "2-5"],
            "gallops ltp 50+ all runners": g[g["band"] == "50+"],
            "gallops ltp [1,2) all runners": g[g["band"] == "1-2"]}
    for k, s in sibs.items():
        r = paired_stats(s, mid, k, full=True)
        s_ = s[s["bsp_ok"]]
        w_, lp_, sp_ = s_["won_b"].to_numpy(), s_["ltp_pre"].to_numpy(float), s_["bsp"].to_numpy(float)
        dd = back(sp_, w_, C) - back(lp_, w_, C)
        dwin = dd[w_]
        o = np.argsort(dwin)
        kk = max(1, int(0.01 * len(dwin)))
        trimmed = np.concatenate([dwin[o[kk:-kk]], np.zeros(int((~w_).sum()))])
        r["trim_1pct_each_tail_of_winners_diff_c5"] = r4(trimmed.mean())
        r["share_total_from_10_most_negative_winners"] = r4(dwin[o[:10]].sum() / dd.sum()) if dd.sum() != 0 else None
        r["ratio_all_vs_winners"] = {k2: (v["mean_ratio_minus_1"] if v else None) for k2, v in ratio_stats(s).items()}
        sib[k] = r
    out({"section": "8_sibling_cells", **sib})

    # 9: multiple testing over the original's paired family -----------------------------------------------------
    mt = {}
    if a.cells_csv and a.cells_csv.exists():
        cdf = pd.read_csv(a.cells_csv)
        fam = cdf[cdf["family"] == "sp_minus_ltp"].copy()
        fam["z_use"] = pd.to_numeric(fam["z_cons"], errors="coerce").fillna(pd.to_numeric(fam["z"], errors="coerce"))
        fam = fam[fam["testable"].astype(str) == "True"]
        z = fam["z_use"].to_numpy(dtype=float)
        p2 = np.array([math.erfc(abs(v) / math.sqrt(2)) if not math.isnan(v) else np.nan for v in z])
        m = int(np.isfinite(p2).sum())
        bonf_z = -math.sqrt(2) * _erfcinv(0.05 / m) if m else None
        rej = bspl.benjamini_hochberg(p2, 0.05)
        neg = fam[rej & (z < 0)]
        pos = fam[rej & (z > 0)]
        mt = {"paired_cells_total": int(len(cdf[cdf["family"] == "sp_minus_ltp"])), "testable": m,
              "bonferroni_two_sided_z": r4(bonf_z), "bh_two_sided_rejections": int(rej.sum()),
              "bh_negative (last price better)": [[r["code"], r["name"], int(r["bets"]), r["roi_c5"], r["se"], r4(r["z_use"])] for _, r in neg.sort_values("z_use").iterrows()],
              "bh_positive_count (SP better)": int(len(pos)),
              "favourite_cells_scanned": int(fam["name"].str.startswith("favourite").sum()),
              "this_cell": [[r["code"], r["name"], int(r["bets"]), r["roi_c5"], r["se"], r4(r["z_use"]), bool(rj)] for (_, r), rj in zip(fam.iterrows(), rej) if r["name"] == "favourite field 6-8"]}
    out({"section": "9_multiple_testing", **mt})
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


def _erfcinv(y: float) -> float:
    """Inverse complementary error function by bisection (y in (0, 2))."""
    lo, hi = -6.0, 6.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erfc(mid) > y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    sys.exit(main())

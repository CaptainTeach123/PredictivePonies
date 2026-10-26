"""Adversarial verification of one claim from experiment "bsp_and_laying"
(scripts/bf_bsp_and_laying.py), lens = leakage and data quality.

CLAIM: AU gallops WIN, favourite (rank 1 at ltp_pre) in a 6-8 runner field: the
paired return at the last traded price beats the return at Betfair SP by
+0.84% of stake (SP-minus-LTP c5 = -0.0084 +/- 0.0010, z -8.5, halves
-0.0082 / -0.0085, n = 17979), conditional on being matched at ltp_pre.

    python scripts/bf_verify_bsp_and_laying_0.py [--end 2018-06-01]

The original script is imported (not edited) so the reproduction uses the same
load(), Rets and ratio_cell code. Everything printed is a summary number.

Checks (all on the gallops favourite 6-8 cell unless stated):
  0. exact reproduction of the cell (n, diff at c0/c5/c8, SE, halves, z)
  1. data quality of the universe: duplicate markets per event/start, rows vs
     field_size (removed runners), dead heats, bsp sanity, ltp_pre == bsp
  2. in-play leakage tail in ltp_pre: winners vs losers with bsp/ltp far from 1
  3. decomposition: systematic (all favourites) vs informational (winners only)
     part of bsp - ltp; sign test on winners
  4. robustness: trimmed / winsorised ratios, cluster-robust SE by date and by
     meeting (venue+date), by year, by field size 6/7/8, by price band
  5. staleness proxy: the market book at ltp_pre (sum 1/ltp_pre) - stale last
     trades make the book inconsistent; and the share of the field with a t30
     trade (liquidity)
  6. removed-runner proxy: a gap in saddlecloth numbers (max program >
     field_size) means a scratching happened at some point
  7. t30 comparisons (CONDITIONAL on the favourite having traded 30 min out):
     bsp vs ltp_t30, ltp_pre vs ltp_t30; and favourite defined at t30
  8. survivorship: recompute without the "all runners priced" market filter
  9. execution: ltp, one tick worse, half-tick, and the share of fills at ltp
     needed to break even with SP; harness 6-8 favourites for contrast
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
import bf_bsp_and_laying as orig  # noqa: E402  the script under test, imported unchanged

C = 0.05


def out(obj) -> None:
    print(json.dumps(obj, default=orig._json_default), flush=True)


def r4(x):
    return orig.r4(x)


def mean_se(x: np.ndarray) -> tuple[float, float]:
    return orig._mean_se(np.asarray(x, dtype=float))


def cluster_se(x: np.ndarray, groups: np.ndarray) -> float:
    """SE of the mean with arbitrary within-group correlation."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    s = pd.Series(x - m).groupby(pd.Series(groups).to_numpy()).sum().to_numpy()
    g = len(s)
    return math.sqrt((s ** 2).sum() * g / max(g - 1, 1)) / n


def paired_summary(bsp, lp, won, first, c: float = C, groups=None, groups2=None) -> dict:
    """Paired (return at bsp) - (return at ltp) for 1 unit backed, at every commission."""
    bsp = np.asarray(bsp, dtype=float)
    lp = np.asarray(lp, dtype=float)
    won = np.asarray(won, dtype=bool)
    first = np.asarray(first, dtype=bool)
    res = {"bets": int(len(lp)), "n_won": int(won.sum())}
    if len(lp) == 0:
        return res
    for cc in orig.COMMISSIONS:
        d = orig.back_returns(bsp, won, cc) - orig.back_returns(lp, won, cc)
        res[f"diff_c{int(cc * 100)}"] = r4(d.mean())
    d = orig.back_returns(bsp, won, c) - orig.back_returns(lp, won, c)
    m, se = mean_se(d)
    res["se"] = r4(se)
    res["z"] = r4(m / se) if se and se > 0 else None
    h = [mean_se(d[first]), mean_se(d[~first])]
    res["halves_c5"] = [r4(h[0][0]), r4(h[1][0])]
    res["halves_se"] = [r4(h[0][1]), r4(h[1][1])]
    res["halves_bets"] = [int(first.sum()), int((~first).sum())]
    res["both_halves_2se_negative"] = bool(all(hm < -2 * hs for hm, hs in h if not math.isnan(hs)))
    res["back_ltp_c5"] = r4(orig.back_returns(lp, won, c).mean())
    res["back_bsp_c5"] = r4(orig.back_returns(bsp, won, c).mean())
    res["back_tick_down_c5"] = r4(orig.back_returns(orig.tick_down(lp), won, c).mean())
    if groups is not None:
        cs = cluster_se(d, groups)
        res["se_cluster_date"] = r4(cs)
        res["z_cluster_date"] = r4(m / cs) if cs > 0 else None
    if groups2 is not None:
        cs = cluster_se(d, groups2)
        res["se_cluster_meeting"] = r4(cs)
        res["z_cluster_meeting"] = r4(m / cs) if cs > 0 else None
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=HERE.parent / "data/raw/betfair/split/au_win.parquet")
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    t0 = time.time()

    # ------------------------------------------------------------------ 1. universe data quality (raw file)
    raw = pd.read_parquet(a.path, columns=["market_id", "event_id", "venue", "start", "date", "field_size",
                                           "selection_id", "program", "ltp_pre", "bsp", "won", "dead_heat_share",
                                           "market_name"])
    n_raw = len(raw)
    raw = raw.drop_duplicates(["market_id", "selection_id"])
    if a.end:
        raw = raw[raw["date"] < pd.Timestamp(a.end)]
    mk = raw.groupby("market_id").agg(event_id=("event_id", "first"), start=("start", "first"), venue=("venue", "first"),
                                      field_size=("field_size", "first"), rows=("selection_id", "size"),
                                      won_sum=("won", "sum"), bsp_nan=("bsp", lambda s: int(s.isna().sum())),
                                      ltp_nan=("ltp_pre", lambda s: int(s.isna().sum())))
    ev_dup = mk.groupby(["event_id", "start"]).size()
    vs_dup = mk.groupby(["venue", "start"]).size()
    out({"section": "1_universe", "rows_raw": int(n_raw), "rows_after_dedupe": int(len(raw)), "markets": int(len(mk)),
         "duplicate_win_markets_same_event_and_start": int((ev_dup > 1).sum()),
         "duplicate_win_markets_same_venue_and_start": int((vs_dup > 1).sum()),
         "markets_rows_ne_field_size": int((mk["rows"] != mk["field_size"]).sum()),
         "markets_won_sum_0": int((mk["won_sum"] == 0).sum()), "markets_won_sum_gt1_dead_heat": int((mk["won_sum"] > 1).sum()),
         "markets_some_bsp_nan": int((mk["bsp_nan"] > 0).sum()), "markets_some_ltp_pre_nan": int((mk["ltp_nan"] > 0).sum()),
         "rows_bsp_le_1.01": int((raw["bsp"] <= 1.01).sum()), "rows_bsp_eq_ltp_pre": r4((raw["bsp"] == raw["ltp_pre"]).mean()),
         "rows_dead_heat_share_lt_1_among_winners": int(((raw["won"] == 1) & (raw["dead_heat_share"] < 1)).sum())})

    # ------------------------------------------------------------------ 0. reproduction via the original code
    df, info, mid = orig.load(a.path, a.end)
    out({"section": "0_load", **{k: info[k] for k in ("markets_used", "runners_used", "markets_gallops", "markets_harness",
                                                       "mid_date_for_halves", "bsp_runner_coverage", "t30_runner_coverage")}})
    gal = df[~df["harness"]].reset_index(drop=True)
    R = orig.Rets(gal)
    fav = gal["fav"].to_numpy()
    fg = gal["fgroup"].to_numpy()
    idx = np.flatnonzero(fav & (fg == "6-8"))
    rc = orig.ratio_cell(R, idx)
    out({"section": "0_reproduction_original_code", "cell": "gallops favourite field 6-8",
         "runners": rc["runners"], "median_ltp_pre": rc["median_ltp_pre"], "strike": rc["strike"], "implied_ltp": rc["implied_ltp"],
         "implied_bsp": rc["implied_bsp"], "share_sp_better_1tick": rc["share_sp_better_1tick"],
         "share_sp_worse_1tick": rc["share_sp_worse_1tick"],
         "back_ltp_pre": orig.compact(rc["back_ltp_pre"]), "back_ltp_tick_down": orig.compact(rc["back_ltp_pre_tick_down"]),
         "back_bsp": orig.compact(rc["back_bsp"]), "sp_minus_ltp": orig.compact(rc["sp_minus_ltp"]),
         "sp_minus_ltp_z": rc["sp_minus_ltp"].get("z"), "sp_minus_ltp_z_cons": rc["sp_minus_ltp"].get("z_cons"),
         "sp_minus_ltp_p_one_sided_as_in_original": rc["sp_minus_ltp"].get("p"),
         "bonferroni_122_two_sided_threshold_z": r4(-orig.math.sqrt(2) * math.erfc(1) if False else 3.5)})

    # convenience frame for the cell (favourites at ltp_pre, 6-8 fields, gallops, bsp present)
    cellf = gal.loc[idx].copy()
    cellf = cellf[cellf["bsp_ok"]]
    cellf["meeting"] = cellf["venue"].astype(str) + "|" + cellf["date"].astype(str)
    lp = cellf["ltp_pre"].to_numpy(float)
    bsp = cellf["bsp"].to_numpy(float)
    won = cellf["won_b"].to_numpy()
    first = cellf["first"].to_numpy()
    dates = cellf["date"].to_numpy()
    meet = cellf["meeting"].to_numpy()
    ratio = bsp / lp
    base = paired_summary(bsp, lp, won, first, groups=dates, groups2=meet)
    out({"section": "0_reproduction_independent", "cell": "gallops favourite field 6-8", **base})

    # ------------------------------------------------------------------ 2. in-play leakage tail in ltp_pre
    lr = np.log(ratio)
    tail = {}
    for lab, m in (("winners", won), ("losers", ~won)):
        q = np.quantile(lr[m], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        tail[lab] = {"n": int(m.sum()), "log_ratio_q01_05_25_50_75_95_99": [r4(v) for v in q],
                     "mean_ratio_minus_1": r4((ratio[m] - 1).mean()),
                     "share_bsp_gt_1.2_ltp": r4((ratio[m] > 1.2).mean()), "share_bsp_lt_0.8_ltp": r4((ratio[m] < 0.8).mean()),
                     "share_bsp_gt_1.5_ltp": r4((ratio[m] > 1.5).mean()), "share_bsp_lt_0.67_ltp": r4((ratio[m] < 1 / 1.5).mean()),
                     "share_bsp_below_ltp": r4((bsp[m] < lp[m] - 1e-9).mean()), "share_bsp_above_ltp": r4((bsp[m] > lp[m] + 1e-9).mean()),
                     "share_bsp_below_ltp_by_tick": r4(R.sp_worse_tick[idx][R.bsp_ok[idx]][m].mean()),
                     "share_bsp_above_ltp_by_tick": r4(R.sp_better_tick[idx][R.bsp_ok[idx]][m].mean())}
    # sign test on winners: bsp < ltp vs bsp > ltp (ties excluded)
    w = won
    below = int((bsp[w] < lp[w] - 1e-9).sum())
    above = int((bsp[w] > lp[w] + 1e-9).sum())
    nz = below + above
    z_sign = (below - above) / math.sqrt(nz) if nz else float("nan")
    out({"section": "2_inplay_tail_and_sign_test", "by_outcome": tail,
         "sign_test_winners": {"bsp_below_ltp": below, "bsp_above_ltp": above, "ties": int(w.sum() - nz), "z_below_minus_above": r4(z_sign)},
         "note": "in-play contamination of ltp_pre would put WINNERS' ltp_pre far BELOW bsp (ratio >> 1): that would push the paired diff POSITIVE, against the claim"})

    # ------------------------------------------------------------------ 3. decomposition systematic vs informational
    k = 1.0 - C
    p_win = won.mean()
    e_all = (bsp - lp).mean()
    e_win = (bsp[won] - lp[won]).mean()
    e_lose = (bsp[~won] - lp[~won]).mean()
    out({"section": "3_decomposition", "p_win": r4(p_win),
         "E_bsp_minus_ltp_all_favs": r4(e_all), "E_bsp_minus_ltp_winners": r4(e_win), "E_bsp_minus_ltp_losers": r4(e_lose),
         "paired_diff_c5": r4(p_win * e_win * k),
         "systematic_part_c5 (P(win) * E[bsp-ltp | all] * 0.95)": r4(p_win * e_all * k),
         "informational_part_c5 (P(win) * (E[win] - E[all]) * 0.95)": r4(p_win * (e_win - e_all) * k),
         "winners_shorten_in_ticks_at_median_price": r4(e_win / 0.02),
         "implied_prob_mean_ltp": r4((1 / lp).mean()), "implied_prob_mean_bsp": r4((1 / bsp).mean()), "strike": r4(p_win),
         "logloss_ltp": r4(-(np.where(won, np.log(1 / lp), np.log(1 - 1 / lp))).mean()),
         "logloss_bsp": r4(-(np.where(won, np.log(1 / bsp), np.log(1 - 1 / bsp))).mean())})

    # ------------------------------------------------------------------ 4. robustness
    rob = {}
    for lab, keep in (("|log ratio| <= 0.05", np.abs(lr) <= 0.05), ("|log ratio| <= 0.10", np.abs(lr) <= 0.10),
                      ("|log ratio| <= 0.20", np.abs(lr) <= 0.20), ("|log ratio| > 0.10 (the excluded tail)", np.abs(lr) > 0.10)):
        rob[lab] = paired_summary(bsp[keep], lp[keep], won[keep], first[keep])
    # winsorised bsp at ltp * exp(+/-0.1)
    bw = lp * np.exp(np.clip(lr, -0.1, 0.1))
    rob["winsorised bsp within +/-10% of ltp"] = paired_summary(bw, lp, won, first)
    years = pd.DatetimeIndex(cellf["date"]).year.to_numpy()
    by_year = {}
    for y in sorted(set(years)):
        m = years == y
        s = paired_summary(bsp[m], lp[m], won[m], first[m])
        by_year[str(y)] = [s["bets"], s["diff_c5"], s["se"], s["z"]]
    fs = cellf["field_size"].to_numpy()
    by_fs = {}
    for f in (6, 7, 8):
        m = fs == f
        s = paired_summary(bsp[m], lp[m], won[m], first[m])
        by_fs[str(f)] = [s["bets"], s["diff_c5"], s["se"], s["z"], s["halves_c5"]]
    band = cellf["band"].to_numpy()
    by_band = {}
    for b in orig.BAND_LABELS:
        m = band == b
        if m.sum() < 50:
            continue
        s = paired_summary(bsp[m], lp[m], won[m], first[m])
        by_band[b] = [s["bets"], s["diff_c5"], s["se"], s["z"], s["halves_c5"], s["back_ltp_c5"], s["back_tick_down_c5"], s["back_bsp_c5"]]
    out({"section": "4_robustness", "trimmed": rob, "by_year [bets, diff_c5, se, z]": by_year,
         "by_field_size [bets, diff_c5, se, z, halves]": by_fs,
         "by_ltp_band [bets, diff_c5, se, z, halves, back_ltp_c5, back_tick_down_c5, back_bsp_c5]": by_band})

    # ------------------------------------------------------------------ 5. staleness proxies
    book = (1.0 / gal["ltp_pre"]).groupby(gal["market_id"]).sum()
    t30_share = gal["has_t30"].groupby(gal["market_id"]).mean()
    cellf["book"] = book.loc[cellf["market_id"]].to_numpy()
    cellf["t30_share"] = t30_share.loc[cellf["market_id"]].to_numpy()
    bk = cellf["book"].to_numpy()
    stale = {}
    for lab, m in (("book < 0.99", bk < 0.99), ("0.99 <= book < 1.01", (bk >= 0.99) & (bk < 1.01)),
                   ("1.01 <= book < 1.03", (bk >= 1.01) & (bk < 1.03)), ("book >= 1.03", bk >= 1.03)):
        s = paired_summary(bsp[m], lp[m], won[m], first[m])
        stale[lab] = {"bets": s["bets"], "diff_c5": s["diff_c5"], "se": s["se"], "z": s["z"], "halves": s["halves_c5"],
                      "median_abs_log_ratio": r4(np.median(np.abs(lr[m]))) if m.any() else None,
                      "share_sp_worse_tick": r4((bsp[m] <= orig.tick_down(lp[m]) + 1e-9).mean()) if m.any() else None,
                      "share_sp_better_tick": r4((bsp[m] >= orig.tick_up(lp[m]) - 1e-9).mean()) if m.any() else None}
    ts = cellf["t30_share"].to_numpy()
    liq = {}
    for lab, m in (("no runner traded at t30", ts == 0), ("0 < share < 0.5", (ts > 0) & (ts < 0.5)),
                   ("0.5 <= share < 1", (ts >= 0.5) & (ts < 1)), ("all runners traded at t30", ts >= 1)):
        s = paired_summary(bsp[m], lp[m], won[m], first[m])
        liq[lab] = {"bets": s["bets"], "diff_c5": s["diff_c5"], "se": s["se"], "z": s["z"], "halves": s["halves_c5"],
                    "median_abs_log_ratio": r4(np.median(np.abs(lr[m]))) if m.any() else None}
    out({"section": "5_staleness_proxies", "book_quantiles_cell_markets": [r4(v) for v in np.quantile(bk, [0.05, 0.25, 0.5, 0.75, 0.95])],
         "by_market_book_at_ltp_pre": stale, "by_share_of_field_traded_at_t30": liq,
         "note": "a stale last trade on some runner makes the book inconsistent; a tight book means the last trades are mutually consistent (fresh)"})

    # ------------------------------------------------------------------ 6. removed-runner proxy (saddlecloth gap)
    prog = pd.to_numeric(gal["program"], errors="coerce")
    maxprog = prog.groupby(gal["market_id"]).max()
    cellf["maxprog"] = maxprog.loc[cellf["market_id"]].to_numpy()
    gap = cellf["maxprog"].to_numpy() > cellf["field_size"].to_numpy()
    scr = {}
    for lab, m in (("no saddlecloth gap (no scratching)", ~gap), ("saddlecloth gap (a scratching at some point)", gap)):
        s = paired_summary(bsp[m], lp[m], won[m], first[m])
        scr[lab] = {"bets": s["bets"], "diff_c5": s["diff_c5"], "se": s["se"], "z": s["z"], "halves": s["halves_c5"],
                    "mean_ratio_minus_1_all": r4((ratio[m] - 1).mean()) if m.any() else None}
    out({"section": "6_scratching_proxy", "share_markets_with_gap": r4(gap.mean()), "program_unparsed_share": r4(prog.isna().mean()), "rows": scr,
         "note": "removed runners are dropped by the parser; a gap in saddlecloth numbers is the only trace. Reduction factors after a late scratching lower BSP relative to earlier trades"})

    # ------------------------------------------------------------------ 7. t30 comparisons (conditional)
    t30 = cellf["ltp_t30"].to_numpy(float)
    has = cellf["has_t30"].to_numpy()
    t30res = {"favourite_at_ltp_pre_with_t30_trade": {"share_with_t30": r4(has.mean()),
                                                        "bsp_minus_ltp_pre": paired_summary(bsp[has], lp[has], won[has], first[has]),
                                                        "bsp_minus_ltp_t30": paired_summary(bsp[has], t30[has], won[has], first[has]),
                                                        "ltp_pre_minus_ltp_t30": paired_summary(lp[has], t30[has], won[has], first[has]),
                                                        "median_ratio_bsp_over_t30": r4(np.median(bsp[has] / t30[has])),
                                                        "E_log_t30_over_pre_winners": r4(np.log(t30[has & won] / lp[has & won]).mean()),
                                                        "E_log_t30_over_pre_losers": r4(np.log(t30[has & ~won] / lp[has & ~won]).mean())}}
    # favourite defined at t30 (rank 1 among runners with a t30 trade, in markets where every runner traded at t30)
    g30 = gal[gal["has_t30"]].copy()
    allt = gal.groupby("market_id")["has_t30"].min()
    g30 = g30[g30["market_id"].map(allt).astype(bool) & (g30["fgroup"] == "6-8")]
    g30 = g30.sort_values(["market_id", "ltp_t30", "selection_id"])
    f30 = g30[g30.groupby("market_id").cumcount() == 0]
    f30 = f30[f30["bsp_ok"]]
    t30res["favourite_at_t30_markets_all_traded_t30"] = {
        "bets": int(len(f30)), "same_runner_as_ltp_pre_fav_share": r4(f30["fav"].mean()),
        "bsp_minus_ltp_pre": paired_summary(f30["bsp"], f30["ltp_pre"], f30["won_b"], f30["first"]),
        "bsp_minus_ltp_t30": paired_summary(f30["bsp"], f30["ltp_t30"], f30["won_b"], f30["first"]),
        "ltp_pre_minus_ltp_t30": paired_summary(f30["ltp_pre"], f30["ltp_t30"], f30["won_b"], f30["first"])}
    out({"section": "7_t30_conditional", **t30res,
         "note": "CONDITIONAL on t30 trading (liquid markets). If bsp also beats ltp_t30 on winners, the last-minute price is the exception, not the rule"})

    # ------------------------------------------------------------------ 8. survivorship: without the all-priced market filter
    cols = ["market_id", "market_name", "date", "field_size", "selection_id", "ltp_pre", "bsp", "won", "venue"]
    full = pd.read_parquet(a.path, columns=cols).drop_duplicates(["market_id", "selection_id"])
    if a.end:
        full = full[full["date"] < pd.Timestamp(a.end)]
    full = full[full["won"].notna()]
    full = full[~full["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b")]
    one = full.groupby("market_id")["won"].transform("sum") == 1
    full = full[one]
    priced = full[full["ltp_pre"].notna() & (full["ltp_pre"] > 1)].sort_values(["market_id", "ltp_pre", "selection_id"])
    favs = priced[priced.groupby("market_id").cumcount() == 0]
    favs = favs[(favs["field_size"] >= 6) & (favs["field_size"] <= 8) & favs["bsp"].notna() & (favs["bsp"] > 1)]
    all_priced = priced.groupby("market_id").size() == full.groupby("market_id").size().loc[priced["market_id"].unique()]
    favs["all_priced"] = favs["market_id"].map(all_priced).astype(bool)
    favs["first"] = favs["date"] < mid
    s_all = paired_summary(favs["bsp"], favs["ltp_pre"], favs["won"] == 1, favs["first"])
    s_drop = paired_summary(favs.loc[~favs["all_priced"], "bsp"], favs.loc[~favs["all_priced"], "ltp_pre"],
                            favs.loc[~favs["all_priced"], "won"] == 1, favs.loc[~favs["all_priced"], "first"])
    out({"section": "8_survivorship", "favourites_6-8_without_all_priced_filter": s_all,
         "favourites_in_markets_dropped_by_filter": s_drop, "dropped_markets_in_cell": int((~favs["all_priced"]).sum())})

    # ------------------------------------------------------------------ 9. execution sensitivity and harness contrast
    r_ltp = orig.back_returns(lp, won, C)
    r_tick = orig.back_returns(orig.tick_down(lp), won, C)
    r_bsp = orig.back_returns(bsp, won, C)
    r_half = 0.5 * (r_ltp + r_tick)
    d_half = r_bsp - r_half
    mh, sh = mean_se(d_half)
    d_tick = r_bsp - r_tick
    mt, st = mean_se(d_tick)
    p_star = (r_bsp.mean() - r_tick.mean()) / (r_ltp.mean() - r_tick.mean())
    # two-tick worse
    r_tick2 = orig.back_returns(orig.tick_down(orig.tick_down(lp)), won, C)
    hcell = df[df["harness"] & df["fav"] & (df["fgroup"] == "6-8") & df["bsp_ok"]]
    out({"section": "9_execution", "roi_c5": {"at_ltp": r4(r_ltp.mean()), "one_tick_worse": r4(r_tick.mean()), "two_ticks_worse": r4(r_tick2.mean()),
                                              "at_bsp": r4(r_bsp.mean())},
         "sp_minus_ltp_c5": r4(base["diff_c5"]), "sp_minus_halftick_c5": [r4(mh), r4(sh)], "sp_minus_one_tick_worse_c5": [r4(mt), r4(st)],
         "share_of_fills_at_exactly_ltp_needed_to_tie_sp (rest one tick worse)": r4(p_star),
         "median_tick_as_share_of_price": r4(np.median(orig._increment(lp) / lp)),
         "harness_favourites_6-8_for_contrast": paired_summary(hcell["bsp"], hcell["ltp_pre"], hcell["won_b"], hcell["first"])})

    # ------------------------------------------------------------------ 10. the other favourite cells quoted in the claim
    other = {}
    for f in orig.FIELD_LABELS + ["all"]:
        i2 = np.flatnonzero(fav & ((fg == f) if f != "all" else True))
        i2 = i2[R.bsp_ok[i2]]
        s = paired_summary(R.bsp[i2], R.lp[i2], R.won[i2], R.first[i2], groups=gal["date"].to_numpy()[i2])
        other[f] = [s["bets"], s["diff_c5"], s["se"], s["se_cluster_date"], s["z"], s["halves_c5"]]
    out({"section": "10_favourites_by_field [bets, diff_c5, se, se_cluster_date, z, halves]", "gallops": other})
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

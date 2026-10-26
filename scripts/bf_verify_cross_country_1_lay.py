"""Adversarial verification of the "cross_country" LAY value-rule claim (mirror of the back one):

    US WIN strict markets (every runner traded pre-off, one winner): lay 1 unit of backer's
    stake at ltp_pre when p_model * ltp_pre - 1 <= -0.10 and ltp_pre <= 50, p_model = the
    walk-forward power recalibration of the proportional market probability.
    Claimed: n 21283, +21.59% of stake after 5% commission, SE 0.0239 (z 9), halves
    0.1989/0.2361, BH survivor of 192 lay cells; 95% of the bets sit in markets whose LTP
    book is > 1.05 (median 1.145), median BSP/ltp_pre 1.14, and the SAME lays at BSP return
    +3.6% (SE 3.8%, n 21262) = nothing. Labelled "artefact, not executable".

    python scripts/bf_verify_cross_country_1_lay.py [--countries US,FR,NZ,ZA] [--burn-in 3] [--bh]

Reuses scripts/bf_cross_country.py (imported, not edited) for the data loading, the
walk-forward beta and the cell summaries, so the reproduction is exact, then adds:

  * the structure of the selection: distinct markets, lays per market, share of bets where
    every runner (every runner <= 50) of the market is laid, the LTP book of the selected
    markets, R^2 of the "edge" on 1/book-1, and a model-free rule "lay every runner <= 50 in
    markets whose LTP book >= b" with its overlap with the claimed rule;
  * honest uncertainty: cluster-robust SEs by market and by race day at ltp_pre and at BSP,
    c0/c5/c8, per half and per calendar year; ROI per unit of LIABILITY (the capital a lay
    puts at risk) with a cluster-robust SE;
  * calibration of the selected runners: strike vs 1/ltp_pre, 1/BSP and p_model;
  * the lay dutch (stake 1/ltp_pre per runner, which locks in (book-1)/book of turnover if
    every runner could be laid at ltp_pre) settled at ltp_pre and at BSP;
  * the same lays at BSP in depth: BSP book of the selected markets, BSP/ltp_pre, by ltp_pre
    band, by LTP-book bin, winners' vs losers' BSP/ltp_pre (in-play leak test: in-play
    trades would make the winners' last trade SHORTER than BSP);
  * subsets where ltp_pre is least likely to be stale: every runner traded 30 min out, BSP
    book within 3% of 100%, LTP book within 3% of 100% (the rule fired because of the
    recalibration, not the book), and ltp_pre == ltp_t30 (no trade in the last 30 min);
  * executability: lay price sensitivity (ltp_pre x 1.00/1.05/1.10/1.14/1.20 and the
    break-even multiplier), one tick above ltp_pre, a limit lay at ltp_pre that is matched
    only when the market does not drift out (BSP <= ltp_pre), and SP lays with a fixed
    liability per bet (the only form an SP lay order takes);
  * date concentration;
  * --bh: rerun all six countries through the original analyse_country to rebuild the
    192-cell lay family and redo Benjamini-Hochberg with the US cell's per-bet SE replaced
    by its market- and day-clustered SE.

Prints JSON lines only; never prints horse names (the loader does not read them).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import bf_cross_country as cc  # noqa: E402

out, r4 = cc.out, cc.r4
BOOK_BINS = [(0.0, 1.05), (1.05, 1.10), (1.10, 1.15), (1.15, 1.25), (1.25, 9.0)]


def cluster_se(r: np.ndarray, g: np.ndarray) -> tuple[float, float, int]:
    """Mean, cluster-robust SE of the mean (clusters g), number of clusters."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n < 2:
        return (float(r.mean()) if n else float("nan")), float("nan"), n
    m = r.mean()
    codes, uniq = pd.factorize(g)
    G = len(uniq)
    s = np.bincount(codes, weights=r - m, minlength=G)
    se = math.sqrt((s * s).sum()) / n * math.sqrt(G / max(G - 1, 1))
    return float(m), float(se), int(G)


def ratio_cluster_se(r: np.ndarray, w: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    """sum(r)/sum(w) (e.g. ROI per unit of liability) with a cluster-robust SE (linearised)."""
    r = np.asarray(r, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(r) < 2 or w.sum() <= 0:
        return float("nan"), float("nan")
    m = r.sum() / w.sum()
    u = r - m * w
    codes, uniq = pd.factorize(g)
    G = len(uniq)
    s = np.bincount(codes, weights=u, minlength=G)
    se = math.sqrt((s * s).sum()) / w.sum() * math.sqrt(G / max(G - 1, 1))
    return float(m), float(se)


def summarise(r: np.ndarray, mk: np.ndarray, dt: np.ndarray, first: np.ndarray) -> dict:
    m, se_m, G = cluster_se(r, mk)
    _, se_d, D = cluster_se(r, dt)
    se_b = float(r.std(ddof=1) / math.sqrt(len(r))) if len(r) > 1 else float("nan")
    res = {"bets": int(len(r)), "markets": G, "race_days": D, "roi": r4(m), "se_per_bet": r4(se_b),
           "se_cluster_market": r4(se_m), "se_cluster_day": r4(se_d),
           "z_per_bet": r4(m / se_b) if se_b else None, "z_cluster_market": r4(m / se_m) if se_m else None,
           "z_cluster_day": r4(m / se_d) if se_d else None}
    halves = []
    for mask in (first, ~first):
        if mask.sum() < 2:
            halves.append({"bets": int(mask.sum())})
            continue
        hm, hs, hg = cluster_se(r[mask], mk[mask])
        _, hd, _ = cluster_se(r[mask], dt[mask])
        hb = float(r[mask].std(ddof=1) / math.sqrt(mask.sum()))
        halves.append({"bets": int(mask.sum()), "markets": hg, "roi": r4(hm), "se_per_bet": r4(hb),
                       "se_cluster_market": r4(hs), "se_cluster_day": r4(hd),
                       "z_per_bet": r4(hm / hb) if hb else None, "z_cluster_market": r4(hm / hs) if hs else None,
                       "z_cluster_day": r4(hm / hd) if hd else None})
    res["halves"] = halves
    res["both_halves_gt_2se_cluster_market"] = bool(all(
        h.get("z_cluster_market") is not None and h["z_cluster_market"] > 2 for h in halves))
    return res


def lay_ret(price: np.ndarray, won: np.ndarray, c: float = 0.05) -> np.ndarray:
    return cc.lay_returns(price, won, c)


def verify_country(code: str, margin: float, burn_in: int) -> dict:
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        df, thin, info = cc.load_country(code, None)
    mid = df.drop_duplicates("market_id")["date"].median()
    rets = cc.Returns(df, mid)
    won = df["won_b"].to_numpy()
    lp = df["ltp_pre"].to_numpy(dtype=float)
    bsp = df["bsp"].to_numpy(dtype=float)
    t30 = df["ltp_t30"].to_numpy(dtype=float)
    book = df["book"].to_numpy(dtype=float)
    codes = df["race"].to_numpy()
    mk = df["market_id"].to_numpy()
    dt = df["date"].to_numpy()
    year = pd.DatetimeIndex(df["date"]).year.to_numpy()
    first = (df["date"] < mid).to_numpy()
    p_model, fits = cc.walk_forward_power(df, burn_in)
    scored = ~np.isnan(p_model)
    edge = np.where(scored & (lp <= cc.VALUE_MAX_PRICE), p_model * lp - 1.0, np.nan)
    sel = np.isfinite(edge) & (edge <= -margin)
    idx = np.flatnonzero(sel)
    bsp_ok = np.isfinite(bsp) & (bsp > 1)
    ib = idx[bsp_ok[idx]]
    if len(idx) < 2:
        out({"section": "reproduction", "country": code, "bets": int(len(idx)), "note": "too few bets"})
        return {}

    # 1. exact reproduction through the original cell()
    a = rets.cell(idx, "ltp_pre")
    b = rets.cell(idx, "bsp")
    keys = ("bets", "lay_roi_c5", "lay_se", "lay_halves_roi_c5", "lay_halves_se_c5", "halves_bets",
            "median_price", "strike", "implied_raw", "implied_prop")
    out({"section": "reproduction", "country": code,
         "rule": f"lay 1u stake at ltp_pre if p_model*ltp_pre-1 <= -{margin:g}, ltp_pre<={cc.VALUE_MAX_PRICE:g}",
         "markets_used": info["markets_used"], "mid_date": mid,
         "ltp_pre": {k: a.get(k) for k in keys}, "bsp": {k: b.get(k) for k in keys},
         "beta_wf_median": r4(np.median([f["beta"] for f in fits])) if fits else None,
         "beta_wf_min_max": ([min(f["beta"] for f in fits), max(f["beta"] for f in fits)] if fits else None)})

    # 2. selection structure: whole markets, lays per market, edge vs 1/book-1, model-free book rule
    n_in_market = pd.Series(sel).groupby(codes).transform("sum").to_numpy()
    field = pd.Series(np.ones(len(df))).groupby(codes).transform("sum").to_numpy()
    n_le50 = pd.Series(lp <= cc.VALUE_MAX_PRICE).groupby(codes).transform("sum").to_numpy()
    inv_book_edge = 1.0 / book[idx] - 1.0
    e = edge[idx]
    corr = float(np.corrcoef(e, inv_book_edge)[0, 1])
    mk_sel = pd.Series(idx).groupby(mk[idx]).size()
    book_rule = {}
    for thr in (1.05, 1.08, 1.10, 1.111, 1.15, 1.25):
        m = scored & (book >= thr) & (lp <= cc.VALUE_MAX_PRICE)
        ii = np.flatnonzero(m)
        jj = ii[bsp_ok[ii]]
        ml, sl, gl = cluster_se(rets.lay5["ltp_pre"][ii], mk[ii])
        mb, sb, _ = cluster_se(rets.lay5["bsp"][jj], mk[jj]) if len(jj) > 1 else (float("nan"),) * 3
        book_rule[f"book>={thr:g}"] = {"bets": int(len(ii)), "markets": gl, "lay_ltp_c5": r4(ml), "se_cl": r4(sl),
                                       "lay_bsp_c5": r4(mb), "bsp_se_cl": r4(sb),
                                       "share_of_these_in_claimed_rule": r4(np.isin(ii, idx).mean()),
                                       "share_of_claimed_rule_covered": r4(np.isin(idx, ii).mean())}
    all_mk = df.drop_duplicates("market_id")["book"]
    out({"section": "selection_structure", "country": code, "bets": int(len(idx)),
         "distinct_markets": int(mk_sel.size), "lays_per_market_mean": r4(mk_sel.mean()),
         "lays_per_market_median": r4(mk_sel.median()), "lays_per_market_max": int(mk_sel.max()),
         "share_bets_whole_market_laid": r4((n_in_market == field)[idx].mean()),
         "share_bets_all_runners_le50_laid": r4((n_in_market == n_le50)[idx].mean()),
         "ltp_book_of_selected": {"median": r4(np.median(book[idx])), "q10": r4(np.quantile(book[idx], 0.1)),
                                  "q90": r4(np.quantile(book[idx], 0.9)), "share_over_1.05": r4((book[idx] > 1.05).mean()),
                                  "share_over_1.10": r4((book[idx] > 1.10).mean())},
         "edge_vs_inv_book_minus_1": {"corr": r4(corr), "r2": r4(corr * corr), "median_edge": r4(np.median(e)),
                                      "median_inv_book_minus_1": r4(np.median(inv_book_edge)),
                                      "median_abs_diff": r4(np.median(np.abs(e - inv_book_edge)))},
         "share_of_all_strict_markets_with_book_over_1.10": r4((all_mk > 1.10).mean()),
         "share_of_all_strict_markets_with_book_over_1.05": r4((all_mk > 1.05).mean()),
         "model_free_book_rule": book_rule})

    # 3. clustered SEs at ltp_pre and BSP (same bets), c0/c5/c8; ROI per unit of liability
    res = {"section": "clustered_se", "country": code, "bsp_coverage_of_selected": r4(bsp_ok[idx].mean())}
    for c in cc.COMMISSIONS:
        res[f"lay_ltp_pre_c{int(c*100)}"] = summarise(lay_ret(lp[idx], won[idx], c), mk[idx], dt[idx], first[idx])
        res[f"lay_bsp_c{int(c*100)}"] = summarise(lay_ret(bsp[ib], won[ib], c), mk[ib], dt[ib], first[ib])
    liab = {}
    for name, ii, pr in (("ltp_pre", idx, lp[idx]), ("bsp", ib, bsp[ib])):
        r5 = lay_ret(pr, won[ii], 0.05)
        m, s = ratio_cluster_se(r5, pr - 1.0, mk[ii])
        hv = []
        for mask in (first[ii], ~first[ii]):
            hm, hs = ratio_cluster_se(r5[mask], pr[mask] - 1.0, mk[ii][mask])
            hv.append([r4(hm), r4(hs)])
        liab[name] = {"roi_on_liability_c5": r4(m), "se_cluster_market": r4(s), "z": r4(m / s) if s else None,
                      "halves": hv, "mean_liability_per_unit_stake": r4((pr - 1.0).mean()),
                      "liability_per_unit_stake_q90": r4(np.quantile(pr - 1.0, 0.9))}
    res["per_unit_liability"] = liab
    out(res)

    # 4. calibration of the selected runners and yearly returns
    cal = {"strike": r4(won[idx].mean()), "mean_1_over_ltp_pre": r4((1.0 / lp[idx]).mean()),
           "mean_1_over_bsp": r4((1.0 / bsp[ib]).mean()), "mean_p_model": r4(p_model[idx].mean()),
           "mean_p_prop": r4(df["p_prop"].to_numpy()[idx].mean()),
           "strike_over_1_over_ltp_pre": r4(won[idx].mean() / (1.0 / lp[idx]).mean()),
           "strike_over_1_over_bsp": r4(won[ib].mean() / (1.0 / bsp[ib]).mean())}
    years = {}
    for y in sorted(set(year[idx].tolist())):
        m = year[idx] == y
        mb = year[ib] == y
        ml, sl, gl = cluster_se(rets.lay5["ltp_pre"][idx][m], mk[idx][m])
        mb_, sb, _ = cluster_se(rets.lay5["bsp"][ib][mb], mk[ib][mb]) if mb.sum() > 1 else (float("nan"),) * 3
        years[str(y)] = {"bets": int(m.sum()), "markets": gl, "lay_ltp_c5": r4(ml), "se_cl": r4(sl),
                         "bsp_bets": int(mb.sum()), "lay_bsp_c5": r4(mb_), "bsp_se_cl": r4(sb)}
    out({"section": "calibration_and_years", "country": code, "calibration_of_selected": cal, "by_year": years})

    # 5. lay dutch: stake 1/ltp_pre per runner locks in (book-1)/book of turnover if EVERY runner is
    #    laid at ltp_pre; settled at ltp_pre and (same stakes) at BSP
    stake = 1.0 / lp[idx]
    d_ltp = np.where(won[idx], -stake * (lp[idx] - 1.0), stake * 0.95)
    stake_b = 1.0 / lp[ib]
    d_bsp = np.where(won[ib], -stake_b * (bsp[ib] - 1.0), stake_b * 0.95)
    out({"section": "dutch_lay_1_over_ltp_pre_stakes_c5", "country": code,
         "roi_on_turnover_ltp_pre": r4(d_ltp.sum() / stake.sum()),
         "roi_on_turnover_bsp": r4(d_bsp.sum() / stake_b.sum()),
         "locked_in_if_whole_market_at_ltp_pre_median": r4(np.median((book[idx] - 1.0) / book[idx]))})

    # 6. BSP settlement in depth
    full_bsp = pd.Series(bsp_ok).groupby(codes).transform("min").to_numpy().astype(bool)
    bb = pd.Series(np.where(bsp_ok, 1.0 / np.where(bsp_ok, bsp, 1.0), 0.0)).groupby(codes).transform("sum").to_numpy()
    bsp_book = np.where(full_bsp, bb, np.nan)
    ratio = bsp[ib] / lp[ib]
    l5_ltp_b = rets.lay5["ltp_pre"][ib]
    l5_bsp = rets.lay5["bsp"][ib]
    band = np.asarray(pd.cut(lp[ib], cc.PRICE_BANDS, right=False, labels=cc.BAND_LABELS).astype(str))
    by_band = {}
    for lab in cc.BAND_LABELS:
        m = band == lab
        if m.sum() >= 2:
            mm, ss, gg = cluster_se(l5_bsp[m], mk[ib][m])
            ml, sl, _ = cluster_se(l5_ltp_b[m], mk[ib][m])
            by_band[lab] = {"bets": int(m.sum()), "markets": gg, "lay_ltp_c5": r4(ml), "ltp_se_cl": r4(sl),
                            "lay_bsp_c5": r4(mm), "bsp_se_cl": r4(ss), "median_bsp_over_ltp": r4(np.median(ratio[m])),
                            "strike": r4(won[ib][m].mean()), "mean_1_over_ltp": r4((1.0 / lp[ib][m]).mean()),
                            "mean_1_over_bsp": r4((1.0 / bsp[ib][m]).mean())}
    by_book = {}
    for lo, hi in BOOK_BINS:
        m = (book[ib] >= lo) & (book[ib] < hi)
        if m.sum() >= 2:
            mm, ss, gg = cluster_se(l5_bsp[m], mk[ib][m])
            ml, sl, _ = cluster_se(l5_ltp_b[m], mk[ib][m])
            by_book[f"{lo:g}-{hi:g}"] = {"bets": int(m.sum()), "markets": gg, "lay_ltp_c5": r4(ml), "ltp_se_cl": r4(sl),
                                         "lay_bsp_c5": r4(mm), "bsp_se_cl": r4(ss),
                                         "median_bsp_over_ltp": r4(np.median(ratio[m])),
                                         "bsp_book_median": r4(np.nanmedian(bsp_book[ib][m]))}
    out({"section": "bsp_settlement", "country": code, "bets_with_bsp": int(len(ib)),
         "bsp_book_of_selected_markets": {"median": r4(np.nanmedian(bsp_book[ib])),
                                          "q10": r4(np.nanquantile(bsp_book[ib], 0.1)),
                                          "q90": r4(np.nanquantile(bsp_book[ib], 0.9)),
                                          "share_markets_full_bsp": r4(np.isfinite(bsp_book[ib]).mean())},
         "bsp_book_all_strict_markets_median": r4(np.nanmedian(pd.Series(bsp_book).groupby(codes).first())),
         "median_bsp_over_ltp_pre": r4(np.median(ratio)), "share_bsp_above_ltp_pre": r4((ratio > 1).mean()),
         "winners_median_bsp_over_ltp": r4(np.median(ratio[won[ib]])),
         "losers_median_bsp_over_ltp": r4(np.median(ratio[~won[ib]])),
         "in_play_leak_note": "in-play trades in ltp_pre would make winners' BSP/ltp_pre > losers' by a lot (winners "
                              "shorten in running); stale pre-off trades move both alike",
         "by_ltp_pre_band": by_band, "by_ltp_book_bin": by_book})

    # 7. where ltp_pre is least likely to be stale
    has_t30 = df["has_t30"].to_numpy()
    all_t30 = pd.Series(has_t30).groupby(codes).transform("min").to_numpy().astype(bool)
    near_bsp = np.isfinite(bsp_book) & (np.abs(bsp_book - 1.0) <= 0.03)
    near_ltp = np.abs(book - 1.0) <= 0.03
    same_t30 = has_t30 & np.isfinite(t30) & (np.abs(t30 - lp) < 1e-9)
    sub = {"share_selected_with_t30": r4(has_t30[idx].mean()),
           "share_selected_ltp_pre_equals_ltp_t30_given_t30": r4(same_t30[idx][has_t30[idx]].mean()) if has_t30[idx].any() else None,
           "share_all_runners_ltp_pre_equals_ltp_t30_given_t30": r4(same_t30[has_t30].mean()) if has_t30.any() else None}
    for name, mask in (("all_runners_t30", all_t30), ("bsp_book_within_3pct", near_bsp),
                       ("ltp_book_within_3pct", near_ltp), ("ltp_pre_equals_ltp_t30", same_t30),
                       ("t30_present_and_moved_since", has_t30 & ~same_t30)):
        ii = idx[mask[idx]]
        jj = ii[bsp_ok[ii]]
        if len(ii) >= 2:
            ml, sl, gl = cluster_se(rets.lay5["ltp_pre"][ii], mk[ii])
            entry = {"bets": int(len(ii)), "markets": gl, "lay_ltp_c5": r4(ml), "ltp_se_cl": r4(sl),
                     "ltp_book_median": r4(np.median(book[ii]))}
            if len(jj) >= 2:
                mb, sb, gb = cluster_se(rets.lay5["bsp"][jj], mk[jj])
                entry.update({"bsp_bets": int(len(jj)), "lay_bsp_c5": r4(mb), "bsp_se_cl": r4(sb),
                              "median_bsp_over_ltp": r4(np.median(bsp[jj] / lp[jj]))})
            sub[name] = entry
        else:
            sub[name] = {"bets": int(len(ii))}
    out({"section": "fresh_price_subsets", "country": code, **sub})

    # 7b. the liquid subset: runners with a trade 30 min out (t30 present for ~15% of US runners). All
    #     scored runners <= 50 split by t30 presence x LTP book (<= / > 1.05), laid at BSP, with cluster
    #     SE, halves, years and ROI per unit of liability; then the claimed rule inside each t30 group and
    #     by the direction of the move since t30. Post-hoc cells found during verification.
    def bsp_cell(mask: np.ndarray) -> dict:
        ii = np.flatnonzero(mask & bsp_ok)
        if len(ii) < 2:
            return {"bets": int(len(ii))}
        r = lay_ret(bsp[ii], won[ii], 0.05)
        m, s, g = cluster_se(r, mk[ii])
        hv = []
        for f in (first[ii], ~first[ii]):
            hm, hs, _ = cluster_se(r[f], mk[ii][f]) if f.sum() > 1 else (float("nan"), float("nan"), 0)
            hv.append([int(f.sum()), r4(hm), r4(hs)])
        lm, ls = ratio_cluster_se(r, bsp[ii] - 1.0, mk[ii])
        yrs = {str(y): [int((year[ii] == y).sum()), r4(r[year[ii] == y].mean())] for y in sorted(set(year[ii].tolist()))}
        return {"bets": int(len(ii)), "markets": g, "lay_bsp_c5": r4(m), "se_cl": r4(s), "z": r4(m / s) if s else None,
                "halves_[n,roi,se_cl]": hv,
                "both_halves_gt_2se": bool(all(h[1] is not None and h[2] and h[1] > 2 * h[2] for h in hv)),
                "roi_on_liability": r4(lm), "liability_se_cl": r4(ls), "by_year_[n,roi]": yrs,
                "strike": r4(won[ii].mean()), "mean_1_over_bsp": r4((1.0 / bsp[ii]).mean()),
                "median_bsp_over_ltp": r4(np.median(bsp[ii] / lp[ii])), "ltp_book_median": r4(np.median(book[ii]))}
    base = scored & (lp <= cc.VALUE_MAX_PRICE)
    shortened = has_t30 & np.isfinite(t30) & (lp < t30 - 1e-9)
    drifted = has_t30 & np.isfinite(t30) & (lp > t30 + 1e-9)
    t30_cells = {
        "all_le50_t30_book_le1.05": bsp_cell(base & has_t30 & (book <= 1.05)),
        "all_le50_t30_book_gt1.05": bsp_cell(base & has_t30 & (book > 1.05)),
        "all_le50_no_t30_book_le1.05": bsp_cell(base & ~has_t30 & (book <= 1.05)),
        "all_le50_no_t30_book_gt1.05": bsp_cell(base & ~has_t30 & (book > 1.05)),
        "rule_t30": bsp_cell(sel & has_t30),
        "rule_no_t30": bsp_cell(sel & ~has_t30),
        "rule_shortened_since_t30": bsp_cell(sel & shortened),
        "rule_drifted_since_t30": bsp_cell(sel & drifted),
        "all_le50_t30_book_gt1.05_drifted": bsp_cell(base & drifted & (book > 1.05)),
        "all_le50_t30_book_gt1.05_shortened": bsp_cell(base & shortened & (book > 1.05)),
    }
    out({"section": "t30_liquidity_split_bsp_lays", "country": code, "share_runners_with_t30": r4(has_t30.mean()),
         "cells": len(t30_cells),
         "note": "post-hoc cells found during verification; conditional on a t30 trade (the liquid subset); "
                 "an SP lay is matched only against SP back money, which these thin markets may not have",
         **t30_cells})

    # 8. executability of a lay at ltp_pre
    sens = {}
    for k in (1.0, 1.05, 1.10, 1.14, 1.20, 1.30):
        pk = np.maximum(1.01, lp[idx] * k)
        m, s, _ = cluster_se(lay_ret(pk, won[idx], 0.05), mk[idx])
        sens[f"x{k:g}"] = [r4(m), r4(s)]
    # break-even multiplier k: mean(0.95*(1-won) - won*(k*lp-1)) = 0
    lose = (~won[idx]).astype(float)
    k_be = (0.95 * lose.mean() + won[idx].mean()) / (won[idx] * lp[idx]).mean()
    tick_up = tick_above(lp[idx])
    m_tu, s_tu, _ = cluster_se(lay_ret(tick_up, won[idx], 0.05), mk[idx])
    # limit lay at ltp_pre: a backer takes it only if the market does not drift out past it (BSP <= ltp_pre);
    # when the market drifts out, backers get a better price elsewhere and the lay sits unmatched
    matched = bsp[ib] <= lp[ib]
    m_m, s_m, g_m = cluster_se(lay_ret(lp[ib][matched], won[ib][matched], 0.05), mk[ib][matched]) if matched.sum() > 1 else (float("nan"),) * 3
    m_u, s_u, _ = cluster_se(lay_ret(lp[ib][~matched], won[ib][~matched], 0.05), mk[ib][~matched]) if (~matched).sum() > 1 else (float("nan"),) * 3
    # SP lay with a fixed liability of 1 unit per bet (MARKET_ON_CLOSE lay orders carry a liability)
    fl = np.where(won[ib], -1.0, 0.95 / (bsp[ib] - 1.0))
    m_fl, s_fl, _ = cluster_se(fl, mk[ib])
    fl_h = []
    for mask in (first[ib], ~first[ib]):
        hm, hs, _ = cluster_se(fl[mask], mk[ib][mask])
        fl_h.append([r4(hm), r4(hs)])
    out({"section": "executability", "country": code,
         "lay_at_ltp_pre_times_k_c5_[roi,se_cl]": sens, "break_even_multiplier_k": r4(k_be),
         "median_bsp_over_ltp_pre": r4(np.median(ratio)),
         "one_tick_above_ltp_pre_c5": [r4(m_tu), r4(s_tu)],
         "limit_lay_matched_only_if_bsp_le_ltp_pre": {"share_matched": r4(matched.mean()), "matched_bets": int(matched.sum()),
                                                      "matched_markets": g_m, "matched_roi_c5": r4(m_m), "se_cl": r4(s_m),
                                                      "unmatched_roi_c5_if_it_had_been": r4(m_u), "unmatched_se_cl": r4(s_u)},
         "sp_lay_fixed_liability_1u_c5": {"bets": int(len(ib)), "roi_on_liability": r4(m_fl), "se_cl": r4(s_fl),
                                          "z": r4(m_fl / s_fl) if s_fl else None, "halves": fl_h}})

    # 9. date concentration
    days = pd.Series(idx).groupby(dt[idx]).size().sort_values(ascending=False)
    out({"section": "date_concentration", "country": code, "race_days": int(days.size),
         "share_top5_days": r4(days.head(5).sum() / len(idx)), "share_top20_days": r4(days.head(20).sum() / len(idx)),
         "max_bets_one_day": int(days.iloc[0]), "share_in_2020": r4((year[idx] == 2020).mean()),
         "seconds": round(time.time() - t0, 1)})
    c5 = res["lay_ltp_pre_c5"]
    return {"bets": int(len(idx)), "ltp_roi": c5["roi"], "ltp_se_cl": c5["se_cluster_market"],
            "ltp_se_day": c5["se_cluster_day"]}


def tick_above(price: np.ndarray) -> np.ndarray:
    """One Betfair tick above ``price``."""
    p = np.asarray(price, dtype=float)
    inc = np.full_like(p, 10.0)
    q = p + 1e-9
    for lo, hi, step in cc.TICKS:
        inc = np.where((q >= lo) & (q < hi), step, inc)
    return np.round(p + inc, 2)


def redo_bh(us_fix: dict | None, burn_in: int) -> None:
    """Rerun the original per-country analysis for all six countries (stdout swallowed) to
    rebuild the 192-cell lay family, then redo BH with the US value cell's SE replaced."""
    cells: list = []
    lays: list = []
    ll: list = []
    with contextlib.redirect_stdout(io.StringIO()):
        for code in cc.COUNTRIES:
            cc.analyse_country(code, None, burn_in, cells, lays, ll)
    rows = []
    for country, fam, label, src, c in lays:
        z = float("nan") if c.get("bets", 0) < 2 or not c.get("lay_se") else c["lay_roi_c5"] / c["lay_se"]
        rows.append({"country": country, "family": fam, "cell": label, "source": src, "bets": c.get("bets"),
                     "lay_roi_c5": c.get("lay_roi_c5"), "se": c.get("lay_se"), "z": z,
                     "halves": c.get("lay_halves_roi_c5"), "halves_se": c.get("lay_halves_se_c5")})
    pv = np.array([cc.one_sided_p(r["z"]) for r in rows])
    rej0 = cc.benjamini_hochberg(pv)
    tested = int((~np.isnan(pv)).sum())
    out({"section": "bh_original_lay_family", "cells": len(rows), "tested": tested, "rejections": int(rej0.sum()),
         "rejected": [f'{r["country"]} {r["family"]} {r["cell"]} {r["source"]} z={r4(r["z"])}' for r, j in zip(rows, rej0) if j],
         "bh_threshold_rank1": r4(cc.FDR / tested)})
    k_us = next((k for k, r in enumerate(rows) if r["country"] == "US" and r["family"] == "value"
                 and r["cell"].startswith("edge<=") and r["source"] == "ltp_pre"), None)
    if us_fix and k_us is not None:
        for name, se in (("market", us_fix["ltp_se_cl"]), ("day", us_fix["ltp_se_day"])):
            p3 = pv.copy()
            p3[k_us] = cc.one_sided_p(us_fix["ltp_roi"] / se)
            rej = cc.benjamini_hochberg(p3)
            out({"section": f"bh_with_us_cell_se_cluster_{name}", "us_cell_z": r4(us_fix["ltp_roi"] / se),
                 "us_cell_p": r4(float(p3[k_us])), "rejections": int(rej.sum()), "us_cell_rejected": bool(rej[k_us])})
    # the BSP twin of the claimed cell and the other value-lay cells
    twins = [r for r in rows if r["family"] == "value"]
    out({"section": "value_lay_cells_all_countries",
         "rows": [{k: (r4(v) if isinstance(v, float) else v) for k, v in r.items()} for r in twins]})
    top = sorted([r for r in rows if not math.isnan(r["z"])], key=lambda r: -r["z"])[:8]
    out({"section": "bh_top_lay_cells", "rows": [{k: (r4(v) if isinstance(v, float) else v) for k, v in r.items()} for r in top]})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default="US,FR,NZ,ZA")
    ap.add_argument("--margin", type=float, default=cc.LAY_MARGIN)
    ap.add_argument("--burn-in", type=int, default=3)
    ap.add_argument("--bh", action="store_true", help="rebuild the 192-cell lay family and redo BH")
    a = ap.parse_args()
    t0 = time.time()
    fixes = {}
    for code in [c.strip().upper() for c in a.countries.split(",") if c.strip()]:
        fixes[code] = verify_country(code, a.margin, a.burn_in)
    if a.bh:
        redo_bh(fixes.get("US"), a.burn_in)
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

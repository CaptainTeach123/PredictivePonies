"""Adversarial verification of the "cross_country" claim:

    US: back 1u at ltp_pre when p_model*ltp_pre-1 >= 0.10 and ltp_pre <= 50
    (+23.7% after 5%, n=3435, SE 0.0648, halves 0.2396/0.2305, BH survivor of 300
    back cells; the SAME bets at BSP +8.2% SE 9.6% -> "stale-LTP artefact").

    python scripts/bf_verify_cross_country_1.py [--countries US,GB,ZA,IE] [--bh]

Reuses scripts/bf_cross_country.py (imported, not edited) for the data loading,
walk-forward beta and the cell summaries, so the reproduction is exact, then adds:

  * cluster-robust SEs (by market and by race date) for the ltp_pre and BSP
    returns, overall and per half; distinct markets, bets per market, share of
    bets where the rule selects EVERY runner of the market;
  * how much of the "edge" is 1/book-1 (R^2 of edge on 1/book-1 within the
    selected bets), and the deterministic dutching return at ltp_pre
    (stake 1/ltp_pre, i.e. 1/book-1 locked in if the prices were available);
  * the same bets at BSP: ROI c0/c5/c8, per-bet and clustered SE, halves, the
    BSP book of the selected markets, the ROI with the top-1% winners removed,
    and by ltp_pre price band;
  * the subset where ltp_pre is least likely to be stale: markets where every
    runner traded 30 min out, and markets whose BSP book is within 3% of 100%;
  * the date concentration of the selected bets (top race days);
  * --bh: rerun all six countries through the original analyse_country to get
    the 300-cell back family and redo Benjamini-Hochberg with the US cell's
    per-bet SE replaced by its market-clustered SE.

Prints JSON lines only; never prints horse names.
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
    return res


def verify_country(code: str, margin: float, burn_in: int) -> None:
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        df, thin, info = cc.load_country(code, None)
    mid = df.drop_duplicates("market_id")["date"].median()
    rets = cc.Returns(df, mid)
    won = df["won_b"].to_numpy()
    lp = df["ltp_pre"].to_numpy(dtype=float)
    bsp = df["bsp"].to_numpy(dtype=float)
    book = df["book"].to_numpy(dtype=float)
    codes = df["race"].to_numpy()
    mk = df["market_id"].to_numpy()
    dt = df["date"].to_numpy()
    first = (df["date"] < mid).to_numpy()
    p_model, fits = cc.walk_forward_power(df, burn_in)
    scored = ~np.isnan(p_model)
    edge = np.where(scored & (lp <= cc.VALUE_MAX_PRICE), p_model * lp - 1.0, np.nan)
    sel = np.isfinite(edge) & (edge >= margin)
    idx = np.flatnonzero(sel)

    # 1. exact reproduction via the original cell()
    a = rets.cell(idx, "ltp_pre")
    b = rets.cell(idx, "bsp")
    out({"section": "reproduction", "country": code, "rule": f"back edge>={margin:g}, ltp_pre<={cc.VALUE_MAX_PRICE:g}",
         "markets_used": info["markets_used"], "mid_date": mid,
         "ltp_pre": {k: a.get(k) for k in ("bets", "roi_c0", "roi_c5", "roi_c8", "se", "halves_roi_c5", "halves_se_c5",
                                           "halves_bets", "positive_both_halves", "strike", "implied_prop", "median_price",
                                           "roi_c5_tick_down")},
         "bsp": {k: b.get(k) for k in ("bets", "roi_c0", "roi_c5", "roi_c8", "se", "halves_roi_c5", "halves_se_c5",
                                       "halves_bets", "positive_both_halves", "strike", "median_price")},
         "beta_wf_median": r4(np.median([f["beta"] for f in fits])) if fits else None,
         "beta_wf_min_max": ([min(f["beta"] for f in fits), max(f["beta"] for f in fits)] if fits else None)})

    # 2. structure of the selection: whole markets, bets per market, edge vs 1/book-1
    n_in_market = pd.Series(sel).groupby(codes).transform("sum").to_numpy()
    field = pd.Series(np.ones(len(df))).groupby(codes).transform("sum").to_numpy()
    whole = (n_in_market == field)[idx]
    inv_book_edge = 1.0 / book[idx] - 1.0
    e = edge[idx]
    corr = float(np.corrcoef(e, inv_book_edge)[0, 1]) if len(idx) > 2 else float("nan")
    mk_sel = pd.Series(idx).groupby(mk[idx]).size()
    out({"section": "selection_structure", "country": code, "bets": int(len(idx)),
         "distinct_markets": int(mk_sel.size), "bets_per_market_mean": r4(mk_sel.mean()),
         "bets_per_market_median": r4(mk_sel.median()), "bets_per_market_max": int(mk_sel.max()) if len(mk_sel) else None,
         "share_bets_where_whole_market_selected": r4(whole.mean()),
         "share_markets_whole_selected": r4((mk_sel.to_numpy() == pd.Series(field[idx]).groupby(mk[idx]).first().to_numpy()).mean()),
         "ltp_book_of_selected": {"median": r4(np.median(book[idx])), "q10": r4(np.quantile(book[idx], 0.1)),
                                  "q90": r4(np.quantile(book[idx], 0.9)), "share_under_0.95": r4((book[idx] < 0.95).mean()),
                                  "share_under_0.91": r4((book[idx] < 0.91).mean())},
         "edge_vs_inv_book_minus_1": {"corr": r4(corr), "r2": r4(corr * corr),
                                      "median_edge": r4(np.median(e)), "median_inv_book_minus_1": r4(np.median(inv_book_edge)),
                                      "median_abs_diff": r4(np.median(np.abs(e - inv_book_edge)))},
         "share_of_all_strict_markets_with_book_under_0.91": r4((df.drop_duplicates("market_id")["book"] < 0.91).mean())})

    # 3. clustered SEs at ltp_pre and at BSP (same bets), c5 and c0
    okb = np.isfinite(bsp[idx]) & (bsp[idx] > 1)
    ib = idx[okb]
    res = {"section": "clustered_se", "country": code}
    for c in (0.0, 0.05, 0.08):
        res[f"ltp_pre_c{int(c*100)}"] = summarise(rets.back["ltp_pre"][c][idx], mk[idx], dt[idx], first[idx])
        res[f"bsp_c{int(c*100)}"] = summarise(rets.back["bsp"][c][ib], mk[ib], dt[ib], first[ib])
    res["bsp_coverage_of_selected"] = r4(okb.mean())
    out(res)

    # 4. dutching arithmetic: stake 1/ltp_pre per runner (so the market return is 1/book-1 if
    #    every runner is backed), settled at ltp_pre and at BSP
    stake = 1.0 / lp[idx]
    ret_ltp = np.where(won[idx], stake * (lp[idx] - 1.0) * 0.95, -stake)
    ret_bsp = np.where(won[ib], (1.0 / lp[ib]) * (bsp[ib] - 1.0) * 0.95, -1.0 / lp[ib])
    out({"section": "dutch_1_over_ltp_pre_stakes_c5", "country": code,
         "roi_on_turnover_ltp_pre": r4(ret_ltp.sum() / stake.sum()),
         "roi_on_turnover_bsp": r4(ret_bsp.sum() / (1.0 / lp[ib]).sum()),
         "expected_if_whole_market_at_ltp_pre_median": r4(np.median(1.0 / book[idx] - 1.0))})

    # 5. BSP settlement in depth: BSP book of the selected markets, trimmed ROI, by band, BSP/ltp ratio
    bsp_ok_all = np.isfinite(bsp) & (bsp > 1)
    full_bsp = pd.Series(bsp_ok_all).groupby(codes).transform("min").to_numpy().astype(bool)
    bsp_book = np.full(len(df), np.nan)
    if full_bsp.any():
        bb = pd.Series(np.where(bsp_ok_all, 1.0 / bsp, 0.0)).groupby(codes).transform("sum").to_numpy()
        bsp_book = np.where(full_bsp, bb, np.nan)
    r_b5 = rets.back["bsp"][0.05][ib]
    order = np.argsort(-r_b5)
    k = max(1, int(round(0.01 * len(r_b5))))
    trimmed = np.delete(r_b5, order[:k])
    ratio = bsp[ib] / lp[ib]
    band = np.asarray(pd.cut(lp[ib], cc.PRICE_BANDS, right=False, labels=cc.BAND_LABELS).astype(str))
    by_band = {}
    for lab in cc.BAND_LABELS:
        m = band == lab
        if m.sum() >= 2:
            mm, ss, gg = cluster_se(r_b5[m], mk[ib][m])
            ml, sl, _ = cluster_se(rets.back["ltp_pre"][0.05][ib][m], mk[ib][m])
            by_band[lab] = {"bets": int(m.sum()), "markets": gg, "bsp_roi_c5": r4(mm), "se_cl": r4(ss),
                            "ltp_roi_c5": r4(ml), "ltp_se_cl": r4(sl), "median_bsp_over_ltp": r4(np.median(ratio[m]))}
    top_contrib = np.sort(r_b5)[::-1][:10]
    out({"section": "bsp_settlement", "country": code, "bets_with_bsp": int(len(ib)),
         "bsp_book_of_selected_markets": {"median": r4(np.nanmedian(bsp_book[ib])), "q10": r4(np.nanquantile(bsp_book[ib], 0.1)),
                                          "q90": r4(np.nanquantile(bsp_book[ib], 0.9)),
                                          "share_markets_full_bsp": r4(np.isfinite(bsp_book[ib]).mean())},
         "median_bsp_over_ltp_pre": r4(np.median(ratio)), "share_bsp_below_ltp_pre": r4((ratio < 1).mean()),
         "bsp_roi_c5": r4(r_b5.mean()), "bsp_roi_c5_trim_top1pct_winners": r4(trimmed.mean()), "trimmed_n": int(len(trimmed)),
         "top10_bet_returns_share_of_gross_profit": r4(top_contrib.sum() / max(r_b5[r_b5 > 0].sum(), 1e-9)),
         "strike_selected": r4(won[ib].mean()), "mean_1_over_bsp_selected": r4((1.0 / bsp[ib]).mean()),
         "mean_1_over_ltp_selected": r4((1.0 / lp[ib]).mean()),
         "by_ltp_pre_band": by_band})

    # 6. where ltp_pre is least likely to be stale: all runners traded at t30; BSP book within 3% of 1
    has_t30 = df["has_t30"].to_numpy()
    all_t30 = pd.Series(has_t30).groupby(codes).transform("min").to_numpy().astype(bool)
    near_bsp = np.isfinite(bsp_book) & (np.abs(bsp_book - 1.0) <= 0.03)
    sub = {}
    for name, mask in (("all_runners_t30", all_t30), ("bsp_book_within_3pct", near_bsp),
                       ("ltp_book_0.90_0.95", (book >= 0.90) & (book < 0.95)), ("ltp_book_under_0.85", book < 0.85)):
        ii = idx[mask[idx]]
        jj = ii[np.isfinite(bsp[ii]) & (bsp[ii] > 1)]
        if len(ii) >= 2:
            ml, sl, gl = cluster_se(rets.back["ltp_pre"][0.05][ii], mk[ii])
            entry = {"bets": int(len(ii)), "markets": gl, "ltp_roi_c5": r4(ml), "ltp_se_cl": r4(sl)}
            if len(jj) >= 2:
                mb, sb, gb = cluster_se(rets.back["bsp"][0.05][jj], mk[jj])
                entry.update({"bsp_bets": int(len(jj)), "bsp_roi_c5": r4(mb), "bsp_se_cl": r4(sb),
                              "median_bsp_over_ltp": r4(np.median(bsp[jj] / lp[jj]))})
            sub[name] = entry
        else:
            sub[name] = {"bets": int(len(ii))}
    out({"section": "fresh_price_subsets", "country": code, **sub})

    # 7b. is the LTP-vs-BSP gap market-wide (every runner shortened by about the same factor, as
    #     after a late withdrawal/reduction factor) or runner-specific (thin-market staleness)?
    #     Within-market SD of log(bsp/ltp_pre) in the selected markets vs normal markets (book within 1%).
    lr = np.where(bsp_ok_all, np.log(np.where(bsp_ok_all, bsp, 1.0) / lp), np.nan)
    sel_mk = np.zeros(len(df), dtype=bool)
    sel_mk[idx] = True
    sel_mk = pd.Series(sel_mk).groupby(codes).transform("max").to_numpy().astype(bool)
    normal_mk = np.abs(book - 1.0) <= 0.01
    disp = {}
    for name, mask in (("selected_markets", sel_mk & full_bsp), ("normal_markets_book_within_1pct", normal_mk & full_bsp)):
        s = pd.Series(lr[mask]).groupby(codes[mask])
        sd = s.std(ddof=1).dropna()
        mean_lr = s.mean()
        disp[name] = {"markets": int(sd.size), "median_within_market_sd_log_ratio": r4(sd.median()),
                      "median_market_mean_log_ratio": r4(mean_lr.median()),
                      "share_markets_sd_under_0.1": r4((sd < 0.1).mean())}
    rem = pd.read_parquet(cc.SPLIT_DIR / f"{code.lower()}_win.parquet", columns=["market_id", "field_size", "won"])
    disp["removed_runners_in_data"] = {"rows_won_nan": int(rem["won"].isna().sum()),
                                       "markets_rows_ne_field_size": int((rem.groupby("market_id").size()
                                                                          != rem.groupby("market_id")["field_size"].first()).sum())}
    out({"section": "gap_dispersion", "country": code, **disp})

    # 7. date concentration
    days = pd.Series(idx).groupby(dt[idx]).size().sort_values(ascending=False)
    out({"section": "date_concentration", "country": code, "race_days": int(days.size),
         "share_top5_days": r4(days.head(5).sum() / len(idx)), "share_top20_days": r4(days.head(20).sum() / len(idx)),
         "max_bets_one_day": int(days.iloc[0]) if len(days) else None,
         "share_in_2020": r4((pd.DatetimeIndex(dt[idx]).year == 2020).mean()),
         "seconds": round(time.time() - t0, 1)})
    return {"idx_n": int(len(idx)), "ltp_se_cl": res["ltp_pre_c5"]["se_cluster_market"],
            "ltp_roi": res["ltp_pre_c5"]["roi"], "ltp_se_day": res["ltp_pre_c5"]["se_cluster_day"]}


def redo_bh(us_fix: dict | None, burn_in: int) -> None:
    """Rerun the original per-country analysis for all six countries (stdout swallowed) to
    rebuild the 300-cell back family, then redo BH with the US value cell's SE replaced."""
    cells: list = []
    lays: list = []
    ll: list = []
    with contextlib.redirect_stdout(io.StringIO()):
        for code in cc.COUNTRIES:
            cc.analyse_country(code, None, burn_in, cells, lays, ll)
    rows = []
    for country, fam, label, src, c in cells:
        z = float("nan") if c.get("bets", 0) < 2 or not c.get("se") else c["roi_c5"] / c["se"]
        rows.append({"country": country, "family": fam, "cell": label, "source": src, "bets": c.get("bets"),
                     "roi_c5": c.get("roi_c5"), "se": c.get("se"), "z": z})
    pv = np.array([cc.one_sided_p(r["z"]) for r in rows])
    rej0 = cc.benjamini_hochberg(pv)
    tested = int((~np.isnan(pv)).sum())
    out({"section": "bh_original", "cells": len(rows), "tested": tested, "rejections": int(rej0.sum()),
         "rejected": [f'{r["country"]} {r["family"]} {r["cell"]} {r["source"]} z={r4(r["z"])}' for r, j in zip(rows, rej0) if j],
         "smallest_p": r4(float(np.nanmin(pv))), "bh_threshold_rank1": r4(cc.FDR / tested)})
    if us_fix:
        pv2 = pv.copy()
        for k, r in enumerate(rows):
            if r["country"] == "US" and r["family"] == "value" and r["cell"] == "edge>=0.1" and r["source"] == "ltp_pre":
                for name, se in (("market", us_fix["ltp_se_cl"]), ("day", us_fix["ltp_se_day"])):
                    p3 = pv.copy()
                    p3[k] = cc.one_sided_p(us_fix["ltp_roi"] / se)
                    rej = cc.benjamini_hochberg(p3)
                    out({"section": f"bh_with_us_cell_se_cluster_{name}", "us_cell_z": r4(us_fix["ltp_roi"] / se),
                         "us_cell_p": r4(float(p3[k])), "rejections": int(rej.sum()), "us_cell_rejected": bool(rej[k])})
        _ = pv2
    # top-10 cells by z, for context
    top = sorted([r for r in rows if not math.isnan(r["z"])], key=lambda r: -r["z"])[:8]
    out({"section": "bh_top_cells", "rows": [{k: (r4(v) if isinstance(v, float) else v) for k, v in r.items()} for r in top]})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default="US,GB,ZA,IE")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--burn-in", type=int, default=3)
    ap.add_argument("--bh", action="store_true", help="rebuild the 300-cell back family and redo BH")
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

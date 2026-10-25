"""Adversarial verification of the "cross_country" value-rule claim:

    US WIN strict markets (every runner traded pre-off, one winner): back 1 unit at
    ltp_pre when p_model * ltp_pre - 1 >= 0.10 and ltp_pre <= 50, p_model = walk-forward
    power recalibration of the proportional market probability.
    Claimed: n 3435, ROI +23.67% after 5% commission, SE 0.0648, halves 0.2396/0.2305,
    BH survivor of 300 back cells; the SAME bets at BSP +8.15% (SE 0.0958) = nothing;
    100% of the bets sit in markets whose LTP book is < 0.95 (stale last trades).

    python scripts/bf_verify_cross_country_2.py [--countries US,GB,ZA,IE] [--burn-in 3] [--boot 2000]

What is re-run (the data pipeline is imported unchanged from scripts/bf_cross_country.py,
so the filters are identical: dedupe, settled WIN, one winner, every runner traded):

  * the cell itself, with an INDEPENDENT maximum-likelihood fit of beta (scipy, on the
    full sample and on three walk-forward months) checked against the script's Newton fit;
  * data quality: rows vs field_size, duplicate WIN markets per event, dead heats, and
    an in-play-leak test (if ltp_pre contained in-play trades the winners' last trade
    would be SHORTER than their BSP; stale pre-off trades make it LONGER);
  * the structure of the selection: how many markets, how many of each market's runners
    are backed (the rule is a dutch of whole fields in markets whose LTP book is far
    under 100%), the LTP and BSP books of those markets, and whether a model-free rule
    "back every runner <= 50 in markets with LTP book < b" gives the same bets/ROI;
  * honest uncertainty: cluster-robust SEs by market and by race day, a race-day
    cluster bootstrap, calendar years, and the expected ROI under proportional
    calibration (0.95/book - 1 for a full dutch) vs observed;
  * executability: the same bets settled at BSP, at one tick under ltp_pre, at
    ltp_pre scaled so the LTP book is 100% (book-normalised), and as a limit order at
    ltp_pre that is matched only when the market does not move through it (BSP >=
    ltp_pre), with the matched share; ltp_pre == ltp_t30 share (nothing traded in the
    last 30 min) where ltp_t30 exists.

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
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import bf_cross_country as orig  # noqa: E402  (imported, not edited)

COMMISSIONS = (0.0, 0.05, 0.08)
VALUE_MARGIN = 0.10
VALUE_MAX_PRICE = 50.0


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) else round(float(o), 4)
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
    return None if math.isnan(x) else round(x, 4)


def mean_se(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan")), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def cluster_se(r: np.ndarray, groups: np.ndarray) -> float:
    """SE of the mean of r with arbitrary within-cluster correlation (sandwich)."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n < 2:
        return float("nan")
    codes, _ = pd.factorize(groups)
    dev = r - r.mean()
    s = np.bincount(codes, weights=dev)
    g = len(s)
    if g < 2:
        return float("nan")
    return float(math.sqrt((s * s).sum() * g / (g - 1)) / n)


def boot_ci(r: np.ndarray, groups: np.ndarray, reps: int, seed: int = 7) -> tuple[float, float, float]:
    """Cluster bootstrap of the mean (percentile 2.5/97.5 and bootstrap SE)."""
    codes, _ = pd.factorize(groups)
    g = codes.max() + 1
    sums = np.bincount(codes, weights=r, minlength=g)
    cnts = np.bincount(codes, minlength=g).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, g, size=(reps, g))
    w = np.zeros((reps, g))
    for k in range(reps):
        w[k] = np.bincount(draws[k], minlength=g)
    rois = (w @ sums) / (w @ cnts)
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5)), float(rois.std(ddof=1))


def back5(price, won):
    return orig.net_returns(price, won, 0.05)


def cell_stats(r: np.ndarray, first: np.ndarray, market: np.ndarray, day: np.ndarray, boot: int) -> dict:
    n = len(r)
    if n == 0:
        return {"bets": 0}
    m, se = mean_se(r)
    res = {"bets": int(n), "roi_c5": r4(m), "se_iid": r4(se),
           "se_cluster_market": r4(cluster_se(r, market)), "se_cluster_day": r4(cluster_se(r, day)),
           "markets": int(len(np.unique(market))), "days": int(len(np.unique(day)))}
    if boot and len(np.unique(day)) > 5:
        lo, hi, bse = boot_ci(r, day, boot)
        res["boot_day_ci95"] = [r4(lo), r4(hi)]
        res["boot_day_se"] = r4(bse)
    halves = []
    for mask in (first, ~first):
        h = r[mask]
        mh, sh = mean_se(h)
        halves.append({"bets": int(mask.sum()), "roi_c5": r4(mh), "se_iid": r4(sh),
                       "se_cluster_market": r4(cluster_se(h, market[mask])) if mask.sum() > 1 else None,
                       "se_cluster_day": r4(cluster_se(h, day[mask])) if mask.sum() > 1 else None})
    res["halves"] = halves
    return res


def nll_beta(beta: float, x: np.ndarray, win: np.ndarray, codes: np.ndarray, n_races: int) -> float:
    p = orig.power_probs(x, beta, codes, n_races)
    return float(-np.log(np.clip(p[win], 1e-300, 1.0)).sum())


def independent_beta(x, win, codes, n_races) -> float:
    res = minimize_scalar(nll_beta, bounds=(0.2, 3.0), args=(x, win, codes, n_races), method="bounded",
                          options={"xatol": 1e-7})
    return float(res.x)


def analyse(code: str, burn_in: int, boot: int) -> dict:
    t0 = time.time()
    df, thin, info = orig.load_country(code, None)
    out({"section": "data", "country": code, "markets_used": info["markets_used"], "runners_used": info["runners_used"],
         "dead_heat_markets": info["dead_heat_markets"], "thin_markets": info["thin_markets"]})

    # ---- data quality on the strict table
    g = df.groupby("market_id")
    rows = g.size()
    fs = g["field_size"].first()
    mk = df.drop_duplicates("market_id")
    ev = pd.read_parquet(orig.SPLIT_DIR / f"{code.lower()}_win.parquet", columns=["market_id", "event_id", "start"])
    ev = ev.drop_duplicates("market_id")
    ev = ev[ev["market_id"].isin(mk["market_id"])]
    dup_events = ev.groupby(["event_id", "start"])["market_id"].nunique()
    dq = {"share_markets_rows_eq_field_size": r4((rows == fs).mean()),
          "markets_rows_lt_field_size": int((rows < fs).sum()), "markets_rows_gt_field_size": int((rows > fs).sum()),
          "events_with_2plus_win_markets": int((dup_events > 1).sum()),
          "winners_per_market_all_one": bool((g["won"].sum() == 1).all()),
          "note": "removed runners are dropped by the parser without a flag; field_size counts non-removed runners only"}

    # ---- reproduce the walk-forward beta and the value rule
    p_model, fits = orig.walk_forward_power(df, burn_in)
    lp = df["ltp_pre"].to_numpy(dtype=float)
    bsp = df["bsp"].to_numpy(dtype=float)
    won = df["won_b"].to_numpy()
    codes = df["race"].to_numpy()
    n_races = int(codes.max()) + 1
    book = df["book"].to_numpy(dtype=float)
    p_prop = df["p_prop"].to_numpy(dtype=float)
    x = np.log(p_prop)
    months = df["month"].to_numpy()
    market = df["market_id"].to_numpy()
    day = df["date"].to_numpy()
    mid = mk["date"].median()
    first = (df["date"] < mid).to_numpy()
    year = df["date"].dt.year.to_numpy()

    # independent beta: full sample and three walk-forward months
    b_full_script, se_full = orig.fit_beta(x, won, codes, n_races)
    b_full_ind = independent_beta(x, won, codes, n_races)
    uniq = np.array(sorted(pd.unique(months)))
    checks = []
    for i in (burn_in, len(uniq) // 2, len(uniq) - 1):
        m = uniq[i]
        train = months < m
        b_s, _ = orig.fit_beta(x[train], won[train], codes[train], n_races)
        b_i = independent_beta(x[train], won[train], codes[train], n_races)
        checks.append({"month": str(m), "beta_script": r4(b_s), "beta_independent": r4(b_i)})
    beta_rep = {"beta_full_script": r4(b_full_script), "beta_full_independent": r4(b_full_ind), "se": r4(se_full),
                "walk_forward_checks": checks, "wf_beta_min_max": [min(f["beta"] for f in fits), max(f["beta"] for f in fits)],
                "train_is_strictly_earlier_months": True}

    edge = np.where(~np.isnan(p_model) & (lp <= VALUE_MAX_PRICE), p_model * lp - 1.0, np.nan)
    sel = np.flatnonzero(~np.isnan(edge) & (edge >= VALUE_MARGIN))
    scored = ~np.isnan(edge)
    r5 = back5(lp, won)
    okb = np.isfinite(bsp) & (bsp > 1)
    r5b = np.where(okb, back5(np.where(okb, bsp, 2.0), won), np.nan)

    cell = {"bets": int(len(sel))}
    for c in COMMISSIONS:
        cell[f"roi_c{int(c * 100)}"] = r4(orig.net_returns(lp[sel], won[sel], c).mean())
    cell.update(cell_stats(r5[sel], first[sel], market[sel], day[sel], boot))
    cell["strike"] = r4(won[sel].mean())
    cell["median_price"] = r4(np.median(lp[sel]))
    selb = sel[okb[sel]]
    bsp_cell = cell_stats(r5b[selb], first[selb], market[selb], day[selb], boot)
    for c in COMMISSIONS:
        bsp_cell[f"roi_c{int(c * 100)}"] = r4(orig.net_returns(bsp[selb], won[selb], c).mean())
    out({"section": "reproduction", "country": code, "mid_date": mid, "claimed": {"bets": 3435, "roi_c5": 0.2367, "se": 0.0648,
         "halves": [0.2396, 0.2305], "bsp_roi_c5": 0.0815, "bsp_se": 0.0958} if code == "US" else None,
         "ltp_pre": cell, "same_bets_at_bsp": bsp_cell, "beta": beta_rep, "data_quality": dq})

    # ---- year split
    years = []
    for y in sorted(np.unique(year[sel])):
        m_ = sel[year[sel] == y]
        mb = m_[okb[m_]]
        a, sa = mean_se(r5[m_])
        b, sb = mean_se(r5b[mb])
        years.append({"year": int(y), "bets": int(len(m_)), "markets": int(len(np.unique(market[m_]))),
                      "roi_c5_ltp_pre": r4(a), "se": r4(sa), "roi_c5_bsp": r4(b), "se_bsp": r4(sb)})
    out({"section": "years", "country": code, "rows": years})

    # ---- structure of the selection: a dutch of whole fields in low-book markets
    sel_df = pd.DataFrame({"market": market[sel], "lp": lp[sel], "won": won[sel], "book": book[sel], "edge": edge[sel]})
    per_mk = sel_df.groupby("market").agg(bets=("lp", "size"), book=("book", "first"), wins=("won", "sum"))
    eligible = pd.Series(lp <= VALUE_MAX_PRICE).groupby(market).sum()  # runners <= 50 per market
    per_mk["eligible"] = eligible.reindex(per_mk.index).to_numpy()
    per_mk["field"] = pd.Series(np.ones(len(df))).groupby(market).sum().reindex(per_mk.index).to_numpy()
    per_mk["full_dutch"] = per_mk["bets"] == per_mk["eligible"]
    book_sel = per_mk["book"].to_numpy()
    inv_edge = 1.0 / book[sel] - 1.0
    bsp_book = pd.Series(np.where(okb, 1.0 / np.where(okb, bsp, 1.0), np.nan)).groupby(market).sum(min_count=1)
    all_bsp = pd.Series(okb).groupby(market).min()
    bsp_book_sel = bsp_book.reindex(per_mk.index)[all_bsp.reindex(per_mk.index).fillna(False).astype(bool)]
    structure = {
        "bets": int(len(sel)), "markets": int(len(per_mk)), "bets_per_market_mean": r4(per_mk["bets"].mean()),
        "share_markets_full_dutch_of_runners_le_50": r4(per_mk["full_dutch"].mean()),
        "share_bets_in_full_dutch_markets": r4(per_mk.loc[per_mk["full_dutch"], "bets"].sum() / len(sel)),
        "share_eligible_runners_backed_in_selected_markets": r4(per_mk["bets"].sum() / per_mk["eligible"].sum()),
        "share_markets_where_a_backed_runner_won": r4((per_mk["wins"] > 0).mean()),
        "ltp_book_of_selected_markets": {"median": r4(np.median(book_sel)), "q10": r4(np.quantile(book_sel, 0.1)),
                                         "q90": r4(np.quantile(book_sel, 0.9)), "max": r4(book_sel.max())},
        "share_bets_book_lt_0.95": r4((book[sel] < 0.95).mean()),
        "bsp_book_of_same_markets_median": r4(bsp_book_sel.median()) if len(bsp_book_sel) else None,
        "corr_edge_vs_1_over_book_minus_1": r4(np.corrcoef(edge[sel], inv_edge)[0, 1]) if len(sel) > 2 else None,
        "share_selected_with_1_over_book_minus_1_ge_0.1": r4((inv_edge >= 0.1).mean()),
        "share_of_all_scored_bets_with_1_over_book_minus_1_ge_0.1_that_are_selected":
            r4(len(sel) / max(1, int((scored & (1.0 / book - 1.0 >= 0.1)).sum()))),
        "share_strict_markets_with_book_lt_0.95": r4((mk["book"] < 0.95).mean()),
        "share_strict_markets_with_book_lt_0.91": r4((mk["book"] < 0.91).mean()),
    }
    out({"section": "selection_structure", "country": code, **structure})

    # ---- model-free equivalent: every runner <= 50 in markets with LTP book < b (same months as the rule)
    free_rows = []
    for b_thr in (0.95, 0.92, 0.91, 0.90, 0.85):
        idx = np.flatnonzero(scored & (book < b_thr))
        if len(idx) == 0:
            free_rows.append({"book_lt": b_thr, "bets": 0})
            continue
        overlap = len(np.intersect1d(idx, sel))
        a = cell_stats(r5[idx], first[idx], market[idx], day[idx], 0)
        ib = idx[okb[idx]]
        b = cell_stats(r5b[ib], first[ib], market[ib], day[ib], 0)
        exp_prop = (0.95 / book[idx] - 1.0 + 0.05 * p_prop[idx]).mean()
        free_rows.append({"book_lt": b_thr, "bets": a["bets"], "markets": a["markets"], "overlap_with_rule": int(overlap),
                          "roi_c5_ltp_pre": a["roi_c5"], "se_iid": a["se_iid"], "se_cluster_market": a["se_cluster_market"],
                          "halves": [h["roi_c5"] for h in a["halves"]],
                          "expected_roi_c5_if_prop_calibrated": r4(exp_prop),
                          "roi_c5_bsp": b.get("roi_c5"), "se_bsp_iid": b.get("se_iid"), "se_bsp_cluster_market": b.get("se_cluster_market"),
                          "halves_bsp": [h["roi_c5"] for h in b["halves"]] if b.get("halves") else None})
    out({"section": "model_free_low_book_dutch", "country": code,
         "note": "no model: back 1u on every runner <= 50 in markets whose sum(1/ltp_pre) < threshold, scored months only",
         "rows": free_rows})

    # ---- expected vs observed, and the in-play-leak test
    exp_prop_sel = (0.95 / book[sel] - 1.0 + 0.05 * p_prop[sel]).mean()
    p_bsp = np.full(len(df), np.nan)
    p_bsp[okb] = 1.0 / bsp[okb]
    p_bsp = p_bsp / pd.Series(p_bsp).groupby(market).transform("sum").to_numpy()
    e_bsp = 0.95 * p_bsp[sel] * lp[sel] - 1.0 + 0.05 * p_bsp[sel]
    ratio = bsp / lp
    r_sel = ratio[sel]
    w_sel = won[sel]
    ok_sel = okb[sel]
    all_ok = okb
    leak = {
        "expected_roi_c5_at_ltp_pre_if_prop_calibrated": r4(exp_prop_sel),
        "expected_roi_c5_at_ltp_pre_if_bsp_is_truth": r4(np.nanmean(e_bsp)),
        "observed_roi_c5_at_ltp_pre": cell["roi_c5"],
        "median_bsp_over_ltp_pre_selected": r4(np.nanmedian(r_sel[ok_sel])),
        "median_bsp_over_ltp_pre_selected_winners": r4(np.nanmedian(r_sel[ok_sel & w_sel])),
        "median_bsp_over_ltp_pre_selected_losers": r4(np.nanmedian(r_sel[ok_sel & ~w_sel])),
        "share_selected_winners_with_bsp_gt_ltp_pre": r4(np.mean(r_sel[ok_sel & w_sel] > 1.0)),
        "share_selected_losers_with_bsp_gt_ltp_pre": r4(np.mean(r_sel[ok_sel & ~w_sel] > 1.0)),
        "median_bsp_over_ltp_pre_all_strict_winners": r4(np.nanmedian(ratio[all_ok & won])),
        "median_bsp_over_ltp_pre_all_strict_losers": r4(np.nanmedian(ratio[all_ok & ~won])),
        "reading": "in-play leakage into ltp_pre would make winners' ltp_pre SHORTER than BSP (ratio > 1); "
                   "stale pre-off trades make it LONGER (ratio < 1)",
    }
    # withdrawal fingerprint: removed runners are dropped silently, so a late scratch leaves the
    # remaining last trades too long (book < 1). A gap in the saddlecloth numbers (max program >
    # field_size) marks a market with a removed runner (at any time); within-market dispersion of
    # log(BSP/ltp_pre) is low when every price shortened by the same factor (one removal) and high
    # when the last trades are individually stale.
    prog = pd.read_parquet(orig.SPLIT_DIR / f"{code.lower()}_win.parquet", columns=["market_id", "selection_id", "program"])
    prog = prog.drop_duplicates(["market_id", "selection_id"])
    prog["program"] = pd.to_numeric(prog["program"], errors="coerce")
    pmax = prog.groupby("market_id")["program"].max()
    gap = (pmax.reindex(mk["market_id"]).to_numpy() > mk["field_size"].to_numpy())
    gap_s = pd.Series(gap, index=mk["market_id"].to_numpy())
    lr = np.where(okb, np.log(np.where(okb, bsp, 1.0) / lp), np.nan)
    disp = pd.Series(lr).groupby(market).std()
    sel_mk = per_mk.index.to_numpy()
    normal_mk = mk.loc[(mk["book"] >= 0.98) & (mk["book"] <= 1.02), "market_id"].to_numpy()
    leak["withdrawal_fingerprint"] = {
        "share_selected_markets_with_saddlecloth_gap": r4(gap_s.reindex(sel_mk).mean()),
        "share_normal_book_markets_with_saddlecloth_gap": r4(gap_s.reindex(normal_mk).mean()),
        "share_all_strict_markets_with_saddlecloth_gap": r4(gap_s.mean()),
        "within_market_sd_log_bsp_over_ltp_pre_selected_median": r4(disp.reindex(sel_mk).median()),
        "within_market_sd_log_bsp_over_ltp_pre_normal_book_median": r4(disp.reindex(normal_mk).median()),
    }
    has_t30 = df["has_t30"].to_numpy()
    t30 = df["ltp_t30"].to_numpy(dtype=float)
    if has_t30[sel].any():
        same = np.isclose(lp, t30)
        leak["share_selected_with_t30"] = r4(has_t30[sel].mean())
        leak["share_ltp_pre_eq_ltp_t30_selected"] = r4(same[sel][has_t30[sel]].mean())
        leak["share_ltp_pre_eq_ltp_t30_all_scored"] = r4(same[scored & has_t30].mean())
    out({"section": "expected_vs_observed_and_leak_test", "country": code, **leak})

    # ---- executability: alternative settlements of the same bets
    tick = orig.tick_down(lp)
    norm_price = lp * book  # scale so the LTP book is 100%
    matched = okb & (bsp >= lp)  # a limit order at ltp_pre is matched only if the market does not move through it
    alt = {}
    for name, price, mask in (("ltp_pre", lp, np.ones(len(df), dtype=bool)), ("tick_down", tick, np.ones(len(df), dtype=bool)),
                              ("book_normalised_ltp_pre", norm_price, np.ones(len(df), dtype=bool)), ("bsp", bsp, okb),
                              ("limit_at_ltp_pre_matched_only_if_bsp_ge_ltp_pre", lp, matched)):
        idx = sel[mask[sel]]
        if len(idx) == 0:
            alt[name] = {"bets": 0}
            continue
        rr = back5(price[idx], won[idx])
        m, se = mean_se(rr)
        alt[name] = {"bets": int(len(idx)), "roi_c5": r4(m), "se_iid": r4(se), "se_cluster_market": r4(cluster_se(rr, market[idx])),
                     "halves": [r4(rr[first[idx]].mean()) if first[idx].any() else None,
                                r4(rr[~first[idx]].mean()) if (~first[idx]).any() else None]}
    alt["limit_at_ltp_pre_matched_only_if_bsp_ge_ltp_pre"]["share_matched"] = r4(matched[sel].mean())
    alt["limit_at_ltp_pre_matched_only_if_bsp_ge_ltp_pre"]["roi_c5_counting_unmatched_as_zero_over_all_bets"] = r4(
        back5(lp[sel], won[sel])[matched[sel]].sum() / len(sel))
    out({"section": "alternative_settlements", "country": code, **alt})

    # ---- same bets, BSP, per market dutch return (market-level unit) - how many markets would need to be bet
    mk_ret = sel_df.groupby("market").apply(lambda d: back5(d["lp"].to_numpy(), d["won"].to_numpy()).sum(), include_groups=False)
    out({"section": "market_level", "country": code, "markets": int(len(mk_ret)),
         "mean_return_per_market_ltp_pre": r4(mk_ret.mean()), "sd_per_market": r4(mk_ret.std(ddof=1)),
         "share_markets_positive": r4((mk_ret > 0).mean()), "seconds": round(time.time() - t0, 1)})
    return {"country": code, "bets": cell["bets"], "roi_c5": cell["roi_c5"], "se_iid": cell["se_iid"],
            "se_cluster_market": cell["se_cluster_market"], "se_cluster_day": cell["se_cluster_day"],
            "bsp_roi_c5": bsp_cell.get("roi_c5"), "bsp_se_iid": bsp_cell.get("se_iid"),
            "bsp_se_cluster_market": bsp_cell.get("se_cluster_market"), "bsp_se_cluster_day": bsp_cell.get("se_cluster_day"),
            "bsp_halves": [h["roi_c5"] for h in bsp_cell.get("halves", [])],
            "share_full_dutch": structure["share_markets_full_dutch_of_runners_le_50"],
            "expected_if_prop": leak["expected_roi_c5_at_ltp_pre_if_prop_calibrated"],
            "book_normalised_roi_c5": alt["book_normalised_ltp_pre"].get("roi_c5"),
            "limit_matched_share": alt["limit_at_ltp_pre_matched_only_if_bsp_ge_ltp_pre"].get("share_matched"),
            "limit_roi_c5_matched": alt["limit_at_ltp_pre_matched_only_if_bsp_ge_ltp_pre"].get("roi_c5")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default="US,GB,ZA,IE")
    ap.add_argument("--burn-in", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    rows = []
    for code in [c.strip().upper() for c in a.countries.split(",") if c.strip()]:
        rows.append(analyse(code, a.burn_in, a.boot))
    out({"section": "summary", "rows": rows, "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

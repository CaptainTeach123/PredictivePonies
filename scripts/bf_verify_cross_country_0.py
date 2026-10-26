"""Adversarial verification of the "cross_country" LAY value-rule claim:

    US WIN strict markets (every runner traded pre-off, one winner): lay 1 unit of
    backer's stake at ltp_pre when p_model * ltp_pre - 1 <= -0.10 and ltp_pre <= 50,
    p_model = walk-forward power recalibration of the proportional market probability.
    Claimed: n 21283, +21.59% of stake after 5% commission, SE 0.0239 (z = 9), halves
    0.1989/0.2361, BH survivor of 192 lay cells; 95% of the bets sit in markets whose
    LTP book is > 1.05 (median 1.145), median BSP/ltp_pre 1.14, and the SAME lays
    settled at BSP return +3.6% (SE 3.8%, n 21262) = nothing, i.e. the ltp_pre result
    is a stale-last-trade artefact and not executable.

    python scripts/bf_verify_cross_country_0.py [--countries US] [--end YYYY-MM-DD] [--burn-in 3]

The data pipeline is imported unchanged from scripts/bf_cross_country.py (dedupe,
settled WIN, one winner, every runner traded, walk-forward beta), so the reproduction
is exact.  Then, under the leakage / data-quality lens:

  * data quality of the selected markets: parallel WIN markets per event, rows vs
    field_size, dead heats, 1000-ceiling prices, BSP coverage, venue and year
    concentration (the original's harness regex finds no US harness; the venues do);
  * the structure of the selection: how many markets, whether whole fields are laid,
    the LTP book of the selected markets, and a MODEL-FREE rule "lay every runner
    <= 50 in markets whose sum(1/ltp_pre) >= 1/0.9" (same bets?);
  * expected lay ROI at ltp_pre if the proportional (stale-book) probabilities were
    right, if BSP were the truth, and if p_model were right, vs observed;
  * in-play-leak test: a leak would make WINNERS' ltp_pre much shorter than their BSP
    (BSP/ltp_pre >> 1 for winners vs losers); stale-short longshot trades do the
    opposite (losers drift out more than winners);
  * honest uncertainty: SEs clustered by market and by race day, both overall and
    per half, for the ltp_pre lay and the BSP lay;
  * executability: the same lays at BSP (backer's stake 1, and fixed liability 1 as an
    SP lay really works), by BSP band (is the +3.6% the known US 50+ BSP over-betting?),
    a limit lay at ltp_pre that is matched only when the price at the off is at or
    below it (BSP <= ltp_pre), lay at max(ltp_pre, BSP), lay at the book-normalised
    price ltp_pre * book, lay at ltp_t30 where it exists, and the baselines (lay every
    scored runner <= 50 at BSP; lay every runner at ltp_pre in markets with a book
    within 3% of 100%);
  * calendar years.

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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import bf_cross_country as orig  # noqa: E402  (imported, not edited)

CLAIM = {"bets": 21283, "lay_roi_c5": 0.2159, "se": 0.0239, "halves": [0.1989, 0.2361],
         "bsp_lay_roi_c5": 0.0358, "bsp_se": 0.0381, "bsp_bets": 21262,
         "share_book_over_1.05": 0.952, "median_bsp_over_ltp_pre": 1.14}
BSP_BANDS = [1.0, 5.0, 10.0, 25.0, 50.0, 1e9]
BSP_LABELS = ["1-5", "5-10", "10-25", "25-50", "50+"]
HARNESS_RE = r"\b(?:Pace|Trot|Pacing|Trotting|Harness)\b"


def out(obj) -> None:
    print(json.dumps(obj, default=orig._json_default), flush=True)


r4 = orig.r4


def cluster_se(r: np.ndarray, groups: np.ndarray) -> float:
    """Cluster-robust SE of the mean of r with clusters ``groups`` (G/(G-1) small-sample factor)."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n < 2:
        return float("nan")
    codes, uniq = pd.factorize(groups)
    g = len(uniq)
    if g < 2:
        return float("nan")
    sums = np.bincount(codes, weights=r - r.mean(), minlength=g)
    return math.sqrt(float((sums ** 2).sum()) * g / (g - 1)) / n


def summarise(r: np.ndarray, first: np.ndarray, mk: np.ndarray, day: np.ndarray, label: str) -> dict:
    r = np.asarray(r, dtype=float)
    n = len(r)
    res = {"label": label, "bets": int(n)}
    if n < 2:
        res["roi"] = r4(r.mean()) if n else None
        return res
    m = float(r.mean())
    se = float(r.std(ddof=1) / math.sqrt(n))
    se_m = cluster_se(r, mk)
    se_d = cluster_se(r, day)
    res.update({"roi": r4(m), "se_iid": r4(se), "se_market": r4(se_m), "se_day": r4(se_d),
                "z_iid": r4(m / se) if se else None, "z_day": r4(m / se_d) if se_d and not math.isnan(se_d) else None})
    halves = []
    for mask in (first, ~first):
        h = r[mask]
        if len(h) < 2:
            halves.append({"bets": int(len(h)), "roi": r4(h.mean()) if len(h) else None})
            continue
        hm = float(h.mean())
        hs = float(h.std(ddof=1) / math.sqrt(len(h)))
        hd = cluster_se(h, day[mask])
        halves.append({"bets": int(len(h)), "roi": r4(hm), "se_iid": r4(hs), "se_day": r4(hd),
                       "z_iid": r4(hm / hs) if hs else None, "z_day": r4(hm / hd) if hd and not math.isnan(hd) else None})
    res["halves"] = halves
    res["positive_both_halves_iid"] = bool(all(h.get("z_iid") is not None and h["z_iid"] > 2 for h in halves))
    res["positive_both_halves_day"] = bool(all(h.get("z_day") is not None and h["z_day"] > 2 for h in halves))
    return res


def lay_liab_returns(price: np.ndarray, won: np.ndarray, c: float) -> np.ndarray:
    """Lay with 1 unit of LIABILITY (backer's stake 1/(price-1)): +(1-c)/(price-1) on a loser, -1 on a winner."""
    price = np.asarray(price, dtype=float)
    return np.where(won, -1.0, (1.0 - c) / (price - 1.0))


def analyse(code: str, end: str | None, burn_in: int) -> dict:
    t0 = time.time()
    df, thin, info = orig.load_country(code, end)
    out({"section": "data", "country": code, **{k: info[k] for k in ("markets_settled", "runners", "markets_used", "runners_used",
                                                                       "dead_heat_markets", "thin_markets", "bsp_coverage_used",
                                                                       "t30_runner_coverage_used", "share_markets_harness")}})
    p_model, fits = orig.walk_forward_power(df, burn_in)
    lp = df["ltp_pre"].to_numpy(dtype=float)
    bsp = df["bsp"].to_numpy(dtype=float)
    okb = np.isfinite(bsp) & (bsp > 1)
    won = df["won_b"].to_numpy()
    book = df["book"].to_numpy(dtype=float)
    p_prop = df["p_prop"].to_numpy(dtype=float)
    market = df["market_id"].to_numpy()
    date = df["date"].to_numpy()
    t30 = df["ltp_t30"].to_numpy(dtype=float)
    has_t30 = df["has_t30"].to_numpy()
    mid = df.drop_duplicates("market_id")["date"].median()
    first = (df["date"] < mid).to_numpy()
    scored = ~np.isnan(p_model) & (lp <= orig.VALUE_MAX_PRICE)
    edge = np.where(scored, p_model * lp - 1.0, np.nan)
    sel = scored & (edge <= -orig.LAY_MARGIN)
    n_sel = int(sel.sum())
    out({"section": "walk_forward", "country": code, "months_fitted": len(fits), "beta_first": fits[0]["beta"] if fits else None,
         "beta_last": fits[-1]["beta"] if fits else None, "beta_median": r4(np.median([f["beta"] for f in fits])) if fits else None,
         "scored_runners_le_50": int(scored.sum()), "selected": n_sel, "mid_date": mid})

    # ---------------------------------------------------------------- 1. reproduction
    l5 = orig.lay_returns(lp, won, 0.05)
    rep = {"bets": n_sel}
    for c in orig.COMMISSIONS:
        rep[f"lay_roi_c{int(c * 100)}"] = r4(orig.lay_returns(lp[sel], won[sel], c).mean())
    r = l5[sel]
    rep["se"] = r4(r.std(ddof=1) / math.sqrt(n_sel))
    rep["halves"] = [r4(r[first[sel]].mean()), r4(r[~first[sel]].mean())]
    sb = sel & okb
    rb = orig.lay_returns(bsp[sb], won[sb], 0.05)
    rep["bsp_bets"] = int(sb.sum())
    rep["bsp_lay_roi_c5"] = r4(rb.mean())
    rep["bsp_se"] = r4(rb.std(ddof=1) / math.sqrt(len(rb)))
    rep["share_book_over_1.05"] = r4((book[sel] > 1.05).mean())
    rep["median_book"] = r4(np.median(book[sel]))
    rep["median_bsp_over_ltp_pre"] = r4(np.median(bsp[sb] / lp[sb]))
    rep["claim"] = CLAIM
    rep["matches_claim"] = bool(abs(rep["lay_roi_c5"] - CLAIM["lay_roi_c5"]) < 0.002 and rep["bets"] == CLAIM["bets"]
                                and abs(rep["bsp_lay_roi_c5"] - CLAIM["bsp_lay_roi_c5"]) < 0.002)
    out({"section": "reproduction", "country": code, **rep})

    # ---------------------------------------------------------------- 2. data quality of the selected markets
    extra = pd.read_parquet(orig.SPLIT_DIR / f"{code.lower()}_win.parquet",
                            columns=["market_id", "selection_id", "event_id", "venue", "market_name", "start", "market_type", "won"])
    extra = extra.drop_duplicates(["market_id", "selection_id"])
    extra = extra[(extra["market_type"] == "WIN") & extra["won"].notna()]
    emk = extra.drop_duplicates("market_id").set_index("market_id")
    per_event_start = emk.groupby(["event_id", "start"]).size()
    dup_keys = set(per_event_start[per_event_start > 1].index)
    emk["parallel_win_market"] = [(e, s) in dup_keys for e, s in zip(emk["event_id"], emk["start"])]
    sel_mk = pd.unique(market[sel])
    all_mk = pd.unique(market)
    rows_per_mk = df.groupby("market_id").size()
    fs = df.drop_duplicates("market_id").set_index("market_id")["field_size"]
    sel_rows = df[np.isin(market, sel_mk)]
    yrs = pd.DatetimeIndex(date[sel]).year
    ven = emk["venue"].reindex(sel_mk)
    harness_name = emk["market_name"].fillna("").str.contains(HARNESS_RE)
    top_ven = ven.value_counts().head(12)
    dq = {"selected_markets": int(len(sel_mk)), "all_strict_markets": int(len(all_mk)),
          "share_selected_markets_with_parallel_win_market_same_event_start": r4(emk["parallel_win_market"].reindex(sel_mk).mean()),
          "share_all_markets_with_parallel_win_market": r4(emk["parallel_win_market"].mean()),
          "share_selected_markets_rows_eq_field_size": r4((rows_per_mk.reindex(sel_mk) == fs.reindex(sel_mk)).mean()),
          "share_all_markets_rows_eq_field_size": r4((rows_per_mk.reindex(all_mk) == fs.reindex(all_mk)).mean()),
          "dead_heat_markets_dropped": info["dead_heat_markets"],
          "share_runners_at_1000_in_selected_markets": r4((sel_rows["ltp_pre"] >= 1000).mean()),
          "share_selected_bets_without_bsp": r4((~okb[sel]).mean()),
          "share_selected_markets_any_t30": r4(sel_rows.groupby("market_id")["has_t30"].max().mean()),
          "share_all_markets_any_t30": r4(df.groupby("market_id")["has_t30"].max().mean()),
          "selected_bets_by_year": {str(k): int(v) for k, v in pd.Series(yrs).value_counts().sort_index().items()},
          "all_strict_markets_by_year": {str(k): int(v) for k, v in pd.Series(pd.DatetimeIndex(df.drop_duplicates("market_id")["date"]).year).value_counts().sort_index().items()},
          "share_selected_markets_harness_by_name": r4(harness_name.reindex(sel_mk).mean()),
          "top_venues_selected_markets": {str(k): int(v) for k, v in top_ven.items()},
          "share_selected_markets_in_top12_venues": r4(top_ven.sum() / len(sel_mk)),
          "distinct_venues_selected": int(ven.nunique()), "distinct_venues_all": int(emk["venue"].nunique()),
          "selected_market_median_field": r4(fs.reindex(sel_mk).median()), "all_market_median_field": r4(fs.median())}
    out({"section": "data_quality", "country": code, **dq})

    # ---------------------------------------------------------------- 3. selection structure and a model-free twin
    n_le50 = pd.Series(lp <= orig.VALUE_MAX_PRICE).groupby(market).transform("sum").to_numpy()
    n_sel_mk = pd.Series(sel.astype(int)).groupby(market).transform("sum").to_numpy()
    whole = sel & (n_sel_mk == n_le50)
    bk_sel = book[sel]
    q = np.quantile(bk_sel, [0.1, 0.25, 0.5, 0.75, 0.9])
    # how much of the edge is just 1/book - 1 ?
    x = 1.0 / book[scored] - 1.0
    y = edge[scored]
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    struct = {"selected_bets": n_sel, "selected_markets": int(len(sel_mk)),
              "bets_per_selected_market_mean": r4(n_sel / len(sel_mk)),
              "share_bets_where_whole_field_le_50_is_laid": r4(whole[sel].mean()),
              "book_of_selected_bets": {"q10": r4(q[0]), "q25": r4(q[1]), "median": r4(q[2]), "q75": r4(q[3]), "q90": r4(q[4]),
                                        "share_gt_1.05": r4((bk_sel > 1.05).mean()), "share_ge_1.111": r4((bk_sel >= 1 / 0.9).mean()),
                                        "share_gt_1.20": r4((bk_sel > 1.2).mean())},
              "r2_of_edge_on_1_over_book_minus_1_scored": r4(r2)}
    free_rows = []
    for thr in (1.05, 1.0 / 0.9, 1.15):
        fsel = scored & (book >= thr)
        if fsel.sum() < 2:
            continue
        inter = int((fsel & sel).sum())
        a = summarise(l5[fsel], first[fsel], market[fsel], date[fsel], f"model-free: lay all <=50 in LTP book >= {thr:.3f}")
        fb = fsel & okb
        b = summarise(orig.lay_returns(bsp[fb], won[fb], 0.05), first[fb], market[fb], date[fb], "same at BSP")
        free_rows.append({"threshold": r4(thr), "bets": a["bets"], "lay_roi_c5": a["roi"], "se_iid": a["se_iid"], "se_day": a["se_day"],
                          "halves": [h.get("roi") for h in a["halves"]],
                          "overlap_with_selected": inter, "jaccard": r4(inter / (fsel.sum() + n_sel - inter)),
                          "bsp_lay_roi_c5": b.get("roi"), "bsp_se_day": b.get("se_day")})
    struct["model_free_rows"] = free_rows
    out({"section": "selection_structure", "country": code, **struct})

    # ---------------------------------------------------------------- 4. expected vs observed, and the in-play-leak test
    p_bsp = np.full(len(df), np.nan)
    p_bsp[okb] = 1.0 / bsp[okb]
    bsp_full = pd.Series(okb).groupby(market).transform("min").to_numpy().astype(bool)
    tot = pd.Series(np.where(bsp_full, p_bsp, 0.0)).groupby(market).transform("sum").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        p_bsp = np.where(bsp_full, p_bsp / np.where(tot > 0, tot, np.nan), np.nan)

    def e_lay(p, price):  # expected lay return (stake 1, 5%) at `price` if the win probability is p
        return 0.95 + 0.05 * p - p * price

    ratio = np.where(okb, bsp / lp, np.nan)
    rs, ws, os_ = ratio[sel], won[sel], okb[sel]
    exp = {"observed_lay_roi_c5_at_ltp_pre": rep["lay_roi_c5"],
           "expected_if_prop_stale_book_calibrated": r4(e_lay(p_prop[sel], lp[sel]).mean()),
           "expected_if_p_model_right": r4(e_lay(p_model[sel], lp[sel]).mean()),
           "expected_if_bsp_is_truth": r4(np.nanmean(e_lay(p_bsp[sel], lp[sel]))),
           "share_selected_with_full_bsp_market": r4(bsp_full[sel].mean()),
           "strike_observed": r4(won[sel].mean()), "mean_1_over_ltp_pre": r4((1.0 / lp[sel]).mean()),
           "mean_p_prop": r4(p_prop[sel].mean()), "mean_p_model": r4(p_model[sel].mean()),
           "mean_p_bsp_normalised": r4(np.nanmean(p_bsp[sel])), "mean_1_over_bsp": r4(np.nanmean(1.0 / bsp[sb])),
           "median_bsp_over_ltp_pre_winners": r4(np.nanmedian(rs[os_ & ws])),
           "median_bsp_over_ltp_pre_losers": r4(np.nanmedian(rs[os_ & ~ws])),
           "q90_bsp_over_ltp_pre_winners": r4(np.nanquantile(rs[os_ & ws], 0.9)),
           "q90_bsp_over_ltp_pre_losers": r4(np.nanquantile(rs[os_ & ~ws], 0.9)),
           "share_winners_ltp_pre_shorter_than_half_bsp": r4(np.mean(rs[os_ & ws] > 2.0)),
           "share_losers_ltp_pre_shorter_than_half_bsp": r4(np.mean(rs[os_ & ~ws] > 2.0)),
           "share_winners_ltp_pre_le_1.5_and_bsp_ge_5": int(((lp[sel] <= 1.5) & (bsp[sel] >= 5) & ws).sum()),
           "all_strict_median_ratio_winners": r4(np.nanmedian(ratio[okb & won])),
           "all_strict_median_ratio_losers": r4(np.nanmedian(ratio[okb & ~won])),
           "reading": "in-play leakage would put the WINNERS' last trade far below BSP (winners' ratio >> losers'); "
                      "stale-short longshot trades put the LOSERS' last trade below BSP (losers' ratio > winners')"}
    # winners' liability: where does the ltp_pre lay profit come from?
    lw = lp[sel & won]
    bw = bsp[sel & won & okb]
    exp["winners"] = {"n": int((sel & won).sum()), "mean_liability_ltp_pre": r4((lw - 1).mean()), "median_ltp_pre": r4(np.median(lw)),
                      "mean_liability_bsp": r4((bw - 1).mean()), "median_bsp": r4(np.median(bw))}
    if has_t30[sel].any():
        same = np.isclose(lp, t30)
        exp["share_selected_with_t30"] = r4(has_t30[sel].mean())
        exp["share_ltp_pre_eq_ltp_t30_selected"] = r4(same[sel & has_t30].mean())
        exp["share_ltp_pre_eq_ltp_t30_all_scored"] = r4(same[scored & has_t30].mean())
    out({"section": "expected_vs_observed_and_leak_test", "country": code, **exp})

    # ---------------------------------------------------------------- 5. honest uncertainty
    a = summarise(l5[sel], first[sel], market[sel], date[sel], "lay 1u stake @ltp_pre c5")
    b = summarise(orig.lay_returns(bsp[sb], won[sb], 0.05), first[sb], market[sb], date[sb], "same lays @BSP c5 (stake 1)")
    out({"section": "clustered_se", "country": code, "ltp_pre": a, "bsp": b,
         "distinct_race_days_selected": int(pd.Series(date[sel]).nunique())})

    # ---------------------------------------------------------------- 6. executability
    ex = {}
    for c in orig.COMMISSIONS:
        ex[f"bsp_stake1_roi_c{int(c * 100)}"] = r4(orig.lay_returns(bsp[sb], won[sb], c).mean())
    ex["bsp_stake1"] = b
    ql = lay_liab_returns(bsp[sb], won[sb], 0.05)
    ex["bsp_liability1"] = summarise(ql, first[sb], market[sb], date[sb], "same lays @BSP c5, 1u LIABILITY each (SP-lay form)")
    ex["bsp_liability1_aggregate_roi_on_liability"] = r4(orig.lay_returns(bsp[sb], won[sb], 0.05).sum() / (bsp[sb] - 1).sum())
    ex["mean_liability_bsp_per_unit_stake"] = r4((bsp[sb] - 1).mean())
    band = pd.cut(bsp, BSP_BANDS, right=False, labels=BSP_LABELS).astype(str)
    rows = []
    for lab in BSP_LABELS:
        m = sb & (band.to_numpy() == lab)
        if m.sum() < 2:
            continue
        s = summarise(orig.lay_returns(bsp[m], won[m], 0.05), first[m], market[m], date[m], f"BSP band {lab}")
        s_l = summarise(l5[m], first[m], market[m], date[m], "ltp_pre")
        rows.append({"bsp_band": lab, "bets": s["bets"], "bsp_lay_roi_c5": s["roi"], "se_iid": s["se_iid"], "se_day": s["se_day"],
                     "halves": [h.get("roi") for h in s["halves"]], "ltp_pre_lay_roi_c5": s_l["roi"],
                     "median_ltp_pre": r4(np.median(lp[m])), "median_bsp": r4(np.median(bsp[m]))})
    ex["bsp_by_bsp_band"] = rows
    m = sb & (bsp < 50)
    ex["bsp_stake1_excluding_bsp_ge_50"] = summarise(orig.lay_returns(bsp[m], won[m], 0.05), first[m], market[m], date[m],
                                                     "same lays @BSP c5, BSP < 50 only")
    # limit lay at ltp_pre: a lay offer at L is matched only if the market trades at or below L
    for tol, name in ((1.0, "fill_if_bsp_le_ltp_pre"), (1.05, "fill_if_bsp_le_1.05x_ltp_pre")):
        filled = sb & (bsp <= lp * tol)
        f = summarise(l5[filled], first[filled], market[filled], date[filled], f"limit lay @ltp_pre, {name}")
        f["share_filled"] = r4(filled.sum() / sb.sum())
        unf = sb & ~filled
        f["unfilled_hypothetical_roi_c5_at_ltp_pre"] = r4(l5[unf].mean()) if unf.sum() else None
        f["unfilled_bets"] = int(unf.sum())
        ex[name] = f
    worst = np.maximum(lp, np.where(okb, bsp, lp))
    ex["lay_at_max_ltp_pre_bsp"] = summarise(orig.lay_returns(worst[sb], won[sb], 0.05), first[sb], market[sb], date[sb],
                                             "lay at max(ltp_pre, BSP) c5")
    norm = lp * book
    ex["lay_at_book_normalised_ltp_pre"] = summarise(orig.lay_returns(norm[sel], won[sel], 0.05), first[sel], market[sel], date[sel],
                                                     "lay at ltp_pre*book (LTP book forced to 100%) c5")
    st = sel & has_t30
    if st.sum() >= 2:
        s30 = summarise(orig.lay_returns(t30[st], won[st], 0.05), first[st], market[st], date[st], "lay at ltp_t30 c5 (where it exists)")
        s30["share_of_selected_with_t30"] = r4(st.sum() / n_sel)
        s30["median_t30_over_ltp_pre"] = r4(np.median(t30[st] / lp[st]))
        ex["lay_at_ltp_t30_conditional"] = s30
    # baselines
    base = scored & okb
    bb = summarise(orig.lay_returns(bsp[base], won[base], 0.05), first[base], market[base], date[base],
                   "BASELINE: lay every scored runner <= 50 at BSP c5")
    ex["baseline_all_scored_at_bsp"] = {k: bb.get(k) for k in ("bets", "roi", "se_iid", "se_day")}
    near = scored & (np.abs(book - 1.0) <= 0.03)
    bn = summarise(l5[near], first[near], market[near], date[near], "BASELINE: lay every runner <= 50 at ltp_pre, book within 3%")
    ex["baseline_book_within_3pct_at_ltp_pre"] = {k: bn.get(k) for k in ("bets", "roi", "se_iid", "se_day")}
    # selected minus baseline at BSP (same source, independent samples approx.)
    if b.get("se_day") and bb.get("se_day"):
        d = b["roi"] - bb["roi"]
        ex["selected_minus_baseline_at_bsp"] = {"diff": r4(d), "se_day": r4(math.hypot(b["se_day"], bb["se_day"])),
                                                "z": r4(d / math.hypot(b["se_day"], bb["se_day"]))}
    out({"section": "executability", "country": code, **ex})

    # ---------------------------------------------------------------- 7. calendar years
    yrs_all = pd.DatetimeIndex(date).year
    yr_rows = []
    for y in sorted(pd.unique(yrs_all[sel])):
        m = sel & (yrs_all == y)
        mb = m & okb
        a_ = summarise(l5[m], first[m], market[m], date[m], "ltp_pre")
        b_ = summarise(orig.lay_returns(bsp[mb], won[mb], 0.05), first[mb], market[mb], date[mb], "bsp")
        yr_rows.append({"year": int(y), "bets": a_["bets"], "ltp_pre_lay_roi_c5": a_.get("roi"), "se_day": a_.get("se_day"),
                        "bsp_lay_roi_c5": b_.get("roi"), "bsp_se_day": b_.get("se_day"), "median_book": r4(np.median(book[m]))})
    out({"section": "by_year", "country": code, "rows": yr_rows})

    verdict = {"country": code, "reproduced": rep["matches_claim"], "ltp_pre_lay_roi_c5": rep["lay_roi_c5"], "se_iid": a.get("se_iid"),
               "se_day": a.get("se_day"), "bsp_lay_roi_c5": b.get("roi"), "bsp_se_iid": b.get("se_iid"), "bsp_se_day": b.get("se_day"),
               "bsp_positive_both_halves_iid": b.get("positive_both_halves_iid"),
               "expected_if_bsp_is_truth": exp["expected_if_bsp_is_truth"],
               "expected_if_prop": exp["expected_if_prop_stale_book_calibrated"],
               "limit_fill_share": ex["fill_if_bsp_le_ltp_pre"].get("share_filled"),
               "limit_filled_roi_c5": ex["fill_if_bsp_le_ltp_pre"].get("roi"),
               "seconds": round(time.time() - t0, 1)}
    out({"section": "verdict", **verdict})
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default="US")
    ap.add_argument("--end", default=None, help="use markets strictly before this date (development)")
    ap.add_argument("--burn-in", type=int, default=3)
    a = ap.parse_args()
    for code in [c.strip().upper() for c in a.countries.split(",") if c.strip()]:
        analyse(code, a.end, a.burn_in)
    return 0


if __name__ == "__main__":
    sys.exit(main())

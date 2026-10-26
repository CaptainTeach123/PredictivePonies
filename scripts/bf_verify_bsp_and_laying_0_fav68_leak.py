"""Adversarial verification (lens: leakage and data quality) of one claim from
experiment "bsp_and_laying" (scripts/bf_bsp_and_laying.py):

    AU gallops WIN, favourite (rank 1 at ltp_pre) in a 6-8 runner field: the
    paired return at the last traded price beats the return at Betfair SP by
    0.84% of stake (SP-minus-LTP c5 = -0.0084 +/- 0.0010, halves -0.0082 /
    -0.0085, n = 17979), conditional on being matched at ltp_pre.

    python scripts/bf_verify_bsp_and_laying_0_fav68_leak.py [--end 2018-06-01]

(The harness named scripts/bf_verify_bsp_and_laying_0.py, which already exists
from an earlier run; nothing existing is edited, so this file has a suffix.)

The original script is imported unchanged for load() / Rets / ratio_cell, and the
cell is also recomputed independently. Summary numbers only, never horse names.

Checks:
  0  reproduction (original code + independent)
  1  universe: duplicate WIN markets for one race (event+start, venue+start,
     venue+date+race number), rows vs field_size, ltp_pre off the price ladder,
     bsp == ltp_pre, bsp <= 1.01, co-favourites (ties at ltp_pre)
  2  staleness / late-scratching proxy: market book at ltp_pre vs at bsp. A late
     scratching after a runner's last pre-off trade leaves its ltp_pre at the
     pre-removal price (reduction factors then cut matched prices), so the book at
     ltp_pre falls below 1 while the BSP book stays ~1, and BSP < ltp_pre
     mechanically. Stale last trades act the same way.
  3  saddlecloth-gap proxy for scratchings
  4  winners vs losers: systematic vs informational parts of bsp - ltp
  5  cluster-robust SE (race day, meeting), by year, by exact field size
  6  in-play / outlier tail: extreme ratios on winners vs losers, trimmed
  7  t30 path (CONDITIONAL on the favourite trading 30 min out)
  8  survivorship: without the "every runner traded" filter; with dead heats
  9  execution: half-tick and one-tick worse, break-even share of fills at ltp
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import bf_bsp_and_laying as orig  # noqa: E402  imported, not edited

C = 0.05


def out(obj) -> None:
    print(json.dumps(obj, default=orig._json_default), flush=True)


r4 = orig.r4


def cluster_se(x, groups) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = pd.Series(x - x.mean()).groupby(np.asarray(groups)).sum().to_numpy()
    g = len(s)
    return math.sqrt((s ** 2).sum() * g / max(g - 1, 1)) / n


def paired(bsp, lp, won, first, groups=None) -> list:
    """[bets, diff_c5, se, z, half1, half2, back_ltp_c5, back_bsp_c5] (+ cluster se, z)."""
    bsp = np.asarray(bsp, float)
    lp = np.asarray(lp, float)
    won = np.asarray(won, bool)
    first = np.asarray(first, bool)
    n = len(lp)
    if n < 2:
        return [n]
    d = orig.back_returns(bsp, won, C) - orig.back_returns(lp, won, C)
    m, se = orig._mean_se(d)
    h1 = orig._mean_se(d[first])[0] if first.sum() > 1 else float("nan")
    h2 = orig._mean_se(d[~first])[0] if (~first).sum() > 1 else float("nan")
    res = [n, r4(m), r4(se), r4(m / se) if se > 0 else None, r4(h1), r4(h2),
           r4(orig.back_returns(lp, won, C).mean()), r4(orig.back_returns(bsp, won, C).mean())]
    if groups is not None:
        cs = cluster_se(d, groups)
        res += [r4(cs), r4(m / cs) if cs > 0 else None]
    return res


COLS = "[bets, sp_minus_ltp_c5, se, z, half1, half2, back_ltp_c5, back_bsp_c5]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=HERE.parent / "data/raw/betfair/split/au_win.parquet")
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    t0 = time.time()

    # ---------------------------------------------------------------- 0. reproduction with the original code
    df, info, mid = orig.load(a.path, a.end)
    gal = df[~df["harness"]].reset_index(drop=True)
    R = orig.Rets(gal)
    idx = np.flatnonzero(gal["fav"].to_numpy() & (gal["fgroup"].to_numpy() == "6-8"))
    rc = orig.ratio_cell(R, idx)
    out({"section": "0_reproduction_original_code", "markets_gallops": info["markets_gallops"], "mid": str(mid.date()),
         "runners": rc["runners"], "median_ltp": rc["median_ltp_pre"], "ratio_q": rc["ratio_q10_25_50_75_90"],
         "strike": rc["strike"], "implied_ltp": rc["implied_ltp"], "implied_bsp": rc["implied_bsp"],
         "back_ltp": orig.compact(rc["back_ltp_pre"]), "back_tick": orig.compact(rc["back_ltp_pre_tick_down"]),
         "back_bsp": orig.compact(rc["back_bsp"]), "sp_minus_ltp": orig.compact(rc["sp_minus_ltp"]),
         "z": rc["sp_minus_ltp"].get("z")})

    # extra columns for the cell (event_id, venue, program, start) from the raw file
    raw = pd.read_parquet(a.path, columns=["market_id", "event_id", "venue", "start", "date", "market_name", "field_size",
                                           "selection_id", "program", "ltp_pre", "ltp_t30", "bsp", "won"])
    n_raw = len(raw)
    raw = raw.drop_duplicates(["market_id", "selection_id"])
    if a.end:
        raw = raw[raw["date"] < pd.Timestamp(a.end)]
    raw["harness"] = raw["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b")
    mk = raw.groupby("market_id").agg(event_id=("event_id", "first"), venue=("venue", "first"), start=("start", "first"),
                                      date=("date", "first"), name=("market_name", "first"), fs=("field_size", "first"),
                                      rows=("selection_id", "size"), harness=("harness", "first"),
                                      maxprog=("program", lambda s: pd.to_numeric(s, errors="coerce").max()))
    mk["race_no"] = mk["name"].fillna("").str.extract(r"^R(\d+)\b", expand=False)

    # ---------------------------------------------------------------- 1. universe data quality
    used = set(gal["market_id"].unique())
    mku = mk[mk.index.isin(used)]
    d_ev = mku.groupby(["event_id", "start"]).size()
    d_vs = mku.groupby(["venue", "start"]).size()
    d_vr = mku.dropna(subset=["race_no"]).groupby(["venue", "date", "race_no"]).size()
    lp_all = gal["ltp_pre"].to_numpy()
    on_ladder = np.isclose(orig.tick_down(orig.tick_up(lp_all)), lp_all) | (lp_all >= 1000)
    cellm = gal.loc[idx, "market_id"]
    lp_sorted = gal.sort_values(["market_id", "ltp_pre"]).groupby("market_id")["ltp_pre"]
    second = lp_sorted.nth(1)
    second = pd.Series(second.to_numpy(), index=gal.sort_values(["market_id", "ltp_pre"]).groupby("market_id").nth(1)["market_id"].to_numpy())
    fav_lp = gal.loc[idx].set_index("market_id")["ltp_pre"]
    cofav = (second.reindex(fav_lp.index).to_numpy() == fav_lp.to_numpy())
    out({"section": "1_universe_gallops_used", "rows_raw_file": int(n_raw), "markets_used": int(len(mku)),
         "dup_win_markets_same_event_start": int((d_ev > 1).sum()), "dup_same_venue_start": int((d_vs > 1).sum()),
         "dup_same_venue_date_race_no": int((d_vr > 1).sum()), "race_no_parsed_share": r4(mku["race_no"].notna().mean()),
         "markets_rows_ne_field_size": int((mku["rows"] != mku["fs"]).sum()),
         "ltp_pre_off_ladder_share": r4(1 - on_ladder.mean()),
         "bsp_eq_ltp_pre_share_all": r4((gal["bsp"] == gal["ltp_pre"]).mean()),
         "bsp_eq_ltp_pre_share_cell": r4((gal.loc[idx, "bsp"] == gal.loc[idx, "ltp_pre"]).mean()),
         "rows_bsp_le_1.01": int((gal["bsp"] <= 1.01).sum()),
         "cell_cofavourite_share": r4(cofav.mean()), "cell_markets": int(cellm.nunique())})

    # cell frame (favourite, 6-8, bsp present)
    cf = gal.loc[idx].copy()
    cf = cf[cf["bsp_ok"]].reset_index(drop=True)
    cf = cf.join(mk[["event_id", "venue", "start", "maxprog"]], on="market_id")
    cf["meeting"] = cf["venue"].astype(str) + "|" + cf["date"].astype(str)
    cf["cofav"] = cf["market_id"].map(pd.Series(cofav, index=fav_lp.index)).astype(bool)
    lp = cf["ltp_pre"].to_numpy(float)
    bsp = cf["bsp"].to_numpy(float)
    won = cf["won_b"].to_numpy()
    first = cf["first"].to_numpy()
    ratio = bsp / lp
    lr = np.log(ratio)
    base = paired(bsp, lp, won, first, cf["date"].to_numpy())
    out({"section": "0_reproduction_independent", "columns": COLS + " + [se_cluster_day, z_cluster_day]", "cell": base,
         "cluster_meeting": [r4(cluster_se(orig.back_returns(bsp, won, C) - orig.back_returns(lp, won, C), cf["meeting"].to_numpy()))],
         "co_favourites_excluded": paired(bsp[~cf["cofav"]], lp[~cf["cofav"]], won[~cf["cofav"]], first[~cf["cofav"]])})

    # ---------------------------------------------------------------- 2. book at ltp_pre vs bsp (staleness / late-scratch proxy)
    g = gal.copy()
    book_l = (1.0 / g["ltp_pre"]).groupby(g["market_id"]).sum()
    gb = g[g["bsp_ok"]]
    book_b = (1.0 / gb["bsp"]).groupby(gb["market_id"]).sum()
    nb = g.groupby("market_id")["bsp_ok"].min()
    cf["book_l"] = cf["market_id"].map(book_l).to_numpy()
    cf["book_b"] = cf["market_id"].map(book_b).to_numpy()
    cf["all_bsp"] = cf["market_id"].map(nb).astype(bool).to_numpy()
    bl = cf["book_l"].to_numpy()
    bb = cf["book_b"].to_numpy()
    gap_b = bb - bl  # > 0: the SP book is fuller than the last-price book (last prices too long)
    rows = {}
    for lab, m in (("book_ltp < 0.97", bl < 0.97), ("0.97-0.99", (bl >= 0.97) & (bl < 0.99)), ("0.99-1.01", (bl >= 0.99) & (bl < 1.01)),
                   ("1.01-1.03", (bl >= 1.01) & (bl < 1.03)), (">= 1.03", bl >= 1.03)):
        rows[lab] = paired(bsp[m], lp[m], won[m], first[m]) + [r4(np.median(lr[m])) if m.any() else None,
                                                               r4(np.median(bb[m])) if m.any() else None]
    rows2 = {}
    for lab, m in (("book_bsp - book_ltp < -0.02", gap_b < -0.02), ("-0.02..0", (gap_b >= -0.02) & (gap_b < 0)),
                   ("0..0.02", (gap_b >= 0) & (gap_b < 0.02)), (">= 0.02 (last prices too long overall)", gap_b >= 0.02)):
        m = m & cf["all_bsp"].to_numpy()
        rows2[lab] = paired(bsp[m], lp[m], won[m], first[m]) + [r4(np.median(lr[m])) if m.any() else None]
    clean = (bl >= 0.99) & (bl < 1.03) & (np.abs(gap_b) < 0.02)
    out({"section": "2_book_consistency", "columns": COLS + " + [median log(bsp/ltp), median book_bsp]",
         "book_ltp_q05_25_50_75_95": [r4(v) for v in np.quantile(bl, [.05, .25, .5, .75, .95])],
         "book_bsp_minus_ltp_q05_50_95": [r4(v) for v in np.nanquantile(gap_b, [.05, .5, .95])],
         "by_book_ltp": rows, "by_book_bsp_minus_book_ltp": rows2,
         "clean_markets (0.99<=book_ltp<1.03 and |book_bsp-book_ltp|<0.02)": paired(bsp[clean], lp[clean], won[clean], first[clean]) + [r4(clean.mean())]})

    # ---------------------------------------------------------------- 3. saddlecloth gap (a scratching happened at some time)
    gap = cf["maxprog"].to_numpy() > cf["field_size"].to_numpy()
    out({"section": "3_scratching_proxy", "columns": COLS, "share_gap": r4(gap.mean()),
         "no_gap": paired(bsp[~gap], lp[~gap], won[~gap], first[~gap]), "gap": paired(bsp[gap], lp[gap], won[gap], first[gap]),
         "no_gap_and_clean_book": paired(bsp[~gap & clean], lp[~gap & clean], won[~gap & clean], first[~gap & clean])})

    # ---------------------------------------------------------------- 4. winners vs losers
    k = 1 - C
    dec = {}
    for lab, m in (("all", np.ones(len(lp), bool)), ("winners", won), ("losers", ~won)):
        dec[lab] = {"n": int(m.sum()), "mean_bsp_minus_ltp": r4((bsp[m] - lp[m]).mean()), "median_ratio": r4(np.median(ratio[m])),
                    "mean_log_ratio": r4(lr[m].mean()), "share_bsp_below_ltp": r4((bsp[m] < lp[m] - 1e-9).mean()),
                    "share_bsp_above_ltp": r4((bsp[m] > lp[m] + 1e-9).mean()),
                    "share_sp_worse_by_tick": r4((bsp[m] <= orig.tick_down(lp[m]) + 1e-9).mean()),
                    "share_sp_better_by_tick": r4((bsp[m] >= orig.tick_up(lp[m]) - 1e-9).mean())}
    pw = won.mean()
    e_all = (bsp - lp).mean()
    e_w = (bsp[won] - lp[won]).mean()
    # what the paired diff would be if winners had the same log ratio as all favourites (no information in the last minute)
    diff_sys = (won * (lp * np.exp(lr.mean()) - lp) * k).mean()
    out({"section": "4_winners_vs_losers", "rows": dec, "paired_c5": r4(pw * e_w * k),
         "systematic_part_c5 (winners get the all-favourite mean log ratio)": r4(diff_sys),
         "informational_part_c5": r4(pw * e_w * k - diff_sys),
         "t_winners_vs_losers_log_ratio": r4((lr[won].mean() - lr[~won].mean()) / math.sqrt(lr[won].var() / won.sum() + lr[~won].var() / (~won).sum()))})

    # ---------------------------------------------------------------- 5. clustered SE, years, exact field size
    years = pd.DatetimeIndex(cf["date"]).year.to_numpy()
    by_year = {str(y): paired(bsp[years == y], lp[years == y], won[years == y], first[years == y]) for y in sorted(set(years))}
    fs = cf["field_size"].to_numpy()
    by_fs = {str(f): paired(bsp[fs == f], lp[fs == f], won[fs == f], first[fs == f]) for f in (6, 7, 8)}
    band = cf["band"].to_numpy()
    by_band = {b: paired(bsp[band == b], lp[band == b], won[band == b], first[band == b]) for b in orig.BAND_LABELS if (band == b).sum() >= 100}
    out({"section": "5_splits", "columns": COLS, "by_year": by_year, "by_field_size": by_fs, "by_ltp_band": by_band})

    # ---------------------------------------------------------------- 6. tails / outliers
    tails = {}
    for lab, m in (("winners", won), ("losers", ~won)):
        tails[lab] = [r4(v) for v in np.quantile(lr[m], [.001, .01, .05, .5, .95, .99, .999])]
    trims = {}
    for t in (0.05, 0.10, 0.20, 0.30):
        keep = np.abs(lr) <= t
        trims[f"|log ratio| <= {t}"] = paired(bsp[keep], lp[keep], won[keep], first[keep]) + [r4(keep.mean())]
    wb = lp * np.exp(np.clip(lr, -0.05, 0.05))
    trims["winsorised at +/-5%"] = paired(wb, lp, won, first)
    out({"section": "6_tails", "columns": COLS + " + [share kept]", "log_ratio_q001_01_05_50_95_99_999": tails, "trimmed": trims})

    # ---------------------------------------------------------------- 7. t30 path (conditional on a t30 trade for the favourite)
    has = cf["has_t30"].to_numpy()
    t30 = cf["ltp_t30"].to_numpy(float)
    mv = np.log(t30 / lp)  # > 0 = firmed from t30 to pre
    t30r = {"share_with_t30": r4(has.mean()),
            "bsp_vs_ltp_pre (t30 subset)": paired(bsp[has], lp[has], won[has], first[has]),
            "no_t30_subset bsp_vs_ltp_pre": paired(bsp[~has], lp[~has], won[~has], first[~has]),
            "bsp_vs_ltp_t30": paired(bsp[has], t30[has], won[has], first[has]),
            "ltp_pre_vs_ltp_t30": paired(lp[has], t30[has], won[has], first[has])}
    for lab, m in (("firmed t30->pre", has & (mv > 0.0)), ("unchanged", has & (mv == 0.0)), ("drifted t30->pre", has & (mv < 0.0))):
        t30r[lab] = paired(bsp[m], lp[m], won[m], first[m]) + [r4(np.median(lr[m])) if m.any() else None]
    out({"section": "7_t30_conditional", "columns": COLS + " (+ median log ratio)", **t30r})

    # ---------------------------------------------------------------- 8. survivorship / filters
    full = raw[~raw["harness"] & raw["won"].notna()].copy()
    full["nw"] = full.groupby("market_id")["won"].transform("sum")
    pr = full[full["ltp_pre"].notna() & (full["ltp_pre"] > 1)].sort_values(["market_id", "ltp_pre", "selection_id"])
    fv = pr[pr.groupby("market_id").cumcount() == 0].copy()
    fv = fv[(fv["field_size"] >= 6) & (fv["field_size"] <= 8) & fv["bsp"].notna() & (fv["bsp"] > 1)]
    n_pr = pr.groupby("market_id").size()
    n_all = full.groupby("market_id").size()
    fv["all_priced"] = (fv["market_id"].map(n_pr) == fv["market_id"].map(n_all)).to_numpy()
    fv["first"] = fv["date"] < mid
    # dead heats kept: winners paid at price / n winners (dead-heat rule), return = (price/nw - 1)*(1-c) approx via share
    def dh(fr):
        w = fr["won"].to_numpy() == 1
        share = np.where(w, 1.0 / fr["nw"].to_numpy(), 0.0)
        def ret(p):
            p = np.asarray(p, float)
            g_ = share * p - 1.0
            return np.where(g_ > 0, g_ * (1 - C), g_)
        d = ret(fr["bsp"]) - ret(fr["ltp_pre"])
        m, se = orig._mean_se(d)
        return [len(d), r4(m), r4(se)]
    one = fv["nw"] == 1
    out({"section": "8_filters", "columns": COLS,
         "orig_filters_rebuilt": paired(fv.loc[one & fv["all_priced"], "bsp"], fv.loc[one & fv["all_priced"], "ltp_pre"],
                                        fv.loc[one & fv["all_priced"], "won"] == 1, fv.loc[one & fv["all_priced"], "first"]),
         "without_all_priced_filter": paired(fv.loc[one, "bsp"], fv.loc[one, "ltp_pre"], fv.loc[one, "won"] == 1, fv.loc[one, "first"]),
         "only_markets_dropped_by_all_priced": paired(fv.loc[one & ~fv["all_priced"], "bsp"], fv.loc[one & ~fv["all_priced"], "ltp_pre"],
                                                      fv.loc[one & ~fv["all_priced"], "won"] == 1, fv.loc[one & ~fv["all_priced"], "first"]),
         "with_dead_heats_dh_rule [n, diff_c5, se]": dh(fv[fv["all_priced"]]),
         "dead_heat_markets_in_cell": int((fv["nw"] > 1).sum())})

    # ---------------------------------------------------------------- 9. execution
    r_l = orig.back_returns(lp, won, C)
    r_t = orig.back_returns(orig.tick_down(lp), won, C)
    r_b = orig.back_returns(bsp, won, C)
    dh_ = r_b - 0.5 * (r_l + r_t)
    dt_ = r_b - r_t
    p_star = (r_b.mean() - r_t.mean()) / (r_l.mean() - r_t.mean())
    out({"section": "9_execution", "roi_c5_ltp_tick_bsp": [r4(r_l.mean()), r4(r_t.mean()), r4(r_b.mean())],
         "sp_minus_half_tick [mean, se]": [r4(dh_.mean()), r4(dh_.std(ddof=1) / math.sqrt(len(dh_)))],
         "sp_minus_one_tick [mean, se]": [r4(dt_.mean()), r4(dt_.std(ddof=1) / math.sqrt(len(dt_)))],
         "share_filled_at_ltp_to_tie_sp (rest one tick worse)": r4(p_star),
         "median_tick_share_of_price": r4(np.median(orig._increment(lp) / lp))})
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

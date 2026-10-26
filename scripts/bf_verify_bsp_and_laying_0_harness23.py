"""Adversarial verification (leakage / data-quality lens) of the "bsp_and_laying" claim:

    AU harness WIN runners with ltp_pre in [2, 3): backing at Betfair SP returns
    +0.64% (+/- 0.13%) of stake more than backing at the last traded price
    (5% commission on both; halves +0.56% / +0.71%).

    python scripts/bf_verify_bsp_and_laying_0_harness23.py [--end 2018-06-01]

(Written to a new file name because scripts/bf_verify_bsp_and_laying_0.py already
holds a sibling verification of a different cell and must not be edited.)

Reuses load() from scripts/bf_bsp_and_laying.py (imported, not edited) so the
reproduction is exact, then attacks the cell:

  0. data quality: duplicated races under two market_ids, rows vs field_size,
     BSP coverage structure (whole markets vs runners), dispersion of the
     last-traded book per market for harness vs gallops (stale-print diagnostic:
     harness markets do not go in-play on Betfair, so ltp_pre is cut at the
     SCHEDULED start, and thin markets leave old prints).
  1. reproduction of the headline cell (c0 / c5 / c8, SE, halves).
  2. decomposition: the paired gain is strike x E[(bsp - ltp)(1 - c) | won];
     mean log(bsp / ltp) for winners vs losers; share of the gain from the top
     1% / 5% of winner price moves; winsorised and trimmed versions.
  3. stratification by the market's last-traded book sum(1 / ltp_pre) (known
     before the off) and, descriptively, by book_bsp / book_ltp.
  4. a pre-race-computable comparator: ltp_pre rescaled so every market's
     last-traded book equals the code's median book (removes the collectively
     stale component of the prints, keeps the relative prices).
  5. time (year, halves) and liquidity proxies (runner / market has a t30 trade).
  6. the same diagnostics for gallops $2-3 (liquid, in-play cutoff) and for the
     two other harness cells named in the claim (favourites, $1-2).
  7. BH over every stratified cell printed here (one family), with the count.

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

COMMS = (0.0, 0.05, 0.08)
C5 = 0.05
BOOK_BINS = [0.0, 0.97, 0.99, 1.0, 1.01, 1.02, 1.04, 9.0]
BOOK_LABELS = ["<0.97", "0.97-0.99", "0.99-1.00", "1.00-1.01", "1.01-1.02", "1.02-1.04", ">1.04"]
RATIO_BINS = [0.0, 0.98, 0.995, 1.005, 1.02, 9.0]
RATIO_LABELS = ["<0.98", "0.98-0.995", "0.995-1.005", "1.005-1.02", ">1.02"]
WINSOR = [(0.8, 1.25), (0.9, 1.1), (0.95, 1.05)]
CELLS: list[dict] = []


def out(obj) -> None:
    print(json.dumps(obj, default=bspl._json_default), flush=True)


def r4(x):
    return bspl.r4(x)


def mean_se(x) -> tuple[float, float]:
    return bspl._mean_se(np.asarray(x, dtype=float))


def paired_stats(bsp, ltp, won, first, label: str | None = None, family: str | None = None) -> dict:
    """Paired (bsp return - ltp return) per unit stake at c = 0 / 5% / 8%: nonzero on winners only."""
    bsp, ltp, won, first = (np.asarray(a) for a in (bsp, ltp, won, first))
    n = len(bsp)
    res = {"bets": int(n), "n_won": int(won.sum())}
    if n == 0:
        return res
    for c in COMMS:
        d = np.where(won, (bsp - ltp) * (1.0 - c), 0.0)
        res[f"diff_c{int(c * 100)}"] = r4(d.mean())
    d5 = np.where(won, (bsp - ltp) * (1.0 - C5), 0.0)
    m, se = mean_se(d5)
    res["se"] = r4(se)
    res["z"] = r4(m / se) if se and se > 0 else None
    halves = []
    for mask in (first, ~first):
        hm, hs = mean_se(d5[mask])
        halves.append([int(mask.sum()), r4(hm), r4(hs)])
    res["halves_c5 [n, diff, se]"] = halves
    res["positive_both_halves"] = bool(all(h[1] is not None and h[2] and h[1] > 2 * h[2] for h in halves))
    res["back_ltp_c5"] = r4(np.where(won, (ltp - 1.0) * (1.0 - C5), -1.0).mean())
    res["back_bsp_c5"] = r4(np.where(won, (bsp - 1.0) * (1.0 - C5), -1.0).mean())
    if label is not None:
        p = bspl.one_sided_p(m / se) if se and se > 0 and n >= 100 else float("nan")
        CELLS.append({"family": family or "stratified", "cell": label, "bets": int(n), "diff_c5": r4(m), "se": r4(se),
                      "p": p, "both_halves": res["positive_both_halves"]})
    return res


def decomposition(sub: pd.DataFrame) -> dict:
    bsp, ltp, won = sub["bsp"].to_numpy(float), sub["ltp_pre"].to_numpy(float), sub["won_b"].to_numpy()
    lr = np.log(bsp / ltp)
    dp = bsp - ltp
    strike = won.mean()
    res = {"strike": r4(strike), "implied_ltp": r4((1.0 / ltp).mean()), "implied_bsp": r4((1.0 / bsp).mean()),
           "mean_log_ratio": {"all": [r4(v) for v in mean_se(lr)], "winners": [r4(v) for v in mean_se(lr[won])],
                              "losers": [r4(v) for v in mean_se(lr[~won])]},
           "mean_bsp_minus_ltp": {"all": [r4(v) for v in mean_se(dp)], "winners": [r4(v) for v in mean_se(dp[won])],
                                  "losers": [r4(v) for v in mean_se(dp[~won])]},
           "gain_identity strike*E[dp|won]*0.95": r4(strike * dp[won].mean() * 0.95),
           "if_shift_were_uniform strike*E[dp|all]*0.95": r4(strike * dp.mean() * 0.95),
           "ratio_quantiles winners q01..q99": [r4(v) for v in np.quantile(bsp[won] / ltp[won], [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])],
           "ratio_quantiles losers q01..q99": [r4(v) for v in np.quantile(bsp[~won] / ltp[~won], [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])]}
    # share of the total paired gain from the largest winner moves
    d5 = np.where(won, dp * 0.95, 0.0)
    total = d5.sum()
    ratio_w = bsp[won] / ltp[won]
    order = np.argsort(-ratio_w)
    dw = d5[won][order]
    nw = len(dw)
    res["gain_share_from_top_winner_moves"] = {f"top {p:g}% of winners (ratio >= {r4(ratio_w[order][max(int(nw * p / 100) - 1, 0)])})":
                                               r4(dw[: max(int(nw * p / 100), 1)].sum() / total) if total else None for p in (1, 5, 10)}
    res["share_winners_ratio_over_1.25"] = r4((ratio_w > 1.25).mean())
    res["share_winners_ratio_over_1.10"] = r4((ratio_w > 1.10).mean())
    return res


def robust_versions(sub: pd.DataFrame, tag: str) -> dict:
    bsp, ltp, won, first = sub["bsp"].to_numpy(float), sub["ltp_pre"].to_numpy(float), sub["won_b"].to_numpy(), sub["first"].to_numpy()
    res = {}
    for lo, hi in WINSOR:
        r = np.clip(bsp / ltp, lo, hi)
        res[f"winsor ratio [{lo:g},{hi:g}]"] = paired_stats(r * ltp, ltp, won, first, f"{tag} winsor [{lo:g},{hi:g}]", "robust")
    r = bsp / ltp
    q1, q99 = np.quantile(r, [0.01, 0.99])
    m = (r > q1) & (r < q99)
    res["trim 1% tails of ratio"] = paired_stats(bsp[m], ltp[m], won[m], first[m], f"{tag} trim 1%", "robust")
    # one tick worse than the last price (the original's attainability check)
    res["sp vs one tick worse than ltp"] = paired_stats(bsp, bspl.tick_down(ltp), won, first)
    return res


def by_strata(sub: pd.DataFrame, col: str, bins, labels, tag: str, family: str) -> dict:
    s = pd.cut(sub[col], bins, labels=labels).astype(str)
    res = {}
    for lab in labels:
        g = sub[s == lab]
        if len(g) == 0:
            continue
        p = paired_stats(g["bsp"], g["ltp_pre"], g["won_b"], g["first"], f"{tag} {col} {lab}", family)
        p["share_of_runners"] = r4(len(g) / len(sub))
        p["mean_log_ratio_all"] = r4(np.log(g["bsp"] / g["ltp_pre"]).mean())
        p["contribution_to_total_diff_c5"] = r4(np.where(g["won_b"], (g["bsp"] - g["ltp_pre"]) * 0.95, 0.0).sum() / len(sub))
        res[lab] = p
    return res


def demargined(sub: pd.DataFrame, target_book: float, tag: str) -> dict:
    """ltp_pre rescaled so the market's last-traded book equals target_book (uses ltp_pre of all
    runners in the market only: computable before the off)."""
    ltp_adj = sub["ltp_pre"].to_numpy(float) * sub["book_l"].to_numpy(float) / target_book
    res = {"target_book": r4(target_book),
           "sp_vs_demargined_ltp": paired_stats(sub["bsp"], ltp_adj, sub["won_b"], sub["first"], f"{tag} demargined", "demargined"),
           "demargined_ltp_vs_raw_ltp (what the rescale alone is worth)": paired_stats(ltp_adj, sub["ltp_pre"], sub["won_b"], sub["first"])}
    return res


def analyse(df: pd.DataFrame, mask: np.ndarray, tag: str, target_book: float, full: bool = True) -> None:
    sub = df[mask & df["bsp_ok"].to_numpy()].copy()
    rep = paired_stats(sub["bsp"], sub["ltp_pre"], sub["won_b"], sub["first"], f"{tag} headline", "headline")
    out({"section": "1 reproduction", "cell": tag, "runners": int(len(sub)), "markets": int(sub["market_id"].nunique()), **rep})
    out({"section": "2 decomposition", "cell": tag, **decomposition(sub)})
    out({"section": "2 robust versions", "cell": tag, **robust_versions(sub, tag)})
    out({"section": "3 by last-traded book of the market (sum 1/ltp_pre; known before the off)", "cell": tag,
         "book_bins": BOOK_LABELS, "rows": by_strata(sub, "book_l", BOOK_BINS, BOOK_LABELS, tag, "book_ltp")})
    consistent = sub[(sub["book_l"] >= 0.99) & (sub["book_l"] < 1.01)]
    out({"section": "3 consistent-book subset (0.99 <= book_ltp < 1.01)", "cell": tag,
         **paired_stats(consistent["bsp"], consistent["ltp_pre"], consistent["won_b"], consistent["first"], f"{tag} book 0.99-1.01", "book_ltp"),
         "robust": {f"winsor [{lo:g},{hi:g}]": paired_stats(np.clip(consistent["bsp"] / consistent["ltp_pre"], lo, hi) * consistent["ltp_pre"],
                                                            consistent["ltp_pre"], consistent["won_b"], consistent["first"]) for lo, hi in WINSOR[:2]}})
    out({"section": "3 DESCRIPTIVE by book_bsp / book_ltp (uses BSP: not known before the off)", "cell": tag,
         "rows": by_strata(sub, "book_ratio", RATIO_BINS, RATIO_LABELS, tag, "book_ratio_descriptive")})
    out({"section": "4 de-margined comparator", "cell": tag, **demargined(sub, target_book, tag)})
    if full:
        sub["year"] = sub["date"].dt.year
        out({"section": "5 by year", "cell": tag,
             "rows": {int(y): paired_stats(g["bsp"], g["ltp_pre"], g["won_b"], g["first"], f"{tag} year {y}", "time") for y, g in sub.groupby("year")}})
        liq = {}
        for name, m in (("runner has t30 trade", sub["has_t30"]), ("runner no t30 trade", ~sub["has_t30"]),
                        ("market: no runner traded at t30", sub["mkt_t30"] == 0), ("market: >= 1/3 of runners traded at t30", sub["mkt_t30"] >= 1 / 3)):
            g = sub[m.to_numpy()]
            liq[name] = paired_stats(g["bsp"], g["ltp_pre"], g["won_b"], g["first"], f"{tag} {name}", "liquidity")
        out({"section": "5 liquidity proxies (t30 trade present for 17% of harness runners)", "cell": tag, "rows": liq})
        fs = {}
        for f in bspl.FIELD_LABELS:
            g = sub[sub["fgroup"] == f]
            fs[f] = paired_stats(g["bsp"], g["ltp_pre"], g["won_b"], g["first"], f"{tag} field {f}", "field")
        out({"section": "5 by field size", "cell": tag, "rows": fs})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=HERE.parent / "data/raw/betfair/split/au_win.parquet")
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    t0 = time.time()
    df, info, mid = bspl.load(a.path, a.end)
    extra = pd.read_parquet(a.path, columns=["market_id", "event_id", "start", "venue"]).drop_duplicates("market_id")
    df = df.merge(extra, on="market_id", how="left")
    df["inv_l"] = 1.0 / df["ltp_pre"]
    df["inv_b"] = np.where(df["bsp_ok"], 1.0 / df["bsp"].fillna(1.0), np.nan)
    mk = df.groupby("market_id").agg(book_l=("inv_l", "sum"), book_b=("inv_b", "sum"), all_bsp=("bsp_ok", "min"),
                                     any_bsp=("bsp_ok", "max"), rows=("inv_l", "size"), fs=("field_size", "first"),
                                     mkt_t30=("has_t30", "mean"))
    mk["book_b"] = mk["book_b"].where(mk["all_bsp"])
    df = df.join(mk[["book_l", "book_b", "mkt_t30"]], on="market_id")
    df["book_ratio"] = df["book_b"] / df["book_l"]
    out({"section": "data", **info, "end": a.end})

    # 0. data quality
    dq = {}
    for code in ("harness", "gallops"):
        h = df[df["harness"] == (code == "harness")]
        m = h.drop_duplicates("market_id")
        mm = mk.loc[m["market_id"]]
        dq[code] = {"markets": int(len(m)), "runners": int(len(h)),
                    "dup_event_id+market_name": int(m.duplicated(["event_id", "market_name"]).sum()),
                    "dup_venue+start": int(m.duplicated(["venue", "start"]).sum()),
                    "dup_selection_id+start": int(h.duplicated(["selection_id", "start"]).sum()),
                    "rows_ne_field_size_markets": int((mm["rows"] != mm["fs"]).sum()),
                    "bsp_markets_all/none/partial": [int(mm["all_bsp"].sum()), int((~mm["any_bsp"]).sum()), int((mm["any_bsp"] & ~mm["all_bsp"]).sum())],
                    "runners_without_bsp_in_partial_markets": int((~h["bsp_ok"] & h["market_id"].isin(mm.index[mm["any_bsp"] & ~mm["all_bsp"]])).sum()),
                    "book_ltp q05/q25/q50/q75/q95": [r4(v) for v in mm["book_l"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])],
                    "book_bsp q05/q25/q50/q75/q95": [r4(v) for v in mm["book_b"].dropna().quantile([0.05, 0.25, 0.5, 0.75, 0.95])],
                    "share_markets_book_ltp_outside_0.99_1.01": r4(((mm["book_l"] < 0.99) | (mm["book_l"] >= 1.01)).mean()),
                    "share_markets_book_ltp_over_1.02": r4((mm["book_l"] >= 1.02).mean()),
                    "share_markets_book_ltp_under_0.98": r4((mm["book_l"] < 0.98).mean()),
                    "t30_runner_coverage": r4(h["has_t30"].mean())}
    out({"section": "0 data quality", **dq})

    band = df["band"].to_numpy()
    harness = df["harness"].to_numpy()
    fav = df["fav"].to_numpy()
    tb_h = float(mk.loc[df.loc[harness, "market_id"].unique(), "book_l"].median())
    tb_g = float(mk.loc[df.loc[~harness, "market_id"].unique(), "book_l"].median())
    analyse(df, harness & (band == "2-3"), "harness 2-3", tb_h, full=True)
    analyse(df, ~harness & (band == "2-3"), "gallops 2-3", tb_g, full=False)
    analyse(df, harness & fav, "harness favourites", tb_h, full=False)
    analyse(df, harness & (band == "1-2"), "harness 1-2", tb_h, full=False)

    # 7. multiple testing over everything stratified here
    cells = pd.DataFrame(CELLS)
    cells["p"] = pd.to_numeric(cells["p"], errors="coerce")
    cells["bh"] = bspl.benjamini_hochberg(cells["p"].to_numpy(), 0.05)
    fam = {}
    for f, g in cells.groupby("family"):
        fam[f] = {"cells": int(len(g)), "testable": int(g["p"].notna().sum()), "bh_rejections": int(g["bh"].sum()),
                  "both_halves": int(g["both_halves"].sum()), "survivors": int((g["bh"] & g["both_halves"]).sum())}
    out({"section": "7 multiple testing (BH 5% over all cells printed by this script, one family and by sub-family)",
         "cells_total": int(len(cells)), "testable": int(cells["p"].notna().sum()), "bh_rejections_overall": int(cells["bh"].sum()),
         "survivors_overall": int((cells["bh"] & cells["both_halves"]).sum()), "by_family": fam,
         "survivor_list": [[r["family"], r["cell"], int(r["bets"]), r["diff_c5"], r["se"]] for _, r in cells[cells["bh"] & cells["both_halves"]].iterrows()]})
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

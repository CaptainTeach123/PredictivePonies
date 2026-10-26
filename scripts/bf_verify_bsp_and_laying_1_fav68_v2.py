"""Adversarial re-check (statistics + execution lens) of the "bsp_and_laying" claim:

    AU gallops WIN favourite (rank 1 at ltp_pre) in a 6-8 runner field: the last traded
    price beats Betfair SP by +0.84% of stake after 5% commission (paired SP-minus-LTP
    -0.0084 +/- 0.0010, n 17,979, halves -0.82% / -0.85%); also gallops $2-3, favourites
    2-5 and $50+ runners; "one tick worse than the last price flips it".

    python scripts/bf_verify_bsp_and_laying_1_fav68_v2.py [--end 2018-06-01] [--boot 300]

(The harness asked for scripts/bf_verify_bsp_and_laying_1.py; that file and a _fav68 variant
already exist from earlier runs, so this independent re-check carries a further suffix.
Nothing existing is edited.)

Same filters as scripts/bf_bsp_and_laying.py, re-derived with event_id / venue / start kept.
Nothing is fitted. What is checked (JSON lines, summary numbers only, never horse names):

  1. reproduction of the cell at commission 0/5/8/10%, per-bet SE, halves;
  2. SEs clustered by market / meeting (venue x date) / race day, day-block bootstrap,
     month-as-unit t;
  3. late-scratching / stale-quote artefact: market book at ltp_pre and at bsp; the effect
     by book bucket, after excluding markets whose ltp_pre book is < 0.97 (a runner priced
     before a removal, so a bet at ltp_pre would have been cut by a reduction factor) or
     whose bsp book / ltp book ratio is extreme;
  4. mechanism: mean log(bsp/ltp_pre) for winners vs losers (general SP shortening of
     favourites vs informed late money), strike vs implied;
  5. influence: winsorised / trimmed ratios, top-1% share;
  6. execution: one/two ticks worse, break-even share matched at exactly ltp_pre,
     the "limit at ltp_pre, SP fallback if unmatched" execution under adverse fill
     (filled only when the price drifted, i.e. bsp >= ltp_pre);
  7. specificity / stability: years, odd/even weeks, exact field 6/7/8, price sub-bands,
     second favourites in 6-8 fields, non-favourites at the same prices;
  8. siblings: gallops $2-3, favourites 2-5, $50+ with clustered SEs, bootstrap, trims;
  9. multiple testing: two-sided Bonferroni / BH over the original's 122 paired cells and
     over all 556 cells it touched.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import bf_bsp_and_laying as bspl  # noqa: E402

SCRATCH = Path("/tmp/claude-0/-home-user-PredictivePonies/43d94981-d22b-59d1-a1cc-00a2ff7cb49d/scratchpad")
COMMS = (0.0, 0.05, 0.08, 0.10)
C = 0.05


def out(o):
    print(json.dumps(o, default=bspl._json_default), flush=True)


def r4(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


def r5(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 5)


def load(path, end):
    cols = ["market_id", "event_id", "venue", "market_name", "date", "start", "field_size", "selection_id",
            "ltp_pre", "ltp_t30", "bsp", "won", "market_type"]
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
    df["won_b"] = df["won"].to_numpy() == 1
    df["harness"] = df["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b")
    df["band"] = pd.cut(df["ltp_pre"], bspl.PRICE_BANDS, right=False, labels=bspl.BAND_LABELS).astype(str)
    df["fgroup"] = pd.cut(df["field_size"], bspl.FIELD_EDGES, right=False, labels=bspl.FIELD_LABELS).astype(str)
    df["bsp_ok"] = df["bsp"].notna() & (df["bsp"] > 1)
    days = sorted(df["date"].unique())
    mid = pd.Timestamp(days[len(days) // 2])
    df["first"] = df["date"] < mid
    df["book_ltp"] = (1.0 / df["ltp_pre"]).groupby(df["market_id"]).transform("sum")
    inv_bsp = (1.0 / df["bsp"].where(df["bsp_ok"]))
    df["book_bsp"] = inv_bsp.groupby(df["market_id"]).transform("sum")
    df["all_bsp"] = df["bsp_ok"].groupby(df["market_id"]).transform("min").astype(bool)
    df["meeting"] = df["venue"].fillna("") + "|" + df["date"].dt.strftime("%Y-%m-%d")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["year"] = df["date"].dt.year
    df["isoweek"] = df["date"].dt.isocalendar().week.astype(int)
    df["day"] = df["date"].dt.strftime("%Y-%m-%d")
    return df, mid


def paired(d, c=C):
    lp, bsp, won = d["ltp_pre"].to_numpy(float), d["bsp"].to_numpy(float), d["won_b"].to_numpy()
    return np.where(won, (bsp - lp) * (1 - c), 0.0)


def back(price, won, c=C):
    return np.where(won, (np.asarray(price, float) - 1) * (1 - c), -1.0)


def mse(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(n))


def cluster_se(x, g):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    s = pd.Series(x - m).groupby(np.asarray(g)).sum().to_numpy()
    G = len(s)
    return float(math.sqrt((s ** 2).sum() * G / max(G - 1, 1)) / n), G


def boot_day(x, day, B, rng):
    x = np.asarray(x, float)
    grp = pd.Series(x).groupby(np.asarray(day))
    sums = grp.sum().to_numpy()
    cnts = grp.size().to_numpy()
    G = len(sums)
    res = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, G, G), minlength=G)
        res[b] = (w * sums).sum() / (w * cnts).sum()
    return float(res.std(ddof=1)), [r5(v) for v in np.quantile(res, [0.025, 0.975])]


def full_cell(d, B, rng, name):
    x = paired(d)
    m, se = mse(x)
    res = {"name": name, "n": len(d), "n_won": int(d["won_b"].sum()),
           "median_ltp": r4(np.median(d["ltp_pre"])),
           "diff_c": {f"c{int(c*100)}": r5(paired(d, c).mean()) for c in COMMS},
           "diff_c5": r5(m), "se": r5(se), "z": r4(m / se) if se > 0 else None}
    h = [mse(x[d["first"].to_numpy()]), mse(x[~d["first"].to_numpy()])]
    res["halves"] = [[r5(a), r5(b)] for a, b in h]
    for key in ("market_id", "meeting", "day"):
        s, G = cluster_se(x, d[key].to_numpy())
        res[f"se_cl_{key}"] = [r5(s), G, r4(m / s)]
    mon = pd.Series(x).groupby(d["month"].to_numpy()).mean()
    res["month_t"] = [len(mon), r5(mon.mean()), r5(mon.std(ddof=1) / math.sqrt(len(mon))),
                      r4(mon.mean() / (mon.std(ddof=1) / math.sqrt(len(mon))))]
    if B:
        bs, ci = boot_day(x, d["day"].to_numpy(), B, rng)
        res["boot_day_se_ci"] = [r5(bs), ci]
    lp, bsp, won = d["ltp_pre"].to_numpy(float), d["bsp"].to_numpy(float), d["won_b"].to_numpy()
    res["back_c5_ltp_tick_2tick_bsp"] = [r5(back(lp, won).mean()), r5(back(bspl.tick_down(lp), won).mean()),
                                         r5(back(bspl.tick_down(bspl.tick_down(lp)), won).mean()), r5(back(bsp, won).mean())]
    a, t1, sp = res["back_c5_ltp_tick_2tick_bsp"][0], res["back_c5_ltp_tick_2tick_bsp"][1], res["back_c5_ltp_tick_2tick_bsp"][3]
    # share p matched exactly at ltp_pre (rest one tick worse) needed to equal SP
    res["breakeven_share_at_ltp_vs_tick"] = r4((sp - t1) / (a - t1)) if a != t1 else None
    lr = np.log(bsp / lp)
    res["mean_log_ratio_winners_losers_all"] = [r5(lr[won].mean()), r5(lr[~won].mean()), r5(lr.mean())]
    res["share_bsp_below_tickdown_winners_losers"] = [r4((bsp[won] <= bspl.tick_down(lp[won]) + 1e-9).mean()),
                                                      r4((bsp[~won] <= bspl.tick_down(lp[~won]) + 1e-9).mean())]
    res["strike_vs_implied_ltp_bsp"] = [r4(won.mean()), r4((1 / lp).mean()), r4((1 / bsp).mean())]
    # influence: winsorise the ratio at 1/99 pct among winners, trim |log ratio| > 0.3
    q1, q99 = np.quantile(lr, [0.01, 0.99])
    lrw = np.clip(lr, q1, q99)
    xw = np.where(won, (lp * np.exp(lrw) - lp) * (1 - C), 0.0)
    res["winsor_1_99"] = list(map(r5, mse(xw)))
    keep = np.abs(lr) <= 0.3
    res["trim_abslogratio_le_0.3"] = [int((~keep).sum()), *map(r5, mse(x[keep]))]
    if won.sum() > 100:
        xa = np.abs(x[won])
        thr = np.quantile(xa, 0.99)
        top = won & (np.abs(x) >= thr)
        res["share_of_total_from_top1pct_winners"] = r4(x[top].sum() / x.sum()) if x.sum() != 0 else None
        res["median_paired_given_won"] = r5(np.median(x[won]))
    # limit order at ltp_pre, SP fallback; adverse-fill model: matched at ltp only when the price later
    # drifted (bsp >= ltp, the backer's limit would be hit by a layer), else SP
    ret_adverse = np.where(bsp >= lp, back(lp, won), back(bsp, won))
    res["limit_at_ltp_sp_fallback_adverse_fill_c5"] = r5(ret_adverse.mean())
    res["limit_minus_sp_c5"] = list(map(r5, mse(ret_adverse - back(bsp, won))))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=HERE.parent / "data/raw/betfair/split/au_win.parquet")
    ap.add_argument("--end", default=None)
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    df, mid = load(a.path, a.end)
    g = df[~df["harness"] & df["bsp_ok"]].reset_index(drop=True)
    out({"section": "data", "markets_gallops": int(df.loc[~df["harness"], "market_id"].nunique()),
         "runners_gallops_bsp": len(g), "mid": str(mid.date())})

    fav68 = g[(g["rank"] == 1) & (g["fgroup"] == "6-8")]
    out({"section": "1-2 cell", **full_cell(fav68, a.boot, rng, "gallops fav 6-8")})

    # 3. book diagnostics
    mk = g.groupby("market_id").agg(book_ltp=("book_ltp", "first"), book_bsp=("book_bsp", "first"),
                                    all_bsp=("all_bsp", "first"))
    mk = mk[mk["all_bsp"]]
    out({"section": "3 book", "markets": len(mk),
         "book_ltp_q01_05_50_95_99": [r4(v) for v in mk["book_ltp"].quantile([0.01, 0.05, 0.5, 0.95, 0.99])],
         "book_bsp_q01_05_50_95_99": [r4(v) for v in mk["book_bsp"].quantile([0.01, 0.05, 0.5, 0.95, 0.99])],
         "share_book_ltp_lt_0.97": r4((mk["book_ltp"] < 0.97).mean()),
         "share_book_bsp_over_ltp_gt_1.03": r4((mk["book_bsp"] / mk["book_ltp"] > 1.03).mean())})
    f = fav68.copy()
    f["bk"] = pd.cut(f["book_ltp"], [0, 0.97, 0.99, 1.0, 1.01, 1.02, 1.04, 9])
    rows = {}
    for k, d in f.groupby("bk", observed=True):
        x = paired(d)
        m, s = mse(x)
        rows[str(k)] = [len(d), r5(m), r5(s)]
    f["rb"] = pd.cut(f["book_bsp"] / f["book_ltp"], [0, 0.97, 0.99, 1.01, 1.03, 9])
    rows2 = {}
    for k, d in f.groupby("rb", observed=True):
        x = paired(d)
        m, s = mse(x)
        rows2[str(k)] = [len(d), r5(m), r5(s)]
    clean = f[(f["book_ltp"] >= 0.97) & (f["book_ltp"] <= 1.04) & f["all_bsp"]
              & ((f["book_bsp"] / f["book_ltp"]).between(0.97, 1.03))]
    out({"section": "3 fav68 by book", "by_book_ltp [n, diff, se]": rows, "by_bookbsp_over_bookltp": rows2})
    out({"section": "3 fav68 clean books", **full_cell(clean, 0, rng, "fav 6-8, book_ltp in [0.97,1.04], bsp/ltp book in [0.97,1.03]")})
    # low-book markets: is every runner's SP shorter than its last price by about the missing book
    # (a removal after the last trades, settled with a reduction factor) or only the favourite?
    low = g[g["all_bsp"] & (g["book_ltp"] < 0.97)]
    lm = low.groupby("market_id").agg(book_ltp=("book_ltp", "first"), book_bsp=("book_bsp", "first"),
                                      med_ratio=("bsp", lambda s: np.nan), fs=("field_size", "first"))
    rat = (low["bsp"] / low["ltp_pre"]).groupby(low["market_id"]).median()
    lm["med_ratio"] = rat
    fav_low = low[low["rank"] == 1]
    out({"section": "3b low-book markets (book_ltp < 0.97)", "markets": len(lm),
         "share_of_gallops_markets": r4(len(lm) / g["market_id"].nunique()),
         "median_book_ltp": r4(lm["book_ltp"].median()), "median_book_bsp": r4(lm["book_bsp"].median()),
         "median_over_markets_of_median_runner_bsp_over_ltp": r4(lm["med_ratio"].median()),
         "corr_med_ratio_vs_book_ltp": r4(np.corrcoef(lm["med_ratio"], lm["book_ltp"])[0, 1]),
         "median_field": r4(lm["fs"].median()),
         "fav_median_bsp_over_ltp_low_vs_rest": [r4((fav_low["bsp"] / fav_low["ltp_pre"]).median()),
                                                 r4((g.loc[(g["rank"] == 1) & (g["book_ltp"] >= 0.97), "bsp"]
                                                     / g.loc[(g["rank"] == 1) & (g["book_ltp"] >= 0.97), "ltp_pre"]).median())],
         "share_low_book_by_fgroup": {k: r4(v) for k, v in g[g["rank"] == 1].assign(lb=lambda d: d["book_ltp"] < 0.97)
                                      .groupby("fgroup")["lb"].mean().items()}})
    # pre-off filters only (the book at ltp_pre is known before the off)
    for lo_b in (0.95, 0.97, 0.98, 0.99):
        d = fav68[fav68["book_ltp"] >= lo_b]
        out({"section": "3c fav68 pre-off book filter", **full_cell(d, a.boot if lo_b == 0.97 else 0, rng,
                                                                     f"fav 6-8, book_ltp >= {lo_b}")})
    # model-based correction: deflate ltp_pre by book_ltp / 1.005 when book_ltp < 0.99 (reduction-factor proxy)
    d = fav68.copy()
    adj = np.where(d["book_ltp"] < 0.99, d["book_ltp"] / 1.005, 1.0)
    d["ltp_pre"] = np.maximum(1.01, d["ltp_pre"] * adj)
    out({"section": "3d fav68 ltp deflated by missing book (proxy for a reduction factor)", **full_cell(d, 0, rng, "fav 6-8 adj")})

    # 7. stability / specificity
    rows = {}
    for y, d in fav68.groupby("year"):
        x = paired(d)
        rows[int(y)] = [len(d), *map(r5, mse(x))]
    wk = {}
    for par, d in fav68.groupby(fav68["isoweek"] % 2):
        wk["odd" if par else "even"] = [len(d), *map(r5, mse(paired(d)))]
    fs = {}
    for s_, d in fav68.groupby("field_size"):
        fs[int(s_)] = [len(d), *map(r5, mse(paired(d)))]
    sb = {}
    f2 = fav68.copy()
    f2["pb"] = pd.cut(f2["ltp_pre"], [1, 1.6, 2.0, 2.4, 2.8, 3.2, 4.0, 99], right=False)
    for k, d in f2.groupby("pb", observed=True):
        sb[str(k)] = [len(d), *map(r5, mse(paired(d)))]
    out({"section": "7 fav68 stability", "by_year [n, diff, se]": rows, "odd_even_week": wk, "exact_field": fs,
         "price_subband": sb})
    # same prices, not favourite / second favourites, in 6-8 fields; favourites in other fields at same prices
    lo, hi = fav68["ltp_pre"].quantile([0.1, 0.9])
    spec = {}
    for name, d in {
        "6-8 rank1": fav68,
        "6-8 rank2": g[(g["rank"] == 2) & (g["fgroup"] == "6-8")],
        "6-8 rank>=2 in fav price range": g[(g["rank"] >= 2) & (g["fgroup"] == "6-8") & g["ltp_pre"].between(lo, hi)],
        "9-11 rank1 in fav68 price range": g[(g["rank"] == 1) & (g["fgroup"] == "9-11") & g["ltp_pre"].between(lo, hi)],
        "12+ rank1 in fav68 price range": g[(g["rank"] == 1) & (g["fgroup"] == "12+") & g["ltp_pre"].between(lo, hi)],
        "2-5 rank1 in fav68 price range": g[(g["rank"] == 1) & (g["fgroup"] == "2-5") & g["ltp_pre"].between(lo, hi)],
        "all fields rank1": g[g["rank"] == 1],
        "all fields rank2": g[g["rank"] == 2],
    }.items():
        x = paired(d)
        m, s = mse(x)
        lr = np.log(d["bsp"].to_numpy() / d["ltp_pre"].to_numpy())
        w = d["won_b"].to_numpy()
        spec[name] = {"n": len(d), "diff": r5(m), "se": r5(s), "z": r4(m / s),
                      "mean_log_ratio_win_lose": [r5(lr[w].mean()), r5(lr[~w].mean())]}
    out({"section": "7 specificity", "fav68_price_p10_p90": [r4(lo), r4(hi)], "rows": spec})

    # 6. execution with the ltp_t30 path, conditional on a t30 trade
    t = fav68[fav68["ltp_t30"].notna() & (fav68["ltp_t30"] > 1)]
    x30 = np.where(t["won_b"], (t["ltp_t30"] - t["ltp_pre"]) * (1 - C), 0.0)
    out({"section": "6 t30 path (conditional on t30 trade)", "n": len(t), "share": r4(len(t) / len(fav68)),
         "t30_minus_ltp_pre_c5": list(map(r5, mse(x30))),
         "sp_minus_ltp_pre_c5_same_rows": list(map(r5, mse(paired(t))))})

    # 8. siblings
    for name, d in {"gallops $2-3 all runners": g[g["band"] == "2-3"],
                    "gallops fav 2-5": g[(g["rank"] == 1) & (g["fgroup"] == "2-5")],
                    "gallops $50+ all runners": g[g["band"] == "50+"],
                    "gallops fav all fields": g[g["rank"] == 1]}.items():
        out({"section": "8 sibling", **full_cell(d, a.boot, rng, name)})
        out({"section": "8 sibling book_ltp >= 0.97", **full_cell(d[d["book_ltp"] >= 0.97], a.boot, rng, name + " | book_ltp>=0.97")})
    # every band, all gallops runners, with and without the pre-off book filter
    rows = {}
    for b in bspl.BAND_LABELS:
        d = g[g["band"] == b]
        dc = d[d["book_ltp"] >= 0.97]
        m0, s0 = mse(paired(d))
        m1, s1 = mse(paired(dc))
        rows[b] = [len(d), r5(m0), r5(s0), len(dc), r5(m1), r5(s1), r4(m1 / s1)]
    rows_f = {}
    for fgp in bspl.FIELD_LABELS:
        d = g[(g["rank"] == 1) & (g["fgroup"] == fgp)]
        dc = d[d["book_ltp"] >= 0.97]
        m0, s0 = mse(paired(d))
        m1, s1 = mse(paired(dc))
        rows_f[fgp] = [len(d), r5(m0), r5(s0), len(dc), r5(m1), r5(s1), r4(m1 / s1)]
    out({"section": "8b bands and fav-by-field, raw vs book_ltp >= 0.97",
         "cols": "[n, diff, se, n_clean, diff_clean, se_clean, z_clean]", "bands": rows, "fav_by_field": rows_f})

    # 9. multiple testing over the original's cells CSV
    csv = SCRATCH / "bspl_full_cells.csv"
    if csv.exists() and not a.end:
        cells = pd.read_csv(csv)
        pc = cells[cells["family"] == "sp_minus_ltp"].copy()
        pc = pc[pc["se"].notna() & (pc["se"] > 0)]
        z = pc["roi_c5"] / pc["se"]
        p2 = np.array([math.erfc(abs(v) / math.sqrt(2)) for v in z])
        rej = bspl.benjamini_hochberg(p2, 0.05)
        nall = len(cells)
        out({"section": "9 multiple testing", "paired_cells_with_se": len(pc), "all_cells": nall,
             "bonferroni_z_two_sided_122": r4(_zcrit(0.05 / 122 / 2)), "bonferroni_z_two_sided_all": r4(_zcrit(0.05 / nall / 2)),
             "two_sided_bh_rejections_paired": int(rej.sum()),
             "neg_direction_bh_rejections": int((rej & (z.to_numpy() < 0)).sum()),
             "top_negative_z": [[r["code"], r["name"], r4(r["roi_c5"]), r4(r["se"]), r4(r["roi_c5"] / r["se"])]
                                for _, r in pc.assign(z=z).sort_values("z").head(10).iterrows()]})
    return 0


def _zcrit(p):
    lo, hi = 0.0, 12.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * math.erfc(mid / math.sqrt(2)) > p:
            lo = mid
        else:
            hi = mid
    return mid


if __name__ == "__main__":
    sys.exit(main())

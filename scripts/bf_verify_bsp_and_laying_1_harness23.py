"""Adversarial verification (statistics + execution lens) of the "bsp_and_laying" claim:

    AU harness WIN runners with ltp_pre in [2, 3): backing 1 unit at Betfair SP
    returns +0.64% (+/- 0.13%) of stake more than backing the same runner at the
    last traded price (5% commission on both; n 20,932; halves +0.56% / +0.71%);
    also harness favourites overall (+0.32% +/- 0.08%) and $1-2 (+0.27% +/- 0.07%).
    A RELATIVE execution finding: backing harness $2-3 at SP still loses -1.4%.

    python scripts/bf_verify_bsp_and_laying_1_harness23.py [--end 2018-06-01] [--boot 1000] [--splits 500]

(The harness asked for scripts/bf_verify_bsp_and_laying_1.py; that file already exists
in git for the gallops $10-15 cell, so this is a new file.)  Same filters as
scripts/bf_bsp_and_laying.py (dedupe, settled WIN, exactly one winner, every runner
traded pre-off, harness = Pace/Trot in the market name).  Nothing is fitted.
What is re-run / added:

  0. exact reproduction of the cell (bets, paired gain c0/5/6/8/10, iid SE, halves at
     the original median day), strike vs implied at both prices;
  1. cluster-robust SEs by market, meeting (event_id), venue-day, race day and month;
     race-day cluster bootstrap; month-as-unit and year-as-unit t-tests; sign test;
  2. the split: calendar years, half-years, thirds, odd/even ISO weeks, and random
     day-level half splits (how often the both-halves > 2 SE rule holds);
  3. multiple testing: p from each SE against Bonferroni over the 122 paired cells
     and the 556 cells the original touched; BH rank of the cell in its family;
  4. the band: 0.1-wide sub-bands, every ladder tick 2.00..2.98 with its share of the
     gain, the band without its edge ticks, shifted / widened windows, inclusive upper
     edge, by field size, favourite vs not;
  5. influence: dropping the k best winners, winsorised SP/LTP ratio, gain by size of
     the SP move (within one tick, 1-2 ticks, ...), share of the gain from big drifts;
  6. mechanism: strike by SP/LTP move bucket vs implied at each price; log loss of
     the LTP book vs the BSP book (is the SP informative, or just longer?);
  7. execution: SP vs one tick worse / one tick better than the last print / the mid;
     "limit at LTP with SP fallback" (worse of the two); liquidity and staleness
     proxies: runner traded 30 min out, market share with a t30 print, unchanged
     print over 30 min, LTP book and BSP book buckets, venue size, runners with no BSP;
  8. companion cells named in the claim: harness $1-2, harness favourites, harness
     all runners, and the same $2-3 cell on gallops (the claimed sign flip).

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
CELLS_CSV = Path("/tmp/claude-0/-home-user-PredictivePonies/43d94981-d22b-59d1-a1cc-00a2ff7cb49d/scratchpad/bspl_full_cells.csv")
COMMISSIONS = (0.0, 0.05, 0.06, 0.08, 0.10)
C_TEST = 0.05
LO, HI = 2.0, 3.0
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1001.0, 10.0)]
FIELD_EDGES = [2, 6, 9, 12, 99]
FIELD_LABELS = ["2-5", "6-8", "9-11", "12+"]
N_PAIRED_FAMILY = 122
N_CELLS_ORIGINAL = 556
CLAIM = {"bets": 20932, "roi_c5": 0.0064, "se": 0.0013, "halves": [0.0056, 0.0071]}


# ----------------------------------------------------------------------------- utilities


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
    """SE of the mean with observations clustered by ``cl`` (G/(G-1) correction)."""
    n = len(ret)
    if n < 2:
        return float("nan")
    d = ret - ret.mean()
    s = pd.Series(d).groupby(np.asarray(cl)).sum().to_numpy()
    g = len(s)
    if g < 2:
        return float("nan")
    return float(math.sqrt((s ** 2).sum() * g / (g - 1)) / n)


def day_aggregates(ret, days):
    df = pd.DataFrame({"r": ret, "d": np.asarray(days)})
    agg = df.groupby("d")["r"].agg(["sum", "count"])
    agg["ss"] = df.assign(r2=df["r"] ** 2).groupby("d")["r2"].sum()
    return agg["sum"].to_numpy(), agg["ss"].to_numpy(), agg["count"].to_numpy()


def cluster_boot(ret, days, n_boot, seed=0):
    s, _, k = day_aggregates(ret, days)
    g = len(s)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(g, np.full(g, 1.0 / g), size=n_boot)
    means = (w @ s) / (w @ k)
    return {"boot_se": r4(means.std(ddof=1)), "ci95": [r4(np.quantile(means, 0.025)), r4(np.quantile(means, 0.975))],
            "share_boot_below_0": r4((means <= 0).mean()), "clusters": int(g), "reps": int(n_boot)}


def random_day_splits(ret, days, n_splits, seed=1):
    """Assign race days at random to two halves; how often is the mean > 2 SE in both?"""
    s, ss, k = day_aggregates(ret, days)
    g = len(s)
    rng = np.random.default_rng(seed)
    both2, both0, min_z = 0, 0, []
    for _ in range(n_splits):
        perm = rng.permutation(g)
        a = np.zeros(g, dtype=bool)
        a[perm[: g // 2]] = True
        zs = []
        for mask in (a, ~a):
            n = k[mask].sum()
            m = s[mask].sum() / n
            var = (ss[mask].sum() - n * m * m) / (n - 1)
            zs.append(m / math.sqrt(var / n))
        both2 += int(min(zs) > 2)
        both0 += int(min(zs) > 0)
        min_z.append(min(zs))
    min_z = np.array(min_z)
    return {"splits": int(n_splits), "share_both_halves_over_2se": r4(both2 / n_splits),
            "share_both_halves_positive": r4(both0 / n_splits),
            "min_half_z_q10_50_90": [r4(v) for v in np.quantile(min_z, [0.1, 0.5, 0.9])]}


def basic(ret, won=None):
    m, se = mean_se(ret)
    res = {"bets": int(len(ret)), "roi": r4(m), "se": r4(se), "z": r4(m / se) if se and se > 0 else None}
    if won is not None:
        res["n_won"] = int(won.sum())
    return res


def paired(sub, c=C_TEST, lp_col="ltp_pre"):
    """bsp return minus ltp_pre return, 1 unit backed on each (same runner)."""
    won = sub["won_b"].to_numpy()
    return back_returns(sub["bsp"].to_numpy(), won, c) - back_returns(sub[lp_col].to_numpy(), won, c)


def sub_cell(sub, mid=None, c=C_TEST):
    d = paired(sub, c)
    won = sub["won_b"].to_numpy()
    b = basic(d, won)
    if len(sub) > 1:
        b["se_cl_date"] = r4(cluster_se(d, sub["date"].to_numpy()))
        b["z_cl_date"] = r4(d.mean() / cluster_se(d, sub["date"].to_numpy())) if b["se_cl_date"] else None
    if mid is not None and len(sub):
        first = (sub["date"] < mid).to_numpy()
        b["halves"] = [r4(d[first].mean()) if first.any() else None, r4(d[~first].mean()) if (~first).any() else None]
        b["halves_se"] = [r4(mean_se(d[first])[1]), r4(mean_se(d[~first])[1])]
        b["halves_won"] = [int(won[first].sum()), int(won[~first].sum())]
        b["both_halves_over_2se"] = bool(all(h is not None and s is not None and s > 0 and h > 2 * s
                                             for h, s in zip(b["halves"], b["halves_se"])))
    return b


# ----------------------------------------------------------------------------- data


def load(path, end):
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
    # the original's median day is over ALL AU WIN markets (gallops + harness)
    days = sorted(df["date"].unique())
    mid = pd.Timestamp(days[len(days) // 2])
    df["harness"] = df["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b")
    df = df.sort_values(["date", "market_id", "ltp_pre", "selection_id"]).reset_index(drop=True)
    df["rank"] = df.groupby("market_id").cumcount() + 1
    df["fav"] = df["rank"] == 1
    df["won_b"] = df["won"].to_numpy() == 1
    df["bsp_ok"] = df["bsp"].notna() & (df["bsp"] > 1)
    df["has_t30"] = df["ltp_t30"].notna() & (df["ltp_t30"] > 1)
    df["fgroup"] = pd.cut(df["field_size"], FIELD_EDGES, right=False, labels=FIELD_LABELS).astype(str)
    df["book"] = (1.0 / df["ltp_pre"]).groupby(df["market_id"]).transform("sum")
    df["book_bsp"] = (1.0 / df["bsp"].where(df["bsp_ok"])).groupby(df["market_id"]).transform("sum")
    df["mkt_t30_share"] = df["has_t30"].groupby(df["market_id"]).transform("mean")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["half_year"] = df["date"].dt.year.astype(str) + "H" + ((df["date"].dt.month > 6).astype(int) + 1).astype(str)
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
    df["venue_day"] = df["venue"].fillna("?") + "|" + df["date"].astype(str)
    info = {"rows_raw": int(n_raw), "markets": int(df["market_id"].nunique()), "runners": int(len(df)),
            "markets_harness": int(df.loc[df["harness"], "market_id"].nunique()),
            "from": str(pd.Timestamp(days[0]).date()), "to": str(pd.Timestamp(days[-1]).date()),
            "mid_date_for_halves": str(mid.date())}
    return df, info, mid


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--splits", type=int, default=500)
    a = ap.parse_args()
    t0 = time.time()
    df_all, info, mid = load(SPLIT / "au_win.parquet", a.end)
    out({"section": "data", **info, "end": a.end})
    h = df_all[df_all["harness"]].copy()
    venue_markets = h.groupby("venue")["market_id"].nunique()
    h["venue_markets"] = h["venue"].map(venue_markets)
    h["venue_q"] = pd.qcut(h["venue_markets"], 4, labels=["small", "q2", "q3", "large"], duplicates="drop").astype(str)
    out({"section": "code", "code": "AU harness", "markets": int(h["market_id"].nunique()), "runners": int(len(h)),
         "bsp_runner_coverage": r4(h["bsp_ok"].mean()), "t30_runner_coverage": r4(h["has_t30"].mean()),
         "venues": int(h["venue"].nunique())})

    lp_all = h["ltp_pre"].to_numpy()
    in_band = (lp_all >= LO) & (lp_all < HI)
    g = h[in_band & h["bsp_ok"].to_numpy()].copy()
    won = g["won_b"].to_numpy()
    lp_g, bsp_g = g["ltp_pre"].to_numpy(), g["bsp"].to_numpy()
    ratio = bsp_g / lp_g
    g["ratio"] = ratio
    d5 = paired(g, C_TEST)
    first = (g["date"] < mid).to_numpy()

    # ---- 0. reproduction ------------------------------------------------------------------------
    rep = {f"paired_c{int(round(c * 100))}": basic(paired(g, c)) for c in COMMISSIONS}
    rep["back_ltp_pre"] = {f"c{int(round(c * 100))}": r4(back_returns(lp_g, won, c).mean()) for c in COMMISSIONS}
    rep["back_bsp"] = {f"c{int(round(c * 100))}": r4(back_returns(bsp_g, won, c).mean()) for c in COMMISSIONS}
    rep["back_ltp_tick_down"] = {f"c{int(round(c * 100))}": r4(back_returns(tick_down(lp_g), won, c).mean()) for c in COMMISSIONS}
    rep["halves_c5"] = [basic(d5[first], won[first]), basic(d5[~first], won[~first])]
    rep["both_halves_over_2se"] = bool(all(b["z"] is not None and b["z"] > 2 for b in rep["halves_c5"]))
    rep["strike"] = r4(won.mean())
    rep["implied_ltp"] = r4((1.0 / lp_g).mean())
    rep["implied_bsp"] = r4((1.0 / bsp_g).mean())
    rep["mean_bsp_minus_ltp_all"] = r4((bsp_g - lp_g).mean())
    rep["mean_bsp_minus_ltp_given_won"] = r4((bsp_g - lp_g)[won].mean())
    rep["mean_bsp_minus_ltp_given_lost"] = r4((bsp_g - lp_g)[~won].mean())
    rep["median_ratio_all"] = r4(np.median(ratio))
    rep["mean_log_ratio_all"] = r4(np.log(ratio).mean())
    rep["mean_log_ratio_won"] = r4(np.log(ratio[won]).mean())
    rep["mean_log_ratio_lost"] = r4(np.log(ratio[~won]).mean())
    rep["gain_if_won_independent_of_move_c5"] = r4(won.mean() * (bsp_g - lp_g).mean() * (1 - C_TEST))
    rep["share_sp_better_1tick"] = r4((bsp_g >= tick_up(lp_g) - 1e-9).mean())
    rep["share_sp_worse_1tick"] = r4((bsp_g <= tick_down(lp_g) + 1e-9).mean())
    rep["share_sp_within_1tick"] = r4(((bsp_g < tick_up(lp_g) - 1e-9) & (bsp_g > tick_down(lp_g) + 1e-9)).mean())
    rep["share_sp_better_1tick_winners"] = r4((bsp_g >= tick_up(lp_g) - 1e-9)[won].mean())
    rep["share_sp_worse_1tick_winners"] = r4((bsp_g <= tick_down(lp_g) + 1e-9)[won].mean())
    rep["band_runners_without_bsp"] = int((in_band & ~h["bsp_ok"].to_numpy()).sum())
    rep["claimed"] = CLAIM
    rep["matches_claim"] = bool(rep["paired_c5"]["bets"] == CLAIM["bets"] and abs(rep["paired_c5"]["roi"] - CLAIM["roi_c5"]) < 2e-4)
    out({"section": "0_reproduction_harness_ltp_2_3", **rep,
         "note": "paired = back 1 at bsp minus back 1 at ltp_pre on the same runner; nonzero only on winners"})

    # ---- 1. clustered SEs, bootstrap, month / year as unit ---------------------------------------
    m5 = float(d5.mean())
    cl = {"mean_c5": r4(m5), "se_iid": r4(mean_se(d5)[1])}
    for name, col in (("market", "market_id"), ("event", "event_id"), ("venue_day", "venue_day"), ("date", "date"), ("month", "month")):
        se = cluster_se(d5, g[col].to_numpy())
        cl[f"se_cl_{name}"] = r4(se)
        cl[f"z_cl_{name}"] = r4(m5 / se)
        cl[f"clusters_{name}"] = int(g[col].nunique())
    cl["boot_by_date"] = cluster_boot(d5, g["date"].to_numpy(), a.boot)
    # binomial null SE as in the original: var = q(1-q)(r_won - r_lost)^2 with q = 1/ltp_pre
    q = 1.0 / lp_g
    r_won = (bsp_g - lp_g) * (1 - C_TEST)
    se_null = math.sqrt(float((q * (1 - q) * r_won ** 2).sum())) / len(g)
    cl["se_null_binomial_as_original"] = r4(se_null)
    cl["z_null"] = r4(m5 / se_null)
    cl["realised_sum_sq_on_winners_over_null_expectation"] = r4(float((r_won[won] ** 2).sum()) / float((q * (1 - q) * r_won ** 2).sum()))
    for unit in ("month", "year", "half_year"):
        per = pd.DataFrame({"d": d5, "u": g[unit].to_numpy()}).groupby("u")["d"].agg(["mean", "count", "sum"])
        mu = per["mean"].to_numpy()
        n_u = len(mu)
        t = mu.mean() / (mu.std(ddof=1) / math.sqrt(n_u)) if n_u > 2 else float("nan")
        pos = int((mu > 0).sum())
        # exact binomial sign test, one-sided
        p_sign = sum(math.comb(n_u, k) for k in range(pos, n_u + 1)) / 2 ** n_u if n_u else float("nan")
        cl[f"{unit}_as_unit"] = {"units": int(n_u), "mean_of_unit_means": r4(mu.mean()), "t": r4(t),
                                 "units_positive": pos, "p_sign_one_sided": r4(p_sign),
                                 "unit_means_rounded": [round(float(v), 4) for v in mu] if unit != "month" else None}
    out({"section": "1_clustered_se_and_units", **cl})

    # ---- 2. splits -------------------------------------------------------------------------------
    sp = {"original_mid_date": str(mid.date())}
    for unit in ("year", "half_year"):
        sp[f"by_{unit}"] = {str(k): sub_cell(g[g[unit] == k]) for k in sorted(g[unit].unique())}
    # thirds by date
    days = np.sort(g["date"].unique())
    cuts = [days[len(days) // 3], days[2 * len(days) // 3]]
    third = np.where(g["date"] < cuts[0], 0, np.where(g["date"] < cuts[1], 1, 2))
    sp["thirds"] = [sub_cell(g[third == i]) for i in range(3)]
    sp["odd_even_iso_week"] = [sub_cell(g[g["iso_week"] % 2 == i]) for i in (1, 0)]
    sp["random_day_splits"] = random_day_splits(d5, g["date"].to_numpy(), a.splits)
    out({"section": "2_splits", **sp})

    # ---- 3. multiple testing ---------------------------------------------------------------------
    mt = {}
    for name, se in (("iid", cl["se_iid"]), ("cl_date", cl["se_cl_date"]), ("cl_month", cl["se_cl_month"]),
                     ("boot_date", cl["boot_by_date"]["boot_se"]), ("null_binomial", se_null)):
        p = one_sided_p(m5 / se)
        mt[name] = {"se": r4(se), "z": r4(m5 / se), "p": p,
                    "bonferroni_122": bool(p < 0.05 / N_PAIRED_FAMILY), "bonferroni_556": bool(p < 0.05 / N_CELLS_ORIGINAL)}
    if CELLS_CSV.exists():
        cells = pd.read_csv(CELLS_CSV)
        fam = cells[cells["family"] == "sp_minus_ltp"].copy()
        fam["p"] = pd.to_numeric(fam["p"], errors="coerce")
        fam = fam.sort_values("p").reset_index(drop=True)
        hit = fam[(fam["code"] == "harness") & (fam["name"] == "band 2-3 field all")]
        if len(hit):
            k = int(hit.index[0]) + 1
            m_test = int(fam["p"].notna().sum())
            mt["bh_rank_in_family"] = {"rank": k, "testable": m_test, "p_cell": r4(hit["p"].iloc[0]),
                                       "bh_threshold_at_rank": r4(0.05 * k / m_test),
                                       "harness_cells_in_family": int((fam["code"] == "harness").sum()),
                                       "harness_cells_p_below_0.05": int(((fam["code"] == "harness") & (fam["p"] < 0.05)).sum())}
    out({"section": "3_multiple_testing", **mt})

    # ---- 4. band robustness ----------------------------------------------------------------------
    hb = h[h["bsp_ok"]].copy()
    lph = hb["ltp_pre"].to_numpy()
    band = {}
    band["sub_bands_0.25"] = {f"[{lo:.2f},{lo + 0.25:.2f})": sub_cell(hb[(lph >= lo) & (lph < lo + 0.25)], mid)
                              for lo in np.arange(2.0, 3.0, 0.25)}
    ticks = {}
    total_gain = float(d5.sum())
    for t in np.round(np.arange(2.0, 3.0, 0.02), 2):
        m = np.isclose(lp_g, t)
        if m.any():
            ticks[f"{t:.2f}"] = [int(m.sum()), int(won[m].sum()), r4(d5[m].mean()), r4(d5[m].sum() / total_gain),
                                 r4(np.log(ratio[m]).mean()), r4(np.log(ratio[m][won[m]]).mean()) if won[m].any() else None]
    band["per_tick [n, won, gain_c5, share_of_total_gain, mean_log_ratio_all, mean_log_ratio_won]"] = ticks
    band["prices_not_on_ladder"] = int((~np.isin(np.round(lp_g, 2), np.round(np.arange(2.0, 3.0, 0.02), 2))).sum())
    band["without_tick_2.00"] = sub_cell(g[~np.isclose(lp_g, 2.0)], mid)
    band["without_ticks_2.00_and_2.98"] = sub_cell(g[~np.isclose(lp_g, 2.0) & ~np.isclose(lp_g, 2.98)], mid)
    band["inclusive_upper_[2,3]"] = sub_cell(hb[(lph >= 2.0) & (lph <= 3.0)], mid)
    band["shifted_windows"] = {f"[{lo:g},{hi:g})": sub_cell(hb[(lph >= lo) & (lph < hi)], mid)
                               for lo, hi in ((1.5, 2.5), (1.8, 2.8), (2.2, 3.2), (2.5, 3.5), (3.0, 4.0), (1.5, 3.5), (2.0, 4.0))}
    band["by_field_size"] = {f: sub_cell(g[g["fgroup"] == f], mid) for f in FIELD_LABELS}
    band["favourite_vs_not"] = {"favourite": sub_cell(g[g["fav"]], mid), "not_favourite": sub_cell(g[~g["fav"]], mid)}
    band["by_rank"] = {str(r): sub_cell(g[g["rank"] == r], mid) for r in (1, 2, 3)}
    out({"section": "4_band_robustness", **band})

    # ---- 5. influence ----------------------------------------------------------------------------
    inf = {}
    order = np.argsort(-d5)
    for k in (1, 5, 10, 25, 50, 100):
        keep = np.ones(len(d5), dtype=bool)
        keep[order[:k]] = False
        inf[f"drop_top_{k}_winners"] = basic(d5[keep])
    inf["top_10_winner_gains"] = [round(float(v), 3) for v in d5[order[:10]]]
    for lo, hi in ((0.9, 1.1), (0.8, 1.25), (0.95, 1.05)):
        rw = np.clip(ratio, lo, hi)
        dw = back_returns(lp_g * rw, won, C_TEST) - back_returns(lp_g, won, C_TEST)
        inf[f"winsorised_ratio_[{lo},{hi}]"] = basic(dw)
    move_t = (bsp_g - lp_g) / 0.02  # in ticks of the 2-3 ladder
    buckets = {"sp_worse_>5_ticks": move_t <= -5, "sp_worse_2-5_ticks": (move_t > -5) & (move_t <= -2),
               "sp_worse_1-2_ticks": (move_t > -2) & (move_t <= -1), "within_1_tick": (move_t > -1) & (move_t < 1),
               "sp_better_1-2_ticks": (move_t >= 1) & (move_t < 2), "sp_better_2-5_ticks": (move_t >= 2) & (move_t < 5),
               "sp_better_>5_ticks": move_t >= 5}
    inf["by_move [n, won, strike, implied_ltp, implied_bsp, gain_c5_total_share, mean_gain_c5]"] = {
        k: [int(m.sum()), int(won[m].sum()), r4(won[m].mean()), r4((1 / lp_g[m]).mean()), r4((1 / bsp_g[m]).mean()),
            r4(d5[m].sum() / total_gain), r4(d5[m].mean())] for k, m in buckets.items() if m.any()}
    for thr in (1.05, 1.1, 1.2):
        inf[f"share_of_gain_from_ratio_over_{thr}"] = r4(d5[ratio > thr].sum() / total_gain)
        inf[f"share_of_gain_lost_to_ratio_under_{round(1 / thr, 3)}"] = r4(d5[ratio < 1 / thr].sum() / total_gain)
    inf["gain_within_1_tick_only"] = basic(d5[buckets["within_1_tick"]])
    # symmetric trimming: drop the k most positive AND the k most negative paired differences
    order_lo = np.argsort(d5)
    for k in (5, 10, 25, 50, 100):
        keep = np.ones(len(d5), dtype=bool)
        keep[order[:k]] = False
        keep[order_lo[:k]] = False
        inf[f"symmetric_trim_{k}_each_side"] = basic(d5[keep])
    inf["bottom_10_winner_gains"] = [round(float(v), 3) for v in d5[order_lo[:10]]]
    # the gain as a function of a cap on the size of the SP move (|log ratio| <= x): runners inside the cap only,
    # and all runners with the ratio winsorised to the cap
    caps = {}
    for x in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        inside = np.abs(np.log(ratio)) <= x
        rw = np.clip(ratio, math.exp(-x), math.exp(x))
        dw = back_returns(lp_g * rw, won, C_TEST) - back_returns(lp_g, won, C_TEST)
        caps[f"{x:g}"] = {"share_inside": r4(inside.mean()), "inside_only": basic(d5[inside]), "winsorised_all": basic(dw)}
    inf["by_move_cap"] = caps
    out({"section": "5_influence", **inf})

    # ---- 6. mechanism ----------------------------------------------------------------------------
    mech = {}
    edges = [0, 0.85, 0.93, 0.97, 0.99, 1.01, 1.03, 1.07, 1.15, 10]
    lab = [f"[{lo},{hi})" for lo, hi in zip(edges, edges[1:])]
    rb = pd.cut(ratio, edges, right=False, labels=lab).astype(str)
    mech["strike_by_ratio_bucket [n, strike, implied_ltp, implied_bsp, back_c0_ltp, back_c0_bsp, gain_c5]"] = {
        b: [int((rb == b).sum()), r4(won[rb == b].mean()), r4((1 / lp_g[rb == b]).mean()), r4((1 / bsp_g[rb == b]).mean()),
            r4(back_returns(lp_g[rb == b], won[rb == b], 0.0).mean()), r4(back_returns(bsp_g[rb == b], won[rb == b], 0.0).mean()),
            r4(d5[rb == b].mean())] for b in lab if (rb == b).any()}
    # log loss of the LTP book vs the BSP book on harness markets where every runner has a BSP
    cnt_h = h.groupby("market_id")["selection_id"].count()
    cnt_b = hb.groupby("market_id")["selection_id"].count().reindex(cnt_h.index).fillna(0)
    all_bsp = cnt_b == cnt_h
    mk = hb[hb["market_id"].isin(all_bsp[all_bsp].index)]
    books = {}
    for src in ("ltp_pre", "bsp"):
        inv = 1.0 / mk[src]
        tot = inv.groupby(mk["market_id"]).transform("sum")
        pr = inv / tot
        mech[f"log_loss_{src}_book"] = r4(-np.log(pr[mk["won_b"]].to_numpy()).mean())
        books[src] = tot.groupby(mk["market_id"]).first()
    mech["log_loss_markets"] = int(mk["market_id"].nunique())
    mech["median_book_ltp_vs_bsp_all_bsp_markets"] = [r4(books["ltp_pre"].median()), r4(books["bsp"].median())]
    mech["mean_book_ltp_vs_bsp_all_bsp_markets"] = [r4(books["ltp_pre"].mean()), r4(books["bsp"].mean())]
    # staleness contrast: how often is the SP more than 10% / 20% away from the last print?
    gal_b = df_all[~df_all["harness"] & df_all["bsp_ok"] & (df_all["ltp_pre"] >= LO) & (df_all["ltp_pre"] < HI)]
    lr_g = np.log(gal_b["bsp"].to_numpy() / gal_b["ltp_pre"].to_numpy())
    lr_h = np.log(ratio)
    mech["share_abs_log_ratio_over_0.1 harness_2_3 vs gallops_2_3"] = [r4((np.abs(lr_h) > 0.1).mean()), r4((np.abs(lr_g) > 0.1).mean())]
    mech["share_abs_log_ratio_over_0.2 harness_2_3 vs gallops_2_3"] = [r4((np.abs(lr_h) > 0.2).mean()), r4((np.abs(lr_g) > 0.2).mean())]
    mech["share_abs_log_ratio_over_0.05 harness_2_3 vs gallops_2_3"] = [r4((np.abs(lr_h) > 0.05).mean()), r4((np.abs(lr_g) > 0.05).mean())]
    out({"section": "6_mechanism", **mech})

    # ---- 7. execution ----------------------------------------------------------------------------
    ex = {}
    for name, px in (("sp_vs_tick_down_ltp (backer matched a tick worse)", tick_down(lp_g)),
                     ("sp_vs_tick_up_ltp (backer got a tick better, patient offer)", tick_up(lp_g)),
                     ("sp_vs_mid_of_ltp_and_tick_up", (lp_g + tick_up(lp_g)) / 2),
                     ("sp_vs_mid_of_ltp_and_tick_down", (lp_g + tick_down(lp_g)) / 2)):
        dd = back_returns(bsp_g, won, C_TEST) - back_returns(px, won, C_TEST)
        ex[name] = basic(dd)
    worse = np.minimum(bsp_g, lp_g)
    ex["limit_at_ltp_with_sp_fallback (worse of the two) vs ltp"] = basic(back_returns(worse, won, C_TEST) - back_returns(lp_g, won, C_TEST))
    ex["sp vs limit_at_ltp_with_sp_fallback"] = basic(back_returns(bsp_g, won, C_TEST) - back_returns(worse, won, C_TEST))
    ex["best_of_two vs ltp (hindsight bound)"] = basic(back_returns(np.maximum(bsp_g, lp_g), won, C_TEST) - back_returns(lp_g, won, C_TEST))
    liq = {}
    liq["runner_traded_30min_out"] = {"yes": sub_cell(g[g["has_t30"]], mid), "no": sub_cell(g[~g["has_t30"]], mid)}
    t30 = g[g["has_t30"]]
    same = np.isclose(t30["ltp_pre"].to_numpy(), t30["ltp_t30"].to_numpy())
    liq["print_unchanged_in_30min (stale candidate, conditional on t30)"] = {"unchanged": sub_cell(t30[same], mid), "changed": sub_cell(t30[~same], mid)}
    mts = g["mkt_t30_share"].to_numpy()
    liq["market_share_of_runners_with_t30"] = {"0": sub_cell(g[mts == 0], mid), "(0,0.5]": sub_cell(g[(mts > 0) & (mts <= 0.5)], mid),
                                               ">0.5": sub_cell(g[mts > 0.5], mid)}
    bk = g["book"].to_numpy()
    liq["ltp_book_bucket"] = {k: sub_cell(g[m], mid) for k, m in (("<0.98", bk < 0.98), ("[0.98,1.0)", (bk >= 0.98) & (bk < 1.0)),
                                                                   ("[1.0,1.02)", (bk >= 1.0) & (bk < 1.02)), ("[1.02,1.05)", (bk >= 1.02) & (bk < 1.05)),
                                                                   (">=1.05", bk >= 1.05))}
    liq["ltp_book_quantiles"] = [r4(v) for v in np.quantile(bk, [0.1, 0.5, 0.9])]
    bb = g["book_bsp"].to_numpy()
    liq["bsp_book_bucket"] = {k: sub_cell(g[m], mid) for k, m in (("<0.99", bb < 0.99), ("[0.99,1.01)", (bb >= 0.99) & (bb < 1.01)),
                                                                   ("[1.01,1.03)", (bb >= 1.01) & (bb < 1.03)), (">=1.03", bb >= 1.03))}
    liq["venue_size_quartile"] = {k: sub_cell(g[g["venue_q"] == k], mid) for k in ("small", "q2", "q3", "large")}
    liq["by_field_size_n"] = {str(fs): sub_cell(g[g["field_size"] == fs]) for fs in sorted(g["field_size"].unique()) if (g["field_size"] == fs).sum() >= 300}
    nob = h[in_band & ~h["bsp_ok"].to_numpy()]
    liq["band_runners_without_bsp"] = {"n": int(len(nob)), "strike": r4(nob["won_b"].mean()) if len(nob) else None,
                                       "back_ltp_c5": r4(back_returns(nob["ltp_pre"].to_numpy(), nob["won_b"].to_numpy(), C_TEST).mean()) if len(nob) else None,
                                       "with_bsp_back_ltp_c5": r4(back_returns(lp_g, won, C_TEST).mean())}
    ex["liquidity_and_staleness"] = liq
    out({"section": "7_execution", **ex})

    # ---- 8. companion cells ----------------------------------------------------------------------
    comp = {}
    comp["harness_1_2"] = sub_cell(hb[(lph >= 1.0) & (lph < 2.0)], mid)
    comp["harness_favourites_all"] = sub_cell(hb[hb["fav"]], mid)
    comp["harness_all_runners"] = sub_cell(hb, mid)
    comp["harness_3_4.5"] = sub_cell(hb[(lph >= 3.0) & (lph < 4.5)], mid)
    for name, sub in (("harness_1_2", hb[(lph >= 1.0) & (lph < 2.0)]), ("harness_favourites_all", hb[hb["fav"]])):
        d = paired(sub, C_TEST)
        comp[name]["se_cl_month"] = r4(cluster_se(d, sub["month"].to_numpy()))
        comp[name]["random_day_splits"] = random_day_splits(d, sub["date"].to_numpy(), a.splits)
        comp[name]["by_year"] = {str(y): r4(d[(sub["year"] == y).to_numpy()].mean()) for y in sorted(sub["year"].unique())}
    gal = df_all[~df_all["harness"] & df_all["bsp_ok"]]
    lpg = gal["ltp_pre"].to_numpy()
    comp["gallops_2_3_same_filters"] = sub_cell(gal[(lpg >= LO) & (lpg < HI)], mid)
    comp["gallops_favourites_all"] = sub_cell(gal[gal["fav"]], mid)
    out({"section": "8_companion_cells", **comp})

    # ---- 9. the "live print" regime: SP within +/-10% of the last print, by period and book; gallops contrast --
    reg = {}
    inside = np.abs(np.log(ratio)) <= 0.1
    gi = g[inside]
    reg["harness_2_3_inside_10pct"] = sub_cell(gi, mid)
    reg["harness_2_3_inside_10pct_by_year"] = {str(y): basic(paired(gi[gi["year"] == y])) for y in sorted(gi["year"].unique())}
    reg["harness_2_3_inside_10pct_random_day_splits"] = random_day_splits(paired(gi), gi["date"].to_numpy(), a.splits)
    bki = gi["book"].to_numpy()
    reg["harness_2_3_inside_10pct_by_ltp_book"] = {k: basic(paired(gi[m])) for k, m in (("<0.98", bki < 0.98), ("[0.98,1.02)", (bki >= 0.98) & (bki < 1.02)), (">=1.02", bki >= 1.02))}
    reg["harness_2_3_outside_10pct"] = sub_cell(g[~inside], mid)
    reg["harness_2_3_outside_10pct_by_year"] = {str(y): basic(paired(g[~inside][g[~inside]["year"] == y])) for y in sorted(g["year"].unique())}
    reg["harness_2_3_outside_10pct_split_drift_vs_firm"] = {"sp_longer_>10pct": basic(d5[np.log(ratio) > 0.1]), "sp_shorter_>10pct": basic(d5[np.log(ratio) < -0.1])}
    # the same decomposition on gallops $2-3 (liquid markets, fresh prints) and on harness favourites / $1-2
    for name, sub in (("gallops_2_3", gal_b), ("harness_favourites_all", hb[hb["fav"]]), ("harness_1_2", hb[(lph >= 1.0) & (lph < 2.0)])):
        r = sub["bsp"].to_numpy() / sub["ltp_pre"].to_numpy()
        ins = np.abs(np.log(r)) <= 0.1
        reg[name] = {"share_inside_10pct": r4(ins.mean()), "inside": basic(paired(sub[ins])), "outside": basic(paired(sub[~ins])),
                     "all": basic(paired(sub))}
    out({"section": "9_live_print_regime", **reg})
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

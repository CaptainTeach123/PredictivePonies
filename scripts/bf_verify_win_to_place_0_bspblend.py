"""Adversarial verification of the "win_to_place" BENCHMARK claim: on the 41,658
fully priced AU place markets with a BSP for every runner, 0.75 * (place BSP
implied, proportionally normalised to n_winners) + 0.25 * p_cal (calibrated
WIN-ltp-derived place probability) beats the place BSP by 0.0023 log loss per
race (SE 0.0003; halves 0.0037 / 0.0012), read as "the win-derived price carries
information beyond the final place market". Lens: leakage and data quality.

    python scripts/bf_verify_win_to_place_0_bspblend.py [--dir data/raw/betfair/split] [--frac 0.1]

Reuses the original loader / lambda fit / calibration unchanged (imported from
scripts/bf_win_to_place.py) so the reproduction is exact, then asks whether the
gain is INFORMATION or merely RE-CALIBRATION of a proportionally normalised
post-race book:
  1. exact reproduction, all three fixed weights the original tried, the full
     weight grid in both halves, a walk-forward fitted weight, cluster-by-date
     SE, median / winsorised gain, top-1% share, by code and by year
  2. data quality: BSP book / n_winners tightness, extreme BSPs, dead heats,
     gain by tightness bucket and after excluding loose books
  3. calibration of the BSP-implied probabilities by band (is q_bsp itself
     miscalibrated?) and two fit-free re-normalisations (power, Shin-like)
  4. the same walk-forward cubic-logit calibration applied to q_bsp itself
     (q_bsp_cal), then blend / grid against q_bsp_cal
  5. nested walk-forward logistic models: cubic(logit q_bsp) with and without
     logit(p_cal) [and logit(place ltp market)] -> incremental gain of p_cal
     beyond a calibrated place BSP, per half
  6. the same nested test with a place probability derived from the WIN BSP
     (post-race) instead of the WIN ltp
Prints summary numbers only.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import bf_win_to_place as O  # noqa: E402

r4 = O.r4
BLEND_GRID = O.BLEND_GRID


def out(obj) -> None:
    print(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)), flush=True)


def gain_stats(ll_model: np.ndarray, ll_base: np.ndarray, dates: np.ndarray, mid, extra_masks: dict | None = None) -> dict:
    """gain = base - model per race: mean, SE, cluster-by-date SE, halves (with
    SE), median, winsorised, top-1% share, by year, optional extra subsets."""
    d = ll_base - ll_model
    N = len(d)
    if N < 2:
        return {"races": int(N)}
    dates = pd.DatetimeIndex(dates)
    fh = np.asarray(dates < mid)
    res = O.paired(ll_model, ll_base, fh)
    s = pd.Series(d).groupby(dates.to_numpy()).sum()
    n = pd.Series(d).groupby(dates.to_numpy()).size()
    m = d.mean()
    res["gain_se_cluster_date"] = r4(math.sqrt(((s - m * n) ** 2).sum()) / N)
    res["z_plain"] = r4(m / (d.std(ddof=1) / math.sqrt(N)))
    res["z_cluster"] = r4(m / res["gain_se_cluster_date"]) if res["gain_se_cluster_date"] else None
    res["median_gain"] = r4(np.median(d))
    res["share_races_gain_pos"] = r4((d > 0).mean())
    lo, hi = np.percentile(d, [1, 99])
    res["winsor_1_99_gain"] = r4(np.clip(d, lo, hi).mean())
    k = max(1, N // 100)
    res["top1pct_share_of_total_gain"] = r4(np.sort(d)[-k:].sum() / d.sum()) if d.sum() != 0 else None
    yr = dates.year.to_numpy()
    res["gain_by_year"] = {int(y): [r4(d[yr == y].mean()), r4(d[yr == y].std(ddof=1) / math.sqrt((yr == y).sum())), int((yr == y).sum())]
                           for y in sorted(set(yr)) if (yr == y).sum() > 1}
    res["halves_pass_2se"] = bool(all(g is not None and se and g > 2 * se for g, se in zip(res["halves_gain"], res["halves_gain_se"])))
    if extra_masks:
        for name, mk in extra_masks.items():
            mk = np.asarray(mk, dtype=bool)
            if mk.sum() > 1:
                dd, ff, yy = d[mk], fh[mk], yr[mk]
                h = [(r4(dd[s].mean()), r4(dd[s].std(ddof=1) / math.sqrt(s.sum()))) if s.sum() > 1 else (None, None) for s in (ff, ~ff)]
                res[f"gain_{name}"] = {"gain": r4(dd.mean()), "se": r4(dd.std(ddof=1) / math.sqrt(len(dd))), "races": int(len(dd)),
                                       "halves": [h[0][0], h[1][0]], "halves_se": [h[0][1], h[1][1]],
                                       "pass_2se_both_halves": bool(all(g is not None and se and g > 2 * se for g, se in h)),
                                       "by_year": {int(y): r4(dd[yy == y].mean()) for y in sorted(set(yy)) if (yy == y).sum() > 1}}
    return res


def norm_nw(df: pd.DataFrame, col: str) -> np.ndarray:
    inv = 1 / df[col]
    return (inv * df["n_winners"] / inv.groupby(df["race"]).transform("sum")).to_numpy()


def padded(df: pd.DataFrame, col: str):
    rc = pd.factorize(df["race"])[0]
    pos = df.groupby("race").cumcount().to_numpy()
    R, W = rc.max() + 1, pos.max() + 1
    M = np.zeros((R, W))
    M[rc, pos] = df[col].to_numpy(dtype=float)
    return M, rc, pos


def power_norm(df: pd.DataFrame, col: str) -> np.ndarray:
    """q_i = (1/price_i)^k with k per race so that sum = n_winners (longshots shrunk more when the book is over)."""
    M, rc, pos = padded(df, col)
    inv = np.where(M > 0, 1 / np.where(M > 0, M, 1), 0.0)
    nw = np.zeros(M.shape[0]); nw[rc] = df["n_winners"].to_numpy(dtype=float)
    lo, hi = np.full(M.shape[0], 0.2), np.full(M.shape[0], 4.0)
    for _ in range(50):
        mid = (lo + hi) / 2
        s = (inv ** mid[:, None]).sum(1)
        over = s > nw
        lo = np.where(over, mid, lo); hi = np.where(over, hi, mid)
    k = (lo + hi) / 2
    Q = inv ** k[:, None]
    Q = Q * (nw / Q.sum(1))[:, None]
    return Q[rc, pos], k[rc]


def wf_logit(df: pd.DataFrame, feats: list[str], months_sorted: list[str], cubic: str | None, window=12, min_rows=5000):
    """Walk-forward logistic regression of placed on logit(feature) columns
    (cubic polynomial in ``cubic``, linear in the rest; standardised on the
    training window), refit monthly on the previous ``window`` months, then
    rescaled within the race to sum to n_winners (as the original calibrator)."""
    Z = {}
    for f in feats:
        p = df[f].clip(1e-6, 1 - 1e-6).to_numpy()
        Z[f] = np.log(p / (1 - p))
    y = df["placed"].to_numpy(dtype=float)
    month = df["month"].to_numpy()
    outp = np.full(len(df), np.nan)
    coefs = {}

    def design(sel, stats):
        cols = [np.ones(sel.sum())]
        for f in feats:
            z = (Z[f][sel] - stats[f][0]) / stats[f][1]
            cols.append(z)
            if f == cubic:
                cols += [z ** 2, z ** 3]
        return np.column_stack(cols)

    for idx, m in enumerate(months_sorted):
        tr = np.isin(month, months_sorted[max(0, idx - window):idx])
        if tr.sum() < min_rows:
            continue
        stats = {f: (Z[f][tr].mean(), Z[f][tr].std()) for f in feats}
        X = design(tr, stats)
        w = np.zeros(X.shape[1])
        for _ in range(15):
            mu = 1 / (1 + np.exp(-(X @ w)))
            g = X.T @ (mu - y[tr])
            H = (X * (mu * (1 - mu))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
            w = w - np.linalg.solve(H, g)
        coefs[m] = [round(float(v), 3) for v in w]
        te = month == m
        outp[te] = 1 / (1 + np.exp(-(design(te, stats) @ w)))
    p = pd.Series(outp, index=df.index)
    scale = df["n_winners"] / p.groupby(df["race"]).transform("sum")
    return (p * scale).clip(upper=0.999).to_numpy(), coefs


def calib_rows(df: pd.DataFrame, cols: list[str], by: str) -> list[dict]:
    bands = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    b = pd.cut(df[by], bands, right=False)
    rows = []
    for bb, g in df.groupby(b, observed=True):
        row = {"band_" + by: str(bb), "runners": int(len(g)), "strike": r4(g["placed"].mean())}
        for c in cols:
            row[c] = r4(g[c].mean())
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/raw/betfair/split"))
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    # ------------------------------------------------------------ exact reproduction of the original pipeline
    j = O.load(a.dir, a.frac, a.seed)
    P, jw, k, i, ok, nw, race = O.placed_set_stats(j)
    months = race["month"].to_numpy()
    codes = race["harness"].to_numpy()
    months_sorted = sorted(set(months))
    ll = O.grid_loglik(P, jw, k, i, ok, nw)
    lam = O.fit_lambdas_walk_forward(ll, ok, months, codes, months_sorted)
    del ll
    j["p_fair"] = O.derived_place_probs(j, P, nw, lam, months, codes)
    # (6) same Harville machinery from the WIN BSP (post-race) for the consistency check
    jb_ok = j.groupby("win_mid")["bsp_win"].transform(lambda s: s.notna().all() and (s > 1).all())
    jb_ok = jb_ok.to_numpy().astype(bool)
    j["p_win_bsp"] = np.where(jb_ok, (1 / j["bsp_win"]) / j.groupby("win_mid")["bsp_win"].transform(lambda s: (1 / s).sum()), j["p_win"])
    Pb = O.pad(j, "p_win_bsp")
    j["p_fair_wbsp"] = O.derived_place_probs(j, Pb, nw, lam, months, codes)
    j.loc[~jb_ok, "p_fair_wbsp"] = np.nan
    j = j[j["p_fair"].notna()].copy()
    j["race"] = pd.factorize(j["win_mid"])[0]
    j["dh"] = j["n_winners"] / j["n_placed"]
    j["p_cal"], cal_params = O.calibrate_walk_forward(j, "p_fair", sorted(j["month"].unique()))
    days = sorted(j["date"].unique())
    mid = days[len(days) // 2]
    grp = j.groupby("race")
    j["all_priced"] = grp["ltp_pl"].transform(lambda s: s.notna().all() and (s > 1).all())
    full = j[j["all_priced"]].copy()
    full["q_mkt"] = norm_nw(full, "ltp_pl")
    fb = full.copy()
    fb["all_bsp"] = fb.groupby("race")["bsp_pl"].transform(lambda s: s.notna().all() and (s > 1).all())
    fb = fb[fb["all_bsp"] & fb["p_cal"].notna()].copy()
    fb["q_bsp"] = norm_nw(fb, "bsp_pl")
    fb["book_nw"] = (1 / fb["bsp_pl"]).groupby(fb["race"]).transform("sum") / fb["n_winners"]
    fb["blend75"] = 0.75 * fb["q_bsp"] + 0.25 * fb["p_cal"]
    yb = fb["placed"].to_numpy(dtype=float)
    rcb = pd.factorize(fb["race"])[0]
    rr = fb.drop_duplicates("race")
    dates_b = rr["date"].to_numpy()
    harness_b = rr["harness"].to_numpy(dtype=bool)
    nw_b = rr["n_winners"].to_numpy()
    book_b = rr["book_nw"].to_numpy()
    dead_b = (rr["n_placed"] > rr["n_winners"]).to_numpy()
    maxbsp_b = fb.groupby("race")["bsp_pl"].max().reindex(rr["race"]).to_numpy()
    minbsp_b = fb.groupby("race")["bsp_pl"].min().reindex(rr["race"]).to_numpy()
    rmonth = rr["month"].to_numpy()
    fb["abs_log_bsp_ltp"] = np.log(fb["bsp_pl"] / fb["ltp_pl"]).abs()
    maxdev_b = fb.groupby("race")["abs_log_bsp_ltp"].max().reindex(rr["race"]).to_numpy()

    LL = lambda col: O.logloss_race(fb[col].to_numpy(), yb, rcb)  # noqa: E731
    ll_bsp = LL("q_bsp"); ll_cal = LL("p_cal"); ll_mkt = LL("q_mkt"); ll_b75 = LL("blend75")
    masks = {"gallops": ~harness_b, "harness": harness_b, "places_3": nw_b == 3, "places_2": nw_b == 2,
             "book_tight_0.97_1.04": (book_b >= 0.97) & (book_b <= 1.04), "book_loose_lt_0.97": book_b < 0.97,
             "book_loose_gt_1.04": book_b > 1.04, "no_dead_heat": ~dead_b, "max_bsp_lt_100": maxbsp_b < 100,
             "max_bsp_ge_100": maxbsp_b >= 100, "max_bsp_lt_50": maxbsp_b < 50,
             "bsp_within_30pct_of_ltp_all_runners": maxdev_b < 0.3, "bsp_far_from_ltp_some_runner": maxdev_b >= 0.3,
             "bsp_far_from_ltp_gt_0.7": maxdev_b >= 0.7,
             "clean_bsp_within_30pct_and_max_bsp_lt_50": (maxdev_b < 0.3) & (maxbsp_b < 50),
             "clean_bsp_within_50pct_all_runners": maxdev_b < 0.5}
    out({"section": "0_data", "runtime_s": round(time.time() - t0, 1), "joined_races": int(j["race"].nunique()),
         "fully_priced_races": int(full["race"].nunique()), "bsp_benchmark_races": int(len(ll_bsp)),
         "mid": str(pd.Timestamp(mid).date()), "harness_share": r4(harness_b.mean()),
         "bsp_book_over_nw_quantiles": {q: r4(np.quantile(book_b, q)) for q in (0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1)},
         "share_races_book_lt_0.97": r4((book_b < 0.97).mean()), "share_races_book_gt_1.04": r4((book_b > 1.04).mean()),
         "share_loose_lt_0.97_that_are_harness": r4(harness_b[book_b < 0.97].mean()) if (book_b < 0.97).any() else None,
         "share_races_max_bsp_ge_100": r4((maxbsp_b >= 100).mean()), "share_races_min_bsp_le_1.1": r4((minbsp_b <= 1.1).mean()),
         "dead_heat_races": int(dead_b.sum()), "runners_q_bsp_clipped_ge_0.999": int((fb["q_bsp"] >= 0.999).sum()),
         "benchmark_races_by_year": {int(y): int(v) for y, v in rr.groupby(rr["date"].dt.year)["race"].size().items()},
         "fully_priced_races_by_year": {int(y): int(v) for y, v in full.drop_duplicates("race").groupby(full.drop_duplicates("race")["date"].dt.year)["race"].size().items()}})

    # ------------------------------------------------------------ 1. reproduction + robustness
    res = {"section": "1_reproduction_vs_place_bsp", "races": int(len(ll_bsp)), "ll_bsp": r4(ll_bsp.mean()),
           "ll_blend75": r4(ll_b75.mean()), "ll_pcal": r4(ll_cal.mean()), "ll_ltp_market": r4(ll_mkt.mean()),
           "blend75_vs_bsp": gain_stats(ll_b75, ll_bsp, dates_b, mid, masks),
           "pcal_vs_bsp": gain_stats(ll_cal, ll_bsp, dates_b, mid, {"gallops": ~harness_b, "harness": harness_b}),
           "ltp_market_vs_bsp": O.paired(ll_mkt, ll_bsp, dates_b < mid)}
    grid = {}
    fh = dates_b < mid
    for w in BLEND_GRID:
        q = w * fb["q_bsp"].to_numpy() + (1 - w) * fb["p_cal"].to_numpy()
        g = ll_bsp - O.logloss_race(q, yb, rcb)
        grid[round(float(w), 2)] = [r4(g.mean()), r4(g[fh].mean()), r4(g[~fh].mean())]
    res["fixed_weight_grid_w_on_bsp [gain, half1, half2]"] = grid
    res["original_tried_weights"] = {w: grid[w] for w in (0.5, 0.75, 0.9)}
    res["best_w_in_sample"] = max(grid, key=lambda w: grid[w][0])
    # walk-forward fitted weight on strictly earlier months
    ll_by_w = {w: O.logloss_race(w * fb["q_bsp"].to_numpy() + (1 - w) * fb["p_cal"].to_numpy(), yb, rcb) for w in BLEND_GRID}
    msum = {w: pd.Series(v).groupby(rmonth).sum() for w, v in ll_by_w.items()}
    mcount = pd.Series(rmonth).value_counts()
    chosen, wf = {}, np.full(len(ll_bsp), np.nan)
    for m in sorted(set(rmonth)):
        prev = [mm for mm in mcount.index if mm < m]
        if sum(mcount[mm] for mm in prev) < 500:
            continue
        best = min(BLEND_GRID, key=lambda w: sum(msum[round(float(w), 2)].get(mm, 0) for mm in prev))
        chosen[m] = round(float(best), 2)
        wf[rmonth == m] = ll_by_w[round(float(best), 2)][rmonth == m]
    has = ~np.isnan(wf)
    res["walk_forward_weight_vs_bsp"] = {**gain_stats(wf[has], ll_bsp[has], dates_b[has], mid, {kk: v[has] for kk, v in masks.items()}),
                                        "median_w_on_bsp": float(np.median(list(chosen.values()))) if chosen else None,
                                        "w_sample": {m: chosen[m] for m in sorted(chosen)[::6]}}
    out(res)

    # ------------------------------------------------------------ 2. gain by book tightness / extreme prices
    rows = []
    edges = [0, 0.9, 0.95, 0.97, 0.99, 1.01, 1.02, 1.04, 1.06, 10]
    d75 = ll_bsp - ll_b75
    dcal = ll_bsp - ll_cal
    for lo, hi in zip(edges, edges[1:]):
        s = (book_b >= lo) & (book_b < hi)
        if s.sum() < 2:
            continue
        rows.append({"book_over_nw": f"{lo:g}-{hi:g}", "races": int(s.sum()), "harness_share": r4(harness_b[s].mean()),
                     "ll_bsp": r4(ll_bsp[s].mean()), "blend75_gain": r4(d75[s].mean()), "se": r4(d75[s].std(ddof=1) / math.sqrt(s.sum())),
                     "pcal_vs_bsp_gain": r4(dcal[s].mean()), "share_of_total_blend_gain": r4(d75[s].sum() / d75.sum())})
    out({"section": "2_gain_by_bsp_book_tightness", "rows": rows})
    # per-runner view: where (by q_bsp band) does the blend gain its log loss?
    lr = -(yb * np.log(np.clip(fb["q_bsp"], 1e-6, 1 - 1e-6)) + (1 - yb) * np.log(1 - np.clip(fb["q_bsp"], 1e-6, 1 - 1e-6)))
    lb = -(yb * np.log(np.clip(fb["blend75"], 1e-6, 1 - 1e-6)) + (1 - yb) * np.log(1 - np.clip(fb["blend75"], 1e-6, 1 - 1e-6)))
    fb["_d"] = lr - lb
    rows = calib_rows(fb, ["q_bsp", "p_cal", "blend75"], "q_bsp")
    for row in rows:
        g = fb[pd.cut(fb["q_bsp"], [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001], right=False).astype(str) == row["band_q_bsp"]]
        row["blend_gain_per_runner"] = r4(g["_d"].mean())
        row["share_of_total_blend_gain"] = r4(g["_d"].sum() / fb["_d"].sum())
        row["strike_over_q_bsp"] = r4(g["placed"].mean() / g["q_bsp"].mean())
    qb = fb["q_bsp"].clip(1e-6, 1 - 1e-6); pc = fb["p_cal"].clip(1e-6, 1 - 1e-6)
    out({"section": "3_calibration_by_q_bsp_band", "rows": rows,
         "q_bsp_vs_p_cal_corr_logit": r4(np.corrcoef(np.log(qb / (1 - qb)), np.log(pc / (1 - pc)))[0, 1])})
    # per-runner decomposition of the blend gain by place-BSP price band x placed / not placed, and by BSP-vs-ltp deviation
    pb_band = pd.cut(fb["bsp_pl"], [1, 1.5, 2, 3, 5, 8, 15, 30, 50, 100, 200, 1001], right=False)
    rows = []
    for (b, pl), g in fb.groupby([pb_band, fb["placed"].astype(int)], observed=True):
        rows.append({"bsp_pl_band": str(b), "placed": int(pl), "runners": int(len(g)), "mean_q_bsp": r4(g["q_bsp"].mean()),
                     "mean_p_cal": r4(g["p_cal"].mean()), "median_bsp_over_ltp": r4((g["bsp_pl"] / g["ltp_pl"]).median()),
                     "gain_per_runner": r4(g["_d"].mean()), "share_of_total_blend_gain": r4(g["_d"].sum() / fb["_d"].sum())})
    dev_band = pd.cut(fb["abs_log_bsp_ltp"], [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 10], right=False)
    rows2 = []
    for b, g in fb.groupby(dev_band, observed=True):
        rows2.append({"abs_log_bsp_over_ltp": str(b), "runners": int(len(g)), "strike": r4(g["placed"].mean()), "mean_q_bsp": r4(g["q_bsp"].mean()),
                      "mean_q_mkt_ltp": r4(g["q_mkt"].mean()), "mean_p_cal": r4(g["p_cal"].mean()),
                      "gain_per_runner": r4(g["_d"].mean()), "share_of_total_blend_gain": r4(g["_d"].sum() / fb["_d"].sum())})
    out({"section": "3a_blend_gain_decomposition_per_runner", "by_bsp_band_x_placed": rows, "by_bsp_vs_ltp_deviation": rows2,
         "share_runners_bsp_far_from_ltp_gt_0.3": r4((fb["abs_log_bsp_ltp"] >= 0.3).mean())})

    # ------------------------------------------------------------ 3b. fit-free re-normalisations of the BSP book
    fb["q_pow"], kpow = power_norm(fb, "bsp_pl")
    ll_pow = LL("q_pow")
    fb["blend75_pow"] = 0.75 * fb["q_pow"] + 0.25 * fb["p_cal"]
    out({"section": "3b_power_normalised_bsp", "power_k_median": r4(np.median(kpow)),
         "power_vs_proportional": gain_stats(ll_pow, ll_bsp, dates_b, mid),
         "blend75_with_power_bsp_vs_power_bsp": gain_stats(LL("blend75_pow"), ll_pow, dates_b, mid, {"gallops": ~harness_b, "harness": harness_b})})

    # ------------------------------------------------------------ 4. the same walk-forward cubic calibration applied to q_bsp
    fbm = sorted(fb["month"].unique())
    fb["q_bsp_cal"], bsp_cal_params = O.calibrate_walk_forward(fb, "q_bsp", fbm)
    hc = fb["q_bsp_cal"].notna().to_numpy()
    rc_has = np.bincount(rcb, weights=hc) == np.bincount(rcb)
    ll_bspc = O.logloss_race(np.nan_to_num(fb["q_bsp_cal"].to_numpy(), nan=0.5), yb, rcb)
    res = {"section": "4_calibrated_bsp", "races_with_calibrated_bsp": int(rc_has.sum()),
           "bsp_cal_params_sample": {m: bsp_cal_params[m] for m in sorted(bsp_cal_params)[::6]},
           "q_bsp_cal_vs_q_bsp": gain_stats(ll_bspc[rc_has], ll_bsp[rc_has], dates_b[rc_has], mid, {kk: v[rc_has] for kk, v in masks.items()}),
           "blend75_raw_vs_q_bsp_cal": gain_stats(ll_b75[rc_has], ll_bspc[rc_has], dates_b[rc_has], mid)}
    grid = {}
    fhc = fh[rc_has]
    for w in BLEND_GRID:
        q = w * np.nan_to_num(fb["q_bsp_cal"].to_numpy(), nan=0.5) + (1 - w) * fb["p_cal"].to_numpy()
        g = (ll_bspc - O.logloss_race(q, yb, rcb))[rc_has]
        grid[round(float(w), 2)] = [r4(g.mean()), r4(g.std(ddof=1) / math.sqrt(len(g))), r4(g[fhc].mean()), r4(g[~fhc].mean())]
        if w == 0.75:
            res["blend75_of_q_bsp_cal_and_p_cal_vs_q_bsp_cal"] = gain_stats(O.logloss_race(q, yb, rcb)[rc_has], ll_bspc[rc_has], dates_b[rc_has], mid,
                                                                             {kk: v[rc_has] for kk, v in masks.items()})
    res["grid_w_on_q_bsp_cal [gain, se, half1, half2]"] = grid
    res["best_w_in_sample"] = max(grid, key=lambda w: grid[w][0])
    out(res)

    # ------------------------------------------------------------ 5. nested walk-forward logistic models
    models = {"A_cubic_bsp": (["q_bsp"], "q_bsp"),
              "B_cubic_bsp+pcal": (["q_bsp", "p_cal"], "q_bsp"),
              "C_cubic_bsp+pcal+ltpmkt": (["q_bsp", "p_cal", "q_mkt"], "q_bsp"),
              "D_cubic_bsp+ltpmkt": (["q_bsp", "q_mkt"], "q_bsp"),
              "E_cubic_bsp+pfair_uncal": (["q_bsp", "p_fair"], "q_bsp")}
    preds, coefs = {}, {}
    for name, (feats, cub) in models.items():
        preds[name], coefs[name] = wf_logit(fb, feats, fbm, cub)
    common = np.ones(len(fb), dtype=bool)
    for v in preds.values():
        common &= ~np.isnan(v)
    rc_c = np.bincount(rcb, weights=common) == np.bincount(rcb)
    lls = {name: O.logloss_race(np.nan_to_num(v, nan=0.5), yb, rcb) for name, v in preds.items()}
    res = {"section": "5_nested_walk_forward_logit", "races": int(rc_c.sum()), "ll_A": r4(lls["A_cubic_bsp"][rc_c].mean()),
           "ll_raw_bsp_same_races": r4(ll_bsp[rc_c].mean()),
           "A_vs_raw_bsp": gain_stats(lls["A_cubic_bsp"][rc_c], ll_bsp[rc_c], dates_b[rc_c], mid),
           "B_vs_A (p_cal beyond calibrated BSP)": gain_stats(lls["B_cubic_bsp+pcal"][rc_c], lls["A_cubic_bsp"][rc_c], dates_b[rc_c], mid,
                                                              {kk: v[rc_c] for kk, v in masks.items()}),
           "E_vs_A (uncalibrated p_fair beyond calibrated BSP)": gain_stats(lls["E_cubic_bsp+pfair_uncal"][rc_c], lls["A_cubic_bsp"][rc_c], dates_b[rc_c], mid),
           "D_vs_A (ltp place market beyond calibrated BSP)": gain_stats(lls["D_cubic_bsp+ltpmkt"][rc_c], lls["A_cubic_bsp"][rc_c], dates_b[rc_c], mid),
           "C_vs_D (p_cal beyond calibrated BSP + ltp place market)": gain_stats(lls["C_cubic_bsp+pcal+ltpmkt"][rc_c], lls["D_cubic_bsp+ltpmkt"][rc_c], dates_b[rc_c], mid,
                                                                        {kk: v[rc_c] for kk, v in masks.items()}),
           "fixed_blend75_vs_A": gain_stats(ll_b75[rc_c], lls["A_cubic_bsp"][rc_c], dates_b[rc_c], mid),
           "coef_samples_std_units": {name: {m: coefs[name][m] for m in sorted(coefs[name])[::8]} for name in models}}
    # calibrated-BSP + p_cal fitted on a tight-book subset only (so loose books cannot drive the p_cal coefficient)
    tight = (fb["book_nw"] >= 0.97) & (fb["book_nw"] <= 1.04)
    ft = fb[tight].copy()
    pa, _ = wf_logit(ft, ["q_bsp"], fbm, "q_bsp")
    pb_, _ = wf_logit(ft, ["q_bsp", "p_cal"], fbm, "q_bsp")
    ok_t = ~np.isnan(pa) & ~np.isnan(pb_)
    rct = pd.factorize(ft["race"])[0]
    rct_ok = np.bincount(rct, weights=ok_t) == np.bincount(rct)
    yt = ft["placed"].to_numpy(dtype=float)
    la = O.logloss_race(np.nan_to_num(pa, nan=0.5), yt, rct); lb_ = O.logloss_race(np.nan_to_num(pb_, nan=0.5), yt, rct)
    res["tight_book_only_B_vs_A"] = gain_stats(lb_[rct_ok], la[rct_ok], ft.drop_duplicates("race")["date"].to_numpy()[rct_ok], mid)
    out(res)

    # ------------------------------------------------------------ 6. same test with the WIN BSP-derived place probability
    fw = fb[fb["p_fair_wbsp"].notna()].copy()
    fw["p_cal_wbsp"], _ = O.calibrate_walk_forward(fw, "p_fair_wbsp", sorted(fw["month"].unique()))
    fw = fw[fw["p_cal_wbsp"].notna()].copy()
    fwm = sorted(fw["month"].unique())
    pa, _ = wf_logit(fw, ["q_bsp"], fwm, "q_bsp")
    pw, _ = wf_logit(fw, ["q_bsp", "p_cal_wbsp"], fwm, "q_bsp")
    pl_, _ = wf_logit(fw, ["q_bsp", "p_cal"], fwm, "q_bsp")
    pwl, cw = wf_logit(fw, ["q_bsp", "p_cal_wbsp", "p_cal"], fwm, "q_bsp")
    okw = ~np.isnan(pa) & ~np.isnan(pw) & ~np.isnan(pl_) & ~np.isnan(pwl)
    rcw = pd.factorize(fw["race"])[0]
    rcw_ok = np.bincount(rcw, weights=okw) == np.bincount(rcw)
    yw = fw["placed"].to_numpy(dtype=float)
    dw = fw.drop_duplicates("race")["date"].to_numpy()
    L = lambda v: O.logloss_race(np.nan_to_num(v, nan=0.5), yw, rcw)  # noqa: E731
    la, lw, ll_, lwl = L(pa), L(pw), L(pl_), L(pwl)
    fw["blend75_wbsp"] = 0.75 * fw["q_bsp"] + 0.25 * fw["p_cal_wbsp"]
    out({"section": "6_win_bsp_derived_consistency_check", "races": int(rcw_ok.sum()),
         "fixed_blend75_bsp_with_winBSP_pcal_vs_bsp": gain_stats(L(fw["blend75_wbsp"].to_numpy())[rcw_ok], L(fw["q_bsp"].to_numpy())[rcw_ok], dw[rcw_ok], mid),
         "winBSP_pcal_beyond_calibrated_place_bsp": gain_stats(lw[rcw_ok], la[rcw_ok], dw[rcw_ok], mid),
         "winLTP_pcal_beyond_calibrated_place_bsp_same_races": gain_stats(ll_[rcw_ok], la[rcw_ok], dw[rcw_ok], mid),
         "winLTP_pcal_beyond_calibrated_place_bsp_and_winBSP_pcal": gain_stats(lwl[rcw_ok], lw[rcw_ok], dw[rcw_ok], mid),
         "coef_sample_[1,z_bsp,z2,z3,z_winbsp_pcal,z_winltp_pcal]": {m: cw[m] for m in sorted(cw)[::8]}})
    out({"section": "totals", "probability_cells_tested_here": "1 claim cell + 3 original weights + 21-point grid x2 + 6 nested model contrasts + 4 win-BSP contrasts + subsets",
         "runtime_s": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

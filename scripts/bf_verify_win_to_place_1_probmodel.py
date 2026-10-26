"""Adversarial verification of the PROBABILITY-MODEL claim of experiment "win_to_place"
(scripts/bf_win_to_place.py), lens: STATISTICS AND EXECUTION.

CLAIM: on fully priced AU place markets (every runner traded at ltp_pre, 41% of markets),
the walk-forward blend w*q_market + (1-w)*p_cal (w chosen each month on earlier months,
median w = 0.25 on the market) has per-race Bernoulli log loss 4.7283 against 4.7385 for
the normalised ltp place market: gain 0.0102 (SE 0.0010), halves 0.0135 (SE 0.0013) and
0.0077 (SE 0.0014), 45,820 races.

    python scripts/bf_verify_win_to_place_1_probmodel.py [--dir data/raw/betfair/split] [--frac 0.1]

The original pipeline is imported unchanged (join, lambda fit, cubic-logit calibration,
walk-forward blend), so section 0 reproduces the headline exactly. Then:
  1. SE robustness: per-race SE (as claimed), SE clustered by day and by meeting, a
     month-block bootstrap; splits not chosen after the fact (calendar years, thirds by
     date, per-month sign counts); the "> 2 SE in both halves" rule under each SE.
  2. Sterner but fair baselines: the place market recalibrated walk-forward with the SAME
     cubic-logit routine the model gets (how much of the gain is the market's own
     miscalibration?); a walk-forward blend of that calibrated market with p_cal (does the
     WIN market add anything beyond recalibration?); the place BSP (post-race, benchmark
     only) against the blend itself, which the original did not report.
  3. Where the gain lives: by the place ltp book / n_winners (an under- or over-round
     book is a stale one-minute snapshot), by post-race staleness of the place ltp
     (max |log(bsp/ltp)| in the race, BSP subset), by code, places paid, field size and
     liquidity (ltp_t30 present); a "well-formed market" subset.
  4. Design sensitivity (tuning): blend weight from the previous 12 months only; linear
     instead of cubic calibration; 6- and 24-month calibration windows; calibration fitted
     on fully priced races only (same population as the market's recalibration); plain
     Harville (lambda = 1) instead of fitted lambdas; fixed in-sample blend weights.
  5. Execution: the blend used as a bet signal (back the place at ltp_pre when
     p_blend * ltp_pl > 1 + k), at ltp_pre and settled at the place BSP, day-clustered SE,
     commission 0/5/6/8/10%.
  6. Multiple testing: BH (q = 0.05) over every probability test (the original's 13 plus
     those added here) and every bet cell added here.
Prints summary numbers only (never horse names or rows).
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("w2p", ROOT / "scripts" / "bf_win_to_place.py")
w2p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w2p)

N_BOOT = 2000
COMMS = [0.0, 0.05, 0.06, 0.08, 0.10]
BOOK_BINS = [0.0, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 10.0]
STALE_BINS = [0.0, 0.05, 0.1, 0.2, 0.4, 10.0]
FIELD_BINS = [0, 7, 10, 14, 100]


def out(obj) -> None:
    print(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)), flush=True)


def r4(x):
    return w2p.r4(x)


def cluster_se(d: np.ndarray, cl) -> float | None:
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return None
    s = pd.Series(d - d.mean()).groupby(np.asarray(cl)).sum().to_numpy()
    return float(math.sqrt((s ** 2).sum()) / len(d))


def block_boot(d: np.ndarray, block, n_boot=N_BOOT, seed=0) -> dict:
    s = pd.Series(np.asarray(d, dtype=float)).groupby(np.asarray(block)).agg(["sum", "count"])
    sums, cnts = s["sum"].to_numpy(), s["count"].to_numpy()
    B = len(sums)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, B, size=(n_boot, B))
    g = sums[idx].sum(1) / cnts[idx].sum(1)
    return {"blocks": int(B), "se": r4(g.std(ddof=1)), "ci95": [r4(np.quantile(g, 0.025)), r4(np.quantile(g, 0.975))],
            "share_resamples_gain_le_0": r4((g <= 0).mean())}


def sub(d: np.ndarray, m: np.ndarray, cl=None) -> dict:
    """gain, per-race SE (and clustered SE when ``cl`` given) on the races selected by mask m."""
    x = d[m]
    if len(x) < 2:
        return {"races": int(len(x)), "gain": None, "se": None, "z": None}
    se = x.std(ddof=1) / math.sqrt(len(x))
    res = {"races": int(len(x)), "gain": r4(x.mean()), "se_race": r4(se), "z_race": r4(x.mean() / se) if se > 0 else None}
    if cl is not None:
        sc = cluster_se(x, np.asarray(cl)[m])
        res["se_cl_date"] = r4(sc)
        res["z_cl_date"] = r4(x.mean() / sc) if sc else None
    return res


def gain_stats(ll_model: np.ndarray, ll_base: np.ndarray, meta: pd.DataFrame, name: str, full_stats=True) -> dict:
    """Paired per-race comparison: gain = base - model (positive = model better)."""
    d = np.asarray(ll_base, dtype=float) - np.asarray(ll_model, dtype=float)
    n = len(d)
    se = d.std(ddof=1) / math.sqrt(n)
    res = {"test": name, "races": int(n), "ll_model": r4(np.mean(ll_model)), "ll_base": r4(np.mean(ll_base)),
           "gain": r4(d.mean()), "se_race": r4(se), "z_race": r4(d.mean() / se) if se > 0 else None}
    date = meta["date"].to_numpy()
    scd = cluster_se(d, date)
    res["se_cl_date"] = r4(scd)
    res["z_cl_date"] = r4(d.mean() / scd) if scd else None
    fh = meta["first_half"].to_numpy()
    res["halves_gain"] = [r4(d[fh].mean()), r4(d[~fh].mean())]
    res["halves_se_race"] = [r4(d[fh].std(ddof=1) / math.sqrt(fh.sum())), r4(d[~fh].std(ddof=1) / math.sqrt((~fh).sum()))]
    res["halves_se_cl_date"] = [r4(cluster_se(d[fh], date[fh])), r4(cluster_se(d[~fh], date[~fh]))]
    res["positive_gain_gt_2se_race"] = bool(se > 0 and d.mean() > 2 * se)
    res["positive_both_halves_2se_race"] = bool(all(g is not None and s is not None and g > 2 * s
                                                     for g, s in zip(res["halves_gain"], res["halves_se_race"])))
    res["positive_both_halves_2se_cl_date"] = bool(all(g is not None and s is not None and g > 2 * s
                                                        for g, s in zip(res["halves_gain"], res["halves_se_cl_date"])))
    if full_stats:
        res["se_cl_meeting"] = r4(cluster_se(d, meta["event_id"].to_numpy()))
        res["month_block_bootstrap"] = block_boot(d, meta["month"].to_numpy())
        years = {}
        for yv in sorted(meta["year"].unique()):
            years[str(yv)] = sub(d, (meta["year"] == yv).to_numpy(), date)
        res["by_year"] = years
        order = np.argsort(date, kind="stable")
        third = np.zeros(n, dtype=int)
        third[order[n // 3:2 * n // 3]] = 1
        third[order[2 * n // 3:]] = 2
        res["thirds_by_date"] = [sub(d, third == t, date) for t in range(3)]
        pm = pd.Series(d).groupby(meta["month"].to_numpy()).agg(["mean", "count"])
        res["months"] = {"n": int(len(pm)), "n_gain_positive": int((pm["mean"] > 0).sum()),
                         "min_month_gain": r4(pm["mean"].min()), "median_month_gain": r4(pm["mean"].median())}
        # heavy tails: is the mean gain carried by a few races with an extreme (stale) price?
        ad = np.abs(d)
        top = np.argsort(ad)[::-1][:max(1, n // 100)]
        lo, hi = np.quantile(d, [0.01, 0.99])
        dw = np.clip(d, lo, hi)
        keep = (d >= lo) & (d <= hi)
        res["heavy_tails"] = {"median_race_gain": r4(np.median(d)), "share_races_gain_positive": r4((d > 0).mean()),
                              "winsorised_1pct_gain": r4(dw.mean()), "winsorised_1pct_se_race": r4(dw.std(ddof=1) / math.sqrt(n)),
                              "trimmed_1pct_gain": r4(d[keep].mean()), "trimmed_1pct_se_race": r4(d[keep].std(ddof=1) / math.sqrt(keep.sum())),
                              "top_1pct_abs_races_share_of_total_gain": r4(d[top].sum() / d.sum()) if d.sum() != 0 else None,
                              "max_abs_race_diff": r4(ad.max()), "q99_abs_race_diff": r4(np.quantile(ad, 0.99))}
    return res


def calibrate_wf(df: pd.DataFrame, col: str, months_sorted: list[str], window=12, degree=3, min_rows=5000):
    """The original's cubic-logit walk-forward recalibration, generalised to ``degree`` and
    ``window`` (degree=3, window=12 reproduces w2p.calibrate_walk_forward)."""
    pc = df[col].clip(1e-6, 1 - 1e-6)
    z = np.log(pc / (1 - pc)).to_numpy()
    y = df["placed"].to_numpy(dtype=float)
    month = df["month"].to_numpy()
    outp = np.full(len(df), np.nan)

    def design(zz):
        return np.column_stack([zz ** k for k in range(degree + 1)])
    for idx, m in enumerate(months_sorted):
        tr = np.isin(month, months_sorted[max(0, idx - window):idx])
        if tr.sum() < min_rows:
            continue
        mu_z, sd_z = z[tr].mean(), z[tr].std()
        X = design((z[tr] - mu_z) / sd_z)
        w = np.zeros(degree + 1)
        for _ in range(12):
            mu = 1 / (1 + np.exp(-(X @ w)))
            g = X.T @ (mu - y[tr])
            H = (X * (mu * (1 - mu))[:, None]).T @ X + 1e-6 * np.eye(degree + 1)
            w = w - np.linalg.solve(H, g)
        te = month == m
        outp[te] = 1 / (1 + np.exp(-(design((z[te] - mu_z) / sd_z) @ w)))
    p = pd.Series(outp, index=df.index)
    scale = df["n_winners"] / p.groupby(df["race"]).transform("sum")
    return (p * scale).clip(upper=0.999).to_numpy()


def wf_blend(qa: np.ndarray, pb: np.ndarray, y: np.ndarray, rc: np.ndarray, rmonth: np.ndarray, has_race: np.ndarray,
             grid=w2p.BLEND_GRID, min_prev=500, lookback: int | None = None):
    """The original's walk-forward blend: for each month choose w (weight on ``qa``) that
    minimises the summed log loss of all strictly earlier months (``lookback`` months when
    given), evaluate on that month's races with ``has_race``. Returns per-race ll (NaN when
    not evaluated), the chosen weights and the per-weight ll arrays."""
    pb = np.nan_to_num(np.asarray(pb, dtype=float), nan=0.5)
    bl = {}
    for w in grid:
        q = w * qa + (1 - w) * pb
        bl[round(float(w), 2)] = w2p.logloss_race(q, y, rc)
    mcount = pd.Series(rmonth).value_counts()
    msum = {w: pd.Series(v[has_race]).groupby(rmonth[has_race]).sum() for w, v in bl.items()}
    chosen, wf = {}, np.full(len(has_race), np.nan)
    for m in sorted(set(rmonth)):
        prev = sorted(mm for mm in mcount.index if mm < m)
        if lookback:
            prev = prev[-lookback:]
        if sum(mcount[mm] for mm in prev) < min_prev:
            continue
        best = min(grid, key=lambda w: sum(msum[round(float(w), 2)].get(mm, 0) for mm in prev))
        chosen[m] = round(float(best), 2)
        sel = (rmonth == m) & has_race
        wf[sel] = bl[chosen[m]][sel]
    return wf, chosen, bl


def race_mask(rc: np.ndarray, runner_ok: np.ndarray) -> np.ndarray:
    return np.bincount(rc, weights=runner_ok.astype(float)) == np.bincount(rc)


def bet_cell(g: pd.DataFrame, mid, name: str) -> dict:
    res = {"rule": name, "bets": int(len(g))}
    if len(g) < 2:
        return res
    r5 = w2p.returns(g["ltp_pl"], g["placed"], g["dh"], 0.05)
    date = g["date"].to_numpy()
    fh = date < mid
    res["ltp"] = {f"roi_c{int(round(c * 100))}": r4(w2p.returns(g["ltp_pl"], g["placed"], g["dh"], c).mean()) for c in COMMS}
    res["ltp"].update({"se_bet": r4(r5.std(ddof=1) / math.sqrt(len(r5))), "se_cl_race": r4(cluster_se(r5, g["race"].to_numpy())),
                       "se_cl_date": r4(cluster_se(r5, date)), "halves_c5": [r4(r5[fh].mean()), r4(r5[~fh].mean())],
                       "one_tick_worse_c5": r4(w2p.returns(w2p.one_tick_worse(g["ltp_pl"]), g["placed"], g["dh"], 0.05).mean()),
                       "strike": r4(g["placed"].mean()), "median_price": r4(g["ltp_pl"].median())})
    has = (g["bsp_pl"].notna() & (g["bsp_pl"] > 1)).to_numpy()
    h = g[has]
    if len(h) > 1:
        rb = w2p.returns(h["bsp_pl"], h["placed"], h["dh"], 0.05)
        dh_ = h["date"].to_numpy()
        fhb = dh_ < mid
        res["settle_bsp"] = {"bets": int(len(h)), "cover": r4(has.mean()),
                             **{f"roi_c{int(round(c * 100))}": r4(w2p.returns(h["bsp_pl"], h["placed"], h["dh"], c).mean()) for c in COMMS},
                             "se_bet": r4(rb.std(ddof=1) / math.sqrt(len(rb))), "se_cl_date": r4(cluster_se(rb, dh_)),
                             "halves_c5": [r4(rb[fhb].mean()), r4(rb[~fhb].mean())],
                             "ltp_roi_c5_same_subset": r4(w2p.returns(h["ltp_pl"], h["placed"], h["dh"], 0.05).mean()),
                             "worse_of_ltp_bsp_c5": r4(w2p.returns(np.minimum(h["ltp_pl"], h["bsp_pl"]), h["placed"], h["dh"], 0.05).mean()),
                             "median_bsp_over_ltp": r4((h["bsp_pl"] / h["ltp_pl"]).median())}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/raw/betfair/split"))
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    # ---- the original pipeline, unchanged -------------------------------------------------------
    j = w2p.load(a.dir, a.frac, a.seed)
    P, jw, k, i, ok, nw, race = w2p.placed_set_stats(j)
    months = race["month"].to_numpy()
    codes = race["harness"].to_numpy()
    months_sorted = sorted(set(months))
    ll = w2p.grid_loglik(P, jw, k, i, ok, nw)
    lam = w2p.fit_lambdas_walk_forward(ll, ok, months, codes, months_sorted)
    del ll
    j["p_fair"] = w2p.derived_place_probs(j, P, nw, lam, months, codes)
    lam1 = {key: (1.0, 1.0, v[2]) for key, v in lam.items()}      # plain Harville, same availability
    j["p_fair_l1"] = w2p.derived_place_probs(j, P, nw, lam1, months, codes)
    del P
    j = j[j["p_fair"].notna()].copy()
    j["race"] = pd.factorize(j["win_mid"])[0]
    j["dh"] = j["n_winners"] / j["n_placed"]
    jm = sorted(j["month"].unique())
    j["p_cal"], cal_params = w2p.calibrate_walk_forward(j, "p_fair", jm)
    days = sorted(j["date"].unique())
    mid = days[len(days) // 2]
    t_pipe = time.time() - t0

    # sensitivity calibrations on the same population as the original (all joined races)
    j["p_cal_lin"] = calibrate_wf(j, "p_fair", jm, window=12, degree=1)
    j["p_cal_w6"] = calibrate_wf(j, "p_fair", jm, window=6, degree=3)
    j["p_cal_w24"] = calibrate_wf(j, "p_fair", jm, window=24, degree=3)
    j["p_cal_l1"] = calibrate_wf(j, "p_fair_l1", jm, window=12, degree=3)
    assert np.allclose(calibrate_wf(j, "p_fair", jm), j["p_cal"].to_numpy(), equal_nan=True), "calibrate_wf != original"

    # ---- fully priced place markets --------------------------------------------------------------
    j["all_priced"] = j.groupby("race")["ltp_pl"].transform(lambda s: s.notna().all() and (s > 1).all())
    full = j[j["all_priced"]].copy()
    book = full.groupby("race")["ltp_pl"].transform(lambda s: (1 / s).sum())
    full["q_mkt"] = (1 / full["ltp_pl"]) * full["n_winners"] / book
    full["book_over_nw"] = book / full["n_winners"]
    full = full.sort_values(["race", "pos"]).reset_index(drop=True)
    fm = sorted(full["month"].unique())
    # the market recalibrated with the same routine (fully priced races only, where q_mkt exists)
    full["q_cal"] = calibrate_wf(full, "q_mkt", fm, window=12, degree=3)
    # p_fair calibrated on the same population as q_cal (fully priced races only)
    full["p_cal_fp"] = calibrate_wf(full, "p_fair", fm, window=12, degree=3)
    del j

    rc = pd.factorize(full["race"])[0]
    R = rc.max() + 1
    races = full.drop_duplicates("race")
    has_bsp_r = (full["bsp_pl"].notna() & (full["bsp_pl"] > 1)).to_numpy()
    stale_pl = np.where(has_bsp_r, np.abs(np.log(full["bsp_pl"].to_numpy(dtype=float) / full["ltp_pl"].to_numpy(dtype=float))), 0.0)
    has_bspw = (full["bsp_win"].notna() & (full["bsp_win"] > 1)).to_numpy()
    stale_win = np.where(has_bspw, np.abs(np.log(full["bsp_win"].to_numpy(dtype=float) / full["ltp_win"].to_numpy(dtype=float))), 0.0)
    meta = pd.DataFrame({
        "date": races["date"].to_numpy(), "event_id": races["event_id"].to_numpy(), "month": races["month"].to_numpy(),
        "year": pd.DatetimeIndex(races["date"]).year, "first_half": (races["date"] < mid).to_numpy(),
        "harness": races["harness"].to_numpy(), "n_winners": races["n_winners"].to_numpy(), "n": races["n"].to_numpy(),
        "book_over_nw": races["book_over_nw"].to_numpy(),
        "all_bsp": race_mask(rc, has_bsp_r), "all_bsp_win": race_mask(rc, has_bspw),
        "stale_pl_max": pd.Series(stale_pl).groupby(rc).max().to_numpy(),
        "stale_win_max": pd.Series(stale_win).groupby(rc).max().to_numpy(),
        "t30_pl_share": pd.Series(full["t30_pl"].notna().to_numpy(dtype=float)).groupby(rc).mean().to_numpy(),
        "t30_win_all": race_mask(rc, full["t30_win"].notna().to_numpy()),
        "min_ltp_pl": pd.Series(full["ltp_pl"].to_numpy(dtype=float)).groupby(rc).min().to_numpy(),
    })
    assert len(meta) == R
    y = full["placed"].to_numpy(dtype=float)
    rmonth = meta["month"].to_numpy()
    q_mkt = full["q_mkt"].to_numpy()
    ll_mkt = w2p.logloss_race(q_mkt, y, rc)
    ll_fair = w2p.logloss_race(full["p_fair"].to_numpy(), y, rc)
    pcal = full["p_cal"].to_numpy()
    rc_has = race_mask(rc, full["p_cal"].notna().to_numpy())
    ll_cal = w2p.logloss_race(np.nan_to_num(pcal, nan=0.5), y, rc)

    # ---- 0. reproduction ---------------------------------------------------------------------------
    wfc, chosen_c, bl_c = wf_blend(q_mkt, pcal, y, rc, rmonth, rc_has)
    hasc = ~np.isnan(wfc)
    wfu, chosen_u, bl_u = wf_blend(q_mkt, full["p_fair"].to_numpy(), y, rc, rmonth, np.ones(R, dtype=bool))
    hasu = ~np.isnan(wfu)
    tests = []  # every probability test (name, gain, se_race, se_cl_date, z_race, z_cl_date)

    def record(res: dict, family: str) -> dict:
        tests.append({"test": res["test"], "family": family, "races": res["races"], "gain": res["gain"], "se_race": res["se_race"],
                      "se_cl_date": res["se_cl_date"], "z_race": res["z_race"], "z_cl_date": res["z_cl_date"],
                      "halves_gain": res["halves_gain"], "halves_se_race": res["halves_se_race"]})
        return res

    head = record(gain_stats(wfc[hasc], ll_mkt[hasc], meta[hasc], "HEADLINE blend_wf_cal vs ltp place market (normalised)"), "original")
    rep = {"section": "0_reproduction", "runtime_pipeline_s": round(t_pipe, 1), "fully_priced_races": int(R), "runners": int(len(full)),
           "races_with_p_cal": int(rc_has.sum()), "races_in_headline": int(hasc.sum()),
           "from": str(pd.Timestamp(meta["date"].min()).date()), "to": str(pd.Timestamp(meta["date"].max()).date()), "mid": str(pd.Timestamp(mid).date()),
           "book_over_nw_median": r4(meta["book_over_nw"].median()),
           "headline": {k_: head[k_] for k_ in ("races", "ll_model", "ll_base", "gain", "se_race", "halves_gain", "halves_se_race")},
           "claimed": {"ll_model": 4.7283, "ll_base": 4.7385, "gain": 0.0102, "se": 0.0010, "halves": [0.0135, 0.0077], "halves_se": [0.0013, 0.0014], "races": 45820, "median_w": 0.25},
           "median_w_market": float(np.median(list(chosen_c.values()))) if chosen_c else None,
           "w_market_by_month_sample": {m: chosen_c[m] for m in sorted(chosen_c)[::6]},
           "w_market_distribution": pd.Series(list(chosen_c.values())).value_counts().sort_index().to_dict(),
           "harville_uncal_vs_market": record(gain_stats(ll_fair, ll_mkt, meta, "harville p_fair vs market", False), "original"),
           "harville_cal_vs_market": record(gain_stats(ll_cal[rc_has], ll_mkt[rc_has], meta[rc_has], "harville p_cal vs market", False), "original"),
           "blend_wf_uncal_vs_market": {**record(gain_stats(wfu[hasu], ll_mkt[hasu], meta[hasu], "blend_wf_uncal vs market", False), "original"),
                                        "median_w_market": float(np.median(list(chosen_u.values()))) if chosen_u else None},
           "fixed_w_uncal": {str(w): record(gain_stats(bl_u[w], ll_mkt, meta, f"fixed blend w_market={w} (p_fair) vs market", False), "original")
                             for w in (0.25, 0.5, 0.75)},
           "fixed_w_cal_in_sample_reference": {str(w): record(gain_stats(bl_c[w][rc_has], ll_mkt[rc_has], meta[rc_has], f"fixed blend w_market={w} (p_cal) vs market", False), "added")
                                               for w in (0.25, 0.5, 0.75)},
           "in_sample_best_w_cal": min(bl_c, key=lambda w: bl_c[w][rc_has].sum())}
    out(rep)

    # ---- 1. SE robustness / splits ----------------------------------------------------------------
    out({"section": "1_se_and_splits_headline", **head})

    # ---- 2. sterner baselines --------------------------------------------------------------------
    qcal = full["q_cal"].to_numpy()
    rc_qc = race_mask(rc, full["q_cal"].notna().to_numpy())
    ll_qcal = w2p.logloss_race(np.nan_to_num(qcal, nan=0.5), y, rc)
    both = rc_qc & hasc
    # blend of the CALIBRATED market with p_cal, walk-forward, available where both exist
    wf2, chosen2, bl2 = wf_blend(np.nan_to_num(qcal, nan=0.5), pcal, y, rc, rmonth, rc_qc & rc_has)
    has2 = ~np.isnan(wf2)
    b2 = both & has2
    # p_fair calibrated on fully priced races only, blended with the calibrated market
    pfp = full["p_cal_fp"].to_numpy()
    rc_fp = race_mask(rc, full["p_cal_fp"].notna().to_numpy())
    wf3, chosen3, _ = wf_blend(np.nan_to_num(qcal, nan=0.5), pfp, y, rc, rmonth, rc_qc & rc_fp)
    has3 = ~np.isnan(wf3)
    b3 = rc_qc & has3
    ll_fp = w2p.logloss_race(np.nan_to_num(pfp, nan=0.5), y, rc)
    sec2 = {"section": "2_sterner_baselines",
            "market_recalibrated_vs_market": record(gain_stats(ll_qcal[both], ll_mkt[both], meta[both], "q_cal (market recalibrated, same routine) vs market"), "added"),
            "headline_blend_on_same_races_vs_market": record(gain_stats(wfc[both], ll_mkt[both], meta[both], "blend_wf_cal vs market (q_cal races)", False), "added"),
            "headline_blend_vs_market_recalibrated": record(gain_stats(wfc[both], ll_qcal[both], meta[both], "blend_wf_cal vs q_cal"), "added"),
            "p_cal_alone_vs_market_recalibrated": record(gain_stats(ll_cal[both], ll_qcal[both], meta[both], "p_cal vs q_cal", False), "added"),
            "blend_of_qcal_and_pcal_vs_market_recalibrated": {**record(gain_stats(wf2[b2], ll_qcal[b2], meta[b2], "blend_wf(q_cal, p_cal) vs q_cal"), "added"),
                                                              "median_w_market": float(np.median(list(chosen2.values()))) if chosen2 else None},
            "blend_of_qcal_and_pcal_vs_market": record(gain_stats(wf2[b2], ll_mkt[b2], meta[b2], "blend_wf(q_cal, p_cal) vs market", False), "added"),
            "p_cal_fullypriced_fit_vs_q_cal": record(gain_stats(ll_fp[b3], ll_qcal[b3], meta[b3], "p_cal (fit on fully priced only) vs q_cal", False), "added"),
            "blend_of_qcal_and_pcal_fp_vs_q_cal": {**record(gain_stats(wf3[b3], ll_qcal[b3], meta[b3], "blend_wf(q_cal, p_cal_fp) vs q_cal"), "added"),
                                                   "median_w_market": float(np.median(list(chosen3.values()))) if chosen3 else None}}
    # place BSP benchmark (post-race price: a benchmark, not a forecast)
    fb = meta["all_bsp"].to_numpy() & hasc & rc_qc
    bsp = full["bsp_pl"].to_numpy(dtype=float)
    inv = np.where(has_bsp_r, 1 / np.where(has_bsp_r, bsp, 1.0), 0.0)
    bsum = np.bincount(rc, weights=inv)
    q_bsp = inv * full["n_winners"].to_numpy() / np.where(bsum > 0, bsum, 1.0)[rc]
    ll_bsp = w2p.logloss_race(np.clip(q_bsp, 1e-6, 1 - 1e-6), y, rc)
    # per-runner blend probabilities for the headline blend (needed for the BSP blend reference and the bets)
    wmap = pd.Series(chosen_c)
    w_r = full["month"].map(wmap).to_numpy(dtype=float)
    p_blend = w_r * q_mkt + (1 - w_r) * pcal
    sec2["place_bsp_benchmark"] = {
        "races": int(fb.sum()), "ll_bsp": r4(ll_bsp[fb].mean()),
        "ltp_market_vs_bsp": record(gain_stats(ll_mkt[fb], ll_bsp[fb], meta[fb], "ltp market vs BSP", False), "original"),
        "q_cal_vs_bsp": record(gain_stats(ll_qcal[fb], ll_bsp[fb], meta[fb], "q_cal vs BSP", False), "added"),
        "p_cal_vs_bsp": record(gain_stats(ll_cal[fb], ll_bsp[fb], meta[fb], "p_cal vs BSP", False), "original"),
        "HEADLINE_blend_vs_bsp": record(gain_stats(wfc[fb], ll_bsp[fb], meta[fb], "blend_wf_cal vs BSP"), "added"),
        "blend_qcal_pcal_vs_bsp": record(gain_stats(wf2[fb & has2], ll_bsp[fb & has2], meta[fb & has2], "blend_wf(q_cal,p_cal) vs BSP", False), "added"),
    }
    for w in (0.75,):
        qq = w * q_bsp + (1 - w) * np.nan_to_num(pcal, nan=0.5)
        sec2["place_bsp_benchmark"][f"oracle_blend_{w}bsp_pcal_vs_bsp"] = record(
            gain_stats(w2p.logloss_race(np.clip(qq, 1e-6, 1 - 1e-6), y, rc)[fb], ll_bsp[fb], meta[fb], f"oracle blend {w}*BSP + p_cal vs BSP", False), "original")
    out(sec2)

    # ---- 3. where the gain lives ------------------------------------------------------------------
    d_head = np.full(R, np.nan)
    d_head[hasc] = ll_mkt[hasc] - wfc[hasc]
    d_qcal = np.full(R, np.nan)
    d_qcal[both] = ll_qcal[both] - wfc[both]
    date_r = meta["date"].to_numpy()

    def by(label_series, dvec, cl):
        rows = {}
        okd = ~np.isnan(dvec)
        for lab, m in label_series.items():
            mm = m & okd
            rows[str(lab)] = sub(dvec, mm, cl)
        return rows
    book_lab = pd.cut(meta["book_over_nw"], BOOK_BINS, right=False)
    groups_book = {str(b): (book_lab == b).to_numpy() for b in book_lab.cat.categories}
    stale_lab = pd.cut(meta["stale_pl_max"], STALE_BINS, right=False)
    groups_stale = {str(b): ((stale_lab == b) & meta["all_bsp"]).to_numpy() for b in stale_lab.cat.categories}
    stalew_lab = pd.cut(meta["stale_win_max"], STALE_BINS, right=False)
    groups_stalew = {str(b): ((stalew_lab == b) & meta["all_bsp_win"]).to_numpy() for b in stalew_lab.cat.categories}
    field_lab = pd.cut(meta["n"], FIELD_BINS, right=True)
    groups_field = {str(b): (field_lab == b).to_numpy() for b in field_lab.cat.categories}
    t30_lab = pd.cut(meta["t30_pl_share"], [-0.01, 0.0, 0.5, 1.0], right=True)
    groups_t30 = {str(b): (t30_lab == b).to_numpy() for b in t30_lab.cat.categories}
    short_lab = pd.cut(meta["min_ltp_pl"], [1.0, 1.1, 1.25, 1.5, 100.0], right=False)
    groups_short = {str(b): (short_lab == b).to_numpy() for b in short_lab.cat.categories}
    well = (meta["book_over_nw"].between(0.98, 1.02) & meta["all_bsp"] & (meta["stale_pl_max"] < 0.1)).to_numpy()
    tight_book = meta["book_over_nw"].between(0.98, 1.02).to_numpy()
    sec3 = {"section": "3_where_the_gain_lives (headline blend vs ltp market; and vs the recalibrated market)",
            "by_place_book_over_n_winners": {"vs_market": by(groups_book, d_head, date_r), "vs_q_cal": by(groups_book, d_qcal, date_r)},
            "by_place_ltp_staleness_max_abs_log_bsp_over_ltp (BSP subset)": {"vs_market": by(groups_stale, d_head, date_r), "vs_q_cal": by(groups_stale, d_qcal, date_r)},
            "by_WIN_ltp_staleness_max_abs_log_bsp_over_ltp (BSP subset)": {"vs_market": by(groups_stalew, d_head, date_r)},
            "by_code": {"vs_market": by({"gallops": (~meta["harness"]).to_numpy(), "harness": meta["harness"].to_numpy()}, d_head, date_r),
                        "vs_q_cal": by({"gallops": (~meta["harness"]).to_numpy(), "harness": meta["harness"].to_numpy()}, d_qcal, date_r)},
            "by_places_paid": {"vs_market": by({"2": (meta["n_winners"] == 2).to_numpy(), "3": (meta["n_winners"] == 3).to_numpy()}, d_head, date_r)},
            "by_field_size": {"vs_market": by(groups_field, d_head, date_r), "vs_q_cal": by(groups_field, d_qcal, date_r)},
            "by_place_t30_share (liquidity)": {"vs_market": by(groups_t30, d_head, date_r), "vs_q_cal": by(groups_t30, d_qcal, date_r)},
            "by_shortest_place_ltp_in_race": {"vs_market": by(groups_short, d_head, date_r), "vs_q_cal": by(groups_short, d_qcal, date_r)},
            "tight_book_0.98_1.02": {"share_of_races": r4(tight_book.mean()),
                                     "vs_market": record({**gain_stats(wfc[tight_book & hasc], ll_mkt[tight_book & hasc], meta[tight_book & hasc], "blend vs market, tight book", False)}, "added"),
                                     "vs_q_cal": record({**gain_stats(wfc[tight_book & both], ll_qcal[tight_book & both], meta[tight_book & both], "blend vs q_cal, tight book", False)}, "added")},
            "well_formed_(tight_book & all BSP & max|log bsp/ltp|<0.1)": {
                "share_of_races": r4(well.mean()),
                "vs_market": record(gain_stats(wfc[well & hasc], ll_mkt[well & hasc], meta[well & hasc], "blend vs market, well-formed markets", False), "added"),
                "vs_q_cal": record(gain_stats(wfc[well & both], ll_qcal[well & both], meta[well & both], "blend vs q_cal, well-formed markets", False), "added"),
                "vs_bsp": record(gain_stats(wfc[well & fb], ll_bsp[well & fb], meta[well & fb], "blend vs BSP, well-formed markets", False), "added")}}
    out(sec3)

    # ---- 4. design sensitivity --------------------------------------------------------------------
    sens = {}
    wfl, chosen_l, _ = wf_blend(q_mkt, pcal, y, rc, rmonth, rc_has, lookback=12)
    hl = ~np.isnan(wfl)
    sens["blend_w_from_previous_12_months_only"] = {**record(gain_stats(wfl[hl], ll_mkt[hl], meta[hl], "blend_wf_cal, w from previous 12 months", False), "added"),
                                                    "median_w_market": float(np.median(list(chosen_l.values()))) if chosen_l else None}
    for col, label in (("p_cal_lin", "linear logit calibration"), ("p_cal_w6", "6-month calibration window"),
                       ("p_cal_w24", "24-month calibration window"), ("p_cal_l1", "plain Harville lambda=1, cubic calibration")):
        pv = full[col].to_numpy()
        rh = race_mask(rc, full[col].notna().to_numpy())
        wfx, chx, _ = wf_blend(q_mkt, pv, y, rc, rmonth, rh)
        hx = ~np.isnan(wfx)
        common = hx & hasc
        sens[label] = {"blend_vs_market": record(gain_stats(wfx[hx], ll_mkt[hx], meta[hx], f"blend_wf, {label}, vs market", False), "added"),
                       "headline_on_common_races": r4((ll_mkt[common] - wfc[common]).mean()),
                       "variant_on_common_races": r4((ll_mkt[common] - wfx[common]).mean()),
                       "median_w_market": float(np.median(list(chx.values()))) if chx else None,
                       "model_alone_vs_market": record(gain_stats(w2p.logloss_race(np.nan_to_num(pv, nan=0.5), y, rc)[rh], ll_mkt[rh], meta[rh], f"{label} alone vs market", False), "added")}
    sens["lambda_path_sample"] = {f"{m}/{'harness' if c else 'gallops'}": [v[0], v[1]] for (m, c), v in sorted(lam.items()) if m[-2:] in ("01", "07")}
    out({"section": "4_design_sensitivity", **sens})

    # ---- 5. execution: the blend as a bet signal -------------------------------------------------
    full["p_blend"] = p_blend
    bt = full[full["p_blend"].notna()]
    cells = []
    for kk in (0.0, 0.05, 0.1):
        g = bt[bt["p_blend"] * bt["ltp_pl"] > 1 + kk]
        cells.append(bet_cell(g, mid, f"back place at ltp_pl when p_blend*ltp_pl>1+{kk:g}"))
    # the mirror: the market says the blend is too long -> lay would be needed; we only report backs
    out({"section": "5_execution_blend_as_bet_signal", "runners_with_blend": int(len(bt)), "cells_tested": len(cells), "cells": cells})

    # ---- 6. multiple testing ----------------------------------------------------------------------
    pv_race = [w2p.z_to_p(t["z_race"]) for t in tests]
    pv_cl = [w2p.z_to_p(t["z_cl_date"]) for t in tests]
    bh_race = w2p.bh(pv_race)
    bh_cl = w2p.bh(pv_cl)
    for t, f1, f2 in zip(tests, bh_race, bh_cl):
        t["bh_q05_race_se"] = f1
        t["bh_q05_cl_date_se"] = f2
    bet_p = []
    for c in cells:
        z = (c["ltp"]["roi_c5"] / c["ltp"]["se_cl_date"]) if c.get("ltp") and c["ltp"].get("se_cl_date") else None
        bet_p.append(w2p.z_to_p(z))
    out({"section": "6_multiple_testing", "probability_tests_here": len(tests), "original_probability_tests": 13,
         "bet_cells_here": len(cells), "bh_bet_cells_ltp_q05": w2p.bh(bet_p),
         "n_bh_pass_race_se": int(sum(bh_race)), "n_bh_pass_cl_date_se": int(sum(bh_cl)),
         "tests": tests})

    # ---- 7. corrected estimate --------------------------------------------------------------------
    hvq = sec2["headline_blend_vs_market_recalibrated"]
    out({"section": "7_corrected_estimate",
         "claimed_gain_vs_ltp_market": {"gain": 0.0102, "se": 0.0010, "halves": [0.0135, 0.0077]},
         "reproduced_gain_vs_ltp_market": {"gain": head["gain"], "se_race": head["se_race"], "se_cl_date": head["se_cl_date"],
                                           "se_cl_meeting": head["se_cl_meeting"], "se_month_block_boot": head["month_block_bootstrap"]["se"],
                                           "halves": head["halves_gain"], "halves_se_cl_date": head["halves_se_cl_date"], "by_year": {k_: v["gain"] for k_, v in head["by_year"].items()}},
         "gain_vs_market_recalibrated_same_routine": {"gain": hvq["gain"], "se_race": hvq["se_race"], "se_cl_date": hvq["se_cl_date"],
                                                      "halves": hvq["halves_gain"], "halves_se_cl_date": hvq["halves_se_cl_date"],
                                                      "by_year": {k_: v["gain"] for k_, v in hvq["by_year"].items()}},
         "share_of_claimed_gain_that_is_market_miscalibration": r4(sec2["market_recalibrated_vs_market"]["gain"] / sec2["headline_blend_on_same_races_vs_market"]["gain"])
         if sec2["headline_blend_on_same_races_vs_market"]["gain"] else None,
         "blend_vs_place_bsp": {"gain": sec2["place_bsp_benchmark"]["HEADLINE_blend_vs_bsp"]["gain"], "se_cl_date": sec2["place_bsp_benchmark"]["HEADLINE_blend_vs_bsp"]["se_cl_date"],
                                "halves": sec2["place_bsp_benchmark"]["HEADLINE_blend_vs_bsp"]["halves_gain"]},
         "runtime_s": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

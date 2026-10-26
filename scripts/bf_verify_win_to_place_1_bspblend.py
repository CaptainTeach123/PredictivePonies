"""Adversarial verification of the BENCHMARK claim of experiment "win_to_place"
(scripts/bf_win_to_place.py), lens: STATISTICS AND EXECUTION.

CLAIM (benchmark only, uses the post-race place BSP): on the 41,658 fully priced AU place
markets where every runner has a place BSP, the blend 0.75 * q_bsp + 0.25 * p_cal has per-race
log loss 4.7814 against 4.7837 for the normalised place BSP: gain 0.0023 (SE 0.0003), halves
0.0037 (SE 0.0004) and 0.0012 (SE 0.0005); p_cal alone is worse than the BSP (-0.0035, SE
0.0011). Interpretation offered: "the win-derived price carries a sliver of information even
beyond the final place market".

    python scripts/bf_verify_win_to_place_1_bspblend.py [--dir data/raw/betfair/split] [--frac 0.1]

The original pipeline is imported unchanged (join, lambda fit, cubic-logit calibration), so
section 0 reproduces the claimed numbers exactly. Then:
  1. SE robustness of the headline: per-race (as claimed), clustered by day and by meeting, a
     month-block bootstrap; splits not chosen after the fact (calendar years, thirds, monthly
     sign counts); heavy tails (median, trimmed and winsorised gain, share of the gain carried
     by the top 1% of races).
  2. Weight selection: the 0.75 was the best of three in-sample weights (0.5, 0.75, 0.9). The
     in-sample optimum over a 0.05 grid and a walk-forward weight (chosen each month on
     strictly earlier months) are reported.
  3. The confound the original did not test: is the gain *information* in the WIN-derived
     price, or just a correction of the place BSP's own calibration (proportional
     normalisation of a place book to n_winners is known to distort)? Controls that contain
     NO win-market information: blending the BSP with a uniform prior (n_winners / n), a
     power re-normalisation q ~ (1/bsp)^a, and a walk-forward cubic-logit recalibration of
     the BSP with the SAME routine p_cal gets. The apples-to-apples test of the information
     claim is then blend(recalibrated BSP, p_cal) vs recalibrated BSP, walk-forward weight.
  4. Is the WIN-derived price special? The same blends with the ltp place market (q_mkt, a
     place-market quantity, no win information), with the uncalibrated Harville p_fair, and
     with a Harville/calibrated price derived from the WIN BSP (the same post-race time point
     as the place BSP; the fairest "does the win market know more than the place market" test).
  5. Where the gain lives: by code, places paid, field size, BSP book over-round, year, and
     the calibration of q_bsp by probability band.
  6. Multiple testing: BH (q = 0.05) over the original's 13 probability tests plus every test
     added here, with the count.
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
from ponies.exotics import order_probs  # noqa: E402

N_BOOT = 2000
W_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
A_GRID = np.round(np.arange(0.5, 1.5001, 0.05), 2)
Q_BANDS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
FIELD_BINS = [0, 7, 10, 14, 100]
BOOK_BINS = [0.0, 0.98, 1.0, 1.02, 1.05, 1.1, 10.0]
# the original's 13 probability tests (gain, per-race SE), from its full run (b_logloss section),
# carried into the BH correction together with the tests added here
ORIGINAL_13 = {
    "orig:harville_vs_ltp_market": (0.0041, 0.0013), "orig:harville_cal_vs_ltp_market": (0.0070, 0.0013),
    "orig:blend_cal_w0.5_vs_ltp_market": (0.0099, 0.0008), "orig:blend_w0.25_vs_ltp_market": (0.0079, 0.0010),
    "orig:blend_w0.5_vs_ltp_market": (0.0084, 0.0007), "orig:blend_w0.75_vs_ltp_market": (0.0058, 0.0004),
    "orig:blend_wf_vs_ltp_market": (0.0086, 0.0009), "orig:blend_cal_wf_vs_ltp_market": (0.0102, 0.0010),
    "orig:ltp_market_vs_bsp": (-0.0109, 0.0014), "orig:p_cal_vs_bsp": (-0.0035, 0.0011),
    "orig:blend_0.5bsp_vs_bsp": (0.0023, 0.0005), "orig:blend_0.75bsp_vs_bsp": (0.0023, 0.0003),
    "orig:blend_0.9bsp_vs_bsp": (0.0012, 0.0002)}


def out(obj) -> None:
    print(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)), flush=True)


def r4(x):
    return w2p.r4(x)


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------
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
    x = d[m]
    if len(x) < 2:
        return {"races": int(len(x)), "gain": None}
    se = x.std(ddof=1) / math.sqrt(len(x))
    res = {"races": int(len(x)), "gain": r4(x.mean()), "se_race": r4(se), "z_race": r4(x.mean() / se) if se > 0 else None}
    if cl is not None:
        sc = cluster_se(x, np.asarray(cl)[m])
        res["se_cl_date"] = r4(sc)
        res["z_cl_date"] = r4(x.mean() / sc) if sc else None
    return res


TESTS: list[dict] = []


def gain_stats(ll_model, ll_base, meta: pd.DataFrame, name: str, full_stats=False, register=True) -> dict:
    """Paired per-race comparison: gain = base - model (positive = model better)."""
    ll_model = np.asarray(ll_model, dtype=float)
    ll_base = np.asarray(ll_base, dtype=float)
    d = ll_base - ll_model
    n = len(d)
    if n < 2:
        return {"test": name, "races": int(n)}
    se = d.std(ddof=1) / math.sqrt(n)
    res = {"test": name, "races": int(n), "ll_model": r4(ll_model.mean()), "ll_base": r4(ll_base.mean()),
           "gain": r4(d.mean()), "se_race": r4(se), "z_race": r4(d.mean() / se) if se > 0 else None}
    date = meta["date"].to_numpy()
    scd = cluster_se(d, date)
    res["se_cl_date"] = r4(scd)
    res["z_cl_date"] = r4(d.mean() / scd) if scd else None
    fh = meta["first_half"].to_numpy()
    res["halves_gain"] = [r4(d[fh].mean()), r4(d[~fh].mean())]
    res["halves_se_race"] = [r4(d[fh].std(ddof=1) / math.sqrt(max(fh.sum(), 2))), r4(d[~fh].std(ddof=1) / math.sqrt(max((~fh).sum(), 2)))]
    res["halves_se_cl_date"] = [r4(cluster_se(d[fh], date[fh])), r4(cluster_se(d[~fh], date[~fh]))]
    res["positive_gain_gt_2se_race"] = bool(se > 0 and d.mean() > 2 * se)
    res["positive_both_halves_2se_race"] = bool(all(g is not None and s is not None and g > 2 * s
                                                     for g, s in zip(res["halves_gain"], res["halves_se_race"])))
    res["positive_both_halves_2se_cl_date"] = bool(all(g is not None and s is not None and g > 2 * s
                                                        for g, s in zip(res["halves_gain"], res["halves_se_cl_date"])))
    if full_stats:
        res["se_cl_meeting"] = r4(cluster_se(d, meta["event_id"].to_numpy()))
        res["month_block_bootstrap"] = block_boot(d, meta["month"].to_numpy())
        res["by_year"] = {str(yv): sub(d, (meta["year"] == yv).to_numpy(), date) for yv in sorted(meta["year"].unique())}
        order = np.argsort(date, kind="stable")
        third = np.zeros(n, dtype=int)
        third[order[n // 3:2 * n // 3]] = 1
        third[order[2 * n // 3:]] = 2
        res["thirds_by_date"] = [sub(d, third == t, date) for t in range(3)]
        pm = pd.Series(d).groupby(meta["month"].to_numpy()).agg(["mean", "count"])
        res["months"] = {"n": int(len(pm)), "n_gain_positive": int((pm["mean"] > 0).sum()),
                         "min_month_gain": r4(pm["mean"].min()), "median_month_gain": r4(pm["mean"].median())}
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
    if register:
        TESTS.append({"test": name, "gain": float(d.mean()), "se": float(se), "races": int(n)})
    return res


def calibrate_wf(df: pd.DataFrame, col: str, months_sorted: list[str], window=12, degree=3, min_rows=5000) -> np.ndarray:
    """The original's cubic-logit walk-forward recalibration (previous ``window`` months,
    Newton), rescaled within the race to n_winners; degree=3, window=12 reproduces
    w2p.calibrate_walk_forward. NaN where no fit is available yet."""
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


def wf_select(lls: dict, rmonth: np.ndarray, min_prev=500, lookback: int | None = None, valid=None):
    """Walk-forward model selection: for each month pick the key whose summed per-race log loss
    over strictly earlier months (``lookback`` of them when given) is lowest; evaluate on that
    month. Races with ``valid`` False are ignored in the selection. Returns per-race ll (NaN
    where no choice yet) and the chosen keys."""
    keys = list(lls)
    months = sorted(set(rmonth))
    vmask = np.ones(len(rmonth)) if valid is None else np.asarray(valid, dtype=float)
    msum = pd.DataFrame({k: lls[k] * vmask for k in keys}).groupby(rmonth).sum()
    mcount = pd.Series(rmonth).value_counts()
    wf = np.full(len(rmonth), np.nan)
    chosen = {}
    for m in months:
        prev = [mm for mm in months if mm < m]
        if lookback:
            prev = prev[-lookback:]
        if not prev or sum(mcount[mm] for mm in prev) < min_prev:
            continue
        tot = msum.loc[prev].sum(axis=0)
        best = tot.idxmin()
        chosen[m] = best
        sel = rmonth == m
        wf[sel] = lls[best][sel]
    return wf, chosen


def derived_from_P(j: pd.DataFrame, P: np.ndarray, nw: np.ndarray, lam: dict, months, codes, sizes) -> np.ndarray:
    """Discounted-Harville place probabilities from the padded win-probability matrix P (one row
    per race, zeros beyond the field); NaN for races whose P is incomplete or without lambdas."""
    R, W = P.shape
    Q = np.full((R, W), np.nan)
    for r in range(R):
        key = (months[r], codes[r])
        if key not in lam:
            continue
        n = int(sizes[r])
        p = P[r, :n]
        if not np.all(np.isfinite(p)) or np.any(p <= 0):
            continue
        l2, l3, _ = lam[key]
        o = order_probs(p, l2, l3)
        Q[r, :n] = o["top3"] if nw[r] == 3 else o["top2"]
    return Q[j["race"].to_numpy(), j["pos"].to_numpy()]


def bh(pvals: list[float], q=0.05) -> list[bool]:
    return w2p.bh(pvals, q)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/raw/betfair/split"))
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    # ---- the original pipeline, unchanged --------------------------------------------------------
    j = w2p.load(a.dir, a.frac, a.seed)
    P, jw, k, i, ok, nw, race = w2p.placed_set_stats(j)
    months = race["month"].to_numpy()
    codes = race["harness"].to_numpy()
    months_sorted = sorted(set(months))
    ll = w2p.grid_loglik(P, jw, k, i, ok, nw)
    lam = w2p.fit_lambdas_walk_forward(ll, ok, months, codes, months_sorted)
    del ll
    j["p_fair"] = w2p.derived_place_probs(j, P, nw, lam, months, codes)
    # added: the same derivation from the WIN BSP (post-race; same time point as the place BSP)
    hbw = (j["bsp_win"].notna() & (j["bsp_win"] > 1)).to_numpy()
    allw = pd.Series(hbw).groupby(j["race"].to_numpy()).transform("all").to_numpy()
    inv = np.where(hbw & allw, 1 / np.where(hbw, j["bsp_win"].to_numpy(dtype=float), 1.0), 0.0)
    j["p_win_bsp"] = inv / np.where(allw, pd.Series(inv).groupby(j["race"].to_numpy()).transform("sum").to_numpy(), 1.0)
    sizes = np.bincount(j["race"].to_numpy())
    Pb = w2p.pad(j, "p_win_bsp", 0.0)
    j["p_fair_bsp"] = derived_from_P(j, Pb, nw, lam, months, codes, sizes)
    del P, Pb
    j = j[j["p_fair"].notna()].copy()
    j["race"] = pd.factorize(j["win_mid"])[0]
    j["dh"] = j["n_winners"] / j["n_placed"]
    jm = sorted(j["month"].unique())
    j["p_cal"], cal_params = w2p.calibrate_walk_forward(j, "p_fair", jm)
    jb = j[j["p_fair_bsp"].notna()]
    j["p_cal_bsp"] = np.nan
    j.loc[jb.index, "p_cal_bsp"] = w2p.calibrate_walk_forward(jb, "p_fair_bsp", sorted(jb["month"].unique()))[0]
    days = sorted(j["date"].unique())
    mid = days[len(days) // 2]
    t_pipe = time.time() - t0

    # ---- fully priced place markets (the original's population) ---------------------------------
    j["all_priced"] = j.groupby("race")["ltp_pl"].transform(lambda s: s.notna().all() and (s > 1).all())
    full = j[j["all_priced"]].copy()
    book = full.groupby("race")["ltp_pl"].transform(lambda s: (1 / s).sum())
    full["q_mkt"] = (1 / full["ltp_pl"]) * full["n_winners"] / book
    full = full.sort_values(["race", "pos"]).reset_index(drop=True)
    full["q_cal"] = calibrate_wf(full, "q_mkt", sorted(full["month"].unique()))
    del j

    # ---- the benchmark population: every runner has a place BSP, p_cal present ------------------
    fb = full.copy()
    fb["all_bsp"] = fb.groupby("race")["bsp_pl"].transform(lambda s: s.notna().all() and (s > 1).all())
    fb = fb[fb["all_bsp"] & fb["p_cal"].notna()].copy().reset_index(drop=True)
    del full
    bsum = fb.groupby("race")["bsp_pl"].transform(lambda s: (1 / s).sum())
    fb["q_bsp"] = (1 / fb["bsp_pl"]) * fb["n_winners"] / bsum
    fb["book_bsp_over_nw"] = bsum / fb["n_winners"]
    fb["q_uni"] = fb["n_winners"] / fb["n"]
    fb["q_bsp_cal"] = calibrate_wf(fb, "q_bsp", sorted(fb["month"].unique()))
    y = fb["placed"].to_numpy(dtype=float)
    rc = pd.factorize(fb["race"])[0]
    races = fb.drop_duplicates("race").copy()
    meta = pd.DataFrame({"date": races["date"].to_numpy(), "event_id": races["event_id"].to_numpy(),
                         "month": races["month"].to_numpy(), "year": pd.DatetimeIndex(races["date"]).year,
                         "harness": races["harness"].to_numpy(), "n_winners": races["n_winners"].to_numpy(),
                         "n": races["n"].to_numpy(), "book": races["book_bsp_over_nw"].to_numpy()})
    meta["first_half"] = meta["date"] < mid
    meta["min_bsp"] = fb.groupby(rc)["bsp_pl"].min().to_numpy()
    meta["max_bsp"] = fb.groupby(rc)["bsp_pl"].max().to_numpy()
    # thin-market proxy: how far the place BSP ended from the last pre-race traded price
    stale = np.abs(np.log(fb["bsp_pl"].to_numpy(dtype=float) / fb["ltp_pl"].to_numpy(dtype=float)))
    meta["stale_max"] = pd.Series(stale).groupby(rc).max().to_numpy()
    meta["t30_present"] = pd.Series(fb["t30_pl"].notna().to_numpy()).groupby(rc).mean().to_numpy() > 0.5
    rmonth = meta["month"].to_numpy()
    date_r = meta["date"].to_numpy()

    def ll_of(q):
        return w2p.logloss_race(np.asarray(q, dtype=float), y, rc)

    def blends(qa, pb, grid=W_GRID):
        pb = np.asarray(pb, dtype=float)
        return {round(float(w), 2): ll_of(w * np.asarray(qa, dtype=float) + (1 - w) * pb) for w in grid}

    q_bsp = fb["q_bsp"].to_numpy(dtype=float)
    p_cal = fb["p_cal"].to_numpy(dtype=float)
    p_fair = fb["p_fair"].to_numpy(dtype=float)
    q_mkt = fb["q_mkt"].to_numpy(dtype=float)
    q_uni = fb["q_uni"].to_numpy(dtype=float)
    ll_bsp = ll_of(q_bsp)
    ll_cal = ll_of(p_cal)
    ll_mkt = ll_of(q_mkt)

    # ---- 0. reproduction ---------------------------------------------------------------------------
    rep = {"section": "0_reproduction", "runtime_pipeline_s": round(t_pipe, 1), "races": int(len(ll_bsp)),
           "runners": int(len(fb)), "from": str(pd.Timestamp(meta["date"].min()).date()),
           "to": str(pd.Timestamp(meta["date"].max()).date()), "mid": str(pd.Timestamp(mid).date()),
           "ll_bsp": r4(ll_bsp.mean()), "cal_params_last": cal_params[max(cal_params)],
           "claimed": {"ll_blend": 4.7814, "ll_bsp": 4.7837, "gain": 0.0023, "se": 0.0003, "halves": [0.0037, 0.0012],
                       "halves_se": [0.0004, 0.0005], "races": 41658, "p_cal_vs_bsp": [-0.0035, 0.0011]}}
    rep["ltp_market_vs_bsp"] = gain_stats(ll_mkt, ll_bsp, meta, "ltp place market vs BSP")
    rep["p_cal_vs_bsp"] = gain_stats(ll_cal, ll_bsp, meta, "p_cal vs BSP")
    bl_cal = blends(q_bsp, p_cal)
    for w in (0.5, 0.75, 0.9):
        rep[f"blend_{w}bsp_pcal_vs_bsp"] = gain_stats(bl_cal[w], ll_bsp, meta, f"oracle blend {w}*BSP+p_cal vs BSP (original)",
                                                       full_stats=(w == 0.75))
    out(rep)

    # ---- 1/2. weight selection ------------------------------------------------------------------
    best_w = min(bl_cal, key=lambda w: bl_cal[w].sum())
    curve = {str(w): r4((ll_bsp - bl_cal[w]).mean()) for w in bl_cal}
    wf, chosen = wf_select(bl_cal, rmonth)
    has = ~np.isnan(wf)
    wf12, chosen12 = wf_select(bl_cal, rmonth, lookback=12)
    has12 = ~np.isnan(wf12)
    sec = {"section": "1_weight_selection", "in_sample_best_w_on_bsp": best_w, "gain_curve_by_w": curve,
           "in_sample_best_vs_bsp": gain_stats(bl_cal[best_w], ll_bsp, meta, f"in-sample best blend w={best_w} vs BSP"),
           "walk_forward_vs_bsp": {**gain_stats(wf[has], ll_bsp[has], meta[has], "walk-forward blend(BSP,p_cal) vs BSP", full_stats=True),
                                   "median_w": float(np.median(list(chosen.values()))) if chosen else None,
                                   "w_by_month_sample": {m: chosen[m] for m in sorted(chosen)[::6]}},
           "walk_forward_12m_vs_bsp": {**gain_stats(wf12[has12], ll_bsp[has12], meta[has12], "walk-forward(12m) blend(BSP,p_cal) vs BSP"),
                                       "median_w": float(np.median(list(chosen12.values()))) if chosen12 else None}}
    out(sec)

    # ---- 3. calibration confound: controls with NO win-market information -------------------------
    sec = {"section": "3_calibration_controls_no_win_information"}
    # 3a uniform prior
    bl_uni = blends(q_bsp, q_uni)
    bw = min(bl_uni, key=lambda w: bl_uni[w].sum())
    wfu, chu = wf_select(bl_uni, rmonth)
    hu = ~np.isnan(wfu)
    sec["blend_bsp_uniform"] = {"in_sample_best_w": bw, "gain_curve_by_w": {str(w): r4((ll_bsp - bl_uni[w]).mean()) for w in bl_uni},
                                "in_sample_best_vs_bsp": gain_stats(bl_uni[bw], ll_bsp, meta, f"in-sample best blend(BSP,uniform) w={bw} vs BSP"),
                                "walk_forward_vs_bsp": {**gain_stats(wfu[hu], ll_bsp[hu], meta[hu], "walk-forward blend(BSP,uniform) vs BSP", full_stats=True),
                                                        "median_w": float(np.median(list(chu.values()))) if chu else None}}
    # 3b power re-normalisation q ~ (1/bsp)^a
    invb = 1 / fb["bsp_pl"].to_numpy(dtype=float)
    nwr = fb["n_winners"].to_numpy(dtype=float)
    bl_pow = {}
    for aa in A_GRID:
        v = invb ** aa
        s = np.bincount(rc, weights=v)[rc]
        bl_pow[round(float(aa), 2)] = ll_of(np.minimum(v * nwr / s, 0.999))
    ba = min(bl_pow, key=lambda w: bl_pow[w].sum())
    wfp, chp = wf_select(bl_pow, rmonth)
    hp = ~np.isnan(wfp)
    sec["power_normalised_bsp"] = {"in_sample_best_a": ba, "gain_curve_by_a": {str(aa): r4((ll_bsp - bl_pow[aa]).mean()) for aa in bl_pow},
                                   "in_sample_best_vs_bsp": gain_stats(bl_pow[ba], ll_bsp, meta, f"in-sample best power a={ba} vs BSP"),
                                   "walk_forward_vs_bsp": {**gain_stats(wfp[hp], ll_bsp[hp], meta[hp], "walk-forward power-normalised BSP vs BSP"),
                                                           "median_a": float(np.median(list(chp.values()))) if chp else None}}
    # 3c cubic-logit recalibration of the BSP (same routine as p_cal, previous 12 months)
    q_bc = fb["q_bsp_cal"].to_numpy(dtype=float)
    hc = np.bincount(rc, weights=np.isfinite(q_bc).astype(float)) == np.bincount(rc)
    ll_bc = ll_of(np.nan_to_num(q_bc, nan=0.5))
    sec["recalibrated_bsp_vs_bsp"] = gain_stats(ll_bc[hc], ll_bsp[hc], meta[hc], "cubic-logit recalibrated BSP vs BSP", full_stats=True)
    sec["recalibrated_bsp_races"] = int(hc.sum())
    # the headline blend on the same races, for a like-for-like comparison with the recalibration
    sec["oracle_blend_0.75_vs_bsp_same_races"] = gain_stats(bl_cal[0.75][hc], ll_bsp[hc], meta[hc], "oracle blend 0.75 vs BSP (recal races)", register=False)
    # recalibrated BSP + uniform: is there shrinkage left after recalibration? (should be ~0)
    bl_bcu = blends(np.nan_to_num(q_bc, nan=0.5), q_uni)
    wfbu, chbu = wf_select(bl_bcu, rmonth, valid=hc)
    hbu = ~np.isnan(wfbu) & hc
    sec["recal_bsp_plus_uniform_wf_vs_recal_bsp"] = {**gain_stats(wfbu[hbu], ll_bc[hbu], meta[hbu], "walk-forward blend(recal BSP,uniform) vs recal BSP"),
                                                     "median_w": float(np.median(list(chbu.values()))) if chbu else None}
    out(sec)

    # ---- 3d. the apples-to-apples information test: blend(recal BSP, p_cal) vs recal BSP -------
    sec = {"section": "3d_information_test_after_recalibration", "races": int(hc.sum())}
    bl_bc_cal = blends(np.nan_to_num(q_bc, nan=0.5), p_cal)
    bwc = min(bl_bc_cal, key=lambda w: bl_bc_cal[w][hc].sum())
    wfc, chc = wf_select(bl_bc_cal, rmonth, valid=hc)
    hcc = ~np.isnan(wfc) & hc
    sec["gain_curve_by_w_vs_recal_bsp"] = {str(w): r4((ll_bc[hc] - bl_bc_cal[w][hc]).mean()) for w in bl_bc_cal}
    sec["in_sample_best_w"] = bwc
    sec["in_sample_best_vs_recal_bsp"] = gain_stats(bl_bc_cal[bwc][hc], ll_bc[hc], meta[hc], f"in-sample best blend(recal BSP,p_cal) w={bwc} vs recal BSP")
    sec["walk_forward_vs_recal_bsp"] = {**gain_stats(wfc[hcc], ll_bc[hcc], meta[hcc], "walk-forward blend(recal BSP,p_cal) vs recal BSP", full_stats=True),
                                        "median_w": float(np.median(list(chc.values()))) if chc else None,
                                        "w_by_month_sample": {m: chc[m] for m in sorted(chc)[::6]}}
    sec["walk_forward_vs_raw_bsp"] = gain_stats(wfc[hcc], ll_bsp[hcc], meta[hcc], "walk-forward blend(recal BSP,p_cal) vs raw BSP")
    sec["p_cal_vs_recal_bsp"] = gain_stats(ll_cal[hc], ll_bc[hc], meta[hc], "p_cal vs recal BSP")
    # p_cal recalibrated on the same population/window as the BSP recalibration (fairness)
    fb["p_cal2"] = calibrate_wf(fb, "p_cal", sorted(fb["month"].unique()))
    p_cal2 = fb["p_cal2"].to_numpy(dtype=float)
    h2 = hc & (np.bincount(rc, weights=np.isfinite(p_cal2).astype(float)) == np.bincount(rc))
    bl2 = blends(np.nan_to_num(q_bc, nan=0.5), np.nan_to_num(p_cal2, nan=0.5))
    wf2, ch2 = wf_select(bl2, rmonth, valid=h2)
    h2c = h2 & ~np.isnan(wf2)
    sec["walk_forward_blend_recal_bsp_recal_pcal_vs_recal_bsp"] = {
        **gain_stats(wf2[h2c], ll_bc[h2c], meta[h2c], "walk-forward blend(recal BSP, re-recal p_cal) vs recal BSP"),
        "median_w": float(np.median(list(ch2.values()))) if ch2 else None}
    out(sec)

    # ---- 4. is the WIN-derived price special? -----------------------------------------------------
    sec = {"section": "4_other_partners_for_the_bsp"}

    def partner(name, pb, mask=None, base=None, base_name="BSP"):
        mask = np.ones(len(ll_bsp), dtype=bool) if mask is None else mask
        base = ll_bsp if base is None else base
        bl = blends(q_bsp if base is ll_bsp else np.nan_to_num(q_bc, nan=0.5), np.nan_to_num(pb, nan=0.5))
        bw_ = min(bl, key=lambda w: bl[w][mask].sum())
        wf_, ch_ = wf_select(bl, rmonth, valid=mask)
        h_ = ~np.isnan(wf_) & mask
        return {"in_sample_best_w": bw_,
                "in_sample_best_vs_base": gain_stats(bl[bw_][mask], base[mask], meta[mask], f"in-sample best blend({base_name},{name}) w={bw_} vs {base_name}"),
                "walk_forward_vs_base": {**gain_stats(wf_[h_], base[h_], meta[h_], f"walk-forward blend({base_name},{name}) vs {base_name}"),
                                         "median_w": float(np.median(list(ch_.values()))) if ch_ else None},
                "partner_alone_vs_base": gain_stats(ll_of(np.nan_to_num(pb, nan=0.5))[mask], base[mask], meta[mask], f"{name} alone vs {base_name}")}
    sec["ltp_place_market_q_mkt"] = partner("q_mkt (ltp place market)", q_mkt)
    q_cal = fb["q_cal"].to_numpy(dtype=float)
    hq = np.bincount(rc, weights=np.isfinite(q_cal).astype(float)) == np.bincount(rc)
    sec["ltp_place_market_recalibrated_q_cal"] = partner("q_cal (recalibrated ltp place market)", q_cal, hq)
    sec["uncalibrated_harville_p_fair"] = partner("p_fair (uncalibrated Harville)", p_fair)
    # WIN BSP-derived (same post-race time point as the place BSP)
    pcb = fb["p_cal_bsp"].to_numpy(dtype=float)
    hpb = np.bincount(rc, weights=np.isfinite(pcb).astype(float)) == np.bincount(rc)
    sec["win_bsp_derived_p_cal_bsp"] = {"races": int(hpb.sum()), **partner("p_cal_bsp (Harville from WIN BSP, calibrated)", pcb, hpb)}
    sec["win_bsp_derived_after_recalibration"] = {"races": int((hpb & hc).sum()),
                                                  **partner("p_cal_bsp", pcb, hpb & hc, base=ll_bc, base_name="recal BSP")}
    # and the ltp-derived p_cal where the WIN-BSP-derived one is available (same races)
    sec["ltp_derived_p_cal_same_races_as_win_bsp"] = partner("p_cal", p_cal, hpb & hc, base=ll_bc, base_name="recal BSP")
    out(sec)

    # ---- 4b. three-way: does p_cal add anything once the BSP is already blended with the
    #      pre-race snapshot of the SAME place market (q_cal)? and the reverse ------------------
    def wf_blend_probs(qa, pb, valid):
        """walk-forward blend as a per-runner probability (NaN where no weight chosen yet)."""
        pb = np.nan_to_num(np.asarray(pb, dtype=float), nan=0.5)
        bl = blends(qa, pb)
        wf_, ch_ = wf_select(bl, rmonth, valid=valid)
        wm = np.array([ch_.get(m, np.nan) for m in rmonth])[rc]
        q = wm * np.asarray(qa, dtype=float) + (1 - wm) * pb
        return q, ~np.isnan(wf_) & valid, ch_

    sec = {"section": "4b_three_way_blends"}
    q_qc, h_qc, _ = wf_blend_probs(q_bsp, q_cal, hq)
    ll_qc2 = ll_of(np.nan_to_num(q_qc, nan=0.5))
    bl3 = blends(np.nan_to_num(q_qc, nan=0.5), p_cal)
    wf3, ch3 = wf_select(bl3, rmonth, valid=h_qc)
    h3 = ~np.isnan(wf3) & h_qc
    bw3 = min(bl3, key=lambda w: bl3[w][h_qc].sum())
    sec["p_cal_added_to_wf_blend(BSP,q_cal)"] = {
        "races": int(h3.sum()), "in_sample_best_w_on_blend": bw3,
        "gain_curve_by_w": {str(w): r4((ll_qc2[h_qc] - bl3[w][h_qc]).mean()) for w in bl3},
        "in_sample_vs_blend_bsp_qcal": gain_stats(bl3[bw3][h_qc], ll_qc2[h_qc], meta[h_qc], f"in-sample best 3-way +p_cal w={bw3} vs blend(BSP,q_cal)"),
        "walk_forward_vs_blend_bsp_qcal": {**gain_stats(wf3[h3], ll_qc2[h3], meta[h3], "walk-forward 3-way +p_cal vs blend(BSP,q_cal)", full_stats=True),
                                           "median_w_on_blend": float(np.median(list(ch3.values()))) if ch3 else None},
        "walk_forward_vs_raw_bsp": gain_stats(wf3[h3], ll_bsp[h3], meta[h3], "walk-forward 3-way (BSP,q_cal)+p_cal vs raw BSP")}
    q_pc, h_pc, _ = wf_blend_probs(q_bsp, p_cal, np.ones(len(ll_bsp), dtype=bool))
    ll_pc2 = ll_of(np.nan_to_num(q_pc, nan=0.5))
    bl3r = blends(np.nan_to_num(q_pc, nan=0.5), np.nan_to_num(q_cal, nan=0.5))
    wf3r, ch3r = wf_select(bl3r, rmonth, valid=h_pc & hq)
    h3r = ~np.isnan(wf3r) & h_pc & hq
    sec["q_cal_added_to_wf_blend(BSP,p_cal)"] = {
        "races": int(h3r.sum()),
        "walk_forward_vs_blend_bsp_pcal": {**gain_stats(wf3r[h3r], ll_pc2[h3r], meta[h3r], "walk-forward 3-way +q_cal vs blend(BSP,p_cal)"),
                                           "median_w_on_blend": float(np.median(list(ch3r.values()))) if ch3r else None}}
    # the WIN-BSP-derived partner with the full statistics (by year etc.)
    q_pb, h_pb2, _ = wf_blend_probs(q_bsp, pcb, hpb)
    sec["win_bsp_derived_wf_blend_full_stats"] = gain_stats(ll_of(np.nan_to_num(q_pb, nan=0.5))[h_pb2], ll_bsp[h_pb2], meta[h_pb2],
                                                            "walk-forward blend(BSP,p_cal_bsp) vs BSP (full stats)", full_stats=True, register=False)
    # p_cal_bsp added on top of (BSP, q_cal)
    bl3b = blends(np.nan_to_num(q_qc, nan=0.5), np.nan_to_num(pcb, nan=0.5))
    wf3b, ch3b = wf_select(bl3b, rmonth, valid=h_qc & hpb)
    h3b = ~np.isnan(wf3b) & h_qc & hpb
    sec["p_cal_bsp_added_to_wf_blend(BSP,q_cal)"] = {
        "races": int(h3b.sum()),
        "walk_forward_vs_blend_bsp_qcal": {**gain_stats(wf3b[h3b], ll_qc2[h3b], meta[h3b], "walk-forward 3-way +p_cal_bsp vs blend(BSP,q_cal)", full_stats=True),
                                           "median_w_on_blend": float(np.median(list(ch3b.values()))) if ch3b else None}}
    out(sec)

    # ---- 5. where the gain lives ----------------------------------------------------------------
    d_head = ll_bsp - bl_cal[0.75]
    d_uni = ll_bsp - bl_uni[bw]
    d_rec = ll_bsp - ll_bc
    d_info = np.where(hcc, ll_bc - wfc, np.nan)
    fs_lab = pd.cut(meta["n"], FIELD_BINS)
    bk_lab = pd.cut(meta["book"], BOOK_BINS)
    groups = {"gallops": (~meta["harness"]).to_numpy(), "harness": meta["harness"].to_numpy(),
              "places_2": (meta["n_winners"] == 2).to_numpy(), "places_3": (meta["n_winners"] == 3).to_numpy(),
              "min_bsp_le_1.1": (meta["min_bsp"] <= 1.1).to_numpy(), "min_bsp_gt_1.1": (meta["min_bsp"] > 1.1).to_numpy(),
              "max_bsp_ge_100": (meta["max_bsp"] >= 100).to_numpy(), "max_bsp_lt_100": (meta["max_bsp"] < 100).to_numpy()}
    for b in fs_lab.cat.categories:
        groups[f"field_{b}"] = (fs_lab == b).to_numpy()
    for b in bk_lab.cat.categories:
        groups[f"bsp_book_over_nw_{b}"] = (bk_lab == b).to_numpy()
    st_lab = pd.cut(meta["stale_max"], [0.0, 0.05, 0.1, 0.2, 0.4, 10.0], right=False)
    for b in st_lab.cat.categories:
        groups[f"max_abs_log_bsp_over_ltp_{b}"] = (st_lab == b).to_numpy()
    groups["t30_present_liquid"] = meta["t30_present"].to_numpy()
    groups["t30_absent"] = (~meta["t30_present"]).to_numpy()
    for yv in sorted(meta["year"].unique()):
        groups[f"year_{yv}"] = (meta["year"] == yv).to_numpy()

    def by(d, name):
        return {g: sub(d, m & np.isfinite(d), date_r) for g, m in groups.items()}
    sec = {"section": "5_where_the_gain_lives",
           "bsp_book_over_nw_quantiles": {q: r4(np.quantile(meta["book"], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
           "oracle_blend_0.75_vs_bsp": by(d_head, "head"),
           "blend_bsp_uniform_in_sample_vs_bsp": by(d_uni, "uni"),
           "recalibrated_bsp_vs_bsp": by(np.where(hc, d_rec, np.nan), "rec"),
           "walk_forward_blend_recal_bsp_pcal_vs_recal_bsp": by(d_info, "info")}
    # calibration of the place BSP by probability band (runner level): is the BSP over-confident?
    qb_lab = pd.cut(fb["q_bsp"], Q_BANDS, right=False)
    rows = []
    q_head = 0.75 * q_bsp + 0.25 * p_cal
    for b, g in fb.assign(_q=q_head, _bc=np.nan_to_num(q_bc, nan=np.nan)).groupby(qb_lab, observed=True):
        rows.append({"q_bsp_band": str(b), "runners": int(len(g)), "mean_q_bsp": r4(g["q_bsp"].mean()), "strike": r4(g["placed"].mean()),
                     "strike_over_q_bsp": r4(g["placed"].mean() / g["q_bsp"].mean()), "mean_p_cal": r4(g["p_cal"].mean()),
                     "mean_blend_0.75": r4(g["_q"].mean()), "mean_recal_bsp": r4(g["_bc"].mean())})
    sec["bsp_calibration_by_band"] = rows
    out(sec)

    # ---- 6. multiple testing -----------------------------------------------------------------------
    names = list(ORIGINAL_13) + [t["test"] for t in TESTS]
    zs = [g / s for g, s in ORIGINAL_13.values()] + [t["gain"] / t["se"] if t["se"] > 0 else 0.0 for t in TESTS]
    pv = [w2p.z_to_p(z) for z in zs]
    flags = bh(pv)
    out({"section": "6_multiple_testing", "tests_total": len(names), "original_tests": len(ORIGINAL_13), "added_tests": len(TESTS),
         "bh_q05_pass": [n for n, f in zip(names, flags) if f],
         "bh_q05_fail": [n for n, f in zip(names, flags) if not f],
         "added_tests_z_race": {t["test"]: r4(t["gain"] / t["se"]) if t["se"] > 0 else None for t in TESTS}})

    # ---- 7. summary ------------------------------------------------------------------------------
    head = rep["blend_0.75bsp_pcal_vs_bsp"]
    out({"section": "7_summary", "runtime_s": round(time.time() - t0, 1),
         "claimed_gain_reproduced": [head["gain"], head["se_race"], head["halves_gain"]],
         "headline_se_cl_date": head["se_cl_date"], "headline_se_cl_meeting": head["se_cl_meeting"],
         "headline_month_boot": head["month_block_bootstrap"], "headline_by_year": {k: v["gain"] for k, v in head["by_year"].items()},
         "walk_forward_blend_bsp_pcal_vs_bsp": sec_wf_gain(TESTS, "walk-forward blend(BSP,p_cal) vs BSP"),
         "blend_bsp_uniform_wf_vs_bsp": sec_wf_gain(TESTS, "walk-forward blend(BSP,uniform) vs BSP"),
         "recalibrated_bsp_vs_bsp": sec_wf_gain(TESTS, "cubic-logit recalibrated BSP vs BSP"),
         "info_after_recal_wf": sec_wf_gain(TESTS, "walk-forward blend(recal BSP,p_cal) vs recal BSP"),
         "info_after_recal_in_sample": sec_wf_gain(TESTS, f"in-sample best blend(recal BSP,p_cal) w={bwc} vs recal BSP"),
         "win_bsp_derived_after_recal_wf": sec_wf_gain(TESTS, "walk-forward blend(recal BSP,p_cal_bsp) vs recal BSP")})
    return 0


def sec_wf_gain(tests: list[dict], name: str):
    for t in tests:
        if t["test"] == name:
            return {"gain": r4(t["gain"]), "se_race": r4(t["se"]), "z": r4(t["gain"] / t["se"]) if t["se"] > 0 else None, "races": t["races"]}
    return None


if __name__ == "__main__":
    sys.exit(main())

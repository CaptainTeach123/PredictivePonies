"""Adversarial verification #1 of experiment "win_to_place" (scripts/bf_win_to_place.py),
lens: STATISTICS AND EXECUTION.

CLAIM under test: back the PLACE at ltp_pre when ltp_pre * p_cal > 1.10 (p_cal =
calibrated discounted-Harville place probability from the WIN market's ltp_pre):
+10.8% ROI after 5% commission on 107,224 bets (SE 1.2%), > 2 SE in both halves,
"paper-only" because the same bets settled at the place BSP return -4.6%.

    python scripts/bf_verify_win_to_place_1.py [--dir data/raw/betfair/split] [--frac 0.1]

The original pipeline is imported unchanged (same join, lambda fit, calibration),
so the headline cell is reproduced exactly; the script then adds what the
original did not run:

  1. SE with bets clustered by race, by meeting (event_id), by day, and a
     month-block bootstrap (46 blocks) at ltp_pre and at the place BSP;
  2. period splits not chosen after the fact: calendar years, thirds, monthly
     sign counts, halves by race count; all at ltp_pre and at BSP;
  3. commission 0/5/6/8/10% (AU exchange racing commission is often 6-10%);
  4. execution: a back limit order at ltp_pre can only be matched if the market
     later trades at or above it; proxy "filled iff place BSP >= ltp_pre" (and
     lenient variants), ROI of the filled bets at ltp_pre; market-order
     fallback (worse of ltp and BSP); SP bet (settle at BSP);
  5. liquidity: ROI at ltp_pre in markets with NO place BSP (no SP could form)
     versus markets with one; staleness gradient: ROI at ltp and at BSP by
     "edge" bucket (if "edge" measured value, both should rise with it);
  6. multiple testing of the executable family: the original's 144 place cells
     re-evaluated at BSP settlement (positive criterion, BH q=0.05);
  7. tuning: pick the best cell on one half at ltp_pre, test the other half at
     BSP; k grid and p choice at BSP;
  8. heavy tails: ROI/SE excluding 15+ prices, winsorised returns, bootstrap SE.
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

K = 0.1
COMMS5 = [0.0, 0.05, 0.06, 0.08, 0.10]
FILL_THRESHOLDS = [1.0, 0.98, 0.95, 0.9, 0.8]
N_BOOT = 1000


def out(obj) -> None:
    print(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)), flush=True)


def r4(x):
    return w2p.r4(x)


def cluster_se(ret: np.ndarray, cl: np.ndarray) -> float | None:
    """SE of the mean of ``ret`` with observations clustered by ``cl`` (sum of
    within-cluster demeaned returns, squared, summed over clusters)."""
    ok = ~np.isnan(ret)
    ret, cl = ret[ok], np.asarray(cl)[ok]
    if len(ret) < 2:
        return None
    d = pd.Series(ret - ret.mean()).groupby(cl).sum().to_numpy()
    return float(math.sqrt((d ** 2).sum()) / len(ret))


def block_boot_se(ret: np.ndarray, block: np.ndarray, n_boot=N_BOOT, seed=0) -> dict:
    """Block bootstrap of the mean return: resample blocks (e.g. months) with
    replacement, ROI = total return / total bets in the resample."""
    ok = ~np.isnan(ret)
    ret, block = ret[ok], np.asarray(block)[ok]
    s = pd.Series(ret).groupby(block).agg(["sum", "count"])
    sums, cnts = s["sum"].to_numpy(), s["count"].to_numpy()
    B = len(sums)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, B, size=(n_boot, B))
    rois = sums[idx].sum(1) / cnts[idx].sum(1)
    return {"blocks": int(B), "se": r4(rois.std(ddof=1)), "ci95": [r4(np.quantile(rois, 0.025)), r4(np.quantile(rois, 0.975))],
            "share_resamples_roi_le_0": r4((rois <= 0).mean())}


def iid_boot_se(ret: np.ndarray, n_boot=400, seed=1) -> float:
    ret = ret[~np.isnan(ret)]
    rng = np.random.default_rng(seed)
    n = len(ret)
    m = np.array([ret[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return r4(m.std(ddof=1))


def basic(ret: np.ndarray) -> dict:
    ret = ret[~np.isnan(ret)]
    if len(ret) < 2:
        return {"bets": int(len(ret)), "roi": None, "se": None}
    se = ret.std(ddof=1) / math.sqrt(len(ret))
    return {"bets": int(len(ret)), "roi": r4(ret.mean()), "se": r4(se), "z": r4(ret.mean() / se) if se > 0 else None}


def cell(g: pd.DataFrame, mid, price_col="ltp_pl", c=0.05, clusters=("race", "event_id", "date")) -> dict:
    """ROI at commission c with per-bet SE, clustered SEs, halves (median-day split)."""
    res = {"bets": int(len(g))}
    if len(g) < 2:
        return res
    r = w2p.returns(g[price_col], g["placed"], g["dh"], c)
    b = basic(r)
    res.update({"roi": b["roi"], "se_per_bet": b["se"], "z_per_bet": b["z"]})
    for cl in clusters:
        se = cluster_se(r, g[cl].to_numpy())
        res[f"se_cl_{cl}"] = r4(se)
    fh = (g["date"] < mid).to_numpy()
    res["halves_roi"] = [basic(r[fh])["roi"], basic(r[~fh])["roi"]]
    res["halves_se_cl_date"] = [r4(cluster_se(r[fh], g["date"].to_numpy()[fh])), r4(cluster_se(r[~fh], g["date"].to_numpy()[~fh]))]
    res["strike"] = r4(g["placed"].mean())
    res["median_price"] = r4(g[price_col].median())
    return res


def comm_table(g: pd.DataFrame, price_col: str) -> dict:
    return {f"c{int(round(c * 100))}": r4(w2p.returns(g[price_col], g["placed"], g["dh"], c).mean()) for c in COMMS5}


def positive(roi, se, halves, bets) -> bool:
    return bool(se and roi is not None and roi > 2 * se and halves and all(h is not None and h > 0 for h in halves) and bets >= 200)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/raw/betfair/split"))
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    # ---- identical pipeline to the original main() up to p_cal -----------------------------
    j = w2p.load(a.dir, a.frac, a.seed)
    P, jw, k, i, ok, nw, race = w2p.placed_set_stats(j)
    months = race["month"].to_numpy()
    codes = race["harness"].to_numpy()
    months_sorted = sorted(set(months))
    ll = w2p.grid_loglik(P, jw, k, i, ok, nw)
    lam = w2p.fit_lambdas_walk_forward(ll, ok, months, codes, months_sorted)
    del ll
    j["p_fair"] = w2p.derived_place_probs(j, P, nw, lam, months, codes)
    j = j[j["p_fair"].notna()].copy()
    j["race"] = pd.factorize(j["win_mid"])[0]
    j["dh"] = j["n_winners"] / j["n_placed"]
    j["p_cal"], cal_params = w2p.calibrate_walk_forward(j, "p_fair", sorted(j["month"].unique()))
    days = sorted(j["date"].unique())
    mid = days[len(days) // 2]
    grp = j.groupby("race")
    j["all_priced"] = grp["ltp_pl"].transform(lambda s: s.notna().all() and (s > 1).all())
    j["year"] = j["date"].dt.year
    j["has_bsp"] = j["bsp_pl"].notna() & (j["bsp_pl"] > 1)
    labels = w2p.band_labels(w2p.PLACE_BANDS)
    j["band_pl"] = pd.cut(j["ltp_pl"], w2p.PLACE_BANDS, right=False, labels=labels).astype(str)
    # share of the race's runners (ALL runners, priced or not) that ever traded in the place market
    j["race_priced_share"] = grp["ltp_pl"].transform(lambda s: (s.notna() & (s > 1)).mean())

    bettable = j[j["ltp_pl"].notna() & (j["ltp_pl"] > 1)].copy()
    edge = (bettable["ltp_pl"] * bettable["p_cal"] - 1)
    bettable["edge"] = edge
    sel = bettable[edge > K].copy()
    hb = sel[sel["has_bsp"]].copy()
    r5 = w2p.returns(sel["ltp_pl"], sel["placed"], sel["dh"], 0.05)
    rb5 = w2p.returns(hb["bsp_pl"], hb["placed"], hb["dh"], 0.05)

    out({"section": "0_data", "runtime_s": round(time.time() - t0, 1), "joined_races": int(j["race"].nunique()),
         "runners": int(len(j)), "mid": str(pd.Timestamp(mid).date()), "from": str(pd.Timestamp(j["date"].min()).date()),
         "to": str(pd.Timestamp(j["date"].max()).date()), "months": int(j["month"].nunique()),
         "last_lambdas": {("harness" if c else "gallops"): v[:2] for (m, c), v in lam.items() if m == max(months_sorted)},
         "cal_params_last": cal_params[max(cal_params)] if cal_params else None})

    # ---- 1. reproduction + clustered SEs ------------------------------------------------------
    rep = cell(sel, mid)
    rep_b = cell(hb, mid, price_col="bsp_pl")
    out({"section": "1_reproduction_and_clustered_se_k0.1_p_cal",
         "at_ltp_pre": {**rep, "bets_per_race": r4(len(sel) / sel["race"].nunique()), "bets_per_day": r4(len(sel) / sel["date"].nunique()),
                        "commission_table": comm_table(sel, "ltp_pl"), "iid_bootstrap_se": iid_boot_se(r5),
                        "month_block_bootstrap": block_boot_se(r5, sel["month"].to_numpy())},
         "same_decision_settled_at_place_bsp": {**rep_b, "bsp_cover": r4(sel["has_bsp"].mean()),
                                               "commission_table": comm_table(hb, "bsp_pl"),
                                               "month_block_bootstrap": block_boot_se(rb5, hb["month"].to_numpy())},
         "ltp_on_bsp_subset": {**cell(hb, mid), "commission_table": comm_table(hb, "ltp_pl")},
         "note": "SE per bet vs clustered by race / meeting / day: bets within a race are weakly negatively dependent "
                 "(at most n_winners of them can win), so race clustering barely moves the SE; day clustering captures "
                 "common feed/staleness shocks."})

    # ---- 2. period splits not chosen after the fact ------------------------------------------
    rows = []
    for y, g in sel.groupby("year"):
        gb = g[g["has_bsp"]]
        rows.append({"year": int(y), "bets": int(len(g)), "roi_c5_ltp": r4(w2p.returns(g["ltp_pl"], g["placed"], g["dh"], 0.05).mean()),
                     "se_cl_date": r4(cluster_se(w2p.returns(g["ltp_pl"], g["placed"], g["dh"], 0.05), g["date"].to_numpy())),
                     "bsp_bets": int(len(gb)), "roi_c5_bsp": r4(w2p.returns(gb["bsp_pl"], gb["placed"], gb["dh"], 0.05).mean()) if len(gb) > 1 else None,
                     "bsp_se_cl_date": r4(cluster_se(w2p.returns(gb["bsp_pl"], gb["placed"], gb["dh"], 0.05), gb["date"].to_numpy())) if len(gb) > 1 else None,
                     "median_place_bsp_over_ltp": r4((gb["bsp_pl"] / gb["ltp_pl"]).median()) if len(gb) else None})
    # thirds by race count
    order = sel.sort_values(["date", "race"])
    cut = np.array_split(np.arange(len(order)), 3)
    thirds = []
    for t, ix in enumerate(cut):
        g = order.iloc[ix]
        gb = g[g["has_bsp"]]
        thirds.append({"third": t + 1, "from": str(pd.Timestamp(g["date"].min()).date()), "to": str(pd.Timestamp(g["date"].max()).date()),
                       "bets": int(len(g)), "roi_c5_ltp": r4(w2p.returns(g["ltp_pl"], g["placed"], g["dh"], 0.05).mean()),
                       "roi_c5_bsp": r4(w2p.returns(gb["bsp_pl"], gb["placed"], gb["dh"], 0.05).mean()) if len(gb) > 1 else None,
                       "bsp_bets": int(len(gb))})
    # monthly sign counts
    m_ltp = pd.Series(r5).groupby(sel["month"].to_numpy()).mean()
    m_bsp = pd.Series(rb5).groupby(hb["month"].to_numpy()).mean()
    m_cnt_b = pd.Series(rb5).groupby(hb["month"].to_numpy()).size()
    # halves by bet count instead of by calendar day
    half_n = len(order) // 2
    g1, g2 = order.iloc[:half_n], order.iloc[half_n:]
    out({"section": "2_period_splits", "by_year": rows, "thirds_by_bet_count": thirds,
         "months_positive_at_ltp": f"{int((m_ltp > 0).sum())}/{len(m_ltp)}",
         "months_positive_at_bsp": f"{int((m_bsp > 0).sum())}/{len(m_bsp)}",
         "months_positive_at_bsp_with_200plus_bets": f"{int(((m_bsp > 0) & (m_cnt_b >= 200)).sum())}/{int((m_cnt_b >= 200).sum())}",
         "halves_by_bet_count_roi_c5_ltp": [r4(w2p.returns(g1["ltp_pl"], g1["placed"], g1["dh"], 0.05).mean()),
                                            r4(w2p.returns(g2["ltp_pl"], g2["placed"], g2["dh"], 0.05).mean())],
         "halves_by_bet_count_roi_c5_bsp": [r4(w2p.returns(g1.loc[g1["has_bsp"], "bsp_pl"], g1.loc[g1["has_bsp"], "placed"], g1.loc[g1["has_bsp"], "dh"], 0.05).mean()),
                                            r4(w2p.returns(g2.loc[g2["has_bsp"], "bsp_pl"], g2.loc[g2["has_bsp"], "placed"], g2.loc[g2["has_bsp"], "dh"], 0.05).mean())],
         "split_day_used_by_original": str(pd.Timestamp(mid).date())})

    # ---- 3. execution: limit-order fill model, market order, SP -------------------------------
    rows = []
    hb["ratio"] = hb["bsp_pl"] / hb["ltp_pl"]
    for thr in FILL_THRESHOLDS:
        filled = hb[hb["ratio"] >= thr]
        unf = hb[hb["ratio"] < thr]
        c = cell(filled, mid) if len(filled) > 1 else {"bets": int(len(filled))}
        rows.append({"fill_rule": f"filled at ltp_pre iff place_bsp >= {thr:g} * ltp_pre", "fill_rate": r4(len(filled) / len(hb)),
                     "bets": c.get("bets"), "roi_c5_at_ltp": c.get("roi"), "se_cl_date": c.get("se_cl_date"), "halves": c.get("halves_roi"),
                     "commission_table": comm_table(filled, "ltp_pl") if len(filled) > 1 else None,
                     "unfilled_bets_roi_c5_at_ltp_(paper_only)": r4(w2p.returns(unf["ltp_pl"], unf["placed"], unf["dh"], 0.05).mean()) if len(unf) > 1 else None,
                     "filled_strike": c.get("strike"), "filled_mean_p_cal": r4(filled["p_cal"].mean()) if len(filled) else None})
    worse = np.minimum(hb["ltp_pl"], hb["bsp_pl"])
    rw = w2p.returns(worse, hb["placed"], hb["dh"], 0.05)
    # "limit at ltp_pre, filled iff the market later touches it (BSP >= ltp), else take SP" is by
    # construction the same price as min(ltp, BSP), so one number covers both.
    out({"section": "3_execution_models_bsp_subset", "bets_with_bsp": int(len(hb)),
         "share_ltp_pre_above_bsp_selected": r4((hb["ratio"] < 1).mean()),
         "share_ltp_pre_above_bsp_all_priced_runners": r4((bettable.loc[bettable["has_bsp"], "bsp_pl"] < bettable.loc[bettable["has_bsp"], "ltp_pl"]).mean()),
         "median_place_bsp_over_ltp_selected": r4(hb["ratio"].median()),
         "quantiles_place_bsp_over_ltp_selected": {str(q): r4(hb["ratio"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
         "median_win_bsp_over_ltp_selected": r4((hb["bsp_win"] / hb["ltp_win"]).median()),
         "one_tick_worse_is_only_pct": r4(1 - (w2p.one_tick_worse(hb["ltp_pl"]) / hb["ltp_pl"]).median()),
         "limit_order_fill_models": rows,
         "worse_of_ltp_and_bsp_(= limit at ltp if touched else SP)": {**basic(rw), "se_cl_date": r4(cluster_se(rw, hb["date"].to_numpy())),
                                                                       "commission_table": comm_table(hb.assign(_w=worse), "_w")},
         "sp_bet": {**basic(rb5), "se_cl_date": r4(cluster_se(rb5, hb["date"].to_numpy())), "commission_table": comm_table(hb, "bsp_pl")}})

    # ---- 4. liquidity: markets with no place BSP (no SP could be formed) ----------------------
    nob = sel[~sel["has_bsp"]]
    rows = {}
    for name, g in {"no_place_bsp": nob, "has_place_bsp": hb, "no_place_bsp_gallops": nob[~nob["harness"]],
                    "no_place_bsp_harness": nob[nob["harness"]], "has_place_bsp_harness": hb[hb["harness"]]}.items():
        c = cell(g, mid) if len(g) > 1 else {"bets": int(len(g))}
        rows[name] = {"bets": c.get("bets"), "roi_c5_at_ltp": c.get("roi"), "se_cl_date": c.get("se_cl_date"), "halves": c.get("halves_roi"),
                      "share_of_selected_bets": r4(len(g) / len(sel)),
                      "mean_share_of_race_runners_ever_traded_in_place": r4(g["race_priced_share"].mean()) if len(g) else None}
    # staleness gradient: edge buckets
    hb_all = bettable[bettable["has_bsp"]]
    eb = pd.cut(hb_all["edge"], [-10, -0.1, 0.0, 0.05, 0.1, 0.2, 0.4, 10])
    grad = []
    for b, g in hb_all.groupby(eb, observed=True):
        grad.append({"edge_bucket_ltp*p_cal-1": str(b), "bets": int(len(g)),
                     "roi_c5_at_ltp": r4(w2p.returns(g["ltp_pl"], g["placed"], g["dh"], 0.05).mean()),
                     "roi_c5_at_bsp": r4(w2p.returns(g["bsp_pl"], g["placed"], g["dh"], 0.05).mean()),
                     "median_place_bsp_over_ltp": r4((g["bsp_pl"] / g["ltp_pl"]).median()),
                     "median_win_bsp_over_ltp": r4((g["bsp_win"] / g["ltp_win"]).median()),
                     "mean_p_cal": r4(g["p_cal"].mean()), "strike": r4(g["placed"].mean()), "mean_1_over_bsp": r4((1 / g["bsp_pl"]).mean())})
    thin = sel["race_priced_share"] < 0.5
    rth = w2p.returns(sel.loc[thin, "ltp_pl"], sel.loc[thin, "placed"], sel.loc[thin, "dh"], 0.05)
    out({"section": "4_liquidity_and_staleness_gradient", "by_bsp_availability": rows, "by_edge_bucket_bsp_subset": grad,
         "share_selected_in_races_with_lt_half_runners_traded_in_place": r4(thin.mean()),
         "roi_c5_at_ltp_in_those_thin_races": basic(rth) if thin.sum() > 1 else None,
         "share_blind_priced_runners_in_races_with_lt_half_traded": r4((bettable["race_priced_share"] < 0.5).mean())})

    # ---- 5. the executable family: 144 original cells at BSP settlement ----------------------
    segments = {"all": np.ones(len(bettable), dtype=bool),
                "places_2": (bettable["n_winners"] == 2).to_numpy(), "places_3": (bettable["n_winners"] == 3).to_numpy(),
                "gallops": (~bettable["harness"]).to_numpy(), "harness": bettable["harness"].to_numpy(),
                "fully_priced_only": bettable["all_priced"].to_numpy(),
                "partially_priced_only": (~bettable["all_priced"]).to_numpy(),
                "place_t30_present_liquid": bettable["t30_pl"].notna().to_numpy(),
                "price_1.5_to_15_post_hoc": ((bettable["ltp_pl"] >= 1.5) & (bettable["ltp_pl"] < 15)).to_numpy()}
    for b in labels:
        segments[f"band_{b}"] = (bettable["band_pl"] == b).to_numpy()
    cells = []
    for pcol in ("p_fair", "p_cal"):
        e = (bettable["ltp_pl"] * bettable[pcol] - 1).to_numpy()
        for kk in w2p.KS:
            for name, mask in segments.items():
                g = bettable[(e > kk) & mask]
                gb = g[g["has_bsp"]]
                c = {"rule": f"ltp_pl*{pcol}>1+{kk:g}", "segment": name, "bets": int(len(g)), "bsp_bets": int(len(gb))}
                r = w2p.returns(g["ltp_pl"], g["placed"], g["dh"], 0.05)
                bl = basic(r)
                c["ltp_roi_c5"] = bl["roi"]
                c["ltp_se_cl_date"] = r4(cluster_se(r, g["date"].to_numpy()))
                fh = (g["date"] < mid).to_numpy()
                c["ltp_halves"] = [basic(r[fh])["roi"], basic(r[~fh])["roi"]]
                c["ltp_positive_per_bet_se"] = positive(bl["roi"], bl["se"], c["ltp_halves"], len(g))
                c["ltp_positive_day_cluster_se"] = positive(bl["roi"], c["ltp_se_cl_date"], c["ltp_halves"], len(g))
                if len(gb) > 1:
                    rb = w2p.returns(gb["bsp_pl"], gb["placed"], gb["dh"], 0.05)
                    bb = basic(rb)
                    fhb = (gb["date"] < mid).to_numpy()
                    c.update({"bsp_roi_c5": bb["roi"], "bsp_se": bb["se"], "bsp_se_cl_date": r4(cluster_se(rb, gb["date"].to_numpy())),
                              "bsp_halves": [basic(rb[fhb])["roi"], basic(rb[~fhb])["roi"]], "bsp_z": bb["z"]})
                    c["bsp_positive"] = positive(bb["roi"], c["bsp_se_cl_date"], c["bsp_halves"], len(gb))
                    rwc = w2p.returns(np.minimum(gb["ltp_pl"], gb["bsp_pl"]), gb["placed"], gb["dh"], 0.05)
                    c["worse_of_roi_c5"] = r4(rwc.mean())
                else:
                    c.update({"bsp_roi_c5": None, "bsp_se": None, "bsp_z": None, "bsp_positive": False, "worse_of_roi_c5": None})
                cells.append(c)
    pv_b = [w2p.z_to_p(c["bsp_z"]) if c["bsp_z"] is not None else 1.0 for c in cells]
    bh_b = w2p.bh(pv_b)
    pv_l = [w2p.z_to_p(c["ltp_roi_c5"] / c["ltp_se_cl_date"]) if c["ltp_se_cl_date"] else 1.0 for c in cells]
    bh_l = w2p.bh(pv_l)
    for c, fb_, fl_ in zip(cells, bh_b, bh_l):
        c["bsp_bh_pass_q05"] = fb_
        c["ltp_bh_pass_q05_day_cluster"] = fl_
    out({"section": "5_executable_family_144_cells", "cells_tested": len(cells),
         "ltp_positive_per_bet_se": sum(c["ltp_positive_per_bet_se"] for c in cells),
         "ltp_positive_day_cluster_se": sum(c["ltp_positive_day_cluster_se"] for c in cells),
         "ltp_bh_pass_day_cluster": sum(c["ltp_bh_pass_q05_day_cluster"] for c in cells),
         "bsp_positive_cells": [(c["rule"], c["segment"], c["bsp_bets"], c["bsp_roi_c5"], c["bsp_halves"]) for c in cells if c["bsp_positive"]],
         "bsp_bh_pass_cells": [(c["rule"], c["segment"], c["bsp_bets"], c["bsp_roi_c5"]) for c in cells if c["bsp_bh_pass_q05"]],
         "bsp_cells_roi_gt_0": sum(1 for c in cells if c["bsp_roi_c5"] is not None and c["bsp_roi_c5"] > 0),
         "bsp_cells_roi_gt_0_with_1000plus_bsp_bets": sum(1 for c in cells if c["bsp_roi_c5"] is not None and c["bsp_roi_c5"] > 0 and c["bsp_bets"] >= 1000),
         "worse_of_cells_roi_gt_0": sum(1 for c in cells if c["worse_of_roi_c5"] is not None and c["worse_of_roi_c5"] > 0),
         "bsp_roi_c5_by_k_all_segment": {c["rule"]: c["bsp_roi_c5"] for c in cells if c["segment"] == "all"},
         "ltp_roi_c5_by_k_all_segment": {c["rule"]: c["ltp_roi_c5"] for c in cells if c["segment"] == "all"},
         "cells": [{kk: c.get(kk) for kk in ("rule", "segment", "bets", "ltp_roi_c5", "ltp_se_cl_date", "ltp_halves", "bsp_bets", "bsp_roi_c5",
                                            "bsp_se_cl_date", "bsp_halves", "worse_of_roi_c5", "bsp_positive", "bsp_bh_pass_q05")} for c in cells]})

    # ---- 6. tuning: develop on one half at ltp_pre, test the other half at BSP ---------------
    checks = []
    for dev_first in (True, False):
        best = None
        for c in cells:
            # pick the cell with the highest dev-half ltp z among cells with >= 1000 dev bets
            pcol = "p_cal" if "p_cal" in c["rule"] else "p_fair"
            kk = float(c["rule"].split(">1+")[1])
            e = (bettable["ltp_pl"] * bettable[pcol] - 1).to_numpy()
            g = bettable[(e > kk) & segments[c["segment"]]]
            d = g[g["date"] < mid] if dev_first else g[g["date"] >= mid]
            if len(d) < 1000:
                continue
            rd = w2p.returns(d["ltp_pl"], d["placed"], d["dh"], 0.05)
            bd = basic(rd)
            if bd["z"] is not None and (best is None or bd["z"] > best[0]):
                best = (bd["z"], c["rule"], c["segment"], bd["roi"], bd["bets"], g)
        if best is None:
            continue
        z, rule, segname, roi_d, n_d, g = best
        h = g[g["date"] >= mid] if dev_first else g[g["date"] < mid]
        hbh = h[h["has_bsp"]]
        rh = w2p.returns(h["ltp_pl"], h["placed"], h["dh"], 0.05)
        rhb = w2p.returns(hbh["bsp_pl"], hbh["placed"], hbh["dh"], 0.05)
        checks.append({"dev_half": "first" if dev_first else "second", "best_cell_by_dev_z_at_ltp": (rule, segname), "dev_bets": n_d,
                       "dev_roi_c5_ltp": roi_d, "dev_z": r4(z), "holdout_bets": int(len(h)), "holdout_roi_c5_ltp": r4(rh.mean()),
                       "holdout_se_cl_date_ltp": r4(cluster_se(rh, h["date"].to_numpy())),
                       "holdout_bsp_bets": int(len(hbh)), "holdout_roi_c5_bsp": r4(rhb.mean()) if len(hbh) > 1 else None,
                       "holdout_bsp_se_cl_date": r4(cluster_se(rhb, hbh["date"].to_numpy())) if len(hbh) > 1 else None})
    out({"section": "6_tuning_dev_holdout", "checks": checks,
         "k_grid_and_p_choice_note": "the claim's k=0.1 and p_cal were chosen among KS x {p_fair,p_cal}; see section 5 "
                                     "bsp_roi_c5_by_k_all_segment: every choice is negative at BSP and gets worse as k rises"})

    # ---- 7. heavy tails -------------------------------------------------------------------------
    r0 = r5
    cap = sel["ltp_pl"].clip(upper=30)
    rcap = w2p.returns(cap, sel["placed"], sel["dh"], 0.05)
    lo = sel["ltp_pl"] < 15
    top = np.sort(r0)[::-1]
    n1 = max(1, int(len(r0) * 0.01))
    out({"section": "7_heavy_tails_at_ltp", "roi_c5": r4(r0.mean()), "roi_c5_price_capped_30": r4(rcap.mean()),
         "roi_c5_excluding_15plus": {**basic(r0[lo.to_numpy()])},
         "share_of_total_profit_from_top_1pct_returns": r4(top[:n1].sum() / max(r0.sum(), 1e-9)) if r0.sum() > 0 else None,
         "max_single_return": r4(top[0]), "kurtosis_excess": r4(pd.Series(r0).kurt()),
         "per_bet_se": r4(r0.std(ddof=1) / math.sqrt(len(r0))), "iid_bootstrap_se": iid_boot_se(r0),
         "roi_c5_at_bsp_excluding_15plus_ltp": basic(w2p.returns(hb.loc[hb["ltp_pl"] < 15, "bsp_pl"], hb.loc[hb["ltp_pl"] < 15, "placed"], hb.loc[hb["ltp_pl"] < 15, "dh"], 0.05))})

    # ---- 8. corrected estimate ------------------------------------------------------------------
    sp = basic(rb5)
    touched = hb[hb["ratio"] >= 1.0]
    rt = w2p.returns(touched["ltp_pl"], touched["placed"], touched["dh"], 0.05)
    out({"section": "8_corrected_estimate",
         "paper_at_ltp_pre": {"bets": rep["bets"], "roi_c5": rep["roi"], "se_per_bet": rep["se_per_bet"], "se_cl_date": rep["se_cl_date"],
                              "halves": rep["halves_roi"], "halves_se_cl_date": rep["halves_se_cl_date"]},
         "executable_sp_bet_decide_ltp_settle_bsp": {"bets": sp["bets"], "roi_c5": sp["roi"], "se_per_bet": sp["se"],
                                                     "se_cl_date": r4(cluster_se(rb5, hb["date"].to_numpy())),
                                                     "roi_c0": r4(w2p.returns(hb["bsp_pl"], hb["placed"], hb["dh"], 0.0).mean()),
                                                     "roi_c6": r4(w2p.returns(hb["bsp_pl"], hb["placed"], hb["dh"], 0.06).mean()),
                                                     "roi_c8": r4(w2p.returns(hb["bsp_pl"], hb["placed"], hb["dh"], 0.08).mean()),
                                                     "roi_c10": r4(w2p.returns(hb["bsp_pl"], hb["placed"], hb["dh"], 0.10).mean()),
                                                     "halves": rep_b["halves_roi"], "halves_se_cl_date": rep_b["halves_se_cl_date"]},
         "limit_order_at_ltp_filled_iff_touched_(bsp>=ltp)": {"fill_rate": r4(len(touched) / len(hb)), **basic(rt),
                                                              "se_cl_date": r4(cluster_se(rt, touched["date"].to_numpy())) if len(touched) > 1 else None},
         "worse_of_ltp_and_bsp": {**basic(rw), "se_cl_date": r4(cluster_se(rw, hb["date"].to_numpy()))},
         "runtime_s": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

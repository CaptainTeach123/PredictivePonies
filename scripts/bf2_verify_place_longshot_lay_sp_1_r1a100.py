"""Adversarial verification #1 of "place_longshot_lay_sp" R1a@100 -- STATISTICS lens (round 2).

    python scripts/bf2_verify_place_longshot_lay_sp_1_r1a100.py [--dir data/raw/betfair/split] [--frac 1.0]

(Written to a new name because scripts/bf2_verify_place_longshot_lay_sp_1.py already exists: it is the R1b@20
verifier of another agent and must not be edited.)

Claim: lay every PLACE runner whose WIN ltp_pre >= 100 at the place BSP, 1 unit of LIABILITY, in AU/GB/IE 2/3-place
markets paired with their WIN market (identical runner sets, settled, G6-clean) whose WIN ltp_pre book is in
[0.97, 1.03] with every runner priced. Claimed ROI8 +0.2646% (SE 0.0544, z 4.86), halves +0.230/+0.299, last 12m
+0.360. NOT the pre-registered threshold (50 was).

The universe is built with the ORIGINAL loader (imported read-only from scripts/bf2_place_longshot_lay_sp.py).
  A  reproduction (orig.eval_cell, liability and stake), by segment
  B  SE by more cluster definitions (market, race day, meeting, venue, HORSE, week, month, two-way horse x day,
     binomial null at q = 1/BSP), block bootstraps; G4 with the most conservative SE; calendar splits
  C  multiple testing: Bonferroni / BH(rank 1) for several programme sizes; the 3-threshold choice
  D  threshold choice: fine grid of WIN ltp_pre thresholds and bands; walk-forward threshold selection
  E  placebos / controls: (1) same rule in ZA, FR, NZ; (2) WIN-market analogue at WIN BSP; (3) shuffled dates;
     (4) price-matched control (WIN ltp_pre < 100, same place-BSP buckets); (5) R1a@100 bets NOT in R1b@20;
     (6) missing-BSP sensitivity; (7) segment heterogeneity
  F  time stability: years, quarters, rolling 12-month windows
Prints summary numbers only (never horse names or rows).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erfcinv
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("orig_pll", ROOT / "scripts/bf2_place_longshot_lay_sp.py")
orig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(orig)

LAST12_FROM = np.datetime64("2019-11-08")
LAST12_TO = np.datetime64("2020-11-08")
C8 = 0.08
THR = 100


def out(o):
    print(json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x)), flush=True)


def r6(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def lay_liab(price, f, c=C8):
    gross = (1 - f) - f * (price - 1)
    net = np.where(gross > 0, gross * (1 - c), gross)
    return net / (price - 1)


def lay_stake(price, f, c=C8):
    gross = (1 - f) - f * (price - 1)
    return np.where(gross > 0, gross * (1 - c), gross)


def cse(r, cl):
    d = pd.Series(r - r.mean()).groupby(np.asarray(cl)).sum().to_numpy()
    G = len(d)
    return float(math.sqrt((d ** 2).sum() * G / (G - 1)) / len(r)) if G > 1 else float("nan")


def cse_codes(r, codes, G):
    d = np.bincount(codes, weights=r - r.mean(), minlength=G)
    d = d[np.bincount(codes, minlength=G) > 0]
    G = len(d)
    return float(math.sqrt((d ** 2).sum() * G / (G - 1)) / len(r)) if G > 1 else float("nan")


def cse_twoway(r, a, b):
    ab = pd.Series(a).astype(str).to_numpy() + "|" + pd.Series(b).astype(str).to_numpy()
    va, vb, vab = cse(r, a) ** 2, cse(r, b) ** 2, cse(r, ab) ** 2
    v = va + vb - vab
    return float(math.sqrt(max(v, va, vb)))


def boot_se(r, cl, reps=2000, seed=0):
    codes, uniq = pd.factorize(np.asarray(cl))
    s = np.bincount(codes, weights=r)
    c = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    G = len(uniq)
    bs = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, G, G)
        bs[i] = s[idx].sum() / c[idx].sum()
    return float(bs.std(ddof=1)), float((bs <= 0).mean())


def se_max(r, g):
    n = len(r)
    if n < 3:
        return float("nan")
    return max(float(r.std(ddof=1) / math.sqrt(n)), cse(r, g["race"].to_numpy()), cse(r, g["date"].to_numpy()))


def se_null_of(price, staking="liability"):
    q = 1 / price
    if staking == "liability":
        vn = q * (1 - q) * (1 + (1 - C8) / (price - 1)) ** 2
    else:
        vn = q * (1 - q) * (price - 1 + (1 - C8)) ** 2
    return float(math.sqrt(vn.sum()) / len(price))


def se_cons(rr, gg, price_col="_px", staking="liability"):
    """max of iid, market, day, meeting, venue, horse, week and the binomial-null SE."""
    if len(rr) < 3:
        return float("nan")
    w_ = gg["week"].to_numpy()
    return max(float(rr.std(ddof=1) / math.sqrt(len(rr))), cse(rr, gg["race"].to_numpy()),
               cse(rr, gg["date"].to_numpy()), cse(rr, gg["horse"].to_numpy()), cse(rr, gg["meeting"].to_numpy()),
               cse(rr, gg["venue_key"].to_numpy()), cse(rr, w_), se_null_of(gg[price_col].to_numpy(), staking))


def summarize(g, price, f, label, extra=None, se_fn=None, verbose=True, staking="liability", c=C8):
    ok = np.isfinite(price)
    g, price, f = g[ok], price[ok], f[ok]
    if len(g) < 10:
        if verbose:
            out({"section": label, "bets": int(len(g))})
        return {"bets": int(len(g))}
    g = g.assign(_px=price)
    se_fn = se_fn or se_max
    fn = lay_liab if staking == "liability" else lay_stake
    r = fn(price, f, c)
    day = g["date"].to_numpy()
    rd = pd.Series(day).groupby(g["race"].to_numpy()).first().to_numpy()
    mid = np.sort(rd)[len(rd) // 2]
    fh = day < mid
    l12 = (day >= LAST12_FROM) & (day <= LAST12_TO)
    se = se_fn(r, g)
    h1, h2 = r[fh], r[~fh]
    s1, s2 = se_fn(h1, g[fh]), se_fn(h2, g[~fh])
    lt = r[l12]
    res = {"section": label, "bets": int(len(r)), "roi8": r6(r.mean()), "se": r6(se), "z": r6(r.mean() / se),
           "roi_0_5_10": [r6(fn(price, f, cc).mean()) for cc in (0.0, 0.05, 0.10)],
           "halves": [r6(h1.mean()), r6(h2.mean())], "halves_se": [r6(s1), r6(s2)],
           "halves_z": [r6(h1.mean() / s1), r6(h2.mean() / s2)], "split": str(mid)[:10],
           "last12": r6(lt.mean()) if len(lt) else None, "last12_bets": int(len(lt)),
           "strike_over_implied": r6(f.mean() / np.mean(1 / price)), "median_price": r6(np.median(price)),
           "g4": bool(h1.mean() > 2 * s1 and h2.mean() > 2 * s2 and len(lt) > 0 and lt.mean() > 0
                      and (f[fh] > 0).sum() >= 10 and (f[~fh] > 0).sum() >= 10)}
    if extra:
        res.update(extra)
    if verbose:
        out(res)
    return res


def add_keys(j, d, cc):
    ven = pd.read_parquet(d / f"{cc}_win.parquet", columns=["market_id", "venue"]).drop_duplicates("market_id")
    ven["market_id"] = ven["market_id"].astype(object)
    j = j.merge(ven.rename(columns={"market_id": "win_mid"}), on="win_mid", how="left")
    j["cc"] = cc
    j["date"] = pd.to_datetime(j["date"]).values.astype("datetime64[D]")
    j["venue_key"] = j["cc"] + ":" + j["venue"].astype(str)
    j["meeting"] = j["venue_key"] + ":" + j["date"].astype(str)
    j["horse"] = j["cc"] + ":" + j["selection_id"].astype(str)
    j["week"] = pd.to_datetime(j["date"]).dt.to_period("W").astype(str)
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data/raw/betfair/split"))
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--skip_other", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir)

    frames = []
    for cc in ("au", "gb", "ie"):
        j, st = orig.load_country(cc, d, a.frac, 7)
        frames.append(add_keys(j, d, cc))
    df = pd.concat(frames, ignore_index=True)
    del frames
    df["year"] = df["start"].dt.year
    u = df[df["win_pre_all"] & df["win_pre_book"].between(0.97, 1.03)].copy()
    n_univ = int(u["race"].nunique())
    sel = u[u["pre_win"] >= THR]
    g = sel[sel["bsp_pl"].notna()].copy().reset_index(drop=True)
    g["_px"] = g["bsp_pl"]
    P, F = g["bsp_pl"].to_numpy(), g["f_pl"].to_numpy()
    out({"section": "universe", "races_g6_clean": int(df["race"].nunique()), "R1a_races": n_univ,
         "decisions": int(len(sel)), "bets": int(len(g)),
         "bets_by_seg": g["seg"].value_counts().to_dict()})

    # ---------------------------------------------------------------- A reproduction
    nm = orig.n_months_of(u)
    for staking in ("liability", "stake"):
        row = orig.eval_cell(sel, sel["bsp_pl"].to_numpy(), sel["f_pl"].to_numpy(), staking, n_univ, nm)
        out({"section": f"A_orig_eval_cell_{staking}", **{k: row.get(k) for k in (
            "bets", "roi_8pct", "se_clustered", "se_race", "se_day", "se_iid", "z", "halves_roi_8pct", "halves_se",
            "half_split_date", "last12m_roi_8pct", "last12m_se", "passes_G4", "roi_0_5_10pct",
            "roi_8pct_market_netted", "strike_over_implied", "mde_80pct", "median_price")}})
    for seg in ("au_gallops", "au_harness", "gb", "ie"):
        m = (g["seg"] == seg).to_numpy()
        summarize(g[m], P[m], F[m], f"A_seg_{seg}")
        summarize(g[m], P[m], F[m], f"A_seg_{seg}_conservative_se", se_fn=se_cons)

    # ---------------------------------------------------------------- B SE by many clusterings
    r = lay_liab(P, F)
    q = 1 / P
    ses = {"iid": float(r.std(ddof=1) / math.sqrt(len(r))), "market": cse(r, g["race"]), "day": cse(r, g["date"]),
           "meeting": cse(r, g["meeting"]), "venue": cse(r, g["venue_key"]), "horse": cse(r, g["horse"]),
           "week": cse(r, g["week"]), "month": cse(r, g["month"]),
           "twoway_horse_day": cse_twoway(r, g["horse"].to_numpy(), g["date"].astype(str).to_numpy()),
           "binomial_null_q_1_over_bsp": se_null_of(P)}
    boots = {}
    for lab, cl in (("day", g["date"].to_numpy()), ("week", g["week"].to_numpy()), ("month", g["month"].to_numpy()),
                    ("horse", g["horse"].to_numpy())):
        s_, p_ = boot_se(r, cl, 2000, seed=len(lab))
        boots[lab] = [r6(s_), r6(p_)]
    hc = g.groupby("horse").size()
    out({"section": "B_se_by_cluster", "roi8": r6(r.mean()), "se": {k: r6(v) for k, v in ses.items()},
         "z": {k: r6(r.mean() / v) for k, v in ses.items()}, "boot_se_and_p_le_0": boots,
         "n_horses": int(len(hc)), "bets_per_horse_mean": r6(hc.mean()), "max_bets_one_horse": int(hc.max()),
         "n_meetings": int(g["meeting"].nunique()), "n_days": int(g["date"].nunique()),
         "bets_per_race_with_bet": r6(len(g) / g["race"].nunique())})
    cons = summarize(g, P, F, "B_conservative_se_max_of_all", se_fn=se_cons)
    day = g["date"].to_numpy()
    for lab, cuts in (("period_median", [np.datetime64("2018-12-09")]),
                      ("thirds", [np.datetime64("2018-04-08"), np.datetime64("2019-07-23")])):
        edges = [np.datetime64("2016-11-30")] + cuts + [np.datetime64("2020-11-09")]
        parts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (day >= lo) & (day < hi)
            sc = se_cons(r[m], g[m])
            parts.append([int(m.sum()), r6(r[m].mean()), r6(sc), r6(r[m].mean() / sc)])
        out({"section": f"B_split_{lab}_bets_roi_se_z", "parts": parts})
    # per stake with the conservative SE (G2 asks for both)
    summarize(g, P, F, "B_per_stake_conservative_se", se_fn=lambda rr, gg: se_cons(rr, gg, staking="stake"),
              staking="stake")

    # ---------------------------------------------------------------- C programme-wide multiple testing
    z_cons = cons["z"]
    p_rep = float(norm.sf(4.86))
    p_cons = float(norm.sf(z_cons))
    rows = {}
    for mm in (324, 1000, 4000, 4324, 6000, 10000, 20000):
        rows[mm] = {"bonf_z_5pct_one_sided": r6(math.sqrt(2) * erfcinv(2 * 0.05 / mm)),
                    "bh_rank1_p_q10": r6(0.10 / mm),
                    "survives_bh_rank1_reported_z": bool(p_rep <= 0.10 / mm),
                    "survives_bh_rank1_conservative_z": bool(p_cons <= 0.10 / mm),
                    "survives_bonf5_conservative_z": bool(p_cons <= 0.05 / mm)}
    out({"section": "C_multiple_testing", "z_reported": 4.86, "p_reported": p_rep, "z_conservative": z_cons,
         "p_conservative": p_cons, "max_programme_cells_bh_rank1_q10_reported": int(0.10 / p_rep),
         "max_programme_cells_bh_rank1_q10_conservative": int(0.10 / p_cons), "by_programme_size": rows})

    # ---------------------------------------------------------------- D threshold choice
    grid = {}
    for T in (20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 110, 120, 150, 200, 300, 500):
        gg = u[(u["pre_win"] >= T) & u["bsp_pl"].notna()]
        s = summarize(gg, gg["bsp_pl"].to_numpy(), gg["f_pl"].to_numpy(), f"D_thr_{T}", verbose=False)
        grid[T] = [s.get("bets"), s.get("roi8"), s.get("z"), s.get("halves_z"), s.get("last12"), s.get("g4")]
    out({"section": "D_threshold_grid_bets_roi_z_halvesz_last12_g4", "table": grid})
    bands = {}
    for lo, hi in ((10, 20), (20, 30), (30, 50), (50, 70), (70, 100), (100, 150), (150, 200), (200, 300), (300, 500),
                   (500, 1001)):
        gg = u[(u["pre_win"] >= lo) & (u["pre_win"] < hi) & u["bsp_pl"].notna()]
        s = summarize(gg, gg["bsp_pl"].to_numpy(), gg["f_pl"].to_numpy(), "band", verbose=False)
        bands[f"[{lo},{hi})"] = [s.get("bets"), s.get("roi8"), s.get("z"), s.get("strike_over_implied"),
                                 s.get("median_price")]
    out({"section": "D_win_ltp_pre_bands_bets_roi_z_strike_over_implied_median_place_bsp", "table": bands})
    # walk-forward threshold choice: for each test year pick T on all EARLIER data (max ROI8), apply to the year
    ub = u[u["bsp_pl"].notna()].copy()
    ub["r8"] = lay_liab(ub["bsp_pl"].to_numpy(), ub["f_pl"].to_numpy())
    for lab, cand in (("preregistered_set_30_50_100", (30, 50, 100)),
                      ("fine_grid", (30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300))):
        picks, oos = {}, []
        for y in (2018, 2019, 2020):
            tr = ub[ub["year"] < y]
            best = max(cand, key=lambda T: tr.loc[tr["pre_win"] >= T, "r8"].mean())
            te = ub[(ub["year"] == y) & (ub["pre_win"] >= best)]
            picks[y] = [best, int(len(te)), r6(te["r8"].mean())]
            oos.append(te)
        oo = pd.concat(oos)
        s = summarize(oo, oo["bsp_pl"].to_numpy(), oo["f_pl"].to_numpy(), f"D_walk_forward_{lab}", verbose=False,
                      se_fn=se_cons)
        out({"section": f"D_walk_forward_threshold_{lab}", "picks_year_T_bets_roi": picks,
             "oos_2018_2020": {k: s.get(k) for k in ("bets", "roi8", "se", "z", "halves", "halves_z", "last12")}})
    # choose on the first half, test on the second half
    mid = np.datetime64(cons["split"])
    tr, te = ub[ub["date"] < mid], ub[ub["date"] >= mid]
    tab = {}
    for T in (30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300):
        a1 = tr.loc[tr["pre_win"] >= T]
        a2 = te.loc[te["pre_win"] >= T]
        tab[T] = [r6(a1["r8"].mean()), r6(a1["r8"].mean() / se_cons(a1["r8"].to_numpy(), a1.assign(_px=a1["bsp_pl"]))),
                  r6(a2["r8"].mean()), r6(a2["r8"].mean() / se_cons(a2["r8"].to_numpy(), a2.assign(_px=a2["bsp_pl"])))]
    out({"section": "D_first_half_vs_second_half_by_T_roi_z_roi_z", "table": tab})

    # ---------------------------------------------------------------- E placebos and controls
    # E5 overlap with R1b@20 and the R1a@100 bets that R1b@20 does not take
    in_r1b = (g["pl_pre_all"] & g["pl_pre_book"].between(0.97, 1.03) & (g["pre_pl"] >= 20)).to_numpy()
    out({"section": "E5_overlap", "share_R1a100_bets_also_in_R1b20": r6(in_r1b.mean()),
         "share_R1a100_in_place_G1_universe": r6((g["pl_pre_all"] & g["pl_pre_book"].between(0.97, 1.03)).mean())})
    summarize(g[in_r1b], P[in_r1b], F[in_r1b], "E5_R1a100_and_R1b20", se_fn=se_cons)
    summarize(g[~in_r1b], P[~in_r1b], F[~in_r1b], "E5_R1a100_not_R1b20", se_fn=se_cons)
    pg1 = (g["pl_pre_all"] & g["pl_pre_book"].between(0.97, 1.03)).to_numpy()
    m = pg1 & ~in_r1b
    summarize(g[m], P[m], F[m], "E5_R1a100_place_G1_but_place_ltp_pre_lt20", se_fn=se_cons)
    m = ~pg1
    summarize(g[m], P[m], F[m], "E5_R1a100_place_market_not_G1", se_fn=se_cons)

    # E4 price-matched control: runners with WIN ltp_pre < 100 in the same universe and same place-BSP buckets
    edges = np.array([1.0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1001])
    ctrl = u[(u["pre_win"] < THR) & u["bsp_pl"].notna()]
    bt = np.digitize(P, edges)
    bc = np.digitize(ctrl["bsp_pl"].to_numpy(), edges)
    rc = lay_liab(ctrl["bsp_pl"].to_numpy(), ctrl["f_pl"].to_numpy())
    tab, wsum, rsum, rsum_t = {}, 0.0, 0.0, 0.0
    for b in np.unique(bt):
        nt, mc = int((bt == b).sum()), bc == b
        rt_b = float(r[bt == b].mean())
        if mc.sum() >= 30:
            rc_b = float(rc[mc].mean())
            wsum += nt
            rsum += nt * rc_b
            rsum_t += nt * rt_b
        else:
            rc_b = None
        hi = edges[b] if b < len(edges) else float("inf")
        tab[f"[{edges[b - 1]:g},{hi:g})"] = [nt, r6(rt_b), int(mc.sum()), r6(rc_b) if rc_b is not None else None]
    out({"section": "E4_price_matched_control", "bucket_nR1a_roiR1a_nctrl_roictrl": tab,
         "R1a_roi_on_matched_buckets": r6(rsum_t / wsum) if wsum else None,
         "ctrl_roi_reweighted_to_R1a_mix": r6(rsum / wsum) if wsum else None, "matched_bets": int(wsum)})

    # E6 missing place BSP sensitivity
    nob = sel[sel["bsp_pl"].isna()]
    mkt_has_bsp = sel.groupby("race")["bsp_pl"].transform(lambda s: s.notna().any())
    nob_in_bsp_mkt = sel[sel["bsp_pl"].isna() & mkt_has_bsp]
    out({"section": "E6_missing_place_bsp", "decisions": int(len(sel)), "no_bsp": int(len(nob)),
         "no_bsp_in_market_with_some_bsp": int(len(nob_in_bsp_mkt)),
         "no_bsp_by_seg": nob["seg"].value_counts().to_dict(),
         "placed_rate_with_bsp": r6(g["f_pl"].mean()), "placed_rate_no_bsp": r6(nob["f_pl"].mean()) if len(nob) else None,
         "placed_rate_no_bsp_in_bsp_mkt": r6(nob_in_bsp_mkt["f_pl"].mean()) if len(nob_in_bsp_mkt) else None})
    # settle the runner-level missing BSPs (market has other BSPs) at the place ltp_pre, else at the place BSP
    alt = sel.copy()
    px = alt["bsp_pl"].to_numpy().copy()
    fill = alt["bsp_pl"].isna().to_numpy() & mkt_has_bsp.to_numpy()
    px[fill] = alt["pre_pl"].to_numpy()[fill]
    summarize(alt, px, alt["f_pl"].to_numpy(), "E6_runner_missing_bsp_settled_at_place_ltp_pre", se_fn=se_max)
    # worst case: every runner-level missing BSP is a lay at place ltp_pre or, if none, at 1/(mean implied)
    px2 = px.copy()
    still = fill & ~np.isfinite(px2)
    px2[still] = float(np.median(P))
    summarize(alt, px2, alt["f_pl"].to_numpy(), "E6_runner_missing_bsp_settled_fallback_median", se_fn=se_max)

    # E7 segment heterogeneity: AU gallops vs GB/IE
    seg_r = {}
    for seg in ("au_gallops", "au_harness", "gb", "ie"):
        mk = (g["seg"] == seg).to_numpy()
        if mk.sum() > 10:
            seg_r[seg] = (float(r[mk].mean()), se_cons(r[mk], g[mk].assign(_px=P[mk])), int(mk.sum()))
    if "au_gallops" in seg_r and "gb" in seg_r:
        diffs = {}
        for s2 in ("gb", "ie"):
            a1, b1 = seg_r["au_gallops"], seg_r[s2]
            diffs[f"au_gallops_minus_{s2}"] = [r6(a1[0] - b1[0]), r6((a1[0] - b1[0]) / math.hypot(a1[1], b1[1]))]
        gbie = np.isin(g["seg"].to_numpy(), ["gb", "ie"])
        summarize(g[gbie], P[gbie], F[gbie], "E7_gb_plus_ie", se_fn=se_cons)
        out({"section": "E7_segment_heterogeneity", "seg_roi_se_n": {k: [r6(v[0]), r6(v[1]), v[2]] for k, v in seg_r.items()},
             "diff_and_z": diffs})

    # E2 WIN-market analogue (lay WIN ltp_pre >= T at the WIN BSP, same WIN-book G1 universe)
    for T in (50, 100, 200):
        gg = u[u["pre_win"] >= T]
        summarize(gg, gg["bsp_win"].to_numpy(), gg["f_win"].to_numpy(), f"E2_win_market_lay_win_bsp_at_{T}",
                  se_fn=se_max)
    summarize(g, g["bsp_win"].to_numpy(), g["f_win"].to_numpy(), "E2_R1a100_runners_laid_in_WIN_at_win_bsp",
              se_fn=se_max)
    # same runners laid at the PLACE ltp_pre (not achievable, descriptive)
    summarize(g, g["pre_pl"].to_numpy(), F, "E2_R1a100_runners_laid_at_place_ltp_pre_descriptive", se_fn=se_max)

    # E3 shuffled-date placebo
    rng = np.random.default_rng(2024)
    days = np.unique(day)
    di = np.searchsorted(days, day)
    real_mid = np.searchsorted(days, np.datetime64(cons["split"]))
    l12_days = int(((days >= LAST12_FROM) & (days <= LAST12_TO)).sum())
    passes, zmin, l12s = 0, [], []
    race_arr = g["race"].to_numpy()
    for it in range(200):
        perm = rng.permutation(len(days))
        pd_ = perm[di]
        fh = pd_ < real_mid
        l12 = pd_ >= len(days) - l12_days
        a1, a2 = r[fh], r[~fh]
        s1 = max(float(a1.std(ddof=1) / math.sqrt(len(a1))), cse(a1, race_arr[fh]), cse(a1, pd_[fh]))
        s2 = max(float(a2.std(ddof=1) / math.sqrt(len(a2))), cse(a2, race_arr[~fh]), cse(a2, pd_[~fh]))
        zmin.append(min(a1.mean() / s1, a2.mean() / s2))
        l12s.append(r[l12].mean())
        passes += int(a1.mean() > 2 * s1 and a2.mean() > 2 * s2 and r[l12].mean() > 0)
    out({"section": "E3_shuffled_dates", "reps": 200, "g4_pass_rate": r6(passes / 200),
         "min_half_z_p5_p50": [r6(np.quantile(zmin, 0.05)), r6(np.median(zmin))],
         "real_half_z": cons["halves_z"], "last12_p5_p50": [r6(np.quantile(l12s, 0.05)), r6(np.median(l12s))]})

    # E8 sign-randomisation null of the 3-threshold x 7-segment search (outcomes redrawn at the break-even
    # probability for each bet, so the true ROI8 is 0); distribution of the max z over the cells
    q0_all = (1 - C8) / (ub["bsp_pl"].to_numpy() - C8)  # E[lay_liab] = 0 at this placed probability
    segs = ub["seg"].to_numpy()
    nwv = ub["n_winners"].to_numpy()
    pw = ub["pre_win"].to_numpy()
    raceu = ub["race"].to_numpy()
    dayu = ub["date"].to_numpy()
    masks = []
    for T in (30, 50, 100):
        for sg in ("all", "au_gallops", "au_harness", "gb", "ie", "nw2", "nw3"):
            mk = pw >= T
            if sg in ("nw2", "nw3"):
                mk = mk & (nwv == int(sg[-1]))
            elif sg != "all":
                mk = mk & (segs == sg)
            if mk.sum() > 100:
                masks.append(mk)
    Pu = ub["bsp_pl"].to_numpy()
    rc_codes, _ = pd.factorize(raceu)
    dc_codes, _ = pd.factorize(dayu)
    mcodes = [(mk, pd.factorize(rc_codes[mk])[0], pd.factorize(dc_codes[mk])[0]) for mk in masks]
    rng = np.random.default_rng(99)
    maxz = []
    for it in range(100):
        ysim = (rng.random(len(Pu)) < q0_all).astype(float)
        rs = lay_liab(Pu, ysim)
        zs = []
        for mk, rcm, dcm in mcodes:
            rr = rs[mk]
            se = max(float(rr.std(ddof=1) / math.sqrt(len(rr))), cse_codes(rr, rcm, rcm.max() + 1),
                     cse_codes(rr, dcm, dcm.max() + 1))
            zs.append(rr.mean() / se)
        maxz.append(max(zs))
    maxz = np.array(maxz)
    out({"section": "E8_null_max_z_over_rule1a_cells", "cells": len(masks), "reps": 100,
         "max_z_p50_p95_p99_max": [r6(np.median(maxz)), r6(np.quantile(maxz, 0.95)), r6(np.quantile(maxz, 0.99)),
                                   r6(maxz.max())], "share_ge_4_86": r6((maxz >= 4.86).mean())})

    # ---------------------------------------------------------------- F time stability
    yr = {}
    for y in sorted(g["year"].unique()):
        mk = (g["year"] == y).to_numpy()
        yr[int(y)] = [int(mk.sum()), r6(r[mk].mean()), r6(se_cons(r[mk], g[mk].assign(_px=P[mk])))]
    out({"section": "F_years_bets_roi_conservative_se", "table": yr})
    qtr = pd.to_datetime(g["date"]).dt.to_period("Q").astype(str).to_numpy()
    tq = {}
    for k in sorted(np.unique(qtr)):
        mk = qtr == k
        tq[k] = [int(mk.sum()), r6(r[mk].mean()), r6(r[mk].mean() / se_max(r[mk], g[mk]))]
    out({"section": "F_quarters_bets_roi_z", "table": tq, "n_negative": int(sum(v[1] < 0 for v in tq.values())),
         "n_quarters": len(tq)})
    starts = pd.date_range("2016-12-01", "2019-11-01", freq="MS")
    roll = []
    for s0 in starts:
        lo, hi = np.datetime64(s0.date()), np.datetime64((s0 + pd.DateOffset(months=12)).date())
        mk = (day >= lo) & (day < hi)
        roll.append(r[mk].mean())
    roll = np.array(roll)
    out({"section": "F_rolling_12m", "n_windows": int(len(roll)), "min": r6(roll.min()), "max": r6(roll.max()),
         "n_negative": int((roll < 0).sum())})
    # last 12 months with the conservative SE, and by segment
    l12 = (day >= LAST12_FROM) & (day <= LAST12_TO)
    summarize(g[l12], P[l12], F[l12], "F_last12m_conservative", se_fn=se_cons)
    # LOC cap 100 with conservative SE
    mk = P <= 100
    summarize(g[mk], P[mk], F[mk], "F_loc_cap_100_conservative", se_fn=se_cons)
    del df, u, ub

    # ---------------------------------------------------------------- E1 other countries
    if not a.skip_other:
        for cc in ("za", "fr", "nz"):
            try:
                j, st = orig.load_country(cc, d, a.frac, 7)
            except Exception as e:  # noqa: BLE001
                out({"section": f"E1_{cc}", "error": str(e)[:200]})
                continue
            j = add_keys(j, d, cc)
            uj = j[j["win_pre_all"] & j["win_pre_book"].between(0.97, 1.03)]
            out({"section": f"E1_{cc}_universe", "races": int(j["race"].nunique()), "R1a_races": int(uj["race"].nunique()),
                 "share_runners_pl_bsp": st.get("share_runners_pl_bsp")})
            for T in (50, 100):
                gg = uj[uj["pre_win"] >= T]
                summarize(gg, gg["bsp_pl"].to_numpy(), gg["f_pl"].to_numpy(), f"E1_{cc}_R1a_at_{T}",
                          {"decisions": int(len(gg)), "with_bsp": int(gg["bsp_pl"].notna().sum())}, se_fn=se_cons)
            gg = j[j["pre_win"] >= 100]
            summarize(gg, gg["bsp_pl"].to_numpy(), gg["f_pl"].to_numpy(), f"E1_{cc}_any_book_at_100_descriptive",
                      se_fn=se_cons)
            del j, uj


if __name__ == "__main__":
    main()

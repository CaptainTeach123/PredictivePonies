"""Adversarial verification #1 (STATISTICS lens) of "place_longshot_lay_sp" R1b@20 PER UNIT OF BACKER'S STAKE.

    python scripts/bf2_verify_place_longshot_lay_sp_1_stake.py [--dir data/raw/betfair/split] [--frac 1.0]
        [--cells path/to/orig_cells.json]

(scripts/bf2_verify_place_longshot_lay_sp_1.py already existed -- written for the per-LIABILITY claim -- so this
is a new file; nothing existing is edited.)

Claim: lay every PLACE runner with PLACE ltp_pre >= 20 at the place BSP, 1 unit of BACKER'S STAKE, in AU/GB/IE
2/3-place markets paired with their WIN market (identical runner sets, settled, G6-clean) whose PLACE ltp_pre
book / n_winners is in [0.97, 1.03] with every runner priced. Claimed ROI8 +11.14% of stake (SE 2.62, z 4.25),
halves +7.37% (SE 4.16) / +14.82% (SE 3.26), last 12 months +17.6%; fails G4 (first half 1.77 SE).

Universe built with the ORIGINAL loader (imported read-only).
  A  reproduction (orig.eval_cell, stake and liability)
  B  SE per stake: iid, market, day, meeting, venue, horse, ISO week, month, two-way horse x day, binomial
     null SE at q = 1/BSP; block bootstraps; PARAMETRIC NULL (outcomes simulated at a uniformly scaled
     1/BSP so the per-stake ROI is 0): exact one-sided p of the observed ROI, overall and in each half
  C  heavy tails / influence: P&L share by BSP bucket, ROI without the k worst bets, LOC caps (50/100/200),
     winsorised returns, per-year
  D  achievable sizing: liability = (PLACE ltp_pre - 1) at the decision (a target backer stake of 1 at the
     selection quote), and capped variants; halves / last 12 / G4
  E  halves at the bets' median, at the period median, thirds; market-netted commission
  F  multiple testing (programme sizes; BH using the original script's own cells if --cells is given)
  G  placebos per stake: shuffled race days (G4 pass rate), ZA/FR place markets, WIN-market analogue,
     price-matched control (runners with PLACE ltp_pre < 20 in the same place-BSP buckets)
  H  threshold grid per stake
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


def out(o):
    print(json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x)), flush=True)


def r6(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def lay_stake(price, f, c=C8):
    gross = (1 - f) - f * (price - 1)
    return np.where(gross > 0, gross * (1 - c), gross)


def lay_liab(price, f, c=C8):
    return lay_stake(price, f, c) / (price - 1)


def cse(r, cl):
    d = pd.Series(r - r.mean()).groupby(np.asarray(cl)).sum().to_numpy()
    G = len(d)
    return float(math.sqrt((d ** 2).sum() * G / (G - 1)) / len(r)) if G > 1 else float("nan")


def cse_twoway(r, a, b):
    ab = pd.Series(a).astype(str).to_numpy() + "|" + pd.Series(b).astype(str).to_numpy()
    va, vb, vab = cse(r, a) ** 2, cse(r, b) ** 2, cse(r, ab) ** 2
    return float(math.sqrt(max(va + vb - vab, va, vb)))


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
    return float(bs.std(ddof=1)), float((bs <= 0).mean()), [r6(np.quantile(bs, 0.025)), r6(np.quantile(bs, 0.975))]


def se_max(r, g):
    if len(r) < 3:
        return float("nan")
    return max(float(r.std(ddof=1) / math.sqrt(len(r))), cse(r, g["race"].to_numpy()), cse(r, g["date"].to_numpy()))


def summarize(g, ret, f, label, extra=None, se_fn=None, verbose=True):
    """ret: per-bet return (already in the chosen staking unit); f: settlement fraction (for loss counts)."""
    ok = np.isfinite(ret)
    g, ret, f = g[ok], ret[ok], f[ok]
    if len(g) < 10:
        if verbose:
            out({"section": label, "bets": int(len(g))})
        return {"bets": int(len(g))}
    se_fn = se_fn or se_max
    day = g["date"].to_numpy()
    rd = pd.Series(day).groupby(g["race"].to_numpy()).first().to_numpy()
    mid = np.sort(rd)[len(rd) // 2]
    fh = day < mid
    l12 = (day >= LAST12_FROM) & (day <= LAST12_TO)
    se = se_fn(ret, g)
    h1, h2 = ret[fh], ret[~fh]
    s1, s2 = se_fn(h1, g[fh]), se_fn(h2, g[~fh])
    lt = ret[l12]
    slt = se_fn(lt, g[l12]) if l12.sum() >= 3 else float("nan")
    res = {"section": label, "bets": int(len(ret)), "roi8": r6(ret.mean()), "se": r6(se), "z": r6(ret.mean() / se),
           "halves": [r6(h1.mean()), r6(h2.mean())], "halves_se": [r6(s1), r6(s2)],
           "halves_z": [r6(h1.mean() / s1), r6(h2.mean() / s2)], "split": str(mid)[:10],
           "last12": r6(lt.mean()) if len(lt) else None, "last12_se": r6(slt), "last12_bets": int(len(lt)),
           "g4": bool(h1.mean() > 2 * s1 and h2.mean() > 2 * s2 and len(lt) > 0 and lt.mean() > 0
                      and (f[fh] > 0).sum() >= 10 and (f[~fh] > 0).sum() >= 10)}
    if extra:
        res.update(extra)
    if verbose:
        out(res)
    return res


def null_sim(P, F_obs_mean_ret, halves_mask, reps, seed, n_winners=None):
    """Parametric null: placed_i ~ Bernoulli(k / P_i) with k such that the expected per-stake ROI at 8% is 0.
    Returns the one-sided p of the observed ROI overall and in each half (dead heats ignored)."""
    n = len(P)
    k = n * (1 - C8) / np.sum((P - C8) / P)
    q = np.clip(k / P, 0, 1)
    rng = np.random.default_rng(seed)
    win_ret, lose_ret = 1 - C8, -(P - 1)
    sims = np.empty((reps, 3))
    for i in range(reps):
        y = rng.random(n) < q
        rr = np.where(y, lose_ret, win_ret)
        sims[i, 0] = rr.mean()
        sims[i, 1] = rr[halves_mask].mean()
        sims[i, 2] = rr[~halves_mask].mean()
    return k, sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data/raw/betfair/split"))
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--reps", type=int, default=4000)
    a = ap.parse_args()
    d = Path(a.dir)

    frames = []
    for cc in ("au", "gb", "ie"):
        j, st = orig.load_country(cc, d, a.frac, 7)
        ven = pd.read_parquet(d / f"{cc}_win.parquet", columns=["market_id", "venue"]).drop_duplicates("market_id")
        ven["market_id"] = ven["market_id"].astype(object)
        j = j.merge(ven.rename(columns={"market_id": "win_mid"}), on="win_mid", how="left")
        j["cc"] = cc
        frames.append(j)
    df = pd.concat(frames, ignore_index=True)
    del frames
    df["date"] = pd.to_datetime(df["date"]).values.astype("datetime64[D]")
    df["venue_key"] = df["cc"] + ":" + df["venue"].astype(str)
    df["meeting"] = df["venue_key"] + ":" + df["date"].astype(str)
    df["horse"] = df["cc"] + ":" + df["selection_id"].astype(str)
    u = df[df["pl_pre_all"] & df["pl_pre_book"].between(0.97, 1.03)].copy()
    n_univ = int(u["race"].nunique())
    sel = u[u["pre_pl"] >= 20]
    g = sel[sel["bsp_pl"].notna()].copy().reset_index(drop=True)
    P, F = g["bsp_pl"].to_numpy(), g["f_pl"].to_numpy()
    out({"section": "universe", "races_g6_clean": int(df["race"].nunique()), "R1b_races": n_univ,
         "decisions": int(len(sel)), "bets": int(len(g)), "median_bsp": r6(np.median(P)), "mean_bsp": r6(P.mean()),
         "p99_bsp": r6(np.quantile(P, 0.99)), "max_bsp": r6(P.max()), "strike": r6(F.mean()),
         "implied": r6(np.mean(1 / P))})

    # ------------------------------------------------------------ A reproduction with the original eval_cell
    nm = orig.n_months_of(u)
    for staking in ("stake", "liability"):
        row = orig.eval_cell(sel, sel["bsp_pl"].to_numpy(), sel["f_pl"].to_numpy(), staking, n_univ, nm)
        out({"section": f"A_orig_eval_cell_{staking}", **{k: row.get(k) for k in (
            "bets", "roi_8pct", "se_clustered", "se_race", "se_day", "se_iid", "z", "halves_roi_8pct", "halves_se",
            "half_split_date", "last12m_roi_8pct", "last12m_se", "passes_G4", "roi_0_5_10pct",
            "roi_8pct_market_netted", "strike_over_implied", "mde_80pct", "roi_8pct_excl_worst_bet")}})

    r = lay_stake(P, F)
    base = summarize(g, r, F, "A_own_per_stake")

    # ------------------------------------------------------------ B SE per stake
    wk = pd.to_datetime(g["date"]).dt.to_period("W").astype(str).to_numpy()
    mo = pd.to_datetime(g["date"]).dt.to_period("M").astype(str).to_numpy()
    q = 1 / P
    var_null = q * (1 - q) * (P - C8) ** 2
    se_null = float(math.sqrt(var_null.sum()) / len(P))
    ses = {"iid": float(r.std(ddof=1) / math.sqrt(len(r))), "market": cse(r, g["race"]), "day": cse(r, g["date"]),
           "meeting": cse(r, g["meeting"]), "venue": cse(r, g["venue_key"]), "horse": cse(r, g["horse"]),
           "week": cse(r, wk), "month": cse(r, mo),
           "twoway_horse_day": cse_twoway(r, g["horse"].to_numpy(), g["date"].astype(str).to_numpy()),
           "binomial_null_q_1_over_bsp": se_null}
    boots = {}
    for lab, cl in (("market", g["race"].to_numpy()), ("day", g["date"].to_numpy()), ("week", wk), ("month", mo),
                    ("horse", g["horse"].to_numpy())):
        s_, p_, ci = boot_se(r, cl, 2000, seed=len(lab))
        boots[lab] = {"se": r6(s_), "p_le_0": r6(p_), "ci95": ci}
    out({"section": "B_se_per_stake_by_cluster", "roi8": r6(r.mean()), "se": {k: r6(v) for k, v in ses.items()},
         "z": {k: r6(r.mean() / v) for k, v in ses.items()}, "boot": boots})

    # parametric null (exact small-probability tail, no CLT)
    day = g["date"].to_numpy()
    fh = day < np.datetime64(base["split"])
    k, sims = null_sim(P, r.mean(), fh, a.reps, 11)
    obs = [r.mean(), r[fh].mean(), r[~fh].mean()]
    pv = [(1 + (sims[:, i] >= obs[i]).sum()) / (a.reps + 1) for i in range(3)]
    sd = sims.std(0, ddof=1)
    out({"section": "B_parametric_null_per_stake", "k_scale_on_1_over_bsp": r6(k), "reps": a.reps,
         "null_sd_all_h1_h2": [r6(x) for x in sd], "obs_all_h1_h2": [r6(x) for x in obs],
         "z_vs_null_sd": [r6(obs[i] / sd[i]) for i in range(3)],
         "p_one_sided_empirical": [r6(x) for x in pv],
         "null_skew_all": r6(float(pd.Series(sims[:, 0]).skew()))})
    # same for per-liability (reference)
    rl = lay_liab(P, F)
    rng = np.random.default_rng(12)
    # null for liability: E[r_liab] = (1-q)(1-c)/(P-1) - q = 0 -> scale k_l
    kl = np.sum((1 - C8) / (P - 1)) / np.sum((1 - C8) / (P * (P - 1)) + 1 / P)
    ql = np.clip(kl / P, 0, 1)
    sl = np.empty(min(a.reps, 2000))
    sl1 = np.empty_like(sl)
    sl2 = np.empty_like(sl)
    for i in range(len(sl)):
        y = rng.random(len(P)) < ql
        rr = np.where(y, -1.0, (1 - C8) / (P - 1))
        sl[i], sl1[i], sl2[i] = rr.mean(), rr[fh].mean(), rr[~fh].mean()
    out({"section": "B_parametric_null_per_liability_reference", "k": r6(kl),
         "null_sd_all_h1_h2": [r6(sl.std(ddof=1)), r6(sl1.std(ddof=1)), r6(sl2.std(ddof=1))],
         "obs_all_h1_h2": [r6(rl.mean()), r6(rl[fh].mean()), r6(rl[~fh].mean())],
         "z_vs_null_sd": [r6(rl.mean() / sl.std(ddof=1)), r6(rl[fh].mean() / sl1.std(ddof=1)),
                          r6(rl[~fh].mean() / sl2.std(ddof=1))]})

    # ------------------------------------------------------------ C heavy tails / influence
    tot = r.sum()
    bk = {}
    for lo, hi in ((20, 30), (30, 50), (50, 100), (100, 200), (200, 500), (500, 1001), (1.01, 20)):
        m = (P >= lo) & (P < hi)
        if m.sum():
            bk[f"[{lo},{hi})"] = {"n": int(m.sum()), "losers": int((F[m] > 0).sum()), "roi_stake": r6(r[m].mean()),
                                  "se": r6(se_max(r[m], g[m])), "pnl_share_of_total": r6(r[m].sum() / tot),
                                  "strike_over_implied": r6(F[m].mean() / np.mean(1 / P[m]))}
    order = np.argsort(r)
    excl = {}
    for kx in (1, 3, 5, 10, 20, 50):
        rr = np.delete(r, order[:kx])
        excl[kx] = r6(rr.mean())
    incl_best = {}
    # symmetric: what ROI would 1, 3, 5 extra losses at the median losing price cost
    med_loss = float(np.median(-r[r < 0]))
    max_loss = float(-r.min())
    for kx in (1, 3, 5):
        incl_best[kx] = r6((tot - kx * max_loss) / len(r))
    out({"section": "C_per_stake_by_bsp_bucket", "table": bk, "total_pnl_units": r6(tot),
         "n_losers": int((F > 0).sum()), "median_loss": r6(med_loss), "max_loss": r6(max_loss),
         "roi_excluding_k_worst": excl, "roi_if_k_more_losses_at_max_loss": incl_best})
    for cap in (50, 100, 200, 500):
        m = P <= cap
        summarize(g[m], r[m], F[m], f"C_loc_cap_{cap}_per_stake", {"share_kept": r6(m.mean())})
    for wq in (0.999, 0.9995):
        lo = np.quantile(r, 1 - wq)
        summarize(g, np.maximum(r, lo), F, f"C_winsorised_{wq}_per_stake_descriptive", {"floor": r6(lo)})
    yr = pd.to_datetime(g["date"]).dt.year.to_numpy()
    ty = {}
    for y in sorted(np.unique(yr)):
        m = yr == y
        ty[int(y)] = [int(m.sum()), r6(r[m].mean()), r6(se_max(r[m], g[m]))]
    out({"section": "C_per_stake_by_year_bets_roi_se", "table": ty})
    seg = {}
    for s in ("au_gallops", "au_harness", "gb", "ie"):
        m = (g["seg"] == s).to_numpy()
        res = summarize(g[m], r[m], F[m], f"C_seg_{s}", verbose=False)
        seg[s] = [res.get("bets"), res.get("roi8"), res.get("se"), res.get("halves_z"), res.get("last12"),
                  res.get("g4")]
    for nw in (2, 3):
        m = (g["n_winners"] == nw).to_numpy()
        res = summarize(g[m], r[m], F[m], f"C_nw{nw}", verbose=False)
        seg[f"nw{nw}"] = [res.get("bets"), res.get("roi8"), res.get("se"), res.get("halves_z"), res.get("last12"),
                          res.get("g4")]
    out({"section": "C_per_stake_segments_bets_roi_se_halvesz_last12_g4", "table": seg})

    # ------------------------------------------------------------ D achievable sizing
    pre = g["pre_pl"].to_numpy()
    for lab, L in (("liab_eq_pre_minus_1", pre - 1), ("liab_eq_min(pre,50)-1", np.minimum(pre, 50) - 1),
                   ("liab_eq_min(pre,100)-1", np.minimum(pre, 100) - 1), ("liab_eq_sqrt(pre-1)", np.sqrt(pre - 1))):
        ret = L * rl  # per unit of intended backer stake (target stake 1 at the selection quote)
        res = summarize(g, ret, F, f"D_{lab}_per_intended_stake",
                        {"roi_per_unit_liability": r6(ret.sum() / L.sum()),
                         "actual_stake_per_bet_mean": r6(np.mean(L / (P - 1))),
                         "actual_stake_per_bet_median": r6(np.median(L / (P - 1)))})
    # D2 conservative SE (max of iid, market, day, meeting, venue, horse, ISO week) for the per-stake claim and
    # the achievable / capped variants; and the achievable sizing across thresholds (knife edge?)
    def se_cons(rr, gg):
        if len(rr) < 3:
            return float("nan")
        w_ = pd.to_datetime(gg["date"]).dt.to_period("W").astype(str).to_numpy()
        return max(float(rr.std(ddof=1) / math.sqrt(len(rr))), cse(rr, gg["race"].to_numpy()),
                   cse(rr, gg["date"].to_numpy()), cse(rr, gg["meeting"].to_numpy()),
                   cse(rr, gg["venue_key"].to_numpy()), cse(rr, gg["horse"].to_numpy()), cse(rr, w_))
    summarize(g, r, F, "D2_claim_per_bsp_stake_conservative_se", se_fn=se_cons)
    summarize(g, (pre - 1) * rl, F, "D2_liab_eq_pre_minus_1_conservative_se", se_fn=se_cons)
    m100 = P <= 100
    summarize(g[m100], r[m100], F[m100], "D2_loc_cap_100_per_stake_conservative_se", se_fn=se_cons)
    gridd = {}
    for T in (10, 15, 18, 20, 22, 25, 30, 40, 50):
        gg = u[(u["pre_pl"] >= T) & u["bsp_pl"].notna()]
        rr = (gg["pre_pl"].to_numpy() - 1) * lay_liab(gg["bsp_pl"].to_numpy(), gg["f_pl"].to_numpy())
        s = summarize(gg, rr, gg["f_pl"].to_numpy(), "x", verbose=False, se_fn=se_cons)
        gridd[T] = [s.get("bets"), s.get("roi8"), s.get("se"), s.get("halves_z"), s.get("last12"), s.get("g4")]
    out({"section": "D2_liab_eq_pre_minus_1_threshold_grid_conservative_se_bets_roi_se_halvesz_last12_g4",
         "table": gridd})
    # BSP-only (unachievable) vs pre: how different is the realised stake?
    out({"section": "D_bsp_over_pre_ratio", "median": r6(np.median(P / pre)), "p10": r6(np.quantile(P / pre, 0.1)),
         "p90": r6(np.quantile(P / pre, 0.9)), "corr_log": r6(np.corrcoef(np.log(P), np.log(pre))[0, 1])})

    # ------------------------------------------------------------ E splits
    for lab, cuts in (("period_median", [np.datetime64("2018-12-09")]),
                      ("thirds", [np.datetime64("2018-04-08"), np.datetime64("2019-07-23")]),
                      ("bets_median_from_claim", [np.datetime64("2019-03-02")])):
        edges = [np.datetime64("2016-11-30")] + cuts + [np.datetime64("2020-11-09")]
        parts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (day >= lo) & (day < hi)
            s_ = se_max(r[m], g[m])
            parts.append([int(m.sum()), r6(r[m].mean()), r6(s_), r6(r[m].mean() / s_)])
        out({"section": f"E_split_{lab}_bets_roi_se_z", "parts": parts})
    # market-netted commission per stake
    gross = (1 - F) - F * (P - 1)
    net = pd.Series(gross).groupby(g["race"].to_numpy()).sum().to_numpy()
    net = np.where(net > 0, net * (1 - C8), net)
    out({"section": "E_market_netted_per_stake", "roi8": r6(net.sum() / len(P)),
         "roi_0_5_10": [r6(lay_stake(P, F, c).mean()) for c in (0.0, 0.05, 0.10)]})

    # ------------------------------------------------------------ F multiple testing
    z_rep = 4.25
    z_cons = r.mean() / max(ses.values())
    rows = {}
    for m in (324, 4000, 4324, 6000, 10000):
        rows[m] = {"bonf_z_5pct_one_sided": r6(math.sqrt(2) * erfcinv(2 * 0.05 / m)),
                   "bh_rank1_z_q10": r6(norm.isf(0.10 / m)),
                   "survives_bh_rank1_reported": bool(norm.sf(z_rep) <= 0.10 / m),
                   "survives_bh_rank1_conservative": bool(norm.sf(z_cons) <= 0.10 / m)}
    mt = {"section": "F_multiple_testing", "z_reported": z_rep, "z_conservative": r6(z_cons),
          "p_conservative": float(norm.sf(z_cons)), "by_programme_size": rows}
    if a.cells and Path(a.cells).exists():
        cells = json.loads(Path(a.cells).read_text())["cells"]
        ps = np.array([c.get("p_one_sided") for c in cells if c.get("p_one_sided") is not None], float)
        ps = ps[np.isfinite(ps)]
        p_this = float(norm.sf(z_rep))
        for m_extra in (4000, 6000):
            # programme = this script's cells + m_extra other cells assumed null (uniform p, so ~0 below p_this)
            allp = np.sort(np.concatenate([ps, np.random.default_rng(5).random(m_extra)]))
            M = len(allp)
            thr = 0.10 * np.arange(1, M + 1) / M
            below = np.flatnonzero(allp <= thr)
            kmax = int(below.max() + 1) if len(below) else 0
            pcrit = float(allp[kmax - 1]) if kmax else 0.0
            mt[f"bh_q10_with_script_cells_plus_{m_extra}_null"] = {
                "M": int(M), "rejections": kmax, "p_crit": pcrit, "z_crit": r6(norm.isf(pcrit)) if pcrit > 0 else None,
                "claim_survives_reported_z": bool(p_this <= pcrit),
                "claim_survives_conservative_z": bool(float(norm.sf(z_cons)) <= pcrit)}
        mt["script_cells_with_p"] = int(len(ps))
        mt["script_cells_p_le_claim"] = int((ps <= float(norm.sf(z_rep))).sum())
    out(mt)

    # ------------------------------------------------------------ G placebos
    # G1 shuffled race days: G4 pass rate when the time order carries no information
    rng = np.random.default_rng(2024)
    days = np.unique(day)
    di = np.searchsorted(days, day)
    real_mid = np.searchsorted(days, np.datetime64(base["split"]))
    l12_days = int(((days >= LAST12_FROM) & (days <= LAST12_TO)).sum())
    passes, zmin, zfirst = 0, [], []
    race_arr = g["race"].to_numpy()
    for it in range(300):
        perm = rng.permutation(len(days))
        pdi = perm[di]
        h = pdi < real_mid
        l12 = pdi >= len(days) - l12_days
        a1, a2 = r[h], r[~h]
        s1 = max(float(a1.std(ddof=1) / math.sqrt(len(a1))), cse(a1, race_arr[h]), cse(a1, pdi[h]))
        s2 = max(float(a2.std(ddof=1) / math.sqrt(len(a2))), cse(a2, race_arr[~h]), cse(a2, pdi[~h]))
        zmin.append(min(a1.mean() / s1, a2.mean() / s2))
        zfirst.append(a1.mean() / s1)
        passes += int(a1.mean() > 2 * s1 and a2.mean() > 2 * s2 and r[l12].mean() > 0)
    zmin = np.array(zmin)
    out({"section": "G1_shuffled_days_per_stake", "reps": 300, "g4_pass_rate": r6(passes / 300),
         "min_half_z_mean_p5_p50_p95": [r6(zmin.mean()), r6(np.quantile(zmin, 0.05)), r6(np.median(zmin)),
                                        r6(np.quantile(zmin, 0.95))],
         "real_min_half_z": r6(min(base["halves_z"])),
         "share_perm_min_half_z_le_real": r6((zmin <= min(base["halves_z"])).mean())})

    # G2 other countries (same loader, same rule) per stake
    for cc in ("za", "fr"):
        try:
            j, st = orig.load_country(cc, d, a.frac, 7)
        except Exception as e:  # noqa: BLE001
            out({"section": f"G2_{cc}", "error": str(e)[:200]})
            continue
        j["date"] = pd.to_datetime(j["date"]).values.astype("datetime64[D]")
        uj = j[j["pl_pre_all"] & j["pl_pre_book"].between(0.97, 1.03)]
        for lab, gg in (("G1_book", uj[uj["pre_pl"] >= 20]), ("any_book_descriptive", j[j["pre_pl"] >= 20])):
            gg = gg[gg["bsp_pl"].notna()]
            if len(gg) < 10:
                out({"section": f"G2_{cc}_{lab}", "bets": int(len(gg))})
                continue
            pp, ffp = gg["bsp_pl"].to_numpy(), gg["f_pl"].to_numpy()
            rr = lay_stake(pp, ffp)
            qn = 1 / pp
            sn = float(math.sqrt((qn * (1 - qn) * (pp - C8) ** 2).sum()) / len(pp))
            summarize(gg, rr, ffp, f"G2_{cc}_{lab}_per_stake",
                      {"se_binomial_null": r6(sn), "z_null": r6(rr.mean() / sn),
                       "roi_liab": r6(lay_liab(pp, ffp).mean()),
                       "strike_over_implied": r6(ffp.mean() / np.mean(1 / pp)), "median_bsp": r6(np.median(pp))})
        del j, uj
    # G3 WIN-market analogue per stake (lay WIN ltp_pre >= T at WIN BSP, WIN book G1)
    uw = df[df["win_pre_all"] & df["win_pre_book"].between(0.97, 1.03)]
    for T in (50, 100):
        gg = uw[(uw["pre_win"] >= T) & uw["bsp_win"].notna()]
        summarize(gg, lay_stake(gg["bsp_win"].to_numpy(), gg["f_win"].to_numpy()), gg["f_win"].to_numpy(),
                  f"G3_win_market_lay_win_bsp_at_{T}_per_stake")
    # G4 price-matched control per stake: PLACE ltp_pre < 20 runners in the same place-BSP buckets, re-weighted
    edges = np.array([1.0, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1001])
    ctrl = u[(u["pre_pl"] < 20) & u["bsp_pl"].notna()]
    Pc, Fc = ctrl["bsp_pl"].to_numpy(), ctrl["f_pl"].to_numpy()
    rc = lay_stake(Pc, Fc)
    bt, bc = np.digitize(P, edges), np.digitize(Pc, edges)
    tab, wsum, rsum, rsum_t = {}, 0.0, 0.0, 0.0
    for b in np.unique(bt):
        nt, mc = int((bt == b).sum()), bc == b
        rt_b = float(r[bt == b].mean())
        rc_b = float(rc[mc].mean()) if mc.sum() >= 30 else None
        if rc_b is not None:
            wsum += nt
            rsum += nt * rc_b
            rsum_t += nt * rt_b
        tab[f"[{edges[b - 1]:g},{edges[b] if b < len(edges) else 'inf'})"] = [nt, r6(rt_b), int(mc.sum()),
                                                                              r6(rc_b) if rc_b is not None else None]
    out({"section": "G4_price_matched_control_per_stake", "bucket_nR1b_roiR1b_nctrl_roictrl": tab,
         "R1b_roi_on_matched_buckets": r6(rsum_t / wsum) if wsum else None,
         "ctrl_roi_reweighted_to_R1b_mix": r6(rsum / wsum) if wsum else None, "matched_bets": int(wsum)})

    # ------------------------------------------------------------ H threshold grid per stake
    grid = {}
    for T in (10, 15, 18, 20, 22, 25, 30, 40, 50, 100):
        gg = u[(u["pre_pl"] >= T) & u["bsp_pl"].notna()]
        s = summarize(gg, lay_stake(gg["bsp_pl"].to_numpy(), gg["f_pl"].to_numpy()), gg["f_pl"].to_numpy(),
                      f"H_{T}", verbose=False)
        grid[T] = [s.get("bets"), s.get("roi8"), s.get("se"), s.get("halves_z"), s.get("last12"), s.get("g4")]
    out({"section": "H_threshold_grid_per_stake_bets_roi_se_halvesz_last12_g4", "table": grid})


if __name__ == "__main__":
    main()

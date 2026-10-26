"""Adversarial verification #1 of "place_longshot_lay_sp" R1b@20 -- STATISTICS lens (round 2).

    python scripts/bf2_verify_place_longshot_lay_sp_1.py [--dir data/raw/betfair/split] [--frac 1.0]

Claim: lay every PLACE runner with PLACE ltp_pre >= 20 at the place BSP, 1 unit of LIABILITY, in AU/GB/IE
2/3-place markets paired with their WIN market (identical runner sets, settled, G6-clean) whose PLACE ltp_pre
book / n_winners is in [0.97, 1.03] with every runner priced. Claimed ROI8 +0.334% (SE 0.063, z 5.28).

The universe is built with the ORIGINAL loader (imported read-only from scripts/bf2_place_longshot_lay_sp.py);
verify_0 already re-implemented it independently and matched. This file probes the statistics:
  A  reproduction (orig.eval_cell, liability and stake)
  B  SE by more cluster definitions: market, race day, meeting, venue, HORSE (selection_id), ISO week, month,
     two-way horse x day (Cameron-Gelbach-Miller), binomial null SE at q = 1/BSP; block bootstraps (day, week,
     month, horse); halves / last 12 months with the most conservative SE
  C  multiple testing: BH / Bonferroni across the programme for several programme sizes
  D  threshold choice: fine grid of PLACE ltp_pre thresholds and bands (is 20 a knife edge?)
  E  placebos: (1) same rule in ZA and FR place markets (other countries, same loader); (2) WIN-market analogue
     (lay WIN ltp_pre >= T at WIN BSP, WIN book G1) in AU/GB/IE; (3) shuffled-date placebo: race days permuted
     at random, G4 re-applied to the permuted halves / "last 12 months" (how often G4 passes by chance for a
     constant edge, and where the real split sits); (4) price-matched placebo: lay runners with PLACE ltp_pre
     < 20 whose place BSP falls in the same BSP buckets as the R1b bets (re-weighted to the R1b BSP mix)
  F  time stability: quarters, rolling 12-month windows
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


def lay_liab(price, f, c=C8):
    gross = (1 - f) - f * (price - 1)
    net = np.where(gross > 0, gross * (1 - c), gross)
    return net / (price - 1)


def cse(r, cl):
    d = pd.Series(r - r.mean()).groupby(np.asarray(cl)).sum().to_numpy()
    G = len(d)
    return float(math.sqrt((d ** 2).sum() * G / (G - 1)) / len(r)) if G > 1 else float("nan")


def cse_twoway(r, a, b):
    """CGM two-way cluster: V = V_a + V_b - V_ab."""
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


def se_with_null(price):
    """se_max combined with the binomial null SE at q = 1/price (used where events are few)."""
    def fn(rr, gg, _p=price):
        pp = gg["_px"].to_numpy()
        qq = 1 / pp
        vn = qq * (1 - qq) * (1 + (1 - C8) / (pp - 1)) ** 2
        return max(se_max(rr, gg), float(math.sqrt(vn.sum()) / len(rr)))
    return fn


def summarize(g, price, f, label, extra=None, se_fn=None, verbose=True, null=False):
    ok = np.isfinite(price)
    g, price, f = g[ok], price[ok], f[ok]
    if null:
        g = g.assign(_px=price)
        se_fn = se_with_null(price)
    if len(g) < 10:
        if verbose:
            out({"section": label, "bets": int(len(g))})
        return {"bets": int(len(g))}
    se_fn = se_fn or se_max
    r = lay_liab(price, f)
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
           "roi_0_5_10": [r6(lay_liab(price, f, c).mean()) for c in (0.0, 0.05, 0.10)],
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data/raw/betfair/split"))
    ap.add_argument("--frac", type=float, default=1.0)
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
         "decisions": int(len(sel)), "bets": int(len(g))})

    # ---------------------------------------------------------------- A reproduction with the original eval_cell
    nm = orig.n_months_of(u)
    for staking in ("liability", "stake"):
        row = orig.eval_cell(sel, sel["bsp_pl"].to_numpy(), sel["f_pl"].to_numpy(), staking, n_univ, nm)
        out({"section": f"A_orig_eval_cell_{staking}", **{k: row.get(k) for k in (
            "bets", "roi_8pct", "se_clustered", "se_race", "se_day", "se_iid", "z", "halves_roi_8pct", "halves_se",
            "half_split_date", "last12m_roi_8pct", "last12m_se", "passes_G4", "roi_0_5_10pct",
            "roi_8pct_market_netted", "strike_over_implied", "mde_80pct")}})

    # ---------------------------------------------------------------- B SE by many clusterings
    r = lay_liab(P, F)
    wk = pd.to_datetime(g["date"]).dt.to_period("W").astype(str).to_numpy()
    mo = pd.to_datetime(g["date"]).dt.to_period("M").astype(str).to_numpy()
    q = 1 / P
    var_null = q * (1 - q) * (1 + (1 - C8) / (P - 1)) ** 2
    se_null = float(math.sqrt(var_null.sum()) / len(P))
    ses = {"iid": float(r.std(ddof=1) / math.sqrt(len(r))), "market": cse(r, g["race"]), "day": cse(r, g["date"]),
           "meeting": cse(r, g["meeting"]), "venue": cse(r, g["venue_key"]), "horse": cse(r, g["horse"]),
           "week": cse(r, wk), "month": cse(r, mo),
           "twoway_horse_day": cse_twoway(r, g["horse"].to_numpy(), g["date"].astype(str).to_numpy()),
           "binomial_null_q_1_over_bsp": se_null}
    boots = {}
    for lab, cl in (("day", g["date"].to_numpy()), ("week", wk), ("month", mo), ("horse", g["horse"].to_numpy())):
        s_, p_ = boot_se(r, cl, 2000, seed=len(lab))
        boots[lab] = [r6(s_), r6(p_)]
    hc = g.groupby("horse").size()
    out({"section": "B_se_by_cluster", "roi8": r6(r.mean()), "se": {k: r6(v) for k, v in ses.items()},
         "z": {k: r6(r.mean() / v) for k, v in ses.items()}, "boot_se_and_p_le_0": boots,
         "n_horses": int(len(hc)), "bets_per_horse_mean": r6(hc.mean()), "bets_per_horse_p99": r6(hc.quantile(0.99)),
         "max_bets_one_horse": int(hc.max()), "n_venues": int(g["venue_key"].nunique()),
         "n_meetings": int(g["meeting"].nunique()), "n_days": int(g["date"].nunique())})

    def se_conservative(rr, gg):
        if len(rr) < 3:
            return float("nan")
        qq = 1 / gg["bsp_pl"].to_numpy()
        vn = qq * (1 - qq) * (1 + (1 - C8) / (gg["bsp_pl"].to_numpy() - 1)) ** 2
        w_ = pd.to_datetime(gg["date"]).dt.to_period("W").astype(str).to_numpy()
        return max(float(rr.std(ddof=1) / math.sqrt(len(rr))), cse(rr, gg["race"].to_numpy()),
                   cse(rr, gg["date"].to_numpy()), cse(rr, gg["horse"].to_numpy()), cse(rr, gg["meeting"].to_numpy()),
                   cse(rr, gg["venue_key"].to_numpy()), cse(rr, w_), float(math.sqrt(vn.sum()) / len(rr)))
    cons = summarize(g, P, F, "B_conservative_se_max_of_all", se_fn=se_conservative)
    # calendar split of the period instead of the bets' median, and thirds
    day = g["date"].to_numpy()
    for lab, cuts in (("period_median", [np.datetime64("2018-12-05")]),
                      ("thirds", [np.datetime64("2018-04-08"), np.datetime64("2019-07-23")])):
        edges = [np.datetime64("2016-11-30")] + cuts + [np.datetime64("2020-11-09")]
        parts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (day >= lo) & (day < hi)
            parts.append([int(m.sum()), r6(r[m].mean()), r6(se_conservative(r[m], g[m])),
                          r6(r[m].mean() / se_conservative(r[m], g[m]))])
        out({"section": f"B_split_{lab}_bets_roi_se_z", "parts": parts})

    # ---------------------------------------------------------------- C programme-wide multiple testing
    z_claim = r.mean() / max(ses.values())
    p_claim = float(norm.sf(z_claim))
    p_reported = float(norm.sf(5.28))
    rows = {}
    for m in (324, 4000, 4324, 6000, 10000, 20000, 50000):
        rows[m] = {"bonf_z_5pct_one_sided": r6(math.sqrt(2) * erfcinv(2 * 0.05 / m)),
                   "bh_rank1_p_threshold_q10": r6(0.10 / m),
                   "survives_bh_q10_as_rank1_reported_z": bool(p_reported <= 0.10 / m),
                   "survives_bh_q10_as_rank1_conservative_z": bool(p_claim <= 0.10 / m),
                   "survives_bonf_5pct_conservative_z": bool(p_claim <= 0.05 / m)}
    out({"section": "C_multiple_testing", "z_reported": 5.28, "p_reported": p_reported,
         "z_conservative_se": r6(z_claim), "p_conservative": p_claim,
         "max_programme_cells_for_bh_q10_rank1": int(0.10 / p_claim), "by_programme_size": rows})

    # ---------------------------------------------------------------- D threshold choice
    grid = {}
    for T in (10, 12, 15, 18, 19, 20, 21, 22, 25, 30, 35, 40, 50, 75, 100):
        gg = u[(u["pre_pl"] >= T) & u["bsp_pl"].notna()]
        s = summarize(gg, gg["bsp_pl"].to_numpy(), gg["f_pl"].to_numpy(), f"D_thr_{T}", verbose=False)
        grid[T] = [s.get("bets"), s.get("roi8"), s.get("z"), s.get("halves_z"), s.get("last12"), s.get("g4")]
    out({"section": "D_threshold_grid_bets_roi_z_halvesz_last12_g4", "table": grid})
    bands = {}
    for lo, hi in ((5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 50), (50, 75), (75, 100),
                   (100, 1001)):
        gg = u[(u["pre_pl"] >= lo) & (u["pre_pl"] < hi) & u["bsp_pl"].notna()]
        s = summarize(gg, gg["bsp_pl"].to_numpy(), gg["f_pl"].to_numpy(), "band", verbose=False)
        bands[f"[{lo},{hi})"] = [s.get("bets"), s.get("roi8"), s.get("z"), s.get("strike_over_implied")]
    out({"section": "D_place_ltp_pre_bands_bets_roi_z_strike_over_implied", "table": bands})

    # ---------------------------------------------------------------- E placebos
    # E1 other countries with place BSPs (same loader: pairing, settled, G6, G1, threshold 20)
    for cc in ("za", "fr"):
        try:
            j, st = orig.load_country(cc, d, a.frac, 7)
        except Exception as e:  # noqa: BLE001
            out({"section": f"E1_{cc}", "error": str(e)[:200]})
            continue
        j["date"] = pd.to_datetime(j["date"]).values.astype("datetime64[D]")
        uj = j[j["pl_pre_all"] & j["pl_pre_book"].between(0.97, 1.03)]
        out({"section": f"E1_{cc}_universe", "races": int(j["race"].nunique()), "R1b_races": int(uj["race"].nunique()),
             "share_runners_pl_bsp": st.get("share_runners_pl_bsp")})
        for T in (20, 50):
            gg = uj[(uj["pre_pl"] >= T)]
            summarize(gg, gg["bsp_pl"].to_numpy(), gg["f_pl"].to_numpy(), f"E1_{cc}_R1b_at_{T}",
                      {"decisions": int(len(gg)), "with_bsp": int(gg["bsp_pl"].notna().sum())}, null=True)
        # looser universe: no book filter (G1 broken, descriptive) to get power
        gg = j[(j["pre_pl"] >= 20)]
        summarize(gg, gg["bsp_pl"].to_numpy(), gg["f_pl"].to_numpy(), f"E1_{cc}_any_book_at_20_descriptive",
                  null=True)
        for T in (10, 15, 30, 50):
            gg = j[(j["pre_pl"] >= T)]
            summarize(gg, gg["bsp_pl"].to_numpy(), gg["f_pl"].to_numpy(), f"E1_{cc}_any_book_at_{T}_descriptive",
                      null=True)
        del j, uj
    # E2 WIN-market analogue in AU/GB/IE: lay WIN ltp_pre >= T at WIN BSP (WIN book G1)
    uw = df[df["win_pre_all"] & df["win_pre_book"].between(0.97, 1.03)]
    for T in (20, 50, 100):
        gg = uw[uw["pre_win"] >= T]
        summarize(gg, gg["bsp_win"].to_numpy(), gg["f_win"].to_numpy(), f"E2_win_market_lay_win_bsp_at_{T}")
    # same races as R1b, WIN-market lay of the very runners R1b lays (cross-market placebo)
    summarize(g, g["bsp_win"].to_numpy(), g["f_win"].to_numpy(), "E2_R1b_runners_laid_in_WIN_at_win_bsp")

    # E3 shuffled-date placebo: permute race days across days; re-apply the G4 half split / last-12 window
    rng = np.random.default_rng(2024)
    days = np.unique(day)
    di = np.searchsorted(days, day)
    real_mid = np.searchsorted(days, np.datetime64(cons["split"]))
    l12_days = int(((days >= LAST12_FROM) & (days <= LAST12_TO)).sum())
    passes, z1s, z2s, l12s = 0, [], [], []
    for it in range(300):
        perm = rng.permutation(len(days))
        pd_ = perm[di]  # permuted day index per bet
        fh = pd_ < real_mid
        l12 = pd_ >= len(days) - l12_days
        a1, a2 = r[fh], r[~fh]
        s1 = max(float(a1.std(ddof=1) / math.sqrt(len(a1))), cse(a1, g["race"].to_numpy()[fh]), cse(a1, pd_[fh]))
        s2 = max(float(a2.std(ddof=1) / math.sqrt(len(a2))), cse(a2, g["race"].to_numpy()[~fh]), cse(a2, pd_[~fh]))
        z1s.append(a1.mean() / s1)
        z2s.append(a2.mean() / s2)
        l12s.append(r[l12].mean())
        passes += int(a1.mean() > 2 * s1 and a2.mean() > 2 * s2 and r[l12].mean() > 0)
    z1s, z2s = np.array(z1s), np.array(z2s)
    out({"section": "E3_shuffled_dates", "reps": 300, "g4_pass_rate": r6(passes / 300),
         "half_z_min_mean_p5_p50": [r6(np.mean(np.minimum(z1s, z2s))), r6(np.quantile(np.minimum(z1s, z2s), 0.05)),
                                    r6(np.median(np.minimum(z1s, z2s)))],
         "real_half_z": cons["halves_z"], "last12_roi_p5_p50": [r6(np.quantile(l12s, 0.05)), r6(np.median(l12s))],
         "real_last12": cons["last12"]})

    # E4 price-matched placebo: runners NOT selected (place ltp_pre < 20) with a place BSP in the R1b BSP buckets,
    # re-weighted to the R1b bucket mix
    edges = np.array([1.0, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 1001])
    ctrl = u[(u["pre_pl"] < 20) & u["bsp_pl"].notna()]
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
        tab[f"[{edges[b - 1]:g},{edges[b] if b < len(edges) else 'inf'})"] = [nt, r6(rt_b), int(mc.sum()),
                                                                              r6(rc_b) if rc_b is not None else None]
    out({"section": "E4_price_matched_placebo", "bucket_nR1b_roiR1b_nctrl_roictrl": tab,
         "R1b_roi_on_matched_buckets": r6(rsum_t / wsum) if wsum else None,
         "ctrl_roi_reweighted_to_R1b_mix": r6(rsum / wsum) if wsum else None, "matched_bets": int(wsum)})

    # ---------------------------------------------------------------- F time stability
    qtr = pd.to_datetime(g["date"]).dt.to_period("Q").astype(str).to_numpy()
    tq = {}
    for k in sorted(np.unique(qtr)):
        m = qtr == k
        tq[k] = [int(m.sum()), r6(r[m].mean()), r6(r[m].mean() / se_max(r[m], g[m]))]
    out({"section": "F_quarters_bets_roi_z", "table": tq, "n_negative": int(sum(v[1] < 0 for v in tq.values())),
         "n_quarters": len(tq)})
    starts = pd.date_range("2016-12-01", "2019-11-01", freq="MS")
    roll = []
    for s0 in starts:
        lo, hi = np.datetime64(s0.date()), np.datetime64((s0 + pd.DateOffset(months=12)).date())
        m = (day >= lo) & (day < hi)
        roll.append(r[m].mean())
    roll = np.array(roll)
    out({"section": "F_rolling_12m", "n_windows": int(len(roll)), "min": r6(roll.min()), "max": r6(roll.max()),
         "n_negative": int((roll < 0).sum())})


if __name__ == "__main__":
    main()

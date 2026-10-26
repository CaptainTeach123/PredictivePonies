"""Adversarial verification of "place_longshot_lay_sp" R1a@100 (round 2), LEAKAGE / DATA QUALITY /
EXECUTION lens.

    python scripts/bf2_verify_place_longshot_lay_sp_0_r1a100.py [--dir data/raw/betfair/split] [--frac 1.0]

Claim: lay every runner with WIN ltp_pre >= 100 (WIN market fully priced at ltp_pre, book in [0.97, 1.03])
in the PLACE market at the place BSP, 1 unit of LIABILITY, AU/GB/IE 2/3-place paired, settled, G6-clean
markets. Claimed ROI8 +0.2646% (SE 0.0544, z 4.86), halves +0.230/+0.299, last 12m +0.360.

The universe is built with the ORIGINAL loader (imported read-only; WCOLS extended with venue so meetings
can be clustered). Probes:
  A  reproduction with orig.eval_cell (liability and stake), by segment
  B  SE by more cluster definitions (meeting, venue, horse, ISO week, month) and block bootstraps; G4 with
     the most conservative SE
  C  price impact / capacity: BSP x (1+x), BSP + k ticks, break-even impact; liability-weighted profit by
     place-BSP bucket and by place BSP / place ltp_pre (where does the money come from?)
  D  where the edge sits on the WIN ladder: WIN ltp_pre bands and a threshold grid (is 100 a peak?)
  E  overlap with R1b@20 (pre-registered): share of bets in common and the ROI of the R1a@100 bets that are
     NOT R1b@20 bets (incremental evidence)
  F  decision timing / staleness: AU harness excluded (its WIN ltp_pre falls back to the scheduled start);
     also require WIN ltp_t30 >= 100 (stable longshots); selection on WIN ltp_t30 alone in the R1a universe
  G  dead heats as full losses for the layer (worst case); missing-BSP decisions settled at place ltp_pre
  H  stability: country x year, half-years, venue concentration
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("orig_pll", ROOT / "scripts/bf2_place_longshot_lay_sp.py")
orig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(orig)
orig.WCOLS = orig.WCOLS + ["venue"]

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


def boot_se(r, cl, reps=1000, seed=0):
    codes, uniq = pd.factorize(np.asarray(cl))
    s = np.bincount(codes, weights=r)
    c = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    G = len(uniq)
    b = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, G, G)
        b[i] = s[idx].sum() / c[idx].sum()
    return float(np.std(b, ddof=1))


def step_at(p):
    p = np.asarray(p, float)
    s = np.full_like(p, 10.0)
    for lo, hi, inc in orig.TICKS:
        s = np.where((p >= lo - 1e-9) & (p < hi - 1e-9), inc, s)
    return s


def se_std(r, g):
    n = len(r)
    return max(float(r.std(ddof=1) / math.sqrt(n)), cse(r, g["race"].to_numpy()), cse(r, g["date"].to_numpy()))


def se_cons(r, g):
    """most conservative of iid, race, day, meeting, week, month clustering"""
    n = len(r)
    cands = [float(r.std(ddof=1) / math.sqrt(n)), cse(r, g["race"].to_numpy()), cse(r, g["date"].to_numpy()),
             cse(r, g["meeting"].to_numpy()), cse(r, g["week"].to_numpy()), cse(r, g["mon"].to_numpy())]
    return max(x for x in cands if np.isfinite(x))


def summ(g, price, f, label, extra=None, se_fn=se_std, verbose=True, c=C8):
    ok = np.isfinite(price)
    g, price, f = g[ok], price[ok], f[ok]
    if len(g) < 5:
        res = {"section": label, "bets": int(len(g))}
        if verbose:
            out(res)
        return res
    r = lay_liab(price, f, c)
    day = g["date"].to_numpy()
    rd = pd.Series(day).groupby(g["race"].to_numpy()).first().to_numpy()
    mid = np.sort(rd)[len(rd) // 2]
    fh = day < mid
    l12 = (day >= LAST12_FROM) & (day <= LAST12_TO)
    se = se_fn(r, g)
    parts = {}
    for nm, m in (("h1", fh), ("h2", ~fh), ("l12", l12)):
        if m.sum() >= 5:
            parts[nm] = (float(r[m].mean()), se_fn(r[m], g[m]), int(m.sum()), int((f[m] > 0).sum()))
        else:
            parts[nm] = (float("nan"), float("nan"), int(m.sum()), 0)
    g4 = bool(parts["h1"][0] > 2 * parts["h1"][1] and parts["h2"][0] > 2 * parts["h2"][1]
              and parts["l12"][0] > 0 and parts["h1"][3] >= 10 and parts["h2"][3] >= 10)
    res = {"section": label, "bets": int(len(r)), "roi8": r6(r.mean()), "se": r6(se),
           "z": r6(r.mean() / se) if se > 0 else None,
           "halves": [r6(parts["h1"][0]), r6(parts["h2"][0])], "halves_se": [r6(parts["h1"][1]), r6(parts["h2"][1])],
           "split": str(mid)[:10], "last12": r6(parts["l12"][0]), "last12_se": r6(parts["l12"][1]),
           "last12_bets": parts["l12"][2],
           "strike_over_implied": r6(f.mean() / np.mean(1 / price)), "median_price": r6(np.median(price)),
           "g4": g4}
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
    for cc in orig.COUNTRIES:
        j, st = orig.load_country(cc, d, a.frac, 7)
        j["cc"] = cc
        frames.append(j)
    df = pd.concat(frames, ignore_index=True)
    del frames
    df["date"] = pd.to_datetime(df["date"]).values.astype("datetime64[D]")
    df["year"] = df["start"].dt.year
    df["meeting"] = df["cc"] + ":" + df["venue"].astype(str) + ":" + df["date"].astype(str)
    df["vkey"] = df["cc"] + ":" + df["venue"].astype(str)
    dts = pd.to_datetime(df["date"])
    df["week"] = dts.dt.to_period("W").astype(str)
    df["mon"] = dts.dt.to_period("M").astype(str)
    df["horse"] = df["selection_id"].astype(str)
    out({"section": "universe_all", "races": int(df["race"].nunique()), "runners": int(len(df))})

    u = df[df["win_pre_all"] & df["win_pre_book"].between(orig.BOOK_LO, orig.BOOK_HI)].copy()
    n_univ = int(u["race"].nunique())
    out({"section": "universe_R1a", "races": n_univ,
         "by_seg": u.drop_duplicates("race")["seg"].value_counts().to_dict()})
    sel = u[u["pre_win"] >= 100].copy()
    hasb = sel["bsp_pl"].notna().to_numpy()
    out({"section": "decisions", "n": int(len(sel)), "with_bsp": int(hasb.sum()),
         "by_seg_with_bsp": sel[hasb]["seg"].value_counts().to_dict()})

    # A reproduction with the original eval_cell
    for staking in ("liability", "stake"):
        row = orig.eval_cell(sel, sel["bsp_pl"].to_numpy(), sel["f_pl"].to_numpy(), staking, n_univ,
                             int(u["month"].nunique()))
        out({"section": f"A_orig_eval_cell_{staking}", **{k: row.get(k) for k in (
            "bets", "roi_8pct", "se_clustered", "se_race", "se_day", "se_iid", "z", "halves_roi_8pct", "halves_se",
            "half_split_date", "last12m_roi_8pct", "last12m_se", "roi_0_5_10pct", "passes_G4",
            "strike_over_implied", "median_price", "roi_8pct_market_netted", "mde_80pct")}})
    g = sel[hasb].copy()
    P, F = g["bsp_pl"].to_numpy(), g["f_pl"].to_numpy()
    r = lay_liab(P, F)
    summ(g, P, F, "A_reproduction")
    for seg in ("au_gallops", "au_harness", "gb", "ie"):
        m = (g["seg"] == seg).to_numpy()
        summ(g[m], P[m], F[m], f"A_seg_{seg}")

    # B SE by cluster definition and block bootstraps
    ses = {"iid": float(r.std(ddof=1) / math.sqrt(len(r)))}
    for k in ("race", "date", "meeting", "vkey", "horse", "week", "mon"):
        ses[k] = cse(r, g[k].to_numpy())
    for k in ("date", "week", "mon"):
        ses[f"boot_{k}"] = boot_se(r, g[k].to_numpy(), seed=hash(k) % 1000)
    out({"section": "B_se_by_cluster", "roi8": r6(r.mean()), "se": {k: r6(v) for k, v in ses.items()},
         "z_min": r6(r.mean() / max(ses.values()))})
    summ(g, P, F, "B_conservative_se_G4", se_fn=se_cons)

    # C price impact / capacity
    imp = {}
    for x in (0.005, 0.01, 0.02, 0.03, 0.05):
        rr = lay_liab(P * (1 + x), F)
        imp[f"bsp_x{1 + x:.3f}"] = [r6(rr.mean()), r6(rr.mean() / se_std(rr, g))]
    for k in (1, 2):
        Pk = P.copy()
        for _ in range(k):
            Pk = np.minimum(Pk + step_at(Pk), 1000.0)
        rr = lay_liab(Pk, F)
        imp[f"bsp_plus_{k}_tick"] = [r6(rr.mean()), r6(rr.mean() / se_std(rr, g))]
    lo, hi = 0.0, 0.5
    for _ in range(40):
        m_ = (lo + hi) / 2
        if lay_liab(P * (1 + m_), F).mean() > 0:
            lo = m_
        else:
            hi = m_
    imp["breakeven_multiplicative_impact"] = r6(lo)
    imp["median_tick_pct_at_bsp"] = r6(np.median(step_at(P) / P))
    out({"section": "C_price_impact_roi_z", **imp})
    # one tick longer with each bet's own G4
    Pk = np.minimum(P + step_at(P), 1000.0)
    summ(g, Pk, F, "C_bsp_plus_1_tick_G4")
    summ(g, P * 1.02, F, "C_bsp_x1.02_G4")
    # backer stake per unit liability, and where the profit comes from
    stake = 1 / (P - 1)
    out({"section": "C_stake_per_unit_liability", "mean": r6(stake.mean()), "median": r6(np.median(stake)),
         "share_of_stake_from_bsp_lt10": r6(stake[P < 10].sum() / stake.sum())})
    tab = {}
    for lo_, hi_ in ((1.01, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 200), (200, 1001)):
        m = (P >= lo_) & (P < hi_)
        if m.sum() > 5:
            tab[f"[{lo_},{hi_})"] = {"n": int(m.sum()), "profit_share": r6(r[m].sum() / r.sum()),
                                     "roi8": r6(r[m].mean()), "z": r6(r[m].mean() / se_std(r[m], g[m])),
                                     "strike_over_implied": r6(F[m].mean() / np.mean(1 / P[m]))}
    out({"section": "C_by_place_bsp_bucket", "total_profit_units": r6(r.sum()), "table": tab})
    pre = g["pre_pl"].to_numpy()
    ratio = P / pre
    tab = {}
    for lo_, hi_ in ((0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.05), (1.05, 1.25), (1.25, 2), (2, 1e9)):
        m = (ratio >= lo_) & (ratio < hi_)
        if m.sum() > 5:
            tab[f"[{lo_},{hi_})"] = {"n": int(m.sum()), "profit_share": r6(r[m].sum() / r.sum()),
                                     "roi8": r6(r[m].mean()), "z": r6(r[m].mean() / se_std(r[m], g[m])),
                                     "strike_over_implied": r6(F[m].mean() / np.mean(1 / P[m]))}
    m = ~np.isfinite(ratio)
    tab["no_place_ltp_pre"] = {"n": int(m.sum()), "profit_share": r6(r[m].sum() / r.sum()) if m.any() else None}
    out({"section": "C_by_bsp_over_place_ltp_pre", "table": tab})
    summ(g, np.where(np.isfinite(pre), pre, np.nan), F, "C_settle_at_place_ltp_pre_descriptive")
    summ(g, np.where(np.isfinite(pre), np.maximum(P, pre), np.nan), F, "C_settle_at_max_bsp_ltp_pre_adversarial")
    m = np.isfinite(ratio) & (ratio >= 0.5) & (ratio <= 2)
    summ(g[m], P[m], F[m], "C_bsp_within_2x_place_ltp_pre")

    # D WIN ladder bands and threshold grid
    tab = {}
    W = g["pre_win"].to_numpy()
    for lo_, hi_ in ((100, 150), (150, 200), (200, 300), (300, 500), (500, 1000), (1000, 1001)):
        m = (W >= lo_) & (W < hi_)
        if m.sum() > 5:
            s = summ(g[m], P[m], F[m], "", verbose=False)
            tab[f"[{lo_},{hi_})"] = [s["bets"], s["roi8"], s["z"], s["strike_over_implied"], s["median_price"],
                                     r6(r[m].sum() / r.sum())]
    out({"section": "D_win_band_bets_roi_z_soi_medbsp_profitshare", "table": tab})
    grid = {}
    for T in (50, 60, 70, 80, 90, 100, 110, 120, 150, 200, 300):
        gg = u[(u["pre_win"] >= T) & u["bsp_pl"].notna()]
        s = summ(gg, gg["bsp_pl"].to_numpy(), gg["f_pl"].to_numpy(), "", verbose=False)
        grid[T] = [s["bets"], s["roi8"], s["z"], s["halves"], s["last12"], s["g4"]]
    out({"section": "D_threshold_grid_bets_roi_z_halves_last12_g4", "table": grid})

    # E overlap with R1b@20
    r1b_ok = g["pl_pre_all"].to_numpy() & (g["pl_pre_book"].between(orig.BOOK_LO, orig.BOOK_HI).to_numpy())
    in_r1b = r1b_ok & (g["pre_pl"].to_numpy() >= 20)
    out({"section": "E_overlap", "share_in_R1b_universe": r6(r1b_ok.mean()), "share_also_R1b20_bet": r6(in_r1b.mean()),
         "profit_share_from_R1b20_bets": r6(r[in_r1b].sum() / r.sum())})
    summ(g[in_r1b], P[in_r1b], F[in_r1b], "E_R1a100_and_R1b20")
    summ(g[~in_r1b], P[~in_r1b], F[~in_r1b], "E_R1a100_not_R1b20")
    m = ~r1b_ok
    summ(g[m], P[m], F[m], "E_R1a100_outside_R1b_universe")
    m = r1b_ok & (g["pre_pl"].to_numpy() < 20)
    summ(g[m], P[m], F[m], "E_R1a100_in_R1b_universe_place_pre_lt20")

    # F timing / staleness
    m = (g["seg"] != "au_harness").to_numpy()
    summ(g[m], P[m], F[m], "F_excl_au_harness")
    m = (g["t30_win"] >= 100).to_numpy()
    summ(g[m], P[m], F[m], "F_also_win_t30_ge_100")
    m = g["t30_win"].isna().to_numpy()
    summ(g[m], P[m], F[m], "F_no_win_t30")
    m = (g["t30_win"] < 100).to_numpy()
    summ(g[m], P[m], F[m], "F_win_t30_lt_100_steamed_in")
    gt = u[(u["t30_win"] >= 100) & u["bsp_pl"].notna()]
    summ(gt, gt["bsp_pl"].to_numpy(), gt["f_pl"].to_numpy(), "F_select_on_win_t30_ge_100_in_R1a_universe")
    # WIN BSP check: ratio of WIN BSP to WIN ltp_pre for the selected runners (descriptive drift at the off)
    lr = np.log(g["bsp_win"] / g["pre_win"])
    out({"section": "F_win_bsp_vs_win_ltp_pre", "mean_log": r6(np.nanmean(lr)), "share_bsp_lt_100": r6(
        np.nanmean(g["bsp_win"] < 100))})

    # G dead heats worst case; missing BSP settled at place ltp_pre
    F2 = np.where(F > 0, 1.0, 0.0)
    summ(g, P, F2, "G_deadheat_full_loss", {"dead_heat_bets": int(((F > 0) & (F < 1)).sum())})
    allp = np.where(sel["bsp_pl"].notna(), sel["bsp_pl"], sel["pre_pl"])
    summ(sel, allp, sel["f_pl"].to_numpy(), "G_missing_bsp_at_place_ltp_pre",
         {"extra_bets": int((sel["bsp_pl"].isna() & sel["pre_pl"].notna()).sum())})

    # H stability
    tab = {}
    for (cc, y), gy in g.groupby([g["seg"], g["year"]]):
        rr = lay_liab(gy["bsp_pl"].to_numpy(), gy["f_pl"].to_numpy())
        tab[f"{cc}|{y}"] = [int(len(gy)), r6(rr.mean()), r6(se_std(rr, gy)) if len(gy) > 5 else None]
    out({"section": "H_seg_year_bets_roi_se", "table": tab})
    hy = (g["year"].astype(str) + "H" + (pd.to_datetime(g["date"]).dt.month > 6).map({False: "1", True: "2"})).to_numpy()
    tab = {}
    for k in sorted(set(hy)):
        m = hy == k
        tab[k] = [int(m.sum()), r6(r[m].mean()), r6(se_std(r[m], g[m]))]
    out({"section": "H_half_year_bets_roi_se", "table": tab, "n_negative": int(sum(v[1] < 0 for v in tab.values()))})
    vkey = g["vkey"].to_numpy()
    vp = pd.Series(r).groupby(vkey).sum().sort_values(ascending=False)
    res = {"venues": int(len(vp)), "total_profit": r6(vp.sum()), "top5_profit": r6(vp.iloc[:5].sum()),
           "top10_profit": r6(vp.iloc[:10].sum()), "share_venues_positive": r6((vp > 0).mean())}
    for k in (5, 10, 20):
        m = ~np.isin(vkey, vp.index[:k])
        res[f"roi_excl_top{k}"] = r6(r[m].mean())
        res[f"z_excl_top{k}"] = r6(r[m].mean() / se_std(r[m], g[m]))
    out({"section": "H_venue_concentration", **res})

    # corrected estimate: conservative SE, at BSP and at +1 tick
    summ(g, P, F, "Z_corrected_conservative_se_bsp", se_fn=se_cons)
    summ(g, Pk, F, "Z_corrected_conservative_se_plus1tick", se_fn=se_cons)
    m = (g["seg"] == "au_gallops").to_numpy()
    summ(g[m], P[m], F[m], "Z_au_gallops_conservative_se", se_fn=se_cons)


if __name__ == "__main__":
    main()

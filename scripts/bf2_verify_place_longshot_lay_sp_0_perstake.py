"""Adversarial verification of "place_longshot_lay_sp" R1b@20 PER UNIT OF BACKER'S STAKE (round 2).

    python scripts/bf2_verify_place_longshot_lay_sp_0_perstake.py [--dir data/raw/betfair/split] [--frac 1.0]

(Written as a new sibling of bf2_verify_place_longshot_lay_sp_0.py, which another verifier already created.)

Claim: lay every PLACE runner with PLACE ltp_pre >= 20 at the place BSP, 1 unit of BACKER'S STAKE, in AU/GB/IE
2- or 3-place markets paired with their WIN market (identical runner sets, settled, G6-clean) whose PLACE
ltp_pre book / n_winners is in [0.97, 1.03] with every runner priced. Claimed ROI8 +11.14% (SE 2.62), halves
+7.37% (4.16) / +14.82% (3.26), last 12 months +17.60%; fails G4.

Independent re-implementation of the universe (does not import the original script). Probes:
  A  reproduction per stake (and per liability as a cross-check)
  B  heavy tail: per-stake P&L by place-BSP bucket, ROI without the k worst lays, day / week block bootstrap
     (percentile CI, skew), number of lays lost at BSP >= 100 / 200 and their cost
  C  EXECUTABLE per-stake analogue: a BSP lay must be sized by liability, so size the liability from a pre-off
     quote, L = (place ltp_pre - 1) per unit of intended stake (also capped variants and a LIMIT_ON_CLOSE cap)
  D  price impact per stake: BSP*(1+x), BSP + k ticks
  E  alternative clustering (meeting, week, month) and an alternative half split (period median)
  F  segments and years per stake
  G  dead heats as full losses; missing-BSP decisions settled at the place ltp_pre
Prints summary numbers only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
L12A, L12B = np.datetime64("2019-11-08"), np.datetime64("2020-11-08")
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1000.0, 10.0)]


def out(o):
    print(json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x)), flush=True)


def r6(x):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, 6)


def step_at(p):
    p = np.asarray(p, float)
    s = np.full_like(p, 10.0)
    for lo, hi, inc in TICKS:
        s = np.where((p >= lo - 1e-9) & (p < hi - 1e-9), inc, s)
    return s


def load(d: Path, cc: str, frac: float) -> pd.DataFrame:
    wc = ["market_id", "event_id", "venue", "market_name", "start", "date", "selection_id", "program", "ltp_pre", "bsp",
          "won"]
    pc = ["market_id", "event_id", "start", "n_winners", "selection_id", "ltp_pre", "ltp_t30", "bsp", "won"]
    w = pd.read_parquet(d / f"{cc}_win.parquet", columns=wc).drop_duplicates(["market_id", "selection_id"])
    p = pd.read_parquet(d / f"{cc}_place.parquet", columns=pc).drop_duplicates(["market_id", "selection_id"])
    if frac < 1:
        ev = w["event_id"].drop_duplicates().sample(frac=frac, random_state=3)
        w, p = w[w["event_id"].isin(ev)], p[p["event_id"].isin(ev)]
    wm = w.groupby("market_id", sort=False).agg(event_id=("event_id", "first"), start=("start", "first")).reset_index()
    pm = p.groupby("market_id", sort=False).agg(event_id=("event_id", "first"), start=("start", "first"),
                                                nw=("n_winners", "first")).reset_index()
    wm = wm[~wm.duplicated(["event_id", "start"], keep=False)]
    pm = pm[~pm.duplicated(["event_id", "start"], keep=False)]
    pr = wm.merge(pm, on=["event_id", "start"], suffixes=("_w", "_p"))
    pr = pr[pr["nw"].isin([2, 3])]
    w = w[w["market_id"].isin(pr["market_id_w"])].rename(columns={"ltp_pre": "pre_w", "bsp": "bsp_w", "won": "won_w"})
    p = p[p["market_id"].isin(pr["market_id_p"])].merge(
        pr[["market_id_p", "market_id_w"]].rename(columns={"market_id_p": "pmid"}), left_on="market_id",
        right_on="pmid").drop(columns=["market_id", "pmid", "event_id", "start"])
    p = p.rename(columns={"market_id_w": "market_id", "ltp_pre": "pre_pl", "ltp_t30": "t30_pl", "bsp": "bsp_pl",
                          "won": "placed", "n_winners": "nw"})
    j = w.merge(p, on=["market_id", "selection_id"], how="outer", indicator=True)
    bad = j.loc[j["_merge"] != "both", "market_id"].unique()
    j = j[~j["market_id"].isin(bad)].drop(columns="_merge")
    g = j.groupby("market_id", sort=False)
    j["n"] = g["selection_id"].transform("size")
    j["npl"] = g["placed"].transform("sum")
    j["nwon"] = g["won_w"].transform("sum")
    na = (j["placed"].isna() | j["won_w"].isna()).groupby(j["market_id"]).transform("max").astype(bool)
    j = j[~na & (j["npl"] >= j["nw"]) & (j["nwon"] >= 1) & (j["n"] >= j["nw"] + 1)].copy()
    for c in ("pre_w", "bsp_w", "pre_pl", "t30_pl", "bsp_pl"):
        j.loc[~(j[c] > 1), c] = np.nan
    g = j.groupby("market_id", sort=False)
    prog = pd.to_numeric(j["program"].astype(object), errors="coerce")
    j["_prog"] = prog
    gap = (j.groupby("market_id")["_prog"].transform("count") == j["n"]) & (
        j.groupby("market_id")["_prog"].transform("max") > j["n"])
    for col in ("pre_w", "pre_pl", "t30_pl"):
        j["_i"] = 1.0 / j[col]
        j[col + "_book"] = j.groupby("market_id")["_i"].transform("sum")
        j[col + "_all"] = j.groupby("market_id")[col].transform("count") == j["n"]
    j["pre_pl_book"] = j["pre_pl_book"] / j["nw"]
    j["t30_pl_book"] = j["t30_pl_book"] / j["nw"]
    g6 = gap & ((j["pre_w_book"] < 0.97) | (j["pre_pl_all"] & (j["pre_pl_book"] < 0.97)))
    j = j[~g6].copy()
    harn = j["market_name"].astype(object).fillna("").str.contains(r"\b(?:Pace|Trot)\b", regex=True)
    j["seg"] = np.where(harn, "au_harness", "au_gallops") if cc == "au" else cc
    j["cc"] = cc
    j["f"] = np.where(j["placed"] == 1, j["nw"] / j["npl"], 0.0)
    j["dh_race"] = j["npl"] > j["nw"]
    j["race"] = cc + ":" + j["market_id"].astype(str)
    j["day"] = pd.to_datetime(j["date"]).values.astype("datetime64[D]")
    j["meeting"] = cc + ":" + j["venue"].astype(str) + ":" + j["day"].astype(str)
    keep = ["race", "day", "meeting", "cc", "seg", "nw", "n", "pre_pl", "t30_pl", "bsp_pl", "f", "dh_race",
            "pre_pl_all", "pre_pl_book", "t30_pl_all", "t30_pl_book", "won_w", "pre_w", "bsp_w"]
    return j[keep].reset_index(drop=True)


def lay_net(price, f, c):
    """per unit of backer's stake"""
    gross = (1 - f) - f * (price - 1)
    return np.where(gross > 0, gross * (1 - c), gross)


def cse(r, cl):
    d = pd.Series(r - r.mean()).groupby(np.asarray(cl)).sum().to_numpy()
    G = len(d)
    return float(math.sqrt((d ** 2).sum() * G / (G - 1)) / len(r)) if G > 1 else float("nan")


def st(r, race, day):
    n = len(r)
    if n < 3:
        return {"bets": int(n), "roi": r6(r.mean()) if n else None, "se": None}
    se = max(float(r.std(ddof=1) / math.sqrt(n)), cse(r, race), cse(r, day))
    return {"bets": int(n), "roi": r6(r.mean()), "se": r6(se)}


def evaluate(label, g, ret_by_c: dict, denom=None, split=None, extra=None, quiet=False):
    """ret_by_c: commission -> per-bet return (already NaN-filtered rows of g). denom: per-bet exposure for a
    ratio estimator (ROI = sum ret / sum denom); default 1 per bet. SE for the ratio by linearisation."""
    r8 = ret_by_c[0.08]
    n = len(r8)
    if n < 3:
        return {}
    if denom is None:
        denom = np.ones(n)
    race, day = g["race"].to_numpy(), g["day"].to_numpy()

    def ratio(mask):
        if mask.sum() < 3:
            return {"bets": int(mask.sum()), "roi": None, "se": None}
        rr, dd = r8[mask], denom[mask]
        roi = rr.sum() / dd.sum()
        lin = (rr - roi * dd) / dd.mean()   # linearised per-bet influence
        s = st(lin, race[mask], day[mask])
        return {"bets": int(mask.sum()), "roi": r6(roi), "se": s["se"]}

    allm = np.ones(n, bool)
    rd = pd.Series(day).groupby(race).first().to_numpy()
    mid = np.sort(rd)[len(rd) // 2] if split is None else split
    fh = day < mid
    l12 = (day >= L12A) & (day <= L12B)
    a, h1, h2, lt = ratio(allm), ratio(fh), ratio(~fh), ratio(l12)
    z = a["roi"] / a["se"] if a["se"] else None
    g4 = bool(h1["roi"] is not None and h2["roi"] is not None and h1["se"] and h2["se"]
              and h1["roi"] > 2 * h1["se"] and h2["roi"] > 2 * h2["se"] and (lt["roi"] or -1) > 0)
    res = {"section": label, "bets": n, "roi_8pct": a["roi"], "se": a["se"], "z": r6(z),
           "roi_0_5_10": [r6(ret_by_c[c].sum() / denom.sum()) for c in (0.0, 0.05, 0.10)],
           "halves": [h1["roi"], h2["roi"]], "halves_se": [h1["se"], h2["se"]], "split": str(mid)[:10],
           "halves_z": [r6(h1["roi"] / h1["se"]) if h1["se"] else None, r6(h2["roi"] / h2["se"]) if h2["se"] else None],
           "last12": lt["roi"], "last12_se": lt["se"], "last12_bets": lt["bets"], "passes_G4": g4}
    if extra:
        res.update(extra)
    if not quiet:
        out(res)
    return res


def rets(price, f, scale=None):
    sc = 1.0 if scale is None else scale
    return {c: lay_net(price, f, c) * sc for c in (0.0, 0.05, 0.08, 0.10)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data/raw/betfair/split"))
    ap.add_argument("--frac", type=float, default=1.0)
    a = ap.parse_args()
    d = Path(a.dir)
    df = pd.concat([load(d, cc, a.frac) for cc in ("au", "gb", "ie")], ignore_index=True)
    out({"section": "universe_all", "races": int(df["race"].nunique()), "runners": int(len(df)),
         "by_cc": df.drop_duplicates("race")["cc"].value_counts().to_dict()})
    u = df[df["pre_pl_all"] & df["pre_pl_book"].between(0.97, 1.03)]
    sel = u[u["pre_pl"] >= 20].copy()
    g = sel[sel["bsp_pl"].notna()].reset_index(drop=True)
    P, F, PRE = g["bsp_pl"].to_numpy(), g["f"].to_numpy(), g["pre_pl"].to_numpy()
    out({"section": "decisions", "n": int(len(sel)), "with_bsp": int(len(g)), "races_universe": int(u["race"].nunique()),
         "median_bsp": r6(np.median(P)), "mean_bsp": r6(P.mean()), "p99_bsp": r6(np.quantile(P, 0.99)),
         "max_bsp": r6(P.max()), "lost_lays": int((F > 0).sum()), "strike_over_implied": r6(F.mean() / np.mean(1 / P))})

    # ---------------- A reproduction
    A = evaluate("A_per_stake_reproduction", g, rets(P, F))
    evaluate("A_per_liability_crosscheck", g, rets(P, F, 1 / (P - 1)))

    # ---------------- B heavy tail
    r8 = lay_net(P, F, 0.08)
    n = len(r8)
    tab = {}
    for lo, hi in ((20, 30), (30, 50), (50, 100), (100, 200), (200, 1001)):
        m = (P >= lo) & (P < hi)
        if m.sum() > 2:
            tab[f"bsp[{lo},{hi})"] = {"bets": int(m.sum()), "lost": int((F[m] > 0).sum()),
                                      "strike_over_implied": r6(F[m].mean() / np.mean(1 / P[m])),
                                      "roi_stake": r6(r8[m].mean()),
                                      "se_stake": r6(max(cse(r8[m], g['day'].to_numpy()[m]), r8[m].std(ddof=1) / math.sqrt(m.sum()))),
                                      "contribution_to_pooled_roi": r6(r8[m].sum() / n),
                                      "expected_losses_if_calibrated": r6(np.sum(1 / P[m]))}
    for lo in (20, 30, 50, 100):
        m = (PRE >= lo) & (PRE < {20: 30, 30: 50, 50: 100, 100: 1001}[lo])
        tab[f"pre_band_from_{lo}"] = {"bets": int(m.sum()), "lost": int((F[m] > 0).sum()), "roi_stake": r6(r8[m].mean()),
                                      "contribution_to_pooled_roi": r6(r8[m].sum() / n)}
    out({"section": "B_by_bucket_per_stake", "table": tab})
    srt = np.sort(r8)
    worst = {f"excl_worst_{k}": r6(srt[k:].mean()) for k in (1, 3, 5, 10, 20, 50)}
    loss = r8 < 0
    out({"section": "B_tail", "total_profit_units": r6(r8.sum()), "gross_win_units": r6(r8[~loss].sum()),
         "gross_loss_units": r6(r8[loss].sum()), "n_losses": int(loss.sum()),
         "worst_10_losses_share_of_loss_units": r6(srt[:10].sum() / r8[loss].sum()),
         "worst_10_cost_in_roi": r6(srt[:10].sum() / n),
         "single_loss_at_bsp_500_cost_in_roi": r6(-499 / n),
         "lost_at_bsp_ge_100": int(((P >= 100) & (F > 0)).sum()),
         "lost_at_bsp_ge_200": int(((P >= 200) & (F > 0)).sum()),
         "expected_lost_at_bsp_ge_200_if_calibrated": r6(np.sum(1 / P[P >= 200])),
         "skew_per_bet": r6(pd.Series(r8).skew()), **worst})
    rng = np.random.default_rng(17)
    for lab, key in (("day", g["day"].astype(str).to_numpy()),
                     ("week", pd.to_datetime(g["day"]).dt.to_period("W").astype(str).to_numpy())):
        codes, uq = pd.factorize(key)
        sums = np.bincount(codes, weights=r8)
        cnts = np.bincount(codes).astype(float)
        B = 4000
        bs = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, len(uq), len(uq))
            bs[b] = sums[idx].sum() / cnts[idx].sum()
        out({"section": f"B_block_bootstrap_{lab}", "blocks": int(len(uq)), "se": r6(bs.std(ddof=1)),
             "ci95": [r6(np.quantile(bs, 0.025)), r6(np.quantile(bs, 0.975))],
             "p_roi_le_0": r6(np.mean(bs <= 0)), "p_roi_le_0.05": r6(np.mean(bs <= 0.05)),
             "boot_skew": r6(pd.Series(bs).skew())})

    # ---------------- C executable per-stake analogues (liability sized from the pre-off place quote)
    # liability L = (ltp_pre - 1) per unit of intended stake; matched backer's stake = L / (BSP - 1)
    rl = {c: lay_net(P, F, c) / (P - 1) for c in (0.0, 0.05, 0.08, 0.10)}
    w = PRE - 1
    exe = {c: rl[c] * w for c in rl}
    evaluate("C1_exec_liab_ltp_pre_minus_1_per_intended_stake", g, exe,
             extra={"mean_matched_stake_per_intended": r6(np.mean(w / (P - 1))),
                    "corr_log_bsp_over_pre_vs_placed": r6(np.corrcoef(np.log(P / PRE), F)[0, 1])})
    evaluate("C2_exec_liab_ltp_pre_minus_1_per_matched_stake", g, exe, denom=w / (P - 1))
    for cap in (50, 100):
        wc = np.minimum(PRE, cap) - 1
        evaluate(f"C3_exec_liab_min(ltp_pre,{cap})-1_per_intended_stake", g, {c: rl[c] * wc for c in rl},
                 denom=wc / (cap - 1) * 0 + 1)
        m = P <= cap
        evaluate(f"C4_exec_loc_cap_{cap}_liab_ltp_pre_minus_1", g[m], {c: exe[c][m] for c in exe})
        evaluate(f"C5_per_stake_at_bsp_loc_cap_{cap}_not_exec_sizing", g[m], rets(P[m], F[m]))
    # per-stake sizing using the t30 place quote where it exists (else ltp_pre)
    q = np.where(np.isfinite(g["t30_pl"].to_numpy()), g["t30_pl"].to_numpy(), PRE)
    evaluate("C6_exec_liab_t30_or_pre_minus_1", g, {c: rl[c] * (q - 1) for c in rl})
    # steamers vs drifters: how per-stake ROI depends on the BSP move (sizing on BSP hides this)
    lr = np.log(P / PRE)
    tb = {}
    for lab, m in (("bsp_shorter_than_pre", lr < -1e-9), ("bsp_equal", np.abs(lr) <= 1e-9), ("bsp_longer", lr > 1e-9)):
        tb[lab] = {"bets": int(m.sum()), "strike_over_implied_bsp": r6(F[m].mean() / np.mean(1 / P[m])),
                   "strike_over_implied_pre": r6(F[m].mean() / np.mean(1 / PRE[m])),
                   "roi_stake_bsp": r6(r8[m].mean()), "roi_exec": r6(exe[0.08][m].mean())}
    out({"section": "C_move_split", "table": tb})

    # ---------------- D price impact per stake
    imp = {}
    day = g["day"].to_numpy()
    for x in (0.01, 0.02, 0.05, 0.10):
        rr = lay_net(P * (1 + x), F, 0.08)
        imp[f"bsp_x{1 + x:.2f}"] = [r6(rr.mean()), r6(rr.mean() / max(cse(rr, day), rr.std(ddof=1) / math.sqrt(n)))]
    for k in (1, 2):
        Pk = P.copy()
        for _ in range(k):
            Pk = np.minimum(Pk + step_at(Pk), 1000.0)
        rr = lay_net(Pk, F, 0.08)
        imp[f"bsp_plus_{k}_tick"] = [r6(rr.mean()), r6(rr.mean() / max(cse(rr, day), rr.std(ddof=1) / math.sqrt(n)))]
        rr = (lay_net(Pk, F, 0.08) / (Pk - 1)) * w
        imp[f"exec_C1_plus_{k}_tick"] = [r6(rr.mean()), r6(rr.mean() / max(cse(rr, day), rr.std(ddof=1) / math.sqrt(n)))]
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mm = (lo + hi) / 2
        lo, hi = (mm, hi) if lay_net(P * (1 + mm), F, 0.08).mean() > 0 else (lo, mm)
    imp["breakeven_mult_impact_per_stake"] = r6(lo)
    out({"section": "D_price_impact_roi_z", **imp})

    # ---------------- E alternative clustering and split
    wk = pd.to_datetime(g["day"]).dt.to_period("W").astype(str).to_numpy()
    mo = pd.to_datetime(g["day"]).dt.to_period("M").astype(str).to_numpy()
    out({"section": "E_alt_se_per_stake", "se_iid": r6(r8.std(ddof=1) / math.sqrt(n)),
         "se_race": r6(cse(r8, g["race"].to_numpy())), "se_day": r6(cse(r8, day)),
         "se_meeting": r6(cse(r8, g["meeting"].to_numpy())), "se_week": r6(cse(r8, wk)), "se_month": r6(cse(r8, mo))})
    evaluate("E_per_stake_split_period_median_2018-12-09", g, rets(P, F), split=np.datetime64("2018-12-09"))
    evaluate("E_exec_C1_split_period_median_2018-12-09", g, exe, split=np.datetime64("2018-12-09"))

    # ---------------- F segments / years per stake and exec
    for seg in ("au_gallops", "au_harness", "gb", "ie"):
        m = (g["seg"] == seg).to_numpy()
        evaluate(f"F_seg_{seg}_per_stake", g[m], rets(P[m], F[m]))
        evaluate(f"F_seg_{seg}_exec_C1", g[m], {c: exe[c][m] for c in exe})
    yr = pd.to_datetime(g["day"]).dt.year.to_numpy()
    hy = np.char.add(yr.astype(str), np.where(pd.to_datetime(g["day"]).dt.month.to_numpy() > 6, "H2", "H1"))
    t1 = {}
    for k in sorted(np.unique(hy)):
        m = hy == k
        rr, re = r8[m], exe[0.08][m]
        t1[k] = {"bets": int(m.sum()), "roi_stake": r6(rr.mean()),
                 "se_stake": r6(max(cse(rr, day[m]), rr.std(ddof=1) / math.sqrt(m.sum()))),
                 "roi_exec": r6(re.mean()), "se_exec": r6(max(cse(re, day[m]), re.std(ddof=1) / math.sqrt(m.sum())))}
    out({"section": "F_half_year", "table": t1, "n_neg_stake": int(sum(v["roi_stake"] < 0 for v in t1.values())),
         "n_neg_exec": int(sum(v["roi_exec"] < 0 for v in t1.values()))})

    # ---------------- G dead heats worst case; missing BSP
    F2 = np.where(F > 0, 1.0, 0.0)
    evaluate("G_deadheat_full_loss_per_stake", g, rets(P, F2),
             extra={"dead_heat_lost_lays": int(((F > 0) & (F < 1)).sum())})
    sp = sel.reset_index(drop=True)
    allp = np.where(sp["bsp_pl"].notna(), sp["bsp_pl"], sp["pre_pl"])
    evaluate("G_missing_bsp_at_ltp_pre_per_stake", sp, rets(allp, sp["f"].to_numpy()),
             extra={"missing": int(sp["bsp_pl"].isna().sum()),
                    "placed_rate_missing": r6(sp.loc[sp["bsp_pl"].isna(), "f"].mean()),
                    "placed_rate_with": r6(sp.loc[sp["bsp_pl"].notna(), "f"].mean())})

    # ---------------- H leakage / timing: is the PLACE ltp_pre informative beyond the BSP because it is stale-but-right,
    # or because it carries post-off information? Compare with the t30 quote (surely pre-off) and the WIN market.
    def move_table(df_, q, b, y, bins=(-9, -0.4, -0.15, -0.03, 0.03, 0.15, 0.4, 9)):
        lr_ = np.log(df_[q] / df_[b]).to_numpy()
        tb_ = {}
        for lo_, hi_ in zip(bins[:-1], bins[1:]):
            m_ = (lr_ >= lo_) & (lr_ < hi_)
            if m_.sum() > 50:
                yy, bb = df_[y].to_numpy()[m_], df_[b].to_numpy()[m_]
                tb_[f"log({q}/{b})[{lo_},{hi_})"] = [int(m_.sum()), r6(yy.mean() / np.mean(1 / bb)),
                                                     r6(math.sqrt(yy.mean() * len(yy)) / len(yy) / np.mean(1 / bb))]
        return tb_
    v = u[u["bsp_pl"].notna() & (u["bsp_pl"] >= 20) & u["pre_pl"].notna()].copy()
    v["y"] = v["f"]
    out({"section": "H1_place_bsp_ge20_strike_over_implied_bsp_by_pre_move", "n_strike/implied_se": move_table(v, "pre_pl", "bsp_pl", "y")})
    vt = v[v["t30_pl"].notna()]
    out({"section": "H1_place_bsp_ge20_by_t30_move", "n_strike/implied_se": move_table(vt, "t30_pl", "bsp_pl", "y"),
         "n_with_t30": int(len(vt))})
    out({"section": "H1_place_bsp_ge20_by_pre_move_same_t30_subset", "n_strike/implied_se": move_table(vt, "pre_pl", "bsp_pl", "y")})
    vw = u[(u["seg"] != "au_harness") & u["bsp_w"].notna() & (u["bsp_w"] >= 30) & u["pre_w"].notna()].drop_duplicates(
        ["race", "pre_w", "bsp_w", "won_w"]).copy()
    vw["y"] = vw["won_w"].astype(float)
    out({"section": "H2_WIN_bsp_ge30_strike_over_implied_by_win_pre_move_(win_pre_cut_at_inplay)",
         "n_strike/implied_se": move_table(vw, "pre_w", "bsp_w", "y")})
    # winners of the race whose PLACE ltp_pre is an 'in-play-like' price far shorter than their place BSP
    ww = u[(u["won_w"] == 1) & u["bsp_pl"].notna() & u["pre_pl"].notna()]
    lw = u[(u["won_w"] == 0) & (u["f"] == 0) & u["bsp_pl"].notna() & u["pre_pl"].notna()]
    out({"section": "H3_extreme_pre_vs_bsp", "winners_pre_lt_third_of_bsp": r6(np.mean(ww["pre_pl"] < ww["bsp_pl"] / 3)),
         "unplaced_pre_lt_third_of_bsp": r6(np.mean(lw["pre_pl"] < lw["bsp_pl"] / 3)),
         "winners_pre_gt_3x_bsp": r6(np.mean(ww["pre_pl"] > 3 * ww["bsp_pl"])),
         "unplaced_pre_gt_3x_bsp": r6(np.mean(lw["pre_pl"] > 3 * lw["bsp_pl"])),
         "winners_pre_le_1.05_and_bsp_ge_1.5": int(((ww["pre_pl"] <= 1.05) & (ww["bsp_pl"] >= 1.5)).sum()),
         "unplaced_pre_le_1.05_and_bsp_ge_1.5": int(((lw["pre_pl"] <= 1.05) & (lw["bsp_pl"] >= 1.5)).sum())})
    # selection decided at t30 on the place ltp_t30 (cannot contain post-off information)
    for lab, mm in (("H4_R1b_and_t30_ge20", (sel["t30_pl"] >= 20) & sel["bsp_pl"].notna()),):
        gg = sel[mm.to_numpy()]
        Pg, Fg = gg["bsp_pl"].to_numpy(), gg["f"].to_numpy()
        evaluate(lab + "_per_stake", gg, rets(Pg, Fg))
        evaluate(lab + "_exec_liab_t30_minus_1", gg, rets(Pg, Fg, (gg["t30_pl"].to_numpy() - 1) / (Pg - 1)))
    for lab, uu in (("H5_t30_ge20_full_t30_book", df[df["t30_pl_all"] & df["t30_pl_book"].between(0.97, 1.03)]),
                    ("H5_t30_ge20_R1b_universe", u)):
        gg = uu[(uu["t30_pl"] >= 20) & uu["bsp_pl"].notna()]
        Pg, Fg = gg["bsp_pl"].to_numpy(), gg["f"].to_numpy()
        evaluate(lab + "_per_stake", gg, rets(Pg, Fg), extra={"by_cc": gg["cc"].value_counts().to_dict()})
        evaluate(lab + "_exec_liab_t30_minus_1", gg, rets(Pg, Fg, (gg["t30_pl"].to_numpy() - 1) / (Pg - 1)))
        evaluate(lab + "_per_liability", gg, rets(Pg, Fg, 1 / (Pg - 1)))
    # per-stake with ratio filter (BSP within 2x of ltp_pre)
    rt = P / PRE
    m = (rt <= 2) & (rt >= 0.5)
    evaluate("H6_per_stake_bsp_within_2x_pre", g[m], rets(P[m], F[m]), extra={"dropped": int((~m).sum())})
    evaluate("H6_exec_C1_bsp_within_2x_pre", g[m], {c: exe[c][m] for c in exe})
    # ---------------- I LIMIT_ON_CLOSE lay with limit = place ltp_pre (matched at BSP only if BSP <= ltp_pre); post-hoc
    m = P <= PRE + 1e-9
    evaluate("I_loc_limit_pre_per_stake", g[m], rets(P[m], F[m]), extra={"fill_rate": r6(m.mean())})
    evaluate("I_loc_limit_pre_per_liability", g[m], rets(P[m], F[m], 1 / (P[m] - 1)))
    evaluate("I_loc_limit_pre_exec_liab_pre_minus_1", g[m], {c: exe[c][m] for c in exe})
    out({"section": "done", "claimed_roi_stake": 0.111433, "reproduced": A.get("roi_8pct")})


if __name__ == "__main__":
    main()

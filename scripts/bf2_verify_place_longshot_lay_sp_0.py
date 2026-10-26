"""Adversarial verification of "place_longshot_lay_sp" R1b@20 (round 2).

    python scripts/bf2_verify_place_longshot_lay_sp_0.py [--dir data/raw/betfair/split]

Claim: lay every PLACE runner with PLACE ltp_pre >= 20 at the place BSP, 1 unit of LIABILITY, in AU/GB/IE
2- or 3-place markets paired with their WIN market (identical runner sets, settled, G6-clean) whose PLACE
ltp_pre book / n_winners is in [0.97, 1.03] with every runner priced. Claimed ROI8 +0.334% (SE 0.063).

This file re-implements the universe from the raw parquet WITHOUT importing the original script, then
probes the claim:
  A  reproduction (ROI at 0/5/8/10%, SE = max(iid, race, day), halves at the bets' median date, last 12m)
  B  alternative clustering: meeting-day, ISO week, calendar month; block bootstrap by week
  C  price impact / capacity: settle at BSP*(1+x) and at BSP + k ticks (k = 1, 2); break-even impact
  D  decision timing: select on PLACE ltp_t30 >= 20 (decided 30 minutes out) instead of ltp_pre
  E  missing-BSP decisions settled at the place ltp_pre (worst case if they had been bets)
  F  stability: by country x year, by half-year, by price band of selection, excluding the top venues
  G  BSP vs ltp_pre disagreement: drop bets whose place BSP is > 2x or < 0.5x the place ltp_pre
  H  dead heats: settle dead-heat places as full losses for the layer (worst case)
Prints summary numbers only (never horse names or rows).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LAST12_FROM = np.datetime64("2019-11-08")
LAST12_TO = np.datetime64("2020-11-08")
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1000.0, 10.0)]


def out(o):
    print(json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x)), flush=True)


def r6(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def step_at(p):
    p = np.asarray(p, float)
    s = np.full_like(p, 10.0)
    for lo, hi, inc in TICKS:
        s = np.where((p >= lo - 1e-9) & (p < hi - 1e-9), inc, s)
    return s


def load(d: Path, cc: str) -> pd.DataFrame:
    wc = ["market_id", "event_id", "venue", "market_name", "start", "date", "selection_id", "program", "ltp_pre",
          "ltp_t30", "bsp", "won"]
    pc = ["market_id", "event_id", "start", "n_winners", "selection_id", "ltp_pre", "ltp_t30", "bsp", "won"]
    w = pd.read_parquet(d / f"{cc}_win.parquet", columns=wc).drop_duplicates(["market_id", "selection_id"])
    p = pd.read_parquet(d / f"{cc}_place.parquet", columns=pc).drop_duplicates(["market_id", "selection_id"])
    wm = w.groupby("market_id").agg(event_id=("event_id", "first"), start=("start", "first")).reset_index()
    pm = p.groupby("market_id").agg(event_id=("event_id", "first"), start=("start", "first"),
                                    nw=("n_winners", "first")).reset_index()
    wm = wm[~wm.duplicated(["event_id", "start"], keep=False)]
    pm = pm[~pm.duplicated(["event_id", "start"], keep=False)]
    pr = wm.merge(pm, on=["event_id", "start"], suffixes=("_w", "_p"))
    pr = pr[pr["nw"].isin([2, 3])]
    w = w[w["market_id"].isin(pr["market_id_w"])]
    p = p[p["market_id"].isin(pr["market_id_p"])].merge(
        pr[["market_id_p", "market_id_w"]].rename(columns={"market_id_p": "market_id"}), on="market_id")
    p = p.rename(columns={"market_id": "pmid", "market_id_w": "market_id", "ltp_pre": "pre_pl", "ltp_t30": "t30_pl",
                          "bsp": "bsp_pl", "won": "placed", "n_winners": "nw"})
    w = w.rename(columns={"ltp_pre": "pre_w", "ltp_t30": "t30_w", "bsp": "bsp_w", "won": "won_w"})
    j = w.merge(p.drop(columns=["event_id", "start"]), on=["market_id", "selection_id"], how="outer", indicator=True)
    badm = set(j.loc[j["_merge"] != "both", "market_id"])
    j = j[~j["market_id"].isin(badm)].drop(columns="_merge")
    g = j.groupby("market_id")
    j["n"] = g["selection_id"].transform("size")
    j["npl"] = g["placed"].transform("sum")
    j["nwon"] = g["won_w"].transform("sum")
    j["anyna"] = (j["placed"].isna() | j["won_w"].isna()).groupby(j["market_id"]).transform("max")
    j = j[(~j["anyna"].astype(bool)) & (j["npl"] >= j["nw"]) & (j["nwon"] >= 1) & (j["n"] >= j["nw"] + 1)].copy()
    for c in ("pre_w", "t30_w", "bsp_w", "pre_pl", "t30_pl", "bsp_pl"):
        j.loc[~(j[c] > 1), c] = np.nan
    g = j.groupby("market_id")
    prog = pd.to_numeric(j["program"].astype(object), errors="coerce")
    j["_prog"] = prog
    gap = (g["_prog"].transform("count") == j["n"]) & (g["_prog"].transform("max") > j["n"])
    for col in ("pre_w", "pre_pl", "t30_pl"):
        j["_i"] = 1 / j[col]
        j[col + "_book"] = j.groupby("market_id")["_i"].transform("sum")
        j[col + "_all"] = j.groupby("market_id")[col].transform("count") == j["n"]
    j["pre_pl_book"] /= j["nw"]
    j["t30_pl_book"] /= j["nw"]
    g6 = gap & ((j["pre_w_book"] < 0.97) | (j["pre_pl_all"] & (j["pre_pl_book"] < 0.97)))
    j = j[~g6].copy()
    j["harness"] = j["market_name"].astype(object).fillna("").str.contains(r"\b(?:Pace|Trot)\b", regex=True)
    j["seg"] = np.where(j["harness"], "au_harness", "au_gallops") if cc == "au" else cc
    j["cc"] = cc
    j["f"] = np.where(j["placed"] == 1, j["nw"] / j["npl"], 0.0)
    j["race"] = cc + ":" + j["market_id"].astype(str)
    j["day"] = pd.to_datetime(j["date"]).values.astype("datetime64[D]")
    j["meeting"] = cc + ":" + j["venue"].astype(str) + ":" + j["day"].astype(str)
    return j.drop(columns=["_prog", "_i", "program", "anyna", "market_name"]).reset_index(drop=True)


def lay_liab(price, f, c):
    gross = (1 - f) - f * (price - 1)
    net = np.where(gross > 0, gross * (1 - c), gross)
    return net / (price - 1)


def cse(r, cl):
    d = pd.Series(r - r.mean()).groupby(np.asarray(cl)).sum().to_numpy()
    G = len(d)
    return float(math.sqrt((d ** 2).sum() * G / (G - 1)) / len(r)) if G > 1 else float("nan")


def stats(r, g):
    n = len(r)
    if n < 3:
        return {"bets": n}
    se = max(float(r.std(ddof=1) / math.sqrt(n)), cse(r, g["race"].to_numpy()), cse(r, g["day"].to_numpy()))
    return {"bets": int(n), "roi": r6(r.mean()), "se": r6(se), "z": r6(r.mean() / se) if se > 0 else None}


def evaluate(g: pd.DataFrame, price: np.ndarray, f: np.ndarray, label: str, extra=None) -> dict:
    ok = np.isfinite(price)
    g, price, f = g[ok], price[ok], f[ok]
    if len(g) < 3:
        out({"section": label, "bets": int(len(g))})
        return {}
    r = lay_liab(price, f, 0.08)
    day = g["day"].to_numpy()
    rd = pd.Series(day).groupby(g["race"].to_numpy()).first().to_numpy()
    mid = np.sort(rd)[len(rd) // 2]
    fh = day < mid
    l12 = (day >= LAST12_FROM) & (day <= LAST12_TO)
    s = stats(r, g)
    h1, h2, lt = stats(r[fh], g[fh]), stats(r[~fh], g[~fh]), stats(r[l12], g[l12])
    res = {"section": label, **s,
           "roi_0_5_10": [r6(lay_liab(price, f, c).mean()) for c in (0.0, 0.05, 0.10)],
           "halves": [h1.get("roi"), h2.get("roi")], "halves_se": [h1.get("se"), h2.get("se")],
           "split": str(mid)[:10], "last12": lt.get("roi"), "last12_se": lt.get("se"),
           "strike_over_implied": r6(f.mean() / np.mean(1 / price)), "median_price": r6(np.median(price)),
           "g4": bool(h1.get("roi") is not None and h2.get("roi") is not None and h1["roi"] > 2 * h1["se"]
                      and h2["roi"] > 2 * h2["se"] and (lt.get("roi") or -1) > 0)}
    if extra:
        res.update(extra)
    out(res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data/raw/betfair/split"))
    a = ap.parse_args()
    d = Path(a.dir)
    df = pd.concat([load(d, cc) for cc in ("au", "gb", "ie")], ignore_index=True)
    out({"section": "universe_all", "races": int(df["race"].nunique()), "runners": int(len(df))})
    u = df[df["pre_pl_all"] & df["pre_pl_book"].between(0.97, 1.03)].copy()
    out({"section": "universe_R1b", "races": int(u["race"].nunique()),
         "by_seg": u.drop_duplicates("race")["seg"].value_counts().to_dict()})
    sel = u[u["pre_pl"] >= 20].copy()
    out({"section": "decisions", "n": int(len(sel)), "with_bsp": int(sel["bsp_pl"].notna().sum())})
    g = sel[sel["bsp_pl"].notna()].copy()
    P, F = g["bsp_pl"].to_numpy(), g["f"].to_numpy()

    # A reproduction
    evaluate(g, P, F, "A_reproduction")
    for seg in ("au_gallops", "au_harness", "gb", "ie"):
        m = (g["seg"] == seg).to_numpy()
        evaluate(g[m], P[m], F[m], f"A_seg_{seg}")

    # B alternative clustering and a weekly block bootstrap
    r = lay_liab(P, F, 0.08)
    wk = pd.to_datetime(g["day"]).dt.to_period("W").astype(str).to_numpy()
    mo = pd.to_datetime(g["day"]).dt.to_period("M").astype(str).to_numpy()
    rng = np.random.default_rng(11)
    wcodes, wuniq = pd.factorize(wk)
    sums = np.bincount(wcodes, weights=r)
    cnts = np.bincount(wcodes).astype(float)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(wuniq), len(wuniq))
        boots.append(sums[idx].sum() / cnts[idx].sum())
    out({"section": "B_alt_se", "roi": r6(r.mean()), "se_iid": r6(r.std(ddof=1) / math.sqrt(len(r))),
         "se_race": r6(cse(r, g["race"].to_numpy())), "se_day": r6(cse(r, g["day"].to_numpy())),
         "se_meeting": r6(cse(r, g["meeting"].to_numpy())), "se_week": r6(cse(r, wk)), "se_month": r6(cse(r, mo)),
         "se_boot_week": r6(np.std(boots, ddof=1)), "boot_p_le_0": r6(np.mean(np.array(boots) <= 0))})

    # C price impact / capacity
    imp = {}
    for x in (0.005, 0.01, 0.02, 0.03, 0.05):
        rr = lay_liab(P * (1 + x), F, 0.08)
        imp[f"bsp_x{1 + x:.3f}"] = [r6(rr.mean()), r6(rr.mean() / cse(rr, g["day"].to_numpy()))]
    for k in (1, 2):
        Pk = P.copy()
        for _ in range(k):
            Pk = np.minimum(Pk + step_at(Pk), 1000.0)
        rr = lay_liab(Pk, F, 0.08)
        imp[f"bsp_plus_{k}_tick"] = [r6(rr.mean()), r6(rr.mean() / cse(rr, g["day"].to_numpy()))]
    lo, hi = 0.0, 0.5
    for _ in range(40):
        m_ = (lo + hi) / 2
        if lay_liab(P * (1 + m_), F, 0.08).mean() > 0:
            lo = m_
        else:
            hi = m_
    imp["breakeven_multiplicative_impact"] = r6(lo)
    out({"section": "C_price_impact_roi_z", **imp})
    # settle at the place ltp_pre (an exchange lay at the selection quote; capacity not modelled)
    pre = g["pre_pl"].to_numpy()
    evaluate(g, pre, F, "C_settle_at_place_ltp_pre_descriptive")
    evaluate(g, np.minimum(P, pre), F, "C_settle_at_min_bsp_ltp_pre_descriptive")
    evaluate(g, np.maximum(P, pre), F, "C_settle_at_max_bsp_ltp_pre_adversarial")

    # D decision timing: select at t30 on the PLACE ltp_t30 (any runner priced; and fully priced book)
    for lab, mask in (
            ("D_t30_any", df["t30_pl"] >= 20),
            ("D_t30_full_book", (df["t30_pl"] >= 20) & df["t30_pl_all"] & df["t30_pl_book"].between(0.97, 1.03)),
            ("D_t30_and_pre_in_R1b", None)):
        if mask is None:
            gg = g[g["t30_pl"] >= 20]
        else:
            gg = df[mask.to_numpy() & df["bsp_pl"].notna().to_numpy()]
        evaluate(gg, gg["bsp_pl"].to_numpy(), gg["f"].to_numpy(), lab,
                 {"by_cc": gg["cc"].value_counts().to_dict()})
    # D2 R1b runners WITHOUT a place t30 price (selection effect)
    gg = g[g["t30_pl"].isna()]
    evaluate(gg, gg["bsp_pl"].to_numpy(), gg["f"].to_numpy(), "D_R1b_no_place_t30")

    # E missing-BSP decisions: settle at the place ltp_pre as if they had been bets
    allp = np.where(sel["bsp_pl"].notna(), sel["bsp_pl"], sel["pre_pl"])
    evaluate(sel, allp, sel["f"].to_numpy(), "E_missing_bsp_at_ltp_pre")

    # F stability
    tab = {}
    for (cc, y), gy in g.groupby([g["cc"], pd.to_datetime(g["day"]).dt.year]):
        rr = lay_liab(gy["bsp_pl"].to_numpy(), gy["f"].to_numpy(), 0.08)
        tab[f"{cc}|{y}"] = [int(len(gy)), r6(rr.mean()), r6(cse(rr, gy["day"].to_numpy()))]
    out({"section": "F_cc_year_bets_roi_se", "table": tab})
    hy = pd.to_datetime(g["day"]).dt.year.astype(str) + "H" + (pd.to_datetime(g["day"]).dt.month > 6).map(
        {False: "1", True: "2"})
    tab = {}
    for k, idx in g.groupby(hy.to_numpy()).groups.items():
        gy = g.loc[idx]
        rr = lay_liab(gy["bsp_pl"].to_numpy(), gy["f"].to_numpy(), 0.08)
        tab[k] = [int(len(gy)), r6(rr.mean()), r6(cse(rr, gy["day"].to_numpy()))]
    out({"section": "F_half_year_bets_roi_se", "table": tab, "n_negative": int(sum(v[1] < 0 for v in tab.values()))})
    for lo_, hi_ in ((20, 30), (30, 50), (50, 100), (100, 1001)):
        m = ((g["pre_pl"] >= lo_) & (g["pre_pl"] < hi_)).to_numpy()
        evaluate(g[m], P[m], F[m], f"F_pre_band_{lo_}_{hi_}")
    # venue concentration: profit share of the top venues and ROI without the top 5 / 10 venues by profit
    vp = pd.Series(r).groupby(g["cc"].to_numpy() + ":" + g["venue"].astype(str).to_numpy()).sum().sort_values(
        ascending=False)
    vkey = g["cc"].to_numpy() + ":" + g["venue"].astype(str).to_numpy()
    res = {"venues": int(len(vp)), "total_profit": r6(vp.sum()), "top5_profit": r6(vp.iloc[:5].sum()),
           "top10_profit": r6(vp.iloc[:10].sum()), "share_venues_positive": r6((vp > 0).mean())}
    for k in (5, 10, 20):
        m = ~np.isin(vkey, vp.index[:k])
        res[f"roi_excl_top{k}"] = r6(r[m].mean())
        res[f"z_excl_top{k}"] = r6(r[m].mean() / cse(r[m], g["day"].to_numpy()[m]))
    out({"section": "F_venue_concentration", **res})

    # G BSP far from ltp_pre
    ratio = P / pre
    m = (ratio <= 2) & (ratio >= 0.5)
    evaluate(g[m], P[m], F[m], "G_bsp_within_2x_of_ltp_pre", {"dropped": int((~m).sum())})
    m = ratio <= 1.0
    evaluate(g[m], P[m], F[m], "G_bsp_le_ltp_pre_descriptive")
    m = ratio > 1.0
    evaluate(g[m], P[m], F[m], "G_bsp_gt_ltp_pre_descriptive")

    # H dead heats as full losses for the layer
    F2 = np.where(F > 0, 1.0, 0.0)
    evaluate(g, P, F2, "H_deadheat_full_loss", {"dead_heat_bets": int(((F > 0) & (F < 1)).sum())})

    # I simple per-race netting with all lays in the race (commission on the race net)
    gross = ((1 - F) - F * (P - 1)) / (P - 1)
    net = pd.Series(gross).groupby(g["race"].to_numpy()).sum().to_numpy()
    net = np.where(net > 0, net * 0.92, net)
    out({"section": "I_market_netted_roi8", "roi": r6(net.sum() / len(P))})


if __name__ == "__main__":
    main()

"""Experiment "t30_early_price" (round 2): act 30 minutes out -- early prices and
closing-line value (CLV) in Betfair WIN markets.

    python scripts/bf2_t30_early_price.py [--dir data/raw/betfair/split] [--frac 0.2] [--json out.json]

Universe: WIN markets of GB, IE (t30 coverage ~99.8%) and AU (only runners with
an ltp_t30; the shares are reported). Deduplicated on (market_id, selection_id);
prices <= 1 set to NaN; settled markets only (won known for every runner, >= 1
winner, >= 3 runners).
  G6       drop markets with a saddlecloth gap AND an ltp_pre book (sum of 1/ltp_pre
           over priced runners) < 0.97.
  G6-t30   (the same idea for an entry struck at t30; primary, reported with and
           without) drop markets with a saddlecloth gap AND a t30 book < 0.97 among
           fully t30-priced markets: a runner was removed after t30, so a bet matched
           at t30 would have been settled with a reduction factor (and unmatched
           orders lapsed), which the archive cannot reproduce (removed runners are
           not in the data).

Rules (decided at t30 with ltp_t30 only, G1):
  (a) back the t30 favourite, 2nd and 3rd favourite (rank by ltp_t30, ties by
      selection_id; markets where every runner has an ltp_t30).
  (b) lay runners with ltp_t30 >= 50, in [30, 50) and in [20, 30).
  (c) back by t30 band [1.01,2) [2,3) [3,5) [5,8) [8,13) [13,20) [20,30) [30,50) [50,+)
      and lay by t30 band [1.01,2) [2,3) [3,5) [5,8) [8,13) [13,20) (the lay bands
      >= 20 are rule (b)).
Settlement (G2):
  limit_then_bsp  limit order at ltp_t30 one tick worse (back: one tick SHORTER, lay:
                  one tick LONGER), filled at the limit iff the later ltp_pre or the
                  bsp reaches it, otherwise settled at BSP.
  bsp             plain BSP.
  Lays per unit of backer's stake and per unit of liability (return / (price - 1)).
Return per unit stake: back net = f(P-1) - (1-f), f = dead_heat_share; lay = minus
that; commission on positive net winnings (per bet) at 0/5/8/10% (8% primary, G3).
Statistics: SE = max(iid, clustered by market, clustered by race day); halves split
at the segment's median race date (races with >= 1 t30-priced runner); last 12
months 2019-11-08..2020-11-08 (G4). G4 also reported at 5% (usual GB/IE rate).
CLV = entry / bsp (> 1 good for a back, < 1 good for a lay).
CLV deciles: every t30-priced runner backed (laid) at the G2 entry and at the paper
t30 quote; bets bucketed by decile of log(entry/bsp); realised ROI vs the ROI
expected if the BSP-implied probability (1/bsp normalised in the market) is true.
Probability: walk-forward (expanding window, refit monthly, one parameter, no
regularisation) power recalibration p ~ q_t30^alpha of the Shin t30 probabilities,
log-loss gain vs the t30 market at the SAME timestamp; plus descriptive log loss
of the t30, ltp_pre and BSP markets (not a same-timestamp comparison).
Prints JSON summary lines only (never horse names or rows).
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
from ponies.odds import shin_probabilities  # noqa: E402,F401  (vectorised copy below, checked in main)

COLS = ["market_id", "market_name", "date", "selection_id", "program", "ltp_pre", "ltp_t30", "bsp", "won",
        "dead_heat_share"]
COUNTRIES = ["gb", "ie", "au"]
COMMS = [0.0, 0.05, 0.08, 0.10]
PRIMARY_C = 0.08
LAST12_FROM = pd.Timestamp("2019-11-08")
Z80 = 0.8416
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1000.0, 10.0)]
BACK_BANDS = [(1.01, 2), (2, 3), (3, 5), (5, 8), (8, 13), (13, 20), (20, 30), (30, 50), (50, 1001)]
LAY_BANDS = [(1.01, 2), (2, 3), (3, 5), (5, 8), (8, 13), (13, 20)]
LAY_LONG = [(20, 30), (30, 50), (50, 1001)]


def out(obj) -> None:
    print(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)), flush=True)


def r6(x):
    return None if x is None or not np.isfinite(float(x)) else round(float(x), 6)


def r4(x):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, 4)


# ---------------------------------------------------------------------------
# tick ladder (as scripts/bf2_aligned_win_to_place.py)
# ---------------------------------------------------------------------------
def tick_shorter(price) -> np.ndarray:
    p = np.round(np.asarray(price, dtype=float), 2)
    step = np.full_like(p, 10.0)
    for lo, hi, inc in TICKS:
        step = np.where((p > lo + 1e-9) & (p <= hi + 1e-9), inc, step)
    return np.maximum(np.round(p - step, 2), 1.01)


def tick_longer(price) -> np.ndarray:
    p = np.round(np.asarray(price, dtype=float), 2)
    step = np.full_like(p, 10.0)
    for lo, hi, inc in reversed(TICKS):
        step = np.where((p >= lo - 1e-9) & (p < hi - 1e-9), inc, step)
    return np.minimum(np.round(p + step, 2), 1000.0)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_country(code: str, d: Path, frac: float, seed: int) -> tuple[pd.DataFrame, dict]:
    w = pd.read_parquet(d / f"{code}_win.parquet", columns=COLS).drop_duplicates(["market_id", "selection_id"])
    if frac < 1:
        mk = w["market_id"].drop_duplicates()
        keep = set(mk.sample(frac=frac, random_state=seed))
        w = w[w["market_id"].isin(keep)]
    w = w.copy()
    for c in ("market_id", "selection_id", "market_name", "program"):
        w[c] = w[c].astype(object)
    for c in ("ltp_pre", "ltp_t30", "bsp"):
        w.loc[~(w[c] > 1), c] = np.nan
    w["date"] = pd.to_datetime(w["date"])
    w["harness"] = w["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b", regex=True)
    w["f"] = np.where(w["won"] == 1, w["dead_heat_share"].fillna(1.0), 0.0).astype(float)
    w["pn"] = pd.to_numeric(w["program"], errors="coerce")
    w["i_pre"] = 1 / w["ltp_pre"]
    w["i_30"] = 1 / w["ltp_t30"]
    w["i_bsp"] = 1 / w["bsp"]
    w["won_na"] = w["won"].isna()
    g = w.groupby("market_id", sort=False)
    m = pd.DataFrame({"n": g.size(), "won_na": g["won_na"].sum(), "won_sum": g["won"].sum(),
                      "pre_n": g["ltp_pre"].count(), "pre_book": g["i_pre"].sum(),
                      "t30_n": g["ltp_t30"].count(), "t30_book": g["i_30"].sum(),
                      "bsp_n": g["bsp"].count(), "bsp_book": g["i_bsp"].sum(),
                      "pn_cnt": g["pn"].count(), "pn_max": g["pn"].max(), "harness": g["harness"].first()})
    m["settled"] = (m["won_na"] == 0) & (m["won_sum"] >= 1) & (m["n"] >= 3)
    m["gap"] = (m["pn_cnt"] == m["n"]) & (m["pn_max"] > m["n"])
    m["g6"] = m["gap"] & (m["pre_book"] < 0.97)
    m["all_t30"] = m["t30_n"] == m["n"]
    m["g6_t30"] = m["gap"] & m["all_t30"] & (m["t30_book"] < 0.97) & ~m["g6"]
    m["all_bsp"] = m["bsp_n"] == m["n"]
    st = {"markets": int(len(m)), "runners": int(m["n"].sum()), "settled_markets": int(m["settled"].sum())}
    s = m[m["settled"]]
    st.update({"g6_dropped": int(s["g6"].sum()), "g6_t30_flagged": int(s["g6_t30"].sum()),
               "markets_any_t30": int((s["t30_n"] > 0).sum()), "markets_all_t30": int(s["all_t30"].sum()),
               "share_markets_any_t30": r4((s["t30_n"] > 0).mean()), "share_markets_all_t30": r4(s["all_t30"].mean()),
               "share_runners_t30": r4(s["t30_n"].sum() / s["n"].sum()),
               "t30_book_all_t30_q05_q50_q95": [r4(x) for x in s.loc[s["all_t30"], "t30_book"].quantile([.05, .5, .95])],
               "pre_book_q05_q50_q95": [r4(x) for x in s.loc[s["pre_n"] == s["n"], "pre_book"].quantile([.05, .5, .95])]})
    if code == "au":
        for h, nm in ((True, "harness"), (False, "gallops")):
            sh = s[s["harness"] == h]
            st[f"{nm}_markets"] = int(len(sh))
            st[f"{nm}_share_runners_t30"] = r4(sh["t30_n"].sum() / sh["n"].sum())
            st[f"{nm}_share_markets_all_t30"] = r4(sh["all_t30"].mean())
    keep_m = m.index[m["settled"] & ~m["g6"]]
    w = w[w["market_id"].isin(keep_m)].copy()
    for c in ("n", "all_t30", "g6_t30", "t30_book", "pre_book", "bsp_book", "all_bsp", "gap"):
        w["m_" + c] = m.loc[w["market_id"], c].to_numpy()
    w["seg"] = code
    w = w.drop(columns=["i_pre", "i_30", "i_bsp", "won_na", "dead_heat_share", "program", "market_name"])
    # t30 rank (fully t30-priced markets only)
    w = w.sort_values(["market_id", "ltp_t30", "selection_id"], kind="mergesort").reset_index(drop=True)
    w["rank30"] = w.groupby("market_id").cumcount() + 1
    w.loc[~w["m_all_t30"], "rank30"] = 0
    return w, st


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def cluster_se(r: np.ndarray, cl: np.ndarray) -> float | None:
    if len(r) < 2:
        return None
    d = pd.Series(r - r.mean()).groupby(cl).sum().to_numpy()
    G = len(d)
    if G < 2:
        return None
    return float(math.sqrt((d ** 2).sum() * G / (G - 1)) / len(r))


def roi_stats(r: np.ndarray, mk: np.ndarray, day: np.ndarray) -> dict:
    n = len(r)
    if n < 2:
        return {"bets": int(n), "roi": r4(r.mean()) if n else None, "se": None}
    se_m = cluster_se(r, mk)
    se_d = cluster_se(r, day)
    se_i = float(r.std(ddof=1) / math.sqrt(n))
    se = max(x for x in (se_m, se_d, se_i) if x is not None)
    return {"bets": int(n), "roi": r4(r.mean()), "se": r4(se), "se_market": r4(se_m), "se_day": r4(se_d),
            "se_iid": r4(se_i)}


def back_net(price, f, c):
    gross = f * (price - 1) - (1 - f)
    return np.where(gross > 0, gross * (1 - c), gross)


def lay_net(price, f, c):
    gross = -(f * (price - 1) - (1 - f))
    return np.where(gross > 0, gross * (1 - c), gross)


def returns(side: str, staking: str, price: np.ndarray, f: np.ndarray, c: float) -> np.ndarray:
    if side == "back":
        return back_net(price, f, c)
    r = lay_net(price, f, c)
    return r / (price - 1) if staking == "liability" else r


def g4(st_h1: dict, st_h2: dict, st_l12: dict) -> bool:
    return bool(st_h1["se"] is not None and st_h2["se"] is not None and st_h1["roi"] is not None
                and st_h2["roi"] is not None and st_h1["roi"] > 2 * st_h1["se"] and st_h2["roi"] > 2 * st_h2["se"]
                and st_l12["roi"] is not None and st_l12["roi"] > 0)


def entry_prices(g: pd.DataFrame, side: str, strict: bool = False):
    t30 = g["ltp_t30"].to_numpy()
    pre = g["ltp_pre"].to_numpy()
    bsp = g["bsp"].to_numpy()
    if side == "back":
        L = tick_shorter(t30)
        via_pre = pre >= L - 1e-9
        via_bsp = bsp >= L - 1e-9
    else:
        L = tick_longer(t30)
        via_pre = pre <= L + 1e-9
        via_bsp = bsp <= L + 1e-9
    filled = via_bsp if strict else (via_pre | via_bsp)
    return L, filled, via_pre, via_bsp, np.where(filled, L, bsp)


def eval_cell(g: pd.DataFrame, side: str, settlement: str, staking: str, mid: pd.Timestamp,
              n_races: int, n_months: int, strict: bool = False, full: bool = True) -> dict:
    L, filled, via_pre, via_bsp, ent_lim = entry_prices(g, side, strict)
    bsp = g["bsp"].to_numpy()
    t30 = g["ltp_t30"].to_numpy()
    price = ent_lim if settlement == "limit_then_bsp" else bsp
    ok = ~np.isnan(price)
    f = g["f"].to_numpy()[ok]
    pr = price[ok]
    mk = g["market_id"].to_numpy()[ok]
    day = g["date"].to_numpy()[ok]
    dts = g["date"].to_numpy()[ok]
    fh = dts < np.datetime64(mid)
    l12 = dts >= np.datetime64(LAST12_FROM)
    rets = {c: returns(side, staking, pr, f, c) for c in COMMS}
    r8 = rets[PRIMARY_C]
    st = roi_stats(r8, mk, day)
    h1 = roi_stats(r8[fh], mk[fh], day[fh])
    h2 = roi_stats(r8[~fh], mk[~fh], day[~fh])
    lt = roi_stats(r8[l12], mk[l12], day[l12])
    r5 = rets[0.05]
    p5 = g4(roi_stats(r5[fh], mk[fh], day[fh]), roi_stats(r5[~fh], mk[~fh], day[~fh]),
            roi_stats(r5[l12], mk[l12], day[l12]))
    res = {"decisions": int(len(g)), "bets": st["bets"], "settled_share": r4(ok.mean()) if len(g) else None,
           "roi_8pct": st["roi"], "se_clustered": st["se"], "se_market": st.get("se_market"),
           "se_day": st.get("se_day"), "z": r4(st["roi"] / st["se"]) if st.get("se") else None,
           "roi_0_5_10pct": [r4(rets[c].mean()) if len(pr) else None for c in (0.0, 0.05, 0.10)],
           "halves_roi_8pct": [h1["roi"], h2["roi"]], "halves_se": [h1["se"], h2["se"]],
           "halves_bets": [h1["bets"], h2["bets"]], "last12m_roi_8pct": lt["roi"], "last12m_bets": lt["bets"],
           "passes_G4_8pct": g4(h1, h2, lt), "passes_G4_5pct": p5}
    if not full:
        return res
    res.update({"mde_80pct_full": r4((2 + Z80) * st["se"]) if st.get("se") else None,
                "mde_80pct_each_half": r4((2 + Z80) * max(h1["se"], h2["se"])) if h1["se"] and h2["se"] else None,
                "fires_per_race": r4(len(g) / n_races) if n_races else None,
                "bets_per_month": r4(len(g) / n_months) if n_months else None,
                "fill_rate": r4(filled.mean()) if len(g) else None,
                "fill_via_ltp_pre": r4(via_pre.mean()) if len(g) else None,
                "fill_via_bsp_only": r4((via_bsp & ~via_pre).mean()) if len(g) else None,
                "ltp_pre_eq_t30_share": r4(np.mean(g["ltp_pre"].to_numpy() == t30)) if len(g) else None,
                "strike": r4((f > 0).mean()) if len(f) else None,
                "median_price": r4(np.median(pr)) if len(pr) else None})
    okb = ok & ~np.isnan(bsp)
    if okb.any():
        clv = price[okb] / bsp[okb]
        res["clv_mean"] = r4(clv.mean())
        res["clv_mean_log"] = r4(np.mean(np.log(clv)))
        res["clv_share_beats_close"] = r4(np.mean(clv > 1 + 1e-9) if side == "back" else np.mean(clv < 1 - 1e-9))
        res["t30_over_bsp_mean_log"] = r4(np.mean(np.log(t30[okb] / bsp[okb])))
        if settlement == "limit_then_bsp" and (okb & filled).any():
            res["clv_filled_mean_log"] = r4(np.mean(np.log(price[okb & filled] / bsp[okb & filled])))
            res["roi_8pct_filled_only"] = r4(returns(side, staking, price[okb & filled], g["f"].to_numpy()[okb & filled],
                                                     PRIMARY_C).mean())
            nf = okb & ~filled
            if nf.any():
                res["roi_8pct_unfilled_at_bsp"] = r4(returns(side, staking, bsp[nf], g["f"].to_numpy()[nf],
                                                             PRIMARY_C).mean())
    # CLV-implied ROI: expected 8% return of the entry if the BSP-implied probability (1/bsp over the
    # market's BSP book; fully BSP-priced markets) is true -- a low-variance read of the entry's edge
    qq = (1 / bsp) / g["m_bsp_book"].to_numpy()
    okq = ok & ~np.isnan(bsp) & g["m_all_bsp"].to_numpy()
    if okq.any():
        E, q = price[okq], qq[okq]
        if side == "back":
            ex = q * (E - 1) * (1 - PRIMARY_C) - (1 - q)
        else:
            ex = (1 - q) * (1 - PRIMARY_C) - q * (E - 1)
            ex = ex / (E - 1) if staking == "liability" else ex
        res["clv_implied_roi_8pct"] = r4(ex.mean())
    # paper references (NOT achievable): the t30 quote itself, and one tick worse taken by everyone
    fa = g["f"].to_numpy()
    if okb.any():
        res["paper_clv_mean_t30_over_bsp"] = r4(np.mean(t30[okb] / bsp[okb]))
        res["paper_clv_mean_tick_worse_over_bsp"] = r4(np.mean(L[okb] / bsp[okb]))
    res["paper_roi_8pct_at_t30"] = r4(returns(side, staking, t30, fa, PRIMARY_C).mean()) if len(g) else None
    res["paper_roi_8pct_at_t30_tick_worse"] = r4(returns(side, staking, L, fa, PRIMARY_C).mean()) if len(g) else None
    if len(r8) > 2:
        res["roi_8pct_excl_top_bet"] = r4(np.delete(r8, np.argmax(r8)).mean())
    return res


# ---------------------------------------------------------------------------
# CLV deciles
# ---------------------------------------------------------------------------
def clv_deciles(df: pd.DataFrame, side: str, entry: str) -> dict:
    """Every t30-priced runner of fully BSP-priced markets: entry = G2 limit (else BSP)
    or the paper t30 quote; deciles of log(entry/bsp)."""
    g = df[df["ltp_t30"].notna() & df["m_all_bsp"]]
    L, filled, _, _, ent_lim = entry_prices(g, side)
    E = ent_lim if entry == "limit_then_bsp" else g["ltp_t30"].to_numpy()
    bsp = g["bsp"].to_numpy()
    f = g["f"].to_numpy()
    q = (1 / bsp) / g["m_bsp_book"].to_numpy()
    lc = np.log(E / bsp)
    staking = "stake" if side == "back" else "liability"
    real8 = returns(side, staking, E, f, PRIMARY_C)
    real0 = returns(side, staking, E, f, 0.0)
    # expected return if the BSP-implied probability q is true (no dead heats)
    if side == "back":
        exp8 = q * (E - 1) * (1 - PRIMARY_C) - (1 - q)
        exp0 = q * E - 1
    else:
        exp8 = ((1 - q) * (1 - PRIMARY_C) - q * (E - 1)) / (E - 1)
        exp0 = ((1 - q) - q * (E - 1)) / (E - 1)
    dec = pd.qcut(pd.Series(lc).rank(method="first"), 10, labels=False).to_numpy()
    mk = g["market_id"].to_numpy()
    day = g["date"].to_numpy()
    rows = []
    for k in range(10):
        s = dec == k
        st = roi_stats(real8[s], mk[s], day[s])
        rows.append({"decile": k + 1, "bets": int(s.sum()), "clv_mean": r4(np.exp(lc[s]).mean()),
                     "clv_log_lo_hi": [r4(lc[s].min()), r4(lc[s].max())],
                     "filled_share": r4(filled[s].mean()) if entry == "limit_then_bsp" else None,
                     "roi_8pct": st["roi"], "se": st["se"], "roi_0pct": r4(real0[s].mean()),
                     "expected_roi_8pct_bsp_fair": r4(exp8[s].mean()), "expected_roi_0pct_bsp_fair": r4(exp0[s].mean()),
                     "median_price": r4(np.median(E[s]))})
    # slope of realised on expected return (0% commission), SE clustered by market
    x = exp0 - exp0.mean()
    y = real0
    b = float((x * (y - y.mean())).sum() / (x * x).sum())
    e = (y - y.mean()) - b * x
    sc = pd.Series(x * e).groupby(mk).sum().to_numpy()
    se_b = float(math.sqrt((sc ** 2).sum()) / (x * x).sum())
    # same against the CLV itself: realised return on log CLV
    xl = lc - lc.mean()
    bl = float((xl * (y - y.mean())).sum() / (xl * xl).sum())
    el = (y - y.mean()) - bl * xl
    scl = pd.Series(xl * el).groupby(mk).sum().to_numpy()
    se_bl = float(math.sqrt((scl ** 2).sum()) / (xl * xl).sum())
    return {"side": side, "entry": entry, "staking": staking, "bets": int(len(g)),
            "all_roi_8pct": r4(real8.mean()), "all_expected_roi_8pct": r4(exp8.mean()),
            "slope_realised_on_expected_0pct": r4(b), "slope_se": r4(se_b),
            "slope_realised_on_logclv_0pct": r4(bl), "slope_logclv_se": r4(se_bl),
            "spearman_decile_clv_vs_roi": r4(pd.Series([r["clv_mean"] for r in rows]).corr(
                pd.Series([r["roi_8pct"] for r in rows]), method="spearman")),
            "deciles": rows}


# ---------------------------------------------------------------------------
# probability: walk-forward power recalibration at t30
# ---------------------------------------------------------------------------
def shin_matrix(O: np.ndarray) -> np.ndarray:
    pi = np.where(np.isnan(O), 0.0, 1.0 / np.where(np.isnan(O), 1.0, O))
    s = pi.sum(1, keepdims=True)
    lo = np.zeros_like(s)
    hi = np.full_like(s, 0.5)

    def probs(z):
        return (np.sqrt(z * z + 4 * (1 - z) * pi * pi / s) - z) / (2 * (1 - z))
    for _ in range(60):
        mid = (lo + hi) / 2
        tot = probs(mid).sum(1, keepdims=True)
        lo = np.where(tot > 1, mid, lo)
        hi = np.where(tot > 1, hi, mid)
    pr = probs((lo + hi) / 2)
    pr = pr / pr.sum(1, keepdims=True)
    under = (s[:, 0] <= 1.0)
    pr[under] = pi[under] / s[under]
    return pr


def matrices(df: pd.DataFrame, col: str):
    d = df.sort_values(["market_id", "selection_id"]).copy()
    d["race"] = pd.factorize(d["market_id"])[0]
    d["pos"] = d.groupby("race").cumcount()
    R, W = int(d["race"].max()) + 1, int(d["pos"].max()) + 1
    O = np.full((R, W), np.nan)
    F = np.zeros((R, W))
    O[d["race"], d["pos"]] = d[col].to_numpy()
    F[d["race"], d["pos"]] = d["f"].to_numpy()
    F = F / F.sum(1, keepdims=True)
    rd = d.drop_duplicates("race").set_index("race").sort_index()
    return O, F, rd["date"].to_numpy(), rd["date"].dt.to_period("M").astype(str).to_numpy()


def ll_rows(P: np.ndarray, F: np.ndarray) -> np.ndarray:
    return -(F * np.log(np.clip(np.where(F > 0, P, 1.0), 1e-12, 1))).sum(1)


def power_fit(lq: np.ndarray, F: np.ndarray, mask: np.ndarray, a0: float = 1.0) -> float:
    """1-D Newton for alpha maximising sum_r sum_i F_ri log softmax(alpha * lq_r)_i."""
    a = a0
    X = np.where(mask, lq, 0.0)
    for _ in range(30):
        z = np.where(mask, a * X, -np.inf)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(1, keepdims=True)
        ex = (p * X).sum(1)
        gr = ((F * X).sum(1) - ex).sum()
        h = -((p * X * X).sum(1) - ex ** 2).sum()
        step = gr / h
        a = a - step
        if abs(step) < 1e-7:
            break
    return float(a)


def prob_section(df: pd.DataFrame, seg: str, mid: pd.Timestamp) -> dict:
    d = df[df["m_all_t30"]]
    if d.empty:
        return {"segment": seg, "races": 0}
    O, F, dates, months = matrices(d, "ltp_t30")
    mask = ~np.isnan(O)
    Q = shin_matrix(O)
    lq = np.log(np.where(mask, np.clip(Q, 1e-12, 1), 1.0))
    ms = sorted(set(months))
    P = np.full_like(Q, np.nan)
    alphas = {}
    a = 1.0
    for idx, m in enumerate(ms):
        if idx < 6:
            continue
        tr = np.isin(months, ms[:idx])
        a = power_fit(lq[tr], F[tr], mask[tr], a)
        alphas[m] = round(a, 4)
        te = months == m
        z = np.where(mask[te], a * lq[te], -np.inf)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        P[te] = e / e.sum(1, keepdims=True)
    ev = ~np.isnan(P[:, 0])
    llm = ll_rows(Q[ev], F[ev])
    llp = ll_rows(np.where(mask[ev], P[ev], 1.0), F[ev])
    dlt = llm - llp
    dts = dates[ev]
    fh = dts < np.datetime64(mid)
    l12 = dts >= np.datetime64(LAST12_FROM)

    def ms_(x):
        return [r6(x.mean()), r6(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None]
    av = np.array(list(alphas.values()))
    res = {"segment": seg, "races": int(ev.sum()), "ll_t30_shin": r4(llm.mean()), "ll_model": r4(llp.mean()),
           "gain": ms_(dlt)[0], "se": ms_(dlt)[1], "z": r4(dlt.mean() / ms_(dlt)[1]) if ms_(dlt)[1] else None,
           "halves_gain": [ms_(dlt[fh])[0], ms_(dlt[~fh])[0]], "halves_se": [ms_(dlt[fh])[1], ms_(dlt[~fh])[1]],
           "last12m_gain": r6(dlt[l12].mean()) if l12.any() else None,
           "alpha_first_last_min_max": [r4(av[0]), r4(av[-1]), r4(av.min()), r4(av.max())] if len(av) else None}
    # descriptive (NOT same timestamp): t30 vs ltp_pre vs BSP market log loss on markets fully priced at all three
    d2 = d[(d["m_pre_book"].between(0.97, 1.03)) & d["m_all_bsp"]]
    d2 = d2[d2.groupby("market_id")["ltp_pre"].transform("count") == d2["m_n"]]
    if len(d2):
        o30, F2, dt2, _ = matrices(d2, "ltp_t30")
        opre, _, _, _ = matrices(d2, "ltp_pre")
        obsp, _, _, _ = matrices(d2, "bsp")
        l30 = ll_rows(shin_matrix(o30), F2)
        lpre = ll_rows(shin_matrix(opre), F2)
        lb = ll_rows(shin_matrix(obsp), F2)
        res["descriptive_markets"] = int(len(l30))
        res["descriptive_ll_t30_pre_bsp"] = [r4(l30.mean()), r4(lpre.mean()), r4(lb.mean())]
        res["descriptive_gain_bsp_over_t30"] = ms_(l30 - lb)
        res["descriptive_gain_pre_over_t30"] = ms_(l30 - lpre)
    return res


# ---------------------------------------------------------------------------
def rule_list():
    rules = []
    for k in (1, 2, 3):
        rules.append({"name": f"back_t30_rank{k}", "side": "back", "group": "a",
                      "sel": (lambda d, k=k: d["rank30"] == k),
                      "text": f"t30 rank-{k} favourite by ltp_t30 (fully t30-priced markets, ties by selection_id)"})
    for lo, hi in LAY_LONG:
        nm = f"lay_t30_{lo}_{hi}" if hi < 1001 else f"lay_t30_ge{lo}"
        rules.append({"name": nm, "side": "lay", "group": "b",
                      "sel": (lambda d, lo=lo, hi=hi: (d["ltp_t30"] >= lo) & (d["ltp_t30"] < hi)),
                      "text": f"ltp_t30 in [{lo},{hi})" if hi < 1001 else f"ltp_t30 >= {lo}"})
    for lo, hi in BACK_BANDS:
        nm = f"back_band_{lo}_{hi}" if hi < 1001 else f"back_band_ge{lo}"
        rules.append({"name": nm, "side": "back", "group": "c",
                      "sel": (lambda d, lo=lo, hi=hi: (d["ltp_t30"] >= lo) & (d["ltp_t30"] < hi)),
                      "text": f"ltp_t30 in [{lo},{hi})" if hi < 1001 else f"ltp_t30 >= {lo}"})
    for lo, hi in LAY_BANDS:
        rules.append({"name": f"lay_band_{lo}_{hi}", "side": "lay", "group": "c",
                      "sel": (lambda d, lo=lo, hi=hi: (d["ltp_t30"] >= lo) & (d["ltp_t30"] < hi)),
                      "text": f"ltp_t30 in [{lo},{hi})"})
    return rules


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "data/raw/betfair/split")
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of markets (development)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    t0 = time.time()

    # tick ladder and Shin sanity checks
    assert tick_shorter(np.array([2.0]))[0] == 1.99 and tick_longer(np.array([2.0]))[0] == 2.02
    assert tick_shorter(np.array([50.0]))[0] == 48.0 and tick_longer(np.array([50.0]))[0] == 55.0
    assert tick_longer(np.array([1000.0]))[0] == 1000.0 and tick_shorter(np.array([1.01]))[0] == 1.01
    tst = np.array([[2.5, 3.2, 6.0, 11.0, np.nan]])
    assert np.allclose(shin_matrix(tst)[0, :4], shin_probabilities(tst[0, :4]), atol=1e-8)

    parts, load = {}, {}
    for c in COUNTRIES:
        w, st = load_country(c, a.dir, a.frac, a.seed)
        parts[c] = w
        load[c] = st
    out({"section": "data", "load": load, "runtime_s": round(time.time() - t0, 1)})

    rules = rule_list()
    cells, claims_rows, sens_rows = [], [], []
    n_cells = 0
    n_cells_sens = 0
    probs, deciles, seg_meta = [], [], {}
    for seg in COUNTRIES:
        dfall = parts[seg]
        dfall = dfall[dfall["ltp_t30"].notna() | dfall["m_all_t30"]]
        df = dfall[~dfall["m_g6_t30"]]
        rd = df[df["ltp_t30"].notna()].drop_duplicates("market_id")
        mid = rd["date"].sort_values().iloc[len(rd) // 2]
        n_races = int(len(rd))
        n_months = int(rd["date"].dt.to_period("M").nunique())
        seg_meta[seg] = {"races_with_t30": n_races, "races_all_t30": int(rd["m_all_t30"].sum()),
                         "t30_runners": int(df["ltp_t30"].notna().sum()), "median_date": str(mid.date()),
                         "months": n_months, "g6_t30_dropped": int(dfall.loc[dfall["m_g6_t30"], "market_id"].nunique()),
                         "races_by_year": rd["date"].dt.year.value_counts().sort_index().to_dict(),
                         "bsp_missing_share_t30_runners": r4(df.loc[df["ltp_t30"].notna(), "bsp"].isna().mean()),
                         "mean_log_t30_over_bsp_by_rank1_2_3": [r4(np.nanmean(np.log(
                             df.loc[df["rank30"] == k, "ltp_t30"] / df.loc[df["rank30"] == k, "bsp"]))) for k in (1, 2, 3)]}
        if seg == "au":
            seg_meta[seg]["harness_share_of_t30_runners"] = r4(df.loc[df["ltp_t30"].notna(), "harness"].mean())
        for ru in rules:
            sel = ru["sel"](df).to_numpy() & df["ltp_t30"].notna().to_numpy()
            g = df[sel]
            sel_all = ru["sel"](dfall).to_numpy() & dfall["ltp_t30"].notna().to_numpy()
            g_all = dfall[sel_all]
            stakings = ["stake"] if ru["side"] == "back" else ["stake", "liability"]
            for settle in ("limit_then_bsp", "bsp"):
                for stk in stakings:
                    res = eval_cell(g, ru["side"], settle, stk, mid, n_races, n_months)
                    n_cells += 1
                    sname = settle if stk == "stake" else f"per_liability_{settle}"
                    row = {"segment": seg, "rule": ru["name"], "group": ru["group"], "side": ru["side"],
                           "settlement": sname, "selection": ru["text"], **res}
                    # sensitivities: keep the G6-t30 markets; strict fill (bsp must reach the limit)
                    s1 = eval_cell(g_all, ru["side"], settle, stk, mid, n_races, n_months, full=False)
                    n_cells_sens += 1
                    row["sens_keep_g6t30"] = {k: s1[k] for k in ("bets", "roi_8pct", "se_clustered", "passes_G4_8pct")}
                    if settle == "limit_then_bsp":
                        s2 = eval_cell(g, ru["side"], settle, stk, mid, n_races, n_months, strict=True, full=False)
                        n_cells_sens += 1
                        row["sens_strict_fill_bsp_only"] = {k: s2[k] for k in ("bets", "roi_8pct", "se_clustered",
                                                                                "passes_G4_8pct")}
                    row["passes_all_gates"] = bool(res["passes_G4_8pct"])
                    cells.append(row)
                    out({"section": "cell", **{k: row[k] for k in (
                        "segment", "rule", "settlement", "bets", "roi_8pct", "se_clustered", "z", "halves_roi_8pct",
                        "halves_se", "last12m_roi_8pct", "roi_0_5_10pct", "passes_G4_8pct", "passes_G4_5pct")},
                         "fill": row.get("fill_rate"), "clv_log": row.get("clv_mean_log"),
                         "clv_beat": row.get("clv_share_beats_close"),
                         "paper_t30": row.get("paper_roi_8pct_at_t30"), "clv_mean": row.get("clv_mean"),
                         "clv_implied": row.get("clv_implied_roi_8pct"),
                         "paper_tw": row.get("paper_roi_8pct_at_t30_tick_worse"),
                         "paper_clv": row.get("paper_clv_mean_t30_over_bsp"),
                         "fill_only": row.get("roi_8pct_filled_only"), "unfilled": row.get("roi_8pct_unfilled_at_bsp"),
                         "mde": row.get("mde_80pct_full"), "fires_per_race": row.get("fires_per_race"),
                         "sens_keep": row["sens_keep_g6t30"]["roi_8pct"],
                         "sens_strict": row.get("sens_strict_fill_bsp_only", {}).get("roi_8pct")})
        probs.append(prob_section(df, seg, mid))
        out({"section": "prob", **probs[-1]})
    out({"section": "segments", **seg_meta})

    # CLV deciles: GB+IE pooled and AU
    gbie = pd.concat([parts["gb"][~parts["gb"]["m_g6_t30"]], parts["ie"][~parts["ie"]["m_g6_t30"]]], ignore_index=True)
    au = parts["au"][~parts["au"]["m_g6_t30"]]
    n_dec = 0
    for nm, d in (("gbie", gbie), ("au", au)):
        for side in ("back", "lay"):
            for entry in ("limit_then_bsp", "paper_t30"):
                r = clv_deciles(d, side, entry)
                r["segment"] = nm
                n_dec += 10
                deciles.append(r)
                out({"section": "clv_deciles", **r})

    passing = [c for c in cells if c["passes_all_gates"]]
    out({"section": "g5_count", "primary_cells": n_cells, "sensitivity_cells": n_cells_sens,
         "clv_decile_cells_descriptive": n_dec, "prob_cells": len(probs),
         "total": n_cells + n_cells_sens + n_dec + len(probs),
         "commission_variants_reported_not_gated": 3,
         "passing_G4_8pct": [(c["segment"], c["rule"], c["settlement"]) for c in passing],
         "passing_G4_5pct_only": [(c["segment"], c["rule"], c["settlement"]) for c in cells
                                  if c["passes_G4_5pct"] and not c["passes_G4_8pct"]]})
    if a.json:
        a.json.write_text(json.dumps({"load": load, "segments": seg_meta, "cells": cells, "prob": probs,
                                      "deciles": deciles, "n_cells": n_cells, "n_cells_sens": n_cells_sens,
                                      "n_dec": n_dec}, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    out({"section": "done", "runtime_s": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

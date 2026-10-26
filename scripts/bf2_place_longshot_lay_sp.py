"""Experiment "place_longshot_lay_sp" (round 2): lay PLACE-market longshots at the
place BSP, selected on pre-off information only; plus a WIN-BSP lay of runners
the place ladder dislikes at t30 and that drifted since t30.

    python scripts/bf2_place_longshot_lay_sp.py [--dir data/raw/betfair/split] [--frac 0.2] [--json out.json]

Universe: AU, GB and IE races whose WIN and PLACE ("To Be Placed") markets pair on
event_id + start, have identical runner sets, pay 2 or 3 places (GB/IE 4-place
markets are excluded and counted), are settled (WIN: >= 1 winner; PLACE: >= n_winners
placed) and have at least n_winners + 1 runners. Deduplicated on (market_id,
selection_id); prices <= 1 set to NaN. AU harness = WIN market_name contains Pace/Trot.
  G6      drop races with a saddlecloth gap (all saddlecloths numeric and max > runners)
          AND an ltp_pre book < 0.97 (WIN book over priced runners, or PLACE book /
          n_winners when every place runner has an ltp_pre).
  G6-t30  for decisions taken at t30 (and rule 2, which uses t30 prices) also drop races
          with a gap AND a t30 book < 0.97 (WIN, or PLACE / n_winners, fully t30-priced).

RULE 1 -- lay the runner in the PLACE market at the place BSP (G2 primary), one bet per
selected runner. Selections (each uses only pre-off information, G1):
  R1a win_pre   WIN ltp_pre >= T (T = 30, 50*, 100); the WIN market must be fully
                priced at ltp_pre with a book in [0.97, 1.03] (G1). Decided at the off.
  R1b place_pre PLACE ltp_pre >= T (T = 10, 20*, 50); the PLACE market must be fully
                priced at ltp_pre with book / n_winners in [0.97, 1.03] (G1). At the off.
  R1c win_t30   WIN ltp_t30 >= T (T = 30, 50*, 100). Decided 30 minutes out; settled
                (a) limit_then_bsp: a lay limit at the PLACE ltp_t30 one tick LONGER,
                    filled at the limit iff the later place ltp_pre or the place bsp is
                    <= the limit, else the place BSP; runners with no place ltp_t30 go
                    straight to the place BSP;
                (b) bsp: plain place BSP.
  (* = the pre-registered primary threshold)
RULE 2 -- lay at the WIN BSP. At t30 (both markets fully t30-priced): p = 1/ltp_t30
  normalised in the WIN market; q_harv = discounted-Harville place probability of p
  (one lambda for 2nd and 3rd, chosen each month on the 12 earlier months by binary
  log loss of the placed flag, pools AU / GB+IE, grid 0.10..1.60, G7); q_mkt = place
  1/ltp_t30 normalised to n_winners. Select ratio = q_mkt / q_harv < R (R = 0.7, 0.8*,
  0.9) AND WIN ltp_pre / WIN ltp_t30 > 1.1 AND WIN ltp_pre <= 20, WIN ltp_pre book in
  [0.97, 1.03] (G1). Controls on the same universe: ratio < 0.8 without the drift
  condition, and the drift condition without the ratio. Secondary universe (more AU
  races): the place market need not be fully t30-priced; ratio_u = (1/place ltp_t30) /
  q_harv (unnormalised) < 0.8 with the same drift and cap conditions.

  Replication of round 1 at ltp_pre (cross-market NOT simultaneous; both ltp_pre books in
  [0.97, 1.03], lambda refitted walk-forward on ltp_pre): ratio_pre < 0.8 with and without
  the drift condition.
ROBUSTNESS (per unit of liability, pooled): rule-1 cells by calendar year; a
LIMIT_ON_CLOSE cap (lay at the place BSP only if it is <= 50 / 100); the place-BSP bucket
of the bets (descriptive); R1b with the PLACE book in [0.95, 1.05] or unfiltered (the
unfiltered one breaks G1 and is not an eligible rule).

Settlement: back net per unit of backer stake = f(P-1) - (1-f); a lay is minus that;
commission on positive net winnings of each bet at 0/5/8/10% (8% primary, G3). Lays
reported per unit of LIABILITY (return / (P-1)) and per unit of backer's stake.
f (PLACE) = n_winners / n_placed for a placed runner (dead heat for the last place),
0 otherwise; f (WIN) = dead_heat_share for a winner.
MISSING BSP: a BSP order in a market (or on a runner) with no reconciled BSP is not
matched, so such decisions are NOT bets (counted as 'no_bsp'); the placed rate and the
WIN-implied probability of the skipped decisions are printed beside the settled ones.
Statistics: SE = max(iid, clustered by race, clustered by race day); halves split at
the median race date of the cell's bets; last 12 months = 2019-11-08 .. 2020-11-08;
G4 = ROI8 > 2 SE in both halves AND ROI8 > 0 in the last 12 months; a half with fewer than 10
lost lays has a degenerate SE and fails. Also printed: ROI8 with commission netted per
race over all of a rule's lays in that market (Betfair charges per market). MDE (80% power,
one-sided 2-SE test) = 2.84 SE. Every evaluated cell is counted (G5) and a one-sided
p-value is printed for the programme-wide BH.
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
from scipy.special import erfcinv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ponies.exotics import order_probs  # noqa: E402

WCOLS = ["market_id", "event_id", "market_name", "start", "date", "selection_id", "program", "ltp_pre",
         "ltp_t30", "bsp", "won", "dead_heat_share"]
PCOLS = ["market_id", "event_id", "start", "n_winners", "selection_id", "ltp_pre", "ltp_t30", "bsp", "won"]
COUNTRIES = ["au", "gb", "ie"]
COMMS = [0.0, 0.05, 0.08, 0.10]
PRIMARY_C = 0.08
LAST12_FROM = np.datetime64("2019-11-08")
LAST12_TO = np.datetime64("2020-11-08")
Z80 = 0.8416
BOOK_LO, BOOK_HI = 0.97, 1.03
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1000.0, 10.0)]
LAMBDA_GRID = np.round(np.arange(0.10, 1.61, 0.05), 2)
LAMBDA_WINDOW = 12
LAMBDA_MIN_MONTHS = 6
R1A_T = [30, 50, 100]
R1B_T = [10, 20, 50]
R1C_T = [30, 50, 100]
R2_R = [0.7, 0.8, 0.9]
DRIFT = 1.1
R2_CAP = 20.0
MIN_LOSSES_PER_HALF = 10
SEGMENTS = ["all", "au_gallops", "au_harness", "gb", "ie", "nw2", "nw3"]


def out(obj) -> None:
    print(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)), flush=True)


def r4(x):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, 4)


def r6(x):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, 6)


# ---------------------------------------------------------------------------
# tick ladder (as scripts/bf2_t30_early_price.py)
# ---------------------------------------------------------------------------
def tick_longer(price) -> np.ndarray:
    p = np.round(np.asarray(price, dtype=float), 2)
    step = np.full_like(p, 10.0)
    for lo, hi, inc in reversed(TICKS):
        step = np.where((p >= lo - 1e-9) & (p < hi - 1e-9), inc, step)
    return np.minimum(np.round(p + step, 2), 1000.0)


# ---------------------------------------------------------------------------
# loading / joining
# ---------------------------------------------------------------------------
def load_country(cc: str, d: Path, frac: float, seed: int) -> tuple[pd.DataFrame, dict]:
    w = pd.read_parquet(d / f"{cc}_win.parquet", columns=WCOLS).drop_duplicates(["market_id", "selection_id"])
    p = pd.read_parquet(d / f"{cc}_place.parquet", columns=PCOLS).drop_duplicates(["market_id", "selection_id"])
    for df in (w, p):
        for c in ("market_id", "event_id", "selection_id"):
            df[c] = df[c].astype(object)
    if frac < 1:
        ev = w["event_id"].drop_duplicates()
        keep = set(ev.sample(frac=frac, random_state=seed))
        w = w[w["event_id"].isin(keep)]
        p = p[p["event_id"].isin(keep)]
    st = {"win_markets": int(w["market_id"].nunique()), "place_markets": int(p["market_id"].nunique())}
    w = w.copy()
    w["harness"] = w["market_name"].astype(object).fillna("").str.contains(r"\b(?:Pace|Trot)\b", regex=True)
    wm = w.drop_duplicates("market_id")[["market_id", "event_id", "start"]]
    pm = p.drop_duplicates("market_id")[["market_id", "event_id", "start", "n_winners"]]
    wm = wm[~wm.duplicated(["event_id", "start"], keep=False)]
    pm = pm[~pm.duplicated(["event_id", "start"], keep=False)]
    pairs = wm.rename(columns={"market_id": "win_mid"}).merge(
        pm.rename(columns={"market_id": "pl_mid"})[["event_id", "start", "pl_mid", "n_winners"]],
        on=["event_id", "start"])
    st["paired_markets"] = int(len(pairs))
    st["paired_nw_counts"] = {int(k): int(v) for k, v in pairs["n_winners"].value_counts().items()}
    pairs = pairs[pairs["n_winners"].isin([2, 3])]
    st["paired_nw_2_3"] = int(len(pairs))
    w = w[w["market_id"].isin(pairs["win_mid"])].rename(columns={
        "market_id": "win_mid", "ltp_pre": "pre_win", "ltp_t30": "t30_win", "bsp": "bsp_win", "won": "won_win"})
    w = w.drop(columns=["market_name"])
    p = p[p["market_id"].isin(pairs["pl_mid"])].rename(columns={
        "market_id": "pl_mid", "ltp_pre": "pre_pl", "ltp_t30": "t30_pl", "bsp": "bsp_pl", "won": "placed"})
    p = p.merge(pairs[["pl_mid", "win_mid"]], on="pl_mid")
    p = p[["win_mid", "pl_mid", "selection_id", "n_winners", "pre_pl", "t30_pl", "bsp_pl", "placed"]]
    j = w.merge(p, on=["win_mid", "selection_id"], how="outer", indicator=True)
    bad = set(j.loc[j["_merge"] != "both", "win_mid"])
    j = j[~j["win_mid"].isin(bad)].drop(columns=["_merge"]).copy()
    st["identical_runner_sets"] = int(j["win_mid"].nunique())
    g = j.groupby("win_mid", sort=False)
    j["n"] = g["selection_id"].transform("size")
    j["n_placed"] = g["placed"].transform("sum")
    j["n_won"] = g["won_win"].transform("sum")
    j["_na"] = j["placed"].isna() | j["won_win"].isna()
    j["any_na"] = g["_na"].transform("max")
    keep = (~j["any_na"].astype(bool)) & (j["n_placed"] >= j["n_winners"]) & (j["n_won"] >= 1) \
        & (j["n"] >= j["n_winners"] + 1)
    j = j[keep].copy()
    st["settled_n_ge_nw_plus_1"] = int(j["win_mid"].nunique())
    for c in ("pre_win", "t30_win", "bsp_win", "pre_pl", "t30_pl", "bsp_pl"):
        j.loc[~(j[c] > 1), c] = np.nan
    j["harness"] = j["harness"].astype(bool)
    j["seg"] = np.where(j["harness"], "au_harness", "au_gallops") if cc == "au" else cc
    j["pool"] = "au" if cc == "au" else "gbie"
    g = j.groupby("win_mid", sort=False)
    prog = pd.to_numeric(j["program"].astype(object), errors="coerce")
    j["_prog"] = prog
    pmax = g["_prog"].transform("max")
    pcnt = g["_prog"].transform("count")
    j["gap"] = (pcnt == j["n"]) & (pmax > j["n"])
    for col, nm in (("pre_win", "win_pre"), ("pre_pl", "pl_pre"), ("t30_win", "win_t30"), ("t30_pl", "pl_t30")):
        j["_i"] = 1.0 / j[col]
        j[nm + "_book"] = g["_i"].transform("sum")
        j[nm + "_cnt"] = g[col].transform("count")
        j[nm + "_all"] = j[nm + "_cnt"] == j["n"]
    j["pl_pre_book"] = j["pl_pre_book"] / j["n_winners"]
    j["pl_t30_book"] = j["pl_t30_book"] / j["n_winners"]
    j["g6"] = j["gap"] & ((j["win_pre_book"] < 0.97) | (j["pl_pre_all"] & (j["pl_pre_book"] < 0.97)))
    j["g6_t30"] = j["gap"] & ~j["g6"] & ((j["win_t30_all"] & (j["win_t30_book"] < 0.97))
                                         | (j["pl_t30_all"] & (j["pl_t30_book"] < 0.97)))
    j["f_win"] = np.where(j["won_win"] == 1, j["dead_heat_share"].fillna(1.0), 0.0).astype(float)
    j["f_pl"] = np.where(j["placed"] == 1, j["n_winners"] / j["n_placed"], 0.0).astype(float)
    j["month"] = j["start"].dt.tz_convert(None).dt.to_period("M").astype(str)
    j["race"] = cc + ":" + j["win_mid"].astype(str)
    races = j.drop_duplicates("race")
    st.update({"races": int(len(races)), "runners": int(len(j)), "g6_dropped_races": int(races["g6"].sum()),
               "g6_t30_flagged_races": int(races["g6_t30"].sum()),
               "share_races_win_pre_all": r4(races["win_pre_all"].mean()),
               "share_races_pl_pre_all": r4(races["pl_pre_all"].mean()),
               "share_races_win_t30_all": r4(races["win_t30_all"].mean()),
               "share_races_pl_t30_all": r4(races["pl_t30_all"].mean()),
               "share_runners_pl_bsp": r4(j["bsp_pl"].notna().mean()),
               "share_runners_win_bsp": r4(j["bsp_win"].notna().mean()),
               "place_dead_heat_races": int((races["n_placed"] > races["n_winners"]).sum())})
    j = j[~j["g6"]].drop(columns=["_prog", "_i", "_na", "any_na", "program", "event_id", "dead_heat_share",
                                  "n_won", "won_win", "pl_mid"])
    return j.reset_index(drop=True), st


# ---------------------------------------------------------------------------
# discounted Harville, batched (round-1 code, checked against ponies.exotics.order_probs)
# ---------------------------------------------------------------------------
def topk_batch(P: np.ndarray, lam: float, k: int) -> np.ndarray:
    P = np.asarray(P, float)
    R, n = P.shape
    a = P ** lam
    A = a.sum(1, keepdims=True)
    den_j = np.maximum(A - a, 1e-300)
    ex = P[:, :, None] * (a[:, None, :] / den_j[:, :, None])
    idx = np.arange(n)
    ex[:, idx, idx] = 0.0
    top2 = P + ex.sum(1)
    if k == 2:
        return top2
    den_jk = np.maximum(A[:, :, None] - a[:, :, None] - a[:, None, :], 1e-300)
    W = ex / den_jk
    S = W.sum((1, 2))
    third = a * (S[:, None] - W.sum(2) - W.sum(1))
    return top2 + third


def harville_grid(h: pd.DataFrame) -> np.ndarray:
    """h sorted by (race, p desc); columns race, n, n_winners, p. Returns rows x len(LAMBDA_GRID)."""
    P = h["p"].to_numpy()
    Q = np.empty((len(h), len(LAMBDA_GRID)))
    key = pd.MultiIndex.from_arrays([h["n"].to_numpy(), h["n_winners"].to_numpy()])
    codes, uniq = pd.factorize(key)
    for u, (n, k) in enumerate(uniq):
        idx = np.flatnonzero(codes == u)
        Pm = P[idx].reshape(-1, int(n))
        for gi, lam in enumerate(LAMBDA_GRID):
            res = np.empty_like(Pm)
            for s in range(0, len(Pm), 3000):
                res[s:s + 3000] = topk_batch(Pm[s:s + 3000], float(lam), int(k))
            Q[idx, gi] = res.ravel()
    return Q


def lambda_path(h: pd.DataFrame, Q: np.ndarray) -> tuple[dict, dict]:
    """lam[(pool, month)] = argmin over the grid of the mean binary log loss of Q vs placed over the
    LAMBDA_WINDOW months strictly before (at least LAMBDA_MIN_MONTHS)."""
    y = h["placed"].to_numpy(float)
    Qc = np.clip(Q, 1e-6, 1 - 1e-6)
    LL = -(y[:, None] * np.log(Qc) + (1 - y[:, None]) * np.log(1 - Qc))
    lam, info = {}, {}
    for pool in sorted(h["pool"].unique()):
        mk = (h["pool"] == pool).to_numpy()
        months = sorted(h.loc[mk, "month"].unique())
        mc = pd.Categorical(h.loc[mk, "month"], categories=months).codes
        S = np.zeros((len(months), len(LAMBDA_GRID)))
        N = np.zeros(len(months))
        np.add.at(S, mc, LL[mk])
        np.add.at(N, mc, 1.0)
        path = []
        for i, m in enumerate(months):
            win = list(range(max(0, i - LAMBDA_WINDOW), i))
            if len(win) < LAMBDA_MIN_MONTHS:
                continue
            ml = S[win].sum(0) / N[win].sum()
            gi = int(np.argmin(ml))
            lam[(pool, m)] = float(LAMBDA_GRID[gi])
            path.append(gi)
        path = np.array(path)
        info[pool] = {"months_fitted": int(len(path)),
                      "lambda_min_median_max": [float(LAMBDA_GRID[path.min()]), float(LAMBDA_GRID[int(np.median(path))]),
                                                float(LAMBDA_GRID[path.max()])] if len(path) else None,
                      "grid_edge_hits": int(((path == 0) | (path == len(LAMBDA_GRID) - 1)).sum()) if len(path) else None,
                      "runners_for_fit": int(mk.sum())}
    return lam, info


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def cluster_se(r: np.ndarray, cl: np.ndarray) -> float | None:
    if len(r) < 2:
        return None
    dsum = pd.Series(r - r.mean()).groupby(cl).sum().to_numpy()
    G = len(dsum)
    if G < 2:
        return None
    return float(math.sqrt((dsum ** 2).sum() * G / (G - 1)) / len(r))


def roi_stats(r: np.ndarray, race: np.ndarray, day: np.ndarray) -> dict:
    n = len(r)
    if n < 2:
        return {"bets": int(n), "roi": r6(r.mean()) if n else None, "se": None}
    se_r = cluster_se(r, race)
    se_d = cluster_se(r, day)
    se_i = float(r.std(ddof=1) / math.sqrt(n))
    se = max(x for x in (se_r, se_d, se_i) if x is not None)
    return {"bets": int(n), "roi": r6(r.mean()), "se": r6(se), "se_race": r6(se_r), "se_day": r6(se_d),
            "se_iid": r6(se_i)}


def lay_ret(price, f, c, staking):
    gross = (1 - f) - f * (price - 1)
    net = np.where(gross > 0, gross * (1 - c), gross)
    return net / (price - 1) if staking == "liability" else net


def g4_pass(h1, h2, lt, ev1: int, ev2: int) -> bool:
    """G4; a half whose lays lost fewer than MIN_LOSSES_PER_HALF times has a degenerate SE (0 when no lay
    lost) and fails."""
    return bool(h1["se"] is not None and h2["se"] is not None and h1["roi"] is not None and h2["roi"] is not None
                and ev1 >= MIN_LOSSES_PER_HALF and ev2 >= MIN_LOSSES_PER_HALF and h1["se"] > 0 and h2["se"] > 0
                and h1["roi"] > 2 * h1["se"] and h2["roi"] > 2 * h2["se"]
                and lt["roi"] is not None and lt["roi"] > 0)


def one_sided_p(z):
    if z is None:
        return None
    return float(0.5 * math.erfc(z / math.sqrt(2)))


class Cells:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, row: dict) -> dict:
        self.rows.append(row)
        return row


def eval_cell(g: pd.DataFrame, price: np.ndarray, f: np.ndarray, staking: str, n_univ_races: int,
              n_months: int, extra: dict | None = None) -> dict:
    """g: decision rows; price: settlement price per decision (NaN = no bet); f: settlement fraction."""
    ok = np.isfinite(price)
    pr, ff = price[ok], f[ok]
    race = g["race"].to_numpy()[ok]
    day = g["date"].to_numpy()[ok]
    res = {"decisions": int(len(g)), "bets": int(ok.sum()), "no_price": int((~ok).sum()),
           "settled_share": r4(ok.mean()) if len(g) else None}
    if ok.sum() < 2:
        res.update({"roi_8pct": None, "se_clustered": None, "passes_G4": False})
        return res
    rets = {c: lay_ret(pr, ff, c, staking) for c in COMMS}
    r8 = rets[PRIMARY_C]
    rd = pd.Series(day).groupby(race).first().to_numpy()
    mid = np.sort(rd)[len(rd) // 2]
    fh = day < mid
    l12 = (day >= LAST12_FROM) & (day <= LAST12_TO)
    stt = roi_stats(r8, race, day)
    h1 = roi_stats(r8[fh], race[fh], day[fh])
    h2 = roi_stats(r8[~fh], race[~fh], day[~fh])
    lt = roi_stats(r8[l12], race[l12], day[l12])
    z = (stt["roi"] / stt["se"]) if stt["se"] else None
    res.update({
        "roi_8pct": stt["roi"], "se_clustered": stt["se"], "se_race": stt.get("se_race"), "se_day": stt.get("se_day"),
        "se_iid": stt.get("se_iid"), "z": r4(z), "p_one_sided": r6(one_sided_p(z)) if z is not None else None,
        "roi_0_5_10pct": [r6(rets[c].mean()) for c in (0.0, 0.05, 0.10)],
        "halves_roi_8pct": [h1["roi"], h2["roi"]], "halves_se": [h1["se"], h2["se"]],
        "halves_bets": [h1["bets"], h2["bets"]], "half_split_date": str(mid)[:10],
        "last12m_roi_8pct": lt["roi"], "last12m_se": lt["se"], "last12m_bets": lt["bets"],
        "losses_per_half": [int((ff[fh] > 0).sum()), int((ff[~fh] > 0).sum())],
        "passes_G4": g4_pass(h1, h2, lt, int((ff[fh] > 0).sum()), int((ff[~fh] > 0).sum())),
        "roi_8pct_market_netted": r6(market_netted(pr, ff, race, staking)),
        "mde_80pct": r6((2 + Z80) * stt["se"]) if stt["se"] else None,
        "mde_80pct_each_half": r6((2 + Z80) * max(h1["se"], h2["se"])) if h1["se"] and h2["se"] else None,
        "strike": r4(ff.mean()), "implied_at_price": r4(np.mean(1 / pr)),
        "strike_over_implied": r4(ff.mean() / np.mean(1 / pr)),
        "mean_price": r4(pr.mean()), "median_price": r4(np.median(pr)), "p90_price": r4(np.quantile(pr, 0.9)),
        "races_with_bet": int(len(np.unique(race))), "bets_per_race_with_bet": r4(len(pr) / len(np.unique(race))),
        "fires_per_universe_race": r4(len(g) / n_univ_races) if n_univ_races else None,
        "bets_per_month": r4(len(pr) / n_months) if n_months else None,
    })
    if len(r8) > 2:
        res["roi_8pct_excl_worst_bet"] = r6(np.delete(r8, np.argmin(r8)).mean())
    if extra:
        res.update(extra)
    return res


def market_netted(pr, ff, race, staking, c=PRIMARY_C) -> float:
    """8% commission charged on each race's positive NET over all of the rule's lays in that race (Betfair
    charges per market), per unit of total stake / liability."""
    gross = (1 - ff) - ff * (pr - 1)
    w = 1 / (pr - 1) if staking == "liability" else np.ones_like(pr)
    net = pd.Series(gross * w).groupby(race).sum().to_numpy()
    net = np.where(net > 0, net * (1 - c), net)
    return float(net.sum() / len(pr))  # each bet carries 1 unit of liability (or of stake)


def seg_mask(df: pd.DataFrame, seg: str) -> np.ndarray:
    if seg == "all":
        return np.ones(len(df), dtype=bool)
    if seg in ("nw2", "nw3"):
        return (df["n_winners"] == int(seg[-1])).to_numpy()
    return (df["seg"] == seg).to_numpy()


def n_months_of(df: pd.DataFrame) -> int:
    return int(df["month"].nunique()) if len(df) else 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data/raw/betfair/split"))
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    t0 = time.time()
    d = Path(a.dir)

    # sanity checks: tick ladder and batched Harville vs ponies.exotics.order_probs
    assert tick_longer(np.array([2.0]))[0] == 2.02 and tick_longer(np.array([50.0]))[0] == 55.0
    assert tick_longer(np.array([20.0]))[0] == 21.0 and tick_longer(np.array([1000.0]))[0] == 1000.0
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(4, 16))
        pp = rng.dirichlet(np.ones(n) * 0.7)
        for lam in (0.3, 0.8, 1.0, 1.4):
            for k in (2, 3):
                ref = order_probs(pp, lam, lam)["top3" if k == 3 else "top2"]
                worst = max(worst, float(np.abs(ref - topk_batch(pp[None, :], lam, k)[0]).max()))
    assert worst < 1e-9, worst
    out({"section": "checks", "harville_batch_vs_order_probs_max_abs": worst})

    frames, stats = [], {}
    for cc in COUNTRIES:
        j, st = load_country(cc, d, a.frac, a.seed)
        stats[cc] = st
        frames.append(j)
        out({"section": "data", "country": cc, **st})
    df = pd.concat(frames, ignore_index=True)
    del frames
    df["date"] = pd.to_datetime(df["date"]).values.astype("datetime64[D]")
    out({"section": "data_pooled", "races": int(df["race"].nunique()), "runners": int(len(df)),
         "date_range": [str(df["date"].min())[:10], str(df["date"].max())[:10]],
         "elapsed_s": round(time.time() - t0, 1)})

    # place BSP coverage by segment and year (all runners of the G6-clean universe)
    df["year"] = df["start"].dt.year
    cov = df.groupby(["seg", "year"])["bsp_pl"].apply(lambda s: s.notna().mean()).round(3)
    out({"section": "place_bsp_coverage_by_seg_year",
         "table": {f"{s}|{y}": float(v) for (s, y), v in cov.items()}})
    covr = df.groupby("seg")["bsp_pl"].apply(lambda s: s.notna().mean()).round(4)
    out({"section": "place_bsp_coverage_by_seg", "table": {k: float(v) for k, v in covr.items()}})

    cells = Cells()

    def record(rule, variant, thr, seg, settlement, staking, decision, row, eligible=True):
        row = {"rule": rule, "variant": variant, "threshold": thr, "segment": seg, "settlement": settlement,
               "staking": staking, "decision_time": decision, "eligible_rule": eligible, **row}
        cells.add(row)
        keys = ["rule", "variant", "threshold", "segment", "settlement", "staking", "decisions", "bets",
                "settled_share", "strike_over_implied", "roi_8pct", "se_clustered", "z", "halves_roi_8pct",
                "halves_se", "last12m_roi_8pct", "passes_G4"]
        out({"section": "cell", **{k: row.get(k) for k in keys}})

    # ------------------------------------------------------------------ RULE 1
    u_r1a = df[df["win_pre_all"] & df["win_pre_book"].between(BOOK_LO, BOOK_HI)]
    u_r1b = df[df["pl_pre_all"] & df["pl_pre_book"].between(BOOK_LO, BOOK_HI)]
    u_r1c = df[~df["g6_t30"]]
    universes = {"R1a_win_pre": (u_r1a, "pre_win", R1A_T, "pre_off"),
                 "R1b_place_pre": (u_r1b, "pre_pl", R1B_T, "pre_off"),
                 "R1c_win_t30": (u_r1c, "t30_win", R1C_T, "t30")}
    for name, (u, col, thrs, dec) in universes.items():
        out({"section": "universe", "rule": name, "races": int(u["race"].nunique()),
             "by_seg_races": u.drop_duplicates("race")["seg"].value_counts().to_dict(),
             "share_of_g6_clean_races": r4(u["race"].nunique() / df["race"].nunique())})
        for thr in thrs:
            sel_all = u[u[col] >= thr]
            for seg in SEGMENTS:
                um = seg_mask(u, seg)
                n_univ = int(u.loc[um, "race"].nunique())
                nm = n_months_of(u.loc[um])
                g = sel_all[seg_mask(sel_all, seg)]
                if len(g) == 0:
                    continue
                f = g["f_pl"].to_numpy()
                settlements = [("bsp", g["bsp_pl"].to_numpy(), {})]
                if name == "R1c_win_t30":
                    has_t30 = g["t30_pl"].notna().to_numpy()
                    L = np.where(has_t30, tick_longer(g["t30_pl"].fillna(2.0).to_numpy()), np.nan)
                    via_pre = has_t30 & (g["pre_pl"].to_numpy() <= L + 1e-9)
                    via_bsp = has_t30 & (g["bsp_pl"].to_numpy() <= L + 1e-9)
                    filled = via_pre | via_bsp
                    ent = np.where(filled, L, g["bsp_pl"].to_numpy())
                    lim_extra = {"share_with_place_t30": r4(has_t30.mean()), "fill_rate": r4(filled.mean()),
                                 "fill_rate_given_place_t30": r4(filled[has_t30].mean()) if has_t30.any() else None,
                                 "fill_via_place_ltp_pre": r4(via_pre.mean())}
                    settlements.append(("limit_then_bsp", ent, lim_extra))
                for sname, price, ex in settlements:
                    for staking in ("liability", "stake"):
                        row = eval_cell(g, price, f, staking, n_univ, nm, ex)
                        record(name, "main", thr, seg, sname, staking, dec, row)
            # missing-BSP treatment diagnostic (pooled, primary thresholds only)
            if (name, thr) in (("R1a_win_pre", 50), ("R1b_place_pre", 20), ("R1c_win_t30", 50)):
                for seg in ["all", "au_gallops", "au_harness", "gb", "ie"]:
                    g = sel_all[seg_mask(sel_all, seg)]
                    if len(g) == 0:
                        continue
                    hb = g["bsp_pl"].notna()
                    pw = (1 / g["pre_win"])
                    rec = {"section": "missing_bsp_check", "rule": name, "threshold": thr, "segment": seg,
                           "decisions": int(len(g)), "with_place_bsp": int(hb.sum()), "share_with_bsp": r4(hb.mean())}
                    for lab, m in (("with_bsp", hb), ("no_bsp", ~hb)):
                        if m.sum():
                            rec[lab] = {"n": int(m.sum()), "placed_rate": r4(g.loc[m, "f_pl"].mean()),
                                        "mean_1_over_win_ltp_pre": r4(pw[m].mean()),
                                        "mean_1_over_place_ltp_pre": r4((1 / g.loc[m, "pre_pl"]).mean()),
                                        "share_place_ltp_pre": r4(g.loc[m, "pre_pl"].notna().mean())}
                    out(rec)
                # how the place BSP compares with the place ltp_pre on the bets (descriptive)
                g = sel_all[sel_all["bsp_pl"].notna() & sel_all["pre_pl"].notna()]
                if len(g):
                    lr = np.log(g["bsp_pl"] / g["pre_pl"])
                    out({"section": "place_bsp_vs_place_ltp_pre", "rule": name, "threshold": thr, "n": int(len(g)),
                         "mean_log_bsp_over_ltp_pre": r4(lr.mean()), "share_bsp_longer": r4((lr > 1e-9).mean()),
                         "share_bsp_shorter": r4((lr < -1e-9).mean())})

    # descriptive reference: round-1 selection ON the place BSP (not achievable, G1 fails) and calibration
    # of the place BSP by pre-off WIN ltp_pre band (G1 universe of R1a)
    for seg in ["all", "au_gallops", "au_harness", "gb", "ie"]:
        g = df[seg_mask(df, seg) & (df["bsp_pl"] > 20).to_numpy()]
        if len(g):
            row = eval_cell(g, g["bsp_pl"].to_numpy(), g["f_pl"].to_numpy(), "liability",
                            int(df.loc[seg_mask(df, seg), "race"].nunique()), n_months_of(g))
            record("REF_place_bsp_gt20", "selected_on_bsp_not_achievable", 20, seg, "bsp", "liability", "post_off",
                   row, eligible=False)
    bands = [(1.01, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 200), (200, 1001)]
    tab = {}
    for lo, hi in bands:
        g = u_r1a[(u_r1a["pre_win"] >= lo) & (u_r1a["pre_win"] < hi) & u_r1a["bsp_pl"].notna()]
        if len(g):
            tab[f"[{lo},{hi})"] = {"n": int(len(g)), "strike": r4(g["f_pl"].mean()),
                                   "implied_place_bsp": r4((1 / g["bsp_pl"]).mean()),
                                   "strike_over_implied": r4(g["f_pl"].mean() / (1 / g["bsp_pl"]).mean())}
    out({"section": "place_bsp_calibration_by_win_ltp_pre_band", "universe": "R1a (WIN book G1)", "table": tab})
    tab = {}
    for lo, hi in [(1.01, 3), (3, 6), (6, 10), (10, 20), (20, 50), (50, 1001)]:
        g = u_r1b[(u_r1b["pre_pl"] >= lo) & (u_r1b["pre_pl"] < hi) & u_r1b["bsp_pl"].notna()]
        if len(g):
            tab[f"[{lo},{hi})"] = {"n": int(len(g)), "strike": r4(g["f_pl"].mean()),
                                   "implied_place_bsp": r4((1 / g["bsp_pl"]).mean()),
                                   "implied_place_ltp_pre_norm": r4((1 / g["pre_pl"] / g["pl_pre_book"]).mean()),
                                   "strike_over_implied_bsp": r4(g["f_pl"].mean() / (1 / g["bsp_pl"]).mean())}
    out({"section": "place_bsp_calibration_by_place_ltp_pre_band", "universe": "R1b (PLACE book G1)", "table": tab})
    # robustness of the pooled rule-1 cells that pass G4 (and the pre-registered thresholds): per calendar
    # year, a LIMIT_ON_CLOSE cap (lay at BSP only if the place BSP <= cap; an order the exchange supports),
    # and a looser / no PLACE-book filter for R1b. Per unit of liability (BSP lays are sized by liability).
    rob = [("R1b_place_pre", u_r1b, "pre_pl", 20), ("R1a_win_pre", u_r1a, "pre_win", 100),
           ("R1c_win_t30", u_r1c, "t30_win", 100), ("R1a_win_pre", u_r1a, "pre_win", 50),
           ("R1c_win_t30", u_r1c, "t30_win", 50)]
    for name, u, col, thr in rob:
        g = u[u[col] >= thr]
        n_univ, nm = int(u["race"].nunique()), n_months_of(u)
        yr = {}
        for y in sorted(g["year"].unique()):
            gy = g[g["year"] == y]
            row = eval_cell(gy, gy["bsp_pl"].to_numpy(), gy["f_pl"].to_numpy(), "liability", n_univ, nm)
            record(name, f"year_{y}", thr, "all", "bsp", "liability", "t30" if "t30" in name else "pre_off", row)
            yr[int(y)] = [row.get("bets"), row.get("roi_8pct"), row.get("se_clustered")]
        out({"section": "robust_by_year", "rule": name, "threshold": thr, "bets_roi8_se": yr})
        for cap in (50, 100):
            price = np.where(g["bsp_pl"].to_numpy() <= cap, g["bsp_pl"].to_numpy(), np.nan)
            row = eval_cell(g, price, g["f_pl"].to_numpy(), "liability", n_univ, nm)
            record(name, f"loc_cap_{cap}", thr, "all", "bsp_if_le_cap", "liability",
                   "t30" if "t30" in name else "pre_off", row)
        bk = {}
        for lo, hi in ((1.01, 30), (30, 50), (50, 100), (100, 200), (200, 1001)):
            m = (g["bsp_pl"] >= lo) & (g["bsp_pl"] < hi)
            gg = g[m]
            if len(gg) > 1:
                rr = lay_ret(gg["bsp_pl"].to_numpy(), gg["f_pl"].to_numpy(), PRIMARY_C, "liability")
                bk[f"[{lo},{hi})"] = [int(len(gg)), r4(gg["f_pl"].mean() / (1 / gg["bsp_pl"]).mean()), r6(rr.mean())]
        out({"section": "by_place_bsp_bucket_descriptive", "rule": name, "threshold": thr,
             "n_strike_over_implied_roi8_liab": bk})
    for lab, lo, hi in (("place_book_0.95_1.05", 0.95, 1.05), ("place_book_any", 0.0, 99.0)):
        u = df[df["pl_pre_all"] & df["pl_pre_book"].between(lo, hi)]
        g = u[u["pre_pl"] >= 20]
        for staking in ("liability", "stake"):
            row = eval_cell(g, g["bsp_pl"].to_numpy(), g["f_pl"].to_numpy(), staking, int(u["race"].nunique()),
                            n_months_of(u))
            record("R1b_place_pre", lab, 20, "all", "bsp", staking, "pre_off", row,
                   eligible=(lab != "place_book_any"))
    out({"section": "progress", "after": "rule1", "elapsed_s": round(time.time() - t0, 1)})

    # ------------------------------------------------------------------ RULE 2
    h = df[df["win_t30_all"]].copy()
    h["p"] = (1 / h["t30_win"]) / h["win_t30_book"]
    h = h.sort_values(["race", "p", "selection_id"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
    Q = harville_grid(h)
    lam, info = lambda_path(h, Q)
    out({"section": "lambda_path", "grid": [float(LAMBDA_GRID[0]), float(LAMBDA_GRID[-1]), 0.05], **info})
    key = list(zip(h["pool"], h["month"]))
    lam_r = np.array([lam.get(k, np.nan) for k in key])
    gi = np.round((np.nan_to_num(lam_r, nan=float(LAMBDA_GRID[0])) - LAMBDA_GRID[0]) / 0.05).astype(int)
    h["q_harv"] = np.where(np.isnan(lam_r), np.nan, Q[np.arange(len(h)), gi])
    del Q
    h = h[h["q_harv"].notna()].copy()
    h["q_mkt"] = np.where(h["pl_t30_all"], h["n_winners"] * (1 / h["t30_pl"]) / (h["pl_t30_book"] * h["n_winners"]),
                          np.nan)
    h["ratio"] = h["q_mkt"] / h["q_harv"]
    h["ratio_u"] = (1 / h["t30_pl"]) / h["q_harv"]
    h["drift"] = h["pre_win"] / h["t30_win"]
    h["p_win_pre"] = (1 / h["pre_win"]) / h["win_pre_book"]
    base = h[~h["g6_t30"] & h["win_pre_all"] & h["win_pre_book"].between(BOOK_LO, BOOK_HI)]
    u2 = base[base["pl_t30_all"]]
    u2u = base[base["t30_pl"].notna()]
    for nm_, u in (("R2_main_universe", u2), ("R2u_universe", u2u)):
        out({"section": "universe", "rule": nm_, "races": int(u["race"].nunique()),
             "by_seg_races": u.drop_duplicates("race")["seg"].value_counts().to_dict(),
             "date_range": [str(u["date"].min())[:10], str(u["date"].max())[:10]] if len(u) else None})
    # same-timestamp calibration diagnostic: hit/market by ratio bin (u2, ltp_pre <= 20)
    e = u2[u2["pre_win"] <= R2_CAP]
    tab = {}
    for lo, hi in [(0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.1), (1.1, 1.3), (1.3, 99)]:
        for dl, dm in (("drift>1.1", e["drift"] > DRIFT), ("no_drift", e["drift"] <= DRIFT)):
            g = e[(e["ratio"] >= lo) & (e["ratio"] < hi) & dm]
            if len(g):
                gb = g[g["bsp_win"].notna()]
                tab[f"[{lo},{hi})|{dl}"] = {"n": int(len(g)), "hit_over_p_ltp_pre": r4(g["f_win"].mean() / g["p_win_pre"].mean()),
                                            "hit_over_p_t30": r4(g["f_win"].mean() / g["p"].mean()),
                                            "hit_over_1_over_bsp": r4(gb["f_win"].mean() / (1 / gb["bsp_win"]).mean()) if len(gb) else None}
    out({"section": "R2_hit_over_market_by_ratio_bin", "table": tab})

    r2_variants = [("main", r, u2, lambda x, r=r: (x["ratio"] < r) & (x["drift"] > DRIFT) & (x["pre_win"] <= R2_CAP))
                   for r in R2_R]
    r2_variants += [("control_no_drift", 0.8, u2, lambda x: (x["ratio"] < 0.8) & (x["pre_win"] <= R2_CAP)),
                    ("control_drift_only", None, u2, lambda x: (x["drift"] > DRIFT) & (x["pre_win"] <= R2_CAP)),
                    ("unnormalised_place_t30", 0.8, u2u,
                     lambda x: (x["ratio_u"] < 0.8) & (x["drift"] > DRIFT) & (x["pre_win"] <= R2_CAP))]
    # round-1 replication at ltp_pre (cross-market, NOT simultaneous: PLACE ltp_pre falls back to the
    # scheduled start, WIN ltp_pre is cut at the in-play switch); both books in [0.97, 1.03] (G1), lambda
    # path refitted on ltp_pre prices (walk-forward, same grid)
    hp = df[df["win_pre_all"] & df["win_pre_book"].between(BOOK_LO, BOOK_HI) & df["pl_pre_all"]
            & df["pl_pre_book"].between(BOOK_LO, BOOK_HI)].copy()
    hp["p"] = (1 / hp["pre_win"]) / hp["win_pre_book"]
    hp = hp.sort_values(["race", "p", "selection_id"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
    Qp = harville_grid(hp)
    lamp, infop = lambda_path(hp, Qp)
    out({"section": "lambda_path_ltp_pre", **infop})
    lr = np.array([lamp.get(k, np.nan) for k in zip(hp["pool"], hp["month"])])
    gi = np.round((np.nan_to_num(lr, nan=float(LAMBDA_GRID[0])) - LAMBDA_GRID[0]) / 0.05).astype(int)
    hp["q_harv"] = np.where(np.isnan(lr), np.nan, Qp[np.arange(len(hp)), gi])
    del Qp
    hp = hp[hp["q_harv"].notna() & ~hp["g6_t30"]].copy()
    hp["ratio_pre"] = ((1 / hp["pre_pl"]) / hp["pl_pre_book"]) / hp["q_harv"]
    hp["drift"] = hp["pre_win"] / hp["t30_win"]
    hp["p_win_pre"] = hp["p"]
    out({"section": "universe", "rule": "R2_replication_ltp_pre_universe", "races": int(hp["race"].nunique()),
         "by_seg_races": hp.drop_duplicates("race")["seg"].value_counts().to_dict()})
    r2_variants += [("replication_ltp_pre", 0.8, hp,
                     lambda x: (x["ratio_pre"] < 0.8) & (x["drift"] > DRIFT) & (x["pre_win"] <= R2_CAP)),
                    ("replication_ltp_pre_no_drift", 0.8, hp,
                     lambda x: (x["ratio_pre"] < 0.8) & (x["pre_win"] <= R2_CAP))]
    for variant, thr, u, fn in r2_variants:
        sel_all = u[fn(u).to_numpy()]
        for seg in SEGMENTS:
            um = seg_mask(u, seg)
            g = sel_all[seg_mask(sel_all, seg)]
            if len(g) == 0:
                continue
            ex = {"hit_over_p_ltp_pre": r4(g["f_win"].mean() / g["p_win_pre"].mean()),
                  "hit_over_p_t30": r4(g["f_win"].mean() / ((1 / g["t30_win"]) / g["win_t30_book"]).mean())}
            for staking in ("liability", "stake"):
                row = eval_cell(g, g["bsp_win"].to_numpy(), g["f_win"].to_numpy(), staking,
                                int(u.loc[um, "race"].nunique()), n_months_of(u.loc[um]), ex)
                record("R2_place_dislike_drift_lay_win_bsp", variant, thr, seg, "bsp", staking, "pre_off", row)

    # ------------------------------------------------------------------ summary
    cdf = pd.DataFrame(cells.rows)
    pv = cdf["p_one_sided"].astype(float).to_numpy()
    okp = np.isfinite(pv)
    bh = np.zeros(len(cdf), dtype=bool)
    if okp.any():
        order = np.argsort(pv[okp])
        m = int(okp.sum())
        below = np.flatnonzero(pv[okp][order] <= 0.10 * np.arange(1, m + 1) / m)
        if len(below):
            bh[np.flatnonzero(okp)[order[: below.max() + 1]]] = True
    cdf["bh_q10_local"] = bh
    elig = cdf[cdf["eligible_rule"]]
    out({"section": "bh_local", "cells_with_p": int(okp.sum()), "rejected_q10": int(bh.sum()),
         "bonferroni_z_one_sided_5pct_this_script": r4(math.sqrt(2) * erfcinv(0.10 / int(okp.sum()))) if okp.any() else None})
    out({"section": "cell_count", "cells_total": int(len(cdf)), "cells_eligible_rules": int(len(elig)),
         "cells_descriptive_not_achievable": int((~cdf["eligible_rule"]).sum()),
         "cells_passing_G4": int(elig["passes_G4"].sum()),
         "passing": elig.loc[elig["passes_G4"], ["rule", "variant", "threshold", "segment", "settlement", "staking",
                                                 "roi_8pct", "se_clustered", "z", "bh_q10_local"]].to_dict("records")})
    if a.json:
        Path(a.json).write_text(json.dumps({"stats": stats, "cells": cells.rows},
                                           default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    out({"section": "done", "elapsed_s": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

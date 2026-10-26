"""Experiment "aligned_win_to_place" (round 2): win-derived fair PLACE price vs the
PLACE market at the SAME timestamp (ltp_t30, 30 minutes before the scheduled
start), for AU gallops, AU harness, GB and IE separately.

    python scripts/bf2_aligned_win_to_place.py [--dir data/raw/betfair/split] [--frac 0.2] [--json out.json]

Pipeline (everything fitted is walk-forward on strictly earlier months, G7):
  1. Pair each WIN market with its PLACE market (same event_id + start, identical
     runner sets, 2-4 places paid, at least n_winners + 2 runners).
     G6: drop races with a saddlecloth gap AND an ltp_pre book < 0.97 (WIN book,
     or PLACE book / n_winners when every place runner has an ltp_pre).
  2. Races whose WIN market has ltp_t30 for every runner: Shin-normalised win
     probabilities from ltp_t30 (ponies.odds.shin_probabilities, vectorised).
  3. Discounted Harville (ponies.exotics.order_probs; extended to 4 places with
     lam3 for 3rd and 4th): lam2/lam3 per segment fitted each month on earlier
     months by maximum likelihood of the observed (winner, placed set) on
     LAMBDA_GRID (pooled AU / GB+IE until a segment has 2000 earlier races).
  4. Walk-forward calibration per segment: logit(placed) = cubic in logit(p_fair)
     fitted on the previous 12 months (as scripts/bf_win_to_place.py), rescaled
     within race to n_winners.
  5. Aligned sample: races where BOTH markets have ltp_t30 for every runner.
     Market probability q = (1/place_ltp_t30) normalised to n_winners.
     Log-loss gain (market - model) per race, SE over races, halves.
  6. Bets decided at t30: back when place_t30 * p_cal > 1 + k, lay when < 1 - k
     (k = 0.05, 0.1, 0.2). Entry (G2): (a) limit one tick worse than place_t30
     (back: one tick shorter; lay: one tick longer), filled at the limit if the
     later place ltp_pre or place bsp reaches it, else place BSP; (b) plain place
     BSP. Lays per unit of backer's stake and per unit of liability. Commission on
     the net winnings of each bet, 0/5/8/10% (8% primary). Dead heats: stake
     share n_winners / n_placed settled as a winner. SE clustered by race and by
     race day (the larger is used); halves split at the segment's median race
     date; last 12 months 2019-11-08 .. 2020-11-08.
Prints summary numbers only (never horse names or rows).
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
from ponies.exotics import order_probs  # noqa: E402
from ponies.odds import shin_probabilities  # noqa: E402

COLS = ["market_id", "event_id", "market_name", "start", "date", "n_winners", "selection_id", "program",
        "ltp_pre", "ltp_t30", "bsp", "won"]
SEGMENTS = ["au_gallops", "au_harness", "gb", "ie"]
POOL = {"au_gallops": "au", "au_harness": "au", "gb": "gbie", "ie": "gbie"}
KS = [0.05, 0.1, 0.2]
COMMS = [0.0, 0.05, 0.08, 0.10]
PRIMARY_C = 0.08
LAST12_FROM = pd.Timestamp("2019-11-08")
MIN_FIT_RACES = 2000
MIN_CAL_RUNNERS = 5000
CAL_WINDOW = 12
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1000.0, 10.0)]
Z80 = 0.8416
# wider than ponies.exotics.LAMBDA_GRID (0.5-1.3): AU t30 fits hit 0.5 on the dev subset (G7: no grid-edge fits)
LAMBDA_GRID = np.round(np.arange(0.1, 1.61, 0.05), 2)


def out(obj) -> None:
    print(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)), flush=True)


def r4(x):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, 4)


# ---------------------------------------------------------------------------
# tick ladder
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
# loading / joining
# ---------------------------------------------------------------------------
def load_country(code: str, d: Path, frac: float, seed: int) -> tuple[pd.DataFrame, dict]:
    w = pd.read_parquet(d / f"{code}_win.parquet", columns=COLS).drop_duplicates(["market_id", "selection_id"])
    p = pd.read_parquet(d / f"{code}_place.parquet", columns=COLS).drop_duplicates(["market_id", "selection_id"])
    if frac < 1:
        ev = w["event_id"].drop_duplicates()
        keep = set(ev.sample(frac=frac, random_state=seed))
        w = w[w["event_id"].isin(keep)]
        p = p[p["event_id"].isin(keep)]
    stats = {"win_markets": int(w["market_id"].nunique()), "place_markets": int(p["market_id"].nunique())}
    w = w.copy()
    w["harness"] = w["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b", regex=True)
    wm = w.drop_duplicates("market_id")[["market_id", "event_id", "start"]]
    pm = p.drop_duplicates("market_id")[["market_id", "event_id", "start", "n_winners"]]
    pm = pm[pm["n_winners"].isin([2, 3, 4])]
    wm = wm[~wm.duplicated(["event_id", "start"], keep=False)]
    pm = pm[~pm.duplicated(["event_id", "start"], keep=False)]
    pairs = wm.rename(columns={"market_id": "win_mid"}).merge(
        pm.rename(columns={"market_id": "pl_mid"})[["event_id", "start", "pl_mid"]], on=["event_id", "start"])
    stats["paired_markets"] = int(len(pairs))
    w = w[w["market_id"].isin(pairs["win_mid"])].rename(columns={
        "market_id": "win_mid", "ltp_pre": "pre_win", "ltp_t30": "t30_win", "bsp": "bsp_win", "won": "won_win"})
    w = w.drop(columns=["market_name", "n_winners"])
    p = p[p["market_id"].isin(pairs["pl_mid"])].rename(columns={
        "market_id": "pl_mid", "ltp_pre": "pre_pl", "ltp_t30": "t30_pl", "bsp": "bsp_pl", "won": "placed"})
    p = p.merge(pairs[["pl_mid", "win_mid"]], on="pl_mid")
    p = p[["win_mid", "pl_mid", "selection_id", "n_winners", "pre_pl", "t30_pl", "bsp_pl", "placed"]]
    j = w.merge(p, on=["win_mid", "selection_id"], how="outer", indicator=True)
    bad = set(j.loc[j["_merge"] != "both", "win_mid"])
    j = j[~j["win_mid"].isin(bad)].drop(columns=["_merge"]).copy()
    stats["identical_runner_sets"] = int(j["win_mid"].nunique())
    g = j.groupby("win_mid")
    j["n"] = g["selection_id"].transform("size")
    j["n_placed"] = g["placed"].transform("sum")
    j["placed_nan"] = g["placed"].transform(lambda s: s.isna().any())
    keep = (~j["placed_nan"]) & (j["n_placed"] >= j["n_winners"]) & (j["n"] >= j["n_winners"] + 2)
    j = j[keep].copy()
    stats["settled_and_n_ge_nw_plus_2"] = int(j["win_mid"].nunique())
    j["harness"] = j["harness"].astype(bool)
    if code == "au":
        j["seg"] = np.where(j["harness"], "au_harness", "au_gallops")
    else:
        j["seg"] = code
    for c in ("pre_win", "t30_win", "bsp_win", "pre_pl", "t30_pl", "bsp_pl"):
        j.loc[~(j[c] > 1), c] = np.nan  # an exchange price must exceed 1
    g = j.groupby("win_mid")
    # G6: saddlecloth gap + ltp_pre book < 0.97
    prog = pd.to_numeric(j["program"], errors="coerce")
    j["_prog"] = prog
    pmax = g["_prog"].transform("max")
    pcnt = g["_prog"].transform("count")
    j["gap"] = (pcnt == j["n"]) & (pmax > j["n"])
    j["_iw"] = 1 / j["pre_win"]
    j["_ip"] = 1 / j["pre_pl"]
    j["win_pre_book"] = g["_iw"].transform("sum")
    j["pl_pre_all"] = g["pre_pl"].transform(lambda s: s.notna().all())
    j["pl_pre_book"] = g["_ip"].transform("sum") / j["n_winners"]
    j["g6_drop"] = j["gap"] & ((j["win_pre_book"] < 0.97) | (j["pl_pre_all"] & (j["pl_pre_book"] < 0.97)))
    j["all_t30_win"] = g["t30_win"].transform(lambda s: s.notna().all())
    j["all_t30_pl"] = g["t30_pl"].transform(lambda s: s.notna().all())
    j["_iw30"] = 1 / j["t30_win"]
    j["_ip30"] = 1 / j["t30_pl"]
    j["win_t30_book"] = g["_iw30"].transform("sum")
    j["pl_t30_book"] = g["_ip30"].transform("sum") / j["n_winners"]
    j["n_win_winners"] = g["won_win"].transform("sum")
    j = j.drop(columns=["_prog", "_iw", "_ip", "_iw30", "_ip30", "placed_nan", "program", "event_id"])
    return j, stats


# ---------------------------------------------------------------------------
# Shin (vectorised over races; identical to ponies.odds.shin_probabilities)
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


# ---------------------------------------------------------------------------
# discounted Harville
# ---------------------------------------------------------------------------
def place_probs(p: np.ndarray, l2: float, l3: float, nw: int) -> np.ndarray:
    """P(runner finishes in the first nw); 2nd ~ p^l2, 3rd and 4th ~ p^l3.
    For nw <= 3 this is ponies.exotics.order_probs top2/top3."""
    if nw in (2, 3):
        o = order_probs(p, l2, l3)
        return o["top2"] if nw == 2 else o["top3"]
    n = len(p)
    o = order_probs(p, l2, l3)
    tri = o["trifecta"]
    b = p ** l3
    D = b.sum() - b[:, None, None] - b[None, :, None] - b[None, None, :]
    T = np.where(tri > 0, tri / np.maximum(D, 1e-300), 0.0)
    excl = T.sum() - T.sum((1, 2)) - T.sum((0, 2)) - T.sum((0, 1))
    return o["top3"] + b * excl


def grid_loglik(P, jw, k, i, ok, nw):
    """Per-race log-likelihood of the observed placed set for every (lam2, lam3)
    on LAMBDA_GRID (copied from scripts/bf_win_to_place.py). Shape (G, G, R)."""
    R = P.shape[0]
    G = len(LAMBDA_GRID)
    ar = np.arange(R)
    kk, ii = np.clip(k, 0, None), np.clip(i, 0, None)
    A = np.empty((G, R)); aj = np.empty((G, R)); ak = np.empty((G, R)); ai = np.empty((G, R))
    for g, lam in enumerate(LAMBDA_GRID):
        a = P ** lam
        A[g] = a.sum(1); aj[g] = a[ar, jw]; ak[g] = a[ar, kk]; ai[g] = a[ar, ii]
    sec_k = ak / np.maximum(A - aj, 1e-300)
    sec_i = ai / np.maximum(A - aj, 1e-300)
    ll = np.zeros((G, G, R))
    two = ok & (nw == 2)
    three = ok & (nw == 3)
    ll[:, :, two] = np.log(np.maximum(sec_k[:, two], 1e-300))[:, None, :]
    thr_i = ai / np.maximum(A - aj - ak, 1e-300)
    thr_k = ak / np.maximum(A - aj - ai, 1e-300)
    for g3 in range(G):
        L = sec_k[:, three] * thr_i[g3, three][None, :] + sec_i[:, three] * thr_k[g3, three][None, :]
        ll[:, g3, three] = np.log(np.maximum(L, 1e-300))
    return ll


def pad(df: pd.DataFrame, col: str, R: int, W: int, fill=0.0) -> np.ndarray:
    M = np.full((R, W), fill, dtype=float)
    M[df["race"].to_numpy(), df["pos"].to_numpy()] = df[col].to_numpy(dtype=float)
    return M


def fit_and_derive(f: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """f: runners of races with complete WIN ltp_t30. Adds p_win (Shin t30),
    p_fair (walk-forward discounted Harville) and p_cal (walk-forward cubic)."""
    f = f.sort_values(["win_mid", "t30_win", "selection_id"]).reset_index(drop=True)
    f["race"] = pd.factorize(f["win_mid"])[0]
    f["pos"] = f.groupby("race").cumcount()
    R, W = int(f["race"].max()) + 1, int(f["pos"].max()) + 1
    O = np.full((R, W), np.nan)
    O[f["race"].to_numpy(), f["pos"].to_numpy()] = f["t30_win"].to_numpy()
    P = shin_matrix(O)
    f["p_win"] = P[f["race"].to_numpy(), f["pos"].to_numpy()]
    race = f.drop_duplicates("race").set_index("race").sort_index()
    nw = race["n_winners"].to_numpy()
    n_placed = race["n_placed"].to_numpy()
    won = pad(f, "won_win", R, W).astype(bool)
    plc = pad(f, "placed", R, W).astype(bool)
    jw = won.argmax(1)
    others = plc & ~won
    cnt = others.sum(1)
    k = np.where(cnt >= 1, others.argmax(1), -1)
    o2 = others.copy()
    o2[np.arange(R), np.clip(k, 0, None)] = False
    i = np.where(cnt >= 2, o2.argmax(1), -1)
    ok = ((race["n_win_winners"].to_numpy() == 1) & won[np.arange(R), jw] & plc[np.arange(R), jw]
          & (n_placed == nw) & (cnt == nw - 1) & np.isin(nw, [2, 3]))
    months = race["month"].to_numpy()
    segs = race["seg"].to_numpy()
    # per (segment, month) summed grid log-likelihood (computed in chunks to bound memory)
    G = len(LAMBDA_GRID)
    groups: dict = {}
    idx_all = np.flatnonzero(ok)
    for s in np.array_split(idx_all, max(1, len(idx_all) // 20000 + 1)):
        if len(s) == 0:
            continue
        ll = grid_loglik(P[s], jw[s], k[s], i[s], np.ones(len(s), dtype=bool), nw[s])
        key = pd.Series(list(zip(segs[s], months[s])))
        for kk, ix in key.groupby(key).indices.items():
            tot, c = groups.get(kk, (np.zeros((G, G)), 0))
            groups[kk] = (tot + ll[:, :, ix].sum(-1), c + len(ix))
        del ll
    months_sorted = sorted(set(months))
    lam = {}
    for m in months_sorted:
        for sg in SEGMENTS:
            tot = np.zeros((G, G)); n = 0
            for (ss, mm), (v, c) in groups.items():
                if mm < m and ss == sg:
                    tot += v; n += c
            src = sg
            if n < MIN_FIT_RACES:
                tot = np.zeros((G, G)); n = 0
                for (ss, mm), (v, c) in groups.items():
                    if mm < m and POOL[ss] == POOL[sg]:
                        tot += v; n += c
                src = POOL[sg]
            if n < MIN_FIT_RACES:
                continue
            g2, g3 = np.unravel_index(np.argmax(tot), tot.shape)
            lam[(sg, m)] = (float(LAMBDA_GRID[g2]), float(LAMBDA_GRID[g3]), n, src)
    # derived place probabilities
    Q = np.full((R, W), np.nan)
    nrun = (~np.isnan(O)).sum(1)
    for r in range(R):
        key = (segs[r], months[r])
        if key not in lam:
            continue
        l2, l3 = lam[key][0], lam[key][1]
        n = nrun[r]
        Q[r, :n] = place_probs(P[r, :n], l2, l3, int(nw[r]))
    f["p_fair"] = Q[f["race"].to_numpy(), f["pos"].to_numpy()]
    # walk-forward calibration per segment
    f["p_cal"] = np.nan
    cal_info = {}
    for sg in SEGMENTS:
        sel = (f["seg"] == sg) & f["p_fair"].notna()
        if not sel.any():
            continue
        fs = f[sel]
        z = np.log(fs["p_fair"].clip(1e-6, 1 - 1e-6) / (1 - fs["p_fair"].clip(1e-6, 1 - 1e-6))).to_numpy()
        y = fs["placed"].to_numpy(dtype=float)
        mo = fs["month"].to_numpy()
        ms = sorted(set(mo))
        pc = np.full(len(fs), np.nan)
        design = lambda zz: np.column_stack([np.ones(len(zz)), zz, zz ** 2, zz ** 3])
        params = {}
        for idx, m in enumerate(ms):
            tr = np.isin(mo, ms[max(0, idx - CAL_WINDOW):idx])
            if tr.sum() < MIN_CAL_RUNNERS:
                continue
            mu_z, sd_z = z[tr].mean(), z[tr].std()
            X = design((z[tr] - mu_z) / sd_z)
            wv = np.zeros(4)
            for _ in range(15):
                mu = 1 / (1 + np.exp(-(X @ wv)))
                gr = X.T @ (mu - y[tr])
                H = (X * (mu * (1 - mu))[:, None]).T @ X + 1e-6 * np.eye(4)
                wv = wv - np.linalg.solve(H, gr)
            params[m] = [round(float(v), 3) for v in wv]
            te = mo == m
            pc[te] = 1 / (1 + np.exp(-(design((z[te] - mu_z) / sd_z) @ wv)))
        f.loc[sel, "p_cal"] = pc
        cal_info[sg] = {"months_calibrated": len(params), "first": min(params) if params else None,
                        "params_sample": {m: params[m] for m in sorted(params)[::12]}}
    s = f.groupby("race")["p_cal"].transform("sum")
    f["p_cal"] = (f["p_cal"] * f["n_winners"] / s).clip(upper=0.999)
    # lambda summary
    lam_info = {}
    for sg in SEGMENTS:
        v = [(m, lam[(sg, m)]) for m in months_sorted if (sg, m) in lam]
        if not v:
            continue
        l2s = np.array([x[1][0] for x in v]); l3s = np.array([x[1][1] for x in v])
        lam_info[sg] = {"months": len(v), "first_month": v[0][0], "last": {"lam2": v[-1][1][0], "lam3": v[-1][1][1],
                        "fit_races": v[-1][1][2], "source": v[-1][1][3]},
                        "lam2_range": [float(l2s.min()), float(l2s.max())], "lam3_range": [float(l3s.min()), float(l3s.max())],
                        "share_at_grid_edge": r4(np.mean((l2s <= LAMBDA_GRID[0]) | (l2s >= LAMBDA_GRID[-1])
                                                         | (l3s <= LAMBDA_GRID[0]) | (l3s >= LAMBDA_GRID[-1]))),
                        "path_every_12m": [(m, x[0], x[1]) for m, x in v[::12]]}
    return f, {"lambdas": lam_info, "calibration": cal_info, "fit_usable_races": int(ok.sum())}


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


def roi_stats(r: np.ndarray, race: np.ndarray, date: np.ndarray) -> dict:
    n = len(r)
    if n < 2:
        return {"bets": int(n), "roi": r4(r.mean()) if n else None, "se": None, "se_race": None, "se_day": None}
    se_race = cluster_se(r, race)
    se_day = cluster_se(r, date)
    se = max(x for x in (se_race, se_day, r.std(ddof=1) / math.sqrt(n)) if x is not None)
    return {"bets": int(n), "roi": r4(r.mean()), "se": r4(se), "se_bet": r4(r.std(ddof=1) / math.sqrt(n)),
            "se_race": r4(se_race), "se_day": r4(se_day)}


def back_net(price, placed, dh, c):
    gross = np.where(placed, dh * (price - 1) - (1 - dh), -1.0)
    return np.where(gross > 0, gross * (1 - c), gross)


def lay_net(price, placed, dh, c):
    gross = np.where(placed, -(dh * (price - 1) - (1 - dh)), 1.0)
    return np.where(gross > 0, gross * (1 - c), gross)


def logloss_race(q, y, rc):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.bincount(rc, weights=-(y * np.log(q) + (1 - y) * np.log(1 - q)))


def gain_stats(ll_model, ll_mkt, first_half, last12) -> dict:
    d = ll_mkt - ll_model
    res = {"races": int(len(d)), "ll_market": r4(ll_mkt.mean()), "ll_model": r4(ll_model.mean()),
           "gain": r4(d.mean()), "se": r4(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None}
    res["halves_gain"] = [r4(d[first_half].mean()), r4(d[~first_half].mean())]
    res["halves_se"] = [r4(d[first_half].std(ddof=1) / math.sqrt(max(first_half.sum(), 1))),
                        r4(d[~first_half].std(ddof=1) / math.sqrt(max((~first_half).sum(), 1)))]
    res["last12m_gain"] = r4(d[last12].mean()) if last12.any() else None
    res["z"] = r4(res["gain"] / res["se"]) if res["se"] else None
    return res


# ---------------------------------------------------------------------------
PRICE_BANDS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 35.0, 60.0, 1001.0]


def blind_band_means(a: pd.DataFrame) -> dict:
    """Mean 8%-commission return of EVERY runner at the place BSP per t30 price band
    (back per stake, lay per stake, lay per liability): the price-matched blind baseline."""
    band = np.digitize(a["t30_pl"].to_numpy(), PRICE_BANDS) - 1
    bb = a["bsp_pl"].to_numpy()
    pl = a["placed"].to_numpy().astype(bool)
    dh = (a["n_winners"] / a["n_placed"]).to_numpy()
    ok = ~np.isnan(bb)
    rb = np.full(len(a), np.nan); rl = np.full(len(a), np.nan); rll = np.full(len(a), np.nan)
    rb[ok] = back_net(bb[ok], pl[ok], dh[ok], PRIMARY_C)
    rl[ok] = lay_net(bb[ok], pl[ok], dh[ok], PRIMARY_C)
    rll[ok] = rl[ok] / (bb[ok] - 1)
    res = {}
    for name, r in (("back", rb), ("lay", rl), ("lay_liab", rll)):
        res[name] = pd.Series(r).groupby(band).mean().to_dict()
    return res


def evaluate_rule(a: pd.DataFrame, sel: np.ndarray, side: str, k: float, seg: str, mid, n_races: int,
                  n_months: int, base: dict | None = None) -> list[dict]:
    """Return one cell per settlement (and staking for lays)."""
    g = a[sel]
    gband = np.digitize(g["t30_pl"].to_numpy(), PRICE_BANDS) - 1
    cells = []
    if side == "back":
        L = tick_shorter(g["t30_pl"].to_numpy())
        filled = (g["pre_pl"].to_numpy() >= L - 1e-9) | (g["bsp_pl"].to_numpy() >= L - 1e-9)
    else:
        L = tick_longer(g["t30_pl"].to_numpy())
        filled = (g["pre_pl"].to_numpy() <= L + 1e-9) | (g["bsp_pl"].to_numpy() <= L + 1e-9)
    bsp = g["bsp_pl"].to_numpy()
    entry_limit = np.where(filled, L, bsp)
    t30 = g["t30_pl"].to_numpy()
    settle_defs = [("limit_then_bsp", entry_limit), ("bsp", bsp)]
    stakings = ["stake"] if side == "back" else ["stake", "liability"]
    placed = g["placed"].to_numpy().astype(bool)
    dh = (g["n_winners"] / g["n_placed"]).to_numpy()
    race = g["win_mid"].to_numpy()
    date = g["date"].to_numpy()
    fh = (g["date"] < mid).to_numpy()
    l12 = (g["date"] >= LAST12_FROM).to_numpy()
    for sname, price in settle_defs:
        ok = ~np.isnan(price)
        for stk in stakings:
            res = {"segment": seg, "side": side, "k": k, "settlement": sname if stk == "stake" else
                   ("per_liability_limit_then_bsp" if sname == "limit_then_bsp" else "per_liability_bsp"),
                   "decisions": int(len(g)), "settled_share": r4(ok.mean()) if len(g) else None}
            pr = price[ok]
            rets = {}
            for c in COMMS:
                rr = back_net(pr, placed[ok], dh[ok], c) if side == "back" else lay_net(pr, placed[ok], dh[ok], c)
                if stk == "liability":
                    rr = rr / (pr - 1)
                rets[c] = rr
            r8 = rets[PRIMARY_C]
            st = roi_stats(r8, race[ok], date[ok])
            res.update({"bets": st["bets"], "roi_8pct": st["roi"], "se_clustered": st["se"], "se_bet": st.get("se_bet"),
                        "se_race": st.get("se_race"), "se_day": st.get("se_day")})
            res["roi_0_5_10pct"] = [r4(rets[c].mean()) if len(rets[c]) else None for c in (0.0, 0.05, 0.10)]
            h1 = roi_stats(r8[fh[ok]], race[ok][fh[ok]], date[ok][fh[ok]])
            h2 = roi_stats(r8[~fh[ok]], race[ok][~fh[ok]], date[ok][~fh[ok]])
            res["halves_roi_8pct"] = [h1["roi"], h2["roi"]]
            res["halves_se"] = [h1["se"], h2["se"]]
            res["halves_bets"] = [h1["bets"], h2["bets"]]
            lt = roi_stats(r8[l12[ok]], race[ok][l12[ok]], date[ok][l12[ok]])
            res["last12m_roi_8pct"] = lt["roi"]
            res["last12m_bets"] = lt["bets"]
            res["z"] = r4(st["roi"] / st["se"]) if st["se"] else None
            passes = (h1["se"] is not None and h2["se"] is not None and h1["roi"] > 2 * h1["se"]
                      and h2["roi"] > 2 * h2["se"] and lt["roi"] is not None and lt["roi"] > 0)
            res["passes_G4"] = bool(passes)
            res["mde_80pct_full"] = r4((2 + Z80) * st["se"]) if st["se"] else None
            res["mde_80pct_each_half"] = r4((2 + Z80) * max(h1["se"], h2["se"])) if h1["se"] and h2["se"] else None
            res["fires_per_race"] = r4(len(g) / n_races) if n_races else None
            res["bets_per_month"] = r4(len(g) / n_months) if n_months else None
            if sname == "limit_then_bsp":
                res["fill_rate"] = r4(filled.mean()) if len(g) else None
            # closing-line value: entry price / place BSP (back: > 1 good; lay: < 1 good)
            okb = ok & ~np.isnan(bsp)
            if okb.any():
                res["clv_entry_over_bsp_mean"] = r4(np.mean(price[okb] / bsp[okb]))
                res["clv_mean_log_entry_over_bsp"] = r4(np.mean(np.log(price[okb] / bsp[okb])))
                if sname == "limit_then_bsp" and (okb & filled).any():
                    res["clv_filled_only_mean"] = r4(np.mean(price[okb & filled] / bsp[okb & filled]))
                res["t30_over_bsp_median"] = r4(np.median(t30[okb] / bsp[okb]))
                res["t30_over_bsp_mean_log"] = r4(np.mean(np.log(t30[okb] / bsp[okb])))
            # paper (NOT achievable) reference: settled at the t30 quote itself
            if stk == "stake":
                rp = back_net(t30, placed, dh, PRIMARY_C) if side == "back" else lay_net(t30, placed, dh, PRIMARY_C)
            else:
                rp = (lay_net(t30, placed, dh, PRIMARY_C)) / (t30 - 1)
            res["paper_roi_8pct_at_t30_quote"] = r4(rp.mean()) if len(rp) else None
            if base is not None and sname == "bsp" and ok.any():
                key = "back" if side == "back" else ("lay" if stk == "stake" else "lay_liab")
                bm = np.array([base[key].get(b, np.nan) for b in gband[ok]])
                res["price_matched_blind_roi_8pct"] = r4(np.nanmean(bm))
            # robustness: one huge (illiquid) BSP can dominate a small cell
            if len(r8) > 2:
                top = np.argmax(r8)
                res["roi_8pct_excl_top_bet"] = r4(np.delete(r8, top).mean())
                res["max_entry_price"] = r4(pr.max())
            res["strike"] = r4(placed[ok].mean()) if ok.any() else None
            res["mean_p_cal"] = r4(g["p_cal"].to_numpy()[ok].mean()) if ok.any() else None
            res["median_entry_price"] = r4(np.median(pr)) if len(pr) else None
            cells.append(res)
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "data/raw/betfair/split")
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of events (development)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None, help="write claims/cells JSON here")
    a = ap.parse_args()
    t0 = time.time()

    parts, load_stats = [], {}
    for code in ("au", "gb", "ie"):
        j, st = load_country(code, a.dir, a.frac, a.seed)
        load_stats[code] = st
        parts.append(j)
    j = pd.concat(parts, ignore_index=True)
    del parts
    j["date"] = pd.to_datetime(j["date"])
    j["month"] = j["date"].dt.to_period("M").astype(str)

    # ---- sample accounting per segment (races)
    rl = j.drop_duplicates("win_mid")
    share = {}
    for sg in SEGMENTS:
        r = rl[rl["seg"] == sg]
        share[sg] = {"joined_races": int(len(r)), "g6_dropped": int(r["g6_drop"].sum()),
                     "gap_races": int(r["gap"].sum()),
                     "all_win_t30": int(r["all_t30_win"].sum()), "all_place_t30": int(r["all_t30_pl"].sum()),
                     "both_all_t30": int((r["all_t30_win"] & r["all_t30_pl"]).sum()),
                     "both_all_t30_after_g6": int((r["all_t30_win"] & r["all_t30_pl"] & ~r["g6_drop"]).sum()),
                     "share_both_all_t30": r4((r["all_t30_win"] & r["all_t30_pl"]).mean()),
                     "share_both_all_t30_after_g6": r4((r["all_t30_win"] & r["all_t30_pl"] & ~r["g6_drop"]).mean())}
    out({"section": "data", "load": load_stats, "races": share, "runtime_s": round(time.time() - t0, 1)})

    # ---- fit set: complete WIN t30, not dropped by G6
    f = j[j["all_t30_win"] & ~j["g6_drop"]].copy()
    del j
    f, fit_info = fit_and_derive(f)
    out({"section": "fit", **fit_info, "runtime_s": round(time.time() - t0, 1)})

    # ---- aligned sample: complete PLACE t30 too, p_cal available
    f["has_cal"] = f.groupby("race")["p_cal"].transform(lambda s: s.notna().all())
    al = f[f["all_t30_pl"] & f["has_cal"]].copy()
    al["q_mkt"] = (1 / al["t30_pl"]) * al["n_winners"] / al.groupby("race")["t30_pl"].transform(lambda s: (1 / s).sum())
    al["all_bsp_pl"] = al.groupby("race")["bsp_pl"].transform(lambda s: s.notna().all())

    prob_results, all_cells, stale = {}, [], {}
    seg_meta = {}
    for sg in SEGMENTS:
        s = al[al["seg"] == sg].copy()
        if s.empty:
            continue
        rdates = s.drop_duplicates("race")["date"]
        mid = rdates.sort_values().iloc[len(rdates) // 2]
        n_races = int(s["race"].nunique())
        n_months = int(s["month"].nunique())
        rl = s.drop_duplicates("race")
        seg_meta[sg] = {"aligned_races": n_races, "runners": int(len(s)), "months": n_months,
                        "from": str(rdates.min().date()), "to": str(rdates.max().date()), "median_date": str(mid.date()),
                        "races_by_year": rl["date"].dt.year.value_counts().sort_index().to_dict(),
                        "n_winners": rl["n_winners"].value_counts().sort_index().to_dict(),
                        "median_field": float(rl["n"].median()),
                        "win_t30_book_median": r4(rl["win_t30_book"].median()),
                        "win_t30_book_p05_p95": [r4(rl["win_t30_book"].quantile(0.05)), r4(rl["win_t30_book"].quantile(0.95))],
                        "place_t30_book_over_nw_median": r4(rl["pl_t30_book"].median()),
                        "place_t30_book_over_nw_p05_p95": [r4(rl["pl_t30_book"].quantile(0.05)), r4(rl["pl_t30_book"].quantile(0.95))],
                        "share_races_place_t30_book_lt_0.97": r4((rl["pl_t30_book"] < 0.97).mean()),
                        "share_races_place_t30_book_gt_1.10": r4((rl["pl_t30_book"] > 1.10).mean()),
                        "place_bsp_coverage_runners": r4(s["bsp_pl"].notna().mean())}
        # ---- log loss at t30
        s = s.sort_values(["race", "pos"])
        rc = pd.factorize(s["race"])[0]
        y = s["placed"].to_numpy(dtype=float)
        rd = s.drop_duplicates("race")["date"].to_numpy()
        fh = rd < np.datetime64(mid)
        l12 = rd >= np.datetime64(LAST12_FROM)
        ll_m = logloss_race(s["q_mkt"].to_numpy(), y, rc)
        ll_c = logloss_race(s["p_cal"].to_numpy(), y, rc)
        ll_f = logloss_race(s["p_fair"].to_numpy(), y, rc)
        pr = {"p_cal_vs_place_t30": gain_stats(ll_c, ll_m, fh, l12),
              "p_fair_vs_place_t30": gain_stats(ll_f, ll_m, fh, l12)}
        # gain by place-book tightness at t30 (decision-time variable)
        rb = s.drop_duplicates("race")["pl_t30_book"].to_numpy()
        rows = []
        for lo, hi in [(0, 0.97), (0.97, 1.03), (1.03, 1.1), (1.1, 99)]:
            m = (rb >= lo) & (rb < hi)
            if m.sum() > 30:
                d = ll_m[m] - ll_c[m]
                rows.append({"place_t30_book_over_nw": f"{lo}-{hi}", "races": int(m.sum()), "gain": r4(d.mean()),
                             "se": r4(d.std(ddof=1) / math.sqrt(m.sum()))})
        pr["p_cal_gain_by_place_t30_book"] = rows
        # post-off benchmark (not a feature): place BSP-implied vs place t30
        sb = s["all_bsp_pl"].to_numpy()
        rsb = s.drop_duplicates("race")["all_bsp_pl"].to_numpy()
        if rsb.sum() > 30:
            s2 = s[sb].copy()
            s2["q_bsp"] = (1 / s2["bsp_pl"]) * s2["n_winners"] / s2.groupby("race")["bsp_pl"].transform(lambda v: (1 / v).sum())
            rc2 = pd.factorize(s2["race"])[0]
            y2 = s2["placed"].to_numpy(dtype=float)
            ll_b = logloss_race(s2["q_bsp"].to_numpy(), y2, rc2)
            d = ll_m[rsb] - ll_b
            pr["benchmark_place_bsp_vs_place_t30"] = {"races": int(len(d)), "gain": r4(d.mean()),
                                                      "se": r4(d.std(ddof=1) / math.sqrt(len(d)))}
            d2 = ll_b - ll_c[rsb]
            pr["benchmark_p_cal_vs_place_bsp"] = {"races": int(len(d2)), "gain": r4(d2.mean()),
                                                  "se": r4(d2.std(ddof=1) / math.sqrt(len(d2)))}
        prob_results[sg] = pr

        # ---- bets
        edge = (s["t30_pl"] * s["p_cal"] - 1).to_numpy()
        st_rows = {}
        okall = s["bsp_pl"].notna().to_numpy()
        ratio_all = (s["bsp_pl"] / s["t30_pl"]).to_numpy()
        st_rows["all_runners"] = {"runners": int(len(s)), "median_bsp_over_t30": r4(np.nanmedian(ratio_all)),
                                  "mean_log_bsp_over_t30": r4(np.nanmean(np.log(ratio_all))),
                                  "share_bsp_lt_t30": r4((ratio_all[okall] < 1).mean()),
                                  "share_pre_eq_t30": r4((s["pre_pl"] == s["t30_pl"]).mean()),
                                  "median_place_t30_book_of_race": r4(s["pl_t30_book"].median()),
                                  "median_win_bsp_over_t30": r4((s["bsp_win"] / s["t30_win"]).median())}
        # blind baselines (descriptive): every runner backed / laid (per liability) at the place BSP
        pl_all = s["placed"].to_numpy().astype(bool)
        dh_all = (s["n_winners"] / s["n_placed"]).to_numpy()
        bb = s["bsp_pl"].to_numpy()
        okb = ~np.isnan(bb)
        rb8 = back_net(bb[okb], pl_all[okb], dh_all[okb], PRIMARY_C)
        rl8 = lay_net(bb[okb], pl_all[okb], dh_all[okb], PRIMARY_C) / (bb[okb] - 1)
        rcl, rdt = s["win_mid"].to_numpy()[okb], s["date"].to_numpy()[okb]
        st_rows["blind_back_all_at_bsp_c8"] = roi_stats(rb8, rcl, rdt)
        st_rows["blind_lay_all_at_bsp_per_liability_c8"] = roi_stats(rl8, rcl, rdt)
        base = blind_band_means(s)
        for side in ("back", "lay"):
            for kk in KS:
                sel = edge > kk if side == "back" else edge < -kk
                cells = evaluate_rule(s, sel, side, kk, sg, mid, n_races, n_months, base)
                all_cells += cells
                g = s[sel]
                rr = (g["bsp_pl"] / g["t30_pl"]).to_numpy()
                okg = ~np.isnan(rr)
                st_rows[f"{side}_k{kk:g}"] = {"runners": int(len(g)),
                                             "median_bsp_over_t30": r4(np.median(rr[okg])) if okg.any() else None,
                                             "mean_log_bsp_over_t30": r4(np.mean(np.log(rr[okg]))) if okg.any() else None,
                                             "share_bsp_lt_t30": r4((rr[okg] < 1).mean()) if okg.any() else None,
                                             "share_pre_eq_t30": r4((g["pre_pl"] == g["t30_pl"]).mean()) if len(g) else None,
                                             "median_place_t30_book_of_race": r4(g["pl_t30_book"].median()) if len(g) else None,
                                             "median_win_bsp_over_t30": r4((g["bsp_win"] / g["t30_win"]).median()) if len(g) else None,
                                             "median_t30_price": r4(g["t30_pl"].median()) if len(g) else None}
        stale[sg] = st_rows

    out({"section": "aligned_sample", "segments": seg_meta})
    out({"section": "logloss_t30", "results": prob_results})
    out({"section": "stale_quote_check_place_bsp_over_t30", "results": stale})
    keys = ["segment", "side", "k", "settlement", "decisions", "bets", "roi_8pct", "se_clustered", "halves_roi_8pct",
            "halves_se", "last12m_roi_8pct", "roi_0_5_10pct", "z", "passes_G4", "fill_rate", "clv_entry_over_bsp_mean",
            "clv_mean_log_entry_over_bsp", "clv_filled_only_mean", "t30_over_bsp_median", "paper_roi_8pct_at_t30_quote", "mde_80pct_full", "fires_per_race", "strike", "mean_p_cal",
            "median_entry_price", "settled_share", "roi_8pct_excl_top_bet", "max_entry_price",
            "price_matched_blind_roi_8pct", "halves_bets", "last12m_bets", "se_race", "se_day", "mde_80pct_each_half",
            "bets_per_month"]
    for c in all_cells:
        out({"cell": {kk: c.get(kk) for kk in keys}})
    n_prob = sum(2 for _ in prob_results)
    out({"section": "totals", "bet_cells": len(all_cells), "probability_cells": n_prob,
         "cells_passing_G4": [(c["segment"], c["side"], c["k"], c["settlement"]) for c in all_cells if c["passes_G4"]],
         "runtime_s": round(time.time() - t0, 1)})
    if a.json:
        a.json.write_text(json.dumps({"cells": all_cells, "prob": prob_results, "stale": stale, "segments": seg_meta,
                                      "share": share, "fit": fit_info},
                                     default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

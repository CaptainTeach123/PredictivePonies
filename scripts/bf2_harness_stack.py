"""Experiment "harness_stack" (round 2): one walk-forward AU harness conditional logit that
stacks every persistent miscalibration found in round 1 on top of the exchange price, with
NZ harness as an out-of-sample replication of the AU-fitted model.

    python scripts/bf2_harness_stack.py [--dev] [--workers 3]

Data: data/raw/betfair/split/{au,nz}_{win,place}.parquet (licensed, local only). Prints JSON
summary lines only; never horse names or rows. Venue names (public) are printed for the
structural gate table.

UNIVERSE  AU harness WIN markets (market_name contains Pace or Trot), deduplicated on
(market_id, selection_id), settled (won known for every runner, >= 1 winner), every runner
traded pre-off (ltp_pre > 1), ltp_pre book sum(1/ltp_pre) in [0.97, 1.03] (G1; this also
removes every G6 market, gap + book < 0.97), saddlecloths numeric and unique, >= 4 runners.

FEATURES (all known before the off)
  lp_pre        log Shin probability at ltp_pre (unpenalised).
  lp30          log Shin probability at ltp_t30 when the whole field traded at t30 (t30 book in
                [0.95, 1.10]); when only part of the field traded at t30, the t30 implied
                probabilities of the traded runners renormalised to the ltp_pre mass of the same
                runners (a within-subset move); runners with no t30 trade: lp30 = lp_pre and
                t30_missing = 1.
  gate          mobile starts only ('M' suffix; standing starts get no gate features). Gate rule
                (worked out from the gaps, see section gate_rule): the front row is saddlecloths
                1..W (W per venue, estimated from the market's price jump between saddlecloth W
                and W+1 in gap-free burn-in fields), front-row runners drawn outside a scratched
                front-row runner move in one gate per scratching (front gate = rank of the
                saddlecloth among the remaining front-row saddlecloths); the second row is never
                promoted (second-row position = rank among the remaining saddlecloths > W), and
                an emergency (saddlecloth above the field) therefore starts on the outside of the
                second row. Dummies: front gates 2..8, 9+ and second-row positions 1, 2, 3+ (ref =
                front gate 1); gate groups {2-3, 4-6, 7+, S1, S2+} x distance band (<1800 m,
                >2200 m; ref 1800-2200 m, metres parsed from market_name) and x venue for the 10
                biggest burn-in venues (others pooled = reference).
  history       horse key = lower-cased stripped name (selection_id is NOT persistent across
                races in this archive), per country, every settled harness run: log1p(days since
                the previous run, capped at 365), back-up flag (previous run <= 7 days earlier),
                no_prior, log1p(prior runs), cum_resid = sum over previous runs of (won - market
                probability at BSP, else ltp_pre) / (prior runs + 5). Only runs with start
                strictly earlier (asserted) are used.
  place ladder  presid30 = ln(q_place / q_harville) at t30: q_place = n_winners * de-margined
                place-market t30 probability, q_harville = plain Harville top-2/top-3 from the
                Shin WIN probabilities at t30; needs every runner traded at t30 in both markets
                (books in [0.95, 1.10]); else presid30 = 0 and presid30_missing = 1.
  EXPLORATORY   (variant STACKX, not pre-registered; added because the t30 residual exists for
                ~0.3% of races) presid_pre: the same residual at ltp_pre, which for AU harness is
                cut at the scheduled start in BOTH the WIN and the PLACE market (place book in
                [0.95, 1.10]); else presid_pre_missing = 1.

MODELS  MKT (lp_pre, lp30, t30_missing: market-only recalibration), STACK (all features above
except presid_pre), STACKX (STACK + presid_pre). Objective = ponies.model.ConditionalLogit's
(mean per-race log loss + 0.5 * l2 * |w|^2 on training-standardised features, lp_pre
unpenalised, dead-heat shares as targets), solved with L-BFGS-B (checked against the repo's
gradient-descent fit in section fit_check).

WALK-FORWARD (G7)  Structural choices (W per venue, top-10 venues) use only the burn-in months
2016-12..2017-05. For every l2 on the grid and every month m from 2017-06 to 2020-11 the model
is fitted on all races starting strictly before the first day of m and predicts m (warm start
from the previous month). The l2 used for month m is the grid value with the lowest per-race
log loss over the out-of-sample predictions of months m-6..m-1, so the evaluation (test)
period is 2017-12..2020-11. NZ month m is predicted by the AU fit used for AU month m.

BETS  back 1 unit at BSP (decision taken pre-off with ltp_pre information; settled at BSP; the
BSP itself is never a feature) when p_model / p_mkt - 1 > k, p_mkt = Shin at ltp_pre, k in
{0.05, 0.10, 0.15, 0.20, 0.30}; flat intersection rules: front gate 4-6 AND back-up (<= 7 days);
front gate 4-6 AND presid30 > 0 (place ladder likes it); exploratory: front gate 4-6 AND
presid_pre > 0. Return per unit stake: net = f(P-1) - (1-f), f = dead_heat_share; commission
on net winnings only. ROI at c in {0, 5, 8, 10}%; SE clustered by race day (the gate) and by
market; halves split at the median race date of the AU test races; last 12 months
2019-11-08..2020-11-08; MDE (80% power, two-sided 5%) = 2.80 * clustered SE.

Sections (every line is JSON): run, data, gate_rule, gate_width, history, place_join, fit_check,
l2_path, refit_summary, coef, logloss, calibration, bet_cell, primary, g5_count, caveat, done.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ponies.model import ConditionalLogit, race_softmax, win_targets, _race_codes  # noqa: E402
from ponies.odds import shin_probabilities  # noqa: E402
from ponies.exotics import order_probs  # noqa: E402

DATA = REPO / "data" / "raw" / "betfair" / "split"
WIN_COLS = ["market_id", "event_id", "venue", "market_name", "start", "date", "selection_id", "program",
            "horse_name", "ltp_pre", "ltp_t30", "bsp", "won", "dead_heat_share"]
PL_COLS = ["market_id", "event_id", "start", "n_winners", "selection_id", "ltp_pre", "ltp_t30", "won"]
HARNESS_RE = r"\b(?:Pace|Trot)\b"
MOBILE_RE = r"\b(?:Pace|Trot)\s+M\s*$"
DIST_RE = r"(\d{3,4})\s*[mM]?\s+(?:Pace|Trot)"
COMMISSIONS = (0.0, 0.05, 0.08, 0.10)
C_MAIN = 0.08
K_VALUES = (0.05, 0.10, 0.15, 0.20, 0.30)
L2_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
BOOK_LO, BOOK_HI = 0.97, 1.03
T30_BOOK = (0.95, 1.10)
PL_BOOK = (0.95, 1.10)
SHRINK = 5.0
BACKUP_DAYS = 7.0
BURN_END = pd.Timestamp("2017-06-01", tz="UTC")
OOS_FIRST = pd.Period("2017-06", "M")
OOS_LAST = pd.Period("2020-11", "M")
L2_WINDOW = 6
LAST12 = (pd.Timestamp("2019-11-08", tz="UTC"), pd.Timestamp("2020-11-09", tz="UTC"))
W_MIN_JUMP = 0.35
W_MIN_RACES = 40
N_TOP_VENUES = 10
GROUPS = ("g23", "g46", "g7p", "s1", "s2p")
FRONT_DUMMIES = ["G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9p"]
SECOND_DUMMIES = ["S1", "S2", "S3p"]
HIST_FEATURES = ["cum_resid", "log_days", "backup7", "no_prior", "log_n_prior"]
MKT_FEATURES = ["lp_pre", "lp30", "t30_missing"]
PLACE30_FEATURES = ["presid30", "presid30_missing"]
PLACEPRE_FEATURES = ["presid_pre", "presid_pre_missing"]

CAVEATS = [
    "Saddlecloth order is the barrier draw only where the venue numbers runners by barrier: the market prices a clear "
    "pole / second-row discontinuity at Queensland, WA, SA, Tasmanian and some NSW country venues, but not at Menangle, "
    "Newcastle, Bathurst or (from 2019) most Victorian venues, nor in NZ; there the 'gate' features measure saddlecloth "
    "order, and the venue interactions are what lets the model treat them differently.",
    "Gate rule (front-row move-in, second row never promoted, emergencies outside the second row) is supported by the "
    "market's prices at the venues with clear barrier coding; elsewhere it is an assumption.",
    "The simultaneous (t30) place-ladder residual exists for ~0.3% of AU harness races, so the spec's place-ladder "
    "feature and the 'gate 4-6 AND place ladder likes it' rule are nearly empty; STACKX's ltp_pre residual is an "
    "exploratory, non-pre-registered substitute (both markets cut at the scheduled start for AU harness).",
    "No t30-decision model: only ~5% of AU harness WIN markets have every runner traded at t30, so every bet here is a "
    "pre-off decision settled at BSP (no limit-order leg).",
    "NZ harness has a BSP for ~0.6% of runners, so the NZ bet replication at BSP is nearly empty; NZ bets at ltp_pre are "
    "paper prices (not an achievable settlement under G2) and are reported for direction only.",
    "Horse key = name: different horses sharing a name are merged; the archive starts 2016-12, so early history is "
    "truncated.",
]


# ----------------------------------------------------------------------------- utilities


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, pd.Period)):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def out(obj) -> None:
    print(json.dumps(obj, default=_json_default), flush=True)


def r5(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, 5)


def one_sided_p(z) -> float | None:
    if z is None or not np.isfinite(z):
        return None
    return float(0.5 * math.erfc(z / math.sqrt(2)))


def cluster_se(x: np.ndarray, groups: np.ndarray) -> float:
    """SE of the mean of x with observations clustered by groups (CR1 small-sample factor)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float("nan")
    codes, uniq = pd.factorize(pd.Series(groups))
    g = len(uniq)
    if g < 2:
        return float("nan")
    s = np.bincount(codes, weights=x - x.mean(), minlength=g)
    return float(math.sqrt((s ** 2).sum() * g / (g - 1)) / n)


def back_ret(price, f, c):
    price = np.asarray(price, float)
    f = np.asarray(f, float)
    net = f * (price - 1.0) - (1.0 - f)
    return np.where(net > 0, net * (1.0 - c), net)


# ----------------------------------------------------------------------------- vectorised market maths


def shin_vec(inv: np.ndarray, codes: np.ndarray, n_m: int) -> np.ndarray:
    """ponies.odds.shin_probabilities applied per market, vectorised (bisection on z per market)."""
    S = np.bincount(codes, weights=inv, minlength=n_m)
    Sr = S[codes]
    lo = np.zeros(n_m)
    hi = np.full(n_m, 0.5)

    def probs(zm):
        z = zm[codes]
        return (np.sqrt(z * z + 4 * (1 - z) * inv * inv / Sr) - z) / (2 * (1 - z))

    for _ in range(60):
        mid = (lo + hi) / 2
        tot = np.bincount(codes, weights=probs(mid), minlength=n_m)
        gt = tot > 1
        lo = np.where(gt, mid, lo)
        hi = np.where(gt, hi, mid)
    p = probs((lo + hi) / 2)
    p = p / np.bincount(codes, weights=p, minlength=n_m)[codes]
    flat = Sr <= 1.0
    p[flat] = inv[flat] / Sr[flat]
    return p


def topk_batch(P: np.ndarray, k: int) -> np.ndarray:
    """Plain Harville (lambda = 1) probability of finishing in the first k (2 or 3), row-wise for
    (R, n) win-probability rows; identical to ponies.exotics.order_probs(p, 1, 1)['top2'/'top3']."""
    R, n = P.shape
    a = P
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
    W[:, idx, idx] = 0.0
    S = W.sum((1, 2))
    third = a * (S[:, None] - W.sum(2) - W.sum(1))
    return top2 + third


def harville_rows(p: np.ndarray, codes: np.ndarray, kk: np.ndarray) -> np.ndarray:
    """Harville top-k for rows sorted so that each market's rows are contiguous."""
    outp = np.empty(len(p))
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    sizes = np.diff(np.r_[starts, len(codes)])
    k_m = kk[starts]
    key = pd.factorize(pd.MultiIndex.from_arrays([sizes, k_m]))[0]
    for u in np.unique(key):
        ms = np.flatnonzero(key == u)
        n, k = int(sizes[ms[0]]), int(k_m[ms[0]])
        rows = (starts[ms][:, None] + np.arange(n)[None, :]).ravel()
        Pm = p[rows].reshape(-1, n)
        res = np.empty_like(Pm)
        for s in range(0, len(Pm), 4000):
            res[s:s + 4000] = topk_batch(Pm[s:s + 4000], k)
        outp[rows] = res.ravel()
    return outp


# ----------------------------------------------------------------------------- loading


def load_country(cc: str, dev: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    w = pd.read_parquet(DATA / f"{cc}_win.parquet", columns=WIN_COLS)
    n_raw = len(w)
    w = w.drop_duplicates(["market_id", "selection_id"]).reset_index(drop=True)
    n_dedup = len(w)
    name = w["market_name"].fillna("").astype(str)
    w = w[name.str.contains(HARNESS_RE)].reset_index(drop=True)
    if dev:
        w = w[w["start"] < pd.Timestamp("2018-09-01", tz="UTC")].reset_index(drop=True)
    for c in ("market_id", "event_id", "selection_id", "venue", "market_name", "program", "horse_name"):
        w[c] = w[c].astype(object)
    name = w["market_name"].fillna("").astype(str)
    w["mobile"] = name.str.contains(MOBILE_RE).to_numpy()
    w["trot"] = name.str.contains(r"\bTrot\b").to_numpy()
    w["dist"] = pd.to_numeric(name.str.extract(DIST_RE)[0], errors="coerce").to_numpy()
    w["pn"] = pd.to_numeric(w["program"], errors="coerce")
    w["f"] = w["dead_heat_share"].fillna(0.0).astype(float)
    w["day"] = w["date"].astype("datetime64[ns]")
    w = w.sort_values(["start", "market_id", "pn", "selection_id"], kind="mergesort").reset_index(drop=True)
    w["mcode"] = pd.factorize(w["market_id"])[0]
    w["priced"] = (w["ltp_pre"] > 1).fillna(False).to_numpy()
    w["inv_pre"] = np.where(w["priced"], 1.0 / w["ltp_pre"].where(w["priced"], 2.0), 0.0)
    g = w.groupby("mcode", sort=False)
    m = pd.DataFrame({
        "n": g.size(), "won_na": g["won"].apply(lambda x: x.isna().sum()), "won_sum": g["won"].sum(),
        "priced": g["priced"].sum(), "book": g["inv_pre"].sum(), "pn_na": g["pn"].apply(lambda x: x.isna().sum()),
        "pn_dup": g["pn"].apply(lambda x: x.duplicated().sum()), "pn_max": g["pn"].max()})
    m["settled"] = (m["won_na"] == 0) & (m["won_sum"] >= 1)
    m["all_priced"] = m["priced"] == m["n"]
    m["book_ok"] = (m["book"] >= BOOK_LO) & (m["book"] <= BOOK_HI)
    m["pn_ok"] = (m["pn_na"] == 0) & (m["pn_dup"] == 0)
    m["gap"] = m["pn_ok"] & (m["pn_max"] > m["n"])
    m["g6_drop"] = m["gap"] & m["all_priced"] & (m["book"] < BOOK_LO)
    m["qual"] = m["settled"] & m["all_priced"] & m["book_ok"] & m["pn_ok"] & (m["n"] >= 4) & ~m["g6_drop"]
    for c in ("settled", "qual", "book", "gap", "all_priced"):
        w["m_" + c] = m[c].to_numpy()[w["mcode"].to_numpy()]
    info = {"section": "data", "country": cc, "rows_raw": n_raw, "rows_dedup_all_codes": n_dedup,
            "harness_rows": int(len(w)), "harness_markets": int(len(m)), "settled": int(m["settled"].sum()),
            "settled_all_priced": int((m["settled"] & m["all_priced"]).sum()),
            "g1_book_in_0.97_1.03": int((m["settled"] & m["all_priced"] & m["book_ok"]).sum()),
            "g6_gap_and_book_lt_0.97_dropped": int((m["settled"] & m["g6_drop"]).sum()),
            "saddlecloth_nan_or_dup_dropped": int((m["settled"] & m["all_priced"] & m["book_ok"] & ~m["pn_ok"]).sum()),
            "qualifying_markets": int(m["qual"].sum()),
            "qualifying_with_gap": int((m["qual"] & m["gap"]).sum()),
            "mobile_share_qual": r5(w.loc[w["m_qual"]].drop_duplicates("mcode")["mobile"].mean()),
            "dist_parsed_share": r5(np.isfinite(w["dist"]).mean()),
            "bsp_present_share_qual": r5((w.loc[w["m_qual"], "bsp"] > 1).mean()),
            "t30_present_share_qual": r5(w.loc[w["m_qual"], "ltp_t30"].notna().mean())}
    pl = pd.read_parquet(DATA / f"{cc}_place.parquet", columns=PL_COLS)
    pl = pl.drop_duplicates(["market_id", "selection_id"])
    for c in ("market_id", "event_id", "selection_id"):
        pl[c] = pl[c].astype(object)
    pl = pl[pl["event_id"].isin(set(w["event_id"].unique())) & pl["n_winners"].isin([2, 3])].reset_index(drop=True)
    return w, pl, info


# ----------------------------------------------------------------------------- structural gate choices


def gate_structure(w: pd.DataFrame, default_w: int | None) -> tuple[dict, int, list]:
    """W per venue from burn-in, gap-free, qualifying, mobile fields: the saddlecloth W with the
    largest jump in mean within-race-demeaned log market probability from W to W+1."""
    b = w[w["m_qual"] & w["mobile"] & ~w["m_gap"] & (w["start"] < BURN_END)].copy()
    b["lp"] = np.log(b["inv_pre"] / b["m_book"])
    b["dlp"] = b["lp"] - b.groupby("mcode")["lp"].transform("mean")
    b["n"] = b.groupby("mcode")["pn"].transform("size")
    rows = []
    widths = {}
    for v, sv in b.groupby("venue"):
        best = None
        for W in range(5, 11):
            x = sv[sv["n"] >= W + 1]
            nr = x["mcode"].nunique()
            if nr < 15:
                continue
            j = x.loc[x["pn"] == W + 1, "dlp"].mean() - x.loc[x["pn"] == W, "dlp"].mean()
            if best is None or j > best[1]:
                best = (W, float(j), int(nr))
        pole = sv.loc[sv["pn"] == 1, "dlp"].mean()
        acc = best is not None and best[1] >= W_MIN_JUMP and best[2] >= W_MIN_RACES
        if acc:
            widths[v] = best[0]
        rows.append({"venue": v, "burn_races": int(sv["mcode"].nunique()), "best_W": best[0] if best else None,
                     "jump": r5(best[1]) if best else None, "races_at_W": best[2] if best else None,
                     "pole_dlp": r5(pole), "accepted": bool(acc)})
    if default_w is None:
        default_w = int(np.median(list(widths.values()))) if widths else 8
    return widths, default_w, rows


def add_gates(w: pd.DataFrame, widths: dict, default_w: int, top_venues: list) -> pd.DataFrame:
    W = w["venue"].map(widths).fillna(default_w).astype(int).to_numpy()
    pn = w["pn"].to_numpy()
    ok = w["mobile"].to_numpy() & np.isfinite(pn)
    front = ok & (pn <= W)
    second = ok & (pn > W)
    # rows are sorted by (start, market, pn): ranks via cumulative counts within market
    mc = w["mcode"].to_numpy()
    fr = pd.Series(front.astype(int)).groupby(mc).cumsum().to_numpy()
    sr = pd.Series(second.astype(int)).groupby(mc).cumsum().to_numpy()
    w["W"] = W
    w["front_gate"] = np.where(front, fr, 0)
    w["second_pos"] = np.where(second, sr, 0)
    fg, sp = w["front_gate"].to_numpy(), w["second_pos"].to_numpy()
    cols = {}
    for k in range(2, 9):
        cols[f"G{k}"] = (fg == k).astype(float)
    cols["G9p"] = (fg >= 9).astype(float)
    cols["S1"] = (sp == 1).astype(float)
    cols["S2"] = (sp == 2).astype(float)
    cols["S3p"] = (sp >= 3).astype(float)
    grp = {"g23": (fg >= 2) & (fg <= 3), "g46": (fg >= 4) & (fg <= 6), "g7p": fg >= 7, "s1": sp == 1, "s2p": sp >= 2}
    d = w["dist"].to_numpy()
    short = np.isfinite(d) & (d < 1800)
    long_ = np.isfinite(d) & (d > 2200)
    ven = w["venue"].to_numpy()
    for gname, gm in grp.items():
        cols[f"{gname}_short"] = (gm & short).astype(float)
        cols[f"{gname}_long"] = (gm & long_).astype(float)
        for i, v in enumerate(top_venues):
            cols[f"{gname}_v{i}"] = (gm & (ven == v)).astype(float)
    w = pd.concat([w, pd.DataFrame(cols, index=w.index)], axis=1)
    w["gate46"] = grp["g46"]
    return w


def gate_feature_names(n_top: int) -> list:
    names = FRONT_DUMMIES + SECOND_DUMMIES
    for gname in GROUPS:
        names += [f"{gname}_short", f"{gname}_long"] + [f"{gname}_v{i}" for i in range(n_top)]
    return names


def gate_rule_evidence(w: pd.DataFrame, widths: dict) -> list:
    """Market-price evidence for the scratching rule at venues with an accepted W (whole archive,
    market prices only: descriptive, not used by the model)."""
    q = w[w["m_qual"] & w["mobile"]].copy()
    q["lp"] = np.log(q["inv_pre"] / q["m_book"])
    q["dlp"] = q["lp"] - q.groupby("mcode")["lp"].transform("mean")
    q["n"] = q.groupby("mcode")["pn"].transform("size")
    res = []
    for v, Wv in widths.items():
        s = q[q["venue"] == v]
        if s.empty:
            continue
        nog = s[~s["m_gap"]]
        prof = nog.groupby("pn")["dlp"].mean()
        pmax = s.groupby("mcode")["pn"].transform("max")
        one_gap = s["m_gap"] & (pmax == s["n"] + 1)
        # the missing saddlecloth of one-gap markets
        miss = (s.groupby("mcode")["pn"].transform(lambda x: (set(range(1, int(x.max()) + 1)) - set(x.astype(int))).pop()
                                                    if (x.max() == len(x) + 1) else np.nan))
        s = s.assign(gap_at=miss.where(one_gap))
        fg = s[s["gap_at"].notna() & (s["gap_at"] <= Wv)]
        x_w1 = fg.loc[fg["pn"] == Wv + 1, "dlp"]
        g1 = fg[(fg["gap_at"] == 1) & (fg["pn"] == 2)]["dlp"]
        res.append({"venue": v, "W": Wv, "front_gap_markets": int(fg["mcode"].nunique()),
                    "sc_W+1_after_front_scratch_dlp": r5(x_w1.mean()), "n_sc_W+1": int(len(x_w1)),
                    "nogap_dlp_at_W(front_outside)": r5(prof.get(Wv)), "nogap_dlp_at_W+1(second_inside)": r5(prof.get(Wv + 1)),
                    "sc2_when_sc1_scratched_dlp": r5(g1.mean()), "n_sc2": int(len(g1)),
                    "nogap_dlp_sc1(pole)": r5(prof.get(1)), "nogap_dlp_sc2": r5(prof.get(2))})
    return res


# ----------------------------------------------------------------------------- history


def add_history(w: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Point-in-time horse history over every settled harness run of the country."""
    h = w[w["m_settled"]].copy()
    h["bsp_ok"] = (h["bsp"] > 1).fillna(False)
    h["inv_bsp"] = np.where(h["bsp_ok"], 1.0 / h["bsp"].where(h["bsp_ok"], 2.0), 0.0)
    g = h.groupby("mcode", sort=False)
    nb = g["bsp_ok"].transform("sum")
    nn = g["bsp_ok"].transform("size")
    bb = g["inv_bsp"].transform("sum")
    p_bsp = h["inv_bsp"] / bb
    p_pre = h["inv_pre"] / h["m_book"]
    h["p_run"] = np.where(nb == nn, p_bsp, np.where(h["m_all_priced"], p_pre, np.nan))
    key = h["horse_name"].fillna("").astype(str).str.strip().str.lower()
    h["key"] = pd.factorize(key)[0]
    h = h.sort_values(["key", "start"], kind="mergesort")
    dup = int(h.duplicated(["key", "start"]).sum())
    if dup:  # two runs of one key at one start (name collision in simultaneous races): keep both, no ordering
        h = h.sort_values(["key", "start", "market_id"], kind="mergesort")
    gk = h.groupby("key", sort=False)
    h["n_prior"] = gk.cumcount()
    h["prev_start"] = gk["start"].shift(1)
    has_prev = h["n_prior"].to_numpy() > 0
    same = has_prev & (h["prev_start"] >= h["start"]).to_numpy()
    # never use a run that did not start strictly earlier
    assert int(same.sum()) == dup, "leakage guard"
    days = (h["start"] - h["prev_start"]).dt.total_seconds().to_numpy() / 86400.0
    days = np.where(same, np.nan, days)
    has_p = h["p_run"].notna().to_numpy()
    resid = np.where(has_p, h["f"].to_numpy() - h["p_run"].to_numpy(), 0.0)
    cs = pd.Series(resid, index=h.index).groupby(h["key"]).cumsum().to_numpy() - resid
    npr = pd.Series(has_p.astype(float), index=h.index).groupby(h["key"]).cumsum().to_numpy() - has_p
    h["cum_resid"] = cs / (npr + SHRINK)
    valid_prev = has_prev & ~same
    h["no_prior"] = (~valid_prev).astype(float)
    h["log_days"] = np.where(valid_prev, np.log1p(np.minimum(np.nan_to_num(days, nan=365.0), 365.0)), 0.0)
    h["backup7"] = (valid_prev & (days <= BACKUP_DAYS)).astype(float)
    h["log_n_prior"] = np.log1p(h["n_prior"].astype(float))
    info = {"section": "history", "settled_runs": int(len(h)), "keys": int(h["key"].nunique()),
            "same_start_key_collisions": dup, "share_with_prior": r5(valid_prev.mean()),
            "share_backup7_of_with_prior": r5(h.loc[valid_prev, "backup7"].mean()),
            "gap_days_median": r5(np.nanmedian(days[valid_prev])), "p_run_missing_share": r5(1 - has_p.mean()),
            "leakage_assert": "every history quantity uses runs with start strictly earlier: passed"}
    feats = h[["cum_resid", "log_days", "backup7", "no_prior", "log_n_prior"]]
    w = w.join(feats)
    for c in HIST_FEATURES:
        w[c] = w[c].fillna(1.0 if c == "no_prior" else 0.0)
    return w, info


# ----------------------------------------------------------------------------- market features and place ladder


def add_market_and_place(w: pd.DataFrame, pl: pd.DataFrame, cc: str) -> tuple[pd.DataFrame, dict]:
    q = w["m_qual"].to_numpy()
    e = w[q].copy()
    codes, uniq = pd.factorize(e["mcode"])
    n_m = len(uniq)
    p_pre = shin_vec(e["inv_pre"].to_numpy(), codes, n_m)
    e["p_mkt"] = p_pre
    e["lp_pre"] = np.log(p_pre)
    # ---- t30
    t = (e["ltp_t30"] > 1).fillna(False).to_numpy()
    inv30 = np.where(t, 1.0 / e["ltp_t30"].where(t, 2.0).to_numpy(), 0.0)
    nt = np.bincount(codes, weights=t.astype(float), minlength=n_m)
    nn = np.bincount(codes, minlength=n_m)
    b30 = np.bincount(codes, weights=inv30, minlength=n_m)
    full30 = (nt == nn) & (b30 >= T30_BOOK[0]) & (b30 <= T30_BOOK[1])
    p30 = np.full(len(e), np.nan)
    fr = full30[codes]
    if fr.any():
        c2, u2 = pd.factorize(codes[fr])
        p30[fr] = shin_vec(inv30[fr], c2, len(u2))
    mass_pre = np.bincount(codes, weights=np.where(t, p_pre, 0.0), minlength=n_m)
    part = t & ~fr
    with np.errstate(divide="ignore", invalid="ignore"):
        p30_part = inv30 / b30[codes] * mass_pre[codes]
    lp30 = np.where(fr, np.log(np.where(fr, p30, 1.0)), np.where(part, np.log(np.where(part, p30_part, 1.0)), e["lp_pre"]))
    e["lp30"] = lp30
    e["t30_missing"] = (~t).astype(float)
    e["full30"] = fr
    # ---- place market match: same event_id, start within 10 minutes, one per WIN market
    wm = e.drop_duplicates("mcode")[["mcode", "market_id", "event_id", "start", "m_book"]]
    nwin = e.groupby("mcode").size()
    wm = wm.assign(n=wm["mcode"].map(nwin).to_numpy())
    pm = pl.groupby("market_id", sort=False).agg(event_id=("event_id", "first"), start_pl=("start", "first"),
                                                 n_winners=("n_winners", "first"), n_pl=("selection_id", "size")).reset_index()
    pm = pm.rename(columns={"market_id": "pl_mid"})
    pairs = wm.merge(pm, on="event_id")
    pairs["dt"] = (pairs["start"] - pairs["start_pl"]).abs()
    pairs = pairs[pairs["dt"] <= pd.Timedelta(minutes=10)]
    pairs["same_n"] = pairs["n"] == pairs["n_pl"]
    pairs = pairs.sort_values(["mcode", "dt", "same_n", "pl_mid"], ascending=[True, True, False, True])
    pairs = pairs.drop_duplicates("mcode", keep="first")
    pairs = pairs[pairs["n"] > pairs["n_winners"]]
    e = e.merge(pairs[["mcode", "pl_mid", "n_winners"]], on="mcode", how="left")
    plr = pl[["market_id", "selection_id", "ltp_pre", "ltp_t30"]].rename(
        columns={"market_id": "pl_mid", "ltp_pre": "pl_pre", "ltp_t30": "pl_t30"})
    e = e.merge(plr, on=["pl_mid", "selection_id"], how="left")
    e = e.sort_values(["start", "market_id", "pn", "selection_id"], kind="mergesort").reset_index(drop=True)
    codes, uniq = pd.factorize(e["mcode"])
    n_m = len(uniq)
    nn = np.bincount(codes, minlength=n_m)
    kk = e["n_winners"].fillna(0).astype(int).to_numpy()
    info = {"section": "place_join", "country": cc, "qual_markets": int(n_m),
            "markets_with_place_market": int(e.drop_duplicates("mcode")["pl_mid"].notna().sum()),
            "full_t30_win_markets": int(e.drop_duplicates("mcode")["full30"].sum())}
    for tag, col, wincol_ok, pwin in (("30", "pl_t30", e["full30"].to_numpy(), None), ("pre", "pl_pre", np.ones(len(e), bool), None)):
        ok_r = (e[col] > 1).fillna(False).to_numpy() & e["pl_mid"].notna().to_numpy() & wincol_ok
        nok = np.bincount(codes, weights=ok_r.astype(float), minlength=n_m)
        inv = np.where(ok_r, 1.0 / e[col].where(ok_r, 2.0).to_numpy(), 0.0)
        bpl = np.bincount(codes, weights=inv, minlength=n_m) / np.maximum(np.bincount(codes, weights=np.where(ok_r, kk, 0), minlength=n_m) / np.maximum(nok, 1), 1)
        mk_ok = (nok == nn) & (bpl >= PL_BOOK[0]) & (bpl <= PL_BOOK[1])
        r_ok = mk_ok[codes]
        resid = np.zeros(len(e))
        if r_ok.any():
            pw = np.exp(e["lp30"].to_numpy()) if tag == "30" else e["p_mkt"].to_numpy()
            sub = np.flatnonzero(r_ok)
            qh = harville_rows(pw[sub], codes[sub], kk[sub])
            qm = kk[sub] * inv[sub] / np.bincount(codes[sub], weights=inv[sub], minlength=n_m)[codes[sub]]
            qm = np.clip(qm, 1e-4, 0.999)
            resid[sub] = np.log(qm / np.clip(qh, 1e-6, 1.0))
        e[f"presid{'30' if tag == '30' else '_pre'}"] = resid
        e[f"presid{'30' if tag == '30' else '_pre'}_missing"] = (~r_ok).astype(float)
        info[f"markets_with_presid_{tag}"] = int(mk_ok.sum())
        info[f"share_markets_presid_{tag}"] = r5(mk_ok.mean())
    # Harville check against ponies.exotics.order_probs on a few races
    sub = e[e["presid_pre_missing"] == 0].drop_duplicates("mcode").head(50)["mcode"]
    worst = 0.0
    for mc_ in sub:
        g = e[e["mcode"] == mc_]
        p = g["p_mkt"].to_numpy()
        k = int(g["n_winners"].iloc[0])
        ref = order_probs(p, 1.0, 1.0)["top3" if k == 3 else "top2"]
        mine = topk_batch(p[None, :], k)[0]
        worst = max(worst, float(np.abs(ref - mine).max()))
    info["harville_check_max_abs_diff"] = r5(worst)
    # Shin check against ponies.odds.shin_probabilities
    worst = 0.0
    for mc_ in e["mcode"].drop_duplicates().head(50):
        g = e[e["mcode"] == mc_]
        worst = max(worst, float(np.abs(shin_probabilities(g["ltp_pre"].to_numpy()) - g["p_mkt"].to_numpy()).max()))
    info["shin_check_max_abs_diff"] = r5(worst)
    return e, info


# ----------------------------------------------------------------------------- model


class FastLogit(ConditionalLogit):
    """ponies.model.ConditionalLogit's objective (mean per-race log loss + 0.5 l2 |w|^2 on standardised
    features, unpenalised names exempt) solved with L-BFGS-B without materialising the standardised
    matrix: score = X @ (w / sd) (the -mu.w/sd constant cancels in the race softmax; the gradient's mu
    term vanishes because p - y sums to zero in every race). Rows of one race must be contiguous."""

    def fit_arrays(self, X, y, starts, codes, w0=None):
        n_races = len(starts)
        self.means = X.mean(0)
        self.stds = X.std(0)
        self.stds[self.stds == 0] = 1.0
        pen = float(self.l2) * np.array([0.0 if f in self.unpenalized else 1.0 for f in self.feature_names])
        sd = self.stds

        def lg(wt):
            s = X @ (wt / sd)
            e = np.exp(s - np.maximum.reduceat(s, starts)[codes])
            p = e / np.add.reduceat(e, starts)[codes]
            loss = -(y @ np.log(p + 1e-300)) / n_races + 0.5 * wt @ (pen * wt)
            grad = (X.T @ (p - y)) / sd / n_races + pen * wt
            return loss, grad

        x0 = np.zeros(X.shape[1]) if w0 is None else np.asarray(w0, float)
        res = minimize(lg, x0, jac=True, method="L-BFGS-B", options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7})
        self.weights = res.x
        self.history = [float(res.fun)]
        self.n_iter = int(res.nit)
        self.converged = bool(res.success)
        self.impute_values = np.zeros(X.shape[1])
        return self

    def predict_arrays(self, X, starts, codes):
        s = X @ (self.weights / self.stds)
        e = np.exp(s - np.maximum.reduceat(s, starts)[codes])
        return e / np.add.reduceat(e, starts)[codes]


def race_index(mcode: np.ndarray):
    starts = np.flatnonzero(np.r_[True, mcode[1:] != mcode[:-1]])
    codes = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(mcode)]))
    return starts, codes


# globals shared with forked workers
_G: dict = {}


def _wf_task(args):
    """Walk-forward over every OOS month for one (variant, l2): returns OOS predictions and per-month fits."""
    variant, l2 = args
    X = _G["X"][variant]
    y, mcode, month_rows = _G["y"], _G["mcode"], _G["month_rows"]
    feats = _G["feats"][variant]
    Xn = _G["Xnz"][variant]
    nz_month_rows = _G["nz_month_rows"]
    pred = np.full(len(y), np.nan)
    pred_nz = np.full(Xn.shape[0], np.nan) if Xn is not None else None
    fits = {}
    w0 = None
    t0 = time.time()
    iters = []
    for m, (a, b) in month_rows.items():
        if a == 0 or b <= a:
            continue
        starts, codes = race_index(mcode[:a])
        mdl = FastLogit(feats, l2=l2, unpenalized=("lp_pre",)).fit_arrays(X[:a], y[:a], starts, codes, w0)
        w0 = mdl.weights
        iters.append(mdl.n_iter)
        s2, c2 = race_index(mcode[a:b])
        pred[a:b] = mdl.predict_arrays(X[a:b], s2, c2)
        if pred_nz is not None and m in nz_month_rows:
            na, nb = nz_month_rows[m]
            if nb > na:
                s3, c3 = race_index(_G["nz_mcode"][na:nb])
                pred_nz[na:nb] = mdl.predict_arrays(Xn[na:nb], s3, c3)
        fits[m] = {"raw": (mdl.weights / mdl.stds).tolist(), "loss": mdl.history[-1], "conv": mdl.converged,
                   "iters": mdl.n_iter}
    return variant, l2, pred, pred_nz, fits, time.time() - t0, iters


# ----------------------------------------------------------------------------- evaluation helpers


def per_race_ll(p: np.ndarray, y: np.ndarray, codes: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(codes, weights=-y * np.log(np.clip(p, 1e-15, 1)), minlength=n)


def ll_cell(name: str, df: pd.DataFrame, p_a: np.ndarray, p_b: np.ndarray, mid_day, bench: str) -> dict:
    """gain = ll(bench b) - ll(model a) per race, SE per race, halves at mid_day, last 12 months."""
    codes, uniq = pd.factorize(df["mcode"])
    n = len(uniq)
    y = df["y"].to_numpy()
    d = per_race_ll(p_b, y, codes, n) - per_race_ll(p_a, y, codes, n)
    rd = df.drop_duplicates("mcode")
    first = (rd["day"] < mid_day).to_numpy()
    last12 = ((rd["start"] >= LAST12[0]) & (rd["start"] < LAST12[1])).to_numpy()
    res = {"section": "logloss", "cell": name, "benchmark": bench, "races": int(n),
           "ll_bench": r5(per_race_ll(p_b, y, codes, n).mean()), "ll_model": r5(per_race_ll(p_a, y, codes, n).mean()),
           "gain": r5(d.mean()), "se": r5(d.std(ddof=1) / math.sqrt(n)) if n > 1 else None}
    res["z"] = r5(res["gain"] / res["se"]) if res["se"] else None
    hs = []
    for mask in (first, ~first):
        x = d[mask]
        hs.append([r5(x.mean()) if len(x) else None, r5(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None, int(len(x))])
    res["halves_gain_se_n"] = hs
    res["last12_gain"] = r5(d[last12].mean()) if last12.any() else None
    res["both_halves_gt_2se"] = bool(all(h[0] is not None and h[1] and h[0] > 2 * h[1] for h in hs))
    return res


def bet_cell(name: str, family: str, g: pd.DataFrame, mid_day, n_races_eval: int, n_months: int,
             settlement: str = "bsp", decision: str = "pre_off", rule: str = "", country: str = "AU",
             exploratory: bool = False) -> dict:
    col = {"bsp": "bsp", "ltp_paper": "ltp_pre"}[settlement]
    price = g[col].to_numpy(float)
    ok = np.isfinite(price) & (price > 1)
    gs = g[ok]
    price = price[ok]
    f = gs["f"].to_numpy()
    day = gs["day"].to_numpy()
    mkt = gs["mcode"].to_numpy()
    res = {"section": "bet_cell", "cell": name, "family": family, "country": country, "rule": rule,
           "decision_time": decision, "settlement": settlement, "exploratory": exploratory,
           "candidates": int(len(g)), "bets": int(len(gs)), "skipped_no_price": int((~ok).sum())}
    n = len(gs)
    if n < 2:
        res.update({"roi_c8": None, "passes_all_gates": False})
        return res
    rets = {c: back_ret(price, f, c) for c in COMMISSIONS}
    for c in COMMISSIONS:
        res[f"roi_c{int(round(c * 100))}"] = r5(rets[c].mean())
    r8 = rets[C_MAIN]
    se_day = cluster_se(r8, day)
    se_mkt = cluster_se(r8, mkt)
    se_iid = float(r8.std(ddof=1) / math.sqrt(n))
    res.update({"se_iid": r5(se_iid), "se_cl_market": r5(se_mkt), "se_cl_day": r5(se_day),
                "se_used": r5(max(se_day, se_mkt)), "races_with_bet": int(len(np.unique(mkt))), "days": int(len(np.unique(day)))})
    se = max(se_day, se_mkt)
    res["z_c8"] = r5(r8.mean() / se) if se > 0 else None
    res["p_one_sided"] = one_sided_p(r8.mean() / se) if se > 0 else None
    first = day < np.datetime64(mid_day)
    hr, hs, hb = [], [], []
    for mask in (first, ~first):
        x = r8[mask]
        hr.append(r5(x.mean()) if len(x) else None)
        hs.append(r5(max(cluster_se(x, day[mask]), cluster_se(x, mkt[mask]))) if len(x) > 2 else None)
        hb.append(int(mask.sum()))
    res["halves_roi_c8"], res["halves_se"], res["halves_bets"] = hr, hs, hb
    l12 = ((gs["start"] >= LAST12[0]) & (gs["start"] < LAST12[1])).to_numpy()
    res["last12_bets"] = int(l12.sum())
    res["last12_roi_c8"] = r5(r8[l12].mean()) if l12.any() else None
    res["strike"] = r5((f > 0).mean())
    res["mean_price"] = r5(price.mean())
    res["median_price"] = r5(np.median(price))
    res["mean_p_mkt"] = r5(gs["p_mkt"].mean())
    if "p_model" in gs:
        res["mean_p_model"] = r5(gs["p_model"].mean())
    res["won_over_p_mkt"] = r5(f.sum() / gs["p_mkt"].sum())
    if settlement == "bsp":
        res["won_over_p_bsp_raw"] = r5(f.sum() / (1.0 / price).sum())
    res["bets_per_race"] = r5(n / n_races_eval) if n_races_eval else None
    res["bets_per_month"] = r5(n / n_months) if n_months else None
    res["mde_80pct_power"] = r5(2.8 * se)
    g4 = (all(a is not None and b is not None and b > 0 and a > 2 * b for a, b in zip(hr, hs))
          and res["last12_roi_c8"] is not None and res["last12_roi_c8"] > 0)
    res["g4_halves_and_last12"] = bool(g4)
    res["passes_all_gates"] = bool(g4 and settlement == "bsp" and not exploratory)
    return res


# ----------------------------------------------------------------------------- main


def build_country(cc: str, dev: bool, widths_override=None, default_w=None, top_venues=None):
    w, pl, info = load_country(cc, dev)
    out(info)
    if widths_override is None:
        widths, default_w, rows = gate_structure(w, None)
        out({"section": "gate_width", "country": cc, "burn_in": "2016-12..2017-05", "min_jump": W_MIN_JUMP,
             "min_races": W_MIN_RACES, "default_W_(median_of_accepted)": default_w,
             "venues": sorted(rows, key=lambda r: -r["burn_races"])[:30]})
    else:
        widths, _, rows = gate_structure(w, default_w)
        acc = {r["venue"]: r for r in rows if r["accepted"]}
        out({"section": "gate_width", "country": cc, "note": "NZ: same estimator on NZ burn-in; default W from AU",
             "default_W": default_w, "accepted": {k: v["best_W"] for k, v in acc.items()},
             "venues": sorted(rows, key=lambda r: -r["burn_races"])[:12]})
        widths = {k: v["best_W"] for k, v in acc.items()}
    if top_venues is None:
        bq = w[w["m_qual"] & (w["start"] < BURN_END)].drop_duplicates("mcode")
        top_venues = bq["venue"].value_counts().index[:N_TOP_VENUES].tolist()
    w = add_gates(w, widths, default_w, top_venues)
    if widths_override is None:
        for r in gate_rule_evidence(w, widths):
            out({"section": "gate_rule", "country": cc, **r})
    w, hinfo = add_history(w)
    hinfo["country"] = cc
    out(hinfo)
    e, pinfo = add_market_and_place(w, pl, cc)
    out(pinfo)
    e["y"] = win_targets(e["won"].to_numpy(), pd.factorize(e["mcode"])[0], e["mcode"].nunique())
    e["month"] = e["start"].dt.tz_convert(None).dt.to_period("M")
    return e, widths, default_w, top_venues


def month_rows_of(e: pd.DataFrame, months) -> dict:
    st = e["start"].to_numpy()
    res = {}
    for m in months:
        lo = np.datetime64(pd.Timestamp(m.start_time, tz="UTC").tz_convert(None))
        hi = np.datetime64(pd.Timestamp((m + 1).start_time, tz="UTC").tz_convert(None))
        stn = e["start"].dt.tz_convert(None).to_numpy()
        res[m] = (int(np.searchsorted(stn, lo, "left")), int(np.searchsorted(stn, hi, "left")))
    return res


def fit_check(e: pd.DataFrame, feats: list) -> None:
    """FastLogit vs the repo's ConditionalLogit (gradient descent) on a small sample."""
    sub = e[e["mcode"].isin(e["mcode"].drop_duplicates().iloc[:1500])].copy()
    sub["race_id"] = sub["mcode"]
    st, co = race_index(sub["mcode"].to_numpy())
    a = FastLogit(feats, l2=0.1, unpenalized=("lp_pre",)).fit_arrays(sub[feats].to_numpy(float), sub["y"].to_numpy(), st, co)
    b = ConditionalLogit(feats, l2=0.1, unpenalized=("lp_pre",)).fit(sub)
    pa = a.predict_arrays(sub[feats].to_numpy(float), st, co)
    pb = b.predict_proba(sub)
    out({"section": "fit_check", "races": 1500, "features": len(feats), "lbfgs_loss": r5(a.history[-1]),
         "repo_gd_loss": r5(b.history[-1]), "repo_gd_iters": len(b.history),
         "max_abs_prob_diff": r5(np.abs(pa - pb).max())})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="data to 2018-08, a 4-value l2 grid")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    t_start = time.time()
    grid = (1e-3, 1e-2, 0.1, 1.0) if args.dev else L2_GRID
    out({"section": "run", "experiment": "harness_stack", "dev": args.dev, "l2_grid": grid, "k_values": K_VALUES,
         "commissions": COMMISSIONS, "book_filter": [BOOK_LO, BOOK_HI], "burn_in_end": BURN_END,
         "oos_months": [OOS_FIRST, OOS_LAST], "l2_window_months": L2_WINDOW})

    au, widths, default_w, top_venues = build_country("au", args.dev)
    out({"section": "top_venues", "venues": top_venues})
    nz, _, _, _ = build_country("nz", args.dev, widths_override=True, default_w=default_w, top_venues=top_venues)

    gate_feats = gate_feature_names(len(top_venues))
    feats = {"MKT": MKT_FEATURES,
             "STACK": MKT_FEATURES + HIST_FEATURES + PLACE30_FEATURES + gate_feats,
             "STACKX": MKT_FEATURES + HIST_FEATURES + PLACE30_FEATURES + PLACEPRE_FEATURES + gate_feats}
    fit_check(au, feats["STACK"])

    last = pd.Period("2018-08", "M") if args.dev else OOS_LAST
    months = list(pd.period_range(OOS_FIRST, last, freq="M"))
    _G["X"] = {v: au[f].to_numpy(float) for v, f in feats.items()}
    _G["Xnz"] = {v: nz[f].to_numpy(float) for v, f in feats.items()}
    _G["y"] = au["y"].to_numpy()
    _G["mcode"] = au["mcode"].to_numpy()
    _G["nz_mcode"] = nz["mcode"].to_numpy()
    _G["feats"] = feats
    _G["month_rows"] = month_rows_of(au, months)
    _G["nz_month_rows"] = month_rows_of(nz, months)
    for v in feats:
        assert np.isfinite(_G["X"][v]).all() and np.isfinite(_G["Xnz"][v]).all(), v

    tasks = [(v, l2) for v in ("STACK", "STACKX", "MKT") for l2 in grid]
    results = {}
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.workers) as pool:
            for v, l2, pred, pnz, fits, secs, iters in pool.imap_unordered(_wf_task, tasks):
                results[(v, l2)] = (pred, pnz, fits)
                out({"section": "refit_summary", "variant": v, "l2": l2, "months": len(fits), "seconds": round(secs, 1),
                     "iters_median": int(np.median(iters)) if iters else None, "iters_max": int(max(iters)) if iters else None,
                     "all_converged": bool(all(x["conv"] for x in fits.values()))})
    else:
        for tsk in tasks:
            v, l2, pred, pnz, fits, secs, iters = _wf_task(tsk)
            results[(v, l2)] = (pred, pnz, fits)
            out({"section": "refit_summary", "variant": v, "l2": l2, "months": len(fits), "seconds": round(secs, 1),
                 "iters_median": int(np.median(iters)) if iters else None})
    out({"section": "timing", "walk_forward_seconds": round(time.time() - t0, 1)})

    # ---- l2 path: month m uses argmin of OOS log loss over months m-6..m-1
    codes_all, _ = pd.factorize(au["mcode"])
    n_r = codes_all.max() + 1
    race_month = au.drop_duplicates("mcode")["month"].to_numpy()
    test_months = months[L2_WINDOW:]
    chosen = {v: {} for v in feats}
    for v in feats:
        llm = {}
        for l2 in grid:
            pred = results[(v, l2)][0]
            ok = np.isfinite(pred)
            ll = np.bincount(codes_all[ok], weights=-au["y"].to_numpy()[ok] * np.log(np.clip(pred[ok], 1e-15, 1)), minlength=n_r)
            llm[l2] = pd.Series(ll).groupby(race_month).agg(["sum", "size"])
        for m in test_months:
            window = [months[i] for i in range(months.index(m) - L2_WINDOW, months.index(m))]
            scores = {l2: llm[l2].loc[llm[l2].index.isin(window), "sum"].sum() for l2 in grid}
            chosen[v][m] = min(scores, key=scores.get)
        vals = list(chosen[v].values())
        out({"section": "l2_path", "variant": v, "test_months": len(vals),
             "chosen_counts": {str(k): int(sum(1 for x in vals if x == k)) for k in grid},
             "at_grid_edge_months": int(sum(1 for x in vals if x in (grid[0], grid[-1]))),
             "path": {str(m): chosen[v][m] for m in test_months}})

    # ---- assemble test predictions
    mr = _G["month_rows"]
    nzr = _G["nz_month_rows"]
    a0 = mr[test_months[0]][0]
    b0 = mr[test_months[-1]][1]
    te = au.iloc[a0:b0].copy()
    nzt_a, nzt_b = nzr[test_months[0]][0], nzr[test_months[-1]][1]
    tn = nz.iloc[nzt_a:nzt_b].copy()
    for v in feats:
        p = np.full(b0 - a0, np.nan)
        pn_ = np.full(nzt_b - nzt_a, np.nan)
        for m in test_months:
            a, b = mr[m]
            p[a - a0:b - a0] = results[(v, chosen[v][m])][0][a:b]
            na, nb = nzr[m]
            if nb > na:
                pn_[na - nzt_a:nb - nzt_a] = results[(v, chosen[v][m])][1][na:nb]
        te[f"p_{v}"] = p
        tn[f"p_{v}"] = pn_
    assert te[[f"p_{v}" for v in feats]].notna().all().all()
    te["p_model"] = te["p_STACK"]
    tn["p_model"] = tn["p_STACK"]
    tn = tn[tn["p_model"].notna()].copy()

    # ---- coefficients (raw units) over the test months, chosen l2
    for v in ("STACK", "STACKX"):
        raws = np.array([results[(v, chosen[v][m])][2][m]["raw"] for m in test_months])
        names = feats[v]
        key = ["lp_pre", "lp30", "t30_missing", "cum_resid", "log_days", "backup7", "no_prior", "log_n_prior",
               "presid30", "presid30_missing", "presid_pre", "presid_pre_missing", "G2", "G3", "G4", "G5", "G6", "G7", "G8",
               "G9p", "S1", "S2", "S3p", "g46_short", "g46_long"] + [f"g46_v{i}" for i in range(len(top_venues))]
        rows = {}
        for k in key:
            if k in names:
                col = raws[:, names.index(k)]
                mu = col.mean()
                rows[k] = [r5(mu), r5(col.min()), r5(col.max()), r5((np.sign(col) == np.sign(mu)).mean())]
        out({"section": "coef", "variant": v, "columns": "[mean, min, max, sign_share] raw-unit coefficient over test refits",
             "refits": len(test_months), "coef": rows})

    # ---- log loss (same timestamp: market = Shin at ltp_pre)
    days_sorted = np.sort(te.drop_duplicates("mcode")["day"].to_numpy())
    mid_day = pd.Timestamp(days_sorted[len(days_sorted) // 2])
    nz_days = np.sort(tn.drop_duplicates("mcode")["day"].to_numpy()) if len(tn) else np.array([])
    mid_nz = pd.Timestamp(nz_days[len(nz_days) // 2]) if len(nz_days) else mid_day
    out({"section": "halves", "au_test_races": int(te["mcode"].nunique()), "au_test_runners": int(len(te)),
         "au_median_race_date": mid_day, "test_months": [test_months[0], test_months[-1]],
         "nz_test_races": int(tn["mcode"].nunique()), "nz_median_race_date": mid_nz})
    ll_cells = []
    for name, df, md in (("au", te, mid_day), ("nz", tn, mid_nz)):
        if df.empty:
            continue
        pm = df["p_mkt"].to_numpy()
        for v in feats:
            ll_cells.append(ll_cell(f"{name}_{v}_vs_market_pre", df, df[f"p_{v}"].to_numpy(), pm, md, "shin_ltp_pre"))
            out(ll_cells[-1])
        ll_cells.append(ll_cell(f"{name}_STACK_vs_MKT", df, df["p_STACK"].to_numpy(), df["p_MKT"].to_numpy(), md, "MKT model"))
        out(ll_cells[-1])
        ll_cells.append(ll_cell(f"{name}_STACKX_vs_STACK", df, df["p_STACKX"].to_numpy(), df["p_STACK"].to_numpy(), md, "STACK model"))
        out(ll_cells[-1])
        fm = df["full30"].to_numpy()
        if fm.any():
            sub = df[fm]
            p30 = np.exp(sub["lp30"].to_numpy())
            ll_cells.append(ll_cell(f"{name}_STACK_vs_market_t30_fullt30_markets", sub, sub["p_STACK"].to_numpy(), p30, md, "shin_ltp_t30"))
            out(ll_cells[-1])
            ll_cells.append(ll_cell(f"{name}_market_pre_vs_market_t30_fullt30_markets", sub, sub["p_mkt"].to_numpy(), p30, md, "shin_ltp_t30"))
            out(ll_cells[-1])

    # ---- calibration of the model edge (descriptive)
    for name, df in (("au", te), ("nz", tn)):
        if df.empty:
            continue
        edge = df["p_model"] / df["p_mkt"] - 1
        bins = pd.cut(edge, [-1, -0.3, -0.15, -0.05, 0.05, 0.10, 0.15, 0.20, 0.30, 10])
        bsp_ok = (df["bsp"] > 1).fillna(False)
        g = df.assign(inv_bsp=np.where(bsp_ok, 1.0 / df["bsp"].where(bsp_ok, 2.0), np.nan)).groupby(bins, observed=True)
        rows = {}
        for k, x in g:
            rows[str(k)] = {"n": int(len(x)), "won": r5(x["f"].sum()), "sum_p_mkt": r5(x["p_mkt"].sum()),
                            "sum_p_model": r5(x["p_model"].sum()), "sum_1/bsp": r5(x["inv_bsp"].sum()),
                            "won/p_mkt": r5(x["f"].sum() / x["p_mkt"].sum())}
        out({"section": "calibration", "country": name, "by": "p_model/p_mkt - 1", "rows": rows})

    # ---- bets
    n_months = len(test_months)
    cells = []

    def add_rules(df, ctry, md, settlements):
        nre = int(df["mcode"].nunique())
        for v in ("STACK", "STACKX", "MKT"):
            edge = df[f"p_{v}"] / df["p_mkt"] - 1
            for k in K_VALUES:
                for s in settlements:
                    g = df[(edge > k).to_numpy()].assign(p_model=df[f"p_{v}"])
                    c = bet_cell(f"{ctry}_{v}_edge_gt_{k:.2f}_{s}", f"{v}_edge", g, md, nre, n_months, s,
                                 rule=f"back at {'BSP' if s == 'bsp' else 'ltp_pre (paper)'} when p_{v}/p_mkt - 1 > {k}",
                                 country=ctry.upper(), exploratory=(v != "STACK"))
                    cells.append(c)
                    out(c)
        g46 = df["gate46"].to_numpy()
        for nm, mask, expl, rule in (
                ("gate46_and_backup7", g46 & (df["backup7"] == 1).to_numpy(), False, "flat back: front gate 4-6 (mobile) AND previous run <= 7 days"),
                ("gate46_and_place30_likes", g46 & ((df["presid30_missing"] == 0) & (df["presid30"] > 0)).to_numpy(), False,
                 "flat back: front gate 4-6 AND t30 place-ladder residual > 0"),
                ("gate46_and_placepre_likes", g46 & ((df["presid_pre_missing"] == 0) & (df["presid_pre"] > 0)).to_numpy(), True,
                 "flat back: front gate 4-6 AND ltp_pre place-ladder residual > 0 (exploratory)"),
                ("gate46_all", g46, True, "flat back: front gate 4-6 (reference, round-1 effect)"),
                ("backup7_all", (df["backup7"] == 1).to_numpy(), True, "flat back: back-up <= 7 days (reference)")):
            for s in settlements:
                c = bet_cell(f"{ctry}_{nm}_{s}", "intersection", df[mask].assign(p_model=df["p_STACK"]), md, nre, n_months, s,
                             rule=rule, country=ctry.upper(), exploratory=expl)
                cells.append(c)
                out(c)

    add_rules(te, "au", mid_day, ("bsp",))
    if len(tn):
        add_rules(tn, "nz", mid_nz, ("bsp", "ltp_paper"))
    for c in cells:
        if c["country"] == "NZ":
            c["passes_all_gates"] = False  # replication only; the claim is the AU rule
    prim = next(c for c in cells if c["cell"] == "au_STACK_edge_gt_0.10_bsp")
    out({"section": "primary", "pre_registered": "AU STACK edge > 0.10 at BSP, 8% commission", "cell": prim})
    n_ll = len(ll_cells)
    out({"section": "g5_count", "bet_cells": len(cells), "logloss_cells": n_ll, "total_cells": len(cells) + n_ll,
         "bet_cells_pre_registered_AU": int(sum(1 for c in cells if c["country"] == "AU" and not c["exploratory"])),
         "passing_all_gates": [c["cell"] for c in cells if c.get("passes_all_gates")],
         "g4_true_any": [c["cell"] for c in cells if c.get("g4_halves_and_last12")]})
    for cv in CAVEATS:
        out({"section": "caveat", "text": cv})
    out({"section": "done", "seconds": round(time.time() - t_start, 1)})


if __name__ == "__main__":
    main()

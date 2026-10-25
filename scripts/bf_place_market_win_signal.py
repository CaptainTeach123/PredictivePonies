"""Experiment "place_market_win_signal": place-market disagreement with the WIN
price as a WIN signal. Lay the runners the place ladder dislikes (settled at
BSP), with a late-money control.

    python scripts/bf_place_market_win_signal.py [--dev] [--dir data/raw/betfair/split]
                                                 [--cells-csv /path/cells.csv] [--no-sensitivity] [--no-gb]

For every race with both a WIN and a PLACE market, the place market's implied
place probability q_mkt (de-margined, times n_winners) is compared with the
place probability the WIN price implies through discounted Harville
(ponies.exotics.order_probs with a monthly, walk-forward lambda):

    resid = ln(q_mkt / q_harville(win))

Negative resid = the place ladder rates the runner worse than its WIN price
does (drifter, or stale-short WIN LTP). Descriptive resid bins, pre-registered
LAY / BACK threshold rules, a per-code ConditionalLogit (market + resid + late
money controls), an IPF two-market consensus, and a GB replication. The whole
pipeline runs twice: MAIN (place book in [0.95, 1.08]) and SENSITIVITY (no
place-book filter), each with its own BH.

COMMON CONVENTIONS (shared with harness_barrier_logit)
  * Dedupe: drop_duplicates(['market_id', 'selection_id']) on every parquet before anything else.
  * Settlement fraction f = WIN dead_heat_share: 1 for a sole winner, 1/k in a k-way dead heat, 0 otherwise.
  * Back 1 unit at price P: net = f(P-1) - (1-f); return = net(1-c) if net > 0 else net.
  * Lay 1 unit of backer's stake at P: net = (1-f) - f(P-1); return = net(1-c) if net > 0 else net;
    per unit liability = the same divided by (P-1). c in {0, 0.05, 0.08}, commission on net winnings only [A-54].
  * Two settlements per rule: 'ltp' = ltp_pre (the last pre-off traded price: a paper price, neither guaranteed
    available nor fresh [A-53]) and 'bsp' = Betfair SP (an SP order placed before the off); rows with no bsp
    are skipped and the skip count is printed.
  * SE = per-bet sample SE of the c = 5% return; z = roi_c5 / SE; one-sided p = P(Z > z).
  * Halves split at the median start (over races) of the evaluation set, printed.
  * Per-cell JSON: bets, n_won, roi_c0, roi_c5, roi_c8, se, z, p, halves_roi_c5, halves_se_c5, halves_bets,
    skipped (bsp missing), settlement; lay cells add liab_* (per unit liability) and races_with_lay.
  * Benjamini-Hochberg at q = 0.10 over ALL bet cells of a run (the count is printed);
    positive = roi_c5 > 2 SE in BOTH halves (half-specific SE) and bh_pass.
  * Log-loss cells: ll_model, ll_market (mean per-race -log p(winner), dead-heat shares as targets),
    gain = market - model, gain_se from the per-race paired difference, halves, z.
  * Walk-forward: everything fitted uses strictly earlier months (lambda window, L2 choice, monthly refits).
  * ConditionalLogit subclass (same loss: mean per-race log loss + 0.5 l2 |w|^2 on standardised features,
    log_p_win unpenalised) fitted with scipy L-BFGS-B.

Prints summary numbers only (JSON lines); never horse names or rows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ponies.exotics import LAMBDA_GRID, order_probs  # noqa: E402
from ponies.model import ConditionalLogit, win_targets  # noqa: E402

COLS = ["market_id", "event_id", "market_name", "start", "n_winners", "field_size",
        "selection_id", "ltp_pre", "ltp_t30", "bsp", "won", "dead_heat_share"]
COMMS = (0.0, 0.05, 0.08)
C_TEST = 0.05
FDR_Q = 0.10
G = len(LAMBDA_GRID)
RESID_LABELS = ["<-0.3", "[-0.3,-0.1)", "[-0.1,0.1]", "(0.1,0.3]", ">0.3"]
M_LABELS = ["firm<-0.1", "[-0.1,0.1]", "drift>0.1"]
LAY_T = (0.1, 0.2, 0.3)
THETAS = (0.05, 0.10, 0.15)
MAX_LAY_PRICE = 20.0
L2_GRID = (0.01, 0.1, 1.0)
IPF_ITERS = 15
LAMBDA_WINDOW = 12
LAMBDA_MIN_MONTHS = 6
FEATURES = {
    "FULL": ["log_p_win", "resid", "resid_neg", "m", "m_missing", "place_liq"],
    "CONTROLS": ["log_p_win", "m", "m_missing", "place_liq"],
    "IPF": ["log_p_win", "d", "d_neg", "m", "m_missing", "place_liq"],
}


# ----------------------------------------------------------------------------- utilities
def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def out(obj) -> None:
    print(json.dumps(obj, default=_json_default), flush=True)


def r4(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else round(x, 4)


def one_sided_p(z) -> float | None:
    if z is None or (isinstance(z, float) and math.isnan(z)):
        return None
    return float(0.5 * math.erfc(z / math.sqrt(2)))


def benjamini_hochberg(pvals, q: float) -> np.ndarray:
    p = np.asarray([np.nan if v is None else v for v in pvals], dtype=float)
    ok = ~np.isnan(p)
    rej = np.zeros(len(p), dtype=bool)
    m = int(ok.sum())
    if m == 0:
        return rej
    order = np.argsort(p[ok])
    sp = p[ok][order]
    below = np.flatnonzero(sp <= q * np.arange(1, m + 1) / m)
    if len(below):
        rej[np.flatnonzero(ok)[order[: below.max() + 1]]] = True
    return rej


def logit(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(x / (1 - x))


# ----------------------------------------------------------------------------- settlement [A-54]
def back_ret(price, f, c):
    price = np.asarray(price, float); f = np.asarray(f, float)
    net = f * (price - 1.0) - (1.0 - f)
    return np.where(net > 0, net * (1.0 - c), net)


def lay_stake_ret(price, f, c):
    price = np.asarray(price, float); f = np.asarray(f, float)
    net = (1.0 - f) - f * (price - 1.0)
    return np.where(net > 0, net * (1.0 - c), net)


def lay_liab_ret(price, f, c):
    return lay_stake_ret(price, f, c) / (np.asarray(price, float) - 1.0)


def _mean_se(x):
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(x[0]), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(n))


def bet_cell(price, f, first, side: str, skipped: int, race=None, p_win=None) -> dict:
    """side in {'back', 'lay'}; lay cells carry per-unit-stake (main) and per-unit-liability fields.
    hit_over_market = mean(f) / mean(p_win): the runners' win rate relative to what the WIN price implied."""
    price = np.asarray(price, float); f = np.asarray(f, float); first = np.asarray(first, bool)
    n = len(price)
    res = {"bets": n, "skipped": int(skipped), "n_won": int((f > 0).sum())}
    if n == 0:
        return res
    if p_win is not None:
        res["hit_over_market"] = r4(f.mean() / np.asarray(p_win, float).mean())
    fn = back_ret if side == "back" else lay_stake_ret
    for c in COMMS:
        res[f"roi_c{int(c * 100)}"] = r4(fn(price, f, c).mean())
    r5 = fn(price, f, C_TEST)
    m, se = _mean_se(r5)
    res["se"] = r4(se)
    res["z"] = r4(m / se) if se and se > 0 else None
    res["p"] = one_sided_p(m / se) if se and se > 0 else None
    hr, hs, hb = [], [], []
    for mask in (first, ~first):
        hm, hse = _mean_se(r5[mask])
        hr.append(r4(hm)); hs.append(r4(hse)); hb.append(int(mask.sum()))
    res["halves_roi_c5"], res["halves_se_c5"], res["halves_bets"] = hr, hs, hb
    res["both_halves_2se"] = bool(all(a is not None and b is not None and b > 0 and a > 2 * b for a, b in zip(hr, hs)))
    res["mean_price"] = r4(price.mean())
    res["strike"] = r4(f.mean())
    if side == "lay":
        for c in COMMS:
            res[f"liab_roi_c{int(c * 100)}"] = r4(lay_liab_ret(price, f, c).mean())
        l5 = lay_liab_ret(price, f, C_TEST)
        lm, lse = _mean_se(l5)
        res["liab_se"] = r4(lse)
        res["liab_z"] = r4(lm / lse) if lse and lse > 0 else None
        res["liab_halves_roi_c5"] = [r4(_mean_se(l5[first])[0]), r4(_mean_se(l5[~first])[0])]
        if race is not None:
            res["races_with_lay"] = int(len(np.unique(np.asarray(race)[: n])))
    return res


class Cells:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, run: str, family: str, name: str, settlement: str, c: dict, **extra) -> dict:
        self.rows.append({"run": run, "family": family, "name": name, "settlement": settlement, **c, **extra})
        return c

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        for col in [c for c in df.columns if isinstance(df[c].iloc[0], list)]:
            df[col] = df[col].map(json.dumps)
        return df


def settle(g: pd.DataFrame, settlement: str):
    """rows usable for a settlement and the price column: 'ltp' = ltp_win, 'bsp' = bsp_win (> 1)."""
    if settlement == "ltp":
        return g, "ltp_win", 0
    ok = g["bsp_win"].notna() & (g["bsp_win"] > 1)
    return g[ok], "bsp_win", int((~ok).sum())


def rule_cells(cells: Cells, run: str, family: str, name: str, g: pd.DataFrame, side: str, mid, **extra) -> None:
    for s in ("ltp", "bsp"):
        gs, col, skipped = settle(g, s)
        c = bet_cell(gs[col].to_numpy(), gs["f"].to_numpy(), (gs["start"] < mid).to_numpy(), side, skipped,
                     race=gs["race"].to_numpy() if side == "lay" else None, p_win=gs["p_win"].to_numpy())
        cells.add(run, family, name, s, c, side=side, **extra)
        out({"run": run, "section": family, "rule": name, "side": side, "settlement": s, **c, **extra})


# ----------------------------------------------------------------------------- batched discounted Harville
def topk_batch(P: np.ndarray, lam: float, k: int) -> np.ndarray:
    """Row-wise identical to order_probs(p, lam, lam)['top2'/'top3'] (checked in main), without the n^3 array.
    P: (R, n) win probabilities summing to 1 along rows."""
    P = np.asarray(P, float)
    R, n = P.shape
    a = P ** lam
    A = a.sum(1, keepdims=True)
    den_j = np.maximum(A - a, 1e-300)
    ex = P[:, :, None] * (a[:, None, :] / den_j[:, :, None])  # ex[r, j, k] = P(j 1st, k 2nd)
    idx = np.arange(n)
    ex[:, idx, idx] = 0.0
    top2 = P + ex.sum(1)
    if k == 2:
        return top2
    den_jk = np.maximum(A[:, :, None] - a[:, :, None] - a[:, None, :], 1e-300)
    W = ex / den_jk
    S = W.sum((1, 2))
    third = a * (S[:, None] - W.sum(2) - W.sum(1))  # exclude j == i and k == i
    return top2 + third


def topk_rows(P_rows, groups, lam_fn, k_fn, chunk=2000) -> np.ndarray:
    """Apply topk_batch to every race block. groups: list of (idx_rows, n, k, lam_key)."""
    outp = np.empty(len(P_rows))
    for idx, n, k, key in groups:
        Pm = P_rows[idx].reshape(-1, n)
        res = np.empty_like(Pm)
        for s in range(0, len(Pm), chunk):
            res[s:s + chunk] = topk_batch(Pm[s:s + chunk], lam_fn(key), k_fn(key, k))
        outp[idx] = res.ravel()
    return outp


def race_groups(j: pd.DataFrame, extra_key=None) -> list:
    """Rows sorted by (race, pos); every race of the same (n, n_winners[, extra]) stacks into one matrix."""
    keys = [j["n"].to_numpy(), j["n_winners"].to_numpy()]
    if extra_key is not None:
        keys.append(extra_key)
    codes, uniques = pd.factorize(pd.MultiIndex.from_arrays(keys))
    groups = []
    for u, tup in enumerate(uniques):
        idx = np.flatnonzero(codes == u)
        groups.append((idx, int(tup[0]), int(tup[1]), tup[2] if extra_key is not None else None))
    return groups


def harville_grid(j: pd.DataFrame) -> np.ndarray:
    """(rows x G) place probability of each runner for every lambda on LAMBDA_GRID."""
    P = j["p_win"].to_numpy()
    Q = np.empty((len(j), G))
    groups = race_groups(j)
    for g, lam in enumerate(LAMBDA_GRID):
        Q[:, g] = topk_rows(P, groups, lambda key, lam=lam: lam, lambda key, k: k)
    return Q


def check_topk_against_order_probs(j: pd.DataFrame, rng: np.random.Generator, n_races: int = 300) -> float:
    races = rng.choice(j["race"].unique(), size=min(n_races, j["race"].nunique()), replace=False)
    sub = j[j["race"].isin(races)].sort_values(["race", "pos"])
    worst = 0.0
    for r, grp in sub.groupby("race", sort=False):
        p = grp["p_win"].to_numpy(); k = int(grp["n_winners"].iloc[0])
        for lam in (0.5, 0.85, 1.0, 1.3):
            ref = order_probs(p, lam, lam)["top3" if k == 3 else "top2"]
            mine = topk_batch(p[None, :], lam, k)[0]
            worst = max(worst, float(np.abs(ref - mine).max()))
    return worst


# ----------------------------------------------------------------------------- loading / joining
def load_joined(d: Path, cc: str, dev: bool) -> tuple[pd.DataFrame, list]:
    w = pd.read_parquet(d / f"{cc}_win.parquet", columns=COLS).drop_duplicates(["market_id", "selection_id"])
    p = pd.read_parquet(d / f"{cc}_place.parquet", columns=COLS).drop_duplicates(["market_id", "selection_id"])
    for df in (w, p):
        for c in ("market_id", "event_id", "selection_id", "market_name"):
            df[c] = df[c].astype(object)
    if dev:  # lambda path for 2019 uses 2018
        w = w[w["start"] >= pd.Timestamp("2018-01-01", tz="UTC")]
        p = p[p["start"] >= pd.Timestamp("2018-01-01", tz="UTC")]
    stages = []

    def stage(name, mids):
        stages.append((name, int(len(mids))))
        return mids

    # WIN market level
    w["won_na"] = w["won"].isna()
    w["priced"] = (w["ltp_pre"] > 1).fillna(False)
    w["inv"] = np.where(w["priced"], 1.0 / w["ltp_pre"], np.nan)
    wm = w.groupby("market_id", sort=False).agg(event_id=("event_id", "first"), start=("start", "first"),
                                                 name=("market_name", "first"), n=("selection_id", "size"),
                                                 won_sum=("won", "sum"), won_na=("won_na", "sum"),
                                                 priced=("priced", "sum"), book=("inv", "sum"))
    settled = wm[(wm["won_na"] == 0) & (wm["won_sum"] >= 1)]
    stage("0_win_settled_with_winner", settled.index)
    s1 = settled[settled["priced"] == settled["n"]]
    stage("1_every_win_runner_ltp_pre_gt_1", s1.index)
    s2 = s1[(s1["book"] >= 0.97) & (s1["book"] <= 1.05)]
    stage("2_win_book_0.97_1.05", s2.index)

    # PLACE market level
    p = p[p["n_winners"].isin([2, 3])].copy()
    p["won_na"] = p["won"].isna()
    p["priced"] = (p["ltp_pre"] > 1).fillna(False)
    pm = p.groupby("market_id", sort=False).agg(event_id=("event_id", "first"), start=("start", "first"),
                                                 n_winners=("n_winners", "first"), n=("selection_id", "size"),
                                                 won_sum=("won", "sum"), won_na=("won_na", "sum"))
    pairs = s2.reset_index()[["market_id", "event_id", "start", "n"]].merge(
        pm.reset_index().rename(columns={"market_id": "pl_mid", "start": "start_pl", "n": "n_pl"}), on="event_id")
    pairs["dt"] = (pairs["start"] - pairs["start_pl"]).abs()
    pairs = pairs[pairs["dt"] <= pd.Timedelta(minutes=10)]
    pairs["same_n"] = pairs["n"] == pairs["n_pl"]
    n_multi = int((pairs.groupby("market_id").size() > 1).sum())
    pairs = pairs.sort_values(["market_id", "dt", "same_n", "pl_mid"], ascending=[True, True, False, True])
    pairs = pairs.drop_duplicates("market_id", keep="first")
    stage("3_matched_place_market_2or3_places_within_10min", pairs["market_id"])
    pairs = pairs[(pairs["won_na"] == 0) & (pairs["won_sum"] >= pairs["n_winners"])]
    stage("4_place_settled", pairs["market_id"])

    # runner level join
    keep = ["market_id", "selection_id", "market_name", "start", "ltp_pre", "ltp_t30", "bsp", "won", "dead_heat_share"]
    j = w.loc[w["market_id"].isin(pairs["market_id"]), keep].merge(
        pairs[["market_id", "pl_mid", "n_winners", "n"]], on="market_id")
    pr = p[["market_id", "selection_id", "ltp_pre", "ltp_t30", "won"]].rename(
        columns={"market_id": "pl_mid", "ltp_pre": "ltp_pl", "ltp_t30": "t30_pl", "won": "placed"})
    j = j.merge(pr, on=["pl_mid", "selection_id"], how="inner")
    j = j.rename(columns={"ltp_pre": "ltp_win", "ltp_t30": "t30_win", "bsp": "bsp_win", "won": "won_win", "dead_heat_share": "f"})
    g = j.groupby("market_id", sort=False)
    j["n_match"] = g["selection_id"].transform("size")
    j["pl_ok"] = (j["ltp_pl"] > 1).fillna(False).astype(int)
    j["pl_priced"] = g["pl_ok"].transform("min")
    ok = (j["n_match"] == j["n"]) & (j["pl_priced"] == 1)
    j = j[ok].drop(columns=["n_match", "pl_priced", "pl_ok"])
    stage("5_every_win_runner_has_place_row_ltp_gt_1", j["market_id"].unique())
    j = j[j["n"] >= j["n_winners"] + 1]
    stage("5b_at_least_one_unplaced_runner", j["market_id"].unique())
    j["inv_pl"] = 1.0 / j["ltp_pl"]
    j["B_pl"] = j.groupby("market_id", sort=False)["inv_pl"].transform("sum") / j["n_winners"]
    j["bpl_ok"] = (j["B_pl"] >= 0.95) & (j["B_pl"] <= 1.08)
    stage("6_place_book_0.95_1.08_(MAIN only)", j.loc[j["bpl_ok"], "market_id"].unique())

    # derived per-runner fields
    j["race"] = pd.factorize(j["market_id"])[0]
    j["harness"] = j["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b").to_numpy()
    j["code"] = np.where(j["harness"], "harness", "gallops")
    j["month"] = j["start"].dt.tz_convert(None).dt.to_period("M").astype(str)
    j["inv_win"] = 1.0 / j["ltp_win"]
    j["p_win"] = j["inv_win"] / j.groupby("race")["inv_win"].transform("sum")
    j["q_mkt"] = (j["n_winners"] * j["inv_pl"] / j.groupby("race")["inv_pl"].transform("sum")).clip(1e-4, 0.999)
    j["m"] = np.log(j["ltp_win"] / j["t30_win"])  # > 0 drifted, < 0 firmed; NaN without a t30 trade
    j["m_missing"] = j["t30_win"].isna().astype(float)
    j["place_liq"] = j["t30_pl"].notna().astype(float).groupby(j["race"]).transform("mean")
    j["f"] = j["f"].fillna(0.0)
    j = j.sort_values(["race", "ltp_win", "selection_id"]).reset_index(drop=True)
    j["pos"] = j.groupby("race").cumcount()
    j = j.drop(columns=["market_name", "inv_win", "inv_pl", "pl_mid"])
    base = stages[0][1]
    out({"section": "data", "country": cc, "dev": dev, "stages": [(s, c, round(c / base, 4)) for s, c in stages],
         "win_markets_with_several_place_candidates": n_multi, "runners": int(len(j)),
         "harness_share_races": r4(j.drop_duplicates("race")["harness"].mean()),
         "t30_win_present_share": r4(1 - j["m_missing"].mean()), "t30_place_present_share": r4(j["t30_pl"].notna().mean()),
         "bsp_present_share": r4((j["bsp_win"] > 1).mean()), "win_dead_heat_runners": int(((j["f"] > 0) & (j["f"] < 1)).sum()),
         "place_dead_heat_races": int((j.groupby("race")["placed"].sum() > j.groupby("race")["n_winners"].first()).sum())})
    return j, stages


# ----------------------------------------------------------------------------- lambda path, resid, IPF
def lambda_path(j: pd.DataFrame, Q: np.ndarray, eval_months: list[str]) -> dict:
    """lambda_m = argmin over the grid of the mean per-runner binary log loss of q_harv(lambda) vs the place
    flag over the LAMBDA_WINDOW months strictly before m (all earlier if fewer, at least LAMBDA_MIN_MONTHS)."""
    y = j["placed"].to_numpy(float)
    Qc = np.clip(Q, 1e-6, 1 - 1e-6)
    LL = -(y[:, None] * np.log(Qc) + (1 - y[:, None]) * np.log(1 - Qc))
    months = sorted(j["month"].unique())
    mcode = pd.Categorical(j["month"], categories=months).codes
    S = np.zeros((len(months), G)); N = np.zeros(len(months))
    np.add.at(S, mcode, LL)
    np.add.at(N, mcode, 1.0)
    lam = {}
    for m in eval_months:
        i = months.index(m) if m in months else None
        if i is None:
            continue
        win = list(range(max(0, i - LAMBDA_WINDOW), i))
        if len(win) < LAMBDA_MIN_MONTHS:
            continue
        mean_ll = S[win].sum(0) / N[win].sum()
        lam[m] = (int(np.argmin(mean_ll)), float(LAMBDA_GRID[np.argmin(mean_ll)]), int(N[win].sum()), float(mean_ll.min()))
    return lam


def ipf_consensus(j: pd.DataFrame, lam_idx: np.ndarray) -> np.ndarray:
    """r = p_win; 15 x { r_i <- r_i q_mkt_i / topk_i(r); renormalise }; returns r per row."""
    P = j["p_win"].to_numpy(); Qm = j["q_mkt"].to_numpy()
    r_out = np.empty(len(j))
    for idx, n, k, g in race_groups(j, extra_key=lam_idx):
        lam = float(LAMBDA_GRID[int(g)])
        Pm = P[idx].reshape(-1, n); Qg = Qm[idx].reshape(-1, n)
        r = Pm.copy()
        for _ in range(IPF_ITERS):
            t = topk_batch(r, lam, k)
            r = np.maximum(r * Qg / np.maximum(t, 1e-12), 1e-9)
            r /= r.sum(1, keepdims=True)
        r_out[idx] = r.ravel()
    return r_out


# ----------------------------------------------------------------------------- model
class LBFGSLogit(ConditionalLogit):
    """The repo's ConditionalLogit (same loss, standardisation and race-median imputation) fitted with
    scipy L-BFGS-B. Rows must be sorted by race_id so that races are contiguous."""

    def _prepare(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        X = df[self.feature_names].to_numpy(dtype=float)
        if fit:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.impute_values = np.nan_to_num(np.nanmean(X, axis=0))
        nan_cols = np.flatnonzero(np.isnan(X).any(0))
        if len(nan_cols):  # [A-07] race median, then the training mean (only for columns that need it)
            med = pd.DataFrame(X[:, nan_cols]).groupby(df["race_id"].to_numpy()).transform("median").to_numpy()
            X[:, nan_cols] = np.where(np.isnan(X[:, nan_cols]), med, X[:, nan_cols])
            X = np.where(np.isnan(X), self.impute_values, X)
        if fit:
            self.means = X.mean(axis=0)
            self.stds = X.std(axis=0)
            self.stds[self.stds == 0] = 1.0
        return (X - self.means) / self.stds

    def fit(self, df: pd.DataFrame, w0: np.ndarray | None = None) -> "LBFGSLogit":
        race = df["race_id"].to_numpy()
        starts = np.flatnonzero(np.r_[True, race[1:] != race[:-1]])
        n_races = len(starts)
        assert n_races == len(np.unique(race)), "rows must be sorted by race_id"
        codes = np.repeat(np.arange(n_races), np.diff(np.r_[starts, len(race)]))
        y = win_targets(df["won"].to_numpy(), codes, n_races)
        Z = self._prepare(df, fit=True)
        l2 = float(self.l2) * np.array([0.0 if f in self.unpenalized else 1.0 for f in self.feature_names])

        def loss_and_grad(w):
            s = Z @ w
            e = np.exp(s - np.maximum.reduceat(s, starts)[codes])
            p = e / np.add.reduceat(e, starts)[codes]
            loss = -(y @ np.log(p + 1e-15)) / n_races + 0.5 * w @ (l2 * w)
            grad = Z.T @ (p - y) / n_races + l2 * w
            return loss, grad

        x0 = np.zeros(Z.shape[1]) if w0 is None else np.asarray(w0, float)
        res = minimize(loss_and_grad, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-8})
        self.weights = res.x
        self.history = [float(res.fun)]
        self.n_iter = int(res.nit)
        return self


def race_logloss(p: np.ndarray, y: np.ndarray, race: np.ndarray) -> np.ndarray:
    """per-race -sum_i y_i log p_i (y = dead-heat shares of the win)."""
    codes, uniq = pd.factorize(race)
    return np.bincount(codes, weights=-y * np.log(np.clip(p, 1e-12, 1)), minlength=len(uniq))


def paired_ll(ll_a: np.ndarray, ll_b: np.ndarray, first: np.ndarray) -> dict:
    """gain = b - a (positive = a is better), per-race paired SE, halves."""
    d = ll_b - ll_a
    m, se = _mean_se(d)
    return {"races": int(len(d)), "ll_a": r4(ll_a.mean()), "ll_b": r4(ll_b.mean()), "gain": r4(m), "gain_se": r4(se),
            "z": r4(m / se) if se and se > 0 else None,
            "halves_gain": [r4(_mean_se(d[first])[0]), r4(_mean_se(d[~first])[0])],
            "halves_gain_se": [r4(_mean_se(d[first])[1]), r4(_mean_se(d[~first])[1])],
            "halves_races": [int(first.sum()), int((~first).sum())]}


def fit_check(j: pd.DataFrame, rng: np.random.Generator) -> None:
    """The L-BFGS subclass reaches the same optimum as the repo's gradient-descent fit (small sample)."""
    races = rng.choice(j["race"].unique(), size=min(1500, j["race"].nunique()), replace=False)
    sub = j[j["race"].isin(races)].sort_values(["race", "pos"]).rename(columns={"race": "race_id", "won_win": "won"})
    sub = sub.dropna(subset=["resid"]).reset_index(drop=True)
    sub["log_p_win"] = np.log(sub["p_win"])
    sub["resid_neg"] = np.minimum(sub["resid"], 0.0)
    feats = FEATURES["FULL"]
    a = LBFGSLogit(feats, l2=0.1, unpenalized=("log_p_win",)).fit(sub)
    b = ConditionalLogit(feats, l2=0.1, unpenalized=("log_p_win",)).fit(sub)
    out({"section": "fit_check", "races": int(len(races)), "lbfgs_loss": round(a.history[-1], 6), "gd_loss": round(b.history[-1], 6),
         "max_abs_weight_diff": r4(np.abs(a.weights - b.weights).max()), "lbfgs_iters": a.n_iter, "gd_iters": len(b.history)})


def model_stage(j: pd.DataFrame, run: str, eval_months: list[str], cells: Cells, l2_force: float | None = None) -> dict:
    """Per code: L2 chosen once on the first 6 evaluation months (test months 2..6, trained on strictly earlier
    evaluation-set months), then monthly refits on all strictly earlier evaluation-set races; test = months 7+."""
    l2_months = eval_months[1:6]
    test_months = eval_months[6:]
    e = j[j["month"].isin(eval_months)].copy()
    e = e.rename(columns={"race": "race_id", "won_win": "won"}).sort_values(["race_id", "pos"]).reset_index(drop=True)
    e["log_p_win"] = np.log(e["p_win"])
    e["resid_neg"] = np.minimum(e["resid"], 0.0)
    e["d_neg"] = np.minimum(e["d"], 0.0)
    mcode = e["month"].to_numpy()

    # ---- L2 choice (once, pooled over codes and the three feature sets)
    tot = {l2: 0.0 for l2 in L2_GRID}
    tot_by = {}
    for l2 in ([] if l2_force is not None else L2_GRID):
        for code in ("gallops", "harness"):
            ec = e[e["code"] == code]
            if ec.empty:
                continue
            for name, feats in FEATURES.items():
                ll = 0.0; nr = 0
                for m in l2_months:
                    tr = ec[(ec["month"] < m) & (ec["month"] >= eval_months[0])]
                    te = ec[ec["month"] == m]
                    if tr["race_id"].nunique() < 200 or te.empty:
                        continue
                    mdl = LBFGSLogit(feats, l2=l2, unpenalized=("log_p_win",)).fit(tr)
                    pr = mdl.predict_proba(te)
                    y = win_targets(te["won"].to_numpy(), pd.factorize(te["race_id"])[0], te["race_id"].nunique())
                    ll += float(race_logloss(pr, y, te["race_id"].to_numpy()).sum()); nr += te["race_id"].nunique()
                tot[l2] += ll
                tot_by[(l2, code, name)] = (r4(ll / nr) if nr else None, nr)
    l2_best = min(tot, key=tot.get) if l2_force is None else float(l2_force)
    out({"run": run, "section": "model_l2", "test_months": [l2_months[0], l2_months[-1]], "total_ll": {str(k): round(v, 2) for k, v in tot.items()},
         "mean_ll_by_l2_code_model": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in tot_by.items()}, "chosen_l2": l2_best,
         "forced": l2_force is not None})

    # ---- monthly refits
    pred = {name: np.full(len(e), np.nan) for name in FEATURES}
    coef_last = {}
    t0 = time.time(); n_fits = 0
    for code in ("gallops", "harness"):
        ec_mask = (e["code"] == code).to_numpy()
        if not ec_mask.any():
            continue
        warm = {name: None for name in FEATURES}
        for m in test_months:
            tr_mask = ec_mask & (mcode < m) & (mcode >= eval_months[0])
            te_mask = ec_mask & (mcode == m)
            if not te_mask.any() or tr_mask.sum() == 0:
                continue
            tr = e[tr_mask]; te = e[te_mask]
            for name, feats in FEATURES.items():
                mdl = LBFGSLogit(feats, l2=l2_best, unpenalized=("log_p_win",)).fit(tr, w0=warm[name])
                warm[name] = mdl.weights
                pred[name][te_mask] = mdl.predict_proba(te)
                n_fits += 1
                if m == test_months[-1]:
                    coef_last[(code, name)] = dict(zip(feats, [r4(v) for v in mdl.weights]))
    out({"run": run, "section": "model_refits", "fits": n_fits, "months": len(test_months), "seconds": round(time.time() - t0, 1),
         "final_standardised_coefficients": {f"{k[0]}|{k[1]}": v for k, v in coef_last.items()}})

    # ---- log-loss cells (test period, per code), market = p_win
    te_mask = np.isin(mcode, test_months) & ~np.isnan(pred["FULL"])
    te = e[te_mask]
    race_start = te.groupby("race_id")["start"].first()
    mid_test = race_start.median()
    out({"run": run, "section": "model_test_period", "months": [test_months[0], test_months[-1]], "races": int(len(race_start)),
         "median_start": str(mid_test)})
    y_all = win_targets(te["won"].to_numpy(), pd.factorize(te["race_id"])[0], te["race_id"].nunique())
    ll_cells = {}
    for code in ("gallops", "harness"):
        cm = (te["code"] == code).to_numpy()
        if not cm.any():
            continue
        race = te["race_id"].to_numpy()[cm]
        first = (te.groupby("race_id")["start"].first().loc[pd.unique(race)] < mid_test).to_numpy()
        ll_mkt = race_logloss(te["p_win"].to_numpy()[cm], y_all[cm], race)
        lls = {}
        for name in FEATURES:
            lls[name] = race_logloss(pred[name][te_mask][cm], y_all[cm], race)
            c = paired_ll(lls[name], ll_mkt, first)
            c = {"ll_model": c.pop("ll_a"), "ll_market": c.pop("ll_b"), **c}
            ll_cells[(code, name)] = c
            out({"run": run, "section": "logloss", "code": code, "model": name, **c})
        inc = paired_ll(lls["FULL"], lls["CONTROLS"], first)
        inc = {"ll_full": inc.pop("ll_a"), "ll_controls": inc.pop("ll_b"), **inc}
        out({"run": run, "section": "logloss_incremental_FULL_minus_CONTROLS", "code": code, **inc})
        inc2 = paired_ll(lls["IPF"], lls["FULL"], first)
        out({"run": run, "section": "logloss_IPF_vs_FULL", "code": code, "gain_IPF_over_FULL": inc2["gain"], "gain_se": inc2["gain_se"], "z": inc2["z"]})

    # ---- model rules (codes pooled, ltp_pre <= 20)
    te = te.assign(p_model=pred["FULL"][te_mask]).rename(columns={"race_id": "race"})
    base = te[te["ltp_win"] <= MAX_LAY_PRICE]
    for theta in THETAS:
        lay = base[base["p_model"] < base["p_win"] * (1 - theta)]
        rule_cells(cells, run, "model_rule", f"LAY p_FULL < p_win*(1-{theta})", lay, "lay", mid_test, theta=theta)
        back = base[base["p_model"] > base["p_win"] * (1 + theta)]
        rule_cells(cells, run, "model_rule", f"BACK p_FULL > p_win*(1+{theta})", back, "back", mid_test, theta=theta)
    return {"l2": l2_best, "ll_cells": ll_cells}


# ----------------------------------------------------------------------------- one full run (MAIN or SENSITIVITY)
def analyse(j: pd.DataFrame, Q: np.ndarray, run: str, cc: str, eval_months: list[str], cells: Cells,
            rng: np.random.Generator, full: bool, l2_force: float | None = None) -> dict:
    lam = lambda_path(j, Q, eval_months)
    out({"run": run, "section": "lambda_path", "country": cc, "lambda_by_month": {m: v[1] for m, v in lam.items()},
         "fit_runners_by_month": {m: v[2] for m, v in lam.items()}})
    lam_idx = j["month"].map({m: v[0] for m, v in lam.items()})
    e = j[lam_idx.notna()].copy()
    li = lam_idx[lam_idx.notna()].to_numpy(int)
    Qe = Q[lam_idx.notna().to_numpy()]
    e["q_harv"] = np.clip(Qe[np.arange(len(e)), li], 1e-4, 0.999)
    e["resid"] = np.log(e["q_mkt"] / e["q_harv"])
    e["resid_bin"] = np.select([e["resid"] < -0.3, e["resid"] < -0.1, e["resid"] <= 0.1, e["resid"] <= 0.3], [0, 1, 2, 3], 4)
    e["m_bin"] = np.select([e["m"] < -0.1, e["m"] <= 0.1], [0, 1], 2)
    if full:
        t0 = time.time()
        r = ipf_consensus(e, li)
        e["d"] = logit(r) - logit(e["p_win"])
        out({"run": run, "section": "ipf", "seconds": round(time.time() - t0, 1), "corr_d_resid": r4(np.corrcoef(e["d"], e["resid"])[0, 1]),
             "corr_d_resid_ltp_le_20": r4(np.corrcoef(e.loc[e["ltp_win"] <= 20, "d"], e.loc[e["ltp_win"] <= 20, "resid"])[0, 1]),
             "d_quantiles": [r4(v) for v in np.quantile(e["d"], [0.01, 0.1, 0.5, 0.9, 0.99])]})
    races = e.drop_duplicates("race")
    mid = races["start"].median()
    e["first"] = e["start"] < mid
    out({"run": run, "section": "evaluation_set", "country": cc, "months": [eval_months[0], eval_months[-1]], "races": int(len(races)),
         "runners": int(len(e)), "median_start": str(mid), "races_by_code": races["code"].value_counts().to_dict(),
         "resid_quantiles": [r4(v) for v in np.quantile(e["resid"], [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])],
         "resid_bin_share": [r4(v) for v in np.bincount(e["resid_bin"], minlength=5) / len(e)],
         "resid_mean_by_code": e.groupby("code")["resid"].mean().round(4).to_dict()})

    if full:
        # ---- descriptive cells: resid bin x code x settlement (back every runner)
        for b, lab in enumerate(RESID_LABELS):
            for code in ("gallops", "harness"):
                g = e[(e["resid_bin"] == b) & (e["code"] == code)]
                for s in ("ltp", "bsp"):
                    gs, col, skipped = settle(g, s)
                    c = bet_cell(gs[col].to_numpy(), gs["f"].to_numpy(), gs["first"].to_numpy(), "back", skipped,
                                 p_win=gs["p_win"].to_numpy())
                    cells.add(run, "descriptive", f"resid {lab} | {code}", s, c, side="back")
                    out({"run": run, "section": "descriptive", "resid_bin": lab, "code": code, "settlement": s, **c})
        # ---- cross-tab resid bin x late-money bin (runners with a t30 trade), not a bet cell
        t = e[e["m_missing"] == 0]
        tab = t.groupby(["resid_bin", "m_bin"]).agg(n=("f", "size"), hit=("f", "mean"), p=("p_win", "mean"))
        tab["hit_over_market"] = tab["hit"] / tab["p"]
        xt = {f"{RESID_LABELS[rb]} x {M_LABELS[mb]}": [int(row["n"]), r4(row["hit_over_market"])] for (rb, mb), row in tab.iterrows()}
        out({"run": run, "section": "crosstab_resid_x_latemoney", "runners_with_t30": int(len(t)), "cells_n_hitOverMarket": xt,
             "corr_resid_m": r4(np.corrcoef(t["resid"], t["m"])[0, 1])})
        # ---- threshold rules (codes pooled)
        for tt in LAY_T:
            g = e[(e["resid"] < -tt) & (e["ltp_win"] <= MAX_LAY_PRICE)]
            rule_cells(cells, run, "threshold", f"LAY resid < -{tt}, ltp_pre <= 20", g, "lay", mid, t=tt)
            g = e[e["resid"] > tt]
            rule_cells(cells, run, "threshold", f"BACK resid > {tt}", g, "back", mid, t=tt)
        if run == "MAIN":
            fit_check(e, rng)
            # share of the 'dislike' signal that survives in the bettable price range (not a bet cell)
            for cap in (10.0, 20.0):
                g = e[e["ltp_win"] <= cap]
                out({"run": run, "section": "hit_over_market_by_resid_bin", "ltp_pre_cap": cap, "runners": int(len(g)),
                     "n_hitOverMarket_by_bin": {RESID_LABELS[b]: [int((g["resid_bin"] == b).sum()),
                                                                  r4(g.loc[g["resid_bin"] == b, "f"].mean() / g.loc[g["resid_bin"] == b, "p_win"].mean())]
                                                for b in range(5)}})
        model_stage(e, run, eval_months, cells, l2_force)
    else:
        g = e[(e["resid"] < -0.2) & (e["ltp_win"] <= MAX_LAY_PRICE)]
        rule_cells(cells, run, "gb_replication", f"GB LAY resid < -0.2, ltp_pre <= 20", g, "lay", mid, t=0.2)
    return {"mid": mid, "lambda": lam}


# ----------------------------------------------------------------------------- BH, positives, summary
def summarise(cells: Cells, run: str) -> None:
    rows = [r for r in cells.rows if r["run"] == run]
    p = [r.get("p") for r in rows]
    passed = benjamini_hochberg(p, FDR_Q)
    for r, bh in zip(rows, passed):
        r["bh_pass"] = bool(bh)
        r["positive"] = bool(bh and r.get("both_halves_2se"))
    table = [{"family": r["family"], "rule": r["name"], "settlement": r["settlement"], "bets": r["bets"], "roi_c0": r.get("roi_c0"),
              "roi_c5": r.get("roi_c5"), "roi_c8": r.get("roi_c8"), "se": r.get("se"), "halves_roi_c5": r.get("halves_roi_c5"),
              "z": r.get("z"), "p": r4(r.get("p")), "bh_pass": r["bh_pass"], "positive": r["positive"],
              **({"liab_roi_c5": r.get("liab_roi_c5"), "liab_se": r.get("liab_se"), "liab_halves_roi_c5": r.get("liab_halves_roi_c5"),
                  "races_with_lay": r.get("races_with_lay")} if r.get("side") == "lay" else {})} for r in rows]
    prim = [r for r in rows if r["family"] == "threshold" and r["name"].startswith("LAY resid < -0.2") and r["settlement"] == "bsp"]
    primary = None
    if prim:
        r = prim[0]
        primary = {"rule": r["name"], "settlement": "bsp", "bets": r["bets"], "roi_c5_per_unit_stake": r.get("roi_c5"), "se": r.get("se"),
                   "halves_roi_c5": r.get("halves_roi_c5"), "halves_se_c5": r.get("halves_se_c5"), "bh_pass": r["bh_pass"],
                   "positive": r["positive"], "roi_c5_per_unit_liability": r.get("liab_roi_c5"), "liab_se": r.get("liab_se"),
                   "liab_halves_roi_c5": r.get("liab_halves_roi_c5"), "races_with_lay": r.get("races_with_lay")}
    # opposite-half development: cells chosen in one half at > 2 SE, evaluated in the other
    opp = []
    for r in rows:
        h, s = r.get("halves_roi_c5") or [None, None], r.get("halves_se_c5") or [None, None]
        for dev_h, test_h in ((0, 1), (1, 0)):
            if h[dev_h] is not None and s[dev_h] and h[dev_h] > 2 * s[dev_h]:
                opp.append({"dev_half": dev_h + 1, "rule": r["name"], "settlement": r["settlement"], "dev_roi": h[dev_h],
                            "test_roi": h[test_h], "test_z": r4(h[test_h] / s[test_h]) if s[test_h] else None})
    out({"run": run, "section": "BH", "q": FDR_Q, "bet_cells_tested": len(rows), "bh_passes": int(passed.sum()),
         "positive_cells": [f"{r['name']} @ {r['settlement']}" for r in rows if r["positive"]], "primary": primary,
         "opposite_half_development": opp, "cells": table})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent.parent / "data/raw/betfair/split")
    ap.add_argument("--dev", action="store_true", help="evaluation set = 2019 only (lambda path uses 2018)")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--no-gb", action="store_true")
    ap.add_argument("--cells-csv", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--l2", type=float, default=None,
                    help="post-hoc: force the L2 penalty instead of the walk-forward choice (run labels carry it)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t_all = time.time()
    if a.dev:
        eval_months = [f"2019-{m:02d}" for m in range(1, 13)]
    else:
        eval_months = [str(p) for p in pd.period_range("2017-06", "2020-11", freq="M")]
    out({"section": "config", "dev": a.dev, "eval_months": [eval_months[0], eval_months[-1]], "lambda_grid": [float(x) for x in LAMBDA_GRID],
         "fdr_q": FDR_Q, "commissions": COMMS, "max_lay_price": MAX_LAY_PRICE, "l2_grid": L2_GRID})
    cells = Cells()
    tag = "" if a.l2 is None else f"_posthoc_l2={a.l2:g}"
    runs = ["MAIN" + tag] + ([] if a.no_sensitivity else ["SENSITIVITY" + tag])

    # ---- AU
    j_au, _ = load_joined(a.dir, "au", a.dev)
    t0 = time.time()
    worst = check_topk_against_order_probs(j_au, rng)
    Q_au = harville_grid(j_au)
    out({"section": "harville_precompute", "country": "au", "runners": int(len(j_au)), "races": int(j_au["race"].nunique()),
         "lambdas": G, "seconds": round(time.time() - t0, 1), "max_abs_diff_vs_order_probs": worst})
    # ---- GB
    j_gb = Q_gb = None
    if not a.no_gb:
        j_gb, _ = load_joined(a.dir, "gb", a.dev)
        t0 = time.time()
        Q_gb = harville_grid(j_gb)
        out({"section": "harville_precompute", "country": "gb", "runners": int(len(j_gb)), "races": int(j_gb["race"].nunique()),
             "seconds": round(time.time() - t0, 1)})

    for run in runs:
        t0 = time.time()
        main = run.startswith("MAIN")
        sel = j_au["bpl_ok"].to_numpy() if main else np.ones(len(j_au), bool)
        analyse(j_au[sel].reset_index(drop=True), Q_au[sel], run, "au", eval_months, cells, rng, full=True, l2_force=a.l2)
        if j_gb is not None:
            selg = j_gb["bpl_ok"].to_numpy() if main else np.ones(len(j_gb), bool)
            analyse(j_gb[selg].reset_index(drop=True), Q_gb[selg], run, "gb", eval_months, cells, rng, full=False)
        summarise(cells, run)
        out({"run": run, "section": "timing", "seconds": round(time.time() - t0, 1)})

    if a.cells_csv:
        cells.frame().to_csv(a.cells_csv, index=False)
    out({"section": "done", "total_seconds": round(time.time() - t_all, 1), "cells_total": len(cells.rows)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

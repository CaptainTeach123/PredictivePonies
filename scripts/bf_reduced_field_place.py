"""Experiment "reduced_field_place": three places still paid on 5-7 runner fields after
scratchings. Harville overlay in the PLACE market versus normal-terms controls, plus the
re-priced WIN market of the same races.

    python scripts/bf_reduced_field_place.py [--dev] [--dir data/raw/betfair/split]
                                             [--cells-csv /path/cells.csv] [--no-sensitivity] [--no-gb]

Betfair fixes the number of places when a PLACE market loads and does not reduce it after
scratchings, so a market that loaded with 8+ runners and lost some keeps paying 3 places with
5-7 runners at the jump (set T). The same rules are run on normal-terms controls: C1 = 3
places with 8-9 runners (nearest field sizes), C2 = 2 places with 5-7 runners (same field
sizes). T4 = 2 places with <= 4 runners (counts and calibration only).

Fair place probability: p_win = normalised 1/ltp_pre of the WIN market; for every lambda on
ponies.exotics.LAMBDA_GRID q_harv(lambda) = order_probs(p_win, lambda, lambda)['top3' or 'top2']
(batched implementation checked against order_probs). lambda_m per month and per n_winners =
argmin of the mean per-runner binary log loss against the place won flag over the previous 12
months (minimum 6; first evaluated month 2017-06) of ALL fully priced AU place markets with the
same n_winners (pooled); lambda_T is fitted the same way on earlier T markets only and drives
a secondary 'lambdaT' version of rule (a). q_mkt = n_winners*(1/ltp_place)/sum(1/ltp_place);
edge e = q_harv*ltp_place - 1.

Cells (PLACE market, 1 unit, settled at place ltp_pre and place bsp): calibration bands x
{T, C1, C2} (36), rules (a) back e > theta for theta in {0.05, 0.10, 0.20}, (b) lay e < -0.10,
(c) back p_win ranks 2-4, on each set (30), WIN re-pricing in T races (8), GB replication (4):
78 bet cells under one Benjamini-Hochberg (MAIN run); the SENSITIVITY run (every priced place
runner in every settled place market, no book guard) repeats them with its own BH.

PRIMARY (pre-registered, both parts required for a 'reduced-field' claim): (i) rule (a)
theta = 0.10 on AU T, settlement bsp, roi_c5 > 2 SE in both halves and bh_pass; (ii) T - C1
difference for the same rule and settlement > 2 SE_diff. (i) without (ii) = general Dr Z
value that belongs to win_to_place, not a reduced-field effect.

COMMON CONVENTIONS (shared with harness_barrier_logit)
  * Dedupe: drop_duplicates(['market_id', 'selection_id']) on every parquet before anything else.
  * Settlement fraction f: WIN market f = dead_heat_share (1 for a sole winner, 1/k in a k-way dead heat,
    0 otherwise); PLACE market f = min(1, n_winners*dead_heat_share) for a placed runner, 0 otherwise.
  * Back 1 unit at price P: net = f(P-1) - (1-f); return = net(1-c) if net > 0 else net.
  * Lay 1 unit of backer's stake at P: net = (1-f) - f(P-1); return = net(1-c) if net > 0 else net;
    per unit liability = the same divided by (P-1). c in {0, 0.05, 0.08}, commission on net winnings only [A-54].
  * Two settlements per rule: 'ltp' = ltp_pre (the last pre-off traded price: a paper price, neither guaranteed
    available nor fresh [A-53]) and 'bsp' = Betfair SP (an SP order placed before the off); rows with no bsp
    are skipped and the skip count is printed. Back/ltp cells also print roi_c5 one Betfair tick below ltp_pre,
    lay/ltp cells one tick above (the realistic side of the spread in each case).
  * SE = per-bet sample SE of the c = 5% return; z = roi_c5 / SE; one-sided p = P(Z > z).
  * Halves split at one fixed date = the median start (over races) of the MAIN T evaluation set, printed,
    used for T, C1 and C2 alike (GB: the median of GB T races).
  * Per-cell JSON: bets, n_won, roi_c0, roi_c5, roi_c8, se, z, p, halves_roi_c5, halves_se_c5, halves_bets,
    skipped (bsp missing), settlement; lay cells add liab_* (per unit liability) and races_with_lay.
  * Benjamini-Hochberg at q = 0.10 over ALL bet cells of a run (the count is printed);
    positive = roi_c5 > 2 SE in BOTH halves (half-specific SE) and bh_pass.
  * Walk-forward: everything fitted (the lambda paths) uses strictly earlier months.

Prints summary numbers only (JSON lines); never horse names or rows.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ponies.exotics import LAMBDA_GRID, order_probs  # noqa: E402

COLS = ["market_id", "event_id", "market_name", "start", "n_winners", "field_size",
        "selection_id", "ltp_pre", "ltp_t30", "bsp", "won", "dead_heat_share"]
COMMS = (0.0, 0.05, 0.08)
C_TEST = 0.05
FDR_Q = 0.10
G = len(LAMBDA_GRID)
LAMBDA_WINDOW = 12
LAMBDA_MIN_MONTHS = 6
THETAS = (0.05, 0.10, 0.20)
LAY_T = 0.10
PL_EDGES = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf]
PL_LABELS = ["<=1.5", "(1.5,2]", "(2,3]", "(3,5]", "(5,10]", ">10"]
WIN_EDGES = [1.0, 3.0, 10.0, np.inf]
WIN_LABELS = ["<=3", "(3,10]", ">10"]
SETS = ("T", "C1", "C2")
RULES = ["a_theta0.05", "a_theta0.10", "a_theta0.20", "b_lay_e<-0.10", "c_rank2-4"]
EXPECTED_CELLS = 36 + 30 + 8 + 4
PRIMARY_RULE, PRIMARY_SET, PRIMARY_SETTLEMENT = "a_theta0.10", "T", "bsp"
POWER = ("Expected power: T fully priced is ~4.4-5.7k markets (~27-35k runner-bets, SE ~0.7%); rule (a) "
         "theta = 0.10 is ~3k bets at prices 1.5-4 (SE ~2%), so the rule can detect a >= 5% edge but not 2-3%.")
# Betfair price ladder (lower inclusive, upper exclusive, increment) for the tick-down line
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1001.0, 10.0)]


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
    """BH over ALL cells of a run: an undefined p (no bets) counts as 1."""
    p = np.asarray([1.0 if v is None else v for v in pvals], dtype=float)
    p = np.where(np.isnan(p), 1.0, p)
    m = len(p)
    rej = np.zeros(m, dtype=bool)
    if m == 0:
        return rej
    order = np.argsort(p)
    below = np.flatnonzero(p[order] <= q * np.arange(1, m + 1) / m)
    if len(below):
        rej[order[: below.max() + 1]] = True
    return rej


def tick_up(price: np.ndarray) -> np.ndarray:
    p = np.asarray(price, dtype=float)
    inc = np.full_like(p, 10.0)
    q = p + 1e-9
    for lo, hi, step in TICKS:
        inc = np.where((q >= lo) & (q < hi), step, inc)
    return np.round(p + inc, 2)


def tick_down(price: np.ndarray) -> np.ndarray:
    p = np.asarray(price, dtype=float)
    inc = np.full_like(p, 10.0)
    q = p - 1e-9
    for lo, hi, step in TICKS:
        inc = np.where((q >= lo) & (q < hi), step, inc)
    return np.maximum(1.01, np.round(p - inc, 2))


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


def bet_cell(price, f, first, side: str, skipped: int, settlement: str, race=None) -> dict:
    price = np.asarray(price, float); f = np.asarray(f, float); first = np.asarray(first, bool)
    n = len(price)
    res = {"bets": n, "skipped": int(skipped), "n_won": int((f > 0).sum())}
    if n == 0:
        res.update({"roi_c0": None, "roi_c5": None, "roi_c8": None, "se": None, "z": None, "p": None,
                    "halves_roi_c5": [None, None], "halves_se_c5": [None, None], "halves_bets": [0, 0],
                    "both_halves_2se": False})
        return res
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
    res["median_price"] = r4(np.median(price))
    res["strike"] = r4((f > 0).mean())
    res["dead_heat_bets"] = int(((f > 0) & (f < 1)).sum())
    res["max_price"] = r4(price.max())
    if side == "back" and settlement == "ltp":
        res["roi_c5_tick_down"] = r4(back_ret(tick_down(price), f, C_TEST).mean())
    if side == "lay" and settlement == "ltp":
        res["roi_c5_tick_up"] = r4(lay_stake_ret(tick_up(price), f, C_TEST).mean())
    if side == "lay":
        for c in COMMS:
            res[f"liab_roi_c{int(c * 100)}"] = r4(lay_liab_ret(price, f, c).mean())
        l5 = lay_liab_ret(price, f, C_TEST)
        lm, lse = _mean_se(l5)
        res["liab_se"] = r4(lse)
        res["liab_z"] = r4(lm / lse) if lse and lse > 0 else None
        res["liab_halves_roi_c5"] = [r4(_mean_se(l5[first])[0]), r4(_mean_se(l5[~first])[0])]
        if race is not None:
            res["races_with_lay"] = int(len(np.unique(np.asarray(race))))
    return res


class Cells:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, run: str, family: str, name: str, set_: str, settlement: str, side: str, c: dict,
            in_bh: bool = True, **extra) -> dict:
        row = {"run": run, "family": family, "name": name, "set": set_, "settlement": settlement, "side": side,
               "in_bh": in_bh, **c, **extra}
        self.rows.append(row)
        out({"section": family, **row})
        return row

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        for col in df.columns:
            if df[col].map(lambda v: isinstance(v, list)).any():
                df[col] = df[col].map(json.dumps)
        return df


def settle(g: pd.DataFrame, settlement: str, market: str):
    """rows usable for a settlement and the price column. market 'pl' -> ltp_pl / bsp_pl, 'win' -> ltp_win / bsp_win."""
    if settlement == "ltp":
        return g, f"ltp_{market}", 0
    col = f"bsp_{market}"
    ok = g[col].notna() & (g[col] > 1)
    return g[ok], col, int((~ok).sum())


def rule_cells(cells: Cells, run: str, family: str, name: str, set_: str, g: pd.DataFrame, side: str,
               market: str = "pl", in_bh: bool = True, **extra) -> None:
    fcol = "f_pl" if market == "pl" else "f_win"
    for s in ("ltp", "bsp"):
        gs, col, skipped = settle(g, s, market)
        c = bet_cell(gs[col].to_numpy(), gs[fcol].to_numpy(), gs["first"].to_numpy(), side, skipped, s,
                     race=gs["race"].to_numpy() if side == "lay" else None)
        cells.add(run, family, name, set_, s, side, c, in_bh=in_bh, **extra)


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
    if k == 2 or n < 3:
        return top2
    den_jk = np.maximum(A[:, :, None] - a[:, :, None] - a[:, None, :], 1e-300)
    W = ex / den_jk
    S = W.sum((1, 2))
    third = a * (S[:, None] - W.sum(2) - W.sum(1))  # exclude j == i and k == i
    return top2 + third


def race_groups(j: pd.DataFrame) -> list:
    """Rows sorted by (race, pos); every race of the same (n, n_winners) stacks into one matrix."""
    codes, uniques = pd.factorize(pd.MultiIndex.from_arrays([j["n"].to_numpy(), j["n_winners"].to_numpy()]))
    return [(np.flatnonzero(codes == u), int(tup[0]), int(tup[1])) for u, tup in enumerate(uniques)]


def harville_grid(j: pd.DataFrame, chunk: int = 2000) -> np.ndarray:
    """(rows x G) place probability of each runner for every lambda on LAMBDA_GRID; k = the place market's n_winners."""
    P = j["p_win"].to_numpy()
    Q = np.empty((len(j), G))
    groups = race_groups(j)
    for g, lam in enumerate(LAMBDA_GRID):
        for idx, n, k in groups:
            Pm = P[idx].reshape(-1, n)
            res = np.empty_like(Pm)
            for s in range(0, len(Pm), chunk):
                res[s:s + chunk] = topk_batch(Pm[s:s + chunk], float(lam), k)
            Q[idx, g] = res.ravel()
    return Q


def check_topk_against_order_probs(j: pd.DataFrame, rng: np.random.Generator, n_races: int = 300) -> float:
    races = rng.choice(j["race"].unique(), size=min(n_races, j["race"].nunique()), replace=False)
    sub = j[j["race"].isin(races)].sort_values(["race", "pos"])
    worst = 0.0
    for _, grp in sub.groupby("race", sort=False):
        p = grp["p_win"].to_numpy(); k = int(grp["n_winners"].iloc[0])
        if len(p) < 3:
            continue
        for lam in (0.5, 0.85, 1.0, 1.3):
            ref = order_probs(p, lam, lam)["top3" if k == 3 else "top2"]
            mine = topk_batch(p[None, :], lam, k)[0]
            worst = max(worst, float(np.abs(ref - mine).max()))
    return worst


# ----------------------------------------------------------------------------- loading / joining (PLACE side)
def classify(nw: pd.Series, fs: pd.Series) -> np.ndarray:
    nw = nw.to_numpy(); fs = fs.to_numpy()
    return np.select([(nw == 3) & (fs >= 5) & (fs <= 7), (nw == 3) & ((fs == 8) | (fs == 9)),
                      (nw == 2) & (fs >= 5) & (fs <= 7), (nw == 2) & (fs <= 4)], ["T", "C1", "C2", "T4"], "other")


def load_joined(d: Path, cc: str, dev: bool, eval_months: list[str]) -> pd.DataFrame:
    w = pd.read_parquet(d / f"{cc}_win.parquet", columns=COLS).drop_duplicates(["market_id", "selection_id"])
    p = pd.read_parquet(d / f"{cc}_place.parquet", columns=COLS).drop_duplicates(["market_id", "selection_id"])
    for df in (w, p):
        for c in ("market_id", "event_id", "selection_id", "market_name"):
            df[c] = df[c].astype(object)
    if dev:  # lambda path for 2019 uses 2018
        cut = pd.Timestamp("2018-01-01", tz="UTC")
        w = w[w["start"] >= cut]
        p = p[p["start"] >= cut]

    # ---- WIN market filters
    w["won_na"] = w["won"].isna()
    w["priced"] = (w["ltp_pre"] > 1).fillna(False)
    w["inv"] = np.where(w["priced"], 1.0 / w["ltp_pre"], np.nan)
    wm = w.groupby("market_id", sort=False).agg(event_id=("event_id", "first"), start=("start", "first"),
                                                 n=("selection_id", "size"), fs=("field_size", "first"),
                                                 won_sum=("won", "sum"), won_na=("won_na", "sum"),
                                                 priced=("priced", "sum"), book=("inv", "sum"))
    win_settled = (wm["won_na"] == 0) & (wm["won_sum"] >= 1)
    win_priced = wm["priced"] == wm["n"]
    win_book = (wm["book"] >= 0.97) & (wm["book"] <= 1.05)
    s2 = wm[win_settled & win_priced & win_book]
    out({"section": "data_win", "country": cc, "dev": dev, "win_markets": int(len(wm)), "settled_with_winner": int(win_settled.sum()),
         "every_runner_ltp_pre_gt_1": int((win_settled & win_priced).sum()), "book_0.97_1.05": int(len(s2))})

    # ---- PLACE market level
    p = p[p["n_winners"].isin([2, 3])].copy()
    p["won_na"] = p["won"].isna()
    p["priced"] = (p["ltp_pre"] > 1).fillna(False)
    p["inv"] = np.where(p["priced"], 1.0 / p["ltp_pre"], 0.0)
    p["has_t30"] = p["ltp_t30"].notna()
    p["bsp_na"] = ~((p["bsp"] > 1).fillna(False))
    pm = p.groupby("market_id", sort=False).agg(event_id=("event_id", "first"), start=("start", "first"),
                                                 n_winners=("n_winners", "first"), fs=("field_size", "first"),
                                                 n=("selection_id", "size"), won_sum=("won", "sum"), won_na=("won_na", "sum"),
                                                 n_priced=("priced", "sum"), inv_sum=("inv", "sum"), any_t30=("has_t30", "max"),
                                                 bsp_na=("bsp_na", "sum"))
    pm["set"] = classify(pm["n_winners"], pm["fs"])
    pm["settled"] = (pm["won_na"] == 0) & (pm["won_sum"] >= pm["n_winners"])
    pm["full"] = pm["n_priced"] == pm["n"]
    pm["book_pl"] = pm["inv_sum"] / pm["n_winners"]
    pm["book_ok"] = pm["full"] & (pm["book_pl"] >= 0.95) & (pm["book_pl"] <= 1.08)
    pm["month"] = pm["start"].dt.tz_convert(None).dt.to_period("M").astype(str)
    pm["in_eval"] = pm["month"].isin(eval_months)

    # pairs: one WIN market per PLACE market (same event_id, |start difference| <= 10 min, nearest start)
    pairs = pm.reset_index()[["market_id", "event_id", "start", "n", "fs"]].merge(
        s2.reset_index()[["market_id", "event_id", "start", "n", "fs"]].rename(
            columns={"market_id": "win_mid", "start": "start_w", "n": "n_w", "fs": "fs_w"}), on="event_id")
    pairs["dt"] = (pairs["start"] - pairs["start_w"]).abs()
    pairs = pairs[pairs["dt"] <= pd.Timedelta(minutes=10)]
    n_multi = int((pairs.groupby("market_id").size() > 1).sum())
    pairs = pairs.sort_values(["market_id", "dt", "win_mid"]).drop_duplicates("market_id", keep="first")
    pm = pm.join(pairs.set_index("market_id")[["win_mid", "n_w", "fs_w"]])
    pm["paired"] = pm["win_mid"].notna()

    # stage counts per set
    stages = {}
    for set_ in ("all", "T", "C1", "C2", "T4"):
        s = pm if set_ == "all" else pm[pm["set"] == set_]
        st = s["settled"]; pr = st & s["paired"]; fu = pr & s["full"]; bk = pr & s["book_ok"]
        row = {"0_place_markets": int(len(s)), "1_place_settled": int(st.sum()), "2_paired_with_filtered_WIN": int(pr.sum()),
               "3_every_place_runner_ltp_pre_gt_1": int(fu.sum()), "4_place_book_0.95_1.08_(MAIN)": int(bk.sum()),
               "5_MAIN_in_eval_window": int((bk & s["in_eval"]).sum()), "5_SENSITIVITY_in_eval_window": int((pr & s["in_eval"]).sum())}
        if set_ in ("T", "C1", "C2", "T4"):
            sp = s[pr]
            row["runners_no_place_ltp_pre_share"] = r4(1 - sp["n_priced"].sum() / sp["n"].sum()) if len(sp) else None
            row["markets_any_place_ltp_t30_share"] = r4(sp["any_t30"].mean()) if len(sp) else None
            row["markets_win_fs_ne_place_fs_share"] = r4((sp["fs_w"] != sp["fs"]).mean()) if len(sp) else None
            row["place_bsp_missing_runner_share"] = r4(sp["bsp_na"].sum() / sp["n"].sum()) if len(sp) else None
        stages[set_] = row
    out({"section": "data_place", "country": cc, "dev": dev, "place_markets_with_several_win_candidates": n_multi, "stages": stages})
    sp = pm[pm["settled"] & pm["paired"]]
    xt = pd.crosstab(sp["n_winners"], sp["fs"].clip(upper=12))
    out({"section": "n_winners_x_field_size_settled_paired", "country": cc,
         "table": {f"nw={k}": {f"fs={c}{'+' if c == 12 else ''}": int(v) for c, v in xt.loc[k].items()} for k in xt.index}})

    # ---- WIN side per pair: p_win, rank, Harville grid over the WIN field
    pairs2 = pm[pm["settled"] & pm["paired"]].reset_index()[["market_id", "win_mid", "n_winners"]].rename(columns={"market_id": "pl_mid"})
    wj = pairs2.merge(w[["market_id", "selection_id", "ltp_pre", "ltp_t30", "bsp", "won", "dead_heat_share"]].rename(
        columns={"market_id": "win_mid", "ltp_pre": "ltp_win", "ltp_t30": "t30_win", "bsp": "bsp_win", "won": "won_win",
                 "dead_heat_share": "f_win"}), on="win_mid")
    wj["inv"] = 1.0 / wj["ltp_win"]
    wj["race"] = pd.factorize(wj["pl_mid"])[0]
    wj["p_win"] = wj["inv"] / wj.groupby("race")["inv"].transform("sum")
    wj = wj.sort_values(["race", "ltp_win", "selection_id"]).reset_index(drop=True)
    wj["pos"] = wj.groupby("race").cumcount()
    wj["rank_win"] = wj["pos"] + 1
    wj["n"] = wj.groupby("race")["pos"].transform("size")
    wj["qrow"] = np.arange(len(wj))

    # ---- runner level: PLACE rows with their WIN row (inner), priced place rows only
    j = p.loc[p["market_id"].isin(pairs2["pl_mid"]),
              ["market_id", "selection_id", "market_name", "start", "n_winners", "field_size", "ltp_pre", "ltp_t30", "bsp", "won", "dead_heat_share"]]
    j = j.rename(columns={"market_id": "pl_mid", "ltp_pre": "ltp_pl", "ltp_t30": "t30_pl", "bsp": "bsp_pl", "won": "placed", "dead_heat_share": "dh"})
    n_place_rows = len(j)
    j = j.merge(wj[["pl_mid", "selection_id", "ltp_win", "t30_win", "bsp_win", "won_win", "f_win", "p_win", "rank_win", "n", "race", "qrow"]],
                on=["pl_mid", "selection_id"], how="inner")
    n_unmatched = n_place_rows - len(j)
    j["f_win"] = j["f_win"].fillna(0.0)
    j["f_pl"] = np.where(j["placed"] > 0, np.minimum(1.0, j["n_winners"] * j["dh"].fillna(0.0)), 0.0)
    j = j[(j["ltp_pl"] > 1).fillna(False)].copy()  # priced place rows only (both runs)
    j["inv_pl"] = 1.0 / j["ltp_pl"]
    j["q_mkt"] = (j["n_winners"] * j["inv_pl"] / j.groupby("pl_mid")["inv_pl"].transform("sum")).clip(1e-4, 0.999)
    for col in ("set", "full", "book_ok", "book_pl", "month", "in_eval", "fs_w", "any_t30"):
        j[col] = j["pl_mid"].map(pm[col])
    for col in ("full", "book_ok", "in_eval", "any_t30"):
        j[col] = j[col].astype(bool)
    j["set"] = j["set"].astype(str)
    j["month"] = j["month"].astype(str)
    j["harness"] = j["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b").to_numpy()
    j["pl_band"] = pd.cut(j["ltp_pl"], PL_EDGES, labels=PL_LABELS).astype(str)
    j["win_band"] = pd.cut(j["ltp_win"], WIN_EDGES, labels=WIN_LABELS).astype(str)
    j = j.drop(columns=["market_name", "inv_pl"]).reset_index(drop=True)
    t0 = time.time()
    Q = harville_grid(wj)
    out({"section": "harville_precompute", "country": cc, "win_rows": int(len(wj)), "races": int(wj["race"].nunique()),
         "lambdas": G, "seconds": round(time.time() - t0, 1),
         "place_rows_without_win_row": n_unmatched, "place_rows_without_win_row_share": r4(n_unmatched / max(n_place_rows, 1)),
         "priced_place_rows": int(len(j)), "harness_share_T_races": r4(j[j["set"] == "T"].drop_duplicates("pl_mid")["harness"].mean()),
         "bsp_pl_present_share_T": r4((j.loc[j["set"] == "T", "bsp_pl"] > 1).mean()),
         "bsp_pl_present_share_C1": r4((j.loc[j["set"] == "C1", "bsp_pl"] > 1).mean()),
         "place_dead_heat_runners": int(((j["f_pl"] > 0) & (j["f_pl"] < 1)).sum())})
    Qj = Q[j["qrow"].to_numpy()]
    return j, Qj, wj


# ----------------------------------------------------------------------------- lambda paths
def lambda_path(months: np.ndarray, y: np.ndarray, Q: np.ndarray, eval_months: list[str]) -> dict:
    """lambda_m = argmin over the grid of the mean per-runner binary log loss of q_harv(lambda) vs the place flag
    over the LAMBDA_WINDOW calendar months strictly before m (at least LAMBDA_MIN_MONTHS with data)."""
    Qc = np.clip(Q, 1e-6, 1 - 1e-6)
    LL = -(y[:, None] * np.log(Qc) + (1 - y[:, None]) * np.log(1 - Qc))
    all_months = [str(x) for x in pd.period_range(min(months.min(), eval_months[0]), max(months.max(), eval_months[-1]), freq="M")]
    mcode = pd.Categorical(months, categories=all_months).codes
    S = np.zeros((len(all_months), G)); N = np.zeros(len(all_months))
    np.add.at(S, mcode, LL)
    np.add.at(N, mcode, 1.0)
    lam = {}
    for m in eval_months:
        i = all_months.index(m)
        win = [k for k in range(max(0, i - LAMBDA_WINDOW), i) if N[k] > 0]
        if len(win) < LAMBDA_MIN_MONTHS:
            continue
        mean_ll = S[win].sum(0) / N[win].sum()
        g = int(np.argmin(mean_ll))
        lam[m] = (g, float(LAMBDA_GRID[g]), int(N[win].sum()), float(mean_ll[g]))
    return lam


def fit_lambdas(j: pd.DataFrame, Q: np.ndarray, eval_months: list[str], cc: str) -> tuple[dict, dict]:
    """Pooled lambda per n_winners on fully priced place markets (any book), and lambda_T on T markets only."""
    full = j["full"].to_numpy(bool)
    paths = {}
    for k in (2, 3):
        sel = full & (j["n_winners"].to_numpy() == k)
        paths[k] = lambda_path(j.loc[sel, "month"].to_numpy(), j.loc[sel, "placed"].to_numpy(float), Q[sel], eval_months)
    selT = full & (j["set"].to_numpy() == "T")
    lamT = lambda_path(j.loc[selT, "month"].to_numpy(), j.loc[selT, "placed"].to_numpy(float), Q[selT], eval_months)
    out({"section": "lambda_path", "country": cc, "fit_set": "fully priced settled place markets paired with a filtered WIN market, any place book",
         "by_month": {m: {"lambda_nw3_pooled": paths[3].get(m, (None, None))[1], "lambda_nw2_pooled": paths[2].get(m, (None, None))[1],
                          "lambda_T_only": lamT.get(m, (None, None))[1], "fit_runners_nw3": paths[3].get(m, (None, None, None))[2],
                          "fit_runners_T": lamT.get(m, (None, None, None))[2]} for m in eval_months}})
    return paths, lamT


def build_eval(j: pd.DataFrame, Q: np.ndarray, paths: dict, lamT: dict) -> pd.DataFrame:
    key = j["n_winners"].astype(str) + "|" + j["month"]
    idx = key.map({f"{k}|{m}": v[0] for k, d in paths.items() for m, v in d.items()})
    e = j[idx.notna() & j["in_eval"]].copy()
    li = idx[idx.notna() & j["in_eval"]].to_numpy(int)
    Qe = Q[(idx.notna() & j["in_eval"]).to_numpy()]
    e["lambda"] = LAMBDA_GRID[li]
    e["q_harv"] = np.clip(Qe[np.arange(len(e)), li], 1e-4, 0.999)
    e["e"] = e["q_harv"] * e["ltp_pl"] - 1.0
    liT = e["month"].map({m: v[0] for m, v in lamT.items()})
    okT = liT.notna().to_numpy()
    qT = np.full(len(e), np.nan)
    qT[okT] = Qe[np.flatnonzero(okT), liT[okT].to_numpy(int)]
    e["q_harv_T"] = np.clip(qT, 1e-4, 0.999)
    e["e_T"] = e["q_harv_T"] * e["ltp_pl"] - 1.0
    return e.reset_index(drop=True)


# ----------------------------------------------------------------------------- one run (MAIN or SENSITIVITY)
def analyse(e: pd.DataFrame, run: str, cells: Cells, mid: pd.Timestamp) -> None:
    e = e.copy()
    e["first"] = e["start"] < mid
    races = e.drop_duplicates("pl_mid")
    out({"run": run, "section": "evaluation_set", "median_start_used_for_halves": str(mid),
         "races_by_set": races["set"].value_counts().to_dict(), "runners_by_set": e["set"].value_counts().to_dict(),
         "T_races_first_half": int(races.loc[races["set"] == "T", "first"].sum()),
         "T_races_second_half": int((~races.loc[races["set"] == "T", "first"]).sum()),
         "mean_by_set": {s: {"q_harv": r4(e.loc[e["set"] == s, "q_harv"].mean()), "q_mkt": r4(e.loc[e["set"] == s, "q_mkt"].mean()),
                             "placed": r4(e.loc[e["set"] == s, "placed"].mean()), "book_pl": r4(races.loc[races["set"] == s, "book_pl"].mean()),
                             "lambda_mean": r4(e.loc[e["set"] == s, "lambda"].mean())} for s in ("T", "C1", "C2", "T4")},
         "edge_quantiles_T": [r4(v) for v in np.quantile(e.loc[e["set"] == "T", "e"], [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])] if (e["set"] == "T").any() else None,
         "edge_quantiles_C1": [r4(v) for v in np.quantile(e.loc[e["set"] == "C1", "e"], [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])] if (e["set"] == "C1").any() else None})

    # ---- 1. calibration: place-price band x set x back every runner x 2 settlements (36 cells)
    for set_ in SETS:
        for band in PL_LABELS:
            g = e[(e["set"] == set_) & (e["pl_band"] == band)]
            extra = {"n": int(len(g)), "hit_rate": r4((g["f_pl"] > 0).mean()), "mean_q_mkt": r4(g["q_mkt"].mean()),
                     "mean_q_harv": r4(g["q_harv"].mean()), "hit_over_q_harv": r4((g["f_pl"] > 0).mean() / g["q_harv"].mean()) if len(g) else None}
            rule_cells(cells, run, "calibration", f"band {band}", set_, g, "back", **extra)
    # T4 calibration (counts only, not bet cells)
    g4 = e[e["set"] == "T4"]
    out({"run": run, "section": "T4_calibration", "runners": int(len(g4)), "markets": int(g4["pl_mid"].nunique()),
         "hit_rate": r4((g4["f_pl"] > 0).mean()), "mean_q_mkt": r4(g4["q_mkt"].mean()), "mean_q_harv": r4(g4["q_harv"].mean()),
         "by_band": {b: [int((g4["pl_band"] == b).sum()), r4((g4.loc[g4["pl_band"] == b, "f_pl"] > 0).mean()),
                         r4(g4.loc[g4["pl_band"] == b, "q_mkt"].mean()), r4(g4.loc[g4["pl_band"] == b, "q_harv"].mean())] for b in PL_LABELS}})
    # edge-bin descriptive (not bet cells): placed vs implied by edge bin, per set
    bins = [-np.inf, -0.10, 0.0, 0.10, 0.25, np.inf]; labs = ["<-0.10", "[-0.10,0)", "[0,0.10)", "[0.10,0.25)", ">=0.25"]
    for set_ in SETS:
        g = e[e["set"] == set_]
        eb = pd.cut(g["e"], bins, labels=labs, right=False).astype(str)
        out({"run": run, "section": "edge_bins", "set": set_,
             "n_placed_qmkt_qharv": {lab: [int((eb == lab).sum()), r4((g.loc[eb == lab, "f_pl"] > 0).mean()),
                                           r4(g.loc[eb == lab, "q_mkt"].mean()), r4(g.loc[eb == lab, "q_harv"].mean())] for lab in labs}})

    # ---- 2. rules on each set (30 cells) + secondary lambdaT (6, outside BH)
    def rule_mask(g, rule, ecol="e"):
        if rule.startswith("a_theta"):
            return g[ecol] > float(rule[len("a_theta"):])
        if rule.startswith("b_lay"):
            return g[ecol] < -LAY_T
        return g["rank_win"].between(2, 4)

    for set_ in SETS:
        g = e[e["set"] == set_]
        for rule in RULES:
            side = "lay" if rule.startswith("b_lay") else "back"
            rule_cells(cells, run, "rule", rule, set_, g[rule_mask(g, rule)], side)
    gT = e[(e["set"] == "T") & e["q_harv_T"].notna()]
    for theta in THETAS:
        rule_cells(cells, run, "rule_lambdaT", f"a_theta{theta:.2f}_lambdaT", "T", gT[gT["e_T"] > theta], "back", in_bh=False)
    # secondary (outside BH): rule (a) theta = 0.10 on T split by whether the place market had traded 30 min out
    gT = e[e["set"] == "T"]
    for flag, lab in ((True, "t30_present"), (False, "t30_absent")):
        g = gT[(gT["any_t30"] == flag) & (gT["e"] > 0.10)]
        rule_cells(cells, run, "rule_T_by_t30", f"a_theta0.10_{lab}", "T", g, "back", in_bh=False,
                   markets=int(gT.loc[gT["any_t30"] == flag, "pl_mid"].nunique()))

    # differences T - C1, T - C2 per rule and settlement
    def find(fam, name, set_, s):
        for r in cells.rows:
            if r["run"] == run and r["family"] == fam and r["name"] == name and r["set"] == set_ and r["settlement"] == s:
                return r
        return None
    for rule in RULES:
        for s in ("ltp", "bsp"):
            t = find("rule", rule, "T", s)
            for ctrl in ("C1", "C2"):
                c = find("rule", rule, ctrl, s)
                if t and c and t.get("roi_c5") is not None and c.get("roi_c5") is not None and t.get("se") and c.get("se"):
                    d = t["roi_c5"] - c["roi_c5"]; sed = math.sqrt(t["se"] ** 2 + c["se"] ** 2)
                    out({"run": run, "section": "difference", "rule": rule, "settlement": s, "pair": f"T-{ctrl}", "roi_c5_T": t["roi_c5"],
                         "roi_c5_C": c["roi_c5"], "bets_T": t["bets"], "bets_C": c["bets"], "diff": r4(d), "se_diff": r4(sed), "z_diff": r4(d / sed),
                         "gt_2se": bool(d > 2 * sed)})
                else:
                    out({"run": run, "section": "difference", "rule": rule, "settlement": s, "pair": f"T-{ctrl}", "diff": None})

    # ---- 3. WIN re-pricing in T races (8 cells) + WIN calibration for T and C1
    gT = e[e["set"] == "T"]
    for band in WIN_LABELS:
        g = gT[gT["win_band"] == band]
        rule_cells(cells, run, "win_repricing", f"WIN band {band}", "T", g, "back", market="win",
                   n=int(len(g)), calib_sum_won_over_sum_pwin=r4(g["f_win"].sum() / g["p_win"].sum()) if len(g) else None)
    g = gT[gT["rank_win"] == 1]
    rule_cells(cells, run, "win_repricing", "WIN favourite", "T", g, "back", market="win",
               n=int(len(g)), calib_sum_won_over_sum_pwin=r4(g["f_win"].sum() / g["p_win"].sum()) if len(g) else None)
    for set_ in ("T", "C1", "C2"):
        g = e[e["set"] == set_]
        out({"run": run, "section": "win_calibration", "set": set_, "races": int(g["pl_mid"].nunique()),
             "by_band_n_calib": {b: [int((g["win_band"] == b).sum()),
                                     r4(g.loc[g["win_band"] == b, "f_win"].sum() / g.loc[g["win_band"] == b, "p_win"].sum()) if (g["win_band"] == b).any() else None]
                                 for b in WIN_LABELS},
             "favourite_n_calib": [int((g["rank_win"] == 1).sum()), r4(g.loc[g["rank_win"] == 1, "f_win"].sum() / g.loc[g["rank_win"] == 1, "p_win"].sum()) if (g["rank_win"] == 1).any() else None],
             "overall_calib": r4(g["f_win"].sum() / g["p_win"].sum()) if len(g) else None})


def analyse_gb(e: pd.DataFrame, run: str, cells: Cells) -> pd.Timestamp | None:
    e = e.copy()
    racesT = e[e["set"] == "T"].drop_duplicates("pl_mid")
    if len(racesT) == 0:
        out({"run": run, "section": "gb_replication", "note": "no GB T races"})
        for set_ in ("T", "C1"):
            for s in ("ltp", "bsp"):
                cells.add(run, "gb_replication", "a_theta0.10", f"GB_{set_}", s, "back", bet_cell([], [], [], "back", 0, s))
        return None
    mid = pd.Timestamp(racesT["start"].sort_values().to_numpy()[len(racesT) // 2])
    e["first"] = e["start"] < mid
    races = e.drop_duplicates("pl_mid")
    out({"run": run, "section": "gb_evaluation_set", "median_start_gb_T": str(mid), "races_by_set": races["set"].value_counts().to_dict(),
         "runners_by_set": e["set"].value_counts().to_dict(),
         "mean_by_set": {s: {"q_harv": r4(e.loc[e["set"] == s, "q_harv"].mean()), "q_mkt": r4(e.loc[e["set"] == s, "q_mkt"].mean()),
                             "placed": r4(e.loc[e["set"] == s, "placed"].mean())} for s in ("T", "C1", "C2")}})
    for set_ in ("T", "C1"):
        g = e[(e["set"] == set_) & (e["e"] > 0.10)]
        rule_cells(cells, run, "gb_replication", "a_theta0.10", f"GB_{set_}", g, "back")
    return mid


# ----------------------------------------------------------------------------- BH, primary, summary
def summarise(cells: Cells, run: str) -> dict:
    rows = [r for r in cells.rows if r["run"] == run]
    bh_rows = [r for r in rows if r["in_bh"]]
    passed = benjamini_hochberg([r.get("p") for r in bh_rows], FDR_Q)
    for r, ok in zip(bh_rows, passed):
        r["bh_pass"] = bool(ok)
        r["positive"] = bool(ok and r.get("both_halves_2se"))
    for r in rows:
        if not r["in_bh"]:
            r["bh_pass"] = None
            r["positive"] = None
            r["positive_if_bh_passed"] = bool(r.get("both_halves_2se"))
    table = [{"family": r["family"], "name": r["name"], "set": r["set"], "settlement": r["settlement"], "side": r["side"], "bets": r["bets"],
              "skipped": r["skipped"], "roi_c0": r.get("roi_c0"), "roi_c5": r.get("roi_c5"), "roi_c8": r.get("roi_c8"), "se": r.get("se"),
              "halves_roi_c5": r.get("halves_roi_c5"), "halves_se_c5": r.get("halves_se_c5"), "z": r.get("z"), "p": r4(r.get("p")),
              "bh_pass": r.get("bh_pass"), "positive": r.get("positive"), "in_bh": r["in_bh"],
              **({"roi_c5_tick_down": r.get("roi_c5_tick_down")} if "roi_c5_tick_down" in r else {}),
              **({"liab_roi_c5": r.get("liab_roi_c5"), "liab_se": r.get("liab_se"), "races_with_lay": r.get("races_with_lay")} if r["side"] == "lay" else {})}
             for r in rows]
    prim = [r for r in bh_rows if r["family"] == "rule" and r["name"] == PRIMARY_RULE and r["set"] == PRIMARY_SET and r["settlement"] == PRIMARY_SETTLEMENT]
    ctrl = [r for r in bh_rows if r["family"] == "rule" and r["name"] == PRIMARY_RULE and r["set"] == "C1" and r["settlement"] == PRIMARY_SETTLEMENT]
    primary = None
    if prim:
        r = prim[0]
        part_i = bool(r["positive"])
        part_ii = None
        diff = None
        if ctrl and r.get("se") and ctrl[0].get("se"):
            c = ctrl[0]
            d = r["roi_c5"] - c["roi_c5"]; sed = math.sqrt(r["se"] ** 2 + c["se"] ** 2)
            diff = {"roi_c5_T": r["roi_c5"], "roi_c5_C1": c["roi_c5"], "diff": r4(d), "se_diff": r4(sed), "z_diff": r4(d / sed)}
            part_ii = bool(d > 2 * sed)
        verdict = ("reduced-field effect (both parts hold)" if part_i and part_ii else
                   "general Dr Z value, belongs to win_to_place (part i holds, part ii does not)" if part_i else
                   "no positive primary (part i fails)")
        primary = {"rule": PRIMARY_RULE, "set": PRIMARY_SET, "settlement": PRIMARY_SETTLEMENT, "bets": r["bets"], "skipped_no_bsp": r["skipped"],
                   "roi_c0": r.get("roi_c0"), "roi_c5": r.get("roi_c5"), "roi_c8": r.get("roi_c8"), "se": r.get("se"), "z": r.get("z"),
                   "halves_roi_c5": r.get("halves_roi_c5"), "halves_se_c5": r.get("halves_se_c5"), "halves_bets": r.get("halves_bets"),
                   "bh_pass": r["bh_pass"], "part_i_positive": part_i, "T_minus_C1": diff, "part_ii_diff_gt_2se": part_ii, "verdict": verdict}
    # opposite-half development: cells at > 2 SE in one half, evaluated in the other
    opp = []
    for r in rows:
        h, s = r.get("halves_roi_c5") or [None, None], r.get("halves_se_c5") or [None, None]
        for dev_h, test_h in ((0, 1), (1, 0)):
            if h[dev_h] is not None and s[dev_h] and h[dev_h] > 2 * s[dev_h]:
                opp.append({"dev_half": dev_h + 1, "family": r["family"], "name": r["name"], "set": r["set"], "settlement": r["settlement"],
                            "dev_roi": h[dev_h], "dev_z": r4(h[dev_h] / s[dev_h]), "test_roi": h[test_h],
                            "test_z": r4(h[test_h] / s[test_h]) if s[test_h] else None,
                            "holds_test_gt_2se": bool(s[test_h] and h[test_h] > 2 * s[test_h])})
    summary = {"run": run, "section": "BH", "q": FDR_Q, "bet_cells_tested": len(bh_rows), "expected_cells": EXPECTED_CELLS,
               "secondary_cells_outside_bh": len(rows) - len(bh_rows), "bh_passes": int(passed.sum()),
               "positive_cells": [f"{r['family']}:{r['name']}:{r['set']}@{r['settlement']}" for r in bh_rows if r["positive"]],
               "secondary_both_halves_2se": [f"{r['name']}:{r['set']}@{r['settlement']}" for r in rows if not r["in_bh"] and r.get("both_halves_2se")],
               "primary": primary, "opposite_half_development": opp, "cells": table}
    out(summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent.parent / "data/raw/betfair/split")
    ap.add_argument("--dev", action="store_true", help="evaluation set = 2019 only (lambda path uses 2018)")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--no-gb", action="store_true")
    ap.add_argument("--cells-csv", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t_all = time.time()
    if a.dev:
        eval_months = [f"2019-{m:02d}" for m in range(1, 13)]
    else:
        eval_months = [str(p) for p in pd.period_range("2017-06", "2020-11", freq="M")]
    out({"section": "config", "dev": a.dev, "eval_months": [eval_months[0], eval_months[-1]], "lambda_grid": [float(x) for x in LAMBDA_GRID],
         "lambda_window_months": LAMBDA_WINDOW, "lambda_min_months": LAMBDA_MIN_MONTHS, "fdr_q": FDR_Q, "commissions": COMMS,
         "thetas": THETAS, "lay_threshold": -LAY_T, "sets": {"T": "n_winners==3 & 5<=field_size<=7", "C1": "n_winners==3 & field_size in {8,9}",
                                                                "C2": "n_winners==2 & 5<=field_size<=7", "T4": "n_winners==2 & field_size<=4 (counts only)"},
         "primary": f"rule {PRIMARY_RULE} on AU {PRIMARY_SET} @ {PRIMARY_SETTLEMENT}: positive (BH + both halves > 2 SE) AND T-C1 diff > 2 SE_diff",
         "power_statement": POWER})
    cells = Cells()
    runs = ["MAIN"] + ([] if a.no_sensitivity else ["SENSITIVITY"])

    # ---- AU
    j_au, Q_au, wj_au = load_joined(a.dir, "au", a.dev, eval_months)
    worst = check_topk_against_order_probs(wj_au, rng)
    out({"section": "harville_check", "country": "au", "max_abs_diff_vs_order_probs": worst})
    del wj_au
    paths_au, lamT_au = fit_lambdas(j_au, Q_au, eval_months, "au")
    e_au = build_eval(j_au, Q_au, paths_au, lamT_au)
    del j_au, Q_au
    main_sel = (e_au["full"] & e_au["book_ok"]).to_numpy(bool)
    racesT = e_au[(e_au["set"] == "T") & main_sel].drop_duplicates("pl_mid")
    mid = pd.Timestamp(racesT["start"].sort_values().to_numpy()[len(racesT) // 2])
    out({"section": "halves", "country": "au", "median_start_of_MAIN_T_races": str(mid), "T_races_main": int(len(racesT))})

    # ---- GB
    e_gb = None
    if not a.no_gb:
        j_gb, Q_gb, wj_gb = load_joined(a.dir, "gb", a.dev, eval_months)
        del wj_gb
        paths_gb, lamT_gb = fit_lambdas(j_gb, Q_gb, eval_months, "gb")
        e_gb = build_eval(j_gb, Q_gb, paths_gb, lamT_gb)
        del j_gb, Q_gb

    summaries = {}
    for run in runs:
        t0 = time.time()
        sel = main_sel if run == "MAIN" else np.ones(len(e_au), bool)
        analyse(e_au[sel].reset_index(drop=True), run, cells, mid)
        if e_gb is not None:
            selg = (e_gb["full"] & e_gb["book_ok"]).to_numpy(bool) if run == "MAIN" else np.ones(len(e_gb), bool)
            analyse_gb(e_gb[selg].reset_index(drop=True), run, cells)
        else:
            for set_ in ("T", "C1"):
                for s in ("ltp", "bsp"):
                    cells.add(run, "gb_replication", "a_theta0.10", f"GB_{set_}", s, "back", bet_cell([], [], [], "back", 0, s), note="--no-gb")
        summaries[run] = summarise(cells, run)
        out({"run": run, "section": "timing", "seconds": round(time.time() - t0, 1)})

    if a.cells_csv is not None:
        cells.frame().to_csv(a.cells_csv, index=False)
    out({"section": "done", "total_seconds": round(time.time() - t_all, 1), "cells_total": len(cells.rows),
         "bh_cells_per_run": {run: s["bet_cells_tested"] for run, s in summaries.items()}})
    return 0


if __name__ == "__main__":
    sys.exit(main())

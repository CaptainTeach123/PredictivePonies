"""Experiment "bsp_and_laying": Betfair SP vs the last traded price, and laying,
on Australian WIN markets (data/raw/betfair/split/au_win.parquet).

    python scripts/bf_bsp_and_laying.py [--end 2018-06-01] [--code gallops|harness|both]
                                        [--cells-csv /path/cells.csv]

A backer who must decide before the jump can either take the last traded price
(ltp_pre: the last trade before the market turned in-play, one-minute snapshots
on the Basic plan, so neither guaranteed available nor fresh [A-53]) or bet at
the Betfair SP (bsp: matched after the off, so not a usable feature, but a
realistic execution for a bet decided before the off). Harness (Pace/Trot in the
market name) and gallops are run separately.

Sections (JSON lines, summary numbers only, never horse names):

  A. bsp/ltp_pre by ltp_pre price band x field size (and favourites, and the
     t30 -> pre move for the ~39% of runners that traded 30 min out): quantiles
     of the ratio, share of runners whose SP beat the last price (by >= 1 tick),
     the back ROI of 1 unit at ltp_pre and at bsp on the same runners, and the
     PAIRED difference (bsp return - ltp_pre return) with its SE and halves.
  A2. Walk-forward chooser: each month, for every band x field cell, take SP if
     the paired difference over strictly earlier months is > k SE (k = 0, 1, 2;
     6-month burn-in, >= 30 earlier bets), else take the last price. Reports the
     realised gain against always-LTP and always-SP, with SE and halves.
  B. Lay returns per unit LIABILITY (stake = 1/(price-1); win stake x (1 - c)
     when the runner loses, lose 1 when it wins) and per unit backer STAKE, at
     ltp_pre (also one tick higher = worse for the layer) and at bsp, selected
     by ltp_pre band (bettable) and by bsp band (descriptive, not bettable);
     band x field size too. Commission 0 / 5% / 8% on net winnings [A-54].
  C. "Lay the field" style rules: lay every runner at ltp_pre >= 30 / 50 / 100
     (by field-size group) and the odds-on favourite (< 1.5, 1.5-2, < 2) in
     small fields (2-5, 6-8, 2-8, all), at ltp_pre and at bsp.
  D. Benjamini-Hochberg (FDR 5%) over every cell of each family, the both-halves
     > 2 SE rule, and the opposite-half development check (cells picked in one
     half at > 2 SE, evaluated in the other).

Nothing is fitted on outcomes except the A2 chooser, which uses strictly earlier
months. Price bands and thresholds are fixed in advance.

Statistics: per-bet sample SE at 5% commission, plus a binomial "null" SE from
the implied probabilities q = 1/ltp_pre (var = q(1-q)(r_won - r_lost)^2), which
is what the sample SE understates when a cell has seen almost no winners (a lay
of 100 runners at 150 with no winner has a sample SE of ~0). BH p-values use
max(sample SE, null SE); cells with fewer than MIN_EVENTS winners (or fewer
than MIN_HALF_EVENTS in a half) are reported but not testable.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ponies.betfair import net_return  # noqa: E402  reference implementation [A-54]

COMMISSIONS = (0.0, 0.05, 0.08)
PRICE_BANDS = [1.0, 2.0, 3.0, 4.5, 6.5, 10.0, 15.0, 25.0, 50.0, 1e9]
BAND_LABELS = [f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+" for lo, hi in zip(PRICE_BANDS, PRICE_BANDS[1:])]
FIELD_EDGES = [2, 6, 9, 12, 99]
FIELD_LABELS = ["2-5", "6-8", "9-11", "12+"]
MOVE_CUTS = [-np.inf, -0.25, -0.1, 0.0, 0.1, 0.25, np.inf]  # log(ltp_t30 / ltp_pre); > 0 = firmed
MOVE_LABELS = ["drift>25%", "drift 10-25%", "drift 0-10%", "firm 0-10%", "firm 10-25%", "firm>25%"]
LONGSHOT_THRESHOLDS = (30.0, 50.0, 100.0)
ODDS_ON = {"<1.5": (1.0, 1.5), "1.5-2": (1.5, 2.0), "<2": (1.0, 2.0)}
SMALL_FIELDS = {"2-5": (2, 5), "6-8": (6, 8), "2-8": (2, 8), "all": (2, 99)}
BURN_IN_MONTHS = 6
MIN_FIT_BETS = 30
K_SE = (0.0, 1.0, 2.0)
FDR = 0.05
MIN_BETS = 100  # cells below this are still listed, but flagged thin
MIN_EVENTS = 10  # winners (= losing lay bets) needed before a cell's SE / p-value is trusted
MIN_HALF_EVENTS = 5  # the same for each half in the both-halves rule
C_TEST = 0.05  # the commission at which SE, z and the both-halves rule are evaluated

# Betfair price ladder: (lower bound inclusive, upper bound exclusive, increment)
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1001.0, 10.0)]


# ----------------------------------------------------------------------------- utilities


def out(obj) -> None:
    print(json.dumps(obj, default=_json_default), flush=True)


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    if isinstance(o, pd.Period):
        return str(o)
    raise TypeError(str(type(o)))


def r4(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) else round(x, 4)


def r6(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) else round(x, 6)


def back_returns(price, won, c: float) -> np.ndarray:
    """1 unit backed: (price - 1)(1 - c) if it won, else -1 (vectorised net_return [A-54])."""
    price = np.asarray(price, dtype=float)
    return np.where(won, (price - 1.0) * (1.0 - c), -1.0)


def lay_liability_returns(price, won, c: float) -> np.ndarray:
    """Lay with 1 unit of liability: backer's stake 1/(price-1); win it less
    commission when the runner loses, pay the liability when it wins."""
    price = np.asarray(price, dtype=float)
    return np.where(won, -1.0, (1.0 - c) / (price - 1.0))


def lay_stake_returns(price, won, c: float) -> np.ndarray:
    """Lay 1 unit of backer's stake: win (1 - c) when the runner loses, pay (price - 1) when it wins."""
    price = np.asarray(price, dtype=float)
    return np.where(won, -(price - 1.0), 1.0 - c)


def _increment(q: np.ndarray) -> np.ndarray:
    inc = np.full_like(q, 10.0)
    for lo, hi, step in TICKS:
        inc = np.where((q >= lo) & (q < hi), step, inc)
    return inc


def tick_down(price) -> np.ndarray:
    """One Betfair tick below (worse for a backer)."""
    p = np.asarray(price, dtype=float)
    return np.maximum(1.01, np.round(p - _increment(p - 1e-9), 2))


def tick_up(price) -> np.ndarray:
    """One Betfair tick above (worse for a layer)."""
    p = np.asarray(price, dtype=float)
    return np.minimum(1000.0, np.round(p + _increment(p + 1e-9), 2))


def one_sided_p(z: float) -> float:
    if z is None or math.isnan(z):
        return float("nan")
    return 0.5 * math.erfc(z / math.sqrt(2))


def benjamini_hochberg(pvals, q: float = FDR) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    rej = np.zeros(len(p), dtype=bool)
    if m == 0:
        return rej
    order = np.argsort(p[ok])
    sorted_p = p[ok][order]
    thresh = q * (np.arange(1, m + 1) / m)
    below = np.flatnonzero(sorted_p <= thresh)
    if len(below):
        k = below.max()
        rej[np.flatnonzero(ok)[order[: k + 1]]] = True
    return rej


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(x[0]), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(n))


def cell(ret: dict, first: np.ndarray, digits: int = 4, won: np.ndarray | None = None,
         q: np.ndarray | None = None, outcomes: tuple | None = None) -> dict:
    """ret = {commission: per-bet return array}. ROI at every commission, per-bet
    sample SE at 5%, both halves at 5%, the both-halves rule and a one-sided
    p-value. ``won`` gates the statistics on the number of events; ``q`` (implied
    probability at ltp_pre) and ``outcomes`` = (return if won, return if lost) at
    5% give the binomial null SE; p uses max(sample SE, null SE)."""
    rnd = r4 if digits == 4 else (r6 if digits == 6 else (lambda x: None if math.isnan(float(x)) else round(float(x), digits)))
    n = int(len(first))
    res = {"bets": n}
    if n == 0:
        return res
    for c in COMMISSIONS:
        res[f"roi_c{int(c * 100)}"] = rnd(ret[c].mean())
    r5 = ret[C_TEST]
    m, se = _mean_se(r5)
    res["se"] = rnd(se)
    n_won = int(won.sum()) if won is not None else None
    res["n_won"] = n_won
    se_null = float("nan")
    if q is not None and outcomes is not None:
        r_won, r_lost = outcomes
        se_null = math.sqrt(float((q * (1.0 - q) * (r_won - r_lost) ** 2).sum())) / n
        res["se_null"] = rnd(se_null)
        res["expected_won_at_ltp"] = r4(q.sum())
    events_ok = n_won is None or n_won >= MIN_EVENTS
    halves, halves_se, halves_n, halves_won = [], [], [], []
    for mask in (first, ~first):
        hm, hs = _mean_se(r5[mask])
        halves_n.append(int(mask.sum()))
        halves.append(rnd(hm))
        halves_se.append(rnd(hs))
        halves_won.append(int(won[mask].sum()) if won is not None else None)
    res["halves_roi_c5"] = halves
    res["halves_se_c5"] = halves_se
    res["halves_bets"] = halves_n
    res["halves_won"] = halves_won
    half_ok = all(w is None or w >= MIN_HALF_EVENTS for w in halves_won)
    res["positive_both_halves"] = bool(half_ok and all(h is not None and s is not None and s > 0 and h > 2 * s
                                                       for h, s in zip(halves, halves_se)))
    se_cons = max(se, se_null) if not math.isnan(se_null) else se
    z = m / se_cons if events_ok and se_cons and se_cons > 0 else float("nan")
    res["z"] = r4(m / se) if se and se > 0 else None
    res["z_cons"] = r4(z)
    res["p"] = one_sided_p(z)
    res["testable"] = bool(events_ok)
    res["thin"] = n < MIN_BETS
    return res


def compact(c: dict) -> list:
    """[bets, roi_c0, roi_c5, roi_c8, se, half1_roi_c5, half2_roi_c5]"""
    return [c.get("bets"), c.get("roi_c0"), c.get("roi_c5"), c.get("roi_c8"), c.get("se"),
            *(c.get("halves_roi_c5") or [None, None])]


class Cells:
    """Every cell tested, for BH and the CSV."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, family: str, code: str, section: str, name: str, src: str, c: dict, **extra) -> dict:
        row = {"family": family, "code": code, "section": section, "name": name, "src": src, **c, **extra}
        self.rows.append(row)
        return c

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        for col in ("halves_roi_c5", "halves_se_c5", "halves_bets", "halves_won"):
            if col in df:
                df[col.replace("halves", "h1")] = df[col].map(lambda v: v[0] if isinstance(v, list) else None)
                df[col.replace("halves", "h2")] = df[col].map(lambda v: v[1] if isinstance(v, list) else None)
                df = df.drop(columns=[col])
        return df


# ----------------------------------------------------------------------------- data


def load(path: Path, end: str | None) -> tuple[pd.DataFrame, dict, pd.Timestamp]:
    cols = ["market_id", "market_name", "date", "field_size", "selection_id", "ltp_pre", "ltp_t30", "bsp",
            "won", "market_type"]
    df = pd.read_parquet(path, columns=cols)
    n_raw = len(df)
    df = df.drop_duplicates(["market_id", "selection_id"])
    n_dedup = len(df)
    df = df[(df["market_type"] == "WIN") & df["won"].notna()]
    if end:
        df = df[df["date"] < pd.Timestamp(end)]
    m_settled = df["market_id"].nunique()
    one_winner = df.groupby("market_id")["won"].transform("sum") == 1  # drops dead heats
    ok = df["ltp_pre"].notna() & (df["ltp_pre"] > 1)
    all_priced = ok.groupby(df["market_id"]).transform("min").astype(bool)
    df = df[one_winner & all_priced].drop(columns=["market_type"]).copy()
    df = df.sort_values(["date", "market_id", "ltp_pre", "selection_id"]).reset_index(drop=True)
    df["fav"] = df.groupby("market_id").cumcount() == 0  # lowest ltp_pre, ties by selection_id
    df["won_b"] = df["won"].to_numpy() == 1
    df["harness"] = df["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b")
    df["band"] = pd.cut(df["ltp_pre"], PRICE_BANDS, right=False, labels=BAND_LABELS).astype(str)
    df["fgroup"] = pd.cut(df["field_size"], FIELD_EDGES, right=False, labels=FIELD_LABELS).astype(str)
    df["month"] = df["date"].dt.to_period("M")
    df["bsp_ok"] = df["bsp"].notna() & (df["bsp"] > 1)
    df["bsp_band"] = pd.cut(df["bsp"].fillna(1.0), PRICE_BANDS, right=False, labels=BAND_LABELS).astype(str)
    df["has_t30"] = df["ltp_t30"].notna() & (df["ltp_t30"] > 1)
    df["move"] = np.where(df["has_t30"], np.log(df["ltp_t30"].fillna(1.0) / df["ltp_pre"]), np.nan)
    days = sorted(df["date"].unique())
    mid = pd.Timestamp(days[len(days) // 2])
    df["first"] = df["date"] < mid
    book = (1.0 / df["ltp_pre"]).groupby(df["market_id"]).sum()
    book_bsp = (1.0 / df.loc[df["bsp_ok"], "bsp"]).groupby(df.loc[df["bsp_ok"], "market_id"]).sum()
    all_bsp = df.groupby("market_id")["bsp_ok"].min()
    info = {"rows_raw": int(n_raw), "rows_after_dedupe": int(n_dedup), "markets_settled": int(m_settled),
            "markets_used": int(df["market_id"].nunique()), "runners_used": int(len(df)),
            "markets_gallops": int(df.loc[~df["harness"], "market_id"].nunique()),
            "markets_harness": int(df.loc[df["harness"], "market_id"].nunique()),
            "dropped": "dead heats and markets with any runner never traded before the off",
            "from": str(pd.Timestamp(days[0]).date()), "to": str(pd.Timestamp(days[-1]).date()),
            "mid_date_for_halves": str(mid.date()),
            "bsp_runner_coverage": r4(df["bsp_ok"].mean()),
            "t30_runner_coverage": r4(df["has_t30"].mean()),
            "book_ltp_pre_median": r4(book.median()),
            "book_bsp_median_all_bsp_markets": r4(book_bsp[all_bsp[all_bsp].index].median())}
    return df, info, mid


# ----------------------------------------------------------------------------- section A


class Rets:
    """Per-row return arrays for one code so a cell is a few numpy means."""

    def __init__(self, df: pd.DataFrame):
        won = df["won_b"].to_numpy()
        lp = df["ltp_pre"].to_numpy(dtype=float)
        bsp = df["bsp"].to_numpy(dtype=float)
        self.won = won
        self.first = df["first"].to_numpy()
        self.lp, self.bsp = lp, bsp
        self.bsp_ok = df["bsp_ok"].to_numpy()
        self.price = {"ltp_pre": lp, "bsp": bsp}
        self.valid = {"ltp_pre": np.ones(len(df), dtype=bool), "bsp": self.bsp_ok}
        self.back = {s: {c: back_returns(p, won, c) for c in COMMISSIONS} for s, p in self.price.items()}
        self.back_tick = {c: back_returns(tick_down(lp), won, c) for c in COMMISSIONS}
        self.lay_liab = {s: {c: lay_liability_returns(p, won, c) for c in COMMISSIONS} for s, p in self.price.items()}
        self.lay_liab_tick = {c: lay_liability_returns(tick_up(lp), won, c) for c in COMMISSIONS}
        self.lay_stake = {s: {c: lay_stake_returns(p, won, c) for c in COMMISSIONS} for s, p in self.price.items()}
        self.lay_stake_tick = {c: lay_stake_returns(tick_up(lp), won, c) for c in COMMISSIONS}
        self.diff = {c: self.back["bsp"][c] - self.back["ltp_pre"][c] for c in COMMISSIONS}
        self.q = 1.0 / lp  # implied probability at the last price, for the null SE
        k = 1.0 - C_TEST
        lpu = tick_up(lp)
        self.o_back = {s: ((p - 1.0) * k, np.full(len(lp), -1.0)) for s, p in self.price.items()}
        self.o_back_tick = ((tick_down(lp) - 1.0) * k, np.full(len(lp), -1.0))
        self.o_lay_liab = {s: (np.full(len(lp), -1.0), k / (p - 1.0)) for s, p in self.price.items()}
        self.o_lay_liab_tick = (np.full(len(lp), -1.0), k / (lpu - 1.0))
        self.o_lay_stake = {s: (-(p - 1.0), np.full(len(lp), k)) for s, p in self.price.items()}
        self.o_diff = ((bsp - lp) * k, np.zeros(len(lp)))
        self.ratio = bsp / lp
        self.sp_better_tick = bsp >= tick_up(lp) - 1e-9
        self.sp_worse_tick = bsp <= tick_down(lp) + 1e-9

    def sub(self, d: dict, idx: np.ndarray) -> dict:
        return {c: a[idx] for c, a in d.items()}

    def osub(self, o: tuple, idx: np.ndarray) -> tuple:
        return o[0][idx], o[1][idx]

    def cell(self, d: dict, o: tuple, idx: np.ndarray, digits: int = 4) -> dict:
        return cell(self.sub(d, idx), self.first[idx], digits, won=self.won[idx], q=self.q[idx], outcomes=self.osub(o, idx))


def ratio_cell(R: Rets, idx: np.ndarray) -> dict:
    """SP vs last price on the runners with both prices."""
    idx = idx[R.bsp_ok[idx]]
    n = len(idx)
    if n == 0:
        return {"runners": 0}
    ratio = R.ratio[idx]
    q = np.quantile(ratio, [0.1, 0.25, 0.5, 0.75, 0.9])
    res = {"runners": int(n), "median_ltp_pre": r4(np.median(R.lp[idx])),
           "ratio_q10_25_50_75_90": [r4(v) for v in q], "mean_log_ratio": r4(np.log(ratio).mean()),
           "share_sp_better_1tick": r4(R.sp_better_tick[idx].mean()),
           "share_sp_worse_1tick": r4(R.sp_worse_tick[idx].mean()),
           "implied_gain_if_ltp_fair": r4((ratio - 1.0).mean()),
           "strike": r4(R.won[idx].mean()), "implied_ltp": r4((1.0 / R.lp[idx]).mean()),
           "implied_bsp": r4((1.0 / R.bsp[idx]).mean())}
    res["back_ltp_pre"] = R.cell(R.back["ltp_pre"], R.o_back["ltp_pre"], idx)
    res["back_ltp_pre_tick_down"] = R.cell(R.back_tick, R.o_back_tick, idx)
    res["back_bsp"] = R.cell(R.back["bsp"], R.o_back["bsp"], idx)
    res["sp_minus_ltp"] = R.cell(R.diff, R.o_diff, idx)
    return res


def section_a(df: pd.DataFrame, R: Rets, code: str, cells: Cells) -> None:
    band = df["band"].to_numpy()
    fg = df["fgroup"].to_numpy()
    fav = df["fav"].to_numpy()
    rows_bf = {}
    for b in BAND_LABELS:
        for f in FIELD_LABELS + ["all"]:
            idx = np.flatnonzero((band == b) & ((fg == f) if f != "all" else True))
            rc = ratio_cell(R, idx)
            rows_bf[f"{b}|{f}"] = rc
            if rc.get("runners", 0):
                cells.add("back_sp", code, "A", f"band {b} field {f}", "bsp", rc["back_bsp"],
                          median_price=rc["median_ltp_pre"])
                cells.add("sp_minus_ltp", code, "A", f"band {b} field {f}", "paired", rc["sp_minus_ltp"],
                          median_price=rc["median_ltp_pre"])
    for f in FIELD_LABELS:
        idx = np.flatnonzero(fg == f)
        rc = ratio_cell(R, idx)
        rows_bf[f"all|{f}"] = rc
        cells.add("back_sp", code, "A", f"all bands field {f}", "bsp", rc["back_bsp"])
        cells.add("sp_minus_ltp", code, "A", f"all bands field {f}", "paired", rc["sp_minus_ltp"])
    rc = ratio_cell(R, np.arange(len(df)))
    rows_bf["all|all"] = rc
    cells.add("back_sp", code, "A", "all runners", "bsp", rc["back_bsp"])
    cells.add("sp_minus_ltp", code, "A", "all runners", "paired", rc["sp_minus_ltp"])

    out({"section": "A", "code": code, "what": "bsp/ltp_pre by ltp_pre band (all field sizes)",
         "columns": {"ratio_q": "q10,q25,q50,q75,q90 of bsp/ltp_pre",
                     "back": "[bets, roi_c0, roi_c5, roi_c8, se_c5, half1_c5, half2_c5]",
                     "sp_minus_ltp": "paired bsp return minus ltp_pre return, same layout"},
         "rows": {b: {"runners": rows_bf[f"{b}|all"]["runners"], "ratio_q": rows_bf[f"{b}|all"]["ratio_q10_25_50_75_90"],
                      "mean_log_ratio": rows_bf[f"{b}|all"]["mean_log_ratio"],
                      "share_sp_better_1tick": rows_bf[f"{b}|all"]["share_sp_better_1tick"],
                      "share_sp_worse_1tick": rows_bf[f"{b}|all"]["share_sp_worse_1tick"],
                      "strike": rows_bf[f"{b}|all"]["strike"], "implied_ltp": rows_bf[f"{b}|all"]["implied_ltp"],
                      "implied_bsp": rows_bf[f"{b}|all"]["implied_bsp"],
                      "back_ltp_pre": compact(rows_bf[f"{b}|all"]["back_ltp_pre"]),
                      "back_ltp_tick_down": compact(rows_bf[f"{b}|all"]["back_ltp_pre_tick_down"]),
                      "back_bsp": compact(rows_bf[f"{b}|all"]["back_bsp"]),
                      "sp_minus_ltp": compact(rows_bf[f"{b}|all"]["sp_minus_ltp"]),
                      "sp_minus_ltp_z": rows_bf[f"{b}|all"]["sp_minus_ltp"].get("z")}
                  for b in BAND_LABELS}})
    out({"section": "A", "code": code, "what": "bsp/ltp_pre by field size (all bands) and overall",
         "rows": {f: {"runners": rows_bf[f"all|{f}"]["runners"], "ratio_q": rows_bf[f"all|{f}"]["ratio_q10_25_50_75_90"],
                      "share_sp_better_1tick": rows_bf[f"all|{f}"]["share_sp_better_1tick"],
                      "share_sp_worse_1tick": rows_bf[f"all|{f}"]["share_sp_worse_1tick"],
                      "back_ltp_pre": compact(rows_bf[f"all|{f}"]["back_ltp_pre"]),
                      "back_bsp": compact(rows_bf[f"all|{f}"]["back_bsp"]),
                      "sp_minus_ltp": compact(rows_bf[f"all|{f}"]["sp_minus_ltp"]),
                      "sp_minus_ltp_z": rows_bf[f"all|{f}"]["sp_minus_ltp"].get("z")}
                  for f in FIELD_LABELS + ["all"]}})
    out({"section": "A", "code": code, "what": "band x field size: median ratio / share SP better by a tick / paired sp_minus_ltp c5 [mean, se, z]",
         "rows": {b: {f: {"n": rows_bf[f"{b}|{f}"].get("runners", 0),
                          "median_ratio": (rows_bf[f"{b}|{f}"].get("ratio_q10_25_50_75_90") or [None] * 5)[2],
                          "sp_better": rows_bf[f"{b}|{f}"].get("share_sp_better_1tick"),
                          "diff": [rows_bf[f"{b}|{f}"].get("sp_minus_ltp", {}).get(k) for k in ("roi_c5", "se", "z")],
                          "back_bsp_c5": rows_bf[f"{b}|{f}"].get("back_bsp", {}).get("roi_c5"),
                          "back_ltp_c5": rows_bf[f"{b}|{f}"].get("back_ltp_pre", {}).get("roi_c5")}
                      for f in FIELD_LABELS} for b in BAND_LABELS}})

    # favourites (rank 1 at ltp_pre) by field size
    fav_rows = {}
    for f in FIELD_LABELS + ["all"]:
        idx = np.flatnonzero(fav & ((fg == f) if f != "all" else True))
        rc = ratio_cell(R, idx)
        fav_rows[f] = rc
        cells.add("back_sp", code, "A", f"favourite field {f}", "bsp", rc["back_bsp"])
        cells.add("sp_minus_ltp", code, "A", f"favourite field {f}", "paired", rc["sp_minus_ltp"])
    out({"section": "A", "code": code, "what": "favourites (rank 1 at ltp_pre) by field size",
         "rows": {f: {"runners": v["runners"], "median_ltp_pre": v["median_ltp_pre"], "ratio_q": v["ratio_q10_25_50_75_90"],
                      "share_sp_better_1tick": v["share_sp_better_1tick"], "strike": v["strike"],
                      "implied_ltp": v["implied_ltp"], "implied_bsp": v["implied_bsp"],
                      "back_ltp_pre": compact(v["back_ltp_pre"]), "back_ltp_tick_down": compact(v["back_ltp_pre_tick_down"]),
                      "back_bsp": compact(v["back_bsp"]), "sp_minus_ltp": compact(v["sp_minus_ltp"]),
                      "sp_minus_ltp_z": v["sp_minus_ltp"].get("z")} for f, v in fav_rows.items()}})

    # move from t30 to pre (conditional on t30 trading)
    has = df["has_t30"].to_numpy()
    mv = pd.cut(df["move"], MOVE_CUTS, labels=MOVE_LABELS).astype(str).to_numpy()
    move_rows = {}
    for lab in MOVE_LABELS:
        idx = np.flatnonzero(has & (mv == lab))
        rc = ratio_cell(R, idx)
        move_rows[lab] = rc
        cells.add("back_sp", code, "A", f"t30 move {lab}", "bsp", rc["back_bsp"], conditional_on_t30=True)
        cells.add("sp_minus_ltp", code, "A", f"t30 move {lab}", "paired", rc["sp_minus_ltp"], conditional_on_t30=True)
    out({"section": "A", "code": code, "what": "by t30 -> pre move (CONDITIONAL on the runner trading 30 min out)",
         "t30_runner_share": r4(has.mean()),
         "rows": {lab: {"runners": v["runners"], "median_ltp_pre": v["median_ltp_pre"], "ratio_q": v["ratio_q10_25_50_75_90"],
                        "share_sp_better_1tick": v["share_sp_better_1tick"], "strike": v["strike"],
                        "implied_ltp": v["implied_ltp"], "implied_bsp": v["implied_bsp"],
                        "back_ltp_pre": compact(v["back_ltp_pre"]), "back_bsp": compact(v["back_bsp"]),
                        "sp_minus_ltp": compact(v["sp_minus_ltp"]), "sp_minus_ltp_z": v["sp_minus_ltp"].get("z")}
                  for lab, v in move_rows.items()}})


# ----------------------------------------------------------------------------- section A2: walk-forward chooser


def chooser(df: pd.DataFrame, R: Rets, code: str, cells: Cells, universe: str) -> None:
    mask = R.bsp_ok.copy()
    if universe == "favourites":
        mask &= df["fav"].to_numpy()
    idx_all = np.flatnonzero(mask)
    key = (df["band"] + "|" + df["fgroup"]).to_numpy()[idx_all]
    month = df["month"].to_numpy()[idx_all]
    d5 = R.diff[0.05][idx_all]
    months = np.array(sorted(pd.unique(month)))
    keys = np.array(sorted(pd.unique(key)))
    mi = pd.Index(months).get_indexer(month)
    ki = pd.Index(keys).get_indexer(key)
    shape = (len(months), len(keys))
    S = np.zeros(shape)
    SS = np.zeros(shape)
    N = np.zeros(shape)
    np.add.at(S, (mi, ki), d5)
    np.add.at(SS, (mi, ki), d5 * d5)
    np.add.at(N, (mi, ki), 1.0)
    # strictly earlier months: cumulative sums shifted by one month
    cS = np.vstack([np.zeros((1, len(keys))), np.cumsum(S, axis=0)[:-1]])
    cSS = np.vstack([np.zeros((1, len(keys))), np.cumsum(SS, axis=0)[:-1]])
    cN = np.vstack([np.zeros((1, len(keys))), np.cumsum(N, axis=0)[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = cS / cN
        var = (cSS - cS * cS / cN) / np.maximum(cN - 1.0, 1.0)
        se = np.sqrt(var / cN)
    evaluable = mi >= BURN_IN_MONTHS
    res_all = {}
    for k in K_SE:
        with np.errstate(invalid="ignore"):
            take_sp = (cN >= MIN_FIT_BETS) & (mean > k * se)
        choice = take_sp[mi, ki] & evaluable
        ev = np.flatnonzero(evaluable)
        rows = idx_all[ev]
        ch = choice[ev]
        first = R.first[rows]
        ret_ch = {c: np.where(ch, R.back["bsp"][c][rows], R.back["ltp_pre"][c][rows]) for c in COMMISSIONS}
        gain_ltp = {c: ret_ch[c] - R.back["ltp_pre"][c][rows] for c in COMMISSIONS}
        gain_sp = {c: ret_ch[c] - R.back["bsp"][c][rows] for c in COMMISSIONS}
        won_r, q_r = R.won[rows], R.q[rows]
        chosen = np.where(ch, R.bsp[rows], R.lp[rows])
        kc = 1.0 - C_TEST
        zeros = np.zeros(len(rows))
        c_ch = cell(ret_ch, first, won=won_r, q=q_r, outcomes=((chosen - 1.0) * kc, np.full(len(rows), -1.0)))
        c_gl = cell(gain_ltp, first, won=won_r, q=q_r, outcomes=((chosen - R.lp[rows]) * kc, zeros))
        c_gs = cell(gain_sp, first, won=won_r, q=q_r, outcomes=((chosen - R.bsp[rows]) * kc, zeros))
        cells.add("chooser", code, "A2", f"chooser k={k:g} {universe} vs always-ltp", "walk-forward", c_gl)
        cells.add("chooser", code, "A2", f"chooser k={k:g} {universe} vs always-sp", "walk-forward", c_gs)
        share_by_band = {}
        bands_rows = df["band"].to_numpy()[rows]
        for b in BAND_LABELS:
            m = bands_rows == b
            share_by_band[b] = r4(ch[m].mean()) if m.any() else None
        res_all[f"k={k:g}"] = {"bets": int(len(rows)), "share_sp_chosen": r4(ch.mean()),
                               "share_sp_chosen_by_band": share_by_band,
                               "roi_chooser": compact(c_ch),
                               "roi_always_ltp": compact(cell({c: R.back["ltp_pre"][c][rows] for c in COMMISSIONS}, first)),
                               "roi_always_sp": compact(cell({c: R.back["bsp"][c][rows] for c in COMMISSIONS}, first)),
                               "gain_vs_always_ltp": compact(c_gl), "gain_vs_always_ltp_z": c_gl.get("z"),
                               "gain_vs_always_sp": compact(c_gs), "gain_vs_always_sp_z": c_gs.get("z")}
    out({"section": "A2", "code": code, "universe": universe,
         "what": "walk-forward take-SP-or-last-price chooser (band x field cells; fitted on strictly earlier months; "
                 f"{BURN_IN_MONTHS}-month burn-in, >= {MIN_FIT_BETS} earlier bets)",
         "months_evaluated": int(max(len(months) - BURN_IN_MONTHS, 0)), "results": res_all})


# ----------------------------------------------------------------------------- section B: lay by band


def lay_cell(R: Rets, idx: np.ndarray, src: str) -> dict:
    idx = idx[R.valid[src][idx]]
    n = len(idx)
    if n == 0:
        return {"bets": 0}
    liab = R.cell(R.lay_liab[src], R.o_lay_liab[src], idx, digits=5)
    stake = R.cell(R.lay_stake[src], R.o_lay_stake[src], idx)
    res = {**liab, "per_stake": compact(stake), "per_stake_z": stake.get("z_cons"),
           "back_roi_c0": r4(R.back[src][0.0][idx].mean()), "back_roi_c5": r4(R.back[src][0.05][idx].mean()),
           "strike": r4(R.won[idx].mean()), "median_price": r4(np.median(R.price[src][idx]))}
    if src == "ltp_pre":
        t = R.cell(R.lay_liab_tick, R.o_lay_liab_tick, idx, digits=5)
        res["tick_up_liab"] = compact(t)
        res["tick_up_liab_z"] = t.get("z_cons")
    return res


def lay_compact(c: dict) -> dict:
    return {"bets": c.get("bets"), "n_won": c.get("n_won"), "expected_won_at_ltp": c.get("expected_won_at_ltp"),
            "strike": c.get("strike"), "median_price": c.get("median_price"),
            "per_liab": compact(c), "se_null": c.get("se_null"), "z": c.get("z"), "z_cons": c.get("z_cons"),
            "testable": c.get("testable"), "both_halves": c.get("positive_both_halves"),
            "per_stake": c.get("per_stake"), "per_stake_z": c.get("per_stake_z"),
            "tick_up_liab": c.get("tick_up_liab"), "back_roi_c0": c.get("back_roi_c0")}


def section_b(df: pd.DataFrame, R: Rets, code: str, cells: Cells) -> None:
    band = df["band"].to_numpy()
    bsp_band = df["bsp_band"].to_numpy()
    fg = df["fgroup"].to_numpy()
    for src in ("ltp_pre", "bsp"):
        rows = {}
        for b in BAND_LABELS:
            c = lay_cell(R, np.flatnonzero(band == b), src)
            cells.add("lay", code, "B", f"lay band {b} (ltp_pre band)", src, c, median_price=c.get("median_price"))
            rows[b] = lay_compact(c)
        out({"section": "B", "code": code, "src": src, "what": f"lay every runner by ltp_pre band, settled at {src}",
             "columns": {"per_liab": "[bets, roi_c0, roi_c5, roi_c8, se_c5, half1_c5, half2_c5] per unit liability",
                         "per_stake": "same per unit backer stake", "tick_up_liab": "lay one tick higher (ltp_pre only)"},
             "rows": rows})
        grid = {}
        for b in BAND_LABELS:
            grid[b] = {}
            for f in FIELD_LABELS:
                c = lay_cell(R, np.flatnonzero((band == b) & (fg == f)), src)
                cells.add("lay", code, "B", f"lay band {b} field {f}", src, c, median_price=c.get("median_price"))
                grid[b][f] = [c.get("bets"), c.get("n_won"), c.get("roi_c5"), c.get("se"), c.get("z_cons"), c.get("positive_both_halves")]
        out({"section": "B", "code": code, "src": src, "what": "lay by ltp_pre band x field size, per unit liability: [bets, n_won, roi_c5, se, z_cons, both_halves]",
             "rows": grid})
    # descriptive: selected by the realised BSP band (not bettable: the SP is not known before the off)
    rows = {}
    for b in BAND_LABELS:
        c = lay_cell(R, np.flatnonzero(bsp_band == b), "bsp")
        rows[b] = lay_compact(c)
    out({"section": "B", "code": code, "src": "bsp", "what": "DESCRIPTIVE (not bettable): lay by realised BSP band, settled at bsp", "rows": rows})


# ----------------------------------------------------------------------------- section C: lay rules


def section_c(df: pd.DataFrame, R: Rets, code: str, cells: Cells) -> None:
    lp = R.lp
    fs = df["field_size"].to_numpy()
    fav = df["fav"].to_numpy()
    # (1) lay the field at long prices
    for thr in LONGSHOT_THRESHOLDS:
        rows = {}
        for f in ["all"] + FIELD_LABELS:
            fm = np.ones(len(df), dtype=bool) if f == "all" else (df["fgroup"].to_numpy() == f)
            idx = np.flatnonzero((lp >= thr) & fm)
            rows[f] = {}
            for src in ("ltp_pre", "bsp"):
                c = lay_cell(R, idx, src)
                cells.add("lay", code, "C", f"lay all >= {thr:g} field {f}", src, c, median_price=c.get("median_price"))
                rows[f][src] = lay_compact(c)
            mk = df.loc[idx, "market_id"]
            rows[f]["markets"] = int(mk.nunique())
            rows[f]["runners_per_market"] = r4(len(idx) / max(mk.nunique(), 1))
        out({"section": "C", "code": code, "rule": f"lay every runner with ltp_pre >= {thr:g}", "rows": rows})
    # (2) odds-on favourites in small fields
    for pname, (plo, phi) in ODDS_ON.items():
        rows = {}
        for fname, (flo, fhi) in SMALL_FIELDS.items():
            idx = np.flatnonzero(fav & (lp >= plo) & (lp < phi) & (fs >= flo) & (fs <= fhi))
            rows[fname] = {}
            for src in ("ltp_pre", "bsp"):
                c = lay_cell(R, idx, src)
                cells.add("lay", code, "C", f"lay odds-on fav {pname} field {fname}", src, c, median_price=c.get("median_price"))
                rows[fname][src] = lay_compact(c)
        out({"section": "C", "code": code, "rule": f"lay the favourite when ltp_pre in {pname}", "rows": rows})


# ----------------------------------------------------------------------------- section D: multiple testing


def half_z(row: pd.Series, h: int) -> float:
    r, s, w = row.get(f"h{h}_roi_c5"), row.get(f"h{h}_se_c5"), row.get(f"h{h}_won")
    if r is None or s is None or not s or (isinstance(s, float) and math.isnan(s)):
        return float("nan")
    if w is not None and not (isinstance(w, float) and math.isnan(w)) and w < MIN_HALF_EVENTS:
        return float("nan")
    return r / s


def section_d(cells: Cells) -> dict:
    df = cells.frame()
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    summary = {"cells_total": int(len(df))}
    for fam, g in df.groupby("family"):
        g = g.copy()
        g["bh"] = benjamini_hochberg(g["p"].to_numpy(), FDR)
        df.loc[g.index, "bh"] = g["bh"]
        surv = g[g["bh"] & g["positive_both_halves"].fillna(False).astype(bool)]
        bh_only = g[g["bh"]]
        fam_res = {"cells": int(len(g)), "cells_testable_with_p": int(g["p"].notna().sum()),
                   "cells_not_testable_too_few_events": int((g["bets"] > 0).sum() - g["p"].notna().sum()),
                   "bh_rejections": int(g["bh"].sum()),
                   "both_halves_positive": int(g["positive_both_halves"].fillna(False).astype(bool).sum()),
                   "survivors_bh_and_both_halves": int(len(surv)),
                   "columns": {"bh_rejections_listed": "[code, section, name, src, bets, n_won, roi_c5, se, se_null, h1_roi_c5, h2_roi_c5, both_halves]",
                               "survivors": "[code, section, name, src, bets, n_won, roi_c0, roi_c5, roi_c8, se, se_null, h1_roi_c5, h1_se, h2_roi_c5, h2_se, thin]"},
                   "bh_rejections_listed": [[r["code"], r["section"], r["name"], r["src"], int(r["bets"]), r.get("n_won"), r.get("roi_c5"), r.get("se"),
                                             r.get("se_null"), r.get("h1_roi_c5"), r.get("h2_roi_c5"), bool(r.get("positive_both_halves"))]
                                            for _, r in bh_only.sort_values("p").head(40).iterrows()],
                   "survivors": [[r["code"], r["section"], r["name"], r["src"], int(r["bets"]), r.get("n_won"), r.get("roi_c0"), r.get("roi_c5"), r.get("roi_c8"),
                                  r.get("se"), r.get("se_null"), r.get("h1_roi_c5"), r.get("h1_se_c5"), r.get("h2_roi_c5"), r.get("h2_se_c5"), bool(r.get("thin"))]
                                 for _, r in surv.sort_values("p").iterrows()]}
        # opposite-half development: pick cells at > 2 SE in one half (>= 50 bets and >= MIN_HALF_EVENTS winners
        # there), evaluate in the other half (same gate; a half with too few events gives z = NaN = not held)
        dev = {}
        for h, o in ((1, 2), (2, 1)):
            picked = g[(g[f"h{h}_bets"].fillna(0) >= 50) & (g.apply(lambda r: half_z(r, h), axis=1) > 2)]
            held = picked[picked.apply(lambda r: half_z(r, o), axis=1) > 2]
            dev[f"dev_half{h}_eval_half{o}"] = {
                "picked": int(len(picked)), "held_over_2se": int(len(held)),
                "picked_listed": [[r["code"], r["section"], r["name"], r["src"], int(r[f"h{h}_bets"]), r[f"h{h}_roi_c5"],
                                   int(r[f"h{o}_bets"]), r[f"h{o}_roi_c5"], r[f"h{o}_se_c5"]]
                                  for _, r in picked.head(25).iterrows()],
                "mean_eval_roi_c5_of_picked": r4(picked[f"h{o}_roi_c5"].astype(float).mean()) if len(picked) else None}
        fam_res["opposite_half"] = dev
        summary[fam] = fam_res
    return summary, df


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=HERE.parent / "data/raw/betfair/split/au_win.parquet")
    ap.add_argument("--end", default=None, help="use only dates before this (development subset)")
    ap.add_argument("--code", choices=["gallops", "harness", "both"], default="both")
    ap.add_argument("--cells-csv", type=Path, default=None)
    a = ap.parse_args()
    t0 = time.time()
    df, info, mid = load(a.path, a.end)
    out({"section": "data", **info, "end": a.end, "commissions": COMMISSIONS,
         "lay_definition": "per unit liability: stake 1/(price-1), win stake*(1-c) when the runner loses, pay 1 when it wins; "
                           "per unit backer stake: win (1-c) / pay (price-1)"})
    cells = Cells()
    codes = ["gallops", "harness"] if a.code == "both" else [a.code]
    for code in codes:
        sub = df[df["harness"] == (code == "harness")].reset_index(drop=True)
        if sub.empty:
            continue
        R = Rets(sub)
        out({"section": "code", "code": code, "markets": int(sub["market_id"].nunique()), "runners": int(len(sub)),
             "bsp_runner_coverage": r4(R.bsp_ok.mean()), "halves_runners": [int(R.first.sum()), int((~R.first).sum())]})
        section_a(sub, R, code, cells)
        for universe in ("all runners", "favourites"):
            chooser(sub, R, code, cells, universe)
        section_b(sub, R, code, cells)
        section_c(sub, R, code, cells)
        out({"section": "timing", "code": code, "seconds": round(time.time() - t0, 1)})
    summary, frame = section_d(cells)
    out({"section": "D", "fdr": FDR, "min_bets_for_not_thin": MIN_BETS, "min_events": MIN_EVENTS, "min_half_events": MIN_HALF_EVENTS,
         "p_value": "one-sided from roi_c5 / max(sample SE, binomial null SE at q = 1/ltp_pre)", **summary})
    if a.cells_csv:
        frame.to_csv(a.cells_csv, index=False)
        out({"section": "cells_csv", "path": str(a.cells_csv), "rows": int(len(frame))})
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Experiment "cross_country": thin markets elsewhere.

    python scripts/bf_cross_country.py [--countries GB,IE,NZ,ZA,FR,US] [--end YYYY-MM-DD] [--burn-in 3]

The same calibration table and favourite-rank backtest as scripts/betfair_backtest.py
(AU) for the GB, IE, NZ, ZA, FR and US WIN markets in data/raw/betfair/split/,
one country at a time:

  * liquidity proxies on every settled WIN market: share of runners with any
    pre-off trade (ltp_pre), with a trade 30 min out (ltp_t30), with a BSP;
    share of markets where every runner traded; runners at the 1000 ceiling;
  * book overround = sum(1/ltp_pre) over the market, on markets where every
    runner traded (median, quartiles, by field size and by how many runners had
    traded 30 min out); also the book at BSP and at ltp_t30 where complete;
  * favourite ranks 1-6 (rank by ltp_pre) and nine ltp_pre bands, each backed
    with 1 unit at ltp_pre and at BSP: bets, ROI at commission 0/5/8% (on net
    winnings only [A-54]), per-bet SE at 5%, ROI/SE in each half of the period
    (split at the country's median market date), strike vs proportional/Shin
    implied probability, ROI one tick below ltp_pre (ltp_pre is the last trade,
    not an available price [A-53]), and the lay side (1 unit backer's stake);
  * favourite-longshot bias, formally: p_i proportional to p_mkt_i^beta within
    the race (beta > 1 = favourites under-bet). beta is fitted walk-forward by
    maximum likelihood on strictly earlier months (expanding window, refit
    monthly, the first --burn-in months are not scored). Reported: beta (full
    sample, with SE, and per half), the walk-forward log-loss gain of the
    recalibrated probabilities vs the proportional market (per-race paired SE),
    the same for Shin, and a value rule from the walk-forward probabilities
    (back when p_model * ltp_pre - 1 >= 5% / 10%; lay when <= -10%);
  * a descriptive FLB index: ROI before commission of runners under 4.5 minus
    runners at 15+, with SE.

Multiple testing: Benjamini-Hochberg at FDR 5% across ALL back cells of all
countries (one-sided p from z = roi_c5 / se); a survivor must also be > 2 SE
above zero after 5% commission in BOTH halves. Lay cells and the log-loss
gains are separate families. Then each half is used as the development set:
cells with z > 2 in one half are re-scored in the other.

Nothing in the rank/band tables is fitted; the only fitted quantity is beta
(walk-forward). Dead heats and markets with an untraded runner are dropped.
Prints summary numbers only; never prints horse names.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ponies import odds  # noqa: E402

SPLIT_DIR = HERE.parent / "data" / "raw" / "betfair" / "split"
COUNTRIES = ["GB", "IE", "NZ", "ZA", "FR", "US"]
COMMISSIONS = (0.0, 0.05, 0.08)
PRICE_BANDS = [1.0, 2.0, 3.0, 4.5, 6.5, 10.0, 15.0, 25.0, 50.0, 1e9]
BAND_LABELS = [f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+" for lo, hi in zip(PRICE_BANDS, PRICE_BANDS[1:])]
FIELD_EDGES = [2, 6, 9, 12, 99]
FIELD_LABELS = ["2-5", "6-8", "9-11", "12+"]
RANKS = 6
VALUE_MARGINS = (0.05, 0.10)
LAY_MARGIN = 0.10
FDR = 0.05
SHORT_MAX, LONG_MIN = 4.5, 15.0  # FLB index bands

# Betfair price ladder: (lower bound inclusive, upper bound exclusive, increment)
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1001.0, 10.0)]


# ----------------------------------------------------------------------------- utilities


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


def out(obj) -> None:
    print(json.dumps(obj, default=_json_default), flush=True)


def r4(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) else round(x, 4)


def net_returns(price: np.ndarray, won: np.ndarray, commission: float) -> np.ndarray:
    """Vectorised ponies.betfair.net_return [A-54]: commission on net winnings only."""
    price = np.asarray(price, dtype=float)
    return np.where(won, (price - 1.0) * (1.0 - commission), -1.0)


def lay_returns(price: np.ndarray, won: np.ndarray, commission: float) -> np.ndarray:
    """Lay 1 unit of backer's stake: keep the stake less commission when the
    runner loses, pay the liability (price - 1) when it wins."""
    price = np.asarray(price, dtype=float)
    return np.where(won, -(price - 1.0), 1.0 - commission)


def tick_down(price: np.ndarray) -> np.ndarray:
    """One Betfair tick below ``price``."""
    p = np.asarray(price, dtype=float)
    inc = np.full_like(p, 10.0)
    q = p - 1e-9
    for lo, hi, step in TICKS:
        inc = np.where((q >= lo) & (q < hi), step, inc)
    return np.maximum(1.01, np.round(p - inc, 2))


def one_sided_p(z: float) -> float:
    if z is None or math.isnan(z):
        return float("nan")
    return 0.5 * math.erfc(z / math.sqrt(2))


def benjamini_hochberg(pvals, q: float = FDR) -> np.ndarray:
    """Boolean mask of rejected hypotheses at FDR q (NaN = not tested)."""
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


def mean_se(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan")), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def shin_vectorised(price: np.ndarray, codes: np.ndarray, n_races: int, iters: int = 60) -> np.ndarray:
    """odds.shin_probabilities [A-51] for every race at once (bisection on z per race)."""
    pi = 1.0 / np.asarray(price, dtype=float)
    book = np.bincount(codes, weights=pi, minlength=n_races)
    br = book[codes]
    lo = np.zeros(n_races)
    hi = np.full(n_races, 0.5)
    for _ in range(iters):
        z = (lo + hi) / 2
        zr = z[codes]
        p = (np.sqrt(zr * zr + 4 * (1 - zr) * pi * pi / br) - zr) / (2 * (1 - zr))
        tot = np.bincount(codes, weights=p, minlength=n_races)
        gt = tot > 1
        lo = np.where(gt, z, lo)
        hi = np.where(gt, hi, z)
    zr = ((lo + hi) / 2)[codes]
    p = (np.sqrt(zr * zr + 4 * (1 - zr) * pi * pi / br) - zr) / (2 * (1 - zr))
    p = p / np.bincount(codes, weights=p, minlength=n_races)[codes]
    return np.where(br <= 1.0, pi / br, p)  # no margin to remove: proportional


def power_probs(x: np.ndarray, beta: float, codes: np.ndarray, n_races: int) -> np.ndarray:
    """p_i = p_mkt_i^beta / sum_race p_mkt^beta, with x = log p_mkt (<= 0, so no overflow)."""
    e = np.exp(beta * x)
    return e / np.bincount(codes, weights=e, minlength=n_races)[codes]


def fit_beta(x: np.ndarray, win: np.ndarray, codes: np.ndarray, n_races: int, iters: int = 30) -> tuple[float, float]:
    """Maximum-likelihood beta of the power recalibration by Newton's method.
    ``win`` is a boolean mask with exactly one winner per race. Returns (beta, se)."""
    beta = 1.0
    wc = codes[win]
    xw = x[win]
    var_sum = float("nan")
    for _ in range(iters):
        p = power_probs(x, beta, codes, n_races)
        ex = np.bincount(codes, weights=p * x, minlength=n_races)
        ex2 = np.bincount(codes, weights=p * x * x, minlength=n_races)
        var_sum = float(np.maximum(ex2 - ex * ex, 1e-12).sum())
        grad = float((xw - ex[wc]).sum())
        step = grad / var_sum
        beta += step
        if abs(step) < 1e-9:
            break
    return beta, 1.0 / math.sqrt(var_sum)


# ----------------------------------------------------------------------------- data


def load_country(code: str, end: str | None) -> tuple[pd.DataFrame, dict]:
    cols = ["market_id", "selection_id", "market_name", "start", "date", "field_size",
            "ltp_pre", "ltp_t30", "bsp", "won", "market_type"]
    df = pd.read_parquet(SPLIT_DIR / f"{code.lower()}_win.parquet", columns=cols)
    n_raw = len(df)
    df = df.drop_duplicates(["market_id", "selection_id"])
    df = df[(df["market_type"] == "WIN") & df["won"].notna()]
    if end:
        df = df[df["date"] < pd.Timestamp(end)]
    df = df.drop(columns=["market_type"])

    # liquidity proxies on every settled market (before any price filter)
    traded = df["ltp_pre"].notna() & (df["ltp_pre"] > 1)
    t30 = df["ltp_t30"].notna() & (df["ltp_t30"] > 1)
    has_bsp = df["bsp"].notna() & (df["bsp"] > 1)
    mk = df["market_id"]
    all_traded = traded.groupby(mk).transform("min").astype(bool)
    all_t30 = t30.groupby(mk).transform("min").astype(bool)
    wins = df.groupby("market_id")["won"].transform("sum")
    harness = df["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b")
    per_market = df.assign(all_traded=all_traded, all_t30=all_t30, harness=harness).drop_duplicates("market_id")
    liquidity = {
        "markets_settled": int(mk.nunique()), "runners": int(len(df)),
        "race_days": int(df["date"].nunique()),
        "markets_per_race_day": r4(mk.nunique() / max(df["date"].nunique(), 1)),
        "median_field": r4(per_market["field_size"].median()),
        "share_runners_any_pre_off_trade": r4(traded.mean()),
        "share_runners_t30_trade": r4(t30.mean()),
        "share_runners_bsp": r4(has_bsp.mean()),
        "share_runners_at_1000_ceiling": r4((df["ltp_pre"] >= 1000).mean()),
        "share_markets_all_runners_traded": r4(per_market["all_traded"].mean()),
        "share_markets_all_runners_t30": r4(per_market["all_t30"].mean()),
        "share_markets_harness": r4(per_market["harness"].mean()),
        "dead_heat_markets": int((per_market.set_index("market_id").index.map(wins.groupby(mk).first()) > 1).sum()),
    }

    # analysis set: one winner and every runner traded before the off
    df = df[(wins == 1) & all_traded].copy()
    df = df.sort_values(["start", "market_id", "ltp_pre", "selection_id"]).reset_index(drop=True)
    df["fav_rank"] = df.groupby("market_id", sort=False).cumcount() + 1
    df["won_b"] = df["won"].to_numpy() == 1
    codes, uniques = pd.factorize(df["market_id"], sort=False)
    n_races = len(uniques)
    df["race"] = codes
    lp = df["ltp_pre"].to_numpy(dtype=float)
    book_row = np.bincount(codes, weights=1.0 / lp, minlength=n_races)[codes]
    df["book"] = book_row
    df["p_prop"] = (1.0 / lp) / book_row
    df["p_shin"] = shin_vectorised(lp, codes, n_races)
    df["has_t30"] = df["ltp_t30"].notna() & (df["ltp_t30"] > 1)
    df["month"] = df["date"].dt.to_period("M")
    # Shin cross-check against the scalar helper on a few markets
    diffs = []
    for rid in range(min(50, n_races)):
        idx = np.flatnonzero(codes == rid)
        diffs.append(float(np.abs(odds.shin_probabilities(lp[idx]) - df["p_shin"].to_numpy()[idx]).max()))
    info = {"country": code, "rows_raw": int(n_raw), **liquidity,
            "markets_used": int(n_races), "runners_used": int(len(df)),
            "dropped": "dead heats and markets with any runner never traded before the off",
            "bsp_coverage_used": r4(((df["bsp"] > 1) & df["bsp"].notna()).mean()),
            "t30_runner_coverage_used": r4(df["has_t30"].mean()),
            "shin_check_max_abs_diff": r4(max(diffs)) if diffs else None}
    return df, info


def book_stats(df: pd.DataFrame) -> dict:
    mk = df.drop_duplicates("market_id")[["market_id", "book", "field_size", "date"]].copy()
    q = mk["book"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    res = {"markets": int(len(mk)),
           "book_ltp_pre": {"mean": r4(mk["book"].mean()), "q10": r4(q[0.1]), "q25": r4(q[0.25]),
                            "median": r4(q[0.5]), "q75": r4(q[0.75]), "q90": r4(q[0.9]),
                            "share_under_100pct": r4((mk["book"] < 1).mean())}}
    fs = pd.cut(mk["field_size"], FIELD_EDGES, right=False, labels=FIELD_LABELS)
    res["book_median_by_field"] = {str(k): r4(v) for k, v in mk.groupby(fs, observed=True)["book"].median().items()}
    # liquidity within the country: how many runners of the market had traded 30 min out
    t30_share = df.groupby("market_id")["has_t30"].mean()
    grp = pd.cut(t30_share, [-0.01, 0.0, 0.999, 1.0], labels=["none", "partial", "all"])
    g = mk.set_index("market_id").join(pd.DataFrame({"t30_grp": grp}))
    res["book_by_t30_coverage"] = {str(k): {"markets": int(len(v)), "book_median": r4(v["book"].median())}
                                   for k, v in g.groupby("t30_grp", observed=True)}
    # book at BSP (markets where every runner has a BSP) and at ltp_t30 (every runner traded 30 min out)
    for col, name in (("bsp", "book_bsp"), ("ltp_t30", "book_t30")):
        ok = df[col].notna() & (df[col] > 1)
        full = ok.groupby(df["market_id"]).transform("min").astype(bool)
        sub = df[full]
        if len(sub):
            b = (1.0 / sub[col]).groupby(sub["market_id"]).sum()
            res[name] = {"markets": int(len(b)), "median": r4(b.median()), "q10": r4(b.quantile(0.1)),
                         "q90": r4(b.quantile(0.9))}
        else:
            res[name] = {"markets": 0}
    # BSP relative to ltp_pre: does the SP drift out (ratio > 1) or in?
    both = df[df["bsp"].notna() & (df["bsp"] > 1)]
    res["bsp_over_ltp_pre_median"] = r4((both["bsp"] / both["ltp_pre"]).median()) if len(both) else None
    return res


# ----------------------------------------------------------------------------- walk-forward power recalibration


def walk_forward_power(df: pd.DataFrame, burn_in: int) -> tuple[np.ndarray, list[dict]]:
    """beta refitted every month on all strictly earlier months; returns the
    out-of-sample recalibrated probability per runner (NaN in the burn-in)."""
    x = np.log(df["p_prop"].to_numpy(dtype=float))
    win = df["won_b"].to_numpy()
    codes = df["race"].to_numpy()
    n_races = int(codes.max()) + 1
    months = df["month"].to_numpy()
    uniq = np.array(sorted(pd.unique(months)))
    p_model = np.full(len(df), np.nan)
    fits = []
    for i, m in enumerate(uniq):
        if i < burn_in:
            continue
        train = months < m
        beta, se = fit_beta(x[train], win[train], codes[train], n_races)
        test = months == m
        p_model[test] = power_probs(x[test], beta, codes[test], n_races)
        fits.append({"month": str(m), "beta": round(beta, 4), "se": round(se, 4), "train_races": int(len(np.unique(codes[train])))})
    return p_model, fits


def race_log_loss(p: np.ndarray, win: np.ndarray, codes: np.ndarray, n_races: int) -> np.ndarray:
    """-log p(winner) per race (NaN where the race has no probability)."""
    ll = np.full(n_races, np.nan)
    ok = win & ~np.isnan(p)
    ll[codes[ok]] = -np.log(np.clip(p[ok], 1e-12, 1.0))
    return ll


# ----------------------------------------------------------------------------- cells


class Returns:
    def __init__(self, df: pd.DataFrame, mid: pd.Timestamp):
        won = df["won_b"].to_numpy()
        lp = df["ltp_pre"].to_numpy(dtype=float)
        bsp = df["bsp"].to_numpy(dtype=float)
        self.won = won
        self.first = (df["date"] < mid).to_numpy()
        self.p_prop = df["p_prop"].to_numpy()
        self.p_shin = df["p_shin"].to_numpy()
        self.price = {"ltp_pre": lp, "bsp": bsp}
        self.valid = {"ltp_pre": np.ones(len(df), dtype=bool), "bsp": np.isfinite(bsp) & (bsp > 1)}
        self.back = {src: {c: net_returns(p, won, c) for c in COMMISSIONS} for src, p in self.price.items()}
        self.lay5 = {src: lay_returns(p, won, 0.05) for src, p in self.price.items()}
        self.tick5 = net_returns(tick_down(lp), won, 0.05)

    def cell(self, idx: np.ndarray, src: str) -> dict:
        idx = idx[self.valid[src][idx]]
        n = len(idx)
        res = {"bets": int(n)}
        if n == 0:
            return res
        won = self.won[idx]
        price = self.price[src][idx]
        for c in COMMISSIONS:
            res[f"roi_c{int(c * 100)}"] = r4(self.back[src][c][idx].mean())
        r5 = self.back[src][0.05][idx]
        res["se"] = r4(r5.std(ddof=1) / math.sqrt(n)) if n > 1 else None
        first = self.first[idx]
        halves, halves_se, halves_n = [], [], []
        for mask in (first, ~first):
            h = r5[mask]
            halves_n.append(int(mask.sum()))
            halves.append(r4(h.mean()) if len(h) else None)
            halves_se.append(r4(h.std(ddof=1) / math.sqrt(len(h))) if len(h) > 1 else None)
        res["halves_roi_c5"] = halves
        res["halves_se_c5"] = halves_se
        res["halves_bets"] = halves_n
        res["positive_both_halves"] = bool(all(h is not None and s is not None and s > 0 and h > 2 * s
                                               for h, s in zip(halves, halves_se)))
        res["strike"] = r4(won.mean())
        res["implied_raw"] = r4((1.0 / price).mean())
        res["implied_prop"] = r4(self.p_prop[idx].mean())
        res["implied_shin"] = r4(self.p_shin[idx].mean())
        res["median_price"] = r4(np.median(price))
        if src == "ltp_pre":
            res["roi_c5_tick_down"] = r4(self.tick5[idx].mean())
        l5 = self.lay5[src][idx]
        res["lay_roi_c5"] = r4(l5.mean())
        res["lay_se"] = r4(l5.std(ddof=1) / math.sqrt(n)) if n > 1 else None
        res["lay_halves_roi_c5"] = [r4(l5[m].mean()) if m.sum() else None for m in (first, ~first)]
        res["lay_halves_se_c5"] = [r4(l5[m].std(ddof=1) / math.sqrt(m.sum())) if m.sum() > 1 else None
                                   for m in (first, ~first)]
        return res


def z_of(cell: dict, key: str = "roi_c5", se_key: str = "se") -> float:
    if cell.get("bets", 0) < 2 or not cell.get(se_key):
        return float("nan")
    return cell[key] / cell[se_key]


def half_z(cell: dict, h: int, key: str = "halves_roi_c5", se_key: str = "halves_se_c5") -> float:
    if key not in cell:
        return float("nan")
    r, s = cell[key][h], cell[se_key][h]
    if r is None or not s:
        return float("nan")
    return r / s


def compact(c: dict) -> list:
    """[bets, roi_c0, roi_c5, roi_c8, se, half1_roi_c5, half2_roi_c5]"""
    return [c.get("bets"), c.get("roi_c0"), c.get("roi_c5"), c.get("roi_c8"), c.get("se"),
            *(c.get("halves_roi_c5") or [None, None])]


# ----------------------------------------------------------------------------- per country


def analyse_country(code: str, end: str | None, burn_in: int, cells: list, lays: list, ll_tests: list) -> dict:
    t0 = time.time()
    df, info = load_country(code, end)
    if df.empty:
        out({"country": code, "error": "no usable markets"})
        return {}
    out({"section": "data", **info})
    books = book_stats(df)
    out({"section": "book", "country": code, **books})

    mid = df.drop_duplicates("market_id")["date"].median()
    first_m = (df["date"] < mid).to_numpy()
    rets = Returns(df, mid)
    won = df["won_b"].to_numpy()
    lp = df["ltp_pre"].to_numpy(dtype=float)
    codes = df["race"].to_numpy()
    n_races = int(codes.max()) + 1

    # favourite ranks
    rank_rows = []
    fav = df["fav_rank"].to_numpy()
    for k in range(1, RANKS + 1):
        idx = np.flatnonzero(fav == k)
        a = rets.cell(idx, "ltp_pre")
        b = rets.cell(idx, "bsp")
        cells.append((code, "rank", f"fav{k}", "ltp_pre", a))
        cells.append((code, "rank", f"fav{k}", "bsp", b))
        lays.append((code, "rank", f"fav{k}", "ltp_pre", a))
        lays.append((code, "rank", f"fav{k}", "bsp", b))
        rank_rows.append({"rank": k, "n": a["bets"], "median_price": a.get("median_price"), "strike": a.get("strike"),
                          "implied_prop": a.get("implied_prop"), "implied_shin": a.get("implied_shin"),
                          "back_ltp_pre": compact(a), "tick_down_c5": a.get("roi_c5_tick_down"),
                          "back_bsp": compact(b), "lay_ltp_pre_c5": [a.get("lay_roi_c5"), a.get("lay_se")]})
    out({"section": "favourite_ranks", "country": code, "mid_date": mid,
         "columns": "back_* = [bets, roi_c0, roi_c5, roi_c8, se_c5, half1_roi_c5, half2_roi_c5]", "rows": rank_rows})

    # price bands
    band = pd.cut(df["ltp_pre"], PRICE_BANDS, right=False, labels=BAND_LABELS).astype(str).to_numpy()
    band_rows = []
    for lab in BAND_LABELS:
        idx = np.flatnonzero(band == lab)
        a = rets.cell(idx, "ltp_pre")
        b = rets.cell(idx, "bsp")
        cells.append((code, "band", lab, "ltp_pre", a))
        cells.append((code, "band", lab, "bsp", b))
        lays.append((code, "band", lab, "ltp_pre", a))
        lays.append((code, "band", lab, "bsp", b))
        band_rows.append({"band": lab, "n": a["bets"], "strike": a.get("strike"), "implied_raw": a.get("implied_raw"),
                          "implied_prop": a.get("implied_prop"), "implied_shin": a.get("implied_shin"),
                          "back_ltp_pre": compact(a), "tick_down_c5": a.get("roi_c5_tick_down"),
                          "back_bsp": compact(b), "lay_ltp_pre_c5": [a.get("lay_roi_c5"), a.get("lay_se")]})
    out({"section": "calibration", "country": code, "rows": band_rows})

    # FLB index: short-priced minus long-priced ROI before commission (not fitted)
    short_i = np.flatnonzero(lp < SHORT_MAX)
    long_i = np.flatnonzero(lp >= LONG_MIN)
    r0 = rets.back["ltp_pre"][0.0]
    ms, ss = mean_se(r0[short_i])
    ml, sl = mean_se(r0[long_i])
    flb_index = {"short_lt_4.5": {"bets": int(len(short_i)), "roi_c0": r4(ms), "se": r4(ss)},
                 "long_ge_15": {"bets": int(len(long_i)), "roi_c0": r4(ml), "se": r4(sl)},
                 "index_short_minus_long": r4(ms - ml), "index_se": r4(math.hypot(ss, sl)),
                 "index_z": r4((ms - ml) / math.hypot(ss, sl)) if ss and sl else None}

    # power recalibration: full-sample and per-half (descriptive) + walk-forward (scored)
    x = np.log(df["p_prop"].to_numpy(dtype=float))
    beta_full, se_full = fit_beta(x, won, codes, n_races)
    beta_h = []
    for mask in (first_m, ~first_m):
        b_, s_ = fit_beta(x[mask], won[mask], codes[mask], n_races)
        beta_h.append([round(b_, 4), round(s_, 4)])
    p_model, fits = walk_forward_power(df, burn_in)
    scored = ~np.isnan(p_model)
    ll_prop = race_log_loss(df["p_prop"].to_numpy(), won, codes, n_races)
    ll_shin = race_log_loss(df["p_shin"].to_numpy(), won, codes, n_races)
    ll_model = race_log_loss(p_model, won, codes, n_races)
    race_first = np.zeros(n_races, dtype=bool)
    race_first[codes[first_m]] = True
    ok_r = ~np.isnan(ll_model)
    gain_model = ll_prop - ll_model
    gain_shin = ll_prop - ll_shin
    gm, gm_se = mean_se(gain_model[ok_r])
    gs, gs_se = mean_se(gain_shin[ok_r])
    gm_h = [mean_se(gain_model[ok_r & m]) for m in (race_first, ~race_first)]
    gs_h = [mean_se(gain_shin[ok_r & m]) for m in (race_first, ~race_first)]
    flb = {"beta_full_sample": [round(beta_full, 4), round(se_full, 4)], "beta_halves": beta_h,
           "beta_walk_forward_first_last": [fits[0]["beta"] if fits else None, fits[-1]["beta"] if fits else None],
           "beta_walk_forward_median": r4(np.median([f["beta"] for f in fits])) if fits else None,
           "scored_races": int(ok_r.sum()), "burn_in_months": burn_in,
           "log_loss_market_prop": r4(np.nanmean(ll_prop[ok_r])), "log_loss_shin": r4(np.nanmean(ll_shin[ok_r])),
           "log_loss_walk_forward_power": r4(np.nanmean(ll_model[ok_r])),
           "gain_power_vs_prop": {"mean": r4(gm), "se": r4(gm_se), "z": r4(gm / gm_se) if gm_se else None,
                                  "halves": [[r4(a), r4(b)] for a, b in gm_h]},
           "gain_shin_vs_prop": {"mean": r4(gs), "se": r4(gs_se), "z": r4(gs / gs_se) if gs_se else None,
                                 "halves": [[r4(a), r4(b)] for a, b in gs_h]},
           "flb_index": flb_index}
    ll_tests.append((code, "power_wf_vs_prop", gm, gm_se, gm_h))
    out({"section": "favourite_longshot", "country": code, **flb})

    # value rule from the walk-forward probabilities (burn-in excluded)
    edge = np.where(scored, p_model * lp - 1.0, np.nan)
    value_rows = []
    for m in VALUE_MARGINS:
        idx = np.flatnonzero(scored & (edge >= m))
        a = rets.cell(idx, "ltp_pre")
        b = rets.cell(idx, "bsp")
        cells.append((code, "value", f"edge>={m:g}", "ltp_pre", a))
        cells.append((code, "value", f"edge>={m:g}", "bsp", b))
        value_rows.append({"rule": f"back if p_model*ltp_pre-1 >= {m:g}", "n": a["bets"],
                           "median_price": a.get("median_price"), "strike": a.get("strike"),
                           "back_ltp_pre": compact(a), "tick_down_c5": a.get("roi_c5_tick_down"), "back_bsp": compact(b)})
    idx = np.flatnonzero(scored & (edge <= -LAY_MARGIN))
    a = rets.cell(idx, "ltp_pre")
    b = rets.cell(idx, "bsp")
    lays.append((code, "value", f"edge<=-{LAY_MARGIN:g}", "ltp_pre", a))
    lays.append((code, "value", f"edge<=-{LAY_MARGIN:g}", "bsp", b))
    value_rows.append({"rule": f"lay if p_model*ltp_pre-1 <= -{LAY_MARGIN:g}", "n": a["bets"],
                       "median_price": a.get("median_price"),
                       "lay_ltp_pre_c5": [a.get("lay_roi_c5"), a.get("lay_se"), *(a.get("lay_halves_roi_c5") or [])],
                       "lay_bsp_c5": [b.get("lay_roi_c5"), b.get("lay_se"), *(b.get("lay_halves_roi_c5") or [])]})
    out({"section": "value_rule", "country": code, "rows": value_rows})

    fav1 = next(c for c in cells if c[0] == code and c[2] == "fav1" and c[3] == "ltp_pre")[4]
    b50 = next(c for c in cells if c[0] == code and c[2] == "50+" and c[3] == "ltp_pre")[4]
    summary = {"country": code, "markets": info["markets_used"], "runners": info["runners_used"],
               "share_runners_traded": info["share_runners_any_pre_off_trade"],
               "share_runners_t30": info["share_runners_t30_trade"],
               "share_markets_all_traded": info["share_markets_all_runners_traded"],
               "book_median": books["book_ltp_pre"]["median"], "book_q90": books["book_ltp_pre"]["q90"],
               "book_bsp_median": books["book_bsp"].get("median"),
               "beta_full": flb["beta_full_sample"], "wf_ll_gain": [r4(gm), r4(gm_se)],
               "shin_ll_gain": [r4(gs), r4(gs_se)],
               "fav1_roi_c0_c5": [fav1.get("roi_c0"), fav1.get("roi_c5")], "fav1_se": fav1.get("se"),
               "band50plus_roi_c0": b50.get("roi_c0"), "band50plus_se": b50.get("se"),
               "flb_index": [flb_index["index_short_minus_long"], flb_index["index_se"]],
               "seconds": round(time.time() - t0, 1)}
    del df, rets
    gc.collect()
    return summary


# ----------------------------------------------------------------------------- families, BH, dev/holdout


def family_report(name: str, items: list, roi_key: str, se_key: str, halves_key: str, halves_se_key: str) -> list[dict]:
    rows = []
    for country, fam, label, src, c in items:
        if c.get("bets", 0) < 2 or not c.get(se_key):
            z = float("nan")
        else:
            z = c[roi_key] / c[se_key]
        rows.append({"country": country, "family": fam, "cell": label, "source": src, "bets": c.get("bets"),
                     "roi_c5": c.get(roi_key), "se": c.get(se_key), "z": r4(z), "p": one_sided_p(z),
                     "halves": c.get(halves_key), "halves_se": c.get(halves_se_key)})
    rej = benjamini_hochberg([r["p"] for r in rows])
    for r, rj in zip(rows, rej):
        r["bh_reject"] = bool(rj)
        hz = [half_z(r, h, "halves", "halves_se") for h in (0, 1)]
        r["both_halves_gt_2se"] = bool(all(not math.isnan(z) and z > 2 for z in hz))
        r["survivor"] = bool(rj and r["both_halves_gt_2se"])
    tested = int(sum(not math.isnan(r["p"]) for r in rows))
    survivors = [r for r in rows if r["survivor"]]
    bh_only = [r for r in rows if r["bh_reject"]]
    out({"section": f"{name}_multiple_testing", "cells_tested": tested, "fdr": FDR,
         "bh_rejections": len(bh_only), "survivors_bh_and_both_halves": len(survivors),
         "bh_rejections_list": [{k: r[k] for k in ("country", "cell", "source", "bets", "roi_c5", "se", "z", "halves")}
                                for r in bh_only],
         "survivors": [{k: r[k] for k in ("country", "cell", "source", "bets", "roi_c5", "se", "z", "halves", "halves_se")}
                       for r in survivors]})
    # development half -> holdout half
    for dev in (0, 1):
        hold = 1 - dev
        picked = []
        for r in rows:
            zd = half_z(r, dev, "halves", "halves_se")
            if not math.isnan(zd) and zd > 2:
                zh = half_z(r, hold, "halves", "halves_se")
                picked.append({"country": r["country"], "cell": r["cell"], "source": r["source"],
                               "dev_roi_c5": r["halves"][dev], "dev_z": r4(zd),
                               "holdout_roi_c5": r["halves"][hold], "holdout_z": r4(zh),
                               "holds": bool(not math.isnan(zh) and zh > 2)})
        out({"section": f"{name}_dev_holdout", "development_half": dev + 1, "picked_z_gt_2": len(picked),
             "hold_in_other_half": int(sum(p["holds"] for p in picked)), "cells": picked})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default=",".join(COUNTRIES))
    ap.add_argument("--end", default=None, help="use markets strictly before this date (development)")
    ap.add_argument("--burn-in", type=int, default=3, help="months before the first walk-forward fit")
    a = ap.parse_args()
    t0 = time.time()
    cells: list = []   # back family
    lays: list = []    # lay family
    ll_tests: list = []
    summaries = []
    for code in [c.strip().upper() for c in a.countries.split(",") if c.strip()]:
        s = analyse_country(code, a.end, a.burn_in, cells, lays, ll_tests)
        if s:
            summaries.append(s)
    out({"section": "summary", "note": "beta>1 = favourites under-bet (favourite-longshot bias); "
         "flb_index = roi_c0(<4.5) - roi_c0(15+); wf_ll_gain = walk-forward log-loss gain of the power "
         "recalibration vs the proportional market, per race [mean, se]", "rows": summaries})
    family_report("back", cells, "roi_c5", "se", "halves_roi_c5", "halves_se_c5")
    family_report("lay", lays, "lay_roi_c5", "lay_se", "lay_halves_roi_c5", "lay_halves_se_c5")
    ll_rows = []
    for code, name, g, se, halves in ll_tests:
        z = g / se if se else float("nan")
        ll_rows.append({"country": code, "test": name, "gain": r4(g), "se": r4(se), "z": r4(z), "p": one_sided_p(z),
                        "halves_gain": [r4(h[0]) for h in halves], "halves_se": [r4(h[1]) for h in halves]})
    rej = benjamini_hochberg([r["p"] for r in ll_rows])
    for r, rj in zip(ll_rows, rej):
        r["bh_reject"] = bool(rj)
        r["both_halves_gt_2se"] = bool(all(s and g_ / s > 2 for g_, s in zip(r["halves_gain"], r["halves_se"])))
    out({"section": "log_loss_multiple_testing", "cells_tested": len(ll_rows), "rows": ll_rows})
    out({"section": "done", "seconds": round(time.time() - t0, 1),
         "back_cells": len(cells), "lay_cells": len(lays), "log_loss_cells": len(ll_rows)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

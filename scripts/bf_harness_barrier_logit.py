"""Experiment "harness_barrier_logit": the harness barrier draw (program number) as
a residual to the exchange price, with the gallops saddlecloth as a placebo and NZ
harness out-of-sample.

    python scripts/bf_harness_barrier_logit.py [--dev] [--l2 0.1]

Data: data/raw/betfair/split/{au,nz}_win.parquet (licensed, local only). Prints JSON
summary lines only; never horse names or rows.

Sections (every line is JSON):
  data             filters and counts per code (AU, NZ)
  halves           the fixed median-start date of each evaluation set
  ratio            per barrier sum(won)/sum(p_mkt) with SE, harness and gallops (not BH cells)
  dist             descriptive ratio by distance bucket x barrier group (not BH cells)
  l2               L2 choice (2016-12..2017-06 train -> 2017-07..2017-12 log loss), frozen
  refit            one line per monthly walk-forward refit (elapsed seconds)
  coef             mean raw coefficient over the refits per barrier dummy, sign stability
  logloss          log-loss cell 'au_harness_full' (model vs market)
  bet_cell         every bet cell (224) with Benjamini-Hochberg at q=0.10 over all of them
  opposite_half    best cell by z in one half evaluated in the other half, per family
  primary          the pre-registered primary and secondary cells
  caveat           verbatim caveat

Conventions (shared with the sibling experiments): settled WIN market = won.notna() on
every row and sum(won) >= 1; settlement fraction f = dead_heat_share; back return =
f*(P-1)*(1-c) - (1-f); lay (1 unit backer's stake) return = (1-f)*(1-c) - f*(P-1),
liability P-1; commission on net winnings only [A-54]; settlements 'ltp' (ltp_pre, a
paper price up to a minute stale) and 'bsp' (executable with an SP order before the
off); ROI = mean return, SE = std(ddof=1)/sqrt(bets); halves at the median start of the
evaluation set; walk-forward fits use only races starting strictly before the first
day of the predicted month.
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
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ponies.model import ConditionalLogit, _race_codes, race_softmax, win_targets  # noqa: E402

DATA = REPO / "data" / "raw" / "betfair" / "split"
COLS = ["market_id", "market_name", "start", "date", "field_size", "selection_id", "program",
        "ltp_pre", "bsp", "won", "dead_heat_share"]
COMMISSIONS = (0.0, 0.05, 0.08)
Q_BH = 0.10
L2_GRID = (0.01, 0.1, 1.0)
HARNESS_RE = r"\b(?:Pace|Trot)\b"
TROT_RE = r"\bTrot\b"
DIST_RE = r"(\d{3,4})[mM]"
MAX_BARRIER = 13
F_LABELS = ("<=8", "9-10", "11+")
K_VALUES = (0.05, 0.10, 0.15)
SETTLEMENTS = (("ltp", "ltp_pre"), ("bsp", "bsp"))
EXPECTED_CELLS = 224
CAVEAT = ("after a scratching, AU harness runners drawn wider on the same row move one gate inward, "
          "so program != actual gate for some runners (noise, not bias); front-row width (7-10) "
          "varies by track and is proxied by F.")

BARRIER_FEATURES = [f"B{k}" for k in range(2, MAX_BARRIER + 1)]
FEATURES = (["log_p"] + BARRIER_FEATURES + [f"{b}_trot" for b in BARRIER_FEATURES]
            + [f"{b}_F910" for b in BARRIER_FEATURES] + [f"{b}_F11" for b in BARRIER_FEATURES])
assert len(FEATURES) == 49

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
        return o.isoformat()
    if isinstance(o, pd.Period):
        return str(o)
    raise TypeError(str(type(o)))


def out(obj) -> None:
    print(json.dumps(obj, default=_json_default), flush=True)


def r5(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) else round(x, 5)


def two_sided_p(z: float) -> float:
    if z is None or math.isnan(z):
        return float("nan")
    return math.erfc(abs(z) / math.sqrt(2))


def benjamini_hochberg(pvals, q: float) -> np.ndarray:
    """BH over ALL cells: an undefined p-value (too few bets) counts as 1."""
    p = np.where(np.isnan(np.asarray(pvals, dtype=float)), 1.0, np.asarray(pvals, dtype=float))
    m = len(p)
    rej = np.zeros(m, dtype=bool)
    if m == 0:
        return rej
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    below = np.flatnonzero(p[order] <= thresh)
    if len(below):
        rej[order[: below.max() + 1]] = True
    return rej


def tick_down(price: np.ndarray) -> np.ndarray:
    p = np.asarray(price, dtype=float)
    inc = np.full_like(p, 10.0)
    q = p - 1e-9
    for lo, hi, step in TICKS:
        inc = np.where((q >= lo) & (q < hi), step, inc)
    return np.maximum(1.01, np.round(p - inc, 2))


def back_returns(price, f, c):
    return f * (price - 1.0) * (1.0 - c) - (1.0 - f)


def lay_returns(price, f, c):
    return (1.0 - f) * (1.0 - c) - f * (price - 1.0)


def median_start(df: pd.DataFrame) -> pd.Timestamp:
    """Median scheduled start over the races (one value per market) of an evaluation set."""
    starts = df.drop_duplicates("market_id")["start"].sort_values().to_numpy()
    return pd.Timestamp(starts[len(starts) // 2])


# ----------------------------------------------------------------------------- data


def load(code: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(DATA / f"{code}_win.parquet", columns=COLS)
    n_raw = len(df)
    df = df.drop_duplicates(["market_id", "selection_id"]).reset_index(drop=True)
    n_dedup = len(df)
    df["_won_na"] = df["won"].isna().astype(int)
    df["_priced"] = (df["ltp_pre"] > 1).astype(int)        # NaN -> False
    df["_inv"] = np.where(df["_priced"] == 1, 1.0 / df["ltp_pre"], 0.0)
    df["program_num"] = pd.to_numeric(df["program"], errors="coerce")
    df["_prog_na"] = df["program_num"].isna().astype(int)
    m = df.groupby("market_id").agg(n=("_won_na", "size"), won_na=("_won_na", "sum"), won_sum=("won", "sum"),
                                    priced=("_priced", "sum"), book=("_inv", "sum"), prog_na=("_prog_na", "sum"),
                                    fs=("field_size", "first"))
    settled = (m["won_na"] == 0) & (m["won_sum"] >= 1)
    priced = m["priced"] == m["n"]
    book_ok = (m["book"] >= 0.95) & (m["book"] <= 1.08)
    field_ok = m["fs"] >= 5
    ok = settled & priced & book_ok & field_ok
    prog_ok = m["prog_na"] == 0
    info = {"section": "data", "code": code, "rows_raw": int(n_raw), "rows_after_dedupe": int(n_dedup),
            "markets_total": int(len(m)), "markets_settled": int(settled.sum()),
            "markets_settled_fully_priced": int((settled & priced).sum()),
            "markets_after_book_guard": int((settled & priced & book_ok).sum()),
            "markets_after_field_ge5": int(ok.sum()),
            "markets_dropped_nan_program": int((ok & ~prog_ok).sum()),
            "markets_used": int((ok & prog_ok).sum())}
    keep = df["market_id"].isin(m.index[ok & prog_ok])
    df = df[keep].drop(columns=["_won_na", "_prog_na", "program"]).copy()
    df["book"] = df["market_id"].map(m["book"])
    df["p_mkt"] = df["_inv"] / df["book"]
    df["log_p"] = np.log(df["p_mkt"])
    df = df.drop(columns=["_inv", "_priced"])
    name = df["market_name"].fillna("")
    df["harness"] = name.str.contains(HARNESS_RE)
    df["trot"] = name.str.contains(TROT_RE)
    df["barrier"] = np.minimum(df["program_num"].astype(int), MAX_BARRIER)
    df["F"] = pd.cut(df["field_size"], [0, 8, 10, 10_000], labels=F_LABELS).astype(str)
    dist = pd.to_numeric(name.str.extract(DIST_RE)[0], errors="coerce")
    df["dist"] = np.select([dist < 1800, dist <= 2200, dist > 2200], ["<1800", "1800-2200", ">2200"], "unknown")
    df["f"] = df["dead_heat_share"].astype(float)
    df["race_id"] = df["market_id"]
    df["month"] = df["start"].dt.tz_localize(None).dt.to_period("M")
    df = df.sort_values(["start", "market_id", "selection_id"]).reset_index(drop=True)
    info.update({"runners_used": int(len(df)), "harness_markets": int(df.loc[df["harness"], "market_id"].nunique()),
                 "trot_markets": int(df.loc[df["trot"], "market_id"].nunique()),
                 "gallops_markets": int(df.loc[~df["harness"], "market_id"].nunique()),
                 "bsp_missing_share": r5(df["bsp"].isna().mean()),
                 "dist_unknown_share": r5((df["dist"] == "unknown").mean()),
                 "start_min": df["start"].min(), "start_max": df["start"].max()})
    return df, info


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    b = df["barrier"].to_numpy()
    trot = df["trot"].to_numpy(dtype=float)
    f910 = (df["F"] == "9-10").to_numpy(dtype=float)
    f11 = (df["F"] == "11+").to_numpy(dtype=float)
    cols = {}
    for k in range(2, MAX_BARRIER + 1):
        d = (b == k).astype(float)
        cols[f"B{k}"] = d
        cols[f"B{k}_trot"] = d * trot
        cols[f"B{k}_F910"] = d * f910
        cols[f"B{k}_F11"] = d * f11
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


# ----------------------------------------------------------------------------- model


class LbfgsLogit(ConditionalLogit):
    """ponies.model.ConditionalLogit with fit() replaced by L-BFGS-B on the identical
    objective (mean per-race log loss + 0.5 * sum_j l2_j w_j^2, l2_j = 0 for the
    unpenalised names). _prepare / predict_proba are inherited unchanged."""

    def fit(self, df: pd.DataFrame) -> "LbfgsLogit":
        codes, n_races = _race_codes(df["race_id"])
        y = win_targets(df["won"].to_numpy(), codes, n_races)
        Z = self._prepare(df, fit=True)
        l2 = float(self.l2) * np.array([0.0 if f in self.unpenalized else 1.0 for f in self.feature_names])

        def loss_and_grad(w):
            p = race_softmax(Z @ w, codes, n_races)
            loss = -np.sum(y * np.log(p + 1e-15)) / n_races + 0.5 * w @ (l2 * w)
            grad = Z.T @ (p - y) / n_races + l2 * w
            return loss, grad

        res = minimize(loss_and_grad, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B",
                       options={"maxiter": 500})
        self.weights = res.x
        self.history = [float(res.fun)]
        self.n_iter = int(res.nit)
        self.converged = bool(res.success)
        return self

    def raw_coefs(self) -> dict:
        """Coefficient per unit of the raw feature (= weight / std): for a dummy, the
        within-race log-odds shift relative to the reference barrier 1."""
        return {f: float(w / s) for f, w, s in zip(self.feature_names, self.weights, self.stds)}


def per_race_logloss(df: pd.DataFrame, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-ln(prob of the winner) per race; dead heats: mean over co-winners (f = 1/n_winners)."""
    codes, n = _race_codes(df["race_id"])
    f = df["f"].to_numpy()
    ll = np.bincount(codes, weights=f * -np.log(np.clip(p, 1e-15, None)), minlength=n)
    first_start = np.full(n, np.nan)
    starts = df["start"].to_numpy().astype("datetime64[ns]").astype("int64").astype(float)
    np.minimum.at(first_start, codes, starts)
    return ll, first_start


def logloss_cell(name: str, df: pd.DataFrame, p_model: np.ndarray, mid: pd.Timestamp) -> dict:
    ll_m, starts = per_race_logloss(df, df["p_mkt"].to_numpy())
    ll_x, _ = per_race_logloss(df, p_model)
    d = ll_m - ll_x
    mid_ns = float(pd.Timestamp(mid).tz_convert("UTC").tz_localize(None).value)
    first = starts < mid_ns
    res = {"section": "logloss", "cell": name, "n_races": int(len(d)), "ll_market": r5(ll_m.mean()),
           "ll_model": r5(ll_x.mean()), "gain": r5(d.mean()), "se": r5(d.std(ddof=1) / math.sqrt(len(d)))}
    for tag, mask in (("h1", first), ("h2", ~first)):
        h = d[mask]
        res[f"gain_{tag}"] = r5(h.mean()) if len(h) else None
        res[f"se_{tag}"] = r5(h.std(ddof=1) / math.sqrt(len(h))) if len(h) > 1 else None
        res[f"n_{tag}"] = int(len(h))
    res["positive"] = bool(res["gain_h1"] is not None and res["gain_h2"] is not None
                           and res["gain_h1"] > 2 * res["se_h1"] and res["gain_h2"] > 2 * res["se_h2"])
    return res


# ----------------------------------------------------------------------------- bet cells


def bet_cell(name: str, family: str, settlement: str, price: np.ndarray, f: np.ndarray, start: np.ndarray,
             mid: pd.Timestamp, side: str = "back", extra: dict | None = None) -> dict:
    price = np.asarray(price, dtype=float)
    f = np.asarray(f, dtype=float)
    ok = np.isfinite(price) & (price > 1)
    res = {"section": "bet_cell", "cell": name, "family": family, "settlement": settlement, "side": side,
           "bets": int(ok.sum()), "skipped": int((~ok).sum())}
    price, f = price[ok], f[ok]
    first = np.asarray(start)[ok] < np.datetime64(pd.Timestamp(mid).tz_convert("UTC").tz_localize(None))
    n = len(price)
    res["dead_heat_bets"] = int(((f > 0) & (f < 1)).sum())
    res["bets_h1"], res["bets_h2"] = int(first.sum()), int((~first).sum())
    fn = back_returns if side == "back" else lay_returns
    rets = {c: fn(price, f, c) for c in COMMISSIONS}
    for c in COMMISSIONS:
        res[f"roi_c{int(c * 100)}"] = r5(rets[c].mean()) if n else None
    if side == "lay":
        liab = (price - 1.0).sum()
        for c in COMMISSIONS:
            res[f"roi_liab_c{int(c * 100)}"] = r5(rets[c].sum() / liab) if n else None
    r = rets[0.05]
    res["se"] = r5(r.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    for tag, mask in (("h1", first), ("h2", ~first)):
        h = r[mask]
        res[f"roi_{tag}_c5"] = r5(h.mean()) if len(h) else None
        res[f"se_{tag}"] = r5(h.std(ddof=1) / math.sqrt(len(h))) if len(h) > 1 else None
    z = res["roi_c5"] / res["se"] if res["se"] else float("nan")
    res["z_c5"] = r5(z)
    res["p_two_sided"] = r5(two_sided_p(z))
    res["strike"] = r5((f > 0).mean()) if n else None
    res["median_price"] = r5(np.median(price)) if n else None
    if side == "back" and settlement == "ltp" and n:
        res["roi_c5_tick_down"] = r5(back_returns(tick_down(price), f, 0.05).mean())
    if extra:
        res.update(extra)
    return res


def finalize(cells: list[dict]) -> list[dict]:
    """Benjamini-Hochberg over every bet cell, then the pre-registered 'positive' rule."""
    p = np.array([c["p_two_sided"] if c["p_two_sided"] is not None else np.nan for c in cells])
    rej = benjamini_hochberg(p, Q_BH)
    for c, ok in zip(cells, rej):
        c["bh_pass"] = bool(ok)
        c["positive"] = bool(ok and c["roi_h1_c5"] is not None and c["se_h1"] is not None
                             and c["roi_h2_c5"] is not None and c["se_h2"] is not None
                             and c["roi_h1_c5"] > 2 * c["se_h1"] and c["roi_h2_c5"] > 2 * c["se_h2"])
    return cells


def opposite_half(cells: list[dict], label: str) -> None:
    """Development on one half, evaluation on the other: the best cell (by z) in H1 is
    reported in H2 and vice versa, per settlement."""
    for settlement in ("ltp", "bsp"):
        fam = [c for c in cells if c["settlement"] == settlement and c["bets_h1"] >= 200 and c["bets_h2"] >= 200
               and c["se_h1"] and c["se_h2"]]
        if not fam:
            continue
        for dev, ev in (("h1", "h2"), ("h2", "h1")):
            best = max(fam, key=lambda c: c[f"roi_{dev}_c5"] / c[f"se_{dev}"])
            zd = best[f"roi_{dev}_c5"] / best[f"se_{dev}"]
            ze = best[f"roi_{ev}_c5"] / best[f"se_{ev}"]
            out({"section": "opposite_half", "family": label, "settlement": settlement, "dev_half": dev,
                 "cell": best["cell"], "dev_roi_c5": best[f"roi_{dev}_c5"], "dev_z": r5(zd),
                 "eval_half": ev, "eval_roi_c5": best[f"roi_{ev}_c5"], "eval_se": best[f"se_{ev}"],
                 "eval_z": r5(ze), "holds_eval_gt_2se": bool(ze > 2), "cells_in_family": len(fam)})


# ----------------------------------------------------------------------------- descriptive


def descriptive(au: pd.DataFrame, dev: bool, cells: list[dict]) -> None:
    ev = au[au["month"].dt.year == 2019] if dev else au
    har = ev[ev["harness"]]
    gal = ev[~ev["harness"]]
    mid = median_start(har)
    out({"section": "halves", "evaluation_set": "descriptive (all qualifying AU races, halves at the median "
         "start of qualifying AU harness races)", "median_start": mid, "period": [ev["month"].min(), ev["month"].max()],
         "harness_races": int(har["market_id"].nunique()), "gallops_races": int(gal["market_id"].nunique())})

    def add(name, family, g, extra=None):
        for settlement, col in SETTLEMENTS:
            cells.append(bet_cell(name, family, settlement, g[col].to_numpy(), g["f"].to_numpy(),
                                  g["start"].dt.tz_localize(None).to_numpy(), mid, "back",
                                  dict(extra or {}, mean_p_mkt=r5(g["p_mkt"].mean()))))

    for b in range(1, MAX_BARRIER + 1):
        hb = har[har["barrier"] == b]
        for typ, mask in (("pace", ~hb["trot"]), ("trot", hb["trot"])):
            add(f"au_harness_B{b}_{typ}", "desc_harness_type", hb[mask], {"barrier": b, "type": typ})
    for b in range(1, MAX_BARRIER + 1):
        hb = har[har["barrier"] == b]
        for F in F_LABELS:
            add(f"au_harness_B{b}_F{F}", "desc_harness_field", hb[hb["F"] == F], {"barrier": b, "F": F})
    for b in range(1, MAX_BARRIER + 1):
        gb = gal[gal["barrier"] == b]
        for F in F_LABELS:
            add(f"au_gallops_B{b}_F{F}", "placebo_gallops_field", gb[gb["F"] == F], {"barrier": b, "F": F})

    # ratio sum(won)/sum(p_mkt) per barrier (not BH cells); SE from the binomial variance
    # of the winners count under the market probabilities, divided by sum(p_mkt)
    for code, g in (("au_harness", har), ("au_gallops", gal)):
        for b in range(1, MAX_BARRIER + 1):
            gb = g[g["barrier"] == b]
            p = gb["p_mkt"].to_numpy()
            w = gb["won"].to_numpy()
            sp = p.sum()
            out({"section": "ratio", "code": code, "barrier": b, "runners": int(len(gb)), "won": int(w.sum()),
                 "sum_p_mkt": r5(sp), "ratio": r5(w.sum() / sp) if sp > 0 else None,
                 "se": r5(math.sqrt((p * (1 - p)).sum()) / sp) if sp > 0 else None,
                 "mean_ltp_pre": r5(gb["ltp_pre"].mean()) if len(gb) else None})
    # descriptive distance split (not BH cells)
    groups = {"1-3": (1, 3), "4-6": (4, 6), "7-9": (7, 9), "10+": (10, 99)}
    for code, g in (("au_harness", har), ("au_gallops", gal)):
        for d in ("<1800", "1800-2200", ">2200", "unknown"):
            gd = g[g["dist"] == d]
            row = {"section": "dist", "code": code, "dist": d, "races": int(gd["market_id"].nunique())}
            for lab, (lo, hi) in groups.items():
                gg = gd[(gd["barrier"] >= lo) & (gd["barrier"] <= hi)]
                sp = gg["p_mkt"].sum()
                row[f"ratio_{lab}"] = r5(gg["won"].sum() / sp) if sp > 0 else None
            out(row)


# ----------------------------------------------------------------------------- walk-forward


def month_bounds(m: pd.Period) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(m.start_time, tz="UTC"), pd.Timestamp((m + 1).start_time, tz="UTC")


def choose_l2(har: pd.DataFrame) -> float:
    train = har[(har["start"] >= pd.Timestamp("2016-12-01", tz="UTC")) & (har["start"] < pd.Timestamp("2017-07-01", tz="UTC"))]
    valid = har[(har["start"] >= pd.Timestamp("2017-07-01", tz="UTC")) & (har["start"] < pd.Timestamp("2018-01-01", tz="UTC"))]
    ll_mkt, _ = per_race_logloss(valid, valid["p_mkt"].to_numpy())
    results = {}
    for l2 in L2_GRID:
        t0 = time.time()
        model = LbfgsLogit(FEATURES, l2=l2, unpenalized=("log_p",)).fit(train)
        ll, _ = per_race_logloss(valid, model.predict_proba(valid))
        results[l2] = {"ll_valid": r5(ll.mean()), "gain_vs_market": r5((ll_mkt - ll).mean()),
                       "iters": model.n_iter, "converged": model.converged, "seconds": round(time.time() - t0, 1)}
    best = min(results, key=lambda k: results[k]["ll_valid"])
    out({"section": "l2", "train": "2016-12..2017-06", "valid": "2017-07..2017-12", "train_races": int(train["market_id"].nunique()),
         "valid_races": int(valid["market_id"].nunique()), "ll_market_valid": r5(ll_mkt.mean()),
         "grid": {str(k): v for k, v in results.items()}, "chosen_l2": best, "frozen": True})
    return best


def walk_forward(har: pd.DataFrame, nz: pd.DataFrame, l2: float, dev: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    months = pd.period_range("2019-01", "2019-03", freq="M") if dev else pd.period_range("2018-01", "2020-11", freq="M")
    p_au = np.full(len(har), np.nan)
    p_nz = np.full(len(nz), np.nan)
    coefs = []
    for m in months:
        t0 = time.time()
        lo, hi = month_bounds(m)
        train = har[har["start"] < lo]
        test = (har["start"] >= lo) & (har["start"] < hi)
        test_nz = (nz["start"] >= lo) & (nz["start"] < hi)
        model = LbfgsLogit(FEATURES, l2=l2, unpenalized=("log_p",)).fit(train)
        if test.any():
            p_au[test.to_numpy()] = model.predict_proba(har[test])
        if test_nz.any():
            p_nz[test_nz.to_numpy()] = model.predict_proba(nz[test_nz])
        rc = model.raw_coefs()
        rc["month"] = str(m)
        coefs.append(rc)
        out({"section": "refit", "month": str(m), "train_races": int(train["market_id"].nunique()),
             "train_rows": int(len(train)), "train_start_max": train["start"].max(),
             "test_races_au": int(har.loc[test, "market_id"].nunique()), "test_races_nz": int(nz.loc[test_nz, "market_id"].nunique()),
             "iters": model.n_iter, "converged": model.converged, "train_loss": r5(model.history[-1]),
             "coef_log_p_raw": r5(rc["log_p"]), "seconds": round(time.time() - t0, 1)})
    co = pd.DataFrame(coefs)
    for b in BARRIER_FEATURES:
        row = {"section": "coef", "barrier": b, "refits": int(len(co))}
        for suffix, lab in (("", "main"), ("_trot", "x_trot"), ("_F910", "x_F9-10"), ("_F11", "x_F11+")):
            v = co[f"{b}{suffix}"].to_numpy()
            mean = v.mean()
            row[f"{lab}_mean"] = r5(mean)
            row[f"{lab}_sign_share"] = r5((np.sign(v) == np.sign(mean)).mean()) if mean != 0 else None
        # implied total for the three field buckets in a pace race
        row["pace_total_F<=8"] = r5(co[b].mean())
        row["pace_total_F9-10"] = r5(co[b].mean() + co[f"{b}_F910"].mean())
        row["pace_total_F11+"] = r5(co[b].mean() + co[f"{b}_F11"].mean())
        out(row)
    out({"section": "coef", "barrier": "log_p", "main_mean": r5(co["log_p"].mean()),
         "main_min": r5(co["log_p"].min()), "main_max": r5(co["log_p"].max()), "refits": int(len(co))})
    au_test = har[~np.isnan(p_au)].copy()
    au_test["p_model"] = p_au[~np.isnan(p_au)]
    nz_test = nz[~np.isnan(p_nz)].copy()
    nz_test["p_model"] = p_nz[~np.isnan(p_nz)]
    return au_test, nz_test, coefs


def rule_cells(test: pd.DataFrame, prefix: str, mid: pd.Timestamp, cells: list[dict], ks, flat_rules: bool) -> None:
    starts = test["start"].dt.tz_localize(None).to_numpy()

    def add(name, family, mask, side, extra=None):
        g = test[mask]
        for settlement, col in SETTLEMENTS:
            cells.append(bet_cell(name, family, settlement, g[col].to_numpy(), g["f"].to_numpy(), starts[mask], mid, side,
                                  dict(extra or {}, mean_p_mkt=r5(g["p_mkt"].mean()) if len(g) else None,
                                       mean_p_model=r5(g["p_model"].mean()) if len(g) else None,
                                       races=int(g["market_id"].nunique()))))

    edge = (test["p_model"] / test["p_mkt"]).to_numpy()
    cheap = (test["ltp_pre"] <= 15).to_numpy()
    for k in ks:
        add(f"{prefix}_a_k{k:.2f}", f"{prefix}_rules", (edge > 1 + k) & cheap, "back", {"rule": "a", "k": k})
    b = test["barrier"].to_numpy()
    add(f"{prefix}_b_back_B4-6", f"{prefix}_rules", (b >= 4) & (b <= 6), "back", {"rule": "b"})
    if flat_rules:
        add(f"{prefix}_c_lay_B1-3", f"{prefix}_rules", (b >= 1) & (b <= 3), "lay", {"rule": "c"})
        add(f"{prefix}_d_lay_B10+", f"{prefix}_rules", b >= 10, "lay", {"rule": "d"})


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="descriptive cells on 2019 only; walk-forward 2019-01..2019-03")
    ap.add_argument("--l2", type=float, default=None, help="override the frozen L2 choice (sensitivity runs only)")
    args = ap.parse_args()
    t_start = time.time()
    out({"section": "run", "experiment": "harness_barrier_logit", "dev": args.dev, "features": len(FEATURES),
         "commissions": COMMISSIONS, "bh_q": Q_BH, "expected_bet_cells": EXPECTED_CELLS})

    au, info_au = load("au")
    out(info_au)
    nz, info_nz = load("nz")
    nz = nz[nz["harness"]].copy()
    info_nz.update({"kept": "harness only, fully priced (every runner ltp_pre > 1) races",
                    "nz_harness_markets_used": int(nz["market_id"].nunique()), "nz_harness_runners_used": int(len(nz)),
                    "nz_bsp_missing_share": r5(nz["bsp"].isna().mean())})
    out(info_nz)

    cells: list[dict] = []
    descriptive(au, args.dev, cells)
    n_desc = len(cells)

    har = add_features(au[au["harness"]].copy())
    nz = add_features(nz)
    l2 = choose_l2(har)
    if args.l2 is not None:
        out({"section": "l2", "override": args.l2, "note": "sensitivity run: the frozen choice above is replaced"})
        l2 = args.l2

    au_test, nz_test, _ = walk_forward(har, nz, l2, args.dev)
    mid_test = median_start(au_test)
    out({"section": "halves", "evaluation_set": "AU harness test period (walk-forward predictions)",
         "median_start": mid_test, "period": [au_test["month"].min(), au_test["month"].max()],
         "races": int(au_test["market_id"].nunique()), "runners": int(len(au_test))})
    out(logloss_cell("au_harness_full", au_test, au_test["p_model"].to_numpy(), mid_test))
    # calibration of the model's edge in the test period (descriptive)
    edge = au_test["p_model"] / au_test["p_mkt"]
    bins = pd.cut(edge, [0, 0.9, 0.95, 1.0, 1.05, 1.10, 1.15, 1.3, 10])
    cal = au_test.groupby(bins, observed=True).agg(n=("won", "size"), strike=("won", "mean"), p_mkt=("p_mkt", "mean"),
                                                   p_model=("p_model", "mean"))
    out({"section": "edge_calibration", "evaluation_set": "AU harness test period",
         "rows": {str(k): {"n": int(v["n"]), "strike": r5(v["strike"]), "p_mkt": r5(v["p_mkt"]), "p_model": r5(v["p_model"])}
                  for k, v in cal.iterrows()}})
    rule_cells(au_test, "au_harness_test", mid_test, cells, K_VALUES, flat_rules=True)

    if len(nz_test):
        mid_nz = median_start(nz_test)
        out({"section": "halves", "evaluation_set": "NZ harness test period (AU model applied out-of-sample)",
             "median_start": mid_nz, "period": [nz_test["month"].min(), nz_test["month"].max()],
             "races": int(nz_test["market_id"].nunique()), "runners": int(len(nz_test))})
        out(logloss_cell("nz_harness_oos", nz_test, nz_test["p_model"].to_numpy(), mid_nz))
        rule_cells(nz_test, "nz_harness_test", mid_nz, cells, (0.10,), flat_rules=False)

    finalize(cells)
    out({"section": "bh", "total_bet_cells": len(cells), "expected": EXPECTED_CELLS, "descriptive_cells": n_desc,
         "q": Q_BH, "bh_pass": int(sum(c["bh_pass"] for c in cells)), "positive": int(sum(c["positive"] for c in cells)),
         "positive_cells": [f"{c['cell']}@{c['settlement']}" for c in cells if c["positive"]],
         "gallops_positive": [f"{c['cell']}@{c['settlement']}" for c in cells if c["positive"] and "gallops" in c["cell"]]})
    for c in cells:
        out(c)
    for fam, label in (("desc_harness_type", "desc_harness_type"), ("desc_harness_field", "desc_harness_field"),
                       ("placebo_gallops_field", "placebo_gallops_field"), ("au_harness_test_rules", "au_harness_test_rules")):
        opposite_half([c for c in cells if c["family"] == fam], label)

    def find(name, settlement):
        return next((c for c in cells if c["cell"] == name and c["settlement"] == settlement), None)

    out({"section": "primary", "pre_registered": "rule (a) k=0.10, settlement bsp, c=0.05, AU harness test period",
         "cell": find("au_harness_test_a_k0.10", "bsp")})
    out({"section": "primary", "pre_registered": "secondary: rule (b) flat back barrier 4-6 at bsp",
         "cell": find("au_harness_test_b_back_B4-6", "bsp")})
    out({"section": "caveat", "text": CAVEAT})
    out({"section": "done", "seconds": round(time.time() - t_start, 1)})


if __name__ == "__main__":
    main()

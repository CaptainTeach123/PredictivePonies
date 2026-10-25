"""Experiment "horse_market_memory": the horse's own exchange history (last-start market
move, beaten favourite, shrunk over/under-performance, recency) as a residual to the
current price.

    python scripts/bf_horse_market_memory.py [--dev] [--l2 0.1]

Data: data/raw/betfair/split/au_win.parquet (licensed, local only). Prints JSON summary
lines only; never horse names or rows. The horse key (lower-cased stripped name + harness
flag) is used internally only; selection_id is not persistent across races, and name
collisions between different horses are accepted (counted, not resolved).

HISTORY uses every settled run in the archive (including runs in markets that fail the
price filters); EVALUATION uses settled WIN markets with every runner ltp_pre > 1 and book
in [0.95, 1.08]. For each run, 'prev' = the same key's run with the latest start strictly
earlier; every cumulative quantity uses runs with start strictly earlier (asserted).

Sections (every line is JSON):
  run              arguments and expected cell counts
  data             filters and counts
  history          horse-key statistics (runs per key, gaps, p_run sources, collisions proxy)
  halves           the fixed median-start date of each evaluation set
  shares           share of evaluation rows with no prior run / with prev_steam_30 present
  desc_thresholds  cum_resid decile thresholds per code (computed in sample: descriptive)
  l2               L2 choice per code (2016-12..2017-06 train -> 2017-07..2017-12 log loss), frozen
  refit            one line per monthly walk-forward refit per code (elapsed seconds)
  coef             mean raw coefficient over the refits per FULL feature, sign stability
  logloss          PREVWON and FULL vs MARKET per code, FULL - PREVWON paired gain
  calibration      decile calibration of p_model(FULL) and edge-bin calibration over the test period
  bet_cell         every bet cell: 36 descriptive + 18 rules = 54 pre-registered (Benjamini-Hochberg #1 at
                   q=0.10 -> bh_pass) plus 18 exploratory rule cells at k in {0.01, 0.02, 0.03} added after the
                   dev run showed model/market disagreements > 5% almost never occur (BH #2 over all 72 ->
                   bh_pass_all); 'positive' uses the BH that applies to the cell's family
  opposite_half    best cell by z in one half evaluated in the other half, per family
  primary          the pre-registered primary cells
  caveat           verbatim caveats

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
from ponies.model import _race_codes, race_softmax, win_targets  # noqa: E402

DATA = REPO / "data" / "raw" / "betfair" / "split"
COLS = ["market_id", "market_name", "start", "date", "field_size", "selection_id", "horse_name",
        "ltp_pre", "ltp_t30", "bsp", "won", "dead_heat_share"]
COMMISSIONS = (0.0, 0.05, 0.08)
Q_BH = 0.10
L2_GRID = (0.01, 0.1, 1.0)
HARNESS_RE = r"\b(?:Pace|Trot)\b"
K_VALUES = (0.05, 0.10, 0.15)
K_EXPLORATORY = (0.01, 0.02, 0.03)   # NOT pre-registered: added after the dev run showed edges > 5% almost never occur
MAX_PRICE = 20.0
SHRINK = 5.0
STEAM_CUT = 0.2
SETTLEMENTS = (("ltp", "ltp_pre"), ("bsp", "bsp"))
CODES = ("gallops", "harness")
EXPECTED_DESC, EXPECTED_RULES = 36, 18
EXPECTED_CELLS = EXPECTED_DESC + EXPECTED_RULES          # pre-registered BH family
EXPECTED_EXPLORATORY = 18                                 # separate, labelled; BH #2 runs over all 72
FULL_FEATURES = ["log_p", "no_prior", "log1p_n_prior", "log_days_since", "prev_won", "prev_surprise", "beaten_fav",
                 "prev_steam_pre", "prev_steam_30", "steam30_missing", "cum_resid", "price_vs_prev", "field_diff",
                 "class_drop"]
PREVWON_FEATURES = ["log_p", "prev_won", "no_prior"]
HIST_FEATURES = [f for f in FULL_FEATURES if f != "log_p"]
DESC_PERIOD = ("2017-07-01", "2020-12-01")
DEV_DESC_PERIOD = ("2019-01-01", "2020-01-01")
TEST_MONTHS = ("2018-01", "2020-11")
DEV_TEST_MONTHS = ("2019-01", "2019-03")
L2_TRAIN = ("2016-12-01", "2017-07-01")
L2_VALID = ("2017-07-01", "2018-01-01")
CAVEATS = [
    "horse key = lower-cased name + harness flag: different horses sharing a name are merged (not resolved); "
    "a horse renamed or moving between codes starts a new key.",
    "prev_steam_30 exists only when the previous run traded 30 minutes before its start (~39% of prior runs); "
    "results involving it are conditional on that liquidity. prev_steam_pre needs the previous run's BSP (~97%).",
    "ltp settlement is a paper price up to a minute stale; roi_c5_tick_down backs one tick shorter. "
    "bsp settlement is executable with an SP order but the SP itself is unknown when the rule fires.",
    "cum_resid deciles for the two descriptive cells are computed on the whole evaluation set (in sample).",
    "the archive starts 2016-12: early history is truncated, so no_prior partly means 'no run since 2016-12'.",
]


def features_for(code: str) -> list[str]:
    return FULL_FEATURES if code == "gallops" else [f for f in FULL_FEATURES if f != "class_drop"]


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


def T(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def naive(ts: pd.Timestamp) -> np.datetime64:
    return np.datetime64(pd.Timestamp(ts).tz_convert("UTC").tz_localize(None))


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


# Betfair price ladder (lower inclusive, upper exclusive, increment) for the tick-down line
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1001.0, 10.0)]


def tick_down(price: np.ndarray) -> np.ndarray:
    p = np.asarray(price, dtype=float)
    inc = np.full_like(p, 10.0)
    q = p - 1e-9
    for lo, hi, step in TICKS:
        inc = np.where((q >= lo) & (q < hi), step, inc)
    return np.maximum(1.01, np.round(p - inc, 2))


def back_returns(price, f, c):
    return f * (price - 1.0) * (1.0 - c) - (1.0 - f)


def median_start(df: pd.DataFrame) -> pd.Timestamp:
    """Median scheduled start over the races (one value per market) of an evaluation set."""
    starts = df.drop_duplicates("market_id")["start"].sort_values().to_numpy()
    return pd.Timestamp(starts[len(starts) // 2])


# ----------------------------------------------------------------------------- data


def load() -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(DATA / "au_win.parquet", columns=COLS)
    n_raw = len(df)
    df = df.drop_duplicates(["market_id", "selection_id"]).reset_index(drop=True)
    name = df["market_name"].fillna("")
    df["harness"] = name.str.contains(HARNESS_RE).to_numpy()
    df["mdn"] = (name.str.contains("Mdn") & ~df["harness"]).astype(float).to_numpy()
    df["_won_na"] = df["won"].isna().astype(int)
    df["_priced"] = (df["ltp_pre"] > 1).astype(int)
    df["_inv"] = np.where(df["_priced"] == 1, 1.0 / df["ltp_pre"].where(df["_priced"] == 1, 2.0), 0.0)
    df["_bsp_ok"] = (df["bsp"] > 1).astype(int)
    df["_inv_bsp"] = np.where(df["_bsp_ok"] == 1, 1.0 / df["bsp"].where(df["_bsp_ok"] == 1, 2.0), 0.0)
    m = df.groupby("market_id").agg(n=("_won_na", "size"), won_na=("_won_na", "sum"), won_sum=("won", "sum"),
                                    priced=("_priced", "sum"), book=("_inv", "sum"),
                                    bsp_ok=("_bsp_ok", "sum"), book_bsp=("_inv_bsp", "sum"))
    settled = (m["won_na"] == 0) & (m["won_sum"] >= 1)
    pre_all = m["priced"] == m["n"]
    bsp_all = m["bsp_ok"] == m["n"]
    book_ok = (m["book"] >= 0.95) & (m["book"] <= 1.08)
    qual = settled & pre_all & book_ok
    mid = df["market_id"]
    df["settled"] = mid.map(settled).to_numpy(dtype=bool)
    df["qual"] = mid.map(qual).to_numpy(dtype=bool)
    mk_bsp_all = mid.map(bsp_all).to_numpy(dtype=bool)
    mk_pre_all = mid.map(pre_all).to_numpy(dtype=bool)
    p_bsp = (df["_inv_bsp"] / mid.map(m["book_bsp"]).to_numpy()).to_numpy()
    p_pre = (df["_inv"] / mid.map(m["book"]).to_numpy()).to_numpy()
    df["p_run"] = np.where(mk_bsp_all, p_bsp, np.where(mk_pre_all, p_pre, np.nan))
    df["p_run_source"] = np.where(mk_bsp_all, "bsp", np.where(mk_pre_all, "ltp_pre", "none"))
    df["p_mkt"] = np.where(df["qual"], p_pre, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["log_p"] = np.log(df["p_mkt"])
    df["f"] = df["dead_heat_share"].astype(float)
    df = df.drop(columns=["_won_na", "_priced", "_inv", "_bsp_ok", "_inv_bsp"])
    info = {"section": "data", "rows_raw": int(n_raw), "rows_after_dedupe": int(len(df)),
            "markets_total": int(len(m)), "markets_settled": int(settled.sum()),
            "markets_settled_fully_priced": int((settled & pre_all).sum()),
            "markets_qualifying_book_guard": int(qual.sum()),
            "history_rows_settled": int(df["settled"].sum()), "evaluation_rows_qualifying": int(df["qual"].sum()),
            "p_run_source_share_history": {k: r5(v) for k, v in
                                           df.loc[df["settled"], "p_run_source"].value_counts(normalize=True).items()},
            "harness_markets": int(df.loc[df["harness"], "market_id"].nunique()),
            "gallops_markets": int(df.loc[~df["harness"], "market_id"].nunique()),
            "qualifying_harness_markets": int(df.loc[df["harness"] & df["qual"], "market_id"].nunique()),
            "qualifying_gallops_markets": int(df.loc[~df["harness"] & df["qual"], "market_id"].nunique()),
            "bsp_missing_share": r5(df["bsp"].isna().mean()), "ltp_t30_present_share": r5(df["ltp_t30"].notna().mean()),
            "start_min": df["start"].min(), "start_max": df["start"].max()}
    return df, info


def add_history(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Point-in-time horse-history features over every settled run. Sorted by (key, start);
    prev = shift(1) within key (no key has two runs at one start: asserted), cumulative
    quantities are exclusive cumsums within key."""
    h = df[df["settled"]].copy()
    key = h["horse_name"].fillna("").str.strip().str.lower() + "|" + h["harness"].astype(str)
    h["key_id"] = pd.factorize(key)[0]
    h = h.sort_values(["key_id", "start"], kind="mergesort")
    assert not h.duplicated(["key_id", "start"]).any(), "a key with two runs at one start: prev would be ambiguous"
    g = h.groupby("key_id", sort=False)
    h["n_prior"] = g.cumcount()
    pmax = h.groupby("market_id")["p_run"].transform("max")
    h["is_fav"] = (h["p_run"].notna() & (h["p_run"] >= pmax - 1e-12)).astype(float)
    h["is_fav"] = h["is_fav"].where(h["p_run"].notna(), np.nan)
    for c in ("start", "f", "p_run", "ltp_pre", "ltp_t30", "bsp", "field_size", "is_fav", "mdn"):
        h["prev_" + c] = g[c].shift(1)
    has_prev = (h["n_prior"] > 0).to_numpy()
    assert (h.loc[has_prev, "prev_start"] < h.loc[has_prev, "start"]).all(), "leakage: prev_start >= start"
    assert h.loc[~has_prev, "prev_start"].isna().all()
    has_p = h["p_run"].notna()
    resid = (h["f"] - h["p_run"]).where(has_p, 0.0)
    h["cum_resid_sum"] = resid.groupby(h["key_id"]).cumsum() - resid           # exclusive: prior runs only
    h["n_prior_resid"] = has_p.astype(int).groupby(h["key_id"]).cumsum() - has_p.astype(int)
    h["cum_resid"] = h["cum_resid_sum"] / (h["n_prior"] + SHRINK)
    days = (h["start"] - h["prev_start"]).dt.total_seconds() / 86400.0
    h["days_since"] = days
    h["no_prior"] = (~has_prev).astype(float)
    h["log1p_n_prior"] = np.log1p(h["n_prior"].astype(float))
    h["log_days_since"] = np.log1p(days)
    h["prev_won"] = np.where(has_prev, (h["prev_f"] > 0).astype(float), 0.0)
    h["prev_p"] = h["prev_p_run"]
    h["prev_surprise"] = h["prev_won"] - h["prev_p"]                            # NaN if no prior / prev p_run NaN
    h["beaten_fav"] = (has_prev & (h["prev_is_fav"] == 1) & (h["prev_won"] == 0)).astype(float)
    pre_ok = (h["prev_ltp_pre"] > 1).to_numpy()
    bsp_ok = (h["prev_bsp"] > 1).to_numpy()
    t30_ok = (h["prev_ltp_t30"] > 1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        h["prev_steam_pre"] = np.where(pre_ok & bsp_ok, np.log(h["prev_ltp_pre"] / h["prev_bsp"]), np.nan)
        h["prev_steam_30"] = np.where(t30_ok & bsp_ok, np.log(h["prev_ltp_t30"] / h["prev_bsp"]), np.nan)
        h["price_vs_prev"] = np.where(pre_ok & (h["ltp_pre"] > 1), np.log(h["ltp_pre"]) - np.log(h["prev_ltp_pre"]), np.nan)
    h["steam30_missing"] = h["prev_steam_30"].isna().astype(float)
    h["field_diff"] = (h["field_size"] - h["prev_field_size"]).astype(float)
    h["class_drop"] = (has_prev & (h["prev_mdn"] == 1) & (h["mdn"] == 0) & ~h["harness"]).astype(float)
    runs = h.groupby("key_id").size()
    info = {"section": "history", "rows": int(len(h)), "keys": int(len(runs)),
            "share_rows_with_prior": r5(has_prev.mean()),
            "runs_per_key_median": r5(runs.median()), "runs_per_key_p90": r5(runs.quantile(0.9)),
            "runs_per_key_max": int(runs.max()),
            "keys_over_80_runs_collision_proxy": int((runs > 80).sum()),
            "rows_in_keys_over_80_runs": int(runs[runs > 80].sum()),
            "gap_days_median": r5(days.median()), "gap_days_p10": r5(days.quantile(0.1)), "gap_days_p90": r5(days.quantile(0.9)),
            "share_prior_rows_prev_p_run_missing": r5(h.loc[has_prev, "prev_p"].isna().mean()),
            "share_prior_rows_prev_steam_pre_present": r5(h.loc[has_prev, "prev_steam_pre"].notna().mean()),
            "share_prior_rows_prev_steam_30_present": r5(h.loc[has_prev, "prev_steam_30"].notna().mean()),
            "share_prior_rows_beaten_fav": r5(h.loc[has_prev, "beaten_fav"].mean()),
            "share_prior_rows_prev_won": r5(h.loc[has_prev, "prev_won"].mean()),
            "share_prior_rows_class_drop_gallops": r5(h.loc[has_prev & ~h["harness"].to_numpy(), "class_drop"].mean()),
            "leakage_assert": "prev_start < start for every row with a prior: passed"}
    h = h.sort_values(["start", "market_id", "selection_id"], kind="mergesort").reset_index(drop=True)
    return h, info


def evaluation_frame(h: pd.DataFrame) -> pd.DataFrame:
    ev = h[h["qual"]].copy()
    ev["race_id"] = ev["market_id"]
    ev["month"] = ev["start"].dt.tz_convert(None).dt.to_period("M")
    return ev.reset_index(drop=True)


def model_frame(ev: pd.DataFrame) -> pd.DataFrame:
    """Race-median imputation applied once (it depends only on the race, so this equals
    ponies.model.ConditionalLogit's per-fit step); remaining NaN (whole field missing)
    are filled with the training mean inside ResidualLogit."""
    keep = ["market_id", "race_id", "start", "month", "harness", "won", "f", "ltp_pre", "bsp", "p_mkt", "log_p"]
    md = ev[keep].copy()
    X = ev[HIST_FEATURES]
    med = X.groupby(ev["market_id"].to_numpy()).transform("median")
    filled = X.where(X.notna(), med)
    for c in HIST_FEATURES:
        md[c] = filled[c].to_numpy()
    return md


# ----------------------------------------------------------------------------- model


class ResidualLogit:
    """Conditional logit with the same objective as ponies.model.ConditionalLogit (mean per-race
    log loss + 0.5 * sum_j l2_j w_j^2 on standardised features, l2_j = 0 for the unpenalised
    names, softmax within race, dead-heat shares as targets) fitted by L-BFGS-B."""

    def __init__(self, feature_names, l2, unpenalized=("log_p",)):
        self.feature_names = list(feature_names)
        self.l2 = float(l2)
        self.unpenalized = tuple(unpenalized)

    def _prepare(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        X = df[self.feature_names].to_numpy(dtype=float)
        if fit:
            self.impute_values = np.nan_to_num(np.nanmean(X, axis=0))
        X = np.where(np.isnan(X), self.impute_values, X)
        if fit:
            self.means = X.mean(axis=0)
            self.stds = X.std(axis=0)
            self.stds[self.stds == 0] = 1.0
        return (X - self.means) / self.stds

    def fit(self, df: pd.DataFrame) -> "ResidualLogit":
        codes, n_races = _race_codes(df["race_id"])
        y = win_targets(df["won"].to_numpy(), codes, n_races)
        Z = self._prepare(df, fit=True)
        l2 = self.l2 * np.array([0.0 if f in self.unpenalized else 1.0 for f in self.feature_names])

        def loss_and_grad(w):
            p = race_softmax(Z @ w, codes, n_races)
            loss = -np.sum(y * np.log(p + 1e-15)) / n_races + 0.5 * w @ (l2 * w)
            grad = Z.T @ (p - y) / n_races + l2 * w
            return loss, grad

        res = minimize(loss_and_grad, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.weights = res.x
        self.train_loss = float(res.fun)
        self.n_iter = int(res.nit)
        self.converged = bool(res.success)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        codes, n_races = _race_codes(df["race_id"])
        return race_softmax(self._prepare(df, fit=False) @ self.weights, codes, n_races)

    def raw_coefs(self) -> dict:
        """Coefficient per unit of the raw feature (= weight / std): the within-race log-odds shift."""
        return {f: float(w / s) for f, w, s in zip(self.feature_names, self.weights, self.stds)}


def per_race_logloss(df: pd.DataFrame, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-ln(prob of the winner) per race; dead heats: mean over co-winners (f = 1/n_winners)."""
    codes, n = _race_codes(df["race_id"])
    f = df["f"].to_numpy()
    ll = np.bincount(codes, weights=f * -np.log(np.clip(p, 1e-15, None)), minlength=n)
    first_start = np.full(n, np.inf)
    starts = df["start"].dt.tz_convert(None).to_numpy().astype("datetime64[ns]").astype("int64").astype(float)
    np.minimum.at(first_start, codes, starts)
    return ll, first_start


def paired_gain(name: str, benchmark: str, df: pd.DataFrame, p_bench: np.ndarray, p_model: np.ndarray,
                mid: pd.Timestamp) -> dict:
    ll_b, starts = per_race_logloss(df, p_bench)
    ll_x, _ = per_race_logloss(df, p_model)
    d = ll_b - ll_x
    first = starts < float(pd.Timestamp(naive(mid)).value)
    res = {"section": "logloss", "cell": name, "benchmark": benchmark, "n_races": int(len(d)),
           "ll_benchmark": r5(ll_b.mean()), "ll_model": r5(ll_x.mean()), "gain": r5(d.mean()),
           "se": r5(d.std(ddof=1) / math.sqrt(len(d)))}
    res["z"] = r5(res["gain"] / res["se"]) if res["se"] else None
    for tag, mask in (("h1", first), ("h2", ~first)):
        h = d[mask]
        res[f"gain_{tag}"] = r5(h.mean()) if len(h) else None
        res[f"se_{tag}"] = r5(h.std(ddof=1) / math.sqrt(len(h))) if len(h) > 1 else None
        res[f"n_{tag}"] = int(len(h))
    res["positive"] = bool(res["gain_h1"] is not None and res["gain_h2"] is not None and res["se_h1"] and res["se_h2"]
                           and res["gain_h1"] > 2 * res["se_h1"] and res["gain_h2"] > 2 * res["se_h2"])
    return res


# ----------------------------------------------------------------------------- bet cells


def bet_cell(name: str, family: str, settlement: str, g: pd.DataFrame, mid: pd.Timestamp, extra: dict | None = None) -> dict:
    col = dict(SETTLEMENTS)[settlement]
    price = g[col].to_numpy(dtype=float)
    f = g["f"].to_numpy(dtype=float)
    ok = np.isfinite(price) & (price > 1)
    res = {"section": "bet_cell", "cell": name, "family": family, "settlement": settlement, "side": "back",
           "rows": int(len(g)), "bets": int(ok.sum()), "skipped_no_price": int((~ok).sum()),
           "races": int(g["market_id"].nunique())}
    price, f = price[ok], f[ok]
    first = g["start"].dt.tz_convert(None).to_numpy()[ok] < naive(mid)
    n = len(price)
    res["bets_h1"], res["bets_h2"] = int(first.sum()), int((~first).sum())
    rets = {c: back_returns(price, f, c) for c in COMMISSIONS}
    for c in COMMISSIONS:
        res[f"roi_c{int(c * 100)}"] = r5(rets[c].mean()) if n else None
    r = rets[0.05]
    res["se"] = r5(r.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    for tag, mask in (("h1", first), ("h2", ~first)):
        hh = r[mask]
        res[f"roi_{tag}_c5"] = r5(hh.mean()) if len(hh) else None
        res[f"se_{tag}"] = r5(hh.std(ddof=1) / math.sqrt(len(hh))) if len(hh) > 1 else None
    z = res["roi_c5"] / res["se"] if res["se"] else float("nan")
    res["z_c5"] = r5(z)
    res["p_two_sided"] = r5(two_sided_p(z))
    res["strike"] = r5((f > 0).mean()) if n else None
    res["median_price"] = r5(np.median(price)) if n else None
    if settlement == "ltp" and n:
        res["roi_c5_tick_down"] = r5(back_returns(tick_down(price), f, 0.05).mean())
    # won/expected ratio under the market probabilities, binomial SE
    p = g["p_mkt"].to_numpy(dtype=float)
    w = g["won"].to_numpy(dtype=float)
    sp = p.sum()
    res["won_over_expected"] = r5(w.sum() / sp) if sp > 0 else None
    res["won_over_expected_se"] = r5(math.sqrt((p * (1 - p)).sum()) / sp) if sp > 0 else None
    res["mean_p_mkt"] = r5(p.mean()) if len(p) else None
    if extra:
        res.update(extra)
    return res


def finalize(cells: list[dict]) -> list[dict]:
    """Benjamini-Hochberg #1 over the 54 pre-registered bet cells (bh_pass), BH #2 over every
    cell including the exploratory family (bh_pass_all), then the 'positive' rule: both halves
    > 2 SE at c=5% and the BH that applies to the cell's family."""
    pre = np.array([c["pre_registered"] for c in cells])
    p = np.array([c["p_two_sided"] if c["p_two_sided"] is not None else np.nan for c in cells])
    rej_pre = np.zeros(len(cells), dtype=bool)
    rej_pre[pre] = benjamini_hochberg(p[pre], Q_BH)
    rej_all = benjamini_hochberg(p, Q_BH)
    for c, ok_pre, ok_all in zip(cells, rej_pre, rej_all):
        c["bh_pass"] = bool(ok_pre) if c["pre_registered"] else None
        c["bh_pass_all"] = bool(ok_all)
        c["both_halves_gt_2se"] = bool(c["roi_h1_c5"] is not None and c["se_h1"] is not None and c["roi_h2_c5"] is not None
                                       and c["se_h2"] is not None and c["roi_h1_c5"] > 2 * c["se_h1"]
                                       and c["roi_h2_c5"] > 2 * c["se_h2"])
        c["positive"] = bool((ok_pre if c["pre_registered"] else ok_all) and c["both_halves_gt_2se"])
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


def descriptive(ev: pd.DataFrame, dev: bool, cells: list[dict]) -> None:
    lo, hi = DEV_DESC_PERIOD if dev else DESC_PERIOD
    d = ev[(ev["start"] >= T(lo)) & (ev["start"] < T(hi))]
    mid = median_start(d)
    out({"section": "halves", "evaluation_set": "descriptive (qualifying AU races, both codes)", "median_start": mid,
         "period": [d["month"].min(), d["month"].max()], "races": int(d["market_id"].nunique()), "runners": int(len(d)),
         "gallops_races": int(d.loc[~d["harness"], "market_id"].nunique()),
         "harness_races": int(d.loc[d["harness"], "market_id"].nunique())})
    for code in CODES:
        g = d[d["harness"] == (code == "harness")]
        prior = g["no_prior"] == 0
        out({"section": "shares", "code": code, "rows": int(len(g)), "share_no_prior": r5(g["no_prior"].mean()),
             "share_prev_steam_30_present": r5(g["prev_steam_30"].notna().mean()),
             "share_prev_steam_30_present_given_prior": r5(g.loc[prior, "prev_steam_30"].notna().mean()),
             "share_prev_steam_pre_present_given_prior": r5(g.loc[prior, "prev_steam_pre"].notna().mean()),
             "median_days_since_given_prior": r5(g.loc[prior, "days_since"].median()),
             "median_n_prior": r5(g["n_prior"].median())})
        q10, q90 = g["cum_resid"].quantile([0.1, 0.9]).to_numpy()
        out({"section": "desc_thresholds", "code": code, "cum_resid_q10": r5(q10), "cum_resid_q90": r5(q90),
             "note": "deciles computed on the whole evaluation set of this code (in sample, descriptive)",
             "cum_resid_mean": r5(g["cum_resid"].mean()), "cum_resid_sd": r5(g["cum_resid"].std())})
        conds = {
            "prev_won": (g["prev_won"] == 1, "last start won"),
            "beaten_fav": (g["beaten_fav"] == 1, "last start the market favourite (p_run max, ties count) and lost"),
            "backed_lost": ((g["prev_steam_pre"] > STEAM_CUT) & (g["prev_won"] == 0), "ln(prev ltp_pre/prev bsp) > 0.2 and lost"),
            "drifted_won": ((g["prev_steam_pre"] < -STEAM_CUT) & (g["prev_won"] == 1), "ln(prev ltp_pre/prev bsp) < -0.2 and won"),
            "no_prior": (g["no_prior"] == 1, "no earlier settled run in the archive under this key"),
            "days_le7": (g["days_since"] <= 7, "back-up within 7 days"),
            "days_ge60": (g["days_since"] >= 60, "60+ days since the last run"),
            "cum_resid_top_decile": (g["cum_resid"] >= q90, "shrunk cumulative (won - p_run) in the top decile (in sample)"),
            "cum_resid_bottom_decile": (g["cum_resid"] <= q10, "shrunk cumulative (won - p_run) in the bottom decile (in sample)"),
        }
        for cname, (mask, desc) in conds.items():
            gg = g[mask.to_numpy()]
            for settlement, _ in SETTLEMENTS:
                cells.append(bet_cell(f"desc_{code}_{cname}", f"desc_{code}", settlement, gg, mid,
                                      {"condition": desc, "descriptive_in_sample": cname.startswith("cum_resid"),
                                       "pre_registered": True}))


# ----------------------------------------------------------------------------- walk-forward


def month_bounds(m: pd.Period) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(m.start_time, tz="UTC"), pd.Timestamp((m + 1).start_time, tz="UTC")


def choose_l2(md: pd.DataFrame, code: str) -> float:
    d = md[md["harness"] == (code == "harness")]
    train = d[(d["start"] >= T(L2_TRAIN[0])) & (d["start"] < T(L2_TRAIN[1]))]
    valid = d[(d["start"] >= T(L2_VALID[0])) & (d["start"] < T(L2_VALID[1]))]
    ll_mkt, _ = per_race_logloss(valid, valid["p_mkt"].to_numpy())
    feats = features_for(code)
    results = {}
    for l2 in L2_GRID:
        t0 = time.time()
        model = ResidualLogit(feats, l2).fit(train)
        ll, _ = per_race_logloss(valid, model.predict_proba(valid))
        results[l2] = {"ll_valid": r5(ll.mean()), "gain_vs_market": r5((ll_mkt - ll).mean()),
                       "iters": model.n_iter, "converged": model.converged, "seconds": round(time.time() - t0, 1)}
    best = min(results, key=lambda k: results[k]["ll_valid"])
    out({"section": "l2", "code": code, "model": "FULL", "train": "2016-12..2017-06", "valid": "2017-07..2017-12",
         "train_races": int(train["market_id"].nunique()), "valid_races": int(valid["market_id"].nunique()),
         "ll_market_valid": r5(ll_mkt.mean()), "grid": {str(k): v for k, v in results.items()}, "chosen_l2": best,
         "frozen": True, "note": "the same L2 is used for PREVWON of this code"})
    return best


def walk_forward(md: pd.DataFrame, code: str, l2: float, months) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = md[md["harness"] == (code == "harness")]
    feats = features_for(code)
    p_full = np.full(len(d), np.nan)
    p_prev = np.full(len(d), np.nan)
    coefs = []
    starts = d["start"].dt.tz_convert(None).to_numpy()
    for m in months:
        t0 = time.time()
        lo, hi = month_bounds(m)
        train_mask = starts < naive(lo)
        test_mask = (starts >= naive(lo)) & (starts < naive(hi))
        train = d[train_mask]
        assert train["start"].max() < lo, "leakage: training race at/after the predicted month"
        full = ResidualLogit(feats, l2).fit(train)
        t1 = time.time()
        prevwon = ResidualLogit(PREVWON_FEATURES, l2).fit(train)
        if test_mask.any():
            test = d[test_mask]
            p_full[test_mask] = full.predict_proba(test)
            p_prev[test_mask] = prevwon.predict_proba(test)
        rc = full.raw_coefs()
        rc["month"] = str(m)
        coefs.append(rc)
        out({"section": "refit", "code": code, "month": str(m), "train_races": int(train["market_id"].nunique()),
             "train_rows": int(len(train)), "train_start_max": train["start"].max(),
             "test_races": int(d.loc[test_mask, "market_id"].nunique()),
             "full_iters": full.n_iter, "full_converged": full.converged, "full_train_loss": r5(full.train_loss),
             "prevwon_iters": prevwon.n_iter, "prevwon_converged": prevwon.converged,
             "coef_log_p_full": r5(rc["log_p"]), "coef_prev_won_prevwon": r5(prevwon.raw_coefs()["prev_won"]),
             "seconds_full": round(t1 - t0, 1), "seconds_total": round(time.time() - t0, 1)})
    co = pd.DataFrame(coefs)
    for f in feats:
        v = co[f].to_numpy()
        mean = v.mean()
        out({"section": "coef", "code": code, "feature": f, "refits": int(len(co)), "mean_raw_coef": r5(mean),
             "min": r5(v.min()), "max": r5(v.max()),
             "sign_agreement_share": r5((np.sign(v) == np.sign(mean)).mean()) if mean != 0 else None,
             "last_refit": r5(v[-1])})
    test = d[~np.isnan(p_full)].copy()
    test["p_full"] = p_full[~np.isnan(p_full)]
    test["p_prevwon"] = p_prev[~np.isnan(p_prev)]
    return test, co


def calibration(test: pd.DataFrame, code: str) -> None:
    dec = pd.qcut(test["p_full"], 10, labels=False, duplicates="drop")
    cal = test.groupby(dec).agg(n=("won", "size"), win_rate=("f", "mean"), p_model=("p_full", "mean"),
                                p_mkt=("p_mkt", "mean"))
    out({"section": "calibration", "kind": "p_model_deciles", "code": code,
         "rows": {str(int(k)): {"n": int(v["n"]), "p_model": r5(v["p_model"]), "p_mkt": r5(v["p_mkt"]),
                                "win_rate": r5(v["win_rate"])} for k, v in cal.iterrows()}})
    edge = test["p_full"] / test["p_mkt"]
    bins = pd.cut(edge, [0, 0.8, 0.9, 0.95, 1.0, 1.05, 1.10, 1.15, 1.3, 10])
    cal = test.groupby(bins, observed=True).agg(n=("won", "size"), win_rate=("f", "mean"), p_mkt=("p_mkt", "mean"),
                                                p_model=("p_full", "mean"))
    out({"section": "calibration", "kind": "edge_bins_p_full_over_p_mkt", "code": code,
         "rows": {str(k): {"n": int(v["n"]), "win_rate": r5(v["win_rate"]), "p_mkt": r5(v["p_mkt"]),
                           "p_model": r5(v["p_model"]), "won_over_expected": r5(v["win_rate"] / v["p_mkt"])}
                  for k, v in cal.iterrows()}})


def rule_cells(test: pd.DataFrame, mid: pd.Timestamp, cells: list[dict]) -> None:
    edge = (test["p_full"] / test["p_mkt"]).to_numpy()
    cheap = (test["ltp_pre"] <= MAX_PRICE).to_numpy()
    harness = test["harness"].to_numpy()
    for code, cmask in (("gallops", ~harness), ("harness", harness), ("pooled", np.ones(len(test), dtype=bool))):
        for family, ks in (("rules", K_VALUES), ("rules_exploratory", K_EXPLORATORY)):
            for k in ks:
                mask = (edge > 1 + k) & cheap & cmask
                g = test[mask]
                for settlement, _ in SETTLEMENTS:
                    cells.append(bet_cell(f"rule_{code}_k{k:.2f}", family, settlement, g, mid,
                                          {"rule": f"back when p_full/p_mkt > {1 + k:.2f} and ltp_pre <= {MAX_PRICE:g}", "k": k,
                                           "code": code, "pre_registered": family == "rules",
                                           "mean_p_model": r5(g["p_full"].mean()) if len(g) else None,
                                           "share_of_test_rows": r5(mask.sum() / cmask.sum()) if cmask.sum() else None}))


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="descriptive cells on 2019 only; walk-forward 2019-01..2019-03")
    ap.add_argument("--l2", type=float, default=None, help="override the frozen L2 choice (sensitivity runs only)")
    args = ap.parse_args()
    t_start = time.time()
    out({"section": "run", "experiment": "horse_market_memory", "dev": args.dev, "full_features": FULL_FEATURES,
         "prevwon_features": PREVWON_FEATURES, "commissions": COMMISSIONS, "bh_q": Q_BH,
         "expected_bet_cells": EXPECTED_CELLS, "max_price": MAX_PRICE, "k_values": K_VALUES})

    df, info = load()
    out(info)
    h, hinfo = add_history(df)
    out(hinfo)
    del df
    ev = evaluation_frame(h)
    del h

    cells: list[dict] = []
    descriptive(ev, args.dev, cells)
    n_desc = len(cells)

    md = model_frame(ev)
    del ev
    l2 = {}
    for code in CODES:
        l2[code] = choose_l2(md, code)
        if args.l2 is not None:
            out({"section": "l2", "code": code, "override": args.l2, "note": "sensitivity run: the frozen choice above is replaced"})
            l2[code] = args.l2

    months = pd.period_range(*(DEV_TEST_MONTHS if args.dev else TEST_MONTHS), freq="M")
    tests = {}
    for code in CODES:
        tests[code], _ = walk_forward(md, code, l2[code], months)
    pooled = pd.concat([tests["gallops"], tests["harness"]]).sort_values(["start", "market_id"]).reset_index(drop=True)
    mid_test = median_start(pooled)
    out({"section": "halves", "evaluation_set": "test period (walk-forward predictions, both codes)", "median_start": mid_test,
         "period": [pooled["month"].min(), pooled["month"].max()], "refits": int(len(months)),
         "races": int(pooled["market_id"].nunique()), "runners": int(len(pooled)),
         "gallops_races": int(tests["gallops"]["market_id"].nunique()), "harness_races": int(tests["harness"]["market_id"].nunique())})

    ll_cells = {}
    for code in CODES:
        t = tests[code]
        ll_cells[(code, "PREVWON")] = paired_gain(f"{code}_PREVWON", "MARKET", t, t["p_mkt"].to_numpy(), t["p_prevwon"].to_numpy(), mid_test)
        ll_cells[(code, "FULL")] = paired_gain(f"{code}_FULL", "MARKET", t, t["p_mkt"].to_numpy(), t["p_full"].to_numpy(), mid_test)
        ll_cells[(code, "FULL-PREVWON")] = paired_gain(f"{code}_FULL_minus_PREVWON", "PREVWON", t, t["p_prevwon"].to_numpy(),
                                                       t["p_full"].to_numpy(), mid_test)
        for c in ("PREVWON", "FULL", "FULL-PREVWON"):
            out(ll_cells[(code, c)])
        calibration(t, code)

    rule_cells(pooled, mid_test, cells)
    finalize(cells)
    pre = [c for c in cells if c["pre_registered"]]
    expl = [c for c in cells if not c["pre_registered"]]
    assert len(pre) == EXPECTED_CELLS and len(expl) == EXPECTED_EXPLORATORY
    out({"section": "bh", "pre_registered_cells": len(pre), "expected": EXPECTED_CELLS, "descriptive_cells": n_desc,
         "rule_cells": len(pre) - n_desc, "exploratory_cells_not_pre_registered": len(expl), "q": Q_BH,
         "bh_pass_pre_registered": int(sum(bool(c["bh_pass"]) for c in pre)),
         "bh_pass_all_72": int(sum(c["bh_pass_all"] for c in cells)),
         "both_halves_gt_2se": int(sum(c["both_halves_gt_2se"] for c in cells)),
         "positive": int(sum(c["positive"] for c in cells)),
         "positive_cells": [f"{c['cell']}@{c['settlement']}" for c in cells if c["positive"]],
         "bh_pass_cells": [f"{c['cell']}@{c['settlement']}:roi_c5={c['roi_c5']}" for c in cells if c["bh_pass"] or c["bh_pass_all"]]})
    for c in cells:
        out(c)
    for fam in ("desc_gallops", "desc_harness", "rules", "rules_exploratory"):
        opposite_half([c for c in cells if c["family"] == fam], fam)

    def find(name, settlement):
        return next((c for c in cells if c["cell"] == name and c["settlement"] == settlement), None)

    primary = find("rule_pooled_k0.10", "bsp")
    out({"section": "primary", "pre_registered": "pooled k=0.10, settlement bsp, c=0.05: > 2 SE in both halves and bh_pass",
         "cell": primary, "success": bool(primary and primary["positive"])})
    model_success = all(ll_cells[(code, "FULL")]["positive"] for code in CODES)
    out({"section": "primary", "pre_registered": "probability-model success: FULL log-loss gain vs MARKET > 2 SE in both halves in both codes "
         "(reported as such, no bankable claim)",
         "success": bool(model_success),
         "cells": {code: {k: ll_cells[(code, "FULL")][k] for k in ("gain", "se", "gain_h1", "se_h1", "gain_h2", "se_h2", "positive")}
                   for code in CODES}})
    for c in CAVEATS:
        out({"section": "caveat", "text": c})
    out({"section": "done", "seconds": round(time.time() - t_start, 1)})


if __name__ == "__main__":
    main()

"""Speed-figure model: predict each horse's next speed figure, then simulate the race.

Why: the conditional logit (model.py) must learn from complete races (every
runner + the winner). Past-performance files only show each past race from
the point of view of today's runners, so complete fields are rare. This model
needs only one horse's own history per training example:

    training example = one past start that has a speed figure
    target           = that start's figure
    inputs           = the horse's EARLIER starts + what was known pre-race
                       (distance, surface, purse, weight, jockey)

Prediction: ridge regression gives an expected figure mu_i per runner; its
uncertainty sigma_i is the residual spread for horses with similar experience
[A-28]. Win probability = share of SIMULATIONS races (figures drawn from
N(mu_i, sigma_i)) that horse i wins [A-29].

Explanations are exact: mu_i - race average = sum of per-feature
contributions w_j * (z_ij - mean_race z_j), in speed-figure points.
"""

from __future__ import annotations

import bisect
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .features import _WindowCounter, id_columns, won_label

FIG_FEATURES: dict[str, str] = {
    "last_fig": "Speed figure in most recent start with a figure",
    "avg3_fig": "Average of last 3 speed figures [A-01]",
    "best3_fig": "Best of last 3 speed figures [A-01]",
    "career_fig": "Career average speed figure",
    "fig_trend": "Last figure minus average of last 3 (improving > 0)",
    "log_starts": "log(1 + prior starts)",
    "log_layoff": "log(days since last start), capped [A-06]",
    "dist_change": "Today's distance minus last start's (furlongs)",
    "sprint_route_switch": "1 if switching between sprint and route [A-27]",
    "surface_switch": "1 if today's surface differs from last start's",
    "turf_today": "1 if today's race is on turf",
    "distance_fit": "Avg figure within ±1f of today's distance minus career avg [A-04]",
    "surface_fit": "Avg figure on today's surface minus career avg",
    "log_purse": "log(today's purse): overall class level",
    "class_change": "log(today's purse / recent average purse)",
    "last_finish_pct": "Last start finishing percentile, 1 = won",
    "last_beaten": "Lengths beaten last start (capped at 20)",
    "weight_change": "Weight today minus last start (lbs)",
    "jockey_win_rate": "Jockey win rate, trailing window, shrunk [A-02][A-03]",
    "age": "Age in years",
    "has_history": "1 if horse has any prior start in the data [A-07]",
}
FIG_FEATURE_NAMES = list(FIG_FEATURES)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def _nan(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _index(history: pd.DataFrame):
    horses: dict[str, list[dict]] = defaultdict(list)
    jockeys = _WindowCounter()
    for r in history.itertuples(index=False):
        if math.isnan(_nan(r.finish_position)):
            continue
        horses[r.horse_id].append({
            "date": r.date, "fig": _nan(r.speed_figure), "finish": _nan(r.finish_position),
            "field": _nan(r.field_size), "dist": _nan(r.distance_furlongs),
            "surface": r.surface if isinstance(r.surface, str) else "",
            "purse": _nan(r.purse), "weight": _nan(getattr(r, "weight_lbs", math.nan)),
            "beaten": _nan(getattr(r, "beaten_lengths", math.nan)),
        })
        jockeys.add(r.jockey_id, r.date, r.finish_position == 1)
    for lines in horses.values():
        lines.sort(key=lambda s: s["date"])
    jockeys.build()
    return horses, jockeys


def _mean(xs):
    return float(np.mean(xs)) if xs else math.nan


def _row_features(row, prior: list[dict], jockeys) -> dict:
    with_fig = [s for s in prior if not math.isnan(s["fig"])]
    figs = [s["fig"] for s in with_fig]
    recent = figs[-config.RECENT_STARTS:]
    career = _mean(figs)
    last = prior[-1] if prior else None
    dist, purse = _nan(row.distance_furlongs), _nan(row.purse)
    surface = row.surface if isinstance(row.surface, str) else ""

    def fit(match):
        sel = [s["fig"] for s in with_fig if match(s)]
        if math.isnan(career):
            return math.nan
        return _mean(sel) - career if sel else 0.0  # [A-17] no evidence = neutral

    purses = [s["purse"] for s in prior[-config.RECENT_STARTS:] if s["purse"] > 0]
    pct = math.nan
    if last and last["field"] > 1:
        pct = 1.0 - (last["finish"] - 1) / (last["field"] - 1)
    sprint = lambda d: d < config.SPRINT_MAX_FURLONGS  # noqa: E731  [A-27]
    return {
        "last_fig": figs[-1] if figs else math.nan,
        "avg3_fig": _mean(recent),
        "best3_fig": max(recent) if recent else math.nan,
        "career_fig": career,
        "fig_trend": figs[-1] - _mean(recent) if recent else math.nan,
        "log_starts": math.log1p(len(prior)),
        "log_layoff": (math.log(max(1, min((row.date - last["date"]).days, config.MAX_LAYOFF_DAYS)))
                       if last else math.nan),
        "dist_change": dist - last["dist"] if last and not math.isnan(last["dist"]) else math.nan,
        "sprint_route_switch": (float(sprint(dist) != sprint(last["dist"]))
                                if last and not math.isnan(dist) and not math.isnan(last["dist"]) else math.nan),
        "surface_switch": float(surface != last["surface"]) if last and surface and last["surface"] else math.nan,
        "turf_today": float(surface == "turf"),
        "distance_fit": fit(lambda s: abs(s["dist"] - dist) <= config.DISTANCE_TOLERANCE_FURLONGS),
        "surface_fit": fit(lambda s: s["surface"] == surface) if surface else math.nan,
        "log_purse": math.log(purse) if purse > 0 else math.nan,
        "class_change": math.log(purse / np.mean(purses)) if purses and purse > 0 else math.nan,
        "last_finish_pct": pct,
        "last_beaten": min(last["beaten"], 20.0) if last and not math.isnan(last["beaten"]) else math.nan,
        "weight_change": (_nan(row.weight_lbs) - last["weight"]) if last else math.nan,
        "jockey_win_rate": jockeys.rate(row.jockey_id, row.date),
        "age": _nan(row.age),
        "has_history": 1.0 if prior else 0.0,
        "n_prior": len(prior),
    }


def build_figure_features(history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time features for each target row (a past start or an entry).

    ``target_fig`` is the row's own speed figure (NaN for entries) and is the
    regression target. Only starts strictly before the row's date are used.
    """
    horses, jockeys = _index(history)
    out = []
    for row in targets.itertuples(index=False):
        lines = horses.get(row.horse_id, [])
        cut = bisect.bisect_left([s["date"] for s in lines], row.date)
        out.append({**id_columns(row), **_row_features(row, lines[:cut], jockeys),
                    "target_fig": _nan(row.speed_figure), "won": won_label(row)})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def _bucket(n_prior: np.ndarray) -> np.ndarray:
    """[A-28] Experience bucket index for each runner."""
    return np.searchsorted(np.array(config.FIGURE_SIGMA_BUCKETS), n_prior, side="left")


@dataclass
class SpeedFigureModel:
    feature_names: list[str] = field(default_factory=lambda: list(FIG_FEATURE_NAMES))
    weights: np.ndarray | None = None
    intercept: float = 0.0
    means: np.ndarray | None = None
    stds: np.ndarray | None = None
    impute_values: np.ndarray | None = None
    sigmas: list[float] | None = None
    n_train: int = 0
    name: str = "speed_figure"
    units: str = "speed-figure points"

    def _prepare(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        X = df[self.feature_names].to_numpy(dtype=float)
        if fit:
            # Training rows come from one data source, so the training mean is
            # on the right figure scale [A-30]; has_history marks debuts [A-07].
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-missing column -> 0
                self.impute_values = np.nan_to_num(np.nanmean(X, axis=0))
        X = np.where(np.isnan(X), self.impute_values, X)
        if fit:
            self.means = X.mean(axis=0)
            self.stds = X.std(axis=0)
            self.stds[self.stds == 0] = 1.0
        return (X - self.means) / self.stds

    def fit(self, feats: pd.DataFrame) -> "SpeedFigureModel":
        train = feats[feats["target_fig"].notna()]
        if len(train) < 20:
            raise ValueError(f"Only {len(train)} past starts with speed figures; need at least 20.")
        Z = self._prepare(train, fit=True)
        y = train["target_fig"].to_numpy(dtype=float)
        n = len(y)
        # [A-26] ridge: (Z'Z/n + lambda I) w = Z'(y - mean y)/n
        A = Z.T @ Z / n + config.FIGURE_RIDGE * np.eye(Z.shape[1])
        self.intercept = float(y.mean())
        self.weights = np.linalg.solve(A, Z.T @ (y - self.intercept) / n)
        resid = y - (self.intercept + Z @ self.weights)
        buckets = _bucket(train["n_prior"].to_numpy())
        overall = float(resid.std())
        self.sigmas = [float(resid[buckets == b].std()) if (buckets == b).sum() >= 10 else overall
                       for b in range(len(config.FIGURE_SIGMA_BUCKETS) + 1)]
        self.n_train = n
        return self

    def predict_fig(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Z = self._prepare(df, fit=False)
        mu = self.intercept + Z @ self.weights
        sigma = np.array(self.sigmas)[_bucket(df["n_prior"].to_numpy())]
        return mu, sigma

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """[A-29] Monte Carlo win probabilities, per race."""
        mu, sigma = self.predict_fig(df)
        probs = np.zeros(len(df))
        for _, idx in df.groupby(df["race_id"].to_numpy(), sort=False).indices.items():
            rng = np.random.default_rng(config.SIMULATION_SEED)
            draws = mu[idx] + sigma[idx] * rng.standard_normal((config.SIMULATIONS, len(idx)))
            wins = np.bincount(draws.argmax(axis=1), minlength=len(idx))
            probs[idx] = wins / config.SIMULATIONS
        return probs

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contribution to expected figure vs. the race average (points)."""
        Z = pd.DataFrame(self._prepare(df, fit=False), columns=self.feature_names)
        Z = Z - Z.groupby(df["race_id"].to_numpy()).transform("mean")
        return Z * self.weights

    @classmethod
    def from_dict(cls, d: dict) -> "SpeedFigureModel":
        return cls(feature_names=d["feature_names"], weights=np.array(d["weights"]),
                   intercept=d["intercept"], means=np.array(d["means"]), stds=np.array(d["stds"]),
                   impute_values=np.array(d["impute_values"]), sigmas=list(d["sigmas"]),
                   n_train=d.get("n_train", 0))

    def to_dict(self) -> dict:
        return {
            "type": self.name, "units": self.units, "feature_names": self.feature_names,
            "weights": self.weights.tolist(), "intercept": self.intercept,
            "means": self.means.tolist(), "stds": self.stds.tolist(),
            "impute_values": self.impute_values.tolist(), "sigmas": self.sigmas,
            "sigma_buckets": list(config.FIGURE_SIGMA_BUCKETS), "n_train": self.n_train,
            "feature_descriptions": FIG_FEATURES,
        }


# --------------------------------------------------------------------------
# Training + evaluation
# --------------------------------------------------------------------------
def figure_metrics(model: SpeedFigureModel, test: pd.DataFrame) -> dict:
    """How well the model predicts figures on unseen starts, vs. naive baselines."""
    t = test[test["target_fig"].notna()]
    if t.empty:
        return {"starts": 0}
    mu, _ = model.predict_fig(t)
    y = t["target_fig"].to_numpy()
    base = t["last_fig"].notna().to_numpy()

    def err(pred, mask):
        e = (pred - y)[mask]
        return {"mae": round(float(np.abs(e).mean()), 2), "rmse": round(float(np.sqrt((e ** 2).mean())), 2)}

    return {
        "starts": int(len(t)), "starts_with_prior_figure": int(base.sum()),
        "model": err(mu, np.ones(len(t), bool)),
        "model_on_horses_with_form": err(mu, base),
        "baseline_last_figure": err(t["last_fig"].to_numpy(), base),
        "baseline_avg_last3": err(t["avg3_fig"].to_numpy(), base),
        "correlation": round(float(np.corrcoef(mu, y)[0, 1]), 3) if len(t) > 2 else None,
    }


def fit_with_holdout(history: pd.DataFrame, test_fraction: float = config.TEST_FRACTION):
    """Fit on all starts with figures, and report accuracy on the latest
    ``test_fraction`` of them using a model fitted on the earlier ones [A-08].

    Returns (model fitted on everything, metrics dict).
    """
    starts = history[history["speed_figure"].notna() & history["finish_position"].notna()]
    feats = build_figure_features(history, starts)
    dates = feats["date"].sort_values()
    cutoff = dates.iloc[int(len(dates) * (1 - test_fraction))] if len(dates) else None
    metrics = {"train_starts": int(len(feats))}
    early, late = feats[feats["date"] < cutoff], feats[feats["date"] >= cutoff]
    if len(early) >= 20 and len(late):
        metrics["holdout"] = {"from": str(cutoff.date()),
                              **figure_metrics(SpeedFigureModel().fit(early), late)}
    return SpeedFigureModel().fit(feats), metrics

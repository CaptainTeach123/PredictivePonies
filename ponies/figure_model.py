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
    "abroad_recent": "Share of last 3 starts run abroad (figures not used) [A-36]",
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
            "abroad": _nan(getattr(r, "abroad", 0.0)) == 1.0,
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
        "abroad_recent": (float(np.mean([s["abroad"] for s in prior[-config.RECENT_STARTS:]]))
                          if prior else 0.0),
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
    temperature: float = 1.0  # [A-34] multiplies sigma
    blend: tuple[float, float] | None = None  # [A-37] (a, b) for model + market
    n_train: int = 0
    name: str = "speed_figure"
    units: str = "speed-figure points"

    def _prepare(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        X = df[self.feature_names].to_numpy(dtype=float)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-missing column -> 0
            if fit:
                self.impute_values = np.nan_to_num(np.nanmean(X, axis=0))
            # [A-07] missing = median of the runner's own race field (a debut
            # horse or an import is "average for this race", whatever the
            # figure level of that race); training mean if the field has none.
            # has_history / abroad_recent let the model adjust from there.
            field_median = (pd.DataFrame(X).groupby(df["race_id"].to_numpy())
                            .transform("median").to_numpy())
        X = np.where(np.isnan(X), field_median, X)
        X = np.where(np.isnan(X), self.impute_values, X)
        if fit:
            self.means = X.mean(axis=0)
            self.stds = X.std(axis=0)
            self.stds[self.stds == 0] = 1.0
        # [A-33] cap extreme inputs
        return np.clip((X - self.means) / self.stds, -config.FEATURE_CLIP_SD, config.FEATURE_CLIP_SD)

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
        sigma = np.array(self.sigmas)[_bucket(df["n_prior"].to_numpy())] * self.temperature
        return mu, sigma

    def predict_proba(self, df: pd.DataFrame, market_col: str = config.BLEND_MARKET_COLUMN) -> np.ndarray:
        """Win probabilities: the model blended with market odds when blend
        weights were learned [A-37]; otherwise the model alone [A-29]."""
        if self.blend is None or market_col not in df:
            return self.predict_model_only(df)
        mu, _ = self.predict_fig(df)
        logq = _log_market(df, market_col)
        model_only = None
        probs = np.zeros(len(df))
        a, b = self.blend
        for _, idx in df.groupby(df["race_id"].to_numpy(), sort=False).indices.items():
            lq = logq[idx]
            if np.isnan(lq).all():  # no odds for this race: model alone
                model_only = self.predict_model_only(df) if model_only is None else model_only
                probs[idx] = model_only[idx]
                continue
            lq = np.where(np.isnan(lq), np.nanmedian(lq), lq)  # [A-38] missing odds = race median
            u = a * mu[idx] + b * lq
            e = np.exp(u - u.max())
            probs[idx] = e / e.sum()
        return probs

    def predict_model_only(self, df: pd.DataFrame) -> np.ndarray:
        """[A-29] Monte Carlo win probabilities from the model alone, per race."""
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
                   temperature=d.get("temperature", 1.0),
                   blend=tuple(d["blend"]) if d.get("blend") else None, n_train=d.get("n_train", 0))

    def to_dict(self) -> dict:
        return {
            "type": self.name, "units": self.units, "feature_names": self.feature_names,
            "weights": self.weights.tolist(), "intercept": self.intercept,
            "means": self.means.tolist(), "stds": self.stds.tolist(),
            "impute_values": self.impute_values.tolist(), "sigmas": self.sigmas,
            "temperature": self.temperature, "blend": list(self.blend) if self.blend else None,
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


def _log_market(df: pd.DataFrame, col: str) -> np.ndarray:
    """log(implied win probability) up to a per-race constant: -log(decimal odds).
    The constant cancels in within-race comparisons. NaN where odds are missing."""
    from .odds import to_decimal

    dec = df[col].map(to_decimal).to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        return np.where(dec > 1, -np.log(dec), np.nan)


def _winner_pairs(t: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(winner, other) pairs: the race winner vs each other runner seen from
    that race. This is the "who wins" question that win betting depends on."""
    ahead, behind = _race_pairs(t)
    keep = t["finish_position"].to_numpy(dtype=float)[ahead] == 1
    return ahead[keep], behind[keep]


def _blend_design(model: "SpeedFigureModel", t: pd.DataFrame):
    """[A-37] Winner pairs with odds on both sides:
    X = [mu_winner - mu_other, logq_winner - logq_other]."""
    t = t[t["finish_position"].notna()].reset_index(drop=True)
    ahead, behind = _winner_pairs(t)
    if len(ahead) == 0:
        return np.zeros((0, 2))
    mu, _ = model.predict_fig(t)
    lq = _log_market(t, "final_odds")
    ok = ~(np.isnan(lq[ahead]) | np.isnan(lq[behind]))
    return np.column_stack([mu[ahead] - mu[behind], lq[ahead] - lq[behind]])[ok]


def fit_blend(X: np.ndarray) -> tuple[float, float] | None:
    """[A-37] Logistic regression without intercept on pair differences:
    P(ahead beats behind) = sigmoid(a * d_mu + b * d_logq). Newton's method."""
    if len(X) < config.MIN_CALIBRATION_PAIRS:
        return None
    w = np.array([0.0, 1.0])  # start at "market only"
    lam = 1e-3 * len(X)
    for _ in range(50):
        p = 1 / (1 + np.exp(-(X @ w)))
        grad = X.T @ (1 - p) - lam * w
        H = -(X.T * (p * (1 - p))) @ X - lam * np.eye(2)
        step = np.linalg.solve(H, grad)
        w = w - step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[0]), float(w[1])


def _race_pairs(t: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(ahead, behind) row indices for every pair of runners from the same race
    with different finishing positions."""
    fin = t["finish_position"].to_numpy(dtype=float)
    ahead, behind = [], []
    for _, idx in t.groupby("race_id").indices.items():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if fin[i] != fin[j]:
                    ahead.append(i if fin[i] < fin[j] else j)
                    behind.append(j if fin[i] < fin[j] else i)
    return np.array(ahead, dtype=int), np.array(behind, dtype=int)


def _phi(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def _pair_log_loss(mu, sigma, ahead, behind) -> float:
    p = _phi((mu[ahead] - mu[behind]) / np.sqrt(sigma[ahead] ** 2 + sigma[behind] ** 2))
    return float(-np.mean(np.log(np.clip(p, 1e-6, 1 - 1e-6))))


def head_to_head(model: "SpeedFigureModel", test: pd.DataFrame) -> dict:
    """Pairwise check against real outcomes and the betting market.

    For every pair of runners from the same unseen past race (past-performance
    files show several runners of many races), did the model rank the horse
    that finished ahead higher? Same question for the market (lower final
    odds) and for "average of last 3 figures". Plus pairwise log loss of
    P(ahead beats behind) = Phi((mu_a - mu_b) / sqrt(sigma_a^2 + sigma_b^2))
    vs the market's p_a / (p_a + p_b). [A-32] ln 2 = 0.693 is a coin flip.
    """
    from .odds import to_decimal

    t = test[test["finish_position"].notna()].reset_index(drop=True)
    ahead, behind = _race_pairs(t)
    if len(ahead) == 0:
        return {"pairs": 0}
    mu, sigma = model.predict_fig(t)
    dec = t["final_odds"].map(to_decimal).to_numpy(dtype=float)
    avg3 = t["avg3_fig"].to_numpy(dtype=float)
    mk = (dec[ahead] > 1) & (dec[behind] > 1)
    q = (1 / dec[ahead][mk]) / (1 / dec[ahead][mk] + 1 / dec[behind][mk])
    bl = ~(np.isnan(avg3[ahead]) | np.isnan(avg3[behind]))
    blend = {}
    lq = _log_market(t, "final_odds")
    if model.blend is not None and mk.any():
        a, b = model.blend
        z = a * (mu[ahead][mk] - mu[behind][mk]) + b * (lq[ahead][mk] - lq[behind][mk])
        blend = {"blend_correct": round(float(np.mean(z > 0)), 4),
                 "blend_pair_log_loss": round(float(np.mean(np.log1p(np.exp(-z)))), 4)}
    # [A-35] the "who wins" test: winner vs each other runner, odds on both sides
    wa, wb = _winner_pairs(t)
    wok = ~(np.isnan(lq[wa]) | np.isnan(lq[wb])) if len(wa) else np.zeros(0, bool)
    wa, wb = wa[wok], wb[wok]
    if len(wa):
        market_loss = np.log1p(np.exp(-(lq[wa] - lq[wb])))
        if model.blend is not None:
            a, b = model.blend
            ours = np.log1p(np.exp(-(a * (mu[wa] - mu[wb]) + b * (lq[wa] - lq[wb]))))
        else:
            p_ = _phi((mu[wa] - mu[wb]) / np.sqrt(sigma[wa] ** 2 + sigma[wb] ** 2))
            ours = -np.log(np.clip(p_, 1e-6, 1 - 1e-6))
        diff = market_loss - ours  # > 0 = we beat the market
        blend.update({
            "win_pairs": int(len(wa)),
            "market_win_pair_log_loss": round(float(market_loss.mean()), 4),
            "ours_win_pair_log_loss": round(float(ours.mean()), 4),
            "win_improvement": round(float(diff.mean()), 5),
            "win_improvement_se": round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 5) if len(diff) > 1 else None,
        })
    return {
        **blend,
        "pairs": int(len(ahead)),
        "model_correct": round(float(np.mean(mu[ahead] > mu[behind])), 4),
        "avg3_correct": round(float(np.mean(avg3[ahead][bl] > avg3[behind][bl])), 4) if bl.any() else None,
        "market_correct": round(float(np.mean(dec[ahead][mk] < dec[behind][mk])), 4) if mk.any() else None,
        "market_pairs": int(mk.sum()),
        "model_pair_log_loss": round(_pair_log_loss(mu, sigma, ahead, behind), 4),
        "market_pair_log_loss": round(float(-np.mean(np.log(q))), 4) if mk.any() else None,
        "coin_flip_log_loss": round(math.log(2), 4),
    }


def calibrate_temperature(model: "SpeedFigureModel", test: pd.DataFrame) -> float | None:
    """[A-34] Temperature that minimises head-to-head log loss on ``test``."""
    t = test[test["finish_position"].notna()].reset_index(drop=True)
    ahead, behind = _race_pairs(t)
    if len(ahead) < config.MIN_CALIBRATION_PAIRS:
        return None
    saved = model.temperature
    model.temperature = 1.0
    mu, sigma = model.predict_fig(t)
    model.temperature = saved
    return min(config.TEMPERATURE_GRID, key=lambda T: _pair_log_loss(mu, T * sigma, ahead, behind))


def fit_with_holdout(history: pd.DataFrame, test_fraction: float = config.TEST_FRACTION):
    """Fit the speed-figure model and report how it does on unseen starts.

    Three chronological slices [A-08][A-34][A-37]:
      A = oldest starts                   -> train a model
      B = next ``test_fraction``          -> tune temperature + blend weights on A's predictions
      C = newest ``test_fraction``        -> score a model trained on A+B with those settings
    The final model is trained on everything with the same settings.

    Returns (final model, metrics dict).
    """
    starts = history[history["speed_figure"].notna() & history["finish_position"].notna()]
    if len(starts) < 20:
        raise ValueError(f"Only {len(starts)} past starts with speed figures; need at least 20.")
    feats = build_figure_features(history, starts)
    dates = feats["date"].sort_values().reset_index(drop=True)
    cut_b = dates.iloc[int(len(dates) * (1 - 2 * test_fraction))]
    cut_c = dates.iloc[int(len(dates) * (1 - test_fraction))]
    A, B, C = feats[feats["date"] < cut_b], feats[(feats["date"] >= cut_b) & (feats["date"] < cut_c)], \
        feats[feats["date"] >= cut_c]
    metrics = {"train_starts": int(len(feats))}
    T, blend = None, None
    if len(A) >= 20 and len(B) and len(C):
        model_a = SpeedFigureModel().fit(A)
        T = calibrate_temperature(model_a, B)
        model_a.temperature = T or 1.0
        blend = fit_blend(_blend_design(model_a, B))
        model_ab = SpeedFigureModel().fit(feats[feats["date"] < cut_c])
        raw = head_to_head(model_ab, C)
        model_ab.temperature, model_ab.blend = T or 1.0, blend
        metrics["holdout"] = {"from": str(cut_c.date()), **figure_metrics(model_ab, C),
                              "head_to_head": raw, "head_to_head_calibrated": head_to_head(model_ab, C)}
        if T is not None:
            metrics["temperature"] = T
        if blend is not None:
            metrics["blend_weights"] = {"a_per_figure_point": round(blend[0], 4), "b_market": round(blend[1], 4)}
    final = SpeedFigureModel().fit(feats)
    final.temperature, final.blend = T or 1.0, blend
    return final, metrics

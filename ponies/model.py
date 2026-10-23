"""Stage 3: conditional-logit win model.

Each runner i in race r gets a score  u_i = w · z_i  where z_i are the
standardised features. Win probability is a softmax within the race:

    P(i wins r) = exp(u_i) / sum_{k in r} exp(u_k)

This is the classic Bolton & Chapman (1986) / Benter (1994) approach. It
guarantees probabilities sum to 1 per race and makes explanations exact: a
runner's score relative to the race average decomposes into per-feature
contributions  w_j * (z_ij - mean_r z_j).

Fitting minimises mean per-race log loss + L2 penalty [A-09] with full-batch
gradient descent [A-10]. No external ML library is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _race_codes(race_ids) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(pd.Series(race_ids), sort=False)
    return codes, len(uniques)


def race_softmax(scores: np.ndarray, codes: np.ndarray, n_races: int) -> np.ndarray:
    """Softmax of ``scores`` within each race group (numerically stable)."""
    maxes = np.full(n_races, -np.inf)
    np.maximum.at(maxes, codes, scores)
    e = np.exp(scores - maxes[codes])
    denom = np.bincount(codes, weights=e, minlength=n_races)
    return e / denom[codes]


def win_targets(won: np.ndarray, codes: np.ndarray, n_races: int) -> np.ndarray:
    """[A-16] Dead heats: each co-winner gets an equal share of the win."""
    won = np.nan_to_num(np.asarray(won, dtype=float))
    counts = np.bincount(codes, weights=won, minlength=n_races)
    return np.divide(won, counts[codes], out=np.zeros_like(won), where=counts[codes] > 0)


@dataclass
class ConditionalLogit:
    feature_names: list[str]
    weights: np.ndarray | None = None
    impute_values: np.ndarray | None = None
    means: np.ndarray | None = None
    stds: np.ndarray | None = None
    history: list[float] = field(default_factory=list)

    # ----- preprocessing ---------------------------------------------------
    def _prepare(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        X = df[self.feature_names].to_numpy(dtype=float)
        if fit:
            # [A-07] impute with training mean
            self.impute_values = np.nan_to_num(np.nanmean(X, axis=0))
        X = np.where(np.isnan(X), self.impute_values, X)
        if fit:
            self.means = X.mean(axis=0)
            self.stds = X.std(axis=0)
            self.stds[self.stds == 0] = 1.0
        return (X - self.means) / self.stds

    # ----- training ----------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "ConditionalLogit":
        codes, n_races = _race_codes(df["race_id"])
        y = win_targets(df["won"].to_numpy(), codes, n_races)
        Z = self._prepare(df, fit=True)

        def loss_and_grad(w):
            p = race_softmax(Z @ w, codes, n_races)
            loss = (-np.sum(y * np.log(p + 1e-15)) / n_races
                    + 0.5 * config.L2_PENALTY * w @ w)
            grad = Z.T @ (p - y) / n_races + config.L2_PENALTY * w
            return loss, grad

        w = np.zeros(Z.shape[1])
        loss, grad = loss_and_grad(w)
        self.history = [float(loss)]
        lr = config.LEARNING_RATE
        for _ in range(config.MAX_ITERATIONS):
            w_new = w - lr * grad
            loss_new, grad_new = loss_and_grad(w_new)
            if loss_new > loss:      # overshoot -> halve step and retry
                lr *= 0.5
                if lr < 1e-10:
                    break
                continue
            improvement = loss - loss_new
            w, loss, grad = w_new, loss_new, grad_new
            self.history.append(float(loss))
            if improvement < config.TOLERANCE:
                break
        self.weights = w
        return self

    # ----- inference ---------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        codes, n_races = _race_codes(df["race_id"])
        Z = self._prepare(df, fit=False)
        return race_softmax(Z @ self.weights, codes, n_races)

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contribution to each runner's score vs. the race average.

        Positive = this feature makes the horse more likely to win than an
        average rival in the same race. Row sums equal u_i - mean_r(u).
        """
        Z = self._prepare(df, fit=False)
        Zc = pd.DataFrame(Z, columns=self.feature_names)
        Zc = Zc - Zc.groupby(df["race_id"].to_numpy()).transform("mean")
        return Zc * self.weights

    # ----- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "type": "conditional_logit",
            "feature_names": self.feature_names,
            "weights": self.weights.tolist(),
            "impute_values": self.impute_values.tolist(),
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "final_train_loss": self.history[-1] if self.history else None,
            "iterations": len(self.history),
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ConditionalLogit":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, d: dict) -> "ConditionalLogit":
        return cls(
            feature_names=d["feature_names"],
            weights=np.array(d["weights"]),
            impute_values=np.array(d["impute_values"]),
            means=np.array(d["means"]),
            stds=np.array(d["stds"]),
        )

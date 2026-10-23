"""Backtest: compare model predictions with what actually happened.

All metrics are computed on the held-out (most recent) races only [A-08].
Baselines:
  * uniform  - every runner equally likely (1 / field size)
  * market   - implied probabilities from FINAL odds [A-12]; a strong
               benchmark, because the betting public is well calibrated.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config, odds
from .model import _race_codes, win_targets


def _race_log_loss(p: np.ndarray, y: np.ndarray, codes: np.ndarray, n: int) -> float:
    """Mean over races of -sum(y * log p)  (= -log P(winner) without dead heats)."""
    per_race = np.bincount(codes, weights=-y * np.log(np.clip(p, 1e-15, 1)), minlength=n)
    return float(per_race.mean())


def market_probabilities(df: pd.DataFrame, column: str = "final_odds") -> np.ndarray:
    dec = df[column].map(odds.to_decimal).to_numpy(dtype=float)
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby("race_id").indices.items():
        out[idx] = odds.implied_probabilities(dec[idx])
    return out


def calibration_table(p: np.ndarray, won: np.ndarray) -> list[dict]:
    """[A-15] Equal-width probability bins: predicted vs. observed win rate."""
    bins = np.linspace(0, 1, config.CALIBRATION_BINS + 1)
    which = np.clip(np.digitize(p, bins) - 1, 0, config.CALIBRATION_BINS - 1)
    rows = []
    for b in range(config.CALIBRATION_BINS):
        m = which == b
        if m.sum() == 0:
            continue
        rows.append({
            "bin_low": float(bins[b]), "bin_high": float(bins[b + 1]),
            "mean_predicted": float(p[m].mean()),
            "observed_win_rate": float(won[m].mean()),
            "count": int(m.sum()),
        })
    return rows


def _roi(stakes_mask: np.ndarray, won: np.ndarray, dec: np.ndarray) -> dict:
    """[A-14] Flat-stake return at final odds."""
    m = stakes_mask & ~np.isnan(dec)
    n = int(m.sum())
    if n == 0:
        return {"bets": 0, "wins": 0, "profit": 0.0, "roi": None}
    stake = config.FLAT_STAKE
    returns = np.where(won[m] == 1, dec[m] * stake, 0.0)
    profit = float(returns.sum() - n * stake)
    return {"bets": n, "wins": int((won[m] == 1).sum()),
            "profit": round(profit, 2), "roi": round(profit / (n * stake), 4)}


def evaluate(df: pd.DataFrame, p: np.ndarray) -> dict:
    """``df`` = test-set features (needs race_id, won, final_odds); ``p`` = model probs."""
    df = df.reset_index(drop=True)
    codes, n = _race_codes(df["race_id"])
    won = np.nan_to_num(df["won"].to_numpy(dtype=float))
    y = win_targets(won, codes, n)
    field = np.bincount(codes, minlength=n)[codes]

    market = market_probabilities(df)
    has_market = df.assign(m=market).groupby("race_id")["m"].transform(lambda s: s.notna().all()).to_numpy()
    mcodes, mn = _race_codes(df.loc[has_market, "race_id"])

    dec = df["final_odds"].map(odds.to_decimal).to_numpy(dtype=float)
    top_model = pd.Series(p).groupby(codes).transform("max").to_numpy() == p
    top_market = (pd.Series(market).groupby(codes).transform("max").to_numpy() == market)
    edge = p * dec - 1

    ll_model = _race_log_loss(p, y, codes, n)
    ll_uniform = _race_log_loss(1.0 / field, y, codes, n)
    result = {
        "races": int(n),
        "runners": int(len(df)),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "log_loss": {
            "model": round(ll_model, 4),
            "uniform": round(ll_uniform, 4),
            "market": (round(_race_log_loss(market[has_market], y[has_market], mcodes, mn), 4)
                       if mn else None),
            "model_on_market_races": (round(_race_log_loss(p[has_market], y[has_market], mcodes, mn), 4)
                                      if mn else None),
        },
        "mcfadden_r2_vs_uniform": round(1 - ll_model / ll_uniform, 4) if ll_uniform else None,
        "brier": round(float(np.mean((p - won) ** 2)), 5),
        "top_pick_win_rate": {
            "model": round(float(won[top_model].sum() / n), 4),
            "market_favourite": round(float(won[top_market & has_market].sum() / max(mn, 1)), 4),
        },
        "betting_backtest": {
            "model_top_pick": _roi(top_model, won, dec),
            "value_bets": {**_roi(edge > config.VALUE_EDGE_THRESHOLD, won, dec),
                           "edge_threshold": config.VALUE_EDGE_THRESHOLD},
        },
        "calibration": calibration_table(p, won),
    }
    return result

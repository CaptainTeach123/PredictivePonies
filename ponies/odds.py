"""Odds conversions. Odds strings follow US "odds-to-one" convention.

Accepted formats (see DATA_SCHEMA.md):
    "5-2", "5/2"  -> fractional, pays 5 for every 2 staked
    "7"           -> 7-1
    "EVEN", "EVN" -> 1-1
"""

from __future__ import annotations

import math

import numpy as np

from . import config


def to_decimal(odds: str | float | None) -> float:
    """Return decimal odds (total return per 1 unit staked, incl. stake).

    Returns NaN for missing/unparseable values.
    """
    if odds is None:
        return math.nan
    if isinstance(odds, (int, float)):
        return float(odds) + 1.0 if not math.isnan(odds) else math.nan
    s = str(odds).strip().upper()
    if not s:
        return math.nan
    if s in {"EVEN", "EVN", "EVENS"}:
        return 2.0
    for sep in ("-", "/"):
        if sep in s:
            num, den = s.split(sep, 1)
            try:
                return float(num) / float(den) + 1.0
            except (ValueError, ZeroDivisionError):
                return math.nan
    try:
        return float(s) + 1.0
    except ValueError:
        return math.nan


def implied_probabilities(decimal_odds: np.ndarray) -> np.ndarray:
    """Implied win probabilities for one race's runners.

    [A-12] If NORMALISE_IMPLIED_PROBS, probabilities are rescaled to sum to 1
    (removes the takeout/overround). Runners with missing odds get NaN and are
    excluded from the normalisation.
    """
    raw = 1.0 / np.asarray(decimal_odds, dtype=float)
    if config.NORMALISE_IMPLIED_PROBS:
        total = np.nansum(raw)
        if total > 0:
            raw = raw / total
    return raw


def fair_odds_string(prob: float) -> str:
    """Model probability -> 'fair' US odds string, e.g. 0.25 -> '3-1'."""
    if not prob or prob <= 0:
        return "-"
    to_one = 1.0 / prob - 1.0
    if to_one < 1:
        return f"{to_one:.2f}-1"
    return f"{to_one:.1f}-1"


def shin_probabilities(decimal_odds: np.ndarray) -> np.ndarray:
    """[A-51] Shin (1993) de-margining: assumes part of the money comes from
    insiders, so the bookmaker's margin falls more heavily on longshots.
    p_i = (sqrt(z^2 + 4(1-z) pi_i^2 / S) - z) / (2(1-z)), pi_i = 1/odds,
    S = sum(pi), with z in [0, 0.5) solved by bisection so sum(p) = 1."""
    pi = 1.0 / np.asarray(decimal_odds, dtype=float)
    s = pi.sum()
    if s <= 1.0:
        return pi / s  # no margin to remove (or an over-broke book)

    def probs(z):
        return (np.sqrt(z * z + 4 * (1 - z) * pi * pi / s) - z) / (2 * (1 - z))
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if probs(mid).sum() > 1:
            lo = mid
        else:
            hi = mid
    p = probs((lo + hi) / 2)
    return p / p.sum()


def power_probabilities(decimal_odds: np.ndarray) -> np.ndarray:
    """[A-51] Power de-margining: p_i = pi_i^k with k >= 1 chosen so the
    probabilities sum to 1 (shrinks longshots more than favourites)."""
    pi = 1.0 / np.asarray(decimal_odds, dtype=float)
    if pi.sum() <= 1.0:
        return pi / pi.sum()
    lo, hi = 1.0, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (pi ** mid).sum() > 1:
            lo = mid
        else:
            hi = mid
    p = pi ** ((lo + hi) / 2)
    return p / p.sum()

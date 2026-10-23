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

"""Finishing-order probabilities for bets beyond the winner (place,
quinella, exacta, trifecta), from win probabilities.

Used by scripts/formfav_bets.py, which also models the expected tote dividend
of each combination [A-50] from past winning dividends.
"""

from __future__ import annotations

import math

import numpy as np

LAMBDA_GRID = np.round(np.arange(0.5, 1.31, 0.05), 2)


def order_probs(p: np.ndarray, lam2: float, lam3: float) -> dict:
    """[A-49] Harville with discounts (Lo & Bacon-Shone 1994):
    P(k 2nd | j 1st) ∝ p_k^lam2, P(i 3rd | j, k) ∝ p_i^lam3."""
    n = len(p)
    a, b = p ** lam2, p ** lam3
    eye = np.eye(n, dtype=bool)
    ex = p[:, None] * a[None, :] / np.maximum(a.sum() - a, 1e-300)[:, None]
    ex[eye] = 0.0
    out = {"win": p, "exacta": ex}
    if n >= 3:
        denom = b.sum() - b[:, None] - b[None, :]
        tri = ex[:, :, None] * b[None, None, :] / np.maximum(denom, 1e-300)[:, :, None]
        idx = np.arange(n)
        tri[idx, idx, :] = 0
        tri[idx, :, idx] = 0
        tri[:, idx, idx] = 0
        out["trifecta"] = tri
        out["top3"] = p + ex.sum(axis=0) + tri.sum(axis=(0, 1))
    out["top2"] = p + ex.sum(axis=0)
    return out


def fit_lambdas(races: list[tuple[np.ndarray, list[int]]]) -> tuple[float, float]:
    """Grid-search lam2, lam3 to maximise the likelihood of the observed 2nd
    and 3rd finishers given the 1st (and 2nd)."""
    def ll2(lam):
        s = 0.0
        for p, order in races:
            if len(order) < 2:
                continue
            a = p ** lam
            s += math.log(a[order[1]] / (a.sum() - a[order[0]]))
        return s

    def ll3(lam):
        s = 0.0
        for p, order in races:
            if len(order) < 3 or len(p) < 3:
                continue
            b = p ** lam
            s += math.log(b[order[2]] / (b.sum() - b[order[0]] - b[order[1]]))
        return s
    return (float(max(LAMBDA_GRID, key=ll2)), float(max(LAMBDA_GRID, key=ll3)))


def places_paid(field: int) -> int:
    """[A-50] TAB place terms: 3 places with 8+ starters, 2 with 5-7, none below."""
    return 3 if field >= 8 else 2 if field >= 5 else 0

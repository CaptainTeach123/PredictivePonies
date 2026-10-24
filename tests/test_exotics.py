"""Finishing-order probabilities [A-49] and place terms [A-50]."""
import numpy as np

from ponies.exotics import fit_lambdas, order_probs, places_paid


def test_order_probabilities_are_consistent():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    for lam in ((1.0, 1.0), (0.8, 0.7)):
        o = order_probs(p, *lam)
        assert abs(o["exacta"].sum() - 1) < 1e-12 and abs(o["trifecta"].sum() - 1) < 1e-12
        assert abs(o["top2"].sum() - 2) < 1e-12 and abs(o["top3"].sum() - 3) < 1e-12
        assert np.allclose(o["trifecta"].sum(axis=2), o["exacta"])
    # Harville (lam = 1): P(1 then 2) = 0.4 * 0.3 / 0.6
    assert abs(order_probs(p, 1, 1)["exacta"][0, 1] - 0.2) < 1e-12


def test_lambda_fit_and_place_terms():
    rng = np.random.default_rng(0)
    races = []
    for _ in range(300):
        p = rng.dirichlet(np.ones(8))
        order = list(rng.choice(8, size=3, replace=False, p=p))  # roughly Harville
        races.append((p, order))
    l2, l3 = fit_lambdas(races)
    assert 0.5 <= l2 <= 1.3 and 0.5 <= l3 <= 1.3
    assert (places_paid(8), places_paid(7), places_paid(5), places_paid(4)) == (3, 2, 2, 0)

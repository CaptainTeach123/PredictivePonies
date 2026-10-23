import math

import numpy as np

from ponies import odds


def test_to_decimal_formats():
    assert odds.to_decimal("5-2") == 3.5
    assert odds.to_decimal("7/2") == 4.5
    assert odds.to_decimal("7") == 8.0
    assert odds.to_decimal("EVEN") == 2.0
    assert math.isnan(odds.to_decimal(None))
    assert math.isnan(odds.to_decimal("scr"))


def test_implied_probabilities_normalised():
    p = odds.implied_probabilities(np.array([2.0, 3.0, 5.0]))
    assert abs(p.sum() - 1) < 1e-12
    assert p[0] > p[1] > p[2]

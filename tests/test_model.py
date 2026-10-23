import numpy as np
import pandas as pd

from ponies.model import ConditionalLogit, race_softmax, win_targets


def test_softmax_sums_to_one_per_race():
    codes = np.array([0, 0, 0, 1, 1])
    p = race_softmax(np.array([1.0, 2.0, 3.0, 100.0, 100.0]), codes, 2)
    assert np.allclose(np.bincount(codes, weights=p), 1.0)
    assert np.allclose(p[3:], 0.5)


def test_dead_heat_targets_split():
    codes = np.array([0, 0, 0])
    assert np.allclose(win_targets(np.array([1.0, 1.0, 0.0]), codes, 1), [0.5, 0.5, 0])


def test_fit_recovers_signal():
    rng = np.random.default_rng(0)
    rows = []
    for r in range(600):
        x = rng.normal(size=8)
        noise = rng.normal(size=8)
        u = 1.5 * x
        p = np.exp(u) / np.exp(u).sum()
        winner = rng.choice(8, p=p)
        for i in range(8):
            rows.append({"race_id": f"r{r}", "good": x[i], "noise": noise[i], "won": float(i == winner)})
    df = pd.DataFrame(rows)
    m = ConditionalLogit(["good", "noise"]).fit(df)
    w_good, w_noise = m.weights / m.stds   # back to raw-feature scale
    assert 1.1 < w_good < 1.9
    assert abs(w_noise) < 0.3
    contrib = m.contributions(df)
    assert np.allclose(contrib.groupby(df["race_id"]).sum().to_numpy(), 0, atol=1e-9)

import numpy as np
import pandas as pd

from ponies.figure_model import SpeedFigureModel, build_figure_features
from ponies.xml_io import START_COLUMNS


def _start(horse, date, fig, race="r", finish=1):
    row = dict.fromkeys(START_COLUMNS)
    row.update(race_id=f"{race}-{date}", date=pd.Timestamp(date), track="T", race_number=1,
               distance_furlongs=6.0, surface="dirt", condition="fast", purse=20000.0, field_size=8,
               horse_id=horse, horse_name=horse, post_position=1.0, weight_lbs=120.0, age=4.0,
               jockey_id="J", finish_position=float(finish), speed_figure=float(fig), source="result")
    return row


def test_features_use_only_earlier_starts():
    hist = pd.DataFrame([_start("A", "2026-01-01", 70), _start("A", "2026-02-01", 80),
                         _start("A", "2026-03-01", 999)])
    f = build_figure_features(hist, hist).set_index("target_fig")
    assert np.isnan(f.loc[70.0, "last_fig"]) and f.loc[70.0, "has_history"] == 0
    assert f.loc[999.0, "last_fig"] == 80 and f.loc[999.0, "avg3_fig"] == 75
    assert f.loc[999.0, "n_prior"] == 2


def test_learns_form_and_simulates_sensible_probs():
    rng = np.random.default_rng(1)
    rows = []
    for h in range(60):
        ability = rng.normal(70, 10)
        for k in range(8):
            rows.append(_start(f"H{h}", f"2025-{1 + k:02d}-15", ability + rng.normal(0, 4)))
    hist = pd.DataFrame(rows)
    m = SpeedFigureModel().fit(build_figure_features(hist, hist))
    card = pd.DataFrame([dict(_start("H0", "2026-01-01", np.nan, race="today"), speed_figure=np.nan),
                         dict(_start("H1", "2026-01-01", np.nan, race="today"), speed_figure=np.nan)])
    f = build_figure_features(hist, card)
    mu, sigma = m.predict_fig(f)
    p = m.predict_proba(f)
    assert abs(p.sum() - 1) < 1e-9 and (sigma > 0).all()
    assert (p[0] > p[1]) == (mu[0] > mu[1])
    assert np.allclose(m.contributions(f).sum(axis=1), mu - mu.mean())

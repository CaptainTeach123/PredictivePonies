import pandas as pd

from ponies.features import build_features
from ponies.xml_io import START_COLUMNS


def _start(race_id, date, horse, finish, fig, field=2, **kw):
    row = dict.fromkeys(START_COLUMNS)
    row.update(race_id=race_id, date=pd.Timestamp(date), track="TST", race_number=1,
               distance_furlongs=6.0, surface="dirt", condition="fast", purse=20000.0,
               field_size=field, horse_id=horse, horse_name=horse, post_position=1.0,
               age=4.0, weight_lbs=120.0, jockey_id="J", trainer_id="T",
               finish_position=float(finish), speed_figure=float(fig), source="result")
    row.update(kw)
    return row


def test_point_in_time_no_leakage():
    hist = pd.DataFrame([
        _start("r1", "2026-01-01", "A", 1, 80),
        _start("r1", "2026-01-01", "B", 2, 70),
        _start("r2", "2026-02-01", "A", 2, 999),   # today's race: must be ignored
        _start("r2", "2026-02-01", "B", 1, 75),
    ])
    feats = build_features(hist, hist[hist.race_id == "r2"]).set_index("horse_id")
    assert feats.loc["A", "speed_last"] == 80
    assert feats.loc["A", "speed_best_recent"] == 80
    assert feats.loc["A", "has_history"] == 1.0
    assert feats.loc["A", "won"] == 0.0 and feats.loc["B", "won"] == 1.0


def test_debut_runner_has_missing_history():
    hist = pd.DataFrame([_start("r1", "2026-01-01", "A", 1, 80)])
    feats = build_features(hist, hist)
    assert feats.loc[0, "has_history"] == 0.0
    assert pd.isna(feats.loc[0, "speed_last"])

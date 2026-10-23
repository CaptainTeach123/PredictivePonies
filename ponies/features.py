"""Stage 2: turn raw starts into per-runner features.

Every feature for a race on date D is computed ONLY from starts strictly
before D (point-in-time), so backtests never peek at the future.

To add a feature: add an entry to ``FEATURES`` (name -> description, which the
dashboard shows in explanations), compute it in ``_runner_features`` and, if it
embeds a judgement call, add an assumption to config.py + ASSUMPTIONS.md.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config, odds

# Order matters: this is the column order of the model's design matrix.
FEATURES: dict[str, str] = {
    "speed_best_recent": "Best speed figure in last starts [A-01]",
    "speed_avg_recent": "Average speed figure in last starts [A-01]",
    "speed_last": "Speed figure in most recent start",
    "finish_pct_recent": "Average finishing percentile in last starts, 1 = won [A-01]",
    "win_rate": "Career win rate, shrunk toward prior [A-03]",
    "log_starts": "log(1 + career starts) - experience",
    "log_layoff": "log(days since last race), capped [A-06]",
    "class_change": "log(today's purse / recent average purse); >0 = moving up in class",
    "distance_fit": "Avg speed fig at similar distance minus overall avg [A-04]",
    "surface_fit": "Avg speed fig on today's surface minus overall avg",
    "condition_fit": "Avg speed fig on today's wet/dry condition minus overall avg [A-05]",
    "jockey_win_rate": "Jockey win rate, trailing window, shrunk [A-02][A-03]",
    "trainer_win_rate": "Trainer win rate, trailing window, shrunk [A-02][A-03]",
    "post_position_rel": "Post position scaled 0 (rail) to 1 (outside)",
    "weight_vs_field": "Weight carried minus race average (lbs)",
    "age": "Age in years",
    "has_history": "1 if horse has any prior start in the data, else 0 [A-07]",
}
if config.USE_MORNING_LINE_FEATURE:  # [A-11]
    FEATURES["log_ml_prob"] = "log(implied win probability from morning line) [A-11]"

FEATURE_NAMES = list(FEATURES)


def _shrunk_rate(wins: float, n: float) -> float:
    """[A-03] Bayesian-average win rate."""
    return (wins + config.PRIOR_RIDES * config.PRIOR_WIN_RATE) / (n + config.PRIOR_RIDES)


def _cond_group(condition: str) -> str:
    """[A-05] Collapse track conditions into 'wet' or 'dry'."""
    return "wet" if (condition or "").lower() in config.WET_CONDITIONS else "dry"


class _WindowCounter:
    """Win counts for a jockey/trainer within a trailing date window."""

    def __init__(self):
        self._events: dict[str, list[tuple[pd.Timestamp, int]]] = defaultdict(list)
        self._built = False

    def add(self, key, date, won):
        if isinstance(key, str) and key:
            self._events[key].append((date, int(won)))

    def build(self):
        self._dates, self._cum = {}, {}
        for key, ev in self._events.items():
            ev.sort(key=lambda e: e[0])
            self._dates[key] = [e[0] for e in ev]
            self._cum[key] = np.concatenate([[0], np.cumsum([e[1] for e in ev])])
        self._built = True

    def rate(self, key, date) -> float:
        if key not in self._dates:
            return _shrunk_rate(0, 0)
        dates = self._dates[key]
        hi = bisect.bisect_left(dates, date)  # strictly before race date
        lo = bisect.bisect_left(dates, date - pd.Timedelta(days=config.CONNECTIONS_WINDOW_DAYS))
        cum = self._cum[key]
        return _shrunk_rate(cum[hi] - cum[lo], hi - lo)


def _index_history(history: pd.DataFrame):
    horses: dict[str, list[dict]] = defaultdict(list)
    jockeys, trainers = _WindowCounter(), _WindowCounter()
    for r in history.itertuples(index=False):
        if math.isnan(r.finish_position):
            continue
        won = r.finish_position == 1
        horses[r.horse_id].append({
            "date": r.date, "fig": r.speed_figure, "finish": r.finish_position,
            "field": r.field_size, "dist": r.distance_furlongs,
            "surface": r.surface, "cond": _cond_group(r.condition),
            "purse": r.purse, "won": won,
        })
        jockeys.add(r.jockey_id, r.date, won)
        trainers.add(r.trainer_id, r.date, won)
    for starts in horses.values():
        starts.sort(key=lambda s: s["date"])
    jockeys.build()
    trainers.build()
    return horses, jockeys, trainers


def _fit(prior: list[dict], match, overall_avg: float) -> float:
    """Avg fig in starts matching a condition minus the overall avg.

    [A-17] 0.0 (neutral) when the horse has figures but none under this
    condition; NaN (-> imputed) when it has no figures at all.
    """
    figs = [s["fig"] for s in prior if match(s) and not math.isnan(s["fig"])]
    if math.isnan(overall_avg):
        return math.nan
    return float(np.mean(figs)) - overall_avg if figs else 0.0


def _runner_features(row, prior: list[dict], jockeys, trainers) -> dict:
    n = config.RECENT_STARTS
    recent = prior[-n:]
    figs_recent = [s["fig"] for s in recent if not math.isnan(s["fig"])]
    figs_all = [s["fig"] for s in prior if not math.isnan(s["fig"])]
    overall = float(np.mean(figs_all)) if figs_all else math.nan

    def pct(s):
        field = s["field"] if s["field"] and not math.isnan(s["field"]) else math.nan
        if math.isnan(field) or field <= 1:
            return math.nan
        return 1.0 - (s["finish"] - 1) / (field - 1)

    pcts = [p for p in (pct(s) for s in recent) if not math.isnan(p)]
    purses = [s["purse"] for s in recent if s["purse"] and not math.isnan(s["purse"])]
    field = row.field_size

    f = {
        "speed_best_recent": max(figs_recent) if figs_recent else math.nan,
        "speed_avg_recent": float(np.mean(figs_recent)) if figs_recent else math.nan,
        "speed_last": prior[-1]["fig"] if prior else math.nan,
        "finish_pct_recent": float(np.mean(pcts)) if pcts else math.nan,
        "win_rate": _shrunk_rate(sum(s["won"] for s in prior), len(prior)),
        "log_starts": math.log1p(len(prior)),
        "log_layoff": (math.log(max(1, min((row.date - prior[-1]["date"]).days,
                                           config.MAX_LAYOFF_DAYS)))
                       if prior else math.nan),
        "class_change": (math.log(row.purse / np.mean(purses))
                         if purses and row.purse and not math.isnan(row.purse) else math.nan),
        "distance_fit": _fit(prior, lambda s: abs(s["dist"] - row.distance_furlongs)
                             <= config.DISTANCE_TOLERANCE_FURLONGS, overall),
        "surface_fit": _fit(prior, lambda s: s["surface"] == row.surface, overall),
        "condition_fit": _fit(prior, lambda s: s["cond"] == _cond_group(row.condition), overall),
        "jockey_win_rate": jockeys.rate(row.jockey_id, row.date),
        "trainer_win_rate": trainers.rate(row.trainer_id, row.date),
        "post_position_rel": ((row.post_position - 1) / (field - 1)
                              if field and field > 1 and not math.isnan(row.post_position)
                              else 0.5),  # [A-18] unknown post = middle
        "weight_vs_field": row.weight_lbs,  # centred per race below
        "age": row.age,
        "has_history": 1.0 if prior else 0.0,
    }
    if config.USE_MORNING_LINE_FEATURE:  # [A-11]
        dec = odds.to_decimal(row.morning_line)
        f["log_ml_prob"] = math.log(1.0 / dec) if dec and not math.isnan(dec) else math.nan
    return f


def id_columns(row) -> dict:
    """Identifying/display columns carried alongside features (used by
    pipeline.race_cards for every model)."""
    return {
        "race_id": row.race_id, "date": row.date, "track": row.track,
        "race_number": row.race_number, "horse_id": row.horse_id,
        "horse_name": row.horse_name, "program": row.program,
        "post_position": row.post_position,
        "jockey_name": row.jockey_name, "trainer_name": row.trainer_name,
        "morning_line": row.morning_line, "final_odds": row.final_odds,
        "finish_position": row.finish_position,
        "distance_furlongs": row.distance_furlongs, "surface": row.surface,
        "condition": row.condition, "race_class": row.race_class,
        "purse": row.purse, "field_size": row.field_size,
    }


def won_label(row) -> float:
    return math.nan if math.isnan(row.finish_position) else float(row.finish_position == 1)


def build_features(history: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Compute features for every row of ``targets`` using ``history``.

    ``targets`` can be historical results (for training/backtesting) or
    upcoming entries (for prediction). Returns one row per runner with id
    columns, FEATURE_NAMES columns, and ``won`` (NaN for entries).
    """
    horses, jockeys, trainers = _index_history(history)
    out = []
    for row in targets.itertuples(index=False):
        starts = horses.get(row.horse_id, [])
        cut = bisect.bisect_left([s["date"] for s in starts], row.date)
        feats = _runner_features(row, starts[:cut], jockeys, trainers)
        out.append({**id_columns(row), **feats, "won": won_label(row)})
    df = pd.DataFrame(out)
    if len(df):
        df["weight_vs_field"] = df["weight_vs_field"] - df.groupby("race_id")["weight_vs_field"].transform("mean")
        df["weight_vs_field"] = df["weight_vs_field"].fillna(0.0)  # [A-18] unknown = race avg
    return df

"""Run the whole pipeline:  XML -> features -> model -> backtest -> dashboard JSON.

    python -m ponies.pipeline

Outputs (all in docs/data/, read by docs/index.html):
    model.json        weights + preprocessing, feature descriptions
    evaluation.json   backtest metrics on held-out races
    predictions.json  win probabilities, rankings, explanations, odds comparison
                      for every race in data/raw/entries/ (plus the test races,
                      so the dashboard can show predicted-vs-actual)
    history.csv       form history used to build features in the browser
    roster.json       known horses / jockeys / trainers (form autocomplete)
    ponies.zip        this package, loaded by the in-browser predictor
                      (ponies/browser.py via Pyodide); generated, not committed
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, odds
from .evaluate import evaluate, market_probabilities
from .features import FEATURES, FEATURE_NAMES, build_features
from .model import ConditionalLogit
from .trackmaster import SOURCE as TRACKMASTER, parse_trackmaster
from .xml_io import combine_history, history_to_csv, load_directory, load_history


def chronological_split(feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """[A-08] Most recent TEST_FRACTION of races (by date) become the test set."""
    races = feats.groupby("race_id")["date"].first().sort_values()
    n_test = max(1, int(round(len(races) * config.TEST_FRACTION)))
    test_ids = set(races.index[-n_test:])
    is_test = feats["race_id"].isin(test_ids)
    return feats[~is_test].copy(), feats[is_test].copy()


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, pd.Timestamp):
        return str(v.date())
    return v


def race_cards(df: pd.DataFrame, model: ConditionalLogit, kind: str) -> list[dict]:
    """Package predictions per race for the dashboard."""
    df = df.reset_index(drop=True)
    df["model_prob"] = model.predict_proba(df)
    contrib = model.contributions(df)
    ml_col = "morning_line"
    df["ml_implied_prob"] = market_probabilities(df, ml_col)
    df["final_implied_prob"] = market_probabilities(df, "final_odds")
    df["ml_decimal"] = df[ml_col].map(odds.to_decimal)
    # [A-13] edge vs. the odds we'd realistically see before the race
    df["edge_vs_ml"] = df["model_prob"] * df["ml_decimal"] - 1

    cards = []
    for race_id, g in df.groupby("race_id", sort=False):
        g = g.sort_values("model_prob", ascending=False)
        first = g.iloc[0]
        runners = []
        for rank, (i, r) in enumerate(g.iterrows(), start=1):
            c = contrib.loc[i]
            runners.append({
                "rank": rank,
                "program": _clean(r["program"]),
                "horse": _clean(r["horse_name"]), "horse_id": r["horse_id"],
                "jockey": _clean(r["jockey_name"]), "trainer": _clean(r["trainer_name"]),
                "post_position": _clean(r["post_position"]),
                "model_prob": _clean(r["model_prob"]),
                "fair_odds": odds.fair_odds_string(r["model_prob"]),
                "morning_line": _clean(r[ml_col]),
                "ml_implied_prob": _clean(r["ml_implied_prob"]),
                "final_odds": r["final_odds"] if isinstance(r["final_odds"], str) else None,
                "final_implied_prob": _clean(r["final_implied_prob"]),
                "edge_vs_ml": _clean(r["edge_vs_ml"]),
                # [A-22] no value flag without form data: imputed "average" features
                # can make an unknown horse look like a bargain at long odds.
                "no_history": not r["has_history"],
                "value_flag": bool(r["edge_vs_ml"] > config.VALUE_EDGE_THRESHOLD and r["has_history"]),
                "finish_position": _clean(r["finish_position"]),
                "features": {f: _clean(r[f]) for f in FEATURE_NAMES},
                "contributions": {f: _clean(c[f]) for f in FEATURE_NAMES},
            })
        cards.append({
            "race_id": race_id, "kind": kind,
            "date": str(first["date"].date()), "track": _clean(first["track"]),
            "race_number": _clean(first["race_number"]),
            "distance_furlongs": _clean(first["distance_furlongs"]),
            "surface": _clean(first["surface"]), "condition": _clean(first["condition"]),
            "race_class": _clean(first["race_class"]), "purse": _clean(first["purse"]),
            "field_size": len(g), "runners": runners,
        })
    return cards


def build_roster(history: pd.DataFrame) -> dict:
    """Latest known name/details per horse, jockey and trainer."""
    h = history.sort_values("date")

    def latest(id_col, name_col, extra=()):
        rows = h[h[id_col].notna() & h[name_col].notna()].drop_duplicates(id_col, keep="last")
        return sorted(({"id": r[id_col], "name": r[name_col], **{k: _clean(r[k]) for k in extra}}
                       for _, r in rows.iterrows()), key=lambda d: d["name"])

    return {
        "horses": latest("horse_id", "horse_name",
                         ("age", "sex", "trainer_id", "trainer_name", "date")),
        "jockeys": latest("jockey_id", "jockey_name"),
        "trainers": latest("trainer_id", "trainer_name"),
    }


def package_source(path: Path) -> None:
    """Zip ponies/*.py so the dashboard can import it in Pyodide."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src in sorted(Path(__file__).parent.glob("*.py")):
            zf.write(src, f"ponies/{src.name}")


def run(out_dir: Path = config.DASHBOARD_DATA_DIR) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 - XML
    history = load_history(config.RESULTS_DIR, config.PP_DIR)
    results = history[history["source"] == "result"]
    entries = load_directory(config.ENTRIES_DIR, "entries") if config.ENTRIES_DIR.exists() else pd.DataFrame()
    # TrackMaster cards: their runners become entries, their PP lines history.
    for path in sorted(config.TRACKMASTER_DIR.glob("*.xml")) if config.TRACKMASTER_DIR.exists() else []:
        tm_entries, tm_pps = parse_trackmaster(ET.parse(path).getroot())
        entries = pd.concat([entries, tm_entries], ignore_index=True)
        history = combine_history(history, tm_pps)

    # Stage 2 - features (training rows are results with a known winner)
    feats = build_features(history, results)
    has_winner = feats.groupby("race_id")["won"].transform("sum") > 0
    feats = feats[has_winner]
    train, test = chronological_split(feats)

    # Stage 3 - model + backtest
    model = ConditionalLogit(FEATURE_NAMES).fit(train)
    metrics = evaluate(test, model.predict_proba(test))
    metrics["train"] = {"races": int(train["race_id"].nunique()), "runners": int(len(train)),
                        "date_range": [str(train["date"].min().date()), str(train["date"].max().date())]}

    # [A-19] Refit on ALL history for live predictions so the latest form counts.
    live_model = ConditionalLogit(FEATURE_NAMES).fit(feats)

    # Stage 4 - dashboard data
    cards = []
    if len(entries):
        cards += race_cards(build_features(history, entries), live_model, "upcoming")
    cards += race_cards(test, model, "backtest")

    synthetic = any(p.name.startswith("SYNTHETIC") for p in config.RAW_DIR.rglob("*.xml"))
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic_data": synthetic,
    }
    model_doc = {**meta, **live_model.to_dict(),
                 "backtest_weights": model.weights.tolist(),
                 "feature_descriptions": FEATURES}
    (out_dir / "model.json").write_text(json.dumps(model_doc, indent=2))
    (out_dir / "evaluation.json").write_text(json.dumps({**meta, **metrics}, indent=2))
    (out_dir / "predictions.json").write_text(json.dumps({**meta, "races": cards}, allow_nan=False))

    # In-browser predictor inputs. Licensed TrackMaster PP lines are not published.
    public = history[history["source"] != TRACKMASTER]
    history_to_csv(public, out_dir / "history.csv")
    (out_dir / "roster.json").write_text(json.dumps(build_roster(public)))
    package_source(out_dir / "ponies.zip")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=config.DASHBOARD_DATA_DIR)
    args = ap.parse_args()
    m = run(args.out)
    print(json.dumps({k: m[k] for k in ("races", "log_loss", "top_pick_win_rate", "betting_backtest")}, indent=2))


if __name__ == "__main__":
    main()

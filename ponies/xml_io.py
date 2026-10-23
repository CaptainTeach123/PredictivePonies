"""Stage 1: read horse-racing XML files into tidy tables.

Three file types are supported (formats documented in DATA_SCHEMA.md):

* results/*.xml            historical race charts with finish positions
* past_performances/*.xml  per-horse lines for races not covered by results
* entries/*.xml            upcoming race cards (no outcomes yet)

Everything is flattened into a single "starts" DataFrame: one row per horse
per race. ``source`` records where the row came from.

To support a different vendor's XML (Equibase, TrackMaster, etc.), write a new
``parse_*`` function that returns rows with the same columns as
``START_COLUMNS`` and register it in ``load_directory``. Nothing downstream
needs to change.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

START_COLUMNS = [
    "race_id", "date", "track", "race_number",
    "distance_furlongs", "surface", "condition", "race_class", "purse",
    "field_size",
    "horse_id", "horse_name", "program", "post_position",
    "age", "sex", "weight_lbs",
    "jockey_id", "jockey_name", "trainer_id", "trainer_name",
    "morning_line", "final_odds",
    "finish_position", "speed_figure", "beaten_lengths",
    "source",
]

NUMERIC_COLUMNS = [
    "race_number", "distance_furlongs", "purse", "field_size", "post_position",
    "age", "weight_lbs", "finish_position", "speed_figure", "beaten_lengths",
]


def make_race_id(track: str, date: str, race_number) -> str:
    return f"{track}-{date}-{int(race_number)}"


def _race_attrs(track: str, date: str, race: ET.Element) -> dict:
    return {
        "track": track,
        "date": date,
        "race_number": race.get("number"),
        "race_id": make_race_id(track, date, race.get("number")),
        "distance_furlongs": race.get("distance_furlongs"),
        "surface": (race.get("surface") or "").lower(),
        "condition": (race.get("condition") or "").lower(),
        "race_class": (race.get("class") or "").lower(),
        "purse": race.get("purse"),
    }


def _starter_attrs(st: ET.Element) -> dict:
    return {
        "horse_id": st.get("horse_id"),
        "horse_name": st.get("horse_name"),
        "program": st.get("program"),
        "post_position": st.get("post_position"),
        "age": st.get("age"),
        "sex": st.get("sex"),
        "weight_lbs": st.get("weight_lbs"),
        "jockey_id": st.get("jockey_id"),
        "jockey_name": st.get("jockey_name"),
        "trainer_id": st.get("trainer_id"),
        "trainer_name": st.get("trainer_name"),
        "morning_line": st.get("morning_line"),
        "final_odds": st.get("final_odds"),
        "finish_position": st.get("finish_position"),
        "speed_figure": st.get("speed_figure"),
        "beaten_lengths": st.get("beaten_lengths"),
    }


def parse_card(path: Path, source: str) -> list[dict]:
    """Parse a results or entries file (same layout; entries lack outcomes)."""
    return parse_card_root(ET.parse(path).getroot(), source)


def parse_card_root(root: ET.Element, source: str) -> list[dict]:
    track, date = root.get("track"), root.get("date")
    rows = []
    for race in root.iter("race"):
        base = _race_attrs(track, date, race)
        # [A-21] scratched runners are dropped and excluded from field_size
        starters = [s for s in race.iter("starter")
                    if (s.get("scratched") or "false").lower() != "true"]
        for st in starters:
            rows.append({**base, **_starter_attrs(st),
                         "field_size": len(starters), "source": source})
    return rows


def parse_past_performances(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    rows = []
    for horse in root.iter("horse"):
        for line in horse.iter("line"):
            track, date = line.get("track"), line.get("date")
            rows.append({
                "race_id": make_race_id(track, date, line.get("race_number")),
                "date": date,
                "track": track,
                "race_number": line.get("race_number"),
                "distance_furlongs": line.get("distance_furlongs"),
                "surface": (line.get("surface") or "").lower(),
                "condition": (line.get("condition") or "").lower(),
                "race_class": (line.get("class") or "").lower(),
                "purse": line.get("purse"),
                "field_size": line.get("field_size"),
                "horse_id": horse.get("id"),
                "horse_name": horse.get("name"),
                "post_position": line.get("post_position"),
                "weight_lbs": line.get("weight_lbs"),
                "jockey_id": line.get("jockey_id"),
                "trainer_id": line.get("trainer_id"),
                "final_odds": line.get("final_odds"),
                "finish_position": line.get("finish_position"),
                "speed_figure": line.get("speed_figure"),
                "beaten_lengths": line.get("beaten_lengths"),
                "source": "pp",
            })
    return rows


def entries_from_text(xml_text: str) -> pd.DataFrame:
    """Entries XML supplied as a string (used by the in-browser predictor)."""
    return _to_frame(parse_card_root(ET.fromstring(xml_text), "entry"))


# Columns the feature builder needs from history (features._index_history).
HISTORY_COLUMNS = [
    "date", "horse_id", "jockey_id", "trainer_id", "distance_furlongs", "surface",
    "condition", "purse", "field_size", "finish_position", "speed_figure",
]


def history_to_csv(history: pd.DataFrame, path: Path) -> None:
    out = history[HISTORY_COLUMNS].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def history_from_csv(source) -> pd.DataFrame:
    """Inverse of ``history_to_csv``. ``source`` is a path or file-like object."""
    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    for col in NUMERIC_COLUMNS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df


def _to_frame(rows: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    df = df.reindex(columns=START_COLUMNS)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_directory(directory: Path, kind: str) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(Path(directory).glob("*.xml")):
        if kind == "results":
            rows += parse_card(path, "result")
        elif kind == "entries":
            rows += parse_card(path, "entry")
        elif kind == "pp":
            rows += parse_past_performances(path)
        else:
            raise ValueError(kind)
    return _to_frame(rows)


def load_history(results_dir: Path, pp_dir: Path) -> pd.DataFrame:
    """All completed starts.

    [A-20] Result charts win over PP lines for the same start (race_id + horse_id).
    """
    results = load_directory(results_dir, "results")
    pps = load_directory(pp_dir, "pp") if Path(pp_dir).exists() else _to_frame([])
    if len(pps):
        known = set(zip(results["race_id"], results["horse_id"]))
        mask = [(r, h) not in known for r, h in zip(pps["race_id"], pps["horse_id"])]
        pps = pps[np.array(mask, dtype=bool)]
    frames = [f for f in (results, pps) if len(f)]
    history = pd.concat(frames, ignore_index=True) if frames else _to_frame([])
    return history.sort_values(["date", "race_id", "post_position"]).reset_index(drop=True)

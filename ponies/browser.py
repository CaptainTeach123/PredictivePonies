"""In-browser predictor, run by the dashboard through Pyodide (Python in WebAssembly).

docs/predict.js loads this package from docs/data/ponies.zip (built by
``pipeline.run``) and calls:

    load_state(history_csv, model_json, figure_model_json)   # once
    predict_card(entries_xml_text) -> JSON string            # per race card

It runs the same feature and model code as the offline pipeline. Ordinary
cards use the fitted model exported to docs/data/ (whichever [A-30] chose).
A TrackMaster file is different: the speed-figure model is trained in the
browser on that file's own past-performance lines (about a second), so the
licensed data never leaves the viewer's machine.
"""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd

from .features import build_features
from .figure_model import SpeedFigureModel, build_figure_features
from .model import ConditionalLogit
from .pipeline import race_cards, trackmaster_cards
from .trackmaster import is_trackmaster, parse_trackmaster
from .xml_io import _to_frame, combine_history, history_from_csv, parse_card_root

_state: dict = {}


def load_state(history_csv: str, model_json: str, figure_model_json: str | None = None) -> None:
    _state["history"] = history_from_csv(io.StringIO(history_csv))
    doc = json.loads(model_json)
    _state["model"] = ConditionalLogit.from_dict(doc)
    _state["chosen"] = doc.get("chosen_model", "conditional_logit")  # [A-30]
    fig_doc = json.loads(figure_model_json) if figure_model_json else None
    _state["figure_model"] = SpeedFigureModel.from_dict(fig_doc) if fig_doc else None
    # model_info for [A-35] value flags: the exported models' own test results
    _state["cl_info"] = {"race_log_loss": doc.get("race_log_loss")}
    _state["fig_info"] = ({k: fig_doc.get(k) for k in ("train_starts", "holdout", "temperature", "race_log_loss")}
                          if fig_doc else None)


def predict_card(entries_xml: str) -> str:
    """Entries XML in either the repo's layout or TrackMaster Plus XML.

    A TrackMaster file brings its runners' past performances with it; they're
    added to the history for this prediction only.
    """
    root = ET.fromstring(entries_xml)
    history = _state["history"]
    if is_trackmaster(root):
        return json.dumps(json.loads(predict_files(json.dumps([entries_xml])))["races"])
    else:
        entries = _to_frame(parse_card_root(root, "entry"))
        if entries.empty:
            raise ValueError("No runners found in the race card.")
        cards = _predict_with_exported(history, entries)
    return json.dumps(cards, allow_nan=False)


# ---- uploads: files are staged in Python so large cards aren't copied back to JS
MAX_XML_BYTES = 60_000_000        # per file
MAX_TOTAL_BYTES = 600_000_000     # per upload


def stage_clear() -> None:
    _state["staged"] = []


def stage_xml(text: str) -> int:
    _state.setdefault("staged", []).append(text)
    return len(_state["staged"])


def stage_zip(data) -> int:
    """Add every .xml inside a zip (zips inside zips too, like TrackMaster's
    daily downloads). ``data`` is bytes, or a JS Uint8Array from Pyodide."""
    raw = bytes(data.to_py()) if hasattr(data, "to_py") else bytes(data)
    total = sum(len(t) for t in _state.get("staged", []))

    def walk(blob: bytes, depth: int = 0):
        nonlocal total
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for info in zf.infolist():
                name = info.filename.lower()
                if name.endswith(".zip") and depth < 3:
                    walk(zf.read(info), depth + 1)
                elif name.endswith(".xml") and info.file_size <= MAX_XML_BYTES:
                    total += info.file_size
                    if total > MAX_TOTAL_BYTES:
                        raise ValueError("Upload too large (over 600 MB of XML).")
                    stage_xml(zf.read(info).decode("utf-8", errors="replace"))

    walk(raw)
    return len(_state.get("staged", []))


def predict_staged() -> str:
    texts = _state.get("staged", [])
    if not texts:
        raise ValueError("No XML files found in the upload.")
    try:
        return predict_files(json.dumps(texts))
    finally:
        stage_clear()


def predict_files(xml_texts_json: str) -> str:
    """Several TrackMaster files at once (JSON list of XML strings).

    All files' PP lines are pooled to train the speed-figure models (one per
    breed [A-31]), so more files = more training data; every file's races are
    predicted. Non-TrackMaster files are predicted one by one as usual.
    """
    entries, pps, cards = [], [], []
    for text in json.loads(xml_texts_json):
        root = ET.fromstring(text)
        if is_trackmaster(root):
            e, p = parse_trackmaster(root)
            entries.append(e)
            pps.append(p)
        else:
            cards += json.loads(predict_card(text))
    skipped = []
    if entries:
        all_entries = pd.concat(entries, ignore_index=True)
        if all_entries.empty:
            raise ValueError("No runners found in the race card.")
        # the same past start can appear in several files (horse entered twice)
        all_pps = pd.concat(pps, ignore_index=True).drop_duplicates(["race_id", "horse_id"])
        tm, skipped = trackmaster_cards(all_pps, all_entries, "custom")
        cards += tm
    # skipped races are reported, not predicted with a model that can't judge them
    return json.dumps({"races": cards, "skipped": skipped}, allow_nan=False)


def _predict_with_exported(history, entries) -> list[dict]:
    if _state["chosen"] == "speed_figure" and _state["figure_model"] is not None:
        return race_cards(build_figure_features(history, entries), _state["figure_model"], "custom",
                          _state["fig_info"])
    return race_cards(build_features(history, entries), _state["model"], "custom", _state["cl_info"])

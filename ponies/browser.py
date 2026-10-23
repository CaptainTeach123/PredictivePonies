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

from .features import build_features
from .figure_model import SpeedFigureModel, build_figure_features, fit_with_holdout
from .model import ConditionalLogit
from .pipeline import race_cards
from .trackmaster import is_trackmaster, parse_trackmaster
from .xml_io import _to_frame, combine_history, history_from_csv, parse_card_root

_state: dict = {}


def load_state(history_csv: str, model_json: str, figure_model_json: str | None = None) -> None:
    _state["history"] = history_from_csv(io.StringIO(history_csv))
    doc = json.loads(model_json)
    _state["model"] = ConditionalLogit.from_dict(doc)
    _state["chosen"] = doc.get("chosen_model", "conditional_logit")  # [A-30]
    _state["figure_model"] = (SpeedFigureModel.from_dict(json.loads(figure_model_json))
                              if figure_model_json else None)


def predict_card(entries_xml: str) -> str:
    """Entries XML in either the repo's layout or TrackMaster Plus XML.

    A TrackMaster file brings its runners' past performances with it; they're
    added to the history for this prediction only.
    """
    root = ET.fromstring(entries_xml)
    history = _state["history"]
    if is_trackmaster(root):
        # [A-30] train the speed-figure model on this file's own PP lines.
        entries, pps = parse_trackmaster(root)
        if entries.empty:
            raise ValueError("No runners found in the race card.")
        try:
            model, info = fit_with_holdout(pps)
            cards = race_cards(build_figure_features(pps, entries), model, "custom", info)
        except ValueError:  # too few figured starts to train on: use the exported model
            cards = _predict_with_exported(combine_history(history, pps), entries)
    else:
        entries = _to_frame(parse_card_root(root, "entry"))
        if entries.empty:
            raise ValueError("No runners found in the race card.")
        cards = _predict_with_exported(history, entries)
    return json.dumps(cards, allow_nan=False)


def _predict_with_exported(history, entries) -> list[dict]:
    if _state["chosen"] == "speed_figure" and _state["figure_model"] is not None:
        return race_cards(build_figure_features(history, entries), _state["figure_model"], "custom")
    return race_cards(build_features(history, entries), _state["model"], "custom")

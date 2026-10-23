"""In-browser predictor, run by the dashboard through Pyodide (Python in WebAssembly).

docs/app.js loads this package from docs/py/ponies.zip (built by
``pipeline.run``) and calls:

    load_state(history_csv_text, model_json_text)   # once
    predict_card(entries_xml_text) -> JSON string   # per race card

It runs the same feature and model code as the offline pipeline, using the
history and fitted weights exported to docs/data/. Nothing is retrained in
the browser.
"""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET

from .features import build_features
from .model import ConditionalLogit
from .pipeline import race_cards
from .trackmaster import is_trackmaster, parse_trackmaster
from .xml_io import _to_frame, combine_history, history_from_csv, parse_card_root

_state: dict = {}


def load_state(history_csv: str, model_json: str) -> None:
    _state["history"] = history_from_csv(io.StringIO(history_csv))
    _state["model"] = ConditionalLogit.from_dict(json.loads(model_json))


def predict_card(entries_xml: str) -> str:
    """Entries XML in either the repo's layout or TrackMaster Plus XML.

    A TrackMaster file brings its runners' past performances with it; they're
    added to the history for this prediction only.
    """
    root = ET.fromstring(entries_xml)
    history = _state["history"]
    if is_trackmaster(root):
        entries, pps = parse_trackmaster(root)
        history = combine_history(history, pps)
    else:
        entries = _to_frame(parse_card_root(root, "entry"))
    if entries.empty:
        raise ValueError("No runners found in the race card.")
    feats = build_features(history, entries)
    return json.dumps(race_cards(feats, _state["model"], "custom"), allow_nan=False)

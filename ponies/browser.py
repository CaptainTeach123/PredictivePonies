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

from .features import build_features
from .model import ConditionalLogit
from .pipeline import race_cards
from .xml_io import entries_from_text, history_from_csv

_state: dict = {}


def load_state(history_csv: str, model_json: str) -> None:
    _state["history"] = history_from_csv(io.StringIO(history_csv))
    _state["model"] = ConditionalLogit.from_dict(json.loads(model_json))


def predict_card(entries_xml: str) -> str:
    entries = entries_from_text(entries_xml)
    if entries.empty:
        raise ValueError("No runners found in the race card.")
    feats = build_features(_state["history"], entries)
    return json.dumps(race_cards(feats, _state["model"], "custom"), allow_nan=False)

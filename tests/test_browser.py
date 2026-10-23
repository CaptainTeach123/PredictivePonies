"""ponies.browser (run in Pyodide by docs/predict.js) must match the offline pipeline."""
import json
import zipfile

from ponies import browser, config, pipeline


def test_browser_matches_pipeline(tmp_path):
    pipeline.run(tmp_path)
    browser.load_state((tmp_path / "history.csv").read_text(), (tmp_path / "model.json").read_text(),
                       (tmp_path / "figure_model.json").read_text())
    entry_file = sorted(config.ENTRIES_DIR.glob("*.xml"))[0]
    cards = json.loads(browser.predict_card(entry_file.read_text()))

    offline = {r["race_id"]: r for r in json.loads((tmp_path / "predictions.json").read_text())["races"]
               if r["kind"] == "upcoming"}
    for card in cards:
        assert card["kind"] == "custom"
        want = [(r["horse_id"], r["model_prob"]) for r in offline[card["race_id"]]["runners"]]
        assert [(r["horse_id"], r["model_prob"]) for r in card["runners"]] == want

    names = zipfile.ZipFile(tmp_path / "ponies.zip").namelist()
    assert "ponies/browser.py" in names and "ponies/model.py" in names


def test_unknown_horse_treated_as_debut(tmp_path):
    pipeline.run(tmp_path)
    browser.load_state((tmp_path / "history.csv").read_text(), (tmp_path / "model.json").read_text(),
                       (tmp_path / "figure_model.json").read_text())
    xml = """<entries track="TST" date="2030-01-01"><race number="1" distance_furlongs="6" surface="dirt" condition="fast">
      <starter program="1" horse_id="NEW-A" horse_name="A" morning_line="2-1"/>
      <starter program="2" horse_id="NEW-B" horse_name="B" morning_line="3-1"/></race></entries>"""
    (card,) = json.loads(browser.predict_card(xml))
    assert all(r["features"]["has_history"] == 0 for r in card["runners"])
    assert abs(sum(r["model_prob"] for r in card["runners"]) - 1) < 1e-3

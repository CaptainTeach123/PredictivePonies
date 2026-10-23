from ponies.xml_io import load_directory, load_history

RESULT = """<?xml version="1.0"?>
<results track="TST" date="2026-01-10">
  <race number="1" distance_furlongs="6" surface="Dirt" condition="Fast" class="claiming" purse="20000">
    <starter program="1" post_position="1" horse_id="A" horse_name="Alpha" jockey_id="J1" trainer_id="T1"
             morning_line="2-1" final_odds="1.5-1" finish_position="1" speed_figure="90"/>
    <starter program="2" post_position="2" horse_id="B" horse_name="Bravo" jockey_id="J2" trainer_id="T2"
             morning_line="3-1" final_odds="4-1" finish_position="2" speed_figure="85"/>
    <starter program="3" post_position="3" horse_id="C" horse_name="Charlie" scratched="true"/>
  </race>
</results>"""

PP = """<?xml version="1.0"?>
<past_performances>
  <horse id="A" name="Alpha">
    <line date="2026-01-10" track="TST" race_number="1" finish_position="1" speed_figure="90"/>
    <line date="2025-12-01" track="TST" race_number="4" distance_furlongs="6" surface="dirt"
          field_size="8" finish_position="3" speed_figure="80"/>
  </horse>
</past_performances>"""


def test_parse_and_merge(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "pp").mkdir()
    (tmp_path / "results" / "r.xml").write_text(RESULT)
    (tmp_path / "pp" / "p.xml").write_text(PP)

    results = load_directory(tmp_path / "results", "results")
    assert len(results) == 2                      # scratched runner dropped
    assert set(results["field_size"]) == {2}
    assert results["surface"].iloc[0] == "dirt"   # lower-cased

    hist = load_history(tmp_path / "results", tmp_path / "pp")
    # PP line duplicating the result chart is dropped; older one kept
    assert len(hist) == 3
    assert (hist["source"] == "pp").sum() == 1

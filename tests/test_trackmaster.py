"""TrackMaster Plus XML reader. The fixture is hand-written in TrackMaster's
layout (field names from plusProXMLData.xsd); it is not licensed data."""
import json
import xml.etree.ElementTree as ET

from ponies import browser, pipeline
from ponies.trackmaster import is_trackmaster, parse_trackmaster

PP = """<ppdata><racedate>{date}</racedate><trackcode>FL</trackcode><racenumber>{rn}</racenumber>
  <racetype>CLM</racetype><purse>15000</purse><trackcondi>{cond}</trackcondi><distance>600</distance>
  <disttype>F</disttype><surface>D</surface><postpositi>3</postpositi><weightcarr>120</weightcarr>
  <jockdisp>Rider Ann</jockdisp><fieldsize>{field}</fieldsize><posttimeod>{odds}</posttimeod>
  <positionfi>{fin}</positionfi><lenbackfin>{lb}</lenbackfin><speedfigur>{fig}</speedfigur></ppdata>"""


def horse(name, prog, ml, pps):
    return f"""<horsedata><horse_name>{name}</horse_name><program>{prog}</program><pp>{prog}</pp>
      <weight>121</weight><sex>G</sex><foal_date>20220301</foal_date><morn_odds>{ml}</morn_odds>
      <jockey><jock_disp>Rider Ann</jock_disp></jockey><trainer><tran_disp>Trainer Bob</tran_disp></trainer>
      {''.join(pps)}</horsedata>"""


CARD = f"""<?xml version="1.0"?><data><racedata><race>1</race><distance>550.0</distance><dist_unit>F</dist_unit>
  <surface>D</surface><purse>20000</purse><track>FL</track><race_date>20261020</race_date><stk_clm_md>CLM</stk_clm_md>
  {horse("FAST ONE", 1, "2/1", [PP.format(date="20261001", rn=3, cond="FT", field=8, odds="2.50", fin=1, lb=0, fig=80),
                                PP.format(date="20260901", rn=5, cond="SY", field=7, odds="4.10", fin=2, lb=150, fig=75)])}
  {horse("SLOW ONE", 2, "8/1", [PP.format(date="20261001", rn=3, cond="FT", field=8, odds="20.00", fin=7, lb=1200, fig=50),
                                PP.format(date="20260801", rn=1, cond="FT", field=0, odds="0.00", fin=0, lb=0, fig=0)])}
  {horse("DEBUT", 3, "5/1", [])}
</racedata></data>"""


def test_parse_units_and_skips():
    root = ET.fromstring(CARD)
    assert is_trackmaster(root)
    entries, pps = parse_trackmaster(root)
    assert len(entries) == 3 and entries["distance_furlongs"].iloc[0] == 5.5
    assert entries["age"].iloc[0] == 4 and entries["morning_line"].iloc[0] == "2/1"
    assert len(pps) == 3                                   # field-size-0 line skipped [A-25]
    fast = pps[pps["horse_name"] == "Fast One"].sort_values("date")
    assert fast["condition"].tolist() == ["sloppy", "fast"]
    assert fast["beaten_lengths"].tolist() == [1.5, 0.0]   # hundredths of a length
    assert fast["distance_furlongs"].iloc[0] == 6.0


def test_browser_predicts_trackmaster_card(tmp_path):
    pipeline.run(tmp_path)
    browser.load_state((tmp_path / "history.csv").read_text(), (tmp_path / "model.json").read_text(),
                       (tmp_path / "figure_model.json").read_text())
    (card,) = json.loads(browser.predict_card(CARD))
    runners = {r["horse"]: r for r in card["runners"]}
    assert runners["Fast One"]["model_prob"] > runners["Slow One"]["model_prob"]
    assert runners["Debut"]["no_history"] and not runners["Fast One"]["no_history"]
    assert abs(sum(r["model_prob"] for r in card["runners"]) - 1) < 1e-3


def _big_card(n_lines=12):
    """Two horses with long histories: 'Steady' runs ~80, 'Plodder' ~55."""
    def lines(base):
        return [PP.format(date=f"2026{m:02d}{d:02d}", rn=r, cond="FT", field=8, odds="3.00",
                          fin=1 if base > 70 else 6, lb=0 if base > 70 else 800, fig=base + (i % 3) - 1)
                for i, (m, d, r) in enumerate((m, d, r) for m in range(1, 10) for d, r in ((5, 2), (20, 4)))][:n_lines]
    return f"""<?xml version="1.0"?><data><racedata><race>1</race><distance>600.0</distance><dist_unit>F</dist_unit>
      <surface>D</surface><purse>20000</purse><track>FL</track><race_date>20261020</race_date><stk_clm_md>CLM</stk_clm_md>
      {horse("STEADY", 1, "2/1", lines(80))}{horse("PLODDER", 2, "3/1", lines(55))}</racedata></data>"""


def test_trackmaster_card_trains_speed_figure_model_in_browser(tmp_path):
    pipeline.run(tmp_path)
    browser.load_state((tmp_path / "history.csv").read_text(), (tmp_path / "model.json").read_text(),
                       (tmp_path / "figure_model.json").read_text())
    (card,) = json.loads(browser.predict_card(_big_card()))
    assert card["model"] == "speed_figure" and card["model_info"]["train_starts"] == 24
    r = {x["horse"]: x for x in card["runners"]}
    assert r["Steady"]["pred_fig"] > r["Plodder"]["pred_fig"] + 10
    assert r["Steady"]["model_prob"] > 0.8

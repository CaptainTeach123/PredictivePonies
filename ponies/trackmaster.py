"""TrackMaster Plus XML (plusProXMLData.xsd) -> the repo's starts tables.

One TrackMaster file is one track's card for one day:

    <data>
      <racedata>                      one per race (today's card)
        <race>, <distance>, <surface>, <purse>, <track>, <race_date>, ...
        <horsedata>                   one per runner
          <horse_name>, <program>, <pp>, <weight>, <morn_odds>, <sex>, <foal_date>,
          <jockey><jock_disp>, <trainer><tran_disp>, ...
          <ppdata>                    one per past start (the horse's form)
            <racedate>, <trackcode>, <racenumber>, <distance>, <surface>,
            <trackcondi>, <fieldsize>, <positionfi>, <speedfigur>,
            <lenbackfin>, <posttimeod>, <jockdisp>, ...

``parse_trackmaster`` returns (entries, past_performances) DataFrames with
xml_io.START_COLUMNS, so the rest of the pipeline is unchanged. The PP lines
give each runner its form, so a TrackMaster card predicts properly even with
no other history loaded.

TrackMaster data is licensed: keep the files out of the public repo
(data/raw/trackmaster/ is git-ignored) and don't publish the PP lines.
Conversions and judgement calls are tagged A-23 to A-25.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pandas as pd

from .xml_io import _to_frame, make_race_id

SOURCE = "trackmaster"
DOMESTIC_COUNTRIES = {"USA", "CAN", "PR", ""}  # [A-36]

_SURFACE = {"D": "dirt", "T": "turf", "A": "synthetic"}
_CONDITION = {
    "FT": "fast", "GD": "good", "SY": "sloppy", "MY": "muddy", "WF": "wet fast",
    "SL": "slow", "HY": "heavy", "FM": "firm", "YL": "yielding", "SF": "soft",
    # [A-25] rarer codes: good-to-firm (dry), good-to-soft (wet), standard
    # all-weather (dry), wet, frozen (dry)
    "GF": "firm", "GS": "soft", "ST": "standard", "WT": "wet fast", "FZ": "frozen",
}


def is_trackmaster(root: ET.Element) -> bool:
    return root.tag == "data" and root.find("racedata") is not None


def _t(el: ET.Element, tag: str) -> str:
    return (el.findtext(tag) or "").strip()


def _num(el: ET.Element, tag: str) -> float | None:
    try:
        return float(_t(el, tag))
    except ValueError:
        return None


def _date(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", s.upper()).strip("-")


def person_id(kind: str, name: str) -> str | None:
    """[A-23] TrackMaster PP lines name jockeys but carry no key, so jockeys
    and trainers are identified by display name ('Davis Jacqueline A')."""
    return f"TM-{kind}-{_slug(name)}" if name else None


def horse_id(horsedata: ET.Element) -> str:
    """[A-23] No horse key in the file: name + foaling date identifies a horse."""
    return f"TM-H-{_slug(_t(horsedata, 'horse_name'))}-{_t(horsedata, 'foal_date')}"


# [A-24] TrackMaster distance units: F = hundredths of a furlong ('550.0' =
# 5.5f), Y = yards (Quarter Horse sprints, '350' = 350 yards), M = metres
# ('1200.0' displayed as '6F*').
_TO_FURLONGS = {"F": 0.01, "": 0.01, "Y": 1 / 220.0, "M": 1 / 201.168}


def _furlongs(el: ET.Element, dist_tag: str, unit_tag: str) -> float | None:
    d = _num(el, dist_tag)
    factor = _TO_FURLONGS.get(_t(el, unit_tag).upper())
    return round(d * factor, 3) if d is not None and factor else None


JUMP_COURSES = {"M", "B", "S"}  # [A-31] hurdle / brush-timber / steeplechase courses


def breed_group(code: str, course_id: str = "") -> str:
    """[A-31] Speed figures are only comparable within a group:
    Thoroughbred flat (TB), Thoroughbred jump races (TB-JUMP) and Quarter Horse
    (QH, AQHA scale). Mixed-breed races (MX) are run at Quarter Horse
    distances, so they count as QH."""
    code = (code or "").upper()
    group = "QH" if code in {"QH", "MX"} else ("TB" if code in {"TB", ""} else code)
    if group == "TB" and (course_id or "").upper() in JUMP_COURSES:
        group = "TB-JUMP"
    return group


def _age(foal_date: str, race_date: str) -> int | None:
    """[A-24] North American ages: everyone turns a year older on 1 January."""
    if len(foal_date) == 8 and len(race_date) == 8:
        return int(race_date[:4]) - int(foal_date[:4])
    return None


def _race_attrs(rd: ET.Element) -> dict:
    track, date = _t(rd, "track"), _date(_t(rd, "race_date"))
    return {
        "track": track, "date": date, "race_number": _t(rd, "race"),
        "race_id": make_race_id(track, date, _t(rd, "race")),
        "distance_furlongs": _furlongs(rd, "distance", "dist_unit"),
        "surface": _SURFACE.get(_t(rd, "surface").upper(), ""),
        "condition": "",  # today's track condition isn't known pre-race
        "race_class": _t(rd, "stk_clm_md").lower(),
        "purse": _num(rd, "purse"),
        "breed": breed_group(_t(rd, "breed_type"), _t(rd, "course_id")),
    }


def _pp_row(p: ET.Element, hid: str, name: str, foal_date: str = "") -> dict | None:
    """One past start. Lines with field size 0 aren't races (no finish, odds or
    figure) and are skipped. [A-25]"""
    field = _num(p, "fieldsize")
    finish = _num(p, "positionfi")
    if not field or not finish:
        return None
    track, date = _t(p, "trackcode"), _date(_t(p, "racedate"))
    fig = _num(p, "speedfigur")
    # [A-36] figures from races abroad run on a higher scale: keep the start
    # (experience, layoff) but not its figure
    abroad = _t(p, "country").upper() not in DOMESTIC_COUNTRIES
    if abroad:
        fig = None
    odds = _num(p, "posttimeod")
    lengths = _num(p, "lenbackfin")
    return {
        "race_id": make_race_id(track, date, _t(p, "racenumber")),
        "date": date, "track": track, "race_number": _t(p, "racenumber"),
        "distance_furlongs": _furlongs(p, "distance", "disttype"),
        "surface": _SURFACE.get(_t(p, "surface").upper(), ""),
        "condition": _CONDITION.get(_t(p, "trackcondi").upper(), _t(p, "trackcondi").lower()),
        "race_class": _t(p, "racetype").lower(),
        "purse": _num(p, "purse"),
        "field_size": field,
        "horse_id": hid, "horse_name": name,
        "post_position": _num(p, "postpositi"),
        "weight_lbs": _num(p, "weightcarr"),
        "age": _age(foal_date, _t(p, "racedate")),
        "jockey_id": person_id("J", _t(p, "jockdisp")),
        "jockey_name": _t(p, "jockdisp") or None,
        "final_odds": f"{odds:.2f}" if odds else None,  # odds-to-one, e.g. 4.70
        "finish_position": finish,
        "speed_figure": fig if fig and fig > 0 else None,  # [A-25] 0 / -97 = no figure
        "beaten_lengths": lengths / 100.0 if lengths is not None else None,  # hundredths
        "breed": breed_group(_t(p, "racebreed"), _t(p, "courseid")),
        "abroad": 1.0 if abroad else 0.0,
        "source": SOURCE,
    }


def parse_trackmaster(root: ET.Element) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries, pps = [], []
    for rd in root.iter("racedata"):
        base = _race_attrs(rd)
        runners = rd.findall("horsedata")
        for h in runners:
            hid, name = horse_id(h), _t(h, "horse_name").title()
            jockey = _t(h.find("jockey"), "jock_disp") if h.find("jockey") is not None else ""
            trainer = _t(h.find("trainer"), "tran_disp") if h.find("trainer") is not None else ""
            entries.append({
                **base, "field_size": len(runners),
                "horse_id": hid, "horse_name": name,
                "program": _t(h, "program"), "post_position": _num(h, "pp"),
                "age": _age(_t(h, "foal_date"), _t(rd, "race_date")),
                "sex": _t(h, "sex"), "weight_lbs": _num(h, "weight"),
                "jockey_id": person_id("J", jockey), "jockey_name": jockey or None,
                "trainer_id": person_id("T", trainer), "trainer_name": trainer or None,
                "morning_line": _t(h, "morn_odds") or None,
                "source": "entry",
            })
            pps += [row for p in h.findall("ppdata")
                    if (row := _pp_row(p, hid, name, _t(h, "foal_date")))]
    return _to_frame(entries), _to_frame(pps)

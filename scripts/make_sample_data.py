"""Generate SYNTHETIC sample XML so the pipeline runs before real data is added.

    python scripts/make_sample_data.py            # writes to data/raw/

Every file is prefixed SYNTHETIC_ and the dashboard shows a warning banner
while any such file is present. Delete them (``rm data/raw/*/SYNTHETIC_*``)
once you've added real XML.

The simulation gives each horse a hidden ability plus distance/surface/wet
preferences, and each jockey/trainer a hidden skill, so the model has real
signal to find. Odds are produced by a noisy "market" that also sees most of
that signal -- so beating the market here is as hard as it should be.
"""

from __future__ import annotations

import argparse
import math
import random
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_HORSES = 800
TRACKS = {"SYN1": "Synthetic Downs", "SYN2": "Mock Park"}
ML_LADDER = [0.2, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.4, 1.5, 1.6, 1.8, 2, 2.5, 3, 3.5, 4,
             4.5, 5, 6, 8, 10, 12, 15, 20, 30, 50]
ADJ = ["Silver", "Midnight", "Lucky", "Golden", "Rapid", "Wild", "Royal", "Iron", "Stormy",
       "Quiet", "Bold", "Crimson", "Electric", "Frosty", "Hidden", "Northern", "Dusty", "Brave",
       "Velvet", "Copper", "Silent", "Southern", "Distant", "Burning", "Lonesome", "Jolly", "Clever",
       "Mighty", "Scarlet", "Hasty"]
NOUN = ["Arrow", "Comet", "Dancer", "Echo", "Falcon", "Gambit", "Harbor", "Legend", "Mirage",
        "Outlaw", "Prospect", "Quest", "Rocket", "Spirit", "Thunder", "Voyage", "Whisper", "Zephyr",
        "Banner", "Cyclone", "Dynasty", "Empire", "Fortune", "Glory", "Horizon", "Jewel", "Knight",
        "Lantern", "Monarch", "Nomad"]
FIRST = ["J.", "M.", "L.", "R.", "T.", "K.", "A.", "D.", "S.", "C."]
LAST = ["Alvarez", "Baker", "Castro", "Dunn", "Ellis", "Flores", "Garcia", "Hughes", "Ito",
        "Jensen", "Kowalski", "Lopez", "Moreno", "Nash", "Ortiz", "Price", "Quinn", "Rossi",
        "Santos", "Turner", "Vega", "Walsh", "Young", "Zhang", "Brooks"]


def to_ml(p: float) -> str:
    target = 1 / p - 1
    best = min(ML_LADDER, key=lambda x: abs(math.log(x) - math.log(max(target, 0.1))))
    frac = {0.2: "1-5", 0.4: "2-5", 0.5: "1-2", 0.6: "3-5", 0.8: "4-5", 1: "1-1", 1.2: "6-5",
            1.4: "7-5", 1.5: "3-2", 1.6: "8-5", 1.8: "9-5", 2.5: "5-2", 3.5: "7-2", 4.5: "9-2"}
    return frac.get(best, f"{int(best)}-1")


def to_tote(p: float, takeout=0.17) -> str:
    to_one = max(0.05, (1 - takeout) / p - 1)
    return f"{round(to_one, 1) if to_one < 10 else round(to_one)}-1"


def simulate(seed: int, start: date, days: int):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    names = random.sample([f"{a} {n}" for a in ADJ for n in NOUN], N_HORSES)
    horses = [{
        "id": f"H{i:04d}", "name": names[i],
        "ability": rng.normal(0, 1), "age": int(rng.integers(2, 7)), "sex": random.choice("CFGMH"),
        "sprinter": rng.normal(0, 1), "turf": rng.normal(0, 1), "wet": rng.normal(0, 1),
        "next_ok": start + timedelta(days=int(rng.integers(0, 40))),
    } for i in range(N_HORSES)]
    jockeys = [{"id": f"J{i:03d}", "name": f"{random.choice(FIRST)} {LAST[i]}",
                "skill": rng.normal(0, 1)} for i in range(len(LAST))]
    trainers = [{"id": f"T{i:03d}", "name": f"{random.choice(FIRST)} {random.choice(LAST)}",
                 "skill": rng.normal(0, 1)} for i in range(20)]
    for h in horses:
        h["trainer"] = random.choice(trainers)
        h["last_run"] = None

    cards = []
    for d in range(days):
        day = start + timedelta(days=d)
        for t_i, track in enumerate(TRACKS):
            if day.weekday() not in ((4, 5) if t_i == 0 else (5, 6)):
                continue
            races = []
            for rn in range(1, 7):
                dist = random.choice([5, 5.5, 6, 6.5, 7, 8, 8.5, 9])
                surface = "turf" if rng.random() < 0.3 else "dirt"
                cond = (random.choice(["firm", "good", "yielding"]) if surface == "turf"
                        else random.choice(["fast"] * 4 + ["sloppy", "muddy", "good"]))
                purse = int(random.choice([15, 20, 25, 40, 60, 80, 100]) * 1000)
                pool = [h for h in horses if h["next_ok"] <= day]
                if len(pool) < 6:
                    continue
                field = random.sample(pool, int(min(len(pool), rng.integers(6, 13))))
                races.append(dict(number=rn, dist=dist, surface=surface, cond=cond,
                                  purse=purse, field=field,
                                  runners=_run_race(rng, field, jockeys, day, dist, surface, cond)))
                for h in field:
                    h["next_ok"] = day + timedelta(days=int(rng.integers(14, 50)))
                    h["last_run"] = day
            if races:
                cards.append({"track": track, "date": day, "races": races})
    return cards, horses, jockeys


def _run_race(rng, field, jockeys, day, dist, surface, cond):
    n = len(field)
    sprint = 1 if dist <= 7 else -1
    wet = cond in {"sloppy", "muddy", "good", "yielding"}
    runners = []
    for post, h in enumerate(field, start=1):
        j = jockeys[int(rng.integers(0, len(jockeys)))]
        layoff = (day - h["last_run"]).days if h["last_run"] else 999
        expected = (8 * h["ability"] + 2.5 * j["skill"] + 2 * h["trainer"]["skill"]
                    + 2.5 * sprint * h["sprinter"] + 2.5 * (1 if surface == "turf" else -1) * h["turf"]
                    + (2.5 * h["wet"] if wet else 0) - 1.5 * (post - 1) / max(n - 1, 1)
                    - (2 if layoff > 120 else 0))
        runners.append({"horse": h, "jockey": j, "post": post, "expected": expected,
                        "perf": expected + rng.normal(0, 7),
                        "weight": int(rng.integers(115, 125))})
    # Market sees expected performance with noise.
    mk = np.array([r["expected"] / 4 + rng.normal(0, 0.35) for r in runners])
    mp = np.exp(mk - mk.max()); mp /= mp.sum()
    ml = np.array([r["expected"] / 4 + rng.normal(0, 0.6) for r in runners])
    mlp = np.exp(ml - ml.max()); mlp /= mlp.sum()
    order = sorted(range(n), key=lambda i: -runners[i]["perf"])
    best = runners[order[0]]["perf"]
    for pos, i in enumerate(order, start=1):
        r = runners[i]
        r.update(finish=pos, fig=int(round(75 + r["perf"])),
                 beaten=round(max(0.0, (best - r["perf"]) / 2.5), 2),
                 final_odds=to_tote(mp[i]), ml=to_ml(mlp[i]))
    return runners


def _starter_el(parent, r, with_result: bool):
    h, j = r["horse"], r["jockey"]
    attrs = {"program": str(r["post"]), "post_position": str(r["post"]),
             "horse_id": h["id"], "horse_name": h["name"], "age": str(h["age"]), "sex": h["sex"],
             "weight_lbs": str(r["weight"]), "jockey_id": j["id"], "jockey_name": j["name"],
             "trainer_id": h["trainer"]["id"], "trainer_name": h["trainer"]["name"],
             "morning_line": r["ml"]}
    if with_result:
        attrs.update(final_odds=r["final_odds"], finish_position=str(r["finish"]),
                     speed_figure=str(r["fig"]), beaten_lengths=str(r["beaten"]))
    ET.SubElement(parent, "starter", attrs)


def _race_el(parent, race):
    return ET.SubElement(parent, "race", {
        "number": str(race["number"]), "distance_furlongs": str(race["dist"]),
        "surface": race["surface"], "condition": race["cond"],
        "class": "allowance" if race["purse"] >= 60000 else "claiming",
        "purse": str(race["purse"])})


def write(cards, out: Path, pp_days: int):
    for sub in ("results", "past_performances", "entries"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for old in (out / sub).glob("SYNTHETIC_*.xml"):
            old.unlink()
    first_day = cards[0]["date"]
    pp_cutoff = first_day + timedelta(days=pp_days)
    pp_root = ET.Element("past_performances", {"generated": str(pp_cutoff)})
    pp_horses: dict[str, ET.Element] = {}

    *history, upcoming = cards
    for card in history:
        if card["date"] < pp_cutoff:
            # Early races exist only as past-performance lines.
            for race in card["races"]:
                for r in race["runners"]:
                    h = r["horse"]
                    if h["id"] not in pp_horses:
                        pp_horses[h["id"]] = ET.SubElement(pp_root, "horse", {"id": h["id"], "name": h["name"]})
                    ET.SubElement(pp_horses[h["id"]], "line", {
                        "date": str(card["date"]), "track": card["track"],
                        "race_number": str(race["number"]), "distance_furlongs": str(race["dist"]),
                        "surface": race["surface"], "condition": race["cond"],
                        "purse": str(race["purse"]), "field_size": str(len(race["runners"])),
                        "post_position": str(r["post"]), "weight_lbs": str(r["weight"]),
                        "jockey_id": r["jockey"]["id"], "trainer_id": h["trainer"]["id"],
                        "final_odds": r["final_odds"], "finish_position": str(r["finish"]),
                        "speed_figure": str(r["fig"]), "beaten_lengths": str(r["beaten"])})
            continue
        root = ET.Element("results", {"track": card["track"], "date": str(card["date"])})
        for race in card["races"]:
            race_el = _race_el(root, race)
            for r in race["runners"]:
                _starter_el(race_el, r, with_result=True)
        _save(root, out / "results" / f"SYNTHETIC_{card['track']}_{card['date']}.xml")
    _save(pp_root, out / "past_performances" / "SYNTHETIC_pp.xml")

    root = ET.Element("entries", {"track": upcoming["track"], "date": str(upcoming["date"])})
    for race in upcoming["races"]:
        race_el = _race_el(root, race)
        for r in race["runners"]:
            _starter_el(race_el, r, with_result=False)
    _save(root, out / "entries" / f"SYNTHETIC_{upcoming['track']}_{upcoming['date']}.xml")


def _save(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--days", type=int, default=300)
    ap.add_argument("--pp-days", type=int, default=60, help="early days written only as PP lines")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    a = ap.parse_args()
    cards, _, _ = simulate(a.seed, date(2025, 9, 5), a.days)
    write(cards, a.out, a.pp_days)
    print(f"wrote {len(cards)} cards to {a.out}")


if __name__ == "__main__":
    main()

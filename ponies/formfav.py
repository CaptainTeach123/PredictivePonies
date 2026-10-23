"""FormFav (formfav.com) race cards + results -> model-ready rows.

FormFav is Australian-focused (also NZ, HK, UK). A race card is a PRE-RACE
snapshot: each runner's career / distance / track / condition / first-up
records, recent form and (Pro tier) speed map and class ratings are already
computed as of that race. Verified: the 2026-10-18 Moe R1 card lists the winner
as a debutant and the runner-up's form string excludes that race. So no
history needs rebuilding and nothing leaks.

A result (Pro tier) gives every runner's finishing position and starting price,
so each race is a complete field for the conditional logit (model.py),
scored against the starting-price market.

Stored raw JSON (data/raw/formfav/, git-ignored: licensed data) holds, per
meeting, {"cards": [get_race_card ...], "results": get_race_results(no race)}.
Judgement calls are tagged [A-39]..[A-41].
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from .features import _shrunk_rate

METRES_PER_FURLONG = 201.168
LBS_PER_KG = 2.20462
_SEX = {"gelding": "G", "mare": "M", "colt": "C", "filly": "F", "horse": "H", "rig": "R"}
_COND_BUCKETS = ("firm", "good", "soft", "heavy", "synthetic", "fast", "slow")

FF_FEATURES: dict[str, str] = {
    "win_rate": "Career win rate, shrunk toward prior [A-03]",
    "place_rate": "Career place rate, shrunk toward prior [A-03]",
    "log_starts": "log(1 + career starts)",
    "distance_win_rate": "Win rate at this distance, shrunk [A-03]",
    "track_win_rate": "Win rate at this track, shrunk [A-03]",
    "condition_win_rate": "Win rate on today's track rating (good/soft/heavy...), shrunk [A-03]",
    "spell_win_rate": "Win rate first-up / second-up when this run is one, shrunk [A-40]",
    "recent_avg_finish": "Average of last 3 finishing positions (0 = 10th+) [A-40]",
    "last_finish": "Last start's finishing position",
    "log_prize_per_start": "log(1 + career prize money per start): class proxy",
    "early_speed": "Early speed index from the speed map (Pro)",
    "settling_position": "Expected settling position from the speed map (Pro)",
    "class_rating": "Current class rating (Pro)",
    "class_difference": "Class rating minus today's race class rating (Pro)",
    "trainer_track_rate": "Trainer win rate at this track, from FormFav's notes [A-41]",
    "jockey_track_rate": "Jockey win rate at this track, from FormFav's notes [A-41]",
    "barrier_rel": "Barrier scaled 0 (inside) to 1 (outside)",
    "weight_vs_field": "Weight carried minus race average (kg, after apprentice claim)",
    "age": "Age in years",
    "debut": "1 if this is the horse's first start",
}
FF_FEATURE_NAMES = list(FF_FEATURES)


def slug(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "-", (s or "").upper()).strip("-")


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _rate(stat: dict | None, key: str = "wins") -> float:
    """[A-03] shrunk rate from a FormFav stats block; NaN when absent."""
    if not stat or not stat.get("starts"):
        return math.nan
    return _shrunk_rate(stat.get(key, 0), stat["starts"])


def furlongs(distance: str | None) -> float:
    m = re.match(r"\s*([\d.]+)\s*m\s*$", distance or "")
    return float(m.group(1)) / METRES_PER_FURLONG if m else math.nan


def surface_for(track: str, condition: str | None) -> str:
    """[A-39] FormFav gives no surface: Polytrack/synthetic tracks or ratings ->
    synthetic, otherwise turf (almost all Australian racing is on turf)."""
    text = f"{track} {condition or ''}".lower()
    return "synthetic" if any(w in text for w in ("poly", "synthetic", "tapeta")) else "turf"


def condition_bucket(condition: str | None) -> str | None:
    word = (condition or "").split(" ")[0].lower()
    return word if word in _COND_BUCKETS else None


def spell_status(last20: str | None) -> str | None:
    """[A-40] Form strings read oldest -> newest; 'X' marks a spell (break).
    Ends with X -> first-up; X then one run -> second-up."""
    s = (last20 or "").upper()
    if s.endswith("X"):
        return "firstUp"
    if len(s) >= 2 and s[-2] == "X":
        return "secondUp"
    return None


def finishes(last20: str | None) -> list[int]:
    """[A-40] Finishing digits, newest last; '0' means 10th or worse -> 10."""
    return [10 if c == "0" else int(c) for c in (last20 or "") if c.isdigit()]


def _decorator_rate(runner: dict, kind: str) -> float:
    """[A-41] FormFav's notes like '22% trainer win rate here (37 starts)'."""
    for d in runner.get("decorators") or []:
        if d.get("type") == kind:
            m = re.search(r"([\d.]+)%", d.get("detail") or "")
            if m:
                return float(m.group(1)) / 100
    return math.nan


def runner_features(runner: dict, card: dict) -> dict:
    stats = runner.get("stats") or {}
    overall = stats.get("overall")
    fins = finishes(runner.get("last20Starts"))
    spell = spell_status(runner.get("last20Starts"))
    starts = (overall or {}).get("starts", 0) or 0
    prize = _num(re.sub(r"[^\d.]", "", runner.get("careerPrizeMoney") or ""))
    speed = runner.get("speedMap") or {}
    cls = runner.get("classProfile") or {}
    fit = runner.get("raceClassFit") or {}
    return {
        "win_rate": _rate(overall),
        "place_rate": _rate(overall, "places"),
        "log_starts": math.log1p(starts),
        "distance_win_rate": _rate(stats.get("distance")),
        "track_win_rate": _rate(stats.get("track")),
        "condition_win_rate": _rate((stats.get("conditions") or {}).get(condition_bucket(card.get("condition")) or "")),
        "spell_win_rate": _rate(stats.get(spell)) if spell else math.nan,
        "recent_avg_finish": sum(fins[-3:]) / len(fins[-3:]) if fins else math.nan,
        "last_finish": fins[-1] if fins else math.nan,
        "log_prize_per_start": math.log1p(prize / starts) if starts and not math.isnan(prize) else math.nan,
        "early_speed": _num(speed.get("earlySpeedIndex")),
        "settling_position": _num(speed.get("settlingPosition")),
        "class_rating": _num(cls.get("currentRating")),
        "class_difference": _num(fit.get("classDifference")),
        "trainer_track_rate": _decorator_rate(runner, "trainer_track"),
        "jockey_track_rate": _decorator_rate(runner, "jockey_track"),
        "age": _num(runner.get("age")),
        "debut": 1.0 if starts == 0 else 0.0,
    }


def race_rows(card: dict, result: dict | None = None) -> list[dict]:
    """One row per runner that started (scratchings dropped [A-21]).

    With ``result``: finish_position, won, and final_odds from the starting
    price (decimal -> odds-to-one). Without: an upcoming race.
    """
    if card.get("abandoned"):
        return []
    placings = {p.get("number"): p for p in (result or {}).get("placings", [])}
    runners = [r for r in card.get("runners", [])
               if not r.get("scratched") and not placings.get(r.get("number"), {}).get("scratched")]
    track = slug(card["track"])
    race_id = f"{track}-{card['date']}-{int(card['raceNumber'])}"
    country = (runners[0].get("country") if runners else "") or "au"
    weights = [(_num(r.get("weight")) - (_num(r.get("claim")) if r.get("claim") else 0.0)) for r in runners]
    mean_w = pd.Series(weights).mean()
    n = len(runners)
    rows = []
    for r, w in zip(runners, weights):
        p = placings.get(r.get("number"), {})
        pos = _num(p.get("position")) if p.get("position") is not None else math.nan
        if result and math.isnan(pos) and p.get("outcome") == "did_not_finish":
            pos = 99.0  # DNF counts as last
        sp = _num(p.get("startingPrice"))
        sex = (r.get("sex") or "").strip()
        rows.append({
            "race_id": race_id, "date": pd.Timestamp(card["date"]), "track": track,
            "race_number": int(card["raceNumber"]),
            "horse_id": f"FF-{slug(country)}-{slug(r['name'])}", "horse_name": r["name"],
            "program": str(r.get("number")), "post_position": _num(r.get("barrier")),
            "jockey_name": r.get("jockey") or None, "trainer_name": r.get("trainer") or None,
            "morning_line": None,  # Pro tier has no pre-race prices
            "final_odds": f"{sp - 1:.2f}" if sp > 1 else None,
            "finish_position": pos,
            "distance_furlongs": furlongs(card.get("distance")),
            "surface": surface_for(card["track"], card.get("condition")),
            "condition": (card.get("condition") or "").lower(),
            "race_class": card.get("raceClass"), "purse": _num(card.get("prizeMoney")),
            "field_size": n, "sex": _SEX.get(sex.lower(), sex[:1].upper()) if sex else None,
            **runner_features(r, card),
            "barrier_rel": (_num(r.get("barrier")) - 1) / (n - 1) if n > 1 else 0.5,
            "weight_vs_field": w - mean_w if not math.isnan(w) else 0.0,
            "won": (float(pos == 1) if result and p.get("position") is not None else math.nan),
        })
    return rows


def load_store(directory: Path) -> pd.DataFrame:
    """All stored meetings -> one DataFrame (completed races + upcoming)."""
    rows = []
    for path in sorted(Path(directory).rglob("*.json")):
        meeting = json.loads(path.read_text())
        results = {r["raceNumber"]: r for r in (meeting.get("results") or {}).get("races", [])}
        for card in meeting.get("cards", []):
            res = results.get(card["raceNumber"])
            final = res if res and res.get("resultStatus") == "final" else None
            rows += race_rows(card, final)
    return pd.DataFrame(rows)

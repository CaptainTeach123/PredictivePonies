"""FormFav card + result conversion. The fixture is trimmed from the real
2026-10-18 Moe R1 card/result shapes (3 runners kept, values as published)."""
import math

from ponies.formfav import race_rows, spell_status, finishes

CARD = {"date": "2026-10-18", "track": "Moe", "raceNumber": 1, "distance": "1217m", "condition": "Good",
        "raceClass": "3YO MDN", "prizeMoney": "32000", "abandoned": False, "runners": [
    {"number": 4, "name": "Nickibi", "jockey": "Zac Spain", "trainer": "Ken & Kasey Keys", "weight": 58,
     "barrier": 10, "age": 3, "sex": "Gelding", "country": "au", "careerPrizeMoney": "$0.00", "scratched": False,
     "decorators": [{"type": "debutant", "detail": "First career start"}]},
    {"number": 6, "name": "Satoshi", "jockey": "Luke Currie", "trainer": "Mick Price", "weight": 58, "barrier": 3,
     "age": 3, "sex": "Gelding", "country": "au", "careerPrizeMoney": "$6345.00", "last20Starts": "0X367",
     "stats": {"overall": {"starts": 4, "wins": 0, "places": 1},
               "conditions": {"good": {"starts": 2, "wins": 0, "places": 1}}},
     "speedMap": {"earlySpeedIndex": 6.25, "settlingPosition": 3.8},
     "classProfile": {"currentRating": 30}, "raceClassFit": {"classDifference": 0},
     "decorators": [{"type": "trainer_track", "detail": "20% trainer win rate here (20 starts)"}]},
    {"number": 11, "name": "Teddy's Trotter", "weight": 56, "claim": 3, "barrier": 14, "scratched": True},
]}
RESULT = {"raceNumber": 1, "resultStatus": "final", "placings": [
    {"number": 4, "position": 1, "startingPrice": 13},
    {"number": 6, "position": 2, "startingPrice": 8},
    {"number": 11, "scratched": True},
]}


def test_rows_from_card_and_result():
    rows = {r["horse_name"]: r for r in race_rows(CARD, RESULT)}
    assert set(rows) == {"Nickibi", "Satoshi"}                 # scratching dropped
    n, s = rows["Nickibi"], rows["Satoshi"]
    assert n["won"] == 1.0 and s["won"] == 0.0 and n["field_size"] == 2
    assert n["final_odds"] == "12.00" and s["final_odds"] == "7.00"   # decimal SP -> odds-to-one
    assert n["debut"] == 1.0 and math.isnan(n["win_rate"])
    assert s["early_speed"] == 6.25 and s["trainer_track_rate"] == 0.2
    assert s["last_finish"] == 7 and s["recent_avg_finish"] == (3 + 6 + 7) / 3
    assert abs(s["distance_furlongs"] - 1217 / 201.168) < 1e-9 and s["surface"] == "turf"


def test_upcoming_race_has_no_outcome():
    rows = race_rows(CARD)
    assert all(math.isnan(r["won"]) and r["final_odds"] is None for r in rows)


def test_form_string_helpers():
    assert spell_status("63X") == "firstUp" and spell_status("0X367") is None and spell_status("7X9") == "secondUp"
    assert finishes("0X367") == [10, 3, 6, 7]

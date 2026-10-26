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


def test_market_price_and_move():
    prices = {"runners": [
        {"number": 4, "prices": [{"marketType": "fixed", "open": 21, "close": 13},
                                 {"marketType": "fixed", "open": 19, "close": 12},
                                 {"marketType": "tote", "close": 30}]},
        {"number": 6, "prices": [{"marketType": "fixed", "open": 6, "close": 8}]}]}
    rows = {r["horse_name"]: r for r in race_rows(CARD, RESULT, prices)}
    assert rows["Nickibi"]["morning_line"] == "12.00"            # median fixed close 13 (tote ignored)
    assert rows["Nickibi"]["price_move"] > 0                      # firmed from 21 to 13
    assert abs(rows["Satoshi"]["price_move"] - math.log(6 / 8)) < 1e-12   # drifted


def test_market_ignores_quotes_after_the_start():
    """[A-42] 'close' can be an in-running quote; replay movements up to the start."""
    from ponies.formfav import market_best, market_odds
    mv = lambda at, px: {"price": px, "at": at}
    prices = {"startTime": "2026-10-22T05:00:00Z", "runners": [{"number": 1, "prices": [
        {"marketType": "fixed", "open": 13, "close": 451,
         "movements": [mv("2026-10-22T04:00:00Z", 10), mv("2026-10-22T04:59:00Z", 13), mv("2026-10-22T05:02:30Z", 451)]},
        {"marketType": "fixed", "open": 10, "close": 12, "movements": [mv("2026-10-22T04:58:00Z", 12)]},
        {"marketType": "fixed", "open": 9, "close": 11, "movements": [mv("2026-10-22T04:30:00Z", 11)]}]}]}
    assert market_odds(prices) == {1: 12.0} and market_best(prices) == {1: 13.0}
    assert market_odds(prices, minutes_before=30) == {1: 10.0}


def test_store_history_uses_earlier_runs_only():
    import pandas as pd
    from ponies.formfav import add_store_history
    df = pd.DataFrame({
        "race_id": ["a", "a", "b", "b", "c", "c"], "horse_id": ["H", "J", "H", "J", "H", "K"],
        "date": pd.to_datetime(["2026-10-01"] * 2 + ["2026-10-08"] * 2 + ["2026-10-20"] * 2),
        "morning_line": ["1.00", "1.00", "3.00", "0.33", "1.0", "1.0"], "field_size": [2] * 6,
        "margin": [0.0, 2.0, 4.0, 0.0, None, None], "won": [1, 0, 0, 1, None, None]})
    out = add_store_history(df)
    h = out[out["horse_id"] == "H"].sort_values("date")
    assert math.isnan(h["prev_margin"].iloc[0]) and h["prev_margin"].iloc[1] == 0.0 and h["prev_margin"].iloc[2] == 4.0
    assert h["days_since_prev"].iloc[2] == 12 and h["prev_won"].iloc[1] == 1
    assert abs(h["prev_market_logp"].iloc[1]) < 1e-12                  # evens in a 2-horse race = average chance
    assert math.isnan(out[out["horse_id"] == "K"]["prev_margin"].iloc[0])


def test_formfav_predictions_only_if_made_before_start():
    from ponies.formfav import formfav_predictions
    pred = {"predictedAt": "2026-10-21T21:33:39Z", "predictions": [{"number": 7, "winProb": 0.2}]}
    assert abs(formfav_predictions(pred, "2026-10-22T05:00:00Z")[7] - math.log(0.2)) < 1e-12
    assert formfav_predictions(pred, "2026-10-21T20:00:00Z") == {}        # made after the start
    assert formfav_predictions({"unavailable": True}, "2026-10-22T05:00:00Z") == {}


def test_race_key_and_code_filter():
    import pandas as pd
    from ponies.formfav import only, race_key
    assert race_key(None, "Moe", "2026-10-18", 1) == "MOE-2026-10-18-1"
    assert race_key({"race_code": "gallops", "country": "au"}, "Moe", "2026-10-18", 1) == "MOE-2026-10-18-1"
    assert race_key({"race_code": "greyhounds", "country": "gb"}, "Romford", "2026-10-18", 3) == "greyhounds-gb-ROMFORD-2026-10-18-3"
    rows = race_rows(CARD, RESULT, meta={"race_code": "harness", "country": "nz"})
    assert rows[0]["race_id"].startswith("harness-nz-") and rows[0]["race_code"] == "harness" and rows[0]["country"] == "nz"
    df = pd.DataFrame(rows + race_rows(CARD, RESULT))
    assert len(only(df)) == 2 and len(only(df, "harness", "nz")) == 2


def _feed_meeting(n_races=80, seed=0):
    """Synthetic greyhound meeting: book 'a' prices each race correctly; book 'x' carries
    the prices of race n-3 (the GB greyhound William Hill fault) [A-67]."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    cards, results, prices, fair = {}, {"races": []}, {}, {}
    for n in range(1, n_races + 1):
        p = rng.dirichlet(np.ones(6) * 1.5)
        fair[n] = 1 / (p * 1.25)
        w = int(rng.choice(6, p=p)) + 1
        start = pd.Timestamp("2026-10-20T12:00:00Z") + pd.Timedelta(minutes=17 * n)
        cards[str(n)] = {"raceNumber": n, "date": "2026-10-20", "track": "Hove",
                         "startTime": start.isoformat().replace("+00:00", "Z"),
                         "runners": [{"number": k, "name": f"d{k}"} for k in range(1, 7)]}
        results["races"].append({"raceNumber": n, "resultStatus": "final",
                                 "placings": [{"number": k, "position": 1 if k == w else 2} for k in range(1, 7)]})
    for n in range(1, n_races + 1):
        at = (pd.Timestamp(cards[str(n)]["startTime"]) - pd.Timedelta(minutes=2)).isoformat().replace("+00:00", "Z")
        src = fair.get(n - 3, fair[n])
        prices[str(n)] = {"runners": [{"number": k, "prices": [
            {"bookmaker": "a", "marketType": "fixed", "movements": [{"price": float(fair[n][k - 1]), "at": at}]},
            {"bookmaker": "x", "marketType": "fixed", "movements": [{"price": float(src[k - 1]), "at": at}]}]}
            for k in range(1, 7)]}
    return {"meeting": {"race_code": "greyhounds", "country": "gb"}, "cards": cards, "results": results, "prices": prices}


def test_misattached_book_is_dropped_and_honest_book_kept():
    from ponies.formfav import book_information, drop_books, misattached_books
    m = _feed_meeting()
    rows = book_information(m)
    bad = misattached_books(rows)
    assert ("greyhounds/gb", "x", "2026-W43") in bad
    assert ("greyhounds/gb", "a", "2026-W43") not in bad
    kept = drop_books(m, bad)
    books = {q["bookmaker"] for pr in kept["prices"].values() for r in pr["runners"] for q in r["prices"]}
    assert books == {"a"}

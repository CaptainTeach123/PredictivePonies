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

Stored raw JSON (data/raw/formfav/, git-ignored: licensed data) is written by
scripts/formfav_collect.py, one file per meeting:
{"cards": {race_no: get_race_card}, "prices": {race_no: get_prices},
 "results": get_race_results(whole meeting)}.
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


def _margin(v) -> float:
    """Lengths behind the winner; FormFav gives a number as text."""
    x = _num(v)
    return min(x, 20.0) if not math.isnan(x) else math.nan


def extra_features(runner: dict, card: dict) -> dict:
    """Candidate inputs [A-45], evaluated by scripts/formfav_experiments.py
    before any is promoted into FF_FEATURES."""
    stats = runner.get("stats") or {}
    cls = runner.get("classProfile") or {}
    fit = runner.get("raceClassFit") or {}
    form = (runner.get("last20Starts") or "").upper()
    prep = form.rsplit("X", 1)[-1] if "X" in form else form
    fins = finishes(form)
    decos = runner.get("decorators") or []
    sent = sum(1 if d.get("sentiment") == "+" else -1 if d.get("sentiment") == "-" else 0 for d in decos)
    style = ((runner.get("speedMap") or {}).get("runningStyle") or "").upper()
    return {
        "place_rate_distance": _rate(stats.get("distance"), "places"),
        "place_rate_track": _rate(stats.get("track"), "places"),
        "place_rate_condition": _rate((stats.get("conditions") or {}).get(condition_bucket(card.get("condition")) or ""), "places"),
        "track_distance_win_rate": _rate(stats.get("trackDistance")),
        "peak_rating_gap": _num(cls.get("peakRating")) - _num(cls.get("currentRating")),
        "best_class_won_rel": _num(cls.get("highestClassWon")) - _num(fit.get("raceClassRating")),
        "in_optimal_range": float(bool(fit.get("withinOptimalRange"))) if fit else math.nan,
        "runs_this_prep": float(sum(c.isdigit() for c in prep)),
        "best_recent_finish": float(min(fins[-3:])) if fins else math.nan,
        "avg_finish_last6": sum(fins[-6:]) / len(fins[-6:]) if fins else math.nan,
        "decorator_sentiment": float(sent),
        "leader_style": 1.0 if style == "L" else 0.0,
        "decorators": ",".join(sorted(f"{d.get('type')}{d.get('sentiment') or ''}" for d in decos)),
    }


def formfav_predictions(pred: dict | None, start: str | None) -> dict:
    """[A-47] FormFav's own win probabilities -> runner number -> log prob.
    Used only when stamped ``predictedAt`` before the scheduled start (so a
    prediction made, or remade, after the race can't leak the result)."""
    if not pred or pred.get("unavailable") or not pred.get("predictions"):
        return {}
    made, begins = _when(pred.get("predictedAt")), _when(start)
    if made is None or begins is None or made >= begins:
        return {}
    return {x.get("number"): math.log(x["winProb"]) for x in pred["predictions"]
            if isinstance(x.get("winProb"), (int, float)) and x["winProb"] > 0}


def _when(ts: str | None) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(ts).tz_convert("UTC") if ts else None
    except (ValueError, TypeError):
        return None


def _quote_before(q: dict, cutoff: pd.Timestamp | None) -> float | None:
    """[A-42] A bookmaker's last fixed price strictly before ``cutoff``.
    FormFav's 'close' can be an in-running or post-race quote (seen: 451 for a
    beaten horse a minute after the start), so with a start time we replay the
    movement history instead. No movement before the cutoff -> None."""
    moves = q.get("movements") or []
    if cutoff is not None and moves:
        before = [m for m in moves if (t := _when(m.get("at"))) is not None and t < cutoff]
        price = before[-1].get("price") if before else None
    elif cutoff is not None and (t := _when(q.get("capturedAt"))) is not None and t >= cutoff:
        price = None  # captured after the cutoff and no history to replay
    else:
        price = q.get("close") or q.get("win")
    return float(price) if isinstance(price, (int, float)) and price > 1 else None


def _quotes(prices: dict | None, minutes_before: float = 0.0) -> dict:
    """runner number -> sorted fixed-odds quotes across bookmakers at
    (scheduled start - minutes_before)."""
    start = _when((prices or {}).get("startTime"))
    cutoff = start - pd.Timedelta(minutes=minutes_before) if start is not None else None
    out = {}
    for r in (prices or {}).get("runners", []):
        qs = [_quote_before(q, cutoff) for q in r.get("prices", [])
              if q.get("marketType", "fixed") == "fixed"]
        qs = sorted(x for x in qs if x is not None)
        if qs:
            out[r.get("number")] = qs
    return out


def market_odds(prices: dict | None, minutes_before: float = 0.0) -> dict:
    """[A-42] Pre-race market price per runner number: the median across
    bookmakers of the last fixed-odds price before the scheduled start (or
    ``minutes_before`` it); the current price for races not yet run. Decimal."""
    return {n: qs[len(qs) // 2] for n, qs in _quotes(prices, minutes_before).items()}


def market_best(prices: dict | None, minutes_before: float = 0.0) -> dict:
    """[A-44] Best (highest) fixed price across bookmakers before the start:
    the price a bettor who shops around actually gets."""
    return {n: qs[-1] for n, qs in _quotes(prices, minutes_before).items()}


def market_open(prices: dict | None) -> dict:
    """[A-43] Opening price per runner: median across bookmakers of the
    fixed-odds 'open'. With market_odds this gives the price move."""
    out = {}
    for r in (prices or {}).get("runners", []):
        quotes = sorted(float(q["open"]) for q in r.get("prices", [])
                        if q.get("marketType", "fixed") == "fixed"
                        and isinstance(q.get("open"), (int, float)) and q["open"] > 1)
        if quotes:
            out[r.get("number")] = quotes[len(quotes) // 2]
    return out


def race_rows(card: dict, result: dict | None = None, prices: dict | None = None,
              predictions: dict | None = None) -> list[dict]:
    """One row per runner that started (scratchings dropped [A-21]).

    With ``result``: finish_position, won, and final_odds from the starting
    price (decimal -> odds-to-one). Without: an upcoming race.
    """
    if card.get("abandoned"):
        return []
    placings = {p.get("number"): p for p in (result or {}).get("placings", [])}
    if prices and not prices.get("startTime") and card.get("startTime"):
        prices = {**prices, "startTime": card["startTime"]}
    market, opening, best = market_odds(prices), market_open(prices), market_best(prices)
    early = market_odds(prices, minutes_before=30)
    books = {n: len(q) for n, q in _quotes(prices).items()}
    ff_pred = formfav_predictions(predictions, card.get("startTime") or (prices or {}).get("startTime"))
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
            # pre-race bookmaker price [A-42], as odds-to-one like the rest of the repo
            "morning_line": f"{market[r.get('number')] - 1:.2f}" if r.get("number") in market else None,
            # [A-43] log(open / close): > 0 = price shortened (money came for it)
            "price_move": (math.log(opening[r.get("number")] / market[r.get("number")])
                           if r.get("number") in market and r.get("number") in opening else math.nan),
            "final_odds": f"{sp - 1:.2f}" if sp > 1 else None,
            "market_best": best.get(r.get("number"), math.nan),       # [A-44] decimal
            "market_t30": early.get(r.get("number"), math.nan),       # 30 min before, decimal
            "n_books": books.get(r.get("number"), 0),
            "start_time": card.get("startTime") or (prices or {}).get("startTime"),
            "margin": _margin(p.get("margin")) if result else math.nan,
            "ff_model_logp": ff_pred.get(r.get("number"), math.nan),   # [A-47]
            "finish_position": pos,
            "distance_furlongs": furlongs(card.get("distance")),
            "surface": surface_for(card["track"], card.get("condition")),
            "condition": (card.get("condition") or "").lower(),
            "race_class": card.get("raceClass"), "purse": _num(card.get("prizeMoney")),
            "field_size": n, "sex": _SEX.get(sex.lower(), sex[:1].upper()) if sex else None,
            **runner_features(r, card),
            **extra_features(r, card),
            "barrier_rel": (_num(r.get("barrier")) - 1) / (n - 1) if n > 1 else 0.5,
            "weight_vs_field": w - mean_w if not math.isnan(w) else 0.0,
            "won": (float(pos == 1) if result and p.get("position") is not None else math.nan),
        })
    return rows


def load_store(directory: Path) -> pd.DataFrame:
    """All stored meetings -> one DataFrame (completed races + upcoming)."""
    rows = []
    for path in sorted(Path(directory).glob("????-??-??/*.json")):  # day folders only
        meeting = json.loads(path.read_text())
        results = {r["raceNumber"]: r for r in (meeting.get("results") or {}).get("races", [])}
        prices = meeting.get("prices", {})
        cards = meeting.get("cards", [])
        for card in (cards.values() if isinstance(cards, dict) else cards):
            res = results.get(card["raceNumber"])
            final = res if res and res.get("resultStatus") == "final" else None
            rows += race_rows(card, final, prices.get(str(card["raceNumber"])),
                              (meeting.get("predictions") or {}).get(str(card["raceNumber"])))
    return add_store_history(pd.DataFrame(rows))


def load_dividends(directory: Path) -> dict[str, list[dict]]:
    """[A-48] race_id -> tote dividends of the final result (winning
    combinations only), from one operator/jurisdiction per race: the first
    listed. Dividends are per $1, runner numbers as text like ``program``."""
    out = {}
    for path in sorted(Path(directory).glob("????-??-??/*.json")):
        meeting = json.loads(path.read_text())
        cards = meeting.get("cards", {})
        for res in (meeting.get("results") or {}).get("races", []):
            if res.get("resultStatus") != "final" or not res.get("dividends"):
                continue
            card = (cards.get(str(res["raceNumber"])) if isinstance(cards, dict) else None) or res
            race_id = f"{slug(card.get('track') or res.get('track'))}-{card.get('date') or res.get('date')}-{int(res['raceNumber'])}"
            first = res["dividends"][0]
            key = (first.get("operator"), first.get("jurisdiction"))
            out[race_id] = [{"pool": d.get("pool"), "combination": [str(x) for x in d.get("combination") or []],
                             "dividend": float(d["dividend"])}
                            for d in res["dividends"]
                            if (d.get("operator"), d.get("jurisdiction")) == key
                            and isinstance(d.get("dividend"), (int, float))]
    return out


STORE_HISTORY = ("prev_margin", "prev_market_logp", "prev_won", "days_since_prev")


def add_store_history(df: pd.DataFrame) -> pd.DataFrame:
    """[A-46] Each runner's previous run that is also in the store (strictly
    earlier date): lengths beaten, the market's view then (log of field size x
    implied win probability: 0 = an average chance) and days since. Missing
    when the horse hasn't run in the stored window."""
    if df.empty:
        return df
    df = df.copy()
    dec = df["morning_line"].map(lambda v: _num(v) + 1)
    inv = 1 / dec
    p = inv / inv.groupby(df["race_id"]).transform("sum")
    done = df["won"].notna()
    df["_mk"] = (p * df["field_size"]).map(lambda x: math.log(x) if x > 0 else math.nan).where(done)
    df["_mg"] = df["margin"].where(done)
    df["_won"] = df["won"].where(done)
    df["_dt"] = df["date"].where(done)
    df = df.sort_values(["horse_id", "date", "race_id"])
    g = df.groupby("horse_id")
    # last completed run on an earlier date (same-day duplicates can't leak:
    # shift then forward-fill only carries values from earlier rows)
    for src, dst in (("_mg", "prev_margin"), ("_mk", "prev_market_logp"), ("_won", "prev_won"), ("_dt", "_prev_dt")):
        df[dst] = g[src].transform(lambda s: s.shift(1).ffill())
    same_day = df["_prev_dt"] == df["date"]
    df.loc[same_day, ["prev_margin", "prev_market_logp", "prev_won", "_prev_dt"]] = math.nan
    df["days_since_prev"] = (df["date"] - df["_prev_dt"]).dt.days
    return df.drop(columns=["_mk", "_mg", "_won", "_dt", "_prev_dt"]).sort_index()

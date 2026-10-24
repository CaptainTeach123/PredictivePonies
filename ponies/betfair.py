"""Betfair Exchange historical data (historicdata.betfair.com) -> one row per runner.

Each downloaded file (``*.bz2``) is one market in Betfair's Stream API format:
one JSON message per line, ``{"op": "mcm", "pt": <ms>, "mc": [{"id": market,
"marketDefinition": {...}, "rc": [{"id": runner, "ltp": price}]}]}``.
The BASIC plan gives last-traded prices (ltp) at one-minute intervals; the
final market definition gives each runner's result status and, where the
market used it, the Betfair Starting Price (bsp).

Judgement calls are tagged [A-53]..[A-54]. Files are licensed: they live in
data/raw/betfair/ (git-ignored) and are never committed or published.
"""

from __future__ import annotations

import bz2
import json
import math
import re
from pathlib import Path

import pandas as pd


def _open(path: Path):
    path = Path(path)
    return bz2.open(path, "rt") if path.suffix == ".bz2" else path.open()


def _ts(ms: int) -> pd.Timestamp:
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def parse_market(lines) -> list[dict]:
    """Stream lines -> runner rows for every market in them (a file can hold
    one market, or a whole event's markets). Removed runners are dropped.

    [A-53] Pre-race price = the last traded price before the market first
    turned in-play (or, if it never did, before the scheduled start);
    ``ltp_t30`` = the last traded price 30 minutes before the start."""
    markets: dict[str, dict] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        pt = msg.get("pt")
        for mc in msg.get("mc", []):
            m = markets.setdefault(mc.get("id"), {"definition": None, "in_play": None, "history": {}})
            md = mc.get("marketDefinition")
            if md:
                m["definition"] = md
                if md.get("inPlay") and m["in_play"] is None:
                    m["in_play"] = pt
            for rc in mc.get("rc", []):
                if isinstance(rc.get("ltp"), (int, float)):
                    m["history"].setdefault(rc["id"], []).append((pt, float(rc["ltp"])))
    rows = []
    for market_id, m in markets.items():
        rows += _market_rows(market_id, m["definition"], m["in_play"], m["history"])
    return rows


def _market_rows(market_id, definition, went_in_play, history) -> list[dict]:
    if not definition:
        return []
    start = pd.Timestamp(definition.get("marketTime")).tz_convert("UTC") if definition.get("marketTime") else None
    start_ms = int(start.timestamp() * 1000) if start is not None else None
    cutoff = went_in_play if went_in_play is not None else start_ms
    t30 = start_ms - 30 * 60 * 1000 if start_ms is not None else None

    def last_before(series, limit):
        if limit is None:
            return math.nan
        before = [p for t, p in series if t < limit]
        return before[-1] if before else math.nan

    runners = [r for r in definition.get("runners", []) if r.get("status") != "REMOVED"]
    winners = sum(r.get("status") == "WINNER" for r in runners)
    settled = definition.get("status") == "CLOSED" and winners > 0
    rows = []
    for r in runners:
        series = history.get(r["id"], [])
        name = r.get("name") or str(r["id"])
        number = re.match(r"\s*(\d+)\.\s*", name)
        rows.append({
            "market_id": market_id, "event_id": definition.get("eventId"),
            "venue": definition.get("venue"), "country": definition.get("countryCode"),
            "market_type": definition.get("marketType"), "market_name": definition.get("name"),
            "start": start, "date": start.tz_localize(None).normalize() if start is not None else pd.NaT,
            "n_winners": definition.get("numberOfWinners"), "field_size": len(runners),
            "selection_id": r["id"], "program": number.group(1) if number else str(r.get("sortPriority", "")),
            "horse_name": name[number.end():] if number else name,
            "ltp_pre": last_before(series, cutoff), "ltp_t30": last_before(series, t30),
            "bsp": float(r["bsp"]) if isinstance(r.get("bsp"), (int, float)) and r["bsp"] > 1 else math.nan,
            "won": (1.0 if r.get("status") == "WINNER" else 0.0) if settled else math.nan,
            "dead_heat_share": (1.0 / winners if r.get("status") == "WINNER" and winners else 0.0) if settled else math.nan,
        })
    return rows


def load_files(directory: Path, pattern: str = "**/*.bz2") -> pd.DataFrame:
    rows = []
    for path in sorted(Path(directory).glob(pattern)):
        with _open(path) as fh:
            rows += parse_market(fh)
    return pd.DataFrame(rows)


def net_return(price, won, commission: float):
    """[A-54] Back bet of 1 unit settled on the exchange: commission is charged
    on net winnings only (losing bets pay none)."""
    return (price - 1) * (1 - commission) if won else -1.0

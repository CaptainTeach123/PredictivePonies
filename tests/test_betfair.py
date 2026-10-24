"""Betfair Stream-format parsing [A-53] and exchange settlement [A-54]."""
import bz2
import json
import math

from ponies.betfair import load_files, net_return, parse_market

START = "2026-10-22T05:00:00.000Z"
T = 1792645200000  # 2026-10-22T05:00:00Z in ms


def _md(in_play=False, status="OPEN", result=None):
    runners = [{"id": 1, "name": "1. Alpha", "sortPriority": 1, "status": "ACTIVE"},
               {"id": 2, "name": "2. Bravo", "sortPriority": 2, "status": "ACTIVE"},
               {"id": 3, "name": "3. Charlie", "sortPriority": 3, "status": "REMOVED"}]
    for r in runners:
        if result and r["status"] != "REMOVED":
            r["status"], r["bsp"] = result[r["id"]]
    return {"marketTime": START, "marketType": "WIN", "venue": "Moe", "countryCode": "AU", "eventId": "9",
            "name": "R1 1200m", "numberOfWinners": 1, "inPlay": in_play, "status": status, "runners": runners}


def _lines():
    msgs = [
        {"op": "mcm", "pt": T - 3_600_000, "mc": [{"id": "1.1", "marketDefinition": _md()}]},
        {"op": "mcm", "pt": T - 2_400_000, "mc": [{"id": "1.1", "rc": [{"id": 1, "ltp": 3.0}, {"id": 2, "ltp": 1.6}]}]},
        {"op": "mcm", "pt": T - 60_000, "mc": [{"id": "1.1", "rc": [{"id": 1, "ltp": 2.8}, {"id": 2, "ltp": 1.7}]}]},
        {"op": "mcm", "pt": T + 30_000, "mc": [{"id": "1.1", "marketDefinition": _md(in_play=True)}]},
        {"op": "mcm", "pt": T + 40_000, "mc": [{"id": "1.1", "rc": [{"id": 1, "ltp": 1.01}, {"id": 2, "ltp": 200}]}]},
        {"op": "mcm", "pt": T + 300_000, "mc": [{"id": "1.1", "marketDefinition": _md(
            True, "CLOSED", {1: ("WINNER", 2.9), 2: ("LOSER", 1.65)})}]},
    ]
    return [json.dumps(m) for m in msgs]


def test_pre_race_price_ignores_in_play_trades(tmp_path):
    rows = {r["horse_name"]: r for r in parse_market(_lines())}
    assert set(rows) == {"Alpha", "Bravo"}                          # removed runner dropped
    a, b = rows["Alpha"], rows["Bravo"]
    assert a["ltp_pre"] == 2.8 and b["ltp_pre"] == 1.7              # not the in-play 1.01 / 200
    assert a["ltp_t30"] == 3.0 and a["bsp"] == 2.9 and a["program"] == "1"
    assert a["won"] == 1.0 and b["won"] == 0.0 and a["field_size"] == 2
    p = tmp_path / "1.1.bz2"
    p.write_bytes(bz2.compress("\n".join(_lines()).encode()))
    assert len(load_files(tmp_path)) == 2


def test_unsettled_market_has_no_result():
    rows = parse_market(_lines()[:3])
    assert all(math.isnan(r["won"]) for r in rows)


def test_commission_on_net_winnings_only():
    assert abs(net_return(3.0, True, 0.08) - 1.84) < 1e-12 and net_return(3.0, False, 0.08) == -1.0

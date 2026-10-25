"""Late scratchings: do bookmakers fully re-price the remaining runners?

    python scripts/formfav_scratchings.py [--store data/raw/formfav]

McAlvanah & Moul (JEBO 2013) found Australian bookmakers anchor on the old
prices after a late scratching and recover only ~80% of the lost margin. Here,
for every scratched runner whose price feed stopped before the start [A-61]:
  t0 = the last time any bookmaker quoted it (the scratching time, approx.)
  before: each book's prices just before t0 (scratched runner included)
  after:  the same book's prices W minutes after t0 (W = 5, 15, 30, 60, jump)
Recovery = (sum of 1/price over the remaining runners after - before) divided
by the scratched runner's implied share before. 1 = fully re-priced, 0 = not
touched. Fair probabilities for the remaining runners are the de-margined
pre-scratching consensus renormalised without the scratched runner (Harville),
and a bet backs the best held price W minutes after t0 when it beats that fair
price. ROI, SE and halves for every cell. Prints summary numbers only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ponies import config, odds  # noqa: E402
from ponies.formfav import _when, effective_start, race_key  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from formfav_price_shopping import book_quotes, cell, summary  # noqa: E402

WINDOWS = (5, 15, 30, 60)


def last_quote_time(runner_prices: list[dict]):
    times = [_when(m.get("at")) for q in runner_prices if q.get("marketType", "fixed") == "fixed"
             for m in (q.get("movements") or [])]
    times = [t for t in times if t is not None]
    return max(times) if times else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path(config.RAW_DIR / "formfav"))
    a = ap.parse_args()
    events, bets = [], []
    races_seen = races_with_scratching = 0
    for path in sorted(a.store.glob("????-??-??/*.json")):
        meeting = json.loads(path.read_text())
        meta = meeting.get("meeting") or {}
        results = {r["raceNumber"]: r for r in (meeting.get("results") or {}).get("races", [])
                   if r.get("resultStatus") == "final"}
        cards = meeting.get("cards", {})
        for n, card in (cards.items() if isinstance(cards, dict) else ((str(c["raceNumber"]), c) for c in cards)):
            res = results.get(card["raceNumber"])
            prices = (meeting.get("prices") or {}).get(str(card["raceNumber"]))
            if not res or not prices or card.get("abandoned"):
                continue
            start = _when(card.get("startTime") or prices.get("startTime"))
            if start is None:
                continue
            start, _shift = effective_start(prices, start)  # [A-64]
            races_seen += 1
            placings = {p.get("number"): p for p in res.get("placings", [])}
            by_num = {r.get("number"): r for r in prices.get("runners", [])}
            scratched = [r for r in card.get("runners", []) if r.get("scratched")
                         or placings.get(r.get("number"), {}).get("scratched")]
            remaining = [r for r in card.get("runners", []) if r not in scratched
                         and placings.get(r.get("number"), {}).get("position") is not None]
            if not scratched or len(remaining) < 2 or sum(placings[r["number"]]["position"] == 1 for r in remaining) != 1:
                continue
            race_id = race_key(meta, card["track"], card["date"], card["raceNumber"])
            any_event = False
            for s in scratched:
                q = (by_num.get(s.get("number")) or {}).get("prices", [])
                t0 = last_quote_time(q)
                if t0 is None or t0 >= start - pd.Timedelta(minutes=2) or t0 < start - pd.Timedelta(hours=24):
                    continue
                before = book_quotes(prices, t0 + pd.Timedelta(seconds=30), 1)   # one minute before t0
                if s["number"] not in before or any(r["number"] not in before for r in remaining):
                    continue
                books = set.intersection(*(set(before[r["number"]]) for r in remaining + [s]))
                if not books:
                    continue
                # fair for the remaining runners: de-margined consensus before the scratching, renormalised
                med = [float(np.median([before[r["number"]][b] for b in books])) for r in remaining + [s]]
                fair_all = odds.shin_probabilities(np.array(med))
                fair = fair_all[:-1] / (1 - fair_all[-1])
                any_event = True
                for w in WINDOWS + ("jump",):
                    at = start if w == "jump" else t0 + pd.Timedelta(minutes=w)
                    if at >= start:
                        at = start
                    after = book_quotes(prices, at)
                    held = book_quotes(prices, at + pd.Timedelta(minutes=5)) if w != "jump" else after
                    rec = []
                    for b in books:
                        if all(r["number"] in after and b in after[r["number"]] for r in remaining):
                            pre_rem = sum(1 / before[r["number"]][b] for r in remaining)
                            post_rem = sum(1 / after[r["number"]][b] for r in remaining)
                            share = 1 / before[s["number"]][b]
                            rec.append((post_rem - pre_rem) / share)
                    best_sum = sum(1 / max(after[r["number"]].values()) for r in remaining
                                   if r["number"] in after) if all(r["number"] in after for r in remaining) else math.nan
                    events.append({"race_id": race_id, "date": pd.Timestamp(card["date"]), "window": str(w),
                                   "group": f"{meta.get('race_code') or 'gallops'}/{meta.get('country') or 'au'}",
                                   "scratched_share": float(fair_all[-1]), "minutes_before_start": (start - t0).total_seconds() / 60,
                                   "recovery": float(np.mean(rec)) if rec else math.nan, "books": len(rec),
                                   "best_booksum_after": best_sum})
                    for i, r in enumerate(remaining):
                        q_now, q_held = after.get(r["number"], {}), held.get(r["number"], {})
                        hb = [min(q_now[b], q_held[b]) for b in q_now if b in q_held]
                        if not hb:
                            continue
                        bets.append({"race_id": race_id, "date": pd.Timestamp(card["date"]), "window": str(w),
                                     "group": f"{meta.get('race_code') or 'gallops'}/{meta.get('country') or 'au'}",
                                     "won": float(placings[r["number"]]["position"] == 1),
                                     "best": max(hb), "fair": float(fair[i]), "scratched_share": float(fair_all[-1])})
            races_with_scratching += any_event
    ev, bt = pd.DataFrame(events), pd.DataFrame(bets)
    print(json.dumps({"races": races_seen, "races_with_a_late_scratching_and_prices": races_with_scratching,
                      "scratching_events": int(ev["race_id"].nunique()) if len(ev) else 0}))
    if ev.empty:
        return 0
    first = ev[ev["window"] == "5"]
    print(json.dumps({"scratched_share_quartiles": [round(float(x), 3) for x in first["scratched_share"].quantile([.25, .5, .75])],
                      "minutes_before_start_quartiles": [round(float(x), 1) for x in first["minutes_before_start"].quantile([.25, .5, .75])]}))
    for w, g in ev.groupby("window", sort=False):
        print(json.dumps({"window_minutes_after_scratching": w, "events_with_full_books": int(g["recovery"].notna().sum()),
                          "recovery_mean": round(float(g["recovery"].mean()), 3) if g["recovery"].notna().any() else None,
                          "recovery_quartiles": [round(float(x), 3) for x in g["recovery"].dropna().quantile([.25, .5, .75])] if g["recovery"].notna().any() else None,
                          "share_best_booksum_below_1": round(float((g["best_booksum_after"] < 1).mean()), 4),
                          "best_booksum_median": round(float(g["best_booksum_after"].median()), 4)}))
    if bt.empty:
        return 0
    days = sorted(bt["date"].unique())
    mid = days[len(days) // 2]
    for w, g in bt.groupby("window", sort=False):
        for k in (0.0, 0.05, 0.1):
            m = (bt["window"] == w) & (bt["best"] * bt["fair"] > 1 + k)
            print(json.dumps({"test": "remaining_runners_after_scratching", "window": w, "k": k, **cell(bt, m, "best", mid),
                              "ev_implied": round(float((bt[m]["best"] * bt[m]["fair"] - 1).mean()), 4) if m.any() else None}))
        m = (bt["window"] == w) & (bt["scratched_share"] >= 0.1) & (bt["best"] * bt["fair"] > 1)
        print(json.dumps({"test": "remaining_runners_after_big_scratching_share>=10%", "window": w, **cell(bt, m, "best", mid)}))
    for grp, g in bt[bt["window"] == "15"].groupby("group"):
        m = (bt["window"] == "15") & (bt["group"] == grp) & (bt["best"] * bt["fair"] > 1)
        print(json.dumps({"test": "by_group_window_15", "group": grp, **cell(bt, m, "best", mid)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

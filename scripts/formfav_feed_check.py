"""Is each bookmaker's price feed attached to the right race and runner? [A-66]

    python scripts/formfav_feed_check.py [--store data/raw/formfav]

The results feed carries each runner's official starting price (SP), attached
by runner number independently of the bookmaker feeds. For every code/country
and bookmaker, using that book's last price before the jump [A-42][A-64]:

  fav_strike        how often the book's own favourite won (races where the
                    book priced every runner), next to 1/field and to the SP
                    favourite's strike in the same races. A correctly attached
                    book's favourite wins far more often than 1/field.
  loglik_gain       the book's de-margined probabilities, mean log likelihood
                    of the winner minus log(1/field), per race. About 0 means
                    the prices carry no information about this race.
  corr_with_sp      correlation of log price with log official SP, same runner.
  offset scan       the same correlation against race N+d of the same meeting
                    file (d = -15..15) and against runner positions rotated or
                    reversed within the race. A peak away from d = 0 / no
                    rotation says the prices belong to another race or trap.
  timing            minutes from the book's last price change to the jump.

Also: duplicate runner numbers (GB greyhound reserves share a trap number),
and whether the card, price and result runner lists agree. Prints summary
numbers only (safe for public CI logs).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ponies import config  # noqa: E402
from ponies.formfav import _quote_before, _when, effective_start  # noqa: E402

OFFSETS = range(-15, 16)


def shin_free(prices: list[float]) -> np.ndarray:
    p = 1 / np.asarray(prices, float)
    return p / p.sum()


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 30 or x.std() == 0 or y.std() == 0:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 3)


def races_in(meeting: dict) -> dict:
    """race number -> dict(start, runners [numbers in card order], winner, sp {num: sp},
    books {book: {num: price}}, last_move {book: minutes from last change to jump}, flags)."""
    meta = meeting.get("meeting") or {}
    results = {r["raceNumber"]: r for r in (meeting.get("results") or {}).get("races", [])
               if r.get("resultStatus") == "final"}
    cards = meeting.get("cards", {})
    out = {}
    for _, card in (cards.items() if isinstance(cards, dict) else ((str(c["raceNumber"]), c) for c in cards)):
        n = card.get("raceNumber")
        res, prices = results.get(n), (meeting.get("prices") or {}).get(str(n))
        if not res or not prices or card.get("abandoned"):
            continue
        start = _when(card.get("startTime") or prices.get("startTime"))
        if start is None:
            continue
        start, _ = effective_start(prices, start)
        placings = {p.get("number"): p for p in res.get("placings", [])}
        card_nums = [r.get("number") for r in card.get("runners", [])]
        price_nums = [r.get("number") for r in prices.get("runners", [])]
        runners = [r.get("number") for r in card.get("runners", []) if not r.get("scratched")
                   and not placings.get(r.get("number"), {}).get("scratched")
                   and placings.get(r.get("number"), {}).get("position") is not None]
        winners = [k for k in runners if placings[k]["position"] == 1]
        books, last_move = defaultdict(dict), {}
        names_card = {r.get("number"): (r.get("name") or "").strip().lower() for r in card.get("runners", [])
                      if not r.get("scratched")}
        name_mismatch = 0
        for r in prices.get("runners", []):
            k = r.get("number")
            nm = (r.get("name") or "").strip().lower()
            if not r.get("scratched") and k in names_card and nm and names_card[k] and nm != names_card[k]:
                name_mismatch += 1
            for q in r.get("prices", []):
                if q.get("marketType", "fixed") != "fixed":
                    continue
                b = q.get("bookmaker") or "?"
                px = _quote_before(q, start)
                if px and px > 1 and k in runners:
                    books[b][k] = px
                ts = [t for m in (q.get("movements") or []) if (t := _when(m.get("at"))) is not None]
                if ts:
                    last = (max(ts) - start).total_seconds() / 60
                    last_move[b] = max(last_move.get(b, -1e9), last)
        out[n] = {"group": f"{meta.get('race_code') or 'gallops'}/{meta.get('country') or 'au'}",
                  "runners": runners, "winner": winners[0] if len(winners) == 1 else None,
                  "sp": {k: float(placings[k]["startingPrice"]) for k in runners
                         if (placings[k].get("startingPrice") or 0) > 1},
                  "books": dict(books), "last_move": last_move,
                  "dup_numbers": len(card_nums) != len(set(card_nums)) or len(price_nums) != len(set(price_nums)),
                  "name_mismatch": name_mismatch,
                  "result_not_on_card": sum(1 for k in placings if k not in set(card_nums))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path(config.RAW_DIR / "formfav"))
    a = ap.parse_args()
    files = []
    for path in sorted(a.store.glob("????-??-??/*.json")):
        files.append(races_in(json.loads(path.read_text())))
    if not any(files):
        print("no races with results and prices")
        return 0

    # per group and book
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    grp_acc = defaultdict(lambda: defaultdict(list))
    for f in files:
        for n, r in f.items():
            g, runners, w = r["group"], r["runners"], r["winner"]
            ga = grp_acc[g]
            ga["races"].append(1)
            ga["dup_numbers"].append(r["dup_numbers"])
            ga["name_mismatch_races"].append(r["name_mismatch"] > 0)
            ga["result_not_on_card"].append(r["result_not_on_card"] > 0)
            sp_full = len(r["sp"]) == len(runners) and len(runners) >= 2
            ga["sp_full"].append(sp_full)
            if sp_full and w is not None:
                sp_fav = min(runners, key=lambda k: (r["sp"][k], k))
                ga["sp_fav_won"].append(sp_fav == w)
                ga["sp_book"].append(sum(1 / v for v in r["sp"].values()))
                ga["inv_field"].append(1 / len(runners))
                p = shin_free([r["sp"][k] for k in runners])
                ga["sp_loglik_gain"].append(math.log(p[runners.index(w)]) - math.log(1 / len(runners)))
            for b, q in r["books"].items():
                ba = acc[g][b]
                ba["quotes"].append(len(q))
                if b in r["last_move"]:
                    ba["last_move_minus_jump"].append(r["last_move"][b])
                for k, px in q.items():
                    if k in r["sp"]:
                        ba["x"].append(math.log(px))
                        ba["y"].append(math.log(r["sp"][k]))
                if len(q) == len(runners) and len(runners) >= 2 and w is not None:
                    fav = min(runners, key=lambda k: (q[k], k))
                    ba["fav_won"].append(fav == w)
                    ba["inv_field"].append(1 / len(runners))
                    p = shin_free([q[k] for k in runners])
                    ba["loglik_gain"].append(math.log(p[runners.index(w)]) - math.log(1 / len(runners)))
                    ba["book_pct"].append(sum(1 / v for v in q.values()))
                    if sp_full:
                        sp_fav = min(runners, key=lambda k: (r["sp"][k], k))
                        ba["sp_fav_won_same_races"].append(sp_fav == w)

    for g in sorted(grp_acc):
        ga = grp_acc[g]
        m = lambda v: round(float(np.mean(v)), 4) if len(v) else None
        print(json.dumps({"test": "results_feed", "group": g, "races": len(ga["races"]),
                          "share_dup_runner_numbers": m(ga["dup_numbers"]),
                          "share_price_names_differ_from_card": m(ga["name_mismatch_races"]),
                          "share_result_runner_not_on_card": m(ga["result_not_on_card"]),
                          "share_full_sp": m(ga["sp_full"]),
                          "sp_fav_strike": m(ga["sp_fav_won"]), "mean_inv_field": m(ga["inv_field"]),
                          "sp_book_pct_median": round(float(np.median(ga["sp_book"])), 4) if ga["sp_book"] else None,
                          "sp_loglik_gain_per_race": m(ga["sp_loglik_gain"])}))
        books = {}
        for b, ba in sorted(acc[g].items(), key=lambda kv: -len(kv[1]["quotes"])):
            if len(ba["quotes"]) < 50:
                continue
            fw = ba["fav_won"]
            lm = np.array(ba["last_move_minus_jump"]) if ba["last_move_minus_jump"] else np.array([])
            books[b] = {"races_quoted": len(ba["quotes"]), "races_full_book": len(fw),
                        "fav_strike": m(fw), "mean_inv_field": m(ba["inv_field"]),
                        "sp_fav_strike_same_races": m(ba["sp_fav_won_same_races"]),
                        "fav_strike_se": round(float(np.std(fw, ddof=1) / math.sqrt(len(fw))), 4) if len(fw) > 1 else None,
                        "loglik_gain_per_race": m(ba["loglik_gain"]),
                        "book_pct_median": round(float(np.median(ba["book_pct"])), 4) if ba["book_pct"] else None,
                        "corr_log_price_with_log_sp": corr(ba["x"], ba["y"]),
                        "last_change_minus_jump_minutes_q10_q50_q90":
                            [round(float(v), 1) for v in np.quantile(lm, [.1, .5, .9])] if len(lm) else None}
        print(json.dumps({"test": "book_attachment", "group": g, "books": books}))

    # offset scans for every book: race N+d of the same meeting file, and runner rotations within the race
    for g in sorted(grp_acc):
        scans = {}
        for b in acc[g]:
            if len(acc[g][b]["quotes"]) < 200:
                continue
            by_d = {}
            for d in OFFSETS:
                x, y = [], []
                for f in files:
                    for n, r in f.items():
                        if r["group"] != g or b not in r["books"]:
                            continue
                        other = f.get(n + d)
                        if not other:
                            continue
                        for i, k in enumerate(r["runners"]):
                            # same trap position in the other race
                            if k in r["books"][b] and i < len(other["runners"]) and other["runners"][i] in other["sp"]:
                                x.append(math.log(r["books"][b][k]))
                                y.append(math.log(other["sp"][other["runners"][i]]))
                c = corr(x, y)
                if c is not None:
                    by_d[d] = (c, len(x))
            rot = {}
            for s in ("rot1", "rot2", "rot-1", "reverse"):
                x, y = [], []
                for f in files:
                    for r in f.values():
                        if r["group"] != g or b not in r["books"]:
                            continue
                        ks = r["runners"]
                        L = len(ks)
                        for i, k in enumerate(ks):
                            j = {"rot1": (i + 1) % L, "rot2": (i + 2) % L, "rot-1": (i - 1) % L, "reverse": L - 1 - i}[s]
                            if k in r["books"][b] and ks[j] in r["sp"]:
                                x.append(math.log(r["books"][b][k]))
                                y.append(math.log(r["sp"][ks[j]]))
                rot[s] = corr(x, y)
            if by_d:
                best = max(by_d, key=lambda d: by_d[d][0])
                scans[b] = {"corr_same_race": by_d.get(0, (None, 0))[0], "best_offset": best,
                            "corr_at_best_offset": by_d[best][0], "pairs_at_best_offset": by_d[best][1],
                            "corr_by_offset_-3_to_3": {str(d): by_d[d][0] for d in range(-3, 4) if d in by_d},
                            "runner_rotations": rot}
        print(json.dumps({"test": "offset_scan_vs_official_sp", "group": g, "books": scans}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Price shopping and closing-line value on FormFav bookmaker prices.

    python scripts/formfav_price_shopping.py [--store data/raw/formfav]

Nothing is fitted, so every completed race with results and prices counts.
For each runner the bookmakers' individual fixed prices are replayed from
their movement histories [A-42] at the jump and at 30 / 60 / 120 minutes
before the scheduled start. "Fair" win probability = Shin de-margining of the
median price at the jump [A-51], or the average of each bookmaker's own
de-margined book [A-55].

Tests (1-unit bets, settled at the price taken, no commission):
  1. best price at the jump when best x fair > 1 + k         (price shopping)
  2. best price T minutes before the jump when it beats the fair price
     implied by the market AT the jump                       (closing-line value)
  3. pure dispersion: best / median >= r
  4. arbitrage: races where sum(1 / best price) < 1
  5. per bookmaker: taking its price whenever it is the best price
Every cell reports ROI, SE and the two halves of the period. Prints summary
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ponies import config, odds  # noqa: E402
from ponies.formfav import _quote_before, _when, race_key  # noqa: E402

EARLY = (30, 60, 120)


def book_quotes(prices: dict | None, start, minutes_before: float = 0.0) -> dict:
    """runner number -> {bookmaker: price} at (start - minutes_before)."""
    cutoff = start - pd.Timedelta(minutes=minutes_before) if start is not None else None
    out = {}
    for r in (prices or {}).get("runners", []):
        qs = {}
        for q in r.get("prices", []):
            if q.get("marketType", "fixed") != "fixed":
                continue
            p = _quote_before(q, cutoff)
            if p is not None:
                qs[q.get("bookmaker") or "?"] = p
        if qs:
            out[r.get("number")] = qs
    return out


def shin(prices: list[float]) -> np.ndarray:
    return odds.shin_probabilities(np.asarray(prices, dtype=float))


def race_rows(path: Path, meeting: dict) -> list[dict]:
    results = {r["raceNumber"]: r for r in (meeting.get("results") or {}).get("races", [])
               if r.get("resultStatus") == "final"}
    cards = meeting.get("cards", {})
    rows = []
    for n, card in (cards.items() if isinstance(cards, dict) else ((str(c["raceNumber"]), c) for c in cards)):
        res = results.get(card["raceNumber"])
        prices = (meeting.get("prices") or {}).get(str(card["raceNumber"]))
        if not res or not prices or card.get("abandoned"):
            continue
        start = _when(card.get("startTime") or prices.get("startTime"))
        if start is None:
            continue
        placings = {p.get("number"): p for p in res.get("placings", [])}
        runners = [r for r in card.get("runners", []) if not r.get("scratched")
                   and not placings.get(r.get("number"), {}).get("scratched")
                   and placings.get(r.get("number"), {}).get("position") is not None]
        if len(runners) < 2 or sum(placings[r["number"]]["position"] == 1 for r in runners) != 1:
            continue
        jump = book_quotes(prices, start)
        if any(r["number"] not in jump for r in runners):
            continue  # need a price for every runner to de-margin the field
        early = {m: book_quotes(prices, start, m) for m in EARLY}
        # [A-57] a price that was also there 5 minutes earlier (not a fleeting or glitchy quote)
        persist = {m: book_quotes(prices, start, m + 5) for m in (0,) + EARLY}
        median = [float(np.median(list(jump[r["number"]].values()))) for r in runners]
        best = [max(jump[r["number"]].values()) for r in runners]
        fair_median = shin(median)
        # [A-55] each bookmaker with a full book, de-margined on its own, then averaged
        books = set.intersection(*(set(jump[r["number"]]) for r in runners))
        own = {b: shin([jump[r["number"]][b] for r in runners]) for b in books}  # each book de-margined alone
        per_book = list(own.values())
        fair_books = np.mean(per_book, axis=0) if per_book else fair_median
        books30 = set.intersection(*(set(early[30].get(r["number"], {})) for r in runners))
        own30 = {b: shin([early[30][r["number"]][b] for r in runners]) for b in books30}
        meta = meeting.get("meeting") or {}
        race_id = race_key(meta, card["track"], card["date"], card["raceNumber"])
        fair_early = {}
        for m in EARLY:  # consensus at T, using only prices known at T
            meds = [float(np.median(list(early[m][r["number"]].values()))) if r["number"] in early[m] else math.nan
                    for r in runners]
            fair_early[m] = shin(meds) if not any(math.isnan(x) for x in meds) else np.full(len(runners), math.nan)
        for i, r in enumerate(runners):
            n_ = r["number"]
            best_book = max(jump[n_], key=jump[n_].get)
            row = {"race_id": race_id, "date": pd.Timestamp(card["date"]), "field": len(runners),
                   "group": f"{meta.get('race_code') or 'gallops'}/{meta.get('country') or 'au'}",
                   "won": float(placings[n_]["position"] == 1),
                   "median": median[i], "best": best[i], "best_book": best_book, "n_books": len(jump[n_]),
                   "fair_median": float(fair_median[i]), "fair_books": float(fair_books[i]),
                   "sum_inv_best": sum(1 / b for b in best)}
            row["books_0"] = dict(jump[n_])
            row["own_fair_0"] = {b: float(v[i]) for b, v in own.items()}
            row["own_fair_30"] = {b: float(v[i]) for b, v in own30.items()}
            for m in (0,) + EARLY:
                now = jump[n_] if m == 0 else early[m].get(n_, {})
                before = persist[m].get(n_, {})
                held = [min(now[b], before[b]) for b in now if b in before]
                row[f"best_persist_{m}"] = max(held) if held else math.nan
            for m in EARLY:
                q = early[m].get(n_)
                row[f"best_{m}"] = max(q.values()) if q else math.nan
                row[f"median_{m}"] = float(np.median(list(q.values()))) if q else math.nan
                row[f"fair_{m}"] = float(fair_early[m][i])
                row[f"books_{m}"] = dict(q) if q else {}
            rows.append(row)
    return rows


def summary(ret: np.ndarray, won: np.ndarray | None = None) -> dict:
    ret = ret[~np.isnan(ret)]
    if len(ret) < 2:
        return {"bets": int(len(ret))}
    out = {"bets": int(len(ret)), "roi": round(float(ret.mean()), 4),
           "roi_se": round(float(ret.std(ddof=1) / math.sqrt(len(ret))), 4)}
    return out


def cell(df: pd.DataFrame, mask, price_col: str, mid) -> dict:
    g = df[mask]
    ret = np.where(g["won"] == 1, g[price_col] - 1, -1.0)
    halves = [np.where(h["won"] == 1, h[price_col] - 1, -1.0) for h in (g[g["date"] < mid], g[g["date"] >= mid])]
    out = summary(ret)
    out["strike"] = round(float(g["won"].mean()), 4) if len(g) else None
    out["halves_roi"] = [round(float(h.mean()), 4) if len(h) else None for h in halves]
    out["both_halves_over_2se"] = bool(all(len(h) > 1 and h.mean() > 2 * h.std(ddof=1) / math.sqrt(len(h)) for h in halves))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path(config.RAW_DIR / "formfav"))
    a = ap.parse_args()
    rows = []
    for path in sorted(a.store.glob("????-??-??/*.json")):
        rows += race_rows(path, json.loads(path.read_text()))
    df = pd.DataFrame(rows)
    if df.empty:
        print("no races with results and prices")
        return 1
    days = sorted(df["date"].unique())
    mid = days[len(days) // 2]
    races = df.drop_duplicates("race_id")
    print(json.dumps({"races": int(len(races)), "runners": int(len(df)), "days": len(days),
                      "books_per_runner_median": float(df["n_books"].median()),
                      "book_pct": {"median_price": round(float((1 / df["median"]).groupby(df["race_id"]).sum().median()), 4),
                                   "best_price": round(float(races["sum_inv_best"].median()), 4)},
                      "arb_races_sum_inv_best_below_1": int((races["sum_inv_best"] < 1).sum()),
                      "best_over_median_median_ratio": round(float((df["best"] / df["median"]).median()), 4),
                      "early_price_coverage": {str(m): round(float(df[f"best_{m}"].notna().mean()), 3) for m in EARLY}}))
    # 1. price shopping at the jump
    for fair in ("fair_median", "fair_books"):
        for k in (0.0, 0.05, 0.1, 0.2):
            m = df["best"] * df[fair] > 1 + k
            print(json.dumps({"test": "best_at_jump", "fair": fair, "k": k, **cell(df, m, "best", mid)}))
    # 2. closing-line value: early best price vs fair at the jump
    for mins in EARLY:
        col = f"best_{mins}"
        have = df[col].notna()
        for k in (0.0, 0.05, 0.1, 0.2):
            m = have & (df[col] * df["fair_median"] > 1 + k)
            print(json.dumps({"test": "early_best_vs_fair_at_jump", "minutes_before": mins, "k": k, **cell(df, m, col, mid)}))
        print(json.dumps({"test": "early_best_all_runners", "minutes_before": mins, **cell(df, have, col, mid)}))
    # 2b. BETTABLE: value by the consensus known at the time of the bet [A-56]
    df["best_0"], df["median_0"], df["fair_0"] = df["best"], df["median"], df["fair_median"]
    for mins in (0,) + EARLY:
        for k in (0.0, 0.05, 0.1, 0.2):
            m = (df[f"best_{mins}"] * df[f"fair_{mins}"] > 1 + k).fillna(False)
            g = df[m]
            print(json.dumps({"test": "value_at_time_of_bet", "minutes_before": mins, "k": k,
                              **cell(df, m, f"best_{mins}", mid),
                              "median_price_taken": round(float(g[f"best_{mins}"].median()), 2) if len(g) else None,
                              "ev_implied_by_consensus": round(float((g[f"best_{mins}"] * g[f"fair_{mins}"] - 1).mean()), 4) if len(g) else None,
                              "best_over_median_quartiles": [round(float(x), 3) for x in (g[f"best_{mins}"] / g[f"median_{mins}"]).quantile([.25, .5, .75])] if len(g) else None}))
        # glitch cap: ignore quotes more than 50% above the median, and require the quote to have held 5 minutes [A-57]
        for k in (0.0, 0.1):
            capped = (df[f"best_{mins}"] / df[f"median_{mins}"] <= 1.5)
            m = (df[f"best_{mins}"] * df[f"fair_{mins}"] > 1 + k).fillna(False) & capped.fillna(False)
            print(json.dumps({"test": "value_at_time_of_bet_capped_1.5", "minutes_before": mins, "k": k, **cell(df, m, f"best_{mins}", mid)}))
            col = f"best_persist_{mins}"
            m = (df[col] * df[f"fair_{mins}"] > 1 + k).fillna(False)
            print(json.dumps({"test": "value_at_time_of_bet_price_held_5min", "minutes_before": mins, "k": k, **cell(df, m, col, mid)}))
            m = m & (df[col] / df[f"median_{mins}"] <= 1.5).fillna(False)
            print(json.dumps({"test": "value_at_time_of_bet_held_5min_and_capped", "minutes_before": mins, "k": k, **cell(df, m, col, mid)}))
    # 2b-2. who is right about the selected runners: the consensus or the outlier book? [A-59]
    for mins in (0, 30):
        col = f"best_persist_{mins}"
        m = ((df[col] * df[f"fair_{mins}"] > 1) & (df[col] / df[f"median_{mins}"] <= 1.5)).fillna(False)
        g = df[m].copy()
        if len(g) < 20:
            continue
        g["best_b"] = [max(b, key=b.get) if b else None for b in g[f"books_{mins}"]]
        g["p_own"] = [o.get(b, math.nan) for o, b in zip(g[f"own_fair_{mins}"], g["best_b"])]
        h = g[g["p_own"].notna()]
        ls = lambda p, y: float(np.mean(np.where(y == 1, np.log(np.clip(p, 1e-9, 1)), np.log(np.clip(1 - p, 1e-9, 1)))))
        print(json.dumps({"test": "consensus_vs_outlier_book_on_selected_bets", "minutes_before": mins, "bets": int(len(g)),
                          "strike": round(float(g["won"].mean()), 4),
                          "mean_consensus_prob": round(float(g[f"fair_{mins}"].mean()), 4),
                          "mean_outlier_book_own_prob": round(float(h["p_own"].mean()), 4) if len(h) else None,
                          "with_full_book_from_outlier": int(len(h)),
                          "log_score_consensus": round(ls(h[f"fair_{mins}"].to_numpy(), h["won"].to_numpy()), 4) if len(h) else None,
                          "log_score_outlier_book": round(ls(h["p_own"].to_numpy(), h["won"].to_numpy()), 4) if len(h) else None,
                          "ev_implied_by_consensus": round(float((g[col] * g[f"fair_{mins}"] - 1).mean()), 4)}))
    # 2b-3. bookmaker profiles: coverage, own margin, and returns when it holds the top price [A-59]
    allb = pd.Series([b for d_ in df["books_0"] for b in d_]).value_counts()
    for b in allb.index:
        px = df["books_0"].map(lambda d_, b=b: d_.get(b, math.nan))
        has = px.notna()
        sub = df[has]
        full = sub.groupby("race_id")["race_id"].transform("size") == sub["field"]
        over = (1 / px[has][full]).groupby(sub[full]["race_id"]).sum()
        is_best = (df["best_book"] == b)
        val = is_best & (df["best_persist_0"] * df["fair_median"] > 1).fillna(False) & (df["best"] / df["median"] <= 1.5)
        print(json.dumps({"test": "bookmaker_profile", "bookmaker": b, "runner_coverage": round(float(has.mean()), 3),
                          "own_book_pct_median": round(float(over.median()), 4) if len(over) else None,
                          "every_runner_at_its_price": summary(np.where(sub["won"] == 1, px[has] - 1, -1.0)),
                          "when_top_price": cell(df, is_best, "best", mid),
                          "when_top_price_and_value_vs_consensus": cell(df, val, "best", mid)}))
    # 2b'. the same, per code and country (soft markets differ) [A-58]
    for grp, g in df.groupby("group"):
        if g["race_id"].nunique() < 50:
            continue
        for mins in (0, 30):
            col = f"best_persist_{mins}"
            m = ((df[col] * df[f"fair_{mins}"] > 1) & (df[col] / df[f"median_{mins}"] <= 1.5) & (df["group"] == grp)).fillna(False)
            print(json.dumps({"test": "value_held_5min_capped_by_group", "group": grp, "minutes_before": mins,
                              "races": int(g["race_id"].nunique()), **cell(df, m, col, mid)}))
        print(json.dumps({"test": "all_runners_by_group", "group": grp, "price": "best",
                          **cell(df, df["group"] == grp, "best", mid),
                          "book_pct_median_price": round(float((1 / g["median"]).groupby(g["race_id"]).sum().median()), 4)}))
    # 2c. BETTABLE: momentum - firmed between 120 and 30 minutes out, bet at the best price 30 minutes out
    ratio = df["median_30"] / df["median_120"]
    for mm in (0.05, 0.1, 0.2):
        firm = (ratio < 1 - mm).fillna(False)
        print(json.dumps({"test": "firmed_120_to_30", "firmed_at_least": mm, **cell(df, firm, "best_30", mid)}))
        both = firm & (df["best_30"] * df["fair_30"] > 1).fillna(False)
        print(json.dumps({"test": "firmed_120_to_30_and_value_at_30", "firmed_at_least": mm, **cell(df, both, "best_30", mid)}))
        drift = (ratio > 1 + mm).fillna(False)
        print(json.dumps({"test": "drifted_120_to_30", "drifted_at_least": mm, **cell(df, drift, "best_30", mid)}))
    # 2d. BETTABLE: a lagging bookmaker - best price well above a given bookmaker's price at the same time
    for mins in (0, 30):
        books = pd.Series([b for d_ in df[f"books_{mins}"] for b in d_]).value_counts()
        for b in books[books >= 0.5 * len(df)].index:
            px = df[f"books_{mins}"].map(lambda d_, b=b: d_.get(b, math.nan))
            for r in (1.1, 1.2):
                m = ((df[f"best_{mins}"] / px >= r) & (df["best_book"] != b if mins == 0 else True)).fillna(False)
                print(json.dumps({"test": "best_vs_bookmaker", "minutes_before": mins, "bookmaker": b, "ratio_at_least": r,
                                  **cell(df, m, f"best_{mins}", mid)}))
    # 2e. DIAGNOSTIC: how often does the early best price beat the closing fair price, by early-price band
    for mins in EARLY:
        beat = (df[f"best_{mins}"] * df["fair_median"] > 1)
        print(json.dumps({"test": "share_of_early_prices_beating_close", "minutes_before": mins,
                          "share": round(float(beat.mean()), 4),
                          "by_band": {str(b): round(float(g.mean()), 3) for b, g in beat.groupby(pd.cut(df[f"best_{mins}"], [1, 2, 3, 4.5, 6.5, 10, 15, 25, 1e9]), observed=True)}}))
    # 3. pure dispersion
    for r in (1.0, 1.1, 1.2, 1.3, 1.5):
        print(json.dumps({"test": "best_over_median", "ratio_at_least": r, **cell(df, df["best"] / df["median"] >= r, "best", mid)}))
    # 4. every runner at best vs median (what shopping alone recovers)
    print(json.dumps({"test": "all_runners", "price": "median", **cell(df, df["won"].notna(), "median", mid)}))
    print(json.dumps({"test": "all_runners", "price": "best", **cell(df, df["won"].notna(), "best", mid)}))
    # 5. per bookmaker when it is the best price
    for b, g in df.groupby("best_book"):
        if len(g) >= 200:
            print(json.dumps({"test": "bookmaker_when_best", "bookmaker": b, **cell(df, df["best_book"] == b, "best", mid)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

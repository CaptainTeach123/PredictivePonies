"""Flat 1-unit bets on the market's N-th favourite in every race.

    python scripts/formfav_rank_bets.py [--store data/raw/formfav] [--max-rank 6]

Favourite rank comes from the pre-jump median bookmaker price [A-42] (ties:
lower runner number). Nothing is fitted, so every completed race counts.
Settled four ways: win at the median fixed price, win at the best fixed price
[A-44], win at the TAB dividend and place at the TAB dividend [A-48][A-50].
Every rank is reported (not just one) and the race days are split in halves,
so a lucky rank shows up as inconsistent. Prints summary numbers only.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ponies import config  # noqa: E402
from ponies.exotics import places_paid  # noqa: E402
from ponies.formfav import load_dividends, load_store  # noqa: E402
from formfav_experiments import prepare  # noqa: E402


def summary(ret: np.ndarray) -> dict:
    ret = ret[~np.isnan(ret)]
    if not len(ret):
        return {"bets": 0}
    return {"bets": int(len(ret)), "strike": round(float((ret > -1).mean()), 3),
            "roi": round(float(ret.mean()), 4),
            "roi_se": round(float(ret.std(ddof=1) / math.sqrt(len(ret))), 4) if len(ret) > 1 else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path(config.RAW_DIR / "formfav"))
    ap.add_argument("--max-rank", type=int, default=6)
    a = ap.parse_args()

    df = prepare(load_store(a.store))
    divs = load_dividends(a.store)
    df["prog_n"] = pd.to_numeric(df["program"], errors="coerce")
    df = df.sort_values(["race_id", "dec", "prog_n"])
    df["fav_rank"] = df.groupby("race_id").cumcount() + 1
    won = df["won"].to_numpy() == 1

    def tote(row, pool):
        d = divs.get(row.race_id)
        if d is None:
            return math.nan
        paid = {tuple(x["combination"]): x["dividend"] for x in d if x["pool"] == pool}
        if not paid:
            return math.nan  # pool not declared for this race
        return paid.get((row.program,), 0.0) - 1.0

    df["ret_win_median"] = np.where(won, df["dec"] - 1, -1.0)
    df["ret_win_best"] = np.where(won, df["best"] - 1, -1.0)
    df["ret_win_tote"] = [tote(r, "Win") for r in df.itertuples()]
    df["ret_place_tote"] = [tote(r, "Place") if places_paid(int(r.field_size)) else math.nan
                            for r in df.itertuples()]

    days = sorted(df["date"].unique())
    mid = days[len(days) // 2]
    print(json.dumps({"races": int(df["race_id"].nunique()), "days": len(days),
                      "halves_split_at": str(pd.Timestamp(mid).date())}))
    cols = ("ret_win_median", "ret_win_best", "ret_win_tote", "ret_place_tote")
    for k in range(1, a.max_rank + 1):
        g = df[df["fav_rank"] == k]
        out = {"favourite_rank": k, "avg_price": round(float(g["dec"].median()), 2)}
        for c in cols:
            out[c[4:]] = summary(g[c].to_numpy(dtype=float))
        out["halves_roi"] = {c[4:]: [summary(h[c].to_numpy(dtype=float)).get("roi")
                                     for h in (g[g["date"] < mid], g[g["date"] >= mid])] for c in cols}
        print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

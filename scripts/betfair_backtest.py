"""Favourite-rank, price-band calibration and margin checks on Betfair
Exchange WIN markets (files from scripts/betfair_download.py).

    python scripts/betfair_backtest.py [--dir data/raw/betfair] [--commission 0.08] [--country AU]

Nothing is fitted, so every settled market counts. Bets are 1 unit backed at
the last pre-race traded price [A-53] and at the Betfair SP, with commission
on net winnings [A-54]. Every rank/band is reported and the period is split in
halves; a cell counts as value only if it is > 2 SE above zero in both.
Prints summary numbers only.
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
from ponies.betfair import load_files, net_return  # noqa: E402

BANDS = [1.0, 2.0, 3.0, 4.5, 6.5, 10.0, 15.0, 25.0, 50.0, 1e9]


def summary(r: np.ndarray) -> dict:
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return {"bets": int(len(r))}
    return {"bets": int(len(r)), "roi": round(float(r.mean()), 4), "roi_se": round(float(r.std(ddof=1) / math.sqrt(len(r))), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=config.RAW_DIR / "betfair")
    ap.add_argument("--commission", type=float, default=0.08, help="your Betfair rate on net winnings")
    ap.add_argument("--country", default=None)
    ap.add_argument("--table", type=Path, default=None, help="compact table from betfair_tar_to_table.py")
    a = ap.parse_args()

    if a.table:
        df = pd.read_csv(a.table, parse_dates=["date"], dtype={"program": str, "market_id": str})
    else:
        df = load_files(a.dir)
    if df.empty:
        print("no files")
        return 1
    df = df[(df["market_type"] == "WIN") & df["won"].notna()]
    if a.country:
        df = df[df["country"] == a.country]
    one = df.groupby("market_id")["won"].transform("sum") == 1
    priced = df.groupby("market_id")["ltp_pre"].transform(lambda s: s.notna().all() and (s > 1).all())
    df = df[one & priced].copy()
    if df.empty:
        print("no settled WIN markets with prices")
        return 1
    won = df["won"].to_numpy() == 1
    for col in ("ltp_pre", "bsp"):
        df[f"ret_{col}"] = [net_return(p, w, a.commission) if p > 1 else math.nan
                            for p, w in zip(df[col].fillna(0), won)]
    book = (1 / df["ltp_pre"]).groupby(df["market_id"]).sum()
    df["p_prop"] = (1 / df["ltp_pre"]) / df.groupby("market_id")["ltp_pre"].transform(lambda s: (1 / s).sum())
    df["p_shin"] = np.nan
    for _, idx in df.groupby("market_id").indices.items():
        df.iloc[idx, df.columns.get_loc("p_shin")] = odds.shin_probabilities(df["ltp_pre"].to_numpy()[idx])
    ll = lambda col: round(float(-np.log(df.loc[won, col].clip(1e-12)).mean()), 4)
    days = sorted(df["date"].unique())
    mid = days[len(days) // 2]
    print(json.dumps({"markets": int(df["market_id"].nunique()), "runners": len(df), "days": len(days),
                      "from": str(pd.Timestamp(days[0]).date()), "to": str(pd.Timestamp(days[-1]).date()),
                      "countries": df["country"].value_counts().head(5).to_dict(),
                      "book_pct_median": round(float(book.median()), 4),
                      "bsp_coverage": round(float(df["bsp"].notna().mean()), 3),
                      "log_loss": {"proportional": ll("p_prop"), "shin": ll("p_shin")},
                      "commission": a.commission}))
    df = df.sort_values(["market_id", "ltp_pre"])
    df["fav_rank"] = df.groupby("market_id").cumcount() + 1
    for k in range(1, 7):
        g = df[df["fav_rank"] == k]
        print(json.dumps({"favourite_rank": k, "median_price": round(float(g["ltp_pre"].median()), 2),
                          "strike": round(float(g["won"].mean()), 4),
                          "back_pre_race": summary(g["ret_ltp_pre"].to_numpy(dtype=float)),
                          "back_bsp": summary(g["ret_bsp"].to_numpy(dtype=float)),
                          "halves_roi_pre": [summary(h["ret_ltp_pre"].to_numpy(dtype=float)).get("roi")
                                             for h in (g[g["date"] < mid], g[g["date"] >= mid])]}))
    df["band"] = pd.cut(df["ltp_pre"], BANDS, right=False,
                        labels=[f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+" for lo, hi in zip(BANDS, BANDS[1:])])
    flagged, rows = [], []
    for band, g in df.groupby("band", observed=True):
        halves = [g[g["date"] < mid]["ret_ltp_pre"].dropna(), g[g["date"] >= mid]["ret_ltp_pre"].dropna()]
        se = lambda x: x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else math.inf
        rows.append({"band": str(band), "runners": int(len(g)), "strike": round(float(g["won"].mean()), 4),
                     "implied_prop": round(float(g["p_prop"].mean()), 4), "implied_shin": round(float(g["p_shin"].mean()), 4),
                     **summary(g["ret_ltp_pre"].to_numpy(dtype=float)),
                     "roi_halves": [round(float(h.mean()), 4) if len(h) else None for h in halves]})
        if all(len(h) > 1 and h.mean() > 2 * se(h) for h in halves):
            flagged.append(str(band))
    print(json.dumps({"calibration": rows}))
    print(json.dumps({"value_bands_both_halves_over_2se": flagged}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Independent rebuild of round 2's one survivor (written without reading its scripts).

    python scripts/bf2_place_lay_independent.py

Rule R1b@20: in AU/GB/IE PLACE markets paying 2 or 3 places, fully traded before the off
(every runner has a place ltp_pre) with sum(1/place ltp_pre) / places paid in [0.97, 1.03],
whose runners match the WIN market and which show no late-scratching sign (saddlecloth gap
together with a WIN ltp_pre book below 0.97), lay at place BSP every runner whose place
ltp_pre is >= 20, with a fixed liability of 1.

Per unit liability at BSP P: backer's stake s = 1/(P-1). Not placed: +s*(1-c). Placed:
fraction f = places paid x dead_heat_share of the bet is settled as a winner (f = 1 without
a dead heat): -f + (1-f)*s*(1-c). Commission per bet on net winnings. Summary numbers only.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "data/raw/betfair/split"


def load(country: str) -> pd.DataFrame:
    w = pd.read_parquet(SPLIT / f"{country}_win.parquet").drop_duplicates(["market_id", "selection_id"])
    p = pd.read_parquet(SPLIT / f"{country}_place.parquet").drop_duplicates(["market_id", "selection_id"])
    p = p[p["n_winners"].isin([2, 3])]
    # WIN market of the same race: same event and start; one WIN market per race
    wm = w.groupby(["event_id", "start"])["market_id"].nunique()
    one = wm[wm == 1].index
    w = w.set_index(["event_id", "start"]).loc[lambda d: d.index.isin(one)].reset_index()
    # late-scratching sign on the WIN market (G6)
    def g6(g):
        progs = pd.to_numeric(g["program"], errors="coerce").dropna().astype(int)
        gap = len(progs) and set(range(1, progs.max() + 1)) - set(progs)
        book = (1 / g["ltp_pre"]).sum() if g["ltp_pre"].notna().all() else np.nan
        return bool(gap) and (book < 0.97)
    flag = w.groupby("market_id").apply(g6, include_groups=False).rename("g6")
    win_sets = w.groupby(["event_id", "start"])["selection_id"].apply(frozenset).rename("win_set")
    win_mid = w.groupby(["event_id", "start"])["market_id"].first().rename("win_market")
    place_sets = p.groupby("market_id").agg(event_id=("event_id", "first"), start=("start", "first"),
                                            place_set=("selection_id", frozenset),
                                            all_priced=("ltp_pre", lambda s: s.notna().all()),
                                            book=("ltp_pre", lambda s: (1 / s).sum()),
                                            nw=("n_winners", "first"))
    place_sets = place_sets.join(win_sets, on=["event_id", "start"]).join(win_mid, on=["event_id", "start"])
    place_sets = place_sets.join(flag, on="win_market")
    ok = (place_sets["place_set"] == place_sets["win_set"]) & place_sets["all_priced"] \
        & (place_sets["book"] / place_sets["nw"]).between(0.97, 1.03) & ~place_sets["g6"].fillna(True).astype(bool)
    keep = place_sets[ok].index
    out = p[p["market_id"].isin(keep)].copy()
    return out, len(place_sets), int(ok.sum())


def lay_return(bsp, won, f, c):
    s = 1.0 / (bsp - 1.0)
    win_part = (1 - f) * s
    ret = np.where(won == 1, -f + win_part, s)
    return np.where(ret > 0, ret * (1 - c), ret)


def cell(df: pd.DataFrame, c: float = 0.08) -> dict:
    r = lay_return(df["bsp"].to_numpy(), df["won"].to_numpy(), df["f"].to_numpy(), c)
    n_m = df["market_id"].nunique()
    # cluster-robust SE of the mean return, clustered by market
    resid = pd.Series(r - r.mean()).groupby(df["market_id"].to_numpy()).sum()
    se = math.sqrt((resid ** 2).sum() * n_m / (n_m - 1)) / len(r)
    return {"bets": int(len(r)), "roi": round(float(r.mean()) * 100, 3), "se": round(se * 100, 3),
            "z": round(float(r.mean()) / se, 2) if se else None}


def main() -> int:
    frames = []
    for cty in ("au", "gb", "ie"):
        d, n_all, n_ok = load(cty)
        print(f"{cty}: place markets {n_all}, passing filters {n_ok}")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["ltp_pre"] >= 20]
    df = df[df["bsp"].notna() & (df["bsp"] > 1)]
    df["f"] = np.where(df["won"] == 1, np.minimum(1.0, df["n_winners"] * df["dead_heat_share"].fillna(0)), 0.0)
    df.loc[(df["won"] == 1) & (df["f"] == 0), "f"] = 1.0
    df["date"] = pd.to_datetime(df["date"])
    mid = df["date"].sort_values().iloc[len(df) // 2]
    last12 = df["date"] >= pd.Timestamp("2019-11-08")
    print("R1b@20 per liability at place BSP")
    for c in (0.0, 0.05, 0.08, 0.10):
        print(f"  commission {c:.0%}: {cell(df, c)}")
    print("  halves at 8% (split", mid.date(), "):", cell(df[df["date"] < mid]), cell(df[df["date"] >= mid]))
    print("  last 12 months at 8%:", cell(df[last12]))
    for cty, g in df.groupby("country"):
        print(f"  {cty}: {cell(g)}")
    for y, g in df.groupby(df["date"].dt.year):
        print(f"  {y}: {cell(g)}")
    strike = df["won"].mean()
    implied = (1 / df["bsp"]).mean()
    print(f"  placed {strike:.4f} vs BSP-implied {implied:.4f} (ratio {strike / implied:.3f}); median BSP {df['bsp'].median():.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

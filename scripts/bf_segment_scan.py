"""Experiment "segment_scan": where (if anywhere) the Betfair Exchange is
inefficient on Australian WIN markets (data/raw/betfair/split/au_win.parquet).

    python scripts/bf_segment_scan.py [--end 2018-11-01] [--cells-csv /path/cells.csv]

Selections (decided before the off, from ltp_pre): the favourite (rank 1 at
ltp_pre) and the nine price bands 1-2, 2-3, 3-4.5, 4.5-6.5, 6.5-10, 10-15,
15-25, 25-50, 50+. Each selection is backed with 1 unit at two prices:

    ltp_pre  last traded price before the off (one-minute snapshots, so it is
             not a guaranteed available price; ROI one tick worse is also shown)
    bsp      Betfair SP (an SP bet placed on the ltp_pre selection; 97.8% of
             runners have a BSP)

and cut by one segment at a time:

    field size (2-5, 6-8, 9-11, 12+) . harness/gallops . top-25 venues + other .
    local start hour (<12, 12-13, 14-15, 16-17, 18-19, 20+; venue timezone) .
    local weekday . year (2016-17, 2018, 2019, 2020) . book-overround quartile
    (sum of 1/ltp_pre; cutoffs are walk-forward: earlier months only) .
    has_t30 (the runner traded 30 min out; a liquidity proxy, present for ~40%
    of runners) . n_winners of the paired PLACE market (2 / 3 / none)

Every cell is reported: bets, ROI at commission 0 / 5% / 8% (commission on net
winnings only [A-54]), per-bet SE at 5%, ROI and SE in each half of the period.
Benjamini-Hochberg at FDR 5% is applied across all back cells (one-sided
p-values from z = roi_c5 / se); a cell is a survivor only if it passes BH AND
is > 2 SE above zero after 5% commission in BOTH halves. The scan is then
repeated with each half as the development set (cells picked in one half,
evaluated in the other). A secondary family (lay side, 1 unit backer's stake)
flags over-bet segments the same way.

Nothing is fitted on outcomes: the only data-derived cutoffs are the top-25
venue list and the book quartiles (both outcome-free; the quartiles use only
earlier months). Prints summary numbers only; never prints horse names.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from ponies import odds  # noqa: E402
from bf_microstructure_logit import TZ_VENUES, _venues  # noqa: E402  venue -> IANA timezone table

COMMISSIONS = (0.0, 0.05, 0.08)
PRICE_BANDS = [1.0, 2.0, 3.0, 4.5, 6.5, 10.0, 15.0, 25.0, 50.0, 1e9]
FIELD_EDGES = [2, 6, 9, 12, 99]
FIELD_LABELS = ["2-5", "6-8", "9-11", "12+"]
HOUR_EDGES = [0, 12, 14, 16, 18, 20, 25]
HOUR_LABELS = ["<12", "12-13", "14-15", "16-17", "18-19", "20+"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TOP_VENUES = 25
BOOK_BURN_IN_MONTHS = 3
FDR = 0.05
MIN_BETS_LIST = 100  # survivors below this are still listed, but flagged as thin

# Betfair price ladder: (lower bound inclusive, upper bound exclusive, increment)
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1001.0, 10.0)]


# ----------------------------------------------------------------------------- utilities


def out(obj) -> None:
    print(json.dumps(obj, default=_json_default), flush=True)


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    raise TypeError(str(type(o)))


def r4(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) else round(x, 4)


def net_returns(price: np.ndarray, won: np.ndarray, commission: float) -> np.ndarray:
    """Vectorised ponies.betfair.net_return [A-54]: commission on net winnings only."""
    price = np.asarray(price, dtype=float)
    return np.where(won, (price - 1.0) * (1.0 - commission), -1.0)


def lay_returns(price: np.ndarray, won: np.ndarray, commission: float) -> np.ndarray:
    """Lay 1 unit of backer's stake: win the stake less commission when the runner
    loses, pay the liability (price - 1) when it wins."""
    price = np.asarray(price, dtype=float)
    return np.where(won, -(price - 1.0), 1.0 - commission)


def tick_down(price: np.ndarray) -> np.ndarray:
    """One Betfair tick below ``price``: what you get if the last trade was one
    tick better than what is on offer."""
    p = np.asarray(price, dtype=float)
    inc = np.full_like(p, 10.0)
    q = p - 1e-9
    for lo, hi, step in TICKS:
        inc = np.where((q >= lo) & (q < hi), step, inc)
    return np.maximum(1.01, np.round(p - inc, 2))


def one_sided_p(z: float) -> float:
    if z is None or math.isnan(z):
        return float("nan")
    return 0.5 * math.erfc(z / math.sqrt(2))


def benjamini_hochberg(pvals: np.ndarray, q: float = FDR) -> np.ndarray:
    """Boolean mask of rejected hypotheses at FDR q (NaN = not tested)."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    rej = np.zeros(len(p), dtype=bool)
    if m == 0:
        return rej
    order = np.argsort(p[ok])
    sorted_p = p[ok][order]
    thresh = q * (np.arange(1, m + 1) / m)
    below = np.flatnonzero(sorted_p <= thresh)
    if len(below):
        k = below.max()
        rej[np.flatnonzero(ok)[order[: k + 1]]] = True
    return rej


# ----------------------------------------------------------------------------- data


def load(path: Path, place_path: Path, end: str | None) -> tuple[pd.DataFrame, dict]:
    cols = ["market_id", "event_id", "venue", "market_name", "start", "date", "field_size",
            "selection_id", "ltp_pre", "ltp_t30", "bsp", "won", "market_type"]
    df = pd.read_parquet(path, columns=cols)
    n_raw = len(df)
    df = df.drop_duplicates(["market_id", "selection_id"])
    n_dedup = len(df)
    df = df[(df["market_type"] == "WIN") & df["won"].notna()]
    if end:
        df = df[df["date"] < pd.Timestamp(end)]
    m_settled = df["market_id"].nunique()
    g = df.groupby("market_id")
    one_winner = g["won"].transform("sum") == 1  # drops dead heats
    df["_ok"] = df["ltp_pre"].notna() & (df["ltp_pre"] > 1)
    all_priced = df.groupby("market_id")["_ok"].transform("min").astype(bool)
    df = df[one_winner & all_priced].drop(columns=["_ok", "market_type"]).copy()
    # deterministic favourite: lowest ltp_pre, ties by selection_id
    df = df.sort_values(["start", "market_id", "ltp_pre", "selection_id"]).reset_index(drop=True)
    df["fav"] = df.groupby("market_id").cumcount() == 0
    df["won_b"] = df["won"].to_numpy() == 1

    lp = df["ltp_pre"].to_numpy()
    p_shin = np.empty(len(df))
    for _, idx in df.groupby("market_id", sort=False).indices.items():
        p_shin[idx] = odds.shin_probabilities(lp[idx])
    df["p_shin"] = p_shin
    df["book"] = (1.0 / df["ltp_pre"]).groupby(df["market_id"]).transform("sum")
    df["harness"] = df["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b")
    df["has_t30"] = df["ltp_t30"].notna() & (df["ltp_t30"] > 1)
    df["month"] = df["date"].dt.to_period("M")

    # local start time from the venue's timezone (unmapped venues -> Sydney)
    tz_of = {v: tz for tz, names in TZ_VENUES.items() for v in _venues(names)}
    venue_tz = df["venue"].map(tz_of)
    unmapped = venue_tz.isna()
    venue_tz = venue_tz.fillna("Australia/Sydney")
    local_hour = np.zeros(len(df))
    local_wday = np.zeros(len(df), dtype=int)
    for tz, idx in df.groupby(venue_tz.to_numpy()).indices.items():
        loc = df["start"].iloc[idx].dt.tz_convert(tz)
        local_hour[idx] = (loc.dt.hour + loc.dt.minute / 60.0).to_numpy()
        local_wday[idx] = loc.dt.weekday.to_numpy()
    df["hour_local"] = local_hour
    df["weekday"] = local_wday

    # paired PLACE market: n_winners (places paid)
    pl = pd.read_parquet(place_path, columns=["market_id", "event_id", "start", "n_winners", "market_type"])
    pl = pl[pl["market_type"] == "PLACE"].drop_duplicates("market_id")
    pl = pl.drop_duplicates(["event_id", "start"])[["event_id", "start", "n_winners"]]
    df = df.merge(pl.rename(columns={"n_winners": "place_n_winners"}), on=["event_id", "start"], how="left")

    info = {"rows_raw": int(n_raw), "rows_after_dedupe": int(n_dedup), "markets_settled": int(m_settled),
            "markets_used": int(df["market_id"].nunique()), "runners_used": int(len(df)),
            "dropped": "dead heats and markets with any runner never traded before the off",
            "bsp_coverage": r4(((df["bsp"] > 1) & df["bsp"].notna()).mean()),
            "t30_runner_coverage": r4(df["has_t30"].mean()),
            "venue_tz_unmapped_market_share": r4(df.loc[unmapped, "market_id"].nunique() / max(df["market_id"].nunique(), 1)),
            "paired_place_market_share": r4(df.drop_duplicates("market_id")["place_n_winners"].notna().mean())}
    return df, info


def walk_forward_book_quartile(df: pd.DataFrame) -> pd.Series:
    """Quartile of the market's book (sum 1/ltp_pre) against the distribution of
    books in strictly earlier months; the first BOOK_BURN_IN_MONTHS months use
    the cutoffs of those months (outcome-free either way)."""
    mk = df.drop_duplicates("market_id")[["market_id", "month", "book"]]
    months = sorted(mk["month"].unique())
    q = pd.Series(index=mk.index, dtype=object)
    burn = months[:BOOK_BURN_IN_MONTHS]
    cuts_burn = mk.loc[mk["month"].isin(burn), "book"].quantile([0.25, 0.5, 0.75]).to_numpy()
    for i, m in enumerate(months):
        if i < BOOK_BURN_IN_MONTHS:
            cuts = cuts_burn
        else:
            cuts = mk.loc[mk["month"] < m, "book"].quantile([0.25, 0.5, 0.75]).to_numpy()
        sel = mk["month"] == m
        q[sel] = np.searchsorted(cuts, mk.loc[sel, "book"].to_numpy(), side="right")
    lab = {0: "Q1 (lowest book)", 1: "Q2", 2: "Q3", 3: "Q4 (highest book)"}
    mk = mk.assign(book_q=q.astype(int).map(lab))
    return df["market_id"].map(mk.set_index("market_id")["book_q"])


def segments(df: pd.DataFrame) -> dict[str, pd.Series]:
    """name -> categorical label per runner row."""
    seg = {}
    seg["field_size"] = pd.cut(df["field_size"], FIELD_EDGES, right=False, labels=FIELD_LABELS).astype(str)
    seg["code"] = np.where(df["harness"], "harness", "gallops")
    top = df.drop_duplicates("market_id")["venue"].value_counts().head(TOP_VENUES).index
    seg["venue"] = np.where(df["venue"].isin(top), df["venue"].fillna("other"), "other")
    seg["hour_local"] = pd.cut(df["hour_local"], HOUR_EDGES, right=False, labels=HOUR_LABELS).astype(str)
    seg["weekday"] = pd.Series(df["weekday"]).map(dict(enumerate(WEEKDAYS))).to_numpy()
    year = df["date"].dt.year
    seg["year"] = np.where(year <= 2017, "2016-17", year.astype(str))
    seg["book_quartile"] = walk_forward_book_quartile(df)
    seg["has_t30"] = np.where(df["has_t30"], "t30 traded", "no t30 trade")
    seg["place_n_winners"] = df["place_n_winners"].map({2.0: "2 places", 3.0: "3 places"}).fillna("no place market")
    return {k: pd.Series(np.asarray(v, dtype=object), index=df.index) for k, v in seg.items()}


# ----------------------------------------------------------------------------- cells


class Returns:
    """Per-row return arrays so each cell is a few numpy means."""

    def __init__(self, df: pd.DataFrame, mid: pd.Timestamp):
        won = df["won_b"].to_numpy()
        lp = df["ltp_pre"].to_numpy(dtype=float)
        bsp = df["bsp"].to_numpy(dtype=float)
        self.won = won
        self.first = (df["date"] < mid).to_numpy()
        self.p_shin = df["p_shin"].to_numpy()
        self.price = {"ltp_pre": lp, "bsp": bsp}
        self.valid = {"ltp_pre": np.ones(len(df), dtype=bool), "bsp": np.isfinite(bsp) & (bsp > 1)}
        self.back = {src: {c: net_returns(p, won, c) for c in COMMISSIONS} for src, p in self.price.items()}
        self.lay5 = {src: lay_returns(p, won, 0.05) for src, p in self.price.items()}
        self.tick5 = net_returns(tick_down(lp), won, 0.05)

    def cell(self, idx: np.ndarray, src: str) -> dict:
        idx = idx[self.valid[src][idx]]
        n = len(idx)
        res = {"bets": int(n)}
        if n == 0:
            return res
        won = self.won[idx]
        price = self.price[src][idx]
        for c in COMMISSIONS:
            res[f"roi_c{int(c * 100)}"] = r4(self.back[src][c][idx].mean())
        r5 = self.back[src][0.05][idx]
        res["se"] = r4(r5.std(ddof=1) / math.sqrt(n)) if n > 1 else None
        first = self.first[idx]
        halves, halves_se, halves_n = [], [], []
        for mask in (first, ~first):
            h = r5[mask]
            halves_n.append(int(mask.sum()))
            halves.append(r4(h.mean()) if len(h) else None)
            halves_se.append(r4(h.std(ddof=1) / math.sqrt(len(h))) if len(h) > 1 else None)
        res["halves_roi_c5"] = halves
        res["halves_se_c5"] = halves_se
        res["halves_bets"] = halves_n
        res["positive_both_halves"] = bool(all(h is not None and s is not None and s > 0 and h > 2 * s
                                               for h, s in zip(halves, halves_se)))
        res["strike"] = r4(won.mean())
        res["implied_shin"] = r4(self.p_shin[idx].mean())
        res["median_price"] = r4(np.median(price))
        if src == "ltp_pre":
            res["roi_c5_tick_down"] = r4(self.tick5[idx].mean())
        l5 = self.lay5[src][idx]
        res["lay_roi_c5"] = r4(l5.mean())
        res["lay_se"] = r4(l5.std(ddof=1) / math.sqrt(n)) if n > 1 else None
        res["lay_halves_roi_c5"] = [r4(l5[m].mean()) if m.sum() else None for m in (first, ~first)]
        res["lay_halves_se_c5"] = [r4(l5[m].std(ddof=1) / math.sqrt(m.sum())) if m.sum() > 1 else None
                                   for m in (first, ~first)]
        return res


def z_of(cell: dict, key: str = "roi_c5", se_key: str = "se") -> float:
    if cell.get("bets", 0) < 2 or not cell.get(se_key):
        return float("nan")
    return cell[key] / cell[se_key]


def half_z(cell: dict, h: int, key: str = "halves_roi_c5", se_key: str = "halves_se_c5") -> float:
    if key not in cell:
        return float("nan")
    r, s = cell[key][h], cell[se_key][h]
    if r is None or not s:
        return float("nan")
    return r / s


def compact(c: dict) -> list:
    """[bets, roi_c0, roi_c5, roi_c8, se, half1_roi_c5, half2_roi_c5] for the per-segment tables."""
    return [c.get("bets"), c.get("roi_c0"), c.get("roi_c5"), c.get("roi_c8"), c.get("se"),
            *(c.get("halves_roi_c5") or [None, None])]


def full(c: dict) -> dict:
    keys = ("bets", "roi_c0", "roi_c5", "roi_c8", "se", "halves_roi_c5", "halves_se_c5", "halves_bets",
            "positive_both_halves", "strike", "implied_shin", "median_price", "roi_c5_tick_down", "roi_c5_bsp",
            "lay_roi_c5", "lay_se")
    return {k: c.get(k) for k in keys if k in c}


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=Path("data/raw/betfair/split/au_win.parquet"))
    ap.add_argument("--place-path", type=Path, default=Path("data/raw/betfair/split/au_place.parquet"))
    ap.add_argument("--end", default=None, help="development: use only dates before this")
    ap.add_argument("--cells-csv", type=Path, default=None, help="write every cell to this CSV")
    a = ap.parse_args()
    t0 = time.time()

    df, info = load(a.path, a.place_path, a.end)
    days = np.sort(df["date"].unique())
    mid = pd.Timestamp(days[len(days) // 2])
    seg = segments(df)
    R = Returns(df, mid)
    df["band"] = pd.cut(df["ltp_pre"], PRICE_BANDS, right=False,
                        labels=[f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+" for lo, hi in zip(PRICE_BANDS, PRICE_BANDS[1:])])

    selections = {"favourite": df["fav"].to_numpy()}
    for b in df["band"].cat.categories:
        selections[f"band {b}"] = (df["band"] == b).to_numpy()

    mk_index = df.drop_duplicates("market_id").index
    level_counts = {name: {str(k): int(v) for k, v in s.loc[mk_index].value_counts().items()}
                    for name, s in seg.items() if name != "has_t30"}
    level_counts["has_t30 (runners)"] = {str(k): int(v) for k, v in seg["has_t30"].value_counts().items()}
    out({"section": "data", **info,
         "from": str(pd.Timestamp(days[0]).date()), "to": str(pd.Timestamp(days[-1]).date()),
         "mid_day_for_halves": str(mid.date()),
         "markets_per_segment_level": level_counts,
         "load_seconds": round(time.time() - t0, 1)})

    # ---------------- the scan ---------------------------------------------------------------------
    cells: list[dict] = []  # one record per tested cell

    def record(src, sel_name, seg_name, level, c):
        cells.append({"price": src, "selection": sel_name, "segment": seg_name, "level": level, **c})

    for src in ("ltp_pre", "bsp"):
        base = {}
        for sel_name, mask in selections.items():
            c = R.cell(np.flatnonzero(mask), src)
            record(src, sel_name, "all", "all", c)
            base[sel_name] = full(c)
        out({"section": "base", "price": src, "note": "no segmentation; selection decided at ltp_pre",
             "cells": base})
        for seg_name, s in seg.items():
            levels = sorted(pd.unique(s.to_numpy()), key=lambda x: (str(x)))
            if seg_name == "field_size":
                levels = FIELD_LABELS
            elif seg_name == "hour_local":
                levels = HOUR_LABELS
            elif seg_name == "weekday":
                levels = WEEKDAYS
            s_arr = s.to_numpy()
            table = {}
            for sel_name, mask in selections.items():
                row = {}
                for lv in levels:
                    idx = np.flatnonzero(mask & (s_arr == lv))
                    c = R.cell(idx, src)
                    record(src, sel_name, seg_name, lv, c)
                    row[lv] = compact(c)
                table[sel_name] = row
            out({"section": "segment", "price": src, "segment": seg_name,
                 "columns": ["bets", "roi_c0", "roi_c5", "roi_c8", "se", "half1_roi_c5", "half2_roi_c5"],
                 "cells": table})

    # ---------------- multiple testing ------------------------------------------------------------
    z = np.array([z_of(c) for c in cells])
    p = np.array([one_sided_p(v) for v in z])
    rej = benjamini_hochberg(p)
    both = np.array([c.get("positive_both_halves", False) for c in cells])
    tested = ~np.isnan(p)
    survivors = [c for c, r, b in zip(cells, rej, both) if r and b]
    bh_only = [c for c, r in zip(cells, rej) if r]
    z_all = z[tested]
    out({"section": "benjamini_hochberg_back",
         "cells_total": len(cells), "cells_with_se": int(tested.sum()),
         "cells_by_price": {src: int(sum(c["price"] == src for c in cells)) for src in ("ltp_pre", "bsp")},
         "fdr": FDR,
         "bh_rejections": int(rej.sum()),
         "cells_z_over_2_full_period": int((z_all > 2).sum()),
         "expected_z_over_2_under_null": round(0.02275 * int(tested.sum()), 1),
         "cells_positive_both_halves_no_bh": int(both.sum()),
         "survivors_bh_and_both_halves": len(survivors),
         "note": "year cells cannot pass the both-halves rule by construction (each year lies mostly in one half)",
         "max_z": r4(np.nanmax(z)), "min_z": r4(np.nanmin(z)),
         "min_p": r4(np.nanmin(p)), "bh_threshold_for_smallest_p": r4(FDR / max(int(tested.sum()), 1))})
    out({"section": "survivors_back", "rows": [{k: c[k] for k in ("price", "selection", "segment", "level")}
                                                | full(c) | {"z": r4(z_of(c)), "thin": c["bets"] < MIN_BETS_LIST}
                                                for c in survivors]})
    out({"section": "bh_rejections_back_regardless_of_halves",
         "rows": [{k: c[k] for k in ("price", "selection", "segment", "level")} | full(c) | {"z": r4(z_of(c))}
                  for c in bh_only]})
    # the best-looking cells even though nothing need pass: top 10 by z with bets >= MIN_BETS_LIST
    order = np.argsort(-np.nan_to_num(z, nan=-99))
    top = [cells[i] for i in order if cells[i]["bets"] >= MIN_BETS_LIST][:10]
    out({"section": "top10_back_cells_by_z_min100_bets",
         "rows": [{k: c[k] for k in ("price", "selection", "segment", "level")} | full(c)
                  | {"z": r4(z_of(c)), "p_one_sided": r4(one_sided_p(z_of(c)))} for c in top]})

    # ---------------- secondary family: lay side (over-bet segments) ----------------------------
    zl = np.array([z_of(c, "lay_roi_c5", "lay_se") for c in cells])
    pl_ = np.array([one_sided_p(v) for v in zl])
    rej_l = benjamini_hochberg(pl_)
    both_l = np.array([all(half_z(c, h, "lay_halves_roi_c5", "lay_halves_se_c5") > 2 for h in (0, 1))
                       if "lay_halves_roi_c5" in c else False for c in cells])
    surv_l = [c for c, r, b in zip(cells, rej_l, both_l) if r and b]
    out({"section": "benjamini_hochberg_lay",
         "note": "lay 1 unit backer's stake at the same price; a positive lay ROI = the segment is over-bet. Realistic lay execution would be one tick worse too.",
         "cells_with_se": int((~np.isnan(pl_)).sum()), "bh_rejections": int(rej_l.sum()),
         "survivors_bh_and_both_halves": len(surv_l),
         "rows": [{k: c[k] for k in ("price", "selection", "segment", "level")}
                  | {"bets": c["bets"], "lay_roi_c5": c["lay_roi_c5"], "lay_se": c["lay_se"],
                     "lay_halves_roi_c5": c["lay_halves_roi_c5"], "back_roi_c0": c["roi_c0"],
                     "strike": c["strike"], "implied_shin": c["implied_shin"], "z": r4(z_of(c, "lay_roi_c5", "lay_se"))}
                  for c in surv_l]})

    # ---------------- opposite-half development check ------------------------------------------
    dev_checks = []
    for dev, test in ((0, 1), (1, 0)):
        zd = np.array([half_z(c, dev) for c in cells])
        zt = np.array([half_z(c, test) for c in cells])
        pd_ = np.array([one_sided_p(v) for v in zd])
        picked = np.flatnonzero(zd > 2)
        picked_bh = np.flatnonzero(benjamini_hochberg(pd_))
        held = [i for i in picked if zt[i] > 2]
        held_bh = [i for i in picked_bh if zt[i] > 2]
        # pooled test-half performance of the picked rules on unique bets
        rows_sel = np.zeros(len(df), dtype=bool)
        for i in picked:
            c = cells[i]
            mask = selections[c["selection"]].copy()
            if c["segment"] != "all":
                mask &= seg[c["segment"]].to_numpy() == c["level"]
            rows_sel |= mask
        test_mask = rows_sel & (R.first if test == 0 else ~R.first)
        pooled = {}
        for src in ("ltp_pre", "bsp"):
            idx = np.flatnonzero(test_mask & R.valid[src])
            if len(idx) > 1:
                r5 = R.back[src][0.05][idx]
                pooled[src] = {"bets": int(len(idx)), "roi_c5": r4(r5.mean()), "se": r4(r5.std(ddof=1) / math.sqrt(len(idx)))}
        dev_checks.append({"dev_half": dev + 1, "test_half": test + 1,
                           "cells_z_over_2_in_dev": int(len(picked)),
                           "expected_under_null": round(0.02275 * int((~np.isnan(zd)).sum()), 1),
                           "held_z_over_2_in_test": len(held),
                           "bh_in_dev": int(len(picked_bh)), "bh_in_dev_held_in_test": len(held_bh),
                           "mean_test_z_of_dev_picks": r4(np.nanmean(zt[picked])) if len(picked) else None,
                           "held_cells": [{k: cells[i][k] for k in ("price", "selection", "segment", "level", "bets")}
                                          | {"dev_roi_c5": cells[i]["halves_roi_c5"][dev], "test_roi_c5": cells[i]["halves_roi_c5"][test],
                                             "test_se": cells[i]["halves_se_c5"][test]} for i in held],
                           "pooled_test_half_unique_bets_of_all_dev_picks": pooled})
    out({"section": "opposite_half_development", "rows": dev_checks})

    if a.cells_csv:
        a.cells_csv.parent.mkdir(parents=True, exist_ok=True)
        flat = []
        for c, zz, pp, rr in zip(cells, z, p, rej):
            rec = {k: v for k, v in c.items() if not isinstance(v, list)}
            rec.update({"half1_roi_c5": (c.get("halves_roi_c5") or [None, None])[0],
                        "half2_roi_c5": (c.get("halves_roi_c5") or [None, None])[1],
                        "half1_se_c5": (c.get("halves_se_c5") or [None, None])[0],
                        "half2_se_c5": (c.get("halves_se_c5") or [None, None])[1],
                        "half1_bets": (c.get("halves_bets") or [None, None])[0],
                        "half2_bets": (c.get("halves_bets") or [None, None])[1],
                        "z": zz, "p_one_sided": pp, "bh_reject": rr})
            flat.append(rec)
        pd.DataFrame(flat).to_csv(a.cells_csv, index=False)
    out({"section": "done", "seconds": round(time.time() - t0, 1), "cells": len(cells)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

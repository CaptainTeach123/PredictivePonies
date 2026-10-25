"""Adversarial verification of the "cross_country" claim:

    US WIN markets (every runner traded pre-off), lay 1 unit of backer's stake
    at BSP on every runner with ltp_pre >= 50: +15.4% of stake after 5%
    commission (n 63,982, SE 0.0375, halves 0.1678 / 0.1438).

    python scripts/bf_verify_cross_country_1.py [--countries US,GB,IE,NZ,ZA,FR] [--boot 4000]

What is re-run (same data filters as scripts/bf_cross_country.py: dedupe,
settled WIN, one winner, every runner traded pre-off):

  * the cell itself (lay at BSP and at ltp_pre), ROI at commission 0/5/8%,
    per-bet SE, halves at the median market date;
  * cluster-robust SEs by market and by race day (runners in one race and
    races on one day are not independent), for the full period and each half;
  * a race-day cluster bootstrap (percentile 95% interval) because the lay
    return is extremely skewed (0.95 almost always, -(BSP-1) up to -999);
  * influence: ROI after removing the k most expensive winners;
  * calendar-year split (the halves split is one of many possible splits);
  * decomposition by the BSP the runner actually settled at (ltp_pre >= 50 is
    a stale last trade; 11% of the selected runners started at BSP < 50);
  * the executable form of the bet.  A Betfair SP lay is placed by LIABILITY,
    not by backer's stake (the backer's stake matched = liability / (BSP - 1)
    is not known until the off), so "1 unit of backer's stake at BSP" cannot
    be placed.  Two liability-based returns are reported with SEs:
      - fixed liability 1 per bet (what an SP lay actually does):
        mean over bets of lay_return_i / (BSP_i - 1);
      - the script's ratio sum(lay_return) / sum(BSP - 1) (equal-stake
        strategy's return per unit of total liability), delta-method SE;
    plus SP lays with a price limit (matched only if BSP <= limit), which are
    post-hoc variants and are labelled as such;
  * the same lay cell on the markets the strict filter drops (thin) and on
    all US markets together.

Prints JSON lines with summary numbers only; never prints horse names.
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

SPLIT_DIR = HERE.parent / "data" / "raw" / "betfair" / "split"
COMMISSIONS = (0.0, 0.05, 0.08)
BSP_BANDS = [1.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1001.0]
BSP_LABELS = [f"{lo:g}-{hi:g}" for lo, hi in zip(BSP_BANDS, BSP_BANDS[1:])]


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    raise TypeError(str(type(o)))


def out(obj) -> None:
    print(json.dumps(obj, default=_json_default), flush=True)


def r5(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) else round(x, 5)


def lay_returns(price: np.ndarray, won: np.ndarray, commission: float) -> np.ndarray:
    """Lay 1 unit of backer's stake: keep stake less commission if the runner loses,
    pay the liability (price - 1) if it wins."""
    price = np.asarray(price, dtype=float)
    return np.where(won, -(price - 1.0), 1.0 - commission)


def cluster_se(x: np.ndarray, groups: np.ndarray) -> float:
    """Cluster-robust SE of the sample mean of x with clusters ``groups``."""
    n = len(x)
    if n < 2:
        return float("nan")
    m = x.mean()
    codes, _ = pd.factorize(groups)
    s = np.bincount(codes, weights=x - m)
    return float(math.sqrt((s * s).sum()) / n)


def ratio_cluster_se(num: np.ndarray, den: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """R = sum(num)/sum(den) with cluster-robust delta-method SE."""
    R = num.sum() / den.sum()
    codes, _ = pd.factorize(groups)
    s = np.bincount(codes, weights=num - R * den)
    return float(R), float(math.sqrt((s * s).sum()) / den.sum())


def cluster_boot(x: np.ndarray, groups: np.ndarray, reps: int, rng: np.random.Generator,
                 weights: np.ndarray | None = None) -> dict:
    """Percentile bootstrap of mean(x) (or sum(x)/sum(weights)) resampling whole clusters."""
    codes, uniq = pd.factorize(groups)
    G = len(uniq)
    S = np.bincount(codes, weights=x, minlength=G)
    W = np.bincount(codes, weights=(np.ones(len(x)) if weights is None else weights), minlength=G)
    stats = np.empty(reps)
    chunk = 500
    for start in range(0, reps, chunk):
        b = min(chunk, reps - start)
        counts = rng.multinomial(G, np.full(G, 1.0 / G), size=b)
        stats[start:start + b] = (counts @ S) / (counts @ W)
    q = np.quantile(stats, [0.025, 0.05, 0.5, 0.95, 0.975])
    return {"clusters": int(G), "reps": int(reps), "q2.5": r5(q[0]), "q5": r5(q[1]), "median": r5(q[2]),
            "q95": r5(q[3]), "q97.5": r5(q[4]), "share_reps_le_0": r5((stats <= 0).mean())}


def load_country(code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["market_id", "selection_id", "start", "date", "field_size", "ltp_pre", "ltp_t30", "bsp", "won", "market_type"]
    df = pd.read_parquet(SPLIT_DIR / f"{code.lower()}_win.parquet", columns=cols)
    df = df.drop_duplicates(["market_id", "selection_id"])
    df = df[(df["market_type"] == "WIN") & df["won"].notna()].drop(columns=["market_type"])
    traded = df["ltp_pre"].notna() & (df["ltp_pre"] > 1)
    mk = df["market_id"]
    all_traded = traded.groupby(mk).transform("min").astype(bool)
    wins = df.groupby("market_id")["won"].transform("sum")
    strict = df[(wins == 1) & all_traded].copy()
    thin = df[(wins == 1) & ~all_traded & traded].copy()
    for d in (strict, thin):
        d["won_b"] = d["won"].to_numpy() == 1
        d["date"] = pd.to_datetime(d["date"])
    return strict, thin


def describe_lay(sel: pd.DataFrame, src: str, mid: pd.Timestamp, boot: int, rng: np.random.Generator,
                 label: str) -> dict:
    """Full description of 'lay 1 unit backer's stake at ``src``' on the rows of ``sel``."""
    price = sel[src].to_numpy(dtype=float)
    ok = np.isfinite(price) & (price > 1)
    sel = sel[ok]
    price = price[ok]
    won = sel["won_b"].to_numpy()
    n = len(sel)
    res = {"cell": label, "source": src, "bets": int(n), "winners": int(won.sum()), "strike": r5(won.mean()),
           "implied_raw_mean_1_over_price": r5((1.0 / price).mean()),
           "expected_winners_if_price_fair": r5((1.0 / price).sum()),
           "back_roi_c0_at_price": r5((price * won).mean() - 1.0)}
    if n < 2:
        return res
    lays = {c: lay_returns(price, won, c) for c in COMMISSIONS}
    for c, l in lays.items():
        res[f"lay_roi_c{int(c * 100)}"] = r5(l.mean())
    l5 = lays[0.05]
    mk = sel["market_id"].to_numpy()
    day = sel["date"].dt.strftime("%Y-%m-%d").to_numpy()
    res["se_per_bet"] = r5(l5.std(ddof=1) / math.sqrt(n))
    res["se_cluster_market"] = r5(cluster_se(l5, mk))
    res["se_cluster_day"] = r5(cluster_se(l5, day))
    res["z_per_bet"] = r5(l5.mean() / (l5.std(ddof=1) / math.sqrt(n)))
    res["z_cluster_day"] = r5(l5.mean() / cluster_se(l5, day))
    first = (sel["date"] < mid).to_numpy()
    halves = []
    for h, mask in enumerate((first, ~first)):
        hh = l5[mask]
        if len(hh) < 2:
            halves.append({"half": h + 1, "bets": int(mask.sum())})
            continue
        se_b = hh.std(ddof=1) / math.sqrt(len(hh))
        se_d = cluster_se(hh, day[mask])
        halves.append({"half": h + 1, "bets": int(len(hh)), "winners": int(won[mask].sum()), "roi_c5": r5(hh.mean()),
                       "se_per_bet": r5(se_b), "z_per_bet": r5(hh.mean() / se_b),
                       "se_cluster_day": r5(se_d), "z_cluster_day": r5(hh.mean() / se_d),
                       "boot_day": cluster_boot(hh, day[mask], boot, rng) if boot else None})
    res["halves"] = halves
    res["both_halves_gt_2se_per_bet"] = bool(all(h.get("z_per_bet") is not None and h["z_per_bet"] > 2 for h in halves))
    res["both_halves_gt_2se_cluster_day"] = bool(all(h.get("z_cluster_day") is not None and h["z_cluster_day"] > 2 for h in halves))
    if boot:
        res["boot_day"] = cluster_boot(l5, day, boot, rng)
    # influence of the most expensive winners
    liab_w = np.sort((price - 1.0)[won])[::-1]
    infl = {}
    for k in (1, 3, 5, 10, 20, 50):
        if k <= len(liab_w):
            infl[f"drop_top_{k}_winners"] = r5((l5.sum() + liab_w[:k].sum()) / n)
    res["influence"] = infl
    res["winner_price_quantiles"] = {q: r5(v) for q, v in zip(("q50", "q90", "q99", "max"), np.quantile(price[won], [0.5, 0.9, 0.99, 1.0]))} if won.any() else None
    res["price_quantiles"] = {q: r5(v) for q, v in zip(("q10", "q50", "q90", "q99", "max"), np.quantile(price, [0.1, 0.5, 0.9, 0.99, 1.0]))}
    # years
    years = sel["date"].dt.year.to_numpy()
    yrs = []
    for y in np.unique(years):
        m = years == y
        hh = l5[m]
        se_b = hh.std(ddof=1) / math.sqrt(len(hh)) if len(hh) > 1 else float("nan")
        yrs.append({"year": int(y), "bets": int(m.sum()), "winners": int(won[m].sum()), "roi_c5": r5(hh.mean()),
                    "se_per_bet": r5(se_b), "se_cluster_day": r5(cluster_se(hh, day[m])) if len(hh) > 1 else None})
    res["years"] = yrs
    # ---- executable forms: liability-based
    liab = price - 1.0
    fixed = l5 / liab  # return per unit of liability with liability 1 on every bet
    fx = {"roi_on_liability": r5(fixed.mean()), "se_per_bet": r5(fixed.std(ddof=1) / math.sqrt(n)),
          "se_cluster_day": r5(cluster_se(fixed, day)), "z_cluster_day": r5(fixed.mean() / cluster_se(fixed, day)),
          "mean_stake_matched_per_unit_liability": r5((1.0 / liab).mean()),
          "halves": []}
    for h, mask in enumerate((first, ~first)):
        hh = fixed[mask]
        se_d = cluster_se(hh, day[mask])
        fx["halves"].append({"half": h + 1, "bets": int(mask.sum()), "roi_on_liability": r5(hh.mean()),
                             "se_cluster_day": r5(se_d), "z_cluster_day": r5(hh.mean() / se_d)})
    if boot:
        fx["boot_day"] = cluster_boot(fixed, day, boot, rng)
    fx["years"] = [{"year": int(y), "roi_on_liability": r5(fixed[years == y].mean()),
                    "se_cluster_day": r5(cluster_se(fixed[years == y], day[years == y]))} for y in np.unique(years)]
    for c in (0.0, 0.08):
        fx[f"roi_on_liability_c{int(c * 100)}"] = r5((lays[c] / liab).mean())
    res["fixed_liability_1_per_bet"] = fx
    R, Rse = ratio_cluster_se(l5, liab, day)
    res["equal_stake_return_per_total_liability"] = {"ratio": r5(R), "se_cluster_day": r5(Rse), "z": r5(R / Rse) if Rse else None,
                                                     "mean_liability": r5(liab.mean()),
                                                     "boot_day": cluster_boot(l5, day, boot, rng, weights=liab) if boot else None}
    return res


def bsp_band_table(sel: pd.DataFrame, mid: pd.Timestamp) -> list[dict]:
    """Decompose the ltp_pre>=50 selection by the BSP it actually settled at."""
    price = sel["bsp"].to_numpy(dtype=float)
    ok = np.isfinite(price) & (price > 1)
    sel = sel[ok]
    price = price[ok]
    won = sel["won_b"].to_numpy()
    l5 = lay_returns(price, won, 0.05)
    day = sel["date"].dt.strftime("%Y-%m-%d").to_numpy()
    first = (sel["date"] < mid).to_numpy()
    band = np.asarray(pd.cut(price, BSP_BANDS, right=False, labels=BSP_LABELS).astype(str))
    rows = []
    for lab in BSP_LABELS:
        m = band == lab
        if m.sum() < 2:
            continue
        hh = l5[m]
        fixed = hh / (price[m] - 1.0)
        rows.append({"bsp_band": lab, "bets": int(m.sum()), "share_of_bets": r5(m.mean()), "winners": int(won[m].sum()),
                     "strike": r5(won[m].mean()), "implied_raw": r5((1.0 / price[m]).mean()),
                     "expected_winners_fair": r5((1.0 / price[m]).sum()),
                     "lay_roi_c5_per_stake": r5(hh.mean()), "se_cluster_day": r5(cluster_se(hh, day[m])),
                     "halves_roi_c5": [r5(hh[first[m]].mean()) if first[m].any() else None,
                                       r5(hh[~first[m]].mean()) if (~first[m]).any() else None],
                     "share_of_total_profit": r5(hh.sum() / l5.sum()) if l5.sum() else None,
                     "fixed_liability_roi": r5(fixed.mean()), "fixed_liability_se_day": r5(cluster_se(fixed, day[m]))})
    return rows


def sp_limit_variants(sel: pd.DataFrame, mid: pd.Timestamp) -> list[dict]:
    """SP lay with a price limit: matched only when BSP <= limit (post-hoc variants)."""
    price = sel["bsp"].to_numpy(dtype=float)
    ok = np.isfinite(price) & (price > 1)
    sel = sel[ok]
    price = price[ok]
    won = sel["won_b"].to_numpy()
    day = sel["date"].dt.strftime("%Y-%m-%d").to_numpy()
    first = (sel["date"] < mid).to_numpy()
    rows = []
    for lim in (60, 80, 100, 150, 200, 300, 1001):
        m = price <= lim
        if m.sum() < 2:
            continue
        l5 = lay_returns(price[m], won[m], 0.05)
        fixed = l5 / (price[m] - 1.0)
        se_s = cluster_se(l5, day[m])
        se_f = cluster_se(fixed, day[m])
        rows.append({"limit": lim, "bets_matched": int(m.sum()), "share_matched": r5(m.mean()), "winners": int(won[m].sum()),
                     "per_stake_roi_c5": r5(l5.mean()), "per_stake_se_day": r5(se_s), "per_stake_z": r5(l5.mean() / se_s),
                     "fixed_liability_roi": r5(fixed.mean()), "fixed_liability_se_day": r5(se_f),
                     "fixed_liability_z": r5(fixed.mean() / se_f) if se_f else None,
                     "fixed_liability_halves": [r5(fixed[first[m]].mean()) if first[m].any() else None,
                                                r5(fixed[~first[m]].mean()) if (~first[m]).any() else None]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default="US,GB,IE,NZ,ZA,FR")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    for code in [c.strip().upper() for c in a.countries.split(",") if c.strip()]:
        strict, thin = load_country(code)
        mid = strict.drop_duplicates("market_id")["date"].median()
        sel = strict[strict["ltp_pre"] >= 50.0]
        out({"section": "cell", "country": code, "family": "strict (every runner traded pre-off)", "mid_date": mid,
             "markets": int(strict["market_id"].nunique()), "selected_rows": int(len(sel)),
             "bets_per_market": {str(k): int(v) for k, v in sel.groupby("market_id").size().value_counts().sort_index().head(8).items()},
             "share_selected_bsp_lt_50": r5((sel["bsp"] < 50).mean()),
             "share_selected_bsp_ge_1000": r5((sel["bsp"] >= 1000).mean()),
             "median_bsp_over_ltp_pre": r5((sel["bsp"] / sel["ltp_pre"]).median())})
        is_us = code == "US"
        boot = a.boot if is_us else max(a.boot // 4, 500)
        out({"section": "lay_bsp", "country": code, **describe_lay(sel, "bsp", mid, boot, rng, "ltp_pre>=50 strict")})
        out({"section": "lay_ltp_pre", "country": code, **describe_lay(sel, "ltp_pre", mid, 0, rng, "ltp_pre>=50 strict")})
        if is_us:
            out({"section": "bsp_band_decomposition", "country": code, "note": "selection ltp_pre>=50, settled at BSP",
                 "rows": bsp_band_table(sel, mid)})
            out({"section": "sp_lay_with_price_limit", "country": code, "note": "POST-HOC variants, not pre-specified",
                 "rows": sp_limit_variants(sel, mid)})
            # the same cell on the thin markets and on all markets together
            tsel = thin[thin["ltp_pre"] >= 50.0]
            out({"section": "lay_bsp_thin", "country": code, **describe_lay(tsel, "bsp", mid, 0, rng, "ltp_pre>=50 thin")})
            allsel = pd.concat([sel, tsel])
            out({"section": "lay_bsp_all_markets", "country": code, **describe_lay(allsel, "bsp", mid, 0, rng, "ltp_pre>=50 strict+thin")})
            # not a pre-race rule: selection by BSP itself, for the calibration story
            for lo, hi in ((50, 100), (100, 200), (200, 1001)):
                s2 = strict[(strict["bsp"] >= lo) & (strict["bsp"] < hi)]
                out({"section": "lay_by_bsp_band_not_pre_race", "country": code,
                     **describe_lay(s2, "bsp", mid, 0, rng, f"bsp {lo}-{hi} strict")})
            # sanity: the same lay at BSP on the <50 bands (expect ~ -5% = commission)
            s3 = strict[(strict["ltp_pre"] >= 15) & (strict["ltp_pre"] < 50)]
            out({"section": "lay_bsp_control_15_50", "country": code, **describe_lay(s3, "bsp", mid, 0, rng, "ltp_pre 15-50 strict")})
        del strict, thin, sel
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

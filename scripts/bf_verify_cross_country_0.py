"""Adversarial verification of the "cross_country" claim:

    US WIN strict markets (every runner traded pre-off, one winner), lay 1 unit of
    backer's stake at BSP on every runner with ltp_pre >= 50:
    n 63982, ROI on stake +15.43% after 5% commission, SE 0.0375, halves 0.1678/0.1438.

    python scripts/bf_verify_cross_country_0.py [--countries US,GB,...] [--boot 400]

Reproduces the cell from the parquet with the same filters as scripts/bf_cross_country.py,
then probes it: duplicate markets per race, rows vs field_size, dead heats, the BSP
distribution (1000 ceiling, BSP < 50 stale-LTP cases), the lay settled with a FIXED
LIABILITY per bet (the only form an SP lay can take on Betfair: MARKET_ON_CLOSE lay
orders carry a liability, not a backer's stake), SP lays with a max-odds limit,
cluster-bootstrap SEs by market and by race day, year and ltp_pre/BSP sub-bands,
largest single-bet contributions, and the same rule extended to the 'thin' markets.
Prints JSON lines only; never prints horse names.
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
SPLIT_DIR = HERE.parent / "data" / "raw" / "betfair" / "split"
COMMISSIONS = (0.0, 0.05, 0.08)


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    raise TypeError(str(type(o)))


def out(obj) -> None:
    print(json.dumps(obj, default=_json_default), flush=True)


def r4(x):
    x = float(x)
    return None if math.isnan(x) else round(x, 4)


def mean_se(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan")), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def lay_stake_returns(price, won, c):
    """Lay 1 unit backer's stake: +1-c when the runner loses, -(price-1) when it wins."""
    price = np.asarray(price, dtype=float)
    return np.where(won, -(price - 1.0), 1.0 - c)


def lay_liab_returns(price, won, c):
    """Lay with 1 unit of LIABILITY (backer's stake = 1/(price-1)): +(1-c)/(price-1) on a loser, -1 on a winner."""
    price = np.asarray(price, dtype=float)
    return np.where(won, -1.0, (1.0 - c) / (price - 1.0))


def summarise(r, first, label):
    m, s = mean_se(r)
    h = [mean_se(r[first]), mean_se(r[~first])]
    return {"cell": label, "bets": int(len(r)), "roi": r4(m), "se": r4(s), "z": r4(m / s) if s else None,
            "halves_roi": [r4(h[0][0]), r4(h[1][0])], "halves_se": [r4(h[0][1]), r4(h[1][1])],
            "halves_bets": [int(first.sum()), int((~first).sum())],
            "both_halves_gt_2se": bool(all(s_ and m_ / s_ > 2 for m_, s_ in h))}


def cluster_boot(r, groups, reps, rng):
    """Bootstrap the mean of r resampling whole clusters (groups = integer codes)."""
    codes, uniq = pd.factorize(groups)
    k = len(uniq)
    sums = np.bincount(codes, weights=r, minlength=k)
    cnts = np.bincount(codes, minlength=k).astype(float)
    means = np.empty(reps)
    for i in range(reps):
        w = rng.multinomial(k, np.full(k, 1.0 / k))
        means[i] = (w * sums).sum() / (w * cnts).sum()
    return float(means.std(ddof=1)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load(code):
    cols = ["market_id", "event_id", "venue", "market_name", "start", "date", "field_size",
            "selection_id", "ltp_pre", "ltp_t30", "bsp", "won", "market_type", "n_winners", "dead_heat_share"]
    df = pd.read_parquet(SPLIT_DIR / f"{code.lower()}_win.parquet", columns=cols)
    n_raw = len(df)
    df = df.drop_duplicates(["market_id", "selection_id"])
    df = df[(df["market_type"] == "WIN") & df["won"].notna()].copy()
    traded = df["ltp_pre"].notna() & (df["ltp_pre"] > 1)
    mk = df["market_id"]
    all_traded = traded.groupby(mk).transform("min").astype(bool)
    wins = df.groupby("market_id")["won"].transform("sum")
    df["all_traded"] = all_traded
    df["wins"] = wins
    df["traded"] = traded
    return df, n_raw


def analyse(code, boot_reps, rng):
    t0 = time.time()
    df, n_raw = load(code)
    # ---- data-quality checks on the settled WIN markets
    per_mk = df.groupby("market_id").agg(rows=("selection_id", "size"), field=("field_size", "first"),
                                         event=("event_id", "first"), start=("start", "first"),
                                         venue=("venue", "first"), name=("market_name", "first"),
                                         wins=("won", "sum"), nw=("n_winners", "first"))
    dup_event_start = per_mk.groupby(["event", "start"]).size()
    dup_venue_start = per_mk.groupby(["venue", "start", "name"]).size()
    dq = {"section": "data_quality", "country": code, "rows_raw": int(n_raw), "rows_settled_win": int(len(df)),
          "markets": int(len(per_mk)),
          "markets_rows_ne_field_size": int((per_mk["rows"] != per_mk["field"]).sum()),
          "markets_sharing_event_and_start": int((dup_event_start > 1).sum()),
          "markets_sharing_venue_start_name": int((dup_venue_start > 1).sum()),
          "dead_heat_markets": int((per_mk["wins"] > 1).sum()),
          "markets_n_winners_ne_1": int((per_mk["nw"] != 1).sum()),
          "share_runners_traded_pre_off": r4(df["traded"].mean()),
          "share_markets_all_traded": r4(df.drop_duplicates("market_id")["all_traded"].mean())}
    out(dq)

    strict = df[(df["wins"] == 1) & df["all_traded"]].copy()
    thin = df[(df["wins"] == 1) & ~df["all_traded"] & df["traded"]].copy()
    mid = strict.drop_duplicates("market_id")["date"].median()
    for name, d in (("strict", strict), ("thin", thin)):
        d.sort_values(["start", "market_id", "ltp_pre", "selection_id"], inplace=True)
        d.reset_index(drop=True, inplace=True)

    def cell_block(d, label):
        sel = d[d["ltp_pre"] >= 50.0]
        okb = sel["bsp"].notna() & (sel["bsp"] > 1)
        sel = sel[okb]
        won = sel["won"].to_numpy() == 1
        bsp = sel["bsp"].to_numpy(dtype=float)
        lp = sel["ltp_pre"].to_numpy(dtype=float)
        first = (sel["date"] < mid).to_numpy()
        res = {"section": "lay50_bsp", "country": code, "set": label, "bets": int(len(sel)),
               "markets": int(sel["market_id"].nunique()), "strike": r4(won.mean()),
               "mean_inv_bsp": r4((1.0 / bsp).mean()), "mean_inv_ltp_pre": r4((1.0 / lp).mean()),
               "sum_bsp_won_over_n": r4((bsp * won).mean()), "sum_ltp_won_over_n": r4((lp * won).mean()),
               "bsp_quantiles_q10_q50_q90_q99": [r4(np.quantile(bsp, q)) for q in (0.1, 0.5, 0.9, 0.99)],
               "share_bsp_ge_1000": r4((bsp >= 1000).mean()), "share_bsp_ge_500": r4((bsp >= 500).mean()),
               "share_bsp_lt_50": r4((bsp < 50).mean()), "share_bsp_lt_25": r4((bsp < 25).mean()),
               "median_bsp_over_ltp": r4(np.median(bsp / lp)), "mean_liability_bsp": r4((bsp - 1).mean()),
               "winners": int(won.sum()), "winners_mean_bsp": r4(bsp[won].mean()) if won.any() else None,
               "winners_median_bsp": r4(np.median(bsp[won])) if won.any() else None,
               "winners_mean_ltp_pre": r4(lp[won].mean()) if won.any() else None}
        for c in COMMISSIONS:
            res[f"lay_stake_roi_c{int(c*100)}"] = r4(lay_stake_returns(bsp, won, c).mean())
        for c in COMMISSIONS:
            res[f"back_bsp_roi_c{int(c*100)}"] = r4(np.where(won, (bsp - 1) * (1 - c), -1.0).mean())
        r5 = lay_stake_returns(bsp, won, 0.05)
        res["stake"] = summarise(r5, first, "lay 1u stake @BSP c5")
        # the same bets laid at ltp_pre (the last trade)
        rl = lay_stake_returns(lp, won, 0.05)
        res["stake_at_ltp_pre"] = summarise(rl, first, "lay 1u stake @ltp_pre c5")
        # fixed LIABILITY per bet (Betfair SP lay orders are liability-based)
        q5 = lay_liab_returns(bsp, won, 0.05)
        res["liability"] = summarise(q5, first, "lay 1u liability @BSP c5 (per-bet)")
        res["liability_aggregate_ratio"] = r4(r5.sum() / (bsp - 1).sum())
        # SP lay with a max-odds limit L: matched only if BSP <= L (LIMIT_ON_CLOSE lay)
        lim = {}
        for L in (100.0, 200.0, 500.0, 999.0):
            m = bsp <= L
            if m.sum() > 1:
                s_ = summarise(r5[m], first[m], f"stake, BSP<={L:g}")
                q_ = summarise(q5[m], first[m], f"liability, BSP<={L:g}")
                lim[f"max_odds_{L:g}"] = {"bets": int(m.sum()), "stake_roi": s_["roi"], "stake_se": s_["se"],
                                         "stake_halves": s_["halves_roi"], "liab_roi": q_["roi"], "liab_se": q_["se"],
                                         "liab_halves": q_["halves_roi"]}
        res["max_odds_limits"] = lim
        # cluster bootstrap SEs
        if len(sel) > 100 and boot_reps:
            se_mk, lo_mk, hi_mk = cluster_boot(r5, sel["market_id"].to_numpy(), boot_reps, rng)
            se_day, lo_day, hi_day = cluster_boot(r5, sel["date"].astype(str).to_numpy(), boot_reps, rng)
            res["boot_se_by_market"] = [r4(se_mk), r4(lo_mk), r4(hi_mk)]
            res["boot_se_by_race_day"] = [r4(se_day), r4(lo_day), r4(hi_day)]
            qse_mk = cluster_boot(q5, sel["market_id"].to_numpy(), boot_reps, rng)
            qse_day = cluster_boot(q5, sel["date"].astype(str).to_numpy(), boot_reps, rng)
            res["liab_boot_se_by_market"] = [r4(v) for v in qse_mk]
            res["liab_boot_se_by_race_day"] = [r4(v) for v in qse_day]
        # by year
        yr = sel["date"].dt.year.to_numpy()
        res["by_year"] = {}
        for y in sorted(set(yr)):
            m = yr == y
            a, b = mean_se(r5[m])
            c_, d_ = mean_se(q5[m])
            res["by_year"][int(y)] = {"bets": int(m.sum()), "stake_roi": r4(a), "stake_se": r4(b),
                                      "liab_roi": r4(c_), "liab_se": r4(d_), "strike": r4(won[m].mean())}
        # by ltp_pre sub-band (selection known pre-off)
        res["by_ltp_band"] = {}
        for lo, hi in ((50, 100), (100, 200), (200, 500), (500, 1000), (1000, 1e9)):
            m = (lp >= lo) & (lp < hi)
            if m.sum() > 1:
                a, b = mean_se(r5[m])
                c_, d_ = mean_se(q5[m])
                res["by_ltp_band"][f"{lo:g}-{hi:g}" if hi < 1e9 else "1000"] = {
                    "bets": int(m.sum()), "strike": r4(won[m].mean()), "mean_inv_bsp": r4((1 / bsp[m]).mean()),
                    "stake_roi": r4(a), "stake_se": r4(b), "liab_roi": r4(c_), "liab_se": r4(d_),
                    "median_bsp": r4(np.median(bsp[m]))}
        # by BSP band (NOT known pre-off: diagnostic of where the stake return comes from)
        res["by_bsp_band_diagnostic"] = {}
        for lo, hi in ((1, 25), (25, 50), (50, 100), (100, 200), (200, 500), (500, 1001)):
            m = (bsp >= lo) & (bsp < hi)
            if m.sum() > 1:
                a, b = mean_se(r5[m])
                res["by_bsp_band_diagnostic"][f"{lo:g}-{hi:g}"] = {
                    "bets": int(m.sum()), "strike": r4(won[m].mean()), "mean_inv_bsp": r4((1 / bsp[m]).mean()),
                    "stake_roi": r4(a), "stake_se": r4(b), "share_of_total_stake_profit": r4(r5[m].sum() / r5.sum()) if r5.sum() else None}
        # influence: largest single winner liabilities as a share of n (each is a hit to ROI)
        wl = np.sort((bsp[won] - 1.0))[::-1]
        res["top10_winner_liabilities_roi_hit_each"] = [r4(v / len(sel)) for v in wl[:10]]
        res["stake_roi_c5_if_top10_winners_excluded"] = r4((r5.sum() + wl[:10].sum()) / (len(sel) - 10))
        # how much of the loss-of-backers is at ltp_pre vs BSP for the same bets
        res["winners_bsp_over_ltp_pre_median"] = r4(np.median(bsp[won] / lp[won])) if won.any() else None
        res["losers_bsp_over_ltp_pre_median"] = r4(np.median(bsp[~won] / lp[~won])) if (~won).any() else None
        # venue concentration of the stake profit
        vs = pd.DataFrame({"v": sel["venue"].to_numpy(), "r": r5}).groupby("v")["r"].agg(["size", "sum", "mean"])
        vs = vs.sort_values("sum", ascending=False)
        res["venues"] = int(len(vs))
        res["top5_venues_share_of_profit"] = r4(vs["sum"].head(5).sum() / r5.sum()) if r5.sum() else None
        res["share_venues_positive"] = r4((vs["mean"] > 0).mean())
        return res, sel

    res_strict, sel_strict = cell_block(strict, "strict (claim)")
    out(res_strict)
    # dead heats included: markets with >1 winner, settled with the dead-heat share
    # (lay of 1u stake at price P with share s: the layer pays s*P-1 on a (part-)winner)
    dh = df[df["all_traded"] & (df["wins"] >= 1)].copy()
    sel = dh[(dh["ltp_pre"] >= 50.0) & dh["bsp"].notna() & (dh["bsp"] > 1)]
    bsp = sel["bsp"].to_numpy(dtype=float)
    share = sel["dead_heat_share"].fillna(0.0).to_numpy(dtype=float)
    won = share > 0
    r5 = np.where(won, -(share * bsp - 1.0), 0.95)
    first = (sel["date"] < mid).to_numpy()
    out({"section": "lay50_bsp_with_dead_heats", "country": code, "bets": int(len(sel)),
         "dead_heat_part_winners": int((won & (share < 1)).sum()),
         **summarise(r5, first, "lay 1u stake @BSP c5, dead heats settled")})
    # selection on the 30-minute-out price instead (present for a liquid minority of runners)
    sel = strict[(strict["ltp_t30"] >= 50.0) & strict["bsp"].notna() & (strict["bsp"] > 1)]
    if len(sel) > 100:
        bsp = sel["bsp"].to_numpy(dtype=float)
        won = sel["won"].to_numpy() == 1
        first = (sel["date"] < mid).to_numpy()
        out({"section": "lay50_bsp_selected_on_ltp_t30", "country": code, "bets": int(len(sel)),
             "share_of_strict_50plus_runners_with_t30": r4(strict.loc[strict["ltp_pre"] >= 50, "ltp_t30"].notna().mean()),
             "strike": r4(won.mean()), "mean_inv_bsp": r4((1 / bsp).mean()),
             "stake": summarise(lay_stake_returns(bsp, won, 0.05), first, "lay 1u stake @BSP c5 (ltp_t30>=50)"),
             "liability": summarise(lay_liab_returns(bsp, won, 0.05), first, "lay 1u liability @BSP c5 (ltp_t30>=50)")})
    # ltp_pre >= 50 runners split by whether the runner had traded 30 min out (liquidity proxy)
    s50 = strict[(strict["ltp_pre"] >= 50.0) & strict["bsp"].notna() & (strict["bsp"] > 1)]
    has = (s50["ltp_t30"].notna() & (s50["ltp_t30"] > 1)).to_numpy()
    for lab, m in (("had_t30_trade", has), ("no_t30_trade", ~has)):
        sub = s50[m]
        if len(sub) > 100:
            bsp = sub["bsp"].to_numpy(dtype=float)
            won = sub["won"].to_numpy() == 1
            first = (sub["date"] < mid).to_numpy()
            out({"section": "lay50_bsp_by_t30_liquidity", "country": code, "group": lab, "bets": int(len(sub)),
                 "strike": r4(won.mean()), "mean_inv_bsp": r4((1 / bsp).mean()),
                 "stake": summarise(lay_stake_returns(bsp, won, 0.05), first, "stake"),
                 "liability": summarise(lay_liab_returns(bsp, won, 0.05), first, "liability")})
    if thin["market_id"].nunique() >= 200:
        res_thin, sel_thin = cell_block(thin, "thin (markets with an untraded runner)")
        out(res_thin)
        both = pd.concat([strict, thin], ignore_index=True)
        res_all, _ = cell_block(both, "strict+thin (rule applied to every settled single-winner market)")
        out(res_all)
    out({"section": "timing", "country": code, "seconds": round(time.time() - t0, 1), "mid_date": mid})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", default="US,GB,IE,ZA,FR,NZ")
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    t0 = time.time()
    for code in [c.strip().upper() for c in a.countries.split(",") if c.strip()]:
        analyse(code, a.boot, rng)
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

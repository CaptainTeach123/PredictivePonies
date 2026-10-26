"""Adversarial verification (statistics + execution lens) of the "bsp_and_laying"
A2 walk-forward chooser claim:

    AU gallops WIN, every runner, equal stakes: each month, per ltp_pre band x
    field-size cell, take Betfair SP if the paired SP-minus-last-price
    difference over strictly earlier months is > 1 SE (6-month burn-in,
    >= 30 earlier bets), else take the last traded price.  Gain over
    always-last-price +0.24% +/- 0.04% of stake (halves +0.44% / +0.11%),
    gain over always-SP +0.80% +/- 0.17% (halves +0.66% / +0.90%).

    python scripts/bf_verify_bsp_and_laying_1_chooser.py [--end 2018-06-01] [--boot 400] [--splits 200]

The data filters, the price ladder and the chooser arithmetic are imported from
scripts/bf_bsp_and_laying.py (never edited), so section 1 is an exact
reproduction.  What is added:

  1. reproduction of the k = 0 / 1 / 2 chooser (bets, SP share, ROI at
     commission 0 / 5 / 8 / 10%, gain vs always-LTP and vs always-SP with the
     original per-bet SE and the original median-day halves);
  2. cluster-robust SEs for both gains: by market, meeting (event_id), race
     day and venue-day; a race-day cluster bootstrap; month-as-unit t-test and
     sign test over the evaluated months; the halves with day-clustered SEs;
  3. time stability: gain by calendar year and half-year; 200 random day-level
     half splits (how often "both halves > 2 SE" holds with per-bet and with
     day-clustered SEs); a balanced split at the median evaluated day; a single
     fit-on-one-half / evaluate-on-the-other version of the rule;
  4. decomposition of both gains by ltp_pre band (which cells carry them);
  5. execution: the same choices with the last-price leg matched one tick worse
     than the last trade (the $50+ and $1-3 legs are where the chooser takes the
     last price), the walk-forward rule refitted against the tick-worse price,
     and the break-even share of last-price legs that must be filled at exactly
     the last trade for the chooser to beat always-SP;
  6. stale-quote diagnostics for the gain vs LTP: by market book at ltp_pre, by
     market liquidity (share of runners traded 30 min out), by field size;
     share of the gain from winners whose SP was > 25% / 50% above the last
     trade; winsorised gain; level shift (mean bsp/ltp on all runners vs on
     winners in the SP-chosen cells);
  7. multiple testing: Bonferroni over the 24 chooser cells and the 556 cells
     of the original family with the day-clustered z.

Prints JSON lines with summary numbers only; never prints horse names.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def _import_original():
    spec = importlib.util.spec_from_file_location("bf_bsp_and_laying", HERE / "bf_bsp_and_laying.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


O = _import_original()
COMMISSIONS = (0.0, 0.05, 0.08, 0.10)
C_TEST = 0.05
BURN_IN = O.BURN_IN_MONTHS
MIN_FIT = O.MIN_FIT_BETS
N_CHOOSER_CELLS = 24
N_CELLS_ORIGINAL = 556
BAND_LABELS = O.BAND_LABELS


# ----------------------------------------------------------------------------- utilities


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, pd.Period)):
        return str(o)
    raise TypeError(str(type(o)))


def out(obj) -> None:
    print(json.dumps(obj, default=_json_default), flush=True)


def r4(x):
    x = float(x)
    return None if math.isnan(x) else round(x, 4)


def r6(x):
    x = float(x)
    return None if math.isnan(x) else round(x, 6)


def mean_se(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else float("nan")), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(n))


def cluster_se(ret: np.ndarray, cl) -> float:
    """SE of the mean with observations clustered by ``cl`` (Liang-Zeger, g/(g-1))."""
    n = len(ret)
    if n < 2:
        return float("nan")
    d = ret - ret.mean()
    s = pd.Series(d).groupby(np.asarray(cl)).sum().to_numpy()
    g = len(s)
    if g < 2:
        return float("nan")
    return float(math.sqrt((s ** 2).sum() * g / (g - 1)) / n)


def cluster_boot(ret: np.ndarray, cl, n_boot: int, seed: int = 0) -> dict:
    df = pd.DataFrame({"r": ret, "c": np.asarray(cl)})
    agg = df.groupby("c")["r"].agg(["sum", "count"])
    s, k = agg["sum"].to_numpy(), agg["count"].to_numpy()
    g = len(s)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, g, g)
        means[b] = s[idx].sum() / k[idx].sum()
    return {"boot_se": r6(means.std(ddof=1)), "ci95": [r6(np.quantile(means, 0.025)), r6(np.quantile(means, 0.975))],
            "share_boot_below_0": r4((means <= 0).mean()), "clusters": int(g)}


def stat(ret: np.ndarray, date=None, won=None) -> dict:
    m, se = mean_se(ret)
    res = {"bets": int(len(ret)), "mean": r6(m), "se": r6(se), "z": r4(m / se) if se and se > 0 else None}
    if date is not None:
        sd = cluster_se(ret, date)
        res["se_cl_date"] = r6(sd)
        res["z_cl_date"] = r4(m / sd) if sd and sd > 0 else None
    if won is not None:
        res["n_won"] = int(won.sum())
    return res


def _safe_z(g: np.ndarray, cl) -> float | None:
    se = cluster_se(g, cl)
    return r4(g.mean() / se) if se and se > 0 and not math.isnan(se) else None


def month_unit(ret: np.ndarray, month) -> dict:
    g = pd.Series(ret).groupby(np.asarray(month))
    mm = g.mean()
    n = len(mm)
    se = float(mm.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return {"months": int(n), "mean_of_monthly_means": r6(mm.mean()), "se_across_months": r6(se),
            "t": r4(mm.mean() / se) if se and se > 0 else None,
            "months_positive": int((mm > 0).sum()), "months_negative": int((mm < 0).sum()),
            "sign_test_p_one_sided": r4(_binom_tail(int((mm > 0).sum()), int((mm != 0).sum())))}


def _binom_tail(k: int, n: int) -> float:
    """P(X >= k) for X ~ Bin(n, 1/2)."""
    if n == 0:
        return float("nan")
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n


# ----------------------------------------------------------------------------- data


def load(path: Path, end: str | None):
    df, info, mid = O.load(path, end)
    extra = pd.read_parquet(path, columns=["market_id", "event_id", "venue"]).drop_duplicates("market_id")
    df = df.merge(extra, on="market_id", how="left")
    sub = df[~df["harness"]].reset_index(drop=True)
    sub["book"] = (1.0 / sub["ltp_pre"]).groupby(sub["market_id"]).transform("sum")
    sub["mkt_t30_share"] = sub["has_t30"].astype(float).groupby(sub["market_id"]).transform("mean")
    sub["venue_day"] = sub["venue"].fillna("") + "|" + sub["date"].astype(str)
    return sub, info, mid


# ----------------------------------------------------------------------------- chooser


def walk_forward(d_fit: np.ndarray, mi: np.ndarray, ki: np.ndarray, n_months: int, n_keys: int, k: float) -> np.ndarray:
    """Exactly the original A2 arithmetic: per month x cell, take SP when the mean
    of ``d_fit`` over strictly earlier months is > k SE (>= MIN_FIT earlier bets)."""
    shape = (n_months, n_keys)
    S, SS, N = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    np.add.at(S, (mi, ki), d_fit)
    np.add.at(SS, (mi, ki), d_fit * d_fit)
    np.add.at(N, (mi, ki), 1.0)
    cS = np.vstack([np.zeros((1, n_keys)), np.cumsum(S, axis=0)[:-1]])
    cSS = np.vstack([np.zeros((1, n_keys)), np.cumsum(SS, axis=0)[:-1]])
    cN = np.vstack([np.zeros((1, n_keys)), np.cumsum(N, axis=0)[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = cS / cN
        var = (cSS - cS * cS / cN) / np.maximum(cN - 1.0, 1.0)
        se = np.sqrt(var / cN)
        take = (cN >= MIN_FIT) & (mean > k * se)
    return take[mi, ki]


class Eval:
    """Per-bet arrays on the chooser's evaluation universe (bsp known, month index >= burn-in)."""

    def __init__(self, sub: pd.DataFrame, universe: str = "all runners"):
        mask = sub["bsp_ok"].to_numpy().copy()
        if universe == "favourites":
            mask &= sub["fav"].to_numpy()
        idx_all = np.flatnonzero(mask)
        u = sub.iloc[idx_all]
        key = (u["band"] + "|" + u["fgroup"]).to_numpy()
        month = u["month"].to_numpy()
        self.months = np.array(sorted(pd.unique(month)))
        keys = np.array(sorted(pd.unique(key)))
        self.mi_all = pd.Index(self.months).get_indexer(month)
        self.ki_all = pd.Index(keys).get_indexer(key)
        self.n_keys = len(keys)
        self.won_all = u["won_b"].to_numpy()
        self.lp_all = u["ltp_pre"].to_numpy(dtype=float)
        self.bsp_all = u["bsp"].to_numpy(dtype=float)
        self.tick_all = O.tick_down(self.lp_all)
        self.fit_ltp = O.back_returns(self.bsp_all, self.won_all, C_TEST) - O.back_returns(self.lp_all, self.won_all, C_TEST)
        self.fit_tick = O.back_returns(self.bsp_all, self.won_all, C_TEST) - O.back_returns(self.tick_all, self.won_all, C_TEST)
        self.evaluable = self.mi_all >= BURN_IN
        ev = np.flatnonzero(self.evaluable)
        self.u = u.iloc[ev].reset_index(drop=True)
        self.won = self.won_all[ev]
        self.lp, self.bsp, self.tick = self.lp_all[ev], self.bsp_all[ev], self.tick_all[ev]
        self.mi = self.mi_all[ev]
        self.back = {"ltp": {c: O.back_returns(self.lp, self.won, c) for c in COMMISSIONS},
                     "sp": {c: O.back_returns(self.bsp, self.won, c) for c in COMMISSIONS},
                     "tick": {c: O.back_returns(self.tick, self.won, c) for c in COMMISSIONS}}
        self.ev = ev

    def choice(self, k: float, fit: str = "ltp") -> np.ndarray:
        d = self.fit_ltp if fit == "ltp" else self.fit_tick
        take = walk_forward(d, self.mi_all, self.ki_all, len(self.months), self.n_keys, k)
        return take[self.ev]

    def ret(self, ch: np.ndarray, c: float, ltp_leg: str = "ltp") -> np.ndarray:
        return np.where(ch, self.back["sp"][c], self.back[ltp_leg][c])


def compact(ret_by_c: dict, first: np.ndarray) -> dict:
    r5 = ret_by_c[C_TEST]
    m, se = mean_se(r5)
    h = [mean_se(r5[first]), mean_se(r5[~first])]
    return {"roi": {f"c{int(c * 100)}": r6(ret_by_c[c].mean()) for c in COMMISSIONS},
            "se_c5": r6(se), "z": r4(m / se) if se and se > 0 else None,
            "halves_c5": [r6(h[0][0]), r6(h[1][0])], "halves_se_c5": [r6(h[0][1]), r6(h[1][1])],
            "halves_z": [r4(h[0][0] / h[0][1]) if h[0][1] else None, r4(h[1][0] / h[1][1]) if h[1][1] else None],
            "positive_both_halves_2se": bool(all(hm > 2 * hs for hm, hs in h if hs and hs > 0) and len(h) == 2)}


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=HERE.parent / "data/raw/betfair/split/au_win.parquet")
    ap.add_argument("--end", default=None)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--splits", type=int, default=200)
    a = ap.parse_args()
    t0 = time.time()
    sub, info, mid = load(a.path, a.end)
    out({"section": "data", "code": "gallops", "markets": int(sub["market_id"].nunique()), "runners": int(len(sub)),
         "from": info["from"], "to": info["to"], "mid_date_for_halves": str(mid.date()), "end": a.end,
         "bsp_runner_coverage": r4(sub["bsp_ok"].mean())})

    E = Eval(sub, "all runners")
    u = E.u
    first = (u["date"] < mid).to_numpy()
    date = u["date"].to_numpy()
    month = u["month"].astype(str).to_numpy()
    band = u["band"].to_numpy()
    year = u["date"].dt.year.to_numpy()
    halfyear = (u["date"].dt.year.astype(str) + "H" + ((u["date"].dt.month > 6).astype(int) + 1).astype(str)).to_numpy()

    # ---- 1. reproduction ---------------------------------------------------------------------------
    choices = {k: E.choice(k, "ltp") for k in (0.0, 1.0, 2.0)}
    rep = {}
    for k, ch in choices.items():
        ret_ch = {c: E.ret(ch, c) for c in COMMISSIONS}
        g_ltp = {c: ret_ch[c] - E.back["ltp"][c] for c in COMMISSIONS}
        g_sp = {c: ret_ch[c] - E.back["sp"][c] for c in COMMISSIONS}
        rep[f"k={k:g}"] = {"bets": int(len(ch)), "share_sp_chosen": r4(ch.mean()),
                           "share_sp_by_band": {b: r4(ch[band == b].mean()) if (band == b).any() else None for b in BAND_LABELS},
                           "roi_chooser": {f"c{int(c * 100)}": r6(ret_ch[c].mean()) for c in COMMISSIONS},
                           "roi_always_ltp_c5": r6(E.back["ltp"][C_TEST].mean()), "roi_always_sp_c5": r6(E.back["sp"][C_TEST].mean()),
                           "gain_vs_ltp": compact(g_ltp, first), "gain_vs_sp": compact(g_sp, first)}
    out({"section": "1_reproduction", "universe": "gallops all runners", "months_evaluated": int(len(E.months) - BURN_IN),
         "halves_bets": [int(first.sum()), int((~first).sum())], "claim": {"k=1 gain_vs_ltp": "+0.0024 +/- 0.0004 [0.0044, 0.0011]",
                                                                             "k=1 gain_vs_sp": "+0.0080 +/- 0.0017 [0.0066, 0.0090]"},
         "results": rep})

    ch = choices[1.0]
    ret1 = {c: E.ret(ch, c) for c in COMMISSIONS}
    gl = {c: ret1[c] - E.back["ltp"][c] for c in COMMISSIONS}
    gs = {c: ret1[c] - E.back["sp"][c] for c in COMMISSIONS}
    gl5, gs5 = gl[C_TEST], gs[C_TEST]

    # ---- 2. clustered SEs -------------------------------------------------------------------------
    cl = {}
    for name, g in (("gain_vs_ltp", gl5), ("gain_vs_sp", gs5)):
        m, se = mean_se(g)
        res = {"mean": r6(m), "se_per_bet": r6(se),
               "se_cl_market": r6(cluster_se(g, u["market_id"].to_numpy())),
               "se_cl_event": r6(cluster_se(g, u["event_id"].to_numpy())),
               "se_cl_venue_day": r6(cluster_se(g, u["venue_day"].to_numpy())),
               "se_cl_date": r6(cluster_se(g, date))}
        res["z_cl_date"] = r4(m / res["se_cl_date"]) if res["se_cl_date"] else None
        res["halves"] = []
        for mask in (first, ~first):
            hm, hs = mean_se(g[mask])
            sd = cluster_se(g[mask], date[mask])
            res["halves"].append({"bets": int(mask.sum()), "mean": r6(hm), "se_per_bet": r6(hs), "z_per_bet": r4(hm / hs) if hs else None,
                                  "se_cl_date": r6(sd), "z_cl_date": r4(hm / sd) if sd else None,
                                  "day_cluster_bootstrap": cluster_boot(g[mask], date[mask], a.boot, seed=1)})
        res["day_cluster_bootstrap"] = cluster_boot(g, date, a.boot)
        res["month_as_unit"] = month_unit(g, month)
        res["both_halves_over_2se_cl_date"] = bool(all(h["z_cl_date"] is not None and h["z_cl_date"] > 2 for h in res["halves"]))
        cl[name] = res
    out({"section": "2_clustered_se_c5", "k": 1, **cl,
         "note": "the gain is nonzero only on winners where the chooser's leg differs, and a market has one winner, so market "
                 "clusters hold at most one nonzero term; meeting / day clusters are the relevant ones"})

    # ---- 3. time stability --------------------------------------------------------------------------
    def by_group(g: np.ndarray, grp: np.ndarray, label: str) -> dict:
        res = {}
        for v in sorted(pd.unique(grp)):
            mask = grp == v
            s = stat(g[mask], date[mask], E.won[mask])
            s["share_sp_chosen"] = r4(ch[mask].mean())
            res[str(v)] = s
        return res

    ts = {"gain_vs_ltp_by_year": by_group(gl5, year, "year"), "gain_vs_sp_by_year": by_group(gs5, year, "year"),
          "gain_vs_ltp_by_halfyear": by_group(gl5, halfyear, "h"), "gain_vs_sp_by_halfyear": by_group(gs5, halfyear, "h")}
    # balanced split at the median evaluated day
    days = np.sort(pd.unique(date))
    mid_eval = days[len(days) // 2]
    f2 = date < mid_eval
    for name, g in (("gain_vs_ltp", gl5), ("gain_vs_sp", gs5)):
        ts[f"{name}_balanced_halves_at_{pd.Timestamp(mid_eval).date()}"] = [stat(g[f2], date[f2]), stat(g[~f2], date[~f2])]
    # random day-level half splits from per-day aggregates
    rng = np.random.default_rng(7)
    agg = pd.DataFrame({"d": date, "gl": gl5, "gl2": gl5 ** 2, "gs": gs5, "gs2": gs5 ** 2, "n": 1.0}).groupby("d").sum()
    nd = len(agg)
    splits = {}
    for name in ("gl", "gs"):
        S, SS, N = agg[name].to_numpy(), agg[name + "2"].to_numpy(), agg["n"].to_numpy()
        both_pb, both_cl = 0, 0
        for _ in range(a.splits):
            pick = rng.random(nd) < 0.5
            ok_pb, ok_cl = True, True
            for mask in (pick, ~pick):
                n = N[mask].sum()
                m = S[mask].sum() / n
                var = (SS[mask].sum() - n * m * m) / (n - 1)
                se_pb = math.sqrt(var / n)
                dev = S[mask] - N[mask] * m
                gg = mask.sum()
                se_cl = math.sqrt((dev ** 2).sum() * gg / (gg - 1)) / n
                ok_pb &= m > 2 * se_pb
                ok_cl &= m > 2 * se_cl
            both_pb += ok_pb
            both_cl += ok_cl
        splits[name] = {"splits": a.splits, "share_both_halves_over_2se_per_bet": r4(both_pb / a.splits),
                        "share_both_halves_over_2se_cl_date": r4(both_cl / a.splits)}
    ts["random_day_half_splits"] = {"gain_vs_ltp": splits["gl"], "gain_vs_sp": splits["gs"]}
    # single fit on one half, evaluate on the other (band x field cells, > 1 SE rule, >= 30 bets)
    key_all = (u["band"] + "|" + u["fgroup"]).to_numpy()
    fit_d = E.fit_ltp[E.ev]
    xh = {}
    for h, (fit_mask, ev_mask) in {"fit_half1_eval_half2": (first, ~first), "fit_half2_eval_half1": (~first, first)}.items():
        fr = pd.DataFrame({"k": key_all[fit_mask], "d": fit_d[fit_mask]}).groupby("k")["d"].agg(["mean", "std", "count"])
        take_keys = set(fr.index[(fr["count"] >= MIN_FIT) & (fr["mean"] > fr["std"] / np.sqrt(fr["count"]))])
        chx = np.isin(key_all, list(take_keys)) & ev_mask
        g = np.where(chx, E.back["sp"][C_TEST], E.back["ltp"][C_TEST])[ev_mask] - E.back["ltp"][C_TEST][ev_mask]
        g2 = np.where(chx, E.back["sp"][C_TEST], E.back["ltp"][C_TEST])[ev_mask] - E.back["sp"][C_TEST][ev_mask]
        xh[h] = {"cells_sp": len(take_keys), "share_sp_chosen": r4(chx[ev_mask].mean()),
                 "gain_vs_ltp": stat(g, date[ev_mask]), "gain_vs_sp": stat(g2, date[ev_mask])}
    ts["fit_one_half_eval_other"] = xh
    out({"section": "3_time_stability", "k": 1, **ts})

    # ---- 4. decomposition by band ---------------------------------------------------------------------
    n_all = len(ch)
    dec = {}
    for b in BAND_LABELS:
        m = band == b
        if not m.any():
            continue
        dec[b] = {"bets": int(m.sum()), "share_of_bets": r4(m.mean()), "share_sp_chosen": r4(ch[m].mean()),
                  "gain_vs_ltp_in_band": r6(gl5[m].mean()), "gain_vs_ltp_se": r6(mean_se(gl5[m])[1]),
                  "contrib_to_total_gain_vs_ltp": r6(gl5[m].sum() / n_all),
                  "gain_vs_sp_in_band": r6(gs5[m].mean()), "gain_vs_sp_se": r6(mean_se(gs5[m])[1]),
                  "contrib_to_total_gain_vs_sp": r6(gs5[m].sum() / n_all)}
    out({"section": "4_decomposition_by_band", "k": 1, "total_gain_vs_ltp": r6(gl5.mean()), "total_gain_vs_sp": r6(gs5.mean()),
         "rows": dec})

    # ---- 5. execution: last-price leg matched one tick worse --------------------------------------------
    ex = {}
    ret_tick = {c: E.ret(ch, c, "tick") for c in COMMISSIONS}
    ex["chooser_ltp_leg_at_tick_down"] = {
        "roi": {f"c{int(c * 100)}": r6(ret_tick[c].mean()) for c in COMMISSIONS},
        "roi_always_tick_down_c5": r6(E.back["tick"][C_TEST].mean()), "roi_always_sp_c5": r6(E.back["sp"][C_TEST].mean()),
        "gain_vs_always_sp": compact({c: ret_tick[c] - E.back["sp"][c] for c in COMMISSIONS}, first),
        "gain_vs_always_sp_se_cl_date": r6(cluster_se(ret_tick[C_TEST] - E.back["sp"][C_TEST], date)),
        "gain_vs_always_tick_down": compact({c: ret_tick[c] - E.back["tick"][c] for c in COMMISSIONS}, first),
        "always_sp_vs_always_tick_down": compact({c: E.back["sp"][c] - E.back["tick"][c] for c in COMMISSIONS}, first)}
    # break-even share f of last-price legs filled at exactly the last trade (rest one tick worse)
    g_exact = gs5.mean()
    g_tick = (ret_tick[C_TEST] - E.back["sp"][C_TEST]).mean()
    ex["breakeven_share_of_ltp_legs_filled_at_last_trade"] = r4(-g_tick / (g_exact - g_tick)) if g_exact != g_tick else None
    # by band, the LTP-leg bets: paired (ltp - sp) and (tick - sp)
    legs = {}
    for b in BAND_LABELS:
        m = (band == b) & ~ch
        if m.sum() < 100:
            continue
        legs[b] = {"ltp_leg_bets": int(m.sum()), "ltp_minus_sp_c5": r6((E.back["ltp"][C_TEST] - E.back["sp"][C_TEST])[m].mean()),
                   "tick_minus_sp_c5": r6((E.back["tick"][C_TEST] - E.back["sp"][C_TEST])[m].mean()),
                   "median_tick_size_pct_of_price": r4(np.median((E.lp[m] - E.tick[m]) / E.lp[m]) * 100)}
    ex["ltp_legs_by_band"] = legs
    # refit the walk-forward rule against the tick-worse last price
    cht = E.choice(1.0, "tick")
    rt = {c: E.ret(cht, c, "tick") for c in COMMISSIONS}
    ex["chooser_refit_vs_tick_down"] = {"share_sp_chosen": r4(cht.mean()),
                                        "share_sp_by_band": {b: r4(cht[band == b].mean()) if (band == b).any() else None for b in BAND_LABELS},
                                        "roi": {f"c{int(c * 100)}": r6(rt[c].mean()) for c in COMMISSIONS},
                                        "gain_vs_always_sp": compact({c: rt[c] - E.back["sp"][c] for c in COMMISSIONS}, first),
                                        "gain_vs_always_tick_down": compact({c: rt[c] - E.back["tick"][c] for c in COMMISSIONS}, first)}
    out({"section": "5_execution_tick_worse", "k": 1, **ex,
         "note": "ltp_pre is a last trade from one-minute snapshots, not an available price; a backer matched one tick worse "
                 "than the last trade is the conservative last-price execution"})

    # ---- 6. stale-quote diagnostics for the gain vs LTP ---------------------------------------------------
    sq = {}
    book = u["book"].to_numpy()
    bk = np.select([np.abs(book - 1) <= 0.01, np.abs(book - 1) <= 0.03], ["|book-1|<=1%", "1-3%"], ">3%")
    sq["by_market_book_at_ltp"] = {}
    for lab in ("|book-1|<=1%", "1-3%", ">3%"):
        m = bk == lab
        sq["by_market_book_at_ltp"][lab] = {**stat(gl5[m], date[m]), "share_of_bets": r4(m.mean()), "share_sp_chosen": r4(ch[m].mean()),
                                            "gain_vs_sp": r6(gs5[m].mean())}
    liq = u["mkt_t30_share"].to_numpy()
    lq = np.select([liq == 0, liq < 1], ["no runner traded t30", "some traded t30"], "all traded t30")
    sq["by_market_liquidity_t30"] = {}
    for lab in ("no runner traded t30", "some traded t30", "all traded t30"):
        m = lq == lab
        if m.any():
            sq["by_market_liquidity_t30"][lab] = {**stat(gl5[m], date[m]), "share_of_bets": r4(m.mean()), "share_sp_chosen": r4(ch[m].mean()),
                                                  "gain_vs_sp": r6(gs5[m].mean())}
    fg = u["fgroup"].to_numpy()
    sq["by_field_size"] = {f: {**stat(gl5[fg == f], date[fg == f]), "share_sp_chosen": r4(ch[fg == f].mean()), "gain_vs_sp": r6(gs5[fg == f].mean())}
                           for f in O.FIELD_LABELS if (fg == f).any()}
    # influence of large SP drifts on winners in SP-chosen bets
    ratio = E.bsp / E.lp
    w_sp = ch & E.won
    tot = gl5.sum()
    infl = {}
    for thr in (1.1, 1.25, 1.5, 2.0):
        m = w_sp & (ratio > thr)
        infl[f"ratio>{thr:g}"] = {"winners": int(m.sum()), "share_of_sp_chosen_winners": r4(m.sum() / max(w_sp.sum(), 1)),
                                  "share_of_total_gain_vs_ltp": r4(gl5[m].sum() / tot) if tot else None}
    wins_gain = np.where(ch, (np.minimum(E.bsp, 1.5 * E.lp) - E.lp) * (1 - C_TEST), 0.0) * E.won
    wins_gain2 = np.where(ch, (np.minimum(E.bsp, 1.25 * E.lp) - E.lp) * (1 - C_TEST), 0.0) * E.won
    sq["influence_of_large_drifts"] = infl
    sq["gain_vs_ltp_winsorised_ratio_1.5"] = stat(wins_gain, date)
    sq["gain_vs_ltp_winsorised_ratio_1.25"] = stat(wins_gain2, date)
    # level shift vs information: in the SP-chosen bets, mean bsp/ltp on all runners vs on winners
    sq["sp_chosen_bets"] = {"bets": int(ch.sum()), "winners": int(w_sp.sum()),
                            "mean_ratio_all": r4(ratio[ch].mean()), "median_ratio_all": r4(np.median(ratio[ch])),
                            "mean_log_ratio_all": r4(np.log(ratio[ch]).mean()),
                            "mean_ratio_winners": r4(ratio[w_sp].mean()), "median_ratio_winners": r4(np.median(ratio[w_sp])),
                            "mean_ratio_losers": r4(ratio[ch & ~E.won].mean()),
                            "share_sp_better_all": r4((ratio[ch] > 1).mean()), "share_sp_better_winners": r4((ratio[w_sp] > 1).mean()),
                            "strike": r4(E.won[ch].mean()), "implied_at_ltp": r4((1 / E.lp[ch]).mean()), "implied_at_bsp": r4((1 / E.bsp[ch]).mean()),
                            "gain_vs_ltp_per_sp_chosen_bet": r6(gl5[ch].mean()),
                            "mean_bsp_minus_ltp_on_winners": r4((E.bsp - E.lp)[w_sp].mean())}
    out({"section": "6_stale_quote_diagnostics", "k": 1, **sq})

    # ---- 7. multiple testing -------------------------------------------------------------------------------
    z_cl = cl["gain_vs_ltp"]["z_cl_date"]
    p_cl = O.one_sided_p(z_cl) if z_cl is not None else float("nan")
    z_cl_sp = cl["gain_vs_sp"]["z_cl_date"]
    p_cl_sp = O.one_sided_p(z_cl_sp) if z_cl_sp is not None else float("nan")
    out({"section": "7_multiple_testing",
         "gain_vs_ltp": {"z_cl_date": z_cl, "p_one_sided": p_cl, "bonferroni_24_chooser_cells": min(1.0, p_cl * N_CHOOSER_CELLS),
                         "bonferroni_556_cells": min(1.0, p_cl * N_CELLS_ORIGINAL)},
         "gain_vs_sp": {"z_cl_date": z_cl_sp, "p_one_sided": p_cl_sp, "bonferroni_24_chooser_cells": min(1.0, p_cl_sp * N_CHOOSER_CELLS),
                        "bonferroni_556_cells": min(1.0, p_cl_sp * N_CELLS_ORIGINAL)}})

    # ---- 8. the other claimed universes, briefly (gallops favourites, harness) -------------------------------
    others = {}
    for code, universe in (("gallops", "favourites"), ("harness", "all runners"), ("harness", "favourites")):
        df_h = sub if code == "gallops" else None
        if code == "harness":
            df_all, _, _ = O.load(a.path, a.end)
            df_h = df_all[df_all["harness"]].reset_index(drop=True)
            if df_h.empty:
                continue
        Eh = Eval(df_h, universe)
        fh = (Eh.u["date"] < mid).to_numpy()
        dh = Eh.u["date"].to_numpy()
        res = {}
        for k in (0.0, 1.0, 2.0):
            c_ = Eh.choice(k)
            r_ = {c: Eh.ret(c_, c) for c in COMMISSIONS}
            g1 = r_[C_TEST] - Eh.back["ltp"][C_TEST]
            g2 = r_[C_TEST] - Eh.back["sp"][C_TEST]
            res[f"k={k:g}"] = {"bets": int(len(c_)), "share_sp_chosen": r4(c_.mean()),
                               "gain_vs_ltp": {**stat(g1, dh), "halves": [r6(g1[fh].mean()), r6(g1[~fh].mean())],
                                               "halves_z_cl_date": [_safe_z(g1[fh], dh[fh]), _safe_z(g1[~fh], dh[~fh])]},
                               "gain_vs_sp": {**stat(g2, dh), "halves": [r6(g2[fh].mean()), r6(g2[~fh].mean())]}}
        others[f"{code} {universe}"] = res
    out({"section": "8_other_universes", **others})
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

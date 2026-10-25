"""Experiment "win_to_place": Betfair AU WIN market -> fair PLACE price via
discounted Harville (ponies.exotics.order_probs), compared with the PLACE
market; and the mirror (does the PLACE market carry information about the
winner beyond the WIN price?).

    python scripts/bf_win_to_place.py [--dir data/raw/betfair/split] [--frac 0.1]

Walk-forward: lam2/lam3 (per code: gallops/harness) and the blend weight are
fitted each month on strictly earlier months only. Bets are 1 unit backed at
the place market's last pre-race traded price [A-53] with commission on net
winnings [A-54]; every cell is reported, with per-bet SE, ROI in both halves
of the period, commission 0/5/8%, plus two execution checks (one tick worse
than ltp_pre; decide on ltp_pre but settle at the place BSP). Dead heats in
the place market are settled with a proportional stake reduction
(n_winners / n_placed) on every placed runner of that market.
Prints summary numbers only (never horse names or rows).
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ponies.betfair import net_return  # noqa: E402
from ponies.exotics import LAMBDA_GRID, order_probs  # noqa: E402
from ponies.model import ConditionalLogit  # noqa: E402

COLS = ["market_id", "event_id", "market_name", "start", "date", "n_winners", "field_size",
        "selection_id", "ltp_pre", "ltp_t30", "bsp", "won"]
PLACE_BANDS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.5, 5.0, 8.0, 15.0, 1e9]
KS = [0.0, 0.05, 0.1, 0.2]
COMMS = [0.0, 0.05, 0.08]
MIN_FIT_RACES = 2000
BLEND_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1000.0, 10.0)]


def out(obj) -> None:
    print(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def r4(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


def one_tick_worse(price: np.ndarray) -> np.ndarray:
    """Next price below ``price`` on the Betfair ladder (a back bet filled one tick worse)."""
    p = np.asarray(price, dtype=float)
    step = np.full_like(p, 10.0)
    for lo, hi, inc in TICKS:
        step = np.where((p > lo) & (p <= hi), inc, step)
    return np.maximum(np.round(p - step, 2), 1.01)


def band_labels(bands):
    return [f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+" for lo, hi in zip(bands, bands[1:])]


# ---------------------------------------------------------------------------
# loading / joining
# ---------------------------------------------------------------------------
def load(d: Path, frac: float, seed: int) -> pd.DataFrame:
    w = pd.read_parquet(d / "au_win.parquet", columns=COLS).drop_duplicates(["market_id", "selection_id"])
    p = pd.read_parquet(d / "au_place.parquet", columns=COLS).drop_duplicates(["market_id", "selection_id"])
    for df in (w, p):
        for c in ("market_id", "event_id", "selection_id", "market_name"):
            df[c] = df[c].astype(object)
    if frac < 1:
        ev = w["event_id"].drop_duplicates()
        keep = set(ev.sample(frac=frac, random_state=seed))
        w = w[w["event_id"].isin(keep)]
        p = p[p["event_id"].isin(keep)]

    # WIN side: exactly one winner, every runner priced (> 1)
    g = w.groupby("market_id")
    one = g["won"].transform("sum") == 1
    priced = g["ltp_pre"].transform(lambda s: s.notna().all() and (s > 1).all())
    w = w[one & priced].copy()
    w["p_win"] = (1 / w["ltp_pre"]) / w.groupby("market_id")["ltp_pre"].transform(lambda s: (1 / s).sum())
    w["harness"] = w["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b")
    w = w.rename(columns={"market_id": "win_mid", "ltp_pre": "ltp_win", "bsp": "bsp_win", "won": "won_win",
                          "ltp_t30": "t30_win"}).drop(columns=["market_name", "n_winners", "field_size"])

    # PLACE side: settled with at least n_winners placed (more = dead heat)
    p = p[p["won"].notna() & p["n_winners"].isin([2, 3])].copy()
    gp = p.groupby("market_id")
    p["n_placed"] = gp["won"].transform("sum")
    p = p[p["n_placed"] >= p["n_winners"]]
    p = p.rename(columns={"market_id": "pl_mid", "ltp_pre": "ltp_pl", "bsp": "bsp_pl", "won": "placed",
                          "ltp_t30": "t30_pl"}).drop(columns=["market_name", "field_size", "date"])

    # one WIN and one PLACE market per (event_id, start)
    wk = w[["event_id", "start", "win_mid"]].drop_duplicates()
    pk = p[["event_id", "start", "pl_mid"]].drop_duplicates()
    wk = wk[~wk.duplicated(["event_id", "start"], keep=False)]
    pk = pk[~pk.duplicated(["event_id", "start"], keep=False)]
    pairs = wk.merge(pk, on=["event_id", "start"])
    w = w[w["win_mid"].isin(pairs["win_mid"])]
    p = p[p["pl_mid"].isin(pairs["pl_mid"])]
    j = w.merge(p.drop(columns=["event_id"]), on=["start", "selection_id"], how="outer", indicator=True)
    j = j.merge(pairs[["win_mid", "pl_mid"]].rename(columns={"win_mid": "_wm"}), on="pl_mid", how="left")
    # require identical runner sets in the two markets
    bad = set(j.loc[j["_merge"] != "both", "win_mid"].dropna()) | set(j.loc[j["_merge"] != "both", "_wm"].dropna())
    j = j[(j["_merge"] == "both") & ~j["win_mid"].isin(bad)].drop(columns=["_merge", "_wm"])
    j["harness"] = j["harness"].astype(bool)  # the outer merge can leave this as object
    j["n"] = j.groupby("win_mid")["selection_id"].transform("size")
    j = j[j["n"] >= j["n_winners"] + 2]  # a bet needs someone who can miss a place
    j["month"] = j["date"].dt.to_period("M").astype(str)
    j["race"] = pd.factorize(j["win_mid"])[0]
    j = j.sort_values(["race", "ltp_win", "selection_id"]).reset_index(drop=True)
    j["pos"] = j.groupby("race").cumcount()
    return j


# ---------------------------------------------------------------------------
# discounted-Harville lambda fit from (winner, placed set)
# ---------------------------------------------------------------------------
def pad(j: pd.DataFrame, col: str, fill=0.0) -> np.ndarray:
    R, W = j["race"].max() + 1, int(j["pos"].max()) + 1
    M = np.full((R, W), fill)
    M[j["race"].to_numpy(), j["pos"].to_numpy()] = j[col].to_numpy(dtype=float)
    return M


def placed_set_stats(j: pd.DataFrame):
    """Per race: P (padded win probs), index of the winner, indices of the one
    or two other placed runners (-1 when none). Races usable for the fit have
    the winner placed and exactly n_winners placed runners."""
    P = pad(j, "p_win")
    won = pad(j, "won_win").astype(bool)
    plc = pad(j, "placed").astype(bool)
    R = P.shape[0]
    race = j.drop_duplicates("race").set_index("race").sort_index()
    nw = race["n_winners"].to_numpy()
    n_placed = race["n_placed"].to_numpy()
    jw = won.argmax(1)
    others = plc & ~won
    cnt = others.sum(1)
    k = np.where(cnt >= 1, others.argmax(1), -1)
    o2 = others.copy()
    o2[np.arange(R), np.clip(k, 0, None)] = False
    i = np.where(cnt >= 2, o2.argmax(1), -1)
    ok = won[np.arange(R), jw] & plc[np.arange(R), jw] & (n_placed == nw) & (cnt == nw - 1)
    return P, jw, k, i, ok, nw, race


def grid_loglik(P, jw, k, i, ok, nw):
    """Per-race log-likelihood of the observed placed set for every (lam2, lam3)
    on LAMBDA_GRID. Shape (G, G, R); 2-place races vary along lam2 only."""
    R = P.shape[0]
    G = len(LAMBDA_GRID)
    ar = np.arange(R)
    kk, ii = np.clip(k, 0, None), np.clip(i, 0, None)
    A = np.empty((G, R)); aj = np.empty((G, R)); ak = np.empty((G, R)); ai = np.empty((G, R))
    for g, lam in enumerate(LAMBDA_GRID):
        a = P ** lam
        A[g] = a.sum(1); aj[g] = a[ar, jw]; ak[g] = a[ar, kk]; ai[g] = a[ar, ii]
    sec_k = ak / np.maximum(A - aj, 1e-300)        # P(k 2nd | j 1st)
    sec_i = ai / np.maximum(A - aj, 1e-300)
    ll = np.zeros((G, G, R))
    two = ok & (nw == 2)
    three = ok & (nw == 3)
    ll[:, :, two] = np.log(np.maximum(sec_k[:, two], 1e-300))[:, None, :]
    # lam3 terms reuse A/aj/ak/ai (same grid), so index them by the lam3 grid position
    thr_i = ai / np.maximum(A - aj - ak, 1e-300)   # P(i 3rd | j, k)
    thr_k = ak / np.maximum(A - aj - ai, 1e-300)   # P(k 3rd | j, i)
    for g3 in range(G):
        L = sec_k[:, three] * thr_i[g3, three][None, :] + sec_i[:, three] * thr_k[g3, three][None, :]
        ll[:, g3, three] = np.log(np.maximum(L, 1e-300))
    return ll


def fit_lambdas_walk_forward(ll, ok, months, codes, months_sorted):
    """For each (month, code): argmax over the grid of the summed log-lik of
    strictly earlier months of the same code (pooled codes if too few races)."""
    G = ll.shape[0]
    keys = pd.Series(list(zip(months, codes)))
    groups = {}
    for key, idx in keys[ok].groupby(keys[ok]).indices.items():
        sel = np.flatnonzero(ok)[idx]
        groups[key] = (ll[:, :, sel].sum(-1), len(sel))
    lam = {}
    for m in months_sorted:
        for c in (False, True):
            tot = np.zeros((G, G)); n = 0
            for (mm, cc), (s, cnt) in groups.items():
                if mm < m and cc == c:
                    tot += s; n += cnt
            if n < MIN_FIT_RACES:  # pool both codes
                tot = np.zeros((G, G)); n = 0
                for (mm, cc), (s, cnt) in groups.items():
                    if mm < m:
                        tot += s; n += cnt
            if n < MIN_FIT_RACES:
                continue
            g2, g3 = np.unravel_index(np.argmax(tot), tot.shape)
            lam[(m, c)] = (float(LAMBDA_GRID[g2]), float(LAMBDA_GRID[g3]), n)
    return lam


def derived_place_probs(j: pd.DataFrame, P, nw, lam, months, codes) -> np.ndarray:
    R, W = P.shape
    Q = np.full((R, W), np.nan)
    for r in range(R):
        key = (months[r], codes[r])
        if key not in lam:
            continue
        l2, l3, _ = lam[key]
        n = int((P[r] > 0).sum())
        o = order_probs(P[r, :n], l2, l3)
        Q[r, :n] = o["top3"] if nw[r] == 3 else o["top2"]
    return Q[j["race"].to_numpy(), j["pos"].to_numpy()]


def calibrate_walk_forward(j: pd.DataFrame, col: str, months_sorted: list[str], window=12) -> np.ndarray:
    """Logistic recalibration logit(P) = cubic polynomial in logit(p_fair)
    (standardised), refitted monthly on the previous ``window`` months (Newton),
    then rescaled within the race to sum to n_winners."""
    z = np.log(j[col].clip(1e-6, 1 - 1e-6) / (1 - j[col].clip(1e-6, 1 - 1e-6))).to_numpy()
    y = j["placed"].to_numpy(dtype=float)
    month = j["month"].to_numpy()
    outp = np.full(len(j), np.nan)
    params = {}
    design = lambda zz: np.column_stack([np.ones(len(zz)), zz, zz ** 2, zz ** 3])
    for idx, m in enumerate(months_sorted):
        tr = np.isin(month, months_sorted[max(0, idx - window):idx])
        if tr.sum() < 5000:
            continue
        mu_z, sd_z = z[tr].mean(), z[tr].std()
        X = design((z[tr] - mu_z) / sd_z)
        w = np.zeros(4)
        for _ in range(12):
            mu = 1 / (1 + np.exp(-(X @ w)))
            g = X.T @ (mu - y[tr])
            H = (X * (mu * (1 - mu))[:, None]).T @ X + 1e-6 * np.eye(4)
            w = w - np.linalg.solve(H, g)
        params[m] = [round(float(v), 3) for v in w]
        te = month == m
        outp[te] = 1 / (1 + np.exp(-(design((z[te] - mu_z) / sd_z) @ w)))
    p = pd.Series(outp, index=j.index)
    scale = j["n_winners"] / p.groupby(j["race"]).transform("sum")
    return (p * scale).clip(upper=0.999).to_numpy(), params


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------
def summ(ret: np.ndarray) -> dict:
    ret = ret[~np.isnan(ret)]
    if len(ret) < 2:
        return {"bets": int(len(ret)), "roi": None, "se": None}
    return {"bets": int(len(ret)), "roi": r4(ret.mean()), "se": r4(ret.std(ddof=1) / math.sqrt(len(ret)))}


def returns(price, won, dh, c):
    """Back 1 unit; dead heat: stake share dh is settled as a winner, 1-dh lost."""
    price = np.asarray(price, dtype=float); won = np.asarray(won, dtype=bool); dh = np.asarray(dh, dtype=float)
    return np.where(won, dh * (price - 1) * (1 - c) - (1 - dh), -1.0)


def bet_cell(g: pd.DataFrame, mid, price_col="ltp_pl", won_col="placed", prob_col=None) -> dict:
    res = {"bets": int(len(g))}
    if len(g) < 2:
        return res
    for c in COMMS:
        r = returns(g[price_col], g[won_col], g["dh"], c)
        s = summ(r)
        res[f"roi_c{int(c*100)}"] = s["roi"]
        if c == 0.05:
            res["se"] = s["se"]
            res["halves_roi_c5"] = [summ(r[(g["date"] < mid).to_numpy()])["roi"], summ(r[(g["date"] >= mid).to_numpy()])["roi"]]
            res["z_c5"] = r4(s["roi"] / s["se"]) if s["se"] else None
    res["roi_c5_one_tick_worse"] = summ(returns(one_tick_worse(g[price_col]), g[won_col], g["dh"], 0.05))["roi"]
    bsp_col = "bsp_pl" if price_col == "ltp_pl" else "bsp_win"
    has = g[bsp_col].notna() & (g[bsp_col] > 1)
    res["bsp_cover"] = r4(has.mean())
    res["roi_c5_settle_bsp"] = summ(returns(g.loc[has, bsp_col], g.loc[has, won_col], g.loc[has, "dh"], 0.05))["roi"]
    res["roi_c5_ltp_same_bsp_subset"] = summ(returns(g.loc[has, price_col], g.loc[has, won_col], g.loc[has, "dh"], 0.05))["roi"]
    if has.sum() > 1:
        rb = returns(g.loc[has, bsp_col], g.loc[has, won_col], g.loc[has, "dh"], 0.05)
        res["bsp_settle_se"] = summ(rb)["se"]
        res["bsp_settle_halves_c5"] = [summ(rb[(g.loc[has, "date"] < mid).to_numpy()])["roi"], summ(rb[(g.loc[has, "date"] >= mid).to_numpy()])["roi"]]
        worse = np.minimum(g.loc[has, price_col], g.loc[has, bsp_col])
        res["roi_c5_worse_of_ltp_and_bsp"] = summ(returns(worse, g.loc[has, won_col], g.loc[has, "dh"], 0.05))["roi"]
        if prob_col:
            res["median_prob_x_bsp"] = r4((g.loc[has, prob_col] * g.loc[has, bsp_col]).median())
            res["median_prob_x_ltp"] = r4((g.loc[has, prob_col] * g.loc[has, price_col]).median())
        hw = has & g["bsp_win"].notna() & (g["bsp_win"] > 1)
        res["median_bsp_win_over_ltp_win"] = r4((g.loc[hw, "bsp_win"] / g.loc[hw, "ltp_win"]).median()) if hw.any() else None
    res["median_bsp_over_ltp"] = r4((g.loc[has, bsp_col] / g.loc[has, price_col]).median()) if has.any() else None
    res["strike"] = r4(g[won_col].mean())
    res["mean_price"] = r4(g[price_col].mean())
    return res


def positive(cell: dict) -> bool:
    h = cell.get("halves_roi_c5")
    return bool(cell.get("se") and cell["roi_c5"] > 2 * cell["se"] and h and all(x is not None and x > 0 for x in h)
                and cell.get("bets", 0) >= 200)


def bh(pvals: list[float], q=0.05) -> list[bool]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    k = np.max(np.flatnonzero(passed)) if passed.any() else -1
    res = np.zeros(n, dtype=bool)
    if k >= 0:
        res[order[:k + 1]] = True
    return res.tolist()


def z_to_p(z) -> float:
    return float(0.5 * math.erfc(z / math.sqrt(2))) if z is not None else 1.0


def logloss_race(q, y, race):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    ll = -(y * np.log(q) + (1 - y) * np.log(1 - q))
    return np.bincount(race, weights=ll)


def paired(a: np.ndarray, b: np.ndarray, first_half=None) -> dict:
    """mean per-race log-loss of a and b, and gain (b - a) with SE over races
    (and per half when ``first_half`` is given)."""
    d = b - a
    res = {"ll_model": r4(a.mean()), "ll_market": r4(b.mean()), "gain": r4(d.mean()),
           "gain_se": r4(d.std(ddof=1) / math.sqrt(len(d))), "races": int(len(d))}
    if first_half is not None:
        res["halves_gain"] = [r4(d[first_half].mean()), r4(d[~first_half].mean())]
        res["halves_gain_se"] = [r4(d[first_half].std(ddof=1) / math.sqrt(first_half.sum())),
                                 r4(d[~first_half].std(ddof=1) / math.sqrt((~first_half).sum()))]
    return res


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/raw/betfair/split"))
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of events (development)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    j = load(a.dir, a.frac, a.seed)
    P, jw, k, i, ok, nw, race = placed_set_stats(j)
    months = race["month"].to_numpy()
    codes = race["harness"].to_numpy()
    months_sorted = sorted(set(months))
    ll = grid_loglik(P, jw, k, i, ok, nw)
    lam = fit_lambdas_walk_forward(ll, ok, months, codes, months_sorted)
    del ll
    j["p_fair"] = derived_place_probs(j, P, nw, lam, months, codes)
    j = j[j["p_fair"].notna()].copy()
    j["race"] = pd.factorize(j["win_mid"])[0]
    j["dh"] = j["n_winners"] / j["n_placed"]
    j["fair_price"] = 1 / j["p_fair"]
    j["p_cal"], cal_params = calibrate_walk_forward(j, "p_fair", sorted(j["month"].unique()))
    days = sorted(j["date"].unique())
    mid = days[len(days) // 2]
    last = {}
    for (m, c), v in lam.items():
        last[("harness" if c else "gallops")] = {"month": m, "lam2": v[0], "lam3": v[1], "fit_races": v[2]}
    lam_hist = sorted({(m, "harness" if c else "gallops"): (v[0], v[1]) for (m, c), v in lam.items()}.items())
    out({"section": "data", "runtime_s": round(time.time() - t0, 1),
         "joined_races": int(j["race"].nunique()), "runners": int(len(j)),
         "fit_usable_races_all_months": int(ok.sum()), "months_with_lambdas": len({m for m, _ in lam}),
         "from": str(pd.Timestamp(j["date"].min()).date()), "to": str(pd.Timestamp(j["date"].max()).date()),
         "mid": str(pd.Timestamp(mid).date()),
         "harness_share_races": r4(j.drop_duplicates("race")["harness"].mean()),
         "n_winners_races": j.drop_duplicates("race")["n_winners"].value_counts().to_dict(),
         "place_ltp_coverage_runners": r4(j["ltp_pl"].notna().mean()),
         "dead_heat_markets": int((j.drop_duplicates("race")["n_placed"] > j.drop_duplicates("race")["n_winners"]).sum()),
         "last_lambdas": last,
         "calibration_params_sample": {m: cal_params[m] for m in sorted(cal_params)[::6]},
         "p_cal_coverage": r4(j["p_cal"].notna().mean()),
         "lambda_path_every_6_months": [{"month": m, "code": c, "lam2": v[0], "lam3": v[1]}
                                        for (m, c), v in lam_hist if int(m[-2:]) in (1, 7)]})

    # ---- place market: normalised implied probabilities (fully priced markets only)
    grp = j.groupby("race")
    j["all_priced"] = grp["ltp_pl"].transform(lambda s: s.notna().all() and (s > 1).all())
    full = j[j["all_priced"]].copy()
    full["q_mkt"] = (1 / full["ltp_pl"]) * full["n_winners"] / full.groupby("race")["ltp_pl"].transform(lambda s: (1 / s).sum())
    full["book_over_nw"] = full.groupby("race")["ltp_pl"].transform(lambda s: (1 / s).sum()) / full["n_winners"]
    j["q_mkt"] = full["q_mkt"]

    # ---- (a) calibration of the place market and of the derived prices
    labels = band_labels(PLACE_BANDS)
    j["band_pl"] = pd.cut(j["ltp_pl"], PLACE_BANDS, right=False, labels=labels)
    j["band_fair"] = pd.cut(j["fair_price"], PLACE_BANDS, right=False, labels=labels)
    rows = []
    for b, g in j[j["ltp_pl"].notna()].groupby("band_pl", observed=True):
        cell = bet_cell(g, mid)
        rows.append({"place_ltp_band": str(b), "runners": int(len(g)), "strike": r4(g["placed"].mean()),
                     "implied_raw_1/ltp": r4((1 / g["ltp_pl"]).mean()),
                     "implied_normalised_fully_priced": r4(g["q_mkt"].mean()) if g["q_mkt"].notna().any() else None,
                     "derived_p_fair": r4(g["p_fair"].mean()), "derived_p_cal": r4(g["p_cal"].mean()),
                     "blind_back_roi_c0": cell.get("roi_c0"), "blind_back_roi_c5": cell.get("roi_c5"),
                     "blind_back_roi_c8": cell.get("roi_c8"), "se": cell.get("se"), "halves_c5": cell.get("halves_roi_c5")})
    out({"section": "a_calibration_place_market_by_place_ltp_band", "rows": rows,
         "fully_priced_place_markets": int(full["race"].nunique()),
         "book_over_n_winners_median": r4(full.drop_duplicates("race")["book_over_nw"].median())})
    rows = []
    for b, g in j.groupby("band_fair", observed=True):
        rows.append({"fair_price_band": str(b), "runners": int(len(g)), "strike": r4(g["placed"].mean()),
                     "derived_p_fair": r4(g["p_fair"].mean()), "strike_over_fair": r4(g["placed"].mean() / g["p_fair"].mean()),
                     "derived_p_cal": r4(g["p_cal"].mean()),
                     "place_ltp_coverage": r4(g["ltp_pl"].notna().mean()),
                     "mean_place_ltp_when_priced": r4(g["ltp_pl"].mean()),
                     "implied_raw_when_priced": r4((1 / g["ltp_pl"]).mean())})
    out({"section": "a_calibration_derived_by_fair_price_band", "rows": rows})
    # the two sides of the same race: where do they disagree?
    full["ratio"] = full["ltp_pl"] * full["p_fair"]
    rb = pd.cut(full["ratio"], [0, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 10])
    rows = []
    for b, g in full.groupby(rb, observed=True):
        rows.append({"ltp_pl_x_p_fair": str(b), "runners": int(len(g)), "strike": r4(g["placed"].mean()),
                     "q_mkt": r4(g["q_mkt"].mean()), "p_fair": r4(g["p_fair"].mean()),
                     "mean_ltp_pl": r4(g["ltp_pl"].mean())})
    out({"section": "a_disagreement_fully_priced", "rows": rows})

    # ---- (b) log loss on fully priced place markets
    y = full["placed"].to_numpy(dtype=float)
    rc = pd.factorize(full["race"])[0]
    ll_mkt = logloss_race(full["q_mkt"].to_numpy(), y, rc)
    ll_fair = logloss_race(full["p_fair"].to_numpy(), y, rc)
    hc = full["p_cal"].notna().to_numpy()
    ll_cal = logloss_race(np.nan_to_num(full["p_cal"].to_numpy(), nan=0.5), y, rc)
    rc_has = np.bincount(rc, weights=hc) == np.bincount(rc)
    fh = (full.drop_duplicates("race")["date"] < mid).to_numpy()
    res = {"section": "b_logloss_fully_priced_place_markets", "races": int(len(ll_mkt)),
           "per_race_logloss": {"market": r4(ll_mkt.mean()), "derived": r4(ll_fair.mean())},
           "derived_vs_market": paired(ll_fair, ll_mkt, fh),
           "derived_calibrated_vs_market": paired(ll_cal[rc_has], ll_mkt[rc_has], fh[rc_has])}
    cal_blend = {}
    for w in BLEND_GRID:
        q = w * full["q_mkt"].to_numpy() + (1 - w) * np.nan_to_num(full["p_cal"].to_numpy(), nan=0.5)
        cal_blend[round(float(w), 2)] = logloss_race(q, y, rc)
    res["blend_calibrated_w_market_0.5"] = paired(cal_blend[0.5][rc_has], ll_mkt[rc_has], fh[rc_has])
    fm = full.drop_duplicates("race").set_index("race")["month"]
    rmonth = fm.reindex(pd.unique(full["race"])).to_numpy()
    # fixed blends and a walk-forward fitted weight (w on the market)
    blend_ll = {}
    for w in BLEND_GRID:
        q = w * full["q_mkt"].to_numpy() + (1 - w) * full["p_fair"].to_numpy()
        blend_ll[round(float(w), 2)] = logloss_race(q, y, rc)
    for w in (0.25, 0.5, 0.75):
        res[f"blend_w_market_{w}"] = paired(blend_ll[w], ll_mkt, fh)
    msum = {w: pd.Series(v).groupby(rmonth).sum() for w, v in blend_ll.items()}
    mcount = pd.Series(rmonth).value_counts()
    chosen, wf = {}, np.full(len(ll_mkt), np.nan)
    for m in sorted(set(rmonth)):
        prev = [mm for mm in mcount.index if mm < m]
        if sum(mcount[mm] for mm in prev) < 500:
            continue
        best = min(BLEND_GRID, key=lambda w: sum(msum[round(float(w), 2)].get(mm, 0) for mm in prev))
        chosen[m] = round(float(best), 2)
        sel = rmonth == m
        wf[sel] = blend_ll[round(float(best), 2)][sel]
    has = ~np.isnan(wf)
    res["blend_walk_forward"] = {**paired(wf[has], ll_mkt[has], fh[has]), "weights_on_market_by_month_sample":
                                 {m: chosen[m] for m in sorted(chosen)[::6]}}
    res["blend_walk_forward"]["median_weight"] = float(np.median(list(chosen.values()))) if chosen else None
    # same walk-forward blend on the calibrated derived probabilities
    msum_c = {w: pd.Series(v[rc_has]).groupby(rmonth[rc_has]).sum() for w, v in cal_blend.items()}
    chosen_c, wfc = {}, np.full(len(ll_mkt), np.nan)
    for m in sorted(set(rmonth)):
        prev = [mm for mm in mcount.index if mm < m]
        if sum(mcount[mm] for mm in prev) < 500:
            continue
        best = min(BLEND_GRID, key=lambda w: sum(msum_c[round(float(w), 2)].get(mm, 0) for mm in prev))
        chosen_c[m] = round(float(best), 2)
        sel = (rmonth == m) & rc_has
        wfc[sel] = cal_blend[round(float(best), 2)][sel]
    hasc = ~np.isnan(wfc)
    res["blend_calibrated_walk_forward"] = {**paired(wfc[hasc], ll_mkt[hasc], fh[hasc]),
                                            "median_weight_on_market": float(np.median(list(chosen_c.values()))) if chosen_c else None}
    out(res)

    # ---- (c) bets: back the place at ltp_pl when ltp_pl * p > 1 + k, p = p_fair or p_cal
    bettable = j[j["ltp_pl"].notna() & (j["ltp_pl"] > 1)].copy()
    segments = {"all": np.ones(len(bettable), dtype=bool),
                "places_2": (bettable["n_winners"] == 2).to_numpy(), "places_3": (bettable["n_winners"] == 3).to_numpy(),
                "gallops": (~bettable["harness"]).to_numpy(), "harness": bettable["harness"].to_numpy(),
                "fully_priced_only": bettable["all_priced"].to_numpy(),
                "partially_priced_only": (~bettable["all_priced"]).to_numpy(),
                "place_t30_present_liquid": bettable["t30_pl"].notna().to_numpy(),
                "price_1.5_to_15_post_hoc": ((bettable["ltp_pl"] >= 1.5) & (bettable["ltp_pl"] < 15)).to_numpy()}
    ref = bettable[bettable["bsp_pl"].notna() & (bettable["bsp_pl"] > 1)]
    out({"section": "c_reference_all_priced_runners", "runners": int(len(bettable)), "with_place_bsp": int(len(ref)),
         "median_bsp_over_ltp_all": r4((ref["bsp_pl"] / ref["ltp_pl"]).median()),
         "share_bsp_below_ltp_all": r4((ref["bsp_pl"] < ref["ltp_pl"]).mean()),
         "median_bsp_win_over_ltp_win_all": r4((ref["bsp_win"] / ref["ltp_win"]).median()),
         "median_p_cal_x_bsp_all": r4((ref["p_cal"] * ref["bsp_pl"]).median()),
         "place_t30_present_share": r4(bettable["t30_pl"].notna().mean()),
         "blind_back_all": bet_cell(bettable, mid)})
    for b in labels:
        segments[f"band_{b}"] = (bettable["band_pl"].astype(str) == b).to_numpy()
    cells, checks = [], {}
    for pcol in ("p_fair", "p_cal"):
        edge = (bettable["ltp_pl"] * bettable[pcol] - 1).to_numpy()
        for kk in KS:
            sel_k = edge > kk
            for name, mask in segments.items():
                g = bettable[sel_k & mask]
                cell = bet_cell(g, mid, prob_col=pcol)
                cell.update({"rule": f"ltp_pl*{pcol}>1+{kk:g}", "prob": pcol, "k": kk, "segment": name, "positive": positive(cell)})
                cells.append(cell)
                # opposite-half development check: > 2 SE (c=5%) in the dev half, does the other half hold?
                for dev_first in (True, False):
                    d = g[g["date"] < mid] if dev_first else g[g["date"] >= mid]
                    h = g[g["date"] >= mid] if dev_first else g[g["date"] < mid]
                    sd = summ(returns(d["ltp_pl"], d["placed"], d["dh"], 0.05))
                    if sd["bets"] >= 100 and sd["roi"] is not None and sd["roi"] > 2 * sd["se"]:
                        sh = summ(returns(h["ltp_pl"], h["placed"], h["dh"], 0.05))
                        checks.setdefault("dev_first_half" if dev_first else "dev_second_half", []).append(
                            {"rule": cell["rule"], "segment": name, "dev_bets": sd["bets"], "dev_roi_c5": sd["roi"],
                             "holdout_bets": sh["bets"], "holdout_roi_c5": sh["roi"], "holdout_se": sh["se"],
                             "holds": bool(sh["roi"] is not None and sh["se"] and sh["roi"] > 2 * sh["se"])})
    n_cells = len(cells)
    pv = [z_to_p(c.get("z_c5")) if c.get("z_c5") is not None else 1.0 for c in cells]
    for c, f in zip(cells, bh(pv)):
        c["bh_pass_q05"] = f
    out({"section": "c_place_bets", "cells_tested": n_cells, "cells": cells,
         "positive_cells": [(c["rule"], c["segment"]) for c in cells if c["positive"]],
         "bh_pass_cells": [(c["rule"], c["segment"]) for c in cells if c["bh_pass_q05"]]})
    out({"section": "c_opposite_half_check",
         **{k: {"n_dev_positive": len(v), "n_hold": sum(x["holds"] for x in v), "detail": v} for k, v in checks.items()}})
    # reference: the market SHORTER than fair by 10% (a lay would be needed) backed anyway
    neg = bettable[bettable["ltp_pl"] * bettable["p_fair"] - 1 < -0.1]
    out({"section": "c_reference_market_shorter_than_fair_by_10pct_back_anyway", **bet_cell(neg, mid)})

    # ---- (d) mirror: does the PLACE market predict WINNERS beyond the WIN price?
    full["x"] = np.log(full["q_mkt"]) - np.log(full["p_fair"])
    full["win_edge_x"] = full["x"]
    xb = pd.cut(full["x"], [-10, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, 10])
    rows = []
    for b, g in full.groupby(xb, observed=True):
        cell = bet_cell(g, mid, price_col="ltp_win", won_col="won_win")
        rows.append({"log_q_mkt_over_p_fair": str(b), "runners": int(len(g)), "win_strike": r4(g["won_win"].mean()),
                     "win_implied": r4(g["p_win"].mean()), "strike_over_implied": r4(g["won_win"].mean() / g["p_win"].mean()),
                     "back_win_roi_c0": cell.get("roi_c0"), "roi_c5": cell.get("roi_c5"), "roi_c8": cell.get("roi_c8"),
                     "se": cell.get("se"), "halves_c5": cell.get("halves_roi_c5")})
    out({"section": "d_mirror_win_calibration_by_place_disagreement", "rows": rows})
    mcells = []
    for t in (0.05, 0.1, 0.2, 0.3):
        g = full[full["x"] > t]
        cell = bet_cell(g, mid, price_col="ltp_win", won_col="won_win")
        cell.update({"rule": f"back WIN when log(q_mkt/p_fair)>{t:g}", "segment": "fully_priced", "positive": positive(cell)})
        mcells.append(cell)
    # walk-forward conditional logit: log p_win (unpenalised) + x, refit monthly on the previous 12 months
    full["race_id"] = full["race"]
    full["won"] = full["won_win"]
    full["lp"] = np.log(full["p_win"])
    full["x_cal"] = np.log(full["q_mkt"]) - np.log(full["p_cal"].fillna(full["p_fair"]))
    full["p_model"] = np.nan
    full["p_model_cal"] = np.nan
    fmonths = sorted(full["month"].unique())
    weights, weights_c = [], []
    for idx_m, m in enumerate(fmonths):
        train_months = [mm for mm in fmonths[max(0, idx_m - 12):idx_m]]
        tr = full[full["month"].isin(train_months)]
        if tr["race"].nunique() < 500:
            continue
        te = full["month"] == m
        model = ConditionalLogit(["lp", "x"], l2=1e-4, unpenalized=("lp",)).fit(tr)
        full.loc[te, "p_model"] = model.predict_proba(full[te])
        weights.append((m, [r4(w) for w in model.weights]))
        model_c = ConditionalLogit(["lp", "x_cal"], l2=1e-4, unpenalized=("lp",)).fit(tr)
        full.loc[te, "p_model_cal"] = model_c.predict_proba(full[te])
        weights_c.append((m, [r4(w) for w in model_c.weights]))
    ev = full[full["p_model"].notna()].copy()
    rc2 = pd.factorize(ev["race"])[0]
    yw = ev["won_win"].to_numpy(dtype=float)
    llw_mkt = np.bincount(rc2, weights=-yw * np.log(np.clip(ev["p_win"].to_numpy(), 1e-9, 1)))
    llw_mod = np.bincount(rc2, weights=-yw * np.log(np.clip(ev["p_model"].to_numpy(), 1e-9, 1)))
    llw_modc = np.bincount(rc2, weights=-yw * np.log(np.clip(ev["p_model_cal"].to_numpy(), 1e-9, 1)))
    rd = ev.drop_duplicates("race")["date"].to_numpy()
    mres = {"section": "d_mirror_win_model", "races": int(len(llw_mkt)),
            "model_vs_market": paired(llw_mod, llw_mkt, rd < mid),
            "weights_sample_std_units": weights[::6], "n_refits": len(weights),
            "model_xcal_vs_market": paired(llw_modc, llw_mkt, rd < mid),
            "weights_xcal_sample_std_units": weights_c[::6]}
    ev["edge_m"] = ev["p_model"] * ev["ltp_win"] - 1
    for kk in KS:
        g = ev[ev["edge_m"] > kk]
        cell = bet_cell(g, mid, price_col="ltp_win", won_col="won_win")
        cell.update({"rule": f"back WIN when p_model*ltp_win>1+{kk:g}", "segment": "fully_priced", "positive": positive(cell)})
        mcells.append(cell)
    out(mres)
    pv2 = [z_to_p(c.get("z_c5")) if c.get("z_c5") is not None else 1.0 for c in mcells]
    for c, f in zip(mcells, bh(pv + pv2)[n_cells:]):
        c["bh_pass_q05_joint"] = f
    out({"section": "d_mirror_win_bets", "cells_tested": len(mcells), "cells": mcells,
         "positive_cells": [c["rule"] for c in mcells if c["positive"]]})
    out({"section": "totals", "bet_cells_tested_total": n_cells + len(mcells),
         "probability_tests": 9, "runtime_s": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

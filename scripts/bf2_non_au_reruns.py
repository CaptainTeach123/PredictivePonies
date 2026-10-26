"""Experiment "non_au_reruns" (round 2): the round-1 ideas rerun on the GB and IE WIN
markets (fresh prices, ~99.8% t30 coverage) and on NZ WIN markets as a third market.

    python scripts/bf2_non_au_reruns.py [--dir data/raw/betfair/split] [--frac 0.2] [--json out.json]

Data: data/raw/betfair/split/{gb,ie,nz}_win.parquet (licensed, local only). Prints JSON
summary lines only; never horse names or rows. The horse key for part 4 is selection_id
(persistent per horse in GB/IE: < 0.2% of ids carry two names), pooled over GB + IE
because horses run in both.

Common preparation (per country): drop_duplicates(market_id, selection_id); prices <= 1 ->
NaN; settled markets only (won known on every row, >= 1 winner, >= 3 runners); local race
day / hour / weekday from the venue country's timezone (Europe/London, Europe/Dublin,
Pacific/Auckland); f = dead_heat_share of a winner (0 for losers).
  G6  drop markets with a gap in the saddlecloth sequence AND an ltp_pre book < 0.97.
  G1  every decision uses ltp_pre, so the evaluation universe is markets whose ltp_pre book
      (sum 1/ltp_pre, every runner priced) is in [0.97, 1.03] ("coherent"). Nothing uses BSP
      or a same-day result (horse history: runs with an earlier start only, asserted).
  G2  every rule is decided before the off and settled at Betfair SP (only markets where
      BSP was offered: NZ offers it on ~52% of markets, all or none). Lays are reported per
      unit of liability (primary) and per unit of backer's stake.
  G3  commission on positive net winnings per bet: 8% primary, 0 / 5 / 10% reported.
  G4  ROI at 8% > 2 SE in both halves (split at the median race day of the rule's
      evaluation universe) and > 0 over 2019-11-08..2020-11-08. SE = max(iid, clustered by
      market, clustered by local race day).
  G5  every cell that yields a z statistic is registered and counted (g5_count line);
      BH (q = 0.05, one-sided for a positive ROI / gain) is reported inside this experiment.
  G7  move deciles, book quartiles and cum_resid deciles use cutoffs from earlier months only
      (expanding window, refreshed monthly); the logits are walk-forward (train on months
      strictly before the predicted month, refit monthly, 9 burn-in months) with L2 chosen
      every 3 months on earlier data only (inner train = all but the last 6 training months,
      validation = those 6 months) from 0.001..10 (half-decade steps), extended by a decade
      (up to 4 times) whenever the optimum sits on a grid edge. The top-15 venue list is
      outcome-free but uses whole-period market counts (flagged).

Parts:
  1 late money: move = log(ltp_t30 / ltp_pre) (> 0 = firmed after t30) deciles (walk-forward
    cutoffs); calibration at ltp_pre (Shin) and at BSP (1/bsp normalised) by decile; ROI at
    BSP (back; lay per liability / per stake). Runners with an ltp_t30 in coherent BSP markets.
  2 price-only residual ConditionalLogit (as scripts/bf_microstructure_logit.py, Newton fit
    of the repo's objective, checked against ponies.model.ConditionalLogit): log Shin p at
    ltp_pre (unpenalised), its square, rank 1-3 dummies, has_t30, move, move*|move|, ladder-
    change print, and log_p interactions with log field, book, local hour, Saturday, has_t30
    (and harness for NZ). Log-loss gain vs the ltp_pre Shin market at the same timestamp;
    bets when the model's expected return at ltp_pre after 8% exceeds k (0, 0.02, 0.05),
    back, or lay per liability, settled at BSP.
  3 one-way segment scan: favourite (rank 1 at ltp_pre) and 9 ltp_pre bands x {all, field
    size (5), venue top 15 + other, local hour (5), weekday (7), year (4), ltp_pre book
    quartile (4, walk-forward cutoffs)} x {back, lay/liability, lay/stake} at BSP.
  4 horse market memory (GB, IE): days since last run, back-up within 7 days, previous
    win / surprise (f - p), beaten favourite, previous steam (log ltp/bsp) at t30 and pre,
    previous late move, shrunk cumulative residual sum(f - p)/(n + 5), price vs previous,
    field change. Walk-forward residual logit as part 2 plus simple rules (back-up <= 7d,
    cum_resid top decile back / bottom decile lay, beaten favourite back, last-start winner
    lay) over the same predicted months.
  5 BSP vs ltp_pre by ltp_pre band: paired per-runner difference (back at BSP minus back at
    ltp_pre, and lay per liability), with the tick-position control: all prints, prints
    exactly on a ladder-change price (2, 3, 4, 6, 10, 20, 30, 50, 100) excluded, and only
    those prints; back / lay ROI at BSP for the excl / only variants (the "all" variant is
    the part-3 "all" cell and is not counted twice).
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ponies.betfair import net_return  # noqa: E402
from ponies.model import ConditionalLogit, race_softmax  # noqa: E402
from ponies.odds import shin_probabilities  # noqa: E402

COLS = ["market_id", "venue", "market_name", "start", "field_size", "selection_id", "program", "ltp_pre",
        "ltp_t30", "bsp", "won", "dead_heat_share"]
COUNTRIES = ("gb", "ie", "nz")
TZ = {"gb": "Europe/London", "ie": "Europe/Dublin", "nz": "Pacific/Auckland"}
COMMS = (0.0, 0.05, 0.08, 0.10)
PC = 0.08
LAST12 = np.datetime64("2019-11-08")
Z80 = 0.8416
BANDS = [(1.01, 2.0), (2.0, 3.0), (3.0, 4.5), (4.5, 6.5), (6.5, 10.0), (10.0, 15.0), (15.0, 25.0), (25.0, 50.0),
         (50.0, 1001.0)]
CHANGE_PRICES = np.array([2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 30.0, 50.0, 100.0])
L2_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
BURN_IN, L2_EVERY, VAL_MONTHS = 9, 3, 6
EV_K = (0.0, 0.02, 0.05)
MIN_BETS, MIN_EVENTS = 100, 10
FIELD_EDGES = [0, 6, 9, 12, 16, 99]
FIELD_LABELS = ["2-5", "6-8", "9-11", "12-15", "16+"]
HOUR_EDGES = [0, 13, 15, 17, 19, 25]
HOUR_LABELS = ["<13", "13-15", "15-17", "17-19", "19+"]
SIDES = (("back", "stake"), ("lay", "liability"), ("lay", "stake"))
SHRINK = 5.0
FEATS_PRICE = ["log_p", "logp_sq", "rank1", "rank2", "rank3", "has_t30", "move", "move_abs", "on_change",
               "logfield_x_logp", "book_x_logp", "hour_x_logp", "sat_x_logp", "t30_x_logp"]
FEATS_MEM = ["log_p", "no_prior", "log1p_n_prior", "log_days_since", "backup7", "prev_won", "prev_surprise",
             "beaten_fav", "prev_steam_pre", "prev_steam_30", "prev_move", "cum_resid", "price_vs_prev",
             "field_diff", "other_country"]

REG: list[dict] = []          # every cell with a z statistic (G5)


# ----------------------------------------------------------------------------- utilities
def out(obj) -> None:
    print(json.dumps(obj, default=_jd), flush=True)


def _jd(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(o).date())
    if isinstance(o, pd.Period):
        return str(o)
    return str(o)


def r4(x):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, 4)


def r6(x):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, 6)


def one_sided_p(z) -> float:
    if z is None or not np.isfinite(z):
        return 1.0
    return 0.5 * math.erfc(z / math.sqrt(2))


def bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    rej = np.zeros(m, dtype=bool)
    if m == 0:
        return rej
    o = np.argsort(p)
    below = np.flatnonzero(p[o] <= q * np.arange(1, m + 1) / m)
    if len(below):
        rej[o[: below.max() + 1]] = True
    return rej


def settle(side: str, staking: str, price: np.ndarray, f: np.ndarray, c: float) -> np.ndarray:
    """Net return per bet [A-54]: back gross = f(P-1) - (1-f); lay = minus that; commission on
    a positive net only; lay per liability = lay / (P-1)."""
    g = f * (price - 1.0) - (1.0 - f)
    if side == "lay":
        g = -g
    r = np.where(g > 0, g * (1.0 - c), g)
    if side == "lay" and staking == "liability":
        r = r / (price - 1.0)
    return r


def cl_se(r: np.ndarray, codes: np.ndarray, ncl: int) -> float:
    if len(r) < 2:
        return float("nan")
    d = np.bincount(codes, weights=r - r.mean(), minlength=ncl)
    G = int(np.count_nonzero(np.bincount(codes, minlength=ncl)))
    if G < 2:
        return float("nan")
    return math.sqrt(float((d * d).sum()) * G / (G - 1)) / len(r)


def on_ladder_change(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.isclose(p[:, None], CHANGE_PRICES[None, :], atol=1e-6).any(1)


def shin_vec(price: np.ndarray, codes: np.ndarray, n: int) -> np.ndarray:
    """ponies.odds.shin_probabilities for every market at once (bisection on z per market)."""
    pi = 1.0 / np.asarray(price, dtype=float)
    s = np.bincount(codes, weights=pi, minlength=n)
    sc = s[codes]
    lo, hi = np.zeros(n), np.full(n, 0.5)

    def probs(z):
        zc = z[codes]
        return (np.sqrt(zc * zc + 4 * (1 - zc) * pi * pi / sc) - zc) / (2 * (1 - zc))
    for _ in range(60):
        mid = (lo + hi) / 2
        above = np.bincount(codes, weights=probs(mid), minlength=n) > 1
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    p = probs((lo + hi) / 2)
    p = np.where(sc <= 1.0, pi / sc, p)
    return p / np.bincount(codes, weights=p, minlength=n)[codes]


# ----------------------------------------------------------------------------- evaluation
class Ev:
    """Arrays of one evaluation universe (one country, one family) for fast cells."""

    def __init__(self, df: pd.DataFrame, label: str):
        self.label = label
        self.n = len(df)
        self.bsp = df["bsp"].to_numpy(float)
        self.pre = df["ltp_pre"].to_numpy(float)
        self.f = df["f"].to_numpy(float)
        self.mc, mu = pd.factorize(df["market_id"])
        self.nm = len(mu)
        self.dc, du = pd.factorize(df["day"])
        self.nd = len(du)
        self.day = df["day"].to_numpy().astype("datetime64[D]")
        md = df.drop_duplicates("market_id")["day"].sort_values().to_numpy().astype("datetime64[D]")
        self.mid = md[len(md) // 2] if len(md) else np.datetime64("2018-11-01")
        self.h1 = self.day < self.mid
        self.l12 = self.day >= LAST12
        self.n_races = self.nm
        self.n_months = int(pd.Series(df["day"]).dt.to_period("M").nunique()) if len(df) else 0

    def _st(self, r, idx):
        n = len(r)
        if n < 2:
            return (float(r.mean()) if n else float("nan")), float("nan"), n
        se = max(float(r.std(ddof=1) / math.sqrt(n)), cl_se(r, self.mc[idx], self.nm), cl_se(r, self.dc[idx], self.nd))
        return float(r.mean()), se, n

    def cell(self, mask, side, staking, price="bsp", reg=None) -> dict:
        P = self.bsp if price == "bsp" else self.pre
        ok = mask & np.isfinite(P) & (P > 1)
        idx = np.flatnonzero(ok)
        pr, f = P[idx], self.f[idx]
        R = {c: settle(side, staking, pr, f, c) for c in COMMS}
        r8 = R[PC]
        h1 = self.h1[idx]
        l12 = self.l12[idx]
        m, se, n = self._st(r8, idx)
        m1, se1, n1 = self._st(r8[h1], idx[h1])
        m2, se2, n2 = self._st(r8[~h1], idx[~h1])
        ml, sel, nl = self._st(r8[l12], idx[l12])
        win = f > 0
        ev1, ev2 = int(win[h1].sum()), int(win[~h1].sum())
        # minimum information: a z needs >= MIN_BETS bets and >= 2 * MIN_EVENTS winners; G4 needs
        # >= MIN_BETS / 2 bets and >= MIN_EVENTS winners in each half (a lay of long shots with no
        # winner in a half has SE 0 and would otherwise "pass")
        info_ok = n >= MIN_BETS and (ev1 + ev2) >= 2 * MIN_EVENTS
        halves_ok = min(n1, n2) >= MIN_BETS // 2 and min(ev1, ev2) >= MIN_EVENTS
        g4 = bool(halves_ok and np.isfinite(se1) and np.isfinite(se2) and se1 > 0 and se2 > 0
                  and m1 > 2 * se1 and m2 > 2 * se2 and nl > 0 and ml > 0)
        z = m / se if info_ok and np.isfinite(se) and se > 0 else float("nan")
        res = {"decisions": int(mask.sum()), "bets": int(n), "roi_8pct": r4(m), "se_clustered": r4(se), "z": r4(z),
               "halves_roi_8pct": [r4(m1), r4(m2)], "halves_se": [r4(se1), r4(se2)], "halves_bets": [n1, n2],
               "halves_winners": [ev1, ev2],
               "last12m_roi_8pct": r4(ml), "last12m_bets": int(nl),
               "roi_0_5_10pct": [r4(R[c].mean()) if n else None for c in (0.0, 0.05, 0.10)],
               "passes_G4": g4, "mde_80pct": r4((2 + Z80) * se),
               "mde_80pct_each_half": r4((2 + Z80) * max(se1, se2)) if np.isfinite(se1) and np.isfinite(se2) else None,
               "fires_per_race": r4(mask.sum() / max(self.n_races, 1)),
               "bets_per_month": r4(n / max(self.n_months, 1)),
               "strike": r4((f > 0).mean()) if n else None, "median_price": r4(np.median(pr)) if n else None,
               "mid": str(self.mid)}
        if reg is not None:
            REG.append({**reg, "side": side, "staking": staking, "settlement": price, "kind": "roi",
                        **{k: res[k] for k in ("bets", "roi_8pct", "se_clustered", "z", "halves_roi_8pct",
                                               "halves_se", "last12m_roi_8pct", "roi_0_5_10pct", "passes_G4",
                                               "mde_80pct", "mde_80pct_each_half", "fires_per_race",
                                               "bets_per_month", "halves_bets", "halves_winners", "median_price", "strike", "mid")}})
        return res

    def paired(self, mask, side, staking, reg=None) -> dict:
        """Same runners: return at BSP minus return at ltp_pre (8%)."""
        ok = mask & np.isfinite(self.bsp) & (self.bsp > 1) & np.isfinite(self.pre)
        idx = np.flatnonzero(ok)
        d = settle(side, staking, self.bsp[idx], self.f[idx], PC) - settle(side, staking, self.pre[idx], self.f[idx], PC)
        d0 = settle(side, staking, self.bsp[idx], self.f[idx], 0.0) - settle(side, staking, self.pre[idx], self.f[idx], 0.0)
        h1 = self.h1[idx]
        m, se, n = self._st(d, idx)
        m1, se1, _ = self._st(d[h1], idx[h1])
        m2, se2, _ = self._st(d[~h1], idx[~h1])
        l12 = self.l12[idx]
        ml, _, _ = self._st(d[l12], idx[l12])
        z = m / se if n >= MIN_BETS and np.isfinite(se) and se > 0 else float("nan")
        lr = np.log(self.bsp[idx] / self.pre[idx])
        res = {"runners": int(n), "diff_8pct": r4(m), "se": r4(se), "z": r4(z), "diff_0pct": r4(d0.mean()) if n else None,
               "halves": [r4(m1), r4(m2)], "halves_se": [r4(se1), r4(se2)], "last12m": r4(ml),
               "mean_log_bsp_over_pre": r4(lr.mean()) if n else None,
               "share_bsp_longer": r4((lr > 1e-9).mean()) if n else None,
               "share_bsp_shorter": r4((lr < -1e-9).mean()) if n else None}
        res["both_halves_gt_2se"] = bool(np.isfinite(se1) and np.isfinite(se2) and m1 > 2 * se1 and m2 > 2 * se2)
        if reg is not None:
            REG.append({**reg, "side": side, "staking": staking, "settlement": "bsp_minus_ltp_pre", "kind": "paired",
                        "bets": int(n), "roi_8pct": r4(m), "se_clustered": r4(se), "z": r4(z),
                        "halves_roi_8pct": [r4(m1), r4(m2)], "halves_se": [r4(se1), r4(se2)], "last12m_roi_8pct": r4(ml),
                        "passes_G4": False})
        return res


def won_over_exp(p: np.ndarray, f: np.ndarray) -> list:
    ok = np.isfinite(p)
    p, f = p[ok], f[ok]
    sp = p.sum()
    if sp <= 0:
        return [None, None]
    return [r4(f.sum() / sp), r4(math.sqrt((p * (1 - p)).sum()) / sp)]


def reg_desc(family, country, rule, segment, z):
    REG.append({"family": family, "country": country, "rule": rule, "segment": segment, "kind": "descriptive",
                "z": r4(z), "passes_G4": False})


# ----------------------------------------------------------------------------- loading
def load_country(code: str, d: Path, frac: float, seed: int) -> tuple[pd.DataFrame, dict]:
    w = pd.read_parquet(d / f"{code}_win.parquet", columns=COLS)
    n_raw = len(w)
    w = w.drop_duplicates(["market_id", "selection_id"])
    if frac < 1:
        mk = w["market_id"].drop_duplicates().sample(frac=frac, random_state=seed)
        w = w[w["market_id"].isin(set(mk))]
    w = w.copy()
    for c in ("ltp_pre", "ltp_t30", "bsp"):
        w.loc[~(w[c] > 1), c] = np.nan
    w["f"] = np.where(w["won"] == 1, w["dead_heat_share"].fillna(1.0), 0.0).astype(float)
    w["harness"] = w["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b", regex=True)
    loc = w["start"].dt.tz_convert(TZ[code])
    w["day"] = loc.dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    w["hour"] = (loc.dt.hour + loc.dt.minute / 60.0).to_numpy()
    w["wday"] = loc.dt.weekday.to_numpy()
    w["mi"] = (w["day"].dt.year * 12 + w["day"].dt.month - 1).astype(int)
    w["year"] = w["day"].dt.year
    w["pn"] = pd.to_numeric(w["program"], errors="coerce")
    w["i_pre"] = 1 / w["ltp_pre"]
    w["i_30"] = 1 / w["ltp_t30"]
    w["i_bsp"] = 1 / w["bsp"]
    w["won_na"] = w["won"].isna()
    g = w.groupby("market_id", sort=False)
    m = pd.DataFrame({"n": g.size(), "won_na": g["won_na"].sum(), "won_sum": g["won"].sum(),
                      "pre_n": g["ltp_pre"].count(), "pre_book": g["i_pre"].sum(),
                      "t30_n": g["ltp_t30"].count(), "t30_book": g["i_30"].sum(),
                      "bsp_n": g["bsp"].count(), "bsp_book": g["i_bsp"].sum(),
                      "pn_cnt": g["pn"].count(), "pn_max": g["pn"].max()})
    m["settled"] = (m["won_na"] == 0) & (m["won_sum"] >= 1) & (m["n"] >= 3)
    m["gap"] = (m["pn_cnt"] == m["n"]) & (m["pn_max"] > m["n"])
    m["g6"] = m["gap"] & (m["pre_book"] < 0.97)
    m["all_pre"] = m["pre_n"] == m["n"]
    m["coh"] = m["all_pre"] & m["pre_book"].between(0.97, 1.03)
    m["bsp_mk"] = m["bsp_n"] > 0
    m["all_bsp"] = m["bsp_n"] == m["n"]
    m["all_t30"] = m["t30_n"] == m["n"]
    s = m[m["settled"]]
    st = {"rows_raw": int(n_raw), "rows_dedup": int(len(w)), "markets": int(len(m)), "settled": int(len(s)),
          "g6_dropped": int(s["g6"].sum()), "gap_markets": int(s["gap"].sum()),
          "coherent": int((s["coh"] & ~s["g6"]).sum()), "bsp_offered": int(s["bsp_mk"].sum()),
          "coherent_and_bsp": int((s["coh"] & s["bsp_mk"] & ~s["g6"]).sum()),
          "share_runners_t30": r4(s["t30_n"].sum() / s["n"].sum()),
          "pre_book_q05_50_95": [r4(x) for x in s.loc[s["all_pre"], "pre_book"].quantile([.05, .5, .95])]}
    w = w[w["market_id"].map(m["settled"]).to_numpy(dtype=bool)].copy()
    for c in ("n", "g6", "coh", "bsp_mk", "all_bsp", "all_pre", "pre_book", "bsp_book", "t30_book", "all_t30"):
        w["m_" + c] = w["market_id"].map(m[c]).to_numpy()
    w["country"] = code
    w = w.drop(columns=["i_pre", "i_30", "i_bsp", "won_na", "pn", "program", "dead_heat_share"])
    w = w.sort_values(["start", "market_id", "ltp_pre", "selection_id"], kind="mergesort").reset_index(drop=True)
    # market-normalised probabilities
    w["p_bsp"] = np.where(w["m_all_bsp"], (1 / w["bsp"]) / w["m_bsp_book"], np.nan)
    w["p_pre_norm"] = np.where(w["m_all_pre"], (1 / w["ltp_pre"]) / w["m_pre_book"], np.nan)
    coh = (w["m_coh"] & ~w["m_g6"]).to_numpy()
    w["p_mkt"] = np.nan
    w["log_p"] = np.nan
    if coh.any():
        sub = w.loc[coh]
        codes, uniq = pd.factorize(sub["market_id"])
        p = shin_vec(sub["ltp_pre"].to_numpy(), codes, len(uniq))
        w.loc[coh, "p_mkt"] = p
        w.loc[coh, "log_p"] = np.log(p)
    w["rank_pre"] = w.groupby("market_id", sort=False).cumcount() + 1   # sorted by ltp_pre, selection_id
    w.loc[~w["m_all_pre"].to_numpy(), "rank_pre"] = 0
    return w, st


# ----------------------------------------------------------------------------- model engine
def newton(Z, codes, nr, y, l2v, w0=None, max_iter=30, tol=1e-10):
    """Repo ConditionalLogit objective (mean per-race log loss + 0.5 sum l2_j w_j^2 on standardised
    features), damped Newton steps."""
    k = Z.shape[1]
    w = np.zeros(k) if w0 is None else w0.copy()

    def lp(w):
        p = race_softmax(Z @ w, codes, nr)
        return -float((y * np.log(p + 1e-15)).sum()) / nr + 0.5 * float((l2v * w * w).sum()), p
    loss, p = lp(w)
    it = 0
    for it in range(1, max_iter + 1):
        g = Z.T @ (p - y) / nr + l2v * w
        zbar = np.empty((nr, k))
        for j in range(k):
            zbar[:, j] = np.bincount(codes, weights=p * Z[:, j], minlength=nr)
        Zc = Z - zbar[codes]
        H = (Zc * p[:, None]).T @ Zc / nr + np.diag(l2v) + 1e-10 * np.eye(k)
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:
            wn = w - t * step
            ln, pn = lp(wn)
            if ln <= loss or t < 1e-4:
                break
            t *= 0.5
        if ln > loss:
            break
        imp = loss - ln
        w, loss, p = wn, ln, pn
        if imp < tol:
            break
    return w, loss, it


class Block:
    """Rows sorted by (month, start, market): contiguous month blocks, monotone race codes."""

    def __init__(self, df: pd.DataFrame, feats: list[str], unpen=("log_p",)):
        self.df = df
        self.feats = feats
        self.X = df[feats].to_numpy(float)
        self.rc = pd.factorize(df["market_id"])[0]
        assert np.all(np.diff(self.rc) >= 0), "race codes must be monotone"
        f = df["f"].to_numpy(float)
        tot = np.bincount(self.rc, weights=f)
        self.y = f / tot[self.rc]
        self.mi = df["mi"].to_numpy()
        self.months = np.unique(self.mi)
        self.start_of = {m: int(np.searchsorted(self.mi, m, "left")) for m in self.months}
        self.end_of = {m: int(np.searchsorted(self.mi, m, "right")) for m in self.months}
        self.l2mask = np.array([0.0 if c in unpen else 1.0 for c in feats])

    def rng(self, a, b):
        X = self.X[a:b]
        c = self.rc[a:b] - self.rc[a]
        return X, c, int(c[-1]) + 1, self.y[a:b]

    @staticmethod
    def standardise(Xtr):
        imp = np.nan_to_num(np.nanmean(Xtr, axis=0))
        Xf = np.where(np.isnan(Xtr), imp, Xtr)
        mu = Xf.mean(0)
        sd = Xf.std(0)
        sd[sd == 0] = 1.0
        return imp, mu, sd

    @staticmethod
    def apply(X, imp, mu, sd):
        return (np.where(np.isnan(X), imp, X) - mu) / sd

    def fit(self, a, b, l2, w0=None):
        X, c, nr, y = self.rng(a, b)
        imp, mu, sd = self.standardise(X)
        w, loss, it = newton(self.apply(X, imp, mu, sd), c, nr, y, l2 * self.l2mask, w0)
        return w, (imp, mu, sd), it

    def predict(self, a, b, w, prep):
        X, c, nr, _ = self.rng(a, b)
        return race_softmax(self.apply(X, *prep) @ w, c, nr)

    def ll_mean(self, a, b, p):
        _, c, nr, y = self.rng(a, b)
        return float(-(y * np.log(np.clip(p, 1e-15, None))).sum() / nr)

    def select_l2(self, i):
        """L2 for prediction month index i, from months strictly before it only."""
        ms = self.months
        a_in, b_in = 0, self.start_of[ms[i - VAL_MONTHS]]
        a_v, b_v = b_in, self.start_of[ms[i]]
        X, c, nr, y = self.rng(a_in, b_in)
        prep = self.standardise(X)
        Z = self.apply(X, *prep)
        Xv, cv, nv, yv = self.rng(a_v, b_v)
        Zv = self.apply(Xv, *prep)
        scores = {}
        w = None

        def score(l2, w0):
            ww, _, _ = newton(Z, c, nr, y, l2 * self.l2mask, w0)
            pv = race_softmax(Zv @ ww, cv, nv)
            return float(-(yv * np.log(np.clip(pv, 1e-15, None))).sum() / nv), ww
        for l2 in sorted(L2_GRID, reverse=True):
            scores[l2], w = score(l2, w)
        ext = 0
        while ext < 4:
            best = min(scores, key=scores.get)
            if best == max(scores):
                new = best * 10.0
            elif best == min(scores):
                new = best / 10.0
            else:
                break
            scores[new], _ = score(new, None)
            ext += 1
        best = min(scores, key=scores.get)
        edge = bool(best == max(scores) or best == min(scores))
        return best, edge, ext, {k: round(v, 6) for k, v in sorted(scores.items())}

    def walk_forward(self):
        n = len(self.X)
        pred = np.full(n, np.nan)
        info = {"refits": 0, "l2_choices": [], "edge_hits": 0, "extensions": 0, "coefs": [], "iters": []}
        w_prev = None
        l2 = None
        for i, m in enumerate(self.months):
            if i < BURN_IN:
                continue
            if (i - BURN_IN) % L2_EVERY == 0 or l2 is None:
                l2, edge, ext, sc = self.select_l2(i)
                info["l2_choices"].append(l2)
                info["edge_hits"] += int(edge)
                info["extensions"] += ext
                if len(info["l2_choices"]) == 1:
                    info["first_selection_scores"] = sc
            w, prep, it = self.fit(0, self.start_of[m], l2, w_prev)
            w_prev = w
            pred[self.start_of[m]:self.end_of[m]] = self.predict(self.start_of[m], self.end_of[m], w, prep)
            info["refits"] += 1
            info["iters"].append(it)
            info["coefs"].append(w / prep[2])
        return pred, info


def per_race_ll(rc, y, p):
    nr = int(rc.max()) + 1
    return np.bincount(rc, weights=-y * np.log(np.clip(p, 1e-15, None)), minlength=nr)


def ll_gain(blk: Block, pred: np.ndarray, label: str) -> dict:
    ok = np.isfinite(pred)
    df = blk.df[ok]
    rc = pd.factorize(df["market_id"])[0]
    y = blk.y[ok]
    gm = per_race_ll(rc, y, df["p_mkt"].to_numpy())
    gx = per_race_ll(rc, y, pred[ok])
    d = gm - gx
    days = df.drop_duplicates("market_id")["day"].to_numpy().astype("datetime64[D]")
    mid = np.sort(days)[len(days) // 2]
    h1 = days < mid
    l12 = days >= LAST12

    def ms(x):
        return r6(x.mean()), (r6(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None)
    g, se = ms(d)
    g1, s1 = ms(d[h1])
    g2, s2 = ms(d[~h1])
    res = {"label": label, "races": int(len(d)), "ll_market": r6(gm.mean()), "ll_model": r6(gx.mean()),
           "gain": g, "se": se, "z": r4(g / se) if se else None, "halves_gain": [g1, g2], "halves_se": [s1, s2],
           "last12m_gain": r6(d[l12].mean()) if l12.any() else None, "mid": str(mid),
           "positive_2se_both_halves": bool(s1 and s2 and g1 > 2 * s1 and g2 > 2 * s2)}
    REG.append({"family": "logloss", "country": label, "rule": "ll_gain", "segment": "all", "kind": "logloss",
                "z": res["z"], "passes_G4": False})
    return res


def model_bets(E: Ev, pred: np.ndarray, pre: np.ndarray, has_bsp: np.ndarray, fam: str, country: str) -> list[dict]:
    p = pred
    evb = p * (pre - 1) * (1 - PC) - (1 - p)
    evl = ((1 - p) * (1 - PC) - p * (pre - 1)) / (pre - 1)
    rows = []
    for k in EV_K:
        for side, stk in SIDES:
            mask = np.isfinite(p) & has_bsp & ((evb > k) if side == "back" else (evl > k))
            nm = f"{fam}_{side}_ev_gt_{k:g}"
            c = E.cell(mask, side, stk, reg={"family": fam, "country": country, "rule": nm, "segment": "all"})
            rows.append({"rule": nm, "staking": stk, **{kk: c[kk] for kk in (
                "bets", "roi_8pct", "se_clustered", "z", "halves_roi_8pct", "halves_se", "last12m_roi_8pct",
                "roi_0_5_10pct", "passes_G4", "fires_per_race", "mde_80pct", "median_price")}})
    return rows


def ratio_calibration(E: Ev, pred, pmkt, fam, country) -> list[dict]:
    ratio = pred / pmkt
    edges = [0, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 100]
    rows = []
    ok = np.isfinite(ratio)
    for lo, hi in zip(edges, edges[1:]):
        s = ok & (ratio >= lo) & (ratio < hi)
        wm, wse = won_over_exp(pmkt[s], E.f[s])
        wx, _ = won_over_exp(pred[s], E.f[s])
        if wse:
            reg_desc(fam + "_calib", country, f"ratio_{lo}_{hi}", "all", (wm - 1) / wse)
        rows.append({"ratio_bin": f"[{lo},{hi})", "runners": int(s.sum()), "won_over_mkt": wm, "se": wse,
                     "won_over_model": wx})
    return rows


# ----------------------------------------------------------------------------- part 1: late money
def part1(w: pd.DataFrame, code: str) -> dict:
    U = w[(w["m_coh"] & ~w["m_g6"] & w["m_bsp_mk"] & w["ltp_t30"].notna()).to_numpy()].copy()
    U["move"] = np.log(U["ltp_t30"] / U["ltp_pre"])
    mv = U["move"].to_numpy()
    mi = U["mi"].to_numpy()
    months = np.unique(mi)
    dec = np.full(len(U), -1)
    cuts_first_last = []
    for i, m in enumerate(months):
        prior = mi < m
        if prior.sum() < 1000:
            continue
        cut = np.quantile(mv[prior], np.arange(1, 10) / 10)
        if not cuts_first_last or i == len(months) - 1:
            cuts_first_last.append([r4(x) for x in cut])
        s = mi == m
        dec[s] = np.searchsorted(cut, mv[s], side="right")
    U = U[dec >= 0].copy()
    dec = dec[dec >= 0]
    E = Ev(U, f"{code}_late")
    pm = U["p_mkt"].to_numpy()
    pb = U["p_bsp"].to_numpy()
    rows = []
    for k in range(10):
        s = dec == k
        wm = won_over_exp(pm[s], E.f[s])
        wb = won_over_exp(pb[s], E.f[s])
        for nm, ww in (("calib_pre", wm), ("calib_bsp", wb)):
            if ww[1]:
                reg_desc("late_money_calib", code, f"D{k + 1}_{nm}", "all", (ww[0] - 1) / ww[1])
        row = {"decile": k + 1, "runners": int(s.sum()), "move_mean": r4(mv_mean(U, s)),
               "won_over_exp_pre": wm, "won_over_exp_bsp": wb,
               "mean_log_bsp_over_pre": r4(np.nanmean(np.log(E.bsp[s] / E.pre[s]))) if s.any() else None,
               "median_pre": r4(np.median(E.pre[s])) if s.any() else None}
        for side, stk in SIDES:
            c = E.cell(s, side, stk, reg={"family": "late_money", "country": code, "rule": f"{side}_D{k + 1}",
                                          "segment": "all"})
            tag = side if side == "back" else f"lay_{stk}"
            row[tag] = {kk: c[kk] for kk in ("bets", "roi_8pct", "se_clustered", "z", "halves_roi_8pct",
                                             "halves_se", "last12m_roi_8pct", "roi_0_5_10pct", "passes_G4")}
        cp = E.cell(s, "back", "stake", price="pre")
        reg_desc("late_money_paper_ltp", code, f"back_D{k + 1}_at_ltp_pre", "all", cp["z"] if cp["z"] is not None else float("nan"))
        row["paper_back_at_ltp_pre_roi8"] = cp["roi_8pct"]
        rows.append(row)
    corr = float(np.corrcoef(U["move"].to_numpy(), np.log(U["p_mkt"].to_numpy()))[0, 1]) if len(U) > 2 else None
    return {"section": "part1_late_money", "country": code, "runners": int(len(U)), "races": E.nm,
            "mid": str(E.mid), "cuts_first_last": cuts_first_last, "corr_move_logp": r4(corr), "deciles": rows}


def mv_mean(U, s):
    return U["move"].to_numpy()[s].mean() if s.any() else float("nan")


# ----------------------------------------------------------------------------- part 2: price-only logit
def price_features(w: pd.DataFrame, code: str) -> pd.DataFrame:
    D = w[(w["m_coh"] & ~w["m_g6"]).to_numpy()].copy()
    D["logp_sq"] = D["log_p"] ** 2
    for r in (1, 2, 3):
        D[f"rank{r}"] = (D["rank_pre"] == r).astype(float)
    D["has_t30"] = D["ltp_t30"].notna().astype(float)
    D["move"] = np.where(D["has_t30"] > 0, np.log(D["ltp_t30"] / D["ltp_pre"]), 0.0)
    D["move_abs"] = D["move"] * np.abs(D["move"])
    D["on_change"] = on_ladder_change(D["ltp_pre"].to_numpy()).astype(float)
    D["logfield_x_logp"] = np.log(D["field_size"].clip(lower=2).astype(float)) * D["log_p"]
    D["book_x_logp"] = (D["m_pre_book"] - 1.0) * D["log_p"]
    D["hour_x_logp"] = (D["hour"] - 15.0) * D["log_p"]
    D["sat_x_logp"] = (D["wday"] == 5).astype(float) * D["log_p"]
    D["t30_x_logp"] = D["has_t30"] * D["log_p"]
    D["harness_x_logp"] = D["harness"].astype(float) * D["log_p"]
    D = D.sort_values(["mi", "start", "market_id", "ltp_pre", "selection_id"], kind="mergesort").reset_index(drop=True)
    return D


def part2(w: pd.DataFrame, code: str, fit_check: bool) -> tuple[dict, dict]:
    D = price_features(w, code)
    feats = FEATS_PRICE + (["harness_x_logp"] if code == "nz" and D["harness"].any() else [])
    blk = Block(D, feats)
    chk = None
    if fit_check:
        a, b = 0, blk.start_of[blk.months[min(2, len(blk.months) - 1)]]
        sub = D.iloc[a:b].assign(race_id=D["market_id"].iloc[a:b])
        t1 = time.time()
        gd = ConditionalLogit(feats, l2=0.01, unpenalized=("log_p",)).fit(sub)
        t_gd = time.time() - t1
        wn, prep, it = blk.fit(a, b, 0.01)
        pn = blk.predict(a, b, wn, prep)
        pg = gd.predict_proba(sub)
        chk = {"section": "fit_check", "country": code, "races": int(sub["market_id"].nunique()),
               "gd_loss": r6(gd.history[-1]), "newton_loss": r6(blk.ll_mean(a, b, pn) + 0.5 * 0.01 * float((wn[blk.l2mask > 0] ** 2).sum())),
               "max_abs_prob_diff": r6(np.max(np.abs(pn - pg))), "gd_seconds": round(t_gd, 1), "newton_iters": it}
    t1 = time.time()
    pred, info = blk.walk_forward()
    ll = ll_gain(blk, pred, f"{code}_price_logit")
    ok = np.isfinite(pred)
    T = D[ok].reset_index(drop=True)
    E = Ev(T, f"{code}_model")
    pr = pred[ok]
    has_bsp = T["m_bsp_mk"].to_numpy(bool)
    bets = model_bets(E, pr, T["ltp_pre"].to_numpy(), has_bsp, "price_logit", code)
    cal = ratio_calibration(E, pr, T["p_mkt"].to_numpy(), "price_logit", code)
    co = np.array(info["coefs"])
    res = {"section": "part2_price_logit", "country": code, "features": feats, "rows": int(len(D)),
           "predicted_rows": int(ok.sum()), "predicted_from": str(T["day"].min().date()) if len(T) else None,
           "refits": info["refits"], "l2_selections": len(info["l2_choices"]),
           "l2_choices": sorted(set(info["l2_choices"])), "l2_choice_counts": pd.Series(info["l2_choices"]).value_counts().to_dict(),
           "l2_edge_hits_after_extension": info["edge_hits"], "l2_extensions": info["extensions"],
           "first_selection_val_scores": info.get("first_selection_scores"),
           "newton_iters_median": float(np.median(info["iters"])) if info["iters"] else None,
           "coef_raw_median": {f: r4(np.median(co[:, j])) for j, f in enumerate(feats)} if len(co) else None,
           "logloss": ll, "bets": bets, "ratio_calibration": cal, "seconds": round(time.time() - t1, 1)}
    return res, chk


# ----------------------------------------------------------------------------- part 3: segment scan
def part3(w: pd.DataFrame, code: str) -> tuple[dict, Ev, pd.DataFrame]:
    U = w[(w["m_coh"] & ~w["m_g6"] & w["m_bsp_mk"]).to_numpy()].copy().reset_index(drop=True)
    E = Ev(U, f"{code}_scan")
    pre = U["ltp_pre"].to_numpy()
    sels = {"fav": (U["rank_pre"] == 1).to_numpy()}
    for lo, hi in BANDS:
        sels[f"band_{lo:g}_{hi:g}" if hi < 1001 else f"band_ge{lo:g}"] = (pre >= lo) & (pre < hi)
    segs: dict[str, np.ndarray] = {"all|all": np.ones(len(U), bool)}
    fb = pd.cut(U["field_size"], FIELD_EDGES, right=False, labels=FIELD_LABELS)
    for lab in FIELD_LABELS:
        segs[f"field|{lab}"] = (fb == lab).to_numpy()
    top = U.drop_duplicates("market_id")["venue"].value_counts().head(15).index.tolist()
    for v in top:
        segs[f"venue|{v}"] = (U["venue"] == v).to_numpy()
    segs["venue|other"] = ~U["venue"].isin(top).to_numpy()
    hb = pd.cut(U["hour"], HOUR_EDGES, right=False, labels=HOUR_LABELS)
    for lab in HOUR_LABELS:
        segs[f"hour|{lab}"] = (hb == lab).to_numpy()
    for d in range(7):
        segs[f"weekday|{d}"] = (U["wday"] == d).to_numpy()
    yr = U["year"].to_numpy()
    segs["year|2016-17"] = yr <= 2017
    for y in (2018, 2019, 2020):
        segs[f"year|{y}"] = yr == y
    # book quartiles: cutoffs from markets in earlier months only
    mk = U.drop_duplicates("market_id")[["market_id", "mi", "m_pre_book"]]
    q = pd.Series(-1, index=mk["market_id"].to_numpy())
    mmi = mk["mi"].to_numpy()
    mbk = mk["m_pre_book"].to_numpy()
    qv = np.full(len(mk), -1)
    for m in np.unique(mmi):
        prior = mmi < m
        if prior.sum() < 200:
            continue
        cut = np.quantile(mbk[prior], [0.25, 0.5, 0.75])
        s = mmi == m
        qv[s] = np.searchsorted(cut, mbk[s], side="right")
    q[:] = qv
    uq = U["market_id"].map(q).to_numpy()
    for k in range(4):
        segs[f"bookq|Q{k + 1}"] = uq == k
    cells = []
    for sn, sm in segs.items():
        for selname, selm in sels.items():
            mask = sm & selm
            for side, stk in SIDES:
                c = E.cell(mask, side, stk, reg={"family": "scan", "country": code, "rule": f"{side}_{selname}",
                                                 "segment": sn})
                cells.append({"segment": sn, "sel": selname, "side": side, "staking": stk, **{k: c[k] for k in (
                    "bets", "roi_8pct", "se_clustered", "z", "halves_roi_8pct", "halves_se", "last12m_roi_8pct",
                    "passes_G4")}})
    allrows = [c for c in cells if c["segment"] == "all|all"]
    venue_names = {f"venue|{v}": f"venue|V{i + 1:02d}" for i, v in enumerate(top)}
    res = {"section": "part3_scan", "country": code, "races": E.nm, "runners": int(len(U)), "mid": str(E.mid),
           "segments": len(segs), "selections": len(sels), "cells": len(cells),
           "top15_venue_ranks": {f"V{i + 1:02d}": v for i, v in enumerate(top)},
           "all_segment": allrows,
           "best_by_z": sorted([c for c in cells if c["z"] is not None and c["bets"] >= 200],
                               key=lambda c: -c["z"])[:8],
           "passing_G4": [c for c in cells if c["passes_G4"]]}
    return res, E, U


# ----------------------------------------------------------------------------- part 5: BSP vs ltp_pre
def part5(E: Ev, U: pd.DataFrame, code: str) -> dict:
    pre = U["ltp_pre"].to_numpy()
    onc = on_ladder_change(pre)
    rows = []
    for lo, hi in BANDS:
        bn = f"band_{lo:g}_{hi:g}" if hi < 1001 else f"band_ge{lo:g}"
        b = (pre >= lo) & (pre < hi)
        row = {"band": bn, "runners": int(b.sum()), "share_on_change_print": r4(onc[b].mean()) if b.any() else None}
        no_change_prices = not (b & onc).any()
        for var, vm in (("all", b), ("excl_change", b & ~onc), ("only_change", b & onc)):
            if var != "all" and no_change_prices:
                row[var] = "identical to all (no ladder-change price in band)" if var == "excl_change" else "empty"
                continue          # not evaluated / not counted
            pb = E.paired(vm, "back", "stake", reg={"family": "bsp_vs_ltp", "country": code,
                                                     "rule": f"paired_back_{bn}", "segment": var})
            pl = E.paired(vm, "lay", "liability", reg={"family": "bsp_vs_ltp", "country": code,
                                                        "rule": f"paired_lay_liab_{bn}", "segment": var})
            row[var] = {"paired_back": pb, "paired_lay_liab": {k: pl[k] for k in ("diff_8pct", "se", "z", "halves")}}
            if var != "all":   # 'all' = the part-3 all-segment cell, not counted twice
                for side, stk in SIDES:
                    c = E.cell(vm, side, stk, reg={"family": "bsp_band_tick", "country": code,
                                                   "rule": f"{side}_{bn}", "segment": var})
                    row[var][side if side == "back" else f"lay_{stk}"] = {k: c[k] for k in (
                        "bets", "roi_8pct", "se_clustered", "z", "halves_roi_8pct", "halves_se", "last12m_roi_8pct",
                        "roi_0_5_10pct", "passes_G4")}
            else:
                for side, stk in SIDES:
                    c = E.cell(vm, side, stk)     # not registered (duplicate of part 3)
                    row[var][side if side == "back" else f"lay_{stk}"] = {k: c[k] for k in (
                        "bets", "roi_8pct", "se_clustered", "z", "halves_roi_8pct", "halves_se", "last12m_roi_8pct",
                        "roi_0_5_10pct", "passes_G4")}
        rows.append(row)
    return {"section": "part5_bsp_vs_ltp", "country": code, "mid": str(E.mid), "bands": rows}


# ----------------------------------------------------------------------------- part 4: horse memory
def memory_history(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    h = pd.concat(frames, ignore_index=True)
    h["p_run"] = np.where(h["m_all_bsp"], h["p_bsp"], h["p_pre_norm"])
    h = h.sort_values(["selection_id", "start"], kind="mergesort").reset_index(drop=True)
    dup = h.duplicated(["selection_id", "start"])
    n_dup = int(dup.sum())
    h = h[~dup].reset_index(drop=True)
    g = h.groupby("selection_id", sort=False)
    h["n_prior"] = g.cumcount()
    pmax = h.groupby("market_id")["p_run"].transform("max")
    h["is_fav"] = np.where(h["p_run"].notna(), (h["p_run"] >= pmax - 1e-12).astype(float), np.nan)
    for c in ("start", "f", "p_run", "ltp_pre", "ltp_t30", "bsp", "field_size", "is_fav", "country"):
        h["prev_" + c] = g[c].shift(1)
    hp = (h["n_prior"] > 0).to_numpy()
    assert (h.loc[hp, "prev_start"] < h.loc[hp, "start"]).all(), "leakage: prev_start >= start"
    resid = (h["f"] - h["p_run"]).where(h["p_run"].notna(), 0.0)
    h["cum_resid"] = (resid.groupby(h["selection_id"]).cumsum() - resid) / (h["n_prior"] + SHRINK)
    days = (h["start"] - h["prev_start"]).dt.total_seconds() / 86400.0
    h["days_since"] = days
    h["no_prior"] = (~hp).astype(float)
    h["log1p_n_prior"] = np.log1p(h["n_prior"].astype(float))
    h["log_days_since"] = np.log1p(days)
    h["backup7"] = (hp & (days <= 7)).astype(float)
    h["prev_won"] = np.where(hp, (h["prev_f"] > 0).astype(float), 0.0)
    h["prev_surprise"] = h["prev_f"] - h["prev_p_run"]
    h["beaten_fav"] = (hp & (h["prev_is_fav"] == 1) & (h["prev_f"] == 0)).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        h["prev_steam_pre"] = np.log(h["prev_ltp_pre"] / h["prev_bsp"])
        h["prev_steam_30"] = np.log(h["prev_ltp_t30"] / h["prev_bsp"])
        h["prev_move"] = np.log(h["prev_ltp_t30"] / h["prev_ltp_pre"])
        h["price_vs_prev"] = np.log(h["ltp_pre"] / h["prev_ltp_pre"])
    h["field_diff"] = (h["field_size"] - h["prev_field_size"]).astype(float)
    h["other_country"] = (hp & (h["prev_country"] != h["country"])).astype(float)
    info = {"section": "part4_history", "rows": int(len(h)), "horses": int(h["selection_id"].nunique()),
            "dup_key_start_dropped": n_dup, "share_with_prior": r4(hp.mean()),
            "gap_days_median": r4(days.median()), "share_backup7_of_prior": r4(h.loc[hp, "backup7"].mean()),
            "share_prev_other_country_of_prior": r4(h.loc[hp, "other_country"].mean()),
            "leakage_assert": "prev_start < start: passed"}
    return h, info


def part4(h: pd.DataFrame, code: str) -> dict:
    D = h[(h["country"] == code) & h["m_coh"] & ~h["m_g6"]].copy()
    hist = [c for c in FEATS_MEM if c != "log_p"]
    med = D[hist].groupby(D["market_id"].to_numpy()).transform("median")
    D[hist] = D[hist].where(D[hist].notna(), med)
    D = D.sort_values(["mi", "start", "market_id", "ltp_pre", "selection_id"], kind="mergesort").reset_index(drop=True)
    blk = Block(D, FEATS_MEM)
    t1 = time.time()
    pred, info = blk.walk_forward()
    ll = ll_gain(blk, pred, f"{code}_memory_logit")
    ok = np.isfinite(pred)
    T = D[ok].reset_index(drop=True)
    E = Ev(T, f"{code}_memory")
    pr = pred[ok]
    has_bsp = T["m_bsp_mk"].to_numpy(bool)
    bets = model_bets(E, pr, T["ltp_pre"].to_numpy(), has_bsp, "memory_logit", code)
    cal = ratio_calibration(E, pr, T["p_mkt"].to_numpy(), "memory_logit", code)
    # n_prior, cum_resid, backup7, beaten_fav, prev_won are never NaN (0 without a prior run), so
    # the race-median imputation left them unchanged
    raw = T
    n_prior = T["n_prior"].to_numpy()
    cum = T["cum_resid"].to_numpy()
    assert not np.isnan(cum).any()
    # cum_resid decile cutoffs from earlier months (rows with >= 1 prior run in this country's universe)
    allD = h[(h["country"] == code) & h["m_coh"] & ~h["m_g6"] & (h["n_prior"] > 0)]
    a_mi, a_cum = allD["mi"].to_numpy(), allD["cum_resid"].to_numpy()
    tmi = T["mi"].to_numpy()
    top = np.zeros(len(T), bool)
    bot = np.zeros(len(T), bool)
    for m in np.unique(tmi):
        prior = a_mi < m
        if prior.sum() < 1000:
            continue
        lo, hi = np.quantile(a_cum[prior], [0.1, 0.9])
        s = (tmi == m) & (n_prior > 0)
        top |= s & (cum > hi)
        bot |= s & (cum < lo)
    rules = {"backup7_back": ("back", raw["backup7"].to_numpy() == 1),
             "cum_resid_top_decile_back": ("back", top),
             "cum_resid_bottom_decile_lay": ("lay", bot),
             "beaten_fav_back": ("back", raw["beaten_fav"].to_numpy() == 1),
             "prev_won_lay": ("lay", raw["prev_won"].to_numpy() == 1),
             "prev_won_back": ("back", raw["prev_won"].to_numpy() == 1),
             "backup7_lay": ("lay", raw["backup7"].to_numpy() == 1)}
    simple = []
    pm = T["p_mkt"].to_numpy()
    for nm, (side, msk) in rules.items():
        mask = msk & has_bsp
        stakings = ["stake"] if side == "back" else ["liability", "stake"]
        wo = won_over_exp(pm[mask], E.f[mask])
        if wo[1]:
            reg_desc("memory_calib", code, nm, "all", (wo[0] - 1) / wo[1])
        for stk in stakings:
            c = E.cell(mask, side, stk, reg={"family": "memory_rule", "country": code, "rule": f"{nm}", "segment": "all"})
            simple.append({"rule": nm, "staking": stk, "won_over_exp_pre": wo, **{k: c[k] for k in (
                "bets", "roi_8pct", "se_clustered", "z", "halves_roi_8pct", "halves_se", "last12m_roi_8pct",
                "roi_0_5_10pct", "passes_G4", "fires_per_race", "mde_80pct")}})
    co = np.array(info["coefs"])
    return {"section": "part4_memory", "country": code, "rows": int(len(D)), "predicted_rows": int(ok.sum()),
            "predicted_from": str(T["day"].min().date()) if len(T) else None, "refits": info["refits"],
            "l2_choice_counts": pd.Series(info["l2_choices"]).value_counts().to_dict(),
            "l2_edge_hits_after_extension": info["edge_hits"], "l2_extensions": info["extensions"],
            "first_selection_val_scores": info.get("first_selection_scores"),
            "coef_raw_median": {f: r4(np.median(co[:, j])) for j, f in enumerate(FEATS_MEM)} if len(co) else None,
            "coef_sign_share_positive": {f: r4((co[:, j] > 0).mean()) for j, f in enumerate(FEATS_MEM)} if len(co) else None,
            "logloss": ll, "bets": bets, "ratio_calibration": cal, "simple_rules": simple,
            "seconds": round(time.time() - t1, 1)}


# ----------------------------------------------------------------------------- primary claims
UNIV = ("coherent-book markets (every runner priced at ltp_pre, sum 1/ltp_pre in [0.97, 1.03], G6 markets "
        "dropped) where BSP is offered")


def primary_specs() -> list[tuple]:
    """(family, country, rule, segment, staking, claim, bet rule): fixed before the full run."""
    sp = []
    for c in COUNTRIES:
        mv = (f"{c.upper()}: runners with an ltp_t30 in {UNIV}; move = log(ltp_t30/ltp_pre) decile from cutoffs "
              f"of earlier months; decided at the off (ltp_pre known), SP bet")
        sp += [("late_money", c, "back_D1", "all", "stake", f"{c.upper()} late money: back the strongest drifters "
                "(move decile 1) at BSP", f"{mv}; back 1 unit at BSP if move decile = 1 (ltp_pre >> ltp_t30)"),
               ("late_money", c, "back_D10", "all", "stake", f"{c.upper()} late money: back the strongest steamers "
                "(move decile 10) at BSP", f"{mv}; back 1 unit at BSP if move decile = 10"),
               ("late_money", c, "lay_D10", "all", "liability", f"{c.upper()} late money: lay the strongest steamers "
                "(decile 10) at BSP", f"{mv}; lay at BSP if move decile = 10, 1 unit liability"),
               ("late_money", c, "lay_D1", "all", "liability", f"{c.upper()} late money: lay the strongest drifters "
                "(decile 1) at BSP", f"{mv}; lay at BSP if move decile = 1, 1 unit liability")]
        pm = (f"{c.upper()}: walk-forward price-only residual logit (monthly refit, L2 chosen on earlier data) at "
              f"ltp_pre in {UNIV}")
        sp += [("price_logit", c, "price_logit_back_ev_gt_0", "all", "stake", f"{c.upper()} price-only logit: back "
                "when model EV at ltp_pre after 8% > 0", f"{pm}; back at BSP if p*(ltp_pre-1)*0.92-(1-p) > 0"),
               ("price_logit", c, "price_logit_lay_ev_gt_0", "all", "liability", f"{c.upper()} price-only logit: lay "
                "when model lay EV after 8% > 0", f"{pm}; lay at BSP if ((1-p)*0.92 - p*(ltp_pre-1))/(ltp_pre-1) > 0")]
        sp += [("scan", c, "back_fav", "all|all", "stake", f"{c.upper()}: back the ltp_pre favourite at BSP",
                f"{c.upper()}: rank-1 runner by ltp_pre in {UNIV}; back at BSP")]
        sp.append(("scan", c, "back_band_6.5_10", "all|all", "stake",
                   f"{c.upper()}: back ltp_pre 6.5-10 at BSP (band holds no ladder-change price)",
                   f"{c.upper()}: runners with ltp_pre in [6.5,10), {UNIV}; back at BSP"))
        for lo, hi in ((4.5, 6.5), (10, 15)):
            sp.append(("bsp_band_tick", c, f"back_band_{lo:g}_{hi:g}", "excl_change", "stake",
                       f"{c.upper()}: back ltp_pre {lo:g}-{hi:g} at BSP (ladder-change prints excluded)",
                       f"{c.upper()}: runners with ltp_pre in [{lo:g},{hi:g}) not exactly on 2/3/4/6/10/20/30/50/100, "
                       f"{UNIV}; back at BSP"))
        sp.append(("bsp_band_tick", c, "lay_band_ge50", "excl_change", "liability",
                   f"{c.upper()}: lay ltp_pre >= 50 at BSP per liability (ladder-change prints excluded)",
                   f"{c.upper()}: runners with ltp_pre >= 50 not on 50/100, {UNIV}; lay at BSP, 1 unit liability"))
    for c in ("gb", "ie"):
        mm = (f"{c.upper()}: walk-forward horse-memory residual logit (selection_id history over GB+IE, runs with "
              f"an earlier start only) at ltp_pre in {UNIV}")
        sp += [("memory_logit", c, "memory_logit_back_ev_gt_0", "all", "stake", f"{c.upper()} horse memory logit: "
                "back when EV after 8% > 0", f"{mm}; back at BSP if model EV at ltp_pre after 8% > 0"),
               ("memory_logit", c, "memory_logit_lay_ev_gt_0", "all", "liability", f"{c.upper()} horse memory "
                "logit: lay when lay EV after 8% > 0", f"{mm}; lay at BSP per liability if lay EV > 0"),
               ("memory_rule", c, "backup7_back", "all", "stake", f"{c.upper()}: back horses backing up within 7 "
                "days at BSP", f"{c.upper()}: previous run (GB/IE) started <= 7 days earlier, {UNIV}; back at BSP"),
               ("memory_rule", c, "cum_resid_top_decile_back", "all", "stake", f"{c.upper()}: back the top decile "
                "of shrunk cumulative residual at BSP", f"{c.upper()}: cum_resid = sum(f - p_run)/(n_prior+5) over "
                f"earlier runs above the 90th percentile of earlier months, {UNIV}; back at BSP"),
               ("memory_rule", c, "cum_resid_bottom_decile_lay", "all", "liability", f"{c.upper()}: lay the bottom "
                "decile of cumulative residual at BSP", f"{c.upper()}: cum_resid below the 10th percentile of earlier "
                f"months, {UNIV}; lay at BSP per liability"),
               ("memory_rule", c, "beaten_fav_back", "all", "stake", f"{c.upper()}: back last-start beaten "
                "favourites at BSP", f"{c.upper()}: previous run favourite (by p_run) and lost, {UNIV}; back at BSP")]
    return sp


def find(family, country, rule, segment, staking):
    for x in REG:
        if (x["kind"] == "roi" and x["family"] == family and x["country"] == country and x["rule"] == rule
                and x["segment"] == segment and x["staking"] == staking):
            return x
    return None


def claim_of(x: dict, claim: str, bet_rule: str, extra_note: str = "") -> dict:
    lay = x["side"] == "lay"
    comp = find(x["family"], x["country"], x["rule"], x["segment"], "stake") if lay and x["staking"] == "liability" else None
    notes = (f"fires {x.get('fires_per_race')} per race, {x.get('bets_per_month')} bets/month; MDE(80%) "
             f"{x.get('mde_80pct')} overall, {x.get('mde_80pct_each_half')} per half; halves bets "
             f"{x.get('halves_bets')} winners {x.get('halves_winners')} split {x.get('mid')}; median price "
             f"{x.get('median_price')}; G4 {x.get('passes_G4')}; BH(within experiment) {x.get('bh_within_experiment')}")
    if comp:
        notes += (f"; per backer stake: roi8 {comp['roi_8pct']} se {comp['se_clustered']} halves "
                  f"{comp['halves_roi_8pct']} G4 {comp['passes_G4']}")
    if extra_note:
        notes += "; " + extra_note
    return {"claim": claim, "bet_rule": bet_rule, "decision_time": "pre_off",
            "settlement": "per_liability_bsp" if lay and x["staking"] == "liability" else "bsp",
            "bets": x["bets"], "roi_8pct": x["roi_8pct"], "se_clustered": x["se_clustered"],
            "halves_roi_8pct": x["halves_roi_8pct"], "halves_se": x["halves_se"],
            "last12m_roi_8pct": x["last12m_roi_8pct"], "roi_0_5_10pct": x["roi_0_5_10pct"], "z": x["z"],
            "passes_all_gates": bool(x["passes_G4"]), "notes": notes,
            "key": [x["family"], x["country"], x["rule"], x["segment"], x["side"], x["staking"]]}


def build_claims() -> list[dict]:
    claims, seen = [], set()
    for fam, c, rule, seg, stk, claim, br in primary_specs():
        x = find(fam, c, rule, seg, stk)
        if x is None:
            continue
        claims.append(claim_of(x, claim, br))
        seen.add(id(x))
    # post-hoc: the best scan cell per country (selected on the full sample) and every G4 pass
    for c in COUNTRIES:
        sc = [x for x in REG if x["kind"] == "roi" and x["family"] == "scan" and x["country"] == c
              and x["z"] is not None and np.isfinite(x["z"])]
        if sc:
            b = max(sc, key=lambda x: x["z"])
            if id(b) not in seen:
                claims.append(claim_of(b, f"{c.upper()} scan, best cell by z (post hoc): {b['side']} {b['rule']} in "
                                       f"{b['segment']} ({b['staking']})", f"{c.upper()}: {b['rule']} selection at "
                                       f"ltp_pre, segment {b['segment']}, {UNIV}; {b['side']} at BSP",
                                       "selected as the max-z cell of the scan"))
                seen.add(id(b))
    for x in REG:
        if x["kind"] == "roi" and x.get("passes_G4") and id(x) not in seen:
            claims.append(claim_of(x, f"{x['country'].upper()} {x['family']}: {x['side']} {x['rule']} in "
                                   f"{x['segment']} ({x['staking']}) passes G4", f"{x['family']} rule {x['rule']}, "
                                   f"segment {x['segment']}, {UNIV}; {x['side']} at BSP", "found by the scan"))
            seen.add(id(x))
    return claims


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "data/raw/betfair/split")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--parts", default="1,2,3,4,5")
    a = ap.parse_args()
    parts = set(a.parts.split(","))
    t0 = time.time()
    # sanity checks: vectorised Shin = repo Shin; settle = ponies.betfair.net_return
    tst = np.array([2.5, 3.2, 6.0, 11.0, 4.4])
    assert np.allclose(shin_vec(tst, np.zeros(5, int), 1), shin_probabilities(tst), atol=1e-9)
    assert abs(settle("back", "stake", np.array([5.0]), np.array([1.0]), 0.08)[0] - net_return(5.0, True, 0.08)) < 1e-12
    assert abs(settle("back", "stake", np.array([5.0]), np.array([0.0]), 0.08)[0] - net_return(5.0, False, 0.08)) < 1e-12
    assert on_ladder_change(np.array([6.0, 6.2, 100.0, 2.02])).tolist() == [True, False, True, False]

    W, load = {}, {}
    for c in COUNTRIES:
        W[c], load[c] = load_country(c, a.dir, a.frac, a.seed)
    out({"section": "data", "load": load, "seconds": round(time.time() - t0, 1)})
    results = {"load": load}

    if "1" in parts:
        for c in COUNTRIES:
            r = part1(W[c], c)
            results[f"p1_{c}"] = r
            out(r)
    if "3" in parts or "5" in parts:
        for c in COUNTRIES:
            r3, E3, U3 = part3(W[c], c)
            results[f"p3_{c}"] = r3
            out({k: v for k, v in r3.items()})
            if "5" in parts:
                r5 = part5(E3, U3, c)
                results[f"p5_{c}"] = r5
                out(r5)
            del E3, U3
    if "2" in parts:
        for i, c in enumerate(COUNTRIES):
            r2, chk = part2(W[c], c, fit_check=(c == "ie"))
            if chk:
                out(chk)
                results["fit_check"] = chk
            results[f"p2_{c}"] = r2
            out(r2)
    if "4" in parts:
        h, hinfo = memory_history([W["gb"], W["ie"]])
        out(hinfo)
        results["p4_history"] = hinfo
        for c in ("gb", "ie"):
            r4_ = part4(h, c)
            results[f"p4_{c}"] = r4_
            out(r4_)
        del h

    # ---- G5 count and BH inside this experiment
    p = np.array([one_sided_p(x["z"]) for x in REG])
    rej = bh(p, 0.05)
    for x, rj in zip(REG, rej):
        x["bh_within_experiment"] = bool(rj)
    fam = pd.Series([x["family"] for x in REG]).value_counts().to_dict()
    kinds = pd.Series([x["kind"] for x in REG]).value_counts().to_dict()
    passing = [x for x in REG if x.get("passes_G4")]
    g5 = {"section": "g5_count", "cells_total": len(REG), "by_family": fam, "by_kind": kinds,
          "bh_q05_survivors": [{k: x[k] for k in ("family", "country", "rule", "segment", "side", "staking", "z")
                                if k in x} for x in REG if x["bh_within_experiment"]],
          "passing_G4": [{k: x.get(k) for k in ("family", "country", "rule", "segment", "side", "staking", "bets",
                                                 "roi_8pct", "se_clustered", "z", "halves_roi_8pct", "halves_se",
                                                 "last12m_roi_8pct", "bh_within_experiment")} for x in passing],
          "bonferroni_z_for_cells": r4(_z_of_p(0.05 / max(len(REG), 1)))}
    out(g5)
    results["g5"] = g5
    claims = build_claims()
    for cl in claims:
        out({"section": "claim", **cl})
    results["claims"] = claims
    results["reg"] = REG
    if a.json:
        a.json.write_text(json.dumps(results, default=_jd))
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


def _z_of_p(p: float) -> float:
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if one_sided_p(mid) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    raise SystemExit(main())

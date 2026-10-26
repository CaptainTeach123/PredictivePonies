"""Experiment "card_scan" (round 2): race-card segments, concurrency and intra-meeting
information on AU (gallops, harness), GB and IE WIN markets, settled at Betfair SP.

    python scripts/bf2_card_scan.py [--dev] [--no-model] [--workers 3] [--cells-csv PATH]

Data: data/raw/betfair/split/{au,gb,ie}_win.parquet (+ nz/za/fr/us WIN start times for the
concurrency count). Licensed: prints JSON summary lines only, never horse names or rows.

UNIVERSE (G1/G6)  WIN markets deduplicated on (market_id, selection_id); settled (won known for
every runner, >= 1 winner); every runner traded pre-off (ltp_pre > 1); ltp_pre book
sum(1/ltp_pre) in [0.97, 1.03] (G1; this also removes every G6 market: saddlecloth gap AND book <
0.97); >= 2 runners. Populations: au_gallops, au_harness (market_name contains Pace|Trot), gb, ie.

SELECTIONS (decided pre-off from ltp_pre)  the favourite (lowest ltp_pre; ties -> lower
saddlecloth) and the round-1 ltp_pre bands 1-2, 2-3, 3-4.5, 4.5-6.5, 6.5-10, 10-15, 15-25, 25-50,
50+ ; AU harness gate cells also use "all runners". Each selection is backed with 1 unit at BSP,
and laid at BSP (per unit of backer's stake and per unit of liability). Runners without a BSP are
void (no bet). Dead heats: stake share f = dead_heat_share settled as a winner.

SEGMENTS (all known before the off)
  class        parsed from market_name (rules in section parse_rules):
                 AU gallops (first match wins): Grp1 'Grp ?1'; Grp2; Grp3; Listed; Jumps 'Hrd|Stpl';
                 Mdn; CL1; CL2; CL3; CL4-6 ('C[Ll]?n'); 2yo; 3yo/4yo; Hcap ('Hcap|Hcp' = all
                 handicaps incl. benchmark races: the archive has no 'BM' token); Cup; Other (Qlty, WFA,
                 unparsed).
                 AU harness: gait x start = Pace M / Pace S / Trot M / Trot S (no class in the names).
                 GB/IE: jumps if 'Hrd|Chs|Hurdle|Chase|NHF|INHF|XC|Hunt|Stpl', else flat. Jumps:
                 J Grd1; J Grd2-3/Listed; J NHF (bumper); J Hunt/XC; J Hcap Chs; J Hcap Hrd;
                 J Nov/Beg/Mdn Chs; J Nov/Mdn/Juv Hrd; J Other. Flat: F Grp1; F Grp2-3; F Listed;
                 F Nursery; F Hcap (incl. App Hcap); F Mdn; F Nov; F Claim/Sell; F Other (Stks, Cond,
                 Class Stks, PA, unparsed).
  distance     metres: AU from '(\\d{3,4})m'; GB/IE from 'XmYf' (1 f = 201.168 m, 1 mile = 1609.344 m).
                 AU gallops <=1100, 1101-1300, 1301-1600, 1601-2000, 2001-2400, 2401+;
                 AU harness <1800, 1800-2199, 2200-2599, 2600+;
                 GB/IE <1350 (5-6f), 1350-1699 (7f-1m), 1700-2299, 2300-3149, 3150-3899, 3900-4699, 4700+.
  race_no      AU: 'R(\\d+)' in market_name (fallback: rank of start in the meeting); GB/IE: rank of
                 the start among the meeting's WIN markets in the archive. Bands 1, 2-3, 4-5, 6-7, 8+.
  last_race    race_no == the meeting's maximum.
  bigday_*     local race day (Australia/Sydney for AU, Europe/London for GB/IE; the meeting's first
                 start). bigday_markets: the region's WIN-market count that day (AU markets; GB+IE
                 markets) >= the 98th percentile ('higher') of strictly earlier days (first 60 days: of
                 those 60 days; a calendar count, no outcomes); bigday_group: the same with the count
                 of Group 1-3 / Grade 1-3 races (AU Grp1-3; GB/IE Grp1-3 + Grd1-3), and count >= 1.
  concurrency  number of OTHER WIN markets in this archive (au gb ie nz za fr us) with scheduled start
                 within +/- 5 minutes: 0, 1, 2, 3, 4+.
  fav_form     intra-meeting: over races at the same meeting (event_id + venue) that started >= 15 min
                 earlier (pre-off decisions; every race gap in the archive is >= ~25 min, so these are
                 settled) and pass G1: F = sum(fav won - Shin p of the fav). Levels: none, <= -1,
                 (-1, 0), [0, 1), >= 1.
  prev_fav     the most recent such earlier race's favourite: none / won / lost.
  gate_resid   AU harness only, runner level, mobile starts ('Pace M'/'Trot M'): gate group of the
                 saddlecloth (1-3, 4-6, 7+); over earlier G1 mobile races at the meeting R_g =
                 sum(winner in g - sum of Shin p of g's runners); the runner's own group's R_g.
                 Levels: standing start, no info (< 2 earlier mobile races), <= -0.75, (-0.75, 0.75),
                 >= 0.75.

STATISTICS  ROI per unit (stake or liability) at commission 0/5/8/10% on net winnings (8% primary);
SE = max(iid, clustered by market, clustered by local race day) (CR1); halves split at the
population's median race date; last 12 months 2019-11-08..2020-11-08; z = roi8 / se; one-sided p;
MDE (80% power at the 2 SE bar) = 2.84 SE. G4: roi8 > 2 SE in both halves and last-12 roi8 > 0.
Every cell (population x selection x side x segment x level) is counted; cells with < 30 bets get p =
1. Benjamini-Hochberg (q = 0.05) within this experiment's family.

T30 REPLICATION (claims only)  the same rule decided at ltp_t30 (fav/band from ltp_t30; markets with
every runner traded at t30, t30 book in [0.95, 1.10], G6 drop; intra-meeting features from races that
started >= 45 min earlier) settled (a) by a limit order one tick worse than ltp_t30 (back: one tick
shorter; lay: one tick longer) filled if the later ltp_pre or the BSP reaches it, else BSP, and (b) at
BSP.

PART B (probability, G7)  per population a conditional logit on qualifying races: lp = log Shin p at
ltp_pre (unpenalised), fav indicator, lp x level and fav x level for every non-reference level of the
static card segments (class, distance, race_no, last_race, bigday_*, concurrency), fav x F (clipped
+/-2), fav x prev-fav won/lost, and for AU harness mobile gate-group dummies (4-6, 7+) and the own
group's R_g (clipped +/-2). Walk-forward: refit monthly on all races starting before the month
(L-BFGS, ConditionalLogit objective), l2 for month m = lowest OOS per-race log loss over months
m-6..m-1 on the grid 1e-4..1000 (15 values, x3 steps); evaluation 2017-12..2020-11. Log-loss gain vs Shin at
ltp_pre (same timestamp) and vs a market-only recalibration (lp alone). Bets: back at BSP when
p_model / p_mkt - 1 > k, lay when < -k, k in {0.05, 0.10, 0.20}.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ponies.odds import shin_probabilities  # noqa: E402

DATA = REPO / "data" / "raw" / "betfair" / "split"
WIN_COLS = ["market_id", "event_id", "venue", "market_name", "start", "date", "selection_id", "program",
            "ltp_pre", "ltp_t30", "bsp", "won", "dead_heat_share", "market_type"]
ALL_WIN = ("au", "gb", "ie", "nz", "za", "fr", "us")
POPS = ("au_gallops", "au_harness", "gb", "ie")
HARNESS_RE = r"\b(?:Pace|Trot)\b"
MOBILE_RE = r"\b(?:Pace|Trot)\s+M\s*$"
COMMS = (0.0, 0.05, 0.08, 0.10)
C8 = 0.08
BOOK_LO, BOOK_HI = 0.97, 1.03
T30_BOOK = (0.95, 1.10)
LAST12_FROM = pd.Timestamp("2019-11-08")
LAST12_TO = pd.Timestamp("2020-11-08")
MIN_BETS_TEST = 30
FDR = 0.05
Z80 = 0.8416
CUT_PRE = 15      # minutes: earlier races used for pre-off decisions
CUT_T30 = 45      # minutes: earlier races used for t30 decisions
CONC_WIN = 5      # +/- minutes
BIG_Q = 0.98
BIG_BURN_DAYS = 60
BANDS = [1.0, 2.0, 3.0, 4.5, 6.5, 10.0, 15.0, 25.0, 50.0, np.inf]
BAND_NAMES = ["1-2", "2-3", "3-4.5", "4.5-6.5", "6.5-10", "10-15", "15-25", "25-50", "50+"]
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1000.0, 10.0)]
L2_GRID = (1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
L2_WINDOW = 6
OOS_FIRST = pd.Period("2017-06", "M")
EVAL_FIRST = pd.Period("2017-12", "M")
K_MODEL = (0.05, 0.10, 0.20)
FAV_RESID_LEVELS = ["none", "<=-1", "(-1,0)", "[0,1)", ">=1"]
GATE_LEVELS = ["standing start", "no info", "<=-0.75", "(-0.75,0.75)", ">=0.75"]

DIMS_ALL = ["all", "class", "distance", "race_no", "last_race", "bigday_markets", "bigday_group",
            "concurrency", "fav_form", "prev_fav"]
STATIC_DIMS = ["class", "distance", "race_no", "last_race", "bigday_markets", "bigday_group", "concurrency"]
INTRA_DIMS = ["fav_form", "prev_fav", "gate_resid"]

CLASS_ORDER = {
    "au_gallops": ["Mdn", "Hcap", "CL1", "CL2", "CL3", "CL4-6", "2yo", "3yo/4yo", "Listed", "Grp3", "Grp2", "Grp1",
                   "Cup", "Jumps", "Other"],
    "au_harness": ["Pace M", "Pace S", "Trot M", "Trot S", "Other"],
    "gb": ["F Grp1", "F Grp2-3", "F Listed", "F Hcap", "F Nursery", "F Mdn", "F Nov", "F Claim/Sell", "F Other",
           "J Grd1", "J Grd2-3/Listed", "J Hcap Hrd", "J Hcap Chs", "J Nov/Mdn/Juv Hrd", "J Nov/Beg/Mdn Chs", "J NHF",
           "J Hunt/XC", "J Other"],
}
CLASS_ORDER["ie"] = CLASS_ORDER["gb"]
DIST_EDGES = {
    "au_gallops": ([0, 1101, 1301, 1601, 2001, 2401, 99999], ["<=1100", "1101-1300", "1301-1600", "1601-2000",
                                                             "2001-2400", "2401+"]),
    "au_harness": ([0, 1800, 2200, 2600, 99999], ["<1800", "1800-2199", "2200-2599", "2600+"]),
    "gb": ([0, 1350, 1700, 2300, 3150, 3900, 4700, 99999], ["<1350", "1350-1699", "1700-2299", "2300-3149",
                                                            "3150-3899", "3900-4699", "4700+"]),
}
DIST_EDGES["ie"] = DIST_EDGES["gb"]
RACE_NO_LEVELS = ["1", "2-3", "4-5", "6-7", "8+"]
CONC_LEVELS = ["0", "1", "2", "3", "4+"]

PARSE_RULES = {
    "au_gallops": "first match wins: Grp1 r'\\bGrp\\s?1\\b'; Grp2; Grp3; Listed r'\\bListed\\b'; Jumps r'\\b(Hrd|Stpl)\\b'; "
                  "Mdn; CL1..CL3 r'\\bC[Ll]?\\s?n\\b'; CL4-6; 2yo; 3yo/4yo r'\\b[34]yo\\b'; Hcap r'\\b(Hcap|Hcp)\\b' "
                  "(all handicaps incl. benchmark: no 'BM' token exists); Cup; else Other (Qlty, WFA, unparsed)",
    "au_harness": "gait r'\\b(Pace|Trot)\\b' x start ' M' (mobile) / ' S' (standing) suffix; no class in names",
    "gb_ie": "jumps if r'\\b(Hrd|Chs|Hurdle|Chase|NHF|INHF|XC|Hunt|Stpl)\\b'. Jumps (first match): J Grd1 "
             "r'\\b(Grd|Grade|G)\\s?1\\b'; J Grd2-3/Listed r'\\b(Grd|G)\\s?[23]\\b|\\bList(ed)?\\b'; J NHF r'\\bI?NHF\\b'; "
             "J Hunt/XC r'\\b(Hunt|XC)\\b'; J Hcap Chs (Hcap & Chs); J Hcap Hrd (Hcap & Hrd); J Nov/Beg/Mdn Chs; "
             "J Nov/Mdn/Juv Hrd; else J Other. Flat (first match): F Grp1 r'\\b(Grp|G)\\s?1\\b'; F Grp2-3; F Listed "
             "r'\\bList(ed)?\\b'; F Nursery; F Hcap r'\\b(Hcap|Hcp)\\b'; F Mdn; F Nov; F Claim/Sell "
             "r'\\b(Claim|Sell|Selling|Auc)\\b'; else F Other",
    "distance": "AU r'(\\d{3,4})\\s*m\\b'; GB/IE leading r'^(\\d+)m(\\d+)f' / '^(\\d+)m' / '^(\\d+)f' (1f = 201.168 m)",
}

CAVEATS = [
    "Saddlecloth is used as the AU harness gate (groups 1-3, 4-6, 7+ incl. the second row); scratchings move "
    "front-row runners in and some venues do not number by barrier, so the gate residual is noisy.",
    "Race number for GB/IE is the rank of the start among the meeting's WIN markets present in this archive; a "
    "missing market shifts later numbers; 'last race' is the last one in the archive.",
    "Big-day thresholds are walk-forward percentiles of a calendar count (no outcomes); the first 60 days use "
    "their own 98th percentile.",
    "BSP bets are assumed placed just before the off on the ltp_pre information; ltp_pre is a one-minute "
    "snapshot that can be a minute stale.",
    "Runners without a BSP are void; G1 removes the stale-quote markets (book < 0.97) that produced every "
    "round-1 ltp_pre 'edge'.",
]


# ----------------------------------------------------------------------------- utilities


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, pd.Period)):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def out(obj) -> None:
    print(json.dumps(obj, default=_json_default), flush=True)


def r5(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, 5)


def one_sided_p(z) -> float:
    if z is None or not np.isfinite(z):
        return 1.0
    return float(0.5 * math.erfc(z / math.sqrt(2)))


def bh_reject(p: np.ndarray, q: float = FDR) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    rej = np.zeros(m, bool)
    if m == 0:
        return rej
    order = np.argsort(p)
    thr = q * np.arange(1, m + 1) / m
    below = np.flatnonzero(p[order] <= thr)
    if len(below):
        rej[order[: below.max() + 1]] = True
    return rej


def cl_se(x: np.ndarray, codes: np.ndarray) -> float:
    """CR1 clustered SE of the mean; codes must be non-decreasing (clusters contiguous)."""
    n = len(x)
    if n < 2:
        return float("nan")
    b = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    G = len(b)
    if G < 2:
        return float("nan")
    s = np.add.reduceat(x - x.mean(), b)
    return float(math.sqrt((s ** 2).sum() * G / (G - 1)) / n)


def tick_shorter(price) -> np.ndarray:
    p = np.round(np.asarray(price, dtype=float), 2)
    step = np.full_like(p, 10.0)
    for lo, hi, inc in TICKS:
        step = np.where((p > lo + 1e-9) & (p <= hi + 1e-9), inc, step)
    return np.maximum(np.round(p - step, 2), 1.01)


def tick_longer(price) -> np.ndarray:
    p = np.round(np.asarray(price, dtype=float), 2)
    step = np.full_like(p, 10.0)
    for lo, hi, inc in reversed(TICKS):
        step = np.where((p >= lo - 1e-9) & (p < hi - 1e-9), inc, step)
    return np.minimum(np.round(p + step, 2), 1000.0)


def shin_vec(inv: np.ndarray, codes: np.ndarray, n_m: int) -> np.ndarray:
    """ponies.odds.shin_probabilities per market, vectorised (bisection on z)."""
    S = np.bincount(codes, weights=inv, minlength=n_m)
    Sr = S[codes]
    lo = np.zeros(n_m)
    hi = np.full(n_m, 0.5)

    def probs(zm):
        z = zm[codes]
        return (np.sqrt(z * z + 4 * (1 - z) * inv * inv / Sr) - z) / (2 * (1 - z))

    for _ in range(60):
        mid = (lo + hi) / 2
        tot = np.bincount(codes, weights=probs(mid), minlength=n_m)
        gt = tot > 1
        lo = np.where(gt, mid, lo)
        hi = np.where(gt, hi, mid)
    p = probs((lo + hi) / 2)
    p = p / np.bincount(codes, weights=p, minlength=n_m)[codes]
    flat = Sr <= 1.0
    p[flat] = inv[flat] / Sr[flat]
    return p


def back_net(price, f, c):
    g = f * (price - 1.0) - (1.0 - f)
    return np.where(g > 0, g * (1.0 - c), g)


def lay_net(price, f, c):
    g = -(f * (price - 1.0) - (1.0 - f))
    return np.where(g > 0, g * (1.0 - c), g)


# ----------------------------------------------------------------------------- parsing


def parse_class(name: pd.Series, pop: pd.Series) -> pd.Series:
    s = name.fillna("")
    cls = pd.Series("Other", index=s.index, dtype=object)

    def has(rx):
        return s.str.contains(rx, regex=True)

    # AU gallops
    au_g = (pop == "au_gallops").to_numpy()
    rules = [("Grp1", r"\bGrp\s?1\b"), ("Grp2", r"\bGrp\s?2\b"), ("Grp3", r"\bGrp\s?3\b"), ("Listed", r"\bListed\b"),
             ("Jumps", r"\b(?:Hrd|Stpl)\b"), ("Mdn", r"\bMdn\b"), ("CL1", r"\bC[Ll]?\s?1\b"), ("CL2", r"\bC[Ll]?\s?2\b"),
             ("CL3", r"\bC[Ll]?\s?3\b"), ("CL4-6", r"\bC[Ll]?\s?[456]\b"), ("2yo", r"\b2yo\b"), ("3yo/4yo", r"\b[34]yo\b"),
             ("Hcap", r"\b(?:Hcap|Hcp)\b"), ("Cup", r"\bCup\b")]
    done = np.zeros(len(s), bool)
    for lab, rx in rules:
        m = au_g & ~done & has(rx).to_numpy()
        cls[m] = lab
        done |= m
    # AU harness
    au_h = (pop == "au_harness").to_numpy()
    gait = np.where(has(r"\bPace\b"), "Pace", np.where(has(r"\bTrot\b"), "Trot", ""))
    start = np.where(has(r"\b(?:Pace|Trot)\s+M\s*$"), " M", np.where(has(r"\b(?:Pace|Trot)\s+S\s*$"), " S", ""))
    hl = pd.Series(np.char.add(gait.astype(str), start.astype(str)), index=s.index)
    hl = hl.where(hl.isin(["Pace M", "Pace S", "Trot M", "Trot S"]), "Other")
    cls[au_h] = hl[au_h]
    # GB / IE
    gi = pop.isin(["gb", "ie"]).to_numpy()
    jumps = has(r"\b(?:Hrd|Chs|Hurdle|Chase|NHF|INHF|XC|Hunt|Stpl)\b").to_numpy()
    chs = has(r"\b(?:Chs|Chase)\b").to_numpy()
    hrd = has(r"\b(?:Hrd|Hurdle)\b").to_numpy()
    hcap = has(r"\b(?:Hcap|Hcp)\b").to_numpy()
    jr = [("J Grd1", has(r"\b(?:Grd|Grade|G)\s?1\b").to_numpy()),
          ("J Grd2-3/Listed", has(r"\b(?:Grd|Grade|G)\s?[23]\b|\bList(?:ed)?\b").to_numpy()),
          ("J NHF", has(r"\bI?NHF\b").to_numpy()),
          ("J Hunt/XC", has(r"\b(?:Hunt|XC)\b").to_numpy()),
          ("J Hcap Chs", hcap & chs),
          ("J Hcap Hrd", hcap & hrd),
          ("J Nov/Beg/Mdn Chs", has(r"\b(?:Nov|Beg|Mdn)\b").to_numpy() & chs),
          ("J Nov/Mdn/Juv Hrd", has(r"\b(?:Nov|Mdn|Juv)\b").to_numpy() & hrd)]
    fr = [("F Grp1", has(r"\b(?:Grp|G)\s?1\b").to_numpy()),
          ("F Grp2-3", has(r"\b(?:Grp|G)\s?[23]\b").to_numpy()),
          ("F Listed", has(r"\bList(?:ed)?\b").to_numpy()),
          ("F Nursery", has(r"\bNursery\b").to_numpy()),
          ("F Hcap", hcap),
          ("F Mdn", has(r"\bMdn\b").to_numpy()),
          ("F Nov", has(r"\bNov\b").to_numpy()),
          ("F Claim/Sell", has(r"\b(?:Claim|Sell|Selling|Auc)\b").to_numpy())]
    done = np.zeros(len(s), bool)
    for lab, m0 in jr:
        m = gi & jumps & ~done & m0
        cls[m] = lab
        done |= m
    cls[gi & jumps & ~done] = "J Other"
    done = np.zeros(len(s), bool)
    for lab, m0 in fr:
        m = gi & ~jumps & ~done & m0
        cls[m] = lab
        done |= m
    cls[gi & ~jumps & ~done] = "F Other"
    return cls


def is_group_race(cls: pd.Series) -> np.ndarray:
    return cls.isin(["Grp1", "Grp2", "Grp3", "F Grp1", "F Grp2-3", "J Grd1"]).to_numpy() | (
        (cls == "J Grd2-3/Listed").to_numpy())


def parse_distance(name: pd.Series, cc: pd.Series) -> np.ndarray:
    s = name.fillna("")
    d = np.full(len(s), np.nan)
    au = (cc == "au").to_numpy()
    m = s.str.extract(r"(\d{3,4})\s*m\b")[0].astype(float).to_numpy()
    d[au] = m[au]
    gi = ~au
    x = s.str.extract(r"^\s*(?:(\d+)m)?(?:(\d+)f)?\b")
    miles = x[0].astype(float).to_numpy()
    fur = x[1].astype(float).to_numpy()
    has = ~np.isnan(miles) | ~np.isnan(fur)
    dm = np.nan_to_num(miles) * 1609.344 + np.nan_to_num(fur) * 201.168
    metres = s.str.extract(r"^\s*(\d{3,4})m\b")[0].astype(float).to_numpy()
    dm = np.where(~np.isnan(metres), metres, np.where(has, dm, np.nan))
    d[gi] = dm[gi]
    return d


# ----------------------------------------------------------------------------- loading


def load(dev: bool, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    t0 = time.time()
    frames = []
    info = {"section": "data"}
    for cc in ("au", "gb", "ie"):
        w = pd.read_parquet(DATA / f"{cc}_win.parquet", columns=WIN_COLS)
        n0 = len(w)
        w = w.drop_duplicates(["market_id", "selection_id"])
        w = w[w["market_type"] == "WIN"].drop(columns="market_type")
        w["cc"] = cc
        info[f"{cc}_rows_raw"] = int(n0)
        info[f"{cc}_rows_dedup_win"] = int(len(w))
        frames.append(w)
    w = pd.concat(frames, ignore_index=True)
    del frames
    for c in ("ltp_pre", "ltp_t30", "bsp"):
        w.loc[~(w[c] > 1), c] = np.nan
    w["pn"] = pd.to_numeric(w["program"], errors="coerce")
    w = w.drop(columns=["program"])
    w["won_na"] = w["won"].isna()
    w["f"] = np.where(w["won"] == 1, w["dead_heat_share"].fillna(1.0), 0.0)
    w["inv_pre"] = 1.0 / w["ltp_pre"]
    w["inv_t30"] = 1.0 / w["ltp_t30"]

    # ---------------- market table (all WIN markets of au/gb/ie)
    g = w.groupby("market_id", sort=False)
    mk = g.agg(cc=("cc", "first"), event_id=("event_id", "first"), venue=("venue", "first"),
               name=("market_name", "first"), start=("start", "first"), date=("date", "first"),
               n=("selection_id", "size"), n_pre=("ltp_pre", "count"), n_t30=("ltp_t30", "count"),
               book_pre=("inv_pre", "sum"), book_t30=("inv_t30", "sum"), won_na=("won_na", "max"),
               n_won=("won", "sum"), pn_cnt=("pn", "count"), pn_max=("pn", "max"), pn_nu=("pn", "nunique"))
    mk = mk.reset_index()
    harness = mk["name"].fillna("").str.contains(HARNESS_RE, regex=True)
    mk["pop"] = np.where(mk["cc"] == "au", np.where(harness, "au_harness", "au_gallops"), mk["cc"])
    mk["mobile"] = (mk["pop"] == "au_harness") & mk["name"].fillna("").str.contains(MOBILE_RE, regex=True)
    mk["settled"] = (~mk["won_na"]) & (mk["n_won"] >= 1)
    mk["gap"] = (mk["pn_cnt"] == mk["n"]) & ((mk["pn_max"] > mk["n"]) | (mk["pn_nu"] < mk["n"]))
    mk["all_pre"] = mk["n_pre"] == mk["n"]
    mk["g6_drop"] = mk["gap"] & (mk["book_pre"] < BOOK_LO)
    mk["g1"] = (mk["settled"] & mk["all_pre"] & (mk["book_pre"] >= BOOK_LO) & (mk["book_pre"] <= BOOK_HI)
                & (mk["n"] >= 2) & ~mk["g6_drop"])
    mk["t30_ok"] = (mk["settled"] & (mk["n_t30"] == mk["n"]) & (mk["book_t30"] >= T30_BOOK[0])
                    & (mk["book_t30"] <= T30_BOOK[1]) & (mk["n"] >= 2) & ~mk["g6_drop"])
    mk["cls"] = parse_class(mk["name"], mk["pop"])
    mk["group_race"] = is_group_race(mk["cls"])
    mk["dist_m"] = parse_distance(mk["name"], mk["cc"])

    # meeting = event_id + venue; race number; local race day
    mk["meet"] = pd.factorize(mk["event_id"].astype(str) + "|" + mk["venue"].fillna("").astype(str))[0]
    mk = mk.sort_values(["meet", "start", "market_id"]).reset_index(drop=True)
    rank = mk.groupby("meet").cumcount() + 1
    rno = mk["name"].fillna("").str.extract(r"^R(\d+)\b")[0].astype(float)
    mk["race_no"] = np.where((mk["cc"] == "au") & rno.notna(), rno, rank).astype(int)
    mk["last_race"] = mk["race_no"] == mk.groupby("meet")["race_no"].transform("max")
    first_start = mk.groupby("meet")["start"].transform("min")
    lday = pd.Series(pd.NaT, index=mk.index, dtype="datetime64[ns]")
    for cc, tz in (("au", "Australia/Sydney"), ("gb", "Europe/London"), ("ie", "Europe/London")):
        sel = mk["cc"] == cc
        lday[sel] = first_start[sel].dt.tz_convert(tz).dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    mk["lday"] = lday
    mk["region"] = np.where(mk["cc"] == "au", "AU", "GBIE")

    # big days (walk-forward 98th percentile of strictly earlier days, calendar counts only)
    big_info = {}
    mk["bigday_markets"] = False
    mk["bigday_group"] = False
    for reg in ("AU", "GBIE"):
        s = mk[mk["region"] == reg]
        dc = s.groupby("lday").agg(n=("market_id", "size"), gcount=("group_race", "sum")).sort_index()
        flags = {}
        for col in ("n", "gcount"):
            c = dc[col].to_numpy().astype(float)
            thr = np.empty(len(c))
            burn = np.quantile(c[:BIG_BURN_DAYS], BIG_Q, method="higher")
            for i in range(len(c)):
                thr[i] = burn if i < BIG_BURN_DAYS else np.quantile(c[:i], BIG_Q, method="higher")
            fl = (c >= thr) & (c >= 1)
            flags[col] = pd.Series(fl, index=dc.index)
            big_info[f"{reg}_{col}"] = {"days": int(len(c)), "big_days": int(fl.sum()), "share": r5(fl.mean()),
                                        "median_threshold": r5(np.median(thr)), "max_count": int(c.max())}
        sel = mk["region"] == reg
        mk.loc[sel, "bigday_markets"] = mk.loc[sel, "lday"].map(flags["n"]).fillna(False).astype(bool).to_numpy()
        mk.loc[sel, "bigday_group"] = mk.loc[sel, "lday"].map(flags["gcount"]).fillna(False).astype(bool).to_numpy()
    info["big_days"] = big_info

    # concurrency over every WIN market in the archive
    st = []
    for cc in ALL_WIN:
        a = pd.read_parquet(DATA / f"{cc}_win.parquet", columns=["market_id", "market_type", "start"])
        a = a[a["market_type"] == "WIN"].drop_duplicates("market_id")
        st.append(a["start"].astype("int64").to_numpy() // 10 ** 6 // 60)
    allmin = np.sort(np.concatenate(st))
    info["concurrency_universe_markets"] = int(len(allmin))
    mm = mk["start"].astype("int64").to_numpy() // 10 ** 6 // 60
    mk["conc"] = (np.searchsorted(allmin, mm + CONC_WIN, "right") - np.searchsorted(allmin, mm - CONC_WIN, "left") - 1)

    # ---------------- dev subset: whole meetings
    if dev:
        meets = mk["meet"].drop_duplicates().sample(frac=0.15, random_state=seed)
        mk = mk[mk["meet"].isin(set(meets))].reset_index(drop=True)
        w = w[w["market_id"].isin(set(mk["market_id"]))]

    # ---------------- runner-level: Shin p at ltp_pre for G1 markets and at t30 for t30_ok markets
    w = w.merge(mk[["market_id", "meet", "start", "g1", "t30_ok", "mobile", "pop"]].rename(columns={"start": "_st"}),
                on="market_id", how="inner")
    w = w.sort_values(["_st", "market_id", "pn", "selection_id"]).reset_index(drop=True)
    w = w.drop(columns=["_st"])
    w["p_pre"] = np.nan
    w["p_t30"] = np.nan
    for col, ok_col, inv in (("p_pre", "g1", "inv_pre"), ("p_t30", "t30_ok", "inv_t30")):
        sel = w[ok_col].to_numpy()
        sub = w.loc[sel]
        codes, uniq = pd.factorize(sub["market_id"])
        w.loc[sel, col] = shin_vec(sub[inv].to_numpy(), codes, len(uniq))
    # check against ponies.odds.shin_probabilities on a few markets
    worst = 0.0
    for mid_ in w.loc[w["g1"], "market_id"].drop_duplicates().head(30):
        gg = w[w["market_id"] == mid_]
        worst = max(worst, float(np.abs(shin_probabilities(gg["ltp_pre"].to_numpy()) - gg["p_pre"].to_numpy()).max()))
    info["shin_vec_vs_ponies_max_abs_diff"] = r5(worst)
    # favourites (lowest price; ties -> lower saddlecloth, then selection_id; rows are sorted that way)
    for col, price in (("fav_pre", "ltp_pre"), ("fav_t30", "ltp_t30")):
        tmp = w[["market_id", price, "pn", "selection_id"]].copy()
        tmp["_o"] = np.arange(len(tmp))
        tmp = tmp.sort_values(["market_id", price, "pn", "selection_id"], na_position="last")
        first = ~tmp["market_id"].duplicated() & tmp[price].notna()
        fv = np.zeros(len(w), bool)
        fv[tmp.loc[first, "_o"].to_numpy()] = True
        w[col] = fv
    w["ggrp"] = np.where(w["mobile"] & w["pn"].notna(),
                         np.where(w["pn"] <= 3, 0, np.where(w["pn"] <= 6, 1, 2)), -1)

    # ---------------- intra-meeting market aggregates (earlier races must pass G1 / have t30 for the t30 version)
    agg = w[w["g1"]].copy()
    fav_rows = agg[agg["fav_pre"]][["market_id", "f", "p_pre"]].rename(columns={"f": "fav_won", "p_pre": "fav_p"})
    mk = mk.merge(fav_rows, on="market_id", how="left")
    for gi in range(3):
        a = agg[agg["mobile"] & (agg["ggrp"] == gi)].groupby("market_id").agg(**{f"gw{gi}": ("f", "sum"),
                                                                                 f"ge{gi}": ("p_pre", "sum")})
        mk = mk.merge(a.reset_index(), on="market_id", how="left")
        mk[f"gw{gi}"] = mk[f"gw{gi}"].fillna(0.0)
        mk[f"ge{gi}"] = mk[f"ge{gi}"].fillna(0.0)
    mk = mk.sort_values(["meet", "start", "market_id"]).reset_index(drop=True)
    info["load_seconds"] = round(time.time() - t0, 1)
    return w, mk, info


def intra_meeting(mk: pd.DataFrame, cutoff_min: int, suffix: str) -> pd.DataFrame:
    """Cumulative earlier-race statistics within each meeting for races started >= cutoff minutes before."""
    key = mk["meet"].to_numpy().astype(np.int64) * 10 ** 8 + (mk["start"].astype("int64").to_numpy() // 10 ** 6 // 60)
    assert np.all(np.diff(key) >= 0)
    n = len(mk)
    first_idx = mk.groupby("meet").cumcount().to_numpy()
    first_idx = np.arange(n) - first_idx  # index of the meeting's first market
    k = np.searchsorted(key, key - cutoff_min, "right") - 1  # last earlier market with start <= start - cutoff
    has_prev = k >= first_idx
    vfav = mk["g1"].to_numpy() & mk["fav_won"].notna().to_numpy()
    resid = np.where(vfav, mk["fav_won"].fillna(0).to_numpy() - mk["fav_p"].fillna(0).to_numpy(), 0.0)
    cs_r = np.r_[0.0, np.cumsum(resid)]
    cs_n = np.r_[0, np.cumsum(vfav.astype(int))]
    kk = np.where(has_prev, k, first_idx - 1)
    n_prev = cs_n[kk + 1] - cs_n[first_idx]
    F = cs_r[kk + 1] - cs_r[first_idx]
    # most recent earlier G1 race (by start) with a valid favourite
    last_valid = np.where(vfav, np.arange(n), -1)
    last_valid = np.maximum.accumulate(last_valid)
    lv = np.where(has_prev, last_valid[np.maximum(kk, 0)], -1)
    lv = np.where(lv >= first_idx, lv, -1)
    prev_won = np.where(lv >= 0, mk["fav_won"].fillna(0).to_numpy()[np.maximum(lv, 0)], np.nan)
    res = pd.DataFrame(index=mk.index)
    res[f"n_prev{suffix}"] = n_prev
    res[f"F{suffix}"] = np.where(n_prev > 0, F, np.nan)
    ff = np.where(n_prev == 0, "none", np.where(F <= -1, "<=-1", np.where(F < 0, "(-1,0)", np.where(F < 1, "[0,1)", ">=1"))))
    res[f"fav_form{suffix}"] = ff
    res[f"prev_fav{suffix}"] = np.where(lv < 0, "none", np.where(prev_won > 0, "won", "lost"))
    res[f"prev_won{suffix}"] = prev_won
    # harness gate residuals over earlier G1 mobile races
    vm = mk["g1"].to_numpy() & mk["mobile"].to_numpy()
    cs_m = np.r_[0, np.cumsum(vm.astype(int))]
    res[f"n_prev_mob{suffix}"] = cs_m[kk + 1] - cs_m[first_idx]
    for gi in range(3):
        rg = np.where(vm, mk[f"gw{gi}"].to_numpy() - mk[f"ge{gi}"].to_numpy(), 0.0)
        cs = np.r_[0.0, np.cumsum(rg)]
        res[f"R{gi}{suffix}"] = cs[kk + 1] - cs[first_idx]
    return res


def band_of(price: np.ndarray) -> np.ndarray:
    b = np.digitize(price, BANDS[1:-1], right=False)
    return np.where(np.isfinite(price), b, -1)


def level_arrays(mk: pd.DataFrame) -> pd.DataFrame:
    """Static segment labels per market."""
    lab = pd.DataFrame(index=mk.index)
    lab["all"] = "all"
    lab["class"] = mk["cls"].to_numpy()
    dist = np.full(len(mk), "unknown", dtype=object)
    for pop, (edges, names) in DIST_EDGES.items():
        sel = (mk["pop"] == pop).to_numpy()
        d = mk["dist_m"].to_numpy()
        idx = np.digitize(d, edges[1:-1], right=False)
        lv = np.array(names, dtype=object)[np.clip(idx, 0, len(names) - 1)]
        lv = np.where(np.isfinite(d), lv, "unknown")
        dist[sel] = lv[sel]
    lab["distance"] = dist
    rn = mk["race_no"].to_numpy()
    lab["race_no"] = np.where(rn <= 1, "1", np.where(rn <= 3, "2-3", np.where(rn <= 5, "4-5", np.where(rn <= 7, "6-7", "8+"))))
    lab["last_race"] = np.where(mk["last_race"], "last", "not last")
    lab["bigday_markets"] = np.where(mk["bigday_markets"], "big", "normal")
    lab["bigday_group"] = np.where(mk["bigday_group"], "big", "normal")
    c = mk["conc"].to_numpy()
    lab["concurrency"] = np.where(c >= 4, "4+", c.astype(str))
    return lab


def gate_level(mobile: np.ndarray, ggrp: np.ndarray, n_prev_mob: np.ndarray, R: np.ndarray) -> np.ndarray:
    lv = np.where(R <= -0.75, "<=-0.75", np.where(R < 0.75, "(-0.75,0.75)", ">=0.75")).astype(object)
    lv = np.where(n_prev_mob < 2, "no info", lv)
    lv = np.where(~mobile | (ggrp < 0), "standing start", lv)
    return lv


# ----------------------------------------------------------------------------- cell engine


class Engine:
    """Per-population return arrays; rows sorted by (local day, start, market) so market and day codes are monotone."""

    def __init__(self, r: pd.DataFrame, price: np.ndarray, mid: pd.Timestamp, n_markets: int, n_months: int,
                 n_days: int, p_dec: np.ndarray | None = None):
        self.f = r["f"].to_numpy(float)
        self.price = price
        self.ok = np.isfinite(price) & (price > 1)
        self.mcode = pd.factorize(r["market_id"])[0]
        self.dcode = pd.factorize(r["lday"])[0]
        d = r["date"].to_numpy()
        self.first = d < np.datetime64(mid)
        self.l12 = (d >= np.datetime64(LAST12_FROM)) & (d <= np.datetime64(LAST12_TO))
        P = np.where(self.ok, price, 2.0)
        self.ret = {}
        for c in COMMS:
            self.ret[("back", c)] = back_net(P, self.f, c)
            ln = lay_net(P, self.f, c)
            self.ret[("lay", c)] = ln
            self.ret[("layliab", c)] = ln / (P - 1.0)
        # binomial SE floor: per-bet variance if the decision-time market probability were true (zero-winner
        # cells otherwise get SE ~ 0)
        pd_ = np.clip(np.nan_to_num(p_dec if p_dec is not None else 1.0 / P, nan=0.0), 1e-4, 0.9999)
        q = pd_ * (1 - pd_)
        d_back = (P - 1.0) * (1 - C8) + 1.0
        d_lay = (P - 1.0) + (1 - C8)
        self.var = {"back": q * d_back ** 2, "lay": q * d_lay ** 2, "layliab": q * (d_lay / (P - 1.0)) ** 2}
        self.n_markets, self.n_months, self.n_days = n_markets, n_months, n_days

    def se(self, x, idx, side):
        s_iid = float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")
        s_m = cl_se(x, self.mcode[idx])
        s_d = cl_se(x, self.dcode[idx])
        s_f = float(math.sqrt(self.var[side][idx].sum()) / len(x))
        vals = [v for v in (s_iid, s_m, s_d, s_f) if np.isfinite(v)]
        return (max(vals) if vals else float("nan")), s_m, s_d, s_f

    def cell(self, mask: np.ndarray, side: str) -> dict:
        idx = np.flatnonzero(mask & self.ok)
        n = len(idx)
        res = {"bets": int(n), "candidates": int(mask.sum())}
        if n == 0:
            res.update({"roi_8pct": None, "se": None, "z": None, "p": 1.0, "g4": False})
            return res
        r8 = self.ret[(side, C8)][idx]
        res["roi_8pct"] = r5(r8.mean())
        res["roi_0_5_10pct"] = [r5(self.ret[(side, c)][idx].mean()) for c in (0.0, 0.05, 0.10)]
        se, s_m, s_d, s_f = self.se(r8, idx, side) if n > 1 else (float("nan"),) * 4
        res["se"] = r5(se)
        res["se_mkt"], res["se_day"], res["se_binom_floor"] = r5(s_m), r5(s_d), r5(s_f)
        z = r8.mean() / se if np.isfinite(se) and se > 0 else float("nan")
        res["z"] = r5(z)
        res["p"] = one_sided_p(z) if n >= MIN_BETS_TEST else 1.0
        hr, hs, hb = [], [], []
        for hm in (self.first[idx], ~self.first[idx]):
            ii = idx[hm]
            x = r8[hm]
            hb.append(int(len(x)))
            hr.append(r5(x.mean()) if len(x) else None)
            hs.append(r5(self.se(x, ii, side)[0]) if len(x) > 1 else None)
        res["halves_roi_8pct"], res["halves_se"], res["halves_bets"] = hr, hs, hb
        lm = self.l12[idx]
        res["last12m_roi_8pct"] = r5(r8[lm].mean()) if lm.any() else None
        res["last12m_bets"] = int(lm.sum())
        res["g4"] = bool(all(a is not None and b is not None and b > 0 and a > 2 * b for a, b in zip(hr, hs))
                         and res["last12m_roi_8pct"] is not None and res["last12m_roi_8pct"] > 0)
        res["strike"] = r5((self.f[idx] > 0).mean())
        res["median_bsp"] = r5(np.median(self.price[idx]))
        res["mde_80pct"] = r5((2 + Z80) * se) if np.isfinite(se) else None
        res["bets_per_month"] = r5(n / self.n_months)
        res["share_markets_with_bet"] = r5(len(np.unique(self.mcode[idx])) / self.n_markets)
        return res


def sel_masks(r: pd.DataFrame, fav_col: str, price_col: str, with_all: bool = False) -> dict:
    b = band_of(r[price_col].to_numpy(float))
    sels = {"fav": r[fav_col].to_numpy()}
    for i, nm in enumerate(BAND_NAMES):
        sels[f"band {nm}"] = b == i
    if with_all:
        sels["all runners"] = np.ones(len(r), bool)
    return sels


# ----------------------------------------------------------------------------- Part B model


class FastLogit:
    """ponies.model.ConditionalLogit objective (mean per-race log loss + 0.5 l2 |w|^2 on standardised features,
    unpenalised names exempt) by L-BFGS-B; rows of one race contiguous."""

    def __init__(self, feats, l2, unpen=("lp",)):
        self.feats, self.l2, self.unpen = feats, l2, unpen

    def fit(self, X, y, starts, codes, w0=None):
        n_r = len(starts)
        self.sd = X.std(0)
        self.sd[self.sd == 0] = 1.0
        pen = float(self.l2) * np.array([0.0 if f in self.unpen else 1.0 for f in self.feats])
        sd = self.sd

        def lg(wt):
            s = X @ (wt / sd)
            e = np.exp(s - np.maximum.reduceat(s, starts)[codes])
            p = e / np.add.reduceat(e, starts)[codes]
            loss = -(y @ np.log(p + 1e-300)) / n_r + 0.5 * wt @ (pen * wt)
            grad = (X.T @ (p - y)) / sd / n_r + pen * wt
            return loss, grad

        x0 = np.zeros(X.shape[1]) if w0 is None else w0
        if w0 is None and "lp" in self.feats:
            x0 = x0.copy()
            x0[self.feats.index("lp")] = sd[self.feats.index("lp")]  # start at the market
        res = minimize(lg, x0, jac=True, method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-7})
        self.w = res.x
        self.conv = bool(res.success)
        self.nit = int(res.nit)
        return self

    def predict(self, X, starts, codes):
        s = X @ (self.w / self.sd)
        e = np.exp(s - np.maximum.reduceat(s, starts)[codes])
        return e / np.add.reduceat(e, starts)[codes]


def race_index(mcode: np.ndarray):
    starts = np.flatnonzero(np.r_[True, mcode[1:] != mcode[:-1]])
    codes = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(mcode)]))
    return starts, codes


_G: dict = {}


def _wf_task(args):
    name, l2 = args
    X = _G["X"][name]
    feats = _G["feats"][name]
    y, mcode, month_rows = _G["y"], _G["mcode"], _G["month_rows"]
    pred = np.full(len(y), np.nan)
    w0 = None
    nits, conv = [], []
    t0 = time.time()
    for m, (a, b) in month_rows.items():
        if m < OOS_FIRST or a == 0 or b <= a:
            continue
        st, cd = race_index(mcode[:a])
        mdl = FastLogit(feats, l2).fit(X[:a], y[:a], st, cd, w0)
        w0 = mdl.w
        nits.append(mdl.nit)
        conv.append(mdl.conv)
        s2, c2 = race_index(mcode[a:b])
        pred[a:b] = mdl.predict(X[a:b], s2, c2)
        last_w = (mdl.w / mdl.sd).copy()
    return name, l2, pred, {"mean_iters": float(np.mean(nits)) if nits else None, "all_converged": bool(all(conv)),
                            "secs": round(time.time() - t0, 1), "last_raw_w": last_w.tolist() if nits else None}


def build_model_matrix(r: pd.DataFrame, pop: str) -> tuple[dict, dict]:
    lp = np.log(np.clip(r["p_pre"].to_numpy(float), 1e-6, 1))
    fav = r["fav_pre"].to_numpy().astype(float)
    cols = {"lp": lp, "fav": fav}
    refs = {}
    for d in STATIC_DIMS:
        lv = r[d].to_numpy().astype(str)
        mk_levels = r.drop_duplicates("market_id")[d].astype(str).value_counts()
        ref = mk_levels.index[0]
        refs[d] = ref
        for l in mk_levels.index[1:]:
            if mk_levels[l] < 200:
                continue
            ind = (lv == l).astype(float)
            cols[f"lp*{d}={l}"] = lp * ind
            cols[f"fav*{d}={l}"] = fav * ind
    F = np.clip(np.nan_to_num(r["F_pre"].to_numpy(float)), -2, 2)
    cols["fav*F"] = fav * F
    pv = r["prev_fav_pre"].to_numpy().astype(str)
    cols["fav*prev_won"] = fav * (pv == "won")
    cols["fav*prev_lost"] = fav * (pv == "lost")
    if pop == "au_harness":
        mob = r["mobile"].to_numpy() & (r["ggrp"].to_numpy() >= 0)
        gg = r["ggrp"].to_numpy()
        cols["gate4-6"] = (mob & (gg == 1)).astype(float)
        cols["gate7+"] = (mob & (gg == 2)).astype(float)
        cols["gate_resid"] = np.where(mob, np.clip(r["Rown_pre"].to_numpy(float), -2, 2), 0.0)
    feats = list(cols)
    X = np.column_stack([cols[c] for c in feats]).astype(float)
    return {"card": (feats, X), "mkt": (["lp"], X[:, :1].copy())}, refs


def run_ablation(r: pd.DataFrame, workers: int) -> dict:
    """AU harness: which part of the card model carries the log-loss gain? Same walk-forward and l2 rule as
    run_model for each variant: gates (lp + gate 4-6 / 7+ dummies), gates+resid (+ intra-meeting gate residual),
    card_nogate (card model without the three gate columns), card (full)."""
    r = r.sort_values(["start", "market_id", "pn", "selection_id"]).reset_index(drop=True)
    mats, _ = build_model_matrix(r, "au_harness")
    feats, X = mats["card"]
    gi = [feats.index(c) for c in ("gate4-6", "gate7+", "gate_resid")]
    variants = {"gates": [0, gi[0], gi[1]], "gates+resid": [0] + gi,
                "card_nogate": [j for j in range(len(feats)) if j not in gi], "card": list(range(len(feats)))}
    mcode = pd.factorize(r["market_id"])[0]
    y = r["f"].to_numpy(float)
    y = y / np.maximum(np.bincount(mcode, weights=y)[mcode], 1e-12)
    months = r["date"].dt.to_period("M").to_numpy()
    month_rows = {}
    for m in pd.unique(months):
        ii = np.flatnonzero(months == m)
        month_rows[m] = (int(ii.min()), int(ii.max()) + 1)
    _G.clear()
    _G.update({"X": {k: X[:, v].copy() for k, v in variants.items()} | {"mkt": X[:, :1].copy()},
               "feats": {k: [feats[j] for j in v] for k, v in variants.items()} | {"mkt": ["lp"]},
               "y": y, "mcode": mcode, "month_rows": month_rows})
    tasks = [(k, l2) for k in variants for l2 in L2_GRID] + [("mkt", 0.0)]
    t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("fork").Pool(max(workers, 1)) as pool:
        res = pool.map(_wf_task, tasks)
    preds = {(nm, l2): p for nm, l2, p, _ in res}
    lastw = {(nm, l2): inf.get("last_raw_w") for nm, l2, _, inf in res}
    n_r = mcode.max() + 1
    race_month = pd.Series(months).groupby(mcode).first().to_numpy()
    yb = y > 0

    def race_ll(p):
        return np.bincount(mcode, weights=np.where(yb, -y * np.log(np.clip(p, 1e-15, 1)), 0.0), minlength=n_r)

    ll = {k: race_ll(p) for k, p in preds.items()}
    ll_mkt = race_ll(r["p_pre"].to_numpy(float))
    month_list = sorted(month_rows)
    ev_r = np.isin(race_month, [m for m in month_list if m >= EVAL_FIRST])
    rd = r.groupby(mcode)["date"].first().to_numpy()[ev_r]
    mid = np.sort(rd)[len(rd) // 2]
    first = rd < mid
    l12 = rd >= np.datetime64(LAST12_FROM)
    chosen_ll, chosen_counts, coefs = {}, {}, {}
    for k in variants:
        llk = np.full(n_r, np.nan)
        ch = []
        for i, m in enumerate(month_list):
            if m < EVAL_FIRST:
                continue
            win = [mm for mm in month_list[max(0, i - L2_WINDOW):i] if mm >= OOS_FIRST]
            msk = np.isin(race_month, win)
            best = min(L2_GRID, key=lambda l2: np.nanmean(ll[(k, l2)][msk]))
            ch.append(best)
            rm = race_month == m
            llk[rm] = ll[(k, best)][rm]
        chosen_ll[k] = llk
        cc = pd.Series(ch).value_counts()
        chosen_counts[k] = {str(a): int(b) for a, b in cc.items()}
        w_last = lastw.get((k, ch[-1])) if ch else None
        if w_last:
            fk = _G["feats"][k]
            coefs[k] = {f: r5(w_last[j]) for j, f in enumerate(fk) if f in ("lp", "gate4-6", "gate7+", "gate_resid", "fav*F")}
    chosen_ll["mkt_recal"] = ll[("mkt", 0.0)]

    def gain(a, b):
        d = (a - b)[ev_r]
        hs = []
        for m_ in (first, ~first):
            x = d[m_]
            hs.append([r5(x.mean()), r5(x.std(ddof=1) / math.sqrt(len(x)))])
        se = d.std(ddof=1) / math.sqrt(len(d))
        return {"gain": r5(d.mean()), "se": r5(se), "z": r5(d.mean() / se), "halves_gain_se": hs,
                "last12m_gain": r5(d[l12].mean())}

    res_out = {"section": "model_ablation_au_harness", "races": int(ev_r.sum()), "mid_date": str(pd.Timestamp(mid).date()),
               "l2_chosen_counts": chosen_counts, "last_fit_coefs_raw": coefs,
               "vs_market_shin_pre": {k: gain(ll_mkt, chosen_ll[k]) for k in list(variants) + ["mkt_recal"]},
               "gates+resid_vs_gates": gain(chosen_ll["gates"], chosen_ll["gates+resid"]),
               "card_vs_gates+resid": gain(chosen_ll["gates+resid"], chosen_ll["card"]),
               "card_nogate_vs_mkt_recal": gain(chosen_ll["mkt_recal"], chosen_ll["card_nogate"]),
               "card_vs_card_nogate": gain(chosen_ll["card_nogate"], chosen_ll["card"]),
               "secs": round(time.time() - t0, 1)}
    out(res_out)
    return res_out


def run_model(pop: str, r: pd.DataFrame, workers: int, cells: list, family_rows: list) -> dict:
    """Walk-forward conditional logit for one population; adds bet cells; returns the probability summary."""
    r = r.sort_values(["start", "market_id", "pn", "selection_id"]).reset_index(drop=True)
    mats, refs = build_model_matrix(r, pop)
    mcode = pd.factorize(r["market_id"])[0]
    y = r["f"].to_numpy(float)
    y = y / np.maximum(np.bincount(mcode, weights=y)[mcode], 1e-12)  # targets sum to 1 per race
    months = r["date"].dt.to_period("M").to_numpy()
    month_rows = {}
    um = pd.unique(months)
    for m in um:
        ii = np.flatnonzero(months == m)
        month_rows[m] = (int(ii.min()), int(ii.max()) + 1)
    _G.clear()
    _G.update({"X": {k: v[1] for k, v in mats.items()}, "feats": {k: v[0] for k, v in mats.items()},
               "y": y, "mcode": mcode, "month_rows": month_rows})
    tasks = [("card", l2) for l2 in L2_GRID] + [("mkt", 0.0)]
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_wf_task, tasks)
    else:
        res = [_wf_task(t) for t in tasks]
    preds = {(nm, l2): p for nm, l2, p, _ in res}
    fit_info = {f"{nm}@{l2}": inf for nm, l2, _, inf in res}
    # per-race log loss by month for l2 selection
    n_r = mcode.max() + 1
    race_month = pd.Series(months).groupby(mcode).first().to_numpy()
    yb = y > 0

    def race_ll(p):
        return np.bincount(mcode, weights=np.where(yb, -y * np.log(np.clip(p, 1e-15, 1)), 0.0), minlength=n_r)

    ll = {k: race_ll(p) for k, p in preds.items()}
    p_mkt = r["p_pre"].to_numpy(float)
    ll_mkt = race_ll(p_mkt)
    month_list = sorted(month_rows)
    chosen = {}
    p_card = np.full(len(y), np.nan)
    for i, m in enumerate(month_list):
        if m < EVAL_FIRST:
            continue
        win = [mm for mm in month_list[max(0, i - L2_WINDOW):i] if mm >= OOS_FIRST]
        msk = np.isin(race_month, win)
        best = min(L2_GRID, key=lambda l2: np.nanmean(ll[("card", l2)][msk]))
        chosen[str(m)] = best
        a, b = month_rows[m]
        p_card[a:b] = preds[("card", best)][a:b]
    ev = np.isfinite(p_card)
    ev_r = np.isin(race_month, [m for m in month_list if m >= EVAL_FIRST])
    ll_card = race_ll(np.where(ev, p_card, 1.0))
    ll_recal = ll[("mkt", 0.0)]
    rd = r.groupby(mcode)["date"].first().to_numpy()
    rd_ev = rd[ev_r]
    mid = np.sort(rd_ev)[len(rd_ev) // 2]
    first = rd_ev < mid
    l12 = (rd_ev >= np.datetime64(LAST12_FROM))

    def gain(a, b):
        d = (a - b)[ev_r]
        hs = []
        for m_ in (first, ~first):
            x = d[m_]
            hs.append([r5(x.mean()), r5(x.std(ddof=1) / math.sqrt(len(x)))])
        se = d.std(ddof=1) / math.sqrt(len(d))
        return {"races": int(len(d)), "gain_nats_per_race": r5(d.mean()), "se": r5(se), "z": r5(d.mean() / se),
                "halves_gain_se": hs, "last12m_gain": r5(d[l12].mean())}

    edge = [L2_GRID[0], L2_GRID[-1]]
    ch = pd.Series(chosen).value_counts().to_dict()
    summ = {"section": "model_logloss", "pop": pop, "features": len(mats["card"][0]), "references": refs,
            "eval_months": f"{EVAL_FIRST}..{month_list[-1]}", "mid_date": str(pd.Timestamp(mid).date()),
            "l2_chosen_counts": {str(k): int(v) for k, v in ch.items()},
            "l2_at_grid_edge_months": int(sum(v for k, v in ch.items() if k in edge)),
            "card_vs_market_shin_pre": gain(ll_mkt, ll_card),
            "card_vs_market_recal": gain(ll_recal, ll_card),
            "recal_vs_market_shin_pre": gain(ll_mkt, ll_recal),
            "ll_market_mean": r5(ll_mkt[ev_r].mean()),
            "fit_info": {k: {kk: vv for kk, vv in v.items() if kk != "last_raw_w"} for k, v in fit_info.items()},
            "secs": round(time.time() - t0, 1)}
    # coefficients of the last fit at the most chosen l2
    top_l2 = max(ch, key=ch.get) if ch else L2_GRID[3]
    lw = fit_info.get(f"card@{top_l2}", {}).get("last_raw_w")
    if lw:
        feats = mats["card"][0]
        order = np.argsort(-np.abs(np.array(lw[1:]))) + 1
        summ["last_fit_top_coefs_raw"] = {feats[j]: r5(lw[j]) for j in [0] + list(order[:12])}
    out(summ)
    # bets from the model (evaluation months only), settled at BSP
    rr = r.loc[ev].copy()
    rr["edge"] = p_card[ev] / rr["p_pre"].to_numpy() - 1.0
    rr = rr.sort_values(["lday", "start", "market_id"]).reset_index(drop=True)
    rd2 = rr["date"].to_numpy()
    mid2 = pd.Timestamp(np.sort(rr.drop_duplicates("market_id")["date"].to_numpy())[rr["market_id"].nunique() // 2])
    n_months = rr["date"].dt.to_period("M").nunique()
    E = Engine(rr, rr["bsp"].to_numpy(float), mid2, rr["market_id"].nunique(), n_months, rr["lday"].nunique(),
               rr["p_pre"].to_numpy(float))
    edge_arr = rr["edge"].to_numpy()
    for k in K_MODEL:
        for side in ("back", "lay", "layliab"):
            msk = edge_arr > k if side == "back" else edge_arr < -k
            c = E.cell(msk, side)
            rec = {"part": "B_model", "pop": pop, "selection": f"model edge {'>' if side == 'back' else '<'} "
                   f"{'' if side == 'back' else '-'}{k}", "dim": "model", "level": f"k={k}", "side": side,
                   "decision": "pre_off", "settlement": "bsp" if side == "back" else
                   ("bsp_per_stake" if side == "lay" else "per_liability_bsp"), **c}
            cells.append(rec)
            family_rows.append(rec)
    return summ


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="15%% of meetings")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--ablation-only", action="store_true",
                    help="only the AU harness card-model ablation (gates / gates+resid / card_nogate / card)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cells-csv", type=Path, default=None)
    a = ap.parse_args()
    T0 = time.time()
    out({"section": "run", "dev": a.dev, "model": not a.no_model, "commission_primary": C8,
         "g1_book": [BOOK_LO, BOOK_HI], "t30_book": T30_BOOK, "intra_cutoff_min": {"pre_off": CUT_PRE, "t30": CUT_T30}})
    w, mk, info = load(a.dev)
    # intra-meeting features (pre-off and t30 versions)
    mk = pd.concat([mk, intra_meeting(mk, CUT_PRE, "_pre"), intra_meeting(mk, CUT_T30, "_t30")], axis=1)
    lab = level_arrays(mk)
    for c in lab.columns:
        mk[c] = lab[c].to_numpy()
    # dead-heat share check
    dh = w[(w["f"] > 0) & (w["f"] < 1)]
    info["dead_heat_winner_rows"] = int(len(dh))
    info["dead_heat_share_values"] = {str(k): int(v) for k, v in dh["f"].round(3).value_counts().head(3).items()}
    by_pop = {}
    for pop in POPS:
        s = mk[mk["pop"] == pop]
        by_pop[pop] = {"win_markets": int(len(s)), "settled": int(s["settled"].sum()),
                       "all_runners_priced_pre": int((s["settled"] & s["all_pre"]).sum()),
                       "g6_gap_and_book_lt_0.97": int((s["settled"] & s["g6_drop"]).sum()),
                       "g1_qualifying": int(s["g1"].sum()), "t30_qualifying": int(s["t30_ok"].sum()),
                       "median_book_pre_g1": r5(s.loc[s["g1"], "book_pre"].median())}
    info["by_pop"] = by_pop
    out(info)
    out({"section": "parse_rules", **PARSE_RULES})

    # runner table with market-level labels
    mcols = ["market_id", "lday", "conc", "F_pre", "prev_fav_pre", "n_prev_mob_pre", "R0_pre", "R1_pre",
             "R2_pre", "F_t30", "prev_fav_t30", "n_prev_mob_t30", "R0_t30", "R1_t30", "R2_t30"] + \
        [c for c in lab.columns if c != "all"] + ["fav_form_pre", "fav_form_t30"]
    r_all = w.merge(mk[mcols], on="market_id", how="inner")
    r_all["all"] = "all"
    gg = r_all["ggrp"].to_numpy()
    for suf in ("_pre", "_t30"):
        Rown = np.select([gg == 0, gg == 1, gg == 2], [r_all[f"R0{suf}"], r_all[f"R1{suf}"], r_all[f"R2{suf}"]], np.nan)
        r_all[f"Rown{suf}"] = Rown
        r_all[f"gate_resid{suf}"] = gate_level(r_all["mobile"].to_numpy(), gg, r_all[f"n_prev_mob{suf}"].to_numpy(), Rown)
    del w
    if a.ablation_only:
        rh = r_all[(r_all["pop"] == "au_harness") & r_all["g1"]].reset_index(drop=True)
        run_ablation(rh, a.workers)
        out({"section": "done", "secs": round(time.time() - T0, 1)})
        return 0

    # level counts (markets) for each segment
    lvl_counts = {}
    for pop in POPS:
        s = mk[(mk["pop"] == pop) & mk["g1"]]
        lvl_counts[pop] = {d: {str(k): int(v) for k, v in s[d if d not in ("fav_form", "prev_fav") else d + "_pre"]
                                .value_counts().items()} for d in DIMS_ALL if d != "all"}
        if pop == "au_harness":
            rr = r_all[(r_all["pop"] == pop) & r_all["g1"]]
            lvl_counts[pop]["gate_resid (runners)"] = {str(k): int(v) for k, v in rr["gate_resid_pre"].value_counts().items()}
    out({"section": "segment_levels_g1_markets", "counts": lvl_counts})

    # ---------------- PART A: pre-off scan at BSP
    cells: list[dict] = []
    family_rows: list[dict] = []
    engines = {}
    pop_rows = {}
    for pop in POPS:
        r = r_all[(r_all["pop"] == pop) & r_all["g1"]].sort_values(["lday", "start", "market_id", "pn"]).reset_index(drop=True)
        pop_rows[pop] = r
        mdates = r.drop_duplicates("market_id")["date"].sort_values().to_numpy()
        mid = pd.Timestamp(mdates[len(mdates) // 2])
        n_months = r["date"].dt.to_period("M").nunique()
        E = Engine(r, r["bsp"].to_numpy(float), mid, r["market_id"].nunique(), n_months, r["lday"].nunique(),
                   r["p_pre"].to_numpy(float))
        engines[pop] = (E, mid)
        sels = sel_masks(r, "fav_pre", "ltp_pre")
        dims = DIMS_ALL + (["gate_resid"] if pop == "au_harness" else [])
        tables = {}
        for d in dims:
            col = d + "_pre" if d in INTRA_DIMS else d
            lv_arr = r[col].to_numpy().astype(str)
            levels = [l for l in (CLASS_ORDER[pop] if d == "class" else DIST_EDGES[pop][1] + ["unknown"] if d == "distance"
                                  else RACE_NO_LEVELS if d == "race_no" else CONC_LEVELS if d == "concurrency"
                                  else FAV_RESID_LEVELS if d == "fav_form" else GATE_LEVELS if d == "gate_resid"
                                  else sorted(pd.unique(lv_arr)))]
            levels = [l for l in levels if (lv_arr == l).any()] + sorted(set(pd.unique(lv_arr)) - set(levels))
            ss = dict(sels)
            if d == "gate_resid":
                ss["all runners"] = np.ones(len(r), bool)
            tab = {}
            for lvl in levels:
                lm = lv_arr == lvl
                row = {}
                for sname, smask in ss.items():
                    for side in ("back", "lay", "layliab"):
                        c = E.cell(smask & lm, side)
                        rec = {"part": "A_scan", "pop": pop, "selection": sname, "dim": d, "level": lvl, "side": side,
                               "decision": "pre_off", "settlement": "bsp" if side == "back" else
                               ("bsp_per_stake" if side == "lay" else "per_liability_bsp"), **c}
                        cells.append(rec)
                        family_rows.append(rec)
                        if side == "back":
                            row[sname] = [c["bets"], c.get("roi_8pct"), c.get("se")]
                        elif side == "layliab":
                            row[sname] += [c.get("roi_8pct"), c.get("se")]
                tab[lvl] = row
            tables[d] = tab
            # compact print: fav full, bands back roi8 and z
            comp = {}
            for lvl, row in tab.items():
                comp[lvl] = {"fav[bets,back,se,layliab,se]": row.get("fav"),
                             "bands_back_roi8": [row[f"band {b}"][1] for b in BAND_NAMES],
                             "bands_back_z": [r5(row[f"band {b}"][1] / row[f"band {b}"][2]) if row[f"band {b}"][2] else None
                                              for b in BAND_NAMES],
                             "bands_layliab_roi8": [row[f"band {b}"][3] for b in BAND_NAMES]}
                if "all runners" in row:
                    comp[lvl]["all_runners[bets,back,se,layliab,se]"] = row["all runners"]
            out({"section": "scan_table", "pop": pop, "dim": d, "mid_date": str(mid.date()), "rows": comp})
    out({"section": "scan_done", "cells": len(cells), "secs": round(time.time() - T0, 1)})

    # ---------------- G4 passers and top cells (Part A)
    A = pd.DataFrame([c for c in cells if c["part"] == "A_scan"])
    A["zf"] = pd.to_numeric(A["z"], errors="coerce")
    passers = A[A["g4"]]
    out({"section": "g4_passers_partA", "n": int(len(passers)),
         "rows": passers[["pop", "selection", "dim", "level", "side", "bets", "roi_8pct", "se", "z", "halves_roi_8pct",
                          "halves_se", "last12m_roi_8pct"]].to_dict("records")})
    top = A[A["bets"] >= 100].sort_values("zf", ascending=False).head(25)
    out({"section": "top25_partA_by_z_min100", "rows": top[["pop", "selection", "dim", "level", "side", "bets", "roi_8pct",
                                                           "se", "z", "halves_roi_8pct", "halves_se", "last12m_roi_8pct",
                                                           "g4"]].to_dict("records")})
    # null-distribution check: how many z > 2 vs expected
    zz = A.loc[A["bets"] >= MIN_BETS_TEST, "zf"].dropna()
    out({"section": "z_distribution_partA", "cells_testable": int(len(zz)), "z_gt_2": int((zz > 2).sum()),
         "expected_iid_null_z_gt_2": r5(0.02275 * len(zz)), "z_gt_3": int((zz > 3).sum()),
         "mean_z": r5(zz.mean()), "sd_z": r5(zz.std()),
         "by_side_mean_z": {k: r5(v) for k, v in A[A["bets"] >= MIN_BETS_TEST].groupby("side")["zf"].mean().items()}})

    # ---------------- claims: best cell per (pop, dim) + pre-registered intra-meeting rules + G4 passers
    claim_keys = []
    for (pop, d), g in A[A["bets"] >= 100].groupby(["pop", "dim"]):
        if d == "all":
            continue
        b = g.sort_values("zf", ascending=False).iloc[0]
        claim_keys.append((pop, b["selection"], d, b["level"], b["side"], "best z in dimension"))
    for pop in POPS:
        for side in ("back", "layliab"):
            claim_keys.append((pop, "fav", "fav_form", "<=-1", side, "pre-registered: favourite after cold earlier favourites"))
            claim_keys.append((pop, "fav", "fav_form", ">=1", side, "pre-registered: favourite after hot earlier favourites"))
            claim_keys.append((pop, "fav", "prev_fav", "lost", side, "pre-registered: favourite after the previous favourite lost"))
            claim_keys.append((pop, "fav", "bigday_group", "big", side, "pre-registered: favourite on a big (Group) day"))
            claim_keys.append((pop, "fav", "concurrency", "4+", side, "pre-registered: favourite when >= 4 other markets start within 5 min"))
            claim_keys.append((pop, "fav", "last_race", "last", side, "pre-registered: favourite in the last race"))
    claim_keys.append(("au_harness", "all runners", "gate_resid", ">=0.75", "back", "pre-registered: AU harness runners whose gate group won more than the market implied earlier today"))
    claim_keys.append(("au_harness", "all runners", "gate_resid", "<=-0.75", "layliab", "pre-registered: AU harness runners whose gate group won less than implied earlier today (lay)"))
    for _, b in passers.iterrows():
        claim_keys.append((b["pop"], b["selection"], b["dim"], b["level"], b["side"], "passed G4 in the scan"))
    seen = set()
    ck = []
    for k in claim_keys:
        if k[:5] in seen:
            continue
        seen.add(k[:5])
        ck.append(k)

    # ---------------- PART A2: t30 replication of the claim cells (limit-then-BSP and BSP)
    t30_eng = {}
    for pop in POPS:
        r = r_all[(r_all["pop"] == pop) & r_all["t30_ok"]].sort_values(["lday", "start", "market_id", "pn"]).reset_index(drop=True)
        if r.empty:
            continue
        mdates = r.drop_duplicates("market_id")["date"].sort_values().to_numpy()
        mid = pd.Timestamp(mdates[len(mdates) // 2])
        n_months = max(r["date"].dt.to_period("M").nunique(), 1)
        t30 = r["ltp_t30"].to_numpy(float)
        pre = r["ltp_pre"].to_numpy(float)
        bsp = r["bsp"].to_numpy(float)
        Lb = tick_shorter(t30)
        fill_b = (np.nan_to_num(pre) >= Lb - 1e-9) | (np.nan_to_num(bsp) >= Lb - 1e-9)
        Ll = tick_longer(t30)
        fill_l = (np.nan_to_num(pre, nan=1e9) <= Ll + 1e-9) | (np.nan_to_num(bsp, nan=1e9) <= Ll + 1e-9)
        ent_b = np.where(fill_b, Lb, bsp)
        ent_l = np.where(fill_l, Ll, bsp)
        args = (mid, r["market_id"].nunique(), n_months, r["lday"].nunique(), r["p_t30"].to_numpy(float))
        t30_eng[pop] = {"r": r, "bsp": Engine(r, bsp, *args), "lim_back": Engine(r, ent_b, *args),
                        "lim_lay": Engine(r, ent_l, *args), "sels": sel_masks(r, "fav_t30", "ltp_t30", with_all=True),
                        "fill_b": fill_b, "fill_l": fill_l}
    claim_rows = []
    for pop, sname, d, lvl, side, why in ck:
        E, mid = engines[pop]
        r = pop_rows[pop]
        col = d + "_pre" if d in INTRA_DIMS else d
        sels = sel_masks(r, "fav_pre", "ltp_pre", with_all=True)
        m = sels[sname] & (r[col].to_numpy().astype(str) == lvl)
        c = E.cell(m, side)
        rep = {}
        if pop in t30_eng:
            T = t30_eng[pop]
            rt = T["r"]
            col30 = d + "_t30" if d in INTRA_DIMS else d
            m30 = T["sels"][sname] & (rt[col30].to_numpy().astype(str) == lvl)
            for sett, eng in (("limit_then_bsp", T["lim_back"] if side == "back" else T["lim_lay"]), ("bsp", T["bsp"])):
                c30 = eng.cell(m30, side)
                rec = {"part": "A2_t30", "pop": pop, "selection": sname, "dim": d, "level": lvl, "side": side,
                       "decision": "t30", "settlement": sett, **c30}
                family_rows.append(rec)
                cells.append(rec)
                fill = (T["fill_b"] if side == "back" else T["fill_l"])[m30 & eng.ok]
                rep[sett] = {k: c30.get(k) for k in ("bets", "roi_8pct", "se", "z", "halves_roi_8pct", "halves_se",
                                                     "last12m_roi_8pct", "g4")}
                if sett == "limit_then_bsp":
                    rep[sett]["fill_rate"] = r5(fill.mean()) if len(fill) else None
        claim_rows.append({"pop": pop, "selection": sname, "dim": d, "level": lvl, "side": side, "why": why,
                           "mid_date": str(mid.date()), **{k: c.get(k) for k in (
                               "bets", "candidates", "roi_8pct", "roi_0_5_10pct", "se", "se_mkt", "se_day", "z", "p",
                               "halves_roi_8pct", "halves_se", "halves_bets", "last12m_roi_8pct", "last12m_bets",
                               "g4", "strike", "median_bsp", "mde_80pct", "bets_per_month", "share_markets_with_bet")},
                           "t30_replication": rep})

    # ---------------- PART B: walk-forward card model
    model_summ = {}
    if not a.no_model:
        for pop in POPS:
            r = pop_rows[pop]
            model_summ[pop] = run_model(pop, r, a.workers, cells, family_rows)
            out({"section": "model_progress", "pop": pop, "secs_total": round(time.time() - T0, 1)})
        Bc = [c for c in cells if c["part"] == "B_model"]
        out({"section": "model_bets", "rows": [{k: c.get(k) for k in (
            "pop", "level", "side", "settlement", "bets", "roi_8pct", "roi_0_5_10pct", "se", "z", "halves_roi_8pct",
            "halves_se", "last12m_roi_8pct", "g4", "mde_80pct", "bets_per_month", "share_markets_with_bet", "median_bsp")}
            for c in Bc]})
        for c in Bc:
            if c["side"] in ("back", "layliab"):
                claim_rows.append({"pop": c["pop"], "selection": c["selection"], "dim": "model", "level": c["level"],
                                   "side": c["side"], "why": "walk-forward card model (Part B)",
                                   **{k: c.get(k) for k in ("bets", "candidates", "roi_8pct", "roi_0_5_10pct", "se",
                                                            "se_mkt", "se_day", "z", "p", "halves_roi_8pct", "halves_se",
                                                            "halves_bets", "last12m_roi_8pct", "last12m_bets", "g4",
                                                            "strike", "median_bsp", "mde_80pct", "bets_per_month",
                                                            "share_markets_with_bet")}, "t30_replication": {}})

    # ---------------- G5: family count and BH within this experiment
    P = np.array([c.get("p", 1.0) if c.get("p") is not None else 1.0 for c in family_rows])
    rej = bh_reject(P)
    for c, rj in zip(family_rows, rej):
        c["bh_family"] = bool(rj)
    by_part = pd.Series([c["part"] for c in family_rows]).value_counts().to_dict()
    n_test = int(sum(1 for c in family_rows if c.get("bets", 0) >= MIN_BETS_TEST))
    out({"section": "g5_count", "cells_total": len(family_rows), "cells_by_part": by_part,
         "cells_with_ge_30_bets": n_test, "bh_q": FDR, "bh_rejections_within_family": int(rej.sum()),
         "min_p": r5(P.min()), "bh_threshold_smallest_p": r5(FDR / len(P)),
         "rejected": [{k: c.get(k) for k in ("part", "pop", "selection", "dim", "level", "side", "settlement", "bets",
                                              "roi_8pct", "se", "z", "g4")} for c, rj in zip(family_rows, rej) if rj][:40]})
    # attach BH flag to claims
    bh_index = {(c["part"], c["pop"], c["selection"], c["dim"], c["level"], c["side"], c.get("settlement")): c["bh_family"]
                for c in family_rows}
    for cr in claim_rows:
        part = "B_model" if cr["dim"] == "model" else "A_scan"
        sett = ("bsp" if cr["side"] == "back" else "bsp_per_stake" if cr["side"] == "lay" else "per_liability_bsp")
        cr["bh_family"] = bh_index.get((part, cr["pop"], cr["selection"], cr["dim"], cr["level"], cr["side"], sett), False)
        cr["passes_all_gates"] = bool(cr["g4"] and cr["bh_family"])
    out({"section": "claims", "n": len(claim_rows), "rows": claim_rows})
    out({"section": "caveats", "rows": CAVEATS})
    if a.cells_csv:
        a.cells_csv.parent.mkdir(parents=True, exist_ok=True)
        flat = []
        for c in family_rows:
            rec = {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in c.items()}
            flat.append(rec)
        pd.DataFrame(flat).to_csv(a.cells_csv, index=False)
    out({"section": "done", "secs": round(time.time() - T0, 1), "cells": len(family_rows)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

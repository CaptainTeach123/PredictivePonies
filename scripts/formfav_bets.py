"""Walk-forward backtest of TAB bets beyond the winner: place, quinella,
exacta, trifecta (and tote win), settled at the real dividends.

    python scripts/formfav_bets.py [--store data/raw/formfav] [--test-share 0.4]

For every test day, using only earlier days [A-08]:
  1. win probabilities: conditional logit on the pre-jump market (+ FormFav's
     own predictions when present) [A-42][A-47]
  2. finishing-order probabilities from them, with Harville discounts fitted
     on past 2nd and 3rd places [A-49]
  3. expected dividend of each combination if it comes in, from a regression
     of past winning dividends on the market's probability of that
     combination and the field size [A-50]
  4. bet 1 unit where probability x expected dividend > 1 + threshold, and
     settle at the actual dividend (0 if the combination missed).

Prints summary numbers only (safe for public CI logs); never race data.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ponies import config  # noqa: E402
from ponies.exotics import fit_lambdas, order_probs, places_paid  # noqa: E402
from ponies.formfav import load_dividends, load_store, only  # noqa: E402
from formfav_experiments import prepare, walk_forward  # noqa: E402

POOLS = ("Win", "Place", "Quinella", "Exacta", "Trifecta")


def combos(pool: str, probs: dict, field: int):
    """(combination tuple of runner indices, probability) for one pool."""
    n = len(probs["win"])
    if pool == "Win":
        yield from (((i,), probs["win"][i]) for i in range(n))
    elif pool == "Place":
        k = places_paid(field)
        if k:
            top = probs["top3"] if k == 3 and "top3" in probs else probs["top2"]
            yield from (((i,), top[i]) for i in range(n))
    elif pool == "Quinella":
        ex = probs["exacta"]
        yield from (((i, j), ex[i, j] + ex[j, i]) for i, j in itertools.combinations(range(n), 2))
    elif pool == "Exacta":
        ex = probs["exacta"]
        yield from (((i, j), ex[i, j]) for i in range(n) for j in range(n) if i != j)
    elif pool == "Trifecta" and "trifecta" in probs:
        tri = probs["trifecta"]
        yield from (((i, j, k), tri[i, j, k]) for i in range(n) for j in range(n) for k in range(n)
                    if len({i, j, k}) == 3 and tri[i, j, k] > 1e-5)


def winning_sets(divs: list[dict], pool: str) -> dict:
    """combination key -> dividend (quinella and place keys are unordered)."""
    out = {}
    for d in divs:
        if d["pool"] != pool:
            continue
        key = tuple(sorted(d["combination"])) if pool in ("Quinella", "Place") else tuple(d["combination"])
        out[key] = d["dividend"]
    return out


def race_table(df: pd.DataFrame, divs: dict, p_model: np.ndarray) -> list[dict]:
    """Per race: runner programs, market and model win probs, finish order, dividends."""
    races = []
    for rid, g in df.groupby("race_id", sort=False):
        if rid not in divs:
            continue
        idx = g.index.to_numpy()
        mk = np.exp(g["mk_logp"].to_numpy())
        pm = p_model[idx]
        fin = g["finish_position"].to_numpy()
        order = [int(i) for i in np.argsort(fin) if not math.isnan(fin[i]) and fin[i] < 90][:3]
        races.append({"race_id": rid, "date": g["date"].iloc[0], "programs": g["program"].tolist(),
                      "mk": mk / mk.sum(), "model": None if np.isnan(pm).any() else pm / pm.sum(),
                      "order": order, "field": len(g), "divs": divs[rid]})
    return races


def fit_dividend_models(train: list[dict], lams: tuple[float, float]) -> dict:
    """[A-50] Per pool: log(dividend) ~ a + b log(market prob of the combination)
    + c log(field size), on past winning combinations. Returns coefficients and
    the residual variance (for the lognormal mean)."""
    models = {}
    for pool in POOLS:
        X, y = [], []
        for r in train:
            probs = order_probs(r["mk"], *lams)
            pos = {prog: i for i, prog in enumerate(r["programs"])}
            for key, div in winning_sets(r["divs"], pool).items():
                try:
                    ix = tuple(pos[k] for k in key)
                except KeyError:
                    continue
                q = combo_prob(pool, probs, ix, r["field"])
                if q and q > 0:
                    X.append([1.0, math.log(q), math.log(r["field"])])
                    y.append(math.log(div))
        if len(y) >= 30:
            X, y = np.array(X), np.array(y)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ coef
            models[pool] = (coef, float(resid.var()), len(y))
    return models


def combo_prob(pool: str, probs: dict, ix: tuple, field: int) -> float | None:
    if pool == "Win":
        return probs["win"][ix[0]]
    if pool == "Place":
        k = places_paid(field)
        top = probs["top3"] if k == 3 and "top3" in probs else probs["top2"]
        return top[ix[0]] if k else None
    if pool == "Quinella":
        return probs["exacta"][ix[0], ix[1]] + probs["exacta"][ix[1], ix[0]]
    if pool == "Exacta":
        return probs["exacta"][ix[0], ix[1]]
    if pool == "Trifecta" and "trifecta" in probs:
        return probs["trifecta"][ix]
    return None


def backtest(races: list[dict], test_days: list, use: str, thresholds=(0.0, 0.2, 0.5),
             min_train: int = 300) -> dict:
    """use: 'mk' (market probabilities) or 'model'."""
    stakes = {(pool, t): [] for pool in POOLS for t in thresholds}
    blind = {pool: [] for pool in POOLS}
    lam_log = []
    for d in test_days:
        train = [r for r in races if r["date"] < d]
        test = [r for r in races if r["date"] == d]
        if len(train) < min_train or not test:
            continue
        lams = fit_lambdas([(r["mk"], r["order"]) for r in train])
        lam_log.append(lams)
        dm = fit_dividend_models(train, lams)
        for r in test:
            p = r[use] if use == "mk" or r["model"] is None else r["model"]
            if use == "model" and r["model"] is None:
                continue
            probs, mprobs = order_probs(p, *lams), order_probs(r["mk"], *lams)
            for pool in POOLS:
                if pool not in dm:
                    continue
                coef, var, _ = dm[pool]
                won = winning_sets(r["divs"], pool)
                if not won:
                    continue  # pool not run / no dividend declared
                for ix, prob in combos(pool, probs, r["field"]):
                    q = combo_prob(pool, mprobs, ix, r["field"])
                    if not q or q <= 0:
                        continue
                    exp_div = math.exp(coef @ [1.0, math.log(q), math.log(r["field"])] + var / 2)
                    key = tuple(r["programs"][i] for i in ix)
                    key = tuple(sorted(key)) if pool in ("Quinella", "Place") else key
                    ret = won.get(key, 0.0) - 1.0
                    if pool in ("Win", "Place"):
                        blind[pool].append(ret)
                    for t in thresholds:
                        if prob * exp_div > 1 + t:
                            stakes[(pool, t)].append(ret)
    summ = lambda xs: ({"bets": len(xs), "hits": int(sum(x > -1 for x in xs)),
                        "roi": round(float(np.mean(xs)), 4),
                        "roi_se": round(float(np.std(xs, ddof=1) / math.sqrt(len(xs))), 4) if len(xs) > 1 else None}
                       if xs else {"bets": 0})
    return {"lambdas_last": lam_log[-1] if lam_log else None,
            "bet_every_runner": {p: summ(v) for p, v in blind.items() if v},
            "value_bets": {f"{pool}>{t}": summ(v) for (pool, t), v in stakes.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path(config.RAW_DIR / "formfav"))
    ap.add_argument("--test-share", type=float, default=0.4)
    a = ap.parse_args()

    df = prepare(only(load_store(a.store)))
    divs = load_dividends(a.store)
    days = sorted(df["date"].unique())
    if len(days) < 5:
        print("too few race days")
        return 0
    test_days = days[int(len(days) * (1 - a.test_share)):]
    p_model = walk_forward(df, ["mk_logp", "ff_model_logp", "has_ff_pred"], test_days, None)
    races = race_table(df, divs, p_model)
    pools_seen = pd.Series([d["pool"] for r in races for d in r["divs"]]).value_counts().to_dict()
    print(json.dumps({"races_with_dividends": len(races), "test_days": len(test_days),
                      "dividend_rows_by_pool": {k: int(v) for k, v in pools_seen.items()}}))
    for use in ("mk", "model"):
        print(json.dumps({"probabilities": "market" if use == "mk" else "market+formfav_pred",
                          **backtest(races, test_days, use)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

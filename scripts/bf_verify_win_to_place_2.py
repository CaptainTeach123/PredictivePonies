"""Adversarial verification #2 of experiment "win_to_place" (scripts/bf_win_to_place.py),
lens: LEAKAGE AND DATA QUALITY, for the k = 0.2 cell.

CLAIM under test: back the PLACE at ltp_pre when ltp_pre * p_cal > 1.20 (p_cal =
calibrated discounted-Harville place probability derived from the WIN market's
ltp_pre): 48,075 bets, +17.9% ROI after 5% commission (SE 2.5%), halves
[+18.9%, +17.2%], one tick worse +15.0%; "paper-only": the same decisions
settled at the place BSP return -5.9% (SE 2.8%), halves [-0.4%, -9.4%].

    python scripts/bf_verify_win_to_place_2.py [--dir data/raw/betfair/split] [--frac 0.1]

The original pipeline is imported unchanged (same join, lambda fit, calibration)
so the headline cell is reproduced exactly; the script then adds what the
original did not run:

  0. raw-file audits: removed runners, duplicated races (same venue+start or
     identical runner set under another market id), places paid vs field size
     at the off (3 places with < 8 runners = late scratching after the market
     loaded), dead heats;
  1. reproduction of the k=0.2 cells (all / gallops / harness) with per-bet,
     race- and day-clustered SE;
  2. leakage tests by label permutation: shuffle the placed labels of one
     month and recompute lambdas and p_cal; nothing dated at or before that
     month may change (walk-forward), and the lambdas never see the permuted
     month;
  3. staleness markers on the selected bets vs blind: no trade in the last 30
     min (ltp_t30 == ltp_pre), under-round place book at ltp_pre, place price
     >= win price (impossible in a consistent book), ltp_pre / BSP; ROI at
     ltp_pre and at BSP by marker; decomposition of the paper ROI into the
     price ratio and the BSP-settled ROI;
  4. scratching proxies: 3 places paid with < 8 runners at the off (2 with
     < 5), saddlecloth-number gaps; share among selected vs blind, ROI by proxy;
  5. model-free null rules (n_winners * p_win, plain Harville lam = 1, place
     price >= win price, and the post-hoc "ltp_pre >= 1.2 x BSP"): does any
     "fair" price give the same ltp_pre edge and the same BSP loss?
  6. corrected executable estimate: decide at ltp_pre, settle at the place
     BSP, at 0/5/8% commission with per-bet and day-clustered SE and halves;
     worse-of (limit at ltp_pre if touched, else SP); subsets.
Prints summary numbers only (never horse names or rows).
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("w2p", ROOT / "scripts" / "bf_win_to_place.py")
w2p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w2p)

K = 0.2
CLAIM = {"bets": 48075, "roi_c0": 0.230, "roi_c5": 0.1794, "roi_c8": 0.149, "se": 0.0251, "halves": [0.189, 0.1723],
         "one_tick_worse": 0.150, "settle_bsp": -0.059, "bsp_se": 0.028, "bsp_halves": [-0.004, -0.094],
         "worse_of": -0.106, "median_place_bsp_over_ltp": 0.803, "median_win_bsp_over_ltp": 1.000, "bsp_cover": 0.62}


def out(obj) -> None:
    print(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)), flush=True)


def r4(x):
    return w2p.r4(x)


def cluster_se(ret: np.ndarray, cl) -> float | None:
    ok = ~np.isnan(ret)
    ret, cl = ret[ok], np.asarray(cl)[ok]
    if len(ret) < 2:
        return None
    d = pd.Series(ret - ret.mean()).groupby(cl).sum().to_numpy()
    return float(math.sqrt((d ** 2).sum()) / len(ret))


def basic(ret: np.ndarray) -> dict:
    ret = ret[~np.isnan(ret)]
    if len(ret) < 2:
        return {"bets": int(len(ret)), "roi": None, "se": None}
    se = ret.std(ddof=1) / math.sqrt(len(ret))
    return {"bets": int(len(ret)), "roi": r4(ret.mean()), "se": r4(se)}


def ret(g: pd.DataFrame, price_col: str, c: float) -> np.ndarray:
    return w2p.returns(g[price_col], g["placed"], g["dh"], c)


def cell(g: pd.DataFrame, mid, prob_col="p_cal", price_col="ltp_pl", full=True) -> dict:
    """bets, ROI c0/c5/c8 at price_col, per-bet / race / day clustered SE, halves,
    one tick worse, the same decisions settled at the place BSP (+SE, halves),
    worse-of, price-ratio medians, strike."""
    res = {"bets": int(len(g)), "races": int(g["race"].nunique()) if len(g) else 0}
    if len(g) < 2:
        return res
    for c in w2p.COMMS:
        r = ret(g, price_col, c)
        res[f"roi_c{int(round(c * 100))}"] = r4(r.mean())
        if c == 0.05:
            res["se"] = r4(r.std(ddof=1) / math.sqrt(len(r)))
            res["se_cl_race"] = r4(cluster_se(r, g["race"].to_numpy()))
            res["se_cl_date"] = r4(cluster_se(r, g["date"].to_numpy()))
            fh = (g["date"] < mid).to_numpy()
            res["halves_c5"] = [basic(r[fh])["roi"], basic(r[~fh])["roi"]]
            res["halves_se_c5"] = [basic(r[fh])["se"], basic(r[~fh])["se"]]
    res["strike"] = r4(g["placed"].mean())
    res["mean_p"] = r4(g[prob_col].mean()) if prob_col in g else None
    res["median_price"] = r4(g[price_col].median())
    if not full:
        return res
    res["roi_c5_one_tick_worse"] = r4(w2p.returns(w2p.one_tick_worse(g[price_col]), g["placed"], g["dh"], 0.05).mean())
    has = (g["bsp_pl"].notna() & (g["bsp_pl"] > 1)).to_numpy()
    res["bsp_cover"] = r4(has.mean())
    if has.sum() > 1:
        gb = g[has]
        rb = ret(gb, "bsp_pl", 0.05)
        res["roi_c5_settle_bsp"] = r4(rb.mean())
        res["bsp_se"] = r4(rb.std(ddof=1) / math.sqrt(len(rb)))
        res["bsp_se_cl_date"] = r4(cluster_se(rb, gb["date"].to_numpy()))
        fhb = (gb["date"] < mid).to_numpy()
        res["bsp_halves_c5"] = [basic(rb[fhb])["roi"], basic(rb[~fhb])["roi"]]
        res["roi_c0_settle_bsp"] = r4(ret(gb, "bsp_pl", 0.0).mean())
        res["roi_c8_settle_bsp"] = r4(ret(gb, "bsp_pl", 0.08).mean())
        res["roi_c5_ltp_on_bsp_subset"] = r4(ret(gb, price_col, 0.05).mean())
        res["roi_c5_worse_of_ltp_bsp"] = r4(w2p.returns(np.minimum(gb[price_col], gb["bsp_pl"]), gb["placed"], gb["dh"], 0.05).mean())
        res["median_place_bsp_over_ltp"] = r4((gb["bsp_pl"] / gb["ltp_pl"]).median())
        hw = gb["bsp_win"].notna() & (gb["bsp_win"] > 1)
        res["median_win_bsp_over_ltp"] = r4((gb.loc[hw, "bsp_win"] / gb.loc[hw, "ltp_win"]).median()) if hw.any() else None
        if prob_col in g:
            res["median_p_x_bsp"] = r4((gb[prob_col] * gb["bsp_pl"]).median())
        res["strike_bsp_subset"] = r4(gb["placed"].mean())
        res["mean_1_over_bsp"] = r4((1 / gb["bsp_pl"]).mean())
        res["mean_1_over_ltp_bsp_subset"] = r4((1 / gb[price_col]).mean())
    return res


def slim(c: dict, keys=("bets", "roi_c5", "se", "se_cl_date", "halves_c5", "roi_c5_settle_bsp", "bsp_se", "bsp_halves_c5",
                        "roi_c5_worse_of_ltp_bsp", "median_place_bsp_over_ltp", "median_win_bsp_over_ltp", "strike", "mean_p",
                        "bsp_cover")) -> dict:
    return {k: c.get(k) for k in keys}


def fit_all(j: pd.DataFrame):
    """The original pipeline from the joined table to p_fair / p_cal (unchanged code)."""
    P, jw, k, i, ok, nw, race = w2p.placed_set_stats(j)
    months = race["month"].to_numpy()
    codes = race["harness"].to_numpy()
    months_sorted = sorted(set(months))
    ll = w2p.grid_loglik(P, jw, k, i, ok, nw)
    lam = w2p.fit_lambdas_walk_forward(ll, ok, months, codes, months_sorted)
    del ll
    p_fair = w2p.derived_place_probs(j, P, nw, lam, months, codes)
    return lam, p_fair, P, nw, months, codes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/raw/betfair/split"))
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    # ---- 0. raw-file audits ---------------------------------------------------------------------
    audit = {}
    raw = {}
    for f in ("au_place", "au_win"):
        x = pd.read_parquet(a.dir / f"{f}.parquet", columns=["market_id", "event_id", "venue", "start", "n_winners", "field_size",
                                                           "selection_id", "program", "won", "ltp_pre", "bsp"])
        n0 = len(x)
        x = x.drop_duplicates(["market_id", "selection_id"])
        g = x.groupby("market_id")
        rows = g.size()
        m = x.drop_duplicates("market_id").set_index("market_id")
        dd = m.groupby(["venue", "start"]).agg(n=("event_id", "size"), ne=("event_id", "nunique"))
        sel_set = g["selection_id"].apply(lambda v: tuple(sorted(v)))
        key = pd.DataFrame({"sel": sel_set.values, "start": m.loc[sel_set.index, "start"].values})
        audit[f] = {"rows": int(n0), "rows_after_dedupe": int(len(x)), "markets": int(len(rows)),
                    "markets_rows_ne_field_size": int((rows != m["field_size"].reindex(rows.index)).sum()),
                    "won_nan_rows": int(x["won"].isna().sum()),
                    "markets_sharing_venue_start": int((dd["n"] > 1).sum()),
                    "markets_sharing_start_and_identical_runner_set": int(key.duplicated(["start", "sel"], keep=False).sum()),
                    "share_ltp_pre_gt_bsp_when_both": r4((x["ltp_pre"] > x["bsp"])[x["ltp_pre"].notna() & x["bsp"].notna()].mean())}
        if f == "au_place":
            fs = m["field_size"]; nw_ = m["n_winners"]; npl = g["won"].sum().reindex(m.index)
            audit[f].update({"places_paid_dist": nw_.value_counts().to_dict(),
                             "markets_3_places_lt_8_runners_at_off": int(((nw_ == 3) & (fs < 8)).sum()),
                             "markets_2_places_lt_5_runners_at_off": int(((nw_ == 2) & (fs < 5)).sum()),
                             "markets_2_places_ge_8_runners": int(((nw_ == 2) & (fs >= 8)).sum()),
                             "markets_n_placed_gt_n_winners_dead_heat": int((npl > nw_).sum()),
                             "markets_n_placed_lt_n_winners": int((npl < nw_).sum())})
            raw["prog"] = x[["market_id", "selection_id", "program"]].rename(columns={"market_id": "pl_mid"})
        del x
    out({"section": "0_raw_audit", "note": "parser drops runners with status REMOVED, so scratched runners are absent and "
         "field_size == rows; a place market loaded with 8+ runners keeps 3 places when scratchings cut the field below 8 (late "
         "scratchings re-price the WIN book immediately but the thin PLACE book's last trade may predate them)", **audit})

    # ---- identical pipeline to the original main() up to p_cal ---------------------------------
    j = w2p.load(a.dir, a.frac, a.seed)
    lam, p_fair, P, nw, months, codes = fit_all(j)
    j["p_fair"] = p_fair
    lam1 = {key: (1.0, 1.0, v[2]) for key, v in lam.items()}
    j["p_harv1"] = w2p.derived_place_probs(j, P, nw, lam1, months, codes)
    keep = j["p_fair"].notna().to_numpy()
    j = j[keep].copy()
    j["race"] = pd.factorize(j["win_mid"])[0]
    j["dh"] = j["n_winners"] / j["n_placed"]
    months_sorted = sorted(j["month"].unique())
    j["p_cal"], cal_params = w2p.calibrate_walk_forward(j, "p_fair", months_sorted)
    j["p_crude"] = (j["n_winners"] * j["p_win"]).clip(upper=0.98)
    days = sorted(j["date"].unique())
    mid = days[len(days) // 2]
    grp = j.groupby("race")
    j["priced_pl"] = j["ltp_pl"].notna() & (j["ltp_pl"] > 1)
    j["n_priced_pl"] = grp["priced_pl"].transform("sum")
    j["all_priced"] = j["n_priced_pl"] == j["n"]
    inv = (1 / j["ltp_pl"]).where(j["priced_pl"], 0.0)
    j["book_lb"] = inv.groupby(j["race"]).transform("sum") / j["n_winners"]
    j["has_bsp"] = j["bsp_pl"].notna() & (j["bsp_pl"] > 1)
    j["t30_state"] = np.where(j["t30_pl"].isna(), "t30_missing",
                              np.where(np.isclose(j["t30_pl"], j["ltp_pl"]), "t30_equals_ltp_pre", "t30_differs"))
    j["pl_ge_win"] = j["priced_pl"] & (j["ltp_pl"] >= j["ltp_win"])
    j["late_scratch"] = ((j["n_winners"] == 3) & (j["n"] < 8)) | ((j["n_winners"] == 2) & (j["n"] < 5))
    prog = raw["prog"].drop_duplicates(["pl_mid", "selection_id"]).copy()
    for c in ("pl_mid", "selection_id"):
        prog[c] = prog[c].astype(object)
    prog["prog_num"] = pd.to_numeric(prog["program"], errors="coerce")
    n_before = len(j)
    j = j.merge(prog[["pl_mid", "selection_id", "prog_num"]], on=["pl_mid", "selection_id"], how="left")
    assert len(j) == n_before
    grp = j.groupby("race")
    j["gap"] = (grp["prog_num"].transform("max") > j["n"]) & (grp["prog_num"].transform("count") == j["n"])
    j["year"] = j["date"].dt.year
    race1 = j.drop_duplicates("race")
    out({"section": "0_joined_data", "runtime_s": round(time.time() - t0, 1), "joined_races": int(j["race"].nunique()),
         "runners": int(len(j)), "mid": str(pd.Timestamp(mid).date()), "from": str(pd.Timestamp(j["date"].min()).date()),
         "to": str(pd.Timestamp(j["date"].max()).date()), "p_cal_coverage": r4(j["p_cal"].notna().mean()),
         "last_lambdas": {("harness" if c else "gallops"): v[:2] for (m, c), v in lam.items() if m == max(months_sorted)},
         "cal_params_last": cal_params[max(cal_params)] if cal_params else None,
         "races_late_scratch_proxy": r4(race1["late_scratch"].mean()), "races_saddlecloth_gap": r4(race1["gap"].mean()),
         "races_fully_priced_place": r4(race1["all_priced"].mean()),
         "runners_place_price_ge_win_price": r4(j.loc[j["priced_pl"], "pl_ge_win"].mean()),
         "joined_races_sharing_start_and_runner_set": int(j.groupby("race")["selection_id"].apply(lambda v: tuple(sorted(v)))
                                                           .to_frame("sel").join(race1.set_index("race")["start"]).duplicated(keep=False).sum())})

    bettable = j[j["priced_pl"]].copy()
    bettable["edge"] = bettable["ltp_pl"] * bettable["p_cal"] - 1
    sel = bettable[bettable["edge"] > K].copy()
    hb = sel[sel["has_bsp"]].copy()

    # ---- 1. reproduction ---------------------------------------------------------------------------
    head = cell(sel, mid)
    rep = {"section": "1_reproduction_k0.2_all_p_cal", **head, "claim": CLAIM,
           "bets_per_race_in_selected_races": r4(len(sel) / max(sel["race"].nunique(), 1)),
           "z_c5_per_bet": r4(head["roi_c5"] / head["se"]) if head.get("se") else None,
           "z_c5_day_clustered": r4(head["roi_c5"] / head["se_cl_date"]) if head.get("se_cl_date") else None,
           "bsp_z_day_clustered": r4(head["roi_c5_settle_bsp"] / head["bsp_se_cl_date"]) if head.get("bsp_se_cl_date") else None}
    if a.frac >= 1:
        rep["matches_claim_bets"] = bool(head["bets"] == CLAIM["bets"])
        rep["matches_claim_roi_c5_within_0.002"] = bool(abs(head["roi_c5"] - CLAIM["roi_c5"]) < 0.002)
    out(rep)
    for name, mask in {"gallops": ~sel["harness"], "harness": sel["harness"]}.items():
        out({"section": f"1_reproduction_k0.2_{name}", **cell(sel[mask], mid)})
    out({"section": "1_reference_blind_back_all_priced_place_runners", **slim(cell(bettable, mid))})

    # ---- 2. leakage tests by label permutation --------------------------------------------------------
    rng = np.random.default_rng(a.seed)
    jj = j.copy()
    m_last = months_sorted[-1]
    m_mid = months_sorted[len(months_sorted) // 2]
    leak = {}
    # (a) calibration: permute placed labels within month m_mid; p_cal must be unchanged for months <= m_mid
    #     and may change only for the 12 months after it (the training window)
    for tag, m_perm in (("permute_last_month", m_last), ("permute_middle_month", m_mid)):
        jp = jj.copy()
        sel_m = (jp["month"] == m_perm).to_numpy()
        lab = jp.loc[sel_m, "placed"].to_numpy().copy()
        rng.shuffle(lab)
        jp.loc[sel_m, "placed"] = lab
        pc, _ = w2p.calibrate_walk_forward(jp, "p_fair", months_sorted)
        d = np.abs(pc - jj["p_cal"].to_numpy())
        mon = jj["month"].to_numpy()
        both = ~np.isnan(d)
        after = [m for m in months_sorted if m > m_perm]
        within = set(after[:12])
        leak[tag] = {"month": m_perm, "max_abs_diff_p_cal_months_le_perm": r4(np.nanmax(d[(mon <= m_perm) & both])) if ((mon <= m_perm) & both).any() else 0.0,
                     "max_abs_diff_p_cal_within_12m_after": r4(np.nanmax(d[np.isin(mon, list(within)) & both])) if within else None,
                     "max_abs_diff_p_cal_beyond_12m_after": r4(np.nanmax(d[np.isin(mon, after[12:]) & both])) if len(after) > 12 else None}
    # (b) lambdas: permute placed labels within the last month -> no lambda may change; within the middle
    #     month -> lambdas for months <= m_mid unchanged
    j0 = w2p.load(a.dir, a.frac, a.seed)
    for tag, m_perm in (("permute_last_month", m_last), ("permute_middle_month", m_mid)):
        jp = j0.copy()
        sel_m = (jp["month"] == m_perm).to_numpy()
        # permute the placed set within each race of that month (keeps n_placed per race)
        part = jp.loc[sel_m, ["race", "placed"]].copy()
        part["placed"] = part.groupby("race")["placed"].transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1 << 30))).to_numpy())
        jp.loc[sel_m, "placed"] = part["placed"].to_numpy()
        lam_p, _, _, _, _, _ = fit_all(jp)
        changed_le = sum(1 for key in lam if key[0] <= m_perm and lam_p.get(key, (None, None))[:2] != lam[key][:2])
        changed_gt = sum(1 for key in lam if key[0] > m_perm and lam_p.get(key, (None, None))[:2] != lam[key][:2])
        leak[tag]["lambda_keys_changed_months_le_perm"] = changed_le
        leak[tag]["lambda_keys_changed_months_gt_perm"] = changed_gt
        leak[tag]["lambda_keys_total"] = len(lam)
    del j0, jj
    # (c) the inputs of p_cal: correlation of p_cal with the same race's BSP-implied place probability,
    #     compared with p_fair (a look-ahead quantity entering p_cal would show as extra BSP correlation)
    fb = j[j["has_bsp"] & j["p_cal"].notna() & j["priced_pl"]].copy()
    fb["q_bsp_raw"] = 1 / fb["bsp_pl"]
    leak["corr_with_1_over_place_bsp"] = {"p_fair": r4(np.corrcoef(fb["p_fair"], fb["q_bsp_raw"])[0, 1]),
                                          "p_cal": r4(np.corrcoef(fb["p_cal"], fb["q_bsp_raw"])[0, 1]),
                                          "1_over_ltp_pl": r4(np.corrcoef(1 / fb["ltp_pl"], fb["q_bsp_raw"])[0, 1])}
    out({"section": "2_leakage_tests", **leak,
         "note": "p_cal for month m is unchanged by permuting labels of months >= m; lambdas never see the permuted month: "
                 "no same-month fitting. The remaining data-quality question is the PRICE, not the probability."})

    # ---- 3. staleness markers ---------------------------------------------------------------------------
    rows = []
    for st in ("t30_missing", "t30_differs", "t30_equals_ltp_pre"):
        g = sel[sel["t30_state"] == st]
        rows.append({"state": st, "share_selected": r4(len(g) / len(sel)), "share_blind": r4((bettable["t30_state"] == st).mean()),
                     **slim(cell(g, mid))})
    out({"section": "3a_selected_bets_by_t30_state", "rows": rows,
         "note": "t30_equals_ltp_pre = no trade in the final 30 minutes (price at least 30 min old); t30_missing = no trade "
                 "at all 30 min before the off (the last trade is somewhere in the final 30 minutes, age unknown)"})
    rows = []
    sf = sel[sel["all_priced"]]
    for lo, hi in [(0, 0.85), (0.85, 0.95), (0.95, 1.0), (1.0, 1.05), (1.05, 1.15), (1.15, 9)]:
        g = sf[(sf["book_lb"] >= lo) & (sf["book_lb"] < hi)]
        rows.append({"ltp_place_book_over_n_winners": f"{lo:g}-{hi:g}", **slim(cell(g, mid))})
    bf_ = bettable[bettable["all_priced"]]
    out({"section": "3b_selected_bets_fully_priced_races_by_ltp_place_book", "rows": rows,
         "share_selected_in_fully_priced_races": r4(sel["all_priced"].mean()),
         "share_selected_fully_priced_in_under_round_book": r4((sf["book_lb"] < 1).mean()),
         "share_blind_fully_priced_in_under_round_book": r4((bf_["book_lb"] < 1).mean()),
         "median_book_selected": r4(sf["book_lb"].median()), "median_book_blind": r4(bf_["book_lb"].median()),
         "note": "sum(1/ltp_pre) over the place book < n_winners is a set of prices that cannot coexist (a bettor could back "
                 "every runner for a sure profit): direct evidence that the quotes are from different times"})
    # impossible prices: place price >= win price of the same runner
    out({"section": "3c_place_price_ge_win_price_impossible_quotes",
         "share_selected": r4(sel["pl_ge_win"].mean()), "share_blind": r4(bettable["pl_ge_win"].mean()),
         "selected_with_impossible_quote": slim(cell(sel[sel["pl_ge_win"]], mid)),
         "selected_without": slim(cell(sel[~sel["pl_ge_win"]], mid)),
         "selected_place_ltp_over_win_ltp_quantiles": {str(q): r4((sel["ltp_pl"] / sel["ltp_win"]).quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
         "blind_place_ltp_over_win_ltp_median": r4((bettable["ltp_pl"] / bettable["ltp_win"]).median()),
         "selected_place_bsp_over_win_bsp_median": r4((hb["bsp_pl"] / hb["bsp_win"]).median())})
    # ltp_pre / BSP
    hb["ratio"] = hb["ltp_pl"] / hb["bsp_pl"]
    rows = []
    for lo, hi in [(0, 0.9), (0.9, 0.98), (0.98, 1.02), (1.02, 1.1), (1.1, 1.25), (1.25, 1.5), (1.5, 2.0), (2.0, 99)]:
        g = hb[(hb["ratio"] >= lo) & (hb["ratio"] < hi)]
        c = cell(g, mid)
        rows.append({"ltp_over_bsp": f"{lo:g}-{hi:g}", "bets": c.get("bets"), "share": r4(len(g) / len(hb)),
                     "roi_c5_at_ltp": c.get("roi_c5"), "se": c.get("se"), "roi_c5_at_bsp": c.get("roi_c5_settle_bsp"),
                     "strike": c.get("strike"), "mean_p_cal": c.get("mean_p"), "mean_1_over_bsp": c.get("mean_1_over_bsp")})
    bb = bettable[bettable["has_bsp"]]
    # decomposition of the paper ROI on the BSP subset
    r_ltp = ret(hb, "ltp_pl", 0.05)
    r_bsp = ret(hb, "bsp_pl", 0.05)
    won = (hb["placed"] == 1).to_numpy()
    out({"section": "3d_selected_bets_with_bsp_by_ltp_over_bsp", "rows": rows,
         "share_ltp_above_bsp_selected": r4((hb["ratio"] > 1).mean()),
         "share_ltp_above_bsp_blind": r4((bb["ltp_pl"] > bb["bsp_pl"]).mean()),
         "quantiles_ltp_over_bsp_selected": {str(q): r4(hb["ratio"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
         "quantiles_ltp_over_bsp_blind": {str(q): r4((bb["ltp_pl"] / bb["bsp_pl"]).quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
         "decomposition_bsp_subset": {"bets": int(len(hb)), "roi_c5_at_ltp": r4(r_ltp.mean()), "roi_c5_at_bsp": r4(r_bsp.mean()),
                                      "mean_ltp_over_bsp_on_winning_bets": r4(hb.loc[won, "ratio"].mean()),
                                      "mean_ltp_over_bsp_all": r4(hb["ratio"].mean()),
                                      "gross_return_ltp_over_gross_return_bsp": r4((1 + r_ltp.mean()) / (1 + r_bsp.mean())),
                                      "strike": r4(won.mean()), "mean_1_over_bsp": r4((1 / hb["bsp_pl"]).mean()),
                                      "mean_1_over_ltp": r4((1 / hb["ltp_pl"]).mean()), "mean_p_cal": r4(hb["p_cal"].mean())},
         "note": "the paper ROI equals the BSP-settled ROI scaled by the ltp/BSP price ratio of the winning bets: the "
                 "selected runners place at the rate the BSP implies, the ltp_pre is simply higher than the BSP"})

    # ---- 4. scratching proxies ------------------------------------------------------------------------------
    rows = {}
    for name, m in {"late_scratch_proxy(3 places & <8 runners, or 2 & <5)": sel["late_scratch"],
                    "no_late_scratch_proxy": ~sel["late_scratch"],
                    "saddlecloth_gap": sel["gap"], "no_saddlecloth_gap": ~sel["gap"],
                    "either_proxy": sel["late_scratch"] | sel["gap"], "neither_proxy": ~(sel["late_scratch"] | sel["gap"])}.items():
        rows[name] = {"share_selected": r4(m.mean()), **slim(cell(sel[m], mid))}
    blind_share = {"late_scratch_proxy": r4(bettable["late_scratch"].mean()), "saddlecloth_gap": r4(bettable["gap"].mean()),
                   "either": r4((bettable["late_scratch"] | bettable["gap"]).mean())}
    out({"section": "4_selected_bets_by_scratching_proxy", "rows": rows, "blind_share": blind_share,
         "blind_roi_c5_late_scratch_proxy": slim(cell(bettable[bettable["late_scratch"]], mid), ("bets", "roi_c5", "roi_c5_settle_bsp", "median_place_bsp_over_ltp")),
         "blind_roi_c5_no_late_scratch_proxy": slim(cell(bettable[~bettable["late_scratch"]], mid), ("bets", "roi_c5", "roi_c5_settle_bsp", "median_place_bsp_over_ltp"))})

    # ---- 5. model-free null rules -------------------------------------------------------------------------------
    rows = []
    for pcol in ("p_cal", "p_fair", "p_harv1", "p_crude"):
        e = (bettable["ltp_pl"] * bettable[pcol] - 1).to_numpy()
        g = bettable[e > K]
        c = cell(g, mid, prob_col=pcol)
        c["rule"] = f"ltp_pl*{pcol}>1.2"
        gneg = bettable[e < -K]
        c2 = cell(gneg, mid, prob_col=pcol)
        c["mirror_market_shorter_than_p_by_20pct_backed_anyway"] = slim(c2, ("bets", "roi_c5", "se", "roi_c5_settle_bsp", "median_place_bsp_over_ltp", "median_win_bsp_over_ltp"))
        rows.append(slim(c, ("rule", "bets", "roi_c0", "roi_c5", "roi_c8", "se", "halves_c5", "roi_c5_one_tick_worse", "roi_c5_settle_bsp", "bsp_se",
                             "bsp_halves_c5", "roi_c5_worse_of_ltp_bsp", "median_place_bsp_over_ltp", "median_win_bsp_over_ltp", "median_p_x_bsp",
                             "strike", "mean_p", "mirror_market_shorter_than_p_by_20pct_backed_anyway")))
    # no model at all: the place price is at or above the runner's own win price
    g = bettable[bettable["pl_ge_win"]]
    c = cell(g, mid)
    c["rule"] = "ltp_pl >= ltp_win (no model; impossible quote)"
    rows.append(slim(c, ("rule", "bets", "roi_c0", "roi_c5", "roi_c8", "se", "halves_c5", "roi_c5_settle_bsp", "bsp_se", "bsp_halves_c5",
                         "roi_c5_worse_of_ltp_bsp", "median_place_bsp_over_ltp", "median_win_bsp_over_ltp", "strike")))
    # post-hoc diagnostic (BSP is not known pre-race): the same k applied to the BSP as "fair" price
    g = bb[bb["ltp_pl"] >= 1.2 * bb["bsp_pl"]]
    c = cell(g, mid)
    c["rule"] = "ltp_pl >= 1.2 * place_BSP (post hoc, not a strategy)"
    rows.append(slim(c, ("rule", "bets", "roi_c0", "roi_c5", "roi_c8", "se", "halves_c5", "roi_c5_settle_bsp", "bsp_se", "bsp_halves_c5",
                         "median_place_bsp_over_ltp", "median_win_bsp_over_ltp", "strike")))
    # k grid for p_cal at ltp and at BSP
    grid = []
    for kk in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6):
        g = bettable[bettable["edge"] > kk]
        c = cell(g, mid)
        grid.append({"k": kk, **slim(c, ("bets", "roi_c5", "se", "halves_c5", "roi_c5_settle_bsp", "bsp_se", "bsp_halves_c5",
                                         "roi_c5_worse_of_ltp_bsp", "median_place_bsp_over_ltp", "median_win_bsp_over_ltp", "bsp_cover"))})
    out({"section": "5_null_models_and_k_grid", "rows": rows, "k_grid_p_cal": grid,
         "note": "if the 'edge' were information in p_cal, the BSP-settled ROI should rise with k; if it is staleness, "
                 "the ltp/BSP gap widens with k and the BSP ROI does not improve"})

    # ---- 6. corrected executable estimate --------------------------------------------------------------------------
    rb5 = ret(hb, "bsp_pl", 0.05)
    rw = w2p.returns(np.minimum(hb["ltp_pl"], hb["bsp_pl"]), hb["placed"], hb["dh"], 0.05)
    touched = hb[hb["ratio"] <= 1.0]
    rt = ret(touched, "ltp_pl", 0.05)
    nob = sel[~sel["has_bsp"]]
    by_year = []
    for y, g in hb.groupby("year"):
        by_year.append({"year": int(y), "bsp_bets": int(len(g)), "roi_c5_bsp": r4(ret(g, "bsp_pl", 0.05).mean()),
                        "roi_c5_ltp": r4(ret(g, "ltp_pl", 0.05).mean()), "median_place_bsp_over_ltp": r4((g["bsp_pl"] / g["ltp_pl"]).median())})
    out({"section": "6_corrected_estimate",
         "paper_at_ltp_pre": slim(head, ("bets", "roi_c0", "roi_c5", "roi_c8", "se", "se_cl_date", "halves_c5")),
         "executable_sp_bet_decide_ltp_settle_bsp": {"bets": int(len(hb)), "bsp_cover": r4(len(hb) / len(sel)),
                                                     "roi_c0": r4(ret(hb, "bsp_pl", 0.0).mean()), "roi_c5": r4(rb5.mean()),
                                                     "roi_c8": r4(ret(hb, "bsp_pl", 0.08).mean()),
                                                     "se_per_bet": r4(rb5.std(ddof=1) / math.sqrt(len(rb5))),
                                                     "se_cl_date": r4(cluster_se(rb5, hb["date"].to_numpy())),
                                                     "se_cl_race": r4(cluster_se(rb5, hb["race"].to_numpy())),
                                                     "halves": head.get("bsp_halves_c5"), "by_year": by_year,
                                                     "gallops": slim(cell(hb[~hb["harness"]], mid), ("bets", "roi_c5_settle_bsp", "bsp_se", "bsp_halves_c5")),
                                                     "harness": slim(cell(hb[hb["harness"]], mid), ("bets", "roi_c5_settle_bsp", "bsp_se", "bsp_halves_c5"))},
         "limit_at_ltp_if_touched_else_sp(worse_of)": {**basic(rw), "se_cl_date": r4(cluster_se(rw, hb["date"].to_numpy())),
                                                       "roi_c0": r4(w2p.returns(np.minimum(hb["ltp_pl"], hb["bsp_pl"]), hb["placed"], hb["dh"], 0.0).mean()),
                                                       "roi_c8": r4(w2p.returns(np.minimum(hb["ltp_pl"], hb["bsp_pl"]), hb["placed"], hb["dh"], 0.08).mean())},
         "limit_at_ltp_filled_iff_bsp_ge_ltp": {"fill_rate": r4(len(touched) / len(hb)), **basic(rt)},
         "paper_only_no_bsp_subset_at_ltp": slim(cell(nob, mid, full=False), ("bets", "roi_c5", "se", "halves_c5", "strike", "mean_p")),
         "runtime_s": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

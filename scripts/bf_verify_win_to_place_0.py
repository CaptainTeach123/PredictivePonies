"""Adversarial verification of experiment "win_to_place" (scripts/bf_win_to_place.py).

CLAIM under test: back the PLACE at ltp_pre when ltp_pre * p_cal > 1.10 (p_cal =
calibrated discounted-Harville place probability from the WIN market's ltp_pre);
+10.8% ROI after 5% commission on 107,224 bets, > 2 SE in both halves, but
negative when the same bets are settled at the place BSP.

    python scripts/bf_verify_win_to_place_0.py [--dir data/raw/betfair/split] [--frac 0.1]

This script imports the original pipeline unchanged (same join, lambda fit,
calibration) so that the headline cell is reproduced exactly, then adds the
diagnostics the original did not run, under the leakage / data-quality lens:

  1. race-clustered SE (several bets per race are not independent);
  2. the place book implied by ltp_pre in the bet's race: an under-round book
     (sum of 1/ltp < n_winners) is a set of prices that could not have been
     available at the same moment, i.e. direct evidence of stale quotes;
  3. staleness marker ltp_t30 == ltp_pre (no trade in the last 30 minutes);
  4. scratching proxy (gap in saddlecloth numbers: a removed runner means the
     thin place market's last trade may predate the reform of the market);
  5. model-free null "fair" probabilities (plain Harville lam=1, and
     n_winners * p_win): does the ltp "edge" appear whatever p is used?;
  6. where the paper profit sits as a function of ltp_pre / BSP (post-hoc
     diagnostic, not a strategy);
  7. winner's curse on p_cal: p_cal * BSP vs realised strike on the bets.
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

K = 0.1


def out(obj) -> None:
    print(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)), flush=True)


def r4(x):
    return w2p.r4(x)


def cluster_se(ret: np.ndarray, race: np.ndarray) -> float | None:
    """SE of the mean return with bets clustered by race."""
    ok = ~np.isnan(ret)
    ret, race = ret[ok], race[ok]
    if len(ret) < 2:
        return None
    m = ret.mean()
    d = pd.Series(ret - m).groupby(race).sum().to_numpy()
    return float(math.sqrt((d ** 2).sum()) / len(ret))


def cell(g: pd.DataFrame, mid, prob_col="p_cal", price_col="ltp_pl") -> dict:
    """Compact bet cell: bets, ROI at 0/5/8%, per-bet SE, race-clustered SE,
    halves, one tick worse, settled at place BSP (same decision), BSP coverage."""
    res = {"bets": int(len(g))}
    if len(g) < 2:
        return res
    for c in w2p.COMMS:
        r = w2p.returns(g[price_col], g["placed"], g["dh"], c)
        res[f"roi_c{int(c * 100)}"] = r4(r.mean())
        if c == 0.05:
            res["se"] = r4(r.std(ddof=1) / math.sqrt(len(r)))
            res["se_race_clustered"] = r4(cluster_se(r, g["race"].to_numpy()))
            fh = (g["date"] < mid).to_numpy()
            res["halves_c5"] = [r4(r[fh].mean()) if fh.sum() > 1 else None, r4(r[~fh].mean()) if (~fh).sum() > 1 else None]
            res["halves_se_c5"] = [r4(r[fh].std(ddof=1) / math.sqrt(fh.sum())) if fh.sum() > 1 else None,
                                   r4(r[~fh].std(ddof=1) / math.sqrt((~fh).sum())) if (~fh).sum() > 1 else None]
    res["roi_c5_one_tick_worse"] = r4(w2p.returns(w2p.one_tick_worse(g[price_col]), g["placed"], g["dh"], 0.05).mean())
    has = (g["bsp_pl"].notna() & (g["bsp_pl"] > 1)).to_numpy()
    res["bsp_cover"] = r4(has.mean())
    if has.sum() > 1:
        gb = g[has]
        rb = w2p.returns(gb["bsp_pl"], gb["placed"], gb["dh"], 0.05)
        res["roi_c5_settle_bsp"] = r4(rb.mean())
        res["bsp_se"] = r4(rb.std(ddof=1) / math.sqrt(len(rb)))
        fhb = (gb["date"] < mid).to_numpy()
        res["bsp_halves_c5"] = [r4(rb[fhb].mean()) if fhb.sum() > 1 else None, r4(rb[~fhb].mean()) if (~fhb).sum() > 1 else None]
        res["roi_c5_ltp_on_bsp_subset"] = r4(w2p.returns(gb[price_col], gb["placed"], gb["dh"], 0.05).mean())
        res["roi_c5_worse_of_ltp_bsp"] = r4(w2p.returns(np.minimum(gb[price_col], gb["bsp_pl"]), gb["placed"], gb["dh"], 0.05).mean())
        res["median_place_bsp_over_ltp"] = r4((gb["bsp_pl"] / gb["ltp_pl"]).median())
        hw = gb["bsp_win"].notna() & (gb["bsp_win"] > 1)
        res["median_win_bsp_over_ltp"] = r4((gb.loc[hw, "bsp_win"] / gb.loc[hw, "ltp_win"]).median()) if hw.any() else None
        if prob_col in g:
            res["median_p_x_bsp"] = r4((gb[prob_col] * gb["bsp_pl"]).median())
            res["mean_p"] = r4(gb[prob_col].mean())
            res["strike_bsp_subset"] = r4(gb["placed"].mean())
            res["mean_1_over_bsp"] = r4((1 / gb["bsp_pl"]).mean())
    res["strike"] = r4(g["placed"].mean())
    res["median_price"] = r4(g[price_col].median())
    res["races"] = int(g["race"].nunique())
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/raw/betfair/split"))
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    # ---- identical pipeline to the original main() up to p_cal -----------------
    j = w2p.load(a.dir, a.frac, a.seed)
    P, jw, k, i, ok, nw, race = w2p.placed_set_stats(j)
    months = race["month"].to_numpy()
    codes = race["harness"].to_numpy()
    months_sorted = sorted(set(months))
    ll = w2p.grid_loglik(P, jw, k, i, ok, nw)
    lam = w2p.fit_lambdas_walk_forward(ll, ok, months, codes, months_sorted)
    del ll
    j["p_fair"] = w2p.derived_place_probs(j, P, nw, lam, months, codes)
    # null models: plain Harville (lam2 = lam3 = 1, nothing fitted) and n_winners * p_win
    lam1 = {key: (1.0, 1.0, v[2]) for key, v in lam.items()}
    j["p_harv1"] = w2p.derived_place_probs(j, P, nw, lam1, months, codes)
    j = j[j["p_fair"].notna()].copy()
    j["race"] = pd.factorize(j["win_mid"])[0]
    j["dh"] = j["n_winners"] / j["n_placed"]
    j["p_cal"], cal_params = w2p.calibrate_walk_forward(j, "p_fair", sorted(j["month"].unique()))
    j["p_crude"] = (j["n_winners"] * j["p_win"]).clip(upper=0.98)
    days = sorted(j["date"].unique())
    mid = days[len(days) // 2]

    # saddlecloth numbers (scratching proxy) from the place parquet
    prog = pd.read_parquet(a.dir / "au_place.parquet", columns=["market_id", "selection_id", "program"])
    prog = prog.drop_duplicates(["market_id", "selection_id"]).rename(columns={"market_id": "pl_mid"})
    for c in ("pl_mid", "selection_id"):
        prog[c] = prog[c].astype(object)
    prog["prog_num"] = pd.to_numeric(prog["program"], errors="coerce")
    n_before = len(j)
    j = j.merge(prog[["pl_mid", "selection_id", "prog_num"]], on=["pl_mid", "selection_id"], how="left")
    assert len(j) == n_before, (len(j), n_before)
    grp = j.groupby("race")
    j["max_prog"] = grp["prog_num"].transform("max")
    j["n_prog"] = grp["prog_num"].transform("count")
    j["gap"] = (j["max_prog"] > j["n"]) & (j["n_prog"] == j["n"])

    # place-market liquidity / consistency measures from ltp_pre only (pre-race information)
    j["priced_pl"] = j["ltp_pl"].notna() & (j["ltp_pl"] > 1)
    j["n_priced_pl"] = grp["priced_pl"].transform("sum")
    j["all_priced"] = j["n_priced_pl"] == j["n"]
    inv = (1 / j["ltp_pl"]).where(j["priced_pl"], 0.0)
    j["book_lb"] = inv.groupby(j["race"]).transform("sum") / j["n_winners"]  # lower bound of the place book / n_winners
    j["race_has_bsp"] = grp["bsp_pl"].transform(lambda s: s.notna().all() and (s > 1).all())
    j["t30_state"] = np.where(j["t30_pl"].isna(), "t30_missing",
                              np.where(np.isclose(j["t30_pl"], j["ltp_pl"]), "t30_equals_ltp_pre(no trade in last 30 min)", "t30_differs"))

    out({"section": "data", "runtime_s": round(time.time() - t0, 1), "joined_races": int(j["race"].nunique()),
         "runners": int(len(j)), "mid": str(pd.Timestamp(mid).date()),
         "from": str(pd.Timestamp(j["date"].min()).date()), "to": str(pd.Timestamp(j["date"].max()).date()),
         "p_cal_coverage": r4(j["p_cal"].notna().mean()), "place_ltp_coverage": r4(j["priced_pl"].mean()),
         "races_with_saddlecloth_gap": r4(j.drop_duplicates("race")["gap"].mean()),
         "races_fully_priced": r4(j.drop_duplicates("race")["all_priced"].mean()),
         "races_all_runners_have_place_bsp": r4(j.drop_duplicates("race")["race_has_bsp"].mean()),
         "last_lambdas": {("harness" if c else "gallops"): v[:2] for (m, c), v in lam.items() if m == max(months_sorted)},
         "cal_params_last": cal_params[max(cal_params)] if cal_params else None})

    bettable = j[j["priced_pl"]].copy()
    edge = (bettable["ltp_pl"] * bettable["p_cal"] - 1).to_numpy()
    sel = bettable[edge > K].copy()

    # ---- 1. reproduction of the headline cell -----------------------------------------
    head = cell(sel, mid)
    out({"section": "1_reproduction_k0.1_all_p_cal", **head,
         "bets_per_race_mean": r4(len(sel) / max(sel["race"].nunique(), 1)),
         "z_c5_per_bet": r4(head["roi_c5"] / head["se"]) if head.get("se") else None,
         "z_c5_race_clustered": r4(head["roi_c5"] / head["se_race_clustered"]) if head.get("se_race_clustered") else None})
    for name, mask in {"gallops": ~sel["harness"], "harness": sel["harness"]}.items():
        out({"section": f"1_reproduction_k0.1_{name}", **cell(sel[mask], mid)})
    # blind reference on the same price field
    out({"section": "1_reference_blind_back_all_priced", **cell(bettable, mid)})

    # ---- 2. under-round place books at ltp_pre ----------------------------------------
    rows = []
    sel_full = sel[sel["all_priced"]]
    for lo, hi in [(0, 0.90), (0.90, 0.95), (0.95, 1.0), (1.0, 1.05), (1.05, 1.15), (1.15, 9)]:
        g = sel_full[(sel_full["book_lb"] >= lo) & (sel_full["book_lb"] < hi)]
        rows.append({"book_over_nw": f"{lo:g}-{hi:g}", **cell(g, mid)})
    out({"section": "2_selected_bets_fully_priced_races_by_ltp_place_book", "rows": rows,
         "share_of_selected_bets_in_fully_priced_races": r4(sel["all_priced"].mean()),
         "share_fully_priced_selected_in_under_round_book": r4((sel_full["book_lb"] < 1).mean()),
         "share_fully_priced_blind_in_under_round_book": r4((bettable.loc[bettable["all_priced"], "book_lb"] < 1).mean()),
         "median_book_selected_fully_priced": r4(sel_full["book_lb"].median()),
         "median_book_all_fully_priced_races": r4(j.loc[j["all_priced"]].drop_duplicates("race")["book_lb"].median())})
    rows = []
    sel_part = sel[~sel["all_priced"]]
    sel_part = sel_part.assign(share_priced=sel_part["n_priced_pl"] / sel_part["n"])
    for lo, hi in [(0, 0.5), (0.5, 0.75), (0.75, 1.0)]:
        g = sel_part[(sel_part["share_priced"] >= lo) & (sel_part["share_priced"] < hi)]
        rows.append({"share_of_runners_priced": f"{lo:g}-{hi:g}", **cell(g, mid)})
    out({"section": "2_selected_bets_partially_priced_races_by_share_priced", "rows": rows})
    # a book that is over-round even at its lower bound is a consistent set of quotes
    cons = sel[sel["book_lb"] >= 1.0]
    incons = sel[sel["book_lb"] < 1.0]
    out({"section": "2_selected_bets_book_lower_bound_split",
         "book_lb_ge_1_consistent_quotes": cell(cons, mid), "book_lb_lt_1": cell(incons, mid)})
    # races with 2+ selected runners: two "10% value" place prices in one race force a stale book
    sel["n_sel_race"] = sel.groupby("race")["selection_id"].transform("size")
    out({"section": "2_selected_bets_by_number_selected_in_race",
         "one_selected": cell(sel[sel["n_sel_race"] == 1], mid), "two_plus_selected": cell(sel[sel["n_sel_race"] >= 2], mid)})

    # ---- 3. staleness marker: ltp_t30 vs ltp_pre --------------------------------------
    rows = []
    for st, g in sel.groupby("t30_state"):
        rows.append({"state": st, **cell(g, mid)})
    out({"section": "3_selected_bets_by_t30_state", "rows": rows,
         "blind_share_t30_equals_ltp_pre": r4((bettable["t30_state"].str.startswith("t30_equals")).mean()),
         "selected_share_t30_equals_ltp_pre": r4((sel["t30_state"].str.startswith("t30_equals")).mean())})

    # ---- 4. scratching proxy ------------------------------------------------------------
    out({"section": "4_selected_bets_by_saddlecloth_gap",
         "gap": cell(sel[sel["gap"]], mid), "no_gap": cell(sel[~sel["gap"]], mid),
         "blind_gap": cell(bettable[bettable["gap"]], mid), "blind_no_gap": cell(bettable[~bettable["gap"]], mid)})

    # ---- 5. null models: does any p give the same ltp edge? -----------------------------
    rows = []
    for pcol in ("p_cal", "p_fair", "p_harv1", "p_crude"):
        e = (bettable["ltp_pl"] * bettable[pcol] - 1).to_numpy()
        g = bettable[e > K]
        rows.append({"prob": pcol, **cell(g, mid, prob_col=pcol)})
        # mirror tail: market shorter than p by 10%, backed anyway
        gneg = bettable[e < -K]
        c2 = cell(gneg, mid, prob_col=pcol)
        rows[-1]["shorter_than_p_by_10pct_backed_anyway"] = {kk: c2.get(kk) for kk in
                                                             ("bets", "roi_c5", "se", "roi_c5_settle_bsp", "median_place_bsp_over_ltp", "median_win_bsp_over_ltp")}
    out({"section": "5_null_models_k0.1_all", "rows": rows})
    # same selected bets, place probability from the place market itself (fully priced races):
    # ltp*q_mkt is constant within a race, so instead report mean q_mkt vs strike for the selected bets
    fp = sel[sel["all_priced"]].copy()
    fp["q_mkt"] = (1 / fp["ltp_pl"]) * fp["n_winners"] / (fp["book_lb"] * fp["n_winners"])
    out({"section": "5_selected_fully_priced_probabilities", "bets": int(len(fp)), "strike": r4(fp["placed"].mean()),
         "mean_p_cal": r4(fp["p_cal"].mean()), "mean_q_mkt_ltp_normalised": r4(fp["q_mkt"].mean()),
         "mean_1_over_ltp": r4((1 / fp["ltp_pl"]).mean()),
         "mean_1_over_bsp_where_present": r4((1 / fp.loc[fp["bsp_pl"] > 1, "bsp_pl"]).mean())})

    # ---- 6. where the paper profit sits: ltp_pre vs BSP (post hoc) --------------------
    hb = sel[sel["bsp_pl"].notna() & (sel["bsp_pl"] > 1)].copy()
    hb["ratio"] = hb["ltp_pl"] / hb["bsp_pl"]
    rows = []
    for lo, hi in [(0, 0.9), (0.9, 0.98), (0.98, 1.02), (1.02, 1.1), (1.1, 1.25), (1.25, 1.5), (1.5, 99)]:
        g = hb[(hb["ratio"] >= lo) & (hb["ratio"] < hi)]
        c = cell(g, mid)
        rows.append({"ltp_over_bsp": f"{lo:g}-{hi:g}", "bets": c.get("bets"), "roi_c5_at_ltp": c.get("roi_c5"), "se": c.get("se"),
                     "roi_c5_at_bsp": c.get("roi_c5_settle_bsp"), "strike": c.get("strike"), "mean_p_cal": r4(g["p_cal"].mean())})
    out({"section": "6_selected_bets_with_bsp_by_ltp_over_bsp", "rows": rows,
         "share_ltp_above_bsp": r4((hb["ratio"] > 1).mean()), "share_ltp_above_bsp_blind": r4(
             (bettable.loc[bettable["bsp_pl"] > 1, "ltp_pl"] > bettable.loc[bettable["bsp_pl"] > 1, "bsp_pl"]).mean()),
         "roi_c5_at_ltp_bets_with_ltp_within_2pct_of_bsp": rows[2]["roi_c5_at_ltp"]})
    # same rule with the WIN market: is the win ltp also stale for these races? (win bsp/ltp for the selected runner)
    hw = hb[hb["bsp_win"].notna() & (hb["bsp_win"] > 1)]
    out({"section": "6_win_side_of_selected_bets", "bets": int(len(hw)),
         "median_win_bsp_over_ltp": r4((hw["bsp_win"] / hw["ltp_win"]).median()),
         "share_win_ltp_within_5pct_of_win_bsp": r4(((hw["bsp_win"] / hw["ltp_win"]).sub(1).abs() < 0.05).mean()),
         "share_place_ltp_within_5pct_of_place_bsp": r4(((hw["bsp_pl"] / hw["ltp_pl"]).sub(1).abs() < 0.05).mean())})

    # ---- 7. winner's curse on p_cal -------------------------------------------------------
    out({"section": "7_winners_curse_bsp_subset", "bets": int(len(hb)),
         "mean_p_cal": r4(hb["p_cal"].mean()), "strike": r4(hb["placed"].mean()),
         "mean_1_over_ltp": r4((1 / hb["ltp_pl"]).mean()), "mean_1_over_bsp": r4((1 / hb["bsp_pl"]).mean()),
         "strike_over_mean_p_cal": r4(hb["placed"].mean() / hb["p_cal"].mean()),
         "blind_bsp_subset": {"mean_p_cal": r4(bettable.loc[bettable["bsp_pl"] > 1, "p_cal"].mean()),
                              "strike": r4(bettable.loc[bettable["bsp_pl"] > 1, "placed"].mean())}})

    # ---- 8. a bettable price proxy: the smaller of ltp_pre and the one-minute-older ltp_t30? (not available)
    # Instead: the rule evaluated on the subset where the place market was liquid enough to have an SP
    # for every runner AND a consistent (over-round) ltp book.
    liq = sel[sel["race_has_bsp"] & (sel["book_lb"] >= 1.0)]
    out({"section": "8_selected_bets_liquid_consistent_races", **cell(liq, mid)})

    # ---- 9. corrected estimate of the executable edge ------------------------------------
    # decide on ltp_pre (as the claim), settle at the place BSP (the price a market order gets), all codes
    out({"section": "9_corrected_estimate", "decide_ltp_settle_bsp": {kk: head.get(kk) for kk in
         ("bets", "bsp_cover", "roi_c5_settle_bsp", "bsp_se", "bsp_halves_c5", "roi_c5_worse_of_ltp_bsp")},
         "roi_c8_settle_bsp": r4(w2p.returns(hb["bsp_pl"], hb["placed"], hb["dh"], 0.08).mean()),
         "roi_c0_settle_bsp": r4(w2p.returns(hb["bsp_pl"], hb["placed"], hb["dh"], 0.0).mean()),
         "bsp_se_race_clustered": r4(cluster_se(w2p.returns(hb["bsp_pl"], hb["placed"], hb["dh"], 0.05), hb["race"].to_numpy())),
         "runtime_s": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

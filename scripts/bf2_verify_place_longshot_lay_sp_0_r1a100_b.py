"""Follow-up checks for the R1a@100 verification (bf2_verify_place_longshot_lay_sp_0_r1a100.py).

    python scripts/bf2_verify_place_longshot_lay_sp_0_r1a100_b.py [--dir data/raw/betfair/split]

  S  loader filter counts per country (paired / identical runner sets / settled / G6)
  T  thin-market share: R1a@100 bets whose runner never traded in the PLACE market before the off
     (no place ltp_pre) or at t30, by segment; ROI restricted to runners that DID trade in the place market
  U  price impact by segment (AU gallops) and after a LOC cap of 100: BSP x 1.02 and + 1 tick, with G4
  V  simple capacity proxy: ROI by number of R1a@100 selections in the race (more selections = more of our
     liability in one place market) and by field size
Prints summary numbers only.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("v0", ROOT / "scripts/bf2_verify_place_longshot_lay_sp_0_r1a100.py")
v0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v0)
orig, out, r6, summ, step_at, lay_liab = v0.orig, v0.out, v0.r6, v0.summ, v0.step_at, v0.lay_liab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data/raw/betfair/split"))
    a = ap.parse_args()
    d = Path(a.dir)
    frames = []
    for cc in orig.COUNTRIES:
        j, st = orig.load_country(cc, d, 1.0, 7)
        out({"section": "S_loader_counts", "country": cc, **{k: st[k] for k in (
            "win_markets", "place_markets", "paired_markets", "paired_nw_2_3", "identical_runner_sets",
            "settled_n_ge_nw_plus_1", "g6_dropped_races", "races")}})
        j["cc"] = cc
        frames.append(j)
    df = pd.concat(frames, ignore_index=True)
    del frames
    df["date"] = pd.to_datetime(df["date"]).values.astype("datetime64[D]")
    u = df[df["win_pre_all"] & df["win_pre_book"].between(orig.BOOK_LO, orig.BOOK_HI)]
    g = u[(u["pre_win"] >= 100) & u["bsp_pl"].notna()].copy()
    P, F = g["bsp_pl"].to_numpy(), g["f_pl"].to_numpy()
    r = lay_liab(P, F)

    # T thin markets
    nopre = g["pre_pl"].isna().to_numpy()
    no30 = g["t30_pl"].isna().to_numpy()
    tab = {}
    for seg in ("au_gallops", "au_harness", "gb", "ie"):
        m = (g["seg"] == seg).to_numpy()
        tab[seg] = {"bets": int(m.sum()), "share_no_place_ltp_pre": r6(nopre[m].mean()),
                    "share_no_place_ltp_t30": r6(no30[m].mean()),
                    "profit_share_no_place_ltp_pre": r6(r[m & nopre].sum() / r[m].sum())}
    out({"section": "T_thin_market_shares", "table": tab})
    summ(g[nopre], P[nopre], F[nopre], "T_no_place_ltp_pre")
    summ(g[~nopre], P[~nopre], F[~nopre], "T_place_traded_pre")
    m = ~nopre & ~no30
    summ(g[m], P[m], F[m], "T_place_traded_pre_and_t30")
    m = ~nopre & (g["seg"] == "au_gallops").to_numpy()
    summ(g[m], P[m], F[m], "T_place_traded_pre_au_gallops")

    # U impact by segment / LOC cap
    P1 = np.minimum(P + step_at(P), 1000.0)
    for lab, m in (("all", np.ones(len(g), bool)), ("au_gallops", (g["seg"] == "au_gallops").to_numpy()),
                   ("loc_cap_100", P <= 100), ("au_gallops_loc_cap_100", (g["seg"] == "au_gallops").to_numpy() & (P <= 100))):
        summ(g[m], P[m], F[m], f"U_{lab}_bsp")
        summ(g[m], P[m] * 1.02, F[m], f"U_{lab}_bsp_x1.02")
        summ(g[m], P1[m], F[m], f"U_{lab}_bsp_plus1tick")

    # V capacity proxy: number of our selections in the race
    k = g.groupby("race")["selection_id"].transform("size").to_numpy()
    tab = {}
    for lo, hi in ((1, 2), (2, 3), (3, 5), (5, 100)):
        m = (k >= lo) & (k < hi)
        s = summ(g[m], P[m], F[m], "", verbose=False)
        tab[f"[{lo},{hi})"] = [s.get("bets"), s.get("roi8"), s.get("z")]
    out({"section": "V_by_selections_in_race_bets_roi_z", "table": tab})
    tab = {}
    n = g["n"].to_numpy()
    for lo, hi in ((2, 9), (9, 12), (12, 15), (15, 40)):
        m = (n >= lo) & (n < hi)
        s = summ(g[m], P[m], F[m], "", verbose=False)
        tab[f"[{lo},{hi})"] = [s.get("bets"), s.get("roi8"), s.get("z")]
    out({"section": "V_by_field_size_bets_roi_z", "table": tab})


if __name__ == "__main__":
    main()

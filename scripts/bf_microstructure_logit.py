"""Experiment "microstructure_logit": price-only residual model on the Betfair
Exchange (AU WIN markets from data/raw/betfair/split/au_win.parquet).

    python scripts/bf_microstructure_logit.py [--end 2018-06-01] [--burn-in 6] [--l2 0.01,0.1,1]

Walk-forward ConditionalLogit (train on every market dated strictly before the
month being predicted, refit monthly) with features derived only from
information available before the jump:

    log_p          log Shin probability at ltp_pre (unpenalised)      [runner]
    rank1..rank3   favourite-rank dummies at ltp_pre                  [runner]
    has_t30        runner traded 30 min before the start              [runner]
    move           log(ltp_t30 / ltp_pre), 0 where no t30 trade       [runner]
    log_field, book (sum 1/ltp_pre), harness, hour_local, weekday     [race]
    logfield_x_logp, harness_x_logp                                   [race x runner]

Race-level columns are constant inside a race, so inside a within-race softmax
their main effects cancel exactly (the gradient is identically zero and the L2
penalty keeps them at 0). They are kept because the spec lists them, and a
second feature set adds the interactions with log_p that let them matter
(book_x_logp, hour_x_logp, sat_x_logp, t30_x_logp).

Sections (JSON lines, summary numbers only):
  data      filters, timezone mapping coverage, feature summary
  fit_check Newton fit of this script vs the repo's gradient-descent fit
  model     per (feature set, l2): coefficients, log-loss gain vs the market
            with a per-race SE and halves, top-pick strike
  bets      model_prob * ltp_pre > 1 + k at commission 0 / 5% / 8% [A-54],
            per-bet SE, halves, one tick below ltp_pre and Betfair SP as
            realistic-execution lines, coherent-book filter, code split,
            edge-bin calibration
  D         Benjamini-Hochberg over every bet cell, the both-halves > 2 SE rule,
            opposite-half development (pick the best cell in one half, evaluate
            in the other)

ltp_pre is the last traded price before the off (one-minute snapshots), not a
price that was guaranteed to be on offer; the tick-down and BSP lines are the
realistic ones.
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
from ponies import odds  # noqa: E402
from ponies.model import ConditionalLogit, _race_codes, race_softmax, win_targets  # noqa: E402

COMMISSIONS = (0.0, 0.05, 0.08)
K_VALUES = (0.0, 0.05, 0.10, 0.20)
COARSE_BANDS = [1.0, 4.0, 10.0, 1e9]
MIN_BETS = 200

FEATURES_A = ["log_p", "rank1", "rank2", "rank3", "log_field", "logfield_x_logp", "book",
              "harness", "harness_x_logp", "hour_local", "weekday", "has_t30", "move"]
FEATURES_B = FEATURES_A + ["book_x_logp", "hour_x_logp", "sat_x_logp", "t30_x_logp"]
FEATURE_SETS = {"A_spec": FEATURES_A, "B_spec_plus_race_interactions": FEATURES_B}

# Betfair price ladder: (lower bound inclusive, upper bound exclusive, increment)
TICKS = [(1.01, 2.0, 0.01), (2.0, 3.0, 0.02), (3.0, 4.0, 0.05), (4.0, 6.0, 0.1), (6.0, 10.0, 0.2),
         (10.0, 20.0, 0.5), (20.0, 30.0, 1.0), (30.0, 50.0, 2.0), (50.0, 100.0, 5.0), (100.0, 1001.0, 10.0)]

# venue -> IANA timezone (Betfair venue names). Anything not listed falls back to
# Australia/Sydney (NSW/VIC/TAS/ACT share it); the share of unmapped markets is printed.
TZ_VENUES = {
    "Australia/Perth": """Gloucester Park Pinjarra Bunbury Ascot Northam Belmont Albany Kalgoorlie Geraldton
        Narrogin Kellerberrin Esperance York Mount Barker Broome Carnarvon Port Hedland Wagin Williams Toodyay
        Collie Busselton Central Wheatbelt Moora Dongara Roebourne Kununurra Derby Leonora Mt Magnet Lark Hill
        Bridgetown Pingrup Katanning Harvey Boyup Brook Corrigin Dowerin Cunderdin Wyndham Laverton Norseman
        Marble Bar Beverley Pinjarra Park Mount Magnet Newman Nannup Landor Yalgoo Mingenew Kojonup Lake Grace
        Narrogin Park Golden Mile""",
    "Australia/Adelaide": """Globe Derby Morphettville Gawler Port Pirie Murray Bridge Port Lincoln Strathalbyn
        Mount Gambier Balaklava Port Augusta Naracoorte Penola Kapunda Mindarie Jamestown Clare Oakbank Bordertown
        Quorn Roxby Downs Victor Harbor Whyalla Ceduna Streaky Bay Lock Kimba Hawker Penong Kangaroo Island
        Kingscote Cummins Tumby Bay Wudinna Morphettville Parks Kadina Murray Bridge GH Broken Hill Mindarie-Halidon
        Cheltenham Victoria Park Gawler Park""",
    "Australia/Darwin": "Darwin Alice Springs Pioneer Park Fannie Bay Adelaide River Katherine Tennant Creek",
    "Australia/Sydney": """Menangle Newcastle Bathurst Wagga Tamworth Penrith Randwick Canberra Dubbo Kembla Grange
        Rosehill Goulburn Grafton Hawkesbury Canterbury Wyong Gosford Warwick Farm Albury Taree Scone Young Parkes
        Muswellbrook Port Macquarie Bankstown Coffs Harbour Ballina Armidale Moruya Nowra Orange Queanbeyan
        Sapphire Coast Lismore Maitland Murwillumbah Gunnedah Cowra Forbes Mudgee Narromine Wellington Gilgandra
        Coonamble Coonabarabran Nyngan Cobar Bourke Brewarrina Walgett Moree Narrabri Inverell Glen Innes Tenterfield
        Casino Kempsey Wauchope Gloucester Cessnock Quirindi Tumut Gundagai Cootamundra Junee Temora Griffith Leeton
        Narrandera Hay Hillston Balranald Deniliquin Moulamein Jerilderie Berrigan Tocumwal Corowa Holbrook
        Tumbarumba Wagga Riverside Braidwood Bombala Cooma Tuncurry Lightning Ridge Louth Wean Come By Chance Talmoi
        Mallawa Carinda Collarenebri Enngonia Tottenham Trangie Warren Condobolin Tullibigeal Lake Cargelligo
        Ardlethan Carrathool Ivanhoe Wentworth Pooncarie Bingara Barraba Wee Waa Mungindi Adaminaby Harden Boorowa
        Murrumburrah Gulgong Dunedoo Coolah Mendooran Tomingley Peak Hill Trundle Grenfell Canowindra Rylstone Lithgow
        Bong Bong Tabulam Bowraville Macksville Dungog Merriwa Nundle Wallabadah Bundarra Deepwater Menangle Park
        Rosehill Gardens Royal Randwick Kensington Fairfield Kembla Newcastle Paceway Tamworth Paceway Wagga Paceway
        Dubbo Paceway Bathurst Paceway Young Paceway Parkes Paceway Goulburn Paceway Leeton Paceway Penrith Paceway
        Bankstown Paceway Albury Paceway Junee Paceway Temora Paceway Cowra Paceway Forbes Paceway Orange Paceway
        Griffith Paceway Queanbeyan Paceway Canberra Paceway Tweed Heads Moss Vale Picton Camden Appin Blayney
        Carcoar Kandos Eugowra Bedgerabong Binalong Boggabri Burren Junction Dalgety Emmaville Tingha Moonan Flat""",
    "Australia/Brisbane": """Albion Park Redcliffe Sunshine Coast Gold Coast Doomben Toowoomba Ipswich Rockhampton
        Townsville Mackay Eagle Farm Cairns Gatton Marburg Beaudesert Kilcoy Dalby Warwick Roma Emerald Bundaberg
        Gympie Nanango Esk Innisfail Mount Isa Longreach Charleville Chinchilla Goondiwindi Home Hill Bowen
        Cloncurry Barcaldine Thangool Yeppoon Clifton Gladstone Mareeba Atherton Ewan Tara Texas Stanthorpe Kumbia
        Wondai Burrandowan Birdsville Bedourie Betoota Jandowae Miles Taroom Moranbah Blackall Winton Hughenden
        Richmond Julia Creek Quilpie Cunnamulla Mitchell Surat St George Dirranbandi Injune Springsure Alpha
        Middlemount Twin Hills Calliope Monto Mount Perry Eidsvold Gayndah Mundubbera Biloela Callaghan Park
        Cluden Ooralea Cannon Park Clifford Park Corbould Park Deagon Lockyer Valley Beenleigh Oakey Bell
        Kilcoy Wallumbilla Augathella Tambo Isisford Aramac Muttaburra Jundah Windorah Stonehenge Yaraka Eulo
        Noorama Thargomindah Morven Bluff Capella Mount Garnet Laura Cooktown Gordonvale Herbert River Ingham
        Tower Hill Townsville Turf Burketown Camooweal Normanton Croydon Georgetown Einasleigh Prairie Pentland
        Charters Towers Collinsville Mount Coolon Clermont Nebo Sarina Mirani St Lawrence Marlborough
        Rockhampton Jockey Club Wowan Dingo Duaringa Theodore Moura Baralaba Rolleston Tieri Capricornia Kilkivan
        Imbil Mooloolah Nambour Caloundra Bribie Kilcoy""",
    "Australia/Hobart": """Launceston Hobart Devonport Elwick Mowbray Spreyton Carrick Burnie Longford King Island
        St Marys Scottsdale Deloraine Smithton Latrobe""",
    "Australia/Melbourne": """Melton Bendigo Ballarat Mildura Cranbourne Geelong Shepparton Pakenham Kilmore Sandown
        Maryborough Echuca Flemington Caulfield Moonee Valley Terang Yarra Valley Swan Hill Warrnambool Sale
        Mornington Stawell Moe Hamilton Wangaratta Ararat Horsham Kyneton Benalla Seymour Cobram Werribee Donald
        Wodonga Charlton Bairnsdale Colac Casterton Camperdown Coleraine Edenhope Penshurst Dunkeld Great Western
        Avoca St Arnaud Wycheproof Manangatang Ouyen Nhill Murtoa Kerang Boort Wedderburn Burrumbeet Balnarring
        Healesville Yarra Glen Alexandra Mansfield Tatura Kyabram Gunbower Hanging Rock Woolamai Stony Creek
        Traralgon Yea Towong Buchan Drouin Dederang Tallangatta Yarrawonga Nyah Merton Mortlake Apsley
        Sandown Lakeside Sandown Hillside Caulfield Heath Flemington Straight Moonee Valley Night Ballarat Synthetic
        Pakenham Synthetic Cranbourne Training Mildura Harness Bendigo Harness Shepparton Harness Geelong Harness
        Ballarat Harness Kilmore Harness Charlton Harness Yarra Valley Harness Maryborough Harness Hamilton Harness
        Bairnsdale Harness Cobram Harness Echuca Harness Swan Hill Harness Terang Harness Boort Harness Ararat Harness
        Horsham Harness Stawell Harness Wangaratta Harness Melton Harness Warragul Cranbourne Harness Kyabram Harness
        Mornington Harness Wedderburn Harness Nyah Harness Ouyen Harness Gunbower Harness Elmore Rochester
        Donald Harness St Arnaud Harness Great Western Harness Warrnambool Harness Mildura Showground
        Werribee Harness Geelong Synthetic""",
}

# ----------------------------------------------------------------------------- utilities


def out(obj) -> None:
    print(json.dumps(obj, default=_json_default), flush=True)


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    raise TypeError(str(type(o)))


def r4(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) else round(x, 4)


def net_returns(price: np.ndarray, won: np.ndarray, commission: float) -> np.ndarray:
    """Vectorised ponies.betfair.net_return [A-54]: commission on net winnings only."""
    price = np.asarray(price, dtype=float)
    return np.where(won, (price - 1.0) * (1.0 - commission), -1.0)


def tick_down(price: np.ndarray) -> np.ndarray:
    """One Betfair tick below ``price``."""
    p = np.asarray(price, dtype=float)
    inc = np.full_like(p, 10.0)
    q = p - 1e-9
    for lo, hi, step in TICKS:
        inc = np.where((q >= lo) & (q < hi), step, inc)
    return np.maximum(1.01, np.round(p - inc, 2))


def bands(series: pd.Series, edges: list[float]) -> pd.Series:
    labels = [f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+" for lo, hi in zip(edges, edges[1:])]
    return pd.cut(series, edges, right=False, labels=labels)


def shin_vectorised(price: np.ndarray, codes: np.ndarray, n: int) -> np.ndarray:
    """odds.shin_probabilities [A-51] for every market at once (bisection on z per market)."""
    pi = 1.0 / np.asarray(price, dtype=float)
    s = np.bincount(codes, weights=pi, minlength=n)
    sc = s[codes]
    lo = np.zeros(n)
    hi = np.full(n, 0.5)

    def probs(z):
        zc = z[codes]
        return (np.sqrt(zc * zc + 4 * (1 - zc) * pi * pi / sc) - zc) / (2 * (1 - zc))
    for _ in range(60):
        mid = (lo + hi) / 2
        tot = np.bincount(codes, weights=probs(mid), minlength=n)
        above = tot > 1
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    p = probs((lo + hi) / 2)
    p = np.where(sc <= 1.0, pi, p)  # no margin to remove: proportional
    tot = np.bincount(codes, weights=p, minlength=n)
    return p / tot[codes]


# ----------------------------------------------------------------------------- Newton fit


class NewtonLogit(ConditionalLogit):
    """The repo's ConditionalLogit (same preprocessing, prediction and objective:
    mean per-race log loss + 0.5 * l2 * |w|^2 with the unpenalised features
    exempt) fitted by damped Newton steps instead of full-batch gradient descent,
    because ~45 monthly refits on ~1M rows would not finish otherwise."""

    def fit(self, df: pd.DataFrame, max_iter: int = 50, tol: float = 1e-10) -> "NewtonLogit":
        codes, n_races = _race_codes(df["race_id"])
        y = win_targets(df["won"].to_numpy(), codes, n_races)
        Z = self._prepare(df, fit=True)
        l2 = (0.01 if self.l2 is None else self.l2) * np.array(
            [0.0 if f in self.unpenalized else 1.0 for f in self.feature_names])
        k = Z.shape[1]

        def loss_of(w):
            p = race_softmax(Z @ w, codes, n_races)
            return (-np.sum(y * np.log(p + 1e-15)) / n_races + 0.5 * w @ (l2 * w)), p

        w = np.zeros(k)
        loss, p = loss_of(w)
        self.history = [float(loss)]
        for _ in range(max_iter):
            grad = Z.T @ (p - y) / n_races + l2 * w
            zbar = np.column_stack([np.bincount(codes, weights=p * Z[:, j], minlength=n_races) for j in range(k)])
            Zc = Z - zbar[codes]
            H = (Zc * p[:, None]).T @ Zc / n_races + np.diag(l2) + 1e-10 * np.eye(k)
            step = np.linalg.solve(H, grad)
            t = 1.0
            while True:
                w_new = w - t * step
                loss_new, p_new = loss_of(w_new)
                if loss_new <= loss or t < 1e-6:
                    break
                t *= 0.5
            improvement = loss - loss_new
            w, loss, p = w_new, loss_new, p_new
            self.history.append(float(loss))
            if improvement < tol:
                break
        self.weights = w
        return self

    def raw_coefs(self) -> dict:
        return {f: float(w / s) for f, w, s in zip(self.feature_names, self.weights, self.stds)}


# ----------------------------------------------------------------------------- data


def load(path: Path, end: str | None) -> tuple[pd.DataFrame, dict]:
    cols = ["market_id", "venue", "market_name", "start", "date", "field_size", "selection_id",
            "ltp_pre", "ltp_t30", "bsp", "won", "market_type"]
    df = pd.read_parquet(path, columns=cols)
    n_raw = len(df)
    df = df.drop_duplicates(["market_id", "selection_id"])
    n_dedup = len(df)
    df = df[(df["market_type"] == "WIN") & df["won"].notna()]
    if end:
        df = df[df["date"] < pd.Timestamp(end)]
    m_all = df["market_id"].nunique()
    g = df.groupby("market_id")
    one_winner = g["won"].transform("sum") == 1  # drops dead heats
    df["_ok"] = df["ltp_pre"].notna() & (df["ltp_pre"] > 1)
    all_priced = df.groupby("market_id")["_ok"].transform("min").astype(bool)
    df = df[one_winner & all_priced].drop(columns=["_ok", "market_type"]).copy()
    df = df.sort_values(["start", "market_id", "ltp_pre"]).reset_index(drop=True)

    codes, n = _race_codes(df["market_id"])
    df["p_shin"] = shin_vectorised(df["ltp_pre"].to_numpy(), codes, n)
    df["log_p"] = np.log(df["p_shin"])
    df["book"] = np.bincount(codes, weights=1.0 / df["ltp_pre"].to_numpy(), minlength=n)[codes]
    rank = df.groupby("market_id")["ltp_pre"].rank(method="min")
    for r in (1, 2, 3):
        df[f"rank{r}"] = (rank == r).astype(float)
    df["log_field"] = np.log(df["field_size"].to_numpy(dtype=float))
    df["logfield_x_logp"] = df["log_field"] * df["log_p"]
    df["harness"] = df["market_name"].fillna("").str.contains(r"\b(?:Pace|Trot)\b").astype(float)
    df["harness_x_logp"] = df["harness"] * df["log_p"]
    df["has_t30"] = (df["ltp_t30"].notna() & (df["ltp_t30"] > 1)).astype(float)
    df["move"] = np.where(df["has_t30"] > 0, np.log(df["ltp_t30"] / df["ltp_pre"]), 0.0)
    # local time of day / weekday from the venue's timezone
    tz_of = {v: tz for tz, names in TZ_VENUES.items() for v in _venues(names)}
    venue_tz = df["venue"].map(tz_of)
    unmapped = venue_tz.isna()
    venue_tz = venue_tz.fillna("Australia/Sydney")
    local_hour = np.zeros(len(df))
    local_wday = np.zeros(len(df))
    for tz, idx in df.groupby(venue_tz.to_numpy()).indices.items():
        loc = df["start"].iloc[idx].dt.tz_convert(tz)
        local_hour[idx] = (loc.dt.hour + loc.dt.minute / 60.0).to_numpy()
        local_wday[idx] = loc.dt.weekday.to_numpy()
    df["hour_local"] = local_hour
    df["weekday"] = local_wday
    df["is_sat"] = (local_wday == 5).astype(float)
    df["book_x_logp"] = (df["book"] - 1.0) * df["log_p"]
    df["hour_x_logp"] = (df["hour_local"] - 15.0) * df["log_p"]
    df["sat_x_logp"] = df["is_sat"] * df["log_p"]
    df["t30_x_logp"] = df["has_t30"] * df["log_p"]
    df["race_id"] = df["market_id"]
    df["month"] = df["date"].dt.to_period("M")
    df["won_b"] = df["won"].to_numpy() == 1
    unmapped_markets = df.loc[unmapped, "market_id"].nunique()
    info = {"rows_raw": int(n_raw), "rows_after_dedupe": int(n_dedup), "markets_settled": int(m_all),
            "markets_used": int(df["market_id"].nunique()), "runners_used": int(len(df)),
            "dropped": "dead heats and markets with any runner never traded pre-race",
            "venue_tz_unmapped_market_share": r4(unmapped_markets / max(df["market_id"].nunique(), 1)),
            "top_unmapped_venues": df.loc[unmapped].drop_duplicates("market_id")["venue"].value_counts().head(8).to_dict()}
    return df, info


def _venues(block: str) -> list[str]:
    # names are separated by two or more spaces / newlines in the table above? No: by
    # single spaces, and several names are multi-word, so split on a curated rule:
    # a word starting a new name is one that follows a complete entry. We keep it
    # simple: a whitelist of multi-word names, everything else is single words.
    multi = ["Gloucester Park", "Mount Barker", "Port Hedland", "Central Wheatbelt", "Mt Magnet", "Lark Hill",
             "Boyup Brook", "Marble Bar", "Pinjarra Park", "Mount Magnet", "Lake Grace", "Narrogin Park",
             "Golden Mile", "Globe Derby", "Port Pirie", "Murray Bridge", "Port Lincoln", "Mount Gambier",
             "Port Augusta", "Roxby Downs", "Victor Harbor", "Streaky Bay", "Kangaroo Island", "Tumby Bay",
             "Morphettville Parks", "Murray Bridge GH", "Broken Hill", "Victoria Park", "Gawler Park",
             "Alice Springs", "Pioneer Park", "Fannie Bay", "Adelaide River", "Tennant Creek", "Albion Park",
             "Sunshine Coast", "Gold Coast", "Eagle Farm", "Mount Isa", "Home Hill", "Julia Creek", "St George",
             "Twin Hills", "Mount Perry", "Callaghan Park", "Cannon Park", "Clifford Park", "Corbould Park",
             "Lockyer Valley", "Mount Garnet", "Herbert River", "Tower Hill", "Townsville Turf", "Charters Towers",
             "Mount Coolon", "St Lawrence", "Rockhampton Jockey Club", "King Island", "St Marys", "Moonee Valley",
             "Yarra Valley", "Swan Hill", "St Arnaud", "Great Western", "Yarra Glen", "Hanging Rock", "Stony Creek",
             "Sandown Lakeside", "Sandown Hillside", "Caulfield Heath", "Flemington Straight", "Moonee Valley Night",
             "Ballarat Synthetic", "Pakenham Synthetic", "Cranbourne Training", "Mildura Showground",
             "Geelong Synthetic", "Mindarie-Halidon", "Kembla Grange", "Warwick Farm", "Port Macquarie",
             "Coffs Harbour", "Sapphire Coast", "Glen Innes", "Lake Cargelligo", "Wagga Riverside", "Lightning Ridge",
             "Come By Chance", "Peak Hill", "Bong Bong", "Menangle Park", "Rosehill Gardens", "Royal Randwick",
             "Newcastle Paceway", "Tamworth Paceway", "Wagga Paceway", "Dubbo Paceway", "Bathurst Paceway",
             "Young Paceway", "Parkes Paceway", "Goulburn Paceway", "Leeton Paceway", "Penrith Paceway",
             "Bankstown Paceway", "Albury Paceway", "Junee Paceway", "Temora Paceway", "Cowra Paceway",
             "Forbes Paceway", "Orange Paceway", "Griffith Paceway", "Queanbeyan Paceway", "Canberra Paceway",
             "Tweed Heads", "Moss Vale", "Burren Junction", "Moonan Flat", "Wee Waa"]
    names = set()
    text = " ".join(block.split())
    for m in sorted(multi, key=len, reverse=True):
        if m in text:
            names.add(m)
            text = text.replace(m, " ")
    for w in text.split():
        if w != "Harness":
            names.add(w)
    for m in list(names):
        names.add(f"{m} Harness")
    return sorted(names)


def mid_day(df: pd.DataFrame) -> pd.Timestamp:
    days = np.sort(df["date"].unique())
    return pd.Timestamp(days[len(days) // 2])


# ----------------------------------------------------------------------------- cells


def bet_cell(g: pd.DataFrame, mid: pd.Timestamp, price_col: str = "ltp_pre") -> dict:
    n = len(g)
    res = {"bets": int(n)}
    if n == 0:
        return res
    won = g["won_b"].to_numpy()
    price = g[price_col].to_numpy(dtype=float)
    for c in COMMISSIONS:
        res[f"roi_c{int(c * 100)}"] = r4(net_returns(price, won, c).mean())
    r5 = net_returns(price, won, 0.05)
    res["se"] = r4(r5.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    first = (g["date"] < mid).to_numpy()
    halves, halves_se = [], []
    for mask in (first, ~first):
        h = r5[mask]
        halves.append(r4(h.mean()) if len(h) else None)
        halves_se.append(r4(h.std(ddof=1) / math.sqrt(len(h))) if len(h) > 1 else None)
    res["halves_roi_c5"] = halves
    res["halves_se_c5"] = halves_se
    res["halves_bets"] = [int(first.sum()), int((~first).sum())]
    res["positive_both_halves"] = bool(all(h is not None and s is not None and h > 2 * s
                                           for h, s in zip(halves, halves_se)))
    res["strike"] = r4(won.mean())
    res["implied_shin"] = r4(g["p_shin"].mean())
    res["model_prob"] = r4(g["p_model"].mean()) if "p_model" in g else None
    res["median_price"] = r4(np.median(price))
    res["roi_c5_tick_down"] = r4(net_returns(tick_down(price), won, 0.05).mean())
    has_bsp = g["bsp"].notna().to_numpy() & (g["bsp"].to_numpy(dtype=float) > 1)
    if has_bsp.sum() > 1:
        res["roi_c5_bsp"] = r4(net_returns(g["bsp"].to_numpy(dtype=float)[has_bsp], won[has_bsp], 0.05).mean())
        res["bsp_bets"] = int(has_bsp.sum())
        res["median_ltp_pre_over_bsp"] = r4(np.median(g["ltp_pre"].to_numpy(dtype=float)[has_bsp] / g["bsp"].to_numpy(dtype=float)[has_bsp]))
    res["share_book_below_0.97"] = r4((g["book"] < 0.97).mean())
    return res


def z_of(cell: dict) -> float:
    if cell.get("bets", 0) < MIN_BETS or not cell.get("se"):
        return float("nan")
    return cell["roi_c5"] / cell["se"]


def half_z(cell: dict, h: int) -> float:
    if "halves_roi_c5" not in cell:
        return float("nan")
    r, s = cell["halves_roi_c5"][h], cell["halves_se_c5"][h]
    if r is None or s is None or cell["halves_bets"][h] < MIN_BETS:
        return float("nan")
    return r / s


def compact(c: dict) -> dict:
    keys = ("bets", "roi_c0", "roi_c5", "roi_c8", "se", "halves_roi_c5", "halves_se_c5", "halves_bets",
            "positive_both_halves", "strike", "implied_shin", "model_prob", "median_price",
            "roi_c5_tick_down", "roi_c5_bsp", "median_ltp_pre_over_bsp", "share_book_below_0.97")
    return {k: c.get(k) for k in keys if k in c}


# ----------------------------------------------------------------------------- walk-forward


def walk_forward(df: pd.DataFrame, features: list[str], l2: float, burn_in: int) -> tuple[np.ndarray, list[dict]]:
    months = sorted(df["month"].unique())
    month_arr = df["month"].to_numpy()
    preds = np.full(len(df), np.nan)
    coefs = []
    for i, m in enumerate(months):
        if i < burn_in:
            continue
        train = df[month_arr < m]
        test_idx = np.flatnonzero(month_arr == m)
        if train["race_id"].nunique() < 500 or len(test_idx) == 0:
            continue
        model = NewtonLogit(features, l2=l2, unpenalized=("log_p",)).fit(train)
        preds[test_idx] = model.predict_proba(df.iloc[test_idx])
        coefs.append({"month": str(m), "train_races": int(train["race_id"].nunique()),
                      "iters": len(model.history) - 1, **model.raw_coefs()})
    return preds, coefs


def log_loss_gain(test: pd.DataFrame, mid: pd.Timestamp) -> dict:
    w = test[test["won_b"]]
    gain = (-np.log(w["p_shin"].clip(1e-12)) + np.log(w["p_model"].clip(1e-12))).to_numpy()
    first = (w["date"] < mid).to_numpy()
    se = gain.std(ddof=1) / math.sqrt(len(gain))
    res = {"races": int(len(w)), "ll_market_shin": r4(-np.log(w["p_shin"].clip(1e-12)).mean()),
           "ll_model": r4(-np.log(w["p_model"].clip(1e-12)).mean()),
           "gain_per_race": r4(gain.mean()), "gain_se": r4(se), "gain_z": r4(gain.mean() / se),
           "gain_halves": [r4(gain[first].mean()), r4(gain[~first].mean())],
           "gain_halves_se": [r4(gain[first].std(ddof=1) / math.sqrt(max(first.sum(), 2))),
                              r4(gain[~first].std(ddof=1) / math.sqrt(max((~first).sum(), 2)))]}
    res["positive_2se_overall"] = bool(gain.mean() > 2 * se)
    res["positive_2se_both_halves"] = bool(all(g > 2 * s for g, s in zip(res["gain_halves"], res["gain_halves_se"])))
    # by code
    for code, mask in (("gallops", (w["harness"] == 0).to_numpy()), ("harness", (w["harness"] == 1).to_numpy())):
        gg = gain[mask]
        res[f"gain_{code}"] = {"races": int(mask.sum()), "gain": r4(gg.mean()),
                               "se": r4(gg.std(ddof=1) / math.sqrt(max(len(gg), 2)))}
    return res


def top_pick(test: pd.DataFrame) -> dict:
    idx_model = test.groupby("market_id")["p_model"].idxmax()
    idx_market = test.groupby("market_id")["p_shin"].idxmax()
    same = (idx_model.to_numpy() == idx_market.to_numpy())
    return {"races": int(len(idx_model)),
            "model_top_pick_strike": r4(test.loc[idx_model, "won_b"].mean()),
            "market_favourite_strike": r4(test.loc[idx_market, "won_b"].mean()),
            "share_same_pick": r4(same.mean()),
            "strike_when_different_model": r4(test.loc[idx_model[~same], "won_b"].mean()) if (~same).sum() else None,
            "strike_when_different_market": r4(test.loc[idx_market[~same], "won_b"].mean()) if (~same).sum() else None,
            "n_different": int((~same).sum())}


# ----------------------------------------------------------------------------- multiple testing


def benjamini_hochberg(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    rej = np.zeros(len(p), dtype=bool)
    if m == 0:
        return rej
    order = np.argsort(p[ok])
    sorted_p = p[ok][order]
    thresh = q * (np.arange(1, m + 1) / m)
    below = np.flatnonzero(sorted_p <= thresh)
    if len(below):
        k = below.max()
        idx = np.flatnonzero(ok)[order[: k + 1]]
        rej[idx] = True
    return rej


def one_sided_p(z: float) -> float:
    if math.isnan(z):
        return float("nan")
    return 0.5 * math.erfc(z / math.sqrt(2))


# ----------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=Path("data/raw/betfair/split/au_win.parquet"))
    ap.add_argument("--end", default=None, help="development: use only dates before this")
    ap.add_argument("--burn-in", type=int, default=6, help="months before the first walk-forward prediction")
    ap.add_argument("--l2", default="0.01,0.1,1", help="comma list of L2 penalties to try")
    ap.add_argument("--feature-sets", default="A_spec,B_spec_plus_race_interactions")
    ap.add_argument("--skip-fit-check", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    l2_grid = [float(x) for x in a.l2.split(",")]
    fsets = {k: FEATURE_SETS[k] for k in a.feature_sets.split(",")}

    df, info = load(a.path, a.end)
    days = np.sort(df["date"].unique())
    out({"section": "data", **info, "from": str(pd.Timestamp(days[0]).date()), "to": str(pd.Timestamp(days[-1]).date()),
         "harness_market_share": r4(df.groupby("market_id")["harness"].first().mean()),
         "runners_with_t30": r4(df["has_t30"].mean()),
         "book_median": r4(df.groupby("market_id")["book"].first().median()),
         "local_hour_quartiles": [r4(x) for x in np.percentile(df["hour_local"], [5, 25, 50, 75, 95])],
         "weekday_market_share": df.groupby("market_id")["weekday"].first().value_counts(normalize=True).sort_index().round(3).to_dict(),
         "rank1_ties_share": r4((df.groupby("market_id")["rank1"].sum() > 1).mean()),
         "load_seconds": round(time.time() - t0, 1)})

    # ---- fit check: Newton vs the repo's gradient descent on the first months ----------------
    if not a.skip_fit_check:
        months = sorted(df["month"].unique())
        sub = df[df["month"].isin(months[:4])]
        t1 = time.time()
        gd = ConditionalLogit(FEATURES_A, l2=0.01, unpenalized=("log_p",)).fit(sub)
        t_gd = time.time() - t1
        t1 = time.time()
        nt = NewtonLogit(FEATURES_A, l2=0.01, unpenalized=("log_p",)).fit(sub)
        t_nt = time.time() - t1
        diff = np.abs(gd.weights - nt.weights)
        top = np.argsort(-diff)[:3]
        out({"section": "fit_check", "races": int(sub["race_id"].nunique()),
             "note": "same objective; the repo's gradient descent stops on a 1e-7 improvement tolerance, so it is slightly short of the optimum along collinear directions (log_p / logfield_x_logp / rank dummies)",
             "gd_final_loss": round(float(gd.history[-1]), 6), "gd_iters": len(gd.history), "gd_seconds": round(t_gd, 1),
             "newton_final_loss": round(float(nt.history[-1]), 6), "newton_iters": len(nt.history) - 1, "newton_seconds": round(t_nt, 1),
             "max_abs_weight_diff_standardised": r4(diff.max()),
             "largest_weight_diffs": {FEATURES_A[i]: [r4(gd.weights[i]), r4(nt.weights[i])] for i in top},
             "max_abs_prob_diff": r4(np.max(np.abs(gd.predict_proba(sub) - nt.predict_proba(sub)))),
             "mean_abs_prob_diff": r4(np.mean(np.abs(gd.predict_proba(sub) - nt.predict_proba(sub))))})

    # ---- walk-forward models -------------------------------------------------------------------
    cells: list[tuple[str, dict]] = []
    model_gains: list[dict] = []
    variants = []
    for fname, feats in fsets.items():
        for l2 in l2_grid:
            t1 = time.time()
            preds, coefs = walk_forward(df, feats, l2, a.burn_in)
            test = df.assign(p_model=preds)
            test = test[test["p_model"].notna()].copy()
            tmid = mid_day(test)
            co = pd.DataFrame(coefs)
            ll = log_loss_gain(test, tmid)
            tag = f"{fname}:l2={l2:g}"
            variants.append((tag, test, tmid))
            model_gains.append({"variant": tag, **{k: ll[k] for k in ("gain_per_race", "gain_se", "gain_z", "gain_halves", "positive_2se_both_halves")}})
            out({"section": "model", "variant": tag, "features": feats, "l2": l2, "refits": len(coefs),
                 "predicted_from": str(test["date"].min().date()), "predicted_to": str(test["date"].max().date()),
                 "test_markets": int(test["market_id"].nunique()), "test_runners": int(len(test)),
                 "mid_day_for_halves": str(tmid.date()),
                 "newton_iters_median": r4(co["iters"].median()) if len(co) else None,
                 "note": "race-level columns (log_field, book, harness, hour_local, weekday) are constant within a race, so their main effects cancel in the softmax and stay at 0; only their interactions with log_p can act",
                 "coef_raw_median": {f: r4(co[f].median()) for f in feats},
                 "coef_raw_first_last": {f: [r4(co[f].iloc[0]), r4(co[f].iloc[-1])] for f in feats
                                         if f not in ("log_field", "book", "harness", "hour_local", "weekday")},
                 "log_loss": ll, "top_pick": top_pick(test), "seconds": round(time.time() - t1, 1)})

            # ---- bets ----
            test["edge"] = test["p_model"] * test["ltp_pre"]
            rows = []
            for k in K_VALUES:
                g = test[test["edge"] > 1 + k]
                c = bet_cell(g, tmid)
                rows.append({"k": k, "share_of_runners": r4(len(g) / len(test)), **compact(c)})
                cells.append((f"{tag}:k={k}", c))
            out({"section": "bets", "variant": tag, "price": "ltp_pre", "rows": rows})
            # coherent book only (last traded prices plausibly still live)
            coh = test[(test["book"] >= 0.98) & (test["book"] <= 1.05)]
            rows = []
            for k in (0.0, 0.05, 0.10):
                g = coh[coh["edge"] > 1 + k]
                c = bet_cell(g, tmid)
                rows.append({"k": k, **compact(c)})
                cells.append((f"{tag}:coherent:k={k}", c))
            out({"section": "bets_coherent_book", "variant": tag, "coherent_share_of_runners": r4(len(coh) / len(test)), "rows": rows})
            # by code
            rows = []
            for k in (0.0, 0.05):
                for code in ("gallops", "harness"):
                    g = test[(test["edge"] > 1 + k) & (test["harness"] == (code == "harness"))]
                    c = bet_cell(g, tmid)
                    rows.append({"k": k, "code": code, **compact(c)})
                    cells.append((f"{tag}:{code}:k={k}", c))
            out({"section": "bets_by_code", "variant": tag, "rows": rows})
            # settled at BSP as the decision price (SP bet on the same selections)
            rows = []
            for k in (0.0, 0.05, 0.10):
                g = test[(test["edge"] > 1 + k) & test["bsp"].notna() & (test["bsp"] > 1)]
                c = bet_cell(g, tmid, price_col="bsp")
                rows.append({"k": k, **{kk: c.get(kk) for kk in ("bets", "roi_c0", "roi_c5", "roi_c8", "se", "halves_roi_c5", "positive_both_halves")}})
                cells.append((f"{tag}:bsp_bet:k={k}", c))
            out({"section": "bets_settled_at_bsp", "variant": tag, "rows": rows})
            # edge calibration
            test["edge_bin"] = pd.cut(test["edge"], [0, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 10], right=False)
            rows = []
            for b, g in test.groupby("edge_bin", observed=True):
                c = bet_cell(g, tmid)
                rows.append({"edge_bin": str(b), **{kk: c.get(kk) for kk in ("bets", "strike", "implied_shin", "model_prob", "roi_c0", "roi_c5", "se", "halves_roi_c5", "roi_c5_bsp")}})
            out({"section": "edge_calibration", "variant": tag, "rows": rows})
            # price band split for the first variant only
            if tag == f"{list(fsets)[0]}:l2={l2_grid[0]:g}":
                rows = []
                for band, g in test[test["edge"] > 1.05].groupby(bands(test.loc[test["edge"] > 1.05, "ltp_pre"], COARSE_BANDS), observed=True):
                    c = bet_cell(g, tmid)
                    rows.append({"band": str(band), **compact(c)})
                    cells.append((f"{tag}:band={band}:k=0.05", c))
                out({"section": "bets_by_price_band", "variant": tag, "k": 0.05, "rows": rows})

    out({"section": "model_gain_summary", "rows": model_gains})

    # ---- D. multiple testing, both halves, opposite-half development ---------------------------
    names = [n for n, _ in cells]
    zs = np.array([z_of(c) for _, c in cells])
    ps = np.array([one_sided_p(z) for z in zs])
    rej = benjamini_hochberg(ps, 0.05)
    tested = int((~np.isnan(ps)).sum())
    both = [n for n, c in cells if c.get("positive_both_halves") and c.get("bets", 0) >= MIN_BETS]
    out({"section": "D_multiple_testing", "bet_cells_tested_with_min_bets": tested, "bet_cells_total": len(cells),
         "min_bets": MIN_BETS, "fdr_q": 0.05,
         "bh_survivors_roi_c5": [names[i] for i in np.flatnonzero(rej)],
         "positive_both_halves_2se_c5": both,
         "best_cells_by_z": sorted([{"cell": n, "z": r4(z), "roi_c5": c["roi_c5"], "bets": c["bets"],
                                     "halves_roi_c5": c["halves_roi_c5"]}
                                    for n, c, z in zip(names, [c for _, c in cells], zs) if not math.isnan(z)],
                                   key=lambda d: -d["z"])[:8]})
    for dev in (0, 1):
        hz = np.array([half_z(c, dev) for _, c in cells])
        if np.all(np.isnan(hz)):
            continue
        i = int(np.nanargmax(hz))
        c = cells[i][1]
        other = 1 - dev
        out({"section": "D_opposite_half_development", "development_half": dev + 1, "chosen_cell": names[i],
             "dev_roi_c5": c["halves_roi_c5"][dev], "dev_z": r4(hz[i]), "dev_bets": c["halves_bets"][dev],
             "holdout_roi_c5": c["halves_roi_c5"][other], "holdout_z": r4(half_z(c, other)),
             "holdout_bets": c["halves_bets"][other],
             "holds": bool(not math.isnan(half_z(c, other)) and half_z(c, other) > 2)})
    out({"section": "done", "seconds": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

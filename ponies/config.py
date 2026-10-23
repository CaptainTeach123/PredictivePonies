"""Every tunable number and modelling assumption lives here.

Each constant is tagged with an assumption ID (``A-xx``) that is described in
``ASSUMPTIONS.md`` at the repo root. If you change a value, update the
registry entry too. ``tests/test_assumptions_registry.py`` fails if an ID is
used in code but missing from the registry (or vice versa).
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = RAW_DIR / "results"            # historical result charts (*.xml)
PP_DIR = RAW_DIR / "past_performances"       # past-performance files (*.xml)
ENTRIES_DIR = RAW_DIR / "entries"            # upcoming race cards (*.xml)
TRACKMASTER_DIR = RAW_DIR / "trackmaster"    # TrackMaster Plus XML (licensed, git-ignored)
DASHBOARD_DATA_DIR = ROOT / "docs" / "data"  # JSON consumed by the dashboard

# --------------------------------------------------------------------------
# Stage 2: feature engineering
# --------------------------------------------------------------------------
# [A-01] Recent form is summarised over the horse's last N starts.
RECENT_STARTS = 3

# [A-02] Jockey/trainer win rates use a trailing window of this many days.
CONNECTIONS_WINDOW_DAYS = 365

# [A-03] Win rates are shrunk toward a prior (Bayesian average) so a jockey
# with 1 win from 2 rides isn't rated 50%. PRIOR_RIDES is the weight of the
# prior expressed as "pseudo-rides"; PRIOR_WIN_RATE is the rate assumed for an
# unknown jockey/trainer/horse (~1 / typical field size).
PRIOR_RIDES = 20
PRIOR_WIN_RATE = 0.11

# [A-04] "Similar distance" for distance-fit means within this many furlongs.
DISTANCE_TOLERANCE_FURLONGS = 1.0

# [A-05] Track conditions are collapsed into two groups for condition-fit.
WET_CONDITIONS = {"sloppy", "muddy", "wet fast", "heavy", "soft", "yielding", "good"}

# [A-06] Layoff is capped (days) before log-transforming, so a 3-year layoff
# isn't treated as wildly different from a 1-year layoff.
MAX_LAYOFF_DAYS = 365

# [A-07] Missing features (e.g. first-time starters) are imputed with the
# median of the runner's own race field (training mean if the whole field is
# missing it), and a ``has_history`` flag lets the model learn a separate debut
# penalty/bonus. Applied in model.ConditionalLogit._prepare.
IMPUTE_STRATEGY = "race_median"

# --------------------------------------------------------------------------
# Stage 3: model
# --------------------------------------------------------------------------
# [A-08] Fraction of races (most recent by date) held out for testing.
# Splitting chronologically avoids training on the future.
TEST_FRACTION = 0.2

# [A-09] L2 regularisation strength for the conditional-logit model.
L2_PENALTY = 0.01

# [A-10] Optimiser settings (full-batch gradient descent).
LEARNING_RATE = 0.5
MAX_ITERATIONS = 2000
TOLERANCE = 1e-7

# [A-11] Market odds are NOT a model input by default, so the model is an
# independent opinion we can compare against the odds. Set True to add
# log(implied probability from morning line) as a feature.
USE_MORNING_LINE_FEATURE = False

# --------------------------------------------------------------------------
# Odds & betting evaluation
# --------------------------------------------------------------------------
# [A-12] Implied probabilities from odds are normalised to sum to 1 within a
# race (removes the bookmaker/track take, a.k.a. overround).
NORMALISE_IMPLIED_PROBS = True

# [A-13] A "value" bet is flagged when model_prob * decimal_odds - 1 exceeds
# this edge threshold.
VALUE_EDGE_THRESHOLD = 0.10

# [A-14] Backtest stakes are a flat 1 unit per bet at FINAL odds, ignoring
# that our own bets would move pari-mutuel odds.
FLAT_STAKE = 1.0

# [A-15] Calibration chart groups predictions into this many probability bins.
CALIBRATION_BINS = 10

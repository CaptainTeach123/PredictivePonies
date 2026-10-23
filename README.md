# Predictive Ponies 🐎

A horse-racing win-probability model built from race XML data, with a public dashboard.

**Dashboard:** https://captainteach123.github.io/PredictivePonies/

```
Horse racing XML files        race entries, past performances, historical results    data/raw/
        ↓
Python data processing        horses, speed figures, form, conditions, jockeys/trainers   ponies/xml_io.py, ponies/features.py
        ↓
Machine-learning model        trained on past races, tested against actual results   ponies/model.py, ponies/evaluate.py
        ↓
Race prediction dashboard     win probabilities, rankings, explanations, odds comparison   docs/ (GitHub Pages)
```

## Quick start

```bash
pip install -r requirements.txt
python -m ponies.pipeline            # builds docs/data/*.json and prints the backtest summary
python -m http.server -d docs 8000   # open http://localhost:8000
```

## Predicting an upcoming race (no install)

Open the dashboard and go to **Enter a race**. Fill in the race details and runners, or load an entries XML file, then press **Predict race**. The model runs in your browser with [Pyodide](https://pyodide.org). The first run downloads about 15 MB, which is cached after that. Use **Download as XML** to save the card so it can be added to `data/raw/entries/`.

**TrackMaster Plus XML** files (US cards with full past performances) can be loaded on the same tab and get real form-based predictions. See [DATA_SCHEMA.md](DATA_SCHEMA.md#trackmaster-plus-xml-supported-directly).

## Using real race data

1. Put XML files into `data/raw/results/`, `data/raw/past_performances/` and `data/raw/entries/`. The layout is in [DATA_SCHEMA.md](DATA_SCHEMA.md).
2. Delete the generated samples: `rm data/raw/*/SYNTHETIC_*.xml`.
3. Run `python -m ponies.pipeline` and commit, or just push and let GitHub Actions rebuild and deploy.

> The repo ships with **synthetic** sample data so everything runs out of the box. The dashboard shows a banner while that data is in use.

## The models

**Speed-figure model** (`ponies/figure_model.py`, currently used): predicts each horse's next speed figure from its own earlier starts (recent and best figures, layoff, distance and surface changes, class, weight, jockey). It then simulates the race 20,000 times, and a horse's win probability is the share of simulations it wins. Because it learns from single starts rather than complete races, it can train straight from past-performance files. A TrackMaster card is predicted by a model trained on that file's own past starts, in the browser.

**Conditional logit** (`ponies/model.py`): scores each runner as a weighted sum of features and takes a softmax within the race (Bolton & Chapman 1986, Benter 1994). It needs complete past races to learn from.

Both are trained on older races and tested on the most recent 20%. The page uses whichever has the lower log loss, and the benchmark is the betting market. Explanations show each feature's exact contribution relative to the average rival.

## Documentation map

| File | Purpose |
|------|---------|
| [AGENTS.md](AGENTS.md) | Pipeline map, commands and rules for changes. **Start here if you are an AI agent.** |
| [ASSUMPTIONS.md](ASSUMPTIONS.md) | Every assumption with its ID, location in code, config key and rationale, plus a change log |
| [DATA_SCHEMA.md](DATA_SCHEMA.md) | XML input formats |
| `ponies/config.py` | All tunable parameters |

## GitHub Pages setup (one time)

Repo **Settings → Pages → Build and deployment → Source: GitHub Actions**. After that, every push to the default branch runs the tests, rebuilds the model and deploys `docs/`.

*For research and entertainment only. Not betting advice.*

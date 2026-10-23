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

## Using real race data

1. Put XML files into `data/raw/results/`, `data/raw/past_performances/` and `data/raw/entries/`. The layout is in [DATA_SCHEMA.md](DATA_SCHEMA.md).
2. Delete the generated samples: `rm data/raw/*/SYNTHETIC_*.xml`.
3. Run `python -m ponies.pipeline` and commit, or just push and let GitHub Actions rebuild and deploy.

> The repo ships with **synthetic** sample data so everything runs out of the box. The dashboard shows a banner while that data is in use.

## The model, in one paragraph

Each runner gets a score that is a weighted sum of standardised features: recent speed figures, finishing percentile, shrunk career/jockey/trainer win rates, distance/surface/condition fit, layoff, class change, post position, weight and age. Win probability is the softmax of those scores **within the race** (a conditional logit, as in Bolton & Chapman 1986 and Benter 1994). Weights are fitted on older races and evaluated on the most recent 20%, which the model never sees during training. The benchmark is the betting market's implied probabilities. Explanations are exact per-feature contributions to a horse's score relative to the average rival.

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

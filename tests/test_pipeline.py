import json

from ponies import pipeline


def test_pipeline_end_to_end(tmp_path):
    metrics = pipeline.run(tmp_path)
    assert metrics["races"] > 0
    preds = json.loads((tmp_path / "predictions.json").read_text())
    for race in preds["races"]:
        total = sum(r["model_prob"] for r in race["runners"])
        assert abs(total - 1) < 1e-3
    assert {"model.json", "evaluation.json", "predictions.json"} <= {p.name for p in tmp_path.iterdir()}

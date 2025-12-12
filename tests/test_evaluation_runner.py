from __future__ import annotations

from pathlib import Path

from codex_audio.evaluation.runner import EvaluationRunner


def test_evaluation_runner_computes_metrics(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    annotation = tmp_path / "ann.csv"
    prediction = tmp_path / "pred.json"

    manifest.write_text(
        "audio_path,annotation_path,prediction_path,station\n"
        f"{tmp_path / 'audio.wav'},{annotation},{prediction},CKNW\n"
    )
    annotation.write_text("start_s,end_s,label\n0,5,a\n5,10,b\n")
    prediction.write_text('{"segments": [{"start": 0, "end": 4.9}, {"start": 4.9, "end": 10}]}')

    runner = EvaluationRunner(config_path=None, tolerance_seconds=0.3)
    metrics = runner.run(manifest)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0

from __future__ import annotations

from pathlib import Path

from codex_audio.evaluation import io


def test_load_manifest_and_segments(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "audio_path,annotation_path,prediction_path,station\n"
        f"{tmp_path / 'a.wav'},{tmp_path / 'ann.csv'},{tmp_path / 'pred.json'},CKNW\n"
    )

    annotation = tmp_path / "ann.csv"
    annotation.write_text("start_s,end_s,label\n0,5,intro\n5,10,story\n")
    prediction = tmp_path / "pred.json"
    prediction.write_text(
        '{"segments": [{"start": 0, "end": 4}, {"start": 4, "end": 10}]}'
    )

    examples = io.load_manifest(manifest)
    assert len(examples) == 1
    example = examples[0]
    assert example.station == "CKNW"

    refs = io.load_reference_segments(annotation)
    preds = io.load_prediction_segments(prediction)
    assert len(refs) == 2
    assert refs[0].label == "intro"
    assert len(preds) == 2

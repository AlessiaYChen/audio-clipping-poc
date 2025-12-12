from __future__ import annotations

import json
from pathlib import Path
from typing import List

import typer

from codex_audio.evaluation.runner import EvaluationRunner
from codex_audio.evaluation.sweep import SweepRunner

app = typer.Typer(help="Evaluation harness for segmentation")


@app.command()
def evaluate(
    manifest: Path = typer.Argument(..., exists=True, readable=True),
    config: Path = typer.Option(None, exists=True, readable=True),
    tolerance: float = typer.Option(3.0, min=0.0, help="Boundary matching tolerance in seconds"),
) -> None:
    runner = EvaluationRunner(config_path=config, tolerance_seconds=tolerance)
    metrics = runner.run(manifest)
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def sweep(
    manifest: Path = typer.Argument(..., exists=True, readable=True),
    config: Path = typer.Option(None, exists=True, readable=True),
    tolerance: List[float] = typer.Option(None, help="Tolerance values to try"),
) -> None:
    tolerances = tolerance or [3.0]
    grid = [{"tolerance_s": value} for value in tolerances]
    runner = SweepRunner(manifest, config)
    results = runner.run(grid)
    for result in results:
        typer.echo(
            f"tol={result.params['tolerance_s']}: F1={result.metrics.get('f1', 0):.3f}"
        )


if __name__ == "__main__":
    app()

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from codex_audio.pipeline import PipelineConfig, StorySegmentationPipeline
from codex_audio.evaluation.runner import EvaluationRunner
from codex_audio.sweeps.grid import SweepRunner

console = Console()
app = typer.Typer(help="News audio story segmentation CLI")


@app.command()
def segment(
    audio_path: Path = typer.Argument(..., exists=True, readable=True),
    station: str = typer.Option("CKNW", "--station", "-s", help="Station identifier"),
    out_dir: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory"),
    config: Optional[Path] = typer.Option(None, help="Station config override"),
) -> None:
    pipeline = StorySegmentationPipeline(
        config=PipelineConfig(station=station, config_path=config)
    )
    result = pipeline.run(audio_path=audio_path, output_dir=out_dir)
    if result.normalized_audio:
        console.print(f"Normalized WAV created at {result.normalized_audio}", style="green")
    if result.segments:
        console.print(f"Generated {len(result.segments)} segments", style="cyan")
    else:
        console.print("No segments generated", style="yellow")
    if result.clip_paths:
        console.print(f"Clips written to {result.clip_paths[0].parent}")
    if result.transcript_path:
        console.print(f"Transcript written to {result.transcript_path}")
    if result.manifest_path:
        console.print(f"Manifest written to {result.manifest_path}")


@app.command()
def eval(
    manifest: Path = typer.Option(..., exists=True, readable=True),
    config: Path = typer.Option(..., help="Station config yaml"),
    tolerance_s: float = typer.Option(3.0, min=0.0),
) -> None:
    runner = EvaluationRunner(config_path=config, tolerance_seconds=tolerance_s)
    metrics = runner.run(manifest)
    console.print(metrics)


@app.command()
def sweep(
    manifest: Path = typer.Option(..., exists=True, readable=True),
    param: list[str] = typer.Option(..., help="Parameter sweep specification"),
    config: Path = typer.Option(...),
) -> None:
    runner = SweepRunner(config_path=config)
    result = runner.run(manifest_path=manifest, param_defs=param)
    console.print(result)


def run() -> None:
    app()

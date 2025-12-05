from codex_audio.pipeline import PipelineConfig


def test_pipeline_config_station() -> None:
    config = PipelineConfig(station="CKNW")
    station_config = config.resolve_station_config()
    assert station_config.name == "CKNW"

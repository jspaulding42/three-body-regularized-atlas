from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_visualizer_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "visualizer_server.py"
    spec = importlib.util.spec_from_file_location("visualizer_server", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


visualizer = _load_visualizer_module()


def test_visualizer_taylor_prediction_compares_with_playback_endpoint() -> None:
    payload = dict(visualizer.DEFAULT_PRESET)
    payload.update(
        {
            "target_time": 0.1,
            "samples": 32,
            "prediction_method": "taylor",
            "order": 18,
        },
    )

    result = visualizer.solve_payload(payload)

    assert result["prediction"]["method"] == "taylor"
    assert result["prediction"]["status"] == "computed"
    assert len(result["simulation"]["positions"]) == 32
    assert len(result["prediction"]["positions"]) == 3
    assert result["comparison"]["max_position_error"] < 1e-8
    assert result["comparison"]["max_velocity_error"] < 1e-8


def test_visualizer_rejects_initial_collision() -> None:
    payload = dict(visualizer.DEFAULT_PRESET)
    payload["positions"] = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]

    with pytest.raises(ValueError, match="noncollision"):
        visualizer.solve_payload(payload)

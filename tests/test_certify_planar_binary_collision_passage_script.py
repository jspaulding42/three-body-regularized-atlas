import json
import subprocess
import sys
from pathlib import Path


def test_collision_passage_script_emits_certified_json():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/certify_planar_binary_collision_passage.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["collision_certified"] is True
    assert result["passage_certified"] is True
    assert result["rho_floors"]["left"] > 0.0
    assert result["rho_floors"]["right"] > 0.0
    assert result["projection_gaps"]["left"] >= 0.0
    assert result["projection_gaps"]["right"] >= 0.0
    assert result["weighted_tube_errors"]["left"]["position"] > 0.0
    assert result["weighted_tube_errors"]["right"]["velocity"] > 0.0

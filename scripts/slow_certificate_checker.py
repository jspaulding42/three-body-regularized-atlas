"""Run the slow generalized-Fuchsian checker target separately."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from three_body_symmetry.public_proof_audit import (  # noqa: E402
    PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS,
)


def _pytest_k_expression(artifact_ids: tuple[str, ...]) -> str:
    return " or ".join(
        artifact_id.split("::", maxsplit=1)[1]
        for artifact_id in artifact_ids
    )


SLOW_GENERALIZED_FUCHSIAN_CHECKER_K = (
    "independent_checker_accepts_serialized_generalized_fuchsian_stop_chart or "
    + _pytest_k_expression(PUBLIC_TC6_REQUIRED_CHECKER_ARTIFACT_IDS)
)

COMMAND = (
    sys.executable,
    "-m",
    "pytest",
    "-q",
    "tests/test_certificate_checker.py",
    "-m",
    "slow",
    "-k",
    SLOW_GENERALIZED_FUCHSIAN_CHECKER_K,
)

PUBLIC_AUDIT_PYTEST = (
    sys.executable,
    "-m",
    "pytest",
    "-q",
    "tests/test_public_proof_audit.py",
)

FULL_CERTIFICATE_CHECKER_PYTEST = (
    sys.executable,
    "-m",
    "pytest",
    "-q",
    "tests/test_certificate_checker.py",
)

COMMANDS = tuple(
    command
    for command in (
        globals().get("COMMAND"),
        PUBLIC_AUDIT_PYTEST,
        FULL_CERTIFICATE_CHECKER_PYTEST,
    )
    if command
)


def main() -> int:
    for command in COMMANDS:
        print("$ " + " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

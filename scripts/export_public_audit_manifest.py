"""Export the public TC4-TC6 audit manifest as deterministic JSON."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from three_body_symmetry.public_proof_audit import (  # noqa: E402
    build_public_tc4_tc6_audit_manifest,
)


def main() -> int:
    manifest = build_public_tc4_tc6_audit_manifest(project_root=ROOT)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

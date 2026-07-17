#!/usr/bin/env python3
"""Export and freshly replay the v0.3 finite planar-chain review bundle.

This same-implementation replay checks only the supplied finite chain.  It is
not an independent verifier and makes no completeness, termination, collision,
or general-solution claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from three_body_symmetry.planar_chain_review_artifact import (  # noqa: E402
    ReviewArtifactError,
    canonical_replay_transcript_json,
    environment_version_report,
    export_review_bundle,
    replay_transcript,
    strict_load_raw_planar_chain,
    verify_review_bundle,
)


DEFAULT_BUNDLE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "artifacts" / "v0.3.0-review" / "planar-chain"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export or replay supplied finite planar-chain certificates. "
            "This is same-implementation replay, not an independent verifier."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export",
        help="build, freshly check, and write the deterministic review bundle",
    )
    export_parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_BUNDLE_DIRECTORY,
    )

    replay_parser = subparsers.add_parser(
        "replay",
        help="strictly load one raw JSON certificate and emit a fresh transcript",
    )
    replay_parser.add_argument("raw_certificate", type=Path)
    replay_parser.add_argument("--output", type=Path)

    verify_parser = subparsers.add_parser(
        "verify-bundle",
        help="verify transport hashes and exact-match fresh replay transcripts",
    )
    verify_parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_BUNDLE_DIRECTORY,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "export":
            export_review_bundle(args.directory)
            print(f"exported and verified {args.directory}")
        elif args.command == "replay":
            certificate = strict_load_raw_planar_chain(args.raw_certificate)
            payload = canonical_replay_transcript_json(replay_transcript(certificate))
            if args.output is None:
                print(payload)
            else:
                args.output.write_bytes(payload.encode("utf-8"))
                print(f"wrote {args.output}")
        elif args.command == "verify-bundle":
            manifest = verify_review_bundle(args.directory)
            environment = environment_version_report(manifest)
            match = "matches" if environment["exact_match"] else "differs from"
            print(f"verified {args.directory}")
            print(
                f"informational runtime environment {match} recorded build environment"
            )
        else:  # pragma: no cover - argparse enforces the command union
            raise ReviewArtifactError(f"unknown command {args.command!r}")
    except (OSError, ReviewArtifactError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

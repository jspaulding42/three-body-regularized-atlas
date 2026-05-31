"""Finite-time chart-regime classification for the validated atlas pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .general_solution import (
    FiniteTimeChartSelectorError,
    evaluate_unrestricted_solution,
)
from .general_solution_theorem import (
    PositiveMassNoncollisionInputDomainCertificate,
    TheoremPipelineObligation,
    certify_positive_mass_noncollision_input_domain,
)
from .validated_atlas import ValidatedAtlasSolution


def _simultaneous_close_pair_branch_partition_from_object(value: object) -> object | None:
    """Recover a close-pair split certificate nested inside an atlas/evaluation."""

    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        candidate = getattr(current, "branch_partition", None)
        if (
            candidate is not None
            and hasattr(candidate, "branches")
            and hasattr(candidate, "binary_distance_threshold")
        ):
            return candidate
        for attribute in (
            "evaluation",
            "branch_union_evaluation",
            "next_ks_evaluation",
            "ks_competing_evaluation",
            "forward_evaluation",
        ):
            child = getattr(current, attribute, None)
            if child is not None:
                pending.append(child)
    return None


def _input_domain_flat_state(
    certificate: PositiveMassNoncollisionInputDomainCertificate | None,
) -> np.ndarray | None:
    if certificate is None or not getattr(certificate, "certified", False):
        return None
    try:
        positions = np.asarray(certificate.positions, dtype=float)
        velocities = np.asarray(certificate.velocities, dtype=float)
    except (TypeError, ValueError):
        return None
    if positions.shape != velocities.shape or positions.ndim != 2:
        return None
    return np.concatenate([positions.reshape(-1), velocities.reshape(-1)])


def _mass_sequences_match(left: object, right: object) -> bool:
    try:
        left_array = np.asarray(left, dtype=float).reshape(-1)
        right_array = np.asarray(right, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(
        left_array.shape == right_array.shape
        and left_array.size > 0
        and np.all(np.isfinite(left_array))
        and np.all(np.isfinite(right_array))
        and np.allclose(
            left_array,
            right_array,
            rtol=0.0,
            atol=64.0 * np.finfo(float).eps,
        )
    )


def _interval_like_contains_flat_point(intervals: object, point: np.ndarray) -> bool:
    try:
        interval_array = np.asarray(intervals, dtype=object).reshape(-1)
    except (TypeError, ValueError):
        return False
    point = np.asarray(point, dtype=float).reshape(-1)
    if interval_array.size != point.size or interval_array.size == 0:
        return False
    for interval, value in zip(interval_array, point):
        try:
            lower, upper = interval.as_tuple()
        except AttributeError:
            if isinstance(interval, (tuple, list)) and len(interval) == 2:
                lower, upper = interval
            else:
                lower = getattr(interval, "lower", np.nan)
                upper = getattr(interval, "upper", np.nan)
        lower = float(lower)
        upper = float(upper)
        if not (np.isfinite(lower) and np.isfinite(upper) and lower <= value <= upper):
            return False
    return True


def _interval_union_contains_flat_point(interval_union: object, point: np.ndarray) -> bool:
    if interval_union is None:
        return False
    try:
        members = tuple(interval_union)
    except TypeError:
        return False
    return any(_interval_like_contains_flat_point(member, point) for member in members)


def _validated_atlas_input_domain_matches_classifier(
    atlas: object | None,
    input_domain: PositiveMassNoncollisionInputDomainCertificate | None,
) -> bool:
    if not isinstance(atlas, ValidatedAtlasSolution):
        return False
    point = _input_domain_flat_state(input_domain)
    if point is None:
        return False
    if not _mass_sequences_match(getattr(input_domain, "masses", ()), atlas.masses):
        return False
    return bool(
        _interval_like_contains_flat_point(atlas.initial_state_interval, point)
        or _interval_union_contains_flat_point(
            atlas.initial_state_interval_union,
            point,
        )
    )


def _certified_proof_ledger_entry(atlas: object | None, name: str) -> object | None:
    proof_ledger = getattr(atlas, "proof_ledger", None)
    for entry in getattr(proof_ledger, "entries", ()):
        if getattr(entry, "name", None) == name and bool(getattr(entry, "certified", False)):
            return entry
    return None


def _append_certified_atlas_proof_obligations(
    obligations: list[TheoremPipelineObligation],
    atlas: object | None,
    *,
    selected_route_id: str | None,
) -> None:
    """Mirror successful local split consumption from the atlas proof ledger.

    These obligations are local evidence for the selected finite-time route.
    They do not assert the separate global theorem that every future
    set-valued branch tree is recursively consumable.
    """

    existing = {obligation.obligation for obligation in obligations}
    for name in (
        "simultaneous_close_pair_partition",
        "finite_time_branch_union_consumption",
        "ks_event_order_partition",
        "finite_time_event_order_branch_union_consumption",
        "finite_time_atlas_loop_progress",
    ):
        if name in existing:
            continue
        entry = _certified_proof_ledger_entry(atlas, name)
        if entry is None:
            continue
        obligations.append(
            TheoremPipelineObligation(
                obligation=name,
                certified=True,
                source=getattr(entry, "source", "ValidatedAtlasSolution.proof_ledger"),
                detail=(
                    f"selected_route_id={selected_route_id}; "
                    f"proof_ledger_entry={name}; "
                    f"{getattr(entry, 'detail', '')}"
                ),
            )
        )
        existing.add(name)


@dataclass(frozen=True)
class FiniteTimeRegimeClassificationCertificate:
    """Constructor-derived finite-time chart classification.

    This is deliberately scoped to one finite target time.  It does not assert
    compact-time coverage, all-future recurrence, or arbitrary-data global
    exhaustion; those remain obligations for the global theorem pipeline.
    """

    input_domain_certificate: PositiveMassNoncollisionInputDomainCertificate | None
    target_time: float
    validated_atlas: ValidatedAtlasSolution | None
    selected_route_id: str | None
    selector_trace: object | None
    branch_partition: object | None
    obligations: tuple[TheoremPipelineObligation, ...]
    failure_reason: str | None = None
    failure_obligations: tuple[str, ...] = ()

    @property
    def input_domain_certified(self) -> bool:
        return bool(getattr(self.input_domain_certificate, "certified", False))

    @property
    def target_time_certified(self) -> bool:
        return bool(np.isfinite(self.target_time))

    @property
    def atlas_certified(self) -> bool:
        return bool(
            isinstance(self.validated_atlas, ValidatedAtlasSolution)
            and getattr(self.validated_atlas, "proof_certified", False)
        )

    @property
    def selector_certified(self) -> bool:
        return bool(getattr(self.selector_trace, "certified", False))

    @property
    def branch_partition_certified(self) -> bool:
        return bool(getattr(self.branch_partition, "certified", False))

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        ))

    @property
    def chart_types(self) -> tuple[str, ...]:
        return tuple(
            str(getattr(chart, "chart_type", ""))
            for chart in getattr(self.validated_atlas, "charts", ())
        )


def classify_finite_time_regime(
    masses: Any,
    positions: Any,
    velocities: Any,
    target_time: float,
    **solver_options: Any,
) -> FiniteTimeRegimeClassificationCertificate:
    """Classify and construct the validated finite-time chart route.

    The constructor consumes only initial data and evaluator options.  It runs
    the public finite-time validated atlas selector and records the selected
    ordinary, planar LC, spatial KS, or Sundman route.  Selector failures are
    returned as missing obligations instead of being promoted to theorem claims.
    """

    if "method" in solver_options:
        raise ValueError("classify_finite_time_regime always uses method='validated_atlas'")
    target_time = float(target_time)
    input_domain: PositiveMassNoncollisionInputDomainCertificate | None
    input_failure: str | None = None
    try:
        input_domain = certify_positive_mass_noncollision_input_domain(
            masses,
            positions,
            velocities,
        )
    except (TypeError, ValueError) as exc:
        input_domain = None
        input_failure = str(exc)

    obligations = [
        TheoremPipelineObligation(
            obligation="positive_mass_noncollision_input_domain",
            certified=bool(getattr(input_domain, "certified", False)),
            source="certify_positive_mass_noncollision_input_domain",
            detail=input_failure or "finite-time classifier input domain",
        ),
        TheoremPipelineObligation(
            obligation="finite_target_time",
            certified=bool(np.isfinite(target_time)),
            source="classify_finite_time_regime",
            detail=f"target_time={target_time!r}",
        ),
    ]
    atlas: ValidatedAtlasSolution | None = None
    selector_trace: object | None = None
    selected_route_id: str | None = None
    failure_reason: str | None = input_failure
    failure_obligations: tuple[str, ...] = ()
    branch_partition: object | None = None
    if input_domain is not None and np.isfinite(target_time):
        try:
            atlas = evaluate_unrestricted_solution(
                masses,
                positions,
                velocities,
                target_time,
                method="validated_atlas",
                **solver_options,
            )
            selector_trace = getattr(atlas, "selector_trace", None)
            selected_route_id = getattr(selector_trace, "selected_route_id", None)
            for attempt in getattr(selector_trace, "attempts", ()):
                candidate_partition = getattr(attempt, "branch_partition", None)
                if candidate_partition is not None:
                    branch_partition = candidate_partition
                    break
            if branch_partition is None:
                branch_partition = _simultaneous_close_pair_branch_partition_from_object(
                    atlas,
                )
            failure_reason = None
        except FiniteTimeChartSelectorError as exc:
            failure_reason = str(exc)
            failure_obligations = exc.missing_obligations
            for attempt in exc.blocking_attempts:
                candidate_partition = getattr(attempt, "branch_partition", None)
                if candidate_partition is not None:
                    branch_partition = candidate_partition
                    break
        except (RuntimeError, ValueError) as exc:
            failure_reason = str(exc)

    obligations.extend(
        [
            TheoremPipelineObligation(
                obligation="finite_time_validated_atlas_type",
                certified=isinstance(atlas, ValidatedAtlasSolution),
                source=(
                    type(atlas).__name__
                    if atlas is not None
                    else "evaluate_unrestricted_solution(method='validated_atlas')"
                ),
                detail="finite-time classifier must consume a real ValidatedAtlasSolution",
            ),
            TheoremPipelineObligation(
                obligation="finite_time_validated_atlas",
                certified=bool(
                    isinstance(atlas, ValidatedAtlasSolution)
                    and atlas.proof_certified
                ),
                source="evaluate_unrestricted_solution(method='validated_atlas')",
                detail=(
                    (
                        f"selected_route_id={selected_route_id}; missing="
                        + ",".join(atlas.missing_certification_obligations)
                    )
                    if atlas is not None and not atlas.proof_certified
                    else f"selected_route_id={selected_route_id}"
                    if atlas is not None
                    else failure_reason or "validated atlas constructor did not run"
                ),
            ),
            TheoremPipelineObligation(
                obligation="finite_time_atlas_input_domain_matches_classifier",
                certified=_validated_atlas_input_domain_matches_classifier(
                    atlas,
                    input_domain,
                ),
                source=(
                    type(atlas).__name__
                    if atlas is not None
                    else "evaluate_unrestricted_solution(method='validated_atlas')"
                ),
                detail=(
                    "finite-time atlas masses and initial interval must bind "
                    "to the classifier input domain"
                ),
            ),
            TheoremPipelineObligation(
                obligation="finite_time_chart_selector",
                certified=bool(selector_trace is not None and selector_trace.certified),
                source=(
                    "FiniteTimeChartSelectorTrace"
                    if selector_trace is not None
                    else "classify_finite_time_regime"
                ),
                detail=(
                    f"selected_route_id={selected_route_id}"
                    if selector_trace is not None
                    else failure_reason or "missing finite-time selector trace"
                ),
            ),
        ]
    )
    if branch_partition is not None:
        obligations.append(
            TheoremPipelineObligation(
                obligation="simultaneous_close_pair_partition",
                certified=bool(getattr(branch_partition, "certified", False)),
                source="certify_simultaneous_close_pair_partition",
                detail=(
                    f"branches={len(getattr(branch_partition, 'branches', ()))}; "
                    f"missing={','.join(getattr(branch_partition, 'missing_obligations', ()))}"
                ),
            )
        )
        if getattr(branch_partition, "certified", False) and atlas is None:
            obligations.append(
                TheoremPipelineObligation(
                    obligation="finite_time_branch_union_consumption",
                    certified=False,
                    source="classify_finite_time_regime",
                    detail=(
                        "simultaneous close-pair branches were constructor-certified, "
                        "but this public finite-time atlas attempt did not certify "
                        "spatial branch-union consumption"
                    ),
                )
            )
    obligations.extend(
        TheoremPipelineObligation(
            obligation=obligation,
            certified=False,
            source="FiniteTimeChartSelectorError",
            detail=failure_reason or "finite-time selector failure",
        )
        for obligation in failure_obligations
    )
    _append_certified_atlas_proof_obligations(
        obligations,
        atlas,
        selected_route_id=selected_route_id,
    )
    return FiniteTimeRegimeClassificationCertificate(
        input_domain_certificate=input_domain,
        target_time=target_time,
        validated_atlas=atlas,
        selected_route_id=selected_route_id,
        selector_trace=selector_trace,
        branch_partition=branch_partition,
        obligations=tuple(obligations),
        failure_reason=failure_reason,
        failure_obligations=failure_obligations,
    )

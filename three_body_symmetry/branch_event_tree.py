"""Normalized finite branch/event-order tree certificates.

The finite-time atlas constructors already produce several branch-like
objects: simultaneous close-pair state partitions, KS event-order bisections,
and ambiguous first-event alternatives.  This module gives those supplied
objects one theorem-facing shape without claiming that arbitrary recursive
refinement terminates or that equality strata have been consumed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BranchEventTreeObligation:
    """One obligation in a supplied finite branch/event tree certificate."""

    obligation: str
    certified: bool
    detail: str
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "certified", self.certified is True)
        object.__setattr__(
            self,
            "required",
            self.required if type(self.required) is bool else True,
        )


@dataclass(frozen=True)
class BranchEventTreeLeafCertificate:
    """Normalized finite leaf from a constructor-derived branch partition."""

    leaf_id: str
    leaf_type: str
    decision: str
    source_type: str
    depth: int
    certified: bool
    missing_obligations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "certified", self.certified is True)

    @property
    def pending(self) -> bool:
        return bool(
            self.missing_obligations
            or "ambiguous" in self.leaf_type
            or "pending" in self.decision
            or "still_ambiguous" in self.decision
            or "not_certified" in self.decision
        )

    @property
    def equality_stratum_pending(self) -> bool:
        return bool(
            self.pending
            and (
                "ambiguous" in self.leaf_type
                or "equality" in self.leaf_type
                or "tie" in self.leaf_type
                or "ambiguous" in self.decision
                or "equality" in self.decision
                or "tie" in self.decision
            )
        )

    @property
    def well_formed(self) -> bool:
        return bool(
            self.leaf_id
            and self.leaf_type
            and self.decision
            and self.source_type
            and self.depth >= 0
            and all(str(obligation) for obligation in self.missing_obligations)
            and (not self.certified or not self.missing_obligations)
        )


@dataclass(frozen=True)
class BranchEventTreeCertificate:
    """Named finite tree consumed by finite atlas-or-stop theorem glue."""

    tree_kind: str
    source_type: str
    leaf_certificates: tuple[BranchEventTreeLeafCertificate, ...]
    cover_certified: bool
    leaf_decisions_certified: bool
    equality_strata_explicit: bool
    obligations: tuple[BranchEventTreeObligation, ...]
    theorem_id: str = "supplied_finite_branch_event_tree"

    def __post_init__(self) -> None:
        object.__setattr__(self, "cover_certified", self.cover_certified is True)
        object.__setattr__(
            self,
            "leaf_decisions_certified",
            self.leaf_decisions_certified is True,
        )
        object.__setattr__(
            self,
            "equality_strata_explicit",
            self.equality_strata_explicit is True,
        )

    @property
    def branches(self) -> tuple[BranchEventTreeLeafCertificate, ...]:
        if self.tree_kind == "simultaneous_close_pair_branch_partition":
            return self.leaf_certificates
        return ()

    @property
    def leaves(self) -> tuple[BranchEventTreeLeafCertificate, ...]:
        if self.tree_kind == "ks_event_order_partition":
            return self.leaf_certificates
        return ()

    @property
    def branch_leaves(self) -> tuple[BranchEventTreeLeafCertificate, ...]:
        if self.tree_kind == "ambiguous_event_order_partition":
            return self.leaf_certificates
        return ()

    @property
    def leaf_count(self) -> int:
        return len(self.leaf_certificates)

    @property
    def certified_leaf_count(self) -> int:
        return sum(
            isinstance(leaf, BranchEventTreeLeafCertificate)
            and leaf.certified is True
            for leaf in self.leaf_certificates
        )

    @property
    def pending_leaf_count(self) -> int:
        return sum(
            isinstance(leaf, BranchEventTreeLeafCertificate) and leaf.pending
            for leaf in self.leaf_certificates
        )

    @property
    def equality_stratum_leaf_count(self) -> int:
        return sum(
            isinstance(leaf, BranchEventTreeLeafCertificate)
            and leaf.equality_stratum_pending
            for leaf in self.leaf_certificates
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.theorem_id
            and self.leaf_certificates
            and self.cover_certified
            and self.leaf_decisions_certified
            and self.obligations
            and any(
                isinstance(obligation, BranchEventTreeObligation)
                and obligation.required
                for obligation in self.obligations
            )
            and all(
                isinstance(leaf, BranchEventTreeLeafCertificate)
                and leaf.certified is True
                for leaf in self.leaf_certificates
            )
            and all(
                isinstance(obligation, BranchEventTreeObligation)
                and obligation.certified is True
                for obligation in self.obligations
                if isinstance(obligation, BranchEventTreeObligation)
                and obligation.required
            )
            and all(
                isinstance(obligation, BranchEventTreeObligation)
                for obligation in self.obligations
            )
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        own: list[str] = []
        if not self.obligations:
            own.append("finite_branch_event_tree_obligations_present")
        if self.obligations and not any(
            isinstance(obligation, BranchEventTreeObligation)
            and obligation.required
            for obligation in self.obligations
        ):
            own.append("finite_branch_event_tree_required_obligation_present")
        for obligation in self.obligations:
            if not isinstance(obligation, BranchEventTreeObligation):
                own.append("finite_branch_event_tree_obligation_type")
                continue
            if obligation.required and obligation.certified is not True:
                own.append(obligation.obligation)
        leaf: list[str] = []
        for certificate in self.leaf_certificates:
            if not isinstance(certificate, BranchEventTreeLeafCertificate):
                leaf.append("finite_branch_event_tree_leaf_type")
                continue
            leaf.extend(
                f"{certificate.leaf_id}:{obligation}"
                for obligation in certificate.missing_obligations
            )
        return tuple(dict.fromkeys((*own, *leaf)))


def certify_supplied_branch_event_tree(
    partition: object,
    *,
    tree_kind: str | None = None,
) -> BranchEventTreeCertificate:
    """Normalize an existing finite branch/event-order partition object."""

    inferred_kind, source_leaves = _partition_kind_and_source_leaves(partition)
    kind = str(tree_kind or inferred_kind)
    leaf_certificates = tuple(
        _normalize_leaf(leaf, fallback_index=index)
        for index, leaf in enumerate(source_leaves)
    )
    duplicate_leaf_ids = _duplicate_ids(
        tuple(leaf.leaf_id for leaf in leaf_certificates)
    )
    cover_certified = _partition_cover_certified(partition, kind)
    leaf_decisions_certified = _partition_leaf_decisions_certified(
        partition,
        leaf_certificates,
        kind,
    )
    equality_strata_explicit = bool(
        leaf_certificates
        and all(
            not leaf.pending or leaf.equality_stratum_pending
            for leaf in leaf_certificates
        )
    )
    obligations = (
        BranchEventTreeObligation(
            obligation="finite_branch_event_tree_kind_supported",
            certified=kind
            in {
                "simultaneous_close_pair_branch_partition",
                "ks_event_order_partition",
                "ambiguous_event_order_partition",
            },
            detail=f"tree_kind={kind}",
        ),
        BranchEventTreeObligation(
            obligation="finite_branch_event_tree_leaves_present",
            certified=bool(leaf_certificates),
            detail=f"leaf_count={len(leaf_certificates)}",
        ),
        BranchEventTreeObligation(
            obligation="finite_branch_event_tree_cover",
            certified=cover_certified,
            detail=f"source_type={type(partition).__name__}",
        ),
        BranchEventTreeObligation(
            obligation="finite_branch_event_tree_leaf_decisions",
            certified=leaf_decisions_certified,
            detail=(
                f"certified_leaf_count={sum(leaf.certified for leaf in leaf_certificates)}; "
                f"pending_leaf_count={sum(leaf.pending for leaf in leaf_certificates)}"
            ),
        ),
        BranchEventTreeObligation(
            obligation="finite_branch_event_tree_leaf_ids_unique",
            certified=not duplicate_leaf_ids,
            detail="duplicate_leaf_ids=" + ",".join(duplicate_leaf_ids),
        ),
        BranchEventTreeObligation(
            obligation="equality_or_pending_strata_explicit",
            certified=equality_strata_explicit,
            detail=(
                f"equality_stratum_leaf_count="
                f"{sum(leaf.equality_stratum_pending for leaf in leaf_certificates)}"
            ),
            required=False,
        ),
        BranchEventTreeObligation(
            obligation="arbitrary_recursive_termination_not_claimed",
            certified=True,
            detail=(
                "normalizes a supplied finite tree only; arbitrary recursive "
                "branch/event-order termination remains a separate theorem"
            ),
            required=False,
        ),
    )
    return BranchEventTreeCertificate(
        tree_kind=kind,
        source_type=type(partition).__name__ if partition is not None else "missing",
        leaf_certificates=leaf_certificates,
        cover_certified=cover_certified,
        leaf_decisions_certified=leaf_decisions_certified,
        equality_strata_explicit=equality_strata_explicit,
        obligations=obligations,
    )


def _duplicate_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _partition_kind_and_source_leaves(partition: object) -> tuple[str, tuple[object, ...]]:
    if partition is None:
        return "missing_partition", ()
    branches = tuple(getattr(partition, "branches", ()) or ())
    if branches:
        return "simultaneous_close_pair_branch_partition", branches
    leaves = tuple(getattr(partition, "leaves", ()) or ())
    if leaves:
        return "ks_event_order_partition", leaves
    branch_leaves = tuple(getattr(partition, "branch_leaves", ()) or ())
    if branch_leaves:
        return "ambiguous_event_order_partition", branch_leaves
    leaf_certificates = tuple(getattr(partition, "leaf_certificates", ()) or ())
    if leaf_certificates:
        return str(getattr(partition, "tree_kind", type(partition).__name__)), leaf_certificates
    return type(partition).__name__, ()


def _normalize_leaf(
    leaf: object,
    *,
    fallback_index: int,
) -> BranchEventTreeLeafCertificate:
    if isinstance(leaf, BranchEventTreeLeafCertificate):
        return leaf
    leaf_id = str(
        getattr(
            leaf,
            "branch_id",
            getattr(
                leaf,
                "leaf_id",
                getattr(leaf, "assumed_first_event_id", f"leaf:{fallback_index}"),
            ),
        )
    )
    leaf_type = str(
        getattr(
            leaf,
            "leaf_type",
            getattr(leaf, "event_type", type(leaf).__name__),
        )
    )
    decision = str(
        getattr(
            leaf,
            "decision",
            getattr(leaf, "selected_pair", getattr(leaf, "assumed_first_event_id", leaf_type)),
        )
    )
    depth = int(getattr(leaf, "depth", 0) or 0)
    missing = tuple(str(value) for value in getattr(leaf, "missing_obligations", ()) or ())
    certified = getattr(leaf, "certified", False) is True and not missing
    return BranchEventTreeLeafCertificate(
        leaf_id=leaf_id,
        leaf_type=leaf_type,
        decision=decision,
        source_type=type(leaf).__name__,
        depth=depth,
        certified=certified,
        missing_obligations=missing,
    )


def _partition_cover_certified(partition: object, kind: str) -> bool:
    if partition is None:
        return False
    if isinstance(partition, BranchEventTreeCertificate):
        return partition.cover_certified
    if kind == "ambiguous_event_order_partition":
        return getattr(partition, "event_alternative_cover_certified", False) is True
    return (
        getattr(
            partition,
            "recursive_bisection_cover_certified",
            getattr(partition, "certified", False),
        )
        is True
    )


def _partition_leaf_decisions_certified(
    partition: object,
    leaf_certificates: tuple[BranchEventTreeLeafCertificate, ...],
    kind: str,
) -> bool:
    if partition is None or not leaf_certificates:
        return False
    if isinstance(partition, BranchEventTreeCertificate):
        return partition.leaf_decisions_certified
    if kind == "ambiguous_event_order_partition":
        return bool(
            getattr(partition, "state_partition_certified", False) is True
            and all(leaf.certified for leaf in leaf_certificates)
        )
    decision_flag = getattr(
        partition,
        "branch_cover_certified",
        getattr(
            partition,
            "leaf_decisions_certified",
            None,
        ),
    )
    if decision_flag is None:
        return all(leaf.certified for leaf in leaf_certificates)
    return bool(
        decision_flag is True and all(leaf.certified for leaf in leaf_certificates)
    )

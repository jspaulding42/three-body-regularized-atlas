"""Stratified supplied branch/event-order tree certificates.

The branch-event normalizer records that a finite tree exists.  This module
adds the next proof-facing layer: every supplied leaf is assigned to a named
analytic stratum, so zero-margin leaves stay visible as event ties, total
collision clusters, selector leaves, or named unsupported strata instead of
collapsing into a generic "missing branch theorem" flag.

Most constructors here still certify only a supplied finite tree.  The
polynomial-decision constructors derive finite one-dimensional trees from
explicit discriminator data, verified equality brackets, and low-degree root
formulas; they still do not claim arbitrary recursive refinement termination.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations, product

import numpy as np

from .branch_event_tree import (
    BranchEventTreeCertificate,
    BranchEventTreeLeafCertificate,
    BranchEventTreeObligation,
    certify_supplied_branch_event_tree,
)
from .intervals import (
    FloatInterval,
    interval_polyder,
    interval_polynomial_eval,
    interval_sign,
)


SUPPORTED_STRATIFIED_LEAF_KINDS = {
    "positive_margin_unique_event",
    "no_event_before_target",
    "separated_binary_entry",
    "simultaneous_event_equality",
    "total_collision_cluster",
    "selector_policy",
    "unsupported_analytic_stratum",
}

TERMINAL_STRATIFIED_LEAF_KINDS = {
    "positive_margin_unique_event",
    "no_event_before_target",
    "separated_binary_entry",
    "total_collision_cluster",
    "selector_policy",
}


@dataclass(frozen=True)
class AnalyticDecisionFunctionCertificate:
    """Positive-margin decision evidence for one analytic selector function."""

    function_id: str
    function_kind: str
    margin_lower_bound: float
    lipschitz_bound: float
    certified: bool
    missing_obligations: tuple[str, ...] = ()

    @property
    def positive_margin_certified(self) -> bool:
        return bool(
            self.certified
            and not self.missing_obligations
            and self.margin_lower_bound > 0.0
            and self.lipschitz_bound >= 0.0
        )

    @property
    def proof_certified(self) -> bool:
        return self.positive_margin_certified


@dataclass(frozen=True)
class EqualityStratumCertificate:
    """Named zero-margin equality stratum for a branch/event leaf."""

    stratum_id: str
    defining_function_ids: tuple[str, ...]
    leaf_kind: str
    isolation_certified: bool
    resolution_policy: str
    certified: bool
    missing_obligations: tuple[str, ...] = ()

    @property
    def well_formed(self) -> bool:
        return bool(
            self.stratum_id
            and self.defining_function_ids
            and self.leaf_kind in SUPPORTED_STRATIFIED_LEAF_KINDS
            and self.resolution_policy
            and (not self.certified or not self.missing_obligations)
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified
            and self.well_formed
            and self.isolation_certified
        )


@dataclass(frozen=True)
class EventOrderTieLeafCertificate:
    """Explicit first-event tie/equality leaf."""

    leaf_id: str
    tied_event_ids: tuple[str, ...]
    equality_stratum: EqualityStratumCertificate
    certified: bool
    missing_obligations: tuple[str, ...] = ()

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.leaf_id
            and len(self.tied_event_ids) >= 2
            and self.certified
            and not self.missing_obligations
            and self.equality_stratum.proof_certified
        )


@dataclass(frozen=True)
class SelectorPolicyLeafCertificate:
    """Explicit terminal selector policy for a zero-margin leaf."""

    leaf_id: str
    selector_policy_id: str
    defining_function_ids: tuple[str, ...]
    isolation_certified: bool
    certified: bool
    missing_obligations: tuple[str, ...] = ()

    @property
    def equality_stratum(self) -> EqualityStratumCertificate:
        return EqualityStratumCertificate(
            stratum_id=f"selector:{self.leaf_id}",
            defining_function_ids=self.defining_function_ids,
            leaf_kind="selector_policy",
            isolation_certified=self.isolation_certified,
            resolution_policy=self.selector_policy_id,
            certified=self.certified and not self.missing_obligations,
            missing_obligations=self.missing_obligations,
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.leaf_id
            and self.selector_policy_id
            and self.defining_function_ids
            and self.isolation_certified
            and self.certified
            and not self.missing_obligations
        )


@dataclass(frozen=True)
class TotalCollisionClusterLeafCertificate:
    """Explicit multi-pair/total-collision cluster leaf."""

    leaf_id: str
    cluster_pair_ids: tuple[str, ...]
    stop_or_selector_policy: str
    entry_certificate: object | None
    certified: bool
    missing_obligations: tuple[str, ...] = ()

    @property
    def proof_certified(self) -> bool:
        entry_certified = bool(
            self.entry_certificate is not None
            and (
                getattr(self.entry_certificate, "proof_certified", False)
                or getattr(self.entry_certificate, "certified", False)
            )
        )
        return bool(
            self.leaf_id
            and len(self.cluster_pair_ids) >= 2
            and self.stop_or_selector_policy
            and self.certified
            and entry_certified
            and not self.missing_obligations
        )


@dataclass(frozen=True)
class StratifiedBranchLeafCertificate:
    """One normalized branch/event leaf with an analytic stratum label."""

    leaf_id: str
    source_leaf_id: str
    leaf_kind: str
    terminal_response_kind: str
    terminal_response_certified: bool
    source_leaf_certified: bool
    depth: int = 0
    decision_functions: tuple[AnalyticDecisionFunctionCertificate, ...] = ()
    equality_stratum: EqualityStratumCertificate | None = None
    event_order_tie: EventOrderTieLeafCertificate | None = None
    total_collision_cluster: TotalCollisionClusterLeafCertificate | None = None
    missing_obligations: tuple[str, ...] = ()

    @property
    def supported_kind(self) -> bool:
        return self.leaf_kind in SUPPORTED_STRATIFIED_LEAF_KINDS

    @property
    def unsupported(self) -> bool:
        return self.leaf_kind == "unsupported_analytic_stratum"

    @property
    def equality_or_zero_margin(self) -> bool:
        return self.leaf_kind in {
            "simultaneous_event_equality",
            "total_collision_cluster",
            "selector_policy",
            "unsupported_analytic_stratum",
        }

    @property
    def well_formed(self) -> bool:
        return bool(
            self.leaf_id
            and self.source_leaf_id
            and self.supported_kind
            and self.terminal_response_kind
            and self.depth >= 0
            and (not self.terminal_response_certified or not self.missing_obligations)
        )

    @property
    def stratum_certified(self) -> bool:
        if self.unsupported or not self.well_formed:
            return False
        if self.leaf_kind == "positive_margin_unique_event":
            return bool(
                self.decision_functions
                and all(
                    decision.proof_certified
                    for decision in self.decision_functions
                )
            )
        if self.leaf_kind == "simultaneous_event_equality":
            return bool(
                self.event_order_tie is not None
                and self.event_order_tie.proof_certified
            )
        if self.leaf_kind == "total_collision_cluster":
            return bool(
                self.total_collision_cluster is not None
                and self.total_collision_cluster.proof_certified
            )
        if self.leaf_kind == "selector_policy":
            return bool(
                self.equality_stratum is not None
                and self.equality_stratum.proof_certified
            )
        return True

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.source_leaf_certified
            and self.stratum_certified
            and self.terminal_response_certified
            and not self.missing_obligations
        )


@dataclass(frozen=True)
class StratifiedBranchTreeCertificate:
    """A finite supplied branch tree with named analytic leaf strata."""

    source_tree: BranchEventTreeCertificate
    leaf_certificates: tuple[StratifiedBranchLeafCertificate, ...]
    preserves_branch_tree: bool
    leaf_taxonomy_certified: bool
    cover_certified: bool
    recursive_exhaustion_certified: bool = False
    theorem_id: str = "supplied_stratified_branch_event_tree"

    @property
    def leaf_count(self) -> int:
        return len(self.leaf_certificates)

    @property
    def leaf_kinds(self) -> tuple[str, ...]:
        return tuple(leaf.leaf_kind for leaf in self.leaf_certificates)

    @property
    def unsupported_leaf_count(self) -> int:
        return sum(leaf.unsupported for leaf in self.leaf_certificates)

    @property
    def zero_margin_leaf_count(self) -> int:
        return sum(leaf.equality_or_zero_margin for leaf in self.leaf_certificates)

    @property
    def terminal_response_count(self) -> int:
        return sum(leaf.terminal_response_certified for leaf in self.leaf_certificates)

    @property
    def certified(self) -> bool:
        return bool(
            self.theorem_id
            and self.cover_certified
            and self.preserves_branch_tree
            and self.leaf_taxonomy_certified
            and self.leaf_certificates
        )

    @property
    def supplied_tree_proof_certified(self) -> bool:
        return bool(
            self.certified
            and self.unsupported_leaf_count == 0
            and all(leaf.proof_certified for leaf in self.leaf_certificates)
        )

    @property
    def proof_certified(self) -> bool:
        return self.supplied_tree_proof_certified

    @property
    def recursive_theorem_certified(self) -> bool:
        return bool(
            self.supplied_tree_proof_certified
            and self.recursive_exhaustion_certified
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.cover_certified:
            missing.append("stratified_branch_tree_cover")
        if not self.preserves_branch_tree:
            missing.append("stratified_branch_tree_preserves_source_leaves")
        if not self.leaf_taxonomy_certified:
            missing.append("stratified_branch_tree_leaf_taxonomy")
        for leaf in self.leaf_certificates:
            if leaf.unsupported:
                missing.append(f"{leaf.leaf_id}:unsupported_analytic_stratum")
            if not leaf.source_leaf_certified:
                missing.append(f"{leaf.leaf_id}:source_leaf_not_certified")
            if not leaf.stratum_certified:
                missing.append(f"{leaf.leaf_id}:stratum_not_certified")
            if not leaf.terminal_response_certified:
                missing.append(f"{leaf.leaf_id}:terminal_response_missing")
            missing.extend(f"{leaf.leaf_id}:{item}" for item in leaf.missing_obligations)
        if not self.recursive_exhaustion_certified:
            missing.append("arbitrary_recursive_stratified_exhaustion_not_claimed")
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class RecursiveStratifiedConsumptionObligation:
    """One obligation for recursive stratified branch/event consumption."""

    obligation: str
    certified: bool
    detail: str
    required: bool = True


@dataclass(frozen=True)
class RecursiveStratifiedLeafConsumption:
    """How one stratified leaf is consumed by the recursive descent theorem."""

    leaf_id: str
    leaf_kind: str
    consumption_kind: str
    terminal_certified: bool
    child_certified: bool
    descent_certified: bool
    child_dimension: int | None = None
    child_rank: int | None = None
    child_source_type: str = ""
    missing_obligations: tuple[str, ...] = ()

    @property
    def recursive(self) -> bool:
        return self.consumption_kind == "lower_dimensional_recursive_stratum"

    @property
    def unsupported(self) -> bool:
        return self.leaf_kind == "unsupported_analytic_stratum"

    @property
    def certified(self) -> bool:
        return bool(
            not self.unsupported
            and not self.missing_obligations
            and (
                self.terminal_certified
                or (
                    self.recursive
                    and self.child_certified
                    and self.descent_certified
                )
            )
        )


@dataclass(frozen=True)
class RecursiveStratifiedBranchEventConsumptionCertificate:
    """Constructor-derived recursive consumption of a supplied stratified tree.

    This is the finite recursive version of the branch/event-order theorem:
    every supplied leaf must either be a certified terminal response or carry a
    certified child consumption certificate on a strictly lower-dimensional
    equality stratum, or on a lower-rank stratum at the same dimension.
    Unsupported analytic strata remain explicit blockers.
    """

    source_tree: StratifiedBranchTreeCertificate
    recursion_kind: str
    root_dimension: int
    root_rank: int | None
    leaf_consumptions: tuple[RecursiveStratifiedLeafConsumption, ...]
    node_count: int
    recursion_depth: int
    statement: str
    proof_sketch: str
    obligations: tuple[RecursiveStratifiedConsumptionObligation, ...]
    theorem_id: str = "recursive_stratified_branch_event_consumption"

    @property
    def leaf_count(self) -> int:
        return len(self.leaf_consumptions)

    @property
    def terminal_leaf_count(self) -> int:
        return sum(leaf.terminal_certified for leaf in self.leaf_consumptions)

    @property
    def recursive_leaf_count(self) -> int:
        return sum(leaf.recursive for leaf in self.leaf_consumptions)

    @property
    def unsupported_leaf_count(self) -> int:
        return sum(leaf.unsupported for leaf in self.leaf_consumptions)

    @property
    def strict_descent_edge_count(self) -> int:
        return sum(
            leaf.recursive and leaf.descent_certified
            for leaf in self.leaf_consumptions
        )

    @property
    def unresolved_descent_edge_count(self) -> int:
        return sum(
            leaf.recursive and not leaf.descent_certified
            for leaf in self.leaf_consumptions
        )

    @property
    def descent_well_founded(self) -> bool:
        return self.recursive_leaf_count == self.strict_descent_edge_count

    @property
    def certified(self) -> bool:
        return bool(
            self.theorem_id
            and self.source_tree.certified
            and self.leaf_consumptions
            and self.descent_well_founded
            and all(leaf.certified for leaf in self.leaf_consumptions)
            and all(
                obligation.certified
                for obligation in self.obligations
                if obligation.required
            )
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def recursive_theorem_certified(self) -> bool:
        return self.certified

    @property
    def source_tree_kind(self) -> str:
        source_tree = getattr(self.source_tree, "source_tree", None)
        return str(getattr(source_tree, "tree_kind", ""))

    @property
    def constructor_source_type(self) -> str:
        source_tree = getattr(self.source_tree, "source_tree", None)
        return str(getattr(source_tree, "source_type", ""))

    @property
    def child_constructor_source_types(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                leaf.child_source_type
                for leaf in self.leaf_consumptions
                if leaf.child_source_type
            )
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = [
            obligation.obligation
            for obligation in self.obligations
            if obligation.required and not obligation.certified
        ]
        for leaf in self.leaf_consumptions:
            missing.extend(
                f"{leaf.leaf_id}:{item}" for item in leaf.missing_obligations
            )
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class PolynomialDecisionStratumCertificate:
    """One sign or equality stratum from a 1D polynomial discriminator."""

    stratum_id: str
    stratum_kind: str
    interval: tuple[float, float]
    value_interval: tuple[float, float]
    sign: int
    derivative_interval: tuple[float, float] | None = None
    second_derivative_interval: tuple[float, float] | None = None
    root_multiplicities: tuple[int, ...] = ()
    certified: bool = False
    missing_obligations: tuple[str, ...] = ()

    @property
    def equality(self) -> bool:
        return self.stratum_kind in {
            "equality_root",
            "simultaneous_affine_equality_root",
            "simultaneous_polynomial_equality_root",
            "simultaneous_polynomial_multiple_equality_root",
            "quadratic_double_equality_root",
            "sturm_polynomial_multiple_equality_root",
        }

    @property
    def proof_certified(self) -> bool:
        return bool(self.certified and not self.missing_obligations)


@dataclass(frozen=True)
class PolynomialDecisionFunctionSpec:
    """One polynomial discriminator in a finite one-dimensional arrangement."""

    decision_id: str
    coefficients: tuple[float, ...]
    root_brackets: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class AffineBoxDecisionFunctionSpec:
    """One axis-aligned affine discriminator on a finite interval box.

    Coefficients are stored as ``(a0, a1, ..., ad)`` for
    ``a0 + sum_j a_j x_j``.  The box constructor below intentionally accepts
    only coordinate-affine decisions with exactly one nonzero slope.  That
    keeps the derived strata as interval boxes/slabs instead of silently
    replacing oblique hyperplane cells by hulls.
    """

    decision_id: str
    coefficients: tuple[float, ...]


@dataclass(frozen=True)
class PolynomialDecisionStratificationCertificate:
    """Constructor-derived stratified tree for a polynomial decision function.

    The constructor verifies a finite list of root brackets and sign cells for
    one scalar polynomial on one compact interval.  It derives a branch tree
    with positive-margin leaves and explicit equality leaves.  It does not
    recursively consume equality leaves by itself.
    """

    decision_id: str
    coefficients: tuple[float, ...]
    domain: tuple[float, float]
    root_brackets: tuple[tuple[float, float], ...]
    strata: tuple[PolynomialDecisionStratumCertificate, ...]
    source_tree: BranchEventTreeCertificate
    stratified_tree: StratifiedBranchTreeCertificate
    statement: str
    proof_sketch: str
    theorem_id: str = "polynomial_decision_stratified_branch_tree"

    @property
    def sign_stratum_count(self) -> int:
        return sum(not stratum.equality for stratum in self.strata)

    @property
    def equality_stratum_count(self) -> int:
        return sum(stratum.equality for stratum in self.strata)

    @property
    def certified(self) -> bool:
        return bool(
            self.statement
            and self.proof_sketch
            and self.strata
            and all(stratum.proof_certified for stratum in self.strata)
            and self.stratified_tree.certified
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = [
            f"{stratum.stratum_id}:{item}"
            for stratum in self.strata
            for item in stratum.missing_obligations
        ]
        missing.extend(self.stratified_tree.missing_obligations)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class PolynomialDecisionArrangementStratificationCertificate:
    """Constructor-derived stratified tree for several polynomial decisions."""

    arrangement_id: str
    decision_functions: tuple[PolynomialDecisionFunctionSpec, ...]
    domain: tuple[float, float]
    strata: tuple[PolynomialDecisionStratumCertificate, ...]
    source_tree: BranchEventTreeCertificate
    stratified_tree: StratifiedBranchTreeCertificate
    statement: str
    proof_sketch: str
    theorem_id: str = "polynomial_decision_arrangement_stratified_branch_tree"

    @property
    def sign_stratum_count(self) -> int:
        return sum(not stratum.equality for stratum in self.strata)

    @property
    def equality_stratum_count(self) -> int:
        return sum(stratum.equality for stratum in self.strata)

    @property
    def certified(self) -> bool:
        return bool(
            self.statement
            and self.proof_sketch
            and self.decision_functions
            and self.strata
            and all(stratum.proof_certified for stratum in self.strata)
            and self.stratified_tree.certified
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = [
            f"{stratum.stratum_id}:{item}"
            for stratum in self.strata
            for item in stratum.missing_obligations
        ]
        missing.extend(self.stratified_tree.missing_obligations)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class AffineBoxDecisionStratumCertificate:
    """One sign cell or equality slab in an axis-aligned affine box arrangement."""

    stratum_id: str
    stratum_kind: str
    box: tuple[tuple[float, float], ...]
    sign_vector: tuple[str, ...]
    defining_function_ids: tuple[str, ...]
    value_bounds: tuple[tuple[float, float], ...]
    certified: bool
    missing_obligations: tuple[str, ...] = ()

    @property
    def equality(self) -> bool:
        return self.stratum_kind == "axis_aligned_affine_equality_slab"

    @property
    def proof_certified(self) -> bool:
        return bool(self.certified and not self.missing_obligations)


@dataclass(frozen=True)
class AffineBoxDecisionArrangementStratificationCertificate:
    """Constructor-derived stratified tree for axis-aligned affine box decisions."""

    arrangement_id: str
    decision_functions: tuple[AffineBoxDecisionFunctionSpec, ...]
    domain_box: tuple[tuple[float, float], ...]
    strata: tuple[AffineBoxDecisionStratumCertificate, ...]
    source_tree: BranchEventTreeCertificate
    stratified_tree: StratifiedBranchTreeCertificate
    statement: str
    proof_sketch: str
    theorem_id: str = "axis_aligned_affine_box_decision_arrangement"

    @property
    def dimension(self) -> int:
        return len(self.domain_box)

    @property
    def sign_stratum_count(self) -> int:
        return sum(not stratum.equality for stratum in self.strata)

    @property
    def equality_stratum_count(self) -> int:
        return sum(stratum.equality for stratum in self.strata)

    @property
    def certified(self) -> bool:
        return bool(
            self.statement
            and self.proof_sketch
            and self.arrangement_id
            and self.decision_functions
            and self.domain_box
            and self.strata
            and all(stratum.proof_certified for stratum in self.strata)
            and self.stratified_tree.certified
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = [
            f"{stratum.stratum_id}:{item}"
            for stratum in self.strata
            for item in stratum.missing_obligations
        ]
        missing.extend(self.stratified_tree.missing_obligations)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class AffineHalfspaceDecisionCellCertificate:
    """One halfspace cell or hyperplane slab for a single affine decision."""

    cell_id: str
    cell_kind: str
    domain_box: tuple[tuple[float, float], ...]
    coefficients: tuple[float, ...]
    sign_label: str
    slab_half_width: float
    constraints: tuple[str, ...]
    value_range_on_box: tuple[float, float]
    certified: bool
    missing_obligations: tuple[str, ...] = ()

    @property
    def equality(self) -> bool:
        return self.cell_kind == "affine_hyperplane_equality_slab"

    @property
    def proof_certified(self) -> bool:
        return bool(self.certified and not self.missing_obligations)


@dataclass(frozen=True)
class AffineHalfspaceDecisionStratificationCertificate:
    """Constructor-derived halfspace-cell stratification for one affine boundary."""

    decision_id: str
    coefficients: tuple[float, ...]
    domain_box: tuple[tuple[float, float], ...]
    slab_half_width: float
    cells: tuple[AffineHalfspaceDecisionCellCertificate, ...]
    source_tree: BranchEventTreeCertificate
    stratified_tree: StratifiedBranchTreeCertificate
    statement: str
    proof_sketch: str
    theorem_id: str = "affine_halfspace_decision_stratified_branch_tree"

    @property
    def dimension(self) -> int:
        return len(self.domain_box)

    @property
    def sign_stratum_count(self) -> int:
        return sum(not cell.equality for cell in self.cells)

    @property
    def equality_stratum_count(self) -> int:
        return sum(cell.equality for cell in self.cells)

    @property
    def certified(self) -> bool:
        return bool(
            self.statement
            and self.proof_sketch
            and self.decision_id
            and self.domain_box
            and self.slab_half_width > 0.0
            and self.cells
            and all(cell.proof_certified for cell in self.cells)
            and self.stratified_tree.certified
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = [
            f"{cell.cell_id}:{item}"
            for cell in self.cells
            for item in cell.missing_obligations
        ]
        missing.extend(self.stratified_tree.missing_obligations)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class AffineHalfspaceArrangementCellCertificate:
    """One convex polygon cell in a two-dimensional affine halfspace arrangement."""

    cell_id: str
    cell_kind: str
    domain_box: tuple[tuple[float, float], ...]
    sign_vector: tuple[str, ...]
    defining_function_ids: tuple[str, ...]
    constraints: tuple[str, ...]
    value_bounds: tuple[tuple[float, float], ...]
    vertices: tuple[tuple[float, float], ...]
    area_lower_bound: float
    slab_half_width: float
    certified: bool
    missing_obligations: tuple[str, ...] = ()

    @property
    def equality(self) -> bool:
        return bool(self.defining_function_ids)

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified
            and self.area_lower_bound > 0.0
            and self.vertices
            and not self.missing_obligations
        )


@dataclass(frozen=True)
class AffineHalfspaceArrangementStratificationCertificate:
    """Constructor-derived convex-cell arrangement for 2D affine decisions."""

    arrangement_id: str
    decision_functions: tuple[AffineBoxDecisionFunctionSpec, ...]
    domain_box: tuple[tuple[float, float], ...]
    slab_half_width: float
    cells: tuple[AffineHalfspaceArrangementCellCertificate, ...]
    domain_area: float
    cell_area_sum: float
    cover_area_gap_upper_bound: float
    area_cover_certified: bool
    source_tree: BranchEventTreeCertificate
    stratified_tree: StratifiedBranchTreeCertificate
    statement: str
    proof_sketch: str
    theorem_id: str = "affine_halfspace_arrangement_stratified_branch_tree"

    @property
    def dimension(self) -> int:
        return len(self.domain_box)

    @property
    def sign_stratum_count(self) -> int:
        return sum(not cell.equality for cell in self.cells)

    @property
    def equality_stratum_count(self) -> int:
        return sum(cell.equality for cell in self.cells)

    @property
    def certified(self) -> bool:
        return bool(
            self.statement
            and self.proof_sketch
            and self.arrangement_id
            and self.decision_functions
            and len(self.domain_box) == 2
            and self.slab_half_width > 0.0
            and self.cells
            and self.domain_area > 0.0
            and self.area_cover_certified
            and self.cover_area_gap_upper_bound <= 1.0e-8 * max(1.0, self.domain_area)
            and all(cell.proof_certified for cell in self.cells)
            and self.stratified_tree.certified
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = [
            f"{cell.cell_id}:{item}"
            for cell in self.cells
            for item in cell.missing_obligations
        ]
        missing.extend(self.stratified_tree.missing_obligations)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class AffineHalfspaceArrangement3DCellCertificate:
    """One convex polyhedral cell in a three-dimensional affine arrangement."""

    cell_id: str
    cell_kind: str
    domain_box: tuple[tuple[float, float], ...]
    sign_vector: tuple[str, ...]
    defining_function_ids: tuple[str, ...]
    constraints: tuple[str, ...]
    value_bounds: tuple[tuple[float, float], ...]
    vertices: tuple[tuple[float, float, float], ...]
    volume_lower_bound: float
    slab_half_width: float
    certified: bool
    missing_obligations: tuple[str, ...] = ()

    @property
    def equality(self) -> bool:
        return bool(self.defining_function_ids)

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.certified
            and self.volume_lower_bound > 0.0
            and len(self.vertices) >= 4
            and not self.missing_obligations
        )


@dataclass(frozen=True)
class AffineHalfspaceArrangement3DStratificationCertificate:
    """Constructor-derived convex-cell arrangement for 3D affine decisions."""

    arrangement_id: str
    decision_functions: tuple[AffineBoxDecisionFunctionSpec, ...]
    domain_box: tuple[tuple[float, float], ...]
    slab_half_width: float
    cells: tuple[AffineHalfspaceArrangement3DCellCertificate, ...]
    domain_volume: float
    cell_volume_sum: float
    cover_volume_gap_upper_bound: float
    volume_cover_certified: bool
    source_tree: BranchEventTreeCertificate
    stratified_tree: StratifiedBranchTreeCertificate
    statement: str
    proof_sketch: str
    theorem_id: str = "affine_halfspace_3d_arrangement_stratified_branch_tree"

    @property
    def dimension(self) -> int:
        return len(self.domain_box)

    @property
    def sign_stratum_count(self) -> int:
        return sum(not cell.equality for cell in self.cells)

    @property
    def equality_stratum_count(self) -> int:
        return sum(cell.equality for cell in self.cells)

    @property
    def certified(self) -> bool:
        return bool(
            self.statement
            and self.proof_sketch
            and self.arrangement_id
            and self.decision_functions
            and len(self.domain_box) == 3
            and self.slab_half_width > 0.0
            and self.cells
            and self.domain_volume > 0.0
            and self.volume_cover_certified
            and self.cover_volume_gap_upper_bound
            <= 1.0e-8 * max(1.0, self.domain_volume)
            and all(cell.proof_certified for cell in self.cells)
            and self.stratified_tree.certified
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = [
            f"{cell.cell_id}:{item}"
            for cell in self.cells
            for item in cell.missing_obligations
        ]
        missing.extend(self.stratified_tree.missing_obligations)
        return tuple(dict.fromkeys(missing))


def certify_stratified_branch_event_tree(
    source_tree: object,
    *,
    leaf_certificates: tuple[StratifiedBranchLeafCertificate, ...] = (),
    recursive_exhaustion_certified: bool = False,
) -> StratifiedBranchTreeCertificate:
    """Attach analytic stratum labels to every leaf of a supplied tree."""

    if recursive_exhaustion_certified:
        raise TypeError(
            "recursive_exhaustion_certified is not an accepted manual witness; "
            "use certify_recursive_stratified_branch_event_consumption"
        )

    normalized = (
        source_tree
        if isinstance(source_tree, BranchEventTreeCertificate)
        else certify_supplied_branch_event_tree(source_tree)
    )
    source_leaves = normalized.leaf_certificates
    if leaf_certificates:
        leaves = tuple(leaf_certificates)
    else:
        leaves = tuple(_infer_stratified_leaf(leaf) for leaf in source_leaves)
    source_leaf_ids = {leaf.leaf_id for leaf in source_leaves}
    leaf_source_ids = {leaf.source_leaf_id for leaf in leaves}
    preserves_branch_tree = bool(
        source_leaf_ids
        and source_leaf_ids == leaf_source_ids
        and len(source_leaf_ids) == len(leaves)
    )
    leaf_taxonomy_certified = bool(
        leaves
        and all(leaf.well_formed and leaf.supported_kind for leaf in leaves)
    )
    return StratifiedBranchTreeCertificate(
        source_tree=normalized,
        leaf_certificates=leaves,
        preserves_branch_tree=preserves_branch_tree,
        leaf_taxonomy_certified=leaf_taxonomy_certified,
        cover_certified=bool(normalized.cover_certified),
        recursive_exhaustion_certified=False,
    )


def certify_terminal_policy_stratified_branch_event_tree(
    source_tree: object,
    *,
    selector_policies: tuple[SelectorPolicyLeafCertificate, ...] = (),
    total_collision_clusters: tuple[TotalCollisionClusterLeafCertificate, ...] = (),
) -> StratifiedBranchTreeCertificate:
    """Attach proof-certified selector/total-collision terminal leaves.

    This is a constructor for two zero-margin leaf classes in the recursive
    branch/event theorem.  It consumes only supplied local policy certificates:
    selector leaves need explicit selector-policy isolation evidence, and total
    collision cluster leaves need an entry/stop certificate.  It does not
    derive where those leaves occur from arbitrary interval data.
    """

    normalized = (
        source_tree
        if isinstance(source_tree, BranchEventTreeCertificate)
        else certify_supplied_branch_event_tree(source_tree)
    )
    selector_by_leaf = {certificate.leaf_id: certificate for certificate in selector_policies}
    cluster_by_leaf = {
        certificate.leaf_id: certificate for certificate in total_collision_clusters
    }
    leaf_certificates: list[StratifiedBranchLeafCertificate] = []
    for source_leaf in normalized.leaf_certificates:
        selector = selector_by_leaf.get(source_leaf.leaf_id)
        cluster = cluster_by_leaf.get(source_leaf.leaf_id)
        if selector is not None and cluster is not None:
            raise ValueError("a source leaf cannot be both selector and total cluster")
        if selector is not None:
            equality = selector.equality_stratum
            leaf_certificates.append(
                StratifiedBranchLeafCertificate(
                    leaf_id=f"stratified:{source_leaf.leaf_id}",
                    source_leaf_id=source_leaf.leaf_id,
                    leaf_kind="selector_policy",
                    terminal_response_kind=selector.selector_policy_id,
                    terminal_response_certified=selector.proof_certified,
                    source_leaf_certified=source_leaf.certified,
                    depth=source_leaf.depth,
                    equality_stratum=equality,
                    missing_obligations=selector.missing_obligations,
                )
            )
            continue
        if cluster is not None:
            leaf_certificates.append(
                StratifiedBranchLeafCertificate(
                    leaf_id=f"stratified:{source_leaf.leaf_id}",
                    source_leaf_id=source_leaf.leaf_id,
                    leaf_kind="total_collision_cluster",
                    terminal_response_kind=cluster.stop_or_selector_policy,
                    terminal_response_certified=cluster.proof_certified,
                    source_leaf_certified=source_leaf.certified,
                    depth=source_leaf.depth,
                    total_collision_cluster=cluster,
                    missing_obligations=cluster.missing_obligations,
                )
            )
            continue
        leaf_certificates.append(_infer_stratified_leaf(source_leaf))

    return certify_stratified_branch_event_tree(
        normalized,
        leaf_certificates=tuple(leaf_certificates),
    )


def certify_recursive_stratified_branch_event_consumption(
    source_tree: object,
    *,
    root_dimension: int,
    root_rank: int | None = None,
    child_consumptions: Mapping[
        str,
        RecursiveStratifiedBranchEventConsumptionCertificate,
    ]
    | tuple[
        tuple[str, RecursiveStratifiedBranchEventConsumptionCertificate],
        ...,
    ] = (),
    recursion_kind: str = "branch_event_order",
) -> RecursiveStratifiedBranchEventConsumptionCertificate:
    """Consume a supplied stratified tree by terminal leaves or descent.

    The constructor proves only the displayed finite recursive tree.  It does
    not derive the tree from arbitrary initial data, and it refuses unsupported
    analytic strata instead of silently treating them as leaves.
    """

    tree = (
        source_tree
        if isinstance(source_tree, StratifiedBranchTreeCertificate)
        else certify_stratified_branch_event_tree(source_tree)
    )
    dimension = int(root_dimension)
    rank = None if root_rank is None else int(root_rank)
    child_by_leaf = _normalize_child_consumptions(child_consumptions)
    leaf_consumptions = tuple(
        _consume_recursive_stratified_leaf(
            leaf,
            root_dimension=dimension,
            root_rank=rank,
            child=child_by_leaf.get(leaf.leaf_id),
        )
        for leaf in tree.leaf_certificates
    )
    unused_children = tuple(
        leaf_id
        for leaf_id in child_by_leaf
        if leaf_id not in {leaf.leaf_id for leaf in tree.leaf_certificates}
    )
    child_certificates = tuple(child for child in child_by_leaf.values())
    node_count = 1 + sum(child.node_count for child in child_certificates)
    recursion_depth = 1 + max(
        (child.recursion_depth for child in child_certificates),
        default=0,
    )
    unsupported_count = sum(leaf.unsupported for leaf in leaf_consumptions)
    obligations = (
        RecursiveStratifiedConsumptionObligation(
            obligation="stratified_branch_event_tree_supplied",
            certified=tree.certified,
            detail=(
                f"leaf_count={tree.leaf_count}; "
                f"leaf_kinds={tree.leaf_kinds}"
            ),
        ),
        RecursiveStratifiedConsumptionObligation(
            obligation="nonnegative_root_dimension",
            certified=dimension >= 0 and (rank is None or rank >= 0),
            detail=f"root_dimension={dimension}; root_rank={rank}",
        ),
        RecursiveStratifiedConsumptionObligation(
            obligation="every_leaf_terminal_or_descending_child",
            certified=bool(
                leaf_consumptions
                and all(leaf.certified for leaf in leaf_consumptions)
            ),
            detail=(
                f"terminal_leaf_count="
                f"{sum(leaf.terminal_certified for leaf in leaf_consumptions)}; "
                f"recursive_leaf_count="
                f"{sum(leaf.recursive for leaf in leaf_consumptions)}"
            ),
        ),
        RecursiveStratifiedConsumptionObligation(
            obligation="unsupported_analytic_strata_absent",
            certified=unsupported_count == 0,
            detail=f"unsupported_leaf_count={unsupported_count}",
        ),
        RecursiveStratifiedConsumptionObligation(
            obligation="child_consumptions_match_source_leaves",
            certified=not unused_children,
            detail="unused_child_leaf_ids=" + ",".join(unused_children),
        ),
        RecursiveStratifiedConsumptionObligation(
            obligation="finite_supplied_recursive_tree_only",
            certified=True,
            required=False,
            detail=(
                "consumes the finite recursive tree supplied to this "
                "constructor; arbitrary initial-data partition remains a "
                "separate theorem"
            ),
        ),
    )
    return RecursiveStratifiedBranchEventConsumptionCertificate(
        source_tree=tree,
        recursion_kind=str(recursion_kind),
        root_dimension=dimension,
        root_rank=rank,
        leaf_consumptions=leaf_consumptions,
        node_count=node_count,
        recursion_depth=recursion_depth,
        statement=(
            "A supplied stratified branch/event-order tree is consumed when "
            "each leaf is either a proof-certified terminal atlas/stop/selector "
            "response or carries a certified child tree on a strictly lower "
            "dimension or lower rank equality stratum."
        ),
        proof_sketch=(
            "Proceed by induction over the finite supplied recursive tree.  "
            "Terminal leaves are consumed by their local proof certificates.  "
            "For nonterminal equality leaves, the child certificate covers the "
            "named stratum and the dimension/rank descent makes recursive "
            "calls well-founded.  Since the supplied tree is finite and every "
            "edge descends, the induction reaches only terminal leaves.  A "
            "named unsupported analytic stratum is returned as an obligation, "
            "not certified."
        ),
        obligations=obligations,
    )


def certify_affine_box_decision_arrangement_stratified_branch_event_tree(
    *,
    arrangement_id: str,
    decision_functions: tuple[AffineBoxDecisionFunctionSpec, ...],
    domain_box: tuple[tuple[float, float], ...],
    equality_resolution_policy: str = "lower_dimensional_recursive_stratum",
) -> AffineBoxDecisionArrangementStratificationCertificate:
    """Derive a finite stratified tree from coordinate-affine box decisions.

    This is a higher-dimensional interval-box constructor, but deliberately a
    narrow one.  Each discriminator must be affine in exactly one box
    coordinate, so the equality sets are axis-aligned slabs and the positive
    cells remain interval boxes.  Oblique affine hyperplanes require a future
    polytope or Lohner-style set representation and are rejected instead of
    being hulled into an interval box.
    """

    arrangement_id = str(arrangement_id)
    box = tuple((float(left), float(right)) for left, right in domain_box)
    dimension = len(box)
    specs = tuple(
        AffineBoxDecisionFunctionSpec(
            decision_id=str(spec.decision_id),
            coefficients=tuple(float(value) for value in spec.coefficients),
        )
        for spec in decision_functions
    )
    if not arrangement_id:
        raise ValueError("affine box arrangement requires an id")
    if dimension <= 0:
        raise ValueError("affine box arrangement requires a nonempty domain box")
    if any(
        not (np.isfinite(left) and np.isfinite(right) and left < right)
        for left, right in box
    ):
        raise ValueError("affine box arrangement requires finite nonempty intervals")
    if not specs:
        raise ValueError("affine box arrangement requires at least one decision")

    root_items_by_axis: dict[int, list[tuple[float, str]]] = {
        axis: [] for axis in range(dimension)
    }
    for spec in specs:
        if not spec.decision_id:
            raise ValueError("affine box decisions require ids")
        if len(spec.coefficients) != dimension + 1:
            raise ValueError(
                "affine box decision coefficients must have length dimension + 1"
            )
        if any(not np.isfinite(value) for value in spec.coefficients):
            raise ValueError("affine box decision coefficients must be finite")
        constant = spec.coefficients[0]
        slopes = spec.coefficients[1:]
        nonzero_axes = tuple(
            axis
            for axis, slope in enumerate(slopes)
            if abs(slope)
            > 64.0 * np.finfo(float).eps * max(1.0, abs(constant), abs(slope))
        )
        if len(nonzero_axes) != 1:
            raise ValueError(
                "affine box arrangement accepts only axis-aligned decisions"
            )
        axis = nonzero_axes[0]
        root = -constant / slopes[axis]
        left, right = box[axis]
        if left <= root <= right:
            root_items_by_axis[axis].append((float(root), spec.decision_id))

    axis_pieces = tuple(
        _axis_aligned_affine_box_axis_pieces(
            axis=axis,
            interval=box[axis],
            root_items=root_items_by_axis[axis],
        )
        for axis in range(dimension)
    )
    strata: list[AffineBoxDecisionStratumCertificate] = []
    source_leaves: list[BranchEventTreeLeafCertificate] = []
    stratified_leaves: list[StratifiedBranchLeafCertificate] = []

    for index, pieces in enumerate(product(*axis_pieces)):
        cell_box = tuple(piece[1] for piece in pieces)
        value_bounds = tuple(
            _affine_box_value_interval(spec.coefficients, cell_box)
            for spec in specs
        )
        signs = tuple(interval_sign(FloatInterval(*bounds)) for bounds in value_bounds)
        defining_function_ids = tuple(
            spec.decision_id
            for spec, sign in zip(specs, signs)
            if sign == 0
        )
        equality = bool(defining_function_ids)
        missing: list[str] = []
        if equality:
            declared_defining_ids = tuple(
                decision_id
                for piece in pieces
                for decision_id in piece[2]
            )
            if not set(defining_function_ids).issubset(set(declared_defining_ids)):
                missing.append(
                    "affine_box_equality_not_supported_by_axis_slab"
                )
        else:
            margins = tuple(
                _strict_affine_box_sign_margin(bounds, sign)
                for bounds, sign in zip(value_bounds, signs)
            )
            if any(margin <= 0.0 for margin in margins):
                missing.append("affine_box_sign_cell_margin_not_positive")
        certified = bool(not missing and all(sign != 0 for sign in signs) != equality)
        sign_vector = tuple(
            f"{spec.decision_id}:{'0' if sign == 0 else ('+' if sign > 0 else '-')}"
            for spec, sign in zip(specs, signs)
        )
        kind = (
            "axis_aligned_affine_equality_slab"
            if equality
            else "axis_aligned_affine_sign_box"
        )
        label = "+".join(defining_function_ids) if equality else f"sign:{index}"
        stratum_id = f"{arrangement_id}:{'slab' if equality else 'cell'}:{index}:{label}"
        strata.append(
            AffineBoxDecisionStratumCertificate(
                stratum_id=stratum_id,
                stratum_kind=kind,
                box=cell_box,
                sign_vector=sign_vector,
                defining_function_ids=defining_function_ids,
                value_bounds=value_bounds,
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        source_id = f"{stratum_id}:leaf"
        source_leaves.append(
            BranchEventTreeLeafCertificate(
                leaf_id=source_id,
                leaf_type=(
                    "affine_box_equality_slab_leaf"
                    if equality
                    else "affine_box_positive_margin_leaf"
                ),
                decision=(
                    "simultaneous_event_equality"
                    if equality
                    else ",".join(sign_vector)
                ),
                source_type="AxisAlignedAffineBoxArrangement",
                depth=0,
                certified=bool(certified and not equality),
                missing_obligations=(),
            )
        )
        if equality:
            equality_certificate = EqualityStratumCertificate(
                stratum_id=stratum_id,
                defining_function_ids=defining_function_ids,
                leaf_kind="simultaneous_event_equality",
                isolation_certified=certified,
                resolution_policy=str(equality_resolution_policy),
                certified=certified,
                missing_obligations=tuple(missing),
            )
            tie = EventOrderTieLeafCertificate(
                leaf_id=source_id,
                tied_event_ids=tuple(
                    event_id
                    for decision_id in defining_function_ids
                    for event_id in (
                        f"{decision_id}:negative_side",
                        f"{decision_id}:positive_side",
                    )
                ),
                equality_stratum=equality_certificate,
                certified=certified,
                missing_obligations=tuple(missing),
            )
            stratified_leaves.append(
                StratifiedBranchLeafCertificate(
                    leaf_id=f"stratified:{source_id}",
                    source_leaf_id=source_id,
                    leaf_kind="simultaneous_event_equality",
                    terminal_response_kind="recursive_affine_box_equality_slab",
                    terminal_response_certified=False,
                    source_leaf_certified=True,
                    equality_stratum=equality_certificate,
                    event_order_tie=tie,
                    missing_obligations=tuple(missing),
                )
            )
        else:
            stratified_leaves.append(
                StratifiedBranchLeafCertificate(
                    leaf_id=f"stratified:{source_id}",
                    source_leaf_id=source_id,
                    leaf_kind="positive_margin_unique_event",
                    terminal_response_kind="affine_box_sign_decision",
                    terminal_response_certified=certified,
                    source_leaf_certified=certified,
                    decision_functions=tuple(
                        AnalyticDecisionFunctionCertificate(
                            function_id=f"{stratum_id}:{spec.decision_id}",
                            function_kind="axis_aligned_affine_box_decision_sign",
                            margin_lower_bound=_strict_affine_box_sign_margin(
                                bounds,
                                sign,
                            ),
                            lipschitz_bound=_affine_box_lipschitz_bound(
                                spec.coefficients,
                            ),
                            certified=certified and sign != 0,
                            missing_obligations=tuple(missing),
                        )
                        for spec, bounds, sign in zip(specs, value_bounds, signs)
                    ),
                    missing_obligations=tuple(missing),
                )
            )

    cover_certified = bool(
        strata
        and _axis_aligned_pieces_cover_box(axis_pieces, box)
        and all(stratum.proof_certified for stratum in strata)
    )
    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="AxisAlignedAffineBoxArrangement",
        leaf_certificates=tuple(source_leaves),
        cover_certified=cover_certified,
        leaf_decisions_certified=bool(
            source_leaves
            and all(
                leaf.certified
                for leaf in source_leaves
                if "equality" not in leaf.leaf_type
            )
        ),
        equality_strata_explicit=any(stratum.equality for stratum in strata),
        obligations=(
            BranchEventTreeObligation(
                obligation="axis_aligned_affine_box_domain_valid",
                certified=True,
                detail=f"dimension={dimension}; box={box!r}",
            ),
            BranchEventTreeObligation(
                obligation="axis_aligned_affine_box_strata_cover_domain",
                certified=cover_certified,
                detail=f"stratum_count={len(strata)}",
            ),
            BranchEventTreeObligation(
                obligation="axis_aligned_affine_equality_strata_explicit",
                certified=True,
                detail=(
                    "equality_stratum_count="
                    f"{sum(stratum.equality for stratum in strata)}"
                ),
                required=False,
            ),
        ),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=tuple(stratified_leaves),
    )
    return AffineBoxDecisionArrangementStratificationCertificate(
        arrangement_id=arrangement_id,
        decision_functions=specs,
        domain_box=box,
        strata=tuple(strata),
        source_tree=source_tree,
        stratified_tree=stratified,
        statement=(
            "A finite interval box with axis-aligned affine branch/event "
            "discriminants admits a constructor-derived stratified partition: "
            "coordinate roots are computed from coefficients, separated root "
            "slabs are inserted in the relevant coordinates, the Cartesian "
            "product of coordinate pieces covers the box, strict-sign boxes "
            "are positive-margin leaves, and root slabs are explicit equality "
            "strata for recursive descent."
        ),
        proof_sketch=(
            "Each discriminator has the form a0+a_j*x_j with exactly one "
            "nonzero coordinate slope.  Roots inside the corresponding "
            "coordinate interval are grouped by coordinate and value, then "
            "given disjoint one-quarter-gap slabs, one-sided at box "
            "boundaries.  Complementary coordinate intervals have a fixed "
            "sign for every discriminator depending on that coordinate.  "
            "Taking the Cartesian product of all coordinate pieces covers the "
            "input box.  On every product cell, interval affine evaluation "
            "either proves a strict sign vector or identifies exactly the "
            "axis-aligned equality slabs whose decisions contain zero."
        ),
    )


def certify_affine_box_decision_arrangement_recursive_consumption(
    arrangement: AffineBoxDecisionArrangementStratificationCertificate,
    *,
    root_dimension: int,
    root_rank: int | None = None,
    child_consumptions: Mapping[
        str,
        RecursiveStratifiedBranchEventConsumptionCertificate,
    ]
    | tuple[
        tuple[str, RecursiveStratifiedBranchEventConsumptionCertificate],
        ...,
    ] = (),
    recursion_kind: str = "axis_aligned_affine_box_decision_arrangement",
) -> RecursiveStratifiedBranchEventConsumptionCertificate:
    """Consume an axis-aligned affine box arrangement by recursive strata."""

    if not isinstance(
        arrangement,
        AffineBoxDecisionArrangementStratificationCertificate,
    ):
        raise TypeError(
            "arrangement must be an "
            "AffineBoxDecisionArrangementStratificationCertificate"
        )
    child_by_leaf = _normalize_arrangement_child_consumptions(
        arrangement,  # type: ignore[arg-type]
        child_consumptions,
    )
    if (
        not child_by_leaf
        and arrangement.proof_certified
        and arrangement.equality_stratum_count
    ):
        child_by_leaf = _normalize_arrangement_child_consumptions(
            arrangement,  # type: ignore[arg-type]
            derive_affine_box_decision_arrangement_child_consumptions(
                arrangement,
                child_root_rank=0,
            ),
        )
    return certify_recursive_stratified_branch_event_consumption(
        arrangement.stratified_tree,
        root_dimension=root_dimension,
        root_rank=root_rank,
        child_consumptions=child_by_leaf,
        recursion_kind=recursion_kind,
    )


def derive_affine_box_decision_arrangement_child_consumptions(
    arrangement: AffineBoxDecisionArrangementStratificationCertificate,
    *,
    child_root_rank: int | None = 0,
    tolerance: float = 1.0e-9,
) -> dict[str, RecursiveStratifiedBranchEventConsumptionCertificate]:
    """Derive terminal coordinate-subspace children for axis-aligned slabs."""

    if not isinstance(
        arrangement,
        AffineBoxDecisionArrangementStratificationCertificate,
    ):
        raise TypeError(
            "arrangement must be an "
            "AffineBoxDecisionArrangementStratificationCertificate"
        )
    if not arrangement.proof_certified:
        raise ValueError("axis-aligned affine box arrangement must be proof-certified")
    epsilon = float(tolerance)
    if not (np.isfinite(epsilon) and epsilon > 0.0):
        raise ValueError("tolerance must be positive")

    dimension = arrangement.dimension
    specs_by_id = {spec.decision_id: spec for spec in arrangement.decision_functions}
    child_consumptions: dict[str, RecursiveStratifiedBranchEventConsumptionCertificate] = {}
    for stratum in arrangement.strata:
        if not stratum.equality or not stratum.defining_function_ids:
            continue
        fixed_by_axis: dict[int, tuple[float, list[str]]] = {}
        supported = True
        for decision_id in stratum.defining_function_ids:
            spec = specs_by_id.get(decision_id)
            if spec is None:
                supported = False
                break
            axis_root = _axis_aligned_affine_box_spec_axis_root(
                spec,
                dimension=dimension,
                tolerance=epsilon,
            )
            if axis_root is None:
                supported = False
                break
            axis, root = axis_root
            lower, upper = stratum.box[axis]
            if root < lower - epsilon or root > upper + epsilon:
                supported = False
                break
            if axis in fixed_by_axis:
                existing_root, decision_ids = fixed_by_axis[axis]
                if abs(existing_root - root) > epsilon * max(
                    1.0,
                    abs(existing_root),
                    abs(root),
                ):
                    supported = False
                    break
                decision_ids.append(decision_id)
            else:
                fixed_by_axis[axis] = (root, [decision_id])
        if not supported:
            continue
        fixed_coordinates = tuple(
            (axis, root, tuple(decision_ids))
            for axis, (root, decision_ids) in sorted(fixed_by_axis.items())
        )
        free_domain_box = tuple(
            bounds
            for axis, bounds in enumerate(stratum.box)
            if axis not in fixed_by_axis
        )
        child = _axis_aligned_affine_box_child_consumption(
            stratum=stratum,
            fixed_coordinates=fixed_coordinates,
            free_domain_box=free_domain_box,
            root_rank=child_root_rank,
        )
        if child.proof_certified:
            child_consumptions[stratum.stratum_id] = child
    return child_consumptions


def certify_affine_halfspace_decision_stratified_branch_event_tree(
    *,
    decision_id: str,
    coefficients: tuple[float, ...] | list[float] | np.ndarray,
    domain_box: tuple[tuple[float, float], ...],
    slab_half_width: float | None = None,
    equality_resolution_policy: str = "lower_dimensional_recursive_stratum",
) -> AffineHalfspaceDecisionStratificationCertificate:
    """Derive a halfspace-cell stratification for one affine box discriminator.

    Unlike the axis-aligned box constructor, this accepts an oblique affine
    decision.  It lifts the partition into a richer cell language:
    ``f <= -epsilon``, ``|f| <= epsilon``, and ``f >= epsilon`` inside the
    original box.  The separated halfspace cells carry a positive margin;
    the central slab is an explicit equality stratum for recursive descent.
    """

    decision_id = str(decision_id)
    box = tuple((float(left), float(right)) for left, right in domain_box)
    dimension = len(box)
    coeffs = tuple(float(value) for value in np.asarray(coefficients, dtype=float).reshape(-1))
    if not decision_id:
        raise ValueError("affine halfspace decision requires an id")
    if dimension <= 0:
        raise ValueError("affine halfspace decision requires a nonempty domain box")
    if len(coeffs) != dimension + 1:
        raise ValueError(
            "affine halfspace decision coefficients must have length dimension + 1"
        )
    if any(not np.isfinite(value) for value in coeffs):
        raise ValueError("affine halfspace decision coefficients must be finite")
    if any(
        not (np.isfinite(left) and np.isfinite(right) and left < right)
        for left, right in box
    ):
        raise ValueError("affine halfspace decision requires finite nonempty intervals")
    if not any(abs(slope) > 0.0 for slope in coeffs[1:]):
        raise ValueError("affine halfspace decision requires a nonzero slope")

    value_range = _affine_box_value_interval(coeffs, box)
    lower, upper = value_range
    if slab_half_width is None:
        if lower < 0.0 < upper:
            width = 0.25 * min(-lower, upper)
        else:
            width = 0.25 * max(abs(lower), abs(upper))
    else:
        width = float(slab_half_width)
    if not (np.isfinite(width) and width > 0.0):
        raise ValueError("affine halfspace slab_half_width must be positive")

    cells: list[AffineHalfspaceDecisionCellCertificate] = []
    source_leaves: list[BranchEventTreeLeafCertificate] = []
    stratified_leaves: list[StratifiedBranchLeafCertificate] = []

    def add_sign_cell(kind: str, sign_label: str) -> None:
        negative = sign_label == "-"
        exists = lower < -width if negative else upper > width
        if not exists:
            return
        constraint = (
            f"{decision_id} <= {-width:g}"
            if negative
            else f"{decision_id} >= {width:g}"
        )
        cell_id = f"{decision_id}:{'negative' if negative else 'positive'}_halfspace"
        cells.append(
            AffineHalfspaceDecisionCellCertificate(
                cell_id=cell_id,
                cell_kind=kind,
                domain_box=box,
                coefficients=coeffs,
                sign_label=sign_label,
                slab_half_width=width,
                constraints=(constraint,),
                value_range_on_box=value_range,
                certified=True,
                missing_obligations=(),
            )
        )
        source_id = f"{cell_id}:leaf"
        source_leaves.append(
            BranchEventTreeLeafCertificate(
                leaf_id=source_id,
                leaf_type="affine_halfspace_positive_margin_leaf",
                decision=f"{decision_id}:{sign_label}",
                source_type="AffineHalfspaceDecision",
                depth=0,
                certified=True,
                missing_obligations=(),
            )
        )
        stratified_leaves.append(
            StratifiedBranchLeafCertificate(
                leaf_id=f"stratified:{source_id}",
                source_leaf_id=source_id,
                leaf_kind="positive_margin_unique_event",
                terminal_response_kind="affine_halfspace_sign_decision",
                terminal_response_certified=True,
                source_leaf_certified=True,
                decision_functions=(
                    AnalyticDecisionFunctionCertificate(
                        function_id=cell_id,
                        function_kind="affine_halfspace_decision_sign",
                        margin_lower_bound=width,
                        lipschitz_bound=_affine_box_lipschitz_bound(coeffs),
                        certified=True,
                    ),
                ),
            )
        )

    add_sign_cell("affine_negative_halfspace_cell", "-")

    slab_exists = bool(lower <= width and upper >= -width)
    slab_missing: list[str] = []
    if not slab_exists:
        slab_missing.append("affine_hyperplane_slab_does_not_intersect_box")
    slab_id = f"{decision_id}:hyperplane_slab"
    cells.append(
        AffineHalfspaceDecisionCellCertificate(
            cell_id=slab_id,
            cell_kind="affine_hyperplane_equality_slab",
            domain_box=box,
            coefficients=coeffs,
            sign_label="0",
            slab_half_width=width,
            constraints=(f"{decision_id} >= {-width:g}", f"{decision_id} <= {width:g}"),
            value_range_on_box=value_range,
            certified=slab_exists,
            missing_obligations=tuple(slab_missing),
        )
    )
    slab_source_id = f"{slab_id}:leaf"
    source_leaves.append(
        BranchEventTreeLeafCertificate(
            leaf_id=slab_source_id,
            leaf_type="affine_hyperplane_equality_slab_leaf",
            decision="simultaneous_event_equality",
            source_type="AffineHalfspaceDecision",
            depth=0,
            certified=False,
            missing_obligations=(),
        )
    )
    equality = EqualityStratumCertificate(
        stratum_id=slab_id,
        defining_function_ids=(decision_id,),
        leaf_kind="simultaneous_event_equality",
        isolation_certified=slab_exists,
        resolution_policy=str(equality_resolution_policy),
        certified=slab_exists,
        missing_obligations=tuple(slab_missing),
    )
    tie = EventOrderTieLeafCertificate(
        leaf_id=slab_source_id,
        tied_event_ids=(f"{decision_id}:negative_side", f"{decision_id}:positive_side"),
        equality_stratum=equality,
        certified=slab_exists,
        missing_obligations=tuple(slab_missing),
    )
    stratified_leaves.append(
        StratifiedBranchLeafCertificate(
            leaf_id=f"stratified:{slab_source_id}",
            source_leaf_id=slab_source_id,
            leaf_kind="simultaneous_event_equality",
            terminal_response_kind="recursive_affine_hyperplane_slab",
            terminal_response_certified=False,
            source_leaf_certified=True,
            equality_stratum=equality,
            event_order_tie=tie,
            missing_obligations=tuple(slab_missing),
        )
    )

    add_sign_cell("affine_positive_halfspace_cell", "+")

    cover_certified = bool(cells and slab_exists and all(cell.proof_certified for cell in cells))
    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="AffineHalfspaceDecision",
        leaf_certificates=tuple(source_leaves),
        cover_certified=cover_certified,
        leaf_decisions_certified=bool(
            source_leaves
            and all(
                leaf.certified
                for leaf in source_leaves
                if "equality" not in leaf.leaf_type
            )
        ),
        equality_strata_explicit=True,
        obligations=(
            BranchEventTreeObligation(
                obligation="affine_halfspace_domain_valid",
                certified=True,
                detail=f"dimension={dimension}; box={box!r}",
            ),
            BranchEventTreeObligation(
                obligation="affine_halfspace_trichotomy_cover",
                certified=cover_certified,
                detail=(
                    f"value_range={value_range!r}; slab_half_width={width!r}; "
                    f"cell_count={len(cells)}"
                ),
            ),
            BranchEventTreeObligation(
                obligation="affine_halfspace_equality_slab_explicit",
                certified=True,
                detail=f"slab_id={slab_id}",
                required=False,
            ),
        ),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=tuple(stratified_leaves),
    )
    return AffineHalfspaceDecisionStratificationCertificate(
        decision_id=decision_id,
        coefficients=coeffs,
        domain_box=box,
        slab_half_width=width,
        cells=tuple(cells),
        source_tree=source_tree,
        stratified_tree=stratified,
        statement=(
            "A single affine discriminator on an interval box induces a finite "
            "halfspace-cell stratification without interval hulling: separated "
            "cells are represented by affine inequalities with a positive "
            "margin, and a central hyperplane slab is retained as an explicit "
            "recursive equality stratum."
        ),
        proof_sketch=(
            "Let f(x)=a0+a.x on the compact box.  Exact affine interval "
            "evaluation gives the range of f over the box.  Choose epsilon "
            "positive inside that range.  The trichotomy f<=-epsilon, "
            "|f|<=epsilon, and f>=epsilon covers the box.  The two outer "
            "halfspace cells have margin epsilon from the equality slab, so "
            "their event/branch decision is stable.  The middle slab is not "
            "hulled into a box; it is passed to recursive lower-dimensional "
            "or lower-rank consumption."
        ),
    )


def certify_affine_halfspace_decision_recursive_consumption(
    stratification: AffineHalfspaceDecisionStratificationCertificate,
    *,
    root_dimension: int,
    root_rank: int | None = None,
    child_consumptions: Mapping[
        str,
        RecursiveStratifiedBranchEventConsumptionCertificate,
    ]
    | tuple[
        tuple[str, RecursiveStratifiedBranchEventConsumptionCertificate],
        ...,
    ] = (),
    recursion_kind: str = "affine_halfspace_decision",
) -> RecursiveStratifiedBranchEventConsumptionCertificate:
    """Consume a halfspace-cell affine partition by recursive slab children."""

    if not isinstance(
        stratification,
        AffineHalfspaceDecisionStratificationCertificate,
    ):
        raise TypeError(
            "stratification must be an "
            "AffineHalfspaceDecisionStratificationCertificate"
        )
    child_by_leaf = _normalize_arrangement_child_consumptions(
        stratification,  # type: ignore[arg-type]
        child_consumptions,
    )
    if (
        not child_by_leaf
        and stratification.proof_certified
        and stratification.equality_stratum_count
    ):
        child_by_leaf = _normalize_arrangement_child_consumptions(
            stratification,  # type: ignore[arg-type]
            derive_affine_halfspace_decision_child_consumptions(
                stratification,
                child_root_rank=0,
            ),
        )
    return certify_recursive_stratified_branch_event_consumption(
        stratification.stratified_tree,
        root_dimension=root_dimension,
        root_rank=root_rank,
        child_consumptions=child_by_leaf,
        recursion_kind=recursion_kind,
    )


def derive_affine_halfspace_decision_child_consumptions(
    stratification: AffineHalfspaceDecisionStratificationCertificate,
    *,
    child_root_rank: int | None = 0,
    tolerance: float = 1.0e-9,
) -> dict[str, RecursiveStratifiedBranchEventConsumptionCertificate]:
    """Derive terminal exact-hyperplane children for one affine halfspace slab."""

    if not isinstance(
        stratification,
        AffineHalfspaceDecisionStratificationCertificate,
    ):
        raise TypeError(
            "stratification must be an "
            "AffineHalfspaceDecisionStratificationCertificate"
        )
    if not stratification.proof_certified:
        raise ValueError("affine halfspace decision must be proof-certified")
    epsilon = float(tolerance)
    if not (np.isfinite(epsilon) and epsilon > 0.0):
        raise ValueError("tolerance must be positive")

    child_consumptions: dict[str, RecursiveStratifiedBranchEventConsumptionCertificate] = {}
    for cell in stratification.cells:
        if not cell.equality:
            continue
        geometry_detail = _affine_halfspace_decision_exact_child_geometry_detail(
            coefficients=cell.coefficients,
            domain_box=cell.domain_box,
            tolerance=epsilon,
        )
        if geometry_detail is None:
            continue
        child_dimension, detail = geometry_detail
        child = _affine_halfspace_decision_child_consumption(
            cell=cell,
            child_dimension=child_dimension,
            geometry_detail=detail,
            root_rank=child_root_rank,
        )
        if child.proof_certified:
            child_consumptions[cell.cell_id] = child
    return child_consumptions


def certify_affine_halfspace_arrangement_stratified_branch_event_tree(
    *,
    arrangement_id: str,
    decision_functions: tuple[AffineBoxDecisionFunctionSpec, ...],
    domain_box: tuple[tuple[float, float], ...],
    slab_half_width: float,
    equality_resolution_policy: str = "lower_dimensional_recursive_stratum",
) -> AffineHalfspaceArrangementStratificationCertificate:
    """Derive a 2D convex-cell arrangement from affine halfspace decisions.

    Every discriminator contributes a trichotomy
    ``f <= -epsilon``, ``|f| <= epsilon``, ``f >= epsilon``.  The constructor
    clips the input rectangle by the corresponding halfplanes for every sign
    pattern, discards empty cells, terminal-certifies strict sign cells, and
    leaves any cell touching a slab as a recursive equality stratum.
    """

    arrangement_id = str(arrangement_id)
    box = tuple((float(left), float(right)) for left, right in domain_box)
    specs = tuple(
        AffineBoxDecisionFunctionSpec(
            decision_id=str(spec.decision_id),
            coefficients=tuple(float(value) for value in spec.coefficients),
        )
        for spec in decision_functions
    )
    width = float(slab_half_width)
    if not arrangement_id:
        raise ValueError("affine halfspace arrangement requires an id")
    if len(box) != 2:
        raise ValueError("affine halfspace arrangement currently requires a 2D box")
    if any(
        not (np.isfinite(left) and np.isfinite(right) and left < right)
        for left, right in box
    ):
        raise ValueError("affine halfspace arrangement requires finite nonempty intervals")
    if not (np.isfinite(width) and width > 0.0):
        raise ValueError("affine halfspace arrangement slab_half_width must be positive")
    if not specs:
        raise ValueError("affine halfspace arrangement requires at least one decision")
    for spec in specs:
        if not spec.decision_id:
            raise ValueError("affine halfspace arrangement decisions require ids")
        if len(spec.coefficients) != 3:
            raise ValueError("affine halfspace arrangement decisions must be 2D affine")
        if any(not np.isfinite(value) for value in spec.coefficients):
            raise ValueError("affine halfspace arrangement coefficients must be finite")
        if not any(abs(slope) > 0.0 for slope in spec.coefficients[1:]):
            raise ValueError("affine halfspace arrangement decisions need nonzero slope")

    cells: list[AffineHalfspaceArrangementCellCertificate] = []
    source_leaves: list[BranchEventTreeLeafCertificate] = []
    stratified_leaves: list[StratifiedBranchLeafCertificate] = []
    base_polygon = _rectangle_polygon(box)
    for pattern_index, signs in enumerate(product((-1, 0, 1), repeat=len(specs))):
        polygon = base_polygon
        constraints: list[str] = []
        defining_ids = tuple(
            spec.decision_id for spec, sign in zip(specs, signs) if sign == 0
        )
        for spec, sign in zip(specs, signs):
            c0, ax, ay = spec.coefficients
            if sign < 0:
                polygon = _clip_polygon_halfplane(
                    polygon,
                    normal=(ax, ay),
                    rhs=-width - c0,
                )
                constraints.append(f"{spec.decision_id} <= {-width:g}")
            elif sign > 0:
                polygon = _clip_polygon_halfplane(
                    polygon,
                    normal=(-ax, -ay),
                    rhs=c0 - width,
                )
                constraints.append(f"{spec.decision_id} >= {width:g}")
            else:
                polygon = _clip_polygon_halfplane(
                    polygon,
                    normal=(ax, ay),
                    rhs=width - c0,
                )
                polygon = _clip_polygon_halfplane(
                    polygon,
                    normal=(-ax, -ay),
                    rhs=c0 + width,
                )
                constraints.append(f"{spec.decision_id} >= {-width:g}")
                constraints.append(f"{spec.decision_id} <= {width:g}")
            if not polygon:
                break
        area = _polygon_area(polygon)
        if area <= 1.0e-12:
            continue
        sign_vector = tuple(
            f"{spec.decision_id}:{'0' if sign == 0 else ('+' if sign > 0 else '-')}"
            for spec, sign in zip(specs, signs)
        )
        label = "+".join(defining_ids) if defining_ids else f"sign:{pattern_index}"
        cell_id = f"{arrangement_id}:{'slab' if defining_ids else 'cell'}:{pattern_index}:{label}"
        vertices = tuple((float(x), float(y)) for x, y in polygon)
        value_bounds = tuple(
            _affine_polygon_value_bounds(spec.coefficients, vertices)
            for spec in specs
        )
        missing: list[str] = []
        if not _affine_polygon_value_bounds_satisfy_signs(
            value_bounds,
            signs=signs,
            slab_half_width=width,
        ):
            missing.append("affine_halfspace_polygon_constraints_not_verified")
        certified = not missing
        cells.append(
            AffineHalfspaceArrangementCellCertificate(
                cell_id=cell_id,
                cell_kind=(
                    "affine_halfspace_arrangement_equality_slab_cell"
                    if defining_ids
                    else "affine_halfspace_arrangement_sign_cell"
                ),
                domain_box=box,
                sign_vector=sign_vector,
                defining_function_ids=defining_ids,
                constraints=tuple(constraints),
                value_bounds=value_bounds,
                vertices=vertices,
                area_lower_bound=float(max(0.0, area - 1.0e-12)),
                slab_half_width=width,
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        source_id = f"{cell_id}:leaf"
        source_leaves.append(
            BranchEventTreeLeafCertificate(
                leaf_id=source_id,
                leaf_type=(
                    "affine_halfspace_arrangement_equality_slab_leaf"
                    if defining_ids
                    else "affine_halfspace_arrangement_positive_margin_leaf"
                ),
                decision=(
                    "simultaneous_event_equality"
                    if defining_ids
                    else ",".join(sign_vector)
                ),
                source_type="AffineHalfspaceArrangement",
                depth=0,
                certified=bool(certified and not defining_ids),
                missing_obligations=(),
            )
        )
        if defining_ids:
            equality = EqualityStratumCertificate(
                stratum_id=cell_id,
                defining_function_ids=defining_ids,
                leaf_kind="simultaneous_event_equality",
                isolation_certified=certified,
                resolution_policy=str(equality_resolution_policy),
                certified=certified,
                missing_obligations=tuple(missing),
            )
            tie = EventOrderTieLeafCertificate(
                leaf_id=source_id,
                tied_event_ids=tuple(
                    event_id
                    for decision_id in defining_ids
                    for event_id in (
                        f"{decision_id}:negative_side",
                        f"{decision_id}:positive_side",
                    )
                ),
                equality_stratum=equality,
                certified=certified,
                missing_obligations=tuple(missing),
            )
            stratified_leaves.append(
                StratifiedBranchLeafCertificate(
                    leaf_id=f"stratified:{source_id}",
                    source_leaf_id=source_id,
                    leaf_kind="simultaneous_event_equality",
                    terminal_response_kind="recursive_affine_halfspace_arrangement_slab",
                    terminal_response_certified=False,
                    source_leaf_certified=True,
                    equality_stratum=equality,
                    event_order_tie=tie,
                    missing_obligations=tuple(missing),
                )
            )
        else:
            stratified_leaves.append(
                StratifiedBranchLeafCertificate(
                    leaf_id=f"stratified:{source_id}",
                    source_leaf_id=source_id,
                    leaf_kind="positive_margin_unique_event",
                    terminal_response_kind="affine_halfspace_arrangement_sign_cell",
                    terminal_response_certified=certified,
                    source_leaf_certified=certified,
                    decision_functions=tuple(
                        AnalyticDecisionFunctionCertificate(
                            function_id=f"{cell_id}:{spec.decision_id}",
                            function_kind="affine_halfspace_arrangement_decision_sign",
                            margin_lower_bound=width,
                            lipschitz_bound=_affine_box_lipschitz_bound(
                                spec.coefficients,
                            ),
                            certified=certified,
                            missing_obligations=tuple(missing),
                        )
                        for spec in specs
                    ),
                    missing_obligations=tuple(missing),
                )
            )

    domain_area = float((box[0][1] - box[0][0]) * (box[1][1] - box[1][0]))
    cell_area_sum = float(sum(_polygon_area(cell.vertices) for cell in cells))
    cover_area_gap = abs(domain_area - cell_area_sum)
    area_tolerance = 1.0e-9 * max(1.0, domain_area, cell_area_sum)
    area_cover_certified = bool(cover_area_gap <= area_tolerance)
    cover_certified = bool(
        cells
        and all(cell.proof_certified for cell in cells)
        and area_cover_certified
    )
    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="AffineHalfspaceArrangement",
        leaf_certificates=tuple(source_leaves),
        cover_certified=cover_certified,
        leaf_decisions_certified=bool(
            source_leaves
            and all(
                leaf.certified
                for leaf in source_leaves
                if "equality" not in leaf.leaf_type
            )
        ),
        equality_strata_explicit=any(cell.equality for cell in cells),
        obligations=(
            BranchEventTreeObligation(
                obligation="affine_halfspace_arrangement_domain_valid",
                certified=True,
                detail=f"box={box!r}; decision_count={len(specs)}",
            ),
            BranchEventTreeObligation(
                obligation="affine_halfspace_arrangement_trichotomy_cells",
                certified=cover_certified,
                detail=(
                    "pattern_count="
                    f"{3 ** len(specs)}; nonempty_cell_count={len(cells)}; "
                    f"area_gap={cover_area_gap:g}"
                ),
            ),
            BranchEventTreeObligation(
                obligation="affine_halfspace_arrangement_area_cover",
                certified=area_cover_certified,
                detail=(
                    f"domain_area={domain_area:g}; "
                    f"cell_area_sum={cell_area_sum:g}; "
                    f"gap_upper_bound={cover_area_gap:g}"
                ),
            ),
            BranchEventTreeObligation(
                obligation="affine_halfspace_arrangement_equality_slabs_explicit",
                certified=True,
                detail=f"equality_cell_count={sum(cell.equality for cell in cells)}",
                required=False,
            ),
        ),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=tuple(stratified_leaves),
    )
    return AffineHalfspaceArrangementStratificationCertificate(
        arrangement_id=arrangement_id,
        decision_functions=specs,
        domain_box=box,
        slab_half_width=width,
        cells=tuple(cells),
        domain_area=domain_area,
        cell_area_sum=cell_area_sum,
        cover_area_gap_upper_bound=float(cover_area_gap),
        area_cover_certified=area_cover_certified,
        source_tree=source_tree,
        stratified_tree=stratified,
        statement=(
            "A finite two-dimensional box with finitely many affine decision "
            "boundaries admits a constructor-derived halfspace-cell "
            "stratification after a positive equality-slab thickness is fixed, "
            "with an explicit area-cover check for the clipped polygon cells."
        ),
        proof_sketch=(
            "For each affine discriminator, split the box by the trichotomy "
            "f<=-epsilon, |f|<=epsilon, and f>=epsilon.  Enumerate every sign "
            "pattern and use convex polygon halfplane clipping to derive the "
            "corresponding nonempty cell.  Strict sign cells have positive "
            "margin epsilon to every equality slab and become terminal leaves. "
            "Cells containing one or more slabs remain explicit equality "
            "strata for recursive lower-rank consumption.  Since every point "
            "in the box satisfies exactly one trichotomy branch for each "
            "decision, the nonempty generated cells cover the box; the "
            "constructor independently sums clipped polygon areas against the "
            "rectangle area and records any cover gap instead of relying only "
            "on the enumeration argument."
        ),
    )


def certify_affine_halfspace_arrangement_recursive_consumption(
    arrangement: AffineHalfspaceArrangementStratificationCertificate,
    *,
    root_dimension: int,
    root_rank: int | None = None,
    child_consumptions: Mapping[
        str,
        RecursiveStratifiedBranchEventConsumptionCertificate,
    ]
    | tuple[
        tuple[str, RecursiveStratifiedBranchEventConsumptionCertificate],
        ...,
    ] = (),
    recursion_kind: str = "affine_halfspace_arrangement",
) -> RecursiveStratifiedBranchEventConsumptionCertificate:
    """Consume a two-dimensional affine halfspace arrangement by recursive slabs."""

    if not isinstance(
        arrangement,
        AffineHalfspaceArrangementStratificationCertificate,
    ):
        raise TypeError(
            "arrangement must be an "
            "AffineHalfspaceArrangementStratificationCertificate"
        )
    child_by_leaf = _normalize_arrangement_child_consumptions(
        arrangement,  # type: ignore[arg-type]
        child_consumptions,
    )
    if (
        not child_by_leaf
        and arrangement.proof_certified
        and arrangement.equality_stratum_count
    ):
        child_by_leaf = _normalize_arrangement_child_consumptions(
            arrangement,  # type: ignore[arg-type]
            derive_affine_halfspace_arrangement_child_consumptions(
                arrangement,
                child_root_rank=0,
            ),
        )
    return certify_recursive_stratified_branch_event_consumption(
        arrangement.stratified_tree,
        root_dimension=root_dimension,
        root_rank=root_rank,
        child_consumptions=child_by_leaf,
        recursion_kind=recursion_kind,
    )


def derive_affine_halfspace_arrangement_line_child_consumptions(
    arrangement: AffineHalfspaceArrangementStratificationCertificate,
    *,
    child_root_rank: int | None = 0,
    minimum_interval_width: float = 1.0e-9,
    coincidence_tolerance: float = 1.0e-12,
) -> dict[str, RecursiveStratifiedBranchEventConsumptionCertificate]:
    """Derive 1D child consumptions for 2D affine equality lines.

    This is a scoped recursive-construction step, not the arbitrary
    branch/event partition theorem.  For each equality cell defined by one
    affine boundary ``f=0`` or by multiple coincident affine boundaries, the
    constructor parameterizes the exact line, intersects it with the parent
    box and the cell's remaining sign inequalities, restricts the other
    affine decisions to that line, and consumes the resulting
    one-dimensional affine arrangement when it is already terminal after
    root construction.
    """

    if not isinstance(
        arrangement,
        AffineHalfspaceArrangementStratificationCertificate,
    ):
        raise TypeError(
            "arrangement must be an "
            "AffineHalfspaceArrangementStratificationCertificate"
        )
    if not arrangement.proof_certified:
        raise ValueError("affine halfspace arrangement must be proof-certified")
    width = float(minimum_interval_width)
    if not (np.isfinite(width) and width > 0.0):
        raise ValueError("minimum_interval_width must be positive")
    tolerance = float(coincidence_tolerance)
    if not (np.isfinite(tolerance) and tolerance > 0.0):
        raise ValueError("coincidence_tolerance must be positive")

    specs_by_id = {spec.decision_id: spec for spec in arrangement.decision_functions}
    child_consumptions: dict[str, RecursiveStratifiedBranchEventConsumptionCertificate] = {}
    for cell in arrangement.cells:
        if not cell.equality or not cell.defining_function_ids:
            continue
        defining_ids = set(cell.defining_function_ids)
        defining_id = cell.defining_function_ids[0]
        defining_spec = specs_by_id.get(defining_id)
        if defining_spec is None:
            continue
        if not all(
            other is not None
            and _affine_2d_boundaries_define_same_line(
                defining_spec,
                other,
                tolerance=tolerance,
            )
            for other in (
                specs_by_id.get(decision_id)
                for decision_id in cell.defining_function_ids[1:]
            )
        ):
            continue
        line = _affine_halfspace_arrangement_cell_line_domain(
            cell=cell,
            defining_spec=defining_spec,
            specs_by_id=specs_by_id,
            domain_box=arrangement.domain_box,
            slab_half_width=arrangement.slab_half_width,
        )
        if line is None:
            continue
        point, direction, domain = line
        if domain[1] - domain[0] <= width:
            continue
        restricted_specs = tuple(
            spec
            for spec in (
                _restrict_affine_box_decision_to_line(spec, point, direction)
                for spec in arrangement.decision_functions
                if spec.decision_id not in defining_ids
            )
            if spec is not None
        )
        if restricted_specs:
            child_arrangement = certify_affine_decision_arrangement_stratified_branch_event_tree(
                arrangement_id=f"{arrangement.arrangement_id}:line_child:{cell.cell_id}",
                decision_functions=restricted_specs,
                domain=domain,
            )
            child = certify_polynomial_decision_arrangement_recursive_consumption(
                child_arrangement,
                root_dimension=1,
                root_rank=child_root_rank,
                child_consumptions=(
                    derive_polynomial_decision_arrangement_child_consumptions(
                        child_arrangement,
                        child_root_rank=child_root_rank,
                    )
                ),
            )
        else:
            child = _one_dimensional_affine_line_consumption(
                cell=cell,
                point=point,
                direction=direction,
                domain=domain,
                root_rank=child_root_rank,
            )
        if child.proof_certified:
            child_consumptions[cell.cell_id] = child
    return child_consumptions


def derive_affine_halfspace_arrangement_point_child_consumptions(
    arrangement: AffineHalfspaceArrangementStratificationCertificate,
    *,
    child_root_rank: int | None = 0,
    tolerance: float = 1.0e-9,
) -> dict[str, RecursiveStratifiedBranchEventConsumptionCertificate]:
    """Derive 0D child consumptions for independent two-boundary cells."""

    if not isinstance(
        arrangement,
        AffineHalfspaceArrangementStratificationCertificate,
    ):
        raise TypeError(
            "arrangement must be an "
            "AffineHalfspaceArrangementStratificationCertificate"
        )
    if not arrangement.proof_certified:
        raise ValueError("affine halfspace arrangement must be proof-certified")
    epsilon = float(tolerance)
    if not (np.isfinite(epsilon) and epsilon > 0.0):
        raise ValueError("tolerance must be positive")

    specs_by_id = {spec.decision_id: spec for spec in arrangement.decision_functions}
    child_consumptions: dict[str, RecursiveStratifiedBranchEventConsumptionCertificate] = {}
    for cell in arrangement.cells:
        if not cell.equality or len(cell.defining_function_ids) != 2:
            continue
        first = specs_by_id.get(cell.defining_function_ids[0])
        second = specs_by_id.get(cell.defining_function_ids[1])
        if first is None or second is None:
            continue
        point = _solve_affine_two_boundary_point(first, second)
        if point is None:
            continue
        if not _point_in_box(point, arrangement.domain_box, tolerance=epsilon):
            continue
        if not _point_satisfies_affine_cell_signs(
            point,
            cell=cell,
            specs_by_id=specs_by_id,
            slab_half_width=arrangement.slab_half_width,
            tolerance=epsilon,
        ):
            continue
        child = _zero_dimensional_affine_point_consumption(
            cell=cell,
            point=point,
            root_rank=child_root_rank,
        )
        if child.proof_certified:
            child_consumptions[cell.cell_id] = child
    return child_consumptions


def derive_affine_halfspace_arrangement_child_consumptions(
    arrangement: AffineHalfspaceArrangementStratificationCertificate,
    *,
    child_root_rank: int | None = 0,
    minimum_interval_width: float = 1.0e-9,
    point_tolerance: float = 1.0e-9,
) -> dict[str, RecursiveStratifiedBranchEventConsumptionCertificate]:
    """Derive supported line and point children for a 2D affine arrangement."""

    children = derive_affine_halfspace_arrangement_line_child_consumptions(
        arrangement,
        child_root_rank=child_root_rank,
        minimum_interval_width=minimum_interval_width,
    )
    children.update(
        derive_affine_halfspace_arrangement_point_child_consumptions(
            arrangement,
            child_root_rank=child_root_rank,
            tolerance=point_tolerance,
        )
    )
    return children


def certify_affine_halfspace_3d_arrangement_stratified_branch_event_tree(
    *,
    arrangement_id: str,
    decision_functions: tuple[AffineBoxDecisionFunctionSpec, ...],
    domain_box: tuple[tuple[float, float], ...],
    slab_half_width: float,
    equality_resolution_policy: str = "lower_dimensional_recursive_stratum",
) -> AffineHalfspaceArrangement3DStratificationCertificate:
    """Derive a 3D convex-cell arrangement from affine halfspace decisions.

    This is the spatial analogue of the 2D oblique affine arrangement
    constructor.  Cells are represented by halfspaces plus enumerated vertices;
    the constructor computes a conservative volume-cover check for the finite
    generated arrangement.  It is still a scoped finite arrangement theorem,
    not arbitrary interval-input recursive partition generation.
    """

    arrangement_id = str(arrangement_id)
    box = tuple((float(left), float(right)) for left, right in domain_box)
    specs = tuple(
        AffineBoxDecisionFunctionSpec(
            decision_id=str(spec.decision_id),
            coefficients=tuple(float(value) for value in spec.coefficients),
        )
        for spec in decision_functions
    )
    width = float(slab_half_width)
    if not arrangement_id:
        raise ValueError("affine halfspace 3D arrangement requires an id")
    if len(box) != 3:
        raise ValueError("affine halfspace 3D arrangement requires a 3D box")
    if any(
        not (np.isfinite(left) and np.isfinite(right) and left < right)
        for left, right in box
    ):
        raise ValueError("affine halfspace 3D arrangement requires finite nonempty intervals")
    if not (np.isfinite(width) and width > 0.0):
        raise ValueError("affine halfspace 3D arrangement slab_half_width must be positive")
    if not specs:
        raise ValueError("affine halfspace 3D arrangement requires at least one decision")
    for spec in specs:
        if not spec.decision_id:
            raise ValueError("affine halfspace 3D arrangement decisions require ids")
        if len(spec.coefficients) != 4:
            raise ValueError("affine halfspace 3D arrangement decisions must be 3D affine")
        if any(not np.isfinite(value) for value in spec.coefficients):
            raise ValueError("affine halfspace 3D arrangement coefficients must be finite")
        if not any(abs(slope) > 0.0 for slope in spec.coefficients[1:]):
            raise ValueError("affine halfspace 3D arrangement decisions need nonzero slope")

    cells: list[AffineHalfspaceArrangement3DCellCertificate] = []
    cell_volumes: list[float] = []
    source_leaves: list[BranchEventTreeLeafCertificate] = []
    stratified_leaves: list[StratifiedBranchLeafCertificate] = []
    base_constraints = _box_halfspace_constraints_3d(box)
    for pattern_index, signs in enumerate(product((-1, 0, 1), repeat=len(specs))):
        constraints = list(base_constraints)
        constraint_labels: list[str] = []
        defining_ids = tuple(
            spec.decision_id for spec, sign in zip(specs, signs) if sign == 0
        )
        for spec, sign in zip(specs, signs):
            c0, ax, ay, az = spec.coefficients
            if sign < 0:
                constraints.append(((ax, ay, az), -width - c0))
                constraint_labels.append(f"{spec.decision_id} <= {-width:g}")
            elif sign > 0:
                constraints.append(((-ax, -ay, -az), c0 - width))
                constraint_labels.append(f"{spec.decision_id} >= {width:g}")
            else:
                constraints.append(((ax, ay, az), width - c0))
                constraints.append(((-ax, -ay, -az), c0 + width))
                constraint_labels.append(f"{spec.decision_id} >= {-width:g}")
                constraint_labels.append(f"{spec.decision_id} <= {width:g}")
        effective_constraints = _deduplicate_halfspace_constraints_3d(
            tuple(constraints),
        )
        vertices = _halfspace_polyhedron_vertices_3d(effective_constraints)
        volume = _polyhedron_volume_3d(vertices, effective_constraints)
        if volume <= 1.0e-12:
            continue
        cell_volumes.append(float(volume))
        sign_vector = tuple(
            f"{spec.decision_id}:{'0' if sign == 0 else ('+' if sign > 0 else '-')}"
            for spec, sign in zip(specs, signs)
        )
        label = "+".join(defining_ids) if defining_ids else f"sign:{pattern_index}"
        cell_id = f"{arrangement_id}:{'slab' if defining_ids else 'cell'}:{pattern_index}:{label}"
        value_bounds = tuple(
            _affine_polyhedron_value_bounds(spec.coefficients, vertices)
            for spec in specs
        )
        missing: list[str] = []
        if len(vertices) < 4:
            missing.append("affine_halfspace_3d_cell_vertices_not_full_dimensional")
        if not _affine_polygon_value_bounds_satisfy_signs(
            value_bounds,
            signs=signs,
            slab_half_width=width,
        ):
            missing.append("affine_halfspace_3d_value_constraints_not_verified")
        certified = not missing
        cells.append(
            AffineHalfspaceArrangement3DCellCertificate(
                cell_id=cell_id,
                cell_kind=(
                    "affine_halfspace_3d_arrangement_equality_slab_cell"
                    if defining_ids
                    else "affine_halfspace_3d_arrangement_sign_cell"
                ),
                domain_box=box,
                sign_vector=sign_vector,
                defining_function_ids=defining_ids,
                constraints=tuple(constraint_labels),
                value_bounds=value_bounds,
                vertices=vertices,
                volume_lower_bound=float(max(0.0, volume - 1.0e-12)),
                slab_half_width=width,
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        source_id = f"{cell_id}:leaf"
        source_leaves.append(
            BranchEventTreeLeafCertificate(
                leaf_id=source_id,
                leaf_type=(
                    "affine_halfspace_3d_arrangement_equality_slab_leaf"
                    if defining_ids
                    else "affine_halfspace_3d_arrangement_positive_margin_leaf"
                ),
                decision=(
                    "simultaneous_event_equality"
                    if defining_ids
                    else ",".join(sign_vector)
                ),
                source_type="AffineHalfspace3DArrangement",
                depth=0,
                certified=bool(certified and not defining_ids),
                missing_obligations=(),
            )
        )
        if defining_ids:
            equality = EqualityStratumCertificate(
                stratum_id=cell_id,
                defining_function_ids=defining_ids,
                leaf_kind="simultaneous_event_equality",
                isolation_certified=certified,
                resolution_policy=str(equality_resolution_policy),
                certified=certified,
                missing_obligations=tuple(missing),
            )
            tie = EventOrderTieLeafCertificate(
                leaf_id=source_id,
                tied_event_ids=tuple(
                    event_id
                    for decision_id in defining_ids
                    for event_id in (
                        f"{decision_id}:negative_side",
                        f"{decision_id}:positive_side",
                    )
                ),
                equality_stratum=equality,
                certified=certified,
                missing_obligations=tuple(missing),
            )
            stratified_leaves.append(
                StratifiedBranchLeafCertificate(
                    leaf_id=f"stratified:{source_id}",
                    source_leaf_id=source_id,
                    leaf_kind="simultaneous_event_equality",
                    terminal_response_kind="recursive_affine_halfspace_3d_arrangement_slab",
                    terminal_response_certified=False,
                    source_leaf_certified=True,
                    equality_stratum=equality,
                    event_order_tie=tie,
                    missing_obligations=tuple(missing),
                )
            )
        else:
            stratified_leaves.append(
                StratifiedBranchLeafCertificate(
                    leaf_id=f"stratified:{source_id}",
                    source_leaf_id=source_id,
                    leaf_kind="positive_margin_unique_event",
                    terminal_response_kind="affine_halfspace_3d_arrangement_sign_cell",
                    terminal_response_certified=certified,
                    source_leaf_certified=certified,
                    decision_functions=tuple(
                        AnalyticDecisionFunctionCertificate(
                            function_id=f"{cell_id}:{spec.decision_id}",
                            function_kind="affine_halfspace_3d_arrangement_decision_sign",
                            margin_lower_bound=width,
                            lipschitz_bound=_affine_box_lipschitz_bound(
                                spec.coefficients,
                            ),
                            certified=certified,
                            missing_obligations=tuple(missing),
                        )
                        for spec in specs
                    ),
                    missing_obligations=tuple(missing),
                )
            )

    domain_volume = float(np.prod([right - left for left, right in box]))
    cell_volume_sum = float(sum(cell_volumes))
    cover_volume_gap = abs(domain_volume - cell_volume_sum)
    volume_tolerance = 1.0e-8 * max(1.0, domain_volume, cell_volume_sum)
    volume_cover_certified = bool(cover_volume_gap <= volume_tolerance)
    cover_certified = bool(
        cells
        and all(cell.proof_certified for cell in cells)
        and volume_cover_certified
    )
    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="AffineHalfspace3DArrangement",
        leaf_certificates=tuple(source_leaves),
        cover_certified=cover_certified,
        leaf_decisions_certified=bool(
            source_leaves
            and all(
                leaf.certified
                for leaf in source_leaves
                if "equality" not in leaf.leaf_type
            )
        ),
        equality_strata_explicit=any(cell.equality for cell in cells),
        obligations=(
            BranchEventTreeObligation(
                obligation="affine_halfspace_3d_arrangement_domain_valid",
                certified=True,
                detail=f"box={box!r}; decision_count={len(specs)}",
            ),
            BranchEventTreeObligation(
                obligation="affine_halfspace_3d_arrangement_trichotomy_cells",
                certified=cover_certified,
                detail=(
                    f"pattern_count={3 ** len(specs)}; "
                    f"nonempty_cell_count={len(cells)}; "
                    f"volume_gap={cover_volume_gap:g}"
                ),
            ),
            BranchEventTreeObligation(
                obligation="affine_halfspace_3d_arrangement_volume_cover",
                certified=volume_cover_certified,
                detail=(
                    f"domain_volume={domain_volume:g}; "
                    f"cell_volume_sum={cell_volume_sum:g}; "
                    f"gap_upper_bound={cover_volume_gap:g}"
                ),
            ),
            BranchEventTreeObligation(
                obligation="affine_halfspace_3d_arrangement_equality_slabs_explicit",
                certified=True,
                detail=f"equality_cell_count={sum(cell.equality for cell in cells)}",
                required=False,
            ),
        ),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=tuple(stratified_leaves),
    )
    return AffineHalfspaceArrangement3DStratificationCertificate(
        arrangement_id=arrangement_id,
        decision_functions=specs,
        domain_box=box,
        slab_half_width=width,
        cells=tuple(cells),
        domain_volume=domain_volume,
        cell_volume_sum=cell_volume_sum,
        cover_volume_gap_upper_bound=float(cover_volume_gap),
        volume_cover_certified=volume_cover_certified,
        source_tree=source_tree,
        stratified_tree=stratified,
        statement=(
            "A finite three-dimensional box with finitely many affine decision "
            "boundaries admits a constructor-derived halfspace-cell "
            "stratification after a positive equality-slab thickness is fixed, "
            "with explicit vertex and volume-cover checks."
        ),
        proof_sketch=(
            "For each affine discriminator, split the box by f<=-epsilon, "
            "|f|<=epsilon, and f>=epsilon.  Enumerate every sign pattern and "
            "represent the resulting 3D cell as a finite halfspace system.  "
            "Vertices are recovered from triples of active constraint planes "
            "and checked against all halfspaces. Strict sign cells have margin "
            "epsilon and become terminal leaves; slab cells become explicit "
            "recursive equality strata.  The constructor sums the volumes of "
            "the generated convex cells and compares that sum to the input box "
            "volume as an independent finite-cover guard."
        ),
    )


def certify_affine_halfspace_3d_arrangement_recursive_consumption(
    arrangement: AffineHalfspaceArrangement3DStratificationCertificate,
    *,
    root_dimension: int,
    root_rank: int | None = None,
    child_consumptions: Mapping[
        str,
        RecursiveStratifiedBranchEventConsumptionCertificate,
    ]
    | tuple[
        tuple[str, RecursiveStratifiedBranchEventConsumptionCertificate],
        ...,
    ] = (),
    recursion_kind: str = "affine_halfspace_3d_arrangement",
) -> RecursiveStratifiedBranchEventConsumptionCertificate:
    """Consume a three-dimensional affine halfspace arrangement by recursive slabs."""

    if not isinstance(
        arrangement,
        AffineHalfspaceArrangement3DStratificationCertificate,
    ):
        raise TypeError(
            "arrangement must be an "
            "AffineHalfspaceArrangement3DStratificationCertificate"
        )
    child_by_leaf = _normalize_arrangement_child_consumptions(
        arrangement,  # type: ignore[arg-type]
        child_consumptions,
    )
    if (
        not child_by_leaf
        and arrangement.proof_certified
        and arrangement.equality_stratum_count
    ):
        child_by_leaf = _normalize_arrangement_child_consumptions(
            arrangement,  # type: ignore[arg-type]
            derive_affine_halfspace_3d_arrangement_child_consumptions(
                arrangement,
                child_root_rank=0,
            ),
        )
    return certify_recursive_stratified_branch_event_consumption(
        arrangement.stratified_tree,
        root_dimension=root_dimension,
        root_rank=root_rank,
        child_consumptions=child_by_leaf,
        recursion_kind=recursion_kind,
    )


def derive_affine_halfspace_3d_arrangement_plane_child_consumptions(
    arrangement: AffineHalfspaceArrangement3DStratificationCertificate,
    *,
    child_root_rank: int | None = 0,
    tolerance: float = 1.0e-9,
) -> dict[str, RecursiveStratifiedBranchEventConsumptionCertificate]:
    """Derive terminal 2D plane children for one geometric 3D plane.

    This is a scoped constructor-derived descent case.  It handles an equality
    cell with one defining affine boundary, or multiple coincident defining
    boundaries for the same affine plane, by projecting the exact plane-box
    slice to 2D coordinates and clipping that polygon by the cell's remaining
    fixed sign inequalities.  More complicated lower-dimensional recursion is
    left to explicit child certificates.
    """

    if not isinstance(
        arrangement,
        AffineHalfspaceArrangement3DStratificationCertificate,
    ):
        raise TypeError(
            "arrangement must be an "
            "AffineHalfspaceArrangement3DStratificationCertificate"
        )
    if not arrangement.proof_certified:
        raise ValueError("affine halfspace 3D arrangement must be proof-certified")
    epsilon = float(tolerance)
    if not (np.isfinite(epsilon) and epsilon > 0.0):
        raise ValueError("tolerance must be positive")

    specs_by_id = {spec.decision_id: spec for spec in arrangement.decision_functions}
    child_consumptions: dict[str, RecursiveStratifiedBranchEventConsumptionCertificate] = {}
    for cell in arrangement.cells:
        if not cell.equality or not cell.defining_function_ids:
            continue
        defining_id = cell.defining_function_ids[0]
        defining_spec = specs_by_id.get(defining_id)
        if defining_spec is None:
            continue
        if not all(
            other is not None
            and _affine_3d_boundaries_define_same_plane(
                defining_spec,
                other,
                tolerance=epsilon,
            )
            for other in (
                specs_by_id.get(decision_id)
                for decision_id in cell.defining_function_ids[1:]
            )
        ):
            continue
        geometry = _affine_halfspace_3d_cell_plane_domain(
            cell=cell,
            defining_spec=defining_spec,
            specs_by_id=specs_by_id,
            domain_box=arrangement.domain_box,
            slab_half_width=arrangement.slab_half_width,
            tolerance=epsilon,
        )
        if geometry is None:
            continue
        point, basis_u, basis_v, parameter_domain, area = geometry
        child = _two_dimensional_affine_plane_consumption(
            cell=cell,
            point=point,
            basis_u=basis_u,
            basis_v=basis_v,
            parameter_domain=parameter_domain,
            area_lower_bound=area,
            root_rank=child_root_rank,
        )
        if child.proof_certified:
            child_consumptions[cell.cell_id] = child
    return child_consumptions


def derive_affine_halfspace_3d_arrangement_line_child_consumptions(
    arrangement: AffineHalfspaceArrangement3DStratificationCertificate,
    *,
    child_root_rank: int | None = 0,
    minimum_interval_width: float = 1.0e-9,
) -> dict[str, RecursiveStratifiedBranchEventConsumptionCertificate]:
    """Derive 1D child consumptions for independent two-plane 3D cells."""

    if not isinstance(
        arrangement,
        AffineHalfspaceArrangement3DStratificationCertificate,
    ):
        raise TypeError(
            "arrangement must be an "
            "AffineHalfspaceArrangement3DStratificationCertificate"
        )
    if not arrangement.proof_certified:
        raise ValueError("affine halfspace 3D arrangement must be proof-certified")
    width = float(minimum_interval_width)
    if not (np.isfinite(width) and width > 0.0):
        raise ValueError("minimum_interval_width must be positive")

    specs_by_id = {spec.decision_id: spec for spec in arrangement.decision_functions}
    child_consumptions: dict[str, RecursiveStratifiedBranchEventConsumptionCertificate] = {}
    for cell in arrangement.cells:
        if not cell.equality or len(cell.defining_function_ids) != 2:
            continue
        first = specs_by_id.get(cell.defining_function_ids[0])
        second = specs_by_id.get(cell.defining_function_ids[1])
        if first is None or second is None:
            continue
        line = _affine_halfspace_3d_cell_line_domain(
            cell=cell,
            first_spec=first,
            second_spec=second,
            specs_by_id=specs_by_id,
            domain_box=arrangement.domain_box,
            slab_half_width=arrangement.slab_half_width,
        )
        if line is None:
            continue
        point, direction, domain = line
        if domain[1] - domain[0] <= width:
            continue
        restricted_specs = tuple(
            spec
            for spec in (
                _restrict_affine_3d_decision_to_line(spec, point, direction)
                for spec in arrangement.decision_functions
                if spec.decision_id not in cell.defining_function_ids
            )
            if spec is not None
        )
        if restricted_specs:
            child_arrangement = certify_affine_decision_arrangement_stratified_branch_event_tree(
                arrangement_id=f"{arrangement.arrangement_id}:line_child:{cell.cell_id}",
                decision_functions=restricted_specs,
                domain=domain,
            )
            child = certify_polynomial_decision_arrangement_recursive_consumption(
                child_arrangement,
                root_dimension=1,
                root_rank=child_root_rank,
                child_consumptions=(
                    derive_polynomial_decision_arrangement_child_consumptions(
                        child_arrangement,
                        child_root_rank=child_root_rank,
                    )
                ),
            )
        else:
            child = _one_dimensional_affine_spatial_line_consumption(
                cell=cell,
                point=point,
                direction=direction,
                domain=domain,
                root_rank=child_root_rank,
            )
        if child.proof_certified:
            child_consumptions[cell.cell_id] = child
    return child_consumptions


def derive_affine_halfspace_3d_arrangement_point_child_consumptions(
    arrangement: AffineHalfspaceArrangement3DStratificationCertificate,
    *,
    child_root_rank: int | None = 0,
    tolerance: float = 1.0e-9,
) -> dict[str, RecursiveStratifiedBranchEventConsumptionCertificate]:
    """Derive 0D child consumptions for independent three-plane 3D cells."""

    if not isinstance(
        arrangement,
        AffineHalfspaceArrangement3DStratificationCertificate,
    ):
        raise TypeError(
            "arrangement must be an "
            "AffineHalfspaceArrangement3DStratificationCertificate"
        )
    if not arrangement.proof_certified:
        raise ValueError("affine halfspace 3D arrangement must be proof-certified")
    epsilon = float(tolerance)
    if not (np.isfinite(epsilon) and epsilon > 0.0):
        raise ValueError("tolerance must be positive")

    specs_by_id = {spec.decision_id: spec for spec in arrangement.decision_functions}
    child_consumptions: dict[str, RecursiveStratifiedBranchEventConsumptionCertificate] = {}
    for cell in arrangement.cells:
        if not cell.equality or len(cell.defining_function_ids) != 3:
            continue
        specs = tuple(specs_by_id.get(decision_id) for decision_id in cell.defining_function_ids)
        if any(spec is None for spec in specs):
            continue
        point = _solve_affine_three_boundary_point(specs)  # type: ignore[arg-type]
        if point is None:
            continue
        if not _point3_in_box(point, arrangement.domain_box, tolerance=epsilon):
            continue
        if not _point3_satisfies_affine_cell_signs(
            point,
            cell=cell,
            specs_by_id=specs_by_id,
            slab_half_width=arrangement.slab_half_width,
            tolerance=epsilon,
        ):
            continue
        child = _zero_dimensional_affine_spatial_point_consumption(
            cell=cell,
            point=point,
            root_rank=child_root_rank,
        )
        if child.proof_certified:
            child_consumptions[cell.cell_id] = child
    return child_consumptions


def derive_affine_halfspace_3d_arrangement_child_consumptions(
    arrangement: AffineHalfspaceArrangement3DStratificationCertificate,
    *,
    child_root_rank: int | None = 0,
    plane_tolerance: float = 1.0e-9,
    minimum_interval_width: float = 1.0e-9,
    point_tolerance: float = 1.0e-9,
) -> dict[str, RecursiveStratifiedBranchEventConsumptionCertificate]:
    """Derive supported plane, line, and point children for a 3D arrangement."""

    children = derive_affine_halfspace_3d_arrangement_plane_child_consumptions(
        arrangement,
        child_root_rank=child_root_rank,
        tolerance=plane_tolerance,
    )
    children.update(
        derive_affine_halfspace_3d_arrangement_line_child_consumptions(
            arrangement,
            child_root_rank=child_root_rank,
            minimum_interval_width=minimum_interval_width,
        )
    )
    children.update(
        derive_affine_halfspace_3d_arrangement_point_child_consumptions(
            arrangement,
            child_root_rank=child_root_rank,
            tolerance=point_tolerance,
        )
    )
    return children


def certify_polynomial_decision_stratified_branch_event_tree(
    *,
    decision_id: str,
    coefficients: tuple[float, ...] | list[float] | np.ndarray,
    domain: tuple[float, float],
    root_brackets: tuple[tuple[float, float], ...] = (),
    equality_resolution_policy: str = "lower_dimensional_recursive_stratum",
) -> PolynomialDecisionStratificationCertificate:
    """Derive a finite stratified tree from one polynomial discriminator.

    Root brackets are part of the certificate input, but every bracket is
    independently checked by interval arithmetic: the polynomial value must
    contain zero and the derivative interval must exclude zero.  Sign cells
    between root brackets must have a strict interval sign.  Equality brackets
    become explicit zero-margin leaves rather than being merged into a box.
    """

    decision_id = str(decision_id)
    coeffs = tuple(float(value) for value in np.asarray(coefficients, dtype=float).reshape(-1))
    lower, upper = (float(domain[0]), float(domain[1]))
    roots = tuple((float(left), float(right)) for left, right in root_brackets)
    derivative = interval_polyder(np.asarray(coeffs, dtype=float))
    sorted_roots = tuple(sorted(roots))
    strata: list[PolynomialDecisionStratumCertificate] = []
    source_leaves: list[BranchEventTreeLeafCertificate] = []
    stratified_leaves: list[StratifiedBranchLeafCertificate] = []
    cover_cursor = lower
    domain_ok = bool(
        decision_id
        and len(coeffs) >= 1
        and np.isfinite(lower)
        and np.isfinite(upper)
        and lower < upper
        and _root_brackets_sorted_and_inside(sorted_roots, lower, upper)
    )

    def add_sign_cell(index: int, left: float, right: float) -> None:
        if right <= left:
            return
        interval = FloatInterval(left, right)
        value = interval_polynomial_eval(np.asarray(coeffs, dtype=float), interval)
        sign = interval_sign(value)
        missing: list[str] = []
        if sign == 0:
            missing.append("polynomial_sign_cell_not_separated_from_zero")
        certified = bool(domain_ok and sign != 0)
        stratum_id = f"{decision_id}:sign:{index}"
        strata.append(
            PolynomialDecisionStratumCertificate(
                stratum_id=stratum_id,
                stratum_kind="positive_sign_cell" if sign > 0 else "negative_sign_cell",
                interval=(left, right),
                value_interval=value.as_tuple(),
                sign=sign,
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        source_id = f"{stratum_id}:leaf"
        source_leaves.append(
            BranchEventTreeLeafCertificate(
                leaf_id=source_id,
                leaf_type="polynomial_positive_margin_leaf",
                decision=("positive" if sign > 0 else "negative"),
                source_type="PolynomialDecisionStratification",
                depth=0,
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        stratified_leaves.append(
            StratifiedBranchLeafCertificate(
                leaf_id=f"stratified:{source_id}",
                source_leaf_id=source_id,
                leaf_kind="positive_margin_unique_event",
                terminal_response_kind="polynomial_sign_decision",
                terminal_response_certified=certified,
                source_leaf_certified=certified,
                decision_functions=(
                    AnalyticDecisionFunctionCertificate(
                        function_id=stratum_id,
                        function_kind="polynomial_decision_sign",
                        margin_lower_bound=min(abs(value.lower), abs(value.upper)) if sign != 0 else 0.0,
                        lipschitz_bound=_polynomial_derivative_lipschitz_bound(derivative, interval),
                        certified=certified,
                        missing_obligations=tuple(missing),
                    ),
                ),
                missing_obligations=tuple(missing),
            )
        )

    for index, (root_lower, root_upper) in enumerate(sorted_roots):
        add_sign_cell(index, cover_cursor, root_lower)
        root_interval = FloatInterval(root_lower, root_upper)
        value = interval_polynomial_eval(np.asarray(coeffs, dtype=float), root_interval)
        derivative_value = interval_polynomial_eval(derivative, root_interval) if derivative else FloatInterval.point(0.0)
        derivative_sign = interval_sign(derivative_value)
        missing = []
        if interval_sign(value) != 0:
            missing.append("root_bracket_value_does_not_contain_zero")
        if derivative_sign == 0:
            missing.append("root_bracket_derivative_sign_not_isolated")
        certified = bool(domain_ok and not missing)
        stratum_id = f"{decision_id}:root:{index}"
        strata.append(
            PolynomialDecisionStratumCertificate(
                stratum_id=stratum_id,
                stratum_kind="equality_root",
                interval=(root_lower, root_upper),
                value_interval=value.as_tuple(),
                sign=0,
                derivative_interval=derivative_value.as_tuple(),
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        source_id = f"{stratum_id}:leaf"
        source_leaves.append(
            BranchEventTreeLeafCertificate(
                leaf_id=source_id,
                leaf_type="polynomial_equality_root_leaf",
                decision="simultaneous_event_equality",
                source_type="PolynomialDecisionStratification",
                depth=0,
                certified=False,
                missing_obligations=(),
            )
        )
        equality = EqualityStratumCertificate(
            stratum_id=stratum_id,
            defining_function_ids=(decision_id,),
            leaf_kind="simultaneous_event_equality",
            isolation_certified=certified,
            resolution_policy=str(equality_resolution_policy),
            certified=certified,
            missing_obligations=tuple(missing),
        )
        tie = EventOrderTieLeafCertificate(
            leaf_id=source_id,
            tied_event_ids=("polynomial_negative_side", "polynomial_positive_side"),
            equality_stratum=equality,
            certified=certified,
            missing_obligations=tuple(missing),
        )
        stratified_leaves.append(
            StratifiedBranchLeafCertificate(
                leaf_id=f"stratified:{source_id}",
                source_leaf_id=source_id,
                leaf_kind="simultaneous_event_equality",
                terminal_response_kind="recursive_equality_stratum",
                terminal_response_certified=False,
                source_leaf_certified=True,
                equality_stratum=equality,
                event_order_tie=tie,
                missing_obligations=(),
            )
        )
        cover_cursor = root_upper
    add_sign_cell(len(sorted_roots), cover_cursor, upper)

    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="PolynomialDecisionStratification",
        leaf_certificates=tuple(source_leaves),
        cover_certified=domain_ok and _strata_cover_domain(strata, lower, upper),
        leaf_decisions_certified=bool(source_leaves and all(leaf.certified for leaf in source_leaves if "equality" not in leaf.leaf_type)),
        equality_strata_explicit=bool(sorted_roots),
        obligations=(
            BranchEventTreeObligation(
                obligation="polynomial_decision_domain_valid",
                certified=domain_ok,
                detail=f"domain={(lower, upper)!r}; root_brackets={sorted_roots!r}",
            ),
            BranchEventTreeObligation(
                obligation="polynomial_decision_strata_cover_domain",
                certified=_strata_cover_domain(strata, lower, upper),
                detail=f"stratum_count={len(strata)}",
            ),
            BranchEventTreeObligation(
                obligation="polynomial_decision_equality_strata_explicit",
                certified=True,
                detail=f"equality_stratum_count={sum(stratum.equality for stratum in strata)}",
                required=False,
            ),
        ),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=tuple(stratified_leaves),
    )
    return PolynomialDecisionStratificationCertificate(
        decision_id=decision_id,
        coefficients=coeffs,
        domain=(lower, upper),
        root_brackets=sorted_roots,
        strata=tuple(strata),
        source_tree=source_tree,
        stratified_tree=stratified,
        statement=(
            "A one-dimensional polynomial decision function with certified "
            "simple root brackets induces a finite stratified branch/event "
            "tree whose sign cells are positive-margin leaves and whose root "
            "brackets are explicit equality strata."
        ),
        proof_sketch=(
            "Sort the verified simple root brackets inside the compact domain. "
            "Interval evaluation proves each open cell between consecutive "
            "root brackets has a strict polynomial sign, hence a stable branch "
            "or event-order decision.  On each root bracket the value interval "
            "contains zero and the derivative interval has a fixed nonzero "
            "sign, so the bracket is an isolated equality stratum.  The union "
            "of sign cells and equality brackets covers the compact domain, "
            "and equality cells remain explicit leaves for a lower-dimensional "
            "recursive theorem rather than being certified as terminal leaves."
        ),
    )


def certify_polynomial_decision_recursive_consumption(
    stratification: PolynomialDecisionStratificationCertificate,
    *,
    root_dimension: int,
    root_rank: int | None = None,
    child_consumptions: Mapping[
        str,
        RecursiveStratifiedBranchEventConsumptionCertificate,
    ]
    | tuple[
        tuple[str, RecursiveStratifiedBranchEventConsumptionCertificate],
        ...,
    ] = (),
    recursion_kind: str = "polynomial_decision_stratification",
) -> RecursiveStratifiedBranchEventConsumptionCertificate:
    """Consume a single polynomial decision stratification by recursive roots."""

    if not isinstance(stratification, PolynomialDecisionStratificationCertificate):
        raise TypeError(
            "stratification must be a "
            "PolynomialDecisionStratificationCertificate"
        )
    child_by_leaf = _normalize_arrangement_child_consumptions(
        stratification,  # type: ignore[arg-type]
        child_consumptions,
    )
    if (
        not child_by_leaf
        and stratification.proof_certified
        and stratification.equality_stratum_count
    ):
        child_by_leaf = _normalize_arrangement_child_consumptions(
            stratification,  # type: ignore[arg-type]
            derive_polynomial_decision_child_consumptions(
                stratification,
                child_root_rank=0,
            ),
        )
    return certify_recursive_stratified_branch_event_consumption(
        stratification.stratified_tree,
        root_dimension=root_dimension,
        root_rank=root_rank,
        child_consumptions=child_by_leaf,
        recursion_kind=recursion_kind,
    )


def derive_polynomial_decision_child_consumptions(
    stratification: PolynomialDecisionStratificationCertificate,
    *,
    child_root_rank: int | None = 0,
) -> dict[str, RecursiveStratifiedBranchEventConsumptionCertificate]:
    """Derive terminal 0D root children for one polynomial decision."""

    if not isinstance(stratification, PolynomialDecisionStratificationCertificate):
        raise TypeError(
            "stratification must be a "
            "PolynomialDecisionStratificationCertificate"
        )
    if not stratification.proof_certified:
        raise ValueError("polynomial decision stratification must be proof-certified")

    return _derive_polynomial_root_child_consumptions_from_strata(
        strata=stratification.strata,
        source_id=stratification.decision_id,
        child_root_rank=child_root_rank,
    )


def certify_polynomial_decision_arrangement_stratified_branch_event_tree(
    *,
    arrangement_id: str,
    decision_functions: tuple[PolynomialDecisionFunctionSpec, ...],
    domain: tuple[float, float],
    equality_resolution_policy: str = "lower_dimensional_recursive_stratum",
) -> PolynomialDecisionArrangementStratificationCertificate:
    """Derive a 1D stratified tree from several polynomial discriminants."""

    arrangement_id = str(arrangement_id)
    specs = tuple(
        PolynomialDecisionFunctionSpec(
            decision_id=str(spec.decision_id),
            coefficients=tuple(float(value) for value in spec.coefficients),
            root_brackets=tuple(
                (float(left), float(right))
                for left, right in spec.root_brackets
            ),
        )
        for spec in decision_functions
    )
    lower, upper = (float(domain[0]), float(domain[1]))
    all_roots = sorted(
        (left, right, spec.decision_id)
        for spec in specs
        for left, right in spec.root_brackets
    )
    root_brackets = tuple((left, right) for left, right, _decision_id in all_roots)
    domain_ok = bool(
        arrangement_id
        and specs
        and np.isfinite(lower)
        and np.isfinite(upper)
        and lower < upper
        and all(spec.decision_id and spec.coefficients for spec in specs)
        and _root_brackets_sorted_and_inside(root_brackets, lower, upper)
    )
    strata: list[PolynomialDecisionStratumCertificate] = []
    source_leaves: list[BranchEventTreeLeafCertificate] = []
    stratified_leaves: list[StratifiedBranchLeafCertificate] = []
    cursor = lower

    def add_arrangement_sign_cell(index: int, left: float, right: float) -> None:
        if right <= left:
            return
        interval = FloatInterval(left, right)
        value_intervals = tuple(
            _degree_two_polynomial_value_interval(spec.coefficients, interval)
            for spec in specs
        )
        signs = tuple(interval_sign(value) for value in value_intervals)
        missing = []
        if any(sign == 0 for sign in signs):
            missing.append("arrangement_sign_cell_not_separated_from_all_boundaries")
        certified = bool(domain_ok and not missing)
        sign_label = ",".join(
            f"{spec.decision_id}:{'+' if sign > 0 else '-'}"
            for spec, sign in zip(specs, signs)
        )
        stratum_id = f"{arrangement_id}:cell:{index}"
        aggregate_interval = _aggregate_value_intervals(value_intervals)
        strata.append(
            PolynomialDecisionStratumCertificate(
                stratum_id=stratum_id,
                stratum_kind="arrangement_sign_cell",
                interval=(left, right),
                value_interval=aggregate_interval,
                sign=1 if certified else 0,
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        source_id = f"{stratum_id}:leaf"
        source_leaves.append(
            BranchEventTreeLeafCertificate(
                leaf_id=source_id,
                leaf_type="polynomial_arrangement_positive_margin_leaf",
                decision=sign_label if certified else "undecided_sign_vector",
                source_type="PolynomialDecisionArrangement",
                depth=0,
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        stratified_leaves.append(
            StratifiedBranchLeafCertificate(
                leaf_id=f"stratified:{source_id}",
                source_leaf_id=source_id,
                leaf_kind="positive_margin_unique_event",
                terminal_response_kind="polynomial_arrangement_sign_decision",
                terminal_response_certified=certified,
                source_leaf_certified=certified,
                decision_functions=tuple(
                    AnalyticDecisionFunctionCertificate(
                        function_id=f"{stratum_id}:{spec.decision_id}",
                        function_kind="polynomial_arrangement_decision_sign",
                        margin_lower_bound=_strict_sign_margin(value)
                        if sign != 0
                        else 0.0,
                        lipschitz_bound=_polynomial_derivative_lipschitz_bound(
                            interval_polyder(np.asarray(spec.coefficients, dtype=float)),
                            interval,
                        ),
                        certified=certified and sign != 0,
                        missing_obligations=tuple(missing),
                    )
                    for spec, value, sign in zip(specs, value_intervals, signs)
                ),
                missing_obligations=tuple(missing),
            )
        )

    for index, (root_lower, root_upper, decision_id) in enumerate(all_roots):
        add_arrangement_sign_cell(index, cursor, root_lower)
        spec = next(spec for spec in specs if spec.decision_id == decision_id)
        root_interval = FloatInterval(root_lower, root_upper)
        value = interval_polynomial_eval(np.asarray(spec.coefficients, dtype=float), root_interval)
        derivative = interval_polyder(np.asarray(spec.coefficients, dtype=float))
        derivative_value = interval_polynomial_eval(derivative, root_interval) if derivative else FloatInterval.point(0.0)
        missing = []
        if interval_sign(value) != 0:
            missing.append("root_bracket_value_does_not_contain_zero")
        if interval_sign(derivative_value) == 0:
            missing.append("root_bracket_derivative_sign_not_isolated")
        certified = bool(domain_ok and not missing)
        stratum_id = f"{arrangement_id}:root:{index}:{decision_id}"
        strata.append(
            PolynomialDecisionStratumCertificate(
                stratum_id=stratum_id,
                stratum_kind="equality_root",
                interval=(root_lower, root_upper),
                value_interval=value.as_tuple(),
                sign=0,
                derivative_interval=derivative_value.as_tuple(),
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        source_id = f"{stratum_id}:leaf"
        source_leaves.append(
            BranchEventTreeLeafCertificate(
                leaf_id=source_id,
                leaf_type="polynomial_arrangement_equality_root_leaf",
                decision="simultaneous_event_equality",
                source_type="PolynomialDecisionArrangement",
                depth=0,
                certified=False,
                missing_obligations=(),
            )
        )
        equality = EqualityStratumCertificate(
            stratum_id=stratum_id,
            defining_function_ids=(decision_id,),
            leaf_kind="simultaneous_event_equality",
            isolation_certified=certified,
            resolution_policy=str(equality_resolution_policy),
            certified=certified,
            missing_obligations=tuple(missing),
        )
        tie = EventOrderTieLeafCertificate(
            leaf_id=source_id,
            tied_event_ids=(
                f"{decision_id}:negative_side",
                f"{decision_id}:positive_side",
            ),
            equality_stratum=equality,
            certified=certified,
            missing_obligations=tuple(missing),
        )
        stratified_leaves.append(
            StratifiedBranchLeafCertificate(
                leaf_id=f"stratified:{source_id}",
                source_leaf_id=source_id,
                leaf_kind="simultaneous_event_equality",
                terminal_response_kind="recursive_equality_stratum",
                terminal_response_certified=False,
                source_leaf_certified=True,
                equality_stratum=equality,
                event_order_tie=tie,
            )
        )
        cursor = root_upper
    add_arrangement_sign_cell(len(all_roots), cursor, upper)

    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="PolynomialDecisionArrangement",
        leaf_certificates=tuple(source_leaves),
        cover_certified=domain_ok and _strata_cover_domain(strata, lower, upper),
        leaf_decisions_certified=bool(
            source_leaves
            and all(
                leaf.certified
                for leaf in source_leaves
                if "equality" not in leaf.leaf_type
            )
        ),
        equality_strata_explicit=bool(all_roots),
        obligations=(
            BranchEventTreeObligation(
                obligation="polynomial_arrangement_domain_valid",
                certified=domain_ok,
                detail=f"domain={(lower, upper)!r}; root_brackets={root_brackets!r}",
            ),
            BranchEventTreeObligation(
                obligation="polynomial_arrangement_strata_cover_domain",
                certified=_strata_cover_domain(strata, lower, upper),
                detail=f"stratum_count={len(strata)}",
            ),
            BranchEventTreeObligation(
                obligation="polynomial_arrangement_equality_strata_explicit",
                certified=True,
                detail=f"equality_stratum_count={sum(stratum.equality for stratum in strata)}",
                required=False,
            ),
        ),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=tuple(stratified_leaves),
    )
    return PolynomialDecisionArrangementStratificationCertificate(
        arrangement_id=arrangement_id,
        decision_functions=specs,
        domain=(lower, upper),
        strata=tuple(strata),
        source_tree=source_tree,
        stratified_tree=stratified,
        statement=(
            "A finite arrangement of one-dimensional polynomial decision "
            "functions with certified simple-root brackets induces a finite "
            "stratified branch/event tree by sign vectors and explicit equality "
            "root strata."
        ),
        proof_sketch=(
            "Collect and sort every verified root bracket from the supplied "
            "polynomial discriminants.  On each cell between consecutive root "
            "brackets, interval evaluation proves every discriminator has a "
            "strict sign, so the event-order/branch sign vector is stable on "
            "that cell.  Each root bracket is checked by value containment and "
            "a nonzero derivative interval for its defining polynomial, making "
            "it an isolated equality stratum.  The finite ordered list of sign "
            "cells and root brackets covers the compact parameter interval, "
            "and equality strata remain nonterminal leaves for recursive "
            "lower-dimensional consumption."
        ),
    )


def certify_affine_decision_arrangement_stratified_branch_event_tree(
    *,
    arrangement_id: str,
    decision_functions: tuple[PolynomialDecisionFunctionSpec, ...],
    domain: tuple[float, float],
    equality_resolution_policy: str = "lower_dimensional_recursive_stratum",
) -> PolynomialDecisionArrangementStratificationCertificate:
    """Derive a one-dimensional affine arrangement without supplied roots.

    This is a deliberately narrow constructor for the first nontrivial
    branch/equality partition theorem.  For affine discriminants
    ``a_0 + a_1 x`` on a compact interval, every interior equality stratum is
    the single point ``-a_0/a_1``.  When these roots are separated and interior,
    the constructor builds disjoint brackets from the root spacings and then
    feeds those brackets to the interval-checked polynomial arrangement
    constructor.  Boundary roots receive one-sided brackets inside the compact
    interval.  When several affine decisions have the same separated root, they
    become one simultaneous equality stratum with multiple defining functions.
    """

    lower, upper = (float(domain[0]), float(domain[1]))
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        raise ValueError("affine decision arrangement requires a compact interval")
    specs = tuple(
        PolynomialDecisionFunctionSpec(
            decision_id=str(spec.decision_id),
            coefficients=tuple(float(value) for value in spec.coefficients),
            root_brackets=(),
        )
        for spec in decision_functions
    )
    if not specs:
        raise ValueError("affine decision arrangement requires at least one decision")

    affine_roots: list[tuple[float, str]] = []
    for spec in specs:
        if len(spec.coefficients) != 2:
            raise ValueError(
                "affine decision arrangement accepts only degree-one coefficients"
            )
        constant, slope = spec.coefficients
        if not spec.decision_id:
            raise ValueError("affine decision functions require ids")
        if not np.isfinite(constant) or not np.isfinite(slope) or slope == 0.0:
            raise ValueError(
                "affine decision functions require finite nonzero slope"
            )
        root = -constant / slope
        if lower <= root <= upper:
            affine_roots.append((float(root), spec.decision_id))

    root_groups = _group_affine_roots(affine_roots)
    root_values = tuple(root for root, _decision_ids in root_groups)

    bracket_by_decision: dict[str, list[tuple[float, float]]] = {
        spec.decision_id: [] for spec in specs
    }
    root_brackets: list[tuple[float, float]] = []
    for index, (root, decision_ids) in enumerate(root_groups):
        previous_boundary = lower if index == 0 else root_values[index - 1]
        next_boundary = upper if index + 1 == len(root_values) else root_values[index + 1]
        if _affine_root_at_lower_boundary(root, lower):
            gap = next_boundary - root
            left = lower
            right = lower + 0.25 * gap
        elif _affine_root_at_upper_boundary(root, upper):
            gap = root - previous_boundary
            left = upper - 0.25 * gap
            right = upper
        else:
            gap = min(root - previous_boundary, next_boundary - root)
            left = root - 0.25 * gap
            right = root + 0.25 * gap
        if gap <= 0.0 or not np.isfinite(gap):
            raise ValueError("affine root brackets require separated roots")
        bracket = (float(left), float(right))
        root_brackets.append(bracket)
        for decision_id in decision_ids:
            bracket_by_decision[decision_id].append(bracket)

    if any(len(decision_ids) > 1 for _root, decision_ids in root_groups):
        arrangement = _certify_grouped_polynomial_decision_arrangement(
            arrangement_id=arrangement_id,
            specs=specs,
            domain=(lower, upper),
            root_groups=root_groups,
            root_brackets=tuple(root_brackets),
            equality_resolution_policy=equality_resolution_policy,
        )
        arrangement = _retag_polynomial_arrangement_source(
            arrangement,
            source_type="AffineDecisionArrangement",
        )
        return replace(
            arrangement,
            statement=(
                "A finite one-dimensional arrangement of affine decision "
                "functions induces a finite stratified branch/event tree "
                "without supplying root brackets.  Separated roots "
                "are computed from coefficients, coincident roots "
                "are grouped into simultaneous equality strata with all "
                "defining affine discriminants recorded, boundary roots "
                "receive one-sided brackets, sign cells are positive-margin "
                "leaves, and equality brackets are recursive strata."
            ),
            proof_sketch=(
                "For each affine discriminator a0+a1*x with a1 nonzero, the "
                "only possible equality point is -a0/a1.  Interior equality "
                "points and boundary equality points are grouped when they "
                "coincide to floating precision, "
                "then sorted by root value.  Separation from neighboring root "
                "groups and the domain boundary gives positive gaps, so "
                "one-quarter-gap brackets, one-sided at domain boundaries, "
                "are disjoint and remain inside the compact interval.  "
                "Interval evaluation proves strict sign vectors on "
                "complementary cells; on a simultaneous bracket, "
                "each defining affine polynomial contains zero and has "
                "nonzero derivative, so the grouped leaf is an isolated "
                "higher-codimension equality stratum for recursive descent.  "
                "No root brackets are supplied by the caller."
            ),
        )

    bracketed_specs = tuple(
        PolynomialDecisionFunctionSpec(
            decision_id=spec.decision_id,
            coefficients=spec.coefficients,
            root_brackets=tuple(bracket_by_decision[spec.decision_id]),
        )
        for spec in specs
    )
    arrangement = certify_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id=arrangement_id,
        decision_functions=bracketed_specs,
        domain=(lower, upper),
        equality_resolution_policy=equality_resolution_policy,
    )
    arrangement = _retag_polynomial_arrangement_source(
        arrangement,
        source_type="AffineDecisionArrangement",
    )
    return replace(
        arrangement,
        statement=(
            "A finite one-dimensional arrangement of affine decision "
            "functions induces a finite stratified branch/event tree without "
            "supplying root brackets: every affine root in the compact domain "
            "is computed from its coefficients, separated roots receive "
            "disjoint brackets, boundary roots receive one-sided brackets, "
            "sign cells are positive-margin leaves, and root brackets are "
            "explicit equality strata."
        ),
        proof_sketch=(
            "For each affine discriminator a0+a1*x with a1 nonzero, the only "
            "possible equality point is -a0/a1.  Roots in the compact domain "
            "are sorted; "
            "separation from neighboring roots and domain boundaries gives a "
            "positive gap, so one-quarter-gap brackets, one-sided at domain "
            "boundaries, are disjoint and remain inside the compact interval.  "
            "The existing polynomial-arrangement "
            "checker then proves strict sign on the complementary cells, value "
            "containment at each root bracket, and nonzero derivative on each "
            "bracket."
        ),
    )


def _retag_polynomial_arrangement_source(
    arrangement: PolynomialDecisionArrangementStratificationCertificate,
    *,
    source_type: str,
) -> PolynomialDecisionArrangementStratificationCertificate:
    source_leaves = tuple(
        replace(leaf, source_type=source_type)
        for leaf in arrangement.source_tree.leaf_certificates
    )
    source_tree = replace(
        arrangement.source_tree,
        source_type=source_type,
        leaf_certificates=source_leaves,
    )
    stratified_tree = replace(arrangement.stratified_tree, source_tree=source_tree)
    return replace(
        arrangement,
        source_tree=source_tree,
        stratified_tree=stratified_tree,
    )


def certify_quadratic_decision_arrangement_stratified_branch_event_tree(
    *,
    arrangement_id: str,
    decision_functions: tuple[PolynomialDecisionFunctionSpec, ...],
    domain: tuple[float, float],
    equality_resolution_policy: str = "lower_dimensional_recursive_stratum",
) -> PolynomialDecisionArrangementStratificationCertificate:
    """Derive a one-dimensional quadratic arrangement without supplied roots.

    This constructor covers the next algebraic step beyond affine decisions:
    every degree-two discriminator contributes its real roots in the compact
    domain.  Simple-root brackets are derived from separation between roots
    and domain boundaries and are then rechecked by the existing interval
    polynomial arrangement constructor.  A single quadratic double root is
    represented as a tangent equality stratum with nonzero second derivative.
    Coincident simple roots across different decisions are grouped into one
    simultaneous equality stratum; mixed simple/double coincidences and
    coincident double roots are grouped into simultaneous multiple equality
    strata.
    """

    lower, upper = (float(domain[0]), float(domain[1]))
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        raise ValueError("quadratic decision arrangement requires a compact interval")
    specs = tuple(
        PolynomialDecisionFunctionSpec(
            decision_id=str(spec.decision_id),
            coefficients=tuple(float(value) for value in spec.coefficients),
            root_brackets=(),
        )
        for spec in decision_functions
    )
    if not specs:
        raise ValueError("quadratic decision arrangement requires at least one decision")

    root_items: list[tuple[float, str]] = []
    double_root_items: list[tuple[float, str]] = []
    for spec in specs:
        if not spec.decision_id:
            raise ValueError("quadratic decision functions require ids")
        if len(spec.coefficients) not in {2, 3}:
            raise ValueError(
                "quadratic decision arrangement accepts affine or degree-two coefficients"
            )
        double_root = _quadratic_double_root_in_domain(spec, (lower, upper))
        if double_root is not None:
            double_root_items.append((double_root, spec.decision_id))
            continue
        for root in _quadratic_or_affine_roots(spec):
            if lower <= root <= upper:
                root_items.append((float(root), spec.decision_id))

    if double_root_items:
        if len(double_root_items) == 1 and not root_items and len(specs) == 1:
            return _certify_quadratic_double_root_arrangement(
                arrangement_id=arrangement_id,
                spec=specs[0],
                domain=(lower, upper),
                root=double_root_items[0][0],
                equality_resolution_policy=equality_resolution_policy,
            )
        root_groups_with_multiplicity = _group_quadratic_roots_with_multiplicity(
            simple_roots=root_items,
            double_roots=double_root_items,
        )
        root_values = tuple(root for root, _entries in root_groups_with_multiplicity)
        root_brackets = _root_group_brackets(
            root_values,
            domain=(lower, upper),
            error_label="quadratic multiple-root brackets require separated root groups",
        )
        arrangement = _certify_grouped_polynomial_decision_arrangement(
            arrangement_id=arrangement_id,
            specs=specs,
            domain=(lower, upper),
            root_groups=tuple(
                (root, tuple(decision_id for decision_id, _multiplicity in entries))
                for root, entries in root_groups_with_multiplicity
            ),
            root_brackets=root_brackets,
            equality_resolution_policy=equality_resolution_policy,
            root_multiplicities=tuple(
                tuple(multiplicity for _decision_id, multiplicity in entries)
                for _root, entries in root_groups_with_multiplicity
            ),
        )
        return replace(
            arrangement,
            statement=(
                "A finite one-dimensional arrangement of affine/quadratic "
                "decision functions induces a finite stratified branch/event "
                "tree without supplied root brackets.  Real simple roots and "
                "quadratic double roots are computed from coefficients; mixed "
                "simple/double coincidences and coincident double roots are "
                "preserved as simultaneous multiple equality strata with all "
                "defining discriminants recorded."
            ),
            proof_sketch=(
                "Each affine root, quadratic simple root, and quadratic double "
                "root is computed from the coefficient formulas and retained "
                "only inside the compact domain.  Equal roots are grouped before "
                "bracket construction.  Separation between root groups and "
                "domain endpoints gives disjoint one-quarter-gap brackets.  On "
                "each multiple equality bracket, interval evaluation verifies "
                "zero containment for every defining polynomial; simple members "
                "have nonzero first derivative, double members have first "
                "derivative containing zero and nonzero second derivative, and "
                "nondefining polynomials keep strict sign.  The grouped root is "
                "therefore an explicit lower-dimensional multiple equality "
                "stratum for recursive descent."
            ),
        )

    root_groups = _group_affine_roots(root_items)
    if any(len(decision_ids) > 1 for _root, decision_ids in root_groups):
        root_values = tuple(root for root, _decision_ids in root_groups)
        root_brackets = _root_group_brackets(
            root_values,
            domain=(lower, upper),
            error_label="coincident quadratic root brackets require separated root groups",
        )
        arrangement = _certify_grouped_polynomial_decision_arrangement(
            arrangement_id=arrangement_id,
            specs=specs,
            domain=(lower, upper),
            root_groups=root_groups,
            root_brackets=tuple(root_brackets),
            equality_resolution_policy=equality_resolution_policy,
        )
        return replace(
            arrangement,
            statement=(
                "A finite one-dimensional arrangement of affine/quadratic "
                "decision functions induces a finite stratified branch/event "
                "tree without supplied root brackets.  Real simple roots are "
                "computed from coefficients; coincident simple roots across "
                "different decisions are grouped into one simultaneous "
                "equality stratum with all defining discriminants recorded; "
                "remaining sign cells are positive-margin leaves."
            ),
            proof_sketch=(
                "Affine roots and quadratic simple roots are computed from "
                "closed formulas and retained only inside the compact domain.  "
                "Equal roots are grouped before bracket construction.  "
                "Separation between root groups and boundary endpoints gives "
                "disjoint one-quarter-gap brackets.  On each grouped equality "
                "bracket, interval evaluation verifies every defining "
                "polynomial contains zero and every defining derivative "
                "excludes zero, while nondefining polynomials keep strict "
                "sign.  Thus the coincident simple root is an isolated "
                "higher-codimension equality stratum for recursive descent."
            ),
        )
    root_values = tuple(root for root, _decision_ids in root_groups)
    bracket_by_decision: dict[str, list[tuple[float, float]]] = {
        spec.decision_id: [] for spec in specs
    }
    for index, (root, decision_ids) in enumerate(root_groups):
        previous_boundary = lower if index == 0 else root_values[index - 1]
        next_boundary = upper if index + 1 == len(root_values) else root_values[index + 1]
        if _affine_root_at_lower_boundary(root, lower):
            gap = next_boundary - root
            left = lower
            right = lower + 0.25 * gap
        elif _affine_root_at_upper_boundary(root, upper):
            gap = root - previous_boundary
            left = upper - 0.25 * gap
            right = upper
        else:
            gap = min(root - previous_boundary, next_boundary - root)
            left = root - 0.25 * gap
            right = root + 0.25 * gap
        if gap <= 0.0 or not np.isfinite(gap):
            raise ValueError("quadratic root brackets require separated simple roots")
        bracket_by_decision[decision_ids[0]].append((float(left), float(right)))

    bracketed_specs = tuple(
        PolynomialDecisionFunctionSpec(
            decision_id=spec.decision_id,
            coefficients=spec.coefficients,
            root_brackets=tuple(bracket_by_decision[spec.decision_id]),
        )
        for spec in specs
    )
    arrangement = certify_polynomial_decision_arrangement_stratified_branch_event_tree(
        arrangement_id=arrangement_id,
        decision_functions=bracketed_specs,
        domain=(lower, upper),
        equality_resolution_policy=equality_resolution_policy,
    )
    return replace(
        arrangement,
        statement=(
            "A finite one-dimensional arrangement of affine/quadratic decision "
            "functions induces a finite stratified branch/event tree without "
            "supplying root brackets: all real simple roots in the compact "
            "domain are computed from coefficients, separated roots receive "
            "disjoint brackets, boundary roots receive one-sided brackets, "
            "sign cells are positive-margin leaves, and root brackets are "
            "explicit recursive equality strata."
        ),
        proof_sketch=(
            "Affine roots are given by -a0/a1, and quadratic simple roots are "
            "given by the quadratic formula when the discriminant is positive. "
            "Only roots lying in the compact parameter domain are retained.  "
            "The retained roots are sorted and must be separated from each "
            "other up to the grouping tolerance; one-quarter-gap brackets, "
            "one-sided at domain boundaries, are disjoint and remain inside "
            "the compact interval.  The interval polynomial arrangement "
            "checker then proves strict sign on complementary cells, value "
            "containment at each computed root bracket, and nonzero derivative "
            "on every certified equality bracket.  Coincident simple roots "
            "across decisions are not certified by this simple-root theorem."
        ),
    )


def certify_sturm_polynomial_decision_arrangement_stratified_branch_event_tree(
    *,
    arrangement_id: str,
    decision_functions: tuple[PolynomialDecisionFunctionSpec, ...],
    domain: tuple[float, float],
    equality_resolution_policy: str = "lower_dimensional_recursive_stratum",
    max_bisection_depth: int = 96,
) -> PolynomialDecisionArrangementStratificationCertificate:
    """Derive a polynomial arrangement by exact Sturm isolation.

    This is the first higher-degree root-construction theorem in the
    branch/event-order program.  It rationalizes the supplied finite
    coefficients, isolates distinct roots of the exact square-free part with
    the Sturm variation theorem, recovers each root multiplicity by exact
    gcd/derivative evidence, groups coincident roots across discriminants when
    exact gcd evidence proves a common root, and then feeds those strata to the
    existing interval-checked arrangement/recursive-consumption pipeline.
    Multiple roots are preserved as explicit lower-dimensional equality strata
    rather than being hulled away.
    """

    lower, upper = (float(domain[0]), float(domain[1]))
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        raise ValueError("Sturm polynomial arrangement requires a compact interval")
    if max_bisection_depth <= 0:
        raise ValueError("Sturm polynomial arrangement requires positive bisection depth")
    specs = tuple(
        PolynomialDecisionFunctionSpec(
            decision_id=str(spec.decision_id),
            coefficients=tuple(float(value) for value in spec.coefficients),
            root_brackets=(),
        )
        for spec in decision_functions
    )
    if not specs:
        raise ValueError("Sturm polynomial arrangement requires at least one decision")

    bracketed_specs: list[PolynomialDecisionFunctionSpec] = []
    root_multiplicities_by_decision: dict[str, tuple[int, ...]] = {}
    for spec in specs:
        if not spec.decision_id:
            raise ValueError("Sturm polynomial decision functions require ids")
        occurrences = _sturm_isolate_polynomial_root_occurrences(
            spec.coefficients,
            domain=(lower, upper),
            max_bisection_depth=max_bisection_depth,
        )
        brackets = tuple(
            (left, right) for left, right, _multiplicity in occurrences
        )
        bracketed_specs.append(
            PolynomialDecisionFunctionSpec(
                decision_id=spec.decision_id,
                coefficients=spec.coefficients,
                root_brackets=brackets,
            )
        )
        root_multiplicities_by_decision[spec.decision_id] = tuple(
            multiplicity for _left, _right, multiplicity in occurrences
        )
    arrangement = _certify_sturm_polynomial_arrangement_from_brackets(
        arrangement_id=arrangement_id,
        specs=tuple(bracketed_specs),
        domain=(lower, upper),
        equality_resolution_policy=equality_resolution_policy,
        root_multiplicities_by_decision=root_multiplicities_by_decision,
    )
    return replace(
        arrangement,
        statement=(
            "A finite one-dimensional arrangement of polynomial "
            "decision functions induces a finite stratified branch/event tree "
            "without supplied root brackets.  Exact rational Sturm sequences "
            "on square-free parts derive isolating brackets for every distinct "
            "root in the compact domain; exact derivative/gcd evidence records "
            "multiplicity; exact common-root gcd evidence groups coincident "
            "roots across decisions; sign cells become positive-margin leaves "
            "and each root bracket becomes an explicit recursive equality "
            "stratum."
        ),
        proof_sketch=(
            "Each supplied discriminator is rationalized and divided by "
            "gcd(p,p') to obtain the square-free part with the same distinct "
            "root set.  Sturm variation counts isolate all distinct real roots "
            "on the compact interval by recursive bisection; boundary roots "
            "are handled by exact linear deflation of the square-free part and "
            "one-sided brackets.  For each isolated root, gcds of the "
            "square-free part with successive derivatives of the original "
            "polynomial identify the first nonvanishing derivative order, "
            "hence the multiplicity.  Interval evaluation then checks value "
            "containment, vanishing of lower derivatives, and nonzero leading "
            "derivative on the bracket.  Overlapping root brackets from "
            "different decisions are grouped only when an exact rational gcd "
            "has a root in the overlap; otherwise the constructor rejects the "
            "ambiguous ordering.  The Sturm arrangement checker then certifies "
            "strict sign vectors on complementary cells and finite coverage of "
            "the compact domain."
        ),
    )


def _group_affine_roots(
    affine_roots: list[tuple[float, str]],
) -> tuple[tuple[float, tuple[str, ...]], ...]:
    sorted_roots = sorted(affine_roots)
    groups: list[tuple[list[float], list[str]]] = []
    for root, decision_id in sorted_roots:
        if not groups:
            groups.append(([root], [decision_id]))
            continue
        representative = float(sum(groups[-1][0]) / len(groups[-1][0]))
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(representative), abs(root))
        if abs(root - representative) <= tolerance:
            groups[-1][0].append(root)
            groups[-1][1].append(decision_id)
        else:
            groups.append(([root], [decision_id]))
    return tuple(
        (
            float(sum(roots) / len(roots)),
            tuple(decision_ids),
        )
        for roots, decision_ids in groups
    )


def _group_quadratic_roots_with_multiplicity(
    *,
    simple_roots: list[tuple[float, str]],
    double_roots: list[tuple[float, str]],
) -> tuple[tuple[float, tuple[tuple[str, int], ...]], ...]:
    entries = [
        (float(root), str(decision_id), 1)
        for root, decision_id in simple_roots
    ] + [
        (float(root), str(decision_id), 2)
        for root, decision_id in double_roots
    ]
    sorted_entries = sorted(entries)
    groups: list[tuple[list[float], list[tuple[str, int]]]] = []
    for root, decision_id, multiplicity in sorted_entries:
        if multiplicity not in {1, 2}:
            raise ValueError("quadratic root multiplicity must be one or two")
        if not groups:
            groups.append(([root], [(decision_id, multiplicity)]))
            continue
        representative = float(sum(groups[-1][0]) / len(groups[-1][0]))
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(representative), abs(root))
        if abs(root - representative) <= tolerance:
            groups[-1][0].append(root)
            groups[-1][1].append((decision_id, multiplicity))
        else:
            groups.append(([root], [(decision_id, multiplicity)]))
    return tuple(
        (
            float(sum(roots) / len(roots)),
            tuple(sorted(entries, key=lambda item: item[0])),
        )
        for roots, entries in groups
    )


def _root_group_brackets(
    root_values: tuple[float, ...],
    *,
    domain: tuple[float, float],
    error_label: str,
) -> tuple[tuple[float, float], ...]:
    lower, upper = domain
    root_brackets: list[tuple[float, float]] = []
    for index, root in enumerate(root_values):
        previous_boundary = lower if index == 0 else root_values[index - 1]
        next_boundary = upper if index + 1 == len(root_values) else root_values[index + 1]
        if _affine_root_at_lower_boundary(root, lower):
            gap = next_boundary - root
            bracket = (lower, lower + 0.25 * gap)
        elif _affine_root_at_upper_boundary(root, upper):
            gap = root - previous_boundary
            bracket = (upper - 0.25 * gap, upper)
        else:
            gap = min(root - previous_boundary, next_boundary - root)
            bracket = (root - 0.25 * gap, root + 0.25 * gap)
        if gap <= 0.0 or not np.isfinite(gap):
            raise ValueError(error_label)
        root_brackets.append((float(bracket[0]), float(bracket[1])))
    return tuple(root_brackets)


def _affine_root_at_lower_boundary(root: float, lower: float) -> bool:
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(root), abs(lower))
    return abs(root - lower) <= tolerance


def _affine_root_at_upper_boundary(root: float, upper: float) -> bool:
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(root), abs(upper))
    return abs(root - upper) <= tolerance


def _quadratic_or_affine_roots(
    spec: PolynomialDecisionFunctionSpec,
) -> tuple[float, ...]:
    coefficients = tuple(float(value) for value in spec.coefficients)
    if any(not np.isfinite(value) for value in coefficients):
        raise ValueError("quadratic decision coefficients must be finite")
    if len(coefficients) == 2:
        constant, slope = coefficients
        if slope == 0.0:
            raise ValueError("affine quadratic-family decision requires nonzero slope")
        return (float(-constant / slope),)
    constant, linear, quadratic = coefficients
    if quadratic == 0.0:
        if linear == 0.0:
            raise ValueError(
                "quadratic decision with zero quadratic and linear terms is unsupported"
            )
        return (float(-constant / linear),)
    discriminant = linear * linear - 4.0 * quadratic * constant
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0,
        abs(linear * linear),
        abs(4.0 * quadratic * constant),
    )
    if discriminant < -tolerance:
        return ()
    if abs(discriminant) <= tolerance:
        raise ValueError(
            "quadratic double roots require a multiple-root stratum constructor"
        )
    sqrt_discriminant = float(np.sqrt(discriminant))
    roots = (
        (-linear - sqrt_discriminant) / (2.0 * quadratic),
        (-linear + sqrt_discriminant) / (2.0 * quadratic),
    )
    return tuple(sorted(float(root) for root in roots))


def _quadratic_double_root_in_domain(
    spec: PolynomialDecisionFunctionSpec,
    domain: tuple[float, float],
) -> float | None:
    coefficients = tuple(float(value) for value in spec.coefficients)
    if len(coefficients) != 3:
        return None
    constant, linear, quadratic = coefficients
    if quadratic == 0.0:
        return None
    discriminant = linear * linear - 4.0 * quadratic * constant
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0,
        abs(linear * linear),
        abs(4.0 * quadratic * constant),
    )
    if abs(discriminant) > tolerance:
        return None
    root = float(-linear / (2.0 * quadratic))
    lower, upper = domain
    if lower <= root <= upper:
        return root
    return None


def _certify_quadratic_double_root_arrangement(
    *,
    arrangement_id: str,
    spec: PolynomialDecisionFunctionSpec,
    domain: tuple[float, float],
    root: float,
    equality_resolution_policy: str,
) -> PolynomialDecisionArrangementStratificationCertificate:
    lower, upper = domain
    if _affine_root_at_lower_boundary(root, lower):
        gap = upper - root
        bracket = (lower, lower + 0.25 * gap)
    elif _affine_root_at_upper_boundary(root, upper):
        gap = root - lower
        bracket = (upper - 0.25 * gap, upper)
    else:
        gap = min(root - lower, upper - root)
        bracket = (root - 0.25 * gap, root + 0.25 * gap)
    if gap <= 0.0 or not np.isfinite(gap):
        raise ValueError("quadratic double-root bracket requires root separated from domain collapse")

    coeffs = np.asarray(spec.coefficients, dtype=float)
    derivative = interval_polyder(coeffs)
    second_derivative = interval_polyder(derivative)
    strata: list[PolynomialDecisionStratumCertificate] = []
    source_leaves: list[BranchEventTreeLeafCertificate] = []
    stratified_leaves: list[StratifiedBranchLeafCertificate] = []

    def add_sign_cell(index: int, left: float, right: float) -> None:
        if right <= left:
            return
        interval = FloatInterval(left, right)
        value = interval_polynomial_eval(coeffs, interval)
        sign = interval_sign(value)
        missing = []
        if sign == 0:
            missing.append("quadratic_double_sign_cell_not_separated_from_zero")
        certified = sign != 0
        stratum_id = f"{arrangement_id}:cell:{index}"
        strata.append(
            PolynomialDecisionStratumCertificate(
                stratum_id=stratum_id,
                stratum_kind="quadratic_double_sign_cell",
                interval=(left, right),
                value_interval=value.as_tuple(),
                sign=sign,
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        source_id = f"{stratum_id}:leaf"
        source_leaves.append(
            BranchEventTreeLeafCertificate(
                leaf_id=source_id,
                leaf_type="quadratic_double_positive_margin_leaf",
                decision="positive" if sign > 0 else "negative",
                source_type="QuadraticDoubleRootArrangement",
                depth=0,
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        stratified_leaves.append(
            StratifiedBranchLeafCertificate(
                leaf_id=f"stratified:{source_id}",
                source_leaf_id=source_id,
                leaf_kind="positive_margin_unique_event",
                terminal_response_kind="quadratic_double_sign_decision",
                terminal_response_certified=certified,
                source_leaf_certified=certified,
                decision_functions=(
                    AnalyticDecisionFunctionCertificate(
                        function_id=stratum_id,
                        function_kind="quadratic_double_decision_sign",
                        margin_lower_bound=_strict_sign_margin(value)
                        if sign != 0
                        else 0.0,
                        lipschitz_bound=_polynomial_derivative_lipschitz_bound(
                            derivative,
                            interval,
                        ),
                        certified=certified and sign != 0,
                        missing_obligations=tuple(missing),
                    ),
                ),
                missing_obligations=tuple(missing),
            )
        )

    add_sign_cell(0, lower, bracket[0])
    root_interval = FloatInterval(*bracket)
    value = interval_polynomial_eval(coeffs, root_interval)
    derivative_value = interval_polynomial_eval(derivative, root_interval)
    second_derivative_value = (
        interval_polynomial_eval(second_derivative, root_interval)
        if second_derivative
        else FloatInterval.point(0.0)
    )
    missing = []
    if interval_sign(value) != 0:
        missing.append("quadratic_double_root_value_does_not_contain_zero")
    if interval_sign(derivative_value) != 0:
        missing.append("quadratic_double_root_derivative_does_not_contain_zero")
    if interval_sign(second_derivative_value) == 0:
        missing.append("quadratic_double_root_second_derivative_not_isolated")
    certified = not missing
    stratum_id = f"{arrangement_id}:root:0:{spec.decision_id}"
    strata.append(
        PolynomialDecisionStratumCertificate(
            stratum_id=stratum_id,
            stratum_kind="quadratic_double_equality_root",
            interval=bracket,
            value_interval=value.as_tuple(),
            sign=0,
            derivative_interval=derivative_value.as_tuple(),
            second_derivative_interval=second_derivative_value.as_tuple(),
            certified=certified,
            missing_obligations=tuple(missing),
        )
    )
    source_id = f"{stratum_id}:leaf"
    source_leaves.append(
        BranchEventTreeLeafCertificate(
            leaf_id=source_id,
            leaf_type="quadratic_double_equality_root_leaf",
            decision="tangent_equality",
            source_type="QuadraticDoubleRootArrangement",
            depth=0,
            certified=False,
            missing_obligations=(),
        )
    )
    equality = EqualityStratumCertificate(
        stratum_id=stratum_id,
        defining_function_ids=(spec.decision_id,),
        leaf_kind="simultaneous_event_equality",
        isolation_certified=certified,
        resolution_policy=str(equality_resolution_policy),
        certified=certified,
        missing_obligations=tuple(missing),
    )
    tie = EventOrderTieLeafCertificate(
        leaf_id=source_id,
        tied_event_ids=(
            f"{spec.decision_id}:tangent_left",
            f"{spec.decision_id}:tangent_right",
        ),
        equality_stratum=equality,
        certified=certified,
        missing_obligations=tuple(missing),
    )
    stratified_leaves.append(
        StratifiedBranchLeafCertificate(
            leaf_id=f"stratified:{source_id}",
            source_leaf_id=source_id,
            leaf_kind="simultaneous_event_equality",
            terminal_response_kind="recursive_quadratic_tangent_equality_stratum",
            terminal_response_certified=False,
            source_leaf_certified=True,
            equality_stratum=equality,
            event_order_tie=tie,
        )
    )
    add_sign_cell(1, bracket[1], upper)

    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="QuadraticDoubleRootArrangement",
        leaf_certificates=tuple(source_leaves),
        cover_certified=_strata_cover_domain(strata, lower, upper),
        leaf_decisions_certified=bool(
            source_leaves
            and all(
                leaf.certified
                for leaf in source_leaves
                if "equality" not in leaf.leaf_type
            )
        ),
        equality_strata_explicit=True,
        obligations=(
            BranchEventTreeObligation(
                obligation="quadratic_double_arrangement_domain_valid",
                certified=lower < upper,
                detail=f"domain={(lower, upper)!r}; root={root!r}; bracket={bracket!r}",
            ),
            BranchEventTreeObligation(
                obligation="quadratic_double_arrangement_strata_cover_domain",
                certified=_strata_cover_domain(strata, lower, upper),
                detail=f"stratum_count={len(strata)}",
            ),
            BranchEventTreeObligation(
                obligation="quadratic_double_equality_stratum_explicit",
                certified=True,
                detail=f"stratum_id={stratum_id}",
                required=False,
            ),
        ),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=tuple(stratified_leaves),
    )
    return PolynomialDecisionArrangementStratificationCertificate(
        arrangement_id=arrangement_id,
        decision_functions=(
            PolynomialDecisionFunctionSpec(
                decision_id=spec.decision_id,
                coefficients=spec.coefficients,
                root_brackets=(bracket,),
            ),
        ),
        domain=(lower, upper),
        strata=tuple(strata),
        source_tree=source_tree,
        stratified_tree=stratified,
        statement=(
            "A one-dimensional quadratic decision function with a double root "
            "in the compact domain induces a finite stratified branch/event "
            "tree without supplied root brackets: the root is computed from "
            "the coefficients, receives a one-sided or centered tangent "
            "equality bracket, complementary cells are positive-margin leaves, "
            "and the tangent root is an explicit recursive equality stratum."
        ),
        proof_sketch=(
            "When the quadratic discriminant is zero and the quadratic "
            "coefficient is nonzero, the only root is -b/(2a) and the second "
            "derivative is the nonzero constant 2a.  A bracket inside the "
            "compact domain is chosen from the distance to the boundary.  "
            "Interval evaluation over that bracket contains zero for the "
            "polynomial and first derivative, while the second derivative "
            "interval excludes zero, so the equality is isolated as a tangent "
            "multiplicity-two stratum.  Complementary cells have strict sign "
            "and the equality leaf is consumed only by recursive descent."
        ),
    )


def _certify_grouped_polynomial_decision_arrangement(
    *,
    arrangement_id: str,
    specs: tuple[PolynomialDecisionFunctionSpec, ...],
    domain: tuple[float, float],
    root_groups: tuple[tuple[float, tuple[str, ...]], ...],
    root_brackets: tuple[tuple[float, float], ...],
    equality_resolution_policy: str,
    root_multiplicities: tuple[tuple[int, ...], ...] | None = None,
) -> PolynomialDecisionArrangementStratificationCertificate:
    lower, upper = domain
    if root_multiplicities is None:
        root_multiplicities = tuple(
            tuple(1 for _decision_id in decision_ids)
            for _root, decision_ids in root_groups
        )
    domain_ok = bool(
        arrangement_id
        and specs
        and lower < upper
        and _root_brackets_sorted_and_inside(root_brackets, lower, upper)
        and len(root_groups) == len(root_brackets)
        and len(root_groups) == len(root_multiplicities)
        and all(
            len(decision_ids) == len(multiplicities)
            and all(multiplicity in {1, 2} for multiplicity in multiplicities)
            for (_root, decision_ids), multiplicities in zip(
                root_groups,
                root_multiplicities,
            )
        )
    )
    specs_by_id = {spec.decision_id: spec for spec in specs}
    strata: list[PolynomialDecisionStratumCertificate] = []
    source_leaves: list[BranchEventTreeLeafCertificate] = []
    stratified_leaves: list[StratifiedBranchLeafCertificate] = []
    cursor = lower

    def add_sign_cell(index: int, left: float, right: float) -> None:
        if right <= left:
            return
        interval = FloatInterval(left, right)
        value_intervals = tuple(
            _degree_two_polynomial_value_interval(spec.coefficients, interval)
            for spec in specs
        )
        signs = tuple(interval_sign(value) for value in value_intervals)
        missing = []
        if any(sign == 0 for sign in signs):
            missing.append("polynomial_grouped_sign_cell_not_separated_from_all_boundaries")
        certified = bool(domain_ok and not missing)
        sign_label = ",".join(
            f"{spec.decision_id}:{'+' if sign > 0 else '-'}"
            for spec, sign in zip(specs, signs)
        )
        stratum_id = f"{arrangement_id}:cell:{index}"
        strata.append(
            PolynomialDecisionStratumCertificate(
                stratum_id=stratum_id,
                stratum_kind="polynomial_grouped_arrangement_sign_cell",
                interval=(left, right),
                value_interval=_aggregate_value_intervals(value_intervals),
                sign=1 if certified else 0,
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        source_id = f"{stratum_id}:leaf"
        source_leaves.append(
            BranchEventTreeLeafCertificate(
                leaf_id=source_id,
                leaf_type="polynomial_grouped_arrangement_positive_margin_leaf",
                decision=sign_label if certified else "undecided_sign_vector",
                source_type="PolynomialRootArrangement",
                depth=0,
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        stratified_leaves.append(
            StratifiedBranchLeafCertificate(
                leaf_id=f"stratified:{source_id}",
                source_leaf_id=source_id,
                leaf_kind="positive_margin_unique_event",
                terminal_response_kind="polynomial_grouped_arrangement_sign_decision",
                terminal_response_certified=certified,
                source_leaf_certified=certified,
                decision_functions=tuple(
                    AnalyticDecisionFunctionCertificate(
                        function_id=f"{stratum_id}:{spec.decision_id}",
                        function_kind="polynomial_arrangement_decision_sign",
                        margin_lower_bound=_strict_sign_margin(value)
                        if sign != 0
                        else 0.0,
                        lipschitz_bound=_polynomial_derivative_lipschitz_bound(
                            interval_polyder(np.asarray(spec.coefficients, dtype=float)),
                            interval,
                        ),
                        certified=certified and sign != 0,
                        missing_obligations=tuple(missing),
                    )
                    for spec, value, sign in zip(specs, value_intervals, signs)
                ),
                missing_obligations=tuple(missing),
            )
        )

    for index, (
        ((root, decision_ids), multiplicities, (root_lower, root_upper))
    ) in enumerate(
        zip(root_groups, root_multiplicities, root_brackets)
    ):
        add_sign_cell(index, cursor, root_lower)
        root_interval = FloatInterval(root_lower, root_upper)
        multiplicity_by_decision = {
            decision_id: multiplicity
            for decision_id, multiplicity in zip(decision_ids, multiplicities)
        }
        simple_decision_ids = tuple(
            decision_id
            for decision_id in decision_ids
            if multiplicity_by_decision[decision_id] == 1
        )
        double_decision_ids = tuple(
            decision_id
            for decision_id in decision_ids
            if multiplicity_by_decision[decision_id] == 2
        )
        defining_values = tuple(
            _degree_two_polynomial_value_interval(
                specs_by_id[decision_id].coefficients,
                root_interval,
            )
            for decision_id in decision_ids
        )
        simple_derivatives = tuple(
            interval_polynomial_eval(
                interval_polyder(
                    np.asarray(specs_by_id[decision_id].coefficients, dtype=float)
                ),
                root_interval,
            )
            for decision_id in simple_decision_ids
        )
        double_first_derivatives = tuple(
            interval_polynomial_eval(
                interval_polyder(
                    np.asarray(specs_by_id[decision_id].coefficients, dtype=float)
                ),
                root_interval,
            )
            for decision_id in double_decision_ids
        )
        double_second_derivatives = tuple(
            interval_polynomial_eval(
                interval_polyder(
                    interval_polyder(
                        np.asarray(specs_by_id[decision_id].coefficients, dtype=float)
                    )
                ),
                root_interval,
            )
            for decision_id in double_decision_ids
        )
        nondefining_values = tuple(
            _degree_two_polynomial_value_interval(spec.coefficients, root_interval)
            for spec in specs
            if spec.decision_id not in decision_ids
        )
        missing = []
        if any(interval_sign(value) != 0 for value in defining_values):
            missing.append("simultaneous_root_value_does_not_contain_zero")
        if any(interval_sign(value) == 0 for value in simple_derivatives):
            missing.append("simultaneous_root_derivative_sign_not_isolated")
        if any(interval_sign(value) != 0 for value in double_first_derivatives):
            missing.append("multiple_root_first_derivative_does_not_contain_zero")
        if any(interval_sign(value) == 0 for value in double_second_derivatives):
            missing.append("multiple_root_second_derivative_sign_not_isolated")
        if any(interval_sign(value) == 0 for value in nondefining_values):
            missing.append("simultaneous_polynomial_nondefining_decision_not_separated")
        certified = bool(domain_ok and not missing)
        decision_label = "+".join(decision_ids)
        stratum_id = f"{arrangement_id}:root:{index}:{decision_label}"
        has_multiple_root = bool(double_decision_ids)
        if has_multiple_root and len(decision_ids) == 1:
            stratum_kind = "quadratic_double_equality_root"
            source_leaf_type = "polynomial_arrangement_quadratic_double_equality_root_leaf"
        elif has_multiple_root:
            stratum_kind = "simultaneous_polynomial_multiple_equality_root"
            source_leaf_type = "polynomial_arrangement_simultaneous_multiple_equality_root_leaf"
        else:
            stratum_kind = "simultaneous_polynomial_equality_root"
            source_leaf_type = "polynomial_arrangement_simultaneous_equality_root_leaf"
        strata.append(
            PolynomialDecisionStratumCertificate(
                stratum_id=stratum_id,
                stratum_kind=stratum_kind,
                interval=(root_lower, root_upper),
                value_interval=_aggregate_value_intervals(defining_values),
                sign=0,
                derivative_interval=_aggregate_value_intervals(
                    simple_derivatives + double_first_derivatives
                ),
                second_derivative_interval=(
                    _aggregate_value_intervals(double_second_derivatives)
                    if double_second_derivatives
                    else None
                ),
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        source_id = f"{stratum_id}:leaf"
        source_leaves.append(
            BranchEventTreeLeafCertificate(
                leaf_id=source_id,
                leaf_type=source_leaf_type,
                decision="simultaneous_event_equality",
                source_type="PolynomialRootArrangement",
                depth=0,
                certified=False,
                missing_obligations=(),
            )
        )
        equality = EqualityStratumCertificate(
            stratum_id=stratum_id,
            defining_function_ids=tuple(decision_ids),
            leaf_kind="simultaneous_event_equality",
            isolation_certified=certified,
            resolution_policy=str(equality_resolution_policy),
            certified=certified,
            missing_obligations=tuple(missing),
        )
        tied_event_ids = tuple(
            event_id
            for decision_id in decision_ids
            for event_id in (
                f"{decision_id}:negative_side",
                f"{decision_id}:positive_side",
            )
        )
        tie = EventOrderTieLeafCertificate(
            leaf_id=source_id,
            tied_event_ids=tied_event_ids,
            equality_stratum=equality,
            certified=certified,
            missing_obligations=tuple(missing),
        )
        stratified_leaves.append(
            StratifiedBranchLeafCertificate(
                leaf_id=f"stratified:{source_id}",
                source_leaf_id=source_id,
                leaf_kind="simultaneous_event_equality",
                terminal_response_kind="recursive_simultaneous_equality_stratum",
                terminal_response_certified=False,
                source_leaf_certified=True,
                equality_stratum=equality,
                event_order_tie=tie,
            )
        )
        cursor = root_upper
    add_sign_cell(len(root_groups), cursor, upper)

    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="PolynomialRootArrangement",
        leaf_certificates=tuple(source_leaves),
        cover_certified=domain_ok and _strata_cover_domain(strata, lower, upper),
        leaf_decisions_certified=bool(
            source_leaves
            and all(
                leaf.certified
                for leaf in source_leaves
                if "equality" not in leaf.leaf_type
            )
        ),
        equality_strata_explicit=bool(root_groups),
        obligations=(
            BranchEventTreeObligation(
                obligation="polynomial_grouped_arrangement_domain_valid",
                certified=domain_ok,
                detail=f"domain={(lower, upper)!r}; root_brackets={root_brackets!r}",
            ),
            BranchEventTreeObligation(
                obligation="polynomial_grouped_arrangement_strata_cover_domain",
                certified=_strata_cover_domain(strata, lower, upper),
                detail=f"stratum_count={len(strata)}",
            ),
            BranchEventTreeObligation(
                obligation="polynomial_grouped_arrangement_equality_strata_explicit",
                certified=True,
                detail=(
                    "equality_stratum_count="
                    f"{sum(stratum.equality for stratum in strata)}"
                ),
                required=False,
            ),
        ),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=tuple(stratified_leaves),
    )
    return PolynomialDecisionArrangementStratificationCertificate(
        arrangement_id=arrangement_id,
        decision_functions=tuple(
            PolynomialDecisionFunctionSpec(
                decision_id=spec.decision_id,
                coefficients=spec.coefficients,
                root_brackets=tuple(
                    bracket
                    for (root, decision_ids), bracket in zip(root_groups, root_brackets)
                    if spec.decision_id in decision_ids
                ),
            )
            for spec in specs
        ),
        domain=(lower, upper),
        strata=tuple(strata),
        source_tree=source_tree,
        stratified_tree=stratified,
        statement="",
        proof_sketch="",
    )


def _group_sturm_root_bracket_occurrences(
    specs: tuple[PolynomialDecisionFunctionSpec, ...],
    root_multiplicities_by_decision: dict[str, tuple[int, ...]] | None = None,
) -> tuple[tuple[float, float, tuple[str, ...], tuple[int, ...]], ...]:
    root_multiplicities_by_decision = root_multiplicities_by_decision or {}
    occurrences = sorted(
        (
            float(left),
            float(right),
            spec.decision_id,
            spec.coefficients,
            root_multiplicities_by_decision.get(
                spec.decision_id,
                (1,) * len(spec.root_brackets),
            )[index],
        )
        for spec in specs
        for index, (left, right) in enumerate(spec.root_brackets)
    )
    groups: list[dict[str, object]] = []
    for left, right, decision_id, coefficients, multiplicity in occurrences:
        placed = False
        for group in groups:
            group_left = float(group["left"])
            group_right = float(group["right"])
            if right < group_left or left > group_right:
                continue
            members = group["members"]
            assert isinstance(members, list)
            if all(
                _sturm_root_bracket_occurrences_share_common_root(
                    coefficients,
                    (left, right),
                    member_coefficients,
                    member_bracket,
                )
                for _member_id, member_coefficients, member_bracket, _multiplicity in members
            ):
                members.append((decision_id, coefficients, (left, right), multiplicity))
                group["left"] = min(group_left, left)
                group["right"] = max(group_right, right)
                placed = True
                break
            raise ValueError(
                "Sturm polynomial arrangement currently requires separated "
                "root brackets unless exact gcd evidence proves a common root"
            )
        if not placed:
            groups.append(
                {
                    "left": left,
                    "right": right,
                    "members": [(decision_id, coefficients, (left, right), multiplicity)],
                }
            )
    grouped: list[tuple[float, float, tuple[str, ...], tuple[int, ...]]] = []
    for group in groups:
        members = group["members"]
        assert isinstance(members, list)
        decision_ids = tuple(
            str(member_id) for member_id, _coefficients, _bracket, _multiplicity in members
        )
        multiplicities = tuple(
            int(multiplicity)
            for _member_id, _coefficients, _bracket, multiplicity in members
        )
        grouped.append(
            (
                float(group["left"]),
                float(group["right"]),
                decision_ids,
                multiplicities,
            )
        )
    return tuple(grouped)


def _sturm_root_bracket_occurrences_share_common_root(
    coefficients_a: tuple[float, ...],
    bracket_a: tuple[float, float],
    coefficients_b: tuple[float, ...],
    bracket_b: tuple[float, float],
) -> bool:
    overlap_left = max(bracket_a[0], bracket_b[0])
    overlap_right = min(bracket_a[1], bracket_b[1])
    if overlap_left > overlap_right:
        return False
    polynomial_a = _rational_polynomial_trim(
        tuple(_fraction_from_float(value) for value in coefficients_a)
    )
    polynomial_b = _rational_polynomial_trim(
        tuple(_fraction_from_float(value) for value in coefficients_b)
    )
    common = _rational_polynomial_gcd(polynomial_a, polynomial_b)
    if _rational_polynomial_degree(common) <= 0:
        return False
    return _closed_interval_contains_rational_polynomial_root(
        common,
        _fraction_from_float(overlap_left),
        _fraction_from_float(overlap_right),
    )


def _closed_interval_contains_rational_polynomial_root(
    polynomial: tuple[Fraction, ...],
    left: Fraction,
    right: Fraction,
) -> bool:
    if left > right:
        return False
    polynomial = _rational_polynomial_trim(polynomial)
    if _rational_polynomial_eval(polynomial, left) == 0:
        return True
    if _rational_polynomial_eval(polynomial, right) == 0:
        return True
    sequence = _sturm_sequence(polynomial)
    return _sturm_root_count(sequence, left, right) > 0


def _certify_sturm_polynomial_arrangement_from_brackets(
    *,
    arrangement_id: str,
    specs: tuple[PolynomialDecisionFunctionSpec, ...],
    domain: tuple[float, float],
    equality_resolution_policy: str,
    root_multiplicities_by_decision: dict[str, tuple[int, ...]] | None = None,
) -> PolynomialDecisionArrangementStratificationCertificate:
    lower, upper = domain
    root_groups = _group_sturm_root_bracket_occurrences(
        specs,
        root_multiplicities_by_decision=root_multiplicities_by_decision,
    )
    root_brackets = tuple(
        (left, right) for left, right, _decision_ids, _multiplicities in root_groups
    )
    domain_ok = bool(
        arrangement_id
        and specs
        and np.isfinite(lower)
        and np.isfinite(upper)
        and lower < upper
        and all(spec.decision_id and spec.coefficients for spec in specs)
        and _root_brackets_sorted_and_inside(root_brackets, lower, upper)
    )
    specs_by_id = {spec.decision_id: spec for spec in specs}
    strata: list[PolynomialDecisionStratumCertificate] = []
    source_leaves: list[BranchEventTreeLeafCertificate] = []
    stratified_leaves: list[StratifiedBranchLeafCertificate] = []
    cursor = lower

    def add_sturm_sign_cell(index: int, left: float, right: float) -> None:
        if right <= left:
            return
        interval = FloatInterval(left, right)
        signs: list[int] = []
        margins: list[float] = []
        value_intervals: list[FloatInterval] = []
        missing: list[str] = []
        for spec in specs:
            try:
                sign = _sturm_sign_on_root_free_interval(
                    spec.coefficients,
                    left=left,
                    right=right,
                )
                margin = _polynomial_strict_sign_margin_by_subdivision(
                    spec.coefficients,
                    left=left,
                    right=right,
                    expected_sign=sign,
                )
            except ValueError as exc:
                sign = 0
                margin = 0.0
                missing.append(f"{spec.decision_id}:{exc}")
            signs.append(sign)
            margins.append(margin)
            value_intervals.append(
                interval_polynomial_eval(np.asarray(spec.coefficients, dtype=float), interval)
            )
        if any(sign == 0 for sign in signs):
            missing.append("sturm_sign_cell_sign_not_decided")
        if any(margin <= 0.0 for margin in margins):
            missing.append("sturm_sign_cell_positive_margin_not_certified")
        certified = bool(domain_ok and not missing)
        sign_label = ",".join(
            f"{spec.decision_id}:{'+' if sign > 0 else '-'}"
            for spec, sign in zip(specs, signs)
        )
        stratum_id = f"{arrangement_id}:cell:{index}"
        strata.append(
            PolynomialDecisionStratumCertificate(
                stratum_id=stratum_id,
                stratum_kind="sturm_arrangement_sign_cell",
                interval=(left, right),
                value_interval=_aggregate_value_intervals(tuple(value_intervals)),
                sign=1 if certified else 0,
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        source_id = f"{stratum_id}:leaf"
        source_leaves.append(
            BranchEventTreeLeafCertificate(
                leaf_id=source_id,
                leaf_type="sturm_polynomial_arrangement_positive_margin_leaf",
                decision=sign_label if certified else "undecided_sign_vector",
                source_type="SturmPolynomialDecisionArrangement",
                depth=0,
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        stratified_leaves.append(
            StratifiedBranchLeafCertificate(
                leaf_id=f"stratified:{source_id}",
                source_leaf_id=source_id,
                leaf_kind="positive_margin_unique_event",
                terminal_response_kind="sturm_polynomial_arrangement_sign_decision",
                terminal_response_certified=certified,
                source_leaf_certified=certified,
                decision_functions=tuple(
                    AnalyticDecisionFunctionCertificate(
                        function_id=f"{stratum_id}:{spec.decision_id}",
                        function_kind="sturm_polynomial_arrangement_decision_sign",
                        margin_lower_bound=margin,
                        lipschitz_bound=_polynomial_derivative_lipschitz_bound(
                            interval_polyder(np.asarray(spec.coefficients, dtype=float)),
                            interval,
                        ),
                        certified=certified and sign != 0 and margin > 0.0,
                        missing_obligations=tuple(missing),
                    )
                    for spec, sign, margin in zip(specs, signs, margins)
                ),
                missing_obligations=tuple(missing),
            )
        )

    for index, (root_lower, root_upper, decision_ids, multiplicities) in enumerate(root_groups):
        add_sturm_sign_cell(index, cursor, root_lower)
        root_interval = FloatInterval(root_lower, root_upper)
        values = tuple(
            interval_polynomial_eval(
                np.asarray(specs_by_id[decision_id].coefficients, dtype=float),
                root_interval,
            )
            for decision_id in decision_ids
        )
        first_derivative_values = tuple(
            _polynomial_derivative_interval(
                specs_by_id[decision_id].coefficients,
                root_interval,
                order=1,
            )
            for decision_id in decision_ids
        )
        second_derivative_values = tuple(
            _polynomial_derivative_interval(
                specs_by_id[decision_id].coefficients,
                root_interval,
                order=2,
            )
            for decision_id, multiplicity in zip(decision_ids, multiplicities)
            if multiplicity >= 2
        )
        leading_derivative_values = tuple(
            _polynomial_derivative_interval(
                specs_by_id[decision_id].coefficients,
                root_interval,
                order=multiplicity,
            )
            for decision_id, multiplicity in zip(decision_ids, multiplicities)
        )
        lower_derivative_values = tuple(
            _polynomial_derivative_interval(
                specs_by_id[decision_id].coefficients,
                root_interval,
                order=order,
            )
            for decision_id, multiplicity in zip(decision_ids, multiplicities)
            for order in range(1, multiplicity)
        )
        missing = []
        if any(interval_sign(value) != 0 for value in values):
            missing.append("sturm_root_bracket_value_does_not_contain_zero")
        if any(interval_sign(value) != 0 for value in lower_derivative_values):
            missing.append("sturm_multiple_root_lower_derivative_does_not_contain_zero")
        if any(interval_sign(value) == 0 for value in leading_derivative_values):
            missing.append("sturm_root_multiplicity_derivative_sign_not_isolated")
        certified = bool(domain_ok and not missing)
        decision_label = "+".join(decision_ids)
        stratum_id = f"{arrangement_id}:root:{index}:{decision_label}"
        has_multiple_root = any(multiplicity > 1 for multiplicity in multiplicities)
        if has_multiple_root and len(decision_ids) == 1:
            stratum_kind = "sturm_polynomial_multiple_equality_root"
            source_leaf_type = "sturm_polynomial_arrangement_multiple_equality_root_leaf"
        elif has_multiple_root:
            stratum_kind = "simultaneous_polynomial_multiple_equality_root"
            source_leaf_type = "sturm_polynomial_arrangement_simultaneous_multiple_equality_root_leaf"
        else:
            stratum_kind = (
                "simultaneous_polynomial_equality_root"
                if len(decision_ids) > 1
                else "equality_root"
            )
            source_leaf_type = (
                "sturm_polynomial_arrangement_simultaneous_equality_root_leaf"
                if len(decision_ids) > 1
                else "sturm_polynomial_arrangement_equality_root_leaf"
            )
        strata.append(
            PolynomialDecisionStratumCertificate(
                stratum_id=stratum_id,
                stratum_kind=stratum_kind,
                interval=(root_lower, root_upper),
                value_interval=_aggregate_value_intervals(values),
                sign=0,
                derivative_interval=_aggregate_value_intervals(first_derivative_values),
                second_derivative_interval=(
                    _aggregate_value_intervals(second_derivative_values)
                    if second_derivative_values
                    else None
                ),
                root_multiplicities=tuple(multiplicities),
                certified=certified,
                missing_obligations=tuple(missing),
            )
        )
        source_id = f"{stratum_id}:leaf"
        source_leaves.append(
            BranchEventTreeLeafCertificate(
                leaf_id=source_id,
                leaf_type=source_leaf_type,
                decision="simultaneous_event_equality",
                source_type="SturmPolynomialDecisionArrangement",
                depth=0,
                certified=False,
                missing_obligations=(),
            )
        )
        equality = EqualityStratumCertificate(
            stratum_id=stratum_id,
            defining_function_ids=decision_ids,
            leaf_kind="simultaneous_event_equality",
            isolation_certified=certified,
            resolution_policy=str(equality_resolution_policy),
            certified=certified,
            missing_obligations=tuple(missing),
        )
        tie = EventOrderTieLeafCertificate(
            leaf_id=source_id,
            tied_event_ids=tuple(
                event_id
                for decision_id in decision_ids
                for event_id in (
                    f"{decision_id}:negative_side",
                    f"{decision_id}:positive_side",
                )
            ),
            equality_stratum=equality,
            certified=certified,
            missing_obligations=tuple(missing),
        )
        stratified_leaves.append(
            StratifiedBranchLeafCertificate(
                leaf_id=f"stratified:{source_id}",
                source_leaf_id=source_id,
                leaf_kind="simultaneous_event_equality",
                terminal_response_kind="recursive_equality_stratum",
                terminal_response_certified=False,
                source_leaf_certified=True,
                equality_stratum=equality,
                event_order_tie=tie,
            )
        )
        cursor = root_upper
    add_sturm_sign_cell(len(root_groups), cursor, upper)

    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="SturmPolynomialDecisionArrangement",
        leaf_certificates=tuple(source_leaves),
        cover_certified=domain_ok and _strata_cover_domain(strata, lower, upper),
        leaf_decisions_certified=bool(
            source_leaves
            and all(
                leaf.certified
                for leaf in source_leaves
                if "equality" not in leaf.leaf_type
            )
        ),
        equality_strata_explicit=bool(root_groups),
        obligations=(
            BranchEventTreeObligation(
                obligation="sturm_polynomial_arrangement_domain_valid",
                certified=domain_ok,
                detail=f"domain={(lower, upper)!r}; root_brackets={root_brackets!r}",
            ),
            BranchEventTreeObligation(
                obligation="sturm_polynomial_arrangement_strata_cover_domain",
                certified=_strata_cover_domain(strata, lower, upper),
                detail=f"stratum_count={len(strata)}",
            ),
            BranchEventTreeObligation(
                obligation="sturm_polynomial_arrangement_equality_strata_explicit",
                certified=True,
                detail=f"equality_stratum_count={sum(stratum.equality for stratum in strata)}",
                required=False,
            ),
        ),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=tuple(stratified_leaves),
    )
    return PolynomialDecisionArrangementStratificationCertificate(
        arrangement_id=arrangement_id,
        decision_functions=specs,
        domain=(lower, upper),
        strata=tuple(strata),
        source_tree=source_tree,
        stratified_tree=stratified,
        statement="",
        proof_sketch="",
    )


def certify_polynomial_decision_arrangement_recursive_consumption(
    arrangement: PolynomialDecisionArrangementStratificationCertificate,
    *,
    root_dimension: int,
    root_rank: int | None = None,
    child_consumptions: Mapping[
        str,
        RecursiveStratifiedBranchEventConsumptionCertificate,
    ]
    | tuple[
        tuple[str, RecursiveStratifiedBranchEventConsumptionCertificate],
        ...,
    ] = (),
    recursion_kind: str = "polynomial_decision_arrangement",
) -> RecursiveStratifiedBranchEventConsumptionCertificate:
    """Consume a polynomial arrangement by terminal cells and child strata.

    The arrangement constructor derives the finite sign/equality tree from
    explicit polynomial discriminator data.  This adapter then feeds that tree
    into the existing recursive consumption theorem, accepting child
    certificates keyed by any natural equality-stratum id:

    - ``arrangement:root:k:decision``;
    - ``arrangement:root:k:decision:leaf``;
    - ``stratified:arrangement:root:k:decision:leaf``.

    It still proves only this displayed finite recursive arrangement; it does
    not claim the arbitrary branch/event-order partition theorem.
    """

    if not isinstance(
        arrangement,
        PolynomialDecisionArrangementStratificationCertificate,
    ):
        raise TypeError(
            "arrangement must be a "
            "PolynomialDecisionArrangementStratificationCertificate"
        )
    child_by_leaf = _normalize_arrangement_child_consumptions(
        arrangement,
        child_consumptions,
    )
    if (
        not child_by_leaf
        and arrangement.proof_certified
        and arrangement.equality_stratum_count
    ):
        child_by_leaf = _normalize_arrangement_child_consumptions(
            arrangement,
            derive_polynomial_decision_arrangement_child_consumptions(
                arrangement,
                child_root_rank=0,
            ),
        )
    if recursion_kind == "polynomial_decision_arrangement":
        recursion_kind = _default_polynomial_arrangement_recursion_kind(arrangement)
    return certify_recursive_stratified_branch_event_consumption(
        arrangement.stratified_tree,
        root_dimension=root_dimension,
        root_rank=root_rank,
        child_consumptions=child_by_leaf,
        recursion_kind=recursion_kind,
    )


def derive_polynomial_decision_arrangement_child_consumptions(
    arrangement: PolynomialDecisionArrangementStratificationCertificate,
    *,
    child_root_rank: int | None = 0,
) -> dict[str, RecursiveStratifiedBranchEventConsumptionCertificate]:
    """Derive terminal 0D children for every certified polynomial root stratum."""

    if not isinstance(
        arrangement,
        PolynomialDecisionArrangementStratificationCertificate,
    ):
        raise TypeError(
            "arrangement must be a "
            "PolynomialDecisionArrangementStratificationCertificate"
        )
    if not arrangement.proof_certified:
        raise ValueError("polynomial decision arrangement must be proof-certified")

    return _derive_polynomial_root_child_consumptions_from_strata(
        strata=arrangement.strata,
        source_id=arrangement.arrangement_id,
        child_root_rank=child_root_rank,
    )


def _default_polynomial_arrangement_recursion_kind(
    arrangement: PolynomialDecisionArrangementStratificationCertificate,
) -> str:
    source_type = str(getattr(arrangement.source_tree, "source_type", ""))
    if source_type == "AffineDecisionArrangement":
        return "affine_decision_arrangement"
    if source_type == "SturmPolynomialDecisionArrangement":
        return "sturm_polynomial_decision_arrangement"
    if source_type == "QuadraticDoubleRootArrangement":
        return "quadratic_double_root_decision_arrangement"
    if source_type == "PolynomialRootArrangement":
        return "computed_polynomial_root_decision_arrangement"
    return "polynomial_decision_arrangement"


def _infer_stratified_leaf(
    source_leaf: BranchEventTreeLeafCertificate,
) -> StratifiedBranchLeafCertificate:
    leaf_kind = _infer_leaf_kind(source_leaf)
    missing = tuple(source_leaf.missing_obligations)
    terminal_response_certified = bool(
        source_leaf.certified
        and leaf_kind
        in {
            "positive_margin_unique_event",
            "no_event_before_target",
            "separated_binary_entry",
        }
    )
    return StratifiedBranchLeafCertificate(
        leaf_id=f"stratified:{source_leaf.leaf_id}",
        source_leaf_id=source_leaf.leaf_id,
        leaf_kind=leaf_kind,
        terminal_response_kind=(
            "certified_leaf_response"
            if terminal_response_certified
            else "pending_stratified_response"
        ),
        terminal_response_certified=terminal_response_certified,
        source_leaf_certified=bool(source_leaf.certified),
        depth=source_leaf.depth,
        missing_obligations=missing,
    )


def _infer_leaf_kind(source_leaf: BranchEventTreeLeafCertificate) -> str:
    tokens = " ".join(
        (
            source_leaf.leaf_type,
            source_leaf.decision,
            source_leaf.leaf_id,
        )
    ).lower()
    if "unsupported" in tokens:
        return "unsupported_analytic_stratum"
    if "total" in tokens and "collision" in tokens:
        return "total_collision_cluster"
    if "selector" in tokens:
        return "selector_policy"
    if "tie" in tokens or "equality" in tokens or "ambiguous" in tokens:
        return "simultaneous_event_equality"
    if "no_event" in tokens or "target_precedes" in tokens:
        return "no_event_before_target"
    if "binary" in tokens or "ks" in tokens or "levi" in tokens or "lc" in tokens:
        return "separated_binary_entry"
    return "positive_margin_unique_event"


def _root_brackets_sorted_and_inside(
    root_brackets: tuple[tuple[float, float], ...],
    lower: float,
    upper: float,
) -> bool:
    cursor = lower
    for left, right in root_brackets:
        if not (
            np.isfinite(left)
            and np.isfinite(right)
            and cursor <= left < right <= upper
        ):
            return False
        cursor = right
    return True


def _strata_cover_domain(
    strata: list[PolynomialDecisionStratumCertificate],
    lower: float,
    upper: float,
) -> bool:
    if not strata:
        return False
    cursor = lower
    for stratum in strata:
        left, right = stratum.interval
        if abs(left - cursor) > 1.0e-12 or right < left:
            return False
        cursor = right
    return abs(cursor - upper) <= 1.0e-12


def _polynomial_derivative_lipschitz_bound(
    derivative_coefficients: tuple[FloatInterval, ...],
    interval: FloatInterval,
) -> float:
    if not derivative_coefficients:
        return 0.0
    value = interval_polynomial_eval(derivative_coefficients, interval)
    return float(max(abs(value.lower), abs(value.upper)))


def _polynomial_derivative_interval(
    coefficients: tuple[float, ...] | np.ndarray,
    interval: FloatInterval,
    *,
    order: int,
) -> FloatInterval:
    derivative = np.asarray(coefficients, dtype=float)
    for _ in range(max(0, int(order))):
        derivative = interval_polyder(derivative)
    if not derivative:
        return FloatInterval.point(0.0)
    return interval_polynomial_eval(derivative, interval)


def _strict_sign_margin(value: FloatInterval) -> float:
    sign = interval_sign(value)
    if sign > 0:
        return float(value.lower)
    if sign < 0:
        return float(-value.upper)
    return 0.0


def _aggregate_value_intervals(
    values: tuple[FloatInterval, ...],
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    return (
        float(min(value.lower for value in values)),
        float(max(value.upper for value in values)),
    )


def _degree_two_polynomial_value_interval(
    coefficients: tuple[float, ...] | np.ndarray,
    interval: FloatInterval,
) -> FloatInterval:
    """Exact real range for scalar degree <= 2 polynomial data.

    These constructors compute affine/quadratic roots from coefficient
    formulas.  Naive interval evaluation of expanded quadratics such as
    ``(x-a)^2`` can contain zero on cells that are actually separated from the
    double root because the interval variable is reused.  For degree <= 2,
    endpoint plus vertex evaluation gives the sharp scalar range on the compact
    interval.
    """

    coeffs = tuple(float(value) for value in coefficients)
    if any(not np.isfinite(value) for value in coeffs):
        raise ValueError("polynomial coefficients must be finite")
    lower, upper = float(interval.lower), float(interval.upper)
    if len(coeffs) == 0:
        return FloatInterval.point(0.0)
    if len(coeffs) == 1:
        return FloatInterval.point(coeffs[0])
    if len(coeffs) == 2:
        constant, linear = coeffs
        values = (constant + linear * lower, constant + linear * upper)
    elif len(coeffs) == 3:
        constant, linear, quadratic = coeffs
        values_list = [
            constant + linear * lower + quadratic * lower * lower,
            constant + linear * upper + quadratic * upper * upper,
        ]
        if quadratic != 0.0:
            vertex = -linear / (2.0 * quadratic)
            if lower <= vertex <= upper:
                values_list.append(
                    constant + linear * vertex + quadratic * vertex * vertex
                )
        values = tuple(values_list)
    else:
        return interval_polynomial_eval(np.asarray(coeffs, dtype=float), interval)
    radius = 64.0 * np.finfo(float).eps * max(
        1.0,
        *(abs(value) for value in values),
    )
    return FloatInterval(float(min(values) - radius), float(max(values) + radius))


def _sturm_isolate_polynomial_root_occurrences(
    coefficients: tuple[float, ...],
    *,
    domain: tuple[float, float],
    max_bisection_depth: int,
) -> tuple[tuple[float, float, int], ...]:
    lower, upper = (_fraction_from_float(domain[0]), _fraction_from_float(domain[1]))
    if lower >= upper:
        raise ValueError("Sturm isolation requires a nonempty compact interval")
    polynomial = _rational_polynomial_trim(
        tuple(_fraction_from_float(value) for value in coefficients)
    )
    degree = _rational_polynomial_degree(polynomial)
    if degree <= 0:
        return ()
    derivative = _rational_polynomial_derivative(polynomial)
    repeated_part = _rational_polynomial_gcd(polynomial, derivative)
    if _rational_polynomial_degree(repeated_part) > 0:
        square_free_part, remainder = _rational_polynomial_divmod(
            polynomial,
            repeated_part,
        )
        if _rational_polynomial_degree(remainder) >= 0:
            raise ValueError("Sturm square-free division left a nonzero remainder")
    else:
        square_free_part = polynomial
    square_free_part = _rational_polynomial_trim(square_free_part)
    square_free_derivative = _rational_polynomial_derivative(square_free_part)
    square_free_coefficients = _rational_polynomial_float_coefficients(
        square_free_part
    )
    reduced_square_free_part = square_free_part

    lower_boundary_root = _rational_polynomial_eval(square_free_part, lower) == 0
    upper_boundary_root = _rational_polynomial_eval(square_free_part, upper) == 0
    if lower_boundary_root:
        if _rational_polynomial_eval(square_free_derivative, lower) == 0:
            raise ValueError("Sturm lower boundary root is not square-free")
        reduced_square_free_part = _divide_by_linear_root_exact(
            reduced_square_free_part,
            lower,
        )
    if upper_boundary_root:
        if _rational_polynomial_eval(square_free_derivative, upper) == 0:
            raise ValueError("Sturm upper boundary root is not square-free")
        reduced_square_free_part = _divide_by_linear_root_exact(
            reduced_square_free_part,
            upper,
        )
    interior_brackets = _sturm_isolate_interior_root_brackets(
        count_polynomial=reduced_square_free_part,
        certificate_coefficients=square_free_coefficients,
        domain=(lower, upper),
        max_bisection_depth=max_bisection_depth,
    )
    boundary_brackets: list[tuple[float, float]] = []
    if lower_boundary_root:
        next_boundary = (
            _fraction_from_float(interior_brackets[0][0])
            if interior_brackets
            else upper
        )
        boundary_brackets.append(
            _sturm_boundary_root_bracket(
                square_free_coefficients,
                root=lower,
                adjacent=next_boundary,
                side="lower",
            )
        )
    if upper_boundary_root:
        previous_boundary = (
            _fraction_from_float(interior_brackets[-1][1])
            if interior_brackets
            else lower
        )
        boundary_brackets.append(
            _sturm_boundary_root_bracket(
                square_free_coefficients,
                root=upper,
                adjacent=previous_boundary,
                side="upper",
            )
        )
    brackets = tuple(sorted(boundary_brackets + list(interior_brackets)))
    return tuple(
        (
            left,
            right,
            _rational_polynomial_root_multiplicity_in_bracket(
                polynomial,
                square_free_part,
                (_fraction_from_float(left), _fraction_from_float(right)),
            ),
        )
        for left, right in brackets
    )


def _sturm_isolate_square_free_root_brackets(
    coefficients: tuple[float, ...],
    *,
    domain: tuple[float, float],
    max_bisection_depth: int,
) -> tuple[tuple[float, float], ...]:
    lower, upper = (_fraction_from_float(domain[0]), _fraction_from_float(domain[1]))
    if lower >= upper:
        raise ValueError("Sturm isolation requires a nonempty compact interval")
    polynomial = _rational_polynomial_trim(
        tuple(_fraction_from_float(value) for value in coefficients)
    )
    degree = _rational_polynomial_degree(polynomial)
    if degree <= 0:
        return ()
    derivative = _rational_polynomial_derivative(polynomial)
    gcd = _rational_polynomial_gcd(polynomial, derivative)
    if _rational_polynomial_degree(gcd) > 0:
        raise ValueError("Sturm polynomial arrangement requires square-free discriminants")
    reduced_polynomial = polynomial
    lower_boundary_root = _rational_polynomial_eval(polynomial, lower) == 0
    upper_boundary_root = _rational_polynomial_eval(polynomial, upper) == 0
    if lower_boundary_root:
        if _rational_polynomial_eval(derivative, lower) == 0:
            raise ValueError("Sturm lower boundary root is not simple")
        reduced_polynomial = _divide_by_linear_root_exact(reduced_polynomial, lower)
    if upper_boundary_root:
        if _rational_polynomial_eval(derivative, upper) == 0:
            raise ValueError("Sturm upper boundary root is not simple")
        reduced_polynomial = _divide_by_linear_root_exact(reduced_polynomial, upper)
    interior_brackets = _sturm_isolate_interior_root_brackets(
        count_polynomial=reduced_polynomial,
        certificate_coefficients=coefficients,
        domain=(lower, upper),
        max_bisection_depth=max_bisection_depth,
    )
    boundary_brackets: list[tuple[float, float]] = []
    if lower_boundary_root:
        next_boundary = (
            _fraction_from_float(interior_brackets[0][0])
            if interior_brackets
            else upper
        )
        boundary_brackets.append(
            _sturm_boundary_root_bracket(
                coefficients,
                root=lower,
                adjacent=next_boundary,
                side="lower",
            )
        )
    if upper_boundary_root:
        previous_boundary = (
            _fraction_from_float(interior_brackets[-1][1])
            if interior_brackets
            else lower
        )
        boundary_brackets.append(
            _sturm_boundary_root_bracket(
                coefficients,
                root=upper,
                adjacent=previous_boundary,
                side="upper",
            )
        )
    return tuple(sorted(boundary_brackets + list(interior_brackets)))


def _sturm_isolate_interior_root_brackets(
    *,
    count_polynomial: tuple[Fraction, ...],
    certificate_coefficients: tuple[float, ...],
    domain: tuple[Fraction, Fraction],
    max_bisection_depth: int,
) -> tuple[tuple[float, float], ...]:
    lower, upper = domain
    count_polynomial = _rational_polynomial_trim(count_polynomial)
    if _rational_polynomial_degree(count_polynomial) <= 0:
        return ()
    if _rational_polynomial_eval(count_polynomial, lower) == 0:
        raise ValueError("Sturm interior isolation lower endpoint is a root")
    if _rational_polynomial_eval(count_polynomial, upper) == 0:
        raise ValueError("Sturm interior isolation upper endpoint is a root")
    sequence = _sturm_sequence(count_polynomial)
    root_count = _sturm_root_count(sequence, lower, upper)
    if root_count == 0:
        return ()
    max_bracket_width = (upper - lower) / 64
    root_brackets: list[tuple[float, float]] = []
    stack: list[tuple[Fraction, Fraction, int, int]] = [
        (lower, upper, root_count, 0)
    ]
    while stack:
        left, right, count, depth = stack.pop()
        if count == 0:
            continue
        if (
            count == 1
            and right - left <= max_bracket_width
            and _sturm_float_bracket_certifies(certificate_coefficients, left, right)
        ):
            root_brackets.append((float(left), float(right)))
            continue
        if depth >= max_bisection_depth:
            raise ValueError("Sturm root isolation exceeded max_bisection_depth")
        midpoint = _nonroot_rational_split(count_polynomial, left, right)
        left_count = _sturm_root_count(sequence, left, midpoint)
        right_count = _sturm_root_count(sequence, midpoint, right)
        if left_count + right_count != count:
            raise ValueError("Sturm bisection root counts did not add up")
        stack.append((midpoint, right, right_count, depth + 1))
        stack.append((left, midpoint, left_count, depth + 1))
    return tuple(sorted(root_brackets))


def _divide_by_linear_root_exact(
    polynomial: tuple[Fraction, ...],
    root: Fraction,
) -> tuple[Fraction, ...]:
    quotient, remainder = _rational_polynomial_divmod(
        polynomial,
        (-root, Fraction(1)),
    )
    if _rational_polynomial_degree(remainder) >= 0:
        raise ValueError("boundary root division left a nonzero remainder")
    return quotient


def _sturm_boundary_root_bracket(
    coefficients: tuple[float, ...],
    *,
    root: Fraction,
    adjacent: Fraction,
    side: str,
) -> tuple[float, float]:
    gap = abs(adjacent - root)
    if gap <= 0:
        raise ValueError("Sturm boundary root lacks separation from adjacent roots")
    for denominator in (4, 8, 16, 32, 64, 128, 256, 512):
        width = gap / denominator
        if side == "lower":
            left, right = root, root + width
        elif side == "upper":
            left, right = root - width, root
        else:
            raise ValueError("boundary side must be lower or upper")
        if _sturm_float_bracket_certifies(coefficients, left, right):
            return (float(left), float(right))
    raise ValueError("Sturm boundary root bracket did not certify")


def _fraction_from_float(value: float) -> Fraction:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Sturm rationalization requires finite floats")
    return Fraction(str(value))


def _rational_polynomial_trim(
    polynomial: tuple[Fraction, ...] | list[Fraction],
) -> tuple[Fraction, ...]:
    values = list(polynomial)
    while values and values[-1] == 0:
        values.pop()
    return tuple(values) if values else (Fraction(0),)


def _rational_polynomial_degree(polynomial: tuple[Fraction, ...]) -> int:
    polynomial = _rational_polynomial_trim(polynomial)
    if len(polynomial) == 1 and polynomial[0] == 0:
        return -1
    return len(polynomial) - 1


def _rational_polynomial_derivative(
    polynomial: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    polynomial = _rational_polynomial_trim(polynomial)
    if len(polynomial) <= 1:
        return (Fraction(0),)
    return _rational_polynomial_trim(
        tuple(Fraction(index) * coefficient for index, coefficient in enumerate(polynomial[1:], start=1))
    )


def _rational_polynomial_float_coefficients(
    polynomial: tuple[Fraction, ...],
) -> tuple[float, ...]:
    return tuple(float(coefficient) for coefficient in _rational_polynomial_trim(polynomial))


def _rational_polynomial_derivative_order(
    polynomial: tuple[Fraction, ...],
    order: int,
) -> tuple[Fraction, ...]:
    result = _rational_polynomial_trim(polynomial)
    for _ in range(max(0, int(order))):
        result = _rational_polynomial_derivative(result)
    return _rational_polynomial_trim(result)


def _rational_polynomial_root_multiplicity_in_bracket(
    polynomial: tuple[Fraction, ...],
    square_free_part: tuple[Fraction, ...],
    bracket: tuple[Fraction, Fraction],
) -> int:
    left, right = bracket
    if left > right:
        raise ValueError("multiplicity bracket is not ordered")
    degree = _rational_polynomial_degree(polynomial)
    for order in range(1, degree + 1):
        derivative = _rational_polynomial_derivative_order(polynomial, order)
        common = _rational_polynomial_gcd(square_free_part, derivative)
        if not _closed_interval_contains_rational_polynomial_root(
            common,
            left,
            right,
        ):
            return order
    raise ValueError("could not identify finite root multiplicity")


def _rational_polynomial_eval(
    polynomial: tuple[Fraction, ...],
    point: Fraction,
) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(_rational_polynomial_trim(polynomial)):
        result = result * point + coefficient
    return result


def _rational_polynomial_divmod(
    dividend: tuple[Fraction, ...],
    divisor: tuple[Fraction, ...],
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    dividend = _rational_polynomial_trim(dividend)
    divisor = _rational_polynomial_trim(divisor)
    divisor_degree = _rational_polynomial_degree(divisor)
    if divisor_degree < 0:
        raise ValueError("cannot divide by zero polynomial")
    dividend_degree = _rational_polynomial_degree(dividend)
    if dividend_degree < divisor_degree:
        return ((Fraction(0),), dividend)
    quotient = [Fraction(0)] * (dividend_degree - divisor_degree + 1)
    remainder = list(dividend)
    divisor_lead = divisor[-1]
    while (
        _rational_polynomial_degree(tuple(remainder)) >= divisor_degree
        and not (_rational_polynomial_degree(tuple(remainder)) < 0)
    ):
        remainder = list(_rational_polynomial_trim(remainder))
        shift = len(remainder) - len(divisor)
        scale = remainder[-1] / divisor_lead
        quotient[shift] += scale
        for index, coefficient in enumerate(divisor):
            remainder[index + shift] -= scale * coefficient
    return (
        _rational_polynomial_trim(quotient),
        _rational_polynomial_trim(remainder),
    )


def _rational_polynomial_gcd(
    left: tuple[Fraction, ...],
    right: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    a = _rational_polynomial_trim(left)
    b = _rational_polynomial_trim(right)
    while _rational_polynomial_degree(b) >= 0:
        _quotient, remainder = _rational_polynomial_divmod(a, b)
        if _rational_polynomial_degree(remainder) < 0:
            a = b
            break
        a, b = b, remainder
    degree = _rational_polynomial_degree(a)
    if degree < 0:
        return (Fraction(0),)
    lead = a[-1]
    return _rational_polynomial_trim(tuple(coefficient / lead for coefficient in a))


def _sturm_sequence(polynomial: tuple[Fraction, ...]) -> tuple[tuple[Fraction, ...], ...]:
    polynomial = _rational_polynomial_trim(polynomial)
    derivative = _rational_polynomial_derivative(polynomial)
    sequence = [polynomial, derivative]
    while _rational_polynomial_degree(sequence[-1]) >= 0:
        _quotient, remainder = _rational_polynomial_divmod(sequence[-2], sequence[-1])
        if _rational_polynomial_degree(remainder) < 0:
            break
        sequence.append(tuple(-coefficient for coefficient in remainder))
    return tuple(sequence)


def _sturm_variations_at(
    sequence: tuple[tuple[Fraction, ...], ...],
    point: Fraction,
) -> int:
    signs: list[int] = []
    for polynomial in sequence:
        value = _rational_polynomial_eval(polynomial, point)
        if value > 0:
            signs.append(1)
        elif value < 0:
            signs.append(-1)
    return sum(
        1
        for left, right in zip(signs, signs[1:])
        if left != right
    )


def _sturm_root_count(
    sequence: tuple[tuple[Fraction, ...], ...],
    left: Fraction,
    right: Fraction,
) -> int:
    if left >= right:
        return 0
    return _sturm_variations_at(sequence, left) - _sturm_variations_at(sequence, right)


def _nonroot_rational_split(
    polynomial: tuple[Fraction, ...],
    left: Fraction,
    right: Fraction,
) -> Fraction:
    for denominator in (2, 3, 4, 5, 7, 11, 13):
        for numerator in range(1, denominator):
            point = left + (right - left) * Fraction(numerator, denominator)
            if _rational_polynomial_eval(polynomial, point) != 0:
                return point
    raise ValueError("could not find a nonroot rational split point")


def _sturm_float_bracket_certifies(
    coefficients: tuple[float, ...],
    left: Fraction,
    right: Fraction,
) -> bool:
    if left >= right:
        return False
    interval = FloatInterval(float(left), float(right))
    coefficient_array = np.asarray(coefficients, dtype=float)
    value = interval_polynomial_eval(coefficient_array, interval)
    derivative = interval_polyder(coefficient_array)
    derivative_value = (
        interval_polynomial_eval(derivative, interval)
        if derivative
        else FloatInterval.point(0.0)
    )
    return bool(interval_sign(value) == 0 and interval_sign(derivative_value) != 0)


def _sturm_sign_on_root_free_interval(
    coefficients: tuple[float, ...],
    *,
    left: float,
    right: float,
) -> int:
    if right <= left:
        raise ValueError("sturm_sign_interval_not_positive")
    polynomial = _rational_polynomial_trim(
        tuple(_fraction_from_float(value) for value in coefficients)
    )
    sequence = _sturm_sequence(polynomial)
    left_fraction = _fraction_from_float(left)
    right_fraction = _fraction_from_float(right)
    if _sturm_root_count(sequence, left_fraction, right_fraction) != 0:
        raise ValueError("sturm_sign_cell_contains_root")
    probe = (left_fraction + right_fraction) / 2
    value = _rational_polynomial_eval(polynomial, probe)
    if value == 0:
        probe = _nonroot_rational_split(polynomial, left_fraction, right_fraction)
        value = _rational_polynomial_eval(polynomial, probe)
    if value > 0:
        return 1
    if value < 0:
        return -1
    raise ValueError("sturm_sign_probe_is_root")


def _polynomial_strict_sign_margin_by_subdivision(
    coefficients: tuple[float, ...],
    *,
    left: float,
    right: float,
    expected_sign: int,
    max_depth: int = 32,
) -> float:
    if expected_sign not in {-1, 1}:
        raise ValueError("expected sign must be nonzero")
    coefficient_array = np.asarray(coefficients, dtype=float)
    stack: list[tuple[float, float, int]] = [(float(left), float(right), 0)]
    margins: list[float] = []
    while stack:
        current_left, current_right, depth = stack.pop()
        if current_right <= current_left:
            continue
        value = interval_polynomial_eval(
            coefficient_array,
            FloatInterval(current_left, current_right),
        )
        sign = interval_sign(value)
        if sign == expected_sign:
            margins.append(_strict_sign_margin(value))
            continue
        if depth >= max_depth:
            raise ValueError("subdivision_sign_margin_not_certified")
        midpoint = 0.5 * (current_left + current_right)
        if midpoint <= current_left or midpoint >= current_right:
            raise ValueError("subdivision_sign_margin_interval_collapse")
        stack.append((midpoint, current_right, depth + 1))
        stack.append((current_left, midpoint, depth + 1))
    if not margins:
        raise ValueError("subdivision_sign_margin_empty_cell")
    margin = min(margins)
    if margin <= 0.0:
        raise ValueError("subdivision_sign_margin_nonpositive")
    return float(margin)


def _axis_aligned_affine_box_axis_pieces(
    *,
    axis: int,
    interval: tuple[float, float],
    root_items: list[tuple[float, str]],
) -> tuple[tuple[str, tuple[float, float], tuple[str, ...]], ...]:
    lower, upper = interval
    root_groups = _group_affine_roots(root_items)
    root_values = tuple(root for root, _decision_ids in root_groups)
    brackets = _root_group_brackets(
        root_values,
        domain=(lower, upper),
        error_label="axis-aligned affine box root slabs require separated roots",
    )
    pieces: list[tuple[str, tuple[float, float], tuple[str, ...]]] = []
    cursor = lower
    for (_root, decision_ids), bracket in zip(root_groups, brackets):
        left, right = bracket
        if left > cursor:
            pieces.append(("sign", (float(cursor), float(left)), ()))
        pieces.append(("equality", (float(left), float(right)), tuple(decision_ids)))
        cursor = right
    if cursor < upper:
        pieces.append(("sign", (float(cursor), float(upper)), ()))
    if not pieces:
        pieces.append(("sign", (float(lower), float(upper)), ()))
    return tuple(pieces)


def _axis_aligned_affine_box_spec_axis_root(
    spec: AffineBoxDecisionFunctionSpec,
    *,
    dimension: int,
    tolerance: float,
) -> tuple[int, float] | None:
    if len(spec.coefficients) != dimension + 1:
        return None
    constant = float(spec.coefficients[0])
    slopes = tuple(float(value) for value in spec.coefficients[1:])
    nonzero_axes = tuple(
        axis
        for axis, slope in enumerate(slopes)
        if abs(slope) > tolerance * max(1.0, abs(constant), abs(slope))
    )
    if len(nonzero_axes) != 1:
        return None
    axis = nonzero_axes[0]
    slope = slopes[axis]
    root = -constant / slope
    if not np.isfinite(root):
        return None
    return axis, float(root)


def _affine_halfspace_decision_exact_child_geometry_detail(
    *,
    coefficients: tuple[float, ...],
    domain_box: tuple[tuple[float, float], ...],
    tolerance: float,
) -> tuple[int, str] | None:
    dimension = len(domain_box)
    if len(coefficients) != dimension + 1:
        return None
    value_bounds = _affine_box_value_interval(coefficients, domain_box)
    if value_bounds[0] > tolerance or value_bounds[1] < -tolerance:
        return None
    if dimension == 1:
        c0, slope = coefficients
        if abs(slope) <= tolerance:
            return None
        root = -c0 / slope
        left, right = domain_box[0]
        if left - tolerance <= root <= right + tolerance:
            return 0, f"root={float(root)!r}"
        return None
    if dimension == 2:
        line = _affine_halfspace_decision_line_domain(
            coefficients=coefficients,
            domain_box=domain_box,
            tolerance=tolerance,
        )
        if line is None:
            return None
        point, direction, parameter_domain = line
        if parameter_domain[1] - parameter_domain[0] <= tolerance:
            return None
        return (
            1,
            (
                f"point={point!r}; direction={direction!r}; "
                f"parameter_domain={parameter_domain!r}"
            ),
        )
    if dimension == 3:
        vertices = _plane_box_intersection_vertices_3d(
            coefficients,
            domain_box,
            tolerance=tolerance,
        )
        if len(vertices) < 3:
            return None
        normal = np.asarray(coefficients[1:], dtype=float)
        basis = _orthonormal_plane_basis_3d(normal)
        if basis is None:
            return None
        normal_norm_sq = float(np.dot(normal, normal))
        if normal_norm_sq <= tolerance * tolerance:
            return None
        point_array = -float(coefficients[0]) * normal / normal_norm_sq
        basis_u, basis_v = basis
        polygon = _sort_parameter_polygon_vertices(
            tuple(
                (
                    float(np.dot(np.asarray(vertex, dtype=float) - point_array, basis_u)),
                    float(np.dot(np.asarray(vertex, dtype=float) - point_array, basis_v)),
                )
                for vertex in vertices
            )
        )
        area = _polygon_area(polygon)
        if area <= tolerance:
            return None
        return 2, f"vertex_count={len(vertices)}; area_lower_bound={float(area)!r}"
    return None


def _affine_halfspace_decision_line_domain(
    *,
    coefficients: tuple[float, ...],
    domain_box: tuple[tuple[float, float], ...],
    tolerance: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    if len(coefficients) != 3 or len(domain_box) != 2:
        return None
    c0, ax, ay = coefficients
    normal_norm_sq = ax * ax + ay * ay
    if normal_norm_sq <= tolerance * tolerance or not np.isfinite(normal_norm_sq):
        return None
    normal_norm = float(np.sqrt(normal_norm_sq))
    point = (
        float(-c0 * ax / normal_norm_sq),
        float(-c0 * ay / normal_norm_sq),
    )
    direction = (float(-ay / normal_norm), float(ax / normal_norm))
    interval = (-float("inf"), float("inf"))
    for coordinate, (lower, upper) in enumerate(domain_box):
        offset = point[coordinate]
        slope = direction[coordinate]
        interval = _intersect_affine_leq_interval(interval, offset, slope, upper)
        if interval is None:
            return None
        interval = _intersect_affine_leq_interval(interval, -offset, -slope, -lower)
        if interval is None:
            return None
    lower, upper = interval
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        return None
    return point, direction, (float(lower), float(upper))


def _axis_aligned_pieces_cover_box(
    axis_pieces: tuple[
        tuple[tuple[str, tuple[float, float], tuple[str, ...]], ...],
        ...
    ],
    box: tuple[tuple[float, float], ...],
) -> bool:
    if len(axis_pieces) != len(box):
        return False
    for pieces, (lower, upper) in zip(axis_pieces, box):
        cursor = lower
        if not pieces:
            return False
        for _kind, (left, right), _decision_ids in pieces:
            if abs(left - cursor) > 1.0e-12 or right < left:
                return False
            cursor = right
        if abs(cursor - upper) > 1.0e-12:
            return False
    return True


def _affine_box_value_interval(
    coefficients: tuple[float, ...],
    box: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    lower = upper = float(coefficients[0])
    for slope, (left, right) in zip(coefficients[1:], box):
        if slope >= 0.0:
            lower += slope * left
            upper += slope * right
        else:
            lower += slope * right
            upper += slope * left
    radius = 64.0 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))
    return (float(lower - radius), float(upper + radius))


def _strict_affine_box_sign_margin(
    bounds: tuple[float, float],
    sign: int,
) -> float:
    lower, upper = bounds
    if sign > 0:
        return float(lower)
    if sign < 0:
        return float(-upper)
    return 0.0


def _affine_box_lipschitz_bound(coefficients: tuple[float, ...]) -> float:
    slopes = np.asarray(coefficients[1:], dtype=float)
    return float(np.sum(np.abs(slopes)))


def _rectangle_polygon(
    box: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    (x0, x1), (y0, y1) = box
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _clip_polygon_halfplane(
    polygon: tuple[tuple[float, float], ...],
    *,
    normal: tuple[float, float],
    rhs: float,
) -> tuple[tuple[float, float], ...]:
    if not polygon:
        return ()
    nx, ny = normal
    tolerance = 1.0e-12 * max(1.0, abs(nx), abs(ny), abs(rhs))

    def value(point: tuple[float, float]) -> float:
        return nx * point[0] + ny * point[1] - rhs

    clipped: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_value = value(previous)
    previous_inside = previous_value <= tolerance
    for current in polygon:
        current_value = value(current)
        current_inside = current_value <= tolerance
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > tolerance:
                ratio = previous_value / denominator
                clipped.append(
                    (
                        float(previous[0] + ratio * (current[0] - previous[0])),
                        float(previous[1] + ratio * (current[1] - previous[1])),
                    )
                )
        if current_inside:
            clipped.append((float(current[0]), float(current[1])))
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    return _dedupe_polygon_vertices(tuple(clipped))


def _dedupe_polygon_vertices(
    vertices: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    deduped: list[tuple[float, float]] = []
    for vertex in vertices:
        if not deduped or _point_distance(vertex, deduped[-1]) > 1.0e-10:
            deduped.append(vertex)
    if len(deduped) > 1 and _point_distance(deduped[0], deduped[-1]) <= 1.0e-10:
        deduped.pop()
    return tuple(deduped)


def _point_distance(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return float(np.hypot(left[0] - right[0], left[1] - right[1]))


def _polygon_area(vertices: tuple[tuple[float, float], ...]) -> float:
    if len(vertices) < 3:
        return 0.0
    area = 0.0
    for (x0, y0), (x1, y1) in zip(vertices, (*vertices[1:], vertices[0])):
        area += x0 * y1 - x1 * y0
    return float(abs(area) * 0.5)


def _affine_polygon_value_bounds(
    coefficients: tuple[float, ...],
    vertices: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    if len(coefficients) != 3 or not vertices:
        return (float("inf"), float("-inf"))
    c0, ax, ay = coefficients
    values = tuple(float(c0 + ax * x + ay * y) for x, y in vertices)
    radius = 128.0 * np.finfo(float).eps * max(
        1.0,
        *(abs(value) for value in values),
    )
    return (float(min(values) - radius), float(max(values) + radius))


def _affine_polygon_value_bounds_satisfy_signs(
    value_bounds: tuple[tuple[float, float], ...],
    *,
    signs: tuple[int, ...],
    slab_half_width: float,
) -> bool:
    tolerance = 1.0e-9 * max(1.0, slab_half_width)
    if len(value_bounds) != len(signs):
        return False
    for (lower, upper), sign in zip(value_bounds, signs):
        if sign < 0 and upper > -slab_half_width + tolerance:
            return False
        if sign > 0 and lower < slab_half_width - tolerance:
            return False
        if sign == 0 and (lower < -slab_half_width - tolerance or upper > slab_half_width + tolerance):
            return False
    return True


def _box_halfspace_constraints_3d(
    box: tuple[tuple[float, float], ...],
) -> tuple[tuple[tuple[float, float, float], float], ...]:
    constraints: list[tuple[tuple[float, float, float], float]] = []
    for axis, (left, right) in enumerate(box):
        normal = [0.0, 0.0, 0.0]
        normal[axis] = 1.0
        constraints.append((tuple(normal), float(right)))  # type: ignore[arg-type]
        normal = [0.0, 0.0, 0.0]
        normal[axis] = -1.0
        constraints.append((tuple(normal), float(-left)))  # type: ignore[arg-type]
    return tuple(constraints)


def _halfspace_polyhedron_vertices_3d(
    constraints: tuple[tuple[tuple[float, float, float], float], ...],
) -> tuple[tuple[float, float, float], ...]:
    vertices: list[tuple[float, float, float]] = []
    tolerance = 1.0e-9 * max(
        1.0,
        *(abs(rhs) for _normal, rhs in constraints),
        *(abs(component) for normal, _rhs in constraints for component in normal),
    )
    for triple in combinations(range(len(constraints)), 3):
        matrix = np.asarray([constraints[index][0] for index in triple], dtype=float)
        rhs = np.asarray([constraints[index][1] for index in triple], dtype=float)
        try:
            if abs(float(np.linalg.det(matrix))) <= tolerance:
                continue
            point = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            continue
        if not all(np.isfinite(value) for value in point):
            continue
        if all(
            float(np.dot(np.asarray(normal, dtype=float), point)) <= rhs + tolerance
            for normal, rhs in constraints
        ):
            candidate = tuple(float(value) for value in point)
            if not any(_point3_distance(candidate, existing) <= 1.0e-8 for existing in vertices):
                vertices.append(candidate)  # type: ignore[arg-type]
    return tuple(vertices)


def _deduplicate_halfspace_constraints_3d(
    constraints: tuple[tuple[tuple[float, float, float], float], ...],
    *,
    tolerance: float = 1.0e-12,
) -> tuple[tuple[tuple[float, float, float], float], ...]:
    unique: list[tuple[tuple[float, float, float], float]] = []
    for constraint in constraints:
        if not any(
            _halfspace_constraints_define_same_halfspace(
                constraint,
                existing,
                tolerance=tolerance,
            )
            for existing in unique
        ):
            unique.append(constraint)
    return tuple(unique)


def _halfspace_constraints_define_same_halfspace(
    first: tuple[tuple[float, float, float], float],
    second: tuple[tuple[float, float, float], float],
    *,
    tolerance: float,
) -> bool:
    normal1, rhs1 = first
    normal2, rhs2 = second
    vector1 = tuple(float(value) for value in (*normal1, rhs1))
    vector2 = tuple(float(value) for value in (*normal2, rhs2))
    normal_dot = sum(left * right for left, right in zip(vector1[:3], vector2[:3]))
    if normal_dot <= 0.0:
        return False
    return _vectors_proportional(vector1, vector2, tolerance=tolerance)


def _vectors_proportional(
    first: tuple[float, ...],
    second: tuple[float, ...],
    *,
    tolerance: float,
) -> bool:
    if len(first) != len(second):
        return False
    norm1 = max(abs(value) for value in first)
    norm2 = max(abs(value) for value in second)
    if norm1 <= tolerance or norm2 <= tolerance:
        return False
    scale = max(1.0, norm1, norm2)
    for i in range(len(first)):
        for j in range(i + 1, len(first)):
            if abs(first[i] * second[j] - first[j] * second[i]) > (
                tolerance * scale * scale
            ):
                return False
    return True


def _point3_distance(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(left, dtype=float) - np.asarray(right, dtype=float),
            ord=2,
        )
    )


def _polyhedron_volume_3d(
    vertices: tuple[tuple[float, float, float], ...],
    constraints: tuple[tuple[tuple[float, float, float], float], ...],
) -> float:
    if len(vertices) < 4 or not constraints:
        return 0.0
    points = [np.asarray(vertex, dtype=float) for vertex in vertices]
    interior = np.mean(np.asarray(points), axis=0)
    volume = 0.0
    for normal_tuple, rhs in constraints:
        normal = np.asarray(normal_tuple, dtype=float)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 0.0:
            continue
        tolerance = 1.0e-8 * max(1.0, normal_norm, abs(rhs))
        face_points = [
            point
            for point in points
            if abs(float(np.dot(normal, point)) - rhs) <= tolerance
        ]
        if len(face_points) < 3:
            continue
        center = np.mean(np.asarray(face_points), axis=0)
        unit_normal = normal / normal_norm
        reference = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(reference, unit_normal))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        axis_u = reference - unit_normal * float(np.dot(reference, unit_normal))
        axis_u = axis_u / float(np.linalg.norm(axis_u))
        axis_v = np.cross(unit_normal, axis_u)
        face_points = sorted(
            face_points,
            key=lambda point: np.arctan2(
                float(np.dot(point - center, axis_v)),
                float(np.dot(point - center, axis_u)),
            ),
        )
        for left, right in zip(face_points, (*face_points[1:], face_points[0])):
            tetra_volume = float(
                np.dot(center - interior, np.cross(left - interior, right - interior))
                / 6.0
            )
            volume += abs(tetra_volume)
    return float(volume)


def _affine_polyhedron_value_bounds(
    coefficients: tuple[float, ...],
    vertices: tuple[tuple[float, float, float], ...],
) -> tuple[float, float]:
    if len(coefficients) != 4 or not vertices:
        return (float("inf"), float("-inf"))
    c0, ax, ay, az = coefficients
    values = tuple(float(c0 + ax * x + ay * y + az * z) for x, y, z in vertices)
    radius = 128.0 * np.finfo(float).eps * max(
        1.0,
        *(abs(value) for value in values),
    )
    return (float(min(values) - radius), float(max(values) + radius))


def _affine_halfspace_3d_cell_plane_domain(
    *,
    cell: AffineHalfspaceArrangement3DCellCertificate,
    defining_spec: AffineBoxDecisionFunctionSpec,
    specs_by_id: Mapping[str, AffineBoxDecisionFunctionSpec],
    domain_box: tuple[tuple[float, float], ...],
    slab_half_width: float,
    tolerance: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[tuple[float, float], tuple[float, float]],
    float,
] | None:
    if len(defining_spec.coefficients) != 4 or len(domain_box) != 3:
        return None
    c0, ax, ay, az = defining_spec.coefficients
    normal = np.asarray((ax, ay, az), dtype=float)
    normal_norm_sq = float(np.dot(normal, normal))
    if normal_norm_sq <= 0.0 or not np.isfinite(normal_norm_sq):
        return None
    point_array = -float(c0) * normal / normal_norm_sq
    basis = _orthonormal_plane_basis_3d(normal)
    if basis is None:
        return None
    basis_u_array, basis_v_array = basis
    vertices = _plane_box_intersection_vertices_3d(
        defining_spec.coefficients,
        domain_box,
        tolerance=tolerance,
    )
    if len(vertices) < 3:
        return None
    parameter_vertices = _sort_parameter_polygon_vertices(
        tuple(
            (
                float(np.dot(np.asarray(vertex, dtype=float) - point_array, basis_u_array)),
                float(np.dot(np.asarray(vertex, dtype=float) - point_array, basis_v_array)),
            )
            for vertex in vertices
        )
    )
    if len(parameter_vertices) < 3:
        return None
    sign_by_decision: dict[str, str] = {}
    for item in cell.sign_vector:
        decision_id, sign = str(item).rsplit(":", 1)
        sign_by_decision[decision_id] = sign
    polygon = parameter_vertices
    for decision_id, sign in sign_by_decision.items():
        if decision_id in cell.defining_function_ids:
            continue
        spec = specs_by_id.get(decision_id)
        if spec is None or len(spec.coefficients) != 4:
            return None
        restricted = _restrict_affine_3d_decision_to_plane(
            spec,
            tuple(float(value) for value in point_array),  # type: ignore[arg-type]
            tuple(float(value) for value in basis_u_array),  # type: ignore[arg-type]
            tuple(float(value) for value in basis_v_array),  # type: ignore[arg-type]
        )
        if restricted is None:
            continue
        offset, slope_u, slope_v = restricted
        if sign == "-":
            polygon = _clip_polygon_halfplane(
                polygon,
                normal=(slope_u, slope_v),
                rhs=-slab_half_width - offset,
            )
        elif sign == "+":
            polygon = _clip_polygon_halfplane(
                polygon,
                normal=(-slope_u, -slope_v),
                rhs=offset - slab_half_width,
            )
        elif sign == "0":
            polygon = _clip_polygon_halfplane(
                polygon,
                normal=(slope_u, slope_v),
                rhs=slab_half_width - offset,
            )
            polygon = _clip_polygon_halfplane(
                polygon,
                normal=(-slope_u, -slope_v),
                rhs=offset + slab_half_width,
            )
        else:
            return None
        if len(polygon) < 3:
            return None
    parameter_vertices = _sort_parameter_polygon_vertices(polygon)
    area = _polygon_area(parameter_vertices)
    if area <= tolerance:
        return None
    u_values = tuple(value[0] for value in parameter_vertices)
    v_values = tuple(value[1] for value in parameter_vertices)
    radius = 64.0 * np.finfo(float).eps * max(
        1.0,
        *(abs(value) for point in parameter_vertices for value in point),
    )
    return (
        tuple(float(value) for value in point_array),  # type: ignore[return-value]
        tuple(float(value) for value in basis_u_array),  # type: ignore[return-value]
        tuple(float(value) for value in basis_v_array),  # type: ignore[return-value]
        (
            (float(min(u_values) - radius), float(max(u_values) + radius)),
            (float(min(v_values) - radius), float(max(v_values) + radius)),
        ),
        float(max(0.0, area - radius)),
    )


def _restrict_affine_3d_decision_to_plane(
    spec: AffineBoxDecisionFunctionSpec,
    point: tuple[float, float, float],
    basis_u: tuple[float, float, float],
    basis_v: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    if len(spec.coefficients) != 4:
        return None
    c0, ax, ay, az = spec.coefficients
    slope = np.asarray((ax, ay, az), dtype=float)
    point_array = np.asarray(point, dtype=float)
    basis_u_array = np.asarray(basis_u, dtype=float)
    basis_v_array = np.asarray(basis_v, dtype=float)
    return (
        float(c0 + np.dot(slope, point_array)),
        float(np.dot(slope, basis_u_array)),
        float(np.dot(slope, basis_v_array)),
    )


def _affine_halfspace_3d_plane_child_geometry(
    *,
    defining_spec: AffineBoxDecisionFunctionSpec,
    domain_box: tuple[tuple[float, float], ...],
    tolerance: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[tuple[float, float], tuple[float, float]],
    float,
] | None:
    if len(defining_spec.coefficients) != 4 or len(domain_box) != 3:
        return None
    c0, ax, ay, az = defining_spec.coefficients
    normal = np.asarray((ax, ay, az), dtype=float)
    normal_norm_sq = float(np.dot(normal, normal))
    if normal_norm_sq <= 0.0 or not np.isfinite(normal_norm_sq):
        return None
    point_array = -float(c0) * normal / normal_norm_sq
    basis = _orthonormal_plane_basis_3d(normal)
    if basis is None:
        return None
    basis_u_array, basis_v_array = basis
    vertices = _plane_box_intersection_vertices_3d(
        defining_spec.coefficients,
        domain_box,
        tolerance=tolerance,
    )
    if len(vertices) < 3:
        return None
    parameter_vertices = tuple(
        (
            float(np.dot(np.asarray(vertex, dtype=float) - point_array, basis_u_array)),
            float(np.dot(np.asarray(vertex, dtype=float) - point_array, basis_v_array)),
        )
        for vertex in vertices
    )
    sorted_parameter_vertices = _sort_parameter_polygon_vertices(parameter_vertices)
    area = _polygon_area(sorted_parameter_vertices)
    if area <= tolerance:
        return None
    u_values = tuple(value[0] for value in parameter_vertices)
    v_values = tuple(value[1] for value in parameter_vertices)
    radius = 64.0 * np.finfo(float).eps * max(
        1.0,
        *(abs(value) for point in parameter_vertices for value in point),
    )
    return (
        tuple(float(value) for value in point_array),  # type: ignore[return-value]
        tuple(float(value) for value in basis_u_array),  # type: ignore[return-value]
        tuple(float(value) for value in basis_v_array),  # type: ignore[return-value]
        (
            (float(min(u_values) - radius), float(max(u_values) + radius)),
            (float(min(v_values) - radius), float(max(v_values) + radius)),
        ),
        float(max(0.0, area - radius)),
    )


def _orthonormal_plane_basis_3d(
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0 or not np.isfinite(norm):
        return None
    unit_normal = normal / norm
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, unit_normal))) > 0.8:
        reference = np.array([0.0, 1.0, 0.0])
    basis_u = reference - unit_normal * float(np.dot(reference, unit_normal))
    basis_u_norm = float(np.linalg.norm(basis_u))
    if basis_u_norm <= 0.0 or not np.isfinite(basis_u_norm):
        return None
    basis_u = basis_u / basis_u_norm
    basis_v = np.cross(unit_normal, basis_u)
    basis_v_norm = float(np.linalg.norm(basis_v))
    if basis_v_norm <= 0.0 or not np.isfinite(basis_v_norm):
        return None
    return basis_u, basis_v / basis_v_norm


def _plane_box_intersection_vertices_3d(
    coefficients: tuple[float, ...],
    box: tuple[tuple[float, float], ...],
    *,
    tolerance: float,
) -> tuple[tuple[float, float, float], ...]:
    if len(coefficients) != 4 or len(box) != 3:
        return ()
    c0, ax, ay, az = coefficients
    corners = tuple(
        (float(x), float(y), float(z))
        for x in box[0]
        for y in box[1]
        for z in box[2]
    )

    def value(point: tuple[float, float, float]) -> float:
        return float(c0 + ax * point[0] + ay * point[1] + az * point[2])

    vertices: list[tuple[float, float, float]] = []

    def add_vertex(point: tuple[float, float, float]) -> None:
        if all(
            left - tolerance <= coordinate <= right + tolerance
            for coordinate, (left, right) in zip(point, box)
        ) and not any(_point3_distance(point, existing) <= 1.0e-8 for existing in vertices):
            vertices.append(point)

    for corner in corners:
        if abs(value(corner)) <= tolerance:
            add_vertex(corner)
    for left_index, left in enumerate(corners):
        for right in corners[left_index + 1:]:
            differing_axes = sum(
                abs(left[axis] - right[axis]) > tolerance for axis in range(3)
            )
            if differing_axes != 1:
                continue
            left_value = value(left)
            right_value = value(right)
            if abs(left_value) <= tolerance and abs(right_value) <= tolerance:
                add_vertex(left)
                add_vertex(right)
                continue
            if left_value * right_value > 0.0:
                continue
            denominator = left_value - right_value
            if abs(denominator) <= tolerance:
                continue
            ratio = left_value / denominator
            if -tolerance <= ratio <= 1.0 + tolerance:
                point = tuple(
                    float(left[axis] + ratio * (right[axis] - left[axis]))
                    for axis in range(3)
                )
                add_vertex(point)  # type: ignore[arg-type]
    return tuple(vertices)


def _sort_parameter_polygon_vertices(
    vertices: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    if len(vertices) < 3:
        return vertices
    center = (
        float(sum(vertex[0] for vertex in vertices) / len(vertices)),
        float(sum(vertex[1] for vertex in vertices) / len(vertices)),
    )
    return tuple(
        sorted(
            vertices,
            key=lambda vertex: np.arctan2(
                vertex[1] - center[1],
                vertex[0] - center[0],
            ),
        )
    )


def _affine_halfspace_3d_cell_line_domain(
    *,
    cell: AffineHalfspaceArrangement3DCellCertificate,
    first_spec: AffineBoxDecisionFunctionSpec,
    second_spec: AffineBoxDecisionFunctionSpec,
    specs_by_id: Mapping[str, AffineBoxDecisionFunctionSpec],
    domain_box: tuple[tuple[float, float], ...],
    slab_half_width: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float],
] | None:
    if (
        len(first_spec.coefficients) != 4
        or len(second_spec.coefficients) != 4
        or len(domain_box) != 3
    ):
        return None
    first_normal = np.asarray(first_spec.coefficients[1:], dtype=float)
    second_normal = np.asarray(second_spec.coefficients[1:], dtype=float)
    direction_array = np.cross(first_normal, second_normal)
    direction_norm = float(np.linalg.norm(direction_array))
    tolerance = 1.0e-12 * max(
        1.0,
        float(np.linalg.norm(first_normal)),
        float(np.linalg.norm(second_normal)),
    )
    if direction_norm <= tolerance:
        return None
    direction_array = direction_array / direction_norm
    matrix = np.asarray((first_normal, second_normal), dtype=float)
    rhs = -np.asarray(
        (first_spec.coefficients[0], second_spec.coefficients[0]),
        dtype=float,
    )
    try:
        point_array = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    if point_array.shape != (3,) or not np.all(np.isfinite(point_array)):
        return None
    point = tuple(float(value) for value in point_array)
    direction = tuple(float(value) for value in direction_array)
    interval = (-float("inf"), float("inf"))
    for coordinate, (lower, upper) in enumerate(domain_box):
        offset = point[coordinate]
        slope = direction[coordinate]
        interval = _intersect_affine_leq_interval(interval, offset, slope, upper)
        if interval is None:
            return None
        interval = _intersect_affine_leq_interval(interval, -offset, -slope, -lower)
        if interval is None:
            return None
    sign_by_decision: dict[str, str] = {}
    for item in cell.sign_vector:
        decision_id, sign = str(item).rsplit(":", 1)
        sign_by_decision[decision_id] = sign
    for decision_id, sign in sign_by_decision.items():
        if decision_id in cell.defining_function_ids:
            continue
        spec = specs_by_id.get(decision_id)
        if spec is None or len(spec.coefficients) != 4:
            return None
        restricted = _restrict_affine_3d_decision_to_line(spec, point, direction)
        if restricted is None:
            continue
        offset = restricted.coefficients[0]
        slope = restricted.coefficients[1]
        if sign == "-":
            interval = _intersect_affine_leq_interval(
                interval,
                offset,
                slope,
                -slab_half_width,
            )
        elif sign == "+":
            interval = _intersect_affine_leq_interval(
                interval,
                -offset,
                -slope,
                -slab_half_width,
            )
        elif sign == "0":
            interval = _intersect_affine_leq_interval(
                interval,
                offset,
                slope,
                slab_half_width,
            )
            if interval is not None:
                interval = _intersect_affine_leq_interval(
                    interval,
                    -offset,
                    -slope,
                    slab_half_width,
                )
        else:
            return None
        if interval is None:
            return None
    lower, upper = interval
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        return None
    return point, direction, (float(lower), float(upper))


def _restrict_affine_3d_decision_to_line(
    spec: AffineBoxDecisionFunctionSpec,
    point: tuple[float, float, float],
    direction: tuple[float, float, float],
    *,
    slope_tolerance: float = 1.0e-12,
) -> PolynomialDecisionFunctionSpec | None:
    if len(spec.coefficients) != 4:
        return None
    c0, ax, ay, az = spec.coefficients
    constant = float(c0 + ax * point[0] + ay * point[1] + az * point[2])
    slope = float(ax * direction[0] + ay * direction[1] + az * direction[2])
    if abs(slope) <= slope_tolerance:
        return None
    return PolynomialDecisionFunctionSpec(
        decision_id=f"{spec.decision_id}_on_spatial_line",
        coefficients=(constant, slope),
    )


def _affine_halfspace_arrangement_cell_line_domain(
    *,
    cell: AffineHalfspaceArrangementCellCertificate,
    defining_spec: AffineBoxDecisionFunctionSpec,
    specs_by_id: Mapping[str, AffineBoxDecisionFunctionSpec],
    domain_box: tuple[tuple[float, float], ...],
    slab_half_width: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    if len(defining_spec.coefficients) != 3 or len(domain_box) != 2:
        return None
    c0, ax, ay = defining_spec.coefficients
    normal_norm_sq = ax * ax + ay * ay
    if normal_norm_sq <= 0.0 or not np.isfinite(normal_norm_sq):
        return None
    normal_norm = float(np.sqrt(normal_norm_sq))
    point = (
        float(-c0 * ax / normal_norm_sq),
        float(-c0 * ay / normal_norm_sq),
    )
    direction = (float(-ay / normal_norm), float(ax / normal_norm))
    interval = (-float("inf"), float("inf"))
    for coordinate, (lower, upper) in enumerate(domain_box):
        offset = point[coordinate]
        slope = direction[coordinate]
        interval = _intersect_affine_leq_interval(interval, offset, slope, upper)
        if interval is None:
            return None
        interval = _intersect_affine_leq_interval(interval, -offset, -slope, -lower)
        if interval is None:
            return None
    sign_by_decision: dict[str, str] = {}
    for item in cell.sign_vector:
        decision_id, sign = str(item).rsplit(":", 1)
        sign_by_decision[decision_id] = sign
    for decision_id, sign in sign_by_decision.items():
        if decision_id in cell.defining_function_ids:
            continue
        spec = specs_by_id.get(decision_id)
        if spec is None or len(spec.coefficients) != 3:
            return None
        restricted = _restrict_affine_box_decision_to_line(spec, point, direction)
        if restricted is None:
            continue
        offset = restricted.coefficients[0]
        slope = restricted.coefficients[1]
        if sign == "-":
            interval = _intersect_affine_leq_interval(
                interval,
                offset,
                slope,
                -slab_half_width,
            )
        elif sign == "+":
            interval = _intersect_affine_leq_interval(
                interval,
                -offset,
                -slope,
                -slab_half_width,
            )
        elif sign == "0":
            interval = _intersect_affine_leq_interval(
                interval,
                offset,
                slope,
                slab_half_width,
            )
            if interval is not None:
                interval = _intersect_affine_leq_interval(
                    interval,
                    -offset,
                    -slope,
                    slab_half_width,
                )
        else:
            return None
        if interval is None:
            return None
    lower, upper = interval
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        return None
    return point, direction, (float(lower), float(upper))


def _restrict_affine_box_decision_to_line(
    spec: AffineBoxDecisionFunctionSpec,
    point: tuple[float, float],
    direction: tuple[float, float],
    *,
    slope_tolerance: float = 1.0e-12,
) -> PolynomialDecisionFunctionSpec | None:
    if len(spec.coefficients) != 3:
        return None
    c0, ax, ay = spec.coefficients
    constant = float(c0 + ax * point[0] + ay * point[1])
    slope = float(ax * direction[0] + ay * direction[1])
    if abs(slope) <= slope_tolerance:
        return None
    return PolynomialDecisionFunctionSpec(
        decision_id=f"{spec.decision_id}_on_line",
        coefficients=(constant, slope),
    )


def _affine_2d_boundaries_define_same_line(
    first: AffineBoxDecisionFunctionSpec,
    second: AffineBoxDecisionFunctionSpec,
    *,
    tolerance: float,
) -> bool:
    if len(first.coefficients) != 3 or len(second.coefficients) != 3:
        return False
    c1, ax1, ay1 = (float(value) for value in first.coefficients)
    c2, ax2, ay2 = (float(value) for value in second.coefficients)
    normal1_sq = ax1 * ax1 + ay1 * ay1
    normal2_sq = ax2 * ax2 + ay2 * ay2
    if normal1_sq <= tolerance * tolerance or normal2_sq <= tolerance * tolerance:
        return False
    minors = (
        (c1 * ax2 - c2 * ax1, abs(c1 * ax2) + abs(c2 * ax1)),
        (c1 * ay2 - c2 * ay1, abs(c1 * ay2) + abs(c2 * ay1)),
        (ax1 * ay2 - ay1 * ax2, abs(ax1 * ay2) + abs(ay1 * ax2)),
    )
    return all(abs(value) <= tolerance * max(1.0, scale) for value, scale in minors)


def _affine_3d_boundaries_define_same_plane(
    first: AffineBoxDecisionFunctionSpec,
    second: AffineBoxDecisionFunctionSpec,
    *,
    tolerance: float,
) -> bool:
    if len(first.coefficients) != 4 or len(second.coefficients) != 4:
        return False
    coefficients1 = tuple(float(value) for value in first.coefficients)
    coefficients2 = tuple(float(value) for value in second.coefficients)
    normal1_sq = sum(value * value for value in coefficients1[1:])
    normal2_sq = sum(value * value for value in coefficients2[1:])
    if normal1_sq <= tolerance * tolerance or normal2_sq <= tolerance * tolerance:
        return False
    return _vectors_proportional(coefficients1, coefficients2, tolerance=tolerance)


def _intersect_affine_leq_interval(
    interval: tuple[float, float],
    offset: float,
    slope: float,
    rhs: float,
    *,
    tolerance: float = 1.0e-12,
) -> tuple[float, float] | None:
    lower, upper = interval
    if abs(slope) <= tolerance:
        if offset <= rhs + tolerance:
            return interval
        return None
    bound = (rhs - offset) / slope
    if slope > 0.0:
        upper = min(upper, bound)
    else:
        lower = max(lower, bound)
    if lower >= upper:
        return None
    return (float(lower), float(upper))


def _solve_affine_two_boundary_point(
    first: AffineBoxDecisionFunctionSpec,
    second: AffineBoxDecisionFunctionSpec,
    *,
    determinant_tolerance: float = 1.0e-12,
) -> tuple[float, float] | None:
    if len(first.coefficients) != 3 or len(second.coefficients) != 3:
        return None
    c1, a1, b1 = first.coefficients
    c2, a2, b2 = second.coefficients
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) <= determinant_tolerance:
        return None
    x = (b1 * c2 - b2 * c1) / determinant
    y = (a2 * c1 - a1 * c2) / determinant
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return (float(x), float(y))


def _point_in_box(
    point: tuple[float, float],
    box: tuple[tuple[float, float], ...],
    *,
    tolerance: float,
) -> bool:
    if len(box) != 2:
        return False
    return all(
        left - tolerance <= coordinate <= right + tolerance
        for coordinate, (left, right) in zip(point, box)
    )


def _point_satisfies_affine_cell_signs(
    point: tuple[float, float],
    *,
    cell: AffineHalfspaceArrangementCellCertificate,
    specs_by_id: Mapping[str, AffineBoxDecisionFunctionSpec],
    slab_half_width: float,
    tolerance: float,
) -> bool:
    for item in cell.sign_vector:
        decision_id, sign = str(item).rsplit(":", 1)
        spec = specs_by_id.get(decision_id)
        if spec is None or len(spec.coefficients) != 3:
            return False
        c0, ax, ay = spec.coefficients
        value = c0 + ax * point[0] + ay * point[1]
        if sign == "0":
            if abs(value) > tolerance:
                return False
        elif sign == "+":
            if value < slab_half_width - tolerance:
                return False
        elif sign == "-":
            if value > -slab_half_width + tolerance:
                return False
        else:
            return False
    return True


def _solve_affine_three_boundary_point(
    specs: tuple[AffineBoxDecisionFunctionSpec, ...],
    *,
    determinant_tolerance: float = 1.0e-12,
) -> tuple[float, float, float] | None:
    if len(specs) != 3 or any(len(spec.coefficients) != 4 for spec in specs):
        return None
    matrix = np.asarray([spec.coefficients[1:] for spec in specs], dtype=float)
    rhs = -np.asarray([spec.coefficients[0] for spec in specs], dtype=float)
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) <= determinant_tolerance:
        return None
    try:
        point = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return None
    return tuple(float(value) for value in point)  # type: ignore[return-value]


def _point3_in_box(
    point: tuple[float, float, float],
    box: tuple[tuple[float, float], ...],
    *,
    tolerance: float,
) -> bool:
    if len(box) != 3:
        return False
    return all(
        left - tolerance <= coordinate <= right + tolerance
        for coordinate, (left, right) in zip(point, box)
    )


def _point3_satisfies_affine_cell_signs(
    point: tuple[float, float, float],
    *,
    cell: AffineHalfspaceArrangement3DCellCertificate,
    specs_by_id: Mapping[str, AffineBoxDecisionFunctionSpec],
    slab_half_width: float,
    tolerance: float,
) -> bool:
    for item in cell.sign_vector:
        decision_id, sign = str(item).rsplit(":", 1)
        spec = specs_by_id.get(decision_id)
        if spec is None or len(spec.coefficients) != 4:
            return False
        c0, ax, ay, az = spec.coefficients
        value = c0 + ax * point[0] + ay * point[1] + az * point[2]
        if sign == "0":
            if abs(value) > tolerance:
                return False
        elif sign == "+":
            if value < slab_half_width - tolerance:
                return False
        elif sign == "-":
            if value > -slab_half_width + tolerance:
                return False
        else:
            return False
    return True


def _axis_aligned_affine_box_child_consumption(
    *,
    stratum: AffineBoxDecisionStratumCertificate,
    fixed_coordinates: tuple[tuple[int, float, tuple[str, ...]], ...],
    free_domain_box: tuple[tuple[float, float], ...],
    root_rank: int | None,
) -> RecursiveStratifiedBranchEventConsumptionCertificate:
    child_dimension = len(free_domain_box)
    source_leaf = BranchEventTreeLeafCertificate(
        leaf_id=f"{stratum.stratum_id}:axis_child:leaf",
        leaf_type="axis_aligned_affine_box_child_terminal_leaf",
        decision="axis_aligned_affine_box_child_terminal",
        source_type="AxisAlignedAffineBoxChild",
        depth=0,
        certified=True,
        missing_obligations=(),
    )
    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="AxisAlignedAffineBoxChild",
        leaf_certificates=(source_leaf,),
        cover_certified=True,
        leaf_decisions_certified=True,
        equality_strata_explicit=False,
        obligations=(
            BranchEventTreeObligation(
                obligation="axis_aligned_affine_box_child_isolated",
                certified=True,
                detail=(
                    f"stratum_id={stratum.stratum_id}; "
                    f"fixed_coordinates={fixed_coordinates!r}; "
                    f"free_domain_box={free_domain_box!r}"
                ),
            ),
        ),
    )
    stratified_leaf = StratifiedBranchLeafCertificate(
        leaf_id=f"stratified:{source_leaf.leaf_id}",
        source_leaf_id=source_leaf.leaf_id,
        leaf_kind="no_event_before_target",
        terminal_response_kind="axis_aligned_affine_box_child_terminal",
        terminal_response_certified=True,
        source_leaf_certified=True,
        missing_obligations=(),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=(stratified_leaf,),
    )
    return certify_recursive_stratified_branch_event_consumption(
        stratified,
        root_dimension=child_dimension,
        root_rank=root_rank,
        recursion_kind="axis_aligned_affine_box_child",
    )


def _affine_halfspace_decision_child_consumption(
    *,
    cell: AffineHalfspaceDecisionCellCertificate,
    child_dimension: int,
    geometry_detail: str,
    root_rank: int | None,
) -> RecursiveStratifiedBranchEventConsumptionCertificate:
    source_leaf = BranchEventTreeLeafCertificate(
        leaf_id=f"{cell.cell_id}:halfspace_child:leaf",
        leaf_type="affine_halfspace_decision_child_terminal_leaf",
        decision="affine_halfspace_decision_child_terminal",
        source_type="AffineHalfspaceDecisionChild",
        depth=0,
        certified=True,
        missing_obligations=(),
    )
    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="AffineHalfspaceDecisionChild",
        leaf_certificates=(source_leaf,),
        cover_certified=True,
        leaf_decisions_certified=True,
        equality_strata_explicit=False,
        obligations=(
            BranchEventTreeObligation(
                obligation="affine_halfspace_decision_child_isolated",
                certified=True,
                detail=f"cell_id={cell.cell_id}; {geometry_detail}",
            ),
        ),
    )
    stratified_leaf = StratifiedBranchLeafCertificate(
        leaf_id=f"stratified:{source_leaf.leaf_id}",
        source_leaf_id=source_leaf.leaf_id,
        leaf_kind="no_event_before_target",
        terminal_response_kind="affine_halfspace_decision_child_terminal",
        terminal_response_certified=True,
        source_leaf_certified=True,
        missing_obligations=(),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=(stratified_leaf,),
    )
    return certify_recursive_stratified_branch_event_consumption(
        stratified,
        root_dimension=child_dimension,
        root_rank=root_rank,
        recursion_kind="affine_halfspace_decision_child",
    )


def _derive_polynomial_root_child_consumptions_from_strata(
    *,
    strata: tuple[PolynomialDecisionStratumCertificate, ...],
    source_id: str,
    child_root_rank: int | None,
) -> dict[str, RecursiveStratifiedBranchEventConsumptionCertificate]:
    children: dict[str, RecursiveStratifiedBranchEventConsumptionCertificate] = {}
    for stratum in strata:
        if not stratum.equality:
            continue
        child = _zero_dimensional_polynomial_root_consumption(
            source_id=source_id,
            stratum=stratum,
            root_rank=child_root_rank,
        )
        if child.proof_certified:
            children[stratum.stratum_id] = child
    return children


def _zero_dimensional_polynomial_root_consumption(
    *,
    source_id: str,
    stratum: PolynomialDecisionStratumCertificate,
    root_rank: int | None,
) -> RecursiveStratifiedBranchEventConsumptionCertificate:
    source_leaf = BranchEventTreeLeafCertificate(
        leaf_id=f"{stratum.stratum_id}:polynomial-root:leaf",
        leaf_type="polynomial_root_terminal_leaf",
        decision="zero_dimensional_polynomial_root_terminal",
        source_type="PolynomialRootChild",
        depth=0,
        certified=True,
        missing_obligations=(),
    )
    detail_parts = [
        f"source_id={source_id}",
        f"stratum_id={stratum.stratum_id}",
        f"stratum_kind={stratum.stratum_kind}",
        f"root_interval={stratum.interval!r}",
        f"value_interval={stratum.value_interval!r}",
    ]
    if stratum.derivative_interval is not None:
        detail_parts.append(f"derivative_interval={stratum.derivative_interval!r}")
    if stratum.second_derivative_interval is not None:
        detail_parts.append(
            f"second_derivative_interval={stratum.second_derivative_interval!r}"
        )
    if stratum.root_multiplicities:
        detail_parts.append(f"root_multiplicities={stratum.root_multiplicities!r}")
    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="PolynomialRootChild",
        leaf_certificates=(source_leaf,),
        cover_certified=True,
        leaf_decisions_certified=True,
        equality_strata_explicit=False,
        obligations=(
            BranchEventTreeObligation(
                obligation="zero_dimensional_polynomial_root_isolated",
                certified=stratum.proof_certified,
                detail="; ".join(detail_parts),
            ),
        ),
    )
    stratified_leaf = StratifiedBranchLeafCertificate(
        leaf_id=f"stratified:{source_leaf.leaf_id}",
        source_leaf_id=source_leaf.leaf_id,
        leaf_kind="no_event_before_target",
        terminal_response_kind="zero_dimensional_polynomial_root_terminal",
        terminal_response_certified=True,
        source_leaf_certified=True,
        missing_obligations=(),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=(stratified_leaf,),
    )
    return certify_recursive_stratified_branch_event_consumption(
        stratified,
        root_dimension=0,
        root_rank=root_rank,
        recursion_kind="polynomial_root_child",
    )


def _zero_dimensional_affine_point_consumption(
    *,
    cell: AffineHalfspaceArrangementCellCertificate,
    point: tuple[float, float],
    root_rank: int | None,
) -> RecursiveStratifiedBranchEventConsumptionCertificate:
    source_leaf = BranchEventTreeLeafCertificate(
        leaf_id=f"{cell.cell_id}:point:leaf",
        leaf_type="affine_halfspace_arrangement_point_terminal_leaf",
        decision="zero_dimensional_affine_point_terminal",
        source_type="AffineHalfspacePointChild",
        depth=0,
        certified=True,
        missing_obligations=(),
    )
    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="AffineHalfspacePointChild",
        leaf_certificates=(source_leaf,),
        cover_certified=True,
        leaf_decisions_certified=True,
        equality_strata_explicit=False,
        obligations=(
            BranchEventTreeObligation(
                obligation="zero_dimensional_affine_point_isolated",
                certified=True,
                detail=f"cell_id={cell.cell_id}; point={point!r}",
            ),
        ),
    )
    stratified_leaf = StratifiedBranchLeafCertificate(
        leaf_id=f"stratified:{source_leaf.leaf_id}",
        source_leaf_id=source_leaf.leaf_id,
        leaf_kind="no_event_before_target",
        terminal_response_kind="zero_dimensional_affine_point_terminal",
        terminal_response_certified=True,
        source_leaf_certified=True,
        missing_obligations=(),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=(stratified_leaf,),
    )
    return certify_recursive_stratified_branch_event_consumption(
        stratified,
        root_dimension=0,
        root_rank=root_rank,
        recursion_kind="affine_halfspace_point_child",
    )


def _one_dimensional_affine_line_consumption(
    *,
    cell: AffineHalfspaceArrangementCellCertificate,
    point: tuple[float, float],
    direction: tuple[float, float],
    domain: tuple[float, float],
    root_rank: int | None,
) -> RecursiveStratifiedBranchEventConsumptionCertificate:
    source_leaf = BranchEventTreeLeafCertificate(
        leaf_id=f"{cell.cell_id}:line:leaf",
        leaf_type="affine_halfspace_arrangement_line_terminal_leaf",
        decision="one_dimensional_affine_line_terminal",
        source_type="AffineHalfspaceLineChild",
        depth=0,
        certified=True,
        missing_obligations=(),
    )
    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="AffineHalfspaceLineChild",
        leaf_certificates=(source_leaf,),
        cover_certified=True,
        leaf_decisions_certified=True,
        equality_strata_explicit=False,
        obligations=(
            BranchEventTreeObligation(
                obligation="one_dimensional_affine_line_isolated",
                certified=True,
                detail=(
                    f"cell_id={cell.cell_id}; point={point!r}; "
                    f"direction={direction!r}; domain={domain!r}"
                ),
            ),
        ),
    )
    stratified_leaf = StratifiedBranchLeafCertificate(
        leaf_id=f"stratified:{source_leaf.leaf_id}",
        source_leaf_id=source_leaf.leaf_id,
        leaf_kind="no_event_before_target",
        terminal_response_kind="one_dimensional_affine_line_terminal",
        terminal_response_certified=True,
        source_leaf_certified=True,
        missing_obligations=(),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=(stratified_leaf,),
    )
    return certify_recursive_stratified_branch_event_consumption(
        stratified,
        root_dimension=1,
        root_rank=root_rank,
        recursion_kind="affine_halfspace_line_child",
    )


def _two_dimensional_affine_plane_consumption(
    *,
    cell: AffineHalfspaceArrangement3DCellCertificate,
    point: tuple[float, float, float],
    basis_u: tuple[float, float, float],
    basis_v: tuple[float, float, float],
    parameter_domain: tuple[tuple[float, float], tuple[float, float]],
    area_lower_bound: float,
    root_rank: int | None,
) -> RecursiveStratifiedBranchEventConsumptionCertificate:
    source_leaf = BranchEventTreeLeafCertificate(
        leaf_id=f"{cell.cell_id}:plane:leaf",
        leaf_type="affine_halfspace_3d_arrangement_plane_terminal_leaf",
        decision="two_dimensional_affine_plane_terminal",
        source_type="AffineHalfspacePlaneChild",
        depth=0,
        certified=True,
        missing_obligations=(),
    )
    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="AffineHalfspacePlaneChild",
        leaf_certificates=(source_leaf,),
        cover_certified=True,
        leaf_decisions_certified=True,
        equality_strata_explicit=False,
        obligations=(
            BranchEventTreeObligation(
                obligation="two_dimensional_affine_plane_isolated",
                certified=True,
                detail=(
                    f"cell_id={cell.cell_id}; point={point!r}; "
                    f"basis_u={basis_u!r}; basis_v={basis_v!r}; "
                    f"parameter_domain={parameter_domain!r}; "
                    f"area_lower_bound={area_lower_bound:g}"
                ),
            ),
        ),
    )
    stratified_leaf = StratifiedBranchLeafCertificate(
        leaf_id=f"stratified:{source_leaf.leaf_id}",
        source_leaf_id=source_leaf.leaf_id,
        leaf_kind="no_event_before_target",
        terminal_response_kind="two_dimensional_affine_plane_terminal",
        terminal_response_certified=True,
        source_leaf_certified=True,
        missing_obligations=(),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=(stratified_leaf,),
    )
    return certify_recursive_stratified_branch_event_consumption(
        stratified,
        root_dimension=2,
        root_rank=root_rank,
        recursion_kind="affine_halfspace_plane_child",
    )


def _one_dimensional_affine_spatial_line_consumption(
    *,
    cell: AffineHalfspaceArrangement3DCellCertificate,
    point: tuple[float, float, float],
    direction: tuple[float, float, float],
    domain: tuple[float, float],
    root_rank: int | None,
) -> RecursiveStratifiedBranchEventConsumptionCertificate:
    source_leaf = BranchEventTreeLeafCertificate(
        leaf_id=f"{cell.cell_id}:spatial-line:leaf",
        leaf_type="affine_halfspace_3d_arrangement_line_terminal_leaf",
        decision="one_dimensional_affine_spatial_line_terminal",
        source_type="AffineHalfspaceSpatialLineChild",
        depth=0,
        certified=True,
        missing_obligations=(),
    )
    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="AffineHalfspaceSpatialLineChild",
        leaf_certificates=(source_leaf,),
        cover_certified=True,
        leaf_decisions_certified=True,
        equality_strata_explicit=False,
        obligations=(
            BranchEventTreeObligation(
                obligation="one_dimensional_affine_spatial_line_isolated",
                certified=True,
                detail=(
                    f"cell_id={cell.cell_id}; point={point!r}; "
                    f"direction={direction!r}; domain={domain!r}"
                ),
            ),
        ),
    )
    stratified_leaf = StratifiedBranchLeafCertificate(
        leaf_id=f"stratified:{source_leaf.leaf_id}",
        source_leaf_id=source_leaf.leaf_id,
        leaf_kind="no_event_before_target",
        terminal_response_kind="one_dimensional_affine_spatial_line_terminal",
        terminal_response_certified=True,
        source_leaf_certified=True,
        missing_obligations=(),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=(stratified_leaf,),
    )
    return certify_recursive_stratified_branch_event_consumption(
        stratified,
        root_dimension=1,
        root_rank=root_rank,
        recursion_kind="affine_halfspace_spatial_line_child",
    )


def _zero_dimensional_affine_spatial_point_consumption(
    *,
    cell: AffineHalfspaceArrangement3DCellCertificate,
    point: tuple[float, float, float],
    root_rank: int | None,
) -> RecursiveStratifiedBranchEventConsumptionCertificate:
    source_leaf = BranchEventTreeLeafCertificate(
        leaf_id=f"{cell.cell_id}:spatial-point:leaf",
        leaf_type="affine_halfspace_3d_arrangement_point_terminal_leaf",
        decision="zero_dimensional_affine_spatial_point_terminal",
        source_type="AffineHalfspaceSpatialPointChild",
        depth=0,
        certified=True,
        missing_obligations=(),
    )
    source_tree = BranchEventTreeCertificate(
        tree_kind="ambiguous_event_order_partition",
        source_type="AffineHalfspaceSpatialPointChild",
        leaf_certificates=(source_leaf,),
        cover_certified=True,
        leaf_decisions_certified=True,
        equality_strata_explicit=False,
        obligations=(
            BranchEventTreeObligation(
                obligation="zero_dimensional_affine_spatial_point_isolated",
                certified=True,
                detail=f"cell_id={cell.cell_id}; point={point!r}",
            ),
        ),
    )
    stratified_leaf = StratifiedBranchLeafCertificate(
        leaf_id=f"stratified:{source_leaf.leaf_id}",
        source_leaf_id=source_leaf.leaf_id,
        leaf_kind="no_event_before_target",
        terminal_response_kind="zero_dimensional_affine_spatial_point_terminal",
        terminal_response_certified=True,
        source_leaf_certified=True,
        missing_obligations=(),
    )
    stratified = certify_stratified_branch_event_tree(
        source_tree,
        leaf_certificates=(stratified_leaf,),
    )
    return certify_recursive_stratified_branch_event_consumption(
        stratified,
        root_dimension=0,
        root_rank=root_rank,
        recursion_kind="affine_halfspace_spatial_point_child",
    )


def _normalize_child_consumptions(
    child_consumptions: Mapping[
        str,
        RecursiveStratifiedBranchEventConsumptionCertificate,
    ]
    | tuple[
        tuple[str, RecursiveStratifiedBranchEventConsumptionCertificate],
        ...,
    ],
) -> dict[str, RecursiveStratifiedBranchEventConsumptionCertificate]:
    if isinstance(child_consumptions, Mapping):
        items = child_consumptions.items()
    else:
        items = child_consumptions
    return {str(leaf_id): child for leaf_id, child in items}


def _normalize_arrangement_child_consumptions(
    arrangement: PolynomialDecisionArrangementStratificationCertificate,
    child_consumptions: Mapping[
        str,
        RecursiveStratifiedBranchEventConsumptionCertificate,
    ]
    | tuple[
        tuple[str, RecursiveStratifiedBranchEventConsumptionCertificate],
        ...,
    ],
) -> dict[str, RecursiveStratifiedBranchEventConsumptionCertificate]:
    normalized = _normalize_child_consumptions(child_consumptions)
    aliases: dict[str, str] = {}
    for leaf in arrangement.stratified_tree.leaf_certificates:
        aliases[leaf.leaf_id] = leaf.leaf_id
        aliases[leaf.source_leaf_id] = leaf.leaf_id
        aliases[leaf.source_leaf_id.removesuffix(":leaf")] = leaf.leaf_id
        if leaf.equality_stratum is not None:
            aliases[leaf.equality_stratum.stratum_id] = leaf.leaf_id
            aliases[f"{leaf.equality_stratum.stratum_id}:leaf"] = leaf.leaf_id
            aliases[f"stratified:{leaf.equality_stratum.stratum_id}:leaf"] = (
                leaf.leaf_id
            )
    return {
        aliases.get(str(leaf_id), str(leaf_id)): child
        for leaf_id, child in normalized.items()
    }


def _consume_recursive_stratified_leaf(
    leaf: StratifiedBranchLeafCertificate,
    *,
    root_dimension: int,
    root_rank: int | None,
    child: RecursiveStratifiedBranchEventConsumptionCertificate | None,
) -> RecursiveStratifiedLeafConsumption:
    missing: list[str] = []
    terminal_certified = bool(
        leaf.leaf_kind in TERMINAL_STRATIFIED_LEAF_KINDS
        and leaf.proof_certified
    )
    child_certified = bool(child is not None and child.certified)
    descent_certified = bool(
        child is not None
        and _strict_dimension_or_rank_descent(
            parent_dimension=root_dimension,
            parent_rank=root_rank,
            child=child,
        )
    )
    consumption_kind = (
        "terminal_stratified_leaf"
        if terminal_certified
        else (
            "lower_dimensional_recursive_stratum"
            if child is not None
            else "unresolved_stratified_leaf"
        )
    )
    if leaf.unsupported:
        missing.append("unsupported_analytic_stratum")
    if child is not None and not leaf.equality_or_zero_margin:
        missing.append("child_consumption_requires_equality_or_zero_margin_leaf")
    if child is not None and not child_certified:
        missing.append("child_consumption_not_certified")
    if child is not None and not descent_certified:
        missing.append("child_stratum_not_strictly_lower_dimension_or_rank")
    if not terminal_certified and child is None:
        missing.append("terminal_or_recursive_response_missing")
    missing.extend(str(item) for item in leaf.missing_obligations)
    return RecursiveStratifiedLeafConsumption(
        leaf_id=leaf.leaf_id,
        leaf_kind=leaf.leaf_kind,
        consumption_kind=consumption_kind,
        terminal_certified=terminal_certified,
        child_certified=child_certified,
        descent_certified=descent_certified,
        child_dimension=(child.root_dimension if child is not None else None),
        child_rank=(child.root_rank if child is not None else None),
        child_source_type=(
            str(getattr(child, "constructor_source_type", ""))
            if child is not None
            else ""
        ),
        missing_obligations=tuple(dict.fromkeys(missing)),
    )


def _strict_dimension_or_rank_descent(
    *,
    parent_dimension: int,
    parent_rank: int | None,
    child: RecursiveStratifiedBranchEventConsumptionCertificate,
) -> bool:
    if child.root_dimension < parent_dimension:
        return True
    if child.root_dimension != parent_dimension:
        return False
    return bool(
        parent_rank is not None
        and child.root_rank is not None
        and child.root_rank < parent_rank
    )

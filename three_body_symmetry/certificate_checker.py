"""Independent checker for serialized three-body proof certificates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, DecimalException, ROUND_CEILING, ROUND_FLOOR, localcontext
from functools import lru_cache
from fractions import Fraction
import math
from typing import Iterable

import numpy as np

from .certificate_language import (
    BranchUnionCertificate,
    ChartChainCertificate,
    EventIsolationCertificate,
    FiniteFuchsianLogPrimitiveCauchyInputsCertificate,
    GeneralizedFuchsianRemainderMajorantCertificate,
    FuchsianLogTermCertificate,
    InitialValueProblemBindingCertificate,
    OrdinaryChartTransitionCertificate,
    OrdinaryAposterioriTubeCertificate,
    WeightedOrdinaryAposterioriTubeCertificate,
    OrdinaryEnclosureTransitionCertificate,
    OrdinaryToPlanarLCEnclosureTransitionCertificate,
    PlanarLCToOrdinaryEnclosureTransitionCertificate,
    PlanarLCExactOverlapAnchorCertificate,
    PlanarLCExactGaugeAtlasCertificate,
    PlanarLCExactCollisionAnchorCertificate,
    PlanarLCTwoSidedCollisionPassageCertificate,
    ValidatedOrdinaryIVPChainCertificate,
    OrdinaryTaylorChartCertificate,
    PlanarLeviCivitaBinaryChartCertificate,
    PlanarLCAposterioriTubeCertificate,
    PlanarLeviCivitaTransitionCertificate,
    PrimitiveCauchyTailInputCertificate,
    SpatialKSBinaryChartCertificate,
    SpatialKSTransitionCertificate,
    TotalCollisionFuchsianStopChartCertificate,
    TotalCollisionGeneralizedFuchsianStopChartCertificate,
    total_collision_generalized_fuchsian_stop_chart_certificate_from_branch,
)
from .lc_gauge_gluing import (
    PlanarLCGaugeGluingCertificate,
    PlanarLCGaugeGluingCheckResult,
    PlanarLCGaugeGluingObligation,
    PlanarLCGaugeOverlapEdge,
    check_planar_lc_gauge_gluing,
)
from .binary_chart import (
    RegularizedBinaryCollisionChartState,
    planar_accelerations_from_regularized_chart_rhs,
    regularized_binary_collision_chart_rhs,
    regularized_binary_collision_chart_to_planar,
    planar_interval_to_regularized_binary_collision_chart_atlas,
)
from .binary_series import (
    RegularizedBinaryTaylorSolution,
    pair_energy_constraint_coefficients as lc_pair_energy_constraint_coefficients,
    regularized_rhs_coefficients as lc_regularized_rhs_coefficients,
)
from .dynamics import accelerations
from .event_recurrence import PrimitiveCauchyTailInput
from .fuchsian import (
    FiniteFuchsianLogBranch,
    FiniteFuchsianLogTotalCollisionIsolationCertificate,
    FuchsianShapeBranch,
    FuchsianLogTerm,
    _finite_fuchsian_log_branch_derivative_envelope,
    certify_finite_fuchsian_log_total_collision_isolation,
    construct_fuchsian_shape_branch,
)
from .intervals import (
    FloatInterval,
    RationalInterval,
    interval_array_derivative_coefficients,
    interval_array_series_eval,
    interval_polyder,
    interval_polynomial_eval,
    interval_sign,
    rational_interval_polyder,
    rational_interval_polynomial_eval,
    rational_interval_sign,
)
from .ks_binary_chart import (
    SpatialKSBinaryChartState,
    ks_binary_chart_to_spatial,
    regularized_ks_binary_chart_rhs,
    spatial_accelerations_from_ks_binary_rhs,
)
from .ks_binary_series import (
    SpatialKSBinaryTaylorSolution,
    ks_horizontal_constraint_coefficients,
    ks_pair_energy_constraint_coefficients,
    regularized_rhs_coefficients as ks_regularized_rhs_coefficients,
)
from .series import acceleration_coefficients
from .stratified_branch_tree import SUPPORTED_STRATIFIED_LEAF_KINDS


@lru_cache(maxsize=2)
def _fast_generalized_fuchsian_branch(max_total_degree: int = 2):
    from .fuchsian import (
        construct_fuchsian_selector_continuation,
        linearized_acceleration_matrix,
        mass_inner_product,
    )

    max_total_degree = int(max_total_degree)
    masses = np.array([1.0, 0.7, 1.4])
    configuration = np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ],
    )
    configuration = configuration - np.average(configuration, axis=0, weights=masses)
    central_lambda = float(np.sum(masses) / (np.sqrt(3.0) ** 3))
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    derivative_matrix = linearized_acceleration_matrix(central_shape, masses)
    beta = float(
        sum(masses[i] * masses[j] for i in range(3) for j in range(i + 1, 3))
        / np.sum(masses) ** 2
    )
    shape_eigenvalues = (
        1.0 / 9.0 + (1.0 / 3.0) * np.sqrt(1.0 - 3.0 * beta),
        1.0 / 9.0 - (1.0 / 3.0) * np.sqrt(1.0 - 3.0 * beta),
    )
    powers = (
        2.0,
        *(
            0.5 * (-1.0 + np.sqrt(9.0 + 36.0 * eigenvalue))
            for eigenvalue in shape_eigenvalues
        ),
    )
    eigenvalues, eigenvectors = np.linalg.eig(derivative_matrix)
    fractional_modes = []
    for eigenvalue in shape_eigenvalues:
        eigenvector_index = int(
            np.argmin(
                np.abs(eigenvalues.real - eigenvalue)
                + np.abs(eigenvalues.imag)
            ),
        )
        mode = eigenvectors[:, eigenvector_index].real.reshape(3, 2)
        mode /= np.sqrt(mass_inner_product(masses, mode, mode))
        fractional_modes.append(mode)
    continuation = construct_fuchsian_selector_continuation(
        masses=masses,
        central_shape=central_shape,
        powers=powers,
        incoming_selected_coefficients={
            (1, 0, 0): 0.02 * central_shape,
            (0, 1, 0): -0.018 * fractional_modes[0],
            (0, 0, 1): 0.022 * fractional_modes[1],
        },
        max_total_degree=max_total_degree,
        scale_index=(1, 0, 0),
    )
    return continuation.incoming


def _generalized_remainder_majorant_certificate_from_majorant(
    majorant,
) -> GeneralizedFuchsianRemainderMajorantCertificate:
    return GeneralizedFuchsianRemainderMajorantCertificate(
        initial_radius=float(majorant.initial_radius),
        shell_contraction=float(majorant.shell_contraction),
        analytic_disk_fraction=float(majorant.analytic_disk_fraction),
        defect_bound=float(majorant.defect_bound),
        linear_inverse_bound=float(majorant.linear_inverse_bound),
        nonlinear_lipschitz_bound=float(majorant.nonlinear_lipschitz_bound),
        remainder_ball_radius=float(majorant.remainder_ball_radius),
        component_effective_exponents=tuple(
            (component, float(exponent))
            for component, exponent in sorted(
                majorant.component_effective_exponents.items(),
            )
        ),
        component_inputs=tuple(
            (component, PrimitiveCauchyTailInputCertificate.from_input(input_))
            for component, input_ in sorted(majorant.component_inputs.items())
        ),
    )


@lru_cache(maxsize=1)
def build_fast_total_collision_generalized_fuchsian_stop_chart_certificate(
) -> TotalCollisionGeneralizedFuchsianStopChartCertificate:
    """Build the canonical fast generalized-Fuchsian total-stop certificate.

    This production fixture is intentionally small but still routes through the
    real generalized-Fuchsian entry, finite-row, majorant, stop-chart, and
    serialized-chart constructors.  It is used by the final theorem package so
    production proof assembly never imports from tests.
    """

    from .finite_target_completeness import (
        certify_supplied_generalized_fuchsian_analytic_remainder_majorant,
        certify_supplied_generalized_fuchsian_entry_data,
        certify_supplied_generalized_fuchsian_finite_row_tail_budget,
        certify_supplied_generalized_fuchsian_stop_chart_for_admissible_entry_data,
    )

    branch = _fast_generalized_fuchsian_branch(2)
    entry = certify_supplied_generalized_fuchsian_entry_data(
        branch=branch,
        radius=0.02,
        sample_taus=(-0.012, 0.012),
        tolerance=1.0e-4,
        energy_tolerance=1.0e-7,
    )
    finite_rows = certify_supplied_generalized_fuchsian_finite_row_tail_budget(
        entry_certificate=entry,
        retained_total_degree=branch.max_total_degree,
        radius=0.018,
    )
    majorant = certify_supplied_generalized_fuchsian_analytic_remainder_majorant(
        entry_certificate=entry,
        finite_row_budget=finite_rows,
        defect_bound=1.0e-14,
        linear_inverse_bound=2.0,
        nonlinear_lipschitz_bound=0.1,
        component_effective_exponents={
            "value": 1.4,
            "first_jet": 0.4,
            "lifted_residual": 1.1,
            "physical_residual": 0.7,
            "regularized_position_value": 2.1,
        },
        step_ratio_bounds={
            "value": 0.2,
            "first_jet": 0.2,
            "lifted_residual": 0.2,
            "physical_residual": 0.2,
            "regularized_position_value": 0.2,
        },
        retained_order_initials={
            "value": 6,
            "first_jet": 6,
            "lifted_residual": 6,
            "physical_residual": 6,
            "regularized_position_value": 6,
        },
        retained_order_increments={
            "value": 1,
            "first_jet": 1,
            "lifted_residual": 1,
            "physical_residual": 1,
            "regularized_position_value": 1,
        },
        initial_radius=0.014,
        shell_contraction=0.5,
        analytic_disk_fraction=0.2,
    )
    stop_chart = certify_supplied_generalized_fuchsian_stop_chart_for_admissible_entry_data(
        entry_certificate=entry,
        finite_row_budget=finite_rows,
        remainder_majorant=majorant,
        residual_tolerance=1.0e-4,
        angular_momentum_tolerance=1.0e-4,
    )
    certificate = total_collision_generalized_fuchsian_stop_chart_certificate_from_branch(
        branch,
        certificate_id="fast-generalized-fuchsian-total-stop-certificate",
        chart_id="fast-generalized-fuchsian-total-stop",
        isolation_radius=entry.radius,
        central_shape_pair_distance_floor=entry.central_shape_pair_distance_floor,
        shape_deviation_bound=entry.shape_deviation_bound,
        shape_pair_distance_floor=entry.shape_pair_distance_floor,
        tau_interval=stop_chart.tau_interval,
        residual_tolerance=stop_chart.residual_tolerance,
        angular_momentum_tolerance=stop_chart.angular_momentum_tolerance,
        tail_bound=stop_chart.tail_bound,
        sample_count=3,
        remainder_majorant=(
            _generalized_remainder_majorant_certificate_from_majorant(majorant)
        ),
    )
    return replace(
        certificate,
        residual_tolerance=1.0e-5,
        projected_residual_tolerance=2.0e4,
    )


@dataclass(frozen=True)
class CertificateCheckObligation:
    """One checker obligation for a serialized certificate."""

    obligation: str
    certified: bool
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "certified", self.certified is True)


def _check_obligation_ledger_certified(
    obligations: tuple[object, ...],
) -> bool:
    return bool(
        obligations
        and all(
            isinstance(obligation, CertificateCheckObligation)
            and obligation.certified is True
            for obligation in obligations
        )
    )


def _check_obligation_ledger_missing(
    obligations: tuple[object, ...],
    *,
    ledger_name: str,
) -> tuple[str, ...]:
    missing: list[str] = []
    if not obligations:
        missing.append(f"{ledger_name}_obligations_present")
    for obligation in obligations:
        if not isinstance(obligation, CertificateCheckObligation):
            missing.append(f"{ledger_name}_obligation_type")
            continue
        if obligation.certified is not True:
            missing.append(obligation.obligation)
    return tuple(dict.fromkeys(missing))


_PLANAR_LC_EXACT_OVERLAP_OBLIGATION_NAMES = (
    "planar_lc_exact_overlap_identifiers_match",
    "planar_lc_exact_overlap_distinct_charts_and_tubes",
    "planar_lc_exact_overlap_anchor_parameters_match",
    "planar_lc_exact_overlap_zero_error_centers",
    "planar_lc_exact_overlap_masses_and_ordered_pair_match",
    "planar_lc_exact_overlap_source_tube_certified",
    "planar_lc_exact_overlap_target_tube_certified",
    "planar_lc_exact_overlap_common_interval_nondegenerate",
    "planar_lc_exact_overlap_exact_anchor_schema",
    "planar_lc_exact_overlap_fixed_components_equal",
    "planar_lc_exact_overlap_unique_deck_relation",
)

_PLANAR_LC_EXACT_GAUGE_ATLAS_OBLIGATION_NAMES = (
    "planar_lc_exact_gauge_atlas_manifest_schema",
    "planar_lc_exact_gauge_atlas_raw_ids_unique_and_positional",
    "planar_lc_exact_gauge_atlas_vertex_bindings_match",
    "planar_lc_exact_gauge_atlas_vertex_tubes_zero_error_certified",
    "planar_lc_exact_gauge_atlas_global_problem_matches",
    "planar_lc_exact_gauge_atlas_overlap_bindings_match_vertices",
    "planar_lc_exact_gauge_atlas_overlaps_recomputed_and_certified",
    "planar_lc_exact_gauge_atlas_derived_edges_match_results",
    "planar_lc_exact_gauge_atlas_derived_graph_certified",
)


def _exact_certified_obligation_manifest(
    obligations: object,
    expected_names: tuple[str, ...],
) -> bool:
    return bool(
        type(obligations) is tuple
        and type(expected_names) is tuple
        and all(type(name) is str and bool(name) for name in expected_names)
        and len(obligations) == len(expected_names)
        and all(
            type(obligation) is CertificateCheckObligation
            and type(obligation.obligation) is str
            and bool(obligation.obligation)
            and type(obligation.certified) is bool
            and obligation.certified is True
            and type(obligation.detail) is str
            for obligation in obligations
        )
        and tuple(
            obligation.obligation
            for obligation in obligations
        )
        == expected_names
    )


@dataclass(frozen=True)
class CertificateCheckResult:
    """Result of checking one serialized certificate."""

    certificate_id: str
    certificate_type: str
    checker_id: str
    obligations: tuple[CertificateCheckObligation, ...]
    max_coefficient_residual: float
    max_sampled_newton_residual: float

    @property
    def certified(self) -> bool:
        return _check_obligation_ledger_certified(self.obligations)

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _check_obligation_ledger_missing(
            self.obligations,
            ledger_name=self.certificate_id or "certificate_check",
        )


@dataclass(frozen=True)
class TransitionCheckResult:
    """Result of checking one serialized chart transition."""

    transition_id: str
    transition_type: str
    checker_id: str
    obligations: tuple[CertificateCheckObligation, ...]
    max_position_gap: float
    max_velocity_gap: float

    @property
    def certified(self) -> bool:
        return _check_obligation_ledger_certified(self.obligations)

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _check_obligation_ledger_missing(
            self.obligations,
            ledger_name=self.transition_id or "transition_check",
        )


@dataclass(frozen=True)
class InitialValueBindingCheckResult:
    """Result of anchoring one checked chart to serialized IVP data."""

    binding_id: str
    chart_id: str
    checker_id: str
    obligations: tuple[CertificateCheckObligation, ...]
    max_position_gap: float
    max_velocity_gap: float
    time_gap: float

    @property
    def certified(self) -> bool:
        return _check_obligation_ledger_certified(self.obligations)

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _check_obligation_ledger_missing(
            self.obligations,
            ledger_name=self.binding_id or "initial_value_binding_check",
        )


@dataclass(frozen=True)
class OrdinaryAposterioriTubeCheckResult:
    """Checked defect-to-solution enclosure data for an ordinary chart."""

    tube_id: str
    chart_id: str
    checker_id: str
    obligations: tuple[CertificateCheckObligation, ...]
    defect_bound: float
    lipschitz_bound: float
    gronwall_error_bound: float
    nominal_pair_distance_floor: float
    tube_pair_distance_floor: float

    @property
    def certified(self) -> bool:
        return _check_obligation_ledger_certified(self.obligations)

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _check_obligation_ledger_missing(
            self.obligations,
            ledger_name=self.tube_id or "ordinary_aposteriori_tube_check",
        )


@dataclass(frozen=True)
class WeightedOrdinaryAposterioriTubeCheckResult:
    """A-posteriori Newton enclosure in a block-weighted infinity norm."""

    tube_id: str
    chart_id: str
    checker_id: str
    obligations: tuple[CertificateCheckObligation, ...]
    position_defect_bound: float
    velocity_defect_bound: float
    scaled_defect_bound: float
    acceleration_lipschitz_bound: float
    scaled_lipschitz_bound: float
    normalized_gronwall_error_bound: float
    proven_position_error_bound: float
    proven_velocity_error_bound: float
    tube_pair_distance_floor: float

    @property
    def certified(self) -> bool:
        return _check_obligation_ledger_certified(self.obligations)

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _check_obligation_ledger_missing(
            self.obligations,
            ledger_name=self.tube_id or "weighted_ordinary_aposteriori_tube",
        )


@dataclass(frozen=True)
class ValidatedOrdinaryIVPChartCheckResult:
    """Aggregate exact-solution enclosure check for one ordinary IVP chart."""

    binding_result: InitialValueBindingCheckResult
    tube_result: OrdinaryAposterioriTubeCheckResult
    chart_result: CertificateCheckResult
    chart_serialization_admissible: bool
    checker_id: str
    obligations: tuple[CertificateCheckObligation, ...]

    @property
    def certified(self) -> bool:
        return bool(
            type(self.binding_result) is InitialValueBindingCheckResult
            and self.binding_result.certified
            and type(self.tube_result) is OrdinaryAposterioriTubeCheckResult
            and self.tube_result.certified
            and type(self.chart_result) is CertificateCheckResult
            and self.chart_serialization_admissible is True
            and _check_obligation_ledger_certified(self.obligations)
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _check_obligation_ledger_missing(
                self.obligations,
                ledger_name="validated_ordinary_ivp_chart",
            )
        )
        missing.extend(self.binding_result.missing_obligations)
        missing.extend(self.tube_result.missing_obligations)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class OrdinaryEnclosureTransitionCheckResult:
    """Validated continuation from an exact source tube into a target tube."""

    transition_id: str
    checker_id: str
    target_chart_result: CertificateCheckResult
    target_tube_result: OrdinaryAposterioriTubeCheckResult
    obligations: tuple[CertificateCheckObligation, ...]
    polynomial_state_gap: float
    required_target_initial_error: float
    time_gap: float

    @property
    def certified(self) -> bool:
        return bool(
            self.target_chart_result.certified
            and self.target_tube_result.certified
            and _check_obligation_ledger_certified(self.obligations)
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _check_obligation_ledger_missing(
                self.obligations,
                ledger_name=self.transition_id or "ordinary_enclosure_transition",
            )
        )
        missing.extend(self.target_chart_result.missing_obligations)
        missing.extend(self.target_tube_result.missing_obligations)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class ValidatedOrdinaryIVPChainCheckResult:
    """Finite induction certificate for one exact ordinary Newtonian branch."""

    chain_id: str
    checker_id: str
    first_chart_result: ValidatedOrdinaryIVPChartCheckResult
    transition_results: tuple[OrdinaryEnclosureTransitionCheckResult, ...]
    obligations: tuple[CertificateCheckObligation, ...]
    covered_physical_time_interval: tuple[float, float]
    target_time: float
    target_position_intervals: tuple[tuple[tuple[float, float], ...], ...]
    target_velocity_intervals: tuple[tuple[tuple[float, float], ...], ...]

    @property
    def certified(self) -> bool:
        return bool(
            self.first_chart_result.certified
            and all(
                type(result) is OrdinaryEnclosureTransitionCheckResult
                and result.certified
                for result in self.transition_results
            )
            and _check_obligation_ledger_certified(self.obligations)
        )

    @property
    def exact_ivp_enclosure_certified(self) -> bool:
        return self.certified

    @property
    def target_state_enclosure_certified(self) -> bool:
        return bool(
            self.certified
            and np.isfinite(self.target_time)
            and self.target_position_intervals
            and self.target_velocity_intervals
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _check_obligation_ledger_missing(
                self.obligations,
                ledger_name=self.chain_id or "validated_ordinary_ivp_chain",
            )
        )
        missing.extend(self.first_chart_result.missing_obligations)
        for result in self.transition_results:
            missing.extend(result.missing_obligations)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class PlanarLCAposterioriTubeCheckResult:
    """Exact lifted-solution enclosure check for a planar LC chart."""

    tube_id: str
    chart_id: str
    checker_id: str
    obligations: tuple[CertificateCheckObligation, ...]
    defect_bound: float
    lipschitz_bound: float
    gronwall_error_bound: float
    third_body_distance_floor: float
    anchor_pair_energy_constraint_residual: float
    pair_energy_constraint_anchor_certified: bool
    anchor_is_polynomial_center: bool

    @property
    def certified(self) -> bool:
        return _check_obligation_ledger_certified(self.obligations)

    @property
    def lifted_exact_solution_enclosure_certified(self) -> bool:
        return self.certified

    @property
    def constrained_newtonian_lift_certified(self) -> bool:
        """Whether the particular center-anchored IVP is constraint preserving.

        A positive initial-error ball also contains off-constraint anchors, so
        an exact constraint check at the polynomial center cannot promote the
        whole tube to a constrained family.
        """
        return bool(
            self.certified
            and self.pair_energy_constraint_anchor_certified
            and self.anchor_is_polynomial_center
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _check_obligation_ledger_missing(
            self.obligations,
            ledger_name=self.tube_id or "planar_lc_aposteriori_tube",
        )


def _rejected_planar_lc_tube_result(
    tube: PlanarLCAposterioriTubeCertificate,
    chart: PlanarLeviCivitaBinaryChartCertificate,
) -> PlanarLCAposterioriTubeCheckResult:
    """Return a deterministic rejection for malformed exact-class inputs."""

    return PlanarLCAposterioriTubeCheckResult(
        tube_id=tube.tube_id if type(tube.tube_id) is str else "",
        chart_id=chart.chart_id if type(chart.chart_id) is str else "",
        checker_id="independent_planar_lc_aposteriori_tube_checker_v1",
        obligations=(
            CertificateCheckObligation(
                "planar_lc_tube_malformed_exact_class_input",
                False,
                "malformed exact-class chart or tube fields were rejected",
            ),
        ),
        defect_bound=np.inf,
        lipschitz_bound=np.inf,
        gronwall_error_bound=np.inf,
        third_body_distance_floor=0.0,
        anchor_pair_energy_constraint_residual=np.inf,
        pair_energy_constraint_anchor_certified=False,
        anchor_is_polynomial_center=False,
    )


def _check_planar_lc_tube_or_reject_malformed(
    tube: PlanarLCAposterioriTubeCertificate,
    chart: PlanarLeviCivitaBinaryChartCertificate,
) -> PlanarLCAposterioriTubeCheckResult:
    try:
        return check_planar_lc_aposteriori_tube(tube, chart)
    except (
        AttributeError,
        DecimalException,
        FloatingPointError,
        IndexError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ):
        return _rejected_planar_lc_tube_result(tube, chart)


@dataclass(frozen=True)
class PlanarLCExactOverlapAnchorCheckResult:
    """Exact deck relation between two independently checked LC IVPs."""

    overlap_id: str
    checker_id: str
    source_chart_id: str
    target_chart_id: str
    raw_overlap_certificate: PlanarLCExactOverlapAnchorCertificate
    raw_source_tube: PlanarLCAposterioriTubeCertificate
    raw_source_chart: PlanarLeviCivitaBinaryChartCertificate
    raw_target_tube: PlanarLCAposterioriTubeCertificate
    raw_target_chart: PlanarLeviCivitaBinaryChartCertificate
    source_tube_result: PlanarLCAposterioriTubeCheckResult
    target_tube_result: PlanarLCAposterioriTubeCheckResult
    obligations: tuple[CertificateCheckObligation, ...]
    source_exact_anchor: tuple[Fraction, ...]
    target_exact_anchor: tuple[Fraction, ...]
    common_parameter_offset_interval: tuple[Fraction, Fraction]
    derived_edge: PlanarLCGaugeOverlapEdge | None
    serialized_masses: tuple[float, ...]
    selected_pair: tuple[int, int]
    source_pair_energy_constraint_exact: bool
    target_pair_energy_constraint_exact: bool
    mass_ratio_arithmetic_exact: bool

    def _anchor_relation_parity(self) -> int | None:
        source = self.source_exact_anchor
        target = self.target_exact_anchor
        if not (
            type(source) is tuple
            and type(target) is tuple
            and len(source) == len(target) == 14
            and all(type(value) is Fraction for value in source + target)
            and source[4:] == target[4:]
        ):
            return None
        same = source[:4] == target[:4]
        antipodal = target[:4] == tuple(-value for value in source[:4])
        if same == antipodal:
            return None
        return 0 if same else 1

    def _common_interval_valid(self) -> bool:
        interval = self.common_parameter_offset_interval
        return bool(
            type(interval) is tuple
            and len(interval) == 2
            and all(type(value) is Fraction for value in interval)
            and interval[0] < interval[1]
            and interval[0] <= 0 <= interval[1]
        )

    def _exact_constraint_status(self) -> tuple[bool, bool] | None:
        if not (
            type(self.serialized_masses) is tuple
            and len(self.serialized_masses) == 3
            and all(type(value) is float and np.isfinite(value) and value > 0.0
                    for value in self.serialized_masses)
            and type(self.selected_pair) is tuple
            and len(self.selected_pair) == 2
            and all(type(index) is int for index in self.selected_pair)
            and self.selected_pair[0] != self.selected_pair[1]
            and set(self.selected_pair).issubset({0, 1, 2})
            and type(self.source_exact_anchor) is tuple
            and type(self.target_exact_anchor) is tuple
            and len(self.source_exact_anchor) == 14
            and len(self.target_exact_anchor) == 14
            and all(
                type(value) is Fraction
                for value in self.source_exact_anchor + self.target_exact_anchor
            )
        ):
            return None
        pair_mass = sum(
            Fraction.from_float(self.serialized_masses[index])
            for index in self.selected_pair
        )

        def exact(anchor: tuple[Fraction, ...]) -> bool:
            z = anchor[0:2]
            w = anchor[2:4]
            h = anchor[4]
            return bool(
                2 * sum(value * value for value in w)
                - pair_mass
                - sum(value * value for value in z) * h
                == 0
            )

        try:
            return exact(self.source_exact_anchor), exact(self.target_exact_anchor)
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            return None

    def _snapshot_lifted_overlap_certified(self) -> bool:
        parity = self._anchor_relation_parity()
        edge = self.derived_edge
        return bool(
            type(self.checker_id) is str
            and self.checker_id == "planar_lc_exact_overlap_anchor_checker_v1"
            and type(self.overlap_id) is str
            and bool(self.overlap_id)
            and type(self.source_chart_id) is str
            and type(self.target_chart_id) is str
            and bool(self.source_chart_id)
            and bool(self.target_chart_id)
            and self.source_chart_id != self.target_chart_id
            and type(self.raw_overlap_certificate)
            is PlanarLCExactOverlapAnchorCertificate
            and type(self.raw_source_tube) is PlanarLCAposterioriTubeCertificate
            and type(self.raw_target_tube) is PlanarLCAposterioriTubeCertificate
            and type(self.raw_source_chart)
            is PlanarLeviCivitaBinaryChartCertificate
            and type(self.raw_target_chart)
            is PlanarLeviCivitaBinaryChartCertificate
            and self.raw_overlap_certificate.overlap_id == self.overlap_id
            and self.raw_overlap_certificate.source_chart_id
            == self.source_chart_id
            == self.raw_source_chart.chart_id
            == self.raw_source_tube.chart_id
            and self.raw_overlap_certificate.target_chart_id
            == self.target_chart_id
            == self.raw_target_chart.chart_id
            == self.raw_target_tube.chart_id
            and self.raw_overlap_certificate.source_tube_id
            == self.raw_source_tube.tube_id
            and self.raw_overlap_certificate.target_tube_id
            == self.raw_target_tube.tube_id
            and self.raw_source_chart.masses == self.raw_target_chart.masses
            and self.raw_source_chart.pair == self.raw_target_chart.pair
            and type(self.raw_source_tube.initial_error_bound) is float
            and type(self.raw_target_tube.initial_error_bound) is float
            and self.raw_source_tube.initial_error_bound == 0.0
            and self.raw_target_tube.initial_error_bound == 0.0
            and type(self.source_tube_result) is PlanarLCAposterioriTubeCheckResult
            and type(self.target_tube_result) is PlanarLCAposterioriTubeCheckResult
            and self.source_tube_result.checker_id
            == "independent_planar_lc_aposteriori_tube_checker_v1"
            and self.target_tube_result.checker_id
            == "independent_planar_lc_aposteriori_tube_checker_v1"
            and self.source_tube_result.chart_id == self.source_chart_id
            and self.target_tube_result.chart_id == self.target_chart_id
            and self.source_tube_result.tube_id == self.raw_source_tube.tube_id
            and self.target_tube_result.tube_id == self.raw_target_tube.tube_id
            and self.source_tube_result.certified
            and self.target_tube_result.certified
            and self.source_tube_result.anchor_is_polynomial_center
            and self.target_tube_result.anchor_is_polynomial_center
            and self._common_interval_valid()
            and parity in (0, 1)
            and type(edge) is PlanarLCGaugeOverlapEdge
            and edge.overlap_id == self.overlap_id
            and edge.source_chart_id == self.source_chart_id
            and edge.target_chart_id == self.target_chart_id
            and type(edge.parity) is int
            and edge.parity == parity
            and _exact_certified_obligation_manifest(
                self.obligations,
                _PLANAR_LC_EXACT_OVERLAP_OBLIGATION_NAMES,
            )
        )

    @property
    def lifted_overlap_certified(self) -> bool:
        """Whether fresh raw evidence proves one exact LC deck relation."""

        if not (
            type(self.raw_overlap_certificate)
            is PlanarLCExactOverlapAnchorCertificate
            and type(self.raw_source_tube) is PlanarLCAposterioriTubeCertificate
            and type(self.raw_target_tube) is PlanarLCAposterioriTubeCertificate
            and type(self.raw_source_chart)
            is PlanarLeviCivitaBinaryChartCertificate
            and type(self.raw_target_chart)
            is PlanarLeviCivitaBinaryChartCertificate
        ):
            return False
        try:
            fresh = check_planar_lc_exact_overlap_anchor(
                self.raw_overlap_certificate,
                self.raw_source_tube,
                self.raw_source_chart,
                self.raw_target_tube,
                self.raw_target_chart,
            )
            return bool(
                type(fresh) is PlanarLCExactOverlapAnchorCheckResult
                and fresh == self
                and self._snapshot_lifted_overlap_certified()
            )
        except Exception:
            # Exact-class dataclass fields are untrusted serialized evidence.
            # In particular, hostile equality objects and arrays must fail
            # closed rather than escape through dataclass equality.
            return False

    @property
    def certified(self) -> bool:
        return self.lifted_overlap_certified

    @property
    def constrained_newtonian_overlap_certified(self) -> bool:
        """Whether the overlap also represents one constrained Newtonian IVP."""

        if not self.lifted_overlap_certified:
            return False
        constraint_status = self._exact_constraint_status()
        try:
            mass_ratio_exact = _planar_lc_mass_ratio_arithmetic_exact(
                self.serialized_masses, self.selected_pair
            )
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            mass_ratio_exact = False
        return bool(
            constraint_status is not None
            and constraint_status == (True, True)
            and self.source_pair_energy_constraint_exact is True
            and self.target_pair_energy_constraint_exact is True
            and mass_ratio_exact is True
            and self.mass_ratio_arithmetic_exact is True
            and self.source_tube_result.constrained_newtonian_lift_certified
            and self.target_tube_result.constrained_newtonian_lift_certified
        )

    @property
    def two_sided_overlap(self) -> bool:
        """Whether the checked common domain extends on both sides of the anchor."""

        interval = self.common_parameter_offset_interval
        return bool(
            self.lifted_overlap_certified
            and interval[0] < 0 < interval[1]
        )

    @property
    def two_sided_overlap_certified(self) -> bool:
        return self.two_sided_overlap

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _check_obligation_ledger_missing(
            self.obligations,
            ledger_name=self.overlap_id or "planar_lc_exact_overlap_anchor",
        )


@dataclass(frozen=True)
class PlanarLCExactGaugeAtlasCheckResult:
    """Raw-evidence aggregate of exact LC overlaps and their derived gauge graph."""

    atlas_id: str
    checker_id: str
    raw_atlas_certificate: PlanarLCExactGaugeAtlasCertificate
    obligations: tuple[CertificateCheckObligation, ...]
    chart_ids: tuple[str, ...]
    tube_ids: tuple[str, ...]
    overlap_ids: tuple[str, ...]
    checked_charts: tuple[PlanarLeviCivitaBinaryChartCertificate, ...]
    checked_tubes: tuple[PlanarLCAposterioriTubeCertificate, ...]
    checked_overlap_certificates: tuple[
        PlanarLCExactOverlapAnchorCertificate, ...
    ]
    vertex_tube_results: tuple[PlanarLCAposterioriTubeCheckResult, ...]
    vertex_problem_data: tuple[
        tuple[tuple[float, ...], tuple[int, int]], ...
    ]
    overlap_results: tuple[PlanarLCExactOverlapAnchorCheckResult, ...]
    anchor_parameter_translations: tuple[
        tuple[Fraction, Fraction, Fraction], ...
    ]
    derived_edges: tuple[PlanarLCGaugeOverlapEdge, ...]
    gauge_result: PlanarLCGaugeGluingCheckResult | None

    def _manifest_schema_valid(self) -> bool:
        return bool(
            type(self.raw_atlas_certificate)
            is PlanarLCExactGaugeAtlasCertificate
            and type(self.atlas_id) is str
            and bool(self.atlas_id)
            and self.raw_atlas_certificate.atlas_id == self.atlas_id
            and type(self.chart_ids) is tuple
            and bool(self.chart_ids)
            and all(type(value) is str and bool(value) for value in self.chart_ids)
            and len(set(self.chart_ids)) == len(self.chart_ids)
            and type(self.tube_ids) is tuple
            and len(self.tube_ids) == len(self.chart_ids)
            and all(type(value) is str and bool(value) for value in self.tube_ids)
            and len(set(self.tube_ids)) == len(self.tube_ids)
            and type(self.overlap_ids) is tuple
            and all(type(value) is str and bool(value) for value in self.overlap_ids)
            and len(set(self.overlap_ids)) == len(self.overlap_ids)
            and self.raw_atlas_certificate.chart_ids == self.chart_ids
            and self.raw_atlas_certificate.tube_ids == self.tube_ids
            and self.raw_atlas_certificate.overlap_ids == self.overlap_ids
        )

    def _vertices_self_check(self) -> bool:
        if not self._manifest_schema_valid() or not (
            type(self.checked_charts) is tuple
            and len(self.checked_charts) == len(self.chart_ids)
            and type(self.checked_tubes) is tuple
            and len(self.checked_tubes) == len(self.chart_ids)
            and type(self.vertex_tube_results) is tuple
            and len(self.vertex_tube_results) == len(self.chart_ids)
            and type(self.vertex_problem_data) is tuple
            and len(self.vertex_problem_data) == len(self.chart_ids)
        ):
            return False
        for index, result in enumerate(self.vertex_tube_results):
            chart = self.checked_charts[index]
            tube = self.checked_tubes[index]
            if not (
                type(chart) is PlanarLeviCivitaBinaryChartCertificate
                and type(tube) is PlanarLCAposterioriTubeCertificate
                and chart.chart_id == self.chart_ids[index]
                and tube.tube_id == self.tube_ids[index]
                and tube.chart_id == chart.chart_id
                and type(tube.initial_error_bound) is float
                and tube.initial_error_bound == 0.0
                and type(result) is PlanarLCAposterioriTubeCheckResult
                and result.checker_id
                == "independent_planar_lc_aposteriori_tube_checker_v1"
                and result.chart_id == self.chart_ids[index]
                and result.tube_id == self.tube_ids[index]
                and result.certified
                and result.anchor_is_polynomial_center
                and self.vertex_problem_data[index]
                == (chart.masses, chart.pair)
            ):
                return False
        first_problem = self.vertex_problem_data[0]
        if not (
            type(first_problem) is tuple
            and len(first_problem) == 2
            and type(first_problem[0]) is tuple
            and len(first_problem[0]) == 3
            and all(
                type(value) is float and np.isfinite(value) and value > 0.0
                for value in first_problem[0]
            )
            and type(first_problem[1]) is tuple
            and len(first_problem[1]) == 2
            and all(type(value) is int for value in first_problem[1])
            and first_problem[1][0] != first_problem[1][1]
            and set(first_problem[1]).issubset({0, 1, 2})
        ):
            return False
        return all(problem == first_problem for problem in self.vertex_problem_data)

    def _overlaps_self_check(self) -> bool:
        if not self._vertices_self_check() or not (
            type(self.checked_overlap_certificates) is tuple
            and len(self.checked_overlap_certificates) == len(self.overlap_ids)
            and type(self.overlap_results) is tuple
            and len(self.overlap_results) == len(self.overlap_ids)
            and type(self.derived_edges) is tuple
            and len(self.derived_edges) == len(self.overlap_ids)
            and type(self.anchor_parameter_translations) is tuple
            and len(self.anchor_parameter_translations) == len(self.overlap_ids)
        ):
            return False
        vertex_by_chart = {
            chart_id: self.vertex_tube_results[index]
            for index, chart_id in enumerate(self.chart_ids)
        }
        for index, result in enumerate(self.overlap_results):
            certificate = self.checked_overlap_certificates[index]
            translation = self.anchor_parameter_translations[index]
            exact_endpoint_ids = bool(
                type(certificate) is PlanarLCExactOverlapAnchorCertificate
                and type(certificate.source_chart_id) is str
                and type(certificate.target_chart_id) is str
            )
            source_tube = vertex_by_chart.get(
                certificate.source_chart_id
                if exact_endpoint_ids
                else None
            )
            target_tube = vertex_by_chart.get(
                certificate.target_chart_id
                if exact_endpoint_ids
                else None
            )
            source_index = (
                self.chart_ids.index(certificate.source_chart_id)
                if exact_endpoint_ids
                and certificate.source_chart_id in self.chart_ids
                else None
            )
            target_index = (
                self.chart_ids.index(certificate.target_chart_id)
                if exact_endpoint_ids
                and certificate.target_chart_id in self.chart_ids
                else None
            )
            translation_valid = False
            if (
                type(certificate) is PlanarLCExactOverlapAnchorCertificate
                and source_tube is not None
                and target_tube is not None
                and type(certificate.source_anchor_parameter) is float
                and type(certificate.target_anchor_parameter) is float
                and np.isfinite(certificate.source_anchor_parameter)
                and np.isfinite(certificate.target_anchor_parameter)
            ):
                source_q = Fraction.from_float(certificate.source_anchor_parameter)
                target_q = Fraction.from_float(certificate.target_anchor_parameter)
                translation_valid = bool(
                    type(translation) is tuple
                    and len(translation) == 3
                    and translation == (source_q, target_q, target_q - source_q)
                    and source_index is not None
                    and target_index is not None
                    and certificate.source_anchor_parameter
                    == self.checked_tubes[source_index].anchor_parameter
                    and certificate.target_anchor_parameter
                    == self.checked_tubes[target_index].anchor_parameter
                )
            if not (
                type(certificate) is PlanarLCExactOverlapAnchorCertificate
                and certificate.overlap_id == self.overlap_ids[index]
                and type(result) is PlanarLCExactOverlapAnchorCheckResult
                and result.overlap_id == self.overlap_ids[index]
                and result.certified
                and source_index is not None
                and target_index is not None
                and result.raw_overlap_certificate == certificate
                and result.raw_source_chart
                == self.checked_charts[source_index]
                and result.raw_target_chart
                == self.checked_charts[target_index]
                and result.raw_source_tube
                == self.checked_tubes[source_index]
                and result.raw_target_tube
                == self.checked_tubes[target_index]
                and type(result.derived_edge) is PlanarLCGaugeOverlapEdge
                and type(self.derived_edges[index]) is PlanarLCGaugeOverlapEdge
                and self.derived_edges[index] == result.derived_edge
                and result.source_chart_id in vertex_by_chart
                and result.target_chart_id in vertex_by_chart
                and result.source_tube_result
                == vertex_by_chart[result.source_chart_id]
                and result.target_tube_result
                == vertex_by_chart[result.target_chart_id]
                and certificate.source_chart_id == result.source_chart_id
                and certificate.target_chart_id == result.target_chart_id
                and certificate.source_tube_id == result.source_tube_result.tube_id
                and certificate.target_tube_id == result.target_tube_result.tube_id
                and translation_valid
            ):
                return False
        return True

    def _graph_self_check(self) -> bool:
        graph = self.gauge_result
        return bool(
            self._overlaps_self_check()
            and type(graph) is PlanarLCGaugeGluingCheckResult
            and graph.certificate_id == f"{self.atlas_id}:derived-gauge-graph"
            and graph.chart_ids == self.chart_ids
            and graph.checked_overlaps == self.derived_edges
            and graph.certified
        )

    def _snapshot_lifted_atlas_certified(self) -> bool:
        return bool(
            type(self.checker_id) is str
            and self.checker_id == "planar_lc_exact_gauge_atlas_checker_v1"
            and self._graph_self_check()
            and _exact_certified_obligation_manifest(
                self.obligations,
                _PLANAR_LC_EXACT_GAUGE_ATLAS_OBLIGATION_NAMES,
            )
        )

    @property
    def lifted_atlas_certified(self) -> bool:
        """Whether fresh raw evidence reproduces this exact gauge atlas."""

        if type(self.raw_atlas_certificate) is not PlanarLCExactGaugeAtlasCertificate:
            return False
        try:
            fresh = check_planar_lc_exact_gauge_atlas(
                self.raw_atlas_certificate,
                self.checked_charts,
                self.checked_tubes,
                self.checked_overlap_certificates,
            )
            return bool(
                type(fresh) is PlanarLCExactGaugeAtlasCheckResult
                and fresh == self
                and self._snapshot_lifted_atlas_certified()
            )
        except Exception:
            # Recomputed dataclass equality is part of the fail-closed check;
            # malformed exact-class fields may themselves define unsafe ==.
            return False

    @property
    def certified(self) -> bool:
        return self.lifted_atlas_certified

    @property
    def constrained_newtonian_atlas_certified(self) -> bool:
        if not self.lifted_atlas_certified:
            return False
        masses, pair = self.vertex_problem_data[0]
        try:
            mass_arithmetic_exact = _planar_lc_mass_ratio_arithmetic_exact(
                masses, pair
            )
        except (IndexError, TypeError, ValueError, ZeroDivisionError):
            mass_arithmetic_exact = False
        return bool(
            mass_arithmetic_exact
            and all(
                result.constrained_newtonian_lift_certified
                for result in self.vertex_tube_results
            )
            and all(
                result.constrained_newtonian_overlap_certified
                for result in self.overlap_results
            )
        )

    @property
    def all_overlaps_two_sided(self) -> bool:
        return bool(
            self.lifted_atlas_certified
            and all(result.two_sided_overlap for result in self.overlap_results)
        )

    @property
    def gauge_assignment(self) -> tuple[tuple[str, int], ...]:
        if not self.lifted_atlas_certified or self.gauge_result is None:
            return ()
        return self.gauge_result.gauge_assignment

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _check_obligation_ledger_missing(
                self.obligations,
                ledger_name=self.atlas_id or "planar_lc_exact_gauge_atlas",
            )
        )
        for result in self.vertex_tube_results:
            if type(result) is PlanarLCAposterioriTubeCheckResult:
                missing.extend(result.missing_obligations)
        for result in self.overlap_results:
            if type(result) is PlanarLCExactOverlapAnchorCheckResult:
                missing.extend(result.missing_obligations)
        if type(self.gauge_result) is PlanarLCGaugeGluingCheckResult:
            missing.extend(self.gauge_result.missing_obligations)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class PlanarLCExactCollisionAnchorCheckResult:
    """Proof that one zero-error lifted IVP is anchored at binary collision."""

    collision_id: str
    checker_id: str
    tube_result: PlanarLCAposterioriTubeCheckResult
    obligations: tuple[CertificateCheckObligation, ...]
    exact_speed_squared: float
    pair_mass: float
    collision_physical_time: float
    collision_parameter: float
    mass_ratio_arithmetic_exact: bool

    @property
    def certified(self) -> bool:
        return bool(
            self.tube_result.certified
            and _check_obligation_ledger_certified(self.obligations)
        )

    @property
    def isolated_binary_collision_certified(self) -> bool:
        return self.certified

    @property
    def local_physical_time_strictly_increasing_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _check_obligation_ledger_missing(
                self.obligations,
                ledger_name=self.collision_id or "planar_lc_exact_collision_anchor",
            )
        )
        missing.extend(self.tube_result.missing_obligations)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class PlanarLCTwoSidedCollisionPassageCheckResult:
    """Two punctured Newton branches selected by one collision-anchored IVP."""

    passage_id: str
    checker_id: str
    collision_result: PlanarLCExactCollisionAnchorCheckResult
    left_target_tube_result: OrdinaryAposterioriTubeCheckResult | WeightedOrdinaryAposterioriTubeCheckResult
    right_target_tube_result: OrdinaryAposterioriTubeCheckResult | WeightedOrdinaryAposterioriTubeCheckResult
    obligations: tuple[CertificateCheckObligation, ...]
    left_rho_lower_bound: float
    right_rho_lower_bound: float
    left_projection_gap: float
    right_projection_gap: float
    left_time_origin_interval: tuple[float, float]
    right_time_origin_interval: tuple[float, float]

    @property
    def certified(self) -> bool:
        return bool(
            self.collision_result.certified
            and self.left_target_tube_result.certified
            and self.right_target_tube_result.certified
            and _check_obligation_ledger_certified(self.obligations)
        )

    @property
    def generalized_binary_collision_continuation_certified(self) -> bool:
        return self.certified

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _check_obligation_ledger_missing(
                self.obligations,
                ledger_name=self.passage_id or "planar_lc_two_sided_collision_passage",
            )
        )
        missing.extend(self.collision_result.missing_obligations)
        missing.extend(self.left_target_tube_result.missing_obligations)
        missing.extend(self.right_target_tube_result.missing_obligations)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class OrdinaryToPlanarLCEnclosureTransitionCheckResult:
    """Checked lift of an exact ordinary branch into an LC tube."""

    transition_id: str
    checker_id: str
    target_tube_result: PlanarLCAposterioriTubeCheckResult
    obligations: tuple[CertificateCheckObligation, ...]
    lift_branch_count: int
    max_lift_box_gap: float
    time_gap: float
    target_initial_time_gap: float
    entry_lift_rho_lower_bound: float

    @property
    def certified(self) -> bool:
        return bool(
            self.target_tube_result.certified
            and _check_obligation_ledger_certified(self.obligations)
        )

    @property
    def constrained_newtonian_lift_certified(self) -> bool:
        return self.certified

    @property
    def constrained_lc_entry_certified(self) -> bool:
        """The source enclosure's algebraic LC lifts are constrained anchors."""
        return self.certified

    @property
    def physical_time_strictly_monotone_certified(self) -> bool:
        return bool(self.certified and self.entry_lift_rho_lower_bound > 0.0)

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _check_obligation_ledger_missing(
                self.obligations,
                ledger_name=self.transition_id or "ordinary_to_planar_lc_enclosure",
            )
        )
        missing.extend(self.target_tube_result.missing_obligations)
        return tuple(dict.fromkeys(missing))


_GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES = (
    "gauge_aware_ordinary_to_lc_exact_input_types",
    "gauge_aware_ordinary_to_lc_identifiers_match",
    "gauge_aware_ordinary_to_lc_source_exact_enclosure_certified",
    "gauge_aware_ordinary_to_lc_common_masses_dimension_ordered_pair",
    "gauge_aware_ordinary_to_lc_exact_mass_ratio_arithmetic",
    "gauge_aware_ordinary_to_lc_parameters_inside_and_anchor_matches",
    "gauge_aware_ordinary_to_lc_declared_handoff_time_cap",
    "gauge_aware_ordinary_to_lc_target_lifted_tube_certified",
    "gauge_aware_ordinary_to_lc_source_box_reconstructed",
    "gauge_aware_ordinary_to_lc_selected_pair_collision_free",
    "gauge_aware_ordinary_to_lc_canonical_branch_grammar",
    "gauge_aware_ordinary_to_lc_canonical_lift_count",
    "gauge_aware_ordinary_to_lc_derived_gauge_graph_certified",
    "gauge_aware_ordinary_to_lc_entry_lift_rho_positive",
    "gauge_aware_ordinary_to_lc_target_initial_error_contains_physical_time",
    "gauge_aware_ordinary_to_lc_one_global_complement_contains_all_lifts",
    "gauge_aware_ordinary_to_lc_existential_selected_lifts_constrained",
)


def _gauge_aware_raw_primitive_schema_valid(
    certificate: OrdinaryToPlanarLCEnclosureTransitionCertificate,
    source_binding: InitialValueProblemBindingCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate,
    source_chart: OrdinaryTaylorChartCertificate,
    target_chart: PlanarLeviCivitaBinaryChartCertificate,
    target_tube: PlanarLCAposterioriTubeCertificate,
) -> bool:
    """Require canonical built-in primitives before any checker normalization."""

    def nonempty_string(value: object) -> bool:
        return type(value) is str and bool(value)

    def finite_float(value: object) -> bool:
        return type(value) is float and math.isfinite(value)

    def float_tuple(
        value: object,
        *,
        length: int | None = None,
        nonempty: bool = False,
    ) -> bool:
        return bool(
            type(value) is tuple
            and (length is None or len(value) == length)
            and (not nonempty or bool(value))
            and all(finite_float(item) for item in value)
        )

    def float_matrix(value: object, rows: int, columns: int) -> bool:
        return bool(
            type(value) is tuple
            and len(value) == rows
            and all(float_tuple(row, length=columns) for row in value)
        )

    def ordinary_coefficient_series(value: object) -> bool:
        return bool(
            type(value) is tuple
            and len(value) >= 2
            and all(float_matrix(coefficient, 3, 2) for coefficient in value)
        )

    def vector_coefficient_series(value: object) -> bool:
        return bool(
            type(value) is tuple
            and len(value) >= 2
            and all(float_tuple(coefficient, length=2) for coefficient in value)
        )

    try:
        source_position_series_ok = ordinary_coefficient_series(
            source_chart.position_coefficients
        )
        source_velocity_series_ok = ordinary_coefficient_series(
            source_chart.velocity_coefficients
        )
        source_order = (
            len(source_chart.position_coefficients)
            if source_position_series_ok
            else -1
        )

        target_vector_series = (
            target_chart.z_coefficients,
            target_chart.z_velocity_coefficients,
            target_chart.binary_center_coefficients,
            target_chart.binary_center_velocity_coefficients,
            target_chart.third_offset_coefficients,
            target_chart.third_offset_velocity_coefficients,
        )
        target_vector_series_ok = all(
            vector_coefficient_series(series) for series in target_vector_series
        )
        target_order = (
            len(target_chart.z_coefficients)
            if target_vector_series_ok
            else -1
        )

        return bool(
            all(
                nonempty_string(value)
                for value in (
                    certificate.transition_id,
                    certificate.source_chart_id,
                    certificate.target_chart_id,
                    certificate.source,
                    source_binding.binding_id,
                    source_binding.chart_id,
                    source_binding.source,
                    source_tube.tube_id,
                    source_tube.chart_id,
                    source_tube.source,
                    source_chart.certificate_id,
                    source_chart.chart_id,
                    source_chart.chart_type,
                    source_chart.source,
                    target_chart.certificate_id,
                    target_chart.chart_id,
                    target_chart.chart_type,
                    target_chart.source,
                    target_tube.tube_id,
                    target_tube.chart_id,
                    target_tube.source,
                )
            )
            and all(
                finite_float(value)
                for value in (
                    certificate.source_parameter,
                    certificate.target_parameter,
                    certificate.handoff_time,
                    certificate.max_time_gap,
                    source_binding.initial_time,
                    source_binding.chart_parameter,
                    source_binding.time_tolerance,
                    source_binding.position_tolerance,
                    source_binding.velocity_tolerance,
                    source_tube.anchor_parameter,
                    source_tube.initial_error_bound,
                    source_tube.tube_radius,
                    source_tube.max_defect_bound,
                    source_tube.max_lipschitz_bound,
                    source_chart.coefficient_tolerance,
                    source_chart.residual_tolerance,
                    source_chart.tail_bound,
                    target_chart.coefficient_tolerance,
                    target_chart.regularized_residual_tolerance,
                    target_chart.projected_residual_tolerance,
                    target_chart.tail_bound,
                    target_chart.projection_rho_lower_bound,
                    target_tube.anchor_parameter,
                    target_tube.initial_error_bound,
                    target_tube.tube_radius,
                    target_tube.max_defect_bound,
                    target_tube.max_lipschitz_bound,
                )
            )
            and float_tuple(source_binding.masses, length=3)
            and float_matrix(source_binding.positions, 3, 2)
            and float_matrix(source_binding.velocities, 3, 2)
            and float_tuple(source_chart.masses, length=3)
            and source_position_series_ok
            and source_velocity_series_ok
            and len(source_chart.velocity_coefficients) == source_order
            and float_tuple(source_chart.parameter_interval, length=2)
            and float_tuple(source_chart.physical_time_interval, length=2)
            and type(source_chart.sample_count) is int
            and float_tuple(target_chart.masses, length=3)
            and type(target_chart.pair) is tuple
            and len(target_chart.pair) == 2
            and all(type(index) is int for index in target_chart.pair)
            and target_vector_series_ok
            and all(len(series) == target_order for series in target_vector_series)
            and float_tuple(
                target_chart.pair_energy_coefficients,
                length=target_order,
            )
            and float_tuple(
                target_chart.physical_time_coefficients,
                length=target_order,
            )
            and float_tuple(target_chart.parameter_interval, length=2)
            and float_tuple(target_chart.physical_time_interval, length=2)
            and type(target_chart.sample_count) is int
            and type(target_tube.require_pair_energy_constraint) is bool
        )
    except Exception:
        return False


def _gauge_aware_direct_ordinary_source_valid(
    binding: InitialValueProblemBindingCertificate,
    tube: OrdinaryAposterioriTubeCertificate,
    source_chart: OrdinaryTaylorChartCertificate,
    source_validation: ValidatedOrdinaryIVPChartCheckResult,
) -> bool:
    """Recompute and bind one direct ordinary IVP solely from raw evidence."""

    try:
        fresh = check_validated_ordinary_ivp_chart(binding, tube, source_chart)
        return bool(
            type(binding) is InitialValueProblemBindingCertificate
            and type(tube) is OrdinaryAposterioriTubeCertificate
            and type(source_chart) is OrdinaryTaylorChartCertificate
            and type(source_validation) is ValidatedOrdinaryIVPChartCheckResult
            and source_validation.checker_id
            == "validated_ordinary_ivp_chart_checker_v1"
            and binding.chart_id == tube.chart_id == source_chart.chart_id
            and type(source_validation.binding_result)
            is InitialValueBindingCheckResult
            and type(source_validation.tube_result)
            is OrdinaryAposterioriTubeCheckResult
            and type(source_validation.chart_result) is CertificateCheckResult
            and source_validation.binding_result.chart_id
            == source_validation.tube_result.chart_id
            == source_chart.chart_id
            and source_validation.chart_result.certificate_id
            == source_chart.certificate_id
            and source_validation.certified
            and type(fresh) is ValidatedOrdinaryIVPChartCheckResult
            and fresh == source_validation
        )
    except Exception:
        return False


def _rejected_gauge_aware_direct_ordinary_source(
    binding: InitialValueProblemBindingCertificate,
    tube: OrdinaryAposterioriTubeCertificate,
    chart: OrdinaryTaylorChartCertificate,
) -> ValidatedOrdinaryIVPChartCheckResult:
    """Deterministic fail-closed result for malformed exact-class raw inputs."""

    chart_id = chart.chart_id if type(chart.chart_id) is str else ""
    binding_id = binding.binding_id if type(binding.binding_id) is str else ""
    tube_id = tube.tube_id if type(tube.tube_id) is str else ""
    failed = CertificateCheckObligation(
        "gauge_aware_direct_ordinary_source_malformed_raw_input",
        False,
        "direct ordinary source reconstruction raised and was rejected",
    )
    return ValidatedOrdinaryIVPChartCheckResult(
        binding_result=InitialValueBindingCheckResult(
            binding_id=binding_id,
            chart_id=chart_id,
            checker_id="independent_initial_value_binding_checker_v1",
            obligations=(failed,),
            max_position_gap=np.inf,
            max_velocity_gap=np.inf,
            time_gap=np.inf,
        ),
        tube_result=OrdinaryAposterioriTubeCheckResult(
            tube_id=tube_id,
            chart_id=chart_id,
            checker_id="independent_ordinary_aposteriori_tube_checker_v1",
            obligations=(failed,),
            defect_bound=np.inf,
            lipschitz_bound=np.inf,
            gronwall_error_bound=np.inf,
            nominal_pair_distance_floor=0.0,
            tube_pair_distance_floor=0.0,
        ),
        chart_result=CertificateCheckResult(
            certificate_id=(
                chart.certificate_id
                if type(chart.certificate_id) is str
                else ""
            ),
            certificate_type="ordinary_taylor",
            checker_id="independent_ordinary_chart_checker_v1",
            obligations=(failed,),
            max_coefficient_residual=np.inf,
            max_sampled_newton_residual=np.inf,
        ),
        chart_serialization_admissible=False,
        checker_id="validated_ordinary_ivp_chart_checker_v1",
        obligations=(failed,),
    )


@dataclass(frozen=True)
class GaugeAwareOrdinaryToPlanarLCEnclosureTransitionCheckResult:
    """Opt-in ordinary-to-LC lift modulo one global deck transformation.

    The result is intentionally separate from the legacy transition checker.
    It retains all raw evidence and certifies only when fresh recomputation
    reproduces the complete immutable snapshot exactly.
    """

    transition_id: str
    checker_id: str
    raw_transition_certificate: OrdinaryToPlanarLCEnclosureTransitionCertificate
    raw_source_binding: InitialValueProblemBindingCertificate
    raw_source_tube: OrdinaryAposterioriTubeCertificate
    raw_source_chart: OrdinaryTaylorChartCertificate
    raw_target_chart: PlanarLeviCivitaBinaryChartCertificate
    source_validation: ValidatedOrdinaryIVPChartCheckResult
    raw_target_tube: PlanarLCAposterioriTubeCertificate
    target_tube_result: PlanarLCAposterioriTubeCheckResult
    obligations: tuple[CertificateCheckObligation, ...]
    source_state_box: tuple[tuple[Fraction, Fraction], ...]
    relative_position_box: tuple[
        tuple[Fraction, Fraction], tuple[Fraction, Fraction]
    ]
    canonical_lift_case: str
    patch_vertex_ids: tuple[str, ...]
    raw_patch_boxes: tuple[
        tuple[tuple[Fraction, Fraction], ...], ...
    ]
    derived_edges: tuple[PlanarLCGaugeOverlapEdge, ...]
    gauge_result: PlanarLCGaugeGluingCheckResult | None
    tested_assignments: tuple[tuple[tuple[str, int], ...], ...]
    tested_lift_max_gaps: tuple[Fraction, ...]
    tested_containments: tuple[bool, ...]
    selected_complement_index: int
    selected_assignment: tuple[tuple[str, int], ...]
    selected_transformed_patch_boxes: tuple[
        tuple[tuple[Fraction, Fraction], ...], ...
    ]
    exact_declared_time_gap: Fraction
    exact_source_target_time_gap: Fraction
    entry_lift_rho_lower_bound: Fraction

    def _snapshot_certified(self) -> bool:
        def fraction_pair(value: object) -> bool:
            return bool(
                type(value) is tuple
                and len(value) == 2
                and type(value[0]) is Fraction
                and type(value[1]) is Fraction
                and value[0] <= value[1]
            )

        def fraction_box(value: object, dimension: int) -> bool:
            return bool(
                type(value) is tuple
                and len(value) == dimension
                and all(fraction_pair(item) for item in value)
            )

        def obligation_ledger_schema(value: object) -> bool:
            return bool(
                type(value) is tuple
                and bool(value)
                and all(
                    type(item) is CertificateCheckObligation
                    and type(item.obligation) is str
                    and bool(item.obligation)
                    and type(item.certified) is bool
                    and type(item.detail) is str
                    for item in value
                )
            )

        source_nested_schema = bool(
            type(self.source_validation) is ValidatedOrdinaryIVPChartCheckResult
            and type(self.source_validation.checker_id) is str
            and self.source_validation.checker_id
            == "validated_ordinary_ivp_chart_checker_v1"
            and type(self.source_validation.chart_serialization_admissible) is bool
            and self.source_validation.chart_serialization_admissible is True
            and obligation_ledger_schema(self.source_validation.obligations)
            and type(self.source_validation.binding_result)
            is InitialValueBindingCheckResult
            and all(
                type(value) is str and bool(value)
                for value in (
                    self.source_validation.binding_result.binding_id,
                    self.source_validation.binding_result.chart_id,
                    self.source_validation.binding_result.checker_id,
                )
            )
            and self.source_validation.binding_result.checker_id
            == "independent_initial_value_binding_checker_v1"
            and obligation_ledger_schema(
                self.source_validation.binding_result.obligations
            )
            and all(
                type(value) is float and np.isfinite(value)
                for value in (
                    self.source_validation.binding_result.max_position_gap,
                    self.source_validation.binding_result.max_velocity_gap,
                    self.source_validation.binding_result.time_gap,
                )
            )
            and type(self.source_validation.tube_result)
            is OrdinaryAposterioriTubeCheckResult
            and all(
                type(value) is str and bool(value)
                for value in (
                    self.source_validation.tube_result.tube_id,
                    self.source_validation.tube_result.chart_id,
                    self.source_validation.tube_result.checker_id,
                )
            )
            and self.source_validation.tube_result.checker_id
            == "independent_ordinary_aposteriori_tube_checker_v1"
            and obligation_ledger_schema(
                self.source_validation.tube_result.obligations
            )
            and all(
                type(value) is float and np.isfinite(value)
                for value in (
                    self.source_validation.tube_result.defect_bound,
                    self.source_validation.tube_result.lipschitz_bound,
                    self.source_validation.tube_result.gronwall_error_bound,
                    self.source_validation.tube_result.nominal_pair_distance_floor,
                    self.source_validation.tube_result.tube_pair_distance_floor,
                )
            )
            and type(self.source_validation.chart_result) is CertificateCheckResult
            and all(
                type(value) is str and bool(value)
                for value in (
                    self.source_validation.chart_result.certificate_id,
                    self.source_validation.chart_result.certificate_type,
                    self.source_validation.chart_result.checker_id,
                )
            )
            and self.source_validation.chart_result.checker_id
            == "independent_ordinary_taylor_checker_interval_v2"
            and obligation_ledger_schema(
                self.source_validation.chart_result.obligations
            )
            and all(
                type(value) is float and np.isfinite(value)
                for value in (
                    self.source_validation.chart_result.max_coefficient_residual,
                    self.source_validation.chart_result.max_sampled_newton_residual,
                )
            )
        )
        target_nested_schema = bool(
            type(self.target_tube_result) is PlanarLCAposterioriTubeCheckResult
            and all(
                type(value) is str and bool(value)
                for value in (
                    self.target_tube_result.tube_id,
                    self.target_tube_result.chart_id,
                    self.target_tube_result.checker_id,
                )
            )
            and obligation_ledger_schema(self.target_tube_result.obligations)
            and all(
                type(value) is float and np.isfinite(value)
                for value in (
                    self.target_tube_result.defect_bound,
                    self.target_tube_result.lipschitz_bound,
                    self.target_tube_result.gronwall_error_bound,
                    self.target_tube_result.third_body_distance_floor,
                    self.target_tube_result.anchor_pair_energy_constraint_residual,
                )
            )
            and type(
                self.target_tube_result.pair_energy_constraint_anchor_certified
            ) is bool
            and type(self.target_tube_result.anchor_is_polynomial_center) is bool
        )

        def assignment(value: object) -> bool:
            return bool(
                type(value) is tuple
                and len(value) == len(self.patch_vertex_ids)
                and all(
                    type(item) is tuple
                    and len(item) == 2
                    and type(item[0]) is str
                    and type(item[1]) is int
                    and item[1] in (0, 1)
                    for item in value
                )
                and tuple(item[0] for item in value) == self.patch_vertex_ids
            )

        case_suffix = {
            "closed_upper_singleton": ("upper",),
            "closed_lower_singleton": ("lower",),
            "right_half_singleton": ("right",),
            "strict_negative_cut_two_patch": ("upper", "lower"),
        }.get(self.canonical_lift_case)
        if case_suffix is None:
            return False
        expected_patch_ids = tuple(
            f"{self.transition_id}:patch:{index}-{suffix}"
            for index, suffix in enumerate(case_suffix)
        )
        two_patch = len(expected_patch_ids) == 2
        exact_edge_structure = False
        if not two_patch:
            exact_edge_structure = type(self.derived_edges) is tuple and not self.derived_edges
        elif type(self.derived_edges) is tuple and len(self.derived_edges) == 1:
            edge = self.derived_edges[0]
            exact_edge_structure = bool(
                type(edge) is PlanarLCGaugeOverlapEdge
                and type(edge.overlap_id) is str
                and edge.overlap_id
                == f"{self.transition_id}:negative-axis-overlap"
                and type(edge.source_chart_id) is str
                and edge.source_chart_id == expected_patch_ids[0]
                and type(edge.target_chart_id) is str
                and edge.target_chart_id == expected_patch_ids[1]
                and type(edge.parity) is int
                and edge.parity == 1
            )

        selected_assignment_map = (
            dict(self.selected_assignment)
            if assignment(self.selected_assignment)
            else {}
        )
        expected_transformed_boxes = tuple(
            tuple(
                (-upper, -lower)
                if selected_assignment_map.get(self.patch_vertex_ids[index]) == 1
                and component < 4
                else (lower, upper)
                for component, (lower, upper) in enumerate(box)
            )
            for index, box in enumerate(self.raw_patch_boxes)
        ) if (
            type(self.raw_patch_boxes) is tuple
            and all(fraction_box(box, 13) for box in self.raw_patch_boxes)
        ) else ()

        return bool(
            type(self.checker_id) is str
            and self.checker_id
            == "gauge_aware_ordinary_to_planar_lc_enclosure_transition_checker_v1"
            and type(self.transition_id) is str
            and bool(self.transition_id)
            and type(self.raw_transition_certificate)
            is OrdinaryToPlanarLCEnclosureTransitionCertificate
            and type(self.raw_source_binding)
            is InitialValueProblemBindingCertificate
            and type(self.raw_source_tube) is OrdinaryAposterioriTubeCertificate
            and type(self.raw_source_chart) is OrdinaryTaylorChartCertificate
            and type(self.raw_target_chart)
            is PlanarLeviCivitaBinaryChartCertificate
            and type(self.source_validation)
            is ValidatedOrdinaryIVPChartCheckResult
            and type(self.raw_target_tube) is PlanarLCAposterioriTubeCertificate
            and _gauge_aware_raw_primitive_schema_valid(
                self.raw_transition_certificate,
                self.raw_source_binding,
                self.raw_source_tube,
                self.raw_source_chart,
                self.raw_target_chart,
                self.raw_target_tube,
            )
            and type(self.raw_transition_certificate.transition_id) is str
            and bool(self.raw_transition_certificate.transition_id)
            and type(self.raw_transition_certificate.source_chart_id) is str
            and bool(self.raw_transition_certificate.source_chart_id)
            and type(self.raw_transition_certificate.target_chart_id) is str
            and bool(self.raw_transition_certificate.target_chart_id)
            and type(self.raw_source_binding.binding_id) is str
            and bool(self.raw_source_binding.binding_id)
            and type(self.raw_source_binding.chart_id) is str
            and bool(self.raw_source_binding.chart_id)
            and type(self.raw_source_tube.tube_id) is str
            and bool(self.raw_source_tube.tube_id)
            and type(self.raw_source_tube.chart_id) is str
            and bool(self.raw_source_tube.chart_id)
            and type(self.raw_source_chart.chart_id) is str
            and bool(self.raw_source_chart.chart_id)
            and type(self.raw_source_chart.certificate_id) is str
            and bool(self.raw_source_chart.certificate_id)
            and type(self.raw_target_chart.chart_id) is str
            and bool(self.raw_target_chart.chart_id)
            and type(self.raw_target_chart.certificate_id) is str
            and bool(self.raw_target_chart.certificate_id)
            and type(self.raw_target_tube.tube_id) is str
            and bool(self.raw_target_tube.tube_id)
            and type(self.raw_target_tube.chart_id) is str
            and bool(self.raw_target_tube.chart_id)
            and self.raw_transition_certificate.transition_id == self.transition_id
            and self.raw_transition_certificate.source_chart_id
            == self.raw_source_binding.chart_id
            == self.raw_source_tube.chart_id
            == self.raw_source_chart.chart_id
            and self.raw_transition_certificate.target_chart_id
            == self.raw_target_chart.chart_id
            and self.raw_target_tube.chart_id == self.raw_target_chart.chart_id
            and source_nested_schema
            and self.source_validation.binding_result.binding_id
            == self.raw_source_binding.binding_id
            and self.source_validation.binding_result.chart_id
            == self.raw_source_chart.chart_id
            and self.source_validation.tube_result.tube_id
            == self.raw_source_tube.tube_id
            and self.source_validation.tube_result.chart_id
            == self.raw_source_chart.chart_id
            and self.source_validation.chart_result.certificate_id
            == self.raw_source_chart.certificate_id
            and _gauge_aware_direct_ordinary_source_valid(
                self.raw_source_binding,
                self.raw_source_tube,
                self.raw_source_chart,
                self.source_validation,
            )
            and target_nested_schema
            and self.target_tube_result.checker_id
            == "independent_planar_lc_aposteriori_tube_checker_v1"
            and self.target_tube_result.chart_id == self.raw_target_chart.chart_id
            and self.target_tube_result.tube_id == self.raw_target_tube.tube_id
            and self.target_tube_result.certified
            and _exact_certified_obligation_manifest(
                self.obligations,
                _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES,
            )
            and fraction_box(self.source_state_box, 12)
            and fraction_box(self.relative_position_box, 2)
            and type(self.canonical_lift_case) is str
            and type(self.patch_vertex_ids) is tuple
            and all(
                type(patch_id) is str and bool(patch_id)
                for patch_id in self.patch_vertex_ids
            )
            and self.patch_vertex_ids == expected_patch_ids
            and len(set(self.patch_vertex_ids)) == len(self.patch_vertex_ids)
            and type(self.raw_patch_boxes) is tuple
            and len(self.raw_patch_boxes) == len(self.patch_vertex_ids)
            and all(fraction_box(box, 13) for box in self.raw_patch_boxes)
            and exact_edge_structure
            and type(self.gauge_result) is PlanarLCGaugeGluingCheckResult
            and type(self.gauge_result.certificate_id) is str
            and self.gauge_result.certificate_id
            == f"{self.transition_id}:derived-gauge-cover"
            and type(self.gauge_result.checker_id) is str
            and self.gauge_result.checker_id
            == "planar_lc_z2_gauge_gluing_checker_v1"
            and type(self.gauge_result.obligations) is tuple
            and bool(self.gauge_result.obligations)
            and all(
                type(obligation) is PlanarLCGaugeGluingObligation
                and type(obligation.obligation) is str
                and bool(obligation.obligation)
                and type(obligation.certified) is bool
                and type(obligation.detail) is str
                for obligation in self.gauge_result.obligations
            )
            and self.gauge_result.chart_ids == self.patch_vertex_ids
            and self.gauge_result.checked_overlaps == self.derived_edges
            and self.gauge_result.certified
            and type(self.tested_assignments) is tuple
            and len(self.tested_assignments) == 2
            and all(assignment(item) for item in self.tested_assignments)
            and self.tested_assignments[0] == self.gauge_result.gauge_assignment
            and self.tested_assignments[1]
            == tuple(
                (chart_id, bit ^ 1)
                for chart_id, bit in self.tested_assignments[0]
            )
            and type(self.tested_lift_max_gaps) is tuple
            and len(self.tested_lift_max_gaps) == 2
            and all(
                type(gap) is Fraction and gap >= 0
                for gap in self.tested_lift_max_gaps
            )
            and type(self.tested_containments) is tuple
            and len(self.tested_containments) == 2
            and all(type(value) is bool for value in self.tested_containments)
            and type(self.selected_complement_index) is int
            and self.selected_complement_index in (0, 1)
            and assignment(self.selected_assignment)
            and self.selected_assignment
            == self.tested_assignments[self.selected_complement_index]
            and self.tested_containments[self.selected_complement_index] is True
            and type(self.selected_transformed_patch_boxes) is tuple
            and len(self.selected_transformed_patch_boxes)
            == len(self.patch_vertex_ids)
            and all(
                fraction_box(box, 13)
                for box in self.selected_transformed_patch_boxes
            )
            and self.selected_transformed_patch_boxes
            == expected_transformed_boxes
            and type(self.exact_declared_time_gap) is Fraction
            and self.exact_declared_time_gap >= 0
            and type(self.exact_source_target_time_gap) is Fraction
            and self.exact_source_target_time_gap >= 0
            and type(self.entry_lift_rho_lower_bound) is Fraction
            and self.entry_lift_rho_lower_bound > 0
        )

    @property
    def certified(self) -> bool:
        try:
            if not self._snapshot_certified():
                return False
            fresh = check_gauge_aware_ordinary_to_planar_lc_enclosure_transition(
                self.raw_transition_certificate,
                self.raw_source_binding,
                self.raw_source_tube,
                self.raw_source_chart,
                self.raw_target_chart,
                self.raw_target_tube,
            )
            return bool(
                type(fresh)
                is GaugeAwareOrdinaryToPlanarLCEnclosureTransitionCheckResult
                and fresh == self
            )
        except Exception:
            # Raw serialized dataclass fields and equality implementations are
            # untrusted.  Every exceptional path must fail closed.
            return False

    @property
    def gauge_aware_lift_certified(self) -> bool:
        return self.certified

    @property
    def constrained_newtonian_lift_certified(self) -> bool:
        """Existential selected lifts, not every interval/tube point."""

        return self.certified

    @property
    def physical_time_strictly_monotone_certified(self) -> bool:
        return bool(self.certified and self.entry_lift_rho_lower_bound > 0)

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _check_obligation_ledger_missing(
                self.obligations,
                ledger_name=self.transition_id
                or "gauge_aware_ordinary_to_planar_lc_enclosure",
            )
        )
        if type(self.target_tube_result) is PlanarLCAposterioriTubeCheckResult:
            missing.extend(self.target_tube_result.missing_obligations)
        if type(self.source_validation) is ValidatedOrdinaryIVPChartCheckResult:
            missing.extend(self.source_validation.missing_obligations)
        if type(self.gauge_result) is PlanarLCGaugeGluingCheckResult:
            missing.extend(self.gauge_result.missing_obligations)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class PlanarLCToOrdinaryEnclosureTransitionCheckResult:
    """Checked punctured LC projection into an elapsed-time ordinary tube."""

    transition_id: str
    checker_id: str
    target_tube_result: OrdinaryAposterioriTubeCheckResult
    obligations: tuple[CertificateCheckObligation, ...]
    source_rho_lower_bound: float
    max_projected_lift_gap: float
    physical_time_origin_interval: tuple[float, float]
    constrained_source_entry_certified: bool

    @property
    def certified(self) -> bool:
        return bool(
            self.target_tube_result.certified
            and _check_obligation_ledger_certified(self.obligations)
        )

    @property
    def exact_newtonian_continuation_certified(self) -> bool:
        # Projection equivalence, parameter ordering, and collision isolation
        # are deliberately not inferred from an accepted endpoint enclosure.
        return False

    @property
    def projected_exit_enclosure_certified(self) -> bool:
        return self.certified

    @property
    def punctured_newton_projection_certified(self) -> bool:
        """The accepted exit meets the proved punctured projection lemma."""
        return bool(self.certified and self.constrained_source_entry_certified)

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing = list(
            _check_obligation_ledger_missing(
                self.obligations,
                ledger_name=self.transition_id or "planar_lc_to_ordinary_enclosure",
            )
        )
        missing.extend(self.target_tube_result.missing_obligations)
        return tuple(dict.fromkeys(missing))


@dataclass(frozen=True)
class EventIsolationCheckResult:
    """Result of checking one serialized event-isolation certificate."""

    event_id: str
    event_type: str
    checker_id: str
    obligations: tuple[CertificateCheckObligation, ...]
    root_interval_width: float

    @property
    def certified(self) -> bool:
        return _check_obligation_ledger_certified(self.obligations)

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _check_obligation_ledger_missing(
            self.obligations,
            ledger_name=self.event_id or "event_isolation_check",
        )


@dataclass(frozen=True)
class BranchUnionCheckResult:
    """Result of checking one serialized finite branch-union certificate."""

    union_id: str
    union_type: str
    checker_id: str
    obligations: tuple[CertificateCheckObligation, ...]
    leaf_count: int

    @property
    def certified(self) -> bool:
        return _check_obligation_ledger_certified(self.obligations)

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _check_obligation_ledger_missing(
            self.obligations,
            ledger_name=self.union_id or "branch_union_check",
        )


@dataclass(frozen=True)
class ChartChainCheckResult:
    """Result of checking one serialized finite chart-chain certificate."""

    chain_id: str
    chain_type: str
    checker_id: str
    obligations: tuple[CertificateCheckObligation, ...]
    chart_count: int

    @property
    def certified(self) -> bool:
        return _check_obligation_ledger_certified(self.obligations)

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        return _check_obligation_ledger_missing(
            self.obligations,
            ledger_name=self.chain_id or "chart_chain_check",
        )


@dataclass(frozen=True)
class ProofGradeArithmeticBackendCertificate:
    """Small certificate for exact rational interval arithmetic support."""

    backend_id: str
    exact_fraction_endpoints: bool
    rational_interval_operations_checked: bool
    rational_interval_polynomial_eval_checked: bool
    rational_interval_sign_trichotomy_checked: bool
    finite_float_to_fraction_embedding_checked: bool
    statement: str
    proof_sketch: str
    witness_source: str = "rational_interval_arithmetic_backend"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exact_fraction_endpoints",
            self.exact_fraction_endpoints is True,
        )
        object.__setattr__(
            self,
            "rational_interval_operations_checked",
            self.rational_interval_operations_checked is True,
        )
        object.__setattr__(
            self,
            "rational_interval_polynomial_eval_checked",
            self.rational_interval_polynomial_eval_checked is True,
        )
        object.__setattr__(
            self,
            "rational_interval_sign_trichotomy_checked",
            self.rational_interval_sign_trichotomy_checked is True,
        )
        object.__setattr__(
            self,
            "finite_float_to_fraction_embedding_checked",
            self.finite_float_to_fraction_embedding_checked is True,
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.backend_id == "python_fraction_rational_interval_backend"
            and self.exact_fraction_endpoints is True
            and self.rational_interval_operations_checked is True
            and self.rational_interval_polynomial_eval_checked is True
            and self.rational_interval_sign_trichotomy_checked is True
            and self.finite_float_to_fraction_embedding_checked is True
            and self.statement
            and self.proof_sketch
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        fields = (
            ("exact_fraction_endpoints", self.exact_fraction_endpoints),
            (
                "rational_interval_operations_checked",
                self.rational_interval_operations_checked,
            ),
            (
                "rational_interval_polynomial_eval_checked",
                self.rational_interval_polynomial_eval_checked,
            ),
            (
                "rational_interval_sign_trichotomy_checked",
                self.rational_interval_sign_trichotomy_checked,
            ),
            (
                "finite_float_to_fraction_embedding_checked",
                self.finite_float_to_fraction_embedding_checked,
            ),
        )
        missing = [name for name, certified in fields if certified is not True]
        if self.backend_id != "python_fraction_rational_interval_backend":
            missing.append("proof_grade_arithmetic_backend_id")
        return tuple(missing)


@dataclass(frozen=True)
class CertificateCheckerKernelSupportCertificate:
    """Theorem-facing support manifest for the independent checker kernel."""

    checker_id: str
    supported_chart_types: tuple[str, ...]
    event_obligation_ids: tuple[str, ...]
    branch_union_obligation_ids: tuple[str, ...]
    chart_chain_obligation_ids: tuple[str, ...]
    transition_obligation_ids: tuple[str, ...]
    proof_grade_arithmetic_backend_certificate: (
        ProofGradeArithmeticBackendCertificate | None
    ) = None
    witness_source: str = "certificate_checker_kernel_support"

    @property
    def proof_grade_arithmetic_backend_sound(self) -> bool:
        return bool(
            isinstance(
                self.proof_grade_arithmetic_backend_certificate,
                ProofGradeArithmeticBackendCertificate,
            )
            and self.proof_grade_arithmetic_backend_certificate.proof_certified
            is True
        )

    @property
    def ordinary_taylor_sound(self) -> bool:
        return "ordinary_taylor" in self.supported_chart_types

    @property
    def levi_civita_sound(self) -> bool:
        return "planar_levi_civita_binary" in self.supported_chart_types

    @property
    def spatial_ks_sound(self) -> bool:
        return "spatial_ks_binary" in self.supported_chart_types

    @property
    def fuchsian_stop_sound(self) -> bool:
        return "total_collision_fuchsian_stop" in self.supported_chart_types

    @property
    def generalized_fuchsian_stop_sound(self) -> bool:
        return (
            "total_collision_generalized_fuchsian_stop"
            in self.supported_chart_types
        )

    @property
    def transition_sound(self) -> bool:
        return bool(self.transition_obligation_ids)

    @property
    def branch_union_sound(self) -> bool:
        return bool(self.branch_union_obligation_ids)

    @property
    def chart_chain_sound(self) -> bool:
        return bool(self.chart_chain_obligation_ids)

    @property
    def verifier_kernel_sound(self) -> bool:
        return bool(
            self.checker_id == "independent_chart_verifier_v1"
            and self.ordinary_taylor_sound
            and self.levi_civita_sound
            and self.spatial_ks_sound
            and self.fuchsian_stop_sound
            and self.generalized_fuchsian_stop_sound
            and self.transition_sound
            and self.branch_union_sound
            and self.chart_chain_sound
        )

    @property
    def proof_certified(self) -> bool:
        return bool(
            self.verifier_kernel_sound
            and self.event_obligation_ids
            and self.proof_grade_arithmetic_backend_sound
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        fields = (
            ("ordinary_taylor_sound", self.ordinary_taylor_sound),
            ("levi_civita_sound", self.levi_civita_sound),
            ("spatial_ks_sound", self.spatial_ks_sound),
            ("fuchsian_stop_sound", self.fuchsian_stop_sound),
            (
                "generalized_fuchsian_stop_sound",
                self.generalized_fuchsian_stop_sound,
            ),
            ("transition_sound", self.transition_sound),
            ("branch_union_sound", self.branch_union_sound),
            ("chart_chain_sound", self.chart_chain_sound),
            ("event_isolation_sound", bool(self.event_obligation_ids)),
            ("verifier_kernel_sound", self.verifier_kernel_sound),
            (
                "proof_grade_arithmetic_backend_sound",
                self.proof_grade_arithmetic_backend_sound,
            ),
        )
        return tuple(name for name, certified in fields if not certified)


_PROOF_GRADE_CHART_OBLIGATIONS: dict[str, tuple[str, ...]] = {
    "ordinary_taylor": (
        "ordinary_taylor_exact_rational_residual_polynomials",
        "interval_taylor_model_newton_residual",
    ),
    "planar_levi_civita_binary": (
        "interval_physical_time_containment",
        "planar_lc_pair_energy_constraint",
        "planar_lc_exact_rational_regularized_residual_polynomials",
        "interval_planar_lc_regularized_rhs_residual",
        "interval_projected_newton_residual_away_from_binary_collision",
    ),
    "spatial_ks_binary": (
        "interval_physical_time_containment",
        "spatial_ks_pair_energy_and_horizontal_constraints",
        "spatial_ks_exact_rational_regularized_residual_polynomials",
        "interval_spatial_ks_regularized_rhs_residual",
        "interval_spatial_ks_projected_newton_residual_away_from_binary_collision",
    ),
    "total_collision_fuchsian_stop": (
        "interval_fuchsian_lifted_residual_on_punctured_shells",
        "interval_fuchsian_projected_residual_on_punctured_shells",
        "interval_zero_angular_momentum_on_punctured_shells",
        "interval_center_of_mass_and_linear_momentum_on_punctured_shells",
        "interval_fuchsian_supported_scale_finite_energy_matching",
        "interval_total_collision_endpoint_collapse_envelope",
        "fuchsian_primitive_cauchy_inputs_certify",
        "fuchsian_primitive_cauchy_shell_inside_isolation",
        "fuchsian_tail_bound_covers_primitive_cauchy_tail",
        "fuchsian_primitive_cauchy_residual_tail_within_tolerance",
    ),
    "total_collision_generalized_fuchsian_stop": (
        "generalized_fuchsian_remainder_majorant_certifies",
        "generalized_fuchsian_remainder_component_inputs_certify",
        "generalized_fuchsian_remainder_shell_inside_isolation",
        "generalized_fuchsian_tail_bound_covers_remainder_tail",
        "interval_generalized_fuchsian_lifted_residual_on_punctured_shells",
        "generalized_fuchsian_remainder_residual_tail_within_tolerance",
        "cauchy_generalized_fuchsian_projected_residual_tail_on_punctured_shells",
        "interval_generalized_zero_angular_momentum_on_punctured_shells",
        "interval_generalized_center_of_mass_and_linear_momentum_on_punctured_shells",
        "generalized_fuchsian_endpoint_collapse_envelope",
    ),
}

_PROOF_GRADE_EVENT_OBLIGATIONS = (
    "event_exact_rational_interval_arithmetic",
)

_PROOF_GRADE_BRANCH_UNION_OBLIGATIONS = (
    "branch_union_exact_rational_interval_aggregation",
)

_PROOF_GRADE_CHART_CHAIN_OBLIGATIONS = (
    "chart_chain_exact_rational_time_coverage",
)

_PROOF_GRADE_TRANSITION_OBLIGATIONS = (
    "transition_exact_rational_state_continuity",
)


def certify_rational_interval_arithmetic_backend_soundness(
    *,
    witness_source: str = "rational_interval_arithmetic_backend",
) -> ProofGradeArithmeticBackendCertificate:
    """Check the exact rational interval operations used by the verifier."""

    third = Fraction(1, 3)
    half = Fraction(1, 2)
    two_fifths = Fraction(2, 5)
    three_fifths = Fraction(3, 5)
    first = RationalInterval(third, half)
    second = RationalInterval(two_fifths, three_fifths)
    exact_fraction_endpoints = bool(
        isinstance(first.lower, Fraction)
        and isinstance(first.upper, Fraction)
        and first.as_tuple() == (third, half)
    )
    rational_interval_operations_checked = bool(
        (first + second).as_tuple() == (Fraction(11, 15), Fraction(11, 10))
        and (first - second).as_tuple() == (Fraction(-4, 15), Fraction(1, 10))
        and (first * second).as_tuple() == (Fraction(2, 15), Fraction(3, 10))
        and second.reciprocal().as_tuple() == (Fraction(5, 3), Fraction(5, 2))
    )
    variable = RationalInterval(third, half)
    polynomial = (
        RationalInterval.point(1),
        RationalInterval.point(2),
        RationalInterval.point(1),
    )
    value = rational_interval_polynomial_eval(polynomial, variable)
    rational_interval_polynomial_eval_checked = bool(
        value.as_tuple() == (Fraction(16, 9), Fraction(9, 4))
        and rational_interval_polyder(polynomial)[0].as_tuple()
        == (Fraction(2, 1), Fraction(2, 1))
    )
    rational_interval_sign_trichotomy_checked = bool(
        rational_interval_sign(RationalInterval(Fraction(1, 5), Fraction(2, 5)))
        == 1
        and rational_interval_sign(
            RationalInterval(Fraction(-2, 5), Fraction(-1, 5))
        )
        == -1
        and rational_interval_sign(RationalInterval(Fraction(-1, 5), Fraction(1, 5)))
        == 0
    )
    embedded = RationalInterval.from_float_interval(0.5, 0.75)
    finite_float_to_fraction_embedding_checked = bool(
        embedded.as_tuple() == (Fraction(1, 2), Fraction(3, 4))
    )
    return ProofGradeArithmeticBackendCertificate(
        backend_id="python_fraction_rational_interval_backend",
        exact_fraction_endpoints=exact_fraction_endpoints,
        rational_interval_operations_checked=rational_interval_operations_checked,
        rational_interval_polynomial_eval_checked=(
            rational_interval_polynomial_eval_checked
        ),
        rational_interval_sign_trichotomy_checked=(
            rational_interval_sign_trichotomy_checked
        ),
        finite_float_to_fraction_embedding_checked=(
            finite_float_to_fraction_embedding_checked
        ),
        statement=(
            "The checker backend represents interval endpoints as exact "
            "Fractions, performs interval arithmetic by exact endpoint "
            "operations, evaluates power-basis polynomials by Horner interval "
            "arithmetic, and uses exact rational sign trichotomy for checker "
            "obligations that claim proof-grade rational arithmetic."
        ),
        proof_sketch=(
            "RationalInterval stores Fraction endpoints and all arithmetic "
            "constructors combine only endpoint Fractions by exact field "
            "operations.  Addition, subtraction, multiplication, reciprocal, "
            "polynomial Horner evaluation, derivative coefficient scaling, and "
            "sign trichotomy are checked here on rational fixtures whose exact "
            "results are known.  Finite binary floats are embedded by "
            "Fraction.from_float, preserving their exact binary-rational value "
            "before the checker performs interval operations."
        ),
        witness_source=str(witness_source),
    )


def certify_certificate_checker_kernel_support(
    *,
    proof_grade_arithmetic_backend_certificate: (
        ProofGradeArithmeticBackendCertificate | None
    ) = None,
    proof_grade_arithmetic_backend_sound: bool = False,
    witness_source: str = "certificate_checker_kernel_support",
) -> CertificateCheckerKernelSupportCertificate:
    """Derive the supported checker-kernel language from the checker tables."""

    if proof_grade_arithmetic_backend_sound and (
        proof_grade_arithmetic_backend_certificate is None
    ):
        raise TypeError(
            "proof_grade_arithmetic_backend_sound=True requires a "
            "ProofGradeArithmeticBackendCertificate"
        )

    return CertificateCheckerKernelSupportCertificate(
        checker_id="independent_chart_verifier_v1",
        supported_chart_types=tuple(sorted(_PROOF_GRADE_CHART_OBLIGATIONS)),
        event_obligation_ids=_PROOF_GRADE_EVENT_OBLIGATIONS,
        branch_union_obligation_ids=_PROOF_GRADE_BRANCH_UNION_OBLIGATIONS,
        chart_chain_obligation_ids=_PROOF_GRADE_CHART_CHAIN_OBLIGATIONS,
        transition_obligation_ids=_PROOF_GRADE_TRANSITION_OBLIGATIONS,
        proof_grade_arithmetic_backend_certificate=(
            proof_grade_arithmetic_backend_certificate
        ),
        witness_source=str(witness_source),
    )


@dataclass(frozen=True)
class IndependentChartVerifierCertificate:
    """Aggregate verifier result for a collection of serialized charts."""

    checker_id: str
    chart_results: tuple[CertificateCheckResult, ...]
    transition_results: tuple[TransitionCheckResult, ...] = ()
    event_results: tuple[EventIsolationCheckResult, ...] = ()
    branch_union_results: tuple[BranchUnionCheckResult, ...] = ()
    chart_chain_results: tuple[ChartChainCheckResult, ...] = ()
    atlas_binding_token: str | None = None

    @property
    def checked_certificate_count(self) -> int:
        return len(self.chart_results)

    @property
    def ordinary_taylor_chart_count(self) -> int:
        return sum(
            1
            for result in self.chart_results
            if result.certificate_type == "ordinary_taylor"
        )

    @property
    def planar_levi_civita_binary_chart_count(self) -> int:
        return sum(
            1
            for result in self.chart_results
            if result.certificate_type == "planar_levi_civita_binary"
        )

    @property
    def spatial_ks_binary_chart_count(self) -> int:
        return sum(
            1
            for result in self.chart_results
            if result.certificate_type == "spatial_ks_binary"
        )

    @property
    def total_collision_fuchsian_stop_chart_count(self) -> int:
        return sum(
            1
            for result in self.chart_results
            if result.certificate_type == "total_collision_fuchsian_stop"
        )

    @property
    def total_collision_generalized_fuchsian_stop_chart_count(self) -> int:
        return sum(
            1
            for result in self.chart_results
            if result.certificate_type == "total_collision_generalized_fuchsian_stop"
        )

    @property
    def checked_event_count(self) -> int:
        return len(self.event_results)

    @property
    def checked_branch_union_count(self) -> int:
        return len(self.branch_union_results)

    @property
    def checked_chart_chain_count(self) -> int:
        return len(self.chart_chain_results)

    @property
    def certified(self) -> bool:
        return bool(
            self.chart_results
            and all(
                type(result) is CertificateCheckResult
                and result.certified is True
                for result in self.chart_results
            )
            and all(
                type(result) is TransitionCheckResult
                and result.certified is True
                for result in self.transition_results
            )
            and all(
                type(result) is EventIsolationCheckResult
                and result.certified is True
                for result in self.event_results
            )
            and all(
                type(result) is BranchUnionCheckResult
                and result.certified is True
                for result in self.branch_union_results
            )
            and all(
                type(result) is ChartChainCheckResult
                and result.certified is True
                for result in self.chart_chain_results
            )
        )

    @property
    def proof_certified(self) -> bool:
        return self.certified

    @property
    def proof_grade_arithmetic_obligation_ids(self) -> tuple[str, ...]:
        obligation_ids: list[str] = []
        for result in self.chart_results:
            if type(result) is not CertificateCheckResult:
                continue
            required = _PROOF_GRADE_CHART_OBLIGATIONS.get(result.certificate_type, ())
            certified = _certified_obligation_ids(result.obligations)
            obligation_ids.extend(
                obligation for obligation in required if obligation in certified
            )
        for result in self.event_results:
            if type(result) is not EventIsolationCheckResult:
                continue
            certified = _certified_obligation_ids(result.obligations)
            obligation_ids.extend(
                obligation
                for obligation in _PROOF_GRADE_EVENT_OBLIGATIONS
                if obligation in certified
            )
        for result in self.branch_union_results:
            if type(result) is not BranchUnionCheckResult:
                continue
            certified = _certified_obligation_ids(result.obligations)
            obligation_ids.extend(
                obligation
                for obligation in _PROOF_GRADE_BRANCH_UNION_OBLIGATIONS
                if obligation in certified
            )
        for result in self.chart_chain_results:
            if type(result) is not ChartChainCheckResult:
                continue
            certified = _certified_obligation_ids(result.obligations)
            obligation_ids.extend(
                obligation
                for obligation in _PROOF_GRADE_CHART_CHAIN_OBLIGATIONS
                if obligation in certified
            )
        for result in self.transition_results:
            if type(result) is not TransitionCheckResult:
                continue
            certified = _certified_obligation_ids(result.obligations)
            obligation_ids.extend(
                obligation
                for obligation in _PROOF_GRADE_TRANSITION_OBLIGATIONS
                if obligation in certified
            )
        return tuple(dict.fromkeys(obligation_ids))

    @property
    def proof_grade_arithmetic_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if not self.chart_results and not self.event_results:
            blockers.append("proof_grade_arithmetic_no_checked_chart_or_event")
        for result in self.chart_results:
            if type(result) is not CertificateCheckResult:
                blockers.append("chart_result_type")
                continue
            required = _PROOF_GRADE_CHART_OBLIGATIONS.get(result.certificate_type)
            if required is None:
                blockers.append(
                    f"{result.certificate_id}:proof_grade_arithmetic_unknown_chart_type"
                )
                continue
            certified = _certified_obligation_ids(result.obligations)
            for obligation in required:
                if obligation not in certified:
                    blockers.append(f"{result.certificate_id}:{obligation}")
        for result in self.event_results:
            if type(result) is not EventIsolationCheckResult:
                blockers.append("event_result_type")
                continue
            certified = _certified_obligation_ids(result.obligations)
            for obligation in _PROOF_GRADE_EVENT_OBLIGATIONS:
                if obligation not in certified:
                    blockers.append(f"{result.event_id}:{obligation}")
        for result in self.transition_results:
            if type(result) is not TransitionCheckResult:
                blockers.append("transition_result_type")
                continue
            certified = _certified_obligation_ids(result.obligations)
            for obligation in _PROOF_GRADE_TRANSITION_OBLIGATIONS:
                if obligation not in certified:
                    blockers.append(f"{result.transition_id}:{obligation}")
        for result in self.branch_union_results:
            if type(result) is not BranchUnionCheckResult:
                blockers.append("branch_union_result_type")
                continue
            certified = _certified_obligation_ids(result.obligations)
            for obligation in _PROOF_GRADE_BRANCH_UNION_OBLIGATIONS:
                if obligation not in certified:
                    blockers.append(f"{result.union_id}:{obligation}")
        for result in self.chart_chain_results:
            if type(result) is not ChartChainCheckResult:
                blockers.append("chart_chain_result_type")
                continue
            certified = _certified_obligation_ids(result.obligations)
            for obligation in _PROOF_GRADE_CHART_CHAIN_OBLIGATIONS:
                if obligation not in certified:
                    blockers.append(f"{result.chain_id}:{obligation}")
        return tuple(dict.fromkeys(blockers))

    @property
    def proof_grade_arithmetic_checked_bundle_certified(self) -> bool:
        return bool(self.certified and not self.proof_grade_arithmetic_blockers)

    @property
    def proof_grade_finite_atlas_blockers(self) -> tuple[str, ...]:
        blockers = list(self.proof_grade_arithmetic_blockers)
        if self.checked_chart_chain_count <= 0:
            blockers.append("finite_atlas_chart_chain_checked")
        if (
            self.checked_certificate_count <= 0
            and self.checked_branch_union_count <= 0
        ):
            blockers.append("finite_atlas_chart_or_branch_union_checked")
        return tuple(dict.fromkeys(blockers))

    @property
    def proof_grade_finite_atlas_bundle_certified(self) -> bool:
        return bool(self.certified and not self.proof_grade_finite_atlas_blockers)

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing: list[str] = []
        for result in self.chart_results:
            if type(result) is not CertificateCheckResult:
                missing.append("chart_result_type")
                continue
            missing.extend(result.missing_obligations)
        for result in self.transition_results:
            if type(result) is not TransitionCheckResult:
                missing.append("transition_result_type")
                continue
            missing.extend(result.missing_obligations)
        for result in self.event_results:
            if type(result) is not EventIsolationCheckResult:
                missing.append("event_result_type")
                continue
            missing.extend(result.missing_obligations)
        for result in self.branch_union_results:
            if type(result) is not BranchUnionCheckResult:
                missing.append("branch_union_result_type")
                continue
            missing.extend(result.missing_obligations)
        for result in self.chart_chain_results:
            if type(result) is not ChartChainCheckResult:
                missing.append("chart_chain_result_type")
                continue
            missing.extend(result.missing_obligations)
        return tuple(dict.fromkeys(missing))


def _certified_obligation_ids(
    obligations: Iterable[CertificateCheckObligation],
) -> set[str]:
    return {
        obligation.obligation
        for obligation in obligations
        if isinstance(obligation, CertificateCheckObligation)
        and obligation.certified is True
    }


def check_ordinary_taylor_chart(
    certificate: OrdinaryTaylorChartCertificate,
) -> CertificateCheckResult:
    """Verify an ordinary Taylor chart from explicit serialized coefficients."""

    q = _coefficient_array(certificate.position_coefficients)
    v = _coefficient_array(certificate.velocity_coefficients)
    masses = np.asarray(certificate.masses, dtype=float)
    parameter_interval = tuple(float(value) for value in certificate.parameter_interval)
    physical_time_interval = tuple(
        float(value) for value in certificate.physical_time_interval
    )
    coefficient_tolerance = float(certificate.coefficient_tolerance)
    residual_tolerance = float(certificate.residual_tolerance)
    tail_bound = float(certificate.tail_bound)
    sample_count = int(certificate.sample_count)
    unit_speed_tolerance = (
        coefficient_tolerance
        if np.isfinite(coefficient_tolerance) and coefficient_tolerance >= 0.0
        else 1.0e-14
    )

    shape_ok = bool(
        q.ndim == 3
        and v.shape == q.shape
        and q.shape[0] >= 2
        and q.shape[1] == 3
        and q.shape[2] in (2, 3)
        and masses.shape == (3,)
    )
    finite_arrays = bool(np.all(np.isfinite(q)) and np.all(np.isfinite(v)))
    finite_masses = bool(masses.shape == (3,) and np.all(np.isfinite(masses)))
    positive_masses = bool(finite_masses and np.all(masses > 0.0))
    finite_intervals = bool(
        _finite_nonempty_interval(parameter_interval)
        and _finite_nonempty_interval(physical_time_interval)
    )
    unit_physical_parameter_speed = bool(
        finite_intervals
        and _ordinary_unit_physical_parameter_speed(
            parameter_interval,
            physical_time_interval,
            tolerance=max(unit_speed_tolerance, 1.0e-14),
        )
    )
    finite_tolerances = bool(
        np.isfinite(coefficient_tolerance)
        and coefficient_tolerance >= 0.0
        and np.isfinite(residual_tolerance)
        and residual_tolerance >= 0.0
    )
    tail_admissible = bool(np.isfinite(tail_bound) and tail_bound >= 0.0)
    initial_noncollision = bool(shape_ok and _initial_noncollision(q[0]))

    max_coefficient_residual = np.inf
    recurrence_certified = False
    if (
        shape_ok
        and finite_arrays
        and positive_masses
        and initial_noncollision
        and finite_tolerances
    ):
        try:
            max_coefficient_residual = _max_coefficient_recurrence_residual(
                q,
                v,
                masses,
            )
            recurrence_certified = bool(
                max_coefficient_residual <= coefficient_tolerance
            )
        except (FloatingPointError, ValueError):
            max_coefficient_residual = np.inf

    max_sampled_newton_residual = np.inf
    sampled_residual_certified = False
    max_interval_newton_residual = np.inf
    interval_residual_certified = False
    exact_rational_interval_residual_certified = False
    if (
        recurrence_certified
        and finite_intervals
        and unit_physical_parameter_speed
        and finite_tolerances
        and sample_count >= 2
    ):
        try:
            max_sampled_newton_residual = _max_sampled_newton_residual(
                q,
                v,
                masses,
                parameter_interval,
                sample_count=sample_count,
            )
            sampled_residual_certified = bool(
                max_sampled_newton_residual <= residual_tolerance
            )
        except (FloatingPointError, ValueError):
            max_sampled_newton_residual = np.inf
    if (
        recurrence_certified
        and finite_intervals
        and unit_physical_parameter_speed
        and finite_tolerances
    ):
        try:
            max_interval_newton_residual = _max_interval_ordinary_taylor_model_residual(
                q,
                v,
                masses,
                parameter_interval,
                tail_bound=tail_bound,
            )
            exact_rational_interval_residual_certified = True
            interval_residual_certified = bool(
                max_interval_newton_residual <= residual_tolerance
            )
        except (FloatingPointError, ValueError):
            max_interval_newton_residual = np.inf
            exact_rational_interval_residual_certified = False

    obligations = (
        CertificateCheckObligation(
            "ordinary_chart_type",
            certificate.chart_type == "ordinary_taylor",
            f"chart_type={certificate.chart_type!r}",
        ),
        CertificateCheckObligation(
            "certificate_identity_present",
            bool(certificate.certificate_id and certificate.chart_id),
            f"certificate_id={certificate.certificate_id!r}; chart_id={certificate.chart_id!r}",
        ),
        CertificateCheckObligation(
            "coefficient_array_shape",
            shape_ok,
            f"position_shape={q.shape}; velocity_shape={v.shape}; masses_shape={masses.shape}",
        ),
        CertificateCheckObligation(
            "finite_coefficients",
            finite_arrays,
            "all position and velocity coefficients are finite",
        ),
        CertificateCheckObligation(
            "positive_masses",
            positive_masses,
            f"masses={tuple(float(value) for value in masses) if masses.ndim == 1 else masses!r}",
        ),
        CertificateCheckObligation(
            "finite_nonempty_time_intervals",
            finite_intervals,
            f"parameter_interval={parameter_interval}; physical_time_interval={physical_time_interval}",
        ),
        CertificateCheckObligation(
            "ordinary_physical_parameter_unit_speed",
            unit_physical_parameter_speed,
            (
                f"parameter_width={parameter_interval[1] - parameter_interval[0]}; "
                f"physical_width={physical_time_interval[1] - physical_time_interval[0]}"
            ),
        ),
        CertificateCheckObligation(
            "finite_checker_tolerances",
            finite_tolerances,
            f"coefficient_tolerance={coefficient_tolerance}; residual_tolerance={residual_tolerance}",
        ),
        CertificateCheckObligation(
            "initial_noncollision",
            initial_noncollision,
            "all initial pair distances are positive",
        ),
        CertificateCheckObligation(
            "ordinary_taylor_coefficient_recurrence",
            recurrence_certified,
            f"max_coefficient_residual={max_coefficient_residual}",
        ),
        CertificateCheckObligation(
            "ordinary_taylor_exact_rational_residual_polynomials",
            exact_rational_interval_residual_certified,
            (
                "q'-v and v'-a(q) residual polynomials are evaluated with "
                "exact rational interval arithmetic over serialized "
                "binary-float coefficients and parameter endpoints"
            ),
        ),
        CertificateCheckObligation(
            "interval_taylor_model_newton_residual",
            interval_residual_certified,
            (
                "max_interval_taylor_model_newton_residual="
                f"{max_interval_newton_residual}; "
                f"diagnostic_max_sampled_newton_residual="
                f"{max_sampled_newton_residual}"
            ),
        ),
        CertificateCheckObligation(
            "tail_bound_admissible",
            tail_admissible,
            f"tail_bound={tail_bound}",
        ),
    )
    return CertificateCheckResult(
        certificate_id=str(certificate.certificate_id),
        certificate_type="ordinary_taylor",
        checker_id="independent_ordinary_taylor_checker_interval_v2",
        obligations=obligations,
        max_coefficient_residual=float(max_coefficient_residual),
        max_sampled_newton_residual=float(max_sampled_newton_residual),
    )


def check_planar_levi_civita_binary_chart(
    certificate: PlanarLeviCivitaBinaryChartCertificate,
) -> CertificateCheckResult:
    """Verify a planar LC binary chart from explicit serialized coefficients."""

    solution = _regularized_binary_solution_from_certificate(certificate)
    masses = np.asarray(certificate.masses, dtype=float)
    parameter_interval = tuple(float(value) for value in certificate.parameter_interval)
    physical_time_interval = tuple(
        float(value) for value in certificate.physical_time_interval
    )
    coefficient_tolerance = float(certificate.coefficient_tolerance)
    regularized_residual_tolerance = float(certificate.regularized_residual_tolerance)
    projected_residual_tolerance = float(certificate.projected_residual_tolerance)
    tail_bound = float(certificate.tail_bound)
    sample_count = int(certificate.sample_count)
    rho_lower_bound = float(certificate.projection_rho_lower_bound)

    arrays = (
        solution.z,
        solution.z_velocity,
        solution.pair_energy,
        solution.binary_center,
        solution.binary_center_velocity,
        solution.third_offset,
        solution.third_offset_velocity,
        solution.physical_time,
    )
    vector_shape = solution.z.shape
    scalar_shape = solution.pair_energy.shape
    shape_ok = bool(
        solution.z.ndim == 2
        and vector_shape[1:] == (2,)
        and solution.z_velocity.shape == vector_shape
        and solution.binary_center.shape == vector_shape
        and solution.binary_center_velocity.shape == vector_shape
        and solution.third_offset.shape == vector_shape
        and solution.third_offset_velocity.shape == vector_shape
        and vector_shape[0] >= 2
        and solution.pair_energy.shape == (vector_shape[0],)
        and solution.physical_time.shape == (vector_shape[0],)
        and scalar_shape == (vector_shape[0],)
    )
    finite_arrays = bool(all(np.all(np.isfinite(array)) for array in arrays))
    finite_masses = bool(masses.shape == (3,) and np.all(np.isfinite(masses)))
    positive_masses = bool(finite_masses and np.all(masses > 0.0))
    pair_valid = bool(
        len(solution.pair) == 2
        and solution.pair[0] != solution.pair[1]
        and set(solution.pair).issubset({0, 1, 2})
    )
    finite_intervals = bool(
        _finite_nonempty_interval(parameter_interval)
        and _finite_nonempty_interval(physical_time_interval)
    )
    finite_tolerances = bool(
        np.isfinite(coefficient_tolerance)
        and coefficient_tolerance >= 0.0
        and np.isfinite(regularized_residual_tolerance)
        and regularized_residual_tolerance >= 0.0
        and np.isfinite(projected_residual_tolerance)
        and projected_residual_tolerance >= 0.0
        and np.isfinite(rho_lower_bound)
        and rho_lower_bound >= 0.0
    )
    tail_admissible = bool(np.isfinite(tail_bound) and tail_bound >= 0.0)
    (
        physical_time_contained,
        physical_time_enclosure,
    ) = _interval_physical_time_contained(
        solution.physical_time,
        parameter_interval,
        physical_time_interval,
        tolerance=tail_bound,
    )
    sampled_physical_time_contained = bool(
        finite_intervals
        and shape_ok
        and _sampled_physical_time_contained(
            solution.physical_time,
            parameter_interval,
            physical_time_interval,
            sample_count=max(sample_count, 2),
        )
    )

    max_coefficient_residual = np.inf
    recurrence_certified = False
    if (
        shape_ok
        and finite_arrays
        and positive_masses
        and pair_valid
        and finite_tolerances
    ):
        try:
            max_coefficient_residual = _max_regularized_binary_coefficient_residual(
                solution,
            )
            recurrence_certified = bool(
                max_coefficient_residual <= coefficient_tolerance
            )
        except (FloatingPointError, ValueError):
            max_coefficient_residual = np.inf

    max_constraint_residual = np.inf
    pair_energy_constraint_certified = False
    if recurrence_certified:
        try:
            max_constraint_residual = float(
                np.max(
                    np.abs(
                        lc_pair_energy_constraint_coefficients(
                            solution,
                            solution.order,
                        ),
                    ),
                ),
            )
            pair_energy_constraint_certified = bool(
                max_constraint_residual <= coefficient_tolerance
            )
        except (FloatingPointError, ValueError):
            max_constraint_residual = np.inf

    max_regularized_residual = np.inf
    regularized_residual_certified = False
    max_interval_regularized_residual = np.inf
    interval_regularized_residual_certified = False
    exact_rational_regularized_residual_certified = False
    if recurrence_certified and finite_intervals and finite_tolerances and sample_count >= 2:
        try:
            max_regularized_residual = _max_sampled_regularized_binary_residual(
                solution,
                parameter_interval,
                sample_count=sample_count,
            )
            regularized_residual_certified = bool(
                max_regularized_residual <= regularized_residual_tolerance
            )
        except (FloatingPointError, ValueError):
            max_regularized_residual = np.inf
    if recurrence_certified and finite_intervals and finite_tolerances:
        try:
            max_interval_regularized_residual = (
                _max_interval_planar_lc_taylor_model_residual(
                    solution,
                    parameter_interval,
                    tail_bound=tail_bound,
                )
            )
            interval_regularized_residual_certified = bool(
                max_interval_regularized_residual <= regularized_residual_tolerance
            )
            exact_rational_regularized_residual_certified = True
        except (FloatingPointError, ValueError):
            max_interval_regularized_residual = np.inf
            exact_rational_regularized_residual_certified = False

    max_projected_residual = np.inf
    projected_sample_count = 0
    projected_residual_certified = False
    max_interval_projected_residual = np.inf
    interval_projected_residual_certified = False
    if recurrence_certified and finite_intervals and finite_tolerances and sample_count >= 2:
        try:
            (
                max_projected_residual,
                projected_sample_count,
            ) = _max_sampled_projected_binary_newton_residual(
                solution,
                parameter_interval,
                sample_count=sample_count,
                rho_lower_bound=rho_lower_bound,
            )
            projected_residual_certified = bool(
                projected_sample_count > 0
                and max_projected_residual <= projected_residual_tolerance
            )
        except (FloatingPointError, ValueError):
            max_projected_residual = np.inf
            projected_sample_count = 0
    if recurrence_certified and finite_intervals and finite_tolerances:
        try:
            max_interval_projected_residual = (
                _max_interval_planar_lc_projected_newton_residual(
                    solution,
                    parameter_interval,
                    tail_bound=tail_bound,
                    rho_lower_bound=rho_lower_bound,
                )
            )
            interval_projected_residual_certified = bool(
                max_interval_projected_residual <= projected_residual_tolerance
            )
        except (FloatingPointError, ValueError):
            max_interval_projected_residual = np.inf

    obligations = (
        CertificateCheckObligation(
            "planar_levi_civita_binary_chart_type",
            certificate.chart_type == "planar_levi_civita_binary",
            f"chart_type={certificate.chart_type!r}",
        ),
        CertificateCheckObligation(
            "certificate_identity_present",
            bool(certificate.certificate_id and certificate.chart_id),
            f"certificate_id={certificate.certificate_id!r}; chart_id={certificate.chart_id!r}",
        ),
        CertificateCheckObligation(
            "planar_lc_coefficient_array_shape",
            shape_ok,
            (
                f"z_shape={solution.z.shape}; z_velocity_shape={solution.z_velocity.shape}; "
                f"pair_energy_shape={solution.pair_energy.shape}; physical_time_shape={solution.physical_time.shape}"
            ),
        ),
        CertificateCheckObligation(
            "finite_coefficients",
            finite_arrays,
            "all regularized binary coefficients are finite",
        ),
        CertificateCheckObligation(
            "positive_masses",
            positive_masses,
            f"masses={tuple(float(value) for value in masses) if masses.ndim == 1 else masses!r}",
        ),
        CertificateCheckObligation(
            "binary_pair_valid",
            pair_valid,
            f"pair={solution.pair!r}",
        ),
        CertificateCheckObligation(
            "finite_nonempty_time_intervals",
            finite_intervals,
            f"parameter_interval={parameter_interval}; physical_time_interval={physical_time_interval}",
        ),
        CertificateCheckObligation(
            "finite_checker_tolerances",
            finite_tolerances,
            (
                f"coefficient_tolerance={coefficient_tolerance}; "
                f"regularized_residual_tolerance={regularized_residual_tolerance}; "
                f"projected_residual_tolerance={projected_residual_tolerance}; "
                f"projection_rho_lower_bound={rho_lower_bound}"
            ),
        ),
        CertificateCheckObligation(
            "interval_physical_time_containment",
            physical_time_contained,
            (
                f"physical_time_interval={physical_time_interval}; "
                f"interval_physical_time_enclosure={physical_time_enclosure}; "
                "diagnostic_sampled_physical_time_contained="
                f"{sampled_physical_time_contained}"
            ),
        ),
        CertificateCheckObligation(
            "planar_lc_regularized_coefficient_recurrence",
            recurrence_certified,
            f"max_coefficient_residual={max_coefficient_residual}",
        ),
        CertificateCheckObligation(
            "planar_lc_pair_energy_constraint",
            pair_energy_constraint_certified,
            f"max_constraint_residual={max_constraint_residual}",
        ),
        CertificateCheckObligation(
            "planar_lc_exact_rational_regularized_residual_polynomials",
            exact_rational_regularized_residual_certified,
            (
                "regularized LC residual polynomials are evaluated with exact "
                "rational interval arithmetic over serialized binary-float "
                "coefficients and parameter endpoints"
            ),
        ),
        CertificateCheckObligation(
            "interval_planar_lc_regularized_rhs_residual",
            interval_regularized_residual_certified,
            (
                "max_interval_regularized_residual="
                f"{max_interval_regularized_residual}; "
                f"diagnostic_max_sampled_regularized_residual="
                f"{max_regularized_residual}"
            ),
        ),
        CertificateCheckObligation(
            "interval_projected_newton_residual_away_from_binary_collision",
            interval_projected_residual_certified,
            (
                f"max_interval_projected_residual={max_interval_projected_residual}; "
                f"diagnostic_max_sampled_projected_residual={max_projected_residual}; "
                f"diagnostic_projected_sample_count={projected_sample_count}"
            ),
        ),
        CertificateCheckObligation(
            "tail_bound_admissible",
            tail_admissible,
            f"tail_bound={tail_bound}",
        ),
    )
    return CertificateCheckResult(
        certificate_id=str(certificate.certificate_id),
        certificate_type="planar_levi_civita_binary",
        checker_id="independent_planar_lc_binary_checker_interval_v2",
        obligations=obligations,
        max_coefficient_residual=float(max_coefficient_residual),
        max_sampled_newton_residual=float(max_projected_residual),
    )


def check_spatial_ks_binary_chart(
    certificate: SpatialKSBinaryChartCertificate,
) -> CertificateCheckResult:
    """Verify a spatial KS binary chart from explicit serialized coefficients."""

    solution = _spatial_ks_solution_from_certificate(certificate)
    masses = np.asarray(certificate.masses, dtype=float)
    parameter_interval = tuple(float(value) for value in certificate.parameter_interval)
    physical_time_interval = tuple(
        float(value) for value in certificate.physical_time_interval
    )
    coefficient_tolerance = float(certificate.coefficient_tolerance)
    regularized_residual_tolerance = float(certificate.regularized_residual_tolerance)
    projected_residual_tolerance = float(certificate.projected_residual_tolerance)
    constraint_tolerance = float(certificate.constraint_tolerance)
    tail_bound = float(certificate.tail_bound)
    sample_count = int(certificate.sample_count)
    rho_lower_bound = float(certificate.projection_rho_lower_bound)

    arrays = (
        solution.u,
        solution.u_velocity,
        solution.pair_energy,
        solution.binary_center,
        solution.binary_center_velocity,
        solution.third_offset,
        solution.third_offset_velocity,
        solution.physical_time,
    )
    shape_ok = bool(
        solution.u.ndim == 2
        and solution.u.shape[1:] == (4,)
        and solution.u.shape[0] >= 2
        and solution.u_velocity.shape == solution.u.shape
        and solution.binary_center.shape == (solution.u.shape[0], 3)
        and solution.binary_center_velocity.shape == solution.binary_center.shape
        and solution.third_offset.shape == solution.binary_center.shape
        and solution.third_offset_velocity.shape == solution.binary_center.shape
        and solution.pair_energy.shape == (solution.u.shape[0],)
        and solution.physical_time.shape == (solution.u.shape[0],)
    )
    finite_arrays = bool(all(np.all(np.isfinite(array)) for array in arrays))
    finite_masses = bool(masses.shape == (3,) and np.all(np.isfinite(masses)))
    positive_masses = bool(finite_masses and np.all(masses > 0.0))
    pair_valid = bool(
        len(solution.pair) == 2
        and solution.pair[0] != solution.pair[1]
        and set(solution.pair).issubset({0, 1, 2})
    )
    finite_intervals = bool(
        _finite_nonempty_interval(parameter_interval)
        and _finite_nonempty_interval(physical_time_interval)
    )
    finite_tolerances = bool(
        np.isfinite(coefficient_tolerance)
        and coefficient_tolerance >= 0.0
        and np.isfinite(regularized_residual_tolerance)
        and regularized_residual_tolerance >= 0.0
        and np.isfinite(projected_residual_tolerance)
        and projected_residual_tolerance >= 0.0
        and np.isfinite(constraint_tolerance)
        and constraint_tolerance >= 0.0
        and np.isfinite(rho_lower_bound)
        and rho_lower_bound >= 0.0
    )
    tail_admissible = bool(np.isfinite(tail_bound) and tail_bound >= 0.0)
    (
        physical_time_contained,
        physical_time_enclosure,
    ) = _interval_physical_time_contained(
        solution.physical_time,
        parameter_interval,
        physical_time_interval,
        tolerance=tail_bound,
    )
    sampled_physical_time_contained = bool(
        finite_intervals
        and shape_ok
        and _sampled_physical_time_contained(
            solution.physical_time,
            parameter_interval,
            physical_time_interval,
            sample_count=max(sample_count, 2),
        )
    )

    max_coefficient_residual = np.inf
    max_relative_coefficient_residual = np.inf
    recurrence_certified = False
    if (
        shape_ok
        and finite_arrays
        and positive_masses
        and pair_valid
        and finite_tolerances
    ):
        try:
            max_coefficient_residual = _max_spatial_ks_coefficient_residual(
                solution,
            )
            max_relative_coefficient_residual = (
                _max_spatial_ks_relative_coefficient_residual(solution)
            )
            recurrence_certified = bool(
                max_coefficient_residual <= coefficient_tolerance
                or max_relative_coefficient_residual <= coefficient_tolerance
            )
        except (FloatingPointError, ValueError):
            max_coefficient_residual = np.inf
            max_relative_coefficient_residual = np.inf

    max_pair_energy_constraint = np.inf
    max_horizontal_constraint = np.inf
    max_interval_pair_energy_constraint = np.inf
    max_interval_horizontal_constraint = np.inf
    constraints_certified = False
    if recurrence_certified:
        try:
            max_pair_energy_constraint = float(
                np.max(
                    np.abs(
                        ks_pair_energy_constraint_coefficients(
                            solution,
                            solution.order,
                        ),
                    ),
                ),
            )
            max_horizontal_constraint = float(
                np.max(
                    np.abs(
                        ks_horizontal_constraint_coefficients(
                            solution,
                            solution.order,
                        ),
                    ),
                ),
            )
            if finite_intervals:
                (
                    max_interval_pair_energy_constraint,
                    max_interval_horizontal_constraint,
                ) = _max_interval_spatial_ks_constraint_residuals(
                    solution,
                    parameter_interval,
                )
            constraints_certified = bool(
                (
                    max_pair_energy_constraint <= constraint_tolerance
                    and max_horizontal_constraint <= constraint_tolerance
                )
                or (
                    max_interval_pair_energy_constraint <= constraint_tolerance
                    and max_interval_horizontal_constraint <= constraint_tolerance
                )
            )
        except (FloatingPointError, ValueError):
            max_pair_energy_constraint = np.inf
            max_horizontal_constraint = np.inf
            max_interval_pair_energy_constraint = np.inf
            max_interval_horizontal_constraint = np.inf

    max_regularized_residual = np.inf
    regularized_residual_certified = False
    max_interval_regularized_residual = np.inf
    interval_regularized_residual_certified = False
    exact_rational_regularized_residual_certified = False
    if recurrence_certified and finite_intervals and finite_tolerances and sample_count >= 2:
        try:
            max_regularized_residual = _max_sampled_spatial_ks_residual(
                solution,
                parameter_interval,
                sample_count=sample_count,
            )
            regularized_residual_certified = bool(
                max_regularized_residual <= regularized_residual_tolerance
            )
        except (FloatingPointError, ValueError):
            max_regularized_residual = np.inf
    if recurrence_certified and finite_intervals and finite_tolerances:
        try:
            max_interval_regularized_residual = (
                _max_interval_spatial_ks_taylor_model_residual(
                    solution,
                    parameter_interval,
                    tail_bound=tail_bound,
                )
            )
            interval_regularized_residual_certified = bool(
                max_interval_regularized_residual <= regularized_residual_tolerance
            )
            exact_rational_regularized_residual_certified = True
        except (FloatingPointError, ValueError):
            max_interval_regularized_residual = np.inf
            exact_rational_regularized_residual_certified = False

    max_projected_residual = np.inf
    projected_sample_count = 0
    projected_residual_certified = False
    max_interval_projected_residual = np.inf
    interval_projected_residual_certified = False
    if recurrence_certified and finite_intervals and finite_tolerances and sample_count >= 2:
        try:
            (
                max_projected_residual,
                projected_sample_count,
            ) = _max_sampled_spatial_ks_projected_newton_residual(
                solution,
                parameter_interval,
                sample_count=sample_count,
                rho_lower_bound=rho_lower_bound,
            )
            projected_residual_certified = bool(
                projected_sample_count > 0
                and max_projected_residual <= projected_residual_tolerance
            )
        except (FloatingPointError, ValueError):
            max_projected_residual = np.inf
            projected_sample_count = 0
    if recurrence_certified and finite_intervals and finite_tolerances:
        try:
            max_interval_projected_residual = (
                _max_interval_spatial_ks_projected_newton_residual(
                    solution,
                    parameter_interval,
                    tail_bound=tail_bound,
                    rho_lower_bound=rho_lower_bound,
                )
            )
            interval_projected_residual_certified = bool(
                max_interval_projected_residual <= projected_residual_tolerance
            )
        except (FloatingPointError, ValueError):
            max_interval_projected_residual = np.inf

    obligations = (
        CertificateCheckObligation(
            "spatial_ks_binary_chart_type",
            certificate.chart_type == "spatial_ks_binary",
            f"chart_type={certificate.chart_type!r}",
        ),
        CertificateCheckObligation(
            "certificate_identity_present",
            bool(certificate.certificate_id and certificate.chart_id),
            f"certificate_id={certificate.certificate_id!r}; chart_id={certificate.chart_id!r}",
        ),
        CertificateCheckObligation(
            "spatial_ks_coefficient_array_shape",
            shape_ok,
            (
                f"u_shape={solution.u.shape}; u_velocity_shape={solution.u_velocity.shape}; "
                f"binary_center_shape={solution.binary_center.shape}; physical_time_shape={solution.physical_time.shape}"
            ),
        ),
        CertificateCheckObligation(
            "finite_coefficients",
            finite_arrays,
            "all spatial KS binary coefficients are finite",
        ),
        CertificateCheckObligation(
            "positive_masses",
            positive_masses,
            f"masses={tuple(float(value) for value in masses) if masses.ndim == 1 else masses!r}",
        ),
        CertificateCheckObligation(
            "binary_pair_valid",
            pair_valid,
            f"pair={solution.pair!r}",
        ),
        CertificateCheckObligation(
            "finite_nonempty_time_intervals",
            finite_intervals,
            f"parameter_interval={parameter_interval}; physical_time_interval={physical_time_interval}",
        ),
        CertificateCheckObligation(
            "finite_checker_tolerances",
            finite_tolerances,
            (
                f"coefficient_tolerance={coefficient_tolerance}; "
                f"regularized_residual_tolerance={regularized_residual_tolerance}; "
                f"projected_residual_tolerance={projected_residual_tolerance}; "
                f"constraint_tolerance={constraint_tolerance}; "
                f"projection_rho_lower_bound={rho_lower_bound}"
            ),
        ),
        CertificateCheckObligation(
            "interval_physical_time_containment",
            physical_time_contained,
            (
                f"physical_time_interval={physical_time_interval}; "
                f"interval_physical_time_enclosure={physical_time_enclosure}; "
                "diagnostic_sampled_physical_time_contained="
                f"{sampled_physical_time_contained}"
            ),
        ),
        CertificateCheckObligation(
            "spatial_ks_regularized_coefficient_recurrence",
            recurrence_certified,
            (
                f"max_coefficient_residual={max_coefficient_residual}; "
                "max_relative_coefficient_residual="
                f"{max_relative_coefficient_residual}"
            ),
        ),
        CertificateCheckObligation(
            "spatial_ks_pair_energy_and_horizontal_constraints",
            constraints_certified,
            (
                f"max_pair_energy_constraint={max_pair_energy_constraint}; "
                f"max_horizontal_constraint={max_horizontal_constraint}; "
                "max_interval_pair_energy_constraint="
                f"{max_interval_pair_energy_constraint}; "
                "max_interval_horizontal_constraint="
                f"{max_interval_horizontal_constraint}"
            ),
        ),
        CertificateCheckObligation(
            "spatial_ks_exact_rational_regularized_residual_polynomials",
            exact_rational_regularized_residual_certified,
            (
                "regularized KS residual polynomials are evaluated with exact "
                "rational interval arithmetic over serialized binary-float "
                "coefficients and parameter endpoints"
            ),
        ),
        CertificateCheckObligation(
            "interval_spatial_ks_regularized_rhs_residual",
            interval_regularized_residual_certified,
            (
                "max_interval_regularized_residual="
                f"{max_interval_regularized_residual}"
            ),
        ),
        CertificateCheckObligation(
            "interval_spatial_ks_projected_newton_residual_away_from_binary_collision",
            interval_projected_residual_certified,
            (
                f"max_interval_projected_residual={max_interval_projected_residual}; "
                f"diagnostic_max_sampled_projected_residual={max_projected_residual}; "
                f"diagnostic_projected_sample_count={projected_sample_count}"
            ),
        ),
        CertificateCheckObligation(
            "tail_bound_admissible",
            tail_admissible,
            f"tail_bound={tail_bound}",
        ),
    )
    return CertificateCheckResult(
        certificate_id=str(certificate.certificate_id),
        certificate_type="spatial_ks_binary",
        checker_id="independent_spatial_ks_binary_checker_interval_v2",
        obligations=obligations,
        max_coefficient_residual=float(max_coefficient_residual),
        max_sampled_newton_residual=float(max_projected_residual),
    )


def check_total_collision_fuchsian_stop_chart(
    certificate: TotalCollisionFuchsianStopChartCertificate,
) -> CertificateCheckResult:
    """Verify a serialized finite Fuchsian-log total-collision stop chart."""

    branch = _fuchsian_log_branch_from_certificate(certificate)
    masses = np.asarray(certificate.masses, dtype=float)
    central_shape = np.asarray(certificate.central_shape, dtype=float)
    tau_interval = tuple(float(value) for value in certificate.tau_interval)
    physical_time_interval = tuple(
        float(value) for value in certificate.physical_time_interval
    )
    event_time = float(certificate.event_physical_time)
    total_collision_tau = float(certificate.total_collision_tau)
    isolation_radius = float(certificate.isolation_radius)
    residual_tolerance = float(certificate.residual_tolerance)
    angular_tolerance = float(certificate.angular_momentum_tolerance)
    tail_bound = float(certificate.tail_bound)
    sample_count = int(certificate.sample_count)

    shape_ok = bool(
        central_shape.ndim == 2
        and central_shape.shape[0] == 3
        and central_shape.shape[1] in (2, 3)
        and masses.shape == (3,)
    )
    finite_masses = bool(masses.shape == (3,) and np.all(np.isfinite(masses)))
    positive_masses = bool(finite_masses and np.all(masses > 0.0))
    finite_terms = _fuchsian_log_terms_finite(certificate.terms, central_shape.shape)
    finite_coefficients = bool(
        shape_ok
        and np.all(np.isfinite(central_shape))
        and np.isfinite(certificate.scale_coefficient)
        and finite_terms
    )
    finite_intervals = bool(
        _finite_nonempty_interval(tau_interval)
        and _finite_nonempty_interval(physical_time_interval)
        and np.isfinite(event_time)
        and np.isfinite(total_collision_tau)
    )
    tau_interval_straddles = bool(
        finite_intervals
        and tau_interval[0] < total_collision_tau < tau_interval[1]
        and abs(total_collision_tau) <= 1.0e-15
    )
    physical_time_matches = bool(
        finite_intervals
        and np.isclose(
            physical_time_interval[0],
            event_time + tau_interval[0] ** 3,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        and np.isclose(
            physical_time_interval[1],
            event_time + tau_interval[1] ** 3,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        and physical_time_interval[0] < event_time < physical_time_interval[1]
    )
    finite_tolerances = bool(
        np.isfinite(residual_tolerance)
        and residual_tolerance >= 0.0
        and np.isfinite(angular_tolerance)
        and angular_tolerance >= 0.0
    )
    tail_admissible = bool(np.isfinite(tail_bound) and tail_bound >= 0.0)
    primitive_cauchy_inputs_present = certificate.primitive_cauchy_inputs is not None
    primitive_cauchy_inputs_certified = False
    primitive_cauchy_tail_bound = np.inf
    primitive_cauchy_detail = "primitive Cauchy inputs not supplied"
    primitive_cauchy_shell_inside_isolation = False
    primitive_residual_tail_certified = False
    primitive_residual_tail_bound = np.inf
    primitive_residual_tail_detail = "primitive Cauchy inputs not supplied"
    endpoint_collapse_certified = False
    endpoint_collapse_detail = "primitive Cauchy inputs not supplied"
    endpoint_first_shell_position_bound = np.inf
    finite_energy_certified = False
    finite_energy_limit = np.inf
    finite_energy_detail = "finite coefficients not certified"
    if finite_coefficients and positive_masses:
        (
            finite_energy_certified,
            finite_energy_limit,
            finite_energy_detail,
        ) = _check_fuchsian_supported_scale_finite_energy_matching(branch)
    if primitive_cauchy_inputs_present:
        (
            primitive_cauchy_inputs_certified,
            primitive_cauchy_tail_bound,
            primitive_cauchy_detail,
        ) = _check_fuchsian_primitive_cauchy_inputs(
            certificate.primitive_cauchy_inputs,
            branch=branch,
        )
        (
            primitive_residual_tail_certified,
            primitive_residual_tail_bound,
            primitive_residual_tail_detail,
        ) = _check_fuchsian_primitive_cauchy_residual_tail(
            certificate.primitive_cauchy_inputs,
            residual_tolerance=residual_tolerance,
        )
        (
            endpoint_collapse_certified,
            endpoint_first_shell_position_bound,
            endpoint_collapse_detail,
        ) = _check_fuchsian_endpoint_collapse_envelope(
            certificate.primitive_cauchy_inputs,
        )
        primitive_cauchy_shell_inside_isolation = bool(
            np.isfinite(certificate.primitive_cauchy_inputs.initial_radius)
            and np.isfinite(isolation_radius)
            and 0.0 < certificate.primitive_cauchy_inputs.initial_radius
            and certificate.primitive_cauchy_inputs.initial_radius <= isolation_radius
        )
    primitive_cauchy_tail_covered = bool(
        primitive_cauchy_inputs_present
        and primitive_cauchy_inputs_certified
        and tail_admissible
        and np.isfinite(primitive_cauchy_tail_bound)
        and tail_bound + 1.0e-15 >= primitive_cauchy_tail_bound
    )

    central_shape_collision_free = bool(
        shape_ok and _minimum_pair_distance(central_shape) > 0.0
    )
    central_floor_gap = np.inf
    shape_floor_gap = np.inf
    deviation_gap = np.inf
    isolation_certified = False
    if (
        shape_ok
        and finite_coefficients
        and positive_masses
        and central_shape_collision_free
        and np.isfinite(isolation_radius)
        and 0.0 < isolation_radius < 1.0
    ):
        try:
            recomputed_isolation = _fuchsian_log_total_collision_isolation_from_certificate(
                certificate,
            )
            central_floor_gap = abs(
                recomputed_isolation.central_shape_pair_distance_floor
                - float(certificate.central_shape_pair_distance_floor)
            )
            shape_floor_gap = abs(
                recomputed_isolation.shape_pair_distance_floor
                - float(certificate.shape_pair_distance_floor)
            )
            deviation_gap = abs(
                recomputed_isolation.shape_deviation_bound
                - float(certificate.shape_deviation_bound)
            )
            isolation_certified = bool(
                recomputed_isolation.certified
                and central_floor_gap <= max(1.0e-12, 1.0e-10 * abs(recomputed_isolation.central_shape_pair_distance_floor))
                and shape_floor_gap <= max(1.0e-12, 1.0e-10 * abs(recomputed_isolation.shape_pair_distance_floor))
                and deviation_gap <= max(1.0e-12, 1.0e-10 * abs(recomputed_isolation.shape_deviation_bound))
            )
        except (FloatingPointError, ValueError):
            central_floor_gap = np.inf
            shape_floor_gap = np.inf
            deviation_gap = np.inf

    max_lifted_residual = np.inf
    max_projected_residual = np.inf
    max_interval_lifted_residual = np.inf
    max_interval_projected_residual = np.inf
    max_interval_angular_momentum = np.inf
    max_interval_center_of_mass = np.inf
    max_interval_linear_momentum = np.inf
    max_angular_momentum = np.inf
    max_position_scale = np.inf
    sampled_count = 0
    interval_lifted_residual_certified = False
    interval_lifted_residual_detail = "primitive Cauchy inputs not supplied"
    interval_projected_residual_certified = False
    interval_projected_residual_detail = "primitive Cauchy inputs not supplied"
    interval_angular_certified = False
    interval_angular_detail = "primitive Cauchy inputs not supplied"
    interval_com_momentum_certified = False
    interval_com_momentum_detail = "primitive Cauchy inputs not supplied"
    sampled_residual_certified = False
    angular_certified = False
    scaling_certified = False
    if (
        isolation_certified
        and tau_interval_straddles
        and physical_time_matches
        and finite_tolerances
        and sample_count >= 2
    ):
        try:
            (
                max_lifted_residual,
                max_projected_residual,
                max_angular_momentum,
                max_position_scale,
                sampled_count,
            ) = _max_sampled_fuchsian_stop_chart_quantities(
                branch,
                tau_interval,
                sample_count=sample_count,
            )
            sampled_residual_certified = bool(
                sampled_count > 0
                and max_lifted_residual <= residual_tolerance
                and max_projected_residual <= residual_tolerance
            )
            angular_certified = bool(
                sampled_count > 0 and max_angular_momentum <= angular_tolerance
            )
            scaling_certified = bool(
                sampled_count > 0
                and np.isfinite(max_position_scale)
                and max_position_scale > 0.0
            )
        except (FloatingPointError, ValueError):
            max_lifted_residual = np.inf
            max_projected_residual = np.inf
            max_angular_momentum = np.inf
            max_position_scale = np.inf
            sampled_count = 0
    if (
        primitive_cauchy_inputs_present
        and isolation_certified
        and tau_interval_straddles
        and finite_tolerances
    ):
        try:
            (
                max_interval_lifted_residual,
                interval_lifted_residual_detail,
            ) = _max_interval_fuchsian_lifted_residual_on_punctured_shells(
                branch,
                tau_interval,
                isolation_radius=float(isolation_radius),
                primitive_cauchy_inputs=certificate.primitive_cauchy_inputs,
            )
            interval_lifted_residual_certified = bool(
                np.isfinite(max_interval_lifted_residual)
                and max_interval_lifted_residual <= residual_tolerance
            )
            (
                max_interval_projected_residual,
                interval_projected_residual_detail,
            ) = _max_interval_fuchsian_projected_residual_on_punctured_shells(
                branch,
                tau_interval,
                isolation_radius=float(isolation_radius),
                primitive_cauchy_inputs=certificate.primitive_cauchy_inputs,
            )
            interval_projected_residual_certified = bool(
                np.isfinite(max_interval_projected_residual)
                and max_interval_projected_residual <= residual_tolerance
            )
            (
                max_interval_angular_momentum,
                interval_angular_detail,
            ) = _max_interval_fuchsian_zero_angular_momentum_on_punctured_shells(
                branch,
                tau_interval,
                isolation_radius=float(isolation_radius),
                primitive_cauchy_inputs=certificate.primitive_cauchy_inputs,
            )
            interval_angular_certified = bool(
                np.isfinite(max_interval_angular_momentum)
                and max_interval_angular_momentum <= angular_tolerance
            )
            (
                max_interval_center_of_mass,
                max_interval_linear_momentum,
                interval_com_momentum_detail,
            ) = _max_interval_fuchsian_center_of_mass_and_linear_momentum_on_punctured_shells(
                branch,
                tau_interval,
                isolation_radius=float(isolation_radius),
                primitive_cauchy_inputs=certificate.primitive_cauchy_inputs,
            )
            interval_com_momentum_certified = bool(
                np.isfinite(max_interval_center_of_mass)
                and np.isfinite(max_interval_linear_momentum)
                and max_interval_center_of_mass <= angular_tolerance
                and max_interval_linear_momentum <= angular_tolerance
            )
        except (FloatingPointError, ValueError):
            max_interval_lifted_residual = np.inf
            max_interval_projected_residual = np.inf
            max_interval_angular_momentum = np.inf
            max_interval_center_of_mass = np.inf
            max_interval_linear_momentum = np.inf
            interval_lifted_residual_detail = (
                "interval lifted residual computation failed"
            )
            interval_projected_residual_detail = (
                "interval projected residual computation failed"
            )
            interval_angular_detail = "interval zero-angular computation failed"
            interval_com_momentum_detail = (
                "interval COM/linear-momentum computation failed"
            )

    obligations = [
        CertificateCheckObligation(
            "total_collision_fuchsian_stop_chart_type",
            certificate.chart_type == "total_collision_fuchsian_stop",
            f"chart_type={certificate.chart_type!r}",
        ),
        CertificateCheckObligation(
            "certificate_identity_present",
            bool(certificate.certificate_id and certificate.chart_id),
            f"certificate_id={certificate.certificate_id!r}; chart_id={certificate.chart_id!r}",
        ),
        CertificateCheckObligation(
            "total_collision_fuchsian_shape",
            shape_ok,
            f"central_shape={central_shape.shape}; masses_shape={masses.shape}",
        ),
        CertificateCheckObligation(
            "finite_coefficients",
            finite_coefficients,
            "central shape, scale coefficient, and finite Fuchsian-log terms are finite",
        ),
        CertificateCheckObligation(
            "positive_masses",
            positive_masses,
            f"masses={tuple(float(value) for value in masses) if masses.ndim == 1 else masses!r}",
        ),
        CertificateCheckObligation(
            "finite_nonempty_tau_interval",
            finite_intervals,
            f"tau_interval={tau_interval}; physical_time_interval={physical_time_interval}",
        ),
        CertificateCheckObligation(
            "punctured_tau_interval_straddles_total_collision",
            tau_interval_straddles,
            f"tau_interval={tau_interval}; total_collision_tau={total_collision_tau}",
        ),
        CertificateCheckObligation(
            "fuchsian_physical_time_interval_matches_cubic_time",
            physical_time_matches,
            f"event_physical_time={event_time}; physical_time_interval={physical_time_interval}",
        ),
        CertificateCheckObligation(
            "finite_checker_tolerances",
            finite_tolerances,
            (
                f"residual_tolerance={residual_tolerance}; "
                f"angular_momentum_tolerance={angular_tolerance}"
            ),
        ),
        CertificateCheckObligation(
            "central_shape_collision_free",
            central_shape_collision_free,
            f"minimum_pair_distance={_minimum_pair_distance(central_shape) if shape_ok else np.inf}",
        ),
        CertificateCheckObligation(
            "punctured_total_collision_isolation_data",
            isolation_certified,
            (
                f"central_floor_gap={central_floor_gap}; "
                f"shape_floor_gap={shape_floor_gap}; deviation_gap={deviation_gap}"
            ),
        ),
        CertificateCheckObligation(
            "interval_fuchsian_lifted_residual_on_punctured_shells",
            interval_lifted_residual_certified,
            (
                f"max_interval_lifted_residual={max_interval_lifted_residual}; "
                f"{interval_lifted_residual_detail}; "
                f"diagnostic_max_sampled_lifted_residual={max_lifted_residual}; "
                f"diagnostic_sampled_count={sampled_count}"
            ),
        ),
        CertificateCheckObligation(
            "interval_fuchsian_projected_residual_on_punctured_shells",
            interval_projected_residual_certified,
            (
                f"max_interval_projected_residual={max_interval_projected_residual}; "
                f"{interval_projected_residual_detail}; "
                f"diagnostic_max_sampled_projected_residual={max_projected_residual}; "
                f"diagnostic_sampled_count={sampled_count}"
            ),
        ),
        CertificateCheckObligation(
            "interval_zero_angular_momentum_on_punctured_shells",
            interval_angular_certified,
            (
                f"max_interval_angular_momentum={max_interval_angular_momentum}; "
                f"{interval_angular_detail}; "
                f"diagnostic_max_sampled_angular_momentum={max_angular_momentum}; "
                f"diagnostic_sampled_count={sampled_count}"
            ),
        ),
        CertificateCheckObligation(
            "interval_center_of_mass_and_linear_momentum_on_punctured_shells",
            interval_com_momentum_certified,
            (
                f"max_interval_center_of_mass={max_interval_center_of_mass}; "
                f"max_interval_linear_momentum={max_interval_linear_momentum}; "
                f"{interval_com_momentum_detail}"
            ),
        ),
        CertificateCheckObligation(
            "interval_fuchsian_supported_scale_finite_energy_matching",
            finite_energy_certified,
            f"finite_energy_limit={finite_energy_limit}; {finite_energy_detail}",
        ),
        CertificateCheckObligation(
            "interval_total_collision_endpoint_collapse_envelope",
            endpoint_collapse_certified,
            (
                "first_shell_position_bound="
                f"{endpoint_first_shell_position_bound}; "
                f"{endpoint_collapse_detail}; "
                f"diagnostic_max_position_over_tau_squared={max_position_scale}; "
                f"diagnostic_sampled_count={sampled_count}"
            ),
        ),
        CertificateCheckObligation(
            "tail_bound_admissible",
            tail_admissible,
            f"tail_bound={tail_bound}",
        ),
        CertificateCheckObligation(
            "fuchsian_primitive_cauchy_inputs_present",
            primitive_cauchy_inputs_present,
            (
                "finite Fuchsian-log stop charts require serialized primitive "
                "Cauchy inputs; sampled residuals alone are not proof-grade"
            ),
        ),
    ]
    if primitive_cauchy_inputs_present:
        obligations.extend(
            (
                CertificateCheckObligation(
                    "fuchsian_primitive_cauchy_inputs_certify",
                    primitive_cauchy_inputs_certified,
                    primitive_cauchy_detail,
                ),
                CertificateCheckObligation(
                    "fuchsian_primitive_cauchy_shell_inside_isolation",
                    primitive_cauchy_shell_inside_isolation,
                    (
                        "initial_radius="
                        f"{certificate.primitive_cauchy_inputs.initial_radius}; "
                        f"isolation_radius={isolation_radius}"
                    ),
                ),
                CertificateCheckObligation(
                    "fuchsian_tail_bound_covers_primitive_cauchy_tail",
                    primitive_cauchy_tail_covered,
                    (
                        f"tail_bound={tail_bound}; "
                        f"primitive_tail_bound={primitive_cauchy_tail_bound}"
                    ),
                ),
                CertificateCheckObligation(
                    "fuchsian_primitive_cauchy_residual_tail_within_tolerance",
                    primitive_residual_tail_certified,
                    (
                        f"residual_tolerance={residual_tolerance}; "
                        f"primitive_residual_tail_bound={primitive_residual_tail_bound}; "
                        f"{primitive_residual_tail_detail}"
                    ),
                ),
            )
        )
    return CertificateCheckResult(
        certificate_id=str(certificate.certificate_id),
        certificate_type="total_collision_fuchsian_stop",
        checker_id="independent_total_collision_fuchsian_stop_checker_interval_v2",
        obligations=tuple(obligations),
        max_coefficient_residual=float(max_interval_lifted_residual),
        max_sampled_newton_residual=float(max_interval_projected_residual),
    )


def check_total_collision_generalized_fuchsian_stop_chart(
    certificate: TotalCollisionGeneralizedFuchsianStopChartCertificate,
) -> CertificateCheckResult:
    """Verify a serialized generalized Fuchsian total-collision stop chart."""

    branch: FuchsianShapeBranch | None = None
    branch_error = ""
    try:
        branch = _generalized_fuchsian_branch_from_certificate(certificate)
    except (FloatingPointError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        branch_error = str(error)

    masses = np.asarray(certificate.masses, dtype=float)
    central_shape = np.asarray(certificate.central_shape, dtype=float)
    tau_interval = tuple(float(value) for value in certificate.tau_interval)
    physical_time_interval = tuple(
        float(value) for value in certificate.physical_time_interval
    )
    event_time = float(certificate.event_physical_time)
    total_collision_tau = float(certificate.total_collision_tau)
    isolation_radius = float(certificate.isolation_radius)
    residual_tolerance = float(certificate.residual_tolerance)
    projected_residual_tolerance_data = getattr(
        certificate,
        "projected_residual_tolerance",
        None,
    )
    projected_residual_tolerance = (
        residual_tolerance
        if projected_residual_tolerance_data is None
        else float(projected_residual_tolerance_data)
    )
    angular_tolerance = float(certificate.angular_momentum_tolerance)
    tail_bound = float(certificate.tail_bound)
    sample_count = int(certificate.sample_count)

    shape_ok = bool(
        central_shape.ndim == 2
        and central_shape.shape[0] == 3
        and central_shape.shape[1] in (2, 3)
        and masses.shape == (3,)
    )
    finite_masses = bool(masses.shape == (3,) and np.all(np.isfinite(masses)))
    positive_masses = bool(finite_masses and np.all(masses > 0.0))
    selected_coefficients_finite = bool(
        shape_ok
        and all(
            len(item.index) == len(certificate.powers)
            and sum(item.index) <= int(certificate.max_total_degree)
            and np.asarray(item.coefficient, dtype=float).shape == central_shape.shape
            and np.all(np.isfinite(np.asarray(item.coefficient, dtype=float)))
            for item in certificate.selected_coefficients
        )
    )
    finite_coefficients = bool(
        shape_ok
        and np.all(np.isfinite(central_shape))
        and all(np.isfinite(float(power)) and float(power) > 0.0 for power in certificate.powers)
        and int(certificate.max_total_degree) >= 0
        and selected_coefficients_finite
    )
    finite_intervals = bool(
        _finite_nonempty_interval(tau_interval)
        and _finite_nonempty_interval(physical_time_interval)
        and np.isfinite(event_time)
        and np.isfinite(total_collision_tau)
    )
    tau_interval_straddles = bool(
        finite_intervals
        and tau_interval[0] < total_collision_tau < tau_interval[1]
        and abs(total_collision_tau) <= 1.0e-15
    )
    physical_time_matches = bool(
        finite_intervals
        and np.isclose(
            physical_time_interval[0],
            event_time + tau_interval[0] ** 3,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        and np.isclose(
            physical_time_interval[1],
            event_time + tau_interval[1] ** 3,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        and physical_time_interval[0] < event_time < physical_time_interval[1]
    )
    finite_tolerances = bool(
        np.isfinite(residual_tolerance)
        and residual_tolerance >= 0.0
        and np.isfinite(projected_residual_tolerance)
        and projected_residual_tolerance >= 0.0
        and np.isfinite(angular_tolerance)
        and angular_tolerance >= 0.0
    )
    tail_admissible = bool(np.isfinite(tail_bound) and tail_bound >= 0.0)
    branch_reconstructs = branch is not None
    recurrence_certified = bool(branch is not None and branch.recurrence_certified)
    finite_energy_certified = bool(
        branch is not None
        and branch.scale_index is not None
        and np.isfinite(branch.finite_energy_limit)
    )
    finite_energy_limit = (
        float(branch.finite_energy_limit) if branch is not None else float("inf")
    )

    central_shape_collision_free = bool(
        shape_ok and _minimum_pair_distance(central_shape) > 0.0
    )
    central_floor_gap = np.inf
    shape_floor_gap = np.inf
    deviation_gap = np.inf
    isolation_certified = False
    if (
        branch is not None
        and shape_ok
        and finite_coefficients
        and positive_masses
        and central_shape_collision_free
        and np.isfinite(isolation_radius)
        and 0.0 < isolation_radius < 1.0
    ):
        try:
            central_floor = _minimum_pair_distance(branch.central_shape)
            deviation = _generalized_fuchsian_shape_deviation_bound_for_checker(
                branch,
                isolation_radius,
            )
            dimension = int(branch.central_shape.shape[1])
            shape_floor = central_floor - 2.0 * np.sqrt(float(dimension)) * deviation
            central_floor_gap = abs(
                central_floor - float(certificate.central_shape_pair_distance_floor)
            )
            shape_floor_gap = abs(
                shape_floor - float(certificate.shape_pair_distance_floor)
            )
            deviation_gap = abs(
                deviation - float(certificate.shape_deviation_bound)
            )
            isolation_certified = bool(
                np.isfinite(central_floor)
                and central_floor > 0.0
                and np.isfinite(deviation)
                and shape_floor > 0.0
                and _finite_close(
                    float(certificate.central_shape_pair_distance_floor),
                    central_floor,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )
                and _finite_close(
                    float(certificate.shape_pair_distance_floor),
                    shape_floor,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )
                and _finite_close(
                    float(certificate.shape_deviation_bound),
                    deviation,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )
            )
        except (FloatingPointError, ValueError):
            central_floor_gap = np.inf
            shape_floor_gap = np.inf
            deviation_gap = np.inf

    majorant_checks = _check_generalized_remainder_majorant(
        certificate.remainder_majorant,
        residual_tolerance=residual_tolerance,
        tail_bound=tail_bound,
    )
    majorant_initial_radius = float(majorant_checks["initial_radius"])
    remainder_shell_inside_isolation = bool(
        np.isfinite(majorant_initial_radius)
        and np.isfinite(isolation_radius)
        and 0.0 < majorant_initial_radius <= isolation_radius
    )

    max_lifted_residual = np.inf
    max_interval_lifted_residual = np.inf
    max_interval_projected_residual = np.inf
    max_interval_angular_momentum = np.inf
    max_interval_center_of_mass = np.inf
    max_interval_linear_momentum = np.inf
    max_angular_momentum = np.inf
    max_position_scale = np.inf
    sampled_count = 0
    sampled_residual_certified = False
    interval_lifted_residual_certified = False
    cauchy_projected_residual_certified = False
    interval_angular_certified = False
    interval_com_momentum_certified = False
    interval_lifted_residual_detail = "remainder majorant not supplied"
    interval_projected_residual_detail = "remainder majorant not supplied"
    interval_angular_detail = "remainder majorant not supplied"
    interval_com_momentum_detail = "remainder majorant not supplied"
    angular_certified = False
    scaling_certified = False
    if (
        branch is not None
        and isolation_certified
        and tau_interval_straddles
        and physical_time_matches
        and finite_tolerances
        and sample_count >= 2
    ):
        try:
            (
                max_lifted_residual,
                max_angular_momentum,
                max_position_scale,
                sampled_count,
            ) = _max_sampled_generalized_fuchsian_stop_chart_quantities(
                branch,
                tau_interval,
                sample_count=sample_count,
            )
            sampled_residual_certified = bool(
                sampled_count > 0
                and np.isfinite(max_lifted_residual)
                and max_lifted_residual + float(majorant_checks["lifted_residual_tail_bound"])
                <= residual_tolerance * (1.0 + 1.0e-12)
            )
            angular_certified = bool(
                sampled_count > 0 and max_angular_momentum <= angular_tolerance
            )
            scaling_certified = bool(
                sampled_count > 0
                and np.isfinite(max_position_scale)
                and max_position_scale > 0.0
            )
        except (FloatingPointError, ValueError):
            max_lifted_residual = np.inf
            max_angular_momentum = np.inf
            max_position_scale = np.inf
            sampled_count = 0
    if (
        branch is not None
        and certificate.remainder_majorant is not None
        and isolation_certified
        and tau_interval_straddles
        and finite_tolerances
    ):
        try:
            (
                max_interval_lifted_residual,
                interval_lifted_residual_detail,
            ) = _max_interval_generalized_fuchsian_lifted_residual_on_punctured_shells(
                branch,
                tau_interval,
                isolation_radius=float(isolation_radius),
                remainder_majorant=certificate.remainder_majorant,
            )
            interval_lifted_residual_certified = bool(
                np.isfinite(max_interval_lifted_residual)
                and max_interval_lifted_residual
                + float(majorant_checks["lifted_residual_tail_bound"])
                <= residual_tolerance * (1.0 + 1.0e-12)
            )
            (
                max_interval_projected_residual,
                interval_projected_residual_detail,
            ) = _max_interval_generalized_fuchsian_projected_residual_on_punctured_shells(
                branch,
                tau_interval,
                isolation_radius=float(isolation_radius),
                remainder_majorant=certificate.remainder_majorant,
            )
            cauchy_projected_residual_certified = bool(
                majorant_checks["required_residual_components_present"]
                and majorant_checks["physical_residual_tail_certified"]
                and np.isfinite(max_interval_projected_residual)
                and max_interval_projected_residual
                + float(majorant_checks["physical_residual_tail_bound"])
                <= projected_residual_tolerance * (1.0 + 1.0e-12)
            )
            (
                max_interval_angular_momentum,
                interval_angular_detail,
            ) = _max_interval_generalized_fuchsian_zero_angular_momentum_on_punctured_shells(
                branch,
                tau_interval,
                isolation_radius=float(isolation_radius),
                remainder_majorant=certificate.remainder_majorant,
            )
            interval_angular_certified = bool(
                np.isfinite(max_interval_angular_momentum)
                and max_interval_angular_momentum <= angular_tolerance
            )
            (
                max_interval_center_of_mass,
                max_interval_linear_momentum,
                interval_com_momentum_detail,
            ) = _max_interval_generalized_fuchsian_center_of_mass_and_linear_momentum_on_punctured_shells(
                branch,
                tau_interval,
                isolation_radius=float(isolation_radius),
                remainder_majorant=certificate.remainder_majorant,
            )
            interval_com_momentum_certified = bool(
                np.isfinite(max_interval_center_of_mass)
                and np.isfinite(max_interval_linear_momentum)
                and max_interval_center_of_mass <= angular_tolerance
                and max_interval_linear_momentum <= angular_tolerance
            )
        except (FloatingPointError, ValueError):
            max_interval_lifted_residual = np.inf
            max_interval_projected_residual = np.inf
            max_interval_angular_momentum = np.inf
            max_interval_center_of_mass = np.inf
            max_interval_linear_momentum = np.inf
            interval_lifted_residual_detail = (
                "interval generalized lifted residual computation failed"
            )
            interval_projected_residual_detail = (
                "diagnostic interval generalized projected residual computation failed"
            )
            interval_angular_detail = "interval generalized angular computation failed"
            interval_com_momentum_detail = (
                "interval generalized COM/linear-momentum computation failed"
            )

    obligations = [
        CertificateCheckObligation(
            "total_collision_generalized_fuchsian_stop_chart_type",
            certificate.chart_type == "total_collision_generalized_fuchsian_stop",
            f"chart_type={certificate.chart_type!r}",
        ),
        CertificateCheckObligation(
            "certificate_identity_present",
            bool(certificate.certificate_id and certificate.chart_id),
            f"certificate_id={certificate.certificate_id!r}; chart_id={certificate.chart_id!r}",
        ),
        CertificateCheckObligation(
            "total_collision_generalized_fuchsian_shape",
            shape_ok,
            f"central_shape={central_shape.shape}; masses_shape={masses.shape}",
        ),
        CertificateCheckObligation(
            "finite_coefficients",
            finite_coefficients,
            "central shape, powers, and selected generalized coefficients are finite",
        ),
        CertificateCheckObligation(
            "positive_masses",
            positive_masses,
            f"masses={tuple(float(value) for value in masses) if masses.ndim == 1 else masses!r}",
        ),
        CertificateCheckObligation(
            "finite_nonempty_tau_interval",
            finite_intervals,
            f"tau_interval={tau_interval}; physical_time_interval={physical_time_interval}",
        ),
        CertificateCheckObligation(
            "punctured_tau_interval_straddles_total_collision",
            tau_interval_straddles,
            f"tau_interval={tau_interval}; total_collision_tau={total_collision_tau}",
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_physical_time_interval_matches_cubic_time",
            physical_time_matches,
            f"event_physical_time={event_time}; physical_time_interval={physical_time_interval}",
        ),
        CertificateCheckObligation(
            "finite_checker_tolerances",
            finite_tolerances,
            (
                f"residual_tolerance={residual_tolerance}; "
                f"angular_momentum_tolerance={angular_tolerance}"
            ),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_branch_reconstructs",
            branch_reconstructs,
            branch_error or "constructor replay accepted serialized selected rows",
        ),
        CertificateCheckObligation(
            "nonresonant_generalized_fuchsian_recurrence",
            recurrence_certified,
            (
                "max_recurrence_residual="
                f"{branch.max_recurrence_residual_norm if branch is not None else np.inf}"
            ),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_finite_energy_scale_row",
            finite_energy_certified,
            (
                f"scale_index={certificate.scale_index!r}; "
                f"finite_energy_limit={finite_energy_limit}"
            ),
        ),
        CertificateCheckObligation(
            "central_shape_collision_free",
            central_shape_collision_free,
            f"minimum_pair_distance={_minimum_pair_distance(central_shape) if shape_ok else np.inf}",
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_punctured_isolation_data",
            isolation_certified,
            (
                f"central_floor_gap={central_floor_gap}; "
                f"shape_floor_gap={shape_floor_gap}; deviation_gap={deviation_gap}"
            ),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_majorant_present",
            bool(certificate.remainder_majorant is not None),
            "serialized generalized stop chart requires a Banach remainder majorant",
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_constants_finite",
            bool(majorant_checks["constants_finite"]),
            str(majorant_checks["constant_detail"]),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_contraction_factor",
            bool(majorant_checks["contraction_certified"]),
            str(majorant_checks["contraction_detail"]),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_self_map",
            bool(majorant_checks["self_map_certified"]),
            str(majorant_checks["self_map_detail"]),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_component_inputs_certify",
            bool(majorant_checks["component_inputs_certified"]),
            str(majorant_checks["component_detail"]),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_majorant_certifies",
            bool(majorant_checks["certified"]),
            str(majorant_checks["detail"]),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_required_residual_components",
            bool(majorant_checks["required_residual_components_present"]),
            str(majorant_checks["required_residual_component_detail"]),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_shell_inside_isolation",
            remainder_shell_inside_isolation,
            (
                f"initial_radius={majorant_initial_radius}; "
                f"isolation_radius={isolation_radius}"
            ),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_tail_bound_covers_remainder_tail",
            bool(majorant_checks["tail_covered"]),
            (
                f"tail_bound={tail_bound}; "
                f"remainder_tail_bound={majorant_checks['tail_bound']}"
            ),
        ),
        CertificateCheckObligation(
            "interval_generalized_fuchsian_lifted_residual_on_punctured_shells",
            interval_lifted_residual_certified,
            (
                f"max_interval_lifted_residual={max_interval_lifted_residual}; "
                f"{interval_lifted_residual_detail}; "
                f"diagnostic_max_sampled_lifted_residual={max_lifted_residual}; "
                f"diagnostic_lifted_residual_tail={majorant_checks['lifted_residual_tail_bound']}; "
                f"diagnostic_sampled_count={sampled_count}"
            ),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_remainder_residual_tail_within_tolerance",
            bool(majorant_checks["lifted_residual_tail_certified"]),
            (
                f"residual_tolerance={residual_tolerance}; "
                f"lifted_residual_tail={majorant_checks['lifted_residual_tail_bound']}"
            ),
        ),
        CertificateCheckObligation(
            "cauchy_generalized_fuchsian_projected_residual_tail_on_punctured_shells",
            cauchy_projected_residual_certified,
            (
                f"projected_residual_tolerance={projected_residual_tolerance}; "
                f"physical_residual_tail={majorant_checks['physical_residual_tail_bound']}; "
                f"{majorant_checks['required_residual_component_detail']}; "
                "diagnostic_direct_interval_projected_residual="
                f"{max_interval_projected_residual}; "
                f"{interval_projected_residual_detail}"
            ),
        ),
        CertificateCheckObligation(
            "interval_generalized_zero_angular_momentum_on_punctured_shells",
            interval_angular_certified,
            (
                f"max_interval_angular_momentum={max_interval_angular_momentum}; "
                f"{interval_angular_detail}; "
                f"diagnostic_max_sampled_angular_momentum={max_angular_momentum}; "
                f"diagnostic_sampled_count={sampled_count}"
            ),
        ),
        CertificateCheckObligation(
            "interval_generalized_center_of_mass_and_linear_momentum_on_punctured_shells",
            interval_com_momentum_certified,
            (
                f"max_interval_center_of_mass={max_interval_center_of_mass}; "
                f"max_interval_linear_momentum={max_interval_linear_momentum}; "
                f"{interval_com_momentum_detail}"
            ),
        ),
        CertificateCheckObligation(
            "generalized_fuchsian_endpoint_collapse_envelope",
            bool(majorant_checks["endpoint_collapse_certified"]),
            (
                f"{majorant_checks['endpoint_detail']}; "
                f"diagnostic_max_position_over_tau_squared={max_position_scale}; "
                f"diagnostic_sampled_count={sampled_count}"
            ),
        ),
        CertificateCheckObligation(
            "tail_bound_admissible",
            tail_admissible,
            f"tail_bound={tail_bound}",
        ),
    ]
    return CertificateCheckResult(
        certificate_id=str(certificate.certificate_id),
        certificate_type="total_collision_generalized_fuchsian_stop",
        checker_id="independent_total_collision_generalized_fuchsian_stop_checker_interval_cauchy_projected_v3",
        obligations=tuple(obligations),
        max_coefficient_residual=float(
            max(
                max_interval_lifted_residual,
                float(majorant_checks["lifted_residual_tail_bound"]),
                float(majorant_checks["physical_residual_tail_bound"]),
            )
        ),
        max_sampled_newton_residual=float(
            max(
                max_interval_lifted_residual,
                max_interval_projected_residual,
                float(majorant_checks["physical_residual_tail_bound"]),
            )
        ),
    )


def check_initial_value_problem_binding(
    binding: InitialValueProblemBindingCertificate,
    charts: Iterable[
        OrdinaryTaylorChartCertificate
        | PlanarLeviCivitaBinaryChartCertificate
        | SpatialKSBinaryChartCertificate
    ],
) -> InitialValueBindingCheckResult:
    """Anchor a serialized chart polynomial to explicit finite IVP data.

    Passing this check proves equality/closeness of the serialized polynomial
    state at one parameter.  Exact-trajectory enclosure still requires the
    separate a-posteriori tube certificate specified in the soundness theorem.
    """

    chart_by_id = {str(chart.chart_id): chart for chart in charts}
    chart = chart_by_id.get(str(binding.chart_id))
    positions = np.asarray(binding.positions, dtype=float)
    velocities = np.asarray(binding.velocities, dtype=float)
    masses = np.asarray(binding.masses, dtype=float)
    parameter = float(binding.chart_parameter)
    initial_time = float(binding.initial_time)
    time_tolerance = float(binding.time_tolerance)
    position_tolerance = float(binding.position_tolerance)
    velocity_tolerance = float(binding.velocity_tolerance)

    identity_present = bool(binding.binding_id and binding.chart_id)
    chart_present = chart is not None
    finite_problem = bool(
        masses.shape == (3,)
        and positions.ndim == 2
        and positions.shape[0] == 3
        and positions.shape == velocities.shape
        and positions.shape[1] in (2, 3)
        and np.all(np.isfinite(masses))
        and np.all(masses > 0.0)
        and np.all(np.isfinite(positions))
        and np.all(np.isfinite(velocities))
        and np.isfinite(parameter)
        and np.isfinite(initial_time)
    )
    finite_tolerances = bool(
        np.isfinite(time_tolerance)
        and time_tolerance >= 0.0
        and np.isfinite(position_tolerance)
        and position_tolerance >= 0.0
        and np.isfinite(velocity_tolerance)
        and velocity_tolerance >= 0.0
    )
    chart_masses_match = bool(
        chart_present
        and tuple(float(value) for value in binding.masses)
        == tuple(float(value) for value in chart.masses)
    )
    parameter_inside = bool(
        chart_present and _parameter_in_interval(chart.parameter_interval, parameter)
    )

    time_gap = np.inf
    max_position_gap = np.inf
    max_velocity_gap = np.inf
    state_shape_matches = False
    binding_matches = False
    if (
        chart_present
        and finite_problem
        and finite_tolerances
        and chart_masses_match
        and parameter_inside
    ):
        try:
            parameter_q = Fraction.from_float(parameter)
            chart_time_q = _exact_rational_chart_physical_time_at_parameter(
                chart, parameter_q
            )
            chart_positions_q, chart_velocities_q = (
                _exact_rational_chart_projected_state_at_parameter(
                    chart, parameter_q
                )
            )
            state_shape_matches = bool(
                chart_positions_q.shape == positions.shape
                and chart_velocities_q.shape == velocities.shape
            )
            if state_shape_matches:
                problem_positions_q = _fraction_array_from_floats(positions)
                problem_velocities_q = _fraction_array_from_floats(velocities)
                time_gap_q = abs(
                    chart_time_q - Fraction.from_float(initial_time)
                )
                position_gap_q = _fraction_array_max_abs_difference(
                    chart_positions_q, problem_positions_q
                )
                velocity_gap_q = _fraction_array_max_abs_difference(
                    chart_velocities_q, problem_velocities_q
                )
                time_gap = _fraction_upper_float(time_gap_q)
                max_position_gap = _fraction_upper_float(position_gap_q)
                max_velocity_gap = _fraction_upper_float(velocity_gap_q)
                binding_matches = bool(
                    time_gap_q <= Fraction.from_float(time_tolerance)
                    and position_gap_q <= Fraction.from_float(position_tolerance)
                    and velocity_gap_q <= Fraction.from_float(velocity_tolerance)
                )
        except (FloatingPointError, ValueError):
            pass

    obligations = (
        CertificateCheckObligation(
            "initial_value_binding_identity_present",
            identity_present,
            f"binding_id={binding.binding_id!r}; chart_id={binding.chart_id!r}",
        ),
        CertificateCheckObligation(
            "initial_value_binding_chart_present",
            chart_present,
            f"known_chart_ids={tuple(chart_by_id)!r}",
        ),
        CertificateCheckObligation(
            "initial_value_problem_finite_positive_mass_state",
            finite_problem,
            f"masses={binding.masses!r}; position_shape={positions.shape}",
        ),
        CertificateCheckObligation(
            "initial_value_binding_tolerances_finite",
            finite_tolerances,
            (
                f"time={time_tolerance}; position={position_tolerance}; "
                f"velocity={velocity_tolerance}"
            ),
        ),
        CertificateCheckObligation(
            "initial_value_binding_chart_masses_match",
            chart_masses_match,
            f"problem_masses={binding.masses!r}",
        ),
        CertificateCheckObligation(
            "initial_value_binding_parameter_inside_chart",
            parameter_inside,
            f"parameter={parameter}",
        ),
        CertificateCheckObligation(
            "initial_value_binding_state_shape_matches",
            state_shape_matches,
            f"problem_shape={positions.shape}",
        ),
        CertificateCheckObligation(
            "initial_value_binding_polynomial_state_matches",
            binding_matches,
            (
                f"time_gap={time_gap}; position_gap={max_position_gap}; "
                f"velocity_gap={max_velocity_gap}"
            ),
        ),
    )
    return InitialValueBindingCheckResult(
        binding_id=str(binding.binding_id),
        chart_id=str(binding.chart_id),
        checker_id="independent_initial_value_binding_checker_v1",
        obligations=obligations,
        max_position_gap=float(max_position_gap),
        max_velocity_gap=float(max_velocity_gap),
        time_gap=float(time_gap),
    )


def check_ordinary_aposteriori_tube(
    certificate: OrdinaryAposterioriTubeCertificate,
    chart: OrdinaryTaylorChartCertificate,
) -> OrdinaryAposterioriTubeCheckResult:
    """Check a Gronwall tube enclosing an exact ordinary Newtonian solution.

    The theorem is the standard defect estimate for a Lipschitz ODE, with a
    self-consistency condition keeping the enclosure inside the collision-free
    tube on which the recomputed Lipschitz bound applies.
    """

    q = _coefficient_array(chart.position_coefficients)
    v = _coefficient_array(chart.velocity_coefficients)
    masses = np.asarray(chart.masses, dtype=float)
    interval = tuple(float(value) for value in chart.parameter_interval)
    anchor_parameter = float(certificate.anchor_parameter)
    initial_error = float(certificate.initial_error_bound)
    radius = float(certificate.tube_radius)
    defect_cap = float(certificate.max_defect_bound)
    lipschitz_cap = float(certificate.max_lipschitz_bound)
    identity_matches = bool(
        certificate.tube_id
        and certificate.chart_id
        and certificate.chart_id == chart.chart_id
        and chart.chart_type == "ordinary_taylor"
    )
    finite_inputs = bool(
        np.isfinite(anchor_parameter)
        and _finite_nonempty_interval(interval)
        and interval[0] <= anchor_parameter <= interval[1]
        and np.isfinite(initial_error)
        and initial_error >= 0.0
        and np.isfinite(radius)
        and radius > 0.0
        and np.isfinite(defect_cap)
        and defect_cap >= 0.0
        and np.isfinite(lipschitz_cap)
        and lipschitz_cap >= 0.0
    )

    nominal_floor = 0.0
    tube_floor = 0.0
    defect = np.inf
    lipschitz = np.inf
    gronwall = np.inf
    defect_within_cap = False
    lipschitz_within_cap = False
    tube_collision_free = False
    self_consistent = False
    if identity_matches and finite_inputs:
        try:
            variable = FloatInterval(interval[0], interval[1])
            position_intervals = interval_array_series_eval(q, variable)
            nominal_floor = _interval_minimum_pair_distance(position_intervals)
            dimension = int(q.shape[2])
            tube_floor = _ordinary_tube_pair_floor_lower(
                nominal_floor, dimension, radius
            )
            tube_collision_free = tube_floor > 0.0
            defect = _max_interval_ordinary_newton_residual(
                q,
                v,
                masses,
                interval,
            )
            defect_within_cap = defect <= defect_cap
            if tube_collision_free:
                acceleration_lipschitz = max(
                    _newton_acceleration_lipschitz_upper(
                        masses,
                        body_index=i,
                        dimension=dimension,
                        pair_distance_floor=tube_floor,
                    )
                    for i in range(3)
                )
                lipschitz = max(1.0, acceleration_lipschitz)
                lipschitz_within_cap = lipschitz <= lipschitz_cap
                anchor_parameter_q = Fraction.from_float(anchor_parameter)
                horizon_q = max(
                    abs(Fraction.from_float(interval[0]) - anchor_parameter_q),
                    abs(Fraction.from_float(interval[1]) - anchor_parameter_q),
                )
                lipschitz_q = Fraction.from_float(lipschitz)
                exponent = _fraction_upper_float(lipschitz_q * horizon_q)
                exponential = _exp_upper(exponent)
                exponential_q = Fraction.from_float(exponential)
                gronwall_q = (
                    exponential_q * Fraction.from_float(initial_error)
                    + Fraction.from_float(defect)
                    * (exponential_q - Fraction(1))
                    / lipschitz_q
                )
                gronwall = _fraction_upper_float(gronwall_q)
                self_consistent = gronwall < radius
        except (
            DecimalException,
            FloatingPointError,
            OverflowError,
            ValueError,
            ZeroDivisionError,
        ):
            pass

    obligations = (
        CertificateCheckObligation(
            "ordinary_tube_identity_matches_chart",
            identity_matches,
            f"tube_id={certificate.tube_id!r}; chart_id={certificate.chart_id!r}",
        ),
        CertificateCheckObligation(
            "ordinary_tube_inputs_finite",
            finite_inputs,
            (
                f"initial_error={initial_error}; radius={radius}; "
                f"defect_cap={defect_cap}; lipschitz_cap={lipschitz_cap}"
            ),
        ),
        CertificateCheckObligation(
            "ordinary_tube_polynomial_defect_within_cap",
            defect_within_cap,
            f"recomputed_defect={defect}; cap={defect_cap}",
        ),
        CertificateCheckObligation(
            "ordinary_tube_collision_free",
            tube_collision_free,
            f"nominal_floor={nominal_floor}; tube_floor={tube_floor}",
        ),
        CertificateCheckObligation(
            "ordinary_tube_lipschitz_within_cap",
            lipschitz_within_cap,
            f"recomputed_lipschitz={lipschitz}; cap={lipschitz_cap}",
        ),
        CertificateCheckObligation(
            "ordinary_tube_gronwall_self_consistent",
            self_consistent,
            f"gronwall_error={gronwall}; radius={radius}",
        ),
    )
    return OrdinaryAposterioriTubeCheckResult(
        tube_id=str(certificate.tube_id),
        chart_id=str(certificate.chart_id),
        checker_id="independent_ordinary_aposteriori_tube_checker_v1",
        obligations=obligations,
        defect_bound=float(defect),
        lipschitz_bound=float(lipschitz),
        gronwall_error_bound=float(gronwall),
        nominal_pair_distance_floor=float(nominal_floor),
        tube_pair_distance_floor=float(tube_floor),
    )


def check_weighted_ordinary_aposteriori_tube(
    certificate: WeightedOrdinaryAposterioriTubeCertificate,
    chart: OrdinaryTaylorChartCertificate,
) -> WeightedOrdinaryAposterioriTubeCheckResult:
    """Check a Newton tube in a position/velocity weighted infinity norm."""

    q = _coefficient_array(chart.position_coefficients)
    v = _coefficient_array(chart.velocity_coefficients)
    masses = np.asarray(chart.masses, dtype=float)
    interval = tuple(float(value) for value in chart.parameter_interval)
    anchor = float(certificate.anchor_parameter)
    eps_q = float(certificate.initial_position_error_bound)
    eps_v = float(certificate.initial_velocity_error_bound)
    radius_q = float(certificate.position_radius)
    radius_v = float(certificate.velocity_radius)
    defect_cap = float(certificate.max_scaled_defect_bound)
    lipschitz_cap = float(certificate.max_scaled_lipschitz_bound)
    identity_matches = bool(
        certificate.tube_id
        and certificate.chart_id == chart.chart_id
        and chart.chart_type == "ordinary_taylor"
    )
    finite_inputs = bool(
        np.isfinite(anchor)
        and _finite_nonempty_interval(interval)
        and interval[0] <= anchor <= interval[1]
        and all(
            np.isfinite(value) and value >= 0.0
            for value in (eps_q, eps_v, defect_cap, lipschitz_cap)
        )
        and np.isfinite(radius_q)
        and radius_q > 0.0
        and np.isfinite(radius_v)
        and radius_v > 0.0
    )
    position_defect = velocity_defect = scaled_defect = np.inf
    acceleration_lipschitz = scaled_lipschitz = np.inf
    normalized_error = position_error = velocity_error = np.inf
    nominal_floor = tube_floor = 0.0
    collision_free = defect_within_cap = lipschitz_within_cap = False
    self_consistent = False
    if identity_matches and finite_inputs:
        try:
            variable = FloatInterval(*interval)
            positions = interval_array_series_eval(q, variable)
            nominal_floor = _interval_minimum_pair_distance(positions)
            dimension = int(q.shape[2])
            tube_floor = _ordinary_tube_pair_floor_lower(
                nominal_floor, dimension, radius_q
            )
            collision_free = tube_floor > 0.0
            position_defect, velocity_defect = (
                _interval_ordinary_newton_residual_blocks(q, v, masses, interval)
            )
            scaled_defect = max(
                _positive_ratio_upper(position_defect, radius_q),
                _positive_ratio_upper(velocity_defect, radius_v),
            )
            defect_within_cap = scaled_defect <= defect_cap
            if collision_free:
                acceleration_lipschitz = max(
                    _newton_acceleration_lipschitz_upper(
                        masses,
                        body_index=index,
                        dimension=dimension,
                        pair_distance_floor=tube_floor,
                    )
                    for index in range(3)
                )
                scaled_lipschitz = max(
                    _positive_ratio_upper(radius_v, radius_q),
                    _positive_ratio_upper(
                        _positive_product_upper(acceleration_lipschitz, radius_q),
                        radius_v,
                    ),
                )
                lipschitz_within_cap = scaled_lipschitz <= lipschitz_cap
                eta0 = max(
                    _positive_ratio_upper(eps_q, radius_q),
                    _positive_ratio_upper(eps_v, radius_v),
                )
                h = max(abs(interval[0] - anchor), abs(interval[1] - anchor))
                exponential = _exp_upper(
                    _positive_product_upper(scaled_lipschitz, h)
                )
                normalized_error = _fraction_upper_float(
                    Fraction.from_float(exponential)
                    * Fraction.from_float(eta0)
                    + Fraction.from_float(scaled_defect)
                    * (Fraction.from_float(exponential) - 1)
                    / Fraction.from_float(scaled_lipschitz)
                )
                position_error = _positive_product_upper(radius_q, normalized_error)
                velocity_error = _positive_product_upper(radius_v, normalized_error)
                self_consistent = normalized_error < 1.0
        except (
            DecimalException,
            FloatingPointError,
            OverflowError,
            ValueError,
            ZeroDivisionError,
        ):
            pass
    obligations = (
        CertificateCheckObligation(
            "weighted_ordinary_tube_identity_matches_chart",
            identity_matches,
            f"tube={certificate.tube_id!r}; chart={chart.chart_id!r}",
        ),
        CertificateCheckObligation(
            "weighted_ordinary_tube_inputs_finite",
            finite_inputs,
            f"position_radius={radius_q}; velocity_radius={radius_v}",
        ),
        CertificateCheckObligation(
            "weighted_ordinary_tube_scaled_defect_within_cap",
            defect_within_cap,
            f"scaled_defect={scaled_defect}; cap={defect_cap}",
        ),
        CertificateCheckObligation(
            "weighted_ordinary_tube_collision_free",
            collision_free,
            f"nominal_floor={nominal_floor}; tube_floor={tube_floor}",
        ),
        CertificateCheckObligation(
            "weighted_ordinary_tube_scaled_lipschitz_within_cap",
            lipschitz_within_cap,
            f"scaled_lipschitz={scaled_lipschitz}; cap={lipschitz_cap}",
        ),
        CertificateCheckObligation(
            "weighted_ordinary_tube_gronwall_self_consistent",
            self_consistent,
            f"normalized_error={normalized_error}",
        ),
    )
    return WeightedOrdinaryAposterioriTubeCheckResult(
        tube_id=str(certificate.tube_id),
        chart_id=str(certificate.chart_id),
        checker_id="weighted_ordinary_aposteriori_tube_checker_v1",
        obligations=obligations,
        position_defect_bound=float(position_defect),
        velocity_defect_bound=float(velocity_defect),
        scaled_defect_bound=float(scaled_defect),
        acceleration_lipschitz_bound=float(acceleration_lipschitz),
        scaled_lipschitz_bound=float(scaled_lipschitz),
        normalized_gronwall_error_bound=float(normalized_error),
        proven_position_error_bound=float(position_error),
        proven_velocity_error_bound=float(velocity_error),
        tube_pair_distance_floor=float(tube_floor),
    )


def check_planar_lc_aposteriori_tube(
    certificate: PlanarLCAposterioriTubeCertificate,
    chart: PlanarLeviCivitaBinaryChartCertificate,
) -> PlanarLCAposterioriTubeCheckResult:
    """Enclose an exact lifted LC solution, including through ``z=0``."""

    solution = _regularized_binary_solution_from_certificate(chart)
    interval = tuple(float(value) for value in chart.parameter_interval)
    anchor = float(certificate.anchor_parameter)
    initial_error = float(certificate.initial_error_bound)
    radius = float(certificate.tube_radius)
    defect_cap = float(certificate.max_defect_bound)
    lipschitz_cap = float(certificate.max_lipschitz_bound)
    identity_matches = bool(
        certificate.tube_id
        and certificate.chart_id == chart.chart_id
        and chart.chart_type == "planar_levi_civita_binary"
    )
    finite_inputs = bool(
        _finite_nonempty_interval(interval)
        and np.isfinite(anchor)
        and interval[0] <= anchor <= interval[1]
        and np.isfinite(initial_error)
        and initial_error >= 0.0
        and np.isfinite(radius)
        and radius > 0.0
        and np.isfinite(defect_cap)
        and defect_cap >= 0.0
        and np.isfinite(lipschitz_cap)
        and lipschitz_cap >= 0.0
    )
    arrays = (
        solution.z,
        solution.z_velocity,
        solution.pair_energy,
        solution.binary_center,
        solution.binary_center_velocity,
        solution.third_offset,
        solution.third_offset_velocity,
        solution.physical_time,
    )
    masses = np.asarray(solution.masses, dtype=float)
    lifted_serialization_admissible = bool(
        solution.z.ndim == 2
        and solution.z.shape[1:] == (2,)
        and solution.z.shape[0] >= 2
        and solution.z_velocity.shape == solution.z.shape
        and solution.pair_energy.shape == (solution.z.shape[0],)
        and solution.binary_center.shape == solution.z.shape
        and solution.binary_center_velocity.shape == solution.z.shape
        and solution.third_offset.shape == solution.z.shape
        and solution.third_offset_velocity.shape == solution.z.shape
        and solution.physical_time.shape == (solution.z.shape[0],)
        and all(np.all(np.isfinite(array)) for array in arrays)
        and masses.shape == (3,)
        and np.all(np.isfinite(masses))
        and np.all(masses > 0.0)
        and len(solution.pair) == 2
        and solution.pair[0] != solution.pair[1]
        and set(solution.pair).issubset({0, 1, 2})
    )
    defect = np.inf
    lipschitz = np.inf
    third_floor = 0.0
    gronwall = np.inf
    defect_within_cap = False
    lipschitz_within_cap = False
    separated_third_body = False
    self_consistent = False
    constraint_residual = np.inf
    constraint_anchor_certified = False
    if identity_matches and finite_inputs and lifted_serialization_admissible:
        try:
            anchor_q = Fraction.from_float(anchor)
            z_anchor = _evaluate_fraction_coefficients(
                chart.z_coefficients, anchor_q
            )
            w_anchor = _evaluate_fraction_coefficients(
                chart.z_velocity_coefficients, anchor_q
            )
            h_anchor = _evaluate_fraction_coefficients(
                chart.pair_energy_coefficients, anchor_q
            )
            pair_mass_q = sum(
                Fraction.from_float(float(chart.masses[index]))
                for index in chart.pair
            )
            rho_anchor = sum(value * value for value in z_anchor)
            constraint_q = (
                2 * sum(value * value for value in w_anchor)
                - pair_mass_q
                - rho_anchor * h_anchor
            )
            constraint_residual = _fraction_upper_float(abs(constraint_q))
            constraint_anchor_certified = constraint_q == 0
        except (OverflowError, ValueError, ZeroDivisionError):
            pass
    if identity_matches and finite_inputs and lifted_serialization_admissible:
        try:
            defect, lipschitz, third_floor = (
                _planar_lc_direct_defect_and_lipschitz(
                    solution,
                    interval,
                    tube_radius=radius,
                )
            )
            defect_within_cap = defect <= defect_cap
            lipschitz_within_cap = lipschitz <= lipschitz_cap
            separated_third_body = third_floor > 0.0
            if separated_third_body and np.isfinite(lipschitz):
                anchor_q = Fraction.from_float(anchor)
                horizon_q = max(
                    abs(Fraction.from_float(interval[0]) - anchor_q),
                    abs(Fraction.from_float(interval[1]) - anchor_q),
                )
                gronwall_q = _gronwall_error_upper_fraction(
                    initial_error,
                    defect,
                    lipschitz,
                    horizon_q,
                )
                gronwall = _fraction_upper_float(gronwall_q)
                self_consistent = gronwall_q < Fraction.from_float(radius)
        except (
            DecimalException,
            FloatingPointError,
            OverflowError,
            ValueError,
            ZeroDivisionError,
        ):
            pass

    obligations = (
        CertificateCheckObligation(
            "planar_lc_tube_identity_matches_chart",
            identity_matches,
            f"tube={certificate.tube_id!r}; chart={certificate.chart_id!r}",
        ),
        CertificateCheckObligation(
            "planar_lc_tube_lifted_serialization_admissible",
            lifted_serialization_admissible,
            (
                f"chart={chart.chart_id!r}; projected physical-velocity "
                "checking is deliberately separate on punctured rho-positive slabs"
            ),
        ),
        CertificateCheckObligation(
            "planar_lc_tube_inputs_finite",
            finite_inputs,
            f"anchor={anchor}; initial_error={initial_error}; radius={radius}",
        ),
        CertificateCheckObligation(
            "planar_lc_tube_direct_lifted_defect_within_cap",
            defect_within_cap,
            f"defect={defect}; cap={defect_cap}",
        ),
        CertificateCheckObligation(
            "planar_lc_tube_separated_third_body",
            separated_third_body,
            f"third_body_distance_floor={third_floor}",
        ),
        CertificateCheckObligation(
            "planar_lc_tube_interval_jacobian_within_cap",
            lipschitz_within_cap,
            f"lipschitz={lipschitz}; cap={lipschitz_cap}",
        ),
        CertificateCheckObligation(
            "planar_lc_tube_pair_energy_constraint_when_required",
            bool(
                not certificate.require_pair_energy_constraint
                or constraint_anchor_certified
            ),
            (
                f"required={certificate.require_pair_energy_constraint}; "
                f"exact_anchor_residual={constraint_residual}"
            ),
        ),
        CertificateCheckObligation(
            "planar_lc_tube_gronwall_self_consistent",
            self_consistent,
            f"gronwall_error={gronwall}; radius={radius}",
        ),
    )
    return PlanarLCAposterioriTubeCheckResult(
        tube_id=str(certificate.tube_id),
        chart_id=str(certificate.chart_id),
        checker_id="independent_planar_lc_aposteriori_tube_checker_v1",
        obligations=obligations,
        defect_bound=float(defect),
        lipschitz_bound=float(lipschitz),
        gronwall_error_bound=float(gronwall),
        third_body_distance_floor=float(third_floor),
        anchor_pair_energy_constraint_residual=float(constraint_residual),
        pair_energy_constraint_anchor_certified=constraint_anchor_certified,
        anchor_is_polynomial_center=bool(initial_error == 0.0),
    )


def check_planar_lc_exact_overlap_anchor(
    certificate: PlanarLCExactOverlapAnchorCertificate,
    source_tube: PlanarLCAposterioriTubeCertificate,
    source_chart: PlanarLeviCivitaBinaryChartCertificate,
    target_tube: PlanarLCAposterioriTubeCertificate,
    target_chart: PlanarLeviCivitaBinaryChartCertificate,
) -> PlanarLCExactOverlapAnchorCheckResult:
    """Derive the exact LC deck bit relating two zero-error tube anchors.

    The two tube checks are recomputed here.  No supplied parity is accepted:
    the checker compares all fourteen exact binary-rational anchor components,
    including physical time, and derives the unique identity or antipodal bit.
    """

    if type(certificate) is not PlanarLCExactOverlapAnchorCertificate:
        raise TypeError("certificate must be an exact LC overlap-anchor certificate")
    if type(source_tube) is not PlanarLCAposterioriTubeCertificate:
        raise TypeError("source_tube must be an exact planar LC tube certificate")
    if type(target_tube) is not PlanarLCAposterioriTubeCertificate:
        raise TypeError("target_tube must be an exact planar LC tube certificate")
    if type(source_chart) is not PlanarLeviCivitaBinaryChartCertificate:
        raise TypeError("source_chart must be an exact planar LC chart certificate")
    if type(target_chart) is not PlanarLeviCivitaBinaryChartCertificate:
        raise TypeError("target_chart must be an exact planar LC chart certificate")

    source_result = _check_planar_lc_tube_or_reject_malformed(
        source_tube, source_chart
    )
    target_result = _check_planar_lc_tube_or_reject_malformed(
        target_tube, target_chart
    )

    identifiers_match = bool(
        type(certificate.overlap_id) is str
        and bool(certificate.overlap_id)
        and type(certificate.source_chart_id) is str
        and type(certificate.source_tube_id) is str
        and type(certificate.target_chart_id) is str
        and type(certificate.target_tube_id) is str
        and type(source_chart.chart_id) is str
        and type(source_tube.chart_id) is str
        and type(source_tube.tube_id) is str
        and type(target_chart.chart_id) is str
        and type(target_tube.chart_id) is str
        and type(target_tube.tube_id) is str
        and certificate.source_chart_id == source_chart.chart_id == source_tube.chart_id
        and certificate.source_tube_id == source_tube.tube_id
        and certificate.target_chart_id == target_chart.chart_id == target_tube.chart_id
        and certificate.target_tube_id == target_tube.tube_id
    )
    distinct_charts_and_tubes = bool(
        identifiers_match
        and source_chart.chart_id != target_chart.chart_id
        and source_tube.tube_id != target_tube.tube_id
    )

    def strict_finite_float(value: object) -> bool:
        return type(value) is float and np.isfinite(value)

    source_interval = source_chart.parameter_interval
    target_interval = target_chart.parameter_interval
    interval_schema = bool(
        type(source_interval) is tuple
        and type(target_interval) is tuple
        and len(source_interval) == len(target_interval) == 2
        and all(strict_finite_float(value) for value in source_interval + target_interval)
        and source_interval[0] <= source_interval[1]
        and target_interval[0] <= target_interval[1]
    )
    anchor_parameters_match = bool(
        interval_schema
        and strict_finite_float(certificate.source_anchor_parameter)
        and strict_finite_float(certificate.target_anchor_parameter)
        and strict_finite_float(source_tube.anchor_parameter)
        and strict_finite_float(target_tube.anchor_parameter)
        and certificate.source_anchor_parameter == source_tube.anchor_parameter
        and certificate.target_anchor_parameter == target_tube.anchor_parameter
        and source_interval[0]
        <= certificate.source_anchor_parameter
        <= source_interval[1]
        and target_interval[0]
        <= certificate.target_anchor_parameter
        <= target_interval[1]
    )
    zero_error_centers = bool(
        strict_finite_float(source_tube.initial_error_bound)
        and strict_finite_float(target_tube.initial_error_bound)
        and source_tube.initial_error_bound == 0.0
        and target_tube.initial_error_bound == 0.0
        and source_result.anchor_is_polynomial_center
        and target_result.anchor_is_polynomial_center
    )
    masses_and_ordered_pair_match = bool(
        type(source_chart.masses) is tuple
        and type(target_chart.masses) is tuple
        and len(source_chart.masses) == len(target_chart.masses) == 3
        and all(
            type(value) is float and np.isfinite(value) and value > 0.0
            for value in source_chart.masses + target_chart.masses
        )
        and source_chart.masses == target_chart.masses
        and type(source_chart.pair) is tuple
        and type(target_chart.pair) is tuple
        and len(source_chart.pair) == len(target_chart.pair) == 2
        and all(type(index) is int for index in source_chart.pair + target_chart.pair)
        and source_chart.pair == target_chart.pair
        and source_chart.pair[0] != source_chart.pair[1]
        and set(source_chart.pair).issubset({0, 1, 2})
    )

    common_interval: tuple[Fraction, Fraction] = (Fraction(0), Fraction(0))
    common_interval_nondegenerate = False
    source_anchor: tuple[Fraction, ...] = ()
    target_anchor: tuple[Fraction, ...] = ()
    exact_anchor_schema = False
    fixed_components_equal = False
    relation_parity: int | None = None
    source_constraint_exact = False
    target_constraint_exact = False
    mass_ratio_arithmetic_exact = False
    if anchor_parameters_match:
        try:
            source_parameter_q = Fraction.from_float(
                certificate.source_anchor_parameter
            )
            target_parameter_q = Fraction.from_float(
                certificate.target_anchor_parameter
            )
            source_relative = (
                Fraction.from_float(source_interval[0]) - source_parameter_q,
                Fraction.from_float(source_interval[1]) - source_parameter_q,
            )
            target_relative = (
                Fraction.from_float(target_interval[0]) - target_parameter_q,
                Fraction.from_float(target_interval[1]) - target_parameter_q,
            )
            common_interval = (
                max(source_relative[0], target_relative[0]),
                min(source_relative[1], target_relative[1]),
            )
            common_interval_nondegenerate = bool(
                common_interval[0] < common_interval[1]
                and common_interval[0] <= 0 <= common_interval[1]
            )
            source_anchor = (
                _exact_rational_planar_lc_lifted_state_at_parameter(
                    source_chart, source_parameter_q
                )
                + (
                    _exact_rational_chart_physical_time_at_parameter(
                        source_chart, source_parameter_q
                    ),
                )
            )
            target_anchor = (
                _exact_rational_planar_lc_lifted_state_at_parameter(
                    target_chart, target_parameter_q
                )
                + (
                    _exact_rational_chart_physical_time_at_parameter(
                        target_chart, target_parameter_q
                    ),
                )
            )
            exact_anchor_schema = bool(
                len(source_anchor) == len(target_anchor) == 14
                and all(
                    type(value) is Fraction
                    for value in source_anchor + target_anchor
                )
            )
        except (IndexError, OverflowError, TypeError, ValueError, ZeroDivisionError):
            pass

    if exact_anchor_schema:
        fixed_components_equal = source_anchor[4:] == target_anchor[4:]
        same = source_anchor[:4] == target_anchor[:4]
        antipodal = target_anchor[:4] == tuple(
            -value for value in source_anchor[:4]
        )
        if fixed_components_equal and same != antipodal:
            relation_parity = 0 if same else 1

    if exact_anchor_schema and masses_and_ordered_pair_match:
        pair_mass_q = sum(
            Fraction.from_float(source_chart.masses[index])
            for index in source_chart.pair
        )

        def constraint_exact(anchor: tuple[Fraction, ...]) -> bool:
            z = anchor[0:2]
            w = anchor[2:4]
            return bool(
                2 * sum(value * value for value in w)
                - pair_mass_q
                - sum(value * value for value in z) * anchor[4]
                == 0
            )

        source_constraint_exact = constraint_exact(source_anchor)
        target_constraint_exact = constraint_exact(target_anchor)
        mass_ratio_arithmetic_exact = bool(
            _planar_lc_mass_ratio_arithmetic_exact(
                source_chart.masses, source_chart.pair
            )
            and _planar_lc_mass_ratio_arithmetic_exact(
                target_chart.masses, target_chart.pair
            )
        )

    lifted_prerequisites = bool(
        identifiers_match
        and distinct_charts_and_tubes
        and anchor_parameters_match
        and zero_error_centers
        and masses_and_ordered_pair_match
        and source_result.certified
        and target_result.certified
        and common_interval_nondegenerate
        and exact_anchor_schema
        and fixed_components_equal
        and relation_parity in (0, 1)
    )
    edge = None
    if lifted_prerequisites:
        edge = PlanarLCGaugeOverlapEdge(
            overlap_id=certificate.overlap_id,
            source_chart_id=source_chart.chart_id,
            target_chart_id=target_chart.chart_id,
            parity=relation_parity,
        )

    obligations = (
        CertificateCheckObligation(
            "planar_lc_exact_overlap_identifiers_match",
            identifiers_match,
            f"overlap={certificate.overlap_id!r}",
        ),
        CertificateCheckObligation(
            "planar_lc_exact_overlap_distinct_charts_and_tubes",
            distinct_charts_and_tubes,
            (
                f"source_chart={source_chart.chart_id!r}; "
                f"target_chart={target_chart.chart_id!r}"
            ),
        ),
        CertificateCheckObligation(
            "planar_lc_exact_overlap_anchor_parameters_match",
            anchor_parameters_match,
            (
                f"source={certificate.source_anchor_parameter!r}; "
                f"target={certificate.target_anchor_parameter!r}"
            ),
        ),
        CertificateCheckObligation(
            "planar_lc_exact_overlap_zero_error_centers",
            zero_error_centers,
            (
                f"source_error={source_tube.initial_error_bound!r}; "
                f"target_error={target_tube.initial_error_bound!r}"
            ),
        ),
        CertificateCheckObligation(
            "planar_lc_exact_overlap_masses_and_ordered_pair_match",
            masses_and_ordered_pair_match,
            f"source_pair={source_chart.pair!r}; target_pair={target_chart.pair!r}",
        ),
        CertificateCheckObligation(
            "planar_lc_exact_overlap_source_tube_certified",
            source_result.certified,
            f"tube={source_tube.tube_id!r}",
        ),
        CertificateCheckObligation(
            "planar_lc_exact_overlap_target_tube_certified",
            target_result.certified,
            f"tube={target_tube.tube_id!r}",
        ),
        CertificateCheckObligation(
            "planar_lc_exact_overlap_common_interval_nondegenerate",
            common_interval_nondegenerate,
            f"relative_interval={common_interval!r}",
        ),
        CertificateCheckObligation(
            "planar_lc_exact_overlap_exact_anchor_schema",
            exact_anchor_schema,
            (
                f"source_dimension={len(source_anchor)}; "
                f"target_dimension={len(target_anchor)}"
            ),
        ),
        CertificateCheckObligation(
            "planar_lc_exact_overlap_fixed_components_equal",
            fixed_components_equal,
            "the h,R,U,y,V,t anchor blocks agree exactly",
        ),
        CertificateCheckObligation(
            "planar_lc_exact_overlap_unique_deck_relation",
            relation_parity in (0, 1),
            f"derived_parity={relation_parity!r}",
        ),
    )
    return PlanarLCExactOverlapAnchorCheckResult(
        overlap_id=certificate.overlap_id if type(certificate.overlap_id) is str else "",
        checker_id="planar_lc_exact_overlap_anchor_checker_v1",
        source_chart_id=(
            source_chart.chart_id if type(source_chart.chart_id) is str else ""
        ),
        target_chart_id=(
            target_chart.chart_id if type(target_chart.chart_id) is str else ""
        ),
        raw_overlap_certificate=certificate,
        raw_source_tube=source_tube,
        raw_source_chart=source_chart,
        raw_target_tube=target_tube,
        raw_target_chart=target_chart,
        source_tube_result=source_result,
        target_tube_result=target_result,
        obligations=obligations,
        source_exact_anchor=source_anchor,
        target_exact_anchor=target_anchor,
        common_parameter_offset_interval=common_interval,
        derived_edge=edge,
        serialized_masses=(
            tuple(source_chart.masses)
            if type(source_chart.masses) is tuple
            else ()
        ),
        selected_pair=(
            tuple(source_chart.pair) if type(source_chart.pair) is tuple else ()
        ),
        source_pair_energy_constraint_exact=source_constraint_exact,
        target_pair_energy_constraint_exact=target_constraint_exact,
        mass_ratio_arithmetic_exact=mass_ratio_arithmetic_exact,
    )


def check_planar_lc_exact_gauge_atlas(
    certificate: PlanarLCExactGaugeAtlasCertificate,
    charts: tuple[PlanarLeviCivitaBinaryChartCertificate, ...],
    tubes: tuple[PlanarLCAposterioriTubeCertificate, ...],
    overlaps: tuple[PlanarLCExactOverlapAnchorCertificate, ...],
) -> PlanarLCExactGaugeAtlasCheckResult:
    """Recompute an exact LC overlap graph from raw vertex and edge evidence.

    No overlap result, gauge edge, or parity is accepted as an input.  Invalid
    raw bindings stop before graph construction, keeping this aggregate wholly
    separate from existing transition and ``proof_certified`` routes.
    """

    if type(certificate) is not PlanarLCExactGaugeAtlasCertificate:
        raise TypeError("certificate must be an exact LC gauge-atlas certificate")
    if not (
        type(charts) is tuple
        and all(type(chart) is PlanarLeviCivitaBinaryChartCertificate for chart in charts)
    ):
        raise TypeError("charts must be a tuple of exact planar LC chart certificates")
    if not (
        type(tubes) is tuple
        and all(type(tube) is PlanarLCAposterioriTubeCertificate for tube in tubes)
    ):
        raise TypeError("tubes must be a tuple of exact planar LC tube certificates")
    if not (
        type(overlaps) is tuple
        and all(
            type(overlap) is PlanarLCExactOverlapAnchorCertificate
            for overlap in overlaps
        )
    ):
        raise TypeError("overlaps must be a tuple of raw exact-overlap certificates")

    chart_ids = certificate.chart_ids
    tube_ids = certificate.tube_ids
    overlap_ids = certificate.overlap_ids

    def exact_nonempty_string_tuple(value: object, *, nonempty: bool) -> bool:
        return bool(
            type(value) is tuple
            and (bool(value) or not nonempty)
            and all(type(item) is str and bool(item) for item in value)
            and len(set(value)) == len(value)
        )

    manifest_schema = bool(
        type(certificate.atlas_id) is str
        and bool(certificate.atlas_id)
        and exact_nonempty_string_tuple(chart_ids, nonempty=True)
        and exact_nonempty_string_tuple(tube_ids, nonempty=True)
        and exact_nonempty_string_tuple(overlap_ids, nonempty=False)
        and len(chart_ids) == len(tube_ids)
    )
    raw_chart_ids = tuple(chart.chart_id for chart in charts)
    raw_tube_ids = tuple(tube.tube_id for tube in tubes)
    raw_overlap_ids = tuple(overlap.overlap_id for overlap in overlaps)
    raw_ids_schema = bool(
        charts
        and all(type(value) is str and bool(value) for value in raw_chart_ids)
        and len(set(raw_chart_ids)) == len(raw_chart_ids)
        and len(tubes) == len(charts)
        and all(type(value) is str and bool(value) for value in raw_tube_ids)
        and len(set(raw_tube_ids)) == len(raw_tube_ids)
        and all(type(value) is str and bool(value) for value in raw_overlap_ids)
        and len(set(raw_overlap_ids)) == len(raw_overlap_ids)
    )
    manifest_matches_raw = bool(
        manifest_schema
        and raw_ids_schema
        and chart_ids == raw_chart_ids
        and tube_ids == raw_tube_ids
        and overlap_ids == raw_overlap_ids
    )
    positional_vertex_binding = bool(
        manifest_matches_raw
        and len(charts) == len(tubes)
        and all(
            type(chart.chart_id) is str
            and type(tube.chart_id) is str
            and tube.chart_id == chart.chart_id
            for chart, tube in zip(charts, tubes)
        )
    )

    vertex_results: list[PlanarLCAposterioriTubeCheckResult] = []
    if len(charts) == len(tubes):
        for chart, tube in zip(charts, tubes):
            vertex_results.append(
                _check_planar_lc_tube_or_reject_malformed(tube, chart)
            )
    vertex_tubes_certified_zero_error = bool(
        positional_vertex_binding
        and len(vertex_results) == len(charts)
        and all(
            type(tube.initial_error_bound) is float
            and tube.initial_error_bound == 0.0
            and result.certified
            and result.anchor_is_polynomial_center
            and result.chart_id == chart.chart_id
            and result.tube_id == tube.tube_id
            for chart, tube, result in zip(charts, tubes, vertex_results)
        )
    )

    def exact_problem(chart: PlanarLeviCivitaBinaryChartCertificate) -> bool:
        return bool(
            type(chart.masses) is tuple
            and len(chart.masses) == 3
            and all(
                type(value) is float and np.isfinite(value) and value > 0.0
                for value in chart.masses
            )
            and type(chart.pair) is tuple
            and len(chart.pair) == 2
            and all(type(index) is int for index in chart.pair)
            and chart.pair[0] != chart.pair[1]
            and set(chart.pair).issubset({0, 1, 2})
        )

    vertex_problem_data = tuple(
        (tuple(chart.masses), tuple(chart.pair))
        if exact_problem(chart)
        else ((), ())
        for chart in charts
    )
    globally_same_problem = bool(
        charts
        and all(exact_problem(chart) for chart in charts)
        and all(problem == vertex_problem_data[0] for problem in vertex_problem_data)
    )

    chart_by_id = (
        {chart.chart_id: chart for chart in charts} if raw_ids_schema else {}
    )
    tube_by_chart_id = (
        {chart.chart_id: tube for chart, tube in zip(charts, tubes)}
        if positional_vertex_binding
        else {}
    )
    overlap_bindings_match_vertices = bool(
        manifest_matches_raw
        and all(
            type(overlap.source_chart_id) is str
            and type(overlap.target_chart_id) is str
            and type(overlap.source_tube_id) is str
            and type(overlap.target_tube_id) is str
            and overlap.source_chart_id in chart_by_id
            and overlap.target_chart_id in chart_by_id
            and overlap.source_chart_id != overlap.target_chart_id
            and overlap.source_tube_id
            == tube_by_chart_id[overlap.source_chart_id].tube_id
            and overlap.target_tube_id
            == tube_by_chart_id[overlap.target_chart_id].tube_id
            for overlap in overlaps
        )
    )

    overlap_results: list[PlanarLCExactOverlapAnchorCheckResult] = []
    if (
        vertex_tubes_certified_zero_error
        and globally_same_problem
        and overlap_bindings_match_vertices
    ):
        for overlap in overlaps:
            source_chart = chart_by_id[overlap.source_chart_id]
            target_chart = chart_by_id[overlap.target_chart_id]
            source_tube = tube_by_chart_id[overlap.source_chart_id]
            target_tube = tube_by_chart_id[overlap.target_chart_id]
            try:
                overlap_results.append(
                    check_planar_lc_exact_overlap_anchor(
                        overlap,
                        source_tube,
                        source_chart,
                        target_tube,
                        target_chart,
                    )
                )
            except (
                AttributeError,
                DecimalException,
                FloatingPointError,
                IndexError,
                KeyError,
                OverflowError,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ):
                break
    all_overlaps_recomputed_and_certified = bool(
        len(overlap_results) == len(overlaps)
        and all(
            result._snapshot_lifted_overlap_certified()
            for result in overlap_results
        )
    )
    anchor_parameter_translations: list[
        tuple[Fraction, Fraction, Fraction]
    ] = []
    for overlap in overlaps:
        if not (
            type(overlap.source_anchor_parameter) is float
            and type(overlap.target_anchor_parameter) is float
            and np.isfinite(overlap.source_anchor_parameter)
            and np.isfinite(overlap.target_anchor_parameter)
        ):
            break
        source_q = Fraction.from_float(overlap.source_anchor_parameter)
        target_q = Fraction.from_float(overlap.target_anchor_parameter)
        anchor_parameter_translations.append(
            (source_q, target_q, target_q - source_q)
        )
    anchor_translations_exact = bool(
        len(anchor_parameter_translations) == len(overlaps)
    )
    derived_edges = tuple(
        result.derived_edge
        for result in overlap_results
        if result._snapshot_lifted_overlap_certified()
        and type(result.derived_edge) is PlanarLCGaugeOverlapEdge
    )
    derived_edges_match_results = bool(
        all_overlaps_recomputed_and_certified
        and len(derived_edges) == len(overlap_results)
        and all(
            type(result.derived_edge) is PlanarLCGaugeOverlapEdge
            and edge == result.derived_edge
            for edge, result in zip(derived_edges, overlap_results)
        )
    )

    gauge_result: PlanarLCGaugeGluingCheckResult | None = None
    ready_for_graph = bool(
        manifest_matches_raw
        and positional_vertex_binding
        and vertex_tubes_certified_zero_error
        and globally_same_problem
        and overlap_bindings_match_vertices
        and all_overlaps_recomputed_and_certified
        and anchor_translations_exact
        and derived_edges_match_results
    )
    if ready_for_graph:
        gauge_result = check_planar_lc_gauge_gluing(
            PlanarLCGaugeGluingCertificate(
                certificate_id=f"{certificate.atlas_id}:derived-gauge-graph",
                chart_ids=tuple(chart_ids),
                overlaps=derived_edges,
                source="derived_from_raw_exact_lc_overlap_anchors",
            )
        )
    derived_graph_certified = bool(
        type(gauge_result) is PlanarLCGaugeGluingCheckResult
        and gauge_result.certified
    )

    obligations = (
        CertificateCheckObligation(
            "planar_lc_exact_gauge_atlas_manifest_schema",
            manifest_schema,
            f"atlas={certificate.atlas_id!r}",
        ),
        CertificateCheckObligation(
            "planar_lc_exact_gauge_atlas_raw_ids_unique_and_positional",
            bool(raw_ids_schema and manifest_matches_raw),
            (
                f"charts={raw_chart_ids!r}; tubes={raw_tube_ids!r}; "
                f"overlaps={raw_overlap_ids!r}"
            ),
        ),
        CertificateCheckObligation(
            "planar_lc_exact_gauge_atlas_vertex_bindings_match",
            positional_vertex_binding,
            f"vertex_count={len(charts)}",
        ),
        CertificateCheckObligation(
            "planar_lc_exact_gauge_atlas_vertex_tubes_zero_error_certified",
            vertex_tubes_certified_zero_error,
            f"checked_vertex_count={len(vertex_results)}",
        ),
        CertificateCheckObligation(
            "planar_lc_exact_gauge_atlas_global_problem_matches",
            globally_same_problem,
            f"problems={vertex_problem_data!r}",
        ),
        CertificateCheckObligation(
            "planar_lc_exact_gauge_atlas_overlap_bindings_match_vertices",
            overlap_bindings_match_vertices,
            f"raw_overlap_count={len(overlaps)}",
        ),
        CertificateCheckObligation(
            "planar_lc_exact_gauge_atlas_overlaps_recomputed_and_certified",
            all_overlaps_recomputed_and_certified,
            f"checked_overlap_count={len(overlap_results)}",
        ),
        CertificateCheckObligation(
            "planar_lc_exact_gauge_atlas_derived_edges_match_results",
            bool(anchor_translations_exact and derived_edges_match_results),
            (
                f"derived_edge_count={len(derived_edges)}; "
                f"anchor_translation_count={len(anchor_parameter_translations)}"
            ),
        ),
        CertificateCheckObligation(
            "planar_lc_exact_gauge_atlas_derived_graph_certified",
            derived_graph_certified,
            "the connected gauge graph was built only from recomputed overlap results",
        ),
    )
    return PlanarLCExactGaugeAtlasCheckResult(
        atlas_id=(certificate.atlas_id if type(certificate.atlas_id) is str else ""),
        checker_id="planar_lc_exact_gauge_atlas_checker_v1",
        raw_atlas_certificate=certificate,
        obligations=obligations,
        chart_ids=(tuple(chart_ids) if type(chart_ids) is tuple else ()),
        tube_ids=(tuple(tube_ids) if type(tube_ids) is tuple else ()),
        overlap_ids=(tuple(overlap_ids) if type(overlap_ids) is tuple else ()),
        checked_charts=tuple(charts),
        checked_tubes=tuple(tubes),
        checked_overlap_certificates=tuple(overlaps),
        vertex_tube_results=tuple(vertex_results),
        vertex_problem_data=vertex_problem_data,
        overlap_results=tuple(overlap_results),
        anchor_parameter_translations=tuple(anchor_parameter_translations),
        derived_edges=derived_edges,
        gauge_result=gauge_result,
    )


def check_planar_lc_exact_collision_anchor(
    certificate: PlanarLCExactCollisionAnchorCertificate,
    tube: PlanarLCAposterioriTubeCertificate,
    chart: PlanarLeviCivitaBinaryChartCertificate,
) -> PlanarLCExactCollisionAnchorCheckResult:
    """Certify an exact, isolated collision of the center-anchored LC IVP."""

    tube_result = check_planar_lc_aposteriori_tube(tube, chart)
    parameter = float(certificate.collision_parameter)
    identifiers_match = bool(
        certificate.collision_id
        and certificate.chart_id == chart.chart_id == tube.chart_id
        and certificate.tube_id == tube.tube_id
    )
    parameter_matches = bool(
        np.isfinite(parameter)
        and _parameter_in_interval(chart.parameter_interval, parameter)
        and parameter == float(tube.anchor_parameter)
    )
    zero_error_center_anchor = bool(
        np.isfinite(tube.initial_error_bound)
        and float(tube.initial_error_bound) == 0.0
        and tube_result.anchor_is_polynomial_center
    )
    exact_zero = False
    exact_constraint = False
    nonzero_velocity = False
    speed_squared = np.inf
    pair_mass_value = np.inf
    collision_time = np.inf
    mass_ratio_arithmetic_exact = _planar_lc_mass_ratio_arithmetic_exact(
        chart.masses, chart.pair
    )
    if identifiers_match and parameter_matches:
        try:
            parameter_q = Fraction.from_float(parameter)
            state = _exact_rational_planar_lc_lifted_state_at_parameter(
                chart, parameter_q
            )
            z = state[0:2]
            w = state[2:4]
            h = state[4]
            rho = z[0] * z[0] + z[1] * z[1]
            speed_q = w[0] * w[0] + w[1] * w[1]
            pair_mass_q = sum(
                Fraction.from_float(float(chart.masses[index]))
                for index in chart.pair
            )
            constraint_q = 2 * speed_q - pair_mass_q - rho * h
            exact_zero = z == (Fraction(0), Fraction(0))
            exact_constraint = constraint_q == 0
            nonzero_velocity = speed_q > 0
            speed_squared = _fraction_lower_float(speed_q)
            pair_mass_value = _fraction_lower_float(pair_mass_q)
            collision_time = float(
                _exact_rational_chart_physical_time_at_parameter(
                    chart, parameter_q
                )
            )
        except (IndexError, OverflowError, ValueError, ZeroDivisionError):
            pass

    obligations = (
        CertificateCheckObligation(
            "planar_lc_collision_identifiers_match",
            identifiers_match,
            f"collision={certificate.collision_id!r}; tube={tube.tube_id!r}",
        ),
        CertificateCheckObligation(
            "planar_lc_collision_parameter_is_tube_anchor",
            parameter_matches,
            f"collision_parameter={parameter}; tube_anchor={tube.anchor_parameter}",
        ),
        CertificateCheckObligation(
            "planar_lc_collision_tube_certified",
            tube_result.certified,
            f"tube={tube.tube_id!r}",
        ),
        CertificateCheckObligation(
            "planar_lc_collision_zero_error_center_anchor",
            zero_error_center_anchor,
            f"initial_error_bound={tube.initial_error_bound}",
        ),
        CertificateCheckObligation(
            "planar_lc_collision_exact_z_zero",
            exact_zero,
            f"collision_parameter={parameter}",
        ),
        CertificateCheckObligation(
            "planar_lc_collision_exact_pair_energy_constraint",
            bool(exact_constraint and tube_result.pair_energy_constraint_anchor_certified),
            f"pair_mass={pair_mass_value}; speed_squared={speed_squared}",
        ),
        CertificateCheckObligation(
            "planar_lc_collision_simple_zero",
            nonzero_velocity,
            f"speed_squared={speed_squared}",
        ),
        CertificateCheckObligation(
            "planar_lc_collision_third_body_separated",
            tube_result.third_body_distance_floor > 0.0,
            f"third_body_distance_floor={tube_result.third_body_distance_floor}",
        ),
        CertificateCheckObligation(
            "planar_lc_collision_exact_mass_ratio_arithmetic",
            mass_ratio_arithmetic_exact,
            "pair mass, alpha, beta, and third/pair mass ratio are exact binary rationals",
        ),
    )
    return PlanarLCExactCollisionAnchorCheckResult(
        collision_id=str(certificate.collision_id),
        checker_id="planar_lc_exact_collision_anchor_checker_v1",
        tube_result=tube_result,
        obligations=obligations,
        exact_speed_squared=float(speed_squared),
        pair_mass=float(pair_mass_value),
        collision_physical_time=float(collision_time),
        collision_parameter=float(parameter),
        mass_ratio_arithmetic_exact=mass_ratio_arithmetic_exact,
    )


def check_planar_lc_two_sided_collision_passage(
    certificate: PlanarLCTwoSidedCollisionPassageCertificate,
    source_chart: PlanarLeviCivitaBinaryChartCertificate,
    collision_result: PlanarLCExactCollisionAnchorCheckResult,
    left_target_chart: OrdinaryTaylorChartCertificate,
    right_target_chart: OrdinaryTaylorChartCertificate,
    left_target_tube: OrdinaryAposterioriTubeCertificate | WeightedOrdinaryAposterioriTubeCertificate,
    right_target_tube: OrdinaryAposterioriTubeCertificate | WeightedOrdinaryAposterioriTubeCertificate,
) -> PlanarLCTwoSidedCollisionPassageCheckResult:
    """Certify two classical branches of one exact LC collision solution."""

    if type(collision_result) is not PlanarLCExactCollisionAnchorCheckResult:
        raise TypeError("collision_result must be an exact collision-anchor result")
    def check_target_tube(tube, chart):
        if type(tube) is WeightedOrdinaryAposterioriTubeCertificate:
            return check_weighted_ordinary_aposteriori_tube(tube, chart)
        if type(tube) is OrdinaryAposterioriTubeCertificate:
            return check_ordinary_aposteriori_tube(tube, chart)
        raise TypeError("passage target tube must be ordinary or weighted ordinary")

    left_tube_result = check_target_tube(left_target_tube, left_target_chart)
    right_tube_result = check_target_tube(right_target_tube, right_target_chart)
    identifiers_match = bool(
        certificate.passage_id
        and certificate.source_chart_id == source_chart.chart_id
        == collision_result.tube_result.chart_id
        and certificate.collision_id == collision_result.collision_id
        and certificate.left_target_chart_id == left_target_chart.chart_id
        == left_target_tube.chart_id
        and certificate.right_target_chart_id == right_target_chart.chart_id
        == right_target_tube.chart_id
    )
    common_problem = bool(
        tuple(source_chart.masses) == tuple(left_target_chart.masses)
        == tuple(right_target_chart.masses)
        and left_target_chart.dimension == right_target_chart.dimension == 2
    )
    left_source_parameter = float(certificate.left_source_parameter)
    right_source_parameter = float(certificate.right_source_parameter)
    left_target_parameter = float(certificate.left_target_parameter)
    right_target_parameter = float(certificate.right_target_parameter)
    ordered_parameters = bool(
        _parameter_in_interval(source_chart.parameter_interval, left_source_parameter)
        and _parameter_in_interval(source_chart.parameter_interval, right_source_parameter)
        and left_source_parameter < collision_result.collision_parameter
        < right_source_parameter
        and _parameter_in_interval(
            left_target_chart.parameter_interval, left_target_parameter
        )
        and _parameter_in_interval(
            right_target_chart.parameter_interval, right_target_parameter
        )
        and float(left_target_tube.anchor_parameter) == left_target_parameter
        and float(right_target_tube.anchor_parameter) == right_target_parameter
    )

    def endpoint(
        source_parameter: float,
        target_parameter: float,
        target_chart: OrdinaryTaylorChartCertificate,
        target_tube: OrdinaryAposterioriTubeCertificate | WeightedOrdinaryAposterioriTubeCertificate,
        target_result: OrdinaryAposterioriTubeCheckResult | WeightedOrdinaryAposterioriTubeCheckResult,
    ) -> tuple[bool, float, float, tuple[float, float]]:
        if not (
            identifiers_match
            and collision_result.certified
            and common_problem
            and ordered_parameters
            and target_result.certified
        ):
            return False, 0.0, np.inf, (np.inf, -np.inf)
        try:
            source_solution = _regularized_binary_solution_from_certificate(
                source_chart
            )
            lifted_state = _planar_lc_state_intervals(
                source_solution,
                (source_parameter, source_parameter),
                inflate=float(collision_result.tube_result.gronwall_error_bound),
            )
            positions, velocities, physical_time, rho = (
                _project_interval_planar_lc_state(
                    lifted_state,
                    np.asarray(source_chart.masses, dtype=float),
                    source_chart.pair,
                )
            )
            rho_floor = float(rho.lower)
            target_parameter_q = Fraction.from_float(target_parameter)
            elapsed_anchor = (
                _exact_rational_chart_physical_time_at_parameter(
                    target_chart, target_parameter_q
                )
                == 0
            )
            target_q, target_v = _exact_rational_chart_projected_state_at_parameter(
                target_chart, target_parameter_q
            )
            if type(target_tube) is WeightedOrdinaryAposterioriTubeCertificate:
                position_allowance = Fraction.from_float(
                    float(target_tube.initial_position_error_bound)
                )
                velocity_allowance = Fraction.from_float(
                    float(target_tube.initial_velocity_error_bound)
                )
            else:
                position_allowance = velocity_allowance = Fraction.from_float(
                    float(target_tube.initial_error_bound)
                )
            gap = Fraction(0)
            contained = True
            for boxes, centers, allowance in (
                (positions, target_q, position_allowance),
                (velocities, target_v, velocity_allowance),
            ):
                for index in np.ndindex(boxes.shape):
                    box = boxes[index]
                    center = centers[index]
                    lower = Fraction.from_float(float(box.lower))
                    upper = Fraction.from_float(float(box.upper))
                    gap = max(gap, abs(lower - center), abs(upper - center))
                    contained = contained and (
                        center - allowance <= lower
                        and upper <= center + allowance
                    )
            return (
                bool(rho_floor > 0.0 and elapsed_anchor and contained),
                rho_floor,
                _fraction_upper_float(gap),
                (float(physical_time.lower), float(physical_time.upper)),
            )
        except (OverflowError, ValueError, ZeroDivisionError):
            return False, 0.0, np.inf, (np.inf, -np.inf)

    left_ok, left_rho, left_gap, left_time = endpoint(
        left_source_parameter,
        left_target_parameter,
        left_target_chart,
        left_target_tube,
        left_tube_result,
    )
    right_ok, right_rho, right_gap, right_time = endpoint(
        right_source_parameter,
        right_target_parameter,
        right_target_chart,
        right_target_tube,
        right_tube_result,
    )
    # The collision anchor gives w(s*) != 0, hence z is a nonzero analytic
    # function.  Its zeros are isolated and integral |z|^2 ds is strictly
    # positive on every nondegenerate interval, even when coarse endpoint-time
    # enclosures overlap the collision time.
    time_ordered = bool(
        collision_result.local_physical_time_strictly_increasing_certified
        and ordered_parameters
        and left_ok
        and right_ok
    )
    obligations = (
        CertificateCheckObligation(
            "planar_lc_passage_identifiers_match",
            identifiers_match,
            f"passage={certificate.passage_id!r}",
        ),
        CertificateCheckObligation(
            "planar_lc_passage_exact_collision_anchor_certified",
            collision_result.certified,
            f"collision={collision_result.collision_id!r}",
        ),
        CertificateCheckObligation(
            "planar_lc_passage_common_planar_problem",
            common_problem,
            "source and both targets share planar masses",
        ),
        CertificateCheckObligation(
            "planar_lc_passage_parameters_straddle_collision",
            ordered_parameters,
            (
                f"left={left_source_parameter}; collision="
                f"{collision_result.collision_parameter}; right={right_source_parameter}"
            ),
        ),
        CertificateCheckObligation(
            "planar_lc_passage_left_punctured_projection_contained",
            left_ok,
            f"rho_floor={left_rho}; gap={left_gap}",
        ),
        CertificateCheckObligation(
            "planar_lc_passage_right_punctured_projection_contained",
            right_ok,
            f"rho_floor={right_rho}; gap={right_gap}",
        ),
        CertificateCheckObligation(
            "planar_lc_passage_physical_times_straddle_collision",
            time_ordered,
            f"left={left_time}; collision={collision_result.collision_physical_time}; right={right_time}",
        ),
    )
    return PlanarLCTwoSidedCollisionPassageCheckResult(
        passage_id=str(certificate.passage_id),
        checker_id="planar_lc_two_sided_collision_passage_checker_v1",
        collision_result=collision_result,
        left_target_tube_result=left_tube_result,
        right_target_tube_result=right_tube_result,
        obligations=obligations,
        left_rho_lower_bound=float(left_rho),
        right_rho_lower_bound=float(right_rho),
        left_projection_gap=float(left_gap),
        right_projection_gap=float(right_gap),
        left_time_origin_interval=left_time,
        right_time_origin_interval=right_time,
    )


def check_ordinary_to_planar_lc_enclosure_transition(
    certificate: OrdinaryToPlanarLCEnclosureTransitionCertificate,
    source_chart: OrdinaryTaylorChartCertificate,
    target_chart: PlanarLeviCivitaBinaryChartCertificate,
    source_validation: ValidatedOrdinaryIVPChartCheckResult
    | OrdinaryEnclosureTransitionCheckResult,
    target_tube: PlanarLCAposterioriTubeCertificate,
) -> OrdinaryToPlanarLCEnclosureTransitionCheckResult:
    """Lift a proven ordinary state enclosure into a constrained LC tube."""

    if type(source_validation) not in {
        ValidatedOrdinaryIVPChartCheckResult,
        OrdinaryEnclosureTransitionCheckResult,
    }:
        raise TypeError("source_validation must be an exact ordinary result")
    target_tube_result = check_planar_lc_aposteriori_tube(target_tube, target_chart)
    source_tube_result = (
        source_validation.tube_result
        if type(source_validation) is ValidatedOrdinaryIVPChartCheckResult
        else source_validation.target_tube_result
    )
    source_parameter = float(certificate.source_parameter)
    target_parameter = float(certificate.target_parameter)
    handoff_time = float(certificate.handoff_time)
    time_cap = float(certificate.max_time_gap)
    identifiers_match = bool(
        certificate.transition_id
        and certificate.source_chart_id == source_chart.chart_id
        and certificate.target_chart_id == target_chart.chart_id
        and target_tube.chart_id == target_chart.chart_id
    )
    source_certified = source_validation.certified
    common_problem = bool(
        tuple(source_chart.masses) == tuple(target_chart.masses)
        and source_chart.dimension == 2
        and set(target_chart.pair).issubset({0, 1, 2})
    )
    mass_ratio_arithmetic_exact = _planar_lc_mass_ratio_arithmetic_exact(
        target_chart.masses, target_chart.pair
    )
    parameters_inside = bool(
        _parameter_in_interval(source_chart.parameter_interval, source_parameter)
        and _parameter_in_interval(target_chart.parameter_interval, target_parameter)
        and float(target_tube.anchor_parameter) == target_parameter
    )
    source_physical_time_parameterization_exact = False
    if type(source_validation) is ValidatedOrdinaryIVPChartCheckResult:
        try:
            parameter_interval_q = tuple(
                Fraction.from_float(value)
                for value in source_chart.parameter_interval
                if type(value) is float and np.isfinite(value)
            )
            physical_time_interval_q = tuple(
                Fraction.from_float(value)
                for value in source_chart.physical_time_interval
                if type(value) is float and np.isfinite(value)
            )
            binding_result = source_validation.binding_result
            source_physical_time_parameterization_exact = bool(
                len(parameter_interval_q) == 2
                and len(physical_time_interval_q) == 2
                and parameter_interval_q[1] - parameter_interval_q[0]
                == physical_time_interval_q[1] - physical_time_interval_q[0]
                and type(source_validation.checker_id) is str
                and source_validation.checker_id
                == "validated_ordinary_ivp_chart_checker_v1"
                and type(binding_result) is InitialValueBindingCheckResult
                and type(binding_result.binding_id) is str
                and bool(binding_result.binding_id)
                and type(binding_result.chart_id) is str
                and type(source_tube_result)
                is OrdinaryAposterioriTubeCheckResult
                and binding_result.chart_id
                == source_tube_result.chart_id
                == source_chart.chart_id
                and type(source_tube_result.tube_id) is str
                and bool(source_tube_result.tube_id)
                and type(source_tube_result.checker_id) is str
                and source_tube_result.checker_id
                == "independent_ordinary_aposteriori_tube_checker_v1"
                and type(binding_result.checker_id) is str
                and binding_result.checker_id
                == "independent_initial_value_binding_checker_v1"
                and type(binding_result.time_gap) is float
                and binding_result.time_gap == 0.0
            )
        except Exception:
            source_physical_time_parameterization_exact = False
    finite_time_cap = bool(np.isfinite(time_cap) and time_cap >= 0.0)
    branch_count = 0
    max_lift_gap = np.inf
    time_gap = np.inf
    target_initial_time_gap = np.inf
    entry_rho_floor = 0.0
    time_matches = False
    target_initial_time_contained = False
    lift_atlas_certified = False
    target_tube_contains_lift_atlas = False
    if (
        identifiers_match
        and source_certified
        and common_problem
        and mass_ratio_arithmetic_exact
        and parameters_inside
        and source_physical_time_parameterization_exact
        and finite_time_cap
        and target_tube_result.certified
    ):
        try:
            source_parameter_q = Fraction.from_float(source_parameter)
            target_parameter_q = Fraction.from_float(target_parameter)
            handoff_q = Fraction.from_float(handoff_time)
            source_time_q = _exact_rational_chart_physical_time_at_parameter(
                source_chart, source_parameter_q
            )
            target_time_q = _exact_rational_chart_physical_time_at_parameter(
                target_chart, target_parameter_q
            )
            time_gap_q = max(
                abs(source_time_q - handoff_q),
                abs(target_time_q - handoff_q),
                abs(source_time_q - target_time_q),
            )
            time_gap = _fraction_upper_float(time_gap_q)
            time_matches = time_gap_q <= Fraction.from_float(time_cap)
            target_initial_time_gap_q = abs(source_time_q - target_time_q)
            target_initial_time_gap = _fraction_upper_float(
                target_initial_time_gap_q
            )
            target_initial_time_contained = bool(
                target_initial_time_gap_q
                <= Fraction.from_float(float(target_tube.initial_error_bound))
            )

            source_q, source_v = _exact_rational_chart_projected_state_at_parameter(
                source_chart, source_parameter_q
            )
            source_radius = float(source_tube_result.gronwall_error_bound)
            position_boxes = _fraction_array_tube_intervals(source_q, source_radius)
            velocity_boxes = _fraction_array_tube_intervals(source_v, source_radius)
            flat_state = tuple(
                interval
                for matrix in (position_boxes, velocity_boxes)
                for row in matrix
                for interval in row
            )
            lift_atlas = planar_interval_to_regularized_binary_collision_chart_atlas(
                flat_state,
                np.asarray(source_chart.masses, dtype=float),
                pair=target_chart.pair,
            )
            branch_count = len(lift_atlas)
            entry_rho_floor = min(
                float(_interval_dot(branch.z, branch.z).lower)
                for branch in lift_atlas
            )
            lift_atlas_certified = bool(
                lift_atlas
                and all(
                    branch.branch_certificate is not None
                    and branch.branch_certificate.certified
                    for branch in lift_atlas
                )
            )
            target_anchor = _exact_rational_planar_lc_lifted_state_at_parameter(
                target_chart, target_parameter_q
            )
            target_error_q = Fraction.from_float(
                float(target_tube.initial_error_bound)
            )
            gap_q = Fraction(0)
            boxes_contained = True
            for branch in lift_atlas:
                branch_values = _flatten_interval_lc_state(branch)
                for interval, center in zip(branch_values, target_anchor):
                    lower_q = Fraction.from_float(float(interval.lower))
                    upper_q = Fraction.from_float(float(interval.upper))
                    gap_q = max(gap_q, abs(lower_q - center), abs(upper_q - center))
                    if not (
                        center - target_error_q <= lower_q
                        and upper_q <= center + target_error_q
                    ):
                        boxes_contained = False
            max_lift_gap = _fraction_upper_float(gap_q)
            target_tube_contains_lift_atlas = bool(
                lift_atlas_certified and boxes_contained
            )
        except (OverflowError, ValueError, ZeroDivisionError):
            pass

    obligations = (
        CertificateCheckObligation(
            "ordinary_to_lc_identifiers_match",
            identifiers_match,
            f"source={certificate.source_chart_id!r}; target={certificate.target_chart_id!r}",
        ),
        CertificateCheckObligation(
            "ordinary_to_lc_source_exact_enclosure_certified",
            source_certified,
            f"source_chart={source_chart.chart_id!r}",
        ),
        CertificateCheckObligation(
            "ordinary_to_lc_common_masses_dimension_pair",
            common_problem,
            f"pair={target_chart.pair!r}",
        ),
        CertificateCheckObligation(
            "ordinary_to_lc_exact_mass_ratio_arithmetic",
            mass_ratio_arithmetic_exact,
            "pair mass, alpha, beta, and third/pair mass ratio are exact binary rationals",
        ),
        CertificateCheckObligation(
            "ordinary_to_lc_parameters_inside_and_anchor_matches",
            parameters_inside,
            f"source_parameter={source_parameter}; target_parameter={target_parameter}",
        ),
        CertificateCheckObligation(
            "ordinary_to_lc_source_physical_time_parameterization_exact",
            source_physical_time_parameterization_exact,
            (
                "direct validated ordinary IVP result has exact binding time, "
                "matching source IDs, and equal exact parameter/time widths"
            ),
        ),
        CertificateCheckObligation(
            "ordinary_to_lc_handoff_time_matches",
            time_matches,
            f"time_gap={time_gap}; cap={time_cap}",
        ),
        CertificateCheckObligation(
            "ordinary_to_lc_target_initial_error_contains_physical_time",
            target_initial_time_contained,
            (
                f"source_target_time_gap={target_initial_time_gap}; "
                f"target_initial_error={target_tube.initial_error_bound}"
            ),
        ),
        CertificateCheckObligation(
            "ordinary_to_lc_target_lifted_tube_certified",
            target_tube_result.certified,
            f"target_tube={target_tube.tube_id!r}",
        ),
        CertificateCheckObligation(
            "ordinary_to_lc_branch_atlas_covers_source_enclosure",
            lift_atlas_certified,
            f"branch_count={branch_count}",
        ),
        CertificateCheckObligation(
            "ordinary_to_lc_entry_lift_rho_positive",
            entry_rho_floor > 0.0,
            f"entry_lift_rho_lower_bound={entry_rho_floor}",
        ),
        CertificateCheckObligation(
            "ordinary_to_lc_target_initial_error_contains_lift_atlas",
            target_tube_contains_lift_atlas,
            (
                f"max_lift_gap={max_lift_gap}; "
                f"target_initial_error={target_tube.initial_error_bound}"
            ),
        ),
        CertificateCheckObligation(
            "ordinary_to_lc_pair_energy_constraint_symbolic",
            target_tube_contains_lift_atlas,
            "h=.5|v|^2-M/rho and w=.25 L(z)^T v imply the exact constraint",
        ),
    )
    return OrdinaryToPlanarLCEnclosureTransitionCheckResult(
        transition_id=str(certificate.transition_id),
        checker_id="ordinary_to_planar_lc_enclosure_transition_checker_v1",
        target_tube_result=target_tube_result,
        obligations=obligations,
        lift_branch_count=int(branch_count),
        max_lift_box_gap=float(max_lift_gap),
        time_gap=float(time_gap),
        target_initial_time_gap=float(target_initial_time_gap),
        entry_lift_rho_lower_bound=float(entry_rho_floor),
    )


def check_gauge_aware_ordinary_to_planar_lc_enclosure_transition(
    certificate: OrdinaryToPlanarLCEnclosureTransitionCertificate,
    source_binding: InitialValueProblemBindingCertificate,
    source_tube: OrdinaryAposterioriTubeCertificate,
    source_chart: OrdinaryTaylorChartCertificate,
    target_chart: PlanarLeviCivitaBinaryChartCertificate,
    target_tube: PlanarLCAposterioriTubeCertificate,
) -> GaugeAwareOrdinaryToPlanarLCEnclosureTransitionCheckResult:
    """Check an ordinary-to-LC lift modulo one global ``Z/2`` gauge.

    This opt-in route derives its branch cover and sign graph from the exact
    source box.  The direct ordinary IVP is recomputed from its retained raw
    binding and tube certificates.  Continuation-result inputs are not in this
    checker's scope.  Ordinary source time is derived from the validated raw
    binding as ``initial_time + source_parameter - chart_parameter``; the
    chart's approximate physical-time interval is not used as exact evidence.
    It never changes, wraps, or promotes the legacy transition checker, and no
    existing proof route consumes its result type.
    """

    if type(certificate) is not OrdinaryToPlanarLCEnclosureTransitionCertificate:
        raise TypeError("certificate must have the exact ordinary-to-LC type")
    if type(source_binding) is not InitialValueProblemBindingCertificate:
        raise TypeError("source_binding must have the exact IVP binding type")
    if type(source_tube) is not OrdinaryAposterioriTubeCertificate:
        raise TypeError("source_tube must have the exact ordinary tube type")
    if type(source_chart) is not OrdinaryTaylorChartCertificate:
        raise TypeError("source_chart must have the exact ordinary chart type")
    if type(target_chart) is not PlanarLeviCivitaBinaryChartCertificate:
        raise TypeError("target_chart must have the exact planar LC chart type")
    if type(target_tube) is not PlanarLCAposterioriTubeCertificate:
        raise TypeError("target_tube must have the exact planar LC tube type")

    raw_primitive_schema_valid = _gauge_aware_raw_primitive_schema_valid(
        certificate,
        source_binding,
        source_tube,
        source_chart,
        target_chart,
        target_tube,
    )
    if raw_primitive_schema_valid:
        try:
            source_validation = check_validated_ordinary_ivp_chart(
                source_binding, source_tube, source_chart
            )
        except Exception:
            source_validation = _rejected_gauge_aware_direct_ordinary_source(
                source_binding, source_tube, source_chart
            )
        try:
            target_tube_result = _check_planar_lc_tube_or_reject_malformed(
                target_tube, target_chart
            )
        except Exception:
            target_tube_result = _rejected_planar_lc_tube_result(
                target_tube, target_chart
            )
    else:
        source_validation = _rejected_gauge_aware_direct_ordinary_source(
            source_binding, source_tube, source_chart
        )
        target_tube_result = _rejected_planar_lc_tube_result(
            target_tube, target_chart
        )
    source_state_box: tuple[tuple[Fraction, Fraction], ...] = ()
    relative_position_box = (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
    )
    canonical_lift_case = ""
    patch_vertex_ids: tuple[str, ...] = ()
    raw_patch_boxes: tuple[
        tuple[tuple[Fraction, Fraction], ...], ...
    ] = ()
    derived_edges: tuple[PlanarLCGaugeOverlapEdge, ...] = ()
    gauge_result: PlanarLCGaugeGluingCheckResult | None = None
    tested_assignments: tuple[tuple[tuple[str, int], ...], ...] = ()
    tested_lift_max_gaps: tuple[Fraction, ...] = ()
    tested_containments: tuple[bool, ...] = ()
    selected_complement_index = -1
    selected_assignment: tuple[tuple[str, int], ...] = ()
    selected_transformed_patch_boxes: tuple[
        tuple[tuple[Fraction, Fraction], ...], ...
    ] = ()
    exact_declared_time_gap = Fraction(0)
    exact_source_target_time_gap = Fraction(0)
    entry_lift_rho_lower_bound = Fraction(0)

    identifiers_match = False
    source_certified = False
    common_problem = False
    mass_ratio_arithmetic_exact = False
    parameters_inside = False
    declared_handoff_time_cap = False
    target_tube_certified = False
    source_box_reconstructed = False
    selected_pair_collision_free = False
    canonical_branch_grammar = False
    canonical_lift_count = False
    derived_gauge_graph_certified = False
    entry_lift_rho_positive = False
    target_initial_time_contained = False
    one_global_complement_contains_all_lifts = False
    existential_selected_lifts_constrained = False

    def square_lower(bounds: tuple[Fraction, Fraction]) -> Fraction:
        lower, upper = bounds
        if lower <= 0 <= upper:
            return Fraction(0)
        return min(lower * lower, upper * upper)

    def transform_patch_box(
        box: tuple[tuple[Fraction, Fraction], ...], bit: int
    ) -> tuple[tuple[Fraction, Fraction], ...]:
        return tuple(
            (-upper, -lower) if bit == 1 and index < 4 else (lower, upper)
            for index, (lower, upper) in enumerate(box)
        )

    try:
        if not raw_primitive_schema_valid:
            raise ValueError("noncanonical raw primitive schema")
        identifiers_match = bool(
            type(certificate.transition_id) is str
            and bool(certificate.transition_id)
            and type(certificate.source_chart_id) is str
            and type(certificate.target_chart_id) is str
            and certificate.source_chart_id == source_chart.chart_id
            and certificate.target_chart_id == target_chart.chart_id
            and source_binding.chart_id == source_chart.chart_id
            and source_tube.chart_id == source_chart.chart_id
            and target_tube.chart_id == target_chart.chart_id
        )
        source_certified = _gauge_aware_direct_ordinary_source_valid(
            source_binding, source_tube, source_chart, source_validation
        )
        ordered_pair = target_chart.pair
        common_problem = bool(
            type(source_chart.masses) is tuple
            and type(target_chart.masses) is tuple
            and source_chart.masses == target_chart.masses
            and len(source_chart.masses) == 3
            and all(
                type(mass) is float and np.isfinite(mass) and mass > 0.0
                for mass in source_chart.masses
            )
            and type(source_chart.dimension) is int
            and source_chart.dimension == 2
            and type(ordered_pair) is tuple
            and len(ordered_pair) == 2
            and all(type(index) is int for index in ordered_pair)
            and ordered_pair[0] != ordered_pair[1]
            and set(ordered_pair).issubset({0, 1, 2})
        )
        if common_problem:
            mass_ratio_arithmetic_exact = _planar_lc_mass_ratio_arithmetic_exact(
                target_chart.masses, ordered_pair
            )

        exact_scalar_fields = bool(
            type(certificate.source_parameter) is float
            and type(certificate.target_parameter) is float
            and type(certificate.handoff_time) is float
            and type(certificate.max_time_gap) is float
            and type(source_binding.initial_time) is float
            and type(source_binding.chart_parameter) is float
            and type(target_tube.anchor_parameter) is float
            and type(target_tube.initial_error_bound) is float
            and all(
                np.isfinite(value)
                for value in (
                    certificate.source_parameter,
                    certificate.target_parameter,
                    certificate.handoff_time,
                    certificate.max_time_gap,
                    source_binding.initial_time,
                    source_binding.chart_parameter,
                    target_tube.anchor_parameter,
                    target_tube.initial_error_bound,
                )
            )
            and certificate.max_time_gap >= 0.0
            and target_tube.initial_error_bound >= 0.0
        )
        parameters_inside = bool(
            exact_scalar_fields
            and _parameter_in_interval(
                source_chart.parameter_interval, certificate.source_parameter
            )
            and _parameter_in_interval(
                target_chart.parameter_interval, certificate.target_parameter
            )
            and target_tube.anchor_parameter == certificate.target_parameter
        )
        target_tube_certified = bool(
            type(target_tube_result) is PlanarLCAposterioriTubeCheckResult
            and target_tube_result.checker_id
            == "independent_planar_lc_aposteriori_tube_checker_v1"
            and target_tube_result.chart_id == target_chart.chart_id
            and target_tube_result.tube_id == target_tube.tube_id
            and target_tube_result.certified
        )

        if parameters_inside:
            source_parameter_q = Fraction.from_float(certificate.source_parameter)
            target_parameter_q = Fraction.from_float(certificate.target_parameter)
            handoff_time_q = Fraction.from_float(certificate.handoff_time)
            source_time_q = (
                Fraction.from_float(source_binding.initial_time)
                + source_parameter_q
                - Fraction.from_float(source_binding.chart_parameter)
            )
            target_time_q = _exact_rational_chart_physical_time_at_parameter(
                target_chart, target_parameter_q
            )
            exact_source_target_time_gap = abs(source_time_q - target_time_q)
            exact_declared_time_gap = max(
                abs(source_time_q - handoff_time_q),
                abs(target_time_q - handoff_time_q),
                exact_source_target_time_gap,
            )
            declared_handoff_time_cap = bool(
                exact_declared_time_gap
                <= Fraction.from_float(certificate.max_time_gap)
            )
            target_initial_time_contained = bool(
                exact_source_target_time_gap
                <= Fraction.from_float(target_tube.initial_error_bound)
            )

        source_tube_result = source_validation.tube_result
        if source_certified and parameters_inside and common_problem:
            if not (
                type(source_tube_result) is OrdinaryAposterioriTubeCheckResult
                and type(source_tube_result.gronwall_error_bound) is float
                and np.isfinite(source_tube_result.gronwall_error_bound)
                and source_tube_result.gronwall_error_bound >= 0.0
            ):
                raise ValueError("source enclosure radius is not an exact finite float")
            source_q, source_v = _exact_rational_chart_projected_state_at_parameter(
                source_chart,
                Fraction.from_float(certificate.source_parameter),
            )
            centers = tuple(
                value
                for array in (source_q, source_v)
                for value in np.asarray(array, dtype=object).reshape(-1)
            )
            if len(centers) != 12 or not all(
                type(value) is Fraction for value in centers
            ):
                raise ValueError("ordinary source state must have 12 exact components")
            source_radius_q = Fraction.from_float(
                source_tube_result.gronwall_error_bound
            )
            rebuilt_box = []
            for center in centers:
                lower_float = _fraction_lower_float(center - source_radius_q)
                upper_float = _fraction_upper_float(center + source_radius_q)
                rebuilt_box.append(
                    (
                        Fraction.from_float(lower_float),
                        Fraction.from_float(upper_float),
                    )
                )
            source_state_box = tuple(rebuilt_box)
            source_box_reconstructed = bool(
                len(source_state_box) == 12
                and all(
                    type(bounds) is tuple
                    and len(bounds) == 2
                    and all(type(value) is Fraction for value in bounds)
                    and bounds[0] <= bounds[1]
                    for bounds in source_state_box
                )
            )

            first, second = ordered_pair
            relative_position_box = tuple(
                (
                    source_state_box[2 * second + axis][0]
                    - source_state_box[2 * first + axis][1],
                    source_state_box[2 * second + axis][1]
                    - source_state_box[2 * first + axis][0],
                )
                for axis in range(2)
            )  # type: ignore[assignment]
            selected_pair_collision_free = bool(
                sum(square_lower(bounds) for bounds in relative_position_box) > 0
            )

            x_low, x_high = relative_position_box[0]
            y_low, y_high = relative_position_box[1]
            expected_branch_count = 0
            if y_low >= 0:
                canonical_lift_case = "closed_upper_singleton"
                expected_branch_count = 1
            elif y_high <= 0:
                canonical_lift_case = "closed_lower_singleton"
                expected_branch_count = 1
            elif x_low > 0:
                canonical_lift_case = "right_half_singleton"
                expected_branch_count = 1
            elif y_low < 0 < y_high and x_high < 0:
                canonical_lift_case = "strict_negative_cut_two_patch"
                expected_branch_count = 2
            canonical_branch_grammar = expected_branch_count in (1, 2)

            if canonical_branch_grammar and selected_pair_collision_free:
                lift_atlas = (
                    planar_interval_to_regularized_binary_collision_chart_atlas(
                        tuple(
                            (float(lower), float(upper))
                            for lower, upper in source_state_box
                        ),
                        np.asarray(source_chart.masses, dtype=float),
                        pair=ordered_pair,
                    )
                )
                raw_patch_boxes = tuple(
                    tuple(
                        (
                            Fraction.from_float(float(interval.lower)),
                            Fraction.from_float(float(interval.upper)),
                        )
                        for interval in _flatten_interval_lc_state(branch)
                    )
                    for branch in lift_atlas
                )
                canonical_lift_count = bool(
                    len(raw_patch_boxes) == expected_branch_count
                    and all(len(box) == 13 for box in raw_patch_boxes)
                )

            if canonical_lift_count:
                prefix = certificate.transition_id
                if canonical_lift_case == "strict_negative_cut_two_patch":
                    patch_vertex_ids = (
                        f"{prefix}:patch:0-upper",
                        f"{prefix}:patch:1-lower",
                    )
                    derived_edges = (
                        PlanarLCGaugeOverlapEdge(
                            overlap_id=f"{prefix}:negative-axis-overlap",
                            source_chart_id=patch_vertex_ids[0],
                            target_chart_id=patch_vertex_ids[1],
                            parity=1,
                        ),
                    )
                else:
                    suffix = {
                        "closed_upper_singleton": "upper",
                        "closed_lower_singleton": "lower",
                        "right_half_singleton": "right",
                    }[canonical_lift_case]
                    patch_vertex_ids = (f"{prefix}:patch:0-{suffix}",)
                    derived_edges = ()
                gauge_result = check_planar_lc_gauge_gluing(
                    PlanarLCGaugeGluingCertificate(
                        certificate_id=f"{prefix}:derived-gauge-cover",
                        chart_ids=patch_vertex_ids,
                        overlaps=derived_edges,
                        source="derived_canonical_ordinary_to_lc_lift_cover",
                    )
                )
                derived_gauge_graph_certified = bool(
                    type(gauge_result) is PlanarLCGaugeGluingCheckResult
                    and gauge_result.chart_ids == patch_vertex_ids
                    and gauge_result.checked_overlaps == derived_edges
                    and gauge_result.certified
                )
                entry_lift_rho_lower_bound = min(
                    sum(square_lower(box[index]) for index in (0, 1))
                    for box in raw_patch_boxes
                )
                entry_lift_rho_positive = entry_lift_rho_lower_bound > 0

            if (
                derived_gauge_graph_certified
                and entry_lift_rho_positive
                and target_tube_certified
                and parameters_inside
            ):
                base_assignment = gauge_result.gauge_assignment  # type: ignore[union-attr]
                complement_assignment = tuple(
                    (chart_id, bit ^ 1) for chart_id, bit in base_assignment
                )
                tested_assignments = (base_assignment, complement_assignment)
                target_anchor = _exact_rational_planar_lc_lifted_state_at_parameter(
                    target_chart,
                    Fraction.from_float(certificate.target_parameter),
                )
                target_radius_q = Fraction.from_float(
                    target_tube.initial_error_bound
                )
                transformed_by_assignment = []
                gaps = []
                containments = []
                for assignment in tested_assignments:
                    assignment_by_id = dict(assignment)
                    transformed_boxes = tuple(
                        transform_patch_box(
                            box, assignment_by_id[patch_vertex_ids[index]]
                        )
                        for index, box in enumerate(raw_patch_boxes)
                    )
                    gap = max(
                        (
                            max(abs(lower - center), abs(upper - center))
                            for box in transformed_boxes
                            for (lower, upper), center in zip(box, target_anchor)
                        ),
                        default=Fraction(0),
                    )
                    contained = bool(
                        gap <= target_radius_q
                        and exact_source_target_time_gap <= target_radius_q
                    )
                    transformed_by_assignment.append(transformed_boxes)
                    gaps.append(gap)
                    containments.append(contained)
                tested_lift_max_gaps = tuple(gaps)
                tested_containments = tuple(containments)
                for index, contained in enumerate(tested_containments):
                    if contained:
                        selected_complement_index = index
                        selected_assignment = tested_assignments[index]
                        selected_transformed_patch_boxes = (
                            transformed_by_assignment[index]
                        )
                        break
                one_global_complement_contains_all_lifts = (
                    selected_complement_index in (0, 1)
                )
                existential_selected_lifts_constrained = bool(
                    one_global_complement_contains_all_lifts
                    and mass_ratio_arithmetic_exact
                    and selected_pair_collision_free
                    and entry_lift_rho_positive
                )
    except Exception:
        # Exact-class inputs still contain untrusted serialized fields.  Any
        # malformed arithmetic, indexing, equality, or interval path rejects.
        pass

    obligations = (
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[0],
            raw_primitive_schema_valid,
            (
                "all six inputs have exact public dataclass types and canonical "
                "built-in primitive field schemas"
            ),
        ),
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[1],
            identifiers_match,
            "transition, source chart, target chart, and target tube identifiers match",
        ),
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[2],
            source_certified,
            "the ordinary result is certified and bound to this exact source serialization",
        ),
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[3],
            common_problem,
            "exact masses, planar dimension, and the ordered LC pair are admissible",
        ),
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[4],
            mass_ratio_arithmetic_exact,
            "pair-mass ratios are exact in the proof-facing binary arithmetic gate",
        ),
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[5],
            parameters_inside,
            "source and target parameters are inside and the target anchor matches",
        ),
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[6],
            declared_handoff_time_cap,
            f"exact_declared_time_gap={exact_declared_time_gap!s}",
        ),
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[7],
            target_tube_certified,
            "the target lifted a-posteriori tube was freshly checked",
        ),
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[8],
            source_box_reconstructed,
            f"source_box_dimension={len(source_state_box)}",
        ),
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[9],
            selected_pair_collision_free,
            f"relative_position_box={relative_position_box!s}",
        ),
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[10],
            canonical_branch_grammar,
            f"canonical_lift_case={canonical_lift_case!r}",
        ),
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[11],
            canonical_lift_count,
            f"derived_patch_count={len(raw_patch_boxes)}",
        ),
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[12],
            derived_gauge_graph_certified,
            f"derived_edge_count={len(derived_edges)}",
        ),
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[13],
            entry_lift_rho_positive,
            f"entry_lift_rho_lower_bound={entry_lift_rho_lower_bound!s}",
        ),
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[14],
            target_initial_time_contained,
            f"exact_source_target_time_gap={exact_source_target_time_gap!s}",
        ),
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[15],
            one_global_complement_contains_all_lifts,
            (
                f"tested_lift_max_gaps={tested_lift_max_gaps!s}; "
                f"selected_complement_index={selected_complement_index}"
            ),
        ),
        CertificateCheckObligation(
            _GAUGE_AWARE_ORDINARY_TO_PLANAR_LC_OBLIGATION_NAMES[16],
            existential_selected_lifts_constrained,
            (
                "for each physical source state, the selected representative "
                "satisfies h=.5|v|^2-M/rho and w=.25 L(z)^T v; this is not "
                "a claim about every point of an interval box or target tube"
            ),
        ),
    )
    transition_id = (
        certificate.transition_id
        if type(certificate.transition_id) is str
        else ""
    )
    return GaugeAwareOrdinaryToPlanarLCEnclosureTransitionCheckResult(
        transition_id=transition_id,
        checker_id=(
            "gauge_aware_ordinary_to_planar_lc_enclosure_transition_checker_v1"
        ),
        raw_transition_certificate=certificate,
        raw_source_binding=source_binding,
        raw_source_tube=source_tube,
        raw_source_chart=source_chart,
        raw_target_chart=target_chart,
        source_validation=source_validation,
        raw_target_tube=target_tube,
        target_tube_result=target_tube_result,
        obligations=obligations,
        source_state_box=source_state_box,
        relative_position_box=relative_position_box,
        canonical_lift_case=canonical_lift_case,
        patch_vertex_ids=patch_vertex_ids,
        raw_patch_boxes=raw_patch_boxes,
        derived_edges=derived_edges,
        gauge_result=gauge_result,
        tested_assignments=tested_assignments,
        tested_lift_max_gaps=tested_lift_max_gaps,
        tested_containments=tested_containments,
        selected_complement_index=selected_complement_index,
        selected_assignment=selected_assignment,
        selected_transformed_patch_boxes=selected_transformed_patch_boxes,
        exact_declared_time_gap=exact_declared_time_gap,
        exact_source_target_time_gap=exact_source_target_time_gap,
        entry_lift_rho_lower_bound=entry_lift_rho_lower_bound,
    )


def check_planar_lc_to_ordinary_enclosure_transition(
    certificate: PlanarLCToOrdinaryEnclosureTransitionCertificate,
    source_chart: PlanarLeviCivitaBinaryChartCertificate,
    target_chart: OrdinaryTaylorChartCertificate,
    source_entry: OrdinaryToPlanarLCEnclosureTransitionCheckResult,
    target_tube: OrdinaryAposterioriTubeCertificate,
) -> PlanarLCToOrdinaryEnclosureTransitionCheckResult:
    """Project a constrained LC branch into an elapsed-time ordinary tube."""

    if type(source_entry) is not OrdinaryToPlanarLCEnclosureTransitionCheckResult:
        raise TypeError("source_entry must be an exact ordinary-to-LC result")
    target_tube_result = check_ordinary_aposteriori_tube(target_tube, target_chart)
    source_parameter = float(certificate.source_parameter)
    target_parameter = float(certificate.target_parameter)
    identifiers_match = bool(
        certificate.transition_id
        and certificate.source_chart_id == source_chart.chart_id
        and certificate.target_chart_id == target_chart.chart_id
        and source_entry.target_tube_result.chart_id == source_chart.chart_id
        and target_tube.chart_id == target_chart.chart_id
    )
    common_problem = bool(
        tuple(source_chart.masses) == tuple(target_chart.masses)
        and target_chart.dimension == 2
    )
    parameters_inside = bool(
        _parameter_in_interval(source_chart.parameter_interval, source_parameter)
        and _parameter_in_interval(target_chart.parameter_interval, target_parameter)
        and float(target_tube.anchor_parameter) == target_parameter
    )
    elapsed_time_anchor = False
    rho_floor = 0.0
    max_gap = np.inf
    time_origin = (np.inf, -np.inf)
    punctured_projection = False
    target_contains_projection = False
    if (
        identifiers_match
        and source_entry.certified
        and common_problem
        and parameters_inside
        and target_tube_result.certified
    ):
        try:
            target_parameter_q = Fraction.from_float(target_parameter)
            elapsed_time_anchor = (
                _exact_rational_chart_physical_time_at_parameter(
                    target_chart, target_parameter_q
                )
                == 0
            )
            source_solution = _regularized_binary_solution_from_certificate(
                source_chart
            )
            source_error = float(
                source_entry.target_tube_result.gronwall_error_bound
            )
            lifted_state = _planar_lc_state_intervals(
                source_solution,
                (source_parameter, source_parameter),
                inflate=source_error,
            )
            positions, velocities, physical_time, rho = (
                _project_interval_planar_lc_state(
                    lifted_state,
                    np.asarray(source_chart.masses, dtype=float),
                    source_chart.pair,
                )
            )
            rho_floor = float(rho.lower)
            punctured_projection = rho_floor > 0.0
            time_origin = (float(physical_time.lower), float(physical_time.upper))
            target_q, target_v = _exact_rational_chart_projected_state_at_parameter(
                target_chart, target_parameter_q
            )
            target_error_q = Fraction.from_float(
                float(target_tube.initial_error_bound)
            )
            gap_q = Fraction(0)
            boxes_contained = True
            for intervals, centers in ((positions, target_q), (velocities, target_v)):
                for index in np.ndindex(intervals.shape):
                    interval = intervals[index]
                    center = centers[index]
                    lower_q = Fraction.from_float(float(interval.lower))
                    upper_q = Fraction.from_float(float(interval.upper))
                    gap_q = max(gap_q, abs(lower_q - center), abs(upper_q - center))
                    if not (
                        center - target_error_q <= lower_q
                        and upper_q <= center + target_error_q
                    ):
                        boxes_contained = False
            max_gap = _fraction_upper_float(gap_q)
            target_contains_projection = bool(
                punctured_projection and elapsed_time_anchor and boxes_contained
            )
        except (OverflowError, ValueError, ZeroDivisionError):
            pass

    obligations = (
        CertificateCheckObligation(
            "lc_to_ordinary_identifiers_match",
            identifiers_match,
            f"source={certificate.source_chart_id!r}; target={certificate.target_chart_id!r}",
        ),
        CertificateCheckObligation(
            "lc_to_ordinary_constrained_source_entry_certified",
            source_entry.constrained_newtonian_lift_certified,
            f"source_transition={source_entry.transition_id!r}",
        ),
        CertificateCheckObligation(
            "lc_to_ordinary_common_masses_dimension",
            common_problem,
            "source LC and target ordinary charts share planar masses",
        ),
        CertificateCheckObligation(
            "lc_to_ordinary_parameters_inside_and_anchor_matches",
            parameters_inside,
            f"source_parameter={source_parameter}; target_parameter={target_parameter}",
        ),
        CertificateCheckObligation(
            "lc_to_ordinary_exit_rho_positive",
            punctured_projection,
            f"source_rho_lower_bound={rho_floor}",
        ),
        CertificateCheckObligation(
            "lc_to_ordinary_target_elapsed_time_anchor_zero",
            elapsed_time_anchor,
            f"target_parameter={target_parameter}",
        ),
        CertificateCheckObligation(
            "lc_to_ordinary_target_tube_certified",
            target_tube_result.certified,
            f"target_tube={target_tube.tube_id!r}",
        ),
        CertificateCheckObligation(
            "lc_to_ordinary_target_initial_error_contains_projection",
            target_contains_projection,
            (
                f"max_projected_lift_gap={max_gap}; "
                f"target_initial_error={target_tube.initial_error_bound}"
            ),
        ),
        CertificateCheckObligation(
            "lc_to_ordinary_autonomous_time_origin_enclosed",
            bool(
                np.isfinite(time_origin[0])
                and np.isfinite(time_origin[1])
                and time_origin[0] <= time_origin[1]
            ),
            f"physical_time_origin_interval={time_origin!r}",
        ),
    )
    return PlanarLCToOrdinaryEnclosureTransitionCheckResult(
        transition_id=str(certificate.transition_id),
        checker_id="planar_lc_to_ordinary_enclosure_transition_checker_v1",
        target_tube_result=target_tube_result,
        obligations=obligations,
        source_rho_lower_bound=float(rho_floor),
        max_projected_lift_gap=float(max_gap),
        physical_time_origin_interval=time_origin,
        constrained_source_entry_certified=bool(
            source_entry.constrained_lc_entry_certified
        ),
    )


def _interval_minimum_pair_distance(positions: np.ndarray) -> float:
    positions = np.asarray(positions, dtype=object)
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError("ordinary position enclosure must have shape (3, dimension)")
    floor = np.inf
    for i in range(3):
        for j in range(i + 1, 3):
            squared = Fraction(0)
            for axis in range(positions.shape[1]):
                difference = positions[j, axis] - positions[i, axis]
                if difference.lower <= 0.0 <= difference.upper:
                    component_floor = 0.0
                else:
                    component_floor = min(
                        abs(float(difference.lower)),
                        abs(float(difference.upper)),
                    )
                component_q = Fraction.from_float(float(component_floor))
                squared += component_q * component_q
            floor = min(
                floor,
                _fraction_sqrt_float(squared, upward=False),
            )
    return float(floor)


@dataclass(frozen=True)
class _IntervalDual:
    value: FloatInterval
    derivative: tuple[FloatInterval, ...]

    def __add__(self, other: object) -> "_IntervalDual":
        right = _as_interval_dual(other, len(self.derivative))
        return _IntervalDual(
            self.value + right.value,
            tuple(a + b for a, b in zip(self.derivative, right.derivative)),
        )

    __radd__ = __add__

    def __neg__(self) -> "_IntervalDual":
        return self.scale(-1.0)

    def __sub__(self, other: object) -> "_IntervalDual":
        return self + (-_as_interval_dual(other, len(self.derivative)))

    def __rsub__(self, other: object) -> "_IntervalDual":
        return _as_interval_dual(other, len(self.derivative)) - self

    def __mul__(self, other: object) -> "_IntervalDual":
        right = _as_interval_dual(other, len(self.derivative))
        return _IntervalDual(
            self.value * right.value,
            tuple(
                a * right.value + self.value * b
                for a, b in zip(self.derivative, right.derivative)
            ),
        )

    __rmul__ = __mul__

    def scale(self, scalar: float) -> "_IntervalDual":
        return _IntervalDual(
            self.value.scale(float(scalar)),
            tuple(item.scale(float(scalar)) for item in self.derivative),
        )

    def positive_power(self, exponent: float) -> "_IntervalDual":
        value = self.value.positive_power(float(exponent))
        multiplier = self.value.positive_power(float(exponent) - 1.0).scale(
            float(exponent)
        )
        return _IntervalDual(
            value,
            tuple(multiplier * item for item in self.derivative),
        )


def _as_interval_dual(value: object, dimension: int) -> _IntervalDual:
    if isinstance(value, _IntervalDual):
        if len(value.derivative) != dimension:
            raise ValueError("dual dimensions do not match")
        return value
    interval = value if isinstance(value, FloatInterval) else FloatInterval.point(float(value))
    return _IntervalDual(
        interval,
        tuple(FloatInterval.point(0.0) for _ in range(dimension)),
    )


def _planar_lc_state_intervals(
    solution: RegularizedBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    inflate: float = 0.0,
) -> tuple[FloatInterval, ...]:
    variable = FloatInterval(*parameter_interval)
    blocks = (
        interval_array_series_eval(solution.z, variable).reshape(-1),
        interval_array_series_eval(solution.z_velocity, variable).reshape(-1),
        np.asarray([interval_polynomial_eval(solution.pair_energy, variable)], dtype=object),
        interval_array_series_eval(solution.binary_center, variable).reshape(-1),
        interval_array_series_eval(solution.binary_center_velocity, variable).reshape(-1),
        interval_array_series_eval(solution.third_offset, variable).reshape(-1),
        interval_array_series_eval(solution.third_offset_velocity, variable).reshape(-1),
        np.asarray([interval_polynomial_eval(solution.physical_time, variable)], dtype=object),
    )
    radius = float(inflate)
    out: list[FloatInterval] = []
    for item in np.concatenate(blocks):
        interval = item if isinstance(item, FloatInterval) else FloatInterval.point(float(item))
        out.append(
            FloatInterval(
                float(np.nextafter(interval.lower - radius, -np.inf)),
                float(np.nextafter(interval.upper + radius, np.inf)),
            )
        )
    if len(out) != 14:
        raise ValueError("planar LC lifted state must have dimension 14")
    return tuple(out)


def _flatten_interval_lc_state(
    state: object,
) -> tuple[FloatInterval, ...]:
    blocks = (
        np.asarray(state.z, dtype=object).reshape(-1),
        np.asarray(state.z_velocity, dtype=object).reshape(-1),
        np.asarray([state.pair_energy], dtype=object),
        np.asarray(state.binary_center, dtype=object).reshape(-1),
        np.asarray(state.binary_center_velocity, dtype=object).reshape(-1),
        np.asarray(state.third_offset, dtype=object).reshape(-1),
        np.asarray(state.third_offset_velocity, dtype=object).reshape(-1),
    )
    out = tuple(item for item in np.concatenate(blocks))
    if len(out) != 13 or not all(isinstance(item, FloatInterval) for item in out):
        raise ValueError("interval LC anchor must have 13 lifted state components")
    return out


def _exact_rational_planar_lc_lifted_state_at_parameter(
    chart: PlanarLeviCivitaBinaryChartCertificate,
    parameter: Fraction,
) -> tuple[Fraction, ...]:
    blocks = (
        np.asarray(_evaluate_fraction_coefficients(chart.z_coefficients, parameter), dtype=object).reshape(-1),
        np.asarray(_evaluate_fraction_coefficients(chart.z_velocity_coefficients, parameter), dtype=object).reshape(-1),
        np.asarray([_evaluate_fraction_coefficients(chart.pair_energy_coefficients, parameter)], dtype=object),
        np.asarray(_evaluate_fraction_coefficients(chart.binary_center_coefficients, parameter), dtype=object).reshape(-1),
        np.asarray(_evaluate_fraction_coefficients(chart.binary_center_velocity_coefficients, parameter), dtype=object).reshape(-1),
        np.asarray(_evaluate_fraction_coefficients(chart.third_offset_coefficients, parameter), dtype=object).reshape(-1),
        np.asarray(_evaluate_fraction_coefficients(chart.third_offset_velocity_coefficients, parameter), dtype=object).reshape(-1),
    )
    out = tuple(item for item in np.concatenate(blocks))
    if len(out) != 13 or not all(isinstance(item, Fraction) for item in out):
        raise ValueError("serialized LC anchor must have 13 rational components")
    return out


def _planar_lc_derivative_intervals(
    solution: RegularizedBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
) -> tuple[FloatInterval, ...]:
    variable = FloatInterval(*parameter_interval)
    blocks = (
        interval_array_series_eval(
            interval_array_derivative_coefficients(solution.z), variable
        ).reshape(-1),
        interval_array_series_eval(
            interval_array_derivative_coefficients(solution.z_velocity), variable
        ).reshape(-1),
        np.asarray([
            interval_polynomial_eval(
                interval_array_derivative_coefficients(solution.pair_energy),
                variable,
            )
        ], dtype=object),
        interval_array_series_eval(
            interval_array_derivative_coefficients(solution.binary_center),
            variable,
        ).reshape(-1),
        interval_array_series_eval(
            interval_array_derivative_coefficients(
                solution.binary_center_velocity
            ),
            variable,
        ).reshape(-1),
        interval_array_series_eval(
            interval_array_derivative_coefficients(solution.third_offset),
            variable,
        ).reshape(-1),
        interval_array_series_eval(
            interval_array_derivative_coefficients(solution.third_offset_velocity),
            variable,
        ).reshape(-1),
        np.asarray([
            interval_polynomial_eval(
                interval_array_derivative_coefficients(solution.physical_time),
                variable,
            )
        ], dtype=object),
    )
    out = tuple(item for item in np.concatenate(blocks))
    if len(out) != 14:
        raise ValueError("planar LC derivative must have dimension 14")
    return out


def _project_interval_planar_lc_state(
    state: tuple[FloatInterval, ...],
    masses: np.ndarray,
    pair: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, FloatInterval, FloatInterval]:
    if len(state) != 14:
        raise ValueError("planar LC state must have dimension 14")
    x, y = state[0:2]
    wx, wy = state[2:4]
    center = np.asarray(state[5:7], dtype=object)
    center_velocity = np.asarray(state[7:9], dtype=object)
    third_offset = np.asarray(state[9:11], dtype=object)
    third_velocity = np.asarray(state[11:13], dtype=object)
    physical_time = state[13]
    rho = x * x + y * y
    if rho.lower <= 0.0:
        raise ValueError("physical LC projection requires a rho-positive slab")
    relative_position = np.asarray(
        [x * x - y * y, (x * y).scale(2.0)], dtype=object
    )
    inverse_rho = rho.reciprocal()
    relative_velocity = np.asarray(
        [
            ((x * wx - y * wy).scale(2.0)) * inverse_rho,
            ((y * wx + x * wy).scale(2.0)) * inverse_rho,
        ],
        dtype=object,
    )
    first, second = pair
    third = ({0, 1, 2} - {first, second}).pop()
    pair_mass = float(masses[first] + masses[second])
    alpha = float(masses[second] / pair_mass)
    beta = float(masses[first] / pair_mass)
    positions = _interval_zero_array((3, 2))
    velocities = _interval_zero_array((3, 2))
    positions[first] = _interval_vector_sub(
        center, _interval_vector_scale(relative_position, alpha)
    )
    positions[second] = _interval_vector_add(
        center, _interval_vector_scale(relative_position, beta)
    )
    positions[third] = _interval_vector_add(center, third_offset)
    velocities[first] = _interval_vector_sub(
        center_velocity, _interval_vector_scale(relative_velocity, alpha)
    )
    velocities[second] = _interval_vector_add(
        center_velocity, _interval_vector_scale(relative_velocity, beta)
    )
    velocities[third] = _interval_vector_add(center_velocity, third_velocity)
    return positions, velocities, physical_time, rho


def _planar_lc_dual_rhs(
    state: tuple[FloatInterval, ...],
    masses: np.ndarray,
    pair: tuple[int, int],
) -> tuple[tuple[_IntervalDual, ...], float]:
    dimension = len(state)
    dual = tuple(
        _IntervalDual(
            interval,
            tuple(
                FloatInterval.point(1.0 if row == column else 0.0)
                for column in range(dimension)
            ),
        )
        for row, interval in enumerate(state)
    )
    x, y = dual[0:2]
    wx, wy = dual[2:4]
    energy = dual[4]
    center = dual[5:7]
    center_velocity = dual[7:9]
    third_offset = dual[9:11]
    third_velocity = dual[11:13]
    del center
    rho = x * x + y * y
    relative = (x * x - y * y, (x * y).scale(2.0))
    first, second = pair
    third = ({0, 1, 2} - {first, second}).pop()
    pair_mass = float(masses[first] + masses[second])
    alpha = float(masses[second] / pair_mass)
    beta = float(masses[first] / pair_mass)
    d_first = tuple(
        third_offset[k] + relative[k].scale(alpha) for k in range(2)
    )
    d_second = tuple(
        third_offset[k] - relative[k].scale(beta) for k in range(2)
    )

    def inverse_square(vector: tuple[_IntervalDual, _IntervalDual]) -> tuple[_IntervalDual, _IntervalDual]:
        norm_square = vector[0] * vector[0] + vector[1] * vector[1]
        if norm_square.value.lower <= 0.0:
            raise ValueError("LC tube reaches a third-body collision denominator")
        inverse_cube = norm_square.positive_power(-1.5)
        return (vector[0] * inverse_cube, vector[1] * inverse_cube)

    field_first = inverse_square(d_first)
    field_second = inverse_square(d_second)
    center_acceleration = tuple(
        (
            field_first[k].scale(float(masses[first]))
            + field_second[k].scale(float(masses[second]))
        ).scale(float(masses[third] / pair_mass))
        for k in range(2)
    )
    third_acceleration = tuple(
        -field_first[k].scale(float(masses[first]))
        - field_second[k].scale(float(masses[second]))
        for k in range(2)
    )
    offset_acceleration = tuple(
        third_acceleration[k] - center_acceleration[k] for k in range(2)
    )
    perturbation = tuple(
        (field_second[k] - field_first[k]).scale(float(masses[third]))
        for k in range(2)
    )
    at_perturbation = (
        (x * perturbation[0] + y * perturbation[1]).scale(2.0),
        (-y * perturbation[0] + x * perturbation[1]).scale(2.0),
    )
    z_acceleration = tuple(
        energy * (x, y)[k] * 0.5 + rho * at_perturbation[k] * 0.25
        for k in range(2)
    )
    a_w = (
        (x * wx - y * wy).scale(2.0),
        (y * wx + x * wy).scale(2.0),
    )
    energy_derivative = a_w[0] * perturbation[0] + a_w[1] * perturbation[1]
    rhs = (
        wx,
        wy,
        *z_acceleration,
        energy_derivative,
        *(rho * item for item in center_velocity),
        *(rho * item for item in center_acceleration),
        *(rho * item for item in third_velocity),
        *(rho * item for item in offset_acceleration),
        rho,
    )
    third_floor = min(
        _fraction_sqrt_float(
            Fraction.from_float(
                max(
                    0.0,
                    (d_first[0] * d_first[0] + d_first[1] * d_first[1]).value.lower,
                )
            ),
            upward=False,
        ),
        _fraction_sqrt_float(
            Fraction.from_float(
                max(
                    0.0,
                    (d_second[0] * d_second[0] + d_second[1] * d_second[1]).value.lower,
                )
            ),
            upward=False,
        ),
    )
    return tuple(rhs), float(third_floor)


def _planar_lc_direct_defect_and_lipschitz(
    solution: RegularizedBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    tube_radius: float,
) -> tuple[float, float, float]:
    polynomial_state = _planar_lc_state_intervals(solution, parameter_interval)
    derivative = _planar_lc_derivative_intervals(solution, parameter_interval)
    polynomial_rhs, _ = _planar_lc_dual_rhs(
        polynomial_state, np.asarray(solution.masses, dtype=float), solution.pair
    )
    defect = max(
        _interval_abs_sup(derivative[index] - polynomial_rhs[index].value)
        for index in range(14)
    )
    tube_state = _planar_lc_state_intervals(
        solution, parameter_interval, inflate=tube_radius
    )
    tube_rhs, third_floor = _planar_lc_dual_rhs(
        tube_state, np.asarray(solution.masses, dtype=float), solution.pair
    )
    lipschitz = max(
        _fraction_upper_float(
            sum(
                (
                    Fraction.from_float(_interval_abs_sup(entry))
                    for entry in component.derivative
                ),
                Fraction(0),
            )
        )
        for component in tube_rhs
    )
    return (
        float(np.nextafter(defect, np.inf)),
        float(np.nextafter(lipschitz, np.inf)),
        third_floor,
    )


def _fraction_upper_float(value: Fraction) -> float:
    """Smallest adjacent binary float we can cheaply prove is >= value."""

    candidate = float(value)
    if Fraction.from_float(candidate) < value:
        candidate = float(np.nextafter(candidate, np.inf))
    return candidate


def _fraction_lower_float(value: Fraction) -> float:
    candidate = float(value)
    if Fraction.from_float(candidate) > value:
        candidate = float(np.nextafter(candidate, -np.inf))
    return candidate


@lru_cache(maxsize=131072)
def _fraction_sqrt_float(value: Fraction, *, upward: bool) -> float:
    """Directed binary64 bound for sqrt of a nonnegative rational.

    Decimal division first encloses the exact rational in the requested input
    direction.  ``Decimal.sqrt`` itself is correctly rounded to nearest rather
    than by the context direction, so the adjacent context Decimal on the same
    side supplies a rigorous square-root endpoint before outward binary64
    conversion.
    """

    value = Fraction(value)
    if value < 0:
        raise ValueError("square-root argument must be nonnegative")
    if value == 0:
        return 0.0
    direction = ROUND_CEILING if upward else ROUND_FLOOR
    with localcontext() as context:
        context.prec = 100
        context.rounding = direction
        decimal_value = context.divide(
            Decimal(value.numerator), Decimal(value.denominator)
        )
        nearest_root = context.sqrt(decimal_value)
        decimal_endpoint = (
            context.next_plus(nearest_root)
            if upward
            else context.next_minus(nearest_root)
        )
    candidate = float(decimal_endpoint)
    candidate_decimal = Decimal.from_float(candidate)
    if upward and candidate_decimal < decimal_endpoint:
        candidate = float(np.nextafter(candidate, np.inf))
    elif not upward and candidate_decimal > decimal_endpoint:
        candidate = float(np.nextafter(candidate, -np.inf))
    return candidate


def _ordinary_tube_pair_floor_lower(
    nominal_floor: float,
    dimension: int,
    position_radius: float,
) -> float:
    sqrt_dimension_upper = _fraction_sqrt_float(
        Fraction(int(dimension)), upward=True
    )
    expansion_upper = (
        Fraction(2)
        * Fraction.from_float(sqrt_dimension_upper)
        * Fraction.from_float(float(position_radius))
    )
    return _fraction_lower_float(
        Fraction.from_float(float(nominal_floor)) - expansion_upper
    )


def _planar_lc_mass_ratio_arithmetic_exact(
    masses: tuple[float, ...] | np.ndarray,
    pair: tuple[int, int],
) -> bool:
    """Whether every binary64 mass ratio used by LC field/projection is exact."""

    values = np.asarray(masses, dtype=float)
    if values.shape != (3,) or np.any(~np.isfinite(values)) or np.any(values <= 0):
        return False
    if len(pair) != 2 or pair[0] == pair[1] or not set(pair).issubset({0, 1, 2}):
        return False
    first, second = pair
    third = ({0, 1, 2} - {first, second}).pop()
    first_q = Fraction.from_float(float(values[first]))
    second_q = Fraction.from_float(float(values[second]))
    third_q = Fraction.from_float(float(values[third]))
    pair_q = first_q + second_q
    pair_float = float(values[first] + values[second])
    if Fraction.from_float(pair_float) != pair_q:
        return False
    return bool(
        Fraction.from_float(float(values[second] / pair_float))
        == second_q / pair_q
        and Fraction.from_float(float(values[first] / pair_float))
        == first_q / pair_q
        and Fraction.from_float(float(values[third] / pair_float))
        == third_q / pair_q
    )


def _fraction_array_from_floats(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    out = np.empty(array.shape, dtype=object)
    for index in np.ndindex(array.shape):
        out[index] = Fraction.from_float(float(array[index]))
    return out


def _fraction_array_tube_intervals(
    values: np.ndarray,
    radius: float,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 2 or radius < 0.0 or not np.isfinite(radius):
        raise ValueError("finite matrix and nonnegative tube radius required")
    radius_q = Fraction.from_float(float(radius))
    return tuple(
        tuple(
            (
                _fraction_lower_float(array[row, axis] - radius_q),
                _fraction_upper_float(array[row, axis] + radius_q),
            )
            for axis in range(array.shape[1])
        )
        for row in range(array.shape[0])
    )


def _exact_rational_ordinary_state_at_physical_time(
    chart: OrdinaryTaylorChartCertificate,
    physical_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    physical = _exact_rational_interval_from_floats(chart.physical_time_interval)
    parameter = _exact_rational_interval_from_floats(chart.parameter_interval)
    if physical is None or parameter is None:
        raise ValueError("ordinary chart intervals must be finite")
    physical_width = physical[1] - physical[0]
    parameter_width = parameter[1] - parameter[0]
    time_q = Fraction.from_float(float(physical_time))
    if physical_width <= 0 or parameter_width <= 0:
        raise ValueError("ordinary chart intervals must have positive width")
    if not physical[0] <= time_q <= physical[1]:
        raise ValueError("target physical time lies outside ordinary chart")
    parameter_q = (
        parameter[0]
        + (time_q - physical[0]) * parameter_width / physical_width
    )
    return _exact_rational_chart_projected_state_at_parameter(chart, parameter_q)


def _positive_product_upper(left: float, right: float) -> float:
    if left < 0.0 or right < 0.0 or not np.isfinite(left + right):
        raise ValueError("positive finite factors required")
    return _fraction_upper_float(
        Fraction.from_float(float(left)) * Fraction.from_float(float(right))
    )


def _positive_ratio_upper(numerator: float, denominator: float) -> float:
    if (
        numerator < 0.0
        or denominator <= 0.0
        or not np.isfinite(numerator)
        or not np.isfinite(denominator)
    ):
        raise ValueError("nonnegative finite numerator and positive denominator required")
    return _fraction_upper_float(
        Fraction.from_float(float(numerator))
        / Fraction.from_float(float(denominator))
    )


def _newton_acceleration_lipschitz_upper(
    masses: np.ndarray,
    *,
    body_index: int,
    dimension: int,
    pair_distance_floor: float,
) -> float:
    """Directed upper bound for one acceleration block in infinity norm."""

    if pair_distance_floor <= 0.0:
        raise ValueError("positive pair-distance floor required")
    sqrt_dimension_upper = _fraction_sqrt_float(
        Fraction(int(dimension)), upward=True
    )
    derivative_factor_upper = _fraction_upper_float(
        Fraction(1) + 3 * Fraction.from_float(sqrt_dimension_upper)
    )
    mass_sum = sum(
        (
            Fraction.from_float(float(masses[j]))
            for j in range(3)
            if j != int(body_index)
        ),
        Fraction(0),
    )
    rho = Fraction.from_float(float(pair_distance_floor))
    bound = (
        2
        * mass_sum
        * Fraction.from_float(derivative_factor_upper)
        / (rho * rho * rho)
    )
    return _fraction_upper_float(bound)


def _exp_upper(value: float) -> float:
    """Rigorous high-precision Decimal enclosure of exp converted upward.

    Python's Decimal exponential is correctly rounded to nearest even when the
    context requests ceiling.  The next context Decimal above that rounded
    value therefore lies above the exact exponential; the final comparison and
    ``nextafter`` convert that proven endpoint outward to binary64.
    """

    if value < 0.0 or not np.isfinite(value):
        raise ValueError("finite nonnegative exponent required")
    with localcontext() as context:
        context.prec = 80
        decimal_value = Decimal.from_float(float(value))
        nearest_exponential = context.exp(decimal_value)
        decimal_upper = context.next_plus(nearest_exponential)
    candidate = float(decimal_upper)
    if Decimal.from_float(candidate) < decimal_upper:
        candidate = float(np.nextafter(candidate, np.inf))
    return candidate


def _gronwall_error_upper_fraction(
    initial_error: float,
    defect: float,
    lipschitz: float,
    horizon: Fraction,
) -> Fraction:
    """Compose the scalar Gronwall bound without binary64 intermediates.

    The three serialized bounds are interpreted as their exact binary64
    rationals.  Only the exponential itself is converted to binary64, through
    the rigorous upper endpoint returned by :func:`_exp_upper`; all remaining
    products, subtraction, division, and addition are exact ``Fraction``
    operations.
    """

    values = (initial_error, defect, lipschitz)
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("finite nonnegative Gronwall inputs required")
    horizon_q = Fraction(horizon)
    if horizon_q < 0:
        raise ValueError("nonnegative exact Gronwall horizon required")
    initial_q = Fraction.from_float(float(initial_error))
    defect_q = Fraction.from_float(float(defect))
    lipschitz_q = Fraction.from_float(float(lipschitz))
    if lipschitz_q == 0:
        return initial_q + defect_q * horizon_q
    exponent = _fraction_upper_float(lipschitz_q * horizon_q)
    exponential_q = Fraction.from_float(_exp_upper(exponent))
    return (
        exponential_q * initial_q
        + defect_q * (exponential_q - Fraction(1)) / lipschitz_q
    )


def check_validated_ordinary_ivp_chart(
    binding: InitialValueProblemBindingCertificate,
    tube: OrdinaryAposterioriTubeCertificate,
    chart: OrdinaryTaylorChartCertificate,
) -> ValidatedOrdinaryIVPChartCheckResult:
    """Prove a supplied ordinary polynomial encloses its bound Newtonian IVP.

    The conclusion is local to the chart's physical interval.  It follows from
    the recomputed defect and collision-free Lipschitz tube by the standard
    continuation/Gronwall argument; it does not imply multi-chart completeness.
    """

    chart_result = check_ordinary_taylor_chart(chart)
    try:
        q_coefficients = _coefficient_array(chart.position_coefficients)
        v_coefficients = _coefficient_array(chart.velocity_coefficients)
        masses = np.asarray(chart.masses, dtype=float)
        chart_serialization_admissible = bool(
            chart.chart_type == "ordinary_taylor"
            and bool(chart.chart_id)
            and q_coefficients.ndim == 3
            and q_coefficients.shape[0] >= 2
            and q_coefficients.shape[1] == 3
            and q_coefficients.shape[2] in (2, 3)
            and v_coefficients.shape == q_coefficients.shape
            and masses.shape == (3,)
            and np.all(np.isfinite(q_coefficients))
            and np.all(np.isfinite(v_coefficients))
            and np.all(np.isfinite(masses))
            and np.all(masses > 0.0)
            and np.isfinite(chart.parameter_interval[0])
            and np.isfinite(chart.parameter_interval[1])
            and chart.parameter_interval[0] < chart.parameter_interval[1]
            and np.isfinite(chart.physical_time_interval[0])
            and np.isfinite(chart.physical_time_interval[1])
            and chart.physical_time_interval[0] < chart.physical_time_interval[1]
        )
    except (TypeError, ValueError, IndexError):
        chart_serialization_admissible = False
    binding_result = check_initial_value_problem_binding(binding, (chart,))
    tube_result = check_ordinary_aposteriori_tube(tube, chart)
    identifiers_match = bool(
        binding.chart_id == chart.chart_id == tube.chart_id
    )
    anchor_matches = bool(
        np.isfinite(binding.chart_parameter)
        and np.isfinite(tube.anchor_parameter)
        and float(binding.chart_parameter) == float(tube.anchor_parameter)
    )
    actual_initial_error = max(
        float(binding_result.max_position_gap),
        float(binding_result.max_velocity_gap),
    )
    initial_error_covered = bool(
        np.isfinite(actual_initial_error)
        and np.isfinite(tube.initial_error_bound)
        and actual_initial_error <= float(tube.initial_error_bound)
    )
    obligations = (
        CertificateCheckObligation(
            "validated_ordinary_chart_serialization_admissible",
            chart_serialization_admissible,
            f"chart_id={chart.chart_id!r}",
        ),
        CertificateCheckObligation(
            "validated_ordinary_ivp_binding_checked",
            binding_result.certified,
            f"binding_id={binding.binding_id!r}",
        ),
        CertificateCheckObligation(
            "validated_ordinary_tube_checked",
            tube_result.certified,
            f"tube_id={tube.tube_id!r}",
        ),
        CertificateCheckObligation(
            "validated_ordinary_component_chart_ids_match",
            identifiers_match,
            (
                f"binding_chart={binding.chart_id!r}; tube_chart={tube.chart_id!r}; "
                f"chart={chart.chart_id!r}"
            ),
        ),
        CertificateCheckObligation(
            "validated_ordinary_anchor_parameter_matches_binding",
            anchor_matches,
            (
                f"binding_parameter={binding.chart_parameter}; "
                f"tube_anchor={tube.anchor_parameter}"
            ),
        ),
        CertificateCheckObligation(
            "validated_ordinary_actual_initial_error_covered",
            initial_error_covered,
            (
                f"actual_initial_error={actual_initial_error}; "
                f"tube_initial_error={tube.initial_error_bound}"
            ),
        ),
    )
    return ValidatedOrdinaryIVPChartCheckResult(
        binding_result=binding_result,
        tube_result=tube_result,
        chart_result=chart_result,
        chart_serialization_admissible=chart_serialization_admissible,
        checker_id="validated_ordinary_ivp_chart_checker_v1",
        obligations=obligations,
    )


def check_ordinary_enclosure_transition(
    certificate: OrdinaryEnclosureTransitionCertificate,
    source_chart: OrdinaryTaylorChartCertificate,
    target_chart: OrdinaryTaylorChartCertificate,
    source_validation: ValidatedOrdinaryIVPChartCheckResult
    | OrdinaryEnclosureTransitionCheckResult,
    target_tube: OrdinaryAposterioriTubeCertificate,
) -> OrdinaryEnclosureTransitionCheckResult:
    """Validate that the target tube continues the exact source IVP branch."""

    if type(source_validation) not in {
        ValidatedOrdinaryIVPChartCheckResult,
        OrdinaryEnclosureTransitionCheckResult,
    }:
        raise TypeError(
            "source_validation must be an exact validated ordinary checker result"
        )
    target_chart_result = check_ordinary_taylor_chart(target_chart)
    target_tube_result = check_ordinary_aposteriori_tube(target_tube, target_chart)
    source_parameter = float(certificate.source_parameter)
    target_parameter = float(certificate.target_parameter)
    handoff_time = float(certificate.handoff_time)
    max_time_gap = float(certificate.max_time_gap)
    source_tube_result = (
        source_validation.tube_result
        if type(source_validation) is ValidatedOrdinaryIVPChartCheckResult
        else source_validation.target_tube_result
    )
    source_result_chart_id = (
        source_validation.binding_result.chart_id
        if type(source_validation) is ValidatedOrdinaryIVPChartCheckResult
        else source_validation.target_tube_result.chart_id
    )
    identifiers_match = bool(
        certificate.transition_id
        and certificate.source_chart_id == source_chart.chart_id
        and certificate.target_chart_id == target_chart.chart_id
        and source_result_chart_id == source_chart.chart_id
        and source_tube_result.chart_id == source_chart.chart_id
        and target_tube.chart_id == target_chart.chart_id
    )
    source_certified = bool(source_validation.certified)
    common_problem = bool(
        tuple(float(value) for value in source_chart.masses)
        == tuple(float(value) for value in target_chart.masses)
        and source_chart.dimension == target_chart.dimension
    )
    parameters_inside = bool(
        _parameter_in_interval(source_chart.parameter_interval, source_parameter)
        and _parameter_in_interval(target_chart.parameter_interval, target_parameter)
        and float(target_tube.anchor_parameter) == target_parameter
    )
    finite_time_cap = bool(np.isfinite(max_time_gap) and max_time_gap >= 0.0)

    time_gap = np.inf
    polynomial_gap = np.inf
    required_initial_error = np.inf
    time_matches = False
    target_error_covers_handoff = False
    if (
        identifiers_match
        and source_certified
        and common_problem
        and parameters_inside
        and finite_time_cap
        and target_chart_result.certified
        and target_tube_result.certified
    ):
        try:
            source_parameter_q = Fraction.from_float(source_parameter)
            target_parameter_q = Fraction.from_float(target_parameter)
            handoff_time_q = Fraction.from_float(handoff_time)
            source_time_q = _exact_rational_chart_physical_time_at_parameter(
                source_chart, source_parameter_q
            )
            target_time_q = _exact_rational_chart_physical_time_at_parameter(
                target_chart, target_parameter_q
            )
            time_gap_q = max(
                abs(source_time_q - handoff_time_q),
                abs(target_time_q - handoff_time_q),
                abs(source_time_q - target_time_q),
            )
            time_gap = _fraction_upper_float(time_gap_q)
            time_matches = time_gap_q <= Fraction.from_float(max_time_gap)
            source_q, source_v = _exact_rational_chart_projected_state_at_parameter(
                source_chart, source_parameter_q
            )
            target_q, target_v = _exact_rational_chart_projected_state_at_parameter(
                target_chart, target_parameter_q
            )
            polynomial_gap_q = max(
                _fraction_array_max_abs_difference(source_q, target_q),
                _fraction_array_max_abs_difference(source_v, target_v),
            )
            polynomial_gap = _fraction_upper_float(polynomial_gap_q)
            required_initial_error_q = (
                Fraction.from_float(source_tube_result.gronwall_error_bound)
                + polynomial_gap_q
            )
            required_initial_error = _fraction_upper_float(
                required_initial_error_q
            )
            target_error_covers_handoff = bool(
                required_initial_error_q
                <= Fraction.from_float(float(target_tube.initial_error_bound))
            )
        except (FloatingPointError, ValueError):
            pass

    obligations = (
        CertificateCheckObligation(
            "ordinary_enclosure_transition_identifiers_match",
            identifiers_match,
            (
                f"source={certificate.source_chart_id!r}; "
                f"target={certificate.target_chart_id!r}"
            ),
        ),
        CertificateCheckObligation(
            "ordinary_enclosure_transition_source_ivp_certified",
            source_certified,
            f"source_chart={source_chart.chart_id!r}",
        ),
        CertificateCheckObligation(
            "ordinary_enclosure_transition_common_problem",
            common_problem,
            "source and target masses/dimension agree exactly",
        ),
        CertificateCheckObligation(
            "ordinary_enclosure_transition_parameters_inside",
            parameters_inside,
            (
                f"source_parameter={source_parameter}; "
                f"target_parameter={target_parameter}"
            ),
        ),
        CertificateCheckObligation(
            "ordinary_enclosure_transition_target_chart_checked",
            target_chart_result.certified,
            f"target_chart={target_chart.chart_id!r}",
        ),
        CertificateCheckObligation(
            "ordinary_enclosure_transition_target_tube_checked",
            target_tube_result.certified,
            f"target_tube={target_tube.tube_id!r}",
        ),
        CertificateCheckObligation(
            "ordinary_enclosure_transition_time_matches",
            time_matches,
            f"time_gap={time_gap}; cap={max_time_gap}",
        ),
        CertificateCheckObligation(
            "ordinary_enclosure_transition_target_error_covers_handoff",
            target_error_covers_handoff,
            (
                f"required={required_initial_error}; "
                f"available={target_tube.initial_error_bound}"
            ),
        ),
    )
    return OrdinaryEnclosureTransitionCheckResult(
        transition_id=str(certificate.transition_id),
        checker_id="ordinary_enclosure_transition_checker_v1",
        target_chart_result=target_chart_result,
        target_tube_result=target_tube_result,
        obligations=obligations,
        polynomial_state_gap=float(polynomial_gap),
        required_target_initial_error=float(required_initial_error),
        time_gap=float(time_gap),
    )


def check_validated_ordinary_ivp_chain(
    certificate: ValidatedOrdinaryIVPChainCertificate,
    binding: InitialValueProblemBindingCertificate,
    charts: Iterable[OrdinaryTaylorChartCertificate],
    tubes: Iterable[OrdinaryAposterioriTubeCertificate],
    transitions: Iterable[OrdinaryEnclosureTransitionCertificate],
) -> ValidatedOrdinaryIVPChainCheckResult:
    """Check a finite forward chain enclosing one exact Newtonian IVP branch."""

    chart_tuple = tuple(charts)
    tube_tuple = tuple(tubes)
    transition_tuple = tuple(transitions)
    if not chart_tuple or not tube_tuple:
        raise ValueError("validated ordinary chain requires at least one chart and tube")

    count_matches = bool(
        len(tube_tuple) == len(chart_tuple)
        and len(transition_tuple) == len(chart_tuple) - 1
    )
    first_result = check_validated_ordinary_ivp_chart(
        binding,
        tube_tuple[0],
        chart_tuple[0],
    )
    transition_results: list[OrdinaryEnclosureTransitionCheckResult] = []
    source_result: ValidatedOrdinaryIVPChartCheckResult | OrdinaryEnclosureTransitionCheckResult = first_result
    if count_matches:
        for index, transition in enumerate(transition_tuple):
            result = check_ordinary_enclosure_transition(
                transition,
                chart_tuple[index],
                chart_tuple[index + 1],
                source_result,
                tube_tuple[index + 1],
            )
            transition_results.append(result)
            source_result = result

    chart_ids = tuple(str(chart.chart_id) for chart in chart_tuple)
    transition_ids = tuple(
        str(transition.transition_id) for transition in transition_tuple
    )
    identities_match = bool(
        certificate.chain_id
        and tuple(certificate.chart_ids) == chart_ids
        and tuple(certificate.transition_ids) == transition_ids
        and _nonempty_ids_unique(chart_ids)
        and _nonempty_ids_unique(transition_ids)
    )
    orientation_supported = certificate.orientation == "forward"
    physical_intervals = tuple(
        tuple(float(value) for value in chart.physical_time_interval)
        for chart in chart_tuple
    )
    intervals_finite = bool(
        all(_finite_nonempty_interval(interval) for interval in physical_intervals)
    )
    oriented_frontier = bool(
        intervals_finite
        and all(
            physical_intervals[index][0] <= physical_intervals[index + 1][0]
            and physical_intervals[index + 1][0] <= physical_intervals[index][1]
            and physical_intervals[index + 1][1] >= physical_intervals[index][1]
            for index in range(len(physical_intervals) - 1)
        )
    )
    if intervals_finite and oriented_frontier:
        covered_interval = (
            float(binding.initial_time),
            max(interval[1] for interval in physical_intervals),
        )
    else:
        covered_interval = (np.inf, -np.inf)
    target_interval = tuple(
        float(value) for value in certificate.target_physical_time_interval
    )
    target_covered = bool(
        _finite_nonempty_interval(target_interval)
        and np.isfinite(binding.initial_time)
        and target_interval[0] == float(binding.initial_time)
        and _finite_nonempty_interval(covered_interval)
        and _interval_contains_interval(covered_interval, target_interval)
    )
    initial_time_in_first = bool(
        intervals_finite
        and physical_intervals[0][0]
        <= float(binding.initial_time)
        <= physical_intervals[0][1]
    )
    all_handoffs_certified = bool(
        count_matches
        and len(transition_results) == len(transition_tuple)
        and all(result.certified for result in transition_results)
    )
    target_time = target_interval[1] if _finite_nonempty_interval(target_interval) else np.inf
    target_positions: tuple[tuple[tuple[float, float], ...], ...] = ()
    target_velocities: tuple[tuple[tuple[float, float], ...], ...] = ()
    target_state_computed = False
    if target_covered and first_result.certified and all_handoffs_certified:
        validation_results: tuple[
            ValidatedOrdinaryIVPChartCheckResult
            | OrdinaryEnclosureTransitionCheckResult,
            ...,
        ] = (first_result, *transition_results)
        target_chart_index = next(
            (
                index
                for index in range(len(chart_tuple) - 1, -1, -1)
                if physical_intervals[index][0]
                <= target_time
                <= physical_intervals[index][1]
            ),
            None,
        )
        if target_chart_index is not None:
            try:
                target_q, target_v = _exact_rational_ordinary_state_at_physical_time(
                    chart_tuple[target_chart_index],
                    target_time,
                )
                validation = validation_results[target_chart_index]
                radius = (
                    validation.tube_result.gronwall_error_bound
                    if type(validation) is ValidatedOrdinaryIVPChartCheckResult
                    else validation.target_tube_result.gronwall_error_bound
                )
                target_positions = _fraction_array_tube_intervals(target_q, radius)
                target_velocities = _fraction_array_tube_intervals(target_v, radius)
                target_state_computed = bool(target_positions and target_velocities)
            except (OverflowError, ValueError, ZeroDivisionError):
                target_state_computed = False

    obligations = (
        CertificateCheckObligation(
            "validated_ordinary_chain_identity_and_order",
            identities_match,
            f"chart_ids={chart_ids!r}; transition_ids={transition_ids!r}",
        ),
        CertificateCheckObligation(
            "validated_ordinary_chain_component_count_matches",
            count_matches,
            (
                f"charts={len(chart_tuple)}; tubes={len(tube_tuple)}; "
                f"transitions={len(transition_tuple)}"
            ),
        ),
        CertificateCheckObligation(
            "validated_ordinary_chain_forward_orientation",
            orientation_supported,
            f"orientation={certificate.orientation!r}",
        ),
        CertificateCheckObligation(
            "validated_ordinary_chain_first_ivp_chart_certified",
            first_result.certified,
            f"chart_id={chart_tuple[0].chart_id!r}",
        ),
        CertificateCheckObligation(
            "validated_ordinary_chain_handoffs_certified",
            all_handoffs_certified,
            f"checked_handoffs={len(transition_results)}",
        ),
        CertificateCheckObligation(
            "validated_ordinary_chain_initial_time_in_first_chart",
            initial_time_in_first,
            f"initial_time={binding.initial_time}; first={physical_intervals[0]!r}",
        ),
        CertificateCheckObligation(
            "validated_ordinary_chain_oriented_frontier_no_gaps",
            oriented_frontier,
            f"physical_intervals={physical_intervals!r}",
        ),
        CertificateCheckObligation(
            "validated_ordinary_chain_target_interval_covered",
            target_covered,
            f"covered={covered_interval!r}; target={target_interval!r}",
        ),
        CertificateCheckObligation(
            "validated_ordinary_chain_target_state_enclosure_computed",
            target_state_computed,
            f"target_time={target_time}",
        ),
    )
    return ValidatedOrdinaryIVPChainCheckResult(
        chain_id=str(certificate.chain_id),
        checker_id="validated_ordinary_ivp_chain_checker_v1",
        first_chart_result=first_result,
        transition_results=tuple(transition_results),
        obligations=obligations,
        covered_physical_time_interval=covered_interval,
        target_time=float(target_time),
        target_position_intervals=target_positions,
        target_velocity_intervals=target_velocities,
    )


def _serialized_chart_problem_identity(
    chart: object,
) -> tuple[tuple[float, float, float], int] | None:
    """Return the exact serialized three-body problem identity for one chart."""

    masses = getattr(chart, "masses", None)
    if not (
        type(masses) is tuple
        and len(masses) == 3
        and all(
            type(mass) is float and np.isfinite(mass) and mass > 0.0
            for mass in masses
        )
    ):
        return None

    if isinstance(chart, OrdinaryTaylorChartCertificate):
        dimension = chart.dimension
    elif isinstance(chart, PlanarLeviCivitaBinaryChartCertificate):
        dimension = 2
    elif isinstance(chart, SpatialKSBinaryChartCertificate):
        dimension = 3
    elif isinstance(
        chart,
        (
            TotalCollisionFuchsianStopChartCertificate,
            TotalCollisionGeneralizedFuchsianStopChartCertificate,
        ),
    ):
        dimension = chart.dimension
    else:
        return None

    if type(dimension) is not int or dimension <= 0:
        return None
    return masses, dimension


def check_ordinary_chart_transition(
    transition: OrdinaryChartTransitionCertificate,
    charts: Iterable[OrdinaryTaylorChartCertificate],
) -> TransitionCheckResult:
    """Verify an ordinary chart handoff from explicit serialized charts."""

    chart_by_id = {chart.chart_id: chart for chart in charts}
    source_chart = chart_by_id.get(transition.source_chart_id)
    target_chart = chart_by_id.get(transition.target_chart_id)
    handoff_time = float(transition.handoff_time)
    position_tolerance = float(transition.position_tolerance)
    velocity_tolerance = float(transition.velocity_tolerance)

    identity_ok = bool(
        transition.transition_id
        and transition.source_chart_id
        and transition.target_chart_id
        and transition.transition_type
    )
    endpoint_charts_present = source_chart is not None and target_chart is not None
    finite_tolerances = bool(
        np.isfinite(position_tolerance)
        and position_tolerance >= 0.0
        and np.isfinite(velocity_tolerance)
        and velocity_tolerance >= 0.0
        and np.isfinite(handoff_time)
    )
    transition_type_supported = bool(
        transition.transition_type == "ordinary_overlap_handoff"
    )
    handoff_inside = bool(
        endpoint_charts_present
        and _physical_time_in_chart(source_chart, handoff_time)
        and _physical_time_in_chart(target_chart, handoff_time)
    )
    chart_types_supported = bool(
        endpoint_charts_present
        and source_chart.chart_type == "ordinary_taylor"
        and target_chart.chart_type == "ordinary_taylor"
    )
    source_problem_identity = (
        _serialized_chart_problem_identity(source_chart)
        if source_chart is not None
        else None
    )
    target_problem_identity = (
        _serialized_chart_problem_identity(target_chart)
        if target_chart is not None
        else None
    )
    common_problem_identity = bool(
        endpoint_charts_present
        and source_problem_identity is not None
        and source_problem_identity == target_problem_identity
    )

    max_position_gap = np.inf
    max_velocity_gap = np.inf
    continuity_certified = False
    exact_rational_continuity_certified = False
    exact_rational_continuity_detail = "transition preconditions not certified"
    if (
        endpoint_charts_present
        and chart_types_supported
        and common_problem_identity
        and transition_type_supported
        and finite_tolerances
        and handoff_inside
    ):
        try:
            source_parameter = _physical_to_parameter(source_chart, handoff_time)
            target_parameter = _physical_to_parameter(target_chart, handoff_time)
            source_q = _evaluate_coefficients(
                _coefficient_array(source_chart.position_coefficients),
                source_parameter,
            )
            target_q = _evaluate_coefficients(
                _coefficient_array(target_chart.position_coefficients),
                target_parameter,
            )
            source_v = _evaluate_coefficients(
                _coefficient_array(source_chart.velocity_coefficients),
                source_parameter,
            )
            target_v = _evaluate_coefficients(
                _coefficient_array(target_chart.velocity_coefficients),
                target_parameter,
            )
            max_position_gap = float(np.max(np.abs(source_q - target_q)))
            max_velocity_gap = float(np.max(np.abs(source_v - target_v)))
            continuity_certified = bool(
                max_position_gap <= position_tolerance
                and max_velocity_gap <= velocity_tolerance
            )
            (
                exact_rational_continuity_certified,
                exact_rational_continuity_detail,
            ) = _exact_rational_transition_state_continuity(
                source_chart,
                target_chart,
                source_parameter=source_parameter,
                target_parameter=target_parameter,
                handoff_time=handoff_time,
                position_tolerance=position_tolerance,
                velocity_tolerance=velocity_tolerance,
                physical_time_tolerance=0.0,
            )
        except (FloatingPointError, ValueError):
            max_position_gap = np.inf
            max_velocity_gap = np.inf

    obligations = (
        CertificateCheckObligation(
            "transition_identity_present",
            identity_ok,
            (
                f"transition_id={transition.transition_id!r}; "
                f"source={transition.source_chart_id!r}; "
                f"target={transition.target_chart_id!r}; "
                f"type={transition.transition_type!r}"
            ),
        ),
        CertificateCheckObligation(
            "transition_endpoint_charts_present",
            endpoint_charts_present,
            f"known_chart_ids={tuple(chart_by_id)}",
        ),
        CertificateCheckObligation(
            "transition_chart_types_supported",
            chart_types_supported,
            "ordinary transition checker currently supports ordinary_taylor endpoints",
        ),
        CertificateCheckObligation(
            "transition_common_problem_identity",
            common_problem_identity,
            (
                f"source_problem_identity={source_problem_identity!r}; "
                f"target_problem_identity={target_problem_identity!r}"
            ),
        ),
        CertificateCheckObligation(
            "transition_type_supported",
            transition_type_supported,
            "supported transition_type is ordinary_overlap_handoff",
        ),
        CertificateCheckObligation(
            "transition_handoff_time_inside_charts",
            handoff_inside,
            f"handoff_time={handoff_time}",
        ),
        CertificateCheckObligation(
            "transition_tolerances_finite",
            finite_tolerances,
            (
                f"position_tolerance={position_tolerance}; "
                f"velocity_tolerance={velocity_tolerance}"
            ),
        ),
        CertificateCheckObligation(
            "transition_state_continuity",
            continuity_certified,
            (
                f"max_position_gap={max_position_gap}; "
                f"max_velocity_gap={max_velocity_gap}"
            ),
        ),
        CertificateCheckObligation(
            "transition_exact_rational_state_continuity",
            exact_rational_continuity_certified,
            exact_rational_continuity_detail,
        ),
    )
    return TransitionCheckResult(
        transition_id=str(transition.transition_id),
        transition_type=str(transition.transition_type),
        checker_id="independent_ordinary_transition_checker_v1",
        obligations=obligations,
        max_position_gap=float(max_position_gap),
        max_velocity_gap=float(max_velocity_gap),
    )


def check_planar_levi_civita_transition(
    transition: PlanarLeviCivitaTransitionCertificate,
    charts: Iterable[
        OrdinaryTaylorChartCertificate | PlanarLeviCivitaBinaryChartCertificate
    ],
) -> TransitionCheckResult:
    """Verify an ordinary/LC binary handoff from explicit serialized charts."""

    chart_by_id = {chart.chart_id: chart for chart in charts}
    source_chart = chart_by_id.get(transition.source_chart_id)
    target_chart = chart_by_id.get(transition.target_chart_id)
    handoff_time = float(transition.handoff_time)
    source_parameter = float(transition.source_parameter)
    target_parameter = float(transition.target_parameter)
    position_tolerance = float(transition.position_tolerance)
    velocity_tolerance = float(transition.velocity_tolerance)
    physical_time_tolerance = float(transition.physical_time_tolerance)

    identity_ok = bool(
        transition.transition_id
        and transition.source_chart_id
        and transition.target_chart_id
        and transition.transition_type
    )
    endpoint_charts_present = source_chart is not None and target_chart is not None
    finite_tolerances = bool(
        np.isfinite(position_tolerance)
        and position_tolerance >= 0.0
        and np.isfinite(velocity_tolerance)
        and velocity_tolerance >= 0.0
        and np.isfinite(physical_time_tolerance)
        and physical_time_tolerance >= 0.0
        and np.isfinite(handoff_time)
        and np.isfinite(source_parameter)
        and np.isfinite(target_parameter)
    )
    source_is_ordinary = isinstance(source_chart, OrdinaryTaylorChartCertificate)
    target_is_ordinary = isinstance(target_chart, OrdinaryTaylorChartCertificate)
    source_is_lc = isinstance(source_chart, PlanarLeviCivitaBinaryChartCertificate)
    target_is_lc = isinstance(target_chart, PlanarLeviCivitaBinaryChartCertificate)
    ordinary_to_lc = source_is_ordinary and target_is_lc
    lc_to_ordinary = source_is_lc and target_is_ordinary
    endpoint_types_supported = bool(ordinary_to_lc or lc_to_ordinary)
    transition_type_supported = bool(
        (
            ordinary_to_lc
            and transition.transition_type
            in {
                "ordinary_to_binary_event_handoff",
                "ordinary_to_planar_lc_binary_event_handoff",
            }
        )
        or (
            lc_to_ordinary
            and transition.transition_type
            in {
                "binary_to_ordinary_event_handoff",
                "planar_lc_binary_to_ordinary_event_handoff",
            }
        )
    )
    parameters_inside = bool(
        endpoint_charts_present
        and _parameter_in_interval(source_chart.parameter_interval, source_parameter)
        and _parameter_in_interval(target_chart.parameter_interval, target_parameter)
    )
    handoff_inside = bool(
        endpoint_charts_present
        and _time_in_interval(source_chart.physical_time_interval, handoff_time)
        and _time_in_interval(target_chart.physical_time_interval, handoff_time)
    )
    source_problem_identity = (
        _serialized_chart_problem_identity(source_chart)
        if source_chart is not None
        else None
    )
    target_problem_identity = (
        _serialized_chart_problem_identity(target_chart)
        if target_chart is not None
        else None
    )
    common_problem_identity = bool(
        endpoint_charts_present
        and source_problem_identity is not None
        and source_problem_identity == target_problem_identity
    )

    max_physical_time_gap = np.inf
    physical_time_match_certified = False
    max_position_gap = np.inf
    max_velocity_gap = np.inf
    continuity_certified = False
    exact_rational_continuity_certified = False
    exact_rational_continuity_detail = "transition preconditions not certified"
    if (
        endpoint_charts_present
        and endpoint_types_supported
        and common_problem_identity
        and transition_type_supported
        and finite_tolerances
        and parameters_inside
        and handoff_inside
    ):
        try:
            source_time = _chart_physical_time_at_parameter(source_chart, source_parameter)
            target_time = _chart_physical_time_at_parameter(target_chart, target_parameter)
            max_physical_time_gap = max(
                abs(source_time - handoff_time),
                abs(target_time - handoff_time),
            )
            physical_time_match_certified = bool(
                max_physical_time_gap <= physical_time_tolerance
            )
            source_q, source_v = _chart_projected_state_at_parameter(
                source_chart,
                source_parameter,
            )
            target_q, target_v = _chart_projected_state_at_parameter(
                target_chart,
                target_parameter,
            )
            max_position_gap = float(np.max(np.abs(source_q - target_q)))
            max_velocity_gap = float(np.max(np.abs(source_v - target_v)))
            continuity_certified = bool(
                physical_time_match_certified
                and max_position_gap <= position_tolerance
                and max_velocity_gap <= velocity_tolerance
            )
            (
                exact_rational_continuity_certified,
                exact_rational_continuity_detail,
            ) = _exact_rational_transition_state_continuity(
                source_chart,
                target_chart,
                source_parameter=source_parameter,
                target_parameter=target_parameter,
                handoff_time=handoff_time,
                position_tolerance=position_tolerance,
                velocity_tolerance=velocity_tolerance,
                physical_time_tolerance=physical_time_tolerance,
            )
        except (FloatingPointError, ValueError):
            max_physical_time_gap = np.inf
            max_position_gap = np.inf
            max_velocity_gap = np.inf

    obligations = (
        CertificateCheckObligation(
            "transition_identity_present",
            identity_ok,
            (
                f"transition_id={transition.transition_id!r}; "
                f"source={transition.source_chart_id!r}; "
                f"target={transition.target_chart_id!r}; "
                f"type={transition.transition_type!r}"
            ),
        ),
        CertificateCheckObligation(
            "transition_endpoint_charts_present",
            endpoint_charts_present,
            f"known_chart_ids={tuple(chart_by_id)}",
        ),
        CertificateCheckObligation(
            "transition_chart_types_supported",
            endpoint_types_supported,
            "LC transition checker supports ordinary_taylor <-> planar_levi_civita_binary",
        ),
        CertificateCheckObligation(
            "transition_common_problem_identity",
            common_problem_identity,
            (
                f"source_problem_identity={source_problem_identity!r}; "
                f"target_problem_identity={target_problem_identity!r}"
            ),
        ),
        CertificateCheckObligation(
            "transition_type_supported",
            transition_type_supported,
            "transition type must match ordinary->LC or LC->ordinary direction",
        ),
        CertificateCheckObligation(
            "transition_parameters_inside_charts",
            parameters_inside,
            (
                f"source_parameter={source_parameter}; "
                f"target_parameter={target_parameter}"
            ),
        ),
        CertificateCheckObligation(
            "transition_handoff_time_inside_charts",
            handoff_inside,
            f"handoff_time={handoff_time}",
        ),
        CertificateCheckObligation(
            "transition_tolerances_finite",
            finite_tolerances,
            (
                f"position_tolerance={position_tolerance}; "
                f"velocity_tolerance={velocity_tolerance}; "
                f"physical_time_tolerance={physical_time_tolerance}"
            ),
        ),
        CertificateCheckObligation(
            "transition_physical_time_match",
            physical_time_match_certified,
            f"max_physical_time_gap={max_physical_time_gap}",
        ),
        CertificateCheckObligation(
            "transition_state_continuity",
            continuity_certified,
            (
                f"max_position_gap={max_position_gap}; "
                f"max_velocity_gap={max_velocity_gap}"
            ),
        ),
        CertificateCheckObligation(
            "transition_exact_rational_state_continuity",
            exact_rational_continuity_certified,
            exact_rational_continuity_detail,
        ),
    )
    return TransitionCheckResult(
        transition_id=str(transition.transition_id),
        transition_type=str(transition.transition_type),
        checker_id="independent_planar_lc_transition_checker_v1",
        obligations=obligations,
        max_position_gap=float(max_position_gap),
        max_velocity_gap=float(max_velocity_gap),
    )


def check_spatial_ks_transition(
    transition: SpatialKSTransitionCertificate,
    charts: Iterable[
        OrdinaryTaylorChartCertificate
        | PlanarLeviCivitaBinaryChartCertificate
        | SpatialKSBinaryChartCertificate
    ],
) -> TransitionCheckResult:
    """Verify an ordinary/KS binary handoff from explicit serialized charts."""

    chart_by_id = {chart.chart_id: chart for chart in charts}
    source_chart = chart_by_id.get(transition.source_chart_id)
    target_chart = chart_by_id.get(transition.target_chart_id)
    handoff_time = float(transition.handoff_time)
    source_parameter = float(transition.source_parameter)
    target_parameter = float(transition.target_parameter)
    position_tolerance = float(transition.position_tolerance)
    velocity_tolerance = float(transition.velocity_tolerance)
    physical_time_tolerance = float(transition.physical_time_tolerance)

    identity_ok = bool(
        transition.transition_id
        and transition.source_chart_id
        and transition.target_chart_id
        and transition.transition_type
    )
    endpoint_charts_present = source_chart is not None and target_chart is not None
    finite_tolerances = bool(
        np.isfinite(position_tolerance)
        and position_tolerance >= 0.0
        and np.isfinite(velocity_tolerance)
        and velocity_tolerance >= 0.0
        and np.isfinite(physical_time_tolerance)
        and physical_time_tolerance >= 0.0
        and np.isfinite(handoff_time)
        and np.isfinite(source_parameter)
        and np.isfinite(target_parameter)
    )
    source_is_ordinary = isinstance(source_chart, OrdinaryTaylorChartCertificate)
    target_is_ordinary = isinstance(target_chart, OrdinaryTaylorChartCertificate)
    source_is_ks = isinstance(source_chart, SpatialKSBinaryChartCertificate)
    target_is_ks = isinstance(target_chart, SpatialKSBinaryChartCertificate)
    ordinary_to_ks = source_is_ordinary and target_is_ks
    ks_to_ordinary = source_is_ks and target_is_ordinary
    ks_to_ks = source_is_ks and target_is_ks
    endpoint_types_supported = bool(ordinary_to_ks or ks_to_ordinary or ks_to_ks)
    transition_type_supported = bool(
        (
            ordinary_to_ks
            and transition.transition_type
            in {
                "spatial_ordinary_to_ks_decreasing_distance_entry",
                "ordinary_to_spatial_ks_binary_event_handoff",
            }
        )
        or (
            ks_to_ordinary
            and transition.transition_type
            in {
                "spatial_ks_to_ordinary_rho_positive_endpoint_projection",
                "spatial_ks_exit_event_to_ordinary_rho_positive_projection",
                "spatial_ks_to_ordinary_safe_handoff",
            }
        )
        or (
            ks_to_ks
            and transition.transition_type
            in {
                "spatial_ks_to_ks_competing_binary_entry",
            }
        )
    )
    parameters_inside = bool(
        endpoint_charts_present
        and _parameter_in_interval(source_chart.parameter_interval, source_parameter)
        and _parameter_in_interval(target_chart.parameter_interval, target_parameter)
    )
    handoff_inside = bool(
        endpoint_charts_present
        and _time_in_interval(source_chart.physical_time_interval, handoff_time)
        and _time_in_interval(target_chart.physical_time_interval, handoff_time)
    )

    max_physical_time_gap = np.inf
    physical_time_match_certified = False
    max_position_gap = np.inf
    max_velocity_gap = np.inf
    continuity_certified = False
    exact_rational_continuity_certified = False
    exact_rational_continuity_detail = "transition preconditions not certified"
    if (
        endpoint_charts_present
        and endpoint_types_supported
        and transition_type_supported
        and finite_tolerances
        and parameters_inside
        and handoff_inside
    ):
        try:
            source_time = _chart_physical_time_at_parameter(source_chart, source_parameter)
            target_time = _chart_physical_time_at_parameter(target_chart, target_parameter)
            max_physical_time_gap = max(
                abs(source_time - handoff_time),
                abs(target_time - handoff_time),
            )
            physical_time_match_certified = bool(
                max_physical_time_gap <= physical_time_tolerance
            )
            source_q, source_v = _chart_projected_state_at_parameter(
                source_chart,
                source_parameter,
            )
            target_q, target_v = _chart_projected_state_at_parameter(
                target_chart,
                target_parameter,
            )
            max_position_gap = float(np.max(np.abs(source_q - target_q)))
            max_velocity_gap = float(np.max(np.abs(source_v - target_v)))
            continuity_certified = bool(
                physical_time_match_certified
                and max_position_gap <= position_tolerance
                and max_velocity_gap <= velocity_tolerance
            )
            (
                exact_rational_continuity_certified,
                exact_rational_continuity_detail,
            ) = _exact_rational_transition_state_continuity(
                source_chart,
                target_chart,
                source_parameter=source_parameter,
                target_parameter=target_parameter,
                handoff_time=handoff_time,
                position_tolerance=position_tolerance,
                velocity_tolerance=velocity_tolerance,
                physical_time_tolerance=physical_time_tolerance,
            )
        except (FloatingPointError, ValueError):
            max_physical_time_gap = np.inf
            max_position_gap = np.inf
            max_velocity_gap = np.inf

    obligations = (
        CertificateCheckObligation(
            "transition_identity_present",
            identity_ok,
            (
                f"transition_id={transition.transition_id!r}; "
                f"source={transition.source_chart_id!r}; "
                f"target={transition.target_chart_id!r}; "
                f"type={transition.transition_type!r}"
            ),
        ),
        CertificateCheckObligation(
            "transition_endpoint_charts_present",
            endpoint_charts_present,
            f"known_chart_ids={tuple(chart_by_id)}",
        ),
        CertificateCheckObligation(
            "transition_chart_types_supported",
            endpoint_types_supported,
            "KS transition checker supports ordinary_taylor <-> spatial_ks_binary and spatial_ks_binary -> spatial_ks_binary",
        ),
        CertificateCheckObligation(
            "transition_type_supported",
            transition_type_supported,
            "transition type must match ordinary->KS, KS->ordinary, or KS->KS direction",
        ),
        CertificateCheckObligation(
            "transition_parameters_inside_charts",
            parameters_inside,
            (
                f"source_parameter={source_parameter}; "
                f"target_parameter={target_parameter}"
            ),
        ),
        CertificateCheckObligation(
            "transition_handoff_time_inside_charts",
            handoff_inside,
            f"handoff_time={handoff_time}",
        ),
        CertificateCheckObligation(
            "transition_tolerances_finite",
            finite_tolerances,
            (
                f"position_tolerance={position_tolerance}; "
                f"velocity_tolerance={velocity_tolerance}; "
                f"physical_time_tolerance={physical_time_tolerance}"
            ),
        ),
        CertificateCheckObligation(
            "transition_physical_time_match",
            physical_time_match_certified,
            f"max_physical_time_gap={max_physical_time_gap}",
        ),
        CertificateCheckObligation(
            "transition_state_continuity",
            continuity_certified,
            (
                f"max_position_gap={max_position_gap}; "
                f"max_velocity_gap={max_velocity_gap}"
            ),
        ),
        CertificateCheckObligation(
            "transition_exact_rational_state_continuity",
            exact_rational_continuity_certified,
            exact_rational_continuity_detail,
        ),
    )
    return TransitionCheckResult(
        transition_id=str(transition.transition_id),
        transition_type=str(transition.transition_type),
        checker_id="independent_spatial_ks_transition_checker_v1",
        obligations=obligations,
        max_position_gap=float(max_position_gap),
        max_velocity_gap=float(max_velocity_gap),
    )


def check_event_isolation(
    certificate: EventIsolationCertificate,
) -> EventIsolationCheckResult:
    """Verify a scalar polynomial event isolation from serialized intervals."""

    pattern = _event_isolation_sign_pattern(certificate.event_type)
    coefficient_intervals = _event_interval_coefficients(
        certificate.coefficient_intervals,
    )
    rational_coefficient_intervals = ()
    search_interval = tuple(float(value) for value in certificate.search_interval)
    root_interval = tuple(float(value) for value in certificate.root_interval)
    root = float(certificate.root)
    event_value = float(certificate.event_value)

    identity_ok = bool(certificate.certificate_id and certificate.event_id)
    event_type_supported = pattern is not None
    coefficient_source_supported = bool(
        event_type_supported
        and certificate.coefficient_source
        in _event_isolation_supported_sources(certificate.event_type)
    )
    finite_event_value = bool(np.isfinite(event_value) and event_value > 0.0)
    pair_valid = _event_isolation_pair_valid(certificate)
    finite_coefficients = bool(
        len(coefficient_intervals) >= 2
        and all(
            np.isfinite(interval.lower)
            and np.isfinite(interval.upper)
            and interval.lower <= interval.upper
            for interval in coefficient_intervals
        )
    )
    finite_intervals = bool(
        _finite_nonempty_interval(search_interval)
        and _finite_nonempty_interval(root_interval)
        and np.isfinite(root)
    )
    root_contained = bool(
        finite_intervals
        and search_interval[0] <= root <= search_interval[1]
        and root_interval[0] <= root <= root_interval[1]
        and search_interval[0] <= root_interval[0] <= root_interval[1] <= search_interval[1]
    )

    left_sign_certified = False
    right_sign_certified = False
    derivative_sign_certified = False
    excludes_earlier_roots = False
    exact_interval_arithmetic_certified = False
    if pattern is not None and finite_coefficients and finite_intervals:
        try:
            rational_coefficient_intervals = _event_rational_interval_coefficients(
                certificate.coefficient_intervals,
            )
            left_value = rational_interval_polynomial_eval(
                rational_coefficient_intervals,
                RationalInterval.from_float_interval(root_interval[0]),
            )
            right_value = rational_interval_polynomial_eval(
                rational_coefficient_intervals,
                RationalInterval.from_float_interval(root_interval[1]),
            )
            derivative_value = rational_interval_polynomial_eval(
                rational_interval_polyder(rational_coefficient_intervals),
                RationalInterval.from_float_interval(*root_interval),
            )
            left_sign_certified = rational_interval_sign(left_value) == pattern["left"]
            right_sign_certified = rational_interval_sign(right_value) == pattern["right"]
            derivative_sign_certified = (
                rational_interval_sign(derivative_value) == pattern["derivative"]
            )
            excludes_earlier_roots = _signed_rational_polynomial_range_certified(
                rational_coefficient_intervals,
                search_interval[0],
                root_interval[0],
                sign=pattern["pre_event"],
            )
            exact_interval_arithmetic_certified = True
        except (FloatingPointError, ValueError):
            left_sign_certified = False
            right_sign_certified = False
            derivative_sign_certified = False
            excludes_earlier_roots = False
            exact_interval_arithmetic_certified = False

    root_interval_width = (
        float(root_interval[1] - root_interval[0])
        if finite_intervals
        else float("inf")
    )
    obligations = (
        CertificateCheckObligation(
            "event_identity_present",
            identity_ok,
            (
                f"certificate_id={certificate.certificate_id!r}; "
                f"event_id={certificate.event_id!r}"
            ),
        ),
        CertificateCheckObligation(
            "event_type_supported",
            event_type_supported,
            f"event_type={certificate.event_type!r}",
        ),
        CertificateCheckObligation(
            "event_coefficient_source_supported",
            coefficient_source_supported,
            f"coefficient_source={certificate.coefficient_source!r}",
        ),
        CertificateCheckObligation(
            "event_value_positive",
            finite_event_value,
            f"event_value={event_value}",
        ),
        CertificateCheckObligation(
            "event_pair_valid",
            pair_valid,
            f"pair={certificate.pair!r}",
        ),
        CertificateCheckObligation(
            "event_coefficients_finite",
            finite_coefficients,
            f"coefficient_count={len(coefficient_intervals)}",
        ),
        CertificateCheckObligation(
            "event_intervals_finite_and_nested",
            root_contained,
            (
                f"search_interval={search_interval}; "
                f"root_interval={root_interval}; root={root}"
            ),
        ),
        CertificateCheckObligation(
            "event_exact_rational_interval_arithmetic",
            exact_interval_arithmetic_certified,
            (
                "polynomial endpoint, derivative, and pre-event sign checks "
                "use exact rational interval arithmetic over the serialized "
                f"binary-float endpoints; coefficient_count={len(rational_coefficient_intervals)}"
            ),
        ),
        CertificateCheckObligation(
            "event_root_endpoint_signs",
            bool(left_sign_certified and right_sign_certified),
            f"expected_pattern={pattern}",
        ),
        CertificateCheckObligation(
            "event_root_derivative_sign",
            derivative_sign_certified,
            f"expected_pattern={pattern}",
        ),
        CertificateCheckObligation(
            "event_excludes_earlier_roots",
            excludes_earlier_roots,
            f"search_interval={search_interval}; root_interval={root_interval}",
        ),
    )
    return EventIsolationCheckResult(
        event_id=str(certificate.event_id),
        event_type=str(certificate.event_type),
        checker_id="independent_polynomial_event_isolation_checker_v2",
        obligations=obligations,
        root_interval_width=root_interval_width,
    )


def check_branch_union(
    certificate: BranchUnionCertificate,
    checked_leaf_results: Iterable[object],
) -> BranchUnionCheckResult:
    """Verify a finite branch union against independently checked leaves."""

    union_type = str(certificate.union_type)
    leaf_ids = tuple(str(value) for value in certificate.leaf_response_certificate_ids)
    leaf_intervals = tuple(
        tuple(float(value) for value in interval)
        for interval in certificate.leaf_target_intervals
    )
    aggregate = tuple(float(value) for value in certificate.aggregate_target_interval)
    checked = tuple(checked_leaf_results)
    certified_leaf_ids = {
        response_id
        for response_id in (
            _checked_response_identifier(result)
            for result in checked
            if getattr(result, "certified", False)
        )
        if response_id
    }
    supported_union_type = union_type in {
        "finite_time_branch_union",
        "finite_time_event_order_branch_union",
        "finite_time_stratified_branch_union",
    }
    leaf_kinds = tuple(str(value) for value in certificate.leaf_kinds)
    stratified_union = union_type == "finite_time_stratified_branch_union"
    leaf_ids_present = bool(leaf_ids)
    leaf_ids_unique = len(set(leaf_ids)) == len(leaf_ids)
    leaf_intervals_match = len(leaf_intervals) == len(leaf_ids)
    leaf_kinds_match = bool(
        not leaf_kinds
        or len(leaf_kinds) == len(leaf_ids)
    )
    stratified_leaf_kinds_present = bool(not stratified_union or leaf_kinds)
    stratified_leaf_kinds_supported = bool(
        not stratified_union
        or (
            leaf_kinds_match
            and all(kind in SUPPORTED_STRATIFIED_LEAF_KINDS for kind in leaf_kinds)
        )
    )
    stratified_unsupported_absent = bool(
        not stratified_union
        or all(kind != "unsupported_analytic_stratum" for kind in leaf_kinds)
    )
    aggregate_finite = _finite_nonempty_interval(aggregate)
    leaf_intervals_finite = bool(
        leaf_intervals
        and all(_finite_nonempty_interval(interval) for interval in leaf_intervals)
    )
    leaf_responses_checked = bool(
        leaf_ids_present
        and all(leaf_id in certified_leaf_ids for leaf_id in leaf_ids)
    )
    aggregate_contains_leaves = bool(
        aggregate_finite
        and leaf_intervals_finite
        and all(_interval_contains_interval(aggregate, interval) for interval in leaf_intervals)
    )
    (
        exact_rational_interval_aggregation,
        exact_rational_interval_aggregation_detail,
    ) = _exact_rational_branch_union_interval_aggregation(
        aggregate,
        leaf_intervals,
    )
    obligations = (
        CertificateCheckObligation(
            "branch_union_type_supported",
            supported_union_type,
            f"union_type={union_type}",
        ),
        CertificateCheckObligation(
            "branch_union_leaf_ids_present",
            leaf_ids_present,
            f"leaf_count={len(leaf_ids)}",
        ),
        CertificateCheckObligation(
            "branch_union_leaf_ids_unique",
            leaf_ids_unique,
            f"unique_leaf_count={len(set(leaf_ids))}; leaf_count={len(leaf_ids)}",
        ),
        CertificateCheckObligation(
            "branch_union_leaf_interval_count_matches",
            leaf_intervals_match,
            (
                f"leaf_interval_count={len(leaf_intervals)}; "
                f"leaf_count={len(leaf_ids)}"
            ),
        ),
        CertificateCheckObligation(
            "branch_union_leaf_kinds_match",
            leaf_kinds_match,
            f"leaf_kind_count={len(leaf_kinds)}; leaf_count={len(leaf_ids)}",
        ),
        CertificateCheckObligation(
            "stratified_branch_union_leaf_kinds_present",
            stratified_leaf_kinds_present,
            f"union_type={union_type}; leaf_kinds={leaf_kinds}",
        ),
        CertificateCheckObligation(
            "stratified_branch_union_leaf_kinds_supported",
            stratified_leaf_kinds_supported,
            (
                f"leaf_kinds={leaf_kinds}; "
                f"supported={tuple(sorted(SUPPORTED_STRATIFIED_LEAF_KINDS))}"
            ),
        ),
        CertificateCheckObligation(
            "stratified_branch_union_unsupported_strata_absent",
            stratified_unsupported_absent,
            f"leaf_kinds={leaf_kinds}",
        ),
        CertificateCheckObligation(
            "branch_union_intervals_finite",
            bool(aggregate_finite and leaf_intervals_finite),
            f"aggregate={aggregate}; leaf_intervals={leaf_intervals}",
        ),
        CertificateCheckObligation(
            "branch_union_leaf_responses_independently_checked",
            leaf_responses_checked,
            (
                f"leaf_ids={leaf_ids}; "
                f"certified_leaf_response_ids={tuple(sorted(certified_leaf_ids))}"
            ),
        ),
        CertificateCheckObligation(
            "branch_union_aggregate_contains_leaf_targets",
            aggregate_contains_leaves,
            f"aggregate={aggregate}; leaf_intervals={leaf_intervals}",
        ),
        CertificateCheckObligation(
            "branch_union_exact_rational_interval_aggregation",
            exact_rational_interval_aggregation,
            exact_rational_interval_aggregation_detail,
        ),
    )
    return BranchUnionCheckResult(
        union_id=str(certificate.union_id),
        union_type=union_type,
        checker_id="branch_union_checker_v1",
        obligations=obligations,
        leaf_count=len(leaf_ids),
    )


def _checked_response_identifier(result: object) -> str:
    for attribute in ("certificate_id", "chain_id", "union_id"):
        value = getattr(result, attribute, "")
        if value:
            return str(value)
    return ""


def check_chart_chain(
    certificate: ChartChainCertificate,
    chart_certificates: Iterable[
        OrdinaryTaylorChartCertificate
        | PlanarLeviCivitaBinaryChartCertificate
        | SpatialKSBinaryChartCertificate
        | TotalCollisionFuchsianStopChartCertificate
        | TotalCollisionGeneralizedFuchsianStopChartCertificate
    ],
    transition_certificates: Iterable[
        OrdinaryChartTransitionCertificate
        | PlanarLeviCivitaTransitionCertificate
        | SpatialKSTransitionCertificate
    ],
    checked_chart_results: Iterable[CertificateCheckResult],
    checked_transition_results: Iterable[TransitionCheckResult],
) -> ChartChainCheckResult:
    """Verify finite chart-chain grammar and physical target coverage."""

    chain_type = str(certificate.chain_type)
    chart_ids = tuple(str(value) for value in certificate.chart_ids)
    transition_ids = tuple(str(value) for value in certificate.transition_ids)
    target_interval = tuple(float(value) for value in certificate.target_physical_time_interval)
    chart_by_id = {str(chart.chart_id): chart for chart in chart_certificates}
    transition_by_id = {
        str(transition.transition_id): transition
        for transition in transition_certificates
    }
    certified_chart_ids = {
        result.certificate_id
        for result in checked_chart_results
        if result.certified and result.certificate_id
    }
    certified_transition_ids = {
        result.transition_id
        for result in checked_transition_results
        if result.certified and result.transition_id
    }

    supported_chain_type = chain_type in {
        "finite_time_chart_chain",
        "ordinary_chart_chain",
        "regularized_atlas_chart_chain",
    }
    identity_present = bool(certificate.certificate_id and certificate.chain_id)
    chart_ids_present = bool(chart_ids)
    chart_ids_unique = len(set(chart_ids)) == len(chart_ids)
    transition_count_matches = len(transition_ids) == max(0, len(chart_ids) - 1)
    charts_present = all(chart_id in chart_by_id for chart_id in chart_ids)
    transitions_present = all(
        transition_id in transition_by_id
        for transition_id in transition_ids
    )
    chart_certificates_checked = bool(
        chart_ids_present
        and charts_present
        and all(
            chart_by_id[chart_id].certificate_id in certified_chart_ids
            for chart_id in chart_ids
        )
    )
    transition_certificates_checked = bool(
        transition_count_matches
        and transitions_present
        and all(transition_id in certified_transition_ids for transition_id in transition_ids)
    )
    transition_adjacency = bool(
        transition_count_matches
        and transitions_present
        and all(
            str(transition_by_id[transition_ids[index]].source_chart_id)
            == chart_ids[index]
            and str(transition_by_id[transition_ids[index]].target_chart_id)
            == chart_ids[index + 1]
            for index in range(len(transition_ids))
        )
    )
    listed_problem_identities = tuple(
        _serialized_chart_problem_identity(chart_by_id[chart_id])
        for chart_id in chart_ids
        if chart_id in chart_by_id
    )
    common_problem_identity = bool(
        chart_ids_present
        and charts_present
        and len(listed_problem_identities) == len(chart_ids)
        and all(identity is not None for identity in listed_problem_identities)
        and len(set(listed_problem_identities)) == 1
    )
    chart_intervals = tuple(
        tuple(float(value) for value in getattr(chart_by_id[chart_id], "physical_time_interval", ()))
        for chart_id in chart_ids
        if chart_id in chart_by_id
    )
    target_interval_finite = _finite_nonempty_interval(target_interval)
    chart_intervals_finite = bool(
        len(chart_intervals) == len(chart_ids)
        and all(_finite_nonempty_interval(interval) for interval in chart_intervals)
    )
    monotone_no_gaps = bool(
        chart_intervals_finite
        and all(
            chart_intervals[index][0] <= chart_intervals[index + 1][0]
            and chart_intervals[index + 1][0] <= chart_intervals[index][1]
            for index in range(len(chart_intervals) - 1)
        )
    )
    if chart_intervals_finite and monotone_no_gaps:
        covered_interval = (
            chart_intervals[0][0],
            max(interval[1] for interval in chart_intervals),
        )
    elif chart_intervals_finite and len(chart_intervals) == 1:
        covered_interval = chart_intervals[0]
    else:
        covered_interval = (float("inf"), float("-inf"))
    target_interval_covered = bool(
        target_interval_finite
        and _finite_nonempty_interval(covered_interval)
        and _interval_contains_interval(covered_interval, target_interval)
    )
    (
        exact_rational_time_coverage,
        exact_rational_time_coverage_detail,
    ) = _exact_rational_chart_chain_time_coverage(
        chart_intervals,
        target_interval,
    )
    obligations = (
        CertificateCheckObligation(
            "chart_chain_identity_present",
            identity_present,
            f"certificate_id={certificate.certificate_id!r}; chain_id={certificate.chain_id!r}",
        ),
        CertificateCheckObligation(
            "chart_chain_type_supported",
            supported_chain_type,
            f"chain_type={chain_type!r}",
        ),
        CertificateCheckObligation(
            "chart_chain_chart_ids_present",
            chart_ids_present,
            f"chart_count={len(chart_ids)}",
        ),
        CertificateCheckObligation(
            "chart_chain_chart_ids_unique",
            chart_ids_unique,
            f"unique_chart_count={len(set(chart_ids))}; chart_count={len(chart_ids)}",
        ),
        CertificateCheckObligation(
            "chart_chain_transition_count_matches",
            transition_count_matches,
            f"transition_count={len(transition_ids)}; chart_count={len(chart_ids)}",
        ),
        CertificateCheckObligation(
            "chart_chain_charts_independently_checked",
            chart_certificates_checked,
            f"chart_ids={chart_ids}; certified_chart_certificate_ids={tuple(sorted(certified_chart_ids))}",
        ),
        CertificateCheckObligation(
            "chart_chain_transitions_independently_checked",
            transition_certificates_checked,
            f"transition_ids={transition_ids}; certified_transition_ids={tuple(sorted(certified_transition_ids))}",
        ),
        CertificateCheckObligation(
            "chart_chain_transition_adjacency",
            transition_adjacency,
            f"chart_ids={chart_ids}; transition_ids={transition_ids}",
        ),
        CertificateCheckObligation(
            "chart_chain_common_problem_identity",
            common_problem_identity,
            f"listed_problem_identities={listed_problem_identities!r}",
        ),
        CertificateCheckObligation(
            "chart_chain_intervals_finite",
            bool(target_interval_finite and chart_intervals_finite),
            f"target_interval={target_interval}; chart_intervals={chart_intervals}",
        ),
        CertificateCheckObligation(
            "chart_chain_no_physical_time_gaps",
            bool(len(chart_intervals) <= 1 or monotone_no_gaps),
            f"chart_intervals={chart_intervals}",
        ),
        CertificateCheckObligation(
            "chart_chain_target_interval_covered",
            target_interval_covered,
            f"covered_interval={covered_interval}; target_interval={target_interval}",
        ),
        CertificateCheckObligation(
            "chart_chain_exact_rational_time_coverage",
            exact_rational_time_coverage,
            exact_rational_time_coverage_detail,
        ),
    )
    return ChartChainCheckResult(
        chain_id=str(certificate.chain_id),
        chain_type=chain_type,
        checker_id="chart_chain_checker_v1",
        obligations=obligations,
        chart_count=len(chart_ids),
    )


def verify_chart_certificates(
    certificates: Iterable[
        OrdinaryTaylorChartCertificate
        | PlanarLeviCivitaBinaryChartCertificate
        | SpatialKSBinaryChartCertificate
        | TotalCollisionFuchsianStopChartCertificate
        | TotalCollisionGeneralizedFuchsianStopChartCertificate
    ],
    transitions: Iterable[
        OrdinaryChartTransitionCertificate
        | PlanarLeviCivitaTransitionCertificate
        | SpatialKSTransitionCertificate
    ] = (),
    events: Iterable[EventIsolationCertificate] = (),
    branch_unions: Iterable[BranchUnionCertificate] = (),
    chart_chains: Iterable[ChartChainCertificate] = (),
) -> IndependentChartVerifierCertificate:
    """Run the independent checker over a finite serialized chart bundle."""

    chart_tuple = tuple(certificates)
    transition_tuple = tuple(transitions)
    event_tuple = tuple(events)
    branch_union_tuple = tuple(branch_unions)
    chart_chain_tuple = tuple(chart_chains)

    chart_ids = tuple(str(certificate.chart_id) for certificate in chart_tuple)
    certificate_ids = tuple(
        str(certificate.certificate_id) for certificate in chart_tuple
    )
    transition_ids = tuple(
        str(certificate.transition_id) for certificate in transition_tuple
    )
    event_ids = tuple(str(certificate.event_id) for certificate in event_tuple)
    union_ids = tuple(
        str(certificate.union_id) for certificate in branch_union_tuple
    )
    chain_ids = tuple(
        str(certificate.chain_id) for certificate in chart_chain_tuple
    )
    chart_ids_unique = _nonempty_ids_unique(chart_ids)
    certificate_ids_unique = _nonempty_ids_unique(certificate_ids)
    transition_ids_unique = _nonempty_ids_unique(transition_ids)
    event_ids_unique = _nonempty_ids_unique(event_ids)
    union_ids_unique = _nonempty_ids_unique(union_ids)
    chain_ids_unique = _nonempty_ids_unique(chain_ids)

    results = tuple(
        _with_bundle_identity_obligations(
            _check_chart_certificate(certificate),
            chart_ids_unique=chart_ids_unique,
            certificate_ids_unique=certificate_ids_unique,
            detail=(
                f"chart_ids={chart_ids!r}; certificate_ids={certificate_ids!r}"
            ),
        )
        for certificate in chart_tuple
    )
    transition_results = tuple(
        _with_unique_namespace_obligation(
            _check_transition_certificate(transition, chart_tuple),
            obligation="bundle_transition_ids_unique",
            certified=transition_ids_unique,
            detail=f"transition_ids={transition_ids!r}",
        )
        for transition in transition_tuple
    )
    event_results = tuple(
        _with_unique_namespace_obligation(
            check_event_isolation(event),
            obligation="bundle_event_ids_unique",
            certified=event_ids_unique,
            detail=f"event_ids={event_ids!r}",
        )
        for event in event_tuple
    )
    chart_chain_results = tuple(
        _with_unique_namespace_obligation(
            check_chart_chain(
                chart_chain,
                chart_tuple,
                transition_tuple,
                results,
                transition_results,
            ),
            obligation="bundle_chart_chain_ids_unique",
            certified=chain_ids_unique,
            detail=f"chain_ids={chain_ids!r}",
        )
        for chart_chain in chart_chain_tuple
    )
    branch_union_results = tuple(
        _with_unique_namespace_obligation(
            check_branch_union(
                branch_union,
                (*results, *chart_chain_results),
            ),
            obligation="bundle_branch_union_ids_unique",
            certified=union_ids_unique,
            detail=f"union_ids={union_ids!r}",
        )
        for branch_union in branch_union_tuple
    )
    return IndependentChartVerifierCertificate(
        checker_id="independent_chart_verifier_v1",
        chart_results=results,
        transition_results=transition_results,
        event_results=event_results,
        branch_union_results=branch_union_results,
        chart_chain_results=chart_chain_results,
    )


def _nonempty_ids_unique(values: tuple[str, ...]) -> bool:
    """Reject duplicate or empty identifiers within one bundle namespace."""

    return bool(all(values) and len(set(values)) == len(values)) if values else True


def _with_bundle_identity_obligations(
    result: CertificateCheckResult,
    *,
    chart_ids_unique: bool,
    certificate_ids_unique: bool,
    detail: str,
) -> CertificateCheckResult:
    return replace(
        result,
        obligations=(
            *result.obligations,
            CertificateCheckObligation(
                "bundle_chart_ids_unique",
                chart_ids_unique,
                detail,
            ),
            CertificateCheckObligation(
                "bundle_certificate_ids_unique",
                certificate_ids_unique,
                detail,
            ),
        ),
    )


def _with_unique_namespace_obligation(
    result: TransitionCheckResult
    | EventIsolationCheckResult
    | BranchUnionCheckResult
    | ChartChainCheckResult,
    *,
    obligation: str,
    certified: bool,
    detail: str,
) -> TransitionCheckResult | EventIsolationCheckResult | BranchUnionCheckResult | ChartChainCheckResult:
    return replace(
        result,
        obligations=(
            *result.obligations,
            CertificateCheckObligation(obligation, certified, detail),
        ),
    )


def _check_chart_certificate(
    certificate: OrdinaryTaylorChartCertificate
    | PlanarLeviCivitaBinaryChartCertificate
    | SpatialKSBinaryChartCertificate
    | TotalCollisionFuchsianStopChartCertificate
    | TotalCollisionGeneralizedFuchsianStopChartCertificate,
) -> CertificateCheckResult:
    if isinstance(certificate, OrdinaryTaylorChartCertificate):
        return check_ordinary_taylor_chart(certificate)
    if isinstance(certificate, PlanarLeviCivitaBinaryChartCertificate):
        return check_planar_levi_civita_binary_chart(certificate)
    if isinstance(certificate, SpatialKSBinaryChartCertificate):
        return check_spatial_ks_binary_chart(certificate)
    if isinstance(certificate, TotalCollisionFuchsianStopChartCertificate):
        return check_total_collision_fuchsian_stop_chart(certificate)
    if isinstance(certificate, TotalCollisionGeneralizedFuchsianStopChartCertificate):
        return check_total_collision_generalized_fuchsian_stop_chart(certificate)
    raise TypeError(f"unsupported chart certificate type: {type(certificate)!r}")


def _check_transition_certificate(
    transition: OrdinaryChartTransitionCertificate
    | PlanarLeviCivitaTransitionCertificate
    | SpatialKSTransitionCertificate,
    charts: tuple[
        OrdinaryTaylorChartCertificate
        | PlanarLeviCivitaBinaryChartCertificate
        | SpatialKSBinaryChartCertificate,
        ...,
    ],
) -> TransitionCheckResult:
    if isinstance(transition, OrdinaryChartTransitionCertificate):
        ordinary_charts = tuple(
            chart for chart in charts if isinstance(chart, OrdinaryTaylorChartCertificate)
        )
        return check_ordinary_chart_transition(transition, ordinary_charts)
    if isinstance(transition, PlanarLeviCivitaTransitionCertificate):
        return check_planar_levi_civita_transition(transition, charts)
    if isinstance(transition, SpatialKSTransitionCertificate):
        return check_spatial_ks_transition(transition, charts)
    raise TypeError(f"unsupported transition certificate type: {type(transition)!r}")


def attach_independent_chart_verifier(
    theorem_certificate: object,
    verifier_certificate: IndependentChartVerifierCertificate,
) -> object:
    """Attach checker evidence to an open-time theorem certificate.

    The returned object is still the same dataclass type as the theorem
    certificate, but it carries the verifier evidence fields consumed by the
    closed-form audit.
    """

    if getattr(theorem_certificate, "theorem_id", None) != "open_time_locally_finite_atlas":
        raise TypeError("independent chart verifier can only be attached to open-time atlas theorem certificates")
    if type(verifier_certificate) is not IndependentChartVerifierCertificate:
        raise TypeError("independent chart verifier must be constructor-derived")
    return replace(
        theorem_certificate,
        independent_chart_verifier_certificate=verifier_certificate,
        independent_chart_verifier_certified=verifier_certificate.certified is True,
    )


def _coefficient_array(values: object) -> np.ndarray:
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return np.asarray((), dtype=float)


def _regularized_binary_solution_from_certificate(
    certificate: PlanarLeviCivitaBinaryChartCertificate,
) -> RegularizedBinaryTaylorSolution:
    return RegularizedBinaryTaylorSolution(
        masses=np.asarray(certificate.masses, dtype=float),
        pair=tuple(int(value) for value in certificate.pair),
        z=_coefficient_array(certificate.z_coefficients),
        z_velocity=_coefficient_array(certificate.z_velocity_coefficients),
        pair_energy=_coefficient_array(certificate.pair_energy_coefficients),
        binary_center=_coefficient_array(certificate.binary_center_coefficients),
        binary_center_velocity=_coefficient_array(
            certificate.binary_center_velocity_coefficients,
        ),
        third_offset=_coefficient_array(certificate.third_offset_coefficients),
        third_offset_velocity=_coefficient_array(
            certificate.third_offset_velocity_coefficients,
        ),
        physical_time=_coefficient_array(certificate.physical_time_coefficients),
    )


def _spatial_ks_solution_from_certificate(
    certificate: SpatialKSBinaryChartCertificate,
) -> SpatialKSBinaryTaylorSolution:
    return SpatialKSBinaryTaylorSolution(
        masses=np.asarray(certificate.masses, dtype=float),
        pair=tuple(int(value) for value in certificate.pair),
        u=_coefficient_array(certificate.u_coefficients),
        u_velocity=_coefficient_array(certificate.u_velocity_coefficients),
        pair_energy=_coefficient_array(certificate.pair_energy_coefficients),
        binary_center=_coefficient_array(certificate.binary_center_coefficients),
        binary_center_velocity=_coefficient_array(
            certificate.binary_center_velocity_coefficients,
        ),
        third_offset=_coefficient_array(certificate.third_offset_coefficients),
        third_offset_velocity=_coefficient_array(
            certificate.third_offset_velocity_coefficients,
        ),
        physical_time=_coefficient_array(certificate.physical_time_coefficients),
    )


def _fuchsian_log_branch_from_certificate(
    certificate: TotalCollisionFuchsianStopChartCertificate,
) -> FiniteFuchsianLogBranch:
    return FiniteFuchsianLogBranch(
        masses=np.asarray(certificate.masses, dtype=float),
        central_shape=np.asarray(certificate.central_shape, dtype=float),
        scale_coefficient=float(certificate.scale_coefficient),
        terms=tuple(
            _fuchsian_log_term_from_certificate(term)
            for term in certificate.terms
        ),
    )


def _generalized_fuchsian_branch_from_certificate(
    certificate: TotalCollisionGeneralizedFuchsianStopChartCertificate,
) -> FuchsianShapeBranch:
    selected_coefficients = {
        tuple(int(value) for value in item.index): np.asarray(
            item.coefficient,
            dtype=float,
        )
        for item in certificate.selected_coefficients
    }
    return construct_fuchsian_shape_branch(
        masses=np.asarray(certificate.masses, dtype=float),
        central_shape=np.asarray(certificate.central_shape, dtype=float),
        powers=tuple(float(power) for power in certificate.powers),
        selected_coefficients=selected_coefficients,
        max_total_degree=int(certificate.max_total_degree),
        scale_index=(
            None
            if certificate.scale_index is None
            else tuple(int(value) for value in certificate.scale_index)
        ),
    )


def _generalized_fuchsian_shape_deviation_bound_for_checker(
    branch: FuchsianShapeBranch,
    radius: float,
) -> float:
    radius = float(radius)
    if radius <= 0.0:
        return float("inf")
    bound = 0.0
    for index, coefficient in branch.coefficients.items():
        if index == branch.zero_index:
            continue
        exponent = branch.exponent(index)
        if not np.isfinite(exponent) or exponent <= 0.0:
            return float("inf")
        bound += (
            float(np.linalg.norm(np.asarray(coefficient, dtype=float), ord=np.inf))
            * radius**exponent
        )
    return float(bound)


def _max_sampled_generalized_fuchsian_stop_chart_quantities(
    branch: FuchsianShapeBranch,
    tau_interval: tuple[float, float],
    *,
    sample_count: int,
) -> tuple[float, float, float, int]:
    max_lifted_residual = 0.0
    max_angular_momentum = 0.0
    max_position_scale = 0.0
    checked = 0
    for tau in _punctured_tau_samples(tau_interval, sample_count=sample_count):
        lifted_residual = branch.shape_equation_residual_at_tau(tau)
        positions = branch.positions_at_tau(tau)
        max_lifted_residual = max(
            max_lifted_residual,
            _array_sup_norm(lifted_residual),
        )
        if branch.central_shape.shape[1] == 2:
            angular = abs(branch.centered_angular_momentum_scalar_at_tau(tau))
        else:
            angular = _spatial_centered_angular_momentum_norm(branch, tau)
        max_angular_momentum = max(max_angular_momentum, float(angular))
        max_position_scale = max(
            max_position_scale,
            float(np.max(np.abs(positions))) / tau**2,
        )
        checked += 1
    if checked == 0:
        return (np.inf, np.inf, np.inf, 0)
    return (
        float(max_lifted_residual),
        float(max_angular_momentum),
        float(max_position_scale),
        int(checked),
    )


def _check_generalized_remainder_majorant(
    majorant: GeneralizedFuchsianRemainderMajorantCertificate | None,
    *,
    residual_tolerance: float,
    tail_bound: float,
) -> dict[str, object]:
    if majorant is None:
        return {
            "initial_radius": np.inf,
            "constants_finite": False,
            "contraction_certified": False,
            "self_map_certified": False,
            "component_inputs_certified": False,
            "certified": False,
            "tail_bound": np.inf,
            "tail_covered": False,
            "lifted_residual_tail_bound": np.inf,
            "lifted_residual_tail_certified": False,
            "physical_residual_tail_bound": np.inf,
            "physical_residual_tail_certified": False,
            "required_residual_components_present": False,
            "required_residual_component_detail": "remainder majorant not supplied",
            "endpoint_collapse_certified": False,
            "constant_detail": "remainder majorant not supplied",
            "contraction_detail": "remainder majorant not supplied",
            "self_map_detail": "remainder majorant not supplied",
            "component_detail": "remainder majorant not supplied",
            "endpoint_detail": "remainder majorant not supplied",
            "detail": "remainder majorant not supplied",
        }

    initial_radius = float(majorant.initial_radius)
    shell_contraction = float(majorant.shell_contraction)
    analytic_disk_fraction = float(majorant.analytic_disk_fraction)
    defect_bound = float(majorant.defect_bound)
    linear_inverse_bound = float(majorant.linear_inverse_bound)
    nonlinear_lipschitz_bound = float(majorant.nonlinear_lipschitz_bound)
    remainder_ball_radius = float(majorant.remainder_ball_radius)
    contraction_factor = float(linear_inverse_bound * nonlinear_lipschitz_bound)
    self_map_bound = float(
        linear_inverse_bound * defect_bound
        + contraction_factor * remainder_ball_radius
    )
    contraction_slack = float(1.0 - contraction_factor)
    self_map_margin = float(remainder_ball_radius - self_map_bound)
    component_inputs = dict(majorant.component_inputs)
    exponents = dict(majorant.component_effective_exponents)
    header_certified = bool(
        np.isfinite(initial_radius)
        and initial_radius > 0.0
        and np.isfinite(shell_contraction)
        and 0.0 < shell_contraction < 1.0
        and np.isfinite(analytic_disk_fraction)
        and 0.0 < analytic_disk_fraction < 1.0
        and component_inputs
        and exponents
    )
    constants_finite = bool(
        np.isfinite(defect_bound)
        and defect_bound >= 0.0
        and np.isfinite(linear_inverse_bound)
        and linear_inverse_bound >= 0.0
        and np.isfinite(nonlinear_lipschitz_bound)
        and nonlinear_lipschitz_bound >= 0.0
        and np.isfinite(remainder_ball_radius)
        and remainder_ball_radius >= 0.0
    )
    contraction_certified = bool(
        constants_finite
        and np.isfinite(contraction_factor)
        and contraction_factor < 1.0
    )
    self_map_certified = bool(
        constants_finite
        and contraction_certified
        and np.isfinite(self_map_bound)
        and self_map_bound <= remainder_ball_radius * (1.0 + 1.0e-12)
    )
    key_sets_match = bool(
        component_inputs and set(component_inputs) == set(exponents)
    )

    component_failures: list[str] = []
    envelope_failures: list[str] = []
    first_shell_bounds: list[float] = []
    primitive_by_component: dict[str, PrimitiveCauchyTailInput] = {}
    for component, serialized in component_inputs.items():
        primitive = PrimitiveCauchyTailInput(
            majorant_initial=float(serialized.majorant_initial),
            majorant_growth=float(serialized.majorant_growth),
            step_ratio_bound=float(serialized.step_ratio_bound),
            retained_order_initial=int(serialized.retained_order_initial),
            retained_order_increment=int(serialized.retained_order_increment),
        )
        primitive_by_component[component] = primitive
        recomputed_first_shell = primitive.first_shell_tail_bound
        recomputed_shell_ratio = primitive.shell_ratio
        first_shell_bounds.append(float(recomputed_first_shell))
        first_shell_matches = _finite_close(
            float(serialized.first_shell_tail_bound),
            recomputed_first_shell,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        shell_ratio_matches = _finite_close(
            float(serialized.shell_ratio),
            recomputed_shell_ratio,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        try:
            expected_initial, expected_growth = (
                _generalized_remainder_shell_majorant_for_checker(
                    remainder_ball_radius=remainder_ball_radius,
                    effective_exponent=float(exponents[component]),
                    initial_radius=initial_radius,
                    shell_contraction=shell_contraction,
                    analytic_disk_fraction=analytic_disk_fraction,
                )
            )
            envelope_covers = bool(
                float(serialized.majorant_initial) + 1.0e-15 >= expected_initial
                and float(serialized.majorant_growth) + 1.0e-15 >= expected_growth
            )
        except (FloatingPointError, KeyError, TypeError, ValueError):
            envelope_covers = False
        if not envelope_covers:
            envelope_failures.append(component)
        if not (
            primitive.certified
            and first_shell_matches
            and shell_ratio_matches
            and envelope_covers
        ):
            component_failures.append(component)

    remainder_tail_bound = float(max(first_shell_bounds, default=np.inf))
    lifted_residual = primitive_by_component.get("lifted_residual")
    lifted_residual_tail_bound = (
        float(lifted_residual.first_shell_tail_bound)
        if lifted_residual is not None
        else float("inf")
    )
    lifted_residual_tail_certified = bool(
        lifted_residual is not None
        and lifted_residual.certified
        and np.isfinite(residual_tolerance)
        and residual_tolerance >= 0.0
        and np.isfinite(lifted_residual_tail_bound)
        and lifted_residual_tail_bound <= residual_tolerance
    )
    physical_residual = primitive_by_component.get("physical_residual")
    physical_residual_tail_bound = (
        float(physical_residual.first_shell_tail_bound)
        if physical_residual is not None
        else float("inf")
    )
    physical_residual_tail_certified = bool(
        physical_residual is not None
        and physical_residual.certified
        and np.isfinite(residual_tolerance)
        and residual_tolerance >= 0.0
        and np.isfinite(physical_residual_tail_bound)
        and physical_residual_tail_bound <= residual_tolerance
    )
    required_residual_components = ("lifted_residual", "physical_residual")
    missing_residual_components = tuple(
        component
        for component in required_residual_components
        if component not in primitive_by_component
    )
    required_residual_components_present = bool(not missing_residual_components)
    endpoint_component = (
        "regularized_position_value"
        if "regularized_position_value" in primitive_by_component
        else "value"
    )
    endpoint_input = primitive_by_component.get(endpoint_component)
    endpoint_collapse_certified = bool(
        endpoint_input is not None
        and endpoint_input.certified
        and np.isfinite(endpoint_input.first_shell_tail_bound)
        and endpoint_input.first_shell_tail_bound >= 0.0
        and np.isfinite(endpoint_input.shell_ratio)
        and endpoint_input.shell_ratio < 1.0
    )
    component_inputs_certified = bool(
        header_certified
        and key_sets_match
        and not component_failures
        and not envelope_failures
        and np.isfinite(remainder_tail_bound)
    )
    tail_covered = bool(
        np.isfinite(tail_bound)
        and tail_bound >= 0.0
        and np.isfinite(remainder_tail_bound)
        and tail_bound + 1.0e-15 >= remainder_tail_bound
    )
    certified = bool(
        constants_finite
        and contraction_certified
        and self_map_certified
        and component_inputs_certified
    )
    return {
        "initial_radius": initial_radius,
        "constants_finite": constants_finite,
        "contraction_certified": contraction_certified,
        "self_map_certified": self_map_certified,
        "component_inputs_certified": component_inputs_certified,
        "certified": certified,
        "tail_bound": remainder_tail_bound,
        "tail_covered": tail_covered,
        "lifted_residual_tail_bound": lifted_residual_tail_bound,
        "lifted_residual_tail_certified": lifted_residual_tail_certified,
        "physical_residual_tail_bound": physical_residual_tail_bound,
        "physical_residual_tail_certified": physical_residual_tail_certified,
        "required_residual_components_present": (
            required_residual_components_present
        ),
        "required_residual_component_detail": (
            f"required_components={required_residual_components}; "
            f"missing_components={missing_residual_components}; "
            f"lifted_residual_tail={lifted_residual_tail_bound}; "
            f"physical_residual_tail={physical_residual_tail_bound}"
        ),
        "endpoint_collapse_certified": endpoint_collapse_certified,
        "constant_detail": (
            f"defect={defect_bound}; inverse={linear_inverse_bound}; "
            f"lipschitz={nonlinear_lipschitz_bound}; radius={remainder_ball_radius}"
        ),
        "contraction_detail": (
            f"q=B*L={contraction_factor}; slack=1-q={contraction_slack}"
        ),
        "self_map_detail": (
            f"B*D+q*R={self_map_bound}; R={remainder_ball_radius}; "
            f"margin=R-(B*D+q*R)={self_map_margin}"
        ),
        "component_detail": (
            f"component_count={len(component_inputs)}; "
            f"key_sets_match={key_sets_match}; "
            f"component_failures={tuple(component_failures)}; "
            f"envelope_failures={tuple(envelope_failures)}"
        ),
        "endpoint_detail": (
            f"endpoint_component={endpoint_component}; "
            f"tail={endpoint_input.first_shell_tail_bound if endpoint_input is not None else np.inf}; "
            f"shell_ratio={endpoint_input.shell_ratio if endpoint_input is not None else np.inf}"
        ),
        "detail": (
            f"header_certified={header_certified}; constants_finite={constants_finite}; "
            f"contraction={contraction_certified}; self_map={self_map_certified}; "
            f"component_inputs={component_inputs_certified}"
        ),
    }


def _generalized_remainder_shell_majorant_for_checker(
    *,
    remainder_ball_radius: float,
    effective_exponent: float,
    initial_radius: float,
    shell_contraction: float,
    analytic_disk_fraction: float,
) -> tuple[float, float]:
    remainder_ball_radius = float(remainder_ball_radius)
    effective_exponent = float(effective_exponent)
    initial_radius = float(initial_radius)
    shell_contraction = float(shell_contraction)
    analytic_disk_fraction = float(analytic_disk_fraction)
    if not (
        np.isfinite(remainder_ball_radius)
        and remainder_ball_radius >= 0.0
        and np.isfinite(effective_exponent)
        and np.isfinite(initial_radius)
        and initial_radius > 0.0
        and np.isfinite(shell_contraction)
        and 0.0 < shell_contraction < 1.0
        and np.isfinite(analytic_disk_fraction)
        and 0.0 < analytic_disk_fraction < 1.0
    ):
        raise ValueError("invalid generalized remainder shell-majorant inputs")
    radius_factor = (
        initial_radius * (1.0 + analytic_disk_fraction)
        if effective_exponent >= 0.0
        else initial_radius * (1.0 - analytic_disk_fraction)
    )
    return (
        float(remainder_ball_radius * radius_factor**effective_exponent),
        float(shell_contraction**effective_exponent),
    )


def _fuchsian_log_term_from_certificate(
    term: FuchsianLogTermCertificate,
) -> FuchsianLogTerm:
    return FuchsianLogTerm(
        power=float(term.power),
        coefficients_by_log_power={
            int(log_power): np.asarray(coefficient, dtype=float)
            for log_power, coefficient in term.coefficients_by_log_power
        },
        selector_basis=tuple(
            np.asarray(basis, dtype=float) for basis in term.selector_basis
        ),
    )


def _fuchsian_log_terms_finite(
    terms: tuple[FuchsianLogTermCertificate, ...],
    expected_shape: tuple[int, ...],
) -> bool:
    for term in terms:
        if not np.isfinite(term.power) or term.power <= 0.0:
            return False
        for log_power, coefficient in term.coefficients_by_log_power:
            array = np.asarray(coefficient, dtype=float)
            if int(log_power) < 0 or array.shape != expected_shape:
                return False
            if not np.all(np.isfinite(array)):
                return False
        for basis in term.selector_basis:
            basis_array = np.asarray(basis, dtype=float)
            if basis_array.shape != expected_shape:
                return False
            if not np.all(np.isfinite(basis_array)):
                return False
    return True


def _fuchsian_log_term_coefficient_sup_norm(term: FuchsianLogTerm) -> float:
    values = [
        _array_sup_norm(np.asarray(coefficient, dtype=float))
        for coefficient in term.coefficients_by_log_power.values()
    ]
    return float(max(values, default=0.0))


def _check_fuchsian_primitive_cauchy_inputs(
    inputs: FiniteFuchsianLogPrimitiveCauchyInputsCertificate,
    *,
    branch: FiniteFuchsianLogBranch | None = None,
) -> tuple[bool, float, str]:
    component_inputs = dict(inputs.component_inputs)
    derivative_orders = dict(inputs.component_derivative_orders)
    component_multipliers = dict(inputs.component_multipliers)
    header_certified = bool(
        np.isfinite(inputs.initial_radius)
        and inputs.initial_radius > 0.0
        and np.isfinite(inputs.shell_contraction)
        and 0.0 < inputs.shell_contraction < 1.0
        and np.isfinite(inputs.analytic_disk_fraction)
        and 0.0 < inputs.analytic_disk_fraction < 1.0
        and np.isfinite(inputs.log_growth_factor)
        and inputs.log_growth_factor > 1.0
        and component_inputs
    )
    key_sets_match = bool(
        component_inputs
        and set(component_inputs) == set(derivative_orders)
        and set(component_inputs) == set(component_multipliers)
    )
    component_failures: list[str] = []
    envelope_failures: list[str] = []
    first_shell_bounds: list[float] = []
    for component, serialized in component_inputs.items():
        primitive = PrimitiveCauchyTailInput(
            majorant_initial=float(serialized.majorant_initial),
            majorant_growth=float(serialized.majorant_growth),
            step_ratio_bound=float(serialized.step_ratio_bound),
            retained_order_initial=int(serialized.retained_order_initial),
            retained_order_increment=int(serialized.retained_order_increment),
        )
        recomputed_first_shell = primitive.first_shell_tail_bound
        recomputed_shell_ratio = primitive.shell_ratio
        first_shell_bounds.append(float(recomputed_first_shell))
        first_shell_matches = _finite_close(
            float(serialized.first_shell_tail_bound),
            recomputed_first_shell,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        shell_ratio_matches = _finite_close(
            float(serialized.shell_ratio),
            recomputed_shell_ratio,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        derivative_order = derivative_orders.get(component, -1)
        multiplier = component_multipliers.get(component, np.inf)
        branch_envelope_covers = True
        if branch is not None:
            try:
                expected_initial, expected_growth = (
                    _finite_fuchsian_log_branch_derivative_envelope(
                        branch,
                        derivative_order=int(derivative_order),
                        initial_radius=float(inputs.initial_radius),
                        shell_contraction=float(inputs.shell_contraction),
                        analytic_disk_fraction=float(inputs.analytic_disk_fraction),
                        log_growth_factor=float(inputs.log_growth_factor),
                    )
                )
                expected_initial *= float(multiplier)
                branch_envelope_covers = bool(
                    np.isfinite(expected_initial)
                    and np.isfinite(expected_growth)
                    and float(serialized.majorant_initial) + 1.0e-15
                    >= expected_initial
                    and float(serialized.majorant_growth) + 1.0e-15
                    >= expected_growth
                )
            except (FloatingPointError, TypeError, ValueError):
                branch_envelope_covers = False
            if not branch_envelope_covers:
                envelope_failures.append(component)
        if not (
            primitive.certified
            and first_shell_matches
            and shell_ratio_matches
            and int(derivative_order) >= 0
            and np.isfinite(float(multiplier))
            and float(multiplier) >= 0.0
            and branch_envelope_covers
        ):
            component_failures.append(component)
    tail_bound = float(max(first_shell_bounds, default=np.inf))
    certified = bool(
        header_certified
        and key_sets_match
        and np.isfinite(tail_bound)
        and not envelope_failures
        and not component_failures
    )
    detail = (
        f"component_count={len(component_inputs)}; "
        f"tail_bound={tail_bound}; "
        f"header_certified={header_certified}; "
        f"key_sets_match={key_sets_match}; "
        f"envelope_failures={tuple(envelope_failures)}; "
        f"component_failures={tuple(component_failures)}"
    )
    return certified, tail_bound, detail


def _check_fuchsian_primitive_cauchy_residual_tail(
    inputs: FiniteFuchsianLogPrimitiveCauchyInputsCertificate,
    *,
    residual_tolerance: float,
) -> tuple[bool, float, str]:
    component_inputs = dict(inputs.component_inputs)
    derivative_orders = dict(inputs.component_derivative_orders)
    component_multipliers = dict(inputs.component_multipliers)
    required_components = ("lifted_residual", "physical_residual")
    missing_components = tuple(
        component
        for component in required_components
        if component not in component_inputs
    )
    metadata_failures: list[str] = []
    component_failures: list[str] = []
    residual_bounds: list[float] = []
    for component in required_components:
        serialized = component_inputs.get(component)
        if serialized is None:
            continue
        derivative_order = int(derivative_orders.get(component, -1))
        multiplier = float(component_multipliers.get(component, np.inf))
        if not (
            derivative_order == 0
            and np.isfinite(multiplier)
            and multiplier >= 1.0
        ):
            metadata_failures.append(component)
            continue
        primitive = PrimitiveCauchyTailInput(
            majorant_initial=float(serialized.majorant_initial),
            majorant_growth=float(serialized.majorant_growth),
            step_ratio_bound=float(serialized.step_ratio_bound),
            retained_order_initial=int(serialized.retained_order_initial),
            retained_order_increment=int(serialized.retained_order_increment),
        )
        recomputed_first_shell = primitive.first_shell_tail_bound
        first_shell_matches = _finite_close(
            float(serialized.first_shell_tail_bound),
            recomputed_first_shell,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        if not (primitive.certified and first_shell_matches):
            component_failures.append(component)
            continue
        residual_bounds.append(float(recomputed_first_shell))
    residual_tail_bound = float(max(residual_bounds, default=np.inf))
    tolerance_ok = bool(
        np.isfinite(residual_tolerance)
        and residual_tolerance >= 0.0
        and np.isfinite(residual_tail_bound)
        and residual_tail_bound <= residual_tolerance
    )
    certified = bool(
        not missing_components
        and not metadata_failures
        and not component_failures
        and tolerance_ok
    )
    detail = (
        f"required_components={required_components}; "
        f"missing_components={missing_components}; "
        f"metadata_failures={tuple(metadata_failures)}; "
        f"component_failures={tuple(component_failures)}"
    )
    return certified, residual_tail_bound, detail


def _check_fuchsian_endpoint_collapse_envelope(
    inputs: FiniteFuchsianLogPrimitiveCauchyInputsCertificate,
) -> tuple[bool, float, str]:
    component_inputs = dict(inputs.component_inputs)
    derivative_orders = dict(inputs.component_derivative_orders)
    value_input = component_inputs.get("value")
    if value_input is None:
        return (
            False,
            np.inf,
            "missing value component for endpoint-collapse envelope",
        )
    initial_radius = float(inputs.initial_radius)
    shell_contraction = float(inputs.shell_contraction)
    value_majorant_growth = float(value_input.majorant_growth)
    value_tail_shell_ratio = float(value_input.shell_ratio)
    value_first_shell_tail_bound = float(value_input.first_shell_tail_bound)
    value_derivative_order = int(derivative_orders.get("value", -1))
    first_shell_shape_bound = float(
        float(value_input.majorant_initial) + value_first_shell_tail_bound
    )
    first_shell_position_bound = float(initial_radius**2 * first_shell_shape_bound)
    finite_part_position_ratio = float(shell_contraction**2 * value_majorant_growth)
    tail_position_ratio = float(shell_contraction**2 * value_tail_shell_ratio)
    primitive = PrimitiveCauchyTailInput(
        majorant_initial=float(value_input.majorant_initial),
        majorant_growth=value_majorant_growth,
        step_ratio_bound=float(value_input.step_ratio_bound),
        retained_order_initial=int(value_input.retained_order_initial),
        retained_order_increment=int(value_input.retained_order_increment),
    )
    certified = bool(
        primitive.certified
        and value_derivative_order == 0
        and np.isfinite(initial_radius)
        and initial_radius > 0.0
        and np.isfinite(shell_contraction)
        and 0.0 < shell_contraction < 1.0
        and np.isfinite(first_shell_shape_bound)
        and first_shell_shape_bound >= 0.0
        and np.isfinite(first_shell_position_bound)
        and np.isfinite(finite_part_position_ratio)
        and finite_part_position_ratio < 1.0
        and np.isfinite(tail_position_ratio)
        and tail_position_ratio < 1.0
    )
    detail = (
        f"value_derivative_order={value_derivative_order}; "
        f"first_shell_shape_bound={first_shell_shape_bound}; "
        f"finite_part_position_ratio={finite_part_position_ratio}; "
        f"tail_position_ratio={tail_position_ratio}"
    )
    return certified, first_shell_position_bound, detail


def _check_fuchsian_supported_scale_finite_energy_matching(
    branch: FiniteFuchsianLogBranch,
) -> tuple[bool, float, str]:
    masses = np.asarray(branch.masses, dtype=float)
    central_shape = _interval_point_array(np.asarray(branch.central_shape, dtype=float))
    unsupported_terms = tuple(
        index
        for index, term in enumerate(branch.terms)
        if _fuchsian_log_term_coefficient_sup_norm(term) > 1.0e-15
    )
    try:
        inertia = _interval_weighted_squared_norm(central_shape, masses)
        potential = _interval_pairwise_newtonian_potential(central_shape, masses)
        leading_gap = inertia.scale(2.0 / 9.0) - potential
        finite_energy = inertia.scale((10.0 / 9.0) * float(branch.scale_coefficient))
    except (FloatingPointError, ValueError):
        return (
            False,
            np.inf,
            "interval finite-energy computation failed",
        )
    leading_gap_bound = _interval_abs_sup(leading_gap)
    finite_energy_bound = _interval_abs_sup(finite_energy)
    certified = bool(
        not unsupported_terms
        and np.isfinite(finite_energy_bound)
        and np.isfinite(float(branch.scale_coefficient))
        and leading_gap_bound <= 1.0e-10
    )
    detail = (
        f"leading_energy_gap_bound={leading_gap_bound}; "
        f"unsupported_non_scale_terms={unsupported_terms}"
    )
    return certified, finite_energy_bound, detail


def _finite_close(
    left: float,
    right: float,
    *,
    rtol: float,
    atol: float,
) -> bool:
    return bool(
        np.isfinite(left)
        and np.isfinite(right)
        and abs(left - right) <= max(atol, rtol * max(abs(left), abs(right)))
    )


def _fuchsian_log_total_collision_isolation_from_certificate(
    certificate: TotalCollisionFuchsianStopChartCertificate,
) -> FiniteFuchsianLogTotalCollisionIsolationCertificate:
    return certify_finite_fuchsian_log_total_collision_isolation(
        _fuchsian_log_branch_from_certificate(certificate),
        radius=float(certificate.isolation_radius),
    )


def _finite_nonempty_interval(interval: tuple[float, float]) -> bool:
    return bool(
        len(interval) == 2
        and np.isfinite(interval[0])
        and np.isfinite(interval[1])
        and interval[0] <= interval[1]
    )


def _interval_contains_interval(
    outer: tuple[float, float],
    inner: tuple[float, float],
    *,
    tolerance: float = 0.0,
) -> bool:
    return bool(
        _finite_nonempty_interval(outer)
        and _finite_nonempty_interval(inner)
        and outer[0] - tolerance <= inner[0]
        and inner[1] <= outer[1] + tolerance
    )


def _exact_rational_interval_from_floats(
    interval: tuple[float, float],
) -> tuple[Fraction, Fraction] | None:
    if not _finite_nonempty_interval(interval):
        return None
    lower = Fraction(float(interval[0]))
    upper = Fraction(float(interval[1]))
    if lower > upper:
        return None
    return lower, upper


def _exact_rational_chart_chain_time_coverage(
    chart_intervals: tuple[tuple[float, float], ...],
    target_interval: tuple[float, float],
) -> tuple[bool, str]:
    target = _exact_rational_interval_from_floats(target_interval)
    charts = tuple(
        rational_interval
        for rational_interval in (
            _exact_rational_interval_from_floats(interval)
            for interval in chart_intervals
        )
        if rational_interval is not None
    )
    if target is None or len(charts) != len(chart_intervals) or not charts:
        return (
            False,
            (
                "target_interval or chart physical-time intervals are not "
                "finite nonempty serialized intervals"
            ),
        )
    no_gaps = all(
        charts[index][0] <= charts[index + 1][0]
        and charts[index + 1][0] <= charts[index][1]
        for index in range(len(charts) - 1)
    )
    covered = (charts[0][0], max(interval[1] for interval in charts))
    covered_target = covered[0] <= target[0] and target[1] <= covered[1]
    certified = bool(no_gaps and covered_target)
    return (
        certified,
        (
            "exact rational interval arithmetic over serialized binary-float "
            f"time endpoints; no_gaps={no_gaps}; "
            f"covered_interval=({covered[0]}, {covered[1]}); "
            f"target_interval=({target[0]}, {target[1]})"
        ),
    )


def _exact_rational_branch_union_interval_aggregation(
    aggregate_interval: tuple[float, float],
    leaf_intervals: tuple[tuple[float, float], ...],
) -> tuple[bool, str]:
    aggregate = _exact_rational_interval_from_floats(aggregate_interval)
    leaves = tuple(
        rational_interval
        for rational_interval in (
            _exact_rational_interval_from_floats(interval)
            for interval in leaf_intervals
        )
        if rational_interval is not None
    )
    if (
        aggregate is None
        or len(leaves) != len(leaf_intervals)
        or not leaves
    ):
        return (
            False,
            (
                "aggregate or leaf target intervals are not finite nonempty "
                "serialized intervals"
            ),
        )
    contains_leaves = all(
        aggregate[0] <= leaf[0] and leaf[1] <= aggregate[1]
        for leaf in leaves
    )
    leaf_hull = (
        min(leaf[0] for leaf in leaves),
        max(leaf[1] for leaf in leaves),
    )
    return (
        bool(contains_leaves),
        (
            "exact rational interval arithmetic over serialized binary-float "
            f"target endpoints; contains_leaves={contains_leaves}; "
            f"aggregate=({aggregate[0]}, {aggregate[1]}); "
            f"leaf_hull=({leaf_hull[0]}, {leaf_hull[1]}); "
            f"leaf_count={len(leaves)}"
        ),
    )


def _exact_rational_transition_state_continuity(
    source_chart: OrdinaryTaylorChartCertificate
    | PlanarLeviCivitaBinaryChartCertificate
    | SpatialKSBinaryChartCertificate,
    target_chart: OrdinaryTaylorChartCertificate
    | PlanarLeviCivitaBinaryChartCertificate
    | SpatialKSBinaryChartCertificate,
    *,
    source_parameter: float,
    target_parameter: float,
    handoff_time: float,
    position_tolerance: float,
    velocity_tolerance: float,
    physical_time_tolerance: float,
) -> tuple[bool, str]:
    try:
        source_parameter_q = Fraction(float(source_parameter))
        target_parameter_q = Fraction(float(target_parameter))
        handoff_time_q = Fraction(float(handoff_time))
        position_tolerance_q = Fraction(float(position_tolerance))
        velocity_tolerance_q = Fraction(float(velocity_tolerance))
        physical_time_tolerance_q = Fraction(float(physical_time_tolerance))
    except (OverflowError, ValueError):
        return False, "transition parameters or tolerances are not finite rationalizable floats"
    try:
        source_time = _exact_rational_chart_physical_time_at_parameter(
            source_chart,
            source_parameter_q,
        )
        target_time = _exact_rational_chart_physical_time_at_parameter(
            target_chart,
            target_parameter_q,
        )
        source_q, source_v = _exact_rational_chart_projected_state_at_parameter(
            source_chart,
            source_parameter_q,
        )
        target_q, target_v = _exact_rational_chart_projected_state_at_parameter(
            target_chart,
            target_parameter_q,
        )
        max_time_gap = max(
            abs(source_time - handoff_time_q),
            abs(target_time - handoff_time_q),
        )
        max_position_gap = _fraction_array_max_abs_difference(source_q, target_q)
        max_velocity_gap = _fraction_array_max_abs_difference(source_v, target_v)
    except (ZeroDivisionError, ValueError, TypeError):
        return False, "exact rational transition state evaluation failed"
    certified = bool(
        max_time_gap <= physical_time_tolerance_q
        and max_position_gap <= position_tolerance_q
        and max_velocity_gap <= velocity_tolerance_q
    )
    return (
        certified,
        (
            "exact rational arithmetic over serialized binary-float chart "
            "coefficients, parameters, masses, and tolerances; "
            f"max_time_gap={max_time_gap}; "
            f"max_position_gap={max_position_gap}; "
            f"max_velocity_gap={max_velocity_gap}; "
            f"physical_time_tolerance={physical_time_tolerance_q}; "
            f"position_tolerance={position_tolerance_q}; "
            f"velocity_tolerance={velocity_tolerance_q}"
        ),
    )


def _exact_rational_chart_physical_time_at_parameter(
    chart: OrdinaryTaylorChartCertificate
    | PlanarLeviCivitaBinaryChartCertificate
    | SpatialKSBinaryChartCertificate,
    parameter: Fraction,
) -> Fraction:
    if isinstance(chart, OrdinaryTaylorChartCertificate):
        physical = _exact_rational_interval_from_floats(chart.physical_time_interval)
        domain = _exact_rational_interval_from_floats(chart.parameter_interval)
        if physical is None or domain is None:
            raise ValueError("ordinary chart time interval is invalid")
        parameter_width = domain[1] - domain[0]
        physical_width = physical[1] - physical[0]
        if parameter_width == 0:
            if parameter != domain[0]:
                raise ValueError("parameter is outside point ordinary chart")
            return physical[0]
        return physical[0] + (parameter - domain[0]) * physical_width / parameter_width
    if isinstance(chart, PlanarLeviCivitaBinaryChartCertificate):
        return _evaluate_fraction_coefficients(chart.physical_time_coefficients, parameter)
    if isinstance(chart, SpatialKSBinaryChartCertificate):
        return _evaluate_fraction_coefficients(chart.physical_time_coefficients, parameter)
    raise TypeError(f"unsupported chart type: {type(chart)!r}")


def _exact_rational_chart_projected_state_at_parameter(
    chart: OrdinaryTaylorChartCertificate
    | PlanarLeviCivitaBinaryChartCertificate
    | SpatialKSBinaryChartCertificate,
    parameter: Fraction,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(chart, OrdinaryTaylorChartCertificate):
        return (
            _evaluate_fraction_coefficients(chart.position_coefficients, parameter),
            _evaluate_fraction_coefficients(chart.velocity_coefficients, parameter),
        )
    if isinstance(chart, PlanarLeviCivitaBinaryChartCertificate):
        return _exact_rational_planar_lc_projected_state_at_parameter(
            chart,
            parameter,
        )
    if isinstance(chart, SpatialKSBinaryChartCertificate):
        return _exact_rational_spatial_ks_projected_state_at_parameter(
            chart,
            parameter,
        )
    raise TypeError(f"unsupported chart type: {type(chart)!r}")


def _exact_rational_planar_lc_projected_state_at_parameter(
    chart: PlanarLeviCivitaBinaryChartCertificate,
    parameter: Fraction,
) -> tuple[np.ndarray, np.ndarray]:
    masses = tuple(Fraction(float(value)) for value in chart.masses)
    if len(masses) != 3:
        raise ValueError("LC chart masses must have length 3")
    first, second = chart.pair
    third = ({0, 1, 2} - {first, second}).pop()
    pair_mass = masses[first] + masses[second]
    if pair_mass == 0:
        raise ValueError("invalid pair mass")
    alpha = masses[second] / pair_mass
    beta = masses[first] / pair_mass
    z = _evaluate_fraction_coefficients(chart.z_coefficients, parameter)
    z_velocity = _evaluate_fraction_coefficients(
        chart.z_velocity_coefficients,
        parameter,
    )
    binary_center = _evaluate_fraction_coefficients(
        chart.binary_center_coefficients,
        parameter,
    )
    binary_center_velocity = _evaluate_fraction_coefficients(
        chart.binary_center_velocity_coefficients,
        parameter,
    )
    third_offset = _evaluate_fraction_coefficients(
        chart.third_offset_coefficients,
        parameter,
    )
    third_offset_velocity = _evaluate_fraction_coefficients(
        chart.third_offset_velocity_coefficients,
        parameter,
    )
    relative_position = _exact_rational_lc_square(z)
    relative_velocity = _exact_rational_lc_velocity(z, z_velocity)
    positions = _fraction_zero_array((3, 2))
    velocities = _fraction_zero_array((3, 2))
    positions[first] = binary_center - alpha * relative_position
    positions[second] = binary_center + beta * relative_position
    positions[third] = binary_center + third_offset
    velocities[first] = binary_center_velocity - alpha * relative_velocity
    velocities[second] = binary_center_velocity + beta * relative_velocity
    velocities[third] = binary_center_velocity + third_offset_velocity
    return positions, velocities


def _exact_rational_spatial_ks_projected_state_at_parameter(
    chart: SpatialKSBinaryChartCertificate,
    parameter: Fraction,
) -> tuple[np.ndarray, np.ndarray]:
    masses = tuple(Fraction(float(value)) for value in chart.masses)
    if len(masses) != 3:
        raise ValueError("KS chart masses must have length 3")
    first, second = chart.pair
    third = ({0, 1, 2} - {first, second}).pop()
    pair_mass = masses[first] + masses[second]
    if pair_mass == 0:
        raise ValueError("invalid pair mass")
    alpha = masses[second] / pair_mass
    beta = masses[first] / pair_mass
    u = _evaluate_fraction_coefficients(chart.u_coefficients, parameter)
    u_velocity = _evaluate_fraction_coefficients(
        chart.u_velocity_coefficients,
        parameter,
    )
    binary_center = _evaluate_fraction_coefficients(
        chart.binary_center_coefficients,
        parameter,
    )
    binary_center_velocity = _evaluate_fraction_coefficients(
        chart.binary_center_velocity_coefficients,
        parameter,
    )
    third_offset = _evaluate_fraction_coefficients(
        chart.third_offset_coefficients,
        parameter,
    )
    third_offset_velocity = _evaluate_fraction_coefficients(
        chart.third_offset_velocity_coefficients,
        parameter,
    )
    relative_position = _exact_rational_ks_project(u)
    relative_velocity = _exact_rational_ks_velocity(u, u_velocity)
    positions = _fraction_zero_array((3, 3))
    velocities = _fraction_zero_array((3, 3))
    positions[first] = binary_center - alpha * relative_position
    positions[second] = binary_center + beta * relative_position
    positions[third] = binary_center + third_offset
    velocities[first] = binary_center_velocity - alpha * relative_velocity
    velocities[second] = binary_center_velocity + beta * relative_velocity
    velocities[third] = binary_center_velocity + third_offset_velocity
    return positions, velocities


def _evaluate_fraction_coefficients(
    coefficients: object,
    parameter: Fraction,
) -> np.ndarray | Fraction:
    values = np.asarray(coefficients, dtype=float)
    if values.ndim == 0:
        return Fraction(float(values))
    if values.ndim == 1:
        out = Fraction(0)
        for coefficient in values[::-1]:
            out = out * parameter + Fraction(float(coefficient))
        return out
    out = _fraction_zero_array(values.shape[1:])
    for coefficient in values[::-1]:
        out = out * parameter + _fraction_array(coefficient)
    return out


def _fraction_array(values: object) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    out = np.empty(numeric.shape, dtype=object)
    for index in np.ndindex(numeric.shape):
        out[index] = Fraction(float(numeric[index]))
    return out


def _fraction_zero_array(shape: tuple[int, ...]) -> np.ndarray:
    out = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        out[index] = Fraction(0)
    return out


def _exact_rational_lc_square(z: np.ndarray) -> np.ndarray:
    x, y = z
    return np.array((x * x - y * y, 2 * x * y), dtype=object)


def _exact_rational_lc_velocity(
    z: np.ndarray,
    z_velocity: np.ndarray,
) -> np.ndarray:
    x, y = z
    vx, vy = z_velocity
    rho = x * x + y * y
    if rho == 0:
        raise ValueError("LC physical velocity is singular at binary collision")
    return np.array(
        (
            (2 * x * vx - 2 * y * vy) / rho,
            (2 * y * vx + 2 * x * vy) / rho,
        ),
        dtype=object,
    )


def _exact_rational_ks_project(u: np.ndarray) -> np.ndarray:
    a, b, c, d = u
    return np.array(
        (
            a * a - b * b - c * c + d * d,
            2 * (a * b - c * d),
            2 * (a * c + b * d),
        ),
        dtype=object,
    )


def _exact_rational_ks_velocity(
    u: np.ndarray,
    u_velocity: np.ndarray,
) -> np.ndarray:
    a, b, c, d = u
    va, vb, vc, vd = u_velocity
    rho = a * a + b * b + c * c + d * d
    if rho == 0:
        raise ValueError("KS physical velocity is singular at binary collision")
    return np.array(
        (
            (2 * a * va - 2 * b * vb - 2 * c * vc + 2 * d * vd) / rho,
            (2 * b * va + 2 * a * vb - 2 * d * vc - 2 * c * vd) / rho,
            (2 * c * va + 2 * d * vb + 2 * a * vc + 2 * b * vd) / rho,
        ),
        dtype=object,
    )


def _fraction_array_max_abs_difference(
    left: np.ndarray,
    right: np.ndarray,
) -> Fraction:
    if left.shape != right.shape:
        raise ValueError("fraction arrays have incompatible shapes")
    maximum = Fraction(0)
    for index in np.ndindex(left.shape):
        maximum = max(maximum, abs(left[index] - right[index]))
    return maximum


def _sampled_physical_time_contained(
    physical_time_coefficients: np.ndarray,
    parameter_interval: tuple[float, float],
    physical_time_interval: tuple[float, float],
    *,
    sample_count: int,
) -> bool:
    if physical_time_coefficients.ndim != 1:
        return False
    lower, upper = physical_time_interval
    for parameter in np.linspace(
        parameter_interval[0],
        parameter_interval[1],
        sample_count,
    ):
        value = _evaluate_scalar_coefficients(
            physical_time_coefficients,
            float(parameter),
        )
        if not (lower <= value <= upper):
            return False
    return True


def _interval_physical_time_contained(
    physical_time_coefficients: np.ndarray,
    parameter_interval: tuple[float, float],
    physical_time_interval: tuple[float, float],
    *,
    tolerance: float,
    pieces: int = 64,
) -> tuple[bool, tuple[float, float]]:
    """Enclose a serialized physical-time polynomial over the full chart slab.

    This is intentionally independent of the certificate sample count.  The
    checker subdivides the serialized parameter interval exactly as rational
    subintervals of the binary-float endpoints, then evaluates the polynomial
    with rational interval Horner arithmetic on every piece.
    """

    coefficients = np.asarray(physical_time_coefficients, dtype=float)
    if (
        coefficients.ndim != 1
        or not np.all(np.isfinite(coefficients))
        or not _finite_nonempty_interval(parameter_interval)
        or not _finite_nonempty_interval(physical_time_interval)
        or not np.isfinite(tolerance)
        or tolerance < 0.0
        or pieces <= 0
    ):
        return False, (float("inf"), float("inf"))
    try:
        parameter = RationalInterval.from_float_interval(
            float(parameter_interval[0]),
            float(parameter_interval[1]),
        )
        coefficient_intervals = tuple(
            RationalInterval.from_float_interval(float(coefficient))
            for coefficient in coefficients
        )
        width = parameter.upper - parameter.lower
        lower = None
        upper = None
        piece_count = int(pieces)
        for index in range(piece_count):
            left = parameter.lower + width * Fraction(index, piece_count)
            right = parameter.lower + width * Fraction(index + 1, piece_count)
            value = rational_interval_polynomial_eval(
                coefficient_intervals,
                RationalInterval(left, right),
            )
            lower = value.lower if lower is None else min(lower, value.lower)
            upper = value.upper if upper is None else max(upper, value.upper)
        if lower is None or upper is None:
            return False, (float("inf"), float("inf"))
        enclosure = RationalInterval(lower, upper)
        outward = enclosure.as_float_interval().as_tuple()
        tol = RationalInterval.from_float_interval(float(tolerance))
        allowed_lower = (
            RationalInterval.from_float_interval(float(physical_time_interval[0])).lower
            - tol.upper
        )
        allowed_upper = (
            RationalInterval.from_float_interval(float(physical_time_interval[1])).upper
            + tol.upper
        )
        contained = bool(
            enclosure.lower >= allowed_lower
            and enclosure.upper <= allowed_upper
        )
        return contained, outward
    except (FloatingPointError, OverflowError, ValueError):
        return False, (float("inf"), float("inf"))


def _ordinary_unit_physical_parameter_speed(
    parameter_interval: tuple[float, float],
    physical_time_interval: tuple[float, float],
    *,
    tolerance: float,
) -> bool:
    """Check that an ordinary chart parameter is physical time up to translation."""

    parameter_width = parameter_interval[1] - parameter_interval[0]
    physical_width = physical_time_interval[1] - physical_time_interval[0]
    if parameter_width < 0.0 or physical_width < 0.0:
        return False
    return bool(abs(parameter_width - physical_width) <= float(tolerance))


def _physical_time_in_chart(
    chart: OrdinaryTaylorChartCertificate,
    physical_time: float,
) -> bool:
    lower, upper = chart.physical_time_interval
    return bool(np.isfinite(physical_time) and lower <= physical_time <= upper)


def _time_in_interval(interval: tuple[float, float], value: float) -> bool:
    lower, upper = interval
    return bool(np.isfinite(value) and lower <= value <= upper)


def _parameter_in_interval(interval: tuple[float, float], value: float) -> bool:
    lower, upper = interval
    return bool(np.isfinite(value) and lower <= value <= upper)


def _physical_to_parameter(
    chart: OrdinaryTaylorChartCertificate,
    physical_time: float,
) -> float:
    physical_lower, physical_upper = chart.physical_time_interval
    parameter_lower, parameter_upper = chart.parameter_interval
    physical_width = physical_upper - physical_lower
    parameter_width = parameter_upper - parameter_lower
    if physical_width == 0.0:
        if physical_time != physical_lower:
            raise ValueError("handoff time is outside point chart")
        return float(parameter_lower)
    return float(
        parameter_lower
        + (physical_time - physical_lower) * parameter_width / physical_width
    )


def _chart_physical_time_at_parameter(
    chart: OrdinaryTaylorChartCertificate
    | PlanarLeviCivitaBinaryChartCertificate
    | SpatialKSBinaryChartCertificate,
    parameter: float,
) -> float:
    if isinstance(chart, OrdinaryTaylorChartCertificate):
        physical_lower, physical_upper = chart.physical_time_interval
        parameter_lower, parameter_upper = chart.parameter_interval
        parameter_width = parameter_upper - parameter_lower
        physical_width = physical_upper - physical_lower
        if parameter_width == 0.0:
            if parameter != parameter_lower:
                raise ValueError("parameter is outside point chart")
            return float(physical_lower)
        return float(physical_lower + (parameter - parameter_lower) * physical_width / parameter_width)
    if isinstance(chart, PlanarLeviCivitaBinaryChartCertificate):
        return _evaluate_scalar_coefficients(
            _coefficient_array(chart.physical_time_coefficients),
            parameter,
        )
    if isinstance(chart, SpatialKSBinaryChartCertificate):
        return _evaluate_scalar_coefficients(
            _coefficient_array(chart.physical_time_coefficients),
            parameter,
        )
    raise TypeError(f"unsupported chart type: {type(chart)!r}")


def _chart_projected_state_at_parameter(
    chart: OrdinaryTaylorChartCertificate
    | PlanarLeviCivitaBinaryChartCertificate
    | SpatialKSBinaryChartCertificate,
    parameter: float,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(chart, OrdinaryTaylorChartCertificate):
        q = _evaluate_coefficients(
            _coefficient_array(chart.position_coefficients),
            parameter,
        )
        v = _evaluate_coefficients(
            _coefficient_array(chart.velocity_coefficients),
            parameter,
        )
        return q, v
    if isinstance(chart, PlanarLeviCivitaBinaryChartCertificate):
        solution = _regularized_binary_solution_from_certificate(chart)
        state = solution.state_at(parameter)
        return regularized_binary_collision_chart_to_planar(state)
    if isinstance(chart, SpatialKSBinaryChartCertificate):
        solution = _spatial_ks_solution_from_certificate(chart)
        state = solution.state_at(parameter)
        return ks_binary_chart_to_spatial(state)
    raise TypeError(f"unsupported chart type: {type(chart)!r}")


def _initial_noncollision(positions: np.ndarray) -> bool:
    for i in range(3):
        for j in range(i + 1, 3):
            if not np.linalg.norm(positions[i] - positions[j]) > 0.0:
                return False
    return True


def _minimum_pair_distance(positions: np.ndarray) -> float:
    if np.asarray(positions).ndim != 2 or positions.shape[0] < 2:
        return float("inf")
    return float(
        min(
            np.linalg.norm(positions[j] - positions[i])
            for i in range(positions.shape[0])
            for j in range(i + 1, positions.shape[0])
        )
    )


def _max_coefficient_recurrence_residual(
    q: np.ndarray,
    v: np.ndarray,
    masses: np.ndarray,
) -> float:
    order = q.shape[0] - 1
    acc = acceleration_coefficients(q, masses, order - 1)
    position_residual = 0.0
    velocity_residual = 0.0
    for n in range(order):
        position_residual = max(
            position_residual,
            float(np.max(np.abs((n + 1) * q[n + 1] - v[n]))),
        )
        velocity_residual = max(
            velocity_residual,
            float(np.max(np.abs((n + 1) * v[n + 1] - acc[n]))),
        )
    return max(position_residual, velocity_residual)


def _max_regularized_binary_coefficient_residual(
    solution: RegularizedBinaryTaylorSolution,
) -> float:
    order = solution.order
    worst = 0.0
    for n in range(order):
        rhs = lc_regularized_rhs_coefficients(solution, n)
        worst = max(
            worst,
            _array_sup_norm((n + 1) * solution.z[n + 1] - rhs.z[n]),
            _array_sup_norm(
                (n + 1) * solution.z_velocity[n + 1] - rhs.z_velocity[n],
            ),
            abs(float((n + 1) * solution.pair_energy[n + 1] - rhs.pair_energy[n])),
            _array_sup_norm(
                (n + 1) * solution.binary_center[n + 1] - rhs.binary_center[n],
            ),
            _array_sup_norm(
                (n + 1) * solution.binary_center_velocity[n + 1]
                - rhs.binary_center_velocity[n],
            ),
            _array_sup_norm(
                (n + 1) * solution.third_offset[n + 1] - rhs.third_offset[n],
            ),
            _array_sup_norm(
                (n + 1) * solution.third_offset_velocity[n + 1]
                - rhs.third_offset_velocity[n],
            ),
            abs(float((n + 1) * solution.physical_time[n + 1] - rhs.physical_time[n])),
        )
    return float(worst)


def _max_interval_planar_lc_taylor_model_residual(
    solution: RegularizedBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    tail_bound: float,
) -> float:
    order = solution.order
    rhs = lc_regularized_rhs_coefficients(solution, order - 1)
    variable = RationalInterval.from_float_interval(
        float(parameter_interval[0]),
        float(parameter_interval[1]),
    )
    residual_blocks = (
        _lc_vector_residual_coefficients(solution.z, rhs.z, order),
        _lc_vector_residual_coefficients(
            solution.z_velocity,
            rhs.z_velocity,
            order,
        ),
        _lc_scalar_residual_coefficients(
            solution.pair_energy,
            rhs.pair_energy,
            order,
        ),
        _lc_vector_residual_coefficients(
            solution.binary_center,
            rhs.binary_center,
            order,
        ),
        _lc_vector_residual_coefficients(
            solution.binary_center_velocity,
            rhs.binary_center_velocity,
            order,
        ),
        _lc_vector_residual_coefficients(
            solution.third_offset,
            rhs.third_offset,
            order,
        ),
        _lc_vector_residual_coefficients(
            solution.third_offset_velocity,
            rhs.third_offset_velocity,
            order,
        ),
        _lc_scalar_residual_coefficients(
            solution.physical_time,
            rhs.physical_time,
            order,
        ),
    )
    worst = RationalInterval.point(0).upper
    for block in residual_blocks:
        block = np.asarray(block, dtype=float)
        if block.ndim == 1:
            worst = max(
                worst,
                _rational_interval_polynomial_abs_sup(block, variable),
            )
        else:
            for index in np.ndindex(block.shape[1:]):
                worst = max(
                    worst,
                    _rational_interval_polynomial_abs_sup(
                        block[(slice(None), *index)],
                        variable,
                    ),
                )
    bound = worst + RationalInterval.from_float_interval(
        max(0.0, float(tail_bound))
    ).upper
    return float(np.nextafter(float(bound), np.inf))


def _max_interval_planar_lc_projected_newton_residual(
    solution: RegularizedBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    tail_bound: float,
    rho_lower_bound: float,
) -> float:
    """Conservative interval check for projected Newton residuals.

    This evaluates the LC chart projection and Newton acceleration over the
    whole parameter interval using interval arithmetic.  It is intentionally a
    small trusted-checker slice: it only certifies slabs whose entire LC rho
    interval is away from the binary collision floor.
    """

    variable = FloatInterval(float(parameter_interval[0]), float(parameter_interval[1]))
    z = interval_array_series_eval(solution.z, variable)
    z_velocity = interval_array_series_eval(solution.z_velocity, variable)
    pair_energy = interval_polynomial_eval(solution.pair_energy, variable)
    binary_center = interval_array_series_eval(solution.binary_center, variable)
    binary_center_velocity = interval_array_series_eval(
        solution.binary_center_velocity,
        variable,
    )
    third_offset = interval_array_series_eval(solution.third_offset, variable)
    third_offset_velocity = interval_array_series_eval(
        solution.third_offset_velocity,
        variable,
    )
    rho = _interval_dot(z, z)
    if rho.lower <= max(0.0, float(rho_lower_bound)):
        return float("inf")

    center_acceleration, third_offset_acceleration, relative_perturbation = (
        _interval_planar_lc_analytic_coordinate_accelerations(
            solution,
            z,
            binary_center,
            third_offset,
        )
    )
    z_acceleration = _interval_vector_add(
        _interval_vector_scale_by_interval(z, pair_energy.scale(0.5)),
        _interval_vector_scale_by_interval(
            _interval_lc_transpose_matvec(z, relative_perturbation),
            rho.scale(0.25),
        ),
    )
    projected = _interval_planar_lc_projected_accelerations(
        solution,
        z,
        z_velocity,
        z_acceleration,
        center_acceleration,
        third_offset_acceleration,
    )
    positions = _interval_planar_lc_positions(
        solution,
        z,
        binary_center,
        third_offset,
    )
    newton = _interval_newton_accelerations(positions, solution.masses)
    worst = 0.0
    for index in np.ndindex(projected.shape):
        worst = max(worst, _interval_abs_sup(projected[index] - newton[index]))
    return float(worst + max(0.0, float(tail_bound)))


def _interval_planar_lc_analytic_coordinate_accelerations(
    solution: RegularizedBinaryTaylorSolution,
    z: np.ndarray,
    binary_center: np.ndarray,
    third_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second = solution.pair
    third = 3 - first - second
    masses = np.asarray(solution.masses, dtype=float)
    pair_mass = float(masses[first] + masses[second])
    alpha = float(masses[second] / pair_mass)
    beta = float(masses[first] / pair_mass)
    relative_position = _interval_lc_square(z)
    from_first_to_third = _interval_vector_add(
        third_offset,
        _interval_vector_scale(relative_position, alpha),
    )
    from_second_to_third = _interval_vector_sub(
        third_offset,
        _interval_vector_scale(relative_position, beta),
    )
    field_first = _interval_inverse_square_field(from_first_to_third)
    field_second = _interval_inverse_square_field(from_second_to_third)
    binary_center_acceleration = _interval_vector_scale(
        _interval_vector_add(
            _interval_vector_scale(field_first, float(masses[first])),
            _interval_vector_scale(field_second, float(masses[second])),
        ),
        float(masses[third] / pair_mass),
    )
    third_acceleration = _interval_vector_scale(
        _interval_vector_add(
            _interval_vector_scale(field_first, float(masses[first])),
            _interval_vector_scale(field_second, float(masses[second])),
        ),
        -1.0,
    )
    third_offset_acceleration = _interval_vector_sub(
        third_acceleration,
        binary_center_acceleration,
    )
    relative_perturbation = _interval_vector_scale(
        _interval_vector_sub(field_second, field_first),
        float(masses[third]),
    )
    return (
        binary_center_acceleration,
        third_offset_acceleration,
        relative_perturbation,
    )


def _interval_planar_lc_positions(
    solution: RegularizedBinaryTaylorSolution,
    z: np.ndarray,
    binary_center: np.ndarray,
    third_offset: np.ndarray,
) -> np.ndarray:
    first, second = solution.pair
    third = 3 - first - second
    masses = np.asarray(solution.masses, dtype=float)
    pair_mass = float(masses[first] + masses[second])
    relative_position = _interval_lc_square(z)
    positions = _interval_zero_array((3, 2))
    positions[first] = _interval_vector_sub(
        binary_center,
        _interval_vector_scale(relative_position, float(masses[second] / pair_mass)),
    )
    positions[second] = _interval_vector_add(
        binary_center,
        _interval_vector_scale(relative_position, float(masses[first] / pair_mass)),
    )
    positions[third] = _interval_vector_add(binary_center, third_offset)
    return positions


def _interval_planar_lc_projected_accelerations(
    solution: RegularizedBinaryTaylorSolution,
    z: np.ndarray,
    z_velocity: np.ndarray,
    z_acceleration: np.ndarray,
    binary_center_acceleration: np.ndarray,
    third_offset_acceleration: np.ndarray,
) -> np.ndarray:
    first, second = solution.pair
    third = 3 - first - second
    masses = np.asarray(solution.masses, dtype=float)
    pair_mass = float(masses[first] + masses[second])
    relative_acceleration = _interval_relative_acceleration_from_lc_acceleration(
        z,
        z_velocity,
        z_acceleration,
    )
    acceleration = _interval_zero_array((3, 2))
    acceleration[first] = _interval_vector_sub(
        binary_center_acceleration,
        _interval_vector_scale(relative_acceleration, float(masses[second] / pair_mass)),
    )
    acceleration[second] = _interval_vector_add(
        binary_center_acceleration,
        _interval_vector_scale(relative_acceleration, float(masses[first] / pair_mass)),
    )
    acceleration[third] = _interval_vector_add(
        binary_center_acceleration,
        third_offset_acceleration,
    )
    return acceleration


def _lc_vector_residual_coefficients(
    values: np.ndarray,
    rhs_values: np.ndarray,
    order: int,
) -> np.ndarray:
    return np.asarray(
        [
            (n + 1) * values[n + 1] - rhs_values[n]
            for n in range(order)
        ],
        dtype=float,
    )


def _lc_scalar_residual_coefficients(
    values: np.ndarray,
    rhs_values: np.ndarray,
    order: int,
) -> np.ndarray:
    return np.asarray(
        [
            float((n + 1) * values[n + 1] - rhs_values[n])
            for n in range(order)
        ],
        dtype=float,
    )


def _max_spatial_ks_coefficient_residual(
    solution: SpatialKSBinaryTaylorSolution,
) -> float:
    order = solution.order
    worst = 0.0
    for n in range(order):
        rhs = ks_regularized_rhs_coefficients(solution, n)
        worst = max(
            worst,
            _array_sup_norm((n + 1) * solution.u[n + 1] - rhs.u[n]),
            _array_sup_norm(
                (n + 1) * solution.u_velocity[n + 1] - rhs.u_velocity[n],
            ),
            abs(float((n + 1) * solution.pair_energy[n + 1] - rhs.pair_energy[n])),
            _array_sup_norm(
                (n + 1) * solution.binary_center[n + 1] - rhs.binary_center[n],
            ),
            _array_sup_norm(
                (n + 1) * solution.binary_center_velocity[n + 1]
                - rhs.binary_center_velocity[n],
            ),
            _array_sup_norm(
                (n + 1) * solution.third_offset[n + 1] - rhs.third_offset[n],
            ),
            _array_sup_norm(
                (n + 1) * solution.third_offset_velocity[n + 1]
                - rhs.third_offset_velocity[n],
            ),
            abs(float((n + 1) * solution.physical_time[n + 1] - rhs.physical_time[n])),
        )
    return float(worst)


def _max_spatial_ks_relative_coefficient_residual(
    solution: SpatialKSBinaryTaylorSolution,
) -> float:
    order = solution.order
    worst = 0.0
    for n in range(order):
        rhs = ks_regularized_rhs_coefficients(solution, n)
        blocks = (
            ((n + 1) * solution.u[n + 1], rhs.u[n]),
            ((n + 1) * solution.u_velocity[n + 1], rhs.u_velocity[n]),
            (
                np.array([(n + 1) * solution.pair_energy[n + 1]], dtype=float),
                np.array([rhs.pair_energy[n]], dtype=float),
            ),
            ((n + 1) * solution.binary_center[n + 1], rhs.binary_center[n]),
            (
                (n + 1) * solution.binary_center_velocity[n + 1],
                rhs.binary_center_velocity[n],
            ),
            ((n + 1) * solution.third_offset[n + 1], rhs.third_offset[n]),
            (
                (n + 1) * solution.third_offset_velocity[n + 1],
                rhs.third_offset_velocity[n],
            ),
            (
                np.array([(n + 1) * solution.physical_time[n + 1]], dtype=float),
                np.array([rhs.physical_time[n]], dtype=float),
            ),
        )
        for left, right in blocks:
            left_array = np.asarray(left, dtype=float)
            right_array = np.asarray(right, dtype=float)
            residual = float(np.max(np.abs(left_array - right_array)))
            scale = max(
                1.0,
                float(np.max(np.abs(left_array))),
                float(np.max(np.abs(right_array))),
            )
            worst = max(worst, residual / scale)
    return float(worst)


def _max_interval_spatial_ks_constraint_residuals(
    solution: SpatialKSBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
) -> tuple[float, float]:
    variable = RationalInterval.from_float_interval(
        float(parameter_interval[0]),
        float(parameter_interval[1]),
    )
    pair_energy_constraint = ks_pair_energy_constraint_coefficients(
        solution,
        solution.order,
    )
    horizontal_constraint = ks_horizontal_constraint_coefficients(
        solution,
        solution.order,
    )
    return (
        float(
            _rational_interval_polynomial_abs_sup(
                pair_energy_constraint,
                variable,
            )
        ),
        float(
            _rational_interval_polynomial_abs_sup(
                horizontal_constraint,
                variable,
            )
        ),
    )


def _max_interval_spatial_ks_taylor_model_residual(
    solution: SpatialKSBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    tail_bound: float,
) -> float:
    order = solution.order
    rhs = ks_regularized_rhs_coefficients(solution, order - 1)
    variable = RationalInterval.from_float_interval(
        float(parameter_interval[0]),
        float(parameter_interval[1]),
    )
    residual_blocks = (
        _lc_vector_residual_coefficients(solution.u, rhs.u, order),
        _lc_vector_residual_coefficients(
            solution.u_velocity,
            rhs.u_velocity,
            order,
        ),
        _lc_scalar_residual_coefficients(
            solution.pair_energy,
            rhs.pair_energy,
            order,
        ),
        _lc_vector_residual_coefficients(
            solution.binary_center,
            rhs.binary_center,
            order,
        ),
        _lc_vector_residual_coefficients(
            solution.binary_center_velocity,
            rhs.binary_center_velocity,
            order,
        ),
        _lc_vector_residual_coefficients(
            solution.third_offset,
            rhs.third_offset,
            order,
        ),
        _lc_vector_residual_coefficients(
            solution.third_offset_velocity,
            rhs.third_offset_velocity,
            order,
        ),
        _lc_scalar_residual_coefficients(
            solution.physical_time,
            rhs.physical_time,
            order,
        ),
    )
    worst = RationalInterval.point(0).upper
    for block in residual_blocks:
        block = np.asarray(block, dtype=float)
        if block.ndim == 1:
            worst = max(
                worst,
                _rational_interval_polynomial_abs_sup(block, variable),
            )
        else:
            for index in np.ndindex(block.shape[1:]):
                worst = max(
                    worst,
                    _rational_interval_polynomial_abs_sup(
                        block[(slice(None), *index)],
                        variable,
                    ),
                )
    bound = worst + RationalInterval.from_float_interval(
        max(0.0, float(tail_bound))
    ).upper
    return float(np.nextafter(float(bound), np.inf))


def _max_interval_spatial_ks_projected_newton_residual(
    solution: SpatialKSBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    tail_bound: float,
    rho_lower_bound: float,
    pieces: int = 128,
) -> float:
    """Conservative projected residual check on nonsingular KS slabs.

    A regularized KS chart may legitimately pass through binary collision,
    where the projected Newtonian coordinates are singular.  The regularized
    RHS is checked on the full chart elsewhere; this projected obligation is
    only meaningful on resolved subintervals whose interval ``rho`` stays
    bounded away from zero.
    """

    lower = float(parameter_interval[0])
    upper = float(parameter_interval[1])
    if not (np.isfinite(lower) and np.isfinite(upper) and lower <= upper):
        return float("inf")
    pieces = max(1, int(pieces))
    worst = 0.0
    checked_piece_count = 0
    for index in range(pieces):
        piece_lower = lower + (upper - lower) * index / pieces
        piece_upper = lower + (upper - lower) * (index + 1) / pieces
        try:
            piece_value = _interval_spatial_ks_projected_newton_residual_piece(
                solution,
                FloatInterval(piece_lower, piece_upper),
                tail_bound=tail_bound,
                rho_lower_bound=rho_lower_bound,
            )
        except (FloatingPointError, ValueError):
            continue
        if np.isfinite(piece_value):
            worst = max(worst, piece_value)
            checked_piece_count += 1
    if checked_piece_count == 0:
        return float("inf")
    return float(worst)


def _interval_spatial_ks_projected_newton_residual_piece(
    solution: SpatialKSBinaryTaylorSolution,
    variable: FloatInterval,
    *,
    tail_bound: float,
    rho_lower_bound: float,
) -> float:
    """Projected Newton residual over one nonsingular KS interval piece."""

    u = interval_array_series_eval(solution.u, variable)
    u_velocity = interval_array_series_eval(solution.u_velocity, variable)
    pair_energy = interval_polynomial_eval(solution.pair_energy, variable)
    binary_center = interval_array_series_eval(solution.binary_center, variable)
    third_offset = interval_array_series_eval(solution.third_offset, variable)
    rho = _interval_dot(u, u)
    if rho.lower <= max(0.0, float(rho_lower_bound)):
        return float("inf")

    center_acceleration, third_offset_acceleration, relative_perturbation = (
        _interval_spatial_ks_analytic_coordinate_accelerations(
            solution,
            u,
            third_offset,
        )
    )
    u_acceleration = _interval_vector_add(
        _interval_vector_scale_by_interval(u, pair_energy.scale(0.5)),
        _interval_vector_scale_by_interval(
            _interval_ks_transpose_matvec(u, relative_perturbation),
            rho.scale(0.25),
        ),
    )
    projected = _interval_spatial_ks_projected_accelerations(
        solution,
        u,
        u_velocity,
        u_acceleration,
        center_acceleration,
        third_offset_acceleration,
    )
    positions = _interval_spatial_ks_positions(
        solution,
        u,
        binary_center,
        third_offset,
    )
    newton = _interval_newton_accelerations(positions, solution.masses)
    worst = 0.0
    for index in np.ndindex(projected.shape):
        worst = max(worst, _interval_abs_sup(projected[index] - newton[index]))
    return float(worst + max(0.0, float(tail_bound)))


def _interval_spatial_ks_analytic_coordinate_accelerations(
    solution: SpatialKSBinaryTaylorSolution,
    u: np.ndarray,
    third_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second = solution.pair
    third = 3 - first - second
    masses = np.asarray(solution.masses, dtype=float)
    pair_mass = float(masses[first] + masses[second])
    alpha = float(masses[second] / pair_mass)
    beta = float(masses[first] / pair_mass)
    relative_position = _interval_ks_project(u)
    from_first_to_third = _interval_vector_add(
        third_offset,
        _interval_vector_scale(relative_position, alpha),
    )
    from_second_to_third = _interval_vector_sub(
        third_offset,
        _interval_vector_scale(relative_position, beta),
    )
    field_first = _interval_inverse_square_field(from_first_to_third)
    field_second = _interval_inverse_square_field(from_second_to_third)
    binary_center_acceleration = _interval_vector_scale(
        _interval_vector_add(
            _interval_vector_scale(field_first, float(masses[first])),
            _interval_vector_scale(field_second, float(masses[second])),
        ),
        float(masses[third] / pair_mass),
    )
    third_acceleration = _interval_vector_scale(
        _interval_vector_add(
            _interval_vector_scale(field_first, float(masses[first])),
            _interval_vector_scale(field_second, float(masses[second])),
        ),
        -1.0,
    )
    third_offset_acceleration = _interval_vector_sub(
        third_acceleration,
        binary_center_acceleration,
    )
    relative_perturbation = _interval_vector_scale(
        _interval_vector_sub(field_second, field_first),
        float(masses[third]),
    )
    return (
        binary_center_acceleration,
        third_offset_acceleration,
        relative_perturbation,
    )


def _interval_spatial_ks_positions(
    solution: SpatialKSBinaryTaylorSolution,
    u: np.ndarray,
    binary_center: np.ndarray,
    third_offset: np.ndarray,
) -> np.ndarray:
    first, second = solution.pair
    third = 3 - first - second
    masses = np.asarray(solution.masses, dtype=float)
    pair_mass = float(masses[first] + masses[second])
    relative_position = _interval_ks_project(u)
    positions = _interval_zero_array((3, 3))
    positions[first] = _interval_vector_sub(
        binary_center,
        _interval_vector_scale(relative_position, float(masses[second] / pair_mass)),
    )
    positions[second] = _interval_vector_add(
        binary_center,
        _interval_vector_scale(relative_position, float(masses[first] / pair_mass)),
    )
    positions[third] = _interval_vector_add(binary_center, third_offset)
    return positions


def _interval_spatial_ks_projected_accelerations(
    solution: SpatialKSBinaryTaylorSolution,
    u: np.ndarray,
    u_velocity: np.ndarray,
    u_acceleration: np.ndarray,
    binary_center_acceleration: np.ndarray,
    third_offset_acceleration: np.ndarray,
) -> np.ndarray:
    first, second = solution.pair
    third = 3 - first - second
    masses = np.asarray(solution.masses, dtype=float)
    pair_mass = float(masses[first] + masses[second])
    relative_acceleration = _interval_relative_acceleration_from_ks_acceleration(
        u,
        u_velocity,
        u_acceleration,
    )
    acceleration = _interval_zero_array((3, 3))
    acceleration[first] = _interval_vector_sub(
        binary_center_acceleration,
        _interval_vector_scale(relative_acceleration, float(masses[second] / pair_mass)),
    )
    acceleration[second] = _interval_vector_add(
        binary_center_acceleration,
        _interval_vector_scale(relative_acceleration, float(masses[first] / pair_mass)),
    )
    acceleration[third] = _interval_vector_add(
        binary_center_acceleration,
        third_offset_acceleration,
    )
    return acceleration


def _max_sampled_newton_residual(
    q: np.ndarray,
    v: np.ndarray,
    masses: np.ndarray,
    parameter_interval: tuple[float, float],
    *,
    sample_count: int,
) -> float:
    q_derivative = _derivative_coefficients(q)
    v_derivative = _derivative_coefficients(v)
    worst = 0.0
    for time in np.linspace(
        parameter_interval[0],
        parameter_interval[1],
        sample_count,
    ):
        positions = _evaluate_coefficients(q, float(time))
        velocities = _evaluate_coefficients(v, float(time))
        dqdt = _evaluate_coefficients(q_derivative, float(time))
        dvdt = _evaluate_coefficients(v_derivative, float(time))
        acceleration = accelerations(positions, masses)
        worst = max(
            worst,
            float(np.max(np.abs(dqdt - velocities))),
            float(np.max(np.abs(dvdt - acceleration))),
        )
    return worst


def _max_interval_ordinary_newton_residual(
    q: np.ndarray,
    v: np.ndarray,
    masses: np.ndarray,
    parameter_interval: tuple[float, float],
) -> float:
    variable = FloatInterval(float(parameter_interval[0]), float(parameter_interval[1]))
    positions = interval_array_series_eval(q, variable)
    velocities = interval_array_series_eval(v, variable)
    position_derivative = interval_array_series_eval(
        interval_array_derivative_coefficients(q),
        variable,
    )
    velocity_derivative = interval_array_series_eval(
        interval_array_derivative_coefficients(v),
        variable,
    )
    acceleration = _interval_newton_accelerations(positions, masses)
    return max(
        _interval_array_difference_sup(position_derivative, velocities),
        _interval_array_difference_sup(velocity_derivative, acceleration),
    )


def _interval_ordinary_newton_residual_blocks(
    q: np.ndarray,
    v: np.ndarray,
    masses: np.ndarray,
    parameter_interval: tuple[float, float],
) -> tuple[float, float]:
    """Return separate outward interval bounds for q'-v and v'-a(q)."""

    variable = FloatInterval(*map(float, parameter_interval))
    positions = interval_array_series_eval(q, variable)
    velocities = interval_array_series_eval(v, variable)
    position_derivative = interval_array_series_eval(
        interval_array_derivative_coefficients(q), variable
    )
    velocity_derivative = interval_array_series_eval(
        interval_array_derivative_coefficients(v), variable
    )
    acceleration = _interval_newton_accelerations(positions, masses)
    return (
        _interval_array_difference_sup(position_derivative, velocities),
        _interval_array_difference_sup(velocity_derivative, acceleration),
    )


def _max_interval_ordinary_taylor_model_residual(
    q: np.ndarray,
    v: np.ndarray,
    masses: np.ndarray,
    parameter_interval: tuple[float, float],
    *,
    tail_bound: float,
) -> float:
    order = q.shape[0] - 1
    acc = acceleration_coefficients(q, masses, order - 1)
    variable = RationalInterval.from_float_interval(
        float(parameter_interval[0]),
        float(parameter_interval[1]),
    )
    worst = RationalInterval.point(0).upper
    for index in np.ndindex(q.shape[1:]):
        position_residual = tuple(
            RationalInterval.from_float_interval(
                float((n + 1) * q[(n + 1, *index)] - v[(n, *index)])
            )
            for n in range(order)
        )
        velocity_residual = tuple(
            RationalInterval.from_float_interval(
                float((n + 1) * v[(n + 1, *index)] - acc[(n, *index)])
            )
            for n in range(order)
        )
        worst = max(
            worst,
            _rational_interval_abs_sup(
                rational_interval_polynomial_eval(position_residual, variable)
            ),
            _rational_interval_abs_sup(
                rational_interval_polynomial_eval(velocity_residual, variable)
            ),
        )
    bound = worst + RationalInterval.from_float_interval(
        max(0.0, float(tail_bound))
    ).upper
    return float(np.nextafter(float(bound), np.inf))


def _interval_newton_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if positions.ndim != 2 or positions.shape[0] != masses.size:
        raise ValueError("position interval array shape does not match masses")
    accelerations_out = np.empty(positions.shape, dtype=object)
    for index in np.ndindex(positions.shape):
        accelerations_out[index] = FloatInterval.point(0.0)
    for i in range(positions.shape[0]):
        for j in range(positions.shape[0]):
            if i == j:
                continue
            diff = tuple(positions[j, axis] - positions[i, axis] for axis in range(positions.shape[1]))
            distance_squared = FloatInterval.point(0.0)
            for component in diff:
                distance_squared = distance_squared + _interval_square(component)
            inv_distance_cubed = distance_squared.positive_power(-1.5)
            for axis, component in enumerate(diff):
                accelerations_out[i, axis] = (
                    accelerations_out[i, axis]
                    + component * inv_distance_cubed.scale(float(masses[j]))
                )
    return accelerations_out


def _interval_square(interval: FloatInterval) -> FloatInterval:
    if interval.lower <= 0.0 <= interval.upper:
        lower = 0.0
        upper = max(interval.lower * interval.lower, interval.upper * interval.upper)
    else:
        values = (interval.lower * interval.lower, interval.upper * interval.upper)
        lower = min(values)
        upper = max(values)
    return FloatInterval(
        float(np.nextafter(lower, -np.inf)),
        float(np.nextafter(upper, np.inf)),
    )


def _interval_array_difference_sup(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != right.shape:
        raise ValueError("interval arrays must have matching shapes")
    worst = 0.0
    for index in np.ndindex(left.shape):
        interval = left[index] - right[index]
        worst = max(worst, _interval_abs_sup(interval))
    return float(worst)


def _interval_abs_sup(interval: FloatInterval) -> float:
    return float(max(abs(interval.lower), abs(interval.upper)))


def _rational_interval_abs_sup(interval: RationalInterval):
    return max(abs(interval.lower), abs(interval.upper))


def _rational_interval_polynomial_abs_sup(
    coefficients: np.ndarray,
    variable: RationalInterval,
):
    coefficient_intervals = tuple(
        RationalInterval.from_float_interval(float(coefficient))
        for coefficient in np.asarray(coefficients, dtype=float)
    )
    return _rational_interval_abs_sup(
        rational_interval_polynomial_eval(coefficient_intervals, variable)
    )


def _interval_zero_array(shape: tuple[int, ...]) -> np.ndarray:
    values = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        values[index] = FloatInterval.point(0.0)
    return values


def _interval_vector_add(*vectors: np.ndarray) -> np.ndarray:
    if not vectors:
        raise ValueError("at least one vector is required")
    out = _interval_zero_array(np.asarray(vectors[0], dtype=object).shape)
    for vector in vectors:
        vector = np.asarray(vector, dtype=object)
        if vector.shape != out.shape:
            raise ValueError("interval vectors must have matching shapes")
        for index in np.ndindex(out.shape):
            out[index] = out[index] + _coerce_float_interval(vector[index])
    return out


def _interval_vector_sub(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != right.shape:
        raise ValueError("interval vectors must have matching shapes")
    out = _interval_zero_array(left.shape)
    for index in np.ndindex(left.shape):
        out[index] = (
            _coerce_float_interval(left[index])
            - _coerce_float_interval(right[index])
        )
    return out


def _interval_vector_scale(vector: np.ndarray, factor: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=object)
    out = _interval_zero_array(vector.shape)
    for index in np.ndindex(vector.shape):
        out[index] = _coerce_float_interval(vector[index]).scale(float(factor))
    return out


def _interval_vector_scale_by_interval(
    vector: np.ndarray,
    factor: FloatInterval,
) -> np.ndarray:
    vector = np.asarray(vector, dtype=object)
    factor = _coerce_float_interval(factor)
    out = _interval_zero_array(vector.shape)
    for index in np.ndindex(vector.shape):
        out[index] = _coerce_float_interval(vector[index]) * factor
    return out


def _interval_dot(left: np.ndarray, right: np.ndarray) -> FloatInterval:
    left = np.asarray(left, dtype=object).reshape(-1)
    right = np.asarray(right, dtype=object).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("interval vectors must have matching shapes")
    value = FloatInterval.point(0.0)
    for index in range(left.shape[0]):
        value = value + (
            _coerce_float_interval(left[index])
            * _coerce_float_interval(right[index])
        )
    return value


def _interval_centered_rows(values: np.ndarray, masses: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if values.ndim != 2 or values.shape[0] != masses.size:
        raise ValueError("values must have one row per mass")
    total_mass = float(np.sum(masses))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("total mass must be positive")
    center = _interval_zero_array((values.shape[1],))
    for body, mass in enumerate(masses):
        for axis in range(values.shape[1]):
            center[axis] = center[axis] + _coerce_float_interval(
                values[body, axis],
            ).scale(float(mass / total_mass))
    out = _interval_zero_array(values.shape)
    for body in range(values.shape[0]):
        for axis in range(values.shape[1]):
            out[body, axis] = _coerce_float_interval(values[body, axis]) - center[axis]
    return out


def _interval_planar_wedge(left: np.ndarray, right: np.ndarray) -> FloatInterval:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != (2,) or right.shape != (2,):
        raise ValueError("planar wedge expects two 2D vectors")
    return _coerce_float_interval(left[0]) * _coerce_float_interval(
        right[1],
    ) - _coerce_float_interval(left[1]) * _coerce_float_interval(right[0])


def _interval_cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    if left.shape != (3,) or right.shape != (3,):
        raise ValueError("cross product expects two 3D vectors")
    return np.array(
        [
            _coerce_float_interval(left[1]) * _coerce_float_interval(right[2])
            - _coerce_float_interval(left[2]) * _coerce_float_interval(right[1]),
            _coerce_float_interval(left[2]) * _coerce_float_interval(right[0])
            - _coerce_float_interval(left[0]) * _coerce_float_interval(right[2]),
            _coerce_float_interval(left[0]) * _coerce_float_interval(right[1])
            - _coerce_float_interval(left[1]) * _coerce_float_interval(right[0]),
        ],
        dtype=object,
    )


def _coerce_float_interval(value: object) -> FloatInterval:
    return value if isinstance(value, FloatInterval) else FloatInterval.point(float(value))


def _interval_lc_square(z: np.ndarray) -> np.ndarray:
    x = _coerce_float_interval(z[0])
    y = _coerce_float_interval(z[1])
    return np.array(
        [
            x * x - y * y,
            (x * y).scale(2.0),
        ],
        dtype=object,
    )


def _interval_lc_matvec(z: np.ndarray, vector: np.ndarray) -> np.ndarray:
    x = _coerce_float_interval(z[0])
    y = _coerce_float_interval(z[1])
    v0 = _coerce_float_interval(vector[0])
    v1 = _coerce_float_interval(vector[1])
    return np.array(
        [
            x.scale(2.0) * v0 - y.scale(2.0) * v1,
            y.scale(2.0) * v0 + x.scale(2.0) * v1,
        ],
        dtype=object,
    )


def _interval_lc_transpose_matvec(z: np.ndarray, vector: np.ndarray) -> np.ndarray:
    x = _coerce_float_interval(z[0])
    y = _coerce_float_interval(z[1])
    v0 = _coerce_float_interval(vector[0])
    v1 = _coerce_float_interval(vector[1])
    return np.array(
        [
            x.scale(2.0) * v0 + y.scale(2.0) * v1,
            y.scale(-2.0) * v0 + x.scale(2.0) * v1,
        ],
        dtype=object,
    )


def _interval_relative_acceleration_from_lc_acceleration(
    z: np.ndarray,
    z_velocity: np.ndarray,
    z_acceleration: np.ndarray,
) -> np.ndarray:
    rho = _interval_dot(z, z)
    if rho.lower <= 0.0:
        raise ValueError("LC projection interval reaches binary collision")
    rho_inv = rho.reciprocal()
    rho_inv_squared = rho_inv * rho_inv
    rho_prime = _interval_dot(z, z_velocity).scale(2.0)
    q_prime = _interval_lc_matvec(z, z_velocity)
    velocity_prime = _interval_vector_sub(
        _interval_vector_scale_by_interval(
            _interval_vector_add(
                _interval_lc_matvec(z_velocity, z_velocity),
                _interval_lc_matvec(z, z_acceleration),
            ),
            rho_inv,
        ),
        _interval_vector_scale_by_interval(
            q_prime,
            rho_prime * rho_inv_squared,
        ),
    )
    return _interval_vector_scale_by_interval(velocity_prime, rho_inv)


def _interval_ks_project(u: np.ndarray) -> np.ndarray:
    a = _coerce_float_interval(u[0])
    b = _coerce_float_interval(u[1])
    c = _coerce_float_interval(u[2])
    d = _coerce_float_interval(u[3])
    return np.array(
        [
            a * a - b * b - c * c + d * d,
            (a * b - c * d).scale(2.0),
            (a * c + b * d).scale(2.0),
        ],
        dtype=object,
    )


def _interval_ks_matvec(u: np.ndarray, vector: np.ndarray) -> np.ndarray:
    a = _coerce_float_interval(u[0])
    b = _coerce_float_interval(u[1])
    c = _coerce_float_interval(u[2])
    d = _coerce_float_interval(u[3])
    v0 = _coerce_float_interval(vector[0])
    v1 = _coerce_float_interval(vector[1])
    v2 = _coerce_float_interval(vector[2])
    v3 = _coerce_float_interval(vector[3])
    return np.array(
        [
            a.scale(2.0) * v0
            - b.scale(2.0) * v1
            - c.scale(2.0) * v2
            + d.scale(2.0) * v3,
            b.scale(2.0) * v0
            + a.scale(2.0) * v1
            - d.scale(2.0) * v2
            - c.scale(2.0) * v3,
            c.scale(2.0) * v0
            + d.scale(2.0) * v1
            + a.scale(2.0) * v2
            + b.scale(2.0) * v3,
        ],
        dtype=object,
    )


def _interval_ks_transpose_matvec(u: np.ndarray, vector: np.ndarray) -> np.ndarray:
    a = _coerce_float_interval(u[0])
    b = _coerce_float_interval(u[1])
    c = _coerce_float_interval(u[2])
    d = _coerce_float_interval(u[3])
    x = _coerce_float_interval(vector[0])
    y = _coerce_float_interval(vector[1])
    z = _coerce_float_interval(vector[2])
    return np.array(
        [
            a.scale(2.0) * x + b.scale(2.0) * y + c.scale(2.0) * z,
            b.scale(-2.0) * x + a.scale(2.0) * y + d.scale(2.0) * z,
            c.scale(-2.0) * x + d.scale(-2.0) * y + a.scale(2.0) * z,
            d.scale(2.0) * x + c.scale(-2.0) * y + b.scale(2.0) * z,
        ],
        dtype=object,
    )


def _interval_relative_acceleration_from_ks_acceleration(
    u: np.ndarray,
    u_velocity: np.ndarray,
    u_acceleration: np.ndarray,
) -> np.ndarray:
    rho = _interval_dot(u, u)
    if rho.lower <= 0.0:
        raise ValueError("KS projection interval reaches binary collision")
    rho_inv = rho.reciprocal()
    rho_inv_squared = rho_inv * rho_inv
    rho_inv_cubed = rho_inv_squared * rho_inv
    rho_prime = _interval_dot(u, u_velocity).scale(2.0)
    q_prime = _interval_ks_matvec(u, u_velocity)
    q_second = _interval_vector_add(
        _interval_ks_matvec(u_velocity, u_velocity),
        _interval_ks_matvec(u, u_acceleration),
    )
    return _interval_vector_sub(
        _interval_vector_scale_by_interval(q_second, rho_inv_squared),
        _interval_vector_scale_by_interval(
            q_prime,
            rho_prime * rho_inv_cubed,
        ),
    )


def _interval_inverse_square_field(vector: np.ndarray) -> np.ndarray:
    norm_square = _interval_dot(vector, vector)
    if norm_square.lower <= 0.0:
        raise ValueError("inverse-square field interval reaches collision")
    inverse_cube = norm_square.positive_power(-1.5)
    return _interval_vector_scale_by_interval(vector, inverse_cube)


def _interval_newton_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if positions.shape[0] != 3 or masses.shape != (3,):
        raise ValueError("expected three positions and three masses")
    accelerations_interval = _interval_zero_array(positions.shape)
    for first in range(3):
        for second in range(first + 1, 3):
            delta = _interval_vector_sub(positions[second], positions[first])
            direction_over_radius_cubed = _interval_inverse_square_field(delta)
            accelerations_interval[first] = _interval_vector_add(
                accelerations_interval[first],
                _interval_vector_scale(
                    direction_over_radius_cubed,
                    float(masses[second]),
                ),
            )
            accelerations_interval[second] = _interval_vector_sub(
                accelerations_interval[second],
                _interval_vector_scale(
                    direction_over_radius_cubed,
                    float(masses[first]),
                ),
            )
    return accelerations_interval


def _max_sampled_regularized_binary_residual(
    solution: RegularizedBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    sample_count: int,
) -> float:
    derivatives = {
        "z": _derivative_coefficients(solution.z),
        "z_velocity": _derivative_coefficients(solution.z_velocity),
        "pair_energy": _derivative_scalar_coefficients(solution.pair_energy),
        "binary_center": _derivative_coefficients(solution.binary_center),
        "binary_center_velocity": _derivative_coefficients(solution.binary_center_velocity),
        "third_offset": _derivative_coefficients(solution.third_offset),
        "third_offset_velocity": _derivative_coefficients(solution.third_offset_velocity),
        "physical_time": _derivative_scalar_coefficients(solution.physical_time),
    }
    worst = 0.0
    for parameter in np.linspace(
        parameter_interval[0],
        parameter_interval[1],
        sample_count,
    ):
        s_value = float(parameter)
        state = solution.state_at(s_value)
        rhs = regularized_binary_collision_chart_rhs(state)
        worst = max(
            worst,
            _array_sup_norm(_evaluate_coefficients(derivatives["z"], s_value) - rhs.z),
            _array_sup_norm(
                _evaluate_coefficients(derivatives["z_velocity"], s_value)
                - rhs.z_velocity,
            ),
            abs(
                _evaluate_scalar_coefficients(derivatives["pair_energy"], s_value)
                - rhs.pair_energy
            ),
            _array_sup_norm(
                _evaluate_coefficients(derivatives["binary_center"], s_value)
                - rhs.binary_center,
            ),
            _array_sup_norm(
                _evaluate_coefficients(derivatives["binary_center_velocity"], s_value)
                - rhs.binary_center_velocity,
            ),
            _array_sup_norm(
                _evaluate_coefficients(derivatives["third_offset"], s_value)
                - rhs.third_offset,
            ),
            _array_sup_norm(
                _evaluate_coefficients(
                    derivatives["third_offset_velocity"],
                    s_value,
                )
                - rhs.third_offset_velocity,
            ),
            abs(
                _evaluate_scalar_coefficients(derivatives["physical_time"], s_value)
                - rhs.physical_time
            ),
        )
    return float(worst)


def _max_sampled_projected_binary_newton_residual(
    solution: RegularizedBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    sample_count: int,
    rho_lower_bound: float,
) -> tuple[float, int]:
    worst = 0.0
    checked = 0
    for parameter in np.linspace(
        parameter_interval[0],
        parameter_interval[1],
        sample_count,
    ):
        state = solution.state_at(float(parameter))
        if state.rho <= rho_lower_bound:
            continue
        derivative = regularized_binary_collision_chart_rhs(state)
        positions, _velocities = regularized_binary_collision_chart_to_planar(state)
        projected_acceleration = planar_accelerations_from_regularized_chart_rhs(
            state,
            derivative,
        )
        newton_acceleration = accelerations(positions, solution.masses)
        worst = max(
            worst,
            _array_sup_norm(projected_acceleration - newton_acceleration),
        )
        checked += 1
    return float(worst if checked else np.inf), checked


def _max_sampled_spatial_ks_residual(
    solution: SpatialKSBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    sample_count: int,
) -> float:
    derivatives = {
        "u": _derivative_coefficients(solution.u),
        "u_velocity": _derivative_coefficients(solution.u_velocity),
        "pair_energy": _derivative_scalar_coefficients(solution.pair_energy),
        "binary_center": _derivative_coefficients(solution.binary_center),
        "binary_center_velocity": _derivative_coefficients(solution.binary_center_velocity),
        "third_offset": _derivative_coefficients(solution.third_offset),
        "third_offset_velocity": _derivative_coefficients(solution.third_offset_velocity),
        "physical_time": _derivative_scalar_coefficients(solution.physical_time),
    }
    worst = 0.0
    for parameter in np.linspace(
        parameter_interval[0],
        parameter_interval[1],
        sample_count,
    ):
        s_value = float(parameter)
        state = solution.state_at(s_value)
        rhs = regularized_ks_binary_chart_rhs(state)
        worst = max(
            worst,
            _array_sup_norm(_evaluate_coefficients(derivatives["u"], s_value) - rhs.u),
            _array_sup_norm(
                _evaluate_coefficients(derivatives["u_velocity"], s_value)
                - rhs.u_velocity,
            ),
            abs(
                _evaluate_scalar_coefficients(derivatives["pair_energy"], s_value)
                - rhs.pair_energy
            ),
            _array_sup_norm(
                _evaluate_coefficients(derivatives["binary_center"], s_value)
                - rhs.binary_center,
            ),
            _array_sup_norm(
                _evaluate_coefficients(derivatives["binary_center_velocity"], s_value)
                - rhs.binary_center_velocity,
            ),
            _array_sup_norm(
                _evaluate_coefficients(derivatives["third_offset"], s_value)
                - rhs.third_offset,
            ),
            _array_sup_norm(
                _evaluate_coefficients(
                    derivatives["third_offset_velocity"],
                    s_value,
                )
                - rhs.third_offset_velocity,
            ),
            abs(
                _evaluate_scalar_coefficients(derivatives["physical_time"], s_value)
                - rhs.physical_time
            ),
        )
    return float(worst)


def _max_sampled_spatial_ks_projected_newton_residual(
    solution: SpatialKSBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    sample_count: int,
    rho_lower_bound: float,
) -> tuple[float, int]:
    worst = 0.0
    checked = 0
    for parameter in np.linspace(
        parameter_interval[0],
        parameter_interval[1],
        sample_count,
    ):
        state = solution.state_at(float(parameter))
        if state.rho <= rho_lower_bound:
            continue
        derivative = regularized_ks_binary_chart_rhs(state)
        positions, _velocities = ks_binary_chart_to_spatial(state)
        projected_acceleration = spatial_accelerations_from_ks_binary_rhs(
            state,
            derivative,
        )
        newton_acceleration = accelerations(positions, solution.masses)
        worst = max(
            worst,
            _array_sup_norm(projected_acceleration - newton_acceleration),
        )
        checked += 1
    return float(worst if checked else np.inf), checked


def _max_sampled_fuchsian_stop_chart_quantities(
    branch: FiniteFuchsianLogBranch,
    tau_interval: tuple[float, float],
    *,
    sample_count: int,
) -> tuple[float, float, float, float, int]:
    max_lifted_residual = 0.0
    max_projected_residual = 0.0
    max_angular_momentum = 0.0
    max_position_scale = 0.0
    checked = 0
    for tau in _punctured_tau_samples(tau_interval, sample_count=sample_count):
        shape, first, second = branch.tau_derivatives(tau)
        positions = branch.positions_at_tau(tau)
        projected_acceleration = (
            tau**2 * second + 2.0 * tau * first - 2.0 * shape
        ) / (9.0 * tau**4)
        lifted_residual = (
            tau**2 * second
            + 2.0 * tau * first
            - 2.0 * shape
            - 9.0 * accelerations(shape, branch.masses)
        )
        projected_residual = projected_acceleration - accelerations(
            positions,
            branch.masses,
        )
        max_lifted_residual = max(
            max_lifted_residual,
            _array_sup_norm(lifted_residual),
        )
        max_projected_residual = max(
            max_projected_residual,
            _array_sup_norm(projected_residual),
        )
        if branch.central_shape.shape[1] == 2:
            max_angular_momentum = max(
                max_angular_momentum,
                abs(branch.centered_angular_momentum_scalar_at_tau(tau)),
            )
        else:
            max_angular_momentum = max(
                max_angular_momentum,
                _spatial_centered_angular_momentum_norm(branch, tau),
            )
        max_position_scale = max(
            max_position_scale,
            float(np.max(np.abs(positions))) / tau**2,
        )
        checked += 1
    if checked == 0:
        return (np.inf, np.inf, np.inf, np.inf, 0)
    return (
        float(max_lifted_residual),
        float(max_projected_residual),
        float(max_angular_momentum),
        float(max_position_scale),
        int(checked),
    )


def _max_interval_fuchsian_lifted_residual_on_punctured_shells(
    branch: FiniteFuchsianLogBranch,
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    primitive_cauchy_inputs: FiniteFuchsianLogPrimitiveCauchyInputsCertificate | None,
) -> tuple[float, str]:
    """Conservative interval check for the lifted Fuchsian equation.

    The primitive Cauchy certificate covers the geometric tail near tau=0.
    This checker covers the resolved compact punctured slabs between that
    tail radius and the declared isolation radius, on both time sides.
    """

    if primitive_cauchy_inputs is None:
        return (np.inf, "primitive Cauchy inputs not supplied")
    initial_radius = float(primitive_cauchy_inputs.initial_radius)
    shell_contraction = float(primitive_cauchy_inputs.shell_contraction)
    radius_upper = min(
        float(isolation_radius),
        max(abs(float(tau_interval[0])), abs(float(tau_interval[1]))),
    )
    if not (
        np.isfinite(initial_radius)
        and np.isfinite(shell_contraction)
        and np.isfinite(radius_upper)
        and 0.0 < shell_contraction < 1.0
        and 0.0 < initial_radius < radius_upper
    ):
        return (
            np.inf,
            (
                f"initial_radius={initial_radius}; "
                f"shell_contraction={shell_contraction}; "
                f"radius_upper={radius_upper}"
            ),
        )

    radius_slabs = (
        (initial_radius, radius_upper),
        (initial_radius * shell_contraction, initial_radius),
    )
    tau_slabs: list[FloatInterval] = []
    for lower, upper in radius_slabs:
        if not 0.0 < lower < upper:
            continue
        tau_slabs.append(FloatInterval(lower, upper))
        tau_slabs.append(FloatInterval(-upper, -lower))

    worst = 0.0
    checked = 0
    for tau_slab in tau_slabs:
        lifted_residual = _interval_fuchsian_lifted_residual(branch, tau_slab)
        for index in np.ndindex(lifted_residual.shape):
            worst = max(worst, _interval_abs_sup(lifted_residual[index]))
        checked += 1
    if checked == 0:
        return (np.inf, "no punctured shell slabs were available")
    return (
        float(worst),
        (
            f"checked_slabs={checked}; "
            f"initial_radius={initial_radius}; radius_upper={radius_upper}"
        ),
    )


def _max_interval_fuchsian_projected_residual_on_punctured_shells(
    branch: FiniteFuchsianLogBranch,
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    primitive_cauchy_inputs: FiniteFuchsianLogPrimitiveCauchyInputsCertificate | None,
) -> tuple[float, str]:
    """Conservative interval check for projected Fuchsian Newton residuals."""

    if primitive_cauchy_inputs is None:
        return (np.inf, "primitive Cauchy inputs not supplied")
    initial_radius = float(primitive_cauchy_inputs.initial_radius)
    shell_contraction = float(primitive_cauchy_inputs.shell_contraction)
    radius_upper = min(
        float(isolation_radius),
        max(abs(float(tau_interval[0])), abs(float(tau_interval[1]))),
    )
    if not (
        np.isfinite(initial_radius)
        and np.isfinite(shell_contraction)
        and np.isfinite(radius_upper)
        and 0.0 < shell_contraction < 1.0
        and 0.0 < initial_radius < radius_upper
    ):
        return (
            np.inf,
            (
                f"initial_radius={initial_radius}; "
                f"shell_contraction={shell_contraction}; "
                f"radius_upper={radius_upper}"
            ),
        )

    radius_slabs = (
        (initial_radius, radius_upper),
        (initial_radius * shell_contraction, initial_radius),
    )
    tau_slabs: list[FloatInterval] = []
    for lower, upper in radius_slabs:
        if not 0.0 < lower < upper:
            continue
        tau_slabs.append(FloatInterval(lower, upper))
        tau_slabs.append(FloatInterval(-upper, -lower))

    worst = 0.0
    checked = 0
    for tau_slab in tau_slabs:
        lifted_residual = _interval_fuchsian_lifted_residual(branch, tau_slab)
        tau_fourth = (tau_slab * tau_slab) * (tau_slab * tau_slab)
        scale = tau_fourth.scale(9.0).reciprocal()
        for index in np.ndindex(lifted_residual.shape):
            projected = _coerce_float_interval(lifted_residual[index]) * scale
            worst = max(worst, _interval_abs_sup(projected))
        checked += 1
    if checked == 0:
        return (np.inf, "no punctured shell slabs were available")
    return (
        float(worst),
        (
            f"checked_slabs={checked}; "
            f"initial_radius={initial_radius}; radius_upper={radius_upper}"
        ),
    )


def _max_interval_fuchsian_zero_angular_momentum_on_punctured_shells(
    branch: FiniteFuchsianLogBranch,
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    primitive_cauchy_inputs: FiniteFuchsianLogPrimitiveCauchyInputsCertificate | None,
) -> tuple[float, str]:
    """Conservative interval check for zero angular momentum near total collision."""

    if primitive_cauchy_inputs is None:
        return (np.inf, "primitive Cauchy inputs not supplied")
    initial_radius = float(primitive_cauchy_inputs.initial_radius)
    shell_contraction = float(primitive_cauchy_inputs.shell_contraction)
    radius_upper = min(
        float(isolation_radius),
        max(abs(float(tau_interval[0])), abs(float(tau_interval[1]))),
    )
    if not (
        np.isfinite(initial_radius)
        and np.isfinite(shell_contraction)
        and np.isfinite(radius_upper)
        and 0.0 < shell_contraction < 1.0
        and 0.0 < initial_radius < radius_upper
    ):
        return (
            np.inf,
            (
                f"initial_radius={initial_radius}; "
                f"shell_contraction={shell_contraction}; "
                f"radius_upper={radius_upper}"
            ),
        )

    radius_slabs = (
        (initial_radius, radius_upper),
        (initial_radius * shell_contraction, initial_radius),
    )
    tau_slabs: list[FloatInterval] = []
    for lower, upper in radius_slabs:
        if not 0.0 < lower < upper:
            continue
        tau_slabs.append(FloatInterval(lower, upper))
        tau_slabs.append(FloatInterval(-upper, -lower))

    worst = 0.0
    checked = 0
    for tau_slab in tau_slabs:
        angular = _interval_fuchsian_centered_angular_momentum(branch, tau_slab)
        if isinstance(angular, FloatInterval):
            worst = max(worst, _interval_abs_sup(angular))
        else:
            for component in angular:
                worst = max(worst, _interval_abs_sup(component))
        checked += 1
    if checked == 0:
        return (np.inf, "no punctured shell slabs were available")
    return (
        float(worst),
        (
            f"checked_slabs={checked}; "
            f"initial_radius={initial_radius}; radius_upper={radius_upper}"
        ),
    )


def _max_interval_fuchsian_center_of_mass_and_linear_momentum_on_punctured_shells(
    branch: FiniteFuchsianLogBranch,
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    primitive_cauchy_inputs: FiniteFuchsianLogPrimitiveCauchyInputsCertificate | None,
) -> tuple[float, float, str]:
    """Conservative interval COM and total-momentum check on punctured slabs."""

    if primitive_cauchy_inputs is None:
        return (np.inf, np.inf, "primitive Cauchy inputs not supplied")
    initial_radius = float(primitive_cauchy_inputs.initial_radius)
    shell_contraction = float(primitive_cauchy_inputs.shell_contraction)
    radius_upper = min(
        float(isolation_radius),
        max(abs(float(tau_interval[0])), abs(float(tau_interval[1]))),
    )
    if not (
        np.isfinite(initial_radius)
        and np.isfinite(shell_contraction)
        and np.isfinite(radius_upper)
        and 0.0 < shell_contraction < 1.0
        and 0.0 < initial_radius < radius_upper
    ):
        return (
            np.inf,
            np.inf,
            (
                f"initial_radius={initial_radius}; "
                f"shell_contraction={shell_contraction}; "
                f"radius_upper={radius_upper}"
            ),
        )

    radius_slabs = (
        (initial_radius, radius_upper),
        (initial_radius * shell_contraction, initial_radius),
    )
    tau_slabs: list[FloatInterval] = []
    for lower, upper in radius_slabs:
        if not 0.0 < lower < upper:
            continue
        tau_slabs.append(FloatInterval(lower, upper))
        tau_slabs.append(FloatInterval(-upper, -lower))

    worst_center = 0.0
    worst_momentum = 0.0
    checked = 0
    for tau_slab in tau_slabs:
        center, momentum = _interval_fuchsian_center_of_mass_and_linear_momentum(
            branch,
            tau_slab,
        )
        for component in center:
            worst_center = max(worst_center, _interval_abs_sup(component))
        for component in momentum:
            worst_momentum = max(worst_momentum, _interval_abs_sup(component))
        checked += 1
    if checked == 0:
        return (np.inf, np.inf, "no punctured shell slabs were available")
    return (
        float(worst_center),
        float(worst_momentum),
        (
            f"checked_slabs={checked}; "
            f"initial_radius={initial_radius}; radius_upper={radius_upper}"
        ),
    )


def _max_interval_generalized_fuchsian_lifted_residual_on_punctured_shells(
    branch: FuchsianShapeBranch,
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    remainder_majorant: GeneralizedFuchsianRemainderMajorantCertificate | None,
) -> tuple[float, str]:
    if remainder_majorant is None:
        return (np.inf, "remainder majorant not supplied")
    tau_slabs, detail = _generalized_fuchsian_punctured_tau_slabs(
        tau_interval,
        isolation_radius=isolation_radius,
        initial_radius=float(remainder_majorant.initial_radius),
        shell_contraction=float(remainder_majorant.shell_contraction),
    )
    worst = 0.0
    checked = 0
    pieces = 128
    for tau_slab in tau_slabs:
        width = (tau_slab.upper - tau_slab.lower) / pieces
        if width <= 0.0:
            continue
        for piece in range(pieces):
            sub_slab = FloatInterval(
                tau_slab.lower + piece * width,
                tau_slab.lower + (piece + 1) * width,
            )
            lifted_residual = _interval_generalized_fuchsian_lifted_residual(
                branch,
                sub_slab,
            )
            for index in np.ndindex(lifted_residual.shape):
                worst = max(worst, _interval_abs_sup(lifted_residual[index]))
            checked += 1
    if checked == 0:
        return (np.inf, "no punctured generalized slabs were available")
    return float(worst), f"checked_subslabs={checked}; pieces_per_slab={pieces}; {detail}"


def _max_interval_generalized_fuchsian_projected_residual_on_punctured_shells(
    branch: FuchsianShapeBranch,
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    remainder_majorant: GeneralizedFuchsianRemainderMajorantCertificate | None,
) -> tuple[float, str]:
    if remainder_majorant is None:
        return (np.inf, "remainder majorant not supplied")
    tau_slabs, detail = _generalized_fuchsian_punctured_tau_slabs(
        tau_interval,
        isolation_radius=isolation_radius,
        initial_radius=float(remainder_majorant.initial_radius),
        shell_contraction=float(remainder_majorant.shell_contraction),
    )
    exact_constant_bound = _exact_constant_generalized_fuchsian_projected_residual_bound(
        branch,
        tau_slabs,
        remainder_majorant,
    )
    if exact_constant_bound is not None:
        bound, exact_detail = exact_constant_bound
        return (
            float(bound),
            (
                f"{exact_detail}; "
                f"checked_slabs={len(tau_slabs)}; {detail}"
            ),
        )
    worst = 0.0
    checked = 0
    pieces = 128
    for tau_slab in tau_slabs:
        for sub_slab in _subdivide_float_interval(tau_slab, pieces):
            lifted_residual = _interval_generalized_fuchsian_lifted_residual(
                branch,
                sub_slab,
            )
            tau_fourth = (sub_slab * sub_slab) * (sub_slab * sub_slab)
            scale = tau_fourth.scale(9.0).reciprocal()
            for index in np.ndindex(lifted_residual.shape):
                projected = _coerce_float_interval(lifted_residual[index]) * scale
                worst = max(worst, _interval_abs_sup(projected))
            checked += 1
    if checked == 0:
        return (np.inf, "no punctured generalized slabs were available")
    return float(worst), f"checked_subslabs={checked}; pieces_per_slab={pieces}; {detail}"


def _exact_constant_generalized_fuchsian_projected_residual_bound(
    branch: FuchsianShapeBranch,
    tau_slabs: tuple[FloatInterval, ...],
    remainder_majorant: GeneralizedFuchsianRemainderMajorantCertificate | None,
) -> tuple[float, str] | None:
    """Projected residual bound for exact constant-shape homothetic charts.

    Exact parabolic homothetic stop charts serialize as a generalized Fuchsian
    branch with only the central coefficient nonzero and a zero Banach
    remainder.  For that case the lifted residual is the central-configuration
    identity ``-2 Q - 9 A(Q)``, independent of the punctured shell.  Checking
    this identity directly avoids amplifying interval force-roundoff by
    ``tau^-4`` on the innermost shell while preserving the generic interval
    gate for nonconstant generalized branches.
    """

    if remainder_majorant is None:
        return None
    if not _generalized_remainder_majorant_is_exact_zero(remainder_majorant):
        return None
    if not tau_slabs:
        return None
    zero_index = branch.zero_index
    for index, coefficient in branch.coefficients.items():
        if index == zero_index:
            continue
        if not np.all(np.asarray(coefficient, dtype=float) == 0.0):
            return None
    min_abs_tau = min(
        min(abs(float(slab.lower)), abs(float(slab.upper)))
        for slab in tau_slabs
    )
    if not np.isfinite(min_abs_tau) or min_abs_tau <= 0.0:
        return None
    lifted_residual = (
        -2.0 * np.asarray(branch.central_shape, dtype=float)
        - 9.0 * accelerations(branch.central_shape, branch.masses)
    )
    lifted_bound = float(np.linalg.norm(lifted_residual, ord=np.inf))
    projected_bound = float(lifted_bound / (9.0 * min_abs_tau**4))
    if not np.isfinite(projected_bound):
        return None
    return (
        projected_bound,
        (
            "exact_constant_homothetic_projected_residual="
            f"{projected_bound}; exact_constant_lifted_residual={lifted_bound}; "
            f"min_abs_tau={min_abs_tau}"
        ),
    )


def _generalized_remainder_majorant_is_exact_zero(
    majorant: GeneralizedFuchsianRemainderMajorantCertificate,
) -> bool:
    if not (
        float(majorant.defect_bound) == 0.0
        and float(majorant.nonlinear_lipschitz_bound) == 0.0
        and float(majorant.remainder_ball_radius) == 0.0
    ):
        return False
    for _component, primitive in majorant.component_inputs:
        if (
            float(primitive.majorant_initial) != 0.0
            or float(primitive.first_shell_tail_bound) != 0.0
        ):
            return False
    return True


def _max_interval_generalized_fuchsian_zero_angular_momentum_on_punctured_shells(
    branch: FuchsianShapeBranch,
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    remainder_majorant: GeneralizedFuchsianRemainderMajorantCertificate | None,
) -> tuple[float, str]:
    if remainder_majorant is None:
        return (np.inf, "remainder majorant not supplied")
    tau_slabs, detail = _generalized_fuchsian_punctured_tau_slabs(
        tau_interval,
        isolation_radius=isolation_radius,
        initial_radius=float(remainder_majorant.initial_radius),
        shell_contraction=float(remainder_majorant.shell_contraction),
    )
    worst = 0.0
    checked = 0
    pieces = 128
    for tau_slab in tau_slabs:
        for sub_slab in _subdivide_float_interval(tau_slab, pieces):
            angular = _interval_generalized_fuchsian_centered_angular_momentum(
                branch,
                sub_slab,
            )
            if isinstance(angular, FloatInterval):
                worst = max(worst, _interval_abs_sup(angular))
            else:
                for component in angular:
                    worst = max(worst, _interval_abs_sup(component))
            checked += 1
    if checked == 0:
        return (np.inf, "no punctured generalized slabs were available")
    return float(worst), f"checked_subslabs={checked}; pieces_per_slab={pieces}; {detail}"


def _max_interval_generalized_fuchsian_center_of_mass_and_linear_momentum_on_punctured_shells(
    branch: FuchsianShapeBranch,
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    remainder_majorant: GeneralizedFuchsianRemainderMajorantCertificate | None,
) -> tuple[float, float, str]:
    if remainder_majorant is None:
        return (np.inf, np.inf, "remainder majorant not supplied")
    tau_slabs, detail = _generalized_fuchsian_punctured_tau_slabs(
        tau_interval,
        isolation_radius=isolation_radius,
        initial_radius=float(remainder_majorant.initial_radius),
        shell_contraction=float(remainder_majorant.shell_contraction),
    )
    worst_center = 0.0
    worst_momentum = 0.0
    checked = 0
    pieces = 128
    for tau_slab in tau_slabs:
        for sub_slab in _subdivide_float_interval(tau_slab, pieces):
            center, momentum = (
                _interval_generalized_fuchsian_center_of_mass_and_linear_momentum(
                    branch,
                    sub_slab,
                )
            )
            for component in center:
                worst_center = max(worst_center, _interval_abs_sup(component))
            for component in momentum:
                worst_momentum = max(worst_momentum, _interval_abs_sup(component))
            checked += 1
    if checked == 0:
        return (np.inf, np.inf, "no punctured generalized slabs were available")
    return (
        float(worst_center),
        float(worst_momentum),
        f"checked_subslabs={checked}; pieces_per_slab={pieces}; {detail}",
    )


def _subdivide_float_interval(
    interval: FloatInterval,
    pieces: int,
) -> tuple[FloatInterval, ...]:
    pieces = int(pieces)
    if pieces <= 0:
        raise ValueError("pieces must be positive")
    width = (interval.upper - interval.lower) / pieces
    if width <= 0.0:
        return ()
    return tuple(
        FloatInterval(
            interval.lower + piece * width,
            interval.lower + (piece + 1) * width,
        )
        for piece in range(pieces)
    )


def _generalized_fuchsian_punctured_tau_slabs(
    tau_interval: tuple[float, float],
    *,
    isolation_radius: float,
    initial_radius: float,
    shell_contraction: float,
) -> tuple[tuple[FloatInterval, ...], str]:
    initial_radius = float(initial_radius)
    shell_contraction = float(shell_contraction)
    radius_upper = min(
        float(isolation_radius),
        max(abs(float(tau_interval[0])), abs(float(tau_interval[1]))),
    )
    if not (
        np.isfinite(initial_radius)
        and np.isfinite(shell_contraction)
        and np.isfinite(radius_upper)
        and 0.0 < shell_contraction < 1.0
        and 0.0 < initial_radius <= radius_upper
    ):
        raise ValueError(
            "invalid generalized Fuchsian punctured shell radii: "
            f"initial_radius={initial_radius}; "
            f"shell_contraction={shell_contraction}; radius_upper={radius_upper}"
        )
    radius_slabs = (
        (initial_radius, radius_upper),
        (initial_radius * shell_contraction, initial_radius),
    )
    tau_slabs: list[FloatInterval] = []
    for lower, upper in radius_slabs:
        if not 0.0 < lower < upper:
            continue
        tau_slabs.append(FloatInterval(lower, upper))
        tau_slabs.append(FloatInterval(-upper, -lower))
    return (
        tuple(tau_slabs),
        f"initial_radius={initial_radius}; radius_upper={radius_upper}",
    )


def _interval_generalized_fuchsian_lifted_residual(
    branch: FuchsianShapeBranch,
    tau_slab: FloatInterval,
) -> np.ndarray:
    shape, first, second = _interval_generalized_fuchsian_tau_derivatives(
        branch,
        tau_slab,
    )
    tau_squared = tau_slab * tau_slab
    left = _interval_vector_add(
        _interval_vector_scale_by_interval(second, tau_squared),
        _interval_vector_scale_by_interval(first, tau_slab.scale(2.0)),
        _interval_vector_scale(shape, -2.0),
    )
    acceleration = _interval_newton_accelerations(shape, branch.masses)
    return _interval_vector_sub(left, _interval_vector_scale(acceleration, 9.0))


def _interval_generalized_fuchsian_center_of_mass_and_linear_momentum(
    branch: FuchsianShapeBranch,
    tau_slab: FloatInterval,
) -> tuple[np.ndarray, np.ndarray]:
    weighted_shape, weighted_first = (
        _interval_generalized_fuchsian_weighted_shape_and_first(
            branch,
            tau_slab,
        )
    )
    masses = np.asarray(branch.masses, dtype=float)
    total_mass = float(np.sum(masses))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("total mass must be positive")
    tau_squared = tau_slab * tau_slab
    center = _interval_vector_scale_by_interval(
        weighted_shape,
        tau_squared.scale(1.0 / total_mass),
    )
    numerator = _interval_vector_add(
        _interval_vector_scale_by_interval(weighted_shape, tau_slab.scale(2.0)),
        _interval_vector_scale_by_interval(weighted_first, tau_squared),
    )
    momentum = _interval_vector_scale_by_interval(
        numerator,
        tau_squared.scale(3.0).reciprocal(),
    )
    return center, momentum


def _interval_generalized_fuchsian_weighted_shape_and_first(
    branch: FuchsianShapeBranch,
    tau_slab: FloatInterval,
) -> tuple[np.ndarray, np.ndarray]:
    if tau_slab.lower > 0.0:
        radius = tau_slab
        sign = 1.0
    elif tau_slab.upper < 0.0:
        radius = FloatInterval(-tau_slab.upper, -tau_slab.lower)
        sign = -1.0
    else:
        raise ValueError("generalized Fuchsian interval check needs a punctured tau slab")
    masses = np.asarray(branch.masses, dtype=float)
    dimension = int(branch.central_shape.shape[1])
    weighted_shape = _interval_zero_array((dimension,))
    weighted_first_radius = _interval_zero_array((dimension,))
    for index, coefficient in branch.coefficients.items():
        exponent = branch.exponent(index)
        if not np.isfinite(exponent):
            raise ValueError("generalized Fuchsian exponent must be finite")
        weighted_coefficient = np.zeros(dimension, dtype=float)
        for body, mass in enumerate(masses):
            weighted_coefficient += float(mass) * np.asarray(
                coefficient[body],
                dtype=float,
            )
        coefficient_interval = _interval_point_array(weighted_coefficient)
        weighted_shape = _interval_vector_add(
            weighted_shape,
            _interval_vector_scale_by_interval(
                coefficient_interval,
                radius.positive_power(exponent),
            ),
        )
        if index == branch.zero_index:
            continue
        weighted_first_radius = _interval_vector_add(
            weighted_first_radius,
            _interval_vector_scale_by_interval(
                coefficient_interval,
                radius.positive_power(exponent - 1.0).scale(exponent),
            ),
        )
    return weighted_shape, _interval_vector_scale(weighted_first_radius, sign)


def _interval_generalized_fuchsian_centered_angular_momentum(
    branch: FuchsianShapeBranch,
    tau_slab: FloatInterval,
) -> FloatInterval | np.ndarray:
    shape, first, _second = _interval_generalized_fuchsian_tau_derivatives(
        branch,
        tau_slab,
    )
    masses = np.asarray(branch.masses, dtype=float)
    centered_shape = _interval_centered_rows(shape, masses)
    centered_first = _interval_centered_rows(first, masses)
    tau_squared_over_three = (tau_slab * tau_slab).scale(1.0 / 3.0)
    if centered_shape.shape[1] == 2:
        angular = FloatInterval.point(0.0)
        for body, mass in enumerate(masses):
            angular = angular + _interval_planar_wedge(
                centered_shape[body],
                centered_first[body],
            ).scale(float(mass))
        return angular * tau_squared_over_three
    if centered_shape.shape[1] == 3:
        angular_vector = _interval_zero_array((3,))
        for body, mass in enumerate(masses):
            angular_vector = _interval_vector_add(
                angular_vector,
                _interval_vector_scale(
                    _interval_cross(centered_shape[body], centered_first[body]),
                    float(mass),
                ),
            )
        return _interval_vector_scale_by_interval(
            angular_vector,
            tau_squared_over_three,
        )
    raise ValueError("generalized Fuchsian angular check supports only 2D and 3D shapes")


def _interval_generalized_fuchsian_tau_derivatives(
    branch: FuchsianShapeBranch,
    tau_slab: FloatInterval,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tau_slab.lower > 0.0:
        radius = tau_slab
        sign = 1.0
    elif tau_slab.upper < 0.0:
        radius = FloatInterval(-tau_slab.upper, -tau_slab.lower)
        sign = -1.0
    else:
        raise ValueError("generalized Fuchsian interval check needs a punctured tau slab")
    shape, radius_first, second = _interval_generalized_fuchsian_radius_derivatives(
        branch,
        radius,
    )
    return shape, _interval_vector_scale(radius_first, sign), second


def _interval_generalized_fuchsian_radius_derivatives(
    branch: FuchsianShapeBranch,
    radius: FloatInterval,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if radius.lower <= 0.0:
        raise ValueError("generalized Fuchsian radius interval must be positive")
    shape = _interval_zero_array(branch.central_shape.shape)
    first = _interval_zero_array(branch.central_shape.shape)
    second = _interval_zero_array(branch.central_shape.shape)
    for index, coefficient in branch.coefficients.items():
        exponent = branch.exponent(index)
        if not np.isfinite(exponent):
            raise ValueError("generalized Fuchsian exponent must be finite")
        coefficient_interval = _interval_point_array(np.asarray(coefficient, dtype=float))
        shape = _interval_vector_add(
            shape,
            _interval_vector_scale_by_interval(
                coefficient_interval,
                radius.positive_power(exponent),
            ),
        )
        if index == branch.zero_index:
            continue
        first = _interval_vector_add(
            first,
            _interval_vector_scale_by_interval(
                coefficient_interval,
                radius.positive_power(exponent - 1.0).scale(exponent),
            ),
        )
        second = _interval_vector_add(
            second,
            _interval_vector_scale_by_interval(
                coefficient_interval,
                radius.positive_power(exponent - 2.0).scale(
                    exponent * (exponent - 1.0),
                ),
            ),
        )
    return shape, first, second


def _interval_fuchsian_center_of_mass_and_linear_momentum(
    branch: FiniteFuchsianLogBranch,
    tau_slab: FloatInterval,
) -> tuple[np.ndarray, np.ndarray]:
    shape, first, _second = _interval_fuchsian_log_branch_tau_derivatives(
        branch,
        tau_slab,
    )
    masses = np.asarray(branch.masses, dtype=float)
    total_mass = float(np.sum(masses))
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("total mass must be positive")
    weighted_shape = _interval_weighted_row_sum(shape, masses)
    weighted_first = _interval_weighted_row_sum(first, masses)
    tau_squared = tau_slab * tau_slab
    center = _interval_vector_scale_by_interval(
        weighted_shape,
        tau_squared.scale(1.0 / total_mass),
    )
    numerator = _interval_vector_add(
        _interval_vector_scale_by_interval(weighted_shape, tau_slab.scale(2.0)),
        _interval_vector_scale_by_interval(weighted_first, tau_squared),
    )
    momentum = _interval_vector_scale_by_interval(
        numerator,
        tau_squared.scale(3.0).reciprocal(),
    )
    return center, momentum


def _interval_weighted_row_sum(values: np.ndarray, masses: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if values.ndim != 2 or values.shape[0] != masses.size:
        raise ValueError("values must have one row per mass")
    out = _interval_zero_array((values.shape[1],))
    for body, mass in enumerate(masses):
        for axis in range(values.shape[1]):
            out[axis] = out[axis] + _coerce_float_interval(
                values[body, axis],
            ).scale(float(mass))
    return out


def _interval_weighted_squared_norm(values: np.ndarray, masses: np.ndarray) -> FloatInterval:
    values = np.asarray(values, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if values.ndim != 2 or values.shape[0] != masses.size:
        raise ValueError("values must have one row per mass")
    out = FloatInterval.point(0.0)
    for body, mass in enumerate(masses):
        out = out + _interval_dot(values[body], values[body]).scale(float(mass))
    return out


def _interval_pairwise_newtonian_potential(
    positions: np.ndarray,
    masses: np.ndarray,
) -> FloatInterval:
    positions = np.asarray(positions, dtype=object)
    masses = np.asarray(masses, dtype=float)
    if positions.ndim != 2 or positions.shape[0] != masses.size:
        raise ValueError("positions must have one row per mass")
    potential = FloatInterval.point(0.0)
    for first in range(positions.shape[0]):
        for second in range(first + 1, positions.shape[0]):
            delta = _interval_vector_sub(positions[second], positions[first])
            radius_squared = _interval_dot(delta, delta)
            inverse_radius = radius_squared.positive_power(-0.5)
            potential = potential + inverse_radius.scale(
                float(masses[first] * masses[second]),
            )
    return potential


def _interval_fuchsian_centered_angular_momentum(
    branch: FiniteFuchsianLogBranch,
    tau_slab: FloatInterval,
) -> FloatInterval | np.ndarray:
    shape, first, _second = _interval_fuchsian_log_branch_tau_derivatives(
        branch,
        tau_slab,
    )
    masses = np.asarray(branch.masses, dtype=float)
    centered_shape = _interval_centered_rows(shape, masses)
    centered_first = _interval_centered_rows(first, masses)
    tau_squared_over_three = (tau_slab * tau_slab).scale(1.0 / 3.0)
    if centered_shape.shape[1] == 2:
        angular = FloatInterval.point(0.0)
        for body, mass in enumerate(masses):
            angular = angular + _interval_planar_wedge(
                centered_shape[body],
                centered_first[body],
            ).scale(float(mass))
        return angular * tau_squared_over_three
    if centered_shape.shape[1] == 3:
        angular_vector = _interval_zero_array((3,))
        for body, mass in enumerate(masses):
            angular_vector = _interval_vector_add(
                angular_vector,
                _interval_vector_scale(
                    _interval_cross(centered_shape[body], centered_first[body]),
                    float(mass),
                ),
            )
        return _interval_vector_scale_by_interval(
            angular_vector,
            tau_squared_over_three,
        )
    raise ValueError("Fuchsian angular check supports only 2D and 3D shapes")


def _interval_fuchsian_lifted_residual(
    branch: FiniteFuchsianLogBranch,
    tau_slab: FloatInterval,
) -> np.ndarray:
    shape, first, second = _interval_fuchsian_log_branch_tau_derivatives(
        branch,
        tau_slab,
    )
    tau_squared = tau_slab * tau_slab
    left = _interval_vector_add(
        _interval_vector_scale_by_interval(second, tau_squared),
        _interval_vector_scale_by_interval(first, tau_slab.scale(2.0)),
        _interval_vector_scale(shape, -2.0),
    )
    acceleration = _interval_newton_accelerations(shape, branch.masses)
    return _interval_vector_sub(left, _interval_vector_scale(acceleration, 9.0))


def _interval_fuchsian_log_branch_tau_derivatives(
    branch: FiniteFuchsianLogBranch,
    tau_slab: FloatInterval,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tau_slab.lower > 0.0:
        radius = tau_slab
        sign = 1.0
    elif tau_slab.upper < 0.0:
        radius = FloatInterval(-tau_slab.upper, -tau_slab.lower)
        sign = -1.0
    else:
        raise ValueError("Fuchsian interval check needs a punctured tau slab")
    shape, radius_first, second = _interval_fuchsian_log_branch_radius_derivatives(
        branch,
        radius,
    )
    return shape, _interval_vector_scale(radius_first, sign), second


def _interval_fuchsian_log_branch_radius_derivatives(
    branch: FiniteFuchsianLogBranch,
    radius: FloatInterval,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if radius.lower <= 0.0:
        raise ValueError("Fuchsian radius interval must be positive")
    central = np.asarray(branch.central_shape, dtype=float)
    shape = _interval_point_array(central)
    first = _interval_vector_scale(
        _interval_point_array(central),
        0.0,
    )
    second = _interval_vector_scale(
        _interval_point_array(central),
        0.0,
    )
    scale = float(branch.scale_coefficient)
    radius_squared = radius * radius
    shape = _interval_vector_add(
        shape,
        _interval_vector_scale_by_interval(
            _interval_point_array(central),
            radius_squared.scale(scale),
        ),
    )
    first = _interval_vector_add(
        first,
        _interval_vector_scale_by_interval(
            _interval_point_array(central),
            radius.scale(2.0 * scale),
        ),
    )
    second = _interval_vector_add(
        second,
        _interval_vector_scale(_interval_point_array(central), 2.0 * scale),
    )
    for term in branch.terms:
        term_value, term_first, term_second = (
            _interval_fuchsian_log_term_radius_derivatives(term, radius)
        )
        shape = _interval_vector_add(shape, term_value)
        first = _interval_vector_add(first, term_first)
        second = _interval_vector_add(second, term_second)
    return shape, first, second


def _interval_fuchsian_log_term_radius_derivatives(
    term: FuchsianLogTerm,
    radius: FloatInterval,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logarithm = _interval_log_positive(radius)
    polynomial, log_first, log_second = (
        _interval_fuchsian_log_polynomial_derivatives(term, logarithm)
    )
    power = float(term.power)
    value_factor = radius.positive_power(power)
    first_factor = radius.positive_power(power - 1.0)
    second_factor = radius.positive_power(power - 2.0)
    first_poly = _interval_vector_add(
        _interval_vector_scale(polynomial, power),
        log_first,
    )
    second_poly = _interval_vector_add(
        _interval_vector_scale(polynomial, power * (power - 1.0)),
        _interval_vector_scale(log_first, 2.0 * power - 1.0),
        log_second,
    )
    return (
        _interval_vector_scale_by_interval(polynomial, value_factor),
        _interval_vector_scale_by_interval(first_poly, first_factor),
        _interval_vector_scale_by_interval(second_poly, second_factor),
    )


def _interval_fuchsian_log_polynomial_derivatives(
    term: FuchsianLogTerm,
    logarithm: FloatInterval,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    polynomial = _interval_zero_array(term.body_shape)
    first = _interval_zero_array(term.body_shape)
    second = _interval_zero_array(term.body_shape)
    for log_power, coefficient in term.coefficients_by_log_power.items():
        coefficient = np.asarray(coefficient, dtype=float)
        polynomial = _interval_vector_add(
            polynomial,
            _interval_vector_scale_by_interval(
                _interval_point_array(coefficient),
                _interval_nonnegative_integer_power(logarithm, int(log_power)),
            ),
        )
        if log_power >= 1:
            first = _interval_vector_add(
                first,
                _interval_vector_scale_by_interval(
                    _interval_point_array(coefficient),
                    _interval_nonnegative_integer_power(
                        logarithm,
                        int(log_power) - 1,
                    ).scale(float(log_power)),
                ),
            )
        if log_power >= 2:
            second = _interval_vector_add(
                second,
                _interval_vector_scale_by_interval(
                    _interval_point_array(coefficient),
                    _interval_nonnegative_integer_power(
                        logarithm,
                        int(log_power) - 2,
                    ).scale(float(log_power * (log_power - 1))),
                ),
            )
    return polynomial, first, second


def _interval_point_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        out[index] = FloatInterval.point(float(values[index]))
    return out


def _interval_log_positive(value: FloatInterval) -> FloatInterval:
    if value.lower <= 0.0:
        raise ValueError("log interval must be positive")
    return FloatInterval(
        float(np.nextafter(np.log(value.lower), -np.inf)),
        float(np.nextafter(np.log(value.upper), np.inf)),
    )


def _interval_nonnegative_integer_power(
    value: FloatInterval,
    exponent: int,
) -> FloatInterval:
    exponent = int(exponent)
    if exponent < 0:
        raise ValueError("integer interval power must be nonnegative")
    result = FloatInterval.point(1.0)
    for _ in range(exponent):
        result = result * value
    return result


def _punctured_tau_samples(
    tau_interval: tuple[float, float],
    *,
    sample_count: int,
) -> tuple[float, ...]:
    lower, upper = tau_interval
    if sample_count < 2 or not lower < 0.0 < upper:
        return ()
    raw_samples = np.linspace(lower, upper, int(sample_count))
    epsilon = max(1.0e-15, 1.0e-12 * max(abs(lower), abs(upper)))
    samples = tuple(float(value) for value in raw_samples if abs(value) > epsilon)
    if samples:
        return samples
    return (float(0.5 * lower), float(0.5 * upper))


def _spatial_centered_angular_momentum_norm(
    branch: FiniteFuchsianLogBranch,
    tau: float,
) -> float:
    positions = branch.positions_at_tau(tau)
    velocities = branch.velocities_at_tau(tau)
    masses = np.asarray(branch.masses, dtype=float)
    center = np.average(positions, axis=0, weights=masses)
    center_velocity = np.average(velocities, axis=0, weights=masses)
    angular = np.zeros(3, dtype=float)
    for mass, position, velocity in zip(masses, positions, velocities):
        angular += mass * np.cross(position - center, velocity - center_velocity)
    return float(np.linalg.norm(angular))


def _derivative_coefficients(coefficients: np.ndarray) -> np.ndarray:
    if coefficients.shape[0] <= 1:
        return np.zeros_like(coefficients)
    out = np.empty((coefficients.shape[0] - 1, *coefficients.shape[1:]), dtype=float)
    for n in range(out.shape[0]):
        out[n] = (n + 1) * coefficients[n + 1]
    return out


def _derivative_scalar_coefficients(coefficients: np.ndarray) -> np.ndarray:
    if coefficients.shape[0] <= 1:
        return np.zeros_like(coefficients)
    out = np.empty((coefficients.shape[0] - 1,), dtype=float)
    for n in range(out.shape[0]):
        out[n] = (n + 1) * coefficients[n + 1]
    return out


def _evaluate_coefficients(coefficients: np.ndarray, time: float) -> np.ndarray:
    if coefficients.size == 0:
        return np.asarray((), dtype=float)
    value = np.zeros(coefficients.shape[1:], dtype=float)
    for coefficient in coefficients[::-1]:
        value = value * time + coefficient
    return value


def _evaluate_scalar_coefficients(coefficients: np.ndarray, time: float) -> float:
    if coefficients.size == 0:
        return np.inf
    value = 0.0
    for coefficient in coefficients[::-1]:
        value = value * time + float(coefficient)
    return float(value)


def _event_isolation_sign_pattern(event_type: str) -> dict[str, int] | None:
    if event_type == "spatial_ks_rho_exit":
        return {"left": -1, "right": 1, "derivative": 1, "pre_event": -1}
    if event_type in {
        "spatial_ordinary_ks_entry",
        "spatial_ks_competing_binary_entry",
    }:
        return {"left": 1, "right": -1, "derivative": -1, "pre_event": 1}
    return None


def _event_isolation_supported_sources(event_type: str) -> set[str]:
    if event_type == "spatial_ks_rho_exit":
        return {"spatial_ks_binary_interval_taylor"}
    if event_type == "spatial_ordinary_ks_entry":
        return {"spatial_ordinary_interval_taylor"}
    if event_type == "spatial_ks_competing_binary_entry":
        return {"spatial_ks_binary_interval_taylor"}
    return set()


def _event_isolation_pair_valid(certificate: EventIsolationCertificate) -> bool:
    if certificate.event_type == "spatial_ks_rho_exit":
        return tuple(certificate.pair) == (-1, -1)
    if certificate.event_type in {
        "spatial_ordinary_ks_entry",
        "spatial_ks_competing_binary_entry",
    }:
        pair = tuple(certificate.pair)
        return bool(
            len(pair) == 2
            and pair[0] != pair[1]
            and set(pair).issubset({0, 1, 2})
        )
    return False


def _event_interval_coefficients(
    coefficient_intervals: tuple[tuple[float, float], ...],
) -> tuple[FloatInterval, ...]:
    return tuple(FloatInterval(float(lower), float(upper)) for lower, upper in coefficient_intervals)


def _event_rational_interval_coefficients(
    coefficient_intervals: tuple[tuple[float, float], ...],
) -> tuple[RationalInterval, ...]:
    return tuple(
        RationalInterval.from_float_interval(float(lower), float(upper))
        for lower, upper in coefficient_intervals
    )


def _signed_polynomial_range_certified(
    coefficients: tuple[FloatInterval, ...],
    lower: float,
    upper: float,
    *,
    sign: int,
    max_subintervals: int = 512,
) -> bool:
    if sign == 0 or not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        return False
    if lower == upper:
        value = interval_polynomial_eval(coefficients, FloatInterval.point(lower))
        return interval_sign(value) == sign
    pieces = 1
    while pieces <= max_subintervals:
        certified = True
        for index in range(pieces):
            piece_lower = lower + (upper - lower) * index / pieces
            piece_upper = lower + (upper - lower) * (index + 1) / pieces
            value = interval_polynomial_eval(
                coefficients,
                FloatInterval(piece_lower, piece_upper),
            )
            if interval_sign(value) != sign:
                certified = False
                break
        if certified:
            return True
        pieces *= 2
    return False


def _signed_rational_polynomial_range_certified(
    coefficients: tuple[RationalInterval, ...],
    lower: float,
    upper: float,
    *,
    sign: int,
    max_subintervals: int = 512,
) -> bool:
    if sign == 0 or not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        return False
    lower_interval = RationalInterval.from_float_interval(lower)
    upper_interval = RationalInterval.from_float_interval(upper)
    rational_lower = lower_interval.lower
    rational_upper = upper_interval.lower
    if rational_lower == rational_upper:
        value = rational_interval_polynomial_eval(
            coefficients,
            RationalInterval(rational_lower, rational_upper),
        )
        return rational_interval_sign(value) == sign
    pieces = 1
    while pieces <= max_subintervals:
        certified = True
        width = rational_upper - rational_lower
        for index in range(pieces):
            piece_lower = rational_lower + width * index / pieces
            piece_upper = rational_lower + width * (index + 1) / pieces
            value = rational_interval_polynomial_eval(
                coefficients,
                RationalInterval(piece_lower, piece_upper),
            )
            if rational_interval_sign(value) != sign:
                certified = False
                break
        if certified:
            return True
        pieces *= 2
    return False


def _array_sup_norm(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.inf
    return float(np.max(np.abs(array)))

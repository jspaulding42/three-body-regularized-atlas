"""Serializable certificate records for independent proof checking.

The constructor modules build rich Python objects while searching for charts.
This module defines a smaller certificate language meant to be checked without
trusting those search objects.  The first supported primitive is an ordinary
Newtonian Taylor chart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .binary_chart import (
    IntervalRegularizedBinaryCollisionChartState,
    RegularizedBinaryCollisionChartState,
    planar_to_regularized_binary_collision_chart,
)
from .binary_series import RegularizedBinaryTaylorSolution
from .binary_series import construct_regularized_binary_taylor_solution
from .event_recurrence import PrimitiveCauchyTailInput
from .fuchsian import (
    FiniteFuchsianLogBranch,
    FiniteFuchsianLogPrimitiveCauchyInputs,
    FiniteFuchsianLogTotalCollisionIsolationCertificate,
    FuchsianShapeBranch,
)
from .ks_binary_series import (
    SpatialKSEntryEventCertificate,
    SpatialKSBinaryTaylorSolution,
    SpatialKSRhoExitEventCertificate,
)
from .intervals import FloatInterval, interval_polynomial_eval
from .series import TaylorSolution, construct_taylor_solution


@dataclass(frozen=True)
class OrdinaryTaylorChartCertificate:
    """Serialized ordinary Taylor chart data for the independent checker."""

    certificate_id: str
    chart_id: str
    chart_type: str
    masses: tuple[float, ...]
    position_coefficients: tuple[tuple[tuple[float, ...], ...], ...]
    velocity_coefficients: tuple[tuple[tuple[float, ...], ...], ...]
    parameter_interval: tuple[float, float]
    physical_time_interval: tuple[float, float]
    coefficient_tolerance: float
    residual_tolerance: float
    tail_bound: float
    sample_count: int = 7
    source: str = "serialized_ordinary_taylor_chart"

    @property
    def order(self) -> int:
        return len(self.position_coefficients) - 1

    @property
    def body_count(self) -> int:
        return len(self.position_coefficients[0]) if self.position_coefficients else 0

    @property
    def dimension(self) -> int:
        if not self.position_coefficients or not self.position_coefficients[0]:
            return 0
        return len(self.position_coefficients[0][0])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrdinaryTaylorChartCertificate":
        return cls(
            certificate_id=str(data.get("certificate_id", "")),
            chart_id=str(data.get("chart_id", "")),
            chart_type=str(data.get("chart_type", "")),
            masses=_tuple_of_float(data.get("masses", ())),
            position_coefficients=_coefficient_tuple(
                data.get("position_coefficients", ()),
            ),
            velocity_coefficients=_coefficient_tuple(
                data.get("velocity_coefficients", ()),
            ),
            parameter_interval=_pair_of_float(data.get("parameter_interval", ())),
            physical_time_interval=_pair_of_float(
                data.get("physical_time_interval", ()),
            ),
            coefficient_tolerance=float(data.get("coefficient_tolerance", np.inf)),
            residual_tolerance=float(data.get("residual_tolerance", np.inf)),
            tail_bound=float(data.get("tail_bound", np.inf)),
            sample_count=int(data.get("sample_count", 7)),
            source=str(data.get("source", "serialized_ordinary_taylor_chart")),
        )


@dataclass(frozen=True)
class OrdinaryChartTransitionCertificate:
    """Serialized ordinary-to-ordinary chart handoff certificate."""

    transition_id: str
    source_chart_id: str
    target_chart_id: str
    transition_type: str
    handoff_time: float
    position_tolerance: float
    velocity_tolerance: float
    source: str = "serialized_ordinary_chart_transition"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrdinaryChartTransitionCertificate":
        return cls(
            transition_id=str(data.get("transition_id", "")),
            source_chart_id=str(data.get("source_chart_id", "")),
            target_chart_id=str(data.get("target_chart_id", "")),
            transition_type=str(data.get("transition_type", "")),
            handoff_time=float(data.get("handoff_time", np.inf)),
            position_tolerance=float(data.get("position_tolerance", np.inf)),
            velocity_tolerance=float(data.get("velocity_tolerance", np.inf)),
            source=str(data.get("source", "serialized_ordinary_chart_transition")),
        )


@dataclass(frozen=True)
class InitialValueProblemBindingCertificate:
    """Bind a serialized chart to a specific three-body initial value problem.

    This is an exact binary-float problem record.  It does not claim arbitrary
    computable-real input support; the checker embeds these finite values
    exactly into its rational comparison backend where available.
    """

    binding_id: str
    chart_id: str
    masses: tuple[float, ...]
    initial_time: float
    chart_parameter: float
    positions: tuple[tuple[float, ...], ...]
    velocities: tuple[tuple[float, ...], ...]
    time_tolerance: float
    position_tolerance: float
    velocity_tolerance: float
    source: str = "serialized_initial_value_problem_binding"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InitialValueProblemBindingCertificate":
        return cls(
            binding_id=str(data.get("binding_id", "")),
            chart_id=str(data.get("chart_id", "")),
            masses=_tuple_of_float(data.get("masses", ())),
            initial_time=float(data.get("initial_time", np.inf)),
            chart_parameter=float(data.get("chart_parameter", np.inf)),
            positions=_coefficient_matrix(data.get("positions", ())),
            velocities=_coefficient_matrix(data.get("velocities", ())),
            time_tolerance=float(data.get("time_tolerance", np.inf)),
            position_tolerance=float(data.get("position_tolerance", np.inf)),
            velocity_tolerance=float(data.get("velocity_tolerance", np.inf)),
            source=str(
                data.get("source", "serialized_initial_value_problem_binding")
            ),
        )


@dataclass(frozen=True)
class OrdinaryAposterioriTubeCertificate:
    """Candidate a-posteriori existence tube around an ordinary chart.

    Bounds use the phase-space infinity norm.  The checker recomputes the
    polynomial defect, nominal pair-distance floor, Newton-field Lipschitz
    bound, and Gronwall enclosure; the supplied numbers are only admissible
    caps/radii.
    """

    tube_id: str
    chart_id: str
    anchor_parameter: float
    initial_error_bound: float
    tube_radius: float
    max_defect_bound: float
    max_lipschitz_bound: float
    source: str = "serialized_ordinary_aposteriori_tube"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrdinaryAposterioriTubeCertificate":
        return cls(
            tube_id=str(data.get("tube_id", "")),
            chart_id=str(data.get("chart_id", "")),
            anchor_parameter=float(data.get("anchor_parameter", np.inf)),
            initial_error_bound=float(data.get("initial_error_bound", np.inf)),
            tube_radius=float(data.get("tube_radius", np.inf)),
            max_defect_bound=float(data.get("max_defect_bound", np.inf)),
            max_lipschitz_bound=float(data.get("max_lipschitz_bound", np.inf)),
            source=str(data.get("source", "serialized_ordinary_aposteriori_tube")),
        )


@dataclass(frozen=True)
class WeightedOrdinaryAposterioriTubeCertificate:
    """Candidate ordinary a-posteriori tube in a weighted phase-space norm."""

    tube_id: str
    chart_id: str
    anchor_parameter: float
    initial_position_error_bound: float
    initial_velocity_error_bound: float
    position_radius: float
    velocity_radius: float
    max_scaled_defect_bound: float
    max_scaled_lipschitz_bound: float
    source: str = "serialized_weighted_ordinary_aposteriori_tube"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "WeightedOrdinaryAposterioriTubeCertificate":
        return cls(
            tube_id=str(data.get("tube_id", "")),
            chart_id=str(data.get("chart_id", "")),
            anchor_parameter=float(data.get("anchor_parameter", np.inf)),
            initial_position_error_bound=float(
                data.get("initial_position_error_bound", np.inf)
            ),
            initial_velocity_error_bound=float(
                data.get("initial_velocity_error_bound", np.inf)
            ),
            position_radius=float(data.get("position_radius", np.inf)),
            velocity_radius=float(data.get("velocity_radius", np.inf)),
            max_scaled_defect_bound=float(
                data.get("max_scaled_defect_bound", np.inf)
            ),
            max_scaled_lipschitz_bound=float(
                data.get("max_scaled_lipschitz_bound", np.inf)
            ),
            source=str(
                data.get(
                    "source", "serialized_weighted_ordinary_aposteriori_tube"
                )
            ),
        )


@dataclass(frozen=True)
class OrdinaryEnclosureTransitionCertificate:
    """Continuation handoff between two validated ordinary chart tubes."""

    transition_id: str
    source_chart_id: str
    target_chart_id: str
    source_parameter: float
    target_parameter: float
    handoff_time: float
    max_time_gap: float
    source: str = "serialized_ordinary_enclosure_transition"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrdinaryEnclosureTransitionCertificate":
        return cls(
            transition_id=str(data.get("transition_id", "")),
            source_chart_id=str(data.get("source_chart_id", "")),
            target_chart_id=str(data.get("target_chart_id", "")),
            source_parameter=float(data.get("source_parameter", np.inf)),
            target_parameter=float(data.get("target_parameter", np.inf)),
            handoff_time=float(data.get("handoff_time", np.inf)),
            max_time_gap=float(data.get("max_time_gap", np.inf)),
            source=str(data.get("source", "serialized_ordinary_enclosure_transition")),
        )


@dataclass(frozen=True)
class ValidatedOrdinaryIVPChainCertificate:
    """Finite forward chain of ordinary exact-solution enclosure charts."""

    chain_id: str
    chart_ids: tuple[str, ...]
    transition_ids: tuple[str, ...]
    target_physical_time_interval: tuple[float, float]
    orientation: str = "forward"
    source: str = "serialized_validated_ordinary_ivp_chain"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidatedOrdinaryIVPChainCertificate":
        return cls(
            chain_id=str(data.get("chain_id", "")),
            chart_ids=tuple(str(value) for value in data.get("chart_ids", ())),
            transition_ids=tuple(
                str(value) for value in data.get("transition_ids", ())
            ),
            target_physical_time_interval=_pair_of_float(
                data.get("target_physical_time_interval", ())
            ),
            orientation=str(data.get("orientation", "forward")),
            source=str(data.get("source", "serialized_validated_ordinary_ivp_chain")),
        )


@dataclass(frozen=True)
class PlanarLCAposterioriTubeCertificate:
    """A-posteriori lifted solution tube for one planar LC chart."""

    tube_id: str
    chart_id: str
    anchor_parameter: float
    initial_error_bound: float
    tube_radius: float
    max_defect_bound: float
    max_lipschitz_bound: float
    require_pair_energy_constraint: bool = False
    source: str = "serialized_planar_lc_aposteriori_tube"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanarLCAposterioriTubeCertificate":
        return cls(
            tube_id=str(data.get("tube_id", "")),
            chart_id=str(data.get("chart_id", "")),
            anchor_parameter=float(data.get("anchor_parameter", np.inf)),
            initial_error_bound=float(data.get("initial_error_bound", np.inf)),
            tube_radius=float(data.get("tube_radius", np.inf)),
            max_defect_bound=float(data.get("max_defect_bound", np.inf)),
            max_lipschitz_bound=float(data.get("max_lipschitz_bound", np.inf)),
            require_pair_energy_constraint=bool(
                data.get("require_pair_energy_constraint", False)
            ),
            source=str(data.get("source", "serialized_planar_lc_aposteriori_tube")),
        )


@dataclass(frozen=True)
class PlanarLCExactOverlapAnchorCertificate:
    """Bind two zero-error LC tubes at one exact gauge-related anchor."""

    overlap_id: str
    source_chart_id: str
    source_tube_id: str
    target_chart_id: str
    target_tube_id: str
    source_anchor_parameter: float
    target_anchor_parameter: float
    source: str = "serialized_planar_lc_exact_overlap_anchor"

    def to_dict(self) -> dict[str, Any]:
        return {
            "overlap_id": self.overlap_id,
            "source_chart_id": self.source_chart_id,
            "source_tube_id": self.source_tube_id,
            "target_chart_id": self.target_chart_id,
            "target_tube_id": self.target_tube_id,
            "source_anchor_parameter": self.source_anchor_parameter,
            "target_anchor_parameter": self.target_anchor_parameter,
            "source": self.source,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "PlanarLCExactOverlapAnchorCertificate":
        # Preserve malformed serialized values for the checker to reject.  In
        # particular, do not coerce truthy objects, booleans, or numeric text
        # into identifiers or anchor parameters.
        return cls(
            overlap_id=data.get("overlap_id", ""),
            source_chart_id=data.get("source_chart_id", ""),
            source_tube_id=data.get("source_tube_id", ""),
            target_chart_id=data.get("target_chart_id", ""),
            target_tube_id=data.get("target_tube_id", ""),
            source_anchor_parameter=data.get("source_anchor_parameter", np.inf),
            target_anchor_parameter=data.get("target_anchor_parameter", np.inf),
            source=data.get(
                "source", "serialized_planar_lc_exact_overlap_anchor"
            ),
        )


@dataclass(frozen=True)
class PlanarLCExactGaugeAtlasCertificate:
    """Bind raw zero-error LC vertices and exact-overlap records into an atlas.

    The three identifier sequences are positional manifests.  They carry no
    supplied gauge bits: the checker must recompute every overlap and derive
    the graph presented to the exact F2 kernel.
    """

    atlas_id: str
    chart_ids: tuple[str, ...]
    tube_ids: tuple[str, ...]
    overlap_ids: tuple[str, ...]
    source: str = "serialized_planar_lc_exact_gauge_atlas"

    def to_dict(self) -> dict[str, Any]:
        return {
            "atlas_id": self.atlas_id,
            "chart_ids": list(self.chart_ids)
            if type(self.chart_ids) is tuple
            else self.chart_ids,
            "tube_ids": list(self.tube_ids)
            if type(self.tube_ids) is tuple
            else self.tube_ids,
            "overlap_ids": list(self.overlap_ids)
            if type(self.overlap_ids) is tuple
            else self.overlap_ids,
            "source": self.source,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "PlanarLCExactGaugeAtlasCertificate":
        # JSON arrays become tuples, but their members are deliberately not
        # coerced.  Non-array values are preserved so the checker, rather than
        # the parser, rejects malformed truthy stand-ins.
        def positional_ids(value: object) -> object:
            if type(value) is list or type(value) is tuple:
                return tuple(value)
            return value

        return cls(
            atlas_id=data.get("atlas_id", ""),
            chart_ids=positional_ids(data.get("chart_ids", ())),  # type: ignore[arg-type]
            tube_ids=positional_ids(data.get("tube_ids", ())),  # type: ignore[arg-type]
            overlap_ids=positional_ids(data.get("overlap_ids", ())),  # type: ignore[arg-type]
            source=data.get(
                "source", "serialized_planar_lc_exact_gauge_atlas"
            ),
        )


@dataclass(frozen=True)
class PlanarLCExactCollisionAnchorCertificate:
    """Bind a zero-error LC tube to an exact isolated binary collision."""

    collision_id: str
    chart_id: str
    tube_id: str
    collision_parameter: float
    source: str = "serialized_planar_lc_exact_collision_anchor"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "PlanarLCExactCollisionAnchorCertificate":
        return cls(
            collision_id=str(data.get("collision_id", "")),
            chart_id=str(data.get("chart_id", "")),
            tube_id=str(data.get("tube_id", "")),
            collision_parameter=float(data.get("collision_parameter", np.inf)),
            source=str(
                data.get("source", "serialized_planar_lc_exact_collision_anchor")
            ),
        )


@dataclass(frozen=True)
class PlanarLCTwoSidedCollisionPassageCertificate:
    """Bind two punctured ordinary projections to one collision-anchored LC branch."""

    passage_id: str
    source_chart_id: str
    collision_id: str
    left_target_chart_id: str
    right_target_chart_id: str
    left_source_parameter: float
    right_source_parameter: float
    left_target_parameter: float
    right_target_parameter: float
    source: str = "serialized_planar_lc_two_sided_collision_passage"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "PlanarLCTwoSidedCollisionPassageCertificate":
        return cls(
            passage_id=str(data.get("passage_id", "")),
            source_chart_id=str(data.get("source_chart_id", "")),
            collision_id=str(data.get("collision_id", "")),
            left_target_chart_id=str(data.get("left_target_chart_id", "")),
            right_target_chart_id=str(data.get("right_target_chart_id", "")),
            left_source_parameter=float(
                data.get("left_source_parameter", np.inf)
            ),
            right_source_parameter=float(
                data.get("right_source_parameter", np.inf)
            ),
            left_target_parameter=float(
                data.get("left_target_parameter", np.inf)
            ),
            right_target_parameter=float(
                data.get("right_target_parameter", np.inf)
            ),
            source=str(
                data.get(
                    "source", "serialized_planar_lc_two_sided_collision_passage"
                )
            ),
        )


@dataclass(frozen=True)
class OrdinaryToPlanarLCEnclosureTransitionCertificate:
    """Bind an exact ordinary enclosure to a noncollision LC lift atlas."""

    transition_id: str
    source_chart_id: str
    target_chart_id: str
    source_parameter: float
    target_parameter: float
    handoff_time: float
    max_time_gap: float
    source: str = "serialized_ordinary_to_planar_lc_enclosure_transition"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "OrdinaryToPlanarLCEnclosureTransitionCertificate":
        return cls(
            transition_id=str(data.get("transition_id", "")),
            source_chart_id=str(data.get("source_chart_id", "")),
            target_chart_id=str(data.get("target_chart_id", "")),
            source_parameter=float(data.get("source_parameter", np.inf)),
            target_parameter=float(data.get("target_parameter", np.inf)),
            handoff_time=float(data.get("handoff_time", np.inf)),
            max_time_gap=float(data.get("max_time_gap", np.inf)),
            source=str(
                data.get(
                    "source",
                    "serialized_ordinary_to_planar_lc_enclosure_transition",
                )
            ),
        )


@dataclass(frozen=True)
class PlanarLCToOrdinaryEnclosureTransitionCertificate:
    """Project a punctured LC exit into an elapsed-time ordinary chart."""

    transition_id: str
    source_chart_id: str
    target_chart_id: str
    source_parameter: float
    target_parameter: float
    source: str = "serialized_planar_lc_to_ordinary_enclosure_transition"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "PlanarLCToOrdinaryEnclosureTransitionCertificate":
        return cls(
            transition_id=str(data.get("transition_id", "")),
            source_chart_id=str(data.get("source_chart_id", "")),
            target_chart_id=str(data.get("target_chart_id", "")),
            source_parameter=float(data.get("source_parameter", np.inf)),
            target_parameter=float(data.get("target_parameter", np.inf)),
            source=str(
                data.get(
                    "source",
                    "serialized_planar_lc_to_ordinary_enclosure_transition",
                )
            ),
        )


@dataclass(frozen=True)
class PlanarLeviCivitaBinaryChartCertificate:
    """Serialized planar Levi-Civita binary chart data for the checker."""

    certificate_id: str
    chart_id: str
    chart_type: str
    masses: tuple[float, ...]
    pair: tuple[int, int]
    z_coefficients: tuple[tuple[float, ...], ...]
    z_velocity_coefficients: tuple[tuple[float, ...], ...]
    pair_energy_coefficients: tuple[float, ...]
    binary_center_coefficients: tuple[tuple[float, ...], ...]
    binary_center_velocity_coefficients: tuple[tuple[float, ...], ...]
    third_offset_coefficients: tuple[tuple[float, ...], ...]
    third_offset_velocity_coefficients: tuple[tuple[float, ...], ...]
    physical_time_coefficients: tuple[float, ...]
    parameter_interval: tuple[float, float]
    physical_time_interval: tuple[float, float]
    coefficient_tolerance: float
    regularized_residual_tolerance: float
    projected_residual_tolerance: float
    tail_bound: float
    sample_count: int = 7
    projection_rho_lower_bound: float = 1.0e-12
    source: str = "serialized_planar_levi_civita_binary_chart"

    @property
    def order(self) -> int:
        return len(self.z_coefficients) - 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanarLeviCivitaBinaryChartCertificate":
        return cls(
            certificate_id=str(data.get("certificate_id", "")),
            chart_id=str(data.get("chart_id", "")),
            chart_type=str(data.get("chart_type", "")),
            masses=_tuple_of_float(data.get("masses", ())),
            pair=_pair_of_int(data.get("pair", ())),
            z_coefficients=_coefficient_vector_tuple(data.get("z_coefficients", ())),
            z_velocity_coefficients=_coefficient_vector_tuple(
                data.get("z_velocity_coefficients", ()),
            ),
            pair_energy_coefficients=_tuple_of_float(
                data.get("pair_energy_coefficients", ()),
            ),
            binary_center_coefficients=_coefficient_vector_tuple(
                data.get("binary_center_coefficients", ()),
            ),
            binary_center_velocity_coefficients=_coefficient_vector_tuple(
                data.get("binary_center_velocity_coefficients", ()),
            ),
            third_offset_coefficients=_coefficient_vector_tuple(
                data.get("third_offset_coefficients", ()),
            ),
            third_offset_velocity_coefficients=_coefficient_vector_tuple(
                data.get("third_offset_velocity_coefficients", ()),
            ),
            physical_time_coefficients=_tuple_of_float(
                data.get("physical_time_coefficients", ()),
            ),
            parameter_interval=_pair_of_float(data.get("parameter_interval", ())),
            physical_time_interval=_pair_of_float(
                data.get("physical_time_interval", ()),
            ),
            coefficient_tolerance=float(data.get("coefficient_tolerance", np.inf)),
            regularized_residual_tolerance=float(
                data.get("regularized_residual_tolerance", np.inf),
            ),
            projected_residual_tolerance=float(
                data.get("projected_residual_tolerance", np.inf),
            ),
            tail_bound=float(data.get("tail_bound", np.inf)),
            sample_count=int(data.get("sample_count", 7)),
            projection_rho_lower_bound=float(
                data.get("projection_rho_lower_bound", 1.0e-12),
            ),
            source=str(
                data.get("source", "serialized_planar_levi_civita_binary_chart"),
            ),
        )


@dataclass(frozen=True)
class PlanarLeviCivitaTransitionCertificate:
    """Serialized ordinary/LC binary chart handoff certificate."""

    transition_id: str
    source_chart_id: str
    target_chart_id: str
    transition_type: str
    handoff_time: float
    source_parameter: float
    target_parameter: float
    position_tolerance: float
    velocity_tolerance: float
    physical_time_tolerance: float
    source: str = "serialized_planar_levi_civita_transition"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanarLeviCivitaTransitionCertificate":
        return cls(
            transition_id=str(data.get("transition_id", "")),
            source_chart_id=str(data.get("source_chart_id", "")),
            target_chart_id=str(data.get("target_chart_id", "")),
            transition_type=str(data.get("transition_type", "")),
            handoff_time=float(data.get("handoff_time", np.inf)),
            source_parameter=float(data.get("source_parameter", np.inf)),
            target_parameter=float(data.get("target_parameter", np.inf)),
            position_tolerance=float(data.get("position_tolerance", np.inf)),
            velocity_tolerance=float(data.get("velocity_tolerance", np.inf)),
            physical_time_tolerance=float(data.get("physical_time_tolerance", np.inf)),
            source=str(data.get("source", "serialized_planar_levi_civita_transition")),
        )


@dataclass(frozen=True)
class SpatialKSBinaryChartCertificate:
    """Serialized spatial KS binary chart data for the checker."""

    certificate_id: str
    chart_id: str
    chart_type: str
    masses: tuple[float, ...]
    pair: tuple[int, int]
    u_coefficients: tuple[tuple[float, ...], ...]
    u_velocity_coefficients: tuple[tuple[float, ...], ...]
    pair_energy_coefficients: tuple[float, ...]
    binary_center_coefficients: tuple[tuple[float, ...], ...]
    binary_center_velocity_coefficients: tuple[tuple[float, ...], ...]
    third_offset_coefficients: tuple[tuple[float, ...], ...]
    third_offset_velocity_coefficients: tuple[tuple[float, ...], ...]
    physical_time_coefficients: tuple[float, ...]
    parameter_interval: tuple[float, float]
    physical_time_interval: tuple[float, float]
    coefficient_tolerance: float
    regularized_residual_tolerance: float
    projected_residual_tolerance: float
    constraint_tolerance: float
    tail_bound: float
    sample_count: int = 7
    projection_rho_lower_bound: float = 1.0e-12
    source: str = "serialized_spatial_ks_binary_chart"

    @property
    def order(self) -> int:
        return len(self.u_coefficients) - 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpatialKSBinaryChartCertificate":
        return cls(
            certificate_id=str(data.get("certificate_id", "")),
            chart_id=str(data.get("chart_id", "")),
            chart_type=str(data.get("chart_type", "")),
            masses=_tuple_of_float(data.get("masses", ())),
            pair=_pair_of_int(data.get("pair", ())),
            u_coefficients=_coefficient_vector_tuple(data.get("u_coefficients", ())),
            u_velocity_coefficients=_coefficient_vector_tuple(
                data.get("u_velocity_coefficients", ()),
            ),
            pair_energy_coefficients=_tuple_of_float(
                data.get("pair_energy_coefficients", ()),
            ),
            binary_center_coefficients=_coefficient_vector_tuple(
                data.get("binary_center_coefficients", ()),
            ),
            binary_center_velocity_coefficients=_coefficient_vector_tuple(
                data.get("binary_center_velocity_coefficients", ()),
            ),
            third_offset_coefficients=_coefficient_vector_tuple(
                data.get("third_offset_coefficients", ()),
            ),
            third_offset_velocity_coefficients=_coefficient_vector_tuple(
                data.get("third_offset_velocity_coefficients", ()),
            ),
            physical_time_coefficients=_tuple_of_float(
                data.get("physical_time_coefficients", ()),
            ),
            parameter_interval=_pair_of_float(data.get("parameter_interval", ())),
            physical_time_interval=_pair_of_float(
                data.get("physical_time_interval", ()),
            ),
            coefficient_tolerance=float(data.get("coefficient_tolerance", np.inf)),
            regularized_residual_tolerance=float(
                data.get("regularized_residual_tolerance", np.inf),
            ),
            projected_residual_tolerance=float(
                data.get("projected_residual_tolerance", np.inf),
            ),
            constraint_tolerance=float(data.get("constraint_tolerance", np.inf)),
            tail_bound=float(data.get("tail_bound", np.inf)),
            sample_count=int(data.get("sample_count", 7)),
            projection_rho_lower_bound=float(
                data.get("projection_rho_lower_bound", 1.0e-12),
            ),
            source=str(data.get("source", "serialized_spatial_ks_binary_chart")),
        )


@dataclass(frozen=True)
class SpatialKSTransitionCertificate:
    """Serialized ordinary/KS binary chart handoff certificate."""

    transition_id: str
    source_chart_id: str
    target_chart_id: str
    transition_type: str
    handoff_time: float
    source_parameter: float
    target_parameter: float
    position_tolerance: float
    velocity_tolerance: float
    physical_time_tolerance: float
    source: str = "serialized_spatial_ks_transition"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpatialKSTransitionCertificate":
        return cls(
            transition_id=str(data.get("transition_id", "")),
            source_chart_id=str(data.get("source_chart_id", "")),
            target_chart_id=str(data.get("target_chart_id", "")),
            transition_type=str(data.get("transition_type", "")),
            handoff_time=float(data.get("handoff_time", np.inf)),
            source_parameter=float(data.get("source_parameter", np.inf)),
            target_parameter=float(data.get("target_parameter", np.inf)),
            position_tolerance=float(data.get("position_tolerance", np.inf)),
            velocity_tolerance=float(data.get("velocity_tolerance", np.inf)),
            physical_time_tolerance=float(data.get("physical_time_tolerance", np.inf)),
            source=str(data.get("source", "serialized_spatial_ks_transition")),
        )


@dataclass(frozen=True)
class EventIsolationCertificate:
    """Serialized scalar polynomial event-isolation certificate."""

    certificate_id: str
    event_id: str
    event_type: str
    coefficient_source: str
    event_value: float
    pair: tuple[int, int]
    root: float
    search_interval: tuple[float, float]
    root_interval: tuple[float, float]
    coefficient_intervals: tuple[tuple[float, float], ...]
    source: str = "serialized_event_isolation"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventIsolationCertificate":
        return cls(
            certificate_id=str(data.get("certificate_id", "")),
            event_id=str(data.get("event_id", "")),
            event_type=str(data.get("event_type", "")),
            coefficient_source=str(data.get("coefficient_source", "")),
            event_value=float(data.get("event_value", np.inf)),
            pair=_pair_of_int(data.get("pair", ())),
            root=float(data.get("root", np.inf)),
            search_interval=_pair_of_float(data.get("search_interval", ())),
            root_interval=_pair_of_float(data.get("root_interval", ())),
            coefficient_intervals=_interval_tuple(
                data.get("coefficient_intervals", ()),
            ),
            source=str(data.get("source", "serialized_event_isolation")),
        )


@dataclass(frozen=True)
class BranchUnionCertificate:
    """Serialized finite branch-union certificate for the checker.

    Each leaf response id must refer to an independently checked chart,
    chart-chain, or nested branch-union response.  The aggregate target
    interval is a scalar physical-time interval enclosing every leaf target
    interval; richer set-valued target hulls can extend this record later
    without changing the basic grammar.
    """

    certificate_id: str
    union_id: str
    union_type: str
    leaf_response_certificate_ids: tuple[str, ...]
    leaf_target_intervals: tuple[tuple[float, float], ...]
    aggregate_target_interval: tuple[float, float]
    leaf_kinds: tuple[str, ...] = ()
    source: str = "serialized_branch_union"

    @property
    def leaf_count(self) -> int:
        return len(self.leaf_response_certificate_ids)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BranchUnionCertificate":
        return cls(
            certificate_id=str(data.get("certificate_id", "")),
            union_id=str(data.get("union_id", "")),
            union_type=str(data.get("union_type", "")),
            leaf_response_certificate_ids=tuple(
                str(value)
                for value in data.get("leaf_response_certificate_ids", ()) or ()
            ),
            leaf_target_intervals=_interval_tuple(
                data.get("leaf_target_intervals", ()),
            ),
            aggregate_target_interval=_pair_of_float(
                data.get("aggregate_target_interval", ()),
            ),
            leaf_kinds=tuple(str(value) for value in data.get("leaf_kinds", ()) or ()),
            source=str(data.get("source", "serialized_branch_union")),
        )


@dataclass(frozen=True)
class ChartChainCertificate:
    """Serialized finite chart-chain coverage certificate."""

    certificate_id: str
    chain_id: str
    chain_type: str
    chart_ids: tuple[str, ...]
    transition_ids: tuple[str, ...]
    target_physical_time_interval: tuple[float, float]
    source: str = "serialized_chart_chain"

    @property
    def chart_count(self) -> int:
        return len(self.chart_ids)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChartChainCertificate":
        return cls(
            certificate_id=str(data.get("certificate_id", "")),
            chain_id=str(data.get("chain_id", "")),
            chain_type=str(data.get("chain_type", "")),
            chart_ids=tuple(str(value) for value in data.get("chart_ids", ()) or ()),
            transition_ids=tuple(
                str(value) for value in data.get("transition_ids", ()) or ()
            ),
            target_physical_time_interval=_pair_of_float(
                data.get("target_physical_time_interval", ()),
            ),
            source=str(data.get("source", "serialized_chart_chain")),
        )


@dataclass(frozen=True)
class FuchsianLogTermCertificate:
    """Serialized finite Fuchsian-log row data for a total-collision chart."""

    power: float
    coefficients_by_log_power: tuple[
        tuple[int, tuple[tuple[float, ...], ...]],
        ...,
    ]
    selector_basis: tuple[tuple[tuple[float, ...], ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FuchsianLogTermCertificate":
        return cls(
            power=float(data.get("power", np.inf)),
            coefficients_by_log_power=_fuchsian_log_coefficients_by_log_power(
                data.get("coefficients_by_log_power", ()),
            ),
            selector_basis=_coefficient_matrix_tuple(
                data.get("selector_basis", ()),
            ),
        )


@dataclass(frozen=True)
class PrimitiveCauchyTailInputCertificate:
    """Serialized primitive Cauchy tail input for one chart component."""

    majorant_initial: float
    majorant_growth: float
    step_ratio_bound: float
    retained_order_initial: int
    retained_order_increment: int
    first_shell_tail_bound: float
    shell_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_input(
        cls,
        input_: PrimitiveCauchyTailInput,
    ) -> "PrimitiveCauchyTailInputCertificate":
        return cls(
            majorant_initial=float(input_.majorant_initial),
            majorant_growth=float(input_.majorant_growth),
            step_ratio_bound=float(input_.step_ratio_bound),
            retained_order_initial=int(input_.retained_order_initial),
            retained_order_increment=int(input_.retained_order_increment),
            first_shell_tail_bound=float(input_.first_shell_tail_bound),
            shell_ratio=float(input_.shell_ratio),
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "PrimitiveCauchyTailInputCertificate":
        return cls(
            majorant_initial=float(data.get("majorant_initial", np.inf)),
            majorant_growth=float(data.get("majorant_growth", np.inf)),
            step_ratio_bound=float(data.get("step_ratio_bound", np.inf)),
            retained_order_initial=int(data.get("retained_order_initial", -1)),
            retained_order_increment=int(data.get("retained_order_increment", -1)),
            first_shell_tail_bound=float(
                data.get("first_shell_tail_bound", np.inf),
            ),
            shell_ratio=float(data.get("shell_ratio", np.inf)),
        )


@dataclass(frozen=True)
class FiniteFuchsianLogPrimitiveCauchyInputsCertificate:
    """Serialized primitive Cauchy envelope for a Fuchsian-log stop chart."""

    initial_radius: float
    shell_contraction: float
    analytic_disk_fraction: float
    log_growth_factor: float
    component_derivative_orders: tuple[tuple[str, int], ...]
    component_multipliers: tuple[tuple[str, float], ...]
    component_inputs: tuple[tuple[str, PrimitiveCauchyTailInputCertificate], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_inputs(
        cls,
        inputs: FiniteFuchsianLogPrimitiveCauchyInputs,
    ) -> "FiniteFuchsianLogPrimitiveCauchyInputsCertificate":
        return cls(
            initial_radius=float(inputs.initial_radius),
            shell_contraction=float(inputs.shell_contraction),
            analytic_disk_fraction=float(inputs.analytic_disk_fraction),
            log_growth_factor=float(inputs.log_growth_factor),
            component_derivative_orders=tuple(
                (str(component), int(order))
                for component, order in sorted(inputs.component_derivative_orders.items())
            ),
            component_multipliers=tuple(
                (str(component), float(multiplier))
                for component, multiplier in sorted(inputs.component_multipliers.items())
            ),
            component_inputs=tuple(
                (
                    str(component),
                    PrimitiveCauchyTailInputCertificate.from_input(input_),
                )
                for component, input_ in sorted(inputs.component_inputs.items())
            ),
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "FiniteFuchsianLogPrimitiveCauchyInputsCertificate":
        return cls(
            initial_radius=float(data.get("initial_radius", np.inf)),
            shell_contraction=float(data.get("shell_contraction", np.inf)),
            analytic_disk_fraction=float(data.get("analytic_disk_fraction", np.inf)),
            log_growth_factor=float(data.get("log_growth_factor", np.inf)),
            component_derivative_orders=tuple(
                (str(component), int(order))
                for component, order in data.get("component_derivative_orders", ()) or ()
            ),
            component_multipliers=tuple(
                (str(component), float(multiplier))
                for component, multiplier in data.get("component_multipliers", ()) or ()
            ),
            component_inputs=tuple(
                (
                    str(component),
                    (
                        PrimitiveCauchyTailInputCertificate.from_dict(input_)
                        if isinstance(input_, dict)
                        else input_
                    ),
                )
                for component, input_ in data.get("component_inputs", ()) or ()
            ),
        )


@dataclass(frozen=True)
class GeneralizedFuchsianSelectedCoefficientCertificate:
    """One selected multivariate generalized Fuchsian coefficient."""

    index: tuple[int, ...]
    coefficient: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "GeneralizedFuchsianSelectedCoefficientCertificate":
        return cls(
            index=tuple(int(value) for value in data.get("index", ()) or ()),
            coefficient=_coefficient_matrix(data.get("coefficient", ())),
        )


@dataclass(frozen=True)
class GeneralizedFuchsianRemainderMajorantCertificate:
    """Serialized Banach majorant for a generalized Fuchsian remainder."""

    initial_radius: float
    shell_contraction: float
    analytic_disk_fraction: float
    defect_bound: float
    linear_inverse_bound: float
    nonlinear_lipschitz_bound: float
    remainder_ball_radius: float
    component_effective_exponents: tuple[tuple[str, float], ...]
    component_inputs: tuple[tuple[str, PrimitiveCauchyTailInputCertificate], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def contraction_factor(self) -> float:
        return float(self.linear_inverse_bound * self.nonlinear_lipschitz_bound)

    @property
    def self_map_bound(self) -> float:
        return float(
            self.linear_inverse_bound * self.defect_bound
            + self.contraction_factor * self.remainder_ball_radius
        )

    @property
    def contraction_slack(self) -> float:
        return float(1.0 - self.contraction_factor)

    @property
    def self_map_margin(self) -> float:
        return float(self.remainder_ball_radius - self.self_map_bound)

    @property
    def picard_first_step_bound(self) -> float:
        return float(self.linear_inverse_bound * self.defect_bound)

    def picard_tail_bound(self, iteration_count: int) -> float:
        iteration_count = int(iteration_count)
        if iteration_count < 0:
            raise ValueError("iteration_count must be nonnegative")
        contraction_factor = self.contraction_factor
        if not (
            np.isfinite(contraction_factor)
            and 0.0 <= contraction_factor < 1.0
            and np.isfinite(self.picard_first_step_bound)
        ):
            return float("inf")
        return float(
            contraction_factor**iteration_count
            * self.picard_first_step_bound
            / (1.0 - contraction_factor)
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "GeneralizedFuchsianRemainderMajorantCertificate":
        return cls(
            initial_radius=float(data.get("initial_radius", np.inf)),
            shell_contraction=float(data.get("shell_contraction", np.inf)),
            analytic_disk_fraction=float(data.get("analytic_disk_fraction", np.inf)),
            defect_bound=float(data.get("defect_bound", np.inf)),
            linear_inverse_bound=float(data.get("linear_inverse_bound", np.inf)),
            nonlinear_lipschitz_bound=float(
                data.get("nonlinear_lipschitz_bound", np.inf),
            ),
            remainder_ball_radius=float(data.get("remainder_ball_radius", np.inf)),
            component_effective_exponents=tuple(
                (str(component), float(exponent))
                for component, exponent in data.get("component_effective_exponents", ()) or ()
            ),
            component_inputs=tuple(
                (
                    str(component),
                    (
                        PrimitiveCauchyTailInputCertificate.from_dict(input_)
                        if isinstance(input_, dict)
                        else input_
                    ),
                )
                for component, input_ in data.get("component_inputs", ()) or ()
            ),
        )


@dataclass(frozen=True)
class TotalCollisionFuchsianStopChartCertificate:
    """Serialized finite Fuchsian-log total-collision stop chart."""

    certificate_id: str
    chart_id: str
    chart_type: str
    masses: tuple[float, ...]
    central_shape: tuple[tuple[float, ...], ...]
    scale_coefficient: float
    terms: tuple[FuchsianLogTermCertificate, ...]
    tau_interval: tuple[float, float]
    physical_time_interval: tuple[float, float]
    event_physical_time: float
    total_collision_tau: float
    isolation_radius: float
    central_shape_pair_distance_floor: float
    shape_deviation_bound: float
    shape_pair_distance_floor: float
    residual_tolerance: float
    angular_momentum_tolerance: float
    tail_bound: float
    sample_count: int = 7
    primitive_cauchy_inputs: FiniteFuchsianLogPrimitiveCauchyInputsCertificate | None = None
    source: str = "serialized_total_collision_fuchsian_stop_chart"

    @property
    def body_count(self) -> int:
        return len(self.central_shape)

    @property
    def dimension(self) -> int:
        if not self.central_shape:
            return 0
        return len(self.central_shape[0])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "TotalCollisionFuchsianStopChartCertificate":
        return cls(
            certificate_id=str(data.get("certificate_id", "")),
            chart_id=str(data.get("chart_id", "")),
            chart_type=str(data.get("chart_type", "")),
            masses=_tuple_of_float(data.get("masses", ())),
            central_shape=_coefficient_matrix(data.get("central_shape", ())),
            scale_coefficient=float(data.get("scale_coefficient", np.inf)),
            terms=tuple(
                FuchsianLogTermCertificate.from_dict(term)
                if isinstance(term, dict)
                else FuchsianLogTermCertificate(
                    power=float(getattr(term, "power", np.inf)),
                    coefficients_by_log_power=_fuchsian_log_coefficients_by_log_power(
                        getattr(term, "coefficients_by_log_power", ()),
                    ),
                    selector_basis=_coefficient_matrix_tuple(
                        getattr(term, "selector_basis", ()),
                    ),
                )
                for term in data.get("terms", ()) or ()
            ),
            tau_interval=_pair_of_float(data.get("tau_interval", ())),
            physical_time_interval=_pair_of_float(
                data.get("physical_time_interval", ()),
            ),
            event_physical_time=float(data.get("event_physical_time", np.inf)),
            total_collision_tau=float(data.get("total_collision_tau", np.inf)),
            isolation_radius=float(data.get("isolation_radius", np.inf)),
            central_shape_pair_distance_floor=float(
                data.get("central_shape_pair_distance_floor", np.inf),
            ),
            shape_deviation_bound=float(data.get("shape_deviation_bound", np.inf)),
            shape_pair_distance_floor=float(
                data.get("shape_pair_distance_floor", np.inf),
            ),
            residual_tolerance=float(data.get("residual_tolerance", np.inf)),
            angular_momentum_tolerance=float(
                data.get("angular_momentum_tolerance", np.inf),
            ),
            tail_bound=float(data.get("tail_bound", np.inf)),
            sample_count=int(data.get("sample_count", 7)),
            primitive_cauchy_inputs=(
                None
                if data.get("primitive_cauchy_inputs") is None
                else FiniteFuchsianLogPrimitiveCauchyInputsCertificate.from_dict(
                    data.get("primitive_cauchy_inputs", {}),
                )
            ),
            source=str(
                data.get(
                    "source",
                    "serialized_total_collision_fuchsian_stop_chart",
                ),
            ),
        )


@dataclass(frozen=True)
class TotalCollisionGeneralizedFuchsianStopChartCertificate:
    """Serialized generalized Fuchsian total-collision stop chart."""

    certificate_id: str
    chart_id: str
    chart_type: str
    masses: tuple[float, ...]
    central_shape: tuple[tuple[float, ...], ...]
    powers: tuple[float, ...]
    selected_coefficients: tuple[GeneralizedFuchsianSelectedCoefficientCertificate, ...]
    max_total_degree: int
    scale_index: tuple[int, ...] | None
    tau_interval: tuple[float, float]
    physical_time_interval: tuple[float, float]
    event_physical_time: float
    total_collision_tau: float
    isolation_radius: float
    central_shape_pair_distance_floor: float
    shape_deviation_bound: float
    shape_pair_distance_floor: float
    residual_tolerance: float
    angular_momentum_tolerance: float
    tail_bound: float
    sample_count: int = 7
    remainder_majorant: GeneralizedFuchsianRemainderMajorantCertificate | None = None
    source: str = "serialized_total_collision_generalized_fuchsian_stop_chart"
    projected_residual_tolerance: float | None = None

    @property
    def body_count(self) -> int:
        return len(self.central_shape)

    @property
    def dimension(self) -> int:
        if not self.central_shape:
            return 0
        return len(self.central_shape[0])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "TotalCollisionGeneralizedFuchsianStopChartCertificate":
        scale_index_data = data.get("scale_index")
        return cls(
            certificate_id=str(data.get("certificate_id", "")),
            chart_id=str(data.get("chart_id", "")),
            chart_type=str(data.get("chart_type", "")),
            masses=_tuple_of_float(data.get("masses", ())),
            central_shape=_coefficient_matrix(data.get("central_shape", ())),
            powers=_tuple_of_float(data.get("powers", ())),
            selected_coefficients=tuple(
                GeneralizedFuchsianSelectedCoefficientCertificate.from_dict(item)
                if isinstance(item, dict)
                else item
                for item in data.get("selected_coefficients", ()) or ()
            ),
            max_total_degree=int(data.get("max_total_degree", -1)),
            scale_index=(
                None
                if scale_index_data is None
                else tuple(int(value) for value in scale_index_data or ())
            ),
            tau_interval=_pair_of_float(data.get("tau_interval", ())),
            physical_time_interval=_pair_of_float(
                data.get("physical_time_interval", ()),
            ),
            event_physical_time=float(data.get("event_physical_time", np.inf)),
            total_collision_tau=float(data.get("total_collision_tau", np.inf)),
            isolation_radius=float(data.get("isolation_radius", np.inf)),
            central_shape_pair_distance_floor=float(
                data.get("central_shape_pair_distance_floor", np.inf),
            ),
            shape_deviation_bound=float(data.get("shape_deviation_bound", np.inf)),
            shape_pair_distance_floor=float(
                data.get("shape_pair_distance_floor", np.inf),
            ),
            residual_tolerance=float(data.get("residual_tolerance", np.inf)),
            angular_momentum_tolerance=float(
                data.get("angular_momentum_tolerance", np.inf),
            ),
            tail_bound=float(data.get("tail_bound", np.inf)),
            sample_count=int(data.get("sample_count", 7)),
            remainder_majorant=(
                None
                if data.get("remainder_majorant") is None
                else GeneralizedFuchsianRemainderMajorantCertificate.from_dict(
                    data.get("remainder_majorant", {}),
                )
            ),
            source=str(
                data.get(
                    "source",
                    "serialized_total_collision_generalized_fuchsian_stop_chart",
                ),
            ),
            projected_residual_tolerance=(
                float(data["projected_residual_tolerance"])
                if data.get("projected_residual_tolerance") is not None
                else None
            ),
        )


def ordinary_taylor_chart_certificate_from_solution(
    solution: TaylorSolution,
    *,
    certificate_id: str,
    chart_id: str,
    parameter_interval: tuple[float, float],
    physical_time_interval: tuple[float, float] | None = None,
    coefficient_tolerance: float = 1.0e-11,
    residual_tolerance: float = 1.0e-8,
    tail_bound: float = 0.0,
    sample_count: int = 7,
    source: str = "taylor_solution_serialization",
) -> OrdinaryTaylorChartCertificate:
    """Serialize a point Taylor solution into the certificate language."""

    physical = (
        tuple(float(value) for value in parameter_interval)
        if physical_time_interval is None
        else tuple(float(value) for value in physical_time_interval)
    )
    return OrdinaryTaylorChartCertificate(
        certificate_id=str(certificate_id),
        chart_id=str(chart_id),
        chart_type="ordinary_taylor",
        masses=tuple(float(value) for value in np.asarray(solution.masses, dtype=float)),
        position_coefficients=_array_to_nested_tuple(solution.position),
        velocity_coefficients=_array_to_nested_tuple(solution.velocity),
        parameter_interval=tuple(float(value) for value in parameter_interval),
        physical_time_interval=physical,
        coefficient_tolerance=float(coefficient_tolerance),
        residual_tolerance=float(residual_tolerance),
        tail_bound=float(tail_bound),
        sample_count=int(sample_count),
        source=str(source),
    )


def planar_levi_civita_binary_chart_certificate_from_solution(
    solution: RegularizedBinaryTaylorSolution,
    *,
    certificate_id: str,
    chart_id: str,
    parameter_interval: tuple[float, float],
    physical_time_interval: tuple[float, float] | None = None,
    coefficient_tolerance: float = 1.0e-11,
    regularized_residual_tolerance: float = 1.0e-8,
    projected_residual_tolerance: float = 1.0e-8,
    tail_bound: float = 0.0,
    sample_count: int = 7,
    projection_rho_lower_bound: float = 1.0e-12,
    source: str = "regularized_binary_solution_serialization",
) -> PlanarLeviCivitaBinaryChartCertificate:
    """Serialize a planar LC binary Taylor chart into the certificate language."""

    physical = (
        _interval_physical_time_range(solution, parameter_interval)
        if physical_time_interval is None
        else tuple(float(value) for value in physical_time_interval)
    )
    return PlanarLeviCivitaBinaryChartCertificate(
        certificate_id=str(certificate_id),
        chart_id=str(chart_id),
        chart_type="planar_levi_civita_binary",
        masses=tuple(float(value) for value in np.asarray(solution.masses, dtype=float)),
        pair=tuple(int(value) for value in solution.pair),
        z_coefficients=_array_to_vector_tuple(solution.z),
        z_velocity_coefficients=_array_to_vector_tuple(solution.z_velocity),
        pair_energy_coefficients=_array_to_scalar_tuple(solution.pair_energy),
        binary_center_coefficients=_array_to_vector_tuple(solution.binary_center),
        binary_center_velocity_coefficients=_array_to_vector_tuple(
            solution.binary_center_velocity,
        ),
        third_offset_coefficients=_array_to_vector_tuple(solution.third_offset),
        third_offset_velocity_coefficients=_array_to_vector_tuple(
            solution.third_offset_velocity,
        ),
        physical_time_coefficients=_array_to_scalar_tuple(solution.physical_time),
        parameter_interval=tuple(float(value) for value in parameter_interval),
        physical_time_interval=physical,
        coefficient_tolerance=float(coefficient_tolerance),
        regularized_residual_tolerance=float(regularized_residual_tolerance),
        projected_residual_tolerance=float(projected_residual_tolerance),
        tail_bound=float(tail_bound),
        sample_count=int(sample_count),
        projection_rho_lower_bound=float(projection_rho_lower_bound),
        source=str(source),
    )


def spatial_ks_binary_chart_certificate_from_solution(
    solution: SpatialKSBinaryTaylorSolution,
    *,
    certificate_id: str,
    chart_id: str,
    parameter_interval: tuple[float, float],
    physical_time_interval: tuple[float, float] | None = None,
    coefficient_tolerance: float = 1.0e-11,
    regularized_residual_tolerance: float = 1.0e-8,
    projected_residual_tolerance: float = 1.0e-8,
    constraint_tolerance: float = 1.0e-8,
    tail_bound: float = 0.0,
    sample_count: int = 7,
    projection_rho_lower_bound: float = 1.0e-12,
    physical_time_shift: float = 0.0,
    source: str = "spatial_ks_binary_solution_serialization",
) -> SpatialKSBinaryChartCertificate:
    """Serialize a spatial KS binary Taylor chart into the certificate language."""

    physical = (
        _interval_physical_time_range(
            solution,
            parameter_interval,
            physical_time_shift=physical_time_shift,
        )
        if physical_time_interval is None
        else tuple(float(value) for value in physical_time_interval)
    )
    return SpatialKSBinaryChartCertificate(
        certificate_id=str(certificate_id),
        chart_id=str(chart_id),
        chart_type="spatial_ks_binary",
        masses=tuple(float(value) for value in np.asarray(solution.masses, dtype=float)),
        pair=tuple(int(value) for value in solution.pair),
        u_coefficients=_array_to_vector_tuple(solution.u),
        u_velocity_coefficients=_array_to_vector_tuple(solution.u_velocity),
        pair_energy_coefficients=_array_to_scalar_tuple(solution.pair_energy),
        binary_center_coefficients=_array_to_vector_tuple(solution.binary_center),
        binary_center_velocity_coefficients=_array_to_vector_tuple(
            solution.binary_center_velocity,
        ),
        third_offset_coefficients=_array_to_vector_tuple(solution.third_offset),
        third_offset_velocity_coefficients=_array_to_vector_tuple(
            solution.third_offset_velocity,
        ),
        physical_time_coefficients=_shifted_scalar_coefficients(
            solution.physical_time,
            physical_time_shift,
        ),
        parameter_interval=tuple(float(value) for value in parameter_interval),
        physical_time_interval=physical,
        coefficient_tolerance=float(coefficient_tolerance),
        regularized_residual_tolerance=float(regularized_residual_tolerance),
        projected_residual_tolerance=float(projected_residual_tolerance),
        constraint_tolerance=float(constraint_tolerance),
        tail_bound=float(tail_bound),
        sample_count=int(sample_count),
        projection_rho_lower_bound=float(projection_rho_lower_bound),
        source=str(source),
    )


def ordinary_chart_transition_certificate(
    *,
    transition_id: str,
    source_chart_id: str,
    target_chart_id: str,
    handoff_time: float,
    position_tolerance: float = 1.0e-8,
    velocity_tolerance: float = 1.0e-8,
    transition_type: str = "ordinary_overlap_handoff",
    source: str = "ordinary_transition_serialization",
) -> OrdinaryChartTransitionCertificate:
    """Create a serialized ordinary-to-ordinary handoff certificate."""

    return OrdinaryChartTransitionCertificate(
        transition_id=str(transition_id),
        source_chart_id=str(source_chart_id),
        target_chart_id=str(target_chart_id),
        transition_type=str(transition_type),
        handoff_time=float(handoff_time),
        position_tolerance=float(position_tolerance),
        velocity_tolerance=float(velocity_tolerance),
        source=str(source),
    )


def planar_levi_civita_transition_certificate(
    *,
    transition_id: str,
    source_chart_id: str,
    target_chart_id: str,
    handoff_time: float,
    source_parameter: float,
    target_parameter: float,
    position_tolerance: float = 1.0e-8,
    velocity_tolerance: float = 1.0e-8,
    physical_time_tolerance: float = 1.0e-10,
    transition_type: str = "ordinary_to_binary_event_handoff",
    source: str = "planar_lc_transition_serialization",
) -> PlanarLeviCivitaTransitionCertificate:
    """Create a serialized ordinary/LC binary handoff certificate."""

    return PlanarLeviCivitaTransitionCertificate(
        transition_id=str(transition_id),
        source_chart_id=str(source_chart_id),
        target_chart_id=str(target_chart_id),
        transition_type=str(transition_type),
        handoff_time=float(handoff_time),
        source_parameter=float(source_parameter),
        target_parameter=float(target_parameter),
        position_tolerance=float(position_tolerance),
        velocity_tolerance=float(velocity_tolerance),
        physical_time_tolerance=float(physical_time_tolerance),
        source=str(source),
    )


def spatial_ks_transition_certificate(
    *,
    transition_id: str,
    source_chart_id: str,
    target_chart_id: str,
    handoff_time: float,
    source_parameter: float,
    target_parameter: float,
    position_tolerance: float = 1.0e-8,
    velocity_tolerance: float = 1.0e-8,
    physical_time_tolerance: float = 1.0e-10,
    transition_type: str = "spatial_ordinary_to_ks_decreasing_distance_entry",
    source: str = "spatial_ks_transition_serialization",
) -> SpatialKSTransitionCertificate:
    """Create a serialized ordinary/KS binary handoff certificate."""

    return SpatialKSTransitionCertificate(
        transition_id=str(transition_id),
        source_chart_id=str(source_chart_id),
        target_chart_id=str(target_chart_id),
        transition_type=str(transition_type),
        handoff_time=float(handoff_time),
        source_parameter=float(source_parameter),
        target_parameter=float(target_parameter),
        position_tolerance=float(position_tolerance),
        velocity_tolerance=float(velocity_tolerance),
        physical_time_tolerance=float(physical_time_tolerance),
        source=str(source),
    )


def spatial_ks_rho_exit_event_isolation_certificate(
    event: SpatialKSRhoExitEventCertificate,
    *,
    certificate_id: str,
    event_id: str,
    source: str = "spatial_ks_rho_exit_event_serialization",
) -> EventIsolationCertificate:
    """Serialize a certified KS ``rho=exit_rho`` event-isolation certificate."""

    if event.root is None or event.root_interval is None:
        raise ValueError("rho-exit event must have an isolated root")
    return EventIsolationCertificate(
        certificate_id=str(certificate_id),
        event_id=str(event_id),
        event_type="spatial_ks_rho_exit",
        coefficient_source=str(event.coefficient_source),
        event_value=float(event.exit_rho),
        pair=(-1, -1),
        root=float(event.root),
        search_interval=tuple(float(value) for value in event.search_interval),
        root_interval=tuple(float(value) for value in event.root_interval),
        coefficient_intervals=_interval_tuple(event.coefficient_intervals),
        source=str(source),
    )


def spatial_ks_entry_event_isolation_certificate(
    event: SpatialKSEntryEventCertificate,
    *,
    certificate_id: str,
    event_id: str,
    source: str = "spatial_ks_entry_event_serialization",
) -> EventIsolationCertificate:
    """Serialize a certified ordinary/KS or competing-KS entry event."""

    if event.root is None or event.root_interval is None:
        raise ValueError("KS entry event must have an isolated root")
    event_type = (
        "spatial_ks_competing_binary_entry"
        if event.coefficient_source == "spatial_ks_binary_interval_taylor"
        else "spatial_ordinary_ks_entry"
    )
    return EventIsolationCertificate(
        certificate_id=str(certificate_id),
        event_id=str(event_id),
        event_type=event_type,
        coefficient_source=str(event.coefficient_source),
        event_value=float(event.enter_distance),
        pair=tuple(int(value) for value in event.pair),
        root=float(event.root),
        search_interval=tuple(float(value) for value in event.search_interval),
        root_interval=tuple(float(value) for value in event.root_interval),
        coefficient_intervals=_interval_tuple(event.coefficient_intervals),
        source=str(source),
    )


def planar_hybrid_chart_chain_certificates_from_solution(
    hybrid_solution: object,
    *,
    certificate_id_prefix: str = "planar-hybrid",
    chain_id: str = "planar-hybrid-chain",
    coefficient_tolerance: float = 1.0e-6,
    ordinary_residual_tolerance: float = 1.0e-6,
    regularized_residual_tolerance: float = 1.0e-8,
    projected_residual_tolerance: float = 2.0e-1,
    physical_time_tolerance: float = 1.0e-10,
    position_tolerance: float = 1.0e-8,
    velocity_tolerance: float = 1.0e-8,
    sample_count: int = 7,
    projection_rho_lower_bound: float = 1.0e-12,
    source: str = "planar_hybrid_solution_serialization",
) -> tuple[
    tuple[OrdinaryTaylorChartCertificate | PlanarLeviCivitaBinaryChartCertificate, ...],
    tuple[OrdinaryChartTransitionCertificate | PlanarLeviCivitaTransitionCertificate, ...],
    ChartChainCertificate,
]:
    """Serialize a finite planar hybrid solution as a checked chart chain.

    The hybrid constructor carries interval proof ledgers.  This serializer
    builds a smaller independent-checker bundle from the point representative:
    ordinary steps become ordinary Taylor charts, binary steps become planar
    Levi-Civita charts, and supported ordinary/LC handoffs become transition
    certificates.  Same-family ordinary handoffs are supported; same-family LC
    continuations are deliberately rejected until the checker has a binary-to-
    binary transition grammar.
    """

    masses = np.asarray(getattr(hybrid_solution, "masses", ()), dtype=float).reshape(-1)
    states = np.asarray(getattr(hybrid_solution, "states", ()), dtype=float)
    steps = tuple(getattr(hybrid_solution, "steps", ()) or ())
    prefix = str(certificate_id_prefix)
    if masses.shape != (3,):
        raise ValueError("planar hybrid serialization requires three masses")
    if states.ndim != 2 or states.shape[0] < len(steps) or states.shape[1] != 12:
        raise ValueError("planar hybrid serialization requires packed planar states")
    if not steps:
        raise ValueError("planar hybrid serialization requires at least one step")

    charts: list[OrdinaryTaylorChartCertificate | PlanarLeviCivitaBinaryChartCertificate] = []
    for index, step in enumerate(steps):
        chart_kind = str(getattr(step, "chart", ""))
        start_state = states[index]
        start_time = float(getattr(step, "start_time", np.nan))
        end_time = float(getattr(step, "end_time", start_time + float(getattr(step, "physical_step", np.nan))))
        physical_interval = _ordered_pair(start_time, end_time)
        tail_bound = _hybrid_step_tail_bound(step)
        order = _hybrid_step_order(step)
        if chart_kind == "ordinary":
            positions, velocities = _planar_state_positions_velocities(start_state)
            solution = construct_taylor_solution(
                positions,
                velocities,
                masses,
                order=order,
            )
            local_step = float(getattr(step, "physical_step", np.nan))
            charts.append(
                ordinary_taylor_chart_certificate_from_solution(
                    solution,
                    certificate_id=f"{prefix}-ordinary-chart-{index}",
                    chart_id=f"{prefix}-ordinary-{index}",
                    parameter_interval=_zero_based_interval(local_step),
                    physical_time_interval=physical_interval,
                    coefficient_tolerance=float(coefficient_tolerance),
                    residual_tolerance=float(ordinary_residual_tolerance),
                    tail_bound=tail_bound,
                    sample_count=int(sample_count),
                    source=source,
                )
            )
        elif chart_kind == "binary":
            pair = tuple(int(value) for value in getattr(step, "pair", ()) or ())
            if len(pair) != 2:
                raise ValueError("binary hybrid step is missing a selected pair")
            positions, velocities = _planar_state_positions_velocities(start_state)
            initial_state = _regularized_point_state_for_planar_hybrid_step(
                step,
                positions,
                velocities,
                masses,
                pair=pair,
            )
            solution = construct_regularized_binary_taylor_solution(
                initial_state,
                order=order,
            )
            local_parameter_step = float(getattr(step, "parameter_step", np.nan))
            parameter_interval = _zero_based_interval(local_parameter_step)
            checked_physical_interval = _padded_interval(
                _union_interval(
                    physical_interval,
                    _interval_physical_time_range(solution, parameter_interval),
                )
            )
            charts.append(
                planar_levi_civita_binary_chart_certificate_from_solution(
                    solution,
                    certificate_id=f"{prefix}-lc-chart-{index}",
                    chart_id=f"{prefix}-lc-{index}",
                    parameter_interval=parameter_interval,
                    physical_time_interval=checked_physical_interval,
                    coefficient_tolerance=float(coefficient_tolerance),
                    regularized_residual_tolerance=float(
                        regularized_residual_tolerance
                    ),
                    projected_residual_tolerance=float(projected_residual_tolerance),
                    tail_bound=tail_bound,
                    sample_count=int(sample_count),
                    projection_rho_lower_bound=float(projection_rho_lower_bound),
                    source=source,
                )
            )
        else:
            raise ValueError(f"unsupported planar hybrid chart kind: {chart_kind!r}")

    transitions: list[
        OrdinaryChartTransitionCertificate | PlanarLeviCivitaTransitionCertificate
    ] = []
    for index, (left_step, right_step, left_chart, right_chart) in enumerate(
        zip(steps, steps[1:], charts, charts[1:]),
    ):
        handoff_time = float(getattr(left_step, "end_time"))
        left_kind = str(getattr(left_step, "chart", ""))
        right_kind = str(getattr(right_step, "chart", ""))
        if left_kind == right_kind == "ordinary":
            transitions.append(
                ordinary_chart_transition_certificate(
                    transition_id=f"{prefix}-ordinary-transition-{index}",
                    source_chart_id=left_chart.chart_id,
                    target_chart_id=right_chart.chart_id,
                    handoff_time=handoff_time,
                    position_tolerance=float(position_tolerance),
                    velocity_tolerance=float(velocity_tolerance),
                    source=source,
                )
            )
        elif left_kind == "ordinary" and right_kind == "binary":
            transitions.append(
                planar_levi_civita_transition_certificate(
                    transition_id=f"{prefix}-ordinary-to-lc-{index}",
                    source_chart_id=left_chart.chart_id,
                    target_chart_id=right_chart.chart_id,
                    transition_type="ordinary_to_binary_event_handoff",
                    handoff_time=handoff_time,
                    source_parameter=float(getattr(left_step, "physical_step")),
                    target_parameter=0.0,
                    position_tolerance=float(position_tolerance),
                    velocity_tolerance=float(velocity_tolerance),
                    physical_time_tolerance=float(physical_time_tolerance),
                    source=source,
                )
            )
        elif left_kind == "binary" and right_kind == "ordinary":
            transitions.append(
                planar_levi_civita_transition_certificate(
                    transition_id=f"{prefix}-lc-to-ordinary-{index}",
                    source_chart_id=left_chart.chart_id,
                    target_chart_id=right_chart.chart_id,
                    transition_type="binary_to_ordinary_event_handoff",
                    handoff_time=handoff_time,
                    source_parameter=float(getattr(left_step, "parameter_step")),
                    target_parameter=0.0,
                    position_tolerance=float(position_tolerance),
                    velocity_tolerance=float(velocity_tolerance),
                    physical_time_tolerance=float(physical_time_tolerance),
                    source=source,
                )
            )
        else:
            raise ValueError(
                "planar hybrid independent checker does not yet support "
                f"{left_kind!r}->{right_kind!r} transitions"
            )

    time_values = [
        float(value)
        for chart in charts
        for value in tuple(chart.physical_time_interval)
    ]
    chain = ChartChainCertificate(
        certificate_id=f"{prefix}-chart-chain",
        chain_id=str(chain_id),
        chain_type="regularized_atlas_chart_chain",
        chart_ids=tuple(chart.chart_id for chart in charts),
        transition_ids=tuple(transition.transition_id for transition in transitions),
        target_physical_time_interval=(min(time_values), max(time_values)),
        source=source,
    )
    return tuple(charts), tuple(transitions), chain


def total_collision_fuchsian_stop_chart_certificate_from_branch(
    branch: FiniteFuchsianLogBranch,
    isolation: FiniteFuchsianLogTotalCollisionIsolationCertificate,
    *,
    certificate_id: str,
    chart_id: str,
    tau_interval: tuple[float, float] | None = None,
    event_physical_time: float = 0.0,
    residual_tolerance: float = 1.0e-8,
    angular_momentum_tolerance: float = 1.0e-8,
    tail_bound: float = 0.0,
    sample_count: int = 7,
    cauchy_inputs: FiniteFuchsianLogPrimitiveCauchyInputs | None = None,
    source: str = "finite_fuchsian_log_total_collision_branch_serialization",
) -> TotalCollisionFuchsianStopChartCertificate:
    """Serialize a finite Fuchsian-log punctured total-collision chart."""

    if not isolation.certified:
        raise ValueError("total-collision isolation certificate must certify")
    radius = float(isolation.radius)
    tau = (-radius, radius) if tau_interval is None else tau_interval
    event_time = float(event_physical_time)
    physical = (
        event_time + float(tau[0]) ** 3,
        event_time + float(tau[1]) ** 3,
    )
    return TotalCollisionFuchsianStopChartCertificate(
        certificate_id=str(certificate_id),
        chart_id=str(chart_id),
        chart_type="total_collision_fuchsian_stop",
        masses=tuple(float(value) for value in np.asarray(branch.masses, dtype=float)),
        central_shape=_matrix_to_tuple(np.asarray(branch.central_shape, dtype=float)),
        scale_coefficient=float(branch.scale_coefficient),
        terms=tuple(
            FuchsianLogTermCertificate(
                power=float(term.power),
                coefficients_by_log_power=tuple(
                    (
                        int(log_power),
                        _matrix_to_tuple(np.asarray(coefficient, dtype=float)),
                    )
                    for log_power, coefficient in sorted(
                        term.coefficients_by_log_power.items(),
                    )
                ),
                selector_basis=tuple(
                    _matrix_to_tuple(np.asarray(basis, dtype=float))
                    for basis in term.selector_basis
                ),
            )
            for term in branch.terms
        ),
        tau_interval=tuple(float(value) for value in tau),
        physical_time_interval=tuple(float(value) for value in physical),
        event_physical_time=event_time,
        total_collision_tau=0.0,
        isolation_radius=radius,
        central_shape_pair_distance_floor=float(
            isolation.central_shape_pair_distance_floor,
        ),
        shape_deviation_bound=float(isolation.shape_deviation_bound),
        shape_pair_distance_floor=float(isolation.shape_pair_distance_floor),
        residual_tolerance=float(residual_tolerance),
        angular_momentum_tolerance=float(angular_momentum_tolerance),
        tail_bound=float(tail_bound),
        sample_count=int(sample_count),
        primitive_cauchy_inputs=(
            None
            if cauchy_inputs is None
            else FiniteFuchsianLogPrimitiveCauchyInputsCertificate.from_inputs(
                cauchy_inputs,
            )
        ),
        source=str(source),
    )


def total_collision_generalized_fuchsian_stop_chart_certificate_from_branch(
    branch: FuchsianShapeBranch,
    *,
    certificate_id: str,
    chart_id: str,
    isolation_radius: float,
    central_shape_pair_distance_floor: float,
    shape_deviation_bound: float,
    shape_pair_distance_floor: float,
    tau_interval: tuple[float, float] | None = None,
    event_physical_time: float = 0.0,
    residual_tolerance: float = 1.0e-8,
    angular_momentum_tolerance: float = 1.0e-8,
    tail_bound: float = 0.0,
    sample_count: int = 7,
    remainder_majorant: GeneralizedFuchsianRemainderMajorantCertificate | None = None,
    source: str = "generalized_fuchsian_total_collision_branch_serialization",
    projected_residual_tolerance: float | None = None,
) -> TotalCollisionGeneralizedFuchsianStopChartCertificate:
    """Serialize a generalized Fuchsian punctured total-collision chart."""

    radius = float(isolation_radius)
    tau = (-radius, radius) if tau_interval is None else tau_interval
    event_time = float(event_physical_time)
    physical = (
        event_time + float(tau[0]) ** 3,
        event_time + float(tau[1]) ** 3,
    )
    return TotalCollisionGeneralizedFuchsianStopChartCertificate(
        certificate_id=str(certificate_id),
        chart_id=str(chart_id),
        chart_type="total_collision_generalized_fuchsian_stop",
        masses=tuple(float(value) for value in np.asarray(branch.masses, dtype=float)),
        central_shape=_matrix_to_tuple(np.asarray(branch.central_shape, dtype=float)),
        powers=tuple(float(power) for power in branch.powers),
        selected_coefficients=tuple(
            GeneralizedFuchsianSelectedCoefficientCertificate(
                index=tuple(int(value) for value in index),
                coefficient=_matrix_to_tuple(np.asarray(branch.coefficients[index], dtype=float)),
            )
            for index in sorted(branch.selected_indices)
        ),
        max_total_degree=int(branch.max_total_degree),
        scale_index=(
            None
            if branch.scale_index is None
            else tuple(int(value) for value in branch.scale_index)
        ),
        tau_interval=tuple(float(value) for value in tau),
        physical_time_interval=tuple(float(value) for value in physical),
        event_physical_time=event_time,
        total_collision_tau=0.0,
        isolation_radius=radius,
        central_shape_pair_distance_floor=float(central_shape_pair_distance_floor),
        shape_deviation_bound=float(shape_deviation_bound),
        shape_pair_distance_floor=float(shape_pair_distance_floor),
        residual_tolerance=float(residual_tolerance),
        angular_momentum_tolerance=float(angular_momentum_tolerance),
        tail_bound=float(tail_bound),
        sample_count=int(sample_count),
        remainder_majorant=remainder_majorant,
        source=str(source),
        projected_residual_tolerance=(
            None
            if projected_residual_tolerance is None
            else float(projected_residual_tolerance)
        ),
    )


def _ordered_pair(left: float, right: float) -> tuple[float, float]:
    left = float(left)
    right = float(right)
    return (min(left, right), max(left, right))


def _padded_interval(interval: tuple[float, float]) -> tuple[float, float]:
    lower, upper = (float(interval[0]), float(interval[1]))
    radius = 64.0 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))
    return (lower - radius, upper + radius)


def _union_interval(
    left: tuple[float, float],
    right: tuple[float, float],
) -> tuple[float, float]:
    return (
        min(float(left[0]), float(right[0])),
        max(float(left[1]), float(right[1])),
    )


def _zero_based_interval(width: float) -> tuple[float, float]:
    width = float(width)
    return _ordered_pair(0.0, width)


def _planar_state_positions_velocities(state: object) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(state, dtype=float).reshape(-1)
    if flat.shape != (12,):
        raise ValueError("packed planar state must contain 12 components")
    return flat[:6].reshape(3, 2), flat[6:].reshape(3, 2)


def _hybrid_step_order(step: object) -> int:
    certificate = getattr(step, "truncation_certificate", None)
    computed_order = int(getattr(certificate, "computed_order", 0) or 0)
    if computed_order > 0:
        return computed_order
    retained_order = int(getattr(certificate, "retained_order", 0) or 0)
    return retained_order if retained_order > 0 else 1


def _hybrid_step_tail_bound(step: object) -> float:
    certificate = getattr(step, "truncation_certificate", None)
    tail = float(getattr(certificate, "tail_bound", 0.0) or 0.0)
    if not np.isfinite(tail) or tail < 0.0:
        raise ValueError("hybrid step has no finite nonnegative tail bound")
    return tail


def _regularized_point_state_for_planar_hybrid_step(
    step: object,
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    *,
    pair: tuple[int, int],
) -> RegularizedBinaryCollisionChartState:
    try:
        return planar_to_regularized_binary_collision_chart(
            positions,
            velocities,
            masses,
            pair=pair,
        )
    except ValueError:
        interval_state = getattr(step, "start_regularized_interval_state", None)
        if interval_state is None:
            interval_union = tuple(
                getattr(step, "start_regularized_interval_state_union", ()) or ()
            )
            interval_state = interval_union[0] if interval_union else None
        if interval_state is None:
            raise
        if not isinstance(interval_state, IntervalRegularizedBinaryCollisionChartState):
            raise ValueError("binary hybrid step carries an unsupported interval LC lift")
        return _regularized_point_state_from_interval(interval_state)


def _regularized_point_state_from_interval(
    interval_state: IntervalRegularizedBinaryCollisionChartState,
) -> RegularizedBinaryCollisionChartState:
    return RegularizedBinaryCollisionChartState(
        masses=np.asarray(interval_state.masses, dtype=float),
        pair=tuple(int(value) for value in interval_state.pair),
        z=_interval_midpoint_array(interval_state.z),
        z_velocity=_interval_midpoint_array(interval_state.z_velocity),
        pair_energy=_interval_midpoint(interval_state.pair_energy),
        binary_center=_interval_midpoint_array(interval_state.binary_center),
        binary_center_velocity=_interval_midpoint_array(
            interval_state.binary_center_velocity,
        ),
        third_offset=_interval_midpoint_array(interval_state.third_offset),
        third_offset_velocity=_interval_midpoint_array(
            interval_state.third_offset_velocity,
        ),
    )


def _interval_midpoint_array(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    return np.asarray(
        [_interval_midpoint(value) for value in array.reshape(-1)],
        dtype=float,
    ).reshape(array.shape)


def _interval_midpoint(value: object) -> float:
    if hasattr(value, "lower") and hasattr(value, "upper"):
        return 0.5 * (float(value.lower) + float(value.upper))
    lower, upper = value
    return 0.5 * (float(lower) + float(upper))


def _array_to_nested_tuple(values: np.ndarray) -> tuple[tuple[tuple[float, ...], ...], ...]:
    array = _coefficient_float_array(values)
    if array.ndim != 3:
        raise ValueError("ordinary Taylor coefficients must have shape (degree, body, dimension)")
    return tuple(
        tuple(
            tuple(float(array[degree, body, axis]) for axis in range(array.shape[2]))
            for body in range(array.shape[1])
        )
        for degree in range(array.shape[0])
    )


def _tuple_of_float(values: object) -> tuple[float, ...]:
    return tuple(float(value) for value in values or ())


def _coefficient_float_array(values: object) -> np.ndarray:
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return _interval_midpoint_array(values)


def _array_to_scalar_tuple(values: object) -> tuple[float, ...]:
    array = _coefficient_float_array(values)
    if array.ndim != 1:
        raise ValueError("scalar coefficients must have shape (degree,)")
    return tuple(float(array[degree]) for degree in range(array.shape[0]))


def _shifted_scalar_coefficients(
    values: object,
    shift: float,
) -> tuple[float, ...]:
    coefficients = list(_array_to_scalar_tuple(values))
    if coefficients:
        coefficients[0] = float(coefficients[0] + float(shift))
    return tuple(coefficients)


def _pair_of_int(values: object) -> tuple[int, int]:
    pair = tuple(int(value) for value in values or ())
    if len(pair) != 2:
        return (-1, -1)
    return pair


def _pair_of_float(values: object) -> tuple[float, float]:
    pair = tuple(float(value) for value in values or ())
    if len(pair) != 2:
        return (np.inf, -np.inf)
    return pair


def _coefficient_tuple(
    values: object,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(
        tuple(tuple(float(component) for component in body) for body in degree)
        for degree in values or ()
    )


def _array_to_vector_tuple(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
    array = _coefficient_float_array(values)
    if array.ndim != 2:
        raise ValueError("vector coefficients must have shape (degree, dimension)")
    return tuple(
        tuple(float(array[degree, axis]) for axis in range(array.shape[1]))
        for degree in range(array.shape[0])
    )


def _coefficient_vector_tuple(values: object) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(component) for component in degree) for degree in values or ())


def _coefficient_matrix(values: object) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(component) for component in row) for row in values or ())


def _coefficient_matrix_tuple(
    values: object,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(_coefficient_matrix(matrix) for matrix in values or ())


def _matrix_to_tuple(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("matrix data must have shape (body, dimension)")
    return tuple(
        tuple(float(array[row, axis]) for axis in range(array.shape[1]))
        for row in range(array.shape[0])
    )


def _fuchsian_log_coefficients_by_log_power(
    values: object,
) -> tuple[tuple[int, tuple[tuple[float, ...], ...]], ...]:
    out = []
    for item in values or ():
        log_power, coefficient = item
        out.append((int(log_power), _coefficient_matrix(coefficient)))
    return tuple(out)


def _interval_tuple(values: object) -> tuple[tuple[float, float], ...]:
    out = []
    for value in values or ():
        interval = tuple(float(component) for component in value)
        if len(interval) != 2:
            return ()
        out.append((interval[0], interval[1]))
    return tuple(out)


def _interval_physical_time_range(
    solution: RegularizedBinaryTaylorSolution | SpatialKSBinaryTaylorSolution,
    parameter_interval: tuple[float, float],
    *,
    physical_time_shift: float = 0.0,
) -> tuple[float, float]:
    enclosure = interval_polynomial_eval(
        solution.physical_time,
        FloatInterval(float(parameter_interval[0]), float(parameter_interval[1])),
    )
    shift = float(physical_time_shift)
    return (float(enclosure.lower + shift), float(enclosure.upper + shift))

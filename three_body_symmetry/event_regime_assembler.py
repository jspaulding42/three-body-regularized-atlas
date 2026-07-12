"""Assemble local event-chart certificates into a global recurrence budget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .event_recurrence import (
    DEFAULT_BUDGET_COMPONENTS,
    GeometricShellEventBudgetCertificate,
    GeometricShellEventIsolationCertificate,
    PrimitiveCauchyTailInput,
    construct_primitive_cauchy_tail_input,
    derive_all_future_event_budget_from_geometric_shell_isolation,
    derive_geometric_shell_event_isolation,
)
from .fuchsian import (
    FiniteFuchsianLogBranch,
    derive_finite_fuchsian_log_branch_primitive_cauchy_inputs,
)


@dataclass(frozen=True)
class EventRegimeObligation:
    """One constructor-derived obligation in event-regime assembly."""

    obligation: str
    certified: bool
    source: str
    detail: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "certified", self.certified is True)
        object.__setattr__(
            self,
            "required",
            self.required if type(self.required) is bool else True,
        )


@dataclass(frozen=True)
class LocalChartFamilyCauchyCertificate:
    """Primitive Cauchy inputs for one event-chart family."""

    kind: str
    component_inputs: Mapping[str, PrimitiveCauchyTailInput]
    source_certificate: object | None = None
    source: str = "local_chart_family_cauchy_constructor"

    @property
    def source_certified(self) -> bool:
        if self.source_certificate is None:
            return False
        return _certificate_field(
            self.source_certificate,
            ("certified", "proof_certified"),
        )

    def certified_for_components(self, components: tuple[str, ...]) -> bool:
        return bool(
            self.kind
            and self.source_certified
            and all(component in self.component_inputs for component in components)
            and all(self.component_inputs[component].certified for component in components)
        )

    def component_input(self, component: str) -> PrimitiveCauchyTailInput:
        return self.component_inputs[str(component)]

    @property
    def certified(self) -> bool:
        return self.certified_for_components(tuple(self.component_inputs))


@dataclass(frozen=True)
class UniformNoncollisionOrdinaryGapCauchyInputs:
    """Ordinary-gap primitive inputs derived from a uniform noncollision envelope."""

    masses: tuple[float, ...]
    pair_distance_lower_bound: float
    pair_diameter_upper_bound: float
    speed_upper_bound: float
    sundman_distance_power: float
    position_radius: float
    sundman_factor_bound: float
    acceleration_bound: float
    velocity_radius_bound: float
    sundman_radius_floor: float
    component_majorants: Mapping[str, float]
    component_inputs: Mapping[str, PrimitiveCauchyTailInput]

    @property
    def certified(self) -> bool:
        return bool(
            self.masses
            and all(np.isfinite(mass) and mass > 0.0 for mass in self.masses)
            and np.isfinite(self.pair_distance_lower_bound)
            and self.pair_distance_lower_bound > 0.0
            and np.isfinite(self.pair_diameter_upper_bound)
            and self.pair_diameter_upper_bound >= self.pair_distance_lower_bound
            and np.isfinite(self.speed_upper_bound)
            and self.speed_upper_bound >= 0.0
            and np.isfinite(self.sundman_distance_power)
            and self.sundman_distance_power > 0.0
            and np.isfinite(self.position_radius)
            and self.position_radius > 0.0
            and np.isfinite(self.sundman_factor_bound)
            and self.sundman_factor_bound > 0.0
            and np.isfinite(self.acceleration_bound)
            and self.acceleration_bound > 0.0
            and np.isfinite(self.velocity_radius_bound)
            and self.velocity_radius_bound > 0.0
            and np.isfinite(self.sundman_radius_floor)
            and self.sundman_radius_floor > 0.0
            and self.component_inputs
            and all(input_.certified for input_ in self.component_inputs.values())
        )

    def component_input(self, component: str) -> PrimitiveCauchyTailInput:
        return self.component_inputs[str(component)]

    def component_majorant(self, component: str) -> float:
        return float(self.component_majorants[str(component)])


@dataclass(frozen=True)
class UniformSeparatedBinaryLeviCivitaCauchyInputs:
    """Separated-binary primitive inputs derived from a uniform LC envelope."""

    masses: tuple[float, float, float]
    pair: tuple[int, int]
    third: int
    alpha: float
    beta: float
    z_bound: float
    z_velocity_bound: float
    pair_energy_bound: float
    binary_center_bound: float
    binary_center_velocity_bound: float
    third_offset_bound: float
    third_offset_velocity_bound: float
    third_body_nominal_distance_lower_bound: float
    radii: Mapping[str, float]
    third_distance_floor: float
    perturbation_bound: float
    center_acceleration_bound: float
    third_offset_acceleration_bound: float
    rho_bound: float
    rhs_bounds: Mapping[str, float]
    levi_civita_radius_floor: float
    state_sup_bound: float
    component_majorants: Mapping[str, float]
    component_inputs: Mapping[str, PrimitiveCauchyTailInput]

    @property
    def certified(self) -> bool:
        return bool(
            len(self.masses) == 3
            and all(np.isfinite(mass) and mass > 0.0 for mass in self.masses)
            and len(set((*self.pair, self.third))) == 3
            and all(index in (0, 1, 2) for index in (*self.pair, self.third))
            and np.isfinite(self.alpha)
            and 0.0 < self.alpha < 1.0
            and np.isfinite(self.beta)
            and 0.0 < self.beta < 1.0
            and np.isfinite(self.z_bound)
            and self.z_bound >= 0.0
            and np.isfinite(self.z_velocity_bound)
            and self.z_velocity_bound >= 0.0
            and np.isfinite(self.pair_energy_bound)
            and self.pair_energy_bound >= 0.0
            and np.isfinite(self.binary_center_bound)
            and self.binary_center_bound >= 0.0
            and np.isfinite(self.binary_center_velocity_bound)
            and self.binary_center_velocity_bound >= 0.0
            and np.isfinite(self.third_offset_bound)
            and self.third_offset_bound >= 0.0
            and np.isfinite(self.third_offset_velocity_bound)
            and self.third_offset_velocity_bound >= 0.0
            and np.isfinite(self.third_body_nominal_distance_lower_bound)
            and self.third_body_nominal_distance_lower_bound > 0.0
            and self.radii
            and all(np.isfinite(radius) and radius > 0.0 for radius in self.radii.values())
            and np.isfinite(self.third_distance_floor)
            and self.third_distance_floor > 0.0
            and np.isfinite(self.perturbation_bound)
            and self.perturbation_bound >= 0.0
            and np.isfinite(self.center_acceleration_bound)
            and self.center_acceleration_bound >= 0.0
            and np.isfinite(self.third_offset_acceleration_bound)
            and self.third_offset_acceleration_bound >= 0.0
            and np.isfinite(self.rho_bound)
            and self.rho_bound >= 0.0
            and self.rhs_bounds
            and all(np.isfinite(bound) and bound >= 0.0 for bound in self.rhs_bounds.values())
            and np.isfinite(self.levi_civita_radius_floor)
            and self.levi_civita_radius_floor > 0.0
            and np.isfinite(self.state_sup_bound)
            and self.state_sup_bound >= 0.0
            and self.component_inputs
            and all(input_.certified for input_ in self.component_inputs.values())
        )

    def component_input(self, component: str) -> PrimitiveCauchyTailInput:
        return self.component_inputs[str(component)]

    def component_majorant(self, component: str) -> float:
        return float(self.component_majorants[str(component)])


@dataclass(frozen=True)
class EventRegimeAssemblyCertificate:
    """All-future event recurrence assembled from local chart certificates."""

    shell_isolation: GeometricShellEventIsolationCertificate
    chart_family_certificates: tuple[LocalChartFamilyCauchyCertificate, ...]
    event_budget: GeometricShellEventBudgetCertificate | None
    components: tuple[str, ...]
    obligations: tuple[EventRegimeObligation, ...]
    time_direction: str = "unspecified"
    provenance: str = "event_regime_assembly_constructor"

    @property
    def family_kinds(self) -> tuple[str, ...]:
        return tuple(certificate.kind for certificate in self.chart_family_certificates)

    @property
    def recurrence_closes(self) -> bool:
        return bool(self.event_budget is not None and self.event_budget.recurrence_closes)

    @property
    def certified(self) -> bool:
        return bool(
            self.obligations
            and any(
                isinstance(obligation, EventRegimeObligation)
                and obligation.required
                for obligation in self.obligations
            )
            and all(
                isinstance(obligation, EventRegimeObligation)
                for obligation in self.obligations
            )
            and all(
                obligation.certified is True
                for obligation in self.obligations
                if obligation.required
            )
            and self.event_budget is not None
            and self.event_budget.certified is True
        )

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.obligations:
            missing.append("event_regime_assembly_obligations_present")
        if self.obligations and not any(
            isinstance(obligation, EventRegimeObligation)
            and obligation.required
            for obligation in self.obligations
        ):
            missing.append("event_regime_assembly_required_obligation_present")
        for obligation in self.obligations:
            if not isinstance(obligation, EventRegimeObligation):
                missing.append("event_regime_assembly_obligation_type")
                continue
            if obligation.required and obligation.certified is not True:
                missing.append(obligation.obligation)
        return tuple(dict.fromkeys(missing))

    @property
    def chart_family_counts(self) -> Mapping[str, int]:
        return self.shell_isolation.chart_family_counts

    def family_certificate(self, kind: str) -> LocalChartFamilyCauchyCertificate:
        for certificate in self.chart_family_certificates:
            if certificate.kind == kind:
                return certificate
        raise KeyError(kind)


def certify_local_chart_family_primitive_cauchy_inputs(
    *,
    kind: str,
    primitive_inputs: Mapping[str, PrimitiveCauchyTailInput | tuple[float, float, float, int, int]]
    | object,
    source_certificate: object | None = None,
    components: tuple[str, ...] = DEFAULT_BUDGET_COMPONENTS,
    source: str = "local_chart_family_cauchy_constructor",
) -> LocalChartFamilyCauchyCertificate:
    """Normalize constructor-derived primitive inputs for one chart family."""

    _reject_raw_bool("source_certificate", source_certificate)
    kind = str(kind)
    if not kind:
        raise ValueError("chart family kind cannot be empty")
    if source_certificate is None:
        raise ValueError(
            "local chart-family Cauchy inputs require a constructor source certificate"
        )
    if kind == "ordinary_gap_taylor" and source_certificate is None:
        raise ValueError(
            "ordinary-gap chart-family Cauchy inputs require a constructor source certificate"
        )
    if _is_separated_binary_family_kind(kind) and source_certificate is None:
        raise ValueError(
            "separated-binary chart-family Cauchy inputs require a constructor source certificate"
        )
    if "total_collision" in kind and source_certificate is None:
        raise ValueError(
            "total-collision chart-family Cauchy inputs require a constructor source certificate"
        )
    component_inputs = {
        component: _coerce_component_input(primitive_inputs, component)
        for component in components
    }
    certificate = LocalChartFamilyCauchyCertificate(
        kind=kind,
        component_inputs=component_inputs,
        source_certificate=source_certificate,
        source=str(source),
    )
    if not certificate.certified_for_components(tuple(components)):
        raise ValueError(f"local chart-family Cauchy inputs do not certify for {kind}")
    return certificate


def certify_uniform_noncollision_ordinary_gap_chart_family(
    *,
    masses: object,
    pair_distance_lower_bound: float,
    pair_diameter_upper_bound: float,
    speed_upper_bound: float,
    step_ratio_bounds: Mapping[str, float],
    retained_order_initials: Mapping[str, int],
    retained_order_increments: Mapping[str, int],
    kind: str = "ordinary_gap_taylor",
    sundman_distance_power: float = 1.0,
    component_majorant_multipliers: Mapping[str, float] | None = None,
    components: tuple[str, ...] = DEFAULT_BUDGET_COMPONENTS,
) -> LocalChartFamilyCauchyCertificate:
    """Derive ordinary-gap Cauchy inputs from a uniform noncollision state envelope."""

    if str(kind) != "ordinary_gap_taylor":
        raise ValueError(
            "uniform noncollision ordinary constructor only supports ordinary_gap_taylor"
        )
    masses_array = np.asarray(masses, dtype=float)
    if masses_array.ndim != 1 or masses_array.size < 2:
        raise ValueError("masses must be a one-dimensional positive array")
    if np.any(masses_array <= 0.0) or not np.all(np.isfinite(masses_array)):
        raise ValueError("masses must be positive and finite")
    pair_distance_lower_bound = float(pair_distance_lower_bound)
    pair_diameter_upper_bound = float(pair_diameter_upper_bound)
    speed_upper_bound = float(speed_upper_bound)
    sundman_distance_power = float(sundman_distance_power)
    if not (
        np.isfinite(pair_distance_lower_bound)
        and pair_distance_lower_bound > 0.0
        and np.isfinite(pair_diameter_upper_bound)
        and pair_diameter_upper_bound >= pair_distance_lower_bound
        and np.isfinite(speed_upper_bound)
        and speed_upper_bound >= 0.0
        and np.isfinite(sundman_distance_power)
        and sundman_distance_power > 0.0
    ):
        raise ValueError("uniform noncollision envelope parameters do not certify")
    position_radius = pair_distance_lower_bound / 10.0
    tube_diameter = pair_diameter_upper_bound + pair_distance_lower_bound / 5.0
    sundman_factor_bound = tube_diameter ** (3.0 * sundman_distance_power)
    acceleration_bound = (
        float(np.sum(masses_array))
        * tube_diameter
        / (((14.0 / 25.0) ** 1.5) * pair_distance_lower_bound**3)
    )
    velocity_radius_bound = float(np.sqrt(acceleration_bound * position_radius))
    sundman_radius_floor = min(
        position_radius
        / (sundman_factor_bound * (speed_upper_bound + velocity_radius_bound)),
        np.sqrt(position_radius / acceleration_bound) / sundman_factor_bound,
    )
    base_majorants = {
        "value": pair_diameter_upper_bound + position_radius,
        "first_jet": sundman_factor_bound * (speed_upper_bound + velocity_radius_bound),
        "lifted_residual": sundman_factor_bound * acceleration_bound,
        "physical_residual": 0.5 * sundman_factor_bound * acceleration_bound,
    }
    if component_majorant_multipliers is not None:
        for component, multiplier in component_majorant_multipliers.items():
            component = str(component)
            multiplier = float(multiplier)
            if component not in base_majorants:
                raise ValueError(f"unknown ordinary component {component!r}")
            if not np.isfinite(multiplier) or multiplier < 0.0:
                raise ValueError(
                    "component majorant multipliers must be finite and nonnegative"
                )
            base_majorants[component] *= multiplier
    component_inputs: dict[str, PrimitiveCauchyTailInput] = {}
    components = tuple(str(component) for component in components)
    for component in components:
        if component not in base_majorants:
            raise ValueError(f"missing ordinary majorant for component {component}")
        if component not in step_ratio_bounds:
            raise ValueError(f"missing step ratio bound for {component}")
        if component not in retained_order_initials:
            raise ValueError(f"missing retained-order initial for {component}")
        if component not in retained_order_increments:
            raise ValueError(f"missing retained-order increment for {component}")
        component_inputs[component] = construct_primitive_cauchy_tail_input(
            majorant_initial=float(base_majorants[component]),
            majorant_growth=1.0,
            step_ratio_bound=float(step_ratio_bounds[component]),
            retained_order_initial=int(retained_order_initials[component]),
            retained_order_increment=int(retained_order_increments[component]),
        )
    ordinary_inputs = UniformNoncollisionOrdinaryGapCauchyInputs(
        masses=tuple(float(mass) for mass in masses_array),
        pair_distance_lower_bound=pair_distance_lower_bound,
        pair_diameter_upper_bound=pair_diameter_upper_bound,
        speed_upper_bound=speed_upper_bound,
        sundman_distance_power=sundman_distance_power,
        position_radius=float(position_radius),
        sundman_factor_bound=float(sundman_factor_bound),
        acceleration_bound=float(acceleration_bound),
        velocity_radius_bound=float(velocity_radius_bound),
        sundman_radius_floor=float(sundman_radius_floor),
        component_majorants=base_majorants,
        component_inputs=component_inputs,
    )
    if not ordinary_inputs.certified:
        raise ValueError("uniform noncollision ordinary Cauchy inputs did not certify")
    return certify_local_chart_family_primitive_cauchy_inputs(
        kind=kind,
        primitive_inputs=ordinary_inputs,
        source_certificate=ordinary_inputs,
        components=components,
        source="uniform_noncollision_ordinary_gap_chart_family_constructor",
    )


def certify_uniform_separated_binary_levi_civita_chart_family(
    *,
    masses: object,
    pair: tuple[int, int] = (0, 1),
    z_bound: float,
    z_velocity_bound: float,
    pair_energy_bound: float,
    binary_center_bound: float,
    binary_center_velocity_bound: float,
    third_offset_bound: float,
    third_offset_velocity_bound: float,
    third_body_nominal_distance_lower_bound: float,
    radii: Mapping[str, float],
    step_ratio_bounds: Mapping[str, float],
    retained_order_initials: Mapping[str, int],
    retained_order_increments: Mapping[str, int],
    kind: str = "separated_binary_levi_civita",
    component_majorant_multipliers: Mapping[str, float] | None = None,
    components: tuple[str, ...] = DEFAULT_BUDGET_COMPONENTS,
) -> LocalChartFamilyCauchyCertificate:
    """Derive LC separated-binary Cauchy inputs from a lifted state envelope."""

    if not _is_separated_binary_family_kind(str(kind)):
        raise ValueError(
            "uniform separated-binary constructor only supports separated_binary_levi_civita families"
        )
    masses_array = np.asarray(masses, dtype=float)
    if masses_array.shape != (3,):
        raise ValueError("separated-binary Levi-Civita envelopes require exactly three masses")
    if np.any(masses_array <= 0.0) or not np.all(np.isfinite(masses_array)):
        raise ValueError("masses must be positive and finite")
    first, second = (int(pair[0]), int(pair[1]))
    if first == second or first not in (0, 1, 2) or second not in (0, 1, 2):
        raise ValueError("pair must contain two distinct body indices")
    third_candidates = tuple(
        index for index in (0, 1, 2) if index not in (first, second)
    )
    if len(third_candidates) != 1:
        raise ValueError("could not identify separated third body")
    third = third_candidates[0]
    pair_mass = float(masses_array[first] + masses_array[second])
    alpha = float(masses_array[second] / pair_mass)
    beta = float(masses_array[first] / pair_mass)
    bounds = {
        "z_bound": float(z_bound),
        "z_velocity_bound": float(z_velocity_bound),
        "pair_energy_bound": float(pair_energy_bound),
        "binary_center_bound": float(binary_center_bound),
        "binary_center_velocity_bound": float(binary_center_velocity_bound),
        "third_offset_bound": float(third_offset_bound),
        "third_offset_velocity_bound": float(third_offset_velocity_bound),
        "third_body_nominal_distance_lower_bound": float(
            third_body_nominal_distance_lower_bound
        ),
    }
    if not all(np.isfinite(value) for value in bounds.values()):
        raise ValueError("separated-binary envelope bounds must be finite")
    if any(
        value < 0.0
        for key, value in bounds.items()
        if key != "third_body_nominal_distance_lower_bound"
    ):
        raise ValueError("separated-binary envelope bounds must be nonnegative")
    if bounds["third_body_nominal_distance_lower_bound"] <= 0.0:
        raise ValueError("third-body nominal distance floor must be positive")
    required_radii = (
        "z",
        "z_velocity",
        "pair_energy",
        "binary_center",
        "binary_center_velocity",
        "third_offset",
        "third_offset_velocity",
    )
    radii_values = {name: float(radii[name]) for name in required_radii}
    if not all(
        np.isfinite(radius) and radius > 0.0 for radius in radii_values.values()
    ):
        raise ValueError("separated-binary majorant radii must be positive and finite")

    relative_variation_bound = (
        2.0 * bounds["z_bound"] * radii_values["z"] + radii_values["z"] ** 2
    )
    third_distance_floor = min(
        bounds["third_body_nominal_distance_lower_bound"]
        - radii_values["third_offset"]
        - alpha * relative_variation_bound,
        bounds["third_body_nominal_distance_lower_bound"]
        - radii_values["third_offset"]
        - beta * relative_variation_bound,
    )
    if not (np.isfinite(third_distance_floor) and third_distance_floor > 0.0):
        raise ValueError("radii do not preserve separated-third-body distance")

    z_majorant = bounds["z_bound"] + radii_values["z"]
    z_velocity_majorant = bounds["z_velocity_bound"] + radii_values["z_velocity"]
    pair_energy_majorant = bounds["pair_energy_bound"] + radii_values["pair_energy"]
    center_bound = bounds["binary_center_bound"] + radii_values["binary_center"]
    center_velocity_majorant = (
        bounds["binary_center_velocity_bound"] + radii_values["binary_center_velocity"]
    )
    third_offset_bound_in_disk = (
        bounds["third_offset_bound"] + radii_values["third_offset"]
    )
    third_offset_velocity_majorant = (
        bounds["third_offset_velocity_bound"] + radii_values["third_offset_velocity"]
    )
    third_offset_upper = (
        third_offset_bound_in_disk + max(alpha, beta) * z_majorant**2
    )
    perturbation_bound = float(
        np.sum(masses_array) * third_offset_upper / third_distance_floor**3
    )
    center_acceleration_bound = float(masses_array[third] * perturbation_bound)
    third_offset_acceleration_bound = float(np.sum(masses_array) * perturbation_bound)
    rho_bound = float(z_majorant**2)
    rhs_bounds = {
        "z": float(z_velocity_majorant),
        "z_velocity": float(
            0.5 * pair_energy_majorant * z_majorant
            + 0.5 * rho_bound * z_majorant * perturbation_bound
        ),
        "pair_energy": float(
            2.0 * z_majorant * z_velocity_majorant * perturbation_bound
        ),
        "binary_center": float(rho_bound * center_velocity_majorant),
        "binary_center_velocity": float(rho_bound * center_acceleration_bound),
        "third_offset": float(rho_bound * third_offset_velocity_majorant),
        "third_offset_velocity": float(rho_bound * third_offset_acceleration_bound),
    }
    radius_candidates = [
        radii_values[name] / rhs_bounds[name]
        for name in required_radii
        if rhs_bounds[name] > 0.0
    ]
    if not radius_candidates:
        raise ValueError("separated-binary RHS has no positive radius candidates")
    levi_civita_radius_floor = float(min(radius_candidates))
    if not (np.isfinite(levi_civita_radius_floor) and levi_civita_radius_floor > 0.0):
        raise ValueError("Levi-Civita radius floor did not certify")
    max_rhs_bound = float(max(rhs_bounds.values()))
    state_sup_bound = float(
        max(
            z_majorant,
            z_velocity_majorant,
            pair_energy_majorant,
            center_bound,
            center_velocity_majorant,
            third_offset_bound_in_disk,
            third_offset_velocity_majorant,
            levi_civita_radius_floor * rho_bound,
        )
    )
    component_majorants = {
        "value": state_sup_bound,
        "first_jet": max_rhs_bound,
        "lifted_residual": max_rhs_bound,
        "physical_residual": rho_bound * max_rhs_bound,
    }
    if component_majorant_multipliers is not None:
        for component, multiplier in component_majorant_multipliers.items():
            component = str(component)
            multiplier = float(multiplier)
            if component not in component_majorants:
                raise ValueError(f"unknown separated-binary component {component!r}")
            if not np.isfinite(multiplier) or multiplier < 0.0:
                raise ValueError(
                    "component majorant multipliers must be finite and nonnegative"
                )
            component_majorants[component] *= multiplier

    component_inputs: dict[str, PrimitiveCauchyTailInput] = {}
    components = tuple(str(component) for component in components)
    for component in components:
        if component not in component_majorants:
            raise ValueError(
                f"missing separated-binary majorant for component {component}"
            )
        if component not in step_ratio_bounds:
            raise ValueError(f"missing step ratio bound for {component}")
        if component not in retained_order_initials:
            raise ValueError(f"missing retained-order initial for {component}")
        if component not in retained_order_increments:
            raise ValueError(f"missing retained-order increment for {component}")
        component_inputs[component] = construct_primitive_cauchy_tail_input(
            majorant_initial=float(component_majorants[component]),
            majorant_growth=1.0,
            step_ratio_bound=float(step_ratio_bounds[component]),
            retained_order_initial=int(retained_order_initials[component]),
            retained_order_increment=int(retained_order_increments[component]),
        )
    separated_binary_inputs = UniformSeparatedBinaryLeviCivitaCauchyInputs(
        masses=tuple(float(mass) for mass in masses_array),
        pair=(first, second),
        third=third,
        alpha=alpha,
        beta=beta,
        z_bound=bounds["z_bound"],
        z_velocity_bound=bounds["z_velocity_bound"],
        pair_energy_bound=bounds["pair_energy_bound"],
        binary_center_bound=bounds["binary_center_bound"],
        binary_center_velocity_bound=bounds["binary_center_velocity_bound"],
        third_offset_bound=bounds["third_offset_bound"],
        third_offset_velocity_bound=bounds["third_offset_velocity_bound"],
        third_body_nominal_distance_lower_bound=bounds[
            "third_body_nominal_distance_lower_bound"
        ],
        radii=radii_values,
        third_distance_floor=float(third_distance_floor),
        perturbation_bound=perturbation_bound,
        center_acceleration_bound=center_acceleration_bound,
        third_offset_acceleration_bound=third_offset_acceleration_bound,
        rho_bound=rho_bound,
        rhs_bounds=rhs_bounds,
        levi_civita_radius_floor=levi_civita_radius_floor,
        state_sup_bound=state_sup_bound,
        component_majorants=component_majorants,
        component_inputs=component_inputs,
    )
    if not separated_binary_inputs.certified:
        raise ValueError("uniform separated-binary Cauchy inputs did not certify")
    return certify_local_chart_family_primitive_cauchy_inputs(
        kind=kind,
        primitive_inputs=separated_binary_inputs,
        source_certificate=separated_binary_inputs,
        components=components,
        source="uniform_separated_binary_levi_civita_chart_family_constructor",
    )


def certify_fuchsian_log_total_collision_chart_family(
    *,
    branch: FiniteFuchsianLogBranch,
    kind: str = "automatic_identity_selector_total_collision",
    initial_radius: float,
    shell_contraction: float,
    analytic_disk_fraction: float,
    log_growth_factor: float,
    step_ratio_bounds: Mapping[str, float],
    retained_order_initials: Mapping[str, int],
    retained_order_increments: Mapping[str, int],
    component_derivative_orders: Mapping[str, int] | None = None,
    component_multipliers: Mapping[str, float] | None = None,
    components: tuple[str, ...] = DEFAULT_BUDGET_COMPONENTS,
) -> LocalChartFamilyCauchyCertificate:
    """Derive a total-collision chart-family Cauchy certificate from a Fuchsian-log branch.

    This is the event-regime bridge for zero-angular identity-selector total
    collision charts: the primitive Cauchy inputs are not supplied as numeric
    tuples, but are computed from the finite Fuchsian-log branch envelope.
    """

    _reject_raw_bool("branch", branch)
    kind = str(kind)
    if "total_collision" not in kind:
        raise ValueError("Fuchsian-log total-collision chart family kind must name total_collision")
    fuchsian_inputs = derive_finite_fuchsian_log_branch_primitive_cauchy_inputs(
        branch,
        initial_radius=initial_radius,
        shell_contraction=shell_contraction,
        analytic_disk_fraction=analytic_disk_fraction,
        log_growth_factor=log_growth_factor,
        step_ratio_bounds=step_ratio_bounds,
        retained_order_initials=retained_order_initials,
        retained_order_increments=retained_order_increments,
        component_derivative_orders=component_derivative_orders,
        component_multipliers=component_multipliers,
    )
    return certify_local_chart_family_primitive_cauchy_inputs(
        kind=kind,
        primitive_inputs=fuchsian_inputs,
        source_certificate=fuchsian_inputs,
        components=components,
        source="fuchsian_log_total_collision_chart_family_constructor",
    )


def derive_event_recurrence_from_chart_family_certificates(
    *,
    shell_isolation: GeometricShellEventIsolationCertificate,
    chart_family_certificates: tuple[LocalChartFamilyCauchyCertificate, ...],
    checked_prefix: int,
    components: tuple[str, ...] = DEFAULT_BUDGET_COMPONENTS,
    time_direction: str = "unspecified",
    provenance: str = "local_chart_family_certificates",
) -> EventRegimeAssemblyCertificate:
    """Compose local event-chart certificates into an all-future budget."""

    _reject_raw_bool("shell_isolation", shell_isolation)
    components = tuple(str(component) for component in components)
    time_direction = str(time_direction)
    if time_direction not in {
        "future",
        "time_reversed_past",
        "time_reversal_invariant",
        "unspecified",
    }:
        raise ValueError("time_direction is not recognized")
    family_certificates = tuple(chart_family_certificates)
    obligations: list[EventRegimeObligation] = [
        EventRegimeObligation(
            obligation="geometric_shell_event_isolation",
            certified=getattr(shell_isolation, "certified", False) is True,
            source=type(shell_isolation).__name__,
        )
    ]
    expected_counts = dict(getattr(shell_isolation, "chart_family_counts", {}))
    by_kind = {certificate.kind: certificate for certificate in family_certificates}
    duplicate_kinds = len(by_kind) != len(family_certificates)
    obligations.append(
        EventRegimeObligation(
            obligation="unique_chart_family_certificates",
            certified=not duplicate_kinds,
            source="EventRegimeAssemblyCertificate",
            detail="family kinds must be unique",
        )
    )
    for kind in sorted(expected_counts):
        certificate = by_kind.get(kind)
        obligations.append(
            EventRegimeObligation(
                obligation=f"chart_family:{kind}",
                certified=bool(
                    certificate is not None
                    and certificate.certified_for_components(components)
                ),
                source=type(certificate).__name__ if certificate is not None else "missing",
                detail="requires certified primitive inputs for "
                + ",".join(components),
            )
        )
    unexpected_kinds = tuple(kind for kind in by_kind if kind not in expected_counts)
    obligations.append(
        EventRegimeObligation(
            obligation="no_unexpected_chart_families",
            certified=not unexpected_kinds,
            source="EventRegimeAssemblyCertificate",
            detail=",".join(unexpected_kinds),
        )
    )

    event_budget = None
    can_derive_budget = bool(
        getattr(shell_isolation, "certified", False) is True
        and not duplicate_kinds
        and not unexpected_kinds
        and expected_counts
        and all(
            by_kind.get(kind) is not None
            and by_kind[kind].certified_for_components(components)
            for kind in expected_counts
        )
    )
    if can_derive_budget:
        primitive_inputs = {
            kind: {
                component: by_kind[kind].component_input(component)
                for component in components
            }
            for kind in expected_counts
        }
        try:
            event_budget = derive_all_future_event_budget_from_geometric_shell_isolation(
                shell_isolation=shell_isolation,
                primitive_inputs=primitive_inputs,
                checked_prefix=int(checked_prefix),
                components=components,
            )
        except ValueError:
            event_budget = None
    obligations.append(
        EventRegimeObligation(
            obligation="all_future_event_budget",
            certified=bool(event_budget is not None and event_budget.certified),
            source=type(event_budget).__name__ if event_budget is not None else "missing",
            detail="derived from shell isolation and local chart-family Cauchy certificates",
        )
    )
    return EventRegimeAssemblyCertificate(
        shell_isolation=shell_isolation,
        chart_family_certificates=family_certificates,
        event_budget=event_budget,
        components=components,
        obligations=tuple(obligations),
        time_direction=time_direction,
        provenance=str(provenance),
    )


def derive_nonzero_angular_event_recurrence_from_uniform_envelopes(
    *,
    masses: object,
    delta_initial: float,
    theta: float,
    event_isolation_initial: float,
    boundary_clearance_initial: float,
    ordinary_pair_distance_lower_bound: float,
    ordinary_pair_diameter_upper_bound: float,
    ordinary_speed_upper_bound: float,
    binary_pair: tuple[int, int],
    binary_z_bound: float,
    binary_z_velocity_bound: float,
    binary_pair_energy_bound: float,
    binary_center_bound: float,
    binary_center_velocity_bound: float,
    binary_third_offset_bound: float,
    binary_third_offset_velocity_bound: float,
    binary_third_body_nominal_distance_lower_bound: float,
    binary_radii: Mapping[str, float],
    step_ratio_bounds: Mapping[str, float],
    retained_order_initials: Mapping[str, int],
    retained_order_increments: Mapping[str, int],
    checked_prefix: int,
    components: tuple[str, ...] = DEFAULT_BUDGET_COMPONENTS,
    ordinary_component_majorant_multipliers: Mapping[str, float] | None = None,
    binary_component_majorant_multipliers: Mapping[str, float] | None = None,
    time_direction: str = "time_reversal_invariant",
) -> EventRegimeAssemblyCertificate:
    """Derive the binary-only nonzero-angular event recurrence from envelopes.

    This is the theorem-facing event constructor for the first all-time target.
    Its hypotheses are still explicit uniform shell and local state envelopes;
    it does not classify arbitrary data into those envelopes.  Within that
    scoped regime, it derives the geometric shell counts, ordinary-gap Cauchy
    inputs, separated-binary Levi-Civita Cauchy inputs, and the all-future tail
    budget without allowing a total-collision chart family.
    """

    masses_array = np.asarray(masses, dtype=float).reshape(-1)
    if masses_array.shape != (3,):
        raise ValueError("nonzero-angular event recurrence currently expects three masses")
    if np.any(masses_array <= 0.0) or not np.all(np.isfinite(masses_array)):
        raise ValueError("masses must be positive and finite")
    shell_isolation = derive_geometric_shell_event_isolation(
        delta_initial=delta_initial,
        theta=theta,
        event_isolation_initial=event_isolation_initial,
        boundary_clearance_initial=boundary_clearance_initial,
        total_collision_kind=None,
    )
    ordinary_family = certify_uniform_noncollision_ordinary_gap_chart_family(
        masses=masses_array,
        pair_distance_lower_bound=ordinary_pair_distance_lower_bound,
        pair_diameter_upper_bound=ordinary_pair_diameter_upper_bound,
        speed_upper_bound=ordinary_speed_upper_bound,
        step_ratio_bounds=step_ratio_bounds,
        retained_order_initials=retained_order_initials,
        retained_order_increments=retained_order_increments,
        component_majorant_multipliers=ordinary_component_majorant_multipliers,
        components=components,
    )
    separated_binary_family = certify_uniform_separated_binary_levi_civita_chart_family(
        masses=masses_array,
        pair=binary_pair,
        z_bound=binary_z_bound,
        z_velocity_bound=binary_z_velocity_bound,
        pair_energy_bound=binary_pair_energy_bound,
        binary_center_bound=binary_center_bound,
        binary_center_velocity_bound=binary_center_velocity_bound,
        third_offset_bound=binary_third_offset_bound,
        third_offset_velocity_bound=binary_third_offset_velocity_bound,
        third_body_nominal_distance_lower_bound=binary_third_body_nominal_distance_lower_bound,
        radii=binary_radii,
        step_ratio_bounds=step_ratio_bounds,
        retained_order_initials=retained_order_initials,
        retained_order_increments=retained_order_increments,
        component_majorant_multipliers=binary_component_majorant_multipliers,
        components=components,
    )
    return derive_event_recurrence_from_chart_family_certificates(
        shell_isolation=shell_isolation,
        chart_family_certificates=(ordinary_family, separated_binary_family),
        checked_prefix=checked_prefix,
        components=components,
        time_direction=time_direction,
        provenance="uniform_nonzero_angular_single_pair_event_envelopes",
    )


def derive_nonzero_angular_event_recurrence_from_uniform_pair_envelopes(
    *,
    masses: object,
    delta_initial: float,
    theta: float,
    event_isolation_initial: float,
    boundary_clearance_initial: float,
    ordinary_pair_distance_lower_bound: float,
    ordinary_pair_diameter_upper_bound: float,
    ordinary_speed_upper_bound: float,
    binary_pair_envelopes: Mapping[tuple[int, int], Mapping[str, object]],
    step_ratio_bounds: Mapping[str, float],
    retained_order_initials: Mapping[str, int],
    retained_order_increments: Mapping[str, int],
    checked_prefix: int,
    components: tuple[str, ...] = DEFAULT_BUDGET_COMPONENTS,
    ordinary_component_majorant_multipliers: Mapping[str, float] | None = None,
    time_direction: str = "time_reversal_invariant",
    provenance: str = "uniform_nonzero_angular_pair_event_envelopes",
) -> EventRegimeAssemblyCertificate:
    """Derive the nonzero-angular event recurrence with one LC row per pair.

    This is the all-pair version of the binary-only nonzero-angular recurrence.
    The shell packing still bounds the total number of binary events by
    ``M_*``; each individual pair family is bounded by the same ``M_*`` because
    it is a subfamily of the total event set.  The constructor requires
    explicit uniform Levi-Civita envelopes for all three binary pairs and
    derives a separate primitive Cauchy row for each pair.
    """

    masses_array = np.asarray(masses, dtype=float).reshape(-1)
    if masses_array.shape != (3,):
        raise ValueError("nonzero-angular pair recurrence currently expects three masses")
    if np.any(masses_array <= 0.0) or not np.all(np.isfinite(masses_array)):
        raise ValueError("masses must be positive and finite")
    required_pairs = ((0, 1), (0, 2), (1, 2))
    normalized_envelopes: dict[tuple[int, int], Mapping[str, object]] = {}
    for pair, envelope in binary_pair_envelopes.items():
        normalized_pair = _normalize_binary_pair(pair)
        if normalized_pair in normalized_envelopes:
            raise ValueError(f"duplicate binary-pair envelope for {normalized_pair}")
        normalized_envelopes[normalized_pair] = envelope
    if set(normalized_envelopes) != set(required_pairs):
        raise ValueError("uniform pair recurrence requires envelopes for pairs 01, 02, and 12")

    binary_kinds = tuple(_binary_pair_family_kind(pair) for pair in required_pairs)
    shell_isolation = derive_geometric_shell_event_isolation(
        delta_initial=delta_initial,
        theta=theta,
        event_isolation_initial=event_isolation_initial,
        boundary_clearance_initial=boundary_clearance_initial,
        binary_kind=binary_kinds[0],
        additional_binary_kinds=binary_kinds[1:],
        total_collision_kind=None,
    )
    ordinary_family = certify_uniform_noncollision_ordinary_gap_chart_family(
        masses=masses_array,
        pair_distance_lower_bound=ordinary_pair_distance_lower_bound,
        pair_diameter_upper_bound=ordinary_pair_diameter_upper_bound,
        speed_upper_bound=ordinary_speed_upper_bound,
        step_ratio_bounds=step_ratio_bounds,
        retained_order_initials=retained_order_initials,
        retained_order_increments=retained_order_increments,
        component_majorant_multipliers=ordinary_component_majorant_multipliers,
        components=components,
    )
    binary_families: list[LocalChartFamilyCauchyCertificate] = []
    for pair, kind in zip(required_pairs, binary_kinds):
        envelope = normalized_envelopes[pair]
        multipliers = envelope.get("component_majorant_multipliers")  # type: ignore[attr-defined]
        binary_families.append(
            certify_uniform_separated_binary_levi_civita_chart_family(
                masses=masses_array,
                pair=pair,
                z_bound=float(envelope["z_bound"]),
                z_velocity_bound=float(envelope["z_velocity_bound"]),
                pair_energy_bound=float(envelope["pair_energy_bound"]),
                binary_center_bound=float(envelope["binary_center_bound"]),
                binary_center_velocity_bound=float(
                    envelope["binary_center_velocity_bound"]
                ),
                third_offset_bound=float(envelope["third_offset_bound"]),
                third_offset_velocity_bound=float(
                    envelope["third_offset_velocity_bound"]
                ),
                third_body_nominal_distance_lower_bound=float(
                    envelope["third_body_nominal_distance_lower_bound"]
                ),
                radii=envelope["radii"],  # type: ignore[arg-type]
                step_ratio_bounds=step_ratio_bounds,
                retained_order_initials=retained_order_initials,
                retained_order_increments=retained_order_increments,
                kind=kind,
                component_majorant_multipliers=multipliers,  # type: ignore[arg-type]
                components=components,
            )
        )
    return derive_event_recurrence_from_chart_family_certificates(
        shell_isolation=shell_isolation,
        chart_family_certificates=(ordinary_family, *binary_families),
        checked_prefix=checked_prefix,
        components=components,
        time_direction=time_direction,
        provenance=provenance,
    )


def _coerce_component_input(
    primitive_inputs: Mapping[str, PrimitiveCauchyTailInput | tuple[float, float, float, int, int]]
    | object,
    component: str,
) -> PrimitiveCauchyTailInput:
    if hasattr(primitive_inputs, "component_input"):
        value = primitive_inputs.component_input(component)
    else:
        value = primitive_inputs[component]  # type: ignore[index]
    if isinstance(value, PrimitiveCauchyTailInput):
        primitive = value
    else:
        if len(value) != 5:
            raise ValueError("primitive Cauchy tuple must have five entries")
        primitive = construct_primitive_cauchy_tail_input(
            majorant_initial=float(value[0]),
            majorant_growth=float(value[1]),
            step_ratio_bound=float(value[2]),
            retained_order_initial=int(value[3]),
            retained_order_increment=int(value[4]),
        )
    if not primitive.certified:
        raise ValueError(f"primitive Cauchy input for {component} does not certify")
    return primitive


def _is_separated_binary_family_kind(kind: str) -> bool:
    return bool(kind == "separated_binary_levi_civita" or kind.startswith("separated_binary_levi_civita_"))


def _normalize_binary_pair(pair: tuple[int, int]) -> tuple[int, int]:
    if len(pair) != 2:
        raise ValueError("binary pair must have two indices")
    first, second = sorted((int(pair[0]), int(pair[1])))
    if first == second or first not in (0, 1, 2) or second not in (0, 1, 2):
        raise ValueError("binary pair must contain two distinct body indices")
    return (first, second)


def _binary_pair_family_kind(pair: tuple[int, int]) -> str:
    first, second = _normalize_binary_pair(pair)
    return f"separated_binary_levi_civita_{first}{second}"


def _certificate_field(certificate: object, field_names: tuple[str, ...]) -> bool:
    return any(
        getattr(certificate, field) is True
        for field in field_names
        if hasattr(certificate, field)
    )


def _reject_raw_bool(name: str, certificate: object | None) -> None:
    if isinstance(certificate, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be a constructor-derived certificate object, not a raw boolean"
        )

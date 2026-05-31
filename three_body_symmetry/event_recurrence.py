"""All-future event-atlas recurrence constructors.

The constructors here turn primitive local Cauchy tail data into a geometric
all-future budget for compact-time shells.  They do not classify arbitrary
three-body dynamics; they close the recurrence once event counts and local
chart-family Cauchy inputs have been proved elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


DEFAULT_BUDGET_COMPONENTS = (
    "value",
    "first_jet",
    "lifted_residual",
    "physical_residual",
)


@dataclass(frozen=True)
class PrimitiveCauchyTailInput:
    """Primitive Cauchy tail envelope for one chart family/component."""

    majorant_initial: float
    majorant_growth: float
    step_ratio_bound: float
    retained_order_initial: int
    retained_order_increment: int

    @property
    def certified(self) -> bool:
        return bool(
            np.isfinite(self.majorant_initial)
            and self.majorant_initial >= 0.0
            and np.isfinite(self.majorant_growth)
            and self.majorant_growth >= 0.0
            and np.isfinite(self.step_ratio_bound)
            and 0.0 < self.step_ratio_bound < 1.0
            and self.retained_order_initial >= 0
            and self.retained_order_increment >= 1
            and self.shell_ratio < 1.0
        )

    @property
    def first_shell_tail_bound(self) -> float:
        if not (
            np.isfinite(self.majorant_initial)
            and 0.0 < self.step_ratio_bound < 1.0
            and self.retained_order_initial >= 0
        ):
            return float("inf")
        sigma = self.step_ratio_bound
        return float(
            self.majorant_initial
            * sigma ** (self.retained_order_initial + 1)
            / (1.0 - sigma)
        )

    @property
    def shell_ratio(self) -> float:
        if not (
            np.isfinite(self.majorant_growth)
            and np.isfinite(self.step_ratio_bound)
            and self.retained_order_increment >= 0
        ):
            return float("inf")
        return float(self.majorant_growth * self.step_ratio_bound**self.retained_order_increment)

    def retained_order_for_shell(self, shell_index: int) -> int:
        shell_index = int(shell_index)
        if shell_index < 0:
            raise ValueError("shell_index must be nonnegative")
        return int(self.retained_order_initial + self.retained_order_increment * shell_index)

    def shell_tail_bound(self, shell_index: int) -> float:
        shell_index = int(shell_index)
        if shell_index < 0:
            raise ValueError("shell_index must be nonnegative")
        if not self.certified:
            return float("inf")
        return float(self.first_shell_tail_bound * self.shell_ratio**shell_index)

    def cauchy_tail_bound(
        self,
        *,
        shell_index: int,
        actual_majorant: float,
        actual_step_ratio: float,
        retained_order: int | None = None,
    ) -> float:
        """Return the direct Cauchy tail for an observed chart in a shell."""

        shell_index = int(shell_index)
        if shell_index < 0:
            raise ValueError("shell_index must be nonnegative")
        actual_majorant = float(actual_majorant)
        actual_step_ratio = float(actual_step_ratio)
        retained_order = (
            self.retained_order_for_shell(shell_index)
            if retained_order is None
            else int(retained_order)
        )
        if not (
            np.isfinite(actual_majorant)
            and actual_majorant >= 0.0
            and np.isfinite(actual_step_ratio)
            and 0.0 <= actual_step_ratio < 1.0
            and retained_order >= 0
        ):
            return float("inf")
        return float(
            actual_majorant
            * actual_step_ratio ** (retained_order + 1)
            / (1.0 - actual_step_ratio)
        )

    def certifies_actual_tail(
        self,
        *,
        shell_index: int,
        actual_majorant: float,
        actual_step_ratio: float,
        retained_order: int | None = None,
        tolerance: float = 1e-12,
    ) -> bool:
        """Check the primitive inequalities imply the geometric tail bound."""

        shell_index = int(shell_index)
        retained_order = (
            self.retained_order_for_shell(shell_index)
            if retained_order is None
            else int(retained_order)
        )
        majorant_cap = self.majorant_initial * self.majorant_growth**shell_index
        return bool(
            self.certified
            and np.isfinite(actual_majorant)
            and np.isfinite(actual_step_ratio)
            and actual_majorant <= majorant_cap * (1.0 + tolerance)
            and 0.0 <= actual_step_ratio <= self.step_ratio_bound * (1.0 + tolerance)
            and retained_order >= self.retained_order_for_shell(shell_index)
            and self.cauchy_tail_bound(
                shell_index=shell_index,
                actual_majorant=actual_majorant,
                actual_step_ratio=actual_step_ratio,
                retained_order=retained_order,
            )
            <= self.shell_tail_bound(shell_index) * (1.0 + tolerance)
        )


@dataclass(frozen=True)
class ChartFamilyPrimitiveCauchyBudget:
    """Primitive Cauchy inputs and chart-count envelope for one family."""

    kind: str
    count_bound: int
    component_inputs: Mapping[str, PrimitiveCauchyTailInput]

    def has_components(self, components: tuple[str, ...]) -> bool:
        return all(component in self.component_inputs for component in components)

    def certified_for_components(self, components: tuple[str, ...]) -> bool:
        return bool(
            self.kind
            and isinstance(self.count_bound, int)
            and self.count_bound >= 0
            and self.has_components(components)
            and all(self.component_inputs[component].certified for component in components)
        )

    def shell_bound(self, component: str, shell_index: int) -> float:
        if component not in self.component_inputs:
            return float("inf")
        return float(self.count_bound * self.component_inputs[component].shell_tail_bound(shell_index))


@dataclass(frozen=True)
class EventIsolationChartCountCertificate:
    """Per-shell chart-family counts derived from uniform event isolation."""

    isolation_fraction: float
    boundary_inclusive: bool = True
    ordinary_kind: str = "ordinary_gap_taylor"
    binary_kind: str = "separated_binary_levi_civita"
    additional_binary_kinds: tuple[str, ...] = ()
    total_collision_kind: str | None = "automatic_identity_selector_total_collision"

    @property
    def certified(self) -> bool:
        family_kinds = (
            self.ordinary_kind,
            self.binary_kind,
            *self.additional_binary_kinds,
            *((self.total_collision_kind,) if self.total_collision_kind is not None else ()),
        )
        return bool(
            np.isfinite(self.isolation_fraction)
            and 0.0 < self.isolation_fraction <= 1.0
            and all(str(kind) for kind in family_kinds)
            and len(set(family_kinds)) == len(family_kinds)
        )

    @property
    def event_count_bound(self) -> int:
        if not self.certified:
            return -1
        boundary_allowance = 2 if self.boundary_inclusive else 0
        return int(np.floor(1.0 / self.isolation_fraction)) + boundary_allowance

    @property
    def chart_family_counts(self) -> dict[str, int]:
        if not self.certified:
            return {}
        event_bound = self.event_count_bound
        counts = {
            self.ordinary_kind: event_bound + 1,
            self.binary_kind: event_bound,
        }
        for kind in self.additional_binary_kinds:
            counts[str(kind)] = event_bound
        if self.total_collision_kind is not None:
            counts[self.total_collision_kind] = event_bound
        return counts


@dataclass(frozen=True)
class AllFutureEventBudgetCertificate:
    """Geometric all-future budget derived from primitive local Cauchy data."""

    family_budgets: tuple[ChartFamilyPrimitiveCauchyBudget, ...]
    checked_prefix: int
    components: tuple[str, ...] = DEFAULT_BUDGET_COMPONENTS
    witness_source: str = "primitive_cauchy_event_recurrence"

    @property
    def family_kinds(self) -> tuple[str, ...]:
        return tuple(family.kind for family in self.family_budgets)

    @property
    def certified(self) -> bool:
        return bool(
            self.checked_prefix >= 0
            and self.components
            and self.family_budgets
            and len(set(self.family_kinds)) == len(self.family_kinds)
            and all(
                family.certified_for_components(self.components)
                for family in self.family_budgets
            )
            and all(self.component_scalar_ratio(component) < 1.0 for component in self.components)
        )

    def family_budget(self, kind: str) -> ChartFamilyPrimitiveCauchyBudget:
        for family in self.family_budgets:
            if family.kind == kind:
                return family
        raise KeyError(kind)

    def component_first_shell_bound(self, component: str) -> float:
        if component not in self.components:
            raise KeyError(component)
        return float(sum(family.shell_bound(component, 0) for family in self.family_budgets))

    def component_scalar_ratio(self, component: str) -> float:
        if component not in self.components:
            raise KeyError(component)
        ratios = [
            family.component_inputs[component].shell_ratio
            for family in self.family_budgets
            if component in family.component_inputs
        ]
        return float(max(ratios)) if ratios else float("inf")

    def component_family_shell_bound(self, component: str, shell_index: int) -> float:
        if component not in self.components:
            raise KeyError(component)
        return float(sum(family.shell_bound(component, shell_index) for family in self.family_budgets))

    def component_scalar_shell_bound(self, component: str, shell_index: int) -> float:
        shell_index = int(shell_index)
        if shell_index < 0:
            raise ValueError("shell_index must be nonnegative")
        return float(
            self.component_first_shell_bound(component)
            * self.component_scalar_ratio(component) ** shell_index
        )

    def component_prefix_scalar_bound(self, component: str, prefix_length: int | None = None) -> float:
        prefix_length = self.checked_prefix if prefix_length is None else int(prefix_length)
        if prefix_length < 0:
            raise ValueError("prefix_length must be nonnegative")
        ratio = self.component_scalar_ratio(component)
        first = self.component_first_shell_bound(component)
        if not np.isfinite(ratio) or ratio >= 1.0:
            return float("inf")
        if prefix_length == 0:
            return 0.0
        return float(first * (1.0 - ratio**prefix_length) / (1.0 - ratio))

    def component_sharp_family_tail_from_prefix(
        self,
        component: str,
        prefix_length: int | None = None,
    ) -> float:
        prefix_length = self.checked_prefix if prefix_length is None else int(prefix_length)
        if prefix_length < 0:
            raise ValueError("prefix_length must be nonnegative")
        total = 0.0
        for family in self.family_budgets:
            primitive = family.component_inputs[component]
            total += (
                family.count_bound
                * primitive.first_shell_tail_bound
                * primitive.shell_ratio**prefix_length
                / (1.0 - primitive.shell_ratio)
            )
        return float(total)

    def component_scalar_tail_from_prefix(
        self,
        component: str,
        prefix_length: int | None = None,
    ) -> float:
        prefix_length = self.checked_prefix if prefix_length is None else int(prefix_length)
        if prefix_length < 0:
            raise ValueError("prefix_length must be nonnegative")
        ratio = self.component_scalar_ratio(component)
        if not np.isfinite(ratio) or ratio >= 1.0:
            return float("inf")
        return float(
            self.component_first_shell_bound(component)
            * ratio**prefix_length
            / (1.0 - ratio)
        )

    def component_infinite_scalar_bound(self, component: str) -> float:
        ratio = self.component_scalar_ratio(component)
        if not np.isfinite(ratio) or ratio >= 1.0:
            return float("inf")
        return float(self.component_first_shell_bound(component) / (1.0 - ratio))

    def component_recurrence_closes(self, component: str) -> bool:
        if component not in self.components:
            return False
        scalar_tail = self.component_scalar_tail_from_prefix(component)
        sharp_tail = self.component_sharp_family_tail_from_prefix(component)
        prefix = self.component_prefix_scalar_bound(component)
        infinite = self.component_infinite_scalar_bound(component)
        return bool(
            self.certified
            and np.isfinite(scalar_tail)
            and np.isfinite(sharp_tail)
            and sharp_tail <= scalar_tail * (1.0 + 1e-12)
            and prefix + scalar_tail <= infinite * (1.0 + 1e-12)
        )

    @property
    def recurrence_closes(self) -> bool:
        return bool(self.certified and all(self.component_recurrence_closes(component) for component in self.components))


@dataclass(frozen=True)
class GeometricShellEventIsolationCertificate:
    """Uniform event-count input derived from geometric compact-time shells."""

    delta_initial: float
    theta: float
    event_isolation_initial: float
    boundary_clearance_initial: float
    isolation_fraction: float
    count_certificate: EventIsolationChartCountCertificate

    @property
    def initial_shell_width(self) -> float:
        return float(self.delta_initial * (1.0 - self.theta))

    @property
    def certified(self) -> bool:
        return bool(
            np.isfinite(self.delta_initial)
            and 0.0 < self.delta_initial <= 2.0
            and np.isfinite(self.theta)
            and 0.0 < self.theta < 1.0
            and np.isfinite(self.event_isolation_initial)
            and self.event_isolation_initial > 0.0
            and np.isfinite(self.boundary_clearance_initial)
            and self.boundary_clearance_initial > 0.0
            and np.isfinite(self.isolation_fraction)
            and 0.0 < self.isolation_fraction <= 1.0
            and self.count_certificate.certified
            and np.isclose(
                self.count_certificate.isolation_fraction,
                self.isolation_fraction,
            )
        )

    @property
    def event_count_bound(self) -> int:
        return self.count_certificate.event_count_bound

    @property
    def chart_family_counts(self) -> dict[str, int]:
        return self.count_certificate.chart_family_counts

    def shell_interval(self, shell_index: int) -> tuple[float, float]:
        shell_index = int(shell_index)
        if shell_index < 0:
            raise ValueError("shell_index must be nonnegative")
        left = 1.0 - self.delta_initial * self.theta**shell_index
        right = 1.0 - self.delta_initial * self.theta ** (shell_index + 1)
        return (float(left), float(right))

    def shell_width(self, shell_index: int) -> float:
        left, right = self.shell_interval(shell_index)
        return float(right - left)

    def event_isolation_lower_bound(self, shell_index: int) -> float:
        shell_index = int(shell_index)
        if shell_index < 0:
            raise ValueError("shell_index must be nonnegative")
        return float(self.event_isolation_initial * self.theta**shell_index)

    def boundary_clearance_lower_bound(self, shell_index: int) -> float:
        shell_index = int(shell_index)
        if shell_index < 0:
            raise ValueError("shell_index must be nonnegative")
        return float(self.boundary_clearance_initial * self.theta**shell_index)

    def certifies_observed_shell(
        self,
        *,
        shell_index: int,
        event_centers: tuple[float, ...] | list[float],
        tolerance: float = 1e-10,
    ) -> bool:
        """Check observed event centers satisfy the geometric packing inputs."""

        if not self.certified:
            return False
        centers = tuple(float(center) for center in event_centers)
        if any(not np.isfinite(center) for center in centers):
            return False
        left, right = self.shell_interval(shell_index)
        sorted_centers = tuple(sorted(centers))
        if sorted_centers != centers:
            return False
        if any(not left < center < right for center in centers):
            return False
        if len(centers) > self.event_count_bound:
            return False
        if not centers:
            return True
        boundary_clearance = min(centers[0] - left, right - centers[-1])
        pair_separation = min(
            (right_center - left_center)
            for left_center, right_center in zip(centers, centers[1:])
        ) if len(centers) > 1 else float("inf")
        return bool(
            boundary_clearance
            >= self.boundary_clearance_lower_bound(shell_index) * (1.0 - tolerance)
            and pair_separation
            >= self.event_isolation_lower_bound(shell_index) * (1.0 - tolerance)
        )

    def observed_chart_family_counts(
        self,
        event_types: tuple[str, ...] | list[str],
    ) -> dict[str, int]:
        """Derive actual chart-family counts from typed events in one shell."""

        event_types = tuple(str(event_type) for event_type in event_types)
        expected_counts = self.chart_family_counts
        ordinary_kind = str(self.count_certificate.ordinary_kind)
        if ordinary_kind not in expected_counts:
            raise ValueError("ordinary chart family is not present in shell counts")
        counts = {kind: 0 for kind in expected_counts}
        counts[ordinary_kind] = len(event_types) + 1
        for event_type in event_types:
            if event_type == ordinary_kind:
                raise ValueError("ordinary gaps are derived between events, not supplied as events")
            if event_type not in expected_counts:
                raise ValueError(f"unknown event chart family {event_type!r}")
            counts[event_type] += 1
        return counts

    def certifies_observed_typed_shell(
        self,
        *,
        shell_index: int,
        event_centers: tuple[float, ...] | list[float],
        event_types: tuple[str, ...] | list[str],
        tolerance: float = 1e-10,
    ) -> bool:
        """Check observed event centers and typed chart counts fit the shell budget."""

        try:
            centers = tuple(float(center) for center in event_centers)
            types = tuple(str(event_type) for event_type in event_types)
            actual_counts = self.observed_chart_family_counts(types)
        except (TypeError, ValueError):
            return False
        if len(centers) != len(types):
            return False
        expected_counts = self.chart_family_counts
        return bool(
            self.certifies_observed_shell(
                shell_index=shell_index,
                event_centers=centers,
                tolerance=tolerance,
            )
            and actual_counts.keys() == expected_counts.keys()
            and all(actual_counts[kind] <= expected_counts[kind] for kind in expected_counts)
        )


@dataclass(frozen=True)
class GeometricShellEventBudgetCertificate:
    """All-future budget closed from geometric shell isolation and Cauchy inputs."""

    shell_isolation: GeometricShellEventIsolationCertificate
    event_budget: AllFutureEventBudgetCertificate

    @property
    def certified(self) -> bool:
        expected_counts = self.shell_isolation.chart_family_counts
        observed_counts = {
            family.kind: family.count_bound
            for family in self.event_budget.family_budgets
        }
        return bool(
            self.shell_isolation.certified
            and self.event_budget.recurrence_closes
            and observed_counts == expected_counts
        )

    @property
    def recurrence_closes(self) -> bool:
        return self.event_budget.recurrence_closes


def construct_primitive_cauchy_tail_input(
    *,
    majorant_initial: float,
    majorant_growth: float,
    step_ratio_bound: float,
    retained_order_initial: int,
    retained_order_increment: int,
) -> PrimitiveCauchyTailInput:
    primitive = PrimitiveCauchyTailInput(
        majorant_initial=float(majorant_initial),
        majorant_growth=float(majorant_growth),
        step_ratio_bound=float(step_ratio_bound),
        retained_order_initial=int(retained_order_initial),
        retained_order_increment=int(retained_order_increment),
    )
    if not primitive.certified:
        raise ValueError("primitive Cauchy tail input does not certify")
    return primitive


def derive_chart_family_counts_from_event_isolation(
    *,
    isolation_fraction: float,
    boundary_inclusive: bool = True,
    ordinary_kind: str = "ordinary_gap_taylor",
    binary_kind: str = "separated_binary_levi_civita",
    additional_binary_kinds: tuple[str, ...] = (),
    total_collision_kind: str | None = "automatic_identity_selector_total_collision",
) -> EventIsolationChartCountCertificate:
    certificate = EventIsolationChartCountCertificate(
        isolation_fraction=float(isolation_fraction),
        boundary_inclusive=bool(boundary_inclusive),
        ordinary_kind=str(ordinary_kind),
        binary_kind=str(binary_kind),
        additional_binary_kinds=tuple(str(kind) for kind in additional_binary_kinds),
        total_collision_kind=None
        if total_collision_kind is None
        else str(total_collision_kind),
    )
    if not certificate.certified:
        raise ValueError("event isolation fraction must be in (0, 1]")
    return certificate


def derive_geometric_shell_event_isolation(
    *,
    delta_initial: float,
    theta: float,
    event_isolation_initial: float,
    boundary_clearance_initial: float,
    boundary_inclusive: bool = True,
    ordinary_kind: str = "ordinary_gap_taylor",
    binary_kind: str = "separated_binary_levi_civita",
    additional_binary_kinds: tuple[str, ...] = (),
    total_collision_kind: str | None = "automatic_identity_selector_total_collision",
) -> GeometricShellEventIsolationCertificate:
    """Derive the uniform packing fraction from geometric shell isolation.

    Shells are ``[1-delta_0 theta^n, 1-delta_0 theta^(n+1)]``.  If every event
    in shell ``n`` has event-free compact-time radius at least ``H_0 theta^n``
    and boundary clearance at least ``B_0 theta^n``, then the packing fraction
    is ``min(H_0,B_0)/(delta_0(1-theta))``.
    """

    delta_initial = float(delta_initial)
    theta = float(theta)
    event_isolation_initial = float(event_isolation_initial)
    boundary_clearance_initial = float(boundary_clearance_initial)
    if not (
        np.isfinite(delta_initial)
        and 0.0 < delta_initial <= 2.0
        and np.isfinite(theta)
        and 0.0 < theta < 1.0
        and np.isfinite(event_isolation_initial)
        and event_isolation_initial > 0.0
        and np.isfinite(boundary_clearance_initial)
        and boundary_clearance_initial > 0.0
    ):
        raise ValueError("geometric shell isolation parameters do not certify")
    initial_shell_width = delta_initial * (1.0 - theta)
    raw_fraction = min(event_isolation_initial, boundary_clearance_initial) / initial_shell_width
    if not np.isfinite(raw_fraction) or raw_fraction <= 0.0:
        raise ValueError("geometric shell isolation fraction did not certify")
    isolation_fraction = float(min(raw_fraction, 1.0))
    count_certificate = derive_chart_family_counts_from_event_isolation(
        isolation_fraction=isolation_fraction,
        boundary_inclusive=boundary_inclusive,
        ordinary_kind=ordinary_kind,
        binary_kind=binary_kind,
        additional_binary_kinds=additional_binary_kinds,
        total_collision_kind=total_collision_kind,
    )
    certificate = GeometricShellEventIsolationCertificate(
        delta_initial=delta_initial,
        theta=theta,
        event_isolation_initial=event_isolation_initial,
        boundary_clearance_initial=boundary_clearance_initial,
        isolation_fraction=isolation_fraction,
        count_certificate=count_certificate,
    )
    if not certificate.certified:
        raise ValueError("geometric shell isolation certificate did not certify")
    return certificate


def derive_all_future_event_budget_from_primitive_cauchy_inputs(
    *,
    counts: Mapping[str, int],
    primitive_inputs: Mapping[str, Mapping[str, PrimitiveCauchyTailInput | tuple[float, float, float, int, int]]],
    checked_prefix: int,
    components: tuple[str, ...] = DEFAULT_BUDGET_COMPONENTS,
) -> AllFutureEventBudgetCertificate:
    families: list[ChartFamilyPrimitiveCauchyBudget] = []
    for kind in sorted(primitive_inputs):
        if kind not in counts:
            raise ValueError(f"missing count bound for chart family {kind}")
        component_inputs: dict[str, PrimitiveCauchyTailInput] = {}
        for component in components:
            if component not in primitive_inputs[kind]:
                raise ValueError(f"missing primitive Cauchy input for {kind}:{component}")
            component_inputs[component] = _coerce_primitive_input(primitive_inputs[kind][component])
        families.append(
            ChartFamilyPrimitiveCauchyBudget(
                kind=str(kind),
                count_bound=int(counts[kind]),
                component_inputs=component_inputs,
            )
        )
    certificate = AllFutureEventBudgetCertificate(
        family_budgets=tuple(families),
        checked_prefix=int(checked_prefix),
        components=tuple(components),
    )
    if not certificate.recurrence_closes:
        raise ValueError("primitive Cauchy inputs do not close an all-future event budget")
    return certificate


def derive_all_future_event_budget_from_geometric_shell_isolation(
    *,
    shell_isolation: GeometricShellEventIsolationCertificate,
    primitive_inputs: Mapping[str, Mapping[str, PrimitiveCauchyTailInput | tuple[float, float, float, int, int]]],
    checked_prefix: int,
    components: tuple[str, ...] = DEFAULT_BUDGET_COMPONENTS,
) -> GeometricShellEventBudgetCertificate:
    """Compose geometric shell isolation with primitive Cauchy tail inputs."""

    if not shell_isolation.certified:
        raise ValueError("shell isolation certificate must certify")
    event_budget = derive_all_future_event_budget_from_primitive_cauchy_inputs(
        counts=shell_isolation.chart_family_counts,
        primitive_inputs=primitive_inputs,
        checked_prefix=checked_prefix,
        components=components,
    )
    certificate = GeometricShellEventBudgetCertificate(
        shell_isolation=shell_isolation,
        event_budget=event_budget,
    )
    if not certificate.certified:
        raise ValueError("geometric shell event budget did not certify")
    return certificate


def _coerce_primitive_input(
    value: PrimitiveCauchyTailInput | tuple[float, float, float, int, int],
) -> PrimitiveCauchyTailInput:
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
        raise ValueError("primitive Cauchy tail input does not certify")
    return primitive

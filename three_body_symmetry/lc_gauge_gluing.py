"""Exact combinatorial certificates for planar Levi-Civita gauge gluing.

The planar Levi-Civita deck transformation flips ``(z, z_velocity)`` and
fixes every other lifted variable.  If independently validated, connected
chart overlaps are labelled by one bit: zero for the same representative and
one for the antipodal representative.  This module checks whether those
*supplied* bits can be removed by choosing one gauge bit per chart.

This is intentionally only an exact graph kernel.  It does not prove that two
charts overlap, describe the same exact solution, or have the claimed edge
parity.  Its results are not used by any existing ``certified`` or
``proof_certified`` path.  A proof-bearing integration must first derive every
edge from a separate analytic overlap checker.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlanarLCGaugeOverlapEdge:
    """One supplied sign relation on a connected LC-chart overlap."""

    overlap_id: str
    source_chart_id: str
    target_chart_id: str
    parity: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "overlap_id": self.overlap_id,
            "source_chart_id": self.source_chart_id,
            "target_chart_id": self.target_chart_id,
            "parity": self.parity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanarLCGaugeOverlapEdge":
        return cls(
            overlap_id=data.get("overlap_id", ""),
            source_chart_id=data.get("source_chart_id", ""),
            target_chart_id=data.get("target_chart_id", ""),
            parity=data.get("parity", -1),
        )


@dataclass(frozen=True)
class PlanarLCGaugeGluingCertificate:
    """Finite overlap graph with supplied exact LC deck-transformation bits."""

    certificate_id: str
    chart_ids: tuple[str, ...]
    overlaps: tuple[PlanarLCGaugeOverlapEdge, ...]
    source: str = "serialized_planar_lc_gauge_gluing"

    def to_dict(self) -> dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "chart_ids": list(self.chart_ids),
            "overlaps": [edge.to_dict() for edge in self.overlaps],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanarLCGaugeGluingCertificate":
        raw_chart_ids = data.get("chart_ids", ())
        chart_ids = (
            tuple(raw_chart_ids)
            if isinstance(raw_chart_ids, (list, tuple))
            else ()
        )
        raw_overlaps = data.get("overlaps", ())
        overlaps: tuple[object, ...]
        if isinstance(raw_overlaps, (list, tuple)):
            overlaps = tuple(
                edge
                if type(edge) is PlanarLCGaugeOverlapEdge
                else PlanarLCGaugeOverlapEdge.from_dict(edge)
                if isinstance(edge, dict)
                else edge
                for edge in raw_overlaps
            )
        else:
            overlaps = ()
        return cls(
            certificate_id=data.get("certificate_id", ""),
            chart_ids=chart_ids,
            overlaps=overlaps,  # type: ignore[arg-type]
            source=data.get("source", "serialized_planar_lc_gauge_gluing"),
        )


@dataclass(frozen=True)
class PlanarLCGaugeGluingObligation:
    """One exact schema or graph equation checked by the gauge kernel."""

    obligation: str
    certified: bool
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "certified", self.certified is True)


_PLANAR_LC_GAUGE_GLUING_OBLIGATION_NAMES = (
    "lc_gauge_certificate_identity",
    "lc_gauge_chart_ids",
    "lc_gauge_overlap_schema",
    "lc_gauge_graph_connected",
    "lc_gauge_equations_compatible",
)


@dataclass(frozen=True)
class PlanarLCGaugeGluingCheckResult:
    """A primal gauge assignment or a self-contained odd-cycle witness."""

    certificate_id: str
    checker_id: str
    obligations: tuple[PlanarLCGaugeGluingObligation, ...]
    chart_ids: tuple[str, ...]
    checked_overlaps: tuple[PlanarLCGaugeOverlapEdge, ...]
    gauge_assignment: tuple[tuple[str, int], ...]
    component_count: int
    obstruction_cycle_chart_ids: tuple[str, ...]
    obstruction_cycle_edges: tuple[PlanarLCGaugeOverlapEdge, ...]

    def _obligation_ledger_schema_valid(self) -> bool:
        return bool(
            type(self.obligations) is tuple
            and len(self.obligations)
            == len(_PLANAR_LC_GAUGE_GLUING_OBLIGATION_NAMES)
            and all(
                type(obligation) is PlanarLCGaugeGluingObligation
                and type(obligation.obligation) is str
                and bool(obligation.obligation)
                and type(obligation.certified) is bool
                and type(obligation.detail) is str
                for obligation in self.obligations
            )
            and tuple(obligation.obligation for obligation in self.obligations)
            == _PLANAR_LC_GAUGE_GLUING_OBLIGATION_NAMES
        )

    def _obligation(self, name: str) -> bool:
        if not self._obligation_ledger_schema_valid():
            return False
        return any(
            obligation.obligation == name
            and obligation.certified is True
            for obligation in self.obligations
        )

    def _assignment(self) -> dict[str, int] | None:
        if not (
            type(self.chart_ids) is tuple
            and all(
                type(chart_id) is str and bool(chart_id)
                for chart_id in self.chart_ids
            )
            and len(set(self.chart_ids)) == len(self.chart_ids)
            and type(self.gauge_assignment) is tuple
            and len(self.gauge_assignment) == len(self.chart_ids)
            and all(
                type(item) is tuple
                and len(item) == 2
                and type(item[0]) is str
                and bool(item[0])
                and type(item[1]) is int
                and item[1] in (0, 1)
                for item in self.gauge_assignment
            )
        ):
            return None
        assignment = dict(self.gauge_assignment)
        if len(assignment) != len(self.gauge_assignment):
            return None
        if set(assignment) != set(self.chart_ids):
            return None
        return assignment

    def _self_contained_schema_valid(self) -> bool:
        if not (
            type(self.chart_ids) is tuple
            and self.chart_ids
            and all(
                type(chart_id) is str and bool(chart_id)
                for chart_id in self.chart_ids
            )
            and len(set(self.chart_ids)) == len(self.chart_ids)
        ):
            return False
        known_chart_ids = set(self.chart_ids)
        if not (
            type(self.checked_overlaps) is tuple
            and all(
                type(edge) is PlanarLCGaugeOverlapEdge
                and type(edge.overlap_id) is str
                and bool(edge.overlap_id)
                and type(edge.source_chart_id) is str
                and type(edge.target_chart_id) is str
                and edge.source_chart_id in known_chart_ids
                and edge.target_chart_id in known_chart_ids
                and edge.source_chart_id != edge.target_chart_id
                and type(edge.parity) is int
                and edge.parity in (0, 1)
                for edge in self.checked_overlaps
            )
        ):
            return False
        overlap_ids = tuple(edge.overlap_id for edge in self.checked_overlaps)
        return len(set(overlap_ids)) == len(overlap_ids)

    def _computed_component_count(self) -> int | None:
        if not self._self_contained_schema_valid():
            return None
        adjacency: dict[str, list[str]] = {
            chart_id: [] for chart_id in self.chart_ids
        }
        for edge in self.checked_overlaps:
            adjacency[edge.source_chart_id].append(edge.target_chart_id)
            adjacency[edge.target_chart_id].append(edge.source_chart_id)
        visited: set[str] = set()
        component_count = 0
        for root in self.chart_ids:
            if root in visited:
                continue
            component_count += 1
            visited.add(root)
            queue: deque[str] = deque([root])
            while queue:
                current = queue.popleft()
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        return component_count

    @property
    def well_formed(self) -> bool:
        return bool(
            type(self.certificate_id) is str
            and bool(self.certificate_id)
            and type(self.checker_id) is str
            and self.checker_id == "planar_lc_z2_gauge_gluing_checker_v1"
            and self._obligation_ledger_schema_valid()
            and self._self_contained_schema_valid()
            and self._obligation("lc_gauge_certificate_identity")
            and self._obligation("lc_gauge_chart_ids")
            and self._obligation("lc_gauge_overlap_schema")
        )

    @property
    def connected(self) -> bool:
        computed_component_count = self._computed_component_count()
        return bool(
            self.well_formed
            and type(self.component_count) is int
            and self.component_count == computed_component_count == 1
            and self._obligation("lc_gauge_graph_connected")
        )

    @property
    def compatible(self) -> bool:
        """Whether the supplied equations are consistent, component by component."""

        assignment = self._assignment()
        return bool(
            self.well_formed
            and assignment is not None
            and self._obligation("lc_gauge_equations_compatible")
            and all(
                assignment[edge.source_chart_id]
                ^ assignment[edge.target_chart_id]
                == edge.parity
                for edge in self.checked_overlaps
            )
        )

    @property
    def certified(self) -> bool:
        """A connected, well-formed overlap graph with a checked primal witness."""

        return bool(
            type(self.checker_id) is str
            and self.checker_id == "planar_lc_z2_gauge_gluing_checker_v1"
            and self._obligation_ledger_schema_valid()
            and self.component_count == 1
            and self.connected
            and self.compatible
            and not self.obstruction_cycle_chart_ids
            and not self.obstruction_cycle_edges
            and self.obligations
            and all(
                type(obligation) is PlanarLCGaugeGluingObligation
                and obligation.certified is True
                for obligation in self.obligations
            )
        )

    @property
    def obstruction_cycle_overlap_ids(self) -> tuple[str, ...]:
        if not (
            type(self.obstruction_cycle_edges) is tuple
            and all(
                type(edge) is PlanarLCGaugeOverlapEdge
                and type(edge.overlap_id) is str
                for edge in self.obstruction_cycle_edges
            )
        ):
            return ()
        return tuple(edge.overlap_id for edge in self.obstruction_cycle_edges)

    @property
    def obstruction_parity(self) -> int:
        if type(self.obstruction_cycle_edges) is not tuple:
            return -1
        parity = 0
        for edge in self.obstruction_cycle_edges:
            if type(edge) is not PlanarLCGaugeOverlapEdge or type(edge.parity) is not int:
                return -1
            parity ^= edge.parity
        return parity

    @property
    def obstruction_certified(self) -> bool:
        """Whether the result itself contains a valid loopless odd closed cycle."""

        vertices = self.obstruction_cycle_chart_ids
        edges = self.obstruction_cycle_edges
        if not (
            type(self.checker_id) is str
            and self.checker_id == "planar_lc_z2_gauge_gluing_checker_v1"
            and self._obligation_ledger_schema_valid()
            and self.well_formed
            and not self.compatible
            and type(vertices) is tuple
            and all(type(vertex) is str and bool(vertex) for vertex in vertices)
            and type(edges) is tuple
            and all(
                type(edge) is PlanarLCGaugeOverlapEdge
                and type(edge.overlap_id) is str
                and bool(edge.overlap_id)
                and type(edge.source_chart_id) is str
                and type(edge.target_chart_id) is str
                and type(edge.parity) is int
                and edge.parity in (0, 1)
                for edge in edges
            )
            and len(edges) >= 2
            and len(vertices) == len(edges) + 1
            and vertices[0] == vertices[-1]
            and len({edge.overlap_id for edge in edges}) == len(edges)
            and self.obstruction_parity == 1
        ):
            return False
        checked_edge_by_id = {
            edge.overlap_id: edge for edge in self.checked_overlaps
        }
        if not all(
            checked_edge_by_id.get(edge.overlap_id) == edge for edge in edges
        ):
            return False
        for left, right, edge in zip(vertices, vertices[1:], edges):
            if not (
                type(edge) is PlanarLCGaugeOverlapEdge
                and type(edge.parity) is int
                and edge.parity in (0, 1)
                and {left, right}
                == {edge.source_chart_id, edge.target_chart_id}
            ):
                return False
        return True

    @property
    def missing_obligations(self) -> tuple[str, ...]:
        if not self._obligation_ledger_schema_valid():
            return ("lc_gauge_gluing_obligation_schema",)
        return tuple(
            obligation.obligation
            for obligation in self.obligations
            if obligation.certified is not True
        )


def _fundamental_cycle(
    source: str,
    target: str,
    conflict: PlanarLCGaugeOverlapEdge,
    parent: dict[str, str | None],
    parent_edge: dict[str, PlanarLCGaugeOverlapEdge],
) -> tuple[tuple[str, ...], tuple[PlanarLCGaugeOverlapEdge, ...]]:
    """Return the tree path from source to target, closed by ``conflict``."""

    source_up = [source]
    while parent[source_up[-1]] is not None:
        source_up.append(parent[source_up[-1]])  # type: ignore[arg-type]
    source_index = {chart_id: index for index, chart_id in enumerate(source_up)}

    target_up = [target]
    while target_up[-1] not in source_index:
        next_chart = parent[target_up[-1]]
        if next_chart is None:
            raise ValueError("conflict endpoints do not share a spanning-tree root")
        target_up.append(next_chart)

    common = target_up[-1]
    source_segment = source_up[: source_index[common] + 1]
    target_segment = target_up[:-1]
    path_vertices = tuple(source_segment + list(reversed(target_segment)))
    source_edges = tuple(parent_edge[chart_id] for chart_id in source_segment[:-1])
    target_edges = tuple(
        parent_edge[chart_id] for chart_id in reversed(target_segment)
    )
    return path_vertices + (source,), source_edges + target_edges + (conflict,)


def check_planar_lc_gauge_gluing(
    certificate: PlanarLCGaugeGluingCertificate,
) -> PlanarLCGaugeGluingCheckResult:
    """Solve ``t_u XOR t_v = parity_e`` and emit a primal or dual witness.

    All arithmetic is on Python integers restricted to zero and one.  Edge
    labels remain supplied hypotheses; this function checks only their global
    consistency.
    """

    exact_certificate_type = type(certificate) is PlanarLCGaugeGluingCertificate
    certificate_id = (
        certificate.certificate_id
        if exact_certificate_type and type(certificate.certificate_id) is str
        else ""
    )
    source_ok = bool(
        exact_certificate_type
        and type(certificate.source) is str
        and bool(certificate.source)
    )
    identity_ok = bool(exact_certificate_type and certificate_id and source_ok)
    chart_ids = certificate.chart_ids if exact_certificate_type else ()
    chart_ids_ok = bool(
        type(chart_ids) is tuple
        and chart_ids
        and all(type(chart_id) is str and chart_id for chart_id in chart_ids)
        and len(set(chart_ids)) == len(chart_ids)
    )
    overlaps = certificate.overlaps if exact_certificate_type else ()
    overlaps_for_check = overlaps if type(overlaps) is tuple else ()
    overlap_types_ok = bool(
        type(overlaps) is tuple
        and all(type(edge) is PlanarLCGaugeOverlapEdge for edge in overlaps_for_check)
    )
    known_chart_ids = set(chart_ids) if chart_ids_ok else set()
    overlap_ids = tuple(
        edge.overlap_id
        for edge in overlaps_for_check
        if type(edge) is PlanarLCGaugeOverlapEdge
    )
    overlap_schema_ok = bool(
        chart_ids_ok
        and overlap_types_ok
        and all(
            type(edge.overlap_id) is str
            and bool(edge.overlap_id)
            and type(edge.source_chart_id) is str
            and type(edge.target_chart_id) is str
            and edge.source_chart_id in known_chart_ids
            and edge.target_chart_id in known_chart_ids
            and edge.source_chart_id != edge.target_chart_id
            and type(edge.parity) is int
            and edge.parity in (0, 1)
            for edge in overlaps_for_check
        )
        and len(set(overlap_ids)) == len(overlap_ids)
    )

    assignment: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    parent_edge: dict[str, PlanarLCGaugeOverlapEdge] = {}
    component_count = 0
    graph_connected = False
    equations_compatible = False
    obstruction_vertices: tuple[str, ...] = ()
    obstruction_edges: tuple[PlanarLCGaugeOverlapEdge, ...] = ()

    if overlap_schema_ok:
        adjacency: dict[str, list[tuple[str, PlanarLCGaugeOverlapEdge]]] = {
            chart_id: [] for chart_id in chart_ids
        }
        for edge in overlaps:
            adjacency[edge.source_chart_id].append((edge.target_chart_id, edge))
            adjacency[edge.target_chart_id].append((edge.source_chart_id, edge))
        for chart_id in chart_ids:
            adjacency[chart_id].sort(
                key=lambda item: (item[1].overlap_id, item[0])
            )

        for root in sorted(chart_ids):
            if root in assignment:
                continue
            component_count += 1
            assignment[root] = 0
            parent[root] = None
            queue: deque[str] = deque([root])
            while queue:
                current = queue.popleft()
                for neighbor, edge in adjacency[current]:
                    if neighbor in assignment:
                        continue
                    assignment[neighbor] = assignment[current] ^ edge.parity
                    parent[neighbor] = current
                    parent_edge[neighbor] = edge
                    queue.append(neighbor)

        graph_connected = component_count == 1
        equations_compatible = True
        for edge in sorted(overlaps, key=lambda item: item.overlap_id):
            if assignment[edge.source_chart_id] ^ assignment[edge.target_chart_id] != edge.parity:
                equations_compatible = False
                obstruction_vertices, obstruction_edges = _fundamental_cycle(
                    edge.source_chart_id,
                    edge.target_chart_id,
                    edge,
                    parent,
                    parent_edge,
                )
                break

    obligations = (
        PlanarLCGaugeGluingObligation(
            "lc_gauge_certificate_identity",
            identity_ok,
            (
                f"certificate_id={certificate_id!r}; "
                f"exact_type={exact_certificate_type}; source_ok={source_ok}"
            ),
        ),
        PlanarLCGaugeGluingObligation(
            "lc_gauge_chart_ids",
            chart_ids_ok,
            f"chart_ids={chart_ids!r}",
        ),
        PlanarLCGaugeGluingObligation(
            "lc_gauge_overlap_schema",
            overlap_schema_ok,
            f"overlap_count={len(overlaps) if type(overlaps) is tuple else 0}",
        ),
        PlanarLCGaugeGluingObligation(
            "lc_gauge_graph_connected",
            graph_connected,
            f"component_count={component_count}",
        ),
        PlanarLCGaugeGluingObligation(
            "lc_gauge_equations_compatible",
            equations_compatible,
            (
                "all supplied equations t_u XOR t_v = parity hold"
                if equations_compatible
                else "a malformed graph or odd-parity cycle prevents a global gauge"
            ),
        ),
    )
    return PlanarLCGaugeGluingCheckResult(
        certificate_id=certificate_id,
        checker_id="planar_lc_z2_gauge_gluing_checker_v1",
        obligations=obligations,
        chart_ids=tuple(chart_ids) if overlap_schema_ok else (),
        checked_overlaps=tuple(overlaps) if overlap_schema_ok else (),
        gauge_assignment=tuple(sorted(assignment.items())),
        component_count=component_count,
        obstruction_cycle_chart_ids=obstruction_vertices,
        obstruction_cycle_edges=obstruction_edges,
    )


__all__ = [
    "PlanarLCGaugeGluingCertificate",
    "PlanarLCGaugeGluingCheckResult",
    "PlanarLCGaugeGluingObligation",
    "PlanarLCGaugeOverlapEdge",
    "check_planar_lc_gauge_gluing",
]

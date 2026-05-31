import itertools

import numpy as np
import pytest

from three_body_symmetry.dynamics import (
    accelerations,
    angular_momentum_z,
    center_of_mass,
    energy,
    linear_momentum,
)
from three_body_symmetry.escape_endpoint import (
    construct_homothetic_escape_dyadic_recurrence,
    construct_homothetic_escape_endpoint_data,
    construct_scattering_endpoint_constants,
    construct_scattering_endpoint_dyadic_recurrence,
    construct_two_ended_scattering_atlas_recurrence,
    find_scattering_endpoint_tail_start,
    homothetic_escape_endpoint_implicit_residual,
    homothetic_escape_radius_at_time,
    homothetic_escape_time_integral,
)
from three_body_symmetry.global_invariants import (
    centered_angular_momentum_components,
    certify_nonzero_angular_momentum_excludes_triple_collision,
)
from three_body_symmetry.fuchsian import (
    FiniteFuchsianLogBranch,
    FiniteFuchsianLogContinuation,
    FuchsianLogBranch,
    FuchsianLogTerm,
    StableResonanceTerm,
    certify_finite_fuchsian_log_compact_time_isolation,
    certify_finite_fuchsian_log_total_collision_isolation,
    construct_finite_fuchsian_log_branch_from_stable_chain,
    derive_finite_fuchsian_log_branch_primitive_cauchy_inputs,
    construct_fuchsian_log_row_solution,
    construct_fuchsian_selector_continuation,
    construct_fuchsian_shape_branch,
    construct_stable_log_selector_chain,
    derive_identity_finite_fuchsian_log_continuation_from_incoming_branch,
    derive_identity_fuchsian_selector_continuation_from_incoming_branch,
)
from three_body_symmetry.event_recurrence import (
    derive_all_future_event_budget_from_primitive_cauchy_inputs,
    derive_all_future_event_budget_from_geometric_shell_isolation,
    derive_chart_family_counts_from_event_isolation,
    derive_geometric_shell_event_isolation,
)
from three_body_symmetry.event_regime_assembler import (
    certify_uniform_separated_binary_levi_civita_chart_family,
)
from three_body_symmetry.intervals import FloatInterval
from three_body_symmetry.obstructions import (
    certify_binary_degenerate_jacobi_kepler_reduction,
    certify_classical_integrals_do_not_determine_vector_field,
    construct_interval_jacobi_cluster_coordinates,
    construct_jacobi_cluster_coordinates,
    equilateral_zero_velocity_state,
    scalene_zero_velocity_state_with_unit_reciprocal_sum,
)
from three_body_symmetry.series import construct_taylor_solution
from three_body_symmetry.tail_bounds import sundman_cauchy_majorant_tail_certificate
from three_body_symmetry.triple_collision import (
    construct_homothetic_total_collision_branch,
    homothetic_energy_series_coefficients,
)
from three_body_symmetry.zero_angular_entry import (
    FiniteJetSelectorSpec,
    certify_finite_jet_identity_selector_entry,
    derive_finite_jet_identity_selector_entry_from_incoming,
    recover_finite_jet_selector_coordinates,
)


def _two_ended_scattering_middle_budgets(middle_charts):
    return {
        "value": sum(chart["value"] for chart in middle_charts),
        "first_jet": sum(chart["first_jet"] for chart in middle_charts),
        "residual": sum(chart["lifted_residual"] for chart in middle_charts),
        "physical_residual": sum(chart["physical_residual"] for chart in middle_charts),
    }


def _interval_vector_contains_point(interval_vector, point):
    return all(
        interval.lower <= float(value) <= interval.upper
        for interval, value in zip(interval_vector, point, strict=True)
    )


def _central_configuration_data(shape):
    if shape == "equilateral":
        configuration = np.array(
            [
                [1.0, 0.0],
                [-0.5, np.sqrt(3.0) / 2.0],
                [-0.5, -np.sqrt(3.0) / 2.0],
            ]
        )
        central_lambda = 1.0 / np.sqrt(3.0)
    elif shape == "euler":
        configuration = np.array(
            [
                [-1.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        )
        central_lambda = 5.0 / 4.0
    else:
        raise ValueError(shape)
    return configuration, central_lambda


def _arbitrary_mass_equilateral_central_configuration():
    masses = np.array([1.0, 0.7, 1.4])
    configuration, central_lambda = _mass_centered_equilateral_central_configuration(masses)
    return masses, configuration, central_lambda


def _mass_centered_equilateral_central_configuration(masses):
    masses = np.array(masses, dtype=float)
    configuration = np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ]
    )
    mass_center = np.average(configuration, axis=0, weights=masses)
    configuration = configuration - mass_center
    side_length = np.sqrt(3.0)
    central_lambda = float(np.sum(masses) / side_length**3)
    return configuration, central_lambda


def _symmetric_euler_central_configuration(middle_mass):
    masses = np.array([1.0, middle_mass, 1.0])
    configuration = np.array(
        [
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )
    central_lambda = middle_mass + 0.25
    return masses, configuration, central_lambda


def _ordered_euler_ratio(masses):
    m1, m2, m3 = masses
    polynomial = np.array(
        [
            m1 + m2,
            3.0 * m1 + 2.0 * m2,
            3.0 * m1 + m2,
            -(m2 + 3.0 * m3),
            -(2.0 * m2 + 3.0 * m3),
            -(m2 + m3),
        ]
    )
    roots = np.roots(polynomial)
    positive_roots = [
        float(root.real)
        for root in roots
        if abs(root.imag) < 1e-10 and root.real > 0.0
    ]
    if len(positive_roots) != 1:
        raise AssertionError(roots)
    return positive_roots[0], polynomial


def _ordered_euler_resonance_mass_ratios(resonance_power, ratio):
    r = ratio
    if resonance_power == 5:
        denominator = 3.0 * r**4 + 18.0 * r**3 + 35.0 * r**2 + 18.0 * r + 3.0
        left_ratio = -(
            (r + 1.0)
            * (3.0 * r**5 - 3.0 * r**4 - 7.0 * r**3 - 15.0 * r**2 - 15.0 * r - 5.0)
            / (r**2 * denominator)
        )
        right_ratio = (
            (r + 1.0)
            * (5.0 * r**5 + 15.0 * r**4 + 15.0 * r**3 + 7.0 * r**2 + 3.0 * r - 3.0)
            / denominator
        )
    elif resonance_power == 6:
        denominator = r**4 - 10.0 * r**3 - 31.0 * r**2 - 10.0 * r + 1.0
        left_ratio = -(
            (r + 1.0)
            * (r**5 + 15.0 * r**4 + 19.0 * r**3 + 27.0 * r**2 + 27.0 * r + 9.0)
            / (r**2 * denominator)
        )
        right_ratio = -(
            (r + 1.0)
            * (9.0 * r**5 + 27.0 * r**4 + 27.0 * r**3 + 19.0 * r**2 + 15.0 * r + 1.0)
            / denominator
        )
    elif resonance_power == 7:
        denominator = 3.0 * r**4 - 13.0 * r**2 + 3.0
        left_ratio = -(
            (r + 1.0)
            * (3.0 * r**5 + 15.0 * r**4 + 17.0 * r**3 + 21.0 * r**2 + 21.0 * r + 7.0)
            / (r**2 * denominator)
        )
        right_ratio = -(
            (r + 1.0)
            * (7.0 * r**5 + 21.0 * r**4 + 21.0 * r**3 + 17.0 * r**2 + 15.0 * r + 3.0)
            / denominator
        )
    else:
        raise ValueError(resonance_power)
    return left_ratio, right_ratio


def _ordered_euler_horizontal_shape_eigenvalue(masses, ratio):
    uncentered_line = np.array([0.0, 1.0, 1.0 + ratio])
    centered_line = uncentered_line - np.average(uncentered_line, weights=masses)
    unscaled_positions = np.column_stack([centered_line, np.zeros(3)])
    unscaled_acceleration = accelerations(unscaled_positions, masses)
    central_lambda = -float(
        np.sum(masses[:, None] * unscaled_positions * unscaled_acceleration)
        / np.sum(masses[:, None] * unscaled_positions * unscaled_positions)
    )
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    normalized_positions = scale_factor * unscaled_positions
    derivative_matrix = _linearized_acceleration_matrix(
        normalized_positions,
        masses,
    )
    horizontal_block = derivative_matrix[np.ix_([0, 2, 4], [0, 2, 4])]
    return float(np.trace(horizontal_block) - 4.0 / 9.0)


def _ordered_euler_scaled_configuration(masses):
    ratio, _polynomial = _ordered_euler_ratio(masses)
    uncentered_line = np.array([0.0, 1.0, 1.0 + ratio])
    centered_line = uncentered_line - np.average(uncentered_line, weights=masses)
    unscaled_positions = np.column_stack([centered_line, np.zeros(3)])
    unscaled_acceleration = accelerations(unscaled_positions, masses)
    central_lambda = -float(
        np.sum(masses[:, None] * unscaled_positions * unscaled_acceleration)
        / np.sum(masses[:, None] * unscaled_positions * unscaled_positions)
    )
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    return ratio, scale_factor * unscaled_positions


def _linearized_acceleration_matrix(positions, masses):
    matrix = np.zeros((positions.size, positions.size))
    for coordinate_index in range(positions.size):
        perturbation = np.zeros_like(positions, dtype=float)
        perturbation.reshape(-1)[coordinate_index] = 1.0
        matrix[:, coordinate_index] = _acceleration_derivative_apply(
            positions,
            masses,
            perturbation,
        ).reshape(-1)
    return matrix


def _parabolic_homothetic_state(shape, time):
    configuration, central_lambda = _central_configuration_data(shape)
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    radius = scale_factor * abs(time) ** (2.0 / 3.0)
    radius_velocity = (
        np.sign(time) * (2.0 / 3.0) * scale_factor * abs(time) ** (-1.0 / 3.0)
    )
    positions = radius * configuration
    velocities = radius_velocity * configuration
    return positions, velocities, central_lambda, scale_factor


def _cubic_time_acceleration(quadratic_coefficient, cubic_coefficient, quartic_coefficient, tau):
    return (
        -(2.0 / 9.0) * quadratic_coefficient * tau**-4
        + (4.0 / 9.0) * quartic_coefficient * tau**-2
    )


def _cubic_time_polynomial_acceleration(coefficients_by_power, tau):
    acceleration = np.zeros_like(next(iter(coefficients_by_power.values())), dtype=float)
    for power, coefficient in coefficients_by_power.items():
        acceleration += (
            power * (power - 3.0) / 9.0
        ) * coefficient * tau ** (power - 6)
    return acceleration


def _shape_series_state(shape_coefficients, tau):
    positions = sum(
        coefficient * tau ** (degree + 2)
        for degree, coefficient in enumerate(shape_coefficients)
    )
    velocities = sum(
        ((degree + 2.0) / 3.0) * coefficient * tau ** (degree - 1)
        for degree, coefficient in enumerate(shape_coefficients)
    )
    return positions, velocities


def _mass_inner_product(masses, left, right):
    return float(np.sum(masses[:, None] * left * right))


def _mass_orthonormal_complement_basis(masses, central_shape, count):
    basis = []
    central_norm = _mass_inner_product(masses, central_shape, central_shape)
    for flat_index in range(central_shape.size):
        candidate = np.zeros_like(central_shape)
        candidate.reshape(-1)[flat_index] = 1.0
        candidate -= (
            _mass_inner_product(masses, candidate, central_shape)
            / central_norm
        ) * central_shape
        for prior in basis:
            candidate -= _mass_inner_product(masses, candidate, prior) * prior
        norm = np.sqrt(_mass_inner_product(masses, candidate, candidate))
        if norm > 1e-10:
            basis.append(candidate / norm)
        if len(basis) == count:
            return tuple(basis)
    raise AssertionError("not enough complement directions")


def _shape_branch_energy_limit(masses, shape_coefficients):
    central_shape = shape_coefficients[0]
    cubic_shape = shape_coefficients[1]
    quartic_shape = shape_coefficients[2]
    linearized_cubic = _acceleration_derivative_apply(
        central_shape,
        masses,
        cubic_shape,
    )
    return (
        0.5 * _mass_inner_product(masses, cubic_shape, cubic_shape)
        - 0.5 * _mass_inner_product(masses, cubic_shape, linearized_cubic)
        + (10.0 / 9.0) * _mass_inner_product(masses, central_shape, quartic_shape)
    )


def _energy_matching_quartic_scale(target_energy, masses, base_shape_coefficients):
    central_shape = base_shape_coefficients[0]
    base_energy = _shape_branch_energy_limit(masses, base_shape_coefficients)
    inertia = _mass_inner_product(masses, central_shape, central_shape)
    return (9.0 / (10.0 * inertia)) * (target_energy - base_energy)


def _equilateral_centered_cubic_kernel(masses, central_shape):
    derivative_matrix = _linearized_acceleration_matrix(central_shape, masses)
    _left_vectors, _singular_values, nullspace_vh = np.linalg.svd(derivative_matrix)
    nullspace = nullspace_vh.T[:, -3:]
    mass_center_constraints = np.zeros((2, 6))
    mass_center_constraints[0, 0::2] = masses
    mass_center_constraints[1, 1::2] = masses
    _constraint_left, _constraint_singular_values, constraint_vh = np.linalg.svd(
        mass_center_constraints @ nullspace
    )
    centered_null_coefficients = constraint_vh.T[:, -1]
    cubic_kernel = (nullspace @ centered_null_coefficients).reshape(3, 2)
    return cubic_kernel / np.sqrt(
        _mass_inner_product(masses, cubic_kernel, cubic_kernel)
    )


def _homothetic_energy_series_coefficients(energy_per_inertia, order=3):
    return homothetic_energy_series_coefficients(energy_per_inertia, order)


def _series_product(left, right, order):
    coefficients = np.zeros(order + 1)
    for degree in range(order + 1):
        coefficients[degree] = sum(left[k] * right[degree - k] for k in range(degree + 1))
    return coefficients


def _series_inverse(series, order):
    inverse = np.zeros(order + 1)
    inverse[0] = 1.0 / series[0]
    for degree in range(1, order + 1):
        inverse[degree] = -sum(series[k] * inverse[degree - k] for k in range(1, degree + 1)) / series[0]
    return inverse


def _series_power_one_plus(series, exponent, order):
    result = np.zeros(order + 1)
    result[0] = 1.0
    term = np.zeros(order + 1)
    term[0] = 1.0
    binomial_coefficient = 1.0
    for power in range(1, order + 1):
        term = _series_product(term, series, order)
        binomial_coefficient *= (exponent - power + 1.0) / power
        result += binomial_coefficient * term
    return result


def _shape_acceleration_series(shape_coefficients, masses, order):
    acceleration_coefficients = np.zeros((order + 1, 3, 2))
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            relative_coefficients = shape_coefficients[:, j, :] - shape_coefficients[:, i, :]
            squared_distance = np.zeros(order + 1)
            for degree in range(order + 1):
                squared_distance[degree] = sum(
                    float(
                        np.dot(
                            relative_coefficients[left_degree],
                            relative_coefficients[degree - left_degree],
                        )
                    )
                    for left_degree in range(degree + 1)
                )
            normalized_tail = np.zeros(order + 1)
            normalized_tail[1:] = squared_distance[1:] / squared_distance[0]
            inverse_distance_cubed = squared_distance[0] ** (-1.5) * _series_power_one_plus(
                normalized_tail,
                -1.5,
                order,
            )
            pair_force = np.zeros((order + 1, 2))
            for degree in range(order + 1):
                pair_force[degree] = sum(
                    relative_coefficients[left_degree]
                    * inverse_distance_cubed[degree - left_degree]
                    for left_degree in range(degree + 1)
                )
            acceleration_coefficients[:, i, :] += masses[j] * pair_force
    return acceleration_coefficients


def _multi_indices_with_total(total_degree, dimension):
    if dimension == 1:
        return [(total_degree,)]
    indices = []
    for first in range(total_degree + 1):
        for tail in _multi_indices_with_total(total_degree - first, dimension - 1):
            indices.append((first, *tail))
    return indices


def _bounded_multi_indices(maximum):
    return list(itertools.product(*(range(limit + 1) for limit in maximum)))


def _add_multi_index(left, right):
    return tuple(left_value + right_value for left_value, right_value in zip(left, right))


def _sub_multi_index(left, right):
    return tuple(left_value - right_value for left_value, right_value in zip(left, right))


def _multi_index_leq(left, right):
    return all(left_value <= right_value for left_value, right_value in zip(left, right))


def _scalar_multi_series_product(left, right, maximum):
    product = {}
    for left_index, left_value in left.items():
        for right_index, right_value in right.items():
            combined = _add_multi_index(left_index, right_index)
            if _multi_index_leq(combined, maximum):
                product[combined] = product.get(combined, 0.0) + (
                    left_value * right_value
                )
    return product


def _scalar_multi_series_power_one_plus(tail, exponent, maximum):
    zero_index = tuple(0 for _ in maximum)
    result = {zero_index: 1.0}
    term = {zero_index: 1.0}
    binomial_coefficient = 1.0
    for power in range(1, sum(maximum) + 1):
        term = _scalar_multi_series_product(term, tail, maximum)
        if not term:
            break
        binomial_coefficient *= (exponent - power + 1.0) / power
        for index, value in term.items():
            result[index] = result.get(index, 0.0) + binomial_coefficient * value
    return result


def _multivariate_acceleration_coefficient(shape_coefficients, masses, target):
    zero_shape = np.zeros_like(next(iter(shape_coefficients.values())))
    zero_index = tuple(0 for _ in target)
    acceleration_coefficient = np.zeros_like(zero_shape)
    indices = _bounded_multi_indices(target)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            relative = {
                index: (
                    shape_coefficients.get(index, zero_shape)[j]
                    - shape_coefficients.get(index, zero_shape)[i]
                )
                for index in indices
            }
            squared_distance = {}
            for index in indices:
                squared_distance[index] = sum(
                    float(
                        np.dot(
                            relative[left],
                            relative[_sub_multi_index(index, left)],
                        )
                    )
                    for left in indices
                    if _multi_index_leq(left, index)
                )
            normalized_tail = {
                index: value / squared_distance[zero_index]
                for index, value in squared_distance.items()
                if index != zero_index
            }
            inverse_distance_cubed = _scalar_multi_series_power_one_plus(
                normalized_tail,
                -1.5,
                target,
            )
            inverse_distance_cubed = {
                index: squared_distance[zero_index] ** (-1.5) * value
                for index, value in inverse_distance_cubed.items()
            }
            pair_force = sum(
                relative[left]
                * inverse_distance_cubed.get(_sub_multi_index(target, left), 0.0)
                for left in indices
                if _multi_index_leq(left, target)
            )
            acceleration_coefficient[i] += masses[j] * pair_force
    return acceleration_coefficient


def _resonant_equilateral_shape_coefficients(
    order,
    cubic_amplitude,
    quartic_scale,
    masses=None,
):
    if masses is None:
        masses = np.array([1.0, 1.0, 2.5])
    else:
        masses = np.array(masses, dtype=float)
    configuration, central_lambda = _mass_centered_equilateral_central_configuration(masses)
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    quadratic_coefficient = scale_factor * configuration
    derivative_matrix = _linearized_acceleration_matrix(quadratic_coefficient, masses)
    _left_vectors, _singular_values, nullspace_vh = np.linalg.svd(derivative_matrix)
    nullspace = nullspace_vh.T[:, -3:]
    mass_center_constraints = np.zeros((2, 6))
    mass_center_constraints[0, 0::2] = masses
    mass_center_constraints[1, 1::2] = masses
    _constraint_left, _constraint_singular_values, constraint_vh = np.linalg.svd(
        mass_center_constraints @ nullspace
    )
    centered_null_coefficients = constraint_vh.T[:, -1]
    cubic_kernel = (nullspace @ centered_null_coefficients).reshape(3, 2)
    cubic_kernel = cubic_kernel / np.sqrt(
        np.sum(masses[:, None] * cubic_kernel * cubic_kernel)
    )
    coefficients = np.zeros((order + 1, 3, 2))
    coefficients[0] = quadratic_coefficient
    coefficients[1] = cubic_amplitude * cubic_kernel
    for degree in range(2, order + 1):
        trial_coefficients = coefficients.copy()
        trial_coefficients[degree] = 0.0
        known_term = _shape_acceleration_series(
            trial_coefficients,
            masses,
            degree,
        )[degree].reshape(-1)
        multiplier = (degree + 2.0) * (degree - 1.0) / 9.0
        operator = multiplier * np.eye(6) - derivative_matrix
        solved, *_ = np.linalg.lstsq(operator, known_term, rcond=None)
        coefficients[degree] = solved.reshape(3, 2)
        if degree == 2:
            coefficients[degree] += quartic_scale * quadratic_coefficient
    return masses, coefficients


def _symmetric_euler_resonant_shape_coefficients(
    resonance_power,
    order,
    resonance_amplitude,
    quartic_scale,
):
    resonance_multiplier = resonance_power * (resonance_power - 3.0)
    middle_mass = (32.0 - resonance_multiplier) / (4.0 * (resonance_multiplier - 4.0))
    masses, configuration, central_lambda = _symmetric_euler_central_configuration(
        middle_mass
    )
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    quadratic_coefficient = scale_factor * configuration
    horizontal_shape_mode = np.column_stack(
        [
            np.array([1.0, -2.0 / middle_mass, 1.0]),
            np.zeros(3),
        ]
    )
    horizontal_shape_mode = horizontal_shape_mode / np.sqrt(
        np.sum(masses[:, None] * horizontal_shape_mode * horizontal_shape_mode)
    )
    derivative_matrix = _linearized_acceleration_matrix(quadratic_coefficient, masses)
    coefficients = np.zeros((order + 1, 3, 2))
    coefficients[0] = quadratic_coefficient
    resonance_degree = resonance_power - 2
    for degree in range(1, order + 1):
        trial_coefficients = coefficients.copy()
        trial_coefficients[degree] = 0.0
        known_term = _shape_acceleration_series(
            trial_coefficients,
            masses,
            degree,
        )[degree].reshape(-1)
        multiplier = (degree + 2.0) * (degree - 1.0) / 9.0
        operator = multiplier * np.eye(6) - derivative_matrix
        solved, *_ = np.linalg.lstsq(operator, known_term, rcond=None)
        coefficients[degree] = solved.reshape(3, 2)
        if degree == 1:
            coefficients[degree] = 0.0
        elif degree == 2:
            coefficients[degree] += quartic_scale * quadratic_coefficient
        elif degree == resonance_degree:
            coefficients[degree] += resonance_amplitude * horizontal_shape_mode
    return masses, coefficients


def _ordered_euler_resonant_shape_coefficients(
    resonance_power,
    ratio,
    order,
    resonance_amplitude,
    quartic_scale,
):
    left_ratio, right_ratio = _ordered_euler_resonance_mass_ratios(
        resonance_power,
        ratio,
    )
    masses = np.array([left_ratio, 1.0, right_ratio])
    uncentered_line = np.array([0.0, 1.0, 1.0 + ratio])
    centered_line = uncentered_line - np.average(uncentered_line, weights=masses)
    configuration = np.column_stack([centered_line, np.zeros(3)])
    unscaled_acceleration = accelerations(configuration, masses)
    central_lambda = -float(
        np.sum(masses[:, None] * configuration * unscaled_acceleration)
        / np.sum(masses[:, None] * configuration * configuration)
    )
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    quadratic_coefficient = scale_factor * configuration
    derivative_matrix = _linearized_acceleration_matrix(quadratic_coefficient, masses)
    horizontal_block = derivative_matrix[np.ix_([0, 2, 4], [0, 2, 4])]
    resonance_multiplier = resonance_power * (resonance_power - 3.0) / 9.0
    eigenvalues, eigenvectors = np.linalg.eig(horizontal_block)
    resonance_index = int(
        np.argmin(np.abs(eigenvalues.real - resonance_multiplier) + np.abs(eigenvalues.imag))
    )
    horizontal_shape_mode = np.column_stack(
        [
            eigenvectors[:, resonance_index].real,
            np.zeros(3),
        ]
    )
    horizontal_shape_mode[:, 0] -= np.average(
        horizontal_shape_mode[:, 0],
        weights=masses,
    )
    horizontal_shape_mode = horizontal_shape_mode / np.sqrt(
        np.sum(masses[:, None] * horizontal_shape_mode * horizontal_shape_mode)
    )
    coefficients = np.zeros((order + 1, 3, 2))
    coefficients[0] = quadratic_coefficient
    resonance_degree = resonance_power - 2
    for degree in range(1, order + 1):
        trial_coefficients = coefficients.copy()
        trial_coefficients[degree] = 0.0
        known_term = _shape_acceleration_series(
            trial_coefficients,
            masses,
            degree,
        )[degree].reshape(-1)
        multiplier = (degree + 2.0) * (degree - 1.0) / 9.0
        operator = multiplier * np.eye(6) - derivative_matrix
        solved, *_ = np.linalg.lstsq(operator, known_term, rcond=None)
        coefficients[degree] = solved.reshape(3, 2)
        if degree == 1:
            coefficients[degree] = 0.0
        elif degree == 2:
            coefficients[degree] += quartic_scale * quadratic_coefficient
        elif degree == resonance_degree:
            coefficients[degree] += resonance_amplitude * horizontal_shape_mode
    return masses, horizontal_shape_mode, coefficients


def _first_resonant_symmetric_euler_shape_coefficients(order, resonance_amplitude, quartic_scale):
    return _symmetric_euler_resonant_shape_coefficients(
        resonance_power=5,
        order=order,
        resonance_amplitude=resonance_amplitude,
        quartic_scale=quartic_scale,
    )


def _acceleration_derivative_apply(positions, masses, perturbation):
    derivative = np.zeros_like(positions, dtype=float)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            relative_position = positions[j] - positions[i]
            relative_perturbation = perturbation[j] - perturbation[i]
            distance = float(np.linalg.norm(relative_position))
            derivative[i] += masses[j] * (
                relative_perturbation / distance**3
                - 3.0
                * relative_position
                * np.dot(relative_position, relative_perturbation)
                / distance**5
            )
    return derivative


def _acceleration_second_order_coefficient(positions, masses, perturbation):
    coefficient = np.zeros_like(positions, dtype=float)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            relative_position = positions[j] - positions[i]
            relative_perturbation = perturbation[j] - perturbation[i]
            distance = float(np.linalg.norm(relative_position))
            radial_velocity = float(np.dot(relative_position, relative_perturbation))
            perturbation_norm_squared = float(np.dot(relative_perturbation, relative_perturbation))
            coefficient[i] += masses[j] * (
                -3.0
                * relative_perturbation
                * radial_velocity
                / distance**5
                + relative_position
                * (
                    -1.5 * perturbation_norm_squared / distance**5
                    + 7.5 * radial_velocity**2 / distance**7
                )
            )
    return coefficient


def _tau_second_derivative_log_monomial_coefficients(power, log_power):
    coefficients = {
        log_power: power * (power - 1.0),
    }
    if log_power >= 1:
        coefficients[log_power - 1] = coefficients.get(log_power - 1, 0.0) + log_power * (
            2.0 * power - 1.0
        )
    if log_power >= 2:
        coefficients[log_power - 2] = coefficients.get(log_power - 2, 0.0) + log_power * (
            log_power - 1.0
        )
    return coefficients


def _solve_log_polynomial_row(power, forcing_by_log_power):
    solved = {}
    highest_log_power = max(forcing_by_log_power)
    divisor = power * (power - 1.0)
    for log_power in range(highest_log_power, -1, -1):
        carry = (
            (log_power + 1.0) * (2.0 * power - 1.0) * solved.get(log_power + 1, 0.0)
            + (log_power + 2.0)
            * (log_power + 1.0)
            * solved.get(log_power + 2, 0.0)
        )
        solved[log_power] = (forcing_by_log_power.get(log_power, 0.0) - carry) / divisor
    return solved


def _equilateral_state(side_length):
    return equilateral_zero_velocity_state(side_length)


def _scalene_state_with_unit_reciprocal_sum():
    return scalene_zero_velocity_state_with_unit_reciprocal_sum()


def test_classical_integrals_do_not_determine_the_vector_field():
    equilateral = _equilateral_state(1.0)
    scalene = _scalene_state_with_unit_reciprocal_sum()

    assert np.linalg.norm(center_of_mass(equilateral), ord=np.inf) < 1e-14
    assert np.linalg.norm(center_of_mass(scalene), ord=np.inf) < 1e-14
    assert np.linalg.norm(linear_momentum(equilateral), ord=np.inf) == 0.0
    assert np.linalg.norm(linear_momentum(scalene), ord=np.inf) == 0.0
    assert angular_momentum_z(equilateral) == 0.0
    assert angular_momentum_z(scalene) == 0.0
    assert abs(energy(equilateral) - energy(scalene)) < 1e-12

    equilateral_acceleration_norms = np.sort(np.linalg.norm(accelerations(equilateral[:6].reshape(3, 2)), axis=1))
    scalene_acceleration_norms = np.sort(np.linalg.norm(accelerations(scalene[:6].reshape(3, 2)), axis=1))

    assert np.linalg.norm(equilateral_acceleration_norms - scalene_acceleration_norms, ord=np.inf) > 0.2


def test_classical_integral_vector_field_separation_certificate():
    certificate = certify_classical_integrals_do_not_determine_vector_field()

    assert certificate.certified
    assert certificate.classical_integrals_match
    assert certificate.vector_field_separated
    assert certificate.reason == "classical integrals match but the Newtonian vector field is different"
    gaps = dict(certificate.classical_integral_gaps)
    assert tuple(gaps) == ("center_of_mass", "linear_momentum", "angular_momentum", "energy")
    assert all(gap <= certificate.integral_tolerance for gap in gaps.values())
    assert gaps["energy"] > 0.0
    assert certificate.acceleration_signature_gap > 0.2


def _rotating_triangle_data():
    masses = np.array([1.0, 1.3, 0.8])
    positions = np.array(
        [
            [0.9, -0.2],
            [-0.4, 0.7],
            [0.1, -0.8],
        ]
    )
    total_mass = np.sum(masses)
    center = np.sum(masses[:, None] * positions, axis=0) / total_mass
    centered = positions - center
    angular_speed = 0.4
    translation_velocity = np.array([0.12, -0.07])
    velocities = translation_velocity + angular_speed * np.column_stack((-centered[:, 1], centered[:, 0]))
    return masses, positions, velocities


def test_centered_angular_momentum_is_translation_invariant():
    masses, positions, velocities = _rotating_triangle_data()
    position_shift = np.array([3.0, -2.0])
    velocity_shift = np.array([-0.4, 0.9])

    original = centered_angular_momentum_components(positions, velocities, masses)
    shifted = centered_angular_momentum_components(
        positions + position_shift,
        velocities + velocity_shift,
        masses,
    )

    assert original.shape == (1,)
    assert original[0] > 0.0
    assert np.linalg.norm(original - shifted, ord=np.inf) < 1e-14


def test_nonzero_angular_momentum_certifies_triple_collision_exclusion_for_interval_box():
    masses, positions, velocities = _rotating_triangle_data()
    point_components = centered_angular_momentum_components(positions, velocities, masses)
    position_intervals = np.empty(positions.shape, dtype=object)
    velocity_intervals = np.empty(velocities.shape, dtype=object)
    for index in np.ndindex(positions.shape):
        position_intervals[index] = FloatInterval(positions[index] - 1e-6, positions[index] + 1e-6)
        velocity_intervals[index] = FloatInterval(velocities[index] - 1e-6, velocities[index] + 1e-6)

    certificate = certify_nonzero_angular_momentum_excludes_triple_collision(
        position_intervals,
        velocity_intervals,
        masses,
    )

    assert certificate.certified
    assert certificate.status == "excluded"
    assert not certificate.undecided
    assert certificate.angular_momentum_components[0].lower <= point_components[0]
    assert point_components[0] <= certificate.angular_momentum_components[0].upper
    assert certificate.angular_momentum_norm_squared_lower_bound > 0.0
    assert certificate.reason == "centered angular momentum is bounded away from zero"


def test_zero_angular_momentum_leaves_triple_collision_exclusion_uncertified():
    masses = np.array([1.0, 1.0, 1.0])
    positions = np.array(
        [
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )
    velocities = np.zeros_like(positions)

    certificate = certify_nonzero_angular_momentum_excludes_triple_collision(
        positions,
        velocities,
        masses,
    )

    assert not certificate.certified
    assert certificate.status == "undecided"
    assert certificate.undecided
    assert certificate.angular_momentum_norm_squared_lower_bound == 0.0
    assert certificate.reason == "centered angular momentum interval contains zero"


def test_regularized_total_collision_germ_forces_zero_angular_momentum_limit():
    masses, separated_positions, rotating_velocities = _rotating_triangle_data()
    exclusion = certify_nonzero_angular_momentum_excludes_triple_collision(
        separated_positions,
        rotating_velocities,
        masses,
    )
    assert exclusion.certified

    central_shape = separated_positions - np.average(
        separated_positions,
        axis=0,
        weights=masses,
    )
    cubic_shape = np.array(
        [
            [0.08, -0.03],
            [-0.04, 0.05],
            [0.02, -0.01],
        ]
    )
    cubic_shape -= np.average(cubic_shape, axis=0, weights=masses)
    quartic_shape = np.array(
        [
            [0.03, 0.01],
            [-0.02, 0.04],
            [0.01, -0.02],
        ]
    )
    quartic_shape -= np.average(quartic_shape, axis=0, weights=masses)

    angular_norms = []
    sundman_products = []
    for regularized_time in (1e-2, 5e-3, 2.5e-3):
        shape = (
            central_shape
            + cubic_shape * regularized_time
            + quartic_shape * regularized_time**2
        )
        shape_derivative = cubic_shape + 2.0 * quartic_shape * regularized_time
        positions = regularized_time**2 * shape
        velocities = (2.0 / (3.0 * regularized_time)) * shape + (
            1.0 / 3.0
        ) * shape_derivative
        centered_L = centered_angular_momentum_components(positions, velocities, masses)
        moment_of_inertia = float(np.sum(masses[:, None] * positions**2))
        kinetic = 0.5 * float(np.sum(masses[:, None] * velocities**2))
        angular_norm_squared = float(np.dot(centered_L, centered_L))

        angular_norms.append(np.sqrt(angular_norm_squared))
        sundman_products.append(2.0 * moment_of_inertia * kinetic)
        assert angular_norm_squared <= 2.0 * moment_of_inertia * kinetic

    assert angular_norms[1] < 0.26 * angular_norms[0]
    assert angular_norms[2] < 0.26 * angular_norms[1]
    assert sundman_products[1] < 0.26 * sundman_products[0]
    assert sundman_products[2] < 0.26 * sundman_products[1]
    assert angular_norms[-1] ** 2 < (
        1e-4 * exclusion.angular_momentum_norm_squared_lower_bound
    )


def test_sundman_normalized_potential_bound_excludes_nonzero_angular_total_collapse():
    masses, separated_positions, rotating_velocities = _rotating_triangle_data()
    exclusion = certify_nonzero_angular_momentum_excludes_triple_collision(
        separated_positions,
        rotating_velocities,
        masses,
    )
    assert exclusion.certified

    centered_shape = separated_positions - np.average(
        separated_positions,
        axis=0,
        weights=masses,
    )
    shape_inertia = float(np.sum(masses[:, None] * centered_shape**2))
    shape_potential = 0.0
    for first in range(3):
        for second in range(first + 1, 3):
            shape_potential += masses[first] * masses[second] / np.linalg.norm(
                centered_shape[first] - centered_shape[second]
            )
    normalized_potential_bound = shape_potential * np.sqrt(shape_inertia)
    finite_energy = 0.37

    assert shape_inertia > 0.0
    assert normalized_potential_bound < 10.0

    sundman_upper_bounds = []
    normalized_potentials = []
    for collapse_scale in (1e-2, 5e-3, 2.5e-3):
        inertia = collapse_scale**2 * shape_inertia
        potential = shape_potential / collapse_scale
        kinetic = finite_energy + potential
        sundman_upper_bounds.append(2.0 * inertia * kinetic)
        normalized_potentials.append(potential * np.sqrt(inertia))

    np.testing.assert_allclose(
        normalized_potentials,
        [normalized_potential_bound] * len(normalized_potentials),
        rtol=1e-14,
        atol=1e-14,
    )
    assert sundman_upper_bounds[1] < 0.51 * sundman_upper_bounds[0]
    assert sundman_upper_bounds[2] < 0.51 * sundman_upper_bounds[1]
    assert sundman_upper_bounds[-1] < (
        0.25 * exclusion.angular_momentum_norm_squared_lower_bound
    )


def test_shape_compact_total_collapse_supplies_normalized_potential_bound():
    masses, separated_positions, _velocities = _rotating_triangle_data()
    centered_shape = separated_positions - np.average(
        separated_positions,
        axis=0,
        weights=masses,
    )
    shape_inertia = float(np.sum(masses[:, None] * centered_shape**2))
    normalized_shape = centered_shape / np.sqrt(shape_inertia)
    normalized_pair_floor = min(
        np.linalg.norm(normalized_shape[first] - normalized_shape[second])
        for first in range(3)
        for second in range(first + 1, 3)
    )
    mass_pair_sum = sum(
        masses[first] * masses[second]
        for first in range(3)
        for second in range(first + 1, 3)
    )
    compact_shape_bound = mass_pair_sum / normalized_pair_floor

    assert normalized_pair_floor > 0.5
    for collapse_scale in (1e-2, 1e-3, 1e-4):
        positions = collapse_scale * normalized_shape
        inertia = float(np.sum(masses[:, None] * positions**2))
        potential = 0.0
        pair_floor = np.inf
        for first in range(3):
            for second in range(first + 1, 3):
                pair_distance = np.linalg.norm(positions[first] - positions[second])
                pair_floor = min(pair_floor, pair_distance / np.sqrt(inertia))
                potential += masses[first] * masses[second] / pair_distance

        normalized_potential = potential * np.sqrt(inertia)
        assert pair_floor == pytest.approx(normalized_pair_floor)
        assert normalized_potential <= compact_shape_bound
        assert normalized_potential == pytest.approx(
            sum(
                masses[first]
                * masses[second]
                / np.linalg.norm(normalized_shape[first] - normalized_shape[second])
                for first in range(3)
                for second in range(first + 1, 3)
            )
        )


def test_binary_degenerate_total_collapse_forces_tight_pair_angular_bound_to_zero():
    masses = np.array([1.0, 0.7, 1.4])
    pair_mass = masses[0] + masses[1]
    reduced_pair_mass = masses[0] * masses[1] / pair_mass
    finite_energy = 0.37
    angular_bounds = []
    normalized_potentials = []

    for cluster_scale in (1e-2, 5e-3, 2.5e-3):
        pair_scale = cluster_scale**2
        pair_center = np.array([0.0, 0.0])
        third_offset = np.array([cluster_scale, 0.2 * cluster_scale])
        positions = np.array(
            [
                pair_center - (masses[1] / pair_mass) * np.array([pair_scale, 0.0]),
                pair_center + (masses[0] / pair_mass) * np.array([pair_scale, 0.0]),
                third_offset,
            ]
        )
        positions -= np.average(positions, axis=0, weights=masses)

        tight_pair = positions[1] - positions[0]
        cluster_third = positions[2] - (
            masses[0] * positions[0] + masses[1] * positions[1]
        ) / pair_mass
        potential = 0.0
        for first in range(3):
            for second in range(first + 1, 3):
                potential += masses[first] * masses[second] / np.linalg.norm(
                    positions[first] - positions[second]
                )
        kinetic_bound = finite_energy + potential
        inertia = float(np.sum(masses[:, None] * positions**2))
        tight_pair_angular_bound = np.sqrt(
            2.0 * reduced_pair_mass * np.linalg.norm(tight_pair) ** 2 * kinetic_bound
        )

        assert np.linalg.norm(tight_pair) / np.linalg.norm(cluster_third) < 0.02
        angular_bounds.append(tight_pair_angular_bound)
        normalized_potentials.append(potential * np.sqrt(inertia))

    assert normalized_potentials[1] > 1.9 * normalized_potentials[0]
    assert normalized_potentials[2] > 1.9 * normalized_potentials[1]
    assert angular_bounds[1] < 0.72 * angular_bounds[0]
    assert angular_bounds[2] < 0.72 * angular_bounds[1]
    assert angular_bounds[-1] < 0.08


def test_jacobi_cluster_coordinates_reconstruct_and_decompose_angular_momentum():
    masses = np.array([1.0, 0.7, 1.4])
    positions = np.array(
        [
            [-0.04, 0.02, 0.01],
            [0.03, -0.01, 0.02],
            [0.2, 0.1, -0.03],
        ]
    )
    velocities = np.array(
        [
            [0.3, -0.4, 0.2],
            [-0.2, 0.5, -0.1],
            [0.1, -0.05, 0.4],
        ]
    )

    certificate = construct_jacobi_cluster_coordinates(
        positions,
        velocities,
        masses,
        pair=(0, 1),
    )

    assert certificate.certified
    assert certificate.reconstruction_error < 1.0e-14
    assert certificate.velocity_reconstruction_error < 1.0e-14
    assert certificate.angular_decomposition_error < 1.0e-14
    assert certificate.outer_reduced_mass == pytest.approx(
        (masses[0] + masses[1]) * masses[2] / np.sum(masses)
    )


def test_interval_jacobi_cluster_coordinates_enclose_box_and_midpoint_lift():
    masses = np.array([1.0, 0.7, 1.4])
    midpoint_positions = np.array(
        [
            [-0.04, 0.02, 0.01],
            [0.03, -0.01, 0.02],
            [0.2, 0.1, -0.03],
        ]
    )
    midpoint_velocities = np.array(
        [
            [0.3, -0.4, 0.2],
            [-0.2, 0.5, -0.1],
            [0.1, -0.05, 0.4],
        ]
    )
    widths = np.array(
        [
            [1.0e-3, 2.0e-3, 1.5e-3],
            [1.5e-3, 1.0e-3, 2.0e-3],
            [2.0e-3, 1.5e-3, 1.0e-3],
        ]
    )
    position_intervals = np.stack(
        (midpoint_positions - widths, midpoint_positions + widths),
        axis=-1,
    )
    velocity_intervals = np.stack(
        (midpoint_velocities - 0.5 * widths, midpoint_velocities + 0.5 * widths),
        axis=-1,
    )

    point_certificate = construct_jacobi_cluster_coordinates(
        midpoint_positions,
        midpoint_velocities,
        masses,
        pair=(0, 1),
    )
    interval_certificate = construct_interval_jacobi_cluster_coordinates(
        position_intervals,
        velocity_intervals,
        masses,
        pair=(0, 1),
    )

    assert interval_certificate.certified
    assert interval_certificate.reconstruction_containment_certified
    assert interval_certificate.velocity_reconstruction_containment_certified
    assert interval_certificate.absolute_reconstruction_containment_certified
    assert interval_certificate.absolute_velocity_reconstruction_containment_certified
    assert interval_certificate.angular_decomposition_residual_contains_zero
    assert _interval_vector_contains_point(
        interval_certificate.relative_pair,
        point_certificate.relative_pair,
    )
    assert _interval_vector_contains_point(
        interval_certificate.cluster_third,
        point_certificate.cluster_third,
    )
    assert _interval_vector_contains_point(
        interval_certificate.relative_pair_velocity,
        point_certificate.relative_pair_velocity,
    )
    assert _interval_vector_contains_point(
        interval_certificate.cluster_third_velocity,
        point_certificate.cluster_third_velocity,
    )


def test_binary_degenerate_outer_angular_barrier_excludes_nonzero_total_collapse():
    masses = np.array([1.0, 0.7, 1.4])
    total_mass = float(np.sum(masses))
    pair_mass = masses[0] + masses[1]
    reduced_pair_mass = masses[0] * masses[1] / pair_mass
    reduced_outer_mass = pair_mass * masses[2] / total_mass
    finite_energy = 0.37
    target_outer_angular = 0.4
    barrier_constant = 0.5 * target_outer_angular**2 / reduced_outer_mass**2
    barrier_energies = []
    radial_acceleration_scales = []

    for cluster_scale in (1e-2, 5e-3, 2.5e-3):
        pair_scale = cluster_scale**2
        relative_pair = np.array([pair_scale, 0.0])
        cluster_third = np.array([cluster_scale, 0.0])
        outer_tangential_speed = target_outer_angular / (
            reduced_outer_mass * cluster_scale
        )
        cluster_third_velocity = np.array([-1.0, outer_tangential_speed])

        cluster_center = -(masses[2] / total_mass) * cluster_third
        third_position = (pair_mass / total_mass) * cluster_third
        positions = np.array(
            [
                cluster_center - (masses[1] / pair_mass) * relative_pair,
                cluster_center + (masses[0] / pair_mass) * relative_pair,
                third_position,
            ]
        )

        potential = 0.0
        for first in range(3):
            for second in range(first + 1, 3):
                potential += masses[first] * masses[second] / np.linalg.norm(
                    positions[first] - positions[second]
                )
        outer_kinetic = 0.5 * reduced_outer_mass * float(
            np.dot(cluster_third_velocity, cluster_third_velocity)
        )
        pair_kinetic = finite_energy + potential - outer_kinetic
        assert pair_kinetic > 0.0
        pair_radial_speed = -np.sqrt(2.0 * pair_kinetic / reduced_pair_mass)
        relative_pair_velocity = np.array([pair_radial_speed, 0.0])

        cluster_center_velocity = -(masses[2] / total_mass) * cluster_third_velocity
        third_velocity = (pair_mass / total_mass) * cluster_third_velocity
        velocities = np.array(
            [
                cluster_center_velocity
                - (masses[1] / pair_mass) * relative_pair_velocity,
                cluster_center_velocity
                + (masses[0] / pair_mass) * relative_pair_velocity,
                third_velocity,
            ]
        )

        kinetic = 0.5 * float(np.sum(masses[:, None] * velocities**2))
        assert kinetic - potential == pytest.approx(finite_energy)
        centered_L = centered_angular_momentum_components(positions, velocities, masses)
        assert centered_L[0] == pytest.approx(target_outer_angular)
        assert (
            reduced_pair_mass
            * (
                relative_pair[0] * relative_pair_velocity[1]
                - relative_pair[1] * relative_pair_velocity[0]
            )
            == pytest.approx(0.0)
        )

        offset_31 = cluster_third + (masses[1] / pair_mass) * relative_pair
        offset_32 = cluster_third - (masses[0] / pair_mass) * relative_pair
        cluster_third_acceleration = -(total_mass / pair_mass) * (
            masses[0] * offset_31 / np.linalg.norm(offset_31) ** 3
            + masses[1] * offset_32 / np.linalg.norm(offset_32) ** 3
        )
        radial_unit = cluster_third / np.linalg.norm(cluster_third)
        radial_speed = float(np.dot(radial_unit, cluster_third_velocity))
        radial_acceleration = (
            (
                float(np.dot(cluster_third_velocity, cluster_third_velocity))
                - radial_speed**2
            )
            / cluster_scale
            + float(np.dot(radial_unit, cluster_third_acceleration))
        )

        assert np.linalg.norm(relative_pair) / np.linalg.norm(cluster_third) < 0.02
        assert radial_acceleration >= barrier_constant / cluster_scale**3
        radial_acceleration_scales.append(radial_acceleration * cluster_scale**3)
        barrier_energies.append(radial_speed**2 + barrier_constant / cluster_scale**2)

    assert min(radial_acceleration_scales) > barrier_constant
    assert barrier_energies[1] > 3.9 * barrier_energies[0]
    assert barrier_energies[2] > 3.9 * barrier_energies[1]


def test_binary_degenerate_jacobi_kepler_reduction_constructor_certifies_tight_cluster():
    masses = np.array([1.0, 0.7, 1.4])
    pair_mass = masses[0] + masses[1]
    total_mass = float(np.sum(masses))
    rho = np.array([1.0e-2, 1.7e-3])
    rho_norm = np.linalg.norm(rho)
    pair_direction = np.array([0.6, 0.8])
    certificates = []

    for pair_to_outer_ratio in (0.08, 0.04, 0.02, 0.01):
        relative_pair = pair_to_outer_ratio * rho_norm * pair_direction
        cluster_center = -(masses[2] / total_mass) * rho
        third_position = (pair_mass / total_mass) * rho
        positions = np.array(
            [
                cluster_center - (masses[1] / pair_mass) * relative_pair,
                cluster_center + (masses[0] / pair_mass) * relative_pair,
                third_position,
            ]
        )
        certificates.append(
            certify_binary_degenerate_jacobi_kepler_reduction(
                positions,
                masses,
                pair=(0, 1),
                max_pair_to_outer_ratio=0.011,
                max_pair_error_ratio=2.0e-6,
                max_outer_error_ratio=5.0e-5,
            )
        )

    assert not certificates[0].certified
    assert certificates[-1].certified
    assert certificates[-1].pair_to_outer_ratio == pytest.approx(0.01)
    assert certificates[-1].kepler_scale_ratio_floor == pytest.approx(
        (pair_mass / total_mass) ** (1.0 / 3.0)
    )
    assert certificates[1].pair_kepler_error_ratio < (
        0.13 * certificates[0].pair_kepler_error_ratio
    )
    assert certificates[2].outer_kepler_error_ratio < (
        0.26 * certificates[1].outer_kepler_error_ratio
    )


def test_binary_degenerate_jacobi_equations_reduce_to_kepler_scale_contradiction():
    masses = np.array([1.0, 0.7, 1.4])
    pair_mass = masses[0] + masses[1]
    total_mass = float(np.sum(masses))
    rho = np.array([1.0e-2, 1.7e-3])
    rho_norm = np.linalg.norm(rho)
    pair_direction = np.array([0.6, 0.8])
    pair_error_ratios = []
    outer_error_ratios = []

    for pair_to_outer_ratio in (0.08, 0.04, 0.02, 0.01):
        relative_pair = pair_to_outer_ratio * rho_norm * pair_direction
        cluster_center = -(masses[2] / total_mass) * rho
        third_position = (pair_mass / total_mass) * rho
        positions = np.array(
            [
                cluster_center - (masses[1] / pair_mass) * relative_pair,
                cluster_center + (masses[0] / pair_mass) * relative_pair,
                third_position,
            ]
        )
        acceleration = accelerations(positions, masses)
        relative_pair_acceleration = acceleration[1] - acceleration[0]
        outer_acceleration = acceleration[2] - (
            masses[0] * acceleration[0] + masses[1] * acceleration[1]
        ) / pair_mass
        pair_kepler_acceleration = (
            -pair_mass
            * relative_pair
            / np.linalg.norm(relative_pair) ** 3
        )
        outer_kepler_acceleration = -total_mass * rho / rho_norm**3

        pair_error_ratios.append(
            np.linalg.norm(relative_pair_acceleration - pair_kepler_acceleration)
            / np.linalg.norm(pair_kepler_acceleration)
        )
        outer_error_ratios.append(
            np.linalg.norm(outer_acceleration - outer_kepler_acceleration)
            / np.linalg.norm(outer_kepler_acceleration)
        )

    kepler_scale_ratio_floor = (pair_mass / total_mass) ** (1.0 / 3.0)

    assert pair_error_ratios[1] < 0.13 * pair_error_ratios[0]
    assert pair_error_ratios[2] < 0.13 * pair_error_ratios[1]
    assert pair_error_ratios[3] < 0.13 * pair_error_ratios[2]
    assert outer_error_ratios[1] < 0.26 * outer_error_ratios[0]
    assert outer_error_ratios[2] < 0.26 * outer_error_ratios[1]
    assert outer_error_ratios[3] < 0.26 * outer_error_ratios[2]
    assert pair_error_ratios[-1] < 2.0e-6
    assert outer_error_ratios[-1] < 5.0e-5
    assert 0.8 < kepler_scale_ratio_floor < 0.85


def test_perturbed_kepler_collision_blow_up_lemma_tracks_parabolic_rate():
    kepler_parameter = 1.7
    leading_scale = (9.0 * kepler_parameter / 2.0) ** (1.0 / 3.0)
    perturbation_amplitude = 0.08
    perturbation_power = 5.0 / 3.0
    radial_direction = np.array([1.0, 0.0])
    transverse_direction = np.array([0.0, 1.0])
    scaled_force_defects = []
    scaled_energy_defects = []
    scaled_angular_defects = []
    scale_errors = []
    moment_equation_errors = []

    for time_to_collision in (1.0e-2, 3.0e-3, 1.0e-3):
        position = (
            leading_scale * time_to_collision ** (2.0 / 3.0) * radial_direction
            + perturbation_amplitude
            * time_to_collision**perturbation_power
            * transverse_direction
        )
        velocity = -(
            (2.0 / 3.0)
            * leading_scale
            * time_to_collision ** (-1.0 / 3.0)
            * radial_direction
            + perturbation_amplitude
            * perturbation_power
            * time_to_collision ** (perturbation_power - 1.0)
            * transverse_direction
        )
        acceleration = (
            -(2.0 / 9.0)
            * leading_scale
            * time_to_collision ** (-4.0 / 3.0)
            * radial_direction
            + perturbation_amplitude
            * perturbation_power
            * (perturbation_power - 1.0)
            * time_to_collision ** (perturbation_power - 2.0)
            * transverse_direction
        )
        radius = np.linalg.norm(position)
        forcing = acceleration + kepler_parameter * position / radius**3
        coordinate_energy = 0.5 * float(np.dot(velocity, velocity)) - (
            kepler_parameter / radius
        )
        angular_momentum = position[0] * velocity[1] - position[1] * velocity[0]
        moment_second_derivative = (
            2.0 * float(np.dot(velocity, velocity))
            + 2.0 * float(np.dot(position, acceleration))
        )
        expected_moment_second_derivative = 2.0 * kepler_parameter / radius

        scaled_force_defects.append(radius**2 * np.linalg.norm(forcing))
        scaled_energy_defects.append(radius * abs(coordinate_energy))
        scaled_angular_defects.append(angular_momentum**2 / radius)
        scale_errors.append(abs(radius / time_to_collision ** (2.0 / 3.0) - leading_scale))
        moment_equation_errors.append(
            abs(moment_second_derivative - expected_moment_second_derivative)
            / expected_moment_second_derivative
        )

    assert scaled_force_defects[1] < 0.5 * scaled_force_defects[0]
    assert scaled_force_defects[2] < 0.5 * scaled_force_defects[1]
    assert scaled_energy_defects[1] < 0.5 * scaled_energy_defects[0]
    assert scaled_energy_defects[2] < 0.5 * scaled_energy_defects[1]
    assert scaled_angular_defects[1] < 0.35 * scaled_angular_defects[0]
    assert scaled_angular_defects[2] < 0.35 * scaled_angular_defects[1]
    assert scale_errors[1] < 0.4 * scale_errors[0]
    assert scale_errors[2] < 0.4 * scale_errors[1]
    assert moment_equation_errors[1] < 0.5 * moment_equation_errors[0]
    assert moment_equation_errors[2] < 0.5 * moment_equation_errors[1]
    assert scaled_force_defects[-1] < 5.0e-4
    assert scaled_energy_defects[-1] < 2.0e-5
    assert scaled_angular_defects[-1] < 1.0e-5
    assert scale_errors[-1] < 2.0e-5
    assert moment_equation_errors[-1] < 3.0e-5


@pytest.mark.parametrize("shape", ("equilateral", "euler"))
def test_parabolic_homothetic_total_collision_branches_solve_newton_away_from_collision(shape):
    time = 0.37
    positions, velocities, central_lambda, scale_factor = _parabolic_homothetic_state(
        shape,
        time,
    )
    configuration, _ = _central_configuration_data(shape)
    expected_radius_acceleration = (
        -(2.0 / 9.0) * scale_factor * time ** (-4.0 / 3.0)
    )
    expected_acceleration = expected_radius_acceleration * configuration

    state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])

    assert np.linalg.norm(accelerations(positions) - expected_acceleration, ord=np.inf) < 1e-12
    assert np.linalg.norm(center_of_mass(state), ord=np.inf) < 1e-14
    assert np.linalg.norm(linear_momentum(state), ord=np.inf) < 1e-14
    assert abs(angular_momentum_z(state)) < 1e-14
    assert abs(energy(state)) < 1e-12
    assert central_lambda > 0.0


def test_zero_angular_total_collision_has_multiple_parabolic_ejection_branches():
    near_collision_time = 1e-9
    comparison_time = 1e-3
    equilateral_near, _equilateral_velocity, *_ = _parabolic_homothetic_state(
        "equilateral",
        near_collision_time,
    )
    euler_near, _euler_velocity, *_ = _parabolic_homothetic_state(
        "euler",
        near_collision_time,
    )
    equilateral_later, _equilateral_later_velocity, *_ = _parabolic_homothetic_state(
        "equilateral",
        comparison_time,
    )
    euler_later, _euler_later_velocity, *_ = _parabolic_homothetic_state(
        "euler",
        comparison_time,
    )

    assert np.linalg.norm(equilateral_near, ord=np.inf) < 2e-6
    assert np.linalg.norm(euler_near, ord=np.inf) < 2e-6
    assert np.linalg.norm(equilateral_later - euler_later, ord=np.inf) > 1e-2


def test_parabolic_total_collision_branches_share_collision_invariants_but_not_shape():
    time = 0.02
    equilateral_positions, equilateral_velocities, *_ = _parabolic_homothetic_state(
        "equilateral",
        time,
    )
    euler_positions, euler_velocities, *_ = _parabolic_homothetic_state(
        "euler",
        time,
    )
    equilateral_state = np.concatenate(
        [equilateral_positions.reshape(-1), equilateral_velocities.reshape(-1)]
    )
    euler_state = np.concatenate([euler_positions.reshape(-1), euler_velocities.reshape(-1)])

    assert np.linalg.norm(center_of_mass(equilateral_state), ord=np.inf) < 1e-14
    assert np.linalg.norm(center_of_mass(euler_state), ord=np.inf) < 1e-14
    assert np.linalg.norm(linear_momentum(equilateral_state), ord=np.inf) < 1e-14
    assert np.linalg.norm(linear_momentum(euler_state), ord=np.inf) < 1e-14
    assert abs(angular_momentum_z(equilateral_state)) < 1e-14
    assert abs(angular_momentum_z(euler_state)) < 1e-14
    assert abs(energy(equilateral_state) - energy(euler_state)) < 1e-12
    assert np.linalg.norm(equilateral_positions - euler_positions, ord=np.inf) > 0.05


def test_zero_angular_total_collision_branch_is_regularized_second_jet_data():
    regularized_time = 1e-3
    branch_jets = {}

    for shape in ("equilateral", "euler"):
        configuration, central_lambda = _central_configuration_data(shape)
        scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
        quadratic_coefficient = scale_factor * configuration
        second_regularized_jet = 2.0 * quadratic_coefficient
        physical_time = regularized_time**3
        expected_positions, _expected_velocities, *_ = _parabolic_homothetic_state(
            shape,
            physical_time,
        )
        regularized_positions = (
            0.5 * second_regularized_jet * regularized_time**2
        )

        assert np.linalg.norm(
            accelerations(quadratic_coefficient) + (2.0 / 9.0) * quadratic_coefficient,
            ord=np.inf,
        ) < 1e-12
        assert np.linalg.norm(
            accelerations(second_regularized_jet)
            + (1.0 / 36.0) * second_regularized_jet,
            ord=np.inf,
        ) < 1e-12
        assert np.linalg.norm(regularized_positions - expected_positions, ord=np.inf) < 1e-14
        branch_jets[shape] = second_regularized_jet

    collapsed_position = np.zeros((3, 2))
    first_regularized_derivative = np.zeros((3, 2))

    assert np.linalg.norm(collapsed_position, ord=np.inf) == 0.0
    assert np.linalg.norm(first_regularized_derivative, ord=np.inf) == 0.0
    assert np.linalg.norm(
        branch_jets["equilateral"] - branch_jets["euler"],
        ord=np.inf,
    ) > 2.0


@pytest.mark.parametrize("shape", ("equilateral", "euler"))
def test_normalized_potential_limit_forces_parabolic_inertia_scale(shape):
    masses = np.ones(3)
    configuration, central_lambda = _central_configuration_data(shape)
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    shape_inertia = float(np.sum(masses[:, None] * configuration**2))
    shape_potential = 0.0
    for first in range(3):
        for second in range(first + 1, 3):
            shape_potential += masses[first] * masses[second] / np.linalg.norm(
                configuration[first] - configuration[second]
            )
    normalized_potential_limit = shape_potential * np.sqrt(shape_inertia)
    predicted_inertia_scale = ((9.0 / 2.0) * normalized_potential_limit) ** (
        2.0 / 3.0
    )
    actual_inertia_scale = scale_factor**2 * shape_inertia

    assert shape_potential == pytest.approx(central_lambda * shape_inertia)
    assert actual_inertia_scale == pytest.approx(predicted_inertia_scale)

    observed_scales = []
    normalized_potentials = []
    inertia_derivatives = []
    for collision_gap in (1e-2, 5e-3, 2.5e-3):
        positions, velocities, _central_lambda, _scale_factor = _parabolic_homothetic_state(
            shape,
            -collision_gap,
        )
        inertia = float(np.sum(masses[:, None] * positions**2))
        inertia_derivative = 2.0 * float(np.sum(masses[:, None] * positions * velocities))
        kinetic = 0.5 * float(np.sum(masses[:, None] * velocities**2))
        potential = 0.0
        for first in range(3):
            for second in range(first + 1, 3):
                potential += masses[first] * masses[second] / np.linalg.norm(
                    positions[first] - positions[second]
                )
        inertia_second_derivative = 4.0 * (kinetic - potential) + 2.0 * potential
        leading_coefficient = positions / collision_gap ** (2.0 / 3.0)

        normalized_potentials.append(potential * np.sqrt(inertia))
        observed_scales.append(inertia / collision_gap ** (4.0 / 3.0))
        inertia_derivatives.append(inertia_derivative)
        expected_inertia_derivative = (
            -(4.0 / 3.0)
            * predicted_inertia_scale
            * collision_gap ** (1.0 / 3.0)
        )
        assert kinetic - potential == pytest.approx(0.0, abs=1e-12)
        assert inertia_derivative < 0.0
        assert inertia_derivative == pytest.approx(expected_inertia_derivative)
        assert inertia_second_derivative == pytest.approx(2.0 * potential)
        assert np.linalg.norm(
            leading_coefficient - scale_factor * configuration,
            ord=np.inf,
        ) < 1e-14

    np.testing.assert_allclose(
        normalized_potentials,
        [normalized_potential_limit] * len(normalized_potentials),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        observed_scales,
        [predicted_inertia_scale] * len(observed_scales),
        rtol=1e-14,
        atol=1e-14,
    )
    assert abs(inertia_derivatives[2]) < abs(inertia_derivatives[1]) < abs(
        inertia_derivatives[0]
    )


def test_shape_compact_total_collapse_has_finite_separated_central_quotient_targets():
    masses = np.array([1.0, 0.7, 1.4])

    def normalized_mutual_distances(configuration):
        inertia = float(np.sum(masses[:, None] * configuration**2))
        normalized = configuration / np.sqrt(inertia)
        return np.array(
            [
                np.linalg.norm(normalized[1] - normalized[0]),
                np.linalg.norm(normalized[2] - normalized[0]),
                np.linalg.norm(normalized[2] - normalized[1]),
            ]
        )

    central_representatives = []
    equilateral_configuration, _central_lambda = _mass_centered_equilateral_central_configuration(
        masses,
    )
    central_representatives.append(equilateral_configuration)

    for permutation in itertools.permutations(range(3)):
        permuted_masses = masses[list(permutation)]
        _ratio, ordered_configuration = _ordered_euler_scaled_configuration(permuted_masses)
        unpermuted_configuration = np.zeros_like(ordered_configuration)
        for ordered_index, original_index in enumerate(permutation):
            unpermuted_configuration[original_index] = ordered_configuration[ordered_index]
        central_representatives.append(unpermuted_configuration)

    quotient_targets = []
    normalized_potential_targets = []
    for configuration in central_representatives:
        acceleration = accelerations(configuration, masses)
        central_multiplier = -float(
            np.sum(masses[:, None] * configuration * acceleration)
            / np.sum(masses[:, None] * configuration * configuration)
        )
        central_residual = acceleration + central_multiplier * configuration
        quotient_target = normalized_mutual_distances(configuration)
        normalized_potential = sum(
            masses[first] * masses[second] / quotient_target[pair_index]
            for pair_index, (first, second) in enumerate(((0, 1), (0, 2), (1, 2)))
        )

        assert central_multiplier > 0.0
        assert np.linalg.norm(central_residual, ord=np.inf) < 1e-11
        assert min(quotient_target) > 0.5
        assert normalized_potential > 0.0

        if not any(np.linalg.norm(quotient_target - known) < 1e-10 for known in quotient_targets):
            quotient_targets.append(quotient_target)
            normalized_potential_targets.append(normalized_potential)

    pairwise_target_separations = [
        np.linalg.norm(quotient_targets[right] - quotient_targets[left])
        for left in range(len(quotient_targets))
        for right in range(left + 1, len(quotient_targets))
    ]
    minimum_target_separation = min(pairwise_target_separations)

    assert len(quotient_targets) == 4
    assert minimum_target_separation > 0.25
    assert min(normalized_potential_targets) > 1.0

    connected_tail_diameter = 0.49 * minimum_target_separation
    for left in range(len(quotient_targets)):
        for right in range(left + 1, len(quotient_targets)):
            assert (
                np.linalg.norm(quotient_targets[right] - quotient_targets[left])
                > 2.0 * connected_tail_diameter
            )


def test_finite_reduced_shape_length_upgrades_to_oriented_shape_limit():
    masses, configuration, _central_lambda = _arbitrary_mass_equilateral_central_configuration()
    target = configuration / np.sqrt(float(np.sum(masses[:, None] * configuration**2)))

    def center_and_project_tangent(vector, existing=()):
        vector = np.array(vector, dtype=float)
        vector -= np.average(vector, axis=0, weights=masses)
        vector -= _mass_inner_product(masses, vector, target) * target
        for basis in existing:
            vector -= _mass_inner_product(masses, vector, basis) * basis
        return vector / np.sqrt(_mass_inner_product(masses, vector, vector))

    tangent_1 = center_and_project_tangent(
        [
            [0.2, -0.1],
            [-0.1, 0.3],
            [0.05, -0.2],
        ],
    )
    tangent_2 = center_and_project_tangent(
        [
            [-0.1, 0.25],
            [0.3, -0.05],
            [-0.2, -0.1],
        ],
        existing=(tangent_1,),
    )

    def normalize_shape(raw, raw_derivative):
        inertia = _mass_inner_product(masses, raw, raw)
        inertia_derivative = 2.0 * _mass_inner_product(masses, raw, raw_derivative)
        scale = np.sqrt(inertia)
        shape = raw / scale
        shape_derivative = raw_derivative / scale - 0.5 * raw * inertia_derivative / (
            inertia * scale
        )
        return shape, shape_derivative

    def reduced_shape_and_derivative(parameter):
        amplitude_1 = 0.18 * np.exp(-parameter)
        amplitude_2 = -0.11 * np.exp(-1.35 * parameter)
        amplitude_1_derivative = -amplitude_1
        amplitude_2_derivative = -1.35 * amplitude_2
        raw = target + amplitude_1 * tangent_1 + amplitude_2 * tangent_2
        raw_derivative = (
            amplitude_1_derivative * tangent_1
            + amplitude_2_derivative * tangent_2
        )
        return normalize_shape(raw, raw_derivative)

    def mass_wedge(left, right):
        return float(
            np.sum(masses * (left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]))
        )

    def rotate(shape, angle):
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        return shape @ rotation.T

    def quarter_turn(shape):
        return np.column_stack((-shape[:, 1], shape[:, 0]))

    def theta_derivative(parameter):
        shape, shape_derivative = reduced_shape_and_derivative(parameter)
        return -mass_wedge(shape, shape_derivative)

    def theta_value(parameter, limit_angle=0.37):
        grid = np.linspace(parameter, 40.0, 4000)
        integral = np.trapz([theta_derivative(value) for value in grid], grid)
        return limit_angle - integral

    orientation_errors = []
    angular_momentum_defects = []
    connection_bounds = []
    shape_speeds = []
    angle_tail_bounds = []
    limit_angle = 0.37
    for parameter in (2.0, 3.0, 4.0, 5.0):
        shape, shape_derivative = reduced_shape_and_derivative(parameter)
        theta = theta_value(parameter, limit_angle)
        theta_s = theta_derivative(parameter)
        oriented_shape = rotate(shape, theta)
        oriented_shape_derivative = rotate(
            theta_s * quarter_turn(shape) + shape_derivative,
            theta,
        )
        shape_speed = np.sqrt(_mass_inner_product(masses, shape_derivative, shape_derivative))
        angular_connection = abs(mass_wedge(shape, shape_derivative))
        tail_grid = np.linspace(parameter, 40.0, 4000)
        shape_tail_length = np.trapz(
            [
                np.sqrt(
                    _mass_inner_product(
                        masses,
                        reduced_shape_and_derivative(value)[1],
                        reduced_shape_and_derivative(value)[1],
                    )
                )
                for value in tail_grid
            ],
            tail_grid,
        )
        angle_tail = abs(limit_angle - theta)

        orientation_errors.append(
            np.linalg.norm(oriented_shape - rotate(target, limit_angle), ord=np.inf)
        )
        angular_momentum_defects.append(abs(mass_wedge(oriented_shape, oriented_shape_derivative)))
        connection_bounds.append(angular_connection - shape_speed)
        shape_speeds.append(shape_speed)
        angle_tail_bounds.append(shape_tail_length - angle_tail)

    assert orientation_errors[0] > orientation_errors[1] > orientation_errors[2] > orientation_errors[3]
    assert shape_speeds[0] > shape_speeds[1] > shape_speeds[2] > shape_speeds[3]
    assert max(angular_momentum_defects) < 1e-15
    assert max(connection_bounds) <= 1e-15
    assert min(angle_tail_bounds) >= -1e-6
    assert orientation_errors[-1] < 2.0e-3


def test_lojasiewicz_angle_criterion_bounds_reduced_shape_length():
    # Model the local reduced-shape tail by an analytic gradient-like slice
    # with an isolated critical point:
    #     Phi(x, y) = x^4/4 + y^2/2.
    # The x-direction has Lojasiewicz exponent 3/4, matching the slowest
    # finite-length decay that the proof has to control.
    x0 = 0.34
    y0 = 0.08
    theta = 0.75
    angle_constant = 1.0

    def coordinates(parameter):
        return np.array(
            [
                1.0 / np.sqrt(x0 ** -2 + 2.0 * parameter),
                y0 * np.exp(-parameter),
            ]
        )

    def potential_gap(point):
        return 0.25 * point[0] ** 4 + 0.5 * point[1] ** 2

    def gradient(point):
        return np.array([point[0] ** 3, point[1]])

    finite_length_bounds = []
    tail_variation_bounds = []
    derivative_margins = []
    for start in (2.0, 5.0, 10.0):
        grid = np.linspace(start, 200.0, 5000)
        points = np.array([coordinates(value) for value in grid])
        gaps = np.array([potential_gap(point) for point in points])
        gradients = np.array([gradient(point) for point in points])
        gradient_norms = np.linalg.norm(gradients, axis=1)
        speeds = gradient_norms

        # Any smaller positive constant than the sampled infimum is a valid
        # Lojasiewicz constant on this tail.
        kappa = 0.25 * float(np.min(gradient_norms / gaps**theta))
        gap_derivative_drop = gradient_norms**2
        derivative_margins.append(
            float(
                np.min(
                    (1.0 - theta)
                    * gaps ** (-theta)
                    * gap_derivative_drop
                    - (1.0 - theta)
                    * angle_constant
                    * kappa
                    * speeds
                )
            )
        )

        finite_length_bound = (
            potential_gap(coordinates(start)) ** (1.0 - theta)
            / ((1.0 - theta) * angle_constant * kappa)
        )
        # The coordinatewise total variation bounds the Euclidean curve length
        # and is explicit for this gradient-flow tail.
        tail_variation_bound = float(np.sum(np.abs(coordinates(start))))
        finite_length_bounds.append(finite_length_bound)
        tail_variation_bounds.append(tail_variation_bound)

    assert min(derivative_margins) >= -1e-12
    assert all(
        tail <= bound
        for tail, bound in zip(tail_variation_bounds, finite_length_bounds)
    )
    assert finite_length_bounds[0] > finite_length_bounds[1] > finite_length_bounds[2]
    assert tail_variation_bounds[0] > tail_variation_bounds[1] > tail_variation_bounds[2]


def test_hyperbolic_damped_shape_entry_has_exponential_finite_length():
    # A local hyperbolic reduced McGehee target linearizes to damped shape
    # modes. This underdamped model deliberately allows oscillation, so finite
    # length is coming from hyperbolic exponential decay rather than a
    # pointwise gradient-angle condition.
    damping = 1.15
    decay_rate = damping / 2.0
    eigenvalues = np.array([1.0, 2.75])
    amplitudes = np.array([0.31, -0.18])
    phase_amplitudes = np.array([0.07, 0.22])
    frequencies = np.sqrt(eigenvalues - decay_rate**2)

    def mode_state(parameter):
        exp_factor = np.exp(-decay_rate * parameter)
        cosines = np.cos(frequencies * parameter)
        sines = np.sin(frequencies * parameter)
        positions = exp_factor * (amplitudes * cosines + phase_amplitudes * sines)
        velocities = exp_factor * (
            (-decay_rate * amplitudes + frequencies * phase_amplitudes) * cosines
            + (-decay_rate * phase_amplitudes - frequencies * amplitudes) * sines
        )
        accelerations = -damping * velocities - eigenvalues * positions
        return positions, velocities, accelerations

    velocity_amplitude = np.sqrt(
        np.sum(
            (-decay_rate * amplitudes + frequencies * phase_amplitudes) ** 2
            + (-decay_rate * phase_amplitudes - frequencies * amplitudes) ** 2
        )
    )

    tail_bounds = []
    observed_tails = []
    energy_decay_defects = []
    stable_state_ratios = []
    for start in (2.0, 4.0, 6.0):
        grid = np.linspace(start, 80.0, 6000)
        states = [mode_state(value) for value in grid]
        speeds = np.array([np.linalg.norm(state[1]) for state in states])
        observed_tail = float(np.trapz(speeds, grid))
        tail_bound = velocity_amplitude * np.exp(-decay_rate * start) / decay_rate
        observed_tails.append(observed_tail)
        tail_bounds.append(tail_bound)

        positions, velocities, accelerations = mode_state(start)
        energy_derivative = float(
            np.dot(velocities, accelerations + eigenvalues * positions)
        )
        expected_energy_derivative = -damping * float(np.dot(velocities, velocities))
        energy_decay_defects.append(abs(energy_derivative - expected_energy_derivative))

        later_positions, later_velocities, _accelerations = mode_state(start + 2.0)
        current_norm = np.linalg.norm(np.concatenate((positions, velocities)))
        later_norm = np.linalg.norm(np.concatenate((later_positions, later_velocities)))
        stable_state_ratios.append(later_norm / current_norm)

    assert all(tail <= bound for tail, bound in zip(observed_tails, tail_bounds))
    assert tail_bounds[1] < np.exp(-decay_rate * 1.9) * tail_bounds[0]
    assert tail_bounds[2] < np.exp(-decay_rate * 1.9) * tail_bounds[1]
    assert max(energy_decay_defects) < 1e-15
    assert max(stable_state_ratios) < 0.65


def test_hyperbolic_stable_eigenrates_recover_fuchsian_selector_powers():
    masses, configuration, _central_lambda = _arbitrary_mass_equilateral_central_configuration()
    central_shape = configuration / np.sqrt(
        float(np.sum(masses[:, None] * configuration**2))
    )

    def center_and_project(vector, existing=()):
        vector = np.array(vector, dtype=float)
        vector -= np.average(vector, axis=0, weights=masses)
        vector -= _mass_inner_product(masses, vector, central_shape) * central_shape
        for basis in existing:
            vector -= _mass_inner_product(masses, vector, basis) * basis
        return vector / np.sqrt(_mass_inner_product(masses, vector, vector))

    mode_one = center_and_project(
        [
            [0.20, -0.08],
            [-0.05, 0.16],
            [0.04, -0.10],
        ],
    )
    mode_two = center_and_project(
        [
            [-0.04, 0.18],
            [0.15, -0.02],
            [-0.08, -0.04],
        ],
        existing=(mode_one,),
    )

    radial_decay = 1.4
    stable_decays = np.array([1.05, 1.96])
    fuchsian_powers = 2.0 * stable_decays / radial_decay
    amplitudes = np.array([0.035, -0.026])
    scale_amplitude = 0.018
    mixed_amplitude = 0.011

    def tau_from_mcgehee_time(parameter):
        return np.exp(-0.5 * radial_decay * parameter)

    def shape_from_tau(tau):
        return (
            central_shape
            + scale_amplitude * tau**2 * central_shape
            + amplitudes[0] * tau ** fuchsian_powers[0] * mode_one
            + amplitudes[1] * tau ** fuchsian_powers[1] * mode_two
            + mixed_amplitude
            * tau ** (fuchsian_powers[0] + fuchsian_powers[1])
            * (mode_one - mode_two)
        )

    exponential_conversion_errors = []
    for parameter in (3.0, 5.0, 7.0):
        tau = tau_from_mcgehee_time(parameter)
        for decay, power in zip(stable_decays, fuchsian_powers):
            exponential_conversion_errors.append(abs(np.exp(-decay * parameter) - tau**power))

    selector_errors = []
    analytic_quartic_quotients = []
    fuchsian_quotients = []
    for tau in (8.0e-2, 2.0e-2, 5.0e-3):
        shape = shape_from_tau(tau)
        centered_shape = shape - central_shape - scale_amplitude * tau**2 * central_shape
        recovered_one = _mass_inner_product(masses, centered_shape, mode_one) / (
            tau ** fuchsian_powers[0]
        )
        recovered_two = _mass_inner_product(masses, centered_shape, mode_two) / (
            tau ** fuchsian_powers[1]
        )
        selector_errors.append(
            max(abs(recovered_one - amplitudes[0]), abs(recovered_two - amplitudes[1]))
        )

        regularized_position = tau**2 * shape
        analytic_quartic_quotients.append(
            abs(
                _mass_inner_product(
                    masses,
                    regularized_position - tau**2 * central_shape,
                    mode_one,
                )
                / tau**4
            )
        )
        fuchsian_quotients.append(abs(recovered_one))

    assert max(exponential_conversion_errors) < 1e-15
    assert fuchsian_powers[0] < 2.0 < fuchsian_powers[1]
    assert selector_errors[1] < 0.14 * selector_errors[0]
    assert selector_errors[2] < 0.14 * selector_errors[1]
    assert fuchsian_quotients[-1] == pytest.approx(abs(amplitudes[0]), rel=2e-4)
    assert analytic_quartic_quotients[0] < analytic_quartic_quotients[1] < analytic_quartic_quotients[2]


def test_resonant_stable_normal_form_recovers_log_subtracted_selector():
    radial_decay = 1.3
    stable_decay = 0.91
    resonant_power = 2
    coupling = -1.1
    first_amplitude = 0.042
    resonant_selector = -0.018
    fuchsian_power = 2.0 * stable_decay / radial_decay
    resonant_fuchsian_power = resonant_power * fuchsian_power
    forced_log_coefficient = (
        -2.0
        * coupling
        * first_amplitude**resonant_power
        / radial_decay
    )

    def tau_from_mcgehee_time(parameter):
        return np.exp(-0.5 * radial_decay * parameter)

    def resonant_modes_from_time(parameter):
        first_mode = first_amplitude * np.exp(-stable_decay * parameter)
        second_mode = np.exp(-resonant_power * stable_decay * parameter) * (
            resonant_selector
            + coupling * first_amplitude**resonant_power * parameter
        )
        return first_mode, second_mode

    conversion_errors = []
    raw_quotients = []
    recovered_selectors = []
    log_terms = []
    for parameter in (3.0, 9.0, 15.0, 21.0):
        tau = tau_from_mcgehee_time(parameter)
        first_mode, second_mode = resonant_modes_from_time(parameter)
        conversion_errors.append(abs(first_mode - first_amplitude * tau**fuchsian_power))

        raw_quotient = second_mode / tau**resonant_fuchsian_power
        log_term = forced_log_coefficient * np.log(tau)
        recovered_selector = raw_quotient - log_term

        raw_quotients.append(raw_quotient)
        recovered_selectors.append(recovered_selector)
        log_terms.append(log_term)

    assert max(conversion_errors) < 1e-15
    assert abs(raw_quotients[-1]) > 2.0 * abs(raw_quotients[0])
    assert abs(log_terms[-1]) > 2.0 * abs(log_terms[0])
    np.testing.assert_allclose(
        recovered_selectors,
        [resonant_selector] * len(recovered_selectors),
        rtol=1e-13,
        atol=1e-13,
    )


def test_finite_resonant_stable_chain_recovers_log_polynomial_selector():
    radial_decay = 1.25
    stable_decay = 0.875
    first_amplitude = 0.038
    first_selector = -0.014
    second_selector = 0.021
    first_coupling = -0.82
    second_coupling = 1.35
    resonant_power = 2
    fuchsian_power = 2.0 * stable_decay / radial_decay
    resonant_fuchsian_power = resonant_power * fuchsian_power

    first_log_coefficient = (
        -2.0
        * first_coupling
        * first_amplitude**resonant_power
        / radial_decay
    )
    second_log_coefficient = -2.0 * second_coupling * first_selector / radial_decay
    second_log_squared_coefficient = (
        2.0
        * second_coupling
        * first_coupling
        * first_amplitude**resonant_power
        / radial_decay**2
    )

    def tau_from_mcgehee_time(parameter):
        return np.exp(-0.5 * radial_decay * parameter)

    def resonant_chain_modes(parameter):
        forcing_amplitude = first_coupling * first_amplitude**resonant_power
        first_source = first_amplitude * np.exp(-stable_decay * parameter)
        first_resonant = np.exp(-resonant_power * stable_decay * parameter) * (
            first_selector + forcing_amplitude * parameter
        )
        second_resonant = np.exp(-resonant_power * stable_decay * parameter) * (
            second_selector
            + second_coupling * first_selector * parameter
            + 0.5 * second_coupling * forcing_amplitude * parameter**2
        )
        return first_source, first_resonant, second_resonant

    recovered_first_selectors = []
    recovered_second_selectors = []
    raw_second_quotients = []
    log_squared_terms = []
    for parameter in (4.0, 10.0, 16.0, 22.0):
        tau = tau_from_mcgehee_time(parameter)
        first_source, first_resonant, second_resonant = resonant_chain_modes(parameter)
        np.testing.assert_allclose(
            first_source,
            first_amplitude * tau**fuchsian_power,
            rtol=1e-14,
            atol=1e-14,
        )

        first_raw = first_resonant / tau**resonant_fuchsian_power
        first_recovered = first_raw - first_log_coefficient * np.log(tau)

        second_raw = second_resonant / tau**resonant_fuchsian_power
        second_forced_logs = (
            second_log_squared_coefficient * np.log(tau) ** 2
            + second_log_coefficient * np.log(tau)
        )
        second_recovered = second_raw - second_forced_logs

        recovered_first_selectors.append(first_recovered)
        recovered_second_selectors.append(second_recovered)
        raw_second_quotients.append(second_raw)
        log_squared_terms.append(second_log_squared_coefficient * np.log(tau) ** 2)

    assert abs(raw_second_quotients[-1]) > 3.0 * abs(raw_second_quotients[0])
    assert abs(log_squared_terms[-1]) > 10.0 * abs(log_squared_terms[0])
    np.testing.assert_allclose(
        recovered_first_selectors,
        [first_selector] * len(recovered_first_selectors),
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        recovered_second_selectors,
        [second_selector] * len(recovered_second_selectors),
        rtol=2e-13,
        atol=2e-13,
    )


def test_poincare_dulac_stable_normal_form_has_complete_log_selectors():
    radial_decay = 1.4
    stable_rates = np.array([0.6, 0.9, 1.2, 1.5, 1.8])
    fuchsian_powers = 2.0 * stable_rates / radial_decay
    source_a = 0.047
    source_b = -0.031
    selector_c = 0.022
    selector_d = -0.018
    selector_e = 0.015
    coupling_c = 0.73
    coupling_d = -1.17
    coupling_e = 0.81
    coupling_b_square = -0.46

    resonant_monomials = {
        2: ((2, 0, 0, 0, 0),),
        3: ((1, 1, 0, 0, 0),),
        4: ((1, 0, 1, 0, 0), (0, 2, 0, 0, 0)),
    }
    for target_index, monomials in resonant_monomials.items():
        target_rate = stable_rates[target_index]
        for powers in monomials:
            assert np.dot(powers, stable_rates) == pytest.approx(target_rate)
            assert all(
                source_index < target_index
                for source_index, power in enumerate(powers)
                if power
            )

    def tau_from_mcgehee_time(parameter):
        return np.exp(-0.5 * radial_decay * parameter)

    def normal_form_modes(parameter):
        first = source_a * np.exp(-stable_rates[0] * parameter)
        second = source_b * np.exp(-stable_rates[1] * parameter)
        third = np.exp(-stable_rates[2] * parameter) * (
            selector_c + coupling_c * source_a**2 * parameter
        )
        fourth = np.exp(-stable_rates[3] * parameter) * (
            selector_d + coupling_d * source_a * source_b * parameter
        )
        fifth = np.exp(-stable_rates[4] * parameter) * (
            selector_e
            + (
                coupling_e * source_a * selector_c
                + coupling_b_square * source_b**2
            )
            * parameter
            + 0.5 * coupling_e * coupling_c * source_a**3 * parameter**2
        )
        return first, second, third, fourth, fifth

    third_log_coefficient = -2.0 * coupling_c * source_a**2 / radial_decay
    fourth_log_coefficient = (
        -2.0 * coupling_d * source_a * source_b / radial_decay
    )
    fifth_log_coefficient = (
        -2.0
        * (coupling_e * source_a * selector_c + coupling_b_square * source_b**2)
        / radial_decay
    )
    fifth_log_squared_coefficient = (
        2.0 * coupling_e * coupling_c * source_a**3 / radial_decay**2
    )

    recovered_selectors = []
    raw_fifth_quotients = []
    forced_log_squared_terms = []
    for parameter in (5.0, 11.0, 17.0, 23.0):
        tau = tau_from_mcgehee_time(parameter)
        modes = normal_form_modes(parameter)
        np.testing.assert_allclose(
            modes[0],
            source_a * tau ** fuchsian_powers[0],
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            modes[1],
            source_b * tau ** fuchsian_powers[1],
            rtol=1e-14,
            atol=1e-14,
        )

        logarithm = np.log(tau)
        third_raw = modes[2] / tau ** fuchsian_powers[2]
        fourth_raw = modes[3] / tau ** fuchsian_powers[3]
        fifth_raw = modes[4] / tau ** fuchsian_powers[4]
        third_recovered = third_raw - third_log_coefficient * logarithm
        fourth_recovered = fourth_raw - fourth_log_coefficient * logarithm
        fifth_recovered = fifth_raw - (
            fifth_log_squared_coefficient * logarithm**2
            + fifth_log_coefficient * logarithm
        )

        recovered_selectors.append(
            (third_recovered, fourth_recovered, fifth_recovered)
        )
        raw_fifth_quotients.append(fifth_raw)
        forced_log_squared_terms.append(fifth_log_squared_coefficient * logarithm**2)

    np.testing.assert_allclose(
        recovered_selectors,
        [(selector_c, selector_d, selector_e)] * len(recovered_selectors),
        rtol=3e-13,
        atol=3e-13,
    )
    assert abs(raw_fifth_quotients[-1]) > 2.0 * abs(raw_fifth_quotients[0])
    assert abs(forced_log_squared_terms[-1]) > 10.0 * abs(forced_log_squared_terms[0])


def test_stable_log_selector_chain_constructor_recovers_coupled_log_selectors():
    radial_decay = 1.4
    stable_rates = (0.6, 0.9, 1.2, 1.5, 1.8)
    source_a = 0.047
    source_b = -0.031
    selector_c = 0.022
    selector_d = -0.018
    selector_e = 0.015
    coupling_c = 0.73
    coupling_d = -1.17
    coupling_e = 0.81
    coupling_b_square = -0.46

    chain = construct_stable_log_selector_chain(
        radial_decay=radial_decay,
        stable_rates=stable_rates,
        source_amplitudes={0: source_a, 1: source_b},
        selectors={2: selector_c, 3: selector_d, 4: selector_e},
        resonant_rows={
            2: (StableResonanceTerm(coupling_c, (2, 0, 0, 0, 0)),),
            3: (StableResonanceTerm(coupling_d, (1, 1, 0, 0, 0)),),
            4: (
                StableResonanceTerm(coupling_e, (1, 0, 1, 0, 0)),
                StableResonanceTerm(coupling_b_square, (0, 2, 0, 0, 0)),
            ),
        },
    )
    third = chain.mode(2)
    fourth = chain.mode(3)
    fifth = chain.mode(4)
    expected_third_log = -2.0 * coupling_c * source_a**2 / radial_decay
    expected_fourth_log = -2.0 * coupling_d * source_a * source_b / radial_decay
    expected_fifth_log = (
        -2.0
        * (coupling_e * source_a * selector_c + coupling_b_square * source_b**2)
        / radial_decay
    )
    expected_fifth_log_squared = (
        2.0 * coupling_e * coupling_c * source_a**3 / radial_decay**2
    )

    assert chain.certified
    assert third.log_coefficients[1] == pytest.approx(expected_third_log, rel=1e-13, abs=1e-13)
    assert fourth.log_coefficients[1] == pytest.approx(expected_fourth_log, rel=1e-13, abs=1e-13)
    assert fifth.log_coefficients[1] == pytest.approx(expected_fifth_log, rel=1e-13, abs=1e-13)
    assert fifth.log_coefficients[2] == pytest.approx(
        expected_fifth_log_squared,
        rel=1e-13,
        abs=1e-13,
    )

    recovered = []
    raw_fifth = []
    log_squared_terms = []
    for parameter in (5.0, 11.0, 17.0, 23.0):
        tau = np.exp(-0.5 * radial_decay * parameter)
        np.testing.assert_allclose(
            chain.mode(0).value_at_time(parameter),
            chain.mode(0).value_at_tau(tau),
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            fifth.value_at_time(parameter),
            fifth.value_at_tau(tau),
            rtol=1e-14,
            atol=1e-14,
        )
        recovered.append(
            (
                third.recovered_selector_at_tau(tau),
                fourth.recovered_selector_at_tau(tau),
                fifth.recovered_selector_at_tau(tau),
            )
        )
        raw_fifth.append(fifth.raw_quotient_at_tau(tau))
        log_squared_terms.append(expected_fifth_log_squared * np.log(tau) ** 2)

    np.testing.assert_allclose(
        recovered,
        [(selector_c, selector_d, selector_e)] * len(recovered),
        rtol=3e-13,
        atol=3e-13,
    )
    assert abs(raw_fifth[-1]) > 2.0 * abs(raw_fifth[0])
    assert abs(log_squared_terms[-1]) > 10.0 * abs(log_squared_terms[0])

    with pytest.raises(ValueError, match="not resonant"):
        construct_stable_log_selector_chain(
            radial_decay=radial_decay,
            stable_rates=stable_rates,
            source_amplitudes={0: source_a, 1: source_b},
            selectors={2: selector_c},
            resonant_rows={
                2: (StableResonanceTerm(coupling_c, (1, 0, 0, 0, 0)),),
            },
        )
    with pytest.raises(ValueError, match="not triangular"):
        construct_stable_log_selector_chain(
            radial_decay=radial_decay,
            stable_rates=stable_rates,
            source_amplitudes={0: source_a},
            selectors={3: selector_d},
            resonant_rows={
                3: (StableResonanceTerm(coupling_d, (1, 1, 0, 0, 0)),),
            },
        )


def test_stable_log_selector_chain_projects_to_finite_fuchsian_log_branch():
    configuration, central_lambda = _central_configuration_data("equilateral")
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    masses = np.ones(3)
    radial_decay = 1.0
    stable_rates = (0.7, 0.9, 1.4, 1.6, 2.1)
    source_a = 0.041
    source_b = -0.029
    selector_c = 0.018
    selector_d = -0.016
    selector_e = 0.012
    coupling_c = 0.73
    coupling_d = -1.17
    coupling_e = 0.81

    chain = construct_stable_log_selector_chain(
        radial_decay=radial_decay,
        stable_rates=stable_rates,
        source_amplitudes={0: source_a, 1: source_b},
        selectors={2: selector_c, 3: selector_d, 4: selector_e},
        resonant_rows={
            2: (StableResonanceTerm(coupling_c, (2, 0, 0, 0, 0)),),
            3: (StableResonanceTerm(coupling_d, (1, 1, 0, 0, 0)),),
            4: (StableResonanceTerm(coupling_e, (1, 0, 1, 0, 0)),),
        },
    )
    mode_basis = _mass_orthonormal_complement_basis(
        masses,
        central_shape,
        count=len(stable_rates),
    )
    mode_shapes = {
        index: mode_basis[index]
        for index in range(len(stable_rates))
    }
    branch = construct_finite_fuchsian_log_branch_from_stable_chain(
        masses=masses,
        central_shape=central_shape,
        scale_coefficient=0.019,
        chain=chain,
        mode_shapes=mode_shapes,
    )
    identity = FiniteFuchsianLogContinuation(branch, branch)
    derived_identity = derive_identity_finite_fuchsian_log_continuation_from_incoming_branch(
        branch,
    )
    expected_selectors = {
        0: source_a,
        1: source_b,
        2: selector_c,
        3: selector_d,
        4: selector_e,
    }

    assert identity.identity_selector_certified
    assert derived_identity.identity_selector_certified
    assert [term.power for term in branch.terms] == pytest.approx(
        [chain.mode(index).fuchsian_power for index in sorted(chain.modes)],
        rel=1e-14,
        abs=1e-14,
    )
    np.testing.assert_allclose(
        derived_identity.outgoing.central_shape,
        branch.central_shape,
        atol=0.0,
        rtol=0.0,
    )
    energy_gaps = []
    angular_values = []
    for sigma in (2.0e-2, 1.0e-2, 4.0e-3):
        assert np.linalg.norm(branch.shape_projection_identity_residual(sigma), ord=np.inf) < 1e-14
        for term_position, mode_index in enumerate(sorted(chain.modes)):
            assert branch.recovered_selectors_for_term(term_position, sigma) == pytest.approx(
                (expected_selectors[mode_index],),
                rel=2e-6,
                abs=2e-6,
            )
    for sigma in (4.0e-3, 1.2e-3, 4.0e-4):
        state = branch.state_at_tau(sigma)
        assert np.linalg.norm(branch.shape_projection_identity_residual(sigma), ord=np.inf) < 1e-14
        energy_gaps.append(abs(energy(state, masses) - branch.finite_energy_limit))
        angular_values.append(abs(branch.centered_angular_momentum_scalar_at_tau(sigma)))

    assert identity.max_selector_gap_at_radius(1e-4) < 1e-12
    assert derived_identity.max_selector_gap_at_radius(1e-4) < 1e-12
    assert energy_gaps[2] < energy_gaps[1] < energy_gaps[0]
    assert energy_gaps[-1] < 5e-4
    assert angular_values[2] < angular_values[1] < angular_values[0]
    assert angular_values[-1] < 1e-5

    invalid_collision_branch = FiniteFuchsianLogBranch(
        masses=masses,
        central_shape=np.zeros_like(central_shape),
        scale_coefficient=0.019,
        terms=branch.terms,
    )
    with pytest.raises(ValueError, match="does not certify"):
        derive_identity_finite_fuchsian_log_continuation_from_incoming_branch(
            invalid_collision_branch,
        )

    with pytest.raises(ValueError, match="missing mode shape"):
        construct_finite_fuchsian_log_branch_from_stable_chain(
            masses=masses,
            central_shape=central_shape,
            scale_coefficient=0.019,
            chain=chain,
            mode_shapes={0: mode_basis[0]},
        )
    with pytest.raises(ValueError, match="mass-orthogonal"):
        construct_finite_fuchsian_log_branch_from_stable_chain(
            masses=masses,
            central_shape=central_shape,
            scale_coefficient=0.019,
            chain=chain,
            mode_shapes={index: central_shape for index in range(len(stable_rates))},
        )
    with pytest.raises(ValueError, match="greater than one"):
        construct_finite_fuchsian_log_branch_from_stable_chain(
            masses=masses,
            central_shape=central_shape,
            scale_coefficient=0.019,
            chain=chain,
            mode_shapes=mode_shapes,
            minimum_non_scale_power=1.5,
        )


def test_fuchsian_log_shape_row_solves_resonant_forcing_triangularly():
    configuration, central_lambda = _central_configuration_data("equilateral")
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    masses = np.ones(3)
    derivative_matrix = _linearized_acceleration_matrix(central_shape, masses)
    resonant_power = 0.5 * (-1.0 + np.sqrt(13.0))
    multiplier = (resonant_power + 2.0) * (resonant_power - 1.0)
    row_operator = multiplier * np.eye(6) - 9.0 * derivative_matrix

    resonant_mode = np.column_stack([central_shape[:, 0], -central_shape[:, 1]])
    resonant_mode /= np.sqrt(
        _mass_inner_product(masses, resonant_mode, resonant_mode)
    )
    scale_mode = central_shape / np.sqrt(
        _mass_inner_product(masses, central_shape, central_shape)
    )
    resonant_vector = resonant_mode.reshape(-1)
    scale_vector = scale_mode.reshape(-1)

    np.testing.assert_allclose(
        row_operator @ resonant_vector,
        np.zeros_like(resonant_vector),
        atol=3e-14,
    )
    assert np.linalg.norm(row_operator @ scale_vector, ord=np.inf) > 0.1

    resonant_forcing = 0.037
    range_forcing = -0.021
    selector = 0.014
    forcing = resonant_forcing * resonant_vector + range_forcing * scale_vector
    log_coefficient = resonant_forcing / (2.0 * resonant_power + 1.0) * resonant_vector
    range_coefficient = np.linalg.lstsq(
        row_operator,
        range_forcing * scale_vector,
        rcond=None,
    )[0]
    constant_coefficient = range_coefficient + selector * resonant_vector

    log_one_equation = row_operator @ log_coefficient
    log_zero_equation = (
        row_operator @ constant_coefficient
        + (2.0 * resonant_power + 1.0) * log_coefficient
    )

    assert np.linalg.norm(log_one_equation, ord=np.inf) < 3e-14
    np.testing.assert_allclose(log_zero_equation, forcing, rtol=1e-12, atol=1e-12)
    recovered_selector = _mass_inner_product(
        masses,
        constant_coefficient.reshape(3, 2) - range_coefficient.reshape(3, 2),
        resonant_mode,
    )
    assert recovered_selector == pytest.approx(selector, rel=1e-12, abs=1e-12)


def test_fuchsian_log_shape_projection_identity_and_zero_angular_limit():
    configuration, central_lambda = _central_configuration_data("equilateral")
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    masses = np.ones(3)
    mode = np.column_stack([-central_shape[:, 1], central_shape[:, 0]])
    mode /= np.sqrt(_mass_inner_product(masses, mode, mode))

    power = 0.5 * (-1.0 + np.sqrt(13.0))
    log_coefficient = 0.024
    constant_coefficient = -0.017

    def mass_wedge(left, right):
        return float(
            np.sum(masses * (left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]))
        )

    def shape_terms(tau):
        logarithm = np.log(tau)
        polynomial = log_coefficient * logarithm + constant_coefficient
        polynomial_prime = log_coefficient / tau
        polynomial_second = -log_coefficient / tau**2

        correction = tau**power * polynomial * mode
        correction_prime = (
            power * tau ** (power - 1.0) * polynomial
            + tau**power * polynomial_prime
        ) * mode
        correction_second = (
            power
            * (power - 1.0)
            * tau ** (power - 2.0)
            * polynomial
            + 2.0 * power * tau ** (power - 1.0) * polynomial_prime
            + tau**power * polynomial_second
        ) * mode
        return (
            central_shape + correction,
            correction_prime,
            correction_second,
        )

    projection_residual_errors = []
    angular_formula_errors = []
    angular_momentum_norms = []
    for tau in (0.04, 0.02, 0.01):
        shape, shape_prime, shape_second = shape_terms(tau)
        positions = tau**2 * shape
        physical_velocity = (2.0 / (3.0 * tau)) * shape + shape_prime / 3.0
        projected_acceleration = (
            tau**2 * shape_second + 2.0 * tau * shape_prime - 2.0 * shape
        ) / (9.0 * tau**4)
        projected_residual = projected_acceleration - accelerations(positions, masses)
        shape_residual = (
            tau**2 * shape_second
            + 2.0 * tau * shape_prime
            - 2.0 * shape
            - 9.0 * accelerations(shape, masses)
        )
        projection_residual_errors.append(
            np.linalg.norm(9.0 * tau**4 * projected_residual - shape_residual, ord=np.inf)
        )

        angular_momentum = mass_wedge(positions, physical_velocity)
        shape_angular_formula = (tau**2 / 3.0) * mass_wedge(shape, shape_prime)
        angular_formula_errors.append(abs(angular_momentum - shape_angular_formula))
        angular_momentum_norms.append(abs(angular_momentum))

    assert max(projection_residual_errors) < 2e-15
    assert max(angular_formula_errors) < 1e-17
    assert angular_momentum_norms[2] < angular_momentum_norms[1] < angular_momentum_norms[0]


def test_fuchsian_log_row_has_no_hidden_nonselector_coefficients():
    configuration, central_lambda = _central_configuration_data("equilateral")
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    masses = np.ones(3)
    derivative_matrix = _linearized_acceleration_matrix(central_shape, masses)
    resonant_power = 0.5 * (-1.0 + np.sqrt(13.0))
    row_operator = (
        (resonant_power + 2.0)
        * (resonant_power - 1.0)
        * np.eye(6)
        - 9.0 * derivative_matrix
    )

    kernel_mode = np.column_stack([central_shape[:, 0], -central_shape[:, 1]])
    kernel_mode /= np.sqrt(_mass_inner_product(masses, kernel_mode, kernel_mode))
    range_mode = np.column_stack([-central_shape[:, 1], central_shape[:, 0]])
    range_mode -= _mass_inner_product(masses, range_mode, kernel_mode) * kernel_mode
    range_mode -= (
        _mass_inner_product(masses, range_mode, central_shape)
        / _mass_inner_product(masses, central_shape, central_shape)
    ) * central_shape
    range_mode /= np.sqrt(_mass_inner_product(masses, range_mode, range_mode))
    kernel_vector = kernel_mode.reshape(-1)
    range_vector = range_mode.reshape(-1)

    forcing = 0.031 * kernel_vector - 0.019 * range_vector
    log_coefficient = 0.031 / (2.0 * resonant_power + 1.0) * kernel_vector
    range_part = np.linalg.lstsq(row_operator, -0.019 * range_vector, rcond=None)[0]
    selector_part = 0.012 * kernel_vector
    constant_coefficient = range_part + selector_part

    solved_log_residual = row_operator @ log_coefficient
    solved_constant_residual = (
        row_operator @ constant_coefficient
        + (2.0 * resonant_power + 1.0) * log_coefficient
        - forcing
    )
    perturbed_log_residual = row_operator @ (
        log_coefficient + 2.0e-3 * kernel_vector
    )
    perturbed_log_constant_residual = (
        row_operator @ constant_coefficient
        + (2.0 * resonant_power + 1.0)
        * (log_coefficient + 2.0e-3 * kernel_vector)
        - forcing
    )
    perturbed_range_residual = (
        row_operator @ (constant_coefficient + 2.0e-3 * range_vector)
        + (2.0 * resonant_power + 1.0) * log_coefficient
        - forcing
    )

    same_selector_constant = constant_coefficient + 3.0e-3 * range_vector
    same_selector_residual = (
        row_operator @ same_selector_constant
        + (2.0 * resonant_power + 1.0) * log_coefficient
        - forcing
    )
    recovered_selector_gap = _mass_inner_product(
        masses,
        (same_selector_constant - constant_coefficient).reshape(3, 2),
        kernel_mode,
    )
    changed_selector_constant = constant_coefficient + 2.0e-3 * kernel_vector
    changed_selector_residual = (
        row_operator @ changed_selector_constant
        + (2.0 * resonant_power + 1.0) * log_coefficient
        - forcing
    )
    changed_selector_gap = _mass_inner_product(
        masses,
        (changed_selector_constant - constant_coefficient).reshape(3, 2),
        kernel_mode,
    )

    assert np.linalg.norm(solved_log_residual, ord=np.inf) < 3e-14
    assert np.linalg.norm(solved_constant_residual, ord=np.inf) < 2e-12
    assert np.linalg.norm(perturbed_log_residual, ord=np.inf) < 3e-14
    assert np.linalg.norm(perturbed_log_constant_residual, ord=np.inf) > 1e-5
    assert np.linalg.norm(perturbed_range_residual, ord=np.inf) > 1e-5
    assert abs(recovered_selector_gap) < 1e-14
    assert np.linalg.norm(same_selector_residual, ord=np.inf) > 1e-5
    assert np.linalg.norm(changed_selector_residual, ord=np.inf) < 2e-12
    assert changed_selector_gap == pytest.approx(2.0e-3, rel=1e-12, abs=1e-12)


def test_fuchsian_log_row_constructor_builds_resonant_selector_branch():
    configuration, central_lambda = _central_configuration_data("equilateral")
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    masses = np.ones(3)
    derivative_matrix = _linearized_acceleration_matrix(central_shape, masses)
    resonant_power = 0.5 * (-1.0 + np.sqrt(13.0))
    row_operator = (
        (resonant_power + 2.0)
        * (resonant_power - 1.0)
        * np.eye(6)
        - 9.0 * derivative_matrix
    )

    kernel_mode = np.column_stack([central_shape[:, 0], -central_shape[:, 1]])
    kernel_mode /= np.sqrt(_mass_inner_product(masses, kernel_mode, kernel_mode))
    range_mode = np.column_stack([-central_shape[:, 1], central_shape[:, 0]])
    range_mode -= _mass_inner_product(masses, range_mode, kernel_mode) * kernel_mode
    range_mode -= (
        _mass_inner_product(masses, range_mode, central_shape)
        / _mass_inner_product(masses, central_shape, central_shape)
    ) * central_shape
    range_mode /= np.sqrt(_mass_inner_product(masses, range_mode, range_mode))
    forcing = 0.031 * kernel_mode - 0.019 * range_mode
    selector = 0.012

    solution = construct_fuchsian_log_row_solution(
        power=resonant_power,
        row_operator=row_operator,
        masses=masses,
        kernel_basis=(kernel_mode,),
        forcing_by_log_power={0: forcing},
        selector_by_basis=(selector,),
    )
    changed_selector_solution = construct_fuchsian_log_row_solution(
        power=resonant_power,
        row_operator=row_operator,
        masses=masses,
        kernel_basis=(kernel_mode,),
        forcing_by_log_power={0: forcing},
        selector_by_basis=(-0.017,),
    )

    expected_log_coefficient = 0.031 / (2.0 * resonant_power + 1.0) * kernel_mode
    expected_range_part = np.linalg.lstsq(
        row_operator,
        (-0.019 * range_mode).reshape(-1),
        rcond=None,
    )[0].reshape(3, 2)

    assert solution.row_certified
    assert solution.max_equation_residual_norm < 2e-12
    np.testing.assert_allclose(
        solution.coefficients_by_log_power[1],
        expected_log_coefficient,
        atol=2e-14,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        solution.coefficients_by_log_power[0] - selector * kernel_mode,
        expected_range_part,
        atol=2e-13,
        rtol=1e-12,
    )
    assert solution.recovered_selector_by_basis == pytest.approx(
        (selector,),
        rel=1e-12,
        abs=1e-12,
    )
    assert changed_selector_solution.recovered_selector_by_basis == pytest.approx(
        (-0.017,),
        rel=1e-12,
        abs=1e-12,
    )
    assert changed_selector_solution.max_equation_residual_norm < 2e-12

    branch = FuchsianLogBranch(
        masses=masses,
        central_shape=central_shape,
        row_solution=solution,
        scale_coefficient=0.027,
    )
    changed_branch = FuchsianLogBranch(
        masses=masses,
        central_shape=central_shape,
        row_solution=changed_selector_solution,
        scale_coefficient=0.027,
    )
    expected_energy = branch.finite_energy_limit
    energy_gaps = []
    angular_values = []
    for sigma in (3.0e-3, 1.0e-3, 3.0e-4):
        incoming_state = branch.state_at_tau(-sigma)
        outgoing_state = branch.state_at_tau(sigma)
        changed_state = changed_branch.state_at_tau(sigma)

        np.testing.assert_allclose(
            incoming_state[:6],
            outgoing_state[:6],
            atol=2e-18,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            incoming_state[6:],
            -outgoing_state[6:],
            atol=2e-12,
            rtol=1e-12,
        )
        assert np.linalg.norm(branch.shape_projection_identity_residual(sigma), ord=np.inf) < 5e-15
        assert np.linalg.norm(branch.shape_projection_identity_residual(-sigma), ord=np.inf) < 5e-15
        angular_values.append(abs(branch.centered_angular_momentum_scalar_at_tau(sigma)))
        assert abs(
            abs(branch.centered_angular_momentum_scalar_at_tau(sigma))
            - abs(branch.centered_angular_momentum_scalar_at_tau(-sigma))
        ) < 1e-20
        assert branch.recovered_selector_by_basis(sigma) == pytest.approx(
            (selector,),
            rel=2e-10,
            abs=2e-10,
        )
        assert changed_branch.recovered_selector_by_basis(sigma) == pytest.approx(
            (-0.017,),
            rel=2e-10,
            abs=2e-10,
        )

        energy_gaps.append(abs(energy(incoming_state, masses) - energy(changed_state, masses)))
        assert abs(energy(incoming_state, masses) - expected_energy) < 2e-4
        assert abs(energy(changed_state, masses) - expected_energy) < 2e-4

    assert energy_gaps[2] < energy_gaps[1] < energy_gaps[0]
    assert angular_values[2] < angular_values[1] < angular_values[0]
    assert angular_values[-1] < 3e-10


def test_fuchsian_log_selector_continuation_preserves_energy_limit():
    configuration, central_lambda = _central_configuration_data("equilateral")
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    masses = np.ones(3)
    resonant_mode = np.column_stack([central_shape[:, 0], -central_shape[:, 1]])
    resonant_mode /= np.sqrt(
        _mass_inner_product(masses, resonant_mode, resonant_mode)
    )

    resonant_power = 0.5 * (-1.0 + np.sqrt(13.0))
    scale_coefficient = 0.027
    forced_log_coefficient = 0.019
    incoming_selector = -0.031
    outgoing_selector = 0.020
    expected_energy = (
        (10.0 / 9.0)
        * scale_coefficient
        * _mass_inner_product(masses, central_shape, central_shape)
    )

    def branch_state(signed_tau, selector):
        sigma = abs(signed_tau)
        orientation = 1.0 if signed_tau > 0.0 else -1.0
        logarithm = np.log(sigma)
        polynomial = forced_log_coefficient * logarithm + selector
        shape = (
            central_shape
            + scale_coefficient * sigma**2 * central_shape
            + sigma**resonant_power * polynomial * resonant_mode
        )
        shape_sigma_derivative = (
            2.0 * scale_coefficient * sigma * central_shape
            + (
                resonant_power * sigma ** (resonant_power - 1.0) * polynomial
                + forced_log_coefficient * sigma ** (resonant_power - 1.0)
            )
            * resonant_mode
        )
        shape_tau_derivative = orientation * shape_sigma_derivative
        positions = signed_tau**2 * shape
        velocities = (
            (2.0 / (3.0 * signed_tau)) * shape + shape_tau_derivative / 3.0
        )
        return np.concatenate([positions.reshape(-1), velocities.reshape(-1)]), shape

    incoming_energy_errors = []
    outgoing_energy_errors = []
    energy_gaps = []
    for sigma in (3.0e-3, 1.0e-3, 3.0e-4):
        incoming_state, incoming_shape = branch_state(-sigma, incoming_selector)
        outgoing_state, outgoing_shape = branch_state(sigma, outgoing_selector)

        incoming_remainder = (
            incoming_shape
            - central_shape
            - scale_coefficient * sigma**2 * central_shape
        )
        outgoing_remainder = (
            outgoing_shape
            - central_shape
            - scale_coefficient * sigma**2 * central_shape
        )
        incoming_recovered_selector = (
            _mass_inner_product(masses, incoming_remainder, resonant_mode)
            / sigma**resonant_power
            - forced_log_coefficient * np.log(sigma)
        )
        outgoing_recovered_selector = (
            _mass_inner_product(masses, outgoing_remainder, resonant_mode)
            / sigma**resonant_power
            - forced_log_coefficient * np.log(sigma)
        )

        incoming_energy = energy(incoming_state, masses)
        outgoing_energy = energy(outgoing_state, masses)
        incoming_energy_errors.append(abs(incoming_energy - expected_energy))
        outgoing_energy_errors.append(abs(outgoing_energy - expected_energy))
        energy_gaps.append(abs(incoming_energy - outgoing_energy))

        assert incoming_recovered_selector == pytest.approx(
            incoming_selector, rel=2e-10, abs=2e-10
        )
        assert outgoing_recovered_selector == pytest.approx(
            outgoing_selector, rel=2e-10, abs=2e-10
        )
        assert abs(angular_momentum_z(incoming_state, masses)) < 1e-14
        assert abs(angular_momentum_z(outgoing_state, masses)) < 1e-14

    assert incoming_energy_errors[2] < incoming_energy_errors[1] < incoming_energy_errors[0]
    assert outgoing_energy_errors[2] < outgoing_energy_errors[1] < outgoing_energy_errors[0]
    assert energy_gaps[2] < energy_gaps[1] < energy_gaps[0]
    assert max(incoming_energy_errors[-1], outgoing_energy_errors[-1]) < 1.4e-4


def test_identity_selector_continuation_matches_lifted_branch_data():
    configuration, central_lambda = _central_configuration_data("equilateral")
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    masses = np.ones(3)
    selector_mode = np.column_stack([central_shape[:, 0], -central_shape[:, 1]])
    selector_mode /= np.sqrt(
        _mass_inner_product(masses, selector_mode, selector_mode)
    )
    transverse_mode = np.column_stack([-central_shape[:, 1], -central_shape[:, 0]])
    transverse_mode -= _mass_inner_product(masses, transverse_mode, selector_mode) * selector_mode
    transverse_mode -= _mass_inner_product(masses, transverse_mode, central_shape) * central_shape
    transverse_mode /= np.sqrt(
        _mass_inner_product(masses, transverse_mode, transverse_mode)
    )

    first_power = 0.5 * (-1.0 + np.sqrt(13.0))
    second_power = first_power + 0.7
    scale_coefficient = 0.021
    forced_log_coefficient = -0.014
    identity_selector = 0.026
    secondary_amplitude = -0.011

    def branch_state(signed_tau):
        sigma = abs(signed_tau)
        sign = 1.0 if signed_tau > 0.0 else -1.0
        logarithm = np.log(sigma)
        selector_polynomial = forced_log_coefficient * logarithm + identity_selector
        shape = (
            central_shape
            + scale_coefficient * sigma**2 * central_shape
            + sigma**first_power * selector_polynomial * selector_mode
            + secondary_amplitude * sigma**second_power * transverse_mode
        )
        shape_sigma_derivative = (
            2.0 * scale_coefficient * sigma * central_shape
            + (
                first_power * sigma ** (first_power - 1.0) * selector_polynomial
                + forced_log_coefficient * sigma ** (first_power - 1.0)
            )
            * selector_mode
            + secondary_amplitude
            * second_power
            * sigma ** (second_power - 1.0)
            * transverse_mode
        )
        shape_tau_derivative = sign * shape_sigma_derivative
        positions = signed_tau**2 * shape
        velocities = (
            (2.0 / (3.0 * signed_tau)) * shape + shape_tau_derivative / 3.0
        )
        state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
        remainder = shape - central_shape - scale_coefficient * sigma**2 * central_shape
        recovered_selector = (
            _mass_inner_product(masses, remainder, selector_mode)
            / sigma**first_power
            - forced_log_coefficient * logarithm
        )
        recovered_secondary = (
            _mass_inner_product(masses, remainder, transverse_mode)
            / sigma**second_power
        )
        return state, recovered_selector, recovered_secondary

    energy_gaps = []
    selector_gaps = []
    secondary_gaps = []
    for sigma in (5.0e-3, 1.5e-3, 5.0e-4):
        incoming_state, incoming_selector, incoming_secondary = branch_state(-sigma)
        outgoing_state, outgoing_selector, outgoing_secondary = branch_state(sigma)
        incoming_positions = incoming_state[:6]
        incoming_velocities = incoming_state[6:]
        outgoing_positions = outgoing_state[:6]
        outgoing_velocities = outgoing_state[6:]

        np.testing.assert_allclose(incoming_positions, outgoing_positions, atol=1e-18)
        np.testing.assert_allclose(incoming_velocities, -outgoing_velocities, rtol=1e-12, atol=1e-12)
        assert abs(angular_momentum_z(incoming_state, masses)) < 1e-13
        assert abs(angular_momentum_z(outgoing_state, masses)) < 1e-13

        selector_gaps.append(abs(incoming_selector - outgoing_selector))
        secondary_gaps.append(abs(incoming_secondary - outgoing_secondary))
        energy_gaps.append(abs(energy(incoming_state, masses) - energy(outgoing_state, masses)))

        assert incoming_selector == pytest.approx(identity_selector, rel=3e-10, abs=3e-10)
        assert outgoing_selector == pytest.approx(identity_selector, rel=3e-10, abs=3e-10)
        assert incoming_secondary == pytest.approx(secondary_amplitude, rel=3e-10, abs=3e-10)
        assert outgoing_secondary == pytest.approx(secondary_amplitude, rel=3e-10, abs=3e-10)

    assert max(selector_gaps) < 1e-14
    assert max(secondary_gaps) < 1e-14
    assert max(energy_gaps) < 1e-12


def test_finite_fuchsian_log_branch_composes_selector_rows_and_projects():
    configuration, central_lambda = _central_configuration_data("equilateral")
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    masses = np.ones(3)
    selector_mode = np.column_stack([central_shape[:, 0], -central_shape[:, 1]])
    selector_mode /= np.sqrt(
        _mass_inner_product(masses, selector_mode, selector_mode)
    )
    transverse_mode = np.column_stack([-central_shape[:, 1], -central_shape[:, 0]])
    transverse_mode -= _mass_inner_product(masses, transverse_mode, selector_mode) * selector_mode
    transverse_mode -= _mass_inner_product(masses, transverse_mode, central_shape) * central_shape
    transverse_mode /= np.sqrt(
        _mass_inner_product(masses, transverse_mode, transverse_mode)
    )

    first_power = 0.5 * (-1.0 + np.sqrt(13.0))
    second_power = first_power + 0.7
    scale_coefficient = 0.021
    forced_log_coefficient = -0.014
    identity_selector = 0.026
    secondary_amplitude = -0.011

    incoming = FiniteFuchsianLogBranch(
        masses=masses,
        central_shape=central_shape,
        scale_coefficient=scale_coefficient,
        terms=(
            FuchsianLogTerm(
                power=first_power,
                coefficients_by_log_power={
                    1: forced_log_coefficient * selector_mode,
                    0: identity_selector * selector_mode,
                },
                selector_basis=(selector_mode,),
            ),
            FuchsianLogTerm(
                power=second_power,
                coefficients_by_log_power={0: secondary_amplitude * transverse_mode},
                selector_basis=(transverse_mode,),
            ),
        ),
    )
    outgoing_identity = FiniteFuchsianLogBranch(
        masses=masses,
        central_shape=central_shape,
        scale_coefficient=scale_coefficient,
        terms=incoming.terms,
    )
    outgoing_changed = FiniteFuchsianLogBranch(
        masses=masses,
        central_shape=central_shape,
        scale_coefficient=scale_coefficient,
        terms=(
            FuchsianLogTerm(
                power=first_power,
                coefficients_by_log_power={
                    1: forced_log_coefficient * selector_mode,
                    0: -0.019 * selector_mode,
                },
                selector_basis=(selector_mode,),
            ),
            FuchsianLogTerm(
                power=second_power,
                coefficients_by_log_power={0: 0.017 * transverse_mode},
                selector_basis=(transverse_mode,),
            ),
        ),
    )
    identity_continuation = FiniteFuchsianLogContinuation(
        incoming=incoming,
        outgoing=outgoing_identity,
    )
    changed_continuation = FiniteFuchsianLogContinuation(
        incoming=incoming,
        outgoing=outgoing_changed,
    )
    derived_identity = derive_identity_finite_fuchsian_log_continuation_from_incoming_branch(
        incoming,
    )
    outgoing_energy_changed = FiniteFuchsianLogBranch(
        masses=masses,
        central_shape=central_shape,
        scale_coefficient=scale_coefficient + 0.003,
        terms=incoming.terms,
    )
    energy_changed_continuation = FiniteFuchsianLogContinuation(
        incoming=incoming,
        outgoing=outgoing_energy_changed,
    )

    assert identity_continuation.identity_selector_certified
    assert derived_identity.identity_selector_certified
    assert not changed_continuation.identity_selector_certified
    assert not energy_changed_continuation.identity_selector_certified
    assert changed_continuation.energy_gap < 1e-12
    assert energy_changed_continuation.energy_gap > 1e-3

    energy_gaps = []
    angular_values = []
    for sigma in (5.0e-3, 1.5e-3, 5.0e-4):
        incoming_state = incoming.state_at_tau(-sigma)
        outgoing_state = outgoing_identity.state_at_tau(sigma)
        derived_state = derived_identity.outgoing.state_at_tau(sigma)
        changed_state = outgoing_changed.state_at_tau(sigma)

        np.testing.assert_allclose(incoming_state[:6], outgoing_state[:6], atol=2e-18)
        np.testing.assert_allclose(incoming_state[:6], derived_state[:6], atol=2e-18)
        np.testing.assert_allclose(incoming_state[6:], -outgoing_state[6:], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(incoming_state[6:], -derived_state[6:], rtol=1e-12, atol=1e-12)
        assert np.linalg.norm(incoming.shape_projection_identity_residual(-sigma), ord=np.inf) < 5e-15
        assert np.linalg.norm(outgoing_identity.shape_projection_identity_residual(sigma), ord=np.inf) < 5e-15
        assert (
            np.linalg.norm(
                derived_identity.outgoing.shape_projection_identity_residual(sigma),
                ord=np.inf,
            )
            < 5e-15
        )

        assert incoming.recovered_selectors_for_term(0, sigma) == pytest.approx(
            (identity_selector,),
            rel=3e-10,
            abs=3e-10,
        )
        assert incoming.recovered_selectors_for_term(1, sigma) == pytest.approx(
            (secondary_amplitude,),
            rel=3e-10,
            abs=3e-10,
        )
        assert outgoing_changed.recovered_selectors_for_term(0, sigma) == pytest.approx(
            (-0.019,),
            rel=3e-10,
            abs=3e-10,
        )
        assert outgoing_changed.recovered_selectors_for_term(1, sigma) == pytest.approx(
            (0.017,),
            rel=3e-10,
            abs=3e-10,
        )

        energy_gaps.append(abs(energy(incoming_state, masses) - energy(changed_state, masses)))
        angular_values.append(abs(incoming.centered_angular_momentum_scalar_at_tau(sigma)))

    assert identity_continuation.max_selector_gap_at_radius(1e-4) < 1e-12
    assert derived_identity.max_selector_gap_at_radius(1e-4) < 1e-12
    assert changed_continuation.max_selector_gap_at_radius(1e-4) > 0.02
    assert energy_gaps[2] < energy_gaps[1] < energy_gaps[0]
    assert energy_gaps[-1] < 6e-5
    assert angular_values[2] < angular_values[1] < angular_values[0]
    assert angular_values[-1] < 2e-7


def test_identity_selector_total_collision_glues_to_ordinary_taylor_pieces():
    masses = np.ones(3)
    compact_windows = (
        (-1.0e-3, -9.0e-4),
        (-8.0e-4, -7.0e-4),
        (7.0e-4, 8.0e-4),
        (9.0e-4, 1.0e-3),
    )
    local_residuals = []
    angular_values = []
    energy_values = []

    for start_time, end_time in compact_windows:
        start_positions, start_velocities, *_ = _parabolic_homothetic_state(
            "equilateral",
            start_time,
        )
        end_positions, end_velocities, *_ = _parabolic_homothetic_state(
            "equilateral",
            end_time,
        )
        ordinary_solution = construct_taylor_solution(
            start_positions,
            start_velocities,
            masses,
            order=32,
        )
        ordinary_positions = ordinary_solution.positions_at(end_time - start_time)
        ordinary_velocities = ordinary_solution.velocities_at(end_time - start_time)
        start_state = np.concatenate(
            [start_positions.reshape(-1), start_velocities.reshape(-1)]
        )
        end_state = np.concatenate([end_positions.reshape(-1), end_velocities.reshape(-1)])

        assert np.sign(start_time) == np.sign(end_time)
        assert min(
            np.linalg.norm(start_positions[j] - start_positions[i])
            for i in range(3)
            for j in range(i + 1, 3)
        ) > 0.0
        local_residuals.append(
            max(
                np.linalg.norm(ordinary_positions - end_positions, ord=np.inf),
                np.linalg.norm(ordinary_velocities - end_velocities, ord=np.inf),
            )
        )
        angular_values.append(abs(angular_momentum_z(start_state, masses)))
        angular_values.append(abs(angular_momentum_z(end_state, masses)))
        energy_values.append(abs(energy(start_state, masses)))
        energy_values.append(abs(energy(end_state, masses)))

    incoming_positions, incoming_velocities, *_ = _parabolic_homothetic_state(
        "equilateral",
        -1.0e-3,
    )
    outgoing_positions, outgoing_velocities, *_ = _parabolic_homothetic_state(
        "equilateral",
        1.0e-3,
    )

    np.testing.assert_allclose(incoming_positions, outgoing_positions, atol=1.0e-18)
    np.testing.assert_allclose(incoming_velocities, -outgoing_velocities, atol=1.0e-12)
    assert max(local_residuals) < 5.0e-12
    assert sum(local_residuals) < 2.0e-11
    assert max(angular_values) < 1.0e-14
    assert max(energy_values) < 1.0e-12


def test_collision_free_shape_limit_supplies_normalized_potential_limit():
    masses, configuration, _central_lambda = _arbitrary_mass_equilateral_central_configuration()
    shape_inertia = float(np.sum(masses[:, None] * configuration**2))
    normalized_limit_shape = configuration / np.sqrt(shape_inertia)
    perturbation = np.array(
        [
            [0.12, -0.04],
            [-0.03, 0.08],
            [0.02, -0.05],
        ]
    )
    perturbation -= np.average(perturbation, axis=0, weights=masses)
    perturbation -= (
        float(np.sum(masses[:, None] * perturbation * normalized_limit_shape))
        * normalized_limit_shape
    )

    gamma = 0.0
    for first in range(3):
        for second in range(first + 1, 3):
            gamma += masses[first] * masses[second] / np.linalg.norm(
                normalized_limit_shape[first] - normalized_limit_shape[second]
            )
    scale_coefficient = ((9.0 / 2.0) * gamma) ** (1.0 / 3.0)
    limiting_second_shape = scale_coefficient * normalized_limit_shape

    normalized_potential_errors = []
    second_shape_errors = []
    pair_floors = []
    for collision_gap in (1e-2, 2.5e-3, 6.25e-4):
        shape_error = collision_gap ** 0.25
        normalized_shape = normalized_limit_shape + shape_error * perturbation
        normalized_shape -= np.average(normalized_shape, axis=0, weights=masses)
        normalized_shape /= np.sqrt(
            float(np.sum(masses[:, None] * normalized_shape**2))
        )
        radius = scale_coefficient * collision_gap ** (2.0 / 3.0)
        positions = radius * normalized_shape
        inertia = float(np.sum(masses[:, None] * positions**2))
        potential = 0.0
        pair_floor = np.inf
        for first in range(3):
            for second in range(first + 1, 3):
                pair_distance = np.linalg.norm(positions[first] - positions[second])
                normalized_pair_distance = pair_distance / np.sqrt(inertia)
                pair_floor = min(pair_floor, normalized_pair_distance)
                potential += masses[first] * masses[second] / pair_distance

        normalized_potential_errors.append(abs(potential * np.sqrt(inertia) - gamma))
        second_shape_errors.append(
            np.linalg.norm(
                positions / collision_gap ** (2.0 / 3.0) - limiting_second_shape,
                ord=np.inf,
            )
        )
        pair_floors.append(pair_floor)

    assert min(pair_floors) > 0.5
    assert normalized_potential_errors[1] < 0.72 * normalized_potential_errors[0]
    assert normalized_potential_errors[2] < 0.72 * normalized_potential_errors[1]
    assert second_shape_errors[1] < 0.72 * second_shape_errors[0]
    assert second_shape_errors[2] < 0.72 * second_shape_errors[1]


def test_collision_free_shape_convergence_forces_central_limit_without_c2_remainder():
    masses, configuration, _central_lambda = _arbitrary_mass_equilateral_central_configuration()
    shape_inertia = float(np.sum(masses[:, None] * configuration**2))
    normalized_limit_shape = configuration / np.sqrt(shape_inertia)
    gamma = 0.0
    for first in range(3):
        for second in range(first + 1, 3):
            gamma += masses[first] * masses[second] / np.linalg.norm(
                normalized_limit_shape[first] - normalized_limit_shape[second]
            )
    radius_coefficient = ((9.0 / 2.0) * gamma) ** (1.0 / 3.0)
    limiting_second_shape = radius_coefficient * normalized_limit_shape

    np.testing.assert_allclose(
        accelerations(normalized_limit_shape, masses),
        -gamma * normalized_limit_shape,
        rtol=1e-13,
        atol=1e-13,
    )

    perturbation = np.array(
        [
            [0.10, -0.03],
            [-0.02, 0.09],
            [0.04, -0.05],
        ]
    )
    perturbation -= np.average(perturbation, axis=0, weights=masses)
    perturbation -= (
        float(np.sum(masses[:, None] * perturbation * normalized_limit_shape))
        * normalized_limit_shape
    )
    perturbation /= np.sqrt(float(np.sum(masses[:, None] * perturbation**2)))

    central_coefficient_errors = []
    for time_to_collision in (1e-3, 1e-5, 1e-7):
        shape_error = time_to_collision ** 0.25
        normalized_shape = normalized_limit_shape + shape_error * perturbation
        normalized_shape -= np.average(normalized_shape, axis=0, weights=masses)
        normalized_shape /= np.sqrt(
            float(np.sum(masses[:, None] * normalized_shape**2))
        )
        positions = radius_coefficient * time_to_collision ** (2.0 / 3.0) * normalized_shape
        scaled_acceleration = time_to_collision ** (4.0 / 3.0) * accelerations(
            positions,
            masses,
        )
        second_shape_from_acceleration = -4.5 * scaled_acceleration

        central_coefficient_errors.append(
            np.linalg.norm(
                second_shape_from_acceleration - limiting_second_shape,
                ord=np.inf,
            )
        )

    assert central_coefficient_errors[1] < 0.75 * central_coefficient_errors[0]
    assert central_coefficient_errors[2] < 0.75 * central_coefficient_errors[1]
    assert central_coefficient_errors[2] < 0.04

    noncentral_shape = normalized_limit_shape + 0.25 * perturbation
    noncentral_shape -= np.average(noncentral_shape, axis=0, weights=masses)
    noncentral_shape /= np.sqrt(float(np.sum(masses[:, None] * noncentral_shape**2)))
    noncentral_gamma = 0.0
    for first in range(3):
        for second in range(first + 1, 3):
            noncentral_gamma += masses[first] * masses[second] / np.linalg.norm(
                noncentral_shape[first] - noncentral_shape[second]
            )
    noncentral_radius_coefficient = ((9.0 / 2.0) * noncentral_gamma) ** (1.0 / 3.0)
    noncentral_second_shape = noncentral_radius_coefficient * noncentral_shape
    noncentral_acceleration_coefficient = (
        accelerations(noncentral_shape, masses) / noncentral_radius_coefficient**2
    )
    noncentral_second_shape_from_acceleration = -4.5 * noncentral_acceleration_coefficient

    assert (
        np.linalg.norm(
            noncentral_second_shape_from_acceleration - noncentral_second_shape,
            ord=np.inf,
        )
        > 0.05
    )


def test_collision_free_shape_convergence_supplies_c1_cubic_time_entry():
    masses, configuration, _central_lambda = _arbitrary_mass_equilateral_central_configuration()
    shape_inertia = float(np.sum(masses[:, None] * configuration**2))
    normalized_limit_shape = configuration / np.sqrt(shape_inertia)
    gamma = 0.0
    for first in range(3):
        for second in range(first + 1, 3):
            gamma += masses[first] * masses[second] / np.linalg.norm(
                normalized_limit_shape[first] - normalized_limit_shape[second]
            )
    radius_coefficient = ((9.0 / 2.0) * gamma) ** (1.0 / 3.0)
    central_second_shape = radius_coefficient * normalized_limit_shape

    perturbation = np.array(
        [
            [0.07, 0.02],
            [-0.03, 0.05],
            [0.01, -0.04],
        ]
    )
    perturbation -= np.average(perturbation, axis=0, weights=masses)
    perturbation -= (
        float(np.sum(masses[:, None] * perturbation * central_second_shape))
        / float(np.sum(masses[:, None] * central_second_shape**2))
        * central_second_shape
    )
    perturbation /= np.sqrt(float(np.sum(masses[:, None] * perturbation**2)))
    perturbation *= 0.15
    perturbation_power = 0.8

    derivative_errors = []
    acceleration_inferred_errors = []
    position_coefficient_errors = []
    for tau in (-1.0e-2, -1.0e-3, -1.0e-4):
        distance_to_collision = -tau
        positions = (
            central_second_shape * tau**2
            + perturbation * distance_to_collision ** (2.0 + perturbation_power)
        )
        regularized_derivative = (
            2.0 * central_second_shape * tau
            - (2.0 + perturbation_power)
            * perturbation
            * distance_to_collision ** (1.0 + perturbation_power)
        )
        position_coefficient = positions / tau**2
        derivative_coefficient = regularized_derivative / tau
        scaled_acceleration = tau**4 * accelerations(positions, masses)
        derivative_coefficient_from_acceleration = -9.0 * scaled_acceleration

        position_coefficient_errors.append(
            np.linalg.norm(position_coefficient - central_second_shape, ord=np.inf)
        )
        derivative_errors.append(
            np.linalg.norm(
                derivative_coefficient - 2.0 * central_second_shape,
                ord=np.inf,
            )
        )
        acceleration_inferred_errors.append(
            np.linalg.norm(
                derivative_coefficient_from_acceleration - 2.0 * central_second_shape,
                ord=np.inf,
            )
        )

    assert position_coefficient_errors[1] < 0.18 * position_coefficient_errors[0]
    assert position_coefficient_errors[2] < 0.18 * position_coefficient_errors[1]
    assert derivative_errors[1] < 0.18 * derivative_errors[0]
    assert derivative_errors[2] < 0.18 * derivative_errors[1]
    assert acceleration_inferred_errors[1] < 0.23 * acceleration_inferred_errors[0]
    assert acceleration_inferred_errors[2] < 0.23 * acceleration_inferred_errors[1]
    assert derivative_errors[-1] < 3.0e-4
    assert acceleration_inferred_errors[-1] < 3.0e-3

    noncentral_second_shape = central_second_shape + perturbation
    noncentral_mismatch = np.linalg.norm(
        -9.0 * accelerations(noncentral_second_shape, masses)
        - 2.0 * noncentral_second_shape,
        ord=np.inf,
    )
    assert noncentral_mismatch > 0.05


def test_collision_free_shape_convergence_supplies_c2_cubic_time_entry():
    masses, configuration, _central_lambda = _arbitrary_mass_equilateral_central_configuration()
    shape_inertia = float(np.sum(masses[:, None] * configuration**2))
    normalized_limit_shape = configuration / np.sqrt(shape_inertia)
    gamma = 0.0
    for first in range(3):
        for second in range(first + 1, 3):
            gamma += masses[first] * masses[second] / np.linalg.norm(
                normalized_limit_shape[first] - normalized_limit_shape[second]
            )
    radius_coefficient = ((9.0 / 2.0) * gamma) ** (1.0 / 3.0)
    central_second_shape = radius_coefficient * normalized_limit_shape

    perturbation = np.array(
        [
            [0.05, -0.01],
            [-0.02, 0.06],
            [0.03, -0.04],
        ]
    )
    perturbation -= np.average(perturbation, axis=0, weights=masses)
    perturbation -= (
        float(np.sum(masses[:, None] * perturbation * central_second_shape))
        / float(np.sum(masses[:, None] * central_second_shape**2))
        * central_second_shape
    )
    perturbation /= np.sqrt(float(np.sum(masses[:, None] * perturbation**2)))
    perturbation *= 0.12
    perturbation_power = 0.9

    direct_second_derivative_errors = []
    chain_rule_second_derivative_errors = []
    remainder_second_derivative_errors = []
    for tau in (-1.0e-2, -1.0e-3, -1.0e-4):
        distance_to_collision = -tau
        positions = (
            central_second_shape * tau**2
            + perturbation * distance_to_collision ** (2.0 + perturbation_power)
        )
        regularized_derivative = (
            2.0 * central_second_shape * tau
            - (2.0 + perturbation_power)
            * perturbation
            * distance_to_collision ** (1.0 + perturbation_power)
        )
        regularized_second_derivative = (
            2.0 * central_second_shape
            + (2.0 + perturbation_power)
            * (1.0 + perturbation_power)
            * perturbation
            * distance_to_collision**perturbation_power
        )
        chain_rule_second_derivative = (
            2.0 * regularized_derivative / tau
            + 9.0 * tau**4 * accelerations(positions, masses)
        )

        direct_second_derivative_errors.append(
            np.linalg.norm(
                regularized_second_derivative - 2.0 * central_second_shape,
                ord=np.inf,
            )
        )
        chain_rule_second_derivative_errors.append(
            np.linalg.norm(
                chain_rule_second_derivative - 2.0 * central_second_shape,
                ord=np.inf,
            )
        )
        remainder_second_derivative_errors.append(
            np.linalg.norm(
                regularized_second_derivative - 2.0 * positions / tau**2,
                ord=np.inf,
            )
        )

    assert direct_second_derivative_errors[1] < 0.21 * direct_second_derivative_errors[0]
    assert direct_second_derivative_errors[2] < 0.21 * direct_second_derivative_errors[1]
    assert chain_rule_second_derivative_errors[1] < 0.27 * chain_rule_second_derivative_errors[0]
    assert chain_rule_second_derivative_errors[2] < 0.27 * chain_rule_second_derivative_errors[1]
    assert remainder_second_derivative_errors[1] < 0.28 * remainder_second_derivative_errors[0]
    assert remainder_second_derivative_errors[2] < 0.28 * remainder_second_derivative_errors[1]
    assert direct_second_derivative_errors[-1] < 3.0e-4
    assert chain_rule_second_derivative_errors[-1] < 4.0e-3
    assert remainder_second_derivative_errors[-1] < 8.0e-4

    noncentral_second_shape = central_second_shape + perturbation
    noncentral_chain_rule_limit = (
        4.0 * noncentral_second_shape
        + 9.0 * accelerations(noncentral_second_shape, masses)
    )
    assert (
        np.linalg.norm(
            noncentral_chain_rule_limit - 2.0 * noncentral_second_shape,
            ord=np.inf,
        )
        > 0.05
    )


def test_c2_cubic_time_asymptotic_forces_central_limiting_shape():
    configuration, central_lambda = _central_configuration_data("equilateral")
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    central_quadratic = scale_factor * configuration
    noncentral_quadratic = central_quadratic + np.array(
        [
            [0.08, 0.01],
            [-0.02, 0.03],
            [0.01, -0.06],
        ]
    )
    c2_remainder_shape = np.array(
        [
            [0.11, -0.04],
            [-0.07, 0.09],
            [0.02, -0.03],
        ]
    )
    c2_remainder_shape -= np.average(c2_remainder_shape, axis=0)
    fractional_power = 2.5

    def residual_limit(quadratic_coefficient, regularized_time):
        positions = (
            quadratic_coefficient * regularized_time**2
            + c2_remainder_shape * regularized_time**fractional_power
        )
        first_derivative = (
            2.0 * quadratic_coefficient * regularized_time
            + fractional_power
            * c2_remainder_shape
            * regularized_time ** (fractional_power - 1.0)
        )
        second_derivative = (
            2.0 * quadratic_coefficient
            + fractional_power
            * (fractional_power - 1.0)
            * c2_remainder_shape
            * regularized_time ** (fractional_power - 2.0)
        )
        physical_acceleration = second_derivative / (9.0 * regularized_time**4) - (
            2.0 * first_derivative
        ) / (9.0 * regularized_time**5)
        return regularized_time**4 * (
            physical_acceleration - accelerations(positions)
        )

    central_residuals = []
    noncentral_residuals = []
    expected_noncentral_obstruction = -(
        accelerations(noncentral_quadratic) + (2.0 / 9.0) * noncentral_quadratic
    )
    for regularized_time in (1e-4, 2.5e-5, 6.25e-6):
        central_residuals.append(
            np.linalg.norm(residual_limit(central_quadratic, regularized_time), ord=np.inf)
        )
        noncentral_residuals.append(
            np.linalg.norm(
                residual_limit(noncentral_quadratic, regularized_time)
                - expected_noncentral_obstruction,
                ord=np.inf,
            )
        )

    assert central_residuals[1] < 0.51 * central_residuals[0]
    assert central_residuals[2] < 0.51 * central_residuals[1]
    assert noncentral_residuals[1] < 0.51 * noncentral_residuals[0]
    assert noncentral_residuals[2] < 0.51 * noncentral_residuals[1]
    assert np.linalg.norm(expected_noncentral_obstruction, ord=np.inf) > 0.01


def test_noninteger_indicial_shape_mode_blocks_automatic_finite_selector_jets():
    configuration, central_lambda = _central_configuration_data("equilateral")
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    central_shape = scale_factor * configuration
    masses = np.ones(3)
    shape_mode = np.column_stack([central_shape[:, 0], -central_shape[:, 1]])
    shape_mode /= np.sqrt(_mass_inner_product(masses, shape_mode, shape_mode))
    derivative = _acceleration_derivative_apply(central_shape, masses, shape_mode)
    shape_eigenvalue = 1.0 / 9.0
    indicial_power = 0.5 * (-1.0 + np.sqrt(13.0))
    wrong_power = indicial_power + 0.4
    amplitude = 1.0e-3

    np.testing.assert_allclose(
        derivative,
        shape_eigenvalue * shape_mode,
        rtol=1e-12,
        atol=1e-12,
    )
    assert 1.0 < indicial_power < 2.0
    assert (indicial_power + 2.0) * (indicial_power - 1.0) == pytest.approx(
        9.0 * shape_eigenvalue,
    )

    def shape_equation_residual(power, tau):
        shape = central_shape + amplitude * tau**power * shape_mode
        first_derivative = amplitude * power * tau ** (power - 1.0) * shape_mode
        second_derivative = (
            amplitude * power * (power - 1.0) * tau ** (power - 2.0) * shape_mode
        )
        return (
            tau**2 * second_derivative
            + 2.0 * tau * first_derivative
            - 2.0 * shape
            - 9.0 * accelerations(shape, masses)
        )

    indicial_scaled_residuals = []
    wrong_scaled_residuals = []
    cubic_quotients = []
    quartic_quotients = []
    for tau in (1.0e-2, 3.0e-3, 1.0e-3):
        indicial_scaled_residuals.append(
            np.linalg.norm(
                shape_equation_residual(indicial_power, tau)
                / (amplitude * tau**indicial_power),
                ord=np.inf,
            )
        )
        wrong_scaled_residuals.append(
            np.linalg.norm(
                shape_equation_residual(wrong_power, tau) / (amplitude * tau**wrong_power),
                ord=np.inf,
            )
        )
        positions = central_shape * tau**2 + amplitude * tau ** (2.0 + indicial_power) * shape_mode
        cubic_quotients.append(
            np.linalg.norm((positions - central_shape * tau**2) / tau**3, ord=np.inf)
        )
        quartic_quotients.append(
            np.linalg.norm((positions - central_shape * tau**2) / tau**4, ord=np.inf)
        )

    assert indicial_scaled_residuals[1] < 0.45 * indicial_scaled_residuals[0]
    assert indicial_scaled_residuals[2] < 0.45 * indicial_scaled_residuals[1]
    assert indicial_scaled_residuals[-1] < 2.0e-6
    assert min(wrong_scaled_residuals) > 0.3
    assert cubic_quotients[0] > cubic_quotients[1] > cubic_quotients[2]
    assert quartic_quotients[0] < quartic_quotients[1] < quartic_quotients[2]


def test_fractional_fuchsian_shape_recurrence_constructs_projected_branch():
    configuration, central_lambda = _central_configuration_data("equilateral")
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    central_shape = scale_factor * configuration
    masses = np.ones(3)
    shape_mode = np.column_stack([central_shape[:, 0], -central_shape[:, 1]])
    shape_mode /= np.sqrt(_mass_inner_product(masses, shape_mode, shape_mode))
    derivative_matrix = _linearized_acceleration_matrix(central_shape, masses)
    indicial_power = 0.5 * (-1.0 + np.sqrt(13.0))
    amplitude = 0.06
    order = 5
    coefficients = np.zeros((order + 1, 3, 2))
    coefficients[0] = central_shape
    coefficients[1] = amplitude * shape_mode

    for degree in range(2, order + 1):
        trial_coefficients = coefficients.copy()
        trial_coefficients[degree] = 0.0
        known_term = _shape_acceleration_series(
            trial_coefficients,
            masses,
            degree,
        )[degree].reshape(-1)
        multiplier = (
            (degree * indicial_power + 2.0)
            * (degree * indicial_power - 1.0)
            / 9.0
        )
        operator = multiplier * np.eye(6) - derivative_matrix
        solved, *_ = np.linalg.lstsq(operator, known_term, rcond=None)
        coefficients[degree] = solved.reshape(3, 2)

        recurrence_residual = (
            operator @ coefficients[degree].reshape(-1) - known_term
        )
        assert np.linalg.norm(recurrence_residual, ord=np.inf) < 1e-12
        assert min(abs(multiplier - eigenvalue) for eigenvalue in (0.0, 4.0 / 9.0, -2.0 / 9.0, 1.0 / 9.0)) > 0.02

    def fuchsian_shape_residual(retained_order, tau):
        shape = sum(
            coefficients[degree] * tau ** (degree * indicial_power)
            for degree in range(retained_order + 1)
        )
        first_derivative = sum(
            degree
            * indicial_power
            * coefficients[degree]
            * tau ** (degree * indicial_power - 1.0)
            for degree in range(1, retained_order + 1)
        )
        second_derivative = sum(
            degree
            * indicial_power
            * (degree * indicial_power - 1.0)
            * coefficients[degree]
            * tau ** (degree * indicial_power - 2.0)
            for degree in range(1, retained_order + 1)
        )
        shape_residual = (
            tau**2 * second_derivative
            + 2.0 * tau * first_derivative
            - 2.0 * shape
            - 9.0 * accelerations(shape, masses)
        )
        return shape_residual, shape_residual / 9.0

    tau = 0.035
    residual_norms = []
    projected_scaled_residuals = []
    for retained_order in (1, 2, 3, 5):
        shape_residual, projected_scaled_residual = fuchsian_shape_residual(
            retained_order,
            tau,
        )
        residual_norms.append(np.linalg.norm(shape_residual, ord=np.inf))
        projected_scaled_residuals.append(
            np.linalg.norm(projected_scaled_residual, ord=np.inf)
        )

    assert residual_norms[1] < 0.02 * residual_norms[0]
    assert residual_norms[2] < 0.05 * residual_norms[1]
    assert residual_norms[3] < 0.02 * residual_norms[2]
    assert projected_scaled_residuals[-1] < 1e-11


def test_fractional_fuchsian_full_eigenspace_projects_on_both_collision_sides():
    configuration, central_lambda = _central_configuration_data("equilateral")
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    central_shape = scale_factor * configuration
    masses = np.ones(3)
    fractional_basis_one = np.column_stack(
        [central_shape[:, 0], -central_shape[:, 1]]
    )
    fractional_basis_one /= np.sqrt(
        _mass_inner_product(masses, fractional_basis_one, fractional_basis_one)
    )
    fractional_basis_two = np.column_stack(
        [central_shape[:, 1], central_shape[:, 0]]
    )
    fractional_basis_two -= (
        _mass_inner_product(masses, fractional_basis_one, fractional_basis_two)
        * fractional_basis_one
    )
    fractional_basis_two /= np.sqrt(
        _mass_inner_product(masses, fractional_basis_two, fractional_basis_two)
    )

    derivative_matrix = _linearized_acceleration_matrix(central_shape, masses)
    indicial_power = 0.5 * (-1.0 + np.sqrt(13.0))
    eigenvalue = 1.0 / 9.0

    for basis in (fractional_basis_one, fractional_basis_two):
        image = (derivative_matrix @ basis.reshape(-1)).reshape(3, 2)
        np.testing.assert_allclose(image, eigenvalue * basis, rtol=1e-10, atol=1e-12)

    assert abs(
        _mass_inner_product(masses, fractional_basis_one, fractional_basis_two)
    ) < 1e-13

    incoming_mode = 0.04 * fractional_basis_one - 0.025 * fractional_basis_two
    outgoing_mode = -0.03 * fractional_basis_one + 0.045 * fractional_basis_two
    order = 5

    def solve_branch(first_coefficient):
        coefficients = np.zeros((order + 1, 3, 2))
        coefficients[0] = central_shape
        coefficients[1] = first_coefficient
        for degree in range(2, order + 1):
            trial_coefficients = coefficients.copy()
            trial_coefficients[degree] = 0.0
            known_term = _shape_acceleration_series(
                trial_coefficients,
                masses,
                degree,
            )[degree].reshape(-1)
            multiplier = (
                (degree * indicial_power + 2.0)
                * (degree * indicial_power - 1.0)
                / 9.0
            )
            operator = multiplier * np.eye(6) - derivative_matrix
            solved = np.linalg.solve(operator, known_term)
            coefficients[degree] = solved.reshape(3, 2)
            recurrence_residual = (
                operator @ coefficients[degree].reshape(-1) - known_term
            )
            assert np.linalg.norm(recurrence_residual, ord=np.inf) < 1e-12
        return coefficients

    incoming_coefficients = solve_branch(incoming_mode)
    outgoing_coefficients = solve_branch(outgoing_mode)

    def signed_shape_residual(coefficients, tau):
        radial_time = abs(tau)
        sign = 1.0 if tau > 0.0 else -1.0
        shape = sum(
            coefficients[degree] * radial_time ** (degree * indicial_power)
            for degree in range(order + 1)
        )
        first_derivative = sum(
            sign
            * degree
            * indicial_power
            * coefficients[degree]
            * radial_time ** (degree * indicial_power - 1.0)
            for degree in range(1, order + 1)
        )
        second_derivative = sum(
            degree
            * indicial_power
            * (degree * indicial_power - 1.0)
            * coefficients[degree]
            * radial_time ** (degree * indicial_power - 2.0)
            for degree in range(1, order + 1)
        )
        shape_residual = (
            tau**2 * second_derivative
            + 2.0 * tau * first_derivative
            - 2.0 * shape
            - 9.0 * accelerations(shape, masses)
        )
        projected_scaled_newton_residual = shape_residual / 9.0
        angular_factor = sum(
            mass
            * (
                shape_row[0] * derivative_row[1]
                - shape_row[1] * derivative_row[0]
            )
            for mass, shape_row, derivative_row in zip(
                masses,
                shape,
                first_derivative,
            )
        )
        angular_momentum = tau**2 * angular_factor / 3.0
        return (
            np.linalg.norm(shape_residual, ord=np.inf),
            np.linalg.norm(projected_scaled_newton_residual, ord=np.inf),
            abs(angular_momentum),
        )

    incoming_residual, incoming_projected, incoming_angular = signed_shape_residual(
        incoming_coefficients,
        -0.028,
    )
    outgoing_residual, outgoing_projected, outgoing_angular = signed_shape_residual(
        outgoing_coefficients,
        0.028,
    )
    shared_entry_gap = np.linalg.norm(
        incoming_coefficients[0] - outgoing_coefficients[0],
        ord=np.inf,
    )
    branch_parameter_gap = np.linalg.norm(
        incoming_coefficients[1] - outgoing_coefficients[1],
        ord=np.inf,
    )

    assert shared_entry_gap < 1e-14
    assert branch_parameter_gap > 0.03
    assert incoming_residual < 1e-11
    assert outgoing_residual < 1e-11
    assert incoming_projected < 2e-12
    assert outgoing_projected < 2e-12
    assert incoming_angular < 1e-6
    assert outgoing_angular < 1e-6


def test_general_nonresonant_fuchsian_eigenmode_builds_unequal_mass_branches():
    masses = np.array([1.0, 0.7, 1.4])
    configuration, central_lambda = _mass_centered_equilateral_central_configuration(
        masses,
    )
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    derivative_matrix = _linearized_acceleration_matrix(central_shape, masses)
    beta = float(
        sum(masses[i] * masses[j] for i in range(3) for j in range(i + 1, 3))
        / np.sum(masses) ** 2
    )
    expected_shape_eigenvalues = (
        1.0 / 9.0 + (1.0 / 3.0) * np.sqrt(1.0 - 3.0 * beta),
        1.0 / 9.0 - (1.0 / 3.0) * np.sqrt(1.0 - 3.0 * beta),
    )
    observed_eigenvalues, observed_eigenvectors = np.linalg.eig(derivative_matrix)
    order = 5
    amplitude = 0.035
    tau = 0.032

    assert 8.0 / 27.0 < beta < 1.0 / 3.0

    for shape_eigenvalue in expected_shape_eigenvalues:
        indicial_power = 0.5 * (
            -1.0 + np.sqrt(9.0 + 36.0 * shape_eigenvalue)
        )
        eigenvector_index = int(
            np.argmin(
                np.abs(observed_eigenvalues.real - shape_eigenvalue)
                + np.abs(observed_eigenvalues.imag)
            )
        )
        mode = observed_eigenvectors[:, eigenvector_index].real.reshape(3, 2)
        mode /= np.sqrt(_mass_inner_product(masses, mode, mode))
        mode_image = (derivative_matrix @ mode.reshape(-1)).reshape(3, 2)
        coefficients = np.zeros((order + 1, 3, 2))
        coefficients[0] = central_shape
        coefficients[1] = amplitude * mode

        np.testing.assert_allclose(
            mode_image,
            shape_eigenvalue * mode,
            rtol=1e-10,
            atol=1e-12,
        )
        assert np.linalg.norm(np.sum(masses[:, None] * mode, axis=0), ord=np.inf) < 1e-12
        assert 1.0 < indicial_power < 2.0
        assert abs(indicial_power - round(indicial_power)) > 0.1

        spectrum = observed_eigenvalues.real
        for degree in range(2, order + 1):
            trial_coefficients = coefficients.copy()
            trial_coefficients[degree] = 0.0
            known_term = _shape_acceleration_series(
                trial_coefficients,
                masses,
                degree,
            )[degree].reshape(-1)
            multiplier = (
                (degree * indicial_power + 2.0)
                * (degree * indicial_power - 1.0)
                / 9.0
            )
            operator = multiplier * np.eye(6) - derivative_matrix
            coefficients[degree] = np.linalg.solve(
                operator,
                known_term,
            ).reshape(3, 2)
            recurrence_residual = (
                operator @ coefficients[degree].reshape(-1) - known_term
            )

            assert min(abs(multiplier - eigenvalue) for eigenvalue in spectrum) > 0.1
            assert np.linalg.norm(recurrence_residual, ord=np.inf) < 1e-12

        shape = sum(
            coefficients[degree] * tau ** (degree * indicial_power)
            for degree in range(order + 1)
        )
        first_derivative = sum(
            degree
            * indicial_power
            * coefficients[degree]
            * tau ** (degree * indicial_power - 1.0)
            for degree in range(1, order + 1)
        )
        second_derivative = sum(
            degree
            * indicial_power
            * (degree * indicial_power - 1.0)
            * coefficients[degree]
            * tau ** (degree * indicial_power - 2.0)
            for degree in range(1, order + 1)
        )
        shape_residual = (
            tau**2 * second_derivative
            + 2.0 * tau * first_derivative
            - 2.0 * shape
            - 9.0 * accelerations(shape, masses)
        )

        assert np.linalg.norm(shape_residual, ord=np.inf) < 2e-12
        assert np.linalg.norm(shape_residual / 9.0, ord=np.inf) < 3e-13


def test_multi_indicial_fuchsian_lift_constructs_simultaneous_fractional_modes():
    masses = np.array([1.0, 0.7, 1.4])
    configuration, central_lambda = _mass_centered_equilateral_central_configuration(
        masses,
    )
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    derivative_matrix = _linearized_acceleration_matrix(central_shape, masses)
    beta = float(
        sum(masses[i] * masses[j] for i in range(3) for j in range(i + 1, 3))
        / np.sum(masses) ** 2
    )
    shape_eigenvalues = (
        1.0 / 9.0 + (1.0 / 3.0) * np.sqrt(1.0 - 3.0 * beta),
        1.0 / 9.0 - (1.0 / 3.0) * np.sqrt(1.0 - 3.0 * beta),
    )
    indicial_powers = tuple(
        0.5 * (-1.0 + np.sqrt(9.0 + 36.0 * eigenvalue))
        for eigenvalue in shape_eigenvalues
    )
    observed_eigenvalues, observed_eigenvectors = np.linalg.eig(derivative_matrix)
    modes = []
    for eigenvalue in shape_eigenvalues:
        eigenvector_index = int(
            np.argmin(
                np.abs(observed_eigenvalues.real - eigenvalue)
                + np.abs(observed_eigenvalues.imag)
            )
        )
        mode = observed_eigenvectors[:, eigenvector_index].real.reshape(3, 2)
        mode /= np.sqrt(_mass_inner_product(masses, mode, mode))
        modes.append(mode)

    def multi_indices(total_degree):
        return [
            (first, total_degree - first)
            for first in range(total_degree + 1)
        ]

    def bounded_indices(maximum):
        return [
            (first, second)
            for first in range(maximum[0] + 1)
            for second in range(maximum[1] + 1)
        ]

    def add_index(left, right):
        return (left[0] + right[0], left[1] + right[1])

    def sub_index(left, right):
        return (left[0] - right[0], left[1] - right[1])

    def index_leq(left, right):
        return left[0] <= right[0] and left[1] <= right[1]

    def scalar_series_product(left, right, maximum):
        product = {}
        for left_index, left_value in left.items():
            for right_index, right_value in right.items():
                combined = add_index(left_index, right_index)
                if index_leq(combined, maximum):
                    product[combined] = product.get(combined, 0.0) + (
                        left_value * right_value
                    )
        return product

    def scalar_series_power_one_plus(tail, exponent, maximum):
        result = {(0, 0): 1.0}
        term = {(0, 0): 1.0}
        binomial_coefficient = 1.0
        for power in range(1, maximum[0] + maximum[1] + 1):
            term = scalar_series_product(term, tail, maximum)
            if not term:
                break
            binomial_coefficient *= (exponent - power + 1.0) / power
            for index, value in term.items():
                result[index] = result.get(index, 0.0) + (
                    binomial_coefficient * value
                )
        return result

    def multivariate_acceleration_coefficient(coefficients, target):
        acceleration_coefficient = np.zeros((3, 2))
        indices = bounded_indices(target)
        for body in range(3):
            for other in range(3):
                if body == other:
                    continue
                relative = {
                    index: (
                        coefficients.get(index, np.zeros((3, 2)))[other]
                        - coefficients.get(index, np.zeros((3, 2)))[body]
                    )
                    for index in indices
                }
                squared_distance = {}
                for index in indices:
                    squared_distance[index] = sum(
                        float(np.dot(relative[left], relative[sub_index(index, left)]))
                        for left in indices
                        if index_leq(left, index)
                    )
                normalized_tail = {
                    index: value / squared_distance[(0, 0)]
                    for index, value in squared_distance.items()
                    if index != (0, 0)
                }
                inverse_distance_cubed = scalar_series_power_one_plus(
                    normalized_tail,
                    -1.5,
                    target,
                )
                inverse_distance_cubed = {
                    index: squared_distance[(0, 0)] ** (-1.5) * value
                    for index, value in inverse_distance_cubed.items()
                }
                pair_force = sum(
                    relative[left] * inverse_distance_cubed.get(
                        sub_index(target, left),
                        0.0,
                    )
                    for left in indices
                    if index_leq(left, target)
                )
                acceleration_coefficient[body] += masses[other] * pair_force
        return acceleration_coefficient

    max_total_degree = 4
    coefficients = {
        (0, 0): central_shape,
        (1, 0): 0.03 * modes[0],
        (0, 1): -0.025 * modes[1],
    }

    assert 1.0 < indicial_powers[1] < indicial_powers[0] < 2.0
    assert abs(indicial_powers[0] - indicial_powers[1]) > 0.2
    assert abs(_mass_inner_product(masses, modes[0], modes[1])) < 1e-12

    for mode, eigenvalue in zip(modes, shape_eigenvalues):
        image = (derivative_matrix @ mode.reshape(-1)).reshape(3, 2)
        np.testing.assert_allclose(image, eigenvalue * mode, rtol=1e-10, atol=1e-12)

    spectrum = observed_eigenvalues.real
    solved_indices = []
    for total_degree in range(2, max_total_degree + 1):
        for index in multi_indices(total_degree):
            weighted_exponent = (
                index[0] * indicial_powers[0] + index[1] * indicial_powers[1]
            )
            multiplier = (
                (weighted_exponent + 2.0)
                * (weighted_exponent - 1.0)
                / 9.0
            )
            trial_coefficients = dict(coefficients)
            trial_coefficients[index] = np.zeros((3, 2))
            known_term = multivariate_acceleration_coefficient(
                trial_coefficients,
                index,
            ).reshape(-1)
            operator = multiplier * np.eye(6) - derivative_matrix
            coefficients[index] = np.linalg.solve(operator, known_term).reshape(3, 2)
            recurrence_residual = operator @ coefficients[index].reshape(-1) - known_term

            solved_indices.append(index)
            assert min(abs(multiplier - eigenvalue) for eigenvalue in spectrum) > 0.05
            assert np.linalg.norm(recurrence_residual, ord=np.inf) < 2e-12

    tau = 0.025
    shape = sum(
        coefficient
        * tau ** (
            index[0] * indicial_powers[0]
            + index[1] * indicial_powers[1]
        )
        for index, coefficient in coefficients.items()
    )
    euler_shape_derivative = sum(
        (
            index[0] * indicial_powers[0]
            + index[1] * indicial_powers[1]
        )
        * coefficient
        * tau ** (
            index[0] * indicial_powers[0]
            + index[1] * indicial_powers[1]
            - 1.0
        )
        for index, coefficient in coefficients.items()
        if index != (0, 0)
    )
    euler_shape_second_derivative = sum(
        (
            index[0] * indicial_powers[0]
            + index[1] * indicial_powers[1]
        )
        * (
            index[0] * indicial_powers[0]
            + index[1] * indicial_powers[1]
            - 1.0
        )
        * coefficient
        * tau ** (
            index[0] * indicial_powers[0]
            + index[1] * indicial_powers[1]
            - 2.0
        )
        for index, coefficient in coefficients.items()
        if index != (0, 0)
    )
    shape_residual = (
        tau**2 * euler_shape_second_derivative
        + 2.0 * tau * euler_shape_derivative
        - 2.0 * shape
        - 9.0 * accelerations(shape, masses)
    )

    assert (1, 1) in solved_indices
    assert np.linalg.norm(coefficients[(1, 1)], ord=np.inf) > 1e-4
    assert np.linalg.norm(shape_residual, ord=np.inf) < 3e-9
    assert np.linalg.norm(shape_residual / 9.0, ord=np.inf) < 4e-10


def test_scale_coordinate_couples_with_multi_indicial_fuchsian_modes():
    masses = np.array([1.0, 0.7, 1.4])
    configuration, central_lambda = _mass_centered_equilateral_central_configuration(
        masses,
    )
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    derivative_matrix = _linearized_acceleration_matrix(central_shape, masses)
    beta = float(
        sum(masses[i] * masses[j] for i in range(3) for j in range(i + 1, 3))
        / np.sum(masses) ** 2
    )
    shape_eigenvalues = (
        1.0 / 9.0 + (1.0 / 3.0) * np.sqrt(1.0 - 3.0 * beta),
        1.0 / 9.0 - (1.0 / 3.0) * np.sqrt(1.0 - 3.0 * beta),
    )
    indicial_powers = (
        2.0,
        *(
            0.5 * (-1.0 + np.sqrt(9.0 + 36.0 * eigenvalue))
            for eigenvalue in shape_eigenvalues
        ),
    )
    observed_eigenvalues, observed_eigenvectors = np.linalg.eig(derivative_matrix)
    fractional_modes = []
    for eigenvalue in shape_eigenvalues:
        eigenvector_index = int(
            np.argmin(
                np.abs(observed_eigenvalues.real - eigenvalue)
                + np.abs(observed_eigenvalues.imag)
            )
        )
        mode = observed_eigenvectors[:, eigenvector_index].real.reshape(3, 2)
        mode /= np.sqrt(_mass_inner_product(masses, mode, mode))
        fractional_modes.append(mode)

    scale_image = (derivative_matrix @ central_shape.reshape(-1)).reshape(3, 2)
    np.testing.assert_allclose(scale_image, (4.0 / 9.0) * central_shape)
    for mode, eigenvalue in zip(fractional_modes, shape_eigenvalues):
        image = (derivative_matrix @ mode.reshape(-1)).reshape(3, 2)
        np.testing.assert_allclose(image, eigenvalue * mode, rtol=1e-10, atol=1e-12)

    scale_coefficient = 0.02
    coefficients = {
        (0, 0, 0): central_shape,
        (1, 0, 0): scale_coefficient * central_shape,
        (0, 1, 0): 0.025 * fractional_modes[0],
        (0, 0, 1): -0.02 * fractional_modes[1],
    }
    selected_basis_indices = {(1, 0, 0), (0, 1, 0), (0, 0, 1)}
    spectrum = observed_eigenvalues.real
    solved_indices = []

    for total_degree in range(2, 4):
        for index in _multi_indices_with_total(total_degree, 3):
            if index in selected_basis_indices:
                continue
            weighted_exponent = sum(
                coordinate * power
                for coordinate, power in zip(index, indicial_powers)
            )
            multiplier = (
                (weighted_exponent + 2.0)
                * (weighted_exponent - 1.0)
                / 9.0
            )
            trial_coefficients = dict(coefficients)
            trial_coefficients[index] = np.zeros((3, 2))
            known_term = _multivariate_acceleration_coefficient(
                trial_coefficients,
                masses,
                index,
            ).reshape(-1)
            operator = multiplier * np.eye(6) - derivative_matrix
            coefficients[index] = np.linalg.solve(operator, known_term).reshape(3, 2)
            recurrence_residual = operator @ coefficients[index].reshape(-1) - known_term

            solved_indices.append(index)
            assert min(abs(multiplier - eigenvalue) for eigenvalue in spectrum) > 0.05
            assert np.linalg.norm(recurrence_residual, ord=np.inf) < 2e-12

    tau = 0.025

    def exponent(index):
        return sum(coordinate * power for coordinate, power in zip(index, indicial_powers))

    shape = sum(
        coefficient * tau ** exponent(index)
        for index, coefficient in coefficients.items()
    )
    first_derivative = sum(
        exponent(index)
        * coefficient
        * tau ** (exponent(index) - 1.0)
        for index, coefficient in coefficients.items()
        if index != (0, 0, 0)
    )
    second_derivative = sum(
        exponent(index)
        * (exponent(index) - 1.0)
        * coefficient
        * tau ** (exponent(index) - 2.0)
        for index, coefficient in coefficients.items()
        if index != (0, 0, 0)
    )
    shape_residual = (
        tau**2 * second_derivative
        + 2.0 * tau * first_derivative
        - 2.0 * shape
        - 9.0 * accelerations(shape, masses)
    )
    expected_energy = (
        (10.0 / 9.0)
        * scale_coefficient
        * _mass_inner_product(masses, central_shape, central_shape)
    )
    energy_errors = []
    for regularized_time in (0.06, 0.04, 0.025):
        sampled_shape = sum(
            coefficient * regularized_time ** exponent(index)
            for index, coefficient in coefficients.items()
        )
        sampled_first_derivative = sum(
            exponent(index)
            * coefficient
            * regularized_time ** (exponent(index) - 1.0)
            for index, coefficient in coefficients.items()
            if index != (0, 0, 0)
        )
        positions = regularized_time**2 * sampled_shape
        velocities = (
            (2.0 / (3.0 * regularized_time)) * sampled_shape
            + sampled_first_derivative / 3.0
        )
        state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
        energy_errors.append(abs(energy(state, masses) - expected_energy))

    assert (1, 1, 0) in solved_indices
    assert (1, 0, 1) in solved_indices
    assert (0, 1, 1) in solved_indices
    assert np.linalg.norm(coefficients[(1, 1, 0)], ord=np.inf) > 5e-5
    assert np.linalg.norm(coefficients[(1, 0, 1)], ord=np.inf) > 1e-5
    assert np.linalg.norm(coefficients[(0, 1, 1)], ord=np.inf) > 1e-4
    assert np.linalg.norm(shape_residual, ord=np.inf) < 1e-11
    assert np.linalg.norm(shape_residual / 9.0, ord=np.inf) < 2e-12
    assert max(energy_errors) < 1e-9


def test_two_sided_mixed_fuchsian_energy_matching_uses_scale_coordinate():
    masses = np.array([1.0, 0.7, 1.4])
    configuration, central_lambda = _mass_centered_equilateral_central_configuration(
        masses,
    )
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    derivative_matrix = _linearized_acceleration_matrix(central_shape, masses)
    beta = float(
        sum(masses[i] * masses[j] for i in range(3) for j in range(i + 1, 3))
        / np.sum(masses) ** 2
    )
    shape_eigenvalues = (
        1.0 / 9.0 + (1.0 / 3.0) * np.sqrt(1.0 - 3.0 * beta),
        1.0 / 9.0 - (1.0 / 3.0) * np.sqrt(1.0 - 3.0 * beta),
    )
    indicial_powers = (
        2.0,
        *(
            0.5 * (-1.0 + np.sqrt(9.0 + 36.0 * eigenvalue))
            for eigenvalue in shape_eigenvalues
        ),
    )
    observed_eigenvalues, observed_eigenvectors = np.linalg.eig(derivative_matrix)
    fractional_modes = []
    for eigenvalue in shape_eigenvalues:
        eigenvector_index = int(
            np.argmin(
                np.abs(observed_eigenvalues.real - eigenvalue)
                + np.abs(observed_eigenvalues.imag)
            )
        )
        mode = observed_eigenvectors[:, eigenvector_index].real.reshape(3, 2)
        mode /= np.sqrt(_mass_inner_product(masses, mode, mode))
        fractional_modes.append(mode)

    def exponent(index):
        return sum(coordinate * power for coordinate, power in zip(index, indicial_powers))

    def solve_branch(scale_coefficient, amplitudes):
        coefficients = {
            (0, 0, 0): central_shape,
            (1, 0, 0): scale_coefficient * central_shape,
            (0, 1, 0): amplitudes[0] * fractional_modes[0],
            (0, 0, 1): amplitudes[1] * fractional_modes[1],
        }
        selected_basis_indices = {(1, 0, 0), (0, 1, 0), (0, 0, 1)}
        spectrum = observed_eigenvalues.real
        solved_indices = []

        for total_degree in range(2, 5):
            for index in _multi_indices_with_total(total_degree, 3):
                if index in selected_basis_indices:
                    continue
                weighted_exponent = exponent(index)
                multiplier = (
                    (weighted_exponent + 2.0)
                    * (weighted_exponent - 1.0)
                    / 9.0
                )
                trial_coefficients = dict(coefficients)
                trial_coefficients[index] = np.zeros((3, 2))
                known_term = _multivariate_acceleration_coefficient(
                    trial_coefficients,
                    masses,
                    index,
                ).reshape(-1)
                operator = multiplier * np.eye(6) - derivative_matrix
                coefficients[index] = np.linalg.solve(operator, known_term).reshape(
                    3,
                    2,
                )
                recurrence_residual = (
                    operator @ coefficients[index].reshape(-1) - known_term
                )

                solved_indices.append(index)
                assert min(abs(multiplier - eigenvalue) for eigenvalue in spectrum) > 0.03
                assert np.linalg.norm(recurrence_residual, ord=np.inf) < 2e-12

        return coefficients, solved_indices

    def sample_branch(coefficients, signed_tau):
        u = abs(signed_tau)
        sign = np.sign(signed_tau)
        shape = sum(
            coefficient * u ** exponent(index)
            for index, coefficient in coefficients.items()
        )
        first_u_derivative = sum(
            exponent(index)
            * coefficient
            * u ** (exponent(index) - 1.0)
            for index, coefficient in coefficients.items()
            if index != (0, 0, 0)
        )
        second_u_derivative = sum(
            exponent(index)
            * (exponent(index) - 1.0)
            * coefficient
            * u ** (exponent(index) - 2.0)
            for index, coefficient in coefficients.items()
            if index != (0, 0, 0)
        )
        signed_shape_derivative = sign * first_u_derivative
        signed_shape_second_derivative = second_u_derivative
        shape_residual = (
            signed_tau**2 * signed_shape_second_derivative
            + 2.0 * signed_tau * signed_shape_derivative
            - 2.0 * shape
            - 9.0 * accelerations(shape, masses)
        )
        positions = signed_tau**2 * shape
        velocities = sign * (
            (2.0 / (3.0 * u)) * shape
            + first_u_derivative / 3.0
        )
        state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
        return state, shape_residual

    incoming_scale = 0.02
    shifted_scale = 0.031
    incoming, incoming_indices = solve_branch(incoming_scale, (-0.018, 0.022))
    outgoing_same_energy, outgoing_indices = solve_branch(
        incoming_scale,
        (0.027, -0.015),
    )
    outgoing_shifted_energy, shifted_indices = solve_branch(
        shifted_scale,
        (0.027, -0.015),
    )
    outgoing_identity_selector, identity_indices = solve_branch(
        incoming_scale,
        (-0.018, 0.022),
    )
    no_fractional_branch, no_fractional_indices = solve_branch(
        incoming_scale,
        (0.0, 0.0),
    )
    inertia = _mass_inner_product(masses, central_shape, central_shape)
    expected_incoming_energy = (10.0 / 9.0) * incoming_scale * inertia
    expected_shifted_energy = (10.0 / 9.0) * shifted_scale * inertia
    expected_energy_gap = (
        (10.0 / 9.0)
        * (shifted_scale - incoming_scale)
        * inertia
    )

    incoming_state, incoming_residual = sample_branch(incoming, -0.035)
    outgoing_state, outgoing_residual = sample_branch(outgoing_same_energy, 0.035)
    shifted_state, shifted_residual = sample_branch(outgoing_shifted_energy, 0.035)
    identity_state, identity_residual = sample_branch(outgoing_identity_selector, 0.035)

    incoming_energy = energy(incoming_state, masses)
    outgoing_energy = energy(outgoing_state, masses)
    shifted_energy = energy(shifted_state, masses)
    identity_energy = energy(identity_state, masses)
    branch_parameter_gap = np.linalg.norm(
        incoming[(0, 1, 0)] - outgoing_same_energy[(0, 1, 0)],
        ord=np.inf,
    ) + np.linalg.norm(
        incoming[(0, 0, 1)] - outgoing_same_energy[(0, 0, 1)],
        ord=np.inf,
    )

    def regularized_position_derivative(
        coefficients,
        signed_regularized_time,
        derivative_order,
    ):
        u = abs(signed_regularized_time)
        sign = np.sign(signed_regularized_time)
        derivative = np.zeros_like(central_shape)
        for index, coefficient in coefficients.items():
            power = exponent(index) + 2.0
            falling_factorial = 1.0
            for offset in range(derivative_order):
                falling_factorial *= power - offset
            derivative += (
                sign**derivative_order
                * falling_factorial
                * coefficient
                * u ** (power - derivative_order)
            )
        return derivative

    finite_jet_gaps = []
    for u in (1e-3, 1e-5, 1e-7):
        finite_jet_gaps.append(
            max(
                np.linalg.norm(
                    regularized_position_derivative(incoming, -u, derivative_order)
                    - regularized_position_derivative(
                        outgoing_same_energy,
                        u,
                        derivative_order,
                    ),
                    ord=np.inf,
                )
                for derivative_order in range(4)
            )
        )
    incoming_near_collision_jets = [
        regularized_position_derivative(incoming, -1e-7, derivative_order)
        for derivative_order in range(4)
    ]
    outgoing_near_collision_jets = [
        regularized_position_derivative(
            outgoing_same_energy,
            1e-7,
            derivative_order,
        )
        for derivative_order in range(4)
    ]

    def sampled_shape(coefficients, u):
        return sum(
            coefficient * u ** exponent(index)
            for index, coefficient in coefficients.items()
        )

    def recovered_fractional_amplitudes(coefficients, u):
        shape_remainder = (
            sampled_shape(coefficients, u)
            - central_shape
            - coefficients[(1, 0, 0)] * u**2
        )
        return tuple(
            _mass_inner_product(masses, shape_remainder, mode)
            / u ** indicial_power
            for mode, indicial_power in zip(fractional_modes, indicial_powers[1:])
        )

    incoming_recovered_amplitudes = recovered_fractional_amplitudes(incoming, 1e-5)
    outgoing_recovered_amplitudes = recovered_fractional_amplitudes(
        outgoing_same_energy,
        1e-5,
    )
    identity_recovered_amplitudes = recovered_fractional_amplitudes(
        outgoing_identity_selector,
        1e-5,
    )

    for required_index in ((1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 0, 0)):
        assert required_index in incoming_indices
        assert required_index in outgoing_indices
        assert required_index in shifted_indices
        assert required_index in identity_indices
        assert required_index in no_fractional_indices
    hidden_index = (1, 1, 0)
    hidden_multiplier = (
        (exponent(hidden_index) + 2.0)
        * (exponent(hidden_index) - 1.0)
        / 9.0
    )
    hidden_operator = hidden_multiplier * np.eye(6) - derivative_matrix
    hidden_singular_values = np.linalg.svd(hidden_operator, compute_uv=False)
    hidden_perturbation = np.array(
        [
            [0.11, -0.07],
            [-0.05, 0.13],
            [0.02, -0.09],
        ]
    )
    hidden_perturbation /= np.linalg.norm(hidden_perturbation.reshape(-1))
    perturbed_coefficients = dict(outgoing_identity_selector)
    perturbed_coefficients[hidden_index] = (
        perturbed_coefficients[hidden_index] + 1e-3 * hidden_perturbation
    )
    trial_coefficients = dict(perturbed_coefficients)
    trial_coefficients[hidden_index] = np.zeros((3, 2))
    perturbed_known_term = _multivariate_acceleration_coefficient(
        trial_coefficients,
        masses,
        hidden_index,
    ).reshape(-1)
    perturbed_recurrence_residual = (
        hidden_operator @ perturbed_coefficients[hidden_index].reshape(-1)
        - perturbed_known_term
    )
    solved_known_term = _multivariate_acceleration_coefficient(
        trial_coefficients,
        masses,
        hidden_index,
    ).reshape(-1)
    solved_recurrence_residual = (
        hidden_operator @ outgoing_identity_selector[hidden_index].reshape(-1)
        - solved_known_term
    )
    assert branch_parameter_gap > 0.03
    assert np.min(hidden_singular_values) > 0.1
    assert np.linalg.norm(solved_recurrence_residual, ord=np.inf) < 2e-12
    assert np.linalg.norm(perturbed_recurrence_residual, ord=np.inf) > 1e-5
    fractional_mode_blowups = []
    for mode_power, mode in zip(indicial_powers[1:], fractional_modes):
        derivative_order = int(np.floor(mode_power + 2.0)) + 1
        derivative_exponent = mode_power + 2.0 - derivative_order
        projections = [
            abs(
                _mass_inner_product(
                    masses,
                    regularized_position_derivative(
                        incoming,
                        -u,
                        derivative_order,
                    ),
                    mode,
                )
            )
            for u in (1e-3, 1e-5, 1e-7)
        ]
        expected_blowup_ratio = (1e-7 / 1e-5) ** derivative_exponent
        fractional_mode_blowups.append((mode_power, derivative_order, projections))

        assert derivative_order == 4
        assert derivative_exponent < 0.0
        assert not np.isclose(mode_power + 2.0, round(mode_power + 2.0))
        assert projections[2] > 0.4 * expected_blowup_ratio * projections[1]
        assert projections[1] > 10.0 * projections[0]

    no_fractional_fourth_derivatives = [
        regularized_position_derivative(no_fractional_branch, u, 4)
        for u in (1e-3, 1e-5, 1e-7)
    ]
    expected_smooth_fourth_derivative = 24.0 * incoming_scale * central_shape
    homothetic_coefficients = _homothetic_energy_series_coefficients(
        (10.0 / 9.0) * incoming_scale,
        order=4,
    )
    assert all(power < 2.0 for power in indicial_powers[1:])
    assert len(fractional_mode_blowups) == len(fractional_modes)
    np.testing.assert_allclose(
        no_fractional_fourth_derivatives[-1],
        expected_smooth_fourth_derivative,
        atol=1e-8,
        rtol=0.0,
    )
    assert (
        np.linalg.norm(
            no_fractional_fourth_derivatives[-1]
            - no_fractional_fourth_derivatives[-2],
            ord=np.inf,
        )
        < 1e-7
    )
    for index, coefficient in no_fractional_branch.items():
        if index[1:] != (0, 0):
            assert np.linalg.norm(coefficient, ord=np.inf) < 1e-13
    for degree, scalar_coefficient in enumerate(homothetic_coefficients):
        np.testing.assert_allclose(
            no_fractional_branch[(degree, 0, 0)],
            scalar_coefficient * central_shape,
            atol=1e-13,
            rtol=1e-13,
        )
    assert finite_jet_gaps[2] < finite_jet_gaps[1] < finite_jet_gaps[0]
    assert finite_jet_gaps[-1] < 0.05
    assert np.linalg.norm(incoming_near_collision_jets[0], ord=np.inf) < 1e-13
    assert np.linalg.norm(outgoing_near_collision_jets[0], ord=np.inf) < 1e-13
    assert np.linalg.norm(incoming_near_collision_jets[1], ord=np.inf) < 5e-7
    assert np.linalg.norm(outgoing_near_collision_jets[1], ord=np.inf) < 5e-7
    np.testing.assert_allclose(
        incoming_near_collision_jets[2],
        2.0 * central_shape,
        atol=2e-7,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        outgoing_near_collision_jets[2],
        2.0 * central_shape,
        atol=2e-7,
        rtol=0.0,
    )
    assert np.linalg.norm(incoming_near_collision_jets[3], ord=np.inf) < 0.04
    assert np.linalg.norm(outgoing_near_collision_jets[3], ord=np.inf) < 0.04
    assert incoming_recovered_amplitudes[0] == pytest.approx(-0.018, abs=2e-6)
    assert incoming_recovered_amplitudes[1] == pytest.approx(0.022, abs=2e-6)
    assert outgoing_recovered_amplitudes[0] == pytest.approx(0.027, abs=2e-6)
    assert outgoing_recovered_amplitudes[1] == pytest.approx(-0.015, abs=2e-6)
    assert identity_recovered_amplitudes[0] == pytest.approx(
        incoming_recovered_amplitudes[0],
        abs=2e-9,
    )
    assert identity_recovered_amplitudes[1] == pytest.approx(
        incoming_recovered_amplitudes[1],
        abs=2e-9,
    )
    for index, coefficient in incoming.items():
        np.testing.assert_allclose(
            outgoing_identity_selector[index],
            coefficient,
            atol=1e-13,
            rtol=1e-13,
        )
    assert np.linalg.norm(incoming_residual, ord=np.inf) < 2e-9
    assert np.linalg.norm(outgoing_residual, ord=np.inf) < 2e-9
    assert np.linalg.norm(shifted_residual, ord=np.inf) < 2e-9
    assert np.linalg.norm(identity_residual, ord=np.inf) < 2e-9
    assert abs(incoming_energy - expected_incoming_energy) < 1e-9
    assert abs(outgoing_energy - expected_incoming_energy) < 1e-9
    assert abs(identity_energy - expected_incoming_energy) < 1e-9
    assert abs(shifted_energy - expected_shifted_energy) < 1e-9
    assert abs(incoming_energy - outgoing_energy) < 2e-9
    assert abs(incoming_energy - identity_energy) < 2e-9
    assert abs((shifted_energy - outgoing_energy) - expected_energy_gap) < 2e-9


def test_fuchsian_selector_constructor_builds_two_sided_identity_branch():
    masses = np.array([1.0, 0.7, 1.4])
    configuration, central_lambda = _mass_centered_equilateral_central_configuration(
        masses,
    )
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    derivative_matrix = _linearized_acceleration_matrix(central_shape, masses)
    beta = float(
        sum(masses[i] * masses[j] for i in range(3) for j in range(i + 1, 3))
        / np.sum(masses) ** 2
    )
    shape_eigenvalues = (
        1.0 / 9.0 + (1.0 / 3.0) * np.sqrt(1.0 - 3.0 * beta),
        1.0 / 9.0 - (1.0 / 3.0) * np.sqrt(1.0 - 3.0 * beta),
    )
    indicial_powers = (
        2.0,
        *(
            0.5 * (-1.0 + np.sqrt(9.0 + 36.0 * eigenvalue))
            for eigenvalue in shape_eigenvalues
        ),
    )
    observed_eigenvalues, observed_eigenvectors = np.linalg.eig(derivative_matrix)
    fractional_modes = []
    for eigenvalue in shape_eigenvalues:
        eigenvector_index = int(
            np.argmin(
                np.abs(observed_eigenvalues.real - eigenvalue)
                + np.abs(observed_eigenvalues.imag)
            )
        )
        mode = observed_eigenvectors[:, eigenvector_index].real.reshape(3, 2)
        mode /= np.sqrt(_mass_inner_product(masses, mode, mode))
        fractional_modes.append(mode)

    scale_coefficient = 0.02
    incoming_selectors = {
        (1, 0, 0): scale_coefficient * central_shape,
        (0, 1, 0): -0.018 * fractional_modes[0],
        (0, 0, 1): 0.022 * fractional_modes[1],
    }
    changed_fractional_selectors = {
        (1, 0, 0): scale_coefficient * central_shape,
        (0, 1, 0): 0.027 * fractional_modes[0],
        (0, 0, 1): -0.015 * fractional_modes[1],
    }

    identity_continuation = construct_fuchsian_selector_continuation(
        masses=masses,
        central_shape=central_shape,
        powers=indicial_powers,
        incoming_selected_coefficients=incoming_selectors,
        max_total_degree=4,
        scale_index=(1, 0, 0),
    )
    changed_continuation = construct_fuchsian_selector_continuation(
        masses=masses,
        central_shape=central_shape,
        powers=indicial_powers,
        incoming_selected_coefficients=incoming_selectors,
        outgoing_selected_coefficients=changed_fractional_selectors,
        max_total_degree=4,
        scale_index=(1, 0, 0),
    )
    derived_identity = derive_identity_fuchsian_selector_continuation_from_incoming_branch(
        identity_continuation.incoming,
    )

    assert identity_continuation.identity_selector_certified
    assert derived_identity.identity_selector_certified
    assert derived_identity.selected_indices == identity_continuation.selected_indices
    assert identity_continuation.incoming.max_recurrence_residual_norm < 2e-12
    assert identity_continuation.incoming.min_solved_singular_value_floor > 0.03
    assert changed_continuation.max_selector_gap > 0.02
    assert changed_continuation.energy_gap < 1e-12

    incoming = identity_continuation.incoming
    outgoing_identity = identity_continuation.outgoing
    outgoing_changed = changed_continuation.outgoing
    expected_energy = incoming.finite_energy_limit

    for branch, signed_tau in (
        (incoming, -0.035),
        (outgoing_identity, 0.035),
        (outgoing_changed, 0.035),
    ):
        assert np.linalg.norm(branch.shape_equation_residual_at_tau(signed_tau), ord=np.inf) < 2e-9
        assert np.linalg.norm(branch.scaled_newton_residual_at_tau(signed_tau), ord=np.inf) < 3e-10
        assert abs(branch.centered_angular_momentum_scalar_at_tau(signed_tau)) < 2e-6
        assert abs(energy(branch.state_at_tau(signed_tau), masses) - expected_energy) < 2e-9

    for index, coefficient in incoming.coefficients.items():
        np.testing.assert_allclose(
            outgoing_identity.coefficients[index],
            coefficient,
            atol=1e-13,
            rtol=1e-13,
        )
        np.testing.assert_allclose(
            derived_identity.outgoing.coefficients[index],
            coefficient,
            atol=1e-13,
            rtol=1e-13,
        )

    finite_jet_gaps = []
    for radius in (1e-3, 1e-5, 1e-7):
        finite_jet_gaps.append(
            max(
                np.linalg.norm(
                    incoming.regularized_position_derivative(-radius, derivative_order)
                    - outgoing_changed.regularized_position_derivative(
                        radius,
                        derivative_order,
                    ),
                    ord=np.inf,
                )
                for derivative_order in range(4)
            )
        )
    assert finite_jet_gaps[2] < finite_jet_gaps[1] < finite_jet_gaps[0]
    assert finite_jet_gaps[-1] < 0.05

    energy_untracked = construct_fuchsian_selector_continuation(
        masses=masses,
        central_shape=central_shape,
        powers=indicial_powers[1:],
        incoming_selected_coefficients={
            (1, 0): -0.018 * fractional_modes[0],
            (0, 1): 0.022 * fractional_modes[1],
        },
        max_total_degree=3,
    )
    assert not energy_untracked.identity_selector_certified
    with pytest.raises(ValueError, match="scale_index"):
        derive_identity_fuchsian_selector_continuation_from_incoming_branch(
            energy_untracked.incoming,
        )


def test_any_analytic_cubic_time_total_collision_branch_forces_second_shape():
    configuration, central_lambda = _central_configuration_data("equilateral")
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    quadratic_coefficient = scale_factor * configuration
    cubic_coefficient = np.array(
        [
            [0.11, -0.04],
            [-0.07, 0.09],
            [0.02, -0.03],
        ]
    )
    quartic_coefficient = np.array(
        [
            [0.03, 0.02],
            [-0.01, 0.04],
            [0.05, -0.02],
        ]
    )
    noncentral_quadratic_coefficient = quadratic_coefficient + np.array(
        [
            [0.08, 0.01],
            [-0.02, 0.03],
            [0.01, -0.06],
        ]
    )

    central_residuals = []
    noncentral_residuals = []
    for regularized_time in (1e-2, 3e-3, 1e-3):
        central_positions = (
            quadratic_coefficient * regularized_time**2
            + cubic_coefficient * regularized_time**3
            + quartic_coefficient * regularized_time**4
        )
        central_acceleration = _cubic_time_acceleration(
            quadratic_coefficient,
            cubic_coefficient,
            quartic_coefficient,
            regularized_time,
        )
        central_residuals.append(
            np.linalg.norm(
                regularized_time**4
                * (central_acceleration - accelerations(central_positions)),
                ord=np.inf,
            )
        )

        noncentral_positions = (
            noncentral_quadratic_coefficient * regularized_time**2
            + cubic_coefficient * regularized_time**3
            + quartic_coefficient * regularized_time**4
        )
        noncentral_acceleration = _cubic_time_acceleration(
            noncentral_quadratic_coefficient,
            cubic_coefficient,
            quartic_coefficient,
            regularized_time,
        )
        noncentral_residuals.append(
            np.linalg.norm(
                regularized_time**4
                * (noncentral_acceleration - accelerations(noncentral_positions)),
                ord=np.inf,
            )
        )

    forced_second_shape_residual = np.linalg.norm(
        accelerations(quadratic_coefficient)
        + (2.0 / 9.0) * quadratic_coefficient,
        ord=np.inf,
    )
    noncentral_forced_shape_residual = np.linalg.norm(
        accelerations(noncentral_quadratic_coefficient)
        + (2.0 / 9.0) * noncentral_quadratic_coefficient,
        ord=np.inf,
    )

    assert forced_second_shape_residual < 1e-14
    assert central_residuals[0] > central_residuals[1] > central_residuals[2]
    assert central_residuals[-1] < 3e-4
    assert noncentral_forced_shape_residual > 0.01
    assert noncentral_residuals[-1] == pytest.approx(noncentral_forced_shape_residual, rel=2e-2)


def test_noncollinear_three_body_central_second_jet_is_equilateral():
    masses = np.array([1.0, 0.7, 1.4])
    equilateral_shape, central_lambda = _mass_centered_equilateral_central_configuration(masses)
    equilateral_scale = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    equilateral_second_shape = equilateral_scale * equilateral_shape
    scalene_shape = np.array(
        [
            [1.1, 0.0],
            [-0.25, 0.9],
            [-0.4, -0.35],
        ]
    )
    scalene_shape = scalene_shape - np.average(scalene_shape, axis=0, weights=masses)

    equilateral_distances = [
        np.linalg.norm(equilateral_second_shape[j] - equilateral_second_shape[i])
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    scalene_distances = [
        np.linalg.norm(scalene_shape[j] - scalene_shape[i])
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    equilateral_best_lambda = -float(
        np.sum(masses[:, None] * equilateral_second_shape * accelerations(equilateral_second_shape, masses))
        / np.sum(masses[:, None] * equilateral_second_shape * equilateral_second_shape)
    )
    scalene_best_lambda = -float(
        np.sum(masses[:, None] * scalene_shape * accelerations(scalene_shape, masses))
        / np.sum(masses[:, None] * scalene_shape * scalene_shape)
    )
    equilateral_central_residual = np.linalg.norm(
        accelerations(equilateral_second_shape, masses)
        + equilateral_best_lambda * equilateral_second_shape,
        ord=np.inf,
    )
    scalene_central_residual = np.linalg.norm(
        accelerations(scalene_shape, masses) + scalene_best_lambda * scalene_shape,
        ord=np.inf,
    )

    assert max(equilateral_distances) - min(equilateral_distances) < 1e-12
    assert abs(equilateral_best_lambda - 2.0 / 9.0) < 1e-12
    assert equilateral_central_residual < 1e-12
    assert max(scalene_distances) - min(scalene_distances) > 0.2
    assert scalene_central_residual > 0.1


def test_ordered_collinear_euler_second_jet_has_unique_positive_ratio():
    base_masses = np.array([1.0, 0.7, 1.4])

    for masses in (
        base_masses,
        base_masses[[0, 2, 1]],
        base_masses[[1, 0, 2]],
        base_masses[[1, 2, 0]],
        base_masses[[2, 0, 1]],
        base_masses[[2, 1, 0]],
    ):
        ratio, polynomial = _ordered_euler_ratio(masses)
        uncentered_line = np.array([0.0, 1.0, 1.0 + ratio])
        centered_line = uncentered_line - np.average(uncentered_line, weights=masses)
        unscaled_positions = np.column_stack([centered_line, np.zeros(3)])
        unscaled_acceleration = accelerations(unscaled_positions, masses)
        central_lambda = -float(
            np.sum(masses[:, None] * unscaled_positions * unscaled_acceleration)
            / np.sum(masses[:, None] * unscaled_positions * unscaled_positions)
        )
        scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
        normalized_positions = scale_factor * unscaled_positions

        assert np.polyval(polynomial, 0.0) < 0.0
        assert np.polyval(polynomial, 10.0) > 0.0
        assert np.linalg.norm(
            unscaled_acceleration + central_lambda * unscaled_positions,
            ord=np.inf,
        ) < 4e-14
        assert np.linalg.norm(
            accelerations(normalized_positions, masses)
            + (2.0 / 9.0) * normalized_positions,
            ord=np.inf,
        ) < 2e-14


def test_ordered_euler_linearized_spectrum_has_single_horizontal_shape_parameter():
    base_masses = np.array([1.0, 0.7, 1.4])

    for masses in (
        base_masses,
        base_masses[[0, 2, 1]],
        base_masses[[1, 0, 2]],
        base_masses[[1, 2, 0]],
        base_masses[[2, 0, 1]],
        base_masses[[2, 1, 0]],
    ):
        ratio, _polynomial = _ordered_euler_ratio(masses)
        uncentered_line = np.array([0.0, 1.0, 1.0 + ratio])
        centered_line = uncentered_line - np.average(uncentered_line, weights=masses)
        unscaled_positions = np.column_stack([centered_line, np.zeros(3)])
        unscaled_acceleration = accelerations(unscaled_positions, masses)
        central_lambda = -float(
            np.sum(masses[:, None] * unscaled_positions * unscaled_acceleration)
            / np.sum(masses[:, None] * unscaled_positions * unscaled_positions)
        )
        scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
        normalized_positions = scale_factor * unscaled_positions
        derivative_matrix = _linearized_acceleration_matrix(
            normalized_positions,
            masses,
        )
        horizontal_indices = [0, 2, 4]
        vertical_indices = [1, 3, 5]
        horizontal_block = derivative_matrix[np.ix_(horizontal_indices, horizontal_indices)]
        vertical_block = derivative_matrix[np.ix_(vertical_indices, vertical_indices)]
        horizontal_shape_eigenvalue = float(np.trace(horizontal_block) - 4.0 / 9.0)
        expected_eigenvalues = np.array(
            [
                0.0,
                0.0,
                4.0 / 9.0,
                -2.0 / 9.0,
                horizontal_shape_eigenvalue,
                -0.5 * horizontal_shape_eigenvalue,
            ]
        )
        observed_eigenvalues = np.linalg.eigvals(derivative_matrix)

        assert horizontal_shape_eigenvalue > 0.0
        assert np.linalg.norm(horizontal_block + 2.0 * vertical_block, ord=np.inf) < 1e-12
        assert np.linalg.norm(observed_eigenvalues.imag, ord=np.inf) < 1e-12
        np.testing.assert_allclose(
            np.sort(observed_eigenvalues.real),
            np.sort(expected_eigenvalues),
            rtol=1e-10,
            atol=1e-10,
        )


def test_ordered_euler_shape_eigenvalue_bounds_limit_higher_resonance_orders():
    mass_cases = (
        np.array([1.0, 0.7, 1.4]),
        np.array([10.0, 1.0, 1.0]),
        np.array([1.0, 10.0, 1.0]),
        np.array([1.0, 1.0, 10.0]),
        np.array([0.01, 1.0, 100.0]),
        np.array([100.0, 1.0, 0.01]),
    )

    for masses in mass_cases:
        ratio, _polynomial = _ordered_euler_ratio(masses)
        uncentered_line = np.array([0.0, 1.0, 1.0 + ratio])
        centered_line = uncentered_line - np.average(uncentered_line, weights=masses)
        unscaled_positions = np.column_stack([centered_line, np.zeros(3)])
        unscaled_acceleration = accelerations(unscaled_positions, masses)
        central_lambda = -float(
            np.sum(masses[:, None] * unscaled_positions * unscaled_acceleration)
            / np.sum(masses[:, None] * unscaled_positions * unscaled_positions)
        )
        scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
        normalized_positions = scale_factor * unscaled_positions
        derivative_matrix = _linearized_acceleration_matrix(
            normalized_positions,
            masses,
        )
        horizontal_block = derivative_matrix[np.ix_([0, 2, 4], [0, 2, 4])]
        horizontal_shape_eigenvalue = float(np.trace(horizontal_block) - 4.0 / 9.0)
        possible_higher_resonance_orders = [
            power
            for power in range(5, 20)
            if 4.0 / 9.0
            < power * (power - 3.0) / 9.0
            < 32.0 / 9.0
        ]

        assert 4.0 / 9.0 < horizontal_shape_eigenvalue < 32.0 / 9.0
        assert possible_higher_resonance_orders == [5, 6, 7]
        assert 8.0 * (8.0 - 3.0) / 9.0 > 32.0 / 9.0


def test_ordered_euler_resonance_surfaces_match_shape_eigenvalue_condition():
    ratio_samples_by_power = {
        5: (0.5, 1.0, 2.0),
        6: (0.4, 1.0, 2.0),
        7: (0.7, 1.0, 1.4),
    }

    for resonance_power, ratio_samples in ratio_samples_by_power.items():
        expected_shape_eigenvalue = resonance_power * (resonance_power - 3.0) / 9.0
        for ratio in ratio_samples:
            left_ratio, right_ratio = _ordered_euler_resonance_mass_ratios(
                resonance_power,
                ratio,
            )
            masses = np.array([left_ratio, 1.0, right_ratio])
            recovered_ratio, euler_polynomial = _ordered_euler_ratio(masses)
            horizontal_shape_eigenvalue = _ordered_euler_horizontal_shape_eigenvalue(
                masses,
                ratio,
            )

            assert left_ratio > 0.0
            assert right_ratio > 0.0
            assert abs(np.polyval(euler_polynomial, ratio)) < 5e-10
            assert abs(recovered_ratio - ratio) < 5e-10
            assert abs(horizontal_shape_eigenvalue - expected_shape_eigenvalue) < 5e-10


def test_analytic_cubic_time_total_collision_branch_forces_cubic_jet_kernel_condition():
    configuration, central_lambda = _central_configuration_data("equilateral")
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    quadratic_coefficient = scale_factor * configuration
    quartic_coefficient = np.array(
        [
            [0.03, 0.02],
            [-0.01, 0.04],
            [0.05, -0.02],
        ]
    )
    translation_cubic_coefficient = np.array(
        [
            [0.13, -0.07],
            [0.13, -0.07],
            [0.13, -0.07],
        ]
    )
    scaling_cubic_coefficient = 0.25 * quadratic_coefficient
    masses = np.ones(3)

    translation_derivative = _acceleration_derivative_apply(
        quadratic_coefficient,
        masses,
        translation_cubic_coefficient,
    )
    scaling_derivative = _acceleration_derivative_apply(
        quadratic_coefficient,
        masses,
        scaling_cubic_coefficient,
    )

    assert np.linalg.norm(translation_derivative, ord=np.inf) < 1e-14
    assert np.linalg.norm(scaling_derivative, ord=np.inf) > 0.01

    translation_scaled_residuals = []
    scaling_scaled_residuals = []
    for regularized_time in (1e-2, 3e-3, 1e-3):
        translation_positions = (
            quadratic_coefficient * regularized_time**2
            + translation_cubic_coefficient * regularized_time**3
            + quartic_coefficient * regularized_time**4
        )
        translation_acceleration = _cubic_time_acceleration(
            quadratic_coefficient,
            translation_cubic_coefficient,
            quartic_coefficient,
            regularized_time,
        )
        translation_scaled_residuals.append(
            regularized_time**3
            * (translation_acceleration - accelerations(translation_positions, masses))
        )

        scaling_positions = (
            quadratic_coefficient * regularized_time**2
            + scaling_cubic_coefficient * regularized_time**3
            + quartic_coefficient * regularized_time**4
        )
        scaling_acceleration = _cubic_time_acceleration(
            quadratic_coefficient,
            scaling_cubic_coefficient,
            quartic_coefficient,
            regularized_time,
        )
        scaling_scaled_residuals.append(
            regularized_time**3
            * (scaling_acceleration - accelerations(scaling_positions, masses))
        )

    translation_norms = [
        np.linalg.norm(residual, ord=np.inf) for residual in translation_scaled_residuals
    ]
    scaling_errors = [
        np.linalg.norm(residual + scaling_derivative, ord=np.inf)
        for residual in scaling_scaled_residuals
    ]

    assert translation_norms[0] > translation_norms[1] > translation_norms[2]
    assert translation_norms[-1] < 1e-4
    assert scaling_errors[0] > scaling_errors[1] > scaling_errors[2]
    np.testing.assert_allclose(
        scaling_scaled_residuals[-1],
        -scaling_derivative,
        rtol=2e-2,
        atol=1e-4,
    )


def test_finite_cubic_asymptotic_forces_cubic_jet_kernel_condition():
    configuration, central_lambda = _central_configuration_data("equilateral")
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    quadratic_coefficient = scale_factor * configuration
    quartic_coefficient = np.array(
        [
            [0.02, -0.01],
            [0.04, 0.01],
            [-0.03, 0.05],
        ]
    )
    fractional_coefficient = np.array(
        [
            [0.04, 0.02],
            [-0.03, 0.01],
            [0.02, -0.05],
        ]
    )
    fractional_coefficient -= np.average(fractional_coefficient, axis=0)
    fractional_power = 4.5
    translation_cubic_coefficient = np.array(
        [
            [0.09, -0.04],
            [0.09, -0.04],
            [0.09, -0.04],
        ]
    )
    scaling_cubic_coefficient = -0.18 * quadratic_coefficient
    masses = np.ones(3)

    translation_derivative = _acceleration_derivative_apply(
        quadratic_coefficient,
        masses,
        translation_cubic_coefficient,
    )
    scaling_derivative = _acceleration_derivative_apply(
        quadratic_coefficient,
        masses,
        scaling_cubic_coefficient,
    )

    assert np.linalg.norm(translation_derivative, ord=np.inf) < 1e-14
    assert np.linalg.norm(scaling_derivative, ord=np.inf) > 0.01

    def finite_asymptotic_state(cubic_coefficient, tau):
        positions = (
            quadratic_coefficient * tau**2
            + cubic_coefficient * tau**3
            + quartic_coefficient * tau**4
            + fractional_coefficient * tau**fractional_power
        )
        first_derivative = (
            2.0 * quadratic_coefficient * tau
            + 3.0 * cubic_coefficient * tau**2
            + 4.0 * quartic_coefficient * tau**3
            + fractional_power
            * fractional_coefficient
            * tau ** (fractional_power - 1.0)
        )
        second_derivative = (
            2.0 * quadratic_coefficient
            + 6.0 * cubic_coefficient * tau
            + 12.0 * quartic_coefficient * tau**2
            + fractional_power
            * (fractional_power - 1.0)
            * fractional_coefficient
            * tau ** (fractional_power - 2.0)
        )
        physical_acceleration = (
            second_derivative / (9.0 * tau**4)
            - 2.0 * first_derivative / (9.0 * tau**5)
        )
        return positions, physical_acceleration

    translation_scaled_residuals = []
    scaling_scaled_residuals = []
    for regularized_time in (1e-2, 3e-3, 1e-3):
        translation_positions, translation_acceleration = finite_asymptotic_state(
            translation_cubic_coefficient,
            regularized_time,
        )
        translation_scaled_residuals.append(
            regularized_time**3
            * (translation_acceleration - accelerations(translation_positions, masses))
        )

        scaling_positions, scaling_acceleration = finite_asymptotic_state(
            scaling_cubic_coefficient,
            regularized_time,
        )
        scaling_scaled_residuals.append(
            regularized_time**3
            * (scaling_acceleration - accelerations(scaling_positions, masses))
        )

    translation_norms = [
        np.linalg.norm(residual, ord=np.inf) for residual in translation_scaled_residuals
    ]
    scaling_errors = [
        np.linalg.norm(residual + scaling_derivative, ord=np.inf)
        for residual in scaling_scaled_residuals
    ]

    assert translation_norms[0] > translation_norms[1] > translation_norms[2]
    assert translation_norms[-1] < 5e-4
    assert scaling_errors[0] > scaling_errors[1] > scaling_errors[2]
    np.testing.assert_allclose(
        scaling_scaled_residuals[-1],
        -scaling_derivative,
        rtol=2e-2,
        atol=5e-4,
    )


def test_equal_mass_equilateral_cubic_jet_kernel_is_only_translation():
    configuration, central_lambda = _central_configuration_data("equilateral")
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    quadratic_coefficient = scale_factor * configuration
    complex_shape = quadratic_coefficient[:, 0] + 1j * quadratic_coefficient[:, 1]
    modes = {
        "scale": (complex_shape, 4.0 / 9.0),
        "rotation": (1j * complex_shape, -2.0 / 9.0),
        "shape_real": (np.conjugate(complex_shape), 1.0 / 9.0),
        "shape_imag": (1j * np.conjugate(complex_shape), 1.0 / 9.0),
    }
    masses = np.ones(3)

    for complex_mode, eigenvalue in modes.values():
        mode = np.column_stack([complex_mode.real, complex_mode.imag])
        derivative = _acceleration_derivative_apply(
            quadratic_coefficient,
            masses,
            mode,
        )
        assert np.linalg.norm(np.sum(mode, axis=0), ord=np.inf) < 1e-14
        assert np.linalg.norm(derivative - eigenvalue * mode, ord=np.inf) < 1e-14
        assert abs(eigenvalue) > 0.0

    for translation in (
        np.tile([1.0, 0.0], (3, 1)),
        np.tile([0.0, 1.0], (3, 1)),
    ):
        derivative = _acceleration_derivative_apply(
            quadratic_coefficient,
            masses,
            translation,
        )
        assert np.linalg.norm(derivative, ord=np.inf) == 0.0

    centered_cubic_coefficient = sum(
        weight * np.column_stack([complex_mode.real, complex_mode.imag])
        for weight, (complex_mode, _eigenvalue) in zip(
            (0.17, -0.11, 0.08, 0.05),
            modes.values(),
        )
    )
    centered_derivative = _acceleration_derivative_apply(
        quadratic_coefficient,
        masses,
        centered_cubic_coefficient,
    )

    assert np.linalg.norm(np.sum(centered_cubic_coefficient, axis=0), ord=np.inf) < 1e-14
    assert np.linalg.norm(centered_derivative, ord=np.inf) > 0.01


def test_equal_mass_equilateral_quartic_jet_is_homothetic_scale_direction():
    configuration, central_lambda = _central_configuration_data("equilateral")
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    quadratic_coefficient = scale_factor * configuration
    complex_shape = quadratic_coefficient[:, 0] + 1j * quadratic_coefficient[:, 1]
    modes = {
        "scale": (complex_shape, 4.0 / 9.0),
        "translation": (np.array([1.0 + 0.0j] * 3), 0.0),
        "rotation": (1j * complex_shape, -2.0 / 9.0),
        "shape_real": (np.conjugate(complex_shape), 1.0 / 9.0),
        "shape_imag": (1j * np.conjugate(complex_shape), 1.0 / 9.0),
    }
    masses = np.ones(3)

    obstructions = {}
    for name, (complex_mode, eigenvalue) in modes.items():
        mode = np.column_stack([complex_mode.real, complex_mode.imag])
        obstruction = (
            (4.0 / 9.0) * mode
            - _acceleration_derivative_apply(quadratic_coefficient, masses, mode)
        )
        obstructions[name] = np.linalg.norm(obstruction, ord=np.inf)
        if name == "scale":
            assert obstructions[name] < 1e-14
        else:
            assert obstructions[name] > 0.1
        assert eigenvalue != 4.0 / 9.0 or name == "scale"

    scale_quartic_coefficient = 0.2 * quadratic_coefficient
    shape_quartic_coefficient = 0.2 * np.column_stack(
        [np.conjugate(complex_shape).real, np.conjugate(complex_shape).imag]
    )
    zero_cubic_coefficient = np.zeros_like(quadratic_coefficient)
    scale_residuals = []
    shape_residuals = []
    shape_obstruction = (
        (4.0 / 9.0) * shape_quartic_coefficient
        - _acceleration_derivative_apply(
            quadratic_coefficient,
            masses,
            shape_quartic_coefficient,
        )
    )

    for regularized_time in (1e-2, 3e-3, 1e-3):
        scale_positions = (
            quadratic_coefficient * regularized_time**2
            + scale_quartic_coefficient * regularized_time**4
        )
        scale_acceleration = _cubic_time_acceleration(
            quadratic_coefficient,
            zero_cubic_coefficient,
            scale_quartic_coefficient,
            regularized_time,
        )
        scale_residuals.append(
            np.linalg.norm(
                regularized_time**2
                * (scale_acceleration - accelerations(scale_positions, masses)),
                ord=np.inf,
            )
        )

        shape_positions = (
            quadratic_coefficient * regularized_time**2
            + shape_quartic_coefficient * regularized_time**4
        )
        shape_acceleration = _cubic_time_acceleration(
            quadratic_coefficient,
            zero_cubic_coefficient,
            shape_quartic_coefficient,
            regularized_time,
        )
        shape_residuals.append(
            regularized_time**2
            * (shape_acceleration - accelerations(shape_positions, masses))
        )

    shape_errors = [
        np.linalg.norm(residual - shape_obstruction, ord=np.inf)
        for residual in shape_residuals
    ]

    assert scale_residuals[0] > scale_residuals[1] > scale_residuals[2]
    assert scale_residuals[-1] < 1e-4
    assert shape_errors[0] > shape_errors[1] > shape_errors[2]
    np.testing.assert_allclose(
        shape_residuals[-1],
        shape_obstruction,
        rtol=2e-2,
        atol=1e-6,
    )


def test_equal_mass_equilateral_quintic_and_sextic_jets_match_homothetic_energy_series():
    configuration, central_lambda = _central_configuration_data("equilateral")
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    quadratic_coefficient = scale_factor * configuration
    complex_shape = quadratic_coefficient[:, 0] + 1j * quadratic_coefficient[:, 1]
    masses = np.ones(3)
    alpha = 0.2
    quartic_coefficient = alpha * quadratic_coefficient
    sextic_coefficient = -(3.0 / 7.0) * alpha**2 * quadratic_coefficient
    shape_sextic_coefficient = 0.1 * np.column_stack(
        [np.conjugate(complex_shape).real, np.conjugate(complex_shape).imag]
    )
    quintic_modes = (
        np.tile([1.0, 0.0], (3, 1)),
        quadratic_coefficient,
        np.column_stack([(-1j * complex_shape).real, (-1j * complex_shape).imag]),
        shape_sextic_coefficient,
    )

    for quintic_coefficient in quintic_modes:
        quintic_obstruction = (
            (10.0 / 9.0) * quintic_coefficient
            - _acceleration_derivative_apply(
                quadratic_coefficient,
                masses,
                quintic_coefficient,
            )
        )
        assert np.linalg.norm(quintic_obstruction, ord=np.inf) > 0.1

    allowed_sextic_obstruction = (
        2.0 * sextic_coefficient
        - _acceleration_derivative_apply(
            quadratic_coefficient,
            masses,
            sextic_coefficient,
        )
        - 3.0
        * alpha**2
        * accelerations(quadratic_coefficient, masses)
    )
    shape_sextic_obstruction = (
        2.0 * shape_sextic_coefficient
        - _acceleration_derivative_apply(
            quadratic_coefficient,
            masses,
            shape_sextic_coefficient,
        )
        - 3.0
        * alpha**2
        * accelerations(quadratic_coefficient, masses)
    )

    assert np.linalg.norm(allowed_sextic_obstruction, ord=np.inf) < 1e-14
    assert np.linalg.norm(shape_sextic_obstruction, ord=np.inf) > 0.1

    allowed_residuals = []
    forbidden_residuals = []
    for regularized_time in (1e-1, 3e-2, 1e-2):
        allowed_positions = (
            quadratic_coefficient * regularized_time**2
            + quartic_coefficient * regularized_time**4
            + sextic_coefficient * regularized_time**6
        )
        allowed_acceleration = _cubic_time_polynomial_acceleration(
            {
                2: quadratic_coefficient,
                4: quartic_coefficient,
                6: sextic_coefficient,
            },
            regularized_time,
        )
        allowed_residuals.append(
            np.linalg.norm(
                allowed_acceleration - accelerations(allowed_positions, masses),
                ord=np.inf,
            )
        )

        forbidden_positions = (
            quadratic_coefficient * regularized_time**2
            + quartic_coefficient * regularized_time**4
            + shape_sextic_coefficient * regularized_time**6
        )
        forbidden_acceleration = _cubic_time_polynomial_acceleration(
            {
                2: quadratic_coefficient,
                4: quartic_coefficient,
                6: shape_sextic_coefficient,
            },
            regularized_time,
        )
        forbidden_residuals.append(
            forbidden_acceleration - accelerations(forbidden_positions, masses)
        )

    forbidden_errors = [
        np.linalg.norm(residual - shape_sextic_obstruction, ord=np.inf)
        for residual in forbidden_residuals
    ]

    assert allowed_residuals[0] > allowed_residuals[1] > allowed_residuals[2]
    assert allowed_residuals[-1] < 1e-5
    assert forbidden_errors[0] > forbidden_errors[1] > forbidden_errors[2]
    np.testing.assert_allclose(
        forbidden_residuals[-1],
        shape_sextic_obstruction,
        rtol=2e-2,
        atol=1e-6,
    )


def test_equal_mass_equilateral_all_order_formal_rigidity_has_only_homothetic_resonance():
    configuration, central_lambda = _central_configuration_data("equilateral")
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    quadratic_coefficient = scale_factor * configuration
    eigenvalues = np.array([0.0, 0.0, 4.0 / 9.0, -2.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0])
    resonances = {
        power: [
            eigenvalue
            for eigenvalue in eigenvalues
            if abs(power * (power - 3.0) / 9.0 - eigenvalue) < 1e-14
        ]
        for power in range(3, 31)
    }

    assert resonances[3] == [0.0, 0.0]
    assert resonances[4] == [4.0 / 9.0]
    assert all(not resonances[power] for power in range(5, 31))

    quartic_scale_coefficient = 0.2
    energy_per_inertia = (10.0 / 9.0) * quartic_scale_coefficient
    homothetic_coefficients = _homothetic_energy_series_coefficients(
        energy_per_inertia,
        order=6,
    )

    assert homothetic_coefficients[1] == pytest.approx(quartic_scale_coefficient)
    assert homothetic_coefficients[2] == pytest.approx(
        -(3.0 / 7.0) * quartic_scale_coefficient**2
    )

    residual_by_order = {}
    regularized_time = 0.08
    for retained_order in (1, 2, 3, 4):
        coefficients_by_power = {
            2 + 2 * coefficient_index: (
                homothetic_coefficients[coefficient_index] * quadratic_coefficient
            )
            for coefficient_index in range(retained_order + 1)
        }
        positions = sum(
            coefficient * regularized_time**power
            for power, coefficient in coefficients_by_power.items()
        )
        acceleration = _cubic_time_polynomial_acceleration(
            coefficients_by_power,
            regularized_time,
        )
        residual_by_order[retained_order] = np.linalg.norm(
            acceleration - accelerations(positions, np.ones(3)),
            ord=np.inf,
        )

    assert residual_by_order[1] > residual_by_order[2] > residual_by_order[3]
    assert residual_by_order[3] > residual_by_order[4]
    assert residual_by_order[4] < 1e-8


@pytest.mark.parametrize(
    "masses",
    (
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 0.7, 1.4]),
        np.array([10.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 2.5]),
    ),
)
def test_arbitrary_mass_equilateral_linearized_spectrum_matches_beta_formula(masses):
    configuration, central_lambda = _mass_centered_equilateral_central_configuration(masses)
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    quadratic_coefficient = scale_factor * configuration
    derivative_matrix = _linearized_acceleration_matrix(quadratic_coefficient, masses)
    observed_eigenvalues = np.linalg.eigvals(derivative_matrix)

    assert np.linalg.norm(observed_eigenvalues.imag, ord=np.inf) < 1e-12

    total_mass = float(np.sum(masses))
    beta = float(
        (masses[0] * masses[1] + masses[0] * masses[2] + masses[1] * masses[2])
        / total_mass**2
    )
    shape_gap = (1.0 / 3.0) * np.sqrt(1.0 - 3.0 * beta)
    expected_eigenvalues = np.array(
        [
            0.0,
            0.0,
            4.0 / 9.0,
            -2.0 / 9.0,
            1.0 / 9.0 + shape_gap,
            1.0 / 9.0 - shape_gap,
        ]
    )

    np.testing.assert_allclose(
        np.sort(observed_eigenvalues.real),
        np.sort(expected_eigenvalues),
        rtol=1e-12,
        atol=1e-12,
    )

    resonances = {
        power: [
            eigenvalue
            for eigenvalue in expected_eigenvalues
            if abs(power * (power - 3.0) / 9.0 - eigenvalue) < 1e-12
        ]
        for power in range(3, 31)
    }
    expected_cubic_resonance_count = 3 if abs(beta - 8.0 / 27.0) < 1e-12 else 2

    assert len(resonances[3]) == expected_cubic_resonance_count
    assert resonances[4] == [4.0 / 9.0]
    assert all(not resonances[power] for power in range(5, 31))


def test_arbitrary_mass_equilateral_beta_resonance_adds_centered_cubic_kernel():
    masses = np.array([1.0, 1.0, 2.5])
    configuration, central_lambda = _mass_centered_equilateral_central_configuration(masses)
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    quadratic_coefficient = scale_factor * configuration
    derivative_matrix = _linearized_acceleration_matrix(quadratic_coefficient, masses)
    _left_vectors, _singular_values, nullspace_vh = np.linalg.svd(derivative_matrix)
    nullspace = nullspace_vh.T[:, -3:]
    mass_center_constraints = np.zeros((2, 6))
    mass_center_constraints[0, 0::2] = masses
    mass_center_constraints[1, 1::2] = masses
    _constraint_left, _constraint_singular_values, constraint_vh = np.linalg.svd(
        mass_center_constraints @ nullspace
    )
    centered_null_coefficients = constraint_vh.T[:, -1]
    centered_cubic_coefficient = (nullspace @ centered_null_coefficients).reshape(3, 2)
    pair_differences = [
        np.linalg.norm(centered_cubic_coefficient[left] - centered_cubic_coefficient[right])
        for left in range(3)
        for right in range(left + 1, 3)
    ]

    np.testing.assert_allclose(
        np.sum(masses[:, None] * centered_cubic_coefficient, axis=0),
        np.zeros(2),
        atol=1e-12,
    )
    assert np.linalg.norm(
        _acceleration_derivative_apply(
            quadratic_coefficient,
            masses,
            centered_cubic_coefficient,
        ),
        ord=np.inf,
    ) < 1e-12
    assert max(pair_differences) > 0.5


def test_resonant_arbitrary_mass_equilateral_cubic_kernel_survives_quartic_solvability():
    masses = np.array([1.0, 1.0, 2.5])
    configuration, central_lambda = _mass_centered_equilateral_central_configuration(masses)
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    quadratic_coefficient = scale_factor * configuration
    derivative_matrix = _linearized_acceleration_matrix(quadratic_coefficient, masses)
    cubic_coefficient = np.array(
        [
            [-1.0, np.sqrt(3.0)],
            [2.0, 0.0],
            [-0.4, -0.4 * np.sqrt(3.0)],
        ]
    )
    cubic_coefficient = cubic_coefficient / np.sqrt(
        np.sum(masses[:, None] * cubic_coefficient * cubic_coefficient)
    )
    second_force_coefficient = _acceleration_second_order_coefficient(
        quadratic_coefficient,
        masses,
        cubic_coefficient,
    )
    scale_left_vector = masses[:, None] * quadratic_coefficient
    quartic_operator = (4.0 / 9.0) * np.eye(6) - derivative_matrix
    quartic_particular, *_ = np.linalg.lstsq(
        quartic_operator,
        second_force_coefficient.reshape(-1),
        rcond=None,
    )
    quartic_particular = quartic_particular.reshape(3, 2)
    quartic_residual = (
        (4.0 / 9.0) * quartic_particular
        - _acceleration_derivative_apply(
            quadratic_coefficient,
            masses,
            quartic_particular,
        )
        - second_force_coefficient
    )

    np.testing.assert_allclose(
        np.sum(masses[:, None] * cubic_coefficient, axis=0),
        np.zeros(2),
        atol=1e-14,
    )
    assert np.linalg.norm(
        _acceleration_derivative_apply(
            quadratic_coefficient,
            masses,
            cubic_coefficient,
        ),
        ord=np.inf,
    ) < 1e-14
    assert abs(float(np.sum(scale_left_vector * second_force_coefficient))) < 1e-14
    assert np.linalg.norm(quartic_residual, ord=np.inf) < 1e-14

    residuals = []
    for regularized_time in (1e-2, 3e-3, 1e-3):
        positions = (
            quadratic_coefficient * regularized_time**2
            + cubic_coefficient * regularized_time**3
            + quartic_particular * regularized_time**4
        )
        acceleration = _cubic_time_polynomial_acceleration(
            {
                2: quadratic_coefficient,
                3: cubic_coefficient,
                4: quartic_particular,
            },
            regularized_time,
        )
        residuals.append(
            np.linalg.norm(
                regularized_time**2 * (acceleration - accelerations(positions, masses)),
                ord=np.inf,
            )
        )

    assert residuals[0] > residuals[1] > residuals[2]
    assert residuals[-1] < 2e-4


def test_resonant_arbitrary_mass_equilateral_series_recurrence_builds_higher_order_branch():
    masses, shape_coefficients = _resonant_equilateral_shape_coefficients(
        order=8,
        cubic_amplitude=0.2,
        quartic_scale=0.07,
    )
    derivative_matrix = _linearized_acceleration_matrix(shape_coefficients[0], masses)
    cubic_pair_differences = [
        np.linalg.norm(shape_coefficients[1, left] - shape_coefficients[1, right])
        for left in range(3)
        for right in range(left + 1, 3)
    ]

    assert max(cubic_pair_differences) > 0.1
    assert np.linalg.norm(
        _acceleration_derivative_apply(shape_coefficients[0], masses, shape_coefficients[1]),
        ord=np.inf,
    ) < 1e-14

    for degree in range(1, 9):
        np.testing.assert_allclose(
            np.sum(masses[:, None] * shape_coefficients[degree], axis=0),
            np.zeros(2),
            atol=1e-12,
        )

    for degree in range(2, 9):
        trial_coefficients = shape_coefficients.copy()
        trial_coefficients[degree] = 0.0
        known_term = _shape_acceleration_series(
            trial_coefficients,
            masses,
            degree,
        )[degree].reshape(-1)
        multiplier = (degree + 2.0) * (degree - 1.0) / 9.0
        recurrence_residual = (
            (multiplier * np.eye(6) - derivative_matrix)
            @ shape_coefficients[degree].reshape(-1)
            - known_term
        )
        assert np.linalg.norm(recurrence_residual, ord=np.inf) < 1e-12

    regularized_time = 0.05
    residual_by_retained_order = {}
    for retained_order in (2, 3, 4, 6, 8):
        coefficients_by_power = {
            degree + 2: shape_coefficients[degree]
            for degree in range(retained_order + 1)
        }
        positions = sum(
            coefficient * regularized_time**power
            for power, coefficient in coefficients_by_power.items()
        )
        acceleration = _cubic_time_polynomial_acceleration(
            coefficients_by_power,
            regularized_time,
        )
        residual_by_retained_order[retained_order] = np.linalg.norm(
            acceleration - accelerations(positions, masses),
            ord=np.inf,
        )

    assert (
        residual_by_retained_order[2]
        > residual_by_retained_order[3]
        > residual_by_retained_order[4]
        > residual_by_retained_order[6]
        > residual_by_retained_order[8]
    )
    assert residual_by_retained_order[8] < 1e-8


@pytest.mark.parametrize(
    "masses",
    (
        np.array([1.0, 1.0, 2.5]),
        np.array(
            [
                0.2,
                0.4 - 4.0 / (15.0 * np.sqrt(3.0)),
                0.4 + 4.0 / (15.0 * np.sqrt(3.0)),
            ]
        ),
    ),
)
def test_equilateral_beta_resonance_surface_builds_local_cubic_branches(masses):
    total_mass = float(np.sum(masses))
    beta = float(
        (masses[0] * masses[1] + masses[0] * masses[2] + masses[1] * masses[2])
        / total_mass**2
    )
    returned_masses, shape_coefficients = _resonant_equilateral_shape_coefficients(
        order=8,
        cubic_amplitude=0.16,
        quartic_scale=0.05,
        masses=masses,
    )
    derivative_matrix = _linearized_acceleration_matrix(shape_coefficients[0], masses)
    cubic_kernel = shape_coefficients[1] / 0.16
    second_force_coefficient = _acceleration_second_order_coefficient(
        shape_coefficients[0],
        masses,
        cubic_kernel,
    )
    scale_left_vector = masses[:, None] * shape_coefficients[0]
    cubic_pair_differences = [
        np.linalg.norm(shape_coefficients[1, left] - shape_coefficients[1, right])
        for left in range(3)
        for right in range(left + 1, 3)
    ]

    np.testing.assert_allclose(returned_masses, masses)
    assert abs(beta - 8.0 / 27.0) < 1e-14
    assert max(cubic_pair_differences) > 0.1
    assert np.linalg.norm(
        _acceleration_derivative_apply(shape_coefficients[0], masses, cubic_kernel),
        ord=np.inf,
    ) < 2e-13
    assert abs(float(np.sum(scale_left_vector * second_force_coefficient))) < 2e-13

    for degree in range(1, 9):
        np.testing.assert_allclose(
            np.sum(masses[:, None] * shape_coefficients[degree], axis=0),
            np.zeros(2),
            atol=1e-12,
        )
        trial_coefficients = shape_coefficients.copy()
        trial_coefficients[degree] = 0.0
        known_term = _shape_acceleration_series(
            trial_coefficients,
            masses,
            degree,
        )[degree].reshape(-1)
        multiplier = (degree + 2.0) * (degree - 1.0) / 9.0
        recurrence_residual = (
            (multiplier * np.eye(6) - derivative_matrix)
            @ shape_coefficients[degree].reshape(-1)
            - known_term
        )
        assert np.linalg.norm(recurrence_residual, ord=np.inf) < 2e-12

    regularized_time = 0.04
    residual_by_retained_order = {}
    for retained_order in (2, 3, 4, 6, 8):
        coefficients_by_power = {
            degree + 2: shape_coefficients[degree]
            for degree in range(retained_order + 1)
        }
        positions = sum(
            coefficient * regularized_time**power
            for power, coefficient in coefficients_by_power.items()
        )
        acceleration = _cubic_time_polynomial_acceleration(
            coefficients_by_power,
            regularized_time,
        )
        residual_by_retained_order[retained_order] = np.linalg.norm(
            acceleration - accelerations(positions, masses),
            ord=np.inf,
        )

    assert residual_by_retained_order[2] > residual_by_retained_order[4]
    assert residual_by_retained_order[4] > residual_by_retained_order[8]
    assert residual_by_retained_order[8] < 1e-8


def test_equal_mass_euler_all_order_formal_rigidity_has_only_homothetic_resonance():
    configuration, central_lambda = _central_configuration_data("euler")
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    quadratic_coefficient = scale_factor * configuration
    line_shape = np.array([-1.0, 0.0, 1.0])
    centered_shape = np.array([1.0, -2.0, 1.0])
    modes = {
        "horizontal_translation": (
            np.column_stack([np.ones(3), np.zeros(3)]),
            0.0,
        ),
        "scale": (quadratic_coefficient, 4.0 / 9.0),
        "horizontal_shape": (
            np.column_stack([centered_shape, np.zeros(3)]),
            16.0 / 15.0,
        ),
        "vertical_translation": (
            np.column_stack([np.zeros(3), np.ones(3)]),
            0.0,
        ),
        "rotation": (
            np.column_stack([np.zeros(3), line_shape]),
            -2.0 / 9.0,
        ),
        "vertical_shape": (
            np.column_stack([np.zeros(3), centered_shape]),
            -8.0 / 15.0,
        ),
    }
    masses = np.ones(3)

    for mode, eigenvalue in modes.values():
        derivative = _acceleration_derivative_apply(
            quadratic_coefficient,
            masses,
            mode,
        )
        assert np.linalg.norm(derivative - eigenvalue * mode, ord=np.inf) < 1e-14

    eigenvalues = np.array([0.0, 0.0, 4.0 / 9.0, -2.0 / 9.0, 16.0 / 15.0, -8.0 / 15.0])
    resonances = {
        power: [
            eigenvalue
            for eigenvalue in eigenvalues
            if abs(power * (power - 3.0) / 9.0 - eigenvalue) < 1e-14
        ]
        for power in range(3, 31)
    }

    assert resonances[3] == [0.0, 0.0]
    assert resonances[4] == [4.0 / 9.0]
    assert all(not resonances[power] for power in range(5, 31))

    quartic_scale_coefficient = 0.2
    energy_per_inertia = (10.0 / 9.0) * quartic_scale_coefficient
    homothetic_coefficients = _homothetic_energy_series_coefficients(
        energy_per_inertia,
        order=6,
    )

    residual_by_order = {}
    regularized_time = 0.08
    for retained_order in (1, 2, 3, 4):
        coefficients_by_power = {
            2 + 2 * coefficient_index: (
                homothetic_coefficients[coefficient_index] * quadratic_coefficient
            )
            for coefficient_index in range(retained_order + 1)
        }
        positions = sum(
            coefficient * regularized_time**power
            for power, coefficient in coefficients_by_power.items()
        )
        acceleration = _cubic_time_polynomial_acceleration(
            coefficients_by_power,
            regularized_time,
        )
        residual_by_order[retained_order] = np.linalg.norm(
            acceleration - accelerations(positions, masses),
            ord=np.inf,
        )

    assert residual_by_order[1] > residual_by_order[2] > residual_by_order[3]
    assert residual_by_order[3] > residual_by_order[4]
    assert residual_by_order[4] < 1e-8


@pytest.mark.parametrize("middle_mass", (1.0, 11.0 / 12.0, 0.25, 1.0 / 24.0, 2.0))
def test_symmetric_unequal_mass_euler_linearized_spectrum_matches_formula(middle_mass):
    masses, configuration, central_lambda = _symmetric_euler_central_configuration(middle_mass)
    scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    quadratic_coefficient = scale_factor * configuration
    derivative_matrix = _linearized_acceleration_matrix(quadratic_coefficient, masses)
    horizontal_shape_eigenvalue = (
        16.0 * (middle_mass + 2.0) / (9.0 * (4.0 * middle_mass + 1.0))
    )
    vertical_shape_eigenvalue = (
        -8.0 * (middle_mass + 2.0) / (9.0 * (4.0 * middle_mass + 1.0))
    )
    expected_eigenvalues = np.array(
        [
            0.0,
            0.0,
            4.0 / 9.0,
            -2.0 / 9.0,
            horizontal_shape_eigenvalue,
            vertical_shape_eigenvalue,
        ]
    )
    observed_eigenvalues = np.linalg.eigvals(derivative_matrix)

    assert np.linalg.norm(observed_eigenvalues.imag, ord=np.inf) < 1e-12
    np.testing.assert_allclose(
        np.sort(observed_eigenvalues.real),
        np.sort(expected_eigenvalues),
        rtol=1e-12,
        atol=1e-12,
    )

    horizontal_shape_mode = np.column_stack(
        [
            np.array([1.0, -2.0 / middle_mass, 1.0]),
            np.zeros(3),
        ]
    )
    derivative = _acceleration_derivative_apply(
        quadratic_coefficient,
        masses,
        horizontal_shape_mode,
    )
    np.testing.assert_allclose(
        np.sum(masses[:, None] * horizontal_shape_mode, axis=0),
        np.zeros(2),
        atol=1e-14,
    )
    assert (
        np.linalg.norm(
            derivative - horizontal_shape_eigenvalue * horizontal_shape_mode,
            ord=np.inf,
        )
        < 1e-12
    )


def test_symmetric_unequal_mass_euler_has_three_higher_order_resonance_masses():
    resonance_cases = {
        5: 11.0 / 12.0,
        6: 0.25,
        7: 1.0 / 24.0,
    }

    for power, middle_mass in resonance_cases.items():
        masses, configuration, central_lambda = _symmetric_euler_central_configuration(middle_mass)
        scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
        quadratic_coefficient = scale_factor * configuration
        derivative_matrix = _linearized_acceleration_matrix(quadratic_coefficient, masses)
        horizontal_shape_mode = np.column_stack(
            [
                np.array([1.0, -2.0 / middle_mass, 1.0]),
                np.zeros(3),
            ]
        )
        multiplier = power * (power - 3.0) / 9.0
        resonance_residual = (
            multiplier * horizontal_shape_mode
            - _acceleration_derivative_apply(
                quadratic_coefficient,
                masses,
                horizontal_shape_mode,
            )
        )

        assert np.linalg.norm(resonance_residual, ord=np.inf) < 1e-12

        eigenvalues = np.linalg.eigvals(derivative_matrix).real
        resonant_powers = [
            candidate_power
            for candidate_power in range(5, 15)
            if any(
                abs(candidate_power * (candidate_power - 3.0) / 9.0 - eigenvalue)
                < 1e-12
                for eigenvalue in eigenvalues
            )
        ]
        assert resonant_powers == [power]

    nonresonant_masses, nonresonant_configuration, nonresonant_lambda = (
        _symmetric_euler_central_configuration(2.0)
    )
    nonresonant_scale = ((9.0 / 2.0) * nonresonant_lambda) ** (1.0 / 3.0)
    nonresonant_eigenvalues = np.linalg.eigvals(
        _linearized_acceleration_matrix(
            nonresonant_scale * nonresonant_configuration,
            nonresonant_masses,
        )
    ).real

    assert all(
        not any(
            abs(power * (power - 3.0) / 9.0 - eigenvalue) < 1e-12
            for eigenvalue in nonresonant_eigenvalues
        )
        for power in range(5, 15)
    )


@pytest.mark.parametrize("resonance_power", (5, 6, 7))
def test_symmetric_euler_resonances_build_higher_order_collinear_branches(
    resonance_power,
):
    masses, shape_coefficients = _symmetric_euler_resonant_shape_coefficients(
        resonance_power=resonance_power,
        order=10,
        resonance_amplitude=0.05,
        quartic_scale=0.04,
    )
    derivative_matrix = _linearized_acceleration_matrix(shape_coefficients[0], masses)
    resonance_degree = resonance_power - 2
    resonance_multiplier = resonance_power * (resonance_power - 3.0) / 9.0
    middle_mass = masses[1]
    horizontal_shape_mode = np.column_stack(
        [
            np.array([1.0, -2.0 / middle_mass, 1.0]),
            np.zeros(3),
        ]
    )

    assert np.linalg.norm(shape_coefficients[1], ord=np.inf) == 0.0
    assert np.linalg.norm(shape_coefficients[resonance_degree], ord=np.inf) > 0.01
    assert np.linalg.norm(shape_coefficients[:, :, 1], ord=np.inf) < 1e-14
    assert (
        np.linalg.norm(
            resonance_multiplier * horizontal_shape_mode
            - _acceleration_derivative_apply(
                shape_coefficients[0],
                masses,
                horizontal_shape_mode,
            ),
            ord=np.inf,
        )
        < 1e-12
    )

    for degree in range(1, 11):
        np.testing.assert_allclose(
            np.sum(masses[:, None] * shape_coefficients[degree], axis=0),
            np.zeros(2),
            atol=1e-12,
        )
        trial_coefficients = shape_coefficients.copy()
        trial_coefficients[degree] = 0.0
        known_term = _shape_acceleration_series(
            trial_coefficients,
            masses,
            degree,
        )[degree].reshape(-1)
        multiplier = (degree + 2.0) * (degree - 1.0) / 9.0
        recurrence_residual = (
            (multiplier * np.eye(6) - derivative_matrix)
            @ shape_coefficients[degree].reshape(-1)
            - known_term
        )
        assert np.linalg.norm(recurrence_residual, ord=np.inf) < 1e-12

    regularized_time = 0.05
    residual_by_retained_order = {}
    retained_orders = tuple(
        order
        for order in (resonance_degree, resonance_degree + 1, resonance_degree + 2, 8, 10)
        if order <= 10
    )
    for retained_order in retained_orders:
        coefficients_by_power = {
            degree + 2: shape_coefficients[degree]
            for degree in range(retained_order + 1)
        }
        positions = sum(
            coefficient * regularized_time**power
            for power, coefficient in coefficients_by_power.items()
        )
        acceleration = _cubic_time_polynomial_acceleration(
            coefficients_by_power,
            regularized_time,
        )
        residual_by_retained_order[retained_order] = np.linalg.norm(
            acceleration - accelerations(positions, masses),
            ord=np.inf,
        )

    assert (
        residual_by_retained_order[retained_orders[0]]
        > residual_by_retained_order[retained_orders[2]]
    )
    assert residual_by_retained_order[retained_orders[-1]] < 1e-8


@pytest.mark.parametrize(
    ("resonance_power", "ratio"),
    (
        (5, 0.5),
        (6, 0.4),
        (7, 0.7),
    ),
)
def test_ordered_euler_resonance_surfaces_build_local_collinear_branches(
    resonance_power,
    ratio,
):
    masses, horizontal_shape_mode, shape_coefficients = (
        _ordered_euler_resonant_shape_coefficients(
            resonance_power=resonance_power,
            ratio=ratio,
            order=10,
            resonance_amplitude=0.04,
            quartic_scale=0.03,
        )
    )
    derivative_matrix = _linearized_acceleration_matrix(shape_coefficients[0], masses)
    resonance_degree = resonance_power - 2
    resonance_multiplier = resonance_power * (resonance_power - 3.0) / 9.0

    assert np.linalg.norm(shape_coefficients[1], ord=np.inf) == 0.0
    assert np.linalg.norm(shape_coefficients[resonance_degree], ord=np.inf) > 0.01
    assert np.linalg.norm(shape_coefficients[:, :, 1], ord=np.inf) < 1e-14
    assert (
        np.linalg.norm(
            resonance_multiplier * horizontal_shape_mode
            - _acceleration_derivative_apply(
                shape_coefficients[0],
                masses,
                horizontal_shape_mode,
            ),
            ord=np.inf,
        )
        < 2e-12
    )

    for degree in range(1, 11):
        np.testing.assert_allclose(
            np.sum(masses[:, None] * shape_coefficients[degree], axis=0),
            np.zeros(2),
            atol=1e-12,
        )
        trial_coefficients = shape_coefficients.copy()
        trial_coefficients[degree] = 0.0
        known_term = _shape_acceleration_series(
            trial_coefficients,
            masses,
            degree,
        )[degree].reshape(-1)
        multiplier = (degree + 2.0) * (degree - 1.0) / 9.0
        recurrence_residual = (
            (multiplier * np.eye(6) - derivative_matrix)
            @ shape_coefficients[degree].reshape(-1)
            - known_term
        )
        assert np.linalg.norm(recurrence_residual, ord=np.inf) < 2e-12

    regularized_time = 0.04
    residual_by_retained_order = {}
    retained_orders = tuple(
        order
        for order in (resonance_degree, resonance_degree + 1, resonance_degree + 2, 8, 10)
        if order <= 10
    )
    for retained_order in retained_orders:
        coefficients_by_power = {
            degree + 2: shape_coefficients[degree]
            for degree in range(retained_order + 1)
        }
        positions = sum(
            coefficient * regularized_time**power
            for power, coefficient in coefficients_by_power.items()
        )
        acceleration = _cubic_time_polynomial_acceleration(
            coefficients_by_power,
            regularized_time,
        )
        residual_by_retained_order[retained_order] = np.linalg.norm(
            acceleration - accelerations(positions, masses),
            ord=np.inf,
        )

    assert (
        residual_by_retained_order[retained_orders[0]]
        > residual_by_retained_order[retained_orders[2]]
    )
    assert residual_by_retained_order[retained_orders[-1]] < 1e-7


def test_nonresonant_ordered_euler_branches_have_only_homothetic_resonance():
    mass_cases = (
        np.array([1.0, 0.7, 1.4]),
        np.array([1.3, 0.9, 2.0]),
        np.array([2.0, 1.0, 1.0]),
    )
    quartic_scale_coefficient = 0.18
    energy_per_inertia = (10.0 / 9.0) * quartic_scale_coefficient
    homothetic_coefficients = _homothetic_energy_series_coefficients(
        energy_per_inertia,
        order=6,
    )

    for masses in mass_cases:
        _ratio, quadratic_coefficient = _ordered_euler_scaled_configuration(masses)
        derivative_matrix = _linearized_acceleration_matrix(quadratic_coefficient, masses)
        eigenvalues = np.linalg.eigvals(derivative_matrix).real
        shape_coefficients = np.zeros((13, 3, 2))
        for coefficient_index, scalar_coefficient in enumerate(homothetic_coefficients):
            shape_coefficients[2 * coefficient_index] = (
                scalar_coefficient * quadratic_coefficient
            )

        assert all(
            not any(
                abs(power * (power - 3.0) / 9.0 - eigenvalue) < 1e-9
                for eigenvalue in eigenvalues
            )
            for power in range(5, 20)
        )
        for degree in range(3, 13):
            multiplier = (degree + 2.0) * (degree - 1.0) / 9.0
            smallest_singular_value = np.linalg.svd(
                multiplier * np.eye(6) - derivative_matrix,
                compute_uv=False,
            )[-1]
            assert smallest_singular_value > 0.05

        for degree in range(1, 13):
            np.testing.assert_allclose(
                np.sum(masses[:, None] * shape_coefficients[degree], axis=0),
                np.zeros(2),
                atol=1e-12,
            )
            trial_coefficients = shape_coefficients.copy()
            trial_coefficients[degree] = 0.0
            known_term = _shape_acceleration_series(
                trial_coefficients,
                masses,
                degree,
            )[degree].reshape(-1)
            multiplier = (degree + 2.0) * (degree - 1.0) / 9.0
            recurrence_residual = (
                (multiplier * np.eye(6) - derivative_matrix)
                @ shape_coefficients[degree].reshape(-1)
                - known_term
            )
            assert np.linalg.norm(recurrence_residual, ord=np.inf) < 2e-12

        regularized_time = 0.05
        residual_by_retained_order = {}
        for retained_even_index in (1, 2, 3, 4, 6):
            coefficients_by_power = {
                2 + 2 * coefficient_index: (
                    homothetic_coefficients[coefficient_index] * quadratic_coefficient
                )
                for coefficient_index in range(retained_even_index + 1)
            }
            positions = sum(
                coefficient * regularized_time**power
                for power, coefficient in coefficients_by_power.items()
            )
            acceleration = _cubic_time_polynomial_acceleration(
                coefficients_by_power,
                regularized_time,
            )
            residual_by_retained_order[retained_even_index] = np.linalg.norm(
                acceleration - accelerations(positions, masses),
                ord=np.inf,
            )

        assert residual_by_retained_order[1] > residual_by_retained_order[2]
        assert residual_by_retained_order[2] > residual_by_retained_order[4]
        assert residual_by_retained_order[6] < 1e-9


def _extra_local_normal_form_resonance_powers(eigenvalues):
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    extra_powers = []
    zero_multiplicity = int(np.sum(np.abs(eigenvalues) < 1e-9))
    extra_powers.extend([3] * max(0, zero_multiplicity - 2))
    for power in range(5, 20):
        multiplier = power * (power - 3.0) / 9.0
        if any(abs(multiplier - eigenvalue) < 1e-9 for eigenvalue in eigenvalues):
            extra_powers.append(power)
    return tuple(extra_powers)


def _indicial_roots_for_linearized_eigenvalue(eigenvalue):
    discriminant = 9.0 + 36.0 * eigenvalue
    root = np.lib.scimath.sqrt(discriminant)
    return ((-1.0 + root) / 2.0, (-1.0 - root) / 2.0)


def test_three_body_zero_angular_local_normal_form_case_split_is_exhaustive():
    beta_resonant_masses = np.array(
        [
            0.2,
            0.4 - 4.0 / (15.0 * np.sqrt(3.0)),
            0.4 + 4.0 / (15.0 * np.sqrt(3.0)),
        ]
    )
    cases = []

    for label, masses in (
        ("equilateral_nonresonant", np.array([1.0, 0.7, 1.4])),
        ("equilateral_beta_resonant", beta_resonant_masses),
    ):
        configuration, central_lambda = _mass_centered_equilateral_central_configuration(masses)
        scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
        cases.append((label, masses, scale_factor * configuration))

    ordered_nonresonant_masses = np.array([1.0, 0.7, 1.4])
    _ratio, ordered_nonresonant_configuration = _ordered_euler_scaled_configuration(
        ordered_nonresonant_masses,
    )
    cases.append(
        (
            "ordered_euler_nonresonant",
            ordered_nonresonant_masses,
            ordered_nonresonant_configuration,
        )
    )

    for resonance_power, ratio in ((5, 0.5), (6, 0.4), (7, 0.7)):
        left_ratio, right_ratio = _ordered_euler_resonance_mass_ratios(
            resonance_power,
            ratio,
        )
        masses = np.array([left_ratio, 1.0, right_ratio])
        _ratio, configuration = _ordered_euler_scaled_configuration(masses)
        cases.append((f"ordered_euler_resonant_{resonance_power}", masses, configuration))

    expected_extra_powers = {
        "equilateral_nonresonant": (),
        "equilateral_beta_resonant": (3,),
        "ordered_euler_nonresonant": (),
        "ordered_euler_resonant_5": (5,),
        "ordered_euler_resonant_6": (6,),
        "ordered_euler_resonant_7": (7,),
    }

    for label, masses, configuration in cases:
        derivative_matrix = _linearized_acceleration_matrix(configuration, masses)
        eigenvalues = np.linalg.eigvals(derivative_matrix)
        real_eigenvalues = eigenvalues.real
        extra_powers = _extra_local_normal_form_resonance_powers(real_eigenvalues)

        assert np.linalg.norm(eigenvalues.imag, ord=np.inf) < 1e-10
        assert np.sum(np.abs(real_eigenvalues) < 1e-9) in (2, 3)
        assert np.sum(np.abs(real_eigenvalues - 4.0 / 9.0) < 1e-9) == 1
        assert extra_powers == expected_extra_powers[label]
        assert len(extra_powers) <= 1


def test_three_body_collision_free_central_targets_are_reduced_hyperbolic():
    beta_resonant_masses = np.array(
        [
            0.2,
            0.4 - 4.0 / (15.0 * np.sqrt(3.0)),
            0.4 + 4.0 / (15.0 * np.sqrt(3.0)),
        ]
    )
    cases = []

    for label, masses in (
        ("equilateral_nonresonant", np.array([1.0, 0.7, 1.4])),
        ("equilateral_equal", np.array([1.0, 1.0, 1.0])),
        ("equilateral_beta_resonant", beta_resonant_masses),
    ):
        configuration, central_lambda = _mass_centered_equilateral_central_configuration(masses)
        scale_factor = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
        cases.append((label, masses, scale_factor * configuration))

    for label, masses in (
        ("ordered_euler_nonresonant", np.array([1.0, 0.7, 1.4])),
        ("ordered_euler_equal", np.array([1.0, 1.0, 1.0])),
    ):
        _ratio, configuration = _ordered_euler_scaled_configuration(masses)
        cases.append((label, masses, configuration))

    for resonance_power, ratio in ((5, 0.5), (6, 0.4), (7, 0.7)):
        left_ratio, right_ratio = _ordered_euler_resonance_mass_ratios(
            resonance_power,
            ratio,
        )
        masses = np.array([left_ratio, 1.0, right_ratio])
        _ratio, configuration = _ordered_euler_scaled_configuration(masses)
        cases.append((f"ordered_euler_resonant_{resonance_power}", masses, configuration))

    for _label, masses, configuration in cases:
        derivative_matrix = _linearized_acceleration_matrix(configuration, masses)
        eigenvalues = np.linalg.eigvals(derivative_matrix)
        real_eigenvalues = eigenvalues.real
        center_indicial_roots = [
            root
            for eigenvalue in real_eigenvalues
            for root in _indicial_roots_for_linearized_eigenvalue(eigenvalue)
            if abs(root.real) < 1.0e-9 and abs(root.imag) < 1.0e-9
        ]

        assert np.linalg.norm(eigenvalues.imag, ord=np.inf) < 1e-10
        assert np.sum(np.abs(real_eigenvalues + 2.0 / 9.0) < 1e-9) == 1
        assert len(center_indicial_roots) == 1
        assert abs(center_indicial_roots[0]) < 1.0e-9

        quotient_eigenvalues = [
            eigenvalue
            for eigenvalue in real_eigenvalues
            if abs(eigenvalue + 2.0 / 9.0) >= 1e-9
        ]
        quotient_roots = [
            root
            for eigenvalue in quotient_eigenvalues
            for root in _indicial_roots_for_linearized_eigenvalue(eigenvalue)
        ]
        assert all(abs(root.real) > 1.0e-6 for root in quotient_roots)


def test_local_normal_form_branch_parameters_project_to_two_sided_zero_angular_continuation():
    cases = []
    masses, coefficients = _resonant_equilateral_shape_coefficients(
        order=10,
        cubic_amplitude=0.12,
        quartic_scale=0.04,
    )
    cases.append(("equilateral_beta_resonant", masses, coefficients))

    for resonance_power, ratio in ((5, 0.5), (6, 0.4), (7, 0.7)):
        masses, _horizontal_shape_mode, coefficients = _ordered_euler_resonant_shape_coefficients(
            resonance_power=resonance_power,
            ratio=ratio,
            order=10,
            resonance_amplitude=0.08,
            quartic_scale=0.03,
        )
        cases.append((f"ordered_euler_resonant_{resonance_power}", masses, coefficients))

    for _label, masses, shape_coefficients in cases:
        central_shape = shape_coefficients[0]
        initial_shape_gap = min(
            np.linalg.norm(central_shape[j] - central_shape[i])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        coefficients_by_power = {
            degree + 2: coefficient
            for degree, coefficient in enumerate(shape_coefficients)
        }
        paired_energies = {}

        assert initial_shape_gap > 0.1
        assert np.linalg.norm(
            accelerations(central_shape, masses) + (2.0 / 9.0) * central_shape,
            ord=np.inf,
        ) < 1e-12

        for regularized_time in (-0.05, 0.05, -0.025, 0.025):
            shape_value = sum(
                coefficient * regularized_time**degree
                for degree, coefficient in enumerate(shape_coefficients)
            )
            shape_gap = min(
                np.linalg.norm(shape_value[j] - shape_value[i])
                for i in range(3)
                for j in range(i + 1, 3)
            )
            positions, velocities = _shape_series_state(
                shape_coefficients,
                regularized_time,
            )
            state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
            projected_acceleration = _cubic_time_polynomial_acceleration(
                coefficients_by_power,
                regularized_time,
            )
            acceleration_residual = np.linalg.norm(
                projected_acceleration - accelerations(positions, masses),
                ord=np.inf,
            )
            acceleration_scale = max(
                1.0,
                np.linalg.norm(projected_acceleration, ord=np.inf),
            )

            assert shape_gap > 0.5 * initial_shape_gap
            assert acceleration_residual / acceleration_scale < 1e-13
            assert np.linalg.norm(center_of_mass(state, masses), ord=np.inf) < 1e-13
            assert np.linalg.norm(linear_momentum(state, masses), ord=np.inf) < 1e-12
            assert abs(angular_momentum_z(state, masses)) < 1e-12
            paired_energies[regularized_time] = energy(state, masses)

        assert abs(paired_energies[-0.05] - paired_energies[0.05]) < 1e-8
        assert abs(paired_energies[-0.025] - paired_energies[0.025]) < 1e-8


def test_spatial_embedding_preserves_zero_angular_total_collision_branch():
    masses, shape_coefficients = _resonant_equilateral_shape_coefficients(
        order=10,
        cubic_amplitude=0.10,
        quartic_scale=0.035,
    )
    first_frame_vector = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0])
    second_frame_vector = np.array(
        [-1.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0), 2.0 / np.sqrt(6.0)]
    )
    embedding = np.column_stack([first_frame_vector, second_frame_vector])
    coefficients_by_power = {
        degree + 2: coefficient
        for degree, coefficient in enumerate(shape_coefficients)
    }

    np.testing.assert_allclose(embedding.T @ embedding, np.eye(2), atol=1e-15)

    for regularized_time in (-0.04, 0.04):
        positions_2d, velocities_2d = _shape_series_state(
            shape_coefficients,
            regularized_time,
        )
        projected_acceleration_2d = _cubic_time_polynomial_acceleration(
            coefficients_by_power,
            regularized_time,
        )
        positions_3d = positions_2d @ embedding.T
        velocities_3d = velocities_2d @ embedding.T
        projected_acceleration_3d = projected_acceleration_2d @ embedding.T
        state_2d = np.concatenate([positions_2d.reshape(-1), velocities_2d.reshape(-1)])
        state_3d = np.concatenate([positions_3d.reshape(-1), velocities_3d.reshape(-1)])

        for first in range(3):
            for second in range(first + 1, 3):
                assert np.linalg.norm(positions_3d[second] - positions_3d[first]) == (
                    pytest.approx(
                        np.linalg.norm(positions_2d[second] - positions_2d[first])
                    )
                )

        acceleration_residual = projected_acceleration_3d - accelerations(
            positions_3d,
            masses,
        )
        acceleration_scale = max(
            1.0,
            np.linalg.norm(projected_acceleration_3d, ord=np.inf),
        )

        assert np.linalg.norm(positions_3d[:, 2], ord=np.inf) > 1.0e-5
        assert np.linalg.norm(acceleration_residual, ord=np.inf) / acceleration_scale < 1e-13
        assert np.linalg.norm(center_of_mass(state_3d, masses), ord=np.inf) < 1e-13
        assert np.linalg.norm(linear_momentum(state_3d, masses), ord=np.inf) < 1e-12
        assert np.linalg.norm(
            centered_angular_momentum_components(positions_3d, velocities_3d, masses),
            ord=np.inf,
        ) < 1e-12
        assert energy(state_3d, masses) == pytest.approx(energy(state_2d, masses))


def test_incoming_regularized_germ_selects_same_outgoing_branch_parameters():
    cubic_amplitude = 0.14
    quartic_scale = -0.025
    masses, incoming_coefficients = _resonant_equilateral_shape_coefficients(
        order=10,
        cubic_amplitude=cubic_amplitude,
        quartic_scale=quartic_scale,
    )
    central_shape = incoming_coefficients[0]
    cubic_kernel = incoming_coefficients[1] / np.sqrt(
        _mass_inner_product(masses, incoming_coefficients[1], incoming_coefficients[1])
    )
    recovered_cubic_amplitude = _mass_inner_product(
        masses,
        incoming_coefficients[1],
        cubic_kernel,
    )
    _base_masses, base_coefficients = _resonant_equilateral_shape_coefficients(
        order=10,
        cubic_amplitude=recovered_cubic_amplitude,
        quartic_scale=0.0,
    )
    recovered_quartic_scale = _mass_inner_product(
        masses,
        incoming_coefficients[2] - base_coefficients[2],
        central_shape,
    ) / _mass_inner_product(masses, central_shape, central_shape)
    _rebuilt_masses, rebuilt_coefficients = _resonant_equilateral_shape_coefficients(
        order=10,
        cubic_amplitude=recovered_cubic_amplitude,
        quartic_scale=recovered_quartic_scale,
    )

    assert recovered_cubic_amplitude == pytest.approx(cubic_amplitude)
    assert recovered_quartic_scale == pytest.approx(quartic_scale)
    np.testing.assert_allclose(rebuilt_coefficients, incoming_coefficients, atol=1e-12)

    for resonance_power, ratio, resonance_amplitude, quartic_scale in (
        (5, 0.5, 0.09, 0.03),
        (6, 0.4, -0.07, -0.02),
        (7, 0.7, 0.05, 0.04),
    ):
        masses, horizontal_shape_mode, incoming_coefficients = (
            _ordered_euler_resonant_shape_coefficients(
                resonance_power=resonance_power,
                ratio=ratio,
                order=10,
                resonance_amplitude=resonance_amplitude,
                quartic_scale=quartic_scale,
            )
        )
        central_shape = incoming_coefficients[0]
        recovered_quartic_scale = _mass_inner_product(
            masses,
            incoming_coefficients[2],
            central_shape,
        ) / _mass_inner_product(masses, central_shape, central_shape)
        _base_masses, _base_mode, base_coefficients = (
            _ordered_euler_resonant_shape_coefficients(
                resonance_power=resonance_power,
                ratio=ratio,
                order=10,
                resonance_amplitude=0.0,
                quartic_scale=recovered_quartic_scale,
            )
        )
        resonance_degree = resonance_power - 2
        recovered_resonance_amplitude = _mass_inner_product(
            masses,
            incoming_coefficients[resonance_degree] - base_coefficients[resonance_degree],
            horizontal_shape_mode,
        )
        _rebuilt_masses, _rebuilt_mode, rebuilt_coefficients = (
            _ordered_euler_resonant_shape_coefficients(
                resonance_power=resonance_power,
                ratio=ratio,
                order=10,
                resonance_amplitude=recovered_resonance_amplitude,
                quartic_scale=recovered_quartic_scale,
            )
        )

        assert recovered_quartic_scale == pytest.approx(quartic_scale)
        assert recovered_resonance_amplitude == pytest.approx(resonance_amplitude)
        np.testing.assert_allclose(rebuilt_coefficients, incoming_coefficients, atol=1e-11)

        coefficients_by_power = {
            degree + 2: coefficient
            for degree, coefficient in enumerate(rebuilt_coefficients)
        }
        for regularized_time in (-0.04, 0.04):
            positions, velocities = _shape_series_state(
                rebuilt_coefficients,
                regularized_time,
            )
            state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
            residual = _cubic_time_polynomial_acceleration(
                coefficients_by_power,
                regularized_time,
            ) - accelerations(positions, masses)

            assert np.linalg.norm(residual, ord=np.inf) < 1e-8
            assert abs(angular_momentum_z(state, masses)) < 1e-12


def test_finite_one_sided_jets_select_local_zero_angular_branch_parameters():
    cubic_amplitude = -0.11
    quartic_scale = 0.035
    masses, equilateral_coefficients = _resonant_equilateral_shape_coefficients(
        order=10,
        cubic_amplitude=cubic_amplitude,
        quartic_scale=quartic_scale,
    )
    central_shape = equilateral_coefficients[0]
    cubic_kernel = _equilateral_centered_cubic_kernel(masses, central_shape)
    _base_masses, equilateral_base_coefficients = _resonant_equilateral_shape_coefficients(
        order=10,
        cubic_amplitude=cubic_amplitude,
        quartic_scale=0.0,
    )
    fractional_remainder = np.array(
        [
            [0.04, -0.01],
            [-0.02, 0.03],
            [0.01, -0.02],
        ]
    )
    fractional_remainder -= np.average(fractional_remainder, axis=0, weights=masses)
    fractional_power = 4.6

    cubic_errors = []
    quartic_errors = []
    for tau in (-1.0e-2, -3.0e-3, -1.0e-3):
        incoming_positions = sum(
            coefficient * tau ** (degree + 2)
            for degree, coefficient in enumerate(equilateral_coefficients)
        ) + fractional_remainder * (-tau) ** fractional_power
        cubic_estimate = (incoming_positions - central_shape * tau**2) / tau**3
        quartic_estimate = (
            incoming_positions
            - central_shape * tau**2
            - equilateral_coefficients[1] * tau**3
        ) / tau**4
        recovered_cubic_amplitude = _mass_inner_product(
            masses,
            cubic_estimate,
            cubic_kernel,
        )
        recovered_quartic_scale = _mass_inner_product(
            masses,
            quartic_estimate - equilateral_base_coefficients[2],
            central_shape,
        ) / _mass_inner_product(masses, central_shape, central_shape)

        cubic_errors.append(abs(recovered_cubic_amplitude - cubic_amplitude))
        quartic_errors.append(abs(recovered_quartic_scale - quartic_scale))

    assert cubic_errors[0] > cubic_errors[1] > cubic_errors[2]
    assert quartic_errors[0] > quartic_errors[1] > quartic_errors[2]
    assert cubic_errors[-1] < 8e-4
    assert quartic_errors[-1] < 0.02

    resonance_power = 6
    resonance_amplitude = 0.075
    euler_quartic_scale = -0.025
    ratio = 0.4
    (
        euler_masses,
        horizontal_shape_mode,
        euler_coefficients,
    ) = _ordered_euler_resonant_shape_coefficients(
        resonance_power=resonance_power,
        ratio=ratio,
        order=10,
        resonance_amplitude=resonance_amplitude,
        quartic_scale=euler_quartic_scale,
    )
    (
        _base_euler_masses,
        _base_horizontal_shape_mode,
        euler_base_coefficients,
    ) = _ordered_euler_resonant_shape_coefficients(
        resonance_power=resonance_power,
        ratio=ratio,
        order=10,
        resonance_amplitude=0.0,
        quartic_scale=euler_quartic_scale,
    )
    central_shape = euler_coefficients[0]
    resonance_degree = resonance_power - 2
    euler_remainder = np.column_stack(
        [
            np.array([0.03, -0.04, 0.02]),
            np.array([0.02, -0.01, 0.03]),
        ]
    )
    euler_remainder -= np.average(euler_remainder, axis=0, weights=euler_masses)
    fractional_power = resonance_power + 0.55

    quartic_errors = []
    resonance_errors = []
    for tau in (-1.0e-2, -3.0e-3, -1.0e-3):
        incoming_positions = sum(
            coefficient * tau ** (degree + 2)
            for degree, coefficient in enumerate(euler_coefficients)
        ) + euler_remainder * (-tau) ** fractional_power
        quartic_estimate = (
            incoming_positions
            - euler_coefficients[0] * tau**2
            - euler_coefficients[1] * tau**3
        ) / tau**4
        lower_resonance_sum = sum(
            euler_coefficients[degree] * tau ** (degree + 2)
            for degree in range(resonance_degree)
        )
        resonance_estimate = (incoming_positions - lower_resonance_sum) / tau**resonance_power
        recovered_quartic_scale = _mass_inner_product(
            euler_masses,
            quartic_estimate,
            central_shape,
        ) / _mass_inner_product(euler_masses, central_shape, central_shape)
        recovered_resonance_amplitude = _mass_inner_product(
            euler_masses,
            resonance_estimate - euler_base_coefficients[resonance_degree],
            horizontal_shape_mode,
        )

        quartic_errors.append(abs(recovered_quartic_scale - euler_quartic_scale))
        resonance_errors.append(abs(recovered_resonance_amplitude - resonance_amplitude))

    assert quartic_errors[0] > quartic_errors[1] > quartic_errors[2]
    assert resonance_errors[0] > resonance_errors[1] > resonance_errors[2]
    assert quartic_errors[-1] < 3e-5
    assert resonance_errors[-1] < 2e-3

    _selected_masses, _selected_mode, selected_coefficients = (
        _ordered_euler_resonant_shape_coefficients(
            resonance_power=resonance_power,
            ratio=ratio,
            order=10,
            resonance_amplitude=resonance_amplitude,
            quartic_scale=euler_quartic_scale,
        )
    )
    coefficients_by_power = {
        degree + 2: coefficient for degree, coefficient in enumerate(selected_coefficients)
    }
    for tau in (-0.04, 0.04):
        positions, velocities = _shape_series_state(selected_coefficients, tau)
        state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
        residual = _cubic_time_polynomial_acceleration(
            coefficients_by_power,
            tau,
        ) - accelerations(positions, euler_masses)

        assert np.linalg.norm(residual, ord=np.inf) < 1e-8
        assert abs(angular_momentum_z(state, euler_masses)) < 1e-12


def test_zero_angular_finite_jet_entry_constructor_certifies_identity_branch():
    cubic_amplitude = -0.11
    quartic_scale = 0.035
    masses, incoming_coefficients = _resonant_equilateral_shape_coefficients(
        order=10,
        cubic_amplitude=cubic_amplitude,
        quartic_scale=quartic_scale,
    )
    central_shape = incoming_coefficients[0]
    cubic_kernel = _equilateral_centered_cubic_kernel(masses, central_shape)
    cubic_spec = FiniteJetSelectorSpec(
        name="equilateral_cubic_selector",
        degree=1,
        basis=cubic_kernel,
    )
    recovered_cubic = recover_finite_jet_selector_coordinates(
        masses=masses,
        coefficients=incoming_coefficients,
        selector_specs=(cubic_spec,),
    )["equilateral_cubic_selector"]
    _base_masses, equilateral_base_coefficients = _resonant_equilateral_shape_coefficients(
        order=10,
        cubic_amplitude=recovered_cubic,
        quartic_scale=0.0,
    )
    scale_spec = FiniteJetSelectorSpec(
        name="equilateral_energy_scale",
        degree=2,
        basis=central_shape,
        reference_coefficients=equilateral_base_coefficients,
    )
    recovered = recover_finite_jet_selector_coordinates(
        masses=masses,
        coefficients=incoming_coefficients,
        selector_specs=(cubic_spec, scale_spec),
    )
    _selected_masses, selected_coefficients = _resonant_equilateral_shape_coefficients(
        order=10,
        cubic_amplitude=recovered["equilateral_cubic_selector"],
        quartic_scale=recovered["equilateral_energy_scale"],
    )
    derived_equilateral_entry = derive_finite_jet_identity_selector_entry_from_incoming(
        masses=masses,
        incoming_coefficients=incoming_coefficients,
        selector_specs=(cubic_spec, scale_spec),
        tolerance=1.0e-7,
    )

    equilateral_entry = certify_finite_jet_identity_selector_entry(
        masses=masses,
        incoming_coefficients=incoming_coefficients,
        selected_coefficients=selected_coefficients,
        selector_specs=(cubic_spec, scale_spec),
        tolerance=1.0e-7,
    )

    assert equilateral_entry.certified
    assert derived_equilateral_entry.certified
    assert derived_equilateral_entry.identity_selector_certified
    assert equilateral_entry.required_regularized_jet_order == 4
    assert derived_equilateral_entry.entry_certificate.required_regularized_jet_order == 4
    assert recovered["equilateral_cubic_selector"] == pytest.approx(cubic_amplitude)
    assert recovered["equilateral_energy_scale"] == pytest.approx(quartic_scale)
    assert dict(derived_equilateral_entry.recovered_selector_values)[
        "equilateral_cubic_selector"
    ] == pytest.approx(cubic_amplitude)
    assert dict(derived_equilateral_entry.recovered_selector_values)[
        "equilateral_energy_scale"
    ] == pytest.approx(quartic_scale)
    assert np.max(
        np.abs(derived_equilateral_entry.selected_branch.coefficients - selected_coefficients)
    ) < 1.0e-10
    assert (
        derived_equilateral_entry.selected_branch.max_selector_operator_residual
        < 1.0e-10
    )
    assert (
        derived_equilateral_entry.selected_branch.min_selector_gram_singular_value_floor
        > 1.0e-3
    )
    assert derived_equilateral_entry.selected_branch.max_coefficient_residual < 1.0e-10
    assert equilateral_entry.max_newton_residual < 1.0e-8
    assert equilateral_entry.max_angular_momentum < 1.0e-12

    resonance_power = 6
    resonance_amplitude = 0.075
    euler_scale = -0.025
    ratio = 0.4
    euler_masses, horizontal_shape_mode, incoming_euler = (
        _ordered_euler_resonant_shape_coefficients(
            resonance_power=resonance_power,
            ratio=ratio,
            order=10,
            resonance_amplitude=resonance_amplitude,
            quartic_scale=euler_scale,
        )
    )
    euler_central_shape = incoming_euler[0]
    euler_scale_spec = FiniteJetSelectorSpec(
        name="ordered_euler_energy_scale",
        degree=2,
        basis=euler_central_shape,
    )
    recovered_scale = recover_finite_jet_selector_coordinates(
        masses=euler_masses,
        coefficients=incoming_euler,
        selector_specs=(euler_scale_spec,),
    )["ordered_euler_energy_scale"]
    (
        _base_euler_masses,
        _base_horizontal_shape_mode,
        euler_base_coefficients,
    ) = _ordered_euler_resonant_shape_coefficients(
        resonance_power=resonance_power,
        ratio=ratio,
        order=10,
        resonance_amplitude=0.0,
        quartic_scale=recovered_scale,
    )
    euler_resonance_spec = FiniteJetSelectorSpec(
        name="ordered_euler_resonance_selector",
        degree=resonance_power - 2,
        basis=horizontal_shape_mode,
        reference_coefficients=euler_base_coefficients,
    )
    recovered_euler = recover_finite_jet_selector_coordinates(
        masses=euler_masses,
        coefficients=incoming_euler,
        selector_specs=(euler_scale_spec, euler_resonance_spec),
    )
    (
        _selected_euler_masses,
        _selected_horizontal_shape_mode,
        selected_euler,
    ) = _ordered_euler_resonant_shape_coefficients(
        resonance_power=resonance_power,
        ratio=ratio,
        order=10,
        resonance_amplitude=recovered_euler["ordered_euler_resonance_selector"],
        quartic_scale=recovered_euler["ordered_euler_energy_scale"],
    )
    derived_euler_entry = derive_finite_jet_identity_selector_entry_from_incoming(
        masses=euler_masses,
        incoming_coefficients=incoming_euler,
        selector_specs=(euler_scale_spec, euler_resonance_spec),
        tolerance=1.0e-7,
    )

    euler_entry = certify_finite_jet_identity_selector_entry(
        masses=euler_masses,
        incoming_coefficients=incoming_euler,
        selected_coefficients=selected_euler,
        selector_specs=(euler_scale_spec, euler_resonance_spec),
        tolerance=1.0e-7,
    )

    assert euler_entry.certified
    assert derived_euler_entry.certified
    assert derived_euler_entry.identity_selector_certified
    assert euler_entry.required_regularized_jet_order == resonance_power
    assert (
        derived_euler_entry.entry_certificate.required_regularized_jet_order
        == resonance_power
    )
    assert recovered_euler["ordered_euler_energy_scale"] == pytest.approx(euler_scale)
    assert recovered_euler["ordered_euler_resonance_selector"] == pytest.approx(
        resonance_amplitude
    )
    assert dict(derived_euler_entry.recovered_selector_values)[
        "ordered_euler_energy_scale"
    ] == pytest.approx(euler_scale)
    assert dict(derived_euler_entry.recovered_selector_values)[
        "ordered_euler_resonance_selector"
    ] == pytest.approx(resonance_amplitude)
    assert np.max(
        np.abs(derived_euler_entry.selected_branch.coefficients - selected_euler)
    ) < 1.0e-10
    assert derived_euler_entry.selected_branch.max_selector_operator_residual < 1.0e-10
    assert (
        derived_euler_entry.selected_branch.min_selector_gram_singular_value_floor
        > 1.0e-3
    )
    assert derived_euler_entry.selected_branch.max_coefficient_residual < 1.0e-10
    assert euler_entry.max_newton_residual < 1.0e-8
    assert euler_entry.max_angular_momentum < 1.0e-12

    bad_selector = FiniteJetSelectorSpec(
        name="nonresonant_cubic_direction",
        degree=1,
        basis=central_shape,
    )
    with pytest.raises(ValueError, match="finite-jet selected branch did not certify"):
        derive_finite_jet_identity_selector_entry_from_incoming(
            masses=masses,
            incoming_coefficients=incoming_coefficients,
            selector_specs=(bad_selector,),
            tolerance=1.0e-7,
        )


def test_finite_jet_identity_selector_total_collisions_compose_compact_atlas():
    collision_events = []

    equilateral_cubic = -0.09
    equilateral_scale = 0.028
    masses, incoming_coefficients = _resonant_equilateral_shape_coefficients(
        order=10,
        cubic_amplitude=equilateral_cubic,
        quartic_scale=equilateral_scale,
    )
    central_shape = incoming_coefficients[0]
    cubic_kernel = _equilateral_centered_cubic_kernel(masses, central_shape)
    recovered_cubic = _mass_inner_product(
        masses,
        incoming_coefficients[1],
        cubic_kernel,
    )
    _base_masses, base_coefficients = _resonant_equilateral_shape_coefficients(
        order=10,
        cubic_amplitude=recovered_cubic,
        quartic_scale=0.0,
    )
    recovered_equilateral_scale = _mass_inner_product(
        masses,
        incoming_coefficients[2] - base_coefficients[2],
        central_shape,
    ) / _mass_inner_product(masses, central_shape, central_shape)
    _selected_masses, selected_coefficients = _resonant_equilateral_shape_coefficients(
        order=10,
        cubic_amplitude=recovered_cubic,
        quartic_scale=recovered_equilateral_scale,
    )
    collision_events.append(
        {
            "time": 0.35,
            "kind": "finite_jet_identity_selector_total_collision",
            "label": "equilateral_beta_resonant",
            "masses": masses,
            "coefficients": selected_coefficients,
            "incoming_energy": _shape_branch_energy_limit(masses, incoming_coefficients),
            "selector": (recovered_cubic, recovered_equilateral_scale),
            "required_regularized_jet_order": 4,
            "value": 3.0e-8,
            "first_jet": 8.0e-8,
            "lifted_residual": 2.0e-7,
            "physical_residual": 2.0e-12,
        }
    )

    resonance_power = 5
    euler_resonance = 0.5
    euler_amplitude = 0.07
    euler_scale = -0.018
    (
        euler_masses,
        horizontal_shape_mode,
        incoming_coefficients,
    ) = _ordered_euler_resonant_shape_coefficients(
        resonance_power=resonance_power,
        ratio=euler_resonance,
        order=10,
        resonance_amplitude=euler_amplitude,
        quartic_scale=euler_scale,
    )
    central_shape = incoming_coefficients[0]
    recovered_euler_scale = _mass_inner_product(
        euler_masses,
        incoming_coefficients[2],
        central_shape,
    ) / _mass_inner_product(euler_masses, central_shape, central_shape)
    (
        _base_euler_masses,
        _base_horizontal_shape_mode,
        base_coefficients,
    ) = _ordered_euler_resonant_shape_coefficients(
        resonance_power=resonance_power,
        ratio=euler_resonance,
        order=10,
        resonance_amplitude=0.0,
        quartic_scale=recovered_euler_scale,
    )
    resonance_degree = resonance_power - 2
    recovered_amplitude = _mass_inner_product(
        euler_masses,
        incoming_coefficients[resonance_degree] - base_coefficients[resonance_degree],
        horizontal_shape_mode,
    )
    (
        _selected_euler_masses,
        _selected_horizontal_shape_mode,
        selected_coefficients,
    ) = _ordered_euler_resonant_shape_coefficients(
        resonance_power=resonance_power,
        ratio=euler_resonance,
        order=10,
        resonance_amplitude=recovered_amplitude,
        quartic_scale=recovered_euler_scale,
    )
    collision_events.append(
        {
            "time": 1.1,
            "kind": "finite_jet_identity_selector_total_collision",
            "label": "ordered_euler_resonant_5",
            "masses": euler_masses,
            "coefficients": selected_coefficients,
            "incoming_energy": _shape_branch_energy_limit(
                euler_masses,
                incoming_coefficients,
            ),
            "selector": (recovered_amplitude, recovered_euler_scale),
            "required_regularized_jet_order": resonance_power,
            "value": 3.5e-8,
            "first_jet": 9.0e-8,
            "lifted_residual": 2.5e-7,
            "physical_residual": 2.5e-12,
        }
    )

    chart_list = (
        {
            "kind": "ordinary_gap_taylor",
            "value": 1.0e-8,
            "first_jet": 3.0e-8,
            "lifted_residual": 7.0e-8,
            "physical_residual": 7.0e-13,
        },
        collision_events[0],
        {
            "kind": "ordinary_gap_taylor",
            "value": 1.2e-8,
            "first_jet": 3.2e-8,
            "lifted_residual": 7.5e-8,
            "physical_residual": 7.5e-13,
        },
        {
            "kind": "separated_binary_levi_civita",
            "value": 1.5e-8,
            "first_jet": 4.0e-8,
            "lifted_residual": 9.0e-8,
            "physical_residual": 9.0e-13,
        },
        {
            "kind": "ordinary_gap_taylor",
            "value": 1.1e-8,
            "first_jet": 3.1e-8,
            "lifted_residual": 7.2e-8,
            "physical_residual": 7.2e-13,
        },
        collision_events[1],
        {
            "kind": "ordinary_gap_taylor",
            "value": 1.0e-8,
            "first_jet": 3.0e-8,
            "lifted_residual": 7.0e-8,
            "physical_residual": 7.0e-13,
        },
    )
    chart_kinds = {chart["kind"] for chart in chart_list}
    budget_components = ("value", "first_jet", "lifted_residual", "physical_residual")
    atlas_budget = {
        component: sum(chart[component] for chart in chart_list)
        for component in budget_components
    }

    assert recovered_cubic == pytest.approx(equilateral_cubic)
    assert recovered_equilateral_scale == pytest.approx(equilateral_scale)
    assert recovered_euler_scale == pytest.approx(euler_scale)
    assert recovered_amplitude == pytest.approx(euler_amplitude)
    assert {
        "ordinary_gap_taylor",
        "separated_binary_levi_civita",
        "finite_jet_identity_selector_total_collision",
    } <= chart_kinds
    assert "unresolved_zero_angular_total_collision" not in chart_kinds
    assert [event["time"] for event in collision_events] == sorted(
        event["time"] for event in collision_events
    )
    assert all(np.isfinite(atlas_budget[component]) for component in budget_components)
    assert atlas_budget["physical_residual"] < atlas_budget["lifted_residual"]

    for event in collision_events:
        masses = event["masses"]
        coefficients = event["coefficients"]
        central_shape = coefficients[0]
        central_gap = min(
            np.linalg.norm(central_shape[j] - central_shape[i])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        coefficients_by_power = {
            degree + 2: coefficient for degree, coefficient in enumerate(coefficients)
        }
        paired_energies = {}

        assert event["required_regularized_jet_order"] in (4, 5)
        assert central_gap > 0.1
        assert np.linalg.norm(
            accelerations(central_shape, masses) + (2.0 / 9.0) * central_shape,
            ord=np.inf,
        ) < 1e-12

        for regularized_time in (-0.04, 0.04):
            positions, velocities = _shape_series_state(
                coefficients,
                regularized_time,
            )
            state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
            projected_acceleration = _cubic_time_polynomial_acceleration(
                coefficients_by_power,
                regularized_time,
            )
            residual = projected_acceleration - accelerations(positions, masses)
            residual_scale = max(
                1.0,
                np.linalg.norm(projected_acceleration, ord=np.inf),
            )
            paired_energies[regularized_time] = energy(state, masses)

            assert np.linalg.norm(residual, ord=np.inf) / residual_scale < 1e-10
            assert abs(angular_momentum_z(state, masses)) < 1e-12
            assert np.linalg.norm(center_of_mass(state, masses), ord=np.inf) < 1e-13
            assert np.linalg.norm(linear_momentum(state, masses), ord=np.inf) < 1e-12

        assert paired_energies[-0.04] == pytest.approx(
            paired_energies[0.04],
            abs=1e-8,
        )
        assert paired_energies[0.04] == pytest.approx(
            event["incoming_energy"],
            abs=1e-5,
        )


def test_automatic_three_body_selector_compact_atlas_needs_no_extra_local_limits():
    events = []
    masses, coefficients = _resonant_equilateral_shape_coefficients(
        order=10,
        cubic_amplitude=0.08,
        quartic_scale=0.025,
    )
    events.append(
        {
            "time": 0.3,
            "kind": "automatic_identity_selector_total_collision",
            "selector_source": "three_body_entry_theorem",
            "external_selector_limit_required": False,
            "masses": masses,
            "coefficients": coefficients,
            "selector_count": 2,
            "value": 3.2e-8,
            "first_jet": 8.5e-8,
            "lifted_residual": 2.2e-7,
            "physical_residual": 2.2e-12,
        }
    )

    (
        euler_masses,
        _horizontal_shape_mode,
        euler_coefficients,
    ) = _ordered_euler_resonant_shape_coefficients(
        resonance_power=6,
        ratio=0.4,
        order=10,
        resonance_amplitude=-0.06,
        quartic_scale=0.02,
    )
    events.append(
        {
            "time": 1.25,
            "kind": "automatic_identity_selector_total_collision",
            "selector_source": "three_body_entry_theorem",
            "external_selector_limit_required": False,
            "masses": euler_masses,
            "coefficients": euler_coefficients,
            "selector_count": 2,
            "value": 3.8e-8,
            "first_jet": 9.5e-8,
            "lifted_residual": 2.6e-7,
            "physical_residual": 2.6e-12,
        }
    )

    chart_list = (
        {
            "kind": "ordinary_gap_taylor",
            "value": 1.1e-8,
            "first_jet": 3.1e-8,
            "lifted_residual": 7.1e-8,
            "physical_residual": 7.1e-13,
        },
        events[0],
        {
            "kind": "ordinary_gap_taylor",
            "value": 1.4e-8,
            "first_jet": 3.5e-8,
            "lifted_residual": 7.9e-8,
            "physical_residual": 7.9e-13,
        },
        {
            "kind": "separated_binary_levi_civita",
            "value": 1.6e-8,
            "first_jet": 4.1e-8,
            "lifted_residual": 9.2e-8,
            "physical_residual": 9.2e-13,
        },
        {
            "kind": "ordinary_gap_taylor",
            "value": 1.2e-8,
            "first_jet": 3.2e-8,
            "lifted_residual": 7.3e-8,
            "physical_residual": 7.3e-13,
        },
        events[1],
        {
            "kind": "ordinary_gap_taylor",
            "value": 1.0e-8,
            "first_jet": 3.0e-8,
            "lifted_residual": 7.0e-8,
            "physical_residual": 7.0e-13,
        },
    )
    components = ("value", "first_jet", "lifted_residual", "physical_residual")
    atlas_budget = {
        component: sum(chart[component] for chart in chart_list)
        for component in components
    }
    chart_kinds = {chart["kind"] for chart in chart_list}

    assert chart_kinds == {
        "ordinary_gap_taylor",
        "separated_binary_levi_civita",
        "automatic_identity_selector_total_collision",
    }
    assert [event["time"] for event in events] == sorted(event["time"] for event in events)
    assert all(not event["external_selector_limit_required"] for event in events)
    assert all(event["selector_source"] == "three_body_entry_theorem" for event in events)
    assert all(np.isfinite(atlas_budget[component]) for component in components)
    assert atlas_budget["physical_residual"] < atlas_budget["lifted_residual"]

    for event in events:
        masses = event["masses"]
        coefficients = event["coefficients"]
        central_shape = coefficients[0]
        derivative_matrix = _linearized_acceleration_matrix(central_shape, masses)
        eigenvalues = np.linalg.eigvals(derivative_matrix)
        real_eigenvalues = eigenvalues.real
        center_indicial_roots = [
            root
            for eigenvalue in real_eigenvalues
            for root in _indicial_roots_for_linearized_eigenvalue(eigenvalue)
            if abs(root.real) < 1.0e-8 and abs(root.imag) < 1.0e-8
        ]
        central_gap = min(
            np.linalg.norm(central_shape[j] - central_shape[i])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        coefficients_by_power = {
            degree + 2: coefficient for degree, coefficient in enumerate(coefficients)
        }
        paired_energies = {}

        assert event["selector_count"] < np.inf
        assert central_gap > 0.1
        assert np.linalg.norm(eigenvalues.imag, ord=np.inf) < 1e-10
        assert len(center_indicial_roots) == 1
        assert np.sum(np.abs(real_eigenvalues + 2.0 / 9.0) < 1e-9) == 1
        assert np.linalg.norm(
            accelerations(central_shape, masses) + (2.0 / 9.0) * central_shape,
            ord=np.inf,
        ) < 1e-12

        for regularized_time in (-0.035, 0.035):
            positions, velocities = _shape_series_state(
                coefficients,
                regularized_time,
            )
            state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
            projected_acceleration = _cubic_time_polynomial_acceleration(
                coefficients_by_power,
                regularized_time,
            )
            residual = projected_acceleration - accelerations(positions, masses)
            residual_scale = max(
                1.0,
                np.linalg.norm(projected_acceleration, ord=np.inf),
            )
            paired_energies[regularized_time] = energy(state, masses)

            assert np.linalg.norm(residual, ord=np.inf) / residual_scale < 1e-10
            assert abs(angular_momentum_z(state, masses)) < 1e-12
            assert np.linalg.norm(center_of_mass(state, masses), ord=np.inf) < 1e-13
            assert np.linalg.norm(linear_momentum(state, masses), ord=np.inf) < 1e-12

        assert paired_energies[-0.035] == pytest.approx(
            paired_energies[0.035],
            abs=1e-8,
        )


def test_local_normal_form_energy_limit_is_fixed_by_branch_jets():
    cases = []
    masses, coefficients = _resonant_equilateral_shape_coefficients(
        order=10,
        cubic_amplitude=0.12,
        quartic_scale=0.04,
    )
    cases.append(("equilateral_beta_resonant", masses, coefficients))

    for resonance_power, ratio in ((5, 0.5), (6, 0.4), (7, 0.7)):
        masses, _horizontal_shape_mode, coefficients = _ordered_euler_resonant_shape_coefficients(
            resonance_power=resonance_power,
            ratio=ratio,
            order=10,
            resonance_amplitude=0.08,
            quartic_scale=0.03,
        )
        cases.append((f"ordered_euler_resonant_{resonance_power}", masses, coefficients))

    for label, masses, shape_coefficients in cases:
        energy_limit = _shape_branch_energy_limit(masses, shape_coefficients)
        central_shape = shape_coefficients[0]
        cubic_shape = shape_coefficients[1]
        linearized_cubic = _acceleration_derivative_apply(
            central_shape,
            masses,
            cubic_shape,
        )
        residual_energies = []

        assert abs(_mass_inner_product(masses, central_shape, cubic_shape)) < 1e-12
        assert np.linalg.norm(linearized_cubic, ord=np.inf) < 1e-12
        assert np.isfinite(energy_limit)

        if label.startswith("ordered_euler"):
            expected_energy_limit = (10.0 / 9.0) * _mass_inner_product(
                masses,
                central_shape,
                shape_coefficients[2],
            )
            assert energy_limit == pytest.approx(expected_energy_limit)

        for regularized_time in (-0.05, 0.05, -0.025, 0.025):
            positions, velocities = _shape_series_state(
                shape_coefficients,
                regularized_time,
            )
            state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
            residual_energies.append(energy(state, masses) - energy_limit)

        assert max(abs(value) for value in residual_energies) < 5e-10


def test_local_normal_form_energy_parameter_matches_arbitrary_incoming_energy():
    branch_builders = [
        (
            "equilateral_beta_resonant",
            lambda quartic_scale: _resonant_equilateral_shape_coefficients(
                order=10,
                cubic_amplitude=0.12,
                quartic_scale=quartic_scale,
            ),
        )
    ]
    for resonance_power, ratio in ((5, 0.5), (6, 0.4), (7, 0.7)):
        branch_builders.append(
            (
                f"ordered_euler_resonant_{resonance_power}",
                lambda quartic_scale, resonance_power=resonance_power, ratio=ratio: (
                    lambda data: (data[0], data[2])
                )(
                    _ordered_euler_resonant_shape_coefficients(
                        resonance_power=resonance_power,
                        ratio=ratio,
                        order=10,
                        resonance_amplitude=0.08,
                        quartic_scale=quartic_scale,
                    )
                ),
            )
        )

    for _label, build_branch in branch_builders:
        masses, base_coefficients = build_branch(0.0)
        base_energy = _shape_branch_energy_limit(masses, base_coefficients)
        central_shape = base_coefficients[0]
        inertia = _mass_inner_product(masses, central_shape, central_shape)

        assert inertia > 0.0

        for target_energy in (base_energy - 0.25, base_energy + 0.4):
            quartic_scale = _energy_matching_quartic_scale(
                target_energy,
                masses,
                base_coefficients,
            )
            matched_masses, matched_coefficients = build_branch(quartic_scale)
            matched_limit = _shape_branch_energy_limit(
                matched_masses,
                matched_coefficients,
            )

            assert np.array_equal(masses, matched_masses)
            assert quartic_scale == pytest.approx(
                (9.0 / (10.0 * inertia)) * (target_energy - base_energy)
            )
            assert matched_limit == pytest.approx(target_energy)

            for regularized_time in (-0.04, 0.04, 0.025):
                positions, velocities = _shape_series_state(
                    matched_coefficients,
                    regularized_time,
                )
                state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])

                assert abs(energy(state, masses) - target_energy) < 1e-9
                assert abs(angular_momentum_z(state, masses)) < 1e-12


def test_resonant_branch_amplitude_is_recovered_from_finite_regularized_jet():
    cubic_amplitude = -0.17
    masses, equilateral_coefficients = _resonant_equilateral_shape_coefficients(
        order=10,
        cubic_amplitude=cubic_amplitude,
        quartic_scale=0.03,
    )
    central_shape = equilateral_coefficients[0]
    cubic_kernel = _equilateral_centered_cubic_kernel(masses, central_shape)
    recovered_cubic_amplitude = _mass_inner_product(
        masses,
        equilateral_coefficients[1],
        cubic_kernel,
    )
    third_regularized_jet = 6.0 * equilateral_coefficients[1]

    assert np.linalg.norm(
        _acceleration_derivative_apply(central_shape, masses, cubic_kernel),
        ord=np.inf,
    ) < 1e-12
    assert _mass_inner_product(masses, cubic_kernel, cubic_kernel) == pytest.approx(1.0)
    assert abs(_mass_inner_product(masses, central_shape, cubic_kernel)) < 1e-12
    assert recovered_cubic_amplitude == pytest.approx(cubic_amplitude)
    assert _mass_inner_product(
        masses,
        third_regularized_jet,
        cubic_kernel,
    ) / 6.0 == pytest.approx(cubic_amplitude)

    for resonance_power, ratio, resonance_amplitude in (
        (5, 0.5, 0.09),
        (6, 0.4, -0.07),
        (7, 0.7, 0.05),
    ):
        masses, horizontal_shape_mode, coefficients = _ordered_euler_resonant_shape_coefficients(
            resonance_power=resonance_power,
            ratio=ratio,
            order=10,
            resonance_amplitude=resonance_amplitude,
            quartic_scale=0.02,
        )
        _base_masses, _base_mode, base_coefficients = _ordered_euler_resonant_shape_coefficients(
            resonance_power=resonance_power,
            ratio=ratio,
            order=10,
            resonance_amplitude=0.0,
            quartic_scale=0.02,
        )
        resonance_degree = resonance_power - 2
        recovered_resonance_amplitude = _mass_inner_product(
            masses,
            coefficients[resonance_degree] - base_coefficients[resonance_degree],
            horizontal_shape_mode,
        )
        regularized_jet_coefficient = coefficients[resonance_degree] - base_coefficients[
            resonance_degree
        ]

        assert np.array_equal(masses, _base_masses)
        assert _mass_inner_product(
            masses,
            horizontal_shape_mode,
            horizontal_shape_mode,
        ) == pytest.approx(1.0)
        assert np.linalg.norm(
            np.sum(masses[:, None] * horizontal_shape_mode, axis=0),
            ord=np.inf,
        ) < 1e-12
        assert recovered_resonance_amplitude == pytest.approx(resonance_amplitude)
        assert _mass_inner_product(
            masses,
            regularized_jet_coefficient,
            horizontal_shape_mode,
        ) == pytest.approx(resonance_amplitude)


@pytest.mark.parametrize("resonance_power", (5, 6, 7))
def test_symmetric_euler_resonant_sign_branches_share_second_jet_and_energy(
    resonance_power,
):
    positive_masses, positive_coefficients = _symmetric_euler_resonant_shape_coefficients(
        resonance_power=resonance_power,
        order=12,
        resonance_amplitude=0.05,
        quartic_scale=0.04,
    )
    negative_masses, negative_coefficients = _symmetric_euler_resonant_shape_coefficients(
        resonance_power=resonance_power,
        order=12,
        resonance_amplitude=-0.05,
        quartic_scale=0.04,
    )
    resonance_degree = resonance_power - 2
    middle_mass = positive_masses[1]
    horizontal_shape_mode = np.column_stack(
        [
            np.array([1.0, -2.0 / middle_mass, 1.0]),
            np.zeros(3),
        ]
    )
    horizontal_shape_mode = horizontal_shape_mode / np.sqrt(
        np.sum(positive_masses[:, None] * horizontal_shape_mode * horizontal_shape_mode)
    )
    energy_limit = (10.0 / 9.0) * float(
        np.sum(
            positive_masses[:, None]
            * positive_coefficients[0]
            * positive_coefficients[2]
        )
    )

    np.testing.assert_allclose(positive_masses, negative_masses)
    np.testing.assert_allclose(positive_coefficients[0], negative_coefficients[0])
    np.testing.assert_allclose(positive_coefficients[1], negative_coefficients[1])
    np.testing.assert_allclose(positive_coefficients[2], negative_coefficients[2])
    np.testing.assert_allclose(
        positive_coefficients[resonance_degree] - negative_coefficients[resonance_degree],
        0.1 * horizontal_shape_mode,
        atol=1e-14,
    )

    for coefficients in (positive_coefficients, negative_coefficients):
        regularized_time = 0.02
        positions = sum(
            coefficient * regularized_time ** (degree + 2)
            for degree, coefficient in enumerate(coefficients)
        )
        velocities = sum(
            ((degree + 2.0) / 3.0)
            * coefficient
            * regularized_time ** (degree - 1)
            for degree, coefficient in enumerate(coefficients)
        )
        state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])

        assert abs(energy(state, positive_masses) - energy_limit) < 1e-9
        assert abs(angular_momentum_z(state, positive_masses)) < 1e-24

    regularized_time = 0.04
    positive_positions = sum(
        coefficient * regularized_time ** (degree + 2)
        for degree, coefficient in enumerate(positive_coefficients)
    )
    negative_positions = sum(
        coefficient * regularized_time ** (degree + 2)
        for degree, coefficient in enumerate(negative_coefficients)
    )

    assert np.linalg.norm(positive_positions - negative_positions, ord=np.inf) > (
        1e-4 * regularized_time**resonance_power
    )


def test_resonant_equilateral_sign_branches_share_second_jet_and_energy():
    positive_masses, positive_coefficients = _resonant_equilateral_shape_coefficients(
        order=12,
        cubic_amplitude=0.05,
        quartic_scale=0.04,
    )
    negative_masses, negative_coefficients = _resonant_equilateral_shape_coefficients(
        order=12,
        cubic_amplitude=-0.05,
        quartic_scale=0.04,
    )

    np.testing.assert_allclose(positive_masses, negative_masses)
    np.testing.assert_allclose(positive_coefficients[0], negative_coefficients[0])
    np.testing.assert_allclose(positive_coefficients[1], -negative_coefficients[1])
    np.testing.assert_allclose(positive_coefficients[2], negative_coefficients[2])

    for degree in range(1, 10):
        expected_negative = ((-1.0) ** degree) * negative_coefficients[degree]
        np.testing.assert_allclose(
            positive_coefficients[degree],
            expected_negative,
            atol=1e-14,
        )

    positive_energies = []
    negative_energies = []
    for regularized_time in (0.04, 0.02):
        positive_positions = sum(
            coefficient * regularized_time ** (degree + 2)
            for degree, coefficient in enumerate(positive_coefficients)
        )
        negative_positions = sum(
            coefficient * regularized_time ** (degree + 2)
            for degree, coefficient in enumerate(negative_coefficients)
        )
        for coefficients, energies in (
            (positive_coefficients, positive_energies),
            (negative_coefficients, negative_energies),
        ):
            positions = sum(
                coefficient * regularized_time ** (degree + 2)
                for degree, coefficient in enumerate(coefficients)
            )
            velocities = sum(
                ((degree + 2.0) / 3.0)
                * coefficient
                * regularized_time ** (degree - 1)
                for degree, coefficient in enumerate(coefficients)
            )
            state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])
            energies.append(energy(state, positive_masses))
            assert abs(angular_momentum_z(state, positive_masses)) < 1e-16

        assert abs(positive_energies[-1] - negative_energies[-1]) < 1e-12
        assert np.linalg.norm(positive_positions - negative_positions, ord=np.inf) > (
            1e-2 * regularized_time**3
        )


@pytest.mark.parametrize("shape", ("equilateral", "euler"))
def test_regularized_second_jet_branch_gives_homothetic_continuation(shape):
    configuration, central_lambda = _central_configuration_data(shape)
    branch = construct_homothetic_total_collision_branch(
        configuration,
        central_lambda,
    )
    quadratic_coefficient = branch.quadratic_coefficient
    second_regularized_jet = branch.second_regularized_jet

    assert branch.certified_scaled_central_configuration
    assert np.linalg.norm(
        accelerations(quadratic_coefficient) + (2.0 / 9.0) * quadratic_coefficient,
        ord=np.inf,
    ) < 1e-12
    assert np.linalg.norm(
        accelerations(second_regularized_jet)
        + (1.0 / 36.0) * second_regularized_jet,
        ord=np.inf,
    ) < 1e-12
    assert np.linalg.norm(0.5 * second_regularized_jet - quadratic_coefficient, ord=np.inf) == 0.0

    for regularized_time in (-0.31, -0.07, 0.05, 0.23):
        positions = branch.positions_at_tau(regularized_time)
        velocities = branch.velocities_at_tau(regularized_time)
        expected_acceleration = -(2.0 / 9.0) * quadratic_coefficient * regularized_time**-4
        state = branch.state_at_tau(regularized_time)
        acceleration_residual = np.linalg.norm(
            accelerations(positions) - expected_acceleration,
            ord=np.inf,
        )
        branch_residual = np.linalg.norm(
            branch.newton_residual_at_tau(regularized_time),
            ord=np.inf,
        )
        acceleration_scale = max(1.0, np.linalg.norm(expected_acceleration, ord=np.inf))

        assert acceleration_residual / acceleration_scale < 1e-14
        assert branch_residual / acceleration_scale < 1e-14
        assert np.linalg.norm(center_of_mass(state), ord=np.inf) < 1e-14
        assert np.linalg.norm(linear_momentum(state), ord=np.inf) < 1e-14
        assert abs(angular_momentum_z(state)) < 1e-14
        assert abs(energy(state)) < 1e-12

    inbound_tau = -0.11
    outbound_tau = -inbound_tau
    inbound_positions = branch.positions_at_tau(inbound_tau)
    outbound_positions = branch.positions_at_tau(outbound_tau)
    inbound_velocities = branch.velocities_at_tau(inbound_tau)
    outbound_velocities = branch.velocities_at_tau(outbound_tau)

    assert np.linalg.norm(inbound_positions - outbound_positions, ord=np.inf) == 0.0
    assert np.linalg.norm(inbound_velocities + outbound_velocities, ord=np.inf) == 0.0


@pytest.mark.parametrize("energy_per_inertia", (-0.35, 0.25))
def test_energy_parameterized_homothetic_series_solves_regularized_energy_recurrence(
    energy_per_inertia,
):
    order = 10
    coefficients = _homothetic_energy_series_coefficients(energy_per_inertia, order)
    configuration, central_lambda = _central_configuration_data("equilateral")
    branch = construct_homothetic_total_collision_branch(
        configuration,
        central_lambda,
        energy_per_inertia=energy_per_inertia,
        order=order,
    )
    derivative_combination = np.array(
        [(degree + 1) * coefficients[degree] for degree in range(order + 1)]
    )
    left = _series_product(derivative_combination, derivative_combination, order)
    right = _series_inverse(coefficients, order)
    right[1] += (9.0 / 2.0) * energy_per_inertia
    first = (9.0 / 10.0) * energy_per_inertia

    assert abs(coefficients[1] - first) < 1e-15
    assert abs(coefficients[2] + (3.0 / 7.0) * first**2) < 1e-15
    assert abs(coefficients[3] - (23.0 / 63.0) * first**3) < 1e-15
    assert np.linalg.norm(left - right, ord=np.inf) < 1e-13
    assert np.linalg.norm(branch.coefficients - coefficients, ord=np.inf) == 0.0
    assert np.linalg.norm(branch.energy_recurrence_residual_coefficients(), ord=np.inf) < 1e-13


@pytest.mark.parametrize("shape", ("equilateral", "euler"))
@pytest.mark.parametrize("energy_per_inertia", (-0.35, 0.25))
def test_energy_parameterized_homothetic_branch_records_energy_in_fourth_jet(
    shape,
    energy_per_inertia,
):
    configuration, central_lambda = _central_configuration_data(shape)
    branch = construct_homothetic_total_collision_branch(
        configuration,
        central_lambda,
        energy_per_inertia=energy_per_inertia,
    )
    coefficients = branch.coefficients
    fourth_regularized_jet = branch.fourth_regularized_jet
    expected_fourth_jet = branch.expected_fourth_regularized_jet
    inertia = branch.inertia

    assert np.linalg.norm(fourth_regularized_jet - expected_fourth_jet, ord=np.inf) < 1e-14

    for regularized_time in (-0.05, 0.05):
        state = branch.state_at_tau(regularized_time)

        assert abs(energy(state) / inertia - energy_per_inertia) < 1e-9
        assert np.linalg.norm(center_of_mass(state), ord=np.inf) < 1e-14
        assert np.linalg.norm(linear_momentum(state), ord=np.inf) < 1e-14
        assert abs(angular_momentum_z(state)) < 1e-14


def test_arbitrary_mass_equilateral_homothetic_branch_continues_through_collision():
    masses, configuration, central_lambda = _arbitrary_mass_equilateral_central_configuration()
    branch = construct_homothetic_total_collision_branch(
        configuration,
        central_lambda,
        masses,
    )
    quadratic_coefficient = branch.quadratic_coefficient
    second_regularized_jet = branch.second_regularized_jet

    assert np.linalg.norm(np.sum(masses[:, None] * configuration, axis=0), ord=np.inf) < 1e-14
    assert np.linalg.norm(
        accelerations(configuration, masses) + central_lambda * configuration,
        ord=np.inf,
    ) < 1e-14
    assert np.linalg.norm(
        accelerations(quadratic_coefficient, masses) + (2.0 / 9.0) * quadratic_coefficient,
        ord=np.inf,
    ) < 1e-14
    assert np.linalg.norm(
        accelerations(second_regularized_jet, masses)
        + (1.0 / 36.0) * second_regularized_jet,
        ord=np.inf,
    ) < 1e-14

    for regularized_time in (-0.13, 0.17):
        positions = quadratic_coefficient * regularized_time**2
        velocities = (2.0 / 3.0) * quadratic_coefficient / regularized_time
        expected_acceleration = -(2.0 / 9.0) * quadratic_coefficient * regularized_time**-4
        state = branch.state_at_tau(regularized_time)
        acceleration_residual = np.linalg.norm(
            accelerations(positions, masses) - expected_acceleration,
            ord=np.inf,
        )
        acceleration_scale = max(1.0, np.linalg.norm(expected_acceleration, ord=np.inf))

        assert acceleration_residual / acceleration_scale < 1e-14
        assert np.linalg.norm(center_of_mass(state, masses), ord=np.inf) < 1e-14
        assert np.linalg.norm(linear_momentum(state, masses), ord=np.inf) < 1e-14
        assert abs(angular_momentum_z(state, masses)) < 1e-14
        assert abs(energy(state, masses)) < 1e-12


@pytest.mark.parametrize("energy_per_inertia", (-0.2, 0.3))
def test_arbitrary_mass_homothetic_energy_branch_uses_mass_weighted_inertia(
    energy_per_inertia,
):
    masses, configuration, central_lambda = _arbitrary_mass_equilateral_central_configuration()
    branch = construct_homothetic_total_collision_branch(
        configuration,
        central_lambda,
        masses,
        energy_per_inertia=energy_per_inertia,
    )
    inertia = branch.inertia

    assert inertia > 0.0

    for regularized_time in (-0.05, 0.05):
        state = branch.state_at_tau(regularized_time)

        assert abs(energy(state, masses) / inertia - energy_per_inertia) < 1e-9
        assert np.linalg.norm(center_of_mass(state, masses), ord=np.inf) < 1e-14
        assert np.linalg.norm(linear_momentum(state, masses), ord=np.inf) < 1e-14
        assert abs(angular_momentum_z(state, masses)) < 1e-14


def test_compact_sundman_time_has_finite_endpoint_on_hyperbolic_homothetic_escape():
    configuration, central_lambda = _central_configuration_data("equilateral")
    masses = np.ones(3)
    initial_scale = 1.0
    initial_radial_speed = 2.0
    positions = initial_scale * configuration
    velocities = initial_radial_speed * configuration
    state = np.concatenate([positions.reshape(-1), velocities.reshape(-1)])

    radial_energy = 0.5 * initial_radial_speed**2 - central_lambda / initial_scale
    asymptotic_radial_speed = np.sqrt(2.0 * radial_energy)
    pair_distances = [
        float(np.linalg.norm(configuration[j] - configuration[i]))
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    sundman_pair_product = float(np.prod(pair_distances))
    sundman_future_length_bound = 1.0 / (
        2.0 * sundman_pair_product * asymptotic_radial_speed * initial_scale**2
    )
    compact_sundman_rate = 1.2
    compact_sundman_future_endpoint_bound = np.tanh(
        compact_sundman_rate * sundman_future_length_bound
    )
    geometric_schedule_upper_after_four_shells = 1.0 - 0.994 * 0.98**4
    scale_bound = 10.0
    time_to_exceed_scale_bound = 1.01 * (
        scale_bound / np.max(np.linalg.norm(configuration, axis=1)) - initial_scale
    ) / asymptotic_radial_speed

    assert radial_energy > 0.0
    assert np.linalg.norm(
        accelerations(positions, masses) + central_lambda * configuration / initial_scale**2,
        ord=np.inf,
    ) < 1e-12
    assert np.linalg.norm(center_of_mass(state), ord=np.inf) < 1e-14
    assert np.linalg.norm(linear_momentum(state), ord=np.inf) < 1e-14
    assert abs(angular_momentum_z(state)) < 1e-14
    assert energy(state, masses) > 0.0
    assert sundman_future_length_bound < 0.06
    assert compact_sundman_future_endpoint_bound < 0.08
    assert geometric_schedule_upper_after_four_shells > compact_sundman_future_endpoint_bound
    assert time_to_exceed_scale_bound < np.inf


def test_unscaled_sundman_cauchy_radius_decays_on_hyperbolic_escape():
    configuration, central_lambda = _central_configuration_data("equilateral")
    masses = np.ones(3)
    radial_energy = 0.5 * 2.0**2 - central_lambda
    asymptotic_radial_speed = np.sqrt(2.0 * radial_energy)
    shape_min_pair_distance = min(
        float(np.linalg.norm(configuration[j] - configuration[i]))
        for i in range(3)
        for j in range(i + 1, 3)
    )
    shape_max_pair_distance = max(
        float(np.linalg.norm(configuration[j] - configuration[i]))
        for i in range(3)
        for j in range(i + 1, 3)
    )
    shape_max_radius = float(max(np.linalg.norm(point) for point in configuration))
    radius_decay_constant = (
        shape_min_pair_distance
        / (10.0 * shape_max_pair_distance**3 * asymptotic_radial_speed * shape_max_radius)
    )
    eventual_uniform_floor_from_compact_atanh_lemma = 2.0 / 3.0

    certified_radii = []
    for scale in (10.0, 100.0, 1000.0):
        radial_speed = np.sqrt(2.0 * radial_energy + 2.0 * central_lambda / scale)
        positions = scale * configuration
        velocities = radial_speed * configuration
        certificate = sundman_cauchy_majorant_tail_certificate(
            positions,
            velocities,
            masses,
            retained_order=6,
            step_size=0.0,
        )
        max_velocity = float(max(np.linalg.norm(velocity) for velocity in velocities))
        position_self_map_radius = certificate.position_radius / (
            certificate.sundman_factor_bound * max_velocity
        )

        assert certificate.s_radius <= position_self_map_radius
        assert certificate.s_radius * scale**2 <= radius_decay_constant
        assert certificate.s_radius < eventual_uniform_floor_from_compact_atanh_lemma
        certified_radii.append(certificate.s_radius)

    assert certified_radii[0] > certified_radii[1] > certified_radii[2]
    assert certified_radii[-1] < 2e-8


def test_physical_time_compactification_keeps_hyperbolic_escape_unbounded_at_endpoint():
    configuration, central_lambda = _central_configuration_data("equilateral")
    initial_scale = 1.0
    initial_radial_speed = 2.0
    radial_energy = 0.5 * initial_radial_speed**2 - central_lambda / initial_scale
    asymptotic_radial_speed = np.sqrt(2.0 * radial_energy)
    compact_time_rate = 0.4
    scale_bound = 10.0
    time_to_exceed_scale_bound = 1.01 * (
        scale_bound / np.max(np.linalg.norm(configuration, axis=1)) - initial_scale
    ) / asymptotic_radial_speed
    compact_time_parameter_at_scale_bound = np.tanh(
        compact_time_rate * time_to_exceed_scale_bound
    )
    logarithmic_growth_lower_bound = initial_scale + (
        asymptotic_radial_speed / (2.0 * compact_time_rate)
    ) * np.log(
        (1.0 + compact_time_parameter_at_scale_bound)
        / (1.0 - compact_time_parameter_at_scale_bound)
    )

    assert radial_energy > 0.0
    assert 0.0 < compact_time_parameter_at_scale_bound < 1.0
    assert compact_time_parameter_at_scale_bound > 0.95
    assert logarithmic_growth_lower_bound * np.max(
        np.linalg.norm(configuration, axis=1)
    ) >= scale_bound


def test_hyperbolic_escape_scaled_position_has_finite_compact_time_endpoint_bound():
    configuration, central_lambda = _central_configuration_data("equilateral")
    initial_scale = 1.0
    initial_radial_speed = 2.0
    radial_energy = 0.5 * initial_radial_speed**2 - central_lambda / initial_scale
    asymptotic_radial_speed = np.sqrt(2.0 * radial_energy)
    compact_time_rate = 0.4
    compact_time_parameter = 0.999999
    physical_time = np.arctanh(compact_time_parameter) / compact_time_rate
    unscaled_position_lower_bound = (
        initial_scale + asymptotic_radial_speed * physical_time
    ) * np.max(np.linalg.norm(configuration, axis=1))
    scaled_position_upper_bound = (
        (initial_scale + initial_radial_speed * physical_time)
        / (1.0 + physical_time)
        * np.max(np.linalg.norm(configuration, axis=1))
    )
    scaled_endpoint_limit_norm = asymptotic_radial_speed * np.max(
        np.linalg.norm(configuration, axis=1)
    )

    assert radial_energy > 0.0
    assert unscaled_position_lower_bound > 30.0
    assert scaled_position_upper_bound <= initial_radial_speed
    assert scaled_endpoint_limit_norm < scaled_position_upper_bound
    assert scaled_endpoint_limit_norm == pytest.approx(
        np.sqrt(2.0 * radial_energy) * np.max(np.linalg.norm(configuration, axis=1))
    )


def test_scaled_escape_variable_equation_projects_to_newtonian_acceleration():
    configuration, central_lambda = _central_configuration_data("equilateral")
    masses = np.ones(3)
    radial_energy = 0.5 * 2.0**2 - central_lambda
    physical_time = 4.0
    scale = 1.0 + physical_time
    radius = 3.0
    radius_velocity = np.sqrt(2.0 * radial_energy + 2.0 * central_lambda / radius)
    radius_acceleration = -central_lambda / radius**2
    positions = radius * configuration
    scaled_positions = positions / scale
    scaled_velocities = (radius_velocity * scale - radius) * configuration / scale**2
    scaled_accelerations = (
        radius_acceleration / scale
        - 2.0 * radius_velocity / scale**2
        + 2.0 * radius / scale**3
    ) * configuration

    projected_acceleration = scale * scaled_accelerations + 2.0 * scaled_velocities
    lifted_rhs = accelerations(scaled_positions, masses) / scale**2

    assert radial_energy > 0.0
    assert np.linalg.norm(projected_acceleration - accelerations(positions, masses), ord=np.inf) < 1e-12
    assert np.linalg.norm(projected_acceleration - lifted_rhs, ord=np.inf) < 1e-12


def test_compact_time_scaled_escape_equation_projects_to_newtonian_acceleration():
    configuration, central_lambda = _central_configuration_data("equilateral")
    masses = np.ones(3)
    compact_time_rate = 0.4
    compact_parameter = 0.7
    physical_time = np.arctanh(compact_parameter) / compact_time_rate
    time_derivative = 1.0 / (compact_time_rate * (1.0 - compact_parameter**2))
    time_second_derivative = (
        2.0
        * compact_parameter
        / (compact_time_rate * (1.0 - compact_parameter**2) ** 2)
    )
    scale = 1.0 + physical_time
    radial_energy = 0.5 * 2.0**2 - central_lambda
    radius = 3.0
    radius_velocity = np.sqrt(2.0 * radial_energy + 2.0 * central_lambda / radius)
    radius_acceleration = -central_lambda / radius**2
    positions = radius * configuration
    scaled_positions = positions / scale
    scaled_velocity_t = (radius_velocity * scale - radius) * configuration / scale**2
    scaled_acceleration_t = (
        radius_acceleration / scale
        - 2.0 * radius_velocity / scale**2
        + 2.0 * radius / scale**3
    ) * configuration
    scaled_velocity_u = scaled_velocity_t * time_derivative
    scaled_acceleration_u = (
        scaled_acceleration_t * time_derivative**2
        + scaled_velocity_t * time_second_derivative
    )

    compact_lhs = scale * scaled_acceleration_u + (
        2.0 * time_derivative
        - scale * time_second_derivative / time_derivative
    ) * scaled_velocity_u
    compact_rhs = (
        time_derivative**2
        * accelerations(scaled_positions, masses)
        / scale**2
    )

    assert radial_energy > 0.0
    assert np.linalg.norm(compact_lhs - compact_rhs, ord=np.inf) < 1e-12
    assert np.linalg.norm(compact_lhs / time_derivative**2 - accelerations(positions, masses), ord=np.inf) < 1e-12


def test_all_real_positive_scale_compact_equation_projects_to_newtonian_acceleration():
    configuration, central_lambda = _central_configuration_data("equilateral")
    masses = np.ones(3)
    compact_time_rate = 0.4
    compact_parameter = 0.7
    physical_time = np.arctanh(compact_parameter) / compact_time_rate
    time_derivative = 1.0 / (compact_time_rate * (1.0 - compact_parameter**2))
    time_second_derivative = (
        2.0
        * compact_parameter
        / (compact_time_rate * (1.0 - compact_parameter**2) ** 2)
    )
    scale = np.sqrt(1.0 + physical_time**2)
    scale_t = physical_time / scale
    scale_tt = 1.0 / scale**3
    scale_u = scale_t * time_derivative
    scale_uu = scale_tt * time_derivative**2 + scale_t * time_second_derivative
    radial_energy = 0.5 * 2.0**2 - central_lambda
    radius = 3.0
    radius_velocity = np.sqrt(2.0 * radial_energy + 2.0 * central_lambda / radius)
    radius_acceleration = -central_lambda / radius**2
    positions = radius * configuration
    scaled_positions = positions / scale
    scaled_velocity_t = (
        radius_velocity / scale
        - radius * scale_t / scale**2
    ) * configuration
    scaled_acceleration_t = (
        radius_acceleration / scale
        - 2.0 * radius_velocity * scale_t / scale**2
        - radius * scale_tt / scale**2
        + 2.0 * radius * scale_t**2 / scale**3
    ) * configuration
    scaled_velocity_u = scaled_velocity_t * time_derivative
    scaled_acceleration_u = (
        scaled_acceleration_t * time_derivative**2
        + scaled_velocity_t * time_second_derivative
    )

    compact_lhs = (
        scale * scaled_acceleration_u
        + (2.0 * scale_u - scale * time_second_derivative / time_derivative)
        * scaled_velocity_u
        + (scale_uu - scale_u * time_second_derivative / time_derivative)
        * scaled_positions
    )
    compact_rhs = (
        time_derivative**2
        * accelerations(scaled_positions, masses)
        / scale**2
    )

    assert radial_energy > 0.0
    assert scale > 0.0
    assert np.linalg.norm(compact_lhs - compact_rhs, ord=np.inf) < 1e-12
    assert np.linalg.norm(compact_lhs / time_derivative**2 - accelerations(positions, masses), ord=np.inf) < 1e-12


def test_hyperbolic_escape_inverse_time_endpoint_requires_log_term():
    configuration, central_lambda = _central_configuration_data("equilateral")
    radial_energy = 0.5 * 2.0**2 - central_lambda
    asymptotic_radial_speed = np.sqrt(2.0 * radial_energy)
    log_coefficient = central_lambda / asymptotic_radial_speed**2
    endpoint_configuration = asymptotic_radial_speed * configuration
    endpoint_acceleration = accelerations(endpoint_configuration, np.ones(3))
    tau = 1e-6
    log_model_second_tau_derivative = -log_coefficient * configuration / tau
    log_model_endpoint_lhs = tau * log_model_second_tau_derivative
    analytic_model_endpoint_lhs = np.zeros_like(configuration)

    assert radial_energy > 0.0
    assert np.linalg.norm(endpoint_acceleration, ord=np.inf) > 0.0
    assert np.linalg.norm(log_model_endpoint_lhs - endpoint_acceleration, ord=np.inf) < 1e-12
    assert np.linalg.norm(analytic_model_endpoint_lhs - endpoint_acceleration, ord=np.inf) > 0.1


def test_log_subtracted_escape_endpoint_cancels_leading_inverse_time_force():
    configuration, central_lambda = _central_configuration_data("equilateral")
    radial_energy = 0.5 * 2.0**2 - central_lambda
    asymptotic_radial_speed = np.sqrt(2.0 * radial_energy)
    log_coefficient = central_lambda / asymptotic_radial_speed**2
    endpoint_configuration = asymptotic_radial_speed * configuration
    tau = 1e-6
    log_subtracted_argument = (
        endpoint_configuration - log_coefficient * tau * np.log(tau) * configuration
    )
    transformed_endpoint_force = (
        accelerations(log_subtracted_argument, np.ones(3))
        + log_coefficient * configuration
    )
    untransformed_endpoint_force = accelerations(endpoint_configuration, np.ones(3))

    assert radial_energy > 0.0
    assert np.linalg.norm(untransformed_endpoint_force, ord=np.inf) > 0.1
    assert np.linalg.norm(transformed_endpoint_force, ord=np.inf) < 1e-5


def test_general_hyperbolic_scattering_log_subtraction_cancels_leading_force():
    masses = np.array([1.0, 0.7, 1.4])
    asymptotic_velocities = np.array(
        [
            [1.2, 0.1],
            [-0.4, 0.9],
            [-0.2, -1.1],
        ]
    )
    relative_velocity_distances = [
        np.linalg.norm(asymptotic_velocities[j] - asymptotic_velocities[i])
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    velocity_force = accelerations(asymptotic_velocities, masses)
    tau = 1e-6
    log_term_second_tau_derivative = velocity_force / tau
    log_model_endpoint_lhs = tau * log_term_second_tau_derivative
    endpoint_log_argument = (
        asymptotic_velocities + velocity_force * tau * np.log(tau)
    )
    transformed_endpoint_force = (
        accelerations(endpoint_log_argument, masses) - velocity_force
    )

    assert min(relative_velocity_distances) > 0.1
    assert np.linalg.norm(velocity_force, ord=np.inf) > 0.1
    assert np.linalg.norm(log_model_endpoint_lhs - velocity_force, ord=np.inf) < 1e-14
    assert np.linalg.norm(transformed_endpoint_force, ord=np.inf) < 2e-5


def test_log_subtracted_escape_equation_projects_exactly_to_newton():
    masses = np.array([1.0, 0.7, 1.4])
    endpoint_velocity = np.array(
        [
            [1.2, 0.1],
            [-0.4, 0.9],
            [-0.2, -1.1],
        ]
    )
    log_vector = accelerations(endpoint_velocity, masses)
    tau = 0.08
    regular_remainder = endpoint_velocity + tau * np.array(
        [
            [0.07, -0.03],
            [-0.02, 0.05],
            [0.04, -0.01],
        ]
    )
    scaled_position = regular_remainder + log_vector * tau * np.log(tau)
    transformed_rhs = accelerations(scaled_position, masses) - log_vector
    regular_remainder_second_tau_derivative = transformed_rhs / tau
    scaled_position_second_tau_derivative = (
        regular_remainder_second_tau_derivative + log_vector / tau
    )
    projected_positions = scaled_position / tau
    projected_acceleration = tau**3 * scaled_position_second_tau_derivative

    assert min(
        np.linalg.norm(scaled_position[j] - scaled_position[i])
        for i in range(3)
        for j in range(i + 1, 3)
    ) > 0.1
    assert np.linalg.norm(
        tau * regular_remainder_second_tau_derivative - transformed_rhs,
        ord=np.inf,
    ) < 1e-14
    assert np.linalg.norm(
        projected_acceleration - accelerations(projected_positions, masses),
        ord=np.inf,
    ) < 1e-14


def test_log_subtracted_escape_first_transseries_coefficients_cancel_first_order():
    masses = np.array([1.0, 0.7, 1.4])
    endpoint_velocity = np.array(
        [
            [1.2, 0.1],
            [-0.4, 0.9],
            [-0.2, -1.1],
        ]
    )
    first_remainder_coefficient = np.array(
        [
            [0.2, -0.05],
            [-0.1, 0.08],
            [0.04, 0.03],
        ]
    )
    log_vector = accelerations(endpoint_velocity, masses)
    derivative_log = _acceleration_derivative_apply(endpoint_velocity, masses, log_vector)
    derivative_free = _acceleration_derivative_apply(
        endpoint_velocity,
        masses,
        first_remainder_coefficient,
    )
    tau_log_coefficient = 0.5 * derivative_log
    tau_coefficient = 0.5 * (derivative_free - 3.0 * tau_log_coefficient)

    assert np.linalg.norm(2.0 * tau_log_coefficient - derivative_log, ord=np.inf) < 1e-14
    assert np.linalg.norm(
        2.0 * tau_coefficient + 3.0 * tau_log_coefficient - derivative_free,
        ord=np.inf,
    ) < 1e-14

    residual_norms = []
    scaled_residuals = []
    for tau in (1e-3, 3e-4, 1e-4):
        logarithm = np.log(tau)
        regular_remainder = (
            endpoint_velocity
            + first_remainder_coefficient * tau
            + tau**2 * (tau_log_coefficient * logarithm + tau_coefficient)
        )
        regular_remainder_second_tau_derivative = (
            2.0 * tau_log_coefficient * logarithm
            + 2.0 * tau_coefficient
            + 3.0 * tau_log_coefficient
        )
        scaled_position = regular_remainder + log_vector * tau * logarithm
        transformed_residual = (
            tau * regular_remainder_second_tau_derivative
            - (accelerations(scaled_position, masses) - log_vector)
        )
        residual_norm = np.linalg.norm(transformed_residual, ord=np.inf)
        residual_norms.append(residual_norm)
        scaled_residuals.append(
            residual_norm / (tau**2 * logarithm**2)
        )

    assert residual_norms[0] > residual_norms[1] > residual_norms[2]
    assert scaled_residuals[0] < 0.6
    assert max(scaled_residuals) < 0.6


def test_log_subtracted_escape_formal_recurrence_is_triangular():
    power = 5
    monomial_log_power = 3
    tau = 0.37
    logarithm = np.log(tau)
    coefficients = _tau_second_derivative_log_monomial_coefficients(
        power,
        monomial_log_power,
    )
    direct_value = tau * (
        tau ** (power - 2)
        * (
            power * (power - 1.0) * logarithm**monomial_log_power
            + monomial_log_power * (2.0 * power - 1.0) * logarithm ** (monomial_log_power - 1)
            + monomial_log_power
            * (monomial_log_power - 1.0)
            * logarithm ** (monomial_log_power - 2)
        )
    )
    coefficient_value = tau ** (power - 1) * sum(
        coefficient * logarithm**log_power
        for log_power, coefficient in coefficients.items()
    )

    assert coefficients[monomial_log_power] == power * (power - 1.0)
    assert coefficients[monomial_log_power - 1] == monomial_log_power * (2.0 * power - 1.0)
    assert coefficients[monomial_log_power - 2] == monomial_log_power * (monomial_log_power - 1.0)
    assert direct_value == pytest.approx(coefficient_value)

    forcing = {
        0: 0.5,
        1: -1.2,
        2: 0.75,
        3: -0.4,
    }
    solved = _solve_log_polynomial_row(power, forcing)
    reconstructed = {}
    for log_power, coefficient in solved.items():
        for produced_log_power, produced_coefficient in _tau_second_derivative_log_monomial_coefficients(
            power,
            log_power,
        ).items():
            reconstructed[produced_log_power] = reconstructed.get(produced_log_power, 0.0) + (
                coefficient * produced_coefficient
            )

    assert power * (power - 1.0) > 0.0
    for log_power, forcing_value in forcing.items():
        assert reconstructed[log_power] == pytest.approx(forcing_value)


def test_nonhomothetic_scattering_fixed_point_has_quantitative_contraction_bounds():
    masses = np.array([1.0, 0.7, 1.4])
    asymptotic_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    log_vector = accelerations(asymptotic_velocities, masses)
    velocity_gap = min(
        np.linalg.norm(asymptotic_velocities[j] - asymptotic_velocities[i])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    velocity_area = abs(
        np.linalg.det(
            np.array(
                [
                    asymptotic_velocities[1] - asymptotic_velocities[0],
                    asymptotic_velocities[2] - asymptotic_velocities[0],
                ]
            )
        )
    )
    start_time = 1.0e4
    log_start = np.log(start_time)
    total_mass = float(np.sum(masses))
    lipschitz_constant = 256.0 * total_mass / velocity_gap**3
    body_error_scale = max(np.linalg.norm(offset) for offset in offsets) / log_start + max(
        np.linalg.norm(row) for row in log_vector
    )
    pair_error_scale = max(
        np.linalg.norm(offsets[j] - offsets[i]) / log_start
        + np.linalg.norm(log_vector[j] - log_vector[i])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    residual_integral_factor = 1.0 / (2.0 * log_start) + 3.0 / (
        4.0 * log_start**2
    )
    contraction_integral_factor = (
        1.0 / 6.0
        + 5.0 / (18.0 * log_start)
        + 19.0 / (108.0 * log_start**2)
    ) / start_time
    contraction_factor = lipschitz_constant * contraction_integral_factor
    zero_residual_bound = (
        lipschitz_constant * body_error_scale * residual_integral_factor
    )
    ball_radius = 2.0 * zero_residual_bound
    worst_ball_separation_loss = 2.0 * ball_radius * log_start**2 / start_time**2

    assert velocity_gap > 1.0
    assert velocity_area > 2.0
    assert pair_error_scale * log_start / start_time < 0.5 * velocity_gap
    assert contraction_factor < 0.01
    assert zero_residual_bound + contraction_factor * ball_radius < ball_radius
    assert worst_ball_separation_loss < 0.25 * velocity_gap

    sampled_residual_bounds = []
    for time in (start_time, 10.0 * start_time, 100.0 * start_time):
        model_position = (
            asymptotic_velocities * time - log_vector * np.log(time) + offsets
        )
        pair_distance = min(
            np.linalg.norm(model_position[j] - model_position[i])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        residual = accelerations(model_position, masses) - log_vector / time**2
        scaled_residual = (
            max(np.linalg.norm(row) for row in residual) * time**3 / np.log(time)
        )
        sampled_residual_bounds.append(scaled_residual)

        assert pair_distance > 0.5 * velocity_gap * time
        assert scaled_residual < lipschitz_constant * body_error_scale

    assert max(sampled_residual_bounds) < 1.0


def test_nonhomothetic_scattering_branch_has_all_future_separation_bound():
    masses = np.array([1.0, 0.7, 1.4])
    asymptotic_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    log_vector = accelerations(asymptotic_velocities, masses)
    velocity_gap = min(
        np.linalg.norm(asymptotic_velocities[j] - asymptotic_velocities[i])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    start_time = 1.0e4
    log_start = np.log(start_time)
    lipschitz_constant = 256.0 * float(np.sum(masses)) / velocity_gap**3
    body_error_scale = max(np.linalg.norm(offset) for offset in offsets) / log_start + max(
        np.linalg.norm(row) for row in log_vector
    )
    pair_error_scale = max(
        np.linalg.norm(offsets[j] - offsets[i]) / log_start
        + np.linalg.norm(log_vector[j] - log_vector[i])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    initial_factor = 1.0 / (2.0 * log_start) + 3.0 / (4.0 * log_start**2)
    initial_correction_bound = lipschitz_constant * body_error_scale * initial_factor
    ball_radius = 2.0 * initial_correction_bound

    def separation_envelope(time):
        return (
            velocity_gap * time
            - pair_error_scale * np.log(time)
            - 2.0 * ball_radius * np.log(time) ** 2 / time
        )

    uniform_separation_rate = (
        velocity_gap
        - pair_error_scale * log_start / start_time
        - 2.0 * ball_radius * log_start**2 / start_time**2
    )

    assert uniform_separation_rate > 0.25 * velocity_gap
    assert pair_error_scale * log_start / start_time < 0.5 * velocity_gap
    assert 2.0 * ball_radius * log_start**2 / start_time**2 < 0.25 * velocity_gap

    for time in (start_time, 10.0 * start_time, 100.0 * start_time):
        model_position = (
            asymptotic_velocities * time - log_vector * np.log(time) + offsets
        )
        model_pair_distance = min(
            np.linalg.norm(model_position[j] - model_position[i])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        correction_pair_radius = 2.0 * ball_radius * np.log(time) ** 2 / time

        assert separation_envelope(time) + 1e-9 >= uniform_separation_rate * time
        assert model_pair_distance - correction_pair_radius >= separation_envelope(time)
        assert separation_envelope(time) > 0.25 * velocity_gap * time


def test_nonhomothetic_scattering_picard_operator_preserves_center_of_mass():
    masses = np.array([1.0, 0.7, 1.4])
    asymptotic_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    log_vector = accelerations(asymptotic_velocities, masses)
    total_mass = float(np.sum(masses))
    asymptotic_center_velocity = (
        np.sum(masses[:, None] * asymptotic_velocities, axis=0) / total_mass
    )
    asymptotic_center_offset = np.sum(masses[:, None] * offsets, axis=0) / total_mass

    assert np.linalg.norm(np.sum(masses[:, None] * log_vector, axis=0), ord=np.inf) < 1e-14

    for time in (1.0e4, 3.0e4, 1.0e5):
        model_position = (
            asymptotic_velocities * time - log_vector * np.log(time) + offsets
        )
        model_velocity = asymptotic_velocities - log_vector / time
        model_center = np.sum(masses[:, None] * model_position, axis=0) / total_mass
        model_momentum = np.sum(masses[:, None] * model_velocity, axis=0)
        trial_correction = np.array(
            [
                [0.2, -0.1],
                [-0.05, 0.08],
                [0.03, 0.06],
            ]
        ) * np.log(time) ** 2 / time
        picard_integrand = (
            accelerations(model_position + trial_correction, masses)
            - log_vector / time**2
        )

        np.testing.assert_allclose(
            model_center,
            asymptotic_center_velocity * time + asymptotic_center_offset,
            rtol=1e-14,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            model_momentum,
            np.sum(masses[:, None] * asymptotic_velocities, axis=0),
            rtol=1e-14,
            atol=1e-12,
        )
        assert np.linalg.norm(
            np.sum(masses[:, None] * picard_integrand, axis=0),
            ord=np.inf,
        ) < 1e-14


def test_nonhomothetic_scattering_branch_matches_angular_momentum_and_energy_limits():
    masses = np.array([1.0, 0.7, 1.4])
    asymptotic_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    log_vector = accelerations(asymptotic_velocities, masses)

    def wedge(left, right):
        return left[0] * right[1] - left[1] * right[0]

    angular_log_coefficient = sum(
        mass * wedge(velocity, log_row)
        for mass, velocity, log_row in zip(masses, asymptotic_velocities, log_vector)
    )
    asymptotic_angular_momentum = sum(
        mass * wedge(offset, velocity)
        for mass, offset, velocity in zip(masses, offsets, asymptotic_velocities)
    )
    model_angular_tail_coefficient = sum(
        mass * wedge(offset, log_row)
        for mass, offset, log_row in zip(masses, offsets, log_vector)
    )
    asymptotic_energy = 0.5 * float(
        np.sum(masses[:, None] * asymptotic_velocities**2)
    )

    velocity_gap = min(
        np.linalg.norm(asymptotic_velocities[j] - asymptotic_velocities[i])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    start_time = 1.0e4
    log_start = np.log(start_time)
    lipschitz_constant = 256.0 * float(np.sum(masses)) / velocity_gap**3
    body_error_scale = max(np.linalg.norm(offset) for offset in offsets) / log_start + max(
        np.linalg.norm(row) for row in log_vector
    )
    pair_error_scale = max(
        np.linalg.norm(offsets[j] - offsets[i]) / log_start
        + np.linalg.norm(log_vector[j] - log_vector[i])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    position_initial_factor = 1.0 / (2.0 * log_start) + 3.0 / (
        4.0 * log_start**2
    )
    position_contraction_factor = (
        1.0 / 6.0
        + 5.0 / (18.0 * log_start)
        + 19.0 / (108.0 * log_start**2)
    ) / start_time
    velocity_initial_factor = 1.0 / (2.0 * log_start) + 1.0 / (
        4.0 * log_start**2
    )
    velocity_transfer_factor = (
        1.0 / 3.0
        + 2.0 / (9.0 * log_start)
        + 2.0 / (27.0 * log_start**2)
    ) / start_time
    initial_position_increment = (
        lipschitz_constant * body_error_scale * position_initial_factor
    )
    contraction_factor = lipschitz_constant * position_contraction_factor
    initial_velocity_increment = (
        lipschitz_constant * body_error_scale * velocity_initial_factor
    )
    velocity_transfer = lipschitz_constant * velocity_transfer_factor
    full_velocity_correction_weight = (
        initial_velocity_increment
        + velocity_transfer * initial_position_increment / (1.0 - contraction_factor)
    )
    ball_radius = 2.0 * initial_position_increment
    uniform_separation_rate = (
        velocity_gap
        - pair_error_scale * log_start / start_time
        - 2.0 * ball_radius * log_start**2 / start_time**2
    )
    pair_mass_sum = sum(
        masses[i] * masses[j] for i in range(3) for j in range(i + 1, 3)
    )

    assert abs(angular_log_coefficient) < 1e-14
    assert uniform_separation_rate > 0.25 * velocity_gap
    assert contraction_factor < 0.01

    angular_model_errors = []
    angular_branch_limit_envelopes = []
    energy_limit_envelopes = []
    for time in (start_time, 10.0 * start_time, 100.0 * start_time):
        model_position = (
            asymptotic_velocities * time - log_vector * np.log(time) + offsets
        )
        model_velocity = asymptotic_velocities - log_vector / time
        model_state = np.concatenate([model_position.reshape(-1), model_velocity.reshape(-1)])
        model_angular_momentum = angular_momentum_z(model_state, masses)
        expected_model_angular_momentum = (
            asymptotic_angular_momentum - model_angular_tail_coefficient / time
        )
        model_energy = energy(model_state, masses)

        position_correction_bound = ball_radius * np.log(time) ** 2 / time
        velocity_correction_bound = (
            np.log(time) ** 2 / time**2 * full_velocity_correction_weight
        )
        model_position_bound = (
            max(np.linalg.norm(row) for row in asymptotic_velocities) * time
            + max(np.linalg.norm(row) for row in log_vector) * np.log(time)
            + max(np.linalg.norm(row) for row in offsets)
        )
        model_velocity_bound = (
            max(np.linalg.norm(row) for row in asymptotic_velocities)
            + max(np.linalg.norm(row) for row in log_vector) / time
        )
        angular_branch_limit_envelopes.append(
            float(np.sum(masses))
            * (
                model_position_bound * velocity_correction_bound
                + position_correction_bound
                * (model_velocity_bound + velocity_correction_bound)
            )
        )
        velocity_error_bound = (
            max(np.linalg.norm(row) for row in log_vector) / time
            + velocity_correction_bound
        )
        kinetic_limit_bound = 0.5 * sum(
            mass * (2.0 * np.linalg.norm(velocity) * velocity_error_bound + velocity_error_bound**2)
            for mass, velocity in zip(masses, asymptotic_velocities)
        )
        potential_limit_bound = pair_mass_sum / (uniform_separation_rate * time)
        energy_limit_envelopes.append(kinetic_limit_bound + potential_limit_bound)
        angular_model_errors.append(abs(model_angular_momentum - asymptotic_angular_momentum))

        assert model_angular_momentum == pytest.approx(expected_model_angular_momentum)
        assert abs(model_energy - asymptotic_energy) < energy_limit_envelopes[-1]

    assert angular_model_errors[0] > angular_model_errors[1] > angular_model_errors[2]
    assert angular_branch_limit_envelopes[0] > angular_branch_limit_envelopes[1]
    assert angular_branch_limit_envelopes[1] > angular_branch_limit_envelopes[2]
    assert energy_limit_envelopes[0] > energy_limit_envelopes[1] > energy_limit_envelopes[2]
    assert energy_limit_envelopes[-1] < 5.0e-6


def test_nonhomothetic_scattering_chart_has_uniform_parameter_neighborhood():
    masses = np.array([1.0, 0.7, 1.4])
    asymptotic_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    velocity_direction = np.array(
        [
            [0.03, -0.02],
            [-0.01, 0.04],
            [0.02, 0.01],
        ]
    )
    offset_direction = np.array(
        [
            [0.02, -0.01],
            [-0.03, 0.02],
            [0.01, 0.04],
        ]
    )
    start_time = 1.0e4
    log_start = np.log(start_time)
    position_initial_factor = 1.0 / (2.0 * log_start) + 3.0 / (
        4.0 * log_start**2
    )
    position_contraction_factor = (
        1.0 / 6.0
        + 5.0 / (18.0 * log_start)
        + 19.0 / (108.0 * log_start**2)
    ) / start_time

    contraction_factors = []
    separation_rates = []
    for scale in (-3.0, -1.0, 0.0, 1.0, 3.0):
        velocities = asymptotic_velocities + scale * velocity_direction
        shifted_offsets = offsets + scale * offset_direction
        log_vector = accelerations(velocities, masses)
        velocity_gap = min(
            np.linalg.norm(velocities[j] - velocities[i])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        lipschitz_constant = 256.0 * float(np.sum(masses)) / velocity_gap**3
        body_error_scale = max(np.linalg.norm(offset) for offset in shifted_offsets) / log_start + max(
            np.linalg.norm(row) for row in log_vector
        )
        pair_error_scale = max(
            np.linalg.norm(shifted_offsets[j] - shifted_offsets[i]) / log_start
            + np.linalg.norm(log_vector[j] - log_vector[i])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        initial_increment_bound = (
            lipschitz_constant * body_error_scale * position_initial_factor
        )
        ball_radius = 2.0 * initial_increment_bound
        contraction_factor = lipschitz_constant * position_contraction_factor
        uniform_separation_rate = (
            velocity_gap
            - pair_error_scale * log_start / start_time
            - 2.0 * ball_radius * log_start**2 / start_time**2
        )

        contraction_factors.append(contraction_factor)
        separation_rates.append(uniform_separation_rate / velocity_gap)

        assert velocity_gap > 1.5
        assert contraction_factor < 0.005
        assert uniform_separation_rate > 0.99 * velocity_gap
        assert initial_increment_bound + contraction_factor * ball_radius < ball_radius

    derivative_log_vector = _acceleration_derivative_apply(
        asymptotic_velocities,
        masses,
        velocity_direction,
    )
    finite_difference_step = 1.0e-5
    finite_difference_log_vector = (
        accelerations(
            asymptotic_velocities + finite_difference_step * velocity_direction,
            masses,
        )
        - accelerations(
            asymptotic_velocities - finite_difference_step * velocity_direction,
            masses,
        )
    ) / (2.0 * finite_difference_step)

    np.testing.assert_allclose(
        finite_difference_log_vector,
        derivative_log_vector,
        rtol=1e-9,
        atol=1e-10,
    )

    for time in (start_time, 3.0 * start_time):
        def model_position(velocities, shifted_offsets):
            return (
                velocities * time
                - accelerations(velocities, masses) * np.log(time)
                + shifted_offsets
            )

        finite_difference_model = (
            model_position(
                asymptotic_velocities + finite_difference_step * velocity_direction,
                offsets + finite_difference_step * offset_direction,
            )
            - model_position(
                asymptotic_velocities - finite_difference_step * velocity_direction,
                offsets - finite_difference_step * offset_direction,
            )
        ) / (2.0 * finite_difference_step)
        expected_model_derivative = (
            velocity_direction * time
            - derivative_log_vector * np.log(time)
            + offset_direction
        )

        np.testing.assert_allclose(
            finite_difference_model,
            expected_model_derivative,
            rtol=1e-10,
            atol=1e-6,
        )

    assert max(contraction_factors) < 1.1 * min(contraction_factors)
    assert min(separation_rates) > 0.99


def test_nonhomothetic_scattering_picard_series_has_all_future_tail_bound():
    masses = np.array([1.0, 0.7, 1.4])
    asymptotic_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    log_vector = accelerations(asymptotic_velocities, masses)
    velocity_gap = min(
        np.linalg.norm(asymptotic_velocities[j] - asymptotic_velocities[i])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    start_time = 1.0e4
    log_start = np.log(start_time)
    lipschitz_constant = 256.0 * float(np.sum(masses)) / velocity_gap**3
    body_error_scale = max(np.linalg.norm(offset) for offset in offsets) / log_start + max(
        np.linalg.norm(row) for row in log_vector
    )
    residual_integral_factor = 1.0 / (2.0 * log_start) + 3.0 / (
        4.0 * log_start**2
    )
    contraction_integral_factor = (
        1.0 / 6.0
        + 5.0 / (18.0 * log_start)
        + 19.0 / (108.0 * log_start**2)
    ) / start_time
    initial_increment_bound = (
        lipschitz_constant * body_error_scale * residual_integral_factor
    )
    contraction_factor = lipschitz_constant * contraction_integral_factor
    weighted_tail_bounds = [
        contraction_factor**retained
        * initial_increment_bound
        / (1.0 - contraction_factor)
        for retained in range(5)
    ]

    assert contraction_factor < 0.01
    assert weighted_tail_bounds[0] > initial_increment_bound
    for previous, current in zip(weighted_tail_bounds, weighted_tail_bounds[1:]):
        assert current == pytest.approx(contraction_factor * previous)
        assert current < 0.005 * previous

    retained_corrections = 3
    weighted_tail = weighted_tail_bounds[retained_corrections]
    pointwise_tail_bounds = [
        np.log(time) ** 2 / time * weighted_tail
        for time in (start_time, 10.0 * start_time, 100.0 * start_time)
    ]

    assert pointwise_tail_bounds[0] < 1.0e-8
    assert pointwise_tail_bounds[0] > pointwise_tail_bounds[1] > pointwise_tail_bounds[2]


def test_nonhomothetic_scattering_picard_truncations_have_newton_residual_bound():
    masses = np.array([1.0, 0.7, 1.4])
    asymptotic_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    log_vector = accelerations(asymptotic_velocities, masses)
    velocity_gap = min(
        np.linalg.norm(asymptotic_velocities[j] - asymptotic_velocities[i])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    start_time = 1.0e4
    log_start = np.log(start_time)
    lipschitz_constant = 256.0 * float(np.sum(masses)) / velocity_gap**3
    body_error_scale = max(np.linalg.norm(offset) for offset in offsets) / log_start + max(
        np.linalg.norm(row) for row in log_vector
    )
    residual_integral_factor = 1.0 / (2.0 * log_start) + 3.0 / (
        4.0 * log_start**2
    )
    contraction_integral_factor = (
        1.0 / 6.0
        + 5.0 / (18.0 * log_start)
        + 19.0 / (108.0 * log_start**2)
    ) / start_time
    initial_increment_bound = (
        lipschitz_constant * body_error_scale * residual_integral_factor
    )
    contraction_factor = lipschitz_constant * contraction_integral_factor

    def residual_envelope(retained_corrections, time):
        return (
            lipschitz_constant
            * initial_increment_bound
            * contraction_factor ** (retained_corrections - 1)
            * np.log(time) ** 2
            / time**4
        )

    first_residual_envelopes = [
        residual_envelope(1, time)
        for time in (start_time, 10.0 * start_time, 100.0 * start_time)
    ]
    third_residual_envelopes = [
        residual_envelope(3, time)
        for time in (start_time, 10.0 * start_time, 100.0 * start_time)
    ]

    assert contraction_factor < 0.01
    assert first_residual_envelopes[0] > first_residual_envelopes[1]
    assert first_residual_envelopes[1] > first_residual_envelopes[2]
    assert third_residual_envelopes[0] < 1.0e-15
    for time in (start_time, 10.0 * start_time, 100.0 * start_time):
        assert residual_envelope(4, time) == pytest.approx(
            contraction_factor * residual_envelope(3, time)
        )


def test_nonhomothetic_scattering_picard_series_has_velocity_tail_bound():
    masses = np.array([1.0, 0.7, 1.4])
    asymptotic_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    log_vector = accelerations(asymptotic_velocities, masses)
    velocity_gap = min(
        np.linalg.norm(asymptotic_velocities[j] - asymptotic_velocities[i])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    start_time = 1.0e4
    log_start = np.log(start_time)
    lipschitz_constant = 256.0 * float(np.sum(masses)) / velocity_gap**3
    body_error_scale = max(np.linalg.norm(offset) for offset in offsets) / log_start + max(
        np.linalg.norm(row) for row in log_vector
    )
    position_initial_factor = 1.0 / (2.0 * log_start) + 3.0 / (
        4.0 * log_start**2
    )
    position_contraction_factor = (
        1.0 / 6.0
        + 5.0 / (18.0 * log_start)
        + 19.0 / (108.0 * log_start**2)
    ) / start_time
    velocity_initial_factor = 1.0 / (2.0 * log_start) + 1.0 / (
        4.0 * log_start**2
    )
    velocity_transfer_factor = (
        1.0 / 3.0
        + 2.0 / (9.0 * log_start)
        + 2.0 / (27.0 * log_start**2)
    ) / start_time
    initial_position_increment = (
        lipschitz_constant * body_error_scale * position_initial_factor
    )
    contraction_factor = lipschitz_constant * position_contraction_factor
    initial_velocity_increment = (
        lipschitz_constant * body_error_scale * velocity_initial_factor
    )
    velocity_transfer = lipschitz_constant * velocity_transfer_factor

    def velocity_tail_envelope(retained_corrections, time):
        return (
            np.log(time) ** 2
            / time**2
            * velocity_transfer
            * initial_position_increment
            * contraction_factor ** (retained_corrections - 1)
            / (1.0 - contraction_factor)
        )

    full_velocity_correction_weight = (
        initial_velocity_increment
        + velocity_transfer * initial_position_increment / (1.0 - contraction_factor)
    )

    def asymptotic_velocity_envelope(time):
        return (
            max(np.linalg.norm(row) for row in log_vector) / time
            + np.log(time) ** 2 / time**2 * full_velocity_correction_weight
        )

    assert contraction_factor < 0.01
    assert velocity_transfer > contraction_factor
    for time in (start_time, 10.0 * start_time, 100.0 * start_time):
        assert velocity_tail_envelope(4, time) == pytest.approx(
            contraction_factor * velocity_tail_envelope(3, time)
        )
        assert velocity_tail_envelope(3, time) < 2e-12

    asymptotic_velocity_bounds = [
        asymptotic_velocity_envelope(time)
        for time in (start_time, 10.0 * start_time, 100.0 * start_time)
    ]

    assert asymptotic_velocity_bounds[0] > asymptotic_velocity_bounds[1]
    assert asymptotic_velocity_bounds[1] > asymptotic_velocity_bounds[2]
    assert asymptotic_velocity_bounds[-1] < 1.0e-6


def test_scattering_chart_glues_to_finite_taylor_handoff_with_future_tail():
    masses = np.array([1.0, 0.7, 1.4])
    asymptotic_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    log_vector = accelerations(asymptotic_velocities, masses)
    velocity_gap = min(
        np.linalg.norm(asymptotic_velocities[j] - asymptotic_velocities[i])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    start_time = 1.0e4
    log_start = np.log(start_time)
    lipschitz_constant = 256.0 * float(np.sum(masses)) / velocity_gap**3
    body_error_scale = max(np.linalg.norm(offset) for offset in offsets) / log_start + max(
        np.linalg.norm(row) for row in log_vector
    )
    pair_error_scale = max(
        np.linalg.norm(offsets[j] - offsets[i]) / log_start
        + np.linalg.norm(log_vector[j] - log_vector[i])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    position_initial_factor = 1.0 / (2.0 * log_start) + 3.0 / (
        4.0 * log_start**2
    )
    position_contraction_factor = (
        1.0 / 6.0
        + 5.0 / (18.0 * log_start)
        + 19.0 / (108.0 * log_start**2)
    ) / start_time
    velocity_initial_factor = 1.0 / (2.0 * log_start) + 1.0 / (
        4.0 * log_start**2
    )
    velocity_transfer_factor = (
        1.0 / 3.0
        + 2.0 / (9.0 * log_start)
        + 2.0 / (27.0 * log_start**2)
    ) / start_time
    initial_position_increment = (
        lipschitz_constant * body_error_scale * position_initial_factor
    )
    contraction_factor = lipschitz_constant * position_contraction_factor
    ball_radius = 2.0 * initial_position_increment
    velocity_transfer = lipschitz_constant * velocity_transfer_factor
    full_velocity_correction_weight = (
        lipschitz_constant * body_error_scale * velocity_initial_factor
        + velocity_transfer * initial_position_increment / (1.0 - contraction_factor)
    )

    model_position = (
        asymptotic_velocities * start_time - log_vector * log_start + offsets
    )
    model_velocity = asymptotic_velocities - log_vector / start_time
    model_pair_distance = min(
        np.linalg.norm(model_position[j] - model_position[i])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    endpoint_position_radius = ball_radius * log_start**2 / start_time
    endpoint_velocity_radius = (
        log_start**2 / start_time**2 * full_velocity_correction_weight
    )
    endpoint_pair_floor = model_pair_distance - 2.0 * endpoint_position_radius
    proven_pair_floor = (
        velocity_gap * start_time
        - pair_error_scale * log_start
        - 2.0 * ball_radius * log_start**2 / start_time
    )
    local_speed_bound = (
        max(np.linalg.norm(row) for row in model_velocity) + endpoint_velocity_radius
    )
    local_acceleration_bound = float(np.sum(masses)) / endpoint_pair_floor**2
    ordinary_handoff_step = 0.01 * min(
        endpoint_pair_floor / local_speed_bound,
        np.sqrt(endpoint_pair_floor / local_acceleration_bound),
    )

    assert contraction_factor < 0.01
    assert endpoint_pair_floor >= proven_pair_floor
    assert proven_pair_floor > 0.25 * velocity_gap * start_time
    assert endpoint_position_radius < 0.2
    assert endpoint_velocity_radius < 1.0e-5
    assert ordinary_handoff_step > 100.0

    retained_corrections = 3
    weighted_tail = (
        contraction_factor**retained_corrections
        * initial_position_increment
        / (1.0 - contraction_factor)
    )
    weighted_velocity_tail = (
        velocity_transfer
        * initial_position_increment
        * contraction_factor ** (retained_corrections - 1)
        / (1.0 - contraction_factor)
    )

    def position_tail_envelope(time):
        return np.log(time) ** 2 / time * weighted_tail

    def velocity_tail_envelope(time):
        return np.log(time) ** 2 / time**2 * weighted_velocity_tail

    def residual_envelope(time):
        return (
            lipschitz_constant
            * initial_position_increment
            * contraction_factor ** (retained_corrections - 1)
            * np.log(time) ** 2
            / time**4
        )

    future_times = (start_time, 10.0 * start_time, 100.0 * start_time)
    position_tails = [position_tail_envelope(time) for time in future_times]
    velocity_tails = [velocity_tail_envelope(time) for time in future_times]
    residual_tails = [residual_envelope(time) for time in future_times]

    assert position_tails[0] < 1.0e-8
    assert velocity_tails[0] < 2.0e-12
    assert residual_tails[0] < 5.0e-16
    assert position_tails[0] > position_tails[1] > position_tails[2]
    assert velocity_tails[0] > velocity_tails[1] > velocity_tails[2]
    assert residual_tails[0] > residual_tails[1] > residual_tails[2]
    assert position_tails[0] < 1.0e-7 * endpoint_position_radius


def test_nonhomothetic_scattering_compact_time_endpoint_has_log_subtracted_tail_bound():
    masses = np.array([1.0, 0.7, 1.4])
    asymptotic_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    log_vector = accelerations(asymptotic_velocities, masses)
    velocity_gap = min(
        np.linalg.norm(asymptotic_velocities[j] - asymptotic_velocities[i])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    start_time = 1.0e4
    log_start = np.log(start_time)
    lipschitz_constant = 256.0 * float(np.sum(masses)) / velocity_gap**3
    body_error_scale = max(np.linalg.norm(offset) for offset in offsets) / log_start + max(
        np.linalg.norm(row) for row in log_vector
    )
    position_initial_factor = 1.0 / (2.0 * log_start) + 3.0 / (
        4.0 * log_start**2
    )
    position_contraction_factor = (
        1.0 / 6.0
        + 5.0 / (18.0 * log_start)
        + 19.0 / (108.0 * log_start**2)
    ) / start_time
    velocity_initial_factor = 1.0 / (2.0 * log_start) + 1.0 / (
        4.0 * log_start**2
    )
    velocity_transfer_factor = (
        1.0 / 3.0
        + 2.0 / (9.0 * log_start)
        + 2.0 / (27.0 * log_start**2)
    ) / start_time
    initial_position_increment = (
        lipschitz_constant * body_error_scale * position_initial_factor
    )
    contraction_factor = lipschitz_constant * position_contraction_factor
    velocity_transfer = lipschitz_constant * velocity_transfer_factor
    ball_radius = 2.0 * initial_position_increment
    retained_corrections = 3
    endpoint_tail_weight = (
        contraction_factor**retained_corrections
        * initial_position_increment
        / (1.0 - contraction_factor)
    )
    full_velocity_weight = (
        lipschitz_constant * body_error_scale * velocity_initial_factor
        + velocity_transfer * initial_position_increment / (1.0 - contraction_factor)
    )
    derivative_tail_weight = (
        endpoint_tail_weight
        + velocity_transfer
        * initial_position_increment
        * contraction_factor ** (retained_corrections - 1)
        / (1.0 - contraction_factor)
    )

    compact_time_rate = 1.0e-6
    future_times = (start_time, 10.0 * start_time, 100.0 * start_time)
    compact_endpoint_tails = []
    finite_picard_endpoint_tails = []
    compact_endpoint_first_jet_tails = []
    finite_picard_first_jet_tails = []
    endpoint_equation_residuals = []
    physical_residuals = []

    for time in future_times:
        compact_parameter = np.tanh(compact_time_rate * time)
        recovered_tau = compact_time_rate / np.arctanh(compact_parameter)
        tau = 1.0 / time
        log_tau = np.log(tau)
        scaled_model = tau * (
            asymptotic_velocities * time - log_vector * np.log(time) + offsets
        )
        log_subtracted_model = scaled_model - log_vector * tau * log_tau
        scaled_model_tau_derivative = offsets + log_vector * (log_tau + 1.0)
        log_subtracted_model_tau_derivative = scaled_model_tau_derivative - log_vector * (
            log_tau + 1.0
        )

        compact_endpoint_tail = ball_radius * tau**2 * np.log(1.0 / tau) ** 2
        finite_picard_endpoint_tail = (
            endpoint_tail_weight * tau**2 * np.log(1.0 / tau) ** 2
        )
        compact_endpoint_first_jet_tail = (
            (ball_radius + full_velocity_weight) * tau * np.log(1.0 / tau) ** 2
        )
        finite_picard_first_jet_tail = (
            derivative_tail_weight * tau * np.log(1.0 / tau) ** 2
        )
        endpoint_equation_residual = (
            lipschitz_constant
            * initial_position_increment
            * contraction_factor ** (retained_corrections - 1)
            * tau**2
            * np.log(1.0 / tau) ** 2
        )
        physical_residual = tau**2 * endpoint_equation_residual
        forced_log_size = max(np.linalg.norm(row) for row in log_vector * tau * log_tau)
        forced_log_derivative_size = max(
            np.linalg.norm(row) for row in log_vector * (log_tau + 1.0)
        )

        assert recovered_tau == pytest.approx(tau)
        assert np.linalg.norm(
            log_subtracted_model - (asymptotic_velocities + offsets * tau),
            ord=np.inf,
        ) < 1.0e-15
        assert np.linalg.norm(log_subtracted_model_tau_derivative - offsets, ord=np.inf) < 1.0e-15
        assert compact_endpoint_tail < forced_log_size
        assert compact_endpoint_first_jet_tail < forced_log_derivative_size
        assert finite_picard_endpoint_tail < 1.0e-6 * compact_endpoint_tail
        assert finite_picard_first_jet_tail < 1.0e-6 * compact_endpoint_first_jet_tail
        assert physical_residual / tau**2 == pytest.approx(endpoint_equation_residual)
        assert endpoint_equation_residual < 1.0e-6 * forced_log_derivative_size
        compact_endpoint_tails.append(compact_endpoint_tail)
        finite_picard_endpoint_tails.append(finite_picard_endpoint_tail)
        compact_endpoint_first_jet_tails.append(compact_endpoint_first_jet_tail)
        finite_picard_first_jet_tails.append(finite_picard_first_jet_tail)
        endpoint_equation_residuals.append(endpoint_equation_residual)
        physical_residuals.append(physical_residual)

    assert contraction_factor < 0.01
    assert compact_endpoint_tails[0] > compact_endpoint_tails[1] > compact_endpoint_tails[2]
    assert (
        finite_picard_endpoint_tails[0]
        > finite_picard_endpoint_tails[1]
        > finite_picard_endpoint_tails[2]
    )
    assert (
        compact_endpoint_first_jet_tails[0]
        > compact_endpoint_first_jet_tails[1]
        > compact_endpoint_first_jet_tails[2]
    )
    assert (
        finite_picard_first_jet_tails[0]
        > finite_picard_first_jet_tails[1]
        > finite_picard_first_jet_tails[2]
    )
    assert (
        endpoint_equation_residuals[0]
        > endpoint_equation_residuals[1]
        > endpoint_equation_residuals[2]
    )
    assert physical_residuals[0] > physical_residuals[1] > physical_residuals[2]
    assert compact_endpoint_tails[-1] < 5.0e-7
    assert compact_endpoint_first_jet_tails[-1] < 2.0e-2
    assert endpoint_equation_residuals[-1] < 1.0e-10


def test_nonhomothetic_scattering_compact_endpoint_shell_tails_are_geometric():
    masses = np.array([1.0, 0.7, 1.4])
    asymptotic_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    start_time = 1.0e4
    retained_corrections = 3
    constants = construct_scattering_endpoint_constants(
        masses,
        asymptotic_velocities,
        offsets,
        start_time=start_time,
    )
    recurrence = construct_scattering_endpoint_dyadic_recurrence(
        constants,
        retained_corrections=retained_corrections,
    )
    compact_time_rate = 1.0e-8

    def shell_tail_sequences(shell_count):
        shell_left_taus = [
            recurrence.shell_left_tau(index) for index in range(shell_count)
        ]
        value_tails = [
            recurrence.shell_tail_bound("value", index)
            for index in range(shell_count)
        ]
        first_jet_tails = [
            recurrence.shell_tail_bound("first_jet", index)
            for index in range(shell_count)
        ]
        residual_tails = [
            recurrence.shell_tail_bound("residual", index)
            for index in range(shell_count)
        ]
        return shell_left_taus, value_tails, first_jet_tails, residual_tails

    shell_left_taus, value_tails, first_jet_tails, residual_tails = shell_tail_sequences(
        12
    )
    compact_shell_left_edges = [
        np.tanh(compact_time_rate / tau) for tau in shell_left_taus
    ]

    assert constants.contraction_factor < 0.01
    assert recurrence.certified
    assert recurrence.component_ratios["value"] < recurrence.component_ratios["first_jet"] < 1.0
    assert recurrence.component_ratios["residual"] == pytest.approx(
        recurrence.component_ratios["value"]
    )
    assert compact_shell_left_edges[0] < compact_shell_left_edges[-1] < 1.0
    for previous, current in zip(
        compact_shell_left_edges,
        compact_shell_left_edges[1:],
    ):
        assert previous < current

    for component, tails in (
        ("value", value_tails),
        ("first_jet", first_jet_tails),
        ("residual", residual_tails),
    ):
        for previous, current in zip(tails, tails[1:]):
            assert current <= recurrence.component_ratios[component] * previous

    _taus, long_value_tails, long_first_jet_tails, long_residual_tails = (
        shell_tail_sequences(40)
    )
    for component, tails in (
        ("value", long_value_tails),
        ("first_jet", long_first_jet_tails),
        ("residual", long_residual_tails),
    ):
        geometric_sum_bound = recurrence.all_future_tail_bound(component)
        assert sum(tails) < geometric_sum_bound

    assert long_value_tails[-1] < 1.0e-25
    assert long_first_jet_tails[-1] < 1.0e-16
    assert long_residual_tails[-1] < 1.0e-25


def test_hyperbolic_scattering_effective_tail_start_closes_future_recurrence():
    masses = np.array([1.0, 0.7, 1.4])
    asymptotic_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    retained_corrections = 3
    constants = find_scattering_endpoint_tail_start(
        masses,
        asymptotic_velocities,
        offsets,
        max_doublings=20,
    )
    recurrence = construct_scattering_endpoint_dyadic_recurrence(
        constants,
        retained_corrections=retained_corrections,
    )
    shell_tails = [
        recurrence.shell_tail_bound("value", index)
        for index in range(50)
    ]

    assert constants.start_time < 1.0e3
    assert constants.certified
    assert constants.ball_radius >= 2.0 * constants.initial_position_increment
    assert recurrence.component_ratios["value"] < 1.0
    for previous, current in zip(shell_tails, shell_tails[1:]):
        assert current <= recurrence.component_ratios["value"] * previous
    assert sum(shell_tails) < recurrence.all_future_tail_bound("value")


def test_hyperbolic_scattering_tail_start_constructor_rejects_uncertified_inputs():
    masses = np.array([1.0, 0.7, 1.4])
    asymptotic_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )

    with pytest.raises(ValueError, match="not certified"):
        construct_scattering_endpoint_constants(
            masses,
            asymptotic_velocities,
            offsets,
            start_time=float(np.exp(2.0)),
        )

    repeated_velocity = asymptotic_velocities.copy()
    repeated_velocity[1] = repeated_velocity[0]
    with pytest.raises(ValueError, match="distinct"):
        construct_scattering_endpoint_constants(
            masses,
            repeated_velocity,
            offsets,
            start_time=1.0e4,
        )

    constants = find_scattering_endpoint_tail_start(
        masses,
        asymptotic_velocities,
        offsets,
    )
    with pytest.raises(ValueError, match="not certified"):
        construct_scattering_endpoint_dyadic_recurrence(
            constants,
            retained_corrections=0,
        )


def test_two_ended_scattering_effective_tail_starts_feed_all_real_recurrence():
    masses = np.array([1.0, 0.7, 1.4])
    past_physical_velocities = np.array(
        [
            [-0.9, 0.25],
            [0.25, 0.95],
            [1.15, -0.5],
        ]
    )
    future_physical_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    past_offsets = np.array(
        [
            [0.05, -0.25],
            [-0.15, 0.08],
            [0.35, 0.18],
        ]
    )
    future_offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    retained_corrections = 3
    atlas = construct_two_ended_scattering_atlas_recurrence(
        masses,
        past_physical_velocities,
        past_offsets,
        future_physical_velocities,
        future_offsets,
        retained_corrections=retained_corrections,
        middle_budgets={
            "value": 1.0e-1,
            "first_jet": 1.0e-1,
            "residual": 1.0e-1,
            "physical_residual": 1.0e-1,
        },
        compact_time_rate=1.0e-8,
    )
    past_tails = [atlas.past.shell_tail_bound("value", index) for index in range(50)]
    future_tails = [atlas.future.shell_tail_bound("value", index) for index in range(50)]

    assert atlas.certified
    assert atlas.past.constants.start_time < 1.0e3
    assert atlas.future.constants.start_time < 1.0e3
    assert -1.0 < atlas.past_handoff_compact_parameter < 0.0
    assert 0.0 < atlas.future_handoff_compact_parameter < 1.0
    assert atlas.past.constants.shell_ratio < 1.0
    assert atlas.future.constants.shell_ratio < 1.0
    assert atlas.past.constants.separation_loss + atlas.past.constants.tube_loss < (
        0.625 * atlas.past.constants.velocity_gap
    )
    assert atlas.future.constants.separation_loss + atlas.future.constants.tube_loss < (
        0.625 * atlas.future.constants.velocity_gap
    )
    for recurrence, tails in ((atlas.past, past_tails), (atlas.future, future_tails)):
        for previous, current in zip(tails, tails[1:]):
            assert current <= recurrence.component_ratios["value"] * previous
        assert sum(tails) < recurrence.all_future_tail_bound("value")
    assert atlas.endpoint_budget("value") < atlas.middle.value


def test_past_infinity_scattering_endpoint_uses_time_reversal_signs():
    masses = np.array([1.0, 0.7, 1.4])
    asymptotic_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    future_log_vector = accelerations(asymptotic_velocities, masses)
    past_log_vector = accelerations(-asymptotic_velocities, masses)
    assert np.linalg.norm(past_log_vector + future_log_vector, ord=np.inf) < 1.0e-15

    velocity_gap = min(
        np.linalg.norm(asymptotic_velocities[j] - asymptotic_velocities[i])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    start_time = 1.0e4
    log_start = np.log(start_time)
    lipschitz_constant = 256.0 * float(np.sum(masses)) / velocity_gap**3
    body_error_scale = max(np.linalg.norm(offset) for offset in offsets) / log_start + max(
        np.linalg.norm(row) for row in past_log_vector
    )
    position_initial_factor = 1.0 / (2.0 * log_start) + 3.0 / (
        4.0 * log_start**2
    )
    position_contraction_factor = (
        1.0 / 6.0
        + 5.0 / (18.0 * log_start)
        + 19.0 / (108.0 * log_start**2)
    ) / start_time
    initial_position_increment = (
        lipschitz_constant * body_error_scale * position_initial_factor
    )
    contraction_factor = lipschitz_constant * position_contraction_factor
    retained_corrections = 3

    compact_time_rate = 1.0e-6
    past_physical_times = (-start_time, -10.0 * start_time, -100.0 * start_time)
    past_endpoint_tails = []
    lifted_residuals = []

    for physical_time in past_physical_times:
        compact_parameter = np.tanh(compact_time_rate * physical_time)
        tau = 1.0 / (-physical_time)
        recovered_tau = -compact_time_rate / np.arctanh(compact_parameter)
        log_tau = np.log(tau)
        scaled_past_model = tau * (
            asymptotic_velocities * physical_time
            + future_log_vector * np.log(-physical_time)
            + offsets
        )
        log_subtracted_past_model = (
            scaled_past_model - past_log_vector * tau * log_tau
        )
        scaled_past_derivative = offsets - future_log_vector * (log_tau + 1.0)
        log_subtracted_past_derivative = (
            scaled_past_derivative - past_log_vector * (log_tau + 1.0)
        )
        endpoint_tail = (
            2.0
            * initial_position_increment
            * tau**2
            * np.log(1.0 / tau) ** 2
        )
        lifted_residual = (
            lipschitz_constant
            * initial_position_increment
            * contraction_factor ** (retained_corrections - 1)
            * tau**2
            * np.log(1.0 / tau) ** 2
        )

        assert recovered_tau == pytest.approx(tau)
        assert np.linalg.norm(
            log_subtracted_past_model - (-asymptotic_velocities + offsets * tau),
            ord=np.inf,
        ) < 1.0e-15
        assert np.linalg.norm(log_subtracted_past_derivative - offsets, ord=np.inf) < 1.0e-15
        assert endpoint_tail > lifted_residual
        past_endpoint_tails.append(endpoint_tail)
        lifted_residuals.append(lifted_residual)

    assert contraction_factor < 0.01
    assert past_endpoint_tails[0] > past_endpoint_tails[1] > past_endpoint_tails[2]
    assert lifted_residuals[0] > lifted_residuals[1] > lifted_residuals[2]
    assert lifted_residuals[-1] < 1.0e-10


def test_two_ended_scattering_atlas_has_piecewise_all_time_tail_bounds():
    masses = np.array([1.0, 0.7, 1.4])
    past_physical_velocities = np.array(
        [
            [-0.9, 0.25],
            [0.25, 0.95],
            [1.15, -0.5],
        ]
    )
    future_physical_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    past_offsets = np.array(
        [
            [0.05, -0.25],
            [-0.15, 0.08],
            [0.35, 0.18],
        ]
    )
    future_offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    start_time = 1.0e4
    retained_corrections = 3

    def endpoint_constants(chart_velocities, offsets):
        log_start = np.log(start_time)
        log_vector = accelerations(chart_velocities, masses)
        velocity_gap = min(
            np.linalg.norm(chart_velocities[j] - chart_velocities[i])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        lipschitz_constant = 256.0 * float(np.sum(masses)) / velocity_gap**3
        body_error_scale = max(np.linalg.norm(offset) for offset in offsets) / log_start + max(
            np.linalg.norm(row) for row in log_vector
        )
        pair_error_scale = max(
            np.linalg.norm(offsets[j] - offsets[i]) / log_start
            + np.linalg.norm(log_vector[j] - log_vector[i])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        position_initial_factor = 1.0 / (2.0 * log_start) + 3.0 / (
            4.0 * log_start**2
        )
        position_contraction_factor = (
            1.0 / 6.0
            + 5.0 / (18.0 * log_start)
            + 19.0 / (108.0 * log_start**2)
        ) / start_time
        velocity_transfer_factor = (
            1.0 / 3.0
            + 2.0 / (9.0 * log_start)
            + 2.0 / (27.0 * log_start**2)
        ) / start_time
        eta = lipschitz_constant * body_error_scale * position_initial_factor
        kappa = lipschitz_constant * position_contraction_factor
        velocity_transfer = lipschitz_constant * velocity_transfer_factor
        radius = 2.0 * eta
        pair_floor = (
            velocity_gap * start_time
            - pair_error_scale * log_start
            - 2.0 * radius * log_start**2 / start_time
        )

        def position_tail(time):
            return (
                np.log(time) ** 2
                / time
                * kappa**retained_corrections
                * eta
                / (1.0 - kappa)
            )

        def velocity_tail(time):
            return (
                np.log(time) ** 2
                / time**2
                * velocity_transfer
                * eta
                * kappa ** (retained_corrections - 1)
                / (1.0 - kappa)
            )

        def lifted_endpoint_residual(time):
            tau = 1.0 / time
            return (
                lipschitz_constant
                * eta
                * kappa ** (retained_corrections - 1)
                * tau**2
                * np.log(1.0 / tau) ** 2
            )

        return {
            "gap": velocity_gap,
            "kappa": kappa,
            "pair_floor": pair_floor,
            "position_tail": position_tail,
            "velocity_tail": velocity_tail,
            "lifted_endpoint_residual": lifted_endpoint_residual,
        }

    past = endpoint_constants(-past_physical_velocities, past_offsets)
    future = endpoint_constants(future_physical_velocities, future_offsets)
    compact_time_rate = 1.0e-6
    past_handoff_u = np.tanh(-compact_time_rate * start_time)
    future_handoff_u = np.tanh(compact_time_rate * start_time)
    middle_cauchy_tail = 5.0e-8
    middle_residual = 1.0e-10

    assert past_handoff_u < 0.0 < future_handoff_u
    assert past["kappa"] < 0.01
    assert future["kappa"] < 0.01
    assert past["pair_floor"] > 0.25 * past["gap"] * start_time
    assert future["pair_floor"] > 0.25 * future["gap"] * start_time
    assert middle_cauchy_tail < 1.0e-6 * min(past["pair_floor"], future["pair_floor"])

    past_times = (start_time, 10.0 * start_time, 100.0 * start_time)
    future_times = (start_time, 10.0 * start_time, 100.0 * start_time)
    past_position_tails = [past["position_tail"](time) for time in past_times]
    future_position_tails = [future["position_tail"](time) for time in future_times]
    past_velocity_tails = [past["velocity_tail"](time) for time in past_times]
    future_velocity_tails = [future["velocity_tail"](time) for time in future_times]
    past_residuals = [past["lifted_endpoint_residual"](time) for time in past_times]
    future_residuals = [future["lifted_endpoint_residual"](time) for time in future_times]

    assert past_position_tails[0] > past_position_tails[1] > past_position_tails[2]
    assert future_position_tails[0] > future_position_tails[1] > future_position_tails[2]
    assert past_velocity_tails[0] > past_velocity_tails[1] > past_velocity_tails[2]
    assert future_velocity_tails[0] > future_velocity_tails[1] > future_velocity_tails[2]
    assert past_residuals[0] > past_residuals[1] > past_residuals[2]
    assert future_residuals[0] > future_residuals[1] > future_residuals[2]
    assert max(past_position_tails[0], future_position_tails[0]) < middle_cauchy_tail
    assert max(past_residuals[-1], future_residuals[-1]) < middle_residual


def test_two_ended_scattering_atlas_has_summable_endpoint_shell_recurrences():
    masses = np.array([1.0, 0.7, 1.4])
    past_physical_velocities = np.array(
        [
            [-0.9, 0.25],
            [0.25, 0.95],
            [1.15, -0.5],
        ]
    )
    future_physical_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    past_offsets = np.array(
        [
            [0.05, -0.25],
            [-0.15, 0.08],
            [0.35, 0.18],
        ]
    )
    future_offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    retained_corrections = 3
    compact_time_rate = 1.0e-8
    atlas = construct_two_ended_scattering_atlas_recurrence(
        masses,
        past_physical_velocities,
        past_offsets,
        future_physical_velocities,
        future_offsets,
        retained_corrections=retained_corrections,
        middle_budgets={
            "value": 5.0e-8,
            "first_jet": 5.0e-7,
            "residual": 1.0e-4,
            "physical_residual": 1.0e-10,
        },
        compact_time_rate=compact_time_rate,
        initial_start_time=1.0e4,
        max_doublings=0,
    )
    past_taus = [atlas.past.shell_left_tau(index) for index in range(14)]
    future_taus = [atlas.future.shell_left_tau(index) for index in range(14)]
    past_compact_edges = [np.tanh(-compact_time_rate / tau) for tau in past_taus]
    future_compact_edges = [np.tanh(compact_time_rate / tau) for tau in future_taus]

    assert atlas.certified
    assert atlas.past.constants.contraction_factor < 0.01
    assert atlas.future.constants.contraction_factor < 0.01
    assert atlas.past.component_ratios["value"] < atlas.past.component_ratios["first_jet"] < 1.0
    assert atlas.future.component_ratios["value"] < atlas.future.component_ratios["first_jet"] < 1.0
    assert -1.0 < past_compact_edges[-1] < past_compact_edges[0] < 0.0
    assert 0.0 < future_compact_edges[0] < future_compact_edges[-1] < 1.0

    for recurrence in (atlas.past, atlas.future):
        for component in ("value", "first_jet", "residual"):
            tails = [recurrence.shell_tail_bound(component, index) for index in range(14)]
            for previous, current in zip(tails, tails[1:]):
                assert current <= recurrence.component_ratios[component] * previous

    for component in ("value", "first_jet", "residual"):
        finite_prefix_sum = sum(
            atlas.past.shell_tail_bound(component, index)
            + atlas.future.shell_tail_bound(component, index)
            for index in range(48)
        )
        assert finite_prefix_sum < atlas.endpoint_budget(component)
        assert atlas.endpoint_budget(component) < atlas.middle.for_component(component)
        assert np.isfinite(atlas.total_budget(component))


def test_two_ended_scattering_allows_finite_mixed_collision_middle_budget():
    masses = np.array([1.0, 0.7, 1.4])
    past_physical_velocities = np.array(
        [
            [-0.9, 0.25],
            [0.25, 0.95],
            [1.15, -0.5],
        ]
    )
    future_physical_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    past_offsets = np.array(
        [
            [0.05, -0.25],
            [-0.15, 0.08],
            [0.35, 0.18],
        ]
    )
    future_offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    start_time = 1.0e4
    retained_corrections = 3
    middle_charts = (
        {
            "kind": "ordinary_taylor",
            "value": 2.0e-8,
            "first_jet": 8.0e-8,
            "lifted_residual": 2.0e-7,
            "physical_residual": 2.0e-12,
        },
        {
            "kind": "separated_binary_levi_civita",
            "value": 4.0e-8,
            "first_jet": 1.2e-7,
            "lifted_residual": 5.0e-7,
            "physical_residual": 5.0e-12,
        },
        {
            "kind": "identity_selector_total_collision",
            "value": 3.0e-8,
            "first_jet": 1.0e-7,
            "lifted_residual": 4.0e-7,
            "physical_residual": 4.0e-12,
        },
        {
            "kind": "ordinary_taylor",
            "value": 2.0e-8,
            "first_jet": 8.0e-8,
            "lifted_residual": 2.0e-7,
            "physical_residual": 2.0e-12,
        },
    )

    middle_kinds = {chart["kind"] for chart in middle_charts}
    middle_budget = _two_ended_scattering_middle_budgets(middle_charts)
    atlas = construct_two_ended_scattering_atlas_recurrence(
        masses,
        past_physical_velocities,
        past_offsets,
        future_physical_velocities,
        future_offsets,
        retained_corrections=retained_corrections,
        middle_budgets=middle_budget,
        initial_start_time=start_time,
        max_doublings=0,
    )

    assert {"ordinary_taylor", "separated_binary_levi_civita", "identity_selector_total_collision"} <= middle_kinds
    assert atlas.certified
    assert atlas.past.constants.contraction_factor < 0.01
    assert atlas.future.constants.contraction_factor < 0.01
    assert (
        atlas.past.component_ratios["physical_residual"]
        < atlas.past.component_ratios["residual"]
        < 1.0
    )
    assert (
        atlas.future.component_ratios["physical_residual"]
        < atlas.future.component_ratios["residual"]
        < 1.0
    )
    assert atlas.endpoint_budget("value") < atlas.middle.value
    assert atlas.endpoint_budget("first_jet") < atlas.middle.first_jet
    assert atlas.endpoint_budget("residual") < atlas.middle.residual
    assert atlas.endpoint_budget("physical_residual") < atlas.middle.physical_residual
    for component in atlas.components:
        assert np.isfinite(atlas.total_budget(component))


def test_two_ended_scattering_uses_nonzero_angular_compact_middle_budget():
    masses = np.array([1.0, 0.7, 1.4])
    middle_positions = np.array(
        [
            [0.9, -0.2],
            [-0.4, 0.7],
            [0.1, -0.8],
        ]
    )
    total_mass = np.sum(masses)
    center = np.sum(masses[:, None] * middle_positions, axis=0) / total_mass
    centered = middle_positions - center
    angular_speed = 0.35
    translation_velocity = np.array([0.08, -0.03])
    middle_velocities = translation_velocity + angular_speed * np.column_stack(
        (-centered[:, 1], centered[:, 0])
    )
    exclusion = certify_nonzero_angular_momentum_excludes_triple_collision(
        middle_positions,
        middle_velocities,
        masses,
    )
    past_physical_velocities = np.array(
        [
            [-0.9, 0.25],
            [0.25, 0.95],
            [1.15, -0.5],
        ]
    )
    future_physical_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    past_offsets = np.array(
        [
            [0.05, -0.25],
            [-0.15, 0.08],
            [0.35, 0.18],
        ]
    )
    future_offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    start_time = 1.0e4
    retained_corrections = 3
    middle_charts = (
        {
            "kind": "ordinary_gap_taylor",
            "value": 2.0e-8,
            "first_jet": 7.0e-8,
            "lifted_residual": 1.5e-7,
            "physical_residual": 1.5e-12,
        },
        {
            "kind": "separated_binary_levi_civita",
            "time": 0.4,
            "value": 3.0e-8,
            "first_jet": 9.0e-8,
            "lifted_residual": 2.0e-7,
            "physical_residual": 2.5e-12,
        },
        {
            "kind": "ordinary_gap_taylor",
            "value": 2.5e-8,
            "first_jet": 7.5e-8,
            "lifted_residual": 1.6e-7,
            "physical_residual": 1.6e-12,
        },
        {
            "kind": "separated_binary_levi_civita",
            "time": 1.2,
            "value": 3.5e-8,
            "first_jet": 1.0e-7,
            "lifted_residual": 2.2e-7,
            "physical_residual": 2.7e-12,
        },
        {
            "kind": "ordinary_gap_taylor",
            "value": 2.0e-8,
            "first_jet": 7.0e-8,
            "lifted_residual": 1.5e-7,
            "physical_residual": 1.5e-12,
        },
    )
    middle_kinds = {chart["kind"] for chart in middle_charts}
    middle_budget = _two_ended_scattering_middle_budgets(middle_charts)
    atlas = construct_two_ended_scattering_atlas_recurrence(
        masses,
        past_physical_velocities,
        past_offsets,
        future_physical_velocities,
        future_offsets,
        retained_corrections=retained_corrections,
        middle_budgets=middle_budget,
        initial_start_time=start_time,
        max_doublings=0,
    )
    binary_event_times = [
        chart["time"]
        for chart in middle_charts
        if chart["kind"] == "separated_binary_levi_civita"
    ]

    assert exclusion.certified
    assert exclusion.status == "excluded"
    assert exclusion.angular_momentum_norm_squared_lower_bound > 0.0
    assert middle_kinds == {"ordinary_gap_taylor", "separated_binary_levi_civita"}
    assert "identity_selector_total_collision" not in middle_kinds
    assert len(binary_event_times) == 2
    assert binary_event_times == sorted(binary_event_times)
    assert atlas.certified
    assert atlas.past.constants.contraction_factor < 0.01
    assert atlas.future.constants.contraction_factor < 0.01
    assert (
        atlas.past.component_ratios["physical_residual"]
        < atlas.past.component_ratios["residual"]
        < 1.0
    )
    assert (
        atlas.future.component_ratios["physical_residual"]
        < atlas.future.component_ratios["residual"]
        < 1.0
    )
    assert atlas.endpoint_budget("value") < atlas.middle.value
    assert atlas.endpoint_budget("first_jet") < atlas.middle.first_jet
    assert atlas.endpoint_budget("residual") < atlas.middle.residual
    assert atlas.endpoint_budget("physical_residual") < atlas.middle.physical_residual
    for component in atlas.components:
        assert np.isfinite(atlas.total_budget(component))


def test_two_ended_scattering_uses_automatic_zero_angular_finite_event_middle():
    masses = np.array([1.0, 0.7, 1.4])
    past_physical_velocities = np.array(
        [
            [-0.9, 0.25],
            [0.25, 0.95],
            [1.15, -0.5],
        ]
    )
    future_physical_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    past_offsets = np.zeros_like(past_physical_velocities)
    future_offsets = np.zeros_like(future_physical_velocities)
    start_time = 1.0e4
    retained_corrections = 3
    def scattering_angular_momentum(velocities, offsets):
        return sum(
            masses[index]
            * (
                offsets[index, 0] * velocities[index, 1]
                - offsets[index, 1] * velocities[index, 0]
            )
            for index in range(3)
        )

    middle_charts = (
        {
            "kind": "ordinary_gap_taylor",
            "value": 2.0e-8,
            "first_jet": 7.0e-8,
            "lifted_residual": 1.5e-7,
            "physical_residual": 1.5e-12,
        },
        {
            "kind": "automatic_identity_selector_total_collision",
            "time": -0.6,
            "external_selector_limit_required": False,
            "value": 3.0e-8,
            "first_jet": 9.0e-8,
            "lifted_residual": 2.0e-7,
            "physical_residual": 2.0e-12,
        },
        {
            "kind": "ordinary_gap_taylor",
            "value": 2.2e-8,
            "first_jet": 7.4e-8,
            "lifted_residual": 1.6e-7,
            "physical_residual": 1.6e-12,
        },
        {
            "kind": "separated_binary_levi_civita",
            "time": 0.25,
            "value": 2.8e-8,
            "first_jet": 8.6e-8,
            "lifted_residual": 1.9e-7,
            "physical_residual": 2.4e-12,
        },
        {
            "kind": "ordinary_gap_taylor",
            "value": 2.1e-8,
            "first_jet": 7.2e-8,
            "lifted_residual": 1.55e-7,
            "physical_residual": 1.55e-12,
        },
        {
            "kind": "automatic_identity_selector_total_collision",
            "time": 0.9,
            "external_selector_limit_required": False,
            "value": 3.2e-8,
            "first_jet": 9.4e-8,
            "lifted_residual": 2.1e-7,
            "physical_residual": 2.1e-12,
        },
        {
            "kind": "ordinary_gap_taylor",
            "value": 2.0e-8,
            "first_jet": 7.0e-8,
            "lifted_residual": 1.5e-7,
            "physical_residual": 1.5e-12,
        },
    )
    middle_kinds = {chart["kind"] for chart in middle_charts}
    middle_budget = _two_ended_scattering_middle_budgets(middle_charts)
    atlas = construct_two_ended_scattering_atlas_recurrence(
        masses,
        past_physical_velocities,
        past_offsets,
        future_physical_velocities,
        future_offsets,
        retained_corrections=retained_corrections,
        middle_budgets=middle_budget,
        initial_start_time=start_time,
        max_doublings=0,
    )
    total_collision_times = [
        chart["time"]
        for chart in middle_charts
        if chart["kind"] == "automatic_identity_selector_total_collision"
    ]

    assert abs(
        scattering_angular_momentum(past_physical_velocities, past_offsets)
    ) < 1e-14
    assert abs(
        scattering_angular_momentum(future_physical_velocities, future_offsets)
    ) < 1e-14
    assert middle_kinds == {
        "ordinary_gap_taylor",
        "separated_binary_levi_civita",
        "automatic_identity_selector_total_collision",
    }
    assert total_collision_times == sorted(total_collision_times)
    assert all(
        not chart["external_selector_limit_required"]
        for chart in middle_charts
        if chart["kind"] == "automatic_identity_selector_total_collision"
    )
    assert atlas.certified
    assert atlas.past.constants.contraction_factor < 0.01
    assert atlas.future.constants.contraction_factor < 0.01
    assert (
        atlas.past.component_ratios["physical_residual"]
        < atlas.past.component_ratios["residual"]
        < 1.0
    )
    assert (
        atlas.future.component_ratios["physical_residual"]
        < atlas.future.component_ratios["residual"]
        < 1.0
    )
    assert atlas.endpoint_budget("value") < atlas.middle.value
    assert atlas.endpoint_budget("first_jet") < atlas.middle.first_jet
    assert atlas.endpoint_budget("residual") < atlas.middle.residual
    assert atlas.endpoint_budget("physical_residual") < atlas.middle.physical_residual
    for component in atlas.components:
        assert np.isfinite(atlas.total_budget(component))


def test_geometric_infinite_event_atlas_recurrence_sums_mixed_collision_tail():
    delta_0 = 0.25
    theta = 0.5
    checked_prefix = 7
    generated_shells = 40
    ratios = {
        "value": 0.38,
        "first_jet": 0.42,
        "lifted_residual": 0.35,
        "physical_residual": 0.22,
    }
    first_shell_chart_budgets = (
        {
            "kind": "ordinary_gap_taylor",
            "value": 1.4e-8,
            "first_jet": 3.0e-8,
            "lifted_residual": 8.0e-8,
            "physical_residual": 5.0e-13,
        },
        {
            "kind": "separated_binary_levi_civita",
            "value": 2.0e-8,
            "first_jet": 5.0e-8,
            "lifted_residual": 1.2e-7,
            "physical_residual": 7.5e-13,
        },
        {
            "kind": "identity_selector_total_collision",
            "value": 1.7e-8,
            "first_jet": 4.4e-8,
            "lifted_residual": 1.0e-7,
            "physical_residual": 6.0e-13,
        },
        {
            "kind": "ordinary_gap_taylor",
            "value": 1.1e-8,
            "first_jet": 2.5e-8,
            "lifted_residual": 7.0e-8,
            "physical_residual": 4.0e-13,
        },
    )
    compact_cut = 1.0 - delta_0 * theta**checked_prefix
    intersecting_prefix_shells = [
        n
        for n in range(generated_shells)
        if 1.0 - delta_0 * theta**n < compact_cut
    ]
    event_kinds = {chart["kind"] for chart in first_shell_chart_budgets}
    first_shell_budget = {
        component: sum(chart[component] for chart in first_shell_chart_budgets)
        for component in ratios
    }

    assert {
        "ordinary_gap_taylor",
        "separated_binary_levi_civita",
        "identity_selector_total_collision",
    } <= event_kinds
    assert len(intersecting_prefix_shells) == checked_prefix

    for n in range(generated_shells):
        shell_width = delta_0 * (1.0 - theta) * theta**n
        mixed_event_width = 0.09 * shell_width + 0.06 * shell_width
        boundary_clearance = 0.2 * shell_width
        assert mixed_event_width + 2.0 * boundary_clearance < shell_width

    for component, ratio in ratios.items():
        shell_budgets = [
            first_shell_budget[component] * ratio**n for n in range(generated_shells)
        ]
        infinite_budget = first_shell_budget[component] / (1.0 - ratio)
        prefix_budget = sum(shell_budgets[:checked_prefix])
        recurrence_tail = (
            first_shell_budget[component]
            * ratio**checked_prefix
            / (1.0 - ratio)
        )
        finite_holdout_tail = sum(shell_budgets[checked_prefix:])

        assert ratio < 1.0
        assert prefix_budget + recurrence_tail == pytest.approx(infinite_budget)
        assert finite_holdout_tail <= recurrence_tail * (1.0 + 1.0e-12)
        assert prefix_budget < infinite_budget
        assert sum(shell_budgets) == pytest.approx(infinite_budget)
        assert np.isfinite(prefix_budget + recurrence_tail)


def test_geometric_infinite_event_tail_uses_automatic_total_collision_selectors():
    delta_0 = 0.25
    theta = 0.5
    checked_prefix = 6
    generated_shells = 36
    ratios = {
        "value": 0.36,
        "first_jet": 0.40,
        "lifted_residual": 0.34,
        "physical_residual": 0.20,
    }
    first_shell_chart_budgets = (
        {
            "kind": "ordinary_gap_taylor",
            "value": 1.3e-8,
            "first_jet": 2.8e-8,
            "lifted_residual": 7.5e-8,
            "physical_residual": 4.5e-13,
        },
        {
            "kind": "separated_binary_levi_civita",
            "value": 1.8e-8,
            "first_jet": 4.8e-8,
            "lifted_residual": 1.1e-7,
            "physical_residual": 7.0e-13,
        },
        {
            "kind": "automatic_identity_selector_total_collision",
            "selector_source": "three_body_entry_theorem",
            "external_selector_limit_required": False,
            "value": 1.9e-8,
            "first_jet": 4.9e-8,
            "lifted_residual": 1.1e-7,
            "physical_residual": 6.5e-13,
        },
        {
            "kind": "ordinary_gap_taylor",
            "value": 1.0e-8,
            "first_jet": 2.4e-8,
            "lifted_residual": 6.5e-8,
            "physical_residual": 3.8e-13,
        },
    )
    event_kinds = {chart["kind"] for chart in first_shell_chart_budgets}
    automatic_total_collision_charts = [
        chart
        for chart in first_shell_chart_budgets
        if chart["kind"] == "automatic_identity_selector_total_collision"
    ]
    first_shell_budget = {
        component: sum(chart[component] for chart in first_shell_chart_budgets)
        for component in ratios
    }

    assert {
        "ordinary_gap_taylor",
        "separated_binary_levi_civita",
        "automatic_identity_selector_total_collision",
    } <= event_kinds
    assert "identity_selector_total_collision" not in event_kinds
    assert automatic_total_collision_charts
    assert all(
        chart["selector_source"] == "three_body_entry_theorem"
        for chart in automatic_total_collision_charts
    )
    assert all(
        not chart["external_selector_limit_required"]
        for chart in automatic_total_collision_charts
    )

    for n in range(generated_shells):
        shell_width = delta_0 * (1.0 - theta) * theta**n
        automatic_total_collision_width = 0.08 * shell_width
        separated_binary_width = 0.06 * shell_width
        ordinary_gap_buffer = 0.18 * shell_width
        assert (
            automatic_total_collision_width
            + separated_binary_width
            + 2.0 * ordinary_gap_buffer
            < shell_width
        )

    for component, ratio in ratios.items():
        shell_budgets = [
            first_shell_budget[component] * ratio**n for n in range(generated_shells)
        ]
        infinite_budget = first_shell_budget[component] / (1.0 - ratio)
        prefix_budget = sum(shell_budgets[:checked_prefix])
        recurrence_tail = (
            first_shell_budget[component]
            * ratio**checked_prefix
            / (1.0 - ratio)
        )
        finite_holdout_tail = sum(shell_budgets[checked_prefix:])

        assert ratio < 1.0
        assert prefix_budget + recurrence_tail == pytest.approx(infinite_budget)
        assert finite_holdout_tail <= recurrence_tail * (1.0 + 1.0e-12)
        assert recurrence_tail < first_shell_budget[component] * 0.01
        assert np.isfinite(prefix_budget + recurrence_tail)


def test_local_chart_envelopes_imply_geometric_shell_budget_hypothesis():
    counts = {
        "ordinary_gap_taylor": 4,
        "separated_binary_levi_civita": 2,
        "automatic_identity_selector_total_collision": 1,
    }
    local_envelopes = {
        "ordinary_gap_taylor": {
            "value": (1.1e-8, 0.32),
            "first_jet": (2.5e-8, 0.36),
            "lifted_residual": (6.0e-8, 0.31),
            "physical_residual": (3.0e-13, 0.18),
        },
        "separated_binary_levi_civita": {
            "value": (1.8e-8, 0.35),
            "first_jet": (4.2e-8, 0.38),
            "lifted_residual": (9.5e-8, 0.33),
            "physical_residual": (6.5e-13, 0.20),
        },
        "automatic_identity_selector_total_collision": {
            "value": (2.0e-8, 0.34),
            "first_jet": (4.8e-8, 0.40),
            "lifted_residual": (1.1e-7, 0.34),
            "physical_residual": (7.0e-13, 0.19),
        },
    }
    components = ("value", "first_jet", "lifted_residual", "physical_residual")
    checked_prefix = 8
    generated_shells = 42

    shell_constants = {}
    for component in components:
        component_ratios = [
            local_envelopes[kind][component][1] for kind in local_envelopes
        ]
        ratio = max(component_ratios)
        first_shell_bound = sum(
            counts[kind] * local_envelopes[kind][component][0]
            for kind in local_envelopes
        )
        shell_constants[component] = (first_shell_bound, ratio)

        assert ratio < 1.0
        for kind in local_envelopes:
            assert local_envelopes[kind][component][1] <= ratio

    for component in components:
        first_shell_bound, ratio = shell_constants[component]
        shell_budgets = []
        for n in range(generated_shells):
            local_sum = sum(
                counts[kind]
                * local_envelopes[kind][component][0]
                * local_envelopes[kind][component][1] ** n
                for kind in local_envelopes
            )
            shell_bound = first_shell_bound * ratio**n
            shell_budgets.append(local_sum)

            assert local_sum <= shell_bound * (1.0 + 1e-14)

        prefix_budget = sum(shell_budgets[:checked_prefix])
        recurrence_tail = (
            first_shell_bound * ratio**checked_prefix / (1.0 - ratio)
        )
        infinite_shell_bound = first_shell_bound / (1.0 - ratio)
        finite_holdout_tail = sum(shell_budgets[checked_prefix:])

        assert prefix_budget + recurrence_tail <= infinite_shell_bound * (
            1.0 + 1e-14
        )
        assert finite_holdout_tail <= recurrence_tail * (1.0 + 1e-12)
        assert np.isfinite(prefix_budget + recurrence_tail)


def test_uniform_cauchy_tail_schedule_gives_local_geometric_chart_envelopes():
    counts = {
        "ordinary_gap_taylor": 4,
        "separated_binary_levi_civita": 2,
        "automatic_identity_selector_total_collision": 1,
    }
    cauchy_data = {
        "ordinary_gap_taylor": {
            "value": (1.2e-5, 1.10, 0.32, 5, 3),
            "first_jet": (2.7e-5, 1.12, 0.34, 5, 3),
            "lifted_residual": (5.5e-5, 1.08, 0.30, 6, 3),
            "physical_residual": (3.0e-9, 1.05, 0.24, 6, 3),
        },
        "separated_binary_levi_civita": {
            "value": (1.9e-5, 1.14, 0.35, 5, 3),
            "first_jet": (4.1e-5, 1.15, 0.36, 5, 3),
            "lifted_residual": (8.0e-5, 1.12, 0.33, 6, 3),
            "physical_residual": (5.0e-9, 1.06, 0.25, 6, 3),
        },
        "automatic_identity_selector_total_collision": {
            "value": (2.2e-5, 1.16, 0.34, 5, 3),
            "first_jet": (4.8e-5, 1.16, 0.37, 5, 3),
            "lifted_residual": (9.2e-5, 1.12, 0.34, 6, 3),
            "physical_residual": (5.8e-9, 1.07, 0.26, 6, 3),
        },
    }
    components = ("value", "first_jet", "lifted_residual", "physical_residual")
    generated_shells = 38
    checked_prefix = 9
    local_envelopes = {}

    for kind, component_data in cauchy_data.items():
        local_envelopes[kind] = {}
        for component, (majorant_0, growth, sigma, retained_0, retained_step) in (
            component_data.items()
        ):
            ratio = growth * sigma**retained_step
            first_bound = majorant_0 * sigma ** (retained_0 + 1) / (1.0 - sigma)
            local_envelopes[kind][component] = (first_bound, ratio)

            assert 0.0 < sigma < 1.0
            assert retained_step >= 1
            assert ratio < 1.0

            for shell in range(generated_shells):
                retained_order = retained_0 + retained_step * shell
                actual_majorant = majorant_0 * growth**shell
                actual_ratio = sigma * (0.92 + 0.04 / (shell + 1.0))
                actual_tail = (
                    actual_majorant
                    * actual_ratio ** (retained_order + 1)
                    / (1.0 - actual_ratio)
                )
                geometric_tail = first_bound * ratio**shell

                assert actual_ratio < sigma
                assert actual_tail <= geometric_tail * (1.0 + 1e-13)

    for component in components:
        shell_ratio = max(
            local_envelopes[kind][component][1] for kind in local_envelopes
        )
        shell_first_bound = sum(
            counts[kind] * local_envelopes[kind][component][0]
            for kind in local_envelopes
        )
        shell_budgets = []

        for shell in range(generated_shells):
            shell_budget = sum(
                counts[kind]
                * local_envelopes[kind][component][0]
                * local_envelopes[kind][component][1] ** shell
                for kind in local_envelopes
            )
            shell_budgets.append(shell_budget)

            assert shell_budget <= shell_first_bound * shell_ratio**shell * (
                1.0 + 1e-14
            )

        recurrence_tail = (
            shell_first_bound * shell_ratio**checked_prefix / (1.0 - shell_ratio)
        )
        finite_holdout_tail = sum(shell_budgets[checked_prefix:])
        infinite_shell_bound = shell_first_bound / (1.0 - shell_ratio)

        assert finite_holdout_tail <= recurrence_tail * (1.0 + 1e-12)
        assert sum(shell_budgets[:checked_prefix]) + recurrence_tail <= (
            infinite_shell_bound * (1.0 + 1e-14)
        )
        assert np.isfinite(infinite_shell_bound)


def test_event_recurrence_constructor_derives_all_future_budget_from_primitive_inputs():
    count_certificate = derive_chart_family_counts_from_event_isolation(
        isolation_fraction=0.16,
    )
    primitive = {
        "ordinary_gap_taylor": {
            "value": (0.9e-5, 1.06, 0.30, 5, 3),
            "first_jet": (2.1e-5, 1.08, 0.32, 5, 3),
            "lifted_residual": (4.6e-5, 1.05, 0.28, 6, 3),
            "physical_residual": (2.4e-9, 1.03, 0.22, 6, 3),
        },
        "separated_binary_levi_civita": {
            "value": (1.4e-5, 1.10, 0.33, 5, 3),
            "first_jet": (3.3e-5, 1.12, 0.34, 5, 3),
            "lifted_residual": (6.8e-5, 1.08, 0.31, 6, 3),
            "physical_residual": (4.1e-9, 1.04, 0.23, 6, 3),
        },
        "automatic_identity_selector_total_collision": {
            "value": (1.8e-5, 1.12, 0.32, 5, 3),
            "first_jet": (4.0e-5, 1.13, 0.35, 5, 3),
            "lifted_residual": (7.8e-5, 1.09, 0.32, 6, 3),
            "physical_residual": (4.9e-9, 1.05, 0.24, 6, 3),
        },
    }
    components = ("value", "first_jet", "lifted_residual", "physical_residual")
    checked_prefix = 7
    generated_shells = 36
    event_patterns = (
        (
            "separated_binary_levi_civita",
            "automatic_identity_selector_total_collision",
            "separated_binary_levi_civita",
        ),
        (
            "automatic_identity_selector_total_collision",
            "separated_binary_levi_civita",
        ),
        (
            "separated_binary_levi_civita",
            "separated_binary_levi_civita",
            "automatic_identity_selector_total_collision",
            "automatic_identity_selector_total_collision",
        ),
    )
    certificate = derive_all_future_event_budget_from_primitive_cauchy_inputs(
        counts=count_certificate.chart_family_counts,
        primitive_inputs=primitive,
        checked_prefix=checked_prefix,
        components=components,
    )

    assert count_certificate.certified
    assert count_certificate.event_count_bound == int(np.floor(1.0 / 0.16)) + 2
    assert count_certificate.chart_family_counts["ordinary_gap_taylor"] == (
        count_certificate.event_count_bound + 1
    )
    assert certificate.certified
    assert certificate.recurrence_closes

    shell_budgets_by_component = {component: [] for component in components}
    for shell in range(generated_shells):
        event_types = event_patterns[shell % len(event_patterns)]
        actual_counts = {
            "ordinary_gap_taylor": len(event_types) + 1,
            "separated_binary_levi_civita": event_types.count(
                "separated_binary_levi_civita"
            ),
            "automatic_identity_selector_total_collision": event_types.count(
                "automatic_identity_selector_total_collision"
            ),
        }
        for kind, actual_count in actual_counts.items():
            assert actual_count <= count_certificate.chart_family_counts[kind]

        for component in components:
            actual_shell_budget = 0.0
            for kind, actual_count in actual_counts.items():
                family = certificate.family_budget(kind)
                primitive_input = family.component_inputs[component]
                retained_order = primitive_input.retained_order_for_shell(shell)
                actual_majorant = (
                    primitive_input.majorant_initial
                    * primitive_input.majorant_growth**shell
                    * (0.91 + 0.05 / (shell + 1.0))
                )
                actual_ratio = primitive_input.step_ratio_bound * (
                    0.88 + 0.08 / (shell + 1.0)
                )
                actual_tail = primitive_input.cauchy_tail_bound(
                    shell_index=shell,
                    actual_majorant=actual_majorant,
                    actual_step_ratio=actual_ratio,
                    retained_order=retained_order,
                )

                assert primitive_input.certifies_actual_tail(
                    shell_index=shell,
                    actual_majorant=actual_majorant,
                    actual_step_ratio=actual_ratio,
                    retained_order=retained_order,
                )
                actual_shell_budget += actual_count * actual_tail

            shell_budgets_by_component[component].append(actual_shell_budget)
            assert actual_shell_budget <= certificate.component_family_shell_bound(
                component,
                shell,
            ) * (1.0 + 1e-13)
            assert certificate.component_family_shell_bound(
                component,
                shell,
            ) <= certificate.component_scalar_shell_bound(component, shell) * (
                1.0 + 1e-14
            )

    for component in components:
        actual_holdout_tail = sum(shell_budgets_by_component[component][checked_prefix:])
        actual_prefix = sum(shell_budgets_by_component[component][:checked_prefix])

        assert certificate.component_scalar_ratio(component) < 1.0
        assert actual_holdout_tail <= certificate.component_sharp_family_tail_from_prefix(
            component
        ) * (1.0 + 1e-12)
        assert certificate.component_sharp_family_tail_from_prefix(
            component
        ) <= certificate.component_scalar_tail_from_prefix(component) * (
            1.0 + 1e-12
        )
        assert actual_prefix + certificate.component_scalar_tail_from_prefix(
            component
        ) <= certificate.component_infinite_scalar_bound(component) * (
            1.0 + 1e-12
        )

    with pytest.raises(ValueError, match="does not certify"):
        derive_all_future_event_budget_from_primitive_cauchy_inputs(
            counts=count_certificate.chart_family_counts,
            primitive_inputs={
                **primitive,
                "ordinary_gap_taylor": {
                    **primitive["ordinary_gap_taylor"],
                    "value": (1.0, 2.0, 0.9, 2, 1),
                },
            },
            checked_prefix=checked_prefix,
            components=components,
        )


def test_fuchsian_log_branch_supplies_total_collision_primitive_cauchy_inputs():
    configuration, central_lambda = _central_configuration_data("equilateral")
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    masses = np.ones(3)
    stable_rates = (0.7, 0.9, 1.4, 1.6, 2.1)
    components = ("value", "first_jet", "lifted_residual", "physical_residual")
    chain = construct_stable_log_selector_chain(
        radial_decay=1.0,
        stable_rates=stable_rates,
        source_amplitudes={0: 0.041, 1: -0.029},
        selectors={2: 0.018, 3: -0.016, 4: 0.012},
        resonant_rows={
            2: (StableResonanceTerm(0.73, (2, 0, 0, 0, 0)),),
            3: (StableResonanceTerm(-1.17, (1, 1, 0, 0, 0)),),
            4: (StableResonanceTerm(0.81, (1, 0, 1, 0, 0)),),
        },
    )
    mode_basis = _mass_orthonormal_complement_basis(
        masses,
        central_shape,
        count=len(stable_rates),
    )
    branch = construct_finite_fuchsian_log_branch_from_stable_chain(
        masses=masses,
        central_shape=central_shape,
        scale_coefficient=0.019,
        chain=chain,
        mode_shapes={index: mode_basis[index] for index in range(len(stable_rates))},
    )
    total_collision_inputs = derive_finite_fuchsian_log_branch_primitive_cauchy_inputs(
        branch,
        initial_radius=0.035,
        shell_contraction=0.55,
        analytic_disk_fraction=0.24,
        log_growth_factor=1.12,
        step_ratio_bounds={
            "value": 0.32,
            "first_jet": 0.34,
            "lifted_residual": 0.32,
            "physical_residual": 0.24,
        },
        retained_order_initials={
            "value": 5,
            "first_jet": 5,
            "lifted_residual": 6,
            "physical_residual": 6,
        },
        retained_order_increments={
            "value": 3,
            "first_jet": 3,
            "lifted_residual": 3,
            "physical_residual": 3,
        },
        component_derivative_orders={
            "value": 0,
            "first_jet": 1,
            "lifted_residual": 1,
            "physical_residual": 0,
        },
        component_multipliers={
            "value": 1.0,
            "first_jet": 1.0,
            "lifted_residual": 0.8,
            "physical_residual": 0.15,
        },
    )
    primitive = {
        "ordinary_gap_taylor": {
            "value": (0.9e-5, 1.06, 0.30, 5, 3),
            "first_jet": (2.1e-5, 1.08, 0.32, 5, 3),
            "lifted_residual": (4.6e-5, 1.05, 0.28, 6, 3),
            "physical_residual": (2.4e-9, 1.03, 0.22, 6, 3),
        },
        "separated_binary_levi_civita": {
            "value": (1.4e-5, 1.10, 0.33, 5, 3),
            "first_jet": (3.3e-5, 1.12, 0.34, 5, 3),
            "lifted_residual": (6.8e-5, 1.08, 0.31, 6, 3),
            "physical_residual": (4.1e-9, 1.04, 0.23, 6, 3),
        },
        "automatic_identity_selector_total_collision": {
            component: total_collision_inputs.component_input(component)
            for component in components
        },
    }
    counts = {
        "ordinary_gap_taylor": 4,
        "separated_binary_levi_civita": 2,
        "automatic_identity_selector_total_collision": 1,
    }
    event_budget = derive_all_future_event_budget_from_primitive_cauchy_inputs(
        counts=counts,
        primitive_inputs=primitive,
        checked_prefix=6,
        components=components,
    )

    assert total_collision_inputs.certified
    assert event_budget.recurrence_closes
    assert event_budget.family_budget(
        "automatic_identity_selector_total_collision"
    ).count_bound == 1

    for shell in range(18):
        radius = total_collision_inputs.initial_radius * (
            total_collision_inputs.shell_contraction**shell
        )
        shape, first, _second = branch.radius_derivatives(radius)
        actual_component_majorants = {
            "value": np.max(np.abs(shape)),
            "first_jet": np.max(np.abs(first)),
            "lifted_residual": 0.8 * np.max(np.abs(first)),
            "physical_residual": 0.15 * np.max(np.abs(shape)),
        }
        for component in components:
            primitive_input = total_collision_inputs.component_input(component)
            assert actual_component_majorants[component] <= (
                total_collision_inputs.component_majorant_bound(component, shell)
                * (1.0 + 1e-12)
            )
            assert primitive_input.certifies_actual_tail(
                shell_index=shell,
                actual_majorant=actual_component_majorants[component],
                actual_step_ratio=0.9 * primitive_input.step_ratio_bound,
                retained_order=primitive_input.retained_order_for_shell(shell),
            )
            assert event_budget.component_family_shell_bound(
                component,
                shell,
            ) <= event_budget.component_scalar_shell_bound(component, shell) * (
                1.0 + 1e-12
            )

    with pytest.raises(ValueError, match="does not certify"):
        derive_finite_fuchsian_log_branch_primitive_cauchy_inputs(
            branch,
            initial_radius=0.035,
            shell_contraction=0.55,
            analytic_disk_fraction=0.24,
            log_growth_factor=1.12,
            step_ratio_bounds={"bad_second": 0.9},
            retained_order_initials={"bad_second": 2},
            retained_order_increments={"bad_second": 1},
            component_derivative_orders={"bad_second": 2},
        )


def test_finite_fuchsian_log_branch_certifies_punctured_total_collision_isolation():
    configuration, central_lambda = _central_configuration_data("equilateral")
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    masses = np.ones(3)
    stable_rates = (0.7, 0.9, 1.4, 1.6, 2.1)
    chain = construct_stable_log_selector_chain(
        radial_decay=1.0,
        stable_rates=stable_rates,
        source_amplitudes={0: 0.041, 1: -0.029},
        selectors={2: 0.018, 3: -0.016, 4: 0.012},
        resonant_rows={
            2: (StableResonanceTerm(0.73, (2, 0, 0, 0, 0)),),
            3: (StableResonanceTerm(-1.17, (1, 1, 0, 0, 0)),),
            4: (StableResonanceTerm(0.81, (1, 0, 1, 0, 0)),),
        },
    )
    mode_basis = _mass_orthonormal_complement_basis(
        masses,
        central_shape,
        count=len(stable_rates),
    )
    branch = construct_finite_fuchsian_log_branch_from_stable_chain(
        masses=masses,
        central_shape=central_shape,
        scale_coefficient=0.019,
        chain=chain,
        mode_shapes={index: mode_basis[index] for index in range(len(stable_rates))},
    )

    isolation = certify_finite_fuchsian_log_total_collision_isolation(
        branch,
        radius=0.035,
    )

    assert isolation.certified
    assert isolation.shape_deviation_bound < isolation.central_shape_pair_distance_floor / 4.0
    assert isolation.shape_pair_distance_floor > 0.0
    for tau in (-0.03, -0.01, -0.001, 0.001, 0.01, 0.03):
        shape, _first, _second = branch.tau_derivatives(tau)
        positions = branch.positions_at_tau(tau)
        shape_pair_floor = min(
            np.linalg.norm(shape[j] - shape[i])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        physical_pair_floor = min(
            np.linalg.norm(positions[j] - positions[i])
            for i in range(3)
            for j in range(i + 1, 3)
        )

        assert shape_pair_floor >= isolation.shape_pair_distance_floor * (1.0 - 1e-12)
        assert physical_pair_floor >= isolation.physical_pair_distance_floor(tau) * (
            1.0 - 1e-12
        )
        assert physical_pair_floor > 0.0

    large_shape_mode = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 0.0],
        ]
    )
    nonisolated = FiniteFuchsianLogBranch(
        masses=masses,
        central_shape=central_shape,
        scale_coefficient=0.019,
        terms=(
            FuchsianLogTerm(
                power=0.2,
                coefficients_by_log_power={0: 3.0 * large_shape_mode},
            ),
        ),
    )
    with pytest.raises(ValueError, match="not isolated"):
        certify_finite_fuchsian_log_total_collision_isolation(
            nonisolated,
            radius=0.035,
        )


def test_fuchsian_log_total_collision_isolation_projects_to_compact_time_shell():
    configuration, central_lambda = _central_configuration_data("equilateral")
    central_shape = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0) * configuration
    masses = np.ones(3)
    stable_rates = (0.7, 0.9, 1.4, 1.6, 2.1)
    chain = construct_stable_log_selector_chain(
        radial_decay=1.0,
        stable_rates=stable_rates,
        source_amplitudes={0: 0.041, 1: -0.029},
        selectors={2: 0.018, 3: -0.016, 4: 0.012},
        resonant_rows={
            2: (StableResonanceTerm(0.73, (2, 0, 0, 0, 0)),),
            3: (StableResonanceTerm(-1.17, (1, 1, 0, 0, 0)),),
            4: (StableResonanceTerm(0.81, (1, 0, 1, 0, 0)),),
        },
    )
    mode_basis = _mass_orthonormal_complement_basis(
        masses,
        central_shape,
        count=len(stable_rates),
    )
    branch = construct_finite_fuchsian_log_branch_from_stable_chain(
        masses=masses,
        central_shape=central_shape,
        scale_coefficient=0.019,
        chain=chain,
        mode_shapes={index: mode_basis[index] for index in range(len(stable_rates))},
    )
    tau_isolation = certify_finite_fuchsian_log_total_collision_isolation(
        branch,
        radius=0.035,
    )
    event_time = 0.4
    compact_rate = 1.3
    event_u = np.tanh(compact_rate * event_time)
    projected_radius_without_shell = certify_finite_fuchsian_log_compact_time_isolation(
        tau_isolation,
        event_physical_time=event_time,
        compact_time_rate=compact_rate,
    ).compact_isolation_radius
    shell_width = projected_radius_without_shell / 0.24
    compact_isolation = certify_finite_fuchsian_log_compact_time_isolation(
        tau_isolation,
        event_physical_time=event_time,
        compact_time_rate=compact_rate,
        shell_interval=(event_u - 0.5 * shell_width, event_u + 0.5 * shell_width),
    )
    count_certificate = derive_chart_family_counts_from_event_isolation(
        isolation_fraction=compact_isolation.shell_isolation_fraction,
    )

    assert compact_isolation.certified
    assert compact_isolation.compact_lower < event_u < compact_isolation.compact_upper
    assert compact_isolation.shell_isolation_fraction == pytest.approx(0.24, rel=1e-12)
    assert count_certificate.certified
    assert count_certificate.event_count_bound == int(
        np.floor(1.0 / compact_isolation.shell_isolation_fraction)
    ) + 2

    for fraction in (-0.9, -0.25, 0.25, 0.9):
        compact_u = event_u + fraction * compact_isolation.compact_isolation_radius
        physical_time = np.arctanh(compact_u) / compact_rate
        tau = np.cbrt(physical_time - event_time)

        assert 0.0 < abs(tau) < tau_isolation.radius
        assert tau_isolation.physical_pair_distance_floor(tau) > 0.0

    with pytest.raises(ValueError, match="inside the shell"):
        certify_finite_fuchsian_log_compact_time_isolation(
            tau_isolation,
            event_physical_time=event_time,
            compact_time_rate=compact_rate,
            shell_interval=(event_u + shell_width, event_u + 2.0 * shell_width),
        )


def test_primitive_cauchy_inputs_close_all_future_event_budget():
    counts = {
        "ordinary_gap_taylor": 5,
        "separated_binary_levi_civita": 2,
        "automatic_identity_selector_total_collision": 1,
    }
    primitive = {
        "ordinary_gap_taylor": {
            "value": (1.0e-5, 1.08, 0.31, 5, 3),
            "first_jet": (2.5e-5, 1.10, 0.33, 5, 3),
            "lifted_residual": (5.0e-5, 1.06, 0.29, 6, 3),
            "physical_residual": (2.8e-9, 1.04, 0.23, 6, 3),
        },
        "separated_binary_levi_civita": {
            "value": (1.7e-5, 1.12, 0.34, 5, 3),
            "first_jet": (3.9e-5, 1.13, 0.35, 5, 3),
            "lifted_residual": (7.6e-5, 1.10, 0.32, 6, 3),
            "physical_residual": (4.8e-9, 1.05, 0.24, 6, 3),
        },
        "automatic_identity_selector_total_collision": {
            "value": (2.0e-5, 1.14, 0.33, 5, 3),
            "first_jet": (4.5e-5, 1.15, 0.36, 5, 3),
            "lifted_residual": (8.8e-5, 1.11, 0.33, 6, 3),
            "physical_residual": (5.6e-9, 1.06, 0.25, 6, 3),
        },
    }
    components = ("value", "first_jet", "lifted_residual", "physical_residual")
    checked_prefix = 7
    generated_shells = 45

    for component in components:
        local_constants = {}
        for kind, data in primitive.items():
            majorant_0, growth, sigma, retained_0, retained_step = data[component]
            local_constant = (
                majorant_0 * sigma ** (retained_0 + 1) / (1.0 - sigma)
            )
            local_ratio = growth * sigma**retained_step
            local_constants[kind] = (local_constant, local_ratio)

            assert 0.0 < sigma < 1.0
            assert local_ratio < 1.0

        scalar_first_bound = sum(
            counts[kind] * local_constants[kind][0] for kind in local_constants
        )
        scalar_ratio = max(local_constants[kind][1] for kind in local_constants)

        sharp_family_tail = sum(
            counts[kind]
            * local_constants[kind][0]
            * local_constants[kind][1] ** checked_prefix
            / (1.0 - local_constants[kind][1])
            for kind in local_constants
        )
        scalar_tail = (
            scalar_first_bound * scalar_ratio**checked_prefix / (1.0 - scalar_ratio)
        )
        shell_budgets = []

        for shell in range(generated_shells):
            actual_shell_budget = 0.0
            shell_bound = 0.0
            for kind, data in primitive.items():
                majorant_0, growth, sigma, retained_0, retained_step = data[component]
                actual_count = counts[kind] - (1 if shell % 3 == 2 else 0)
                retained_order = retained_0 + retained_step * shell
                actual_majorant = majorant_0 * growth**shell
                actual_ratio = sigma * (0.9 + 0.06 / (shell + 1.0))
                actual_chart_tail = (
                    actual_majorant
                    * actual_ratio ** (retained_order + 1)
                    / (1.0 - actual_ratio)
                )
                local_constant, local_ratio = local_constants[kind]

                assert 0 <= actual_count <= counts[kind]
                assert actual_ratio < sigma
                assert actual_chart_tail <= (
                    local_constant * local_ratio**shell * (1.0 + 1e-13)
                )

                actual_shell_budget += actual_count * actual_chart_tail
                shell_bound += counts[kind] * local_constant * local_ratio**shell

            shell_budgets.append(actual_shell_budget)

            assert actual_shell_budget <= shell_bound * (1.0 + 1e-13)
            assert shell_bound <= (
                scalar_first_bound * scalar_ratio**shell * (1.0 + 1e-14)
            )

        actual_holdout_tail = sum(shell_budgets[checked_prefix:])
        scalar_infinite_bound = scalar_first_bound / (1.0 - scalar_ratio)

        assert actual_holdout_tail <= sharp_family_tail * (1.0 + 1e-12)
        assert sharp_family_tail <= scalar_tail * (1.0 + 1e-12)
        assert sum(shell_budgets[:checked_prefix]) + scalar_tail <= (
            scalar_infinite_bound * (1.0 + 1e-14)
        )
        assert np.isfinite(scalar_infinite_bound)


def test_uniform_event_isolation_supplies_chart_counts_for_primitive_budget():
    alpha = 0.16
    event_count_bound = int(np.floor(1.0 / alpha)) + 2
    derived_counts = {
        "ordinary_gap_taylor": event_count_bound + 1,
        "separated_binary_levi_civita": event_count_bound,
        "automatic_identity_selector_total_collision": event_count_bound,
    }
    primitive = {
        "ordinary_gap_taylor": {
            "value": (0.9e-5, 1.06, 0.30, 5, 3),
            "first_jet": (2.1e-5, 1.08, 0.32, 5, 3),
            "lifted_residual": (4.6e-5, 1.05, 0.28, 6, 3),
            "physical_residual": (2.4e-9, 1.03, 0.22, 6, 3),
        },
        "separated_binary_levi_civita": {
            "value": (1.4e-5, 1.10, 0.33, 5, 3),
            "first_jet": (3.3e-5, 1.12, 0.34, 5, 3),
            "lifted_residual": (6.8e-5, 1.08, 0.31, 6, 3),
            "physical_residual": (4.1e-9, 1.04, 0.23, 6, 3),
        },
        "automatic_identity_selector_total_collision": {
            "value": (1.8e-5, 1.12, 0.32, 5, 3),
            "first_jet": (4.0e-5, 1.13, 0.35, 5, 3),
            "lifted_residual": (7.8e-5, 1.09, 0.32, 6, 3),
            "physical_residual": (4.9e-9, 1.05, 0.24, 6, 3),
        },
    }
    delta_0 = 0.24
    theta = 0.5
    generated_shells = 34
    checked_prefix = 6
    components = ("value", "first_jet", "lifted_residual", "physical_residual")

    event_patterns = (
        (
            "separated_binary_levi_civita",
            "automatic_identity_selector_total_collision",
            "separated_binary_levi_civita",
            "automatic_identity_selector_total_collision",
        ),
        (
            "automatic_identity_selector_total_collision",
            "separated_binary_levi_civita",
            "separated_binary_levi_civita",
        ),
        (
            "separated_binary_levi_civita",
            "separated_binary_levi_civita",
            "automatic_identity_selector_total_collision",
            "separated_binary_levi_civita",
            "automatic_identity_selector_total_collision",
        ),
    )

    actual_counts_by_shell = []
    for shell in range(generated_shells):
        left = 1.0 - delta_0 * theta**shell
        right = 1.0 - delta_0 * theta ** (shell + 1)
        shell_width = right - left
        isolation_radius = alpha * shell_width
        event_types = event_patterns[shell % len(event_patterns)]
        centers = [
            left + (index + 1.0) * shell_width / (len(event_types) + 1.0)
            for index in range(len(event_types))
        ]

        assert len(centers) <= event_count_bound
        assert centers[0] - left >= isolation_radius
        assert right - centers[-1] >= isolation_radius
        for a, b in zip(centers, centers[1:]):
            assert b - a >= isolation_radius

        type_counts = {
            "ordinary_gap_taylor": len(centers) + 1,
            "separated_binary_levi_civita": event_types.count(
                "separated_binary_levi_civita"
            ),
            "automatic_identity_selector_total_collision": event_types.count(
                "automatic_identity_selector_total_collision"
            ),
        }
        actual_counts_by_shell.append(type_counts)

        for kind, actual_count in type_counts.items():
            assert actual_count <= derived_counts[kind]

    assert derived_counts["ordinary_gap_taylor"] == event_count_bound + 1
    assert derived_counts["separated_binary_levi_civita"] == event_count_bound
    assert (
        derived_counts["automatic_identity_selector_total_collision"]
        == event_count_bound
    )

    for component in components:
        local_constants = {}
        for kind, data in primitive.items():
            majorant_0, growth, sigma, retained_0, retained_step = data[component]
            first_bound = (
                majorant_0 * sigma ** (retained_0 + 1) / (1.0 - sigma)
            )
            ratio = growth * sigma**retained_step
            local_constants[kind] = (first_bound, ratio)

            assert ratio < 1.0

        scalar_first_bound = sum(
            derived_counts[kind] * local_constants[kind][0]
            for kind in local_constants
        )
        scalar_ratio = max(local_constants[kind][1] for kind in local_constants)
        shell_budgets = []

        for shell, actual_counts in enumerate(actual_counts_by_shell):
            shell_budget = 0.0
            for kind, actual_count in actual_counts.items():
                first_bound, ratio = local_constants[kind]
                shell_budget += actual_count * first_bound * ratio**shell
            shell_budgets.append(shell_budget)

            assert shell_budget <= (
                scalar_first_bound * scalar_ratio**shell * (1.0 + 1e-14)
            )

        recurrence_tail = (
            scalar_first_bound * scalar_ratio**checked_prefix / (1.0 - scalar_ratio)
        )
        finite_holdout_tail = sum(shell_budgets[checked_prefix:])

        assert finite_holdout_tail <= recurrence_tail * (1.0 + 1e-12)
        assert np.isfinite(recurrence_tail)


def test_uniform_noncollision_state_envelope_supplies_ordinary_cauchy_inputs():
    mass_total = 3.4
    distance_floor = 0.72
    diameter_bound = 2.8
    speed_bound = 1.35
    sundman_distance_power = 1.0
    rho = distance_floor / 10.0
    g_bound = (diameter_bound + distance_floor / 5.0) ** (
        3.0 * sundman_distance_power
    )
    acceleration_bound = (
        mass_total
        * (diameter_bound + distance_floor / 5.0)
        / (((14.0 / 25.0) ** 1.5) * distance_floor**3)
    )
    velocity_radius_bound = np.sqrt(acceleration_bound * rho)
    sundman_radius_floor = min(
        rho / (g_bound * (speed_bound + velocity_radius_bound)),
        np.sqrt(rho / acceleration_bound) / g_bound,
    )
    sigma = 0.42
    retained_0 = 6
    retained_step = 3
    ordinary_count_bound = 9
    checked_prefix = 8
    generated_shells = 40
    component_majorants = {
        "value": diameter_bound + rho,
        "first_jet": g_bound * (speed_bound + velocity_radius_bound),
        "lifted_residual": g_bound * acceleration_bound,
        "physical_residual": 0.5 * g_bound * acceleration_bound,
    }

    assert sundman_radius_floor > 0.0
    assert 0.0 < sigma < 1.0

    local_ratio = sigma**retained_step
    assert local_ratio < 1.0

    for component, majorant_0 in component_majorants.items():
        first_bound = majorant_0 * sigma ** (retained_0 + 1) / (1.0 - sigma)
        shell_budgets = []

        for shell in range(generated_shells):
            retained_order = retained_0 + retained_step * shell
            actual_radius = sundman_radius_floor * (1.0 + 0.02 / (shell + 1.0))
            step = sigma * sundman_radius_floor * (0.91 + 0.03 / (shell + 1.0))
            actual_ratio = step / actual_radius
            actual_majorant = majorant_0 * (0.94 + 0.04 / (shell + 1.0))
            actual_tail = (
                actual_majorant
                * actual_ratio ** (retained_order + 1)
                / (1.0 - actual_ratio)
            )
            local_tail_bound = first_bound * local_ratio**shell

            assert actual_radius >= sundman_radius_floor
            assert actual_ratio < sigma
            assert actual_majorant <= majorant_0
            assert actual_tail <= local_tail_bound * (1.0 + 1e-13)

            shell_budgets.append(ordinary_count_bound * actual_tail)

        scalar_first_bound = ordinary_count_bound * first_bound
        recurrence_tail = (
            scalar_first_bound * local_ratio**checked_prefix / (1.0 - local_ratio)
        )
        finite_holdout_tail = sum(shell_budgets[checked_prefix:])
        infinite_bound = scalar_first_bound / (1.0 - local_ratio)

        assert finite_holdout_tail <= recurrence_tail * (1.0 + 1e-12)
        assert sum(shell_budgets[:checked_prefix]) + recurrence_tail <= (
            infinite_bound * (1.0 + 1e-14)
        )
        assert np.isfinite(infinite_bound), component


def test_uniform_separated_binary_envelope_supplies_levi_civita_cauchy_inputs():
    masses = np.array([1.1, 0.9, 1.4])
    pair_mass = masses[0] + masses[1]
    alpha = masses[1] / pair_mass
    beta = masses[0] / pair_mass
    z_bound = 0.36
    z_velocity_bound = 1.25
    pair_energy_bound = 1.4
    binary_center_bound = 2.6
    binary_center_velocity_bound = 0.75
    third_offset_bound = 2.4
    third_offset_velocity_bound = 0.85
    radii = {
        "z": 0.055,
        "z_velocity": 0.080,
        "pair_energy": 0.100,
        "binary_center": 0.120,
        "binary_center_velocity": 0.070,
        "third_offset": 0.110,
        "third_offset_velocity": 0.070,
    }
    third_body_nominal_floor = 1.55
    relative_variation_bound = 2.0 * z_bound * radii["z"] + radii["z"] ** 2
    third_distance_floor = min(
        third_body_nominal_floor
        - radii["third_offset"]
        - alpha * relative_variation_bound,
        third_body_nominal_floor
        - radii["third_offset"]
        - beta * relative_variation_bound,
    )
    z_majorant = z_bound + radii["z"]
    z_velocity_majorant = z_velocity_bound + radii["z_velocity"]
    pair_energy_majorant = pair_energy_bound + radii["pair_energy"]
    center_velocity_majorant = (
        binary_center_velocity_bound + radii["binary_center_velocity"]
    )
    third_offset_velocity_majorant = (
        third_offset_velocity_bound + radii["third_offset_velocity"]
    )
    third_offset_upper = (
        third_offset_bound
        + radii["third_offset"]
        + max(alpha, beta) * z_majorant**2
    )
    perturbation_bound = float(np.sum(masses) * third_offset_upper / third_distance_floor**3)
    center_acceleration_bound = masses[2] * perturbation_bound
    third_offset_acceleration_bound = float(np.sum(masses) * perturbation_bound)
    rho_bound = z_majorant**2
    rhs_bounds = {
        "z": z_velocity_majorant,
        "z_velocity": 0.5 * pair_energy_majorant * z_majorant
        + 0.5 * rho_bound * z_majorant * perturbation_bound,
        "pair_energy": 2.0 * z_majorant * z_velocity_majorant * perturbation_bound,
        "binary_center": rho_bound * center_velocity_majorant,
        "binary_center_velocity": rho_bound * center_acceleration_bound,
        "third_offset": rho_bound * third_offset_velocity_majorant,
        "third_offset_velocity": rho_bound * third_offset_acceleration_bound,
    }
    radius_candidates = [
        radii[name] / bound
        for name, bound in rhs_bounds.items()
        if bound > 0.0
    ]
    levi_civita_radius_floor = min(radius_candidates)
    state_sup_bound = max(
        z_majorant,
        z_velocity_majorant,
        pair_energy_majorant,
        binary_center_bound + radii["binary_center"],
        center_velocity_majorant,
        third_offset_bound + radii["third_offset"],
        third_offset_velocity_majorant,
        levi_civita_radius_floor * rho_bound,
    )
    sigma = 0.38
    retained_0 = 5
    retained_step = 3
    binary_count_bound = 7
    checked_prefix = 7
    generated_shells = 38
    component_majorants = {
        "value": state_sup_bound,
        "first_jet": max(rhs_bounds.values()),
        "lifted_residual": 0.8 * max(rhs_bounds.values()),
        "physical_residual": 0.5 * max(rhs_bounds.values()),
    }

    assert third_distance_floor > 0.0
    assert perturbation_bound > 0.0
    assert levi_civita_radius_floor > 0.0

    local_ratio = sigma**retained_step
    assert local_ratio < 1.0

    for component, majorant_0 in component_majorants.items():
        first_bound = majorant_0 * sigma ** (retained_0 + 1) / (1.0 - sigma)
        shell_budgets = []

        for shell in range(generated_shells):
            retained_order = retained_0 + retained_step * shell
            actual_radius = levi_civita_radius_floor * (1.0 + 0.03 / (shell + 1.0))
            step = sigma * levi_civita_radius_floor * (0.90 + 0.04 / (shell + 1.0))
            actual_ratio = step / actual_radius
            actual_majorant = majorant_0 * (0.93 + 0.05 / (shell + 1.0))
            actual_tail = (
                actual_majorant
                * actual_ratio ** (retained_order + 1)
                / (1.0 - actual_ratio)
            )
            local_tail_bound = first_bound * local_ratio**shell

            assert actual_radius >= levi_civita_radius_floor
            assert actual_ratio < sigma
            assert actual_majorant <= majorant_0
            assert actual_tail <= local_tail_bound * (1.0 + 1e-13)

            shell_budgets.append(binary_count_bound * actual_tail)

        scalar_first_bound = binary_count_bound * first_bound
        recurrence_tail = (
            scalar_first_bound * local_ratio**checked_prefix / (1.0 - local_ratio)
        )
        finite_holdout_tail = sum(shell_budgets[checked_prefix:])
        infinite_bound = scalar_first_bound / (1.0 - local_ratio)

        assert finite_holdout_tail <= recurrence_tail * (1.0 + 1e-12)
        assert sum(shell_budgets[:checked_prefix]) + recurrence_tail <= (
            infinite_bound * (1.0 + 1e-14)
        )
        assert np.isfinite(infinite_bound), component

    constructor = certify_uniform_separated_binary_levi_civita_chart_family(
        masses=masses,
        pair=(0, 1),
        z_bound=z_bound,
        z_velocity_bound=z_velocity_bound,
        pair_energy_bound=pair_energy_bound,
        binary_center_bound=binary_center_bound,
        binary_center_velocity_bound=binary_center_velocity_bound,
        third_offset_bound=third_offset_bound,
        third_offset_velocity_bound=third_offset_velocity_bound,
        third_body_nominal_distance_lower_bound=third_body_nominal_floor,
        radii=radii,
        step_ratio_bounds={
            "value": sigma,
            "first_jet": sigma,
            "lifted_residual": sigma,
            "physical_residual": sigma,
        },
        retained_order_initials={
            "value": retained_0,
            "first_jet": retained_0,
            "lifted_residual": retained_0,
            "physical_residual": retained_0,
        },
        retained_order_increments={
            "value": retained_step,
            "first_jet": retained_step,
            "lifted_residual": retained_step,
            "physical_residual": retained_step,
        },
        component_majorant_multipliers={
            "lifted_residual": 0.8,
            "physical_residual": 0.5 / rho_bound,
        },
        components=tuple(component_majorants),
    )
    source = constructor.source_certificate

    assert constructor.certified
    assert constructor.source == "uniform_separated_binary_levi_civita_chart_family_constructor"
    assert source.third_distance_floor == pytest.approx(third_distance_floor)
    assert source.perturbation_bound == pytest.approx(perturbation_bound)
    assert source.levi_civita_radius_floor == pytest.approx(levi_civita_radius_floor)
    assert source.state_sup_bound == pytest.approx(state_sup_bound)
    for component, majorant in component_majorants.items():
        primitive_input = constructor.component_input(component)
        assert source.component_majorant(component) == pytest.approx(majorant)
        assert primitive_input.majorant_initial == pytest.approx(majorant)
        assert primitive_input.majorant_growth == pytest.approx(1.0)
        assert primitive_input.step_ratio_bound == pytest.approx(sigma)
        assert primitive_input.retained_order_initial == retained_0
        assert primitive_input.retained_order_increment == retained_step
        assert primitive_input.certifies_actual_tail(
            shell_index=checked_prefix,
            actual_majorant=majorant,
            actual_step_ratio=sigma,
        )


def test_uniform_event_isolation_gives_geometric_shell_count_bound():
    delta_0 = 0.2
    theta = 0.5
    alpha = 0.18
    shell_count_bound = int(np.floor(1.0 / alpha)) + 2
    per_event_budget = {
        "value": 3.0e-9,
        "first_jet": 8.0e-9,
        "lifted_residual": 1.5e-8,
        "physical_residual": 2.0e-13,
    }
    per_gap_budget = {
        "value": 1.0e-9,
        "first_jet": 3.0e-9,
        "lifted_residual": 5.0e-9,
        "physical_residual": 8.0e-14,
    }
    shell_ratio = 0.44
    checked_prefix = 6
    event_centers_by_shell = []

    for n in range(checked_prefix):
        left = 1.0 - delta_0 * theta**n
        right = 1.0 - delta_0 * theta ** (n + 1)
        shell_width = right - left
        isolation_radius = alpha * shell_width
        # Pack fewer than the boundary-inclusive upper bound, with gaps wider
        # than the lower isolation scale.
        centers = [left + fraction * shell_width for fraction in (0.2, 0.4, 0.6, 0.8)]
        event_centers_by_shell.append(centers)

        assert centers[0] - left >= isolation_radius
        assert right - centers[-1] >= isolation_radius
        for a, b in zip(centers, centers[1:]):
            assert b - a >= isolation_radius
        assert len(centers) <= shell_count_bound
        assert len(centers) * alpha * shell_width <= shell_width

    first_shell_event_count = max(len(centers) for centers in event_centers_by_shell)
    assert first_shell_event_count == 4
    assert first_shell_event_count <= shell_count_bound

    for component in per_event_budget:
        first_shell_budget = (
            shell_count_bound * per_event_budget[component]
            + (shell_count_bound + 1) * per_gap_budget[component]
        )
        prefix_budget = sum(
            first_shell_budget * shell_ratio**n for n in range(checked_prefix)
        )
        recurrence_tail = (
            first_shell_budget
            * shell_ratio**checked_prefix
            / (1.0 - shell_ratio)
        )
        infinite_budget = first_shell_budget / (1.0 - shell_ratio)

        assert prefix_budget + recurrence_tail == pytest.approx(infinite_budget)
        assert recurrence_tail < first_shell_budget * 0.02
        assert np.isfinite(infinite_budget)


def test_geometric_shell_isolation_constructor_closes_all_future_event_budget():
    components = ("value", "first_jet", "lifted_residual", "physical_residual")
    shell_isolation = derive_geometric_shell_event_isolation(
        delta_initial=0.2,
        theta=0.5,
        event_isolation_initial=0.018,
        boundary_clearance_initial=0.02,
    )
    primitive = {
        "ordinary_gap_taylor": {
            "value": (0.9e-5, 1.06, 0.30, 5, 3),
            "first_jet": (2.1e-5, 1.08, 0.32, 5, 3),
            "lifted_residual": (4.6e-5, 1.05, 0.28, 6, 3),
            "physical_residual": (2.4e-9, 1.03, 0.22, 6, 3),
        },
        "separated_binary_levi_civita": {
            "value": (1.4e-5, 1.10, 0.33, 5, 3),
            "first_jet": (3.3e-5, 1.12, 0.34, 5, 3),
            "lifted_residual": (6.8e-5, 1.08, 0.31, 6, 3),
            "physical_residual": (4.1e-9, 1.04, 0.23, 6, 3),
        },
        "automatic_identity_selector_total_collision": {
            "value": (1.8e-5, 1.12, 0.32, 5, 3),
            "first_jet": (4.0e-5, 1.13, 0.35, 5, 3),
            "lifted_residual": (7.8e-5, 1.09, 0.32, 6, 3),
            "physical_residual": (4.9e-9, 1.05, 0.24, 6, 3),
        },
    }
    certificate = derive_all_future_event_budget_from_geometric_shell_isolation(
        shell_isolation=shell_isolation,
        primitive_inputs=primitive,
        checked_prefix=6,
        components=components,
    )

    assert shell_isolation.certified
    assert shell_isolation.isolation_fraction == pytest.approx(0.18)
    assert shell_isolation.event_count_bound == int(np.floor(1.0 / 0.18)) + 2
    assert shell_isolation.chart_family_counts == {
        "ordinary_gap_taylor": shell_isolation.event_count_bound + 1,
        "separated_binary_levi_civita": shell_isolation.event_count_bound,
        "automatic_identity_selector_total_collision": shell_isolation.event_count_bound,
    }
    assert certificate.certified
    assert certificate.recurrence_closes

    event_patterns = (
        (
            "separated_binary_levi_civita",
            "automatic_identity_selector_total_collision",
            "separated_binary_levi_civita",
            "automatic_identity_selector_total_collision",
        ),
        (
            "automatic_identity_selector_total_collision",
            "separated_binary_levi_civita",
            "separated_binary_levi_civita",
        ),
    )
    generated_shells = 12
    shell_budgets = {component: [] for component in components}
    for shell in range(generated_shells):
        left, right = shell_isolation.shell_interval(shell)
        width = right - left
        event_types = event_patterns[shell % len(event_patterns)]
        centers = tuple(
            left + (index + 1.0) * width / (len(event_types) + 1.0)
            for index in range(len(event_types))
        )
        dense_centers = tuple(left + fraction * width for fraction in (0.2, 0.25))
        actual_counts = shell_isolation.observed_chart_family_counts(event_types)

        assert shell_isolation.certifies_observed_shell(
            shell_index=shell,
            event_centers=centers,
        )
        assert shell_isolation.certifies_observed_typed_shell(
            shell_index=shell,
            event_centers=centers,
            event_types=event_types,
        )
        assert not shell_isolation.certifies_observed_shell(
            shell_index=shell,
            event_centers=dense_centers,
        )
        assert not shell_isolation.certifies_observed_typed_shell(
            shell_index=shell,
            event_centers=centers,
            event_types=("ordinary_gap_taylor", *event_types),
        )
        assert not shell_isolation.certifies_observed_typed_shell(
            shell_index=shell,
            event_centers=centers[:-1],
            event_types=event_types,
        )
        assert not shell_isolation.certifies_observed_typed_shell(
            shell_index=shell,
            event_centers=centers,
            event_types=("unsupported_chart_family", *event_types[1:]),
        )
        for kind, actual_count in actual_counts.items():
            assert actual_count <= shell_isolation.chart_family_counts[kind]

        for component in components:
            actual_shell_budget = 0.0
            for kind, actual_count in actual_counts.items():
                primitive_input = certificate.event_budget.family_budget(
                    kind
                ).component_inputs[component]
                actual_majorant = (
                    primitive_input.majorant_initial
                    * primitive_input.majorant_growth**shell
                    * (0.92 + 0.04 / (shell + 1.0))
                )
                actual_ratio = primitive_input.step_ratio_bound * (
                    0.90 + 0.06 / (shell + 1.0)
                )
                actual_tail = primitive_input.cauchy_tail_bound(
                    shell_index=shell,
                    actual_majorant=actual_majorant,
                    actual_step_ratio=actual_ratio,
                )

                assert primitive_input.certifies_actual_tail(
                    shell_index=shell,
                    actual_majorant=actual_majorant,
                    actual_step_ratio=actual_ratio,
                )
                actual_shell_budget += actual_count * actual_tail

            shell_budgets[component].append(actual_shell_budget)
            assert actual_shell_budget <= (
                certificate.event_budget.component_scalar_shell_bound(
                    component,
                    shell,
                )
                * (1.0 + 1e-13)
            )

    for component in components:
        actual_holdout = sum(
            shell_budgets[component][certificate.event_budget.checked_prefix:]
        )
        assert actual_holdout <= (
            certificate.event_budget.component_scalar_tail_from_prefix(component)
            * (1.0 + 1e-12)
        )

    with pytest.raises(ValueError, match="geometric shell isolation"):
        derive_geometric_shell_event_isolation(
            delta_initial=0.2,
            theta=1.0,
            event_isolation_initial=0.018,
            boundary_clearance_initial=0.02,
        )


def test_compact_event_accumulation_forces_isolation_degeneracy():
    compact_left = 0.35
    compact_right = 0.75
    accumulation_point = 0.6
    uniform_isolation_radius = 0.01
    finite_count_bound = int(
        np.floor((compact_right - compact_left) / uniform_isolation_radius)
    ) + 2
    centers = [
        accumulation_point - 0.1 / (k + 2.0)
        for k in range(2 * finite_count_bound)
    ]
    local_isolation_radii = [
        0.45 * min(center - prev_center, accumulation_point - center)
        for prev_center, center in zip([compact_left] + centers[:-1], centers)
    ]
    binary_chart_constants = [
        {
            "kind": "separated_binary_levi_civita",
            "zeta_norm_squared": 0.5 * (1.0 + 0.7),
            "third_body_separation_floor": radius,
            "analytic_radius": radius,
        }
        for radius in local_isolation_radii
    ]

    assert len(centers) > finite_count_bound
    assert all(compact_left < center < compact_right for center in centers)
    assert all(center < accumulation_point for center in centers)
    assert min(chart["zeta_norm_squared"] for chart in binary_chart_constants) > 0.0
    assert min(local_isolation_radii) < uniform_isolation_radius
    assert local_isolation_radii[-1] < local_isolation_radii[0] / 10.0
    assert min(
        chart["third_body_separation_floor"] for chart in binary_chart_constants
    ) < uniform_isolation_radius
    assert min(chart["analytic_radius"] for chart in binary_chart_constants) < (
        uniform_isolation_radius
    )

    separated_subfamily = []
    for center in centers:
        if not separated_subfamily or center - separated_subfamily[-1] >= uniform_isolation_radius:
            separated_subfamily.append(center)

    assert len(separated_subfamily) <= finite_count_bound
    assert len(separated_subfamily) < len(centers)


def test_finite_time_binary_accumulation_forces_total_collision_limit():
    accumulation_time = 1.0
    event_times = np.array(
        [accumulation_time - 2.0 ** (-(n + 2)) for n in range(30)]
    )
    repeated_pair_labels = [(0, 1)] * len(event_times)
    terminal_pair_distances = {
        (0, 1): 0.0,
        (0, 2): 0.4,
        (1, 2): 0.4,
    }
    separated_third_floor_at_limit = min(
        terminal_pair_distances[(0, 2)],
        terminal_pair_distances[(1, 2)],
    )
    local_binary_isolation_radius = 0.05
    events_inside_isolation = [
        time
        for time in event_times
        if 0.0 < accumulation_time - time < local_binary_isolation_radius
    ]

    assert len(set(repeated_pair_labels)) == 1
    assert terminal_pair_distances[(0, 1)] == 0.0
    assert separated_third_floor_at_limit > 0.0
    assert len(events_inside_isolation) > 1
    # A separated binary limit supplies a punctured LC chart around the limit,
    # so this many same-pair events inside its isolation radius is impossible.
    separated_binary_limit_contradiction = (
        terminal_pair_distances[(0, 1)] == 0.0
        and separated_third_floor_at_limit > 0.0
        and len(events_inside_isolation) > 1
    )
    assert separated_binary_limit_contradiction

    alternating_pair_labels = [(0, 1), (1, 2)] * 15
    repeated_pairs = {
        pair for pair in set(alternating_pair_labels) if alternating_pair_labels.count(pair) > 1
    }
    terminal_pair_distances = {
        pair: 0.0 if pair in repeated_pairs else np.inf for pair in [(0, 1), (0, 2), (1, 2)]
    }
    terminal_pair_distances[(0, 2)] = min(
        terminal_pair_distances[(0, 1)] + terminal_pair_distances[(1, 2)],
        terminal_pair_distances[(0, 2)],
    )

    assert repeated_pairs == {(0, 1), (1, 2)}
    assert terminal_pair_distances[(0, 1)] == 0.0
    assert terminal_pair_distances[(1, 2)] == 0.0
    assert terminal_pair_distances[(0, 2)] == 0.0


def test_nonzero_angular_momentum_makes_compact_binary_event_set_finite():
    masses, positions, velocities = _rotating_triangle_data()
    exclusion = certify_nonzero_angular_momentum_excludes_triple_collision(
        positions,
        velocities,
        masses,
    )
    compact_interval = (0.0, 1.0)
    accumulating_event_times = np.array(
        [compact_interval[1] - 2.0 ** (-(n + 2)) for n in range(32)]
    )
    event_pairs = [(0, 1)] * len(accumulating_event_times)
    repeated_pair = max(set(event_pairs), key=event_pairs.count)
    subsequence_accumulates_in_interval = (
        compact_interval[0] <= accumulating_event_times[-1] < compact_interval[1]
    )
    binary_accumulation_forces_total_collision = (
        event_pairs.count(repeated_pair) > 1 and subsequence_accumulates_in_interval
    )

    assert exclusion.certified
    assert exclusion.angular_momentum_norm_squared_lower_bound > 0.0
    assert binary_accumulation_forces_total_collision
    assert exclusion.status == "excluded"
    total_collision_excluded = exclusion.status == "excluded"
    infinite_binary_events_compatible = (
        binary_accumulation_forces_total_collision and not total_collision_excluded
    )
    assert not infinite_binary_events_compatible


def test_nonzero_angular_compact_interval_has_finite_mixed_atlas_budget():
    masses, positions, velocities = _rotating_triangle_data()
    exclusion = certify_nonzero_angular_momentum_excludes_triple_collision(
        positions,
        velocities,
        masses,
    )
    compact_interval = (0.0, 2.0)
    binary_events = (
        {
            "time": 0.45,
            "pair": (0, 1),
            "value": 2.0e-8,
            "first_jet": 4.0e-8,
            "lifted_residual": 8.0e-8,
            "physical_residual": 1.0e-12,
        },
        {
            "time": 1.35,
            "pair": (1, 2),
            "value": 3.0e-8,
            "first_jet": 5.0e-8,
            "lifted_residual": 9.0e-8,
            "physical_residual": 1.5e-12,
        },
    )
    ordinary_components = (
        {
            "interval": (0.0, 0.4),
            "value": 1.0e-8,
            "first_jet": 2.0e-8,
            "lifted_residual": 5.0e-8,
            "physical_residual": 4.0e-13,
        },
        {
            "interval": (0.5, 1.3),
            "value": 1.5e-8,
            "first_jet": 2.5e-8,
            "lifted_residual": 6.0e-8,
            "physical_residual": 5.0e-13,
        },
        {
            "interval": (1.4, 2.0),
            "value": 1.2e-8,
            "first_jet": 2.2e-8,
            "lifted_residual": 5.5e-8,
            "physical_residual": 4.5e-13,
        },
    )
    components = ("value", "first_jet", "lifted_residual", "physical_residual")
    atlas_budget = {
        component: sum(piece[component] for piece in ordinary_components)
        + sum(event[component] for event in binary_events)
        for component in components
    }

    assert exclusion.certified
    assert exclusion.status == "excluded"
    assert all(
        compact_interval[0] < event["time"] < compact_interval[1]
        for event in binary_events
    )
    assert len({event["pair"] for event in binary_events}) == 2
    assert len(ordinary_components) == len(binary_events) + 1
    assert all(np.isfinite(atlas_budget[component]) for component in components)
    assert atlas_budget["physical_residual"] < atlas_budget["lifted_residual"]


def test_two_ended_scattering_projected_newton_residual_has_stronger_shell_decay():
    masses = np.array([1.0, 0.7, 1.4])
    past_physical_velocities = np.array(
        [
            [-0.9, 0.25],
            [0.25, 0.95],
            [1.15, -0.5],
        ]
    )
    future_physical_velocities = np.array(
        [
            [-1.0, 0.2],
            [0.35, 0.9],
            [1.2, -0.55],
        ]
    )
    past_offsets = np.array(
        [
            [0.05, -0.25],
            [-0.15, 0.08],
            [0.35, 0.18],
        ]
    )
    future_offsets = np.array(
        [
            [0.1, -0.3],
            [-0.2, 0.05],
            [0.4, 0.2],
        ]
    )
    start_time = 1.0e4
    retained_corrections = 3
    atlas = construct_two_ended_scattering_atlas_recurrence(
        masses,
        past_physical_velocities,
        past_offsets,
        future_physical_velocities,
        future_offsets,
        retained_corrections=retained_corrections,
        middle_budgets={
            "value": 0.0,
            "first_jet": 0.0,
            "residual": 0.0,
            "physical_residual": 1.0e-10,
        },
        initial_start_time=start_time,
        max_doublings=0,
    )
    tau_start = 1.0 / start_time

    assert atlas.certified
    for recurrence in (atlas.past, atlas.future):
        assert recurrence.constants.contraction_factor < 0.01
        assert (
            recurrence.component_ratios["physical_residual"]
            < recurrence.component_ratios["residual"]
            < 1.0
        )

        lifted = [recurrence.shell_tail_bound("residual", index) for index in range(18)]
        physical = [
            recurrence.physical_residual_shell_bound(index) for index in range(18)
        ]
        for shell_index, (lifted_value, physical_value) in enumerate(zip(lifted, physical)):
            tau = recurrence.shell_left_tau(shell_index)
            assert physical_value == pytest.approx(tau**2 * lifted_value)
            assert physical_value <= tau_start**2 * lifted_value * (1.0 + 1.0e-14)
        for previous, current in zip(lifted, lifted[1:]):
            assert current <= recurrence.component_ratios["residual"] * previous
        for previous, current in zip(physical, physical[1:]):
            assert current <= (
                recurrence.component_ratios["physical_residual"] * previous
            )

        long_physical = sum(
            recurrence.physical_residual_shell_bound(index) for index in range(60)
        )
        assert long_physical < recurrence.all_future_tail_bound("physical_residual")

    endpoint_physical_residual_bound = atlas.endpoint_budget("physical_residual")
    endpoint_lifted_residual_bound = atlas.endpoint_budget("residual")

    assert endpoint_physical_residual_bound < tau_start**2 * endpoint_lifted_residual_bound
    assert endpoint_physical_residual_bound < 1.0e-4 * atlas.middle.physical_residual
    assert np.isfinite(atlas.total_budget("physical_residual"))


def test_two_ended_scattering_atlas_requires_matching_asymptotic_invariants():
    masses = np.array([1.0, 0.7, 1.4])
    velocities = np.array(
        [
            [-0.9, 0.25],
            [0.25, 0.95],
            [1.15, -0.5],
        ]
    )
    offsets = np.array(
        [
            [0.05, -0.25],
            [-0.15, 0.08],
            [0.35, 0.18],
        ]
    )

    def wedge(left, right):
        return left[0] * right[1] - left[1] * right[0]

    def scattering_invariants(endpoint_velocities, endpoint_offsets):
        return {
            "momentum": np.sum(masses[:, None] * endpoint_velocities, axis=0),
            "center_offset": np.sum(masses[:, None] * endpoint_offsets, axis=0),
            "angular": sum(
                mass * wedge(offset, velocity)
                for mass, offset, velocity in zip(
                    masses, endpoint_offsets, endpoint_velocities
                )
            ),
            "energy": 0.5
            * float(np.sum(masses[:, None] * endpoint_velocities**2)),
        }

    past = scattering_invariants(velocities, offsets)
    matched_future = scattering_invariants(velocities.copy(), offsets.copy())

    np.testing.assert_allclose(past["momentum"], matched_future["momentum"])
    np.testing.assert_allclose(
        past["center_offset"],
        matched_future["center_offset"],
    )
    assert past["angular"] == pytest.approx(matched_future["angular"])
    assert past["energy"] == pytest.approx(matched_future["energy"])

    log_vector = accelerations(velocities, masses)
    np.testing.assert_allclose(
        np.sum(masses[:, None] * log_vector, axis=0),
        np.zeros(2),
        atol=1.0e-14,
    )
    angular_log_drift = sum(
        mass * wedge(velocity, row)
        for mass, velocity, row in zip(masses, velocities, log_vector)
    )
    assert angular_log_drift == pytest.approx(0.0, abs=1.0e-14)

    future_velocity_mismatch = velocities.copy()
    future_velocity_mismatch[0, 0] += 0.03
    velocity_mismatch = scattering_invariants(future_velocity_mismatch, offsets)

    future_offset_mismatch = offsets.copy()
    future_offset_mismatch[1, 1] += 0.04
    offset_mismatch = scattering_invariants(velocities, future_offset_mismatch)

    assert (
        np.linalg.norm(velocity_mismatch["momentum"] - past["momentum"], ord=np.inf)
        > 1.0e-3
    )
    assert abs(velocity_mismatch["energy"] - past["energy"]) > 1.0e-3
    assert (
        np.linalg.norm(
            offset_mismatch["center_offset"] - past["center_offset"],
            ord=np.inf,
        )
        > 1.0e-3
    )
    assert abs(offset_mismatch["angular"] - past["angular"]) > 1.0e-3


def test_homothetic_escape_log_subtracted_endpoint_has_convergent_implicit_series():
    configuration, central_lambda = _central_configuration_data("equilateral")
    initial_radius = 1.0
    initial_speed = 2.0
    endpoint = construct_homothetic_escape_endpoint_data(
        gravitational_parameter=central_lambda,
        initial_radius=initial_radius,
        initial_speed=initial_speed,
    )
    endpoint_h_x = 1.0 / endpoint.asymptotic_speed
    endpoint_h_log = endpoint.beta / (2.0 * endpoint.asymptotic_speed)
    forced_log_coefficient = -endpoint_h_log / endpoint_h_x

    assert endpoint.certified
    assert forced_log_coefficient == pytest.approx(endpoint.forced_log_coefficient)

    corrected_errors = []
    inverse_times = []
    residuals = []
    for radius in (1e3, 1e5, 1e7):
        physical_time = (
            homothetic_escape_time_integral(
                radius,
                central_lambda,
                endpoint.asymptotic_speed,
            )
            + endpoint.time_shift
        )
        inverse_time = 1.0 / physical_time
        log_variable = inverse_time * np.log(inverse_time)
        scaled_radius = radius * inverse_time
        residuals.append(
            abs(
                homothetic_escape_endpoint_implicit_residual(
                    scaled_radius,
                    inverse_time,
                    log_variable,
                    endpoint,
                )
            )
        )
        corrected_errors.append(
            abs(
                scaled_radius
                - forced_log_coefficient * log_variable
                - endpoint.asymptotic_speed
            )
        )
        inverse_times.append(inverse_time)

    assert max(residuals) < 1e-12
    assert corrected_errors[0] > corrected_errors[1] > corrected_errors[2]
    assert max(error / tau for error, tau in zip(corrected_errors, inverse_times)) < 1.1
    assert np.linalg.norm(
        accelerations(endpoint.asymptotic_speed * configuration)
        - forced_log_coefficient * configuration,
        ord=np.inf,
    ) < 1e-14


def test_homothetic_escape_dyadic_endpoint_constructor_bounds_all_future_tail():
    _configuration, central_lambda = _central_configuration_data("equilateral")
    initial_radius = 1.0
    initial_speed = 2.0
    endpoint = construct_homothetic_escape_endpoint_data(
        gravitational_parameter=central_lambda,
        initial_radius=initial_radius,
        initial_speed=initial_speed,
    )
    start_time = 1.0e4
    recurrence = construct_homothetic_escape_dyadic_recurrence(
        endpoint,
        start_time=start_time,
        tau_radius=1.0,
        rho_radius=1.0,
        cauchy_majorant=2.0,
        retained_degree=0,
    )

    corrected_errors = []
    implicit_residuals = []
    for shell_index in range(22):
        physical_time = start_time * 2.0**shell_index
        radius = homothetic_escape_radius_at_time(
            physical_time,
            endpoint,
        )
        inverse_time = 1.0 / physical_time
        log_variable = inverse_time * np.log(inverse_time)
        scaled_radius = radius * inverse_time
        corrected_errors.append(
            abs(
                scaled_radius
                - endpoint.forced_log_coefficient * log_variable
                - endpoint.asymptotic_speed
            )
        )
        implicit_residuals.append(
            abs(
                homothetic_escape_endpoint_implicit_residual(
                    scaled_radius,
                    inverse_time,
                    log_variable,
                    endpoint,
                )
            )
        )

    assert recurrence.certified
    assert max(implicit_residuals[:12]) < 2.0e-12
    for shell_index, error in enumerate(corrected_errors):
        assert error <= recurrence.shell_tail_bound(shell_index)
        assert recurrence.shell_tail_bound(shell_index) <= (
            recurrence.geometric_shell_tail_bound(shell_index) * (1.0 + 1e-12)
        )

    assert recurrence.shell_tail_ratio < 1.0
    assert sum(corrected_errors) < recurrence.all_future_tail_bound
    assert recurrence.future_tail_from_shell(4) < recurrence.all_future_tail_bound
    assert corrected_errors[-1] < 1.0e-10


def test_homothetic_escape_dyadic_endpoint_constructor_rejects_uncertified_inputs():
    _configuration, central_lambda = _central_configuration_data("equilateral")
    endpoint = construct_homothetic_escape_endpoint_data(
        gravitational_parameter=central_lambda,
        initial_radius=1.0,
        initial_speed=2.0,
    )

    with pytest.raises(ValueError, match="positive energy"):
        construct_homothetic_escape_endpoint_data(
            gravitational_parameter=central_lambda,
            initial_radius=1.0,
            initial_speed=0.5,
        )

    with pytest.raises(ValueError, match="not certified"):
        construct_homothetic_escape_dyadic_recurrence(
            endpoint,
            start_time=3.0,
            tau_radius=1.0e-8,
            rho_radius=1.0e-8,
            cauchy_majorant=1.0,
            retained_degree=1,
        )

    with pytest.raises(ValueError, match="positive tail exponent"):
        construct_homothetic_escape_dyadic_recurrence(
            endpoint,
            start_time=1.0e4,
            tau_radius=1.0,
            rho_radius=1.0,
            cauchy_majorant=1.0,
            retained_degree=0,
            derivative_order=1,
        )

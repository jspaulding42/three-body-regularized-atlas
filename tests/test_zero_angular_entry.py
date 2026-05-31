import numpy as np

from three_body_symmetry.zero_angular_entry import (
    certify_cubic_time_total_collision_cubic_jet_kernel,
    certify_cubic_time_total_collision_leading_jet,
)


def _equal_mass_equilateral_cubic_time_shape():
    configuration = np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ],
    )
    central_lambda = 1.0 / np.sqrt(3.0)
    scale = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    return scale * configuration


def test_cubic_time_leading_jet_certifies_central_second_shape_only():
    masses = np.ones(3)
    central_shape = _equal_mass_equilateral_cubic_time_shape()

    certificate = certify_cubic_time_total_collision_leading_jet(
        masses=masses,
        quadratic_coefficient=central_shape,
        tolerance=1.0e-12,
    )
    wrong_scale = certify_cubic_time_total_collision_leading_jet(
        masses=masses,
        quadratic_coefficient=1.04 * central_shape,
        tolerance=1.0e-12,
    )
    shifted = certify_cubic_time_total_collision_leading_jet(
        masses=masses,
        quadratic_coefficient=central_shape + np.array([0.02, -0.03]),
        tolerance=1.0e-12,
    )

    assert certificate.certified
    assert certificate.proof_certified
    assert certificate.central_equation_residual < 1.0e-14
    assert not certificate.arbitrary_fuchsian_entry_theorem_claimed
    assert "A(C)=-(2/9)C" in certificate.statement
    assert not wrong_scale.certified
    assert "central_configuration_normalization_A_plus_two_ninths_C" in (
        wrong_scale.missing_obligations
    )
    assert not shifted.certified
    assert "centered_quadratic_total_collision_shape" in shifted.missing_obligations


def test_cubic_time_cubic_jet_kernel_accepts_translation_and_rejects_scaling():
    masses = np.ones(3)
    central_shape = _equal_mass_equilateral_cubic_time_shape()
    translation_cubic = np.array(
        [
            [0.13, -0.07],
            [0.13, -0.07],
            [0.13, -0.07],
        ],
    )
    scaling_cubic = 0.25 * central_shape

    translation = certify_cubic_time_total_collision_cubic_jet_kernel(
        masses=masses,
        quadratic_coefficient=central_shape,
        cubic_coefficient=translation_cubic,
        tolerance=1.0e-12,
    )
    scaling = certify_cubic_time_total_collision_cubic_jet_kernel(
        masses=masses,
        quadratic_coefficient=central_shape,
        cubic_coefficient=scaling_cubic,
        tolerance=1.0e-12,
    )

    assert translation.certified
    assert translation.kernel_residual < 1.0e-14
    assert not translation.arbitrary_fuchsian_entry_theorem_claimed
    assert "DA(C)D=0" in translation.statement
    assert not scaling.certified
    assert scaling.kernel_residual > 0.01
    assert "linearized_cubic_jet_kernel_condition" in scaling.missing_obligations


def test_cubic_time_cubic_jet_kernel_rejects_stale_leading_certificate():
    masses = np.ones(3)
    central_shape = _equal_mass_equilateral_cubic_time_shape()
    translation_cubic = np.repeat([[0.03, 0.01]], 3, axis=0)
    stale_leading = certify_cubic_time_total_collision_leading_jet(
        masses=masses,
        quadratic_coefficient=central_shape,
        tolerance=1.0e-12,
    )

    certificate = certify_cubic_time_total_collision_cubic_jet_kernel(
        masses=masses,
        quadratic_coefficient=central_shape + np.array([0.02, -0.03]),
        cubic_coefficient=translation_cubic,
        leading_certificate=stale_leading,
        tolerance=1.0e-12,
    )

    assert not certificate.certified
    assert "leading_cubic_time_entry_certificate" in certificate.missing_obligations

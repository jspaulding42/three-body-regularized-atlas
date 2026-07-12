from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from three_body_symmetry.dynamics import accelerations
from three_body_symmetry.zero_angular_entry import (
    CubicTimeEntryObligation,
    FiniteJetDerivedIdentitySelectorEntryCertificate,
    FiniteJetIdentitySelectorEntryCertificate,
    FiniteJetSelectedBranchCertificate,
    FiniteJetSelectorCoordinateCertificate,
    FiniteJetSelectorSpec,
    certify_cubic_time_total_collision_cubic_jet_kernel,
    certify_cubic_time_total_collision_leading_jet,
)


def _equal_mass_equilateral_cubic_time_leading_shape():
    masses = np.ones(3)
    shape = np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ]
    )
    acceleration = accelerations(shape, masses)
    central_lambda = -float(
        np.sum(masses[:, None] * shape * acceleration)
        / np.sum(masses[:, None] * shape * shape)
    )
    scale = ((9.0 / 2.0) * central_lambda) ** (1.0 / 3.0)
    return masses, scale * shape


def test_cubic_time_entry_ledgers_reject_spoofed_and_optional_only_obligations():
    masses, leading_shape = _equal_mass_equilateral_cubic_time_leading_shape()
    certificate = certify_cubic_time_total_collision_leading_jet(
        masses=masses,
        quadratic_coefficient=leading_shape,
        tolerance=1.0e-10,
    )
    fake_obligation = SimpleNamespace(
        obligation="attribute_compatible_fake",
        certified=True,
        required=True,
    )
    optional_only = CubicTimeEntryObligation(
        obligation="optional_only",
        certified=True,
        detail="not a required proof row",
        required=False,
    )
    truthy_obligation = CubicTimeEntryObligation(
        obligation="truthy_certified_row",
        certified="yes",
        detail="truthy strings must not certify",
    )

    fake_ledger = replace(certificate, obligations=(fake_obligation,))
    optional_ledger = replace(certificate, obligations=(optional_only,))
    truthy_ledger = replace(certificate, obligations=(truthy_obligation,))

    assert certificate.certified
    assert truthy_obligation.certified is False
    assert not fake_ledger.certified
    assert "cubic_time_leading_jet_entry_obligation_type" in (
        fake_ledger.missing_obligations
    )
    assert not optional_ledger.certified
    assert "cubic_time_leading_jet_entry_required_obligation_present" in (
        optional_ledger.missing_obligations
    )
    assert not truthy_ledger.certified
    assert "truthy_certified_row" in truthy_ledger.missing_obligations


def test_cubic_jet_kernel_rejects_fake_leading_certificate_and_spoofed_ledger():
    masses, leading_shape = _equal_mass_equilateral_cubic_time_leading_shape()
    leading = certify_cubic_time_total_collision_leading_jet(
        masses=masses,
        quadratic_coefficient=leading_shape,
        tolerance=1.0e-10,
    )
    translation_cubic = np.array(
        [
            [0.03, -0.02],
            [0.03, -0.02],
            [0.03, -0.02],
        ]
    )
    certificate = certify_cubic_time_total_collision_cubic_jet_kernel(
        masses=masses,
        quadratic_coefficient=leading_shape,
        cubic_coefficient=translation_cubic,
        leading_certificate=leading,
        tolerance=1.0e-10,
    )
    fake_leading = SimpleNamespace(certified=True, missing_obligations=())
    fake_obligation = SimpleNamespace(
        obligation="attribute_compatible_fake",
        certified=True,
        required=True,
    )

    fake_leading_certificate = replace(
        certificate,
        leading_certificate=fake_leading,
    )
    fake_ledger = replace(certificate, obligations=(fake_obligation,))

    assert leading.certified
    assert certificate.certified
    assert not fake_leading_certificate.certified
    assert "cubic_time_leading_jet_entry_certificate_type" in (
        fake_leading_certificate.missing_obligations
    )
    assert not fake_ledger.certified
    assert "cubic_time_cubic_jet_kernel_obligation_type" in (
        fake_ledger.missing_obligations
    )


def test_finite_jet_identity_selector_rejects_fake_coordinate_certificates():
    fake_coordinate = SimpleNamespace(
        name="selector",
        degree=0,
        regularized_power=2,
        selector_gap=0.0,
        certified=True,
    )
    coordinate = FiniteJetSelectorCoordinateCertificate(
        name="selector",
        degree=0,
        regularized_power=2,
        recovered_value=0.0,
        selected_value=0.0,
        selector_gap=0.0,
        basis_norm=1.0,
    )
    certificate = FiniteJetIdentitySelectorEntryCertificate(
        masses=(1.0, 1.0, 1.0),
        incoming_energy_limit=0.0,
        selected_energy_limit=0.0,
        energy_gap=0.0,
        coordinate_certificates=(coordinate,),
        required_regularized_jet_order=2,
        sample_taus=(0.1,),
        newton_residual_bounds=(0.0,),
        angular_momentum_bounds=(0.0,),
        tolerance=1.0e-8,
    )
    spoofed = replace(certificate, coordinate_certificates=(fake_coordinate,))

    assert coordinate.certified
    assert certificate.certified
    assert not spoofed.certified


def test_finite_jet_selected_branch_and_derived_entry_require_constructor_types():
    masses, leading_shape = _equal_mass_equilateral_cubic_time_leading_shape()
    coefficients = np.zeros((3, *leading_shape.shape), dtype=float)
    coefficients[0] = leading_shape
    selector_spec = FiniteJetSelectorSpec(
        name="selector",
        degree=0,
        basis=np.ones_like(leading_shape),
    )
    selected_branch = FiniteJetSelectedBranchCertificate(
        masses=tuple(float(mass) for mass in masses),
        coefficients=coefficients,
        selector_specs=(selector_spec,),
        selected_values=(("selector", 0.0),),
        selector_operator_residual_bounds=(("selector", 0.0),),
        selector_gram_singular_value_floors=((0, 1.0),),
        coefficient_residual_bounds=(0.0,),
        sample_taus=(0.1,),
        newton_residual_bounds=(0.0,),
        angular_momentum_bounds=(0.0,),
        tolerance=1.0e-8,
    )
    entry_certificate = FiniteJetIdentitySelectorEntryCertificate(
        masses=tuple(float(mass) for mass in masses),
        incoming_energy_limit=0.0,
        selected_energy_limit=0.0,
        energy_gap=0.0,
        coordinate_certificates=(
            FiniteJetSelectorCoordinateCertificate(
                name="selector",
                degree=0,
                regularized_power=2,
                recovered_value=0.0,
                selected_value=0.0,
                selector_gap=0.0,
                basis_norm=1.0,
            ),
        ),
        required_regularized_jet_order=2,
        sample_taus=(0.1,),
        newton_residual_bounds=(0.0,),
        angular_momentum_bounds=(0.0,),
        tolerance=1.0e-8,
    )
    derived = FiniteJetDerivedIdentitySelectorEntryCertificate(
        incoming_coefficients=coefficients,
        selected_branch=selected_branch,
        entry_certificate=entry_certificate,
        recovered_selector_values=selected_branch.selected_values,
    )
    fake_spec = SimpleNamespace(
        name="selector",
        degree=0,
        basis=np.ones_like(leading_shape),
    )
    fake_branch = SimpleNamespace(
        certified=True,
        selected_values=selected_branch.selected_values,
        max_newton_residual=0.0,
        max_angular_momentum=0.0,
    )
    fake_entry = SimpleNamespace(
        certified=True,
        max_newton_residual=0.0,
        max_angular_momentum=0.0,
    )

    assert selected_branch.certified
    assert entry_certificate.certified
    assert derived.certified
    assert not replace(selected_branch, selector_specs=(fake_spec,)).certified
    assert not replace(derived, selected_branch=fake_branch).certified
    assert not replace(derived, entry_certificate=fake_entry).certified

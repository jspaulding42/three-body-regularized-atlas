from scripts.verify_lc_projection_identities import main, verify_identities


def test_all_lc_projection_identities_hold_exactly() -> None:
    results = verify_identities()
    assert results
    assert {
        "full_newton_first_body_acceleration",
        "full_newton_second_body_acceleration",
        "full_newton_third_body_acceleration",
        "deck_third_body_displacement_first",
        "deck_third_body_displacement_second",
    } <= results.keys()
    assert all(results.values()), results


def test_lc_projection_identity_verifier_exits_successfully() -> None:
    assert main() == 0

from scripts.verify_lc_projection_identities import main, verify_identities


def test_all_lc_projection_identities_hold_exactly() -> None:
    results = verify_identities()
    assert results
    assert all(results.values()), results


def test_lc_projection_identity_verifier_exits_successfully() -> None:
    assert main() == 0

use num_rational::BigRational;
use num_traits::Zero;

use three_body_planar_chain_verifier::raw_schema::{
    decode_raw_chain, RawPlanarChainWire, SchemaProfile,
};
use three_body_planar_chain_verifier::{
    ordinary_chart_input_from_wire, ordinary_tube_input_from_wire, parse_wire_json,
    replay_initial_value_binding_exact_rational_v04,
    replay_validated_ordinary_root_exact_rational_v04, InitialValueBindingObligation,
    ValidatedOrdinaryRootObligation, DEFAULT_WIRE_JSON_LIMITS,
    EXACT_RATIONAL_INITIAL_VALUE_BINDING_V04_PROFILE_ID,
    EXACT_RATIONAL_VALIDATED_ORDINARY_ROOT_V04_PROFILE_ID, INITIAL_VALUE_BINDING_OBLIGATION_IDS,
    VALIDATED_ORDINARY_ROOT_OBLIGATION_IDS,
};

const SUCCESS_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../artifacts/v0.3.0-review/planar-chain/success.raw.json"
));
const FAILED_REVISIT_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../artifacts/v0.3.0-review/planar-chain/failed-revisit.raw.json"
));

fn decode(bytes: &[u8]) -> RawPlanarChainWire {
    let syntax = parse_wire_json(bytes, DEFAULT_WIRE_JSON_LIMITS).unwrap();
    decode_raw_chain(syntax, SchemaProfile::V03Compatible).unwrap()
}

#[test]
fn public_exact_binding_and_validated_root_profiles_accept_both_canonical_roots() {
    for (name, bytes) in [
        ("success", SUCCESS_RAW),
        ("failed-revisit", FAILED_REVISIT_RAW),
    ] {
        let chain = decode(bytes);
        let chart = ordinary_chart_input_from_wire(&chain.initial_chart).unwrap();
        let tube = ordinary_tube_input_from_wire(&chain.initial_tube);

        let binding =
            replay_initial_value_binding_exact_rational_v04(&chain.root_binding, &chart).unwrap();
        assert_eq!(
            binding.profile_id(),
            EXACT_RATIONAL_INITIAL_VALUE_BINDING_V04_PROFILE_ID,
            "{name}"
        );
        assert_eq!(
            binding
                .obligations()
                .iter()
                .map(InitialValueBindingObligation::id)
                .collect::<Vec<_>>(),
            INITIAL_VALUE_BINDING_OBLIGATION_IDS,
            "{name}"
        );
        assert!(
            binding
                .obligations()
                .iter()
                .all(InitialValueBindingObligation::satisfied),
            "{name}: {binding:?}"
        );
        assert!(binding.profile_satisfied(), "{name}: {binding:?}");
        assert_eq!(binding.time_gap(), Some(&BigRational::zero()), "{name}");
        assert_eq!(
            binding.maximum_position_gap(),
            Some(&BigRational::zero()),
            "{name}"
        );
        assert_eq!(
            binding.maximum_velocity_gap(),
            Some(&BigRational::zero()),
            "{name}"
        );

        let root =
            replay_validated_ordinary_root_exact_rational_v04(&chain.root_binding, &chart, &tube)
                .unwrap();
        assert_eq!(
            root.profile_id(),
            EXACT_RATIONAL_VALIDATED_ORDINARY_ROOT_V04_PROFILE_ID,
            "{name}"
        );
        assert_eq!(
            root.obligations()
                .iter()
                .map(ValidatedOrdinaryRootObligation::id)
                .collect::<Vec<_>>(),
            VALIDATED_ORDINARY_ROOT_OBLIGATION_IDS,
            "{name}"
        );
        assert!(
            root.obligations()
                .iter()
                .all(ValidatedOrdinaryRootObligation::satisfied),
            "{name}: {root:?}"
        );
        assert!(root.validated_root_satisfied(), "{name}: {root:?}");
        assert_eq!(
            root.actual_initial_error(),
            Some(&BigRational::zero()),
            "{name}"
        );
        assert_eq!(
            root.root_clock_origin(),
            Some(&BigRational::zero()),
            "{name}"
        );
    }
}

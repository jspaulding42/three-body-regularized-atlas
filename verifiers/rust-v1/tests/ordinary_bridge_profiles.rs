use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

use three_body_planar_chain_verifier::raw_schema::{
    decode_raw_chain, RawPlanarChainWire, SchemaProfile, SegmentWire,
};
use three_body_planar_chain_verifier::{
    ordinary_chart_input_from_wire, ordinary_tube_input_from_wire, parse_wire_json,
    replay_carried_ordinary_bridge_exact_rational_v04, RationalInterval, DEFAULT_WIRE_JSON_LIMITS,
    EXACT_RATIONAL_CARRIED_ORDINARY_BRIDGE_V04_PROFILE_ID,
    ORDINARY_AUTONOMOUS_UNIQUENESS_BRIDGE_KERNEL_V1_ID, ORDINARY_BRIDGE_OBLIGATION_IDS,
};

const SUCCESS_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../conformance/raw-v1/inputs/success.raw.json"
));
const FAILED_REVISIT_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../conformance/raw-v1/inputs/failed-revisit.raw.json"
));

fn decode(bytes: &[u8]) -> RawPlanarChainWire {
    let syntax = parse_wire_json(bytes, DEFAULT_WIRE_JSON_LIMITS).unwrap();
    decode_raw_chain(syntax, SchemaProfile::V03Compatible).unwrap()
}

#[test]
fn public_api_replays_the_first_carried_ordinary_bridge_in_both_frozen_chains() {
    for (case_id, bytes) in [
        ("success", SUCCESS_RAW),
        ("failed-revisit", FAILED_REVISIT_RAW),
    ] {
        let chain = decode(bytes);
        let SegmentWire::OrdinaryBridge(segment) = &chain.segments[0] else {
            panic!("{case_id} first segment is not an ordinary bridge");
        };
        let source_chart = ordinary_chart_input_from_wire(&chain.initial_chart).unwrap();
        let source_tube = ordinary_tube_input_from_wire(&chain.initial_tube);
        let target_chart = ordinary_chart_input_from_wire(&segment.target_chart).unwrap();
        let target_tube = ordinary_tube_input_from_wire(&segment.target_tube);
        let parent_clock = RationalInterval::try_point(BigRational::zero()).unwrap();
        let replay = replay_carried_ordinary_bridge_exact_rational_v04(
            &segment.transition,
            &source_chart,
            &source_tube,
            &target_chart,
            &target_tube,
            &parent_clock,
        )
        .unwrap_or_else(|error| panic!("{case_id} replay failed: {error}"));

        assert!(replay.local_bridge_satisfied(), "{case_id}");
        assert_eq!(
            replay.profile_id(),
            EXACT_RATIONAL_CARRIED_ORDINARY_BRIDGE_V04_PROFILE_ID
        );
        assert_eq!(
            replay.analytic_kernel_id(),
            ORDINARY_AUTONOMOUS_UNIQUENESS_BRIDGE_KERNEL_V1_ID
        );
        assert_eq!(
            replay
                .obligations()
                .iter()
                .map(|obligation| obligation.id())
                .collect::<Vec<_>>(),
            ORDINARY_BRIDGE_OBLIGATION_IDS
        );
        assert_eq!(
            replay.target_clock_origin(),
            Some(
                &RationalInterval::try_point(BigRational::new(BigInt::one(), BigInt::one() << 40,))
                    .unwrap()
            ),
            "{case_id}"
        );
    }
}

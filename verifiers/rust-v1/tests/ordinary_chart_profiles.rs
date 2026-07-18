use std::collections::{BTreeMap, BTreeSet};

use three_body_planar_chain_verifier::raw_schema::{
    decode_raw_chain, OrdinaryChartWire, RawPlanarChainWire, SchemaProfile, SegmentWire,
};
use three_body_planar_chain_verifier::{
    ordinary_chart_input_from_wire, parse_wire_json,
    replay_ordinary_chart_exact_rational_claimed_tail_v04, validate_canonical_wire_json,
    WireJsonValue, DEFAULT_WIRE_JSON_LIMITS,
    EXACT_RATIONAL_ORDINARY_CHART_CLAIMED_TAIL_V04_PROFILE_ID, ORDINARY_CHART_OBLIGATION_IDS,
};

const EXPECTATIONS: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../conformance/raw-v1/ordinary-chart-profile-expectations.json"
));
const SUCCESS_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../conformance/raw-v1/inputs/success.raw.json"
));
const FAILED_REVISIT_RAW: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../conformance/raw-v1/inputs/failed-revisit.raw.json"
));

struct CaseInput {
    case_id: &'static str,
    input_file: &'static str,
    input_sha256: &'static str,
    bytes: &'static [u8],
}

const CASE_INPUTS: [CaseInput; 2] = [
    CaseInput {
        case_id: "failed-revisit",
        input_file: "inputs/failed-revisit.raw.json",
        input_sha256: "c6830919726c22fbca6e45d5781b2465e54876bfa89c25bc26e97284ff918727",
        bytes: FAILED_REVISIT_RAW,
    },
    CaseInput {
        case_id: "success",
        input_file: "inputs/success.raw.json",
        input_sha256: "ede15b0f35cf741f542a6cd260470a93ee5ff85dc5ae2371db1b88819b821f11",
        bytes: SUCCESS_RAW,
    },
];

#[test]
fn rust_claimed_tail_profile_matches_the_neutral_positional_expectations() {
    let expectation = parse_wire_json(EXPECTATIONS, DEFAULT_WIRE_JSON_LIMITS)
        .expect("ordinary-chart expectation must be strict JSON");
    let root = object(&expectation, "$ expectation");
    exact_fields(
        root,
        &["expectation_schema", "obligation_ids", "cases"],
        "$ expectation",
    );
    assert_eq!(
        string(
            field(root, "expectation_schema", "$ expectation"),
            "$.expectation_schema"
        ),
        "raw-v1-ordinary-chart-profile-expectation-v1"
    );

    let expected_ids = string_array(
        field(root, "obligation_ids", "$ expectation"),
        "$.obligation_ids",
    );
    assert_eq!(expected_ids, ORDINARY_CHART_OBLIGATION_IDS);
    let cases = array(field(root, "cases", "$ expectation"), "$.cases");
    assert_eq!(cases.len(), CASE_INPUTS.len());
    let case_ids = cases
        .iter()
        .map(|case| {
            let case = object(case, "$.cases[]");
            string(field(case, "case_id", "$.cases[]"), "$.cases[].case_id")
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        case_ids,
        CASE_INPUTS
            .iter()
            .map(|case| case.case_id)
            .collect::<BTreeSet<_>>()
    );

    for input in &CASE_INPUTS {
        let case_value = cases
            .iter()
            .find(|case| {
                let case = object(case, "$.cases[]");
                string(field(case, "case_id", "$.cases[]"), "$.cases[].case_id") == input.case_id
            })
            .unwrap_or_else(|| panic!("missing expectation case {}", input.case_id));
        check_case(case_value, input, &expected_ids);
    }
}

fn check_case(case_value: &WireJsonValue, input: &CaseInput, expected_ids: &[&str]) {
    let case = object(case_value, "$.cases[]");
    exact_fields(
        case,
        &[
            "case_id",
            "input_file",
            "input_sha256",
            "chart_order",
            "profiles",
            "comparison",
        ],
        "$.cases[]",
    );
    assert_eq!(
        string(field(case, "case_id", "$.cases[]"), "$.cases[].case_id"),
        input.case_id
    );
    assert_eq!(
        string(
            field(case, "input_file", "$.cases[]"),
            "$.cases[].input_file"
        ),
        input.input_file
    );
    assert_eq!(
        string(
            field(case, "input_sha256", "$.cases[]"),
            "$.cases[].input_sha256"
        ),
        input.input_sha256
    );

    let raw = validate_canonical_wire_json(input.bytes, DEFAULT_WIRE_JSON_LIMITS).unwrap_or_else(
        |error| panic!("{} is not canonical raw-v1 JSON: {error:?}", input.case_id),
    );
    let chain = decode_raw_chain(raw, SchemaProfile::V03Compatible)
        .unwrap_or_else(|error| panic!("{} did not decode: {error:?}", input.case_id));
    let charts = ordinary_charts(&chain);
    assert_eq!(charts.len(), 6);
    check_chart_order(case, &charts, input.case_id);

    let profiles = array(field(case, "profiles", "$.cases[]"), "$.cases[].profiles");
    assert_eq!(profiles.len(), 2, "{} profile count", input.case_id);
    let mut profile_ids = BTreeSet::new();
    for profile_value in profiles {
        let profile = object(profile_value, "$.cases[].profiles[]");
        exact_fields(
            profile,
            &[
                "profile_id",
                "runtime",
                "profile_scope",
                "chart_satisfaction",
            ],
            "$.cases[].profiles[]",
        );
        profile_ids.insert(string(
            field(profile, "profile_id", "$.cases[].profiles[]"),
            "$.cases[].profiles[].profile_id",
        ));
        assert!(!string(
            field(profile, "runtime", "$.cases[].profiles[]"),
            "$.cases[].profiles[].runtime",
        )
        .is_empty());
        assert!(!string(
            field(profile, "profile_scope", "$.cases[].profiles[]"),
            "$.cases[].profiles[].profile_scope",
        )
        .is_empty());
        let satisfaction = array(
            field(profile, "chart_satisfaction", "$.cases[].profiles[]"),
            "$.cases[].profiles[].chart_satisfaction",
        );
        assert_eq!(satisfaction.len(), charts.len());
        for ledger in satisfaction {
            assert_eq!(
                boolean_array(ledger, "$.cases[].profiles[].chart_satisfaction[]").len(),
                expected_ids.len()
            );
        }
    }

    let rust_profile_value = profiles
        .iter()
        .find(|profile| {
            let profile = object(profile, "$.cases[].profiles[]");
            string(
                field(profile, "profile_id", "$.cases[].profiles[]"),
                "$.cases[].profiles[].profile_id",
            ) == EXACT_RATIONAL_ORDINARY_CHART_CLAIMED_TAIL_V04_PROFILE_ID
        })
        .unwrap_or_else(|| panic!("{} has no Rust claimed-tail profile", input.case_id));
    let rust_profile = object(rust_profile_value, "$.cases[].profiles[]");
    assert_eq!(
        string(
            field(rust_profile, "runtime", "$.cases[].profiles[]"),
            "$.cases[].profiles[].runtime",
        ),
        "rust-v1"
    );
    assert_eq!(
        string(
            field(rust_profile, "profile_scope", "$.cases[].profiles[]"),
            "$.cases[].profiles[].profile_scope",
        ),
        "conditional_claimed_tail"
    );
    let expected_ledgers = array(
        field(rust_profile, "chart_satisfaction", "$.cases[].profiles[]"),
        "$.cases[].profiles[].chart_satisfaction",
    );

    for (position, (chart, expected_ledger)) in charts.iter().zip(expected_ledgers).enumerate() {
        let semantic = ordinary_chart_input_from_wire(chart).unwrap_or_else(|error| {
            panic!(
                "{} chart {position} failed semantic admission: {error:?}",
                input.case_id
            )
        });
        let replay = replay_ordinary_chart_exact_rational_claimed_tail_v04(&semantic)
            .unwrap_or_else(|error| {
                panic!(
                    "{} chart {position} failed replay: {error:?}",
                    input.case_id
                )
            });
        assert_eq!(
            replay.profile_id(),
            EXACT_RATIONAL_ORDINARY_CHART_CLAIMED_TAIL_V04_PROFILE_ID
        );
        assert_eq!(
            replay
                .obligations()
                .iter()
                .map(|obligation| obligation.id())
                .collect::<Vec<_>>(),
            expected_ids
        );
        assert_eq!(
            replay
                .obligations()
                .iter()
                .map(|obligation| obligation.satisfied())
                .collect::<Vec<_>>(),
            boolean_array(expected_ledger, "$.cases[].profiles[].chart_satisfaction[]",),
            "{} chart {position}",
            input.case_id
        );
    }

    let comparison = object(
        field(case, "comparison", "$.cases[]"),
        "$.cases[].comparison",
    );
    exact_fields(
        comparison,
        &[
            "left_profile_id",
            "right_profile_id",
            "observed_ordered_boolean_ledgers_equal",
            "mathematical_outcomes_comparable",
            "release_gate_status",
            "blocker_id",
        ],
        "$.cases[].comparison",
    );
    let left_profile_id = string(
        field(comparison, "left_profile_id", "$.cases[].comparison"),
        "$.cases[].comparison.left_profile_id",
    );
    let right_profile_id = string(
        field(comparison, "right_profile_id", "$.cases[].comparison"),
        "$.cases[].comparison.right_profile_id",
    );
    assert!(profile_ids.contains(left_profile_id));
    assert!(profile_ids.contains(right_profile_id));
    assert_eq!(
        right_profile_id,
        EXACT_RATIONAL_ORDINARY_CHART_CLAIMED_TAIL_V04_PROFILE_ID
    );
    assert!(boolean(
        field(
            comparison,
            "observed_ordered_boolean_ledgers_equal",
            "$.cases[].comparison",
        ),
        "$.cases[].comparison.observed_ordered_boolean_ledgers_equal",
    ));
    assert!(!boolean(
        field(
            comparison,
            "mathematical_outcomes_comparable",
            "$.cases[].comparison",
        ),
        "$.cases[].comparison.mathematical_outcomes_comparable",
    ));
    assert_eq!(
        string(
            field(comparison, "release_gate_status", "$.cases[].comparison"),
            "$.cases[].comparison.release_gate_status",
        ),
        "BLOCKED"
    );
    assert_eq!(
        string(
            field(comparison, "blocker_id", "$.cases[].comparison"),
            "$.cases[].comparison.blocker_id",
        ),
        "profile_semantics_differ_open_v1_06"
    );
}

fn ordinary_charts(chain: &RawPlanarChainWire) -> Vec<&OrdinaryChartWire> {
    let mut charts = Vec::with_capacity(chain.segments.len() + 1);
    charts.push(&chain.initial_chart);
    for segment in &chain.segments {
        charts.push(match segment {
            SegmentWire::OrdinaryBridge(segment) => &segment.target_chart,
            SegmentWire::PlanarLcPassage(segment) => &segment.target_chart,
        });
    }
    charts
}

fn check_chart_order(
    case: &BTreeMap<String, WireJsonValue>,
    charts: &[&OrdinaryChartWire],
    case_id: &str,
) {
    let order = array(
        field(case, "chart_order", "$.cases[]"),
        "$.cases[].chart_order",
    );
    assert_eq!(order.len(), charts.len());
    for (position, (entry_value, chart)) in order.iter().zip(charts).enumerate() {
        let entry = object(entry_value, "$.cases[].chart_order[]");
        exact_fields(
            entry,
            &["path", "certificate_id", "chart_id"],
            "$.cases[].chart_order[]",
        );
        let expected_path = if position == 0 {
            "$.initial_chart".to_owned()
        } else {
            format!("$.segments[{}].target_chart", position - 1)
        };
        assert_eq!(
            string(
                field(entry, "path", "$.cases[].chart_order[]"),
                "$.cases[].chart_order[].path"
            ),
            expected_path,
            "{case_id} chart {position} path"
        );
        assert_eq!(
            string(
                field(entry, "certificate_id", "$.cases[].chart_order[]"),
                "$.cases[].chart_order[].certificate_id",
            ),
            chart.certificate_id,
            "{case_id} chart {position} certificate"
        );
        assert_eq!(
            string(
                field(entry, "chart_id", "$.cases[].chart_order[]"),
                "$.cases[].chart_order[].chart_id",
            ),
            chart.chart_id,
            "{case_id} chart {position} id"
        );
    }
}

fn object<'a>(value: &'a WireJsonValue, path: &str) -> &'a BTreeMap<String, WireJsonValue> {
    match value {
        WireJsonValue::Object(value) => value,
        other => panic!("{path} must be an object, got {other:?}"),
    }
}

fn array<'a>(value: &'a WireJsonValue, path: &str) -> &'a [WireJsonValue] {
    match value {
        WireJsonValue::Array(value) => value,
        other => panic!("{path} must be an array, got {other:?}"),
    }
}

fn string<'a>(value: &'a WireJsonValue, path: &str) -> &'a str {
    match value {
        WireJsonValue::String(value) => value,
        other => panic!("{path} must be a string, got {other:?}"),
    }
}

fn boolean(value: &WireJsonValue, path: &str) -> bool {
    match value {
        WireJsonValue::Bool(value) => *value,
        other => panic!("{path} must be a Boolean, got {other:?}"),
    }
}

fn field<'a>(
    object: &'a BTreeMap<String, WireJsonValue>,
    name: &str,
    path: &str,
) -> &'a WireJsonValue {
    object
        .get(name)
        .unwrap_or_else(|| panic!("{path} is missing field {name:?}"))
}

fn string_array<'a>(value: &'a WireJsonValue, path: &str) -> Vec<&'a str> {
    array(value, path)
        .iter()
        .enumerate()
        .map(|(index, value)| string(value, &format!("{path}[{index}]")))
        .collect()
}

fn boolean_array(value: &WireJsonValue, path: &str) -> Vec<bool> {
    array(value, path)
        .iter()
        .enumerate()
        .map(|(index, value)| boolean(value, &format!("{path}[{index}]")))
        .collect()
}

fn exact_fields(object: &BTreeMap<String, WireJsonValue>, expected: &[&str], path: &str) {
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(actual, expected, "{path} has the wrong exact field set");
}

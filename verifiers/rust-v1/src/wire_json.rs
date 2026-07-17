use core::fmt;
use std::collections::BTreeMap;

use crate::{
    checked_real_binary64_from_json, parse_json_number_lexeme, CheckedJsonBinary64, JsonNumberKind,
    JsonNumberLimits, NumericError, ParsedJsonNumber, DEFAULT_JSON_NUMBER_LIMITS,
};

/// Resource limits for the untrusted raw-v1 JSON byte layer.
///
/// String bytes are counted after JSON escape decoding and include object
/// keys. Array and object member limits apply independently to each container.
/// A nesting limit of zero admits scalars but no arrays or objects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WireJsonLimits {
    pub max_input_bytes: usize,
    pub max_nesting_depth: usize,
    pub max_total_nodes: usize,
    pub max_decoded_string_bytes: usize,
    pub max_array_members: usize,
    pub max_object_members: usize,
    pub number_limits: JsonNumberLimits,
}

pub const DEFAULT_WIRE_JSON_LIMITS: WireJsonLimits = WireJsonLimits {
    max_input_bytes: 16 * 1024 * 1024,
    max_nesting_depth: 128,
    max_total_nodes: 1_000_000,
    max_decoded_string_bytes: 8 * 1024 * 1024,
    max_array_members: 250_000,
    max_object_members: 250_000,
    number_limits: DEFAULT_JSON_NUMBER_LIMITS,
};

/// A strict RFC 8259 syntax tree retaining the wire distinction between
/// integer and real-valued number tokens.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WireJsonValue {
    Null,
    Bool(bool),
    String(String),
    Integer(ParsedJsonNumber),
    /// A real token together with its independently proved binary64 value.
    Real(CheckedJsonBinary64),
    Array(Vec<Self>),
    Object(BTreeMap<String, Self>),
}

/// Deterministic failures from strict JSON parsing and canonical validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WireJsonError {
    InputByteLimitExceeded,
    InvalidUtf8,
    ByteOrderMarkForbidden,
    UnexpectedEnd { offset: usize },
    UnexpectedByte { offset: usize, byte: u8 },
    TrailingData { offset: usize },
    InvalidLiteral { offset: usize },
    InvalidNumber { offset: usize, source: NumericError },
    UnescapedControlCharacter { offset: usize },
    InvalidEscape { offset: usize },
    InvalidUnicodeEscape { offset: usize },
    LoneUnicodeSurrogate { offset: usize },
    ExpectedObjectKey { offset: usize },
    ExpectedColon { offset: usize },
    ExpectedArrayDelimiter { offset: usize },
    ExpectedObjectDelimiter { offset: usize },
    DuplicateObjectKey { key: String },
    NestingDepthLimitExceeded,
    TotalNodeLimitExceeded,
    DecodedStringByteLimitExceeded,
    ArrayMemberLimitExceeded,
    ObjectMemberLimitExceeded,
    ScalarClassMismatch,
    NonCanonicalBytes { first_difference: usize },
}

impl fmt::Display for WireJsonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputByteLimitExceeded => formatter.write_str("JSON input byte limit exceeded"),
            Self::InvalidUtf8 => formatter.write_str("JSON input is not valid UTF-8"),
            Self::ByteOrderMarkForbidden => {
                formatter.write_str("a UTF-8 byte-order mark is forbidden")
            }
            Self::UnexpectedEnd { offset } => {
                write!(formatter, "unexpected end of JSON input at byte {offset}")
            }
            Self::UnexpectedByte { offset, byte } => {
                write!(
                    formatter,
                    "unexpected JSON byte 0x{byte:02x} at byte {offset}"
                )
            }
            Self::TrailingData { offset } => {
                write!(formatter, "trailing data after JSON value at byte {offset}")
            }
            Self::InvalidLiteral { offset } => {
                write!(formatter, "invalid JSON literal at byte {offset}")
            }
            Self::InvalidNumber { offset, source } => {
                write!(formatter, "invalid JSON number at byte {offset}: {source}")
            }
            Self::UnescapedControlCharacter { offset } => write!(
                formatter,
                "unescaped control character in JSON string at byte {offset}"
            ),
            Self::InvalidEscape { offset } => {
                write!(formatter, "invalid JSON string escape at byte {offset}")
            }
            Self::InvalidUnicodeEscape { offset } => {
                write!(formatter, "invalid JSON Unicode escape at byte {offset}")
            }
            Self::LoneUnicodeSurrogate { offset } => {
                write!(formatter, "lone JSON Unicode surrogate at byte {offset}")
            }
            Self::ExpectedObjectKey { offset } => {
                write!(formatter, "expected a JSON object key at byte {offset}")
            }
            Self::ExpectedColon { offset } => {
                write!(
                    formatter,
                    "expected ':' after JSON object key at byte {offset}"
                )
            }
            Self::ExpectedArrayDelimiter { offset } => write!(
                formatter,
                "expected ',' or ']' after JSON array member at byte {offset}"
            ),
            Self::ExpectedObjectDelimiter { offset } => write!(
                formatter,
                "expected ',' or '}}' after JSON object member at byte {offset}"
            ),
            Self::DuplicateObjectKey { key } => {
                write!(formatter, "duplicate decoded JSON object key {key:?}")
            }
            Self::NestingDepthLimitExceeded => {
                formatter.write_str("JSON nesting depth limit exceeded")
            }
            Self::TotalNodeLimitExceeded => formatter.write_str("JSON total node limit exceeded"),
            Self::DecodedStringByteLimitExceeded => {
                formatter.write_str("JSON decoded string byte limit exceeded")
            }
            Self::ArrayMemberLimitExceeded => {
                formatter.write_str("JSON array member limit exceeded")
            }
            Self::ObjectMemberLimitExceeded => {
                formatter.write_str("JSON object member limit exceeded")
            }
            Self::ScalarClassMismatch => {
                formatter.write_str("JSON syntax tree scalar class mismatch")
            }
            Self::NonCanonicalBytes { first_difference } => write!(
                formatter,
                "JSON bytes are not canonical; first difference at byte {first_difference}"
            ),
        }
    }
}

impl std::error::Error for WireJsonError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidNumber { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Parse exactly one UTF-8 RFC 8259 value while retaining every number
/// lexeme. RFC whitespace is accepted here; use [`validate_canonical_wire_json`]
/// when raw-v1 canonical bytes are required.
pub fn parse_wire_json(
    input: &[u8],
    limits: WireJsonLimits,
) -> Result<WireJsonValue, WireJsonError> {
    if input.len() > limits.max_input_bytes {
        return Err(WireJsonError::InputByteLimitExceeded);
    }
    if input.starts_with(&[0xef, 0xbb, 0xbf]) {
        return Err(WireJsonError::ByteOrderMarkForbidden);
    }
    let text = std::str::from_utf8(input).map_err(|_| WireJsonError::InvalidUtf8)?;
    let mut parser = Parser {
        input,
        text,
        index: 0,
        limits,
        total_nodes: 0,
        decoded_string_bytes: 0,
    };
    parser.skip_whitespace();
    let value = parser.parse_value(0)?;
    parser.skip_whitespace();
    if parser.index != input.len() {
        return Err(WireJsonError::TrailingData {
            offset: parser.index,
        });
    }
    Ok(value)
}

/// Parse a value and require exact equality with the raw-v1 CPython-v1
/// canonical serializer profile.
pub fn validate_canonical_wire_json(
    input: &[u8],
    limits: WireJsonLimits,
) -> Result<WireJsonValue, WireJsonError> {
    let value = parse_wire_json(input, limits)?;
    let canonical = to_canonical_bytes(&value)?;
    if input != canonical {
        let first_difference = input
            .iter()
            .zip(&canonical)
            .position(|(actual, expected)| actual != expected)
            .unwrap_or_else(|| input.len().min(canonical.len()));
        return Err(WireJsonError::NonCanonicalBytes { first_difference });
    }
    Ok(value)
}

/// Serialize an already checked syntax tree using the raw-v1 CPython-v1
/// canonical JSON profile.
pub fn to_canonical_bytes(value: &WireJsonValue) -> Result<Vec<u8>, WireJsonError> {
    let mut output = Vec::new();
    write_canonical_value(value, &mut output)?;
    Ok(output)
}

struct Parser<'a> {
    input: &'a [u8],
    text: &'a str,
    index: usize,
    limits: WireJsonLimits,
    total_nodes: usize,
    decoded_string_bytes: usize,
}

impl Parser<'_> {
    fn parse_value(&mut self, container_depth: usize) -> Result<WireJsonValue, WireJsonError> {
        self.charge_node()?;
        match self.input.get(self.index).copied() {
            Some(b'n') => {
                self.parse_literal(b"null")?;
                Ok(WireJsonValue::Null)
            }
            Some(b't') => {
                self.parse_literal(b"true")?;
                Ok(WireJsonValue::Bool(true))
            }
            Some(b'f') => {
                self.parse_literal(b"false")?;
                Ok(WireJsonValue::Bool(false))
            }
            Some(b'\"') => self.parse_string().map(WireJsonValue::String),
            Some(b'[') => {
                self.check_container_depth(container_depth)?;
                self.parse_array(container_depth + 1)
            }
            Some(b'{') => {
                self.check_container_depth(container_depth)?;
                self.parse_object(container_depth + 1)
            }
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(byte) => Err(WireJsonError::UnexpectedByte {
                offset: self.index,
                byte,
            }),
            None => Err(WireJsonError::UnexpectedEnd { offset: self.index }),
        }
    }

    fn parse_literal(&mut self, literal: &[u8]) -> Result<(), WireJsonError> {
        let start = self.index;
        if self.input.get(start..start + literal.len()) != Some(literal) {
            return Err(WireJsonError::InvalidLiteral { offset: start });
        }
        self.index += literal.len();
        Ok(())
    }

    fn parse_number(&mut self) -> Result<WireJsonValue, WireJsonError> {
        let start = self.index;
        while let Some(byte) = self.input.get(self.index).copied() {
            if matches!(byte, b',' | b']' | b'}' | b' ' | b'\t' | b'\n' | b'\r') {
                break;
            }
            self.index += 1;
        }
        let lexeme = &self.text[start..self.index];
        let parsed =
            parse_json_number_lexeme(lexeme, self.limits.number_limits).map_err(|source| {
                WireJsonError::InvalidNumber {
                    offset: start,
                    source,
                }
            })?;
        match parsed.kind() {
            JsonNumberKind::Integer => Ok(WireJsonValue::Integer(parsed)),
            JsonNumberKind::Real => {
                checked_real_binary64_from_json(lexeme, self.limits.number_limits)
                    .map(WireJsonValue::Real)
                    .map_err(|source| WireJsonError::InvalidNumber {
                        offset: start,
                        source,
                    })
            }
        }
    }

    fn parse_array(&mut self, container_depth: usize) -> Result<WireJsonValue, WireJsonError> {
        debug_assert_eq!(self.input.get(self.index), Some(&b'['));
        self.index += 1;
        self.skip_whitespace();
        let mut values = Vec::new();
        if self.consume_byte(b']') {
            return Ok(WireJsonValue::Array(values));
        }
        loop {
            if values.len() >= self.limits.max_array_members {
                return Err(WireJsonError::ArrayMemberLimitExceeded);
            }
            values.push(self.parse_value(container_depth)?);
            self.skip_whitespace();
            match self.input.get(self.index).copied() {
                Some(b',') => {
                    self.index += 1;
                    self.skip_whitespace();
                }
                Some(b']') => {
                    self.index += 1;
                    return Ok(WireJsonValue::Array(values));
                }
                Some(_) => {
                    return Err(WireJsonError::ExpectedArrayDelimiter { offset: self.index });
                }
                None => {
                    return Err(WireJsonError::UnexpectedEnd { offset: self.index });
                }
            }
        }
    }

    fn parse_object(&mut self, container_depth: usize) -> Result<WireJsonValue, WireJsonError> {
        debug_assert_eq!(self.input.get(self.index), Some(&b'{'));
        self.index += 1;
        self.skip_whitespace();
        let mut values = BTreeMap::new();
        if self.consume_byte(b'}') {
            return Ok(WireJsonValue::Object(values));
        }
        loop {
            if values.len() >= self.limits.max_object_members {
                return Err(WireJsonError::ObjectMemberLimitExceeded);
            }
            if self.input.get(self.index) != Some(&b'\"') {
                return Err(WireJsonError::ExpectedObjectKey { offset: self.index });
            }
            let key = self.parse_string()?;
            if values.contains_key(&key) {
                return Err(WireJsonError::DuplicateObjectKey { key });
            }
            self.skip_whitespace();
            if !self.consume_byte(b':') {
                return Err(WireJsonError::ExpectedColon { offset: self.index });
            }
            self.skip_whitespace();
            let value = self.parse_value(container_depth)?;
            values.insert(key, value);
            self.skip_whitespace();
            match self.input.get(self.index).copied() {
                Some(b',') => {
                    self.index += 1;
                    self.skip_whitespace();
                }
                Some(b'}') => {
                    self.index += 1;
                    return Ok(WireJsonValue::Object(values));
                }
                Some(_) => {
                    return Err(WireJsonError::ExpectedObjectDelimiter { offset: self.index });
                }
                None => {
                    return Err(WireJsonError::UnexpectedEnd { offset: self.index });
                }
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, WireJsonError> {
        debug_assert_eq!(self.input.get(self.index), Some(&b'\"'));
        self.index += 1;
        let mut decoded = String::new();
        loop {
            let byte = self
                .input
                .get(self.index)
                .copied()
                .ok_or(WireJsonError::UnexpectedEnd { offset: self.index })?;
            match byte {
                b'\"' => {
                    self.index += 1;
                    return Ok(decoded);
                }
                b'\\' => self.parse_escape(&mut decoded)?,
                0x00..=0x1f => {
                    return Err(WireJsonError::UnescapedControlCharacter { offset: self.index });
                }
                0x20..=0x7f => {
                    self.push_decoded_char(&mut decoded, char::from(byte))?;
                    self.index += 1;
                }
                _ => {
                    let character = self.text[self.index..]
                        .chars()
                        .next()
                        .expect("the complete input was validated as UTF-8");
                    self.push_decoded_char(&mut decoded, character)?;
                    self.index += character.len_utf8();
                }
            }
        }
    }

    fn parse_escape(&mut self, decoded: &mut String) -> Result<(), WireJsonError> {
        let slash_offset = self.index;
        self.index += 1;
        let escape = self
            .input
            .get(self.index)
            .copied()
            .ok_or(WireJsonError::UnexpectedEnd { offset: self.index })?;
        self.index += 1;
        let character = match escape {
            b'\"' => '\"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{0008}',
            b'f' => '\u{000c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => {
                let first_offset = self.index;
                let first = self.parse_hex_quad(first_offset)?;
                if (0xd800..=0xdbff).contains(&first) {
                    if self.input.get(self.index..self.index + 2) != Some(b"\\u") {
                        return Err(WireJsonError::LoneUnicodeSurrogate {
                            offset: first_offset,
                        });
                    }
                    self.index += 2;
                    let second_offset = self.index;
                    let second = self.parse_hex_quad(second_offset)?;
                    if !(0xdc00..=0xdfff).contains(&second) {
                        return Err(WireJsonError::LoneUnicodeSurrogate {
                            offset: second_offset,
                        });
                    }
                    let scalar = 0x1_0000
                        + ((u32::from(first) - 0xd800) << 10)
                        + (u32::from(second) - 0xdc00);
                    char::from_u32(scalar).expect("a paired surrogate is a Unicode scalar")
                } else if (0xdc00..=0xdfff).contains(&first) {
                    return Err(WireJsonError::LoneUnicodeSurrogate {
                        offset: first_offset,
                    });
                } else {
                    char::from_u32(u32::from(first)).expect("a non-surrogate u16 is a scalar")
                }
            }
            _ => {
                return Err(WireJsonError::InvalidEscape {
                    offset: slash_offset,
                })
            }
        };
        self.push_decoded_char(decoded, character)
    }

    fn parse_hex_quad(&mut self, offset: usize) -> Result<u16, WireJsonError> {
        let mut value = 0_u16;
        for _ in 0..4 {
            let byte = self
                .input
                .get(self.index)
                .copied()
                .ok_or(WireJsonError::InvalidUnicodeEscape { offset })?;
            let digit = match byte {
                b'0'..=b'9' => u16::from(byte - b'0'),
                b'a'..=b'f' => u16::from(byte - b'a') + 10,
                b'A'..=b'F' => u16::from(byte - b'A') + 10,
                _ => return Err(WireJsonError::InvalidUnicodeEscape { offset }),
            };
            value = value * 16 + digit;
            self.index += 1;
        }
        Ok(value)
    }

    fn push_decoded_char(
        &mut self,
        decoded: &mut String,
        character: char,
    ) -> Result<(), WireJsonError> {
        self.decoded_string_bytes = self
            .decoded_string_bytes
            .checked_add(character.len_utf8())
            .ok_or(WireJsonError::DecodedStringByteLimitExceeded)?;
        if self.decoded_string_bytes > self.limits.max_decoded_string_bytes {
            return Err(WireJsonError::DecodedStringByteLimitExceeded);
        }
        decoded.push(character);
        Ok(())
    }

    fn charge_node(&mut self) -> Result<(), WireJsonError> {
        self.total_nodes = self
            .total_nodes
            .checked_add(1)
            .ok_or(WireJsonError::TotalNodeLimitExceeded)?;
        if self.total_nodes > self.limits.max_total_nodes {
            return Err(WireJsonError::TotalNodeLimitExceeded);
        }
        Ok(())
    }

    fn check_container_depth(&self, container_depth: usize) -> Result<(), WireJsonError> {
        if container_depth >= self.limits.max_nesting_depth {
            return Err(WireJsonError::NestingDepthLimitExceeded);
        }
        Ok(())
    }

    fn skip_whitespace(&mut self) {
        while matches!(
            self.input.get(self.index),
            Some(b' ' | b'\t' | b'\n' | b'\r')
        ) {
            self.index += 1;
        }
    }

    fn consume_byte(&mut self, expected: u8) -> bool {
        if self.input.get(self.index) == Some(&expected) {
            self.index += 1;
            true
        } else {
            false
        }
    }
}

fn write_canonical_value(value: &WireJsonValue, output: &mut Vec<u8>) -> Result<(), WireJsonError> {
    match value {
        WireJsonValue::Null => output.extend_from_slice(b"null"),
        WireJsonValue::Bool(false) => output.extend_from_slice(b"false"),
        WireJsonValue::Bool(true) => output.extend_from_slice(b"true"),
        WireJsonValue::String(value) => write_canonical_string(value, output),
        WireJsonValue::Integer(value) => {
            if value.kind() != JsonNumberKind::Integer {
                return Err(WireJsonError::ScalarClassMismatch);
            }
            output.extend_from_slice(value.exact_decimal().numer().to_string().as_bytes());
        }
        WireJsonValue::Real(value) => {
            output.extend_from_slice(cpython_v1_float_string(value).as_bytes());
        }
        WireJsonValue::Array(values) => {
            output.push(b'[');
            for (index, item) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_canonical_value(item, output)?;
            }
            output.push(b']');
        }
        WireJsonValue::Object(values) => {
            output.push(b'{');
            for (index, (key, item)) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_canonical_string(key, output);
                output.push(b':');
                write_canonical_value(item, output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

fn write_canonical_string(value: &str, output: &mut Vec<u8>) {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    output.push(b'\"');
    for character in value.chars() {
        match character {
            '\"' => output.extend_from_slice(b"\\\""),
            '\\' => output.extend_from_slice(b"\\\\"),
            '\u{0008}' => output.extend_from_slice(b"\\b"),
            '\t' => output.extend_from_slice(b"\\t"),
            '\n' => output.extend_from_slice(b"\\n"),
            '\u{000c}' => output.extend_from_slice(b"\\f"),
            '\r' => output.extend_from_slice(b"\\r"),
            '\u{0000}'..='\u{001f}' => {
                let byte = character as u8;
                output.extend_from_slice(b"\\u00");
                output.push(HEX[usize::from(byte >> 4)]);
                output.push(HEX[usize::from(byte & 0x0f)]);
            }
            _ => {
                let mut encoded = [0_u8; 4];
                output.extend_from_slice(character.encode_utf8(&mut encoded).as_bytes());
            }
        }
    }
    output.push(b'\"');
}

fn cpython_v1_float_string(value: &CheckedJsonBinary64) -> String {
    // OPEN-V1-01 does not define a portable normative float spelling. Rust's
    // finite-f64 Debug output supplies the shortest digits and the same
    // fixed/scientific cutoff here; the exponent syntax is normalized below.
    // This is an implementation choice that must remain guarded by a pinned
    // release toolchain and the corpus plus all-exponent cross-checks below.
    let rendered = format!("{:?}", value.value());
    let Some(exponent_marker) = rendered.find('e') else {
        return rendered;
    };

    let (significand, exponent_with_marker) = rendered.split_at(exponent_marker);
    let exponent = &exponent_with_marker[1..];
    let (sign, digits) = match exponent.strip_prefix('-') {
        Some(digits) => ('-', digits),
        None => ('+', exponent.strip_prefix('+').unwrap_or(exponent)),
    };
    let mut canonical = String::with_capacity(rendered.len() + 2);
    canonical.push_str(significand);
    canonical.push('e');
    canonical.push(sign);
    if digits.len() < 2 {
        canonical.push('0');
    }
    canonical.push_str(digits);
    canonical
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFORMANCE_SUCCESS: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/raw-v1/inputs/success.raw.json"
    ));
    const CONFORMANCE_FAILED_REVISIT: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/raw-v1/inputs/failed-revisit.raw.json"
    ));

    #[derive(Default)]
    struct AstProfile {
        total_nodes: usize,
        integer_numbers: usize,
        real_numbers: usize,
        decoded_string_bytes: usize,
        max_nesting_depth: usize,
        max_array_members: usize,
        max_object_members: usize,
    }

    fn parse(input: &[u8]) -> Result<WireJsonValue, WireJsonError> {
        parse_wire_json(input, DEFAULT_WIRE_JSON_LIMITS)
    }

    fn canonical(input: &[u8]) -> Result<WireJsonValue, WireJsonError> {
        validate_canonical_wire_json(input, DEFAULT_WIRE_JSON_LIMITS)
    }

    fn canonicalized(input: &[u8]) -> Vec<u8> {
        to_canonical_bytes(&parse(input).unwrap()).unwrap()
    }

    fn profile_and_check_reals(
        value: &WireJsonValue,
        container_depth: usize,
        profile: &mut AstProfile,
    ) {
        profile.total_nodes += 1;
        match value {
            WireJsonValue::Null | WireJsonValue::Bool(_) => {}
            WireJsonValue::String(value) => profile.decoded_string_bytes += value.len(),
            WireJsonValue::Integer(_) => profile.integer_numbers += 1,
            WireJsonValue::Real(value) => {
                profile.real_numbers += 1;
                let rendered = cpython_v1_float_string(value);
                let reparsed = parse(rendered.as_bytes()).unwrap();
                let WireJsonValue::Real(reparsed_real) = &reparsed else {
                    panic!("real renderer changed the scalar class: {rendered}")
                };
                assert_eq!(
                    reparsed_real.bits(),
                    value.bits(),
                    "real renderer changed binary64 bits for {}",
                    value.lexeme()
                );
                assert_eq!(
                    to_canonical_bytes(&reparsed).unwrap(),
                    rendered.as_bytes(),
                    "real renderer was not idempotent for {}",
                    value.lexeme()
                );
            }
            WireJsonValue::Array(values) => {
                let nested_depth = container_depth + 1;
                profile.max_nesting_depth = profile.max_nesting_depth.max(nested_depth);
                profile.max_array_members = profile.max_array_members.max(values.len());
                for item in values {
                    profile_and_check_reals(item, nested_depth, profile);
                }
            }
            WireJsonValue::Object(values) => {
                let nested_depth = container_depth + 1;
                profile.max_nesting_depth = profile.max_nesting_depth.max(nested_depth);
                profile.max_object_members = profile.max_object_members.max(values.len());
                for (key, item) in values {
                    profile.decoded_string_bytes += key.len();
                    profile_and_check_reals(item, nested_depth, profile);
                }
            }
        }
    }

    #[test]
    fn parses_all_scalar_classes_and_retains_number_lexemes() {
        let value = parse(br#"[null,true,false,"text",123,-0,1.25e+2]"#).unwrap();
        let WireJsonValue::Array(values) = value else {
            panic!("expected an array")
        };
        assert!(matches!(values[0], WireJsonValue::Null));
        assert!(matches!(values[1], WireJsonValue::Bool(true)));
        assert!(matches!(values[2], WireJsonValue::Bool(false)));
        assert!(matches!(&values[3], WireJsonValue::String(text) if text == "text"));
        assert!(matches!(&values[4], WireJsonValue::Integer(number) if number.lexeme() == "123"));
        assert!(matches!(&values[5], WireJsonValue::Integer(number) if number.lexeme() == "-0"));
        assert!(matches!(&values[6], WireJsonValue::Real(number)
            if number.lexeme() == "1.25e+2" && number.bits() == 125_f64.to_bits()));
    }

    #[test]
    fn rejects_malformed_json_and_nonfinite_or_overflowing_numbers() {
        for input in [
            b"".as_slice(),
            b"nul".as_slice(),
            b"[1,]".as_slice(),
            b"{\"a\" 1}".as_slice(),
            b"01".as_slice(),
            b"+1".as_slice(),
            b"NaN".as_slice(),
            b"Infinity".as_slice(),
            b"true false".as_slice(),
        ] {
            assert!(parse(input).is_err(), "unexpected acceptance of {input:?}");
        }
        assert!(matches!(
            parse(b"1e309"),
            Err(WireJsonError::InvalidNumber {
                source: NumericError::Binary64Overflow,
                ..
            })
        ));
        assert_eq!(
            parse(&[b'\"', 0xff, b'\"']),
            Err(WireJsonError::InvalidUtf8)
        );
        assert_eq!(
            parse(b"\xef\xbb\xbfnull"),
            Err(WireJsonError::ByteOrderMarkForbidden)
        );
        assert!(matches!(
            parse(&[b'\"', 0x1f, b'\"']),
            Err(WireJsonError::UnescapedControlCharacter { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_decoded_keys_at_every_depth() {
        assert_eq!(
            parse(br#"{"a":1,"a":2}"#),
            Err(WireJsonError::DuplicateObjectKey { key: "a".into() })
        );
        assert_eq!(
            parse(br#"{"outer":{"a":1,"\u0061":2}}"#),
            Err(WireJsonError::DuplicateObjectKey { key: "a".into() })
        );
    }

    #[test]
    fn decodes_unicode_escapes_and_requires_well_formed_surrogate_pairs() {
        let decoded = parse(r#"["é","\u00e9","\ud834\udd1e"]"#.as_bytes()).unwrap();
        assert_eq!(
            decoded,
            WireJsonValue::Array(vec![
                WireJsonValue::String("é".into()),
                WireJsonValue::String("é".into()),
                WireJsonValue::String("𝄞".into()),
            ])
        );
        for input in [
            br#""\ud800""#.as_slice(),
            br#""\ud800\u0041""#.as_slice(),
            br#""\udc00""#.as_slice(),
            br#""\ud800\ud800""#.as_slice(),
        ] {
            assert!(matches!(
                parse(input),
                Err(WireJsonError::LoneUnicodeSurrogate { .. })
            ));
        }
        assert!(matches!(
            parse(br#""\u12x4""#),
            Err(WireJsonError::InvalidUnicodeEscape { .. })
        ));
        assert!(matches!(
            parse(br#""\q""#),
            Err(WireJsonError::InvalidEscape { .. })
        ));
    }

    #[test]
    fn canonical_strings_are_utf8_direct_with_cpython_control_escapes() {
        let input = r#""é/\"\\\b\t\n\f\r\u0000\u000b""#.as_bytes();
        let value = parse(input).unwrap();
        assert_eq!(to_canonical_bytes(&value).unwrap(), input);
        assert_eq!(canonical(input), Ok(value));
        assert_eq!(canonicalized(br#""\u00e9\/""#), "\"é/\"".as_bytes());
    }

    #[test]
    fn object_keys_use_python_unicode_code_point_order() {
        let input = "{\"𐀀\":4,\"\":3,\"é\":2,\"z\":1}".as_bytes();
        assert_eq!(
            canonicalized(input),
            "{\"z\":1,\"é\":2,\"\":3,\"𐀀\":4}".as_bytes()
        );
        assert!(canonical(input).is_err());
        assert!(canonical("{\"z\":1,\"é\":2,\"\":3,\"𐀀\":4}".as_bytes()).is_ok());
    }

    #[test]
    fn cpython_v1_number_rendering_is_not_original_lexeme_reemission() {
        for (input, expected) in [
            ("-0", "0"),
            ("1e0", "1.0"),
            ("-0e0", "-0.0"),
            ("0.0", "0.0"),
            ("-0.0", "-0.0"),
            ("1e-4", "0.0001"),
            ("1e-5", "1e-05"),
            ("1e-6", "1e-06"),
            ("1e-7", "1e-07"),
            ("1e15", "1000000000000000.0"),
            ("1e16", "1e+16"),
            ("1E+16", "1e+16"),
            ("5e-324", "5e-324"),
            ("1.7976931348623157e308", "1.7976931348623157e+308"),
        ] {
            assert_eq!(
                canonicalized(input.as_bytes()),
                expected.as_bytes(),
                "unexpected canonical form for {input}"
            );
            assert!(
                canonical(expected.as_bytes()).is_ok(),
                "canonical renderer output was rejected for {input}"
            );
        }
    }

    #[test]
    fn manually_misclassified_number_ast_fails_closed() {
        let real = parse_json_number_lexeme("1.5", DEFAULT_JSON_NUMBER_LIMITS).unwrap();
        assert_eq!(
            to_canonical_bytes(&WireJsonValue::Integer(real)),
            Err(WireJsonError::ScalarClassMismatch)
        );
    }

    #[test]
    fn canonical_validation_rejects_any_byte_level_difference() {
        for input in [
            b" null".as_slice(),
            b"null ".as_slice(),
            b"null\n".as_slice(),
            b"[1, 2]".as_slice(),
            br#"{"b":1,"a":2}"#.as_slice(),
            br#""\u0061""#.as_slice(),
            br#""\/""#.as_slice(),
            b"-0".as_slice(),
            b"1e0".as_slice(),
        ] {
            assert!(matches!(
                canonical(input),
                Err(WireJsonError::NonCanonicalBytes { .. })
            ));
        }
        assert!(canonical("{\"a\":2,\"b\":1,\"é\":\"𝄞\"}".as_bytes()).is_ok());
    }

    #[test]
    fn every_parser_resource_limit_is_enforced() {
        let input_limit = WireJsonLimits {
            max_input_bytes: 3,
            ..DEFAULT_WIRE_JSON_LIMITS
        };
        assert_eq!(
            parse_wire_json(b"null", input_limit),
            Err(WireJsonError::InputByteLimitExceeded)
        );

        let depth_limit = WireJsonLimits {
            max_nesting_depth: 1,
            ..DEFAULT_WIRE_JSON_LIMITS
        };
        assert_eq!(
            parse_wire_json(b"[[]]", depth_limit),
            Err(WireJsonError::NestingDepthLimitExceeded)
        );

        let node_limit = WireJsonLimits {
            max_total_nodes: 1,
            ..DEFAULT_WIRE_JSON_LIMITS
        };
        assert_eq!(
            parse_wire_json(b"[null]", node_limit),
            Err(WireJsonError::TotalNodeLimitExceeded)
        );

        let string_limit = WireJsonLimits {
            max_decoded_string_bytes: 3,
            ..DEFAULT_WIRE_JSON_LIMITS
        };
        assert_eq!(
            parse_wire_json("{\"é\":\"é\"}".as_bytes(), string_limit),
            Err(WireJsonError::DecodedStringByteLimitExceeded)
        );

        let array_limit = WireJsonLimits {
            max_array_members: 1,
            ..DEFAULT_WIRE_JSON_LIMITS
        };
        assert_eq!(
            parse_wire_json(b"[0,1]", array_limit),
            Err(WireJsonError::ArrayMemberLimitExceeded)
        );

        let object_limit = WireJsonLimits {
            max_object_members: 1,
            ..DEFAULT_WIRE_JSON_LIMITS
        };
        assert_eq!(
            parse_wire_json(br#"{"a":0,"b":1}"#, object_limit),
            Err(WireJsonError::ObjectMemberLimitExceeded)
        );

        let number_limit = WireJsonLimits {
            number_limits: JsonNumberLimits {
                max_significand_digits: 1,
                ..DEFAULT_JSON_NUMBER_LIMITS
            },
            ..DEFAULT_WIRE_JSON_LIMITS
        };
        assert!(matches!(
            parse_wire_json(b"12", number_limit),
            Err(WireJsonError::InvalidNumber {
                source: NumericError::JsonNumberSignificandDigitLimitExceeded,
                ..
            })
        ));
    }

    #[test]
    fn conformance_raw_inputs_are_canonical_and_have_resource_headroom() {
        for (name, input) in [
            ("success.raw.json", CONFORMANCE_SUCCESS),
            ("failed-revisit.raw.json", CONFORMANCE_FAILED_REVISIT),
        ] {
            let value = validate_canonical_wire_json(input, DEFAULT_WIRE_JSON_LIMITS)
                .unwrap_or_else(|error| panic!("{name} was rejected: {error}"));
            assert_eq!(
                to_canonical_bytes(&value).unwrap(),
                input,
                "{name} changed on canonical reserialization"
            );

            let mut profile = AstProfile::default();
            profile_and_check_reals(&value, 0, &mut profile);
            assert!(profile.total_nodes < DEFAULT_WIRE_JSON_LIMITS.max_total_nodes);
            assert!(
                profile.decoded_string_bytes < DEFAULT_WIRE_JSON_LIMITS.max_decoded_string_bytes
            );
            assert!(profile.max_nesting_depth < DEFAULT_WIRE_JSON_LIMITS.max_nesting_depth);
            assert!(profile.max_array_members < DEFAULT_WIRE_JSON_LIMITS.max_array_members);
            assert!(profile.max_object_members < DEFAULT_WIRE_JSON_LIMITS.max_object_members);

            println!(
                "{name}: numbers={} (integer={}, real={}); input={} (headroom={}); \
                 nodes={} (headroom={}); depth={} (headroom={}); decoded-string-bytes={} \
                 (headroom={}); max-array-members={} (headroom={}); max-object-members={} \
                 (headroom={})",
                profile.integer_numbers + profile.real_numbers,
                profile.integer_numbers,
                profile.real_numbers,
                input.len(),
                DEFAULT_WIRE_JSON_LIMITS.max_input_bytes - input.len(),
                profile.total_nodes,
                DEFAULT_WIRE_JSON_LIMITS.max_total_nodes - profile.total_nodes,
                profile.max_nesting_depth,
                DEFAULT_WIRE_JSON_LIMITS.max_nesting_depth - profile.max_nesting_depth,
                profile.decoded_string_bytes,
                DEFAULT_WIRE_JSON_LIMITS.max_decoded_string_bytes - profile.decoded_string_bytes,
                profile.max_array_members,
                DEFAULT_WIRE_JSON_LIMITS.max_array_members - profile.max_array_members,
                profile.max_object_members,
                DEFAULT_WIRE_JSON_LIMITS.max_object_members - profile.max_object_members,
            );
        }
    }

    #[test]
    fn cpython_renderer_stress_covers_all_finite_exponents_and_both_signs() {
        const FINITE_EXPONENT_COUNT: u64 = 2047;
        const FRACTION_MASK: u64 = (1_u64 << 52) - 1;

        let mut state = 0x7d34_2c81_a65f_91e3_u64;
        for sign in [0_u64, 1_u64 << 63] {
            for exponent_field in 0..FINITE_EXPONENT_COUNT {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let bits = sign | (exponent_field << 52) | (state & FRACTION_MASK);
                let source = format!("{:?}", f64::from_bits(bits));
                let checked = checked_real_binary64_from_json(&source, DEFAULT_JSON_NUMBER_LIMITS)
                    .unwrap_or_else(|error| {
                        panic!("bits 0x{bits:016x} source {source:?} failed proof: {error}")
                    });
                assert_eq!(checked.bits(), bits);

                let rendered = cpython_v1_float_string(&checked);
                let reparsed = parse(rendered.as_bytes()).unwrap();
                let WireJsonValue::Real(reparsed_real) = &reparsed else {
                    panic!("bits 0x{bits:016x} rendered as non-real {rendered:?}")
                };
                assert_eq!(reparsed_real.bits(), bits);
                assert_eq!(to_canonical_bytes(&reparsed).unwrap(), rendered.as_bytes());
            }
        }
    }
}

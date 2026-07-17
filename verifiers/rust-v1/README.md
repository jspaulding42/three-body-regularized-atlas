# Rust verifier v1

This standalone crate is the independent verifier's numeric foundation. The
current slice contains only:

- exact decoding of finite IEEE-754 binary64 bit patterns;
- manual, resource-bounded RFC 8259 number parsing and exact nearest-even
  decimal-to-binary64 proof, retaining the exact decimal lexeme value
  separately from the theorem-facing exact binary64 dyadic;
- arbitrary-size exact rational intervals and Horner evaluation; and
- integer-arithmetic dyadic square-root enclosures with exact postconditions.

It does not parse certificates, call Python, check three-body equations, or
make any certificate-level claim.

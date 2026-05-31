# Nonzero-Angular All-Pair Binary Coverage Lemma

Nonzero centered angular momentum excludes total collision, but it does not
select which binary pair may collide.  In a labeled three-body problem the
possible binary collision pairs are

```text
(0,1), (0,2), (1,2).
```

A Levi-Civita chart for one pair regularizes only that pair's relative
coordinate.  For example, a chart built around `r_01=z^2` gives analytic
coordinates through `q_0=q_1` while the third body is separated.  It does not
regularize the different singular hypersurfaces `q_0=q_2` or `q_1=q_2`.

Therefore an arbitrary-data nonzero-angular theorem has two options:

1. include separated-binary chart families for all three pairs in every
   compact-time endpoint recurrence; or
2. provide a separate constructor-derived regime classifier proving that only a
   proper subset of pairs can occur on the branch being certified.

Without the second classifier, a one-pair event recurrence is only a scoped
subregime.  It may prove that the tail arithmetic closes for branches whose
binary events all use that pair, but it cannot certify the all-time
nonzero-angular theorem for arbitrary labeled initial data.

The executable obligation is
`nonzero_angular_all_pair_binary_coverage`.  It checks each time direction of
the two-sided event recurrence.  A pair-specific family
`separated_binary_levi_civita_01`, `_02`, or `_12` covers its named pair.  A
generic `separated_binary_levi_civita` family covers only the pair recorded by
its constructor source certificate.  The `all_time_nonzero_angular` path
certifies only when each endpoint recurrence covers all three pairs, while the
single-pair recurrence remains available as an explicit incomplete scoped
regime hypothesis.

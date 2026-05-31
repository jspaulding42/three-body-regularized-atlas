# Recursive Stratified Branch/Event Consumption Theorem

This note records the finite recursive theorem used by the set-valued
finite-target constructor.  It is deliberately narrower than the missing
arbitrary interval-input partition theorem: it consumes a finite stratified
tree that has already been constructed by a supported branch/event partition
constructor, and it refuses unsupported analytic strata.

## Theorem

Let `S` be a finite branch/event-order tree over a compact represented input
set.  Every leaf of `S` is labeled as exactly one of:

- a positive-margin terminal leaf, such as a unique event, no-event response, or
  separated-binary entry;
- a proof-certified terminal singular/policy leaf, such as a selector policy or
  total-collision stop cluster;
- an equality leaf carrying an explicit semi-analytic stratum
  `f_1=...=f_k=0` together with a recursive child tree on that stratum;
- an unsupported analytic stratum.

Assume each terminal leaf has a proof-certified atlas/stop/selector response.
Assume each recursive equality leaf has a child recursive-consumption
certificate whose root measure is strictly smaller in the lexicographic measure

```text
mu = (dimension, rank),
```

where dimension decreases first, and rank is used only among strata of the same
dimension.  Then the whole finite tree is consumed by finite union and ledger
aggregation.  If any leaf is unsupported, any terminal response is missing, or
any child fails strict dimension/rank descent, the theorem returns a named
obligation instead of certifying the tree.

## Proof

The proof is induction on the finite recursive tree.  A terminal leaf is
consumed directly by its local proof certificate.  A recursive equality leaf is
not certified as a terminal box: the equality constraints remain part of the
child input.  Since the child has smaller `(dimension, rank)`, recursive descent
cannot cycle along a branch.  Since the displayed tree is finite, every branch
therefore reaches a terminal proof-certified leaf or a named unsupported leaf.

For a certified branch, the finite union of leaf responses covers the parent
input set because the parent source tree covers the represented input set and
the stratified leaves preserve the source-leaf ids.  The atlas, stop,
invariant, residual, projection, and tail ledgers are then aggregated leafwise.
No equality stratum is hulled back into an ambient box during this argument.

For constructor-derived arrangements, the same proof applies after the
arrangement constructor has produced the finite source tree.  The currently
implemented constructor sources include verified polynomial root brackets,
affine coefficient-derived roots, quadratic simple/double-root strata, exact
Sturm-isolated polynomial arrangements, axis-aligned affine box arrangements,
and 2D/3D oblique affine halfspace arrangements.  Their equality leaves remain
semi-analytic cells or slabs with recorded defining functions, and their child
certificates must descend in dimension or rank.

Axis-aligned affine box arrangements now have their own automatic child
constructor.  Each equality slab records coordinate-affine defining functions
whose roots fix one or more coordinate axes.  The constructor recovers those
exact coordinate roots from the affine coefficients, groups coincident
same-axis defining ids, leaves the remaining coordinate intervals as the
lower-dimensional domain, and emits a terminal child of dimension equal to the
number of free coordinates.  This consumes represented coordinate slabs and
coordinate intersections without supplied placeholder children; oblique
hyperplanes remain outside the axis-aligned grammar and are handled only by the
separate affine halfspace constructors.

The single affine halfspace decision constructor now has the matching automatic
child for its central equality slab.  The parent tree still represents the
search partition as `f <= -epsilon`, `|f| <= epsilon`, and `f >= epsilon`, but
the recursive child is tied to the exact hyperplane `f=0`.  In dimensions one,
two, and three the constructor verifies the point, line segment, or plane slice
inside the parent box and emits a terminal lower-dimensional
`AffineHalfspaceDecisionChild`.  This closes the represented one-discriminator
oblique slab case without hulling the slab back into ambient boxes.

The affine recursive-consumption adapters now invoke their automatic child
constructors by default when no child map is supplied and the displayed parent
stratification is proof-certified.  This covers axis-aligned affine box
arrangements, single affine halfspace decisions, two-dimensional oblique
affine halfspace arrangements, and three-dimensional oblique affine halfspace
arrangements.  Explicit partial child maps remain partial, so omitted equality
leaves are still reported as unresolved rather than silently filled.

One-dimensional polynomial decisions and polynomial decision arrangements now
have an automatic root-child constructor for their certified equality strata.
Whenever the parent constructor has already isolated an equality root by a
simple-root bracket, a tangent double-root certificate, a grouped
coincident/multiple-root certificate, or a Sturm multiplicity certificate, the
child constructor emits a terminal zero-dimensional `PolynomialRootChild`
recording the root interval, value interval, derivative/second-derivative data
when available, and multiplicity data when present.  This removes another class
of supplied positive-margin placeholder children while keeping the represented
scope explicit: the root child consumes only equality strata already certified
by the displayed one-dimensional polynomial arrangement.  The polynomial
recursive-consumption adapters now invoke this root-child constructor
automatically when no child map is supplied and the parent polynomial
stratification is proof-certified; partial child maps remain partial, so omitted
equality leaves are still reported as unresolved.

The first automatic lower-dimensional child constructor is implemented for
one geometric equality line in a 2D oblique affine halfspace arrangement.  Given
a parent cell with one defining affine equation `f=0`, or several coincident
defining affine equations for the same line, the constructor parameterizes the
exact line, intersects that parameter line with the parent box and the cell's
remaining sign inequalities, pulls every nondefining affine decision back to a
one-dimensional affine discriminator, and then uses the 1D affine arrangement
constructor on that line.  This closes parent equality slabs whose
line-restricted child arrangement is already terminal after coefficient root
construction, including coincident-boundary slabs that record multiple decision
ids on the same line.  If there are no remaining decisions on the equality
line, the exact line stratum itself is emitted as a terminal one-dimensional
child.  Parallel but distinct affine boundaries are not collapsed by this
constructor.  If the restricted child contains equality roots, the line-child
constructor now feeds the restricted affine arrangement through
`derive_polynomial_decision_arrangement_child_consumptions(...)`, so those
one-dimensional roots are consumed by terminal `PolynomialRootChild`
certificates rather than by supplied positive-margin placeholders.

The companion zero-dimensional child constructor handles independent
two-boundary cells in the same 2D affine arrangement.  When two affine equations
have a nonzero determinant, their exact intersection point is computed,
checked against the parent box and the cell's sign vector, and emitted as a
terminal zero-dimensional child.  Combining the line and point constructors
consumes a transverse two-line arrangement without manually supplied child
certificates: the four one-boundary slab cells descend to terminal line
children, and the central two-boundary cell descends to the terminal point
child.

The first spatial child constructor handles the terminal one-geometric-plane
case for a 3D oblique affine arrangement.  Given a parent cell with one
defining affine plane, or several coincident defining affine equations for the
same plane, the constructor computes the exact plane, intersects it with the
parent box by enumerating box-edge crossings, projects the slice to plane
coordinates, clips that polygon by the remaining fixed sign inequalities,
checks that the induced two-dimensional polygon has positive area, and emits
that exact plane stratum as a terminal two-dimensional child.  Duplicate
halfspace constraints are deduplicated in the 3D polyhedron volume check so
coincident decision ids do not break the represented cover certificate.  This
removes manual child evidence for represented single-plane or same-plane
equality cells whose other decisions are already fixed to strict signs.
Parallel but distinct planes are not collapsed by this constructor.

The same spatial arrangement layer now handles independent two-plane and
three-plane equality cells.  Two independent affine planes are intersected to
an exact line, clipped against the parent box and remaining sign inequalities,
and either emitted as a terminal one-dimensional spatial line or, when other
affine decisions remain along the line, consumed through the one-dimensional
affine arrangement constructor.  Three independent affine planes are solved as
an exact point, checked against the parent box and sign vector, and emitted as
a terminal zero-dimensional spatial point child.  These are represented affine
subcases; lower-dimensional cells outside the finite affine arrangement grammar
still need explicit child certificates.

## Boundary

This theorem does not prove arbitrary interval-input recursive partition
generation.  It assumes a finite stratified tree is supplied or produced by a
supported constructor.  The stronger missing theorem must still derive the
appropriate branch/event tree from arbitrary interval input data, including
simultaneous first events, ambiguous binary pairs, selector boundaries, and
total-collision clusters.

## Executable Evidence

`RecursiveStratifiedBranchEventConsumptionCertificate` records:

- `terminal_leaf_count`,
- `recursive_leaf_count`,
- `strict_descent_edge_count`,
- `unresolved_descent_edge_count`,
- `descent_well_founded`,
- `unsupported_leaf_count`,
- `node_count`,
- `recursion_depth`.

The certificate is proof-certified only when every recursive edge is a strict
dimension/rank descent, every terminal leaf is locally certified, unsupported
strata are absent, and supplied child ids match equality leaves of the source
tree.

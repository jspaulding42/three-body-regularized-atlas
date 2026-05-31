# Research Note: Unit-Distance Principles Applied to Three-Body Orbits

## Source Principle

OpenAI's unit-distance result disproves Erdos's expected `n^(1+o(1))` upper behavior by constructing infinitely many point sets with at least `n^(1+delta)` unit distances. The proof does not simply optimize a square grid. It moves the problem into number fields with many symmetries, builds many norm-one elements in a high-dimensional Minkowski lattice, intersects with a controlled region, and then projects back to one complex coordinate where those elements become unit planar segments.

The transferable research pattern is:

- Work constructively, even when the community expects a negative or near-optimal barrier.
- Lift the problem into a representation where hidden symmetry is abundant.
- Separate existence/construction from projection/verification.
- Verify the projected object in the original problem, rather than trusting the lifted story.

## Three-Body Translation

The unrestricted Newtonian three-body problem is not made integrable by this idea. A realistic target is a structured family: planar equal-mass periodic orbits.

For a choreography, each body follows the same planar curve with a one-third period shift:

```text
q1(t) = q(t)
q2(t) = q(t + T/3)
q3(t) = q(t + 2T/3)
```

Using Fourier modes, the center-of-mass identity

```text
q(t) + q(t + T/3) + q(t + 2T/3) = 0
```

is automatic when harmonics divisible by `3` are excluded. The figure-eight's half-turn symmetry

```text
q(t + T/2) = -q(t)
```

is automatic when even harmonics are excluded. The permitted harmonic indices are therefore:

```text
n odd and 3 does not divide n
```

This is the small analogue of the unit-distance proof's "rich field first, planar projection second" strategy: first choose the character subspace where structural constraints are exact, then solve the physical equations in that reduced space.

## Implemented Conclusion

The code in this folder verifies three concrete claims:

1. The cyclic Fourier filter exactly enforces the choreography center-of-mass and half-turn identities.
2. The known Moore-Chenciner figure-eight initial condition is periodic under direct Newtonian integration, conserves energy, and has zero total linear/angular momentum to numerical tolerance.
3. A least-squares shooting correction starting from a rough symmetric seed converges to a nearby scaled member of the figure-eight family.

That is a meaningful application, not a general solution: the hidden-symmetry construction gives a reliable route to a nontrivial exact-looking orbit, and the tests check the claim in the original Newtonian coordinates.


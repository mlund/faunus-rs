# Extending Faunus

Analyses, Monte Carlo moves and energy terms are Rust types inside the crate, named from the YAML
input. There is no plugin interface: extending Faunus means adding a type and rebuilding.

Each of the three has a worked template in the source tree. `cargo test` compiles and runs every
template, so none can fall out of step with the interface it demonstrates.

| To add | Copy | Implement |
|---|---|---|
| an analysis | `src/analysis/template.rs` | `Analyze` |
| a move | `src/montecarlo/template.rs` | `MoveProposal` |
| an energy term | `src/energy/template.rs` | `EnergyChange`, or `StatefulEnergy` to cache |

Read the template first. What follows explains why they are shaped as they are.

## Each extension sees only what it needs

The compiler, not convention, decides what each of the three may do to the system.

Analyses and moves receive an *observable* system. They read positions, atom kinds, distances, the
cell and the topology, and they resolve selections. They cannot move a particle, resize a group,
change an atom's kind or rescale the volume, because the interface they are handed omits those
operations. A move therefore cannot mutate the system while proposing a change to it; it returns a
description of the change, and the framework applies it.

Analyses that need a trial move, such as a Widom insertion, are the exception. They receive a
*perturbable* system: a copy they may translate, rotate or rescale to measure an energy difference.
The copy is theirs; the real system is untouched. `src/analysis/virtual_translate.rs` is the smallest
example.

A single call, `measure`, owns the whole excursion. It states the intent — translate this molecule,
reorient that one, scale the volume — evaluates a closure on the perturbed system, and restores it.
Nothing else mutates the copy, so a perturbation cannot outlive the measurement that needed it.

That closes a class of error. An energy term caches each molecule's interactions and cannot see
that a molecule has moved, so a displacement followed by an energy evaluation returns the energy of
the configuration before it. The force such a measurement reports is exactly zero, and the
orientational landscape perfectly flat — the quantities `virtual_translate` and `widom_rotation`
exist to measure, and neither looks wrong on sight.

Restoring the copy is not the same as perturbing it back. `measure` rolls each cache back to a
snapshot; reversing the incremental update instead would compute $\infty + (-\infty) = \mathrm{NaN}$
for any trial pose that overlapped a neighbour, and one forbidden orientation would poison every
reading after it.

Energy terms also receive an observable system. A term that changed the system it was asked to
evaluate would corrupt the simulation, and no test would catch it.

## What an analysis must get right

Resolve selections once, in `build`. A selection such as `atomtype Na` resolves to a list of
particle indices. Resolving it every frame wastes time; caching it naively is wrong. Store it in a
`CachedSelection`, which re-resolves when — and only when — the system changes in a way that
selection can see.

Consider a titration move that turns a sodium into a chlorine. Nothing is added or removed, so the
groups are unchanged, and a cache keyed on group composition alone would serve the pre-swap atoms
for the rest of the run. `CachedSelection` watches atom kinds as well. The template tests exactly
this case.

Do not count your own samples. The framework owns the frame counter and the sampling frequency.
Implement `perform_sample`, which runs only when a sample is due. `results` runs only after at least
one sample, so an average never guards against dividing by zero.

## What a move must get right

A proposed move carries two things: the geometric transform to apply, and a description of what
changed, which tells each energy term how much to recompute. Disagreement between them corrupts the
simulation.

Suppose a move rotates part of a molecule but reports a rigid-body displacement. The bonded term
recomputes only when a molecule's internal geometry changes, so it returns the same value twice. The
non-bonded term serves a cached whole-molecule energy, likewise unchanged. The energy difference
comes out as zero, the move is always accepted, and the drift check at the end of the run passes —
because it adds up the same wrong numbers on both sides.

A move therefore never states the two separately. It calls a constructor — `translate_group`,
`rotate_group`, `translate_atoms` or `rotate_atoms` — which derives the description of the change
from the transform it builds. A mismatched pair cannot be written.

Resolve molecule names in `finalize`, before the run starts. A typo in the input file should stop
the simulation, not yield a move that never fires.

## What an energy term must get right

The framework calls a term twice per trial move, before and after, and subtracts. Both calls must
measure the same thing.

A term is told which part of the system changed and may recompute only what that requires. Then it
must return the energy of exactly the particles the change touches, and use the same partition on
both calls. A term that gets this wrong returns plausible numbers and passes the drift check.

The safe starting point, which the stateless template term takes, is to ignore the description of
the change and recompute the total. That is correct and slow.

A term that caches move-mutable state to go faster implements `StatefulEnergy` instead. The trait
splits the energy into a fresh `total_energy` and an incremental `partial_energy`, and the framework
routes a whole-system evaluation to `total_energy` — so a cache cannot hide its own drift from the
energy-drift check. The cached state lives behind a `Snapshot`, which reduces the
`save_backup`/`undo`/`discard_backup` reject protocol to one line each and covers every field, so a
rejected move always restores exactly what was cached. The template's second term demonstrates this.

Caching inside a term is fine, including through a `RefCell`. Changing the system through one is not.

## Registering a new extension

An analysis is named from YAML through the `AnalysisBuilder` enum and a move through `MoveBuilder`:
add a variant pointing at your builder struct, and the name becomes available in the input file. An
energy term needs a variant in the `EnergyTerm` enum and an optional field on `HamiltonianBuilder`,
which is how the `energy:` section maps names to terms.

Every template shows the builder shape: a struct deserialized from the input, and a `build` method
that takes the system and returns the finished object.

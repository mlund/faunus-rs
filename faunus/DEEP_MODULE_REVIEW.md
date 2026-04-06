# Deep Module Design Review

Review of the faunus-rs codebase through the lens of "deep module" design
(simple interfaces hiding complex implementations, per Ousterhout's _A Philosophy of Software Design_).

## Module Depth Summary

| Module | Public Items | Lines (approx) | Depth |
|---|---|---|---|
| `selection/` | 2 | 1,100 | Very Deep |
| `energy/` | ~13 | 3,980 | Deep |
| `analysis/` | ~10 | 2,080 | Deep |
| `collective_variable` | 4 | 767 | Deep |
| `topology/` | ~12 | 3,700 | Deep |
| `backend` | 2 | ~400 | Deep |
| `montecarlo/` | ~9 | 1,630 | Moderate |
| `group` | ~7 | 954 | Moderate |
| `cell/` | ~9 | 800 | Moderate |
| `context` | 5 traits, ~28 methods | 189 + 392 impl | Wide (deep in spots) |
| `propagate` | ~8 | 4,613 | Deep |
| `change`, `state`, `particle`, `info`, `time`, `dimension` | 1-3 each | 50-104 each | Shallow |

---

## Detailed Module Reviews

### `selection/` — Very Deep

The standout example. Only 2 public types (`Selection`, `SelectionError`) hide ~1,100 lines
containing a full lexer, recursive descent parser, AST, and evaluator. A user just writes
`Selection::parse("molecule water").resolve_atoms(...)`.

### `energy/` — Deep (layered)

~13 public items wrap 3,980 lines across 11 files. Three-layer architecture:

**Layer 1 — Public interface (thin).** `EnergyChange` trait has a single method
(`energy(&self, context, change) -> f64`). `Hamiltonian` is a `Vec<EnergyTerm>` that sums
energies with early exit on non-finite values. `HamiltonianBuilder` is `pub(crate)`.

**Layer 2 — Term dispatch (mechanical).** `EnergyTerm` enum (8 variants) forwards `energy()`,
`update()`, and `sync_from()` via match arms. Each term independently handles the `Change` type
to compute only what's needed (e.g., `IntramolecularBonded` returns 0 for `RigidBody` changes
since bond lengths don't change under rigid translation).

**Layer 3 — Individual terms (where the real depth lives).**

- **`NonbondedMatrix<P>`** (1,244 lines): Generic over potential type (`ArcPotential` or
  `SplinedPotential`). `NonbondedTerm` trait provides 10 composable helpers
  (`particle_with_particle` → `particle_with_group` → `group_with_other_groups` → etc.)
  building up from the single pair interaction. `ExclusionMatrix` (128 lines, `pub(super)`)
  is invisible to callers. Spline tabulation via `from_nonbonded()` transparently replaces
  trait-object potentials with cubic Hermite splines in r² space (4 FMA per evaluation).

- **`builder.rs`** (933 lines): The deepest single file. `PairPotentialBuilder` resolves
  atom pairs via `HashMap<DefaultOrPair, Vec<PairInteraction>>` with fallback to "default".
  `PairInteraction` (9 variants) converts to `Box<dyn IsotropicTwobodyEnergy>` via mixing
  rules (`DirectOrMixing<T>`), Coulomb scheme setup with medium/permittivity/Debye length,
  and trait object summation. Custom serde visitor for `DefaultOrPair`. All `pub(crate)`.

- **`ExternalPressure`** (337 lines): NPT energy `P·V - (N+1)·kT·ln(V)` with 5 pressure
  units and custom serde for single-key YAML maps. Smart entity counting (atomic groups
  contribute per-particle, molecular groups contribute 1).

- **`IntermolecularBonded`** (part of bonded.rs, 593 total): Maintains `particles_status`
  activation array, updated selectively on `Resize` changes — the only stateful bonded term.

- **`SasaEnergy`** (202 lines): Voronoi tessellation via `voronota-ltr`, per-particle
  surface tensions, auto-offset from first configuration.

- **`Constrain`** (206 lines): Hard (infinity outside range) or soft harmonic
  (`0.5·k·(eq-value)²`) constraints on collective variables.

- **`CellOverlap`** (52 lines): Appropriately shallow sentinel term, always first in
  Hamiltonian for early rejection.

**Strengths:**
- Change-driven lazy evaluation: each term computes only what the `Change` requires
- `HamiltonianBuilder` fully `pub(crate)`, `ExclusionMatrix` `pub(super)` — tight visibility
- `NonbondedMatrix<P>` generic lets spline optimization be a transparent swap
- Construction order in `Hamiltonian::new()` is deliberate: CellOverlap first for early exit

**Issues:**
- `todo!()` in `NonbondedMatrix::single_group_change` (nonbonded.rs:327): `Resize` and
  `UpdateIdentity` unimplemented — GCMC with nonbonded interactions will panic.
- `SasaEnergy::update` ignores `Change` granularity (sasa.rs:107-111): every change triggers
  full Voronoi recomputation. The match arms are identical.
- ~70 lines of commented-out tests in builder.rs (lines 603-670).

### `analysis/` — Deep

~10 public items wrap 2,080 lines behind a single 5-method `Analyze<T>` trait:

- **`ShapeAnalysis`** (584 lines, 3 config fields): eigendecomposition of mass-weighted
  gyration tensor, 7 shape descriptors, all computed and averaged behind `sample()`.
- **`VirtualTranslate`** (382 lines, 6 config fields): Widom perturbation method with unit
  conversion, overflow protection, and context cloning.
- **`RadialDistribution`** (347 lines, 6 config fields): dual atom-atom / COM-COM modes,
  generic `collect_pair_distances()` helper, histogram normalization for GC ensemble.
- **`CollectiveVariableAnalysis`** (300 lines, 3 config fields): Welford's online mean,
  composing with the CV abstraction.
- **`StructureWriter`** (144 lines, 2 config fields): thin I/O wrapper — appropriately
  shallow.

**Issues:**
- `MassCenterDistance` duplicates the pair iteration logic that `collect_pair_distances()`
  already generalizes in `radial_distribution.rs`.
- Every `sample()` impl repeats the same `frequency.should_perform(step)` guard — could be
  lifted into the caller or a default trait method.
- `num_samples` semantics differ: per-pair in `MassCenterDistance`, per-step everywhere else.

### `context` — Wide rather than deep

The `Context` trait itself has 1 method with a default impl. The real API is spread across
5 supertraits (`ParticleSystem`, `GroupCollection`, `WithCell`, `WithTopology`,
`WithHamiltonian`) totaling ~28 methods. A new backend must provide ~20 required impls.

**Genuine depth exists** in `scale_volume_and_positions` (4-step PBC-aware algorithm) and
`sync_group_from` (recursive dispatch across 5 `GroupChange` variants). The
`Backend` constructor hides YAML parsing, topology building, hamiltonian
construction, and group insertion.

**Issues:**
- `todo!()` in production paths: `sync_group_from` (group.rs:519,522) and
  `sync_from_groupcollection` (group.rs:557,559) will panic on untested `Change` variants.
- `particle()` returns a clone via the trait; `Backend` works around it with a
  concrete `particles()` returning `&[Particle]`, but trait-level code pays the copy cost.
- `WithHamiltonian` returns `Ref<'_>`/`RefMut<'_>`, baking `RefCell` into the trait contract.

### `backend` — Deep (facade)

2 public items, ~400 lines. `Backend` implements 5 traits and is the concrete glue
tying topology, energy, cell, and particles together. Two-phase construction:
`Hamiltonian::new()` for topology-only terms, then `Hamiltonian::finalize()` for
context-dependent terms after particles are placed.

### `collective_variable` — Deep

4 public items, 767 lines. Six concrete CV implementations and their builder logic are fully
internal.

### `propagate` — Deep

~8 public items, 4,613 lines. Three-layer architecture: `Propagate` owns
`Vec<PropagationBlock<T>>` (MC or Langevin), MC blocks hold `Vec<MoveRunner<T>>` wrapping
`Box<dyn MoveProposal<T>>` trait objects, and the `langevin/` subdirectory (3,727 lines,
80% of module) implements GPU-accelerated BAOAB integration via CubeCL with mixed
rigid/flexible dynamics. Builder pipeline (`PropagateBuilder` → `MoveCollectionBuilder` →
`MoveBuilder`) handles YAML deserialization. Only 5 forwarding methods remain in
`PropagationBlock`.

**Strengths:**
- GPU logic fully isolated in `langevin/` subdirectory
- `MoveRunner` encapsulates stats/weight/repeat; `MoveProposal` trait gives clean dispatch
- Builder pipeline separates deserialization from runtime construction

**Issues:**

*Medium priority:*
- `langevin/mod.rs:353-432`: `extract_spline_data` / `extract_bonded_data` are stateless
  methods on `LangevinRunner`; move to `utils.rs` to reduce file size and eliminate
  explicit `drop()` calls needed for borrow-checker workarounds.
- `langevin/mod.rs:68-71`: `pub(crate) config` and `pub(in ...) elapsed` are mutated
  directly from the parent module; add accessors and make fields private.
- `mod.rs:208`: `Option<StdRng>` is always `Some` after construction; change to `StdRng`.
- `mod.rs:47`: `MoveCollection` is `pub` but never in the public API; downgrade to
  `pub(crate)`.

*Low priority:*
- `langevin/pipeline.rs:520-888`: Six `dispatch_*` methods repeat `CubeCount`/`CubeDim`
  boilerplate; a helper would reduce ~150 lines.
- `moveproposal.rs:103`: `Displacement::AngleDistance` has no producer; dead code.
- `langevin/pipeline.rs:740,854`: `0xDEAD_BEEF` RNG seed needs a comment.

---

## Structural Observations

### Enum dispatch vs. trait objects

`EnergyTerm`, `Cell`, `PropagationBlock`, `Change` all use enums with manual match dispatch.
This avoids dynamic dispatch overhead, but every new variant requires updating match arms
everywhere. For `EnergyTerm` this is manageable (3 match blocks in energyterm.rs).
MC moves use trait-object dispatch via `Box<dyn MoveProposal<T>>`.

### Single concrete `Context` implementation

The trait hierarchy is rich but only `Backend` implements it. Designed for future
GPU/parallel backends, but 5 traits and ~20 required methods for 1 implementor is a gap.
The `Clone` bound further limits extensibility.

### Builder pattern consistency

The YAML → builder → runtime pipeline is a deep module pattern applied consistently across
energy, analysis, and topology. `HamiltonianBuilder` being `pub(crate)` is the gold standard;
the same treatment could be applied to other builders only used internally.

### Small data types are appropriately shallow

`Change`, `State`, `Particle`, `Dimension`, `Time` are small, focused types. Not every module
needs to be deep; these are the "leaf" types that deep modules compose.

---

## Recommendations

### Done

1. ~~**Tighten `topology/` exports**~~ — Public surface reduced from ~25 to ~12 items.
   Glob re-exports replaced with selective lists. Dead code removed.

5. ~~**Factor `PairInteraction::to_boxed` duplication**~~ — Added `FromMixing` trait on
   `DirectOrMixing<T>` encapsulating per-type mixing logic. Five match arms replaced with
   single-line delegations.

6. ~~**Implement `EnergyChange` consistently**~~ — Not a real issue. All energy types use
   inherent `energy()` methods; `EnergyTerm` enum dispatch is the intended pattern.

9. ~~**Extract deferred energy construction**~~ — Moved context-dependent energy term
   construction from `Backend::new()` into `Hamiltonian::finalize()`. Unified scattered
   `thermal_energy` wrappers into a single `R_IN_KJ_PER_MOL` constant. Removed unused
   `WithTemperature` trait.

3. ~~**Reduce `Move` enum boilerplate**~~ — Already addressed: `MoveRunner` wraps shared
   fields (stats, weight, repeat), `MoveProposal` trait provides trait-object dispatch.
   Only 5 forwarding methods remain in `PropagationBlock`.

4. ~~**Collapse `propagate` into `montecarlo/`**~~ — No longer relevant. Module grew to
   4,613 lines with deep GPU Langevin dynamics; it is a substantial module in its own right.

10. ~~**Fix propagate high-priority issues**~~ — Replaced `unwrap()` on just-assigned
    `Option` with local binding in `LangevinRunner::propagate`. Converted 10 kernel
    `.expect()` panics to `?` error propagation. Replaced `build_move!` macro with
    `BuildableMove` trait making the weight/repeat/finalize contract compiler-enforced.

### Cross-cutting

2. **Replace `todo!()` with proper error handling** — Found in multiple modules:
   - `sync_group_from` (group.rs:519,522) and `sync_from_groupcollection` (group.rs:557,559)
   - `NonbondedMatrix::single_group_change` (nonbonded.rs:327) for `Resize`/`UpdateIdentity`
   All will panic at runtime on GCMC or identity swap operations.

### `analysis/` specific

7. **Lift `collect_pair_distances` to shared scope** — Currently private to
   `radial_distribution.rs` but duplicated in `MassCenterDistance::sample()`.

### `context` specific

8. **Split `ParticleSystem` trait** — Its 12 methods span distance calculation, mass center
   computation, volume scaling, and particle manipulation. Smaller, focused traits would
   each be deeper.

---

## `Move` Dispatch: Current Design (resolved)

The original `Move` enum boilerplate has been addressed. The current architecture uses
`MoveRunner<T>` (shared stats/weight/repeat wrapper) around `Box<dyn MoveProposal<T>>`
trait objects. `MoveBuilder` enum handles tagged YAML deserialization (`!TranslateMolecule
{ ... }`), then `.build()` produces trait objects. `PropagationBlock<T>` has only 5
forwarding methods across MC and Langevin variants — acceptable for a 2-variant enum.

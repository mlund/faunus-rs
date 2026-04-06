# Plan: Implement `ReactionCoordinate` Analysis

## Context

The C++ Faunus `FileReactionCoordinate` analysis evaluates a reaction coordinate (collective variable) at each sample step, tracks its running average, and streams `{step, value, average}` to a file. The Rust codebase already has the `CollectiveVariable` infrastructure but lacks this analysis type. Adding it enables users to monitor any CV over time during a simulation.

## Approach

Create a new `ReactionCoordinate` analysis in `src/analysis/reaction_coordinate.rs` following the exact patterns of `VirtualTranslate` and `ConstrainBuilder`.

## YAML Configuration

```yaml
analysis:
  - !ReactionCoordinate
    property: mass_center_position
    selection: "molecule protein"
    dimension: z
    range: [-50.0, 50.0]
    file: rc.dat          # optional; if omitted, only mean is tracked
    frequency: !Every 100
```

The CV fields (`property`, `range`, `dimension`, `selection`, `selection2`, `resolution`) are flattened from `CollectiveVariableBuilder`, same as `ConstrainBuilder` does.

## Files to Create/Modify

### 1. Create `src/analysis/reaction_coordinate.rs`

**Struct: `ReactionCoordinate`**
- `cv: ConcreteCollectiveVariable` — the resolved collective variable
- `stream: Option<Box<dyn Write>>` — optional output file
- `output_file: Option<PathBuf>` — filename (for reporting)
- `frequency: Frequency`
- `mean: Mean` — running average (`average::Mean`)
- `num_samples: usize`

**Builder: `ReactionCoordinateBuilder`** (manual, not derive_builder)

Since we need `#[serde(flatten)]` for the CV fields (like `ConstrainBuilder`), and `derive_builder` doesn't support flatten well, use a manual builder struct:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionCoordinateBuilder {
    #[serde(flatten)]
    cv: CollectiveVariableBuilder,
    file: Option<PathBuf>,
    frequency: Frequency,
}
```

The `build()` method takes `&impl Context` to resolve selections via `cv.build_concrete(context)`.

**`Analyze<T>` impl:**
- `sample()`: evaluate CV, add to `Mean`, write `"{step} {value:.6} {avg:.6}\n"` to stream
- `frequency()`: return stored frequency
- `num_samples()`: return counter
- `flush()`: flush stream

**`Info` impl:**
- `short_name`: `"reactioncoordinate"`
- `long_name`: `"Reaction coordinate time series"`

### 2. Modify `src/analysis/mod.rs`

- Add `mod reaction_coordinate;`
- Add `pub use reaction_coordinate::{ReactionCoordinate, ReactionCoordinateBuilder};`
- Add variant to `AnalysisBuilder`:
  ```rust
  ReactionCoordinate(ReactionCoordinateBuilder),
  ```
- Add match arm in `AnalysisBuilder::build()`:
  ```rust
  Self::ReactionCoordinate(builder) => Box::new(builder.build(context)?),
  ```

### 3. Modify `src/collective_variable.rs`

- Change `ConcreteCollectiveVariable` visibility from `pub(crate)` to `pub` so the analysis module can use it (or keep `pub(crate)` if analysis is in the same crate — it is, so `pub(crate)` suffices).

No change needed — `pub(crate)` already allows access from `src/analysis/`.

## Existing Code to Reuse

| What | Where |
|------|-------|
| `CollectiveVariableBuilder` + `build_concrete()` | `src/collective_variable.rs` |
| `ConcreteCollectiveVariable::evaluate()` | `src/collective_variable.rs` |
| `average::Mean` | `average` crate (already in deps) |
| `open_compressed()` | `src/auxiliary.rs` |
| `Analyze<T>` trait | `src/analysis/mod.rs` |
| `Frequency` enum | `src/analysis/mod.rs` |
| `Info` trait | `src/info.rs` |

## Verification

1. **Unit tests** in `reaction_coordinate.rs`:
   - YAML deserialization round-trip of `ReactionCoordinateBuilder`
   - Builder validation (missing frequency → error)
   - Deserialization via `AnalysisBuilder` enum (`!ReactionCoordinate` tag)
2. **Integration test** (behind `#[cfg(all(test, feature = "chemfiles"))]`):
   - Build from YAML against `ReferencePlatform` context
   - Call `sample()`, verify `num_samples` increments and mean is correct
3. `cargo test` — all existing tests pass
4. `cargo clippy` — no warnings

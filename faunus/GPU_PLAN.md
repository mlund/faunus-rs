# GPU MC Propagation Plan

## 1. New Platform or Purely Compute?

**Purely compute — no new platform needed.** The `SimdPlatform`'s SoA layout (`x: Vec<f64>`, `y: Vec<f64>`, `z: Vec<f64>`, `atom_kinds: Vec<u32>`) already maps directly to GPU buffer layout. The MC loop (propose move → evaluate energy → Metropolis accept/reject → backup/undo) is inherently sequential and stays on CPU. What changes is _where_ the nonbonded energy is computed.

The right abstraction is a **GPU-backed energy term** — a `GpuNonbonded` that implements `EnergyChange` and replaces (or wraps) `NonbondedMatrix` for the nonbonded evaluation. This matches how mcgpu separates the `EnergyBackend` (GPU/CPU) from the MC logic.

```
SimdPlatform (data owner, unchanged)
  └── Hamiltonian
        ├── NonbondedMatrix       ← CPU path (existing)
        └── GpuNonbonded          ← GPU path (new, same trait)
              ├── wgpu device/queue/pipelines
              ├── GPU buffers mirroring SoA positions
              ├── spline coefficient buffers
              └── GroupEnergyCache (CPU-side, same as now)
```

## 2. CPU↔GPU Transfer Cost for Single-Molecule Perturbations

For 100 macromolecules × 1000 particles:

| Transfer | Size | Apple Silicon (UMA) | Discrete GPU (PCIe 4) |
|----------|------|--------------------|-----------------------|
| Upload moved group (1000 × 3 × f32) | 12 KB | ~0.1 μs (shared memory) | ~1 μs |
| Upload all positions (100K × 3 × f32) | 1.2 MB | ~6 μs | ~50 μs |
| Download energy row (100 × f32) | 400 B | negligible | negligible |
| GPU dispatch latency | — | ~5-10 μs | ~10-20 μs |

**Key findings from mcgpu benchmarks** (Apple M4, 699 sites/molecule):

| Molecules | CPU (steps/s) | GPU (steps/s) | GPU speedup |
|-----------|--------------|---------------|-------------|
| 50 | 186 | 209 | 1.1× |
| 100 | 89 | 131 | 1.5× |
| 200 | 45 | 86 | 1.9× |
| 400 | 16 | 52 | 3.3× |

The energy cache is the bigger win: **3.7–6.7× speedup** independent of backend. GPU only outperforms CPU at ~100+ molecules because dispatch latency dominates for small systems.

## 3. Implementation Strategy

### What mcgpu does well (and faunus should adopt):

1. **Pairwise row computation**: When molecule `i` moves, dispatch one compute shader that evaluates E(i, j) for all j in parallel. Each workgroup handles one target molecule; 256 threads reduce over site pairs.

2. **Energy cache on CPU**: The N×N `GroupEnergyCache` stays host-side. GPU computes the new energy row, CPU does the symmetric cache update. This is exactly what faunus already does.

3. **Spline coefficients on GPU**: Precompute `NonbondedMatrixSplined` → upload coefficient buffer once. mcgpu's `gpu_spline.rs` already converts faunus's spline format.

4. **Selective upload**: Only write the moved group's positions to GPU, not the entire system. faunus's `Change::SingleGroup(gi, RigidBody)` tells you exactly what changed.

5. **Batch moves**: mcgpu's `compute_batch` shader evaluates multiple independent molecules in a single dispatch (up to 16). This amortizes the ~10 μs dispatch latency and is the key to GPU scaling.

### What the implementation would look like:

```rust
GpuNonbonded {
    // wgpu resources
    device, queue, pipeline,

    // GPU buffers (mirroring SimdPlatform SoA)
    pos_buffer,          // all positions [f32; 4] × N_total
    atom_type_buffer,    // all atom kinds [u32; 4] × N_total
    spline_params,       // per-type-pair spline metadata
    spline_coeffs,       // contiguous coefficient array

    // CPU-side cache (same as existing GroupEnergyCache)
    cache: RefCell<Option<GroupEnergyCache>>,
}

impl EnergyChange for GpuNonbonded {
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        match change {
            Change::SingleGroup(gi, GroupChange::RigidBody) => {
                // O(1) cache lookup for old energy
                self.cache.borrow().group_energies[*gi]
            }
            _ => { /* full recalculation */ }
        }
    }

    fn update(&mut self, context: &impl Context, change: &Change) {
        // Upload only moved group positions to GPU
        // Dispatch compute shader: one row of N-1 pair energies
        // Read back N-1 f32 values
        // Update cache symmetrically (CPU)
    }
}
```

## 4. Bottom Line

- **Not a new platform** — it's a new `EnergyTerm` variant that delegates to GPU compute
- **Transfer cost is small** (~12 KB up, ~400 B down per step), especially on Apple Silicon UMA
- **Dispatch latency is the bottleneck** for small systems; batch moves amortize it
- **Energy cache is orthogonal to GPU** and provides the largest single speedup
- mcgpu's `energy.wgsl` shader and `gpu_spline.rs` can be adapted directly since they already consume faunus spline data

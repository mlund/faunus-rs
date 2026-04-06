# GPU Acceleration via wgpu — Architecture Analysis

Analysis of GPU acceleration strategies for faunus-rs using the `wgpu` Rust crate,
covering both Langevin dynamics (MD) and Monte Carlo propagation. Informed by
OpenMM's GPU force architecture and validated against a working wgpu MC
implementation (`mcgpu`).

---

## 1. OpenMM's Cross-Backend Architecture

OpenMM maintains a `platforms/common/` layer with ~12K lines of platform-neutral
kernel code (`.cc` files using abstract types: `real`, `real3`, `GLOBAL`, `LOCAL`).
At runtime, `replaceStrings()` substitutes platform-specific tokens and
`compileProgram()` JIT-compiles for CUDA (NVRTC), OpenCL (`clBuildProgram`), or
HIP (hiprtc). Per-backend layers (~2–3K lines each) supply only tile-based
dispatch, FFT, sorting, and memory primitives.

**wgpu already solves this problem.** WGSL shaders are written once and `naga`
transpiles to the native backend at runtime: SPIR-V (Vulkan), MSL (Metal),
HLSL (DX12), GLSL (OpenGL). No macro substitution or per-backend kernel code
is needed.

| OpenMM mechanism | wgpu equivalent |
|---|---|
| Abstract `.cc` kernels with type macros | WGSL — already platform-neutral |
| `replaceStrings()` runtime substitution | `naga_oil` provides `#import`, `#define`, `#ifdef` |
| `compileProgram()` per backend | `device.create_shader_module()` — naga handles translation |
| Per-backend tile dispatch (`.cu`/`.cl`/`.hip`) | Single WGSL compute shader with `@workgroup_size` |

---

## 2. OpenMM's Langevin Integrator

**Source:** `platforms/common/src/kernels/langevinMiddle.cc`

OpenMM implements the **BAOAB (Langevin Middle)** scheme, split into three GPU
kernels with constraint enforcement between them:

### Kernel 1 — Velocity half-step (lines 7–19)
```
v += (dt/2) * F/m
```

### Kernel 2 — Position + stochastic kick (lines 26–50)
```
r += (dt/2) * v
v  = vscale * v + noisescale * sqrt(1/m) * R_gaussian
r += (dt/2) * v_new
```
Where `vscale = exp(-dt * γ)` and `noisescale = sqrt(kT * (1 - vscale²))`.

### Kernel 3 — Constraint correction (lines 57–90)
```
v += (constrained_delta - unconstrained_delta) / dt
r += constraint_correction
```

### Execution sequence per step
```
prepareRandomNumbers()       → generate Gaussian noise
kernel1()                    → v half-step with forces
applyVelocityConstraints()   → SHAKE on velocities
kernel2()                    → stochastic position/velocity update
applyConstraints()           → SHAKE/SETTLE on positions
kernel3()                    → velocity correction from constraints
computeVirtualSites()        → virtual site positions
```

### Buffers passed to GPU
- `posq`: positions as `real4` (xyz + charge)
- `velm`: velocities as `mixed4` (vxyz + 1/mass)
- `force`: forces as `mm_long` (int64 fixed-point)
- `params`: `[vscale, noisescale]` — recomputed only when T, γ, or dt change
- `random`: pre-generated `float4` Gaussian deviates
- `dt`: timestep as `mixed2`

### Random number generation
**Source:** `platforms/common/src/kernels/integrationUtilities.cc`

Uses MLCG + Xorshift → Box-Muller transform. Per-thread `uint4` state produces
4 Gaussian deviates per `float4` output. State persisted in a seed buffer
across steps.

---

## 3. Force Calculation — The Actual Bottleneck

### 3.1 OpenMM's approach: analytical kernels for built-in forces

OpenMM evaluates all built-in forces analytically on GPU:

| Force | Kernel | Method |
|---|---|---|
| Lennard-Jones | `coulombLennardJones.cc` | Direct `σ⁶/r⁶` computation |
| Coulomb (PME) | `coulombLennardJones.cc` + `pme.cc` | Hastings erfc + 5th-order B-spline grid |
| Harmonic bond | `harmonicBondForce.cc` | `k*(r - r0)` |
| Harmonic angle | `harmonicAngleForce.cc` | `k*(θ - θ0)` with acos/asin |
| Periodic torsion | `periodicTorsionForce.cc` | `k*n*sin(nφ - phase)` |
| CMAP | `cmapTorsionForce.cc` | Bicubic spline on (φ,ψ) grid |

For user-defined potentials, OpenMM uses `TabulatedFunction` — precomputed cubic
spline coefficients uploaded as `float4` arrays, evaluated via Horner's method.

### 3.2 The interatomic crate already has splines

The `interatomic` crate's `SplinedPotential` (`src/twobody/hermite.rs`) provides
exactly the mechanism needed to bridge Rust potentials to GPU:

```
Any IsotropicTwobodyEnergy + Cutoff
        │
        ▼  SplinedPotential::new()
   Vec<SplineCoeffs> { u: [f64; 4], f: [f64; 4] }
        │
        ▼  to_simd_f32()
   Vec<[f32; 4]>          ← GPU-ready buffer
        │
        ▼  upload to wgpu storage buffer
   WGSL: one generic kernel evaluates ALL potentials
```

Key properties of `SplinedPotential`:
- `#[repr(C, align(32))]` coefficients — maps directly to `array<vec4<f32>>` in WGSL
- Precomputed energy *and* force coefficients (4 FMAs each)
- Multiple grid strategies: `UniformR`, `UniformRsq`, `PowerLaw2`, `InverseRsq`
- `to_simd_f32()` already converts to single precision
- Wraps any `IsotropicTwobodyEnergy` — LJ, WCA, Mie, Combined, Box<dyn>, etc.
- Default 2000 points gives ~1e-6 relative error; `high_accuracy()` available

### 3.3 Angular potentials via 1D splines

Threebody (`ThreebodyAngleEnergy`) and fourbody (`FourbodyAngleEnergy`) potentials
are 1D functions of angle with bounded domains, simpler than pair potentials:

| Property | Twobody U(r²) | Threebody U(θ) | Fourbody U(φ) |
|---|---|---|---|
| Domain | [r_min, r_cutoff] — varies | [0, π] — fixed | [-π, π] — fixed |
| Grid type needed | PowerLaw2 (steep core) | UniformR (smooth) | UniformR, periodic |
| One table per... | atom-type pair | bond-angle type | dihedral type |

A `SplinedAnglePotential` analogous to `SplinedPotential` would spline any
angular potential over its fixed domain. For dihedrals, periodic boundary
conditions ensure `U(-π) = U(π)` with continuous derivatives — matching
OpenMM's CMAP approach.

The advantage of uniformity: one WGSL kernel handles all angle types, one handles
all dihedral types. Adding a new potential (e.g., `CosineTorsion`) requires zero
GPU code changes — just spline it and upload coefficients.

---

## 4. Proposed wgpu Architecture

### 4.1 Shader structure

Using `naga_oil` for modular WGSL composition:

```
shaders/
  common.wgsl             — shared struct definitions, constants
  rng.wgsl                — Philox4x32-10 + Box-Muller
  langevin_part1.wgsl     — velocity half-step
  langevin_part2.wgsl     — position + stochastic kick  (#import rng)
  langevin_part3.wgsl     — constraint correction
  pair_forces.wgsl        — nonbonded pair force evaluation
  angle_forces.wgsl       — threebody angle forces
  dihedral_forces.wgsl    — fourbody dihedral forces
```

### 4.2 GPU buffer layout

```
Particle data:
  positions:       array<vec4<f32>>   // xyz + charge (or w unused)
  velocities:      array<vec4<f32>>   // vxyz + 1/mass
  forces:          array<vec4<f32>>   // accumulated forces (atomic writes)

Spline coefficient tables (all potentials unified):
  pair_coeffs:     array<vec4<f32>>   // [f32; 4] per interval, all pair types concatenated
  angle_coeffs:    array<vec4<f32>>   // all angle types concatenated
  dihedral_coeffs: array<vec4<f32>>   // all dihedral types concatenated

Topology / interaction lists:
  pair_list:       array<vec2<u32>>   // from neighbor list (atom indices + pair type)
  angle_list:      array<vec4<u32>>   // (i, j, k, table_offset)
  dihedral_list:   array<vec4<u32>>   // (i, j, k, l) + table_offset packed

Simulation parameters:
  params:          uniform Params     // dt, vscale, noisescale, N, rng_seed, step
  table_meta:      array<TableMeta>   // per-type: offset, n_points, r_min, inv_delta
```

### 4.3 Compute dispatch

```
Per MD step:
  1. pair_forces.wgsl       dispatch ceil(N_pairs / 64)      — spline eval + force projection
  2. angle_forces.wgsl      dispatch ceil(N_angles / 64)     — spline eval + angle geometry
  3. dihedral_forces.wgsl   dispatch ceil(N_dihedrals / 64)  — spline eval + dihedral geometry
  4. langevin_part1.wgsl    dispatch ceil(N_atoms / 64)      — v half-step
  5. [CPU: velocity constraints if needed]
  6. langevin_part2.wgsl    dispatch ceil(N_atoms / 64)      — stochastic update
  7. [CPU: position constraints if needed]
  8. langevin_part3.wgsl    dispatch ceil(N_atoms / 64)      — constraint correction
```

Each force kernel follows the same pattern:

```wgsl
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> forces: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> coeffs: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> table_meta: array<TableMeta>;

@compute @workgroup_size(64)
fn pair_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair = pair_list[gid.x];
    let r = distance(positions[pair.i], positions[pair.j]);

    // Look up spline table for this pair type
    let meta = table_meta[pair.type_id];
    let t = (r - meta.r_min) * meta.inv_delta;
    let i = u32(t);
    let eps = t - f32(i);

    // 4 FMAs — same Horner evaluation as hermite.rs
    let c = coeffs[meta.offset + i];
    let force_mag = c.x + eps * (c.y + eps * (c.z + eps * c.w));

    // Project onto Cartesian force components
    let r_hat = normalize(positions[pair.j].xyz - positions[pair.i].xyz);
    let f = force_mag * r_hat;

    // Atomic accumulation (Newton's third law)
    atomicAdd(&forces[pair.i].xyz, f);
    atomicAdd(&forces[pair.j].xyz, -f);
}
```

### 4.4 Random number generation

Implement Philox4x32-10 in WGSL (~20 lines). Stateless design with counter =
`(step, particle_id)` and a global seed key — no per-particle RNG state needed.
Apply Box-Muller transform for Gaussian deviates:

```wgsl
fn rand_gaussian(counter: vec2<u32>, key: u32) -> vec2<f32> {
    let u = philox4x32(counter, key);  // 4 uniform u32
    let u1 = max(f32(u.x) / 4294967295.0, 1e-10);
    let u2 = f32(u.y) / 4294967295.0;
    let mag = sqrt(-2.0 * log(u1));
    return vec2<f32>(mag * cos(6.2831853 * u2), mag * sin(6.2831853 * u2));
}
```

---

## 5. Precision and Limitations

### 5.1 f32-only constraint

WGSL has no runtime `f64`. The WebGPU f64 feature request (gpuweb/gpuweb#2805)
is Milestone 4+ with no timeline. Metal has no native f64 support.

**Mitigations:**
- Use f32 for all GPU computation (forces, integration)
- Kahan compensated summation for energy accumulation
- Perform constraint solving (SHAKE/SETTLE) and total energy on CPU
- `SplinedPotential::to_simd_f32()` already exists and is tested
- OpenMM's Ewald `erfc` Hastings approximation was designed for f32

### 5.2 No warp-level shuffles

OpenMM's tile-based nonbonded dispatch uses `__shfl` (CUDA) for atom position
broadcasting within warps. WGSL has no direct equivalent.

**Mitigations:**
- Use `var<workgroup>` shared memory for tile-based approaches
- WGSL subgroup operations (added January 2025) provide `subgroupShuffle()`,
  `subgroupAdd()`, etc., but subgroup sizes vary (32 NVIDIA, 64 AMD, 32 Apple)
- For the integrator kernels this is irrelevant — they are embarrassingly parallel

### 5.3 Atomic float accumulation

Force accumulation from multiple pairs onto the same atom requires atomic writes.
WGSL supports `atomicAdd` on `i32`/`u32` storage buffers. Float atomics require
either:
- Fixed-point integer encoding (OpenMM's approach: `mm_long` int64)
- Atomic compare-and-swap loop on `u32` (bitcast f32 ↔ u32)
- Per-interaction output buffer with CPU-side reduction (simple but memory-heavy)

### 5.4 Grid type recommendation for GPU

For GPU inner loops, minimize per-evaluation arithmetic:

| GridType | Index computation | GPU suitability |
|---|---|---|
| `UniformR` | `(r - r_min) * inv_delta` | Best — 1 multiply + 1 subtract |
| `UniformRsq` | `(rsq - rsq_min) * inv_delta` | Good — no sqrt needed |
| `InverseRsq` | `(1/rsq - w_min) * inv_delta` | Good — 1 division |
| `PowerLaw2` | `sqrt((r - r_min) / range) * (n-1)` | Acceptable — 1 extra sqrt |
| `PowerLaw(p)` | `powf(x, 1/p) * (n-1)` | Avoid — `powf` is expensive |

For pair potentials with steep repulsive cores, `UniformRsq` or `InverseRsq` are
preferred since they avoid sqrt and provide denser sampling at short range.
For angular potentials (smooth, bounded domain), `UniformR` is ideal.

---

## 6. Neighbor List Considerations

The neighbor list is the most complex GPU component, separate from potential
evaluation. Options in order of implementation complexity:

1. **CPU neighbor list, GPU force evaluation** — simplest starting point.
   Rebuild neighbor list on CPU, upload pair list each step (or every N steps).
   Viable for systems up to ~10K atoms.

2. **Cell list on GPU** — partition space into cells of side ≥ r_cutoff.
   Each thread processes one atom, iterating over neighboring cells.
   Standard approach for moderate system sizes.

3. **Tile-based (OpenMM style)** — warp-sized atom tiles with exclusion bitmasks.
   Maximum throughput but requires careful shared-memory management and is
   harder to implement in WGSL without warp intrinsics.

---

## 7. GPU-Accelerated Monte Carlo via Pairwise Caching

### 7.1 When GPU helps MC

Standard MC moves one molecule per step — the delta energy calculation is O(N)
pair evaluations. For small molecules (water, 3 sites), this is ~3K pairs per
move, too little to saturate a GPU. But for **large rigid bodies** the work
per move scales as `N_sites_moved × N_sites_other`:

| System | Sites/molecule | Pairs per move | GPU benefit |
|---|---|---|---|
| Water × 1000 | 3 | ~3K | None — kernel launch overhead dominates |
| Protein × 100 | 500 | ~25M | 1.5–3x speedup |
| Colloid × 100 | 1000 | ~100M | 3–7x speedup |

Benchmarks from the `mcgpu` project (Apple M4, LJ + Yukawa, 699 sites/molecule):

| Molecules | Atoms | CPU (steps/s) | GPU cached (steps/s) | Speedup |
|---|---|---|---|---|
| 50 | 34,950 | 185.8 | 209.0 | 1.1x |
| 100 | 69,900 | 88.7 | 130.6 | 1.5x |
| 200 | 139,800 | 44.5 | 85.6 | 1.9x |
| 400 | 279,600 | 15.8 | 51.5 | 3.3x |

### 7.2 Pairwise energy caching strategy

The key optimization is an N×N molecule-molecule energy cache that eliminates
redundant computation of `e_old`. Both CPU and GPU backends benefit.

**Cache structure:**
- `pairwise_cache[i * N + j]`: energy between molecules i and j
- `mol_energies[i]`: cached row sum (total energy of molecule i with all others)
- `dirty[i]`: flag indicating molecule i has moved since last computation

**Per MC step:**
```
1. e_old = mol_energies[i]                          → O(1) cache lookup
2. invalidate(i)                                     → set dirty[i] = true
3. propose move, update positions of molecule i
4. e_new = recompute row i of pairwise matrix        → O(N) molecule-molecule energies
5. accept/reject via Metropolis criterion
6. if rejected: invalidate(i), revert, recompute row → O(N) again
```

**Complexity reduction:**

| Operation | Without cache | With cache |
|---|---|---|
| Get e_old | O(N × sites²) | O(1) |
| Get e_new | O(N × sites²) | O(N × sites²) |
| **Per step** | **2 × O(N × sites²)** | **O(N × sites²)** |

Caching alone halves the work. On GPU, the improvement is larger because it
eliminates a synchronous GPU round-trip for `e_old`:

| Backend | 100 mol | 200 mol | 400 mol |
|---|---|---|---|
| GPU uncached | 33.9 | 18.0 | 9.0 |
| GPU cached | 127.6 | 68.8 | 32.7 |
| **Cache improvement** | **3.8x** | **3.8x** | **3.6x** |

### 7.3 GPU kernel design for MC energy

**Source:** `mcgpu/src/energy.wgsl`

The compute shader dispatches one workgroup (256 threads) per target molecule j.
Each workgroup computes E(mol_i, mol_j) by distributing sites of molecule i
across threads, each iterating over all sites of molecule j:

```wgsl
@compute @workgroup_size(256)
fn compute_pairwise(...) {
    let mol_j = wid.x;       // one workgroup per target molecule

    // COM-based early rejection
    if (com_distance_sq(mol_i, mol_j) > mol_cutoff_sq) { skip; }

    // Each thread handles a stride of sites in molecule i
    var site_i = thread_id;
    while (site_i < n_sites) {
        for (var s = 0u; s < n_sites; s++) {   // all sites in molecule j
            let r_sq = min_image_dist_sq(site_i_pos, site_j_pos);
            if (r_sq < cutoff_sq) {
                energy += spline_energy(type_i, type_j, r_sq);
            }
        }
        site_i += 256u;
    }

    // Workgroup parallel reduction → pairwise_output[mol_j]
}
```

Dispatch: `dispatch_workgroups(N_molecules, 1, 1)` computes one full row of
the pairwise matrix in a single GPU submission.

**Spline evaluation in the shader** uses the same Horner method as the CPU:

```wgsl
fn spline_energy(type_i: u32, type_j: u32, r_sq: f32) -> f32 {
    let params = spline_params[type_i * n_types + type_j];
    let r = sqrt(r_sq);
    let t = (r - params.r_min) / (params.r_max - params.r_min);
    let x = sqrt(t);                           // PowerLaw2 grid mapping
    let idx = min(u32(x * f32(n_intervals)), n_intervals - 1u);
    let frac = x * f32(n_intervals) - f32(idx);
    let c = spline_coeffs[params.coeff_offset + idx].u;
    return c.x + frac * (c.y + frac * (c.z + frac * c.w));  // 4 FMAs
}
```

### 7.4 Batched independent moves

A further optimization: identify molecules whose interaction neighborhoods
don't overlap (COM distance > 2 × cutoff), propose moves on all of them
simultaneously, and compute all `e_new` values in a single GPU dispatch.

```
1. select_independent_molecules() → [i₁, i₂, ..., iₖ]  (k ≤ 16)
2. e_old[k] = mol_energies[iₖ]                          → k × O(1) from cache
3. propose moves on all k molecules
4. dispatch_workgroups(k, N_molecules, 1)                → single GPU submission
5. accept/reject each independently
```

This amortizes GPU launch and readback overhead across multiple moves. For
400 molecules with ~16 independent moves per batch, throughput increases
significantly compared to sequential single-molecule moves.

### 7.5 Spline data bridge: interatomic → GPU

`GpuSplineData::from_matrix()` extracts spline coefficients from faunus-rs's
`NonbondedMatrixSplined` into GPU-aligned buffers:

```rust
pub struct GpuSplineParams {    // 32 bytes, vec4-aligned
    pub r_min: f32,
    pub r_max: f32,
    pub n_coeffs: u32,
    pub coeff_offset: u32,      // offset into global coefficients buffer
    pub u_at_rmin: f32,
    pub _pad: [f32; 3],
}

pub struct GpuSplineCoeffs {    // 16 bytes = vec4<f32>
    pub u: [f32; 4],            // Horner coefficients
}
```

One `GpuSplineParams` per atom-type pair (N_types × N_types), all spline
intervals concatenated into a single `Vec<GpuSplineCoeffs>`. The pair index
`type_i * n_types + type_j` gives O(1) lookup into the params array, and
`coeff_offset + interval_index` indexes into the coefficients array.

### 7.6 Integration with faunus-rs

The GPU MC cache does **not** require a new `Context` platform. `Context` is a
storage/transaction abstraction (backup/undo for particles); the energy cache
sits inside the energy term:

```
Context (unchanged — CPU-side state + backup/undo)
  └─ Hamiltonian
       └─ NonbondedMatrixSplined
            └─ PairwiseEnergyCache (new, optional)
                 ├─ pairwise_cache: Vec<f64>     (N_groups × N_groups)
                 ├─ mol_energies: Vec<f64>        (cached row sums)
                 ├─ dirty: Vec<bool>
                 └─ backend: CpuBackend | GpuBackend(wgpu)
```

**Mapping to faunus-rs concepts:**

| mcgpu | faunus-rs |
|---|---|
| `System.molecules[i]` | `context.groups()[i]` |
| `mol_type.sites_lab` | particles in group range |
| `EnergyBackend::molecule_energy(sys, i)` | `NonbondedMatrixSplined::group_with_other_groups(ctx, i)` |
| `invalidate_molecule(i)` | triggered by `Change::SingleGroup(i, RigidBody)` |
| `multi_step` (independent moves) | new `Propagate` enhancement |
| `EnergyBackend` enum (Gpu/Cpu) | same pattern inside energy term |

**What faunus-rs already has:**
- `NonbondedMatrixSplined` stores splined potentials per atom-type pair
- `Change::SingleGroup(i, GroupChange::RigidBody)` signals inter-group-only changes
- `group_with_other_groups()` computes exactly the "one row" the cache needs
- Energy terms have `save_backup()` / `undo()` for cache state management

**What would be needed:**
1. Pairwise group-group energy cache (benefits CPU too, independent of GPU)
2. GPU backend adapter using wgpu (following mcgpu's proven pattern)
3. Batched independent moves in `Propagate` (optional, for throughput)

---

## 8. CPU SIMD Inner Loop — No New Platform Needed

### 8.1 Why a SimdPlatform with SoA storage is not the answer

A natural question is whether a second `Context` implementation with
Structure-of-Arrays (SoA) particle layout would enable SIMD. It would not,
because the bottleneck is the **scalar loop structure**, not the memory layout.

The current hot path in `NonbondedMatrix::particle_with_particle()` processes
one pair at a time:

```rust
fn particle_with_particle(&self, context: &impl Context, i: usize, j: usize) -> f64 {
    let distance_squared = context.get_distance_squared(i, j);   // 1 pair
    self.potentials
        .get((context.get_atomkind(i), context.get_atomkind(j)))
        .isotropic_twobody_energy(distance_squared)               // 1 eval
}
```

Called inside a scalar iterator:

```rust
group.iter_active()
    .filter(|j| *j != i)
    .map(|j| self.particle_with_particle(context, i, j))
    .sum()
```

Even with SoA storage (`xs: Vec<f64>, ys: Vec<f64>, zs: Vec<f64>`), this
iterator still processes one `j` per iteration. SIMD requires processing 4+
values of `j` simultaneously — a loop restructure, not a storage change.

### 8.2 What SIMD actually needs

| Requirement | Current AoS `Vec<Particle>` | Hypothetical SoA | What matters |
|---|---|---|---|
| Contiguous x for SIMD load | Strided (x,y,z,id interleaved) | Yes | Loop must batch 4+ j's |
| Distance² for 4 pairs | 4 gathers | 4 contiguous loads/axis | Loop must emit `f64x4` |
| Potential eval for 4 r² | 4 scalar lookups | 4 scalar lookups | `energy_x4()` already exists |

The `interatomic` crate already provides SIMD spline evaluation:
- `SplineTableSimd::energy_x4(rsq: f64x4)` — 4-wide f64
- `SplineTableSimdF32::energy_x4(rsq: f32x4)` — 4-wide f32
- `SplineTableSimdF32::energy_simd(rsq: SimdF32)` — 4 or 8 wide depending on arch

The missing piece is a **batched inner loop** that feeds these functions.

### 8.3 The right fix: SIMD inner loop in the energy term

The SIMD loop belongs in `NonbondedMatrix<SplinedPotential>`, not in Context:

```rust
fn group_with_group_simd(
    &self, particles: &[Particle], cell: &Cell,
    group_i: &Group, group_j: &Group,
) -> f64 {
    let spline = &self.simd_table;  // SplineTableSimdF32
    let mut sum = 0.0f32;

    for i in group_i.iter_active() {
        let pi = particles[i].pos;

        // Process 4 j-particles at a time
        let active_j: Vec<usize> = group_j.iter_active().collect();
        for chunk in active_j.chunks(4) {
            let mut rsq = f32x4::splat(f32::MAX);  // beyond cutoff by default

            for (lane, &j) in chunk.iter().enumerate() {
                let dr = cell.distance(pi, particles[j].pos);
                rsq = rsq.replace(lane, dr.norm_squared() as f32);
            }

            sum += spline.energy_x4(rsq).reduce_add();
        }
    }
    sum as f64
}
```

This uses the existing AoS particle layout — the 4 gathers per chunk are
cheap compared to the spline evaluation. On modern CPUs (Apple M-series,
Intel with AVX2), gather from AoS is ~2 cycles per element vs ~8 cycles for
the Horner polynomial, so SoA would save <10% while a SIMD inner loop gives
2–4x.

### 8.4 Particle data is already contiguous per group

Groups store contiguous particle ranges, so positions within a group are
already sequential in memory:

```rust
let range = group.range_active();
let particles: &[Particle] = &context.particles()[range];
```

No SoA transformation needed for the energy term to access a batch of
positions. The `Context` trait's `particles()` slice provides the raw data;
the SIMD batching is an internal optimization of the energy calculation.

### 8.5 A second platform is not justified

| Approach | Effort | Expected speedup | Verdict |
|---|---|---|---|
| SimdPlatform (SoA storage) | High — new Context impl, duplicate backup/undo, all analysis must work with both | ~5–10% from aligned loads | Not worth it |
| SIMD inner loop in energy term | Moderate — new methods on `NonbondedMatrix<SplinedPotential>` | 2–4x on hot loop | Correct approach |
| Pairwise group-group cache (§7.2) | Low — CPU-only optimization | 2x (eliminates e_old) | Do first |

The `Context` trait is valuable as a single-platform abstraction for storage
and transactions (backup/undo). The generic `T: Context` costs almost nothing
and leaves the door open for a genuine future need (e.g. distributed/MPI),
but SIMD is not that need — it belongs in the energy evaluation layer.

### 8.6 Optimization priority order

1. **Pairwise energy cache** (§7.2) — 2x from eliminating redundant `e_old`,
   CPU-only, benefits all system sizes
2. **SIMD inner loop** (§8.3) — 2–4x on the remaining `e_new` computation,
   using existing `SplineTableSimd` infrastructure
3. **GPU backend** (§7.3) — additional 1.5–3x for large rigid bodies (>100
   sites/molecule), using spline coefficients already extracted in step 2

These are composable: cache + SIMD gives ~4x on CPU before touching GPU code.
For large rigid bodies, cache + GPU gives 3–7x (mcgpu benchmarks).

---

## 9. Alternative: CubeCL (write-once Rust → GPU)

The CubeCL crate (github.com/tracel-ai/cubecl) allows writing GPU kernels in
Rust with a `#[cube]` macro that compiles to WGSL (wgpu), CUDA, and ROCm.
This would give the same write-once property without maintaining WGSL files.
Still maturing but architecturally interesting — could replace hand-written WGSL
for the integrator while keeping the spline coefficient upload strategy.

---

## 10. Summary

### MD (Langevin dynamics)

| Component | Strategy |
|---|---|
| Pair forces (bottleneck) | Spline any `IsotropicTwobodyEnergy` via existing `SplinedPotential`, upload `[f32; 4]` coefficients, one generic WGSL kernel |
| Angle forces | Spline `ThreebodyAngleEnergy` over [0, π], same coefficient format and kernel pattern |
| Dihedral forces | Spline `FourbodyAngleEnergy` over [-π, π] with periodic wrapping, same pattern |
| Langevin integrator | BAOAB 3-kernel split (following OpenMM), embarrassingly parallel, f32-safe |
| RNG | Philox4x32-10 in WGSL, stateless with (step, particle_id) counter |
| Cross-backend | wgpu/naga handles Vulkan/Metal/DX12/OpenGL — no per-backend code needed |
| Precision | f32 throughout; Kahan summation for energy; constraints on CPU |
| New potentials | Zero GPU code changes — spline on CPU, upload coefficients |

### MC (Monte Carlo)

| Component | Strategy |
|---|---|
| When GPU helps | Large rigid bodies (>100 sites/molecule, >50 molecules) |
| When GPU does not help | Small molecules (water, ions) — kernel launch overhead dominates |
| Pairwise energy cache | N×N group-group matrix with dirty flags; e_old is O(1), e_new recomputes one row on GPU |
| GPU kernel | One workgroup per target molecule, 256 threads distribute sites, workgroup reduction |
| Spline bridge | `GpuSplineData::from_matrix()` extracts from `NonbondedMatrixSplined` |
| Batched moves | Select spatially independent molecules, single `dispatch_workgroups(k, N, 1)` |
| Architecture impact | Cache inside energy term, not a new platform; `Context` unchanged |
| CPU benefit | Pairwise cache alone gives 1.9x on CPU (independent of GPU) |

### CPU SIMD and platform architecture

| Component | Strategy |
|---|---|
| SIMD inner loop | Batch 4+ j-particles per iteration, feed `SplineTableSimd::energy_x4()` |
| SoA particle layout | Not needed — AoS gather cost is <10% of spline evaluation |
| SimdPlatform | Not justified — SIMD belongs in the energy term, not in Context |
| Single platform | `ReferencePlatform` is sufficient; `T: Context` generic costs nothing |
| Optimization order | 1. Pairwise cache (2x) → 2. SIMD inner loop (2–4x) → 3. GPU backend (1.5–3x) |

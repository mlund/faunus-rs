# Phase-Switch Monte Carlo for Solubility of Rigid Molecules

## Background and Related Work

The phase-switch method originates from a family of lattice-switch techniques:

1. **Lattice-switch MC** ([doi:10.1103/PhysRevLett.79.3002](https://doi.org/10.1103/PhysRevLett.79.3002), Bruce et al., 1997).
   Computes free energy differences between two solid phases (e.g., FCC vs HCP) by switching
   lattice vectors while preserving displacements. Extended to soft potentials (LJ, Morse) in
   [doi:10.1103/PhysRevE.65.036710](https://doi.org/10.1103/PhysRevE.65.036710).

2. **Phase-switch MC** ([doi:10.1103/PhysRevLett.85.5138](https://doi.org/10.1103/PhysRevLett.85.5138), Wilding & Bruce, 2000).
   Generalizes lattice-switch to solid↔fluid coexistence by introducing an overlap/tether order
   parameter and multicanonical biasing.

3. **Solid-liquid coexistence for soft potentials** ([doi:10.1063/1.1642588](https://doi.org/10.1063/1.1642588), Wilding & Bruce, 2004).
   Extended phase-switch to Lennard-Jones using transition matrix methods.

Existing implementations include [`monteswitch`](https://github.com/tomlunderwood/monteswitch) (Fortran,
solid-solid only) and [DL_MONTE](https://dl_monte.gitlab.io/dl_monte-tutorials-pages/tutorial7.html)
(general-purpose MC with lattice-switch support).

The phase-switch approach has not been applied to molecular solubility. The current state of the art
decomposes solubility via a thermodynamic cycle through the gas phase:

```
Delta_G_sol = Delta_G_sublimation + Delta_G_solvation
            = (G_gas - G_crystal) + (G_solution - G_gas)
```

Computing crystal↔fluid free energy directly — bypassing the gas-phase intermediate — is the
novelty of applying phase-switch to solubility.

## Method Summary

The phase-switch method computes the free energy difference between crystal (CS) and fluid (F) phases
in a single MC simulation. Solubility follows from the transfer free energy per molecule:

```
Delta_g = g_CS - g_F = (1/N) ln(R_{F,CS})
```

where `R_{F,CS}` is the ratio of configurational weights of the two phases (Eq. 5-6 in the original paper).

The method requires:
1. An order parameter `M` that distinguishes crystal-like from fluid-like configurations
2. Flat histogram sampling (Wang-Landau) on `M` to bridge the free energy barrier
3. A phase-switch MC move that swaps the reference lattice between phases

## Existing Faunus Infrastructure

| Component | Status | Location |
|-----------|--------|----------|
| Rigid body MC moves | Done | `src/montecarlo/{translate,rotate}.rs` |
| Volume moves (NPT) | Done | `src/montecarlo/volume.rs` |
| Wang-Landau / flat histogram | Done | `src/wang_landau.rs`, `src/flat_histogram.rs` |
| Collective variable framework | Done | `src/collective_variable/` |
| Penalty bias energy | Done | `src/energy/penalty.rs` |
| Insertion policies | Done | `src/topology/block.rs` |

## New Components

### 1. Insertion Policy: `CrystalCOM`

New variant in `InsertionPolicy` (`src/topology/block.rs`) that places rigid molecules on crystal
lattice sites derived from PDB symmetry:

```rust
CrystalCOM {
    pdb: InputPath,                   // PDB with CRYST1 + asymmetric unit
    supercell: [usize; 3],           // unit cell replicas [nx, ny, nz]
    #[serde(default)]
    rotate: bool,
}
```

Implementation in `get_positions()`:
1. Parse CRYST1 record for unit cell parameters (a, b, c, alpha, beta, gamma)
2. Look up space group symmetry operations (common protein space groups: P2_12_12_1, P2_1, C2, ...)
3. Apply symmetry ops to ASU coordinates to generate full unit cell
4. Replicate over supercell dimensions
5. For each molecule: extract COM position and orientation quaternion (by fitting to ASU)
6. Store lattice sites as the crystal reference configuration {R}^CS

The `pdbtbx` crate handles PDB parsing including CRYST1 and symmetry records.

YAML usage:
```yaml
insertmolecules:
  - molecule: lysozyme
    count: 32
    insert: !CrystalCOM
      pdb: 1dpx.pdb
      supercell: [2, 2, 2]
```

### 2. Collective Variable: `PhaseOverlap`

New `CvKind` implementor (`src/collective_variable/`) measuring the overlap order parameter `M`:

```
M = sum_i [ O_i (1 - theta(u_i - u_c)) + T_i theta(u_i - u_c) ]
```

where:
- `u_i` = displacement from reference site (translational + orientational for rigid molecules)
- `O_i` = overlap with neighbors (counts clashes that a phase switch would create)
- `T_i = alpha * u_i` = tether to lattice site
- `u_c` = cutoff separating overlap and tether domains
- `theta` = step function

For rigid molecules, the displacement combines COM distance and quaternion distance:
```
u_i = sqrt(|r_i - R_i|^2 + lambda^2 (1 - |q_i . Q_i|^2))
```
where `lambda` weights orientational vs. translational contributions.

### 3. MC Move: `PhaseSwitch`

New `Move<T>` implementation (`src/montecarlo/`) — a global move that:

1. Stores two reference configurations: `{R}^CS` (from `CrystalCOM` insertion) and `{R}^F` (a fluid snapshot)
2. Computes displacement vectors `{u_i} = {r_i - R_i^current}` (position + orientation)
3. Proposes the switched state: `{r_i'} = {R_i^other + scaled(u_i)}`
4. Scales volume: `V' = V * (V_other / V_current)` where `V_gamma` is the equilibrium volume of phase gamma
5. Accepts/rejects via Metropolis on the effective energy change

The phase label gamma becomes a stochastic variable toggled by this move.

### 4. MC Move: `AssociationSwap`

In the crystal phase, each molecule is "associated" with a lattice site. This move picks two molecules
and swaps their site assignments. It changes the representation (which molecule sits on which site)
without changing the physical configuration. Required for ergodic sampling of the crystal phase
where direct particle permutations are suppressed.

## Implementation Plan

### Phase 1: Hard Spheres (validation)
Reproduce the original paper's hard-sphere freezing results.
- `CrystalCOM` with FCC lattice (no PDB needed, just lattice constant)
- `PhaseOverlap` CV with translational overlap only
- `PhaseSwitch` move for point particles
- Wang-Landau on M to compute coexistence pressure

### Phase 2: Rigid Molecules
Extend to rigid bodies with orientational degrees of freedom.
- Full `CrystalCOM` with PDB symmetry parsing
- Quaternion-aware displacement vectors in `PhaseOverlap`
- Orientation handling in `PhaseSwitch`

### Phase 3: Implicit Solvent
Add solvation free energy for the fluid phase.
- SASA-based or GB/SA implicit solvent energy term
- Crystal contacts replace solvent contacts; the free energy difference captures this
- Solubility: `S = exp(-Delta_g / kT)` per molecule

## Solvation Approximation for Protein Crystals

Protein crystals are ~50% solvent. The protein-water interactions are therefore partially captured
in both phases, and the solvation contributions may largely cancel in the free energy difference.
This suggests a zeroth-order approximation:

- **Crystal phase**: protein-protein interactions at lattice contacts + implicit bulk water in channels
- **Fluid phase**: single protein in the same implicit solvent model
- **Assumption**: solvation environments are similar enough that the implicit solvent terms cancel,
  leaving the phase-switch free energy dominated by crystal packing contacts

If experimental solubility `S_exp` is known, the effective solvation correction can be extracted:

```
Delta_G_solvation_correction = -kT ln(S_exp) - Delta_g_phase_switch
```

This enables a calibration workflow:
1. Run phase-switch without solvation correction for proteins with known solubility
2. Fit the systematic offset as a function of protein properties (e.g., surface area, net charge)
3. Use the calibrated correction predictively for unknown systems

## Design Considerations

- **Crystal reference from insertion**: `CrystalCOM` generates lattice sites at system setup and stores
  them for later use by `PhaseSwitch`. No separate reference file needed.
- **Fluid reference**: captured automatically as a snapshot during fluid-phase equilibration.
- **Space group library**: a small lookup table for common protein space groups suffices
  (P2_12_12_1, P2_1, C2, P4_32_12 cover ~70% of protein crystals).
- **Protein flexibility**: start with fully rigid molecules. Internal flexibility could later be treated
  as additional displacement degrees of freedom relative to a reference conformation.
- **Solvent content**: protein crystals are ~50% solvent; the implicit solvent model must be
  consistent between phases.

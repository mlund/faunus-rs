# Nonbonded Electrostatics: Factored Spline Design

## Current Architecture

`PairPot` = `ShortRange` + `Coulomb` evaluated as a sum per pair. When splined, the **combined** SR+Coulomb is tabulated as **one spline per atom-type pair**. Problems:

1. **Spline accuracy** — Coulomb is long-ranged, requiring many nodes (default 2000) even though SR potentials could get away with far fewer
2. **Obscured contributions** — can't decompose energy into SR vs electrostatic
3. **Redundant data** — the Coulomb part is `q_i*q_j * f(r)` where `f(r)` is the **same function for all pairs**, yet we store a separate spline for every `(kind_i, kind_j)` entry

## Proposed: Factored Coulomb Spline

Store the Coulomb scheme as **one shared spline** for `f(r)` (without charge product), and keep per-pair SR splines:

```
u(r^2) = spline_SR[kind_i, kind_j](r^2) + q_i*q_j * spline_coulomb(r^2)
```

### Data structure change

```rust
pub struct NonbondedMatrix<P = PairPot> {
    potentials: Array2<P>,                     // SR-only splines (per pair)
    coulomb_spline: Option<SplinedPotential>,  // shared scheme spline f(r)
    charge_products: Array2<f64>,              // q_i * q_j matrix
    // ... rest unchanged
}
```

### Hot path change

```rust
// Current
energy += potentials[kind_i, kind_j].energy(rsq);

// Proposed
energy += sr_potentials[kind_i, kind_j].energy(rsq)
        + charge_products[kind_i, kind_j] * coulomb_spline.energy(rsq);
```

### Benefits

- SR splines can use ~200-500 nodes instead of 2000 (short-ranged potentials are smooth)
- Coulomb spline can use 5000+ nodes cheaply (only one copy)
- Net memory reduction despite better accuracy
- Separate energy reporting for free
- GPU kernel gets the Coulomb spline in constant memory
- `ExcludedCoulomb` term could use the same shared spline

### Performance

- Hot path: 2 spline lookups (~8 FMA) + 1 multiply vs current 1 lookup (4 FMA)
- Better accuracy-per-byte due to factored node allocation
- GPU: shared Coulomb spline fits in shared/constant memory, reducing global memory bandwidth

### Migration path

Only changes `NonbondedMatrixSplined` — the un-splined `NonbondedMatrix<PairPot>` already has clean SR/Coulomb separation via enum dispatch.

## Alternatives Considered

### Direct Coulomb evaluation (no spline)

Keep SR splined, evaluate Coulomb analytically. Clean separation but `erfc` (Ewald) is expensive per pair on both CPU and GPU.

### Hybrid: direct for simple, spline for complex

Plain Coulomb and reaction field are cheap to evaluate directly; only spline Ewald real-space. Pragmatic but adds branching complexity.

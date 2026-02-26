# Analysis

Analysis objects sample the system at a given frequency during a simulation
and optionally write results to output files.
They are defined in the `analysis` section of the YAML input.

## Collective Variable

Monitors a collective variable (CV) over the course of a simulation,
recording the instantaneous value and a running average at each sampled step.
This is useful for verifying that a CV remains within the expected range,
for convergence checking, and for post-processing of time/step series data.

If `file` is given, each sampled step writes a line with columns
`step`, `value`, and `running_average`.
The file may be gzip-compressed by using a `.gz` extension.

### Example

```yaml
analysis:
  - !CollectiveVariable
    property: mass_center_position
    selection: "molecule protein"
    dimension: z
    range: [-50.0, 50.0]
    file: cv.dat
    frequency: !Every 100
```

### Options

Key          | Required | Default | Description
------------ | -------- | ------- | -------------------------------------------
`property`   | yes      |         | CV type (see table below)
`range`      | yes      |         | Allowed `[min, max]` interval
`frequency`  | yes      |         | Sample frequency, e.g. `!Every 100`
`dimension`  | no       | `xyz`   | Axis projection (`x`, `y`, `z`, `xy`, …)
`selection`  | depends  |         | Selection expression for one atom or group
`selection2` | depends  |         | Second selection (for two-group properties)
`resolution` | no       |         | Bin width (only used by Penalty)
`file`       | no       |         | Output file path; omit to only track the mean

### Supported properties

Property                 | Selection       | Description
------------------------ | --------------- | -------------------------------------------
`volume`                 | none            | Simulation cell volume
`box_length`             | none            | Cell side length along `dimension`
`atom_position`          | one atom        | Atom position projected onto `dimension`
`size`                   | one group       | Number of active particles in a group
`end_to_end`             | one group       | End-to-end distance of a molecular group
`mass_center_position`   | one group       | Mass center position along `dimension`
`mass_center_separation` | two groups      | Distance between two group mass centers

## Polymer Shape

Computes size and shape anisotropy descriptors from the mass-weighted gyration tensor
([doi:10/d6ff](https://doi.org/10/d6ff)).
For each matching group the 3×3 symmetric gyration tensor is built as
`S_ij = (1/M) Σ m_k r_i r_j`, where `r` is the PBC-aware displacement from the
centre of mass. Eigenvalue decomposition yields the principal moments
`λ₁ ≤ λ₂ ≤ λ₃` from which the following descriptors are derived.

Following [IUPAC 2014](https://doi.org/10.1515/pac-2013-0201) nomenclature,
the per-configuration squared radius of gyration is `s² = Tr(S)` (Def. 1.14),
and the reported `Rg = √⟨s²⟩` is the root-mean-square radius of gyration
(Def. 1.16). Note that `⟨s⟩ ≠ √⟨s²⟩` in general.
The reported ratio `⟨Re²⟩/⟨Rg²⟩` equals 6 for ideal Gaussian chains
([Wall & Erpenbeck 1959](https://doi.org/10.1063/1.1730022)).

| Descriptor                    | Formula                                                 | Reference |
| ----------------------------- | ------------------------------------------------------- | --------- |
| Radius of gyration squared    | `Rg² = λ₁ + λ₂ + λ₃`                                   | [IUPAC 2014](https://doi.org/10.1515/pac-2013-0201) Def. 1.14 |
| Asphericity                   | `b = λ₃ − (λ₁ + λ₂)/2`                                 | [Aronovitz & Nelson 1986](https://doi.org/10.1051/jphys:019860047090156100) |
| Acylindricity                 | `c = λ₂ − λ₁`                                           | [Aronovitz & Nelson 1986](https://doi.org/10.1051/jphys:019860047090156100) |
| Relative shape anisotropy     | `κ² = (b² + ¾c²) / Rg⁴`  ∈ [0, 1]                      | [Rudnick & Gaspari 1986](https://doi.org/10.1007/BF01012872) |
| Prolateness                   | `S = 27(λ₁−λ̄)(λ₂−λ̄)(λ₃−λ̄) / Rg⁶`  ∈ [−0.25, 2]      | [Theodorou & Suter 1985](https://doi.org/10.1021/ma00164a001) |
| Westin linear (rod-like)      | `Cl = (λ₃ − λ₂) / Rg²`                                 | [Westin 1997](https://doi.org/10.1006/cviu.1997.0640) |
| Westin planar (disc-like)     | `Cp = 2(λ₂ − λ₁) / Rg²`                                | [Westin 1997](https://doi.org/10.1006/cviu.1997.0640) |
| Westin spherical              | `Cs = 3λ₁ / Rg²`  (Cl + Cp + Cs = 1)                   | [Westin 1997](https://doi.org/10.1006/cviu.1997.0640) |

If `file` is given (single-molecule selection only), each sampled step writes a line with
columns `step`, `Rg`, and the upper triangle of the gyration tensor
(`Sxx Sxy Sxz Syy Syz Szz`).
The file may be gzip-compressed by using a `.gz` extension.

### Example

```yaml
analysis:
  - !PolymerShape
    selection: "molecule polymer"
    file: shape.dat.gz
    frequency: !Every 100
```

### Options

Key          | Required | Default | Description
------------ | -------- | ------- | -------------------------------------------
`selection`  | yes      |         | Selection expression for molecule group(s)
`file`       | no       |         | Streaming output file (single molecule only)
`frequency`  | yes      |         | Sample frequency, e.g. `!Every 100`

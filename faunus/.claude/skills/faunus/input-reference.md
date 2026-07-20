# Faunus YAML Input Reference

For full details, read the documentation in `docs/` and working examples in `examples/`.
Regression-tested inputs (`tests/files/*/input.yaml`) are the most reliable references
since they are validated against known output on every release. The top-level file
structure is in `SKILL.md`.

## Valid example input files

All examples and regression tests use `input.yaml` as the main input file; find them with
`examples/*/input.yaml` and `tests/files/*/input.yaml`. Always read the relevant example
when generating a new config to ensure accuracy.

Unit-test topologies — partial configs for specific features:
- `tests/files/speciation_test.yaml` — speciation move setup
- `tests/files/topology_pass.yaml` — valid topology with includes
- `tests/files/translate_molecules_simulation.yaml` — molecule translation
- `tests/files/bonded_interactions.yaml` — bonded energy terms
- `tests/files/nonbonded_interactions.yaml` — nonbonded energy terms
- `tests/files/nonbonded_kimhummer.yaml` — Kim-Hummer potential
- `tests/files/nonbonded_custom.yaml` — custom pair potential
- `tests/files/sasa_interactions.yaml` — SASA energy terms
- `tests/files/cell_sphere.yaml` — spherical cell geometry

Include file fragments (not standalone):
- `examples/calvados3/calvados3.yaml` — CALVADOS3 forcefield
- `examples/sticks/duello-topology.yaml` — stick molecule topology
- `tests/files/top2.yaml`, `tests/files/top3.yaml` — partial topologies

## Where to Find Details

| Topic | Documentation | Example files |
|-------|--------------|---------------|
| Atoms, molecules, bonds | `docs/topology.md` | `tests/files/topology_pass.yaml`, `tests/files/bonded_interactions.yaml` |
| Pair potentials, Ewald, SASA | `docs/energy.md` | `tests/files/nonbonded_interactions.yaml`, `tests/files/npt_water_ewald/input.yaml` |
| Custom external / pair-COM bias (MC + LD) | `docs/energy.md#custom-external-potential`, `docs/energy.md#custom-pair-potential-com-com` | — |
| MC moves, Langevin, Gibbs | `docs/moves.md` | `examples/langevin/input.yaml`, `tests/files/gibbs_ensemble/input.yaml` |
| Trajectory, RDF, energy output | `docs/analysis.md` | `examples/twobody/input.yaml`, `examples/calvados3/input.yaml` |
| Selection expressions | `docs/selection_language.md` | used in analysis and constraints |
| NPT water | — | `tests/files/npt_water/input.yaml` |
| NPT polymers | — | `tests/files/npt_polymers/input.yaml` |
| Coarse-grained proteins | — | `examples/calvados3/input.yaml`, `examples/protein_ions/input.yaml` |
| Kim-Hummer potential | — | `examples/kimhummer/input.yaml` |
| GCMC / speciation | — | `tests/files/gcmc_ideal_gas/input.yaml`, `tests/files/gcmc_swap/input.yaml` |
| Force field include | — | `examples/calvados3/calvados3.yaml` (included by `input.yaml`) |

## Cell Types

```yaml
cell: !Cuboid [30, 30, 30]                        # 3D PBC
cell: !HexagonalPrism {side: 15, height: 30}       # hexagonal PBC
cell: !Slit [30, 30, 50]                           # PBC XY, hard walls Z
cell: !Cylinder {radius: 10, height: 50}           # PBC Z, hard walls XY
cell: !Sphere {radius: 20}                         # no PBC, hard wall
cell: !Endless                                     # infinite
```

## Unit Conventions

| Quantity | Unit |
|----------|------|
| Distance | angstrom |
| Energy | kJ/mol |
| Temperature | Kelvin |
| Mass | g/mol |
| Charge | elementary charges |
| MC displacement (`dp`) | angstrom (translate) or radians (rotate) |
| Torsion/dihedral angles | degrees |
| Langevin friction | 1/ps |
| Langevin timestep | ps |
| Pressure | specify: `!atm`, `!bar`, `!Pa`, `!kT`, `!mM` |

## Mixing Rules

`LorentzBerthelot` / `LB`, `Arithmetic`, `Geometric`, `FenderHalsey` / `FH`

## Tips

- Use `spline` tabulation for performance; add `bounding_spheres: true` for rigid molecules
- Use `replace:` for pairs that fully override `default`; `append:` for pairs that extend it
- `!Stochastic` collections for mixed molecule types; `!Deterministic` with `cycles` for sweeps
- Unknown keys are rejected (`deny_unknown_fields`): a mistyped key errors rather than being ignored

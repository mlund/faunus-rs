# Faunus YAML Quick Reference

For full details, read the documentation in `docs/` and working examples in `examples/`.

## Top-Level Structure

```yaml
include: [forcefield.yaml]      # optional: merge external YAML files
atoms: [...]                    # required: atom/bead type definitions
molecules: [...]                # required: molecular topologies
system:
  medium: {...}                 # temperature, dielectric, salt
  cell: ...                     # simulation box geometry
  blocks: [...]                 # molecule placement
  energy: {...}                 # interaction potentials
  intermolecular: {...}         # optional: cross-molecule bonds
propagate: {...}                # MC moves or Langevin dynamics
analysis: [...]                 # optional: output and sampling
```

## Where to Find Details

| Topic | Documentation | Example files |
|-------|--------------|---------------|
| Atoms, molecules, bonds | `docs/topology.md` | `tests/files/topology_pass.yaml`, `tests/files/bonded_interactions.yaml` |
| Pair potentials, Ewald, SASA | `docs/energy.md` | `tests/files/nonbonded_interactions.yaml`, `tests/files/npt_water_ewald/input.yaml` |
| MC moves, Langevin, Gibbs | `docs/moves.md` | `examples/langevin/input.yaml`, `tests/files/gibbs_ensemble/input.yaml` |
| Trajectory, RDF, energy output | `docs/analysis.md` | `examples/twobody/input.yaml`, `examples/calvados3/input.yaml` |
| Selection expressions | `docs/selection_language.md` | used in analysis and constraints |
| NPT water | — | `tests/files/npt_water/input.yaml` |
| NPT polymers | — | `tests/files/npt_polymers/input.yaml` |
| Coarse-grained proteins | — | `examples/calvados3/input.yaml`, `examples/protein_ions/input.yaml` |
| Kim-Hummer potential | — | `examples/kimhummer/input.yaml` |
| GCMC / speciation | — | `tests/files/gcmc_ideal_gas/input.yaml`, `tests/files/gcmc_swap/input.yaml` |
| Force field include | — | `examples/calvados3/calvados3.yaml` (included by `input.yaml`) |

Regression-tested inputs (`tests/files/*/input.yaml`) are the most reliable references
since they are validated against known output on every release.

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

# Units and Conventions

Faunus inputs use explicit physical units where a field requires them. Unless a
field or YAML tag states otherwise, use the conventions below.

| Quantity | Unit or convention | Common fields |
|----------|--------------------|---------------|
| Length | Å | Cell dimensions, positions, cutoffs, `sigma`, bond lengths, histogram bins |
| Energy | kJ/mol | Pair potentials, bonded terms, tabulated potentials, energy output |
| Temperature | K | `system.medium.temperature` |
| Charge | Elementary charge, `e` | Atom `charge` values and electrostatic analyses |
| Mass | g/mol | Atom `mass` values and PSF output |
| Activity or concentration | mol/L | Atom and molecule `activity`, salt concentrations, molarity collective variables |
| Pressure | Tagged units | `!atm`, `!bar`, `!Pa`, `!kT`, and `!mM` in `system.energy.pressure` |
| Topology angles | Degrees | Torsion and dihedral equilibrium angles such as `aeq` and `phi` |
| Move angles | Radians | Angular MC move sizes such as `!RotateMolecule {max_angle: ...}` |

## Input conventions

YAML tags select physical models, units, and constructors. Examples include
`!Water` for the dielectric model, `!NaCl` for salt, `!Cuboid` for a cell, and
`!atm` for pressure.

In a complete simulation input, energy terms belong under `system.energy`.
Included force-field fragments may use a top-level `energy` section that is
merged into the complete input.

Selection expressions are strings, for example `"molecule protein and name CA"`.
They use the selection language described in
[Selection Language](selection_language.md).

Use YAML comments (`#`) for lines or blocks. Faunus also ignores top-level keys
whose name starts with `_`, which is useful for temporarily disabling whole
sections while keeping the input valid YAML. Template inputs can also use
MiniJinja block comments; see [Comments](index.md#comments).

## Names

Atom type names come from the top-level `atoms` list and are used by force-field
parameters and `atomtype` selections. Per-atom names come from molecule
definitions, structure files, or `atom_names`, and are used by `name`
selections.

Molecule names identify molecule types in `molecules`, `system.blocks`, moves,
and analyses. In selection strings, quote names that contain spaces or collide
with selection keywords.

## Output conventions

The main `output.yaml` file reports the final simulation summary, energy terms,
move statistics, and analysis summaries. Analyses with a `file:` field write
their own data files. Column headers include units where possible, for example
`z/Å`, `W/kJ/mol`, or `molarity/mol·L⁻¹`.

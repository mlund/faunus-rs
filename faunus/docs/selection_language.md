# Selection Language

Faunus uses a VMD-like selection language for picking atoms and groups
in analysis, collective variables, and other contexts.
Expressions are case-insensitive and support boolean logic and glob patterns.

## Boolean operators

Combine expressions with `and`, `or`, `not`, and parentheses:

```text
protein and backbone
not molecule water
(chain A or chain B) and hydrophobic
```

## Keyword reference

### Parametric keywords

These take one or more arguments (names, glob patterns, or numeric ranges).

Keyword              | Aliases                       | Description
-------------------- | ----------------------------- | -----------
`chain`              | `segid`                       | Chain / segment identifier
`resname`            | `resn`                        | Residue name
`resid`              | `resi`, `resseq`, `resnum`    | Residue sequence number (ranges with `to` or `:`)
`name`               | `atomname`                    | Per-instance atom name
`element`            | `elem`                        | Chemical element symbol
`atomtype`           | `type`                        | Force-field atom type name
`atomid`             |                               | Atom kind id (ranges with `to` or `:`)
`molecule`           |                               | Molecule kind name

#### Glob patterns

Name arguments support glob wildcards:

- `*` matches any number of characters
- `?` matches exactly one character
- `[abc]` matches any character in the set
- `[a-z]` matches any character in the range

Examples: `atomtype C*`, `name "C[AB]"`, `resname A??`.

#### Numeric ranges

`resid` and `atomid` accept single values, `to` ranges, or colon ranges:

```text
resid 42
resid 10 to 20
resid 10:20
atomid 0 to 5
```

### Standalone keywords

These take no arguments and match by residue name.
For coarse-grained models without residue information, the atom type name
is used as fallback.

Keyword        | Matched residues
-------------- | ----------------
`all`          | Everything
`none`         | Nothing
`protein`      | ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL HIE HID HIP CYX ASH GLH LYN NTR CTR
`backbone`     | Protein residues with atom name C, CA, N, or O
`sidechain`    | Protein residues excluding backbone atoms
`nucleic`      | DA DT DG DC DU A U G C RA RU RG RC
`hydrophobic`  | ALA VAL ILE LEU MET PHE TRP PRO GLY
`aromatic`     | PHE TYR TRP HIS HIE HID HIP
`acidic`       | ASP GLU ASH GLH CTR
`basic`        | ARG LYS HIS HIE HID HIP LYN NTR
`polar`        | SER THR ASN GLN CYS CYX TYR
`charged`      | Acidic or basic residues

## Resolution modes

A selection can be resolved in two ways:

- **Atom-level** (`resolve_atoms`): returns absolute particle indices of all matching atoms.
- **Group-level** (`resolve_groups`): returns group indices where _any_ active atom matches.

The resolution mode depends on the analysis.
For example, partial energy analysis uses atom-level resolution
so that atom-type filters (e.g. `hydrophobic and molecule MOL1`) select
only the matching atoms rather than the whole molecule.

## Examples

```yaml
# All sodium atoms
selections: ["atomtype Na", "atomtype Cl"]

# Specific molecule
selections: ["molecule MOL1", "molecule MOL2"]

# Atom-level subset of a molecule
selections: ["hydrophobic and molecule MOL1", "hydrophobic and molecule MOL2"]

# Residue range in chain A
selection: "resid 10 to 50 and chain A"

# Boolean combination
selection: "(protein and not backbone) or molecule ligand"
```

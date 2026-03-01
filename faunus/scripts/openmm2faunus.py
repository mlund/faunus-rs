#!/usr/bin/env python3
"""
Convert an OpenMM-parametrized system to a Faunus YAML topology + XYZ structure file.

Usage:
    python openmm2faunus.py input.pdb [--ff amber14-all.xml] [-o topology.yaml]

Requires:
    - OpenMM: conda install -c conda-forge openmm
    - PyYAML: pip install pyyaml

Extracted:
    - Atom types with nonbonded parameters (charge, sigma, epsilon)
    - Harmonic bonds, harmonic angles, periodic (proper) dihedrals
    - excluded_neighbours (1-2, 1-3, 1-4 exclusions for multi-atom molecules)
    - 1-4 scaling factors (electrostatic_scaling, lj_scaling) on dihedrals
    - Residue structure, simulation cell dimensions

TODO:
    - Improper dihedrals (separate from proper)
    - CMAP cross-terms
    - Urey-Bradley angle terms
    - Energy section (nonbonded pair potentials)
    - Chain annotations
    - degrees_of_freedom
"""

import argparse
import math
import os
import sys
from collections import defaultdict, deque, namedtuple

try:
    from openmm.app import PDBFile, ForceField, Modeller
    from openmm import (
        NonbondedForce,
        HarmonicBondForce,
        HarmonicAngleForce,
        PeriodicTorsionForce,
        unit,
    )
except ImportError:
    sys.exit("Error: OpenMM required. Install: conda install -c conda-forge openmm")

try:
    import yaml
except ImportError:
    sys.exit("Error: PyYAML required. Install: pip install pyyaml")

_ZERO_TOL = 1e-12
_DEFAULT_NB = {"charge": 0.0, "sigma": 0.0, "epsilon": 0.0}


def _get_force(system, force_type):
    """Return the first force of the given type, or None."""
    return next((f for f in system.getForces() if isinstance(f, force_type)), None)


def extract_nonbonded(system):
    """Extract per-atom charge, sigma (Å), epsilon (kJ/mol) from NonbondedForce."""
    force = _get_force(system, NonbondedForce)
    if force is None:
        return {}
    params = {}
    for i in range(force.getNumParticles()):
        charge, sigma, epsilon = force.getParticleParameters(i)
        params[i] = {
            "charge": charge.value_in_unit(unit.elementary_charge),
            "sigma": sigma.value_in_unit(unit.angstrom),
            "epsilon": epsilon.value_in_unit(unit.kilojoule_per_mole),
        }
    return params


def extract_bonds(system):
    """Extract harmonic bonds as global index pairs with parameters."""
    force = _get_force(system, HarmonicBondForce)
    if force is None:
        return []
    bonds = []
    for i in range(force.getNumBonds()):
        idx1, idx2, req, k = force.getBondParameters(i)
        bonds.append(
            {
                "i": idx1,
                "j": idx2,
                "req": req.value_in_unit(unit.angstrom),
                "k": k.value_in_unit(unit.kilojoule_per_mole / unit.angstrom**2),
            }
        )
    return bonds


def extract_angles(system):
    """Extract harmonic angles as global index triples with parameters.

    Converts equilibrium angle from radians to degrees and spring constant
    from kJ/mol/rad² to kJ/mol/deg².
    """
    force = _get_force(system, HarmonicAngleForce)
    if force is None:
        return []
    angles = []
    for i in range(force.getNumAngles()):
        idx1, idx2, idx3, aeq, k = force.getAngleParameters(i)
        angles.append(
            {
                "idx": [idx1, idx2, idx3],
                "aeq": aeq.value_in_unit(unit.degree),
                "k": k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)
                * (math.pi / 180.0) ** 2,
            }
        )
    return angles


def extract_dihedrals(system):
    """Extract periodic torsion dihedrals as global index quartets with parameters.

    Converts phase angle from radians to degrees. Spring constant stays in kJ/mol.
    AMBER uses multiple entries per atom quartet (Fourier series), each becomes
    a separate dihedral entry.
    """
    force = _get_force(system, PeriodicTorsionForce)
    if force is None:
        return []
    dihedrals = []
    for i in range(force.getNumTorsions()):
        idx1, idx2, idx3, idx4, n, phase, k = force.getTorsionParameters(i)
        dihedrals.append(
            {
                "idx": [idx1, idx2, idx3, idx4],
                "n": float(n),
                "phi": phase.value_in_unit(unit.degree),
                "k": k.value_in_unit(unit.kilojoule_per_mole),
            }
        )
    return dihedrals


def extract_14_scaling(system):
    """Extract 1-4 exception scaling factors from NonbondedForce.

    OpenMM stores pre-scaled exception parameters for 1-4 pairs; recover the
    scaling factors by comparing to unscaled particle parameters.

    Returns dict mapping frozenset({i, j}) -> {electrostatic_scaling, lj_scaling}.
    """
    nb_force = _get_force(system, NonbondedForce)
    if nb_force is None:
        return {}

    exceptions = {}
    for idx in range(nb_force.getNumExceptions()):
        i, j, chargeprod, _sigma, epsilon = nb_force.getExceptionParameters(idx)
        chargeprod = chargeprod.value_in_unit(unit.elementary_charge**2)
        epsilon = epsilon.value_in_unit(unit.kilojoule_per_mole)

        # Skip fully excluded pairs (1-2 and 1-3 bonds)
        if abs(chargeprod) < _ZERO_TOL and abs(epsilon) < _ZERO_TOL:
            continue

        # Recover scaling factors from particle parameters
        qi = nb_force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        qj = nb_force.getParticleParameters(j)[0].value_in_unit(unit.elementary_charge)
        eps_i = nb_force.getParticleParameters(i)[2].value_in_unit(
            unit.kilojoule_per_mole
        )
        eps_j = nb_force.getParticleParameters(j)[2].value_in_unit(
            unit.kilojoule_per_mole
        )

        q_full = qi * qj
        eps_full = math.sqrt(eps_i * eps_j) if eps_i > 0 and eps_j > 0 else 0.0

        elec_scale = chargeprod / q_full if abs(q_full) > _ZERO_TOL else 0.0
        lj_scale = epsilon / eps_full if abs(eps_full) > _ZERO_TOL else 0.0

        exceptions[frozenset((i, j))] = {
            "electrostatic_scaling": round(elec_scale, 6),
            "lj_scaling": round(lj_scale, 6),
        }
    return exceptions


def _param_key(mass, element, charge, sigma, epsilon):
    """Round parameters to create a hashable identity for deduplication."""
    return (
        round(mass, 3),
        element,
        round(charge, 4),
        round(sigma, 3),
        round(epsilon, 4),
    )


def collect_atom_kinds(topology, nb_params):
    """Build unique atom kinds, merging those with identical parameters.

    Atoms with the same name AND the same physical parameters (mass, charge,
    sigma, epsilon) are merged into a single kind. When two atoms share a name
    but differ in parameters (e.g. backbone "C" in ACE vs ALA with slightly
    different charges), the residue name is appended to disambiguate.

    Returns (atomkinds dict, per-atom kind name list).
    """
    # First pass: collect (name, residue) -> parameters
    raw_kinds = {}  # (atom_name, res_name) -> entry dict
    for atom in topology.atoms():
        idx = atom.index
        kind_key = (atom.name, atom.residue.name)
        if kind_key not in raw_kinds:
            element = atom.element.symbol if atom.element else "X"
            mass = atom.element.mass.value_in_unit(unit.dalton) if atom.element else 0.0
            nb = nb_params.get(idx, _DEFAULT_NB)
            raw_kinds[kind_key] = {
                "mass": mass,
                "element": element,
                "charge": nb["charge"],
                "sigma": nb["sigma"],
                "epsilon": nb["epsilon"],
            }

    # Second pass: group by (atom_name, param_key) to merge across residues
    # e.g. "C" in ACE and "C" in ALA are merged if parameters match
    merged = {}  # param_identity -> entry dict with final name
    kind_key_to_name = {}  # (atom_name, res_name) -> final kind name

    for (atom_name, res_name), params in raw_kinds.items():
        pk = _param_key(
            params["mass"],
            params["element"],
            params["charge"],
            params["sigma"],
            params["epsilon"],
        )
        merge_key = (atom_name, pk)

        if merge_key in merged:
            # Same name, same parameters — reuse
            kind_key_to_name[(atom_name, res_name)] = merged[merge_key]["name"]
        else:
            # Check if atom_name already used with *different* parameters
            name_conflict = any(k[0] == atom_name and k[1] != pk for k in merged)
            final_name = f"{atom_name}_{res_name}" if name_conflict else atom_name
            entry = {"name": final_name, "mass": round(params["mass"], 4)}
            if params["element"]:
                entry["element"] = params["element"]
            if abs(params["charge"]) > 1e-8:
                entry["charge"] = round(params["charge"], 6)
            if params["sigma"] > 0:
                entry["sigma"] = round(params["sigma"], 4)
            if params["epsilon"] > 0:
                entry["epsilon"] = round(params["epsilon"], 6)
            merged[merge_key] = entry
            kind_key_to_name[(atom_name, res_name)] = final_name

    per_atom_kind_name = [
        kind_key_to_name[(atom.name, atom.residue.name)] for atom in topology.atoms()
    ]

    # Deduplicate the final list (preserve order)
    seen = set()
    atomkinds = {}
    for entry in merged.values():
        if entry["name"] not in seen:
            seen.add(entry["name"])
            atomkinds[entry["name"]] = entry

    return atomkinds, per_atom_kind_name


class MoleculeInstance:
    """A single molecule instance with topology and coordinates."""

    def __init__(
        self,
        global_indices,
        atom_kinds,
        atom_names,
        residues,
        bonds,
        torsions,
        dihedrals,
        positions,
    ):
        self.global_indices = global_indices
        self.atom_kinds = atom_kinds  # list of atom kind names
        self.atom_names = atom_names  # list of per-atom PDB names
        self.residues = residues  # list of {name, number, range}
        self.bonds = bonds  # list of {index, k, req} with local indices
        # "torsion" is faunus terminology for 3-body angle potentials
        self.torsions = torsions  # list of {index, k, aeq} with local indices
        self.dihedrals = dihedrals  # list of {index, k, n, phi} with local indices
        self.positions = positions  # list of (x, y, z) in angstrom

    def fingerprint(self):
        """Identity fingerprint for deduplication.

        Two molecules are the same kind if they share identical atom kind
        sequences, bond/torsion/dihedral topology, and residue structure.
        """
        bond_tuples = tuple(
            (*b["index"], round(b["k"], 2), round(b["req"], 3)) for b in self.bonds
        )
        torsion_tuples = tuple(
            (*t["index"], round(t["k"], 2), round(t["aeq"], 3)) for t in self.torsions
        )
        dihedral_tuples = tuple(
            (
                *d["index"],
                round(d["k"], 2),
                round(d["n"], 1),
                round(d["phi"], 3),
                round(d.get("electrostatic_scaling", 0), 4),
                round(d.get("lj_scaling", 0), 4),
            )
            for d in self.dihedrals
        )
        residue_tuples = tuple(
            (r["name"], r["range"][1] - r["range"][0]) for r in self.residues
        )
        return (
            tuple(self.atom_kinds),
            bond_tuples,
            torsion_tuples,
            dihedral_tuples,
            residue_tuples,
        )


MoleculeGroup = namedtuple("MoleculeGroup", ["template", "instances", "name"])


def _find_connected_components(topology):
    """Find connected components of the molecular bond graph via BFS.

    Returns list of sorted atom index lists, one per molecule.
    """
    n_atoms = topology.getNumAtoms()

    adjacency = [[] for _ in range(n_atoms)]
    for bond in topology.bonds():
        i, j = bond[0].index, bond[1].index
        adjacency[i].append(j)
        adjacency[j].append(i)

    visited = [False] * n_atoms
    components = []
    for start in range(n_atoms):
        if visited[start]:
            continue
        component = []
        queue = deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        component.sort()
        components.append(component)

    return components


def _build_molecule_instance(
    component,
    per_atom_kind_name,
    atom_by_index,
    positions_ang,
    bond_by_atom,
    angle_by_atom,
    dihedral_by_atom,
    all_14_exceptions,
):
    """Build a MoleculeInstance from a connected component of global atom indices."""
    index_set = set(component)
    offset = component[0]

    atom_kinds = [per_atom_kind_name[i] for i in component]
    atom_names = [atom_by_index[i].name for i in component]
    mol_positions = [positions_ang[i] for i in component]

    # Residue ranges (local indices within molecule)
    residues = []
    current_res_key = None
    for local_idx, global_idx in enumerate(component):
        atom_obj = atom_by_index[global_idx]
        res_key = (
            atom_obj.residue.chain.id,
            atom_obj.residue.id,
            atom_obj.residue.name,
        )
        if res_key != current_res_key:
            if residues:
                residues[-1]["range"][1] = local_idx
            res_number = atom_obj.residue.id
            if isinstance(res_number, str):
                try:
                    res_number = int(res_number)
                except ValueError:
                    res_number = None
            residues.append(
                {
                    "name": atom_obj.residue.name,
                    "number": res_number,
                    "range": [local_idx, 0],
                }
            )
            current_res_key = res_key
    if residues:
        residues[-1]["range"][1] = len(component)

    # Intramolecular harmonic bonds (local indices)
    local_bonds = []
    seen_bonds = set()
    for global_idx in component:
        for b in bond_by_atom[global_idx]:
            if b["i"] in index_set and b["j"] in index_set:
                pair = (min(b["i"], b["j"]), max(b["i"], b["j"]))
                if pair not in seen_bonds:
                    seen_bonds.add(pair)
                    local_bonds.append(
                        {
                            "index": [b["i"] - offset, b["j"] - offset],
                            "k": round(b["k"], 4),
                            "req": round(b["req"], 4),
                        }
                    )

    # Intramolecular harmonic angles (faunus calls these "torsions")
    local_torsions = []
    seen_torsions = set()
    for global_idx in component:
        for a in angle_by_atom[global_idx]:
            ai, aj, ak = a["idx"]
            if ai in index_set and aj in index_set and ak in index_set:
                triple = (ai, aj, ak)
                if triple not in seen_torsions:
                    seen_torsions.add(triple)
                    local_torsions.append(
                        {
                            "index": [ai - offset, aj - offset, ak - offset],
                            "k": round(a["k"], 6),
                            "aeq": round(a["aeq"], 4),
                        }
                    )

    # Intramolecular periodic dihedrals (local indices)
    local_dihedrals = []
    seen_dihedrals = set()
    for global_idx in component:
        for d in dihedral_by_atom[global_idx]:
            di, dj, dk, dl = d["idx"]
            if all(x in index_set for x in (di, dj, dk, dl)):
                # Full tuple as key since AMBER has multiple terms per quartet
                dih_key = (di, dj, dk, dl, d["n"], round(d["phi"], 4))
                if dih_key not in seen_dihedrals:
                    seen_dihedrals.add(dih_key)
                    local_dihedrals.append(
                        {
                            "index": [
                                di - offset,
                                dj - offset,
                                dk - offset,
                                dl - offset,
                            ],
                            "k": round(d["k"], 6),
                            "n": d["n"],
                            "phi": round(d["phi"], 4),
                        }
                    )

    # Annotate dihedrals with 1-4 scaling factors from NonbondedForce exceptions.
    # Only the first dihedral per unique 1-4 atom pair carries the scaling
    # (AMBER has multiple Fourier terms per quartet).
    annotated_14_pairs = set()
    for dih in local_dihedrals:
        gi = dih["index"][0] + offset
        gl = dih["index"][3] + offset
        pair = frozenset((gi, gl))
        if pair not in annotated_14_pairs and pair in all_14_exceptions:
            annotated_14_pairs.add(pair)
            exc = all_14_exceptions[pair]
            dih["electrostatic_scaling"] = exc["electrostatic_scaling"]
            dih["lj_scaling"] = exc["lj_scaling"]

    return MoleculeInstance(
        component,
        atom_kinds,
        atom_names,
        residues,
        local_bonds,
        local_torsions,
        local_dihedrals,
        mol_positions,
    )


def extract_molecules(
    topology,
    per_atom_kind_name,
    all_bonds,
    all_angles,
    all_dihedrals,
    positions_ang,
    all_14_exceptions=None,
):
    """Extract molecule instances using connected components of the bond graph.

    Uses OpenMM topology bonds (connectivity) to identify molecules via BFS.
    Each connected component becomes a MoleculeInstance with local indices,
    bonds, torsions, dihedrals, residues, and coordinates.
    """
    if all_14_exceptions is None:
        all_14_exceptions = {}

    components = _find_connected_components(topology)

    # Index force-field terms by atom for fast per-molecule lookup
    bond_by_atom = defaultdict(list)
    for b in all_bonds:
        bond_by_atom[b["i"]].append(b)
        bond_by_atom[b["j"]].append(b)

    angle_by_atom = defaultdict(list)
    for a in all_angles:
        for atom_idx in a["idx"]:
            angle_by_atom[atom_idx].append(a)

    dihedral_by_atom = defaultdict(list)
    for d in all_dihedrals:
        for atom_idx in d["idx"]:
            dihedral_by_atom[atom_idx].append(d)

    atom_by_index = {atom.index: atom for atom in topology.atoms()}

    return [
        _build_molecule_instance(
            component,
            per_atom_kind_name,
            atom_by_index,
            positions_ang,
            bond_by_atom,
            angle_by_atom,
            dihedral_by_atom,
            all_14_exceptions,
        )
        for component in components
    ]


def deduplicate_molecules(instances):
    """Group identical molecules into MoleculeGroup(template, instances, name).

    Returns list of MoleculeGroup tuples, preserving order of first appearance.
    All instances are kept so their coordinates can be written to the structure file.
    """
    groups = {}

    for inst in instances:
        fp = inst.fingerprint()
        if fp in groups:
            groups[fp]["instances"].append(inst)
        else:
            # Name after residue for single-residue molecules, otherwise generic
            if len(inst.residues) == 1:
                name = inst.residues[0]["name"]
            else:
                name = f"mol_{len(groups)}"
            groups[fp] = {"template": inst, "instances": [inst], "name": name}

    # Resolve name collisions by appending a suffix
    seen_names = {}
    for group in groups.values():
        name = group["name"]
        if name in seen_names:
            seen_names[name] += 1
            group["name"] = f"{name}_{seen_names[name]}"
        else:
            seen_names[name] = 1

    return [
        MoleculeGroup(g["template"], g["instances"], g["name"]) for g in groups.values()
    ]


def build_molecule_yaml(template, name):
    """Convert a MoleculeInstance template to a YAML-compatible dict."""
    mol = {"name": name, "atoms": template.atom_kinds}

    if template.atom_names != template.atom_kinds:
        mol["atom_names"] = template.atom_names

    if len(template.atom_kinds) > 1 and template.bonds:
        mol["excluded_neighbours"] = 3

    if template.residues:
        mol["residues"] = [
            {"name": r["name"], "number": r["number"], "range": r["range"]}
            for r in template.residues
        ]

    if template.bonds:
        mol["bonds"] = [
            {"index": b["index"], "kind": {"Harmonic": {"k": b["k"], "req": b["req"]}}}
            for b in template.bonds
        ]

    if template.torsions:
        mol["torsions"] = [
            {"index": t["index"], "kind": {"Harmonic": {"k": t["k"], "aeq": t["aeq"]}}}
            for t in template.torsions
        ]

    if template.dihedrals:
        dih_list = []
        for d in template.dihedrals:
            dih_entry = {
                "index": d["index"],
                "kind": {"ProperPeriodic": {"k": d["k"], "n": d["n"], "phi": d["phi"]}},
            }
            if "electrostatic_scaling" in d:
                dih_entry["electrostatic_scaling"] = d["electrostatic_scaling"]
            if "lj_scaling" in d:
                dih_entry["lj_scaling"] = d["lj_scaling"]
            dih_list.append(dih_entry)
        mol["dihedrals"] = dih_list

    return mol


def write_xyz(path, groups, comment="Generated by openmm2faunus"):
    """Write all atom positions to an XYZ structure file.

    Atoms are written in block order: for each molecule kind, all instances
    are written sequentially. This matches how faunus reads the structure
    file — positions are consumed in the order blocks appear.
    """
    # Collect all atoms in block order
    all_names = []
    all_positions = []
    for group in groups:
        for inst in group.instances:
            all_names.extend(inst.atom_kinds)
            all_positions.extend(inst.positions)

    with open(path, "w") as f:
        f.write(f"{len(all_names)}\n")
        f.write(f"{comment}\n")
        for name, (x, y, z) in zip(all_names, all_positions):
            f.write(f"{name} {x:.6f} {y:.6f} {z:.6f}\n")

    return len(all_names)


def extract_box_vectors(topology):
    """Extract periodic box dimensions from OpenMM topology (angstrom)."""
    vecs = topology.getPeriodicBoxVectors()
    if vecs is None:
        return None
    lengths = [vecs[i][i].value_in_unit(unit.angstrom) for i in range(3)]
    return [round(x, 4) for x in lengths]


def build_topology_yaml(pdb, forcefield_xml, add_hydrogens=False, delete_water=False):
    """Build complete faunus YAML topology from PDB + force field.

    Returns the YAML dict and the deduplicated molecule groups (for XYZ export).
    """
    pdb_obj = PDBFile(pdb)
    topology = pdb_obj.topology
    positions = pdb_obj.positions
    ff = ForceField(*forcefield_xml)

    if add_hydrogens or delete_water:
        modeller = Modeller(topology, positions)
        if delete_water:
            modeller.deleteWater()
            print(
                f"  Removed crystal waters: {modeller.topology.getNumAtoms()} atoms remaining"
            )
        if add_hydrogens:
            modeller.addHydrogens(ff)
            print(f"  Added hydrogens: {modeller.topology.getNumAtoms()} atoms")
        topology = modeller.topology
        positions = modeller.positions

    system = ff.createSystem(topology)

    positions_ang = [tuple(pos.value_in_unit(unit.angstrom)) for pos in positions]

    nb_params = extract_nonbonded(system)
    harmonic_bonds = extract_bonds(system)
    harmonic_angles = extract_angles(system)
    periodic_dihedrals = extract_dihedrals(system)
    exceptions_14 = extract_14_scaling(system)

    atomkinds, per_atom_kind_name = collect_atom_kinds(topology, nb_params)
    instances = extract_molecules(
        topology,
        per_atom_kind_name,
        harmonic_bonds,
        harmonic_angles,
        periodic_dihedrals,
        positions_ang,
        exceptions_14,
    )
    groups = deduplicate_molecules(instances)

    molecules_yaml = []
    blocks_yaml = []

    for group in groups:
        molecules_yaml.append(build_molecule_yaml(group.template, group.name))
        blocks_yaml.append({"molecule": group.name, "N": len(group.instances)})

    system_yaml = {"blocks": blocks_yaml}

    # Add cell dimensions if periodic box is present
    box = extract_box_vectors(topology)
    if box:
        system_yaml["cell"] = {"Cuboid": box}

    yaml_top = {
        "atoms": list(atomkinds.values()),
        "molecules": molecules_yaml,
        "system": system_yaml,
    }

    return yaml_top, groups


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenMM-parametrized system to Faunus YAML topology + XYZ structure"
    )
    parser.add_argument("pdb", help="Input PDB file")
    parser.add_argument(
        "--ff",
        nargs="+",
        default=["amber14-all.xml", "amber14/tip3pfb.xml"],
        help="OpenMM force field XML files (default: amber14-all.xml amber14/tip3pfb.xml)",
    )
    parser.add_argument(
        "-o", "--output", default="topology.yaml", help="Output YAML file"
    )
    parser.add_argument(
        "--add-hydrogens",
        action="store_true",
        help="Add missing hydrogens using OpenMM Modeller (needed for crystal structures)",
    )
    parser.add_argument(
        "--delete-water",
        action="store_true",
        help="Remove crystal water molecules before processing",
    )
    args = parser.parse_args()

    # Derive structure filename from topology filename
    base = os.path.splitext(args.output)[0]
    structure_file = base + ".xyz"

    print(f"Loading {args.pdb} with force fields: {', '.join(args.ff)}")
    yaml_top, groups = build_topology_yaml(
        args.pdb, args.ff, args.add_hydrogens, args.delete_water
    )

    # Write XYZ structure file
    n_atoms = write_xyz(structure_file, groups, comment=f"From {args.pdb}")
    print(f"Written {structure_file}: {n_atoms} atoms")

    # Write YAML topology
    with open(args.output, "w") as f:
        f.write("# Faunus topology generated from OpenMM\n")
        f.write(f"# Source: {args.pdb}\n")
        f.write(f"# Force fields: {', '.join(args.ff)}\n")
        f.write(f"# Structure file: {structure_file}\n\n")
        yaml.dump(yaml_top, f, default_flow_style=False, sort_keys=False)

    n_kinds = len(yaml_top["atoms"])
    n_mols = len(yaml_top["molecules"])
    total = sum(b["N"] for b in yaml_top["system"]["blocks"])
    print(f"Written {args.output}:")
    print(f"  {n_kinds} atom kinds, {n_mols} molecule types, {total} molecules total")
    print(f"\nUsage: faunus {args.output} --structure {structure_file}")


if __name__ == "__main__":
    main()

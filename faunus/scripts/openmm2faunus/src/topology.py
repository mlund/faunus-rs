"""
Copyright 2023-2024 Mikael Lund & Ladislav Bartos

Licensed under the Apache license, version 2.0 (the "license");
you may not use this file except in compliance with the license.
You may obtain a copy of the license at

    http://www.apache.org/licenses/license-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the license is distributed on an "as is" basis,
without warranties or conditions of any kind, either express or implied.
See the license for the specific language governing permissions and
limitations under the license.
"""

"""
Faunus Topology represented by Python classes.
"""

# ruff: noqa: E402
from to_yaml import yaml_tag, yaml_unit, yaml_default  # type: ignore
import yaml  # type: ignore
from martini_openmm import MartiniTopFile  # type: ignore


@yaml_default()
class FaunusDefault:
    """Represents generic default value."""

    pass


class FaunusHydrophobicity:
    """Represents hydrophobicity of an atom."""

    class Base:
        pass

    @yaml_unit("!Hydrophobic")
    class Hydrophobic(Base):
        pass

    @yaml_unit("!Hydrophilic")
    class Hydrophilic(Base):
        pass

    @yaml_tag("!SurfaceTension")
    class SurfaceTension(Base):
        def __init__(self, tension: float):
            self.tension = tension


class FaunusAtomKind:
    """Defines an atom kind and its properties."""

    def __init__(
        self,
        name: str,
        mass: float,
        charge: float = 0.0,
        element: str | None = None,
        sigma: float | None = None,
        epsilon: float | None = None,
        hydrophobicity: FaunusHydrophobicity | None = None,
        custom: dict | None = None,
    ):
        self.name = name
        self.mass = mass
        self.charge = charge
        self.element = element
        self.sigma = sigma
        self.epsilon = epsilon
        self.hydrophobicity = hydrophobicity
        self.custom = custom


class FaunusBondKind:
    """Represents different types of bonds in Faunus."""

    class Base:
        pass

    @yaml_tag("!Harmonic")
    class Harmonic(Base):
        def __init__(self, k: float, req: float):
            self.k = k
            self.req = req

    @yaml_tag("!FENE")
    class FENE(Base):
        def __init__(self, req: float, rmax: float, k: float):
            self.req = req
            self.rmax = rmax
            self.k = k


class FaunusBondOrder:
    """Represents the order of chemical bonds between atoms."""

    class Base:
        pass

    @yaml_unit("!Single")
    class Single(Base):
        pass

    @yaml_unit("!Double")
    class Double(Base):
        pass

    @yaml_unit("!Triple")
    class Triple(Base):
        pass

    @yaml_unit("!Quadruple")
    class Quadruple(Base):
        pass

    @yaml_unit("!Quintuple")
    class Quintuple(Base):
        pass

    @yaml_unit("!Sextuple")
    class Sextuple(Base):
        pass

    @yaml_unit("!Amide")
    class Amide(Base):
        pass

    @yaml_unit("!Aromatic")
    class Aromatic(Base):
        pass

    @yaml_unit("!Custom")
    class Custom(Base):
        def __init__(self, value: float):
            self.value = value


class FaunusTorsionKind:
    """Represents different types of torsions used in Faunus."""

    class Base:
        pass

    @yaml_tag("!Harmonic")
    class Harmonic(Base):
        def __init__(self, k: float, aeq: float):
            self.k = k
            self.aeq = aeq

    @yaml_tag("!Cosine")
    class Cosine(Base):
        def __init__(self, k: float, aeq: float):
            self.k = k
            self.aeq = aeq


class FaunusDihedralKind:
    """Represents different types of dihedral angles used in Faunus."""

    class Base:
        pass

    @yaml_tag("!ProperHarmonic")
    class ProperHarmonic(Base):
        def __init__(self, k: float, aeq: float):
            self.k = k
            self.aeq = aeq

    @yaml_tag("!ProperPeriodic")
    class ProperPeriodic(Base):
        def __init__(self, k: float, n: float, phi: float):
            self.k = k
            self.n = n
            self.phi = phi

    @yaml_tag("!ImproperHarmonic")
    class ImproperHarmonic(Base):
        def __init__(self, k: float, aeq: float):
            self.k = k
            self.aeq = aeq

    @yaml_tag("!ImproperPeriodic")
    class ImproperPeriodic(Base):
        def __init__(self, k: float, n: float, phi: float):
            self.k = k
            self.n = n
            self.phi = phi

    """
    @yaml_tag("!ImproperAmber")
    class ImproperAmber(Base):
        def __init__(self, k: float, n: float, phi: float):
            self.k = k
            self.n = n
            self.phi = phi

    @yaml_tag("!ImproperCharmm")
    class ImproperCharmm(Base):
        def __init__(self, k: float, n: float, phi: float):
            self.k = k
            self.n = n
            self.phi = phi
    """


class FaunusBond:
    """Represents a bond in a molecular system with specified indices and optional bond kind and order."""

    def __init__(
        self,
        index: list[int],
        kind: FaunusBondKind.Base | None = None,
        order: FaunusBondOrder.Base | None = None,
    ):
        self.index = index
        self.kind = kind
        self.order = order


class FaunusTorsion:
    """Represents a torsional interaction within a molecular system, identified by indices and optional kind."""

    def __init__(self, index: list[int], kind: FaunusTorsionKind.Base | None = None):
        self.index = index
        self.kind = kind


class FaunusDihedral:
    """Represents dihedral interactions within a molecular system, with additional properties for scaling."""

    def __init__(
        self,
        index: list[int],
        kind: FaunusDihedralKind.Base | None = None,
        electrostatic_scaling: float | None = None,
        lj_scaling: float | None = None,
    ):
        self.index = index
        self.kind = kind
        self.electrostatic_scaling = electrostatic_scaling
        self.lj_scaling = lj_scaling


class FaunusDegreesOfFreedom:
    """Represents the degrees of freedom status for a molecule in simulations."""

    class Base:
        pass

    @yaml_unit("!Free")
    class Free(Base):
        pass

    @yaml_unit("!Frozen")
    class Frozen(Base):
        pass

    @yaml_unit("!Rigid")
    class Rigid(Base):
        pass

    @yaml_unit("!RigidAlchemical")
    class RigidAlchemical(Base):
        pass


class FaunusResidue:
    """Represents a residue within a molecule, defined by a name and a range of atom indices."""

    def __init__(self, name: str, range: list[int], number: int | None = None):
        self.name = name
        self.number = number
        self.range = range


class FaunusChain:
    """Represents a chain within a molecule, defined by a name and a range of atom indices."""

    def __init__(self, name: str, range: list[int]):
        self.name = name
        self.range = range


class FaunusMoleculeKind:
    """Represents a molecule kind in the system, including details about its composition and bonded interactions."""

    def __init__(
        self,
        name: str,
        atoms: list[str],
        bonds: list[FaunusBond] | None = None,
        torsions: list[FaunusTorsion] | None = None,
        dihedrals: list[FaunusDihedral] | None = None,
        excluded_neighbours: int | None = None,
        exclusions: list[list[int]] | None = None,
        degrees_of_freedom: FaunusDegreesOfFreedom.Base | None = None,
        atom_names: list[str | None] | None = None,
        residues: list[FaunusResidue] | None = None,
        chains: list[FaunusChain] | None = None,
        custom: dict | None = None,
        has_com: bool | None = None,
    ):
        self.name = name
        self.atoms = atoms
        self.bonds = bonds
        self.torsions = torsions
        self.dihedrals = dihedrals
        self.excluded_neighbours = excluded_neighbours
        self.exclusions = exclusions
        self.degrees_of_freedom = degrees_of_freedom
        self.atom_names = atom_names
        self.residues = residues
        self.chains = chains
        self.custom = custom
        self.has_com = has_com


class FaunusNonbondedInteraction:
    """Represents nonbonded interaction between atoms in the simulated system."""

    class Base:
        pass

    @yaml_tag("!LennardJones")
    class LennardJones(Base):
        def __init__(self, sigma: float, eps: float):
            self.sigma = sigma
            self.eps = eps

    @yaml_tag("!CoulombReactionField")
    class CoulombReactionField(Base):
        def __init__(self, epsr: float, epsrf: float, cutoff: float, shift: bool):
            self.epsr = epsr
            self.epsrf = epsrf
            self.cutoff = cutoff
            self.shift = shift

    # TODO: add other supported nonbonded interactions


class FaunusIntermolecularBonded:
    """Represents intermolecular bonded interactions for the simulation."""

    def __init__(
        self,
        bonds: list[FaunusBond] | None = None,
        torsions: list[FaunusTorsion] | None = None,
        dihedrals: list[FaunusDihedral] | None = None,
    ):
        self.bonds = bonds
        self.torsions = torsions
        self.dihedrals = dihedrals


class FaunusMoleculeBlock:
    """Represents a block of molecules for simulation purposes, specifying the molecule type and count."""

    def __init__(self, molecule: str, number: int):
        self.molecule = molecule
        self.N = number


class FaunusEnergy:
    """Manages hamiltonian of the system."""

    def __init__(
        self,
        nonbonded: dict[
            tuple[str, str] | FaunusDefault, list[FaunusNonbondedInteraction.Base]
        ],
    ):
        self.nonbonded = nonbonded


class FaunusSystem:
    """Manages intermolecular bonded interactions, hamiltonian and the molecule blocks in the system."""

    def __init__(
        self,
        intermolecular: FaunusIntermolecularBonded | None = None,
        energy: FaunusEnergy | None = None,
        blocks: list[FaunusMoleculeBlock] | None = None,
    ):
        self.intermolecular = intermolecular
        self.energy = energy
        self.blocks = blocks


class FaunusTopology:
    """Manages the overall topology of a simulation, including molecules, atoms, and intermolecular interactions."""

    def __init__(
        self,
        atom_kinds: list[FaunusAtomKind] | None = None,
        molecule_kinds: list[FaunusMoleculeKind] | None = None,
        intermolecular: FaunusIntermolecularBonded | None = None,
        energy: FaunusEnergy | None = None,
        molecule_blocks: list[FaunusMoleculeBlock] | None = None,
    ):
        self.atoms = atom_kinds
        self.molecules = molecule_kinds
        self.system = FaunusSystem(intermolecular, energy, molecule_blocks)

    @classmethod
    def _martini_get_atoms(
        cls,
        moltype: MartiniTopFile._MoleculeType,
        atom_types: dict[str, list[str]],
        martini_faunus_names: dict[str, list[str]],
    ) -> tuple[list[FaunusAtomKind], list[str], list[str | None]]:
        """
        Get atom kinds from a single Martini molecule type.
        Returns: list of atom kinds, list of atoms of the molecule and list of their names.
        """
        # Gromacs supports redefining masses and charges of atom types for specific molecules,
        # we therefore rename atom types to contain name, charge and mass information
        # if there are multiple Gromacs atoms with the same atom kind but redefined charge
        # and/or mass, we create separate atom kinds for these atoms

        # list of faunus atom kinds
        faunus_atoms: list[FaunusAtomKind] = []
        # atoms of the molecule type
        moltype_atoms = []
        # names of the atoms of the molecule
        atom_names = []

        for atom in moltype.atoms:
            if len(atom) >= 8:
                mass = float(atom[7])
                charge = float(atom[6])
            elif len(atom) >= 7:
                mass = float(atom_types[atom[1]][3])
                charge = float(atom[6])
            else:
                mass = float(atom_types[atom[1]][3])
                charge = float(atom_types[atom[1]][4])

            faunus_atom_name = f"{atom[1]}_{charge}_{mass}"
            moltype_atoms.append(faunus_atom_name)
            atom_names.append(atom[4])

            try:
                martini_faunus_names[atom[1]].append(faunus_atom_name)
            except KeyError:
                martini_faunus_names[atom[1]] = [faunus_atom_name]

            exists = False
            for atom2 in faunus_atoms:
                if atom2.name == faunus_atom_name:
                    exists = True
                    break

            if not exists:
                faunus_atoms.append(FaunusAtomKind(faunus_atom_name, mass, charge))

        return (faunus_atoms, moltype_atoms, atom_names)

    @classmethod
    def _martini_get_bonds(
        cls, moltype: MartiniTopFile._MoleculeType
    ) -> list[FaunusBond]:
        """Get bonds from a Martini molecule type."""

        bonds = []
        for bond in moltype.bonds:
            bonds.append(
                FaunusBond(
                    [int(x) - 1 for x in bond[:2]],
                    FaunusBondKind.Harmonic(float(bond[4]), float(bond[3]) * 10.0),
                )
            )

        return bonds

    @classmethod
    def _martini_get_torsions(
        cls, moltype: MartiniTopFile._MoleculeType
    ) -> list[FaunusTorsion]:
        """Get torsions from a Martini molecule type."""

        torsions = []
        for torsion in moltype.g96_angles:
            torsions.append(
                FaunusTorsion(
                    [int(x) - 1 for x in torsion[:3]],
                    FaunusTorsionKind.Cosine(float(torsion[5]), float(torsion[4])),
                )
            )

        for torsion in moltype.harmonic_angles:
            torsions.append(
                FaunusTorsion(
                    [int(x) - 1 for x in torsion[:3]],
                    FaunusTorsionKind.Harmonic(float(torsion[5]), float(torsion[4])),
                )
            )

        return torsions

    @classmethod
    def _martini_get_dihedrals(
        cls, moltype: MartiniTopFile._MoleculeType
    ) -> list[FaunusDihedral]:
        """Get dihedrals from a Martini molecule type."""

        dihedrals = []
        for dihedral in moltype.dihedrals:
            atoms = [int(x) - 1 for x in dihedral[:4]]
            dihedral_type = int(dihedral[4])

            if dihedral_type == 1:
                dihedrals.append(
                    FaunusDihedral(
                        atoms,
                        FaunusDihedralKind.ProperPeriodic(
                            float(dihedral[6]), float(dihedral[7]), float(dihedral[5])
                        ),
                    )
                )
            elif dihedral_type == 2:
                dihedrals.append(
                    FaunusDihedral(
                        atoms,
                        FaunusDihedralKind.ImproperHarmonic(
                            float(dihedral[6]), float(dihedral[5])
                        ),
                    )
                )
            elif dihedral_type == 4:
                dihedrals.append(
                    FaunusDihedral(
                        atoms,
                        FaunusDihedralKind.ImproperPeriodic(
                            float(dihedral[6]), float(dihedral[7]), float(dihedral[5])
                        ),
                    )
                )
            else:
                raise Exception(f"Dihedral type '{dihedral_type}' is not supported.")

        return dihedrals

    @classmethod
    def _martini_get_residues(
        cls, moltype: MartiniTopFile._MoleculeType
    ) -> list[FaunusResidue]:
        """Get residues from a Martini molecule type."""

        class CurrResidue:
            """Helper class for obtaining residues from the Martini topology."""

            def __init__(self, range: list[int], name: str, number: int):
                self.range = range
                self.name = name
                self.number = number

            def to_faunus(self) -> FaunusResidue:
                return FaunusResidue(self.name, self.range.copy(), self.number)

        last_resnum = None
        residues = []
        curr_residue = CurrResidue([0, 0], "", 1)
        for i, atom in enumerate(moltype.atoms):
            resnum = int(atom[2])
            if resnum != last_resnum:
                if last_resnum is not None:
                    curr_residue.range[1] = i
                    residues.append(curr_residue.to_faunus())

                last_resnum = resnum
                curr_residue.range[0] = i
                curr_residue.name = atom[3]
                curr_residue.number = resnum

        # add the last residue
        curr_residue.range[1] = len(moltype.atoms)
        curr_residue.name = moltype.atoms[-1][3]
        curr_residue.number = int(moltype.atoms[-1][2])
        residues.append(curr_residue.to_faunus())

        return residues

    @classmethod
    def _martini_get_exclusions(
        cls, moltype: MartiniTopFile._MoleculeType
    ) -> list[list[int]]:
        """Get exclusions from a Martini molecule type."""

        return [[int(excl[0]) - 1, int(excl[1]) - 1] for excl in moltype.exclusions]

    @classmethod
    def _martini_get_constraints_as_bonds(
        cls, moltype: MartiniTopFile._MoleculeType
    ) -> list[FaunusBond]:
        """
        Get constraints from a Martini molecule type and
        convert them to harmonic bonds with a high force constant.
        """

        bonds = []
        for constraint in moltype.constraints:
            bonds.append(
                FaunusBond(
                    [int(x) - 1 for x in constraint[:2]],
                    FaunusBondKind.Harmonic(50_000, float(constraint[3]) * 10.0),
                )
            )

        return bonds

    @classmethod
    def _martini_get_nonbonded(
        cls,
        martini_faunus_names: dict[str, list[str]],
        nonbonded: dict[tuple[str, str], list[str]],
    ) -> dict[tuple[str, str] | FaunusDefault, list[FaunusNonbondedInteraction.Base]]:
        """
        Get nonbonded interactions from the Martini topology.
        """

        faunus_nonbonded: dict[
            tuple[str, str] | FaunusDefault, list[FaunusNonbondedInteraction.Base]
        ] = {}

        for name1 in martini_faunus_names:
            for name2 in martini_faunus_names:
                try:
                    try:
                        interaction = nonbonded[(name1, name2)]
                    except KeyError:
                        interaction = nonbonded[(name2, name1)]
                except KeyError:
                    raise Exception(
                        f"Could not find LJ interaction between atoms {name1} and {name2}."
                    )

                for faunus_name1 in martini_faunus_names[name1]:
                    for faunus_name2 in martini_faunus_names[name2]:
                        if (faunus_name1, faunus_name2) in faunus_nonbonded.keys() or (
                            faunus_name2,
                            faunus_name1,
                        ) in faunus_nonbonded.keys():
                            continue

                        faunus_nonbonded[(faunus_name1, faunus_name2)] = [
                            FaunusNonbondedInteraction.LennardJones(
                                float(interaction[3]) * 10.0, float(interaction[4])
                            )
                        ]

        # add default electrostatic interactions
        faunus_nonbonded[FaunusDefault()] = [
            FaunusNonbondedInteraction.CoulombReactionField(
                15.0, float("inf"), 11.0, True
            )
        ]

        return faunus_nonbonded

    @classmethod
    def from_martini(cls, martini_top: MartiniTopFile):
        """
        Convert Martini topology parsed by martini_openmm into Faunus topology.

        Notes:
        - Constraints are converted to harmonic bonds with a force constant of 50000.
        - Pairs and restricted angles are not supported and will raise an exception.
        - Cmaps, vsites, and intermolecular bonded interactions are ignored.
        - Electrostatic interactions are ignored.
        """

        atoms = []
        molecules = []
        blocks = []
        molnames = []

        # dictionary mapping martini atom type names to all faunus atom kind names
        martini_faunus_names: dict[str, list[str]] = {}

        for molname, number in martini_top._molecules:
            blocks.append(FaunusMoleculeBlock(molname, number))

            # if there are multiple blocks with the same molecule,
            # only create the molecule once
            if molname in molnames:
                continue

            molnames.append(molname)
            moltype = martini_top._moleculeTypes[molname]

            # TODO: pairs are currently not supported by Faunus
            if len(moltype.pairs) != 0:
                raise NotImplementedError("Pairs are currently not supported.")

            # TODO: implement restricted angles
            if len(moltype.restricted_angles) != 0:
                raise NotImplementedError(
                    "Restricted angles are currently not supported."
                )

            # get atom kinds for Faunus
            faunus_atoms, moltype_atoms, atom_names = FaunusTopology._martini_get_atoms(
                moltype, martini_top._atom_types, martini_faunus_names
            )

            atoms.extend(faunus_atoms)

            # get bonds, torsions, dihedrals, constraints
            bonds = FaunusTopology._martini_get_bonds(moltype)
            bonds.extend(FaunusTopology._martini_get_constraints_as_bonds(moltype))
            torsions = FaunusTopology._martini_get_torsions(moltype)
            dihedrals = FaunusTopology._martini_get_dihedrals(moltype)

            # get exclusions
            exclusions = FaunusTopology._martini_get_exclusions(moltype)

            # get residues
            residues = FaunusTopology._martini_get_residues(moltype)

            molecules.append(
                FaunusMoleculeKind(
                    moltype.molecule_name,
                    moltype_atoms,
                    bonds if len(bonds) != 0 else None,
                    torsions if len(torsions) != 0 else None,
                    dihedrals if len(dihedrals) != 0 else None,
                    # excluded_neighbours should always be 1 for Martini
                    1,
                    exclusions if len(exclusions) != 0 else None,
                    FaunusDegreesOfFreedom.Free(),
                    atom_names,
                    residues,
                )
            )

        # get nonbonded interactions (LJ only)
        nonbonded = FaunusTopology._martini_get_nonbonded(
            martini_faunus_names, martini_top._nonbond_types
        )

        return FaunusTopology(
            atoms,
            molecules,
            None,
            FaunusEnergy(nonbonded) if len(nonbonded) != 0 else None,
            blocks if len(blocks) != 0 else None,
        )

    def to_yaml(self) -> str:
        """Serialize the Topology as a yaml structure readable by Faunus."""
        return yaml.dump(self, sort_keys=False).replace("''", "")

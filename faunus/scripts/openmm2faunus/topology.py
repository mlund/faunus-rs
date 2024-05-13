"""
Copyright 2023-2024 Mikael Lund

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

from to_yaml import yaml_tag, yaml_unit
import yaml
from martini_openmm import *

class FaunusHydrophobicity:
    """Represents hydrophobicity of an atom."""

    @yaml_unit("!Hydrophobic")
    class Hydrophobic:
        pass

    @yaml_unit("!Hydrophilic")
    class Hydrophilic:
        pass

    @yaml_tag("!SurfaceTension")
    class SurfaceTension:
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

    @yaml_tag("!Harmonic")
    class Harmonic:
        def __init__(self, k: float, req: float):
            self.k = k
            self.req = req

    @yaml_tag("!FENE")
    class FENE:
        def __init__(self, req: float, rmax: float, k: float):
            self.req = req
            self.rmax = rmax
            self.k = k

class FaunusBondOrder:
    """Represents the order of chemical bonds between atoms."""

    @yaml_unit("!Single")
    class Single:
        pass

    @yaml_unit("!Double")
    class Double:
        pass

    @yaml_unit("!Triple")
    class Triple:
        pass

    @yaml_unit("!Quadruple")
    class Quadruple:
        pass

    @yaml_unit("!Quintuple")
    class Quintuple:
        pass

    @yaml_unit("!Sextuple")
    class Sextuple:
        pass

    @yaml_unit("!Amide")
    class Amide:
        pass

    @yaml_unit("!Aromatic")
    class Aromatic:
        pass

    @yaml_unit("!Custom")
    class Custom:
        def __init__(self, value: float):
            self.value = value

class FaunusTorsionKind:
    """Represents different types of torsions used in Faunus."""

    @yaml_tag("!Harmonic")
    class Harmonic:
        def __init__(self, k: float, aeq: float):
            self.k = k
            self.aeq = aeq

    @yaml_tag("!Cosine")
    class Cosine:
        def __init__(self, k: float, aeq: float):
            self.k = k
            self.aeq = aeq

class FaunusDihedralKind:
    """Represents different types of dihedral angles used in Faunus."""

    @yaml_tag("!Harmonic")
    class Harmonic:
        def __init__(self, k: float, aeq: float):
            self.k = k
            self.aeq = aeq

    @yaml_tag("!ProperPeriodic")
    class ProperPeriodic:
        def __init__(self, k: float, n: float, phi: float):
            self.k = k
            self.n = n
            self.phi = phi

    @yaml_tag("!ImproperPeriodic")
    class ImproperHarmonic:
        def __init__(self, k: float, aeq: float):
            self.k = k
            self.aeq = aeq

    @yaml_tag("!ImproperAmber")
    class ImproperAmber:
        def __init__(self, k: float, n: float, phi: float):
            self.k = k
            self.n = n
            self.phi = phi

    @yaml_tag("!ImproperCharmm")
    class ImproperCharmm:
        def __init__(self, k: float, n: float, phi: float):
            self.k = k
            self.n = n
            self.phi = phi


class FaunusBond:
    """Represents a bond in a molecular system with specified indices and optional bond kind and order."""
    def __init__(self, 
                index: list[int], 
                kind: FaunusBondKind | None = None, 
                order: FaunusBondOrder | None = None):
        self.index = index
        self.kind = kind
        self.order = order

class FaunusTorsion:
    """Represents a torsional interaction within a molecular system, identified by indices and optional kind."""
    def __init__(self, 
                index: list[int],
                kind: FaunusTorsionKind | None = None):
        self.index = index
        self.kind = kind

class FaunusDihedral:
    """Represents dihedral interactions within a molecular system, with additional properties for scaling."""
    def __init__(self,
                index: list[int],
                kind: FaunusDihedralKind | None = None,
                electrostatic_scaling: float | None = None,
                lj_scaling: float | None = None):
        self.index = index
        self.kind = kind
        self.electrostatic_scaling = electrostatic_scaling
        self.lj_scaling = lj_scaling

class FaunusDegreesOfFreedom:
    """Represents the degrees of freedom status for a molecule in simulations."""
    @yaml_unit("!Free")
    class Free:
        pass

    @yaml_unit("!Frozen")
    class Frozen:
        pass

    @yaml_unit("!Rigid")
    class Rigid:
        pass

    @yaml_unit("!RigidAlchemical")
    class RigidAlchemical:
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
    def __init__(self,
            name: str,
            atoms: list[str], 
            bonds: list[FaunusBond] | None = None,
            torsions: list[FaunusTorsion] | None = None,
            dihedrals: list[FaunusDihedral] | None = None,
            excluded_neighbours: int | None = None,
            exclusions: list[list[int]] | None = None,
            degrees_of_freedom: FaunusDegreesOfFreedom | None = None,
            atom_names: list[str | None] | None = None,
            residues: list[FaunusResidue] | None = None,
            chains: list[FaunusChain] | None = None,
            custom: dict | None = None,
            has_com: bool | None = None):
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

class FaunusIntermolecularBonded:
    """Represents intermolecular bonded interactions for the simulation."""
    def __init__(self, 
                bonds: list[FaunusBond] | None = None,
                torsions: list[FaunusTorsion] | None = None,
                dihedrals: list[FaunusDihedral] | None = None):
        self.bonds = bonds
        self.torsions = torsions
        self.dihedrals = dihedrals

class FaunusMoleculeBlock:
    """Represents a block of molecules for simulation purposes, specifying the molecule type and count."""
    def __init__(self, molecule: str, number: int):
        self.molecule = molecule
        self.N = number

class FaunusSystem:
    """Manages intermolecular bonded interactions and the molecule blocks in the system."""
    def __init__(self, intermolecular: FaunusIntermolecularBonded | None = None, blocks: list[FaunusMoleculeBlock] | None = None):
        self.intermolecular = intermolecular
        self.blocks = blocks

class FaunusTopology:
    """Manages the overall topology of a simulation, including molecules, atoms, and intermolecular interactions."""
    def __init__(self, 
                atom_kinds: list[FaunusAtomKind] | None = None,
                molecule_kinds: list[FaunusMoleculeKind] | None = None,
                intermolecular: FaunusIntermolecularBonded | None = None, 
                molecule_blocks: list[FaunusMoleculeBlock] | None = None):
        self.atoms = atom_kinds
        self.molecules = molecule_kinds
        self.system = FaunusSystem(intermolecular, molecule_blocks)

    def _martini_get_atoms(self,
                    moltype: MartiniTopFile._MoleculeType, 
                    atom_types: dict[list[str]]
        ) -> tuple[list[str], list[str]]:
        """
        Get atom kinds from a single Martini molecule type and add them to the topology.
        Return list of atoms of the molecule and list of their names.
        """
        # Gromacs supports redefining masses and charges of atom types for specific molecules,
        # we therefore rename atom types to contain name, charge and mass information
        # if there are multiple Gromacs atoms with the same atom kind but redefined charge
        # and/or mass, we create separate atom kinds for these atoms

        moltype_atoms = []
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
                charge = float(atom_types[atom[1][4]])
            
            faunus_atom_name = f"{atom[1]}_{charge}_{mass}"
            moltype_atoms.append(faunus_atom_name)
            atom_names.append(atom[4])

            exists = False
            for atom2 in self.atoms:
                if atom2.name == faunus_atom_name:
                    exists = True
                    break
                    
            if not exists:
                self.atoms.append(FaunusAtomKind(faunus_atom_name, mass, charge))
        
        return (moltype_atoms, atom_names)

    def _martini_get_bonds(self, moltype: MartiniTopFile._MoleculeType) -> list[FaunusBond]:
        """Get bonds from a Martini molecule type."""

        bonds = []
        for bond in moltype.bonds:
            bonds.append(FaunusBond(
                [int(x) - 1 for x in bond[:2]],
                FaunusBondKind.Harmonic(float(bond[4]), float(bond[3]))))
        
        return bonds

    def _martini_get_torsions(self, moltype: MartiniTopFile._MoleculeType) -> list[FaunusTorsion]:
        """Get torsions from a Martini molecule type."""

        torsions = []
        for torsion in moltype.g96_angles:
            torsions.append(
                FaunusTorsion(
                    [int(x) - 1 for x in torsion[:3]],
                    FaunusTorsionKind.Cosine(float(torsion[5]), float(torsion[4]))))
        
        for torsion in moltype.harmonic_angles:
            torsions.append(
                FaunusTorsion(
                    [int(x) - 1 for x in torsion[:3]],
                    FaunusTorsionKind.Harmonic(float(torsion[5]), float(torsion[4]))))
            
        return torsions

    def _martini_get_dihedrals(self, moltype: MartiniTopFile._MoleculeType) -> list[FaunusDihedral]:
        """Get dihedrals from a Martini molecule type."""

        dihedrals = []
        for dihedral in moltype.dihedrals:
            dihedrals.append(
                FaunusDihedral(
                    [int(x) - 1 for x in dihedral[:4]],
                    FaunusDihedralKind.ProperPeriodic(float(dihedral[6]), float(dihedral[7]), float(dihedral[5]))))
        
        return dihedrals

    def _martini_get_residues(self, moltype: MartiniTopFile._MoleculeType) -> list[FaunusResidue]:
        """Get residues from a Martini molecule type."""

        last_resnum = None
        residues = []
        curr_residue = [0, 0, None, 1]
        for (i, atom) in enumerate(moltype.atoms):
            resnum = int(atom[2])
            if resnum != last_resnum:
                if last_resnum is not None:
                    curr_residue[1] = i
                    residues.append(
                        FaunusResidue(
                            curr_residue[2], 
                            [curr_residue[0], curr_residue[1]], 
                            curr_residue[3]))
                
                last_resnum = resnum
                curr_residue[0] = i
                curr_residue[2] = atom[3]
                curr_residue[3] = resnum

        # add last residue
        curr_residue[1] = len(moltype.atoms)
        curr_residue[2] = moltype.atoms[-1][3]
        curr_residue[3] = int(moltype.atoms[-1][2])
        residues.append(FaunusResidue(curr_residue[2], [curr_residue[0], curr_residue[1]], curr_residue[3]))

        return residues

    def _martini_get_exclusions(self, moltype: MartiniTopFile._MoleculeType) -> list[list[int]]:
        """Get exclusions from a Martini molecule type."""

        return [[int(excl[0]) - 1, int(excl[1]) - 1] for excl in moltype.exclusions]
    
    def _martini_get_constraints_as_bonds(self, moltype: MartiniTopFile._MoleculeType) -> list[FaunusBond]:
        """
        Get constraints from a Martini molecule type and 
        convert them to harmonic bonds with a high force constant.
        """

        bonds = []
        for constraint in moltype.constraints:
            bonds.append(FaunusBond(
                [int(x) - 1 for x in constraint[:2]],
                FaunusBondKind.Harmonic(50_000, float(constraint[3]))))

        return bonds

    def __init__(self, martini_top: MartiniTopFile):
        """
        Convert Martini topology parsed by martini_openmm into Faunus topology.
        
        Notes:
        - Constraints are converted to harmonic bonds with a force constant of 50000.
        - Pairs and restricted angles are not supported and will raise an exception.
        - Nonbonded interactions, cmaps, vsites, and intermolecular bonded interactions are ignored.
        """

        self.atoms = []
        self.molecules = []
        blocks = []
        molnames = []

        for (molname, number) in martini_top._molecules:
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
                raise NotImplementedError("Restricted angles are currently not supported.")

            # get atom kinds for Faunus
            moltype_atoms, atom_names = self._martini_get_atoms(moltype, martini_top._atom_types)

            # get bonds, torsions, dihedrals, constraints
            bonds = self._martini_get_bonds(moltype)
            bonds.extend(self._martini_get_constraints_as_bonds(moltype))
            torsions = self._martini_get_torsions(moltype)
            dihedrals = self._martini_get_dihedrals(moltype)

            # get exclusions
            exclusions = self._martini_get_exclusions(moltype)
            
            # get residues 
            residues = self._martini_get_residues(moltype)
            
            self.molecules.append(
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
                    residues)) 

        if len(blocks) != 0:
            self.system = FaunusSystem(None, blocks)


    def to_yaml(self) -> str:
        """Serialize the Topology as a yaml structure readable by Faunus."""
        return yaml.dump(self, sort_keys = False).replace("''", "")
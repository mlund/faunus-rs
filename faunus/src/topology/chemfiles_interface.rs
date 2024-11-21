// Copyright 2023 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

//! # Interface to the [`chemfiles`] crate

use std::path::Path;

use crate::{
    cell::{Cell, Cuboid, Endless, Shape, SimulationCell, Sphere},
    group::{Group, GroupCollection},
    platform::reference::ReferencePlatform,
    topology::Residue,
    Point, PointParticle, WithCell, WithTopology,
};
use chemfiles::Frame;
use nalgebra::Vector3;

use super::{molecule::MoleculeKind, AtomKind, IndexRange};

/// Create a new chemfiles::Frame from an input file in a supported format.
pub(super) fn frame_from_file(filename: &impl AsRef<Path>) -> anyhow::Result<chemfiles::Frame> {
    let mut trajectory = chemfiles::Trajectory::open(filename, 'r')?;
    let mut frame = chemfiles::Frame::new();
    trajectory.read(&mut frame)?;
    Ok(frame)
}

/// Get positions of particles from chemfiles::Frame.
/// If `cell` is provided, all the positions of particles from the frame are shifted by -half_cell.
pub(super) fn positions_from_frame(
    frame: &chemfiles::Frame,
    cell: Option<&impl SimulationCell>,
) -> Vec<Point> {
    let shift = if let Some(cell) = cell {
        cell.bounding_box().map(|b| -0.5 * b).unwrap_or_default()
    } else {
        Vector3::default()
    };

    frame
        .positions()
        .iter()
        .map(|pos| <[f64; 3] as Into<Point>>::into(*pos) + shift)
        .collect()
}

/// A trait for structure that can be converted to chemfiles Frame.
pub trait ChemFrameConvert: WithCell + WithTopology + GroupCollection {
    /// Convert system to chemfiles::Frame structure.
    ///
    /// ## Notes
    /// - Positions, residues, atom types and bonds are converted.
    /// - Custom properties of atoms and residues are not converted.
    /// - Angles and dihedrals are not converted.
    /// - Residues are renumbered starting from 1.
    fn to_frame(&self) -> Frame {
        let mut frame = Frame::new();
        self.add_atoms_to_frame(&mut frame);
        self.add_residues_to_frame(&mut frame);
        self.add_bonds_to_frame(&mut frame);
        frame.set_cell(&self.cell().to_chem_cell());

        frame
    }

    /// Convert all faunus particles to chemfiles particles and add them to the chemfiles Frame.
    /// This converts all atoms, both active and inactive.
    /// This also shifts the particles so they fit into chemfiles cell.
    fn add_atoms_to_frame(&self, frame: &mut Frame) {
        let topology = self.topology_ref();
        let mut particles = self.get_all_particles();

        // shift the particles
        // we need to shift them because faunus treats [0,0,0] as the center of the cell,
        // while chemfiles treats [half_cell, half_cell, half_cell] as the center of the cell
        // no shifting is needed if the box is infinite
        if let Some(shift) = self.cell().bounding_box().map(|b| -0.5 * b) {
            particles
                .iter_mut()
                .for_each(|particle| particle.pos += shift);
        }

        // add atoms to the frame
        let to_atomkind = |i: usize| &topology.atomkinds()[i];
        self.groups().iter().for_each(|group| {
            let molecule = &topology.moleculekinds()[group.molecule()];
            let atoms = molecule.atom_indices().iter().cloned().map(to_atomkind);
            for (i, atom) in atoms.enumerate() {
                frame.add_atom(
                    &atom.to_chem_atom(molecule.atom_names()[i].as_deref()),
                    (*particles[i + group.start()].pos()).into(),
                    None,
                );
            }
        });
    }

    /// Convert faunus residues to chemfiles residues and add them to the chemfiles Frame.
    fn add_residues_to_frame(&self, frame: &mut Frame) {
        let topology = self.topology_ref();

        self.groups()
            .iter()
            .fold((1, 0), |(residue_index, atom_index), group| {
                let molecule = &topology.moleculekinds()[group.molecule()];
                group
                    .to_chem_residues(atom_index, residue_index, molecule)
                    .iter()
                    .for_each(|residue| {
                        frame
                            .add_residue(residue)
                            .expect("Faunus residue could not be converted to chemfiles topology.");
                    });

                (
                    residue_index + molecule.residues().len() as i64,
                    atom_index + group.capacity(),
                )
            });
    }

    // Convert faunus bonds to chemfiles bonds and add them to the chemfiles Frame.
    fn add_bonds_to_frame(&self, frame: &mut Frame) {
        let topology = self.topology_ref();

        self.groups().iter().fold(0, |atom_index, group| {
            let molecule = &topology.moleculekinds()[group.molecule()];

            molecule.bonds().iter().for_each(|bond| {
                frame.add_bond(bond.index()[0] + atom_index, bond.index()[1] + atom_index)
            });

            atom_index + group.capacity()
        });

        topology
            .intermolecular()
            .bonds()
            .iter()
            .for_each(|bond| frame.add_bond(bond.index()[0], bond.index()[1]));
    }
}

impl Group {
    /// Convert Group to N chemfiles Residues.
    ///
    /// ## Parameters
    /// `abs_atom_index` - absolute index of the first atom of the group
    /// `first_resid` - absolute index of the first residue of the group
    /// `molecule` - MoleculeKind forming this group
    fn to_chem_residues(
        &self,
        abs_atom_index: usize,
        first_resid: i64,
        molecule: &MoleculeKind,
    ) -> Vec<chemfiles::Residue> {
        molecule
            .residues()
            .iter()
            .enumerate()
            .map(|(index, resid)| resid.to_chem_residue(abs_atom_index, first_resid + index as i64))
            .collect()
    }
}

impl Residue {
    /// Convert Residue to chemfiles Residue.
    ///
    /// ## Parameters
    /// `abs_atom_index` - absolute index of the first atom of the group this residue is part of
    /// `resid` - absolute index of the residue
    fn to_chem_residue(&self, abs_atom_index: usize, resid: i64) -> chemfiles::Residue {
        let mut chemfiles_residue = chemfiles::Residue::with_id(self.name(), resid);

        self.range()
            .for_each(|atom| chemfiles_residue.add_atom(atom + abs_atom_index));

        chemfiles_residue
    }
}

impl AtomKind {
    /// Convert topology atom to chemfiles atom.
    /// Does not convert custom properties.
    /// `name` is the name of the particle itself, not of the AtomKind.
    /// If `name` is not provided, the name of the AtomKind itself is used.
    fn to_chem_atom(&self, name: Option<&str>) -> chemfiles::Atom {
        let mut chemfiles_atom = match name {
            Some(name) => chemfiles::Atom::new(name),
            None => chemfiles::Atom::new(self.name()),
        };
        chemfiles_atom.set_mass(self.mass());
        chemfiles_atom.set_charge(self.charge());
        if let Some(element) = self.element() {
            chemfiles_atom.set_atomic_type(element);
        }

        chemfiles_atom
    }
}

impl ChemFrameConvert for ReferencePlatform {}

/// Convert topology Residue to chemfiles residue.
impl core::convert::From<&Residue> for chemfiles::Residue {
    fn from(residue: &Residue) -> Self {
        let mut chemfiles_residue = match residue.number() {
            None => Self::new(residue.name()),
            Some(n) => Self::with_id(residue.name(), n as i64),
        };

        residue
            .range()
            .for_each(|atom| chemfiles_residue.add_atom(atom));

        chemfiles_residue
    }
}

/// Any Shape implementing this trait may be converted into chemfiles::UnitCell.
pub trait CellToChemCell: Shape {
    fn to_chem_cell(&self) -> chemfiles::UnitCell {
        match self.bounding_box() {
            Some(x) => chemfiles::UnitCell::new(x.into()),
            None => chemfiles::UnitCell::infinite(),
        }
    }
}

impl CellToChemCell for Cuboid {}
impl CellToChemCell for Sphere {}
impl CellToChemCell for Endless {}
impl CellToChemCell for Cell {}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use crate::topology::{
        block::{BlockActivationStatus, InsertionPolicy, MoleculeBlock},
        molecule::MoleculeKindBuilder,
        AtomKindBuilder, Bond, BondKind, BondOrder, IntermolecularBonded, Topology,
    };

    use float_cmp::assert_approx_eq;

    use super::*;

    #[test]
    fn atom_to_chemfiles() {
        let atom = AtomKindBuilder::default()
            .name("OW")
            .mass(16.0)
            .charge(-1.0)
            .element("O")
            .build()
            .unwrap();

        let converted = atom.to_chem_atom(Some("OX"));

        assert_eq!(converted.name(), "OX");
        assert_eq!(converted.mass(), 16.0);
        assert_eq!(converted.charge(), -1.0);
        assert_eq!(converted.atomic_type(), "O");

        let converted = atom.to_chem_atom(None);
        assert_eq!(converted.name(), "OW");
        assert_eq!(converted.mass(), 16.0);
        assert_eq!(converted.charge(), -1.0);
        assert_eq!(converted.atomic_type(), "O");
    }

    #[test]
    fn residue_to_chemfiles() {
        let residue = Residue::new("ALA", Some(4), 3..7);
        let converted = residue.to_chem_residue(7, 2);

        assert_eq!(converted.name(), "ALA");
        assert_eq!(converted.id(), Some(2));
        assert_eq!(converted.atoms(), vec![10, 11, 12, 13]);
    }

    #[test]
    fn group_to_chemfiles() {
        let molecule = MoleculeKindBuilder::default()
            .name("MOL".to_string())
            .atoms(
                ["OW", "OW", "OW", "HW", "HW", "OW"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
            )
            .atom_indices(vec![0, 0, 0, 1, 1, 0])
            .residues(vec![
                Residue::new("ALA", Some(4), 1..3),
                Residue::new("SER", Some(5), 5..6),
            ])
            .build()
            .unwrap();

        let group = Group::new(0, 0, 0..6);
        let converted = group.to_chem_residues(17, 3, &molecule);

        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].name(), "ALA");
        assert_eq!(converted[0].id(), Some(3));
        assert_eq!(converted[0].atoms(), vec![18, 19]);
        assert_eq!(converted[1].name(), "SER");
        assert_eq!(converted[1].id(), Some(4));
        assert_eq!(converted[1].atoms(), vec![22]);
    }

    #[test]
    fn reference_platform_to_chemfiles() {
        let molecule = MoleculeKindBuilder::default()
            .name("MOL")
            .atoms(
                ["OW", "OW", "OW", "HW", "HW", "OW"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
            )
            .atom_indices(vec![0, 0, 0, 1, 1, 0])
            .residues(vec![
                Residue::new("ALA", Some(4), 1..3),
                Residue::new("SER", Some(5), 5..6),
            ])
            .bonds(vec![
                Bond::new([1, 3], BondKind::Unspecified, BondOrder::Unspecified),
                Bond::new([3, 5], BondKind::Unspecified, BondOrder::Unspecified),
            ])
            .atom_names(vec![None, None, Some("O3".to_owned()), None, None, None])
            .build()
            .unwrap();

        let atom1 = AtomKindBuilder::default()
            .name("OW")
            .mass(16.0)
            .charge(-1.0)
            .element("O")
            .build()
            .unwrap();

        let atom2 = AtomKindBuilder::default()
            .name("HW")
            .mass(1.0)
            .charge(0.0)
            .element("H")
            .build()
            .unwrap();

        let block = MoleculeBlock::new(
            "MOL",
            0,
            3,
            BlockActivationStatus::Partial(2),
            Some(InsertionPolicy::Manual(vec![Point::default(); 21])),
        );

        let intermolecular_bond = Bond::new([7, 17], BondKind::Unspecified, BondOrder::Unspecified);

        let intermolecular = IntermolecularBonded::new(vec![intermolecular_bond], vec![], vec![]);

        let topology = Topology::new(
            vec![atom1, atom2].as_slice(),
            vec![molecule].as_slice(),
            intermolecular,
            vec![block],
        );

        let mut rng = rand::thread_rng();

        let context = ReferencePlatform::from_raw_parts(
            Rc::new(topology),
            Cell::Cuboid(Cuboid::new(10.0, 5.0, 2.5)),
            RefCell::new(vec![].into()),
            None::<&str>,
            &mut rng,
        )
        .unwrap();

        let converted = context.to_frame();

        // check atoms
        let atom_names = ["OW", "OW", "O3", "HW", "HW", "OW"];
        let atom_masses = [16.0, 16.0, 16.0, 1.0, 1.0, 16.0];
        let atom_charges = [-1.0, -1.0, -1.0, 0.0, 0.0, -1.0];
        let atomic_types = ["O", "O", "O", "H", "H", "O"];

        for (a, atom) in converted.iter_atoms().enumerate() {
            assert_eq!(atom.name(), atom_names[a % 6]);
            assert_eq!(atom.mass(), atom_masses[a % 6]);
            assert_eq!(atom.charge(), atom_charges[a % 6]);
            assert_eq!(atom.atomic_type(), atomic_types[a % 6]);
        }

        // check residues
        let residue_names = ["ALA", "SER"];
        let topology = converted.topology();
        for i in 0..6 {
            let residue = topology.residue(i).unwrap();
            assert_eq!(residue.name(), residue_names[i % 2]);
            assert_eq!(residue.id(), Some((i + 1) as i64));
            if i % 2 == 0 {
                assert_eq!(residue.atoms(), vec![(i / 2) * 6 + 1, (i / 2) * 6 + 2]);
            } else {
                assert_eq!(residue.atoms(), vec![(i / 2) * 6 + 5]);
            }
        }

        // check cell
        let frame_cell = converted.cell().lengths();
        assert_approx_eq!(f64, frame_cell[0], 10.0);
        assert_approx_eq!(f64, frame_cell[1], 5.0);
        assert_approx_eq!(f64, frame_cell[2], 2.5);

        // check bonds
        let bonds = [(1, 3), (3, 5), (7, 9), (7, 17), (9, 11), (13, 15), (15, 17)];
        for (expected, bond) in bonds.iter().zip(topology.bonds().iter()) {
            assert_eq!(expected.0, bond[0]);
            assert_eq!(expected.1, bond[1]);
        }
    }
}

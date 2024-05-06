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

//! # Inteface to the [`chemfiles`] crate

use std::path::Path;

use crate::{
    cell::{Cuboid, Endless, Shape, Sphere},
    group::GroupCollection,
    platform::reference::ReferencePlatform,
    topology::Residue,
    Context, Point, PointParticle,
};
use chemfiles::Frame;
use itertools::Itertools;

use crate::topology::TopologyLike;

use super::{AtomKind, NonOverlapping};

/// Create a new chemfiles::Frame from an input file in a supported format.
pub(crate) fn frame_from_file(filename: &impl AsRef<Path>) -> anyhow::Result<chemfiles::Frame> {
    let mut trajectory = chemfiles::Trajectory::open(filename, 'r')?;
    let mut frame = chemfiles::Frame::new();
    trajectory.read(&mut frame)?;
    Ok(frame)
}

/// Get positions of particles from chemfiles::Frame.
pub(crate) fn positions_from_frame(frame: &chemfiles::Frame) -> Vec<Point> {
    frame.positions().iter().map(|pos| (*pos).into()).collect()
}

pub trait ContextToChemFrame: Context {
    /// Convert system to chemfiles::Frame structure.
    ///
    /// ## Notes
    /// - Positions, residues, atom types and bonds are converted.
    /// - Custom properties of atoms and residues are not converted.
    /// - Angles and dihedrals are not converted.
    fn to_frame(&self) -> Frame {
        let mut frame = Frame::new();
        self.add_atoms_to_frame(&mut frame);
        //self.add_residues_to_frame(&mut frame);
        frame.set_cell(&self.cell().to_chem_cell());

        // todo! connectivity

        frame
    }

    /// Get all atom types defined for the system as chemfiles::Atom structures.
    fn get_chemfiles_atoms(&self) -> Vec<chemfiles::Atom> {
        self.topology().atoms().iter().map(|x| x.into()).collect()
    }

    /// Convert all faunus particles to chemfiles particles and add them to the chemfiles Frame.
    /// This converts all atoms, both active and inactive.
    fn add_atoms_to_frame(&self, frame: &mut Frame) {
        for particle in self.get_particles_all().iter() {
            frame.add_atom(
                self.get_chemfiles_atoms().get(particle.atom_id()).unwrap(),
                particle.pos.into(),
                None,
            )
        }
    }

    ///Convert faunus residues to chemfiles residues and add them to the chemfiles Frame.
    fn add_residues_to_frame(&self, frame: &mut Frame) {
        let topology = self.topology();

        let mut index = 0;
        for group in self.groups() {
            let molecule = &topology.molecules()[group.molecule()];
            for residue in molecule.residues().iter() {
                frame.add_residue(&residue.to_chem_residue(index));
                index = residue.range().end;
            }
        }
    }

    //fn add_bonds_to_frame(&self, frame: &mut Frame) {}
}

pub trait ResidueToChemResidue {
    /// Convert faunus residue to chemfiles residue.
    fn to_chem_residue(&self, init_atom_index: usize) -> chemfiles::Residue;
}

impl ResidueToChemResidue for Residue {
    fn to_chem_residue(&self, init_atom_index: usize) -> chemfiles::Residue {
        let mut chemfiles_residue = match self.number() {
            None => chemfiles::Residue::new(self.name()),
            Some(n) => chemfiles::Residue::with_id(self.name(), n as i64),
        };

        self.range()
            .for_each(|atom| chemfiles_residue.add_atom(atom + init_atom_index));

        chemfiles_residue
    }
}

impl ContextToChemFrame for ReferencePlatform {
    /// Convert all faunus particles to chemfiles particles and add them to the chemfiles Frame.
    ///
    /// ## Notes
    /// This implementation is specific to the ReferencePlatform and works with reference to the
    /// particles of the system instead of their copy. This should make the conversion faster.
    fn add_atoms_to_frame(&self, frame: &mut Frame) {
        for particle in self.particles() {
            frame.add_atom(
                self.get_chemfiles_atoms().get(particle.atom_id()).unwrap(),
                particle.pos.into(),
                None,
            )
        }
    }
}

/// Convert topology atom to chemfiles atom.
/// Does not convert custom properties.
impl core::convert::From<&AtomKind> for chemfiles::Atom {
    fn from(atom: &AtomKind) -> Self {
        let mut chemfiles_atom = Self::new(atom.name());
        chemfiles_atom.set_mass(atom.mass());
        chemfiles_atom.set_charge(atom.charge());
        if let Some(element) = atom.element() {
            chemfiles_atom.set_atomic_type(element);
        }

        chemfiles_atom
    }
}

/// Convert topology Residue to chemfiles residue.
impl core::convert::From<&Residue> for chemfiles::Residue {
    fn from(residue: &Residue) -> Self {
        let mut chemfiles_residue = match residue.number() {
            None => Self::new(residue.name()),
            Some(n) => Self::with_id(residue.name(), n as i64),
        };

        residue.range().map(|atom| chemfiles_residue.add_atom(atom));

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

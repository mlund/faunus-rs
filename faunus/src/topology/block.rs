// Copyright 2023-2024 Mikael Lund
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

//! This module implements:
//! a) the `MoleculeBlock` structure which is used to define the topology of the system,
//! b) the `InsertionPolicy` used to specify the construction of the molecule blocks.

use std::iter::zip;
use std::{cmp::Ordering, path::Path};

use super::structure;
use super::{molecule::MoleculeKind, AtomKind, InputPath};
use crate::dimension::Dimension;
use crate::transform;
use crate::{cell::SimulationCell, group::GroupSize, Context, Particle, Point};
use rand::rngs::ThreadRng;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

/// Describes the activation status of a MoleculeBlock.
/// Partial(n) means that only the first 'n' molecules of the block are active.
/// All means that all molecules of the block are active.
#[derive(Debug, Clone, PartialEq, Copy, Serialize, Deserialize, Default)]
#[serde(untagged)]
pub enum BlockActivationStatus {
    Partial(usize),
    #[default]
    All,
}

/// Specifies how the structure of molecules of a molecule block should be obtained or generated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InsertionPolicy {
    /// Read molecule block from a file.
    FromFile(InputPath),
    /// Place the atoms of each molecule of the block to random positions in the simulation cell.
    RandomAtomPos {
        #[serde(default)]
        directions: Dimension,
    },
    /// Read the structure of the molecule. Then place all molecules of the block
    /// to random positions in the simulation cell.
    RandomCOM {
        /// File containing the structure of the molecule.
        filename: InputPath,
        #[serde(default)]
        /// Rotate the molecule randomly; default is false.
        rotate: bool,
        #[serde(default)]
        /// Random directions to place the molecule.
        directions: Dimension,
        /// Optional offset vector to add to the molecule _after_ random COM has been chosen.
        offset: Option<Point>,
    },
    FixedCOM {
        /// File containing the structure of the molecule.
        filename: InputPath,
        /// Mass center position.
        position: Point,
        /// Rotate the molecule randomly; default is false.
        #[serde(default)]
        rotate: bool,
    },

    /// Define the positions of the atoms of all molecules manually, directly in the topology file.
    Manual(Vec<Point>),
}

impl InsertionPolicy {
    /// Obtain or generate positions of particles of a molecule block using the target InsertionPolicy.
    fn get_positions(
        &self,
        atoms: &[AtomKind],
        molecule_kind: &MoleculeKind,
        number: usize,
        cell: &impl SimulationCell,
        rng: &mut ThreadRng,
    ) -> anyhow::Result<Vec<Point>> {
        match self {
            Self::FromFile(filename) => {
                structure::positions_from_structure_file(filename.path().unwrap(), Some(cell))
            }

            Self::RandomAtomPos { directions } => Ok((0..(molecule_kind.atom_indices().len()
                * number))
                .map(|_| directions.filter(cell.get_point_inside(rng)))
                .collect::<Vec<_>>()),

            Self::RandomCOM {
                filename,
                rotate,
                directions,
                offset,
            } => InsertionPolicy::generate_random_com(
                molecule_kind,
                atoms,
                number,
                cell,
                rng,
                filename,
                *rotate,
                directions,
                offset,
            ),
            Self::FixedCOM {
                filename,
                position,
                rotate,
            } => InsertionPolicy::generate_fixed_com(
                molecule_kind,
                atoms,
                number,
                cell,
                filename,
                *rotate,
                position,
            ),

            // the coordinates should already be validated that they are compatible with the topology
            Self::Manual(positions) => Ok(positions.to_owned()),
        }
    }

    /// Read molecule positions from file and translate COM to origin (0,0,0)
    fn load_positions_to_origin(
        filename: &InputPath,
        molecule_kind: &MoleculeKind,
        atoms: &[AtomKind],
        cell: Option<&impl SimulationCell>,
    ) -> anyhow::Result<Vec<Point>> {
        // read coordinates of the molecule from input file
        let mut ref_positions =
            structure::positions_from_structure_file(filename.path().unwrap(), cell)?;

        // get the center of mass of the molecule
        let com = crate::aux::center_of_mass(
            &ref_positions,
            &molecule_kind
                .atom_indices()
                .iter()
                .map(|index| atoms[*index].mass())
                .collect::<Vec<_>>(),
        );

        // get positions relative to the center of mass
        ref_positions.iter_mut().for_each(|pos| *pos -= com);
        Ok(ref_positions)
    }

    /// Generate positions using the insertion policy FixedCOM.
    #[allow(clippy::too_many_arguments)]
    fn generate_fixed_com(
        molecule_kind: &MoleculeKind,
        atoms: &[AtomKind],
        num_molecules: usize,
        cell: &impl SimulationCell,
        filename: &InputPath,
        rotate: bool,
        position: &Point,
    ) -> anyhow::Result<Vec<Point>> {
        if num_molecules != 1 {
            anyhow::bail!("FixedCOM policy can only be used to insert a single molecule.");
        }
        Self::generate_random_com(
            molecule_kind,
            atoms,
            num_molecules,
            cell,
            &mut rand::thread_rng(),
            filename,
            rotate,
            &Dimension::None, // no random directions
            &Some(*position),
        )
    }

    /// Generate positions using the insertion policy RandomCOM.
    #[allow(clippy::too_many_arguments)]
    fn generate_random_com(
        molecule_kind: &MoleculeKind,
        atoms: &[AtomKind],
        num_molecules: usize,
        cell: &impl SimulationCell,
        rng: &mut ThreadRng,
        filename: &InputPath,
        rotate: bool,
        directions: &Dimension,
        offset: &Option<Point>,
    ) -> anyhow::Result<Vec<Point>> {
        let centered_positions =
            Self::load_positions_to_origin(filename, molecule_kind, atoms, Some(cell))?;

        // generate positions for a single molecule
        let gen_pos = |_| {
            let new_com =
                directions.filter(cell.get_point_inside(rng)) + offset.unwrap_or(Point::zeros());

            let mut positions: Vec<_> =
                centered_positions.iter().map(|pos| pos + new_com).collect();

            // Rotate the molecule
            // TODO: Optimize. We possibly want to do this before translating the molecule out
            //       of the origin.
            if rotate {
                transform::rotate_random(&mut positions, &new_com, rng);
            }

            // wrap particles into simulation cell
            positions.iter_mut().for_each(|pos| cell.boundary(pos));
            positions
        };

        // return a flat list of positions for `num_molecules` molecules
        Ok((0..num_molecules).flat_map(gen_pos).collect::<Vec<_>>())
    }

    /// Finalize path to the provided structure file (if it is provided) treating it either as an absolute path
    /// (if it is absolute) or as a path relative to `filename`.
    pub(super) fn finalize_path(&mut self, filename: impl AsRef<Path>) {
        match self {
            Self::FromFile(x) => x.finalize(filename),
            Self::RandomCOM { filename: x, .. } => x.finalize(filename),
            Self::FixedCOM { filename: x, .. } => x.finalize(filename),
            Self::RandomAtomPos { .. } | Self::Manual(_) => (),
        }
    }
}

/// A block of molecules of the same molecule kind.
#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate)]
#[serde(deny_unknown_fields)]
pub struct MoleculeBlock {
    /// Name of the molecule kind of molecules in this block.
    #[serde(rename = "molecule")]
    molecule_name: String,
    /// Index of the molecule kind.
    /// Only defined for MoleculeBlock in a specific Topology.
    #[serde(skip)]
    molecule_id: usize,
    /// Number of molecules in this block.
    #[serde(rename = "N")]
    num_molecules: usize,
    /// Number of active molecules in this block.
    #[serde(default)]
    active: BlockActivationStatus,
    /// Specifies how the structure of the molecule block should be obtained.
    /// None => structure should be read from a separately provided structure file
    ///         TODO: Replace Option with variant so that it's more explicit what `None` means.
    insert: Option<InsertionPolicy>,
}

impl MoleculeBlock {
    pub fn molecule(&self) -> &str {
        &self.molecule_name
    }

    pub fn molecule_index(&self) -> usize {
        self.molecule_id
    }

    pub fn num_molecules(&self) -> usize {
        self.num_molecules
    }

    pub fn active(&self) -> BlockActivationStatus {
        self.active
    }

    pub fn insert_policy(&self) -> Option<&InsertionPolicy> {
        self.insert.as_ref()
    }

    /// Create a new MoleculeBlock structure. This function does not perform any sanity checks.
    #[allow(dead_code)]
    pub(crate) fn new(
        molecule: &str,
        molecule_id: usize,
        num_molecules: usize,
        active: BlockActivationStatus,
        insert: Option<InsertionPolicy>,
    ) -> MoleculeBlock {
        MoleculeBlock {
            molecule_name: molecule.to_owned(),
            molecule_id,
            num_molecules,
            active,
            insert,
        }
    }

    /// Create groups from a MoleculeBlock and insert them into Context.
    ///
    /// ## Parameters
    /// - `context` - structure into which the groups should be added
    /// - `molecules` - list of all molecule kinds in the system
    /// - `external_positions` - list of particle coordinates to use;
    ///    must match exactly the number of coordinates that are required
    pub(crate) fn insert_block(
        &self,
        context: &mut impl Context,
        external_positions: &[Point],
        rng: &mut ThreadRng,
    ) -> anyhow::Result<()> {
        // let molecule = &molecules[self.molecule_id];
        let topology = context.topology();
        let molecule = &topology.moleculekinds()[self.molecule_id];
        let mut particle_counter = context.num_particles();
        let n_particles = molecule.len();

        // get flat list of positions of *all* molecules in the block
        let mut positions = match &self.insert {
            None => external_positions.to_owned(),
            Some(policy) => policy.get_positions(
                context.topology().atomkinds(),
                molecule,
                self.num_molecules,
                context.cell(),
                rng,
            )?,
        }
        .into_iter();

        // Make particles for a single molecule
        let mut make_particles = || {
            let particles: Vec<_> = zip(
                molecule.atom_indices(),
                positions.by_ref().take(n_particles),
            )
            .map(|(atom_id, position)| {
                let particle = Particle::new(*atom_id, particle_counter, position);
                particle_counter += 1;
                particle
            })
            .collect();
            particles
        };

        // create groups and populate them with particles
        for i in 0..self.num_molecules {
            let particles = make_particles();
            let group_id = context.add_group(molecule.id(), &particles)?.index();

            // deactivate the groups that should not be active
            match self.active {
                BlockActivationStatus::Partial(x) if i >= x => {
                    context.resize_group(group_id, GroupSize::Empty).unwrap()
                }
                _ => (),
            }
        }

        Ok(())
    }

    /// Get total number of atoms in a block.
    /// Panics if the molecule kind defined in the block does not exist.
    pub(crate) fn num_atoms(&self, molecules: &[MoleculeKind]) -> usize {
        self.num_molecules * molecules[self.molecule_id].atom_indices().len()
    }

    /// Set id (kind) of the molecules in the block.
    pub(super) fn set_molecule_id(&mut self, molecule_id: usize) {
        self.molecule_id = molecule_id;
    }

    /// Finalize MoleculeBlock parsing.
    pub(super) fn finalize(&mut self, filename: impl AsRef<Path>) -> Result<(), ValidationError> {
        // finalize the paths to input structure files
        match self.insert.as_mut() {
            None => (),
            Some(x) => x.finalize_path(filename),
        }

        // check that the number of active particles is not higher than the total number of particles
        if let BlockActivationStatus::Partial(active_mol) = self.active {
            match active_mol.cmp(&self.num_molecules) {
                Ordering::Greater => return Err(ValidationError::new("")
                    .with_message("the specified number of active molecules in a block is higher than the total number of molecules".into())),
                Ordering::Equal => self.active = BlockActivationStatus::All,
                Ordering::Less => (),
            }
        }
        Ok(())
    }
}

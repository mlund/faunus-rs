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

use std::{cmp::Ordering, path::Path};

use rand::rngs::ThreadRng;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

use crate::dimension::Dimension;
use crate::{cell::SimulationCell, group::GroupSize, Context, Particle, Point};

use super::AtomKind;
use super::{molecule::MoleculeKind, InputPath};

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
        filename: InputPath,
        #[serde(default)]
        rotate: bool,
        #[serde(default)]
        directions: Dimension,
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
        cell: &dyn SimulationCell,
        rng: &mut ThreadRng,
    ) -> anyhow::Result<Vec<Point>> {
        match self {
            Self::FromFile(filename) => super::structure::positions_from_structure_file(
                filename.path().unwrap(),
                Some(cell),
            ),

            Self::RandomAtomPos { directions } => Ok((0..(molecule_kind.atom_indices().len()
                * number))
                .map(|_| directions.filter(cell.get_point_inside(rng)))
                .collect::<Vec<Point>>()),

            Self::RandomCOM {
                filename,
                rotate,
                directions,
            } => InsertionPolicy::generate_random_com(
                molecule_kind,
                atoms,
                number,
                cell,
                rng,
                filename,
                *rotate,
                directions,
            ),

            // the coordinates should already be validated that they are compatible with the topology
            Self::Manual(positions) => Ok(positions.to_owned()),
        }
    }

    /// Generate positions using the insertion policy RandomCOM.
    #[allow(clippy::too_many_arguments)]
    fn generate_random_com(
        molecule_kind: &MoleculeKind,
        atoms: &[AtomKind],
        number: usize,
        cell: &dyn SimulationCell,
        rng: &mut ThreadRng,
        filename: &InputPath,
        rotate: bool,
        directions: &Dimension,
    ) -> anyhow::Result<Vec<Point>> {
        // read coordinates of the molecule from input file
        let mut ref_positions =
            super::structure::positions_from_structure_file(filename.path().unwrap(), Some(cell))?;

        // get the center of mass of the molecule
        let com = crate::aux::center_of_mass(
            &ref_positions,
            &molecule_kind
                .atom_indices()
                .iter()
                .map(|index| atoms[*index].mass())
                .collect::<Vec<f64>>(),
        );

        // get positions relative to the center of mass
        ref_positions.iter_mut().for_each(|pos| *pos -= com);

        // generate random positions for the molecules
        Ok((0..number)
            .flat_map(|_| {
                let random_com = directions.filter(cell.get_point_inside(rng));
                let mut molecule_positions = ref_positions
                    .iter()
                    .map(|pos| random_com + pos)
                    .collect::<Vec<_>>();

                // rotate the molecule
                if rotate {
                    crate::transform::rotate_random(&mut molecule_positions, &random_com, rng);
                }

                // wrap particles into simulation cell
                molecule_positions
                    .iter_mut()
                    .for_each(|pos| cell.boundary(pos));

                molecule_positions
            })
            .collect::<Vec<_>>())
    }

    /// Finalize path to the provided structure file (if it is provided) treating it either as an absolute path
    /// (if it is absolute) or as a path relative to `filename`.
    pub(super) fn finalize_path(&mut self, filename: impl AsRef<Path>) {
        match self {
            Self::FromFile(x) => x.finalize(filename),
            Self::RandomCOM { filename: x, .. } => x.finalize(filename),
            Self::RandomAtomPos { .. } | Self::Manual(_) => (),
        }
    }
}

/// A block of molecules of the same molecule kind.
#[derive(Debug, Clone, Serialize, Deserialize, Default, Validate)]
#[serde(deny_unknown_fields)]
pub struct MoleculeBlock {
    /// Name of the molecule kind of molecules in this block.
    molecule: String,
    /// Index of the molecule kind.
    /// Only defined for MoleculeBlock in a specific Topology.
    #[serde(skip)]
    molecule_index: usize,
    /// Number of molecules in this block.
    #[serde(rename = "N")]
    number: usize,
    /// Number of active molecules in this block.
    #[serde(default)]
    active: BlockActivationStatus,
    /// Specifies how the structure of the molecule block should be obtained.
    /// None => structure should be read from a separately provided structure file
    insert: Option<InsertionPolicy>,
}

impl MoleculeBlock {
    pub fn molecule(&self) -> &str {
        &self.molecule
    }

    pub fn molecule_index(&self) -> usize {
        self.molecule_index
    }

    pub fn number(&self) -> usize {
        self.number
    }

    pub fn active(&self) -> BlockActivationStatus {
        self.active
    }

    pub fn insert(&self) -> Option<&InsertionPolicy> {
        self.insert.as_ref()
    }

    /// Create a new MoleculeBlock structure. This function does not perform any sanity checks.
    #[allow(dead_code)]
    pub(crate) fn new(
        molecule: &str,
        molecule_index: usize,
        number: usize,
        active: BlockActivationStatus,
        insert: Option<InsertionPolicy>,
    ) -> MoleculeBlock {
        MoleculeBlock {
            molecule: molecule.to_owned(),
            molecule_index,
            number,
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
        atoms: &[AtomKind],
        molecules: &[MoleculeKind],
        external_positions: &[Point],
        rng: &mut ThreadRng,
    ) -> anyhow::Result<()> {
        let molecule = &molecules[self.molecule_index];
        let mut particle_counter = context.num_particles();

        // get positions of the particles in the block
        let mut positions = match &self.insert {
            None => external_positions.to_owned(),
            Some(policy) => {
                policy.get_positions(atoms, molecule, self.number, context.cell(), rng)?
            }
        }
        .into_iter();

        // create groups and populate them with particles
        for i in 0..self.number {
            // create the particles
            let particles: Vec<Particle> = molecule
                .atom_indices()
                .iter()
                .zip(positions.by_ref())
                .map(|(index, position)| {
                    let particle = Particle::new(*index, particle_counter, position);
                    particle_counter += 1;
                    particle
                })
                .collect();

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

    /// Get the number of atoms in a block.
    /// Panics if the molecule kind defined in the block does not exist.
    pub(crate) fn num_atoms(&self, molecules: &[MoleculeKind]) -> usize {
        self.number * molecules[self.molecule_index].atom_indices().len()
    }

    /// Set index of the molecule of the block.
    pub(super) fn set_molecule_index(&mut self, index: usize) {
        self.molecule_index = index;
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
            match active_mol.cmp(&self.number) {
                Ordering::Greater => return Err(ValidationError::new("")
                    .with_message("the specified number of active molecules in a block is higher than the total number of molecules".into())),
                Ordering::Equal => self.active = BlockActivationStatus::All,
                Ordering::Less => (),
            }
        }

        Ok(())
    }
}

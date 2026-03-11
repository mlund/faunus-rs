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
use crate::{cell::SimulationCell, group::GroupSize, Context, Particle, Point, UnitQuaternion};
use rand::rngs::ThreadRng;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

/// Describes the activation status of a MoleculeBlock.
/// Partial(n) means that only the first 'n' molecules of the block are active.
/// All means that all molecules of the block are active.
#[derive(Debug, Clone, PartialEq, Eq, Copy, Serialize, Deserialize, Default)]
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
        /// Optional minimum distance (Å) between bounding spheres of placed molecules.
        /// Uses rejection sampling to avoid overlaps in dense systems.
        min_distance: Option<f64>,
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
    /// Generate positions as a random walk with a fixed step size.
    /// Useful for linear polymer chains built from FASTA sequences.
    RandomWalk {
        /// Step size between consecutive atoms (Å)
        bond_length: f64,
        #[serde(default)]
        /// Random directions for placing the chain center.
        directions: Dimension,
    },
    /// Place molecules on a simple cubic grid. Requires a cuboidal cell.
    GridCOM {
        /// File containing the structure of the molecule.
        filename: InputPath,
        #[serde(default)]
        /// Rotate each molecule randomly; default is false.
        rotate: bool,
    },
}

impl InsertionPolicy {
    /// Obtain or generate positions and per-molecule quaternions.
    fn get_positions(
        &self,
        atoms: &[AtomKind],
        molecule_kind: &MoleculeKind,
        number: usize,
        cell: &impl SimulationCell,
        rng: &mut ThreadRng,
    ) -> anyhow::Result<(Vec<Point>, Vec<UnitQuaternion>)> {
        match self {
            Self::FromFile(filename) => {
                let pos =
                    structure::positions_from_structure_file(filename.path().unwrap(), Some(cell))?;
                Ok((pos, vec![UnitQuaternion::identity(); number]))
            }

            Self::RandomAtomPos { directions } => Ok((
                (0..(molecule_kind.atom_indices().len() * number))
                    .map(|_| directions.filter(cell.get_point_inside(rng)))
                    .collect(),
                vec![UnitQuaternion::identity(); number],
            )),

            Self::RandomCOM {
                filename,
                rotate,
                directions,
                offset,
                min_distance,
            } => Self::generate_random_com(
                molecule_kind,
                atoms,
                number,
                cell,
                rng,
                filename,
                *rotate,
                directions,
                offset,
                *min_distance,
            ),
            Self::FixedCOM {
                filename,
                position,
                rotate,
            } => Self::generate_fixed_com(
                molecule_kind,
                atoms,
                number,
                cell,
                filename,
                *rotate,
                position,
            ),

            Self::Manual(positions) => Ok((
                positions.to_owned(),
                vec![UnitQuaternion::identity(); number],
            )),

            Self::RandomWalk {
                bond_length,
                directions,
            } => {
                let pos = Self::generate_random_walk(
                    molecule_kind,
                    number,
                    cell,
                    rng,
                    *bond_length,
                    directions,
                )?;
                Ok((pos, vec![UnitQuaternion::identity(); number]))
            }

            Self::GridCOM { filename, rotate } => {
                Self::generate_grid_com(molecule_kind, atoms, number, cell, rng, filename, *rotate)
            }
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
        let com = crate::auxiliary::mass_center(
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
    ) -> anyhow::Result<(Vec<Point>, Vec<UnitQuaternion>)> {
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
            None,
        )
    }

    /// Generate positions using the insertion policy RandomCOM.
    ///
    /// If `min_distance` is set, rejection sampling ensures bounding spheres
    /// of placed molecules do not overlap. The check uses COM–COM distance
    /// vs. the sum of bounding radii plus `min_distance`.
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
        min_distance: Option<f64>,
    ) -> anyhow::Result<(Vec<Point>, Vec<UnitQuaternion>)> {
        const MAX_ATTEMPTS: usize = 1_000_000;

        let centered_positions =
            Self::load_positions_to_origin(filename, molecule_kind, atoms, Some(cell))?;

        let bounding_radius = centered_positions
            .iter()
            .map(|p| p.norm())
            .fold(0.0_f64, f64::max);

        let mut gen_pos = || {
            let new_com =
                directions.filter(cell.get_point_inside(rng)) + offset.unwrap_or(Point::zeros());
            let (positions, q) =
                Self::place_molecule_at(&centered_positions, &new_com, rotate, cell, rng);
            (new_com, positions, q)
        };

        let mut placed_coms: Vec<Point> = Vec::with_capacity(num_molecules);
        let mut all_positions: Vec<Point> =
            Vec::with_capacity(num_molecules * centered_positions.len());
        let mut quaternions: Vec<UnitQuaternion> = Vec::with_capacity(num_molecules);

        let threshold_sq = min_distance.map(|d| {
            let t = 2.0 * bounding_radius + d;
            t * t
        });

        for i in 0..num_molecules {
            let (com, positions, q) = if let Some(tsq) = threshold_sq {
                let mut attempts = 0;
                loop {
                    let (com, positions, q) = gen_pos();
                    let overlaps = placed_coms
                        .iter()
                        .any(|other| cell.distance(&com, other).norm_squared() < tsq);
                    if !overlaps {
                        break (com, positions, q);
                    }
                    attempts += 1;
                    if attempts >= MAX_ATTEMPTS {
                        anyhow::bail!(
                            "failed to place molecule {} of {} after {} attempts; \
                             consider reducing min_distance ({:.2} Å) or molecule count",
                            i + 1,
                            num_molecules,
                            MAX_ATTEMPTS,
                            min_distance.unwrap_or(0.0),
                        );
                    }
                }
            } else {
                gen_pos()
            };

            placed_coms.push(com);
            all_positions.extend(positions);
            quaternions.push(q);
        }

        Ok((all_positions, quaternions))
    }

    /// Translate centered molecule positions to a given COM, optionally rotate, and wrap into cell.
    ///
    /// Returns the placed positions and the applied rotation quaternion so
    /// that the group's orientation is correct from the start (needed by LD
    /// and 6D tabulated energies).
    fn place_molecule_at(
        centered_positions: &[Point],
        com: &Point,
        rotate: bool,
        cell: &impl SimulationCell,
        rng: &mut ThreadRng,
    ) -> (Vec<Point>, UnitQuaternion) {
        let mut positions: Vec<_> = centered_positions.iter().map(|pos| pos + com).collect();
        let quaternion = if rotate {
            let q = transform::random_rotation(rng);
            let matrix = q.to_rotation_matrix();
            positions
                .iter_mut()
                .for_each(|pos| *pos = matrix * (*pos - com) + com);
            q
        } else {
            UnitQuaternion::identity()
        };
        positions.iter_mut().for_each(|pos| cell.boundary(pos));
        (positions, quaternion)
    }

    /// Place molecules on a simple cubic grid within a cuboidal cell.
    ///
    /// Grid spacing is auto-calculated from box dimensions and molecule count.
    /// Fails if the cell has no bounding box or if any grid point falls outside the cell.
    fn generate_grid_com(
        molecule_kind: &MoleculeKind,
        atoms: &[AtomKind],
        num_molecules: usize,
        cell: &impl SimulationCell,
        rng: &mut ThreadRng,
        filename: &InputPath,
        rotate: bool,
    ) -> anyhow::Result<(Vec<Point>, Vec<UnitQuaternion>)> {
        let box_lengths = cell
            .bounding_box()
            .ok_or_else(|| anyhow::anyhow!("GridCOM requires a cell with a bounding box"))?;

        let centered_positions =
            Self::load_positions_to_origin(filename, molecule_kind, atoms, Some(cell))?;

        let n_per_axis = (num_molecules as f64).cbrt().ceil() as usize;
        let spacing = Point::new(
            box_lengths.x / n_per_axis as f64,
            box_lengths.y / n_per_axis as f64,
            box_lengths.z / n_per_axis as f64,
        );
        let half_box = box_lengths * 0.5;

        let mut grid_points = Vec::with_capacity(n_per_axis.pow(3));
        for ix in 0..n_per_axis {
            for iy in 0..n_per_axis {
                for iz in 0..n_per_axis {
                    let point = Point::new(
                        (ix as f64 + 0.5) * spacing.x - half_box.x,
                        (iy as f64 + 0.5) * spacing.y - half_box.y,
                        (iz as f64 + 0.5) * spacing.z - half_box.z,
                    );
                    if cell.is_outside(&point) {
                        anyhow::bail!(
                            "GridCOM grid point falls outside cell; \
                             this policy requires a cuboidal cell (Cuboid or Slit)"
                        );
                    }
                    grid_points.push(point);
                }
            }
        }

        let mut all_positions = Vec::with_capacity(num_molecules * centered_positions.len());
        let mut quaternions = Vec::with_capacity(num_molecules);

        for com in grid_points.iter().take(num_molecules) {
            let (pos, q) = Self::place_molecule_at(&centered_positions, com, rotate, cell, rng);
            all_positions.extend(pos);
            quaternions.push(q);
        }

        Ok((all_positions, quaternions))
    }

    /// Generate positions as a self-avoiding random walk from a random origin.
    ///
    /// Each molecule starts at a random point inside the cell, then each
    /// subsequent atom is placed `bond_length` away in a random direction.
    /// Steps that would place an atom outside the cell or closer than
    /// `bond_length` to any earlier bead in the same chain are rejected
    /// and retried with a new random direction.
    fn generate_random_walk(
        molecule_kind: &MoleculeKind,
        num_molecules: usize,
        cell: &impl SimulationCell,
        rng: &mut ThreadRng,
        bond_length: f64,
        directions: &Dimension,
    ) -> anyhow::Result<Vec<Point>> {
        let n_atoms = molecule_kind.atom_indices().len();
        let max_attempts = 1000 * n_atoms;
        let min_sq = bond_length * bond_length;
        let mut all_positions = Vec::with_capacity(n_atoms * num_molecules);

        for _ in 0..num_molecules {
            let chain_start = all_positions.len();
            let mut pos = directions.filter(cell.get_point_inside(rng));
            all_positions.push(pos);
            for _ in 1..n_atoms {
                pos = (0..max_attempts)
                    .map(|_| pos + transform::random_unit_vector(rng) * bond_length)
                    .find(|candidate| {
                        cell.is_inside(candidate)
                            // Exclude last element (bonded neighbor at exactly bond_length)
                            && !all_positions[chain_start..all_positions.len() - 1]
                                .iter()
                                .any(|p| (candidate - p).norm_squared() < min_sq)
                    })
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "RandomWalk: failed to place bead after {max_attempts} attempts"
                        )
                    })?;
                all_positions.push(pos);
            }
        }
        Ok(all_positions)
    }

    /// Finalize path to the provided structure file (if it is provided) treating it either as an absolute path
    /// (if it is absolute) or as a path relative to `filename`.
    pub(super) fn finalize_path(&mut self, filename: impl AsRef<Path>) {
        match self {
            Self::FromFile(x) => x.finalize(filename),
            Self::RandomCOM { filename: x, .. } => x.finalize(filename),
            Self::FixedCOM { filename: x, .. } => x.finalize(filename),
            Self::GridCOM { filename: x, .. } => x.finalize(filename),
            Self::RandomAtomPos { .. } | Self::Manual(_) | Self::RandomWalk { .. } => (),
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

    pub const fn molecule_index(&self) -> usize {
        self.molecule_id
    }

    pub const fn num_molecules(&self) -> usize {
        self.num_molecules
    }

    pub const fn active(&self) -> BlockActivationStatus {
        self.active
    }

    pub(crate) const fn insert_policy(&self) -> Option<&InsertionPolicy> {
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
    ) -> Self {
        Self {
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
    ///   must match exactly the number of coordinates that are required
    pub(crate) fn insert_block(
        &self,
        context: &mut impl Context,
        external_positions: &[Point],
        rng: &mut ThreadRng,
    ) -> anyhow::Result<()> {
        let topology = context.topology();
        let molecule = &topology.moleculekinds()[self.molecule_id];
        let n_particles = molecule.len();

        log::debug!(
            "Attempting to insert N={} '{}' molecules each with n={} particles.",
            self.num_molecules,
            molecule.name(),
            n_particles
        );

        // get flat list of positions and per-molecule quaternions
        let (positions_vec, quaternions) = match &self.insert {
            None => (
                external_positions.to_owned(),
                vec![UnitQuaternion::identity(); self.num_molecules],
            ),
            Some(policy) => policy.get_positions(
                context.topology().atomkinds(),
                molecule,
                self.num_molecules,
                context.cell(),
                rng,
            )?,
        };
        let mut positions = positions_vec.into_iter();

        // Make particles for a single molecule
        let mut make_particles = || {
            let particles: Vec<_> = zip(
                molecule.atom_indices(),
                positions.by_ref().take(n_particles),
            )
            .map(|(atom_id, position)| Particle::new(*atom_id, position))
            .collect();
            log::debug!(
                "Generated {} particles for molecule '{}'",
                particles.len(),
                molecule.name()
            );
            particles
        };

        if molecule.atomic() {
            // Atomic molecule: pool all atoms into a single group (quaternion meaningless)
            let particles: Vec<_> = (0..self.num_molecules)
                .flat_map(|_| make_particles())
                .collect();
            let group_index = context.add_group(molecule.id(), &particles)?.index();
            if let BlockActivationStatus::Partial(active) = self.active {
                context
                    .resize_group(group_index, GroupSize::Partial(active))
                    .unwrap();
            }
        } else {
            // Standard: one group per molecule
            for (i, q) in quaternions.into_iter().enumerate() {
                let particles = make_particles();
                let group_index = context.add_group(molecule.id(), &particles)?.index();
                // Sync quaternion with the rotation applied during placement so that
                // LD and 6D tabulated energies see the correct orientation from the start.
                context.groups_mut()[group_index].set_quaternion(q);
                context.update_mass_center(group_index);
                if let BlockActivationStatus::Partial(x) = self.active {
                    if i >= x {
                        context.resize_group(group_index, GroupSize::Empty).unwrap();
                    }
                }
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
    pub(super) const fn set_molecule_id(&mut self, molecule_id: usize) {
        self.molecule_id = molecule_id;
    }

    /// Finalize MoleculeBlock parsing.
    pub(super) fn finalize(&mut self, filename: impl AsRef<Path>) -> Result<(), ValidationError> {
        if let Some(x) = self.insert.as_mut() {
            x.finalize_path(filename);
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

#[cfg(test)]
mod tests {
    use crate::backend::Backend;
    use crate::group::GroupCollection;

    fn backend_from_str(yaml: &str) -> Backend {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml).unwrap();
        let mut rng = rand::thread_rng();
        Backend::new(tmp.path(), None, &mut rng).unwrap()
    }

    #[test]
    fn atomic_block_creates_single_group() {
        let ctx = backend_from_str(
            r#"
atoms:
  - {name: X, mass: 1.0, sigma: 1.0}
molecules:
  - name: particle
    atoms: [X]
    atomic: true
system:
  cell: !Cuboid [10.0, 10.0, 10.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: particle
      N: 20
      active: 8
      insert: !RandomAtomPos {}
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#,
        );
        assert_eq!(ctx.groups().len(), 1);
        assert_eq!(ctx.groups()[0].capacity(), 20);
        assert_eq!(ctx.groups()[0].len(), 8);
    }

    #[test]
    fn non_atomic_block_creates_many_groups() {
        let ctx = backend_from_str(
            r#"
atoms:
  - {name: X, mass: 1.0, sigma: 1.0}
molecules:
  - name: particle
    atoms: [X]
system:
  cell: !Cuboid [10.0, 10.0, 10.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: particle
      N: 20
      active: 8
      insert: !RandomAtomPos {}
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#,
        );
        assert_eq!(ctx.groups().len(), 20);
        assert_eq!(ctx.groups()[0].capacity(), 1);
    }
}

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

//! Implementation of the bonded interactions.

use crate::{
    group::Group,
    topology::{block::BlockActivationStatus, Topology},
    Change, Context, GroupChange, SyncFrom,
};

use super::{EnergyChange, EnergyTerm};

/// Energy term for computing intramolecular bonded interactions.
#[derive(Debug, Clone, Default)]
pub struct IntramolecularBonded {}

impl EnergyChange for IntramolecularBonded {
    /// Compute the energy associated with the intramolecular
    /// bonded interactions relevant to some change in the system.
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        match change {
            Change::Everything | Change::Volume(_, _) => self.all_groups(context),
            Change::None | Change::SingleGroup(_, GroupChange::RigidBody) => 0.0,
            // TODO! optimization: not all bonds have to be recalculated if a single particle inside a group changes
            Change::SingleGroup(id, change) => match change {
                GroupChange::None | GroupChange::RigidBody => 0.0,
                _ => self.one_group(context, &context.groups()[*id]),
            },
            Change::Groups(groups) => self.multiple_groups(
                context,
                &groups
                    .iter()
                    .filter_map(|(index, change)| match change {
                        GroupChange::None | GroupChange::RigidBody => None,
                        _ => Some(*index),
                    })
                    .collect::<Vec<usize>>(),
            ),
        }
    }
}

impl IntramolecularBonded {
    /// Calculate energy of all active bonded interactions of the specified groups.
    #[inline(always)]
    fn multiple_groups(&self, context: &impl Context, groups: &[usize]) -> f64 {
        groups
            .iter()
            .map(|&id| self.one_group(context, &context.groups()[id]))
            .sum()
    }

    /// Calculate energy of all active bonded interactions of all groups.
    #[inline(always)]
    fn all_groups(&self, context: &impl Context) -> f64 {
        context
            .groups()
            .iter()
            .map(|group| self.one_group(context, group))
            .sum()
    }

    /// Calculate energy of all active bonded interactions of target group.
    fn one_group(&self, context: &impl Context, group: &Group) -> f64 {
        let topology = context.topology_ref();
        let molecule = &topology.moleculekinds()[group.molecule()];

        let bond_energy: f64 = molecule
            .bonds()
            .iter()
            .map(|bond| bond.energy(context, group))
            .sum();

        let torsion_energy: f64 = molecule
            .torsions()
            .iter()
            .map(|torsion| torsion.energy(context, group))
            .sum();

        let dihedral_energy: f64 = molecule
            .dihedrals()
            .iter()
            .map(|dihedral| dihedral.energy(context, group))
            .sum();

        bond_energy + torsion_energy + dihedral_energy
    }
}

impl From<IntramolecularBonded> for EnergyTerm {
    fn from(term: IntramolecularBonded) -> Self {
        EnergyTerm::IntramolecularBonded(term)
    }
}

/// Energy term for computing intermolecular bonded interactions.
#[derive(Debug, Clone)]
pub struct IntermolecularBonded {
    /// Stores whether each particle of the system is active (true) or inactive (false).
    particles_status: Vec<bool>,
}

impl EnergyChange for IntermolecularBonded {
    /// Compute the energy associated with the intermolecular
    /// bonded interactions relevant to some change in the system.
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        match change {
            Change::None => 0.0,
            _ => {
                let intermolecular = context.topology_ref().intermolecular();
                let bond_energy: f64 = intermolecular
                    .bonds()
                    .iter()
                    .map(|bond| bond.energy_intermolecular(context, self))
                    .sum();

                let torsion_energy: f64 = intermolecular
                    .torsions()
                    .iter()
                    .map(|torsion| torsion.energy_intermolecular(context, self))
                    .sum();

                let dihedral_energy: f64 = intermolecular
                    .dihedrals()
                    .iter()
                    .map(|dihedral| dihedral.energy_intermolecular(context, self))
                    .sum();

                bond_energy + torsion_energy + dihedral_energy
            }
        }
    }
}

impl IntermolecularBonded {
    /// Create a new IntermolecularBonded energy term from Topology.
    #[allow(clippy::new_ret_no_self)]
    pub(super) fn new(topology: &Topology) -> EnergyTerm {
        let particles_status: Vec<bool> = topology
            .blocks()
            .iter()
            .flat_map(|block| {
                let molecule = &topology.moleculekinds()[block.molecule_index()];
                let num_atoms = molecule.atoms().len();
                match block.active() {
                    BlockActivationStatus::All => vec![true; block.num_molecules() * num_atoms],
                    BlockActivationStatus::Partial(x) => {
                        let mut status = vec![true; x * num_atoms];
                        status.extend(vec![false; (block.num_molecules() - x) * num_atoms]);
                        status
                    }
                }
            })
            .collect();

        EnergyTerm::IntermolecularBonded(IntermolecularBonded { particles_status })
    }

    /// Update the energy term. The update is needed if at least one particle was activated or deactivated.
    //
    // TODO:
    // Currently this updates the entire group upon any change in the size of the group.
    // However, `change` contains information about the number of (de)activated particles in the group.
    // We should probably use this information to only update the relevant part of the group.
    pub(super) fn update(&mut self, context: &impl Context, change: &Change) -> anyhow::Result<()> {
        match change {
            Change::SingleGroup(i, GroupChange::Resize(_)) => {
                self.update_status_one_group(&context.groups()[*i])
            }
            Change::Groups(groups) => self.update_status_multiple_groups(
                context,
                &groups
                    .iter()
                    .filter_map(|x| match x.1 {
                        // filter out groups which were not resized
                        GroupChange::Resize(_) => Some(x.0),
                        _ => None,
                    })
                    .collect::<Vec<usize>>(),
            ),
            Change::Everything => self.update_status_all(context),
            Change::SingleGroup(_, _) | Change::None | Change::Volume { .. } => (),
        }

        Ok(())
    }

    /// Check whether the particle with the provided absolute index is active.
    pub(crate) fn is_active(&self, abs_index: usize) -> bool {
        self.particles_status[abs_index]
    }

    /// Update the status of particles from a single group.
    fn update_status_one_group(&mut self, group: &Group) {
        group
            .iter_active()
            .for_each(|x| self.particles_status[x] = true);
        group
            .iter_inactive()
            .for_each(|x| self.particles_status[x] = false);
    }

    /// Update the status of particles from multiple groups.
    fn update_status_multiple_groups(&mut self, context: &impl Context, groups: &[usize]) {
        groups
            .iter()
            .map(|&i| &context.groups()[i])
            .for_each(|group| self.update_status_one_group(group));
    }

    /// Update the status of particles from all groups.
    fn update_status_all(&mut self, context: &impl Context) {
        context
            .groups()
            .iter()
            .for_each(|group| self.update_status_one_group(group));
    }
}

impl From<IntermolecularBonded> for EnergyTerm {
    fn from(term: IntermolecularBonded) -> Self {
        EnergyTerm::IntermolecularBonded(term)
    }
}

impl SyncFrom for IntermolecularBonded {
    fn sync_from(&mut self, other: &IntermolecularBonded, change: &Change) -> anyhow::Result<()> {
        match change {
            // TODO: this can be optimized to only update the relevant parts of the status array
            Change::Everything | Change::SingleGroup(_, GroupChange::Resize(_)) => {
                self.particles_status.clone_from(&other.particles_status)
            }
            Change::Groups(vec)
                if vec
                    .iter()
                    .any(|group| matches!(group.1, GroupChange::Resize(_))) =>
            {
                self.particles_status.clone_from(&other.particles_status)
            }
            Change::None | Change::Volume(_, _) | Change::SingleGroup(_, _) | Change::Groups(_) => {
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests_intramolecular {
    use std::{cell::RefCell, rc::Rc};

    use float_cmp::assert_approx_eq;

    use crate::{
        cell::{Cell, Cuboid},
        energy::Hamiltonian,
        group::{GroupCollection, GroupSize},
        montecarlo::NewOld,
        platform::reference::ReferencePlatform,
    };

    use super::*;

    /// Get intramolecular bonded structure for testing.
    fn get_intramolecular_bonded() -> (ReferencePlatform, IntramolecularBonded) {
        let topology = Topology::from_file("tests/files/bonded_interactions.yaml").unwrap();

        let mut rng = rand::thread_rng();
        let system = ReferencePlatform::from_raw_parts(
            Rc::new(topology),
            Cell::Cuboid(Cuboid::cubic(20.0)),
            RefCell::new(Hamiltonian::default()),
            None::<&str>,
            &mut rng,
        )
        .unwrap();

        let bonded = IntramolecularBonded::default();
        (system, bonded)
    }

    #[test]
    fn test_intramolecular_one_group() {
        let (system, bonded) = get_intramolecular_bonded();
        let expected = [1559328.708422025, 1433671.4698209586];

        assert_approx_eq!(
            f64,
            bonded.one_group(&system, &system.groups()[0]),
            expected[0]
        );
        assert_approx_eq!(
            f64,
            bonded.one_group(&system, &system.groups()[1]),
            expected[1]
        );
        assert_approx_eq!(f64, bonded.one_group(&system, &system.groups()[2]), 0.0)
    }

    #[test]
    fn test_intramolecular_multiple_groups() {
        let (system, bonded) = get_intramolecular_bonded();
        let expected = 1559328.708422025 + 1433671.4698209586;

        assert_approx_eq!(f64, bonded.multiple_groups(&system, &[0, 1]), expected);
    }

    #[test]
    fn test_intramolecular_all_groups() {
        let (mut system, bonded) = get_intramolecular_bonded();
        let expected = 1559328.708422025 + 1433671.4698209586;

        assert_approx_eq!(f64, bonded.all_groups(&system), expected);

        system.resize_group(2, GroupSize::Expand(4)).unwrap();
        let expected = 4112541.544583845;
        assert_approx_eq!(f64, bonded.all_groups(&system), expected);
    }

    #[test]
    fn test_intramolecular_energy() {
        let (system, bonded) = get_intramolecular_bonded();

        // no change
        let change = Change::None;
        assert_approx_eq!(f64, bonded.energy(&system, &change), 0.0);

        // change everything
        let change = Change::Everything;
        let expected = bonded.all_groups(&system);
        assert_approx_eq!(f64, bonded.energy(&system, &change), expected);

        // change volume
        let change = Change::Volume(
            crate::cell::VolumeScalePolicy::Isotropic,
            NewOld {
                old: 104.0,
                new: 108.0,
            },
        );
        assert_approx_eq!(f64, bonded.energy(&system, &change), expected);

        // single group with no change
        let change = Change::SingleGroup(1, GroupChange::None);
        assert_approx_eq!(f64, bonded.energy(&system, &change), 0.0);

        // multiple groups with no change
        let change = Change::Groups(vec![(0, GroupChange::None), (1, GroupChange::None)]);
        assert_approx_eq!(f64, bonded.energy(&system, &change), 0.0);

        // change single rigid group
        let change = Change::SingleGroup(1, GroupChange::RigidBody);
        assert_approx_eq!(f64, bonded.energy(&system, &change), 0.0);

        // change multiple rigid groups
        let change = Change::Groups(vec![
            (0, GroupChange::RigidBody),
            (1, GroupChange::RigidBody),
        ]);
        assert_approx_eq!(f64, bonded.energy(&system, &change), 0.0);

        // change several particles within a single group
        let change = Change::SingleGroup(1, GroupChange::PartialUpdate(vec![1, 2]));
        let expected = bonded.one_group(&system, &system.groups()[1]);
        assert_approx_eq!(f64, bonded.energy(&system, &change), expected);

        // change several particles in multiple groups
        let change = Change::Groups(vec![
            (0, GroupChange::PartialUpdate(vec![1])),
            (1, GroupChange::PartialUpdate(vec![0, 2])),
        ]);
        let expected = bonded.multiple_groups(&system, &[0, 1]);
        assert_approx_eq!(f64, bonded.energy(&system, &change), expected);
    }
}

#[cfg(test)]
mod tests_intermolecular {
    use std::{cell::RefCell, rc::Rc};

    use float_cmp::assert_approx_eq;

    use crate::{
        cell::{Cell, Cuboid},
        energy::Hamiltonian,
        group::{GroupCollection, GroupSize},
        montecarlo::NewOld,
        platform::reference::ReferencePlatform,
    };

    use super::*;

    #[test]
    fn test_intermolecular_new() {
        let topology = Topology::from_file("tests/files/bonded_interactions.yaml").unwrap();
        let intermolecular = match IntermolecularBonded::new(&topology) {
            EnergyTerm::IntermolecularBonded(e) => e,
            _ => panic!("IntermolecularBonded not constructed."),
        };

        for i in 0..8 {
            assert!(intermolecular.is_active(i))
        }

        for i in 8..12 {
            assert!(!intermolecular.is_active(i));
        }
    }

    /// Get intermolecular bonded structure for testing.
    fn get_intermolecular_bonded() -> (ReferencePlatform, IntermolecularBonded) {
        let topology = Topology::from_file("tests/files/bonded_interactions.yaml").unwrap();

        let mut rng = rand::thread_rng();
        let system = ReferencePlatform::from_raw_parts(
            Rc::new(topology.clone()),
            Cell::Cuboid(Cuboid::cubic(20.0)),
            RefCell::new(Hamiltonian::default()),
            None::<&str>,
            &mut rng,
        )
        .unwrap();

        let bonded = match IntermolecularBonded::new(&topology) {
            EnergyTerm::IntermolecularBonded(e) => e,
            _ => panic!("IntermolecularBonded not constructed."),
        };

        (system, bonded)
    }

    #[test]
    fn test_intermolecular_update() {
        let (mut system, mut bonded) = get_intermolecular_bonded();

        let original_bonded = bonded.clone();

        // we have to resize the groups at the start of the test so we can actually
        // see what changes cause the energy term to get updated
        system.resize_group(1, GroupSize::Shrink(2)).unwrap();
        system.resize_group(2, GroupSize::Expand(3)).unwrap();

        // no change
        let change = Change::None;
        let expected_status = [
            true, true, true, true, // first group
            true, true, true, true, // second group
            false, false, false, false, // third group
        ];
        bonded.update(&system, &change).unwrap();
        assert_eq!(bonded.particles_status, expected_status);

        // volume change
        let change = Change::Volume(
            crate::cell::VolumeScalePolicy::Isotropic,
            NewOld {
                old: 104.0,
                new: 108.0,
            },
        );
        bonded.update(&system, &change).unwrap();
        assert_eq!(bonded.particles_status, expected_status);

        // irrelevant single group change
        let change = Change::SingleGroup(2, GroupChange::PartialUpdate(vec![0, 1, 3]));
        bonded.update(&system, &change).unwrap();
        assert_eq!(bonded.particles_status, expected_status);

        // irrelevant changes in multiple groups
        let change = Change::Groups(vec![
            (0, GroupChange::PartialUpdate(vec![2])),
            (2, GroupChange::RigidBody),
        ]);
        bonded.update(&system, &change).unwrap();
        assert_eq!(bonded.particles_status, expected_status);

        // resize single group (irrelevant one)
        let change = Change::SingleGroup(0, GroupChange::Resize(GroupSize::Shrink(2)));
        bonded.update(&system, &change).unwrap();
        assert_eq!(bonded.particles_status, expected_status);

        // resize single group (relevant)
        let change = Change::SingleGroup(1, GroupChange::Resize(GroupSize::Shrink(2)));
        bonded.update(&system, &change).unwrap();
        let expected_status = [
            true, true, true, true, // first group
            true, true, false, false, // second group
            false, false, false, false, // third group
        ];
        assert_eq!(bonded.particles_status, expected_status);

        let mut bonded = original_bonded.clone();

        // resize multiple groups
        let change = Change::Groups(vec![
            (0, GroupChange::RigidBody),
            (1, GroupChange::Resize(GroupSize::Shrink(2))),
            (2, GroupChange::Resize(GroupSize::Expand(3))),
        ]);
        bonded.update(&system, &change).unwrap();
        let expected_status = [
            true, true, true, true, // first group
            true, true, false, false, // second group
            true, true, true, false, // third group
        ];
        assert_eq!(bonded.particles_status, expected_status);

        let mut bonded = original_bonded.clone();

        // everything changes
        let change = Change::Everything;
        bonded.update(&system, &change).unwrap();
        assert_eq!(bonded.particles_status, expected_status);
    }

    #[test]
    fn test_intermolecular_sync() {
        let (_, mut bonded) = get_intermolecular_bonded();

        let original_bonded = bonded.clone();

        let mut opposite_bonded = bonded.clone();
        for status in opposite_bonded.particles_status.iter_mut() {
            *status = false;
        }

        // no change
        let change = Change::None;
        bonded.sync_from(&opposite_bonded, &change).unwrap();
        assert_eq!(bonded.particles_status, original_bonded.particles_status);

        // volume change
        let change = Change::Volume(
            crate::cell::VolumeScalePolicy::Isotropic,
            NewOld {
                old: 104.0,
                new: 108.0,
            },
        );
        bonded.sync_from(&opposite_bonded, &change).unwrap();
        assert_eq!(bonded.particles_status, original_bonded.particles_status);

        // irrelevant single group change
        let change = Change::SingleGroup(2, GroupChange::PartialUpdate(vec![0, 1, 3]));
        bonded.sync_from(&opposite_bonded, &change).unwrap();
        assert_eq!(bonded.particles_status, original_bonded.particles_status);

        // irrelevant changes in multiple groups
        let change = Change::Groups(vec![
            (0, GroupChange::PartialUpdate(vec![2])),
            (2, GroupChange::RigidBody),
        ]);
        bonded.sync_from(&opposite_bonded, &change).unwrap();
        assert_eq!(bonded.particles_status, original_bonded.particles_status);

        // resize single group
        let change = Change::SingleGroup(1, GroupChange::Resize(GroupSize::Shrink(2)));
        bonded.sync_from(&opposite_bonded, &change).unwrap();
        assert_eq!(bonded.particles_status, opposite_bonded.particles_status);

        let mut bonded = original_bonded.clone();

        // resize multiple groups
        let change = Change::Groups(vec![
            (0, GroupChange::RigidBody),
            (1, GroupChange::Resize(GroupSize::Shrink(2))),
            (2, GroupChange::Resize(GroupSize::Expand(3))),
        ]);
        bonded.sync_from(&opposite_bonded, &change).unwrap();
        assert_eq!(bonded.particles_status, opposite_bonded.particles_status);

        let mut bonded = original_bonded.clone();

        // everything changes
        let change = Change::Everything;
        bonded.sync_from(&opposite_bonded, &change).unwrap();
        assert_eq!(bonded.particles_status, opposite_bonded.particles_status);
    }

    #[test]
    fn test_intermolecular_energy() {
        let (mut system, mut bonded) = get_intermolecular_bonded();

        // no change
        let change = Change::None;
        assert_approx_eq!(f64, bonded.energy(&system, &change), 0.0);

        // any other change
        let change = Change::Everything;
        let expected = 4349.90721737715;
        assert_approx_eq!(f64, bonded.energy(&system, &change), expected);

        let change = Change::SingleGroup(1, GroupChange::PartialUpdate(vec![0, 2]));
        assert_approx_eq!(f64, bonded.energy(&system, &change), expected);

        // resize group
        let resize = GroupSize::Expand(2);
        system.resize_group(2, resize).unwrap();
        let change = Change::SingleGroup(2, GroupChange::Resize(resize));
        bonded.update(&system, &change).unwrap();
        let expected = 4362.58996700314;
        assert_approx_eq!(f64, bonded.energy(&system, &change), expected);
    }
}

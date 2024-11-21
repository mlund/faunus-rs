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

//! Implementation of the Nonbonded energy terms.

use interatomic::twobody::{IsotropicTwobodyEnergy, NoInteraction};
use ndarray::Array2;
use std::fmt::Debug;
use std::path::Path;
use std::sync::Arc;

use crate::{
    energy::{builder::PairPotentialBuilder, EnergyTerm},
    topology::Topology,
    Change, Context, Group, GroupChange, SyncFrom,
};

use super::{builder::HamiltonianBuilder, exclusions::ExclusionMatrix, EnergyChange};

/// Trait implemented by all Energy Terms dealing with nonbonded interactions.
pub(super) trait NonbondedTerm {
    /// Calculates the energy between two interacting particles given by absolute indices.
    fn particle_with_particle(&self, context: &impl Context, i: usize, j: usize) -> f64;

    /// Compute the energy of a particle interacting with particles of the specified group.
    /// Ensures self-avoidance, i.e. makes sure that the particle does not interact with itself.
    ///
    /// ## Parameters
    /// - `context` - simulated system
    /// - `i` - absolute index of the particle
    /// - `group` - group to calculate interactions with
    ///
    /// ## Example
    /// - Group 1 contains three active particles: A, B, C.
    /// - Calling this method with particle A and group 1 will return the sum of interactions A-B and A-C.
    #[inline(always)]
    fn particle_with_group(&self, context: &impl Context, i: usize, group: &Group) -> f64 {
        group
            .iter_active()
            .filter(|j| *j != i)
            .map(|j| self.particle_with_particle(context, i, j))
            .sum()
    }

    /// Compute the energy of a particle interacting with particles of the specified group.
    ///
    /// ## Warning
    /// **Does not ensure self-avoidance!**
    /// Do not use if particle `i` belongs to `group`. Instead, use `particle_with_group`.
    ///
    /// ## Parameters
    /// - `context` - simulated system
    /// - `i` - absolute index of the particle
    /// - `group` - group the particle interacts with
    ///
    /// ## Example
    /// - Group 1 contains active particles A, and B.
    ///   Group 2 contains active particles C, D, and E.
    /// - Calling this method with particle A and group 2 will return the sum of interactions
    ///   A-C, A-D, and A-E.
    /// - Calling this method with particle A and group 1 will return the sum of interactions
    ///   A-A, A-B. To get just A-B, use `particle_with_group`.
    #[inline(always)]
    fn particle_with_group_unchecked(
        &self,
        context: &impl Context,
        i: usize,
        group: &Group,
    ) -> f64 {
        group
            .iter_active()
            .map(|j| self.particle_with_particle(context, i, j))
            .sum()
    }

    /// Compute the energy of a particle interacting with particles of all other groups.
    ///
    /// ## Parameters
    /// - `context` - simulated system
    /// - `i` - absolute index of the particle
    /// - `group` - group the particle is part of
    ///
    /// ## Example
    /// - The system consists of three groups (1-3) containing the following active particles:
    /// - Group 1 = A, B
    /// - Group 2 = C, D, E
    /// - Group 3 = F
    /// - Calling this method with particle A will return the sum of interactions
    ///   A-C, A-D, A-E, and A-F.
    #[inline(always)]
    fn particle_with_other_groups(&self, context: &impl Context, i: usize, group: &Group) -> f64 {
        context
            .groups()
            .iter()
            .filter(|group_j| group_j.index() != group.index())
            .map(|group_j| self.particle_with_group(context, i, group_j))
            .sum()
    }

    /// Compute the energy of a particle interacting with all other particles.
    ///
    /// ## Parameters
    /// - `context` - simulated system
    /// - `i` - absolute index of the particle
    /// - `group` - group the particle is part of
    ///
    /// ## Example
    /// - The system consists of three groups (1-3) containing the following active particles:
    /// - Group 1 = A, B
    /// - Group 2 = C, D, E
    /// - Group 3 = F
    /// - Calling this method with particle A will return the sum of interactions
    ///   A-B, A-C, A-D, A-E, and A-F.
    #[inline(always)]
    fn particle_with_all(&self, context: &impl Context, i: usize, group: &Group) -> f64 {
        self.particle_with_other_groups(context, i, group)
            + self.particle_with_group(context, i, group)
    }

    /// Compute the energy of a group interacting with a different group.
    ///
    /// ## Warning
    /// **Does not ensure self-avoidance!**
    /// Do not use if `group1` and `group2` are the same group. Instead, use `group_with_itself`.
    ///
    /// ## Parameters
    /// - `context` - simulated system
    /// - `group1` - the first interacting group
    /// - `group2` - the second interacting group
    ///
    /// ## Example
    /// - Group 1 contains active particles A, and B.
    ///   Group 2 contains active particles C, D, and E.
    /// - Calling this method with group 1 and group 2 will return the sum of interactions
    ///   A-C, A-D, A-E, B-C, B-D, and B-E.
    #[inline(always)]
    fn group_with_group(&self, context: &impl Context, group1: &Group, group2: &Group) -> f64 {
        group1
            .iter_active()
            .map(|i| self.particle_with_group_unchecked(context, i, group2))
            .sum()
    }

    /// Compute the energy of a group interacting with itself.
    ///
    /// ## Parameters
    /// - `context` - simulated system
    /// - `group` - interacting group
    ///
    /// ## Example
    /// - Group contains active particles A, B, and C.
    /// - Calling this method will return the sum of interactions A-B, A-C, B-C.
    #[inline(always)]
    fn group_with_itself(&self, context: &impl Context, group: &Group) -> f64 {
        group
            .iter_active()
            .enumerate()
            .flat_map(|(i, p1)| {
                group
                    .iter_active()
                    .skip(i + 1)
                    .map(move |p2| self.particle_with_particle(context, p1, p2))
            })
            .sum()
    }

    /// Compute the energy of a single group interacting with all other groups (not itself).
    ///
    /// ## Parameters
    /// - `context` - simulated system
    /// - `group` - interacting group
    ///
    /// ## Example
    /// - The system consists of three groups (1-3) containing the following active particles:
    /// - Group 1 = A, B
    /// - Group 2 = C, D, E
    /// - Group 3 = F
    /// - Calling this method with group 1 will return the sum of interactions
    ///   A-C, A-D, A-E, A-F, B-C, B-D, B-E and B-F.
    #[inline(always)]
    fn group_with_other_groups(&self, context: &impl Context, group: &Group) -> f64 {
        group
            .iter_active()
            .map(|i| self.particle_with_other_groups(context, i, group))
            .sum()
    }

    /// Compute the energy of a single group interacting with all particles
    /// (including particles of the group itself).
    ///
    /// ## Parameters
    /// - `context` - simulated system
    /// - `group` - interacting group
    ///
    /// ## Example
    /// - The system consists of three groups (1-3) containing the following active particles:
    /// - Group 1 = A, B
    /// - Group 2 = C, D, E
    /// - Group 3 = F
    /// - Calling this method with group 1 will return the sum of interactions
    ///   A-B, A-C, A-D, A-E, A-F, B-C, B-D, B-E and B-F.
    #[inline(always)]
    #[allow(dead_code)]
    fn group_with_all(&self, context: &impl Context, group: &Group) -> f64 {
        self.group_with_other_groups(context, group) + self.group_with_itself(context, group)
    }

    /// Compute the energy of all nonbonded interactions.
    ///
    /// ## Parameters
    /// - `context` - simulated system
    ///
    /// ## Example
    /// - The system consists of three groups (1-3) containing the following active particles:
    /// - Group 1 = A, B
    /// - Group 2 = C, D, E
    /// - Group 3 = F
    /// - Calling this method will return the sum of interactions
    ///   A-B, A-C, A-D, A-E, A-F, B-C, B-D, B-E, B-F, C-D, C-E, C-F, D-E, D-F, E-F.
    #[inline(always)]
    fn total_nonbonded(&self, context: &impl Context) -> f64 {
        context
            .groups()
            .iter()
            .enumerate()
            .flat_map(|(i, group_i)| {
                context
                    .groups()
                    .iter()
                    .skip(i + 1)
                    .map(move |group_j| self.group_with_group(context, group_i, group_j))
                    .chain(std::iter::once(self.group_with_itself(context, group_i)))
            })
            .sum()
    }
}

/// Energy term for computing nonbonded interactions
/// using a matrix of `IsotropicTwobodyEnergy` trait objects.
///
/// # Note
///
/// We use `Arc<dyn IsotropicTwobodyEnergy>` for thread-safety.
/// `Box` is not thread-safe but perhaps more performant(?).
#[derive(Debug, Clone)]
pub struct NonbondedMatrix {
    /// Matrix of pair potentials based on atom type ids.
    potentials: Array2<Arc<dyn IsotropicTwobodyEnergy>>,
    /// Matrix of excluded interactions.
    exclusions: ExclusionMatrix,
}

impl EnergyChange for NonbondedMatrix {
    /// Compute the energy of the EnergyTerm relevant to some change in the system.
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        match change {
            Change::Everything | Change::Volume(_, _) => self.total_nonbonded(context),
            Change::SingleGroup(group_index, group_change) => {
                self.single_group_change(context, *group_index, group_change)
            }
            Change::Groups(vec) => vec
                .iter()
                .map(|(group, change)| self.single_group_change(context, *group, change))
                .sum(),
            Change::None => 0.0,
        }
    }
}

impl NonbondedTerm for NonbondedMatrix {
    /// Calculates the energy between two particles given by indices.
    #[inline(always)]
    fn particle_with_particle(&self, context: &impl Context, i: usize, j: usize) -> f64 {
        let distance_squared = context.get_distance_squared(i, j);
        self.exclusions.get((i, j)) as f64
            * self
                .potentials
                .get((context.get_atomkind(i), context.get_atomkind(j)))
                .expect("Atom kinds should exist in the nonbonded matrix.")
                .isotropic_twobody_energy(distance_squared)
    }
}

impl NonbondedMatrix {
    /// Create from YAML file and a topology
    ///
    /// Can be used to generate a new `EnergyTerm` with:
    ///
    /// ```ignore
    /// let energy = EnergyTerm::From(NonbondedMatrix::from_file(...).unwrap());
    /// ```
    pub fn from_file(
        file: impl AsRef<Path>,
        topology: &Topology,
    ) -> anyhow::Result<NonbondedMatrix> {
        Self::new(
            &HamiltonianBuilder::from_file(file)?.pairpot_builder,
            topology,
        )
    }

    /// Create a new NonbondedReference structure wrapped in an EnergyTerm enum.
    #[allow(clippy::new_ret_no_self)]
    pub(super) fn new(
        pairpot_builder: &PairPotentialBuilder,
        topology: &Topology,
    ) -> anyhow::Result<NonbondedMatrix> {
        let atoms = topology.atomkinds();
        let n_atom_types = atoms.len();

        let mut potentials: Array2<Arc<dyn IsotropicTwobodyEnergy>> = Array2::from_elem(
            (n_atom_types, n_atom_types),
            Arc::<NoInteraction>::default(),
        );

        for i in 0..n_atom_types {
            for j in 0..n_atom_types {
                let interaction = pairpot_builder.get_interaction(&atoms[i], &atoms[j])?;
                potentials[(i, j)] = interaction.into();
            }
        }

        let exclusions = ExclusionMatrix::from_topology(topology);

        Ok(NonbondedMatrix {
            potentials,
            exclusions,
        })
    }

    /// Matches all possible single group perturbations and returns the energy.
    fn single_group_change(
        &self,
        context: &impl Context,
        group_index: usize,
        change: &GroupChange,
    ) -> f64 {
        match change {
            GroupChange::RigidBody => {
                self.group_with_other_groups(context, &context.groups()[group_index])
            }
            GroupChange::Resize(_) | GroupChange::UpdateIdentity(_) => {
                todo!("Resize and UpdateIdentity changes are not yet implemented for NonbondedMatrix.")
            }
            GroupChange::PartialUpdate(x) => x
                .iter()
                .map(|&particle| {
                    let group = &context.groups()[group_index];
                    // the PartialUpdate stores relative indices of particles
                    match group.to_absolute_index(particle) {
                        Ok(abs_index) => self.particle_with_all(context, abs_index, group),
                        // if the particle is not active, return 0.0
                        Err(_) => 0.0,
                    }
                })
                .sum(),
            GroupChange::None => 0.0,
        }
    }
    /// Get square matrix of pair potentials for all atom type combinations.
    pub fn get_potentials(&self) -> &Array2<Arc<dyn IsotropicTwobodyEnergy>> {
        &self.potentials
    }
    /// Get square matrix of pair potentials for all atom type combinations.
    pub fn get_potentials_mut(&mut self) -> &mut Array2<Arc<dyn IsotropicTwobodyEnergy>> {
        &mut self.potentials
    }
}

impl From<NonbondedMatrix> for EnergyTerm {
    fn from(nonbonded: NonbondedMatrix) -> Self {
        EnergyTerm::NonbondedMatrix(nonbonded)
    }
}

impl SyncFrom for NonbondedMatrix {
    fn sync_from(&mut self, other: &NonbondedMatrix, change: &Change) -> anyhow::Result<()> {
        match change {
            Change::Everything => self.potentials.clone_from(&other.potentials),
            Change::None | Change::Volume(_, _) | Change::SingleGroup(_, _) | Change::Groups(_) => {
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use float_cmp::assert_approx_eq;

    use crate::{
        cell::{Cell, Cuboid},
        energy::{builder::HamiltonianBuilder, Hamiltonian},
        group::{GroupCollection, GroupSize},
        montecarlo::NewOld,
        platform::reference::ReferencePlatform,
        topology::Topology,
    };

    use super::*;

    /// Compare behavior of two `IsotropicTwobodyEnergy` trait objects.
    fn assert_behavior(obj1: &dyn IsotropicTwobodyEnergy, obj2: &dyn IsotropicTwobodyEnergy) {
        let testing_distances = [0.00201, 0.7, 12.3, 12457.6];

        for &dist in testing_distances.iter() {
            assert_approx_eq!(
                f64,
                obj1.isotropic_twobody_energy(dist),
                obj2.isotropic_twobody_energy(dist)
            );
        }
    }

    #[test]
    fn test_nonbonded_matrix_new() {
        let topology = Topology::from_file("tests/files/topology_pass.yaml").unwrap();
        let pairpot_builder = HamiltonianBuilder::from_file("tests/files/topology_pass.yaml")
            .unwrap()
            .pairpot_builder;

        let nonbonded = NonbondedMatrix::new(&pairpot_builder, &topology).unwrap();

        assert_eq!(
            nonbonded.potentials.len(),
            topology.atomkinds().len() * topology.atomkinds().len()
        );

        for i in 0..topology.atomkinds().len() {
            for j in (i + 1)..topology.atomkinds().len() {
                assert_behavior(
                    nonbonded.potentials.get((i, j)).unwrap(),
                    nonbonded.potentials.get((j, i)).unwrap(),
                );
            }
        }

        // O, C with anything: default interaction
        let o_index = topology
            .atomkinds()
            .iter()
            .position(|x| x.name() == "O")
            .unwrap();
        let c_index = topology
            .atomkinds()
            .iter()
            .position(|x| x.name() == "C")
            .unwrap();

        let default = nonbonded.potentials.get((o_index, o_index)).unwrap();

        for i in [o_index, c_index] {
            for j in 0..topology.atomkinds().len() {
                assert_behavior(nonbonded.potentials.get((i, j)).unwrap(), default);
            }
        }

        // X interacts slightly differently with charged atoms because it is itself charged
        let x_index = topology
            .atomkinds()
            .iter()
            .position(|x| x.name() == "X")
            .unwrap();
        let ow_index = topology
            .atomkinds()
            .iter()
            .position(|x| x.name() == "OW")
            .unwrap();

        for i in 0..topology.atomkinds().len() {
            if i == x_index || i == ow_index {
                continue;
            }

            assert_behavior(nonbonded.potentials.get((x_index, i)).unwrap(), default);
        }
    }

    /// Assert particle-particle interaction energy.
    fn assert_part_part(
        system: &impl Context,
        nonbonded: &NonbondedMatrix,
        i: usize,
        j: usize,
        expected: f64,
    ) {
        assert_approx_eq!(
            f64,
            nonbonded.particle_with_particle(system, i, j),
            expected
        );
    }

    /// Get nonbonded matrix for testing.
    fn get_test_matrix() -> (ReferencePlatform, NonbondedMatrix) {
        let topology = Topology::from_file("tests/files/nonbonded_interactions.yaml").unwrap();
        let builder = HamiltonianBuilder::from_file("tests/files/nonbonded_interactions.yaml")
            .unwrap()
            .pairpot_builder;

        let nonbonded = NonbondedMatrix::new(&builder, &topology).unwrap();

        let mut rng = rand::thread_rng();
        let system = ReferencePlatform::from_raw_parts(
            Rc::new(topology),
            Cell::Cuboid(Cuboid::cubic(20.0)),
            RefCell::new(Hamiltonian::from(vec![nonbonded.clone().into()])),
            None::<&str>,
            &mut rng,
        )
        .unwrap();

        (system, nonbonded)
    }

    #[test]
    fn test_nonbonded_matrix_particle_particle() {
        let (system, nonbonded) = get_test_matrix();

        // intramolecular

        let intramolecular_a1b_energy = -0.356652949245542;
        for (i, j) in [(0, 1), (3, 4), (6, 7)] {
            assert_part_part(&system, &nonbonded, i, j, intramolecular_a1b_energy);
            assert_part_part(&system, &nonbonded, j, i, intramolecular_a1b_energy);
        }

        let intramolecular_a1a2_energy = 0.0;
        for (i, j) in [(0, 2), (3, 5), (6, 8)] {
            assert_part_part(&system, &nonbonded, i, j, intramolecular_a1a2_energy);
            assert_part_part(&system, &nonbonded, j, i, intramolecular_a1a2_energy);
        }

        let intramolecular_a2b_energy = -0.000233230711693257;
        for (i, j) in [(1, 2), (4, 5), (7, 8)] {
            assert_part_part(&system, &nonbonded, i, j, intramolecular_a2b_energy);
            assert_part_part(&system, &nonbonded, j, i, intramolecular_a2b_energy);
        }

        // intermolecular

        let intermolecular_a1a1_energy =
            [5.00761780387822, 5.00761780387822, -0.000090421636081691];
        for ((i, j), energy) in [(0, 3), (0, 6), (3, 6)]
            .into_iter()
            .zip(intermolecular_a1a1_energy)
        {
            assert_part_part(&system, &nonbonded, i, j, energy);
            assert_part_part(&system, &nonbonded, j, i, energy);
        }

        let intermolecular_a1b_energy = [
            -0.000508026822504991,
            -0.356652949245542,
            -0.000508026822504991,
            -2.3703647517146784e-5,
            -0.356652949245542,
            -0.000508026822504991,
        ];
        for ((i, j), energy) in [(0, 4), (0, 7), (3, 7), (4, 6), (1, 3), (1, 6)]
            .into_iter()
            .zip(intermolecular_a1b_energy)
        {
            assert_part_part(&system, &nonbonded, i, j, energy);
            assert_part_part(&system, &nonbonded, j, i, energy);
        }

        let intermolecular_a1a2_energy = [
            -6.406572630990959e-6,
            3.88211425347351,
            6.120644662447584,
            -1.2499998437500003e-6,
            3.88211425347351,
            -6.406572630990959e-6,
        ];
        for ((i, j), energy) in [(0, 5), (0, 8), (3, 8), (5, 6), (2, 3), (2, 6)]
            .into_iter()
            .zip(intermolecular_a1a2_energy)
        {
            assert_part_part(&system, &nonbonded, i, j, energy);
            assert_part_part(&system, &nonbonded, j, i, energy);
        }

        let intermolecular_bb_energy =
            [-0.713305898491084, -0.713305898491084, -0.01156737611454047];
        for ((i, j), energy) in [(1, 4), (1, 7), (4, 7)]
            .into_iter()
            .zip(intermolecular_bb_energy)
        {
            assert_part_part(&system, &nonbonded, i, j, energy);
            assert_part_part(&system, &nonbonded, j, i, energy);
        }

        let intermolecular_a2b_energy = [
            -1.748899941931173e-5,
            -0.0075075032697152,
            -0.0075075032697152,
            -2.6853740564936314e-6,
            -0.0075075032697152,
            -1.748899941931173e-5,
        ];
        for ((i, j), energy) in [(1, 5), (1, 8), (4, 8), (7, 5), (4, 2), (7, 2)]
            .into_iter()
            .zip(intermolecular_a2b_energy)
        {
            assert_part_part(&system, &nonbonded, i, j, energy);
            assert_part_part(&system, &nonbonded, j, i, energy);
        }

        let intermolecular_a2a2_energy =
            [5.00761780387822, 5.007617803878217, -9.042163608169031e-5];
        for ((i, j), energy) in [(2, 5), (2, 8), (5, 8)]
            .into_iter()
            .zip(intermolecular_a2a2_energy)
        {
            assert_part_part(&system, &nonbonded, i, j, energy);
            assert_part_part(&system, &nonbonded, j, i, energy);
        }
    }

    #[test]
    fn test_nonbonded_matrix_particle_with_self_group() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 3, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let expected = nonbonded.particle_with_particle(&system, 0, 1);
        assert_approx_eq!(
            f64,
            nonbonded.particle_with_group(&system, 0, &system.groups()[0]),
            expected
        );

        let expected = nonbonded.particle_with_particle(&system, 3, 4)
            + nonbonded.particle_with_particle(&system, 4, 5);
        assert_approx_eq!(
            f64,
            nonbonded.particle_with_group(&system, 4, &system.groups()[1]),
            expected
        )
    }

    #[test]
    fn test_nonbonded_matrix_particle_with_group() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let expected = nonbonded.particle_with_particle(&system, 1, 3)
            + nonbonded.particle_with_particle(&system, 1, 4)
            + nonbonded.particle_with_particle(&system, 1, 5);
        assert_approx_eq!(
            f64,
            nonbonded.particle_with_group(&system, 1, &system.groups()[1]),
            expected
        );

        let expected = 0.0;
        assert_approx_eq!(
            f64,
            nonbonded.particle_with_group(&system, 0, &system.groups()[2]),
            expected
        );

        let expected = nonbonded.particle_with_particle(&system, 5, 0)
            + nonbonded.particle_with_particle(&system, 5, 1);
        assert_approx_eq!(
            f64,
            nonbonded.particle_with_group(&system, 5, &system.groups()[0]),
            expected
        );
    }

    #[test]
    fn test_nonbonded_matrix_particle_with_other_groups() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let expected = nonbonded.particle_with_group(&system, 0, &system.groups()[1])
            + nonbonded.particle_with_group(&system, 0, &system.groups()[2]);
        assert_approx_eq!(
            f64,
            nonbonded.particle_with_other_groups(&system, 0, &system.groups()[0]),
            expected
        );

        let expected = nonbonded.particle_with_group(&system, 3, &system.groups()[0])
            + nonbonded.particle_with_group(&system, 3, &system.groups()[2]);
        assert_approx_eq!(
            f64,
            nonbonded.particle_with_other_groups(&system, 3, &system.groups()[1]),
            expected
        );
    }

    #[test]
    fn test_nonbonded_matrix_particle_with_all() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let expected = nonbonded.particle_with_particle(&system, 1, 0)
            + nonbonded.particle_with_particle(&system, 1, 3)
            + nonbonded.particle_with_particle(&system, 1, 4)
            + nonbonded.particle_with_particle(&system, 1, 5);

        assert_approx_eq!(
            f64,
            nonbonded.particle_with_all(&system, 1, &system.groups()[0]),
            expected
        );
    }

    #[test]
    fn test_nonbonded_matrix_group_with_group() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let expected = nonbonded.particle_with_group(&system, 0, &system.groups()[1])
            + nonbonded.particle_with_group(&system, 1, &system.groups()[1]);

        assert_approx_eq!(
            f64,
            nonbonded.group_with_group(&system, &system.groups()[0], &system.groups()[1]),
            expected
        );
        assert_approx_eq!(
            f64,
            nonbonded.group_with_group(&system, &system.groups()[1], &system.groups()[0]),
            expected
        );

        let expected = 0.0;
        assert_approx_eq!(
            f64,
            nonbonded.group_with_group(&system, &system.groups()[0], &system.groups()[2]),
            expected
        );
        assert_approx_eq!(
            f64,
            nonbonded.group_with_group(&system, &system.groups()[2], &system.groups()[0]),
            expected
        );
    }

    #[test]
    fn test_nonbonded_matrix_group_with_itself() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let expected = nonbonded.particle_with_particle(&system, 0, 1);
        assert_approx_eq!(
            f64,
            nonbonded.group_with_itself(&system, &system.groups()[0]),
            expected
        );

        let expected = nonbonded.particle_with_particle(&system, 3, 4)
            + nonbonded.particle_with_particle(&system, 3, 5)
            + nonbonded.particle_with_particle(&system, 4, 5);
        assert_approx_eq!(
            f64,
            nonbonded.group_with_itself(&system, &system.groups()[1]),
            expected
        );

        let expected = 0.0;
        assert_approx_eq!(
            f64,
            nonbonded.group_with_itself(&system, &system.groups()[2]),
            expected
        );
    }

    #[test]
    fn test_nonbonded_matrix_group_with_other_groups() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let expected =
            nonbonded.group_with_group(&system, &system.groups()[0], &system.groups()[1]);
        assert_approx_eq!(
            f64,
            nonbonded.group_with_other_groups(&system, &system.groups()[0]),
            expected
        );
        assert_approx_eq!(
            f64,
            nonbonded.group_with_other_groups(&system, &system.groups()[1]),
            expected
        );

        let expected = 0.0;
        assert_approx_eq!(
            f64,
            nonbonded.group_with_other_groups(&system, &system.groups()[2]),
            expected
        );
    }

    #[test]
    fn test_nonbonded_matrix_group_with_all() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let expected =
            nonbonded.group_with_group(&system, &system.groups()[0], &system.groups()[1])
                + nonbonded.group_with_itself(&system, &system.groups()[0]);
        assert_approx_eq!(
            f64,
            nonbonded.group_with_all(&system, &system.groups()[0]),
            expected
        );

        let expected =
            nonbonded.group_with_group(&system, &system.groups()[0], &system.groups()[1])
                + nonbonded.group_with_itself(&system, &system.groups()[1]);
        assert_approx_eq!(
            f64,
            nonbonded.group_with_all(&system, &system.groups()[1]),
            expected
        );

        let expected = 0.0;
        assert_approx_eq!(
            f64,
            nonbonded.group_with_all(&system, &system.groups()[2]),
            expected
        );
    }

    #[test]
    fn test_nonbonded_matrix_total_nonbonded() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        let interactions = [
            (0, 1),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 3),
            (1, 4),
            (1, 5),
            (3, 4),
            (3, 5),
            (4, 5),
        ];

        let expected = interactions
            .into_iter()
            .map(|(i, j)| nonbonded.particle_with_particle(&system, i, j))
            .sum();
        assert_approx_eq!(f64, nonbonded.total_nonbonded(&system), expected);
    }

    #[test]
    fn test_nonbonded_matrix_energy() {
        let (mut system, nonbonded) = get_test_matrix();

        // deactivate particles 2, 6, 7, 8
        system.resize_group(0, GroupSize::Shrink(1)).unwrap();
        system.resize_group(2, GroupSize::Shrink(3)).unwrap();

        // no change
        let change = Change::None;
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), 0.0);

        // change everything
        let change = Change::Everything;
        let expected = nonbonded.total_nonbonded(&system);
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);

        // change volume
        let change = Change::Volume(
            crate::cell::VolumeScalePolicy::Isotropic,
            NewOld {
                old: 104.0,
                new: 108.0,
            },
        );
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);

        // multiple groups with no change
        let change = Change::Groups(vec![(0, GroupChange::None), (1, GroupChange::None)]);
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), 0.0);

        // change single rigid group
        let change = Change::SingleGroup(1, GroupChange::RigidBody);
        let expected = nonbonded.group_with_other_groups(&system, &system.groups()[1]);
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);

        // change multiple rigid groups
        let change = Change::Groups(vec![
            (0, GroupChange::RigidBody),
            (1, GroupChange::RigidBody),
        ]);
        let expected = nonbonded.group_with_other_groups(&system, &system.groups()[0])
            + nonbonded.group_with_other_groups(&system, &system.groups()[1]);
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);

        // change several particles within a single group
        let change = Change::SingleGroup(1, GroupChange::PartialUpdate(vec![0, 1]));
        let expected = nonbonded.particle_with_all(&system, 3, &system.groups()[1])
            + nonbonded.particle_with_all(&system, 4, &system.groups()[1]);
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);

        // change several particles in multiple groups
        let change = Change::Groups(vec![
            (0, GroupChange::PartialUpdate(vec![1])),
            (1, GroupChange::PartialUpdate(vec![0, 1])),
        ]);
        let expected = nonbonded.particle_with_all(&system, 3, &system.groups()[1])
            + nonbonded.particle_with_all(&system, 4, &system.groups()[1])
            + nonbonded.particle_with_all(&system, 1, &system.groups()[0]);
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);

        // change several particles in multiple groups, some of which are inactive
        let change = Change::Groups(vec![
            (0, GroupChange::PartialUpdate(vec![1, 2])),
            (1, GroupChange::PartialUpdate(vec![0, 1])),
            (2, GroupChange::PartialUpdate(vec![0])),
        ]);
        assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);
    }
}

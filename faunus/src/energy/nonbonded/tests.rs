use std::{cell::RefCell, sync::Arc};

use float_cmp::assert_approx_eq;
use interatomic::twobody::{IsotropicTwobodyEnergy, SplineConfig};

use crate::{
    backend::Backend,
    cell::{Cell, Cuboid, SimulationCell},
    energy::{builder::HamiltonianBuilder, Hamiltonian},
    group::{GroupCollection, GroupSize},
    montecarlo::NewOld,
    topology::Topology,
    Change, Context, Group, GroupChange,
};

use super::*;

/// Conservative lower bound on group-group distance using bounding spheres.
fn min_group_distance(gi: &Group, gj: &Group, cell: &impl SimulationCell) -> Option<f64> {
    let com_i = gi.mass_center()?;
    let com_j = gj.mass_center()?;
    let ri = gi.bounding_radius()?;
    let rj = gj.bounding_radius()?;
    let com_dist = cell.distance(com_i, com_j).norm();
    Some((com_dist - ri - rj).max(0.0))
}

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
    let file = "tests/files/topology_pass.yaml";
    let topology = Topology::from_file(file).unwrap();
    let pairpot_builder = HamiltonianBuilder::from_file(file)
        .unwrap()
        .pairpot_builder
        .unwrap();
    let medium: Option<interatomic::coulomb::Medium> =
        serde_yml::from_reader(std::fs::File::open(file).unwrap())
            .ok()
            .and_then(|s: serde_yml::Value| {
                let medium = s.get("system")?.get("medium")?;
                serde_yml::from_value(medium.clone()).ok()
            });

    let nonbonded = NonbondedMatrix::new(&pairpot_builder, &topology, medium).unwrap();

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
fn get_test_matrix() -> (Backend, NonbondedMatrix) {
    let file = "tests/files/nonbonded_interactions.yaml";
    let topology = Topology::from_file(file).unwrap();
    let builder = HamiltonianBuilder::from_file(file)
        .unwrap()
        .pairpot_builder
        .unwrap();

    let medium = interatomic::coulomb::Medium::new(
        298.15,
        interatomic::coulomb::permittivity::Permittivity::Vacuum,
        None,
    );

    let nonbonded = NonbondedMatrix::new(&builder, &topology, Some(medium)).unwrap();

    let mut rng = rand::thread_rng();
    let system = Backend::from_raw_parts(
        Arc::new(topology),
        Cell::Cuboid(Cuboid::cubic(20.0)),
        RefCell::new(Hamiltonian::from(vec![nonbonded.clone().into()])),
        None,
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

    let intermolecular_a1a1_energy = [
        401.06633566678175,
        401.06633566678175,
        -0.000090421636081691,
    ];
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
        310.66787793413096,
        491.1915281349669,
        -1.2499998437500003e-6,
        310.66787793413096,
        -6.406572630990959e-6,
    ];
    for ((i, j), energy) in [(0, 5), (0, 8), (3, 8), (5, 6), (2, 3), (2, 6)]
        .into_iter()
        .zip(intermolecular_a1a2_energy)
    {
        assert_part_part(&system, &nonbonded, i, j, energy);
        assert_part_part(&system, &nonbonded, j, i, energy);
    }

    let intermolecular_bb_energy = [-0.713305898491084, -0.713305898491084, -0.01156737611454047];
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

    let intermolecular_a2a2_energy = [
        401.06633566678175,
        401.06633566678175,
        -9.042163608169031e-5,
    ];
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

    let expected = nonbonded.group_with_group(&system, &system.groups()[0], &system.groups()[1]);
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

    let expected = nonbonded.group_with_group(&system, &system.groups()[0], &system.groups()[1])
        + nonbonded.group_with_itself(&system, &system.groups()[0]);
    assert_approx_eq!(
        f64,
        nonbonded.group_with_all(&system, &system.groups()[0]),
        expected
    );

    let expected = nonbonded.group_with_group(&system, &system.groups()[0], &system.groups()[1])
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

    // change multiple rigid groups — cross-term group0↔group1 counted once, not twice
    let change = Change::Groups(vec![
        (0, GroupChange::RigidBody),
        (1, GroupChange::RigidBody),
    ]);
    let expected = nonbonded.group_with_other_groups(&system, &system.groups()[0])
        + nonbonded.group_with_other_groups(&system, &system.groups()[1])
        - nonbonded.group_with_group(&system, &system.groups()[0], &system.groups()[1]);
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

// ====== NonbondedMatrixSplined tests ======

/// Get splined nonbonded matrix for testing.
fn get_test_splined_matrix() -> (Backend, NonbondedMatrix, NonbondedMatrixSplined) {
    let (system, nonbonded) = get_test_matrix();
    let cutoff = 15.0; // Use a cutoff that covers all test distances
    let splined = NonbondedMatrixSplined::from_nonbonded(&nonbonded, cutoff, None);
    (system, nonbonded, splined)
}

#[test]
fn test_nonbonded_matrix_splined_new() {
    let (_, nonbonded, splined) = get_test_splined_matrix();

    // Check that the splined matrix has the same dimensions as the original
    assert_eq!(
        splined.get_potentials().raw_dim(),
        nonbonded.get_potentials().raw_dim()
    );
}

#[test]
fn test_nonbonded_matrix_splined_particle_particle() {
    let (system, nonbonded, splined) = get_test_splined_matrix();

    // Use tolerance since splines are approximations
    let relative_tolerance = 2e-3; // 0.2% relative error for larger values
    let absolute_tolerance = 1e-5; // Absolute tolerance for very small values

    // Test some representative pairs
    let test_pairs = [(0, 1), (0, 3), (1, 4), (3, 4), (0, 5), (1, 5)];

    for (i, j) in test_pairs {
        let analytical = nonbonded.particle_with_particle(&system, i, j);
        let splined_energy = splined.particle_with_particle(&system, i, j);
        let abs_diff = (analytical - splined_energy).abs();

        // For very small energies, check absolute difference
        // For larger energies, check relative difference
        if analytical.abs() < 1e-4 {
            assert!(
                abs_diff < absolute_tolerance,
                "Pair ({}, {}): analytical={}, splined={}, abs_diff={}",
                i,
                j,
                analytical,
                splined_energy,
                abs_diff
            );
        } else {
            let relative_error = abs_diff / analytical.abs();
            assert!(
                relative_error < relative_tolerance,
                "Pair ({}, {}): analytical={}, splined={}, relative_error={}",
                i,
                j,
                analytical,
                splined_energy,
                relative_error
            );
        }
    }
}

#[test]
fn test_nonbonded_matrix_splined_total_nonbonded() {
    let (mut system, nonbonded, splined) = get_test_splined_matrix();

    // Deactivate some particles like in the original test
    system.resize_group(0, GroupSize::Shrink(1)).unwrap();
    system.resize_group(2, GroupSize::Shrink(3)).unwrap();

    let analytical_total = nonbonded.total_nonbonded(&system);
    let splined_total = splined.total_nonbonded(&system);

    // Allow for some tolerance due to spline approximation
    let tolerance = 1e-3;
    let relative_error = ((analytical_total - splined_total) / analytical_total).abs();

    assert!(
        relative_error < tolerance,
        "Total energy: analytical={}, splined={}, relative_error={}",
        analytical_total,
        splined_total,
        relative_error
    );
}

#[test]
fn test_nonbonded_matrix_splined_energy_changes() {
    let (mut system, nonbonded, splined) = get_test_splined_matrix();

    // Deactivate some particles
    system.resize_group(0, GroupSize::Shrink(1)).unwrap();
    system.resize_group(2, GroupSize::Shrink(3)).unwrap();

    let tolerance = 1e-3;

    // Test Change::None
    let change = Change::None;
    assert_approx_eq!(f64, splined.energy(&system, &change), 0.0);

    // Test Change::Everything
    let change = Change::Everything;
    let analytical = nonbonded.energy(&system, &change);
    let splined_energy = splined.energy(&system, &change);
    let relative_error = ((analytical - splined_energy) / analytical).abs();
    assert!(
        relative_error < tolerance,
        "Change::Everything: analytical={}, splined={}, relative_error={}",
        analytical,
        splined_energy,
        relative_error
    );

    // Test single rigid group change
    let change = Change::SingleGroup(1, GroupChange::RigidBody);
    let analytical = nonbonded.energy(&system, &change);
    let splined_energy = splined.energy(&system, &change);
    let relative_error = ((analytical - splined_energy) / analytical).abs();
    assert!(
        relative_error < tolerance,
        "SingleGroup RigidBody: analytical={}, splined={}, relative_error={}",
        analytical,
        splined_energy,
        relative_error
    );

    // Test partial update
    let change = Change::SingleGroup(1, GroupChange::PartialUpdate(vec![0, 1]));
    let analytical = nonbonded.energy(&system, &change);
    let splined_energy = splined.energy(&system, &change);
    let relative_error = ((analytical - splined_energy) / analytical).abs();
    assert!(
        relative_error < tolerance,
        "SingleGroup PartialUpdate: analytical={}, splined={}, relative_error={}",
        analytical,
        splined_energy,
        relative_error
    );
}

#[test]
fn test_min_group_distance() {
    let cell = Cuboid::cubic(20.0);

    let mut g1 = Group::new(0, 0, 0..3);
    g1.set_mass_center(crate::Point::new(0.0, 0.0, 0.0));
    g1.set_bounding_radius(1.0);

    let mut g2 = Group::new(1, 0, 3..6);
    g2.set_mass_center(crate::Point::new(5.0, 0.0, 0.0));
    g2.set_bounding_radius(1.5);

    // COM distance = 5.0, sum of radii = 2.5, min distance = 2.5
    let d = min_group_distance(&g1, &g2, &cell).unwrap();
    assert_approx_eq!(f64, d, 2.5);

    // Overlapping spheres → 0
    let mut g3 = Group::new(2, 0, 6..9);
    g3.set_mass_center(crate::Point::new(1.0, 0.0, 0.0));
    g3.set_bounding_radius(2.0);
    let d = min_group_distance(&g1, &g3, &cell).unwrap();
    assert_approx_eq!(f64, d, 0.0);

    // PBC wrapping: groups near opposite edges
    let mut g4 = Group::new(3, 0, 9..12);
    g4.set_mass_center(crate::Point::new(9.0, 0.0, 0.0));
    g4.set_bounding_radius(0.5);
    // COM distance via PBC = 20 - 9 = 11, but PBC gives min image = 9
    // Actually: g1 at 0, g4 at 9 → distance via PBC in [-10,10] box: 9.0
    let d = min_group_distance(&g1, &g4, &cell).unwrap();
    assert_approx_eq!(f64, d, 9.0 - 1.0 - 0.5);

    // Missing bounding radius → None
    let g5 = Group::new(4, 0, 12..15);
    assert!(min_group_distance(&g1, &g5, &cell).is_none());
}

#[test]
fn test_nonbonded_matrix_splined_with_config() {
    let (system, nonbonded, _) = get_test_splined_matrix();

    // Test with high accuracy config
    let config = SplineConfig::high_accuracy();
    let splined_high = NonbondedMatrixSplined::from_nonbonded(&nonbonded, 15.0, Some(config));

    // Test with fast config
    let config = SplineConfig::fast();
    let splined_fast = NonbondedMatrixSplined::from_nonbonded(&nonbonded, 15.0, Some(config));

    // Both should produce reasonable energies
    let energy_high = splined_high.total_nonbonded(&system);
    let energy_fast = splined_fast.total_nonbonded(&system);
    let analytical = nonbonded.total_nonbonded(&system);

    // High accuracy should be closer to analytical
    let error_high = ((analytical - energy_high) / analytical).abs();
    let error_fast = ((analytical - energy_fast) / analytical).abs();

    // Both should be within reasonable bounds
    assert!(
        error_high < 1e-3,
        "High accuracy error too large: {}",
        error_high
    );
    assert!(
        error_fast < 1e-2,
        "Fast config error too large: {}",
        error_fast
    );

    // High accuracy should generally be better (or at least not significantly worse)
    // Note: this isn't always guaranteed but should hold for most cases
    assert!(
        error_high <= error_fast * 1.1,
        "High accuracy ({}) should be better than fast ({})",
        error_high,
        error_fast
    );
}

/// Verify nonbonded forces against analytical pair forces.
#[test]
fn test_nonbonded_forces() {
    use crate::context::ParticleSystem;
    let (system, nonbonded) = get_test_matrix();

    let forces = nonbonded.forces(&system);

    // Newton's third law: total force on the system must be zero
    let total: crate::Point = forces.iter().sum();
    assert!(
        total.norm() < 1e-8,
        "Total force should be zero (Newton III), got {total:?}"
    );

    // Verify force on particle 0 against sum of analytical pair forces.
    let mut expected_force = crate::Point::zeros();
    for group in system.groups() {
        for j in group.iter_active() {
            if j == 0 {
                continue;
            }
            if nonbonded.exclusions.get((0, j)) == 0 {
                continue;
            }
            let dr = system.get_distance(0, j); // p0 - pj (from j to 0)
            let rsq = dr.norm_squared();
            let potential = nonbonded
                .potentials
                .get((system.atom_kind(0), system.atom_kind(j)))
                .unwrap();
            let f_mag = potential.isotropic_twobody_force(rsq);
            expected_force += dr * (2.0 * f_mag);
        }
    }

    let computed = forces[0];
    let diff = (computed - expected_force).norm();
    assert!(
        diff < 1e-10,
        "Force on particle 0 mismatch: computed={computed:?}, expected={expected_force:?}, diff={diff}"
    );
}

/// Verify what `isotropic_twobody_force(r²)` returns by comparing with
/// numerical derivatives in both r and r² spaces.
#[test]
fn test_force_convention() {
    use interatomic::twobody::LennardJones;
    let lj = LennardJones::new(1.0, 1.0);

    for r in [0.9, 1.0, 1.1, 1.5, 2.0] {
        let rsq = r * r;
        let f = lj.isotropic_twobody_force(rsq);

        // Numerical -dU/dr
        let eps = 1e-7;
        let neg_dudr = -(lj.isotropic_twobody_energy((r + eps).powi(2))
            - lj.isotropic_twobody_energy((r - eps).powi(2)))
            / (2.0 * eps);

        // Numerical -dU/d(r²)
        let neg_dudrsq = -(lj.isotropic_twobody_energy(rsq + eps)
            - lj.isotropic_twobody_energy(rsq - eps))
            / (2.0 * eps);

        // f should match -dU/d(r²), not -dU/dr
        let ratio_rsq = f / neg_dudrsq;
        let ratio_r = f / neg_dudr;

        assert!(
            (ratio_rsq - 1.0).abs() < 1e-4,
            "r={r}: f/(-dU/d(r²))={ratio_rsq} (expected 1.0)"
        );
        // -dU/dr = 2r * (-dU/d(r²)), so f/(-dU/dr) = 1/(2r)
        assert!(
            (ratio_r - 1.0 / (2.0 * r)).abs() < 1e-4,
            "r={r}: f/(-dU/dr)={ratio_r} (expected {})",
            1.0 / (2.0 * r)
        );
    }
}

/// Verify force vectors against energy finite differences.
///
/// p0 = (0,0,0), p1 = (r,0,0). Displace p0 along x and compute F = -∂U/∂x₀.
#[test]
fn test_force_vector_vs_energy_gradient() {
    use interatomic::twobody::LennardJones;
    let lj = LennardJones::new(1.0, 1.0);

    for r in [0.95, 1.0, 1.5, 2.0] {
        let rsq = r * r;
        let f_mag = lj.isotropic_twobody_force(rsq); // = -dU/d(r²)

        // Our formula: F_on_0 = 2 * f_mag * (p0 - p1)
        // p0 - p1 = (0,0,0) - (r,0,0) = (-r, 0, 0)
        let our_force_x = 2.0 * f_mag * (-r);

        // Numerical: F_x = -[U(x₀+ε) - U(x₀-ε)] / (2ε)
        // x₀ = 0, x₁ = r. When x₀ → +ε, distance = r-ε. When x₀ → -ε, distance = r+ε.
        let eps = 1e-7;
        let u_at_x0_plus = lj.isotropic_twobody_energy((r - eps).powi(2)); // closer
        let u_at_x0_minus = lj.isotropic_twobody_energy((r + eps).powi(2)); // farther
        let numerical_force_x = -(u_at_x0_plus - u_at_x0_minus) / (2.0 * eps);

        let rel_err = ((our_force_x - numerical_force_x) / numerical_force_x).abs();
        assert!(
            rel_err < 1e-4,
            "r={r}: our={our_force_x:.8}, numerical={numerical_force_x:.8}, rel_err={rel_err}"
        );
    }
}

/// Verify that splined forces closely match analytical forces.
#[test]
fn test_splined_forces_match_analytical() {
    let (system, nonbonded) = get_test_matrix();
    let splined = NonbondedMatrixSplined::from(&nonbonded);

    let analytical_forces = nonbonded.forces(&system);
    let splined_forces = splined.forces(&system);

    assert_eq!(analytical_forces.len(), splined_forces.len());

    for (i, (a, s)) in analytical_forces
        .iter()
        .zip(splined_forces.iter())
        .enumerate()
    {
        let diff = (a - s).norm();
        let magnitude = a.norm().max(1e-10);
        assert!(
            diff / magnitude < 1e-2,
            "Force mismatch on particle {i}: analytical={a:?}, splined={s:?}, rel_err={}",
            diff / magnitude
        );
    }
}

/// Verify Hamiltonian::forces() dispatches to nonbonded term.
#[test]
fn test_hamiltonian_forces() {
    let (system, nonbonded) = get_test_matrix();
    let hamiltonian = Hamiltonian::from(vec![nonbonded.clone().into()]);

    let term_forces = nonbonded.forces(&system);
    let ham_forces = hamiltonian.forces(&system);

    assert_eq!(term_forces.len(), ham_forces.len());
    for (t, h) in term_forces.iter().zip(ham_forces.iter()) {
        assert!((t - h).norm() < 1e-12);
    }
}

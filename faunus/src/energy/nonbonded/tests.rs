use std::{cell::RefCell, sync::Arc};

use float_cmp::assert_approx_eq;
use interatomic::twobody::{IsotropicTwobodyEnergy, SplineConfig};

use crate::{
    backend::Backend,
    cell::{Cell, Cuboid},
    energy::{builder::HamiltonianBuilder, Hamiltonian},
    group::{GroupCollection, GroupCollectionMut, GroupSize, RelIndex},
    montecarlo::NewOld,
    topology::Topology,
    Change, GroupChange,
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
    let file = "tests/files/topology_pass.yaml";
    let topology = Topology::from_file(file).unwrap();
    let pairpot_builder = HamiltonianBuilder::from_file(file)
        .unwrap()
        .nonbonded()
        .cloned()
        .unwrap();
    let medium: Option<interatomic::coulomb::Medium> =
        yaml_serde::from_reader(std::fs::File::open(file).unwrap())
            .ok()
            .and_then(|s: yaml_serde::Value| {
                let medium = s.get("system")?.get("medium")?;
                yaml_serde::from_value(medium.clone()).ok()
            });

    let nonbonded = NonbondedMatrix::new(&pairpot_builder, &topology, medium).unwrap();

    assert_eq!(nonbonded.potentials.order(), topology.atomkinds().len());

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
    system: &impl ObserveContext,
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
        .nonbonded()
        .cloned()
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
        nonbonded.particle_with_group_reference(&system, 0, &system.groups()[0]),
        expected
    );

    let expected = nonbonded.particle_with_particle(&system, 3, 4)
        + nonbonded.particle_with_particle(&system, 4, 5);
    assert_approx_eq!(
        f64,
        nonbonded.particle_with_group_reference(&system, 4, &system.groups()[1]),
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
        nonbonded.particle_with_group_reference(&system, 1, &system.groups()[1]),
        expected
    );

    let expected = 0.0;
    assert_approx_eq!(
        f64,
        nonbonded.particle_with_group_reference(&system, 0, &system.groups()[2]),
        expected
    );

    let expected = nonbonded.particle_with_particle(&system, 5, 0)
        + nonbonded.particle_with_particle(&system, 5, 1);
    assert_approx_eq!(
        f64,
        nonbonded.particle_with_group_reference(&system, 5, &system.groups()[0]),
        expected
    );
}

#[test]
fn test_nonbonded_matrix_particle_with_other_groups() {
    let (mut system, nonbonded) = get_test_matrix();

    // deactivate particles 2, 6, 7, 8
    system.resize_group(0, GroupSize::Shrink(1)).unwrap();
    system.resize_group(2, GroupSize::Shrink(3)).unwrap();

    let expected = nonbonded.particle_with_group_reference(&system, 0, &system.groups()[1])
        + nonbonded.particle_with_group_reference(&system, 0, &system.groups()[2]);
    assert_approx_eq!(
        f64,
        nonbonded.particle_with_other_groups_reference(&system, 0, &system.groups()[0]),
        expected
    );

    let expected = nonbonded.particle_with_group_reference(&system, 3, &system.groups()[0])
        + nonbonded.particle_with_group_reference(&system, 3, &system.groups()[2]);
    assert_approx_eq!(
        f64,
        nonbonded.particle_with_other_groups_reference(&system, 3, &system.groups()[1]),
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
        nonbonded.particle_with_all_reference(&system, 1, &system.groups()[0]),
        expected
    );
}

#[test]
fn test_nonbonded_matrix_group_with_group() {
    let (mut system, nonbonded) = get_test_matrix();

    // deactivate particles 2, 6, 7, 8
    system.resize_group(0, GroupSize::Shrink(1)).unwrap();
    system.resize_group(2, GroupSize::Shrink(3)).unwrap();

    let expected = nonbonded.particle_with_group_reference(&system, 0, &system.groups()[1])
        + nonbonded.particle_with_group_reference(&system, 1, &system.groups()[1]);

    assert_approx_eq!(
        f64,
        nonbonded.group_with_group_reference(&system, &system.groups()[0], &system.groups()[1]),
        expected
    );
    assert_approx_eq!(
        f64,
        nonbonded.group_with_group_reference(&system, &system.groups()[1], &system.groups()[0]),
        expected
    );

    let expected = 0.0;
    assert_approx_eq!(
        f64,
        nonbonded.group_with_group_reference(&system, &system.groups()[0], &system.groups()[2]),
        expected
    );
    assert_approx_eq!(
        f64,
        nonbonded.group_with_group_reference(&system, &system.groups()[2], &system.groups()[0]),
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
        nonbonded.group_with_itself_reference(&system, &system.groups()[0]),
        expected
    );

    let expected = nonbonded.particle_with_particle(&system, 3, 4)
        + nonbonded.particle_with_particle(&system, 3, 5)
        + nonbonded.particle_with_particle(&system, 4, 5);
    assert_approx_eq!(
        f64,
        nonbonded.group_with_itself_reference(&system, &system.groups()[1]),
        expected
    );

    let expected = 0.0;
    assert_approx_eq!(
        f64,
        nonbonded.group_with_itself_reference(&system, &system.groups()[2]),
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
        nonbonded.group_with_group_reference(&system, &system.groups()[0], &system.groups()[1]);
    assert_approx_eq!(
        f64,
        nonbonded.group_with_other_groups_reference(&system, &system.groups()[0]),
        expected
    );
    assert_approx_eq!(
        f64,
        nonbonded.group_with_other_groups_reference(&system, &system.groups()[1]),
        expected
    );

    let expected = 0.0;
    assert_approx_eq!(
        f64,
        nonbonded.group_with_other_groups_reference(&system, &system.groups()[2]),
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
        nonbonded.group_with_group_reference(&system, &system.groups()[0], &system.groups()[1])
            + nonbonded.group_with_itself_reference(&system, &system.groups()[0]);
    assert_approx_eq!(
        f64,
        nonbonded.group_with_all_reference(&system, &system.groups()[0]),
        expected
    );

    let expected =
        nonbonded.group_with_group_reference(&system, &system.groups()[0], &system.groups()[1])
            + nonbonded.group_with_itself_reference(&system, &system.groups()[1]);
    assert_approx_eq!(
        f64,
        nonbonded.group_with_all_reference(&system, &system.groups()[1]),
        expected
    );

    let expected = 0.0;
    assert_approx_eq!(
        f64,
        nonbonded.group_with_all_reference(&system, &system.groups()[2]),
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
    assert_approx_eq!(f64, nonbonded.total_nonbonded_reference(&system), expected);
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
    let expected = nonbonded.total_nonbonded_reference(&system);
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
    let expected = nonbonded.group_with_other_groups_reference(&system, &system.groups()[1]);
    assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);

    // change multiple rigid groups — cross-term group0↔group1 counted once, not twice
    let change = Change::Groups(vec![
        (0, GroupChange::RigidBody),
        (1, GroupChange::RigidBody),
    ]);
    let expected = nonbonded.group_with_other_groups_reference(&system, &system.groups()[0])
        + nonbonded.group_with_other_groups_reference(&system, &system.groups()[1])
        - nonbonded.group_with_group_reference(&system, &system.groups()[0], &system.groups()[1]);
    assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);

    // change several particles within a single group
    let change = Change::SingleGroup(
        1,
        GroupChange::PartialUpdate(vec![RelIndex::new(0), RelIndex::new(1)]),
    );
    let expected = nonbonded.particle_with_all_reference(&system, 3, &system.groups()[1])
        + nonbonded.particle_with_all_reference(&system, 4, &system.groups()[1]);
    assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);

    // change several particles in multiple groups.
    //
    // Affected atoms: 1 (g0), 3 (g1), 4 (g1). g2 has no active particles in
    // this fixture, so every pair in `total_nonbonded_reference` involves at least one
    // atom in a changed group — `multi_group_change` must therefore
    // return `total_nonbonded_reference` exactly once per pair, with no double-counting
    // of the cross terms (1,3), (1,4), or the intra pair (3,4).
    let change = Change::Groups(vec![
        (0, GroupChange::PartialUpdate(vec![RelIndex::new(1)])),
        (
            1,
            GroupChange::PartialUpdate(vec![RelIndex::new(0), RelIndex::new(1)]),
        ),
    ]);
    let expected = nonbonded.total_nonbonded_reference(&system);
    assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);

    // Same affected atoms via a different relative-index spelling: g0 rel 2
    // and g2 rel 0 both point to inactive particles (silently ignored), and
    // g1 already covered. Result must match the previous case.
    let change = Change::Groups(vec![
        (
            0,
            GroupChange::PartialUpdate(vec![RelIndex::new(1), RelIndex::new(2)]),
        ),
        (
            1,
            GroupChange::PartialUpdate(vec![RelIndex::new(0), RelIndex::new(1)]),
        ),
        (2, GroupChange::PartialUpdate(vec![RelIndex::new(0)])),
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
        splined.get_potentials().order(),
        nonbonded.get_potentials().order()
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

    let analytical_total = nonbonded.total_nonbonded_reference(&system);
    let splined_total = splined.total_nonbonded_reference(&system);

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
    let change = Change::SingleGroup(
        1,
        GroupChange::PartialUpdate(vec![RelIndex::new(0), RelIndex::new(1)]),
    );
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
fn test_nonbonded_matrix_splined_with_config() {
    let (system, nonbonded, _) = get_test_splined_matrix();

    // Test with high accuracy config
    let config = SplineConfig::high_accuracy();
    let splined_high = NonbondedMatrixSplined::from_nonbonded(&nonbonded, 15.0, Some(config));

    // Test with fast config
    let config = SplineConfig::fast();
    let splined_fast = NonbondedMatrixSplined::from_nonbonded(&nonbonded, 15.0, Some(config));

    // Both should produce reasonable energies
    let energy_high = splined_high.total_nonbonded_reference(&system);
    let energy_fast = splined_fast.total_nonbonded_reference(&system);
    let analytical = nonbonded.total_nonbonded_reference(&system);

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
    use crate::context::ObserveContext;
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
                .get((system.atom_kind(0).get(), system.atom_kind(j).get()))
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

// ====== Group-to-group cutoff culling ======

/// Build a small COM-bearing system with bounded Coulomb for culling tests.
fn culling_system() -> (Backend, NonbondedMatrix) {
    let file = "tests/files/nonbonded_culling.yaml";
    let topology = Topology::from_file(file).unwrap();
    let builder = HamiltonianBuilder::from_file(file)
        .unwrap()
        .nonbonded()
        .cloned()
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
        Cell::Cuboid(Cuboid::cubic(60.0)),
        RefCell::new(Hamiltonian::from(vec![nonbonded.clone().into()])),
        None,
        &mut rng,
    )
    .unwrap();

    (system, nonbonded)
}

#[test]
fn cutoff_culling_is_lossless_and_gated() {
    let (system, base) = culling_system();
    assert_approx_eq!(f64, base.required_cull_cutoff(), 6.0);

    // Culling acts in the full-energy path, so evaluate via `energy`.
    let everything = Change::Everything;
    let baseline = base.energy(&system, &everything);

    // A cutoff at the longest pair range only culls pairs that contribute zero.
    let mut safe = base.clone();
    safe.set_cutoff(6.0);
    assert_approx_eq!(f64, safe.energy(&system, &everything), baseline);

    // A too-short cutoff drops genuinely in-range interactions — proves culling
    // is active and that the build-time guard against it is meaningful.
    let mut too_short = base.clone();
    too_short.set_cutoff(2.0);
    assert!((too_short.energy(&system, &everything) - baseline).abs() > 1e-6);

    // bounding_spheres = false makes any cutoff inert.
    let mut disabled = base.clone();
    disabled.set_cutoff(2.0);
    disabled.set_bounding_spheres(false);
    assert_approx_eq!(f64, disabled.energy(&system, &everything), baseline);
}

#[test]
fn cutoff_validation_rejects_too_short() {
    let file = "tests/files/nonbonded_culling.yaml";
    let topology = Topology::from_file(file).unwrap();
    let mut builder = HamiltonianBuilder::from_file(file).unwrap();
    let medium = interatomic::coulomb::Medium::new(
        298.15,
        interatomic::coulomb::permittivity::Permittivity::Vacuum,
        None,
    );

    // Longest pair range is 6 Å; a 3 Å group cutoff would silently drop interactions.
    builder.set_nonbonded_cutoff(Some(3.0));
    assert!(Hamiltonian::new(&builder, &topology, Some(medium.clone())).is_err());

    // Exactly at the required range is accepted.
    builder.set_nonbonded_cutoff(Some(6.0));
    assert!(Hamiltonian::new(&builder, &topology, Some(medium)).is_ok());
}

#[test]
fn cutoff_validation_rejects_unbounded_potential() {
    // Bare Lennard-Jones has infinite range, so no finite group cutoff is safe.
    let file = "tests/files/nonbonded_interactions.yaml";
    let topology = Topology::from_file(file).unwrap();
    let mut builder = HamiltonianBuilder::from_file(file).unwrap();
    let medium = interatomic::coulomb::Medium::new(
        298.15,
        interatomic::coulomb::permittivity::Permittivity::Vacuum,
        None,
    );

    builder.set_nonbonded_cutoff(Some(50.0));
    assert!(Hamiltonian::new(&builder, &topology, Some(medium)).is_err());
}

/// Regression: `multi_group_change` resolved a group's `PartialUpdate` atoms into a fixed
/// `[usize; 8]` buffer, panicking (index out of bounds) for more than eight affected atoms. A
/// `Change::Groups` partial update must match the equivalent `SingleGroup` partial update — which
/// handles any atom count — for a molecule with more than eight atoms.
#[test]
fn multi_group_partial_update_handles_more_than_eight_affected_atoms() {
    use std::io::Write;
    const N: usize = 12; // > 8

    let atom_list = ["A"; N].join(", ");
    let row = |dy: f64| {
        (0..N)
            .map(|i| format!("[{i}.0, {dy}, 0.0]"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let yaml = format!(
        "atoms:\n  - {{name: A, mass: 1.0, sigma: 1.0, eps: 1.0}}\n\
         molecules:\n  - {{name: BIG, atoms: [{atom_list}]}}\n\
         system:\n  cell: !Cuboid [40.0, 40.0, 40.0]\n  \
         medium: {{permittivity: !Vacuum, temperature: 298.15}}\n  \
         energy:\n    - !Nonbonded\n        default:\n          - !LennardJones {{mixing: LB}}\n  \
         blocks:\n    - molecule: BIG\n      N: 2\n      insert: !Manual [{m0}, {m1}]\n",
        m0 = row(0.0),
        m1 = row(3.0),
    );
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(yaml.as_bytes()).unwrap();
    let path = tmp.path();

    let topology = Topology::from_file(path).unwrap();
    let builder = HamiltonianBuilder::from_file(path)
        .unwrap()
        .nonbonded()
        .cloned()
        .unwrap();
    let medium = interatomic::coulomb::Medium::new(
        298.15,
        interatomic::coulomb::permittivity::Permittivity::Vacuum,
        None,
    );
    let nonbonded = NonbondedMatrix::new(&builder, &topology, Some(medium)).unwrap();
    let system = Backend::from_raw_parts(
        Arc::new(topology),
        Cell::Cuboid(Cuboid::cubic(40.0)),
        RefCell::new(Hamiltonian::from(vec![nonbonded.clone().into()])),
        None,
        &mut rand::thread_rng(),
    )
    .unwrap();

    // Both groups fully partial-updated: every pair involves a changed atom, so the deduplicated
    // multi-group sum must equal the total nonbonded energy exactly once per pair (cf. the
    // multi-group test above). With >8 affected atoms per group this exercised the panicking buffer.
    let all: Vec<RelIndex> = (0..N).map(RelIndex::new).collect();
    let change = Change::Groups(vec![
        (0, GroupChange::PartialUpdate(all.clone())),
        (1, GroupChange::PartialUpdate(all)),
    ]);
    let expected = nonbonded.total_nonbonded_reference(&system);
    assert!(
        expected.is_finite() && expected != 0.0,
        "sanity: molecules interact"
    );
    assert_approx_eq!(f64, nonbonded.energy(&system, &change), expected);
}

/// A cubic system from YAML input, with a factory for matrices independent of the system's own.
fn backend_from_yaml(yaml: &str, side: f64) -> (Backend, impl Fn() -> NonbondedMatrix) {
    use std::io::Write;

    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(yaml.as_bytes()).unwrap();
    let path = tmp.path();

    let topology = Arc::new(Topology::from_file(path).unwrap());
    let builder = HamiltonianBuilder::from_file(path)
        .unwrap()
        .nonbonded()
        .cloned()
        .unwrap();
    let new_matrix = {
        let topology = Arc::clone(&topology);
        move || NonbondedMatrix::new(&builder, &topology, None).unwrap()
    };
    let system = Backend::from_raw_parts(
        topology,
        Cell::Cuboid(Cuboid::cubic(side)),
        RefCell::new(Hamiltonian::from(vec![new_matrix().into()])),
        None,
        &mut rand::thread_rng(),
    )
    .unwrap();
    (system, new_matrix)
}

/// A rerun installs each frame with `restore_configuration`, with no backup or move in between. The
/// frame's group energies must not come from the cache built on the previous frame. With two
/// single-atom groups, the group energy equals the total, which is computed without the cache.
#[test]
fn restored_configuration_is_not_served_from_the_previous_cache() {
    use crate::context::{Context, WithHamiltonian};
    use crate::energy::EnergyChange;
    use crate::{Particle, Point};

    let yaml = "atoms:\n  - {name: A, mass: 1.0, sigma: 2.0, eps: 1.0}\n\
                molecules:\n  - {name: M, atoms: [A], degrees_of_freedom: Rigid}\n\
                system:\n  cell: !Cuboid [20.0, 20.0, 20.0]\n  \
                medium: {permittivity: !Vacuum, temperature: 298.15}\n  \
                energy:\n    - !Nonbonded\n        default:\n          - !LennardJones {mixing: LB}\n  \
                blocks:\n    - molecule: M\n      N: 2\n      \
                insert: !Manual [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]\n";
    let (mut system, _) = backend_from_yaml(yaml, 20.0);

    let group_energy = |system: &Backend| {
        system
            .hamiltonian()
            .energy(system, &Change::SingleGroup(0, GroupChange::RigidBody))
    };
    let first_frame = group_energy(&system); // builds the cache

    let kind = system.atom_kind(0).get();
    let particles = [
        Particle::new(kind, Point::new(0.0, 0.0, 0.0)),
        Particle::new(kind, Point::new(5.0, 0.0, 0.0)),
    ];
    let sizes: Vec<_> = system
        .groups()
        .iter()
        .map(|g| GroupSize::from_count(g.len(), g.capacity()))
        .collect();
    let quaternions: Vec<_> = system.groups().iter().map(|g| *g.quaternion()).collect();
    system
        .restore_configuration(None, &particles, &sizes, &quaternions)
        .unwrap();

    let total = system.hamiltonian().energy(&system, &Change::Everything);
    assert!(
        (total - first_frame).abs() > 1e-3,
        "the frames must differ for this test to mean anything"
    );
    assert_approx_eq!(f64, group_energy(&system), total, epsilon = 1e-9);
}

const M_SLOT: usize = 2;
const IONS: usize = 3;

/// Two active rigid molecules, one empty slot and a partly active atomic group in a 12 Å box. The
/// last ion wraps across the x face to reach the molecule at x = 4.
fn resizable_system_yaml(nonbonded_extra: &str) -> String {
    let head = "atoms:\n  - {name: A, mass: 1.0, sigma: 2.0, eps: 1.0}\n  \
                - {name: I, mass: 1.0, sigma: 1.0, eps: 1.0}\n\
                molecules:\n  - {name: M, atoms: [A, A], degrees_of_freedom: Rigid}\n  \
                - {name: ION, atoms: [I], atomic: true}\n\
                system:\n  cell: !Cuboid [12.0, 12.0, 12.0]\n  \
                medium: {permittivity: !Vacuum, temperature: 298.15}\n  \
                energy:\n    - !Nonbonded\n        default:\n          - !LennardJones {mixing: LB}\n";
    let blocks = "  blocks:\n    - molecule: M\n      N: 3\n      active: 2\n      \
                insert: !Manual [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [2.0, 4.0, 0.0], \
                [0.0, -4.0, 0.0], [2.0, -4.0, 0.0]]\n    \
                - molecule: ION\n      N: 4\n      active: 3\n      \
                insert: !Manual [[0.0, 0.0, 3.0], [2.0, 2.0, 3.0], [-5.5, 0.0, 0.0], [1.0, 4.0, 3.0]]\n";
    format!("{head}{nonbonded_extra}{blocks}")
}

/// One trial of each kind of reaction on the [`resizable_system_yaml`] system.
fn speciation_trial(system: &Backend, kind: usize) -> crate::propagate::ProposedMove {
    use crate::propagate::ProposedMove;
    use crate::transform::SpeciationAction;
    use crate::Point;

    let ions = &system.groups()[IONS];
    let first_ion = ions.iter_active().next().unwrap();
    match kind {
        0 => ProposedMove::speciation(
            vec![SpeciationAction::ActivateAtom {
                group_index: IONS,
                position: Point::new(-3.0, 1.0, 2.0),
            }],
            vec![(
                IONS,
                GroupChange::ResizePartial(GroupSize::Expand(1), vec![RelIndex::new(ions.len())]),
            )],
        ),
        1 => ProposedMove::speciation(
            vec![SpeciationAction::DeactivateAtom {
                group_index: IONS,
                abs_index: first_ion,
            }],
            vec![(
                IONS,
                GroupChange::AtomicShrink {
                    rels: vec![RelIndex::new(0)],
                    n_old: ions.len(),
                },
            )],
        ),
        2 => ProposedMove::speciation(
            vec![SpeciationAction::ActivateGroup {
                group_index: M_SLOT,
                positions: vec![Point::new(0.0, -4.0, 0.0), Point::new(2.0, -4.0, 0.0)],
            }],
            vec![(M_SLOT, GroupChange::Resize(GroupSize::Full))],
        ),
        3 => ProposedMove::speciation(
            vec![SpeciationAction::DeactivateGroup(M_SLOT)],
            vec![(M_SLOT, GroupChange::Resize(GroupSize::Empty))],
        ),
        _ => ProposedMove::speciation(
            vec![SpeciationAction::SwapAtomKind {
                group_index: IONS,
                abs_index: first_ion,
                new_atom_id: system.atom_kind(system.groups()[0].iter_active().next().unwrap()),
            }],
            vec![(IONS, GroupChange::UpdateIdentity(vec![RelIndex::new(0)]))],
        ),
    }
}

/// A reaction only marks the rows it names as stale. After each kind, rejected then accepted, every
/// cached group energy must equal a cache built from scratch. Swap-and-pop and activation change the
/// active set, so a row has to be recomputed from whichever set is current when it is read.
#[test]
fn group_energy_cache_survives_speciation() {
    use crate::context::{Context, WithHamiltonian};
    use crate::energy::EnergyChange;

    let (mut system, new_matrix) = backend_from_yaml(&resizable_system_yaml(""), 12.0);

    let rigid = |g| Change::SingleGroup(g, GroupChange::RigidBody);
    let check = |system: &Backend| {
        let fresh = new_matrix();
        for g in 0..=IONS {
            assert_approx_eq!(
                f64,
                system.hamiltonian().energy(system, &rigid(g)),
                fresh.partial_energy(system, &rigid(g)),
                epsilon = 1e-9
            );
        }
    };
    check(&system); // builds the cache

    for kind in 0..5 {
        for accept in [false, true] {
            let proposed = speciation_trial(&system, kind);
            let before = system.hamiltonian().energy(&system, proposed.change());
            system.save_energy_backups(proposed.change());
            proposed.apply_with_backup(&mut system).unwrap();
            system.update(proposed.change()).unwrap();
            let after = system.hamiltonian().energy(&system, proposed.change());
            assert!(
                (after - before).abs() > 1e-6,
                "kind {kind} must change the energy for this test to mean anything"
            );
            if accept {
                system.discard_backup();
            } else {
                system.undo().unwrap();
            }
            check(&system);
        }
    }
}

const SPLINE_5A_CELL_LIST: &str = "        spline: {cutoff: 5.0, cell_list: true}\n";

fn backend_from_input(yaml: &str) -> Backend {
    use std::io::Write;

    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(yaml.as_bytes()).unwrap();
    Backend::new(tmp.path(), None, &mut rand::thread_rng()).unwrap()
}

fn backend_with_spline(yaml: &str) -> (Backend, NonbondedMatrixSplined) {
    let (_, new_matrix) = backend_from_yaml(yaml, 12.0);
    let spline = NonbondedMatrixSplined::from_nonbonded(&new_matrix(), 5.0, None);
    (backend_from_input(yaml), spline)
}

/// Every entry of a swept row against the all-pairs sum, in a box of only two cells per side, where
/// the neighbour map wraps and a pair can only be reached through the periodic boundary.
#[test]
fn swept_row_matches_all_pairs_entry_by_entry() {
    let (system, matrix) = backend_with_spline(&resizable_system_yaml(SPLINE_5A_CELL_LIST));
    assert!(system.cell_list().is_some());

    let pairs = matrix.cached_pairs(&system);
    let mut interacting = 0;
    for m in 0..=IONS {
        for first in [0, m + 1] {
            let mut out = [0.0; IONS + 1];
            pairs.row(m, first, &mut out);
            for j in (first..=IONS).filter(|&j| j != m) {
                let expected = pairs.pair(m, j);
                assert_approx_eq!(f64, out[j], expected, epsilon = 1e-12);
                interacting += usize::from(expected != 0.0);
            }
        }
    }
    assert!(interacting > 0, "the groups must interact");
}

/// Literal energies, so sweep and all-pairs can't agree on a wrong answer. An ion at x = 3 with
/// molecules at x = 0 and 9 and a 5 Å cutoff: only the first pair interacts, and LJ(σ=2, ε=1) at
/// 3 Å is -0.3203, less the energy shift of -0.0163 at the cutoff.
#[test]
fn swept_row_bins_each_pair_by_group_and_stops_at_the_cutoff() {
    let yaml = format!(
        "atoms:\n  - {{name: A, mass: 1.0, sigma: 2.0, eps: 1.0}}\n\
         molecules:\n  - {{name: M, atoms: [A], degrees_of_freedom: Rigid}}\n  \
         - {{name: ION, atoms: [A], atomic: true}}\n\
         system:\n  cell: !Cuboid [30.0, 30.0, 30.0]\n  \
         medium: {{permittivity: !Vacuum, temperature: 298.15}}\n  \
         energy:\n    - !Nonbonded\n        default:\n          - !LennardJones {{mixing: LB}}\n\
         {SPLINE_5A_CELL_LIST}  \
         blocks:\n    - molecule: M\n      N: 2\n      \
         insert: !Manual [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]\n    \
         - molecule: ION\n      N: 1\n      insert: !Manual [[3.0, 0.0, 0.0]]\n"
    );
    let (system, matrix) = backend_with_spline(&yaml);
    assert!(system.cell_list().is_some());
    assert!(system.groups()[2].mass_center().is_none(), "must sweep");

    let mut out = [0.0; 3];
    matrix.cached_pairs(&system).row(2, 0, &mut out);
    assert_approx_eq!(f64, out[0], -0.3203 + 0.0163, epsilon = 1e-3);
    assert_eq!(out[1], 0.0);
}

/// An excluded molecule pair has no energy, and a cell list alone can't tell which group a
/// neighbour belongs to.
#[test]
fn swept_row_applies_molecule_pair_exclusions() {
    let (system, mut matrix) = backend_with_spline(&resizable_system_yaml(SPLINE_5A_CELL_LIST));
    assert!(system.groups()[IONS].mass_center().is_none(), "must sweep");
    let swept = |matrix: &NonbondedMatrixSplined| {
        let mut out = [0.0; IONS + 1];
        matrix.cached_pairs(&system).row(IONS, 0, &mut out);
        out
    };
    assert!(swept(&matrix)[0] != 0.0);

    let ions = system.groups()[IONS].molecule();
    matrix.exclude_molecule_pair(system.groups()[0].molecule(), ions);
    assert_eq!(swept(&matrix)[0], 0.0);
}

/// Two systems differing only by the cell list must read the same cached energies through rigid
/// moves and every kind of reaction, rejected then accepted. Swap-and-pop and activation change
/// both the active set and the neighbour lists a sweep walks.
#[test]
fn cell_list_rows_track_all_pairs_through_trials() {
    use crate::context::{Context, WithHamiltonian};
    use crate::energy::EnergyChange;
    use crate::propagate::ProposedMove;
    use crate::Point;

    let mut swept = backend_from_input(&resizable_system_yaml(SPLINE_5A_CELL_LIST));
    let mut all_pairs = backend_from_input(&resizable_system_yaml(
        "        spline: {cutoff: 5.0, cell_list: false}\n",
    ));
    assert!(swept.cell_list().is_some() && all_pairs.cell_list().is_none());

    let check = |swept: &Backend, all_pairs: &Backend| {
        for g in 0..=IONS {
            let rigid = Change::SingleGroup(g, GroupChange::RigidBody);
            assert_approx_eq!(
                f64,
                swept.hamiltonian().energy(swept, &rigid),
                all_pairs.hamiltonian().energy(all_pairs, &rigid),
                epsilon = 1e-9
            );
        }
    };
    check(&swept, &all_pairs); // builds both caches

    let trial = |system: &mut Backend, proposed: &ProposedMove, accept: bool| {
        system.hamiltonian().energy(system, proposed.change());
        system.save_energy_backups(proposed.change());
        proposed.apply_with_backup(system).unwrap();
        system.update(proposed.change()).unwrap();
        system.hamiltonian().energy(system, proposed.change());
        if accept {
            system.discard_backup();
        } else {
            system.undo().unwrap();
        }
    };

    for kind in 0..7 {
        for accept in [false, true] {
            let proposed = match kind {
                5 => ProposedMove::translate_group(0, Point::new(0.5, 3.0, -1.0)),
                // A full cell width, so a reject has to take the tracked cell-list undo.
                6 => ProposedMove::translate_atoms(
                    IONS,
                    vec![RelIndex::new(0)],
                    Point::new(6.0, 0.0, 0.0),
                ),
                _ => speciation_trial(&swept, kind),
            };
            trial(&mut swept, &proposed, accept);
            trial(&mut all_pairs, &proposed, accept);
            check(&swept, &all_pairs);
        }
    }
}

/// Partial updates only mark their cache row stale, so a run of single-atom moves of an atomic
/// group, accepted or rejected, followed by rigid-body moves must leave every cached group energy
/// equal to a cache built from scratch.
#[test]
fn group_energy_cache_survives_partial_updates_and_rejects() {
    use crate::energy::stateful::StatefulEnergy;
    use crate::Point;
    use rand::{Rng, SeedableRng};

    let yaml = "atoms:\n  - {name: A, mass: 1.0, sigma: 2.0, eps: 1.0}\n  \
                - {name: I, mass: 1.0, sigma: 1.0, eps: 1.0}\n\
                molecules:\n  - {name: M, atoms: [A, A, A], degrees_of_freedom: Rigid}\n  \
                - {name: ION, atoms: [I], atomic: true}\n\
                system:\n  cell: !Cuboid [12.0, 12.0, 12.0]\n  \
                medium: {permittivity: !Vacuum, temperature: 298.15}\n  \
                energy:\n    - !Nonbonded\n        default:\n          - !LennardJones {mixing: LB}\n  \
                blocks:\n    - molecule: M\n      N: 2\n      \
                insert: !Manual [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], \
                [0.0, 4.0, 0.0], [2.0, 4.0, 0.0], [4.0, 4.0, 0.0]]\n    \
                - molecule: ION\n      N: 4\n      \
                insert: !Manual [[0.0, 0.0, 3.0], [2.0, 2.0, 3.0], [4.0, 0.0, -3.0], [1.0, 4.0, 3.0]]\n";
    let (mut system, new_matrix) = backend_from_yaml(yaml, 12.0);

    let mut cached = new_matrix();
    let rigid = |g| Change::SingleGroup(g, GroupChange::RigidBody);
    let check = |cached: &NonbondedMatrix, system: &Backend| {
        let fresh = new_matrix();
        for g in 0..2 {
            assert_approx_eq!(
                f64,
                cached.partial_energy(system, &rigid(g)),
                fresh.partial_energy(system, &rigid(g)),
                epsilon = 1e-9
            );
        }
    };
    cached.partial_energy(&system, &rigid(0)); // builds the cache

    let n_ions = system.groups()[2].iter_active().count();
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    for trial in 0..40 {
        let accept = trial % 3 != 0;
        // Ion moves in a burst, then a rigid-body move, as in a GCMC sweep
        let (change, indices, new): (Change, Vec<usize>, Vec<Point>) = if trial % 4 == 3 {
            let g = &system.groups()[trial % 2];
            let shift = Point::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.5);
            let idx: Vec<usize> = g.iter_active().collect();
            let pos = idx.iter().map(|&i| system.position(i) + shift).collect();
            (rigid(trial % 2), idx, pos)
        } else {
            let rel = rng.gen_range(0..n_ions);
            let i = system.groups()[2].iter_active().nth(rel).unwrap();
            let pos = Point::new(
                rng.gen_range(-6.0..6.0),
                rng.gen_range(-6.0..6.0),
                rng.gen_range(-6.0..6.0),
            );
            let change =
                Change::SingleGroup(2, GroupChange::PartialUpdate(vec![RelIndex::new(rel)]));
            (change, vec![i], vec![pos])
        };
        let old: Vec<Point> = indices.iter().map(|&i| system.position(i)).collect();

        cached.partial_energy(&system, &change);
        cached.save_backup(&system, &change);
        system.set_positions(indices.iter().copied(), new.iter());
        cached.refresh(&system, &change).unwrap();
        cached.partial_energy(&system, &change);
        if accept {
            cached.discard_backup();
        } else {
            system.set_positions(indices.iter().copied(), old.iter());
            cached.undo();
        }
        check(&cached, &system);
    }
}

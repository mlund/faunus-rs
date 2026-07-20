use std::{cell::RefCell, sync::Arc};

use float_cmp::assert_approx_eq;

use crate::{
    backend::Backend,
    cell::{Cell, Cuboid},
    energy::Hamiltonian,
    group::{GroupCollection, GroupCollectionMut, GroupSize, RelIndex},
    montecarlo::NewOld,
    topology::Topology,
    GroupChange,
};

use super::*;
use crate::energy::stateful::StatefulEnergy;

fn make_system() -> (Topology, Backend) {
    let topology = Topology::from_file("tests/files/bonded_interactions.yaml").unwrap();
    let mut rng = rand::thread_rng();
    let system = Backend::from_raw_parts(
        Arc::new(topology.clone()),
        Cell::Cuboid(Cuboid::cubic(20.0)),
        RefCell::new(Hamiltonian::default()),
        None,
        &mut rng,
    )
    .unwrap();
    (topology, system)
}

fn get_intramolecular_bonded() -> (Backend, IntramolecularBonded) {
    let (_topology, system) = make_system();
    (system, IntramolecularBonded::default())
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
    let change = Change::SingleGroup(
        1,
        GroupChange::PartialUpdate(vec![RelIndex::new(1), RelIndex::new(2)]),
    );
    let expected = bonded.one_group(&system, &system.groups()[1]);
    assert_approx_eq!(f64, bonded.energy(&system, &change), expected);

    // change several particles in multiple groups
    let change = Change::Groups(vec![
        (0, GroupChange::PartialUpdate(vec![RelIndex::new(1)])),
        (
            1,
            GroupChange::PartialUpdate(vec![RelIndex::new(0), RelIndex::new(2)]),
        ),
    ]);
    let expected = bonded.multiple_groups(&system, &[0, 1]);
    assert_approx_eq!(f64, bonded.energy(&system, &change), expected);
}

#[test]
fn test_intermolecular_new() {
    let (topology, _) = make_system();
    let EnergyTerm::IntermolecularBonded(intermolecular) = IntermolecularBonded::new(&topology)
    else {
        panic!("IntermolecularBonded not constructed.");
    };

    for i in 0..8 {
        assert!(intermolecular.is_active(i))
    }

    for i in 8..12 {
        assert!(!intermolecular.is_active(i));
    }
}

fn get_intermolecular_bonded() -> (Backend, IntermolecularBonded) {
    let (topology, system) = make_system();
    let EnergyTerm::IntermolecularBonded(bonded) = IntermolecularBonded::new(&topology) else {
        panic!("IntermolecularBonded not constructed.");
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
    bonded.refresh(&system, &change).unwrap();
    assert_eq!(*bonded.particles_status, expected_status);

    // volume change
    let change = Change::Volume(
        crate::cell::VolumeScalePolicy::Isotropic,
        NewOld {
            old: 104.0,
            new: 108.0,
        },
    );
    bonded.refresh(&system, &change).unwrap();
    assert_eq!(*bonded.particles_status, expected_status);

    // irrelevant single group change
    let change = Change::SingleGroup(
        2,
        GroupChange::PartialUpdate(vec![RelIndex::new(0), RelIndex::new(1), RelIndex::new(3)]),
    );
    bonded.refresh(&system, &change).unwrap();
    assert_eq!(*bonded.particles_status, expected_status);

    // irrelevant changes in multiple groups
    let change = Change::Groups(vec![
        (0, GroupChange::PartialUpdate(vec![RelIndex::new(2)])),
        (2, GroupChange::RigidBody),
    ]);
    bonded.refresh(&system, &change).unwrap();
    assert_eq!(*bonded.particles_status, expected_status);

    // resize single group (irrelevant one)
    let change = Change::SingleGroup(0, GroupChange::Resize(GroupSize::Shrink(2)));
    bonded.refresh(&system, &change).unwrap();
    assert_eq!(*bonded.particles_status, expected_status);

    // resize single group (relevant)
    let change = Change::SingleGroup(1, GroupChange::Resize(GroupSize::Shrink(2)));
    bonded.refresh(&system, &change).unwrap();
    let expected_status = [
        true, true, true, true, // first group
        true, true, false, false, // second group
        false, false, false, false, // third group
    ];
    assert_eq!(*bonded.particles_status, expected_status);

    let mut bonded = original_bonded.clone();

    // resize multiple groups
    let change = Change::Groups(vec![
        (0, GroupChange::RigidBody),
        (1, GroupChange::Resize(GroupSize::Shrink(2))),
        (2, GroupChange::Resize(GroupSize::Expand(3))),
    ]);
    bonded.refresh(&system, &change).unwrap();
    let expected_status = [
        true, true, true, true, // first group
        true, true, false, false, // second group
        true, true, true, false, // third group
    ];
    assert_eq!(*bonded.particles_status, expected_status);

    let mut bonded = original_bonded.clone();

    // everything changes
    let change = Change::Everything;
    bonded.refresh(&system, &change).unwrap();
    assert_eq!(*bonded.particles_status, expected_status);
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

    let change = Change::SingleGroup(
        1,
        GroupChange::PartialUpdate(vec![RelIndex::new(0), RelIndex::new(2)]),
    );
    assert_approx_eq!(f64, bonded.energy(&system, &change), expected);

    // resize group
    let resize = GroupSize::Expand(2);
    system.resize_group(2, resize).unwrap();
    let change = Change::SingleGroup(2, GroupChange::Resize(resize));
    bonded.refresh(&system, &change).unwrap();
    let expected = 4362.58996700314;
    assert_approx_eq!(f64, bonded.energy(&system, &change), expected);
}

/// Pins how `IntramolecularBonded` gates on the reported `GroupChange`.
///
/// The term recomputes only when `GroupChange::internal_change()` is true. A move that perturbs a
/// molecule's internal geometry but reports `RigidBody` therefore gets **zero** bonded energy for
/// both the old and the new state, so the bonded contribution to ΔU vanishes silently. Nothing in
/// `ProposedMove` prevents that pairing today; these tests record both branches before the typed
/// constructors make the wrong one unrepresentable.
#[cfg(test)]
mod change_gating_characterization {
    use crate::backend::Backend;
    use crate::energy::bonded::IntramolecularBonded;
    use crate::energy::EnergyChange;
    use crate::group::RelIndex;
    use crate::{Change, GroupChange};

    /// A 4-mer whose last bond is stretched 1 Å beyond `req`, so the bonded energy is non-zero.
    const STRETCHED_CHAIN: &str = r#"
atoms:
  - {name: A, mass: 1.0, charge: 0.0, sigma: 1.0}
molecules:
  - name: POLY
    atoms: [A, A, A, A]
    bonds:
      - {index: [0, 1], kind: !Harmonic {k: 100.0, req: 1.0}}
      - {index: [1, 2], kind: !Harmonic {k: 100.0, req: 1.0}}
      - {index: [2, 3], kind: !Harmonic {k: 100.0, req: 1.0}}
system:
  cell: !Cuboid [30.0, 30.0, 30.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: []
  blocks:
    - molecule: POLY
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, steps: 0, collections: []}
"#;

    fn context() -> Backend {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), STRETCHED_CHAIN).unwrap();
        Backend::new(tmp.path(), None, &mut rand::thread_rng()).unwrap()
    }

    #[test]
    fn partial_update_sees_the_stretched_bond() {
        let context = context();
        let term = IntramolecularBonded::default();
        let energy = term.energy(
            &context,
            &Change::SingleGroup(0, GroupChange::PartialUpdate(vec![RelIndex::new(3)])),
        );
        // One bond stretched by 1 Å: ½·k·Δ² = ½·100·1² = 50 kJ/mol.
        assert!((energy - 50.0).abs() < 1e-9, "bonded energy = {energy}");
    }

    /// The bug: the very same configuration reports **zero** bonded energy when the change is
    /// labelled `RigidBody`, because `internal_change()` is false for it. Both the old and the new
    /// state return 0.0, so the bonded part of ΔU disappears and the drift check cannot see it.
    #[test]
    fn rigid_body_silently_reports_zero_for_the_same_configuration() {
        let context = context();
        let term = IntramolecularBonded::default();
        let energy = term.energy(&context, &Change::SingleGroup(0, GroupChange::RigidBody));
        assert_eq!(energy, 0.0);

        // And that is exactly the label a rigid translate/rotate legitimately uses — which is why
        // the pairing, not the label, has to be constrained.
        assert!(!GroupChange::RigidBody.internal_change());
        assert!(GroupChange::PartialUpdate(vec![RelIndex::new(3)]).internal_change());
    }
}

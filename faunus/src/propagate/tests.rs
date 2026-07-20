#[cfg(feature = "gpu")]
use super::builder::MoveCollectionBuilder;
use super::builder::{CollectionBuilder, Seed};
use super::Propagate;
use crate::backend::Backend;
use crate::montecarlo::AcceptanceCriterion;
use std::path::Path;

/// System thermal energy RT (kJ/mol) at 298.15 K, matching the test inputs.
const THERMAL_ENERGY: f64 = crate::R_IN_KJ_PER_MOL * 298.15;

#[test]
fn seed_parse() {
    let string = "!Fixed 49786352";
    let seed: Seed = yaml_serde::from_str(string).unwrap();
    assert!(matches!(seed, Seed::Fixed(49786352)));

    let string = "Hardware";
    let seed: Seed = yaml_serde::from_str(string).unwrap();
    assert!(matches!(seed, Seed::Hardware));
}

/// `Stochastic`/`Deterministic` is a tag on the wrapping `MoveCollectionBuilder` (see
/// `builder.rs`), not on `CollectionBuilder` itself — that routing is exercised indirectly by
/// `propagate_parse` below, which loads a file containing both tags. This test only pins
/// `CollectionBuilder`'s own field parsing.
#[test]
fn collection_builder_parse() {
    let string = "cycles: 20
moves:
   - !TranslateMolecule { molecule: Water, max_displacement: 0.4, weight: 1.0 }
   - !TranslateMolecule { molecule: Protein, max_displacement: 0.6, weight: 2.0 }
   - !TranslateMolecule { molecule: Lipid, max_displacement: 0.5, weight: 0.5 }";

    let collection: CollectionBuilder = yaml_serde::from_str(string).unwrap();
    assert_eq!(collection.repeat, 20);
    assert_eq!(collection.moves.len(), 3);
}

#[test]
#[cfg(feature = "gpu")]
fn langevin_config_parse() {
    let yaml = "!LangevinDynamics
timestep: 0.002
friction: 1.0
steps: 500
temperature: 300.0";
    let builder: MoveCollectionBuilder = yaml_serde::from_str(yaml).unwrap();
    assert!(matches!(
        builder,
        MoveCollectionBuilder::LangevinDynamics(_)
    ));
    if let MoveCollectionBuilder::LangevinDynamics(config) = builder {
        assert_eq!(config.timestep, 0.002);
        assert_eq!(config.friction, 1.0);
        assert_eq!(config.steps, 500);
    }
}

#[test]
fn propagate_parse() {
    let mut rng = rand::thread_rng();
    let context = Backend::new(
        "tests/files/topology_pass.yaml",
        Some(Path::new("tests/files/structure.xyz")),
        &mut rng,
    )
    .unwrap();
    let propagate =
        Propagate::from_file("tests/files/topology_pass.yaml", &context, THERMAL_ENERGY).unwrap();

    assert_eq!(propagate.max_repeats, 10000);
    assert_eq!(propagate.seed, Seed::Hardware);
    assert_eq!(propagate.current_repeat, 0);
    assert_eq!(propagate.criterion, AcceptanceCriterion::MetropolisHastings);
    assert_eq!(propagate.blocks.len(), 2);

    let stochastic = &propagate.blocks[0];
    assert_eq!(stochastic.moves().len(), 3);
    assert_eq!(stochastic.moves()[0].repeat(), 2);
    assert_eq!(stochastic.moves()[0].weight(), 0.5);
    assert_eq!(stochastic.moves()[1].repeat(), 1);
    assert_eq!(stochastic.moves()[1].weight(), 1.0);

    let deterministic = &propagate.blocks[1];
    assert_eq!(deterministic.repeat(), 5);
    assert_eq!(deterministic.moves().len(), 1);
}

#[test]
fn propagate_parse_fail() {
    let mut rng = rand::thread_rng();
    let context = Backend::new(
        "tests/files/topology_invalid_propagate.yaml",
        Some(Path::new("tests/files/structure.xyz")),
        &mut rng,
    )
    .unwrap();

    assert!(Propagate::from_file(
        "tests/files/topology_invalid_propagate.yaml",
        &context,
        THERMAL_ENERGY
    )
    .is_err());
}

#[test]
fn propagate_translate_atom_parse_fail1() {
    let mut rng = rand::thread_rng();
    let context = Backend::new(
        "tests/files/topology_invalid_translate_atom1.yaml",
        Some(Path::new("tests/files/structure.xyz")),
        &mut rng,
    )
    .unwrap();

    assert!(Propagate::from_file(
        "tests/files/topology_invalid_translate_atom1.yaml",
        &context,
        THERMAL_ENERGY
    )
    .is_err());
}

#[test]
fn propagate_translate_atom_parse_fail2() {
    let mut rng = rand::thread_rng();
    let context = Backend::new(
        "tests/files/topology_invalid_translate_atom2.yaml",
        Some(Path::new("tests/files/structure.xyz")),
        &mut rng,
    )
    .unwrap();

    assert!(Propagate::from_file(
        "tests/files/topology_invalid_translate_atom2.yaml",
        &context,
        THERMAL_ENERGY
    )
    .is_err());
}

#[test]
fn propagate_translate_atom_parse_fail3() {
    let mut rng = rand::thread_rng();
    let context = Backend::new(
        "tests/files/topology_invalid_translate_atom3.yaml",
        Some(Path::new("tests/files/structure.xyz")),
        &mut rng,
    )
    .unwrap();

    assert!(Propagate::from_file(
        "tests/files/topology_invalid_translate_atom3.yaml",
        &context,
        THERMAL_ENERGY
    )
    .is_err());
}

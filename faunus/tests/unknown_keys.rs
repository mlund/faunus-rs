//! Regression guard for issue #64: input parsing must reject unknown/misspelled
//! keys rather than silently ignoring them.
use faunus::Simulation;

/// A complete, self-contained input. Each test injects one bogus key into a
/// copy of this and asserts the parse fails.
const BASE: &str = r#"
atoms:
  - name: OW
    mass: 15.999
    charge: -0.8476
    sigma: 3.166
    epsilon: 0.650
  - name: HW
    mass: 1.007
    charge: 0.4238
    sigma: 2.0
    epsilon: 0.0

molecules:
  - name: water
    from_structure:
      - OW: [2.30, 6.28, 1.13]
      - HW: [1.37, 6.26, 1.50]
      - HW: [2.31, 5.89, 0.21]
    exclusions: [[0, 1], [0, 2], [1, 2]]
    has_com: true

system:
  cell: !Cuboid [18.6, 18.6, 18.6]
  medium:
    temperature: 300
    permittivity: !Vacuum
  blocks:
    - molecule: water
      N: 128
      insert:
        !RandomCOM { rotate: true }
  energy:
    nonbonded:
      default:
        - !LennardJones {mixing: LB}
        - !Fanourgakis {cutoff: 9.0}
    spline:
      cutoff: 14.0
      shift_energy: false

analysis:
  - !RadialDistribution
    selections: ["atomtype OW", "atomtype OW"]
    file: rdf.dat
    dr: 0.1
    frequency: !Every 10

propagate:
  seed: !Fixed 42
  criterion: Metropolis
  repeat: 2000
  collections:
    - !Deterministic
      moves:
        - !TranslateMolecule {molecule: water, dp: 0.4, repeat: 64}
"#;

#[test]
fn base_input_parses() {
    Simulation::from_yaml(BASE, None).expect("BASE input must parse cleanly");
}

/// Replace `find` (first occurrence) with `repl` and assert the resulting input
/// is rejected — i.e. the injected `bogus_zz` key is not silently swallowed.
fn assert_rejected(section: &str, find: &str, repl: &str) {
    let yaml = BASE.replacen(find, repl, 1);
    assert_ne!(yaml, BASE, "probe for `{section}` did not modify the input");
    assert!(
        Simulation::from_yaml(&yaml, None).is_err(),
        "unknown key in `{section}` was silently accepted",
    );
}

#[test]
fn rejects_unknown_energy_key() {
    assert_rejected(
        "energy",
        "energy:\n    nonbonded",
        "energy:\n    bogus_zz: 1\n    nonbonded",
    );
}

#[test]
fn rejects_unknown_nonbonded_key() {
    assert_rejected(
        "nonbonded",
        "nonbonded:\n      default",
        "nonbonded:\n      bogus_zz: 1\n      default",
    );
}

#[test]
fn rejects_unknown_spline_key() {
    assert_rejected("spline", "cutoff: 14.0", "cutoff: 14.0\n      bogus_zz: 1");
}

#[test]
fn rejects_unknown_pair_potential_mixing_key() {
    // The mixing form of a pair potential (untagged `DirectOrMixing`) used to
    // swallow extra keys; the direct form always rejected them.
    assert_rejected(
        "LennardJones",
        "!LennardJones {mixing: LB}",
        "!LennardJones {mixing: LB, bogus_zz: 1}",
    );
}

#[test]
fn rejects_unknown_insert_policy_key() {
    assert_rejected(
        "insert",
        "!RandomCOM { rotate: true }",
        "!RandomCOM { rotate: true, bogus_zz: 1 }",
    );
}

#[test]
fn rejects_unknown_system_key() {
    assert_rejected(
        "system",
        "system:\n  cell",
        "system:\n  bogus_zz: 1\n  cell",
    );
}

#[test]
fn rejects_unknown_toplevel_key() {
    assert_rejected("root", "atoms:", "bogus_zz: 1\natoms:");
}

#[test]
fn misspelled_section_names_the_key_and_suggests_a_fix() {
    let yaml = BASE.replacen("analysis:", "analysiss:", 1);
    let err = Simulation::from_yaml(&yaml, None)
        .expect_err("a misspelled section must not be silently ignored")
        .to_string();
    assert!(
        err.contains("analysiss"),
        "error should name the typo: {err}"
    );
    assert!(
        err.contains("analysis"),
        "error should suggest the fix: {err}"
    );
}

#[test]
fn rejects_non_string_section_key() {
    // A non-string root key can never name a valid section and must be rejected,
    // not silently skipped.
    let yaml = format!("{BASE}\n1234: oops\n");
    let err = Simulation::from_yaml(&yaml, None)
        .expect_err("a non-string root key must be rejected")
        .to_string();
    assert!(err.contains("non-string key"), "unexpected error: {err}");
}

#[test]
fn rejects_unknown_key_in_included_file() {
    // Included files are parsed piecemeal too; a misspelled section there must
    // not vanish silently.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("lib.yaml"),
        "atoms:\n  - {name: OW, mass: 1.0, sigma: 1.0}\nmoleculess: []\n",
    )
    .unwrap();
    // Main input takes its atoms from the include, so it defines none itself.
    let main = format!(
        "include: [lib.yaml]\n{}",
        BASE.split_once("molecules:")
            .map(|(_, rest)| format!("molecules:{rest}"))
            .unwrap(),
    );
    let path = dir.path().join("input.yaml");
    std::fs::write(&path, main).unwrap();
    let err = Simulation::from_file(&path, None)
        .expect_err("misspelled section in an include must be rejected")
        .to_string();
    assert!(err.contains("moleculess"), "should name the typo: {err}");
    assert!(
        err.contains("lib.yaml"),
        "should name the include file: {err}"
    );
}

#[test]
fn disabled_underscore_section_is_allowed() {
    // `_`-prefixed keys intentionally disable a section and must not be rejected
    // by the whole-document key check.
    let yaml = format!("{BASE}\n_disabled_extra:\n  foo: 1\n");
    Simulation::from_yaml(&yaml, None).expect("`_`-prefixed key must be allowed");
}

//! Guards for #122: dimensional/chemical input keys that are hard to type must
//! have an ASCII spelling, and every unicode spelling must remain accepted.
//!
//! Both fixes live in the `coulomb` crate, but the surface they affect is
//! faunus YAML input, so the regression guard lives here — `coulomb`'s own CI
//! runs `cargo test` with default features, which exclude `serde`.
use faunus::Simulation;

/// Complete, self-contained input. Tests substitute the salt tag or the Ewald
/// alpha key into a copy of this.
const BASE: &str = r#"
atoms:
  - name: Na
    mass: 22.99
    charge: 1.0
    sigma: 3.3
    epsilon: 0.5

molecules:
  - name: ion
    atoms: [Na]

system:
  cell: !Cuboid [30.0, 30.0, 30.0]
  medium:
    temperature: 300
    permittivity: !Water
    salt: [!NaCl, 0.1]
  blocks:
    - molecule: ion
      N: 8
      insert: !RandomAtomPos {}
  energy:
    - !Nonbonded
      default:
        - !LennardJones {mixing: LB}
        - !CoulombEwald {alpha: 0.1, cutoff: 11.0}

analysis:
  - !Energy
    file: energy.dat
    frequency: !Every 10

propagate:
  seed: !Fixed 42
  steps: 1
  collections:
    - !Deterministic
      moves:
        - !TranslateAtom {molecule: ion, max_displacement: 0.1, repeat: 1}
"#;

fn parses(name: &str, find: &str, repl: &str) {
    let yaml = BASE.replacen(find, repl, 1);
    assert_ne!(yaml, BASE, "{name}: probe did not modify the input");
    if let Err(e) = Simulation::from_yaml(&yaml, None) {
        panic!("{name}: must parse, but failed: {e}");
    }
}

#[test]
fn base_input_parses() {
    Simulation::from_yaml(BASE, None).expect("BASE input must parse cleanly");
}

// --- Salt: ASCII spellings must be accepted (#122) ---------------------------

/// Every salt must be reachable as a YAML tag. Only ASCII can be: a YAML tag
/// shorthand is restricted to URI characters, so a subscript makes the document
/// unscannable (`!CaCl₂` fails in the parser, before serde is ever reached).
/// This is why the unicode spellings could not be used at all.
#[test]
fn salt_tag_accepts_ascii_spelling() {
    for ascii in ["CaCl2", "CaSO4", "Na2SO4", "KAl(SO4)2", "LaCl3"] {
        parses(
            &format!("salt !{ascii}"),
            "salt: [!NaCl, 0.1]",
            &format!("salt: [!{ascii}, 0.1]"),
        );
    }
}

/// The unicode spelling stays accepted via the untagged (bare string) form,
/// which the scanner does allow.
#[test]
fn salt_bare_string_accepts_unicode_alias() {
    for unicode in ["CaCl₂", "CaSO₄", "Na₂SO₄", "KAl(SO₄)₂", "LaCl₃"] {
        parses(
            &format!("salt {unicode}"),
            "salt: [!NaCl, 0.1]",
            &format!("salt: ['{unicode}', 0.1]"),
        );
    }
}

/// ...and so does the ASCII spelling in that same form.
#[test]
fn salt_bare_string_accepts_ascii() {
    for ascii in ["CaCl2", "CaSO4", "Na2SO4", "KAl(SO4)2", "LaCl3"] {
        parses(
            &format!("salt {ascii}"),
            "salt: [!NaCl, 0.1]",
            &format!("salt: ['{ascii}', 0.1]"),
        );
    }
}

// --- EwaldTruncated: the `α` alias must actually work (#122) ------------------

#[test]
fn ewald_accepts_unicode_alpha_alias() {
    parses(
        "!CoulombEwald α",
        "!CoulombEwald {alpha: 0.1, cutoff: 11.0}",
        "!CoulombEwald {α: 0.1, cutoff: 11.0}",
    );
}

#[test]
fn ewald_accepts_ascii_alpha() {
    Simulation::from_yaml(BASE, None).expect("`alpha` must remain the canonical key");
}

/// The alias must not weaken strictness: an unknown key is still an error.
#[test]
fn ewald_still_rejects_unknown_key() {
    let yaml = BASE.replacen(
        "!CoulombEwald {alpha: 0.1, cutoff: 11.0}",
        "!CoulombEwald {alpha: 0.1, cutoff: 11.0, bogus_zz: 1}",
        1,
    );
    assert!(
        Simulation::from_yaml(&yaml, None).is_err(),
        "unknown key in !CoulombEwald was silently accepted",
    );
}

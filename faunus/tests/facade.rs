//! The public simulation interface, driven the way a downstream crate would drive it.
//!
//! Integration tests compile as a separate crate, so nothing `pub(crate)` is reachable here. If
//! these pass, `faunus::Simulation` is a usable door into the engine and not just a compiling one.
//!
//! `gcmc_ideal_gas` is the fixture throughout: it is deterministic (`seed: !Fixed 42`),
//! self-contained (no `include:`, no external structure), and none of its analyses write files —
//! so these tests neither race with the other test binaries over the working directory nor
//! leave artefacts behind.

// shared test helpers; only the YAML comparator is used here
#[allow(dead_code)]
mod common;

use std::path::Path;

use faunus::{Error, Simulation};
use serde_yml::Value;

/// Wall-clock measurements and the git revision differ between any two runs of the same input.
const VOLATILE_KEYS: &[&str] = &["timer", "elapsed_seconds", "energy_timers", "version"];

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/files/{name}"))
}

fn parse(yaml: &str) -> Value {
    serde_yml::from_str(yaml).expect("output is valid YAML")
}

/// Running from a checkpoint through `Simulation` reproduces what the CLI wrote for the same input.
#[test]
fn from_file_reproduces_the_reference_output() {
    let dir = fixture("gcmc_ideal_gas");
    let temp = tempfile::tempdir().expect("temp dir");
    let state = temp.path().join("state.yaml");
    std::fs::copy(dir.join("state.yaml"), &state).expect("copy state");

    let mut simulation =
        Simulation::from_file(&dir.join("input.yaml"), Some(&state)).expect("load simulation");
    let output = simulation.run().expect("run simulation");

    let reference = std::fs::read_to_string(dir.join("reference_output.yaml")).expect("reference");
    common::assert_yaml_eq(
        &parse(&reference),
        &parse(output.to_yaml()),
        1e-8,
        VOLATILE_KEYS,
    );

    // A single box, and its document is the whole result.
    assert_eq!(output.boxes().len(), 1);
    assert_eq!(output.boxes()[0].to_yaml(), output.to_yaml());
}

/// The accessors agree with the `energy_change` block they are derived from.
#[test]
fn box_result_accessors_match_the_yaml() {
    let dir = fixture("gcmc_ideal_gas");
    let mut simulation =
        Simulation::from_file(&dir.join("input.yaml"), None).expect("load simulation");
    let output = simulation.run().expect("run simulation");

    let result = &output.boxes()[0];
    let change = &parse(result.to_yaml())["energy_change"];

    assert_eq!(change["initial"].as_f64(), Some(result.initial_energy()));
    assert_eq!(change["final"].as_f64(), Some(result.final_energy()));
    assert_eq!(change["drift"].as_f64(), Some(result.drift()));
}

/// The string path builds the same system as the file path: identical seed, identical trajectory.
///
/// Compared without a tolerance on the move statistics, so a divergence of even one accepted
/// move fails here rather than drifting silently.
#[test]
fn from_yaml_matches_from_file() {
    let input = fixture("gcmc_ideal_gas").join("input.yaml");
    let yaml = std::fs::read_to_string(&input).expect("read input");

    let from_file = Simulation::from_file(&input, None)
        .expect("load from file")
        .run()
        .expect("run from file");
    let from_yaml = Simulation::from_yaml(&yaml, None)
        .expect("load from yaml")
        .run()
        .expect("run from yaml");

    common::assert_yaml_eq(
        &parse(from_file.to_yaml()),
        &parse(from_yaml.to_yaml()),
        0.0,
        VOLATILE_KEYS,
    );
}

/// `write_to` lays down exactly the file the CLI's `-o` produces.
#[test]
fn write_to_writes_the_summary_document() {
    let temp = tempfile::tempdir().expect("temp dir");
    let output_path = temp.path().join("output.yaml");

    let mut simulation = Simulation::from_file(&fixture("gcmc_ideal_gas").join("input.yaml"), None)
        .expect("load simulation");
    let output = simulation.run().expect("run simulation");
    output.write_to(&output_path).expect("write output");

    let written = std::fs::read_to_string(&output_path).expect("read output");
    assert_eq!(written, output.to_yaml());
    // A single box writes one file, with no `box0_` sibling.
    assert!(!temp.path().join("box0_output.yaml").exists());
}

/// Gibbs needs a directory to anchor its per-box outputs, so the string path must refuse it
/// rather than silently running a single box.
#[test]
fn from_yaml_rejects_a_gibbs_section() {
    let yaml =
        std::fs::read_to_string(fixture("gibbs_ensemble").join("input.yaml")).expect("read input");

    match Simulation::from_yaml(&yaml, None) {
        Err(Error::Unsupported(message)) => assert!(
            message.contains("gibbs"),
            "message should name the offending section: {message}"
        ),
        Err(other) => panic!("expected Error::Unsupported, got {other:?}"),
        Ok(_) => panic!("expected a `propagate.gibbs` section to be rejected"),
    }
}

/// `Error::Run` keeps the failure underneath as a source, rather than flattening it to a string,
/// so a caller can walk the chain instead of grepping a message.
#[test]
fn run_errors_expose_their_source() {
    use std::error::Error as _;

    let run = Error::Run("underlying cause".into());
    assert_eq!(run.to_string(), "simulation failed");
    assert_eq!(
        run.source().map(ToString::to_string).as_deref(),
        Some("underlying cause")
    );
}

/// A missing input is an `Error::Io` naming the path, not an opaque parse failure.
#[test]
fn missing_input_names_the_file() {
    let missing = Path::new("definitely/not/here.yaml");
    match Simulation::from_file(missing, None) {
        Err(Error::Io { path, source }) => {
            assert_eq!(path, missing);
            assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
        }
        other => panic!("expected Error::Io, got {other:?}"),
    }
}

/// `run_with_progress` reports monotonically increasing steps against a fixed total.
#[test]
fn progress_reports_every_step() {
    let mut simulation = Simulation::from_file(&fixture("gcmc_ideal_gas").join("input.yaml"), None)
        .expect("load simulation");

    let mut steps = Vec::new();
    let mut totals = Vec::new();
    simulation
        .run_with_progress(&mut |step, total| {
            steps.push(step);
            totals.push(total);
        })
        .expect("run simulation");

    let total = *totals.first().expect("at least one callback");
    assert!(totals.iter().all(|&t| t == total), "total must not change");
    assert_eq!(steps, (1..=total).collect::<Vec<_>>());
}

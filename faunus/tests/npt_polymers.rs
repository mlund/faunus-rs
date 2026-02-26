//! Regression test for NPT polymer simulation.
//!
//! Runs the `faunus` CLI binary from a pre-equilibrated state with a fixed
//! seed, then compares the generated output against a committed reference.
//! If any code change alters MC behavior, the output will differ and the test
//! fails.
//!
//! # Usage
//! ```sh
//! # Generate fixtures (one-time, or after intentional changes):
//! cargo test generate_npt_polymers_fixtures -- --ignored
//!
//! # Run the regression test:
//! cargo test npt_polymers_regression -- --ignored
//! ```

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

/// Directory containing the test input files.
fn test_files_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("files")
        .join("npt_polymers")
}

/// Path to the compiled `faunus` binary.
fn faunus_binary() -> PathBuf {
    // `cargo test` puts the binary in the same target directory
    let mut path = PathBuf::from(env!("CARGO_BIN_EXE_faunus"));
    // Fallback: resolve via CARGO_BIN_EXE (set by cargo for integration tests)
    if !path.exists() {
        path = PathBuf::from("target/debug/faunus");
    }
    path
}

/// Run faunus with the given arguments and assert success.
fn run_faunus(input: &Path, state: &Path, output: &Path) {
    let status = Command::new(faunus_binary())
        .arg("-o")
        .arg(output)
        .arg("run")
        .arg("-i")
        .arg(input)
        .arg("-s")
        .arg(state)
        .status()
        .expect("failed to execute faunus binary");
    assert!(status.success(), "faunus exited with status: {status}");
}

/// Generate the fixture files (state.yaml and reference_output.yaml).
///
/// This runs a short simulation without a pre-existing state to produce
/// the state file, then runs again *from* that state to produce the
/// deterministic reference output.
///
/// Run with: `cargo test generate_npt_polymers_fixtures -- --ignored`
#[test]
#[ignore]
fn generate_npt_polymers_fixtures() {
    let dir = test_files_dir();
    let input = dir.join("input.yaml");
    let state = dir.join("state.yaml");
    let reference_output = dir.join("reference_output.yaml");

    // Step 1: Generate state.yaml by running from scratch
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let tmp_output = tmp.path().join("output.yaml");
    run_faunus(&input, &state, &tmp_output);

    assert!(state.exists(), "state.yaml was not created");
    println!("Generated {}", state.display());

    // Step 2: Generate reference_output.yaml by running from the state
    let tmp2 = tempfile::tempdir().expect("failed to create temp dir");
    let tmp_state = tmp2.path().join("state.yaml");
    std::fs::copy(&state, &tmp_state).expect("failed to copy state.yaml");
    run_faunus(&input, &tmp_state, &reference_output);

    assert!(
        reference_output.exists(),
        "reference_output.yaml was not created"
    );
    println!("Generated {}", reference_output.display());
}

/// Regression test: run from pre-equilibrated state and compare output.
///
/// Run with: `cargo test npt_polymers_regression -- --ignored`
#[test]
#[ignore]
fn npt_polymers_regression() {
    let dir = test_files_dir();
    let input = dir.join("input.yaml");
    let state_source = dir.join("state.yaml");
    let reference_output = dir.join("reference_output.yaml");

    assert!(
        state_source.exists(),
        "state.yaml not found. Run `cargo test generate_npt_polymers_fixtures -- --ignored` first."
    );
    assert!(
        reference_output.exists(),
        "reference_output.yaml not found. Run `cargo test generate_npt_polymers_fixtures -- --ignored` first."
    );

    // Copy state to a temp dir so the CLI can write back to it without modifying the fixture
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let tmp_state = tmp.path().join("state.yaml");
    let tmp_output = tmp.path().join("output.yaml");
    std::fs::copy(&state_source, &tmp_state).expect("failed to copy state.yaml");

    run_faunus(&input, &tmp_state, &tmp_output);

    // Parse both YAML files
    let actual_yaml = std::fs::read_to_string(&tmp_output).expect("failed to read output.yaml");
    let actual: serde_yaml::Value =
        serde_yaml::from_str(&actual_yaml).expect("failed to parse output.yaml");

    let reference_yaml =
        std::fs::read_to_string(&reference_output).expect("failed to read reference_output.yaml");
    let reference: serde_yaml::Value =
        serde_yaml::from_str(&reference_yaml).expect("failed to parse reference_output.yaml");

    // Compare entire YAML output, ignoring non-deterministic keys
    common::assert_yaml_eq(&reference, &actual, 1e-10, &["timer", "num_accepted"]);
}

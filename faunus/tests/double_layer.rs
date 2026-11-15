//! Osmotic pressure between charged planes against Monte Carlo literature values.
//!
//! Reference: Guldbrand, Jönsson, Wennerström & Linse, J. Chem. Phys. 80, 2221 (1984),
//! doi:10.1063/1.446912. Table I, σ = 0.2244 C/m², 2a = 21 Å, P_osm/RT from Eq. 5.

// shared test helpers; only `faunus_binary` is used here
#[allow(dead_code)]
mod common;

use std::path::Path;
use std::process::Command;

/// Runs `examples/double_layer/input.yaml` rather than a test copy, so the example can't drift
/// from what is validated. Overrides go in front of the template, whose `| default(...)` keeps
/// an already-set variable.
fn osmotic_pressure(overrides: &str) -> (f64, f64) {
    let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/double_layer/input.yaml");
    let template = std::fs::read_to_string(example).expect("failed to read double-layer example");
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    std::fs::write(
        tmp.path().join("input.yaml"),
        format!("{overrides}\n{template}"),
    )
    .expect("failed to write input");

    // Analyses write their files relative to the working directory: keep them in the tempdir.
    // The first run equilibrates into state.yaml; the second samples from it.
    for _ in 0..2 {
        let status = Command::new(common::faunus_binary())
            .current_dir(tmp.path())
            .args(["run", "-i", "input.yaml", "-s", "state.yaml"])
            .status()
            .expect("failed to execute faunus binary");
        assert!(status.success(), "faunus exited with status: {status}");
    }

    let output: yaml_serde::Value = yaml_serde::from_str(
        &std::fs::read_to_string(tmp.path().join("output.yaml")).expect("missing output.yaml"),
    )
    .expect("invalid output.yaml");
    let pressure = output["analysis"]
        .as_sequence()
        .expect("analysis should be a sequence")
        .iter()
        .find_map(|entry| entry.get("double_layer_pressure"))
        .expect("no double_layer_pressure in output")["p_osm/mM"]
        .clone();
    let value = |key: &str| {
        pressure[key]
            .as_f64()
            .expect("p_osm should be {mean, error}")
    };
    (value("mean"), value("error"))
}

/// Within two combined standard errors, since both values are themselves MC estimates.
fn assert_matches_guldbrand(overrides: &str, reference: f64, reference_error: f64) {
    let (mean, error) = osmotic_pressure(overrides);
    let tolerance = 2.0 * reference_error.hypot(error);
    assert!(
        (mean - reference).abs() < tolerance,
        "P_osm/RT = {mean:.1} ± {error:.1} mM, Guldbrand: {reference} ± {reference_error} mM"
    );
}

/// Monovalent counterions repel the planes, as Poisson–Boltzmann predicts.
#[test]
#[ignore]
fn regression_monovalent() {
    assert_matches_guldbrand("{% set dG = 50.0 %}", 388.0, 40.0);
}

/// Divalent ion correlations cancel the repulsion, which Poisson–Boltzmann misses entirely.
#[test]
#[ignore]
fn regression_divalent() {
    assert_matches_guldbrand("{% set dG = -50.0 %}", 2.0, 9.0);
}

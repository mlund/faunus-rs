//! Gibbs ensemble regression test for LJ phase coexistence.
//!
//! Reference: Panagiotopoulos, Mol. Phys. 61, 813 (1987), doi:10/cvzgw9
//! Table 1, T*=1.25, N=300: rho*(gas)=0.148, rho*(liquid)=0.526

// shared test helpers; only `run_faunus` is used here
#[allow(dead_code)]
mod common;

use std::path::Path;

const RHO_GAS: f64 = 0.148;
const RHO_LIQUID: f64 = 0.526;

/// Extract cell volume from per-box output YAML (!Cuboid [Lx, Ly, Lz]).
fn cell_volume(yaml: &serde_yml::Value) -> f64 {
    let cell = &yaml["cell"];
    let dims = match cell {
        serde_yml::Value::Tagged(t) => &t.value,
        other => other,
    };
    dims.as_sequence()
        .expect("cell should be a sequence")
        .iter()
        .map(|v| v.as_f64().expect("cell dim should be float"))
        .product()
}

/// Count active groups (Full-sized) in a state file by counting `size: Full` lines.
fn count_active_groups(state_path: &Path) -> usize {
    std::fs::read_to_string(state_path)
        .expect("read state file")
        .lines()
        .filter(|line| line.trim() == "size: Full")
        .count()
}

/// Count active atoms in atomic mega-groups from a state file.
///
/// Atomic groups use `size: !Partial N` where N is the active atom count.
/// In a Gibbs ensemble, mega-groups are never completely Full (particles
/// are split between two boxes), so only Partial sizes need counting.
fn count_active_atoms(state_path: &Path) -> usize {
    std::fs::read_to_string(state_path)
        .expect("read state file")
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix("size: !Partial ")
                .and_then(|rest| rest.trim().parse::<usize>().ok())
        })
        .sum()
}

/// Run a Gibbs ensemble test and check conservation laws and phase separation.
fn run_gibbs_test(test_dir: &Path, atomic: bool) {
    let input = test_dir.join("input.yaml");
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let output = tmp.path().join("output.yaml");
    let state = tmp.path().join("state.yaml");

    // copy pre-equilibrated per-box state files into temp dir
    for name in ["box0_state.yaml", "box1_state.yaml"] {
        let src = test_dir.join(name);
        if src.exists() {
            std::fs::copy(&src, tmp.path().join(name)).expect("copy state file");
        }
    }

    common::run_faunus(&input, &state, &output);

    // --- per-box output files ---
    let box0_yaml: serde_yml::Value = serde_yml::from_str(
        &std::fs::read_to_string(tmp.path().join("box0_output.yaml")).expect("read box0"),
    )
    .expect("parse box0");
    let box1_yaml: serde_yml::Value = serde_yml::from_str(
        &std::fs::read_to_string(tmp.path().join("box1_output.yaml")).expect("read box1"),
    )
    .expect("parse box1");

    // --- volumes ---
    let v0 = cell_volume(&box0_yaml);
    let v1 = cell_volume(&box1_yaml);
    let v_total = v0 + v1;
    let initial_volume = 2.0 * 10.0_f64.powi(3);

    assert!(
        (v_total - initial_volume).abs() < 1.0,
        "Total volume not conserved: {v_total:.1} vs expected {initial_volume:.1}"
    );

    // --- particle counts from state files ---
    let (n0, n1) = if atomic {
        (
            count_active_atoms(&tmp.path().join("box0_state.yaml")),
            count_active_atoms(&tmp.path().join("box1_state.yaml")),
        )
    } else {
        (
            count_active_groups(&tmp.path().join("box0_state.yaml")),
            count_active_groups(&tmp.path().join("box1_state.yaml")),
        )
    };
    let n_total = n0 + n1;

    assert_eq!(n_total, 600, "Total particles not conserved: {n_total}");
    assert!(n0 > 0 && n1 > 0, "Both boxes must have particles");

    // --- reduced densities vs literature ---
    let rho0 = n0 as f64 / v0;
    let rho1 = n1 as f64 / v1;
    let (rho_low, rho_high) = if rho0 < rho1 {
        (rho0, rho1)
    } else {
        (rho1, rho0)
    };

    println!("Volume: box0={v0:.1}, box1={v1:.1}, total={v_total:.1}");
    println!("Particles: box0={n0}, box1={n1}, total={n_total}");
    println!(
        "Reduced density: gas={rho_low:.3} (ref {RHO_GAS}), liquid={rho_high:.3} (ref {RHO_LIQUID})"
    );

    let gas_tol = 0.15;
    let liquid_tol = 0.20;
    assert!(
        (rho_low - RHO_GAS).abs() < gas_tol,
        "Gas density {rho_low:.3} too far from reference {RHO_GAS} (tol={gas_tol})"
    );
    assert!(
        (rho_high - RHO_LIQUID).abs() < liquid_tol,
        "Liquid density {rho_high:.3} too far from reference {RHO_LIQUID} (tol={liquid_tol})"
    );
}

/// Run the Gibbs ensemble and check conservation laws and phase separation.
#[test]
#[ignore]
fn regression() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/files/gibbs_ensemble");
    run_gibbs_test(&dir, false);
}

/// Same test but with atomic groups.
#[test]
#[ignore]
fn regression_atomic() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/files/gibbs_ensemble_atomic");
    run_gibbs_test(&dir, true);
}

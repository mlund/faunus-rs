//! Wang–Landau free energy against an analytic profile.

// shared test helpers; only `faunus_binary` is used here
#[allow(dead_code)]
mod common;

use std::path::Path;
use std::process::Command;

/// A particle in U = f·x has F(x)/kT = f·x/kT + c. The slope also fixes the sign:
/// writing ln g instead of −ln g flips it.
#[test]
fn free_energy_slope_matches_linear_potential() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/files/wang_landau_1d");
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("free_energy.csv");
    let status = Command::new(common::faunus_binary())
        .current_dir(tmp.path())
        .arg("wang-landau")
        .arg("-i")
        .arg(dir.join("input.yaml"))
        .arg("-s")
        .arg(tmp.path().join("wl_states"))
        .arg("-o")
        .arg(&output)
        .arg("-j")
        .arg("1")
        .status()
        .unwrap();
    assert!(status.success());

    let (cv, free_energy): (Vec<f64>, Vec<f64>) = std::fs::read_to_string(&output)
        .unwrap()
        .lines()
        .skip(1)
        .map(|line| {
            let mut columns = line.split(',').map(|c| c.trim().parse::<f64>().unwrap());
            (columns.next().unwrap(), columns.next().unwrap())
        })
        .unzip();

    // Least-squares slope of F(x)
    let n = cv.len() as f64;
    let (mean_x, mean_f) = (
        cv.iter().sum::<f64>() / n,
        free_energy.iter().sum::<f64>() / n,
    );
    let slope = cv
        .iter()
        .zip(&free_energy)
        .map(|(x, f)| (x - mean_x) * (f - mean_f))
        .sum::<f64>()
        / cv.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();

    // f = 1 kJ/mol/Å at 300 K
    let expected = 1.0 / (faunus::R_IN_KJ_PER_MOL * 300.0);
    assert!(
        (slope - expected).abs() < 0.05 * expected,
        "slope {slope:.3} kT/Å, expected {expected:.3}"
    );
}

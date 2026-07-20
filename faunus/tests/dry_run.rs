//! `--check` dry run: the CLI must validate the input and stop before the simulation,
//! writing no results. These drive the compiled binary because the short-circuit lives in
//! the CLI dispatch, above the `Simulation`/`replay` library seam the other tests exercise.

use std::process::Command;

/// Path to the compiled `faunus` binary, injected by Cargo for integration tests.
const FAUNUS: &str = env!("CARGO_BIN_EXE_faunus");

const VALID_INPUT: &str = "examples/phosphate-titration/input.yaml";

/// A `--check` run on valid input exits successfully and leaves the `-o` file empty: the
/// output probe may create it, but no simulation results are ever written. Guards the
/// `if args.check { return }` short-circuit — drop it and a full run fills the file.
#[test]
fn check_validates_without_running() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("out.yaml");

    let status = Command::new(FAUNUS)
        .arg("--check")
        .arg("-o")
        .arg(&output)
        .args(["run", "-i", VALID_INPUT])
        .status()
        .expect("failed to spawn faunus");

    assert!(status.success(), "--check on valid input should exit 0");
    let bytes_written = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    assert_eq!(
        bytes_written, 0,
        "--check must not write simulation results"
    );
}

/// `--check` must not destroy existing analysis output. Analysis writers open their `file:`
/// lazily (on the first sampled row), so a dry run — which never samples — leaves a populated
/// output file untouched. Guards against the writers reverting to eager, truncate-at-build open.
#[test]
fn check_preserves_existing_analysis_output() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.yaml");
    let base = std::fs::read_to_string(VALID_INPUT).unwrap();
    // The input loads `phosphate.xyz` relative to itself; copy it beside the modified input.
    std::fs::copy(
        "examples/phosphate-titration/phosphate.xyz",
        dir.path().join("phosphate.xyz"),
    )
    .unwrap();
    // Add a file-writing analysis whose output already holds data we must not lose.
    std::fs::write(
        &input,
        base.replacen(
            "analysis:\n",
            "analysis:\n  - !Energy { file: energy.dat, frequency: !Every 10 }\n",
            1,
        ),
    )
    .unwrap();
    let energy = dir.path().join("energy.dat");
    std::fs::write(&energy, "PRIOR DATA\n").unwrap();

    let status = Command::new(FAUNUS)
        .arg("--check")
        .arg("-o")
        .arg(dir.path().join("out.yaml"))
        .arg("run")
        .arg("-i")
        .arg(&input)
        .current_dir(dir.path()) // the analysis `file:` resolves against the working directory
        .status()
        .expect("failed to spawn faunus");

    assert!(status.success(), "--check on valid input should exit 0");
    assert_eq!(
        std::fs::read_to_string(&energy).unwrap(),
        "PRIOR DATA\n",
        "--check must not truncate an existing analysis output file"
    );
}

/// `--check` shares the real validation path, and validation runs *before* the check
/// short-circuits: an unknown key is rejected exactly as in a full run, so the process
/// exits non-zero and names the offending key rather than reporting a clean dry run.
#[test]
fn check_rejects_invalid_input() {
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.yaml");
    let base = std::fs::read_to_string(VALID_INPUT).unwrap();
    std::fs::write(&bad, format!("{base}\nbogus_toplevel_key: 42\n")).unwrap();

    let out = Command::new(FAUNUS)
        .arg("--check")
        .arg("-o")
        .arg(dir.path().join("out.yaml"))
        .arg("run")
        .arg("-i")
        .arg(&bad)
        .output()
        .expect("failed to spawn faunus");

    assert!(!out.status.success(), "--check must fail on invalid input");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("bogus_toplevel_key"),
        "--check should report the unknown key, got: {stderr}"
    );
}

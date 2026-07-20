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
use yaml_serde::Value;

/// Wall-clock measurements and the git revision differ between any two runs of the same input.
const VOLATILE_KEYS: &[&str] = &["timer", "elapsed_seconds", "energy_timers", "version"];

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/files/{name}"))
}

fn parse(yaml: &str) -> Value {
    yaml_serde::from_str(yaml).expect("output is valid YAML")
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

/// A tiny self-contained system. `trajectory` is the analysis output path; when it equals the
/// replayed trajectory, `replay` must refuse (issue #60). `with_moves` adds a propagate section
/// so the input can also generate a trajectory.
fn trajectory_input(trajectory: &Path, with_moves: bool) -> String {
    let propagate = if with_moves {
        "
propagate:
  seed: !Fixed 42
  criterion: Metropolis
  steps: 20
  collections:
    - !Deterministic
      moves:
        - !TranslateMolecule {molecule: particle, max_displacement: 1.0, repeat: 1}"
    } else {
        ""
    };
    format!(
        "
atoms:
  - {{name: X, mass: 1.0, sigma: 2.0, epsilon: 0.5}}
molecules:
  - name: particle
    atoms: [X]
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium: {{permittivity: !Vacuum, temperature: 298.15}}
  energy: []
  blocks:
    - {{molecule: particle, N: 4, insert: !RandomAtomPos {{}}}}
analysis:
  - !Trajectory {{file: \"{}\", save_frame_state: true, frequency: !Every 5}}{}",
        trajectory.display(),
        propagate,
    )
}

/// A rerun whose analysis writes the very trajectory it replays truncates that file mid-read,
/// so `replay` rejects it up front rather than silently processing a single frame (issue #60).
#[test]
fn replay_rejects_an_analysis_that_writes_the_replayed_trajectory() {
    let temp = tempfile::tempdir().expect("temp dir");
    let trajectory = temp.path().join("traj.xtc");
    let input = temp.path().join("input.yaml");
    std::fs::write(&input, trajectory_input(&trajectory, false)).expect("write input");

    // The collision is caught before any frame is read, so the trajectory need not even exist.
    match faunus::replay(&input, &trajectory, None) {
        Err(Error::Input(message)) => assert!(
            message.contains("traj.xtc"),
            "error should name the trajectory: {message}"
        ),
        other => panic!("expected Error::Input naming the collision, got {other:?}"),
    }
}

/// A rerun whose analyses write elsewhere replays every frame — the collision guard does not
/// reject a legitimate rerun.
#[test]
fn replay_processes_every_frame_without_a_collision() {
    let temp = tempfile::tempdir().expect("temp dir");
    let trajectory = temp.path().join("traj.xtc");

    // Generate a short trajectory (with its .aux) by running the input that writes it.
    let generator = temp.path().join("generate.yaml");
    std::fs::write(&generator, trajectory_input(&trajectory, true)).expect("write generator");
    Simulation::from_file(&generator, None)
        .expect("load generator")
        .run()
        .expect("generate trajectory");
    assert!(trajectory.exists(), "generator wrote no trajectory");

    // Replay it through an input whose only analysis writes no files at all.
    let rerun = temp.path().join("rerun.yaml");
    std::fs::write(
        &rerun,
        "
atoms:
  - {name: X, mass: 1.0, sigma: 2.0, epsilon: 0.5}
molecules:
  - name: particle
    atoms: [X]
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium: {permittivity: !Vacuum, temperature: 298.15}
  energy: []
  blocks:
    - {molecule: particle, N: 4, insert: !RandomAtomPos {}}
analysis:
  - !CollectiveVariable {property: volume, frequency: !Every 1}",
    )
    .expect("write rerun input");

    let output = faunus::replay(&rerun, &trajectory, None).expect("replay should succeed");
    let frames = parse(output.to_yaml())["rerun"]["frames"]
        .as_f64()
        .expect("rerun block reports a frame count");
    assert!(
        frames >= 2.0,
        "expected several frames replayed, got {frames}"
    );
}

/// An NPT trajectory carries a box that fluctuates away from the input's. Replay must evaluate each
/// frame in the box it was generated in, not in the one the input file happens to declare (#89).
#[test]
fn replay_uses_the_box_each_frame_was_generated_in() {
    const INPUT_VOLUME: f64 = 8000.0; // the 20 Å cube both inputs declare

    let temp = tempfile::tempdir().expect("temp dir");
    let trajectory = temp.path().join("npt.xtc");

    let system = "
atoms:
  - {name: X, mass: 1.0, sigma: 2.0, epsilon: 0.5}
molecules:
  - name: particle
    atoms: [X]
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium: {permittivity: !Vacuum, temperature: 298.15}
  energy:
    - !Nonbonded
      default:
        - !LennardJones {mixing: LB}
    - !Pressure {atm: 1.0}
  blocks:
    - {molecule: particle, N: 20, insert: !RandomAtomPos {}}";

    let generator = temp.path().join("generate.yaml");
    std::fs::write(
        &generator,
        format!(
            "{system}
analysis:
  - !Trajectory {{file: \"{}\", save_frame_state: true, frequency: !Every 10}}
  - !CollectiveVariable {{property: volume, frequency: !Every 10}}
propagate:
  seed: !Fixed 42
  criterion: Metropolis
  steps: 200
  collections:
    - !Stochastic
      moves:
        - !TranslateMolecule {{molecule: particle, max_displacement: 1.0, repeat: 1}}
        - !VolumeMove {{volume_displacement: 0.2, weight: 1.0}}",
            trajectory.display()
        ),
    )
    .expect("write generator");

    let run = Simulation::from_file(&generator, None)
        .expect("load generator")
        .run()
        .expect("generate trajectory");
    let sampled_volume = parse(run.to_yaml())["analysis"][1]["collective_variable"]["mean"]
        .as_f64()
        .expect("generator samples a mean volume");
    assert!(
        (sampled_volume - INPUT_VOLUME).abs() > 1.0,
        "the volume move never moved the box, so the test proves nothing"
    );

    let rerun = temp.path().join("rerun.yaml");
    std::fs::write(
        &rerun,
        format!(
            "{system}
analysis:
  - !CollectiveVariable {{property: volume, frequency: !Every 1}}"
        ),
    )
    .expect("write rerun input");

    let output = faunus::replay(&rerun, &trajectory, None).expect("replay should succeed");
    let replayed_volume = parse(output.to_yaml())["analysis"][0]["collective_variable"]["mean"]
        .as_f64()
        .expect("replay samples a mean volume");

    assert!(
        (replayed_volume - sampled_volume).abs() < 1e-2 * sampled_volume,
        "replayed in {replayed_volume} Å³, but the frames were generated in {sampled_volume} Å³"
    );
}

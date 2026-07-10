//! A fixed `propagate.seed` must reproduce a run exactly.
//!
//! Every fixture uses an ideal gas (`energy: {}`), so they say nothing about
//! physics and finish in milliseconds — cheap enough to run on every
//! `cargo test` rather than behind `--ignored`.

// shared test helpers; only `run_faunus` is used here
#[allow(dead_code)]
mod common;

use std::path::Path;

/// Name and contents of each `*state.yaml` a run wrote.
type States = Vec<(String, String)>;

/// Run `input.yaml` with `seed` substituted, returning each state file's contents.
///
/// State files hold the coordinates, so they detect a diverging trajectory even
/// though the ideal-gas energy is degenerate. Anything in `preload` is written
/// into the working directory first, which restores a run from a fixed
/// configuration instead of generating one.
fn run_with_seed(fixture: &str, seed: u32, preload: &[(String, String)]) -> States {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/files")
        .join(fixture);
    let tmp = tempfile::tempdir().expect("failed to create temp dir");

    let input = std::fs::read_to_string(dir.join("input.yaml")).expect("read input.yaml");
    let input = input.replace("seed: !Fixed 7", &format!("seed: !Fixed {seed}"));
    let input_path = tmp.path().join("input.yaml");
    std::fs::write(&input_path, input).expect("write input.yaml");

    for (name, contents) in preload {
        std::fs::write(tmp.path().join(name), contents).expect("write preloaded state");
    }

    let state = tmp.path().join("state.yaml");
    let output = tmp.path().join("output.yaml");
    common::run_faunus(&input_path, &state, &output);

    let mut states: States = std::fs::read_dir(tmp.path())
        .expect("read temp dir")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| is_state_file(path))
        .map(|path| {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            (name, std::fs::read_to_string(&path).expect("read state file"))
        })
        .collect();
    states.sort();

    assert!(
        !states.is_empty(),
        "{fixture}: no state files written, nothing to compare"
    );
    states
}

fn is_state_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with("state.yaml"))
}

fn state(states: &States, name: &str) -> String {
    states
        .iter()
        .find(|(file, _)| file == name)
        .unwrap_or_else(|| panic!("no {name} among {:?}", states.iter().map(|s| &s.0)))
        .1
        .clone()
}

/// Same seed twice must agree; a different seed must not. The second half keeps
/// the first from passing vacuously on a state file that omits coordinates.
fn assert_seed_determines_run(fixture: &str) {
    let first = run_with_seed(fixture, 7, &[]);
    let second = run_with_seed(fixture, 7, &[]);
    assert_eq!(
        first, second,
        "{fixture}: identical seeds produced different state files"
    );

    let other = run_with_seed(fixture, 8, &[]);
    assert_ne!(
        first, other,
        "{fixture}: changing the seed changed nothing, so this test proves nothing"
    );
}

/// Guards the seeded initial configuration, which once came from `thread_rng()`.
#[test]
fn single_box_run_is_reproducible() {
    assert_seed_determines_run("determinism_single");
}

/// Guards the inter-box move stream, which once came from `from_entropy()`.
#[test]
fn gibbs_run_is_reproducible() {
    assert_seed_determines_run("determinism_gibbs");
}

/// Each Gibbs box's own Markov chain must follow `propagate.seed`.
///
/// The fixture has no inter-box moves, and both boxes are restored from the same
/// state files, so the coordinates and the inter-box stream are held fixed. Only
/// the per-box chain seeds remain, which is what makes this able to catch a box
/// seeded from a hardcoded constant — such a box repeats one trajectory for every
/// seed, and the whole-run comparisons above would still pass, because its
/// coordinates are cloned from the (seed-dependent) initial configuration.
#[test]
fn gibbs_per_box_chains_follow_seed() {
    let fixture = "determinism_gibbs_chain";
    let pinned = run_with_seed(fixture, 7, &[]);

    let a = run_with_seed(fixture, 7, &pinned);
    let b = run_with_seed(fixture, 7, &pinned);
    let c = run_with_seed(fixture, 8, &pinned);

    for box_state in ["box0_state.yaml", "box1_state.yaml"] {
        assert_eq!(
            state(&a, box_state),
            state(&b, box_state),
            "{box_state}: identical seeds diverged"
        );
        assert_ne!(
            state(&a, box_state),
            state(&c, box_state),
            "{box_state}: unchanged across seeds, so its chain ignores propagate.seed"
        );
    }
}

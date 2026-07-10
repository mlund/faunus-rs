//! A written record of what faunus exports.
//!
//! Integration tests are compiled as a separate crate, so this file sees exactly what a downstream
//! user sees. Every item named below is part of the supported surface; adding one here is a
//! deliberate widening, and a demotion that breaks a consumer shows up as a compile error rather
//! than as a surprise in `duello`.
//!
//! Everything else — the backend, the context traits, moves, analyses, selections and the group
//! machinery — is `pub(crate)` and deliberately unreachable from here. A simulation is driven
//! through `faunus::Simulation`, which hides all of it.

use std::path::Path;

use faunus::topology::{AtomKind, CustomProperty, FindByName, Topology};
use faunus::{cell::Cell, energy::NonbondedMatrix, Info, Particle, PointParticle};
use faunus::{BoxResult, Simulation, SimulationOutput};

/// Name every supported export, so that hiding one fails the build.
#[test]
fn public_api_is_reachable() {
    // Types `duello` and the binary depend on.
    fn _types(_: Option<(AtomKind, Topology, NonbondedMatrix, Cell, Particle)>) {}

    // Traits, named through the methods a consumer calls on them.
    fn _traits<T: Info + CustomProperty, U: PointParticle, V: FindByName<AtomKind>>() {}

    // The command-line entry point, which `src/bin/faunus.rs` is built on.
    #[cfg(feature = "cli")]
    let _: fn() -> anyhow::Result<()> = faunus::cli::do_main;

    // Free functions.
    let _ = faunus::transform::random_unit_vector(&mut rand::thread_rng());

    // Re-exported so downstream crates cannot end up with two incompatible `interatomic` versions.
    let _ = faunus::interatomic::ELECTRIC_PREFACTOR;
}

/// Pin the simulation interface.
///
/// Coercing each item to a function pointer fixes its signature: adding a type parameter or a
/// lifetime later stops compiling here rather than downstream. A future pyo3 wrapper needs exactly
/// this — no generics, no borrows escaping, and a handle it can move onto another thread.
#[test]
fn simulation_api_is_pinned() {
    fn _assert_send_static<T: Send + 'static>() {}
    _assert_send_static::<Simulation>();
    _assert_send_static::<SimulationOutput>();
    _assert_send_static::<BoxResult>();
    _assert_send_static::<faunus::Error>();

    // Rust API guidelines C-DEBUG: every public type is printable.
    fn _assert_debug<T: std::fmt::Debug>() {}
    _assert_debug::<Simulation>();
    _assert_debug::<SimulationOutput>();
    _assert_debug::<BoxResult>();
    _assert_debug::<faunus::Error>();

    // `Error` is a real error type, so `?` and `anyhow` accept it.
    fn _assert_error<T: std::error::Error + Send + Sync + 'static>() {}
    _assert_error::<faunus::Error>();

    let _: fn(&Path, Option<&Path>) -> faunus::Result<Simulation> = Simulation::from_file;
    let _: fn(&str, Option<&Path>) -> faunus::Result<Simulation> = Simulation::from_yaml;
    let _: fn(&mut Simulation) -> faunus::Result<SimulationOutput> = Simulation::run;
    let _: fn(&Simulation, &Path) -> faunus::Result<()> = Simulation::save_state;
    let _: fn(&Path, &Path, Option<&Path>) -> faunus::Result<SimulationOutput> = faunus::replay;

    let _: fn(&SimulationOutput) -> &str = SimulationOutput::to_yaml;
    let _: fn(&SimulationOutput) -> &[BoxResult] = SimulationOutput::boxes;
    let _: fn(&SimulationOutput, &Path) -> faunus::Result<()> = SimulationOutput::write_to;

    let _: fn(&BoxResult) -> f64 = BoxResult::initial_energy;
    let _: fn(&BoxResult) -> f64 = BoxResult::final_energy;
    let _: fn(&BoxResult) -> f64 = BoxResult::drift;
    let _: fn(&BoxResult) -> &str = BoxResult::to_yaml;
}

/// The progress callback is a trait object, not a type parameter, so the method stays callable
/// from a pyo3 wrapper. Naming the type here makes a change to it a compile error.
type RunWithProgress =
    fn(&mut Simulation, &mut dyn FnMut(usize, usize)) -> faunus::Result<SimulationOutput>;

#[test]
fn progress_callback_is_not_generic() {
    let _: RunWithProgress = Simulation::run_with_progress;
}

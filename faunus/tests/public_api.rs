//! A written record of what faunus exports.
//!
//! Integration tests are compiled as a separate crate, so this file sees exactly what a downstream
//! user sees. Every item named below is part of the supported surface; adding one here is a
//! deliberate widening, and a demotion that breaks a consumer shows up as a compile error rather
//! than as a surprise in `duello`.
//!
//! Everything else — the backend, the context traits, moves, analyses, selections and the group
//! machinery — is `pub(crate)` and deliberately unreachable from here. Running a simulation from
//! library code is not supported yet; see issue #54.

use faunus::topology::{AtomKind, CustomProperty, FindByName, Topology};
use faunus::{cell::Cell, energy::NonbondedMatrix, Info, Particle, PointParticle};

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

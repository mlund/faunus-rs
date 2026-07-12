// Copyright 2025 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

//! The contract shared by every *stateful* (caching) energy term.
//!
//! A stateful term caches move-mutable state — a reciprocal-space summator, a
//! pairwise energy matrix, a tessellation — so that a trial move costs less than
//! a full re-evaluation. Implementing [`StatefulEnergy`] instead of hand-writing
//! the energy/update/backup methods makes three latent bugs impossible by
//! construction:
//!
//! 1. The outward energy is *derived* by [`derived_energy`], which routes
//!    [`Change::Everything`] to a fresh [`total_energy`](StatefulEnergy::total_energy)
//!    recompute. A cache can therefore never mask its own drift from the
//!    energy-drift check (which evaluates `Change::Everything`).
//! 2. Splitting `total_energy` (fresh) from
//!    [`partial_energy`](StatefulEnergy::partial_energy) (incremental) names the
//!    two evaluation modes a term must provide, so neither is forgotten.
//! 3. Move-mutable state held behind [`Snapshot`](super::backup::Snapshot) is
//!    backed up whole, so a field cannot be left out of the reject path.

use crate::{Change, ObserveContext};

/// Energy term that caches move-mutable state across Monte Carlo trials.
///
/// The framework calls [`total_energy`](Self::total_energy) for whole-system
/// evaluations and [`partial_energy`](Self::partial_energy) for the incremental
/// subset touched by a move (see [`derived_energy`]). During a trial the runner
/// calls [`save_backup`](Self::save_backup) (pre-move), [`refresh`](Self::refresh)
/// (post-move), then either [`discard_backup`](Self::discard_backup) (accept) or
/// [`undo`](Self::undo) (reject).
pub(crate) trait StatefulEnergy {
    /// Fresh, from-scratch total energy. Must not read the move caches, so a
    /// `Change::Everything` evaluation can expose cache divergence.
    fn total_energy(&self, context: &impl ObserveContext) -> f64;

    /// Incremental energy for the subset a non-`Everything`/`None` change touches.
    /// May read the caches maintained by [`refresh`](Self::refresh).
    fn partial_energy(&self, context: &impl ObserveContext, change: &Change) -> f64;

    /// Update internal caches after the system has changed (post-move).
    fn refresh(&mut self, context: &impl ObserveContext, change: &Change) -> anyhow::Result<()>;

    /// Snapshot move-mutable state so [`undo`](Self::undo) can restore it.
    /// Called pre-move, while the context still holds the old configuration.
    fn save_backup(&mut self, context: &impl ObserveContext, change: &Change);

    /// Restore state from the snapshot (MC reject).
    fn undo(&mut self);

    /// Drop the snapshot, committing the current state (MC accept).
    fn discard_backup(&mut self);
}

/// Derive the outward energy of a [`StatefulEnergy`] for a given change.
///
/// `Change::Everything` always takes the fresh [`total_energy`] path, so a stale
/// or drifting cache can never hide from the energy-drift check. `Change::Volume`
/// stays on the incremental path: a volume move's `refresh` already rebuilds the
/// whole cache, so routing it through `total_energy` would only recompute the
/// same quantity two extra times per move.
pub(crate) fn derived_energy<T: StatefulEnergy>(
    term: &T,
    context: &impl ObserveContext,
    change: &Change,
) -> f64 {
    match change {
        Change::Everything => term.total_energy(context),
        Change::None => 0.0,
        Change::Volume(..) | Change::SingleGroup(..) | Change::Groups(..) => {
            term.partial_energy(context, change)
        }
    }
}

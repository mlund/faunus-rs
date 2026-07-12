//! A minimal, working energy term, kept as a starting point for new ones.
//!
//! It applies a harmonic restraint pulling every particle towards the z = 0 plane. Nothing about
//! that is interesting; what the file demonstrates is the contract every energy term must keep.
//!
//! **A term reads the system; it never writes to it.** `energy` receives a read-only
//! [`ObserveContext`](crate::ObserveContext), so the mutating methods are not merely discouraged,
//! they are absent from the type. Interior mutability for *caching* is still allowed — see the
//! `RefCell<ComSelection>` in `custom_external.rs` — but never for changing the physics.
//!
//! **`energy` must be consistent with the [`Change`] it is given.** The framework calls it twice
//! per trial move, once before and once after, and subtracts. If a term returns a cached value for
//! a change that actually affected it, ΔU is wrong in a way the energy-drift check cannot see: it
//! sums the same wrong numbers. When in doubt, recompute — as this term does, by ignoring `change`
//! and always summing over the whole system. That is correct but O(N) per move; the real terms
//! narrow it, and pay for that with the `save_backup`/`undo` cache protocol.
//!
//! A term that *caches* move-mutable state (a pairwise matrix, a reciprocal summator) instead of
//! recomputing in full implements [`StatefulEnergy`](super::stateful::StatefulEnergy) rather than
//! [`EnergyChange`] directly — see [`CachedZRestraint`] below. The trait splits the outward energy
//! into a fresh `total_energy` and an incremental `partial_energy`, and the framework derives
//! `energy` so that `Change::Everything` always recomputes; a cache therefore cannot hide its own
//! drift from the energy-drift check. The move-mutable state lives behind a
//! [`Snapshot`](super::backup::Snapshot), which makes `save_backup`/`undo`/`discard_backup`
//! one-liners and covers every field automatically.
//!
//! The tests at the bottom exercise both terms against a real system, so this file cannot rot. Copy
//! it, rename it, add a variant to [`EnergyTerm`](super::EnergyTerm), and register the builder.

// Nothing but the tests below drives these terms; this is a template, not a registered energy term.
#![allow(dead_code)]

use super::backup::Snapshot;
use super::stateful::{derived_energy, StatefulEnergy};
use super::EnergyChange;
use crate::{Change, ObserveContext};
use serde::{Deserialize, Serialize};

/// Deserialized from the `energy:` section of the input file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZRestraintBuilder {
    /// Force constant in kJ/(mol·Å²).
    spring_constant: f64,
}

impl ZRestraintBuilder {
    pub fn build(&self) -> anyhow::Result<ZRestraint> {
        anyhow::ensure!(
            self.spring_constant >= 0.0,
            "ZRestraint: spring constant must be non-negative"
        );
        Ok(ZRestraint {
            spring_constant: self.spring_constant,
        })
    }
}

/// Harmonic restraint towards the z = 0 plane: `U = ½k Σᵢ zᵢ²`.
#[derive(Debug, Clone)]
pub struct ZRestraint {
    spring_constant: f64,
}

impl EnergyChange for ZRestraint {
    /// The context is immutable, and `Change` is ignored because this term recomputes in full.
    ///
    /// A term that *does* use `change` must return an energy for exactly the particles the change
    /// touches, and must return the same partition before and after the move.
    fn energy(&self, context: &impl ObserveContext, _change: &Change) -> f64 {
        let sum_z_squared: f64 = context
            .groups()
            .iter()
            .flat_map(|group| group.iter_active())
            .map(|i| context.position(i).z.powi(2))
            .sum();
        0.5 * self.spring_constant * sum_z_squared
    }
}

/// A *stateful* sibling of [`ZRestraint`] that caches the restraint energy, so a move costs less
/// than a full recompute. It models the contract a real caching term (nonbonded, Ewald, …) must
/// keep — the same three energy modes and the [`Snapshot`] backup protocol — on physics simple
/// enough to check by hand.
///
/// The cached, move-mutable state lives behind a [`Snapshot`]: `save_backup`/`undo`/
/// `discard_backup` are one-liners, and a field added to [`CachedState`] is covered by undo
/// automatically, so the forget-a-field backup bug cannot recur.
pub struct CachedZRestraint {
    spring_constant: f64,
    state: Snapshot<CachedState>,
}

/// Everything a trial move mutates lives here, so [`Snapshot`] backs it up whole.
#[derive(Clone)]
struct CachedState {
    /// Cached total restraint energy (kJ/mol), refreshed after every accepted move.
    energy: f64,
}

impl CachedZRestraint {
    pub fn new(spring_constant: f64, context: &impl ObserveContext) -> Self {
        let mut term = Self {
            spring_constant,
            state: Snapshot::new(CachedState { energy: 0.0 }),
        };
        term.state.energy = term.compute(context);
        term
    }

    /// Fresh ½k Σᵢ zᵢ² over the whole system.
    fn compute(&self, context: &impl ObserveContext) -> f64 {
        let sum_z_squared: f64 = context
            .groups()
            .iter()
            .flat_map(|group| group.iter_active())
            .map(|i| context.position(i).z.powi(2))
            .sum();
        0.5 * self.spring_constant * sum_z_squared
    }

    /// Per-term info for output reporting, assembled with the shared `yaml_map!` macro
    /// (the same helper the real terms use for their `to_yaml`).
    pub fn to_yaml(&self) -> serde_yml::Value {
        yaml_map! { "spring_constant" => self.spring_constant }
    }
}

impl EnergyChange for CachedZRestraint {
    /// The framework derives the outward energy: `Change::Everything` recomputes fresh, everything
    /// else reads the cache — so drift is always visible to the drift check.
    fn energy(&self, context: &impl ObserveContext, change: &Change) -> f64 {
        derived_energy(self, context, change)
    }
}

impl StatefulEnergy for CachedZRestraint {
    /// From-scratch total; never reads the cache.
    fn total_energy(&self, context: &impl ObserveContext) -> f64 {
        self.compute(context)
    }

    /// The incremental path. A real term recomputes only the part `change` touches; this template
    /// keeps the contract honest by returning the value [`refresh`](Self::refresh) last cached.
    fn partial_energy(&self, _context: &impl ObserveContext, _change: &Change) -> f64 {
        self.state.energy
    }

    /// Refresh the cache after the move has been applied (positions are now the new ones). A real
    /// term updates only the affected rows here; the template recomputes in full for clarity.
    fn refresh(&mut self, context: &impl ObserveContext, _change: &Change) -> anyhow::Result<()> {
        self.state.energy = self.compute(context);
        Ok(())
    }

    fn save_backup(&mut self, _context: &impl ObserveContext, _change: &Change) {
        self.state.save();
    }

    fn undo(&mut self) {
        self.state.undo();
    }

    fn discard_backup(&mut self) {
        self.state.discard();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::context::PerturbContext;
    use crate::Point;

    /// Two atoms at z = ±2, so the restraint energy is ½·k·(4 + 4) = 4k.
    const TWO_ATOMS: &str = r#"
atoms:
  - {name: A, mass: 1.0, charge: 0.0, sigma: 1.0}
molecules:
  - name: dimer
    atoms: [A, A]
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: dimer
      N: 1
      insert: !Manual [[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;

    fn backend() -> Backend {
        Backend::from_yaml_str(TWO_ATOMS, None, &mut rand::thread_rng()).unwrap()
    }

    #[test]
    fn restrains_particles_towards_the_plane() {
        let context = backend();
        let term = ZRestraintBuilder {
            spring_constant: 2.0,
        }
        .build()
        .unwrap();
        assert_eq!(term.energy(&context, &Change::Everything), 8.0);
    }

    /// The energy must follow the positions, not a stale cache.
    #[test]
    fn the_energy_follows_the_particles() {
        let mut context = backend();
        let term = ZRestraintBuilder {
            spring_constant: 2.0,
        }
        .build()
        .unwrap();

        // Slide both atoms onto the plane; the restraint energy must vanish.
        context.translate_particles(&[0], &Point::new(0.0, 0.0, 2.0));
        context.translate_particles(&[1], &Point::new(0.0, 0.0, -2.0));
        assert_eq!(term.energy(&context, &Change::Everything), 0.0);
    }

    #[test]
    fn a_negative_spring_constant_is_rejected() {
        assert!(ZRestraintBuilder {
            spring_constant: -1.0
        }
        .build()
        .is_err());
    }

    /// The `save_backup` → mutate → `undo` protocol must restore the cache exactly, so a rejected
    /// move leaves the term as if it never happened. This is the invariant every stateful term relies
    /// on; verifying it on the template keeps the backup pattern from silently rotting.
    #[test]
    fn undo_restores_the_cached_energy_after_a_rejected_move() {
        let mut context = backend();
        let mut term = CachedZRestraint::new(2.0, &context);
        assert_eq!(term.energy(&context, &Change::Everything), 8.0);

        let change = Change::SingleGroup(0, crate::GroupChange::RigidBody);
        term.save_backup(&context, &change);

        // Apply a trial move (slide both atoms onto the plane) and refresh the cache.
        context.translate_particles(&[0], &Point::new(0.0, 0.0, 2.0));
        context.translate_particles(&[1], &Point::new(0.0, 0.0, -2.0));
        term.refresh(&context, &change).unwrap();
        assert_eq!(term.partial_energy(&context, &change), 0.0);

        // Reject: the term's undo restores its cache (the caller restores the positions).
        term.undo();
        assert_eq!(term.partial_energy(&context, &change), 8.0);
    }
}

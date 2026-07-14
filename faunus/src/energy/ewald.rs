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

//! Ewald reciprocal-space energy term.
//!
//! Wraps `coulomb::reciprocal::EwaldReciprocal` to provide the k-space
//! contribution to the electrostatic energy in Monte Carlo simulations.
//! Uses the Nymand–Linse O(M·N_k) partial structure factor update for
//! single-group moves instead of O(N·N_k) full rebuilds.
//!
//! When `optimize` is enabled, the splitting parameter α and wave-vector
//! cutoff n_max are jointly optimized at startup to minimize the number of
//! k-vectors while preserving energy accuracy. This is only effective for
//! Yukawa electrostatics (κ > 0) with PBC policy. If optimization changes α,
//! the nonbonded real-space pair potential is automatically rebuilt to match.

use super::backup::Snapshot;
use super::stateful::{derived_energy, StatefulEnergy};
use crate::cell::Shape;
use crate::change::GroupChange;
use crate::Change;
use crate::ObserveContext;
use interatomic::coulomb::reciprocal::{EwaldPolicy, EwaldReciprocal};
use interatomic::coulomb::DebyeLength;
use serde::{Deserialize, Serialize};

/// Move-mutable Ewald state, backed up whole by [`Snapshot`] so the reject path
/// cannot forget a field. A resize (`refresh_charges`) zeroes a deactivated
/// molecule's charges; restoring `charges` on reject keeps a later rigid move of
/// that molecule from reading `charge == 0.0` and silently untracking it.
#[derive(Clone)]
struct EwaldState {
    ewald: EwaldReciprocal,
    /// Cached reciprocal + self energy (kJ/mol)
    cached_energy: f64,
    /// Charges indexed by global particle index (includes zeros)
    charges: Vec<f64>,
}

/// Ewald reciprocal-space energy with backup/undo for MC moves.
#[derive(Clone)]
pub struct EwaldReciprocalEnergy {
    state: Snapshot<EwaldState>,
    /// Electric prefactor: e²/(4πε₀ε_r) in kJ/mol·Å (immutable)
    prefactor: f64,
    /// Reusable position buffers to avoid per-update allocations (scratch).
    pos_buf: (Vec<f64>, Vec<f64>, Vec<f64>),
    /// Old positions of affected particles: (global_index, [x, y, z]). Move-scoped
    /// scratch captured in `save_backup` and consumed by the incremental `refresh`;
    /// *not* restorable state, so it lives outside the `Snapshot`.
    old_positions: Vec<(usize, [f64; 3])>,
}

impl std::fmt::Debug for EwaldReciprocalEnergy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EwaldReciprocalEnergy")
            .field("cached_energy", &self.state.cached_energy)
            .field("prefactor", &self.prefactor)
            .field("num_charges", &self.state.charges.len())
            .finish()
    }
}

/// YAML configuration for Ewald summation (both real-space and reciprocal).
///
/// When present, a matching real-space Ewald pair potential is automatically
/// injected into the nonbonded defaults (before splining), and the reciprocal-space
/// term is added as a separate energy contribution.
///
/// # Example
///
/// ```yaml
/// ewald:
///   cutoff: 12.0
///   accuracy: 1e-5
///   policy: PBC
///   optimize: true  # reduce k-vectors for Yukawa; no-op for pure Coulomb or IPBC
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EwaldBuilder {
    /// Real-space cutoff (Å)
    pub cutoff: f64,
    /// Target relative accuracy (e.g. 1e-5)
    #[serde(default = "default_accuracy")]
    pub accuracy: f64,
    /// Ewald policy: PBC (default) or IPBC
    #[serde(default)]
    pub policy: EwaldPolicy,
    /// Jointly optimize α and n_max to minimize k-vectors at startup.
    ///
    /// Scans (α, n_max) pairs against actual particle data, picking the
    /// fewest k-vectors that reproduce the reference total energy within
    /// the accuracy target. Only effective for Yukawa (κ > 0) with PBC;
    /// silently ignored for pure Coulomb or IPBC.
    #[serde(default)]
    pub optimize: bool,
}

const fn default_accuracy() -> f64 {
    1e-5
}

impl EwaldReciprocalEnergy {
    /// Create from builder, context, and medium.
    pub fn new(
        builder: &EwaldBuilder,
        context: &impl ObserveContext,
        medium: &interatomic::coulomb::Medium,
    ) -> anyhow::Result<Self> {
        let box_length = Self::box_length_from_context(context)?;
        let kappa = medium.debye_length().map(|d| 1.0 / d);
        let mut ewald = EwaldReciprocal::new(box_length, builder.cutoff, builder.accuracy, kappa);
        ewald.set_policy(builder.policy);
        let prefactor = interatomic::coulomb::TO_CHEMISTRY_UNIT / medium.permittivity();

        log::info!(
            "Ewald reciprocal ({:?}): α={:.4}, n_max={}, k-vectors={}",
            builder.policy,
            ewald.alpha(),
            ewald.n_max(),
            ewald.num_k_vectors()
        );

        let charges = Self::extract_charges(context);
        let n = charges.len();
        let mut term = Self {
            state: Snapshot::new(EwaldState {
                ewald,
                cached_energy: 0.0,
                charges,
            }),
            prefactor,
            pos_buf: (vec![0.0; n], vec![0.0; n], vec![0.0; n]),
            old_positions: Vec::new(),
        };
        term.full_update_impl(context, builder.optimize);
        if builder.optimize {
            log::info!(
                "Ewald optimized: α={:.4}, n_max={}, k-vectors={}",
                term.state.ewald.alpha(),
                term.state.ewald.n_max(),
                term.state.ewald.num_k_vectors()
            );
        }
        Ok(term)
    }

    /// Current Ewald splitting parameter α.
    pub fn alpha(&self) -> f64 {
        self.state.ewald.alpha()
    }

    /// Return the real-space Ewald scheme matching the current α.
    ///
    /// After optimization this may differ from the initial scheme derived from
    /// `accuracy` alone, so callers should use this to rebuild the nonbonded
    /// pair matrix when `optimize` is enabled.
    pub fn real_space_scheme(&self) -> interatomic::coulomb::pairwise::RealSpaceEwald {
        self.state.ewald.real_space_scheme()
    }

    fn box_length_from_context(context: &impl ObserveContext) -> anyhow::Result<[f64; 3]> {
        let bb = context
            .cell()
            .bounding_box()
            .ok_or_else(|| anyhow::anyhow!("Ewald requires a cuboid cell with finite volume"))?;
        Ok([bb.x, bb.y, bb.z])
    }

    /// Re-extract charges and resize position buffers to match.
    fn refresh_charges(&mut self, context: &impl ObserveContext) {
        self.state.charges = Self::extract_charges(context);
        let n = self.state.charges.len();
        self.pos_buf.0.resize(n, 0.0);
        self.pos_buf.1.resize(n, 0.0);
        self.pos_buf.2.resize(n, 0.0);
    }

    fn extract_charges(context: &impl ObserveContext) -> Vec<f64> {
        let topology = context.topology_ref();
        let atomkinds = topology.atomkinds();
        let n = context.groups().iter().map(|g| g.capacity()).sum();
        let mut charges = vec![0.0; n];
        for group in context.groups() {
            for i in group.iter_active() {
                charges[i] = atomkinds[context.atom_kind(i).get()].charge();
            }
        }
        charges
    }

    /// Full recompute of structure factors and cached energy.
    fn full_update(&mut self, context: &impl ObserveContext) {
        self.full_update_impl(context, false);
    }

    fn full_update_impl(&mut self, context: &impl ObserveContext, optimize: bool) {
        self.fill_positions(context);
        let (x, y, z) = &self.pos_buf;
        let st = &mut *self.state;
        st.ewald.update_all(x, y, z, &st.charges, None, optimize);
        self.update_cached_energy();
    }

    /// Full recompute with new box dimensions (volume change).
    fn full_update_with_box(&mut self, context: &impl ObserveContext) -> anyhow::Result<()> {
        let box_length = Self::box_length_from_context(context)?;
        self.fill_positions(context);
        let (x, y, z) = &self.pos_buf;
        let st = &mut *self.state;
        st.ewald
            .update_all(x, y, z, &st.charges, Some(box_length), false);
        self.update_cached_energy();
        Ok(())
    }

    fn update_cached_energy(&mut self) {
        let energy = self.prefactor
            * (self.state.ewald.energy() + self.state.ewald.self_energy(&self.state.charges));
        self.state.cached_energy = energy;
    }

    /// Fill pre-allocated position buffers from the current context.
    fn fill_positions(&mut self, context: &impl ObserveContext) {
        let (x, y, z) = &mut self.pos_buf;
        x.iter_mut().for_each(|v| *v = 0.0);
        y.iter_mut().for_each(|v| *v = 0.0);
        z.iter_mut().for_each(|v| *v = 0.0);
        for group in context.groups() {
            for i in group.iter_active() {
                let pos = context.position(i);
                x[i] = pos.x;
                y[i] = pos.y;
                z[i] = pos.z;
            }
        }
    }

    /// Compute energy relevant to a change (`kJ/mol`).
    pub(crate) fn energy(&self, context: &impl ObserveContext, change: &Change) -> f64 {
        derived_energy(self, context, change)
    }

    /// Whether a group change can alter any particle's charge, invalidating the incremental
    /// structure-factor update: a resize (de)activates particles, an identity swap changes an
    /// atom kind. Both require re-extracting charges and a full rebuild.
    fn changes_charges(gc: &GroupChange) -> bool {
        gc.is_resize() || matches!(gc, GroupChange::UpdateIdentity(_))
    }

    /// Resolve affected global particle indices from a group change.
    fn affected_indices(group: &crate::group::Group, gc: &GroupChange) -> Vec<usize> {
        match gc {
            GroupChange::RigidBody => group.iter_active().collect(),
            GroupChange::PartialUpdate(rel)
            | GroupChange::UpdateIdentity(rel)
            | GroupChange::ResizePartial(_, rel) => {
                let offset = group.iter_active().next().unwrap_or(0);
                rel.iter().map(|ri| offset + ri.get()).collect()
            }
            GroupChange::AtomicShrink { rels, .. } => {
                let offset = group.iter_active().next().unwrap_or(group.start());
                rels.iter().map(|rel| offset + rel.get()).collect()
            }
            GroupChange::None | GroupChange::Resize(_) | GroupChange::ResizeExcludeIntra(_) => {
                Vec::new()
            }
        }
    }

    /// O(M·N_k) incremental update for a single-group move.
    ///
    /// Uses old positions from backup and new positions from context to
    /// call `update_particle` for each affected charged particle.
    fn incremental_update(
        &mut self,
        context: &impl ObserveContext,
        group_index: usize,
        group_change: &GroupChange,
    ) {
        if self.old_positions.is_empty() {
            self.full_update(context);
            return;
        }

        let group = &context.groups()[group_index];
        let affected = Self::affected_indices(group, group_change);

        for &idx in &affected {
            let charge = self.state.charges[idx];
            if charge == 0.0 {
                continue;
            }
            // Linear scan is efficient for the typical small number of affected particles
            if let Some(&(_, old)) = self.old_positions.iter().find(|(i, _)| *i == idx) {
                let new_pos = context.position(idx);
                let new = [new_pos.x, new_pos.y, new_pos.z];
                self.state.ewald.update_particle(charge, old, new);
            }
        }
        self.update_cached_energy();
    }

    /// Collect positions of particles that will be affected by the change.
    fn collect_affected_positions(
        &self,
        change: &Change,
        context: &impl ObserveContext,
    ) -> Vec<(usize, [f64; 3])> {
        match change {
            Change::SingleGroup(gi, gc) => {
                let group = &context.groups()[*gi];
                Self::affected_indices(group, gc)
                    .into_iter()
                    .filter(|&i| self.state.charges[i] != 0.0)
                    .map(|i| {
                        let pos = context.position(i);
                        (i, [pos.x, pos.y, pos.z])
                    })
                    .collect()
            }
            Change::None | Change::Everything | Change::Volume(..) | Change::Groups(..) => {
                Vec::new()
            }
        }
    }

    /// Fresh reciprocal + self energy on a scratch summator (`kJ/mol`).
    ///
    /// Recomputes from re-extracted charges, current positions, and the box read from the
    /// context (not the cloned summator's cached dimensions), so the energy-drift check sees
    /// accumulated structure-factor drift, a stale charge after an identity swap, or a box that
    /// has drifted from `context.cell()`.
    fn compute_total(&self, context: &impl ObserveContext) -> f64 {
        let charges = Self::extract_charges(context);
        let n = charges.len();
        let (mut x, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for group in context.groups() {
            for i in group.iter_active() {
                let pos = context.position(i);
                x[i] = pos.x;
                y[i] = pos.y;
                z[i] = pos.z;
            }
        }
        let box_length = Self::box_length_from_context(context).ok();
        let mut ewald = self.state.ewald.clone();
        ewald.update_all(&x, &y, &z, &charges, box_length, false);
        self.prefactor * (ewald.energy() + ewald.self_energy(&charges))
    }

    /// Report Ewald parameters as YAML.
    pub(super) fn to_yaml(&self) -> serde_yml::Value {
        yaml_map! {
            "alpha" => self.state.ewald.alpha(),
            "n_max" => (self.state.ewald.n_max() as u64),
            "k_vectors" => (self.state.ewald.num_k_vectors() as u64),
        }
    }
}

impl StatefulEnergy for EwaldReciprocalEnergy {
    fn total_energy(&self, context: &impl ObserveContext) -> f64 {
        self.compute_total(context)
    }

    fn partial_energy(&self, _context: &impl ObserveContext, _change: &Change) -> f64 {
        self.state.cached_energy
    }

    /// Update internal state after a system change.
    ///
    /// For single-group moves, uses O(M·N_k) incremental structure factor
    /// updates via `update_particle` (Nymand & Linse, JCP 112, 6152, 2000).
    fn refresh(&mut self, context: &impl ObserveContext, change: &Change) -> anyhow::Result<()> {
        match change {
            Change::None => {}
            Change::Volume(..) | Change::Everything => {
                self.full_update_with_box(context)?;
            }
            Change::Groups(changes) => {
                if changes.iter().any(|(_, gc)| Self::changes_charges(gc)) {
                    self.refresh_charges(context);
                }
                self.full_update(context);
            }
            // A resize (de)activates particles and an identity swap can change a charge; either way
            // the incremental structure-factor update — which only moves a fixed charge between
            // positions — is invalid, so re-extract charges and rebuild from scratch (issue #66).
            Change::SingleGroup(_, gc) if Self::changes_charges(gc) => {
                self.refresh_charges(context);
                self.full_update(context);
            }
            Change::SingleGroup(gi, gc) => match gc {
                GroupChange::None => {}
                // The charge-changing variants are handled by the guarded arm above; the rest are
                // position-only moves the incremental structure-factor update handles.
                GroupChange::RigidBody
                | GroupChange::PartialUpdate(_)
                | GroupChange::UpdateIdentity(_)
                | GroupChange::ResizePartial(..)
                | GroupChange::AtomicShrink { .. }
                | GroupChange::Resize(_)
                | GroupChange::ResizeExcludeIntra(_) => {
                    self.incremental_update(context, *gi, gc);
                }
            },
        }
        Ok(())
    }

    /// Snapshot state for later undo. Context has OLD positions (called before move).
    /// `old_positions` is move-scoped scratch outside the `Snapshot`; it drives the
    /// incremental `refresh`, and must be captured here or that path silently falls
    /// back to a full rebuild.
    fn save_backup(&mut self, context: &impl ObserveContext, change: &Change) {
        self.old_positions = self.collect_affected_positions(change, context);
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
    use crate::energy::stateful::StatefulEnergy;
    use crate::group::{GroupCollectionMut, GroupSize};
    use crate::GroupChange;
    use std::io::Write;

    /// Two single-atom molecules of opposite charge (net neutral), individually deactivatable.
    fn charged_context() -> Backend {
        let yaml = "atoms:\n  - {name: Cat, mass: 1.0, charge: 1.0, sigma: 1.0}\n  \
                    - {name: Ani, mass: 1.0, charge: -1.0, sigma: 1.0}\n\
                    molecules:\n  - {name: CAT, atoms: [Cat]}\n  - {name: ANI, atoms: [Ani]}\n\
                    system:\n  cell: !Cuboid [30.0, 30.0, 30.0]\n  \
                    medium: {permittivity: !Vacuum, temperature: 300.0}\n  energy: {}\n  \
                    blocks:\n    - {molecule: CAT, N: 1, insert: !Manual [[0.0, 0.0, 0.0]]}\n    \
                    - {molecule: ANI, N: 1, insert: !Manual [[5.0, 0.0, 0.0]]}\n\
                    propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}\n";
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(yaml.as_bytes()).unwrap();
        Backend::new(tmp.path(), None, &mut rand::thread_rng()).unwrap()
    }

    #[test]
    fn undo_restores_charges_after_a_rejected_resize() {
        let mut context = charged_context();
        let mut term =
            EwaldReciprocalEnergy::new(&test_builder(), &context, &vacuum_medium()).unwrap();

        let original = term.state.charges.clone();
        assert!(original.iter().any(|&q| q != 0.0), "molecules carry charge");

        // Simulate a rejected deletion of molecule 0: back up, deactivate it, refresh (which
        // re-extracts and zeroes its charge), then undo.
        let change = Change::SingleGroup(0, GroupChange::Resize(GroupSize::Shrink(1)));
        term.save_backup(&context, &change);
        context.resize_group(0, GroupSize::Shrink(1)).unwrap();
        term.refresh(&context, &change).unwrap();
        assert_ne!(
            term.state.charges, original,
            "resize should zero the deactivated molecule's charge"
        );

        term.undo();
        assert_eq!(
            term.state.charges, original,
            "undo must restore charges, not just the structure factors"
        );
    }

    /// Pin the Nymand–Linse partial structure-factor update against a from-scratch
    /// recompute. This invariant — incremental single-group updates stay numerically
    /// equal to a full rebuild — must survive the `StatefulEnergy` migration, which
    /// routes single-group moves through `partial_energy`/`refresh` rather than a
    /// fresh reciprocal sum.
    #[test]
    fn incremental_update_matches_fresh_recompute() {
        use crate::group::GroupCollection;
        use crate::Point;

        let mut context = charged_context();
        let builder = test_builder();
        let medium = vacuum_medium();
        let mut term = EwaldReciprocalEnergy::new(&builder, &context, &medium).unwrap();

        let change = Change::SingleGroup(0, GroupChange::RigidBody);
        let shifts = [
            Point::new(0.7, -0.3, 0.5),
            Point::new(-0.4, 0.8, -0.2),
            Point::new(0.1, 0.1, 0.9),
            Point::new(-0.6, -0.5, 0.3),
        ];
        for shift in shifts {
            let indices: Vec<usize> = context.groups()[0].iter_active().collect();
            term.save_backup(&context, &change);
            context.translate_particles(&indices, &shift);
            term.refresh(&context, &change).unwrap(); // incremental (Nymand–Linse) path
            term.discard_backup(); // accept the move

            let incremental = term.energy(&context, &change);
            let fresh = EwaldReciprocalEnergy::new(&builder, &context, &medium)
                .unwrap()
                .energy(&context, &change);
            approx::assert_relative_eq!(incremental, fresh, epsilon = 1e-9, max_relative = 1e-9);
        }
    }

    fn test_builder() -> EwaldBuilder {
        EwaldBuilder {
            cutoff: 12.0,
            accuracy: 1e-4,
            policy: EwaldPolicy::default(),
            optimize: false,
        }
    }

    fn vacuum_medium() -> interatomic::coulomb::Medium {
        interatomic::coulomb::Medium::new(
            300.0,
            interatomic::coulomb::permittivity::Permittivity::Vacuum,
            None,
        )
    }

    /// #67: `energy(Everything)` must be a fresh recompute, not the cache — otherwise
    /// accumulated structure-factor drift is invisible to the energy-drift check.
    #[test]
    fn everything_energy_ignores_a_corrupted_cache() {
        let context = charged_context();
        let mut term =
            EwaldReciprocalEnergy::new(&test_builder(), &context, &vacuum_medium()).unwrap();

        let correct = term.energy(&context, &Change::Everything);
        // Simulate drift by corrupting the cached energy the incremental path maintains.
        term.state.cached_energy = correct + 1234.0;

        let recomputed = term.energy(&context, &Change::Everything);
        approx::assert_relative_eq!(recomputed, correct, epsilon = 1e-9, max_relative = 1e-9);
        assert_ne!(
            recomputed, term.state.cached_energy,
            "Everything must recompute, not return the (corrupted) cache"
        );
    }

    /// #66: an identity swap that changes a particle's charge must re-extract charges;
    /// the incremental structure-factor update cannot represent a charge change.
    #[test]
    fn identity_swap_refreshes_charges() {
        use crate::group::{AtomKindId, GroupCollectionMut, RelIndex};

        let mut context = charged_context();
        let builder = test_builder();
        let medium = vacuum_medium();
        let mut term = EwaldReciprocalEnergy::new(&builder, &context, &medium).unwrap();
        assert_eq!(term.state.charges[0], 1.0, "atom 0 starts as the +1 cation");

        // Swap atom 0 from the +1 cation kind (0) to the -1 anion kind (1). The real speciation
        // move delivers this as `Change::Groups`, not `SingleGroup` (see `ProposedMove::speciation`).
        let change = Change::Groups(vec![(
            0,
            GroupChange::UpdateIdentity(vec![RelIndex::new(0)]),
        )]);
        term.save_backup(&context, &change);
        context.set_atom_kind(0, AtomKindId::new(1));
        term.refresh(&context, &change).unwrap();

        assert_eq!(
            term.state.charges[0], -1.0,
            "identity swap must refresh the cached charge"
        );
        let fresh = EwaldReciprocalEnergy::new(&builder, &context, &medium)
            .unwrap()
            .energy(&context, &change);
        approx::assert_relative_eq!(
            term.energy(&context, &change),
            fresh,
            epsilon = 1e-9,
            max_relative = 1e-9
        );
    }
}

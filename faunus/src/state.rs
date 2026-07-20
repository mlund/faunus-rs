// Copyright 2023-2026 Mikael Lund
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

//! Save and load simulation state for checkpointing and resuming simulations.

use crate::{
    cell::Cell,
    context::WithSimulationCell,
    group::{GroupCollection, GroupSize},
    Particle, UnitQuaternion,
};
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Saved state of a single group (serde detail, not part of the public API).
///
/// Immutable topology fields (`molecule`, `capacity`) are stored to detect
/// mismatches early on load, rather than silently producing wrong energies.
#[derive(Debug, Serialize, Deserialize)]
struct GroupState {
    molecule: crate::group::MoleculeId,
    capacity: usize,
    size: GroupSize,
    /// Rigid-body orientation needed by LD and 6D tabulated energies.
    /// Defaults to identity for backward compatibility with pre-quaternion state files.
    #[serde(default = "UnitQuaternion::identity")]
    quaternion: UnitQuaternion,
}

/// Checkpoint of the simulation state.
///
/// Only runtime-mutable quantities are saved. Topology, hamiltonian, analyses,
/// and moves are rebuilt from the YAML input file on resume, keeping the state
/// file small and avoiding serialization of trait objects.
#[derive(Debug, Serialize, Deserialize)]
pub struct State {
    particles: Vec<Particle>,
    cell: Cell,
    groups: Vec<GroupState>,
    step: usize,
}

impl State {
    /// Capture the current simulation state for checkpointing.
    pub fn save(context: &(impl GroupCollection + WithSimulationCell), step: usize) -> Self {
        State {
            particles: (0..context.num_particles())
                .map(|i| Particle::new(context.atom_kind(i).get(), context.position(i)))
                .collect(),
            cell: context.cell().clone(),
            groups: context
                .groups()
                .iter()
                .map(|g| GroupState {
                    molecule: g.molecule(),
                    capacity: g.capacity(),
                    size: g.size(),
                    quaternion: *g.quaternion(),
                })
                .collect(),
            step,
        }
    }

    /// Restore state into a context. Returns the saved step counter.
    ///
    /// Validates topology compatibility before modifying any state,
    /// so a mismatched state file is rejected cleanly.
    pub fn load(self, context: &mut impl crate::Context) -> anyhow::Result<usize> {
        let num_particles = context.num_particles();
        let num_groups = context.groups().len();

        if self.particles.len() != num_particles {
            anyhow::bail!(
                "Particle count mismatch: state has {}, context has {}",
                self.particles.len(),
                num_particles
            );
        }
        if self.groups.len() != num_groups {
            anyhow::bail!(
                "Group count mismatch: state has {}, context has {}",
                self.groups.len(),
                num_groups
            );
        }

        // Expected after atom swap reactions (titration); not actionable at warn level
        for (i, state_p) in self.particles.iter().enumerate() {
            let ctx_id = context.atom_kind(i).get();
            if state_p.atom_id != ctx_id {
                log::debug!(
                    "Particle {} atom_id differs: state has {}, topology has {} (atom swap?)",
                    i,
                    state_p.atom_id,
                    ctx_id
                );
            }
        }

        // Catch molecule reordering or resized molecule definitions
        for (i, (gs, group)) in self.groups.iter().zip(context.groups().iter()).enumerate() {
            if gs.molecule != group.molecule() {
                anyhow::bail!(
                    "Group {} molecule mismatch: state has {}, topology has {}",
                    i,
                    gs.molecule,
                    group.molecule()
                );
            }
            if gs.capacity != group.capacity() {
                anyhow::bail!(
                    "Group {} capacity mismatch: state has {}, topology has {}",
                    i,
                    gs.capacity,
                    group.capacity()
                );
            }
        }

        let sizes: Vec<_> = self.groups.iter().map(|gs| gs.size).collect();
        let quaternions: Vec<_> = self.groups.iter().map(|gs| gs.quaternion).collect();
        context.restore_configuration(Some(self.cell), &self.particles, &sizes, &quaternions)?;

        log::info!("Restored simulation state");
        Ok(self.step)
    }

    /// Override the step counter (e.g. reset to 0 after umbrella drive phase).
    // Only the `cli`-gated umbrella / Wang-Landau drivers use this.
    #[cfg(feature = "cli")]
    pub fn with_step(mut self, step: usize) -> Self {
        self.step = step;
        self
    }

    /// Load a state from a YAML file.
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let yaml = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read state file {:?}", path.as_ref()))?;
        Ok(yaml_serde::from_str(&yaml)?)
    }

    /// Save the state to a YAML file.
    pub fn to_file(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let yaml = yaml_serde::to_string(self)?;
        std::fs::write(path.as_ref(), yaml)
            .with_context(|| format!("Failed to write state file {:?}", path.as_ref()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::context::{ObserveContext, WithTopology};
    use crate::Point;
    use rand::SeedableRng;

    /// A checkpoint whose orientation disagrees with its own coordinates heals on load.
    ///
    /// State files written before orientations were tracked default to identity, and the
    /// molecular-swap path used to leave them behind entirely, so files in the wild carry
    /// orientations that describe a molecule they no longer hold. The coordinates are the ground
    /// truth: restore recomputes the orientation from them rather than importing the lie.
    #[test]
    fn a_checkpoint_orientation_that_contradicts_its_coordinates_is_recomputed() {
        let yaml = r#"
atoms:
  - {name: T1, mass: 1.0, sigma: 2.0}
  - {name: T2, mass: 2.0, sigma: 2.0}
  - {name: T3, mass: 3.0, sigma: 2.0}
molecules:
  - name: bent
    from_structure: [{T1: [-3.0, 0.0, 0.0]}, {T2: [0.0, 2.5, 0.0]}, {T3: [3.0, 0.0, 0.0]}]
system:
  cell: !Cuboid [30.0, 30.0, 30.0]
  medium: {permittivity: !Vacuum, temperature: 298.15}
  energy: []
  blocks:
    - {molecule: bent, N: 4, active: 4, insert: !RandomCOM {rotate: true}}
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut context = Backend::new(tmp.path(), None, &mut rng).unwrap();

        let truth = *context.groups()[0].quaternion();
        assert_ne!(truth, UnitQuaternion::identity(), "insertion rotated it");

        // A pre-quaternion state file: coordinates as they are, orientations claiming "unrotated".
        let mut state = State::save(&context, 0);
        for group in &mut state.groups {
            group.quaternion = UnitQuaternion::identity();
        }
        state.load(&mut context).unwrap();

        // The coordinates never moved, so the orientation must come back to what they imply.
        assert!(
            context.groups()[0].quaternion().angle_to(&truth) < 1e-9,
            "restore imported an orientation contradicting the coordinates it restored"
        );
    }

    /// Largest separation between any two atoms of a molecule, as stored.
    ///
    /// Measured under minimum image, so it is what the code itself would reconstruct. Taken over
    /// the group's whole capacity rather than its active atoms: the *inactive* slots are what
    /// molecular swap reads as its overlay template, so their geometry has to be sound too — a
    /// dormant group is exactly where a torn molecule hides.
    fn stored_extent(context: &Backend, group: &crate::group::Group) -> f64 {
        let indices: Vec<usize> = (group.start()..group.start() + group.capacity()).collect();
        let mut extent = 0.0_f64;
        for (n, &i) in indices.iter().enumerate() {
            for &j in &indices[n + 1..] {
                extent = extent.max(context.get_distance(i, j).norm());
            }
        }
        extent
    }

    /// Largest separation between any two atoms of the molecule's reference conformation.
    fn reference_extent(reference: &[Point]) -> f64 {
        let mut extent = 0.0_f64;
        for (n, p) in reference.iter().enumerate() {
            for q in &reference[n + 1..] {
                extent = extent.max((p - q).norm());
            }
        }
        extent
    }

    /// Every committed checkpoint must hold intact molecules.
    ///
    /// Scanning the committed fixtures rather than a live run keeps a state file with torn
    /// geometry from being reused as the starting point of the next simulation — which is how
    /// one such file survived in the tree unnoticed.
    ///
    /// The yardstick is the molecule's own reference conformation, not the box: a minimum-image
    /// separation is bounded by half the box *by construction*, so comparing it against half the
    /// box could never report a tear. Restricted to molecules with a reference conformation and
    /// no bonded potentials, whose shape therefore nothing can legitimately change — a bonded
    /// chain under a pivot move is free to extend well beyond its reference, and a kind declared
    /// as a bare list of atoms has no reference shape at all. (Some fixtures deliberately scatter
    /// such atoms across the cell with `insert: !RandomAtomPos`, now confined to geometry-free
    /// kinds — no `from_structure`, no COM — that carry nothing to tear; see issue #83.)
    #[test]
    fn committed_state_files_hold_intact_molecules() {
        const TOLERANCE: f64 = 1e-6;
        let files_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/files");
        let mut checked = 0usize;
        // Collected rather than asserted one at a time: a regeneration can tear many molecules at
        // once, and the whole list is what tells you which fixtures to look at.
        let mut torn: Vec<String> = Vec::new();

        for entry in std::fs::read_dir(&files_dir).unwrap() {
            let dir = entry.unwrap().path();
            let (input, state) = (dir.join("input.yaml"), dir.join("state.yaml"));
            if !input.is_file() || !state.is_file() {
                continue;
            }
            let name = dir.file_name().unwrap().to_string_lossy();
            let mut rng = rand::rngs::StdRng::seed_from_u64(0);
            let mut context = Backend::new(&input, None, &mut rng)
                .unwrap_or_else(|e| panic!("{name}: cannot build system: {e:#}"));
            State::from_file(&state)
                .unwrap()
                .load(&mut context)
                .unwrap_or_else(|e| panic!("{name}: cannot load state: {e:#}"));

            let topology = context.topology();
            for (gi, group) in context.groups().iter().enumerate() {
                let kind = topology.moleculekind(group.molecule());
                let reference = kind.reference_positions();
                if !kind.has_com() || reference.len() < 2 || kind.has_bonded_potentials() {
                    continue;
                }
                checked += 1;

                let expected = reference_extent(reference);
                let found = stored_extent(&context, group);
                if (found - expected).abs() > TOLERANCE {
                    torn.push(format!(
                        "{name}: group {gi} spans {found:.1} Å; its conformation is {expected:.1} Å across"
                    ));
                    // A torn molecule has no meaningful orientation either; one complaint is enough.
                    continue;
                }

                // A checkpoint carries the orientation forward into the next run, so a stored
                // quaternion that disagrees with the stored coordinates imports the lie wholesale.
                // Measured as a residual, not an angle against a fit: a symmetric molecule has
                // several orientations its coordinates cannot tell apart, and all of them are true.
                if group.is_empty() {
                    continue;
                }
                let current: Vec<Point> =
                    group.iter_active().map(|i| context.position(i)).collect();
                let gathered = crate::geometry::gather_molecule(&current, context.cell());
                let Some(error) =
                    crate::geometry::orientation_residual(reference, &gathered, group.quaternion())
                else {
                    continue;
                };
                if error > TOLERANCE {
                    torn.push(format!(
                        "{name}: group {gi} stores an orientation that misses its coordinates by {error:.2e} Å"
                    ));
                }
            }
        }

        // Guards against the filter above silently excluding everything.
        assert!(checked > 0, "no molecular groups were checked");
        assert!(
            torn.is_empty(),
            "{} problems across {checked} rigid molecules in committed state files:\n{}",
            torn.len(),
            torn.join("\n")
        );
    }
}

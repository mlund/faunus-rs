// Copyright 2023-2024 Mikael Lund
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

use crate::cell::VolumeScalePolicy;
use crate::group::{ParticleSelection, RelIndex};
use crate::transform::SpeciationAction;
use crate::ObserveContext;
use crate::{
    montecarlo::{Bias, NewOld},
    transform::Transform,
    Change, Context, GroupChange, Info, Point, UnitQuaternion,
};
use core::fmt::Debug;
use rand::RngCore;
use serde::Serialize;

/// Default value of `repeat` for various structures.
pub(crate) const fn default_repeat() -> usize {
    1
}

/// Default value of `weight` for move selection.
pub(crate) const fn default_weight() -> f64 {
    1.0
}

/// Target for a proposed Monte Carlo move.
#[derive(Clone, Debug)]
pub enum MoveTarget {
    /// Apply to a single group.
    Group(usize),
    /// Apply to the entire system.
    System,
}

/// A fully described but unapplied Monte Carlo move.
///
/// The `transform` mutates the system while the `change` tells the energy terms what to
/// recompute; if the two disagree the move silently gets the wrong energy. Reporting `RigidBody`
/// while applying a `PartialRotate`, say, makes the bonded term return `0.0` for *both* states
/// (it recomputes only when [`GroupChange::internal_change`] holds) and lets the nonbonded term
/// serve the same cached group energy twice, so ΔU ≈ 0 and the move is always accepted. The energy
/// drift check cannot see it, because it sums the same wrong numbers.
///
/// The fields are therefore private, and every constructor derives the change from the transform
/// it builds. [`Self::speciation`] is the one exception — its `GroupChange`s encode physics the
/// transform does not carry — and it is checked in debug builds instead.
///
/// Constructing and inspecting a move goes through the constructors and the accessors — see
/// `translate_group_announces_a_rigid_body_change`. A mismatched pairing cannot be written at all:
/// the fields are private, so there is no struct literal in which to announce `RigidBody` while
/// supplying a `PartialRotate`.
#[derive(Clone, Debug)]
pub struct ProposedMove {
    change: Change,
    displacement: Displacement,
    transform: Transform,
    target: MoveTarget,
}

impl ProposedMove {
    /// Rigidly translate every active particle of a group.
    pub fn translate_group(group: usize, shift: Point) -> Self {
        Self {
            change: Change::SingleGroup(group, GroupChange::RigidBody),
            displacement: Displacement::Distance(shift),
            transform: Transform::Translate(shift),
            target: MoveTarget::Group(group),
        }
    }

    /// Rigidly rotate a group about its mass center.
    pub fn rotate_group(group: usize, rotation: UnitQuaternion, angle: f64) -> Self {
        Self {
            change: Change::SingleGroup(group, GroupChange::RigidBody),
            displacement: Displacement::Angle(angle),
            transform: Transform::Rotate(rotation),
            target: MoveTarget::Group(group),
        }
    }

    /// Move a set of groups together as one rigid cluster (roto-translation).
    ///
    /// `new_mass_centers` are the wrapped target mass centers (parallel to `groups`), which the
    /// caller computes in *unwrapped* coordinates so a cluster spanning the periodic boundary
    /// rotates correctly. `rotation` is the common orientation change (`None` for translation only).
    /// `angle`/`translation` are recorded only for displacement statistics. The move is announced as
    /// a [`Change::Groups`] of rigid bodies — the energy terms then compute both the cluster-vs-rest
    /// and the (rigidly comoving) intra-cluster interactions, so the ΔU is correct for any cluster
    /// geometry, including a cluster whose self-interaction through the periodic boundary changes.
    pub fn cluster(
        groups: Vec<usize>,
        new_mass_centers: Vec<Point>,
        rotation: Option<UnitQuaternion>,
        angle: f64,
        translation: Point,
    ) -> Self {
        debug_assert_eq!(groups.len(), new_mass_centers.len());
        let change = Change::Groups(
            groups
                .iter()
                .map(|&gi| (gi, GroupChange::RigidBody))
                .collect(),
        );
        Self {
            change,
            displacement: Displacement::AngleDistance(angle, translation),
            transform: Transform::ClusterTransform {
                groups,
                new_mass_centers,
                rotation,
            },
            target: MoveTarget::System,
        }
    }

    /// Translate a subset of a group's particles, changing its internal geometry.
    pub fn translate_atoms(group: usize, relative: Vec<RelIndex>, shift: Point) -> Self {
        Self {
            change: Change::SingleGroup(group, GroupChange::PartialUpdate(relative.clone())),
            displacement: Displacement::Distance(shift),
            transform: Transform::PartialTranslate(shift, ParticleSelection::Relative(relative)),
            target: MoveTarget::Group(group),
        }
    }

    /// Move a subset of a group's particles to `positions`, changing its internal geometry.
    ///
    /// `angle` is the rotation the positions were derived from, and is reported to the acceptance
    /// statistics. The positions themselves are computed by the move, which alone knows how to
    /// unwrap the affected atoms across the periodic boundary (see [`Transform::SetPositions`]).
    pub fn move_atoms(
        group: usize,
        relative: Vec<RelIndex>,
        positions: Vec<Point>,
        angle: f64,
    ) -> Self {
        Self {
            change: Change::SingleGroup(group, GroupChange::PartialUpdate(relative.clone())),
            displacement: Displacement::Angle(angle),
            transform: Transform::SetPositions(positions, ParticleSelection::Relative(relative)),
            target: MoveTarget::Group(group),
        }
    }

    /// Scale the cell and every particle position to `volumes.new`.
    pub fn scale_volume(policy: VolumeScalePolicy, volumes: NewOld<f64>) -> Self {
        Self {
            displacement: Displacement::Custom(volumes.new - volumes.old),
            transform: Transform::VolumeScale(policy, volumes.new),
            change: Change::Volume(policy, volumes),
            target: MoveTarget::System,
        }
    }

    /// A reaction step. The `GroupChange`s cannot be derived from the actions — `AtomicShrink`
    /// carries the pre-shrink size and `ResizeExcludeIntra` absorbs the intramolecular energy into
    /// the equilibrium constant — so they are supplied, and cross-checked in debug builds.
    pub fn speciation(actions: Vec<SpeciationAction>, changes: Vec<(usize, GroupChange)>) -> Self {
        debug_assert!(
            speciation_changes_match_actions(&actions, &changes),
            "each group named in the change must be touched by an action, and vice versa"
        );
        Self {
            change: Change::Groups(changes),
            displacement: Displacement::None,
            transform: Transform::Speciation(actions),
            target: MoveTarget::System,
        }
    }

    pub fn change(&self) -> &Change {
        &self.change
    }

    pub fn displacement(&self) -> &Displacement {
        &self.displacement
    }

    #[allow(dead_code)] // asserted by the move-proposal tests
    pub fn transform(&self) -> &Transform {
        &self.transform
    }

    #[allow(dead_code)] // asserted by the move-proposal tests
    pub fn target(&self) -> &MoveTarget {
        &self.target
    }

    /// Apply the transform to the context, saving backup for undo.
    pub fn apply_with_backup(&self, context: &mut impl Context) -> anyhow::Result<()> {
        match self.target {
            MoveTarget::Group(i) => self.transform.on_group_with_backup(i, context),
            MoveTarget::System => self.transform.on_system_with_backup(context),
        }
    }
}

/// Every group named in a speciation `GroupChange` must be touched by some action, and vice versa.
fn speciation_changes_match_actions(
    actions: &[SpeciationAction],
    changes: &[(usize, GroupChange)],
) -> bool {
    let touched: std::collections::BTreeSet<usize> =
        actions.iter().map(SpeciationAction::group_index).collect();
    let described: std::collections::BTreeSet<usize> = changes.iter().map(|(gi, _)| *gi).collect();
    touched == described
}

/// Narrow trait for the unique logic of each Monte Carlo move.
pub trait MoveProposal<T: ObserveContext>: Debug + Info {
    /// Describe a move without applying it; context is read-only.
    fn propose_move(&mut self, context: &T, rng: &mut dyn RngCore) -> Option<ProposedMove>;

    /// Re-check, before proposing, any invariant the system can break under the move.
    ///
    /// What held at build time need not still hold: a titration swap alters atom identities, GCMC
    /// inserts and removes groups. `Err` aborts the run and means the move can no longer sample
    /// correctly — not merely that it has nothing to do, which is a `None` from
    /// [`propose_move`](Self::propose_move).
    fn revalidate(&mut self, _context: &T) -> anyhow::Result<()> {
        Ok(())
    }

    /// Optional bias added to the trial energy for acceptance.
    fn bias(&self, _change: &Change, _energies: &NewOld<f64>) -> Bias {
        Bias::None
    }

    /// Number of steps to advance after attempting the move.
    fn step_by(&self) -> usize {
        1
    }

    /// Called once the trial has been resolved, with the context in its settled state.
    ///
    /// The context is whatever the outcome made it: the trial configuration if the move was
    /// accepted, the original one if it was rejected and rolled back. This is the only point at
    /// which a move can bring state it derives from the configuration back in step, and the only
    /// point at which it *knows* the change was its own — so it may update that state cheaply
    /// instead of rebuilding it. Also used to track per-sub-move statistics (per-reaction in
    /// speciation, squared displacement in cluster moves).
    fn on_trial_outcome(&mut self, _context: &T, _accepted: bool) {}

    /// Serialize the move-specific fields to a tagged YAML value.
    fn to_yaml(&self) -> Option<yaml_serde::Value>;
}

/// Wrap a serializable value in a YAML tag.
pub(crate) fn tagged_yaml(tag: &str, value: &impl Serialize) -> Option<yaml_serde::Value> {
    let value = yaml_serde::to_value(value).ok()?;
    Some(yaml_serde::Value::Tagged(Box::new(
        yaml_serde::value::TaggedValue {
            tag: yaml_serde::value::Tag::new(tag),
            value,
        },
    )))
}

/// Enum used to store the extent of displacement of a move.
///
/// This is used for collecting statistics about for far moves change
/// the system. Used to track mean squared displacements.
#[derive(Clone, Debug)]
pub enum Displacement {
    /// Displacement vector; typically due to a translation
    Distance(Point),
    /// Angular displacement; typically due to a rotation
    Angle(f64),
    /// Displacement vector and angular displacement; typically due to a rototranslational move
    AngleDistance(f64, Point),
    /// A custom displacement
    Custom(f64),
    /// Zero displacement - typically used for rejected moves
    Zero,
    /// No displacement appropriate
    None,
}

impl TryFrom<Displacement> for f64 {
    type Error = &'static str;
    fn try_from(value: Displacement) -> Result<Self, Self::Error> {
        match value {
            Displacement::Distance(x) => Ok(x.norm()),
            Displacement::Angle(x) => Ok(x),
            Displacement::Custom(x) => Ok(x),
            Displacement::Zero => Ok(0.0),
            _ => Err("Cannot convert displacement to floating point number"),
        }
    }
}

#[cfg(test)]
mod pairing_tests {
    use super::*;
    // Only the `#[cfg(debug_assertions)]` speciation tests below use this; gate it so
    // `cargo test --release` (where those tests vanish) doesn't see an unused import.
    #[cfg(debug_assertions)]
    use crate::group::GroupSize;

    /// Was the `ProposedMove` doc example, before `propagate` became crate-private.
    #[test]
    fn translate_group_announces_a_rigid_body_change() {
        let proposed = ProposedMove::translate_group(0, crate::Point::new(0.1, 0.0, 0.0));
        assert!(matches!(
            proposed.change(),
            Change::SingleGroup(0, GroupChange::RigidBody)
        ));
    }

    #[test]
    fn translate_atoms_reports_exactly_the_relative_indices_it_moves() {
        let relative = vec![RelIndex::new(2), RelIndex::new(5)];
        let proposed =
            ProposedMove::translate_atoms(3, relative.clone(), Point::new(1.0, 0.0, 0.0));

        let Change::SingleGroup(group, GroupChange::PartialUpdate(changed)) = proposed.change()
        else {
            panic!("expected SingleGroup/PartialUpdate");
        };
        let Transform::PartialTranslate(_, ParticleSelection::Relative(moved)) =
            proposed.transform()
        else {
            panic!("expected PartialTranslate/Relative");
        };
        assert_eq!(*group, 3);
        assert_eq!(changed, &relative);
        assert_eq!(moved, &relative);
    }

    /// A change of internal geometry must never be labelled `RigidBody`: the bonded term recomputes
    /// only when `internal_change()` holds, so the mislabelled move would drop its bonded ΔU.
    #[test]
    fn move_atoms_never_labels_itself_rigid_body() {
        let proposed =
            ProposedMove::move_atoms(0, vec![RelIndex::new(1)], vec![Point::zeros()], 0.1);
        let Change::SingleGroup(_, group_change) = proposed.change() else {
            panic!("expected SingleGroup");
        };
        assert!(group_change.internal_change());
        assert!(!matches!(group_change, GroupChange::RigidBody));
    }

    #[test]
    fn rotate_group_is_rigid_and_carries_no_internal_change() {
        let proposed = ProposedMove::rotate_group(0, UnitQuaternion::identity(), 0.1);
        let Change::SingleGroup(_, group_change) = proposed.change() else {
            panic!("expected SingleGroup");
        };
        assert!(matches!(group_change, GroupChange::RigidBody));
        assert!(!group_change.internal_change());
    }

    /// The escape hatch is still checked: a `GroupChange` naming a group no action touches would
    /// make the energy terms recompute the wrong groups.
    ///
    /// The validator is a `debug_assert!`, so it — and this test — exist only where debug
    /// assertions do. `cargo test --release` compiles both away rather than running a test that
    /// can never panic.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "each group named in the change must be touched by an action")]
    fn speciation_rejects_a_change_for_an_untouched_group() {
        let actions = vec![SpeciationAction::DeactivateGroup(7)];
        let changes = vec![(9, GroupChange::Resize(GroupSize::Empty))];
        let _ = ProposedMove::speciation(actions, changes);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "each group named in the change must be touched by an action")]
    fn speciation_rejects_an_action_with_no_change() {
        let actions = vec![
            SpeciationAction::DeactivateGroup(7),
            SpeciationAction::DeactivateGroup(8),
        ];
        let changes = vec![(7, GroupChange::Resize(GroupSize::Empty))];
        let _ = ProposedMove::speciation(actions, changes);
    }
}

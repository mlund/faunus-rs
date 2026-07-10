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
/// Constructing and inspecting a move goes through the constructors and accessors:
///
/// ```
/// use faunus::propagate::ProposedMove;
/// use faunus::{Change, GroupChange, Point};
///
/// let proposed = ProposedMove::translate_group(0, Point::new(0.1, 0.0, 0.0));
/// assert!(matches!(
///     proposed.change(),
///     Change::SingleGroup(0, GroupChange::RigidBody)
/// ));
/// ```
///
/// The same code fails to compile if it reaches for the field instead — the only difference from
/// the example above, so the failure can only be the privacy of `change`:
///
/// ```compile_fail
/// use faunus::propagate::ProposedMove;
/// use faunus::{Change, GroupChange, Point};
///
/// let proposed = ProposedMove::translate_group(0, Point::new(0.1, 0.0, 0.0));
/// assert!(matches!(
///     proposed.change,
///     Change::SingleGroup(0, GroupChange::RigidBody)
/// ));
/// ```
///
/// and a mismatched pairing cannot be written at all, because there is no struct literal to write:
///
/// ```compile_fail
/// use faunus::propagate::{Displacement, MoveTarget, ProposedMove};
/// use faunus::transform::Transform;
/// use faunus::{Change, GroupChange, UnitQuaternion};
///
/// // A partial rotation announced as `RigidBody` would silently drop the bonded ΔU.
/// let _ = ProposedMove {
///     change: Change::SingleGroup(0, GroupChange::RigidBody),
///     transform: Transform::Rotate(UnitQuaternion::identity()),
///     displacement: Displacement::Angle(0.1),
///     target: MoveTarget::Group(0),
/// };
/// ```
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

    /// Translate a subset of a group's particles, changing its internal geometry.
    pub fn translate_atoms(group: usize, relative: Vec<RelIndex>, shift: Point) -> Self {
        Self {
            change: Change::SingleGroup(group, GroupChange::PartialUpdate(relative.clone())),
            displacement: Displacement::Distance(shift),
            transform: Transform::PartialTranslate(shift, ParticleSelection::Relative(relative)),
            target: MoveTarget::Group(group),
        }
    }

    /// Rotate a subset of a group's particles about `center`, changing its internal geometry.
    pub fn rotate_atoms(
        group: usize,
        relative: Vec<RelIndex>,
        center: Point,
        rotation: UnitQuaternion,
        angle: f64,
    ) -> Self {
        Self {
            change: Change::SingleGroup(group, GroupChange::PartialUpdate(relative.clone())),
            displacement: Displacement::Angle(angle),
            transform: Transform::PartialRotate(
                center,
                rotation,
                ParticleSelection::Relative(relative),
            ),
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

    pub fn transform(&self) -> &Transform {
        &self.transform
    }

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
pub trait MoveProposal<T: Context>: Debug + Info {
    /// Describe a move without applying it; context is read-only.
    fn propose_move(&mut self, context: &T, rng: &mut dyn RngCore) -> Option<ProposedMove>;

    /// Optional bias added to the trial energy for acceptance.
    fn bias(&self, _change: &Change, _energies: &NewOld<f64>) -> Bias {
        Bias::None
    }

    /// Number of steps to advance after attempting the move.
    fn step_by(&self) -> usize {
        1
    }

    /// Called after a trial move is accepted or rejected.
    /// Override to track per-sub-move statistics (e.g. per-reaction in speciation).
    fn on_trial_outcome(&mut self, _accepted: bool) {}

    /// Serialize the move-specific fields to a tagged YAML value.
    fn to_yaml(&self) -> Option<serde_yml::Value>;
}

/// Wrap a serializable value in a YAML tag.
pub(crate) fn tagged_yaml(tag: &str, value: &impl Serialize) -> Option<serde_yml::Value> {
    let value = serde_yml::to_value(value).ok()?;
    Some(serde_yml::Value::Tagged(Box::new(
        serde_yml::value::TaggedValue {
            tag: serde_yml::value::Tag::new(tag),
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
    use crate::group::GroupSize;

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

    /// A partial rotation must never be labelled `RigidBody`: the bonded term recomputes only when
    /// `internal_change()` holds, so the mislabelled move would drop its bonded ΔU entirely.
    #[test]
    fn rotate_atoms_never_labels_itself_rigid_body() {
        let proposed = ProposedMove::rotate_atoms(
            0,
            vec![RelIndex::new(1)],
            Point::zeros(),
            UnitQuaternion::identity(),
            0.1,
        );
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

    #[test]
    fn speciation_accepts_changes_that_match_its_actions() {
        let actions = vec![SpeciationAction::DeactivateGroup(7)];
        let changes = vec![(7, GroupChange::Resize(GroupSize::Empty))];
        let _ = ProposedMove::speciation(actions, changes);
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

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

use crate::{
    montecarlo::{Bias, NewOld},
    transform::Transform,
    Change, Context, Info, Point,
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
#[derive(Clone, Debug)]
pub struct ProposedMove {
    pub change: Change,
    pub displacement: Displacement,
    pub transform: Transform,
    pub target: MoveTarget,
}

impl ProposedMove {
    /// Apply the transform to the context, saving backup for undo.
    pub fn apply_with_backup(&self, context: &mut impl Context) -> anyhow::Result<()> {
        match self.target {
            MoveTarget::Group(i) => self.transform.on_group_with_backup(i, context),
            MoveTarget::System => self.transform.on_system_with_backup(context),
        }
    }
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

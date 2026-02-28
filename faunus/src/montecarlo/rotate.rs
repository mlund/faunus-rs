// Copyright 2023 Mikael Lund
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

use crate::montecarlo;
use crate::propagate::{tagged_yaml, Displacement, MoveProposal, MoveTarget, ProposedMove};
use crate::transform::{random_quaternion, Transform};
use crate::{Change, Context, GroupChange};
use rand::RngCore;
use serde::{Deserialize, Serialize};

/// Move for rotating a random molecule.
///
/// This will pick a random molecule of type `molecule_id` and rotate it by a random angle.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotateMolecule {
    /// Name of the molecule type to rotate.
    #[serde(rename = "molecule")]
    molecule_name: String,
    /// Id of the molecule type to rotate.
    #[serde(skip)]
    molecule_id: usize,
    /// Maximum angular displacement (radians).
    #[serde(alias = "dp")]
    max_displacement: f64,
    /// Move selection weight.
    #[serde(skip_serializing)]
    pub(crate) weight: f64,
    /// Repeat the move N times.
    #[serde(default = "crate::propagate::default_repeat")]
    #[serde(skip_serializing)]
    pub(crate) repeat: usize,
}

impl RotateMolecule {
    /// Validate and finalize the move.
    pub(crate) fn finalize(&mut self, context: &impl Context) -> anyhow::Result<()> {
        self.molecule_id =
            montecarlo::find_molecule_id(context, &self.molecule_name, "RotateMolecule")?;
        Ok(())
    }
}

impl<T: Context> MoveProposal<T> for RotateMolecule {
    fn propose_move(&mut self, context: &T, rng: &mut dyn RngCore) -> Option<ProposedMove> {
        let group_index = montecarlo::random_group(context, rng, self.molecule_id)?;
        let (quaternion, angle) = random_quaternion(rng, self.max_displacement);
        Some(ProposedMove {
            change: Change::SingleGroup(group_index, GroupChange::RigidBody),
            displacement: Displacement::Angle(angle),
            transform: Transform::Rotate(quaternion),
            target: MoveTarget::Group(group_index),
        })
    }

    fn to_yaml(&self) -> Option<serde_yaml::Value> {
        tagged_yaml("RotateMolecule", self)
    }
}

impl crate::Info for RotateMolecule {
    fn short_name(&self) -> Option<&'static str> {
        Some("rotate_molecule")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Rigid body rotation of random molecule")
    }
}

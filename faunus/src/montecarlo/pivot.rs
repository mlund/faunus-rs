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

use crate::group::ParticleSelection;
use crate::montecarlo;
use crate::propagate::{tagged_yaml, Displacement, MoveProposal};
use crate::topology::BondGraph;
use crate::transform::{random_unit_vector, Transform};
use crate::{Change, Context, GroupChange};
use nalgebra::UnitVector3;
use rand::prelude::*;
use rand::RngCore;
use serde::{Deserialize, Serialize};

/// Move for performing pivot rotations on flexible polymer chains.
///
/// Picks a random backbone atom as pivot and rotates one tail of the chain
/// around it, efficiently decorrelating end-to-end distance.
/// See Madras & Sokal, J. Stat. Phys. 50, 109–186 (1988).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PivotMove {
    /// Name of the molecule type to pivot.
    #[serde(rename = "molecule")]
    molecule_name: String,
    /// Id of the molecule type to pivot.
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
    /// Cached bond graph for topology-aware pivot selection.
    #[serde(skip)]
    bond_graph: BondGraph,
}

impl PivotMove {
    /// Validate and finalize the move.
    pub(crate) fn finalize(&mut self, context: &impl Context) -> anyhow::Result<()> {
        self.molecule_id = montecarlo::find_molecule_id(context, &self.molecule_name, "PivotMove")?;
        self.bond_graph = context.topology().moleculekinds()[self.molecule_id]
            .bond_graph()
            .clone();
        Ok(())
    }
}

impl<T: Context> MoveProposal<T> for PivotMove {
    fn propose_move(
        &mut self,
        context: &mut T,
        rng: &mut dyn RngCore,
    ) -> Option<(Change, Displacement)> {
        if self.bond_graph.is_empty() {
            return None;
        }

        let group_index = montecarlo::random_group(context, rng, self.molecule_id)?;
        let group = &context.groups()[group_index];
        let n = group.iter_active().count();
        if n < 3 {
            return None;
        }

        let pivot_rel = rng.gen_range(0..n);
        let &direction = self.bond_graph.neighbors(pivot_rel).choose(rng)?;

        let side_a = self.bond_graph.connected_from(direction, pivot_rel);
        let side_b = self.bond_graph.connected_from(pivot_rel, direction);
        let rotated_rel = if side_a.len() <= side_b.len() {
            side_a
        } else {
            side_b
        };

        let pivot_pos = context.position(group.start() + pivot_rel);

        let axis = random_unit_vector(rng);
        let uaxis = UnitVector3::new_normalize(axis);
        let angle = self.max_displacement * 2.0 * (rng.r#gen::<f64>() - 0.5);
        let quaternion = crate::UnitQuaternion::from_axis_angle(&uaxis, angle);

        Transform::PartialRotate(
            pivot_pos,
            quaternion,
            ParticleSelection::RelIndex(rotated_rel.clone()),
        )
        .on_group_with_backup(group_index, context)
        .unwrap();

        Some((
            Change::SingleGroup(group_index, GroupChange::PartialUpdate(rotated_rel)),
            Displacement::Angle(angle),
        ))
    }

    fn to_yaml(&self) -> Option<serde_yaml::Value> {
        tagged_yaml("PivotMove", self)
    }
}

impl crate::Info for PivotMove {
    fn short_name(&self) -> Option<&'static str> {
        Some("pivot")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Pivot rotation of polymer chain")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaml_parsing() {
        let yaml = "!PivotMove {molecule: Polymer, dp: 1.5, weight: 2.0}";
        let pivot: PivotMove = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(pivot.molecule_name, "Polymer");
        assert_eq!(pivot.max_displacement, 1.5);
        assert_eq!(pivot.weight, 2.0);
        assert_eq!(pivot.repeat, 1); // default
        assert_eq!(pivot.molecule_id, 0); // skipped during deserialization
    }

    #[test]
    fn yaml_unknown_field_rejected() {
        let yaml = "!PivotMove {molecule: Polymer, dp: 1.5, weight: 2.0, unknown: 42}";
        assert!(serde_yaml::from_str::<PivotMove>(yaml).is_err());
    }
}

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

use super::MoveStatistics;
use crate::montecarlo;
use crate::propagate::Displacement;
use crate::transform::random_unit_vector;
use crate::{Change, Context, GroupChange};
use nalgebra::UnitVector3;
use rand::prelude::*;
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
    weight: f64,
    /// Repeat the move N times.
    #[serde(default = "crate::propagate::default_repeat")]
    repeat: usize,
    /// Move statistics.
    #[serde(skip_deserializing)]
    statistics: MoveStatistics,
}

impl PivotMove {
    /// Propose a pivot rotation on a polymer chain.
    ///
    /// Picks a random group of the specified molecule type, selects a random
    /// pivot atom (excluding the first atom), and rotates the segment before
    /// the pivot around the pivot atom's position.
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(crate) fn propose_move(
        &mut self,
        context: &mut impl Context,
        rng: &mut impl Rng,
    ) -> Option<(Change, Displacement)> {
        let group_index = montecarlo::random_group(context, rng, self.molecule_id)?;
        let group = &context.groups()[group_index];
        let n = group.iter_active().count();

        if n < 3 {
            return None;
        }

        // Pick random pivot index i in 1..N (relative; atom 0 is excluded as pivot)
        let i = rng.gen_range(1..n);
        let pivot_abs = group.start() + i;
        let pivot_pos = context.particle(pivot_abs).pos;

        // Generate random rotation
        let axis = random_unit_vector(rng);
        let uaxis = UnitVector3::new_normalize(axis);
        let angle = self.max_displacement * 2.0 * (rng.r#gen::<f64>() - 0.5);
        let quaternion = crate::UnitQuaternion::from_axis_angle(&uaxis, angle);

        // Absolute indices of atoms to rotate (segment before pivot)
        let abs_indices: Vec<usize> = (group.start()..group.start() + i).collect();

        // Rotate the segment around the pivot position
        context.rotate_particles(&abs_indices, &quaternion, Some(-pivot_pos));
        context.update_mass_center(group_index);

        // Return change with relative indices of rotated atoms
        Some((
            Change::SingleGroup(group_index, GroupChange::PartialUpdate((0..i).collect())),
            Displacement::Angle(angle),
        ))
    }

    /// Get immutable reference to the statistics of the move.
    pub(crate) const fn get_statistics(&self) -> &MoveStatistics {
        &self.statistics
    }

    /// Get mutable reference to the statistics of the move.
    pub(crate) const fn get_statistics_mut(&mut self) -> &mut MoveStatistics {
        &mut self.statistics
    }

    /// Get weight of the move.
    pub(crate) const fn weight(&self) -> f64 {
        self.weight
    }

    /// Number of times the move should be repeated if selected.
    pub(crate) const fn repeat(&self) -> usize {
        self.repeat
    }

    /// Validate and finalize the move.
    pub(crate) fn finalize(&mut self, context: &impl Context) -> anyhow::Result<()> {
        self.molecule_id = context
            .topology()
            .moleculekinds()
            .iter()
            .position(|x| x.name() == &self.molecule_name)
            .ok_or_else(|| {
                anyhow::Error::msg(
                    "Molecule kind in the definition of 'PivotMove' move does not exist.",
                )
            })?;
        Ok(())
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

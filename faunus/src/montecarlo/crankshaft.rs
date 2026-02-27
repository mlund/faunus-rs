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
use crate::topology::BondGraph;
use crate::{Change, Context, GroupChange};
use nalgebra::UnitVector3;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Move for performing crankshaft rotations around dihedral axes.
///
/// Picks a random proper dihedral, then rotates the smaller sub-tree
/// around the middle bond vector. This preserves bond lengths and angles
/// by construction.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrankshaftMove {
    /// Name of the molecule type.
    #[serde(rename = "molecule")]
    molecule_name: String,
    /// Id of the molecule type.
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
    /// Cached bond graph for sub-tree selection.
    #[serde(skip)]
    bond_graph: BondGraph,
    /// Middle bonds of proper dihedrals, stored as [i, j] pairs.
    #[serde(skip)]
    dihedral_bonds: Vec<[usize; 2]>,
}

impl CrankshaftMove {
    /// Propose a crankshaft rotation around a dihedral axis.
    ///
    /// Picks a random proper dihedral's middle bond, BFS-walks the bond graph
    /// from both sides, and rotates the smaller sub-tree around the bond vector.
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(crate) fn propose_move(
        &mut self,
        context: &mut impl Context,
        rng: &mut impl Rng,
    ) -> Option<(Change, Displacement)> {
        if self.dihedral_bonds.is_empty() {
            return None;
        }

        let group_index = montecarlo::random_group(context, rng, self.molecule_id)?;
        let group = &context.groups()[group_index];
        let n = group.iter_active().count();
        if n < 4 {
            return None;
        }

        let &[i, j] = self.dihedral_bonds.choose(rng)?;

        // Rotate the smaller side for better acceptance
        let side_a = self.bond_graph.connected_from(i, j);
        let side_b = self.bond_graph.connected_from(j, i);
        let (pivot_rel, dir_rel, rotated_rel) = if side_a.len() <= side_b.len() {
            (j, i, side_a)
        } else {
            (i, j, side_b)
        };

        let group_start = group.start();
        let pivot_pos = context.particle(group_start + pivot_rel).pos;
        let dir_pos = context.particle(group_start + dir_rel).pos;

        let uaxis = UnitVector3::new_normalize(dir_pos - pivot_pos);
        let angle = self.max_displacement * 2.0 * (rng.r#gen::<f64>() - 0.5);
        let quaternion = crate::UnitQuaternion::from_axis_angle(&uaxis, angle);

        let abs_indices: Vec<usize> = rotated_rel.iter().map(|&r| group_start + r).collect();

        context.rotate_particles(&abs_indices, &quaternion, Some(-pivot_pos));
        context.update_mass_center(group_index);

        Some((
            Change::SingleGroup(group_index, GroupChange::PartialUpdate(rotated_rel)),
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
        self.molecule_id =
            montecarlo::find_molecule_id(context, &self.molecule_name, "CrankshaftMove")?;
        let topology = context.topology();
        let mol_kind = &topology.moleculekinds()[self.molecule_id];
        self.bond_graph = mol_kind.bond_graph().clone();
        self.dihedral_bonds = mol_kind
            .dihedrals()
            .iter()
            .filter(|d| !d.is_improper())
            .map(|d| [d.index()[1], d.index()[2]])
            .collect();
        self.dihedral_bonds.sort_unstable();
        self.dihedral_bonds.dedup();
        Ok(())
    }
}

impl crate::Info for CrankshaftMove {
    fn short_name(&self) -> Option<&'static str> {
        Some("crankshaft")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Crankshaft rotation around dihedral axis")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yaml_parsing() {
        let yaml = "!CrankshaftMove {molecule: Peptide, dp: 0.5, weight: 1.0}";
        let m: CrankshaftMove = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(m.molecule_name, "Peptide");
        assert_eq!(m.max_displacement, 0.5);
        assert_eq!(m.weight, 1.0);
        assert_eq!(m.repeat, 1);
        assert_eq!(m.molecule_id, 0);
        assert!(m.dihedral_bonds.is_empty());
    }

    #[test]
    fn yaml_unknown_field_rejected() {
        let yaml = "!CrankshaftMove {molecule: Peptide, dp: 0.5, weight: 1.0, unknown: 42}";
        assert!(serde_yaml::from_str::<CrankshaftMove>(yaml).is_err());
    }
}

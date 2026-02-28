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
use crate::propagate::{tagged_yaml, Displacement, MoveProposal, MoveTarget, ProposedMove};
use crate::topology::BondGraph;
use crate::transform::{random_displacement, Transform};
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
    #[serde(skip_serializing, default = "crate::propagate::default_weight")]
    pub(crate) weight: f64,
    /// Repeat the move N times.
    #[serde(default = "crate::propagate::default_repeat")]
    #[serde(skip_serializing)]
    pub(crate) repeat: usize,
    /// Cached bond graph for sub-tree selection.
    #[serde(skip)]
    bond_graph: BondGraph,
    /// Middle bonds of proper dihedrals, stored as [i, j] pairs.
    #[serde(skip)]
    dihedral_bonds: Vec<[usize; 2]>,
}

impl CrankshaftMove {
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

impl<T: Context> MoveProposal<T> for CrankshaftMove {
    fn propose_move(&mut self, context: &T, rng: &mut dyn RngCore) -> Option<ProposedMove> {
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

        let side_a = self.bond_graph.connected_from(i, j);
        let side_b = self.bond_graph.connected_from(j, i);
        let (pivot_rel, dir_rel, rotated_rel) = if side_a.len() <= side_b.len() {
            (j, i, side_a)
        } else {
            (i, j, side_b)
        };

        let group_start = group.start();
        let pivot_pos = context.position(group_start + pivot_rel);
        let dir_pos = context.position(group_start + dir_rel);

        let uaxis = UnitVector3::new_normalize(dir_pos - pivot_pos);
        let angle = random_displacement(rng, self.max_displacement);
        let quaternion = crate::UnitQuaternion::from_axis_angle(&uaxis, angle);

        Some(ProposedMove {
            change: Change::SingleGroup(
                group_index,
                GroupChange::PartialUpdate(rotated_rel.clone()),
            ),
            displacement: Displacement::Angle(angle),
            transform: Transform::PartialRotate(
                pivot_pos,
                quaternion,
                ParticleSelection::RelIndex(rotated_rel),
            ),
            target: MoveTarget::Group(group_index),
        })
    }

    fn to_yaml(&self) -> Option<serde_yaml::Value> {
        tagged_yaml("CrankshaftMove", self)
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

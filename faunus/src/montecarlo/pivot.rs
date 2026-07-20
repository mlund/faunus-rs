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

use crate::group::MoleculeId;
use crate::montecarlo::{self, branch};
use crate::propagate::{tagged_yaml, MoveProposal, ProposedMove};
use crate::topology::BondGraph;
use crate::transform::random_quaternion;
use crate::ObserveContext;
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
    molecule_id: MoleculeId,
    /// Maximum rotation angle (radians).
    #[serde(alias = "dprot")]
    max_angle: f64,
    /// Move selection weight.
    #[serde(skip_serializing, default = "crate::propagate::default_weight")]
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
    pub(crate) fn finalize(&mut self, context: &impl ObserveContext) -> anyhow::Result<()> {
        self.molecule_id = montecarlo::find_molecule_id(context, &self.molecule_name, "PivotMove")?;
        montecarlo::validate_max_angle(self.max_angle, "PivotMove")?;
        let topology = context.topology();
        let molecule = topology.moleculekind(self.molecule_id);
        montecarlo::validate_flexible(molecule, "PivotMove")?;
        self.bond_graph = molecule.bond_graph().clone();
        Ok(())
    }
}

impl<T: ObserveContext> MoveProposal<T> for PivotMove {
    fn propose_move(&mut self, context: &T, rng: &mut dyn RngCore) -> Option<ProposedMove> {
        let group_index = montecarlo::random_group(context, rng, self.molecule_id)?;
        let group = &context.groups()[group_index];
        if !branch::is_intact(group, &self.bond_graph, 3) {
            return None;
        }

        let pivot = rng.gen_range(0..group.len());
        let &neighbour = self.bond_graph.neighbors(pivot).choose(rng)?;

        // Turning the pivot's own side is equivalent to turning the other one, and cheaper when it
        // is the smaller. A leaf pivot, though, has nothing on its side but itself, and it sits at
        // the centre and cannot move: that proposal would be an always-accepted no-op.
        let branch = if self.bond_graph.degree(pivot) == 1 {
            self.bond_graph.connected_from(neighbour, pivot)
        } else {
            self.bond_graph.smaller_branch(neighbour, pivot).atoms
        };

        let rotation = random_quaternion(rng, self.max_angle).into();
        Some(branch::propose_rotation(
            context,
            &self.bond_graph,
            group_index,
            group.start(),
            pivot,
            &branch,
            &rotation,
        ))
    }

    fn to_yaml(&self) -> Option<yaml_serde::Value> {
        tagged_yaml("PivotMove", self)
    }
}

impl_info!(PivotMove, "pivot", "Pivot rotation of polymer chain");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::{GroupCollectionMut, GroupSize};
    use crate::montecarlo::chain_fixture::*;
    use crate::Point;
    use rand::rngs::StdRng;

    /// Number of independent proposals each geometric invariant is checked over.
    const PROPOSALS: usize = 64;

    fn pivot(max_angle: f64) -> PivotMove {
        yaml_serde::from_str(&format!("{{molecule: Chain, max_angle: {max_angle}}}")).unwrap()
    }

    /// A pivot rotates a sub-tree rigidly about one of its own atoms, so no bond can change length.
    fn assert_bond_lengths_conserved(positions: &[Point], spec: ChainSpec) {
        let context = chain_context(positions, spec);
        let reference = bond_lengths(&context);
        let mut move_ = pivot(1.0);
        move_.finalize(&context).unwrap();

        for seed in 0..PROPOSALS {
            let mut context = context.clone();
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let proposal = move_.propose_move(&context, &mut rng).unwrap();
            apply(&mut context, &proposal);
            for (bond, (after, before)) in bond_lengths(&context).iter().zip(&reference).enumerate()
            {
                assert!(
                    (after - before).abs() < 1e-9,
                    "seed {seed}: bond {bond} changed from {before} to {after}"
                );
            }
        }
    }

    #[test]
    fn conserves_bond_lengths_for_a_chain_wrapped_across_the_boundary() {
        let spec = ChainSpec::default();
        assert_bond_lengths_conserved(&wrapped_chain(6, spec.box_length), spec);
    }

    /// The rotated sub-tree then reaches beyond half the box from the pivot, where a
    /// minimum-image convention taken atom-by-atom picks the wrong periodic image.
    #[test]
    fn conserves_bond_lengths_for_a_chain_longer_than_half_the_box() {
        let spec = ChainSpec {
            box_length: 12.0,
            ..ChainSpec::default()
        };
        assert_bond_lengths_conserved(&overlong_chain(32, spec.box_length), spec);
    }

    /// A proposal that cannot move any bead is accepted with ΔU = 0, inflating the acceptance ratio.
    #[test]
    fn never_proposes_a_noop() {
        let context = chain_context(&chain(6), ChainSpec::default());
        let mut move_ = pivot(1.0);
        move_.finalize(&context).unwrap();

        for seed in 0..PROPOSALS {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let proposal = move_.propose_move(&context, &mut rng).unwrap();
            assert!(
                !is_noop(&context, &proposal),
                "seed {seed}: proposal leaves every selected bead on the rotation axis"
            );
        }
    }

    /// The bond graph spans the molecule kind, so its offsets may address deactivated beads.
    #[test]
    fn skips_partially_active_group() {
        let mut context = chain_context(&chain(8), ChainSpec::default());
        let mut move_ = pivot(1.0);
        move_.finalize(&context).unwrap();
        context.resize_group(0, GroupSize::Partial(5)).unwrap();

        for seed in 0..PROPOSALS {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            assert!(
                move_.propose_move(&context, &mut rng).is_none(),
                "seed {seed}: proposed a move on a partially active group"
            );
        }
    }

    /// Zero is the dangerous one: every proposal is then the identity, so the move is always
    /// accepted and the run reports 100 % acceptance while sampling nothing.
    #[test]
    fn rejects_unusable_max_angle() {
        let context = chain_context(&chain(6), ChainSpec::default());
        for max_angle in ["0", "-1.0", "4.0", ".nan", ".inf"] {
            let mut move_: PivotMove =
                yaml_serde::from_str(&format!("{{molecule: Chain, max_angle: {max_angle}}}"))
                    .unwrap();
            assert!(
                move_.finalize(&context).is_err(),
                "max_angle {max_angle} should be rejected"
            );
        }
    }

    #[test]
    fn rejects_molecule_without_bonds() {
        let context = chain_context(
            &chain(6),
            ChainSpec {
                bonds: false,
                ..ChainSpec::default()
            },
        );
        assert!(pivot(1.0).finalize(&context).is_err());
    }

    #[test]
    fn rejects_rigid_molecule() {
        let context = chain_context(
            &chain(6),
            ChainSpec {
                rigid: true,
                ..ChainSpec::default()
            },
        );
        assert!(pivot(1.0).finalize(&context).is_err());
    }

    #[test]
    fn yaml_parsing() {
        let yaml = "!PivotMove {molecule: Polymer, max_angle: 1.5, weight: 2.0}";
        let pivot: PivotMove = yaml_serde::from_str(yaml).unwrap();
        assert_eq!(pivot.molecule_name, "Polymer");
        assert_eq!(pivot.max_angle, 1.5);
        assert_eq!(pivot.weight, 2.0);
        assert_eq!(pivot.repeat, 1); // default
        assert_eq!(pivot.molecule_id, MoleculeId::new(0)); // skipped during deserialization
    }

    #[test]
    fn yaml_unknown_field_rejected() {
        let yaml = "!PivotMove {molecule: Polymer, max_angle: 1.5, weight: 2.0, unknown: 42}";
        assert!(yaml_serde::from_str::<PivotMove>(yaml).is_err());
    }
}

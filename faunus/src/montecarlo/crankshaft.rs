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

use crate::cell::BoundaryConditions;
use crate::group::MoleculeId;
use crate::montecarlo::{self, branch};
use crate::propagate::{tagged_yaml, MoveProposal, ProposedMove};
use crate::topology::{BondGraph, Dihedral};
use crate::transform::random_displacement;
use crate::ObserveContext;
use nalgebra::UnitVector3;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

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
    /// Cached bond graph for sub-tree selection.
    #[serde(skip)]
    bond_graph: BondGraph,
    /// Middle bonds of proper dihedrals, stored as [i, j] pairs.
    #[serde(skip)]
    dihedral_bonds: Vec<[usize; 2]>,
}

impl CrankshaftMove {
    /// Validate and finalize the move.
    pub(crate) fn finalize(&mut self, context: &impl ObserveContext) -> anyhow::Result<()> {
        self.molecule_id =
            montecarlo::find_molecule_id(context, &self.molecule_name, "CrankshaftMove")?;
        montecarlo::validate_max_angle(self.max_angle, "CrankshaftMove")?;
        let topology = context.topology();
        let molecule = topology.moleculekind(self.molecule_id);
        montecarlo::validate_flexible(molecule, "CrankshaftMove")?;
        self.bond_graph = molecule.bond_graph().clone();

        let proper: Vec<&Dihedral> = molecule
            .dihedrals()
            .iter()
            .filter(|dihedral| !dihedral.is_improper())
            .collect();
        let candidates: Vec<[usize; 2]> = if proper.is_empty() {
            // Fall back to all bonds when no proper dihedrals exist (e.g. FASTA chains)
            molecule
                .bonds()
                .iter()
                .map(|bond| [bond.index()[0], bond.index()[1]])
                .collect()
        } else {
            proper
                .iter()
                .map(|dihedral| [dihedral.index()[1], dihedral.index()[2]])
                .collect()
        };

        // A bond with a terminal end has no torsion about it: one side of the cut is the lone end
        // atom, which lies on the rotation axis, and the other is the whole rest of the molecule,
        // whose rotation is a rigid-body one. Either way, nothing internal changes.
        let has_torsion =
            |&[i, j]: &[usize; 2]| self.bond_graph.degree(i) > 1 && self.bond_graph.degree(j) > 1;
        // orient each bond before collecting: two dihedrals may traverse the same bond either way,
        // and the set must not then weight it twice
        let oriented = |[i, j]: [usize; 2]| if i > j { [j, i] } else { [i, j] };
        let bonds: BTreeSet<[usize; 2]> = candidates
            .into_iter()
            .map(oriented)
            .filter(has_torsion)
            .collect();

        anyhow::ensure!(
            !bonds.is_empty(),
            "CrankshaftMove: molecule '{}' has no bond with a torsion about it",
            self.molecule_name
        );
        self.dihedral_bonds = bonds.into_iter().collect();
        Ok(())
    }
}

impl<T: ObserveContext> MoveProposal<T> for CrankshaftMove {
    fn propose_move(&mut self, context: &T, rng: &mut dyn RngCore) -> Option<ProposedMove> {
        let group_index = montecarlo::random_group(context, rng, self.molecule_id)?;
        let group = &context.groups()[group_index];
        if !branch::is_intact(group, &self.bond_graph, 4) {
            return None;
        }

        let &[i, j] = self.dihedral_bonds.choose(rng)?;
        // Turn the smaller side about the bond. The bond's other end anchors it and, lying on the
        // axis, holds every bond length and bond angle at the joint fixed.
        let branch = self.bond_graph.smaller_branch(i, j);

        let start = group.start();
        // minimum image: the bond may straddle a cell face, in which case the stored coordinates
        // differ by nearly a box length and point nowhere near along the bond
        let axis = context.cell().distance(
            &context.position(start + branch.root),
            &context.position(start + branch.cut_at),
        );
        let angle = random_displacement(rng, self.max_angle);
        let quaternion =
            crate::UnitQuaternion::from_axis_angle(&UnitVector3::new_normalize(axis), angle);

        Some(branch::propose_rotation(
            context,
            &self.bond_graph,
            group_index,
            start,
            branch.cut_at,
            &branch.atoms,
            &(quaternion, angle).into(),
        ))
    }

    fn to_yaml(&self) -> Option<serde_yml::Value> {
        tagged_yaml("CrankshaftMove", self)
    }
}

impl_info!(
    CrankshaftMove,
    "crankshaft",
    "Crankshaft rotation around dihedral axis"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::group::{GroupCollectionMut, GroupSize};
    use crate::montecarlo::chain_fixture::*;
    use crate::Point;
    use rand::rngs::StdRng;

    /// Number of independent proposals each geometric invariant is checked over.
    const PROPOSALS: usize = 64;

    fn crankshaft(max_angle: f64) -> CrankshaftMove {
        serde_yml::from_str(&format!("{{molecule: Chain, dprot: {max_angle}}}")).unwrap()
    }

    /// A crankshaft turns a sub-tree about one of its own bonds, so it conserves every bond length
    /// *and* every bond angle by construction — only torsions about the chosen bond may change.
    fn assert_internal_geometry_conserved(positions: &[Point], spec: ChainSpec) {
        let context = chain_context(positions, spec);
        let (lengths, angles) = (bond_lengths(&context), bond_angles(&context));
        let mut move_ = crankshaft(1.0);
        move_.finalize(&context).unwrap();

        for seed in 0..PROPOSALS {
            let mut context = context.clone();
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let proposal = move_.propose_move(&context, &mut rng).unwrap();
            apply(&mut context, &proposal);
            for (bond, (after, before)) in bond_lengths(&context).iter().zip(&lengths).enumerate() {
                assert!(
                    (after - before).abs() < 1e-9,
                    "seed {seed}: bond {bond} changed length from {before} to {after}"
                );
            }
            for (bead, (after, before)) in bond_angles(&context).iter().zip(&angles).enumerate() {
                assert!(
                    (after - before).abs() < 1e-9,
                    "seed {seed}: angle at bead {} changed from {before} to {after}",
                    bead + 1
                );
            }
        }
    }

    #[test]
    fn conserves_internal_geometry_for_a_chain_wrapped_across_the_boundary() {
        let spec = ChainSpec::default();
        assert_internal_geometry_conserved(&wrapped_chain(6, spec.box_length), spec);
    }

    /// The rotated sub-tree then reaches beyond half the box from the axis, where a
    /// minimum-image convention taken atom-by-atom picks the wrong periodic image.
    #[test]
    fn conserves_internal_geometry_for_a_chain_longer_than_half_the_box() {
        let spec = ChainSpec {
            box_length: 12.0,
            ..ChainSpec::default()
        };
        assert_internal_geometry_conserved(&overlong_chain(32, spec.box_length), spec);
    }

    /// Without proper dihedrals the move falls back to plain bonds, where a terminal bond would
    /// rotate its lone leaf atom about an axis the atom itself lies on.
    #[test]
    fn never_proposes_a_noop_without_dihedrals() {
        let context = chain_context(
            &chain(6),
            ChainSpec {
                dihedrals: false,
                ..ChainSpec::default()
            },
        );
        let mut move_ = crankshaft(1.0);
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
        let mut move_ = crankshaft(1.0);
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

    /// Dihedrals traversing a shared bond in opposite directions must not weight it twice.
    #[test]
    fn dihedral_bonds_are_orientation_independent() {
        let yaml = "
atoms:
  - {name: A, mass: 1.0, sigma: 1.0, eps: 0.1}
molecules:
  - name: Chain
    atoms: [A, A, A, A, A, A]
    has_com: true
    bonds:
      - {index: [0, 1], kind: !Harmonic {k: 100.0, req: 1.0}}
      - {index: [1, 2], kind: !Harmonic {k: 100.0, req: 1.0}}
      - {index: [2, 3], kind: !Harmonic {k: 100.0, req: 1.0}}
      - {index: [1, 4], kind: !Harmonic {k: 100.0, req: 1.0}}
      - {index: [2, 5], kind: !Harmonic {k: 100.0, req: 1.0}}
    dihedrals:
      - {index: [0, 1, 2, 3], kind: !ProperHarmonic {k: 5.0, aeq: 120.0}}
      - {index: [5, 2, 1, 4], kind: !ProperHarmonic {k: 5.0, aeq: 120.0}}
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium:
    permittivity: !Vacuum
    temperature: 300.0
  energy:
    nonbonded:
      default:
        - !LennardJones {mixing: LB}
  blocks:
    - molecule: Chain
      N: 1
      insert: !Manual
        - [0.0, 0.0, 0.0]
        - [1.0, 0.0, 0.0]
        - [1.5, 0.9, 0.0]
        - [2.5, 0.9, 0.3]
        - [0.7, -0.5, 0.8]
        - [1.1, 1.5, -0.8]
";
        let context = Backend::from_yaml_str(yaml, None, &mut rand::thread_rng()).unwrap();
        let mut move_ = crankshaft(1.0);
        move_.finalize(&context).unwrap();

        // both dihedrals turn about bond 1–2, listed as [1,2] and [2,1]
        assert_eq!(move_.dihedral_bonds, vec![[1, 2]]);
    }

    /// Zero is the dangerous one: every proposal is then the identity, so the move is always
    /// accepted and the run reports 100 % acceptance while sampling nothing.
    #[test]
    fn rejects_unusable_max_angle() {
        let context = chain_context(&chain(6), ChainSpec::default());
        for max_angle in ["0", "-1.0", "4.0", ".nan", ".inf"] {
            let mut move_: CrankshaftMove =
                serde_yml::from_str(&format!("{{molecule: Chain, dprot: {max_angle}}}")).unwrap();
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
        assert!(crankshaft(1.0).finalize(&context).is_err());
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
        assert!(crankshaft(1.0).finalize(&context).is_err());
    }

    #[test]
    fn yaml_parsing() {
        let yaml = "!CrankshaftMove {molecule: Peptide, dprot: 0.5, weight: 1.0}";
        let m: CrankshaftMove = serde_yml::from_str(yaml).unwrap();
        assert_eq!(m.molecule_name, "Peptide");
        assert_eq!(m.max_angle, 0.5);
        assert_eq!(m.weight, 1.0);
        assert_eq!(m.repeat, 1);
        assert_eq!(m.molecule_id, MoleculeId::new(0));
        assert!(m.dihedral_bonds.is_empty());
    }

    #[test]
    fn yaml_unknown_field_rejected() {
        let yaml = "!CrankshaftMove {molecule: Peptide, dprot: 0.5, weight: 1.0, unknown: 42}";
        assert!(serde_yml::from_str::<CrankshaftMove>(yaml).is_err());
    }
}

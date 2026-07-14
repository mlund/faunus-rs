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

//! Rigid rotation of a bonded sub-tree, shared by the pivot and crankshaft moves.

use crate::cell::BoundaryConditions;
use crate::group::{Group, RelIndex};
use crate::propagate::ProposedMove;
use crate::topology::BondGraph;
use crate::{ObserveContext, UnitQuaternion};
use std::collections::VecDeque;

/// Whether `group` may have its internal geometry sampled right now.
///
/// The bond graph describes the molecule *kind*, so its offsets address every atom of the kind,
/// including any the group has deactivated. Rotating those would displace particles that the
/// energy of the change does not account for, so a partially active group is left alone.
pub(super) fn is_intact(group: &Group, bond_graph: &BondGraph, min_atoms: usize) -> bool {
    group.is_full() && group.len() >= min_atoms && group.len() == bond_graph.num_atoms()
}

/// A rotation and the angle it turns through, kept together so the two cannot drift apart: the
/// quaternion moves the atoms, while the angle is what the acceptance statistics report.
pub(super) struct Rotation {
    pub quaternion: UnitQuaternion,
    pub angle: f64,
}

impl From<(UnitQuaternion, f64)> for Rotation {
    fn from((quaternion, angle): (UnitQuaternion, f64)) -> Self {
        Self { quaternion, angle }
    }
}

/// Propose a rigid rotation of `branch` about `anchor`.
///
/// The proposal carries positions rather than a rotation because the sub-tree must first be
/// unwrapped by *following bonds*, walking the graph outward from `anchor` and accumulating the
/// minimum-image vector of each bond in turn. Bonds are short, so every step is unambiguous, and
/// the sub-tree stays contiguous however far it reaches across the cell. Taking the minimum image
/// of each atom independently — the obvious reading of "rotate about a centre", and what a
/// rotation of the bare coordinates does — instead folds
/// atoms more than half a box length from `anchor` into the wrong periodic image, tearing the
/// chain apart at exactly the chain lengths these moves exist to sample.
///
/// `anchor` stays fixed and need not belong to `branch`, but must be bonded into it, so that
/// `{anchor} ∪ branch` is connected.
pub(super) fn propose_rotation(
    context: &impl ObserveContext,
    bond_graph: &BondGraph,
    group_index: usize,
    group_start: usize,
    anchor: usize,
    branch: &[usize],
    rotation: &Rotation,
) -> ProposedMove {
    let position = |atom: usize| context.position(group_start + atom);
    let centre = position(anchor);

    let mut in_branch = vec![false; bond_graph.num_atoms()];
    branch.iter().for_each(|&atom| in_branch[atom] = true);

    let mut unwrapped = vec![None; bond_graph.num_atoms()];
    unwrapped[anchor] = Some(centre);
    let mut queue = VecDeque::from([anchor]);
    while let Some(atom) = queue.pop_front() {
        let (unwrapped_atom, position_atom) = (unwrapped[atom].expect("dequeued"), position(atom));
        for &neighbour in bond_graph.neighbors(atom) {
            if unwrapped[neighbour].is_some() || !in_branch[neighbour] {
                continue;
            }
            let bond = context
                .cell()
                .distance(&position(neighbour), &position_atom);
            unwrapped[neighbour] = Some(unwrapped_atom + bond);
            queue.push_back(neighbour);
        }
    }

    let positions = branch
        .iter()
        .map(|&atom| {
            let position = unwrapped[atom]
                .expect("every atom of a branch is reachable from its anchor through bonds");
            centre + rotation.quaternion.transform_vector(&(position - centre))
        })
        .collect();
    let branch = branch.iter().copied().map(RelIndex::new).collect();

    ProposedMove::move_atoms(group_index, branch, positions, rotation.angle)
}

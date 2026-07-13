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

//! Bonded-chain fixtures shared by the pivot and crankshaft move tests.
//!
//! The moves under test claim two geometric invariants: every intramolecular bond length is
//! conserved, and every proposal actually displaces something. Both are checked here against a
//! chain whose bead positions the test dictates exactly, so that periodic-boundary cases
//! (a chain wrapped across a face, a chain longer than half the box) can be constructed on demand.

use crate::backend::Backend;
use crate::cell::BoundaryConditions;
use crate::context::{WithSimulationCell, WithTopology};
use crate::group::GroupCollection;
use crate::propagate::ProposedMove;
use crate::transform::Transform;
use crate::{ObserveContext, Point};

/// How the chain molecule is declared in the topology.
#[derive(Clone, Copy)]
pub(super) struct ChainSpec {
    /// Edge length of the cubic cell.
    pub box_length: f64,
    /// Declare proper dihedrals over every consecutive quadruple.
    pub dihedrals: bool,
    /// Declare the molecule rigid.
    pub rigid: bool,
    /// Declare consecutive harmonic bonds.
    pub bonds: bool,
}

impl Default for ChainSpec {
    fn default() -> Self {
        Self {
            box_length: 20.0,
            dihedrals: true,
            rigid: false,
            bonds: true,
        }
    }
}

/// Build a context holding a single chain with the given bead positions.
pub(super) fn chain_context(positions: &[Point], spec: ChainSpec) -> Backend {
    let n = positions.len();
    let atoms = vec!["A"; n].join(", ");
    let bonds = if spec.bonds {
        (0..n.saturating_sub(1))
            .map(|i| {
                format!(
                    "      - {{index: [{i}, {}], kind: !Harmonic {{k: 100.0, req: 1.0}}}}\n",
                    i + 1
                )
            })
            .collect::<String>()
    } else {
        String::new()
    };
    let dihedrals = if spec.dihedrals && spec.bonds {
        (0..n.saturating_sub(3))
            .map(|i| {
                format!(
                    "      - {{index: [{i}, {}, {}, {}], kind: !ProperHarmonic {{k: 5.0, aeq: 120.0}}}}\n",
                    i + 1,
                    i + 2,
                    i + 3
                )
            })
            .collect::<String>()
    } else {
        String::new()
    };
    let manual = positions
        .iter()
        .map(|p| format!("        - [{}, {}, {}]\n", p.x, p.y, p.z))
        .collect::<String>();

    let yaml = format!(
        "atoms:
  - {{name: A, mass: 1.0, sigma: 1.0, eps: 0.1}}

molecules:
  - name: Chain
    atoms: [{atoms}]
    has_com: true
    degrees_of_freedom: {dof}
{bonds_section}{dihedrals_section}
system:
  cell: !Cuboid [{l}, {l}, {l}]
  medium:
    permittivity: !Vacuum
    temperature: 300.0
  energy:
    nonbonded:
      default:
        - !LennardJones {{mixing: LB}}
  blocks:
    - molecule: Chain
      N: 1
      insert: !Manual
{manual}
",
        dof = if spec.rigid { "Rigid" } else { "Free" },
        bonds_section = if bonds.is_empty() {
            String::new()
        } else {
            format!("    bonds:\n{bonds}")
        },
        dihedrals_section = if dihedrals.is_empty() {
            String::new()
        } else {
            format!("    dihedrals:\n{dihedrals}")
        },
        l = spec.box_length,
    );

    Backend::from_yaml_str(&yaml, None, &mut rand::thread_rng())
        .expect("chain fixture should be a valid input")
}

/// An open helix of `n` beads, 1 Å apart, with a bond angle of ≈124°, laid out from `origin`.
///
/// Helical rather than straight because a straight chain is degenerate for a crankshaft: every
/// bead lies on every bond axis, so every rotation is trivially the identity. The helix axis is
/// tilted off the box axes as well — a chain along a box axis makes the stored coordinate
/// difference of a bond crossing that face merely *antiparallel* to the true bond vector, which
/// spans the same rotation axis and so hides a missing minimum-image convention.
fn helix(n: usize, origin: Point) -> Vec<Point> {
    const TURN: f64 = 2.0; // rad per bead
    const RADIUS: f64 = 0.4;
    const RISE: f64 = 1.0;
    let scale = 1.0 / (2.0 * RADIUS * RADIUS * (1.0 - TURN.cos()) + RISE * RISE).sqrt();
    let tilt = crate::UnitQuaternion::from_axis_angle(
        &nalgebra::UnitVector3::new_normalize(Point::new(1.0, 1.0, 0.3)),
        0.9,
    );
    (0..n)
        .map(|i| {
            let turn = i as f64 * TURN;
            let bead =
                Point::new(RADIUS * turn.cos(), RADIUS * turn.sin(), RISE * i as f64) * scale;
            origin + tilt.transform_vector(&bead)
        })
        .collect()
}

/// Fold into [-L/2, L/2), as the cell would.
fn wrap(beads: Vec<Point>, box_length: f64) -> Vec<Point> {
    beads
        .into_iter()
        .map(|mut p| {
            p.iter_mut()
                .for_each(|x| *x -= box_length * (*x / box_length).round());
            p
        })
        .collect()
}

/// A chain centred on the origin, well inside any cell used here.
pub(super) fn chain(n: usize) -> Vec<Point> {
    let beads = helix(n, Point::zeros());
    let centre = beads.iter().sum::<Point>() / n as f64;
    beads.iter().map(|p| p - centre).collect()
}

/// A chain laid out from near the +x face, so that it wraps around the cell.
///
/// Consecutive minimum-image distances remain 1 Å, but the stored coordinates of the bonded pairs
/// straddling a face differ by nearly a full box length.
pub(super) fn wrapped_chain(n: usize, box_length: f64) -> Vec<Point> {
    let beads = wrap(
        helix(n, Point::new(0.5 * box_length - 1.5, 0.0, 0.0)),
        box_length,
    );
    assert!(
        beads
            .windows(2)
            .any(|b| (b[1] - b[0]).amax() > 0.5 * box_length),
        "fixture is vacuous: no bond straddles a cell face"
    );
    beads
}

/// A chain whose halves each reach beyond half the box along a Cartesian axis.
///
/// This is the regime where a minimum-image convention applied per atom, rather than followed
/// along the bonds, folds part of a rotated sub-tree into the wrong periodic image.
pub(super) fn overlong_chain(n: usize, box_length: f64) -> Vec<Point> {
    let beads = helix(n, Point::new(0.5 * box_length - 1.5, 0.0, 0.0));
    let middle = beads[n / 2];
    let reach = beads
        .iter()
        .map(|p| (p - middle).amax())
        .fold(0.0, f64::max);
    assert!(
        reach > 0.5 * box_length,
        "fixture is vacuous: the chain reaches only {reach:.1} Å from its middle bead, \
         short of the {:.1} Å half-box",
        0.5 * box_length
    );
    wrap(beads, box_length)
}

/// Minimum-image length of every intramolecular bond of the (single) chain.
pub(super) fn bond_lengths(context: &Backend) -> Vec<f64> {
    let start = context.groups()[0].start();
    context
        .topology_ref()
        .moleculekind(crate::group::MoleculeId::new(0))
        .bonds()
        .iter()
        .map(|bond| {
            let [i, j] = [bond.index()[0], bond.index()[1]];
            context.get_distance_squared(start + i, start + j).sqrt()
        })
        .collect()
}

/// Angle at each interior bead of the (linear) chain, i.e. ∠(i−1, i, i+1).
///
/// Bond lengths alone cannot detect a wrong rotation axis: any rotation about any axis through the
/// pivot conserves every distance from the pivot. The bond angles are what a crankshaft must
/// conserve, and what a mis-oriented axis destroys.
pub(super) fn bond_angles(context: &Backend) -> Vec<f64> {
    let group = &context.groups()[0];
    let (start, n) = (group.start(), group.len());
    let bond_vector = |i: usize, j: usize| {
        context
            .cell()
            .distance(&context.position(start + i), &context.position(start + j))
    };
    (1..n - 1)
        .map(|i| bond_vector(i - 1, i).angle(&bond_vector(i + 1, i)))
        .collect()
}

/// Apply a proposed move to the context.
pub(super) fn apply(context: &mut Backend, proposal: &ProposedMove) {
    proposal
        .apply_with_backup(context)
        .expect("transform should apply");
}

/// Whether a proposal leaves every particle it selects exactly where it was.
///
/// True when every selected bead lies on the rotation axis: the rotation is then the identity on
/// the selected set, so the move is accepted with ΔU = 0 without having sampled anything.
///
/// Measured as the longest lever arm — the perpendicular distance to the axis, recovered from the
/// displacement a rotation by `angle` produced, `2·d·sin(angle/2)`. Going through the lever arm
/// rather than the displacement keeps the verdict a property of the *selection*, so that an
/// unluckily small angle cannot make a sound proposal look degenerate.
pub(super) fn is_noop(context: &Backend, proposal: &ProposedMove) -> bool {
    let Transform::SetPositions(positions, selection) = proposal.transform() else {
        panic!("rotation moves propose new positions");
    };
    let crate::propagate::MoveTarget::Group(group) = proposal.target() else {
        panic!("rotation moves target a single group");
    };
    let crate::propagate::Displacement::Angle(angle) = proposal.displacement() else {
        panic!("rotation moves report an angular displacement");
    };
    let chord = 2.0 * (0.5 * angle).sin().abs();
    assert!(
        chord > 1e-6,
        "degenerate rotation angle in fixture: {angle}"
    );

    context.groups()[*group]
        .select(selection, context.topology_ref())
        .expect("selection should resolve")
        .iter()
        .zip(positions)
        .all(|(&i, new)| {
            let displacement = context.cell().distance(new, &context.position(i)).norm();
            displacement / chord < 1e-9
        })
}

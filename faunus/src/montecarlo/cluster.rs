// Copyright 2024 Mikael Lund
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

use super::{find_molecule_id, random_group, Bias};
use crate::cell::BoundaryConditions;
use crate::group::ParticleSelection;
use crate::montecarlo::NewOld;
use crate::propagate::{tagged_yaml, Displacement, MoveProposal, MoveTarget, ProposedMove};
use crate::transform::{random_displacement, random_quaternion, random_unit_vector, Transform};
use crate::{Change, Context, GroupChange, Point};
use rand::RngCore;
use serde::{Deserialize, Serialize};

const fn default_true() -> bool {
    true
}

/// Rigid-body cluster move.
///
/// Picks a random seed molecule, grows a cluster by BFS using a distance threshold,
/// then translates and rotates the entire cluster as a rigid unit. Ensures detailed
/// balance via bias rejection: if the cluster composition changes after the move,
/// the move is rejected with certainty.
///
/// Two clustering criteria are supported via the `com` flag:
/// - `com: true` (default): mass-center to mass-center distance — O(N_mol) per BFS step, fast.
/// - `com: false`: closest bead-to-bead distance — O(N_mol × N_beads²), physically transferable
///   (threshold directly means surface separation, e.g. 6 Å = in contact).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterMove {
    /// Name of the molecule type to cluster.
    #[serde(rename = "molecule")]
    molecule_name: String,
    /// Resolved molecule id (filled in `finalize`).
    #[serde(skip)]
    molecule_id: usize,
    /// Maximum translational displacement (Å).
    #[serde(alias = "dp")]
    max_displacement: f64,
    /// Maximum rotational displacement (radians).
    #[serde(alias = "dprot")]
    max_angle: f64,
    /// Distance threshold for cluster membership (Å).
    /// Interpreted as COM-to-COM when `com: true`, or closest bead-to-bead when `com: false`.
    threshold: f64,
    /// Move selection weight.
    #[serde(skip_serializing, default = "crate::propagate::default_weight")]
    pub(crate) weight: f64,
    /// Repeat the move N times per sweep.
    #[serde(default = "crate::propagate::default_repeat")]
    #[serde(skip_serializing)]
    pub(crate) repeat: usize,
    /// If true (default), use mass-center distance for cluster criterion.
    /// If false, use closest bead-to-bead distance.
    #[serde(default = "default_true")]
    com: bool,
    /// Set by `propose_move`, read by `bias`: true if cluster composition is stable after move.
    #[serde(skip)]
    cluster_stable: bool,
}

impl crate::Info for ClusterMove {
    fn short_name(&self) -> Option<&'static str> {
        Some("cluster")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Rigid-body cluster move")
    }
}

impl ClusterMove {
    pub(crate) fn finalize(&mut self, context: &impl Context) -> anyhow::Result<()> {
        self.molecule_id = find_molecule_id(context, &self.molecule_name, "ClusterMove")?;
        anyhow::ensure!(self.threshold > 0.0, "ClusterMove: threshold must be positive");
        anyhow::ensure!(
            self.max_displacement >= 0.0,
            "ClusterMove: max_displacement must be non-negative"
        );
        anyhow::ensure!(
            self.max_angle >= 0.0,
            "ClusterMove: max_angle must be non-negative"
        );
        Ok(())
    }

    /// True if groups `gi` and `gj` are within the threshold distance.
    fn in_contact(&self, gi: usize, gj: usize, context: &impl Context) -> bool {
        let threshold_sq = self.threshold * self.threshold;
        if self.com {
            let ci = context.groups()[gi].mass_center().copied().unwrap_or_default();
            let cj = context.groups()[gj].mass_center().copied().unwrap_or_default();
            context.cell().distance_squared(&ci, &cj) <= threshold_sq
        } else {
            let ai = context.groups()[gi]
                .select(&ParticleSelection::Active, context.topology_ref())
                .unwrap_or_default();
            let aj = context.groups()[gj]
                .select(&ParticleSelection::Active, context.topology_ref())
                .unwrap_or_default();
            ai.iter().any(|&a| {
                aj.iter()
                    .any(|&b| context.get_distance_squared(a, b) <= threshold_sq)
            })
        }
    }

    /// Grow a cluster from `seed` via BFS. Returns sorted group indices.
    fn find_cluster(&self, seed: usize, context: &impl Context) -> Vec<usize> {
        let mut cluster = vec![seed];
        let mut pool: Vec<usize> = context
            .groups()
            .iter()
            .enumerate()
            .filter(|(i, g)| g.molecule() == self.molecule_id && *i != seed)
            .map(|(i, _)| i)
            .collect();

        let mut ci = 0;
        while ci < cluster.len() {
            pool.retain(|&j| {
                if self.in_contact(cluster[ci], j, context) {
                    cluster.push(j);
                    false
                } else {
                    true
                }
            });
            ci += 1;
        }
        cluster.sort_unstable();
        cluster
    }
}

impl<T: Context> MoveProposal<T> for ClusterMove {
    fn propose_move(&mut self, context: &T, rng: &mut dyn RngCore) -> Option<ProposedMove> {
        // 1. Pick random seed molecule
        let seed = random_group(context, rng, self.molecule_id)?;

        // 2. Grow cluster via BFS
        let cluster = self.find_cluster(seed, context);

        // 3. Compute unweighted cluster mass center
        let cluster_com = {
            let coms: Vec<Point> = cluster
                .iter()
                .map(|&i| context.groups()[i].mass_center().copied().unwrap_or_default())
                .collect();
            let n = coms.len() as f64;
            coms.iter().fold(Point::zeros(), |acc, p| acc + p) / n
        };

        // 4. Random translation and rotation
        let translation =
            random_unit_vector(rng) * random_displacement(rng, self.max_displacement);
        let (quaternion, angle) = random_quaternion(rng, self.max_angle);

        // 5. Pre-compute bias: mentally apply transform to cluster positions and
        //    check whether any non-cluster molecule would enter the threshold.
        //    If yes → cluster composition would change → bias = infinity (reject).
        let threshold_sq = self.threshold * self.threshold;
        let transform_point = |p: Point| -> Point {
            cluster_com + quaternion * (p - cluster_com) + translation
        };

        self.cluster_stable = if self.com {
            // COM mode: check new cluster COMs against non-cluster COMs
            let new_coms: Vec<Point> = cluster
                .iter()
                .map(|&i| {
                    transform_point(
                        context.groups()[i].mass_center().copied().unwrap_or_default(),
                    )
                })
                .collect();
            !context
                .groups()
                .iter()
                .enumerate()
                .filter(|(i, g)| g.molecule() == self.molecule_id && !cluster.contains(i))
                .any(|(_, g)| {
                    let other = g.mass_center().copied().unwrap_or_default();
                    new_coms.iter().any(|nc| {
                        context.cell().distance_squared(nc, &other) <= threshold_sq
                    })
                })
        } else {
            // Bead mode: check new cluster bead positions against non-cluster beads
            let new_positions: Vec<Point> = cluster
                .iter()
                .flat_map(|&i| {
                    context.groups()[i]
                        .select(&ParticleSelection::Active, context.topology_ref())
                        .unwrap_or_default()
                        .into_iter()
                        .map(|abs| transform_point(context.position(abs)))
                })
                .collect();
            !context
                .groups()
                .iter()
                .enumerate()
                .filter(|(i, g)| g.molecule() == self.molecule_id && !cluster.contains(i))
                .any(|(_, g)| {
                    let other_beads = g
                        .select(&ParticleSelection::Active, context.topology_ref())
                        .unwrap_or_default();
                    new_positions.iter().any(|np| {
                        other_beads.iter().any(|&b| {
                            let bp = context.position(b);
                            context.cell().distance_squared(np, &bp) <= threshold_sq
                        })
                    })
                })
        };

        // 6. Build change and proposed move
        let change =
            Change::Groups(cluster.iter().map(|&i| (i, GroupChange::RigidBody)).collect());

        Some(ProposedMove {
            change,
            displacement: Displacement::AngleDistance(angle, translation),
            transform: Transform::ClusterTransform {
                groups: cluster,
                translation,
                rotation: Some((cluster_com, quaternion)),
            },
            target: MoveTarget::System,
        })
    }

    fn bias(&self, _change: &Change, _energies: &NewOld<f64>) -> Bias {
        if self.cluster_stable {
            Bias::None
        } else {
            // Cluster composition changed after move → reject to maintain detailed balance
            Bias::Energy(f64::INFINITY)
        }
    }

    fn to_yaml(&self) -> Option<serde_yml::Value> {
        tagged_yaml("ClusterMove", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use std::path::Path;

    #[test]
    fn test_parse_com_mode_default() {
        let s = "{ molecule: MOL1, dp: 5.0, dprot: 0.3, threshold: 30.0 }";
        let m: ClusterMove = serde_yml::from_str(s).unwrap();
        assert_eq!(m.molecule_name, "MOL1");
        assert_eq!(m.max_displacement, 5.0);
        assert_eq!(m.max_angle, 0.3);
        assert_eq!(m.threshold, 30.0);
        assert!(m.com, "com should default to true");
        assert_eq!(m.weight, 1.0);
        assert_eq!(m.repeat, 1);
    }

    #[test]
    fn test_parse_bead_mode_explicit() {
        let s = "{ molecule: MOL1, dp: 5.0, dprot: 0.3, threshold: 6.0, com: false }";
        let m: ClusterMove = serde_yml::from_str(s).unwrap();
        assert!(!m.com, "com should be false");
        assert_eq!(m.threshold, 6.0);
    }

    #[test]
    fn test_parse_dp_alias() {
        // Verify "dp" and "dprot" aliases work
        let s1 = "{ molecule: M, dp: 2.0, dprot: 0.5, threshold: 10.0 }";
        let s2 = "{ molecule: M, max_displacement: 2.0, max_angle: 0.5, threshold: 10.0 }";
        let m1: ClusterMove = serde_yml::from_str(s1).unwrap();
        let m2: ClusterMove = serde_yml::from_str(s2).unwrap();
        assert_eq!(m1.max_displacement, m2.max_displacement);
        assert_eq!(m1.max_angle, m2.max_angle);
    }

    #[test]
    fn test_cluster_move_finalize() {
        let mut rng = rand::thread_rng();
        let context = Backend::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap();

        let mut m: ClusterMove =
            serde_yml::from_str("{ molecule: MOL, dp: 5.0, dprot: 0.3, threshold: 30.0 }")
                .unwrap();
        m.finalize(&context).unwrap();
        assert_eq!(m.molecule_id, 0); // MOL is the first molecule in topology_pass.yaml
    }

    #[test]
    fn test_cluster_move_finalize_unknown_molecule() {
        let mut rng = rand::thread_rng();
        let context = Backend::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap();

        let mut m: ClusterMove =
            serde_yml::from_str("{ molecule: DOESNOTEXIST, dp: 5.0, dprot: 0.3, threshold: 30.0 }")
                .unwrap();
        assert!(m.finalize(&context).is_err());
    }
}

// Copyright 2026 Mikael Lund
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
use crate::auxiliary::BlockAverage;
use crate::cell::BoundaryConditions;
use crate::group::{MoleculeId, ParticleSelection};
use crate::montecarlo::NewOld;
use crate::propagate::{tagged_yaml, MoveProposal, ProposedMove};
use crate::transform::{random_displacement, random_quaternion, random_unit_vector};
use crate::{Change, Context, Point, UnitQuaternion};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const fn default_true() -> bool {
    true
}

/// Rigid-body cluster move.
///
/// Picks a random seed molecule, grows a cluster by BFS using a distance threshold, then moves the
/// whole cluster as a rigid unit. Detailed balance is enforced by rejecting any move that would
/// change the cluster membership (`bias = ∞`); combined with symmetric roto-translation proposals,
/// this samples the correct Boltzmann distribution. See Dress & Krauth,
/// <https://doi.org/10.1088/0305-4470/28/23/001>.
///
/// The cluster is grown in *unwrapped* coordinates — each recruited neighbour is placed by minimum
/// image relative to the molecule that recruited it — so the rotation pivot (the cluster mass
/// center) is well defined and the whole cluster rotates correctly even when it spans the periodic
/// boundary. A single molecule type means equal group masses, so the unweighted centroid of the
/// group mass centers *is* the cluster mass center.
///
/// Two clustering criteria are selectable via the `use_com` flag:
/// - `use_com: true` (default): mass-center to mass-center distance — O(N_mol) per BFS step, fast.
/// - `use_com: false`: closest bead-to-bead distance — O(N_mol × N_beads²), physically transferable
///   (the threshold is a surface separation, e.g. 6 Å = in contact).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterMove {
    /// Name of the molecule type to cluster.
    #[serde(rename = "molecule")]
    molecule_name: String,
    /// Resolved molecule id (filled in `finalize`).
    #[serde(skip)]
    molecule_id: MoleculeId,
    /// Maximum translational displacement (Å).
    #[serde(alias = "dp")]
    max_displacement: f64,
    /// Maximum rotational displacement (radians).
    #[serde(alias = "dprot")]
    max_angle: f64,
    /// Distance threshold for cluster membership (Å).
    /// Interpreted as COM-to-COM when `use_com: true`, or closest bead-to-bead when `use_com: false`.
    threshold: f64,
    /// Move selection weight.
    #[serde(skip_serializing, default = "crate::propagate::default_weight")]
    pub(crate) weight: f64,
    /// Repeat the move N times per sweep.
    #[serde(default = "crate::propagate::default_repeat")]
    #[serde(skip_serializing)]
    pub(crate) repeat: usize,
    /// If true (default), use mass-center distance for the cluster criterion.
    /// If false, use closest bead-to-bead distance.
    #[serde(default = "default_true")]
    use_com: bool,
    /// Set by `propose_move`, read by `bias`: true if the cluster membership is unchanged by the move.
    #[serde(skip)]
    cluster_stable: bool,
    /// Runtime diagnostics reported to `output.yaml`.
    #[serde(skip)]
    stats: ClusterStats,
}

/// Running diagnostics for the cluster move, mirroring the C++ Faunus reporting.
#[derive(Clone, Debug, Default)]
struct ClusterStats {
    /// Mean cluster size ⟨N⟩ with standard error.
    size: BlockAverage,
    /// Mean-square translational and rotational displacement (0 on rejected trials).
    msd_translation: BlockAverage,
    msd_rotation: BlockAverage,
    /// Squared displacement of the pending trial, banked by `on_trial_outcome`.
    pending_sq_translation: f64,
    pending_sq_rotation: f64,
    n_proposals: u64,
    n_bias_rejected: u64,
}

impl_info!(ClusterMove, "cluster", "Rigid-body cluster move");

impl ClusterMove {
    pub(crate) fn finalize(&mut self, context: &impl Context) -> anyhow::Result<()> {
        self.molecule_id = find_molecule_id(context, &self.molecule_name, "ClusterMove")?;
        anyhow::ensure!(
            context
                .topology_ref()
                .moleculekind(self.molecule_id)
                .has_com(),
            "ClusterMove requires a molecule with a mass center; '{}' has none",
            self.molecule_name
        );
        anyhow::ensure!(
            self.threshold > 0.0,
            "ClusterMove: threshold must be positive"
        );
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

    /// Mass center of group `gi` (all cluster groups are molecular and have one).
    fn group_com(&self, gi: usize, context: &impl Context) -> Point {
        context.groups()[gi]
            .mass_center()
            .copied()
            .expect("cluster molecule has a mass center")
    }

    /// True if groups `gi` and `gj` are within the threshold distance.
    fn in_contact(&self, gi: usize, gj: usize, context: &impl Context) -> bool {
        let threshold_sq = self.threshold * self.threshold;
        if self.use_com {
            let ci = self.group_com(gi, context);
            let cj = self.group_com(gj, context);
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

    /// Grow a cluster from `seed` via BFS.
    ///
    /// Returns the sorted group indices together with each group's *unwrapped* mass center: the
    /// seed keeps its wrapped mass center, and every recruited member is placed by minimum image
    /// relative to the member that recruited it. The result is contiguous in Euclidean space (for a
    /// non-percolating cluster), independent of the recruitment order, so rotating about the
    /// cluster mass center is well defined regardless of periodic boundaries.
    fn find_cluster(&self, seed: usize, context: &impl Context) -> (Vec<usize>, Vec<Point>) {
        let mut cluster = vec![seed];
        let mut unwrapped = vec![self.group_com(seed, context)];
        let mut pool: Vec<usize> = context
            .groups()
            .iter()
            .enumerate()
            .filter(|(i, g)| g.molecule() == self.molecule_id && *i != seed && !g.is_empty())
            .map(|(i, _)| i)
            .collect();

        let mut ci = 0;
        while ci < cluster.len() {
            let parent = cluster[ci];
            let parent_unwrapped = unwrapped[ci];
            let parent_com = self.group_com(parent, context);
            pool.retain(|&j| {
                if self.in_contact(parent, j, context) {
                    let disp = context
                        .cell()
                        .distance(&self.group_com(j, context), &parent_com);
                    unwrapped.push(parent_unwrapped + disp);
                    cluster.push(j);
                    false
                } else {
                    true
                }
            });
            ci += 1;
        }

        // Sort by group index (keeping the unwrapped centers parallel) for deterministic output.
        let mut order: Vec<usize> = (0..cluster.len()).collect();
        order.sort_unstable_by_key(|&k| cluster[k]);
        (
            order.iter().map(|&k| cluster[k]).collect(),
            order.iter().map(|&k| unwrapped[k]).collect(),
        )
    }

    /// Would the cluster membership survive the proposed move? If a non-cluster molecule of the
    /// same type would fall within the threshold of the moved cluster, membership changes and the
    /// move must be rejected to preserve detailed balance.
    #[allow(clippy::too_many_arguments)]
    fn cluster_membership_stable(
        &self,
        context: &impl Context,
        cluster: &[usize],
        new_mass_centers: &[Point],
        rotation: UnitQuaternion,
    ) -> bool {
        let threshold_sq = self.threshold * self.threshold;
        let members: HashSet<usize> = cluster.iter().copied().collect();
        let outsiders = || {
            context.groups().iter().enumerate().filter(|(i, g)| {
                g.molecule() == self.molecule_id && !members.contains(i) && !g.is_empty()
            })
        };

        if self.use_com {
            !outsiders().any(|(_, g)| {
                let other = g
                    .mass_center()
                    .copied()
                    .expect("molecule has a mass center");
                new_mass_centers
                    .iter()
                    .any(|nc| context.cell().distance_squared(nc, &other) <= threshold_sq)
            })
        } else {
            // Moved bead positions, consistent with `Transform::ClusterTransform`: each bead is
            // rotated about its own group mass center, then placed at the group's new mass center.
            let moved_beads: Vec<Point> = cluster
                .iter()
                .zip(new_mass_centers)
                .flat_map(|(&gi, &new_com)| {
                    let old_com = self.group_com(gi, context);
                    context.groups()[gi]
                        .select(&ParticleSelection::Active, context.topology_ref())
                        .unwrap_or_default()
                        .into_iter()
                        .map(move |b| {
                            new_com
                                + rotation * context.cell().distance(&context.position(b), &old_com)
                        })
                })
                .collect();
            !outsiders().any(|(_, g)| {
                let beads = g
                    .select(&ParticleSelection::Active, context.topology_ref())
                    .unwrap_or_default();
                moved_beads.iter().any(|mb| {
                    beads.iter().any(|&b| {
                        context.cell().distance_squared(mb, &context.position(b)) <= threshold_sq
                    })
                })
            })
        }
    }
}

impl<T: Context> MoveProposal<T> for ClusterMove {
    fn propose_move(&mut self, context: &T, rng: &mut dyn RngCore) -> Option<ProposedMove> {
        let seed = random_group(context, rng, self.molecule_id)?;
        let (cluster, unwrapped) = self.find_cluster(seed, context);

        self.stats.size.add(cluster.len() as f64);
        self.stats.n_proposals += 1;

        // Cluster mass center in unwrapped space (= centroid, since one molecule type ⇒ equal mass).
        let pivot = unwrapped.iter().fold(Point::zeros(), |acc, p| acc + p) / cluster.len() as f64;

        let translation = random_unit_vector(rng) * random_displacement(rng, self.max_displacement);
        let (quaternion, angle) = random_quaternion(rng, self.max_angle);
        let rotation = (self.max_angle > 0.0).then_some(quaternion);

        // Each group's new mass center: rotate its unwrapped center about the pivot, translate, wrap.
        let new_mass_centers: Vec<Point> = unwrapped
            .iter()
            .map(|&u| {
                let mut p = pivot + quaternion * (u - pivot) + translation;
                context.cell().boundary(&mut p);
                p
            })
            .collect();

        self.cluster_stable =
            self.cluster_membership_stable(context, &cluster, &new_mass_centers, quaternion);
        if !self.cluster_stable {
            self.stats.n_bias_rejected += 1;
        }
        self.stats.pending_sq_translation = translation.norm_squared();
        self.stats.pending_sq_rotation = angle * angle;

        Some(ProposedMove::cluster(
            cluster,
            new_mass_centers,
            rotation,
            angle,
            translation,
        ))
    }

    fn bias(&self, _change: &Change, _energies: &NewOld<f64>) -> Bias {
        if self.cluster_stable {
            Bias::None
        } else {
            // Membership would change → reject with certainty to maintain detailed balance.
            Bias::Energy(f64::INFINITY)
        }
    }

    fn on_trial_outcome(&mut self, accepted: bool) {
        let (dr2, dtheta2) = if accepted {
            (
                self.stats.pending_sq_translation,
                self.stats.pending_sq_rotation,
            )
        } else {
            (0.0, 0.0)
        };
        self.stats.msd_translation.add(dr2);
        self.stats.msd_rotation.add(dtheta2);
    }

    fn to_yaml(&self) -> Option<serde_yml::Value> {
        let mut value = tagged_yaml("ClusterMove", self)?;
        if let serde_yml::Value::Tagged(tagged) = &mut value {
            if let serde_yml::Value::Mapping(map) = &mut tagged.value {
                let rate = if self.stats.n_proposals > 0 {
                    self.stats.n_bias_rejected as f64 / self.stats.n_proposals as f64
                } else {
                    0.0
                };
                map.insert("cluster_size".into(), self.stats.size.to_yaml()?);
                map.insert("bias_rejection_rate".into(), rate.into());
                map.insert(
                    "rmsd_translation".into(),
                    self.stats.msd_translation.mean().sqrt().into(),
                );
                map.insert(
                    "rmsd_rotation".into(),
                    self.stats.msd_rotation.mean().sqrt().into(),
                );
            }
        }
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::context::WithSimulationCell;
    use crate::group::GroupCollection;
    use float_cmp::assert_approx_eq;
    use rand::{rngs::StdRng, SeedableRng};
    use std::path::Path;

    /// Build a system of single-bead `MOL` molecules at the given (box-centred) positions.
    fn system(positions: &[[f64; 3]]) -> Backend {
        let inserts = positions
            .iter()
            .map(|p| format!("[{}, {}, {}]", p[0], p[1], p[2]))
            .collect::<Vec<_>>()
            .join(", ");
        let yaml = format!(
            r#"
atoms:
  - {{name: B, mass: 1.0, charge: 0.0, sigma: 1.0}}
molecules:
  - name: MOL
    atoms: [B]
system:
  cell: !Cuboid [30.0, 30.0, 30.0]
  medium: {{permittivity: !Vacuum, temperature: 300.0}}
  energy: {{}}
  blocks:
    - molecule: MOL
      N: {}
      insert: !Manual [{}]
propagate: {{seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}}
"#,
            positions.len(),
            inserts
        );
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml).unwrap();
        Backend::new(tmp.path(), None, &mut rand::thread_rng()).unwrap()
    }

    fn cluster_move(yaml: &str, context: &Backend) -> ClusterMove {
        let mut m: ClusterMove = serde_yml::from_str(yaml).unwrap();
        m.finalize(context).unwrap();
        m
    }

    /// All pairwise minimum-image distances between group mass centers.
    fn pairwise_distances(context: &Backend) -> Vec<f64> {
        let coms: Vec<Point> = context
            .groups()
            .iter()
            .map(|g| g.mass_center().copied().unwrap())
            .collect();
        let mut out = Vec::new();
        for i in 0..coms.len() {
            for j in (i + 1)..coms.len() {
                out.push(context.cell().distance(&coms[i], &coms[j]).norm());
            }
        }
        out
    }

    #[test]
    fn find_cluster_grows_by_single_linkage() {
        // 0—3—6 are linked (gaps of 3 < threshold 4); 12 is isolated.
        let context = system(&[
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ]);
        let m = cluster_move(
            "{molecule: MOL, dp: 1.0, dprot: 0.1, threshold: 4.0}",
            &context,
        );
        let (cluster, _) = m.find_cluster(0, &context);
        assert_eq!(cluster, vec![0, 1, 2], "transitive single-linkage cluster");

        let (isolated, _) = m.find_cluster(3, &context);
        assert_eq!(isolated, vec![3], "the far molecule clusters alone");
    }

    #[test]
    fn bias_rejects_membership_change() {
        // Cluster {0,1}; outsider 2 sits at x=10.
        let context = system(&[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [10.0, 0.0, 0.0]]);
        let m = cluster_move(
            "{molecule: MOL, dp: 1.0, dprot: 0.1, threshold: 4.0}",
            &context,
        );
        let cluster = vec![0, 1];
        let id = UnitQuaternion::identity();

        // Shift the cluster +5 x → member reaches x=8, within 4 Å of the outsider ⇒ membership grows.
        let toward = vec![Point::new(5.0, 0.0, 0.0), Point::new(8.0, 0.0, 0.0)];
        assert!(!m.cluster_membership_stable(&context, &cluster, &toward, id));

        // Shift the cluster away ⇒ membership unchanged.
        let away = vec![Point::new(-5.0, 0.0, 0.0), Point::new(-2.0, 0.0, 0.0)];
        assert!(m.cluster_membership_stable(&context, &cluster, &away, id));
    }

    #[test]
    fn rigid_motion_preserves_distances_across_pbc() {
        // A 12-molecule cluster straddling the ±15 boundary. A rigid roto-translation must preserve
        // every intra-cluster minimum-image distance exactly, validating the unwrapped-space pivot
        // and rotation for a box-spanning cluster.
        let positions = [
            [14.0, 0.0, 0.0],
            [14.0, 2.0, 0.0],
            [13.0, 0.0, 2.0],
            [-14.0, 0.0, 0.0],
            [-14.0, 2.0, 0.0],
            [-13.0, 0.0, 2.0],
            [14.0, -2.0, 1.0],
            [-14.0, -2.0, 1.0],
            [13.0, 2.0, -2.0],
            [-13.0, 2.0, -2.0],
            [14.0, 0.0, -3.0],
            [-14.0, 0.0, -3.0],
        ];
        let mut context = system(&positions);
        // Huge threshold ⇒ every molecule joins one cluster regardless of geometry.
        let mut m = cluster_move(
            "{molecule: MOL, dp: 5.0, dprot: 1.0, threshold: 100.0}",
            &context,
        );

        let before = pairwise_distances(&context);
        let proposed = m
            .propose_move(&context, &mut StdRng::seed_from_u64(7))
            .unwrap();
        match proposed.change() {
            Change::Groups(v) => assert_eq!(v.len(), 12, "all 12 molecules in one cluster"),
            other => panic!("expected Groups, got {other:?}"),
        }
        proposed.apply_with_backup(&mut context).unwrap();

        let after = pairwise_distances(&context);
        for (b, a) in before.iter().zip(&after) {
            assert_approx_eq!(f64, *b, *a, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_parse_com_mode_default() {
        let s = "{ molecule: MOL1, dp: 5.0, dprot: 0.3, threshold: 30.0 }";
        let m: ClusterMove = serde_yml::from_str(s).unwrap();
        assert_eq!(m.molecule_name, "MOL1");
        assert_eq!(m.max_displacement, 5.0);
        assert_eq!(m.max_angle, 0.3);
        assert_eq!(m.threshold, 30.0);
        assert!(m.use_com, "use_com should default to true");
        assert_eq!(m.weight, 1.0);
        assert_eq!(m.repeat, 1);
    }

    #[test]
    fn test_parse_bead_mode_explicit() {
        let s = "{ molecule: MOL1, dp: 5.0, dprot: 0.3, threshold: 6.0, use_com: false }";
        let m: ClusterMove = serde_yml::from_str(s).unwrap();
        assert!(!m.use_com, "use_com should be false");
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
    fn finalize_rejects_molecule_without_mass_center() {
        // `MOL` in topology_pass.yaml has `has_com: false`; the rotation pivot needs a mass center,
        // so the move must fail at startup rather than panic mid-simulation.
        let mut rng = rand::thread_rng();
        let context = Backend::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap();

        let mut m: ClusterMove =
            serde_yml::from_str("{ molecule: MOL, dp: 5.0, dprot: 0.3, threshold: 30.0 }").unwrap();
        assert!(m.finalize(&context).is_err());
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

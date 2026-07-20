// Copyright 2025 Mikael Lund
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

//! Polymer shape analysis via the mass-weighted gyration tensor.
//!
//! Computes size and shape anisotropy descriptors from eigenvalues of the
//! gyration tensor, streaming per-step data to an optional file and reporting
//! averages in YAML output.

use super::{Analyze, Frequency, Sampling};
use crate::auxiliary::{ColumnWriter, MappingExt, WeightedMean};
use crate::cell::BoundaryConditions;
use crate::geometry::GyrationTensor;
use crate::selection::{CachedSelection, Groups, Selection};
use crate::ObserveContext;
use anyhow::Result;
use derive_more::Debug;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// YAML builder for [`ShapeAnalysis`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShapeAnalysisBuilder {
    pub selection: Selection,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<PathBuf>,
    pub frequency: Frequency,
}

impl ShapeAnalysisBuilder {
    pub fn apply_output_dir(&mut self, dir: &std::path::Path) -> Result<()> {
        crate::analysis::prefix_opt(&mut self.file, dir)
    }

    pub fn build(&self, context: &impl ObserveContext) -> Result<ShapeAnalysis> {
        let topology = context.topology_ref();
        let groups = context.groups();
        let group_indices = self
            .selection
            .resolve_groups(topology, groups, &|i| context.atom_kind(i));
        if group_indices.is_empty() {
            anyhow::bail!(
                "ShapeAnalysis: selection '{}' matched no groups",
                self.selection.source()
            );
        }

        let stream = if let Some(path) = &self.file {
            if group_indices.len() > 1 {
                anyhow::bail!(
                    "ShapeAnalysis: file output requires a single-molecule selection, \
                     but '{}' matched {} groups",
                    self.selection.source(),
                    group_indices.len()
                );
            }
            Some(ColumnWriter::open(
                path,
                &["step", "Rg", "Sxx", "Sxy", "Sxz", "Syy", "Syz", "Szz"],
            )?)
        } else {
            None
        };

        Ok(ShapeAnalysis {
            selection: CachedSelection::groups(self.selection.clone()),
            stream,
            sampling: Sampling::new(self.frequency),
            num_groups_sampled: 0,
            gyration_radius_squared: WeightedMean::new(),
            gyration_radius: WeightedMean::new(),
            end_to_end_squared: WeightedMean::new(),
            asphericity: WeightedMean::new(),
            acylindricity: WeightedMean::new(),
            relative_shape_anisotropy: WeightedMean::new(),
            prolateness: WeightedMean::new(),
            westin_cl: WeightedMean::new(),
            westin_cp: WeightedMean::new(),
            westin_cs: WeightedMean::new(),
            tensor_xx: WeightedMean::new(),
            tensor_xy: WeightedMean::new(),
            tensor_xz: WeightedMean::new(),
            tensor_yy: WeightedMean::new(),
            tensor_yz: WeightedMean::new(),
            tensor_zz: WeightedMean::new(),
        })
    }
}

/// Polymer shape analysis via the mass-weighted gyration tensor.
#[derive(Debug)]
pub struct ShapeAnalysis {
    selection: CachedSelection<Groups>,
    #[debug(skip)]
    stream: Option<ColumnWriter>,
    /// Frequency and frame count, owned by the framework.
    sampling: Sampling,
    /// Gyration tensors accumulated, one per matching molecule per frame. This, not `num_samples`,
    /// is the count behind every mean below.
    num_groups_sampled: usize,
    gyration_radius_squared: WeightedMean,
    gyration_radius: WeightedMean,
    end_to_end_squared: WeightedMean,
    asphericity: WeightedMean,
    acylindricity: WeightedMean,
    relative_shape_anisotropy: WeightedMean,
    prolateness: WeightedMean,
    westin_cl: WeightedMean,
    westin_cp: WeightedMean,
    westin_cs: WeightedMean,
    tensor_xx: WeightedMean,
    tensor_xy: WeightedMean,
    tensor_xz: WeightedMean,
    tensor_yy: WeightedMean,
    tensor_yz: WeightedMean,
    tensor_zz: WeightedMean,
}

/// Minimum Rg² to guard against division by zero.
const RG2_EPSILON: f64 = 1e-20;

/// Compute the mass-weighted gyration tensor for a group of particles.
fn gyration_tensor(
    group: &crate::group::Group,
    context: &impl ObserveContext,
) -> Option<GyrationTensor> {
    let com = group.mass_center()?;
    if group.len() < 2 {
        return None;
    }
    let positions_masses = group
        .iter_active()
        .map(|i| (context.position(i), context.atom_mass(i)));
    GyrationTensor::from_positions_masses_com(positions_masses, com, context.cell())
}

/// Compute shape descriptors from sorted eigenvalues λ₁ ≤ λ₂ ≤ λ₃.
struct ShapeDescriptors {
    asphericity: f64,
    acylindricity: f64,
    relative_shape_anisotropy: f64,
    prolateness: f64,
    westin_cl: f64,
    westin_cp: f64,
    westin_cs: f64,
}

fn compute_descriptors(evals: &[f64; 3], rg_squared: f64) -> Option<ShapeDescriptors> {
    if rg_squared < RG2_EPSILON {
        return None;
    }
    let [l1, l2, l3] = *evals;
    let rg4 = rg_squared * rg_squared;
    let rg6 = rg4 * rg_squared;

    let b = l3 - (l1 + l2) / 2.0;
    let c = l2 - l1;
    let kappa2 = (b * b + 0.75 * c * c) / rg4;

    let l_mean = rg_squared / 3.0;
    let s = 27.0 * (l1 - l_mean) * (l2 - l_mean) * (l3 - l_mean) / rg6;

    let cl = (l3 - l2) / rg_squared;
    let cp = 2.0 * (l2 - l1) / rg_squared;
    let cs = 3.0 * l1 / rg_squared;

    Some(ShapeDescriptors {
        asphericity: b,
        acylindricity: c,
        relative_shape_anisotropy: kappa2,
        prolateness: s,
        westin_cl: cl,
        westin_cp: cp,
        westin_cs: cs,
    })
}

impl_info!(
    ShapeAnalysis,
    "polymer_shape",
    "Polymer shape via gyration tensor",
    "doi:10/d6ff"
);

impl<T: ObserveContext> Analyze<T> for ShapeAnalysis {
    impl_sampling_accessors!();

    fn perform_sample(&mut self, context: &T, step: usize, weight: f64) -> Result<()> {
        let group_indices = self.selection.resolve(context).to_vec();

        for &gi in &group_indices {
            let group = context.group(gi);
            let Some(result) = gyration_tensor(group, context) else {
                continue;
            };

            self.gyration_radius_squared.add(result.rg_squared, weight);
            self.gyration_radius.add(result.rg_squared.sqrt(), weight);

            let first = group.iter_active().next().unwrap();
            let last = group.iter_active().last().unwrap();
            if first != last {
                let re2 = context
                    .cell()
                    .distance_squared(&context.position(first), &context.position(last));
                self.end_to_end_squared.add(re2, weight);
            }

            let s = &result.tensor;
            self.tensor_xx.add(s[(0, 0)], weight);
            self.tensor_xy.add(s[(0, 1)], weight);
            self.tensor_xz.add(s[(0, 2)], weight);
            self.tensor_yy.add(s[(1, 1)], weight);
            self.tensor_yz.add(s[(1, 2)], weight);
            self.tensor_zz.add(s[(2, 2)], weight);

            if let Some(desc) = compute_descriptors(&result.eigenvalues, result.rg_squared) {
                self.asphericity.add(desc.asphericity, weight);
                self.acylindricity.add(desc.acylindricity, weight);
                self.relative_shape_anisotropy
                    .add(desc.relative_shape_anisotropy, weight);
                self.prolateness.add(desc.prolateness, weight);
                self.westin_cl.add(desc.westin_cl, weight);
                self.westin_cp.add(desc.westin_cp, weight);
                self.westin_cs.add(desc.westin_cs, weight);
            }

            if let Some(ref mut stream) = self.stream {
                let s = &result.tensor;
                let rg = result.rg_squared.sqrt();
                stream.write_row(&[
                    &step,
                    &format_args!("{rg:.6}"),
                    &format_args!("{:.6}", s[(0, 0)]),
                    &format_args!("{:.6}", s[(0, 1)]),
                    &format_args!("{:.6}", s[(0, 2)]),
                    &format_args!("{:.6}", s[(1, 1)]),
                    &format_args!("{:.6}", s[(1, 2)]),
                    &format_args!("{:.6}", s[(2, 2)]),
                ])?;
            }

            self.num_groups_sampled += 1;
        }
        Ok(())
    }

    fn results(&self) -> Option<yaml_serde::Value> {
        // A frame in which no molecule matched leaves every accumulator empty.
        if self.num_groups_sampled == 0 {
            return None;
        }
        let mut map = yaml_serde::Mapping::new();
        let rg2 = self.gyration_radius_squared.mean();
        let re2 = self.end_to_end_squared.mean();

        map.try_insert("Rg", rg2.sqrt())?;
        map.try_insert("Re", re2.sqrt())?;
        map.try_insert("Re2/Rg2", re2 / rg2)?;
        map.try_insert("asphericity", self.asphericity.mean())?;
        map.try_insert("acylindricity", self.acylindricity.mean())?;
        map.try_insert(
            "relative_shape_anisotropy",
            self.relative_shape_anisotropy.mean(),
        )?;
        map.try_insert("prolateness", self.prolateness.mean())?;
        map.try_insert("Cl", self.westin_cl.mean())?;
        map.try_insert("Cp", self.westin_cp.mean())?;
        map.try_insert("Cs", self.westin_cs.mean())?;
        map.try_insert("Sxx", self.tensor_xx.mean())?;
        map.try_insert("Sxy", self.tensor_xy.mean())?;
        map.try_insert("Sxz", self.tensor_xz.mean())?;
        map.try_insert("Syy", self.tensor_yy.mean())?;
        map.try_insert("Syz", self.tensor_yz.mean())?;
        map.try_insert("Szz", self.tensor_zz.mean())?;
        map.try_insert("num_samples", self.sampling.num_samples())?;
        map.try_insert("num_groups_sampled", self.num_groups_sampled)?;

        Some(yaml_serde::Value::Mapping(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisBuilder;
    use crate::geometry::GyrationTensor;
    use approx::assert_relative_eq;

    #[test]
    fn deserialize_builder() {
        let yaml = r#"
selection: "molecule polymer"
frequency: !Every 100
"#;
        let builder: ShapeAnalysisBuilder = yaml_serde::from_str(yaml).unwrap();
        assert!(builder.file.is_none());
        assert!(matches!(builder.frequency, Frequency::Every(100)));
    }

    #[test]
    fn deserialize_builder_with_file() {
        let yaml = r#"
selection: "molecule polymer"
file: shape.dat.gz
frequency: !Every 50
"#;
        let builder: ShapeAnalysisBuilder = yaml_serde::from_str(yaml).unwrap();
        assert_eq!(
            builder.file.as_ref().unwrap().to_str().unwrap(),
            "shape.dat.gz"
        );
    }

    #[test]
    fn deserialize_via_analysis_builder() {
        let yaml = r#"
- !PolymerShape
  selection: "molecule polymer"
  frequency: !Every 100
"#;
        let builders: Vec<AnalysisBuilder> = yaml_serde::from_str(yaml).unwrap();
        assert!(matches!(builders[0], AnalysisBuilder::PolymerShape(_)));
    }

    /// Helper: build gyration tensor from equal-mass positions in a single image.
    fn gyration_from_positions(positions: &[nalgebra::Vector3<f64>]) -> GyrationTensor {
        let com = positions.iter().sum::<crate::Point>() / positions.len() as f64;
        GyrationTensor::from_positions_masses_com(
            positions.iter().map(|&p| (p, 1.0)),
            &com,
            &crate::cell::Endless,
        )
        .unwrap()
    }

    #[test]
    fn collinear_rod() {
        // 3 equal-mass particles on x-axis: perfect rod
        let positions = vec![
            nalgebra::Vector3::new(-1.0, 0.0, 0.0),
            nalgebra::Vector3::new(0.0, 0.0, 0.0),
            nalgebra::Vector3::new(1.0, 0.0, 0.0),
        ];
        let result = gyration_from_positions(&positions);
        let desc = compute_descriptors(&result.eigenvalues, result.rg_squared).unwrap();

        assert!(result.rg_squared > 0.0);
        assert_relative_eq!(result.eigenvalues[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.eigenvalues[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(desc.relative_shape_anisotropy, 1.0, epsilon = 1e-10);
        assert!(desc.asphericity >= 0.0);
    }

    #[test]
    fn equilateral_triangle_planar() {
        // 3 particles at equilateral triangle vertices in xy-plane
        let positions = vec![
            nalgebra::Vector3::new(1.0, 0.0, 0.0),
            nalgebra::Vector3::new(-0.5, 3.0_f64.sqrt() / 2.0, 0.0),
            nalgebra::Vector3::new(-0.5, -(3.0_f64.sqrt()) / 2.0, 0.0),
        ];
        let result = gyration_from_positions(&positions);
        let desc = compute_descriptors(&result.eigenvalues, result.rg_squared).unwrap();

        // No z-extent → λ₁ ≈ 0
        assert_relative_eq!(result.eigenvalues[0], 0.0, epsilon = 1e-10);
        // λ₂ ≈ λ₃ for equilateral triangle
        assert_relative_eq!(
            result.eigenvalues[1],
            result.eigenvalues[2],
            epsilon = 1e-10
        );
        // Oblate → prolateness < 0
        assert!(desc.prolateness < 0.0);
    }

    #[test]
    fn regular_tetrahedron_spherical() {
        // 4 particles at regular tetrahedron vertices → isotropic
        let positions = vec![
            nalgebra::Vector3::new(1.0, 1.0, 1.0),
            nalgebra::Vector3::new(1.0, -1.0, -1.0),
            nalgebra::Vector3::new(-1.0, 1.0, -1.0),
            nalgebra::Vector3::new(-1.0, -1.0, 1.0),
        ];
        let result = gyration_from_positions(&positions);
        let desc = compute_descriptors(&result.eigenvalues, result.rg_squared).unwrap();

        // All eigenvalues equal → perfect sphere
        assert_relative_eq!(
            result.eigenvalues[0],
            result.eigenvalues[1],
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.eigenvalues[1],
            result.eigenvalues[2],
            epsilon = 1e-10
        );
        assert_relative_eq!(desc.relative_shape_anisotropy, 0.0, epsilon = 1e-10);
        assert_relative_eq!(desc.asphericity, 0.0, epsilon = 1e-10);
        assert_relative_eq!(desc.acylindricity, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn descriptor_value_ranges() {
        let configs: Vec<Vec<nalgebra::Vector3<f64>>> = vec![
            // Rod
            vec![
                nalgebra::Vector3::new(-2.0, 0.0, 0.0),
                nalgebra::Vector3::new(0.0, 0.0, 0.0),
                nalgebra::Vector3::new(2.0, 0.0, 0.0),
            ],
            // Planar
            vec![
                nalgebra::Vector3::new(1.0, 0.0, 0.0),
                nalgebra::Vector3::new(-0.5, 0.866, 0.0),
                nalgebra::Vector3::new(-0.5, -0.866, 0.0),
            ],
            // Spherical
            vec![
                nalgebra::Vector3::new(1.0, 1.0, 1.0),
                nalgebra::Vector3::new(1.0, -1.0, -1.0),
                nalgebra::Vector3::new(-1.0, 1.0, -1.0),
                nalgebra::Vector3::new(-1.0, -1.0, 1.0),
            ],
            // General asymmetric
            vec![
                nalgebra::Vector3::new(3.0, 0.0, 0.0),
                nalgebra::Vector3::new(0.0, 1.0, 0.0),
                nalgebra::Vector3::new(0.0, 0.0, 0.5),
                nalgebra::Vector3::new(-1.0, 0.5, 0.2),
            ],
        ];

        for positions in &configs {
            let result = gyration_from_positions(positions);
            let desc = compute_descriptors(&result.eigenvalues, result.rg_squared).unwrap();

            assert!(result.rg_squared > 0.0, "Rg² must be positive");
            assert!(
                desc.asphericity >= -1e-10,
                "asphericity must be non-negative"
            );
            assert!(
                desc.relative_shape_anisotropy >= -1e-10
                    && desc.relative_shape_anisotropy <= 1.0 + 1e-10,
                "κ² must be in [0, 1], got {}",
                desc.relative_shape_anisotropy
            );
            assert!(
                desc.prolateness >= -0.25 - 1e-10 && desc.prolateness <= 2.0 + 1e-10,
                "S must be in [-0.25, 2], got {}",
                desc.prolateness
            );

            let westin_sum = desc.westin_cl + desc.westin_cp + desc.westin_cs;
            assert_relative_eq!(westin_sum, 1.0, epsilon = 1e-10);
            assert!(desc.westin_cl >= -1e-10);
            assert!(desc.westin_cp >= -1e-10);
            assert!(desc.westin_cs >= -1e-10);
        }
    }

    #[test]
    fn two_particles() {
        // Two particles separated by distance d → Rg² = d²/4, κ² = 1
        let d = 4.0;
        let positions = vec![
            nalgebra::Vector3::new(0.0, 0.0, 0.0),
            nalgebra::Vector3::new(d, 0.0, 0.0),
        ];
        let result = gyration_from_positions(&positions);
        let desc = compute_descriptors(&result.eigenvalues, result.rg_squared).unwrap();

        assert_relative_eq!(result.rg_squared, d * d / 4.0, epsilon = 1e-10);
        assert_relative_eq!(desc.relative_shape_anisotropy, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn coincident_particles_no_panic() {
        // All particles at same position → Rg² ≈ 0, descriptors should return None
        let positions = vec![
            nalgebra::Vector3::new(1.0, 2.0, 3.0),
            nalgebra::Vector3::new(1.0, 2.0, 3.0),
            nalgebra::Vector3::new(1.0, 2.0, 3.0),
        ];
        let result = gyration_from_positions(&positions);
        assert!(result.rg_squared < RG2_EPSILON);
        assert!(compute_descriptors(&result.eigenvalues, result.rg_squared).is_none());
    }

    #[test]
    fn unequal_masses_shift_tensor() {
        let positions = vec![
            nalgebra::Vector3::new(-1.0, 0.0, 0.0),
            nalgebra::Vector3::new(1.0, 0.0, 0.0),
        ];
        let equal = gyration_from_positions(&positions);

        let masses = [1.0, 3.0];
        let total_mass: f64 = masses.iter().sum();
        let com = (masses[0] * positions[0] + masses[1] * positions[1]) / total_mass;

        let mut tensor = nalgebra::Matrix3::<f64>::zeros();
        for (p, &m) in positions.iter().zip(masses.iter()) {
            let r = p - com;
            for i in 0..3 {
                for j in 0..3 {
                    tensor[(i, j)] += m * r[i] * r[j];
                }
            }
        }
        tensor /= total_mass;

        let rg2_weighted = tensor.trace();
        // With unequal masses, COM shifts toward heavier particle → smaller Rg²
        assert!(rg2_weighted < equal.rg_squared);
    }
}

#[cfg(test)]
mod counter_semantics {
    use super::*;
    use crate::backend::Backend;

    /// Two polymers, so a frame contributes one sample but two gyration tensors.
    const TWO_POLYMERS: &str = r#"
atoms:
  - {name: A, mass: 1.0, charge: 0.0, sigma: 1.0}
molecules:
  - name: POLY
    atoms: [A, A, A]
system:
  cell: !Cuboid [30.0, 30.0, 30.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: []
  blocks:
    - molecule: POLY
      N: 2
      insert: !Manual [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                       [8.0, 0.0, 0.0], [9.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, steps: 0, collections: []}
"#;

    /// `num_samples` counts frames everywhere. The per-molecule tally, which is the count behind
    /// every reported mean, is published separately as `num_groups_sampled`.
    #[test]
    fn num_samples_counts_frames_not_molecules() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), TWO_POLYMERS).unwrap();
        let context = Backend::new(tmp.path(), None, &mut rand::thread_rng()).unwrap();

        let builder: ShapeAnalysisBuilder =
            yaml_serde::from_str("{selection: \"molecule POLY\", frequency: !Every 1}").unwrap();
        let mut analysis = builder.build(&context).unwrap();

        for step in 0..3 {
            analysis.sample(&context, step).unwrap();
        }
        assert_eq!(Analyze::<Backend>::num_samples(&analysis), 3, "frames");
        assert_eq!(analysis.num_groups_sampled, 6, "3 frames x 2 molecules");

        let yaml = Analyze::<Backend>::to_yaml(&analysis).unwrap();
        assert_eq!(yaml["num_samples"].as_u64(), Some(3));
        assert_eq!(yaml["num_groups_sampled"].as_u64(), Some(6));
    }
}

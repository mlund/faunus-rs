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

use super::{Analyze, Frequency};
use crate::cell::BoundaryConditions;
use crate::particle::PointParticle;
use crate::selection::Selection;
use crate::Context;
use anyhow::Result;
use average::{Estimate, Mean};
use derive_more::Debug;
use nalgebra::{Matrix3, SymmetricEigen};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;

/// YAML builder for [`ShapeAnalysis`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeAnalysisBuilder {
    pub selection: Selection,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<PathBuf>,
    pub frequency: Frequency,
}

impl ShapeAnalysisBuilder {
    pub fn build(&self, context: &impl Context) -> Result<ShapeAnalysis> {
        let topology = context.topology_ref();
        let groups = context.groups();
        let group_indices = self.selection.resolve_groups(topology, groups);
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
            let mut stream = crate::auxiliary::open_compressed(path)?;
            writeln!(stream, "# step Rg Sxx Sxy Sxz Syy Syz Szz")?;
            Some(stream)
        } else {
            None
        };

        Ok(ShapeAnalysis {
            selection: self.selection.clone(),
            stream,
            frequency: self.frequency,
            num_samples: 0,
            gyration_radius_squared: Mean::new(),
            gyration_radius: Mean::new(),
            end_to_end_squared: Mean::new(),
            asphericity: Mean::new(),
            acylindricity: Mean::new(),
            relative_shape_anisotropy: Mean::new(),
            prolateness: Mean::new(),
            westin_cl: Mean::new(),
            westin_cp: Mean::new(),
            westin_cs: Mean::new(),
        })
    }
}

/// Polymer shape analysis via the mass-weighted gyration tensor.
#[derive(Debug)]
pub struct ShapeAnalysis {
    selection: Selection,
    #[debug(skip)]
    stream: Option<Box<dyn Write>>,
    frequency: Frequency,
    num_samples: usize,
    gyration_radius_squared: Mean,
    gyration_radius: Mean,
    end_to_end_squared: Mean,
    asphericity: Mean,
    acylindricity: Mean,
    relative_shape_anisotropy: Mean,
    prolateness: Mean,
    westin_cl: Mean,
    westin_cp: Mean,
    westin_cs: Mean,
}

/// Eigenvalues sorted ascending and the gyration tensor.
struct GyrationResult {
    eigenvalues: [f64; 3],
    tensor: Matrix3<f64>,
    rg_squared: f64,
}

/// Minimum Rg² to guard against division by zero.
const RG2_EPSILON: f64 = 1e-20;

/// Eigendecompose a symmetric tensor and return sorted results.
fn decompose_tensor(tensor: Matrix3<f64>) -> GyrationResult {
    let eigen = SymmetricEigen::new(tensor);
    let mut evals = [
        eigen.eigenvalues[0],
        eigen.eigenvalues[1],
        eigen.eigenvalues[2],
    ];
    evals.sort_by(f64::total_cmp);

    let rg_squared = evals.iter().sum();
    GyrationResult {
        eigenvalues: evals,
        tensor,
        rg_squared,
    }
}

/// Compute the mass-weighted gyration tensor for a group of particles.
fn gyration_tensor(
    group: &crate::group::Group,
    context: &impl Context,
) -> Option<GyrationResult> {
    let com = group.mass_center()?;
    if group.len() < 2 {
        return None;
    }

    let topology = context.topology_ref();
    let atomkinds = topology.atomkinds();
    let cell = context.cell();

    let mut tensor = Matrix3::<f64>::zeros();
    let mut total_mass = 0.0;

    for i in group.iter_active() {
        let particle = context.particle(i);
        let mass = atomkinds[particle.atom_id()].mass();
        let r = cell.distance(particle.pos(), com);
        total_mass += mass;
        tensor += r * r.transpose() * mass;
    }

    if total_mass <= 0.0 {
        return None;
    }
    tensor /= total_mass;

    Some(decompose_tensor(tensor))
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

impl crate::Info for ShapeAnalysis {
    fn short_name(&self) -> Option<&'static str> {
        Some("polymershape")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Polymer shape via gyration tensor")
    }
    fn citation(&self) -> Option<&'static str> {
        Some("doi:10/d6ff")
    }
}

impl<T: Context> Analyze<T> for ShapeAnalysis {
    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn sample(&mut self, context: &T, step: usize) -> Result<()> {
        if !self.frequency.should_perform(step) {
            return Ok(());
        }

        let topology = context.topology_ref();
        let groups = context.groups();
        let group_indices = self.selection.resolve_groups(topology, groups);

        for &gi in &group_indices {
            let group = &groups[gi];
            let Some(result) = gyration_tensor(group, context) else {
                continue;
            };

            self.gyration_radius_squared.add(result.rg_squared);
            self.gyration_radius.add(result.rg_squared.sqrt());

            let first = group.iter_active().next().unwrap();
            let last = group.iter_active().last().unwrap();
            if first != last {
                let re2 = context.cell().distance_squared(
                    context.particle(first).pos(),
                    context.particle(last).pos(),
                );
                self.end_to_end_squared.add(re2);
            }

            if let Some(desc) = compute_descriptors(&result.eigenvalues, result.rg_squared) {
                self.asphericity.add(desc.asphericity);
                self.acylindricity.add(desc.acylindricity);
                self.relative_shape_anisotropy
                    .add(desc.relative_shape_anisotropy);
                self.prolateness.add(desc.prolateness);
                self.westin_cl.add(desc.westin_cl);
                self.westin_cp.add(desc.westin_cp);
                self.westin_cs.add(desc.westin_cs);
            }

            if let Some(ref mut stream) = self.stream {
                let s = &result.tensor;
                writeln!(
                    stream,
                    "{} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                    step,
                    result.rg_squared.sqrt(),
                    s[(0, 0)],
                    s[(0, 1)],
                    s[(0, 2)],
                    s[(1, 1)],
                    s[(1, 2)],
                    s[(2, 2)]
                )?;
            }

            self.num_samples += 1;
        }
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }

    fn flush(&mut self) {
        if let Some(ref mut stream) = self.stream {
            let _ = stream.flush();
        }
    }

    fn to_yaml(&self) -> Option<serde_yaml::Value> {
        if self.num_samples == 0 {
            return None;
        }
        let mut map = serde_yaml::Mapping::new();
        let rg2 = self.gyration_radius_squared.mean();
        let re2 = self.end_to_end_squared.mean();

        map.insert("Rg".into(), serde_yaml::to_value(rg2.sqrt()).ok()?);
        map.insert("Re".into(), serde_yaml::to_value(re2.sqrt()).ok()?);
        map.insert(
            "Re2/Rg2".into(),
            serde_yaml::to_value(re2 / rg2).ok()?,
        );
        map.insert(
            "asphericity".into(),
            serde_yaml::to_value(self.asphericity.mean()).ok()?,
        );
        map.insert(
            "acylindricity".into(),
            serde_yaml::to_value(self.acylindricity.mean()).ok()?,
        );
        map.insert(
            "relative_shape_anisotropy".into(),
            serde_yaml::to_value(self.relative_shape_anisotropy.mean()).ok()?,
        );
        map.insert(
            "prolateness".into(),
            serde_yaml::to_value(self.prolateness.mean()).ok()?,
        );
        map.insert(
            "Cl".into(),
            serde_yaml::to_value(self.westin_cl.mean()).ok()?,
        );
        map.insert(
            "Cp".into(),
            serde_yaml::to_value(self.westin_cp.mean()).ok()?,
        );
        map.insert(
            "Cs".into(),
            serde_yaml::to_value(self.westin_cs.mean()).ok()?,
        );
        map.insert(
            "num_samples".into(),
            serde_yaml::Value::Number(self.num_samples.into()),
        );

        Some(serde_yaml::Value::Mapping(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisBuilder;
    use approx::assert_relative_eq;

    #[test]
    fn deserialize_builder() {
        let yaml = r#"
selection: "molecule polymer"
frequency: !Every 100
"#;
        let builder: ShapeAnalysisBuilder = serde_yaml::from_str(yaml).unwrap();
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
        let builder: ShapeAnalysisBuilder = serde_yaml::from_str(yaml).unwrap();
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
        let builders: Vec<AnalysisBuilder> = serde_yaml::from_str(yaml).unwrap();
        assert!(matches!(builders[0], AnalysisBuilder::PolymerShape(_)));
    }

    /// Helper: build gyration result from equal-mass positions.
    fn gyration_from_positions(positions: &[nalgebra::Vector3<f64>]) -> GyrationResult {
        let n = positions.len() as f64;
        let com: nalgebra::Vector3<f64> =
            positions.iter().sum::<nalgebra::Vector3<f64>>() / n;

        let mut tensor = Matrix3::<f64>::zeros();
        for p in positions {
            let r = p - com;
            tensor += r * r.transpose();
        }
        tensor /= n;
        decompose_tensor(tensor)
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
            assert!(desc.asphericity >= -1e-10, "asphericity must be non-negative");
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

        let mut tensor = Matrix3::<f64>::zeros();
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

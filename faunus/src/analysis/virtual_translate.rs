// Copyright 2023-2024 Mikael Lund
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

//! Virtual translate move analysis for force measurement
//!
//! This module implements the virtual translate move analysis, which performs
//! a virtual displacement of a single molecule in a specified direction and
//! measures the force using the Widom perturbation method.
//!
//! The force is calculated as:
//! ```text
//! f = kT * ln<exp(-dU/kT)> / dL
//! ```
//!
//! where `dU` is the energy change due to the perturbation and `dL` is the
//! displacement magnitude.

use super::{Analyze, Frequency};
use crate::change::{Change, GroupChange};
use crate::dimension::Dimension;
use crate::energy::EnergyChange;
use crate::group::{GroupSelection, GroupSize};
use crate::topology::Topology;
use crate::{Context, Point};
use anyhow::Result;
use average::{Estimate, Mean};
use derive_builder::Builder;
use derive_more::Debug;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;

/// Virtual translate move analysis.
///
/// Performs a virtual displacement of a single molecule in a specified direction
/// and measures the force by perturbation using the Widom method:
///
/// `f = kT * ln<exp(-dU/kT)> / dL`
///
/// where `dU` is the energy change and `dL` is the displacement magnitude.
///
/// # Requirements
/// - Exactly one active molecule of the specified type must exist in the system.
/// - The molecule must not be atomic (multi-atom molecules only).
#[derive(Debug, Builder)]
#[builder(build_fn(skip), derive(Deserialize, Serialize))]
pub struct VirtualTranslate {
    /// Molecule name to translate
    #[allow(dead_code)]
    molecule: String,

    /// Molecule id (resolved from name)
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    molecule_id: usize,

    /// Displacement magnitude in Angstrom
    #[builder_field_attr(serde(rename = "dL"))]
    displacement: f64,

    /// Displacement directions. Defaults to z-axis.
    #[allow(dead_code)]
    #[builder_field_attr(serde(default = "default_directions"))]
    directions: Dimension,

    /// Normalized displacement direction
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    unit_direction: Point,

    /// Output file for streaming results
    #[allow(dead_code)]
    #[builder_field_attr(serde(rename = "file"))]
    output_file: Option<PathBuf>,

    /// Stream object for output
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    #[debug(skip)]
    stream: Option<Box<dyn Write>>,

    /// Sample frequency
    frequency: Frequency,

    /// Running average of exp(-dU/kT)
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    mean_exp_energy: Mean,

    /// Number of samples taken
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip_deserializing))]
    num_samples: usize,

    /// Temperature in Kelvin (needed to convert energy to kT).
    /// Default is 298.15 K if not specified.
    #[builder_field_attr(serde(default = "default_temperature"))]
    temperature: f64,
}

/// Default temperature in Kelvin (298.15 K = 25°C).
/// Returns `Option` because derive_builder wraps fields in `Option`.
fn default_temperature() -> Option<f64> {
    Some(298.15)
}

/// Default displacement directions (z-axis).
/// Returns `Option` because derive_builder wraps fields in `Option`.
fn default_directions() -> Option<Dimension> {
    Some(Dimension::Z)
}

/// Convert Dimension to a normalized direction vector
fn dimension_to_unit_vector(dim: &Dimension) -> Result<Point> {
    if *dim == Dimension::None {
        anyhow::bail!("Direction cannot be 'None'");
    }
    // Filter a (1,1,1) vector to get direction and normalize
    let dir_vec = dim.filter(Point::new(1.0, 1.0, 1.0));
    let norm = dir_vec.norm();
    if norm < 1e-10 {
        anyhow::bail!("Direction vector cannot be zero");
    }
    Ok(dir_vec / norm)
}

impl VirtualTranslateBuilder {
    fn validate(&self) -> Result<()> {
        if self.molecule.is_none() {
            anyhow::bail!("Missing required field 'molecule' for VirtualTranslate analysis");
        }
        if self.displacement.is_none() {
            anyhow::bail!("Missing required field 'dL' for VirtualTranslate analysis");
        }
        if self.frequency.is_none() {
            anyhow::bail!("Missing required field 'frequency' for VirtualTranslate analysis");
        }
        Ok(())
    }

    /// Build the VirtualTranslate analysis
    pub fn build(&self, topology: &Topology) -> Result<VirtualTranslate> {
        self.validate()?;

        let molecule_name = self.molecule.clone().unwrap();
        let molecule_id = topology
            .find_molecule(&molecule_name)
            .ok_or_else(|| anyhow::anyhow!("Molecule '{}' not found in topology", molecule_name))?
            .id();

        // Check that the molecule is not atomic
        let molkind = &topology.moleculekinds()[molecule_id];
        if molkind.atom_indices().len() < 2 {
            anyhow::bail!(
                "VirtualTranslate requires non-atomic molecules; '{}' has only {} atom(s)",
                molecule_name,
                molkind.atom_indices().len()
            );
        }

        let displacement = self.displacement.unwrap();
        let directions = self.directions.clone().unwrap_or(Dimension::Z);
        let temperature = self.temperature.unwrap_or(298.15);

        // Convert Dimension to unit direction vector
        let unit_direction = dimension_to_unit_vector(&directions)?;

        // Open output stream if file specified
        let stream = if let Some(path) = self.output_file.as_ref().and_then(|p| p.as_ref()) {
            let mut stream = crate::auxiliary::open_compressed(path)?;
            // Write header
            writeln!(stream, "# step dL/Å dU/kT <force>/kT/Å")?;
            Some(stream)
        } else {
            None
        };

        Ok(VirtualTranslate {
            molecule: molecule_name,
            molecule_id,
            displacement,
            directions,
            unit_direction,
            output_file: self.output_file.clone().flatten(),
            stream,
            frequency: self.frequency.unwrap(),
            mean_exp_energy: Mean::new(),
            num_samples: 0,
            temperature,
        })
    }
}

impl crate::Info for VirtualTranslate {
    fn short_name(&self) -> Option<&'static str> {
        Some("virtualtranslate")
    }

    fn long_name(&self) -> Option<&'static str> {
        Some("Virtual translate move for force measurement by perturbation")
    }

    fn citation(&self) -> Option<&'static str> {
        // The virtual translate method is based on Widom insertion
        Some("doi:10.1063/1.1749657")
    }
}

impl VirtualTranslate {
    /// Calculate the mean free energy from the Widom average
    /// Returns -ln(<exp(-dU/kT)>) in units of kT
    fn mean_free_energy(&self) -> f64 {
        let mean = self.mean_exp_energy.mean();
        if !mean.is_finite() || mean <= 0.0 {
            log::warn!(
                "VirtualTranslate: invalid Widom average <exp(-dU/kT)> = {}; returning +inf free energy",
                mean
            );
            return f64::INFINITY;
        }
        -mean.ln()
    }

    /// Calculate the mean force in units of kT/Å
    fn mean_force(&self) -> f64 {
        if self.displacement.abs() > f64::EPSILON {
            -self.mean_free_energy() / self.displacement
        } else {
            0.0
        }
    }

    /// Perform the virtual perturbation and return the energy change in kT
    fn perturb<T: Context>(&self, context: &mut T, group_index: usize) -> Result<f64> {
        let displacement_vector = self.displacement * self.unit_direction;

        // Calculate change descriptor for energy calculations
        let change = Change::SingleGroup(group_index, GroupChange::RigidBody);

        // Calculate old energy (in kJ/mol)
        let old_energy = context.hamiltonian().energy(context, &change);

        // Translate the group
        let particle_indices: Vec<usize> = context.groups()[group_index].iter_active().collect();
        context.translate_particles(&particle_indices, &displacement_vector);

        // Calculate new energy (in kJ/mol)
        let new_energy = context.hamiltonian().energy(context, &change);

        // Restore original positions
        context.translate_particles(&particle_indices, &(-displacement_vector));

        // Convert energy change from kJ/mol to kT
        // kT = R * T where R is the molar gas constant
        const KILO_JOULE_PER_JOULE: f64 = 1e-3;
        let k_t = physical_constants::MOLAR_GAS_CONSTANT * KILO_JOULE_PER_JOULE * self.temperature;
        let delta_u = (new_energy - old_energy) / k_t;

        Ok(delta_u)
    }

    /// Add exp(-dU) to the Widom average, with overflow protection
    fn collect_widom_average(&mut self, energy_change: f64) -> bool {
        if energy_change < -(f64::MAX_EXP as f64) {
            log::warn!(
                "VirtualTranslate: skipping sample due to too negative energy; consider decreasing dL"
            );
            return false;
        }
        if energy_change > f64::MAX_EXP as f64 {
            log::warn!(
                "VirtualTranslate: skipping sample due to too positive energy; consider decreasing dL"
            );
            return false;
        }
        self.mean_exp_energy.add((-energy_change).exp());
        true
    }

    /// Write data to the output stream
    fn write_to_stream(&mut self, step: usize, energy_change: f64) -> Result<()> {
        // Calculate mean_force before mutable borrow of stream
        let mean_force = self.mean_force();
        let displacement = self.displacement;

        if let Some(ref mut stream) = self.stream {
            writeln!(
                stream,
                "{} {:.3e} {:.6e} {:.6e}",
                step, displacement, energy_change, mean_force
            )?;
        }
        Ok(())
    }
}

impl<T: Context> Analyze<T> for VirtualTranslate {
    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn sample(&mut self, context: &T, step: usize) -> Result<()> {
        if !self.frequency.should_perform(step) {
            return Ok(());
        }

        // Check for zero displacement
        if self.displacement.abs() < f64::EPSILON {
            return Ok(());
        }

        // Find active molecules of the specified type
        let group_indices = context.select(&GroupSelection::ByMoleculeId(self.molecule_id));
        let active_groups: Vec<usize> = group_indices
            .into_iter()
            .filter(|&idx| context.groups()[idx].size() != GroupSize::Empty)
            .collect();

        if active_groups.is_empty() {
            return Ok(());
        }

        if active_groups.len() > 1 {
            anyhow::bail!(
                "VirtualTranslate requires exactly ONE active molecule of type '{}', found {}",
                self.molecule,
                active_groups.len()
            );
        }

        let group_index = active_groups[0];

        // Clone context because `Analyze::sample` receives `&T` but perturbation
        // needs `&mut T` to translate particles. The perturbation is virtual
        // (translate + restore), so the clone is discarded unchanged.
        let mut trial_context = context.clone();
        let energy_change = self.perturb(&mut trial_context, group_index)?;

        if self.collect_widom_average(energy_change) {
            self.num_samples += 1;
            self.write_to_stream(step, energy_change)?;
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
}

impl<T: Context> From<VirtualTranslate> for Box<dyn Analyze<T>> {
    fn from(analysis: VirtualTranslate) -> Box<dyn Analyze<T>> {
        Box::new(analysis)
    }
}

impl std::fmt::Display for VirtualTranslate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Virtual Translate Analysis:")?;
        writeln!(f, "  Molecule:    {}", self.molecule)?;
        writeln!(f, "  dL:          {} Å", self.displacement)?;
        writeln!(
            f,
            "  Direction:   [{:.3}, {:.3}, {:.3}]",
            self.unit_direction.x, self.unit_direction.y, self.unit_direction.z
        )?;
        writeln!(f, "  Samples:     {}", self.num_samples)?;
        if !self.mean_exp_energy.is_empty() {
            writeln!(f, "  <force>:     {:.6} kT/Å", self.mean_force())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisBuilder;
    use crate::topology::MoleculeKindBuilder;
    use crate::Info;
    use float_cmp::assert_approx_eq;

    /// Assert approximate equality of a Point's x, y, z components.
    macro_rules! assert_point_approx_eq {
        ($point:expr, $x:expr, $y:expr, $z:expr) => {
            assert_approx_eq!(f64, $point.x, $x);
            assert_approx_eq!(f64, $point.y, $y);
            assert_approx_eq!(f64, $point.z, $z);
        };
    }

    /// Create a topology with a single molecule of the given name and atom count.
    fn make_topology(name: &str, num_atoms: usize) -> Topology {
        let molecule = MoleculeKindBuilder::default()
            .name(name)
            .atoms((0..num_atoms).map(|i| format!("A{i}")).collect())
            .atom_indices((0..num_atoms).collect())
            .build()
            .unwrap();
        let mut topology = Topology::default();
        topology.add_moleculekind(molecule);
        topology
    }

    /// Helper to build a VirtualTranslate with common defaults for testing.
    fn build_vt(topology: &Topology, displacement: f64) -> VirtualTranslate {
        VirtualTranslateBuilder::default()
            .molecule("MOL".to_string())
            .displacement(displacement)
            .frequency(Frequency::Every(1))
            .build(topology)
            .unwrap()
    }

    /// Deserialize YAML into AnalysisBuilder list and extract the VirtualTranslateBuilder at `index`.
    fn deserialize_vt_builder(yaml: &str, index: usize) -> VirtualTranslateBuilder {
        let builders: Vec<AnalysisBuilder> = serde_yaml::from_str(yaml).unwrap();
        match &builders[index] {
            AnalysisBuilder::VirtualTranslate(b) => b.clone(),
            _ => panic!("expected VirtualTranslate variant"),
        }
    }

    #[test]
    fn test_dimension_to_unit_vector() {
        assert_point_approx_eq!(
            dimension_to_unit_vector(&Dimension::Z).unwrap(),
            0.0,
            0.0,
            1.0
        );
        assert_point_approx_eq!(
            dimension_to_unit_vector(&Dimension::X).unwrap(),
            1.0,
            0.0,
            0.0
        );
        assert_point_approx_eq!(
            dimension_to_unit_vector(&Dimension::Y).unwrap(),
            0.0,
            1.0,
            0.0
        );

        let s2 = 1.0 / 2.0_f64.sqrt();
        assert_point_approx_eq!(
            dimension_to_unit_vector(&Dimension::XY).unwrap(),
            s2,
            s2,
            0.0
        );

        let s3 = 1.0 / 3.0_f64.sqrt();
        assert_point_approx_eq!(
            dimension_to_unit_vector(&Dimension::XYZ).unwrap(),
            s3,
            s3,
            s3
        );

        assert!(dimension_to_unit_vector(&Dimension::None).is_err());
    }

    #[test]
    fn build_with_valid_fields() {
        let topology = make_topology("MOL", 2);
        let vt = VirtualTranslateBuilder::default()
            .molecule("MOL".to_string())
            .displacement(0.01)
            .frequency(Frequency::Every(10))
            .build(&topology)
            .unwrap();
        assert_approx_eq!(f64, vt.displacement, 0.01);
        assert_eq!(vt.molecule_id, 0);
        assert_eq!(vt.molecule, "MOL");
        assert_approx_eq!(f64, vt.temperature, 298.15);
        assert_eq!(vt.directions, Dimension::Z);
        assert_eq!(vt.num_samples, 0);
    }

    #[test]
    fn build_missing_molecule() {
        let topology = make_topology("MOL", 2);
        let result = VirtualTranslateBuilder::default()
            .displacement(0.01)
            .frequency(Frequency::Every(10))
            .build(&topology);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("molecule"));
    }

    #[test]
    fn build_missing_displacement() {
        let topology = make_topology("MOL", 2);
        let result = VirtualTranslateBuilder::default()
            .molecule("MOL".to_string())
            .frequency(Frequency::Every(10))
            .build(&topology);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("dL"));
    }

    #[test]
    fn build_missing_frequency() {
        let topology = make_topology("MOL", 2);
        let result = VirtualTranslateBuilder::default()
            .molecule("MOL".to_string())
            .displacement(0.01)
            .build(&topology);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("frequency"));
    }

    #[test]
    fn build_unknown_molecule() {
        let topology = make_topology("MOL", 2);
        let result = VirtualTranslateBuilder::default()
            .molecule("UNKNOWN".to_string())
            .displacement(0.01)
            .frequency(Frequency::Every(10))
            .build(&topology);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn build_rejects_atomic_molecule() {
        let topology = make_topology("ATOM", 1);
        let result = VirtualTranslateBuilder::default()
            .molecule("ATOM".to_string())
            .displacement(0.01)
            .frequency(Frequency::Every(10))
            .build(&topology);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("non-atomic molecules"));
    }

    #[test]
    fn build_with_custom_temperature_and_direction() {
        let topology = make_topology("MOL", 2);
        let vt = VirtualTranslateBuilder::default()
            .molecule("MOL".to_string())
            .displacement(0.05)
            .frequency(Frequency::Every(5))
            .temperature(310.0)
            .directions(Dimension::X)
            .build(&topology)
            .unwrap();
        assert_approx_eq!(f64, vt.temperature, 310.0);
        assert_eq!(vt.directions, Dimension::X);
        assert_point_approx_eq!(vt.unit_direction, 1.0, 0.0, 0.0);
    }

    #[test]
    fn collect_widom_average_normal() {
        let topology = make_topology("MOL", 2);
        let mut vt = build_vt(&topology, 0.1);

        // Zero energy change → exp(0) = 1.0
        assert!(vt.collect_widom_average(0.0));
        assert_approx_eq!(f64, vt.mean_exp_energy.mean(), 1.0);

        // Positive energy change → exp(-dU) < 1
        assert!(vt.collect_widom_average(1.0));
        let expected = (1.0 + (-1.0_f64).exp()) / 2.0;
        assert_approx_eq!(f64, vt.mean_exp_energy.mean(), expected, epsilon = 1e-12);
    }

    #[test]
    fn collect_widom_average_overflow_protection() {
        let topology = make_topology("MOL", 2);
        let mut vt = build_vt(&topology, 0.1);

        // Too negative energy → would overflow exp()
        assert!(!vt.collect_widom_average(-(f64::MAX_EXP as f64 + 1.0)));
        assert!(vt.mean_exp_energy.is_empty());

        // Too positive energy → exp(-dU) underflows
        assert!(!vt.collect_widom_average(f64::MAX_EXP as f64 + 1.0));
        assert!(vt.mean_exp_energy.is_empty());
    }

    #[test]
    fn mean_free_energy_and_force() {
        let topology = make_topology("MOL", 2);
        let mut vt = build_vt(&topology, 0.5);

        // Add samples with zero energy change → exp(0) = 1.0
        // mean = 1.0, free_energy = -ln(1.0) = 0.0
        vt.collect_widom_average(0.0);
        vt.collect_widom_average(0.0);
        assert_approx_eq!(f64, vt.mean_free_energy(), 0.0);
        assert_approx_eq!(f64, vt.mean_force(), 0.0);
    }

    #[test]
    fn mean_free_energy_with_nonzero_energy() {
        let topology = make_topology("MOL", 2);
        let mut vt = build_vt(&topology, 0.1);

        // Add single sample with energy_change = 2.0
        // exp(-2.0) ≈ 0.1353
        // free_energy = -ln(exp(-2.0)) = 2.0
        // force = -free_energy / dL = -2.0 / 0.1 = -20.0
        vt.collect_widom_average(2.0);
        assert_approx_eq!(f64, vt.mean_free_energy(), 2.0, epsilon = 1e-12);
        assert_approx_eq!(f64, vt.mean_force(), -20.0, epsilon = 1e-10);
    }

    #[test]
    fn mean_free_energy_empty_returns_infinity() {
        let topology = make_topology("MOL", 2);
        let vt = build_vt(&topology, 0.1);
        // No samples → mean is NaN → should return +inf
        assert!(vt.mean_free_energy().is_infinite());
    }

    #[test]
    fn mean_force_zero_displacement() {
        let topology = make_topology("MOL", 2);
        let vt = build_vt(&topology, 0.0);
        assert_approx_eq!(f64, vt.mean_force(), 0.0);
    }

    #[test]
    fn info_trait() {
        let topology = make_topology("MOL", 2);
        let vt = build_vt(&topology, 0.01);
        assert_eq!(vt.short_name(), Some("virtualtranslate"));
        assert!(vt.long_name().unwrap().contains("force measurement"));
        assert!(vt.citation().unwrap().starts_with("doi:"));
    }

    #[test]
    fn display_without_samples() {
        let topology = make_topology("MOL", 2);
        let vt = build_vt(&topology, 0.01);
        let output = format!("{}", vt);
        assert!(output.contains("MOL"));
        assert!(output.contains("0.01"));
        assert!(output.contains("Samples:     0"));
        assert!(!output.contains("<force>"));
    }

    #[test]
    fn display_with_samples() {
        let topology = make_topology("MOL", 2);
        let mut vt = build_vt(&topology, 0.1);
        vt.collect_widom_average(0.0);
        vt.num_samples = 1;
        let output = format!("{}", vt);
        assert!(output.contains("Samples:     1"));
        assert!(output.contains("<force>"));
    }

    #[test]
    fn deserialize_virtual_translate_builders() {
        let yaml = std::fs::read_to_string("tests/files/virtual_translate.yaml").unwrap();
        let topology = make_topology("MOL", 2);

        // First: z-direction with explicit temperature
        let vt = deserialize_vt_builder(&yaml, 0).build(&topology).unwrap();
        assert_eq!(vt.molecule, "MOL");
        assert_approx_eq!(f64, vt.displacement, 0.01);
        assert_eq!(vt.directions, Dimension::Z);
        assert_approx_eq!(f64, vt.temperature, 298.15);
        assert!(matches!(vt.frequency, Frequency::Every(10)));

        // Second: x-direction with default temperature
        let vt = deserialize_vt_builder(&yaml, 1).build(&topology).unwrap();
        assert_approx_eq!(f64, vt.displacement, 0.05);
        assert_eq!(vt.directions, Dimension::X);
        assert_approx_eq!(f64, vt.temperature, 298.15);
        assert!(matches!(vt.frequency, Frequency::Every(5)));
    }

    #[test]
    fn deserialize_missing_required_fields() {
        let yaml = r#"
- !VirtualTranslate
  molecule: MOL
  frequency: !Every 10
"#;
        let topology = make_topology("MOL", 2);
        assert!(deserialize_vt_builder(yaml, 0).build(&topology).is_err());
    }

    #[test]
    fn deserialize_default_direction_is_z() {
        let yaml = r#"
- !VirtualTranslate
  molecule: MOL
  dL: 0.1
  frequency: !Every 1
"#;
        let topology = make_topology("MOL", 2);
        let vt = deserialize_vt_builder(yaml, 0).build(&topology).unwrap();
        assert_eq!(vt.directions, Dimension::Z);
    }

    #[test]
    fn roundtrip_serialize_deserialize_builder() {
        let yaml = r#"
molecule: MOL
dL: 0.05
directions: !x
temperature: 310.0
frequency: !Every 5
"#;
        let builder: VirtualTranslateBuilder = serde_yaml::from_str(yaml).unwrap();
        let serialized = serde_yaml::to_string(&builder).unwrap();
        let roundtrip: VirtualTranslateBuilder = serde_yaml::from_str(&serialized).unwrap();
        let topology = make_topology("MOL", 2);
        let vt = roundtrip.build(&topology).unwrap();
        assert_approx_eq!(f64, vt.displacement, 0.05);
        assert_eq!(vt.directions, Dimension::X);
        assert_approx_eq!(f64, vt.temperature, 310.0);
    }

}

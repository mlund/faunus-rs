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
use crate::selection::Selection;
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
/// - Exactly one active molecule matching the selection must exist in the system.
#[derive(Debug, Builder)]
#[builder(build_fn(skip), derive(Deserialize, Serialize))]
pub struct VirtualTranslate {
    /// Selection expression for the molecule to translate
    selection: Selection,

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
const fn default_temperature() -> Option<f64> {
    Some(298.15)
}

/// Default displacement directions (z-axis).
/// Returns `Option` because derive_builder wraps fields in `Option`.
const fn default_directions() -> Option<Dimension> {
    Some(Dimension::Z)
}

/// Convert Dimension to a normalized direction vector so that
/// the displacement magnitude is controlled solely by `dL`.
fn dimension_to_unit_vector(dim: &Dimension) -> Result<Point> {
    if *dim == Dimension::None {
        anyhow::bail!("Direction cannot be 'None'");
    }
    let dir_vec = dim.filter(Point::new(1.0, 1.0, 1.0));
    let norm = dir_vec.norm();
    if norm < 1e-10 {
        anyhow::bail!("Direction vector cannot be zero");
    }
    Ok(dir_vec / norm)
}

impl VirtualTranslateBuilder {
    fn validate(&self) -> Result<()> {
        if self.selection.is_none() {
            anyhow::bail!("Missing required field 'selection' for VirtualTranslate analysis");
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
    pub fn build(&self) -> Result<VirtualTranslate> {
        self.validate()?;

        let displacement = self.displacement.unwrap();
        let directions = self.directions.clone().unwrap_or(Dimension::Z);
        let temperature = self.temperature.unwrap_or(298.15);

        let unit_direction = dimension_to_unit_vector(&directions)?;

        let stream = if let Some(path) = self.output_file.as_ref().and_then(|p| p.as_ref()) {
            let mut stream = crate::auxiliary::open_compressed(path)?;
            writeln!(stream, "# step dL/Å dU/kT <force>/kT/Å")?;
            Some(stream)
        } else {
            None
        };

        Ok(VirtualTranslate {
            selection: self.selection.clone().unwrap(),
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
        Some("doi:10.1063/1.1749657") // Widom insertion method
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
        let change = Change::SingleGroup(group_index, GroupChange::RigidBody);

        let old_energy = context.hamiltonian().energy(context, &change); // kJ/mol

        let particle_indices: Vec<usize> = context.groups()[group_index].iter_active().collect();
        context.translate_particles(&particle_indices, &displacement_vector);
        let new_energy = context.hamiltonian().energy(context, &change); // kJ/mol
        context.translate_particles(&particle_indices, &(-displacement_vector));

        // Convert kJ/mol → kT using kT = R·T
        const KILO_JOULE_PER_JOULE: f64 = 1e-3;
        let k_t = physical_constants::MOLAR_GAS_CONSTANT * KILO_JOULE_PER_JOULE * self.temperature;
        let delta_u = (new_energy - old_energy) / k_t;

        Ok(delta_u)
    }

    /// Add exp(-dU) to the Widom average, with overflow protection.
    ///
    /// Returns `true` if the sample was added to the running average,
    /// `false` if it was skipped due to extreme energy values that would
    /// cause overflow in `exp()`. Skipped samples are still written to
    /// the output stream to keep it synchronized with other analyses.
    fn collect_widom_average(&mut self, energy_change: f64) -> bool {
        if energy_change < -(f64::MAX_EXP as f64) {
            log::warn!(
                "VirtualTranslate: skipping Widom average due to too negative energy; consider decreasing dL"
            );
            return false;
        }
        if energy_change > f64::MAX_EXP as f64 {
            log::warn!(
                "VirtualTranslate: skipping Widom average due to too positive energy; consider decreasing dL"
            );
            return false;
        }
        self.mean_exp_energy.add((-energy_change).exp());
        true
    }

    /// Write data to the output stream.
    ///
    /// Called for every sampled step, including those where the Widom average
    /// was skipped due to overflow. This ensures the output file has one row
    /// per sampled step, staying in sync with other analyses at the same frequency.
    fn write_to_stream(&mut self, step: usize, energy_change: f64) -> Result<()> {
        // Bind values before mutable borrow of self.stream
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

        if self.displacement.abs() < f64::EPSILON {
            return Ok(());
        }

        let active_groups = self
            .selection
            .resolve_groups(context.topology_ref(), context.groups());

        if active_groups.is_empty() {
            return Ok(());
        }

        if active_groups.len() > 1 {
            anyhow::bail!(
                "VirtualTranslate requires exactly ONE active molecule matching '{}', found {}",
                self.selection,
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
        }
        // Always write to stream to stay synchronized with other analyses
        // (e.g. MassCenterDistance) that sample at the same frequency.
        self.write_to_stream(step, energy_change)?;

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
    fn from(analysis: VirtualTranslate) -> Self {
        Box::new(analysis)
    }
}

impl std::fmt::Display for VirtualTranslate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Virtual Translate Analysis:")?;
        writeln!(f, "  Selection:   {}", self.selection)?;
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
    use crate::Info;
    use float_cmp::assert_approx_eq;

    /// Assert approximate equality of a Point's x, y, z components.
    macro_rules! assert_point_approx_eq {
        ($point:expr_2021, $x:expr_2021, $y:expr_2021, $z:expr_2021) => {
            assert_approx_eq!(f64, $point.x, $x);
            assert_approx_eq!(f64, $point.y, $y);
            assert_approx_eq!(f64, $point.z, $z);
        };
    }

    /// Helper to build a VirtualTranslate with common defaults for testing.
    fn build_vt(displacement: f64) -> VirtualTranslate {
        VirtualTranslateBuilder::default()
            .selection(Selection::parse("molecule MOL").unwrap())
            .displacement(displacement)
            .frequency(Frequency::Every(1))
            .build()
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
        let vt = VirtualTranslateBuilder::default()
            .selection(Selection::parse("molecule MOL").unwrap())
            .displacement(0.01)
            .frequency(Frequency::Every(10))
            .build()
            .unwrap();
        assert_approx_eq!(f64, vt.displacement, 0.01);
        assert_eq!(vt.selection.source(), "molecule MOL");
        assert_approx_eq!(f64, vt.temperature, 298.15);
        assert_eq!(vt.directions, Dimension::Z);
        assert_eq!(vt.num_samples, 0);
    }

    #[test]
    fn build_missing_selection() {
        let result = VirtualTranslateBuilder::default()
            .displacement(0.01)
            .frequency(Frequency::Every(10))
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("selection"));
    }

    #[test]
    fn build_missing_displacement() {
        let result = VirtualTranslateBuilder::default()
            .selection(Selection::parse("molecule MOL").unwrap())
            .frequency(Frequency::Every(10))
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("dL"));
    }

    #[test]
    fn build_missing_frequency() {
        let result = VirtualTranslateBuilder::default()
            .selection(Selection::parse("molecule MOL").unwrap())
            .displacement(0.01)
            .build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("frequency"));
    }

    #[test]
    fn build_with_custom_temperature_and_direction() {
        let vt = VirtualTranslateBuilder::default()
            .selection(Selection::parse("molecule MOL").unwrap())
            .displacement(0.05)
            .frequency(Frequency::Every(5))
            .temperature(310.0)
            .directions(Dimension::X)
            .build()
            .unwrap();
        assert_approx_eq!(f64, vt.temperature, 310.0);
        assert_eq!(vt.directions, Dimension::X);
        assert_point_approx_eq!(vt.unit_direction, 1.0, 0.0, 0.0);
    }

    #[test]
    fn collect_widom_average_normal() {
        let mut vt = build_vt(0.1);

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
        let mut vt = build_vt(0.1);

        // Too negative energy → would overflow exp()
        assert!(!vt.collect_widom_average(-(f64::MAX_EXP as f64 + 1.0)));
        assert!(vt.mean_exp_energy.is_empty());

        // Too positive energy → exp(-dU) underflows
        assert!(!vt.collect_widom_average(f64::MAX_EXP as f64 + 1.0));
        assert!(vt.mean_exp_energy.is_empty());
    }

    #[test]
    fn mean_free_energy_and_force() {
        let mut vt = build_vt(0.5);

        // Add samples with zero energy change → exp(0) = 1.0
        // mean = 1.0, free_energy = -ln(1.0) = 0.0
        vt.collect_widom_average(0.0);
        vt.collect_widom_average(0.0);
        assert_approx_eq!(f64, vt.mean_free_energy(), 0.0);
        assert_approx_eq!(f64, vt.mean_force(), 0.0);
    }

    #[test]
    fn mean_free_energy_with_nonzero_energy() {
        let mut vt = build_vt(0.1);

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
        let vt = build_vt(0.1);
        // No samples → mean is NaN → should return +inf
        assert!(vt.mean_free_energy().is_infinite());
    }

    #[test]
    fn mean_force_zero_displacement() {
        let vt = build_vt(0.0);
        assert_approx_eq!(f64, vt.mean_force(), 0.0);
    }

    #[test]
    fn info_trait() {
        let vt = build_vt(0.01);
        assert_eq!(vt.short_name(), Some("virtualtranslate"));
        assert!(vt.long_name().unwrap().contains("force measurement"));
        assert!(vt.citation().unwrap().starts_with("doi:"));
    }

    #[test]
    fn display_without_samples() {
        let vt = build_vt(0.01);
        let output = format!("{vt}");
        assert!(output.contains("molecule MOL"));
        assert!(output.contains("0.01"));
        assert!(output.contains("Samples:     0"));
        assert!(!output.contains("<force>"));
    }

    #[test]
    fn display_with_samples() {
        let mut vt = build_vt(0.1);
        vt.collect_widom_average(0.0);
        vt.num_samples = 1;
        let output = format!("{vt}");
        assert!(output.contains("Samples:     1"));
        assert!(output.contains("<force>"));
    }

    #[test]
    fn deserialize_virtual_translate_builders() {
        let yaml = std::fs::read_to_string("tests/files/virtual_translate.yaml").unwrap();

        // First: z-direction with explicit temperature
        let vt = deserialize_vt_builder(&yaml, 0).build().unwrap();
        assert_eq!(vt.selection.source(), "molecule MOL");
        assert_approx_eq!(f64, vt.displacement, 0.01);
        assert_eq!(vt.directions, Dimension::Z);
        assert_approx_eq!(f64, vt.temperature, 298.15);
        assert!(matches!(vt.frequency, Frequency::Every(10)));

        // Second: x-direction with default temperature
        let vt = deserialize_vt_builder(&yaml, 1).build().unwrap();
        assert_approx_eq!(f64, vt.displacement, 0.05);
        assert_eq!(vt.directions, Dimension::X);
        assert_approx_eq!(f64, vt.temperature, 298.15);
        assert!(matches!(vt.frequency, Frequency::Every(5)));
    }

    #[test]
    fn deserialize_missing_required_fields() {
        let yaml = r#"
- !VirtualTranslate
  selection: "molecule MOL"
  frequency: !Every 10
"#;
        assert!(deserialize_vt_builder(yaml, 0).build().is_err());
    }

    #[test]
    fn deserialize_default_direction_is_z() {
        let yaml = r#"
- !VirtualTranslate
  selection: "molecule MOL"
  dL: 0.1
  frequency: !Every 1
"#;
        let vt = deserialize_vt_builder(yaml, 0).build().unwrap();
        assert_eq!(vt.directions, Dimension::Z);
    }

    #[test]
    fn roundtrip_serialize_deserialize_builder() {
        let yaml = r#"
selection: "molecule MOL"
dL: 0.05
directions: !x
temperature: 310.0
frequency: !Every 5
"#;
        let builder: VirtualTranslateBuilder = serde_yaml::from_str(yaml).unwrap();
        let serialized = serde_yaml::to_string(&builder).unwrap();
        let roundtrip: VirtualTranslateBuilder = serde_yaml::from_str(&serialized).unwrap();
        let vt = roundtrip.build().unwrap();
        assert_approx_eq!(f64, vt.displacement, 0.05);
        assert_eq!(vt.directions, Dimension::X);
        assert_approx_eq!(f64, vt.temperature, 310.0);
    }
}

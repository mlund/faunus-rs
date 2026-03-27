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

use super::widom::WidomAccumulator;
use super::{Analyze, Frequency};
use crate::auxiliary::{ColumnWriter, MappingExt};
use crate::axes::Axes;
use crate::change::{Change, GroupChange};
use crate::energy::EnergyChange;
use crate::selection::{Selection, SelectionCache};
use crate::{Context, Point};
use anyhow::Result;
use derive_builder::Builder;
use derive_more::Debug;
use serde::{Deserialize, Serialize};
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
    directions: Axes,

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
    stream: Option<ColumnWriter>,

    /// Sample frequency
    frequency: Frequency,

    /// Widom exponential average accumulator (log-sum-exp)
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    widom: WidomAccumulator,

    /// Cached resolved group indices to avoid re-resolution when N hasn't changed.
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    group_cache: SelectionCache,

    /// Thermal energy R*T in kJ/mol.
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    thermal_energy: f64,
}

/// Default displacement directions (z-axis).
/// Returns `Option` because derive_builder wraps fields in `Option`.
const fn default_directions() -> Option<Axes> {
    Some(Axes::Z)
}

/// Convert Axes to a normalized direction vector so that
/// the displacement magnitude is controlled solely by `dL`.
fn axes_to_unit_vector(dim: &Axes) -> Result<Point> {
    if *dim == Axes::None {
        anyhow::bail!("Direction cannot be 'None'");
    }
    let dir_vec = dim.project(Point::new(1.0, 1.0, 1.0));
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

    /// Build the VirtualTranslate analysis.
    ///
    /// `thermal_energy` is R*T in kJ/mol, typically from `Medium::temperature()`.
    pub fn build(&self, thermal_energy: f64) -> Result<VirtualTranslate> {
        self.validate()?;

        let displacement = self.displacement.unwrap();
        let directions = self.directions.unwrap_or(Axes::Z);

        let unit_direction = axes_to_unit_vector(&directions)?;

        let stream = if let Some(path) = self.output_file.as_ref().and_then(|p| p.as_ref()) {
            Some(ColumnWriter::open(
                path,
                &["step", "dL/Å", "dU/kT", "<force>/kT/Å"],
            )?)
        } else {
            None
        };

        Ok(VirtualTranslate {
            selection: self.selection.as_ref().unwrap().clone(),
            displacement,
            directions,
            unit_direction,
            output_file: self.output_file.as_ref().and_then(|p| p.clone()),
            stream,
            frequency: self.frequency.unwrap(),
            widom: WidomAccumulator::new(),
            group_cache: SelectionCache::default(),
            thermal_energy,
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
        Some("doi:10.1063/1.1734110") // Widom insertion method
    }
}

impl VirtualTranslate {
    /// Calculate the mean force in units of kT/Å
    fn mean_force(&self) -> f64 {
        if self.displacement.abs() > f64::EPSILON {
            -self.widom.mean_free_energy() / self.displacement
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

        Ok((new_energy - old_energy) / self.thermal_energy)
    }

    /// Write data to the output stream.
    ///
    /// Called for every sampled step, including those where the Widom average
    /// was skipped due to overflow. This ensures the output file has one row
    /// per sampled step, staying in sync with other analyses at the same frequency.
    fn write_to_stream(&mut self, step: usize, energy_change: f64) -> Result<()> {
        let mean_force = self.mean_force();
        let displacement = self.displacement;

        if let Some(ref mut stream) = self.stream {
            stream.write_row(&[
                &step,
                &format_args!("{displacement:.3e}"),
                &format_args!("{energy_change:.6e}"),
                &format_args!("{mean_force:.6e}"),
            ])?;
        }
        Ok(())
    }
}

impl<T: Context> Analyze<T> for VirtualTranslate {
    fn frequency(&self) -> Frequency {
        self.frequency
    }
    fn set_frequency(&mut self, freq: Frequency) {
        self.frequency = freq;
    }

    fn perform_sample(&mut self, context: &T, step: usize, weight: f64) -> Result<()> {
        if self.displacement.abs() < f64::EPSILON {
            return Ok(());
        }

        let gen = context.group_lists_generation();
        let selection = &self.selection;
        let active_groups = self
            .group_cache
            .get_or_resolve(gen, || context.resolve_groups_live(selection));

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

        let mut trial_context = context.clone();
        let energy_change = self.perturb(&mut trial_context, group_index)?;

        self.widom.collect(energy_change, weight);
        self.write_to_stream(step, energy_change)?;

        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.widom.len()
    }

    fn to_yaml(&self) -> Option<serde_yml::Value> {
        let mut map = serde_yml::Mapping::new();
        map.try_insert("displacement", self.displacement)?;
        map.try_insert("num_samples", self.widom.len())?;
        map.try_insert("mean_force", self.mean_force())?;
        map.try_insert("mean_free_energy", self.widom.mean_free_energy())?;
        Some(serde_yml::Value::Mapping(map))
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
        writeln!(f, "  Samples:     {}", self.widom.len())?;
        if !self.widom.is_empty() {
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

    const RT_298: f64 = crate::R_IN_KJ_PER_MOL * 298.15;

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
            .build(RT_298)
            .unwrap()
    }

    /// Deserialize YAML into AnalysisBuilder list and extract the VirtualTranslateBuilder at `index`.
    fn deserialize_vt_builder(yaml: &str, index: usize) -> VirtualTranslateBuilder {
        let builders: Vec<AnalysisBuilder> = serde_yml::from_str(yaml).unwrap();
        match &builders[index] {
            AnalysisBuilder::VirtualTranslate(b) => b.clone(),
            _ => panic!("expected VirtualTranslate variant"),
        }
    }

    #[test]
    fn test_axes_to_unit_vector() {
        assert_point_approx_eq!(axes_to_unit_vector(&Axes::Z).unwrap(), 0.0, 0.0, 1.0);
        assert_point_approx_eq!(axes_to_unit_vector(&Axes::X).unwrap(), 1.0, 0.0, 0.0);
        assert_point_approx_eq!(axes_to_unit_vector(&Axes::Y).unwrap(), 0.0, 1.0, 0.0);

        let s2 = 1.0 / 2.0_f64.sqrt();
        assert_point_approx_eq!(axes_to_unit_vector(&Axes::XY).unwrap(), s2, s2, 0.0);

        let s3 = 1.0 / 3.0_f64.sqrt();
        assert_point_approx_eq!(axes_to_unit_vector(&Axes::XYZ).unwrap(), s3, s3, s3);

        assert!(axes_to_unit_vector(&Axes::None).is_err());
    }

    #[test]
    fn build_with_valid_fields() {
        let vt = VirtualTranslateBuilder::default()
            .selection(Selection::parse("molecule MOL").unwrap())
            .displacement(0.01)
            .frequency(Frequency::Every(10))
            .build(RT_298)
            .unwrap();
        assert_approx_eq!(f64, vt.displacement, 0.01);
        assert_eq!(vt.selection.source(), "molecule MOL");
        assert_approx_eq!(f64, vt.thermal_energy, RT_298);
        assert_eq!(vt.directions, Axes::Z);
        assert_eq!(vt.widom.len(), 0);
    }

    #[test]
    fn build_missing_selection() {
        let result = VirtualTranslateBuilder::default()
            .displacement(0.01)
            .frequency(Frequency::Every(10))
            .build(RT_298);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("selection"));
    }

    #[test]
    fn build_missing_displacement() {
        let result = VirtualTranslateBuilder::default()
            .selection(Selection::parse("molecule MOL").unwrap())
            .frequency(Frequency::Every(10))
            .build(RT_298);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("dL"));
    }

    #[test]
    fn build_missing_frequency() {
        let result = VirtualTranslateBuilder::default()
            .selection(Selection::parse("molecule MOL").unwrap())
            .displacement(0.01)
            .build(RT_298);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("frequency"));
    }

    #[test]
    fn build_with_custom_direction() {
        let rt_310 = crate::R_IN_KJ_PER_MOL * 310.0;
        let vt = VirtualTranslateBuilder::default()
            .selection(Selection::parse("molecule MOL").unwrap())
            .displacement(0.05)
            .frequency(Frequency::Every(5))
            .directions(Axes::X)
            .build(rt_310)
            .unwrap();
        assert_approx_eq!(f64, vt.thermal_energy, rt_310);
        assert_eq!(vt.directions, Axes::X);
        assert_point_approx_eq!(vt.unit_direction, 1.0, 0.0, 0.0);
    }

    #[test]
    fn mean_free_energy_and_force() {
        let mut vt = build_vt(0.5);
        vt.widom.collect(0.0, 1.0);
        vt.widom.collect(0.0, 1.0);
        assert_approx_eq!(f64, vt.widom.mean_free_energy(), 0.0);
        assert_approx_eq!(f64, vt.mean_force(), 0.0);
    }

    #[test]
    fn mean_free_energy_with_nonzero_energy() {
        let mut vt = build_vt(0.1);
        // free_energy = -ln(exp(-2.0)) = 2.0, force = -2.0/0.1 = -20.0
        vt.widom.collect(2.0, 1.0);
        assert_approx_eq!(f64, vt.widom.mean_free_energy(), 2.0, epsilon = 1e-12);
        assert_approx_eq!(f64, vt.mean_force(), -20.0, epsilon = 1e-10);
    }

    #[test]
    fn mean_free_energy_empty_returns_infinity() {
        let vt = build_vt(0.1);
        assert!(vt.widom.mean_free_energy().is_infinite());
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
        vt.widom.collect(0.0, 1.0);
        let output = format!("{vt}");
        assert!(output.contains("Samples:     1"));
        assert!(output.contains("<force>"));
    }

    #[test]
    fn deserialize_virtual_translate_builders() {
        let yaml = std::fs::read_to_string("tests/files/virtual_translate.yaml").unwrap();

        let vt = deserialize_vt_builder(&yaml, 0).build(RT_298).unwrap();
        assert_eq!(vt.selection.source(), "molecule MOL");
        assert_approx_eq!(f64, vt.displacement, 0.01);
        assert_eq!(vt.directions, Axes::Z);
        assert!(matches!(vt.frequency, Frequency::Every(10)));

        let vt = deserialize_vt_builder(&yaml, 1).build(RT_298).unwrap();
        assert_approx_eq!(f64, vt.displacement, 0.05);
        assert_eq!(vt.directions, Axes::X);
        assert!(matches!(vt.frequency, Frequency::Every(5)));
    }

    #[test]
    fn deserialize_missing_required_fields() {
        let yaml = r#"
- !VirtualTranslate
  selection: "molecule MOL"
  frequency: !Every 10
"#;
        assert!(deserialize_vt_builder(yaml, 0).build(RT_298).is_err());
    }

    #[test]
    fn deserialize_default_direction_is_z() {
        let yaml = r#"
- !VirtualTranslate
  selection: "molecule MOL"
  dL: 0.1
  frequency: !Every 1
"#;
        let vt = deserialize_vt_builder(yaml, 0).build(RT_298).unwrap();
        assert_eq!(vt.directions, Axes::Z);
    }

    #[test]
    fn roundtrip_serialize_deserialize_builder() {
        let yaml = r#"
selection: "molecule MOL"
dL: 0.05
directions: !x
frequency: !Every 5
"#;
        let builder: VirtualTranslateBuilder = serde_yml::from_str(yaml).unwrap();
        let serialized = serde_yml::to_string(&builder).unwrap();
        let roundtrip: VirtualTranslateBuilder = serde_yml::from_str(&serialized).unwrap();
        let vt = roundtrip.build(RT_298).unwrap();
        assert_approx_eq!(f64, vt.displacement, 0.05);
        assert_eq!(vt.directions, Axes::X);
    }
}

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

//! Virtual volume move analysis for excess pressure measurement.
//!
//! Performs a virtual volume perturbation and measures the excess pressure
//! using the Widom method ([doi:10.1063/1.472721](https://doi.org/10.1063/1.472721)):
//!
//! ```text
//! Pex = kT * ln<exp(-dU/kT)> / dV
//! ```

use super::widom::WidomAccumulator;
use super::{Analyze, Frequency};
use crate::auxiliary::{ColumnWriter, MappingExt};
use crate::cell::{Shape, VolumeScalePolicy};
use crate::change::Change;
use crate::energy::EnergyChange;
use crate::Context;
use anyhow::Result;
use derive_builder::Builder;
use derive_more::Debug;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::path::PathBuf;

/// Virtual volume move analysis for excess pressure measurement.
///
/// Performs a virtual volume displacement and measures the excess pressure
/// by perturbation using the [Widom method](https://doi.org/10.1063/1.472721):
///
/// `Pex = kT * ln<exp(-dU/kT)> / dV`
///
/// where `dU` is the energy change and `dV` is the volume displacement.
/// All particle positions are scaled according to the chosen `method`.
#[derive(Debug, Builder)]
#[builder(build_fn(skip), derive(Deserialize, Serialize))]
pub struct VirtualVolumeMove {
    /// Volume displacement in Angstrom^3
    #[builder_field_attr(serde(rename = "dV"))]
    volume_displacement: f64,

    /// Volume scaling policy
    #[builder_field_attr(serde(default))]
    method: VolumeScalePolicy,

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

    /// Number of samples per block for variance estimation.
    #[allow(dead_code)]
    #[builder_field_attr(serde(default = "serde_default_block_size"))]
    block_size: usize,

    /// Widom exponential average accumulator (log-sum-exp). Drives block
    /// segmentation internally via [`WidomAccumulator::with_block_size`].
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    widom: WidomAccumulator,

    /// Thermal energy R*T in kJ/mol.
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    thermal_energy: f64,
}

fn default_block_size() -> usize {
    100
}

fn serde_default_block_size() -> Option<usize> {
    Some(default_block_size())
}

impl VirtualVolumeMoveBuilder {
    pub fn apply_output_dir(&mut self, dir: &std::path::Path) -> Result<()> {
        if let Some(path) = self.output_file.as_mut().and_then(Option::as_mut) {
            crate::analysis::prefix_in_place(path, dir)?;
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        if self.volume_displacement.is_none() {
            anyhow::bail!("Missing required field 'dV' for VirtualVolumeMove analysis");
        }
        if self.frequency.is_none() {
            anyhow::bail!("Missing required field 'frequency' for VirtualVolumeMove analysis");
        }
        Ok(())
    }

    /// Build the VirtualVolumeMove analysis.
    ///
    /// `thermal_energy` is R*T in kJ/mol, typically from `Medium::temperature()`.
    pub fn build(&self, thermal_energy: f64) -> Result<VirtualVolumeMove> {
        self.validate()?;

        let output_file = self.output_file.clone().flatten();
        let stream = output_file
            .as_deref()
            .map(|p| ColumnWriter::open(p, &["step", "dV/Å³", "dU/kT", "<Pex>/kT/Å³"]))
            .transpose()?;

        let block_size = self.block_size.unwrap_or_else(default_block_size);
        let block_size_nz = NonZeroUsize::new(block_size)
            .ok_or_else(|| anyhow::anyhow!("VirtualVolumeMove: 'block_size' must be > 0, got 0"))?;

        Ok(VirtualVolumeMove {
            volume_displacement: self.volume_displacement.unwrap(),
            method: self.method.unwrap_or_default(),
            output_file,
            stream,
            frequency: self.frequency.unwrap(),
            block_size,
            widom: WidomAccumulator::default().with_block_size(block_size_nz),
            thermal_energy,
        })
    }
}

impl crate::Info for VirtualVolumeMove {
    fn short_name(&self) -> Option<&'static str> {
        Some("virtualvolumemove")
    }

    fn long_name(&self) -> Option<&'static str> {
        Some("Virtual volume move for pressure measurement by perturbation")
    }

    fn citation(&self) -> Option<&'static str> {
        Some("doi:10.1063/1.472721")
    }
}

impl VirtualVolumeMove {
    /// Mean excess pressure in kT/Å³.
    fn mean_pressure(&self) -> f64 {
        if self.volume_displacement.abs() > f64::EPSILON {
            -self.widom.mean_free_energy() / self.volume_displacement
        } else {
            0.0
        }
    }

    /// Linear unit conversion from kT/Å³ to Pascal. Applies equally to a
    /// pressure mean and a pressure error (Var(cX) = c²Var(X)).
    ///
    /// P\[Pa\] = P\[kT/ų\] · kT\[J\] / (1 ų in m³).
    fn to_pascal(&self, p_kt_per_a3: f64) -> f64 {
        p_kt_per_a3 * self.thermal_energy * 1e6 / crate::MOLAR_TO_INV_ANGSTROM3
    }

    /// Linear unit conversion from kT/Å³ to millimolar (mM). Exploits
    /// P = c·kT, so c\[1/ų\] = P\[kT/ų\]. Applies equally to means and errors.
    fn to_millimolar(&self, p_kt_per_a3: f64) -> f64 {
        p_kt_per_a3 * 1e3 / crate::MOLAR_TO_INV_ANGSTROM3
    }

    /// Sample standard deviation of pressure across blocks in kT/Å³.
    ///
    /// Returns `None` until at least two blocks have been completed.
    fn pressure_stddev(&self) -> Option<f64> {
        if self.widom.free_energy().n() >= 2 {
            Some(self.widom.free_energy().stddev() / self.volume_displacement.abs())
        } else {
            None
        }
    }

    /// Perform the virtual volume perturbation and return the energy change in kT.
    ///
    /// `old_energy` is pre-computed on the original (immutable) context to avoid
    /// a redundant energy evaluation on the clone before mutation.
    fn perturb<T: Context>(&self, context: &mut T, old_energy: f64) -> Result<f64> {
        let old_volume = context
            .cell()
            .volume()
            .ok_or_else(|| anyhow::anyhow!("VirtualVolumeMove: cell has no defined volume"))?;
        let new_volume = old_volume + self.volume_displacement;

        context.scale_volume_and_positions(new_volume, self.method)?;
        // Ewald reciprocal-space caches k-vectors for the old box; must refresh before energy eval
        context.update(&Change::Everything)?;

        let new_energy = context.hamiltonian().energy(context, &Change::Everything);

        Ok((new_energy - old_energy) / self.thermal_energy)
    }

    /// One row per sampled step, keeping the file in sync with other analyses
    /// at the same frequency.
    fn write_to_stream(&mut self, step: usize, energy_change: f64) -> Result<()> {
        let mean_pressure = self.mean_pressure();
        let dv = self.volume_displacement;

        if let Some(stream) = self.stream.as_mut() {
            stream.write_row(&[
                &step,
                &format_args!("{dv:.3e}"),
                &format_args!("{energy_change:.6e}"),
                &format_args!("{mean_pressure:.6e}"),
            ])?;
        }
        Ok(())
    }
}

impl<T: Context> Analyze<T> for VirtualVolumeMove {
    fn frequency(&self) -> Frequency {
        self.frequency
    }
    fn set_frequency(&mut self, freq: Frequency) {
        self.frequency = freq;
    }

    fn perform_sample(&mut self, context: &T, step: usize, weight: f64) -> Result<()> {
        if self.volume_displacement.abs() < f64::EPSILON {
            return Ok(());
        }

        let old_energy = context.hamiltonian().energy(context, &Change::Everything);
        let mut trial_context = context.clone();
        let energy_change = self.perturb(&mut trial_context, old_energy)?;

        self.widom.collect(energy_change, weight);
        self.write_to_stream(step, energy_change)?;

        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.widom.len()
    }

    fn to_yaml(&self) -> Option<serde_yml::Value> {
        let mut map = serde_yml::Mapping::new();
        map.try_insert("dV", self.volume_displacement)?;
        map.try_insert("method", format!("{:?}", self.method))?;
        map.try_insert("num_samples", self.widom.len())?;
        map.try_insert("mean_free_energy", self.widom.mean_free_energy())?;

        // Pressure = -F / dV. Per-unit `{mean, error}` mappings come from
        // multiplying the free-energy BlockAverage by the pressure-unit
        // scale; variance scales by c² so the error scales by |c|.
        let fe = self.widom.free_energy();
        let s_kt = -1.0 / self.volume_displacement;
        map.try_insert("Pex (kT/Å³)", fe * s_kt)?;
        map.try_insert("Pex (Pa)", fe * self.to_pascal(s_kt))?;
        map.try_insert("Pex (mM)", fe * self.to_millimolar(s_kt))?;

        Some(serde_yml::Value::Mapping(map))
    }
}

impl<T: Context> From<VirtualVolumeMove> for Box<dyn Analyze<T>> {
    fn from(analysis: VirtualVolumeMove) -> Self {
        Box::new(analysis)
    }
}

impl std::fmt::Display for VirtualVolumeMove {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Virtual Volume Move Analysis:")?;
        writeln!(f, "  dV:          {} ų", self.volume_displacement)?;
        writeln!(f, "  Method:      {:?}", self.method)?;
        writeln!(f, "  Samples:     {}", self.widom.len())?;
        if !self.widom.is_empty() {
            writeln!(f, "  <Pex>:       {:.6} kT/ų", self.mean_pressure())?;
        }
        if let Some(s) = self.pressure_stddev() {
            writeln!(f, "  std(Pex):    {:.6} kT/ų", s)?;
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

    fn build_vvm(dv: f64) -> VirtualVolumeMove {
        VirtualVolumeMoveBuilder::default()
            .volume_displacement(dv)
            .frequency(Frequency::Every(1))
            .build(RT_298)
            .unwrap()
    }

    fn deserialize_vvm_builder(yaml: &str, index: usize) -> VirtualVolumeMoveBuilder {
        let builders: Vec<AnalysisBuilder> = serde_yml::from_str(yaml).unwrap();
        match &builders[index] {
            AnalysisBuilder::VirtualVolumeMove(b) => b.clone(),
            _ => panic!("expected VirtualVolumeMove variant"),
        }
    }

    #[test]
    fn apply_output_dir_prefixes_file() {
        let yaml = "
- !VirtualVolumeMove
  dV: 0.2
  file: pressure.csv
  frequency: !Every 10
";
        let mut builders: Vec<AnalysisBuilder> = serde_yml::from_str(yaml).unwrap();
        builders[0]
            .apply_output_dir(std::path::Path::new("box0"))
            .unwrap();
        let AnalysisBuilder::VirtualVolumeMove(b) = &builders[0] else {
            panic!("expected VirtualVolumeMove variant");
        };
        assert_eq!(
            b.output_file.clone().flatten(),
            Some(std::path::Path::new("box0").join("pressure.csv"))
        );
    }

    #[test]
    fn file_output_writes_header_and_row() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pressure.csv");
        let mut vvm = VirtualVolumeMoveBuilder::default()
            .volume_displacement(0.2)
            .output_file(Some(path.clone()))
            .frequency(Frequency::Every(1))
            .build(RT_298)
            .unwrap();
        vvm.write_to_stream(0, -1.5).unwrap();
        drop(vvm);

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert!(lines[0].contains("step"), "header missing: {contents:?}");
        assert_eq!(lines.len(), 2, "expected header + 1 row: {contents:?}");
    }

    #[test]
    fn build_with_valid_fields() {
        let vvm = VirtualVolumeMoveBuilder::default()
            .volume_displacement(0.5)
            .frequency(Frequency::Every(10))
            .build(RT_298)
            .unwrap();
        assert_approx_eq!(f64, vvm.volume_displacement, 0.5);
        assert_eq!(vvm.method, VolumeScalePolicy::Isotropic);
        assert!(vvm.widom.is_empty());
    }

    #[test]
    fn build_missing_dv() {
        let result = VirtualVolumeMoveBuilder::default()
            .frequency(Frequency::Every(10))
            .build(RT_298);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("dV"));
    }

    #[test]
    fn build_missing_frequency() {
        let result = VirtualVolumeMoveBuilder::default()
            .volume_displacement(0.5)
            .build(RT_298);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("frequency"));
    }

    #[test]
    fn build_rejects_zero_block_size() {
        let yaml = "
- !VirtualVolumeMove
  dV: 0.5
  frequency: !Every 1
  block_size: 0
";
        let result = deserialize_vvm_builder(yaml, 0).build(RT_298);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("block_size"));
    }

    #[test]
    fn build_with_custom_method() {
        let vvm = VirtualVolumeMoveBuilder::default()
            .volume_displacement(1.0)
            .frequency(Frequency::Every(5))
            .method(VolumeScalePolicy::ScaleZ)
            .build(RT_298)
            .unwrap();
        assert_eq!(vvm.method, VolumeScalePolicy::ScaleZ);
    }

    #[test]
    fn mean_pressure_zero_energy() {
        let mut vvm = build_vvm(0.5);
        vvm.widom.collect(0.0, 1.0);
        vvm.widom.collect(0.0, 1.0);
        assert_approx_eq!(f64, vvm.mean_pressure(), 0.0);
    }

    #[test]
    fn mean_pressure_nonzero_energy() {
        let mut vvm = build_vvm(0.2);
        // dU = 2.0 kT → free_energy = 2.0 → Pex = -2.0 / 0.2 = -10.0 kT/ų
        vvm.widom.collect(2.0, 1.0);
        assert_approx_eq!(f64, vvm.mean_pressure(), -10.0, epsilon = 1e-10);
    }

    #[test]
    fn mean_pressure_zero_dv() {
        let vvm = build_vvm(0.0);
        assert_approx_eq!(f64, vvm.mean_pressure(), 0.0);
    }

    #[test]
    fn pressure_unit_conversions() {
        let vvm = build_vvm(1.0);
        // 1 kT/ų at 298.15 K
        let p_pa = vvm.to_pascal(1.0);
        let p_mm = vvm.to_millimolar(1.0);
        // kT(298.15 K) ≈ 4.116e-21 J, 1 ų = 1e-30 m³ → 1 kT/ų ≈ 4.116e9 Pa
        assert_approx_eq!(f64, p_pa, 4.116e9, epsilon = 5.0e6);
        // 1 1/ų = 1/(N_A · 1e-27 L) ≈ 1.66e3 mol/L = 1.66e6 mM
        assert_approx_eq!(f64, p_mm, 1.66e6, epsilon = 1.0e4);
    }

    #[test]
    fn info_trait() {
        let vvm = build_vvm(0.5);
        assert_eq!(vvm.short_name(), Some("virtualvolumemove"));
        assert!(vvm.long_name().unwrap().contains("pressure"));
        assert!(vvm.citation().unwrap().starts_with("doi:"));
    }

    #[test]
    fn display_without_samples() {
        let vvm = build_vvm(0.5);
        let output = format!("{vvm}");
        assert!(output.contains("0.5"));
        assert!(output.contains("Samples:     0"));
        assert!(!output.contains("<Pex>"));
    }

    #[test]
    fn display_with_samples() {
        let mut vvm = build_vvm(0.5);
        vvm.widom.collect(0.0, 1.0);
        let output = format!("{vvm}");
        assert!(output.contains("Samples:     1"));
        assert!(output.contains("<Pex>"));
    }

    #[test]
    fn deserialize_virtual_volume_move_builders() {
        let yaml = std::fs::read_to_string("tests/files/virtual_volume_move.yaml").unwrap();

        let vvm = deserialize_vvm_builder(&yaml, 0).build(RT_298).unwrap();
        assert_approx_eq!(f64, vvm.volume_displacement, 0.5);
        assert_eq!(vvm.method, VolumeScalePolicy::Isotropic);
        assert!(matches!(vvm.frequency, Frequency::Every(10)));

        let vvm = deserialize_vvm_builder(&yaml, 1).build(RT_298).unwrap();
        assert_approx_eq!(f64, vvm.volume_displacement, 1.0);
        assert_eq!(vvm.method, VolumeScalePolicy::ScaleZ);
        assert!(matches!(vvm.frequency, Frequency::Every(5)));
    }

    #[test]
    fn pressure_stddev_no_blocks() {
        let mut vvm = build_vvm(0.5);
        vvm.widom.collect(1.0, 1.0);
        assert!(vvm.pressure_stddev().is_none());
    }

    #[test]
    fn pressure_stddev_with_two_blocks() {
        let mut vvm = build_vvm(0.5);
        // Block 1: dU = 0 → free_energy = 0
        vvm.widom.collect(0.0, 1.0);
        vvm.widom.end_block();
        // Block 2: dU = 2 → free_energy = 2
        vvm.widom.collect(2.0, 1.0);
        vvm.widom.end_block();
        // stddev of [0, 2] = 1.414..., Pex_std = 1.414... / 0.5 ≈ 2.828
        let s = vvm.pressure_stddev().expect("two blocks should yield Some");
        assert!(s > 2.0 && s < 4.0);
    }

    #[test]
    fn pressure_stddev_via_perform_sample() {
        // Uses build_vvm which sets block_size = 100 (default).
        // We need to drive 200 samples through perform_sample without a real Context,
        // so instead wire up end_block directly via the public API.
        let mut vvm = VirtualVolumeMoveBuilder::default()
            .volume_displacement(1.0)
            .frequency(Frequency::Every(1))
            .block_size(2)
            .build(RT_298)
            .unwrap();
        assert_eq!(vvm.block_size, 2);
        // Manually simulate what perform_sample does after perturb():
        for _ in 0..2 {
            vvm.widom.collect(0.0, 1.0);
            vvm.widom.collect(2.0, 1.0);
            vvm.widom.end_block();
        }
        assert!(vvm.widom.free_energy().n() >= 2);
        assert!(vvm.pressure_stddev().is_some());
    }

    #[test]
    fn to_yaml_emits_pex_mapping_per_unit() {
        let mut vvm = VirtualVolumeMoveBuilder::default()
            .volume_displacement(0.5)
            .frequency(Frequency::Every(1))
            .build(RT_298)
            .unwrap();
        // Close two distinct blocks so the BlockAverage has finite mean and SEM.
        vvm.widom.collect(0.0, 1.0);
        vvm.widom.end_block();
        vvm.widom.collect(2.0, 1.0);
        vvm.widom.end_block();

        let yaml = <VirtualVolumeMove as Analyze<crate::backend::Backend>>::to_yaml(&vvm)
            .expect("to_yaml returns Some");
        let map = yaml.as_mapping().expect("top-level mapping");

        for key in ["Pex (kT/Å³)", "Pex (Pa)", "Pex (mM)"] {
            let entry = map.get(key).unwrap_or_else(|| panic!("missing {key}"));
            let parsed: crate::auxiliary::BlockSummary =
                serde_yml::from_value(entry.clone()).expect("entry parses as BlockSummary");
            assert!(parsed.mean.is_finite(), "{key} mean must be finite");
            assert!(parsed.error.is_finite(), "{key} error must be finite");
            assert!(parsed.error >= 0.0, "{key} error must be non-negative");
        }
    }

    #[test]
    fn deserialize_custom_block_size() {
        let yaml = r#"
- !VirtualVolumeMove
  dV: 0.5
  frequency: !Every 10
  block_size: 50
"#;
        let vvm = deserialize_vvm_builder(yaml, 0).build(RT_298).unwrap();
        assert_eq!(vvm.block_size, 50);
    }

    #[test]
    fn deserialize_default_block_size() {
        let yaml = r#"
- !VirtualVolumeMove
  dV: 0.5
  frequency: !Every 10
"#;
        let vvm = deserialize_vvm_builder(yaml, 0).build(RT_298).unwrap();
        assert_eq!(vvm.block_size, 100);
    }

    #[test]
    fn deserialize_missing_required_fields() {
        let yaml = r#"
- !VirtualVolumeMove
  method: Isotropic
  frequency: !Every 10
"#;
        assert!(deserialize_vvm_builder(yaml, 0).build(RT_298).is_err());
    }

    #[test]
    fn deserialize_default_method_is_isotropic() {
        let yaml = r#"
- !VirtualVolumeMove
  dV: 0.5
  frequency: !Every 1
"#;
        let vvm = deserialize_vvm_builder(yaml, 0).build(RT_298).unwrap();
        assert_eq!(vvm.method, VolumeScalePolicy::Isotropic);
    }

    #[test]
    fn roundtrip_serialize_deserialize_builder() {
        let yaml = r#"
dV: 0.5
method: ScaleZ
frequency: !Every 5
"#;
        let builder: VirtualVolumeMoveBuilder = serde_yml::from_str(yaml).unwrap();
        let serialized = serde_yml::to_string(&builder).unwrap();
        let roundtrip: VirtualVolumeMoveBuilder = serde_yml::from_str(&serialized).unwrap();
        let vvm = roundtrip.build(RT_298).unwrap();
        assert_approx_eq!(f64, vvm.volume_displacement, 0.5);
        assert_eq!(vvm.method, VolumeScalePolicy::ScaleZ);
    }
}

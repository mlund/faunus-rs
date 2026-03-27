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
use crate::auxiliary::MappingExt;
use crate::cell::{Shape, VolumeScalePolicy};
use crate::change::Change;
use crate::energy::EnergyChange;
use crate::Context;
use anyhow::Result;
use derive_builder::Builder;
use derive_more::Debug;
use serde::{Deserialize, Serialize};

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

    /// Sample frequency
    frequency: Frequency,

    /// Widom exponential average accumulator (log-sum-exp)
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    widom: WidomAccumulator,

    /// Thermal energy R*T in kJ/mol.
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    thermal_energy: f64,
}

impl VirtualVolumeMoveBuilder {
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

        Ok(VirtualVolumeMove {
            volume_displacement: self.volume_displacement.unwrap(),
            method: self.method.unwrap_or_default(),
            frequency: self.frequency.unwrap(),
            widom: WidomAccumulator::new(),
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

    /// Convert pressure from kT/Å³ to Pascal.
    ///
    /// P\[Pa\] = P\[kT/ų\] · kT\[J\] / (1 ų in m³).
    fn pressure_to_pa(&self, p_kt_per_a3: f64) -> f64 {
        p_kt_per_a3 * self.thermal_energy * 1e6 / crate::MOLAR_TO_INV_ANGSTROM3
    }

    /// Convert pressure from kT/Å³ to millimolar (mM).
    ///
    /// Exploits P = c·kT, so c\[1/ų\] = P\[kT/ų\].
    fn pressure_to_mm(&self, p_kt_per_a3: f64) -> f64 {
        p_kt_per_a3 * 1e3 / crate::MOLAR_TO_INV_ANGSTROM3
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
}

impl<T: Context> Analyze<T> for VirtualVolumeMove {
    fn frequency(&self) -> Frequency {
        self.frequency
    }
    fn set_frequency(&mut self, freq: Frequency) {
        self.frequency = freq;
    }

    fn perform_sample(&mut self, context: &T, _step: usize, weight: f64) -> Result<()> {
        if self.volume_displacement.abs() < f64::EPSILON {
            return Ok(());
        }

        let old_energy = context.hamiltonian().energy(context, &Change::Everything);
        let mut trial_context = context.clone();
        let energy_change = self.perturb(&mut trial_context, old_energy)?;

        self.widom.collect(energy_change, weight);

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

        let p = self.mean_pressure();
        map.try_insert("Pex/kT/Å³", p)?;
        map.try_insert("Pex/Pa", self.pressure_to_pa(p))?;
        map.try_insert("Pex/mM", self.pressure_to_mm(p))?;

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
        let p_pa = vvm.pressure_to_pa(1.0);
        let p_mm = vvm.pressure_to_mm(1.0);
        // kT at 298.15 K ≈ 4.116e-21 J, 1 ų = 1e-30 m³
        // P ≈ 4.116e-21 / 1e-30 ≈ 4.116e9 Pa
        assert!(p_pa > 4.0e9 && p_pa < 4.2e9);
        // c = 1/ų → mol/L = 1/(N_A * 1e-27) ≈ 1.66e3 mol/L → 1.66e6 mM
        assert!(p_mm > 1.6e6 && p_mm < 1.7e6);
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

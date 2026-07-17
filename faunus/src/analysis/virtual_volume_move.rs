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
use super::{Analyze, Sampling};
use crate::auxiliary::{BlockSummary, ColumnWriter, MappingExt};
use crate::cell::{Shape, VolumeScalePolicy};
use crate::change::Change;
use crate::context::{PerturbContext, Perturbation};
use crate::energy::EnergyChange;
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
#[builder_struct_attr(serde(deny_unknown_fields))]
pub struct VirtualVolumeMove {
    /// Volume displacement in Angstrom^3
    #[builder_field_attr(serde(rename = "volume_displacement", alias = "dV"))]
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

    /// Frequency and frame count, owned by the framework. Deserialized from `frequency`.
    #[builder(setter(name = "frequency", into))]
    #[builder_field_attr(serde(rename = "frequency"))]
    sampling: Sampling,

    /// Number of samples per block for variance estimation.
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

const DEFAULT_BLOCK_SIZE: usize = 100;

fn serde_default_block_size() -> Option<usize> {
    Some(DEFAULT_BLOCK_SIZE)
}

impl VirtualVolumeMoveBuilder {
    pub fn apply_output_dir(&mut self, dir: &std::path::Path) -> Result<()> {
        if let Some(path) = self.output_file.as_mut().and_then(Option::as_mut) {
            crate::analysis::prefix_in_place(path, dir)?;
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        let dv = self.volume_displacement.ok_or_else(|| {
            anyhow::anyhow!(
                "Missing required field 'volume_displacement' for VirtualVolumeMove analysis"
            )
        })?;
        if dv.abs() < f64::EPSILON {
            anyhow::bail!("VirtualVolumeMove: 'volume_displacement' must be non-zero, got {dv}");
        }
        if self.sampling.is_none() {
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

        let block_size = self.block_size.unwrap_or(DEFAULT_BLOCK_SIZE);
        let block_size_nz = NonZeroUsize::new(block_size)
            .ok_or_else(|| anyhow::anyhow!("VirtualVolumeMove: 'block_size' must be > 0, got 0"))?;

        Ok(VirtualVolumeMove {
            volume_displacement: self.volume_displacement.unwrap(),
            method: self.method.unwrap_or_default(),
            output_file,
            stream,
            sampling: self.sampling.unwrap(),
            block_size,
            widom: WidomAccumulator::new(block_size_nz),
            thermal_energy,
        })
    }
}

impl_info!(
    VirtualVolumeMove,
    "virtual_volume_move",
    "Virtual volume move for pressure measurement by perturbation",
    "doi:10.1063/1.472721"
);

impl VirtualVolumeMove {
    /// Mean excess pressure in kT/Å³. `dV ≠ 0` is enforced at build time.
    fn mean_pressure(&self) -> f64 {
        -self.widom.mean_free_energy() / self.volume_displacement
    }

    /// kT/Å³ → Pascal. Linear, so applies equally to means and errors
    /// (Var(cX) = c²Var(X)).
    fn to_pascal(&self, p_kt_per_a3: f64) -> f64 {
        p_kt_per_a3 * self.thermal_energy * 1e6 / crate::MOLAR_TO_INV_ANGSTROM3
    }

    /// kT/Å³ → millimolar. Linear, so applies equally to means and errors.
    /// Exploits P = c·kT, so c[1/Å³] = P[kT/Å³].
    fn to_millimolar(&self, p_kt_per_a3: f64) -> f64 {
        p_kt_per_a3 * 1e3 / crate::MOLAR_TO_INV_ANGSTROM3
    }

    /// Perform the virtual volume perturbation and return the energy change in kT.
    ///
    /// `old_energy` is pre-computed on the original (immutable) context to avoid
    /// a redundant energy evaluation on the clone before mutation.
    fn perturb<T: PerturbContext>(&self, context: &mut T, old_energy: f64) -> Result<f64> {
        let old_volume = context
            .cell()
            .volume()
            .ok_or_else(|| anyhow::anyhow!("VirtualVolumeMove: cell has no defined volume"))?;
        let new_volume = old_volume + self.volume_displacement;

        let new_energy = context.measure(
            &Perturbation::ScaleVolume {
                volume: new_volume,
                policy: self.method,
            },
            |scaled, change| scaled.hamiltonian().energy(scaled, change),
        )?;

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

impl<T: PerturbContext> Analyze<T> for VirtualVolumeMove {
    impl_sampling_accessors!();

    fn perform_sample(&mut self, context: &T, step: usize, weight: f64) -> Result<()> {
        let old_energy = context.hamiltonian().energy(context, &Change::Everything);
        let mut trial_context = context.clone();
        let energy_change = self.perturb(&mut trial_context, old_energy)?;

        self.widom.collect(energy_change, weight);
        self.write_to_stream(step, energy_change)?;

        Ok(())
    }

    fn results(&self) -> Option<serde_yml::Value> {
        if self.widom.is_empty() {
            return None;
        }
        let mut map = serde_yml::Mapping::new();
        map.try_insert("volume_displacement", self.volume_displacement)?;
        map.try_insert("method", format!("{:?}", self.method))?;
        map.try_insert("block_size", self.block_size)?;
        map.try_insert("num_samples", self.sampling.num_samples())?;
        map.try_insert("num_perturbations", self.widom.len())?;
        map.try_insert("mean_free_energy", self.widom.mean_free_energy())?;

        // Mean comes from the total accumulator (finite from sample 1);
        // error comes from the block aggregator (NaN until ≥ 1 block
        // closes, ~0 with 1 block, real SEM with ≥ 2). dV ≠ 0 is
        // enforced at build time.
        let mean_kt = self.mean_pressure();
        let err_kt = self.widom.free_energy().error() / self.volume_displacement.abs();
        let pex_units = [
            ("Pex/kT/Å³", mean_kt, err_kt),
            ("Pex/Pa", self.to_pascal(mean_kt), self.to_pascal(err_kt)),
            (
                "Pex/mM",
                self.to_millimolar(mean_kt),
                self.to_millimolar(err_kt),
            ),
        ];
        for (key, m, e) in pex_units {
            map.try_insert(key, BlockSummary { mean: m, error: e })?;
        }

        Some(serde_yml::Value::Mapping(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisBuilder;
    use crate::analysis::Frequency;
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

    /// The pre-3.0 spelling of `volume_displacement` is still accepted, so that older input keeps
    /// running. Nothing else pins it: the inputs under `tests/` all use the canonical key.
    #[test]
    fn legacy_dv_key_is_still_accepted() {
        let builder = deserialize_vvm_builder(
            "- !VirtualVolumeMove {dV: 0.2, file: pressure.csv, frequency: !Every 10}",
            0,
        );
        assert_approx_eq!(f64, builder.build(RT_298).unwrap().volume_displacement, 0.2);
    }

    #[test]
    fn apply_output_dir_prefixes_file() {
        let yaml = "
- !VirtualVolumeMove
  volume_displacement: 0.2
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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("volume_displacement"));
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
  volume_displacement: 0.5
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
    fn build_rejects_zero_dv() {
        let result = VirtualVolumeMoveBuilder::default()
            .volume_displacement(0.0)
            .frequency(Frequency::Every(1))
            .build(RT_298);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("volume_displacement"));
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
        assert_eq!(vvm.short_name(), Some("virtual_volume_move"));
        assert!(vvm.long_name().unwrap().contains("pressure"));
        assert!(vvm.citation().unwrap().starts_with("doi:"));
    }

    #[test]
    fn deserialize_virtual_volume_move_builders() {
        let yaml = std::fs::read_to_string("tests/files/virtual_volume_move.yaml").unwrap();

        let vvm = deserialize_vvm_builder(&yaml, 0).build(RT_298).unwrap();
        assert_approx_eq!(f64, vvm.volume_displacement, 0.5);
        assert_eq!(vvm.method, VolumeScalePolicy::Isotropic);
        assert!(matches!(vvm.sampling.frequency(), Frequency::Every(10)));

        let vvm = deserialize_vvm_builder(&yaml, 1).build(RT_298).unwrap();
        assert_approx_eq!(f64, vvm.volume_displacement, 1.0);
        assert_eq!(vvm.method, VolumeScalePolicy::ScaleZ);
        assert!(matches!(vvm.sampling.frequency(), Frequency::Every(5)));
    }

    #[test]
    fn pressure_stddev_via_perform_sample() {
        // A block size of 2 lets two collects close a block, so this reaches the same
        // state perform_sample would after 200 samples at the default size.
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
        }
        assert!(vvm.widom.free_energy().n() >= 2);
        assert!(vvm.widom.free_energy().stddev().is_finite());
    }

    #[test]
    fn to_yaml_emits_pex_mapping_per_unit() {
        // Size 1 so sampling alone closes each block, as production does.
        let mut vvm = VirtualVolumeMoveBuilder::default()
            .volume_displacement(0.5)
            .frequency(Frequency::Every(1))
            .block_size(1)
            .build(RT_298)
            .unwrap();
        vvm.widom.collect(0.0, 1.0);
        vvm.widom.collect(2.0, 1.0);
        // The accumulator is driven directly here, so tell the framework two frames were sampled.
        vvm.sampling.set_num_samples(2);

        let yaml = <VirtualVolumeMove as Analyze<crate::backend::Backend>>::to_yaml(&vvm)
            .expect("to_yaml returns Some");
        let map = yaml.as_mapping().expect("top-level mapping");

        for key in ["Pex/kT/Å³", "Pex/Pa", "Pex/mM"] {
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
  volume_displacement: 0.5
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
  volume_displacement: 0.5
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
  volume_displacement: 0.5
  frequency: !Every 1
"#;
        let vvm = deserialize_vvm_builder(yaml, 0).build(RT_298).unwrap();
        assert_eq!(vvm.method, VolumeScalePolicy::Isotropic);
    }

    #[test]
    fn roundtrip_serialize_deserialize_builder() {
        let yaml = r#"
volume_displacement: 0.5
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

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::analysis::Frequency;
    use crate::backend::Backend;
    use crate::context::{Context, WithHamiltonian, WithSimulationCell};
    use crate::energy::EnergyChange;
    use float_cmp::assert_approx_eq;

    const RT_300: f64 = crate::R_IN_KJ_PER_MOL * 300.0;

    /// Two Lennard-Jones molecules: `nonbonded` is stateful, so a missed refresh would show.
    fn lj_backend() -> Backend {
        Backend::from_yaml_str(
            r#"
atoms:
  - {name: A, mass: 1.0, charge: 0.0}
molecules:
  - name: MOL
    atoms: [A]
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy:
    nonbonded:
      default:
        - !LennardJones {sigma: 3.0, eps: 2.5}
  blocks:
    - molecule: MOL
      N: 2
      insert: !Manual [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#,
            None,
            &mut rand::thread_rng(),
        )
        .unwrap()
    }

    #[test]
    fn perturbation_energy_matches_the_true_energy_change() {
        let context = lj_backend();
        let vvm = VirtualVolumeMoveBuilder::default()
            .volume_displacement(10.0)
            .frequency(Frequency::Every(1))
            .build(RT_300)
            .unwrap();

        let total = |ctx: &Backend| ctx.hamiltonian().energy(ctx, &Change::Everything);
        let old_energy = total(&context);

        let mut reference = context.clone();
        let old_volume = reference.cell().volume().unwrap();
        reference
            .scale_volume_and_positions(old_volume + 10.0, vvm.method)
            .unwrap();
        reference.update(&Change::Everything).unwrap();
        let expected = (total(&reference) - old_energy) / RT_300;

        let mut trial = context.clone();
        let measured = vvm.perturb(&mut trial, old_energy).unwrap();

        assert!(
            expected.abs() > 1e-6,
            "the test system must have a real ΔU to measure, got {expected}"
        );
        assert_approx_eq!(f64, measured, expected, epsilon = 1e-12);
    }
}

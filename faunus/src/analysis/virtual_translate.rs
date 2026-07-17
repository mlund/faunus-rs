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
use super::{Analyze, Sampling};
use crate::auxiliary::{BlockSummary, ColumnWriter, MappingExt};
use crate::axes::Axes;
use crate::context::{PerturbContext, Perturbation};
use crate::energy::EnergyChange;
use crate::selection::{CachedSelection, Groups, Selection};
use crate::Point;
use anyhow::Result;
use derive_builder::Builder;
use derive_more::Debug;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
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
#[builder_struct_attr(serde(deny_unknown_fields))]
pub struct VirtualTranslate {
    /// Selection expression for the molecule to translate
    selection: Selection,

    /// Displacement magnitude in Angstrom. Canonical `displacement` matches the
    /// `volume_displacement` moves; `dL` kept as their `dV`-style short alias.
    #[builder_field_attr(serde(alias = "dL"))]
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

    /// Frequency and frame count, owned by the framework. Deserialized from `frequency`.
    #[builder(setter(name = "frequency", into))]
    #[builder_field_attr(serde(rename = "frequency"))]
    sampling: Sampling,

    /// Widom exponential average accumulator (log-sum-exp)
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    widom: WidomAccumulator,

    /// The selection, its resolved groups, and its cache key. Built with the analysis.
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    group_cache: CachedSelection<Groups>,

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
    pub fn apply_output_dir(&mut self, dir: &std::path::Path) -> Result<()> {
        if let Some(path) = self.output_file.as_mut().and_then(Option::as_mut) {
            crate::analysis::prefix_in_place(path, dir)?;
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        if self.selection.is_none() {
            anyhow::bail!("Missing required field 'selection' for VirtualTranslate analysis");
        }
        if self.displacement.is_none() {
            anyhow::bail!("Missing required field 'displacement' for VirtualTranslate analysis");
        }
        if self.sampling.is_none() {
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
            sampling: self.sampling.unwrap(),
            // One block per sampled configuration: each frame yields one force estimate,
            // and the hierarchy then finds the correlation scale on its own.
            widom: WidomAccumulator::new(NonZeroUsize::MIN),
            group_cache: CachedSelection::groups(self.selection.as_ref().unwrap().clone()),
            thermal_energy,
        })
    }
}

impl_info!(
    VirtualTranslate,
    "virtual_translate",
    "Virtual translate move for force measurement by perturbation",
    "doi:10.1063/1.1734110" // Widom insertion method
);

impl VirtualTranslate {
    /// Mean force ± SEM in units of kT/Å.
    ///
    /// Mean comes from the total accumulator (finite from sample 1); the error comes from
    /// the block aggregator, which blocks one free energy per sampled configuration and
    /// corrects for correlation between them once the run is long enough to resolve it.
    /// Both scale by 1/dL, the error by |dL| since the sign carries no information about
    /// the spread.
    ///
    /// With one sample per block the exponential average collapses to `F = dU`, so this is
    /// the SEM of ⟨dU⟩ rather than of `-ln⟨exp(-dU)⟩`; the two agree to leading order in
    /// dL, which is the only regime where a finite-difference force is meaningful. See
    /// issue #117.
    fn mean_force(&self) -> BlockSummary {
        if self.displacement.abs() > f64::EPSILON {
            BlockSummary {
                mean: -self.widom.mean_free_energy() / self.displacement,
                error: self.widom.free_energy().error() / self.displacement.abs(),
            }
        } else {
            BlockSummary::default()
        }
    }

    /// Perform the virtual perturbation and return the energy change in kT.
    ///
    /// Both energies are the molecule's interaction with its surroundings, one displaced and one
    /// not. Both are read against the *same* change — the one the displacement derives — or the
    /// difference would subtract two different quantities. The unperturbed one needs no refresh:
    /// nothing has moved, so the caches still describe the coordinates.
    fn perturb<T: PerturbContext>(&self, context: &mut T, group_index: usize) -> Result<f64> {
        let displace = Perturbation::Translate {
            group: group_index,
            shift: self.displacement * self.unit_direction,
        };

        let old_energy = context.hamiltonian().energy(context, &displace.change()); // kJ/mol
        let new_energy = context.measure(&displace, |displaced, change| {
            displaced.hamiltonian().energy(displaced, change) // kJ/mol
        })?;

        Ok((new_energy - old_energy) / self.thermal_energy)
    }

    /// One row per sampled step, keeping the file in sync with other analyses
    /// at the same frequency.
    fn write_to_stream(&mut self, step: usize, energy_change: f64) -> Result<()> {
        let displacement = self.displacement;

        if let Some(stream) = self.stream.as_mut() {
            // The column wants the mean alone; `mean_force()` would also run the whole
            // plateau scan per sampled step only to discard it. `perform_sample` has
            // already rejected a zero displacement.
            let mean_force = -self.widom.mean_free_energy() / displacement;
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

impl<T: PerturbContext> Analyze<T> for VirtualTranslate {
    impl_sampling_accessors!();

    fn perform_sample(&mut self, context: &T, step: usize, weight: f64) -> Result<()> {
        if self.displacement.abs() < f64::EPSILON {
            return Ok(());
        }

        let active_groups = self.group_cache.resolve(context);

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

        let group_index = active_groups[0].get();

        let mut trial_context = context.clone();
        let energy_change = self.perturb(&mut trial_context, group_index)?;

        self.widom.collect(energy_change, weight);
        self.write_to_stream(step, energy_change)?;

        Ok(())
    }

    fn results(&self) -> Option<serde_yml::Value> {
        // A frame is counted even when `perform_sample` returns early, so the framework's
        // frame-count guard is not enough: key on the accumulator that feeds every number below.
        if self.widom.is_empty() {
            return None;
        }
        let mut map = serde_yml::Mapping::new();
        map.try_insert("displacement", self.displacement)?;
        map.try_insert("num_samples", self.sampling.num_samples())?;
        map.try_insert("num_perturbations", self.widom.len())?;
        map.try_insert("mean_force", self.mean_force())?;
        map.try_insert("mean_free_energy", self.widom.mean_free_energy())?;
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

    /// A frame can be counted without contributing: `perform_sample` returns early when the
    /// displacement is zero or nothing matches the selection. Reporting must key on the
    /// accumulator, not the frame counter, or an empty Widom average (+inf) reaches output.yaml.
    #[test]
    fn counted_but_uncollected_frames_do_not_publish_infinities() {
        let mut vt = build_vt(0.0); // zero displacement: perform_sample early-returns
        vt.sampling.set_num_samples(5); // ...yet five frames were counted

        assert!(vt.widom.is_empty());
        assert!(
            <VirtualTranslate as Analyze<crate::backend::Backend>>::to_yaml(&vt).is_none(),
            "an empty accumulator must not be reported"
        );
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
        assert!(result.unwrap_err().to_string().contains("displacement"));
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
        assert_approx_eq!(f64, vt.mean_force().mean, 0.0);
    }

    #[test]
    fn mean_free_energy_with_nonzero_energy() {
        let mut vt = build_vt(0.1);
        // free_energy = -ln(exp(-2.0)) = 2.0, force = -2.0/0.1 = -20.0
        vt.widom.collect(2.0, 1.0);
        assert_approx_eq!(f64, vt.widom.mean_free_energy(), 2.0, epsilon = 1e-12);
        assert_approx_eq!(f64, vt.mean_force().mean, -20.0, epsilon = 1e-10);
    }

    #[test]
    fn mean_free_energy_empty_returns_infinity() {
        let vt = build_vt(0.1);
        assert!(vt.widom.mean_free_energy().is_infinite());
    }

    #[test]
    fn mean_force_zero_displacement() {
        let vt = build_vt(0.0);
        let force = vt.mean_force();
        assert_approx_eq!(f64, force.mean, 0.0);
        assert_approx_eq!(f64, force.error, 0.0);
    }

    /// Sampling alone must produce a spread. Closing blocks by hand here would prove
    /// nothing about the built analysis: it is exactly what production never did, leaving
    /// the aggregator empty and `mean_force` reporting an exact zero uncertainty.
    #[test]
    fn sampling_alone_yields_a_finite_force_uncertainty() {
        let mut vt = build_vt(0.1);
        vt.widom.collect(0.0, 1.0);
        vt.widom.collect(2.0, 1.0);
        // The accumulator is driven directly here, so tell the framework two frames were sampled.
        vt.sampling.set_num_samples(2);

        assert_eq!(
            vt.widom.free_energy().n(),
            2,
            "each sample closes its own block"
        );
        assert!(
            vt.mean_force().error > 0.0,
            "two differing samples must not report an exact zero uncertainty"
        );

        let yaml = <VirtualTranslate as Analyze<crate::backend::Backend>>::to_yaml(&vt)
            .expect("to_yaml returns Some");
        let entry = yaml
            .as_mapping()
            .and_then(|map| map.get("mean_force"))
            .expect("mean_force entry");
        let parsed: BlockSummary =
            serde_yml::from_value(entry.clone()).expect("mean_force parses as BlockSummary");
        assert!(parsed.mean.is_finite(), "mean must be finite");
        assert!(parsed.error.is_finite(), "error must be finite");
        assert!(parsed.error >= 0.0, "error must be non-negative");
    }

    #[test]
    fn info_trait() {
        let vt = build_vt(0.01);
        assert_eq!(vt.short_name(), Some("virtual_translate"));
        assert!(vt.long_name().unwrap().contains("force measurement"));
        assert!(vt.citation().unwrap().starts_with("doi:"));
    }

    #[test]
    fn deserialize_virtual_translate_builders() {
        let yaml = std::fs::read_to_string("tests/files/virtual_translate.yaml").unwrap();

        let vt = deserialize_vt_builder(&yaml, 0).build(RT_298).unwrap();
        assert_eq!(vt.selection.source(), "molecule MOL");
        assert_approx_eq!(f64, vt.displacement, 0.01);
        assert_eq!(vt.directions, Axes::Z);
        assert!(matches!(vt.sampling.frequency(), Frequency::Every(10)));

        let vt = deserialize_vt_builder(&yaml, 1).build(RT_298).unwrap();
        assert_approx_eq!(f64, vt.displacement, 0.05);
        assert_eq!(vt.directions, Axes::X);
        assert!(matches!(vt.sampling.frequency(), Frequency::Every(5)));
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
  displacement: 0.1
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

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::analysis::Frequency;
    use crate::backend::Backend;
    use crate::change::Change;
    use crate::context::{Context, ObserveContext, WithHamiltonian};
    use crate::energy::EnergyChange;
    use float_cmp::assert_approx_eq;

    const RT_300: f64 = crate::R_IN_KJ_PER_MOL * 300.0;

    /// A Lennard-Jones pair: the tagged molecule at the origin and a neighbour 4 Å up the z-axis.
    ///
    /// `nonbonded` is a stateful term — its group energies are cached — so a stale cache shows up
    /// here as a force of exactly zero. A stateless external field cannot expose that.
    fn lj_pair_backend() -> Backend {
        Backend::from_yaml_str(
            r#"
atoms:
  - {name: A, mass: 1.0, charge: 0.0}
  - {name: B, mass: 1.0, charge: 0.0}
molecules:
  - name: MOL
    atoms: [A]
  - name: NEIGHBOUR
    atoms: [B]
system:
  cell: !Cuboid [30.0, 30.0, 30.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy:
    nonbonded:
      default:
        - !LennardJones {sigma: 3.0, eps: 2.5}
  blocks:
    - molecule: MOL
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0]]
    - molecule: NEIGHBOUR
      N: 1
      insert: !Manual [[0.0, 0.0, 4.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#,
            None,
            &mut rand::thread_rng(),
        )
        .unwrap()
    }

    fn vt_z(displacement: f64) -> VirtualTranslate {
        VirtualTranslateBuilder::default()
            .selection(Selection::parse("molecule MOL").unwrap())
            .displacement(displacement)
            .frequency(Frequency::Every(1))
            .build(RT_300)
            .unwrap()
    }

    /// The measured ΔU must be the ΔU the system actually undergoes.
    ///
    /// The reference is a fresh whole-system energy either side of the same displacement — the
    /// one evaluation no cache can serve — so the two agree only if the perturbed energy describes
    /// the displaced coordinates.
    #[test]
    fn perturbation_energy_matches_the_true_energy_change() {
        let context = lj_pair_backend();
        let vt = vt_z(0.5);
        let group = context.resolve_groups(&Selection::parse("molecule MOL").unwrap())[0];

        let mut reference = context.clone();
        let total = |ctx: &Backend| ctx.hamiltonian().energy(ctx, &Change::Everything);
        let before = total(&reference);
        reference
            .translate_group(group, &(0.5 * Point::new(0.0, 0.0, 1.0)))
            .unwrap();
        let expected = (total(&reference) - before) / RT_300;

        let mut trial = context.clone();
        let measured = vt.perturb(&mut trial, group).unwrap();

        assert!(
            expected.abs() > 1e-3,
            "the test system must have a real ΔU to measure, got {expected}"
        );
        assert_approx_eq!(f64, measured, expected, epsilon = 1e-9);
    }

    /// A molecule 4 Å from a Lennard-Jones neighbour feels a force. Reporting exactly zero is the
    /// signature of an energy read back from a cache that never saw the displacement.
    #[test]
    fn mean_force_is_nonzero_next_to_a_neighbour() {
        let context = lj_pair_backend();
        let mut vt = vt_z(0.1);

        vt.sample(&context, 1).unwrap();

        let force = vt.mean_force().mean;
        assert!(
            force.abs() > 1e-3,
            "expected a finite force towards the neighbour, got {force}"
        );
    }
}

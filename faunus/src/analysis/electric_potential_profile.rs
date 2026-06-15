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

//! Laterally-averaged electric potential φ(z) along the z-axis of a slab.
//!
//! For an implicit-salt (Debye–Hückel / Yukawa) electrolyte the potential of a uniformly
//! charged plane decays exponentially, so the laterally-averaged potential is a screened
//! convolution of the charge density along z (see [`crate::energy::slab_potential`]). This
//! analysis accumulates the per-slab charge density from sampled configurations and writes
//! the resulting profile — charge density, potential, and electric field — to a column file.
//!
//! The walls (if any) are taken to be neutral: only the explicit ions contribute. The
//! method follows the spirit of Greberg, Åkesson & co-workers,
//! <https://doi.org/10/dhb9mj>, specialised to a screened electrolyte (no infinite-slab
//! correction is needed because screening makes the lateral integral convergent).

use super::{Analyze, Frequency};
use crate::auxiliary::{BlockAverage, BlockSummary, ColumnWriter, MappingExt};
use crate::energy::slab_potential::{SlabGrid, SlabKernel};
use crate::selection::Selection;
use crate::Context;
use anyhow::Result;
use derive_more::Debug;
use interatomic::coulomb::{DebyeLength, Medium, Temperature};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

fn all_atoms() -> Selection {
    Selection::parse("all").expect("'all' is a valid selection")
}

fn default_resolution() -> f64 {
    0.5
}

fn default_file() -> PathBuf {
    "potential.csv".into()
}

/// YAML builder for [`ElectricPotentialProfile`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectricPotentialProfileBuilder {
    /// Atoms whose charge contributes to the profile. Defaults to all atoms.
    #[serde(default = "all_atoms")]
    selection: Selection,
    /// Bin width Δz along z (Å). Defaults to 0.5 Å.
    #[serde(default = "default_resolution")]
    resolution: f64,
    /// Restore the finite-box (Greberg) correction: report φ_box = φ_∞ − φ_ext, the potential
    /// of the finite minimum-image cross-section instead of an infinite plane. Enable when the
    /// box is not much larger than the Debye length and the simulation has no Åkesson external
    /// term. Defaults to `false` (infinite-plane kernel).
    #[serde(default)]
    finite_box_correction: bool,
    /// Output column file (use a `.csv` extension for comma-separated values).
    #[serde(default = "default_file")]
    file: PathBuf,
    /// Sampling frequency.
    frequency: Frequency,
}

impl ElectricPotentialProfileBuilder {
    pub fn apply_output_dir(&mut self, dir: &Path) -> Result<()> {
        crate::analysis::prefix_in_place(&mut self.file, dir)
    }

    /// Build the analysis.
    ///
    /// Needs `context` for the cell geometry and `medium` for the Bjerrum length and the
    /// Debye screening length; the screened kernel is undefined without finite ionic strength.
    pub fn build(
        &self,
        context: &impl Context,
        medium: Option<&Medium>,
    ) -> Result<ElectricPotentialProfile> {
        if self.resolution <= 0.0 {
            anyhow::bail!("resolution must be positive");
        }
        let medium =
            medium.ok_or_else(|| anyhow::anyhow!("a medium is required for the Bjerrum length"))?;
        let bjerrum_length = medium.bjerrum_length();
        // Salt present ⇒ screened (Yukawa) kernel; salt-free ⇒ bare-Coulomb (Greberg/Åkesson).
        let debye_length = medium.debye_length();
        let kernel = match debye_length {
            Some(length) => SlabKernel::screened(bjerrum_length, 1.0 / length),
            None => SlabKernel::unscreened(bjerrum_length),
        };

        let mut grid = SlabGrid::from_cell(context.cell(), self.resolution, kernel)?;
        if self.finite_box_correction {
            // The correction handles a thin box exactly, so the infinite-plane warning is moot.
            grid = grid.with_finite_box_correction()?;
        } else if let Some(length) = debye_length {
            // The infinite-plane caveat only applies to the screened kernel; the unscreened one
            // is the exact 1-D Poisson Green's function.
            if grid.is_laterally_thin(length) {
                log::warn!(
                    "electric potential profile: lateral box spans only {:.1} Debye lengths; the \
                     infinite-plane approximation assumes many more (enable finite_box_correction)",
                    grid.lateral_debye_lengths(length),
                );
            }
        }
        let n_bins = grid.n_bins();

        Ok(ElectricPotentialProfile {
            selection: self.selection.clone(),
            grid,
            millivolt_per_kt: kt_per_charge_in_millivolt(medium.temperature()),
            bjerrum_length,
            debye_length,
            slab_charge_density: new_accumulators(n_bins),
            potential: new_accumulators(n_bins),
            num_samples: 0,
            output_file: self.file.clone(),
            frequency: self.frequency,
        })
    }
}

/// kT/e expressed in millivolts at temperature `kelvin`: `RT/F` converted to mV.
fn kt_per_charge_in_millivolt(kelvin: f64) -> f64 {
    crate::R_IN_KJ_PER_MOL * kelvin * 1.0e6 / physical_constants::FARADAY_CONSTANT
}

fn new_accumulators(n: usize) -> Vec<BlockAverage> {
    (0..n).map(|_| BlockAverage::new()).collect()
}

/// Electric potential profile φ(z) for a screened slab.
#[derive(Debug)]
pub struct ElectricPotentialProfile {
    /// Atoms whose charge contributes to the profile.
    selection: Selection,
    /// z-grid and screened kernel doing the convolution.
    grid: SlabGrid,
    /// Conversion factor from potential in kT/e to millivolts.
    millivolt_per_kt: f64,
    /// Bjerrum length (Å), reported for reference.
    bjerrum_length: f64,
    /// Debye length (Å), reported for reference; `None` for the unscreened (bare-Coulomb) kernel.
    debye_length: Option<f64>,
    /// Per-slab areal charge density σ(z) (e·Å⁻²): mean and error across samples.
    slab_charge_density: Vec<BlockAverage>,
    /// Per-slab potential φ(z) (kT/e): mean and error across samples.
    potential: Vec<BlockAverage>,
    /// Number of samples taken.
    num_samples: usize,
    #[debug(skip)]
    output_file: PathBuf,
    frequency: Frequency,
}

/// Centred finite-difference derivative of `values` at index `i`, one-sided at the ends.
fn derivative(values: &[f64], i: usize, spacing: f64) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    if i == 0 {
        (values[1] - values[0]) / spacing
    } else if i == n - 1 {
        (values[n - 1] - values[n - 2]) / spacing
    } else {
        (values[i + 1] - values[i - 1]) / (2.0 * spacing)
    }
}

impl ElectricPotentialProfile {
    /// Potential at bin `index` in millivolts, as `{mean, error}`.
    fn potential_millivolt(&self, index: usize) -> BlockSummary {
        &self.potential[index] * self.millivolt_per_kt
    }

    /// Mean potential profile in millivolts (drives the field column and the YAML scalars).
    fn mean_potential_millivolt(&self) -> Vec<f64> {
        self.potential
            .iter()
            .map(|p| p.mean() * self.millivolt_per_kt)
            .collect()
    }

    /// Build the YAML results mapping (inherent so it is callable without choosing a
    /// `Context` type; the [`Analyze`] trait method delegates here).
    fn report(&self) -> Option<serde_yml::Value> {
        if self.num_samples == 0 {
            return None;
        }
        let n = self.grid.n_bins();
        let lower = self.potential_millivolt(0);
        let upper = self.potential_millivolt(n - 1);
        let midplane = self.potential_millivolt(n / 2);
        // Drop = wall − midplane; treat the two as independent for the error estimate.
        let drop = |wall: &BlockSummary| BlockSummary {
            mean: wall.mean - midplane.mean,
            error: wall.error.hypot(midplane.error),
        };

        let mut map = serde_yml::Mapping::new();
        map.try_insert("num_samples", self.num_samples)?;
        map.try_insert("bjerrum_length/Å", self.bjerrum_length)?;
        if let Some(debye_length) = self.debye_length {
            map.try_insert("debye_length/Å", debye_length)?;
        }
        map.try_insert("num_bins", n)?;
        map.try_insert("potential_lower_wall/mV", lower)?;
        map.try_insert("potential_upper_wall/mV", upper)?;
        map.try_insert("potential_midplane/mV", midplane)?;
        map.try_insert("potential_drop_lower/mV", drop(&lower))?;
        map.try_insert("potential_drop_upper/mV", drop(&upper))?;
        Some(serde_yml::Value::Mapping(map))
    }

    fn write_profile(&self) -> Result<()> {
        let mut writer = ColumnWriter::open(
            &self.output_file,
            &[
                "z/Å",
                "slab_charge_density/e·Å⁻²",
                "charge_density/e·Å⁻³",
                "potential/mV",
                "potential_error/mV",
                "field/mV·Å⁻¹",
            ],
        )?;
        let potential_mv = self.mean_potential_millivolt();
        let bin_width = self.grid.bin_width();
        for i in 0..self.grid.n_bins() {
            let z = self.grid.bin_center(i);
            let sigma = self.slab_charge_density[i].mean();
            let summary = self.potential_millivolt(i);
            // E = −dφ/dz.
            let field = -derivative(&potential_mv, i, bin_width);
            writer.write_row(&[
                &format_args!("{z:.4}"),
                &format_args!("{sigma:.6e}"),
                &format_args!("{:.6e}", sigma / bin_width),
                &format_args!("{:.6}", summary.mean),
                &format_args!("{:.6}", summary.error),
                &format_args!("{field:.6}"),
            ])?;
        }
        writer.flush()?;
        Ok(())
    }
}

impl crate::Info for ElectricPotentialProfile {
    fn short_name(&self) -> Option<&'static str> {
        Some("electricpotentialprofile")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Electric potential profile along z (screened slab)")
    }
    fn citation(&self) -> Option<&'static str> {
        Some("doi:10/dhb9mj")
    }
}

impl<T: Context> Analyze<T> for ElectricPotentialProfile {
    fn frequency(&self) -> Frequency {
        self.frequency
    }
    fn set_frequency(&mut self, freq: Frequency) {
        self.frequency = freq;
    }

    fn perform_sample(&mut self, context: &T, _step: usize, _weight: f64) -> Result<()> {
        // Instantaneous total charge per slab. Only currently-active atoms are resolved, so
        // a fluctuating particle number (GCMC) is handled automatically.
        let mut slab_charge = vec![0.0; self.grid.n_bins()];
        for index in context.resolve_atoms_live(&self.selection) {
            let bin = self.grid.bin_index(context.position(index).z);
            slab_charge[bin] += context.atom_charge(index);
        }

        let area = self.grid.area();
        for (accumulator, &charge) in self.slab_charge_density.iter_mut().zip(&slab_charge) {
            accumulator.add(charge / area);
        }

        // Accumulate this configuration's full φ(z); computing it per sample lets the error
        // capture cross-slab charge correlations a per-bin variance would miss.
        for (accumulator, potential) in self
            .potential
            .iter_mut()
            .zip(self.grid.potential_profile(&slab_charge))
        {
            accumulator.add(potential);
        }

        self.num_samples += 1;
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }

    fn write_to_disk(&mut self) -> Result<()> {
        if self.num_samples == 0 {
            return Ok(());
        }
        self.write_profile()
    }

    fn to_yaml(&self) -> Option<serde_yml::Value> {
        self.report()
    }
}

impl<T: Context> From<ElectricPotentialProfile> for Box<dyn Analyze<T>> {
    fn from(analysis: ElectricPotentialProfile) -> Self {
        Box::new(analysis)
    }
}

impl std::fmt::Display for ElectricPotentialProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Electric Potential Profile:")?;
        writeln!(f, "  Samples:  {}", self.num_samples)?;
        writeln!(f, "  Bins:     {}", self.grid.n_bins())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisBuilder;

    /// Build an analysis directly (10 bins over [−5, 5]) for testing the reporting helpers.
    fn dummy(output_file: &str) -> ElectricPotentialProfile {
        let kernel = SlabKernel::screened(7.0, 0.1);
        // 10 Å cube ⇒ half-length 5, 10 bins at Δz = 1.
        let grid = SlabGrid::from_cell(
            &crate::cell::Cell::Cuboid(crate::cell::Cuboid::new(10.0, 10.0, 10.0)),
            1.0,
            kernel,
        )
        .unwrap();
        let n = grid.n_bins();
        assert_eq!(n, 10);
        ElectricPotentialProfile {
            selection: all_atoms(),
            grid,
            millivolt_per_kt: 25.7,
            bjerrum_length: 7.0,
            debye_length: Some(10.0),
            slab_charge_density: new_accumulators(n),
            potential: new_accumulators(n),
            num_samples: 0,
            output_file: output_file.into(),
            frequency: Frequency::Every(1),
        }
    }

    #[test]
    fn millivolt_conversion_is_about_25_7_at_room_temperature() {
        let mv = kt_per_charge_in_millivolt(298.15);
        assert!((mv - 25.7).abs() < 0.1, "kT/e = {mv} mV");
    }

    #[test]
    fn derivative_of_linear_ramp_is_constant_slope() {
        let values = [0.0, 2.0, 4.0, 6.0];
        let spacing = 1.0;
        // Interior central difference and one-sided ends all recover slope 2.
        for i in 0..values.len() {
            assert!((derivative(&values, i, spacing) - 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn yaml_is_none_without_samples() {
        let analysis = dummy("/tmp/faunus_epp_none.csv");
        assert!(analysis.report().is_none());
    }

    #[test]
    fn error_is_zero_for_identical_samples_and_positive_otherwise() {
        // Two identical configurations ⇒ zero variance ⇒ zero SEM.
        let mut same = dummy("/tmp/faunus_epp_same.csv");
        for accumulator in &mut same.potential {
            accumulator.add(1.0);
            accumulator.add(1.0);
        }
        same.num_samples = 2;
        let yaml = same.report().unwrap();
        let error = yaml["potential_midplane/mV"]["error"].as_f64().unwrap();
        assert_eq!(error, 0.0);

        // Two distinct configurations ⇒ positive SEM.
        let mut differ = dummy("/tmp/faunus_epp_differ.csv");
        for accumulator in &mut differ.potential {
            accumulator.add(1.0);
            accumulator.add(3.0);
        }
        differ.num_samples = 2;
        let yaml = differ.report().unwrap();
        let error = yaml["potential_midplane/mV"]["error"].as_f64().unwrap();
        assert!(error > 0.0, "error = {error}");
    }

    #[test]
    fn yaml_reports_walls_midplane_and_drops() {
        let mut analysis = dummy("/tmp/faunus_epp_walls.csv");
        for accumulator in &mut analysis.potential {
            accumulator.add(2.0);
        }
        analysis.num_samples = 1;
        let yaml = analysis.report().unwrap();
        for key in [
            "potential_lower_wall/mV",
            "potential_upper_wall/mV",
            "potential_midplane/mV",
            "potential_drop_lower/mV",
            "potential_drop_upper/mV",
        ] {
            assert!(yaml.get(key).is_some(), "missing {key}");
            assert!(yaml[key].get("mean").is_some());
            assert!(yaml[key].get("error").is_some());
        }
        assert!(yaml.get("bjerrum_length/Å").is_some());
        assert!(yaml.get("debye_length/Å").is_some());
    }

    #[test]
    fn write_profile_emits_a_row_per_bin_with_headers() {
        let path = "/tmp/faunus_epp_profile.csv";
        let mut analysis = dummy(path);
        for accumulator in &mut analysis.slab_charge_density {
            accumulator.add(1e-3);
        }
        for accumulator in &mut analysis.potential {
            accumulator.add(0.5);
        }
        analysis.num_samples = 1;
        analysis.write_profile().unwrap();

        let contents = std::fs::read_to_string(path).unwrap();
        let mut lines = contents.lines();
        let header = lines.next().unwrap();
        assert!(header.contains("z/Å"));
        assert!(header.contains("slab_charge_density/e·Å⁻²"));
        assert!(header.contains("potential_error/mV"));
        assert!(header.contains("field/mV·Å⁻¹"));
        assert_eq!(lines.count(), analysis.grid.n_bins());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn deserialize_applies_defaults_when_only_frequency_given() {
        // selection, resolution and file all default.
        let yaml = r#"
- !ElectricPotentialProfile
  frequency: !Every 10
"#;
        let builders: Vec<AnalysisBuilder> = serde_yml::from_str(yaml).unwrap();
        match &builders[0] {
            AnalysisBuilder::ElectricPotentialProfile(b) => {
                assert_eq!(b.selection.source(), "all");
                assert_eq!(b.resolution, 0.5);
                assert_eq!(b.file, PathBuf::from("potential.csv"));
                assert!(!b.finite_box_correction);
            }
            _ => panic!("expected ElectricPotentialProfile variant"),
        }
    }

    #[test]
    fn deserialize_finite_box_correction_flag() {
        let yaml = r#"
- !ElectricPotentialProfile
  frequency: !Every 10
  finite_box_correction: true
"#;
        let builders: Vec<AnalysisBuilder> = serde_yml::from_str(yaml).unwrap();
        match &builders[0] {
            AnalysisBuilder::ElectricPotentialProfile(b) => assert!(b.finite_box_correction),
            _ => panic!("expected ElectricPotentialProfile variant"),
        }
    }

    #[test]
    fn info_trait() {
        use crate::Info;
        let analysis = dummy("/tmp/faunus_epp_info.csv");
        assert_eq!(analysis.short_name(), Some("electricpotentialprofile"));
        assert!(analysis.citation().unwrap().starts_with("doi:"));
    }
}

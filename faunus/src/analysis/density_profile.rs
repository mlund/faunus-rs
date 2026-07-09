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

//! Average density of a selected species along the z-axis.
//!
//! Each selected atom — or, with `use_com`, the mass centre of each selected molecule — counts
//! towards the slab holding it. Dividing the mean count by the slab volume gives ρ(z), reported
//! as a number density, a molar concentration, and a mass density.
//!
//! Each configuration contributes its own count, which is then averaged rather than accumulated.
//! An insertion or deletion therefore only changes the count of the configuration it occurs in,
//! leaving the mean equal to the grand-canonical ⟨N(z)⟩ when the particle number fluctuates.

use super::{Analyze, Frequency};
use crate::auxiliary::{BlockAverage, BlockSummary, ColumnWriter, MappingExt};
use crate::cell::Shape;
use crate::group::GroupSize;
use crate::selection::{CachedSelection, Selection};
use crate::z_grid::ZGrid;
use crate::Context;
use anyhow::Result;
use derive_more::Debug;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// One (g/mol)·Å⁻³ expressed in g·mL⁻¹ (≈ 1.66054).
const GRAM_PER_MOL_PER_CUBIC_ANGSTROM_TO_GRAM_PER_MILLILITER: f64 =
    1.0e24 / physical_constants::AVOGADRO_CONSTANT;

fn default_resolution() -> f64 {
    1.0
}

/// YAML builder for [`DensityProfile`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DensityProfileBuilder {
    /// Atoms, or molecules when `use_com` is set, whose density is profiled.
    selection: Selection,
    /// Output column file (use a `.csv` extension for comma-separated values).
    file: PathBuf,
    /// Bin the mass centre of each selected molecule instead of each selected atom.
    #[serde(default)]
    use_com: bool,
    /// Slab thickness Δz along z (Å). Defaults to 1.0 Å.
    #[serde(default = "default_resolution")]
    resolution: f64,
    /// Sampling frequency.
    frequency: Frequency,
}

impl DensityProfileBuilder {
    pub fn apply_output_dir(&mut self, dir: &Path) -> Result<()> {
        crate::analysis::prefix_in_place(&mut self.file, dir)
    }

    /// Build the analysis, taking the slab layout from the cell of `context`.
    pub fn build(&self, context: &impl Context) -> Result<DensityProfile> {
        let grid = ZGrid::from_cell(context.cell(), self.resolution)?;
        if self.use_com {
            self.reject_selections_without_mass_center(context)?;
        }
        let n_bins = grid.n_bins();
        Ok(DensityProfile {
            selection: CachedSelection::for_com(self.selection.clone(), self.use_com),
            grid,
            use_com: self.use_com,
            counts: new_accumulators(n_bins),
            masses: new_accumulators(n_bins),
            total_count: BlockAverage::new(),
            warned_volume_change: false,
            num_samples: 0,
            output_file: self.file.clone(),
            frequency: self.frequency,
        })
    }

    /// Reject a `use_com` selection that can match a group without a mass centre.
    ///
    /// A selection matches only the atoms that are present, so a species that starts out empty —
    /// a grand-canonical reservoir, say — would slip past a check made against the initial
    /// state and abort the run once its first particle appears. Testing against a fully
    /// populated copy of the groups catches it now instead.
    fn reject_selections_without_mass_center(&self, context: &impl Context) -> Result<()> {
        let mut groups = context.groups().to_vec();
        for group in &mut groups {
            group.resize(GroupSize::Full)?;
        }
        let topology = context.topology();
        let matches_atomic_group = self
            .selection
            .resolve_groups(&topology, &groups)
            .into_iter()
            .any(|index| !topology.moleculekinds()[groups[index].molecule()].has_com());
        if matches_atomic_group {
            anyhow::bail!(
                "DensityProfile: selection '{}' matches a group without a center of mass \
                 (atomic groups are not supported with use_com)",
                self.selection.source()
            );
        }
        Ok(())
    }
}

fn new_accumulators(n: usize) -> Vec<BlockAverage> {
    (0..n).map(|_| BlockAverage::new()).collect()
}

/// Density profile ρ(z) of a selected species.
#[derive(Debug)]
pub struct DensityProfile {
    /// Atoms, or molecules when `use_com` is set, whose density is profiled.
    selection: CachedSelection,
    /// Slab layout along z.
    grid: ZGrid,
    /// Bin molecular mass centres rather than individual atoms.
    use_com: bool,
    /// Particles per slab: mean and error across samples.
    counts: Vec<BlockAverage>,
    /// Mass per slab (g/mol): mean and error across samples.
    masses: Vec<BlockAverage>,
    /// Total number of selected particles, reported as a consistency check.
    total_count: BlockAverage,
    warned_volume_change: bool,
    num_samples: usize,
    #[debug(skip)]
    output_file: PathBuf,
    frequency: Frequency,
}

impl DensityProfile {
    /// Instantaneous count and mass per slab for the current configuration.
    fn tally(&mut self, context: &impl Context) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut counts = vec![0.0; self.grid.n_bins()];
        let mut masses = vec![0.0; self.grid.n_bins()];
        let grid = &self.grid;
        let mut add = |z: f64, mass: f64| {
            let bin = grid.bin_index(z);
            counts[bin] += 1.0;
            masses[bin] += mass;
        };
        let indices = self.selection.resolve(context).to_vec();
        if self.use_com {
            for group_index in indices {
                let group = &context.groups()[group_index];
                let mass_center = group.mass_center().ok_or_else(|| {
                    anyhow::anyhow!("DensityProfile: group {group_index} has no center of mass")
                })?;
                let mass = group.iter_active().map(|i| context.atom_mass(i)).sum();
                add(mass_center.z, mass);
            }
        } else {
            for index in indices {
                add(context.position(index).z, context.atom_mass(index));
            }
        }
        Ok((counts, masses))
    }

    /// Number density (Å⁻³) of each slab, mean and error.
    fn number_density(&self, bin: usize) -> BlockSummary {
        &self.counts[bin] * self.grid.bin_volume().recip()
    }

    /// Molar concentration (mol·L⁻¹) of each slab, mean and error.
    fn molarity(&self, bin: usize) -> BlockSummary {
        &self.counts[bin] * (self.grid.bin_volume() * crate::MOLAR_TO_INV_ANGSTROM3).recip()
    }

    /// Mass density (g·mL⁻¹) of each slab, mean and error.
    fn mass_density(&self, bin: usize) -> BlockSummary {
        &self.masses[bin]
            * (GRAM_PER_MOL_PER_CUBIC_ANGSTROM_TO_GRAM_PER_MILLILITER / self.grid.bin_volume())
    }

    /// Build the YAML results mapping (inherent so it is callable without choosing a
    /// `Context` type; the [`Analyze`] trait method delegates here).
    fn report(&self) -> Option<serde_yml::Value> {
        if self.num_samples == 0 {
            return None;
        }
        let mut map = serde_yml::Mapping::new();
        map.try_insert("num_samples", self.num_samples)?;
        map.try_insert("num_bins", self.grid.n_bins())?;
        map.try_insert("bin_width/Å", self.grid.bin_width())?;
        map.try_insert("area/Å²", self.grid.area())?;
        map.try_insert("mean_count", self.total_count.summary())?;
        Some(serde_yml::Value::Mapping(map))
    }

    fn write_profile(&self) -> Result<()> {
        let mut writer = ColumnWriter::open(
            &self.output_file,
            &[
                "z/Å",
                "density/Å⁻³",
                "density_error/Å⁻³",
                "molarity/mol·L⁻¹",
                "molarity_error/mol·L⁻¹",
                "mass_density/g·mL⁻¹",
                "mass_density_error/g·mL⁻¹",
            ],
        )?;
        for bin in 0..self.grid.n_bins() {
            let density = self.number_density(bin);
            let molarity = self.molarity(bin);
            let mass_density = self.mass_density(bin);
            writer.write_row(&[
                &format_args!("{:.4}", self.grid.bin_center(bin)),
                &format_args!("{:.6e}", density.mean),
                &format_args!("{:.6e}", density.error),
                &format_args!("{:.6e}", molarity.mean),
                &format_args!("{:.6e}", molarity.error),
                &format_args!("{:.6e}", mass_density.mean),
                &format_args!("{:.6e}", mass_density.error),
            ])?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Warn once if the cell has changed size, which the fixed slab layout cannot follow.
    fn check_volume(&mut self, context: &impl Context) {
        if self.warned_volume_change {
            return;
        }
        let initial = self.grid.volume();
        let Some(current) = context.cell().volume() else {
            return;
        };
        if (current - initial).abs() > 1e-6 * initial {
            log::warn!(
                "density profile: the cell volume changed from {initial:.1} to {current:.1} Å³; \
                 the slabs keep their initial size, so the profile is only meaningful at \
                 constant volume"
            );
            self.warned_volume_change = true;
        }
    }
}

impl crate::Info for DensityProfile {
    fn short_name(&self) -> Option<&'static str> {
        Some("densityprofile")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Density profile along z")
    }
}

impl<T: Context> Analyze<T> for DensityProfile {
    fn frequency(&self) -> Frequency {
        self.frequency
    }
    fn set_frequency(&mut self, freq: Frequency) {
        self.frequency = freq;
    }

    fn perform_sample(&mut self, context: &T, _step: usize, _weight: f64) -> Result<()> {
        self.check_volume(context);
        let (counts, masses) = self.tally(context)?;
        self.total_count.add(counts.iter().sum::<f64>());
        for (accumulator, &count) in self.counts.iter_mut().zip(&counts) {
            accumulator.add(count);
        }
        for (accumulator, &mass) in self.masses.iter_mut().zip(&masses) {
            accumulator.add(mass);
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

impl<T: Context> From<DensityProfile> for Box<dyn Analyze<T>> {
    fn from(analysis: DensityProfile) -> Self {
        Box::new(analysis)
    }
}

impl std::fmt::Display for DensityProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Density Profile:")?;
        writeln!(f, "  Selection: {}", self.selection.selection())?;
        writeln!(f, "  Samples:   {}", self.num_samples)?;
        writeln!(f, "  Bins:      {}", self.grid.n_bins())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisBuilder;
    use crate::backend::Backend;
    use approx::assert_relative_eq;
    use tempfile::NamedTempFile;

    /// One DIMER in a 10 Å cube: a heavy atom A (mass 3) at z = −2 and a light atom B (mass 1)
    /// at z = +2, so the mass centre sits at z = −1. With Δz = 1 Å the ten bins span [−5, 5],
    /// putting A in bin 3, B in bin 7, and the mass centre in bin 4.
    const DIMER: &str = r#"
atoms:
  - {name: A, mass: 3.0, charge: 0.0, sigma: 1.0}
  - {name: B, mass: 1.0, charge: 0.0, sigma: 1.0}
molecules:
  - name: DIMER
    atoms: [A, B]
system:
  cell: !Cuboid [10.0, 10.0, 10.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: DIMER
      N: 1
      insert: !Manual [[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;

    /// Two independent atoms in an atomic mega-group, which carries no mass centre.
    const ATOMIC_GAS: &str = r#"
atoms:
  - {name: A, mass: 2.0, charge: 0.0, sigma: 1.0}
molecules:
  - name: GAS
    atomic: true
    atoms: [A]
system:
  cell: !Cuboid [10.0, 10.0, 10.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: GAS
      N: 2
      insert: !Manual [[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;

    /// Three atoms inserted, one of them deactivated — the state a grand-canonical run reaches
    /// after a deletion. The inactive atom sits at z = +4, alone in bin 9.
    const PARTLY_ACTIVE_GAS: &str = r#"
atoms:
  - {name: A, mass: 2.0, charge: 0.0, sigma: 1.0}
molecules:
  - name: GAS
    atomic: true
    atoms: [A]
system:
  cell: !Cuboid [10.0, 10.0, 10.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: GAS
      N: 3
      active: 2
      insert: !Manual [[0.0, 0.0, -2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;

    fn backend_from_str(yaml: &str) -> Backend {
        let tmp = NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml).unwrap();
        Backend::new(tmp.path(), None, &mut rand::thread_rng()).unwrap()
    }

    fn builder(selection: &str, use_com: bool) -> DensityProfileBuilder {
        DensityProfileBuilder {
            selection: Selection::parse(selection).unwrap(),
            file: "density.csv".into(),
            use_com,
            resolution: 1.0,
            frequency: Frequency::Every(1),
        }
    }

    /// Bins holding at least one particle, as `(bin, mean_count)`.
    fn occupied(analysis: &DensityProfile) -> Vec<(usize, f64)> {
        analysis
            .counts
            .iter()
            .enumerate()
            .filter(|(_, accumulator)| accumulator.mean() > 0.0)
            .map(|(bin, accumulator)| (bin, accumulator.mean()))
            .collect()
    }

    #[test]
    fn atoms_are_binned_individually() {
        let context = backend_from_str(DIMER);
        let mut analysis = builder("all", false).build(&context).unwrap();
        analysis.sample(&context, 0).unwrap();
        assert_eq!(occupied(&analysis), vec![(3, 1.0), (7, 1.0)]);
        // Bin 3 holds the heavy atom, bin 7 the light one.
        assert_relative_eq!(analysis.masses[3].mean(), 3.0);
        assert_relative_eq!(analysis.masses[7].mean(), 1.0);
    }

    #[test]
    fn mass_centers_are_binned_when_use_com_is_set() {
        let context = backend_from_str(DIMER);
        let mut analysis = builder("molecule DIMER", true).build(&context).unwrap();
        analysis.sample(&context, 0).unwrap();
        // A single count at the mass-weighted centre z = −1, carrying the whole molecular mass.
        assert_eq!(occupied(&analysis), vec![(4, 1.0)]);
        assert_relative_eq!(analysis.masses[4].mean(), 4.0);
    }

    #[test]
    fn selecting_one_atom_kind_ignores_the_other() {
        let context = backend_from_str(DIMER);
        let mut analysis = builder("atomtype B", false).build(&context).unwrap();
        analysis.sample(&context, 0).unwrap();
        assert_eq!(occupied(&analysis), vec![(7, 1.0)]);
    }

    #[test]
    fn densities_integrate_back_to_the_particle_count() {
        let context = backend_from_str(DIMER);
        let mut analysis = builder("all", false).build(&context).unwrap();
        analysis.sample(&context, 0).unwrap();
        let bin_volume = analysis.grid.bin_volume();
        let particles: f64 = (0..analysis.grid.n_bins())
            .map(|bin| analysis.number_density(bin).mean * bin_volume)
            .sum();
        assert_relative_eq!(particles, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn concentration_and_mass_density_follow_from_the_number_density() {
        let context = backend_from_str(DIMER);
        let mut analysis = builder("all", false).build(&context).unwrap();
        analysis.sample(&context, 0).unwrap();
        // Bin 3: one atom of mass 3 in a 100 Å³ slab.
        let bin_volume = analysis.grid.bin_volume();
        assert_relative_eq!(bin_volume, 100.0);
        assert_relative_eq!(analysis.number_density(3).mean, 0.01);
        // 1 Å⁻³ ≈ 1660.54 M, so 0.01 Å⁻³ ≈ 16.6 M.
        assert_relative_eq!(analysis.molarity(3).mean, 16.605, epsilon = 1e-3);
        // 3 g/mol in 100 Å³ ≈ 0.0498 g/mL.
        assert_relative_eq!(analysis.mass_density(3).mean, 0.049816, epsilon = 1e-5);
    }

    #[test]
    fn error_is_zero_for_identical_samples_and_positive_otherwise() {
        let context = backend_from_str(DIMER);
        let mut same = builder("all", false).build(&context).unwrap();
        same.sample(&context, 0).unwrap();
        same.sample(&context, 1).unwrap();
        assert_eq!(same.number_density(3).error, 0.0);

        // Emptying bin 3 in a second sample makes its mean fluctuate.
        let mut differ = builder("all", false).build(&context).unwrap();
        differ.sample(&context, 0).unwrap();
        differ.counts[3].add(0.0);
        assert!(differ.number_density(3).error > 0.0);
    }

    /// Under grand-canonical sampling the profile must count only the particles that are
    /// present, so that its mean is ⟨N(z)⟩ rather than the number of allocated slots.
    #[test]
    fn inactive_particles_are_not_counted() {
        let context = backend_from_str(PARTLY_ACTIVE_GAS);
        let mut analysis = builder("all", false).build(&context).unwrap();
        analysis.sample(&context, 0).unwrap();
        assert_eq!(occupied(&analysis), vec![(3, 1.0), (7, 1.0)]);
        assert_relative_eq!(analysis.total_count.mean(), 2.0);
        // Bin 9 holds the deactivated atom and must stay empty.
        assert_relative_eq!(analysis.number_density(9).mean, 0.0);
    }

    /// A speciation or titration move swaps an atom's kind in place, leaving every group's
    /// active count — and hence the group-list generation — untouched. A profile selecting on
    /// atom type must still follow the swap.
    #[test]
    fn an_atom_kind_swap_is_picked_up() {
        use crate::group::GroupCollection;
        let mut context = backend_from_str(DIMER);
        let mut analysis = builder("atomtype A", false).build(&context).unwrap();
        analysis.sample(&context, 0).unwrap();
        assert_eq!(occupied(&analysis), vec![(3, 1.0)]);

        // Turn the one A atom (bin 3) into a B atom; nothing of kind A is left.
        context.set_atom_kind(0, 1);
        analysis.sample(&context, 1).unwrap();
        assert_relative_eq!(analysis.counts[3].mean(), 0.5);
        assert_relative_eq!(analysis.total_count.mean(), 0.5);
    }

    /// A titration swap can change an atom's mass, which moves the molecule's mass centre. The
    /// centre cached on the group is not refreshed by such a swap, so it must be recomputed.
    #[test]
    fn a_mass_changing_atom_kind_swap_moves_the_mass_center() {
        use crate::group::GroupCollection;
        let mut context = backend_from_str(DIMER);
        let mut analysis = builder("molecule DIMER", true).build(&context).unwrap();
        analysis.sample(&context, 0).unwrap();
        // A (mass 3) at z=−2, B (mass 1) at z=+2 ⇒ centre at z=−1, bin 4.
        assert_eq!(occupied(&analysis), vec![(4, 1.0)]);

        // Turn B into a second A (mass 1 → 3); the centre moves to z = 0, bin 5.
        context.set_atom_kind(1, 0);
        analysis.sample(&context, 1).unwrap();
        assert_relative_eq!(analysis.counts[4].mean(), 0.5);
        assert_relative_eq!(analysis.counts[5].mean(), 0.5);
        // The mass follows the new kinds too: 3 + 3 = 6.
        assert_relative_eq!(analysis.masses[5].mean(), 3.0); // 6.0 in one of two samples
    }

    #[test]
    fn use_com_rejects_a_group_without_a_mass_center() {
        let context = backend_from_str(ATOMIC_GAS);
        let error = builder("molecule GAS", true).build(&context).unwrap_err();
        assert!(error.to_string().contains("center of mass"), "{error}");
        // The same selection is fine atom by atom.
        assert!(builder("molecule GAS", false).build(&context).is_ok());
    }

    /// A species that is empty at startup matches nothing, so the rejection above must be
    /// decided from the topology rather than from the initial configuration — otherwise a
    /// grand-canonical insertion would abort the run much later.
    #[test]
    fn use_com_rejects_an_atomic_species_that_starts_out_empty() {
        use crate::context::ParticleSystem;
        let yaml = ATOMIC_GAS.replace("N: 2", "N: 2\n      active: 0");
        let context = backend_from_str(&yaml);
        let selection = Selection::parse("molecule GAS").unwrap();
        assert!(context.resolve_groups_live(&selection).is_empty());
        let error = builder("molecule GAS", true).build(&context).unwrap_err();
        assert!(error.to_string().contains("center of mass"), "{error}");
    }

    #[test]
    fn non_prismatic_cell_is_rejected() {
        let yaml = DIMER.replace("!Cuboid [10.0, 10.0, 10.0]", "!Sphere {radius: 10.0}");
        let context = backend_from_str(&yaml);
        assert!(builder("all", false).build(&context).is_err());
    }

    #[test]
    fn yaml_is_none_without_samples() {
        let context = backend_from_str(DIMER);
        let analysis = builder("all", false).build(&context).unwrap();
        assert!(analysis.report().is_none());
    }

    #[test]
    fn yaml_reports_the_grid_and_the_mean_count() {
        let context = backend_from_str(DIMER);
        let mut analysis = builder("all", false).build(&context).unwrap();
        analysis.sample(&context, 0).unwrap();
        let yaml = analysis.report().unwrap();
        assert_eq!(yaml["num_samples"].as_u64(), Some(1));
        assert_eq!(yaml["num_bins"].as_u64(), Some(10));
        assert_relative_eq!(yaml["bin_width/Å"].as_f64().unwrap(), 1.0);
        assert_relative_eq!(yaml["area/Å²"].as_f64().unwrap(), 100.0);
        assert_relative_eq!(yaml["mean_count"]["mean"].as_f64().unwrap(), 2.0);
    }

    #[test]
    fn write_profile_emits_a_row_per_bin_with_headers() {
        let path = std::env::temp_dir().join("faunus_density_profile.csv");
        let context = backend_from_str(DIMER);
        let mut analysis = builder("all", false).build(&context).unwrap();
        analysis.output_file = path.clone();
        analysis.sample(&context, 0).unwrap();
        Analyze::<Backend>::write_to_disk(&mut analysis).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let mut lines = contents.lines();
        let header = lines.next().unwrap();
        assert!(header.contains("z/Å"));
        assert!(header.contains("density/Å⁻³"));
        assert!(header.contains("molarity/mol·L⁻¹"));
        assert!(header.contains("mass_density_error/g·mL⁻¹"));
        assert_eq!(lines.count(), analysis.grid.n_bins());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn deserialize_applies_defaults() {
        let yaml = r#"
selection: "atomtype Na"
file: sodium.csv
frequency: !Every 10
"#;
        let builder: DensityProfileBuilder = serde_yml::from_str(yaml).unwrap();
        assert_eq!(builder.selection.source(), "atomtype Na");
        assert_relative_eq!(builder.resolution, 1.0);
        assert!(!builder.use_com);
    }

    #[test]
    fn file_is_required_and_unknown_fields_are_rejected() {
        let no_file = "selection: all\nfrequency: !Every 10\n";
        assert!(serde_yml::from_str::<DensityProfileBuilder>(no_file).is_err());
        let unknown = "selection: all\nfile: d.csv\nfrequency: !Every 10\noops: 1\n";
        assert!(serde_yml::from_str::<DensityProfileBuilder>(unknown).is_err());
    }

    #[test]
    fn deserialize_via_analysis_builder() {
        let yaml = r#"
- !DensityProfile
  selection: "molecule water"
  file: water.csv
  use_com: true
  resolution: 0.5
  frequency: !Every 10
"#;
        let builders: Vec<AnalysisBuilder> = serde_yml::from_str(yaml).unwrap();
        match &builders[0] {
            AnalysisBuilder::DensityProfile(builder) => {
                assert!(builder.use_com);
                assert_relative_eq!(builder.resolution, 0.5);
            }
            _ => panic!("expected DensityProfile variant"),
        }
    }

    #[test]
    fn info_trait() {
        use crate::Info;
        let context = backend_from_str(DIMER);
        let analysis = builder("all", false).build(&context).unwrap();
        assert_eq!(analysis.short_name(), Some("densityprofile"));
    }
}

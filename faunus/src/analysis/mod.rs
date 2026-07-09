// Copyright 2023 Mikael Lund
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

//! # System analysis and reporting

use crate::{Context, Info};
use anyhow::Result;
use core::fmt::Debug;
use interatomic::coulomb::Temperature;
use serde::{Deserialize, Serialize};
use serde_yml::Value;
use std::path::{Path, PathBuf};

mod collective_variable;
mod density_profile;
mod double_layer_pressure;
mod electric_potential_profile;
mod energy;
mod mean_along_coordinate;
mod multipole;
mod multipole_distribution;
mod radial_distribution;
pub mod reweight;
mod rotational_diffusion;
mod scaled_widom_insertion;
mod shape;
mod spatial_distribution;
mod structure_writer;
mod virtual_translate;
mod virtual_volume_move;
mod widom;
mod widom_rotation;
pub use collective_variable::{CollectiveVariableAnalysis, CollectiveVariableAnalysisBuilder};
pub use density_profile::{DensityProfile, DensityProfileBuilder};
pub use double_layer_pressure::{DoubleLayerPressure, DoubleLayerPressureBuilder};
pub use electric_potential_profile::{ElectricPotentialProfile, ElectricPotentialProfileBuilder};
pub use energy::{EnergyAnalysis, EnergyAnalysisBuilder};
pub use mean_along_coordinate::{MeanAlongCoordinate, MeanAlongCoordinateBuilder};
pub use multipole_distribution::{MultipoleDistribution, MultipoleDistributionBuilder};
pub use radial_distribution::{RadialDistribution, RadialDistributionBuilder};
pub use rotational_diffusion::{RotationalDiffusion, RotationalDiffusionBuilder};
pub use scaled_widom_insertion::{ScaledWidomInsertion, ScaledWidomInsertionBuilder};
pub use shape::{ShapeAnalysis, ShapeAnalysisBuilder};
pub use spatial_distribution::{SpatialDistribution, SpatialDistributionBuilder};
pub use structure_writer::{StructureWriter, StructureWriterBuilder};
pub use virtual_translate::{VirtualTranslate, VirtualTranslateBuilder};
pub use virtual_volume_move::{VirtualVolumeMove, VirtualVolumeMoveBuilder};
pub use widom_rotation::{WidomRotation, WidomRotationBuilder};

/// Frequency of analysis.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Frequency {
    /// Every `n` steps
    Every(usize),
    /// Once at step `n`
    Once(usize),
    /// Once at the very last step
    End,
}

impl Frequency {
    /// Check if action, typically a move or analysis, should be performed at given step.
    ///
    /// Handles `Every(n)` and `Once(n)`. `End` samples once from `finalize` instead; see
    /// [`should_perform_at_end`](Self::should_perform_at_end).
    #[must_use]
    #[allow(clippy::manual_is_multiple_of)] // is_multiple_of is not const
    pub const fn should_perform(&self, step: usize) -> bool {
        match self {
            Self::Every(n) => step % *n == 0,
            Self::Once(n) => step == *n,
            Self::End => false,
        }
    }

    /// Check if action should be performed at the final step.
    ///
    /// Returns `true` only for the `End` variant.
    #[must_use]
    pub const fn should_perform_at_end(&self) -> bool {
        matches!(self, Self::End)
    }
}

/// Sampling state every analysis carries: when to sample, and how often it has.
///
/// Owned by the framework. `num_samples` counts *frames* and is incremented once per successful
/// `perform_sample`, so it cannot drift into meaning something else per analysis.
#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    frequency: Frequency,
    num_samples: usize,
}

impl From<Frequency> for Sampling {
    fn from(frequency: Frequency) -> Self {
        Self::new(frequency)
    }
}

/// On the wire a `Sampling` *is* its frequency; the sample count is runtime state.
impl Serialize for Sampling {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        self.frequency.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Sampling {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        Frequency::deserialize(deserializer).map(Self::new)
    }
}

impl Sampling {
    pub const fn new(frequency: Frequency) -> Self {
        Self {
            frequency,
            num_samples: 0,
        }
    }
    pub const fn frequency(&self) -> Frequency {
        self.frequency
    }
    pub const fn set_frequency(&mut self, frequency: Frequency) {
        self.frequency = frequency;
    }
    /// Frames sampled so far.
    pub const fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Fake a sample count so a unit test can exercise reporting without driving a whole run.
    #[cfg(test)]
    pub const fn set_num_samples(&mut self, num_samples: usize) {
        self.num_samples = num_samples;
    }
}

/// Helper to deserialize analysis input and create a boxed `Analyze` object.
#[derive(Clone, Deserialize)]
pub enum AnalysisBuilder {
    /// Structure writer
    #[serde(rename = "Trajectory")]
    StructureWriter(StructureWriterBuilder),
    /// Virtual translate analysis for force measurement
    VirtualTranslate(VirtualTranslateBuilder),
    /// Collective variable time series
    CollectiveVariable(CollectiveVariableAnalysisBuilder),
    /// Polymer shape analysis via gyration tensor
    PolymerShape(ShapeAnalysisBuilder),
    /// Radial distribution function g(r)
    RadialDistribution(RadialDistributionBuilder),
    /// Spatial distribution function on a body-fixed grid
    SpatialDistribution(SpatialDistributionBuilder),
    /// Energy time series (total or partial)
    Energy(EnergyAnalysisBuilder),
    /// Mean of one CV binned along another
    MeanAlongCoordinate(MeanAlongCoordinateBuilder),
    /// Scaled Widom insertion for single-ion chemical potential
    ScaledWidomInsertion(ScaledWidomInsertionBuilder),
    /// Virtual volume move for excess pressure measurement
    VirtualVolumeMove(VirtualVolumeMoveBuilder),
    /// Rotational diffusion via quaternion covariance matrix
    RotationalDiffusion(RotationalDiffusionBuilder),
    /// Per-group charge and dipole moment analysis
    Multipole(multipole::MultipoleAnalysisBuilder),
    /// Multipolar decomposition and orientational correlations vs. COM separation
    MultipoleDistribution(MultipoleDistributionBuilder),
    /// Osmotic pressure between two charged planes (Guldbrand midplane method)
    DoubleLayerPressure(DoubleLayerPressureBuilder),
    /// Electric potential profile φ(z) along z (screened slab)
    ElectricPotentialProfile(ElectricPotentialProfileBuilder),
    /// Density profile ρ(z) of a selected species along z
    DensityProfile(DensityProfileBuilder),
    /// Widom rotational perturbation about the center of mass
    WidomRotation(WidomRotationBuilder),
}

/// Prefix `dir` onto a relative output path that stays within `dir`.
///
/// Per-window/per-walker/per-box drivers (umbrella, Wang-Landau, Gibbs)
/// route every analysis output through this so files cannot land in the
/// same place across parallel workers. A leading `/`, a drive prefix, or a
/// `..` component would let the joined path escape `dir` and reintroduce the
/// cross-worker collision, so such paths are rejected.
pub(crate) fn prefix_in_place(p: &mut PathBuf, dir: &Path) -> Result<()> {
    use std::path::Component;
    let escapes = p.components().any(|c| {
        matches!(
            c,
            Component::Prefix(_) | Component::RootDir | Component::ParentDir
        )
    });
    if escapes {
        anyhow::bail!(
            "Output path {p:?} must be relative and stay within the run directory \
             (no absolute path, drive prefix, or `..`)"
        );
    }
    *p = dir.join(&*p);
    Ok(())
}

/// As [`prefix_in_place`], for builders that store an optional path.
pub(crate) fn prefix_opt(p: &mut Option<PathBuf>, dir: &Path) -> Result<()> {
    if let Some(inner) = p {
        prefix_in_place(inner, dir)?;
    }
    Ok(())
}

/// As [`prefix_in_place`], for the `String` path that [`StructureWriter`]
/// stores (kept as `String` for direct use by the XTC writer).
pub(crate) fn prefix_string(s: &mut String, dir: &Path) -> Result<()> {
    let mut path = PathBuf::from(std::mem::take(s));
    prefix_in_place(&mut path, dir)?;
    *s = path
        .into_os_string()
        .into_string()
        .map_err(|os| anyhow::anyhow!("Non-UTF-8 path after prefix: {os:?}"))?;
    Ok(())
}

impl AnalysisBuilder {
    /// Prepend `dir` to every output path on this builder, in place.
    /// Returns an error if any path is absolute.
    pub fn apply_output_dir(&mut self, dir: &Path) -> Result<()> {
        match self {
            Self::StructureWriter(b) => b.apply_output_dir(dir),
            Self::VirtualTranslate(b) => b.apply_output_dir(dir),
            Self::CollectiveVariable(b) => b.apply_output_dir(dir),
            Self::PolymerShape(b) => b.apply_output_dir(dir),
            Self::RadialDistribution(b) => b.apply_output_dir(dir),
            Self::SpatialDistribution(b) => b.apply_output_dir(dir),
            Self::Energy(b) => b.apply_output_dir(dir),
            Self::MeanAlongCoordinate(b) => b.apply_output_dir(dir),
            Self::ScaledWidomInsertion(b) => b.apply_output_dir(dir),
            Self::VirtualVolumeMove(b) => b.apply_output_dir(dir),
            Self::RotationalDiffusion(b) => b.apply_output_dir(dir),
            Self::Multipole(b) => b.apply_output_dir(dir),
            Self::MultipoleDistribution(b) => b.apply_output_dir(dir),
            Self::DoubleLayerPressure(b) => b.apply_output_dir(dir),
            Self::ElectricPotentialProfile(b) => b.apply_output_dir(dir),
            Self::DensityProfile(b) => b.apply_output_dir(dir),
            Self::WidomRotation(b) => b.apply_output_dir(dir),
        }
    }

    /// Build analysis object
    #[must_use = "this returns a Result that should be handled"]
    pub fn build<T: Context>(
        &self,
        context: &T,
        medium: Option<&interatomic::coulomb::Medium>,
    ) -> Result<Box<dyn Analyze<T> + Send>> {
        let rt = medium
            .map(|m| crate::R_IN_KJ_PER_MOL * m.temperature())
            .unwrap_or(crate::R_IN_KJ_PER_MOL * 298.15);
        Ok(match self {
            Self::StructureWriter(builder) => Box::new(builder.build()?),
            Self::VirtualTranslate(builder) => Box::new(builder.build(rt)?),
            Self::CollectiveVariable(builder) => Box::new(builder.build(context)?),
            Self::PolymerShape(builder) => Box::new(builder.build(context)?),
            Self::RadialDistribution(builder) => Box::new(builder.build(context)?),
            Self::SpatialDistribution(builder) => Box::new(builder.build(context)?),
            Self::Energy(builder) => Box::new(builder.build(context)?),
            Self::MeanAlongCoordinate(builder) => Box::new(builder.build(context)?),
            Self::ScaledWidomInsertion(builder) => Box::new(builder.build(context, medium)?),
            Self::VirtualVolumeMove(builder) => Box::new(builder.build(rt)?),
            Self::RotationalDiffusion(builder) => Box::new(builder.build(context)?),
            Self::Multipole(builder) => Box::new(builder.build(context)?),
            Self::MultipoleDistribution(builder) => Box::new(builder.build(context, medium)?),
            Self::DoubleLayerPressure(builder) => Box::new(builder.build(context, medium)?),
            Self::ElectricPotentialProfile(builder) => Box::new(builder.build(context, medium)?),
            Self::DensityProfile(builder) => Box::new(builder.build(context)?),
            Self::WidomRotation(builder) => Box::new(builder.build(context, rt)?),
        })
    }
}

/// Collection of analysis objects. Send-bound required for Gibbs ensemble scoped threads.
pub type AnalysisCollection<T> = Vec<Box<dyn Analyze<T> + Send>>;

/// Create analysis collection from yaml file containing a list of analysis objects under an "analysis" key.
#[must_use = "this returns a Result that should be handled"]
pub fn from_file<T: Context>(
    path: &Path,
    context: &T,
    medium: Option<&interatomic::coulomb::Medium>,
) -> Result<AnalysisCollection<T>> {
    from_file_in_dir(path, context, medium, None)
}

/// As [`from_file`], but creates `output_dir` and prefixes every output
/// path with it before any file is opened. Used by parallel drivers
/// (umbrella, Wang-Landau, Gibbs) to keep per-worker outputs apart.
#[must_use = "this returns a Result that should be handled"]
pub fn from_file_creating_dir<T: Context>(
    path: &Path,
    context: &T,
    medium: Option<&interatomic::coulomb::Medium>,
    output_dir: &Path,
) -> Result<AnalysisCollection<T>> {
    std::fs::create_dir_all(output_dir)?;
    from_file_in_dir(path, context, medium, Some(output_dir))
}

/// As [`from_file`], but additionally prefixes every output path with
/// `output_dir` (when supplied) before any file is opened.
#[must_use = "this returns a Result that should be handled"]
pub fn from_file_in_dir<T: Context>(
    path: &Path,
    context: &T,
    medium: Option<&interatomic::coulomb::Medium>,
    output_dir: Option<&Path>,
) -> Result<AnalysisCollection<T>> {
    let yaml = crate::auxiliary::read_yaml(path)
        .map_err(|err| anyhow::anyhow!("Error reading file {:?}: {}", &path, err))?;
    let value = serde_yml::from_str::<Value>(&yaml)?
        .get("analysis")
        .ok_or_else(|| anyhow::anyhow!("No 'analysis' key found in input yaml file."))?
        .clone();
    let mut builders = crate::auxiliary::from_tagged_list::<AnalysisBuilder>("analysis", &value)?;
    if let Some(dir) = output_dir {
        for b in &mut builders {
            b.apply_output_dir(dir)?;
        }
    }
    builders
        .into_iter()
        .map(|builder| builder.build(context, medium))
        .collect()
}

/// Interface for system analysis.
pub trait Analyze<T: Context>: Debug + Info {
    /// The analysis' sampling state. The only bookkeeping an implementation must store.
    fn sampling(&self) -> &Sampling;
    /// Mutable access, so the framework can count a sample.
    fn sampling_mut(&mut self) -> &mut Sampling;

    /// Perform the actual sampling logic. Called only when the frequency check passes, and never
    /// responsible for counting itself.
    fn perform_sample(&mut self, context: &T, step: usize, weight: f64) -> Result<()>;

    /// Sample system. Checks frequency, then delegates to `perform_sample` with weight 1.
    fn sample(&mut self, context: &T, step: usize) -> Result<()> {
        self.sample_weighted(context, step, 1.0)
    }

    /// Sample with a reweighting factor. Checks frequency, then delegates to `perform_sample`.
    fn sample_weighted(&mut self, context: &T, step: usize, weight: f64) -> Result<()> {
        if self.frequency().should_perform(step) {
            self.sample_now(context, step, weight)?;
        }
        Ok(())
    }

    /// Sample unconditionally and count the frame. Not meant to be overridden.
    fn sample_now(&mut self, context: &T, step: usize, weight: f64) -> Result<()> {
        self.perform_sample(context, step, weight)?;
        self.sampling_mut().num_samples += 1;
        Ok(())
    }

    /// Sampling frequency.
    fn frequency(&self) -> Frequency {
        self.sampling().frequency()
    }

    /// Frames sampled so far.
    fn num_samples(&self) -> usize {
        self.sampling().num_samples()
    }

    /// Called once after the simulation ends, at the final `step`.
    ///
    /// The default samples once when the frequency is `End`. Analyses used to have to override
    /// this to honour `End` at all, and exactly one of them did.
    fn finalize(&mut self, context: &T, step: usize) -> Result<()> {
        if self.frequency().should_perform_at_end() {
            self.sample_now(context, step, 1.0)?;
        }
        Ok(())
    }

    /// Write accumulated results to disk.
    ///
    /// Called once at end of simulation. Analyses that append per-sample
    /// (e.g. energy time series) need only flush; analyses that rewrite
    /// an entire file (e.g. RDF, binned averages) should do the write here
    /// instead of in `sample()`.
    fn write_to_disk(&mut self) -> Result<()> {
        Ok(())
    }

    /// Build the results mapping. Called only when at least one sample was taken, so an
    /// implementation never has to guard against dividing by zero or publishing a mean of nothing.
    fn results(&self) -> Option<serde_yml::Value> {
        None
    }

    /// Results for `output.yaml`, or `None` when nothing was sampled.
    ///
    /// Analyses used to guard this themselves and three of them forgot, publishing `.inf` and
    /// `.nan` — an empty [`WidomAccumulator`](crate::analysis::widom::WidomAccumulator) reports a
    /// free energy of `+inf`, and an empty `WeightedMean` reports `NaN`. The guard lives here now.
    fn to_yaml(&self) -> Option<serde_yml::Value> {
        if self.num_samples() == 0 {
            return None;
        }
        self.results()
    }

    /// Override the sampling frequency. Used by `rerun` to sample every frame.
    fn set_frequency(&mut self, frequency: Frequency) {
        self.sampling_mut().set_frequency(frequency);
    }
}

/// Collect YAML results from all analyses, keyed by short name.
pub fn analyses_to_yaml<T: Context>(analyses: &AnalysisCollection<T>) -> Vec<serde_yml::Value> {
    analyses
        .iter()
        .filter_map(|a| {
            let yaml = a.to_yaml()?;
            let name = a.short_name().unwrap_or("unknown");
            let mut map = serde_yml::Mapping::new();
            map.insert(serde_yml::Value::String(name.to_string()), yaml);
            Some(serde_yml::Value::Mapping(map))
        })
        .collect()
}

impl<T: Context> crate::Info for AnalysisCollection<T> {
    fn short_name(&self) -> Option<&'static str> {
        Some("analysis")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Collection of analysis objects")
    }
}

/// Extension trait for [`AnalysisCollection`].
/// A collection of analyses is not itself an analysis: it has no frequency and no sample count of
/// its own. It only fans out.
pub trait AnalysisCollectionExt<T: Context> {
    /// Override sampling frequency on all analyses. Used by `rerun` to sample every frame.
    fn override_frequencies(&mut self, freq: Frequency);
    fn sample(&mut self, context: &T, step: usize) -> Result<()>;
    fn sample_weighted(&mut self, context: &T, step: usize, weight: f64) -> Result<()>;
    fn finalize(&mut self, context: &T, step: usize) -> Result<()>;
    fn write_to_disk(&mut self) -> Result<()>;
    /// Summed frames across all analyses. Comparable now that every analysis counts frames.
    fn num_samples(&self) -> usize;
}

impl<T: Context> AnalysisCollectionExt<T> for AnalysisCollection<T> {
    fn override_frequencies(&mut self, freq: Frequency) {
        self.iter_mut().for_each(|a| a.set_frequency(freq));
    }
    fn sample(&mut self, context: &T, step: usize) -> Result<()> {
        self.iter_mut().try_for_each(|a| a.sample(context, step))
    }
    fn sample_weighted(&mut self, context: &T, step: usize, weight: f64) -> Result<()> {
        self.iter_mut()
            .try_for_each(|a| a.sample_weighted(context, step, weight))
    }
    fn finalize(&mut self, context: &T, step: usize) -> Result<()> {
        self.iter_mut().try_for_each(|a| a.finalize(context, step))
    }
    fn write_to_disk(&mut self) -> Result<()> {
        self.iter_mut().try_for_each(|a| a.write_to_disk())
    }
    fn num_samples(&self) -> usize {
        self.iter().map(|a| a.num_samples()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const YAML_WITH_FILES: &str = r#"
- !Trajectory
  file: traj.xtc
  frequency: !Every 10
- !Energy
  file: energy.dat
  frequency: !Every 5
- !RadialDistribution
  selections: ["atomtype Na", "atomtype Cl"]
  file: rdf.dat
  dr: 0.1
  frequency: !Every 100
- !SpatialDistribution
  reference: "all"
  selection: "atomtype Na"
  file: spatial.dx
  frequency: !Every 100
- !CollectiveVariable
  property: volume
  range: [1000.0, 5000.0]
  file: cv.dat
  frequency: !Every 1
- !MeanAlongCoordinate
  property: volume
  coordinate:
    property: volume
    resolution: 0.5
  file: mean.dat
  frequency: !Every 1
"#;

    #[test]
    fn prefix_in_place_joins_relative() {
        let mut p = PathBuf::from("traj.xtc");
        prefix_in_place(&mut p, Path::new("window0")).unwrap();
        assert_eq!(p, Path::new("window0").join("traj.xtc"));
    }

    #[test]
    fn prefix_in_place_rejects_absolute() {
        let mut p = PathBuf::from("/tmp/traj.xtc");
        assert!(prefix_in_place(&mut p, Path::new("window0")).is_err());
    }

    #[test]
    fn prefix_in_place_rejects_parent_escape() {
        let mut p = PathBuf::from("../shared/traj.xtc");
        assert!(prefix_in_place(&mut p, Path::new("window0")).is_err());
    }

    #[test]
    fn prefix_opt_skips_none() {
        let mut p: Option<PathBuf> = None;
        prefix_opt(&mut p, Path::new("window0")).unwrap();
        assert!(p.is_none());
    }

    #[test]
    fn prefix_opt_joins_some() {
        let mut p = Some(PathBuf::from("cv.dat"));
        prefix_opt(&mut p, Path::new("window0")).unwrap();
        assert_eq!(p, Some(Path::new("window0").join("cv.dat")));
    }

    #[test]
    fn prefix_string_joins_relative() {
        let mut s = String::from("traj.xtc");
        prefix_string(&mut s, Path::new("window0")).unwrap();
        assert_eq!(PathBuf::from(s), Path::new("window0").join("traj.xtc"));
    }

    #[test]
    fn apply_output_dir_succeeds_for_every_file_bearing_variant() {
        let mut builders: Vec<AnalysisBuilder> = serde_yml::from_str(YAML_WITH_FILES).unwrap();
        for b in &mut builders {
            b.apply_output_dir(Path::new("window7")).unwrap();
        }
    }

    #[test]
    fn apply_output_dir_publicly_observable_via_collective_variable() {
        let yaml = r#"
- !CollectiveVariable
  property: volume
  range: [1000.0, 5000.0]
  file: cv.dat
  frequency: !Every 1
"#;
        let mut builders: Vec<AnalysisBuilder> = serde_yml::from_str(yaml).unwrap();
        builders[0].apply_output_dir(Path::new("window7")).unwrap();
        let AnalysisBuilder::CollectiveVariable(b) = &builders[0] else {
            panic!("expected CollectiveVariable variant");
        };
        assert_eq!(b.file, Some(Path::new("window7").join("cv.dat")));
    }

    #[test]
    fn apply_output_dir_rejects_absolute_path() {
        let yaml = "
- !Energy
  file: /tmp/energy.dat
  frequency: !Every 5
";
        let mut builders: Vec<AnalysisBuilder> = serde_yml::from_str(yaml).unwrap();
        assert!(builders[0].apply_output_dir(Path::new("window0")).is_err());
    }
}

/// Pins the framework behaviour Tier 2 is about to change: which `Frequency` variants actually
/// sample, who owns `num_samples`, and what `to_yaml` emits before any sample has been taken.
///
/// Several assertions here record *bugs*. They are written down so the fix shows up as a
/// deliberate flip in the diff rather than as a test invented to match new code.
#[cfg(test)]
mod framework_characterization {
    use super::*;
    use crate::backend::Backend;

    /// The smallest possible analysis: counts calls, nothing else.
    #[derive(Debug)]
    struct Counter {
        sampling: Sampling,
        finalized: bool,
    }

    impl Counter {
        fn new(frequency: Frequency) -> Self {
            Self {
                sampling: Sampling::new(frequency),
                finalized: false,
            }
        }
    }

    impl crate::Info for Counter {
        fn short_name(&self) -> Option<&'static str> {
            Some("counter")
        }
        fn long_name(&self) -> Option<&'static str> {
            Some("counter")
        }
    }

    impl<T: Context> Analyze<T> for Counter {
        fn sampling(&self) -> &Sampling {
            &self.sampling
        }
        fn sampling_mut(&mut self) -> &mut Sampling {
            &mut self.sampling
        }
        fn perform_sample(&mut self, _context: &T, _step: usize, _weight: f64) -> Result<()> {
            Ok(())
        }
        fn finalize(&mut self, context: &T, step: usize) -> Result<()> {
            self.finalized = true;
            if self.sampling.frequency().should_perform_at_end() {
                self.sample_now(context, step, 1.0)?;
            }
            Ok(())
        }
    }

    fn context() -> Backend {
        let yaml = r#"
atoms:
  - {name: A, mass: 1.0, charge: 0.0, sigma: 1.0}
molecules:
  - name: MOL
    atoms: [A]
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: MOL
      N: 2
      insert: !Manual [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml).unwrap();
        Backend::new(tmp.path(), None, &mut rand::thread_rng()).unwrap()
    }

    fn sample_over_steps(frequency: Frequency, steps: usize) -> usize {
        let context = context();
        let mut counter = Counter::new(frequency);
        for step in 0..steps {
            Analyze::sample(&mut counter, &context, step).unwrap();
        }
        Analyze::<Backend>::num_samples(&counter)
    }

    #[test]
    fn every_and_once_gate_exactly() {
        assert_eq!(sample_over_steps(Frequency::Every(1), 10), 10);
        assert_eq!(sample_over_steps(Frequency::Every(3), 10), 4); // steps 0,3,6,9
        assert_eq!(sample_over_steps(Frequency::Once(4), 10), 1);
        assert_eq!(sample_over_steps(Frequency::Once(99), 10), 0);
    }

    /// `Frequency::Probability` is gone. It never sampled anything — `should_perform` returned
    /// false for it and `should_perform_randomly` was called from nowhere — so a YAML file that
    /// asked for it got silence. It is now a parse error instead.
    #[test]
    fn probability_frequency_no_longer_parses() {
        assert!(serde_yml::from_str::<Frequency>("!Probability 1.0").is_err());
        assert!(serde_yml::from_str::<Frequency>("!Every 10").is_ok());
        assert!(serde_yml::from_str::<Frequency>("!End").is_ok());
    }

    /// Was a bug: `End` never sampled during the run and the default `finalize` ignored it, so
    /// only `structure_writer` honoured it. The default now samples once at the final step.
    #[test]
    fn end_frequency_samples_exactly_once_from_finalize() {
        let context = context();
        let mut counter = Counter::new(Frequency::End);
        for step in 0..10 {
            Analyze::sample(&mut counter, &context, step).unwrap();
        }
        assert_eq!(
            Analyze::<Backend>::num_samples(&counter),
            0,
            "End does not sample during the run"
        );

        Analyze::finalize(&mut counter, &context, 9).unwrap();
        assert_eq!(Analyze::<Backend>::num_samples(&counter), 1);
        assert!(Frequency::End.should_perform_at_end());
    }

    /// ...and a normal frequency must not sample an extra time at the end.
    #[test]
    fn finalize_does_not_double_sample_a_periodic_analysis() {
        let context = context();
        let mut counter = Counter::new(Frequency::Every(1));
        for step in 0..3 {
            Analyze::sample(&mut counter, &context, step).unwrap();
        }
        Analyze::finalize(&mut counter, &context, 2).unwrap();
        assert_eq!(Analyze::<Backend>::num_samples(&counter), 3);
    }

    /// Was a bug: three analyses skipped the `num_samples == 0` guard and published `.inf`/`.nan`
    /// (`WidomAccumulator::mean_free_energy` is `+inf` with no samples). The guard now lives in the
    /// framework's `to_yaml`, so `results()` is only ever called with something to report.
    #[test]
    fn zero_sample_yaml_is_omitted_rather_than_non_finite() {
        let builder: crate::analysis::VirtualVolumeMoveBuilder =
            serde_yml::from_str("{dV: 0.5, method: Isotropic, frequency: !Every 10}").unwrap();
        let analysis = builder.build(2.5).unwrap();
        assert_eq!(Analyze::<Backend>::num_samples(&analysis), 0);
        assert!(Analyze::<Backend>::to_yaml(&analysis).is_none());

        // The framework's frame-count guard is not sufficient on its own: an analysis whose
        // `perform_sample` can return early counts a frame without feeding its accumulator, so it
        // also guards `results()` on the accumulator. Both layers must agree on "nothing to say".
        assert!(Analyze::<Backend>::results(&analysis).is_none());
    }

    /// Every analysis that reports something must be silent before its first sample.
    #[test]
    fn no_analysis_reports_before_its_first_sample() {
        let counter = Counter::new(Frequency::Every(1));
        assert_eq!(Analyze::<Backend>::num_samples(&counter), 0);
        assert!(Analyze::<Backend>::to_yaml(&counter).is_none());
    }

    /// `num_samples` means *frames* in most analyses but *group·frames* in `shape` and
    /// `widom_rotation`, and `AnalysisCollection::num_samples()` sums them regardless.
    #[test]
    fn collection_num_samples_sums_incomparable_counters() {
        let context = context();
        let mut collection: AnalysisCollection<Backend> = vec![
            Box::new(Counter::new(Frequency::Every(1))),
            Box::new(Counter::new(Frequency::Every(1))),
        ];
        for step in 0..5 {
            collection.sample(&context, step).unwrap();
        }
        // Two analyses, five frames each: the sum is 10, not 5.
        assert_eq!(collection.num_samples(), 10);
    }
}

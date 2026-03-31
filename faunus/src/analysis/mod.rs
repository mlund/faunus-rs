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
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_yml::Value;
use std::path::Path;

mod collective_variable;
mod energy;
mod mean_along_coordinate;
mod multipole;
mod radial_distribution;
pub mod reweight;
mod rotational_diffusion;
mod scaled_widom_insertion;
mod shape;
mod structure_writer;
mod virtual_translate;
mod virtual_volume_move;
mod widom;
pub use collective_variable::{CollectiveVariableAnalysis, CollectiveVariableAnalysisBuilder};
pub use energy::{EnergyAnalysis, EnergyAnalysisBuilder};
pub use mean_along_coordinate::{MeanAlongCoordinate, MeanAlongCoordinateBuilder};
pub use radial_distribution::{RadialDistribution, RadialDistributionBuilder};
pub use rotational_diffusion::{RotationalDiffusion, RotationalDiffusionBuilder};
pub use scaled_widom_insertion::{ScaledWidomInsertion, ScaledWidomInsertionBuilder};
pub use shape::{ShapeAnalysis, ShapeAnalysisBuilder};
pub use structure_writer::{StructureWriter, StructureWriterBuilder};
pub use virtual_translate::{VirtualTranslate, VirtualTranslateBuilder};
pub use virtual_volume_move::{VirtualVolumeMove, VirtualVolumeMoveBuilder};

/// Frequency of analysis.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Frequency {
    /// Every `n` steps
    Every(usize),
    /// With probability `p` regardless of number of affected molecules or atoms
    Probability(f64),
    /// Once at step `n`
    Once(usize),
    /// Once at the very last step
    End,
}

impl Frequency {
    /// Check if action, typically a move or analysis, should be performed at given step.
    ///
    /// This handles `Every(n)` and `Once(n)` variants. For `Probability` use
    /// [`should_perform_randomly`](Self::should_perform_randomly), and for `End` use
    /// [`should_perform_at_end`](Self::should_perform_at_end).
    #[must_use]
    #[allow(clippy::manual_is_multiple_of)] // is_multiple_of is not const
    pub const fn should_perform(&self, step: usize) -> bool {
        match self {
            Self::Every(n) => step % *n == 0,
            Self::Once(n) => step == *n,
            Self::Probability(_) | Self::End => false,
        }
    }

    /// Check if action should be performed at the final step.
    ///
    /// Returns `true` only for the `End` variant.
    #[must_use]
    pub const fn should_perform_at_end(&self) -> bool {
        matches!(self, Self::End)
    }

    /// Check if action should be performed based on probability.
    ///
    /// For `Probability(p)`, returns `true` with probability `p`.
    /// Returns `false` for all other variants.
    #[must_use]
    pub fn should_perform_randomly(&self, rng: &mut impl Rng) -> bool {
        match self {
            Self::Probability(p) => rng.gen_bool(*p),
            _ => false,
        }
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
}

impl AnalysisBuilder {
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
            Self::Energy(builder) => Box::new(builder.build(context)?),
            Self::MeanAlongCoordinate(builder) => Box::new(builder.build(context)?),
            Self::ScaledWidomInsertion(builder) => Box::new(builder.build(context, medium)?),
            Self::VirtualVolumeMove(builder) => Box::new(builder.build(rt)?),
            Self::RotationalDiffusion(builder) => Box::new(builder.build(context)?),
            Self::Multipole(builder) => Box::new(builder.build(context)?),
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
    let yaml = crate::auxiliary::read_yaml(path)
        .map_err(|err| anyhow::anyhow!("Error reading file {:?}: {}", &path, err))?;
    let value = serde_yml::from_str::<Value>(&yaml)?
        .get("analysis")
        .ok_or_else(|| anyhow::anyhow!("No 'analysis' key found in input yaml file."))?
        .clone();
    serde_yml::from_value::<Vec<AnalysisBuilder>>(value)?
        .into_iter()
        .map(|builder| builder.build(context, medium))
        .collect()
}

/// Interface for system analysis.
pub trait Analyze<T: Context>: Debug + Info {
    /// Get analysis frequency
    ///
    /// This is the frequency at which the analysis should be performed.
    fn frequency(&self) -> Frequency;

    /// Perform the actual sampling logic. Called only when the frequency check passes.
    fn perform_sample(&mut self, context: &T, step: usize, weight: f64) -> Result<()>;

    /// Sample system. Checks frequency, then delegates to `perform_sample` with weight 1.
    fn sample(&mut self, context: &T, step: usize) -> Result<()> {
        self.sample_weighted(context, step, 1.0)
    }

    /// Sample with a reweighting factor. Checks frequency, then delegates to `perform_sample`.
    fn sample_weighted(&mut self, context: &T, step: usize, weight: f64) -> Result<()> {
        if self.frequency().should_perform(step) {
            self.perform_sample(context, step, weight)
        } else {
            Ok(())
        }
    }

    /// Total number of samples which is the sum of successful calls to `sample()`.
    fn num_samples(&self) -> usize;

    /// Called once after the simulation ends for `End`-frequency work.
    fn finalize(&mut self, context: &T) -> Result<()> {
        let _ = context;
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

    /// Return a YAML representation of the analysis results, if any.
    fn to_yaml(&self) -> Option<serde_yml::Value> {
        None
    }

    /// Override the sampling frequency. Used by `rerun` to sample every frame.
    fn set_frequency(&mut self, _freq: Frequency) {}
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
pub trait AnalysisCollectionExt<T: Context> {
    /// Override sampling frequency on all analyses. Used by `rerun` to sample every frame.
    fn override_frequencies(&mut self, freq: Frequency);
}

impl<T: Context> AnalysisCollectionExt<T> for AnalysisCollection<T> {
    fn override_frequencies(&mut self, freq: Frequency) {
        self.iter_mut().for_each(|a| a.set_frequency(freq));
    }
}

impl<T: Context> Analyze<T> for AnalysisCollection<T> {
    fn perform_sample(&mut self, context: &T, step: usize, weight: f64) -> Result<()> {
        self.iter_mut()
            .try_for_each(|a| a.sample_weighted(context, step, weight))
    }
    fn sample(&mut self, context: &T, step: usize) -> Result<()> {
        self.iter_mut().try_for_each(|a| a.sample(context, step))
    }
    fn sample_weighted(&mut self, context: &T, step: usize, weight: f64) -> Result<()> {
        self.iter_mut()
            .try_for_each(|a| a.sample_weighted(context, step, weight))
    }
    /// Summed number of samples for all analysis objects
    fn num_samples(&self) -> usize {
        self.iter().map(|a| a.num_samples()).sum()
    }
    fn frequency(&self) -> Frequency {
        Frequency::Every(1)
    }
    fn finalize(&mut self, context: &T) -> Result<()> {
        self.iter_mut().try_for_each(|a| a.finalize(context))
    }
    fn write_to_disk(&mut self) -> Result<()> {
        self.iter_mut().try_for_each(|a| a.write_to_disk())
    }
}

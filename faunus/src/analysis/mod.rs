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
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_yaml::Value;
use std::path::PathBuf;

mod distance;
mod structure_writer;
mod virtual_translate;
pub use distance::{MassCenterDistance, MassCenterDistanceBuilder};
#[cfg(feature = "chemfiles")]
pub use structure_writer::{StructureWriter, StructureWriterBuilder};
pub use virtual_translate::{VirtualTranslate, VirtualTranslateBuilder};

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
    /// Mass center distance analysis
    MassCenterDistance(MassCenterDistanceBuilder),
    /// Structure writer
    #[cfg(feature = "chemfiles")]
    #[serde(rename = "Trajectory")]
    StructureWriter(StructureWriterBuilder),
    /// Virtual translate analysis for force measurement
    VirtualTranslate(VirtualTranslateBuilder),
}

impl AnalysisBuilder {
    /// Build analysis object
    #[must_use = "this returns a Result that should be handled"]
    pub fn build<T: Context>(&self, context: &T) -> Result<Box<dyn Analyze<T>>> {
        let analysis: Box<dyn Analyze<T>> = match self {
            Self::MassCenterDistance(builder) => Box::new(builder.build(&context.topology())?),
            #[cfg(feature = "chemfiles")]
            Self::StructureWriter(builder) => Box::new(builder.build()?),
            Self::VirtualTranslate(builder) => {
                Box::new(builder.build(&context.topology())?)
            }
        };
        Ok(analysis)
    }
}

/// Collection of analysis objects.
pub type AnalysisCollection<T> = Vec<Box<dyn Analyze<T>>>;

/// Create analysis collection from yaml file containing a list of analysis objects under an "analysis" key.
#[must_use = "this returns a Result that should be handled"]
pub fn from_file<T: Context>(path: &PathBuf, context: &T) -> Result<AnalysisCollection<T>> {
    let yaml = std::fs::read_to_string(path)
        .map_err(|err| anyhow::anyhow!("Error reading file {:?}: {}", &path, err))?;
    let value = serde_yaml::from_str::<Value>(&yaml)?
        .get("analysis")
        .ok_or_else(|| anyhow::anyhow!("No 'analysis' key found in input yaml file."))?
        .clone();
    serde_yaml::from_value::<Vec<AnalysisBuilder>>(value)?
        .into_iter()
        .map(|builder| builder.build(context))
        .collect()
}

/// Interface for system analysis.
pub trait Analyze<T: Context>: Debug + Info {
    /// Get analysis frequency
    ///
    /// This is the frequency at which the analysis should be performed.
    fn frequency(&self) -> Frequency;

    /// Sample system.
    fn sample(&mut self, context: &T, step: usize) -> Result<()>;

    /// Total number of samples which is the sum of successful calls to `sample()`.
    fn num_samples(&self) -> usize;

    /// Flush output stream, if any, ensuring that all intermediately buffered contents reach their destination.
    fn flush(&mut self) {}
}

impl<T: Context> crate::Info for AnalysisCollection<T> {
    fn short_name(&self) -> Option<&'static str> {
        Some("analysis")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Collection of analysis objects")
    }
}

impl<T: Context> Analyze<T> for AnalysisCollection<T> {
    fn sample(&mut self, context: &T, step: usize) -> Result<()> {
        self.iter_mut().try_for_each(|a| a.sample(context, step))
    }
    /// Summed number of samples for all analysis objects
    fn num_samples(&self) -> usize {
        self.iter().map(|a| a.num_samples()).sum()
    }
    fn frequency(&self) -> Frequency {
        Frequency::Every(1)
    }
    fn flush(&mut self) {
        self.iter_mut().for_each(|a| a.flush())
    }
}

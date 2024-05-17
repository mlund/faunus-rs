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

use super::montecarlo::Frequency;
use crate::{Context, Info};
use anyhow::Result;
use core::fmt::Debug;
use rand::rngs::ThreadRng;

/// Collection of analysis objects.
pub type AnalysisCollection<T> = Vec<Box<dyn Analyze<T>>>;

/// Interface for system analysis.
pub trait Analyze<T: Context>: Debug + Info {
    /// Get analysis frequency
    ///
    /// This is the frequency at which the analysis should be performed.
    fn frequency(&self) -> Frequency;

    /// Sample system.
    fn sample(&mut self, context: &T, step: usize, rng: &mut ThreadRng) -> Result<()>;

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
    fn sample(&mut self, context: &T, step: usize, rng: &mut ThreadRng) -> Result<()> {
        self.iter_mut()
            .try_for_each(|a| a.sample(context, step, rng))
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

/// Writes structure of the system in the specified format during the simulation.
#[cfg(feature = "chemfiles")]
#[derive(Debug)]
pub struct StructureWriter {
    output_file: String,
    trajectory: Option<chemfiles::Trajectory>,
    frequency: Frequency,
    num_samples: usize,
}

impl StructureWriter {
    pub fn new(output_file: &str, frequency: Frequency) -> StructureWriter {
        StructureWriter {
            output_file: output_file.to_owned(),
            frequency,
            trajectory: None,
            num_samples: 0,
        }
    }
}

#[cfg(feature = "chemfiles")]
impl crate::Info for StructureWriter {
    fn short_name(&self) -> Option<&'static str> {
        Some("structure printer")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Writes structure of the system at specified frequency into an output trajectory.")
    }
}

#[cfg(feature = "chemfiles")]
impl<T: Context> Analyze<T> for StructureWriter {
    fn sample(&mut self, context: &T, step: usize, rng: &mut ThreadRng) -> anyhow::Result<()> {
        if !self.frequency.should_perform(step, rng) {
            return Ok(());
        }

        let frame = context.to_frame();

        if self.trajectory.is_none() {
            self.trajectory = Some(chemfiles::Trajectory::open(&self.output_file, 'w')?);
        }

        self.trajectory.as_mut().unwrap().write(&frame)?;
        self.num_samples += 1;
        Ok(())
    }

    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }
}

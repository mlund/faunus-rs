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

/// Collection of analysis objects.
pub type AnalysisCollection<T> = Vec<Box<dyn Analyze<T>>>;

/// Interface for system analysis.
pub trait Analyze<T: Context>: Debug + Info {
    /// Get analysis frequency
    ///
    /// This is the frequency at which the analysis should be performed.
    fn frequency(&self) -> Frequency;

    /// Sample system.
    fn sample(&mut self, context: &T) -> Result<()>;

    /// Total number of samples which is the sum of successful calls to `sample()`.
    fn num_samples(&self) -> usize;

    /// Flush output stream, if any, ensuring that all intermediately buffered contents reach their destination.
    fn flush(&mut self) {}

    /// Report analysis as JSON object
    fn to_json(&self) -> Option<serde_json::Map<String, serde_json::Value>> {
        None
    }
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
    fn sample(&mut self, context: &T) -> Result<()> {
        self.iter_mut().try_for_each(|a| a.sample(context))
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
    fn to_json(&self) -> Option<serde_json::Map<String, serde_json::Value>> {
        let mut j = serde_json::Map::new();
        for a in self.iter() {
            if let Some(mut j2) = a.to_json() {
                j.append(&mut j2);
            }
        }
        Some(j)
    }
}

impl<T: Context> From<Box<dyn Analyze<T>>> for serde_json::Value {
    fn from(analyze: Box<dyn Analyze<T>>) -> Self {
        let mut j = analyze.to_json().unwrap_or_default();
        j.insert("samples".into(), analyze.num_samples().into());
        j.insert(
            "frequency".into(),
            serde_json::to_value(analyze.frequency()).unwrap_or_default(),
        );
        j.into()
    }
}

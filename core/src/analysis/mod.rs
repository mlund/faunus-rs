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
use anyhow::Result;
use core::fmt::Debug;
use serde::Serialize;

pub trait Analyze: Debug {
    /// Get analysis frequency
    fn frequency(&self) -> Frequency;

    /// Sample system
    fn sample(&mut self) -> Result<()>;

    /// Total number of samples
    fn samples(&self) -> usize;

    /// Flush output stream, if any, ensuring that all intermediately buffered contents reach their destination.
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    /// Report analysis as JSON object
    fn to_json(&self) -> Option<serde_json::Map<String, serde_json::Value>> {
        None
    }
}

impl From<Box<dyn Analyze>> for serde_json::Map<String, serde_json::Value> {
    fn from(analyze: Box<dyn Analyze>) -> Self {
        analyze.to_json().unwrap_or_default()
    }
}

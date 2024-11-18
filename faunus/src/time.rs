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

//! Support for operations dealing with *time*.

use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::time::Duration;

/// Helper class to keep track of time spent in a Monte Carlo move
///
/// The reported time is the accumulated time spent between multiple calls to `start` and `stop`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Timer {
    /// Start time
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    start: Option<std::time::Instant>,
    /// Accumulated time
    #[serde(skip_deserializing)]
    accumulated: Duration,
}

impl Timer {
    /// Start the timer
    pub fn start(&mut self) {
        self.start = Some(std::time::Instant::now());
    }

    /// Stop the timer and accumulate the time. Errors if the timer was not started.
    pub fn stop(&mut self) -> Result<(), anyhow::Error> {
        if let Some(start) = self.start {
            self.accumulated += start.elapsed();
            self.start = None;
            Ok(())
        } else {
            anyhow::bail!("Timer was not started");
        }
    }

    /// Get the accumulated time spent between all calls to `start` and `stop`
    pub const fn accumulated(&self) -> Duration {
        self.accumulated
    }

    /// Clear the accumulated time
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl Display for Timer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let elapsed = self.accumulated.as_secs_f32();
        write!(f, "⏱️ elapsed = {:.2} s", elapsed)
    }
}

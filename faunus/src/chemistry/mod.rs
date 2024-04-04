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

//! # Chemistry module
//!
//! This contains support for chemical systems, including electrolyte solutions.
mod electrolyte;
pub mod reaction;

pub use electrolyte::{bjerrum_length, Electrolyte, PermittivityNR, Salt};

/// Trait for the relative permittivity
pub trait RelativePermittivity {
    /// Relative permittivity of the medium at a given temperature in K
    /// # Errors
    /// If the temperature is outside the valid range
    fn relative_permittivity(&self, temperature: f64) -> anyhow::Result<f64>;
}

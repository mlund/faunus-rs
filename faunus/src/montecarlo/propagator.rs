// Copyright 2023-2024 Mikael Lund
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

use serde::{Deserialize, Serialize};

use crate::Context;

use super::{Move, TranslateMolecule};

/// An enum for all supported MC moves and MD integrators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Propagator {
    TranslateMolecule(TranslateMolecule),
}

impl Propagator {
    /// Validates and finalizes the propagator.
    ///
    /// Validation:
    /// Checks that the definition of the propagator is valid, i.e., it does not reference any undefined atoms, molecules etc.
    ///
    /// Finalization:
    /// Sets required properties of the propagator that could not be read from the input file.
    pub(super) fn finalize(&mut self, context: &impl Context) -> anyhow::Result<()> {
        match self {
            Propagator::TranslateMolecule(x) => x.finalize(context)?,
        }

        Ok(())
    }
}

impl<T: Context> From<Propagator> for Box<dyn Move<T>> {
    /// Converts the propagator into a Move trait object.
    fn from(value: Propagator) -> Self {
        match value {
            Propagator::TranslateMolecule(x) => Box::from(x) as Box<dyn Move<T>>,
        }
    }
}

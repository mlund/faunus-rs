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

use crate::{Change, Info, SyncFromAny};
use as_any::AsAny;

/// Trait for describing terms in the Hamiltonian.
pub trait EnergyTerm: Info + AsAny + std::fmt::Debug + SyncFromAny {
    /// Compute the energy change of the term due to a change in the system.
    /// The energy is returned in units of kJ/mol.
    fn energy_change(&self, change: &Change) -> Option<f64>;
}

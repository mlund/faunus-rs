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

//! # Energy calculation and Hamiltonian

use crate::{Change, SyncFrom};
use as_any::AsAny;

/// Collection of energy terms.
///
/// The Hamiltonian is a collection of energy terms,
/// that itself implements the `EnergyTerm` trait for summing them up.
pub type Hamiltonian = Vec<Box<dyn EnergyTerm>>;

/// Trait for describing terms in the Hamiltonian.
pub trait EnergyTerm: AsAny + SyncFrom {
    /// Compute the energy change of the term due to a change in the system.
    /// The energy is returned in units of kJ/mol.
    fn energy_change(&self, change: &Change) -> f64;

    /// Update internal state due to a change in the system.
    ///
    /// After a system change, the energy term may need to update its internal state.
    /// For example, in Ewald summation, the reciprocal space energy needs to be updated.
    /// By default, this does nothing.
    #[allow(unused_variables)]
    fn update(&mut self, change: &Change) -> anyhow::Result<()> {
        Ok(())
    }
}

impl Clone for Box<dyn EnergyTerm> {
    #[allow(unconditional_recursion)]
    fn clone(&self) -> Self {
        self.as_any().downcast_ref::<Self>().unwrap().clone()
    }
}

impl EnergyTerm for Hamiltonian {
    fn energy_change(&self, change: &Change) -> f64 {
        let mut sum = 0.0;
        for term in self.iter() {
            sum += term.energy_change(change);
            // early exit if sum is NaN or infinite
            if sum.is_nan() || sum.is_infinite() {
                break;
            }
        }
        sum
    }

    fn update(&mut self, change: &Change) -> anyhow::Result<()> {
        self.iter_mut().try_for_each(|term| term.update(change))
    }
}

impl SyncFrom for Hamiltonian {
    fn sync_from(&mut self, other: &dyn as_any::AsAny, change: &Change) -> anyhow::Result<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        for (term, other_term) in self.iter_mut().zip(other.iter()) {
            term.sync_from(other_term, change)?;
        }
        Ok(())
    }
}

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

pub mod bonded;
pub(crate) mod builder;

use std::path::Path;

use crate::{
    platform::reference::nonbonded::NonbondedReference, topology::Topology, Change, Context,
    SyncFrom,
};

use self::{
    bonded::{IntermolecularBonds, IntramolecularBonds},
    builder::HamiltonianBuilder,
};

/// Collection of energy terms.
///
/// The Hamiltonian is a collection of energy terms,
/// that itself implements the `EnergyTerm` trait for summing them up.
#[derive(Debug, Clone, Default)]
pub struct Hamiltonian {
    energy_terms: Vec<EnergyTerm>,
}

impl SyncFrom for Hamiltonian {
    /// Synchronize the Hamiltonian from other Hamiltonian.
    fn sync_from(&mut self, other: &Hamiltonian, change: &Change) -> anyhow::Result<()> {
        for (term, other_term) in self.energy_terms.iter_mut().zip(other.energy_terms.iter()) {
            term.sync_from(other_term, change)?;
        }

        Ok(())
    }
}

impl Hamiltonian {
    /// Create a Hamiltonian from the provided energy terms.
    pub fn new(terms: Vec<EnergyTerm>) -> Self {
        Hamiltonian {
            energy_terms: terms,
        }
    }

    /// Add an energy term into the Hamiltonian.
    pub fn add_energy_term(&mut self, term: EnergyTerm) -> anyhow::Result<()> {
        self.energy_terms.push(term);
        Ok(())
    }

    /// Compute the energy change of the Hamiltonian due to a change in the system.
    /// The energy is returned in the units of kJ/mol.
    pub fn energy_change(&self, context: &impl Context, change: &Change) -> f64 {
        self.energy_terms
            .iter()
            .map(|term| term.energy_change(context, change))
            // early return if the total energy becomes none or infinite
            .take_while(|&energy| !energy.is_nan() && !energy.is_infinite())
            .sum()
    }

    /// Update internal state due to a change in the system.
    ///
    /// After a system change, the internal state of the energy terms may need to be updated.
    /// For example, in Ewald summation, the reciprocal space energy needs to be updated.
    pub fn update(&mut self, change: &Change) -> anyhow::Result<()> {
        self.energy_terms
            .iter_mut()
            .try_for_each(|term| term.update(change))?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum EnergyTerm {
    /// Non-bonded interactions between particles for reference platform.
    NonbondedReference(NonbondedReference),
    /// Intramolecular bonded interactions.
    IntramolecularBonds(IntramolecularBonds),
    /// Intermolecular bonded interactions.
    IntermolecularBonds(IntermolecularBonds),
}

impl EnergyTerm {
    /// Create an EnergyTerm for Nonbonded interactions by reading an input file.
    /// The input file must contain topology of the System and an `energy` section.
    pub fn nonbonded_from_file(filename: impl AsRef<Path> + Clone) -> anyhow::Result<Self> {
        let hamiltonian_builder = HamiltonianBuilder::from_file(filename.clone())?;
        let topology = Topology::from_file(filename)?;

        NonbondedReference::new(&hamiltonian_builder.nonbonded, &topology)
    }

    /// Compute the energy change of the EnergyTerm due to a change in the system.
    /// The energy is returned in the units of kJ/mol.
    fn energy_change(&self, context: &impl Context, change: &Change) -> f64 {
        0.0
    }

    /// Update internal state due to a change in the system.
    fn update(&mut self, _change: &Change) -> anyhow::Result<()> {
        match self {
            EnergyTerm::NonbondedReference(_)
            | EnergyTerm::IntramolecularBonds(_)
            | EnergyTerm::IntermolecularBonds(_) => (),
        }

        Ok(())
    }
}

impl SyncFrom for EnergyTerm {
    /// Synchronize the EnergyTerm from other EnergyTerm.
    ///
    /// Panics if the EnergyTerms are not compatible with each other.
    fn sync_from(&mut self, other: &EnergyTerm, change: &Change) -> anyhow::Result<()> {
        match (self, other) {
            (EnergyTerm::NonbondedReference(x), EnergyTerm::NonbondedReference(y)) => {
                x.sync_from(y, change)?
            }
            (EnergyTerm::IntramolecularBonds(x), EnergyTerm::IntramolecularBonds(y)) => {
                x.sync_from(y, change)?
            }
            (EnergyTerm::IntermolecularBonds(x), EnergyTerm::IntermolecularBonds(y)) => {
                x.sync_from(y, change)?
            }
            _ => panic!("Trying to sync incompatible energy terms."),
        }

        Ok(())
    }
}

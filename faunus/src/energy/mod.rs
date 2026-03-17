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

pub(crate) mod bonded;
pub(crate) mod builder;
mod celloverlap;
mod constrain;
mod custom_external;
mod energyterm;
pub(crate) mod ewald;
mod excluded_coulomb;
pub mod exclusions;
mod external_pressure;
mod hamiltonian;
mod nonbonded;
#[cfg(feature = "gpu")]
pub(crate) mod nonbonded_kernel;
mod pairpot;
mod polymer_depletion;
mod sasa;
mod tabulated6d;

pub use bonded::{IntermolecularBonded, IntramolecularBonded};
pub use celloverlap::CellOverlap;
pub use constrain::{Constrain, ConstrainBuilder};
pub use custom_external::{CustomExternal, CustomExternalBuilder};
pub use energyterm::EnergyTerm;
pub use ewald::{EwaldBuilder, EwaldReciprocalEnergy};
pub use external_pressure::{ExternalPressure, Pressure};
pub use hamiltonian::{EnergyChange, Hamiltonian};
pub use nonbonded::{NonbondedMatrix, NonbondedMatrixSplined};
pub use pairpot::PairPot;
pub use polymer_depletion::{PolymerDepletion, PolymerDepletionBuilder};
pub use sasa::{SasaEnergy, SasaEnergyBuilder};
pub use tabulated6d::{Tabulated6D, Tabulated6DBuilder};

// Re-export spline types from interatomic for convenience
pub use interatomic::twobody::{GridType, SplineConfig};

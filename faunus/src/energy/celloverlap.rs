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

//! Implementation of the Nonbonded energy terms.

use std::fmt::Debug;

use crate::{
    cell::{BoundaryConditions, PeriodicDirections, Shape},
    energy::EnergyTerm,
    Change, Context,
};

/// Returns infinite energy if particles are outside the simulation cell boundaries; zero otherwise.
#[derive(Copy, Clone, Default, PartialEq, Debug)]
pub struct CellOverlap;

impl CellOverlap {
    pub fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        let cell = context.cell();
        // Assume no overlap if no change or PBC in all directions
        if matches!(change, Change::None) || cell.pbc() == PeriodicDirections::PeriodicXYZ {
            0.0
        } else if context
            .get_active_particles()
            .iter()
            .any(|particle| cell.is_outside(&particle.pos))
        {
            f64::INFINITY // ensure MC rejection
        } else {
            0.0
        }
    }
}

impl From<CellOverlap> for EnergyTerm {
    fn from(celloverlap: CellOverlap) -> Self {
        EnergyTerm::CellOverlap(celloverlap)
    }
}

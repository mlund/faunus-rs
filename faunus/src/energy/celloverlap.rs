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
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
pub struct CellOverlap;

impl CellOverlap {
    pub fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        let cell = context.cell();
        // Assume no overlap if no change or PBC in all directions
        if matches!(change, Change::None) || cell.pbc() == PeriodicDirections::PeriodicXYZ {
            0.0
        } else if context
            .groups()
            .iter()
            .flat_map(|g| g.iter_active())
            .any(|i| cell.is_outside(&context.position(i)))
        {
            f64::INFINITY // ensure MC rejection
        } else {
            0.0
        }
    }
}

impl From<CellOverlap> for EnergyTerm {
    fn from(celloverlap: CellOverlap) -> Self {
        Self::CellOverlap(celloverlap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::Backend, group::GroupCollection, Point};

    /// Verify CellOverlap returns infinity when a particle is outside the cell.
    #[test]
    fn celloverlap_detects_outside() {
        // Slit cell has open z-boundary — particles outside z should trigger infinity
        let yaml = r#"
atoms:
  - {name: X, mass: 1.0, sigma: 1.0}
molecules:
  - name: particle
    atoms: [X]
    atomic: true
system:
  cell: !Slit [10.0, 10.0, 10.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {celloverlap: {}}
  blocks:
    - molecule: particle
      N: 1
      insert: !RandomAtomPos {}
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), yaml).unwrap();
        let mut ctx = Backend::new(tmp.path(), None, &mut rand::thread_rng()).unwrap();

        // Place particle well inside the cell
        ctx.set_positions(0..1, [Point::new(0.0, 0.0, 0.0)].iter());
        let overlap = CellOverlap;
        assert_eq!(overlap.energy(&ctx, &Change::Everything), 0.0);

        // Place particle outside the slit z-boundary
        ctx.set_positions(0..1, [Point::new(0.0, 0.0, 100.0)].iter());
        assert_eq!(overlap.energy(&ctx, &Change::Everything), f64::INFINITY);
    }
}

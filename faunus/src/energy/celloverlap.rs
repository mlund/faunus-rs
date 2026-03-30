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
        if matches!(change, Change::None) || cell.pbc() == PeriodicDirections::PeriodicXYZ {
            return 0.0;
        }
        let is_outside = |i: usize| cell.is_outside(&context.position(i));
        // Only check particles involved in the change; checking all active particles
        // incorrectly rejects moves when unchanged groups have out-of-bounds stored positions.
        let outside = match change {
            Change::Everything | Change::Volume(_, _) => context
                .groups()
                .iter()
                .flat_map(|g| g.iter_active())
                .any(is_outside),
            Change::Groups(changes) => changes
                .iter()
                .any(|(gi, _)| context.groups()[*gi].iter_active().any(is_outside)),
            Change::SingleGroup(gi, _) => context.groups()[*gi].iter_active().any(is_outside),
            Change::None => false,
        };
        if outside { f64::INFINITY } else { 0.0 }
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

    // Two non-atomic single-atom molecules so each gets its own group index.
    const SLIT_YAML: &str = r#"
atoms:
  - {name: X, mass: 1.0, sigma: 1.0}
molecules:
  - name: A
    atoms: [X]
  - name: B
    atoms: [X]
system:
  cell: !Slit [10.0, 10.0, 10.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {celloverlap: {}}
  blocks:
    - molecule: A
      N: 1
      insert: !RandomAtomPos {}
    - molecule: B
      N: 1
      insert: !RandomAtomPos {}
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;

    fn make_context() -> Backend {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), SLIT_YAML).unwrap();
        Backend::new(tmp.path(), None, &mut rand::thread_rng()).unwrap()
    }

    /// Verify CellOverlap returns infinity when a particle is outside the cell.
    #[test]
    fn celloverlap_detects_outside() {
        let mut ctx = make_context();
        ctx.set_positions(0..2, [Point::new(0.0, 0.0, 0.0), Point::new(0.0, 0.0, 0.0)].iter());
        let overlap = CellOverlap;
        assert_eq!(overlap.energy(&ctx, &Change::Everything), 0.0);

        // Particle 0 outside the slit z-boundary
        ctx.set_positions(0..1, [Point::new(0.0, 0.0, 100.0)].iter());
        assert_eq!(overlap.energy(&ctx, &Change::Everything), f64::INFINITY);
    }

    /// Only the changed group is checked — an out-of-bounds particle in an unchanged group
    /// must not cause rejection.
    #[test]
    fn celloverlap_only_checks_changed_group() {
        use crate::GroupChange;
        let mut ctx = make_context();
        // Group 0 (particle 0) inside, group 1 (particle 1) outside
        ctx.set_positions(0..1, [Point::new(0.0, 0.0, 0.0)].iter());
        ctx.set_positions(1..2, [Point::new(0.0, 0.0, 100.0)].iter());
        let overlap = CellOverlap;
        // Change only group 0 — group 1's out-of-bounds position must be ignored
        let change = Change::SingleGroup(0, GroupChange::RigidBody);
        assert_eq!(overlap.energy(&ctx, &change), 0.0);
        // Change only group 1 — its out-of-bounds position must be detected
        let change = Change::SingleGroup(1, GroupChange::RigidBody);
        assert_eq!(overlap.energy(&ctx, &change), f64::INFINITY);
    }
}

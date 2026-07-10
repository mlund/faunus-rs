//! A minimal, working energy term, kept as a starting point for new ones.
//!
//! It applies a harmonic restraint pulling every particle towards the z = 0 plane. Nothing about
//! that is interesting; what the file demonstrates is the contract every energy term must keep.
//!
//! **A term reads the system; it never writes to it.** `energy` receives a read-only
//! [`ObserveContext`](crate::ObserveContext), so the mutating methods are not merely discouraged,
//! they are absent from the type. Interior mutability for *caching* is still allowed — see the
//! `RefCell<ComSelection>` in `custom_external.rs` — but never for changing the physics.
//!
//! **`energy` must be consistent with the [`Change`] it is given.** The framework calls it twice
//! per trial move, once before and once after, and subtracts. If a term returns a cached value for
//! a change that actually affected it, ΔU is wrong in a way the energy-drift check cannot see: it
//! sums the same wrong numbers. When in doubt, recompute — as this term does, by ignoring `change`
//! and always summing over the whole system. That is correct but O(N) per move; the real terms
//! narrow it, and pay for that with the `save_backup`/`undo` cache protocol.
//!
//! The test at the bottom exercises the term against a real system, so this file cannot rot. Copy
//! it, rename it, add a variant to [`EnergyTerm`](super::EnergyTerm), and register the builder.

// Nothing but the tests below drives this term; it is a template, not a registered energy term.
#![allow(dead_code)]

use super::EnergyChange;
use crate::{Change, ObserveContext};
use serde::{Deserialize, Serialize};

/// Deserialized from the `energy:` section of the input file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZRestraintBuilder {
    /// Force constant in kJ/(mol·Å²).
    spring_constant: f64,
}

impl ZRestraintBuilder {
    pub fn build(&self) -> anyhow::Result<ZRestraint> {
        anyhow::ensure!(
            self.spring_constant >= 0.0,
            "ZRestraint: spring constant must be non-negative"
        );
        Ok(ZRestraint {
            spring_constant: self.spring_constant,
        })
    }
}

/// Harmonic restraint towards the z = 0 plane: `U = ½k Σᵢ zᵢ²`.
#[derive(Debug, Clone)]
pub struct ZRestraint {
    spring_constant: f64,
}

impl EnergyChange for ZRestraint {
    /// The context is immutable, and `Change` is ignored because this term recomputes in full.
    ///
    /// A term that *does* use `change` must return an energy for exactly the particles the change
    /// touches, and must return the same partition before and after the move.
    fn energy(&self, context: &impl ObserveContext, _change: &Change) -> f64 {
        let sum_z_squared: f64 = context
            .groups()
            .iter()
            .flat_map(|group| group.iter_active())
            .map(|i| context.position(i).z.powi(2))
            .sum();
        0.5 * self.spring_constant * sum_z_squared
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::context::PerturbContext;
    use crate::Point;

    /// Two atoms at z = ±2, so the restraint energy is ½·k·(4 + 4) = 4k.
    const TWO_ATOMS: &str = r#"
atoms:
  - {name: A, mass: 1.0, charge: 0.0, sigma: 1.0}
molecules:
  - name: dimer
    atoms: [A, A]
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: dimer
      N: 1
      insert: !Manual [[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;

    fn backend() -> Backend {
        Backend::from_yaml_str(TWO_ATOMS, None, &mut rand::thread_rng()).unwrap()
    }

    #[test]
    fn restrains_particles_towards_the_plane() {
        let context = backend();
        let term = ZRestraintBuilder {
            spring_constant: 2.0,
        }
        .build()
        .unwrap();
        assert_eq!(term.energy(&context, &Change::Everything), 8.0);
    }

    /// The energy must follow the positions, not a stale cache.
    #[test]
    fn the_energy_follows_the_particles() {
        let mut context = backend();
        let term = ZRestraintBuilder {
            spring_constant: 2.0,
        }
        .build()
        .unwrap();

        // Slide both atoms onto the plane; the restraint energy must vanish.
        context.translate_particles(&[0], &Point::new(0.0, 0.0, 2.0));
        context.translate_particles(&[1], &Point::new(0.0, 0.0, -2.0));
        assert_eq!(term.energy(&context, &Change::Everything), 0.0);
    }

    #[test]
    fn a_negative_spring_constant_is_rejected() {
        assert!(ZRestraintBuilder {
            spring_constant: -1.0
        }
        .build()
        .is_err());
    }
}

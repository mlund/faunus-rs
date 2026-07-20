//! A minimal, working [`MoveProposal`] implementation, kept as a starting point for new moves.
//!
//! It translates a random molecule of one kind along the z axis only. Nothing about that is
//! interesting; what the file demonstrates is the contract every move must keep.
//!
//! A move *proposes*; it never mutates. `propose_move` receives a read-only
//! [`ObserveContext`](crate::ObserveContext), and returns a [`ProposedMove`] describing what should
//! happen. The framework applies it, evaluates ΔU, and either keeps it or calls `undo`.
//!
//! Build the [`ProposedMove`] with one of its constructors — `translate_group`, `rotate_group`,
//! `translate_atoms`, `rotate_atoms`. Each derives the [`Change`](crate::Change) that drives the
//! energy recomputation *from* the transform it builds, so the two cannot disagree. A move that set
//! them separately could announce `RigidBody` while rotating part of a molecule, and every energy
//! term would then return the same number for both states: ΔU ≈ 0, the move is always accepted, and
//! the energy-drift check stays green because it sums the same wrong numbers.
//!
//! The test at the bottom runs the move against a real system, so this file cannot rot. Copy it,
//! rename it, and register it in [`MoveBuilder`](super::builder::MoveBuilder).

// Nothing but the tests below drives this move; it is a template, not a registered move.
#![allow(dead_code)]

use super::{find_molecule_id, random_group};
use crate::group::MoleculeId;
use crate::propagate::{tagged_yaml, MoveProposal, ProposedMove};
use crate::{ObserveContext, Point};
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};

/// Deserialized from the `propagate:` section of the input file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranslateAlongZ {
    /// Name of the molecule kind to move.
    #[serde(rename = "molecule")]
    molecule_name: String,
    /// Maximum displacement along z, in Å.
    max_displacement: f64,
    /// Resolved in `finalize`, once the context is known.
    #[serde(skip)]
    molecule_id: MoleculeId,
}

impl TranslateAlongZ {
    /// Resolve names against the topology before the first move is proposed.
    ///
    /// Failing here turns a typo in the input file into a startup error rather than a move that
    /// silently never fires.
    pub(crate) fn finalize(&mut self, context: &impl ObserveContext) -> anyhow::Result<()> {
        self.molecule_id = find_molecule_id(context, &self.molecule_name, "TranslateAlongZ")?;
        Ok(())
    }
}

impl<T: ObserveContext> MoveProposal<T> for TranslateAlongZ {
    /// `context` is `&T` and `T: ObserveContext`, so no mutation is reachable from here — not by
    /// convention, but because the methods that would do it are not on the trait.
    fn propose_move(&mut self, context: &T, rng: &mut dyn RngCore) -> Option<ProposedMove> {
        let group_index = random_group(context, rng, self.molecule_id)?;
        let shift = Point::new(
            0.0,
            0.0,
            rng.gen_range(-self.max_displacement..self.max_displacement),
        );
        Some(ProposedMove::translate_group(group_index, shift))
    }

    fn to_yaml(&self) -> Option<yaml_serde::Value> {
        tagged_yaml("TranslateAlongZ", self)
    }
}

// The shared `impl_info!` macro models the blessed pattern: real moves use it rather than
// hand-writing the `short_name`/`long_name` accessors.
impl_info!(
    TranslateAlongZ,
    "translate_along_z",
    "Translation of a random molecule along z"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::{Change, GroupChange};

    const TWO_DIMERS: &str = r#"
atoms:
  - {name: A, mass: 1.0, charge: 0.0, sigma: 1.0}
molecules:
  - name: dimer
    atoms: [A, A]
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: []
  blocks:
    - molecule: dimer
      N: 2
      insert: !Manual [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, steps: 0, collections: []}
"#;

    fn move_and_context() -> (TranslateAlongZ, Backend) {
        let context = Backend::from_yaml_str(TWO_DIMERS, None, &mut rand::thread_rng()).unwrap();
        let mut mv = TranslateAlongZ {
            molecule_name: "dimer".to_owned(),
            max_displacement: 0.5,
            molecule_id: MoleculeId::new(0),
        };
        mv.finalize(&context).unwrap();
        (mv, context)
    }

    /// A whole-molecule translation is a `RigidBody` change: no internal geometry moved, so the
    /// bonded terms can skip recomputation. The constructor decides this, not the move.
    #[test]
    fn a_group_translation_announces_a_rigid_body_change() {
        let (mut mv, context) = move_and_context();
        let proposed = MoveProposal::propose_move(&mut mv, &context, &mut rand::thread_rng())
            .expect("two dimers are present");
        assert!(matches!(
            proposed.change(),
            Change::SingleGroup(_, GroupChange::RigidBody)
        ));
    }

    #[test]
    fn the_displacement_stays_within_dp_and_along_z() {
        let (mut mv, context) = move_and_context();
        let mut rng = rand::thread_rng();
        for _ in 0..32 {
            let proposed = MoveProposal::propose_move(&mut mv, &context, &mut rng).unwrap();
            let crate::propagate::Displacement::Distance(shift) = proposed.displacement() else {
                panic!("a translation must report a distance");
            };
            assert_eq!(shift.x, 0.0);
            assert_eq!(shift.y, 0.0);
            assert!(shift.z.abs() < 0.5);
        }
    }

    #[test]
    fn an_unknown_molecule_name_fails_at_startup() {
        let (_, context) = move_and_context();
        let mut mv = TranslateAlongZ {
            molecule_name: "trimer".to_owned(),
            max_displacement: 0.5,
            molecule_id: MoleculeId::new(0),
        };
        assert!(mv.finalize(&context).is_err());
    }
}

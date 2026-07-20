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

use crate::group::MoleculeId;
use crate::montecarlo;
use crate::propagate::{tagged_yaml, MoveProposal, ProposedMove};
use crate::transform::random_quaternion;
use crate::ObserveContext;
use rand::RngCore;
use serde::{Deserialize, Serialize};

/// Move for rotating a random molecule.
///
/// This will pick a random molecule of type `molecule_id` and rotate it by a random angle.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotateMolecule {
    /// Name of the molecule type to rotate.
    #[serde(rename = "molecule")]
    molecule_name: String,
    /// Id of the molecule type to rotate.
    #[serde(skip)]
    molecule_id: MoleculeId,
    /// Maximum rotation angle (radians).
    #[serde(alias = "dprot")]
    max_angle: f64,
    /// Move selection weight.
    #[serde(skip_serializing, default = "crate::propagate::default_weight")]
    pub(crate) weight: f64,
    /// Repeat the move N times.
    #[serde(default = "crate::propagate::default_repeat")]
    #[serde(skip_serializing)]
    pub(crate) repeat: usize,
}

impl RotateMolecule {
    /// Validate and finalize the move.
    pub(crate) fn finalize(&mut self, context: &impl ObserveContext) -> anyhow::Result<()> {
        self.molecule_id =
            montecarlo::find_molecule_id(context, &self.molecule_name, "RotateMolecule")?;
        montecarlo::validate_max_angle(self.max_angle, "RotateMolecule")?;
        let topology = context.topology();
        montecarlo::validate_orientable(topology.moleculekind(self.molecule_id), "RotateMolecule")?;
        Ok(())
    }
}

impl<T: ObserveContext> MoveProposal<T> for RotateMolecule {
    fn propose_move(&mut self, context: &T, rng: &mut dyn RngCore) -> Option<ProposedMove> {
        let group_index = montecarlo::random_group(context, rng, self.molecule_id)?;
        let (quaternion, angle) = random_quaternion(rng, self.max_angle);
        Some(ProposedMove::rotate_group(group_index, quaternion, angle))
    }

    fn to_yaml(&self) -> Option<yaml_serde::Value> {
        tagged_yaml("RotateMolecule", self)
    }
}

impl_info!(
    RotateMolecule,
    "rotate_molecule",
    "Rigid body rotation of random molecule"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::montecarlo::chain_fixture::{chain, chain_context, ChainSpec};

    fn rotate(max_angle: f64) -> RotateMolecule {
        yaml_serde::from_str(&format!("{{molecule: Chain, max_angle: {max_angle}}}")).unwrap()
    }

    /// Zero is the dangerous one: every proposal is then the identity, so the move is always
    /// accepted and the run reports 100 % acceptance while sampling nothing.
    #[test]
    fn rejects_unusable_max_angle() {
        let context = chain_context(&chain(4), ChainSpec::default());
        for max_angle in ["0", "-1.0", "4.0", ".nan", ".inf"] {
            let mut move_: RotateMolecule =
                yaml_serde::from_str(&format!("{{molecule: Chain, max_angle: {max_angle}}}"))
                    .unwrap();
            assert!(
                move_.finalize(&context).is_err(),
                "max_angle {max_angle} should be rejected"
            );
        }
        assert!(rotate(0.5).finalize(&context).is_ok());
    }

    /// A single-particle molecule has no orientation to sample; `RotateMolecule` would spin it
    /// about its own mass centre forever, always accepting.
    #[test]
    fn rejects_atomic_molecule() {
        let yaml = "
atoms:
  - {name: A, mass: 1.0, sigma: 1.0, eps: 0.1}
molecules:
  - name: Chain
    atoms: [A]
    atomic: true
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium:
    permittivity: !Vacuum
    temperature: 300.0
  energy:
    - !Nonbonded
        default:
          - !LennardJones {mixing: LB}
  blocks:
    - molecule: Chain
      N: 4
      insert: !RandomAtomPos {}
";
        let context =
            crate::backend::Backend::from_yaml_str(yaml, None, &mut rand::thread_rng()).unwrap();
        assert!(rotate(0.5).finalize(&context).is_err());
    }
}

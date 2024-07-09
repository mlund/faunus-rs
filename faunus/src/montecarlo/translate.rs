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

use super::{Frequency, MoveStatistics};
use crate::group::*;
use crate::transform::{random_unit_vector, Transform};
use crate::{Change, Context, GroupChange};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Move for translating a random molecule
///
/// This will pick a random molecule of type `id` and translate it by a random displacement.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranslateMolecule {
    /// Name of the molecule type to translate.
    #[serde(rename = "molecule")]
    molecule_name: String,
    /// Id of the molecule type to translate.
    #[serde(skip)]
    molecule_id: usize,
    /// Maximum displacement.
    #[serde(alias = "dp")]
    max_displacement: f64,
    /// Move frequency.
    frequency: Frequency,
    /// Move statisticcs.
    #[serde(skip_deserializing)]
    statistics: MoveStatistics,
}

impl TranslateMolecule {
    pub fn new(
        molecule_name: &str,
        molecule_id: usize,
        max_displacement: f64,
        frequency: Frequency,
    ) -> Self {
        Self {
            molecule_name: molecule_name.to_owned(),
            molecule_id,
            max_displacement,
            frequency,
            statistics: MoveStatistics::default(),
        }
    }

    /// Validate and finalize the move.
    pub(crate) fn finalize(&mut self, context: &impl Context) -> anyhow::Result<()> {
        self.molecule_id = context
            .topology()
            .moleculekinds()
            .iter()
            .position(|x| x.name() == &self.molecule_name)
            .ok_or(anyhow::Error::msg(
                "Molecule kind in the definition of 'TranslateMolecule' move does not exist.",
            ))?;

        Ok(())
    }

    /// Pick a random group index with the correct molecule type
    fn random_group(&self, context: &impl Context, rng: &mut ThreadRng) -> Option<usize> {
        let select = GroupSelection::ByMoleculeId(self.molecule_id);
        context.select(&select).iter().copied().choose(rng)
    }
}

#[test]
fn test_finalize() {
    use crate::platform::reference::ReferencePlatform;

    let mut rng = rand::thread_rng();
    let context = ReferencePlatform::new(
        "tests/files/topology_pass.yaml",
        Some("tests/files/structure.xyz"),
        &mut rng,
    )
    .unwrap();

    let mut propagator = TranslateMolecule::new("MOL2", 0, 0.5, super::Frequency::Every(4));

    propagator.finalize(&context).unwrap();

    assert_eq!(propagator.molecule_name, "MOL2");
    assert_eq!(propagator.molecule_id, 1);
    assert_eq!(propagator.max_displacement, 0.5);
    assert!(matches!(propagator.frequency, super::Frequency::Every(4)));
}

impl crate::Info for TranslateMolecule {
    fn short_name(&self) -> Option<&'static str> {
        Some("translate")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Translate a random molecule")
    }
}

impl TranslateMolecule {
    pub(crate) fn do_move(
        &mut self,
        context: &mut impl Context,
        step: &mut usize,
        rng: &mut ThreadRng,
    ) -> Option<Change> {
        *step += 1;

        if let Some(index) = self.random_group(context, rng) {
            let displacement =
                random_unit_vector(rng) * self.max_displacement * 2.0 * (rng.gen::<f64>() - 0.5);
            Transform::Translate(displacement)
                .on_group(index, context)
                .unwrap();
            Some(Change::SingleGroup(index, GroupChange::RigidBody))
        } else {
            None
        }
    }
    pub(crate) fn statistics(&self) -> &MoveStatistics {
        &self.statistics
    }
    pub(crate) fn statistics_mut(&mut self) -> &mut MoveStatistics {
        &mut self.statistics
    }
    pub(crate) fn frequency(&self) -> Frequency {
        self.frequency
    }
}

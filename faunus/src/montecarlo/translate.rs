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

use super::MoveStatistics;
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
    /// Move selection weight.
    weight: f64,
    /// Repeat the move N times.
    #[serde(default = "crate::propagate::default_repeat")]
    repeat: usize,
    /// Move statisticcs.
    #[serde(skip_deserializing)]
    statistics: MoveStatistics,
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
    /// Create a new `TranslateMolecule` move.
    pub fn new(
        molecule_name: &str,
        molecule_id: usize,
        max_displacement: f64,
        weight: f64,
    ) -> Self {
        Self {
            molecule_name: molecule_name.to_owned(),
            molecule_id,
            max_displacement,
            weight,
            repeat: 1,
            statistics: MoveStatistics::default(),
        }
    }

    /// Pick a random group index with the correct molecule type
    fn random_group(&self, context: &impl Context, rng: &mut impl Rng) -> Option<usize> {
        let select = GroupSelection::ByMoleculeId(self.molecule_id);
        context.select(&select).iter().copied().choose(rng)
    }

    /// Propose a translation of a molecule.
    pub(crate) fn propose_move(
        &mut self,
        context: &mut impl Context,
        rng: &mut impl Rng,
    ) -> Option<Change> {
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

    /// Get immutable reference to the statistics of the move.
    pub(crate) fn statistics(&self) -> &MoveStatistics {
        &self.statistics
    }

    /// Get mutable reference to the statistics of the move.
    pub(crate) fn statistics_mut(&mut self) -> &mut MoveStatistics {
        &mut self.statistics
    }

    /// Get weight of the move.
    pub(crate) fn weight(&self) -> f64 {
        self.weight
    }

    /// Number of times the move should be repeated if selected.
    pub(crate) fn repeat(&self) -> usize {
        self.repeat
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
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_parse() {
        let string = "{ molecule: Water, dp: 0.5, weight: 0.7 }";
        let translate: TranslateMolecule = serde_yaml::from_str(string).unwrap();

        assert_eq!(translate.molecule_name, "Water");
        assert_eq!(translate.max_displacement, 0.5);
        assert_eq!(translate.weight, 0.7);
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

        let mut propagator = TranslateMolecule::new("MOL2", 0, 0.5, 4.0);

        propagator.finalize(&context).unwrap();

        assert_eq!(propagator.molecule_name, "MOL2");
        assert_eq!(propagator.molecule_id, 1);
        assert_eq!(propagator.max_displacement, 0.5);
        assert_eq!(propagator.weight, 4.0);
    }
}

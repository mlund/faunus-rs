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

/// Pick a random group index of the specified molecule type.
fn random_group(context: &impl Context, rng: &mut impl Rng, molecule_id: usize) -> Option<usize> {
    let select = GroupSelection::ByMoleculeId(molecule_id);
    context.select(&select).iter().copied().choose(rng)
}

/// Pick a random atom from the specified group.
/// Returns an absolute index of the atom.
fn random_atom(
    context: &impl Context,
    rng: &mut impl Rng,
    group_index: usize,
    atom_id: Option<usize>,
) -> Option<usize> {
    let select = match atom_id {
        Some(a) => ParticleSelection::ById(a),
        None => ParticleSelection::Active,
    };

    context
        .groups()
        .get(group_index)
        .expect("Group should exist.")
        .select(&select, context)
        .expect("Selection should be successful.")
        .iter()
        .copied()
        .choose(rng)
}

/// Move for translating a random molecule.
///
/// This will pick a random molecule of type `molecule_id` and translate it by a random displacement.
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
        Some("trans-mol")
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
        repeat: usize,
    ) -> Self {
        Self {
            molecule_name: molecule_name.to_owned(),
            molecule_id,
            max_displacement,
            weight,
            repeat,
            statistics: MoveStatistics::default(),
        }
    }

    /// Propose a translation of a molecule.
    pub(crate) fn propose_move(
        &mut self,
        context: &mut impl Context,
        rng: &mut impl Rng,
    ) -> Option<Change> {
        if let Some(index) = random_group(context, rng, self.molecule_id) {
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
    pub(crate) fn get_statistics(&self) -> &MoveStatistics {
        &self.statistics
    }

    /// Get mutable reference to the statistics of the move.
    pub(crate) fn get_statistics_mut(&mut self) -> &mut MoveStatistics {
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

/// Move for translating a random atom.
///
/// This will pick a random atom of either
/// a) any type from any molecule (neither atom_name nor molecule_name are specified),
/// a) type `atom_id` from any molecule (only atom_name is specified),
/// b) any type from molecule of type `molecule_id` (only molecule_name is specified), or
/// c) type `atom_id` from molecule of type `molecule_id` (both atom_name and molecule_name are specified)
///
/// and translate it by a random displacement.
///
// TODO! what should be done if a molecule becomes partially deactivated, no longer containing an atom of the specified kind
// currently, the `propose_move` method attempts to select a new atom until it succeeds but that's not ideal
//
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranslateAtom {
    /// Name of the atom type to translate.
    #[serde(rename = "atom")]
    atom_name: Option<String>,
    /// Id of the atom type to translate.
    #[serde(skip)]
    atom_id: Option<usize>,
    /// Name of the molecule type to select the atom from.
    #[serde(rename = "molecule")]
    molecule_name: Option<String>,
    /// Id of the molecule type to select the atom from.
    #[serde(skip)]
    molecule_id: Option<usize>,
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
    /// Molecule types to select from. Only used if `molecule_name` is not provided.
    #[serde(skip)]
    #[serde(default = "default_select_molecule_ids")]
    select_molecule_ids: GroupSelection,
}

// TODO different default option might be better (we want any group that is not empty)
fn default_select_molecule_ids() -> GroupSelection {
    GroupSelection::Size(GroupSize::Full)
}

impl crate::Info for TranslateAtom {
    fn short_name(&self) -> Option<&'static str> {
        Some("trans-atom")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Translate a random atom")
    }
}

impl TranslateAtom {
    /// Create a new `TranslateAtom` move.
    pub fn new(
        molecule_name: Option<&str>,
        molecule_id: Option<usize>,
        atom_name: Option<&str>,
        atom_id: Option<usize>,
        max_displacement: f64,
        weight: f64,
        repeat: usize,
    ) -> Self {
        Self {
            atom_name: atom_name.map(|s| s.to_string()),
            atom_id,
            molecule_name: molecule_name.map(|s| s.to_string()),
            molecule_id,
            max_displacement,
            weight,
            repeat,
            statistics: MoveStatistics::default(),
            select_molecule_ids: GroupSelection::Size(GroupSize::Full),
        }
    }

    /// Returns group id and absolute index of an atom to act on.
    /// If no atom could be selected, returns None.
    fn get_group_atom(
        &self,
        context: &mut impl Context,
        rng: &mut impl Rng,
    ) -> Option<(usize, usize)> {
        let group = match self.molecule_id {
            Some(m) => random_group(context, rng, m)?,
            // if molecule is not specified, we choose from all molecules containing
            // at least one of the requested atoms
            // todo! what if the target atom becomes deactivated?
            None => context
                .select(&self.select_molecule_ids)
                .iter()
                .copied()
                .choose(rng)?,
        };

        Some((group, random_atom(context, rng, group, self.atom_id)?))
    }

    /// Propose a translation of an atom.
    pub(crate) fn propose_move(
        &mut self,
        context: &mut impl Context,
        rng: &mut impl Rng,
    ) -> Option<Change> {
        // repeat until you find a suitable atom
        let (group, absolute_atom) = std::iter::repeat_with(|| self.get_group_atom(context, rng))
            .find_map(|group_atom| group_atom)
            .unwrap();

        let displacement =
            random_unit_vector(rng) * self.max_displacement * 2.0 * (rng.gen::<f64>() - 0.5);

        Transform::PartialTranslate(
            displacement,
            ParticleSelection::AbsIndex(vec![absolute_atom]),
        )
        .on_group(group, context)
        .unwrap();

        // we need to convert to relative index because `GroupChange` expects it
        let relative_atom = context.groups()[group]
            .to_relative_index(absolute_atom)
            .expect("Atom should be part of the group.");

        Some(Change::SingleGroup(
            group,
            GroupChange::PartialUpdate(vec![relative_atom]),
        ))
    }

    /// Get immutable reference to the statistics of the move.
    pub(crate) fn get_statistics(&self) -> &MoveStatistics {
        &self.statistics
    }

    /// Get mutable reference to the statistics of the move.
    pub(crate) fn get_statistics_mut(&mut self) -> &mut MoveStatistics {
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
        if let Some(molecule_name) = &self.molecule_name {
            self.molecule_id = Some(
                context
                    .topology()
                    .moleculekinds()
                    .iter()
                    .position(|x| x.name() == molecule_name)
                    .ok_or(anyhow::Error::msg(
                        "Molecule kind in the definition of 'TranslateAtom' move does not exist.",
                    ))?,
            );
        }

        if let Some(atom_name) = &self.atom_name {
            self.atom_id = Some(
                context
                    .topology()
                    .atomkinds()
                    .iter()
                    .position(|x| x.name() == atom_name)
                    .ok_or(anyhow::Error::msg(
                        "Atom kind in the definition of 'TranslateAtom' move does not exist.",
                    ))?,
            );
        }

        match (self.atom_id, self.molecule_id) {
            // check that the atom kind exists inside the molecule
            (Some(a), Some(m)) => {
                if !context
                    .topology()
                    .moleculekinds()
                    .get(m)
                    .expect("Molecule kind should exist.")
                    .atom_indices()
                    .contains(&a)
                {
                    anyhow::bail!("Atom kind in the definition of 'TranslateAtom' move does not exist in the specified molecule kind.");
                }
            }
            (Some(a), None) => {
                // get molecule kinds containing the requested atom kind
                let molecule_indices = context
                    .topology_ref()
                    .moleculekinds()
                    .iter()
                    .filter_map(|mol| {
                        if mol.atom_indices().contains(&a) {
                            Some(mol.id())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<usize>>();

                self.select_molecule_ids = GroupSelection::ByMoleculeIds(molecule_indices);
            }
            _ => (),
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::platform::reference::ReferencePlatform;

    #[test]
    fn test_translate_molecule_parse() {
        let string = "{ molecule: Water, dp: 0.5, weight: 0.7 }";
        let translate: TranslateMolecule = serde_yaml::from_str(string).unwrap();

        assert_eq!(translate.molecule_name, "Water");
        assert_eq!(translate.max_displacement, 0.5);
        assert_eq!(translate.weight, 0.7);
    }

    #[test]
    fn test_translate_atom_parse() {
        let string = "{ atom: O, dp: 0.1, weight: 1.0, repeat: 4}";
        let translate: TranslateAtom = serde_yaml::from_str(string).unwrap();

        assert_eq!(translate.molecule_name, None);
        assert_eq!(translate.atom_name.unwrap(), "O");
        assert_eq!(translate.max_displacement, 0.1);
        assert_eq!(translate.weight, 1.0);
        assert_eq!(translate.repeat, 4);
    }

    #[test]
    fn test_translate_molecule_finalize() {
        let mut rng = rand::thread_rng();
        let context = ReferencePlatform::new(
            "tests/files/topology_pass.yaml",
            Some("tests/files/structure.xyz"),
            &mut rng,
        )
        .unwrap();

        let mut propagator = TranslateMolecule::new("MOL2", 0, 0.5, 4.0, 1);

        propagator.finalize(&context).unwrap();

        assert_eq!(propagator.molecule_name, "MOL2");
        assert_eq!(propagator.molecule_id, 1);
        assert_eq!(propagator.max_displacement, 0.5);
        assert_eq!(propagator.weight, 4.0);
        assert_eq!(propagator.repeat, 1);
    }

    #[test]
    fn test_translate_atom_finalize() {
        let mut rng = rand::thread_rng();
        let context = ReferencePlatform::new(
            "tests/files/topology_pass.yaml",
            Some("tests/files/structure.xyz"),
            &mut rng,
        )
        .unwrap();

        let mut propagator = TranslateAtom::new(None, None, Some("X"), None, 0.5, 4.0, 1);

        propagator.finalize(&context).unwrap();
        assert_eq!(propagator.molecule_name, None);
        assert_eq!(propagator.molecule_id, None);
        assert_eq!(propagator.atom_name, Some(String::from("X")));
        assert_eq!(propagator.atom_id, Some(2));
        assert_eq!(propagator.max_displacement, 0.5);
        assert_eq!(propagator.weight, 4.0);
        assert_eq!(propagator.repeat, 1);
    }

    #[test]
    fn test_translate_atom_selections() {
        let mut rng = rand::thread_rng();
        let mut context = ReferencePlatform::new(
            "tests/files/topology_pass.yaml",
            Some("tests/files/structure.xyz"),
            &mut rng,
        )
        .unwrap();

        let mut seedable = rand::rngs::StdRng::seed_from_u64(12345);

        // anything can move
        let mut move1 = TranslateAtom::new(None, None, None, None, 0.1, 1.0, 1);
        move1.finalize(&context).unwrap();

        let expected_groups = [25, 30, 29, 66, 27, 31, 64, 32, 23, 12];
        let expected_indices = [2, 2, 1, 2, 1, 2, 1, 1, 1, 2];

        for i in 0..10 {
            let change = move1.propose_move(&mut context, &mut seedable).unwrap();
            match change {
                Change::SingleGroup(group, group_change) => {
                    assert_eq!(group, expected_groups[i]);
                    match group_change {
                        GroupChange::PartialUpdate(x) => assert_eq!(x[0], expected_indices[i]),
                        _ => panic!("Invalid Group Change."),
                    }
                }
                _ => panic!("Invalid Change."),
            };
        }

        // only MOL can move
        let mut move2 = TranslateAtom::new(Some("MOL"), None, None, None, 0.1, 1.0, 1);
        move2.finalize(&context).unwrap();

        let expected_groups = [61, 60, 0, 61, 1, 0, 0, 61, 2, 1];
        let expected_indices = [2, 4, 1, 2, 3, 2, 0, 2, 5, 2];

        for i in 0..10 {
            let change = move2.propose_move(&mut context, &mut seedable).unwrap();
            match change {
                Change::SingleGroup(group, group_change) => {
                    assert_eq!(group, expected_groups[i]);
                    match group_change {
                        GroupChange::PartialUpdate(x) => assert_eq!(x[0], expected_indices[i]),
                        _ => panic!("Invalid Group Change."),
                    }
                }
                _ => panic!("Invalid Change."),
            };
        }

        // only MOL and OW can move
        let mut move3 = TranslateAtom::new(Some("MOL"), None, Some("OW"), None, 0.1, 1.0, 1);
        move3.finalize(&context).unwrap();

        let expected_groups = [60, 61, 0, 60, 0, 61, 61, 2, 1, 2];
        let expected_indices = [0, 5, 4, 5, 0, 0, 5, 5, 4, 5];

        for i in 0..10 {
            let change = move3.propose_move(&mut context, &mut seedable).unwrap();
            match change {
                Change::SingleGroup(group, group_change) => {
                    assert_eq!(group, expected_groups[i]);
                    match group_change {
                        GroupChange::PartialUpdate(x) => assert_eq!(x[0], expected_indices[i]),
                        _ => panic!("Invalid Group Change."),
                    }
                }
                _ => panic!("Invalid Change."),
            };
        }

        // only HW can move
        let mut move4 = TranslateAtom::new(None, None, Some("HW"), None, 0.1, 1.0, 1);
        move4.finalize(&context).unwrap();

        assert_eq!(
            move4.select_molecule_ids,
            GroupSelection::ByMoleculeIds(vec![0])
        );

        let expected_groups = [1, 1, 60, 61, 2, 60, 2, 2, 2, 60];
        let expected_indices = [3, 3, 1, 3, 2, 2, 3, 2, 2, 3];
        for i in 0..10 {
            let change = move4.propose_move(&mut context, &mut seedable).unwrap();
            match change {
                Change::SingleGroup(group, group_change) => {
                    assert_eq!(group, expected_groups[i]);
                    match group_change {
                        GroupChange::PartialUpdate(x) => assert_eq!(x[0], expected_indices[i]),
                        _ => panic!("Invalid Group Change."),
                    }
                }
                _ => panic!("Invalid Change."),
            };
        }
    }
}

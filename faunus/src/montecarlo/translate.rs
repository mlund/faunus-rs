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

use super::preferential::PreferentialSampling;
use super::{find_molecule_id, random_atom, random_group, Bias};
use crate::group::*;
use crate::montecarlo::NewOld;
use crate::propagate::{tagged_yaml, MoveProposal, ProposedMove};
use crate::transform::{random_displacement, random_unit_vector};
use crate::Change;
use crate::ObserveContext;
use crate::Point;

use rand::prelude::*;
use serde::{Deserialize, Serialize};

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
    molecule_id: MoleculeId,
    /// Maximum displacement.
    #[serde(alias = "dp")]
    max_displacement: f64,
    /// Move selection weight.
    #[serde(skip_serializing, default = "crate::propagate::default_weight")]
    pub(crate) weight: f64,
    /// Repeat the move N times.
    #[serde(default = "crate::propagate::default_repeat")]
    #[serde(skip_serializing)]
    pub(crate) repeat: usize,
    /// Move directions
    #[serde(default)]
    directions: crate::axes::Axes,
}

impl_info!(
    TranslateMolecule,
    "trans_mol",
    "Translate a random molecule"
);

impl TranslateMolecule {
    /// Create a new `TranslateMolecule` move.
    #[allow(dead_code)] // constructed by tests
    pub fn new(
        molecule_name: &str,
        molecule_id: MoleculeId,
        max_displacement: f64,
        weight: f64,
        directions: crate::axes::Axes,
        repeat: usize,
    ) -> Self {
        Self {
            molecule_name: molecule_name.to_owned(),
            molecule_id,
            max_displacement,
            weight,
            repeat,
            directions,
        }
    }

    /// Validate and finalize the move.
    pub(crate) fn finalize(&mut self, context: &impl ObserveContext) -> anyhow::Result<()> {
        self.molecule_id = find_molecule_id(context, &self.molecule_name, "TranslateMolecule")?;
        if context
            .topology_ref()
            .moleculekind(self.molecule_id)
            .atomic()
        {
            anyhow::bail!(
                "TranslateMolecule cannot be used with atomic molecule '{}'; use TranslateAtom instead",
                self.molecule_name
            );
        }
        Ok(())
    }
}

impl<T: ObserveContext> MoveProposal<T> for TranslateMolecule {
    fn propose_move(&mut self, context: &T, rng: &mut dyn RngCore) -> Option<ProposedMove> {
        let group_index = random_group(context, rng, self.molecule_id)?;
        let displacement = self
            .directions
            .project(random_unit_vector(rng) * random_displacement(rng, self.max_displacement));
        Some(ProposedMove::translate_group(group_index, displacement))
    }

    fn to_yaml(&self) -> Option<serde_yml::Value> {
        tagged_yaml("TranslateMolecule", self)
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
    atom_id: Option<AtomKindId>,
    /// Name of the molecule type to select the atom from.
    #[serde(rename = "molecule")]
    molecule_name: Option<String>,
    /// Id of the molecule type to select the atom from.
    #[serde(skip)]
    molecule_id: Option<MoleculeId>,
    /// Maximum displacement.
    #[serde(alias = "dp")]
    max_displacement: f64,
    /// Move selection weight.
    #[serde(skip_serializing, default = "crate::propagate::default_weight")]
    pub(crate) weight: f64,
    /// Repeat the move N times.
    #[serde(default = "crate::propagate::default_repeat")]
    #[serde(skip_serializing)]
    pub(crate) repeat: usize,
    /// Molecule types to select from. Only used if `molecule_name` is not provided.
    #[serde(skip)]
    #[serde(default = "default_select_molecule_ids")]
    select_molecule_ids: GroupSelection,
    /// Optional preferential sampling bias toward a reference molecule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    preferential: Option<PreferentialSampling>,
}

// TODO different default option might be better (we want any group that is not empty)
const fn default_select_molecule_ids() -> GroupSelection {
    GroupSelection::Size(GroupSize::Full)
}

impl_info!(TranslateAtom, "trans_atom", "Translate a random atom");

impl TranslateAtom {
    /// Create a new `TranslateAtom` move.
    #[allow(dead_code)] // constructed by tests
    pub fn new(
        molecule_name: Option<&str>,
        molecule_id: Option<MoleculeId>,
        atom_name: Option<&str>,
        atom_id: Option<AtomKindId>,
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
            select_molecule_ids: GroupSelection::Size(GroupSize::Full),
            preferential: None,
        }
    }

    /// Pick a random group index matching the molecule/selection filter.
    fn pick_group(
        &self,
        context: &impl ObserveContext,
        rng: &mut (impl Rng + ?Sized),
    ) -> Option<usize> {
        match self.molecule_id {
            Some(m) => random_group(context, rng, m),
            None => context
                .select(&self.select_molecule_ids)
                .iter()
                .copied()
                .choose(rng),
        }
    }

    /// A random displacement of at most `max_displacement`, isotropic in direction.
    fn trial_displacement(&self, rng: &mut dyn RngCore) -> Point {
        random_unit_vector(rng) * random_displacement(rng, self.max_displacement)
    }

    /// The groups this move may draw from.
    ///
    /// Preferential sampling weighs candidates against one another, so its candidate set has to
    /// span every eligible group: the normalized weight that selects an atom (Allen & Tildesley
    /// eqn 9.43) is defined relative to all the atoms it competes with. Narrowing to one randomly
    /// chosen group first would leave the solvent molecules near the solute competing only with
    /// their own atoms — no bias toward the solute at all, and for a one-atom set, none possible.
    fn eligible_groups(&self) -> GroupSelection {
        match self.molecule_id {
            Some(molecule) => GroupSelection::ByMoleculeId(molecule),
            None => self.select_molecule_ids.clone(),
        }
    }

    /// Returns group id and absolute index of a uniformly chosen atom.
    fn get_group_atom(
        &self,
        context: &impl ObserveContext,
        rng: &mut (impl Rng + ?Sized),
    ) -> Option<(usize, usize)> {
        let group = self.pick_group(context, rng)?;
        Some((group, random_atom(context, rng, group, self.atom_id)?))
    }

    /// Validate and finalize the move.
    pub(crate) fn finalize(&mut self, context: &impl ObserveContext) -> anyhow::Result<()> {
        if let Some(molecule_name) = &self.molecule_name {
            self.molecule_id = Some(
                context
                    .topology()
                    .moleculekinds()
                    .iter()
                    .position(|x| x.name() == molecule_name)
                    .map(MoleculeId::new)
                    .ok_or_else(|| {
                        anyhow::Error::msg(
                            "Molecule kind in the definition of 'TranslateAtom' move does not exist.",
                        )
                    })?,
            );
        }

        if let Some(atom_name) = &self.atom_name {
            self.atom_id = Some(
                context
                    .topology()
                    .atomkinds()
                    .iter()
                    .position(|x| x.name() == atom_name)
                    .map(AtomKindId::new)
                    .ok_or_else(|| {
                        anyhow::Error::msg(
                            "Atom kind in the definition of 'TranslateAtom' move does not exist.",
                        )
                    })?,
            );
        }

        match (self.atom_id, self.molecule_id) {
            // check that the atom kind exists inside the molecule
            (Some(a), Some(m)) => {
                if !context
                    .topology()
                    .moleculekinds()
                    .get(m.get())
                    .expect("Molecule kind should exist.")
                    .atom_indices()
                    .contains(&a.get())
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
                        if mol.atom_indices().contains(&a.get()) {
                            Some(MoleculeId::new(mol.id()))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<MoleculeId>>();

                self.select_molecule_ids = GroupSelection::ByMoleculeIds(molecule_indices);
            }
            _ => (),
        }

        // The sampler takes the move's own filter and derives its candidates from it, so the two
        // cannot drift apart and the reference is checked against the set actually drawn from.
        let groups = self.eligible_groups();
        let atom = self.atom_id;
        if let Some(preferential) = self.preferential.as_mut() {
            preferential.finalize(context, groups, atom)?;
        }

        Ok(())
    }
}

impl<T: ObserveContext> MoveProposal<T> for TranslateAtom {
    #[allow(clippy::unnecessary_unwrap)] // split borrow: eligible_atoms borrows self, then pref
    fn propose_move(&mut self, context: &T, rng: &mut dyn RngCore) -> Option<ProposedMove> {
        let (group, absolute_atom, displacement) = if self.preferential.is_some() {
            // Drawn before the pick: the acceptance correction of eqn 9.44 needs both endpoints
            // of the move, and the displacement does not depend on which atom is chosen.
            let displacement = self.trial_displacement(rng);
            let pref = self.preferential.as_mut().unwrap();
            let atom = pref.propose(context, &displacement, rng)?;
            // The atom is picked from the whole eligible set, so its group follows from it.
            let group = context
                .group_of_particle(atom)
                .expect("a selected atom belongs to the group it was selected from");
            (group, atom, displacement)
        } else {
            // Bounded retries: GCMC may leave the eligible groups empty
            // (count fluctuates to zero). Spinning forever in that case
            // hangs the run; reporting `None` lets the runner count this
            // attempt as rejected and continue. 16 is enough to survive
            // multi-group selection where the first few picks miss; an
            // empty system fails fast.
            const MAX_RETRIES: usize = 16;
            let (group, atom) = (0..MAX_RETRIES).find_map(|_| self.get_group_atom(context, rng))?;
            (group, atom, self.trial_displacement(rng))
        };

        let absolute_atom = AbsIndex::new(absolute_atom);
        let relative_atom = context.groups()[group]
            .to_relative(absolute_atom)
            .expect("Atom should be part of the group.");

        Some(ProposedMove::translate_atoms(
            group,
            vec![relative_atom],
            displacement,
        ))
    }

    fn bias(&self, _change: &Change, _energies: &NewOld<f64>) -> Bias {
        match &self.preferential {
            Some(pref) => Bias::Dimensionless(pref.ln_bias()),
            None => Bias::None,
        }
    }

    fn revalidate(&mut self, context: &T) -> anyhow::Result<()> {
        match self.preferential.as_mut() {
            Some(preferential) => preferential.revalidate(context),
            None => Ok(()),
        }
    }

    fn on_trial_outcome(&mut self, context: &T, accepted: bool) {
        if let Some(preferential) = self.preferential.as_mut() {
            preferential.on_trial_outcome(context, accepted);
        }
    }

    fn to_yaml(&self) -> Option<serde_yml::Value> {
        tagged_yaml("TranslateAtom", self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Change, GroupChange};

    use std::path::Path;

    use super::*;
    use crate::backend::Backend;
    use crate::context::WithTopology;

    /// Preferential candidates must span every matching group, not one picked at random.
    ///
    /// Two Solvent molecules, one 3 Å from the solute and one 28 Å away. Weighing their atoms
    /// against each other puts the near molecule at ~98% of selections, since W'(3) : W'(28) is
    /// (3+1)⁻² : (28+1)⁻² ≈ 53 : 1. Choosing a group uniformly first and only weighting inside it
    /// caps the near molecule at ~50% — and weighs one molecule's atoms with the other's weights.
    #[test]
    fn preferential_candidates_span_all_matching_groups() {
        use rand::SeedableRng;

        const PROPOSALS: usize = 2_000;

        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let context =
            Backend::new("tests/files/preferential_two_groups.yaml", None, &mut rng).unwrap();

        // Positions are pinned in the input: group 0 is the solute, 1 the near Solvent, 2 the far.
        let distance_to_solute = |group: usize| {
            context.groups()[group]
                .mass_center()
                .unwrap()
                .metric_distance(context.groups()[0].mass_center().unwrap())
        };
        assert!(distance_to_solute(1) < 5.0, "group 1 is the near Solvent");
        assert!(distance_to_solute(2) > 25.0, "group 2 is the far Solvent");

        let mut translate: TranslateAtom = serde_yml::from_str(
            "{molecule: Solvent, max_displacement: 0.5, preferential: \
             {reference: \"molecule Macromolecule\", exponent: 2, offset: 1.0}}",
        )
        .unwrap();
        translate.finalize(&context).unwrap();

        let near_picks = (0..PROPOSALS)
            .filter(|_| {
                let proposed = translate.propose_move(&context, &mut rng).unwrap();
                let Change::SingleGroup(group, _) = proposed.change() else {
                    panic!("TranslateAtom proposes a single-group change");
                };
                *group == 1
            })
            .count();

        let fraction = near_picks as f64 / PROPOSALS as f64;
        assert!(
            fraction > 0.9,
            "the near molecule took only {:.0}% of selections; candidates are not being \
             weighed across both groups",
            100.0 * fraction
        );
    }

    /// A move may not displace its own reference.
    ///
    /// Moving a reference atom shifts the sphere the distances are measured from, so every other
    /// candidate's weight changes at once — while the acceptance correction assumes only the moved
    /// atom's did. The result would be a silently non-Boltzmann chain, so the input is refused.
    /// An unrestricted `TranslateAtom` is the dangerous case: with neither `molecule` nor `atom`
    /// set, its candidates are every atom in the system, the reference among them.
    #[test]
    fn a_move_may_not_displace_its_own_reference() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let context =
            Backend::new("tests/files/preferential_two_groups.yaml", None, &mut rng).unwrap();

        for yaml in [
            // Unrestricted: candidates default to every group, including the reference.
            "{max_displacement: 0.5, preferential: \
             {reference: \"molecule Macromolecule\", exponent: 2, offset: 1.0}}",
            // Explicitly pointed at itself.
            "{molecule: Macromolecule, max_displacement: 0.5, preferential: \
             {reference: \"molecule Macromolecule\", exponent: 2, offset: 1.0}}",
        ] {
            let mut translate: TranslateAtom = serde_yml::from_str(yaml).unwrap();
            let error = translate
                .finalize(&context)
                .expect_err("a move overlapping its own reference must be refused");
            assert!(
                error.to_string().contains("distinct from the atoms"),
                "unexpected error: {error}"
            );
        }
    }

    /// The weight exponent must bias selection *toward* the reference.
    #[test]
    fn a_non_positive_exponent_is_refused() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let context =
            Backend::new("tests/files/preferential_two_groups.yaml", None, &mut rng).unwrap();

        // nu <= 0 makes W'(r) grow with r, biasing selection toward the atoms furthest away.
        let mut translate: TranslateAtom = serde_yml::from_str(
            "{molecule: Solvent, max_displacement: 0.5, preferential: \
             {reference: \"molecule Macromolecule\", exponent: -2, offset: 1.0}}",
        )
        .unwrap();
        let error = translate
            .finalize(&context)
            .expect_err("a negative exponent must be refused");
        assert!(
            error
                .to_string()
                .contains("exponent must be finite and positive"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_translate_molecule_parse() {
        let string = "{ molecule: Water, max_displacement: 0.5, weight: 0.7 }";
        let translate: TranslateMolecule = serde_yml::from_str(string).unwrap();

        assert_eq!(translate.molecule_name, "Water");
        assert_eq!(translate.max_displacement, 0.5);
        assert_eq!(translate.weight, 0.7);
    }

    #[test]
    fn test_translate_atom_parse() {
        let string = "{ atom: O, max_displacement: 0.1, weight: 1.0, repeat: 4}";
        let translate: TranslateAtom = serde_yml::from_str(string).unwrap();

        assert_eq!(translate.molecule_name, None);
        assert_eq!(translate.atom_name.unwrap(), "O");
        assert_eq!(translate.max_displacement, 0.1);
        assert_eq!(translate.weight, 1.0);
        assert_eq!(translate.repeat, 4);
    }

    #[test]
    fn test_translate_molecule_finalize() {
        let mut rng = rand::thread_rng();
        let context = Backend::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap();

        let mut propagator = TranslateMolecule::new(
            "MOL2",
            MoleculeId::new(0),
            0.5,
            4.0,
            crate::axes::Axes::XYZ,
            1,
        );

        propagator.finalize(&context).unwrap();

        assert_eq!(propagator.molecule_name, "MOL2");
        assert_eq!(propagator.molecule_id, MoleculeId::new(1));
        assert_eq!(propagator.max_displacement, 0.5);
        assert_eq!(propagator.weight, 4.0);
        assert_eq!(propagator.repeat, 1);
    }

    #[test]
    fn test_translate_atom_finalize() {
        let mut rng = rand::thread_rng();
        let context = Backend::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap();

        let mut propagator = TranslateAtom::new(None, None, Some("X"), None, 0.5, 4.0, 1);

        propagator.finalize(&context).unwrap();
        assert_eq!(propagator.molecule_name, None);
        assert_eq!(propagator.molecule_id, None);
        assert_eq!(propagator.atom_name, Some(String::from("X")));
        assert_eq!(propagator.atom_id, Some(AtomKindId::new(2)));
        assert_eq!(propagator.max_displacement, 0.5);
        assert_eq!(propagator.weight, 4.0);
        assert_eq!(propagator.repeat, 1);
    }

    /// A `molecule`/`atom` filter must be honoured by every proposal it selects, not just the
    /// literal sequence one seed happens to draw — pinning the latter would break on any
    /// unrelated change to the selection algorithm while saying nothing about the filter itself.
    #[test]
    fn atom_selection_respects_molecule_and_atom_filters() {
        let mut rng = rand::thread_rng();
        let context = Backend::new(
            "tests/files/topology_pass.yaml",
            Some(Path::new("tests/files/structure.xyz")),
            &mut rng,
        )
        .unwrap();

        let molecule_of = |group: usize| -> String {
            let molecule_id = context.groups()[group].molecule();
            context
                .topology_ref()
                .moleculekind(molecule_id)
                .name()
                .clone()
        };
        let atom_kind_of = |group: usize, index: RelIndex| -> String {
            let absolute = context.groups()[group].to_absolute(index).unwrap();
            let kind = context.atom_kind(absolute.get());
            context.topology_ref().atomkinds()[kind.get()]
                .name()
                .to_string()
        };

        let mut seedable = rand::rngs::StdRng::seed_from_u64(12345);
        let mut assert_filters_hold =
            |move_: &mut TranslateAtom, allowed_molecules: &[&str], allowed_atoms: &[&str]| {
                for _ in 0..10 {
                    let change = move_
                        .propose_move(&context, &mut seedable)
                        .unwrap()
                        .change()
                        .clone();
                    let Change::SingleGroup(group, GroupChange::PartialUpdate(indices)) = change
                    else {
                        panic!("expected a single-group partial update");
                    };
                    if !allowed_molecules.is_empty() {
                        assert!(allowed_molecules.contains(&molecule_of(group).as_str()));
                    }
                    if !allowed_atoms.is_empty() {
                        for &index in &indices {
                            assert!(allowed_atoms.contains(&atom_kind_of(group, index).as_str()));
                        }
                    }
                }
            };

        // Unrestricted: any molecule and atom kind may be drawn.
        let mut move1 = TranslateAtom::new(None, None, None, None, 0.1, 1.0, 1);
        move1.finalize(&context).unwrap();
        assert_filters_hold(&mut move1, &[], &[]);

        // Only MOL may move.
        let mut move2 = TranslateAtom::new(Some("MOL"), None, None, None, 0.1, 1.0, 1);
        move2.finalize(&context).unwrap();
        assert_filters_hold(&mut move2, &["MOL"], &[]);

        // Only MOL, and only its OW atoms.
        let mut move3 = TranslateAtom::new(Some("MOL"), None, Some("OW"), None, 0.1, 1.0, 1);
        move3.finalize(&context).unwrap();
        assert_filters_hold(&mut move3, &["MOL"], &["OW"]);

        // Only HW atoms, drawn from whichever molecule kind carries them.
        let mut move4 = TranslateAtom::new(None, None, Some("HW"), None, 0.1, 1.0, 1);
        move4.finalize(&context).unwrap();
        assert_eq!(
            move4.select_molecule_ids,
            GroupSelection::ByMoleculeIds(vec![MoleculeId::new(0)])
        );
        assert_filters_hold(&mut move4, &[], &["HW"]);
    }
}

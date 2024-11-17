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

//! # Implementation of the deserialization of the hamiltonian.

use std::{
    collections::HashMap,
    fmt::{self, Debug},
    marker::PhantomData,
    path::Path,
};

use crate::topology::AtomKind;
use anyhow::Context as AnyhowContext;
use coulomb::pairwise::MultipoleEnergy;
use interatomic::{
    twobody::{
        AshbaughHatch, HardSphere, IonIon, IsotropicTwobodyEnergy, LennardJones, NoInteraction,
        WeeksChandlerAndersen,
    },
    CombinationRule,
};
use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use unordered_pair::UnorderedPair;

/// Structure storing information about the nonbonded interactions in the system in serializable format.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub(crate) struct NonbondedBuilder(
    #[serde(with = "::serde_with::rust::maps_duplicate_key_is_error")]
    // defining interactions between the same atom kinds multiple times causes an error
    HashMap<DefaultOrPair, Vec<NonbondedInteraction>>,
);

impl NonbondedBuilder {
    /// Get interactions for a specific pair of atoms.
    /// If this pair of atoms is not defined, get interactions for Default.
    /// If Default is not defined, return an empty array.
    fn collect_interactions(&self, atom1: &str, atom2: &str) -> &[NonbondedInteraction] {
        let key = DefaultOrPair::Pair(UnorderedPair(atom1.to_owned(), atom2.to_owned()));

        match self.0.get(&key) {
            Some(x) => x,
            None => match self.0.get(&DefaultOrPair::Default) {
                Some(x) => x,
                None => &[],
            },
        }
    }

    /// Get interactions for a specific pair of atoms and collect them into a single `IsotropicTwobodyEnergy` trait object.
    /// If this pair of atoms has no explicitly defined interactions, get interactions for Default.
    /// If Default is not defined or no interactions have been found, return `NoInteraction` structure and log a warning.
    pub(crate) fn get_interaction(
        &self,
        atom1: &AtomKind,
        atom2: &AtomKind,
    ) -> anyhow::Result<Box<dyn IsotropicTwobodyEnergy>> {
        let interactions = self.collect_interactions(atom1.name(), atom2.name());

        if interactions.is_empty() {
            log::warn!(
                "No nonbonded interaction defined for '{} <-> {}'.",
                atom1.name(),
                atom2.name()
            );
            return Ok(Box::from(NoInteraction::default()));
        }

        let mut iterator = interactions.iter();
        let mut total_interaction = loop {
            // find the first existing interaction and use it to initialize the `total_interaction`
            if let Some(interaction) = iterator.next() {
                if let Some(converted) = interaction.to_dyn_energy(atom1, atom2)? {
                    break converted;
                }
            } else {
                // no interactions left
                log::warn!(
                    "No nonbonded interaction defined for '{} <-> {}'.",
                    atom1.name(),
                    atom2.name()
                );
                return Ok(Box::from(NoInteraction::default()));
            }
        };

        // loop through the rest of the interactions and sum them together
        for interaction in iterator {
            if let Some(converted) = interaction.to_dyn_energy(atom1, atom2)? {
                total_interaction = total_interaction + converted;
            }
        }

        Ok(total_interaction)
    }
}

/// Structure used for (de)serializing the Hamiltonian of the system.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub(crate) struct HamiltonianBuilder {
    /// Nonbonded interactions defined for the system.
    pub nonbonded: NonbondedBuilder,
}

/// Specifies pair of atom kinds interacting with each other.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DefaultOrPair {
    /// All pairs of atom kinds.
    Default,
    /// Pair of atom kinds.
    Pair(UnorderedPair<String>),
}

impl Serialize for DefaultOrPair {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match *self {
            DefaultOrPair::Default => serializer.serialize_str("default"),
            DefaultOrPair::Pair(ref pair) => pair.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for DefaultOrPair {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct DefaultOrPairVisitor;

        impl<'de> Visitor<'de> for DefaultOrPairVisitor {
            type Value = DefaultOrPair;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("\"default\" or a pair of atom kinds")
            }

            // parse default as string
            fn visit_str<E>(self, value: &str) -> Result<DefaultOrPair, E>
            where
                E: de::Error,
            {
                if value == "default" {
                    Ok(DefaultOrPair::Default)
                } else {
                    Err(E::invalid_value(de::Unexpected::Str(value), &self))
                }
            }

            // parse pair of atom kinds
            fn visit_seq<A>(self, seq: A) -> Result<DefaultOrPair, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let pair =
                    UnorderedPair::deserialize(serde::de::value::SeqAccessDeserializer::new(seq))?;
                Ok(DefaultOrPair::Pair(pair))
            }
        }

        deserializer.deserialize_any(DefaultOrPairVisitor)
    }
}

/// Type of nonbonded interaction.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) enum NonbondedInteraction {
    /// Ashbaugh-Hatch potential.
    AshbaughHatch(DirectOrMixing<AshbaughHatch>),
    /// Lennard-Jones potential.
    LennardJones(DirectOrMixing<LennardJones>),
    /// Weeks-Chandler-Andersen potential.
    #[serde(alias = "WCA")]
    WeeksChandlerAndersen(DirectOrMixing<WeeksChandlerAndersen>),
    /// Hard sphere potential.
    HardSphere(DirectOrMixing<HardSphere>),
    /// Truncated Ewald potential.
    CoulombEwald(coulomb::pairwise::EwaldTruncated),
    /// Real-space Ewald potential.
    CoulombRealSpaceEwald(coulomb::pairwise::RealSpaceEwald),
    /// Plain coulombic potential.
    CoulombPlain(coulomb::pairwise::Plain),
    /// Reaction field.
    CoulombReactionField(coulomb::pairwise::ReactionField),
}

impl NonbondedInteraction {
    /// Converts a `NonbondedInteraction` structure to `IsotropicTwobodyEnergy` trait object.
    ///
    /// ## Notes
    /// - A mixing rule is applied, if needed.
    /// - Returns `None` for coulombic interactions with uncharged particles.
    fn to_dyn_energy(
        &self,
        atom1: &AtomKind,
        atom2: &AtomKind,
    ) -> anyhow::Result<Option<Box<dyn IsotropicTwobodyEnergy>>> {
        let charges = Some((atom1.charge(), atom2.charge()));
        let epsilons = match (atom1.epsilon(), atom2.epsilon()) {
            (Some(x), Some(y)) => Some((x, y)),
            _ => None,
        };
        let sigmas = match (atom1.sigma(), atom2.sigma()) {
            (Some(x), Some(y)) => Some((x, y)),
            _ => None,
        };
        let _lambdas = match (atom1.lambda(), atom2.lambda()) {
            (Some(x), Some(y)) => Some((x, y)),
            _ => None,
        };

        match self {
            Self::LennardJones(x) => Ok(Some(x.convert_and_mix_sigma_epsilon(
                epsilons,
                sigmas,
                LennardJones::from_combination_rule,
            )?)),
            Self::WeeksChandlerAndersen(x) => Ok(Some(x.convert_and_mix_sigma_epsilon(
                epsilons,
                sigmas,
                WeeksChandlerAndersen::from_combination_rule,
            )?)),
            Self::HardSphere(x) => Ok(Some(
                x.convert_and_mix_sigma(sigmas, HardSphere::from_combination_rule)?,
            )),
            Self::CoulombPlain(x) => NonbondedInteraction::convert_coulomb(charges, x.clone()),
            Self::CoulombEwald(x) => NonbondedInteraction::convert_coulomb(charges, x.clone()),
            Self::CoulombRealSpaceEwald(x) => {
                NonbondedInteraction::convert_coulomb(charges, x.clone())
            }
            Self::CoulombReactionField(x) => {
                NonbondedInteraction::convert_coulomb(charges, x.clone())
            }
            _ => anyhow::bail!("Unsupported nonbonded interaction."),
        }
    }

    /// Convert coulombic interaction to `IonIon` interaction.
    ///
    /// ## Notes
    /// - If any of the charges is `0.0`, returns None.
    fn convert_coulomb<T: MultipoleEnergy + Debug + Clone + PartialEq + 'static>(
        charges: Option<(f64, f64)>,
        scheme: T,
    ) -> anyhow::Result<Option<Box<dyn IsotropicTwobodyEnergy>>> {
        let charges = charges
            .context("Charges were not provided but are required for the coulombic interaction.")?;

        if (charges.0 * charges.1).abs() <= std::f64::EPSILON {
            // disable interaction between pairs of particles where at least one particle is uncharged
            Ok(None)
        } else {
            Ok(Some(Box::new(IonIon::new(charges.0 * charges.1, scheme))
                as Box<dyn IsotropicTwobodyEnergy>))
        }
    }
}

/// Specifies whether the parameters for the interaction are
/// directly provided or should be calculated using a combination rule.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub(crate) enum DirectOrMixing<T: IsotropicTwobodyEnergy> {
    /// Calculate the parameters using the provided combination rule.
    Mixing {
        /// Combination rule to use for mixing.
        mixing: CombinationRule,
        #[serde(skip)]
        /// Marker specifying the interaction type.
        _phantom: PhantomData<T>,
    },
    /// The parameters for the interaction are specifically provided.
    Direct(T),
}

impl<T> DirectOrMixing<T>
where
    T: IsotropicTwobodyEnergy + Clone + 'static,
{
    /// Converts `DirectOrMixing` enum to appropriate `IsotropicTwobodyEnergy` trait object
    /// for which `from_combination_rule` function exists.
    ///
    /// Used for mixing `sigmas` and `epsilons` (e.g. LJ, WCA potentials).
    ///
    /// In case the parameters of the potential are directly provided, no mixing is performed.
    fn convert_and_mix_sigma_epsilon(
        &self,
        epsilons: Option<(f64, f64)>,
        sigmas: Option<(f64, f64)>,
        from_combination_rule: impl Fn(CombinationRule, (f64, f64), (f64, f64)) -> T,
    ) -> anyhow::Result<Box<dyn IsotropicTwobodyEnergy>> {
        match self {
            DirectOrMixing::Direct(inner) => Ok(Box::new(inner.clone())),
            DirectOrMixing::Mixing {
                mixing: rule,
                _phantom: _,
            } => Ok(Box::new(from_combination_rule(
                *rule,
                epsilons.context("Epsilons not provided but required for mixing.")?,
                sigmas.context("Sigmas not provided but required for mixing.")?,
            ))),
        }
    }

    /// Converts `DirectOrMixing` enum to appropriate `IsotropicTwobodyEnergy` trait object
    /// for which `from_combination_rule` function exists.
    ///
    /// Used for mixing `sigmas` (e.g. HardSphere potential).
    ///
    /// In case the parameters of the potential are directly provided, no mixing is performed.
    fn convert_and_mix_sigma(
        &self,
        sigmas: Option<(f64, f64)>,
        from_combination_rule: impl Fn(CombinationRule, (f64, f64)) -> T,
    ) -> anyhow::Result<Box<dyn IsotropicTwobodyEnergy>> {
        match self {
            DirectOrMixing::Direct(inner) => Ok(Box::new(inner.clone())),
            DirectOrMixing::Mixing {
                mixing: rule,
                _phantom: _,
            } => Ok(Box::new(from_combination_rule(
                *rule,
                sigmas.context("Sigmas not provided but required for mixing.")?,
            ))),
        }
    }
}

impl HamiltonianBuilder {
    /// Get hamiltonian from faunus input file.
    pub(crate) fn from_file(filename: impl AsRef<Path>) -> anyhow::Result<HamiltonianBuilder> {
        let yaml = std::fs::read_to_string(filename)?;
        let full: serde_yaml::Value = serde_yaml::from_str(&yaml)?;

        let mut current = &full;
        for key in ["system", "energy"] {
            current = match current.get(key) {
                Some(x) => x,
                None => anyhow::bail!("Could not find `{}` in the YAML file.", key),
            }
        }

        serde_yaml::from_value(current.clone()).map_err(anyhow::Error::msg)
    }

    /// Check that all atom kinds referred to in the hamiltonian exist.
    pub(crate) fn validate(&self, atom_kinds: &[AtomKind]) -> anyhow::Result<()> {
        for pair in self.nonbonded.0.keys() {
            if let DefaultOrPair::Pair(UnorderedPair(x, y)) = pair {
                if !atom_kinds.iter().any(|atom| atom.name() == x)
                    || !atom_kinds.iter().any(|atom| atom.name() == y)
                {
                    anyhow::bail!("Atom kind specified in `nonbonded` does not exist.")
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::topology::AtomKindBuilder;
    use float_cmp::assert_approx_eq;

    use super::*;

    #[test]
    fn hamiltonian_deserialization_pass() {
        let builder = HamiltonianBuilder::from_file("tests/files/topology_pass.yaml").unwrap();

        assert!(builder.nonbonded.0.contains_key(&DefaultOrPair::Default));
        assert!(builder
            .nonbonded
            .0
            .contains_key(&DefaultOrPair::Pair(UnorderedPair(
                String::from("OW"),
                String::from("OW")
            ))));
        assert!(builder
            .nonbonded
            .0
            .contains_key(&DefaultOrPair::Pair(UnorderedPair(
                String::from("OW"),
                String::from("HW")
            ))));

        assert_eq!(builder.nonbonded.0.len(), 3);

        for (pair, interactions) in builder.nonbonded.0 {
            if let DefaultOrPair::Default = pair {
                assert_eq!(
                    interactions,
                    vec![
                        NonbondedInteraction::LennardJones(DirectOrMixing::Direct(
                            LennardJones::new(1.5, 6.0)
                        )),
                        NonbondedInteraction::WeeksChandlerAndersen(DirectOrMixing::Direct(
                            WeeksChandlerAndersen::new(1.3, 8.0)
                        )),
                        NonbondedInteraction::CoulombPlain(coulomb::pairwise::Plain::new(
                            11.0,
                            Some(1.0),
                        ))
                    ]
                );
            }

            if let DefaultOrPair::Pair(UnorderedPair(x, y)) = pair {
                if x == y {
                    assert_eq!(
                        interactions,
                        [
                            NonbondedInteraction::WeeksChandlerAndersen(DirectOrMixing::Direct(
                                WeeksChandlerAndersen::new(1.5, 3.0)
                            )),
                            NonbondedInteraction::HardSphere(DirectOrMixing::Mixing {
                                mixing: CombinationRule::Geometric,
                                _phantom: Default::default()
                            }),
                            NonbondedInteraction::CoulombReactionField(
                                coulomb::pairwise::ReactionField::new(11.0, 100.0, 1.5, true)
                            ),
                        ]
                    )
                } else {
                    assert_eq!(
                        interactions,
                        [
                            NonbondedInteraction::HardSphere(DirectOrMixing::Mixing {
                                mixing: CombinationRule::LorentzBerthelot,
                                _phantom: Default::default()
                            }),
                            NonbondedInteraction::CoulombEwald(
                                coulomb::pairwise::EwaldTruncated::new(11.0, 0.1)
                            ),
                        ]
                    )
                }
            }
        }
    }

    #[test]
    fn hamiltonian_deserialization_fail_duplicate() {
        let error =
            HamiltonianBuilder::from_file("tests/files/nonbonded_duplicate.yaml").unwrap_err();
        assert_eq!(&error.to_string(), "invalid entry: found duplicate key");
    }

    #[test]
    fn hamiltonian_deserialization_fail_duplicate_default() {
        let error = HamiltonianBuilder::from_file("tests/files/nonbonded_duplicate_default.yaml")
            .unwrap_err();
        assert!(error.to_string().contains("duplicate entry with key"));
    }

    #[test]
    fn hamiltonian_builder_validate() {
        let builder = HamiltonianBuilder::from_file("tests/files/topology_pass.yaml").unwrap();

        let atom_ow = AtomKindBuilder::default()
            .name("OW")
            .id(0)
            .mass(16.0)
            .charge(1.0)
            .build()
            .unwrap();

        let atom_hw = AtomKindBuilder::default()
            .name("HW")
            .id(1)
            .mass(1.0)
            .charge(0.0)
            .build()
            .unwrap();

        let atoms = [atom_ow.clone(), atom_hw.clone()];
        builder.validate(&atoms).unwrap();

        let atoms = [atom_ow.clone()];
        let error = builder.validate(&atoms).unwrap_err();
        assert_eq!(
            &error.to_string(),
            "Atom kind specified in `nonbonded` does not exist."
        );

        let atoms = [atom_hw.clone()];
        let error = builder.validate(&atoms).unwrap_err();
        assert_eq!(
            &error.to_string(),
            "Atom kind specified in `nonbonded` does not exist."
        );
    }

    // we can not (easily) test equality of the trait objects so we test the equality of their behavior
    fn assert_behavior(
        obj1: Box<dyn IsotropicTwobodyEnergy>,
        obj2: Box<dyn IsotropicTwobodyEnergy>,
    ) {
        let testing_distances = [0.00201, 0.7, 12.3, 12457.6];

        for &dist in testing_distances.iter() {
            assert_approx_eq!(
                f64,
                obj1.isotropic_twobody_energy(dist),
                obj2.isotropic_twobody_energy(dist)
            );
        }
    }

    // TODO: These tests are commented out as they test a private interface that was
    // subsequently refactored. They should be re-enabled using the public interface
    // once it is stable.

    // #[test]
    // fn test_convert_nonbonded() {
    //     // Lennard Jones -- direct
    //     let expected = LennardJones::new(1.5, 3.2);
    //     let nonbonded =
    //         NonbondedInteraction::LennardJones(DirectOrMixing::Direct(expected.clone()));

    //     let converted = nonbonded.convert(None, None, None, None).unwrap().unwrap();

    //     assert_behavior(converted, Box::new(expected));

    //     // Lennard Jones -- mixing
    //     let expected = LennardJones::new(1.5, 4.5);
    //     let nonbonded = NonbondedInteraction::LennardJones(DirectOrMixing::Mixing {
    //         mixing: CombinationRule::Arithmetic,
    //         _phantom: PhantomData,
    //     });

    //     let converted = nonbonded
    //         .convert(None, Some((2.0, 1.0)), Some((8.2, 0.8)), None)
    //         .unwrap()
    //         .unwrap();

    //     assert_behavior(converted, Box::new(expected));

    //     // Hard Sphere -- mixing
    //     let expected = HardSphere::new(3.0);
    //     let nonbonded = NonbondedInteraction::HardSphere(DirectOrMixing::Mixing {
    //         mixing: CombinationRule::Geometric,
    //         _phantom: PhantomData,
    //     });

    //     let converted = nonbonded
    //         .convert(None, None, Some((4.5, 2.0)), None)
    //         .unwrap()
    //         .unwrap();

    //     assert_behavior(converted, Box::new(expected));

    //     // Coulomb Reaction Field -- charged atoms
    //     let expected = coulomb::pairwise::ReactionField::new(11.0, 15.0, 1.5, false);
    //     let nonbonded = NonbondedInteraction::CoulombReactionField(expected.clone());
    //     let charge = (1.0, -1.0);

    //     let converted = nonbonded
    //         .convert(Some(charge), None, None, None)
    //         .unwrap()
    //         .unwrap();

    //     assert_behavior(
    //         converted,
    //         Box::new(IonIon::new(charge.0 * charge.1, expected)),
    //     );

    //     // Coulomb Reaction Field -- uncharged atom => should result in None
    //     let coulomb = coulomb::pairwise::ReactionField::new(11.0, 15.0, 1.5, false);
    //     let nonbonded = NonbondedInteraction::CoulombReactionField(coulomb.clone());
    //     let charge = (0.0, -1.0);

    //     assert!(nonbonded
    //         .convert(Some(charge), None, None, None)
    //         .unwrap()
    //         .is_none());
    // }

    #[test]
    fn test_get_interaction() {
        let mut interactions = HashMap::new();

        let interaction1 = NonbondedInteraction::WeeksChandlerAndersen(DirectOrMixing::Direct(
            WeeksChandlerAndersen::new(1.5, 3.2),
        ));
        let interaction2 =
            NonbondedInteraction::CoulombPlain(coulomb::pairwise::Plain::new(11.0, None));

        let interaction3 = NonbondedInteraction::HardSphere(DirectOrMixing::Mixing {
            mixing: CombinationRule::Arithmetic,
            _phantom: PhantomData,
        });

        let for_pair = vec![
            interaction1.clone(),
            interaction2.clone(),
            interaction3.clone(),
        ];

        let for_default = vec![interaction1.clone(), interaction2.clone()];

        interactions.insert(
            DefaultOrPair::Pair(UnorderedPair(String::from("NA"), String::from("CL"))),
            for_pair,
        );

        interactions.insert(DefaultOrPair::Default, for_default);

        let atom1 = AtomKindBuilder::default()
            .name("NA")
            .id(0)
            .mass(12.0)
            .charge(1.0)
            .sigma(1.0)
            .build()
            .unwrap();

        let atom2 = AtomKindBuilder::default()
            .name("CL")
            .id(1)
            .mass(16.0)
            .charge(-1.0)
            .sigma(3.0)
            .build()
            .unwrap();

        let atom3 = AtomKindBuilder::default()
            .name("K")
            .id(2)
            .mass(32.0)
            .charge(0.0)
            .sigma(2.0)
            .build()
            .unwrap();

        let mut nonbonded = NonbondedBuilder(interactions);
        let expected = interaction1.to_dyn_energy(&atom1, &atom2).unwrap().unwrap()
            + interaction2.to_dyn_energy(&atom1, &atom2).unwrap().unwrap()
            + interaction3.to_dyn_energy(&atom1, &atom2).unwrap().unwrap();

        let interaction = nonbonded.get_interaction(&atom1, &atom2).unwrap();
        assert_behavior(interaction, expected.clone());

        // changed order of atoms = same result
        let interaction = nonbonded.get_interaction(&atom2, &atom1).unwrap();
        assert_behavior(interaction, expected);

        // default
        let expected = interaction1.to_dyn_energy(&atom2, &atom1).unwrap().unwrap();
        let interaction = nonbonded.get_interaction(&atom1, &atom3).unwrap();
        assert_behavior(interaction, expected);

        // no interaction
        nonbonded.0.remove(&DefaultOrPair::Default);
        let expected = Box::<NoInteraction>::default();
        let interaction = nonbonded.get_interaction(&atom1, &atom3).unwrap();
        assert_behavior(interaction, expected);
    }

    #[test]
    fn test_get_interaction_empty() {
        let mut interactions = HashMap::new();

        let interaction1 = coulomb::pairwise::Plain::new(11.0, None);

        let interaction2 = coulomb::pairwise::EwaldTruncated::new(11.0, 0.2);

        let interaction3 =
            HardSphere::from_combination_rule(CombinationRule::Arithmetic, (1.0, 3.0));

        let for_pair = vec![
            NonbondedInteraction::CoulombPlain(interaction1.clone()),
            NonbondedInteraction::CoulombEwald(interaction2.clone()),
            NonbondedInteraction::HardSphere(DirectOrMixing::Direct(interaction3.clone())),
        ];

        interactions.insert(
            DefaultOrPair::Pair(UnorderedPair(String::from("NA"), String::from("BB"))),
            for_pair,
        );

        let atom1 = AtomKindBuilder::default()
            .name("NA")
            .id(0)
            .mass(12.0)
            .charge(1.0)
            .sigma(1.0)
            .build()
            .unwrap();

        let atom2 = AtomKindBuilder::default()
            .name("BB")
            .id(1)
            .mass(16.0)
            .charge(0.0)
            .sigma(3.0)
            .build()
            .unwrap();

        // first two interactions evaluate to 0
        let mut nonbonded = NonbondedBuilder(interactions);
        let expected = Box::new(IonIon::new(0.0, interaction1.clone()))
            as Box<dyn IsotropicTwobodyEnergy>
            + Box::new(IonIon::new(0.0, interaction2.clone())) as Box<dyn IsotropicTwobodyEnergy>
            + Box::new(interaction3) as Box<dyn IsotropicTwobodyEnergy>;

        let interaction = nonbonded.get_interaction(&atom1, &atom2).unwrap();
        assert_behavior(interaction, expected);

        // all interactions evaluate to 0
        let for_pair = vec![
            NonbondedInteraction::CoulombPlain(interaction1.clone()),
            NonbondedInteraction::CoulombEwald(interaction2.clone()),
        ];

        nonbonded.0.insert(
            DefaultOrPair::Pair(UnorderedPair(String::from("NA"), String::from("BB"))),
            for_pair,
        );

        let expected = Box::new(IonIon::new(0.0, interaction1)) as Box<dyn IsotropicTwobodyEnergy>
            + Box::new(IonIon::new(0.0, interaction2)) as Box<dyn IsotropicTwobodyEnergy>;

        let interaction = nonbonded.get_interaction(&atom1, &atom2).unwrap();
        assert_behavior(interaction, expected);
    }
}

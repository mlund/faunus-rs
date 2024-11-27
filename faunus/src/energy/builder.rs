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

/// Specifies whether the parameters for the interaction are
/// directly provided or should be calculated using a combination rule.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub(crate) enum DirectOrMixing<T: IsotropicTwobodyEnergy> {
    /// Calculate the parameters using the provided combination rule.
    Mixing {
        /// Combination rule to use for mixing.
        mixing: CombinationRule,
        /// Optional cutoff for the interaction.
        cutoff: Option<f64>,
        #[serde(skip)]
        /// Marker specifying the interaction type.
        _phantom: PhantomData<T>,
    },
    /// The parameters for the interaction are specifically provided.
    Direct(T),
}

/// Types of pair interactions
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) enum PairInteraction {
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

impl PairInteraction {
    /// Convert to a boxed `IsotropicTwobodyEnergy` trait object for a given pair of atom types.
    ///
    /// ## Notes
    /// - A mixing rule is applied, if needed.
    fn to_boxed(
        &self,
        atom1: &AtomKind,
        atom2: &AtomKind,
    ) -> anyhow::Result<Box<dyn IsotropicTwobodyEnergy>> {
        let mixed = AtomKind::combine(CombinationRule::Arithmetic, atom1, atom2);
        let charge_product = mixed.charge();

        match self {
            Self::LennardJones(x) => match x {
                DirectOrMixing::Direct(inner) => Ok(Box::new(inner.clone())),
                DirectOrMixing::Mixing {
                    mixing: rule,
                    cutoff: _,
                    _phantom: _,
                } => {
                    let combined = AtomKind::combine(*rule, atom1, atom2);
                    Ok(Box::new(LennardJones::new(
                        combined.epsilon().context("Epsilons not defined!")?,
                        combined.sigma().context("Sigmas not defined!")?,
                    )))
                }
            },
            Self::WeeksChandlerAndersen(x) => match x {
                DirectOrMixing::Direct(inner) => Ok(Box::new(inner.clone())),
                DirectOrMixing::Mixing {
                    mixing: rule,
                    cutoff: _,
                    _phantom: _,
                } => {
                    let combined = AtomKind::combine(*rule, atom1, atom2);
                    Ok(Box::new(WeeksChandlerAndersen::new(
                        combined.epsilon().context("Epsilons not defined!")?,
                        combined.sigma().context("Sigmas not defined!")?,
                    )))
                }
            },
            Self::AshbaughHatch(x) => match x {
                DirectOrMixing::Direct(inner) => Ok(Box::new(inner.clone())),
                DirectOrMixing::Mixing {
                    mixing,
                    cutoff,
                    _phantom,
                } => {
                    let combined = AtomKind::combine(*mixing, atom1, atom2);
                    let lj = LennardJones::new(
                        combined.epsilon().context("Epsilons not defined!")?,
                        combined.sigma().context("Sigmas not defined!")?,
                    );
                    let ah = AshbaughHatch::new(
                        lj,
                        combined.lambda().context("No lambda defined!")?,
                        cutoff.context("Cutoff undefined!")?,
                    );
                    log::trace!("{}-{}: {}", atom1.name(), atom2.name(), ah);
                    Ok(Box::new(ah))
                }
            },
            Self::HardSphere(x) => match x {
                DirectOrMixing::Direct(inner) => Ok(Box::new(inner.clone())),
                DirectOrMixing::Mixing {
                    mixing,
                    cutoff: _,
                    _phantom,
                } => {
                    let combined = AtomKind::combine(*mixing, atom1, atom2);
                    Ok(Box::new(HardSphere::new(
                        combined.sigma().context("Sigmas not defined!")?,
                    )))
                }
            },
            Self::CoulombPlain(scheme) => Self::make_coulomb(charge_product, scheme.clone()),
            Self::CoulombEwald(scheme) => Self::make_coulomb(charge_product, scheme.clone()),
            Self::CoulombRealSpaceEwald(scheme) => {
                Self::make_coulomb(charge_product, scheme.clone())
            }
            Self::CoulombReactionField(scheme) => {
                Self::make_coulomb(charge_product, scheme.clone())
            }
        }
    }
    /// Helper to create a coulombic interaction with a generic scheme
    fn make_coulomb<
        T: coulomb::pairwise::MultipoleEnergy
            + Clone
            + Debug
            + std::cmp::PartialEq
            + 'static
            + Sync
            + Send,
    >(
        charge_product: f64,
        scheme: T,
    ) -> anyhow::Result<Box<dyn IsotropicTwobodyEnergy>> {
        let ionion = IonIon::new(charge_product, scheme.clone());
        Ok(Box::new(ionion))
    }
}

/// Structure storing information about the nonbonded interactions in the system in serializable format.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub(crate) struct PairPotentialBuilder(
    #[serde(with = "::serde_with::rust::maps_duplicate_key_is_error")]
    // defining interactions between the same atom kinds multiple times causes an error
    HashMap<DefaultOrPair, Vec<PairInteraction>>,
);

impl PairPotentialBuilder {
    /// Get interactions for a specific pair of atoms.
    /// If this pair of atoms is not defined, get interactions for Default.
    /// If Default is not defined, return an empty array.
    fn get_pair_interactions(&self, atom1: &str, atom2: &str) -> &[PairInteraction] {
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
        let interactions = self.get_pair_interactions(atom1.name(), atom2.name());

        if interactions.is_empty() {
            log::warn!(
                "No nonbonded interaction defined for '{} <-> {}'.",
                atom1.name(),
                atom2.name()
            );
            return Ok(Box::from(NoInteraction::default()));
        }

        let total_interaction = interactions
            .iter()
            .map(|interact| interact.to_boxed(atom1, atom2).unwrap())
            .sum();

        Ok(total_interaction)
    }
}

/// Structure used for (de)serializing the Hamiltonian of the system.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub(crate) struct HamiltonianBuilder {
    /// Nonbonded interactions defined for the system.
    #[serde(rename = "nonbonded")]
    pub pairpot_builder: PairPotentialBuilder,
}

impl HamiltonianBuilder {
    /// Get hamiltonian from faunus input file.
    ///
    /// This assumes this YAML layout:
    /// ```yaml
    /// system:
    ///   energy:
    ///     nonbonded:
    ///       ...
    /// ```
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
        for pair in self.pairpot_builder.0.keys() {
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

/// Specifies pair of atom kinds interacting with each other.
///
/// TODO: Why not just `Option<UnorderedPair<String>>`?
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

#[cfg(test)]
mod tests {
    use crate::topology::AtomKindBuilder;
    use float_cmp::assert_approx_eq;

    use super::*;

    #[test]
    fn hamiltonian_deserialization_pass() {
        let builder = HamiltonianBuilder::from_file("tests/files/topology_pass.yaml").unwrap();

        assert!(builder
            .pairpot_builder
            .0
            .contains_key(&DefaultOrPair::Default));
        assert!(builder
            .pairpot_builder
            .0
            .contains_key(&DefaultOrPair::Pair(UnorderedPair(
                String::from("OW"),
                String::from("OW")
            ))));
        assert!(builder
            .pairpot_builder
            .0
            .contains_key(&DefaultOrPair::Pair(UnorderedPair(
                String::from("OW"),
                String::from("HW")
            ))));

        assert_eq!(builder.pairpot_builder.0.len(), 3);

        for (pair, interactions) in builder.pairpot_builder.0 {
            if let DefaultOrPair::Default = pair {
                assert_eq!(
                    interactions,
                    vec![
                        PairInteraction::LennardJones(DirectOrMixing::Direct(LennardJones::new(
                            1.5, 6.0
                        ))),
                        PairInteraction::WeeksChandlerAndersen(DirectOrMixing::Direct(
                            WeeksChandlerAndersen::new(1.3, 8.0)
                        )),
                        PairInteraction::CoulombPlain(coulomb::pairwise::Plain::new(
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
                            PairInteraction::WeeksChandlerAndersen(DirectOrMixing::Direct(
                                WeeksChandlerAndersen::new(1.5, 3.0)
                            )),
                            PairInteraction::HardSphere(DirectOrMixing::Mixing {
                                mixing: CombinationRule::Geometric,
                                cutoff: None,
                                _phantom: Default::default()
                            }),
                            PairInteraction::CoulombReactionField(
                                coulomb::pairwise::ReactionField::new(11.0, 100.0, 1.5, true)
                            ),
                        ]
                    )
                } else {
                    assert_eq!(
                        interactions,
                        [
                            PairInteraction::HardSphere(DirectOrMixing::Mixing {
                                mixing: CombinationRule::LorentzBerthelot,
                                cutoff: None,
                                _phantom: Default::default()
                            }),
                            PairInteraction::CoulombEwald(coulomb::pairwise::EwaldTruncated::new(
                                11.0, 0.1
                            )),
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

        let interaction1 = PairInteraction::WeeksChandlerAndersen(DirectOrMixing::Direct(
            WeeksChandlerAndersen::new(1.5, 3.2),
        ));
        let interaction2 = PairInteraction::CoulombPlain(coulomb::pairwise::Plain::new(11.0, None));

        let interaction3 = PairInteraction::HardSphere(DirectOrMixing::Mixing {
            mixing: CombinationRule::Arithmetic,
            cutoff: None,
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

        let mut nonbonded = PairPotentialBuilder(interactions);
        let expected = interaction1.to_boxed(&atom1, &atom2).unwrap()
            + interaction2.to_boxed(&atom1, &atom2).unwrap()
            + interaction3.to_boxed(&atom1, &atom2).unwrap();

        let interaction = nonbonded.get_interaction(&atom1, &atom2).unwrap();
        assert_behavior(interaction, expected.clone());

        // changed order of atoms = same result
        let interaction = nonbonded.get_interaction(&atom2, &atom1).unwrap();
        assert_behavior(interaction, expected);

        // default
        let expected = interaction1.to_boxed(&atom2, &atom1).unwrap();
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
            PairInteraction::CoulombPlain(interaction1.clone()),
            PairInteraction::CoulombEwald(interaction2.clone()),
            PairInteraction::HardSphere(DirectOrMixing::Direct(interaction3.clone())),
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
        let mut nonbonded = PairPotentialBuilder(interactions);
        let expected = Box::new(IonIon::new(0.0, interaction1.clone()))
            as Box<dyn IsotropicTwobodyEnergy>
            + Box::new(IonIon::new(0.0, interaction2.clone())) as Box<dyn IsotropicTwobodyEnergy>
            + Box::new(interaction3) as Box<dyn IsotropicTwobodyEnergy>;

        let interaction = nonbonded.get_interaction(&atom1, &atom2).unwrap();
        assert_behavior(interaction, expected);

        // all interactions evaluate to 0
        let for_pair = vec![
            PairInteraction::CoulombPlain(interaction1.clone()),
            PairInteraction::CoulombEwald(interaction2.clone()),
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

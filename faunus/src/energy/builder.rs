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

use std::{collections::HashMap, fmt, path::Path};

use interatomic::{twobody::IsotropicTwobodyEnergy, CombinationRule};
use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use unordered_pair::UnorderedPair;

use crate::topology::AtomKind;

/// Structure used for (de)serializing the Hamiltonian of the system.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub(super) struct HamiltonianBuilder {
    #[serde(with = "::serde_with::rust::maps_duplicate_key_is_error")]
    // defining interactions between the same atom kinds multiple times causes an error
    nonbonded: HashMap<DefaultOrPair, Vec<NonbondedInteraction>>,
}

/// Specifies pair of atom kinds interacting with each other.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum DefaultOrPair {
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
pub(super) enum NonbondedInteraction {
    /// Lennard-Jones potential.
    LennardJones(DirectOrMixing<interatomic::twobody::LennardJones>),
    /// Weeks-Chandler-Andersen potential.
    #[serde(alias = "WCA")]
    WeeksChandlerAndersen(DirectOrMixing<interatomic::twobody::WeeksChandlerAndersen>),
    /// Hard sphere potential.
    HardSphere(DirectOrMixing<interatomic::twobody::HardSphere>),
}

/// Specifies whether the parameters for the interaction are
/// directly provided or should be calculated using a combination rule.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub(super) enum DirectOrMixing<T: IsotropicTwobodyEnergy> {
    /// Calculate the parameters using the provided combination rule.
    Mixing {
        mixing: CombinationRule,
        #[serde(skip)]
        _phantom: T,
    },
    /// The parameters for the interaction are specifically provided.
    Direct(T),
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
        for pair in self.nonbonded.keys() {
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
    use super::*;

    #[test]
    fn hamiltonian_deserialization_pass() {
        let builder = HamiltonianBuilder::from_file("tests/files/topology_pass.yaml").unwrap();

        assert!(builder.nonbonded.contains_key(&DefaultOrPair::Default));
        assert!(builder
            .nonbonded
            .contains_key(&DefaultOrPair::Pair(UnorderedPair(
                String::from("OW"),
                String::from("OW")
            ))));
        assert!(builder
            .nonbonded
            .contains_key(&DefaultOrPair::Pair(UnorderedPair(
                String::from("OW"),
                String::from("HW")
            ))));

        assert_eq!(builder.nonbonded.len(), 3);

        for (pair, interactions) in builder.nonbonded {
            if let DefaultOrPair::Default = pair {
                assert_eq!(
                    interactions,
                    vec![
                        NonbondedInteraction::LennardJones(DirectOrMixing::Direct(
                            interatomic::twobody::LennardJones::new(1.5, 6.0)
                        )),
                        NonbondedInteraction::WeeksChandlerAndersen(DirectOrMixing::Mixing {
                            mixing: CombinationRule::LorentzBerthelot,
                            _phantom: Default::default()
                        })
                    ]
                );
            }

            if let DefaultOrPair::Pair(UnorderedPair(x, y)) = pair {
                if x == y {
                    assert_eq!(
                        interactions,
                        vec![
                            NonbondedInteraction::WeeksChandlerAndersen(DirectOrMixing::Direct(
                                interatomic::twobody::WeeksChandlerAndersen::new(1.5, 3.0)
                            )),
                            NonbondedInteraction::HardSphere(DirectOrMixing::Mixing {
                                mixing: CombinationRule::Geometric,
                                _phantom: Default::default()
                            })
                        ]
                    )
                } else {
                    assert_eq!(
                        interactions,
                        vec![NonbondedInteraction::HardSphere(DirectOrMixing::Mixing {
                            mixing: CombinationRule::Arithmetic,
                            _phantom: Default::default()
                        })]
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
    fn hamiltonian_builder_validate() {
        let builder = HamiltonianBuilder::from_file("tests/files/topology_pass.yaml").unwrap();

        let atoms = [
            AtomKind::new("OW", 0, 16.0, 1.0, None, None, None, None, HashMap::new()),
            AtomKind::new("HW", 1, 1.0, 0.0, None, None, None, None, HashMap::new()),
        ];

        builder.validate(&atoms).unwrap();

        let atoms = [AtomKind::new(
            "OW",
            0,
            16.0,
            1.0,
            None,
            None,
            None,
            None,
            HashMap::new(),
        )];

        let error = builder.validate(&atoms).unwrap_err();
        assert_eq!(
            &error.to_string(),
            "Atom kind specified in `nonbonded` does not exist."
        );

        let atoms = [AtomKind::new(
            "HW",
            1,
            1.0,
            0.0,
            None,
            None,
            None,
            None,
            HashMap::new(),
        )];

        let error = builder.validate(&atoms).unwrap_err();
        assert_eq!(
            &error.to_string(),
            "Atom kind specified in `nonbonded` does not exist."
        );
    }
}

// TEST duplicate definitions, duplicate default

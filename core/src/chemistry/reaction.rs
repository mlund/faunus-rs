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

//! # Chemical reactions
//!
//! This module contains support for chemical reactions, including
//! parsing and handling of reaction strings.
//! This is used for speciation moves in the grand canonical ensemble.
//!
//! # Examples
//!
//! Description | Example | Notes
//! ------------|---------------- | -----
//! Molecular participants | `A + A = D`            | Possible arrows: `=`, `⇌`, `⇄`, `→`
//! Implicit participants  | `RCOO- + 👻H+ ⇌ RCOOH` | Mark with `👻` or `~`
//! Atomic participants    | `⚛Pb ⇄ ⚛Au`            | Mark with `⚛` or `.`

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Participant in a reaction
///
/// A participant is either an atom, a molecule or an implicit participant.
/// When parsing a reaction, atoms are prefixed with a dot or an atom sign, e.g. ".Na" or "⚛Na".
/// Implicit participants are prefixed with a tilde or a ghost, e.g. "~H" or "👻H".
/// Molecules are not prefixed, e.g. "Cl".
/// The prefix is not stored in the participant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Participant {
    /// Atomic participant like "Au"
    Atom(String),
    /// Molecular participant, like "Water"
    Molecule(String),
    /// Implicit participant, like "H⁺" or "e⁻"
    Implicit(String),
}

impl std::string::ToString for Participant {
    fn to_string(&self) -> String {
        match self {
            Self::Atom(s) => s.clone(),
            Self::Implicit(s) => s.clone(),
            Self::Molecule(s) => s.clone(),
        }
    }
}

impl std::str::FromStr for Participant {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        let p = if let Some(s) = s.strip_prefix(['.', '⚛']) {
            Self::Atom(s.to_string())
        } else if let Some(s) = s.strip_prefix(['~', '👻']) {
            Self::Implicit(s.to_string())
        } else {
            Self::Molecule(s.to_string())
        };
        Ok(p)
    }
}

/// Direction of a reaction
#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq, Default)]
pub enum Direction {
    /// Forward reaction, i.e. left to right
    #[default]
    Forward,
    /// Backward reaction, i.e. right to left
    Backward,
}

/// Chemical reaction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Reaction {
    /// Reaction string in forward direction, e.g. "A + B -> C + D"
    reaction: String,
    /// Participants on the left side of the forward reaction
    left: Vec<Participant>,
    /// Participants on the right side of the forward reaction
    right: Vec<Participant>,
    /// Free energy of the forward reaction
    free_energy: f64,
    /// Current direction of the reaction
    direction: Direction,
}

impl Reaction {
    /// Parse a reaction from a string representation, e.g. "A + 👻B + ⚛C = D + E"
    ///
    /// - Participants are separated by a plus sign, e.g. "A + B + C" clambed by whitespace
    /// - Reactants are separated from products by an equal sign, e.g. "A + B = C + D"
    /// - Atomic participants are prefixed with a dot or an atom sign, e.g. ".C" or "⚛C".
    /// - Implicit participants are prefixed with a tilde or a ghost, e.g. "~B" or "👻B".
    ///
    /// See topology for more information about atomic and implicit participants.
    pub fn from_reaction(forward_reaction: &str, free_energy: f64) -> Result<Self> {
        let sides: Vec<&str> = forward_reaction.split(&['=', '⇌', '⇄', '→']).collect();

        if sides.len() != 2 {
            anyhow::bail!("Invalid reaction: missing '=' separator");
        }

        let reactants = sides[0]
            .trim()
            .split_terminator('+')
            .map(|s| s.trim().parse::<Participant>())
            .collect::<Result<Vec<Participant>>>()?;

        let products = sides[1]
            .trim()
            .split_terminator('+')
            .map(|s| s.trim().parse::<Participant>())
            .collect::<Result<Vec<Participant>>>()?;

        if reactants.is_empty() && products.is_empty() {
            anyhow::bail!("Invalid reaction: no reactants or products");
        }
        Ok(Self {
            reaction: forward_reaction
                .to_string()
                .replace('.', "⚛")
                .replace('~', "👻")
                .replace(['=', '⇄', '→'], "⇌"),
            left: reactants,
            right: products,
            free_energy,
            direction: Direction::Forward,
        })
    }

    /// Set the direction of the reaction
    ///
    /// # Examples
    /// ~~~
    /// use std::str::FromStr;
    /// use faunus::chemistry::reaction::{Reaction, Direction, Participant};
    /// let mut reaction = Reaction::from_reaction("A = B", 1.0).unwrap();
    /// reaction.set_direction(Direction::Backward);
    /// let (reactants, products) = reaction.get();
    /// assert_eq!(reactants[0], Participant::from_str("B").unwrap());
    /// assert_eq!(products[0], Participant::from_str("A").unwrap());
    /// assert_eq!(reaction.free_energy(), -1.0);
    /// ~~~
    pub fn set_direction(&mut self, direction: Direction) {
        self.direction = direction;
    }
    /// Get the current direction of the reaction
    pub fn direction(&self) -> Direction {
        self.direction
    }
    /// Set a random direction
    pub fn random_direction(&mut self, rng: &mut impl rand::Rng) {
        self.direction = if rng.gen_bool(0.5) {
            Direction::Forward
        } else {
            Direction::Backward
        }
    }
    /// Get the free energy of the reaction in the current direction
    pub fn free_energy(&self) -> f64 {
        match self.direction {
            Direction::Forward => self.free_energy,
            Direction::Backward => -self.free_energy,
        }
    }
    /// Get the reactants and products of the reaction in the current direction
    pub fn get(&self) -> (&Vec<Participant>, &Vec<Participant>) {
        match self.direction {
            Direction::Forward => (&self.left, &self.right),
            Direction::Backward => (&self.right, &self.left),
        }
    }
}

// test parsing of reactions like "A + ~B + .C = D + E"
#[test]
fn test_parse_reaction() {
    let reaction = Reaction::from_reaction("A + ~B + .C = D + E", 1.0).unwrap();
    assert_eq!(
        reaction,
        Reaction {
            reaction: "A + 👻B + ⚛C ⇌ D + E".to_string(),
            left: vec![
                Participant::Molecule("A".to_string()),
                Participant::Implicit("B".to_string()),
                Participant::Atom("C".to_string())
            ],
            right: vec![
                Participant::Molecule("D".to_string()),
                Participant::Molecule("E".to_string())
            ],
            free_energy: 1.0,
            direction: Direction::Forward,
        }
    );
}

// Test reaction edge cases where there are no reactants or no products
#[test]
fn test_reaction_edge_cases() {
    // neither reactants nor products NOT OK!
    assert!(Reaction::from_reaction(" = ", 1.0).is_err());

    // empty products OK
    let reaction = Reaction::from_reaction("A = ", 1.0).unwrap();
    assert_eq!(
        reaction,
        Reaction {
            reaction: "A ⇌ ".to_string(),
            left: vec![Participant::Molecule("A".to_string())],
            right: vec![],
            free_energy: 1.0,
            direction: Direction::Forward,
        }
    );
    // empty reactants OK
    let reaction = Reaction::from_reaction(" ⇄ A", 1.0).unwrap();
    assert_eq!(
        reaction,
        Reaction {
            reaction: " ⇌ A".to_string(),
            left: vec![],
            right: vec![Participant::Molecule("A".to_string())],
            free_energy: 1.0,
            direction: Direction::Forward,
        }
    );
}

// test conversion of participants to and from strings
#[test]
fn test_participant() {
    let participant = String::from("~H").parse::<Participant>().unwrap();
    assert_eq!(participant, Participant::Implicit("H".to_string()));
    assert_eq!(participant.to_string(), "H");

    let participant = String::from("👻H").parse::<Participant>().unwrap();
    assert_eq!(participant, Participant::Implicit("H".to_string()));
    assert_eq!(participant.to_string(), "H");

    let participant = String::from(".Na").parse::<Participant>().unwrap();
    assert_eq!(participant, Participant::Atom("Na".to_string()));
    assert_eq!(participant.to_string(), "Na");

    let participant = String::from("⚛Na").parse::<Participant>().unwrap();
    assert_eq!(participant, Participant::Atom("Na".to_string()));
    assert_eq!(participant.to_string(), "Na");

    let participant = String::from("Cl").parse::<Participant>().unwrap();
    assert_eq!(participant, Participant::Molecule("Cl".to_string()));
    assert_eq!(participant.to_string(), "Cl");
}

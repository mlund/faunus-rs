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
//! Participant | Example                |  Notes
//! ------------|----------------------- | ------------------------------------
//! Molecular   | `A + A ⇌ D`            | Possible arrows: `=`, `⇌`, `⇄`, `→`
//! Implicit    | `RCOO- + 👻H+ ⇌ RCOOH` | Mark with `👻` or `~`
//! Atomic      | `⚛Pb ⇄ ⚛Au`            | Mark with `⚛` or `.`

use anyhow::Result;
use num::traits::Inv;
use regex::Regex;
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

impl std::fmt::Display for Participant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Atom(s) => write!(f, "{}", s),
            Self::Implicit(s) => write!(f, "{}", s),
            Self::Molecule(s) => write!(f, "{}", s),
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
    /// Equilibrium constant of the forward reaction
    equilibrium_const: f64,
    /// Current direction of the reaction
    direction: Direction,
}

/// Repeat a reaction participant, e.g. "2A" -> ["A", "A"]
fn repeat_participant(participant: &str) -> Vec<String> {
    let re = Regex::new(r"^(?P<number>\d+)(?P<remaining>.*)").unwrap();
    match re.captures(participant) {
        Some(captured) => {
            let n: usize = captured
                .name("number")
                .unwrap()
                .as_str()
                .parse::<usize>()
                .unwrap_or(1);
            let remaining = captured.name("remaining").unwrap().as_str().trim();
            vec![remaining.to_string(); n]
        }
        None => vec![participant.to_string()],
    }
}

fn parse_side(side: &str) -> Result<Vec<Participant>> {
    side.trim()
        .split_terminator(" + ")
        .map(|i| i.trim())
        .flat_map(repeat_participant)
        .map(|s| s.parse::<Participant>())
        .collect::<Result<Vec<Participant>>>()
}

fn check_sides(sides: &[&str]) -> Result<()> {
    if sides.len() != 2 {
        anyhow::bail!("Invalid reaction: exactly one '=' separator required");
    }
    // check that participants are separated by a plus sign
    for s in sides {
        let num_plusses = s.matches(" + ").count();
        let num_words = s.split_whitespace().count() - num_plusses;
        if num_words > 0 && num_words != num_plusses + 1 {
            anyhow::bail!("Invalid reaction: missing '+' separator");
        }
    }
    Ok(())
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
    pub fn from_reaction(forward_reaction: &str, equilibrium_const: f64) -> Result<Self> {
        let sides: Vec<&str> = forward_reaction.split(&['=', '⇌', '⇄', '→']).collect();
        check_sides(&sides)?;
        let reactants = parse_side(sides[0])?;
        let products = parse_side(sides[1])?;

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
            equilibrium_const,
            direction: Direction::Forward,
        })
    }

    /// Set the direction of the reaction
    ///
    /// # Examples
    /// ~~~
    /// use std::str::FromStr;
    /// use faunus::chemistry::reaction::{Reaction, Direction, Participant};
    /// let mut reaction = Reaction::from_reaction("A = B", 2.0).unwrap();
    /// reaction.set_direction(Direction::Backward);
    /// let (reactants, products) = reaction.get();
    /// assert_eq!(reactants[0], Participant::from_str("B").unwrap());
    /// assert_eq!(products[0], Participant::from_str("A").unwrap());
    /// assert_eq!(reaction.equilibrium_const(), 1.0 / 2.0);
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
    pub fn equilibrium_const(&self) -> f64 {
        match self.direction {
            Direction::Forward => self.equilibrium_const,
            Direction::Backward => self.equilibrium_const.inv(),
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
            equilibrium_const: 1.0,
            direction: Direction::Forward,
        }
    );

    let reaction = Reaction::from_reaction("A + 2~B + .C = D + 1E", 1.0).unwrap();
    assert_eq!(
        reaction,
        Reaction {
            reaction: "A + 2👻B + ⚛C ⇌ D + 1E".to_string(),
            left: vec![
                Participant::Molecule("A".to_string()),
                Participant::Implicit("B".to_string()),
                Participant::Implicit("B".to_string()),
                Participant::Atom("C".to_string())
            ],
            right: vec![
                Participant::Molecule("D".to_string()),
                Participant::Molecule("E".to_string())
            ],
            equilibrium_const: 1.0,
            direction: Direction::Forward,
        }
    );
}

// Test reaction edge cases where there are no reactants or no products
#[test]
fn test_reaction_edge_cases() {
    // neither reactants nor products NOT OK!
    assert!(Reaction::from_reaction(" = ", 1.0).is_err());

    // missing plus sign NOT OK
    assert!(Reaction::from_reaction("H+ Cl- =", 1.0).is_err());

    // plus sign in species is OK
    assert!(Reaction::from_reaction("H+ + Cl- =", 1.0).is_ok());

    // empty products OK
    let reaction = Reaction::from_reaction("A = ", 1.0).unwrap();
    assert_eq!(
        reaction,
        Reaction {
            reaction: "A ⇌ ".to_string(),
            left: vec![Participant::Molecule("A".to_string())],
            right: vec![],
            equilibrium_const: 1.0,
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
            equilibrium_const: 1.0,
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

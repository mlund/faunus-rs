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

use serde::{Deserialize, Serialize};

/// Participant in a reaction
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Participant {
    Atom(String),
    Molecule(String),
    Implicit(String),
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
    /// Set the direction of the reaction
    pub fn set_direction(&mut self, direction: Direction) {
        self.direction = direction;
    }
    /// Get the direction of the reaction
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
    /// Get the free energy of the reaction
    pub fn free_energy(&self) -> f64 {
        match self.direction {
            Direction::Forward => self.free_energy,
            Direction::Backward => -self.free_energy,
        }
    }
    /// Get the reactants and products of the reaction
    pub fn get(&self) -> (&Vec<Participant>, &Vec<Participant>) {
        match self.direction {
            Direction::Forward => (&self.left, &self.right),
            Direction::Backward => (&self.right, &self.left),
        }
    }
    /// Parse a reaction from a string representation, e.g. "A + B = C + D"
    pub fn from_reaction(_reaction: &str, _free_energy: f64) -> anyhow::Result<Self, String> {
        unimplemented!()
    }
}

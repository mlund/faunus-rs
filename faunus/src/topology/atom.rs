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

use crate::topology::{CustomProperty, Value};
use serde::{Deserialize, Serialize};

/// Description of atom properties
///
/// Atoms need not be chemical elements, but can be custom atoms representing interaction sites.
/// This does _not_ include positions; indices etc., but is rather
/// used to represent static properties used for templating atoms.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct AtomKind {
    /// Unique name.
    name: String,
    /// Unique identifier.
    /// Only defined if the AtomKind is inside of Topology.
    #[serde(skip)]
    id: usize,
    /// Atomic mass (g/mol).
    mass: f64,
    /// Atomic charge.
    #[serde(default)]
    charge: f64,
    /// Atomic symbol if appropriate (He, C, O, Fe, etc.).
    element: Option<String>,
    /// Lennard-Jones diameter, σٖᵢᵢ (angstrom).
    sigma: Option<f64>,
    /// Lennard-Jones well depth, εᵢᵢ (kJ/mol).
    epsilon: Option<f64>,
    /// Hydrophobicity information.
    hydrophobicity: Option<Hydrophobicity>,
    /// Map of custom properties.
    #[serde(default)]
    custom: std::collections::HashMap<String, Value>,
}

impl AtomKind {
    /// New atom type with given name but with otherwise default values
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn mass(&self) -> f64 {
        self.mass
    }

    pub fn charge(&self) -> f64 {
        self.charge
    }

    pub fn element(&self) -> Option<&str> {
        self.element.as_deref()
    }

    pub fn sigma(&self) -> Option<f64> {
        self.sigma
    }

    pub fn epsilon(&self) -> Option<f64> {
        self.epsilon
    }

    pub fn hydrophobicity(&self) -> Option<Hydrophobicity> {
        self.hydrophobicity
    }

    pub fn custom(&self) -> &std::collections::HashMap<String, Value> {
        &self.custom
    }

    /// Set unique identifier
    pub(super) fn set_id(&mut self, id: usize) {
        self.id = id;
    }

    /// Set sigma.
    pub fn set_sigma(&mut self, sigma: f64) {
        self.sigma = Some(sigma);
    }

    /// Set epsilon.
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = Some(epsilon);
    }
}

impl CustomProperty for AtomKind {
    fn set_property(&mut self, key: &str, value: Value) -> anyhow::Result<()> {
        self.custom.insert(key.to_string(), value);
        Ok(())
    }
    fn get_property(&self, key: &str) -> Option<Value> {
        self.custom.get(key).cloned()
    }
}

/// Enum to store hydrophobicity information of an atom or residue
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Copy)]
pub enum Hydrophobicity {
    /// Item is hydrophobic
    Hydrophobic,
    /// Item is hydrophilic
    Hydrophilic,
    /// Stores information about surface tension
    SurfaceTension(f64),
}

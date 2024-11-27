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
use derive_builder::Builder;
pub use interatomic::CombinationRule;
use serde::{Deserialize, Serialize};

/// Description of atom properties
///
/// Atoms need not be chemical elements, but can be custom atoms representing interaction sites.
/// This does _not_ include positions; indices etc., but is rather
/// used to represent static properties used for templating atoms.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default, Builder)]
#[serde(deny_unknown_fields)]
#[builder(default)]
pub struct AtomKind {
    /// Unique name.
    #[builder(setter(into))]
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
    #[builder(setter(into, strip_option), default)]
    element: Option<String>,
    /// Lennard-Jones diameter, σٖᵢᵢ (angstrom).
    #[serde(alias = "σ")]
    #[builder(setter(strip_option), default)]
    sigma: Option<f64>,
    /// Lennard-Jones well depth, εᵢᵢ (kJ/mol).
    #[serde(alias = "ε", alias = "eps")]
    #[builder(setter(strip_option), default)]
    epsilon: Option<f64>,
    /// Hydrophobicity information.
    #[builder(setter(strip_option), default)]
    hydrophobicity: Option<Hydrophobicity>,
    /// Map of custom properties.
    #[serde(default)]
    custom: std::collections::HashMap<String, Value>,
}

impl AtomKind {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn id(&self) -> usize {
        self.id
    }

    pub const fn mass(&self) -> f64 {
        self.mass
    }

    pub const fn charge(&self) -> f64 {
        self.charge
    }

    pub fn element(&self) -> Option<&str> {
        self.element.as_deref()
    }

    /// Get the optional Ashbaugh-Hatch scaling factor, λ.
    pub fn lambda(&self) -> Option<f64> {
        match self.hydrophobicity {
            Some(Hydrophobicity::Lambda(lambda)) => Some(lambda),
            _ => None,
        }
    }

    /// Set the Ashbaugh-Hatch scaling factor, λ.
    pub fn set_lambda(&mut self, lambda: f64) -> anyhow::Result<()> {
        self.hydrophobicity = Some(Hydrophobicity::Lambda(lambda));
        Ok(())
    }

    /// Get the particle diameter
    pub const fn sigma(&self) -> Option<f64> {
        self.sigma
    }

    pub const fn epsilon(&self) -> Option<f64> {
        self.epsilon
    }

    pub const fn hydrophobicity(&self) -> Option<Hydrophobicity> {
        self.hydrophobicity
    }

    pub const fn custom(&self) -> &std::collections::HashMap<String, Value> {
        &self.custom
    }

    /// Set unique identifier
    pub(super) fn set_id(&mut self, id: usize) {
        self.id = id;
    }

    /// Set sigma.
    pub fn set_sigma(&mut self, sigma: Option<f64>) {
        self.sigma = sigma;
    }

    /// Set epsilon.
    pub fn set_epsilon(&mut self, epsilon: Option<f64>) {
        self.epsilon = epsilon;
    }

    /// Combine two atom kinds using a combination rule.
    ///
    /// The properties below are combined; all others have default (empty) values.
    ///
    /// - `charge` (product)
    /// - `epsilon` (epsilon rule)
    /// - `lambda` (sigma rule)
    /// - `mass` (sum)
    /// - `sigma` (sigma rule)
    ///
    pub fn combine(rule: CombinationRule, atom1: &Self, atom2: &Self) -> Self {
        let mut atomkind = AtomKind::default();

        if let (Some(a), Some(b)) = (atom1.epsilon(), atom2.epsilon()) {
            atomkind.epsilon = Some(rule.mix_epsilons((a, b)));
        }

        if let (Some(a), Some(b)) = (atom1.sigma(), atom2.sigma()) {
            atomkind.sigma = Some(rule.mix_sigmas((a, b)));
        }

        if let (Some(a), Some(b)) = (atom1.lambda(), atom2.lambda()) {
            atomkind.set_lambda(rule.mix_sigmas((a, b))).unwrap();
        }

        atomkind.charge = atom1.charge() * atom2.charge();
        atomkind.mass = atom1.mass() + atom2.mass();
        atomkind
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
    /// Ashbaugh-Hatch scaling factor
    #[serde(alias = "λ")]
    Lambda(f64),
}

/// Set sigma for a list of atomkinds with `None` sigma
pub fn set_missing_sigma(atomkinds: &mut [AtomKind], default_sigma: f64) {
    atomkinds
        .iter_mut()
        .filter(|i| i.sigma().is_none())
        .for_each(|i| {
            i.set_sigma(Some(default_sigma));
        });
}

/// Set epsilon for for a list of atomkinds with `None` epsilon.
pub fn set_missing_epsilon(atomkinds: &mut [AtomKind], default_epsilon: f64) {
    atomkinds
        .iter_mut()
        .filter(|i| i.epsilon().is_none())
        .for_each(|i| {
            i.set_epsilon(Some(default_epsilon));
        });
}

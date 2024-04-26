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
use chemfiles;
use derive_getters::Getters;
use serde::{Deserialize, Serialize};

/// Description of atom properties
///
/// Atoms need not be chemical elements, but can be custom atoms representing interaction sites.
/// This does _not_ include positions; indices etc., but is rather
/// used to represent static properties used for templating atoms.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default, Getters)]
pub struct AtomKind {
    /// Unique name.
    name: String,
    /// Unique identifier.
    /// Only defined if the AtomKind is inside of Topology.
    #[serde(skip_deserializing)]
    id: usize,
    /// Atomic mass (g/mol).
    mass: f64,
    /// Atomic charge.
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
    pub fn from_str(string: &str) -> Result<AtomKind, anyhow::Error> {
        serde_yaml::from_str::<AtomKind>(string).map_err(anyhow::Error::msg)
    }

    /// New atom type with given name but with otherwise default values
    pub(super) fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }
    /// Set unique identifier
    pub(super) fn set_id(&mut self, id: usize) {
        self.id = id;
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

/// Convert from chemfiles atom to topology atom
impl core::convert::From<chemfiles::AtomRef<'_>> for AtomKind {
    fn from(atom: chemfiles::AtomRef) -> Self {
        AtomKind {
            name: atom.name(),
            id: 0,
            mass: atom.mass(),
            charge: atom.charge(),
            sigma: Some(2.0 * atom.vdw_radius()),
            element: Some(atom.atomic_type()),
            ..Default::default()
        }
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

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn read_atomkind() {
        let string = std::fs::read_to_string("tests/files/atom_input.yaml").unwrap();
        let atom = AtomKind::from_str(&string).unwrap();

        // TODO! write a proper test
        println!("{:?}", atom);
    }

    #[test]
    fn read_atomkind_minimal() {
        let string = std::fs::read_to_string("tests/files/minimal_atom.yaml").unwrap();
        let atom = AtomKind::from_str(&string).unwrap();

        assert_eq!(atom.name, "OW");
        assert_eq!(atom.charge, -1.0);
        assert_eq!(atom.mass, 16.0);
        assert_eq!(atom.id, 0);
        assert!(atom.element.is_none());
        assert!(atom.sigma.is_none());
        assert!(atom.epsilon.is_none());
        assert!(atom.hydrophobicity.is_none());
        assert!(atom.custom.is_empty());
    }
}

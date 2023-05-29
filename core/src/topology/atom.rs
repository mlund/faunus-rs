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

use crate::topology::{CustomProperty, Indices, Value};
use crate::Point;
use chemfiles;
use serde::{Deserialize, Serialize};

/// Enum to store hydrophobicity information of an atom or residue
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Copy)]
pub enum Hydrophobicity {
    Hydrophobic,
    Hydrophilic,
    /// Stores information about surface tension
    SurfaceTension(f64),
}

/// Information about atoms
///
/// Atoms need not be chemical elements, but can be custom atoms representing interaction sites.
/// The `AtomType` does _not_ include positions; indices etc., but is rather
/// used to represent static properties used for templating atoms.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct AtomKind {
    /// Unique name
    name: String,
    /// Unique identifier
    id: usize,
    /// Atomic mass
    mass: f64,
    /// Atomic charge
    charge: f64,
    /// Atomic number
    atomic_number: Option<usize>,
    /// Atomic symbol (He, C, O, Fe, etc.)
    element: Option<String>,
    /// Lennard-Jones diameter
    sigma: Option<f64>,
    /// Lennaard-Jones well depth
    epsilon: Option<f64>,
    /// Hydrophobicity information
    hydrophobicity: Option<Hydrophobicity>,
    /// Map of custom properties
    properties: std::collections::HashMap<String, Value>,
}

impl AtomKind {
    /// New atom type with given name but with otherwise default values
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }
    /// Unique identifier
    pub fn id(&self) -> usize {
        self.id
    }
    /// Set unique identifier
    pub fn set_id(&mut self, id: usize) {
        self.id = id;
    }
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl CustomProperty for AtomKind {
    fn set_property(&mut self, key: &str, value: Value) -> anyhow::Result<()> {
        self.properties.insert(key.to_string(), value);
        Ok(())
    }
    fn get_property(&self, key: &str) -> Option<Value> {
        self.properties.get(key).cloned()
    }
}

/// An atom is the smallest particle entity.
///
/// It does not have to be a real chemical element, but can be a dummy or custom atom.
/// The type or name of the atom is given by an `AtomType`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Atom {
    /// Unique index
    index: usize,
    /// Atom type
    kind: AtomKind,
    /// Position of atom (x, y, z)
    pos: Point,
}

impl Atom {
    /// New atom with given name but with otherwise default values
    pub fn new(atomtype: AtomKind) -> Self {
        Self {
            kind: atomtype,
            ..Default::default()
        }
    }
    /// Atom type
    pub fn kind(&self) -> &AtomKind {
        &self.kind
    }
    /// Position of atom (x, y, z)
    pub fn pos(&self) -> &Point {
        &self.pos
    }
    /// Position of atom (x, y, z)
    pub fn pos_mut(&mut self) -> &mut Point {
        &mut self.pos
    }
    /// Unique identifier
    pub fn id(&self) -> usize {
        self.kind.id
    }
}

impl Indices for Atom {
    fn index(&self) -> usize {
        self.index
    }
    fn set_index(&mut self, index: usize) {
        self.index = index;
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
            atomic_number: Some(atom.atomic_number() as usize),
            ..Default::default()
        }
    }
}

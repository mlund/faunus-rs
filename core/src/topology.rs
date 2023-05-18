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

use crate::Point;
pub use chemfiles::{Atom, AtomMut, AtomRef, Property, Residue, ResidueRef, Topology, TopologyRef};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct AtomProperties {
    pub id: usize,
    pub name: String,
    pub mass: f64,
    pub charge: f64,
    pub dipole_moment_scalar: Option<f64>,
    /// Atomic diameter
    pub sigma: f64,
    pub implicit: bool,
}

/// Properties for molecular groups
#[derive(Debug, Serialize, Default, Clone, Deserialize)]
pub struct MolecularProperties {
    pub compressible: bool,
    pub rigid: bool,
    pub chemical_potential: Option<f64>,
    pub conformations: Vec<Vec<Point>>,
    pub atom_ids: Vec<usize>,
    /// Internal bonds between particles in the group; relative indices
    pub bonds: Vec<[usize; 2]>,
    /// Angle between three particles in the group; relative indices
    pub angles: Vec<[usize; 2]>,
    /// Dihedral angle between four particles in the group; relative indices
    pub dihedrals: Vec<[usize; 4]>,
    /// Improper dihedral angle between four particles in the group; relative indices
    pub impropers: Vec<[usize; 4]>,
}

/// Properties for atomic groups which is is a special case of group
/// consisting of a collection of single particles, e.g. ions; a lennard-jones fluid;
/// or atoms in a lattice.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AtomicProperties {
    pub compressible: bool,
    pub atom_id: usize,
}

#[derive(Debug, Serialize, Clone)]
pub enum GroupKind {
    Atomic(AtomicProperties),
    Molecular(MolecularProperties),
    Implicit,
}

#[derive(Serialize, Clone)]
pub struct Properties {
    /// Unique id corresponding to the index in the molecule collection
    pub id: usize,
    /// Kind of group
    pub kind: GroupKind,
    /// Name of the group
    pub name: String,
    #[serde(skip)]
    pub topology: Topology,
}

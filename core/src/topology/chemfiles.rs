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

//! # Inteface to the [`chemfiles`] crate

use crate::topology;
use chemfiles::{AtomRef, ResidueRef, Topology};
use itertools::Itertools;

impl core::convert::From<AtomRef<'_>> for topology::AtomType {
    fn from(atom: AtomRef) -> Self {
        topology::AtomType {
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

// Convert a chemfiles residue to a topology residue
impl core::convert::From<ResidueRef<'_>> for topology::ResidueType {
    fn from(residue: ResidueRef) -> Self {
        topology::ResidueType {
            name: residue.name(),
            id: residue.id().unwrap() as usize,
            atom_names: topology::Selection::Ids(residue.atoms()),
            bonds: Default::default(),
            properties: Default::default(),
        }
    }
}

impl core::convert::From<Topology> for topology::Topology {
    fn from(value: Topology) -> Self {
        let mut _atom_types: Vec<topology::AtomType> = (0..value.size())
            .map(|i| value.atom(i))
            .unique_by(|atom| atom.name())
            .map(|atom| atom.into())
            .collect();

        for (i, atom) in _atom_types.iter_mut().enumerate() {
            atom.id = i;
        }

        let mut _residue_types: Vec<topology::ResidueType> = (0..value.residues_count())
            .map(|i| value.residue(i).unwrap())
            .unique_by(|residue| residue.name())
            .map(|residue| residue.into())
            .collect();

        for (i, residue) in _residue_types.iter_mut().enumerate() {
            residue.id = i;
        }

        let _bonds: Vec<topology::Bond> = value
            .bonds()
            .iter()
            .map(|bond| {
                topology::Bond::new(
                    [bond[0], bond[1]],
                    topology::BondKind::None,
                    topology::BondOrder::None,
                )
            })
            .collect();

        unimplemented!()
    }
}

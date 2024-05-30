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

//! Loading molecular structures.

use std::path::Path;

use super::{chemfiles_interface, AtomKind, MoleculeKind, MoleculeKindBuilder};
use crate::cell::SimulationCell;
use crate::Point;

/// Obtain positions of particles from the provided structure file using the `chemfiles` crate.
#[cfg(feature = "chemfiles")]
pub fn positions_from_structure_file(
    filename: &impl AsRef<Path>,
    cell: Option<&impl SimulationCell>,
) -> anyhow::Result<Vec<Point>> {
    let frame = chemfiles_interface::frame_from_file(filename)?;
    let positions = chemfiles_interface::positions_from_frame(&frame, cell);
    Ok(positions)
}

#[cfg(not(feature = "chemfiles"))]
pub(crate) fn positions_from_structure_file(
    _filename: &impl AsRef<Path>,
    _cell: Option<&impl SimulationCell>,
) -> anyhow::Result<Vec<Point>> {
    todo!("Not implemented. Use the `chemfiles` feature.")
}

/// Make `MoleculeKind` from structure file populated with atom ids and names
///
/// Atom names must already exist in the list of `AtomKind` objects.
pub fn molecule_from_file(
    molname: &str,
    filename: &impl AsRef<Path>,
    atomkinds: &[AtomKind],
    cell: Option<&impl SimulationCell>,
) -> anyhow::Result<(MoleculeKind, Vec<Point>)> {
    let frame = chemfiles_interface::frame_from_file(filename)?;
    let ok_name = |n: &_| atomkinds.iter().any(|a| a.name() == n);
    let (good_names, bad_names) = frame
        .iter_atoms()
        .map(|a| a.name())
        .partition::<Vec<String>, _>(ok_name);
    if !bad_names.is_empty() {
        anyhow::bail!("Unknown atom names: {:?}", bad_names);
    };

    let get_atom_id = |name| atomkinds.iter().find(|a| a.name() == name).unwrap().id();
    let atom_ids = good_names.iter().map(get_atom_id);
    let molecule = MoleculeKindBuilder::default()
        .name(molname)
        .atom_indices(atom_ids.collect())
        .atoms(good_names)
        .build()?;

    let positions = chemfiles_interface::positions_from_frame(&frame, cell);
    Ok((molecule, positions))
}

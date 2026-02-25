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

use super::io;
use super::{AtomKind, MoleculeKind, MoleculeKindBuilder};
use crate::cell::SimulationCell;
use crate::Point;
use nalgebra::Vector3;

/// Obtain positions of particles from the provided structure file.
///
/// If a simulation cell is given, coordinates are shifted by `-0.5 * bounding_box`
/// to convert from file convention (corner origin) to Faunus convention (center origin).
pub fn positions_from_structure_file(
    filename: &impl AsRef<Path>,
    cell: Option<&impl SimulationCell>,
) -> anyhow::Result<Vec<Point>> {
    let data = io::read_structure(filename)?;

    let shift = cell.map_or_else(Vector3::default, |cell| {
        cell.bounding_box()
            .map_or_else(Vector3::default, |b| -0.5 * b)
    });

    let positions = data.positions.into_iter().map(|pos| pos + shift).collect();
    Ok(positions)
}

/// Make `MoleculeKind` from structure file populated with atom ids and names.
///
/// Atom names must already exist in the list of `AtomKind` objects.
pub fn molecule_from_file(
    molname: &str,
    filename: &impl AsRef<Path>,
    atomkinds: &[AtomKind],
    cell: Option<&impl SimulationCell>,
) -> anyhow::Result<(MoleculeKind, Vec<Point>)> {
    let data = io::read_structure(filename)?;

    let ok_name = |n: &_| atomkinds.iter().any(|a| a.name() == n);
    let (good_names, bad_names) = data
        .names
        .iter()
        .cloned()
        .partition::<Vec<String>, _>(ok_name);
    if !bad_names.is_empty() {
        anyhow::bail!("Unknown atom names: {:?}", bad_names);
    };

    let get_atom_id = |name: &String| atomkinds.iter().find(|a| a.name() == name).unwrap().id();
    let atom_ids: Vec<usize> = good_names.iter().map(get_atom_id).collect();
    let molecule = MoleculeKindBuilder::default()
        .name(molname)
        .atom_indices(atom_ids)
        .atoms(good_names)
        .build()?;

    let shift = cell.map_or_else(Vector3::default, |cell| {
        cell.bounding_box()
            .map_or_else(Vector3::default, |b| -0.5 * b)
    });

    let positions = data.positions.into_iter().map(|pos| pos + shift).collect();
    Ok((molecule, positions))
}

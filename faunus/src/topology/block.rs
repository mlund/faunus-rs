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

use derive_getters::Getters;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(untagged)]
pub enum ActivationStatus {
    Partial(usize),
    #[default]
    All,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, Getters)]
pub struct MoleculeBlock {
    /// Name of the molecule kind of molecules in this block.
    molecule: String,
    /// Index of the molecule kind.
    /// Only defined for MoleculeBlock in a specific Topology.
    #[serde(skip)]
    molecule_index: usize,
    /// Number of molecules in this block.
    #[serde(rename = "N")]
    number: usize,
    /// Should the block of molecules be treated as a single group?
    /// Groups composed of multiple molecules have undefined center of mass.
    /// The default value is `false`.
    // TODO! Does the above make sense?
    #[serde(default = "bool::default")]
    group: bool,
    /// Number of active molecules in this block.
    /// Only coordinates for active molecules are required to be present in the structure file.
    // TODO! Is it so? (We should also allow inactive molecules)
    // What if the partial block is between other blocks?
    #[serde(default)]
    active: ActivationStatus,
}

impl MoleculeBlock {
    pub(super) fn set_molecule_index(&mut self, index: usize) {
        self.molecule_index = index;
    }
}

/*
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_molecule_block() {
        let block = MoleculeBlock {
            molecule: "MOL".to_string(),
            number: 10,
            group: false,
            active: ActivationStatus::All,
        };

        println!("{}", serde_yaml::to_string(&block).unwrap());
    }
}
*/

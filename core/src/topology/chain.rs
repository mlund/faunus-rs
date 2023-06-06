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

use super::Connectivity;
use serde::{Deserialize, Serialize};

/// Chain of connected residues
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Chain {
    /// Unique name, e.g. _"A"_, _"B"_, etc.
    pub name: String,
    /// Unique identifier
    pub id: usize,
    /// List of residue ids in the chain
    pub residue_ids: Vec<usize>,
    /// Connectivity information _between_ residues
    pub connectivity: Vec<Connectivity>,
}

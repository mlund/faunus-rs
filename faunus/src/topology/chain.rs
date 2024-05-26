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

use serde::{Deserialize, Serialize};

use std::ops::Range;

/// Continuous range of atoms with a non-unique name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Chain {
    /// Name of the chain
    name: String,
    /// Atom indices forming the chain.
    /// Range of indices relating to the atoms of a molecule.
    #[serde(
        serialize_with = "crate::topology::serialize_range_as_array",
        deserialize_with = "crate::topology::deserialize_range_from_array"
    )]
    range: Range<usize>,
}

impl Chain {
    #[inline(always)]
    pub fn new(name: &str, range: Range<usize>) -> Self {
        Self {
            name: name.to_owned(),
            range,
        }
    }

    #[inline(always)]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl crate::topology::IndexRange for Chain {
    #[inline(always)]
    fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

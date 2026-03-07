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

//! # Platform specific code.
//!
//! Implementations of `Context` and the Hamiltonian for different hardware platforms.

use std::path::Path;

pub mod soa;

/// Extract medium from system/medium in YAML file
pub fn get_medium(path: impl AsRef<Path>) -> anyhow::Result<interatomic::coulomb::Medium> {
    let file = std::fs::File::open(&path)
        .map_err(|err| anyhow::anyhow!("Could not open {:?}: {}", path.as_ref(), err))?;
    serde_yaml::from_reader(file)
        .ok()
        .and_then(|s: serde_yaml::Value| {
            let val = s.get("system")?.get("medium")?;
            serde_yaml::from_value(val.clone()).ok()
        })
        .ok_or_else(|| anyhow::anyhow!("Could not find `system/medium` in input file"))
}

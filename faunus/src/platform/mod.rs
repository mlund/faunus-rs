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
//! This containes implementations of e.g. `Context` and the Hamiltonian
//! for different hardware platforms.
//! This allows for e.g. GPU acceleration, or special parallelization schemes.

use std::path::Path;

pub mod aos;
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

/// Implement `WithCell`, `WithTopology`, and `WithHamiltonian` for a platform type.
///
/// Requires the type to have fields: `topology`, `cell`, `hamiltonian`.
/// For platforms with cached cell-derived state (e.g. `pbc_params`),
/// implement `WithCell` manually and use `impl_platform_topology_hamiltonian!` instead.
macro_rules! impl_platform_shared {
    ($T:ty) => {
        impl crate::WithCell for $T {
            type SimCell = crate::cell::Cell;
            #[inline(always)]
            fn cell(&self) -> &Self::SimCell {
                &self.cell
            }
            fn cell_mut(&mut self) -> &mut Self::SimCell {
                &mut self.cell
            }
        }

        crate::platform::impl_platform_topology_hamiltonian!($T);
    };
}

/// Implement `WithTopology` and `WithHamiltonian` only (not `WithCell`).
macro_rules! impl_platform_topology_hamiltonian {
    ($T:ty) => {
        impl crate::WithTopology for $T {
            fn topology(&self) -> std::sync::Arc<crate::topology::Topology> {
                self.topology.clone()
            }
            fn topology_ref(&self) -> &std::sync::Arc<crate::topology::Topology> {
                &self.topology
            }
        }

        impl crate::WithHamiltonian for $T {
            fn hamiltonian(&self) -> std::cell::Ref<'_, crate::energy::Hamiltonian> {
                self.hamiltonian.borrow()
            }
            fn hamiltonian_mut(&self) -> std::cell::RefMut<'_, crate::energy::Hamiltonian> {
                self.hamiltonian.borrow_mut()
            }
        }
    };
}

pub(crate) use impl_platform_shared;
pub(crate) use impl_platform_topology_hamiltonian;

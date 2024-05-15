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

//! Implementation of the bonded interactions.

use interatomic::twobody::IsotropicTwobodyEnergy;

use crate::{Change, SyncFrom};

#[derive(Debug, Clone)]
pub struct IntramolecularBonds {
    potentials: Vec<Vec<Box<dyn IsotropicTwobodyEnergy>>>,
}

impl SyncFrom for IntramolecularBonds {
    fn sync_from(&mut self, other: &IntramolecularBonds, change: &Change) -> anyhow::Result<()> {
        match change {
            Change::Everything => self.potentials = other.potentials.clone(),
            Change::None => (),
            _ => todo!("Implement other changes."),
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct IntermolecularBonds {
    potentials: Vec<Vec<Box<dyn IsotropicTwobodyEnergy>>>,
}

impl SyncFrom for IntermolecularBonds {
    fn sync_from(&mut self, other: &IntermolecularBonds, change: &Change) -> anyhow::Result<()> {
        match change {
            Change::Everything => self.potentials = other.potentials.clone(),
            Change::None => (),
            _ => todo!("Implement other changes."),
        }

        Ok(())
    }
}

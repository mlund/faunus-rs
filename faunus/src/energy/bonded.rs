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

use crate::{Change, Context, SyncFrom};

#[derive(Debug, Clone)]
pub struct IntramolecularBonds {}

impl IntramolecularBonds {
    pub(crate) fn energy_change(&self, context: &impl Context, change: &Change) -> f64 {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct IntermolecularBonds {}

impl IntermolecularBonds {
    pub(crate) fn energy_change(&self, context: &impl Context, change: &Change) -> f64 {
        todo!()
    }
}

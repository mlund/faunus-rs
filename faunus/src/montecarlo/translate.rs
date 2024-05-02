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

use super::{Frequency, MoveStatistics};
use crate::group::*;
use crate::transform::{random_unit_vector, Transform};
use crate::{Change, Context, GroupChange};
use rand::prelude::*;

/// Move for translating a random molecule
///
/// This will pick a random molecule of type `id` and translate it by a random displacement.
#[derive(Clone, Debug)]
pub struct TranslateGroup {
    /// Group id to translate
    id: usize,
    /// Maximum displacement
    max_displacement: f64,
    /// Move statisticcs
    statistics: MoveStatistics,
    /// Move frequency
    frequency: Frequency,
}

impl TranslateGroup {
    pub fn new(id: usize, max_displacement: f64, frequency: Frequency) -> Self {
        Self {
            id,
            max_displacement,
            frequency,
            statistics: MoveStatistics::default(),
        }
    }
    /// Pick a random group index with the correct molecule type
    fn random_group(&self, context: &impl Context, rng: &mut ThreadRng) -> Option<usize> {
        let select = GroupSelection::ByMoleculeId(self.id.clone());
        context.select(&select).iter().copied().choose(rng)
    }
}

impl crate::Info for TranslateGroup {
    fn short_name(&self) -> Option<&'static str> {
        Some("translate")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Translate a random molecule")
    }
}

impl<T: Context> super::Move<T> for TranslateGroup {
    fn do_move(&mut self, context: &mut T, rng: &mut ThreadRng) -> Option<Change> {
        if let Some(index) = self.random_group(context, rng) {
            let displacement =
                random_unit_vector(rng) * self.max_displacement * 2.0 * (rng.gen::<f64>() - 0.5);
            Transform::Translate(displacement)
                .on_group(index, context)
                .unwrap();
            Some(Change::SingleGroup(index, GroupChange::RigidBody))
        } else {
            None
        }
    }
    fn statistics(&self) -> &MoveStatistics {
        &self.statistics
    }
    fn statistics_mut(&mut self) -> &mut MoveStatistics {
        &mut self.statistics
    }
    fn frequency(&self) -> Frequency {
        self.frequency
    }
}

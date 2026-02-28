// Copyright 2024 Mikael Lund
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

//! # Energy due to solvent-accessible surface area and atomic-level tensions.

use crate::Point;
use crate::{Change, Context};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use voronota_ltr::{compute_tessellation, Ball, TessellationResult};

#[derive(Clone, Debug)]
struct SasaBackup {
    balls: Vec<Ball>,
    tessellation: TessellationResult,
    tensions: Vec<f64>,
}

#[derive(Debug, Clone, Builder)]
#[builder(derive(Deserialize, Serialize, Debug))]
#[builder_struct_attr(serde(deny_unknown_fields))]
pub struct SasaEnergy {
    /// Probe radius for the tessellation
    probe_radius: f64,
    /// Input balls for tessellation
    #[builder_field_attr(serde(skip))]
    #[builder(default)]
    balls: Vec<Ball>,
    /// Voronoi tessellation of the particles
    #[builder_field_attr(serde(skip))]
    #[builder(default)]
    tessellation: TessellationResult,
    /// Surface tension for each particle
    #[builder_field_attr(serde(skip_serializing))]
    #[builder(default)]
    tensions: Vec<f64>,
    /// Optionally shift calculated energy by this value (kJ/mol)
    #[builder(default = None, setter(strip_option))]
    energy_offset: Option<f64>,
    /// Set offset from first SASA energy calculation event
    #[builder(default = "false")]
    offset_from_first: bool,
    /// Backup for undo on MC reject
    #[builder_field_attr(serde(skip))]
    #[builder(default)]
    backup: Option<SasaBackup>,
}

impl SasaEnergy {
    /// Create from positions, radii, and tensions
    pub fn new<'a>(
        probe_radius: f64,
        positions: impl IntoIterator<Item = &'a Point>,
        radii: impl IntoIterator<Item = f64>,
        tensions: impl IntoIterator<Item = f64>,
        energy_offset: Option<f64>,
        offset_from_first: bool,
    ) -> Self {
        let balls = Self::make_balls(positions, radii);
        let tessellation = compute_tessellation(&balls, probe_radius, None, None);
        Self {
            probe_radius,
            balls,
            tessellation,
            tensions: tensions.into_iter().collect(),
            energy_offset,
            offset_from_first,
            backup: None,
        }
    }

    fn make_balls<'a>(
        positions: impl IntoIterator<Item = &'a Point>,
        radii: impl IntoIterator<Item = f64>,
    ) -> Vec<Ball> {
        std::iter::zip(positions, radii)
            .map(|(pos, radius)| Ball::new(pos.x, pos.y, pos.z, radius))
            .collect()
    }

    /// Update positions only; radii and tensions are left unchanged.
    pub fn update_positions<'a>(&mut self, positions: impl IntoIterator<Item = &'a Point>) {
        std::iter::zip(positions, self.balls.iter_mut()).for_each(|(pos, ball)| {
            ball.x = pos.x;
            ball.y = pos.y;
            ball.z = pos.z;
        });
        self.tessellation = compute_tessellation(&self.balls, self.probe_radius, None, None);
    }

    /// Calculate the surface energy based in the available surface area (kJ/mol)
    pub fn energy(&self, _context: &impl Context, _change: &Change) -> f64 {
        // TODO: calculate only for changed positions
        self.tensions
            .iter()
            .zip(self.tessellation.cells.iter())
            .map(|(tension, cell)| tension * cell.sas_area)
            .sum::<f64>()
            + self.energy_offset.unwrap_or(0.0)
    }
}

impl SasaEnergy {
    /// Update internal state related to given change
    /// TODO: Implement partial updates
    pub(super) fn update(&mut self, context: &impl Context, change: &Change) -> anyhow::Result<()> {
        // TODO: Update only the positions that have changed
        match change {
            Change::Everything => self.update_all(context),
            _ => self.update_all(context),
        }
    }

    /// Update internal state, considering all particles (expensive)
    fn update_all(&mut self, context: &impl Context) -> anyhow::Result<()> {
        let particles = context.get_active_particles();
        let positions = particles.iter().map(|p| -> &Point { &p.pos });

        let radii = particles.iter().map(|p| {
            context.topology().atomkinds()[p.atom_id]
                .sigma()
                .map(|sigma| sigma / 2.0)
                .unwrap_or(0.0)
        });
        self.balls = Self::make_balls(positions, radii);
        self.tessellation = compute_tessellation(&self.balls, self.probe_radius, None, None);
        self.tensions = particles
            .iter()
            .map(|p| {
                context.topology().atomkinds()[p.atom_id]
                    .surface_tension()
                    .unwrap_or(0.0)
            })
            .collect();

        // Set energy offset from the first configuration if requested and only if not already set.
        // This is useful for the SASA energy to be zero for the first configuration, e.g. when
        // molecules are not in contact and fully exposed to the solvent.
        if self.offset_from_first && self.energy_offset.is_none() && !particles.is_empty() {
            let energy = self.energy(context, &Change::Everything);
            self.energy_offset = Some(-energy);
            log::info!(
                "SASA energy offset set from first configuration = {:.2} kJ/mol",
                self.energy_offset.unwrap()
            );
        }
        Ok(())
    }
}

impl SasaEnergy {
    // TODO: implement partial sync
    pub(crate) fn sync_from(&mut self, other: &Self, _change: &Change) -> anyhow::Result<()> {
        self.balls.clone_from(&other.balls);
        self.tessellation.clone_from(&other.tessellation);
        self.tensions.clone_from(&other.tensions);
        Ok(())
    }

    pub(super) fn save_backup(&mut self) {
        assert!(self.backup.is_none(), "backup already exists");
        self.backup = Some(SasaBackup {
            balls: self.balls.clone(),
            tessellation: self.tessellation.clone(),
            tensions: self.tensions.clone(),
        });
    }

    pub(super) fn undo(&mut self) {
        let backup = self.backup.take().expect("undo called without backup");
        self.balls = backup.balls;
        self.tessellation = backup.tessellation;
        self.tensions = backup.tensions;
    }

    pub(super) fn discard_backup(&mut self) {
        self.backup = None;
    }
}

#[cfg(test)]
mod tests_sasaenergy {
    use crate::{energy::EnergyChange, platform::reference::ReferencePlatform, WithHamiltonian};
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_sasa() {
        let context = ReferencePlatform::new(
            "tests/files/sasa_interactions.yaml",
            None,
            &mut rand::thread_rng(),
        )
        .unwrap();

        let energy = context
            .hamiltonian()
            .energy(&context, &crate::Change::Everything);

        assert_approx_eq!(f64, energy, 248.32404971035157);
    }

    #[test]
    fn test_sasa_offset() {
        let context = ReferencePlatform::new(
            "tests/files/sasa_interactions_offset.yaml",
            None,
            &mut rand::thread_rng(),
        )
        .unwrap();

        let energy = context
            .hamiltonian()
            .energy(&context, &crate::Change::Everything);

        assert_approx_eq!(f64, energy, 0.0);
    }
}

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

use crate::{Change, Context, SyncFrom};
use crate::{Particle, Point};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use voronota::{Ball, RadicalTessellation};

#[derive(Debug, Clone, Builder)]
#[builder(derive(Deserialize, Serialize, Debug))]
#[builder_struct_attr(serde(deny_unknown_fields))]
pub struct SasaEnergy {
    /// Probe radius for the tessellation
    probe_radius: f64,
    /// Voronoi tessellation of the particles
    #[builder_field_attr(serde(skip))]
    #[builder(default)]
    tesselation: RadicalTessellation,
    /// Surface tension for each particle
    #[builder_field_attr(serde(skip_serializing))]
    #[builder(default)]
    tensions: Vec<f64>,
}

impl SasaEnergy {
    /// Create from positions, radii, and tensions
    pub fn new<'a>(
        probe_radius: f64,
        positions: impl IntoIterator<Item = &'a Point>,
        radii: impl IntoIterator<Item = f64>,
        tensions: impl IntoIterator<Item = f64>,
    ) -> Self {
        let balls = Self::make_balls(positions, radii);
        Self {
            probe_radius,
            tesselation: RadicalTessellation::from_balls(probe_radius, &balls, None),
            tensions: tensions.into_iter().collect(),
        }
    }

    fn make_balls<'a>(
        positions: impl IntoIterator<Item = &'a Point>,
        radii: impl IntoIterator<Item = f64>,
    ) -> Vec<Ball> {
        let to_ball = |(pos, radius): (&Point, f64)| Ball {
            x: pos.x,
            y: pos.y,
            z: pos.z,
            r: radius,
        };
        std::iter::zip(positions, radii).map(to_ball).collect()
    }

    /// Update positions only; radii and tensions are left unchanged.
    pub fn update_positions<'a>(&mut self, positions: impl IntoIterator<Item = &'a Point>) {
        std::iter::zip(positions, self.tesselation.balls.iter_mut()).for_each(|(pos, ball)| {
            ball.x = pos.x;
            ball.y = pos.y;
            ball.z = pos.z;
        });
        self.tesselation =
            RadicalTessellation::from_balls(self.probe_radius, &self.tesselation.balls, None);
    }

    /// Calculate the surface energy based in the available surface area (kJ/mol)
    pub fn energy(&self, _context: &impl Context, _change: &Change) -> f64 {
        // TODO: calculate only for changed positions
        self.tensions
            .iter()
            .enumerate()
            .map(|(i, tension)| tension * self.tesselation.available_area(i))
            .sum()
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
        let positions = particles
            .iter()
            .map(|particle: &Particle| -> &Point { &particle.pos });

        // Closure to extract radius from topology for a single particle
        let get_radius = |particle: &Particle| -> f64 {
            let atom_id = particle.atom_id;
            context.topology().atomkinds()[atom_id]
                .sigma()
                .map(|sigma| sigma / 2.0)
                .unwrap_or(0.0)
        };
        let radii = particles.iter().map(get_radius);

        // Closure to extract tension from topology for a single particle
        let get_tension = |particle: &Particle| -> f64 {
            context.topology().atomkinds()[particle.atom_id]
                .hydrophobicity()
                .and_then(|h| h.surface_tension())
                .unwrap_or(0.0)
        };

        let balls = Self::make_balls(positions, radii);
        self.tesselation = RadicalTessellation::from_balls(self.probe_radius, &balls, None);
        self.tensions = particles.iter().map(get_tension).collect();
        Ok(())
    }
}

impl SyncFrom for SasaEnergy {
    fn sync_from(&mut self, other: &SasaEnergy, change: &Change) -> anyhow::Result<()> {
        // TODO: implement partial sync
        match change {
            Change::Everything => {
                self.tesselation = other.tesselation.clone();
                self.tensions = other.tensions.clone();
            }
            _ => self.sync_from(other, &Change::Everything)?,
        }
        Ok(())
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

        assert_approx_eq!(f64, energy, 262.3481193159764);
    }
}

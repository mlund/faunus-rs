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

use crate::{topology::Hydrophobicity, Change, Context, SyncFrom};
use crate::{Particle, Point};
use voronota::{Ball, RadicalTessellation};

#[derive(Debug, Clone)]
pub struct SasaEnergy {
    tesselation: RadicalTessellation,
    /// Surface tension for each particle
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
            RadicalTessellation::from_balls(self.tesselation.probe, &self.tesselation.balls, None);
    }

    /// Calculate the surface energy based in the available surface area
    pub fn energy(&self, _: &impl Context, _: &Change) -> f64 {
        // @TODO: calculate only for changed positions
        self.tensions
            .iter()
            .enumerate()
            .map(|(i, tension)| tension * self.tesselation.available_area(i))
            .sum()
    }
}

impl SasaEnergy {
    pub(super) fn update(&mut self, context: &impl Context, _: &Change) -> anyhow::Result<()> {
        // @TODO: Update only the positions that have changed
        let particles = context.get_active_particles();
        let positions = particles
            .iter()
            .map(|particle: &Particle| -> &Point { &particle.pos });
        let radii = particles.iter().map(|particle: &Particle| -> f64 {
            let atom_id = particle.atom_id;
            context.topology().atomkinds()[atom_id]
                .sigma()
                .unwrap_or(0.0)
        });
        let balls = Self::make_balls(positions, radii);

        let tensions = particles.iter().map(|particle: &Particle| -> f64 {
            let atom_id = particle.atom_id;
            let h = context.topology().atomkinds()[atom_id]
                .hydrophobicity()
                .unwrap_or(Hydrophobicity::SurfaceTension(0.0));
            match h {
                Hydrophobicity::SurfaceTension(tension) => tension,
                _ => 0.0,
            }
        });
        self.tesselation = RadicalTessellation::from_balls(self.tesselation.probe, &balls, None);
        self.tensions = tensions.collect();
        Ok(())
    }
}

impl SyncFrom for SasaEnergy {
    fn sync_from(&mut self, other: &SasaEnergy, _: &Change) -> anyhow::Result<()> {
        // @TODO only sync changes
        self.tesselation = other.tesselation.clone();
        self.tensions = other.tensions.clone();
        Ok(())
    }
}

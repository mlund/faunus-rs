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

//! # Solvent-accessible surface area calculation

use super::{exclusions::ExclusionMatrix, EnergyChange};
use crate::{
    energy::{builder::NonbondedBuilder, EnergyTerm},
    topology::Topology,
    Change, Context, Group, GroupChange, SyncFrom,
    topology::Hydrophobicity,
};
use crate::{Particle, Point};
use interatomic::twobody::{IsotropicTwobodyEnergy, NoInteraction};
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

    fn make_balls<'a>(positions: impl IntoIterator<Item = &'a Point>, radii: impl IntoIterator<Item = f64>) -> Vec<Ball> {
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
    pub fn calc_energy(&self) -> f64 {
        self.tensions
            .iter()
            .enumerate()
            .map(|(i, tension)| tension * self.tesselation.available_area(i))
            .sum()
    }
}

impl EnergyChange for SasaEnergy {
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        let particles = context.get_active_particles();
        let positions = particles.iter().map(|particle: &Particle| -> &Point { &particle.pos });
        let radii = particles.iter().map(|particle: &Particle| -> f64 {
            let atom_id = particle.atom_id;
            context.topology().atomkinds()[atom_id].sigma().unwrap_or(0.0)
        });
        let balls = Self::make_balls(positions, radii);

        let tensions = particles.iter().map(|particle: &Particle| -> f64 {
            let atom_id = particle.atom_id;
            let h = context.topology().atomkinds()[atom_id].hydrophobicity().unwrap_or(Hydrophobicity::SurfaceTension(0.0));
            match h {
                Hydrophobicity::SurfaceTension(tension) => tension,
                _ => 0.0,
            }
        });
        let tesselation = RadicalTessellation::from_balls(1.4, &balls, None);
        // self.update_positions(positions);
        0.0
    }
}

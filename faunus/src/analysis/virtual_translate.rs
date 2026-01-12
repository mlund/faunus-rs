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

//! Virtual translate move analysis for force measurement
//!
//! This module implements the virtual translate move analysis, which performs
//! a virtual displacement of a single molecule in a specified direction and
//! measures the force using the Widom perturbation method.
//!
//! The force is calculated as:
//! ```text
//! f = kT * ln<exp(-dU/kT)> / dL
//! ```
//!
//! where `dU` is the energy change due to the perturbation and `dL` is the
//! displacement magnitude.

use super::{Analyze, Frequency};
use average::{Estimate, Mean};
use crate::change::{Change, GroupChange};
use crate::dimension::Dimension;
use crate::energy::EnergyChange;
use crate::group::{GroupSelection, GroupSize};
use crate::topology::Topology;
use crate::{Context, Point};
use anyhow::Result;
use derive_builder::Builder;
use derive_more::Debug;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;


/// Virtual translate move analysis.
///
/// Performs a virtual displacement of a single molecule in a specified direction
/// and measures the force by perturbation using the Widom method:
///
/// `f = kT * ln<exp(-dU/kT)> / dL`
///
/// where `dU` is the energy change and `dL` is the displacement magnitude.
///
/// # Requirements
/// - Exactly one active molecule of the specified type must exist in the system.
/// - The molecule must not be atomic (multi-atom molecules only).
#[derive(Debug, Builder)]
#[builder(build_fn(skip), derive(Deserialize, Serialize))]
pub struct VirtualTranslate {
    /// Molecule name to translate
    #[allow(dead_code)]
    molecule: String,

    /// Molecule id (resolved from name)
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    molecule_id: usize,

    /// Displacement magnitude in Angstrom
    #[builder_field_attr(serde(rename = "dL"))]
    displacement: f64,

    /// Displacement directions. Defaults to z-axis.
    #[allow(dead_code)]
    #[builder_field_attr(serde(default = "default_directions"))]
    directions: Dimension,

    /// Normalized displacement direction
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    unit_direction: Point,

    /// Output file for streaming results
    #[allow(dead_code)]
    #[builder_field_attr(serde(rename = "file"))]
    output_file: Option<PathBuf>,

    /// Stream object for output
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    #[debug(skip)]
    stream: Option<Box<dyn Write>>,

    /// Sample frequency
    frequency: Frequency,

    /// Running average of exp(-dU/kT)
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip))]
    mean_exp_energy: Mean,

    /// Number of samples taken
    #[builder(setter(skip))]
    #[builder_field_attr(serde(skip_deserializing))]
    num_samples: usize,

    /// Temperature in Kelvin (needed to convert energy to kT).
    /// Default is 298.15 K if not specified.
    temperature: f64,
}

/// Default temperature in Kelvin (298.15 K = 25°C)
fn default_temperature() -> f64 {
    298.15
}

/// Default displacement directions (z-axis)
fn default_directions() -> Option<Dimension> {
    Some(Dimension::Z)
}

/// Convert Dimension to a normalized direction vector
fn dimension_to_unit_vector(dim: &Dimension) -> Result<Point> {
    if *dim == Dimension::None {
        anyhow::bail!("Direction cannot be 'None'");
    }
    // Filter a (1,1,1) vector to get direction and normalize
    let dir_vec = dim.filter(Point::new(1.0, 1.0, 1.0));
    let norm = dir_vec.norm();
    if norm < 1e-10 {
        anyhow::bail!("Direction vector cannot be zero");
    }
    Ok(dir_vec / norm)
}

impl VirtualTranslateBuilder {
    fn validate(&self) -> Result<()> {
        if self.molecule.is_none() {
            anyhow::bail!("Missing required field 'molecule' for VirtualTranslate analysis");
        }
        if self.displacement.is_none() {
            anyhow::bail!("Missing required field 'dL' for VirtualTranslate analysis");
        }
        if self.frequency.is_none() {
            anyhow::bail!("Missing required field 'frequency' for VirtualTranslate analysis");
        }
        Ok(())
    }

    /// Build the VirtualTranslate analysis
    pub fn build(&self, topology: &Topology) -> Result<VirtualTranslate> {
        self.validate()?;

        let molecule_name = self.molecule.clone().unwrap();
        let molecule_id = topology
            .find_molecule(&molecule_name)
            .ok_or_else(|| anyhow::anyhow!("Molecule '{}' not found in topology", molecule_name))?
            .id();

        // Check that the molecule is not atomic
        let molkind = &topology.moleculekinds()[molecule_id];
        if molkind.atom_indices().len() < 2 {
            anyhow::bail!(
                "VirtualTranslate requires non-atomic molecules; '{}' has only {} atom(s)",
                molecule_name,
                molkind.atom_indices().len()
            );
        }

        let displacement = self.displacement.unwrap();
        let directions = self.directions.clone().unwrap_or(Dimension::Z);
        let temperature = self.temperature.unwrap_or_else(default_temperature);

        // Convert Dimension to unit direction vector
        let unit_direction = dimension_to_unit_vector(&directions)?;

        // Open output stream if file specified
        let stream = if let Some(ref path) = self.output_file {
            let path = path.clone().unwrap();
            let mut stream = crate::aux::open_compressed(&path)?;
            // Write header
            writeln!(stream, "# step dL/Å dU/kT <force>/kT/Å")?;
            Some(stream)
        } else {
            None
        };

        Ok(VirtualTranslate {
            molecule: molecule_name,
            molecule_id,
            displacement,
            directions,
            unit_direction,
            output_file: self.output_file.clone().flatten(),
            stream,
            frequency: self.frequency.unwrap(),
            mean_exp_energy: Mean::new(),
            num_samples: 0,
            temperature,
        })
    }
}

impl crate::Info for VirtualTranslate {
    fn short_name(&self) -> Option<&'static str> {
        Some("virtualtranslate")
    }

    fn long_name(&self) -> Option<&'static str> {
        Some("Virtual translate move for force measurement by perturbation")
    }

    fn citation(&self) -> Option<&'static str> {
        // The virtual translate method is based on Widom insertion
        Some("doi:10.1063/1.1749657")
    }
}

impl VirtualTranslate {
    /// Calculate the mean free energy from the Widom average
    /// Returns -ln(<exp(-dU/kT)>) in units of kT
    fn mean_free_energy(&self) -> f64 {
        -self.mean_exp_energy.mean().ln()
    }

    /// Calculate the mean force in units of kT/Å
    fn mean_force(&self) -> f64 {
        if self.displacement.abs() > f64::EPSILON {
            -self.mean_free_energy() / self.displacement
        } else {
            0.0
        }
    }

    /// Perform the virtual perturbation and return the energy change in kT
    fn perturb<T: Context>(&self, context: &mut T, group_index: usize) -> Result<f64> {
        let displacement_vector = self.displacement * self.unit_direction;

        // Calculate change descriptor for energy calculations
        let change = Change::SingleGroup(group_index, GroupChange::RigidBody);

        // Calculate old energy (in kJ/mol)
        let old_energy = context.hamiltonian().energy(context, &change);

        // Translate the group
        let particle_indices: Vec<usize> = context.groups()[group_index].iter_active().collect();
        context.translate_particles(&particle_indices, &displacement_vector);

        // Calculate new energy (in kJ/mol)
        let new_energy = context.hamiltonian().energy(context, &change);

        // Restore original positions
        context.translate_particles(&particle_indices, &(-displacement_vector));

        // Convert energy change from kJ/mol to kT
        // kT = R * T where R is the molar gas constant
        const KILO_JOULE_PER_JOULE: f64 = 1e-3;
        let k_t = physical_constants::MOLAR_GAS_CONSTANT * KILO_JOULE_PER_JOULE * self.temperature;
        let delta_u = (new_energy - old_energy) / k_t;

        Ok(delta_u)
    }

    /// Add exp(-dU) to the Widom average, with overflow protection
    fn collect_widom_average(&mut self, energy_change: f64) -> bool {
        if energy_change < -(f64::MAX_EXP as f64) {
            log::warn!(
                "VirtualTranslate: skipping sample due to too negative energy; consider decreasing dL"
            );
            return false;
        }
        self.mean_exp_energy.add((-energy_change).exp());
        true
    }

    /// Write data to the output stream
    fn write_to_stream(&mut self, step: usize, energy_change: f64) -> Result<()> {
        // Calculate mean_force before mutable borrow of stream
        let mean_force = self.mean_force();
        let displacement = self.displacement;

        if let Some(ref mut stream) = self.stream {
            writeln!(
                stream,
                "{} {:.3e} {:.6e} {:.6e}",
                step, displacement, energy_change, mean_force
            )?;
        }
        Ok(())
    }
}

impl<T: Context> Analyze<T> for VirtualTranslate {
    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn sample(&mut self, context: &T, step: usize) -> Result<()> {
        // Check for zero displacement
        if self.displacement.abs() < f64::EPSILON {
            return Ok(());
        }

        // Find active molecules of the specified type
        let group_indices = context.select(&GroupSelection::ByMoleculeId(self.molecule_id));
        let active_groups: Vec<usize> = group_indices
            .into_iter()
            .filter(|&idx| context.groups()[idx].size() != GroupSize::Empty)
            .collect();

        if active_groups.is_empty() {
            return Ok(());
        }

        if active_groups.len() > 1 {
            anyhow::bail!(
                "VirtualTranslate requires exactly ONE active molecule of type '{}', found {}",
                self.molecule,
                active_groups.len()
            );
        }

        let group_index = active_groups[0];

        // We need a mutable clone of the context for perturbation
        // This is a virtual move - we perturb and restore
        let mut trial_context = context.clone();

        let energy_change = self.perturb(&mut trial_context, group_index)?;

        if self.collect_widom_average(energy_change) {
            self.num_samples += 1;
            self.write_to_stream(step, energy_change)?;
        }

        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }

    fn flush(&mut self) {
        if let Some(ref mut stream) = self.stream {
            let _ = stream.flush();
        }
    }
}

impl<T: Context> From<VirtualTranslate> for Box<dyn Analyze<T>> {
    fn from(analysis: VirtualTranslate) -> Box<dyn Analyze<T>> {
        Box::new(analysis)
    }
}

impl std::fmt::Display for VirtualTranslate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Virtual Translate Analysis:")?;
        writeln!(f, "  Molecule:    {}", self.molecule)?;
        writeln!(f, "  dL:          {} Å", self.displacement)?;
        writeln!(
            f,
            "  Direction:   [{:.3}, {:.3}, {:.3}]",
            self.unit_direction.x, self.unit_direction.y, self.unit_direction.z
        )?;
        writeln!(f, "  Samples:     {}", self.num_samples)?;
        if !self.mean_exp_energy.is_empty() {
            writeln!(f, "  <force>:     {:.6} kT/Å", self.mean_force())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_directions() {
        let dir = default_directions();
        assert_eq!(dir, Some(Dimension::Z));
    }

    #[test]
    fn test_dimension_to_unit_vector() {
        use float_cmp::assert_approx_eq;

        // Test z-axis
        let z_vec = dimension_to_unit_vector(&Dimension::Z).unwrap();
        assert_approx_eq!(f64, z_vec.x, 0.0);
        assert_approx_eq!(f64, z_vec.y, 0.0);
        assert_approx_eq!(f64, z_vec.z, 1.0);

        // Test x-axis
        let x_vec = dimension_to_unit_vector(&Dimension::X).unwrap();
        assert_approx_eq!(f64, x_vec.x, 1.0);
        assert_approx_eq!(f64, x_vec.y, 0.0);
        assert_approx_eq!(f64, x_vec.z, 0.0);

        // Test xy-plane (should be normalized)
        let xy_vec = dimension_to_unit_vector(&Dimension::XY).unwrap();
        let expected = 1.0 / 2.0_f64.sqrt();
        assert_approx_eq!(f64, xy_vec.x, expected);
        assert_approx_eq!(f64, xy_vec.y, expected);
        assert_approx_eq!(f64, xy_vec.z, 0.0);

        // Test None should fail
        assert!(dimension_to_unit_vector(&Dimension::None).is_err());
    }
}

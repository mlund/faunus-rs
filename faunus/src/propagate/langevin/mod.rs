// Copyright 2023-2024 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// you may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

//! Mixed rigid-body + per-atom Langevin dynamics via CubeCL.
//!
//! Orchestrates the BAOAB integration pipeline on the compute device:
//! 1. Compute per-atom pairwise and bonded forces
//! 2. Reduce rigid-molecule forces to COM forces and torques
//! 3. BAOAB integrator: rigid (COM+quaternion) and flexible (per-atom) paths
//! 4. Reconstruct rigid-body atom positions from COM + quaternion

mod kernels;
mod pipeline;
#[cfg(test)]
mod tests;
mod utils;

use pipeline::*;
use utils::*;

use crate::cell::{BoundaryConditions, Shape};
use crate::energy::nonbonded_kernel::{build_neighbor_cell_table, compute_n_cells_1d};
use crate::topology::DegreesOfFreedom;
use crate::Context;
use average::{Estimate, Variance};
use cubecl::prelude::*;
use serde::{Deserialize, Serialize};

/// Configuration for Langevin dynamics propagation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LangevinConfig {
    /// Integration timestep in ps
    pub timestep: f64,
    /// Friction coefficient in 1/ps
    pub friction: f64,
    /// Number of LD steps per propagation cycle
    pub steps: usize,
    /// Temperature in Kelvin
    pub temperature: f64,
    /// Cell list rebuild interval (steps). Default: 20.
    #[serde(default = "default_cell_list_rebuild")]
    pub cell_list_rebuild: u32,
}

fn default_cell_list_rebuild() -> u32 {
    20
}

// ============================================================================
// LangevinRunner: high-level orchestration between MC and LD
// ============================================================================

/// Rigid-body Langevin dynamics runner (CubeCL compute backend).
#[derive(Debug)]
pub struct LangevinRunner {
    config: LangevinConfig,
    elapsed: std::time::Duration,
    /// Accumulated energy change across all LD blocks (for drift tracking).
    energy_change_sum: f64,
    /// Running average of translational temperature (K) across LD blocks.
    t_trans: Variance,
    /// Running average of rotational temperature (K) across LD blocks.
    t_rot: Variance,
    gpu: Option<LangevinGpu<cubecl::wgpu::WgpuRuntime>>,
}

impl LangevinRunner {
    pub(in crate::propagate) fn steps(&self) -> usize {
        self.config.steps
    }

    pub(in crate::propagate) fn elapsed(&self) -> std::time::Duration {
        self.elapsed
    }

    pub(in crate::propagate) fn energy_change_sum(&self) -> f64 {
        self.energy_change_sum
    }

    pub(in crate::propagate) fn add_elapsed(&mut self, duration: std::time::Duration) {
        self.elapsed += duration;
    }

    pub(in crate::propagate) fn add_energy_change(&mut self, delta: f64) {
        self.energy_change_sum += delta;
    }

    pub fn new(config: LangevinConfig) -> Self {
        Self {
            config,
            elapsed: std::time::Duration::default(),
            energy_change_sum: 0.0,
            t_trans: Variance::new(),
            t_rot: Variance::new(),
            gpu: None,
        }
    }

    pub(in crate::propagate) fn propagate<T: Context>(
        &mut self,
        context: &mut T,
    ) -> anyhow::Result<()> {
        // Bind locally before storing to avoid unwrap() on the just-assigned Option
        if self.gpu.is_none() {
            let mut gpu = Self::init_gpu(context, &self.config)?;
            Self::upload_full_state(context, &mut gpu)?;
            self.gpu = Some(gpu);
        } else {
            Self::upload_context_state(context, self.gpu.as_mut().unwrap());
        }
        // Safe: always Some after the branch above
        let gpu = self.gpu.as_mut().unwrap();
        let steps = self.config.steps;

        if gpu.has_gpu_forces {
            gpu.run_steps(steps)?;
        } else {
            let mut force_callback = |positions: &[[f32; 4]]| -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
                Self::write_positions(context, positions);
                let n_groups = context.groups().len();
                for g in 0..n_groups {
                    context.update_mass_center(g);
                }
                let forces = context.hamiltonian().forces(context);
                reduce_forces_to_com(context, &forces)
            };
            gpu.run_steps_with_cpu_forces(steps, &mut force_callback)?;
        }

        // LD -> MC: download positions and write back to context
        let gpu = self.gpu.as_ref().unwrap();
        let positions = gpu.download_positions();

        // Check for explosion (NaN or extreme coordinates) in a single pass
        let (has_nan, max_coord) = positions.iter().fold((false, 0.0f32), |(nan, max), p| {
            let [x, y, z, _] = *p;
            (
                nan || x.is_nan() || y.is_nan() || z.is_nan(),
                max.max(x.abs()).max(y.abs()).max(z.abs()),
            )
        });

        if has_nan {
            anyhow::bail!(
                "Langevin dynamics produced NaN positions. \
                 Likely cause: overlapping particles or timestep too large."
            );
        }
        if max_coord > 1e6 {
            log::warn!(
                "Langevin dynamics: max coordinate = {:.1e}. \
                 System may be exploding -- consider reducing timestep or resolving overlaps first.",
                max_coord
            );
        }

        Self::write_positions(context, &positions);

        // LD->MC: download quaternions and write to rigid groups only
        let gpu_quats = gpu.download_quaternions();
        let mol_is_rigid = &gpu.mol_is_rigid_host;
        for (i, (group, q)) in context.groups_mut().iter_mut().zip(&gpu_quats).enumerate() {
            if mol_is_rigid[i] != 0 {
                group.set_quaternion(gpu_to_quat(q));
            }
        }

        // Recompute mass centers for all groups
        let n_groups = context.groups().len();
        for g in 0..n_groups {
            context.update_mass_center(g);
        }

        // Invalidate energy caches since all molecules moved
        context.hamiltonian_mut().invalidate_caches();

        let (t_trans, t_rot) = gpu.download_temperature();
        self.t_trans.add(t_trans as f64);
        self.t_rot.add(t_rot as f64);
        log::debug!("LD block: T_trans={t_trans:.1} K, T_rot={t_rot:.1} K");
        Ok(())
    }

    /// Initialize CubeCL wgpu client and create the compute pipeline.
    fn init_gpu<T: Context>(
        context: &T,
        config: &LangevinConfig,
    ) -> anyhow::Result<LangevinGpu<cubecl::wgpu::WgpuRuntime>> {
        let n_atoms = context.num_particles() as u32;
        let n_molecules = context.groups().len() as u32;
        let box_length = context
            .cell()
            .bounding_box()
            .ok_or_else(|| anyhow::anyhow!("LangevinDynamics requires a bounded cell"))?;
        let kt = (crate::R_IN_KJ_PER_MOL * config.temperature) as f32;

        let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
        let client = cubecl::wgpu::WgpuRuntime::client(&device);
        log::info!("CubeCL wgpu runtime initialized");

        Ok(LangevinGpu::new(
            client,
            config.clone(),
            n_atoms,
            n_molecules,
            box_length.x as f32,
            kt,
        ))
    }

    /// Build upload data from the simulation context and write it to device buffers.
    fn upload_full_state<T: Context>(
        context: &T,
        gpu: &mut LangevinGpu<cubecl::wgpu::WgpuRuntime>,
    ) -> anyhow::Result<()> {
        let n = context.num_particles();
        let groups = context.groups();
        let n_mol = groups.len();
        let topology = context.topology();
        let cell = context.cell();

        let positions = pack_positions_f32(context);

        // Extract spline data for on-device force computation if available
        let spline = extract_spline_data(context);

        // Extract bonded topology into CSR layout for on-device bonded forces
        let (bond_data, angle_data, dihedral_data) = extract_bonded_data(context);

        // Exclusion CSR for intra-molecular NB pairs in flexible molecules
        let (excl_offsets, excl_atoms) =
            crate::energy::bonded::kernel::repack_exclusions(&context.topology(), context.groups());

        // COM positions, reference-frame coordinates, molecule→atom offsets,
        // molecule masses, and diagonal inertia tensors — all in a single pass.
        let mut com_positions = Vec::with_capacity(n_mol * 4);
        let mut ref_positions = Vec::with_capacity(n * 4);
        let mut mol_atom_offsets = Vec::with_capacity(n_mol + 1);
        let mut mol_masses = Vec::with_capacity(n_mol);
        let mut mol_inertia = Vec::with_capacity(n_mol * 4);

        for group in groups {
            mol_atom_offsets.push(group.start() as u32);
            let com = group
                .mass_center()
                .copied()
                .unwrap_or_else(|| context.mass_center(&group.iter_active().collect::<Vec<_>>()));
            com_positions.extend_from_slice(&[com.x as f32, com.y as f32, com.z as f32, 0.0]);

            let mol_kind = &topology.moleculekinds()[group.molecule()];
            let atom_indices = mol_kind.atom_indices();
            let mut total_mass = 0.0f64;
            let mut ixx = 0.0f64;
            let mut iyy = 0.0f64;
            let mut izz = 0.0f64;
            for (idx, i) in group.iter_active().enumerate() {
                let rel = cell.distance(&context.position(i), &com);
                let (rx, ry, rz) = (rel.x, rel.y, rel.z);
                ref_positions.extend_from_slice(&[rx as f32, ry as f32, rz as f32, 0.0]);

                let m = atom_mass(&topology, atom_indices, idx) as f64;
                total_mass += m;
                ixx += m * (ry * ry + rz * rz);
                iyy += m * (rx * rx + rz * rz);
                izz += m * (rx * rx + ry * ry);
            }
            mol_masses.push(total_mass as f32);
            mol_inertia.extend_from_slice(&[ixx as f32, iyy as f32, izz as f32, 0.0f32]);
        }
        if let Some(last) = groups.last() {
            mol_atom_offsets.push((last.start() + last.len()) as u32);
        } else {
            mol_atom_offsets.push(0);
        }

        // Read quaternions from groups
        let quaternions: Vec<f32> = groups
            .iter()
            .flat_map(|g| quat_to_gpu(g.quaternion()))
            .collect();

        // Classify molecules: Rigid|RigidAlchemical → rigid, Free → per-atom, Frozen → skip
        let mol_is_rigid: Vec<u32> = groups
            .iter()
            .map(|g| {
                u32::from(
                    topology.moleculekinds()[g.molecule()]
                        .degrees_of_freedom()
                        .is_rigid(),
                )
            })
            .collect();

        // Per-atom flexible flag and masses (Free molecules get flexible atoms)
        let mut atom_is_flexible = Vec::with_capacity(n);
        let mut atom_masses_vec = Vec::with_capacity(n);
        for group in groups {
            let mol_kind = &topology.moleculekinds()[group.molecule()];
            let is_free = u32::from(matches!(
                mol_kind.degrees_of_freedom(),
                DegreesOfFreedom::Free
            ));
            let atom_indices = mol_kind.atom_indices();
            for idx in 0..group.len() {
                atom_is_flexible.push(is_free);
                atom_masses_vec.push(atom_mass(&topology, atom_indices, idx));
            }
        }
        let has_flexible = atom_is_flexible.iter().any(|&f| f != 0);

        if let (true, Some(bonds)) = (has_flexible, &bond_data) {
            warn_timestep_stability(
                gpu.config.timestep,
                &atom_masses_vec,
                &atom_is_flexible,
                bonds,
            );
        }

        let (com_velocities, angular_velocities, atom_velocities) = generate_mb_velocities(
            gpu.kt as f64,
            &mol_is_rigid,
            &mol_masses,
            &mol_inertia,
            &atom_is_flexible,
            &atom_masses_vec,
        );

        gpu.upload_state(LangevinUploadData {
            positions,
            ref_positions,
            com_positions,
            mol_atom_offsets,
            mol_masses,
            mol_inertia,
            quaternions,
            com_velocities,
            angular_velocities,
            atom_type_ids: spline.atom_type_ids,
            mol_ids: spline.mol_ids,
            spline_params: spline.params,
            spline_coeffs: spline.coeffs,
            n_atom_types: spline.n_atom_types,
            bond_data,
            angle_data,
            dihedral_data,
            excl_offsets,
            excl_atoms,
            mol_is_rigid,
            atom_velocities,
            atom_masses: atom_masses_vec,
            atom_is_flexible,
            has_flexible,
        });
        Ok(())
    }

    /// Upload positions, COM, quaternions, and box length from context to device.
    ///
    /// Box length must be synced because NPT volume moves can change it between LD blocks.
    fn upload_context_state<T: Context>(
        context: &T,
        gpu: &mut LangevinGpu<cubecl::wgpu::WgpuRuntime>,
    ) {
        let positions = pack_positions_f32(context);
        let (com_positions, quaternions) = pack_com_and_quaternions(context);
        gpu.upload_positions_com_quaternions(&positions, &com_positions, &quaternions);

        // Sync box length (may have changed via NPT volume moves)
        if let Some(bb) = context.cell().bounding_box() {
            let new_box = bb.x as f32;
            if (new_box - gpu.box_length).abs() > f32::EPSILON {
                gpu.box_length = new_box;
                if gpu.has_gpu_forces {
                    let new_n = compute_n_cells_1d(gpu.box_length, gpu.max_cutoff);
                    if new_n != gpu.n_cells_1d {
                        gpu.n_cells_1d = new_n;
                        let nb_table = build_neighbor_cell_table(gpu.n_cells_1d);
                        gpu.neighbor_cells = gpu
                            .client
                            .create_from_slice(bytemuck::cast_slice(&nb_table));
                    }
                    log::debug!(
                        "Box length changed to {:.4}, cell grid {}³",
                        gpu.box_length,
                        gpu.n_cells_1d,
                    );
                }
            }
        }
    }

    fn write_positions<T: Context>(context: &mut T, positions: &[[f32; 4]]) {
        let points: Vec<crate::Point> = positions
            .iter()
            .map(|p| {
                let mut pos = crate::Point::new(p[0] as f64, p[1] as f64, p[2] as f64);
                context.cell().boundary(&mut pos);
                pos
            })
            .collect();

        context.set_positions(0..points.len(), points.iter());
    }

    pub(in crate::propagate) fn to_yaml(&self) -> serde_yml::Value {
        let mut map = serde_yml::Mapping::new();
        map.insert("timestep".into(), self.config.timestep.into());
        map.insert("friction".into(), self.config.friction.into());
        map.insert("steps".into(), self.config.steps.into());
        map.insert("temperature".into(), self.config.temperature.into());
        map.insert(
            "cell_list_rebuild".into(),
            self.config.cell_list_rebuild.into(),
        );
        map.insert(
            "elapsed_seconds".into(),
            serde_yml::Value::Number(serde_yml::Number::from(self.elapsed.as_secs_f64())),
        );
        if !self.t_trans.is_empty() {
            let mut temp_map = serde_yml::Mapping::new();
            temp_map.insert(
                "translational".into(),
                format!(
                    "{:.1} +/- {:.1}",
                    self.t_trans.mean(),
                    self.t_trans.sample_variance().sqrt()
                )
                .into(),
            );
            temp_map.insert(
                "rotational".into(),
                format!(
                    "{:.1} +/- {:.1}",
                    self.t_rot.mean(),
                    self.t_rot.sample_variance().sqrt()
                )
                .into(),
            );
            map.insert(
                "measured_temperature".into(),
                serde_yml::Value::Mapping(temp_map),
            );
        }
        serde_yml::Value::Tagged(Box::new(serde_yml::value::TaggedValue {
            tag: serde_yml::value::Tag::new("LangevinDynamics"),
            value: serde_yml::Value::Mapping(map),
        }))
    }
}

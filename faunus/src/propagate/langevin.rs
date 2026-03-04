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

#[cfg(feature = "gpu")]
use crate::cell::{BoundaryConditions, Shape};
use crate::Context;
#[cfg(feature = "gpu")]
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for GPU-accelerated Langevin dynamics propagation.
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
}

/// GPU-accelerated rigid-body Langevin dynamics runner.
#[derive(Debug)]
pub struct LangevinRunner {
    pub(crate) config: LangevinConfig,
    pub(super) elapsed: std::time::Duration,
    /// Accumulated energy change across all LD blocks (for drift tracking).
    pub(super) energy_change_sum: f64,
    #[cfg(feature = "gpu")]
    gpu_state: Option<crate::gpu::langevin::LangevinGpu>,
}

impl LangevinRunner {
    pub fn new(config: LangevinConfig) -> Self {
        Self {
            config,
            elapsed: std::time::Duration::default(),
            energy_change_sum: 0.0,
            #[cfg(feature = "gpu")]
            gpu_state: None,
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub(super) fn propagate<T: Context>(&mut self, _context: &mut T) -> anyhow::Result<()> {
        anyhow::bail!("GPU Langevin dynamics requires the `gpu` feature")
    }

    #[cfg(feature = "gpu")]
    pub(super) fn propagate<T: Context>(&mut self, context: &mut T) -> anyhow::Result<()> {
        // First call: initialize GPU and upload full state (positions, velocities=0, quaternions=identity)
        // Subsequent calls: skip upload — GPU state (velocities, quaternions) persists between blocks
        let first_call = self.gpu_state.is_none();
        if first_call {
            self.gpu_state = Some(Self::init_gpu(context, &self.config)?);
            Self::upload_state(context, self.gpu_state.as_ref().unwrap())?;
        }
        let gpu_ld = self.gpu_state.as_mut().unwrap();

        // Run LD steps on GPU
        gpu_ld.run_steps(self.config.steps);

        // LD → MC: download positions and write back to context
        let positions = gpu_ld.download_positions();

        // Check for explosion (NaN or extreme coordinates)
        let has_nan = positions
            .iter()
            .any(|p| p[0].is_nan() || p[1].is_nan() || p[2].is_nan());
        let max_coord = positions
            .iter()
            .flat_map(|p| [p[0].abs(), p[1].abs(), p[2].abs()])
            .fold(0.0f32, f32::max);

        if has_nan {
            anyhow::bail!(
                "Langevin dynamics produced NaN positions. \
                 Likely cause: overlapping particles or timestep too large."
            );
        }
        if max_coord > 1e6 {
            log::warn!(
                "Langevin dynamics: max coordinate = {:.1e}. \
                 System may be exploding — consider reducing timestep or resolving overlaps first.",
                max_coord
            );
        }

        Self::write_positions(context, &positions)?;

        // Recompute mass centers for all groups
        let n_groups = context.groups().len();
        for g in 0..n_groups {
            context.update_mass_center(g);
        }

        // Invalidate energy caches since all molecules moved
        context.hamiltonian_mut().invalidate_caches();

        if log::log_enabled!(log::Level::Debug) {
            let (t_trans, t_rot) = gpu_ld.download_temperature();
            log::debug!("LD block: T_trans={t_trans:.1} K, T_rot={t_rot:.1} K");
        }
        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn init_gpu<T: Context>(
        context: &T,
        config: &LangevinConfig,
    ) -> anyhow::Result<crate::gpu::langevin::LangevinGpu> {
        let gpu_ctx = std::sync::Arc::new(crate::gpu::GpuContext::new()?);

        // Extract spline data from the hamiltonian's nonbonded term
        let hamiltonian = context.hamiltonian();
        let spline_data = hamiltonian
            .energy_terms()
            .iter()
            .find_map(|term| match term {
                crate::energy::EnergyTerm::NonbondedMatrixSplined(nb) => {
                    Some(crate::gpu::spline::GpuSplineData::from_matrix(nb))
                }
                _ => None,
            })
            .ok_or_else(|| {
                anyhow::anyhow!("LangevinDynamics requires a splined nonbonded potential")
            })?;

        let spline_buffers = gpu_ctx.upload_spline_data(&spline_data);

        let groups = context.groups();
        let n_atoms = context.num_particles() as u32;
        let n_molecules = groups.len() as u32;

        let topology = context.topology();
        let n_atom_types = topology.atomkinds().len() as u32;

        let box_length = context
            .cell()
            .bounding_box()
            .ok_or_else(|| anyhow::anyhow!("LangevinDynamics requires a bounded cell"))?;

        // Use the spline cutoff from the first pair's r_max
        let cutoff = if !spline_data.params.is_empty() {
            spline_data.params[0].r_max
        } else {
            15.0
        };

        let kt = (physical_constants::MOLAR_GAS_CONSTANT * 1e-3 * config.temperature) as f32; // kJ/mol

        drop(hamiltonian);

        Ok(crate::gpu::langevin::LangevinGpu::new(
            gpu_ctx,
            config.clone(),
            spline_buffers,
            n_atoms,
            n_molecules,
            n_atom_types,
            box_length.x as f32,
            cutoff,
            kt,
        ))
    }

    /// Build GPU upload data from the simulation context and write it to GPU buffers.
    #[cfg(feature = "gpu")]
    fn upload_state<T: Context>(
        context: &T,
        gpu_ld: &crate::gpu::langevin::LangevinGpu,
    ) -> anyhow::Result<()> {
        use crate::gpu::langevin::LangevinUploadData;

        let n = context.num_particles();
        let groups = context.groups();
        let n_mol = groups.len();
        let topology = context.topology();
        let cell = context.cell();

        // Per-atom positions [x, y, z, atom_type_bits]
        let mut positions = Vec::with_capacity(n * 4);
        if let (Some((x, y, z)), Some(atom_kinds)) =
            (context.positions_soa(), context.atom_kinds_u32())
        {
            for i in 0..n {
                positions.push(x[i] as f32);
                positions.push(y[i] as f32);
                positions.push(z[i] as f32);
                positions.push(f32::from_bits(atom_kinds[i]));
            }
        } else {
            for i in 0..n {
                let pos = context.position(i);
                let kind = context.get_atomkind(i);
                positions.push(pos.x as f32);
                positions.push(pos.y as f32);
                positions.push(pos.z as f32);
                positions.push(f32::from_bits(kind as u32));
            }
        }

        let atom_type_ids: Vec<u32> = if let Some(kinds) = context.atom_kinds_u32() {
            kinds.to_vec()
        } else {
            (0..n).map(|i| context.get_atomkind(i) as u32).collect()
        };

        let mut atom_mol_ids: Vec<u32> = Vec::with_capacity(n);
        for (mol_idx, group) in groups.iter().enumerate() {
            for _ in group.iter_active() {
                atom_mol_ids.push(mol_idx as u32);
            }
        }

        // COM positions, reference-frame coordinates, and molecule→atom offsets
        let mut com_positions = Vec::with_capacity(n_mol * 4);
        let mut ref_positions = Vec::with_capacity(n * 4);
        let mut mol_atom_offsets = Vec::with_capacity(n_mol + 1);

        for group in groups {
            mol_atom_offsets.push(group.start() as u32);
            let com = group
                .mass_center()
                .copied()
                .unwrap_or_else(|| context.mass_center(&group.iter_active().collect::<Vec<_>>()));
            com_positions.extend_from_slice(&[com.x as f32, com.y as f32, com.z as f32, 0.0]);

            for i in group.iter_active() {
                let rel = cell.distance(&context.position(i), &com);
                ref_positions.extend_from_slice(&[rel.x as f32, rel.y as f32, rel.z as f32, 0.0]);
            }
        }
        if let Some(last) = groups.last() {
            mol_atom_offsets.push((last.start() + last.len()) as u32);
        } else {
            mol_atom_offsets.push(0);
        }

        // Molecule masses and diagonal inertia tensors from reference positions
        let mut mol_masses = Vec::with_capacity(n_mol);
        let mut mol_inertia = Vec::with_capacity(n_mol * 4);
        let mut ref_offset = 0usize;
        for group in groups {
            let mol_kind = &topology.moleculekinds()[group.molecule()];
            let atom_indices = mol_kind.atom_indices();
            let mut total_mass = 0.0f64;
            let mut ixx = 0.0f64;
            let mut iyy = 0.0f64;
            let mut izz = 0.0f64;
            for idx in 0..group.len() {
                let m = if idx < atom_indices.len() {
                    topology.atomkinds()[atom_indices[idx]].mass()
                } else {
                    1.0
                };
                total_mass += m;
                let rx = ref_positions[ref_offset] as f64;
                let ry = ref_positions[ref_offset + 1] as f64;
                let rz = ref_positions[ref_offset + 2] as f64;
                ref_offset += 4;
                ixx += m * (ry * ry + rz * rz);
                iyy += m * (rx * rx + rz * rz);
                izz += m * (rx * rx + ry * ry);
            }
            mol_masses.push(total_mass as f32);
            mol_inertia.extend_from_slice(&[ixx as f32, iyy as f32, izz as f32, 0.0f32]);
        }

        // Identity quaternions
        let quaternions: Vec<f32> = (0..n_mol).flat_map(|_| [0.0f32, 0.0, 0.0, 1.0]).collect();

        // Maxwell-Boltzmann velocities: sigma = sqrt(kT * 100 / M) in Å/ps
        let kt = gpu_ld.kt() as f64;
        let conv = 100.0; // kJ/mol → amu·Å²/ps²
        let mut rng = rand::thread_rng();
        let gauss = |rng: &mut rand::rngs::ThreadRng| -> f64 {
            let u1: f64 = rng.gen::<f64>().max(1e-30);
            let u2: f64 = rng.gen();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };

        let mut com_velocities = Vec::with_capacity(n_mol * 4);
        for &mass in &mol_masses {
            let sigma = ((kt * conv) / mass as f64).sqrt();
            com_velocities.push((sigma * gauss(&mut rng)) as f32);
            com_velocities.push((sigma * gauss(&mut rng)) as f32);
            com_velocities.push((sigma * gauss(&mut rng)) as f32);
            com_velocities.push(0.0f32);
        }

        let mut angular_velocities = Vec::with_capacity(n_mol * 4);
        for m_idx in 0..n_mol {
            let sigma_from_inertia = |i: f64| -> f64 {
                if i > 0.0 {
                    ((kt * conv) / i).sqrt()
                } else {
                    0.0
                }
            };
            let sx = sigma_from_inertia(mol_inertia[m_idx * 4] as f64);
            let sy = sigma_from_inertia(mol_inertia[m_idx * 4 + 1] as f64);
            let sz = sigma_from_inertia(mol_inertia[m_idx * 4 + 2] as f64);
            angular_velocities.push((sx * gauss(&mut rng)) as f32);
            angular_velocities.push((sy * gauss(&mut rng)) as f32);
            angular_velocities.push((sz * gauss(&mut rng)) as f32);
            angular_velocities.push(0.0f32);
        }

        gpu_ld.upload_state(&LangevinUploadData {
            positions,
            atom_type_ids,
            atom_mol_ids,
            ref_positions,
            com_positions,
            mol_atom_offsets,
            mol_masses,
            mol_inertia,
            quaternions,
            com_velocities,
            angular_velocities,
        });
        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn write_positions<T: Context>(context: &mut T, positions: &[[f32; 4]]) -> anyhow::Result<()> {
        let particles: Vec<crate::Particle> = positions
            .iter()
            .map(|p| {
                let kind = f32::to_bits(p[3]) as usize;
                let mut pos = crate::Point::new(p[0] as f64, p[1] as f64, p[2] as f64);
                context.cell().boundary(&mut pos);
                crate::Particle::new(kind, pos)
            })
            .collect();

        context.set_particles(0..particles.len(), particles.iter())?;
        Ok(())
    }

    pub(super) fn to_yaml(&self) -> serde_yaml::Value {
        let mut map = serde_yaml::Mapping::new();
        map.insert("timestep".into(), self.config.timestep.into());
        map.insert("friction".into(), self.config.friction.into());
        map.insert("steps".into(), self.config.steps.into());
        map.insert("temperature".into(), self.config.temperature.into());
        map.insert(
            "elapsed_seconds".into(),
            serde_yaml::Value::Number(serde_yaml::Number::from(self.elapsed.as_secs_f64())),
        );
        serde_yaml::Value::Tagged(Box::new(serde_yaml::value::TaggedValue {
            tag: serde_yaml::value::Tag::new("LangevinDynamics"),
            value: serde_yaml::Value::Mapping(map),
        }))
    }
}

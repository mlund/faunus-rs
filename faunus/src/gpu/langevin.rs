//! GPU-accelerated rigid-body Langevin dynamics runner.
//!
//! Orchestrates the BAOAB integration pipeline:
//! 1. Pair forces → per-atom forces
//! 2. Force reduction → per-molecule forces and torques
//! 3. BAOAB integrator steps (B-A-O-A-B)
//! 4. Rigid body reconstruction → atom positions from COM + quaternion

use super::{buffers, GpuContext, SplineBuffers};
use crate::propagate::LangevinConfig;
use std::sync::Arc;

/// Push constants for the pairwise force/energy kernel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ForceUniforms {
    pub n_atoms: u32,
    pub n_atom_types: u32,
    pub box_length: f32,
    pub cutoff_sq: f32,
}

/// Push constants for the BAOAB kernels.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LangevinUniforms {
    pub n_molecules: u32,
    pub dt: f32,
    pub friction: f32,
    pub kt: f32,
    pub rng_seed: u32,
    pub rng_step: u32,
    pub box_length: f32,
    _pad: u32,
}

/// Pre-computed data for initial GPU state upload.
///
/// All vectors use GPU layout (vec4-padded f32) so they can be written
/// directly to GPU buffers without further transformation.
pub struct LangevinUploadData {
    /// Per-atom positions as [x, y, z, atom_type_bits] (n_atoms × 4)
    pub positions: Vec<f32>,
    /// Per-atom type IDs (n_atoms)
    pub atom_type_ids: Vec<u32>,
    /// Per-atom molecule IDs for intramolecular exclusions (n_atoms)
    pub atom_mol_ids: Vec<u32>,
    /// Per-atom body-frame reference positions [rx, ry, rz, 0] (n_atoms × 4)
    pub ref_positions: Vec<f32>,
    /// Per-molecule COM positions [x, y, z, 0] (n_molecules × 4)
    pub com_positions: Vec<f32>,
    /// Molecule→atom offset table (n_molecules + 1)
    pub mol_atom_offsets: Vec<u32>,
    /// Per-molecule total mass (n_molecules)
    pub mol_masses: Vec<f32>,
    /// Per-molecule diagonal inertia [Ixx, Iyy, Izz, 0] (n_molecules × 4)
    pub mol_inertia: Vec<f32>,
    /// Per-molecule quaternions [x, y, z, w] (n_molecules × 4)
    pub quaternions: Vec<f32>,
    /// Per-molecule COM velocities [vx, vy, vz, 0] (n_molecules × 4)
    pub com_velocities: Vec<f32>,
    /// Per-molecule angular velocities [wx, wy, wz, 0] (n_molecules × 4)
    pub angular_velocities: Vec<f32>,
}

/// GPU resources for the Langevin dynamics pipeline.
///
/// Some buffer/pipeline fields appear unused but must stay alive so that
/// bind groups referencing them remain valid for GPU dispatch.
#[allow(dead_code)]
pub struct LangevinGpu {
    gpu: Arc<GpuContext>,
    config: LangevinConfig,

    // Persistent GPU buffers
    positions: wgpu::Buffer,
    forces: wgpu::Buffer,
    com_positions: wgpu::Buffer,
    com_velocities: wgpu::Buffer,
    quaternions: wgpu::Buffer,
    angular_velocities: wgpu::Buffer,
    com_forces: wgpu::Buffer,
    torques: wgpu::Buffer,
    ref_positions: wgpu::Buffer,
    mol_masses: wgpu::Buffer,
    mol_inertia: wgpu::Buffer,
    atom_type_ids: wgpu::Buffer,
    atom_mol_ids: wgpu::Buffer,
    mol_atom_offsets: wgpu::Buffer,
    spline_buffers: SplineBuffers,

    // Pipelines (created once)
    force_pipeline: wgpu::ComputePipeline,
    force_bind_group: wgpu::BindGroup,
    reduce_pipeline: wgpu::ComputePipeline,
    reduce_bind_group: wgpu::BindGroup,
    langevin_pipeline: wgpu::ComputePipeline,
    half_kick_pipeline: wgpu::ComputePipeline,
    langevin_bind_group: wgpu::BindGroup,
    reconstruct_pipeline: wgpu::ComputePipeline,
    reconstruct_bind_group: wgpu::BindGroup,

    n_atoms: u32,
    n_molecules: u32,
    n_atom_types: u32,
    box_length: f32,
    cutoff_sq: f32,
    kt: f32,
    step_counter: u32,
}

impl std::fmt::Debug for LangevinGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangevinGpu")
            .field("n_atoms", &self.n_atoms)
            .field("n_molecules", &self.n_molecules)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl LangevinGpu {
    /// Create pipelines and bind groups for the Langevin dynamics pipeline.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: Arc<GpuContext>,
        config: LangevinConfig,
        spline_buffers: SplineBuffers,
        n_atoms: u32,
        n_molecules: u32,
        n_atom_types: u32,
        box_length: f32,
        cutoff: f32,
        kt: f32,
    ) -> Self {
        let device = &gpu.device;

        // Allocate all persistent buffers
        let vec4_size = 16u64; // 4 × f32
        let positions =
            buffers::storage_buffer_rw(device, "ld_positions", n_atoms as u64 * vec4_size);
        let forces = buffers::storage_buffer_rw(device, "ld_forces", n_atoms as u64 * vec4_size);
        let com_positions =
            buffers::storage_buffer_rw(device, "ld_com_pos", n_molecules as u64 * vec4_size);
        let com_velocities =
            buffers::storage_buffer_rw(device, "ld_com_vel", n_molecules as u64 * vec4_size);
        let quaternions =
            buffers::storage_buffer_rw(device, "ld_quaternions", n_molecules as u64 * vec4_size);
        let angular_velocities =
            buffers::storage_buffer_rw(device, "ld_omega", n_molecules as u64 * vec4_size);
        let com_forces =
            buffers::storage_buffer_rw(device, "ld_com_forces", n_molecules as u64 * vec4_size);
        let torques =
            buffers::storage_buffer_rw(device, "ld_torques", n_molecules as u64 * vec4_size);
        let ref_positions =
            buffers::storage_buffer_rw(device, "ld_ref_positions", n_atoms as u64 * vec4_size);
        let mol_masses = buffers::storage_buffer_rw(
            device,
            "ld_mol_masses",
            n_molecules as u64 * 4, // f32 per molecule
        );
        let mol_inertia = buffers::storage_buffer_rw(
            device,
            "ld_mol_inertia",
            n_molecules as u64 * vec4_size, // vec4<f32> (Ixx, Iyy, Izz, 0)
        );
        let atom_type_ids = buffers::storage_buffer_rw(
            device,
            "ld_atom_types",
            n_atoms as u64 * 4, // u32 per atom
        );
        let atom_mol_ids = buffers::storage_buffer_rw(
            device,
            "ld_atom_mol_ids",
            n_atoms as u64 * 4, // u32 per atom: which molecule each atom belongs to
        );
        let mol_atom_offsets = buffers::storage_buffer_rw(
            device,
            "ld_mol_offsets",
            (n_molecules as u64 + 1) * 4, // u32 prefix sums
        );

        // Load shaders
        let force_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pair_forces"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pair_forces.wgsl").into()),
        });
        let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("force_reduce"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/force_reduce.wgsl").into()),
        });
        let langevin_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rigid_langevin"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rigid_langevin.wgsl").into()),
        });
        let reconstruct_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rigid_reconstruct"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rigid_reconstruct.wgsl").into()),
        });

        // --- Force pipeline ---
        let force_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("force_bgl"),
            entries: &[
                buffers::bgl_storage(0, true),  // positions
                buffers::bgl_storage(1, false), // forces (output)
                buffers::bgl_storage(2, true),  // atom_type_ids
                buffers::bgl_storage(3, true),  // spline_params
                buffers::bgl_storage(4, true),  // spline_coeffs
                buffers::bgl_storage(5, true),  // atom_mol_ids (for exclusions)
            ],
        });
        let force_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("force_pl"),
                bind_group_layouts: &[&force_bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..std::mem::size_of::<ForceUniforms>() as u32,
                }],
            });
        let force_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("force_pipeline"),
            layout: Some(&force_pipeline_layout),
            module: &force_shader,
            entry_point: Some("compute_forces"),
            compilation_options: Default::default(),
            cache: None,
        });
        let force_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("force_bg"),
            layout: &force_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: forces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: atom_type_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: spline_buffers.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: spline_buffers.coeffs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: atom_mol_ids.as_entire_binding(),
                },
            ],
        });

        // --- Force reduction pipeline ---
        let reduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reduce_bgl"),
            entries: &[
                buffers::bgl_storage(0, true),  // forces (per-atom)
                buffers::bgl_storage(1, true),  // positions
                buffers::bgl_storage(2, true),  // com_positions
                buffers::bgl_storage(3, false), // com_forces (output)
                buffers::bgl_storage(4, false), // torques (output)
                buffers::bgl_storage(5, true),  // mol_atom_offsets
            ],
        });
        let reduce_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("reduce_pl"),
                bind_group_layouts: &[&reduce_bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..std::mem::size_of::<[u32; 2]>() as u32,
                }],
            });
        let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reduce_pipeline"),
            layout: Some(&reduce_pipeline_layout),
            module: &reduce_shader,
            entry_point: Some("reduce_forces"),
            compilation_options: Default::default(),
            cache: None,
        });
        let reduce_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reduce_bg"),
            layout: &reduce_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: forces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: com_positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: com_forces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: torques.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: mol_atom_offsets.as_entire_binding(),
                },
            ],
        });

        // --- Langevin BAOAB pipeline ---
        let langevin_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("langevin_bgl"),
            entries: &[
                buffers::bgl_storage(0, false), // com_positions
                buffers::bgl_storage(1, false), // com_velocities
                buffers::bgl_storage(2, false), // quaternions
                buffers::bgl_storage(3, false), // angular_velocities
                buffers::bgl_storage(4, true),  // com_forces
                buffers::bgl_storage(5, true),  // torques
                buffers::bgl_storage(6, true),  // mol_masses
                buffers::bgl_storage(7, true),  // mol_inertia
            ],
        });
        let langevin_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("langevin_pl"),
                bind_group_layouts: &[&langevin_bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..std::mem::size_of::<LangevinUniforms>() as u32,
                }],
            });
        let langevin_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("langevin_pipeline"),
            layout: Some(&langevin_pipeline_layout),
            module: &langevin_shader,
            entry_point: Some("baoab_step"),
            compilation_options: Default::default(),
            cache: None,
        });
        let half_kick_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("half_kick_pipeline"),
            layout: Some(&langevin_pipeline_layout),
            module: &langevin_shader,
            entry_point: Some("half_kick"),
            compilation_options: Default::default(),
            cache: None,
        });
        let langevin_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("langevin_bg"),
            layout: &langevin_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: com_positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: com_velocities.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: quaternions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: angular_velocities.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: com_forces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: torques.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: mol_masses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: mol_inertia.as_entire_binding(),
                },
            ],
        });

        // --- Reconstruct pipeline ---
        let reconstruct_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reconstruct_bgl"),
            entries: &[
                buffers::bgl_storage(0, true),  // com_positions
                buffers::bgl_storage(1, true),  // quaternions
                buffers::bgl_storage(2, true),  // ref_positions
                buffers::bgl_storage(3, false), // positions (output)
                buffers::bgl_storage(4, true),  // mol_atom_offsets
            ],
        });
        let reconstruct_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("reconstruct_pl"),
                bind_group_layouts: &[&reconstruct_bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..std::mem::size_of::<[u32; 2]>() as u32,
                }],
            });
        let reconstruct_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("reconstruct_pipeline"),
                layout: Some(&reconstruct_pipeline_layout),
                module: &reconstruct_shader,
                entry_point: Some("reconstruct_positions"),
                compilation_options: Default::default(),
                cache: None,
            });
        let reconstruct_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reconstruct_bg"),
            layout: &reconstruct_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: com_positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: quaternions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ref_positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: mol_atom_offsets.as_entire_binding(),
                },
            ],
        });

        Self {
            gpu,
            config,
            positions,
            forces,
            com_positions,
            com_velocities,
            quaternions,
            angular_velocities,
            com_forces,
            torques,
            ref_positions,
            mol_masses,
            mol_inertia,
            atom_type_ids,
            atom_mol_ids,
            mol_atom_offsets,
            spline_buffers,
            force_pipeline,
            force_bind_group,
            reduce_pipeline,
            reduce_bind_group,
            langevin_pipeline,
            half_kick_pipeline,
            langevin_bind_group,
            reconstruct_pipeline,
            reconstruct_bind_group,
            n_atoms,
            n_molecules,
            n_atom_types,
            box_length,
            cutoff_sq: cutoff * cutoff,
            kt,
            step_counter: 0,
        }
    }

    pub fn kt(&self) -> f32 {
        self.kt
    }

    /// Write initial particle and molecule state to GPU buffers.
    pub fn upload_state(&self, data: &LangevinUploadData) {
        let q = &self.gpu.queue;
        q.write_buffer(&self.positions, 0, bytemuck::cast_slice(&data.positions));
        q.write_buffer(
            &self.atom_type_ids,
            0,
            bytemuck::cast_slice(&data.atom_type_ids),
        );
        q.write_buffer(
            &self.atom_mol_ids,
            0,
            bytemuck::cast_slice(&data.atom_mol_ids),
        );
        q.write_buffer(
            &self.ref_positions,
            0,
            bytemuck::cast_slice(&data.ref_positions),
        );
        q.write_buffer(
            &self.com_positions,
            0,
            bytemuck::cast_slice(&data.com_positions),
        );
        q.write_buffer(
            &self.mol_atom_offsets,
            0,
            bytemuck::cast_slice(&data.mol_atom_offsets),
        );
        q.write_buffer(&self.mol_masses, 0, bytemuck::cast_slice(&data.mol_masses));
        q.write_buffer(
            &self.mol_inertia,
            0,
            bytemuck::cast_slice(&data.mol_inertia),
        );
        q.write_buffer(
            &self.quaternions,
            0,
            bytemuck::cast_slice(&data.quaternions),
        );
        q.write_buffer(
            &self.com_velocities,
            0,
            bytemuck::cast_slice(&data.com_velocities),
        );
        q.write_buffer(
            &self.angular_velocities,
            0,
            bytemuck::cast_slice(&data.angular_velocities),
        );
    }

    /// Run `steps` BAOAB integration steps entirely on GPU.
    ///
    /// Full BAOAB per step: B(F_old) → A → O → A → reconstruct → F_new → B(F_new)
    /// The closing B of step N and opening B of step N+1 share the same forces,
    /// giving the correct merged velocity kick.
    pub fn run_steps(&mut self, steps: usize) {
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;
        let wg_atoms = self.n_atoms.div_ceil(64);
        let wg_mols = self.n_molecules.div_ceil(64);

        let fu = ForceUniforms {
            n_atoms: self.n_atoms,
            n_atom_types: self.n_atom_types,
            box_length: self.box_length,
            cutoff_sq: self.cutoff_sq,
        };
        let reduce_push = [self.n_molecules, self.box_length.to_bits()];
        let reconstruct_push = [self.n_atoms, self.n_molecules];

        if self.step_counter == 0 {
            log::info!(
                "GPU Langevin: box_length={}, dt={}, friction={}, kT={}, n_mol={}, n_atoms={}",
                self.box_length,
                self.config.timestep,
                self.config.friction,
                self.kt,
                self.n_molecules,
                self.n_atoms
            );
        }

        // Initial force computation at current positions
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("initial_forces"),
            });
            Self::encode_forces(&mut encoder, self, &fu);
            Self::encode_reduce(&mut encoder, self, &reduce_push);
            queue.submit(Some(encoder.finish()));
        }

        let mut lu = LangevinUniforms {
            n_molecules: self.n_molecules,
            dt: self.config.timestep as f32,
            friction: self.config.friction as f32,
            kt: self.kt,
            rng_seed: 0xDEAD_BEEF,
            rng_step: 0,
            box_length: self.box_length,
            _pad: 0,
        };

        for _ in 0..steps {
            lu.rng_step = self.step_counter;

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("baoab_step"),
            });

            // B-A-O-A: opening half-kick + drift + thermostat + drift
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.langevin_pipeline);
                pass.set_bind_group(0, &self.langevin_bind_group, &[]);
                pass.set_push_constants(0, bytemuck::bytes_of(&lu));
                pass.dispatch_workgroups(wg_mols, 1, 1);
            }

            // Reconstruct atom positions from COM + quaternion
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.reconstruct_pipeline);
                pass.set_bind_group(0, &self.reconstruct_bind_group, &[]);
                pass.set_push_constants(0, bytemuck::cast_slice(&reconstruct_push));
                pass.dispatch_workgroups(wg_atoms, 1, 1);
            }

            // Forces at new positions
            Self::encode_forces(&mut encoder, self, &fu);
            Self::encode_reduce(&mut encoder, self, &reduce_push);

            // Closing B: half-kick with new forces
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.half_kick_pipeline);
                pass.set_bind_group(0, &self.langevin_bind_group, &[]);
                pass.set_push_constants(0, bytemuck::bytes_of(&lu));
                pass.dispatch_workgroups(wg_mols, 1, 1);
            }

            queue.submit(Some(encoder.finish()));
            self.step_counter += 1;
        }

        device.poll(wgpu::Maintain::Wait);
    }

    fn readback(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        super::readback_f32(&self.gpu.device, &self.gpu.queue, buffer, count)
    }

    /// Takes `&Self` instead of `&self` to allow calling from `run_steps(&mut self)`
    /// without conflicting borrows on `self.step_counter`.
    fn encode_forces(encoder: &mut wgpu::CommandEncoder, s: &Self, fu: &ForceUniforms) {
        let wg_atoms = s.n_atoms.div_ceil(64);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&s.force_pipeline);
        pass.set_bind_group(0, &s.force_bind_group, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(fu));
        pass.dispatch_workgroups(wg_atoms, 1, 1);
    }

    fn encode_reduce(encoder: &mut wgpu::CommandEncoder, s: &Self, push: &[u32; 2]) {
        let wg_mols = s.n_molecules.div_ceil(64);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&s.reduce_pipeline);
        pass.set_bind_group(0, &s.reduce_bind_group, &[]);
        pass.set_push_constants(0, bytemuck::cast_slice(push));
        pass.dispatch_workgroups(wg_mols, 1, 1);
    }

    /// Compute translational and rotational temperatures from GPU velocities.
    ///
    /// Returns (T_trans, T_rot) in Kelvin using equipartition:
    ///   T = (2 * KE) / (n_dof * kB)
    pub fn download_temperature(&self) -> (f32, f32) {
        let n = self.n_molecules as usize;
        let vel = self.readback(&self.com_velocities, n * 4);
        let omega = self.readback(&self.angular_velocities, n * 4);
        let masses = self.readback(&self.mol_masses, n);
        let inertia = self.readback(&self.mol_inertia, n * 4);

        // KE_trans = sum 0.5 * M * |v|^2 in amu·Å²/ps²
        let ke_trans: f32 = (0..n)
            .map(|m| {
                let vx = vel[m * 4];
                let vy = vel[m * 4 + 1];
                let vz = vel[m * 4 + 2];
                0.5 * masses[m] * (vx * vx + vy * vy + vz * vz)
            })
            .sum();

        // KE_rot = sum 0.5 * omega_i * I_i * omega_i (diagonal inertia tensor)
        let ke_rot: f32 = (0..n)
            .map(|m| {
                let wx = omega[m * 4];
                let wy = omega[m * 4 + 1];
                let wz = omega[m * 4 + 2];
                let ix = inertia[m * 4];
                let iy = inertia[m * 4 + 1];
                let iz = inertia[m * 4 + 2];
                0.5 * (ix * wx * wx + iy * wy * wy + iz * wz * wz)
            })
            .sum();

        // amu·Å²/ps² → kJ/mol: divide by 100; T = 2*KE / (3*N * R)
        const R_KJ_PER_MOL_K: f32 = physical_constants::MOLAR_GAS_CONSTANT as f32 * 1e-3;
        const KJ_MOL_TO_INTERNAL: f32 = 100.0;
        let dof = 3.0 * n as f32;
        let t_trans = 2.0 * ke_trans / (KJ_MOL_TO_INTERNAL * dof * R_KJ_PER_MOL_K);
        let t_rot = 2.0 * ke_rot / (KJ_MOL_TO_INTERNAL * dof * R_KJ_PER_MOL_K);
        (t_trans, t_rot)
    }

    /// Download a GPU buffer as a vector of `[f32; 4]` elements.
    fn readback_vec4(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<[f32; 4]> {
        self.readback(buffer, count * 4)
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect()
    }

    /// Download atom positions from GPU (vec4 per atom: x, y, z, atom_type_bits).
    pub fn download_positions(&self) -> Vec<[f32; 4]> {
        self.readback_vec4(&self.positions, self.n_atoms as usize)
    }

    /// Download per-molecule quaternions [x, y, z, w] from GPU.
    pub fn download_quaternions(&self) -> Vec<[f32; 4]> {
        self.readback_vec4(&self.quaternions, self.n_molecules as usize)
    }

    /// Upload only positions, COM positions, and quaternions (velocities untouched).
    ///
    /// Used on MC→LD transitions to sync atom positions and rigid-body orientations
    /// that may have changed during MC moves, without disturbing GPU velocities.
    pub fn upload_positions_com_quaternions(
        &self,
        positions: &[f32],
        com_positions: &[f32],
        quaternions: &[f32],
    ) {
        debug_assert_eq!(positions.len(), self.n_atoms as usize * 4);
        debug_assert_eq!(com_positions.len(), self.n_molecules as usize * 4);
        debug_assert_eq!(quaternions.len(), self.n_molecules as usize * 4);
        let q = &self.gpu.queue;
        q.write_buffer(&self.positions, 0, bytemuck::cast_slice(positions));
        q.write_buffer(&self.com_positions, 0, bytemuck::cast_slice(com_positions));
        q.write_buffer(&self.quaternions, 0, bytemuck::cast_slice(quaternions));
    }
}

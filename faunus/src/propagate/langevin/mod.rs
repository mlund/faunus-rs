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

#[cfg(feature = "gpu")]
mod kernels;

#[cfg(feature = "gpu")]
use crate::cell::{BoundaryConditions, Shape};
#[cfg(feature = "gpu")]
use crate::topology::{DegreesOfFreedom, Topology};
use crate::Context;
#[cfg(feature = "gpu")]
use average::Estimate;
use average::Variance;
#[cfg(feature = "gpu")]
use cubecl::prelude::*;
#[cfg(feature = "gpu")]
use cubecl::server::Handle;
#[cfg(feature = "gpu")]
use rand::Rng;
use serde::{Deserialize, Serialize};
#[cfg(feature = "gpu")]
use std::marker::PhantomData;

#[cfg(feature = "gpu")]
const WORKGROUP_SIZE: u32 = 64;

/// Atom mass from topology, falling back to 1.0 for out-of-range indices.
#[cfg(feature = "gpu")]
fn atom_mass(topology: &Topology, atom_indices: &[usize], idx: usize) -> f32 {
    atom_indices
        .get(idx)
        .map(|&ai| topology.atomkinds()[ai].mass() as f32)
        .unwrap_or(1.0)
}

/// GPU spline data extracted from the Hamiltonian: (atom_type_ids, mol_ids, params, coeffs, n_types).
#[cfg(feature = "gpu")]
type SplineData = (
    Option<Vec<u32>>,
    Option<Vec<u32>>,
    Option<Vec<f32>>,
    Option<Vec<f32>>,
    u32,
);

/// Callback computing per-molecule COM forces and torques from atom positions.
#[cfg(feature = "gpu")]
type ForceCallback<'a> = &'a mut dyn FnMut(&[[f32; 4]]) -> (Vec<[f32; 4]>, Vec<[f32; 4]>);

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
}

/// Pre-computed data for initial state upload.
///
/// All vectors use vec4-padded f32 layout so they can be written
/// directly to device buffers without further transformation.
#[cfg(feature = "gpu")]
struct LangevinUploadData {
    positions: Vec<f32>,
    ref_positions: Vec<f32>,
    com_positions: Vec<f32>,
    mol_atom_offsets: Vec<u32>,
    mol_masses: Vec<f32>,
    mol_inertia: Vec<f32>,
    quaternions: Vec<f32>,
    com_velocities: Vec<f32>,
    angular_velocities: Vec<f32>,
    /// Per-atom type indices for spline lookup (None = CPU forces only).
    atom_type_ids: Option<Vec<u32>>,
    /// Per-atom molecule index for intra-molecular exclusion.
    mol_ids: Option<Vec<u32>>,
    /// Flat spline params array (stride 8 per type pair).
    spline_params: Option<Vec<f32>>,
    /// Flat spline coefficients array (stride 8 per interval).
    spline_coeffs: Option<Vec<f32>>,
    /// Number of atom types (matrix dimension for spline_params).
    n_atom_types: u32,
    /// CSR bond data.
    bond_data: Option<crate::energy::bonded::kernel::CsrData>,
    /// CSR angle data.
    angle_data: Option<crate::energy::bonded::kernel::CsrData>,
    /// CSR dihedral data.
    dihedral_data: Option<crate::energy::bonded::kernel::CsrData>,
    /// Per-molecule flag: 1 = rigid-body integration, 0 = per-atom or frozen.
    mol_is_rigid: Vec<u32>,
    /// Per-atom velocities for flexible atoms (stride 3: vx, vy, vz).
    atom_velocities: Vec<f32>,
    /// Per-atom masses for flexible integration.
    atom_masses: Vec<f32>,
    /// Per-atom flag: 1 = flexible (per-atom BAOAB), 0 = rigid or frozen.
    atom_is_flexible: Vec<u32>,
    /// Whether any flexible atoms exist.
    has_flexible: bool,
}

/// CubeCL-based Langevin dynamics pipeline, generic over runtime backend.
#[cfg(feature = "gpu")]
struct LangevinGpu<R: Runtime> {
    client: ComputeClient<R>,
    config: LangevinConfig,

    // Integrator state buffers
    positions: Handle,
    com_positions: Handle,
    com_velocities: Handle,
    quaternions: Handle,
    angular_velocities: Handle,
    com_forces: Handle,
    torques: Handle,
    ref_positions: Handle,
    mol_masses: Handle,
    mol_inertia: Handle,
    mol_atom_offsets: Handle,

    // Force computation buffers (populated when GPU forces enabled)
    atom_forces: Handle,
    atom_type_ids: Handle,
    mol_ids: Handle,
    spline_params: Handle,
    spline_coeffs: Handle,
    n_atom_types: u32,
    has_gpu_forces: bool,

    // Bonded force CSR buffers — separate from nonbonded because bonded
    // kernels accumulate (`+=`) on top of pair forces already in atom_forces.
    bond_offsets: Handle,
    bond_atoms: Handle,
    bond_params: Handle,
    has_bonds: bool,
    angle_offsets: Handle,
    angle_atoms: Handle,
    angle_params: Handle,
    has_angles: bool,
    dihedral_offsets: Handle,
    dihedral_atoms: Handle,
    dihedral_params: Handle,
    has_dihedrals: bool,

    // Per-atom flexible integration buffers
    mol_is_rigid: Handle,
    atom_velocities: Handle,
    atom_masses_buf: Handle,
    atom_is_flexible: Handle,
    has_flexible: bool,
    has_rigid: bool,

    // Host-side copies of constant buffers (avoids GPU readbacks in download_temperature)
    mol_is_rigid_host: Vec<u32>,
    mol_masses_host: Vec<f32>,
    mol_inertia_host: Vec<f32>,
    atom_masses_host: Vec<f32>,
    atom_is_flexible_host: Vec<u32>,

    n_atoms: u32,
    n_molecules: u32,
    box_length: f32,
    kt: f32,
    step_counter: u32,
    _runtime: PhantomData<R>,
}

#[cfg(feature = "gpu")]
impl<R: Runtime> std::fmt::Debug for LangevinGpu<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangevinGpu")
            .field("n_atoms", &self.n_atoms)
            .field("n_molecules", &self.n_molecules)
            .field("has_gpu_forces", &self.has_gpu_forces)
            .field("has_flexible", &self.has_flexible)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "gpu")]
impl<R: Runtime> LangevinGpu<R> {
    fn new(
        client: ComputeClient<R>,
        config: LangevinConfig,
        n_atoms: u32,
        n_molecules: u32,
        box_length: f32,
        kt: f32,
    ) -> Self {
        let vec4_bytes = 16usize;

        let positions = client.empty(n_atoms as usize * vec4_bytes);
        let com_positions = client.empty(n_molecules as usize * vec4_bytes);
        let com_velocities = client.empty(n_molecules as usize * vec4_bytes);
        let quaternions = client.empty(n_molecules as usize * vec4_bytes);
        let angular_velocities = client.empty(n_molecules as usize * vec4_bytes);
        let com_forces = client.empty(n_molecules as usize * vec4_bytes);
        let torques = client.empty(n_molecules as usize * vec4_bytes);
        let ref_positions = client.empty(n_atoms as usize * vec4_bytes);
        let mol_masses = client.empty(n_molecules as usize * 4);
        let mol_inertia = client.empty(n_molecules as usize * vec4_bytes);
        let mol_atom_offsets = client.empty((n_molecules as usize + 1) * 4);

        // Force buffers (empty placeholders until spline data is uploaded)
        let atom_forces = client.empty(n_atoms as usize * vec4_bytes);
        let atom_type_ids = client.empty(4); // placeholder
        let mol_ids = client.empty(4);
        let spline_params = client.empty(4);
        let spline_coeffs = client.empty(4);

        // Minimal placeholders — real CSR data is uploaded later if bonded terms exist
        let bond_offsets = client.empty(4);
        let bond_atoms = client.empty(4);
        let bond_params = client.empty(4);
        let angle_offsets = client.empty(4);
        let angle_atoms = client.empty(4);
        let angle_params = client.empty(4);
        let dihedral_offsets = client.empty(4);
        let dihedral_atoms = client.empty(4);
        let dihedral_params = client.empty(4);

        // Placeholders for flexible integration buffers
        let mol_is_rigid = client.empty(n_molecules as usize * 4);
        let atom_velocities = client.empty(n_atoms as usize * 3 * 4);
        let atom_masses_buf = client.empty(n_atoms as usize * 4);
        let atom_is_flexible = client.empty(n_atoms as usize * 4);

        Self {
            client,
            config,
            positions,
            com_positions,
            com_velocities,
            quaternions,
            angular_velocities,
            com_forces,
            torques,
            ref_positions,
            mol_masses,
            mol_inertia,
            mol_atom_offsets,
            atom_forces,
            atom_type_ids,
            mol_ids,
            spline_params,
            spline_coeffs,
            n_atom_types: 0,
            has_gpu_forces: false,
            bond_offsets,
            bond_atoms,
            bond_params,
            has_bonds: false,
            angle_offsets,
            angle_atoms,
            angle_params,
            has_angles: false,
            dihedral_offsets,
            dihedral_atoms,
            dihedral_params,
            has_dihedrals: false,
            mol_is_rigid,
            atom_velocities,
            atom_masses_buf,
            atom_is_flexible,
            has_flexible: false,
            has_rigid: true,
            mol_is_rigid_host: Vec::new(),
            mol_masses_host: Vec::new(),
            mol_inertia_host: Vec::new(),
            atom_masses_host: Vec::new(),
            atom_is_flexible_host: Vec::new(),
            n_atoms,
            n_molecules,
            box_length,
            kt,
            step_counter: 0,
            _runtime: PhantomData,
        }
    }

    fn upload_state(&mut self, data: LangevinUploadData) {
        self.positions = self
            .client
            .create_from_slice(bytemuck::cast_slice(&data.positions));
        self.ref_positions = self
            .client
            .create_from_slice(bytemuck::cast_slice(&data.ref_positions));
        self.com_positions = self
            .client
            .create_from_slice(bytemuck::cast_slice(&data.com_positions));
        self.mol_atom_offsets = self
            .client
            .create_from_slice(bytemuck::cast_slice(&data.mol_atom_offsets));
        self.mol_masses = self
            .client
            .create_from_slice(bytemuck::cast_slice(&data.mol_masses));
        self.mol_inertia = self
            .client
            .create_from_slice(bytemuck::cast_slice(&data.mol_inertia));
        self.quaternions = self
            .client
            .create_from_slice(bytemuck::cast_slice(&data.quaternions));
        self.com_velocities = self
            .client
            .create_from_slice(bytemuck::cast_slice(&data.com_velocities));
        self.angular_velocities = self
            .client
            .create_from_slice(bytemuck::cast_slice(&data.angular_velocities));

        // Upload spline force data if provided
        if let (Some(types), Some(mol_ids), Some(params), Some(coeffs)) = (
            &data.atom_type_ids,
            &data.mol_ids,
            &data.spline_params,
            &data.spline_coeffs,
        ) {
            self.atom_type_ids = self.client.create_from_slice(bytemuck::cast_slice(types));
            self.mol_ids = self.client.create_from_slice(bytemuck::cast_slice(mol_ids));
            self.spline_params = self.client.create_from_slice(bytemuck::cast_slice(params));
            self.spline_coeffs = self.client.create_from_slice(bytemuck::cast_slice(coeffs));
            self.n_atom_types = data.n_atom_types;
            self.has_gpu_forces = true;
            log::info!(
                "On-device forces enabled: {} atom types, {} spline params, {} spline coeffs",
                data.n_atom_types,
                params.len() / 8,
                coeffs.len() / 8,
            );
        }

        // Upload bonded CSR data
        let upload_csr = |client: &ComputeClient<R>,
                          data: &crate::energy::bonded::kernel::CsrData| {
            (
                client.create_from_slice(bytemuck::cast_slice(&data.offsets)),
                client.create_from_slice(bytemuck::cast_slice(&data.atoms)),
                client.create_from_slice(bytemuck::cast_slice(&data.params)),
            )
        };
        if let Some(csr) = &data.bond_data {
            (self.bond_offsets, self.bond_atoms, self.bond_params) = upload_csr(&self.client, csr);
            self.has_bonds = true;
            log::info!("On-device bond forces: {} entries", csr.atoms.len());
        }
        if let Some(csr) = &data.angle_data {
            (self.angle_offsets, self.angle_atoms, self.angle_params) =
                upload_csr(&self.client, csr);
            self.has_angles = true;
            log::info!("On-device angle forces: {} entries", csr.atoms.len() / 2);
        }
        if let Some(csr) = &data.dihedral_data {
            (
                self.dihedral_offsets,
                self.dihedral_atoms,
                self.dihedral_params,
            ) = upload_csr(&self.client, csr);
            self.has_dihedrals = true;
            log::info!("On-device dihedral forces: {} entries", csr.atoms.len() / 3);
        }

        // Upload rigid/flexible classification and per-atom integration buffers
        self.mol_is_rigid = self
            .client
            .create_from_slice(bytemuck::cast_slice(&data.mol_is_rigid));
        self.atom_is_flexible = self
            .client
            .create_from_slice(bytemuck::cast_slice(&data.atom_is_flexible));
        self.atom_masses_buf = self
            .client
            .create_from_slice(bytemuck::cast_slice(&data.atom_masses));
        self.atom_velocities = self
            .client
            .create_from_slice(bytemuck::cast_slice(&data.atom_velocities));
        self.has_flexible = data.has_flexible;

        let n_rigid = data.mol_is_rigid.iter().filter(|&&r| r != 0).count();
        self.has_rigid = n_rigid > 0;

        let n_flex_atoms = data.atom_is_flexible.iter().filter(|&&f| f != 0).count();
        let n_mol = data.mol_is_rigid.len();
        log::info!(
            "Molecule classification: {n_rigid} rigid, {} flexible/frozen ({n_flex_atoms} flexible atoms)",
            n_mol - n_rigid,
        );

        // Cache constant buffers on host to avoid GPU readbacks in download_temperature.
        // Moved (not cloned) since upload_state consumes the data.
        self.mol_is_rigid_host = data.mol_is_rigid;
        self.mol_masses_host = data.mol_masses;
        self.mol_inertia_host = data.mol_inertia;
        self.atom_masses_host = data.atom_masses;
        self.atom_is_flexible_host = data.atom_is_flexible;
    }

    /// Run `steps` BAOAB steps with on-device force computation.
    ///
    /// Forces, reduction, integration, and reconstruction all run on the compute
    /// device without any host readback during the integration loop.
    /// Rigid and flexible integration paths operate on disjoint atom ranges.
    fn run_steps(&mut self, steps: usize) {
        assert!(self.has_gpu_forces, "on-device forces not initialized");
        self.log_first_step();

        // Nonbonded writes atom_forces (=), then bonded accumulates (+=),
        // then reduce sums rigid per-atom forces into per-molecule COM forces and torques.
        self.dispatch_pair_forces();
        self.dispatch_bonded_forces();
        self.dispatch_reduce();

        for _ in 0..steps {
            self.dispatch_baoab();
            self.dispatch_baoab_atoms();
            self.dispatch_reconstruct();

            self.dispatch_pair_forces();
            self.dispatch_bonded_forces();
            self.dispatch_reduce();

            self.dispatch_half_kick();
            self.dispatch_half_kick_atoms();
            self.step_counter += 1;
        }

        // Sync to ensure all work is complete
        let _ = self.client.read_one(self.com_forces.clone());
    }

    /// Run `steps` BAOAB steps with CPU-computed forces from a callback.
    ///
    /// CPU forces path provides COM forces/torques directly; flexible atoms
    /// are not supported in this path (requires on-device forces).
    fn run_steps_with_cpu_forces(&mut self, steps: usize, compute_forces: ForceCallback<'_>) {
        self.log_first_step();

        let positions = self.download_positions();
        let (com_forces, torques) = compute_forces(&positions);
        self.upload_com_forces_torques(&com_forces, &torques);

        for _ in 0..steps {
            self.dispatch_baoab();
            self.dispatch_reconstruct();

            let positions = self.download_positions();
            let (com_forces, torques) = compute_forces(&positions);
            self.upload_com_forces_torques(&com_forces, &torques);

            self.dispatch_half_kick();
            self.step_counter += 1;
        }

        let _ = self.client.read_one(self.com_forces.clone());
    }

    fn upload_com_forces_torques(&mut self, com_forces: &[[f32; 4]], torques: &[[f32; 4]]) {
        self.com_forces = self
            .client
            .create_from_slice(bytemuck::cast_slice(com_forces));
        self.torques = self.client.create_from_slice(bytemuck::cast_slice(torques));
    }

    // ========================================================================
    // Kernel dispatch helpers
    // ========================================================================

    fn dispatch_pair_forces(&self) {
        let count = CubeCount::Static(self.n_atoms.div_ceil(WORKGROUP_SIZE), 1, 1);
        let dim = CubeDim::new_1d(WORKGROUP_SIZE);
        let inv_box = 1.0f32 / self.box_length;

        unsafe {
            crate::energy::nonbonded_kernel::pair_forces_kernel::launch_unchecked::<R>(
                &self.client,
                count,
                dim,
                ArrayArg::from_raw_parts::<f32>(&self.positions, self.n_atoms as usize * 4, 1),
                ArrayArg::from_raw_parts::<u32>(&self.atom_type_ids, self.n_atoms as usize, 1),
                ArrayArg::from_raw_parts::<u32>(&self.mol_ids, self.n_atoms as usize, 1),
                ArrayArg::from_raw_parts::<f32>(
                    &self.spline_params,
                    (self.n_atom_types * self.n_atom_types) as usize * 8,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(
                    &self.spline_coeffs,
                    self.spline_coeffs.size() as usize / 4,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(&self.atom_forces, self.n_atoms as usize * 4, 1),
                ScalarArg::new(self.n_atoms),
                ScalarArg::new(self.n_atom_types),
                ScalarArg::new(self.box_length),
                ScalarArg::new(inv_box),
            )
        }
        .expect("pair_forces launch failed");
    }

    /// Dispatch bond, angle, and dihedral force kernels.
    ///
    /// Must run after `dispatch_pair_forces` because the bonded kernels
    /// accumulate (`+=`) into the same `atom_forces` buffer that the
    /// nonbonded kernel initializes (`=`).
    fn dispatch_bonded_forces(&self) {
        let inv_box = 1.0f32 / self.box_length;
        // CubeCount isn't Copy, so we recreate per-launch
        let atom_count = || CubeCount::Static(self.n_atoms.div_ceil(WORKGROUP_SIZE), 1, 1);
        let dim = || CubeDim::new_1d(WORKGROUP_SIZE);

        if self.has_bonds {
            let n_bond_atoms = self.bond_atoms.size() as usize / 4;
            let n_params = self.bond_params.size() as usize / 4;
            unsafe {
                crate::energy::bonded::kernel::bond_forces_kernel::launch_unchecked::<R>(
                    &self.client,
                    atom_count(),
                    dim(),
                    ArrayArg::from_raw_parts::<f32>(&self.positions, self.n_atoms as usize * 4, 1),
                    ArrayArg::from_raw_parts::<u32>(
                        &self.bond_offsets,
                        self.n_atoms as usize + 1,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<u32>(&self.bond_atoms, n_bond_atoms, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.bond_params, n_params, 1),
                    ArrayArg::from_raw_parts::<f32>(
                        &self.atom_forces,
                        self.n_atoms as usize * 4,
                        1,
                    ),
                    ScalarArg::new(self.n_atoms),
                    ScalarArg::new(self.box_length),
                    ScalarArg::new(inv_box),
                )
            }
            .expect("bond_forces launch failed");
        }

        if self.has_angles {
            let n_atoms_idx = self.angle_atoms.size() as usize / 4;
            let n_params = self.angle_params.size() as usize / 4;
            unsafe {
                crate::energy::bonded::kernel::angle_forces_kernel::launch_unchecked::<R>(
                    &self.client,
                    atom_count(),
                    dim(),
                    ArrayArg::from_raw_parts::<f32>(&self.positions, self.n_atoms as usize * 4, 1),
                    ArrayArg::from_raw_parts::<u32>(
                        &self.angle_offsets,
                        self.n_atoms as usize + 1,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<u32>(&self.angle_atoms, n_atoms_idx, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.angle_params, n_params, 1),
                    ArrayArg::from_raw_parts::<f32>(
                        &self.atom_forces,
                        self.n_atoms as usize * 4,
                        1,
                    ),
                    ScalarArg::new(self.n_atoms),
                    ScalarArg::new(self.box_length),
                    ScalarArg::new(inv_box),
                )
            }
            .expect("angle_forces launch failed");
        }

        if self.has_dihedrals {
            let n_atoms_idx = self.dihedral_atoms.size() as usize / 4;
            let n_params = self.dihedral_params.size() as usize / 4;
            unsafe {
                crate::energy::bonded::kernel::dihedral_forces_kernel::launch_unchecked::<R>(
                    &self.client,
                    atom_count(),
                    dim(),
                    ArrayArg::from_raw_parts::<f32>(&self.positions, self.n_atoms as usize * 4, 1),
                    ArrayArg::from_raw_parts::<u32>(
                        &self.dihedral_offsets,
                        self.n_atoms as usize + 1,
                        1,
                    ),
                    ArrayArg::from_raw_parts::<u32>(&self.dihedral_atoms, n_atoms_idx, 1),
                    ArrayArg::from_raw_parts::<f32>(&self.dihedral_params, n_params, 1),
                    ArrayArg::from_raw_parts::<f32>(
                        &self.atom_forces,
                        self.n_atoms as usize * 4,
                        1,
                    ),
                    ScalarArg::new(self.n_atoms),
                    ScalarArg::new(self.box_length),
                    ScalarArg::new(inv_box),
                )
            }
            .expect("dihedral_forces launch failed");
        }
    }

    fn dispatch_reduce(&self) {
        let count = CubeCount::Static(self.n_molecules.div_ceil(WORKGROUP_SIZE), 1, 1);
        let dim = CubeDim::new_1d(WORKGROUP_SIZE);
        let inv_box = 1.0f32 / self.box_length;

        unsafe {
            kernels::reduce_forces_kernel::launch_unchecked::<R>(
                &self.client,
                count,
                dim,
                ArrayArg::from_raw_parts::<f32>(&self.atom_forces, self.n_atoms as usize * 4, 1),
                ArrayArg::from_raw_parts::<f32>(&self.positions, self.n_atoms as usize * 4, 1),
                ArrayArg::from_raw_parts::<f32>(
                    &self.com_positions,
                    self.n_molecules as usize * 4,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(
                    &self.mol_atom_offsets,
                    self.n_molecules as usize + 1,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(&self.com_forces, self.n_molecules as usize * 4, 1),
                ArrayArg::from_raw_parts::<f32>(&self.torques, self.n_molecules as usize * 4, 1),
                ArrayArg::from_raw_parts::<u32>(&self.mol_is_rigid, self.n_molecules as usize, 1),
                ScalarArg::new(self.n_molecules),
                ScalarArg::new(self.box_length),
                ScalarArg::new(inv_box),
            )
        }
        .expect("reduce_forces launch failed");
    }

    fn dispatch_baoab(&self) {
        if !self.has_rigid {
            return;
        }
        let count = CubeCount::Static(self.n_molecules.div_ceil(WORKGROUP_SIZE), 1, 1);
        let dim = CubeDim::new_1d(WORKGROUP_SIZE);

        unsafe {
            kernels::baoab_step::launch_unchecked::<R>(
                &self.client,
                count,
                dim,
                ArrayArg::from_raw_parts::<f32>(
                    &self.com_positions,
                    self.n_molecules as usize * 4,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(
                    &self.com_velocities,
                    self.n_molecules as usize * 4,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(
                    &self.quaternions,
                    self.n_molecules as usize * 4,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(
                    &self.angular_velocities,
                    self.n_molecules as usize * 4,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(&self.com_forces, self.n_molecules as usize * 4, 1),
                ArrayArg::from_raw_parts::<f32>(&self.torques, self.n_molecules as usize * 4, 1),
                ArrayArg::from_raw_parts::<f32>(&self.mol_masses, self.n_molecules as usize, 1),
                ArrayArg::from_raw_parts::<f32>(
                    &self.mol_inertia,
                    self.n_molecules as usize * 4,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&self.mol_is_rigid, self.n_molecules as usize, 1),
                ScalarArg::new(self.n_molecules),
                ScalarArg::new(self.config.timestep as f32),
                ScalarArg::new(self.config.friction as f32),
                ScalarArg::new(self.kt),
                ScalarArg::new(0xDEAD_BEEFu32),
                ScalarArg::new(self.step_counter),
                ScalarArg::new(self.box_length),
            )
        }
        .expect("baoab launch failed");
    }

    fn dispatch_reconstruct(&self) {
        if !self.has_rigid {
            return;
        }
        let count = CubeCount::Static(self.n_atoms.div_ceil(WORKGROUP_SIZE), 1, 1);
        let dim = CubeDim::new_1d(WORKGROUP_SIZE);

        unsafe {
            kernels::reconstruct_positions::launch_unchecked::<R>(
                &self.client,
                count,
                dim,
                ArrayArg::from_raw_parts::<f32>(
                    &self.com_positions,
                    self.n_molecules as usize * 4,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(
                    &self.quaternions,
                    self.n_molecules as usize * 4,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(&self.ref_positions, self.n_atoms as usize * 4, 1),
                ArrayArg::from_raw_parts::<f32>(&self.positions, self.n_atoms as usize * 4, 1),
                ArrayArg::from_raw_parts::<u32>(
                    &self.mol_atom_offsets,
                    self.n_molecules as usize + 1,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&self.mol_is_rigid, self.n_molecules as usize, 1),
                ScalarArg::new(self.n_atoms),
                ScalarArg::new(self.n_molecules),
            )
        }
        .expect("reconstruct launch failed");
    }

    fn dispatch_half_kick(&self) {
        if !self.has_rigid {
            return;
        }
        let count = CubeCount::Static(self.n_molecules.div_ceil(WORKGROUP_SIZE), 1, 1);
        let dim = CubeDim::new_1d(WORKGROUP_SIZE);

        unsafe {
            kernels::half_kick::launch_unchecked::<R>(
                &self.client,
                count,
                dim,
                ArrayArg::from_raw_parts::<f32>(
                    &self.com_velocities,
                    self.n_molecules as usize * 4,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(
                    &self.angular_velocities,
                    self.n_molecules as usize * 4,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(&self.com_forces, self.n_molecules as usize * 4, 1),
                ArrayArg::from_raw_parts::<f32>(&self.torques, self.n_molecules as usize * 4, 1),
                ArrayArg::from_raw_parts::<f32>(
                    &self.quaternions,
                    self.n_molecules as usize * 4,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(&self.mol_masses, self.n_molecules as usize, 1),
                ArrayArg::from_raw_parts::<f32>(
                    &self.mol_inertia,
                    self.n_molecules as usize * 4,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&self.mol_is_rigid, self.n_molecules as usize, 1),
                ScalarArg::new(self.n_molecules),
                ScalarArg::new(self.config.timestep as f32),
            )
        }
        .expect("half_kick launch failed");
    }

    /// Dispatch per-atom BAOAB step for flexible atoms.
    fn dispatch_baoab_atoms(&self) {
        if !self.has_flexible {
            return;
        }
        let count = CubeCount::Static(self.n_atoms.div_ceil(WORKGROUP_SIZE), 1, 1);
        let dim = CubeDim::new_1d(WORKGROUP_SIZE);

        unsafe {
            kernels::baoab_atom_step::launch_unchecked::<R>(
                &self.client,
                count,
                dim,
                ArrayArg::from_raw_parts::<f32>(&self.positions, self.n_atoms as usize * 4, 1),
                ArrayArg::from_raw_parts::<f32>(
                    &self.atom_velocities,
                    self.n_atoms as usize * 3,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(&self.atom_forces, self.n_atoms as usize * 4, 1),
                ArrayArg::from_raw_parts::<f32>(&self.atom_masses_buf, self.n_atoms as usize, 1),
                ArrayArg::from_raw_parts::<u32>(&self.atom_is_flexible, self.n_atoms as usize, 1),
                ScalarArg::new(self.n_atoms),
                ScalarArg::new(self.config.timestep as f32),
                ScalarArg::new(self.config.friction as f32),
                ScalarArg::new(self.kt),
                ScalarArg::new(0xDEAD_BEEFu32),
                ScalarArg::new(self.step_counter),
                ScalarArg::new(self.box_length),
            )
        }
        .expect("baoab_atom_step launch failed");
    }

    /// Dispatch closing B half-kick for flexible atoms.
    fn dispatch_half_kick_atoms(&self) {
        if !self.has_flexible {
            return;
        }
        let count = CubeCount::Static(self.n_atoms.div_ceil(WORKGROUP_SIZE), 1, 1);
        let dim = CubeDim::new_1d(WORKGROUP_SIZE);

        unsafe {
            kernels::half_kick_atoms::launch_unchecked::<R>(
                &self.client,
                count,
                dim,
                ArrayArg::from_raw_parts::<f32>(
                    &self.atom_velocities,
                    self.n_atoms as usize * 3,
                    1,
                ),
                ArrayArg::from_raw_parts::<f32>(&self.atom_forces, self.n_atoms as usize * 4, 1),
                ArrayArg::from_raw_parts::<f32>(&self.atom_masses_buf, self.n_atoms as usize, 1),
                ArrayArg::from_raw_parts::<u32>(&self.atom_is_flexible, self.n_atoms as usize, 1),
                ScalarArg::new(self.n_atoms),
                ScalarArg::new(self.config.timestep as f32),
            )
        }
        .expect("half_kick_atoms launch failed");
    }

    // ========================================================================
    // Data readback
    // ========================================================================

    fn readback_f32(&self, handle: &Handle, count: usize) -> Vec<f32> {
        let bytes = self.client.read_one(handle.clone());
        let all: &[f32] = bytemuck::cast_slice(&bytes);
        all[..count].to_vec()
    }

    fn readback_vec4(&self, handle: &Handle, count: usize) -> Vec<[f32; 4]> {
        let bytes = self.client.read_one(handle.clone());
        let all: &[[f32; 4]] = bytemuck::cast_slice(&bytes);
        all[..count].to_vec()
    }

    fn download_positions(&self) -> Vec<[f32; 4]> {
        self.readback_vec4(&self.positions, self.n_atoms as usize)
    }

    fn download_quaternions(&self) -> Vec<[f32; 4]> {
        self.readback_vec4(&self.quaternions, self.n_molecules as usize)
    }

    fn upload_positions_com_quaternions(
        &mut self,
        positions: &[f32],
        com_positions: &[f32],
        quaternions: &[f32],
    ) {
        debug_assert_eq!(positions.len(), self.n_atoms as usize * 4);
        debug_assert_eq!(com_positions.len(), self.n_molecules as usize * 4);
        debug_assert_eq!(quaternions.len(), self.n_molecules as usize * 4);
        self.positions = self
            .client
            .create_from_slice(bytemuck::cast_slice(positions));
        self.com_positions = self
            .client
            .create_from_slice(bytemuck::cast_slice(com_positions));
        self.quaternions = self
            .client
            .create_from_slice(bytemuck::cast_slice(quaternions));
    }

    /// Compute translational and rotational temperatures from device velocities.
    ///
    /// Uses host-cached copies of constant buffers (masses, inertia, flags)
    /// to avoid unnecessary GPU readbacks — only velocities are downloaded.
    fn download_temperature(&self) -> (f32, f32) {
        let n = self.n_molecules as usize;

        // Only download velocities (which change every step)
        let vel = self.readback_vec4(&self.com_velocities, n);
        let omega = self.readback_vec4(&self.angular_velocities, n);

        // Rigid-body translational + rotational KE from host-cached constants
        let mut ke_trans: f32 = 0.0;
        let mut dof_trans: f32 = 0.0;
        let mut ke_rot: f32 = 0.0;
        let mut dof_rot: f32 = 0.0;
        for ((((&is_rigid, &mass), &v), &w), inertia) in self
            .mol_is_rigid_host
            .iter()
            .zip(&self.mol_masses_host)
            .zip(&vel)
            .zip(&omega)
            .zip(self.mol_inertia_host.chunks_exact(4))
        {
            if is_rigid != 0 {
                let [vx, vy, vz, _] = v;
                ke_trans += 0.5 * mass * (vx * vx + vy * vy + vz * vz);
                dof_trans += 3.0;

                let [wx, wy, wz, _] = w;
                ke_rot +=
                    0.5 * (inertia[0] * wx * wx + inertia[1] * wy * wy + inertia[2] * wz * wz);
                dof_rot += 3.0;
            }
        }

        // Flexible per-atom translational KE
        if self.has_flexible {
            let n_at = self.n_atoms as usize;
            let atom_vel = self.readback_f32(&self.atom_velocities, n_at * 3);

            for ((&is_flex, &mass), v) in self
                .atom_is_flexible_host
                .iter()
                .zip(&self.atom_masses_host)
                .zip(atom_vel.chunks_exact(3))
            {
                if is_flex != 0 {
                    let (vx, vy, vz) = (v[0], v[1], v[2]);
                    ke_trans += 0.5 * mass * (vx * vx + vy * vy + vz * vz);
                    dof_trans += 3.0;
                }
            }
        }

        const R_KJ_PER_MOL_K: f32 = physical_constants::MOLAR_GAS_CONSTANT as f32 * 1e-3;
        const KJ_MOL_TO_INTERNAL: f32 = 100.0;
        let t_trans = if dof_trans > 0.0 {
            2.0 * ke_trans / (KJ_MOL_TO_INTERNAL * dof_trans * R_KJ_PER_MOL_K)
        } else {
            0.0
        };
        let t_rot = if dof_rot > 0.0 {
            2.0 * ke_rot / (KJ_MOL_TO_INTERNAL * dof_rot * R_KJ_PER_MOL_K)
        } else {
            0.0
        };
        (t_trans, t_rot)
    }

    fn log_first_step(&self) {
        if self.step_counter == 0 {
            log::info!(
                "Langevin: box={}, dt={}, friction={}, kT={}, n_mol={}, n_atoms={}, device_forces={}",
                self.box_length,
                self.config.timestep,
                self.config.friction,
                self.kt,
                self.n_molecules,
                self.n_atoms,
                self.has_gpu_forces,
            );
        }
    }
}

// ============================================================================
// LangevinRunner: high-level orchestration between MC and LD
// ============================================================================

/// Rigid-body Langevin dynamics runner (CubeCL compute backend).
#[derive(Debug)]
pub struct LangevinRunner {
    pub(crate) config: LangevinConfig,
    pub(in crate::propagate) elapsed: std::time::Duration,
    /// Accumulated energy change across all LD blocks (for drift tracking).
    pub(in crate::propagate) energy_change_sum: f64,
    /// Running average of translational temperature (K) across LD blocks.
    t_trans: Variance,
    /// Running average of rotational temperature (K) across LD blocks.
    t_rot: Variance,
    #[cfg(feature = "gpu")]
    gpu: Option<LangevinGpu<cubecl::wgpu::WgpuRuntime>>,
}

impl LangevinRunner {
    pub fn new(config: LangevinConfig) -> Self {
        Self {
            config,
            elapsed: std::time::Duration::default(),
            energy_change_sum: 0.0,
            t_trans: Variance::new(),
            t_rot: Variance::new(),
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    #[cfg(not(feature = "gpu"))]
    pub(in crate::propagate) fn propagate<T: Context>(
        &mut self,
        _context: &mut T,
    ) -> anyhow::Result<()> {
        anyhow::bail!("Langevin dynamics requires the `gpu` feature")
    }

    #[cfg(feature = "gpu")]
    pub(in crate::propagate) fn propagate<T: Context>(
        &mut self,
        context: &mut T,
    ) -> anyhow::Result<()> {
        let first_call = self.gpu.is_none();
        if first_call {
            self.gpu = Some(Self::init_gpu(context, &self.config)?);
            Self::upload_full_state(context, self.gpu.as_mut().unwrap())?;
        } else {
            Self::upload_context_state(context, self.gpu.as_mut().unwrap());
        }

        let gpu = self.gpu.as_mut().unwrap();
        let steps = self.config.steps;

        if gpu.has_gpu_forces {
            gpu.run_steps(steps);
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
            gpu.run_steps_with_cpu_forces(steps, &mut force_callback);
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
    #[cfg(feature = "gpu")]
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
        let kt = (physical_constants::MOLAR_GAS_CONSTANT * 1e-3 * config.temperature) as f32;

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
    #[cfg(feature = "gpu")]
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
        let (atom_type_ids, mol_ids, spline_params, spline_coeffs, n_atom_types) =
            Self::extract_spline_data(context);

        // Extract bonded topology into CSR layout for on-device bonded forces
        let (bond_data, angle_data, dihedral_data) = Self::extract_bonded_data(context);

        // COM positions, reference-frame coordinates, and molecule->atom offsets
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
                let m = atom_mass(&topology, atom_indices, idx) as f64;
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

        // Maxwell-Boltzmann velocities: sigma = sqrt(kT * 100 / M) in A/ps
        let kt = gpu.kt as f64;
        let conv = 100.0; // kJ/mol -> amu*A^2/ps^2
        let mut rng = rand::thread_rng();
        // Generate a single MB velocity component: v ~ N(0, sqrt(kT*conv/mass))
        let mut mb_velocity = |mass: f64| -> f32 {
            let u1: f64 = rng.gen::<f64>().max(1e-30);
            let u2: f64 = rng.gen();
            let gauss = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            let sigma = if mass > 0.0 {
                ((kt * conv) / mass).sqrt()
            } else {
                0.0
            };
            (sigma * gauss) as f32
        };

        // Rigid COM velocities (zeros for non-rigid)
        let mut com_velocities = Vec::with_capacity(n_mol * 4);
        for (&is_rigid, &mass) in mol_is_rigid.iter().zip(&mol_masses) {
            if is_rigid != 0 {
                com_velocities.extend_from_slice(&[
                    mb_velocity(mass as f64),
                    mb_velocity(mass as f64),
                    mb_velocity(mass as f64),
                ]);
            } else {
                com_velocities.extend_from_slice(&[0.0, 0.0, 0.0]);
            }
            com_velocities.push(0.0f32);
        }

        let mut angular_velocities = Vec::with_capacity(n_mol * 4);
        for (m_idx, &is_rigid) in mol_is_rigid.iter().enumerate() {
            if is_rigid != 0 {
                let base = m_idx * 4;
                angular_velocities.extend_from_slice(&[
                    mb_velocity(mol_inertia[base] as f64),
                    mb_velocity(mol_inertia[base + 1] as f64),
                    mb_velocity(mol_inertia[base + 2] as f64),
                ]);
            } else {
                angular_velocities.extend_from_slice(&[0.0, 0.0, 0.0]);
            }
            angular_velocities.push(0.0f32);
        }

        // Per-atom velocities (Maxwell-Boltzmann for flexible, zeros for rigid/frozen)
        let mut atom_velocities = Vec::with_capacity(n * 3);
        for (&is_flex, &mass) in atom_is_flexible.iter().zip(&atom_masses_vec) {
            if is_flex != 0 {
                let m = mass as f64;
                atom_velocities.extend_from_slice(&[
                    mb_velocity(m),
                    mb_velocity(m),
                    mb_velocity(m),
                ]);
            } else {
                atom_velocities.extend_from_slice(&[0.0, 0.0, 0.0]);
            }
        }

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
            atom_type_ids,
            mol_ids,
            spline_params,
            spline_coeffs,
            n_atom_types,
            bond_data,
            angle_data,
            dihedral_data,
            mol_is_rigid,
            atom_velocities,
            atom_masses: atom_masses_vec,
            atom_is_flexible,
            has_flexible,
        });
        Ok(())
    }

    /// Extract spline data from the Hamiltonian for on-device force computation.
    ///
    /// Returns `(atom_type_ids, mol_ids, spline_params, spline_coeffs, n_atom_types)`.
    /// If no splined nonbonded term exists, returns None and forces fall back to CPU.
    #[cfg(feature = "gpu")]
    fn extract_spline_data<T: Context>(context: &T) -> SplineData {
        let hamiltonian = context.hamiltonian();
        let nonbonded = hamiltonian
            .energy_terms()
            .iter()
            .find_map(|term| match term {
                crate::energy::EnergyTerm::NonbondedMatrixSplined(nb) => Some(nb),
                _ => None,
            });

        let Some(nb) = nonbonded else {
            log::info!("No splined nonbonded term found; using CPU forces");
            return (None, None, None, None, 0);
        };

        let n_atom_types = context.topology().atomkinds().len() as u32;

        let spline_data =
            interatomic::gpu::GpuSplineData::<interatomic::gpu::PowerLaw2>::from_potentials(
                nb.get_potentials().iter(),
            );
        let spline_params =
            crate::energy::nonbonded_kernel::repack_spline_params(&spline_data.params);
        let spline_coeffs =
            crate::energy::nonbonded_kernel::repack_spline_coeffs(&spline_data.coefficients);

        let atom_type_ids: Vec<u32> = context.atom_kinds_u32().to_vec();

        // Rigid-body atoms within the same molecule must not interact via nonbonded forces
        let mol_ids: Vec<u32> = context
            .groups()
            .iter()
            .enumerate()
            .flat_map(|(mol_idx, g)| std::iter::repeat_n(mol_idx as u32, g.capacity()))
            .collect();

        drop(hamiltonian);
        (
            Some(atom_type_ids),
            Some(mol_ids),
            Some(spline_params),
            Some(spline_coeffs),
            n_atom_types,
        )
    }

    /// Extract bonded topology (bonds, angles, dihedrals) into CSR layout.
    ///
    /// Returns `None` for each interaction type that has no entries.
    #[cfg(feature = "gpu")]
    fn extract_bonded_data<T: Context>(
        context: &T,
    ) -> (
        Option<crate::energy::bonded::kernel::CsrData>,
        Option<crate::energy::bonded::kernel::CsrData>,
        Option<crate::energy::bonded::kernel::CsrData>,
    ) {
        use crate::energy::bonded::kernel;
        let topology = context.topology();
        let groups = context.groups();

        let to_option = |csr: kernel::CsrData| {
            if csr.is_empty() {
                None
            } else {
                Some(csr)
            }
        };
        let bonds = to_option(kernel::repack_bonds(&topology, groups));
        let angles = to_option(kernel::repack_angles(&topology, groups));
        let dihedrals = to_option(kernel::repack_dihedrals(&topology, groups));

        drop(topology);
        (bonds, angles, dihedrals)
    }

    /// Upload positions, COM, and quaternions from context to device.
    #[cfg(feature = "gpu")]
    fn upload_context_state<T: Context>(
        context: &T,
        gpu: &mut LangevinGpu<cubecl::wgpu::WgpuRuntime>,
    ) {
        let positions = pack_positions_f32(context);
        let (com_positions, quaternions) = pack_com_and_quaternions(context);
        gpu.upload_positions_com_quaternions(&positions, &com_positions, &quaternions);
    }

    #[cfg(feature = "gpu")]
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

    pub(in crate::propagate) fn to_yaml(&self) -> serde_yaml::Value {
        let mut map = serde_yaml::Mapping::new();
        map.insert("timestep".into(), self.config.timestep.into());
        map.insert("friction".into(), self.config.friction.into());
        map.insert("steps".into(), self.config.steps.into());
        map.insert("temperature".into(), self.config.temperature.into());
        map.insert(
            "elapsed_seconds".into(),
            serde_yaml::Value::Number(serde_yaml::Number::from(self.elapsed.as_secs_f64())),
        );
        if !self.t_trans.is_empty() {
            let mut temp_map = serde_yaml::Mapping::new();
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
                serde_yaml::Value::Mapping(temp_map),
            );
        }
        serde_yaml::Value::Tagged(Box::new(serde_yaml::value::TaggedValue {
            tag: serde_yaml::value::Tag::new("LangevinDynamics"),
            value: serde_yaml::Value::Mapping(map),
        }))
    }
}

/// Convert a `UnitQuaternion` to device layout `[i, j, k, w]` as f32.
#[cfg(any(test, feature = "gpu"))]
fn quat_to_gpu(q: &crate::UnitQuaternion) -> [f32; 4] {
    [q.i as f32, q.j as f32, q.k as f32, q.w as f32]
}

/// Convert device layout `[i, j, k, w]` back to a `UnitQuaternion`.
#[cfg(any(test, feature = "gpu"))]
fn gpu_to_quat(q: &[f32; 4]) -> crate::UnitQuaternion {
    nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
        q[3] as f64,
        q[0] as f64,
        q[1] as f64,
        q[2] as f64,
    ))
}

/// Pack all atom positions from context into vec4 layout `[x, y, z, atom_type_bits]`.
#[cfg(feature = "gpu")]
fn pack_positions_f32<T: Context>(context: &T) -> Vec<f32> {
    let n = context.num_particles();
    let mut positions = Vec::with_capacity(n * 4);
    let (x, y, z) = context.positions_soa();
    let atom_kinds = context.atom_kinds_u32();
    for i in 0..n {
        positions.push(x[i] as f32);
        positions.push(y[i] as f32);
        positions.push(z[i] as f32);
        positions.push(f32::from_bits(atom_kinds[i]));
    }
    positions
}

/// Pack per-molecule COM positions and quaternions from context groups.
#[cfg(feature = "gpu")]
fn pack_com_and_quaternions<T: Context>(context: &T) -> (Vec<f32>, Vec<f32>) {
    let groups = context.groups();
    let mut com_positions = Vec::with_capacity(groups.len() * 4);
    let mut quaternions = Vec::with_capacity(groups.len() * 4);
    for group in groups {
        let com = group
            .mass_center()
            .copied()
            .unwrap_or_else(|| context.mass_center(&group.iter_active().collect::<Vec<_>>()));
        com_positions.extend_from_slice(&[com.x as f32, com.y as f32, com.z as f32, 0.0]);
        quaternions.extend_from_slice(&quat_to_gpu(group.quaternion()));
    }
    (com_positions, quaternions)
}

/// Reduce per-atom forces to per-molecule COM forces and torques (CPU path).
///
/// - COM force = sum of atomic forces in each molecule
/// - Torque = sum of (lever_arm x force) with PBC-aware lever arms
///
/// Returns `(com_forces, torques)` in vec4 layout `[x, y, z, 0]`.
#[cfg(feature = "gpu")]
fn reduce_forces_to_com<T: Context>(
    context: &T,
    forces: &[crate::Point],
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let groups = context.groups();
    let cell = context.cell();
    let mut com_forces = Vec::with_capacity(groups.len());
    let mut torques = Vec::with_capacity(groups.len());

    for group in groups {
        let com = group
            .mass_center()
            .copied()
            .unwrap_or_else(|| context.mass_center(&group.iter_active().collect::<Vec<_>>()));
        let mut f_total = crate::Point::zeros();
        let mut tau_total = crate::Point::zeros();

        for i in group.iter_active() {
            let f = if i < forces.len() {
                forces[i]
            } else {
                crate::Point::zeros()
            };
            f_total += f;
            // PBC-aware lever arm: minimum image of (r_i - r_com)
            let lever = cell.distance(&context.position(i), &com);
            tau_total += lever.cross(&f);
        }

        com_forces.push([f_total.x as f32, f_total.y as f32, f_total.z as f32, 0.0]);
        torques.push([
            tau_total.x as f32,
            tau_total.y as f32,
            tau_total.z as f32,
            0.0,
        ]);
    }

    (com_forces, torques)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that quaternion roundtrip through f32 layout [x,y,z,w] is accurate.
    #[test]
    fn quaternion_roundtrip_f32_f64() {
        let axis = nalgebra::UnitVector3::new_normalize(crate::Point::new(1.0, 2.0, 3.0));
        let q = crate::UnitQuaternion::from_axis_angle(&axis, 1.23);

        let gpu = quat_to_gpu(&q);
        let reconstructed = gpu_to_quat(&gpu);

        // f32 precision ~ 1e-7, so angle error should be small
        assert!(
            q.angle_to(&reconstructed) < 1e-6,
            "Roundtrip angle error: {}",
            q.angle_to(&reconstructed)
        );
    }
}

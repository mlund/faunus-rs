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

//! CubeCL compute pipeline: GPU buffer management, kernel dispatch, and data readback.

use super::kernels;
use super::{compute_temperature, LangevinConfig};
use crate::energy::nonbonded_kernel::{
    build_cell_list, build_neighbor_cell_table, compute_n_cells_1d, max_cutoff_from_spline_params,
};
use cubecl::prelude::*;
use cubecl::server::Handle;
use std::marker::PhantomData;

pub(super) const WORKGROUP_SIZE: u32 = 64;

/// Callback computing per-molecule COM forces and torques from atom positions.
pub(super) type ForceCallback<'a> =
    &'a mut dyn FnMut(&[[f32; 4]]) -> (Vec<[f32; 4]>, Vec<[f32; 4]>);

/// Pre-computed data for initial state upload.
///
/// All vectors use vec4-padded f32 layout so they can be written
/// directly to device buffers without further transformation.
pub(super) struct LangevinUploadData {
    pub(super) positions: Vec<f32>,
    pub(super) ref_positions: Vec<f32>,
    pub(super) com_positions: Vec<f32>,
    pub(super) mol_atom_offsets: Vec<u32>,
    pub(super) mol_masses: Vec<f32>,
    pub(super) mol_inertia: Vec<f32>,
    pub(super) quaternions: Vec<f32>,
    pub(super) com_velocities: Vec<f32>,
    pub(super) angular_velocities: Vec<f32>,
    /// Per-atom type indices for spline lookup (None = CPU forces only).
    pub(super) atom_type_ids: Option<Vec<u32>>,
    /// Per-atom molecule index for intra-molecular exclusion.
    pub(super) mol_ids: Option<Vec<u32>>,
    /// Flat spline params array (stride 8 per type pair).
    pub(super) spline_params: Option<Vec<f32>>,
    /// Flat spline coefficients array (stride 8 per interval).
    pub(super) spline_coeffs: Option<Vec<f32>>,
    /// Number of atom types (matrix dimension for spline_params).
    pub(super) n_atom_types: u32,
    /// CSR bond data.
    pub(super) bond_data: Option<crate::energy::bonded::kernel::CsrData>,
    /// CSR angle data.
    pub(super) angle_data: Option<crate::energy::bonded::kernel::CsrData>,
    /// CSR dihedral data.
    pub(super) dihedral_data: Option<crate::energy::bonded::kernel::CsrData>,
    /// Per-atom exclusion CSR offsets for intra-molecular NB skip.
    pub(super) excl_offsets: Vec<u32>,
    /// Per-atom exclusion CSR neighbor indices.
    pub(super) excl_atoms: Vec<u32>,
    /// Per-molecule flag: 1 = rigid-body integration, 0 = per-atom or frozen.
    pub(super) mol_is_rigid: Vec<u32>,
    /// Per-atom velocities for flexible atoms (stride 3: vx, vy, vz).
    pub(super) atom_velocities: Vec<f32>,
    /// Per-atom masses for flexible integration.
    pub(super) atom_masses: Vec<f32>,
    /// Per-atom flag: 1 = flexible (per-atom BAOAB), 0 = rigid or frozen.
    pub(super) atom_is_flexible: Vec<u32>,
    /// Whether any flexible atoms exist.
    pub(super) has_flexible: bool,
}

/// CubeCL-based Langevin dynamics pipeline, generic over runtime backend.
pub(super) struct LangevinGpu<R: Runtime> {
    pub(super) client: ComputeClient<R>,
    pub(super) config: LangevinConfig,

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
    pub(super) has_gpu_forces: bool,

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

    // Exclusion CSR for intra-molecular NB pairs in flexible molecules
    excl_offsets: Handle,
    excl_atoms: Handle,

    // Per-atom flexible integration buffers
    mol_is_rigid: Handle,
    atom_velocities: Handle,
    atom_masses_buf: Handle,
    atom_is_flexible: Handle,
    pub(super) has_flexible: bool,
    has_rigid: bool,

    // Cell list CSR for O(n·k) neighbor iteration
    cell_offsets: Handle,
    cell_atoms: Handle,
    pub(super) neighbor_cells: Handle,
    pub(super) n_cells_1d: u32,
    /// Cached max cutoff from spline params (constant, avoids GPU readback)
    pub(super) max_cutoff: f32,

    // Host-side copies of constant buffers (avoids GPU readbacks in download_temperature)
    pub(super) mol_is_rigid_host: Vec<u32>,
    mol_masses_host: Vec<f32>,
    mol_inertia_host: Vec<f32>,
    atom_masses_host: Vec<f32>,
    atom_is_flexible_host: Vec<u32>,

    pub(super) n_atoms: u32,
    pub(super) n_molecules: u32,
    pub(super) box_length: f32,
    pub(super) kt: f32,
    step_counter: u32,
    _runtime: PhantomData<R>,
}

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

impl<R: Runtime> LangevinGpu<R> {
    pub(super) fn new(
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

        // Exclusion CSR placeholders (real data uploaded if flexible molecules exist)
        let excl_offsets = client.empty((n_atoms as usize + 1) * 4);
        let excl_atoms = client.empty(4);

        // Placeholders for flexible integration buffers
        let mol_is_rigid = client.empty(n_molecules as usize * 4);
        let atom_velocities = client.empty(n_atoms as usize * 3 * 4);
        let atom_masses_buf = client.empty(n_atoms as usize * 4);
        let atom_is_flexible = client.empty(n_atoms as usize * 4);

        // Cell list placeholders (real data uploaded after spline params are available)
        let cell_offsets = client.empty(4);
        let cell_atoms = client.empty(4);
        let neighbor_cells = client.empty(4);

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
            excl_offsets,
            excl_atoms,
            mol_is_rigid,
            atom_velocities,
            atom_masses_buf,
            atom_is_flexible,
            has_flexible: false,
            has_rigid: true,
            cell_offsets,
            cell_atoms,
            neighbor_cells,
            n_cells_1d: 1,
            max_cutoff: 0.0,
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

    pub(super) fn upload_state(&mut self, data: LangevinUploadData) {
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

            // Neighbor table is constant (depends only on grid size), so built once here.
            // The per-step cell_offsets/cell_atoms CSR is rebuilt in rebuild_cell_list().
            self.max_cutoff = max_cutoff_from_spline_params(params);
            self.n_cells_1d = compute_n_cells_1d(self.box_length, self.max_cutoff);
            let nb_table = build_neighbor_cell_table(self.n_cells_1d);
            self.neighbor_cells = self
                .client
                .create_from_slice(bytemuck::cast_slice(&nb_table));
            log::info!(
                "On-device forces enabled: {} atom types, {} spline params, {} spline coeffs, \
                 cell grid {}³ (cutoff={:.2})",
                data.n_atom_types,
                params.len() / 8,
                coeffs.len() / 8,
                self.n_cells_1d,
                self.max_cutoff,
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

        // Upload exclusion CSR for intra-molecular NB pairs
        self.excl_offsets = self
            .client
            .create_from_slice(bytemuck::cast_slice(&data.excl_offsets));
        if !data.excl_atoms.is_empty() {
            self.excl_atoms = self
                .client
                .create_from_slice(bytemuck::cast_slice(&data.excl_atoms));
            log::info!("On-device exclusion CSR: {} entries", data.excl_atoms.len());
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

    /// Download positions and rebuild the cell list CSR on CPU, then upload.
    fn rebuild_cell_list(&mut self) {
        let positions = self.download_positions();
        let (offsets, atoms) = build_cell_list(&positions, self.box_length, self.n_cells_1d);
        self.cell_offsets = self
            .client
            .create_from_slice(bytemuck::cast_slice(&offsets));
        self.cell_atoms = self.client.create_from_slice(bytemuck::cast_slice(&atoms));
    }

    /// Run `steps` BAOAB steps with on-device force computation.
    ///
    /// Forces, reduction, integration, and reconstruction all run on the compute
    /// device without any host readback during the integration loop.
    /// Rigid and flexible integration paths operate on disjoint atom ranges.
    pub(super) fn run_steps(&mut self, steps: usize) -> anyhow::Result<()> {
        anyhow::ensure!(self.has_gpu_forces, "on-device forces not initialized");
        self.log_first_step();
        let rebuild_interval = self.config.cell_list_rebuild;

        // Initial cell list required before first force evaluation
        self.rebuild_cell_list();
        self.dispatch_pair_forces()?;
        self.dispatch_bonded_forces()?;
        self.dispatch_reduce()?;

        for step in 0..steps as u32 {
            self.dispatch_baoab()?;
            self.dispatch_baoab_atoms()?;
            self.dispatch_reconstruct()?;

            // Periodic rebuild keeps cell list valid as particles diffuse;
            // costs one GPU sync per rebuild (~0.1ms) vs force computation
            if rebuild_interval > 0 && (step + 1) % rebuild_interval == 0 {
                self.rebuild_cell_list();
            }

            self.dispatch_pair_forces()?;
            self.dispatch_bonded_forces()?;
            self.dispatch_reduce()?;

            self.dispatch_half_kick()?;
            self.dispatch_half_kick_atoms()?;
            self.step_counter += 1;
        }

        // Sync to ensure all work is complete
        let _ = self.client.read_one(self.com_forces.clone());
        Ok(())
    }

    /// Run `steps` BAOAB steps with CPU-computed forces from a callback.
    ///
    /// CPU forces path provides COM forces/torques directly; flexible atoms
    /// are not supported in this path (requires on-device forces).
    pub(super) fn run_steps_with_cpu_forces(
        &mut self,
        steps: usize,
        compute_forces: ForceCallback<'_>,
    ) -> anyhow::Result<()> {
        self.log_first_step();

        let positions = self.download_positions();
        let (com_forces, torques) = compute_forces(&positions);
        self.upload_com_forces_torques(&com_forces, &torques);

        for _ in 0..steps {
            self.dispatch_baoab()?;
            self.dispatch_reconstruct()?;

            let positions = self.download_positions();
            let (com_forces, torques) = compute_forces(&positions);
            self.upload_com_forces_torques(&com_forces, &torques);

            self.dispatch_half_kick()?;
            self.step_counter += 1;
        }

        let _ = self.client.read_one(self.com_forces.clone());
        Ok(())
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

    fn dispatch_pair_forces(&self) -> anyhow::Result<()> {
        let count = CubeCount::Static(self.n_atoms.div_ceil(WORKGROUP_SIZE), 1, 1);
        let dim = CubeDim::new_1d(WORKGROUP_SIZE);
        let inv_box = 1.0f32 / self.box_length;
        // CubeCL requires non-zero array length; clamp to 1 when CSR is empty
        let n_excl_atoms = (self.excl_atoms.size() as usize / 4).max(1);
        let n3 = (self.n_cells_1d * self.n_cells_1d * self.n_cells_1d) as usize;

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
                ArrayArg::from_raw_parts::<u32>(&self.excl_offsets, self.n_atoms as usize + 1, 1),
                ArrayArg::from_raw_parts::<u32>(&self.excl_atoms, n_excl_atoms, 1),
                ArrayArg::from_raw_parts::<u32>(&self.mol_is_rigid, self.n_molecules as usize, 1),
                ArrayArg::from_raw_parts::<u32>(&self.cell_offsets, n3 + 1, 1),
                ArrayArg::from_raw_parts::<u32>(&self.cell_atoms, self.n_atoms as usize, 1),
                ArrayArg::from_raw_parts::<u32>(&self.neighbor_cells, n3 * 27, 1),
                ArrayArg::from_raw_parts::<f32>(&self.atom_forces, self.n_atoms as usize * 4, 1),
                ScalarArg::new(self.n_atoms),
                ScalarArg::new(self.n_atom_types),
                ScalarArg::new(self.box_length),
                ScalarArg::new(inv_box),
                ScalarArg::new(self.n_cells_1d),
            )
        }?;
        Ok(())
    }

    /// Dispatch bond, angle, and dihedral force kernels.
    ///
    /// Must run after `dispatch_pair_forces` because the bonded kernels
    /// accumulate (`+=`) into the same `atom_forces` buffer that the
    /// nonbonded kernel initializes (`=`).
    fn dispatch_bonded_forces(&self) -> anyhow::Result<()> {
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
            }?;
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
            }?;
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
            }?;
        }
        Ok(())
    }

    fn dispatch_reduce(&self) -> anyhow::Result<()> {
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
        }?;
        Ok(())
    }

    fn dispatch_baoab(&self) -> anyhow::Result<()> {
        if !self.has_rigid {
            return Ok(());
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
        }?;
        Ok(())
    }

    fn dispatch_reconstruct(&self) -> anyhow::Result<()> {
        if !self.has_rigid {
            return Ok(());
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
        }?;
        Ok(())
    }

    fn dispatch_half_kick(&self) -> anyhow::Result<()> {
        if !self.has_rigid {
            return Ok(());
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
        }?;
        Ok(())
    }

    /// Dispatch per-atom BAOAB step for flexible atoms.
    fn dispatch_baoab_atoms(&self) -> anyhow::Result<()> {
        if !self.has_flexible {
            return Ok(());
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
        }?;
        Ok(())
    }

    /// Dispatch closing B half-kick for flexible atoms.
    fn dispatch_half_kick_atoms(&self) -> anyhow::Result<()> {
        if !self.has_flexible {
            return Ok(());
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
        }?;
        Ok(())
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

    pub(super) fn download_positions(&self) -> Vec<[f32; 4]> {
        self.readback_vec4(&self.positions, self.n_atoms as usize)
    }

    #[cfg(test)]
    pub(super) fn download_com_velocities(&self) -> Vec<[f32; 4]> {
        self.readback_vec4(&self.com_velocities, self.n_molecules as usize)
    }

    pub(super) fn download_quaternions(&self) -> Vec<[f32; 4]> {
        self.readback_vec4(&self.quaternions, self.n_molecules as usize)
    }

    pub(super) fn upload_positions_com_quaternions(
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

    /// Download velocities from device and compute temperatures.
    pub(super) fn download_temperature(&self) -> (f32, f32) {
        let n = self.n_molecules as usize;
        let vel = self.readback_vec4(&self.com_velocities, n);
        let omega = self.readback_vec4(&self.angular_velocities, n);
        let atom_vel = if self.has_flexible {
            self.readback_f32(&self.atom_velocities, self.n_atoms as usize * 3)
        } else {
            Vec::new()
        };
        compute_temperature(
            &self.mol_is_rigid_host,
            &self.mol_masses_host,
            &self.mol_inertia_host,
            &vel,
            &omega,
            &self.atom_is_flexible_host,
            &self.atom_masses_host,
            &atom_vel,
        )
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

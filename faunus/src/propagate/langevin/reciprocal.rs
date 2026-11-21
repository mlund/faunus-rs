//! On-device Ewald reciprocal-space forces, following the formula in
//! [`interatomic::coulomb::gpu`]. Two passes without atomics: S(k) per k-vector,
//! then the force per charged atom.

use super::pipeline::WORKGROUP_SIZE;
use crate::energy::ReciprocalDeviceData;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// k-space data resident on the device; rebuild per LD block, since MC may have
/// changed the box or the charges.
pub(super) struct DeviceReciprocal {
    client: Client,
    kvectors: Handle,
    structure_factor: Handle,
    atoms: Handle,
    charges: Handle,
    n_k: u32,
    n_charged: u32,
    force_prefactor: f32,
    box_length: f32,
}

impl DeviceReciprocal {
    pub(super) fn new(client: &Client, data: &ReciprocalDeviceData) -> Self {
        let n_k = data.kvectors.len() as u32;
        let n_charged = data.atoms.len() as u32;
        // Empty buffers are invalid; `add_forces` skips the passes in that case
        let upload = |bytes: &[u8]| {
            if bytes.is_empty() {
                client.empty(16)
            } else {
                client.create_from_slice(bytes)
            }
        };
        Self {
            client: client.clone(),
            kvectors: upload(bytemuck::cast_slice(&data.kvectors)),
            structure_factor: client.empty((n_k.max(1) as usize) * 2 * 4),
            atoms: upload(bytemuck::cast_slice(&data.atoms)),
            charges: upload(bytemuck::cast_slice(&data.charges)),
            n_k,
            n_charged,
            force_prefactor: data.force_prefactor,
            box_length: data.box_length,
        }
    }

    /// Add the reciprocal force to `atom_forces` (vec4 per atom) at `positions`.
    pub(super) fn add_forces(&self, positions: &Handle, atom_forces: &Handle) {
        if self.n_k == 0 || self.n_charged == 0 {
            return;
        }
        let f32_len = |h: &Handle| h.size() as usize / 4;
        let n_k = self.n_k as usize;
        let n_charged = self.n_charged as usize;
        let dim = || CubeDim::new_1d(WORKGROUP_SIZE);
        let inv_box = 1.0 / self.box_length;
        unsafe {
            structure_factor_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(self.n_k.div_ceil(WORKGROUP_SIZE), 1, 1),
                dim(),
                BufferArg::from_raw_parts(positions.clone(), f32_len(positions)),
                BufferArg::from_raw_parts(self.atoms.clone(), n_charged),
                BufferArg::from_raw_parts(self.charges.clone(), n_charged),
                BufferArg::from_raw_parts(self.kvectors.clone(), n_k * 4),
                BufferArg::from_raw_parts(self.structure_factor.clone(), n_k * 2),
                self.n_charged,
                self.n_k,
                self.box_length,
                inv_box,
            );
            reciprocal_forces_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(self.n_charged.div_ceil(WORKGROUP_SIZE), 1, 1),
                dim(),
                BufferArg::from_raw_parts(positions.clone(), f32_len(positions)),
                BufferArg::from_raw_parts(self.atoms.clone(), n_charged),
                BufferArg::from_raw_parts(self.charges.clone(), n_charged),
                BufferArg::from_raw_parts(self.kvectors.clone(), n_k * 4),
                BufferArg::from_raw_parts(self.structure_factor.clone(), n_k * 2),
                BufferArg::from_raw_parts(atom_forces.clone(), f32_len(atom_forces)),
                self.n_charged,
                self.n_k,
                self.force_prefactor,
                self.box_length,
                inv_box,
            );
        }
    }
}

/// Image of a coordinate in the box: f32 sin/cos lose precision as |k·r| grows,
/// and k ∈ 2π/L·ℤ³ makes the wrap exact.
#[cube]
fn wrap(x: f32, box_length: f32, inv_box: f32) -> f32 {
    x - box_length * f32::round(x * inv_box)
}

/// S(k) = Σⱼ qⱼ e^{ik·rⱼ}, stored as (Re, Im); one thread per k-vector.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn structure_factor_kernel(
    positions: &[f32],
    atoms: &[u32],
    charges: &[f32],
    kvectors: &[f32],
    structure_factor: &mut [f32],
    n_charged: u32,
    n_k: u32,
    box_length: f32,
    inv_box: f32,
) {
    let k = ABSOLUTE_POS;
    if k < n_k as usize {
        let kx = kvectors[k * 4];
        let ky = kvectors[k * 4 + 1];
        let kz = kvectors[k * 4 + 2];
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for j in 0..n_charged as usize {
            let a = atoms[j] as usize * 4;
            let kr = kx * wrap(positions[a], box_length, inv_box)
                + ky * wrap(positions[a + 1], box_length, inv_box)
                + kz * wrap(positions[a + 2], box_length, inv_box);
            re += charges[j] * f32::cos(kr);
            im += charges[j] * f32::sin(kr);
        }
        structure_factor[k * 2] = re;
        structure_factor[k * 2 + 1] = im;
    }
}

/// Fᵢ = prefactor·qᵢ Σₖ A(k) k Im[e^{ik·rᵢ} S(k)*], added to `atom_forces`;
/// one thread per charged atom.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn reciprocal_forces_kernel(
    positions: &[f32],
    atoms: &[u32],
    charges: &[f32],
    kvectors: &[f32],
    structure_factor: &[f32],
    atom_forces: &mut [f32],
    n_charged: u32,
    n_k: u32,
    force_prefactor: f32,
    box_length: f32,
    inv_box: f32,
) {
    let j = ABSOLUTE_POS;
    if j < n_charged as usize {
        let a = atoms[j] as usize * 4;
        let x = wrap(positions[a], box_length, inv_box);
        let y = wrap(positions[a + 1], box_length, inv_box);
        let z = wrap(positions[a + 2], box_length, inv_box);
        let mut fx = 0.0f32;
        let mut fy = 0.0f32;
        let mut fz = 0.0f32;
        for k in 0..n_k as usize {
            let kx = kvectors[k * 4];
            let ky = kvectors[k * 4 + 1];
            let kz = kvectors[k * 4 + 2];
            let kr = kx * x + ky * y + kz * z;
            let term = kvectors[k * 4 + 3]
                * (f32::sin(kr) * structure_factor[k * 2]
                    - f32::cos(kr) * structure_factor[k * 2 + 1]);
            fx += term * kx;
            fy += term * ky;
            fz += term * kz;
        }
        let scale = force_prefactor * charges[j];
        atom_forces[a] += scale * fx;
        atom_forces[a + 1] += scale * fy;
        atom_forces[a + 2] += scale * fz;
    }
}

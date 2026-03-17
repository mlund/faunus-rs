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

//! Pure helper functions for data packing, velocity generation, and temperature
//! computation. No GPU or CubeCL dependency — testable on the host.

use crate::cell::BoundaryConditions;
use crate::topology::Topology;
use crate::Context;
use rand::Rng;

/// Atom mass from topology, falling back to 1.0 for out-of-range indices.
pub(super) fn atom_mass(topology: &Topology, atom_indices: &[usize], idx: usize) -> f32 {
    // For atomic mega-groups, all atoms share the same kind (index 0)
    atom_indices
        .get(idx)
        .or_else(|| atom_indices.first())
        .map(|&ai| topology.atomkinds()[ai].mass() as f32)
        .unwrap_or(1.0)
}

/// Log a warning if the timestep exceeds the harmonic stability limit for flexible atoms.
pub(super) fn warn_timestep_stability(
    timestep: f64,
    atom_masses: &[f32],
    atom_is_flexible: &[u32],
    bonds: &crate::energy::bonded::kernel::CsrData,
) {
    let min_mass = atom_masses
        .iter()
        .zip(atom_is_flexible)
        .filter(|(_, &f)| f != 0)
        .map(|(&m, _)| m)
        .reduce(f32::min)
        .unwrap_or(1.0);
    let k_max = bonds
        .params
        .chunks(2)
        .map(|kv| kv[0])
        .reduce(f32::max)
        .unwrap_or(0.0);
    if k_max > 0.0 && min_mass > 0.0 {
        // Leapfrog stability: dt < 2/omega where omega = sqrt(k*conv/m)
        let conv = 100.0_f64; // kJ/mol → amu·Å²/ps²
        let dt_max = 2.0 / (k_max as f64 * conv / min_mass as f64).sqrt();
        if timestep > dt_max {
            log::warn!(
                "Timestep {:.4} ps exceeds harmonic stability limit {:.4} ps \
                 (k_max={:.1} kJ/mol/Å², m_min={:.2} amu). Reduce dt or increase mass.",
                timestep,
                dt_max,
                k_max,
                min_mass,
            );
        }
    }
}

/// Generate Maxwell-Boltzmann velocities for all DOFs.
///
/// Returns `(com_velocities, angular_velocities, atom_velocities)` in their
/// respective GPU layouts (vec4 stride for COM/angular, stride-3 for atoms).
pub(super) fn generate_mb_velocities(
    kt: f64,
    mol_is_rigid: &[u32],
    mol_masses: &[f32],
    mol_inertia: &[f32],
    atom_is_flexible: &[u32],
    atom_masses: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let conv = 100.0_f64; // kJ/mol → amu·Å²/ps²
    let mut rng = rand::thread_rng();
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

    let n_mol = mol_is_rigid.len();
    let mut com_velocities = Vec::with_capacity(n_mol * 4);
    for (&is_rigid, &mass) in mol_is_rigid.iter().zip(mol_masses) {
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

    let n_atoms = atom_is_flexible.len();
    let mut atom_velocities = Vec::with_capacity(n_atoms * 3);
    for (&is_flex, &mass) in atom_is_flexible.iter().zip(atom_masses) {
        if is_flex != 0 {
            let m = mass as f64;
            atom_velocities.extend_from_slice(&[mb_velocity(m), mb_velocity(m), mb_velocity(m)]);
        } else {
            atom_velocities.extend_from_slice(&[0.0, 0.0, 0.0]);
        }
    }

    (com_velocities, angular_velocities, atom_velocities)
}

/// Compute translational and rotational temperatures from velocities and constant data.
///
/// Pure computation with no GPU access — testable independently of the device pipeline.
/// Temperature is derived from kinetic energy via T = 2·KE / (dof · kB) where
/// KE uses internal units (amu·Å²/ps²) with conversion factor 100 kJ/mol.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_temperature(
    mol_is_rigid: &[u32],
    mol_masses: &[f32],
    mol_inertia: &[f32],
    com_velocities: &[[f32; 4]],
    angular_velocities: &[[f32; 4]],
    atom_is_flexible: &[u32],
    atom_masses: &[f32],
    atom_velocities: &[f32],
) -> (f32, f32) {
    let mut ke_trans: f32 = 0.0;
    let mut dof_trans: f32 = 0.0;
    let mut ke_rot: f32 = 0.0;
    let mut dof_rot: f32 = 0.0;

    for ((((&is_rigid, &mass), &v), &w), inertia) in mol_is_rigid
        .iter()
        .zip(mol_masses)
        .zip(com_velocities)
        .zip(angular_velocities)
        .zip(mol_inertia.chunks_exact(4))
    {
        if is_rigid != 0 {
            let [vx, vy, vz, _] = v;
            ke_trans += 0.5 * mass * (vx * vx + vy * vy + vz * vz);
            dof_trans += 3.0;

            let [wx, wy, wz, _] = w;
            ke_rot += 0.5 * (inertia[0] * wx * wx + inertia[1] * wy * wy + inertia[2] * wz * wz);
            dof_rot += 3.0;
        }
    }

    for ((&is_flex, &mass), v) in atom_is_flexible
        .iter()
        .zip(atom_masses)
        .zip(atom_velocities.chunks_exact(3))
    {
        if is_flex != 0 {
            let (vx, vy, vz) = (v[0], v[1], v[2]);
            ke_trans += 0.5 * mass * (vx * vx + vy * vy + vz * vz);
            dof_trans += 3.0;
        }
    }

    const R_KJ_PER_MOL_K: f32 = crate::R_IN_KJ_PER_MOL as f32;
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

/// Convert a `UnitQuaternion` to device layout `[i, j, k, w]` as f32.
pub(super) fn quat_to_gpu(q: &crate::UnitQuaternion) -> [f32; 4] {
    [q.i as f32, q.j as f32, q.k as f32, q.w as f32]
}

/// Convert device layout `[i, j, k, w]` back to a `UnitQuaternion`.
pub(super) fn gpu_to_quat(q: &[f32; 4]) -> crate::UnitQuaternion {
    nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
        q[3] as f64,
        q[0] as f64,
        q[1] as f64,
        q[2] as f64,
    ))
}

/// Pack all atom positions from context into vec4 layout `[x, y, z, atom_type_bits]`.
pub(super) fn pack_positions_f32<T: Context>(context: &T) -> Vec<f32> {
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
pub(super) fn pack_com_and_quaternions<T: Context>(context: &T) -> (Vec<f32>, Vec<f32>) {
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
pub(super) fn reduce_forces_to_com<T: Context>(
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

/// GPU spline data extracted from the Hamiltonian for on-device force computation.
pub(super) struct GpuSplineUpload {
    pub(super) atom_type_ids: Option<Vec<u32>>,
    pub(super) mol_ids: Option<Vec<u32>>,
    pub(super) params: Option<Vec<f32>>,
    pub(super) coeffs: Option<Vec<f32>>,
    pub(super) n_atom_types: u32,
}

/// Extract spline data from the Hamiltonian for on-device force computation.
///
/// Returns `None` fields if no splined nonbonded term exists, falling back to CPU forces.
pub(super) fn extract_spline_data<T: Context>(context: &T) -> GpuSplineUpload {
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
        return GpuSplineUpload {
            atom_type_ids: None,
            mol_ids: None,
            params: None,
            coeffs: None,
            n_atom_types: 0,
        };
    };

    let n_atom_types = context.topology().atomkinds().len() as u32;

    let spline_data =
        interatomic::gpu::GpuSplineData::<interatomic::gpu::PowerLaw2>::from_potentials(
            nb.get_potentials().iter(),
        );
    let spline_params = crate::energy::nonbonded_kernel::repack_spline_params(&spline_data.params);
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

    GpuSplineUpload {
        atom_type_ids: Some(atom_type_ids),
        mol_ids: Some(mol_ids),
        params: Some(spline_params),
        coeffs: Some(spline_coeffs),
        n_atom_types,
    }
}

/// Extract bonded topology (bonds, angles, dihedrals) into CSR layout for GPU upload.
pub(super) fn extract_bonded_data<T: Context>(
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

    (bonds, angles, dihedrals)
}

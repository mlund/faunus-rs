//! CubeCL kernels for rigid-body Langevin (BAOAB) integration.
//!
//! Contains the integrator steps (baoab_step, half_kick), position reconstruction,
//! and atom→COM force/torque reduction. Uses Philox 4x32-10 counter-based RNG.

use cubecl::prelude::*;

// 1 kJ/mol = 100 amu*A^2/ps^2
const KJ_MOL_TO_INTERNAL: f32 = 100.0;

// ============================================================================
// Philox 4x32-10 counter-based RNG
// ============================================================================

/// Full 32x32 -> 64-bit multiply returning (lo, hi) via 16-bit limbs.
#[cube]
fn mulhilo(a: u32, b: u32, lo: &mut u32, hi: &mut u32) {
    let a_lo = a & 0xFFFFu32;
    let a_hi = a >> 16u32;
    let b_lo = b & 0xFFFFu32;
    let b_hi = b >> 16u32;

    let lo_lo = a_lo * b_lo;
    let hi_lo = a_hi * b_lo;
    let lo_hi = a_lo * b_hi;
    let hi_hi = a_hi * b_hi;

    let mid = (lo_lo >> 16u32) + (hi_lo & 0xFFFFu32) + (lo_hi & 0xFFFFu32);
    *lo = (lo_lo & 0xFFFFu32) | ((mid & 0xFFFFu32) << 16u32);
    *hi = hi_hi + (hi_lo >> 16u32) + (lo_hi >> 16u32) + (mid >> 16u32);
}

/// One Philox round: apply S-box permutation with the given key.
#[cube]
fn philox_round(
    c0: u32,
    c1: u32,
    c2: u32,
    c3: u32,
    k0: u32,
    k1: u32,
    o0: &mut u32,
    o1: &mut u32,
    o2: &mut u32,
    o3: &mut u32,
) {
    let mut m0_lo = 0u32;
    let mut m0_hi = 0u32;
    let mut m1_lo = 0u32;
    let mut m1_hi = 0u32;
    mulhilo(0xD2511F53u32, c0, &mut m0_lo, &mut m0_hi);
    mulhilo(0xCD9E8D57u32, c2, &mut m1_lo, &mut m1_hi);
    *o0 = m1_hi ^ c1 ^ k0;
    *o1 = m1_lo;
    *o2 = m0_hi ^ c3 ^ k1;
    *o3 = m0_lo;
}

/// Philox 4x32-10: full 10-round permutation.
#[cube]
fn philox4x32(
    c0: u32,
    c1: u32,
    c2: u32,
    c3: u32,
    k0: u32,
    k1: u32,
    o0: &mut u32,
    o1: &mut u32,
    o2: &mut u32,
    o3: &mut u32,
) {
    let mut r0 = c0;
    let mut r1 = c1;
    let mut r2 = c2;
    let mut r3 = c3;
    let mut kk0 = k0;
    let mut kk1 = k1;

    #[unroll]
    for _ in 0..10u32 {
        let mut n0 = 0u32;
        let mut n1 = 0u32;
        let mut n2 = 0u32;
        let mut n3 = 0u32;
        philox_round(r0, r1, r2, r3, kk0, kk1, &mut n0, &mut n1, &mut n2, &mut n3);
        r0 = n0;
        r1 = n1;
        r2 = n2;
        r3 = n3;
        kk0 += 0x9E3779B9u32;
        kk1 += 0xBB67AE85u32;
    }
    *o0 = r0;
    *o1 = r1;
    *o2 = r2;
    *o3 = r3;
}

/// Box-Muller: two uniform u32 -> two Gaussian f32.
#[cube]
fn box_muller(u0: u32, u1: u32, g0: &mut f32, g1: &mut f32) {
    let uf0 = f32::cast_from(u0) * 2.3283064e-10f32;
    let uf1 = f32::cast_from(u1) * 2.3283064e-10f32;
    let clamped = select(uf0 < 1e-30f32, 1e-30f32, uf0);
    let r = f32::sqrt(-2.0f32 * f32::ln(clamped));
    #[allow(clippy::approx_constant)]
    let theta = 6.283_185_5_f32 * uf1;
    *g0 = r * f32::cos(theta);
    *g1 = r * f32::sin(theta);
}

/// Generate 4 Gaussian random numbers for molecule `mol_id` at step `step`.
#[cube]
fn gaussian4(
    mol_id: u32,
    step: u32,
    seed: u32,
    stream: u32,
    g0: &mut f32,
    g1: &mut f32,
    g2: &mut f32,
    g3: &mut f32,
) {
    let mut r0 = 0u32;
    let mut r1 = 0u32;
    let mut r2 = 0u32;
    let mut r3 = 0u32;
    philox4x32(
        mol_id,
        step,
        stream,
        0u32,
        seed,
        0x12345678u32,
        &mut r0,
        &mut r1,
        &mut r2,
        &mut r3,
    );
    box_muller(r0, r1, g0, g1);
    box_muller(r2, r3, g2, g3);
}

// ============================================================================
// Quaternion utilities (shared by integrator and reconstruction)
// ============================================================================

#[cube]
fn quat_rotate(
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    out_x: &mut f32,
    out_y: &mut f32,
    out_z: &mut f32,
) {
    let tx = 2.0f32 * (qy * vz - qz * vy);
    let ty = 2.0f32 * (qz * vx - qx * vz);
    let tz = 2.0f32 * (qx * vy - qy * vx);
    *out_x = vx + qw * tx + (qy * tz - qz * ty);
    *out_y = vy + qw * ty + (qz * tx - qx * tz);
    *out_z = vz + qw * tz + (qx * ty - qy * tx);
}

#[cube]
fn quat_rotate_inv(
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    out_x: &mut f32,
    out_y: &mut f32,
    out_z: &mut f32,
) {
    quat_rotate(-qx, -qy, -qz, qw, vx, vy, vz, out_x, out_y, out_z);
}

#[cube]
fn quat_mul(
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    px: f32,
    py: f32,
    pz: f32,
    pw: f32,
    out_x: &mut f32,
    out_y: &mut f32,
    out_z: &mut f32,
    out_w: &mut f32,
) {
    *out_x = qw * px + qx * pw + qy * pz - qz * py;
    *out_y = qw * py - qx * pz + qy * pw + qz * px;
    *out_z = qw * pz + qx * py - qy * px + qz * pw;
    *out_w = qw * pw - qx * px - qy * py - qz * pz;
}

#[cube]
fn quat_normalize(
    x: f32,
    y: f32,
    z: f32,
    w: f32,
    ox: &mut f32,
    oy: &mut f32,
    oz: &mut f32,
    ow: &mut f32,
) {
    let len = f32::sqrt(x * x + y * y + z * z + w * w);
    let inv = 1.0f32 / len;
    *ox = x * inv;
    *oy = y * inv;
    *oz = z * inv;
    *ow = w * inv;
}

#[cube]
fn safe_inv(v: f32) -> f32 {
    select(v == 0.0f32, 0.0f32, 1.0f32 / v)
}

// ============================================================================
// BAOAB integrator kernels
// ============================================================================

/// Full BAOAB step: B(half-kick) -> A(drift) -> O(thermostat) -> A(drift).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn baoab_step(
    com_positions: &mut Array<f32>,
    com_velocities: &mut Array<f32>,
    quaternions: &mut Array<f32>,
    angular_velocities: &mut Array<f32>,
    com_forces: &Array<f32>,
    torques_buf: &Array<f32>,
    mol_masses: &Array<f32>,
    mol_inertia: &Array<f32>,
    n_molecules: u32,
    dt: f32,
    friction: f32,
    kt: f32,
    rng_seed: u32,
    rng_step: u32,
    box_length: f32,
) {
    let m = ABSOLUTE_POS;
    if m < n_molecules as usize {
        let half_dt = 0.5f32 * dt;
        let m4 = m * 4;

        let mass = mol_masses[m];
        let inv_mass = 1.0f32 / mass;
        let inv_ix = safe_inv(mol_inertia[m4]);
        let inv_iy = safe_inv(mol_inertia[m4 + 1]);
        let inv_iz = safe_inv(mol_inertia[m4 + 2]);

        let mut rx = com_positions[m4];
        let mut ry = com_positions[m4 + 1];
        let mut rz = com_positions[m4 + 2];
        let mut vx = com_velocities[m4];
        let mut vy = com_velocities[m4 + 1];
        let mut vz = com_velocities[m4 + 2];
        let mut qx = quaternions[m4];
        let mut qy = quaternions[m4 + 1];
        let mut qz = quaternions[m4 + 2];
        let mut qw = quaternions[m4 + 3];
        let mut ox = angular_velocities[m4];
        let mut oy = angular_velocities[m4 + 1];
        let mut oz = angular_velocities[m4 + 2];
        let fx = com_forces[m4];
        let fy = com_forces[m4 + 1];
        let fz = com_forces[m4 + 2];
        let taux = torques_buf[m4];
        let tauy = torques_buf[m4 + 1];
        let tauz = torques_buf[m4 + 2];

        // B step: half-kick velocities
        let kick = half_dt * inv_mass * KJ_MOL_TO_INTERNAL;
        vx += kick * fx;
        vy += kick * fy;
        vz += kick * fz;

        let mut tb_x = 0.0f32;
        let mut tb_y = 0.0f32;
        let mut tb_z = 0.0f32;
        quat_rotate_inv(
            qx, qy, qz, qw, taux, tauy, tauz, &mut tb_x, &mut tb_y, &mut tb_z,
        );
        ox += half_dt * inv_ix * tb_x * KJ_MOL_TO_INTERNAL;
        oy += half_dt * inv_iy * tb_y * KJ_MOL_TO_INTERNAL;
        oz += half_dt * inv_iz * tb_z * KJ_MOL_TO_INTERNAL;

        // A step: half-drift
        rx += half_dt * vx;
        ry += half_dt * vy;
        rz += half_dt * vz;

        // Quaternion drift: dq/dt = 0.5 * q (x) (omega_body, 0)
        let s = half_dt * 0.5f32;
        let mut dqx = 0.0f32;
        let mut dqy = 0.0f32;
        let mut dqz = 0.0f32;
        let mut dqw = 0.0f32;
        quat_mul(
            qx,
            qy,
            qz,
            qw,
            ox * s,
            oy * s,
            oz * s,
            0.0f32,
            &mut dqx,
            &mut dqy,
            &mut dqz,
            &mut dqw,
        );
        qx += dqx;
        qy += dqy;
        qz += dqz;
        qw += dqw;
        let mut nqx = 0.0f32;
        let mut nqy = 0.0f32;
        let mut nqz = 0.0f32;
        let mut nqw = 0.0f32;
        quat_normalize(qx, qy, qz, qw, &mut nqx, &mut nqy, &mut nqz, &mut nqw);
        qx = nqx;
        qy = nqy;
        qz = nqz;
        qw = nqw;

        // O step: Ornstein-Uhlenbeck thermostat
        let a = f32::exp(-friction * dt);
        let b = f32::sqrt(1.0f32 - a * a);

        let m_u32 = m as u32;
        let mut n0 = 0.0f32;
        let mut n1 = 0.0f32;
        let mut n2 = 0.0f32;
        let mut n3 = 0.0f32;
        gaussian4(
            m_u32, rng_step, rng_seed, 0u32, &mut n0, &mut n1, &mut n2, &mut n3,
        );
        let mut nr0 = 0.0f32;
        let mut nr1 = 0.0f32;
        let mut nr2 = 0.0f32;
        let mut nr3 = 0.0f32;
        gaussian4(
            m_u32, rng_step, rng_seed, 1u32, &mut nr0, &mut nr1, &mut nr2, &mut nr3,
        );

        let sigma_trans = f32::sqrt(kt * inv_mass * KJ_MOL_TO_INTERNAL);
        vx = a * vx + b * sigma_trans * n0;
        vy = a * vy + b * sigma_trans * n1;
        vz = a * vz + b * sigma_trans * n2;

        let sigma_ox = f32::sqrt(kt * inv_ix * KJ_MOL_TO_INTERNAL);
        let sigma_oy = f32::sqrt(kt * inv_iy * KJ_MOL_TO_INTERNAL);
        let sigma_oz = f32::sqrt(kt * inv_iz * KJ_MOL_TO_INTERNAL);
        ox = a * ox + b * sigma_ox * nr0;
        oy = a * oy + b * sigma_oy * nr1;
        oz = a * oz + b * sigma_oz * nr2;

        // A step: second half-drift
        rx += half_dt * vx;
        ry += half_dt * vy;
        rz += half_dt * vz;

        let mut dqx2 = 0.0f32;
        let mut dqy2 = 0.0f32;
        let mut dqz2 = 0.0f32;
        let mut dqw2 = 0.0f32;
        quat_mul(
            qx,
            qy,
            qz,
            qw,
            ox * s,
            oy * s,
            oz * s,
            0.0f32,
            &mut dqx2,
            &mut dqy2,
            &mut dqz2,
            &mut dqw2,
        );
        qx += dqx2;
        qy += dqy2;
        qz += dqz2;
        qw += dqw2;
        let mut fqx = 0.0f32;
        let mut fqy = 0.0f32;
        let mut fqz = 0.0f32;
        let mut fqw = 0.0f32;
        quat_normalize(qx, qy, qz, qw, &mut fqx, &mut fqy, &mut fqz, &mut fqw);

        // PBC wrap
        let inv_box = 1.0f32 / box_length;
        rx -= box_length * f32::round(rx * inv_box);
        ry -= box_length * f32::round(ry * inv_box);
        rz -= box_length * f32::round(rz * inv_box);

        com_positions[m4] = rx;
        com_positions[m4 + 1] = ry;
        com_positions[m4 + 2] = rz;
        com_positions[m4 + 3] = 0.0f32;
        com_velocities[m4] = vx;
        com_velocities[m4 + 1] = vy;
        com_velocities[m4 + 2] = vz;
        com_velocities[m4 + 3] = 0.0f32;
        quaternions[m4] = fqx;
        quaternions[m4 + 1] = fqy;
        quaternions[m4 + 2] = fqz;
        quaternions[m4 + 3] = fqw;
        angular_velocities[m4] = ox;
        angular_velocities[m4 + 1] = oy;
        angular_velocities[m4 + 2] = oz;
        angular_velocities[m4 + 3] = 0.0f32;
    }
}

/// Closing B half-kick using forces at the new positions.
#[cube(launch_unchecked)]
pub fn half_kick(
    com_velocities: &mut Array<f32>,
    angular_velocities: &mut Array<f32>,
    com_forces: &Array<f32>,
    torques_buf: &Array<f32>,
    quaternions: &Array<f32>,
    mol_masses: &Array<f32>,
    mol_inertia: &Array<f32>,
    n_molecules: u32,
    dt: f32,
) {
    let m = ABSOLUTE_POS;
    if m < n_molecules as usize {
        let half_dt = 0.5f32 * dt;
        let m4 = m * 4;

        let inv_mass = 1.0f32 / mol_masses[m];
        let inv_ix = safe_inv(mol_inertia[m4]);
        let inv_iy = safe_inv(mol_inertia[m4 + 1]);
        let inv_iz = safe_inv(mol_inertia[m4 + 2]);

        let mut vx = com_velocities[m4];
        let mut vy = com_velocities[m4 + 1];
        let mut vz = com_velocities[m4 + 2];
        let mut ox = angular_velocities[m4];
        let mut oy = angular_velocities[m4 + 1];
        let mut oz = angular_velocities[m4 + 2];

        let kick = half_dt * inv_mass * KJ_MOL_TO_INTERNAL;
        vx += kick * com_forces[m4];
        vy += kick * com_forces[m4 + 1];
        vz += kick * com_forces[m4 + 2];

        let qx = quaternions[m4];
        let qy = quaternions[m4 + 1];
        let qz = quaternions[m4 + 2];
        let qw = quaternions[m4 + 3];
        let mut tb_x = 0.0f32;
        let mut tb_y = 0.0f32;
        let mut tb_z = 0.0f32;
        quat_rotate_inv(
            qx,
            qy,
            qz,
            qw,
            torques_buf[m4],
            torques_buf[m4 + 1],
            torques_buf[m4 + 2],
            &mut tb_x,
            &mut tb_y,
            &mut tb_z,
        );
        ox += half_dt * inv_ix * tb_x * KJ_MOL_TO_INTERNAL;
        oy += half_dt * inv_iy * tb_y * KJ_MOL_TO_INTERNAL;
        oz += half_dt * inv_iz * tb_z * KJ_MOL_TO_INTERNAL;

        com_velocities[m4] = vx;
        com_velocities[m4 + 1] = vy;
        com_velocities[m4 + 2] = vz;
        com_velocities[m4 + 3] = 0.0f32;
        angular_velocities[m4] = ox;
        angular_velocities[m4 + 1] = oy;
        angular_velocities[m4 + 2] = oz;
        angular_velocities[m4 + 3] = 0.0f32;
    }
}

// ============================================================================
// Position reconstruction kernel
// ============================================================================

/// Binary search to find which molecule owns atom `atom_idx`.
#[cube]
fn find_molecule(mol_atom_offsets: &Array<u32>, atom_idx: u32, n_molecules: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = n_molecules;
    loop {
        if lo >= hi {
            break;
        }
        let mid = (lo + hi) / 2;
        if mol_atom_offsets[mid + 1] <= atom_idx {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Reconstruct atom positions from COM positions and quaternion rotations.
#[cube(launch_unchecked)]
pub fn reconstruct_positions(
    com_positions: &Array<f32>,
    quaternions: &Array<f32>,
    ref_positions: &Array<f32>,
    positions: &mut Array<f32>,
    mol_atom_offsets: &Array<u32>,
    n_atoms: u32,
    n_molecules: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_atoms as usize {
        let m = find_molecule(mol_atom_offsets, i as u32, n_molecules as usize);
        let m4 = m * 4;
        let i4 = i * 4;

        let qx = quaternions[m4];
        let qy = quaternions[m4 + 1];
        let qz = quaternions[m4 + 2];
        let qw = quaternions[m4 + 3];

        let mut rot_x = 0.0f32;
        let mut rot_y = 0.0f32;
        let mut rot_z = 0.0f32;
        quat_rotate(
            qx,
            qy,
            qz,
            qw,
            ref_positions[i4],
            ref_positions[i4 + 1],
            ref_positions[i4 + 2],
            &mut rot_x,
            &mut rot_y,
            &mut rot_z,
        );

        let atom_type_w = positions[i4 + 3];
        positions[i4] = com_positions[m4] + rot_x;
        positions[i4 + 1] = com_positions[m4 + 1] + rot_y;
        positions[i4 + 2] = com_positions[m4 + 2] + rot_z;
        positions[i4 + 3] = atom_type_w;
    }
}

// ============================================================================
// Force reduction kernel
// ============================================================================

/// Reduce per-atom forces to per-molecule COM forces and torques.
///
/// Each thread processes one molecule, using `mol_atom_offsets` to determine
/// which atoms belong to it. PBC minimum image applied for lever arms.
#[cube(launch_unchecked)]
pub fn reduce_forces_kernel(
    forces: &Array<f32>,
    positions: &Array<f32>,
    com_positions: &Array<f32>,
    mol_atom_offsets: &Array<u32>,
    com_forces: &mut Array<f32>,
    torques: &mut Array<f32>,
    n_molecules: u32,
    box_length: f32,
    inv_box: f32,
) {
    let m = ABSOLUTE_POS;
    if m < n_molecules as usize {
        let start = mol_atom_offsets[m] as usize;
        let end = mol_atom_offsets[m + 1] as usize;
        let m4 = m * 4;
        let cx = com_positions[m4];
        let cy = com_positions[m4 + 1];
        let cz = com_positions[m4 + 2];

        let mut fx_sum = 0.0f32;
        let mut fy_sum = 0.0f32;
        let mut fz_sum = 0.0f32;
        let mut tx_sum = 0.0f32;
        let mut ty_sum = 0.0f32;
        let mut tz_sum = 0.0f32;

        for a in start..end {
            let a4 = a * 4;
            let fi_x = forces[a4];
            let fi_y = forces[a4 + 1];
            let fi_z = forces[a4 + 2];

            fx_sum += fi_x;
            fy_sum += fi_y;
            fz_sum += fi_z;

            // PBC-aware lever arm: r_atom - r_com
            let mut lx = positions[a4] - cx;
            let mut ly = positions[a4 + 1] - cy;
            let mut lz = positions[a4 + 2] - cz;
            lx -= box_length * f32::round(lx * inv_box);
            ly -= box_length * f32::round(ly * inv_box);
            lz -= box_length * f32::round(lz * inv_box);

            // torque = lever x force
            tx_sum += ly * fi_z - lz * fi_y;
            ty_sum += lz * fi_x - lx * fi_z;
            tz_sum += lx * fi_y - ly * fi_x;
        }

        com_forces[m4] = fx_sum;
        com_forces[m4 + 1] = fy_sum;
        com_forces[m4 + 2] = fz_sum;
        com_forces[m4 + 3] = 0.0;

        torques[m4] = tx_sum;
        torques[m4 + 1] = ty_sum;
        torques[m4 + 2] = tz_sum;
        torques[m4 + 3] = 0.0;
    }
}

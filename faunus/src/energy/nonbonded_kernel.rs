//! CubeCL kernel for pairwise nonbonded forces using spline lookup.
//!
//! Each thread computes the total force on one atom by looping over all other
//! atoms. This avoids atomic accumulation at the cost of evaluating each pair
//! twice (no Newton's 3rd law). Intra-molecular handling differs by molecule
//! type: rigid bodies skip all intra-molecular pairs (internal distances are
//! constant), while flexible molecules skip only topology-excluded pairs
//! (e.g. directly bonded neighbors) via a per-atom exclusion CSR.

use cubecl::prelude::*;

/// PowerLaw2 spline force evaluation: returns `-dU/d(r^2)` for the pair.
///
/// Spline params layout (stride 8): `[r_min, r_max, n_coeffs, coeff_offset, f_at_rmin, 0, 0, 0]`
/// Spline coeffs layout (stride 8): `[u0, u1, u2, u3, f0, f1, f2, f3]`
#[cube]
fn spline_force(
    rsq: f32,
    kind_i: u32,
    kind_j: u32,
    n_atom_types: u32,
    spline_params: &Array<f32>,
    spline_coeffs: &Array<f32>,
    result: &mut f32,
) {
    let base = (kind_i * n_atom_types + kind_j) as usize * 8;
    let r_min = spline_params[base];
    let r_max = spline_params[base + 1];
    let n_coeffs = spline_params[base + 2];
    let coeff_offset = spline_params[base + 3];
    let f_at_rmin = spline_params[base + 4];

    let rsq_max = r_max * r_max;

    if rsq >= rsq_max {
        *result = 0.0;
    } else {
        let r = f32::sqrt(rsq);
        let r_clamped = select(r < r_min, r_min, r);
        let r_range = r_max - r_min;
        let x = f32::sqrt((r_clamped - r_min) / r_range);
        let t = x * (n_coeffs - 1.0);
        let idx = u32::cast_from(t);
        let max_idx = u32::cast_from(n_coeffs) - 2;
        let idx_clamped = select(idx > max_idx, max_idx, idx);
        let eps = t - f32::cast_from(idx_clamped);

        let c_base = (u32::cast_from(coeff_offset) + idx_clamped) as usize * 8;
        let f0 = spline_coeffs[c_base + 4];
        let f1 = spline_coeffs[c_base + 5];
        let f2 = spline_coeffs[c_base + 6];
        let f3 = spline_coeffs[c_base + 7];

        // Horner's method; below r_min, scale by r_min/r to match constant -dU/dr
        let f_spline = f0 + eps * (f1 + eps * (f2 + eps * f3));
        *result = select(r < r_min, f_at_rmin * r_min / r, f_spline);
    }
}

/// Check whether atom pair (i, j) is in the exclusion CSR list.
/// Returns 1 if excluded, 0 otherwise.
///
/// Uses u32 instead of bool because CubeCL's `#[cube]` expander cannot
/// return native `bool` from branching expressions. Linear scan is fine
/// since lists are short (typically 1–3 entries for `excluded_neighbours=1`).
#[cube]
fn is_excluded(i: u32, j: u32, excl_offsets: &Array<u32>, excl_atoms: &Array<u32>) -> u32 {
    let start = excl_offsets[i as usize] as usize;
    let end = excl_offsets[i as usize + 1] as usize;
    let mut found = 0u32;
    for k in start..end {
        if excl_atoms[k] == j {
            found = 1u32;
        }
    }
    found
}

/// Compute per-atom pairwise nonbonded forces using PowerLaw2 spline lookup.
///
/// Each thread handles one atom `i`, looping over all atoms `j`.
/// Intra-molecular pairs are handled per molecule type:
/// - Rigid molecules: skip all intra-molecular pairs (via `mol_is_rigid`)
/// - Flexible molecules: compute unless the pair is in the exclusion CSR
/// PBC minimum image convention applied. Output forces in `[fx, fy, fz, 0]` layout.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn pair_forces_kernel(
    positions: &Array<f32>,
    atom_types: &Array<u32>,
    mol_ids: &Array<u32>,
    spline_params: &Array<f32>,
    spline_coeffs: &Array<f32>,
    excl_offsets: &Array<u32>,
    excl_atoms: &Array<u32>,
    mol_is_rigid: &Array<u32>,
    forces: &mut Array<f32>,
    n_atoms: u32,
    n_atom_types: u32,
    box_length: f32,
    inv_box: f32,
) {
    let i = ABSOLUTE_POS;
    if i < n_atoms as usize {
        let i4 = i * 4;
        let xi = positions[i4];
        let yi = positions[i4 + 1];
        let zi = positions[i4 + 2];
        let kind_i = atom_types[i];
        let mol_i = mol_ids[i];
        // Hoist rigid flag to avoid repeated array read in the inner loop
        let rigid_i = mol_is_rigid[mol_i as usize];

        let mut fx = 0.0f32;
        let mut fy = 0.0f32;
        let mut fz = 0.0f32;

        for j in 0..n_atoms as usize {
            if j != i {
                // Rigid: all intra-mol distances are constant, so NB forces
                // cancel in the COM reduction and can be skipped entirely.
                // Flexible: intra-mol NB forces are physical (e.g. WCA between
                // non-bonded monomers), skip only topology-excluded pairs.
                let mut skip = 0u32;
                if mol_ids[j] == mol_i {
                    if rigid_i != 0u32 {
                        skip = 1u32;
                    } else {
                        skip = is_excluded(i as u32, j as u32, excl_offsets, excl_atoms);
                    }
                }

                if skip == 0u32 {
                    let j4 = j * 4;
                    let mut dx = xi - positions[j4];
                    let mut dy = yi - positions[j4 + 1];
                    let mut dz = zi - positions[j4 + 2];
                    dx -= box_length * f32::round(dx * inv_box);
                    dy -= box_length * f32::round(dy * inv_box);
                    dz -= box_length * f32::round(dz * inv_box);

                    let rsq = dx * dx + dy * dy + dz * dz;
                    let kind_j = atom_types[j];

                    let mut f_mag = 0.0f32;
                    spline_force(
                        rsq,
                        kind_i,
                        kind_j,
                        n_atom_types,
                        spline_params,
                        spline_coeffs,
                        &mut f_mag,
                    );

                    // f_mag = -dU/d(r^2); dr = ri - rj (j->i); F_i = 2*f_mag*dr
                    let scale = 2.0 * f_mag;
                    fx += scale * dx;
                    fy += scale * dy;
                    fz += scale * dz;
                }
            }
        }

        forces[i4] = fx;
        forces[i4 + 1] = fy;
        forces[i4 + 2] = fz;
        forces[i4 + 3] = 0.0;
    }
}

/// Repack `PowerLaw2Params` from interatomic into a flat f32 array for CubeCL.
///
/// Layout per pair (stride 8): `[r_min, r_max, n_coeffs, coeff_offset, f_at_rmin, 0, 0, 0]`
pub fn repack_spline_params(params: &[interatomic::gpu::PowerLaw2Params]) -> Vec<f32> {
    params
        .iter()
        .flat_map(|p| {
            [
                p.r_min,
                p.r_max,
                p.n_coeffs as f32,
                p.coeff_offset as f32,
                p.f_at_rmin,
                0.0,
                0.0,
                0.0,
            ]
        })
        .collect()
}

/// Repack `GpuSplineCoeffs` into a flat f32 array for CubeCL.
///
/// Layout per interval (stride 8): `[u0, u1, u2, u3, f0, f1, f2, f3]`
pub fn repack_spline_coeffs(coeffs: &[interatomic::gpu::GpuSplineCoeffs]) -> Vec<f32> {
    coeffs
        .iter()
        .flat_map(|c| {
            [
                c.u[0], c.u[1], c.u[2], c.u[3], c.f[0], c.f[1], c.f[2], c.f[3],
            ]
        })
        .collect()
}

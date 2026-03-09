//! CubeCL kernel for pairwise nonbonded forces using spline lookup.
//!
//! Each thread computes the total force on one atom by iterating over
//! 27 neighbor cells from a CPU-built cell list (CSR layout). This gives
//! O(n·k) scaling instead of O(n²). Intra-molecular handling differs by
//! molecule type: rigid bodies skip all intra-molecular pairs (internal
//! distances are constant), while flexible molecules skip only
//! topology-excluded pairs via a per-atom exclusion CSR.

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

/// Compute per-atom pairwise nonbonded forces using cell-list neighbor iteration.
///
/// Each thread handles one atom `i`, iterating over 27 neighbor cells via
/// a CSR cell list. Sentinel value `u32::MAX` marks unused neighbor slots.
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
    cell_offsets: &Array<u32>,
    cell_atoms: &Array<u32>,
    neighbor_cells: &Array<u32>,
    forces: &mut Array<f32>,
    n_atoms: u32,
    n_atom_types: u32,
    box_length: f32,
    inv_box: f32,
    n_cells_1d: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_atoms as usize {
        let i4 = i * 4;
        let xi = positions[i4];
        let yi = positions[i4 + 1];
        let zi = positions[i4 + 2];
        let kind_i = atom_types[i];
        let mol_i = mol_ids[i];
        let rigid_i = mol_is_rigid[mol_i as usize];

        // Shift from [-L/2, L/2] to [0, L] so floor() gives non-negative cell indices
        let cell_size = box_length / f32::cast_from(n_cells_1d);
        let inv_cell = 1.0 / cell_size;
        let half_box = 0.5 * box_length;
        let cix = u32::cast_from(f32::floor((xi + half_box) * inv_cell)) % n_cells_1d;
        let ciy = u32::cast_from(f32::floor((yi + half_box) * inv_cell)) % n_cells_1d;
        let ciz = u32::cast_from(f32::floor((zi + half_box) * inv_cell)) % n_cells_1d;
        let ci = (cix * n_cells_1d + ciy) * n_cells_1d + ciz;

        let mut fx = 0.0f32;
        let mut fy = 0.0f32;
        let mut fz = 0.0f32;

        // Iterate over 27 neighbor cells
        let nb_base = ci as usize * 27;
        for k in 0u32..27u32 {
            let nc = neighbor_cells[nb_base + k as usize];
            // When n_cells_1d < 3, PBC wrapping produces duplicate neighbors;
            // these are deduplicated on CPU and unused slots are sentinel-filled
            if nc != u32::MAX {
                let start = cell_offsets[nc as usize];
                let end = cell_offsets[nc as usize + 1];
                for idx in start..end {
                    let j = cell_atoms[idx as usize] as usize;
                    if j != i {
                        // Rigid: intra-mol NB forces cancel in COM reduction.
                        // Flexible: only skip topology-excluded pairs (e.g. bonded neighbors)
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

                            let scale = 2.0 * f_mag;
                            fx += scale * dx;
                            fy += scale * dy;
                            fz += scale * dz;
                        }
                    }
                }
            }
        }

        forces[i4] = fx;
        forces[i4 + 1] = fy;
        forces[i4 + 2] = fz;
        forces[i4 + 3] = 0.0;
    }
}

/// Extract max cutoff (r_max) from flat spline params array.
///
/// Spline params stride is 8; index 1 in each stride is r_max.
pub fn max_cutoff_from_spline_params(spline_params: &[f32]) -> f32 {
    spline_params
        .chunks_exact(8)
        .map(|chunk| chunk[1])
        .reduce(f32::max)
        .unwrap_or(0.0)
}

/// Compute number of cells per dimension so that cell_size >= cutoff.
pub fn compute_n_cells_1d(box_length: f32, cutoff: f32) -> u32 {
    if cutoff <= 0.0 {
        return 1;
    }
    (box_length / cutoff).floor().max(1.0) as u32
}

/// Build flat neighbor cell table `[n³ × 27]` with PBC-wrapped indices.
///
/// For each cell, stores up to 27 neighbor cell indices (including self).
/// When `n < 3`, some neighbors are duplicates; these are deduplicated and
/// remaining slots filled with `u32::MAX` sentinel.
pub fn build_neighbor_cell_table(n: u32) -> Vec<u32> {
    let n3 = (n * n * n) as usize;
    let mut table = vec![u32::MAX; n3 * 27];
    for cx in 0..n {
        for cy in 0..n {
            for cz in 0..n {
                let ci = ((cx * n + cy) * n + cz) as usize;
                let mut slot = 0usize;
                let mut written = [u32::MAX; 27];
                // Use n-1 instead of -1 to stay in unsigned arithmetic with PBC wrapping
                for dx in [n - 1, 0, 1] {
                    for dy in [n - 1, 0, 1] {
                        for dz in [n - 1, 0, 1] {
                            let nx = (cx + dx) % n;
                            let ny = (cy + dy) % n;
                            let nz = (cz + dz) % n;
                            let nc = (nx * n + ny) * n + nz;
                            // Deduplicate (relevant when n < 3)
                            if !written[..slot].contains(&nc) {
                                written[slot] = nc;
                                table[ci * 27 + slot] = nc;
                                slot += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    table
}

/// Build cell list CSR from positions in `[x, y, z, _]` layout.
///
/// Returns `(offsets, atoms)` where `offsets` has length `n_cells³ + 1`.
/// Positions are assumed centered around origin (range `[-L/2, L/2]`).
pub fn build_cell_list(
    positions: &[[f32; 4]],
    box_length: f32,
    n_cells_1d: u32,
) -> (Vec<u32>, Vec<u32>) {
    let n3 = (n_cells_1d * n_cells_1d * n_cells_1d) as usize;
    let cell_size = box_length / n_cells_1d as f32;
    let inv_cell = 1.0 / cell_size;
    let half_box = 0.5 * box_length;

    // Count atoms per cell
    let mut counts = vec![0u32; n3];
    for pos in positions {
        let ci = cell_index(pos[0], pos[1], pos[2], half_box, inv_cell, n_cells_1d);
        counts[ci as usize] += 1;
    }

    // Prefix sum → offsets
    let mut offsets = Vec::with_capacity(n3 + 1);
    offsets.push(0u32);
    for &c in &counts {
        offsets.push(offsets.last().unwrap() + c);
    }

    // Scatter atom indices
    let mut atoms = vec![0u32; positions.len()];
    let mut cursors = vec![0u32; n3];
    for (atom_idx, pos) in positions.iter().enumerate() {
        let ci = cell_index(pos[0], pos[1], pos[2], half_box, inv_cell, n_cells_1d) as usize;
        atoms[(offsets[ci] + cursors[ci]) as usize] = atom_idx as u32;
        cursors[ci] += 1;
    }

    (offsets, atoms)
}

/// Duplicated in `pair_forces_kernel` (CubeCL cannot call non-`#[cube]` functions).
fn cell_index(x: f32, y: f32, z: f32, half_box: f32, inv_cell: f32, n: u32) -> u32 {
    // rem_euclid handles particles that drifted slightly outside [-L/2, L/2]
    let cx = ((x + half_box) * inv_cell).floor().rem_euclid(n as f32) as u32;
    let cy = ((y + half_box) * inv_cell).floor().rem_euclid(n as f32) as u32;
    let cz = ((z + half_box) * inv_cell).floor().rem_euclid(n as f32) as u32;
    (cx * n + cy) * n + cz
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neighbor_cell_table_n3() {
        let table = build_neighbor_cell_table(3);
        assert_eq!(table.len(), 27 * 27);
        // Each cell should have exactly 27 unique neighbors (including self)
        for ci in 0..27 {
            let neighbors: Vec<u32> = table[ci * 27..(ci + 1) * 27]
                .iter()
                .copied()
                .filter(|&v| v != u32::MAX)
                .collect();
            assert_eq!(neighbors.len(), 27, "cell {ci} should have 27 neighbors");
        }
    }

    #[test]
    fn test_neighbor_cell_table_n1() {
        let table = build_neighbor_cell_table(1);
        assert_eq!(table.len(), 27);
        // Only 1 cell total, so only 1 unique neighbor
        let unique: Vec<u32> = table.iter().copied().filter(|&v| v != u32::MAX).collect();
        assert_eq!(unique.len(), 1);
        assert_eq!(unique[0], 0);
    }

    #[test]
    fn test_neighbor_cell_table_n2() {
        let table = build_neighbor_cell_table(2);
        // n=2: each cell wraps to all 8 cells
        for ci in 0..8 {
            let unique: Vec<u32> = table[ci * 27..(ci + 1) * 27]
                .iter()
                .copied()
                .filter(|&v| v != u32::MAX)
                .collect();
            assert_eq!(unique.len(), 8, "cell {ci} should have 8 neighbors for n=2");
        }
    }

    #[test]
    fn test_build_cell_list() {
        let positions = vec![
            [-4.0f32, -4.0, -4.0, 0.0],
            [4.0, 4.0, 4.0, 0.0],
            [-4.0, 4.0, -4.0, 0.0],
        ];
        let box_length = 10.0;
        let n = 3; // 3 cells per dim, cell_size ~ 3.33

        let (offsets, atoms) = build_cell_list(&positions, box_length, n);
        assert_eq!(offsets.len(), 28); // 27 cells + 1
        assert_eq!(atoms.len(), 3);
        // All atoms should appear exactly once
        let mut sorted = atoms.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_compute_n_cells_1d() {
        assert_eq!(compute_n_cells_1d(10.0, 3.0), 3);
        assert_eq!(compute_n_cells_1d(10.0, 10.0), 1);
        assert_eq!(compute_n_cells_1d(10.0, 0.0), 1);
        assert_eq!(compute_n_cells_1d(10.0, 2.0), 5);
    }

    #[test]
    fn test_max_cutoff_from_spline_params() {
        // stride 8, index 1 is r_max
        let params = vec![
            0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // pair 1: r_max=5
            0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // pair 2: r_max=8
        ];
        assert_eq!(max_cutoff_from_spline_params(&params), 8.0);
    }
}

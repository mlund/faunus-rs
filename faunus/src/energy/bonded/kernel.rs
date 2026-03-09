//! CubeCL kernels for intramolecular bonded forces.
//!
//! Uses a per-atom CSR layout so each GPU thread handles one atom, avoiding
//! float atomics which CubeCL does not support. OpenMM instead launches one
//! thread per interaction and uses `atomicAdd` for shared-atom accumulation,
//! but that requires float atomics. Our trade-off is redundant work: each
//! bond is visited 2×, each angle 3×, each dihedral 4×, which is acceptable
//! for the O(N) bonded interaction count.

use cubecl::prelude::*;

const DIHEDRAL_TYPE_HARMONIC: u32 = 0;
const DIHEDRAL_TYPE_PERIODIC: u32 = 1;

// ============================================================
// Kernel: harmonic bond forces
// ============================================================

/// Per-atom harmonic bond force accumulation using CSR neighbor list.
///
/// CSR layout:
/// - `offsets[i]..offsets[i+1]`: bond entries for atom i
/// - `neighbors[b]`: partner atom index
/// - `params[b*2]`: spring constant k, `params[b*2+1]`: equilibrium distance
///
/// Forces are added to the existing `forces` buffer (`+=`).
#[cube(launch_unchecked)]
pub fn bond_forces_kernel(
    positions: &Array<f32>,
    offsets: &Array<u32>,
    neighbors: &Array<u32>,
    params: &Array<f32>,
    forces: &mut Array<f32>,
    n_atoms: u32,
    box_length: f32,
    inv_box: f32,
) {
    let i = ABSOLUTE_POS;
    // CubeCL lacks `return`; guard with `if` instead of early exit
    if i < n_atoms as usize {
        // Stride-4 layout matches nonbonded kernel (xyz + padding)
        let i4 = i * 4;
        let xi = positions[i4];
        let yi = positions[i4 + 1];
        let zi = positions[i4 + 2];

        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;

        let mut fx = 0.0f32;
        let mut fy = 0.0f32;
        let mut fz = 0.0f32;

        for b in start..end {
            let j = neighbors[b] as usize;
            let j4 = j * 4;

            // Minimum image convention
            let mut dx = xi - positions[j4];
            let mut dy = yi - positions[j4 + 1];
            let mut dz = zi - positions[j4 + 2];
            dx -= box_length * f32::round(dx * inv_box);
            dy -= box_length * f32::round(dy * inv_box);
            dz -= box_length * f32::round(dz * inv_box);

            let rsq = dx * dx + dy * dy + dz * dz;
            let p = b * 2;
            let k = params[p];
            let req = params[p + 1];

            // F = -dU/dr · r̂ = -k(r - req) · dr/r
            let r = f32::sqrt(rsq);
            let scale = -k * (r - req) / r;
            fx += scale * dx;
            fy += scale * dy;
            fz += scale * dz;
        }

        // Accumulate on top of nonbonded forces already in buffer
        forces[i4] += fx;
        forces[i4 + 1] += fy;
        forces[i4 + 2] += fz;
    }
}

// ============================================================
// Kernel: valence angle forces
// ============================================================

/// Per-atom angle force accumulation using CSR angle list.
///
/// Uses the OpenMM angle force Jacobian: given vertex at atom 2 with
/// v0 = pos2−pos1, v1 = pos2−pos3, cp = v0×v1,
/// F₁ = de_dangle · (v0×cp)/(|v0|²·|cp|),
/// F₃ = de_dangle · (cp×v1)/(|v1|²·|cp|),
/// F₂ = −(F₁+F₃).
///
/// CSR layout per entry:
/// - `atoms[e*2]`, `atoms[e*2+1]`: two other atom indices
/// - `params[e*4]`: role (0=end_a, 1=vertex, 2=end_c)
/// - `params[e*4+1]`: k_eff (radian-based spring constant)
/// - `params[e*4+2]`: θ_eq in radians
// CubeCL requires pre-initialized `let mut` for branch-assigned variables,
// triggering false-positive "value never read" warnings.
#[cube(launch_unchecked)]
#[allow(unused_assignments)]
pub fn angle_forces_kernel(
    positions: &Array<f32>,
    offsets: &Array<u32>,
    atoms: &Array<u32>,
    params: &Array<f32>,
    forces: &mut Array<f32>,
    n_atoms: u32,
    box_length: f32,
    inv_box: f32,
) {
    let i = ABSOLUTE_POS;
    if i < n_atoms as usize {
        let i4 = i * 4;
        let my_x = positions[i4];
        let my_y = positions[i4 + 1];
        let my_z = positions[i4 + 2];

        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;

        let mut fx = 0.0f32;
        let mut fy = 0.0f32;
        let mut fz = 0.0f32;

        for e in start..end {
            let o1 = atoms[e * 2] as usize;
            let o2 = atoms[e * 2 + 1] as usize;
            let p = e * 4;
            let role = u32::cast_from(params[p]);
            let k_eff = params[p + 1];
            let theta_eq = params[p + 2];

            let o14 = o1 * 4;
            let o24 = o2 * 4;
            let o1x = positions[o14];
            let o1y = positions[o14 + 1];
            let o1z = positions[o14 + 2];
            let o2x = positions[o24];
            let o2y = positions[o24 + 1];
            let o2z = positions[o24 + 2];

            // Reconstruct v0 = vertex − end_a, v1 = vertex − end_c
            // role 0 (I'm end_a): vertex=other1, end_c=other2
            // role 1 (I'm vertex): end_a=other1, end_c=other2
            // role 2 (I'm end_c): end_a=other1, vertex=other2
            let mut v0x = 0.0f32;
            let mut v0y = 0.0f32;
            let mut v0z = 0.0f32;
            let mut v1x = 0.0f32;
            let mut v1y = 0.0f32;
            let mut v1z = 0.0f32;

            if role == 0 {
                v0x = o1x - my_x;
                v0y = o1y - my_y;
                v0z = o1z - my_z;
                v1x = o1x - o2x;
                v1y = o1y - o2y;
                v1z = o1z - o2z;
            } else if role == 1 {
                v0x = my_x - o1x;
                v0y = my_y - o1y;
                v0z = my_z - o1z;
                v1x = my_x - o2x;
                v1y = my_y - o2y;
                v1z = my_z - o2z;
            } else {
                v0x = o2x - o1x;
                v0y = o2y - o1y;
                v0z = o2z - o1z;
                v1x = o2x - my_x;
                v1y = o2y - my_y;
                v1z = o2z - my_z;
            }

            // PBC
            v0x -= box_length * f32::round(v0x * inv_box);
            v0y -= box_length * f32::round(v0y * inv_box);
            v0z -= box_length * f32::round(v0z * inv_box);
            v1x -= box_length * f32::round(v1x * inv_box);
            v1y -= box_length * f32::round(v1y * inv_box);
            v1z -= box_length * f32::round(v1z * inv_box);

            // cp = v0 × v1
            let cpx = v0y * v1z - v0z * v1y;
            let cpy = v0z * v1x - v0x * v1z;
            let cpz = v0x * v1y - v0y * v1x;

            // Clamp to avoid NaN from degenerate (collinear) angles
            let rp = f32::max(f32::sqrt(cpx * cpx + cpy * cpy + cpz * cpz), 1.0e-6);
            let r21 = v0x * v0x + v0y * v0y + v0z * v0z;
            let r23 = v1x * v1x + v1y * v1y + v1z * v1z;
            let d = v0x * v1x + v0y * v1y + v0z * v1z;

            let cos_theta = (d / f32::sqrt(r21 * r23)).clamp(-1.0, 1.0);
            let theta = f32::acos(cos_theta);
            let de_dangle = k_eff * (theta - theta_eq);

            // OpenMM angle Jacobian (angleForce.cc):
            // F_end_a = de_dangle · (v0 × cp) / (|v0|² · |cp|)
            let c1x = v0y * cpz - v0z * cpy;
            let c1y = v0z * cpx - v0x * cpz;
            let c1z = v0x * cpy - v0y * cpx;
            let s1 = de_dangle / (r21 * rp);

            // F_end_c = de_dangle · (cp × v1) / (r23 · rp)
            let c3x = cpy * v1z - cpz * v1y;
            let c3y = cpz * v1x - cpx * v1z;
            let c3z = cpx * v1y - cpy * v1x;
            let s3 = de_dangle / (r23 * rp);

            // Pick force for this atom's role; vertex gets -(F_a + F_c) by Newton's 3rd law
            if role == 0 {
                fx += c1x * s1;
                fy += c1y * s1;
                fz += c1z * s1;
            } else if role == 2 {
                fx += c3x * s3;
                fy += c3y * s3;
                fz += c3z * s3;
            } else {
                fx -= c1x * s1 + c3x * s3;
                fy -= c1y * s1 + c3y * s3;
                fz -= c1z * s1 + c3z * s3;
            }
        }

        forces[i4] += fx;
        forces[i4 + 1] += fy;
        forces[i4 + 2] += fz;
    }
}

// ============================================================
// Kernel: dihedral forces
// ============================================================

/// Per-atom dihedral force accumulation using CSR dihedral list.
///
/// Uses the OpenMM torsion force Jacobian with vectors
/// v0 = pos1−pos2, v1 = pos3−pos2, v2 = pos3−pos4,
/// plane normals cp0 = v0×v1, cp1 = v1×v2.
///
/// CSR layout per entry:
/// - `atoms[e*3]..atoms[e*3+2]`: three other atom indices
/// - `params[e*8+0]`: role (0–3, which atom in the quartet)
/// - `params[e*8+1]`: type tag (0=harmonic, 1=periodic)
/// - `params[e*8+2..7]`: type-dependent parameters
///   - harmonic: [k_eff, φ_eq_rad, -, -]
///   - periodic: [k, φ₀_rad, n, -]
// Same CubeCL branch-init pattern as the angle kernel
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, unused_assignments)]
pub fn dihedral_forces_kernel(
    positions: &Array<f32>,
    offsets: &Array<u32>,
    atoms: &Array<u32>,
    params: &Array<f32>,
    forces: &mut Array<f32>,
    n_atoms: u32,
    box_length: f32,
    inv_box: f32,
) {
    let i = ABSOLUTE_POS;
    if i < n_atoms as usize {
        let i4 = i * 4;
        let my_x = positions[i4];
        let my_y = positions[i4 + 1];
        let my_z = positions[i4 + 2];

        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;

        let mut fx = 0.0f32;
        let mut fy = 0.0f32;
        let mut fz = 0.0f32;

        for e in start..end {
            let a3 = e * 3;
            let o1 = atoms[a3] as usize;
            let o2 = atoms[a3 + 1] as usize;
            let o3 = atoms[a3 + 2] as usize;
            let p = e * 8;
            let role = u32::cast_from(params[p]);
            let dtype = u32::cast_from(params[p + 1]);

            // Load positions of the three "other" atoms
            let o14 = o1 * 4;
            let o24 = o2 * 4;
            let o34 = o3 * 4;
            let o1x = positions[o14];
            let o1y = positions[o14 + 1];
            let o1z = positions[o14 + 2];
            let o2x = positions[o24];
            let o2y = positions[o24 + 1];
            let o2z = positions[o24 + 2];
            let o3x = positions[o34];
            let o3y = positions[o34 + 1];
            let o3z = positions[o34 + 2];

            // Reconstruct positions of atoms 1,2,3,4 from role.
            // role 0: me=1, other1=2, other2=3, other3=4
            // role 1: me=2, other1=1, other2=3, other3=4
            // role 2: me=3, other1=1, other2=2, other3=4
            // role 3: me=4, other1=1, other2=2, other3=3
            let mut p1x = 0.0f32;
            let mut p1y = 0.0f32;
            let mut p1z = 0.0f32;
            let mut p2x = 0.0f32;
            let mut p2y = 0.0f32;
            let mut p2z = 0.0f32;
            let mut p3x = 0.0f32;
            let mut p3y = 0.0f32;
            let mut p3z = 0.0f32;
            let mut p4x = 0.0f32;
            let mut p4y = 0.0f32;
            let mut p4z = 0.0f32;

            if role == 0 {
                p1x = my_x;
                p1y = my_y;
                p1z = my_z;
                p2x = o1x;
                p2y = o1y;
                p2z = o1z;
                p3x = o2x;
                p3y = o2y;
                p3z = o2z;
                p4x = o3x;
                p4y = o3y;
                p4z = o3z;
            } else if role == 1 {
                p1x = o1x;
                p1y = o1y;
                p1z = o1z;
                p2x = my_x;
                p2y = my_y;
                p2z = my_z;
                p3x = o2x;
                p3y = o2y;
                p3z = o2z;
                p4x = o3x;
                p4y = o3y;
                p4z = o3z;
            } else if role == 2 {
                p1x = o1x;
                p1y = o1y;
                p1z = o1z;
                p2x = o2x;
                p2y = o2y;
                p2z = o2z;
                p3x = my_x;
                p3y = my_y;
                p3z = my_z;
                p4x = o3x;
                p4y = o3y;
                p4z = o3z;
            } else {
                p1x = o1x;
                p1y = o1y;
                p1z = o1z;
                p2x = o2x;
                p2y = o2y;
                p2z = o2z;
                p3x = o3x;
                p3y = o3y;
                p3z = o3z;
                p4x = my_x;
                p4y = my_y;
                p4z = my_z;
            }

            // v0 = pos1 - pos2, v1 = pos3 - pos2, v2 = pos3 - pos4
            let mut v0x = p1x - p2x;
            let mut v0y = p1y - p2y;
            let mut v0z = p1z - p2z;
            let mut v1x = p3x - p2x;
            let mut v1y = p3y - p2y;
            let mut v1z = p3z - p2z;
            let mut v2x = p3x - p4x;
            let mut v2y = p3y - p4y;
            let mut v2z = p3z - p4z;

            // PBC
            v0x -= box_length * f32::round(v0x * inv_box);
            v0y -= box_length * f32::round(v0y * inv_box);
            v0z -= box_length * f32::round(v0z * inv_box);
            v1x -= box_length * f32::round(v1x * inv_box);
            v1y -= box_length * f32::round(v1y * inv_box);
            v1z -= box_length * f32::round(v1z * inv_box);
            v2x -= box_length * f32::round(v2x * inv_box);
            v2y -= box_length * f32::round(v2y * inv_box);
            v2z -= box_length * f32::round(v2z * inv_box);

            // cp0 = v0 × v1 (normal to plane 1-2-3)
            let cp0x = v0y * v1z - v0z * v1y;
            let cp0y = v0z * v1x - v0x * v1z;
            let cp0z = v0x * v1y - v0y * v1x;
            // cp1 = v1 × v2 (normal to plane 2-3-4)
            let cp1x = v1y * v2z - v1z * v2y;
            let cp1y = v1z * v2x - v1x * v2z;
            let cp1z = v1x * v2y - v1y * v2x;

            // Clamp to avoid NaN from degenerate (collinear) geometries
            let norm_cross1 = cp0x * cp0x + cp0y * cp0y + cp0z * cp0z;
            let norm_cross2 = cp1x * cp1x + cp1y * cp1y + cp1z * cp1z;
            let inv_nc1 = 1.0 / f32::max(f32::sqrt(norm_cross1), 1.0e-6);
            let inv_nc2 = 1.0 / f32::max(f32::sqrt(norm_cross2), 1.0e-6);

            // Dihedral angle from dot product of normalized plane normals
            let cos_angle = (cp0x * cp1x + cp0y * cp1y + cp0z * cp1z) * inv_nc1 * inv_nc2;
            let cos_angle_clamped = cos_angle.clamp(-1.0, 1.0);
            let mut theta = f32::acos(cos_angle_clamped);

            // Sign convention: positive when v0 and cp1 point in the same half-space
            let sign_dot = v0x * cp1x + v0y * cp1y + v0z * cp1z;
            theta = select(sign_dot >= 0.0, theta, -theta);

            // dU/dθ depends on type
            let mut de_dangle = 0.0f32;
            if dtype == DIHEDRAL_TYPE_HARMONIC {
                let k_eff = params[p + 2];
                let phi_eq = params[p + 3];
                de_dangle = k_eff * (theta - phi_eq);
            } else if dtype == DIHEDRAL_TYPE_PERIODIC {
                let k = params[p + 2];
                let phi0 = params[p + 3];
                let n = params[p + 4];
                de_dangle = -k * n * f32::sin(n * theta - phi0);
            }

            // OpenMM torsion Jacobian (torsionForce.cc / periodicTorsionForce.cc):
            // projects dU/dθ onto Cartesian forces via the two plane normals
            let norm_sqr_bc = v1x * v1x + v1y * v1y + v1z * v1z;
            let norm_bc = f32::sqrt(norm_sqr_bc);
            let dp = 1.0 / f32::max(norm_sqr_bc, 1.0e-12);

            let ff_x = (-de_dangle * norm_bc) / f32::max(norm_cross1, 1.0e-12);
            let ff_y = (v0x * v1x + v0y * v1y + v0z * v1z) * dp;
            let ff_z = (v2x * v1x + v2y * v1y + v2z * v1z) * dp;
            let ff_w = (de_dangle * norm_bc) / f32::max(norm_cross2, 1.0e-12);

            // force1 = ff_x * cp0, force4 = ff_w * cp1
            let f1x = ff_x * cp0x;
            let f1y = ff_x * cp0y;
            let f1z = ff_x * cp0z;
            let f4x = ff_w * cp1x;
            let f4y = ff_w * cp1y;
            let f4z = ff_w * cp1z;

            // s = ff_y * force1 - ff_z * force4
            let sx = ff_y * f1x - ff_z * f4x;
            let sy = ff_y * f1y - ff_z * f4y;
            let sz = ff_y * f1z - ff_z * f4z;

            // force2 = s - force1, force3 = -s - force4
            if role == 0 {
                fx += f1x;
                fy += f1y;
                fz += f1z;
            } else if role == 1 {
                fx += sx - f1x;
                fy += sy - f1y;
                fz += sz - f1z;
            } else if role == 2 {
                fx += -sx - f4x;
                fy += -sy - f4y;
                fz += -sz - f4z;
            } else {
                fx += f4x;
                fy += f4y;
                fz += f4z;
            }
        }

        forces[i4] += fx;
        forces[i4 + 1] += fy;
        forces[i4 + 2] += fz;
    }
}

// ============================================================
// Host-side repacking
// ============================================================

use crate::group::Group;
use crate::topology::{BondKind, DihedralKind, Topology, TorsionKind};

/// Per-atom CSR (Compressed Sparse Row) layout for bonded interactions.
///
/// - `offsets[i]..offsets[i+1]`: entries for atom i
/// - `atoms`: neighbor/partner atom indices (stride depends on interaction type)
/// - `params`: interaction parameters (stride depends on interaction type)
pub struct CsrData {
    pub offsets: Vec<u32>,
    pub atoms: Vec<u32>,
    pub params: Vec<f32>,
}

impl CsrData {
    /// True if no interactions were packed.
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }
}

// Topology stores k in degree units; GPU kernels use radians, so k_eff = k * (180/π)²
const RAD_PER_DEG_SQ: f64 = (180.0 / std::f64::consts::PI) * (180.0 / std::f64::consts::PI);

/// Build CSR bond data from topology for the bond forces kernel.
///
/// Each bond creates two directed entries (one per endpoint).
/// Includes both intra-molecular and intermolecular bonds.
/// Only harmonic bonds are supported.
pub fn repack_bonds(topology: &Topology, groups: &[Group]) -> CsrData {
    let n_atoms: usize = groups.iter().map(|g| g.capacity()).sum();
    let mut edges: Vec<Vec<(u32, f32, f32)>> = vec![Vec::new(); n_atoms];

    for group in groups {
        let molecule = &topology.moleculekinds()[group.molecule()];
        for bond in molecule.bonds() {
            let BondKind::Harmonic(h) = bond.kind() else {
                continue;
            };
            let [i_loc, j_loc] = *bond.index();
            if i_loc >= group.len() || j_loc >= group.len() {
                continue;
            }
            let i_abs = (group.start() + i_loc) as u32;
            let j_abs = (group.start() + j_loc) as u32;
            let k = h.spring_constant() as f32;
            let req = h.eq_distance() as f32;
            // Two directed entries per bond so each endpoint's thread sees it
            edges[i_abs as usize].push((j_abs, k, req));
            edges[j_abs as usize].push((i_abs, k, req));
        }
    }

    // Intermolecular bonds use absolute atom indices
    for bond in topology.intermolecular().bonds() {
        let BondKind::Harmonic(h) = bond.kind() else {
            continue;
        };
        let [i_abs, j_abs] = *bond.index();
        if i_abs >= n_atoms || j_abs >= n_atoms {
            continue;
        }
        let k = h.spring_constant() as f32;
        let req = h.eq_distance() as f32;
        edges[i_abs].push((j_abs as u32, k, req));
        edges[j_abs].push((i_abs as u32, k, req));
    }

    let mut offsets = Vec::with_capacity(n_atoms + 1);
    let mut atoms = Vec::new();
    let mut params = Vec::new();
    offsets.push(0u32);
    for atom_edges in &edges {
        for &(j, k, req) in atom_edges {
            atoms.push(j);
            params.push(k);
            params.push(req);
        }
        offsets.push(atoms.len() as u32);
    }
    CsrData {
        offsets,
        atoms,
        params,
    }
}

/// Build CSR angle data from topology for the angle forces kernel.
///
/// Each angle creates three entries (one per participating atom).
/// Includes both intra-molecular and intermolecular torsions (angles).
/// Spring constant is converted to radian units.
pub fn repack_angles(topology: &Topology, groups: &[Group]) -> CsrData {
    let n_atoms: usize = groups.iter().map(|g| g.capacity()).sum();
    type AngleEntry = (u32, u32, f32, f32, f32); // (other1, other2, role, k_eff, theta_eq_rad)
    let mut entries: Vec<Vec<AngleEntry>> = vec![Vec::new(); n_atoms];

    for group in groups {
        let molecule = &topology.moleculekinds()[group.molecule()];
        for torsion in molecule.torsions() {
            let TorsionKind::Harmonic(h) = torsion.kind() else {
                continue;
            };
            let [a, b, c] = *torsion.index();
            if a >= group.len() || b >= group.len() || c >= group.len() {
                continue;
            }
            let aa = (group.start() + a) as u32;
            let ba = (group.start() + b) as u32;
            let ca = (group.start() + c) as u32;
            // Convert from degree-based k to radian-based for the kernel
            let k_eff = (h.spring_constant() * RAD_PER_DEG_SQ) as f32;
            let theta_eq_rad = h.eq_angle().to_radians() as f32;

            // Three entries per angle so each participating atom's thread sees it
            entries[aa as usize].push((ba, ca, 0.0, k_eff, theta_eq_rad));
            entries[ba as usize].push((aa, ca, 1.0, k_eff, theta_eq_rad));
            entries[ca as usize].push((aa, ba, 2.0, k_eff, theta_eq_rad));
        }
    }

    // Intermolecular torsions (angles) use absolute atom indices
    for torsion in topology.intermolecular().torsions() {
        let TorsionKind::Harmonic(h) = torsion.kind() else {
            continue;
        };
        let [a, b, c] = *torsion.index();
        if a >= n_atoms || b >= n_atoms || c >= n_atoms {
            continue;
        }
        let k_eff = (h.spring_constant() * RAD_PER_DEG_SQ) as f32;
        let theta_eq_rad = h.eq_angle().to_radians() as f32;

        entries[a].push((b as u32, c as u32, 0.0, k_eff, theta_eq_rad));
        entries[b].push((a as u32, c as u32, 1.0, k_eff, theta_eq_rad));
        entries[c].push((a as u32, b as u32, 2.0, k_eff, theta_eq_rad));
    }

    let mut offsets = Vec::with_capacity(n_atoms + 1);
    let mut atoms = Vec::new();
    let mut params = Vec::new();
    offsets.push(0u32);
    for atom_entries in &entries {
        for &(o1, o2, role, k_eff, theta_eq) in atom_entries {
            atoms.push(o1);
            atoms.push(o2);
            params.push(role);
            params.push(k_eff);
            params.push(theta_eq);
            params.push(0.0); // padding to stride 4
        }
        // Two atom indices per entry
        offsets.push((atoms.len() / 2) as u32);
    }
    CsrData {
        offsets,
        atoms,
        params,
    }
}

/// Build CSR dihedral data from topology for the dihedral forces kernel.
///
/// Each dihedral creates four entries (one per atom).
/// Includes both intra-molecular and intermolecular dihedrals.
/// Supports harmonic and periodic dihedral types.
pub fn repack_dihedrals(topology: &Topology, groups: &[Group]) -> CsrData {
    let n_atoms: usize = groups.iter().map(|g| g.capacity()).sum();
    type DihedralEntry = (u32, u32, u32, [f32; 8]); // (other1, other2, other3, params)
    let mut entries: Vec<Vec<DihedralEntry>> = vec![Vec::new(); n_atoms];

    /// Convert a dihedral to base_params, returning None for unsupported types.
    fn dihedral_base_params(kind: &DihedralKind) -> Option<[f32; 8]> {
        match kind {
            DihedralKind::ProperHarmonic(h) | DihedralKind::ImproperHarmonic(h) => {
                let k_eff = (h.spring_constant() * RAD_PER_DEG_SQ) as f32;
                let phi_eq = h.eq_angle().to_radians() as f32;
                Some([
                    0.0,
                    DIHEDRAL_TYPE_HARMONIC as f32,
                    k_eff,
                    phi_eq,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ])
            }
            DihedralKind::ProperPeriodic(p) | DihedralKind::ImproperPeriodic(p) => {
                let k = p.spring_constant() as f32;
                let phi0 = p.phase_angle().to_radians() as f32;
                let n = p.periodicity() as f32;
                Some([
                    0.0,
                    DIHEDRAL_TYPE_PERIODIC as f32,
                    k,
                    phi0,
                    n,
                    0.0,
                    0.0,
                    0.0,
                ])
            }
            DihedralKind::Unspecified => None,
        }
    }

    /// Add four CSR entries for a dihedral with absolute atom indices.
    fn push_dihedral_entries(
        entries: &mut [Vec<DihedralEntry>],
        abs: [u32; 4],
        base_params: [f32; 8],
    ) {
        for role in 0u32..4 {
            let others: [u32; 3] = match role {
                0 => [abs[1], abs[2], abs[3]],
                1 => [abs[0], abs[2], abs[3]],
                2 => [abs[0], abs[1], abs[3]],
                _ => [abs[0], abs[1], abs[2]],
            };
            let mut p = base_params;
            p[0] = role as f32;
            entries[abs[role as usize] as usize].push((others[0], others[1], others[2], p));
        }
    }

    for group in groups {
        let molecule = &topology.moleculekinds()[group.molecule()];
        for dihedral in molecule.dihedrals() {
            let [i0, i1, i2, i3] = *dihedral.index();
            if [i0, i1, i2, i3].iter().any(|&idx| idx >= group.len()) {
                continue;
            }
            let abs = [i0, i1, i2, i3].map(|i| (group.start() + i) as u32);
            if let Some(base_params) = dihedral_base_params(dihedral.kind()) {
                push_dihedral_entries(&mut entries, abs, base_params);
            }
        }
    }

    // Intermolecular dihedrals use absolute atom indices
    for dihedral in topology.intermolecular().dihedrals() {
        let [i0, i1, i2, i3] = *dihedral.index();
        if [i0, i1, i2, i3].iter().any(|&idx| idx >= n_atoms) {
            continue;
        }
        let abs = [i0 as u32, i1 as u32, i2 as u32, i3 as u32];
        if let Some(base_params) = dihedral_base_params(dihedral.kind()) {
            push_dihedral_entries(&mut entries, abs, base_params);
        }
    }

    let mut offsets = Vec::with_capacity(n_atoms + 1);
    let mut atoms = Vec::new();
    let mut params = Vec::new();
    offsets.push(0u32);
    for atom_entries in &entries {
        for &(o1, o2, o3, ref p) in atom_entries {
            atoms.push(o1);
            atoms.push(o2);
            atoms.push(o3);
            params.extend_from_slice(p);
        }
        // Three atom indices per entry
        offsets.push((atoms.len() / 3) as u32);
    }
    CsrData {
        offsets,
        atoms,
        params,
    }
}

// ============================================================
// Host-side force computation (reference / CPU fallback)
// ============================================================

/// Compute angle forces using the OpenMM Jacobian (f64, host-side reference).
///
/// Given atoms a-b-c with vertex at b (positions after PBC unwrapping),
/// returns `(force_a, force_b, force_c)` where each force is `[fx, fy, fz]`.
/// `de_dangle` = dU/dθ in radian units.
#[cfg(test)]
fn angle_forces_reference(
    pos_a: [f64; 3],
    pos_b: [f64; 3],
    pos_c: [f64; 3],
    de_dangle: f64,
) -> ([f64; 3], [f64; 3], [f64; 3]) {
    // v0 = vertex - end_a, v1 = vertex - end_c
    let v0 = sub3(pos_b, pos_a);
    let v1 = sub3(pos_b, pos_c);

    let cp = cross3(v0, v1);
    let rp = (cp[0] * cp[0] + cp[1] * cp[1] + cp[2] * cp[2])
        .sqrt()
        .max(1e-12);
    let r21 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
    let r23 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];

    let fa = scale3(cross3(v0, cp), de_dangle / (r21 * rp));
    let fc = scale3(cross3(cp, v1), de_dangle / (r23 * rp));
    let fb = [-(fa[0] + fc[0]), -(fa[1] + fc[1]), -(fa[2] + fc[2])];
    (fa, fb, fc)
}

/// Compute dihedral forces using the OpenMM torsion Jacobian (f64, host-side reference).
///
/// Given atoms 1-2-3-4, returns `(F1, F2, F3, F4)`.
/// `de_dangle` = dU/dθ in radian units.
#[cfg(test)]
fn dihedral_forces_reference(
    pos1: [f64; 3],
    pos2: [f64; 3],
    pos3: [f64; 3],
    pos4: [f64; 3],
    de_dangle: f64,
) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
    let v0 = sub3(pos1, pos2);
    let v1 = sub3(pos3, pos2);
    let v2 = sub3(pos3, pos4);

    let cp0 = cross3(v0, v1);
    let cp1 = cross3(v1, v2);

    let norm_cross1 = dot3(cp0, cp0);
    let norm_sqr_bc = dot3(v1, v1);
    let norm_bc = norm_sqr_bc.sqrt();
    let norm_cross2 = dot3(cp1, cp1);

    let dp = 1.0 / norm_sqr_bc.max(1e-30);

    let ff_x = (-de_dangle * norm_bc) / norm_cross1.max(1e-30);
    let ff_y = dot3(v0, v1) * dp;
    let ff_z = dot3(v2, v1) * dp;
    let ff_w = (de_dangle * norm_bc) / norm_cross2.max(1e-30);

    let f1 = scale3(cp0, ff_x);
    let f4 = scale3(cp1, ff_w);

    let s = [
        ff_y * f1[0] - ff_z * f4[0],
        ff_y * f1[1] - ff_z * f4[1],
        ff_y * f1[2] - ff_z * f4[2],
    ];

    let f2 = sub3(s, f1);
    let f3 = [-(s[0] + f4[0]), -(s[1] + f4[1]), -(s[2] + f4[2])];

    (f1, f2, f3, f4)
}

#[cfg(test)]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[cfg(test)]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    /// Signed dihedral angle from four positions via plane normals.
    fn signed_dihedral(p1: [f64; 3], p2: [f64; 3], p3: [f64; 3], p4: [f64; 3]) -> f64 {
        let v0 = sub3(p1, p2);
        let v1 = sub3(p3, p2);
        let v2 = sub3(p3, p4);
        let cp0 = cross3(v0, v1);
        let cp1 = cross3(v1, v2);
        let cos_angle = dot3(cp0, cp1) / (dot3(cp0, cp0).sqrt() * dot3(cp1, cp1).sqrt());
        let theta = cos_angle.clamp(-1.0, 1.0).acos();
        if dot3(v0, cp1) < 0.0 {
            -theta
        } else {
            theta
        }
    }

    /// C++ test: HarmonicBond(k=100, req=4), distance=(0,3,0)
    #[test]
    fn harmonic_bond_force() {
        let k = 100.0_f64;
        let req = 4.0;
        let dr = [0.0, 3.0, 0.0];
        let r: f64 = dr.iter().map(|x| x * x).sum::<f64>().sqrt();
        let scale = -k * (r - req) / r;
        let force_on_i = dr.map(|d| scale * d);
        assert_approx_eq!(f64, force_on_i[0], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, force_on_i[1], 100.0, epsilon = 1e-6);
        assert_approx_eq!(f64, force_on_i[2], 0.0, epsilon = 1e-6);
    }

    /// C++ test: HarmonicTorsion(k=1, aeq=45°), atoms at 90° angle
    #[test]
    fn harmonic_angle_force() {
        let pos_a = [0.0, 1.0, 0.0];
        let pos_b = [0.0, 0.0, 0.0];
        let pos_c = [1.0, 0.0, 0.0];

        let theta = std::f64::consts::FRAC_PI_2;
        let theta_eq = std::f64::consts::FRAC_PI_4;
        let de_dangle = theta - theta_eq; // k=1

        let (fa, fb, fc) = angle_forces_reference(pos_a, pos_b, pos_c, de_dangle);

        let pi4 = std::f64::consts::FRAC_PI_4;
        assert_approx_eq!(f64, fa[0], pi4, epsilon = 1e-6);
        assert_approx_eq!(f64, fa[1], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, fa[2], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, fb[0], -pi4, epsilon = 1e-6);
        assert_approx_eq!(f64, fb[1], -pi4, epsilon = 1e-6);
        assert_approx_eq!(f64, fb[2], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, fc[0], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, fc[1], pi4, epsilon = 1e-6);
        assert_approx_eq!(f64, fc[2], 0.0, epsilon = 1e-6);
    }

    /// C++ test: PeriodicDihedral(k=100, phi=0°, n=3), atoms at 90° dihedral
    #[test]
    fn periodic_dihedral_force() {
        let pos1 = [5.0, 0.0, 0.0];
        let pos2 = [0.0, 0.0, 0.0];
        let pos3 = [0.0, 0.0, 2.0];
        let pos4 = [0.0, 10.0, 2.0];

        let theta = signed_dihedral(pos1, pos2, pos3, pos4);
        // U = k*(1 + cos(n*θ - φ₀)), dU/dθ = -k*n*sin(n*θ - φ₀)
        let de_dangle = -100.0 * 3.0 * (3.0 * theta).sin();

        let (f1, f2, f3, f4) = dihedral_forces_reference(pos1, pos2, pos3, pos4, de_dangle);

        assert_approx_eq!(f64, f1[0], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, f1[1], 60.0, epsilon = 1e-6);
        assert_approx_eq!(f64, f1[2], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, f2[0], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, f2[1], -60.0, epsilon = 1e-6);
        assert_approx_eq!(f64, f2[2], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, f3[0], -30.0, epsilon = 1e-6);
        assert_approx_eq!(f64, f3[1], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, f3[2], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, f4[0], 30.0, epsilon = 1e-6);
        assert_approx_eq!(f64, f4[1], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, f4[2], 0.0, epsilon = 1e-6);
    }

    /// C++ test: HarmonicDihedral(k=100, deq=90°), atoms at 120° dihedral
    /// p_45deg[3]={10,10,2}, p_60deg: y*=√3, p_120deg: x*=-1
    #[test]
    fn harmonic_dihedral_force() {
        let pos1 = [5.0, 0.0, 0.0];
        let pos2 = [0.0, 0.0, 0.0];
        let pos3 = [0.0, 0.0, 2.0];
        let pos4 = [-10.0, 10.0 * 3.0_f64.sqrt(), 2.0];

        let theta = signed_dihedral(pos1, pos2, pos3, pos4);
        // U = 0.5*k*(θ - θ_eq)², radian-based k
        let de_dangle = 100.0 * (theta - std::f64::consts::FRAC_PI_2);

        let (f1, f2, f3, f4) = dihedral_forces_reference(pos1, pos2, pos3, pos4, de_dangle);

        assert_approx_eq!(f64, f1[0], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, f1[1], 10.471975512, epsilon = 1e-6);
        assert_approx_eq!(f64, f1[2], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, f2[0], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, f2[1], -10.471975512, epsilon = 1e-6);
        assert_approx_eq!(f64, f2[2], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, f3[0], -2.2672492053, epsilon = 1e-6);
        assert_approx_eq!(f64, f3[1], -1.308996939, epsilon = 1e-6);
        assert_approx_eq!(f64, f3[2], 0.0, epsilon = 1e-6);
        assert_approx_eq!(f64, f4[0], 2.2672492053, epsilon = 1e-6);
        assert_approx_eq!(f64, f4[1], 1.308996939, epsilon = 1e-6);
        assert_approx_eq!(f64, f4[2], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn angle_force_sum_zero() {
        let (fa, fb, fc) =
            angle_forces_reference([1.5, 2.3, -0.7], [0.0, 0.0, 0.0], [-1.2, 0.8, 1.4], 3.14);
        for k in 0..3 {
            assert_approx_eq!(f64, fa[k] + fb[k] + fc[k], 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn dihedral_force_sum_zero() {
        let (f1, f2, f3, f4) = dihedral_forces_reference(
            [2.0, 1.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [-1.0, 2.5, 3.0],
            42.0,
        );
        for k in 0..3 {
            assert_approx_eq!(f64, f1[k] + f2[k] + f3[k] + f4[k], 0.0, epsilon = 1e-6);
        }
    }
}

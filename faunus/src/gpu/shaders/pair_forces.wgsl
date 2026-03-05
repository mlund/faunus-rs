// Per-atom pairwise nonbonded force computation using splined potentials.
//
// Each thread handles one atom i, loops over all atoms j, and accumulates
// the force on atom i. Uses branchless PBC and PowerLaw2 spline lookup.

struct SplineParams {
    r_min: f32,
    r_max: f32,
    n_coeffs: u32,
    coeff_offset: u32,
    f_at_rmin: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// Interleaved energy + force coefficients per interval
struct SplineCoeffs {
    u: vec4<f32>,  // energy [u0, u1, u2, u3]
    f: vec4<f32>,  // force  [f0, f1, f2, f3]
}

struct ForceUniforms {
    n_atoms: u32,
    n_atom_types: u32,
    box_length: f32,
    cutoff_sq: f32,
}

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> forces: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> atom_type_ids: array<u32>;
@group(0) @binding(3) var<storage, read> spline_params: array<SplineParams>;
@group(0) @binding(4) var<storage, read> spline_coeffs: array<SplineCoeffs>;
@group(0) @binding(5) var<storage, read> atom_mol_ids: array<u32>;

var<push_constant> pc: ForceUniforms;

// Evaluate spline force magnitude: -dU/dr (scalar, positive = repulsive)
fn spline_force(type_i: u32, type_j: u32, r: f32) -> f32 {
    let pair_idx = type_i * pc.n_atom_types + type_j;
    let params = spline_params[pair_idx];

    if (r >= params.r_max) {
        return 0.0;
    }

    // Clamp to r_min: below r_min returns f_at_rmin (matching CPU extrapolation)
    let r_clamped = max(r, params.r_min);

    // PowerLaw2 grid mapping
    let t = (r_clamped - params.r_min) / (params.r_max - params.r_min);
    let x = sqrt(t);

    let n_intervals = params.n_coeffs - 1u;
    let idx_f = x * f32(n_intervals);
    let idx = min(u32(idx_f), n_intervals - 1u);
    let frac = idx_f - f32(idx);

    let c = spline_coeffs[params.coeff_offset + idx].f;

    // Horner: f0 + frac*(f1 + frac*(f2 + frac*f3))
    return c.x + frac * (c.y + frac * (c.z + frac * c.w));
}

@compute @workgroup_size(64)
fn compute_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.n_atoms) {
        return;
    }

    let pi = positions[i].xyz;
    let type_i = atom_type_ids[i];
    let mol_i = atom_mol_ids[i];
    let box_len = pc.box_length;
    let inv_box = 1.0 / box_len;

    var force = vec3<f32>(0.0, 0.0, 0.0);

    for (var j = 0u; j < pc.n_atoms; j++) {
        // Skip intramolecular pairs (handles self-exclusion too)
        if (atom_mol_ids[j] == mol_i) {
            continue;
        }

        var dr = positions[j].xyz - pi;
        // Branchless PBC: dr -= L * round(dr / L)
        dr -= box_len * round(dr * inv_box);

        let r_sq = dot(dr, dr);
        if (r_sq < pc.cutoff_sq && r_sq > 1e-8) {
            let r = sqrt(r_sq);
            let type_j = atom_type_ids[j];
            let f_mag = spline_force(type_i, type_j, r);
            // f_mag = -dU/dr (positive = repulsive); dr points i→j, so negate
            force -= (f_mag / r) * dr;
        }
    }

    forces[i] = vec4<f32>(force, 0.0);
}

// Reconstruct atom positions from COM + quaternion.
//
// Each thread handles one atom:
//   r[i] = r_com[m] + rotate(q[m], r_ref[i])
// where r_ref[i] is the atom's position in the molecule's body frame.

struct ReconstructUniforms {
    n_atoms: u32,
    n_molecules: u32,
}

@group(0) @binding(0) var<storage, read> com_positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> quaternions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> ref_positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> mol_atom_offsets: array<u32>;

var<push_constant> pc: ReconstructUniforms;

// Rotate vector v by unit quaternion q: q * (0,v) * q*
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let qw = q.w;
    let t = 2.0 * cross(qv, v);
    return v + qw * t + cross(qv, t);
}

// Binary search to find which molecule owns atom `atom_idx`
fn find_molecule(atom_idx: u32) -> u32 {
    var lo = 0u;
    var hi = pc.n_molecules;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (mol_atom_offsets[mid + 1u] <= atom_idx) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

@compute @workgroup_size(64)
fn reconstruct_positions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.n_atoms) {
        return;
    }

    let m = find_molecule(i);
    let r_com = com_positions[m].xyz;
    let q = quaternions[m];
    let r_ref = ref_positions[i].xyz;

    let r_rotated = quat_rotate(q, r_ref);
    let atom_type = bitcast<u32>(positions[i].w); // preserve atom type in w

    positions[i] = vec4<f32>(r_com + r_rotated, bitcast<f32>(atom_type));
}

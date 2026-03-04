// Reduce per-atom forces to per-molecule COM forces and torques.
//
// Each thread handles one molecule. For rigid bodies:
//   F_com[m] = sum(f[i]) for i in molecule m
//   tau[m]   = sum((r[i] - r_com[m]) × f[i]) for i in molecule m

struct ReduceUniforms {
    n_molecules: u32,
    box_length: f32,
}

@group(0) @binding(0) var<storage, read> forces: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> com_positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> com_forces: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> torques: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> mol_atom_offsets: array<u32>;

var<push_constant> pc: ReduceUniforms;

@compute @workgroup_size(64)
fn reduce_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    if (m >= pc.n_molecules) {
        return;
    }

    let start = mol_atom_offsets[m];
    let end = mol_atom_offsets[m + 1u];
    let r_com = com_positions[m].xyz;

    var f_total = vec3<f32>(0.0);
    var tau_total = vec3<f32>(0.0);

    let box_len = pc.box_length;
    let inv_box = 1.0 / box_len;

    for (var i = start; i < end; i++) {
        let fi = forces[i].xyz;
        // Minimum-image lever arm: handles molecules split across PBC boundaries
        var dr = positions[i].xyz - r_com;
        dr -= box_len * round(dr * inv_box);
        f_total += fi;
        tau_total += cross(dr, fi);
    }

    com_forces[m] = vec4<f32>(f_total, 0.0);
    torques[m] = vec4<f32>(tau_total, 0.0);
}

// BAOAB (Langevin Middle) integrator for rigid bodies.
//
// Two entry points:
//   baoab_step: B-A-O-A (opening half-kick, drift, thermostat, drift)
//   half_kick:  B only (closing half-kick after force recomputation)
//
// The full BAOAB step across the run loop is:
//   baoab_step(F_old) → reconstruct → compute F_new → half_kick(F_new)
//
// Uses Philox 4x32 counter-based RNG for deterministic Gaussian noise.

struct LangevinUniforms {
    n_molecules: u32,
    dt: f32,
    friction: f32,
    kT: f32,
    rng_seed: u32,
    rng_step: u32,
    box_length: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> com_positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> com_velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> quaternions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> angular_velocities: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> com_forces: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> torques: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> mol_masses: array<f32>;
@group(0) @binding(7) var<storage, read> mol_inertia: array<vec4<f32>>;

var<push_constant> pc: LangevinUniforms;

// ============================================================================
// Philox 4x32-10 counter-based RNG
// ============================================================================

// Full 32x32 → 64-bit multiply using 16-bit limbs (WGSL lacks u64)
fn mulhilo(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let lo_lo = a_lo * b_lo;
    let hi_lo = a_hi * b_lo;
    let lo_hi = a_lo * b_hi;
    let hi_hi = a_hi * b_hi;

    let mid = (lo_lo >> 16u) + (hi_lo & 0xFFFFu) + (lo_hi & 0xFFFFu);
    let lo = (lo_lo & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
    let hi = hi_hi + (hi_lo >> 16u) + (lo_hi >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

fn philox_round(ctr: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    let m0 = mulhilo(0xD2511F53u, ctr.x);
    let m1 = mulhilo(0xCD9E8D57u, ctr.z);
    return vec4<u32>(
        m1.y ^ ctr.y ^ key.x,
        m1.x,
        m0.y ^ ctr.w ^ key.y,
        m0.x
    );
}

fn philox4x32(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    var ctr = counter;
    var k = key;
    for (var i = 0u; i < 10u; i++) {
        ctr = philox_round(ctr, k);
        k.x += 0x9E3779B9u;
        k.y += 0xBB67AE85u;
    }
    return ctr;
}

// Box-Muller: two uniform u32 -> two Gaussian f32
fn box_muller(u0: u32, u1: u32) -> vec2<f32> {
    let uf0 = f32(u0) * 2.3283064e-10; // 1 / 2^32
    let uf1 = f32(u1) * 2.3283064e-10;
    let r = sqrt(-2.0 * log(max(uf0, 1e-30)));
    let theta = 6.28318530718 * uf1;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

// Generate 4 Gaussian random numbers for molecule `mol_id` at step `step`.
// `stream` separates translational (0) from rotational (1) noise.
fn gaussian4(mol_id: u32, step: u32, seed: u32, stream: u32) -> vec4<f32> {
    let ctr = vec4<u32>(mol_id, step, stream, 0u);
    let key = vec2<u32>(seed, 0x12345678u);
    let rng = philox4x32(ctr, key);
    let g01 = box_muller(rng.x, rng.y);
    let g23 = box_muller(rng.z, rng.w);
    return vec4<f32>(g01.x, g01.y, g23.x, g23.y);
}

// 1 kJ/mol = 100 amu·Å²/ps²
const KJ_MOL_TO_INTERNAL: f32 = 100.0;

// Invert each component, mapping zero inertia to zero (skip rotation for point particles)
fn safe_inv_inertia(inertia: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        select(1.0 / inertia.x, 0.0, inertia.x == 0.0),
        select(1.0 / inertia.y, 0.0, inertia.y == 0.0),
        select(1.0 / inertia.z, 0.0, inertia.z == 0.0)
    );
}

// Rotate vector v by quaternion q: q ⊗ (v,0) ⊗ q*
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Rotate vector v by inverse quaternion q*: q* ⊗ (v,0) ⊗ q
fn quat_rotate_inv(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qc = vec4<f32>(-q.xyz, q.w);
    return quat_rotate(qc, v);
}

// Quaternion multiplication: q * p
fn quat_mul(q: vec4<f32>, p: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        q.w * p.x + q.x * p.w + q.y * p.z - q.z * p.y,
        q.w * p.y - q.x * p.z + q.y * p.w + q.z * p.x,
        q.w * p.z + q.x * p.y - q.y * p.x + q.z * p.w,
        q.w * p.w - q.x * p.x - q.y * p.y - q.z * p.z
    );
}

@compute @workgroup_size(64)
fn baoab_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    if (m >= pc.n_molecules) {
        return;
    }

    let dt = pc.dt;
    let half_dt = 0.5 * dt;
    let gamma = pc.friction;
    let mass = mol_masses[m];
    let inv_mass = 1.0 / mass;
    let inv_inertia = safe_inv_inertia(mol_inertia[m].xyz);

    var r = com_positions[m].xyz;
    var v = com_velocities[m].xyz;
    var q = quaternions[m];
    var omega = angular_velocities[m].xyz;
    let f = com_forces[m].xyz;
    let tau = torques[m].xyz;

    // B step: half-kick velocities
    v += half_dt * inv_mass * f * KJ_MOL_TO_INTERNAL;
    let tau_body = quat_rotate_inv(q, tau);
    omega += half_dt * inv_inertia * tau_body * KJ_MOL_TO_INTERNAL;

    // A step: half-drift
    r += half_dt * v;
    // Quaternion drift using body-frame omega: dq/dt = 0.5 * q ⊗ (0, omega_body)
    let omega_quat = vec4<f32>(omega * half_dt * 0.5, 0.0);
    q = q + quat_mul(q, omega_quat);
    q = normalize(q);

    // O step: Ornstein-Uhlenbeck thermostat (body frame)
    let a = exp(-gamma * dt);
    let b = sqrt(1.0 - a * a);

    let noise_trans = gaussian4(m, pc.rng_step, pc.rng_seed, 0u);
    let noise_rot = gaussian4(m, pc.rng_step, pc.rng_seed, 1u);

    let sigma_trans = sqrt(pc.kT * inv_mass * KJ_MOL_TO_INTERNAL);
    v = a * v + b * sigma_trans * noise_trans.xyz;

    let sigma_rot = sqrt(pc.kT * inv_inertia * KJ_MOL_TO_INTERNAL);
    omega = a * omega + b * sigma_rot * noise_rot.xyz;

    // A step: second half-drift
    r += half_dt * v;
    let omega_quat2 = vec4<f32>(omega * half_dt * 0.5, 0.0);
    q = q + quat_mul(q, omega_quat2);
    q = normalize(q);

    // PBC wrap: keep COM inside the primary cell [-L/2, L/2]
    let box_len = pc.box_length;
    let inv_box = 1.0 / box_len;
    r -= box_len * round(r * inv_box);

    com_positions[m] = vec4<f32>(r, 0.0);
    com_velocities[m] = vec4<f32>(v, 0.0);
    quaternions[m] = q;
    angular_velocities[m] = vec4<f32>(omega, 0.0);
}

// Closing B half-kick using forces at the new positions.
@compute @workgroup_size(64)
fn half_kick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    if (m >= pc.n_molecules) {
        return;
    }

    let half_dt = 0.5 * pc.dt;
    let inv_mass = 1.0 / mol_masses[m];
    let inv_inertia = safe_inv_inertia(mol_inertia[m].xyz);

    var v = com_velocities[m].xyz;
    var omega = angular_velocities[m].xyz;

    v += half_dt * inv_mass * com_forces[m].xyz * KJ_MOL_TO_INTERNAL;
    let q = quaternions[m];
    let tau_body = quat_rotate_inv(q, torques[m].xyz);
    omega += half_dt * inv_inertia * tau_body * KJ_MOL_TO_INTERNAL;

    com_velocities[m] = vec4<f32>(v, 0.0);
    angular_velocities[m] = vec4<f32>(omega, 0.0);
}

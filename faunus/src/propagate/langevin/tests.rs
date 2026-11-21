use super::*;
use crate::context::WithHamiltonian as _;
use crate::energy::ld_case::{ld_system, LdCase};
use cubecl::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

/// Seeded generator so the statistical assertions below cannot flake.
fn test_rng() -> StdRng {
    StdRng::seed_from_u64(0xFA11_5EED)
}

// ============================================================================
// Quaternion roundtrip tests
// ============================================================================

#[test]
fn quaternion_roundtrip_f32_f64() {
    let axis = nalgebra::UnitVector3::new_normalize(crate::Point::new(1.0, 2.0, 3.0));
    let q = crate::UnitQuaternion::from_axis_angle(&axis, 1.23);
    let reconstructed = gpu_to_quat(&quat_to_gpu(&q));
    assert!(
        q.angle_to(&reconstructed) < 1e-6,
        "Roundtrip angle error: {}",
        q.angle_to(&reconstructed)
    );
}

#[test]
fn quaternion_identity_roundtrip() {
    let q = crate::UnitQuaternion::identity();
    let gpu = quat_to_gpu(&q);
    assert_eq!(gpu, [0.0, 0.0, 0.0, 1.0]);
    let back = gpu_to_quat(&gpu);
    assert!(q.angle_to(&back) < 1e-10);
}

/// q and -q represent the same rotation — roundtrip must preserve orientation.
#[test]
fn quaternion_sign_invariance() {
    let axis = nalgebra::UnitVector3::new_normalize(crate::Point::new(0.0, 0.0, 1.0));
    let q = crate::UnitQuaternion::from_axis_angle(&axis, std::f64::consts::FRAC_PI_2);

    let gpu_pos = quat_to_gpu(&q);
    let gpu_neg = [-gpu_pos[0], -gpu_pos[1], -gpu_pos[2], -gpu_pos[3]];

    let q_pos = gpu_to_quat(&gpu_pos);
    let q_neg = gpu_to_quat(&gpu_neg);
    assert!(
        q_pos.angle_to(&q_neg) < 1e-6,
        "q and -q should represent the same rotation"
    );
}

// ============================================================================
// Temperature computation tests
// ============================================================================

/// Helper: compute expected T from ½mv² = ½·dof·kB·T.
fn expected_temperature(kinetic_energy: f32, dof: f32) -> f32 {
    const R_KJ_PER_MOL_K: f32 = crate::R_IN_KJ_PER_MOL as f32;
    const KJ_MOL_TO_INTERNAL: f32 = 100.0;
    2.0 * kinetic_energy / (KJ_MOL_TO_INTERNAL * dof * R_KJ_PER_MOL_K)
}

#[test]
fn temperature_zero_velocity() {
    let (t_trans, t_rot) = compute_temperature(
        &[1],                  // one rigid molecule
        &[10.0],               // mass
        &[1.0, 1.0, 1.0, 0.0], // inertia
        &[[0.0, 0.0, 0.0, 0.0]],
        &[[0.0, 0.0, 0.0, 0.0]],
        &[],
        &[],
        &[],
    );
    assert_eq!(t_trans, 0.0);
    assert_eq!(t_rot, 0.0);
}

#[test]
fn temperature_no_molecules() {
    let (t_trans, t_rot) = compute_temperature(&[], &[], &[], &[], &[], &[], &[], &[]);
    assert_eq!(t_trans, 0.0);
    assert_eq!(t_rot, 0.0);
}

#[test]
fn temperature_single_rigid_translational() {
    let mass = 18.0f32; // water-like
    let vx = 2.0f32;
    let ke = 0.5 * mass * vx * vx;

    let (t_trans, _) = compute_temperature(
        &[1],
        &[mass],
        &[1.0, 1.0, 1.0, 0.0],
        &[[vx, 0.0, 0.0, 0.0]],
        &[[0.0, 0.0, 0.0, 0.0]],
        &[],
        &[],
        &[],
    );
    let expected = expected_temperature(ke, 3.0);
    assert!(
        (t_trans - expected).abs() < 1e-4,
        "T_trans={t_trans}, expected={expected}"
    );
}

#[test]
fn temperature_single_rigid_rotational() {
    let inertia = [5.0f32, 10.0, 15.0, 0.0];
    let omega = [1.0f32, 2.0, 3.0, 0.0];
    let ke_rot = 0.5
        * (inertia[0] * omega[0].powi(2)
            + inertia[1] * omega[1].powi(2)
            + inertia[2] * omega[2].powi(2));

    let (_, t_rot) = compute_temperature(
        &[1],
        &[18.0],
        &inertia,
        &[[0.0, 0.0, 0.0, 0.0]],
        &[omega],
        &[],
        &[],
        &[],
    );
    let expected = expected_temperature(ke_rot, 3.0);
    assert!(
        (t_rot - expected).abs() < 1e-4,
        "T_rot={t_rot}, expected={expected}"
    );
}

#[test]
fn temperature_multiple_rigid_molecules() {
    let masses = [18.0f32, 44.0];
    let inertia = [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0];
    let vel = [[1.0f32, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]];
    let omega = [[0.0f32; 4]; 2];

    let ke = 0.5 * masses[0] * 1.0 + 0.5 * masses[1] * 4.0;
    let expected = expected_temperature(ke, 6.0); // 2 molecules × 3 DOF

    let (t_trans, _) = compute_temperature(&[1, 1], &masses, &inertia, &vel, &omega, &[], &[], &[]);
    assert!(
        (t_trans - expected).abs() < 1e-3,
        "T_trans={t_trans}, expected={expected}"
    );
}

/// Non-rigid molecules should not contribute to the temperature.
#[test]
fn temperature_skips_nonrigid() {
    let (t_trans, t_rot) = compute_temperature(
        &[0], // NOT rigid
        &[18.0],
        &[1.0, 1.0, 1.0, 0.0],
        &[[10.0, 10.0, 10.0, 0.0]], // large velocity, but should be ignored
        &[[5.0, 5.0, 5.0, 0.0]],
        &[],
        &[],
        &[],
    );
    assert_eq!(t_trans, 0.0);
    assert_eq!(t_rot, 0.0);
}

#[test]
fn temperature_flexible_atoms_only() {
    let mass = 12.0f32;
    let vx = 3.0f32;
    let ke = 0.5 * mass * vx * vx;

    let (t_trans, t_rot) = compute_temperature(
        &[],
        &[],
        &[],
        &[],
        &[],
        &[1], // one flexible atom
        &[mass],
        &[vx, 0.0, 0.0], // stride-3 velocity
    );
    let expected = expected_temperature(ke, 3.0);
    assert!(
        (t_trans - expected).abs() < 1e-4,
        "T_trans={t_trans}, expected={expected}"
    );
    assert_eq!(t_rot, 0.0, "flexible atoms have no rotational DOF");
}

/// Mixed system: one rigid molecule + two flexible atoms. Both contribute to T_trans.
#[test]
fn temperature_mixed_rigid_and_flexible() {
    let rigid_mass = 18.0f32;
    let rigid_vel = [1.0f32, 0.0, 0.0, 0.0];
    let ke_rigid = 0.5 * rigid_mass * 1.0;

    let flex_mass = 12.0f32;
    let flex_vel = [0.0f32, 2.0, 0.0]; // stride-3
    let ke_flex = 0.5 * flex_mass * 4.0;

    let total_ke = ke_rigid + ke_flex;
    let total_dof = 3.0 + 3.0; // one rigid COM + one flex atom

    let (t_trans, _) = compute_temperature(
        &[1],
        &[rigid_mass],
        &[1.0, 1.0, 1.0, 0.0],
        &[rigid_vel],
        &[[0.0; 4]],
        &[1],
        &[flex_mass],
        &flex_vel,
    );
    let expected = expected_temperature(total_ke, total_dof);
    assert!(
        (t_trans - expected).abs() < 1e-3,
        "T_trans={t_trans}, expected={expected}"
    );
}

/// Frozen flexible atoms (flag=0) should not contribute.
#[test]
fn temperature_frozen_flexible_atoms_excluded() {
    let (t_trans, _) = compute_temperature(
        &[],
        &[],
        &[],
        &[],
        &[],
        &[0, 0], // both frozen
        &[12.0, 14.0],
        &[5.0, 5.0, 5.0, 3.0, 3.0, 3.0],
    );
    assert_eq!(t_trans, 0.0);
}

/// T_trans and T_rot should be consistent with equipartition for isotropic velocities.
#[test]
fn temperature_equipartition_isotropic() {
    let mass = 18.0f32;
    let v = 1.5f32;
    let inertia = [2.0f32, 2.0, 2.0, 0.0]; // spherical top
    let omega = 1.5f32;

    let ke_trans = 0.5 * mass * 3.0 * v * v;
    let ke_rot = 0.5 * 2.0 * 3.0 * omega * omega;

    let (t_trans, t_rot) = compute_temperature(
        &[1],
        &[mass],
        &inertia,
        &[[v, v, v, 0.0]],
        &[[omega, omega, omega, 0.0]],
        &[],
        &[],
        &[],
    );
    let expected_trans = expected_temperature(ke_trans, 3.0);
    let expected_rot = expected_temperature(ke_rot, 3.0);

    assert!(
        (t_trans - expected_trans).abs() < 1e-3,
        "T_trans={t_trans}, expected={expected_trans}"
    );
    assert!(
        (t_rot - expected_rot).abs() < 1e-3,
        "T_rot={t_rot}, expected={expected_rot}"
    );
    // For isotropic velocities with same sigma, T_trans ≈ T_rot only if m·v² ≈ I·ω²
    // Here they differ because mass ≠ inertia, which is expected.
}

// ============================================================================
// LangevinConfig serialization
// ============================================================================

#[test]
fn langevin_config_deserialize() {
    let yaml = r#"
        timestep: 0.002
        friction: 10.0
        steps: 1000
        temperature: 300.0
    "#;
    let config: LangevinConfig = yaml_serde::from_str(yaml).unwrap();
    assert_eq!(config.timestep, 0.002);
    assert_eq!(config.friction, 10.0);
    assert_eq!(config.steps, 1000);
    assert_eq!(config.temperature, 300.0);
    assert_eq!(config.cell_list_rebuild, 20, "default should be 20");
}

#[test]
fn langevin_config_with_cell_list_rebuild() {
    let yaml = r#"
        timestep: 0.001
        friction: 5.0
        steps: 500
        temperature: 298.0
        cell_list_rebuild: 50
    "#;
    let config: LangevinConfig = yaml_serde::from_str(yaml).unwrap();
    assert_eq!(config.cell_list_rebuild, 50);
}

#[test]
fn langevin_config_rejects_unknown_fields() {
    let yaml = r#"
        timestep: 0.002
        friction: 10.0
        steps: 1000
        temperature: 300.0
        bogus_field: 42
    "#;
    let result: Result<LangevinConfig, _> = yaml_serde::from_str(yaml);
    assert!(result.is_err(), "unknown fields should be rejected");
}

// ============================================================================
// LangevinRunner YAML output
// ============================================================================

#[test]
fn langevin_runner_to_yaml_without_temperature() {
    let config = LangevinConfig {
        timestep: 0.002,
        friction: 10.0,
        steps: 500,
        temperature: 300.0,
        cell_list_rebuild: 20,
    };
    let runner = LangevinRunner::new(config);
    let yaml = runner.to_yaml();

    // Should be a tagged mapping with expected fields
    if let yaml_serde::Value::Tagged(tagged) = &yaml {
        assert_eq!(tagged.tag.to_string(), "!LangevinDynamics");
        if let yaml_serde::Value::Mapping(map) = &tagged.value {
            assert!(map.contains_key(yaml_serde::Value::from("timestep")));
            assert!(map.contains_key(yaml_serde::Value::from("friction")));
            assert!(map.contains_key(yaml_serde::Value::from("steps")));
            assert!(map.contains_key(yaml_serde::Value::from("temperature")));
            assert!(!map.contains_key(yaml_serde::Value::from("measured_temperature")));
        } else {
            panic!("expected mapping");
        }
    } else {
        panic!("expected tagged value");
    }
}

#[test]
fn langevin_runner_to_yaml_with_temperature() {
    let config = LangevinConfig {
        timestep: 0.002,
        friction: 10.0,
        steps: 500,
        temperature: 300.0,
        cell_list_rebuild: 20,
    };
    let mut runner = LangevinRunner::new(config);
    runner.t_trans.add(295.0);
    runner.t_trans.add(305.0);
    runner.t_rot.add(290.0);
    runner.t_rot.add(310.0);

    let yaml = runner.to_yaml();
    if let yaml_serde::Value::Tagged(tagged) = &yaml {
        if let yaml_serde::Value::Mapping(map) = &tagged.value {
            assert!(
                map.contains_key(yaml_serde::Value::from("measured_temperature")),
                "should include measured_temperature after adding samples"
            );
        } else {
            panic!("expected mapping");
        }
    } else {
        panic!("expected tagged value");
    }
}

// ============================================================================
// Maxwell-Boltzmann velocity generation
// ============================================================================

/// Helper: expected MB velocity variance = kT * conv / mass (in (Å/ps)²).
fn mb_variance(kt: f64, mass: f64) -> f64 {
    let conv = 100.0; // kJ/mol → amu·Å²/ps²
    kt * conv / mass
}

/// Rigid COM velocities should have zero mean and correct MB variance.
#[test]
fn mb_velocities_rigid_com_statistics() {
    let n_mol = 5000;
    let mass = 18.0f32;
    let kt = 2.494; // ~300 K in kJ/mol

    let mol_is_rigid = vec![1u32; n_mol];
    let mol_masses = vec![mass; n_mol];
    let mol_inertia = vec![0.0f32; n_mol * 4]; // not used for COM
    let (com_vel, _, _) = generate_mb_velocities(
        kt,
        &mol_is_rigid,
        &mol_masses,
        &mol_inertia,
        &[],
        &[],
        &mut test_rng(),
    );

    // Extract vx components (stride 4, offset 0)
    let vx: Vec<f64> = com_vel.iter().step_by(4).map(|&v| v as f64).collect();
    assert_eq!(vx.len(), n_mol);

    let mean: f64 = vx.iter().sum::<f64>() / vx.len() as f64;
    let var: f64 = vx.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vx.len() as f64;
    let expected_var = mb_variance(kt, mass as f64);

    // Statistical tolerance: 5σ for mean, 20% for variance
    let sigma_mean = (expected_var / n_mol as f64).sqrt();
    assert!(
        mean.abs() < 5.0 * sigma_mean,
        "mean={mean:.4}, 5σ={:.4}",
        5.0 * sigma_mean
    );
    assert!(
        (var - expected_var).abs() / expected_var < 0.2,
        "var={var:.4}, expected={expected_var:.4}"
    );
}

/// Non-rigid molecules should get zero COM velocities.
#[test]
fn mb_velocities_nonrigid_are_zero() {
    let (com_vel, ang_vel, _) = generate_mb_velocities(
        2.494,
        &[0, 0],
        &[18.0, 44.0],
        &[1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0],
        &[],
        &[],
        &mut test_rng(),
    );
    assert!(
        com_vel.iter().all(|&v| v == 0.0),
        "non-rigid COM velocities should be zero"
    );
    assert!(
        ang_vel.iter().all(|&v| v == 0.0),
        "non-rigid angular velocities should be zero"
    );
}

/// Flexible atom velocities should have correct MB statistics.
#[test]
fn mb_velocities_flexible_atom_statistics() {
    let n_atoms = 5000;
    let mass = 12.0f32;
    let kt = 2.494;

    let atom_is_flexible = vec![1u32; n_atoms];
    let atom_masses = vec![mass; n_atoms];
    let (_, _, atom_vel) = generate_mb_velocities(
        kt,
        &[],
        &[],
        &[],
        &atom_is_flexible,
        &atom_masses,
        &mut test_rng(),
    );

    // Extract vx components (stride 3, offset 0)
    let vx: Vec<f64> = atom_vel.iter().step_by(3).map(|&v| v as f64).collect();
    assert_eq!(vx.len(), n_atoms);

    let mean: f64 = vx.iter().sum::<f64>() / vx.len() as f64;
    let var: f64 = vx.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vx.len() as f64;
    let expected_var = mb_variance(kt, mass as f64);

    let sigma_mean = (expected_var / n_atoms as f64).sqrt();
    assert!(
        mean.abs() < 5.0 * sigma_mean,
        "mean={mean:.4}, 5σ={:.4}",
        5.0 * sigma_mean
    );
    assert!(
        (var - expected_var).abs() / expected_var < 0.2,
        "var={var:.4}, expected={expected_var:.4}"
    );
}

/// Frozen flexible atoms (flag=0) should get zero velocities.
#[test]
fn mb_velocities_frozen_atoms_are_zero() {
    let (_, _, atom_vel) = generate_mb_velocities(
        2.494,
        &[],
        &[],
        &[],
        &[0, 0, 0],
        &[12.0, 14.0, 16.0],
        &mut test_rng(),
    );
    assert!(
        atom_vel.iter().all(|&v| v == 0.0),
        "frozen atom velocities should be zero"
    );
}

/// Angular velocities should scale with 1/sqrt(inertia).
#[test]
fn mb_velocities_angular_variance_scales_with_inertia() {
    let n_mol = 5000;
    let kt = 2.494;
    let inertia_x = 5.0f32;
    let inertia_y = 50.0f32; // 10× larger → 10× smaller variance

    let mol_is_rigid = vec![1u32; n_mol];
    let mol_masses = vec![18.0f32; n_mol];
    let mol_inertia: Vec<f32> = (0..n_mol)
        .flat_map(|_| [inertia_x, inertia_y, 1.0, 0.0])
        .collect();

    let (_, ang_vel, _) = generate_mb_velocities(
        kt,
        &mol_is_rigid,
        &mol_masses,
        &mol_inertia,
        &[],
        &[],
        &mut test_rng(),
    );

    // omega_x variance should be ~10× omega_y variance
    let ox: Vec<f64> = ang_vel.chunks(4).map(|c| c[0] as f64).collect();
    let oy: Vec<f64> = ang_vel.chunks(4).map(|c| c[1] as f64).collect();

    let var_ox: f64 = ox.iter().map(|v| v * v).sum::<f64>() / n_mol as f64;
    let var_oy: f64 = oy.iter().map(|v| v * v).sum::<f64>() / n_mol as f64;

    let ratio = var_ox / var_oy;
    let expected_ratio = inertia_y as f64 / inertia_x as f64; // 10.0
    assert!(
        (ratio - expected_ratio).abs() / expected_ratio < 0.3,
        "variance ratio={ratio:.2}, expected={expected_ratio:.1}"
    );
}

/// vec4 layout: w-component should always be zero for all velocity arrays.
#[test]
fn mb_velocities_w_components_are_zero() {
    let (com_vel, ang_vel, _) = generate_mb_velocities(
        2.494,
        &[1, 1, 0],
        &[18.0, 44.0, 28.0],
        &[1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        &[1, 0],
        &[12.0, 14.0],
        &mut test_rng(),
    );

    for (i, chunk) in com_vel.chunks(4).enumerate() {
        assert_eq!(chunk[3], 0.0, "com_vel[{i}].w != 0");
    }
    for (i, chunk) in ang_vel.chunks(4).enumerate() {
        assert_eq!(chunk[3], 0.0, "ang_vel[{i}].w != 0");
    }
}

// ============================================================================
// GPU kernel physics tests
// ============================================================================

/// Create a wgpu client for kernel tests.
fn test_client() -> Client {
    pipeline::wgpu_client().unwrap()
}

/// Reconstruct with identity quaternion should reproduce COM + ref_positions exactly.
#[test]
fn reconstruct_identity_quaternion() {
    let client = test_client();
    let n_atoms = 3u32;
    let n_molecules = 1u32;

    // One molecule at COM = (1, 2, 3) with 3 atoms at relative offsets
    let com = [1.0f32, 2.0, 3.0, 0.0];
    let refs = [
        0.5f32, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0,
    ];
    let identity_quat = [0.0f32, 0.0, 0.0, 1.0]; // [i, j, k, w]
    let offsets = [0u32, 3];
    let mol_rigid = [1u32];

    // Initial positions (will be overwritten by reconstruct)
    let positions = vec![0.0f32; n_atoms as usize * 4];

    let h_pos = client.create_from_slice(bytemuck::cast_slice(&positions));
    let h_com = client.create_from_slice(bytemuck::cast_slice(&com));
    let h_ref = client.create_from_slice(bytemuck::cast_slice(&refs));
    let h_quat = client.create_from_slice(bytemuck::cast_slice(&identity_quat));
    let h_offsets = client.create_from_slice(bytemuck::cast_slice(&offsets));
    let h_rigid = client.create_from_slice(bytemuck::cast_slice(&mol_rigid));

    let count = CubeCount::Static(1, 1, 1);
    let dim = CubeDim::new_1d(WORKGROUP_SIZE);

    unsafe {
        kernels::reconstruct_positions::launch_unchecked(
            &client,
            count,
            dim,
            BufferArg::from_raw_parts(h_com.clone(), 4),
            BufferArg::from_raw_parts(h_quat.clone(), 4),
            BufferArg::from_raw_parts(h_ref.clone(), 12),
            BufferArg::from_raw_parts(h_pos.clone(), 12),
            BufferArg::from_raw_parts(h_offsets.clone(), 2),
            BufferArg::from_raw_parts(h_rigid.clone(), 1),
            n_atoms,
            n_molecules,
        )
    };

    let result: Vec<f32> = bytemuck::cast_slice(&client.read_one(h_pos).unwrap()).to_vec();

    // Atom 0: COM + ref = (1.5, 2.0, 3.0)
    assert!((result[0] - 1.5).abs() < 1e-5, "atom 0 x: {}", result[0]);
    assert!((result[1] - 2.0).abs() < 1e-5, "atom 0 y: {}", result[1]);
    assert!((result[2] - 3.0).abs() < 1e-5, "atom 0 z: {}", result[2]);

    // Atom 1: COM + ref = (0.5, 2.0, 3.0)
    assert!((result[4] - 0.5).abs() < 1e-5, "atom 1 x: {}", result[4]);

    // Atom 2: COM + ref = (1.0, 2.3, 3.0)
    assert!((result[9] - 2.3).abs() < 1e-5, "atom 2 y: {}", result[9]);
}

/// Reconstruct with 90° z-rotation should rotate ref_positions in the xy-plane.
#[test]
fn reconstruct_90deg_z_rotation() {
    let client = test_client();

    let com = [0.0f32, 0.0, 0.0, 0.0];
    // Single atom at ref = (1, 0, 0); after 90° z-rotation → (0, 1, 0)
    let refs = [1.0f32, 0.0, 0.0, 0.0];
    // 90° around z: q = (0, 0, sin(π/4), cos(π/4)) = (0, 0, 0.7071, 0.7071) in [i,j,k,w]
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let quat = [0.0f32, 0.0, s, s];
    let offsets = [0u32, 1];
    let mol_rigid = [1u32];

    let positions = vec![0.0f32; 4];
    let h_pos = client.create_from_slice(bytemuck::cast_slice(&positions));
    let h_com = client.create_from_slice(bytemuck::cast_slice(&com));
    let h_ref = client.create_from_slice(bytemuck::cast_slice(&refs));
    let h_quat = client.create_from_slice(bytemuck::cast_slice(&quat));
    let h_offsets = client.create_from_slice(bytemuck::cast_slice(&offsets));
    let h_rigid = client.create_from_slice(bytemuck::cast_slice(&mol_rigid));

    unsafe {
        kernels::reconstruct_positions::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(h_com.clone(), 4),
            BufferArg::from_raw_parts(h_quat.clone(), 4),
            BufferArg::from_raw_parts(h_ref.clone(), 4),
            BufferArg::from_raw_parts(h_pos.clone(), 4),
            BufferArg::from_raw_parts(h_offsets.clone(), 2),
            BufferArg::from_raw_parts(h_rigid.clone(), 1),
            1u32,
            1u32,
        )
    };

    let result: Vec<f32> = bytemuck::cast_slice(&client.read_one(h_pos).unwrap()).to_vec();
    assert!(
        (result[0]).abs() < 1e-5,
        "x should be ~0, got {}",
        result[0]
    );
    assert!(
        (result[1] - 1.0).abs() < 1e-5,
        "y should be ~1, got {}",
        result[1]
    );
    assert!(
        (result[2]).abs() < 1e-5,
        "z should be ~0, got {}",
        result[2]
    );
}

/// Force reduction: two atoms with known forces → correct COM force and torque.
#[test]
fn reduce_forces_dimer() {
    let client = test_client();

    // Dimer: 2 atoms in 1 molecule, COM at origin
    // Atom 0 at (-1, 0, 0), Atom 1 at (1, 0, 0)
    let positions = [-1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let com = [0.0f32, 0.0, 0.0, 0.0];
    let offsets = [0u32, 2];
    let mol_rigid = [1u32];

    // Equal y-forces → zero COM force, nonzero z-torque
    // Atom 0: F=(0, 1, 0), Atom 1: F=(0, -1, 0)
    let forces = [0.0f32, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0];

    let com_forces = vec![0.0f32; 4];
    let torques = vec![0.0f32; 4];

    let h_forces = client.create_from_slice(bytemuck::cast_slice(&forces));
    let h_pos = client.create_from_slice(bytemuck::cast_slice(&positions));
    let h_com = client.create_from_slice(bytemuck::cast_slice(&com));
    let h_offsets = client.create_from_slice(bytemuck::cast_slice(&offsets));
    let h_com_forces = client.create_from_slice(bytemuck::cast_slice(&com_forces));
    let h_torques = client.create_from_slice(bytemuck::cast_slice(&torques));
    let h_rigid = client.create_from_slice(bytemuck::cast_slice(&mol_rigid));

    let box_length = 100.0f32;
    let inv_box = 1.0f32 / box_length;

    unsafe {
        kernels::reduce_forces_kernel::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(h_forces.clone(), 8),
            BufferArg::from_raw_parts(h_pos.clone(), 8),
            BufferArg::from_raw_parts(h_com.clone(), 4),
            BufferArg::from_raw_parts(h_offsets.clone(), 2),
            BufferArg::from_raw_parts(h_com_forces.clone(), 4),
            BufferArg::from_raw_parts(h_torques.clone(), 4),
            BufferArg::from_raw_parts(h_rigid.clone(), 1),
            1u32,
            box_length,
            inv_box,
        )
    };

    let cf: Vec<f32> = bytemuck::cast_slice(&client.read_one(h_com_forces).unwrap()).to_vec();
    let tau: Vec<f32> = bytemuck::cast_slice(&client.read_one(h_torques).unwrap()).to_vec();

    // COM force: (0,1,0) + (0,-1,0) = (0,0,0)
    assert!(cf[0].abs() < 1e-5, "Fx={}", cf[0]);
    assert!(cf[1].abs() < 1e-5, "Fy={}", cf[1]);
    assert!(cf[2].abs() < 1e-5, "Fz={}", cf[2]);

    // Torque: (-1,0,0)×(0,1,0) = (0,0,-1) and (1,0,0)×(0,-1,0) = (0,0,-1)
    // Total τ = (0, 0, -2)
    assert!(tau[0].abs() < 1e-5, "τx={}", tau[0]);
    assert!(tau[1].abs() < 1e-5, "τy={}", tau[1]);
    assert!((tau[2] - (-2.0)).abs() < 1e-5, "τz={}, expected -2", tau[2]);
}

/// Half-kick: known force → exact velocity change Δv = (dt/2) * F / m * conv.
#[test]
fn half_kick_exact_impulse() {
    let client = test_client();

    let mass = 18.0f32;
    let dt = 0.002f32;
    let force = [3.0f32, -1.0, 2.0, 0.0]; // COM force
    let torque = [0.0f32; 4]; // no torque for this test
    let vel = [0.0f32; 4]; // start from rest
    let omega = [0.0f32; 4];
    let quat = [0.0f32, 0.0, 0.0, 1.0]; // identity
    let inertia = [1.0f32, 1.0, 1.0, 0.0];
    let mol_rigid = [1u32];

    let h_vel = client.create_from_slice(bytemuck::cast_slice(&vel));
    let h_omega = client.create_from_slice(bytemuck::cast_slice(&omega));
    let h_force = client.create_from_slice(bytemuck::cast_slice(&force));
    let h_torque = client.create_from_slice(bytemuck::cast_slice(&torque));
    let h_quat = client.create_from_slice(bytemuck::cast_slice(&quat));
    let h_mass = client.create_from_slice(bytemuck::cast_slice(&[mass]));
    let h_inertia = client.create_from_slice(bytemuck::cast_slice(&inertia));
    let h_rigid = client.create_from_slice(bytemuck::cast_slice(&mol_rigid));

    unsafe {
        kernels::half_kick::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(h_vel.clone(), 4),
            BufferArg::from_raw_parts(h_omega.clone(), 4),
            BufferArg::from_raw_parts(h_force.clone(), 4),
            BufferArg::from_raw_parts(h_torque.clone(), 4),
            BufferArg::from_raw_parts(h_quat.clone(), 4),
            BufferArg::from_raw_parts(h_mass.clone(), 1),
            BufferArg::from_raw_parts(h_inertia.clone(), 4),
            BufferArg::from_raw_parts(h_rigid.clone(), 1),
            1u32,
            dt,
        )
    };

    let result: Vec<f32> = bytemuck::cast_slice(&client.read_one(h_vel).unwrap()).to_vec();

    // Expected: Δv = (dt/2) / m * 100 * F
    let conv = 100.0f32;
    let kick = 0.5 * dt / mass * conv;
    assert!(
        (result[0] - kick * force[0]).abs() < 1e-6,
        "vx={}, expected={}",
        result[0],
        kick * force[0]
    );
    assert!(
        (result[1] - kick * force[1]).abs() < 1e-6,
        "vy={}, expected={}",
        result[1],
        kick * force[1]
    );
    assert!(
        (result[2] - kick * force[2]).abs() < 1e-6,
        "vz={}, expected={}",
        result[2],
        kick * force[2]
    );
}

/// Single-atom rigid molecules avoid spline/bonded complexity while exercising
/// the core BAOAB integrator and thermostat.
fn make_free_particles(n: usize, mass: f32, kt: f32, rng: &mut impl Rng) -> LangevinUploadData {
    let mol_is_rigid = vec![1u32; n];
    let mol_masses = vec![mass; n];
    // Zero inertia is safe: `safe_inv(0)` returns 0 in the kernel,
    // so angular noise vanishes for point particles.
    let mol_inertia = vec![0.0f32; n * 4];
    let atom_is_flexible = vec![0u32; n];
    let atom_masses = vec![mass; n];

    let (com_velocities, angular_velocities, atom_velocities) = generate_mb_velocities(
        kt as f64,
        &mol_is_rigid,
        &mol_masses,
        &mol_inertia,
        &atom_is_flexible,
        &atom_masses,
        rng,
    );

    let positions = vec![0.0f32; n * 4];
    let ref_positions = vec![0.0f32; n * 4];
    let com_positions = vec![0.0f32; n * 4];
    let mol_atom_offsets: Vec<u32> = (0..=n as u32).collect();
    let quaternions: Vec<f32> = (0..n).flat_map(|_| [0.0, 0.0, 0.0, 1.0]).collect();
    let excl_offsets = vec![0u32; n + 1];

    LangevinUploadData {
        positions,
        ref_positions,
        com_positions,
        mol_atom_offsets,
        mol_masses,
        mol_inertia,
        quaternions,
        com_velocities,
        angular_velocities,
        atom_type_ids: None,
        mol_ids: None,
        spline_params: None,
        spline_coeffs: None,
        n_atom_types: 0,
        bond_data: None,
        angle_data: None,
        dihedral_data: None,
        excl_offsets,
        excl_atoms: Vec::new(),
        mol_is_rigid,
        atom_velocities,
        atom_masses,
        atom_is_flexible,
        has_flexible: false,
    }
}

/// Per-atom half-kick: same physics as rigid half-kick but different kernel.
#[test]
fn half_kick_atoms_exact_impulse() {
    let client = test_client();

    let mass = 12.0f32;
    let dt = 0.002f32;
    let force = [5.0f32, -2.0, 1.0, 0.0]; // vec4 layout
    let vel = [0.0f32; 3]; // stride-3 layout
    let is_flex = [1u32];

    let h_vel = client.create_from_slice(bytemuck::cast_slice(&vel));
    let h_force = client.create_from_slice(bytemuck::cast_slice(&force));
    let h_mass = client.create_from_slice(bytemuck::cast_slice(&[mass]));
    let h_flex = client.create_from_slice(bytemuck::cast_slice(&is_flex));

    unsafe {
        kernels::half_kick_atoms::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            BufferArg::from_raw_parts(h_vel.clone(), 3),
            BufferArg::from_raw_parts(h_force.clone(), 4),
            BufferArg::from_raw_parts(h_mass.clone(), 1),
            BufferArg::from_raw_parts(h_flex.clone(), 1),
            1u32,
            dt,
        )
    };

    let result: Vec<f32> = bytemuck::cast_slice(&client.read_one(h_vel).unwrap()).to_vec();

    let conv = 100.0f32;
    let kick = 0.5 * dt / mass * conv;
    assert!(
        (result[0] - kick * force[0]).abs() < 1e-6,
        "vx={}, expected={}",
        result[0],
        kick * force[0]
    );
    assert!(
        (result[1] - kick * force[1]).abs() < 1e-6,
        "vy={}, expected={}",
        result[1],
        kick * force[1]
    );
    assert!(
        (result[2] - kick * force[2]).abs() < 1e-6,
        "vz={}, expected={}",
        result[2],
        kick * force[2]
    );
}

// ============================================================================
// End-to-end physics regression tests (BAOAB integrator + thermostat)
// ============================================================================

/// Shared fixture for physics regression tests.
struct PhysicsTestSetup {
    gpu: LangevinGpu,
    n: usize,
    mass: f32,
    kt: f32,
    friction: f64,
    dt: f64,
}

impl PhysicsTestSetup {
    /// N=200 argon-like particles at 300K with standard LD parameters.
    ///
    /// `seed` fixes both the initial velocities and the thermostat's noise, so a
    /// run is reproducible. Each physics test below passes a different one: the
    /// assertions are statistical, and a single realization could in principle
    /// satisfy them while a regression is present.
    fn with_seed(seed: u64) -> Self {
        let n = 200;
        let mass = 39.948f32; // argon
        let temperature = 300.0;
        let friction = 10.0;
        let dt = 0.002;
        // Large box to avoid PBC artifacts in free-particle tests
        let box_length = 1000.0f32;
        let kt = (crate::R_IN_KJ_PER_MOL * temperature) as f32;

        let client = test_client();
        let config = LangevinConfig {
            timestep: dt,
            friction,
            steps: 0,
            temperature,
            cell_list_rebuild: 0,
        };
        let data = make_free_particles(n, mass, kt, &mut StdRng::seed_from_u64(seed));
        let mut gpu = LangevinGpu::new(
            client,
            config,
            n as u32,
            n as u32,
            box_length,
            kt,
            seed as u32,
        );
        gpu.upload_state(data);
        Self {
            gpu,
            n,
            mass,
            kt,
            friction,
            dt,
        }
    }
}

/// Force callback for free particles (pure thermostat-driven diffusion).
type ForceResult = (Vec<[f32; 4]>, Vec<[f32; 4]>);

fn zero_forces(n: usize) -> impl FnMut(&[[f32; 4]]) -> ForceResult {
    move |_| (vec![[0.0f32; 4]; n], vec![[0.0f32; 4]; n])
}

/// Isotropic harmonic trap F=-k·r confines particles for equipartition tests.
fn harmonic_forces(k: f32) -> impl FnMut(&[[f32; 4]]) -> ForceResult {
    move |positions| {
        let forces: Vec<[f32; 4]> = positions
            .iter()
            .map(|p| [-k * p[0], -k * p[1], -k * p[2], 0.0])
            .collect();
        let torques = vec![[0.0f32; 4]; positions.len()];
        (forces, torques)
    }
}

/// Einstein relation: MSD(t) = 6·D·t where D = kT·conv/(m·γ).
/// Verifies that the thermostat produces correct diffusive transport.
#[test]
fn free_particle_diffusion() {
    let PhysicsTestSetup {
        mut gpu,
        n,
        mass,
        kt,
        friction,
        dt,
    } = PhysicsTestSetup::with_seed(0x0000_0001);
    let total_steps = 2000;

    let initial_pos = gpu.download_positions().unwrap();
    gpu.run_steps_with_cpu_forces(total_steps, &mut zero_forces(n))
        .unwrap();
    let pos = gpu.download_positions().unwrap();

    let msd: f32 = initial_pos
        .iter()
        .zip(pos.iter())
        .map(|(a, b)| {
            let dx = b[0] - a[0];
            let dy = b[1] - a[1];
            let dz = b[2] - a[2];
            dx * dx + dy * dy + dz * dz
        })
        .sum::<f32>()
        / n as f32;

    let t = total_steps as f64 * dt;
    let d = kt as f64 * 100.0 / (mass as f64 * friction); // 100 = kJ/mol → amu·Å²/ps²
    let expected_msd = 6.0 * d * t;
    let ratio = msd as f64 / expected_msd;
    assert!(
        (0.5..=1.8).contains(&ratio),
        "MSD={msd:.2}, expected={expected_msd:.2}, ratio={ratio:.3}"
    );
}

/// Equipartition: ⟨x²⟩ = kT/k for a harmonic oscillator.
/// Verifies that the thermostat samples the correct Boltzmann distribution.
#[test]
fn harmonic_oscillator_position_variance() {
    let PhysicsTestSetup { mut gpu, n, kt, .. } = PhysicsTestSetup::with_seed(0x0000_0002);
    let k_spring = 1.0f32; // kJ/mol/Å²
    let mut force_fn = harmonic_forces(k_spring);

    gpu.run_steps_with_cpu_forces(1000, &mut force_fn).unwrap();

    // Sample in blocks to decorrelate snapshots
    let n_blocks = 10;
    let mut sum_x2 = 0.0f64;
    for _ in 0..n_blocks {
        gpu.run_steps_with_cpu_forces(500, &mut force_fn).unwrap();
        for p in &gpu.download_positions().unwrap() {
            sum_x2 += (p[0] as f64).powi(2);
        }
    }

    let var_x = sum_x2 / (n_blocks * n) as f64;
    let expected_var = kt as f64 / k_spring as f64;
    let ratio = var_x / expected_var;
    assert!(
        (0.5..=1.8).contains(&ratio),
        "⟨x²⟩={var_x:.3}, expected={expected_var:.3}, ratio={ratio:.3}"
    );
}

/// Verifies that the O-step thermostat drives translational kinetic energy to kT.
#[test]
fn temperature_equilibrium() {
    let PhysicsTestSetup { mut gpu, .. } = PhysicsTestSetup::with_seed(0x0000_0003);
    // Harmonic trap prevents unbounded drift that would make KE sampling noisy
    let mut force_fn = harmonic_forces(1.0);

    gpu.run_steps_with_cpu_forces(2000, &mut force_fn).unwrap();

    let (t_trans, _) = gpu.download_temperature().unwrap();
    let ratio = t_trans as f64 / 300.0;
    assert!(
        (0.7..=1.3).contains(&ratio),
        "T_trans={t_trans:.1}K, expected≈300K, ratio={ratio:.3}"
    );
}

/// C_v(t)/C_v(0) = exp(-γt) for free Brownian particles.
/// Verifies that the friction coefficient is correctly applied in the O-step.
#[test]
fn velocity_autocorrelation_decay() {
    let PhysicsTestSetup {
        mut gpu,
        n,
        friction,
        dt,
        ..
    } = PhysicsTestSetup::with_seed(0x0000_0004);
    // 50 steps × 0.002 ps = 0.1 ps → exactly one friction time (1/γ)
    let steps_per_block = 50;

    gpu.run_steps_with_cpu_forces(1000, &mut zero_forces(n))
        .unwrap();

    let v0 = gpu.download_com_velocities().unwrap();
    let cv0: f64 = v0
        .iter()
        .map(|v| (v[0] as f64).powi(2) + (v[1] as f64).powi(2) + (v[2] as f64).powi(2))
        .sum::<f64>()
        / n as f64;

    // Run one block and check decay at t=0.1ps (γ=10 → exp(-1) ≈ 0.368)
    gpu.run_steps_with_cpu_forces(steps_per_block, &mut zero_forces(n))
        .unwrap();
    let vt = gpu.download_com_velocities().unwrap();
    let cv: f64 = v0
        .iter()
        .zip(vt.iter())
        .map(|(a, b)| {
            a[0] as f64 * b[0] as f64 + a[1] as f64 * b[1] as f64 + a[2] as f64 * b[2] as f64
        })
        .sum::<f64>()
        / n as f64;

    let t = steps_per_block as f64 * dt;
    let ratio = cv / cv0;
    let expected = (-friction * t).exp();
    assert!(
        (0.15..=0.65).contains(&ratio),
        "C_v(t)/C_v(0)={ratio:.3}, expected≈{expected:.3} at t={t}ps"
    );
}

// ---------------------------------------------------------------------------
// Reproducibility: a fixed `propagate.seed` must pin a Langevin run.
// ---------------------------------------------------------------------------

/// Initial Maxwell-Boltzmann velocities must come from the run's generator.
#[test]
fn mb_velocities_follow_the_seed() {
    let draw = |seed: u64| {
        let mut rng = StdRng::seed_from_u64(seed);
        generate_mb_velocities(
            2.494,
            &[1, 1, 1],
            &[18.0, 44.0, 28.0],
            &[1.0; 12],
            &[1, 1],
            &[12.0, 14.0],
            &mut rng,
        )
    };

    assert_eq!(draw(7), draw(7), "same seed gave different velocities");
    assert_ne!(
        draw(7),
        draw(8),
        "velocities ignore the seed, so they are drawn from somewhere else"
    );
}

/// Run free particles under the thermostat alone and return their positions.
///
/// The initial velocities are held fixed, so the only remaining source of
/// randomness is the Philox key driving the O-step noise.
fn thermostat_positions(rng_seed: u32) -> Vec<[f32; 4]> {
    let (n, mass, temperature) = (32usize, 39.948f32, 300.0);
    let kt = (crate::R_IN_KJ_PER_MOL * temperature) as f32;
    let config = LangevinConfig {
        timestep: 0.002,
        friction: 10.0,
        steps: 0,
        temperature,
        cell_list_rebuild: 0,
    };
    let data = make_free_particles(n, mass, kt, &mut test_rng());
    let mut gpu = LangevinGpu::new(
        test_client(),
        config,
        n as u32,
        n as u32,
        1000.0,
        kt,
        rng_seed,
    );
    gpu.upload_state(data);
    gpu.run_steps_with_cpu_forces(50, &mut zero_forces(n))
        .unwrap();
    gpu.download_positions().unwrap()
}

/// The thermostat noise must follow the seed, not a hardcoded constant.
///
/// Identical initial velocities are used for every run, so a difference in the
/// final positions can only come from the O-step's random forces.
#[test]
fn thermostat_noise_follows_the_seed() {
    assert_eq!(
        thermostat_positions(7),
        thermostat_positions(7),
        "same Philox key gave different trajectories"
    );
    assert_ne!(
        thermostat_positions(7),
        thermostat_positions(8),
        "thermostat noise ignores its key, so every run shares one noise stream"
    );
}

// ============================================================================
// Rigid-body reference frame
// ============================================================================

/// A rigid trimer, rotated so its carried orientation is *not* the identity.
///
/// Every other reconstruct test starts from the identity quaternion, where a lab-frame and a
/// body-frame reference coincide — which is exactly why the two could be confused for each other.
fn rotated_trimer() -> crate::backend::Backend {
    use crate::context::Context;

    let mut context = ld_system(
        r#"
atoms:
  - {name: A, mass: 1.0, charge: 0.0}
molecules:
  - name: TRIMER
    atoms: [A, A, A]
    degrees_of_freedom: Rigid
system:
  cell: !Cuboid [40.0, 40.0, 40.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: []
  blocks:
    - molecule: TRIMER
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.5, 0.0]]
"#,
    );

    // An orientation off every axis, so a lab/body mix-up cannot cancel by symmetry.
    let rotation = crate::UnitQuaternion::from_euler_angles(0.3, -0.7, 1.1);
    context.rotate_group(0, &rotation).unwrap();
    context
}

/// Uploading a state and reconstructing it must return the coordinates it was given.
///
/// The kernel computes `r = com + R(q)·ref`, so `ref` must be the atom's position in the *body*
/// frame. Uploading the lab-frame displacement `r − com` instead makes the first reconstruct apply
/// `R(q)` to coordinates that already carry it, rotating every rigid molecule a second time by its
/// own orientation — silently, and before a single dynamics step has been taken.
#[test]
fn upload_and_reconstruct_preserves_a_rotated_rigid_molecule() {
    use crate::group::GroupCollection;
    use crate::Point;

    let context = rotated_trimer();
    let config = LangevinConfig {
        timestep: 0.001,
        friction: 1.0,
        steps: 0,
        temperature: 300.0,
        cell_list_rebuild: 20,
    };
    let mut rng = test_rng();

    let expected: Vec<Point> = (0..context.num_particles())
        .map(|i| context.position(i))
        .collect();

    let mut gpu = LangevinRunner::init_gpu(&context, &config, &mut rng).unwrap();
    LangevinRunner::upload_full_state(&context, &mut gpu, &mut rng).unwrap();
    gpu.dispatch_reconstruct();

    for (i, (got, want)) in gpu
        .download_positions()
        .unwrap()
        .iter()
        .zip(&expected)
        .enumerate()
    {
        let got = Point::new(got[0] as f64, got[1] as f64, got[2] as f64);
        assert!(
            (got - want).norm() < 1e-4,
            "atom {i} reconstructed at {got:?}, uploaded at {want:?}"
        );
    }
}

/// The kernels use one box length, so only a cubic, fully periodic cell is valid.
#[test]
fn check_supported_requires_cubic_cuboid() {
    let check = |cell| {
        LangevinRunner::check_supported(
            &LdCase {
                cell,
                ..Default::default()
            }
            .context(),
        )
    };
    assert!(check("!Cuboid [40.0, 40.0, 40.0]").is_ok());
    assert!(check("!Cuboid [40.0, 40.0, 50.0]").is_err());
    assert!(check("!Slit [40.0, 40.0, 40.0]").is_err());
}

#[test]
fn check_supported_names_unsupported_terms() {
    let context = LdCase {
        extra: r#"    - !CustomExternal {selection: "all", function: "0.1 * z"}"#,
        ..Default::default()
    }
    .context();
    let err = LangevinRunner::check_supported(&context).unwrap_err();
    assert!(err.to_string().contains("custom_external"), "{err}");
}

// ============================================================================
// On-device forces vs the CPU Hamiltonian
// ============================================================================

/// The LD settings the tests below vary from.
fn ld_config(steps: usize) -> LangevinConfig {
    LangevinConfig {
        timestep: 0.005,
        friction: 10.0,
        steps,
        temperature: 298.15,
        cell_list_rebuild: 10,
    }
}

/// Largest deviation of `got` from `want`, relative to the largest `want`.
fn relative_deviation(got: &[[f32; 4]], want: &[[f32; 4]]) -> f32 {
    let norm = |v: &[f32; 4]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let diff = |a: &[f32; 4], b: &[f32; 4]| norm(&[a[0] - b[0], a[1] - b[1], a[2] - b[2], 0.0]);
    let scale = want.iter().map(norm).fold(0.0, f32::max);
    assert!(
        scale > 1e-3,
        "nothing to compare: largest reference {scale}"
    );
    got.iter()
        .zip(want)
        .map(|(a, b)| diff(a, b))
        .fold(0.0, f32::max)
        / scale
}

/// The first force pass of a Langevin block must match the CPU Hamiltonian.
///
/// Rigid molecules compare by COM force and torque, since the device skips their internal pairs
/// (which cancel there); flexible atoms compare one by one.
fn assert_device_forces_match_cpu(mut context: crate::backend::Backend) {
    let mut gpu = uploaded_gpu(&context);
    assert_block_forces_match_cpu(&mut context, &mut gpu);
}

/// A pipeline holding the first block's upload of `context`.
fn uploaded_gpu(context: &crate::backend::Backend) -> LangevinGpu {
    let mut rng = test_rng();
    let mut gpu = LangevinRunner::init_gpu(context, &ld_config(0), &mut rng).unwrap();
    LangevinRunner::upload_full_state(context, &mut gpu, &mut rng).unwrap();
    assert!(gpu.has_gpu_forces, "case must run the on-device force path");
    gpu
}

fn assert_block_forces_match_cpu(context: &mut crate::backend::Backend, gpu: &mut LangevinGpu) {
    LangevinRunner::integrate(context, gpu, 0).unwrap();
    let device = gpu.download_forces().unwrap();

    let cpu_atoms = context.hamiltonian().forces(context);
    let (cpu_com, cpu_torque) = reduce_forces_to_com(context, &cpu_atoms);
    // Compare what the device integrates, by the upload's own classification
    let flagged = |flags: &[u32]| {
        (0..flags.len())
            .filter(|&i| flags[i] != 0)
            .collect::<Vec<_>>()
    };
    let rigid = flagged(&gpu.mol_is_rigid_host);
    let flexible = flagged(&gpu.atom_is_flexible_host);

    let pick = |v: &[[f32; 4]], idx: &[usize]| idx.iter().map(|&i| v[i]).collect::<Vec<_>>();
    // Software Vulkan (lavapipe) f32 trig is ~1e-3 coarse; a dropped term deviates by ≳0.25
    let tolerance = 5e-3;
    if !rigid.is_empty() {
        let com = relative_deviation(&pick(&device.com, &rigid), &pick(&cpu_com, &rigid));
        assert!(com < tolerance, "rigid COM force deviates by {com:.2e}");
        let tau = relative_deviation(&pick(&device.torques, &rigid), &pick(&cpu_torque, &rigid));
        assert!(tau < tolerance, "rigid torque deviates by {tau:.2e}");
    }
    if !flexible.is_empty() {
        let want: Vec<_> = flexible
            .iter()
            .map(|&i| numeric_force(context, i))
            .collect();
        let atoms = relative_deviation(&pick(&device.atoms, &flexible), &want);
        assert!(
            atoms < tolerance,
            "flexible atom force deviates by {atoms:.2e}"
        );
    }
}

/// −∂U/∂rᵢ by central difference of the total energy. Bonded terms have no CPU
/// `forces()`, so this is the only reference that covers every term.
fn numeric_force(context: &mut crate::backend::Backend, i: usize) -> [f32; 4] {
    use crate::context::Context as _;
    use crate::energy::EnergyChange as _;
    use crate::group::{GroupCollection as _, GroupCollectionMut as _};
    const H: f64 = 1e-4;
    let origin = context.position(i);
    let mut energy_at = |dr: crate::Point| {
        context.set_positions([i], [&(origin + dr)]);
        context.update(&crate::Change::Everything).unwrap();
        context
            .hamiltonian()
            .energy(context, &crate::Change::Everything)
    };
    let mut force = [0.0f32; 4];
    for (axis, f) in force.iter_mut().take(3).enumerate() {
        let dr = crate::Point::ith(axis, H);
        *f = (-(energy_at(dr) - energy_at(-dr)) / (2.0 * H)) as f32;
    }
    energy_at(crate::Point::zeros());
    force
}

/// Splined LJ + Coulomb, with a rigid dimer and free ions interacting across the box faces.
#[test]
fn device_pair_forces_across_the_periodic_boundary() {
    assert_device_forces_match_cpu(ld_system(
        r#"
atoms:
  - {name: A, charge: 1.0, mass: 10.0, sigma: 2.5, epsilon: 0.5}
  - {name: B, charge: -1.0, mass: 12.0, sigma: 2.5, epsilon: 0.5}
molecules:
  - {name: dimer, atoms: [A, B], degrees_of_freedom: Rigid}
  - {name: ion, atoms: [B], degrees_of_freedom: Free}
system:
  medium: {permittivity: !Vacuum, temperature: 298.15}
  cell: !Cuboid [40.0, 40.0, 40.0]
  blocks:
    - {molecule: dimer, N: 1, insert: !Manual [[18.5, 0.0, 0.0], [-19.0, 0.8, 0.0]]}
    - {molecule: ion, N: 2, insert: !Manual [[-16.0, 1.5, 0.5], [16.0, -1.0, 19.2]]}
  energy:
    - !Nonbonded
      default:
        - !LennardJones {mixing: LorentzBerthelot}
        - !CoulombPlain {cutoff: 12.0}
      spline: {cutoff: 12.0}
"#,
    ));
}

/// A free trimer: the bonded 0–1 and 1–2 pairs are excluded, the 1–3 pair 0–2 is not.
#[test]
fn device_pair_forces_skip_excluded_flexible_pairs() {
    assert_device_forces_match_cpu(ld_system(
        r#"
atoms:
  - {name: A, charge: 1.0, mass: 10.0, sigma: 3.0, epsilon: 0.5}
  - {name: B, charge: -1.0, mass: 12.0, sigma: 3.0, epsilon: 0.5}
molecules:
  - name: chain
    atoms: [A, B, A]
    degrees_of_freedom: Free
    excluded_neighbours: 1
    bonds:
      - {index: [0, 1], kind: !Harmonic {k: 5.0, req: 2.0}}
      - {index: [1, 2], kind: !Harmonic {k: 5.0, req: 2.0}}
system:
  medium: {permittivity: !Vacuum, temperature: 298.15}
  cell: !Cuboid [40.0, 40.0, 40.0]
  blocks:
    - {molecule: chain, N: 1, insert: !Manual [[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [2.8, 3.1, 0.0]]}
  energy:
    - !Nonbonded
      default:
        - !LennardJones {mixing: LorentzBerthelot}
        - !CoulombPlain {cutoff: 12.0}
      spline: {cutoff: 12.0}
"#,
    ));
}

/// A free, non-planar 4-mer whose only forces are bonded: all intramolecular pairs are excluded.
#[test]
fn device_bonded_forces() {
    assert_device_forces_match_cpu(ld_system(
        r#"
atoms:
  - {name: A, charge: 0.0, mass: 10.0, sigma: 3.0, epsilon: 0.5}
molecules:
  - name: chain
    atoms: [A, A, A, A]
    degrees_of_freedom: Free
    excluded_neighbours: 3
    bonds:
      - {index: [0, 1], kind: !Harmonic {k: 5.0, req: 2.0}}
      - {index: [1, 2], kind: !Harmonic {k: 5.0, req: 2.0}}
      - {index: [2, 3], kind: !Harmonic {k: 5.0, req: 2.0}}
    torsions:
      - {index: [0, 1, 2], kind: !Harmonic {k: 0.02, aeq: 109.0}}
      - {index: [1, 2, 3], kind: !Harmonic {k: 0.02, aeq: 120.0}}
    dihedrals:
      - {index: [0, 1, 2, 3], kind: !ProperHarmonic {k: 0.01, aeq: 60.0}}
system:
  medium: {permittivity: !Vacuum, temperature: 298.15}
  cell: !Cuboid [40.0, 40.0, 40.0]
  blocks:
    - {molecule: chain, N: 1, insert: !Manual [[0.0, 0.0, 0.0], [2.3, 0.0, 0.0], [3.0, 1.9, 0.0], [4.6, 2.2, 1.5]]}
  energy:
    - !Nonbonded
      default: [!LennardJones {mixing: LorentzBerthelot}]
      spline: {cutoff: 12.0}
"#,
    ));
}

/// `CustomPair` has host forces only; they reach the rigid bodies through the COM overlay.
#[test]
fn device_forces_include_the_host_overlay() {
    assert_device_forces_match_cpu(ld_system(
        r#"
atoms:
  - {name: A, charge: 0.0, mass: 10.0, sigma: 3.0, epsilon: 0.5}
molecules:
  - {name: P, atoms: [A, A], degrees_of_freedom: Rigid}
  - {name: Q, atoms: [A, A], degrees_of_freedom: Rigid}
system:
  medium: {permittivity: !Vacuum, temperature: 298.15}
  cell: !Cuboid [40.0, 40.0, 40.0]
  blocks:
    - {molecule: P, N: 1, insert: !Manual [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]]}
    - {molecule: Q, N: 1, insert: !Manual [[8.0, 0.0, 1.0], [9.0, 2.0, 0.0]]}
  energy:
    - !Nonbonded
      default: [!LennardJones {mixing: LorentzBerthelot}]
      spline: {cutoff: 12.0}
    - !CustomPair
        selections: ["molecule P", "molecule Q"]
        function: "0.5 * k * (r - r0)^2"
        constants: {k: 10.0, r0: 5.0}
"#,
    ));
}

/// Reciprocal-dominated Ewald (α·r_c = 3.6, sites ≥ 5 Å apart) on rigid dimers and free ions,
/// so a missing or mis-scaled k-space force shows above the f32 tolerance.
const EWALD_SYSTEM: &str = r#"
atoms:
  - {name: A, charge: 1.0, mass: 10.0, sigma: 1.0, epsilon: 0.1}
  - {name: B, charge: -1.0, mass: 12.0, sigma: 1.0, epsilon: 0.1}
molecules:
  - {name: dimer, atoms: [A, B], degrees_of_freedom: Rigid}
  - {name: ion, atoms: [A], degrees_of_freedom: Free}
  - {name: counterion, atoms: [B], degrees_of_freedom: Free}
system:
  medium: {permittivity: !Vacuum, temperature: 298.15}
  cell: !Cuboid [30.0, 30.0, 30.0]
  blocks:
    - {molecule: dimer, N: 2, insert: !Manual [[1.0, 0.5, 0.0], [6.0, 2.0, 1.0], [-9.0, -8.0, 4.0], [-13.5, -10.0, 6.0]]}
    - {molecule: ion, N: 2, insert: !Manual [[12.0, -11.0, -7.0], [-4.0, 9.0, 13.5]]}
    - {molecule: counterion, N: 2, insert: !Manual [[7.0, 12.0, -3.0], [-12.0, 3.0, -14.0]]}
  energy:
    - !Nonbonded
      default: [!LennardJones {mixing: LorentzBerthelot}]
      spline: {cutoff: 8.0}
    - !Ewald {cutoff: 8.0, accuracy: 1e-5, alpha: 0.45}
"#;

#[test]
fn device_ewald_reciprocal_forces() {
    assert_device_forces_match_cpu(ld_system(EWALD_SYSTEM));
}

/// MC may resize the box between LD blocks; the k-vectors must follow it.
#[test]
fn device_ewald_forces_follow_a_resized_box() {
    use crate::context::Context as _;
    let mut context = ld_system(EWALD_SYSTEM);
    let mut gpu = uploaded_gpu(&context);
    context
        .scale_volume_and_positions(34.0f64.powi(3), crate::cell::VolumeScalePolicy::Isotropic)
        .unwrap();
    context.update(&crate::Change::Everything).unwrap();
    LangevinRunner::upload_context_state(&context, &mut gpu).unwrap();
    assert_block_forces_match_cpu(&mut context, &mut gpu);
}

// ============================================================================
// Langevin blocks with on-device forces
// ============================================================================

/// Centres of an `n³` cubic lattice filling a box of side `n · spacing`.
fn lattice_sites(n: usize, spacing: f64) -> impl Iterator<Item = (f64, f64, f64)> {
    let at = move |d: usize| (d as f64 + 0.5 - 0.5 * n as f64) * spacing;
    itertools::iproduct!(0..n, 0..n, 0..n).map(move |(i, j, k)| (at(i), at(j), at(k)))
}

/// Each site plus each offset in `layout`, as a YAML `!Manual` position list.
fn manual_positions(sites: impl Iterator<Item = (f64, f64, f64)>, layout: &[[f64; 3]]) -> String {
    sites
        .flat_map(|(x, y, z)| {
            layout
                .iter()
                .map(move |d| format!("[{}, {}, {}]", x + d[0], y + d[1], z + d[2]))
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// `n³` rigid dipolar trimers on a lattice, plus `m³` free, bonded 4-mers on a centred
/// `m`-lattice of the same spacing (in the trimer voids for odd `n`, even `m`). Splined LJ +
/// Coulomb throughout.
fn trimers_and_chains(n: usize, spacing: f64, m: usize) -> crate::backend::Backend {
    let box_length = n as f64 * spacing;
    let trimer = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.5, 0.5]];
    let chain = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.6, 1.8, 0.0],
        [4.0, 2.2, 1.4],
    ];
    let mut blocks = format!(
        "    - {{molecule: trimer, N: {}, insert: !Manual [{}]}}",
        n * n * n,
        manual_positions(lattice_sites(n, spacing), &trimer)
    );
    if m > 0 {
        blocks += &format!(
            "\n    - {{molecule: chain, N: {}, insert: !Manual [{}]}}",
            m * m * m,
            manual_positions(lattice_sites(m, spacing), &chain)
        );
    }
    ld_system(&format!(
        r#"
atoms:
  - {{name: P, charge: 1.0, mass: 10.0, sigma: 3.0, epsilon: 0.5}}
  - {{name: N, charge: -1.0, mass: 12.0, sigma: 3.0, epsilon: 0.5}}
  - {{name: O, charge: 0.0, mass: 14.0, sigma: 3.0, epsilon: 0.5}}
molecules:
  - {{name: trimer, atoms: [P, N, O], degrees_of_freedom: Rigid}}
  - name: chain
    atoms: [P, O, O, N]
    degrees_of_freedom: Free
    excluded_neighbours: 1
    bonds:
      - {{index: [0, 1], kind: !Harmonic {{k: 5.0, req: 2.0}}}}
      - {{index: [1, 2], kind: !Harmonic {{k: 5.0, req: 2.0}}}}
      - {{index: [2, 3], kind: !Harmonic {{k: 5.0, req: 2.0}}}}
    torsions:
      - {{index: [0, 1, 2], kind: !Harmonic {{k: 0.02, aeq: 109.0}}}}
    dihedrals:
      - {{index: [0, 1, 2, 3], kind: !ProperHarmonic {{k: 0.01, aeq: 60.0}}}}
system:
  medium: {{permittivity: !Water, temperature: 298.15}}
  cell: !Cuboid [{box_length}, {box_length}, {box_length}]
  blocks:
{blocks}
  energy:
    - !Nonbonded
      default:
        - !LennardJones {{mixing: LorentzBerthelot}}
        - !CoulombPlain {{cutoff: 12.0}}
      spline: {{cutoff: 12.0}}
"#
    ))
}

/// Block temperatures `t` must average to `target` within 3σ, for `dof` degrees of freedom
/// sampled in `blocks` near-independent blocks.
fn assert_thermalised(name: &str, t: &Variance, target: f64, dof: f64, blocks: usize) {
    let sigma = target * (2.0 / dof).sqrt() / (blocks as f64).sqrt();
    assert!(
        (t.mean() - target).abs() < 3.0 * sigma,
        "{name} T = {:.1} K, expected {target} ± {sigma:.1} K",
        t.mean()
    );
}

/// Run `blocks` Langevin blocks from a fixed seed; return the runner and final positions.
fn run_blocks(
    context: &mut crate::backend::Backend,
    config: &LangevinConfig,
    seed: u64,
    blocks: usize,
) -> (LangevinRunner, Vec<[f32; 4]>) {
    let mut runner = LangevinRunner::new(config.clone());
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..blocks {
        runner.propagate(context, &mut rng).unwrap();
    }
    let positions = runner.gpu.as_ref().unwrap().download_positions().unwrap();
    (runner, positions)
}

/// Full blocks through `run_steps`: thermalised and reproducible from the seed.
#[test]
fn rigid_blocks_with_device_forces_thermalise() {
    let (n, blocks) = (4, 20);
    let config = ld_config(50);
    let (runner, positions) = run_blocks(&mut trimers_and_chains(n, 15.0, 0), &config, 11, blocks);
    assert!(runner.gpu.as_ref().unwrap().has_gpu_forces);

    // Kinetic energy decorrelates as e^{−2γt}, and γt = 2.5 per block, so block
    // temperatures are near-independent draws with relative spread √(2/3N).
    let dof = 3.0 * (n * n * n) as f64;
    let t = config.temperature;
    assert_thermalised("translational", &runner.t_trans, t, dof, blocks);
    assert_thermalised("rotational", &runner.t_rot, t, dof, blocks);

    let (_, again) = run_blocks(&mut trimers_and_chains(n, 15.0, 0), &config, 11, blocks);
    assert_eq!(positions, again, "same seed gave a different trajectory");
}

/// `n³` free molecules of `atoms` on a lattice, each at a site plus the offsets in `layout`.
/// The LJ is weak and short-ranged, so the molecules hardly interact.
fn dilute_free_molecules(n: usize, molecule: &str, layout: &[[f64; 3]]) -> crate::backend::Backend {
    let spacing = 50.0;
    let box_length = n as f64 * spacing;
    let positions = manual_positions(lattice_sites(n, spacing), layout);
    ld_system(&format!(
        r#"
atoms:
  - {{name: Ar, charge: 0.0, mass: 39.948, sigma: 1.0, epsilon: 0.01}}
molecules:
  - {molecule}
system:
  medium: {{permittivity: !Vacuum, temperature: 298.15}}
  cell: !Cuboid [{box_length}, {box_length}, {box_length}]
  blocks:
    - {{molecule: free, N: {}, insert: !Manual [{}]}}
  energy:
    - !Nonbonded
      default: [!LennardJones {{mixing: LorentzBerthelot}}]
      spline: {{cutoff: 3.0}}
"#,
        n * n * n,
        positions
    ))
}

fn flexible_config(steps: usize) -> LangevinConfig {
    LangevinConfig {
        timestep: 0.002,
        cell_list_rebuild: 20,
        ..ld_config(steps)
    }
}

/// Einstein relation for free flexible atoms, including the ballistic start from
/// Maxwell–Boltzmann velocities: MSD = 6D(t − (1 − e^{−γt})/γ), D = kT/(mγ).
#[test]
fn flexible_atom_diffusion() {
    use crate::group::GroupCollection as _;
    use crate::WithSimulationCell as _;
    let n = 8;
    let config = flexible_config(2000);
    let mut context = dilute_free_molecules(
        n,
        "{name: free, atoms: [Ar], degrees_of_freedom: Free}",
        &[[0.0; 3]],
    );
    let initial: Vec<_> = (0..context.num_particles())
        .map(|i| context.position(i))
        .collect();
    run_blocks(&mut context, &config, 21, 1);

    let msd = initial
        .iter()
        .enumerate()
        .map(|(i, r0)| {
            context
                .cell()
                .distance(&context.position(i), r0)
                .norm_squared()
        })
        .sum::<f64>()
        / initial.len() as f64;
    let (gamma, t) = (config.friction, config.steps as f64 * config.timestep);
    let d = crate::R_IN_KJ_PER_MOL * config.temperature * 100.0 / (39.948 * gamma);
    let expected = 6.0 * d * (t - (1.0 - (-gamma * t).exp()) / gamma);
    // Per-atom |Δr|² has relative spread √(2/3); 512 atoms give ≈ 4 %
    assert!(
        (msd / expected - 1.0).abs() < 0.12,
        "MSD = {msd:.2} Å², expected {expected:.2} Å²"
    );
}

/// A harmonic dimer with zero rest length: the bond vector is a 3D oscillator, so
/// ⟨r²⟩ = 3kT/k exactly, and the flexible atoms' kinetic temperature is T.
#[test]
fn flexible_dimer_equipartition() {
    use crate::group::GroupCollection as _;
    use crate::WithSimulationCell as _;
    let (n, k) = (5, 10.0);
    let config = flexible_config(200);
    let mut context = dilute_free_molecules(
        n,
        &format!(
            "{{name: free, atoms: [Ar, Ar], degrees_of_freedom: Free, excluded_neighbours: 1, \
             bonds: [{{index: [0, 1], kind: !Harmonic {{k: {k}, req: 0.0}}}}]}}"
        ),
        &[[0.0; 3], [0.5, 0.3, -0.2]],
    );
    let mut runner = LangevinRunner::new(config.clone());
    let mut rng = StdRng::seed_from_u64(31);
    // The bond relaxes in γμ/k ≈ 0.2 ps, so a 0.4 ps block decorrelates it
    for _ in 0..5 {
        runner.propagate(&mut context, &mut rng).unwrap();
    }
    runner.t_trans = Variance::new();
    let blocks = 20;
    let mut r2 = Variance::new();
    for _ in 0..blocks {
        runner.propagate(&mut context, &mut rng).unwrap();
        for g in context.groups() {
            let r = context.cell().distance(
                &context.position(g.start()),
                &context.position(g.start() + 1),
            );
            r2.add(r.norm_squared());
        }
    }
    let kt = crate::R_IN_KJ_PER_MOL * config.temperature;
    let expected = 3.0 * kt / k;
    // r² ∝ χ²₃ has relative spread √(2/3); 2500 samples give ≈ 2 %
    assert!(
        (r2.mean() / expected - 1.0).abs() < 0.08,
        "⟨r²⟩ = {:.3} Å², expected {expected:.3} Å²",
        r2.mean()
    );
    let dof = 3.0 * context.num_particles() as f64;
    assert_thermalised("kinetic", &runner.t_trans, config.temperature, dof, blocks);
}

/// A fixed-seed trajectory through every on-device kernel, recorded so a compute-backend
/// upgrade can be checked against it. Regenerate with `FAUNUS_REGENERATE=1`.
#[test]
#[ignore = "f32 rounding differs between GPU drivers; run on the recording machine around a backend upgrade"]
fn trajectory_matches_the_recorded_one() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/files/langevin_trajectory.csv");
    let (_, positions) = run_blocks(&mut trimers_and_chains(3, 18.0, 2), &ld_config(50), 41, 1);

    if std::env::var_os("FAUNUS_REGENERATE").is_some_and(|v| v == "1") {
        let csv: String = std::iter::once("x,y,z\n".to_string())
            .chain(
                positions
                    .iter()
                    .map(|p| format!("{},{},{}\n", p[0], p[1], p[2])),
            )
            .collect();
        std::fs::write(&path, csv).unwrap();
    }
    let recorded: Vec<[f32; 3]> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .skip(1)
        .map(|line| {
            let v: Vec<f32> = line.split(',').map(|x| x.parse().unwrap()).collect();
            [v[0], v[1], v[2]]
        })
        .collect();
    assert_eq!(recorded.len(), positions.len());
    for (i, (got, want)) in positions.iter().zip(&recorded).enumerate() {
        let deviation = (0..3).map(|d| (got[d] - want[d]).abs()).fold(0.0, f32::max);
        assert!(deviation < 1e-4, "atom {i} at {got:?}, recorded {want:?}");
    }
}

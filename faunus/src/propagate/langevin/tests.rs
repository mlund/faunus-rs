use super::*;

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

#[test]
fn quaternion_90_degree_rotations() {
    for (label, axis) in [
        ("x", crate::Point::x_axis()),
        ("y", crate::Point::y_axis()),
        ("z", crate::Point::z_axis()),
    ] {
        let q = crate::UnitQuaternion::from_axis_angle(&axis, std::f64::consts::FRAC_PI_2);
        let back = gpu_to_quat(&quat_to_gpu(&q));
        assert!(
            q.angle_to(&back) < 1e-6,
            "90° rotation around {label} failed roundtrip"
        );
    }
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
    let config: LangevinConfig = serde_yml::from_str(yaml).unwrap();
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
    let config: LangevinConfig = serde_yml::from_str(yaml).unwrap();
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
    let result: Result<LangevinConfig, _> = serde_yml::from_str(yaml);
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
    if let serde_yml::Value::Tagged(tagged) = &yaml {
        assert_eq!(tagged.tag.to_string(), "!LangevinDynamics");
        if let serde_yml::Value::Mapping(map) = &tagged.value {
            assert!(map.contains_key(serde_yml::Value::from("timestep")));
            assert!(map.contains_key(serde_yml::Value::from("friction")));
            assert!(map.contains_key(serde_yml::Value::from("steps")));
            assert!(map.contains_key(serde_yml::Value::from("temperature")));
            assert!(!map.contains_key(serde_yml::Value::from("measured_temperature")));
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
    if let serde_yml::Value::Tagged(tagged) = &yaml {
        if let serde_yml::Value::Mapping(map) = &tagged.value {
            assert!(
                map.contains_key(serde_yml::Value::from("measured_temperature")),
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
    let (com_vel, _, _) =
        generate_mb_velocities(kt, &mol_is_rigid, &mol_masses, &mol_inertia, &[], &[]);

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
    let (_, _, atom_vel) =
        generate_mb_velocities(kt, &[], &[], &[], &atom_is_flexible, &atom_masses);

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
    let (_, _, atom_vel) =
        generate_mb_velocities(2.494, &[], &[], &[], &[0, 0, 0], &[12.0, 14.0, 16.0]);
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

    let (_, ang_vel, _) =
        generate_mb_velocities(kt, &mol_is_rigid, &mol_masses, &mol_inertia, &[], &[]);

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
    );

    for (i, chunk) in com_vel.chunks(4).enumerate() {
        assert_eq!(chunk[3], 0.0, "com_vel[{i}].w != 0");
    }
    for (i, chunk) in ang_vel.chunks(4).enumerate() {
        assert_eq!(chunk[3], 0.0, "ang_vel[{i}].w != 0");
    }
}

/// Velocity magnitude should increase with temperature (sanity check).
#[test]
fn mb_velocities_increase_with_temperature() {
    let mol_is_rigid = vec![1u32; 1000];
    let mol_masses = vec![18.0f32; 1000];
    let mol_inertia = vec![0.0f32; 4000];

    let rms = |kt: f64| -> f64 {
        let (com_vel, _, _) =
            generate_mb_velocities(kt, &mol_is_rigid, &mol_masses, &mol_inertia, &[], &[]);
        let sum_sq: f64 = com_vel
            .chunks(4)
            .map(|c| (c[0] as f64).powi(2))
            .sum::<f64>();
        (sum_sq / 1000.0).sqrt()
    };

    let rms_cold = rms(0.5); // ~60 K
    let rms_hot = rms(5.0); // ~600 K
    assert!(
        rms_hot > rms_cold * 2.0,
        "rms_hot={rms_hot:.4} should be > 2× rms_cold={rms_cold:.4} (10× kT ratio → ~3.2× rms)"
    );
}

// ============================================================================
// GPU kernel physics tests
// ============================================================================

/// Create a wgpu client for kernel tests.
fn test_client() -> ComputeClient<cubecl::wgpu::WgpuRuntime> {
    let device = cubecl::wgpu::WgpuDevice::DefaultDevice;
    cubecl::wgpu::WgpuRuntime::client(&device)
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
        kernels::reconstruct_positions::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            &client,
            count,
            dim,
            ArrayArg::from_raw_parts::<f32>(&h_com, 4, 1),
            ArrayArg::from_raw_parts::<f32>(&h_quat, 4, 1),
            ArrayArg::from_raw_parts::<f32>(&h_ref, 12, 1),
            ArrayArg::from_raw_parts::<f32>(&h_pos, 12, 1),
            ArrayArg::from_raw_parts::<u32>(&h_offsets, 2, 1),
            ArrayArg::from_raw_parts::<u32>(&h_rigid, 1, 1),
            ScalarArg::new(n_atoms),
            ScalarArg::new(n_molecules),
        )
    }
    .unwrap();

    let result: Vec<f32> = bytemuck::cast_slice(&client.read_one(h_pos)).to_vec();

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
        kernels::reconstruct_positions::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            ArrayArg::from_raw_parts::<f32>(&h_com, 4, 1),
            ArrayArg::from_raw_parts::<f32>(&h_quat, 4, 1),
            ArrayArg::from_raw_parts::<f32>(&h_ref, 4, 1),
            ArrayArg::from_raw_parts::<f32>(&h_pos, 4, 1),
            ArrayArg::from_raw_parts::<u32>(&h_offsets, 2, 1),
            ArrayArg::from_raw_parts::<u32>(&h_rigid, 1, 1),
            ScalarArg::new(1u32),
            ScalarArg::new(1u32),
        )
    }
    .unwrap();

    let result: Vec<f32> = bytemuck::cast_slice(&client.read_one(h_pos)).to_vec();
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
        kernels::reduce_forces_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            ArrayArg::from_raw_parts::<f32>(&h_forces, 8, 1),
            ArrayArg::from_raw_parts::<f32>(&h_pos, 8, 1),
            ArrayArg::from_raw_parts::<f32>(&h_com, 4, 1),
            ArrayArg::from_raw_parts::<u32>(&h_offsets, 2, 1),
            ArrayArg::from_raw_parts::<f32>(&h_com_forces, 4, 1),
            ArrayArg::from_raw_parts::<f32>(&h_torques, 4, 1),
            ArrayArg::from_raw_parts::<u32>(&h_rigid, 1, 1),
            ScalarArg::new(1u32),
            ScalarArg::new(box_length),
            ScalarArg::new(inv_box),
        )
    }
    .unwrap();

    let cf: Vec<f32> = bytemuck::cast_slice(&client.read_one(h_com_forces)).to_vec();
    let tau: Vec<f32> = bytemuck::cast_slice(&client.read_one(h_torques)).to_vec();

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
        kernels::half_kick::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            ArrayArg::from_raw_parts::<f32>(&h_vel, 4, 1),
            ArrayArg::from_raw_parts::<f32>(&h_omega, 4, 1),
            ArrayArg::from_raw_parts::<f32>(&h_force, 4, 1),
            ArrayArg::from_raw_parts::<f32>(&h_torque, 4, 1),
            ArrayArg::from_raw_parts::<f32>(&h_quat, 4, 1),
            ArrayArg::from_raw_parts::<f32>(&h_mass, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&h_inertia, 4, 1),
            ArrayArg::from_raw_parts::<u32>(&h_rigid, 1, 1),
            ScalarArg::new(1u32),
            ScalarArg::new(dt),
        )
    }
    .unwrap();

    let result: Vec<f32> = bytemuck::cast_slice(&client.read_one(h_vel)).to_vec();

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
fn make_free_particles(n: usize, mass: f32, kt: f32) -> LangevinUploadData {
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
        kernels::half_kick_atoms::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(WORKGROUP_SIZE),
            ArrayArg::from_raw_parts::<f32>(&h_vel, 3, 1),
            ArrayArg::from_raw_parts::<f32>(&h_force, 4, 1),
            ArrayArg::from_raw_parts::<f32>(&h_mass, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&h_flex, 1, 1),
            ScalarArg::new(1u32),
            ScalarArg::new(dt),
        )
    }
    .unwrap();

    let result: Vec<f32> = bytemuck::cast_slice(&client.read_one(h_vel)).to_vec();

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
    gpu: LangevinGpu<cubecl::wgpu::WgpuRuntime>,
    n: usize,
    mass: f32,
    kt: f32,
    friction: f64,
    dt: f64,
}

impl PhysicsTestSetup {
    /// N=200 argon-like particles at 300K with standard LD parameters.
    fn new() -> Self {
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
        let data = make_free_particles(n, mass, kt);
        let mut gpu = LangevinGpu::new(client, config, n as u32, n as u32, box_length, kt);
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
    } = PhysicsTestSetup::new();
    let total_steps = 2000;

    let initial_pos = gpu.download_positions();
    gpu.run_steps_with_cpu_forces(total_steps, &mut zero_forces(n))
        .unwrap();
    let pos = gpu.download_positions();

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
    let PhysicsTestSetup { mut gpu, n, kt, .. } = PhysicsTestSetup::new();
    let k_spring = 1.0f32; // kJ/mol/Å²
    let mut force_fn = harmonic_forces(k_spring);

    gpu.run_steps_with_cpu_forces(1000, &mut force_fn).unwrap();

    // Sample in blocks to decorrelate snapshots
    let n_blocks = 10;
    let mut sum_x2 = 0.0f64;
    for _ in 0..n_blocks {
        gpu.run_steps_with_cpu_forces(500, &mut force_fn).unwrap();
        for p in &gpu.download_positions() {
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
    let PhysicsTestSetup { mut gpu, .. } = PhysicsTestSetup::new();
    // Harmonic trap prevents unbounded drift that would make KE sampling noisy
    let mut force_fn = harmonic_forces(1.0);

    gpu.run_steps_with_cpu_forces(2000, &mut force_fn).unwrap();

    let (t_trans, _) = gpu.download_temperature();
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
    } = PhysicsTestSetup::new();
    // 50 steps × 0.002 ps = 0.1 ps → exactly one friction time (1/γ)
    let steps_per_block = 50;

    gpu.run_steps_with_cpu_forces(1000, &mut zero_forces(n))
        .unwrap();

    let v0 = gpu.download_com_velocities();
    let cv0: f64 = v0
        .iter()
        .map(|v| (v[0] as f64).powi(2) + (v[1] as f64).powi(2) + (v[2] as f64).powi(2))
        .sum::<f64>()
        / n as f64;

    // Run one block and check decay at t=0.1ps (γ=10 → exp(-1) ≈ 0.368)
    gpu.run_steps_with_cpu_forces(steps_per_block, &mut zero_forces(n))
        .unwrap();
    let vt = gpu.download_com_velocities();
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

use criterion::{criterion_group, criterion_main, Criterion};
use faunus::montecarlo::MarkovChain;
use faunus::platform::reference::ReferencePlatform;
use faunus::platform::simd::SimdPlatform;
use faunus::simulation::Simulation;
use std::io::Write;
use std::path::Path;

fn yaml_config(spline: bool, n_molecules: usize) -> String {
    let xyz = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/twobody/cppm-p18.xyz");
    let xyz = xyz.to_str().unwrap();
    let spline_section = if spline {
        "\n    spline:\n      cutoff: 90.0\n      n_points: 500"
    } else {
        ""
    };
    // Scale box so density stays roughly constant relative to 20 molecules in 200^3
    let box_len = (200.0_f64.powi(3) * n_molecules as f64 / 20.0).cbrt() as usize;
    format!(
        r#"
atoms:
  - name: PP
    mass: 1.0
    charge: 1.0
    sigma: 4.0
    epsilon: 0.5
  - name: NP
    mass: 1.0
    charge: -1.0
    sigma: 4.0
    epsilon: 0.5
  - name: MP
    mass: 1e9
    charge: 0.0
    sigma: 40.0
    epsilon: 0.5

molecules:
  - name: MOL
    degrees_of_freedom: Rigid
    has_com: true
    from_structure: "{xyz}"

system:
  cell: !Cuboid [{box_len}, {box_len}, {box_len}]
  medium:
    permittivity: !Water
    temperature: 298.15
    salt: [!NaCl, 0.005]
  energy:
    nonbonded:
      default:
        - !Coulomb {{cutoff: 90}}
        - !WeeksChandlerAndersen {{mixing: LB}}{spline_section}
  blocks:
    - molecule: MOL
      N: {n_molecules}
      insert:
        !RandomCOM {{
          filename: "{xyz}",
          rotate: true,
        }}

analysis: []

propagate:
  seed: !Fixed 42
  criterion: Metropolis
  repeat: 1000000
  collections:
    - !Stochastic
      moves:
        - !TranslateMolecule {{molecule: MOL, dp: 40.0, weight: 1.0}}
        - !RotateMolecule {{molecule: MOL, dp: 1.0, weight: 1.0}}
"#
    )
}

fn write_tmp_yaml(yaml: &str) -> tempfile::NamedTempFile {
    let mut tmp = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
    tmp.write_all(yaml.as_bytes()).unwrap();
    tmp
}

fn build_mc_reference(spline: bool, n_molecules: usize) -> MarkovChain<ReferencePlatform> {
    let yaml = yaml_config(spline, n_molecules);
    let tmp = write_tmp_yaml(&yaml);
    let (sim, _medium) = Simulation::from_file(tmp.path(), None).unwrap();
    match sim {
        Simulation::SingleBox(mc) => mc,
        _ => unreachable!(),
    }
}

fn build_mc_simd(n_molecules: usize) -> MarkovChain<SimdPlatform> {
    let yaml = yaml_config(true, n_molecules);
    let tmp = write_tmp_yaml(&yaml);

    let medium = faunus::platform::reference::get_medium(tmp.path()).unwrap();
    let kt = faunus::simulation::thermal_energy(&medium);
    let context = SimdPlatform::new(tmp.path(), None, &mut rand::thread_rng()).unwrap();
    let propagate = faunus::propagate::Propagate::from_file(tmp.path(), &context).unwrap();
    let analyses = faunus::analysis::from_file(tmp.path(), &context).unwrap();
    MarkovChain::new(context, propagate, kt, analyses).unwrap()
}

/// Single MC move benchmarks for 20 CPPM molecules
fn bench_mc_sweep(c: &mut Criterion) {
    let mut mc_ref_nospline = build_mc_reference(false, 20);
    let mut mc_ref_spline = build_mc_reference(true, 20);
    let mut mc_simd = build_mc_simd(20);

    let mut group = c.benchmark_group("mc_sweep_20cppm");
    group.bench_function("reference", |b| {
        b.iter(|| mc_ref_nospline.run_n_steps(1).unwrap());
    });
    group.bench_function("reference_splined", |b| {
        b.iter(|| mc_ref_spline.run_n_steps(1).unwrap());
    });
    group.bench_function("simd_soa", |b| {
        b.iter(|| mc_simd.run_n_steps(1).unwrap());
    });
    group.finish();
}

/// Multi-step MC benchmarks (20 moves per iteration).
/// With a group energy cache, old_energy lookups become O(1) after the first
/// step, so the per-move cost should drop by ~50% for nonbonded-dominated systems.
fn bench_mc_multistep(c: &mut Criterion) {
    let mut mc_simd_20 = build_mc_simd(20);
    let mut mc_ref_20 = build_mc_reference(true, 20);

    let mut group = c.benchmark_group("mc_multistep_20cppm");
    group.bench_function("reference_splined_20steps", |b| {
        b.iter(|| mc_ref_20.run_n_steps(20).unwrap());
    });
    group.bench_function("simd_soa_20steps", |b| {
        b.iter(|| mc_simd_20.run_n_steps(20).unwrap());
    });
    group.finish();
}

/// Scaling benchmark: 80 molecules to measure how performance scales with
/// system size. The group cache saves O(N_groups) work per step, so the
/// absolute time saved grows linearly with molecule count.
fn bench_mc_scaling(c: &mut Criterion) {
    let mut mc_simd_80 = build_mc_simd(80);

    let mut group = c.benchmark_group("mc_sweep_80cppm");
    group.bench_function("simd_soa", |b| {
        b.iter(|| mc_simd_80.run_n_steps(1).unwrap());
    });
    group.bench_function("simd_soa_20steps", |b| {
        b.iter(|| mc_simd_80.run_n_steps(20).unwrap());
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_mc_sweep,
    bench_mc_multistep,
    bench_mc_scaling
);
criterion_main!(benches);

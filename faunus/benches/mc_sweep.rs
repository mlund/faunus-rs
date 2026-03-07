use criterion::{criterion_group, criterion_main, Criterion};
use faunus::montecarlo::MarkovChain;
use faunus::platform::soa::SoaPlatform;
use std::io::Write;
use std::path::Path;

fn yaml_config(n_molecules: usize) -> String {
    let xyz = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/twobody/cppm-p18.xyz");
    let xyz = xyz.to_str().unwrap();
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
        - !WeeksChandlerAndersen {{mixing: LB}}
    spline:
      cutoff: 90.0
      n_points: 500
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

fn build_mc(n_molecules: usize) -> MarkovChain<SoaPlatform> {
    let yaml = yaml_config(n_molecules);
    let tmp = write_tmp_yaml(&yaml);
    let context = SoaPlatform::new(tmp.path(), None, &mut rand::thread_rng()).unwrap();
    let medium = faunus::platform::get_medium(tmp.path()).unwrap();
    let kt = faunus::simulation::thermal_energy(&medium);
    faunus::simulation::build_markov_chain(tmp.path(), context, kt, None).unwrap()
}

/// Single MC move benchmarks for 20 CPPM molecules
fn bench_mc_sweep(c: &mut Criterion) {
    let mut mc = build_mc(20);

    let mut group = c.benchmark_group("mc_sweep_20cppm");
    group.bench_function("soa", |b| {
        b.iter(|| mc.run_n_steps(1).unwrap());
    });
    group.finish();
}

/// Multi-step MC benchmarks (20 moves per iteration).
fn bench_mc_multistep(c: &mut Criterion) {
    let mut mc = build_mc(20);

    let mut group = c.benchmark_group("mc_multistep_20cppm");
    group.bench_function("soa_20steps", |b| {
        b.iter(|| mc.run_n_steps(20).unwrap());
    });
    group.finish();
}

/// Scaling benchmark: 80 molecules to measure how performance scales with
/// system size.
fn bench_mc_scaling(c: &mut Criterion) {
    let mut mc_soa_80 = build_mc(80);

    let mut group = c.benchmark_group("mc_sweep_80cppm");
    group.bench_function("soa", |b| {
        b.iter(|| mc_soa_80.run_n_steps(1).unwrap());
    });
    group.bench_function("soa_20steps", |b| {
        b.iter(|| mc_soa_80.run_n_steps(20).unwrap());
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

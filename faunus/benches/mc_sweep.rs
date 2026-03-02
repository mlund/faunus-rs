use criterion::{criterion_group, criterion_main, Criterion};
use faunus::montecarlo::MarkovChain;
use faunus::platform::reference::ReferencePlatform;
use faunus::platform::simd::SimdPlatform;
use faunus::simulation::Simulation;
use std::io::Write;
use std::path::Path;

fn yaml_config() -> String {
    let xyz = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/twobody/cppm-p18.xyz");
    let xyz = xyz.to_str().unwrap();
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
  cell: !Cuboid [200, 200, 200]
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
      N: 20
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
        - !TranslateMolecule {{molecule: MOL, dp: 2.0, weight: 1.0}}
        - !RotateMolecule {{molecule: MOL, dp: 1.0, weight: 1.0}}
"#
    )
}

fn write_tmp_yaml(yaml: &str) -> tempfile::NamedTempFile {
    let mut tmp = tempfile::Builder::new()
        .suffix(".yaml")
        .tempfile()
        .unwrap();
    tmp.write_all(yaml.as_bytes()).unwrap();
    tmp
}

fn build_mc_reference() -> MarkovChain<ReferencePlatform> {
    let yaml = yaml_config();
    let tmp = write_tmp_yaml(&yaml);
    let (sim, _medium) = Simulation::from_file(tmp.path(), None).unwrap();
    match sim {
        Simulation::SingleBox(mc) => mc,
        _ => unreachable!(),
    }
}

fn build_mc_simd() -> MarkovChain<SimdPlatform> {
    let yaml = yaml_config();
    let tmp = write_tmp_yaml(&yaml);

    let medium = faunus::platform::reference::get_medium(tmp.path()).unwrap();
    let kt = faunus::simulation::thermal_energy(&medium);
    let context = SimdPlatform::new(tmp.path(), None, &mut rand::thread_rng()).unwrap();
    let propagate = faunus::propagate::Propagate::from_file(tmp.path(), &context).unwrap();
    let analyses = faunus::analysis::from_file(tmp.path(), &context).unwrap();
    MarkovChain::new(context, propagate, kt, analyses).unwrap()
}

fn bench_mc_sweep(c: &mut Criterion) {
    let mut mc_ref = build_mc_reference();
    let mut mc_simd = build_mc_simd();

    let mut group = c.benchmark_group("mc_sweep_20cppm");
    group.bench_function("reference", |b| {
        b.iter(|| mc_ref.run_n_steps(1).unwrap());
    });
    group.bench_function("simd_soa", |b| {
        b.iter(|| mc_simd.run_n_steps(1).unwrap());
    });
    group.finish();
}

criterion_group!(benches, bench_mc_sweep);
criterion_main!(benches);

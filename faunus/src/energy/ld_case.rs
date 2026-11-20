//! Test fixture for the Langevin-dynamics force checks.

/// A system from the YAML `atoms:` … `system:` body, with an inert `propagate:` section.
pub(crate) fn ld_system(body: &str) -> crate::backend::Backend {
    use rand::SeedableRng;
    let yaml = format!(
        "{body}\npropagate: {{seed: !Fixed 1, criterion: Metropolis, steps: 0, collections: []}}\n"
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    crate::backend::Backend::from_yaml_str(&yaml, None, &mut rng).unwrap()
}

/// A charged dimer (bonded, excluded pair kept), a rigid probe ion, and an
/// uninserted flexible kind whose terms must not count.
pub(crate) struct LdCase {
    pub dimer_dof: &'static str,
    pub bond: &'static str,
    pub spline: bool,
    pub cell: &'static str,
    pub extra: &'static str,
}

impl Default for LdCase {
    fn default() -> Self {
        Self {
            dimer_dof: "Rigid",
            bond: "!Harmonic {k: 1.0, req: 3.0}",
            spline: true,
            cell: "!Cuboid [40.0, 40.0, 40.0]",
            extra: "",
        }
    }
}

impl LdCase {
    pub fn context(&self) -> crate::backend::Backend {
        let Self {
            dimer_dof,
            bond,
            spline,
            cell,
            extra,
        } = self;
        let spline = if *spline {
            "spline: {cutoff: 15.0}"
        } else {
            ""
        };
        let yaml = format!(
            r#"
atoms:
  - {{name: A, charge: 1.0, mass: 1.0, sigma: 2.0}}
molecules:
  - name: dimer
    atoms: [A, A]
    degrees_of_freedom: {dimer_dof}
    keep_excluded_coulomb: true
    excluded_neighbours: 1
    bonds:
      - {{index: [0, 1], kind: {bond}}}
  - {{name: probe, atoms: [A], degrees_of_freedom: Rigid}}
  - name: spare
    atoms: [A, A]
    degrees_of_freedom: Free
    keep_excluded_coulomb: true
    excluded_neighbours: 1
    bonds:
      - {{index: [0, 1], kind: !FENE {{k: 1.0, req: 3.0, rmax: 5.0}}}}
system:
  medium: {{permittivity: !Vacuum, temperature: 298.15}}
  cell: {cell}
  blocks:
    - {{molecule: dimer, N: 1, insert: !Manual [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]}}
    - {{molecule: probe, N: 1, insert: !Manual [[10.0, 0.0, 0.0]]}}
  energy:
    - !Nonbonded
      default:
        - !CoulombPlain {{cutoff: 15.0}}
      {spline}
{extra}
"#
        );
        ld_system(&yaml)
    }
}

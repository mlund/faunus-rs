//! Coulomb correction for excluded (bonded) pairs.
//!
//! When the combined nonbonded potential (SR + Coulomb) is splined as one entity,
//! excluded pairs skip everything. This term adds back exact Coulomb for
//! molecules that opt in via `keep_excluded_coulomb: true`, enabling charge
//! titration and alchemical moves on molecules with exclusions.

use interatomic::twobody::{ArcPotential, IsotropicTwobodyEnergy};
use ndarray::Array2;

use crate::{group::Group, topology::Topology, Change, Context};

use super::{builder::PairPotentialBuilder, EnergyChange, EnergyTerm};

/// Evaluates analytical Coulomb for excluded pairs in opted-in molecules.
///
/// Uses exact (non-splined) Coulomb so that energy changes from charge
/// mutations are computed without spline approximation error cancellation
/// issues — the main nonbonded term contributes zero for these pairs.
#[derive(Debug, Clone)]
pub struct ExcludedCoulomb {
    /// Coulomb-only potentials indexed by atom-kind pair.
    potentials: Array2<ArcPotential>,
}

impl ExcludedCoulomb {
    /// Build from a pair potential builder, extracting only Coulomb interactions.
    pub(crate) fn new(
        pairpot_builder: &PairPotentialBuilder,
        topology: &Topology,
        medium: Option<interatomic::coulomb::Medium>,
        combine_with_default: bool,
    ) -> anyhow::Result<Self> {
        let atoms = topology.atomkinds();
        let n = atoms.len();
        let mut potentials = Array2::from_elem(
            (n, n),
            ArcPotential::new(interatomic::twobody::NoInteraction),
        );

        for i in 0..n {
            for j in i..n {
                if let Some(pot) = pairpot_builder.get_coulomb_interaction(
                    &atoms[i],
                    &atoms[j],
                    medium.clone(),
                    combine_with_default,
                )? {
                    let arc = ArcPotential(pot.into());
                    potentials[(i, j)] = arc.clone();
                    potentials[(j, i)] = arc;
                }
            }
        }

        Ok(Self { potentials })
    }

    /// Coulomb energy for excluded pairs in a single group.
    fn one_group(&self, context: &impl Context, group: &Group) -> f64 {
        let topology = context.topology_ref();
        let molecule = &topology.moleculekinds()[group.molecule()];

        if !molecule.keep_excluded_coulomb() || molecule.exclusions().is_empty() {
            return 0.0;
        }

        molecule
            .exclusions()
            .iter()
            .map(|pair| {
                let (i, j) = pair.into_ordered_tuple();
                let abs_i = group.start() + i;
                let abs_j = group.start() + j;
                let dist_sq = context.get_distance_squared(abs_i, abs_j);
                let kind_i = context.atom_kind(abs_i);
                let kind_j = context.atom_kind(abs_j);
                self.potentials[(kind_i, kind_j)].isotropic_twobody_energy(dist_sq)
            })
            .sum()
    }
}

impl EnergyChange for ExcludedCoulomb {
    fn energy(&self, context: &impl Context, change: &Change) -> f64 {
        match change {
            Change::Everything | Change::Volume(_, _) => context
                .groups()
                .iter()
                .map(|g| self.one_group(context, g))
                .sum(),
            Change::None => 0.0,
            Change::SingleGroup(id, gc) if gc.internal_change() => {
                self.one_group(context, &context.groups()[*id])
            }
            Change::SingleGroup(..) => 0.0,
            Change::Groups(groups) => groups
                .iter()
                .filter(|(_, gc)| gc.internal_change())
                .map(|(id, _)| self.one_group(context, &context.groups()[*id]))
                .sum(),
        }
    }
}

impl From<ExcludedCoulomb> for EnergyTerm {
    fn from(term: ExcludedCoulomb) -> Self {
        Self::ExcludedCoulomb(term)
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::Backend;
    use crate::context::WithHamiltonian;
    use crate::Change;
    use float_cmp::assert_approx_eq;
    use interatomic::twobody::IsotropicTwobodyEnergy;

    /// Build a system with two bonded charged particles and an exclusion.
    fn make_dimer_system(keep_excluded_coulomb: bool) -> Backend {
        let distance = 10.0;
        let yaml = format!(
            r#"
atoms:
  - {{name: A, charge: 1.0, mass: 1.0}}

molecules:
  - name: dimer
    atoms: [A, A]
    keep_excluded_coulomb: {keep_excluded_coulomb}
    excluded_neighbours: 1
    bonds:
      - {{index: [0, 1], kind: !Harmonic {{k: 1.0, req: {distance}}}}}

system:
  medium:
    permittivity: !Vacuum
    temperature: 298.15
  geometry: {{type: cuboid, length: 100.0}}
  blocks:
    - molecule: dimer
      N: 1
      insert: !Manual
        - [0.0, 0.0, 0.0]
        - [{distance}, 0.0, 0.0]
  energy:
    nonbonded:
      default:
        - !CoulombPlain {{cutoff: 50.0}}
"#
        );

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("input.yaml");
        std::fs::write(&file, yaml).unwrap();

        let mut rng = rand::thread_rng();
        Backend::new(&file, None, &mut rng).unwrap()
    }

    /// Two bonded charged particles with an exclusion — verify correction equals analytical Coulomb.
    #[test]
    fn excluded_coulomb_correction() {
        let system = make_dimer_system(true);
        let distance = 10.0;

        let medium = interatomic::coulomb::Medium::new(
            298.15,
            interatomic::coulomb::permittivity::Permittivity::Vacuum,
            None,
        );

        // Expected: analytical Coulomb at this distance
        let ionion = interatomic::twobody::IonIon::new(
            1.0,
            medium.into(),
            interatomic::coulomb::pairwise::Plain::default(),
        );
        let expected = ionion.isotropic_twobody_energy(distance * distance);

        let per_term = system
            .hamiltonian()
            .per_term_energies(&system, &Change::Everything);
        let excl_energy = per_term
            .iter()
            .find(|(name, _)| *name == "excluded_coulomb")
            .map(|(_, e)| *e)
            .unwrap();

        assert_approx_eq!(f64, excl_energy, expected, epsilon = 1e-10);
    }

    /// Default `keep_excluded_coulomb: false` produces zero correction.
    #[test]
    fn excluded_coulomb_default_off() {
        let system = make_dimer_system(false);

        let per_term = system
            .hamiltonian()
            .per_term_energies(&system, &Change::Everything);

        // Term should not be present when keep_excluded_coulomb is false
        let excl = per_term
            .iter()
            .find(|(name, _)| *name == "excluded_coulomb");
        assert!(excl.is_none());
    }
}

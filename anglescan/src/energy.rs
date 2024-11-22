use coulomb::pairwise::MultipolePotential;
use faunus::energy::NonbondedMatrix;
use faunus::topology::AtomKind;
use interatomic::twobody::{IonIon, IsotropicTwobodyEnergy};
use interatomic::Vector3;

use crate::structure::Structure;

// type alias for the pair potential
type CoulombMethod = coulomb::pairwise::Plain;
type CoulombPotential = IonIon<CoulombMethod>;

/// Pair-matrix of twobody energies for pairs of atom ids
pub struct PairMatrix {
    nonbonded: NonbondedMatrix,
}

impl PairMatrix {
    /// Create a new pair matrix with added Coulomb potential
    pub fn new_append_ionion(
        mut nonbonded: NonbondedMatrix,
        atomkinds: &[AtomKind],
        coulomb_method: CoulombMethod,
    ) -> Self {
        nonbonded
            .get_potentials_mut()
            .indexed_iter_mut()
            .for_each(|((i, j), pairpot)| {
                let charge_product = atomkinds[i].charge() * atomkinds[j].charge();
                let coulomb = Box::new(CoulombPotential::new(
                    charge_product,
                    coulomb_method.clone(),
                )) as Box<dyn IsotropicTwobodyEnergy>;
                let combined = coulomb + Box::new(pairpot.clone());
                *pairpot = std::sync::Arc::new(combined);
            });
        Self { nonbonded }
    }

    // Sum energy between two set of atomic structures (kJ/mol)
    pub fn sum_energy(&self, a: &Structure, b: &Structure) -> f64 {
        let potentials = self.nonbonded.get_potentials();
        let mut energy = 0.0;
        for i in 0..a.pos.len() {
            for j in 0..b.pos.len() {
                let distance_sq = (a.pos[i] - b.pos[j]).norm_squared();
                let a = a.atom_ids[i];
                let b = b.atom_ids[j];
                energy += potentials[(a, b)].isotropic_twobody_energy(distance_sq);
            }
        }
        trace!("molecule-molecule energy: {:.2} kJ/mol", energy);
        energy
    }
}

/// Calculate accumulated electric potential at point `r` due to charges in `structure`
pub fn electric_potential(structure: &Structure, r: &Vector3, multipole: &CoulombMethod) -> f64 {
    std::iter::zip(structure.pos.iter(), structure.charges.iter())
        .map(|(pos, charge)| multipole.ion_potential(*charge, (pos - r).norm()))
        .sum()
}

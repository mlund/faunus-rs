use faunus::topology::AtomKind;
use interact::multipole::Yukawa;
use interact::twobody::{IonIon, IsotropicTwobodyEnergy, LennardJones, YukawaLennardJones};
use interact::CombinationRule;

use crate::structure::Structure;

/// Pair-matrix of twobody energies for pairs of atom ids
pub struct PairMatrix<'a> {
    /// Matrix of twobody energy terms
    pub matrix: Vec<Vec<YukawaLennardJones<'a>>>,
}

impl PairMatrix<'_> {
    /// Create a new pair matrix
    pub fn new(atomkinds: &[AtomKind], multipole: &'static Yukawa) -> Self {
        let default =
            YukawaLennardJones::new(IonIon::new(0.0, multipole), LennardJones::new(0.0, 0.0));
        let n = atomkinds.len();
        let mut matrix = vec![vec![default; n]; n];
        for i in 0..n {
            for j in 0..n {
                let a = &atomkinds[i];
                let b = &atomkinds[j];
                let charge_product = a.charge * b.charge;
                let epsilons = (a.epsilon.unwrap_or_default(), b.epsilon.unwrap_or_default());
                let sigmas = (a.sigma.unwrap_or_default(), b.sigma.unwrap_or_default());
                let ionion = IonIon::new(charge_product, multipole);
                let lj = LennardJones::from_combination_rule(
                    CombinationRule::LorentzBerthelot,
                    epsilons,
                    sigmas,
                );
                matrix[i][j] = YukawaLennardJones::new(ionion, lj);
            }
        }
        Self { matrix }
    }

    // Sum energy between two set of atomic structures
    pub fn sum_energy(&self, a: &Structure, b: &Structure) -> f64 {
        let mut energy = 0.0;
        for i in 0..a.positions.len() {
            for j in 0..b.positions.len() {
                let distance_sq = (a.positions[i] - b.positions[j]).norm_squared();
                let id_a = a.atom_ids[i];
                let id_b = b.atom_ids[j];
                energy += self.matrix[id_a][id_b].isotropic_twobody_energy(distance_sq);
            }
        }
        energy
    }
}

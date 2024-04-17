use faunus::topology::AtomKind;
use interatomic::twobody::{IonIon, IsotropicTwobodyEnergy, LennardJones};
use interatomic::CombinationRule;

use crate::structure::Structure;

// type alias for the pair potential
type CoulombMethod = interatomic::multipole::Coulomb;
type ShortRange = interatomic::twobody::WeeksChandlerAndersen;
type PairPotential<'a> = interatomic::twobody::Combined<IonIon<'a, CoulombMethod>, ShortRange>;

/// Pair-matrix of twobody energies for pairs of atom ids
pub struct PairMatrix<'a> {
    /// Matrix of twobody energy terms
    pub matrix: Vec<Vec<PairPotential<'a>>>,
}

impl<'a> PairMatrix<'a> {
    /// Create a new pair matrix
    pub fn new(atomkinds: &[AtomKind], multipole: &'a CoulombMethod) -> Self {
        let lj_default = LennardJones::new(0.0, 0.0);
        let default = PairPotential::new(IonIon::new(0.0, multipole), ShortRange::new(lj_default));
        let n = atomkinds.len();
        let mut matrix = vec![vec![default; n]; n];
        for i in 0..n {
            for j in 0..n {
                let a = &atomkinds[i];
                let b = &atomkinds[j];

                let ionion = IonIon::new(a.charge * b.charge, multipole);
                let epsilons = (a.epsilon.unwrap_or(0.0), b.epsilon.unwrap_or(0.0));
                let sigmas = (a.sigma.unwrap_or(0.0), b.sigma.unwrap_or(0.0));
                let lj = ShortRange::from_combination_rule(
                    CombinationRule::LorentzBerthelot,
                    epsilons,
                    sigmas,
                );
                matrix[i][j] = PairPotential::new(ionion, lj);
            }
        }
        Self { matrix }
    }

    // Sum energy between two set of atomic structures (kJ/mol)
    pub fn sum_energy(&self, a: &Structure, b: &Structure) -> f64 {
        let mut energy = 0.0;
        for i in 0..a.pos.len() {
            for j in 0..b.pos.len() {
                let distance_sq = (a.pos[i] - b.pos[j]).norm_squared();
                let id_a = a.atom_ids[i];
                let id_b = b.atom_ids[j];
                energy += self.matrix[id_a][id_b].isotropic_twobody_energy(distance_sq);
            }
        }
        trace!("molecule-molecule energy: {:.2} kJ/mol", energy);
        energy
    }
}

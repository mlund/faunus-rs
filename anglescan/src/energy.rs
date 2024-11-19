use coulomb::pairwise::MultipolePotential;
use faunus::topology::AtomKind;
use interatomic::twobody::{Combined, IonIon, IsotropicTwobodyEnergy, WeeksChandlerAndersen};
use interatomic::{CombinationRule, Vector3};

use crate::structure::Structure;

// type alias for the pair potential
type CoulombMethod = coulomb::pairwise::Plain;
type CoulombPotential = IonIon<CoulombMethod>;
type ShortRange = WeeksChandlerAndersen;
type PairPotential = Combined<CoulombPotential, ShortRange>;

/// Pair-matrix of twobody energies for pairs of atom ids
pub struct PairMatrix {
    /// Matrix of twobody energy terms
    pub matrix: Vec<Vec<PairPotential>>,
}

impl PairMatrix {
    /// Create a new pair matrix
    pub fn new(atomkinds: &[AtomKind], multipole: &CoulombMethod) -> Self {
        let default = PairPotential::new(
            CoulombPotential::new(0.0, multipole.clone()),
            ShortRange::new(0.0, 0.0),
        );
        let n = atomkinds.len();
        let mut matrix = vec![vec![default; n]; n];
        for i in 0..n {
            for j in 0..n {
                let a = &atomkinds[i];
                let b = &atomkinds[j];

                let ionion = IonIon::new(a.charge() * b.charge(), multipole.clone());
                let mixed = AtomKind::combine(CombinationRule::LorentzBerthelot, a, b);
                let lj = ShortRange::new(mixed.epsilon().unwrap(), mixed.sigma().unwrap());
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

/// Calculate accumulated electric potential at point `r` due to charges in `structure`
pub fn electric_potential(structure: &Structure, r: &Vector3, multipole: &CoulombMethod) -> f64 {
    std::iter::zip(structure.pos.iter(), structure.charges.iter())
        .map(|(pos, charge)| multipole.ion_potential(*charge, (pos - r).norm()))
        .sum()
}

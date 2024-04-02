use crate::structure::AtomKinds;
use faunus::topology::AtomKind;
use interact::multipole::{self, MultipoleEnergy, Yukawa};
use interact::twobody::{IonIonYukawa, IsotropicTwobodyEnergy, LennardJones, YukawaLennardJones};
use interact::CombinationRule;
use serde::de;
use std::rc::Rc;

/// Pair-matrix of twobody energies for pairs of atom ids
struct PairMatrix {
    /// Matrix of twobody energies
    pub matrix: Vec<Vec<YukawaLennardJones>>,
    pub multipole: Yukawa,
}

impl PairMatrix {
    // fn build(&mut self, atomkinds: &[AtomKind]) {
    //     let n = atomkinds.len();
    //     // let multipole = Rc::new(Yukawa::new(0.0, Some(0.0)));
    //     let default = YukawaLennardJones::new(
    //         IonIonYukawa::new(0.0, &self.multipole),
    //         LennardJones::new(0.0, 0.0),
    //     );
    //     self.matrix = vec![vec![default; n]; n];
    //     for i in 0..n {
    //         for j in 0..n {
    //             let qq = atomkinds[i].charge * atomkinds[j].charge;
    //             let epsilons = (
    //                 atomkinds[i].epsilon.unwrap_or_default(),
    //                 atomkinds[j].epsilon.unwrap_or_default(),
    //             );
    //             let sigmas = (
    //                 atomkinds[i].sigma.unwrap_or_default(),
    //                 atomkinds[j].sigma.unwrap_or_default(),
    //             );
    //             let lj = LennardJones::from_combination_rule(
    //                 CombinationRule::LorentzBerthelot,
    //                 epsilons,
    //                 sigmas,
    //             );
    //             let ionion = IonIonYukawa::new(qq, &self.multipole);
    //             self.matrix[i][j] = YukawaLennardJones::new(ionion, lj);
    //         }
    //     }
    // }

    /// Create a new pair matrix
    fn new(cutoff: f64, debye_length: f64, atomkinds: &[AtomKind]) -> Self {
        // let multipole: &'static Yukawa = Yukawa::new(cutoff, Some(debye_length));
        let n = atomkinds.len();

        let multipole = Yukawa::new(cutoff, Some(debye_length));
        Self {
            matrix: Vec::default(),
            multipole: multipole.into(),
        }

        // let default = YukawaLennardJones::new(
        //     IonIonYukawa::new(0.0, &result.multipole),
        //     LennardJones::new(0.0, 0.0),
        // );
        // result.matrix = vec![vec![default; n]; n];
        // for i in 0..n {
        //     for j in 0..n {
        //         let qq = atomkinds[i].charge * atomkinds[j].charge;
        //         let epsilons = (
        //             atomkinds[i].epsilon.unwrap_or_default(),
        //             atomkinds[j].epsilon.unwrap_or_default(),
        //         );
        //         let sigmas = (
        //             atomkinds[i].sigma.unwrap_or_default(),
        //             atomkinds[j].sigma.unwrap_or_default(),
        //         );
        //         let lj = LennardJones::from_combination_rule(
        //             CombinationRule::LorentzBerthelot,
        //             epsilons,
        //             sigmas,
        //         );
        //         let ionion = IonIonYukawa::new(qq, &result.multipole);
        //         result.matrix[i][j] = YukawaLennardJones::new(ionion, lj);
        //     }
        // }
        // result
    }
}

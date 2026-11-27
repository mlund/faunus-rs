// Copyright 2025 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

//! Self energy of each charge under a truncated pair Coulomb scheme.
//!
//! A truncated scheme such as Fanourgakis gives every charge a constant interaction with its
//! own neutralising cloud, u_self = c₁ z² l_B kT / r_c. It is constant at fixed charges but
//! biases every move that changes Σz²: grand-canonical insertion, titration and molecule swaps.

use super::builder::{PairInteraction, PairPotentialBuilder};
use super::EnergyChange;
use crate::change::{Change, Dependence};
use crate::context::ObserveContext;
use crate::topology::AtomKind;
use anyhow::Context;
use interatomic::coulomb::{Medium, Temperature};
use interatomic::Cutoff;

/// Per-kind self energy of the pair Coulomb scheme each kind interacts with itself through.
#[derive(Clone, Debug)]
pub struct CoulombSelfEnergy {
    /// kJ/mol, indexed by atom kind.
    per_kind: Vec<f64>,
}

impl CoulombSelfEnergy {
    /// The self term of the user's nonbonded pair schemes, or `None` when no kind has one.
    ///
    /// `nonbonded` must be the builder *before* the real-space Ewald injection: reciprocal
    /// Ewald adds its own self term, so counting the injected scheme too would double it.
    ///
    /// # Errors
    /// A charged kind with Coulomb but no medium, or with a composite Coulomb potential.
    pub(crate) fn new(
        nonbonded: Option<&PairPotentialBuilder>,
        atomkinds: &[AtomKind],
        medium: Option<&Medium>,
    ) -> anyhow::Result<Option<Self>> {
        let Some(nonbonded) = nonbonded else {
            return Ok(None);
        };
        let kind_energy = |kind: &AtomKind| -> anyhow::Result<f64> {
            let z = kind.charge();
            let schemes = nonbonded.resolve_interactions(kind, kind, PairInteraction::is_coulomb);
            if z == 0.0 || schemes.is_empty() {
                return Ok(0.0);
            }
            let medium = medium.context("Coulomb interactions need a medium")?;
            let range = nonbonded.pair_range();
            schemes
                .into_iter()
                .map(|scheme| {
                    let coulomb = scheme.to_coulomb(kind, kind, medium.clone())?;
                    let self_energy = coulomb.self_energy(z)?;
                    // A shorter range truncates the pair again, so the scheme's c₁ no longer applies
                    if self_energy != 0.0 && range < coulomb.cutoff() {
                        anyhow::bail!(
                            "the self energy assumes the scheme's {} Å cutoff, but the pair sum \
                             stops at {range} Å (spline or nonbonded cutoff)",
                            coulomb.cutoff()
                        );
                    }
                    Ok(self_energy)
                })
                .sum::<anyhow::Result<f64>>()
                .with_context(|| format!("Coulomb self energy of atom kind {}", kind.name()))
        };
        let per_kind = atomkinds
            .iter()
            .map(kind_energy)
            .collect::<anyhow::Result<Vec<_>>>()?;
        if per_kind.iter().all(|&u| u == 0.0) {
            return Ok(None);
        }
        if let Err(error) = nonbonded.coulomb_scheme(atomkinds, medium) {
            log::warn!(
                "Coulomb self energy uses each kind's own-pair scheme, which may not be the one \
                 it interacts with others through: {error}"
            );
        }
        if let Some(medium) = medium {
            let thermal_energy = crate::R_IN_KJ_PER_MOL * medium.temperature();
            let nonzero: Vec<_> = atomkinds
                .iter()
                .zip(&per_kind)
                .filter(|(_, &u)| u != 0.0)
                .map(|(kind, u)| format!("{} {:.4}", kind.name(), u / thermal_energy))
                .collect();
            log::info!("Coulomb self energy (kT): {}", nonzero.join(", "));
        }
        Ok(Some(Self { per_kind }))
    }
}

impl EnergyChange for CoulombSelfEnergy {
    fn energy(&self, context: &impl ObserveContext, change: &Change) -> f64 {
        change.sum_over_touched(context, Dependence::IdentityOnly, |i| {
            self.per_kind[context.atom_kind(i).get()]
        })
    }
}

/// Energy (kJ/mol) of the self term alone for `change`; 0 when the term is absent.
#[cfg(test)]
pub(crate) fn self_term_energy(context: &crate::backend::Backend, change: &Change) -> f64 {
    use crate::WithHamiltonian;
    let hamiltonian = context.hamiltonian();
    hamiltonian
        .energy_terms()
        .iter()
        .filter(|term| matches!(term, super::EnergyTerm::CoulombSelf(_)))
        .map(|term| term.energy(context, change))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::self_term_energy;
    use crate::backend::Backend;
    use crate::change::GroupChange;
    use crate::energy::EnergyChange;
    use crate::group::{AtomKindId, GroupCollectionMut, RelIndex};
    use crate::{Change, WithHamiltonian};
    use approx::assert_relative_eq;

    /// A neutral atom, a cation and an anion, each its own group, at `positions`.
    fn ions(side: f64, positions: [[f64; 3]; 3], medium: &str, energy: &str) -> Backend {
        let [n, p, m] = positions.map(|r| format!("{r:?}"));
        let yaml = format!(
            r#"
atoms:
  - {{name: N, mass: 1.0, charge: 0.0, sigma: 1.0}}
  - {{name: P, mass: 1.0, charge: 1.0, sigma: 1.0}}
  - {{name: M, mass: 1.0, charge: -1.0, sigma: 1.0}}
molecules:
  - {{name: n, atoms: [N]}}
  - {{name: p, atoms: [P]}}
  - {{name: m, atoms: [M]}}
system:
  cell: !Cuboid [{side:?}, {side:?}, {side:?}]
  medium: {{permittivity: !Fixed 80.0, temperature: 300.0{medium}}}
  energy:
{energy}
  blocks:
    - {{molecule: n, N: 1, insert: !Manual [{n}]}}
    - {{molecule: p, N: 1, insert: !Manual [{p}]}}
    - {{molecule: m, N: 1, insert: !Manual [{m}]}}
propagate: {{seed: !Fixed 1, criterion: Metropolis, steps: 0, collections: []}}
"#
        );
        Backend::from_yaml_str(&yaml, None, &mut rand::thread_rng()).unwrap()
    }

    /// The ions sit beyond an 80 Å cutoff of one another, so only self terms remain.
    fn isolated(medium: &str, energy: &str) -> Backend {
        ions(
            400.0,
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, -100.0, 0.0]],
            medium,
            energy,
        )
    }

    /// Energy change (kJ/mol) of retyping particle 0 (the neutral `N`) to `kind`.
    fn swap_energy(context: &mut Backend, kind: usize) -> f64 {
        let change = Change::SingleGroup(0, GroupChange::UpdateIdentity(vec![RelIndex::new(0)]));
        let before = context.hamiltonian().energy(context, &change);
        context.set_atom_kind(0, AtomKindId::new(kind));
        let after = context.hamiltonian().energy(context, &change);
        after - before
    }

    /// −7/8 z² e²/(4πε₀εᵣ r_c) in kJ/mol at εᵣ = 80 (Fanourgakis 2009).
    fn fanourgakis_self_energy(z: f64, cutoff: f64) -> f64 {
        -7.0 / 8.0 * z * z * interatomic::coulomb::TO_CHEMISTRY_UNIT / 80.0 / cutoff
    }

    fn system_energy(context: &Backend) -> f64 {
        context.hamiltonian().energy(context, &Change::Everything)
    }

    // Invariant 1: retyping a neutral atom to a unit charge costs −7/8 l_B/r_c kT.
    #[test]
    fn atom_swap_neutral_to_unit_charge_costs_the_self_energy() {
        let mut context = isolated(
            "",
            "    - !Nonbonded {default: [!Fanourgakis {cutoff: 80.0}]}",
        );
        assert_relative_eq!(
            swap_energy(&mut context, 1),
            fanourgakis_self_energy(1.0, 80.0),
            max_relative = 1e-12
        );
    }

    // Invariant 5: a `replace:` entry for the kind's own pair sets its scheme.
    #[test]
    fn replaced_pair_scheme_sets_the_kind_self_energy() {
        let mut context = isolated(
            "",
            "    - !Nonbonded\n        default: [!Coulomb {cutoff: 80.0}]\n        replace:\n          [P, P]: [!Fanourgakis {cutoff: 60.0}]",
        );
        assert_relative_eq!(
            swap_energy(&mut context, 1),
            fanourgakis_self_energy(1.0, 60.0),
            max_relative = 1e-12
        );
    }

    // Invariant 2: medium salt screens the self term exactly as it screens the pair.
    #[test]
    fn medium_salt_screens_the_self_energy_like_the_pair() {
        use interatomic::coulomb::{permittivity::ConstantPermittivity, DebyeLength};
        let fanourgakis = "    - !Nonbonded {default: [!Fanourgakis {cutoff: 80.0}]}";
        let salt = ", salt: [!NaCl, 0.005]";
        let mut context = isolated(salt, fanourgakis);
        let medium = crate::backend::get_medium_str(&format!(
            "system:\n  medium: {{permittivity: !Fixed 80.0, temperature: 300.0{salt}}}"
        ))
        .unwrap();
        let debye_length = medium.debye_length();
        let pair = interatomic::twobody::IonIon::new(
            1.0,
            ConstantPermittivity::new(80.0),
            interatomic::coulomb::pairwise::Fanourgakis::new(80.0, debye_length),
        );
        assert!(pair.kappa().is_some());
        let screened = swap_energy(&mut context, 1);
        assert_relative_eq!(screened, pair.self_energy(1.0), max_relative = 1e-12);
        // Screening must matter here, or the check above could pass unscreened
        assert!(screened.abs() < 0.5 * fanourgakis_self_energy(1.0, 80.0).abs());
    }

    // Invariant 3: moving charges never touches the self term.
    #[test]
    fn translation_leaves_the_self_term_at_zero() {
        use crate::cell::VolumeScalePolicy;
        use crate::montecarlo::NewOld;
        let context = isolated(
            "",
            "    - !Nonbonded {default: [!Fanourgakis {cutoff: 80.0}]}",
        );
        assert!(self_term_energy(&context, &Change::Everything) != 0.0);
        let moves = [
            Change::SingleGroup(1, GroupChange::RigidBody),
            Change::SingleGroup(1, GroupChange::PartialUpdate(vec![RelIndex::new(0)])),
            Change::Volume(VolumeScalePolicy::Isotropic, NewOld::from(2.0, 1.0)),
        ];
        for change in moves {
            assert_eq!(self_term_energy(&context, &change), 0.0, "{change:?}");
        }
    }

    // Invariant 6: reciprocal `!Ewald` carries its own self term, so none is added.
    // Pinned on the code before the pair self term existed.
    #[test]
    fn reciprocal_ewald_energy_is_unchanged() {
        let context = ions(
            30.0,
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 7.0, 0.0]],
            "",
            "    - !Nonbonded {default: []}\n    - !Ewald {cutoff: 12.0, accuracy: 1e-5}",
        );
        assert_relative_eq!(
            system_energy(&context),
            -2.242222080154793,
            max_relative = 1e-12
        );
    }

    /// Build a single cation with `energy`, returning the error.
    fn build_error(energy: &str) -> String {
        let yaml = format!(
            r#"
atoms:
  - {{name: P, mass: 1.0, charge: 1.0, sigma: 1.0}}
molecules:
  - {{name: p, atoms: [P]}}
system:
  cell: !Cuboid [30.0, 30.0, 30.0]
  medium: {{permittivity: !Fixed 80.0, temperature: 300.0}}
  energy:
{energy}
  blocks:
    - {{molecule: p, N: 1, insert: !Manual [[0.0, 0.0, 0.0]]}}
propagate: {{seed: !Fixed 1, criterion: Metropolis, steps: 0, collections: []}}
"#
        );
        let error = Backend::from_yaml_str(&yaml, None, &mut rand::thread_rng()).unwrap_err();
        format!("{error:#}")
    }

    // Invariant 6: a pairwise Ewald beside reciprocal `!Ewald` would count both the
    // real-space pair energy and the self term twice.
    #[test]
    fn pairwise_ewald_beside_reciprocal_ewald_is_an_error() {
        for pairwise in [
            "!Ewald {cutoff: 12.0, accuracy: 1e-5}",
            "!CoulombEwald {cutoff: 12.0, alpha: 0.3}",
        ] {
            let error = build_error(&format!(
                "    - !Nonbonded {{default: [{pairwise}]}}\n    - !Ewald {{cutoff: 12.0, accuracy: 1e-5}}"
            ));
            assert!(error.contains("real-space"), "{pairwise}: {error}");
        }
    }

    // Invariant 1: a spline or culling cutoff below the scheme's truncates the pair a second
    // time, so the scheme's self energy no longer matches the pair in use.
    #[test]
    fn pair_range_shorter_than_the_self_energy_scheme_is_an_error() {
        for range in ["spline: {cutoff: 14.0}", "cutoff: 14.0"] {
            let error = build_error(&format!(
                "    - !Nonbonded\n        default: [!Fanourgakis {{cutoff: 80.0}}]\n        {range}"
            ));
            assert!(error.contains("self energy"), "{range}: {error}");
        }
    }
}

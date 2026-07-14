//! A minimal, working [`Analyze`] implementation, kept as a starting point for new analyses.
//!
//! It measures the mean charge of the selected atoms. Nothing about that is interesting; what the
//! file demonstrates is the shape every analysis shares:
//!
//! 1. A `…Builder` struct deserialized from the `analysis:` section of the input, holding a
//!    [`Selection`] and a [`Frequency`].
//! 2. A `build()` method that takes the context and resolves the selection **once**, eagerly.
//! 3. The analysis proper, which owns a [`Sampling`] and nothing else about bookkeeping.
//!
//! The test at the bottom runs it against a real system, so this file cannot rot: it is compiled
//! and executed by `cargo test`. Copy it, rename it, and register the builder in
//! [`AnalysisBuilder`](super::AnalysisBuilder).

// Nothing but the tests below drives this analysis; it is a template, not a registered analysis.
#![allow(dead_code)]

use super::{Analyze, Frequency, Sampling};
use crate::selection::{Atoms, CachedSelection, Selection};
use crate::{Info, ObserveContext};
use anyhow::Result;

/// Deserialized from the input file; holds no resolved state.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeanChargeBuilder {
    selection: Selection,
    frequency: Frequency,
}

impl MeanChargeBuilder {
    /// Resolve the selection once, here, rather than on every sample.
    ///
    /// The cache invalidates itself when the system changes in a way the selection can see — a
    /// grand-canonical resize, or an atom-kind swap for an `atomtype` selection. That is why the
    /// resolved indices must live in a [`CachedSelection`] and never in a plain `Vec`.
    pub fn build(&self, context: &impl ObserveContext) -> Result<MeanCharge> {
        anyhow::ensure!(
            !context.resolve_atoms(&self.selection).is_empty(),
            "MeanCharge: selection '{}' matched no atoms",
            self.selection.source()
        );
        Ok(MeanCharge {
            selection: CachedSelection::atoms(self.selection.clone()),
            sum: 0.0,
            sampling: Sampling::new(self.frequency),
        })
    }
}

/// Mean charge of the selected atoms, averaged over frames.
#[derive(Debug)]
pub struct MeanCharge {
    selection: CachedSelection<Atoms>,
    sum: f64,
    /// Frequency and frame count. The framework increments the count; never do it here.
    sampling: Sampling,
}

impl Info for MeanCharge {
    fn short_name(&self) -> Option<&'static str> {
        Some("mean_charge")
    }
    fn long_name(&self) -> Option<&'static str> {
        Some("Mean charge of the selected atoms")
    }
}

impl<T: ObserveContext> Analyze<T> for MeanCharge {
    fn sampling(&self) -> &Sampling {
        &self.sampling
    }

    fn sampling_mut(&mut self) -> &mut Sampling {
        &mut self.sampling
    }

    /// Called only when the frequency check passes. The context is read-only: an analysis that
    /// needs a trial move takes `T: PerturbContext` instead and perturbs a clone through
    /// `measure` — see `analysis/virtual_translate.rs`.
    fn perform_sample(&mut self, context: &T, _step: usize, _weight: f64) -> Result<()> {
        let atoms = self.selection.resolve(context);
        let charge: f64 = atoms.iter().map(|i| context.atom_charge(i.get())).sum();
        self.sum += charge / atoms.len() as f64;
        Ok(())
    }

    /// Called only when at least one sample was taken, so dividing by the frame count is safe.
    fn results(&self) -> Option<serde_yml::Value> {
        let mean = self.sum / self.sampling.num_samples() as f64;
        let mut map = serde_yml::Mapping::new();
        map.insert("mean_charge".into(), mean.into());
        Some(serde_yml::Value::Mapping(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::group::GroupCollectionMut;

    /// One anion and one cation, so the mean charge is zero until a kind is swapped.
    const SALT: &str = r#"
atoms:
  - {name: Na, mass: 22.99, charge: 1.0, sigma: 3.3}
  - {name: Cl, mass: 35.45, charge: -1.0, sigma: 3.3}
molecules:
  - name: pair
    atoms: [Na, Cl]
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: pair
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;

    fn backend() -> Backend {
        Backend::from_yaml_str(SALT, None, &mut rand::thread_rng()).unwrap()
    }

    fn analysis(source: &str) -> MeanCharge {
        let context = backend();
        MeanChargeBuilder {
            selection: Selection::parse(source).unwrap(),
            frequency: Frequency::Every(1),
        }
        .build(&context)
        .unwrap()
    }

    #[test]
    fn averages_the_charge_over_frames() {
        let context = backend();
        let mut analysis = analysis("all");
        analysis.sample(&context, 0).unwrap();
        analysis.sample(&context, 1).unwrap();

        assert_eq!(Analyze::<Backend>::num_samples(&analysis), 2);
        // +1 and −1 average to zero, every frame.
        assert_eq!(analysis.sum, 0.0);
    }

    /// The point of the eager `CachedSelection`: a swap that leaves group composition untouched
    /// still has to be seen by an `atomtype` selection.
    #[test]
    fn an_atom_kind_swap_is_picked_up() {
        let mut context = backend();
        let mut analysis = analysis("atomtype Na");
        analysis.sample(&context, 0).unwrap();
        assert_eq!(analysis.sum, 1.0);

        // Turn the chloride into a second sodium; the selection now matches both atoms.
        context.set_atom_kind(1, crate::group::AtomKindId::new(0));
        analysis.sample(&context, 1).unwrap();
        assert_eq!(analysis.sum, 2.0);
    }

    #[test]
    fn an_empty_selection_is_rejected_at_build_time() {
        let context = backend();
        let builder = MeanChargeBuilder {
            selection: Selection::parse("atomtype Ca").unwrap(),
            frequency: Frequency::Every(1),
        };
        assert!(builder.build(&context).is_err());
    }
}

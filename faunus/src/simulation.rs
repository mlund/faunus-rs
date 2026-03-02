//! Simulation runners for single-box and (future) multi-box ensemble simulations.

use crate::{
    analysis,
    montecarlo::MarkovChain,
    platform::reference::ReferencePlatform,
    propagate::Propagate,
    state::State,
};
use anyhow::Result;
use std::path::Path;

/// Thermal energy kT in kJ/mol from a medium's temperature.
pub fn thermal_energy(medium: &interatomic::coulomb::Medium) -> f64 {
    use interatomic::coulomb::Temperature;
    const KILO_JOULE_PER_JOULE: f64 = 1e-3;
    physical_constants::MOLAR_GAS_CONSTANT * KILO_JOULE_PER_JOULE * medium.temperature()
}

/// Top-level simulation driver.
///
/// `SingleBox` covers standard NVT / NPT simulations.
/// `Gibbs` (Phase 2) will hold two coupled boxes for Gibbs ensemble MC.
pub enum Simulation {
    SingleBox(MarkovChain<ReferencePlatform>),
}

impl Simulation {
    /// Build a simulation from an input YAML file.
    pub fn from_file(
        input: &Path,
        state: Option<&Path>,
    ) -> Result<(Self, interatomic::coulomb::Medium)> {
        let medium = crate::platform::reference::get_medium(input)?;

        let context = ReferencePlatform::new(input, None, &mut rand::thread_rng())?;
        let propagate = Propagate::from_file(input, &context)?;

        let analyses = analysis::from_file(input, &context)?;
        let mut mc =
            MarkovChain::new(context, propagate, thermal_energy(&medium), analyses)?;

        if let Some(state_path) = state {
            if state_path.exists() {
                mc.load_state(State::from_file(state_path)?)?;
            }
        }

        Ok((Self::SingleBox(mc), medium))
    }

    /// Access the inner `MarkovChain` (single-box mode).
    pub fn as_single_box(&self) -> &MarkovChain<ReferencePlatform> {
        match self {
            Self::SingleBox(mc) => mc,
        }
    }

    /// Mutable access to the inner `MarkovChain` (single-box mode).
    pub fn as_single_box_mut(&mut self) -> &mut MarkovChain<ReferencePlatform> {
        match self {
            Self::SingleBox(mc) => mc,
        }
    }
}

/// Serialize data to a YAML file under an optional key.
pub(crate) fn write_yaml<T: serde::Serialize>(
    data: &T,
    output: &mut std::fs::File,
    key: Option<&str>,
) -> Result<()> {
    use std::io::Write;
    match key {
        Some(key) => {
            let mut wrapper = std::collections::BTreeMap::new();
            wrapper.insert(key.to_string(), data);
            let yaml = serde_yaml::to_string(&wrapper)?;
            output.write_all(yaml.as_bytes())?;
        }
        None => {
            let yaml = serde_yaml::to_string(data)?;
            output.write_all(yaml.as_bytes())?;
        }
    }
    Ok(())
}

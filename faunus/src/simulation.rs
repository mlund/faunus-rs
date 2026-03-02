//! Simulation runners for single-box and Gibbs ensemble simulations.

use crate::{
    analysis,
    montecarlo::{gibbs::GibbsEnsemble, MarkovChain},
    platform::reference::ReferencePlatform,
    propagate::{self, Propagate},
    state::State,
};
use anyhow::Result;
use rand::SeedableRng;
use std::path::Path;

/// Thermal energy kT in kJ/mol from a medium's temperature.
pub fn thermal_energy(medium: &interatomic::coulomb::Medium) -> f64 {
    use interatomic::coulomb::Temperature;
    const KILO_JOULE_PER_JOULE: f64 = 1e-3;
    physical_constants::MOLAR_GAS_CONSTANT * KILO_JOULE_PER_JOULE * medium.temperature()
}

/// Top-level simulation driver.
#[allow(clippy::large_enum_variant)]
pub enum Simulation {
    /// Standard single-box NVT / NPT simulation.
    SingleBox(MarkovChain<ReferencePlatform>),
    /// Two coupled boxes for Gibbs ensemble MC.
    Gibbs(Box<GibbsEnsemble<ReferencePlatform>>),
}

impl Simulation {
    /// Build a simulation from an input YAML file.
    ///
    /// If `propagate.gibbs` is present, constructs a two-box Gibbs ensemble;
    /// otherwise falls back to single-box mode.
    pub fn from_file(
        input: &Path,
        state: Option<&Path>,
    ) -> Result<(Self, interatomic::coulomb::Medium)> {
        let medium = crate::platform::reference::get_medium(input)?;
        let kt = thermal_energy(&medium);

        if let Some(gibbs_cfg) = propagate::gibbs_config_from_file(input)? {
            let sim = Self::build_gibbs(input, state, kt, gibbs_cfg)?;
            return Ok((sim, medium));
        }

        // --- single-box path (unchanged) ---
        let context = ReferencePlatform::new(input, None, &mut rand::thread_rng())?;
        let propagate = Propagate::from_file(input, &context)?;
        let analyses = analysis::from_file(input, &context)?;
        let mut mc = MarkovChain::new(context, propagate, kt, analyses)?;

        if let Some(state_path) = state {
            if state_path.exists() {
                mc.load_state(State::from_file(state_path)?)?;
            }
        }

        Ok((Self::SingleBox(mc), medium))
    }

    /// Build Gibbs ensemble from two cloned boxes.
    fn build_gibbs(
        input: &Path,
        state: Option<&Path>,
        kt: f64,
        gibbs_cfg: crate::montecarlo::gibbs::GibbsConfig,
    ) -> Result<Self> {
        let context0 = ReferencePlatform::new(input, None, &mut rand::thread_rng())?;
        let context1 = context0.clone();

        let propagate0 = Propagate::from_file(input, &context0)?;
        let mut propagate1 = Propagate::from_file(input, &context1)?;
        // give box 1 a distinct seed so intra-box trajectories diverge
        propagate1.reseed(0x000D_EADB_EEFC_AFE1);

        let analyses0 = analysis::from_file(input, &context0)?;
        let analyses1 = analysis::from_file(input, &context1)?;

        let max_sweeps = propagate0.max_repeats() / gibbs_cfg.intra_steps.max(1);
        log::info!(
            "Gibbs ensemble: {} sweeps, {} intra-steps, {} inter-moves",
            max_sweeps,
            gibbs_cfg.intra_steps,
            gibbs_cfg.moves.len()
        );

        let inter_moves: Vec<_> = gibbs_cfg
            .moves
            .into_iter()
            .map(|b| b.build(&context0))
            .collect::<Result<_>>()?;

        let mut mc0 = MarkovChain::new(context0, propagate0, kt, analyses0)?;
        let mut mc1 = MarkovChain::new(context1, propagate1, kt, analyses1)?;

        // load per-box state files if they exist
        if let Some(state_path) = state {
            for (i, mc) in [&mut mc0, &mut mc1].iter_mut().enumerate() {
                let box_state = box_prefixed_path(state_path, i);
                if box_state.exists() {
                    mc.load_state(State::from_file(&box_state)?)?;
                    log::info!("Restored box {} state from {}", i, box_state.display());
                }
            }
        }

        let rng = rand::rngs::StdRng::from_entropy();
        let ensemble = GibbsEnsemble::new(
            [mc0, mc1],
            inter_moves,
            gibbs_cfg.intra_steps,
            max_sweeps,
            kt,
            rng,
        );

        Ok(Self::Gibbs(Box::new(ensemble)))
    }
}

/// Prefix a file path with `box{i}_`, e.g. `dir/output.yaml` → `dir/box0_output.yaml`.
pub(crate) fn box_prefixed_path(path: &Path, box_index: usize) -> std::path::PathBuf {
    let stem = path
        .file_stem()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("file");
    let ext = path
        .extension()
        .unwrap_or_default()
        .to_str()
        .unwrap_or("yaml");
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!("box{box_index}_{stem}.{ext}"))
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

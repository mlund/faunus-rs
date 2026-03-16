//! Simulation runners for single-box and Gibbs ensemble simulations.

use crate::{
    analysis::{self, AnalysisCollection},
    backend::Backend,
    montecarlo::{gibbs::GibbsEnsemble, MarkovChain},
    propagate::{self, Propagate},
    state::State,
};
use anyhow::Result;
use interatomic::coulomb::Temperature;
use rand::SeedableRng;
use std::path::Path;

/// Top-level simulation driver.
#[allow(clippy::large_enum_variant)]
pub enum Simulation {
    /// Standard single-box simulation.
    SingleBox(MarkovChain<Backend>),
    /// Two coupled boxes for Gibbs ensemble MC.
    Gibbs(Box<GibbsEnsemble<Backend>>),
}

/// Build a MarkovChain from an input file, context, and thermal energy,
/// optionally restoring from a state checkpoint.
pub fn build_markov_chain<T: crate::Context + 'static>(
    input: &Path,
    context: T,
    rt: f64,
    state: Option<&Path>,
) -> Result<MarkovChain<T>> {
    let propagate = Propagate::from_file(input, &context)?;
    let analyses = analysis::from_file(input, &context)?;
    let mut mc = MarkovChain::new(context, propagate, rt, analyses)?;
    if let Some(state_path) = state {
        if state_path.exists() {
            mc.load_state(State::from_file(state_path)?)?;
        }
    }
    Ok(mc)
}

/// Build a context and analysis collection from an input YAML file,
/// without constructing a `Propagate`. Used by `rerun`.
pub fn build_context_and_analyses(
    input: &Path,
) -> Result<(
    Backend,
    AnalysisCollection<Backend>,
    interatomic::coulomb::Medium,
)> {
    let medium = crate::backend::get_medium(input)?;
    let context = Backend::new(input, None, &mut rand::thread_rng())?;
    let analyses = analysis::from_file(input, &context)?;
    Ok((context, analyses, medium))
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
        let medium = crate::backend::get_medium(input)?;
        let rt = crate::R_IN_KJ_PER_MOL * medium.temperature();

        if let Some(gibbs_cfg) = propagate::gibbs_config_from_file(input)? {
            let sim = Self::build_gibbs(input, state, rt, gibbs_cfg)?;
            return Ok((sim, medium));
        }

        let context = Backend::new(input, None, &mut rand::thread_rng())?;
        let mc = build_markov_chain(input, context, rt, state)?;
        Ok((Self::SingleBox(mc), medium))
    }

    /// Build Gibbs ensemble from two cloned boxes.
    fn build_gibbs(
        input: &Path,
        state: Option<&Path>,
        rt: f64,
        gibbs_cfg: crate::montecarlo::gibbs::GibbsConfig,
    ) -> Result<Self> {
        let context0 = Backend::new(input, None, &mut rand::thread_rng())?;
        let context1 = context0.clone();

        let inter_moves: Vec<_> = gibbs_cfg
            .moves
            .into_iter()
            .map(|b| b.build(&context0))
            .collect::<Result<_>>()?;

        let mut mc0 = build_markov_chain(input, context0, rt, None)?;
        let mut mc1 = build_markov_chain(input, context1, rt, None)?;
        // give box 1 a distinct seed so intra-box trajectories diverge
        mc1.propagation_mut().reseed(0x000D_EADB_EEFC_AFE1);

        let max_sweeps = mc0.propagation().max_repeats() / gibbs_cfg.intra_steps.max(1);
        log::info!(
            "Gibbs ensemble: {} sweeps, {} intra-steps, {} inter-moves",
            max_sweeps,
            gibbs_cfg.intra_steps,
            inter_moves.len()
        );

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
            rt,
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

//! Loading, running and reporting a simulation.
//!
//! [`Simulation`] is the crate's only public door into the engine: it assembles a system from an
//! input, drives it to completion, and hands back an opaque [`SimulationOutput`]. The Monte Carlo
//! machinery it builds on — contexts, moves, analyses — stays internal.

use crate::{
    analysis::{self, AnalysisCollection, AnalysisCollectionExt, Frequency},
    backend::Backend,
    context::PerturbContext,
    energy::EnergyChange,
    error::{Error, Result},
    group::{GroupCollection, GroupCollectionMut},
    montecarlo::{gibbs::GibbsEnsemble, MarkovChain},
    propagate::{self, Propagate},
    state::State,
    topology::io::frame_state::{self, FrameStateReader},
    Change, Context, ObserveContext, Particle, Point, WithHamiltonian,
};
use interatomic::coulomb::{Medium, Temperature};
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// Thermal energy kT in kJ/mol.
fn thermal_energy(medium: &Medium) -> f64 {
    crate::R_IN_KJ_PER_MOL * medium.temperature()
}

/// Sum of charges of all active particles in the system.
fn net_charge(context: &impl ObserveContext) -> f64 {
    context
        .groups()
        .iter()
        .flat_map(|g| g.iter_active())
        .map(|i| context.atom_charge(i))
        .sum()
}

/// Git revision the binary was built from, e.g. `v0.2.0-3-gabc123def-dirty`.
///
/// Falls back to "unknown" when built outside a git checkout (source archive or
/// published crate), where vergen emits its idempotent placeholder.
pub(crate) fn git_revision() -> &'static str {
    match env!("VERGEN_GIT_DESCRIBE") {
        "VERGEN_IDEMPOTENT_OUTPUT" => "unknown",
        revision => revision,
    }
}

/// A failed Monte Carlo step, boxed so the caller keeps the whole `anyhow` context chain.
fn run_failed(error: anyhow::Error) -> Error {
    Error::Run(error.into())
}

/// Whether two paths denote the same file. Canonicalizes where the files exist (resolving `..`
/// and symlinks) and falls back to lexical absolutization where they do not, so a relative
/// `traj.xtc` and an absolute one under the same directory still compare equal.
fn same_file(a: &Path, b: &Path) -> bool {
    let resolve = |path: &Path| {
        std::fs::canonicalize(path)
            .or_else(|_| std::path::absolute(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    resolve(a) == resolve(b)
}

/// Serialize data as a YAML document, optionally nested under a single key.
fn yaml_block<T: serde::Serialize>(data: &T, key: Option<&str>) -> anyhow::Result<String> {
    Ok(match key {
        Some(key) => {
            let mut wrapper = BTreeMap::new();
            wrapper.insert(key.to_string(), data);
            serde_yml::to_string(&wrapper)?
        }
        None => serde_yml::to_string(data)?,
    })
}

/// Serialize data to a YAML file under an optional key.
// Only the `cli`-gated umbrella / Wang-Landau drivers use this; the facade builds strings.
#[cfg(feature = "cli")]
pub fn write_yaml<T: serde::Serialize>(
    data: &T,
    output: &mut std::fs::File,
    key: Option<&str>,
) -> anyhow::Result<()> {
    use std::io::Write;
    output.write_all(yaml_block(data, key)?.as_bytes())?;
    Ok(())
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

/// Where an input document comes from.
///
/// Every section of the input can be parsed from either a path or a string, but the two are not
/// interchangeable: `include:` directives resolve against the input file's directory, so the
/// `File` arm must reach the path-based readers rather than route through the string ones.
#[derive(Clone, Copy)]
enum Source<'a> {
    File(&'a Path),
    Yaml(&'a str),
}

impl Source<'_> {
    fn medium(&self) -> anyhow::Result<Medium> {
        match *self {
            Self::File(path) => crate::backend::get_medium(path),
            Self::Yaml(yaml) => crate::backend::get_medium_str(yaml),
        }
    }

    /// Parse the whole document into a `Value` for whole-section key validation,
    /// applying the same template/underscore preprocessing as the file readers.
    fn root_value(&self) -> anyhow::Result<serde_yml::Value> {
        let yaml = match *self {
            Self::File(path) => crate::auxiliary::read_yaml(path)?,
            Self::Yaml(yaml) => yaml.to_string(),
        };
        Ok(serde_yml::from_str(&yaml)?)
    }

    fn setup_rng(&self) -> anyhow::Result<rand::rngs::StdRng> {
        match *self {
            Self::File(path) => propagate::setup_rng_from_file(path),
            Self::Yaml(yaml) => propagate::setup_rng_from_str(yaml),
        }
    }

    fn gibbs_config(&self) -> anyhow::Result<Option<crate::montecarlo::gibbs::GibbsConfig>> {
        match *self {
            Self::File(path) => propagate::gibbs_config_from_file(path),
            Self::Yaml(yaml) => propagate::gibbs_config_from_str(yaml),
        }
    }

    fn backend(&self, structure: Option<&Path>, rng: &mut impl Rng) -> anyhow::Result<Backend> {
        match *self {
            Self::File(path) => Backend::new(path, structure, rng),
            Self::Yaml(yaml) => Backend::from_yaml_str(yaml, structure, rng),
        }
    }

    fn propagate<T: Context>(
        &self,
        context: &T,
        thermal_energy: f64,
    ) -> anyhow::Result<Propagate<T>> {
        match *self {
            Self::File(path) => Propagate::from_file(path, context, thermal_energy),
            Self::Yaml(yaml) => Propagate::from_str(yaml, context, thermal_energy),
        }
    }

    fn analyses<T: Context>(
        &self,
        context: &T,
        medium: Option<&Medium>,
        output_dir: Option<&Path>,
    ) -> anyhow::Result<AnalysisCollection<T>> {
        if let Some(dir) = output_dir {
            std::fs::create_dir_all(dir).map_err(|err| {
                anyhow::anyhow!("Cannot create output directory {:?}: {}", dir, err)
            })?;
        }
        match *self {
            Self::File(path) => analysis::from_file_in_dir(path, context, medium, output_dir),
            Self::Yaml(yaml) => analysis::from_str_in_dir(yaml, context, medium, output_dir),
        }
    }
}

/// Build a Markov chain from an input, context and thermal energy,
/// optionally restoring from a state checkpoint.
fn build_markov_chain<T: Context + 'static>(
    source: Source,
    context: T,
    rt: f64,
    state: Option<&Path>,
    medium: Option<&Medium>,
    output_dir: Option<&Path>,
) -> anyhow::Result<MarkovChain<T>> {
    let propagate = source.propagate(&context, rt)?;
    let analyses = source.analyses(&context, medium, output_dir)?;
    let mut mc = MarkovChain::new(context, propagate, rt, analyses)?;
    if let Some(state_path) = state {
        if state_path.exists() {
            mc.load_state(State::from_file(state_path)?)?;
        }
    }
    Ok(mc)
}

/// What a [`Simulation`] drives: one box, or two coupled ones.
#[allow(clippy::large_enum_variant)]
enum Engine {
    /// Standard single-box simulation.
    SingleBox(MarkovChain<Backend>),
    /// Two coupled boxes for Gibbs ensemble MC.
    Gibbs(Box<GibbsEnsemble<Backend>>),
}

impl Engine {
    const fn num_boxes(&self) -> usize {
        match self {
            Self::SingleBox(_) => 1,
            Self::Gibbs(_) => 2,
        }
    }
}

/// A loaded simulation, ready to run.
///
/// Build one with [`from_file`](Self::from_file) or [`from_yaml`](Self::from_yaml), drive it with
/// [`run`](Self::run), and checkpoint it with [`save_state`](Self::save_state).
///
/// ```no_run
/// let mut simulation = faunus::Simulation::from_file("input.yaml".as_ref(), None)?;
/// let output = simulation.run()?;
/// output.write_to("output.yaml".as_ref())?;
/// # Ok::<(), faunus::Error>(())
/// ```
pub struct Simulation {
    engine: Engine,
    medium: Medium,
}

// Summarize rather than derive: the engine holds the whole system, and printing it would bury
// the caller in coordinates.
impl std::fmt::Debug for Simulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Simulation")
            .field("boxes", &self.engine.num_boxes())
            .field("medium", &self.medium)
            .finish()
    }
}

impl Simulation {
    /// Load a simulation from an input YAML file.
    ///
    /// A `propagate.gibbs` section selects a two-box Gibbs ensemble; otherwise the simulation is
    /// single-box. When `state` names an existing checkpoint it is restored, and
    /// [`save_state`](Self::save_state) will write back to it.
    pub fn from_file(input: &Path, state: Option<&Path>) -> Result<Self> {
        // Name the offending path now; further in, a missing file surfaces as a parse error.
        std::fs::File::open(input).map_err(|source| Error::io(input, source))?;
        Self::build(Source::File(input), None, state)
    }

    /// Load a single-box simulation from a self-contained YAML document.
    ///
    /// The document must carry every section inline: `include:` directives are not resolved,
    /// having no directory to resolve against. A `propagate.gibbs` section yields
    /// [`Error::Unsupported`] — use [`from_file`](Self::from_file) for Gibbs ensembles.
    ///
    /// Analyses that write files resolve their paths against the current working directory.
    pub fn from_yaml(yaml: &str, structure: Option<&Path>) -> Result<Self> {
        Self::build(Source::Yaml(yaml), structure, None)
    }

    fn build(source: Source, structure: Option<&Path>, state: Option<&Path>) -> Result<Self> {
        crate::auxiliary::validate_section_keys(&source.root_value()?)?;
        let medium = source.medium()?;
        let rt = crate::R_IN_KJ_PER_MOL * medium.temperature();

        let engine = match source.gibbs_config()? {
            // Gibbs anchors its per-box outputs beside the state file, which a bare string has not
            // got. Rejecting here rather than in `from_yaml` keeps the document parsed only once.
            Some(_) if matches!(source, Source::Yaml(_)) => {
                return Err(Error::Unsupported(
                    "a `propagate.gibbs` section needs `Simulation::from_file`".to_string(),
                ))
            }
            Some(config) => build_gibbs(source, structure, state, rt, config, &medium)?,
            None => {
                let context = source.backend(structure, &mut source.setup_rng()?)?;
                let mc = build_markov_chain(source, context, rt, state, Some(&medium), None)?;
                Engine::SingleBox(mc)
            }
        };
        Ok(Self { engine, medium })
    }

    /// Run the simulation to completion.
    ///
    /// Logs the medium, thermal energy, initial and final energy, net charge and energy drift.
    pub fn run(&mut self) -> Result<SimulationOutput> {
        self.run_with_progress(&mut |_, _| {})
    }

    /// As [`run`](Self::run), invoking `on_step(step, total)` after each completed cycle —
    /// Monte Carlo steps for a single box, sweeps for a Gibbs ensemble.
    pub fn run_with_progress(
        &mut self,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<SimulationOutput> {
        log::info!("{}", self.medium);
        log::info!("Thermal energy: {:.2} kJ/mol", thermal_energy(&self.medium));
        match &mut self.engine {
            Engine::SingleBox(mc) => run_single_box(mc, &self.medium, on_step),
            Engine::Gibbs(ensemble) => run_gibbs(ensemble, &self.medium, on_step),
        }
    }

    /// Write a checkpoint that [`from_file`](Self::from_file) can restore.
    ///
    /// A Gibbs ensemble writes one file per box, `box0_`/`box1_`-prefixed, beside `path`.
    pub fn save_state(&self, path: &Path) -> Result<()> {
        match &self.engine {
            Engine::SingleBox(mc) => {
                mc.save_state().to_file(path)?;
                log::info!("Saved simulation state to {}", path.display());
            }
            Engine::Gibbs(ensemble) => {
                for (i, mc) in ensemble.boxes().iter().enumerate() {
                    let box_state = box_prefixed_path(path, i);
                    mc.save_state().to_file(&box_state)?;
                    log::info!("Saved box {i} state to {}", box_state.display());
                }
            }
        }
        Ok(())
    }
}

/// Build a Gibbs ensemble from two cloned boxes.
fn build_gibbs(
    source: Source,
    structure: Option<&Path>,
    state: Option<&Path>,
    rt: f64,
    gibbs_cfg: crate::montecarlo::gibbs::GibbsConfig,
    medium: &Medium,
) -> Result<Engine> {
    // One master stream for the ensemble: initial coordinates, box 1's chain
    // and the inter-box moves all draw from it, so `propagate.seed` pins the
    // whole run. Box 0 keeps the chain `Propagate` seeds from the same value.
    let mut setup_rng = source.setup_rng()?;
    let context0 = source.backend(structure, &mut setup_rng)?;
    let context1 = context0.clone();

    let inter_moves: Vec<_> = gibbs_cfg
        .moves
        .into_iter()
        .map(|b| b.build(&context0))
        .collect::<anyhow::Result<_>>()?;

    // Per-box analyses share their `file:` paths and would race under
    // parallel intra-box MC; route each box's outputs into its own
    // subdirectory, anchored the same way `box_prefixed_path` anchors
    // per-box state files.
    let anchor = state
        .and_then(|p| p.parent())
        .unwrap_or_else(|| Path::new("."));
    let out_dir0 = anchor.join("box0");
    let out_dir1 = anchor.join("box1");
    let mut mc0 = build_markov_chain(source, context0, rt, None, Some(medium), Some(&out_dir0))?;
    let mut mc1 = build_markov_chain(source, context1, rt, None, Some(medium), Some(&out_dir1))?;
    // Give box 1 a distinct seed so intra-box trajectories diverge. Drawing it
    // from the master stream keeps box 1 responsive to `propagate.seed`; a
    // hardcoded constant would give it the same chain for every input.
    mc1.propagation_mut().reseed(setup_rng.gen());

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

    let rng = rand::rngs::StdRng::from_rng(&mut setup_rng).map_err(anyhow::Error::from)?;
    let ensemble = GibbsEnsemble::new(
        [mc0, mc1],
        inter_moves,
        gibbs_cfg.intra_steps,
        max_sweeps,
        rt,
        rng,
    );

    Ok(Engine::Gibbs(Box::new(ensemble)))
}

/// Per-box YAML: version, medium, blocks, cell, energy, propagate, analysis.
///
/// The blocks are concatenated in a fixed order; the regression fixtures compare against this
/// document, so a reordering reads as a physics change.
fn box_result<T: Context + 'static>(
    mc: &MarkovChain<T>,
    medium: &Medium,
    initial_energy: f64,
) -> anyhow::Result<BoxResult> {
    let version = BTreeMap::from([
        ("faunus", env!("CARGO_PKG_VERSION")),
        ("git_revision", git_revision()),
    ]);
    let final_energy = mc.system_energy();
    let drift = mc.energy_drift(initial_energy);
    let energy_summary = BTreeMap::from([
        ("initial".to_string(), initial_energy),
        ("final".to_string(), final_energy),
        ("drift".to_string(), drift),
    ]);

    let mut yaml = yaml_block(&version, Some("version"))?;
    yaml.push_str(&yaml_block(medium, Some("medium"))?);
    yaml.push_str(&yaml_block(
        &mc.context().topology().blocks(),
        Some("blocks"),
    )?);
    yaml.push_str(&yaml_block(&mc.context().cell(), Some("cell"))?);
    yaml.push_str(&yaml_block(&energy_summary, Some("energy_change"))?);

    let energy_info = mc.context().hamiltonian().info_to_yaml();
    if energy_info.as_mapping().is_some_and(|map| !map.is_empty()) {
        yaml.push_str(&yaml_block(&energy_info, Some("energy"))?);
    }
    yaml.push_str(&yaml_block(
        &mc.context().hamiltonian().timing_to_yaml(),
        Some("energy_timers"),
    )?);
    yaml.push_str(&yaml_block(&mc.propagation().to_yaml(), Some("propagate"))?);

    let analysis_yaml = analysis::analyses_to_yaml(mc.analyses());
    if !analysis_yaml.is_empty() {
        yaml.push_str(&yaml_block(&analysis_yaml, Some("analysis"))?);
    }

    Ok(BoxResult {
        initial_energy,
        final_energy,
        drift,
        yaml: yaml.into(),
    })
}

/// Reject a non-finite starting energy (out-of-range penalty/constraint or overlap → `+∞`/NaN),
/// which would otherwise render the drift report as `inf`/`NaN` instead of a clear error.
fn require_finite_initial_energy(energy: f64) -> anyhow::Result<f64> {
    anyhow::ensure!(
        energy.is_finite(),
        "initial energy is non-finite ({energy}): the starting configuration is outside the \
         collective-variable range or has overlapping particles — start from a valid configuration"
    );
    Ok(energy)
}

fn run_single_box(
    mc: &mut MarkovChain<Backend>,
    medium: &Medium,
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<SimulationOutput> {
    let penalty_active = mc.context().hamiltonian().penalty().is_some();
    if penalty_active {
        log::info!("Penalty bias active: analysis averages are biased (use rerun for reweighting)");
    }

    // A `-∞` starting energy is unphysical (a singular/overlapping potential); fail loudly here
    // rather than let the first move to a finite state be silently auto-accepted.
    let initial_energy = crate::montecarlo::ensure_physical_energy(mc.system_energy())?;
    // A penalty/constraint CV that starts out of range gives `+∞`/NaN here, which would poison
    // the drift report — reject it. Plain runs may legitimately start from an overlap (`+∞`)
    // and relax on the first auto-accepted move, so this stricter check is penalty-only.
    if penalty_active {
        require_finite_initial_energy(initial_energy)?;
    }
    log::info!("Initial energy = {initial_energy:.2} kJ/mol");
    log::info!("Net charge = {:.4e} e", net_charge(mc.context()));

    let total = mc.propagation().max_repeats();
    for (i, step) in mc.iter().enumerate() {
        step.map_err(run_failed)?;
        on_step(i + 1, total);
    }

    mc.finalize_analyses()?;
    let result = box_result(mc, medium, initial_energy)?;

    let relative_drift = if result.final_energy.abs() > f64::EPSILON {
        result.drift / result.final_energy.abs()
    } else {
        result.drift
    };
    log::info!("Final energy = {:.2} kJ/mol", result.final_energy);
    log::info!("Net charge = {:.4e} e", net_charge(mc.context()));
    if relative_drift > 1e-9 {
        log::warn!(
            "Energy drift: {:.2e} kJ/mol (relative: {relative_drift:.2e}) ⚠️",
            result.drift
        );
    } else {
        log::info!(
            "Energy drift: {:.2e} kJ/mol (relative: {relative_drift:.2e}) ✅",
            result.drift
        );
    }

    // For a single box the summary *is* the box document; share it rather than copy it.
    Ok(SimulationOutput {
        yaml: Arc::clone(&result.yaml),
        boxes: vec![result],
    })
}

fn run_gibbs(
    ensemble: &mut GibbsEnsemble<Backend>,
    medium: &Medium,
    on_step: &mut dyn FnMut(usize, usize),
) -> Result<SimulationOutput> {
    let initial_energies: [f64; 2] = std::array::from_fn(|i| {
        let mc = &ensemble.boxes()[i];
        let energy = mc.system_energy();
        let charge = net_charge(mc.context());
        log::info!("Box {i}: initial energy = {energy:.2} kJ/mol, net charge = {charge:.4e} e");
        energy
    });
    for &energy in &initial_energies {
        crate::montecarlo::ensure_physical_energy(energy)?;
    }

    let max_sweeps = ensemble.max_sweeps();
    for sweep in 0..max_sweeps {
        // parallel intra-box + sequential inter-box (one sweep)
        let intra_steps = ensemble.intra_steps();
        let [box0, box1] = ensemble.boxes_mut();
        std::thread::scope(|scope| -> Result<()> {
            let handle0 = scope.spawn(|| box0.run_n_steps(intra_steps));
            let handle1 = scope.spawn(|| box1.run_n_steps(intra_steps));
            handle0
                .join()
                .map_err(|_| Error::Run("box 0 thread panicked".into()))?
                .map_err(run_failed)?;
            handle1
                .join()
                .map_err(|_| Error::Run("box 1 thread panicked".into()))?
                .map_err(run_failed)?;
            Ok(())
        })?;

        ensemble.perform_inter_moves().map_err(run_failed)?;
        on_step(sweep + 1, max_sweeps);
    }

    ensemble.finalize_analyses()?;

    let mut boxes = Vec::with_capacity(2);
    for (i, (mc, &initial_energy)) in ensemble.boxes().iter().zip(&initial_energies).enumerate() {
        let result = box_result(mc, medium, initial_energy)?;
        let charge = net_charge(mc.context());
        log::info!(
            "Box {i}: final energy = {:.2} kJ/mol, drift = {:.2e}, net charge = {charge:.4e} e",
            result.final_energy,
            result.drift
        );
        boxes.push(result);
    }

    // The main document carries only what is not per-box: the medium and the inter-box moves.
    let mut yaml = yaml_block(medium, Some("medium"))?;
    let inter_yaml = ensemble.inter_moves_to_yaml();
    if !inter_yaml.is_empty() {
        yaml.push_str(&yaml_block(&inter_yaml, Some("gibbs_moves"))?);
    }

    Ok(SimulationOutput {
        yaml: yaml.into(),
        boxes,
    })
}

/// The results of a completed run.
pub struct SimulationOutput {
    yaml: Arc<str>,
    boxes: Vec<BoxResult>,
}

// The YAML documents are megabytes for a large system; show the shape, not the contents.
impl std::fmt::Debug for SimulationOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimulationOutput")
            .field("boxes", &self.boxes)
            .finish_non_exhaustive()
    }
}

impl SimulationOutput {
    /// The summary document.
    ///
    /// For a single box this is the complete result. For a Gibbs ensemble it holds the medium and
    /// the inter-box move statistics, with the rest split across [`boxes`](Self::boxes).
    pub fn to_yaml(&self) -> &str {
        &self.yaml
    }

    /// One entry per simulation box: one for a single box, two for a Gibbs ensemble,
    /// none for a [`replay`].
    pub fn boxes(&self) -> &[BoxResult] {
        &self.boxes
    }

    /// Write the results to `path`.
    ///
    /// A Gibbs ensemble additionally writes each box beside it, as `box0_`/`box1_`-prefixed files.
    pub fn write_to(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.yaml.as_bytes()).map_err(|source| Error::io(path, source))?;
        if self.boxes.len() > 1 {
            for (i, result) in self.boxes.iter().enumerate() {
                let box_path = box_prefixed_path(path, i);
                std::fs::write(&box_path, result.yaml.as_bytes())
                    .map_err(|source| Error::io(&box_path, source))?;
            }
        }
        Ok(())
    }
}

/// The results for one simulation box.
pub struct BoxResult {
    initial_energy: f64,
    final_energy: f64,
    drift: f64,
    yaml: Arc<str>,
}

impl std::fmt::Debug for BoxResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoxResult")
            .field("initial_energy", &self.initial_energy)
            .field("final_energy", &self.final_energy)
            .field("drift", &self.drift)
            .finish_non_exhaustive()
    }
}

impl BoxResult {
    /// Total potential energy before the first step, in kJ/mol.
    pub const fn initial_energy(&self) -> f64 {
        self.initial_energy
    }

    /// Total potential energy after the last step, in kJ/mol.
    pub const fn final_energy(&self) -> f64 {
        self.final_energy
    }

    /// Discrepancy between the accumulated energy changes and the recomputed total, in kJ/mol.
    ///
    /// A well-behaved simulation keeps this near zero; a large value means some move reports an
    /// energy change that does not match what it did to the system.
    pub const fn drift(&self) -> f64 {
        self.drift
    }

    /// This box's result document.
    pub fn to_yaml(&self) -> &str {
        &self.yaml
    }
}

/// Check that the frame state file describes the same system as the input.
fn validate_aux_header(header: &frame_state::FrameStateHeader, context: &Backend) -> Result<()> {
    let groups = context.groups();
    if header.n_groups as usize != groups.len() {
        return Err(Error::Input(format!(
            "frame state group count ({}) doesn't match input ({})",
            header.n_groups,
            groups.len()
        )));
    }
    if header.n_particles as usize != context.num_particles() {
        return Err(Error::Input(format!(
            "frame state particle count ({}) doesn't match input ({})",
            header.n_particles,
            context.num_particles()
        )));
    }
    for (i, (&(mol_id, capacity), group)) in header.groups.iter().zip(groups).enumerate() {
        if crate::group::MoleculeId::new(mol_id as usize) != group.molecule()
            || capacity as usize != group.capacity()
        {
            return Err(Error::Input(format!(
                "frame state group {i} topology mismatch: aux=({mol_id}, {capacity}), \
                 input=({}, {})",
                group.molecule(),
                group.capacity()
            )));
        }
    }
    Ok(())
}

/// Load a single frame's state into the context from XTC positions + aux data.
fn load_frame_into_context(
    context: &mut Backend,
    particles: &[Particle],
    frame_state: &frame_state::FrameStateFrame,
) -> anyhow::Result<()> {
    use crate::group::GroupSize;

    let sizes: Vec<GroupSize> = frame_state
        .group_sizes
        .iter()
        .zip(context.groups().iter())
        .map(|(&size, g)| GroupSize::from_count(size as usize, g.capacity()))
        .collect();

    context.apply_particles_and_groups(particles, &sizes, &frame_state.quaternions)?;
    context.update(&Change::Everything)?;
    Ok(())
}

/// Replay a trajectory through an input's Hamiltonian and analyses.
///
/// The trajectory's Hamiltonian need not be the one that generated it, which is what makes this
/// useful: analyses run again over saved frames, reweighted if the input carries a penalty bias.
/// `aux` defaults to `traj` with its extension replaced by `.aux`.
///
/// This is a free function rather than a [`Simulation`] method because a replay is driven by the
/// trajectory, not by the input's `propagate` section.
pub fn replay(input: &Path, traj: &Path, aux: Option<&Path>) -> Result<SimulationOutput> {
    use crate::topology::io::NM_TO_ANGSTROM;

    std::fs::File::open(input).map_err(|source| Error::io(input, source))?;
    let source = Source::File(input);
    let medium = source.medium()?;
    let mut context = source.backend(None, &mut source.setup_rng()?)?;
    let mut analyses = source.analyses(&context, Some(&medium), None)?;

    log::info!("{}", medium);
    log::info!("Thermal energy: {:.2} kJ/mol", thermal_energy(&medium));

    // Every frame must be sampled since we can't skip frames in the XTC/aux pair
    analyses.override_frequencies(Frequency::Every(1));

    let aux = aux
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| frame_state::aux_path_from_traj(traj));

    // An analysis pointed at the trajectory being replayed would truncate it mid-read, silently
    // yielding one frame instead of many (issue #60). Refuse before opening the readers.
    for analysis in &analyses {
        for written in analysis.trajectory_outputs() {
            if same_file(&written, traj) || same_file(&written, &aux) {
                return Err(Error::Input(format!(
                    "analysis writes {}, which is the trajectory being replayed; \
                     change the analysis `file:` or drop it from the rerun input",
                    written.display()
                )));
            }
        }
    }

    let mut aux_reader = FrameStateReader::open(&aux)?;

    validate_aux_header(aux_reader.header(), &context)?;

    let mut xtc_reader = molly::XTCReader::open(traj).map_err(|source| Error::io(traj, source))?;
    let mut frame = molly::Frame::default();

    let n_particles = context.num_particles();
    let mut frame_index = 0usize;
    // Pre-allocate and reuse across frames to avoid per-frame heap allocation
    let mut particles = Vec::with_capacity(n_particles);
    let mut aux_frame = frame_state::FrameStateFrame::default();

    // Detect penalty bias for reweighting
    let weight_source = if let Some(penalty) = context.hamiltonian().penalty() {
        let inv_kt = 1.0 / thermal_energy(&medium);
        let state = penalty.state();
        let state_guard = state.read().expect("poisoned lock");
        log::info!(
            "Reweighting enabled: penalty bias detected (\u{0394}g={:.1} kT)",
            state_guard.ln_g_range()
        );
        drop(state_guard);
        // Clone is cheap (state is behind Arc) and avoids borrowing from context
        analysis::reweight::WeightSource::Penalty {
            penalty: Box::new(penalty.clone()),
            inv_thermal_energy: inv_kt,
        }
    } else {
        analysis::reweight::WeightSource::Uniform
    };

    log::info!("Replaying trajectory: {}", traj.display());

    loop {
        match xtc_reader.read_frame(&mut frame) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(source) => return Err(Error::io(traj, source)),
        }

        if frame.natoms() != n_particles {
            return Err(Error::Input(format!(
                "XTC frame {frame_index} has {} atoms, expected {n_particles}",
                frame.natoms()
            )));
        }

        if !aux_reader.read_frame_into(&mut aux_frame)? {
            return Err(Error::Input(format!(
                "aux file ended before XTC at frame {frame_index}"
            )));
        }

        // XTC stores positions in nm with corner at origin; Faunus uses Å with
        // center at origin. Column-major layout: [c0x, c0y, c0z, c1x, ...].
        let cols = frame.boxvec.to_cols_array();
        let box_half = Point::new(
            cols[0] as f64 * NM_TO_ANGSTROM * 0.5,
            cols[4] as f64 * NM_TO_ANGSTROM * 0.5,
            cols[8] as f64 * NM_TO_ANGSTROM * 0.5,
        );

        particles.clear();
        particles.extend((0..n_particles).map(|i| {
            let pos = Point::new(
                frame.positions[3 * i] as f64 * NM_TO_ANGSTROM - box_half.x,
                frame.positions[3 * i + 1] as f64 * NM_TO_ANGSTROM - box_half.y,
                frame.positions[3 * i + 2] as f64 * NM_TO_ANGSTROM - box_half.z,
            );
            Particle::new(aux_frame.atom_ids[i] as usize, pos)
        }));

        load_frame_into_context(&mut context, &particles, &aux_frame)?;
        let weight = weight_source.weight(&context);
        analyses.sample_weighted(&context, frame_index, weight)?;
        frame_index += 1;
    }

    log::info!("Processed {frame_index} frames");

    // `frame_index` counts frames, so the last one carries step `frame_index - 1`, matching
    // what the Monte Carlo loop passes.
    analyses.finalize(&context, frame_index.saturating_sub(1))?;
    analyses.write_to_disk()?;

    let energy = context.hamiltonian().energy(&context, &Change::Everything);
    let energy_summary = BTreeMap::from([
        ("last_frame_energy".to_string(), energy),
        ("frames".to_string(), frame_index as f64),
    ]);

    let mut yaml = yaml_block(&medium, Some("medium"))?;
    yaml.push_str(&yaml_block(&energy_summary, Some("rerun"))?);
    let analysis_yaml = analysis::analyses_to_yaml(&analyses);
    if !analysis_yaml.is_empty() {
        yaml.push_str(&yaml_block(&analysis_yaml, Some("analysis"))?);
    }

    Ok(SimulationOutput {
        yaml: yaml.into(),
        boxes: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::require_finite_initial_energy;

    #[test]
    fn require_finite_initial_energy_rejects_non_finite() {
        assert_eq!(require_finite_initial_energy(-123.4).unwrap(), -123.4);
        assert!(require_finite_initial_energy(f64::INFINITY).is_err());
        assert!(require_finite_initial_energy(f64::NEG_INFINITY).is_err());
        assert!(require_finite_initial_energy(f64::NAN).is_err());
    }
}

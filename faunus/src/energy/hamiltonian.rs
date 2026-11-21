use super::{
    bonded::{IntermolecularBonded, IntramolecularBonded},
    builder::HamiltonianBuilder,
    energyterm::LdForces,
    external_pressure::ExternalPressure,
    nonbonded::{NonbondedMatrix, NonbondedMatrixSplined},
    CellOverlap, EnergyTerm,
};
use crate::time::Instant;
use crate::ObserveContext;
use crate::{
    topology::{DegreesOfFreedom, Topology},
    Change,
};
use interatomic::coulomb::{DebyeLength, Temperature};
use std::cell::Cell;
use std::path::Path;
use std::time::Duration;

/// Trait implemented by structures that can compute
/// and return an energy relevant to some change in the system.
pub trait EnergyChange {
    /// Compute the energy associated with some change in the system.
    fn energy(&self, context: &impl ObserveContext, change: &Change) -> f64;
}

/// Collection of energy terms.
///
/// The Hamiltonian is a collection of energy terms,
/// that itself implements the `EnergyTerm` trait for summing them up.
#[derive(Debug, Clone, Default)]
pub struct Hamiltonian {
    energy_terms: Vec<EnergyTerm>,
    /// Accumulated wall-clock time spent in each energy term's `energy()` call.
    energy_timers: Vec<Cell<Duration>>,
    /// Accumulated wall-clock time spent in each energy term's `update()` call.
    update_timers: Vec<Cell<Duration>>,
}

impl Hamiltonian {
    /// Create a Hamiltonian from the provided HamiltonianBuilder and topology.
    pub(crate) fn new(
        builder: &HamiltonianBuilder,
        topology: &Topology,
        medium: Option<interatomic::coulomb::Medium>,
    ) -> anyhow::Result<Self> {
        let mut hamiltonian = Self::default();
        let temperature = medium.as_ref().map(|m| m.temperature());

        hamiltonian.push(CellOverlap.into());

        // If Ewald is configured, inject real-space pair potential into nonbonded
        // defaults *before* building (and potentially splining) the pair matrix.
        let mut pairpot_builder;
        let pairpot_ref = if let Some(ewald) = builder.ewald() {
            let debye_length = medium.as_ref().and_then(|m| m.debye_length());
            let real_space = ewald.splitting(debye_length)?.real_space_scheme();
            log::info!(
                "Ewald: injected real-space pair potential (cutoff={}, α={:.4})",
                ewald.cutoff,
                real_space.alpha()
            );
            pairpot_builder = builder.nonbonded().cloned().unwrap_or_default();
            pairpot_builder.push_default(super::builder::PairInteraction::CoulombRealSpaceEwald(
                real_space,
            ));
            Some(&pairpot_builder)
        } else {
            builder.nonbonded()
        };

        if let Some(nonbonded_matrix) = pairpot_ref {
            hamiltonian.push(Self::build_nonbonded_term(
                nonbonded_matrix,
                topology,
                medium.clone(),
            )?);

            // Splined nonbonded skips excluded pairs entirely (SR + Coulomb).
            // For titration/alchemical moves, we need the Coulomb part back.
            let any_keep = topology
                .moleculekinds()
                .iter()
                .any(|m| m.keep_excluded_coulomb());
            if any_keep && nonbonded_matrix.has_coulomb() {
                let term = super::excluded_coulomb::ExcludedCoulomb::new(
                    nonbonded_matrix,
                    topology,
                    medium.clone(),
                )?;
                hamiltonian.push(term.into());
                log::info!("Added excluded-pair Coulomb correction");
            }
        }

        if topology
            .moleculekinds()
            .iter()
            .any(|m| m.has_bonded_potentials())
        {
            hamiltonian.push(IntramolecularBonded::default().into());
        }

        // IntermolecularBonded term should only be added if it is actually needed
        if !topology.intermolecular().is_empty() {
            hamiltonian.push(IntermolecularBonded::new(topology));
        }

        for sasa_builder in builder.sasas() {
            hamiltonian.push(sasa_builder.build()?.into());
        }

        if let Some(pressure) = builder.pressure() {
            let temperature = temperature.ok_or_else(|| {
                anyhow::anyhow!("Medium with temperature required for isobaric energy term")
            })?;
            let rt = crate::R_IN_KJ_PER_MOL * temperature;
            hamiltonian.push(ExternalPressure::new(pressure, rt).into());
        }

        Ok(hamiltonian)
    }

    /// Build a nonbonded energy term (plain or splined) from a pair potential builder.
    ///
    /// The `spline`/`cutoff`/`bounding_spheres` knobs are read from the pair
    /// potential builder itself, so an Ewald-augmented clone carries them along.
    fn build_nonbonded_term(
        pairpot_builder: &super::builder::PairPotentialBuilder,
        topology: &Topology,
        medium: Option<interatomic::coulomb::Medium>,
    ) -> anyhow::Result<EnergyTerm> {
        let mut nonbonded = NonbondedMatrix::new(pairpot_builder, topology, medium.clone())?;
        if let Some(spline_opts) = pairpot_builder.spline() {
            let config = spline_opts.to_spline_config();
            let mut splined = NonbondedMatrixSplined::from_nonbonded(
                &nonbonded,
                spline_opts.cutoff,
                Some(config),
            );
            // Without a medium the temperature is unknown; room temperature still
            // gives a meaningful scale for the tolerance.
            let thermal_energy = medium.as_ref().map_or(
                crate::R_IN_KJ_PER_MOL * 298.15,
                crate::simulation::thermal_energy,
            );
            warn_if_spline_inaccurate(&nonbonded, &splined, thermal_energy, topology);
            splined.set_bounding_spheres(spline_opts.bounding_spheres);
            log::info!(
                "Using splined nonbonded potentials (cutoff={}, table_points={})",
                spline_opts.cutoff,
                spline_opts.table_points
            );
            Ok(splined.into())
        } else {
            if let Some(cutoff) = pairpot_builder.cutoff() {
                // The documented exactness condition, enforced rather than trusted:
                // a culled group pair drops *all* its atom-atom terms, so the cutoff
                // must reach past the longest-ranged pair potential.
                let required = nonbonded.required_cull_cutoff();
                anyhow::ensure!(
                    required.is_finite(),
                    "nonbonded.cutoff is set but a pair potential has unbounded range \
                     (e.g. Lennard-Jones or Coulomb without a cutoff); group-to-group \
                     culling cannot be exact. Remove nonbonded.cutoff or give every \
                     potential a cutoff."
                );
                anyhow::ensure!(
                    cutoff >= required,
                    "nonbonded.cutoff ({cutoff} Å) is shorter than the longest pair-potential \
                     range ({required} Å); culling would drop interactions still in range. \
                     Use a cutoff of at least {required} Å."
                );

                if !pairpot_builder.bounding_spheres() {
                    log::warn!(
                        "nonbonded.cutoff ({cutoff} Å) is set but bounding_spheres is false; \
                         no group-to-group culling will be applied."
                    );
                } else if topology.moleculekinds().iter().any(|m| m.has_com()) {
                    nonbonded.set_cutoff(cutoff);
                    log::info!("Group-to-group cutoff culling enabled (cutoff = {cutoff} Å)");
                } else {
                    log::warn!(
                        "nonbonded.cutoff ({cutoff} Å) is set but no molecule has has_com: true; \
                         group-to-group culling will not be applied."
                    );
                }
            }
            Ok(nonbonded.into())
        }
    }

    /// Create a Hamiltonian from a YAML file and topology.
    pub fn from_file(
        filename: impl AsRef<Path>,
        topology: &Topology,
        medium: Option<interatomic::coulomb::Medium>,
    ) -> anyhow::Result<Self> {
        let builder = HamiltonianBuilder::from_file(filename)?;
        builder.validate(topology.atomkinds())?;
        Self::new(&builder, topology, medium)
    }

    /// Appends an energy term to the back of the Hamiltonian.
    pub(crate) fn push(&mut self, term: EnergyTerm) {
        self.energy_terms.push(term);
        self.energy_timers.push(Cell::new(Duration::ZERO));
        self.update_timers.push(Cell::new(Duration::ZERO));
    }

    /// Inserts an energy term at the front for early rejection of infinite energy.
    pub(crate) fn push_front(&mut self, term: EnergyTerm) {
        self.energy_terms.insert(0, term);
        self.energy_timers.insert(0, Cell::new(Duration::ZERO));
        self.update_timers.insert(0, Cell::new(Duration::ZERO));
    }

    /// Removes the first energy term. Returns `None` if empty.
    pub fn pop_front(&mut self) -> Option<EnergyTerm> {
        if self.energy_terms.is_empty() {
            return None;
        }
        self.energy_timers.remove(0);
        self.update_timers.remove(0);
        Some(self.energy_terms.remove(0))
    }

    /// Access the individual energy terms.
    pub fn energy_terms(&self) -> &[EnergyTerm] {
        &self.energy_terms
    }

    /// Return a reference to the Penalty term, if present.
    pub fn penalty(&self) -> Option<&super::penalty::Penalty> {
        #[allow(clippy::wildcard_enum_match_arm)] // searching specifically for the Penalty term
        #[allow(clippy::wildcard_enum_match_arm)] // only one variant is splined
        self.energy_terms.iter().find_map(|t| match t {
            EnergyTerm::Penalty(p) => Some(p),
            _ => None,
        })
    }

    /// Replace the nonbonded energy term with a rebuilt version using an updated real-space scheme.
    ///
    /// Called after Ewald optimization changes α, requiring a new pair matrix and splines.
    pub(crate) fn rebuild_nonbonded(
        &mut self,
        builder: &HamiltonianBuilder,
        topology: &Topology,
        medium: Option<interatomic::coulomb::Medium>,
        real_space: interatomic::coulomb::pairwise::RealSpaceEwald,
    ) -> anyhow::Result<()> {
        let mut pairpot_builder = builder.nonbonded().cloned().unwrap_or_default();
        pairpot_builder.push_default(super::builder::PairInteraction::CoulombRealSpaceEwald(
            real_space,
        ));

        let new_term = Self::build_nonbonded_term(&pairpot_builder, topology, medium)?;

        for term in &mut self.energy_terms {
            // Ewald rebuild replaces the entire nonbonded term; carry over
            // molecule-pair exclusions so tabulated pairs stay excluded.
            if let Some(old) = term.molecule_pair_exclusions() {
                let old = old.to_vec();
                *term = new_term;
                for [a, b] in old {
                    term.exclude_molecule_pair(a, b);
                }
                log::info!("Rebuilt nonbonded interactions with optimized Ewald real-space scheme");
                return Ok(());
            }
        }
        Ok(())
    }

    /// Exclude a molecule-type pair from the nonbonded energy term.
    ///
    /// All inter-group interactions between groups of these two molecule kinds
    /// will be skipped by the nonbonded term. Use when the pair is handled by
    /// another energy term (e.g. [`TabulatedEnergy`]).
    pub(crate) fn exclude_nonbonded_molecule_pair(
        &mut self,
        mol_a: crate::group::MoleculeId,
        mol_b: crate::group::MoleculeId,
    ) {
        self.energy_terms
            .iter_mut()
            .for_each(|t| t.exclude_molecule_pair(mol_a, mol_b));
    }

    /// Compute each term's energy, returning `(name, energy)` pairs.
    pub fn per_term_energies(
        &self,
        context: &impl ObserveContext,
        change: &Change,
    ) -> Vec<(&'static str, f64)> {
        self.energy_terms
            .iter()
            .filter_map(|term| {
                let name = crate::Info::short_name(term)?;
                Some((name, term.energy(context, change)))
            })
            .collect()
    }

    /// Compute per-atom forces from all energy terms.
    ///
    /// Returns a dense vector indexed by absolute particle index, with contributions
    /// from all force-providing terms summed together.
    // Only the Langevin integrator needs forces, and it is gpu-gated.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn forces(&self, context: &impl ObserveContext) -> Vec<crate::Point> {
        self.forces_filtered(context, |_, _| true)
    }

    /// The splined nonbonded term. Its presence puts LD pair and bonded forces on the GPU.
    pub(crate) fn splined_nonbonded(&self) -> Option<&NonbondedMatrixSplined> {
        #[allow(clippy::wildcard_enum_match_arm)] // only one variant is splined
        self.energy_terms.iter().find_map(|t| match t {
            EnergyTerm::NonbondedMatrixSplined(nb) => Some(nb),
            _ => None,
        })
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn ewald_reciprocal(&self) -> Option<&super::EwaldReciprocalEnergy> {
        #[allow(clippy::wildcard_enum_match_arm)]
        self.energy_terms.iter().find_map(|t| match t {
            EnergyTerm::EwaldReciprocal(e) => Some(e.as_ref()),
            _ => None,
        })
    }

    fn ld_terms_of<'a>(
        &'a self,
        context: &'a impl ObserveContext,
        class: LdForces,
    ) -> impl Iterator<Item = (usize, &'a EnergyTerm)> {
        let on_device = self.splined_nonbonded().is_some();
        // Host forces reach LD as rigid-body COM forces and torques, which
        // per-atom-integrated flexible atoms never see
        let has_flexible = context.groups().iter().any(|g| {
            context
                .topology_ref()
                .moleculekind(g.molecule())
                .degrees_of_freedom()
                == DegreesOfFreedom::Free
        });
        self.energy_terms.iter().enumerate().filter(move |(_, t)| {
            let source = t.ld_forces(context, on_device);
            let source = if has_flexible && source == LdForces::Host {
                LdForces::Unsupported
            } else {
                source
            };
            source == class
        })
    }

    /// Indices of terms whose forces LD sums on the host each step. Classify once
    /// per LD block, not per step, since classification scans the groups.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn ld_host_terms(&self, context: &impl ObserveContext) -> Vec<usize> {
        self.ld_terms_of(context, LdForces::Host)
            .map(|(i, _)| i)
            .collect()
    }

    /// Per-atom forces from the [`ld_host_terms`](Self::ld_host_terms), summed for
    /// the overlay onto the GPU's COM forces/torques.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn ld_overlay_forces(
        &self,
        context: &impl ObserveContext,
        host_terms: &[usize],
    ) -> Vec<crate::Point> {
        self.forces_filtered(context, |i, _| host_terms.contains(&i))
    }

    /// Names of terms LD cannot take forces from; LD must refuse to run with any.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn ld_unsupported_terms(&self, context: &impl ObserveContext) -> Vec<&'static str> {
        self.ld_terms_of(context, LdForces::Unsupported)
            // A nameless term must still block LD
            .map(|(_, t)| crate::Info::short_name(t).unwrap_or("unnamed"))
            .collect()
    }

    fn forces_filtered(
        &self,
        context: &impl ObserveContext,
        keep: impl Fn(usize, &EnergyTerm) -> bool,
    ) -> Vec<crate::Point> {
        let mut result: Option<Vec<crate::Point>> = None;
        for (_, term) in self
            .energy_terms
            .iter()
            .enumerate()
            .filter(|(i, t)| keep(*i, t))
        {
            let term_forces = term.forces(context);
            if term_forces.is_empty() {
                continue;
            }
            match &mut result {
                None => result = Some(term_forces),
                Some(acc) => {
                    // Extend if a later term has more particles
                    if term_forces.len() > acc.len() {
                        acc.resize(term_forces.len(), crate::Point::zeros());
                    }
                    for (a, f) in acc.iter_mut().zip(term_forces.iter()) {
                        *a += f;
                    }
                }
            }
        }
        result.unwrap_or_default()
    }

    /// Update internal state due to a change in the system.
    ///
    /// After a system change, the internal state of the energy terms may need to be updated.
    /// For example, in Ewald summation, the reciprocal space energy needs to be updated.
    pub(crate) fn update(
        &mut self,
        context: &impl ObserveContext,
        change: &Change,
    ) -> anyhow::Result<()> {
        for (term, timer) in self.energy_terms.iter_mut().zip(self.update_timers.iter()) {
            let start = Instant::now();
            term.update(context, change)?;
            timer.set(timer.get() + start.elapsed());
        }
        Ok(())
    }

    /// Save backups of all energy terms for later undo.
    ///
    /// Call before applying a move so that terms like Ewald can snapshot
    /// positions of affected particles in their pre-move state.
    pub(crate) fn save_backups(&mut self, change: &Change, context: &impl ObserveContext) {
        self.energy_terms
            .iter_mut()
            .for_each(|term| term.save_backup(change, context));
    }

    /// Update with backup for later undo on MC reject.
    pub(crate) fn update_with_backup(
        &mut self,
        context: &impl ObserveContext,
        change: &Change,
    ) -> anyhow::Result<()> {
        self.save_backups(change, context);
        self.update(context, change)
    }

    /// Restore all energy terms from their internal backups.
    pub(crate) fn undo(&mut self) {
        self.energy_terms.iter_mut().for_each(|term| term.undo());
    }

    /// Drop all energy term backups.
    pub(crate) fn discard_backup(&mut self) {
        self.energy_terms
            .iter_mut()
            .for_each(|term| term.discard_backup());
    }

    /// Per-term information as a YAML sequence of single-key mappings (`- name: info`).
    ///
    /// A sequence rather than a map so repeated terms (e.g. two `!Sasa`) each get their
    /// own entry instead of colliding on a shared `short_name` key; this mirrors the
    /// `energy:` list input and the `analysis:` output convention. Only includes terms
    /// that provide information via `EnergyTerm::to_yaml()`.
    pub fn info_to_yaml(&self) -> yaml_serde::Value {
        let seq: Vec<yaml_serde::Value> = self
            .energy_terms
            .iter()
            .filter_map(|term| {
                let name = crate::Info::short_name(term)?;
                let info = term.to_yaml()?;
                Some(yaml_map! { yaml_serde::Value::String(name.to_string()) => info })
            })
            .collect();
        yaml_serde::Value::Sequence(seq)
    }

    /// Per-term energy timing as a YAML sequence of single-key mappings
    /// (`- name: percentage of total`).
    ///
    /// A sequence for the same reason as [`info_to_yaml`](Self::info_to_yaml): repeated
    /// terms must not collide on a shared name key. Reports both `energy()` and
    /// `update()` time per term.
    pub fn timing_to_yaml(&self) -> yaml_serde::Value {
        let total: f64 = self
            .energy_timers
            .iter()
            .chain(self.update_timers.iter())
            .map(|t| t.get().as_secs_f64())
            .sum();
        let seq: Vec<yaml_serde::Value> = self
            .energy_terms
            .iter()
            .zip(self.energy_timers.iter())
            .zip(self.update_timers.iter())
            .filter_map(|((term, e_timer), u_timer)| {
                let name = crate::Info::short_name(term)?;
                let to_pct = |secs: f64| {
                    if total > 0.0 {
                        (secs / total * 10000.0).round() / 100.0
                    } else {
                        0.0
                    }
                };
                let e_secs = e_timer.get().as_secs_f64();
                let u_secs = u_timer.get().as_secs_f64();
                let e_pct = to_pct(e_secs);
                let u_pct = to_pct(u_secs);
                // Skip only when both components round away, so a term never shows a
                // non-zero `total` over an all-zero breakdown. `total` itself comes from
                // the raw seconds, not `e_pct + u_pct`, so it carries no float noise
                // (e.g. 0.15000000000000002).
                if e_pct == 0.0 && u_pct == 0.0 {
                    return None;
                }
                let breakdown = yaml_map! {
                    "total" => to_pct(e_secs + u_secs),
                    "energy" => e_pct,
                    "update" => u_pct,
                };
                Some(yaml_map! { yaml_serde::Value::String(name.to_string()) => breakdown })
            })
            .collect();
        yaml_serde::Value::Sequence(seq)
    }
    /// Add energy terms that require a live context (particles already placed).
    ///
    /// Must be called after `Hamiltonian::new()` and particle insertion.
    pub(crate) fn finalize(
        &mut self,
        builder: &HamiltonianBuilder,
        context: &impl ObserveContext,
        medium: Option<&interatomic::coulomb::Medium>,
    ) -> anyhow::Result<()> {
        let require_thermal_energy = |term: &str| -> anyhow::Result<f64> {
            let m = medium.ok_or_else(|| {
                anyhow::anyhow!("Medium with temperature required for {term} energy term")
            })?;
            Ok(crate::R_IN_KJ_PER_MOL * m.temperature())
        };

        if let Some(penalty_builder) = builder.penalty() {
            let thermal_energy = require_thermal_energy("penalty")?;
            // Front placement: out-of-range CVs short-circuit before expensive terms
            self.push_front(penalty_builder.build(context, thermal_energy)?.into());
        }
        for cb in builder.constrains() {
            self.push(cb.build(context)?.into());
        }
        for eb in builder.custom_externals() {
            self.push(eb.build()?.into());
        }
        for pb in builder.custom_pairs() {
            self.push(pb.build(context)?.into());
        }

        if let Some(pm_builder) = builder.polymer_depletion() {
            let thermal_energy = require_thermal_energy("polymer_depletion")?;
            self.push(pm_builder.build(context, thermal_energy)?.into());
        }
        if let Some(ct_builder) = builder.contact_tessellation() {
            let ct = super::contact_tessellation::ContactTessellationEnergy::from_builder(
                ct_builder, context,
            )?;
            self.push(ct.into());
        }
        // Build tabulated energy entries (6D molecule-molecule and 3D molecule-atom)
        let (tab6d, tab3d) = (builder.tabulated6d(), builder.tabulated3d());
        if tab6d.is_some() || tab3d.is_some() {
            let inv_thermal_energy = 1.0 / require_thermal_energy("tabulated")?;
            let topology = context.topology();
            let mut tab_entries = Vec::new();
            if let Some(tab6d) = tab6d {
                tab_entries.extend(tab6d.build_entries(&topology, inv_thermal_energy)?);
            }
            if let Some(tab3d) = tab3d {
                tab_entries.extend(tab3d.build_entries(&topology, inv_thermal_energy)?);
            }
            let tab = super::TabulatedEnergy::new(tab_entries, inv_thermal_energy);
            for (mol_a, mol_b) in tab.molecule_pairs() {
                self.exclude_nonbonded_molecule_pair(
                    crate::group::MoleculeId::new(mol_a),
                    crate::group::MoleculeId::new(mol_b),
                );
                log::info!(
                    "Excluded molecule pair ({}, {}) from nonbonded (handled by tabulated energy)",
                    mol_a,
                    mol_b
                );
            }
            self.push(tab.into());
        }
        if let Some(ewald_builder) = builder.ewald() {
            let medium = medium
                .ok_or_else(|| anyhow::anyhow!("Ewald requires a medium with permittivity"))?;
            let splitting = ewald_builder.splitting(medium.debye_length())?;
            let ewald = super::EwaldReciprocalEnergy::from_splitting(
                ewald_builder,
                &splitting,
                context,
                medium,
            )?;
            // The injected scheme came from this splitting; they differ only if `optimize` moved α
            if ewald.real_space_scheme() != splitting.real_space_scheme() {
                self.rebuild_nonbonded(
                    builder,
                    context.topology_ref(),
                    Some(medium.clone()),
                    ewald.real_space_scheme(),
                )?;
            }
            self.push(ewald.into());
        }
        Ok(())
    }
}

impl<T: Into<Vec<EnergyTerm>>> From<T> for Hamiltonian {
    fn from(energy_terms: T) -> Self {
        let terms: Vec<EnergyTerm> = energy_terms.into();
        let n = terms.len();
        Self {
            energy_terms: terms,
            energy_timers: vec![Cell::new(Duration::ZERO); n],
            update_timers: vec![Cell::new(Duration::ZERO); n],
        }
    }
}

impl EnergyChange for Hamiltonian {
    /// Compute the energy of the Hamiltonian associated with a change in the system.
    /// The energy is returned in the units of kJ/mol.
    fn energy(&self, context: &impl ObserveContext, change: &Change) -> f64 {
        let mut sum: f64 = 0.0;
        for (term, timer) in self.energy_terms.iter().zip(self.energy_timers.iter()) {
            let start = Instant::now();
            let energy = term.energy(context, change);
            timer.set(timer.get() + start.elapsed());
            if energy.is_finite() {
                sum += energy;
            } else {
                return energy;
            }
        }
        sum
    }
}

/// Warns about the atom pair whose spline table departs most from the exact potential.
fn warn_if_spline_inaccurate(
    exact: &NonbondedMatrix,
    splined: &NonbondedMatrixSplined,
    thermal_energy: f64,
    topology: &Topology,
) {
    let Some(worst) =
        super::nonbonded::spline_accuracy::worst_deviation(exact, splined, thermal_energy)
    else {
        return;
    };
    let name = |i: usize| topology.atomkinds()[i].name();
    let remedy = if worst.distance < worst.table_edge {
        format!(
            "This is below the table's inner edge at {:.2} Å, where it extrapolates linearly.",
            worst.table_edge
        )
    } else {
        "Increase `table_points` or try another `grid_type`.".to_string()
    };
    log::warn!(
        "Spline table for {}–{} is off by {:.3} kT at r = {:.2} Å \
         (exact {:.4}, table {:.4} kJ/mol). {remedy}",
        name(worst.atom_kinds.0),
        name(worst.atom_kinds.1),
        worst.error() / thermal_energy,
        worst.distance,
        worst.exact,
        worst.splined,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `timing_to_yaml` reports each term as a `{total, energy, update}` mapping of
    /// numbers (never the old packed string), with `total` derived from the raw
    /// seconds rather than the sum of two rounded percentages — so it carries no
    /// float noise such as `0.15000000000000002`. This output is scrubbed from every
    /// integration fixture, so this is its only guard.
    #[test]
    fn timing_to_yaml_is_a_numeric_breakdown_without_rounding_noise() {
        let h = Hamiltonian::from(vec![
            EnergyTerm::CellOverlap(CellOverlap),
            EnergyTerm::CellOverlap(CellOverlap),
        ]);
        // Second term dominates, so the first rounds to small percentages whose naive
        // sum (0.1 + 0.05) is the noisy 0.15000000000000002 the fix removes.
        h.energy_timers[0].set(Duration::from_secs_f64(0.1));
        h.update_timers[0].set(Duration::from_secs_f64(0.05));
        h.energy_timers[1].set(Duration::from_secs_f64(100.0));

        let yaml = h.timing_to_yaml();
        let entry = &yaml[0]["cell_overlap"];

        assert_eq!(entry["energy"].as_f64(), Some(0.1));
        assert_eq!(entry["update"].as_f64(), Some(0.05));
        // Clean 0.15, not 0.1 + 0.05 == 0.15000000000000002.
        assert_eq!(entry["total"].as_f64(), Some(0.15));
    }

    use crate::energy::ld_case::LdCase;

    fn ld_unsupported(case: LdCase) -> Vec<&'static str> {
        use crate::WithHamiltonian;
        let context = case.context();
        let names = context.hamiltonian().ld_unsupported_terms(&context);
        names
    }

    #[test]
    fn ld_accepts_splined_nonbonded_and_bonded() {
        assert_eq!(ld_unsupported(LdCase::default()), Vec::<&str>::new());
        let flexible = LdCase {
            dimer_dof: "Free",
            ..Default::default()
        };
        // Excluded Coulomb still blocks the flexible dimer; bonded must not.
        assert_eq!(ld_unsupported(flexible), ["excluded_coulomb"]);
    }

    #[test]
    fn ld_names_terms_without_forces() {
        let external = LdCase {
            extra: r#"    - !CustomExternal {selection: "all", function: "0.1 * z"}"#,
            ..Default::default()
        };
        assert_eq!(ld_unsupported(external), ["custom_external"]);
    }

    /// PBC Ewald has an LD force path; IPBC has none.
    #[test]
    fn ld_accepts_ewald_only_with_pbc() {
        let ewald = |extra| LdCase {
            extra,
            ..Default::default()
        };
        assert_eq!(
            ld_unsupported(ewald("    - !Ewald {cutoff: 12.0, accuracy: 1e-5}")),
            Vec::<&str>::new()
        );
        assert_eq!(
            ld_unsupported(ewald(
                "    - !Ewald {cutoff: 12.0, accuracy: 1e-5, policy: IPBC}"
            )),
            ["ewald_reciprocal"]
        );
    }

    /// With splines the k-space forces run on-device per atom, so flexible molecules
    /// get them too; the host overlay only reaches rigid bodies.
    #[test]
    fn ld_takes_ewald_on_device_with_splines() {
        use crate::WithHamiltonian;
        let ewald = "    - !Ewald {cutoff: 12.0, accuracy: 1e-5}";
        let host_terms = |spline| {
            let context = LdCase {
                spline,
                extra: ewald,
                ..Default::default()
            }
            .context();
            let host = context.hamiltonian().ld_host_terms(&context);
            host.len()
        };
        // Unsplined: nonbonded and Ewald both go through the CPU force path
        assert_eq!(host_terms(false), 2);
        assert_eq!(host_terms(true), 0);

        let flexible = |extra| LdCase {
            dimer_dof: "Free",
            extra,
            ..Default::default()
        };
        assert_eq!(
            ld_unsupported(flexible(ewald)),
            ld_unsupported(flexible(""))
        );
    }

    /// The GPU bonded kernels skip non-harmonic bonds, so a moving FENE bond has no force.
    #[test]
    fn ld_refuses_bonds_the_device_skips() {
        let fene = |dimer_dof| LdCase {
            dimer_dof,
            bond: "!FENE {k: 1.0, req: 3.0, rmax: 5.0}",
            ..Default::default()
        };
        assert_eq!(ld_unsupported(fene("Rigid")), Vec::<&str>::new());
        assert_eq!(
            ld_unsupported(fene("Free")),
            ["excluded_coulomb", "intramolecular"]
        );
    }

    /// Without splines LD takes the CPU-force path: no bonded forces, and pair forces
    /// reach only rigid bodies.
    #[test]
    fn ld_refuses_moving_bonds_without_splines() {
        let unsplined = |dimer_dof| LdCase {
            dimer_dof,
            spline: false,
            ..Default::default()
        };
        assert_eq!(ld_unsupported(unsplined("Rigid")), Vec::<&str>::new());
        assert_eq!(
            ld_unsupported(unsplined("Free")),
            ["nonbonded", "excluded_coulomb", "intramolecular"]
        );
    }

    #[test]
    fn ld_host_terms_select_custom_pair() {
        use crate::WithHamiltonian;
        let has_overlay = |extra| {
            let context = LdCase {
                extra,
                ..Default::default()
            }
            .context();
            let host = context.hamiltonian().ld_host_terms(&context);
            !host.is_empty()
        };
        assert!(has_overlay(
            r#"    - !CustomPair {selections: ["molecule dimer", "molecule probe"], function: "0.5 * r"}"#
        ));
        assert!(!has_overlay(""));
    }
}

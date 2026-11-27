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

//! Åkesson mean-field correction for the charge outside a slit's minimum-image box
//! (Greberg, Kjellander & Åkesson, Mol. Phys. 87, 407 (1996)).

use super::slab_potential::{square_base_width, SlabGrid, SlabKernel};
use crate::cell::{Cell, Shape};
use crate::change::{Change, GroupChange};
use crate::context::ObserveContext;
use anyhow::{Context, Result};
use interatomic::coulomb::{Medium, Temperature};
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

/// The pair Coulomb scheme shared by every charged atom pair: all the kernel choice needs.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CoulombScheme {
    pub kind: CoulombSchemeKind,
    pub cutoff: f64,
    /// Inverse Debye length (Å⁻¹), if the scheme screens.
    pub kappa: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CoulombSchemeKind {
    Plain,
    /// Any other scheme, by YAML tag, for messages.
    Other(&'static str),
}

/// How the potential of the exterior charge is computed. The choice follows from the pair
/// scheme: the mean field must supply exactly the Coulomb the pair sum leaves out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum MeanFieldKernel {
    /// Pair Coulomb covers the whole minimum-image box; the field is the charge outside the
    /// square (Greberg Eq. 15).
    Greberg,
}

impl MeanFieldKernel {
    pub(crate) fn select(scheme: Option<&CoulombScheme>, cell: &Cell) -> Result<Self> {
        let Some(scheme) = scheme else {
            anyhow::bail!("Åkesson: requires a pair Coulomb scheme");
        };
        if scheme.kappa.is_some() {
            anyhow::bail!("Åkesson: the bare-Coulomb exterior field cannot be Debye screened");
        }
        // A cuboid is also periodic in z, so its exterior is not one square sheet per slab.
        let Cell::Slit(slit) = cell else {
            anyhow::bail!("Åkesson: requires a !Slit cell");
        };
        let width = square_base_width(cell).context("Åkesson")?;
        let height = slit.bounding_box().map_or(f64::INFINITY, |b| b.z);
        // Laterally periodic, walls in z: the farthest minimum-image pair is half a lateral
        // diagonal sideways and the full height apart.
        let span = (0.5 * width * width + height * height).sqrt();
        if scheme.kind == CoulombSchemeKind::Plain && scheme.cutoff >= span {
            return Ok(Self::Greberg);
        }
        anyhow::bail!(
            "Åkesson: needs plain Coulomb with cutoff ≥ {span:.1} Å (the minimum-image span)"
        )
    }
}

/// When the exterior field φ_ext(z) is recomputed from the sampled σ(z).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum AkessonUpdate {
    /// Sample σ(z) every step; refresh φ and rewrite the file every `n` steps.
    Every(NonZeroUsize),
    /// Use the σ(z) loaded from file; never sample, refresh or write.
    Fixed,
}

/// YAML input for the [`Akesson`] energy term.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkessonBuilder {
    /// Slab width (Å).
    pub resolution: f64,
    /// σ(z) profile: loaded if present, rewritten on every refresh.
    pub file: PathBuf,
    pub update: AkessonUpdate,
}

/// Mean-field energy of every charge in the field φ_ext(z) of the charge outside the box.
#[derive(Clone, Debug)]
pub struct Akesson {
    kernel: MeanFieldKernel,
    grid: SlabGrid,
    /// σ(z) as per-slab sums of the areal charge density (e·Å⁻²) over `samples` steps. Every
    /// slab is sampled together, so one count serves all.
    sigma_sum: Vec<f64>,
    samples: u64,
    /// φ_ext at bin centres (kT/e). Changes only in `advance`, so between refreshes a move's
    /// energy change is Σ q Δφ of the moved charges alone.
    phi: Vec<f64>,
    update: AkessonUpdate,
    /// `file` as given in the input; a relative one is resolved by [`anchor`](Self::anchor).
    requested_file: PathBuf,
    /// Where σ(z) is read from and written to; `None` until a relative path is anchored, so a
    /// runner that forgets cannot fall back to an arbitrary working directory.
    file: Option<PathBuf>,
    thermal_energy: f64,
}

impl Akesson {
    /// Builds the term. An absolute `file` is loaded now; a relative one waits for the runner
    /// to [`anchor`](Self::anchor) it beside the state.
    pub(crate) fn new(
        builder: &AkessonBuilder,
        context: &impl ObserveContext,
        medium: &Medium,
        scheme: Option<&CoulombScheme>,
    ) -> Result<Self> {
        let cell = context.cell();
        let kernel = MeanFieldKernel::select(scheme, cell)?;
        // The correction table is cached once here rather than rebuilt on every refresh.
        let grid = SlabGrid::from_cell(
            cell,
            builder.resolution,
            SlabKernel::unscreened(medium.bjerrum_length()),
        )?
        .with_finite_box_correction()?;
        let mut term = Self {
            sigma_sum: vec![0.0; grid.n_bins()],
            samples: 0,
            phi: vec![0.0; grid.n_bins()],
            kernel,
            grid,
            update: builder.update,
            requested_file: builder.file.clone(),
            file: None,
            thermal_energy: crate::R_IN_KJ_PER_MOL * medium.temperature(),
        };
        if builder.file.is_absolute() {
            term.load(builder.file.clone())?;
        }
        Ok(term)
    }

    /// Resolve a relative `file` in `dir` (the state's directory), `box{i}_`-prefixed for a
    /// Gibbs box, and reload σ(z) from there. Restarts and boxes then keep their own profile.
    pub(crate) fn anchor(&mut self, dir: &Path, box_index: Option<usize>) -> Result<()> {
        if self.requested_file.is_absolute() {
            return Ok(());
        }
        let path = dir.join(&self.requested_file);
        self.load(match box_index {
            Some(i) => crate::simulation::box_prefixed_path(&path, i),
            None => path,
        })
    }

    /// Replace σ(z) by the profile in `file`, or an empty one if it is absent and `Every` will
    /// build one. `Fixed` has nothing to build from, so a missing profile is an error.
    fn load(&mut self, file: PathBuf) -> Result<()> {
        (self.sigma_sum, self.samples) = if file.exists() {
            load_profile(&file, &self.grid)
                .with_context(|| format!("Åkesson: reading {}", file.display()))?
        } else {
            (vec![0.0; self.grid.n_bins()], 0)
        };
        if self.samples == 0 {
            if self.update == AkessonUpdate::Fixed {
                anyhow::bail!(
                    "Åkesson: `update: Fixed` needs a sampled σ(z) profile, but {} has none; \
                     equilibrate one with `update: !Every N` first",
                    file.display()
                );
            }
            log::warn!("Åkesson: no σ(z) in {}; starting empty", file.display());
        }
        self.phi = self.exterior_potential();
        self.file = Some(file);
        self.report();
        Ok(())
    }

    fn report(&self) {
        log::info!(
            "Åkesson: {:?} kernel (a = {:.1} Å), {} bins × {:.2} Å, update {:?}, σ(z) from {} ({} samples)",
            self.kernel,
            0.5 * self.grid.lateral_extent(),
            self.grid.n_bins(),
            self.grid.bin_width(),
            self.update,
            self.file.as_deref().unwrap_or(&self.requested_file).display(),
            self.samples,
        );
    }

    /// Energy (kJ/mol) of the charges `change` touches.
    pub(crate) fn energy(&self, context: &impl ObserveContext, change: &Change) -> f64 {
        let atom = |i: usize| context.atom_charge(i) * self.potential(context.position(i).z);
        let group_energy = |group_index: usize| -> f64 {
            context.groups()[group_index].iter_active().map(atom).sum()
        };
        let changed = |group_index: usize, group_change: &GroupChange| -> f64 {
            match group_change {
                GroupChange::None => 0.0,
                // φ is fixed between refreshes, so only the changed slots contribute to ΔU.
                GroupChange::PartialUpdate(rels) | GroupChange::UpdateIdentity(rels) => {
                    let start = context.groups()[group_index].start();
                    rels.iter().map(|rel| atom(start + rel.get())).sum()
                }
                // Summing what is active in each state keeps ΔU right across a size change.
                GroupChange::RigidBody
                | GroupChange::Resize(_)
                | GroupChange::ResizeExcludeIntra(_)
                | GroupChange::ResizePartial(..)
                | GroupChange::AtomicShrink { .. } => group_energy(group_index),
            }
        };
        let reduced: f64 = match change {
            Change::None => 0.0,
            Change::Everything | Change::Volume(..) => {
                (0..context.groups().len()).map(group_energy).sum()
            }
            Change::SingleGroup(group_index, group_change) => changed(*group_index, group_change),
            Change::Groups(changes) => changes
                .iter()
                .map(|(group_index, group_change)| changed(*group_index, group_change))
                .sum(),
        };
        self.thermal_energy * reduced
    }

    /// φ_ext(z) (kT/e), linear between bin centres and flat beyond the outermost ones.
    fn potential(&self, z: f64) -> f64 {
        let width = self.grid.bin_width();
        let x = ((z - self.grid.bin_center(0)) / width).clamp(0.0, (self.phi.len() - 1) as f64);
        let lower = (x.floor() as usize).min(self.phi.len().saturating_sub(2));
        let t = x - lower as f64;
        self.phi[lower] * (1.0 - t) + self.phi[(lower + 1).min(self.phi.len() - 1)] * t
    }

    /// φ_ext from the mean σ(z); zero before any sample.
    fn exterior_potential(&self) -> Vec<f64> {
        if self.samples == 0 {
            return vec![0.0; self.grid.n_bins()];
        }
        let to_charge = self.grid.area() / self.samples as f64;
        let slab_charges: Vec<f64> = self.sigma_sum.iter().map(|s| s * to_charge).collect();
        self.grid.exterior_profile(&slab_charges)
    }

    /// True unless `Fixed`: the field then changes during a run.
    pub(crate) const fn is_adaptive(&self) -> bool {
        matches!(self.update, AkessonUpdate::Every(_))
    }

    /// Sample σ(z) and, on a refresh step, recompute φ and rewrite the file. Returns the energy
    /// change the refresh caused (kJ/mol): external work, outside any move's ΔU.
    pub(crate) fn advance(&mut self, context: &impl ObserveContext, step: usize) -> Result<f64> {
        if self.file.is_none() {
            anyhow::bail!(
                "Åkesson: {} was never anchored to the run's state directory",
                self.requested_file.display()
            );
        }
        // Bins, area and φ are frozen at construction, so after a committed volume move both the
        // energy and σ would be wrong; stop rather than run on.
        self.grid
            .ensure_cell_unchanged(context.cell())
            .context("Åkesson")?;
        let AkessonUpdate::Every(every) = self.update else {
            return Ok(0.0);
        };
        let inv_area = 1.0 / self.grid.area();
        for group in context.groups() {
            for i in group.iter_active() {
                let bin = self.grid.bin_index(context.position(i).z);
                self.sigma_sum[bin] += context.atom_charge(i) * inv_area;
            }
        }
        self.samples += 1;
        if !step.is_multiple_of(every.get()) {
            return Ok(0.0);
        }
        let before = self.energy(context, &Change::Everything);
        self.phi = self.exterior_potential();
        self.save()?;
        Ok(self.energy(context, &Change::Everything) - before)
    }

    /// Write samples taken since the last refresh, so a checkpoint between refreshes keeps them.
    /// φ is left alone: it changes only on refresh steps, where the work is booked.
    pub(crate) fn flush(&self) -> Result<()> {
        if self.is_adaptive() && self.samples > 0 {
            self.save()?;
        }
        Ok(())
    }

    /// Write via a sibling temp file, so an interrupted run never leaves a truncated profile.
    fn save(&self) -> Result<()> {
        use std::fmt::Write as _;
        let mut csv = String::with_capacity(32 * (self.sigma_sum.len() + 1));
        csv.push_str(CSV_HEADER);
        csv.push('\n');
        for (bin, sum) in self.sigma_sum.iter().enumerate() {
            let mean = sum / self.samples as f64;
            let _ = writeln!(
                csv,
                "{},{mean:e},{}",
                self.grid.bin_center(bin),
                self.samples
            );
        }
        let file = self
            .file
            .as_ref()
            .context("Åkesson: no anchored profile file")?;
        let tmp = file.with_extension("csv.tmp");
        file.parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(&tmp, csv))
            .and_then(|()| std::fs::rename(&tmp, file))
            .with_context(|| format!("Åkesson: writing {}", file.display()))
    }
}

const CSV_HEADER: &str = "z/Å,σ/e·Å⁻²,samples";

/// Read `z, mean σ, samples` rows. The z column must match the grid: a profile from another
/// resolution or box would silently put σ in the wrong slabs.
fn load_profile(path: &Path, grid: &SlabGrid) -> Result<(Vec<f64>, u64)> {
    let text = std::fs::read_to_string(path)?;
    let rows: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && *line != CSV_HEADER)
        .collect();
    if rows.len() != grid.n_bins() {
        anyhow::bail!(
            "{} rows, but the grid has {} bins",
            rows.len(),
            grid.n_bins()
        );
    }
    let mut samples = None;
    let mut sigma_sum = Vec::with_capacity(rows.len());
    for (bin, row) in rows.iter().enumerate() {
        let fields: Vec<&str> = row.split(',').map(str::trim).collect();
        let [z, sigma, n] = fields[..] else {
            anyhow::bail!("row {}: expected 3 columns, got '{row}'", bin + 1);
        };
        let (z, sigma, n): (f64, f64, u64) = (z.parse()?, sigma.parse()?, n.parse()?);
        if (z - grid.bin_center(bin)).abs() > 1e-6 * grid.bin_width() {
            anyhow::bail!(
                "row {}: z = {z} Å, but the grid's bin centre is {} Å",
                bin + 1,
                grid.bin_center(bin)
            );
        }
        if *samples.get_or_insert(n) != n {
            anyhow::bail!("row {}: sample count differs from the first row", bin + 1);
        }
        sigma_sum.push(sigma * n as f64);
    }
    Ok((sigma_sum, samples.unwrap_or(0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{get_medium_str, Backend};
    use crate::cell::{Cuboid, Cylinder, Slit};
    use crate::energy::slab_potential::brute_force_exterior;

    // Two unit ions, each its own group, at bin centres of a 20×20×40 slit (1 Å bins:
    // centres at −19.5, −18.5, …). Plain Coulomb spans the box.
    const IONS: &str = r#"
atoms:
  - {name: I, mass: 1.0, charge: 1.0, sigma: 1.0}
molecules:
  - {name: ION, atoms: [I]}
system:
  cell: !Slit [20.0, 20.0, 40.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: []
  blocks:
    - molecule: ION
      N: 2
      insert: !Manual [[0.0, 0.0, Z0], [0.0, 0.0, Z1]]
propagate: {seed: !Fixed 1, criterion: Metropolis, steps: 0, collections: []}
"#;

    fn context_at(z0: f64, z1: f64) -> Backend {
        let yaml = IONS
            .replace("Z0", &format!("{z0:?}"))
            .replace("Z1", &format!("{z1:?}"));
        Backend::from_yaml_str(&yaml, None, &mut rand::thread_rng()).unwrap()
    }

    fn context() -> Backend {
        context_at(-6.5, -14.5)
    }

    /// σ = 0.01 e·Å⁻² in the slab at z = −9.5 only, one sample.
    fn single_slab_profile(dir: &Path) -> PathBuf {
        let file = dir.join("akesson.csv");
        let mut csv = format!("{CSV_HEADER}\n");
        for i in 0..40 {
            let z = -19.5 + i as f64;
            let sigma = if i == 10 { 0.01 } else { 0.0 };
            csv.push_str(&format!("{z},{sigma},1\n"));
        }
        std::fs::write(&file, csv).unwrap();
        file
    }

    fn term(file: &Path, update: AkessonUpdate, context: &Backend) -> Akesson {
        let builder = AkessonBuilder {
            resolution: 1.0,
            file: file.to_path_buf(),
            update,
        };
        let medium = get_medium_str(IONS).unwrap();
        Akesson::new(&builder, context, &medium, Some(&plain(1e4))).unwrap()
    }

    /// kT·σ·g(d): energy of a unit ion `d` from a slab of areal charge `sigma`, with g from the
    /// exterior integral rather than the kernel under test.
    fn exterior_energy(sigma: f64, d: f64) -> f64 {
        let bjerrum = get_medium_str(IONS).unwrap().bjerrum_length();
        crate::R_IN_KJ_PER_MOL * 300.0 * sigma * brute_force_exterior(bjerrum, 10.0, d)
    }

    fn assert_close(got: f64, want: f64) {
        assert!((got - want).abs() < 1e-2 * want.abs(), "{got} vs {want}");
    }

    #[test]
    fn loaded_profile_acts_on_each_moved_charge() {
        // The ions sit 3 and 5 Å from the charged slab.
        let dir = tempfile::tempdir().unwrap();
        let file = single_slab_profile(dir.path());
        let context = context();
        let term = term(&file, AkessonUpdate::Fixed, &context);
        let (near, far) = (exterior_energy(0.01, 3.0), exterior_energy(0.01, 5.0));
        let moved_near = Change::SingleGroup(0, GroupChange::RigidBody);
        assert_close(term.energy(&context, &moved_near), near);
        assert_close(term.energy(&context, &Change::Everything), near + far);
        assert_eq!(term.energy(&context, &Change::None), 0.0);
    }

    #[test]
    fn a_partial_update_counts_only_the_changed_slots() {
        // Both ions in one atomic group; slot 1 (5 Å from the charged slab) is the one moved.
        let dir = tempfile::tempdir().unwrap();
        let file = single_slab_profile(dir.path());
        let yaml = IONS
            .replace(
                "{name: ION, atoms: [I]}",
                "{name: ION, atoms: [I], atomic: true}",
            )
            .replace("Z0", "-6.5")
            .replace("Z1", "-14.5");
        let context = Backend::from_yaml_str(&yaml, None, &mut rand::thread_rng()).unwrap();
        let term = term(&file, AkessonUpdate::Fixed, &context);
        let moved = GroupChange::PartialUpdate(vec![crate::group::RelIndex::new(1)]);
        assert_close(
            term.energy(&context, &Change::SingleGroup(0, moved)),
            exterior_energy(0.01, 5.0),
        );
    }

    #[test]
    fn anchoring_reads_the_profile_beside_the_state_box_prefixed() {
        // A relative `file` follows the state, so Gibbs boxes and restarts keep their own σ(z).
        let dir = tempfile::tempdir().unwrap();
        let written = single_slab_profile(dir.path());
        std::fs::rename(&written, dir.path().join("box1_akesson.csv")).unwrap();
        let context = context();
        let mut term = term(Path::new("akesson.csv"), AkessonUpdate::Fixed, &context);
        assert_eq!(term.energy(&context, &Change::Everything), 0.0);

        term.anchor(dir.path(), Some(1)).unwrap();
        let expected = exterior_energy(0.01, 3.0) + exterior_energy(0.01, 5.0);
        assert_close(term.energy(&context, &Change::Everything), expected);
    }

    #[test]
    fn field_is_linear_between_bin_centres_and_flat_beyond_the_last() {
        // z = −6.0 lies midway between the centres 3 and 4 Å from the charged slab; z = −19.9
        // is below the first centre (−19.5, 10 Å away).
        let dir = tempfile::tempdir().unwrap();
        let file = single_slab_profile(dir.path());
        let context = context_at(-6.0, -19.9);
        let term = term(&file, AkessonUpdate::Fixed, &context);
        let g = |d: f64| exterior_energy(0.01, d);
        let first = term.energy(&context, &Change::SingleGroup(0, GroupChange::RigidBody));
        let second = term.energy(&context, &Change::SingleGroup(1, GroupChange::RigidBody));
        assert_close(first, 0.5 * (g(3.0) + g(4.0)));
        assert_close(second, g(10.0));
    }

    #[test]
    fn refresh_returns_the_energy_it_adds_and_writes_the_profile() {
        // One sample: σ = 1/400 e·Å⁻² in each ion's slab, 8 Å apart. Each ion then feels
        // σ·(g(0) + g(8)); U was 0 before, so the returned work is all of U.
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("akesson.csv");
        let context = context();
        let every = AkessonUpdate::Every(NonZeroUsize::new(1).unwrap());
        let mut term = term(&file, every, &context);
        assert_eq!(term.energy(&context, &Change::Everything), 0.0);

        let work = term.advance(&context, 1).unwrap();
        let sigma = 1.0 / 400.0;
        assert_close(
            work,
            2.0 * (exterior_energy(sigma, 0.0) + exterior_energy(sigma, 8.0)),
        );
        assert_eq!(work, term.energy(&context, &Change::Everything));
        assert!(file.exists());
    }

    #[test]
    fn fixed_mode_without_a_profile_is_an_error() {
        // Running on with φ = 0 would silently switch the correction off.
        let dir = tempfile::tempdir().unwrap();
        let context = context();
        let builder = AkessonBuilder {
            resolution: 1.0,
            file: dir.path().join("missing.csv"),
            update: AkessonUpdate::Fixed,
        };
        let medium = get_medium_str(IONS).unwrap();
        let err = Akesson::new(&builder, &context, &medium, Some(&plain(1e4))).unwrap_err();
        assert!(format!("{err:#}").contains("Fixed"), "{err:#}");

        let mut relative = term(Path::new("akesson.csv"), AkessonUpdate::Fixed, &context);
        assert!(relative.anchor(dir.path(), None).is_err());
    }

    #[test]
    fn stepping_an_unanchored_relative_profile_is_an_error() {
        // A runner that forgot to anchor would otherwise run without the field it asked for.
        let context = context();
        let every = AkessonUpdate::Every(NonZeroUsize::new(1).unwrap());
        let mut term = term(Path::new("akesson.csv"), every, &context);
        let err = term.advance(&context, 1).unwrap_err();
        assert!(err.to_string().contains("anchor"), "{err}");
    }

    #[test]
    fn a_committed_volume_change_stops_the_run_in_either_mode() {
        use crate::cell::{Shape, VolumeScalePolicy};
        use crate::context::{Context as _, WithSimulationCell};
        let dir = tempfile::tempdir().unwrap();
        let file = single_slab_profile(dir.path());
        let every = AkessonUpdate::Every(NonZeroUsize::new(1).unwrap());
        for update in [AkessonUpdate::Fixed, every] {
            let mut context = context();
            let mut term = term(&file, update, &context);
            let volume = context.cell().volume().unwrap();
            context
                .scale_volume_and_positions(volume * 1.2, VolumeScalePolicy::Isotropic)
                .unwrap();
            assert!(term.advance(&context, 1).is_err(), "{update:?}");
        }
    }

    #[test]
    fn saving_creates_the_state_directory() {
        let dir = tempfile::tempdir().unwrap();
        let state_dir = dir.path().join("not/yet/there");
        let context = context();
        let every = AkessonUpdate::Every(NonZeroUsize::new(1).unwrap());
        let mut term = term(Path::new("akesson.csv"), every, &context);
        term.anchor(&state_dir, None).unwrap();
        term.advance(&context, 1).unwrap();
        assert!(state_dir.join("akesson.csv").exists());
    }

    #[test]
    fn flush_keeps_samples_taken_since_the_last_refresh() {
        // A run ending between refreshes must not drop its tail on restart.
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("akesson.csv");
        let context = context();
        let every = AkessonUpdate::Every(NonZeroUsize::new(10).unwrap());
        let mut term = term(&file, every, &context);
        for step in 1..=13 {
            term.advance(&context, step).unwrap();
        }
        term.flush().unwrap();
        assert_eq!(load_profile(&file, &term.grid).unwrap().1, 13);
    }

    #[test]
    fn fixed_mode_reloads_the_saved_field_and_never_writes() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("akesson.csv");
        let context = context();
        let every = AkessonUpdate::Every(NonZeroUsize::new(1).unwrap());
        let mut sampled = term(&file, every, &context);
        sampled.advance(&context, 1).unwrap();
        let saved = std::fs::read_to_string(&file).unwrap();

        let mut fixed = term(&file, AkessonUpdate::Fixed, &context);
        let energy = fixed.energy(&context, &Change::Everything);
        assert!(
            (energy - sampled.energy(&context, &Change::Everything)).abs() < 1e-12 * energy.abs()
        );
        assert_eq!(fixed.advance(&context, 1).unwrap(), 0.0);
        assert_eq!(fixed.energy(&context, &Change::Everything), energy);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), saved);
        std::fs::remove_file(&file).unwrap();
        fixed.flush().unwrap();
        assert!(!file.exists(), "a fixed profile was written");
    }

    fn plain(cutoff: f64) -> CoulombScheme {
        CoulombScheme {
            kind: CoulombSchemeKind::Plain,
            cutoff,
            kappa: None,
        }
    }

    #[test]
    fn greberg_needs_cutoff_spanning_the_minimum_image_box() {
        // 50×50×100 slit: the farthest minimum-image pair is √(25² + 25² + 100²) ≈ 106.1 Å apart.
        let cell = Cell::Slit(Slit::new(50.0, 50.0, 100.0));
        assert_eq!(
            MeanFieldKernel::select(Some(&plain(107.0)), &cell).unwrap(),
            MeanFieldKernel::Greberg
        );
        assert!(MeanFieldKernel::select(Some(&plain(105.0)), &cell).is_err());
    }

    #[test]
    fn only_a_square_slit_qualifies() {
        // A cuboid is periodic in z, so its exterior charge is not one square sheet per slab.
        let scheme = plain(1e4);
        for cell in [
            Cell::Cuboid(Cuboid::new(50.0, 50.0, 100.0)),
            Cell::Cylinder(Cylinder::new(25.0, 100.0)),
            Cell::Slit(Slit::new(50.0, 60.0, 100.0)),
        ] {
            assert!(
                MeanFieldKernel::select(Some(&scheme), &cell).is_err(),
                "{cell:?}"
            );
        }
    }

    #[test]
    fn rejects_screened_or_truncated_pair_schemes() {
        // Greberg's bare-Coulomb exterior double-counts a screened pair term, and a scheme
        // that is not 1/r across the box leaves a gap no square-sheet field fills.
        let cell = Cell::Slit(Slit::new(50.0, 50.0, 100.0));
        let screened = CoulombScheme {
            kappa: Some(0.1),
            ..plain(1e4)
        };
        let fanourgakis = CoulombScheme {
            kind: CoulombSchemeKind::Other("CoulombFanourgakis"),
            ..plain(1e4)
        };
        for scheme in [screened, fanourgakis] {
            assert!(
                MeanFieldKernel::select(Some(&scheme), &cell).is_err(),
                "{scheme:?}"
            );
        }
        assert!(MeanFieldKernel::select(None, &cell).is_err());
    }
}

/// The term reached through YAML and `Hamiltonian::finalize`.
#[cfg(test)]
mod wiring_tests {
    use crate::backend::Backend;
    use crate::WithHamiltonian;

    // Neutral 1:1 salt in a 20×20×40 slit. NONBONDED and UPDATE are filled in per test.
    const SALT: &str = r#"
atoms:
  - {name: Na, mass: 1.0, charge: 1.0, sigma: 1.0}
  - {name: Cl, mass: 1.0, charge: -1.0, sigma: 1.0}
molecules:
  - {name: NA, atoms: [Na]}
  - {name: CL, atoms: [Cl]}
system:
  cell: !Slit [20.0, 20.0, 40.0]
  medium: {permittivity: !Fixed 80.0, temperature: 300.0}
  energy:
    - !Nonbonded
NONBONDED
    - !Akesson {resolution: 1.0, file: FILE, update: UPDATE}
  blocks:
    - {molecule: NA, N: 2, insert: !Manual [[0.0, 0.0, -5.0], [0.0, 0.0, 5.0]]}
    - {molecule: CL, N: 2, insert: !Manual [[0.0, 0.0, -10.0], [0.0, 0.0, 10.0]]}
propagate: {seed: !Fixed 1, criterion: Metropolis, steps: 0, collections: []}
"#;

    const FULL_COULOMB: &str = "      default: [!Coulomb {cutoff: 1000.0}]";

    fn build(nonbonded: &str, update: &str) -> anyhow::Result<Backend> {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("akesson.csv");
        let yaml = SALT
            .replace("NONBONDED", nonbonded)
            .replace("UPDATE", update)
            .replace("FILE", &format!("{:?}", file.display().to_string()));
        Backend::from_yaml_str(&yaml, None, &mut rand::thread_rng())
    }

    #[test]
    fn a_refreshing_term_makes_the_hamiltonian_adaptive() {
        let every = build(FULL_COULOMB, "!Every 10").unwrap();
        assert!(every.hamiltonian().is_adaptive());
    }

    #[test]
    fn rejects_pair_setups_the_field_cannot_complement() {
        let mixed = "      default: [!Coulomb {cutoff: 1000.0}]
      replace:
        [Na, Cl]: [!Fanourgakis {cutoff: 1000.0}]";
        let splined = "      default: [!Coulomb {cutoff: 1000.0}]
      spline: {cutoff: 20.0}";
        for (nonbonded, reason) in [(mixed, "other pairs use"), (splined, "cutoff ≥")] {
            let err = build(nonbonded, "!Every 10").unwrap_err();
            assert!(format!("{err:#}").contains(reason), "{err:#}");
        }
    }

    #[test]
    fn refresh_work_is_booked_so_the_run_conserves_energy() {
        // Every step refreshes φ, changing the energy outside any move. Unbooked, that work
        // would show up as drift; σ is nonzero from step 1, so the work is too.
        use crate::analysis::AnalysisCollection;
        use crate::montecarlo::MarkovChain;
        use crate::propagate::Propagate;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("akesson.csv");
        let yaml = SALT
            .replace("NONBONDED", FULL_COULOMB)
            .replace("UPDATE", "!Every 1")
            .replace("FILE", &format!("{:?}", file.display().to_string()))
            .replace(
                "propagate: {seed: !Fixed 1, criterion: Metropolis, steps: 0, collections: []}",
                "propagate:
  seed: !Fixed 1
  criterion: Metropolis
  steps: 20
  collections:
    - !Stochastic
      cycles: 4
      moves:
        - !TranslateMolecule {molecule: NA, dp: 3.0, weight: 1}
        - !TranslateMolecule {molecule: CL, dp: 3.0, weight: 1}",
            );
        let thermal_energy = crate::R_IN_KJ_PER_MOL * 300.0;
        let context = Backend::from_yaml_str(&yaml, None, &mut rand::thread_rng()).unwrap();
        let propagate = Propagate::from_str(&yaml, &context, thermal_energy).unwrap();
        let mut chain = MarkovChain::new(
            context,
            propagate,
            thermal_energy,
            AnalysisCollection::default(),
        )
        .unwrap();
        let initial = chain.system_energy();
        chain.run_n_steps(20).unwrap();
        let change = (chain.system_energy() - initial).abs();
        let drift = chain.energy_drift(initial);
        assert!(file.exists(), "the chain never refreshed φ");
        assert!(change > 1e-3, "the run should change the energy: {change}");
        assert!(drift < 1e-9 * change.max(1.0), "drift {drift:e}");
    }

    #[test]
    fn rejects_medium_salt_screening() {
        // Screening comes from the medium here, not the pair scheme, so the scheme alone looks bare.
        let salty = SALT.replace(
            "temperature: 300.0}",
            "temperature: 300.0, salt: [!NaCl, 0.1]}",
        );
        let yaml = salty
            .replace("NONBONDED", FULL_COULOMB)
            .replace("UPDATE", "Fixed")
            .replace("FILE", "unused.csv");
        let err = Backend::from_yaml_str(&yaml, None, &mut rand::thread_rng()).unwrap_err();
        assert!(format!("{err:#}").contains("screened"), "{err:#}");
    }
}

/// Samplers that assume a fixed energy landscape must refuse a refreshing field.
#[cfg(all(test, feature = "cli"))]
mod sampler_guard_tests {
    /// A driver fixture made adaptive: one charged particle in a slit with plain Coulomb.
    fn adaptive_input(fixture: &str, dir: &std::path::Path) -> std::path::PathBuf {
        let csv = dir.join("akesson.csv");
        let yaml = std::fs::read_to_string(fixture)
            .unwrap()
            .replace("{name: A, mass: 1.0}", "{name: A, mass: 1.0, charge: 1.0}")
            .replace("!Cuboid [40.0, 40.0, 40.0]", "!Slit [40.0, 40.0, 40.0]")
            .replace(
                "  energy:\n",
                &format!(
                    "  energy:\n    - !Nonbonded {{default: [!Coulomb {{cutoff: 1000.0}}]}}\n    \
                     - !Akesson {{resolution: 1.0, file: {:?}, update: !Every 10}}\n",
                    csv.display().to_string()
                ),
            );
        let input = dir.join("input.yaml");
        std::fs::write(&input, yaml).unwrap();
        input
    }

    fn assert_refused(result: anyhow::Result<impl Sized>, sampler: &str) {
        let Err(err) = result else {
            panic!("{sampler} accepted a refreshing Åkesson field");
        };
        assert!(format!("{err:#}").contains("Fixed"), "{err:#}");
    }

    #[test]
    fn wang_landau_refuses_a_refreshing_field() {
        let dir = tempfile::tempdir().unwrap();
        let input = adaptive_input("tests/files/wang_landau_1d/input.yaml", dir.path());
        assert_refused(
            crate::wang_landau::WangLandau::prepare(&input),
            "Wang–Landau",
        );
    }

    #[test]
    fn umbrella_refuses_a_refreshing_field() {
        let dir = tempfile::tempdir().unwrap();
        let input = adaptive_input("tests/files/umbrella_1d/input.yaml", dir.path());
        assert_refused(crate::umbrella::Umbrella::prepare(&input, None), "umbrella");
    }
}

/// The artefact the term removes: with the wall charge inside the minimum-image box, its
/// exterior part is missing, so the wall is under-screened from afar and counterions float
/// into the bulk. Far from the wall a screened electrolyte is neutral.
#[cfg(test)]
mod screening_tests {
    use crate::analysis::AnalysisCollection;
    use crate::backend::Backend;
    use crate::group::GroupCollection;
    use crate::montecarlo::MarkovChain;
    use crate::propagate::Propagate;
    use crate::ObserveContext;

    /// Charge excess (N₊ − N₋)/(N₊ + N₋) of the mobile ions in the upper half of the slit.
    fn bulk_charge_excess(akesson: bool, steps: usize, seed: u64, dir: &std::path::Path) -> f64 {
        let (side, height, salt) = (40.0, 120.0, 40);
        let wall_z = -0.5 * height + 0.5;
        let beads: Vec<String> = (0..5)
            .flat_map(|ix| (0..4).map(move |iy| (ix, iy)))
            .map(|(ix, iy)| {
                format!(
                    "[{}, {}, {wall_z}]",
                    -16.0 + 8.0 * ix as f64,
                    -15.0 + 10.0 * iy as f64
                )
            })
            .collect();
        let akesson = if akesson {
            format!(
                "    - !Akesson {{resolution: 0.5, file: {:?}, update: !Every 20}}\n",
                dir.join("akesson.csv").display().to_string()
            )
        } else {
            String::new()
        };
        let yaml = format!(
            r#"
atoms:
  - {{name: W, charge: -1.0, sigma: 4.0, epsilon: 0.5, mass: 1.0}}
  - {{name: Na, charge: 1.0, sigma: 4.0, epsilon: 0.5, mass: 1.0}}
  - {{name: Cl, charge: -1.0, sigma: 4.0, epsilon: 0.5, mass: 1.0}}
molecules:
  - {{name: wall, atoms: [W]}}
  - {{name: na, atoms: [Na]}}
  - {{name: cl, atoms: [Cl]}}
system:
  cell: !Slit [{side}, {side}, {height}]
  medium: {{permittivity: !Fixed 78.4, temperature: 298.15}}
  energy:
    - !Nonbonded {{default: [!Coulomb {{cutoff: 1.0e5}}, !WCA {{mixing: LB}}]}}
{akesson}  blocks:
    - {{molecule: wall, N: 20, insert: !Manual [{beads}]}}
    - {{molecule: na, N: {na}, insert: !RandomAtomPos {{directions: xyz}}}}
    - {{molecule: cl, N: {salt}, insert: !RandomAtomPos {{directions: xyz}}}}
analysis:
propagate:
  seed: !Fixed {seed}
  criterion: Metropolis
  steps: {steps}
  collections:
    - !Stochastic
      cycles: 100
      moves:
        - !TranslateMolecule {{molecule: na, dp: 5.0, weight: 3}}
        - !TranslateMolecule {{molecule: cl, dp: 5.0, weight: 2}}
"#,
            beads = beads.join(", "),
            na = 20 + salt,
        );
        let kt = crate::R_IN_KJ_PER_MOL * 298.15;
        let context = Backend::from_yaml_str(&yaml, None, &mut rand::thread_rng()).unwrap();
        let propagate = Propagate::from_str(&yaml, &context, kt).unwrap();
        let mut chain =
            MarkovChain::new(context, propagate, kt, AnalysisCollection::default()).unwrap();
        let (mut excess, mut total) = (0.0, 0.0);
        for step in 0..steps {
            chain.run_n_steps(1).unwrap();
            if step < steps / 5 {
                continue;
            }
            let context = chain.context();
            for i in 0..context.num_particles() {
                if context.position(i).z > 0.0 {
                    let q = context.atom_charge(i);
                    excess += q;
                    total += q.abs();
                }
            }
        }
        excess / total
    }

    /// Release mode, ~15 s. Fixed-seed runs scatter by ±0.004 with the field and land at
    /// +0.03…+0.04 without it.
    #[test]
    #[ignore]
    fn counterions_float_into_the_bulk_only_without_the_field() {
        let dir = tempfile::tempdir().unwrap();
        let with = bulk_charge_excess(true, 20_000, 1, dir.path());
        let without = bulk_charge_excess(false, 20_000, 1, dir.path());
        assert!(
            with.abs() < 0.01,
            "with the field the bulk should be neutral: {with:.4}"
        );
        assert!(
            without > 0.02,
            "without it counterions should float: {without:.4}"
        );
    }
}

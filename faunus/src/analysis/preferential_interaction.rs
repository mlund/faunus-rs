// Copyright 2026 Mikael Lund
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

//! Preferential interaction coefficient Γ of a ligand with a molecular substrate.
//!
//! Γ is the thermodynamic derivative behind preferential binding and exclusion: it fixes how the
//! chemical potential of the substrate responds to the ligand concentration, and hence — through
//! the Wyman linkage — the ligand dependence of any equilibrium the substrate takes part in.
//!
//! Let `D(δ)` be the region within `r_lig + δ` of the substrate surface. In implicit solvent,
//!
//! ```text
//! Γ(δ) = ⟨N(δ)⟩ − c·Vol(D(δ))
//! ```
//!
//! is the cumulative ligand excess. In explicit solvent, the counted-solvent estimator instead is
//!
//! ```text
//! Γ(δ) = ⟨N₃(δ) − [N₃ᵇ/N₁ᵇ] N₁(δ)⟩,
//! ```
//!
//! where components 1 and 3 are solvent and ligand, and superscript `b` denotes the region outside
//! the widest domain. The ligand and solvent are partitioned through the identical `D(δ)` in each
//! frame. Scanning δ removes the need to *choose* a domain: Γ(δ) approaches a plateau once δ
//! outruns the range over which the solvent composition is perturbed, and that plateau is Γ. The
//! approach need not be monotonic, so a profile still varying at the widest δ is unconverged, not
//! a bound.
//!
//! # Substrate geometry, one copy
//!
//! The reference geometry — the per-atom cell volumes `vᵢ(p)`, the surface areas, the engulfed
//! flags — is cached for a rigid substrate because translation and rotation preserve its internal
//! geometry. For a flexible substrate it is rebuilt from the live coordinates at every sampled
//! frame. Per-sample counting always reads live positions through the minimum image, so either
//! substrate may translate and wrap across the periodic boundary.
//!
//! Only one substrate copy is allowed (`validate_substrate`). Several identical rigid copies
//! would share the same body-frame reference and could multiply the ligand statistics per frame,
//! but only while their domains stay disjoint: once two `D(δ)` overlap, a ligand between them
//! falls in both (breaking the tile-the-space partition behind Σγᵢ = Γ), and each copy's `vᵢ`
//! reaches into space the other copy occupies (biasing the volume). At that separation Γ also
//! stops being the infinite-dilution coefficient and picks up a protein–protein crowding term.
//! Supporting N > 1 therefore needs a nearest-copy tie-break plus an inter-copy overlap guard, or
//! a joint tessellation of all copies that forfeits the compute-once optimization — a deliberate
//! follow-up, not a free win.

use super::{Analyze, Frequency, Sampling};
use crate::auxiliary::{BlockSummary, ColumnWriter, MappingExt, WeightedBlockAverage};
use crate::cell::{BoundaryConditions, Shape};
use crate::selection::{ComSelection, Selection};
use crate::{ObserveContext, Point};
use anyhow::Result;
use derive_more::Debug;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Ladder of domain thicknesses δ.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shell {
    /// Largest δ (Å). Beyond this a ligand counts as bulk; must be a multiple of `resolution`.
    max: f64,
    /// Spacing of the ladder (Å).
    resolution: f64,
}

/// A counted solvent species used as the finite-box reference for Γ.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExplicitSolvent {
    /// Solvent molecules or atoms whose local and bulk populations are compared with the ligand.
    selection: Selection,
    /// Count one mass centre per selected molecule instead of every selected atom.
    use_com: bool,
}

/// Hidden state for the choice of thermodynamic reference medium.
#[derive(Debug)]
enum SolventReference {
    /// McMillan–Mayer estimator, with the solvent integrated out.
    Implicit,
    /// Finite-box estimator using a counted solvent species in the same spatial partition as the
    /// ligand.
    Explicit(Box<ExplicitSolventReference>),
}

#[derive(Debug)]
struct ExplicitSolventReference {
    selection: ComSelection,
    /// Per-frame solvent tally `[residue * n_shells + shell]`.
    counts: Vec<f64>,
    shell_totals: Vec<f64>,
    positions: Vec<Point>,
    /// Mean solvent population owned by each residue and shell.
    residue_counts: Vec<Vec<WeightedBlockAverage>>,
    concentration: WeightedBlockAverage,
    bulk_ligand_to_solvent_ratio: WeightedBlockAverage,
}

impl SolventReference {
    fn is_explicit(&self) -> bool {
        matches!(self, Self::Explicit(_))
    }
}

impl Shell {
    /// Unknown user input is an error, so a ladder that cannot be walked is refused here rather
    /// than overflowing `len` or silently collapsing to a single rung.
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.resolution.is_finite() && self.resolution > 0.0,
            "PreferentialInteraction: shell.resolution must be positive, got {}",
            self.resolution
        );
        anyhow::ensure!(
            self.max.is_finite() && self.max >= self.resolution,
            "PreferentialInteraction: shell.max must be at least shell.resolution, got {} and {}",
            self.max,
            self.resolution
        );
        // Each rung is a tessellation and an allocation, so an enormous max/resolution ratio would
        // hang the build before any later guard runs. A few thousand rungs is already far past any
        // sensible ladder.
        const MAX_RUNGS: f64 = 10_000.0;
        anyhow::ensure!(
            self.max / self.resolution <= MAX_RUNGS,
            "PreferentialInteraction: shell.max / shell.resolution = {:.0} exceeds {MAX_RUNGS:.0}; \
             coarsen the ladder",
            self.max / self.resolution
        );
        // shell.max is the bulk boundary, so the ladder must land on it exactly. If it were not a
        // multiple of the resolution, the outermost rung δ = round(max/resolution)·resolution would
        // sit up to half a step from shell.max, quietly shifting where a ligand stops counting.
        let rungs = self.max / self.resolution;
        anyhow::ensure!(
            (rungs - rungs.round()).abs() <= 1e-6 * rungs.max(1.0),
            "PreferentialInteraction: shell.max ({}) must be an integer multiple of \
             shell.resolution ({})",
            self.max,
            self.resolution
        );
        Ok(())
    }

    /// Number of rungs, including δ = 0.
    fn len(&self) -> usize {
        (self.max / self.resolution).round() as usize + 1
    }

    /// δ of rung `k`.
    fn delta(&self, k: usize) -> f64 {
        k as f64 * self.resolution
    }

    /// Rung a ligand sitting `gap` beyond the exclusion boundary belongs to, or `None` for bulk.
    ///
    /// A ligand that has penetrated the boundary (`gap < 0`) lands in rung zero, so the count and
    /// the volume keep referring to the same region even when the ligand is soft.
    fn index(&self, gap: f64) -> Option<usize> {
        let k = if gap <= 0.0 {
            0
        } else {
            (gap / self.resolution).ceil() as usize
        };
        (k < self.len()).then_some(k)
    }
}

/// YAML builder for [`PreferentialInteraction`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreferentialInteractionBuilder {
    /// The molecular substrate the ligand is counted around.
    substrate: Selection,
    /// The species whose excess is measured. One per analysis; declare the analysis again for more.
    ligand: Selection,
    /// Count the mass centre of each selected molecule instead of each selected atom.
    #[serde(default)]
    use_com: bool,
    /// Counted solvent for the finite-box explicit-solvent estimator. If absent, use the implicit
    /// solvent estimator.
    #[serde(default)]
    solvent: Option<ExplicitSolvent>,
    /// Ligand radius (Å). Defaults to σ/2 for atoms and to zero for mass centres.
    #[serde(default)]
    radius: Option<f64>,
    /// The δ ladder.
    shell: Shell,
    /// Water radius (Å), setting the surface area against which hydration is reported.
    /// Spelled `probe_radius` to match the SASA/tessellation energy terms.
    #[serde(rename = "probe_radius", default = "default_solvent_probe")]
    solvent_probe: f64,
    /// Γ(δ), the convergence profile. Read this before trusting the reported Γ.
    #[serde(default)]
    profile: Option<PathBuf>,
    /// One row per residue, at the widest δ.
    #[serde(default)]
    file: Option<PathBuf>,
    /// Sampling frequency.
    frequency: Frequency,
}

const fn default_solvent_probe() -> f64 {
    1.4
}

/// Volume of one water molecule (Å³): 18.015 g/mol at 1 g/mL.
const WATER_VOLUME: f64 = 18.015e24 / physical_constants::AVOGADRO_CONSTANT;

impl PreferentialInteractionBuilder {
    pub fn apply_output_dir(&mut self, dir: &Path) -> Result<()> {
        crate::analysis::prefix_opt(&mut self.profile, dir)?;
        crate::analysis::prefix_opt(&mut self.file, dir)
    }

    pub fn build(&self, context: &impl ObserveContext) -> Result<PreferentialInteraction> {
        self.shell.validate()?;
        anyhow::ensure!(
            self.solvent_probe.is_finite() && self.solvent_probe >= 0.0,
            "PreferentialInteraction: probe_radius must be finite and non-negative, got {}",
            self.solvent_probe
        );
        if let Some(radius) = self.radius {
            anyhow::ensure!(
                radius.is_finite() && radius >= 0.0,
                "PreferentialInteraction: radius must be finite and non-negative, got {radius}"
            );
        }
        // The σ-derived radius path already rejects an empty ligand selection, but the explicit
        // `radius` and `use_com` paths skip it, so check here for every path.
        let matched = if self.use_com {
            !context.resolve_groups(&self.ligand).is_empty()
        } else {
            !context.resolve_atoms(&self.ligand).is_empty()
        };
        anyhow::ensure!(
            matched,
            "PreferentialInteraction: ligand selection '{}' matched nothing",
            self.ligand.source()
        );
        if let Some(solvent) = &self.solvent {
            let matched = if solvent.use_com {
                !context.resolve_groups(&solvent.selection).is_empty()
            } else {
                !context.resolve_atoms(&solvent.selection).is_empty()
            };
            anyhow::ensure!(
                matched,
                "PreferentialInteraction: solvent selection '{}' matched nothing",
                solvent.selection.source()
            );
            let ligand_atoms: std::collections::HashSet<_> =
                context.resolve_atoms(&self.ligand).into_iter().collect();
            let solvent_atoms = context.resolve_atoms(&solvent.selection);
            anyhow::ensure!(
                solvent_atoms
                    .iter()
                    .all(|atom| !ligand_atoms.contains(atom)),
                "PreferentialInteraction: ligand selection '{}' and solvent selection '{}' \
                 overlap; they must count distinct species",
                self.ligand.source(),
                solvent.selection.source()
            );
            let substrate_atoms: std::collections::HashSet<_> =
                context.resolve_atoms(&self.substrate).into_iter().collect();
            anyhow::ensure!(
                solvent_atoms
                    .iter()
                    .all(|atom| !substrate_atoms.contains(atom)),
                "PreferentialInteraction: substrate selection '{}' and solvent selection '{}' \
                 overlap; the substrate cannot be its own solvent",
                self.substrate.source(),
                solvent.selection.source()
            );
        }
        let ligand_radius = self.ligand_radius(context)?;
        let reference = SurfaceReference::new(
            context,
            &self.substrate,
            ligand_radius,
            self.shell,
            self.solvent_probe,
        )?;
        let n_shells = self.shell.len();
        let accumulators = || (0..n_shells).map(|_| WeightedBlockAverage::new()).collect();
        let n_residues = reference.residues.len();
        let solvent_reference =
            self.solvent
                .as_ref()
                .map_or(SolventReference::Implicit, |solvent| {
                    SolventReference::Explicit(Box::new(ExplicitSolventReference {
                        selection: ComSelection::new(solvent.selection.clone(), solvent.use_com),
                        counts: vec![0.0; n_residues * n_shells],
                        shell_totals: vec![0.0; n_shells],
                        positions: Vec::new(),
                        residue_counts: (0..n_residues).map(|_| accumulators()).collect(),
                        concentration: WeightedBlockAverage::new(),
                        bulk_ligand_to_solvent_ratio: WeightedBlockAverage::new(),
                    }))
                });
        Ok(PreferentialInteraction {
            ligand: ComSelection::new(self.ligand.clone(), self.use_com),
            solvent_reference,
            residue_gamma: (0..n_residues).map(|_| accumulators()).collect(),
            residue_counts: (0..n_residues).map(|_| accumulators()).collect(),
            reference_counts: (0..n_residues).map(|_| accumulators()).collect(),
            residue_volumes: (0..n_residues).map(|_| accumulators()).collect(),
            residue_water_volumes: (0..n_residues).map(|_| accumulators()).collect(),
            residue_asa: (0..n_residues)
                .map(|_| WeightedBlockAverage::new())
                .collect(),
            domain_volume: accumulators(),
            counts: vec![0.0; n_residues * n_shells],
            shell_totals: vec![0.0; n_shells],
            positions: Vec::new(),
            reference,
            gamma: accumulators(),
            concentration: WeightedBlockAverage::new(),
            stopped: false,
            profile_file: self.profile.clone(),
            residue_file: self.file.clone(),
            sampling: Sampling::new(self.frequency),
        })
    }

    /// The single radius that sets the whole ladder.
    ///
    /// An averaged radius would corrupt every reference volume, so a selection spanning atom kinds
    /// of different size is refused rather than reconciled.
    fn ligand_radius(&self, context: &impl ObserveContext) -> Result<f64> {
        if let Some(radius) = self.radius {
            return Ok(radius);
        }
        if self.use_com {
            return Ok(0.0);
        }
        let topology = context.topology_ref();
        let kinds = topology.atomkinds();
        let atoms = context.resolve_atoms(&self.ligand);
        anyhow::ensure!(
            !atoms.is_empty(),
            "PreferentialInteraction: ligand selection '{}' matched no atoms",
            self.ligand.source()
        );
        let mut radii = atoms.into_iter().map(|i| {
            let kind = &kinds[context.atom_kind(i).get()];
            kind.sigma().map(|sigma| sigma / 2.0).ok_or_else(|| {
                anyhow::anyhow!(
                    "PreferentialInteraction: ligand atom kind '{}' has no σ; set `radius` \
                     explicitly",
                    kind.name()
                )
            })
        });
        let first = radii.next().unwrap()?;
        for radius in radii {
            anyhow::ensure!(
                (radius? - first).abs() < 1e-9,
                "PreferentialInteraction: ligand selection '{}' spans atom kinds of different σ; \
                 set `radius` explicitly",
                self.ligand.source()
            );
        }
        Ok(first)
    }
}

/// A residue of the substrate, and the geometry it offers a ligand.
///
/// Residues partition the substrate's atoms, so summing a per-atom excess over them loses nothing:
/// Σ γ stays exactly Γ. A substrate without residue records — a coarse-grained bead model — gets
/// one residue per atom, which is the same statement with a finer grain.
#[derive(Debug)]
pub(crate) struct Residue {
    pub name: String,
    pub number: usize,
    /// Water-accessible surface area (Å²).
    pub asa: f64,
}

/// Refuse a cell whose tessellation and minimum image would disagree.
///
/// The domain volumes come from a tessellation of the periodic box, while the counting uses the
/// cell's own minimum image. A partially periodic cell — a slit, a cylinder — would wrap the
/// tessellation through its hard walls while the counting does not, so volume and count would
/// refer to different regions and Γ would be biased with nothing to show for it.
fn reject_unsupported_cell(context: &impl ObserveContext) -> Result<()> {
    use crate::cell::{Cell, PeriodicDirections};
    let cell = context.cell();
    anyhow::ensure!(
        matches!(cell, Cell::Cuboid(_) | Cell::Sphere(_)),
        "PreferentialInteraction: needs a cuboid or spherical cell; a partially periodic cell \
         would tessellate through its own walls"
    );
    anyhow::ensure!(
        matches!(
            cell.pbc(),
            PeriodicDirections::PeriodicXYZ | PeriodicDirections::None
        ),
        "PreferentialInteraction: needs a fully periodic or fully aperiodic cell"
    );
    Ok(())
}

/// Radius of a sphere about the substrate centroid enclosing every atom, including its own radius.
///
/// The molecule is unwrapped by growing a spanning tree: each atom is placed on the minimum image
/// of the *nearest already-placed* atom, so a rigid body straddling a periodic boundary is
/// measured at its true extent. Unwrapping against a single anchor instead would fold any atom
/// farther than half the box from that anchor onto the wrong image, underestimating the radius —
/// and defeating the self-overlap guard for exactly the oversized substrate it must reject, since
/// a molecule wider than the box then reports a true, large radius here.
fn bounding_radius(
    cell: &crate::cell::Cell,
    atoms: &[usize],
    radii: &[f64],
    position: impl Fn(usize) -> Point,
) -> f64 {
    let raw: Vec<Point> = atoms.iter().map(|&i| position(i)).collect();
    let n = raw.len();
    let mut unwrapped = vec![Point::zeros(); n];
    let mut in_tree = vec![false; n];
    // Prim's algorithm: `best[i]` is the unwrapped position atom i would take against its nearest
    // in-tree atom, and its minimum-image distance to that atom.
    let mut best = vec![(Point::zeros(), f64::INFINITY); n];
    unwrapped[0] = raw[0];
    in_tree[0] = true;
    let mut last = 0;
    for _ in 1..n {
        for i in 0..n {
            if in_tree[i] {
                continue;
            }
            let disp = cell.distance(&unwrapped[last], &raw[i]);
            let d = disp.norm();
            if d < best[i].1 {
                best[i] = (unwrapped[last] - disp, d);
            }
        }
        last = (0..n)
            .filter(|&i| !in_tree[i])
            .min_by(|&a, &b| best[a].1.total_cmp(&best[b].1))
            .expect("substrate is non-empty");
        unwrapped[last] = best[last].0;
        in_tree[last] = true;
    }
    let centroid = unwrapped.iter().sum::<Point>() / n as f64;
    unwrapped
        .iter()
        .zip(radii)
        .map(|(p, &r)| (p - centroid).norm() + r)
        .fold(0.0, f64::max)
}

/// How the substrate geometry must be maintained while sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeometryMode {
    /// Internal geometry is invariant, so the initial tessellation remains valid.
    Static,
    /// Internal geometry may change, so every sampled frame needs a fresh tessellation.
    Dynamic,
}

/// Validate the substrate and decide how its geometry must be maintained.
///
/// A single complete molecular group is required so its changing surface remains one connected
/// thermodynamic reference. Rigid geometry can be cached; every other degree-of-freedom model is
/// conservatively refreshed at each sample.
fn validate_substrate(
    context: &impl ObserveContext,
    substrate: &Selection,
    atoms: &[usize],
) -> Result<GeometryMode> {
    let groups = context.resolve_groups(substrate);
    anyhow::ensure!(
        groups.len() == 1,
        "PreferentialInteraction: substrate selection '{}' matched {} molecules; it must name \
         exactly one",
        substrate.source(),
        groups.len()
    );
    let group = context.group(crate::group::GroupIndex::new(groups[0]));
    let molecule = context.topology_ref().moleculekind(group.molecule());
    anyhow::ensure!(
        !molecule.atomic(),
        "PreferentialInteraction: substrate '{}' is an atomic group, which has no fixed shape",
        molecule.name()
    );
    // A selection reaching only part of the molecule would leave the rest of the body out of the
    // tessellation, and the domain would close over surface that is really buried.
    anyhow::ensure!(
        atoms.len() == group.len(),
        "PreferentialInteraction: substrate selection '{}' matched {} of the {} atoms of '{}'; \
         it must cover the whole molecule",
        substrate.source(),
        atoms.len(),
        group.len(),
        molecule.name()
    );
    Ok(if molecule.degrees_of_freedom().is_rigid() {
        GeometryMode::Static
    } else {
        GeometryMode::Dynamic
    })
}

/// Residue of each substrate atom, and the residues themselves.
///
/// Faunus keeps residues as metadata on the molecule *template*, so an absolute particle index has
/// to be walked back through the group to reach them.
fn residues_of(
    context: &impl ObserveContext,
    atoms: &[usize],
) -> Result<(Vec<Residue>, Vec<usize>)> {
    use crate::topology::IndexRange as _;

    /// What an atom belongs to. A residue *position* and an atom's *template index* are different
    /// index spaces, so they must not share a key: an atom outside every residue range would
    /// otherwise be folded into whichever residue happens to sit at that position.
    #[derive(PartialEq, Eq, Hash)]
    enum Key {
        Residue(usize, usize),
        LoneAtom(usize, usize),
    }

    let topology = context.topology_ref();
    let mut residues: Vec<Residue> = Vec::new();
    let mut owner = Vec::with_capacity(atoms.len());
    let mut seen: std::collections::HashMap<Key, usize> = std::collections::HashMap::new();

    for &atom in atoms {
        let group_index = context.group_of_particle(atom).ok_or_else(|| {
            anyhow::anyhow!("PreferentialInteraction: atom {atom} belongs to no group")
        })?;
        let group = context.group(crate::group::GroupIndex::new(group_index));
        let kind = &topology.moleculekinds()[group.molecule().get()];
        let template = kind.topology_index(atom - group.start());

        let found = kind
            .residues()
            .iter()
            .position(|residue| residue.range().contains(&template));

        let key = match found {
            Some(position) => Key::Residue(group_index, position),
            None => Key::LoneAtom(group_index, template),
        };
        let index = *seen.entry(key).or_insert_with(|| {
            let (name, number) = match found.map(|i| &kind.residues()[i]) {
                Some(residue) => (
                    residue.name().to_owned(),
                    residue.number().unwrap_or(residues.len() + 1),
                ),
                // Outside every residue record — a cap, an ion, a coarse-grained bead. Fall back to
                // the atom kind, as the selection language does.
                None => (
                    topology.atomkinds()[context.atom_kind(atom).get()]
                        .name()
                        .to_owned(),
                    residues.len() + 1,
                ),
            };
            residues.push(Residue {
                name,
                number,
                asa: 0.0,
            });
            residues.len() - 1
        });
        owner.push(index);
    }
    Ok((residues, owner))
}

/// Accumulate per-atom quantities onto their owning residues without reallocating.
fn accumulate_by_owner(target: &mut [f64], per_atom: &[f64], owner: &[usize]) {
    target.fill(0.0);
    for (&value, &residue) in per_atom.iter().zip(owner) {
        target[residue] += value;
    }
}

/// Geometry of the substrate: the domain ladder D(δ) and its partition among substrate atoms.
///
/// Rigid-body invariants are built once. Flexible geometry is replaced from the live coordinates
/// before every sample. Computing a cell's volume needs the diagram, but testing whether a point
/// lies in that cell is an argmin.
#[derive(Debug)]
struct SurfaceReference {
    mode: GeometryMode,
    /// Absolute indices of the substrate atoms.
    atoms: Vec<usize>,
    /// Radius of each substrate atom (Å).
    radii: Vec<f64>,
    /// Residue owning each substrate atom.
    owner: Vec<usize>,
    /// The residues, carrying their water-accessible surface area.
    residues: Vec<Residue>,
    /// `v(r_lig + δ_k)`, the domain volume owned by each residue, indexed `[shell][residue]`.
    ///
    /// This is voronota's own partition, which is why the ownership argmin in `locate` must use
    /// the power distance: numerator and denominator have to name the same region.
    volumes: Vec<Vec<f64>>,
    /// `v(p_w + δ_k)`, the water-accessible domain volume owned by each residue at the water probe
    /// `p_w = solvent_probe`, indexed `[shell][residue]`. Backs the hydration density b₁, which is
    /// a substrate + water property independent of the ligand.
    water_volumes: Vec<Vec<f64>>,
    /// `Vol(D(δ_k)) = Σ v(r_lig + δ_k)`, the domain volume at each rung.
    domain_volume: Vec<f64>,
    /// Box edge lengths the reference was built against (Å), for the constant-cell check — an
    /// anisotropic move can reshape the box at constant volume.
    box_dimensions: Point,
    /// How far any cached tessellation reaches beyond an atom surface,
    /// `max(r_lig, solvent_probe) + shell.max` (Å). Used by the spherical-cell wall check.
    domain_reach: f64,
    /// Box volume outside the widest domain (Å³).
    bulk_volume: f64,
    ligand_radius: f64,
    solvent_probe: f64,
    shell: Shell,
    /// Per-atom `dᵢ² − Rᵢ²` for the ligand being placed, the probe-free part of the power distance;
    /// reused so `credit` allocates nothing.
    scratch: Vec<f64>,
}

impl SurfaceReference {
    fn new(
        context: &impl ObserveContext,
        substrate: &Selection,
        ligand_radius: f64,
        shell: Shell,
        solvent_probe: f64,
    ) -> Result<Self> {
        reject_unsupported_cell(context)?;
        let atoms = context.resolve_atoms(substrate);
        anyhow::ensure!(
            !atoms.is_empty(),
            "PreferentialInteraction: substrate selection '{}' matched no atoms",
            substrate.source()
        );
        let mode = validate_substrate(context, substrate, &atoms)?;

        let topology = context.topology_ref();
        let kinds = topology.atomkinds();
        // A missing σ would collapse the ball to a point and shift the whole domain ladder inwards.
        // The mixed-σ ligand is refused for the same reason; an absent σ is the same corruption.
        let radii: Vec<f64> = atoms
            .iter()
            .map(|&i| {
                let kind = &kinds[context.atom_kind(i).get()];
                kind.sigma().map(|sigma| sigma / 2.0).ok_or_else(|| {
                    anyhow::anyhow!(
                        "PreferentialInteraction: substrate atom kind '{}' has no σ, so its size \
                         is undefined",
                        kind.name()
                    )
                })
            })
            .collect::<Result<_>>()?;

        let periodic_box = crate::energy::make_periodic_box(context.cell());
        let balls: Vec<voronota_ltr::Ball> = atoms
            .iter()
            .zip(&radii)
            .map(|(&i, &r)| {
                let p = context.position(i);
                voronota_ltr::Ball::new(p.x, p.y, p.z, r)
            })
            .collect();
        let (mut residues, owner) = residues_of(context, &atoms)?;

        // Roll a per-atom quantity up onto the residues that partition the atoms.
        let per_residue = |per_atom: &[f64]| {
            let mut sums = vec![0.0; residues.len()];
            for (&value, &residue) in per_atom.iter().zip(&owner) {
                sums[residue] += value;
            }
            sums
        };

        // One tessellation per rung. For a rigid substrate this happens only at construction; a
        // flexible substrate takes the same path before every sampled frame.
        let ladder = |probe_base: f64| -> Result<Vec<Vec<f64>>> {
            (0..shell.len())
                .map(|k| {
                    let volumes =
                        cell_volumes(&balls, probe_base + shell.delta(k), periodic_box.as_ref())?;
                    Ok(per_residue(&volumes))
                })
                .collect()
        };

        let volumes = ladder(ligand_radius)?;
        let domain_volume: Vec<f64> = volumes.iter().map(|v| v.iter().sum()).collect();

        // b₁ is Record's hydration density — a property of the substrate surface and water alone,
        // with no ligand in it, which is what lets him tabulate it once and transfer it between
        // solutes. It therefore uses the water probe for *both* the shell volume and the area; a
        // ligand-probe volume over a water-probe area would inject a spurious curvature factor
        // (a + r_lig)² / (a + p_w)² on top of the real shell curvature. This second ladder is at
        // `solvent_probe + δ`, in parallel with the ligand ladder above.
        let water_volumes = ladder(solvent_probe)?;

        let asa = per_residue(&surface_areas(
            &balls,
            solvent_probe,
            periodic_box.as_ref(),
        )?);
        for (residue, area) in residues.iter_mut().zip(asa) {
            residue.asa = area;
        }

        let box_volume = context.cell().volume().ok_or_else(|| {
            anyhow::anyhow!("PreferentialInteraction: the cell has no volume, so there is no bulk")
        })?;
        // Under periodic boundaries the domain saturates at the cell volume, so a ladder wider than
        // the box does not overflow — it quietly consumes the bulk, leaving Γ referenced against
        // nothing. A positive remainder is not enough to catch that; it has to be a usable one.
        const MIN_BULK_FRACTION: f64 = 0.01;
        let bulk_volume = box_volume - domain_volume[shell.len() - 1];
        anyhow::ensure!(
            bulk_volume > MIN_BULK_FRACTION * box_volume,
            "PreferentialInteraction: the domain leaves only {:.1}% of the cell as bulk, which is \
             too little to reference Γ against. Reduce `shell.max` or enlarge the cell",
            100.0 * bulk_volume / box_volume
        );
        if bulk_volume < 0.1 * box_volume {
            log::warn!(
                "preferential interaction: only {:.0}% of the cell is bulk; Γ is referenced \
                 against a thin shell and its error bar will be optimistic",
                100.0 * bulk_volume / box_volume
            );
        }

        // Widest of the two ladders (the water ladder wins when `use_com` makes r_lig = 0). Under
        // PBC the domain must fit within half the shortest edge or it wraps onto its own image —
        // the bulk-fraction check does not catch this (a thin box can stay 99 % bulk).
        let domain_reach = ligand_radius.max(solvent_probe) + shell.delta(shell.len() - 1);
        let bound = bounding_radius(context.cell(), &atoms, &radii, |i| context.position(i));
        if context.cell().pbc().is_some() {
            let min_edge = context
                .cell()
                .bounding_box()
                .map(|b| b.x.min(b.y).min(b.z))
                .unwrap_or(f64::INFINITY);
            anyhow::ensure!(
                bound + domain_reach < 0.5 * min_edge,
                "PreferentialInteraction: the substrate plus its domain (radius {:.1} Å) exceeds \
                 half the shortest box edge ({:.1} Å); the domain would wrap onto its periodic \
                 image. Enlarge the cell or reduce `shell.max`",
                bound + domain_reach,
                0.5 * min_edge
            );
        }

        let n = atoms.len();
        Ok(Self {
            mode,
            atoms,
            radii,
            domain_reach,
            owner,
            residues,
            volumes,
            water_volumes,
            domain_volume,
            box_dimensions: context.cell().bounding_box().unwrap_or_else(Point::zeros),
            bulk_volume,
            ligand_radius,
            solvent_probe,
            shell,
            scratch: vec![0.0; n],
        })
    }

    /// Refresh all geometry derived from coordinates when the substrate can change shape.
    fn refresh(&mut self, context: &impl ObserveContext) -> Result<()> {
        if self.mode == GeometryMode::Dynamic {
            reject_unsupported_cell(context)?;

            let shell = self.shell;
            let owner = &self.owner;
            let residues_len = self.residues.len();
            let periodic_box = crate::energy::make_periodic_box(context.cell());
            let balls: Vec<voronota_ltr::Ball> = self
                .atoms
                .iter()
                .zip(&self.radii)
                .map(|(&i, &r)| {
                    let p = context.position(i);
                    voronota_ltr::Ball::new(p.x, p.y, p.z, r)
                })
                .collect();

            let refill_ladder = |target: &mut [Vec<f64>],
                                 mut domain_volume: Option<&mut [f64]>,
                                 probe_base: f64|
             -> Result<()> {
                for k in 0..shell.len() {
                    let values =
                        cell_volumes(&balls, probe_base + shell.delta(k), periodic_box.as_ref())?;
                    let rung = &mut target[k];
                    if rung.len() != residues_len {
                        rung.resize(residues_len, 0.0);
                    }
                    accumulate_by_owner(rung, &values, owner);
                    if let Some(domain_volume) = domain_volume.as_deref_mut() {
                        domain_volume[k] = rung.iter().sum();
                    }
                }
                Ok(())
            };

            refill_ladder(
                &mut self.volumes,
                Some(&mut self.domain_volume),
                self.ligand_radius,
            )?;
            refill_ladder(&mut self.water_volumes, None, self.solvent_probe)?;

            let areas = surface_areas(&balls, self.solvent_probe, periodic_box.as_ref())?;
            let mut asa = vec![0.0; self.residues.len()];
            accumulate_by_owner(&mut asa, &areas, &self.owner);
            for (residue, area) in self.residues.iter_mut().zip(asa) {
                residue.asa = area;
            }

            let box_volume = context.cell().volume().ok_or_else(|| {
                anyhow::anyhow!(
                    "PreferentialInteraction: the cell has no volume, so there is no bulk"
                )
            })?;
            self.box_dimensions = context.cell().bounding_box().unwrap_or_else(Point::zeros);
            self.bulk_volume = box_volume - self.domain_volume[self.shell.len() - 1];
            const MIN_BULK_FRACTION: f64 = 0.01;
            anyhow::ensure!(
                self.bulk_volume > MIN_BULK_FRACTION * box_volume,
                "PreferentialInteraction: the domain leaves only {:.1}% of the cell as bulk, \
                 which is too little to reference Γ against. Reduce `shell.max` or enlarge the \
                 cell",
                100.0 * self.bulk_volume / box_volume
            );
            if self.bulk_volume < 0.1 * box_volume {
                log::warn!(
                    "preferential interaction: only {:.0}% of the cell is bulk; Γ is referenced \
                     against a thin shell and its error bar will be optimistic",
                    100.0 * self.bulk_volume / box_volume
                );
            }
        }
        Ok(())
    }

    fn is_dynamic(&self) -> bool {
        self.mode == GeometryMode::Dynamic
    }

    /// Credit a ligand at `r` to the residue that owns it, at every rung whose domain contains it.
    ///
    /// Ownership is recomputed per rung: the radical cell boundary between two balls of unequal
    /// radius moves with the probe, so a ligand near an interface can belong to different residues
    /// at different δ. Counting it once at its innermost rung and carrying that owner outward — the
    /// tempting shortcut — would misattribute it wherever the partition shifts. `counts` is the
    /// flat `[residue][rung]` tally; returns `true` if the ligand entered the domain at all.
    fn credit(&mut self, context: &impl ObserveContext, r: &Point, counts: &mut [f64]) -> bool {
        let n_shells = self.shell.len();
        let cell = context.cell();
        let mut surface_distance = f64::INFINITY;
        for ((slot, &i), &radius) in self.scratch.iter_mut().zip(&self.atoms).zip(&self.radii) {
            let d = cell.distance(r, &context.position(i)).norm();
            // Proximal distance: `r ∈ D(p)` iff `minᵢ(|r − rᵢ| − Rᵢ) ≤ p`, exactly. Its square,
            // less the atom radius squared, is the probe-free part of the power distance below.
            surface_distance = surface_distance.min(d - radius);
            *slot = d * d - radius * radius;
        }
        let Some(first) = self.shell.index(surface_distance - self.ligand_radius) else {
            return false;
        };

        for k in first..n_shells {
            // Power distance `dᵢ² − (Rᵢ + p)²` decides ownership at probe `p_k`, the same partition
            // that gave `volumes[k]`. Dropping the common `−p²` term leaves `(dᵢ² − Rᵢ²) − 2Rᵢp`,
            // so the cached `scratch` needs only a linear correction per rung, no squaring.
            let probe = self.ligand_radius + self.shell.delta(k);
            let atom = self
                .scratch
                .iter()
                .zip(&self.radii)
                .map(|(cached, r)| cached - 2.0 * r * probe)
                .enumerate()
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .expect("substrate is non-empty");
            counts[self.owner[atom] * n_shells + k] += 1.0;
        }
        true
    }
}

/// Per-ball tessellation quantity at `probe`.
///
/// `voronota-ltr` distinguishes a computed cell from a geometrically empty one. An unavailable
/// cell is an error because silently assigning it zero would bias the reference domain.
fn per_ball(
    balls: &[voronota_ltr::Ball],
    probe: f64,
    periodic_box: Option<&voronota_ltr::PeriodicBox>,
    extract: impl Fn(&voronota_ltr::TessellationResult) -> Vec<voronota_ltr::CellMeasure>,
) -> Result<Vec<f64>> {
    let result = voronota_ltr::compute_tessellation(balls, probe, periodic_box, None, false);
    extract(&result)
        .into_iter()
        .enumerate()
        .map(|(index, measure)| match measure {
            voronota_ltr::CellMeasure::Computed(value) => Ok(value),
            voronota_ltr::CellMeasure::Empty => Ok(0.0),
            voronota_ltr::CellMeasure::NotComputed => {
                anyhow::bail!(
                    "PreferentialInteraction: voronota-ltr did not compute cell {index} at probe \
                     {probe} Å"
                )
            }
        })
        .collect()
}

/// Per-ball cell volume at `probe`, clipped to the solvent-accessible surface.
fn cell_volumes(
    balls: &[voronota_ltr::Ball],
    probe: f64,
    periodic_box: Option<&voronota_ltr::PeriodicBox>,
) -> Result<Vec<f64>> {
    use voronota_ltr::Results as _;
    per_ball(balls, probe, periodic_box, |result| result.volumes())
}

/// Per-ball solvent-accessible surface area at `probe`.
fn surface_areas(
    balls: &[voronota_ltr::Ball],
    probe: f64,
    periodic_box: Option<&voronota_ltr::PeriodicBox>,
) -> Result<Vec<f64>> {
    use voronota_ltr::Results as _;
    per_ball(balls, probe, periodic_box, |result| result.sas_areas())
}

/// Preferential interaction coefficient of a ligand with a molecular substrate.
#[derive(Debug)]
pub struct PreferentialInteraction {
    ligand: ComSelection,
    solvent_reference: SolventReference,
    reference: SurfaceReference,
    /// Γ(δ_k), one accumulator per rung.
    gamma: Vec<WeightedBlockAverage>,
    /// γ(δ_k), the excess owned by each residue, indexed `[residue][shell]`.
    residue_gamma: Vec<Vec<WeightedBlockAverage>>,
    /// Mean ligand population owned by each residue and shell when the geometry or solvent
    /// reference is sampled.
    residue_counts: Vec<Vec<WeightedBlockAverage>>,
    /// Mean reference population for each residue and shell: `c₃ᵇv` in implicit solvent or
    /// `(N₃ᵇ/N₁ᵇ)N₁` in explicit solvent.
    reference_counts: Vec<Vec<WeightedBlockAverage>>,
    /// Frame-dependent residue geometry, indexed `[residue][shell]`.
    residue_volumes: Vec<Vec<WeightedBlockAverage>>,
    residue_water_volumes: Vec<Vec<WeightedBlockAverage>>,
    residue_asa: Vec<WeightedBlockAverage>,
    /// Frame-dependent total domain volume at each shell.
    domain_volume: Vec<WeightedBlockAverage>,
    /// Bulk concentration of the ligand (Å⁻³).
    concentration: WeightedBlockAverage,
    /// Per-sample scratch, reused to keep sampling allocation-free: ligand tally
    /// `[residue * n_shells + shell]`, per-shell totals, and the current ligand positions.
    counts: Vec<f64>,
    shell_totals: Vec<f64>,
    positions: Vec<Point>,
    /// Set once a frame violates a geometric constraint; further samples are skipped.
    stopped: bool,
    #[debug(skip)]
    profile_file: Option<PathBuf>,
    #[debug(skip)]
    residue_file: Option<PathBuf>,
    sampling: Sampling,
}

impl PreferentialInteraction {
    /// Γ at rung `k`, mean and standard error.
    pub(crate) fn gamma(&self, shell: usize) -> BlockSummary {
        self.gamma[shell].summary().unwrap_or_default()
    }

    /// Whether the analysis has stopped sampling after a geometry violation.
    #[cfg(test)]
    pub(crate) fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Mean ligands owned by `residue` at rung `shell` — the Kₚ numerator.
    ///
    fn residue_count(&self, residue: usize, shell: usize) -> f64 {
        if self.reference.is_dynamic() || self.solvent_reference.is_explicit() {
            self.residue_counts[residue][shell].mean()
        } else {
            self.residue_gamma[residue][shell].mean()
                + self.reference.volumes[shell][residue] * self.concentration.mean()
        }
    }

    /// γ of `residue` at rung `shell`. These sum to [`gamma`](Self::gamma) by construction: the
    /// counts partition the ligands and the volumes partition the domain.
    pub(crate) fn residue_gamma(&self, residue: usize, shell: usize) -> BlockSummary {
        self.residue_gamma[residue][shell]
            .summary()
            .unwrap_or_default()
    }

    /// Volume `residue` offers a ligand centre out to rung `shell`, `v(r_lig + δ) − v(r_lig)`.
    ///
    /// Occlusion lives here. The reference is built at probe `r_lig + δ`, so the envelope is the
    /// surface a ligand of *that* radius can reach: a bead whose cell is walled in by its
    /// neighbours never gains volume as δ grows, and a pocket that admits a small ion is sealed
    /// over for a larger one. It is the denominator of Kₚ, and the numerator of b₁.
    pub(crate) fn accessible_volume(&self, residue: usize, shell: usize) -> f64 {
        if self.reference.is_dynamic() {
            self.residue_volumes[residue][shell].mean() - self.residue_volumes[residue][0].mean()
        } else {
            self.reference.volumes[shell][residue] - self.reference.volumes[0][residue]
        }
    }

    /// Local-to-bulk partition coefficient Kₚ of `residue` at rung `shell`.
    ///
    /// Above one the ligand accumulates against this residue, below one it is excluded. Numerator
    /// and denominator both count only the *accessible* slab, so the rung-0 count and volume — the
    /// excluded interior, which a soft or off-centre ligand can still penetrate — are subtracted
    /// from each. An occluded residue offers no volume and so has no Kₚ, which is not Kₚ = 0.
    ///
    /// Both the accessible volume and the accessible count can turn negative when the radical
    /// partition transfers ownership between unequal-radius neighbours as the probe grows. Kₚ is a
    /// local-to-bulk concentration ratio, so a negative numerator or denominator has no meaning; it
    /// is reported as `None` rather than as a negative Kₚ.
    pub(crate) fn partition_coefficient(&self, residue: usize, shell: usize) -> Option<f64> {
        let reference = if self.reference.is_dynamic() || self.solvent_reference.is_explicit() {
            self.reference_counts[residue][shell].mean() - self.reference_counts[residue][0].mean()
        } else {
            self.concentration.mean() * self.accessible_volume(residue, shell)
        };
        let count = self.residue_count(residue, shell) - self.residue_count(residue, 0);
        (reference > 0.0 && count >= 0.0).then_some(count / reference)
    }

    fn surface_area(&self, residue: usize) -> f64 {
        if self.reference.is_dynamic() {
            self.residue_asa[residue].mean()
        } else {
            self.reference.residues[residue].asa
        }
    }

    /// Hydration per unit area b₁, in waters per Å².
    ///
    /// In implicit solvent the water population is the water-probe shell volume divided by the
    /// molecular volume of water. In explicit solvent it is the sampled solvent population in the
    /// same ligand-accessible slab used by Γ. A buried residue exposes no surface, so its b₁ is
    /// undefined rather than zero. A negative shell volume or population can arise when the
    /// radical partition transfers ownership between unequal-radius neighbours; a hydration
    /// density cannot be negative, so that too reads as undefined.
    pub(crate) fn hydration_density(&self, residue: usize, shell: usize) -> Option<f64> {
        let asa = self.surface_area(residue);
        match &self.solvent_reference {
            SolventReference::Implicit => {
                let water_shell = if self.reference.is_dynamic() {
                    self.residue_water_volumes[residue][shell].mean()
                        - self.residue_water_volumes[residue][0].mean()
                } else {
                    self.reference.water_volumes[shell][residue]
                        - self.reference.water_volumes[0][residue]
                };
                (asa > 0.0 && water_shell >= 0.0).then(|| water_shell / (WATER_VOLUME * asa))
            }
            SolventReference::Explicit(explicit) => {
                let waters = explicit.residue_counts[residue][shell].mean()
                    - explicit.residue_counts[residue][0].mean();
                (asa > 0.0 && waters >= 0.0).then_some(waters / asa)
            }
        }
    }

    /// Why the current reference geometry is invalid in the live cell, or `Ok` if it remains valid.
    ///
    /// A `VolumeMove` cannot be rejected at build (the analysis never sees the propagator), so a
    /// changed cell is caught here at the first drifted sample. A spherical cell adds a second
    /// check: voronota cannot clip to a hard wall, so a substrate whose domain reaches the wall has
    /// its free-space volume counting space no ligand can occupy. The caller stops *this* analysis
    /// on a violation rather than aborting the whole run — earlier samples remain valid.
    fn check_geometry_still_valid(&self, context: &impl ObserveContext) -> Result<()> {
        // Dimensions, not just volume: an anisotropic move can reshape a cuboid at constant volume.
        if let Some(current) = context.cell().bounding_box() {
            let reference = self.reference.box_dimensions;
            let drifted = (current - reference).abs().max() > 1e-6 * reference.min();
            anyhow::ensure!(
                !drifted,
                "PreferentialInteraction: the cell changed from {:.1?} to {:.1?} Å. The reference \
                 geometry is fixed at build, so this analysis requires a constant cell — remove \
                 the volume move",
                reference.as_slice(),
                current.as_slice()
            );
        }
        if let crate::cell::Cell::Sphere(sphere) = context.cell() {
            // Origin-centred; radius = half the bounding-box edge.
            let wall = sphere.bounding_box().map_or(f64::INFINITY, |b| b.x / 2.0);
            let farthest = self
                .reference
                .atoms
                .iter()
                .zip(&self.reference.radii)
                .map(|(&i, &r)| context.position(i).norm() + r)
                .fold(0.0, f64::max);
            anyhow::ensure!(
                farthest + self.reference.domain_reach <= wall,
                "PreferentialInteraction: the substrate's domain has reached the spherical wall \
                 (extends to {:.1} Å of the {:.1} Å radius); the domain volume then counts space \
                 outside the cell and Γ is biased. Keep the substrate away from the wall, enlarge \
                 the cell, or reduce `shell.max`",
                farthest + self.reference.domain_reach,
                wall
            );
        }
        Ok(())
    }

    /// Γ(δ) across the ladder: the profile that says whether Γ has converged.
    fn write_profile(&self, path: &Path) -> Result<()> {
        let mut writer = ColumnWriter::open(path, &["delta/Å", "gamma", "gamma_error"])?;
        for k in 0..self.reference.shell.len() {
            let gamma = self.gamma(k);
            writer.write_row(&[
                &format_args!("{:.4}", self.reference.shell.delta(k)),
                &format_args!("{:.6e}", gamma.mean),
                &format_args!("{:.6e}", gamma.error),
            ])?;
        }
        writer.flush()?;
        Ok(())
    }

    /// One row per residue, at the widest δ.
    fn write_residues(&self, path: &Path) -> Result<()> {
        let mut writer = ColumnWriter::open(
            path,
            &[
                "residue",
                "number",
                "asa/Å²",
                "accessible_volume/Å³",
                "b1/Å⁻²",
                "kp",
                "gamma",
                "gamma_error",
            ],
        )?;
        // A residue with no accessible surface offers neither a partition coefficient nor a
        // hydration density; both are undefined rather than zero, and go out as `nan`.
        let optional =
            |value: Option<f64>| value.map_or_else(|| "nan".to_owned(), |v| format!("{v:.6e}"));
        let last = self.reference.shell.len() - 1;
        for (index, residue) in self.reference.residues.iter().enumerate() {
            let gamma = self.residue_gamma(index, last);
            writer.write_row(&[
                &residue.name,
                &residue.number,
                &format_args!("{:.4}", self.surface_area(index)),
                &format_args!("{:.4}", self.accessible_volume(index, last)),
                &optional(self.hydration_density(index, last)),
                &optional(self.partition_coefficient(index, last)),
                &format_args!("{:.6e}", gamma.mean),
                &format_args!("{:.6e}", gamma.error),
            ])?;
        }
        writer.flush()?;
        Ok(())
    }

    fn report(&self) -> Option<serde_yml::Value> {
        let last = self.reference.shell.len() - 1;
        let mut map = serde_yml::Mapping::new();
        map.try_insert("num_samples", self.sampling.num_samples())?;
        map.try_insert("gamma", self.gamma(last))?;
        map.try_insert("concentration/Å⁻³", self.concentration.summary()?)?;
        if let SolventReference::Explicit(explicit) = &self.solvent_reference {
            map.try_insert(
                "solvent_concentration/Å⁻³",
                explicit.concentration.summary()?,
            )?;
            map.try_insert(
                "bulk_ligand_to_solvent_ratio",
                explicit.bulk_ligand_to_solvent_ratio.summary()?,
            )?;
        }
        let excluded_volume = if self.reference.is_dynamic() {
            self.domain_volume[0].mean()
        } else {
            self.reference.domain_volume[0]
        };
        map.try_insert("excluded_volume/Å³", excluded_volume)?;
        map.try_insert("ligand_radius/Å", self.reference.ligand_radius)?;
        map.try_insert("num_residues", self.reference.residues.len())?;
        Some(serde_yml::Value::Mapping(map))
    }

    /// Refill `self.positions` with the ligands of the current configuration.
    fn load_ligand_positions(&mut self, context: &impl ObserveContext) -> Result<()> {
        load_positions(&mut self.ligand, &mut self.positions, context, "ligand")
    }
}

fn load_positions(
    selection: &mut ComSelection,
    positions: &mut Vec<Point>,
    context: &impl ObserveContext,
    species: &str,
) -> Result<()> {
    positions.clear();
    match selection {
        ComSelection::Atoms(cache) => {
            for i in cache.resolve(context) {
                positions.push(context.position(i.get()));
            }
        }
        ComSelection::Groups(cache) => {
            for &g in cache.resolve(context) {
                let center = context.group(g).mass_center().copied().ok_or_else(|| {
                    anyhow::anyhow!(
                        "PreferentialInteraction: {species} group {g} has no center of mass"
                    )
                })?;
                positions.push(center);
            }
        }
    }
    Ok(())
}

impl_info!(
    PreferentialInteraction,
    "preferential_interaction",
    "Preferential interaction coefficient of a ligand with a molecular substrate"
);

impl<T: ObserveContext> Analyze<T> for PreferentialInteraction {
    impl_sampling_accessors!();

    fn perform_sample(&mut self, context: &T, step: usize, weight: f64) -> Result<()> {
        if self.stopped {
            return Ok(());
        }
        self.reference.refresh(context)?;
        if let Err(reason) = self.check_geometry_still_valid(context) {
            log::warn!("{reason}; this analysis stops sampling while the run continues");
            self.stopped = true;
            return Ok(());
        }
        let n_shells = self.reference.shell.len();
        // `counts[residue * n_shells + k]` holds the ligands owned by `residue` at rung k, already
        // resolved per rung by `credit`, so no prefix sum is needed here.
        self.counts.fill(0.0);
        self.load_ligand_positions(context)?;

        let mut bulk = 0.0f64;
        // A raw index keeps `self.positions` borrowed immutably while `credit` borrows `self.counts`
        // and `self.reference` mutably.
        for p in 0..self.positions.len() {
            let position = self.positions[p];
            if !self.reference.credit(context, &position, &mut self.counts) {
                bulk += 1.0;
            }
        }

        // Γ is a per-sample estimator, so the fluctuating bulk composition enters each sample
        // rather than being divided out afterwards. This retains its covariance with the local
        // populations and gives the block error estimator the quantity actually being averaged.
        let concentration = bulk / self.reference.bulk_volume;

        let bulk_ligand_to_solvent_ratio = match &mut self.solvent_reference {
            SolventReference::Implicit => 0.0,
            SolventReference::Explicit(explicit) => {
                explicit.counts.fill(0.0);
                load_positions(
                    &mut explicit.selection,
                    &mut explicit.positions,
                    context,
                    "solvent",
                )?;
                let mut solvent_bulk = 0.0;
                for p in 0..explicit.positions.len() {
                    let position = explicit.positions[p];
                    if !self
                        .reference
                        .credit(context, &position, &mut explicit.counts)
                    {
                        solvent_bulk += 1.0;
                    }
                }
                anyhow::ensure!(
                    solvent_bulk > 0.0,
                    "PreferentialInteraction: explicit solvent has no molecules in the bulk at \
                     step {step}; enlarge the cell or reduce `shell.max`"
                );
                explicit
                    .concentration
                    .add(solvent_bulk / self.reference.bulk_volume, weight);
                let ratio = bulk / solvent_bulk;
                explicit.bulk_ligand_to_solvent_ratio.add(ratio, weight);
                explicit.shell_totals.fill(0.0);
                ratio
            }
        };
        self.concentration.add(concentration, weight);

        self.shell_totals.fill(0.0);
        let dynamic = self.reference.is_dynamic();
        let sample_populations = dynamic || self.solvent_reference.is_explicit();
        for (residue, accumulators) in self.residue_gamma.iter_mut().enumerate() {
            if dynamic {
                self.residue_asa[residue].add(self.reference.residues[residue].asa, weight);
            }
            for (k, accumulator) in accumulators.iter_mut().enumerate() {
                let count = self.counts[residue * n_shells + k];
                let volume = self.reference.volumes[k][residue];
                let reference_count = match &mut self.solvent_reference {
                    SolventReference::Implicit => concentration * volume,
                    SolventReference::Explicit(explicit) => {
                        let solvent_count = explicit.counts[residue * n_shells + k];
                        explicit.residue_counts[residue][k].add(solvent_count, weight);
                        explicit.shell_totals[k] += solvent_count;
                        bulk_ligand_to_solvent_ratio * solvent_count
                    }
                };
                if sample_populations {
                    self.residue_counts[residue][k].add(count, weight);
                    self.reference_counts[residue][k].add(reference_count, weight);
                }
                if dynamic {
                    self.residue_volumes[residue][k].add(volume, weight);
                    self.residue_water_volumes[residue][k]
                        .add(self.reference.water_volumes[k][residue], weight);
                }
                accumulator.add(count - reference_count, weight);
                self.shell_totals[k] += count;
            }
        }
        for (k, accumulator) in self.gamma.iter_mut().enumerate() {
            if dynamic {
                self.domain_volume[k].add(self.reference.domain_volume[k], weight);
            }
            let reference_count = match &self.solvent_reference {
                SolventReference::Implicit => concentration * self.reference.domain_volume[k],
                SolventReference::Explicit(explicit) => {
                    bulk_ligand_to_solvent_ratio * explicit.shell_totals[k]
                }
            };
            accumulator.add(self.shell_totals[k] - reference_count, weight);
        }
        Ok(())
    }

    fn write_to_disk(&mut self) -> Result<()> {
        if self.sampling.num_samples() == 0 {
            return Ok(());
        }
        if let Some(path) = self.profile_file.clone() {
            self.write_profile(&path)?;
        }
        if let Some(path) = self.residue_file.clone() {
            self.write_residues(&path)?;
        }
        Ok(())
    }

    fn results(&self) -> Option<serde_yml::Value> {
        self.report()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;

    /// Eight ligands parked far enough out to be bulk at every rung of the ladder.
    const BULK: [[f64; 3]; 8] = [
        [25.0, 0.0, 0.0],
        [-25.0, 0.0, 0.0],
        [0.0, 25.0, 0.0],
        [0.0, -25.0, 0.0],
        [0.0, 0.0, 25.0],
        [0.0, 0.0, -25.0],
        [20.0, 20.0, 0.0],
        [-20.0, -20.0, 0.0],
    ];

    fn manual(positions: &[[f64; 3]]) -> String {
        positions
            .iter()
            .map(|p| format!("[{}, {}, {}]", p[0], p[1], p[2]))
            .collect::<Vec<_>>()
            .join(", ")
    }

    const BOX: f64 = 60.0;

    /// A rigid substrate and free ligands. The substrate is one rigid molecule — the analysis
    /// requires it, since it freezes the reference geometry once — with `sub_sigma` beads at the
    /// given body-frame positions; the ligands are a separate atomic species. Nothing interacts;
    /// the analysis is fed a configuration, not a trajectory.
    fn build_system(
        cell: f64,
        sub_sigma: f64,
        substrate: &[[f64; 3]],
        ligands: &[[f64; 3]],
    ) -> Backend {
        build_system_with_dof(cell, sub_sigma, substrate, ligands, "Rigid")
    }

    fn build_system_with_dof(
        cell: f64,
        sub_sigma: f64,
        substrate: &[[f64; 3]],
        ligands: &[[f64; 3]],
        degrees_of_freedom: &str,
    ) -> Backend {
        let structure = substrate
            .iter()
            .map(|p| format!("      - SUB: [{}, {}, {}]", p[0], p[1], p[2]))
            .collect::<Vec<_>>()
            .join("\n");
        let input = format!(
            r#"
atoms:
  - {{name: SUB, mass: 1.0, charge: 0.0, sigma: {sub_sigma}}}
  - {{name: LIG, mass: 1.0, charge: 0.0, sigma: 3.0}}
molecules:
  - name: substrate
    degrees_of_freedom: {degrees_of_freedom}
    from_structure:
{structure}
  - name: ligand
    atoms: [LIG]
    atomic: true
system:
  cell: !Cuboid [{cell}, {cell}, {cell}]
  medium: {{permittivity: !Vacuum, temperature: 300.0}}
  energy: {{}}
  blocks:
    - molecule: substrate
      N: 1
      insert: !Manual [{substrate_positions}]
    - molecule: ligand
      N: {n_ligands}
      insert: !Manual [{ligand_positions}]
propagate: {{seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}}
"#,
            substrate_positions = manual(substrate),
            n_ligands = ligands.len(),
            ligand_positions = manual(ligands),
        );
        Backend::from_yaml_str(&input, None, &mut rand::thread_rng()).unwrap()
    }

    fn build_flexible_dimer(separation: f64, ligands: &[[f64; 3]]) -> Backend {
        let ligand_positions = manual(ligands);
        let input = format!(
            r#"
atoms:
  - {{name: SUB, mass: 1.0, charge: 0.0, sigma: 6.0}}
  - {{name: LIG, mass: 1.0, charge: 0.0, sigma: 3.0}}
molecules:
  - name: substrate
    atoms: [SUB, SUB]
  - name: ligand
    atoms: [LIG]
    atomic: true
system:
  cell: !Cuboid [{BOX}, {BOX}, {BOX}]
  medium: {{permittivity: !Vacuum, temperature: 300.0}}
  energy: {{}}
  blocks:
    - molecule: substrate
      N: 1
      insert: !Manual [[{left}, 0.0, 0.0], [{right}, 0.0, 0.0]]
    - molecule: ligand
      N: {n_ligands}
      insert: !Manual [{ligand_positions}]
propagate: {{seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}}
"#,
            left = -0.5 * separation,
            right = 0.5 * separation,
            n_ligands = ligands.len(),
        );
        Backend::from_yaml_str(&input, None, &mut rand::thread_rng()).unwrap()
    }

    fn build_explicit_system(ligands: &[[f64; 3]], solvents: &[[f64; 3]]) -> Backend {
        let input = format!(
            r#"
atoms:
  - {{name: SUB, mass: 1.0, charge: 0.0, sigma: 6.0}}
  - {{name: LIG, mass: 1.0, charge: 0.0, sigma: 3.0}}
  - {{name: SOL, mass: 1.0, charge: 0.0, sigma: 2.8}}
molecules:
  - name: substrate
    degrees_of_freedom: Rigid
    atoms: [SUB]
  - name: ligand
    atoms: [LIG]
    atomic: true
  - name: solvent
    atoms: [SOL]
    atomic: true
system:
  cell: !Cuboid [{BOX}, {BOX}, {BOX}]
  medium: {{permittivity: !Vacuum, temperature: 300.0}}
  energy: {{}}
  blocks:
    - molecule: substrate
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0]]
    - molecule: ligand
      N: {n_ligands}
      insert: !Manual [{ligand_positions}]
    - molecule: solvent
      N: {n_solvents}
      insert: !Manual [{solvent_positions}]
propagate: {{seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}}
"#,
            n_ligands = ligands.len(),
            ligand_positions = manual(ligands),
            n_solvents = solvents.len(),
            solvent_positions = manual(solvents),
        );
        Backend::from_yaml_str(&input, None, &mut rand::thread_rng()).unwrap()
    }

    fn build_molecular_solvent_system() -> Backend {
        let input = r#"
atoms:
  - {name: SUB, mass: 1.0, charge: 0.0, sigma: 6.0}
  - {name: LIG, mass: 1.0, charge: 0.0, sigma: 3.0}
  - {name: SOL, mass: 1.0, charge: 0.0, sigma: 2.8}
molecules:
  - name: substrate
    degrees_of_freedom: Rigid
    atoms: [SUB]
  - name: ligand
    atoms: [LIG]
    atomic: true
  - name: water
    atoms: [SOL, SOL]
    has_com: true
system:
  cell: !Cuboid [60.0, 60.0, 60.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: substrate
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0]]
    - molecule: ligand
      N: 2
      insert: !Manual [[8.0, 0.0, 0.0], [25.0, 0.0, 0.0]]
    - molecule: water
      N: 4
      insert: !Manual
        - [-8.5, 0.0, 0.0]
        - [-7.5, 0.0, 0.0]
        - [-0.5, 8.0, 0.0]
        - [0.5, 8.0, 0.0]
        - [-25.5, 0.0, 0.0]
        - [-24.5, 0.0, 0.0]
        - [-0.5, 25.0, 0.0]
        - [0.5, 25.0, 0.0]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;
        Backend::from_yaml_str(input, None, &mut rand::thread_rng()).unwrap()
    }

    fn build_flexible_explicit_dimer(
        separation: f64,
        ligands: &[[f64; 3]],
        solvents: &[[f64; 3]],
    ) -> Backend {
        let input = format!(
            r#"
atoms:
  - {{name: SUB, mass: 1.0, charge: 0.0, sigma: 6.0}}
  - {{name: LIG, mass: 1.0, charge: 0.0, sigma: 3.0}}
  - {{name: SOL, mass: 1.0, charge: 0.0, sigma: 2.8}}
molecules:
  - name: substrate
    atoms: [SUB, SUB]
  - name: ligand
    atoms: [LIG]
    atomic: true
  - name: solvent
    atoms: [SOL]
    atomic: true
system:
  cell: !Cuboid [60.0, 60.0, 60.0]
  medium: {{permittivity: !Vacuum, temperature: 300.0}}
  energy: {{}}
  blocks:
    - molecule: substrate
      N: 1
      insert: !Manual [[{left}, 0.0, 0.0], [{right}, 0.0, 0.0]]
    - molecule: ligand
      N: {n_ligands}
      insert: !Manual [{ligand_positions}]
    - molecule: solvent
      N: {n_solvents}
      insert: !Manual [{solvent_positions}]
propagate: {{seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}}
"#,
            left = -0.5 * separation,
            right = 0.5 * separation,
            n_ligands = ligands.len(),
            ligand_positions = manual(ligands),
            n_solvents = solvents.len(),
            solvent_positions = manual(solvents),
        );
        Backend::from_yaml_str(&input, None, &mut rand::thread_rng()).unwrap()
    }

    fn equal_sphere_union_volume(radius: f64, separation: f64) -> f64 {
        let sphere = 4.0 * std::f64::consts::PI * radius.powi(3) / 3.0;
        if separation >= 2.0 * radius {
            return 2.0 * sphere;
        }
        let overlap = std::f64::consts::PI
            * (4.0 * radius + separation)
            * (2.0 * radius - separation).powi(2)
            / 12.0;
        2.0 * sphere - overlap
    }

    fn set_flexible_dimer_separation(context: &mut Backend, separation: f64) {
        use crate::Context as _;

        context
            .set_group_conformation(
                0,
                &[0, 1],
                &[
                    Point::new(-0.5 * separation, 0.0, 0.0),
                    Point::new(0.5 * separation, 0.0, 0.0),
                ],
            )
            .unwrap();
    }

    /// σ = 6 beads (R = 3), σ = 3 ligands (r = 1.5), in a 60 Å box.
    fn system(substrate: &[[f64; 3]], ligands: &[[f64; 3]]) -> Backend {
        build_system(BOX, 6.0, substrate, ligands)
    }

    fn system_in_box(cell: f64, substrate: &[[f64; 3]], ligands: &[[f64; 3]]) -> Backend {
        build_system(cell, 6.0, substrate, ligands)
    }

    /// One substrate bead at the origin.
    fn one_bead(ligands: &[[f64; 3]]) -> Backend {
        system(&[[0.0, 0.0, 0.0]], ligands)
    }

    /// A single substrate bead of the given radius, alone in a box large enough that its widest
    /// domain cannot reach its own periodic image.
    fn lone_bead(radius: f64) -> Backend {
        build_system(
            200.0,
            2.0 * radius,
            &[[0.0, 0.0, 0.0]],
            &[[90.0, 90.0, 90.0]],
        )
    }

    const SHELL: Shell = Shell {
        max: 10.0,
        resolution: 1.0,
    };

    const SOLVENT_PROBE: f64 = 1.4;

    fn analysis_with_radius(context: &Backend, radius: Option<f64>) -> PreferentialInteraction {
        PreferentialInteractionBuilder {
            substrate: Selection::parse("atomtype SUB").unwrap(),
            ligand: Selection::parse("atomtype LIG").unwrap(),
            use_com: false,
            solvent: None,
            radius,
            shell: SHELL,
            solvent_probe: SOLVENT_PROBE,
            profile: None,
            file: None,
            frequency: Frequency::Every(1),
        }
        .build(context)
        .unwrap()
    }

    fn analysis(context: &Backend) -> PreferentialInteraction {
        analysis_with_radius(context, None)
    }

    fn explicit_analysis(
        context: &Backend,
        solvent: &str,
        use_com: bool,
    ) -> PreferentialInteraction {
        PreferentialInteractionBuilder {
            substrate: Selection::parse("atomtype SUB").unwrap(),
            ligand: Selection::parse("atomtype LIG").unwrap(),
            use_com: false,
            solvent: Some(ExplicitSolvent {
                selection: Selection::parse(solvent).unwrap(),
                use_com,
            }),
            radius: None,
            shell: SHELL,
            solvent_probe: SOLVENT_PROBE,
            profile: None,
            file: None,
            frequency: Frequency::Every(1),
        }
        .build(context)
        .unwrap()
    }

    /// A bead at the origin walled in by six neighbours at ±`spacing` along the axes. Its Voronoi
    /// cell is the bounded cube the six bisector planes cut out, so it is occluded by
    /// construction — how thoroughly depends on the ligand's size.
    fn caged_bead(spacing: f64) -> Backend {
        let substrate = [
            [0.0, 0.0, 0.0],
            [spacing, 0.0, 0.0],
            [-spacing, 0.0, 0.0],
            [0.0, spacing, 0.0],
            [0.0, -spacing, 0.0],
            [0.0, 0.0, spacing],
            [0.0, 0.0, -spacing],
        ];
        system(&substrate, &BULK)
    }

    /// Volume of a sphere; the domain around a lone bead is one, so every reference volume in
    /// these tests is analytic and independent of the code under test.
    fn sphere(radius: f64) -> f64 {
        4.0 / 3.0 * std::f64::consts::PI * radius.powi(3)
    }

    const SUBSTRATE_RADIUS: f64 = 3.0;
    const LIGAND_RADIUS: f64 = 1.5;

    /// Vol(D(δ_k)) around the lone bead.
    fn domain(k: usize) -> f64 {
        sphere(SUBSTRATE_RADIUS + LIGAND_RADIUS + SHELL.delta(k))
    }

    /// Bulk concentration when `n` ligands sit outside the widest domain.
    fn concentration(n: f64) -> f64 {
        n / (BOX.powi(3) - domain(SHELL.len() - 1))
    }

    /// With every ligand in bulk, Γ(δ) is pure excluded volume — and for a lone sphere that volume
    /// is analytic, so the whole path (tessellation, ladder, bulk concentration, arithmetic) is
    /// pinned to a number computed independently of the code under test.
    #[test]
    fn gamma_of_a_lone_sphere_is_minus_the_excluded_volume() {
        let context = one_bead(&BULK);
        let mut analysis = analysis(&context);
        analysis.sample(&context, 0).unwrap();

        for k in 0..SHELL.len() {
            let expected = -concentration(8.0) * domain(k);
            assert!(
                (analysis.gamma(k).mean - expected).abs() < 1e-9,
                "rung {k}: Γ = {}, expected {expected}",
                analysis.gamma(k).mean
            );
        }
    }

    /// In explicit solvent the finite-box estimator compares ligand and solvent populations in
    /// the same domain. Matching the local 1:2 ratio to the bulk 1:2 ratio therefore gives Γ = 0,
    /// independently of the geometric domain volume.
    #[test]
    fn explicit_solvent_uniform_composition_has_zero_gamma() {
        let ligands = [[8.0, 0.0, 0.0], [25.0, 0.0, 0.0]];
        let solvents = [
            [-8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [-25.0, 0.0, 0.0],
            [0.0, 25.0, 0.0],
        ];
        let context = build_explicit_system(&ligands, &solvents);
        let mut analysis = explicit_analysis(&context, "atomtype SOL", false);

        analysis.sample(&context, 0).unwrap();
        let last = SHELL.len() - 1;
        let gamma = analysis.gamma(last).mean;
        assert!(gamma.abs() < 1e-12, "Γ = {gamma}, expected zero");
        float_cmp::assert_approx_eq!(
            f64,
            analysis.partition_coefficient(0, last).unwrap(),
            1.0,
            epsilon = 1e-12
        );
        float_cmp::assert_approx_eq!(
            f64,
            analysis.hydration_density(0, last).unwrap(),
            2.0 / analysis.surface_area(0),
            epsilon = 1e-12
        );

        let report = analysis.report().unwrap();
        let report = report.as_mapping().unwrap();
        let ratio = &report["bulk_ligand_to_solvent_ratio"];
        let ratio = ratio.as_mapping().unwrap()["mean"].as_f64().unwrap();
        float_cmp::assert_approx_eq!(f64, ratio, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn explicit_solvent_enrichment_matches_the_finite_box_estimator() {
        let ligands = [[8.0, 0.0, 0.0], [0.0, -8.0, 0.0], [25.0, 0.0, 0.0]];
        let solvents = [
            [-8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [-25.0, 0.0, 0.0],
            [0.0, 25.0, 0.0],
        ];
        let context = build_explicit_system(&ligands, &solvents);
        let mut analysis = explicit_analysis(&context, "atomtype SOL", false);

        analysis.sample(&context, 0).unwrap();
        let gamma = analysis.gamma(SHELL.len() - 1).mean;
        float_cmp::assert_approx_eq!(f64, gamma, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn explicit_solvent_requires_a_bulk_population() {
        let ligands = [[8.0, 0.0, 0.0], [25.0, 0.0, 0.0]];
        let solvents = [[-8.0, 0.0, 0.0], [0.0, 8.0, 0.0]];
        let context = build_explicit_system(&ligands, &solvents);
        let mut analysis = explicit_analysis(&context, "atomtype SOL", false);

        let error = analysis.sample(&context, 7).unwrap_err().to_string();
        assert!(
            error.contains("no molecules in the bulk at step 7"),
            "{error}"
        );
    }

    #[test]
    fn explicit_solvent_can_count_one_center_per_molecule() {
        let context = build_molecular_solvent_system();
        let mut analysis = explicit_analysis(&context, "molecule water", true);

        analysis.sample(&context, 0).unwrap();
        let last = SHELL.len() - 1;
        assert!(analysis.gamma(last).mean.abs() < 1e-12);
        let SolventReference::Explicit(explicit) = &analysis.solvent_reference else {
            unreachable!()
        };
        float_cmp::assert_approx_eq!(
            f64,
            explicit.residue_counts[0][last].mean(),
            2.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn rigid_alchemical_substrate_stays_static() {
        let context = build_system_with_dof(
            80.0,
            6.0,
            &[[0.0, 0.0, 0.0]],
            &[[25.0, 0.0, 0.0]],
            "RigidAlchemical",
        );
        let analysis = analysis(&context);
        assert!(
            !analysis.reference.is_dynamic(),
            "RigidAlchemical substrates should use the cached static geometry path"
        );
    }

    #[test]
    fn explicit_solvent_decomposition_is_exact_for_a_flexible_substrate() {
        let ligands = [[-11.0, 0.0, 0.0], [-6.0, 5.0, 0.0], [25.0, 0.0, 0.0]];
        let solvents = [
            [-6.0, -5.0, 0.0],
            [6.0, 5.0, 0.0],
            [-25.0, 0.0, 0.0],
            [0.0, 25.0, 0.0],
        ];
        let context = build_flexible_explicit_dimer(12.0, &ligands, &solvents);
        let mut analysis = explicit_analysis(&context, "atomtype SOL", false);

        analysis.sample(&context, 0).unwrap();
        let last = SHELL.len() - 1;
        let total = analysis.gamma(last).mean;
        let residues: Vec<f64> = (0..2)
            .map(|residue| analysis.residue_gamma(residue, last).mean)
            .collect();
        float_cmp::assert_approx_eq!(f64, total, 1.0, epsilon = 1e-12);
        float_cmp::assert_approx_eq!(f64, residues.iter().sum::<f64>(), total, epsilon = 1e-12);
        assert!(residues[0] > 0.0, "left residue γ = {}", residues[0]);
        assert!(residues[1] < 0.0, "right residue γ = {}", residues[1]);
    }

    #[test]
    fn explicit_solvent_honors_rerun_weights() {
        let solvents = [
            [-8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [-25.0, 0.0, 0.0],
            [0.0, 25.0, 0.0],
        ];
        let bulk_rich = build_explicit_system(
            &[[8.0, 0.0, 0.0], [25.0, 0.0, 0.0], [0.0, -25.0, 0.0]],
            &solvents,
        );
        let local_rich = build_explicit_system(
            &[[8.0, 0.0, 0.0], [0.0, -8.0, 0.0], [25.0, 0.0, 0.0]],
            &solvents,
        );
        let mut analysis = explicit_analysis(&bulk_rich, "atomtype SOL", false);

        analysis.sample_weighted(&bulk_rich, 0, 3.0).unwrap();
        analysis.sample_weighted(&local_rich, 1, 1.0).unwrap();
        float_cmp::assert_approx_eq!(
            f64,
            analysis.gamma(SHELL.len() - 1).mean,
            -0.5,
            epsilon = 1e-12
        );
    }

    #[test]
    fn ligand_and_explicit_solvent_must_be_distinct() {
        let context = build_explicit_system(
            &[[8.0, 0.0, 0.0], [25.0, 0.0, 0.0]],
            &[[-8.0, 0.0, 0.0], [-25.0, 0.0, 0.0]],
        );
        let builder = PreferentialInteractionBuilder {
            substrate: Selection::parse("atomtype SUB").unwrap(),
            ligand: Selection::parse("atomtype LIG").unwrap(),
            use_com: false,
            solvent: Some(ExplicitSolvent {
                selection: Selection::parse("atomtype LIG").unwrap(),
                use_com: false,
            }),
            radius: None,
            shell: SHELL,
            solvent_probe: SOLVENT_PROBE,
            profile: None,
            file: None,
            frequency: Frequency::Every(1),
        };
        let error = builder.build(&context).unwrap_err().to_string();
        assert!(error.contains("overlap"), "{error}");
    }

    /// Kₚ against a known local density. Six ligands sit 8 Å from a lone bead, inside the
    /// accessible slab and all owned by the single residue; eight more in bulk set the reference
    /// concentration. Then Kₚ = 6 / (c · v_acc) with c and v_acc analytic for the lone sphere.
    #[test]
    fn partition_coefficient_matches_a_known_local_density() {
        let slab = [
            [8.0, 0.0, 0.0],
            [-8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [0.0, -8.0, 0.0],
            [0.0, 0.0, 8.0],
            [0.0, 0.0, -8.0],
        ];
        let mut ligands = slab.to_vec();
        ligands.extend_from_slice(&BULK);
        let context = one_bead(&ligands);
        let mut analysis = analysis(&context);
        analysis.sample(&context, 0).unwrap();

        let last = SHELL.len() - 1;
        let v_acc = sphere(SUBSTRATE_RADIUS + LIGAND_RADIUS + SHELL.max)
            - sphere(SUBSTRATE_RADIUS + LIGAND_RADIUS);
        let expected = slab.len() as f64 / (concentration(BULK.len() as f64) * v_acc);
        let kp = analysis.partition_coefficient(0, last).unwrap();
        assert!(
            (kp - expected).abs() < 1e-6,
            "Kₚ = {kp}, expected {expected}"
        );
    }

    /// The same eight bulk ligands, plus one placed 8 Å from the bead centre. Its surface distance
    /// is 8 − 3 = 5 Å, so it clears the exclusion boundary by 5 − 1.5 = 3.5 Å and first enters the
    /// domain at δ = 4. Γ must therefore step by exactly one there and nowhere else — which pins
    /// the shell binning and the prefix sum together.
    #[test]
    fn a_ligand_enters_the_domain_at_the_rung_matching_its_gap() {
        let mut ligands = BULK.to_vec();
        ligands.push([8.0, 0.0, 0.0]);
        let context = one_bead(&ligands);
        let mut analysis = analysis(&context);
        analysis.sample(&context, 0).unwrap();

        for k in 0..SHELL.len() {
            let counted = f64::from(u8::from(k >= 4));
            let expected = counted - concentration(8.0) * domain(k);
            assert!(
                (analysis.gamma(k).mean - expected).abs() < 1e-9,
                "rung {k}: Γ = {}, expected {expected} (count {counted})",
                analysis.gamma(k).mean
            );
        }
    }

    /// Two overlapping beads: the domain is the union of two inflated spheres, whose volume is
    /// analytic. Their tessellation genuinely partitions a shared surface, so this pins
    /// `Results::volumes()` and the ownership argmin that must agree with it.
    #[test]
    fn gamma_of_two_overlapping_beads_matches_the_union_volume() {
        // A box wide enough that the widest domain (radius 14.5 Å) cannot reach its own image.
        const CELL: f64 = 80.0;
        const SEPARATION: f64 = 4.0;
        let substrate = [[-SEPARATION / 2.0, 0.0, 0.0], [SEPARATION / 2.0, 0.0, 0.0]];
        let bulk = [
            [35.0, 0.0, 0.0],
            [-35.0, 0.0, 0.0],
            [0.0, 35.0, 0.0],
            [0.0, -35.0, 0.0],
            [0.0, 0.0, 35.0],
            [0.0, 0.0, -35.0],
        ];
        let context = system_in_box(CELL, &substrate, &bulk);
        let mut analysis = analysis(&context);
        analysis.sample(&context, 0).unwrap();

        // Union of two equal spheres of radius `a` whose centres are `d` apart, with the lens
        // (their intersection) counted once: V = 2·(4/3)πa³ − π(4a + d)(2a − d)²/12.
        let union = |a: f64| {
            let lens =
                std::f64::consts::PI * (4.0 * a + SEPARATION) * (2.0 * a - SEPARATION).powi(2)
                    / 12.0;
            2.0 * sphere(a) - lens
        };
        let widest = union(SUBSTRATE_RADIUS + LIGAND_RADIUS + SHELL.max);
        let concentration = bulk.len() as f64 / (CELL.powi(3) - widest);

        for k in 0..SHELL.len() {
            let expected =
                -concentration * union(SUBSTRATE_RADIUS + LIGAND_RADIUS + SHELL.delta(k));
            let gamma = analysis.gamma(k).mean;
            assert!(
                (gamma - expected).abs() < 1e-6,
                "rung {k}: Γ = {gamma}, expected {expected}"
            );
        }
    }

    /// The decomposition is only worth having if it is exact. Σᵢ γᵢ = Γ holds because the counts
    /// partition the ligands and the volumes partition the domain — so it must hold for a lumpy
    /// multi-bead substrate with ligands scattered through the shells, not just for a lone sphere.
    #[test]
    fn per_atom_excess_sums_to_the_total() {
        let substrate = [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [2.0, 4.0, 0.0],
            [2.0, 1.5, 4.0],
        ];
        let ligands = [
            [9.0, 0.0, 0.0],
            [-6.0, 0.0, 0.0],
            [2.0, 11.0, 0.0],
            [2.0, 1.5, 9.5],
            [0.0, -7.0, 3.0],
            [25.0, 0.0, 0.0],
            [-25.0, 0.0, 0.0],
            [0.0, 25.0, 0.0],
        ];
        let context = system(&substrate, &ligands);
        let mut analysis = analysis(&context);
        analysis.sample(&context, 0).unwrap();

        for k in 0..SHELL.len() {
            let sum: f64 = (0..substrate.len())
                .map(|residue| analysis.residue_gamma(residue, k).mean)
                .sum();
            let total = analysis.gamma(k).mean;
            assert!(
                (sum - total).abs() < 1e-9,
                "rung {k}: Σᵢ γᵢ = {sum}, Γ = {total}"
            );
        }
    }

    /// On a substrate of unequal radii the radical cell boundary moves with the probe, so a ligand
    /// near an interface can belong to different residues at different δ. Ownership must therefore
    /// be resolved per rung; carrying a single innermost owner outward misattributes it. The
    /// expected per-rung owner is recomputed here from the power distance, independently of the
    /// analysis, so the test fails if the analysis ever reverts to a fixed owner.
    #[test]
    #[allow(clippy::needless_range_loop)] // the brute-force reference indexes [rung][residue] directly
    fn ownership_follows_the_radical_partition_at_each_rung() {
        // Two beads: small A (σ = 2, R = 1) at the origin, large B (σ = 10, R = 5) at 10 Å. The
        // A–B radical plane sweeps from x ≈ 3.2 at δ = 0 toward A as δ grows, so ligands parked
        // between those positions change hands.
        const MIXED: &str = r#"
atoms:
  - {name: A, mass: 1.0, charge: 0.0, sigma: 2.0}
  - {name: B, mass: 1.0, charge: 0.0, sigma: 10.0}
  - {name: LIG, mass: 1.0, charge: 0.0, sigma: 3.0}
molecules:
  - name: substrate
    degrees_of_freedom: Rigid
    from_structure:
      - A: [0.0, 0.0, 0.0]
      - B: [10.0, 0.0, 0.0]
  - name: ligand
    atoms: [LIG]
    atomic: true
system:
  cell: !Cuboid [200.0, 200.0, 200.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: substrate
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
    - molecule: ligand
      N: 4
      insert: !Manual [[3.0, 1.0, 0.0], [4.0, 0.0, 2.0], [5.0, 2.0, 0.0], [2.5, 0.0, 1.5]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;
        let ligands = [
            [3.0, 1.0, 0.0],
            [4.0, 0.0, 2.0],
            [5.0, 2.0, 0.0],
            [2.5, 0.0, 1.5],
        ];
        let beads = [([0.0, 0.0, 0.0], 1.0), ([10.0, 0.0, 0.0], 5.0)];
        let r_lig = 1.5;

        let context = Backend::from_yaml_str(MIXED, None, &mut rand::thread_rng()).unwrap();
        let mut analysis = PreferentialInteractionBuilder {
            substrate: Selection::parse("molecule substrate").unwrap(),
            ligand: Selection::parse("atomtype LIG").unwrap(),
            use_com: false,
            solvent: None,
            radius: Some(r_lig),
            shell: SHELL,
            solvent_probe: SOLVENT_PROBE,
            profile: None,
            file: None,
            frequency: Frequency::Every(1),
        }
        .build(&context)
        .unwrap();
        analysis.sample(&context, 0).unwrap();

        // Independent brute force: per ligand, per rung it is inside, the owner is the bead
        // minimising the power distance at that rung's probe.
        let mut expected = vec![[0.0f64; 2]; SHELL.len()];
        for lig in ligands {
            let dist = |b: usize| {
                let c: [f64; 3] = beads[b].0;
                ((lig[0] - c[0]).powi(2) + (lig[1] - c[1]).powi(2) + (lig[2] - c[2]).powi(2)).sqrt()
            };
            let surface = (0..2)
                .map(|b| dist(b) - beads[b].1)
                .fold(f64::INFINITY, f64::min);
            let Some(first) = SHELL.index(surface - r_lig) else {
                continue;
            };
            for k in first..SHELL.len() {
                let probe = r_lig + SHELL.delta(k);
                let owner = (0..2)
                    .min_by(|&a, &b| {
                        let power = |i: usize| dist(i).powi(2) - (beads[i].1 + probe).powi(2);
                        power(a).total_cmp(&power(b))
                    })
                    .unwrap();
                expected[k][owner] += 1.0;
            }
        }

        for k in 0..SHELL.len() {
            for residue in 0..2 {
                assert!(
                    (analysis.residue_count(residue, k) - expected[k][residue]).abs() < 1e-9,
                    "rung {k}, residue {residue}: count {}, expected {}",
                    analysis.residue_count(residue, k),
                    expected[k][residue]
                );
            }
        }
        // The point of the test: at least one ligand actually changes hands across the ladder.
        let flips = (0..2).any(|residue| {
            (1..SHELL.len()).any(|k| expected[k][residue] != expected[k - 1][residue])
        });
        assert!(flips, "test is vacuous: no ligand changed owner across δ");
    }

    /// Accumulation at one residue and exclusion at another, in the same substrate and the same
    /// frame. This is the shape every real result takes — the Hofmeister series is a table of such
    /// reversals — so the decomposition has to be able to represent it, not average it away.
    #[test]
    fn one_atom_can_accumulate_while_another_excludes() {
        let substrate = [[-8.0, 0.0, 0.0], [8.0, 0.0, 0.0]];
        // Four ligands crowded against the first bead; none near the second.
        let ligands = [
            [-13.0, 0.0, 0.0],
            [-8.0, 5.0, 0.0],
            [-8.0, -5.0, 0.0],
            [-8.0, 0.0, 5.0],
            [25.0, 25.0, 0.0],
            [-25.0, -25.0, 0.0],
        ];
        let context = system(&substrate, &ligands);
        let mut analysis = analysis(&context);
        analysis.sample(&context, 0).unwrap();

        let last = SHELL.len() - 1;
        let crowded = analysis.residue_gamma(0, last).mean;
        let bare = analysis.residue_gamma(1, last).mean;
        assert!(
            crowded > 0.0,
            "crowded atom should accumulate, got γ = {crowded}"
        );
        assert!(bare < 0.0, "bare atom should exclude, got γ = {bare}");
    }

    /// A bead walled in tightly enough that its cell lies wholly inside its own solvent-accessible
    /// sphere offers a ligand *no* volume, at any δ. A distance cutoff would still count ligands
    /// against it; the Voronoi reference cannot, because the geometry gives it nothing to hold.
    #[test]
    fn a_buried_atom_offers_no_accessible_volume() {
        let context = caged_bead(5.0);
        let mut analysis = analysis(&context);
        analysis.sample(&context, 0).unwrap();

        for k in 0..SHELL.len() {
            assert!(
                analysis.accessible_volume(0, k) < 1e-9,
                "rung {k}: buried atom has accessible volume {}",
                analysis.accessible_volume(0, k)
            );
        }
        // A neighbour on the outside of the cage is not occluded, so the test cannot pass by
        // reporting zero for everything.
        assert!(analysis.accessible_volume(1, SHELL.len() - 1) > 0.0);
    }

    /// The same cage, loosened. A small ligand reaches the caged bead through the gaps; a larger
    /// one does not, because the accessible surface seals over. Occlusion is therefore a property
    /// of the *pair*, not of the substrate alone — which is exactly what a distance cutoff, blind
    /// to the ligand's size, cannot express.
    #[test]
    fn occlusion_depends_on_the_size_of_the_ligand() {
        let context = caged_bead(6.5);
        let last = SHELL.len() - 1;

        let mut small = analysis_with_radius(&context, Some(1.5));
        small.sample(&context, 0).unwrap();
        let reachable = small.accessible_volume(0, last);

        let mut large = analysis_with_radius(&context, Some(4.0));
        large.sample(&context, 0).unwrap();
        let sealed = large.accessible_volume(0, last);

        assert!(
            reachable > 1.0,
            "the small ligand should reach the caged bead, got {reachable} Å³"
        );
        assert!(
            sealed < 1e-9,
            "the large ligand should be sealed out, got {sealed} Å³"
        );
    }

    /// A tiny bead buried inside a large one has an empty weighted cell and therefore zero volume,
    /// while a detached bead has a computed full-sphere cell. Confusing those states would inflate
    /// the domain and bias Γ by −c·V_spurious.
    #[test]
    fn an_engulfed_atom_has_an_empty_cell() {
        // A σ = 1 bead (R = 0.5) sitting 5 Å off the centre of a σ = 30 bead (R = 15): its whole
        // inflated sphere lies inside the big one, so its radical cell is empty.
        const ENGULFED: &str = r#"
atoms:
  - {name: BIG, mass: 1.0, charge: 0.0, sigma: 30.0}
  - {name: DOT, mass: 1.0, charge: 0.0, sigma: 1.0}
  - {name: LIG, mass: 1.0, charge: 0.0, sigma: 3.0}
molecules:
  - name: substrate
    degrees_of_freedom: Rigid
    from_structure:
      - BIG: [0.0, 0.0, 0.0]
      - DOT: [5.0, 0.0, 0.0]
  - name: ligand
    atoms: [LIG]
    atomic: true
system:
  cell: !Cuboid [200.0, 200.0, 200.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: substrate
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    - molecule: ligand
      N: 1
      insert: !Manual [[90.0, 90.0, 90.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;
        let context = Backend::from_yaml_str(ENGULFED, None, &mut rand::thread_rng()).unwrap();
        let mut analysis = PreferentialInteractionBuilder {
            substrate: Selection::parse("molecule substrate").unwrap(),
            ligand: Selection::parse("atomtype LIG").unwrap(),
            use_com: false,
            solvent: None,
            radius: Some(1.5),
            shell: SHELL,
            solvent_probe: SOLVENT_PROBE,
            profile: None,
            file: None,
            frequency: Frequency::Every(1),
        }
        .build(&context)
        .unwrap();
        analysis.sample(&context, 0).unwrap();

        // The engulfed dot (residue 1) offers essentially nothing; the big bead (residue 0) offers
        // the whole domain. A free-sphere fallback would have given the dot ~14 Å³ at δ = 0.
        let last = SHELL.len() - 1;
        assert!(
            analysis.accessible_volume(1, last) < 1e-6,
            "engulfed atom should offer no volume, got {} Å³",
            analysis.accessible_volume(1, last)
        );
    }

    /// With unequal radii the radical partition shifts with the probe: a larger neighbour's cell
    /// grows faster and can take volume from a smaller bead, so its accessible shell volume goes
    /// negative. Kₚ (a concentration ratio) and b₁ (a hydration density) cannot be negative, so an
    /// unphysical shell volume must read as `nan`, not as a negative number.
    #[test]
    fn ownership_transfer_never_yields_a_negative_kp_or_b1() {
        // A σ = 6 bead half-caged by five σ = 30 beads: open on −z, so it keeps a water-accessible
        // face (asa > 0), yet the five neighbours overrun its cell as the probe grows, driving its
        // water shell volume negative. This is the lysozyme regime in miniature — a partly exposed
        // bead whose hydration density would read negative without the guard.
        const CLUSTER: &str = r#"
atoms:
  - {name: BIG, mass: 1.0, charge: 0.0, sigma: 30.0}
  - {name: MED, mass: 1.0, charge: 0.0, sigma: 6.0}
  - {name: LIG, mass: 1.0, charge: 0.0, sigma: 3.0}
molecules:
  - name: substrate
    degrees_of_freedom: Rigid
    from_structure:
      - MED: [0.0, 0.0, 0.0]
      - BIG: [17.0, 0.0, 0.0]
      - BIG: [-17.0, 0.0, 0.0]
      - BIG: [0.0, 17.0, 0.0]
      - BIG: [0.0, -17.0, 0.0]
      - BIG: [0.0, 0.0, 17.0]
  - name: ligand
    atoms: [LIG]
    atomic: true
system:
  cell: !Cuboid [200.0, 200.0, 200.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: substrate
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0], [17.0, 0.0, 0.0], [-17.0, 0.0, 0.0],
                       [0.0, 17.0, 0.0], [0.0, -17.0, 0.0], [0.0, 0.0, 17.0]]
    - molecule: ligand
      N: 1
      insert: !Manual [[90.0, 90.0, 90.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;
        let context = Backend::from_yaml_str(CLUSTER, None, &mut rand::thread_rng()).unwrap();
        let mut analysis = PreferentialInteractionBuilder {
            substrate: Selection::parse("molecule substrate").unwrap(),
            ligand: Selection::parse("atomtype LIG").unwrap(),
            use_com: false,
            solvent: None,
            radius: Some(1.5),
            shell: SHELL,
            solvent_probe: SOLVENT_PROBE,
            profile: None,
            file: None,
            frequency: Frequency::Every(1),
        }
        .build(&context)
        .unwrap();
        analysis.sample(&context, 0).unwrap();

        // The medium beads lose shell volume as the probe grows; the reported partition coefficient
        // and hydration density must stay `None`-or-non-negative rather than turning negative.
        let mut saw_transfer = false;
        for residue in 0..analysis.reference.residues.len() {
            for shell in 0..SHELL.len() {
                if analysis.accessible_volume(residue, shell) < 0.0 {
                    saw_transfer = true;
                }
                if let Some(kp) = analysis.partition_coefficient(residue, shell) {
                    assert!(
                        kp >= 0.0,
                        "negative Kₚ = {kp} at residue {residue}, shell {shell}"
                    );
                }
                if let Some(b1) = analysis.hydration_density(residue, shell) {
                    assert!(
                        b1 >= 0.0,
                        "negative b₁ = {b1} at residue {residue}, shell {shell}"
                    );
                }
            }
        }
        assert!(
            saw_transfer,
            "geometry did not exercise ownership transfer; test is vacuous"
        );
    }

    /// Several neighbours can eliminate a weighted cell even though none contains the atom by
    /// itself. Such an atom offers zero reference volume; treating the missing cell as a detached
    /// full sphere creates the large-probe δ³ divergence seen for molecular substrates.
    #[test]
    fn a_collectively_hidden_atom_has_zero_excess() {
        const COLLECTIVELY_HIDDEN: &str = r#"
atoms:
  - {name: A, mass: 1.0, charge: 0.0, sigma: 4.00}
  - {name: B, mass: 1.0, charge: 0.0, sigma: 5.18}
  - {name: C, mass: 1.0, charge: 0.0, sigma: 5.58}
  - {name: D, mass: 1.0, charge: 0.0, sigma: 4.00}
  - {name: E, mass: 1.0, charge: 0.0, sigma: 4.50}
  - {name: F, mass: 1.0, charge: 0.0, sigma: 6.56}
  - {name: G, mass: 1.0, charge: 0.0, sigma: 5.62}
  - {name: LIG, mass: 1.0, charge: 0.0, sigma: 3.0}
molecules:
  - name: substrate
    degrees_of_freedom: Rigid
    from_structure:
      - A: [0.4025, 0.1675, -3.5540]
      - B: [2.8625, -3.5723, 0.7652]
      - C: [0.5942, 1.6715, 0.1783]
      - D: [0.0, 0.0, 0.0]
      - E: [-1.3506, 3.6633, 2.9246]
      - F: [-3.8649, 0.5584, -0.3824]
      - G: [-1.2191, -2.0496, 3.8379]
  - name: ligand
    atoms: [LIG]
    atomic: true
system:
  cell: !Cuboid [100.0, 100.0, 100.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: substrate
      N: 1
      insert: !Manual
        - [0.4025, 0.1675, -3.5540]
        - [2.8625, -3.5723, 0.7652]
        - [0.5942, 1.6715, 0.1783]
        - [0.0, 0.0, 0.0]
        - [-1.3506, 3.6633, 2.9246]
        - [-3.8649, 0.5584, -0.3824]
        - [-1.2191, -2.0496, 3.8379]
    - molecule: ligand
      N: 6
      insert: !Manual
        - [45.0, 0.0, 0.0]
        - [-45.0, 0.0, 0.0]
        - [0.0, 45.0, 0.0]
        - [0.0, -45.0, 0.0]
        - [0.0, 0.0, 45.0]
        - [0.0, 0.0, -45.0]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;
        let context =
            Backend::from_yaml_str(COLLECTIVELY_HIDDEN, None, &mut rand::thread_rng()).unwrap();
        let mut analysis = PreferentialInteractionBuilder {
            substrate: Selection::parse("molecule substrate").unwrap(),
            ligand: Selection::parse("atomtype LIG").unwrap(),
            use_com: false,
            solvent: None,
            radius: Some(5.0),
            shell: Shell {
                max: 1.0,
                resolution: 1.0,
            },
            solvent_probe: SOLVENT_PROBE,
            profile: None,
            file: None,
            frequency: Frequency::Every(1),
        }
        .build(&context)
        .unwrap();

        analysis.sample(&context, 0).unwrap();

        assert!(
            analysis.residue_gamma(3, 0).mean.abs() < 1e-12,
            "collectively hidden atom should have γ = 0, got {}",
            analysis.residue_gamma(3, 0).mean
        );
    }

    #[test]
    fn a_degenerate_shell_ladder_is_refused() {
        let build = |shell: Shell| {
            PreferentialInteractionBuilder {
                substrate: Selection::parse("molecule substrate").unwrap(),
                ligand: Selection::parse("atomtype LIG").unwrap(),
                use_com: false,
                solvent: None,
                radius: Some(1.5),
                shell,
                solvent_probe: SOLVENT_PROBE,
                profile: None,
                file: None,
                frequency: Frequency::Every(1),
            }
            .build(&one_bead(&BULK))
        };
        assert!(build(Shell {
            max: 10.0,
            resolution: 0.0
        })
        .is_err());
        assert!(build(Shell {
            max: 10.0,
            resolution: -1.0
        })
        .is_err());
        assert!(build(Shell {
            max: -5.0,
            resolution: 1.0
        })
        .is_err());
        // shell.max not a whole multiple of the resolution would misplace the bulk boundary.
        assert!(build(Shell {
            max: 8.0,
            resolution: 0.3
        })
        .is_err());
    }

    #[test]
    fn each_flexible_conformation_uses_its_own_domain_volume() {
        let mut context = build_flexible_dimer(4.0, &BULK);
        let mut analysis = analysis(&context);
        analysis.sample(&context, 0).unwrap();

        let separation = 12.0;
        set_flexible_dimer_separation(&mut context, separation);
        analysis.sample(&context, 1).unwrap();

        let inflated_radius = 3.0 + 1.5 + SHELL.max;
        let box_volume = BOX.powi(3);
        let expected = [4.0, separation]
            .map(|distance| {
                let domain = equal_sphere_union_volume(inflated_radius, distance);
                -(BULK.len() as f64) * domain / (box_volume - domain)
            })
            .into_iter()
            .sum::<f64>()
            / 2.0;
        float_cmp::assert_approx_eq!(
            f64,
            analysis.gamma(SHELL.len() - 1).mean,
            expected,
            epsilon = 1e-8
        );
    }

    #[test]
    fn flexible_geometry_output_is_averaged_over_sampled_conformations() {
        let mut context = build_flexible_dimer(4.0, &BULK);
        let mut analysis = analysis(&context);
        analysis.sample(&context, 0).unwrap();

        let separation = 12.0;
        set_flexible_dimer_separation(&mut context, separation);
        analysis.sample(&context, 1).unwrap();

        let inflated_radius = 3.0 + 1.5;
        let expected = [4.0, separation]
            .map(|distance| equal_sphere_union_volume(inflated_radius, distance))
            .into_iter()
            .sum::<f64>()
            / 2.0;
        let report = analysis.report().unwrap();
        let key = serde_yml::Value::String("excluded_volume/Å³".to_owned());
        let actual = report.as_mapping().unwrap()[&key].as_f64().unwrap();
        float_cmp::assert_approx_eq!(f64, actual, expected, epsilon = 1e-8);
    }

    #[test]
    fn flexible_conformations_honor_rerun_weights() {
        let mut context = build_flexible_dimer(4.0, &BULK);
        let mut analysis = analysis(&context);
        analysis.sample_weighted(&context, 0, 3.0).unwrap();

        let separation = 12.0;
        set_flexible_dimer_separation(&mut context, separation);
        analysis.sample_weighted(&context, 1, 1.0).unwrap();

        let inflated_radius = 3.0 + 1.5 + SHELL.max;
        let box_volume = BOX.powi(3);
        let gamma = [4.0, separation].map(|distance| {
            let domain = equal_sphere_union_volume(inflated_radius, distance);
            -(BULK.len() as f64) * domain / (box_volume - domain)
        });
        let expected = (3.0 * gamma[0] + gamma[1]) / 4.0;
        float_cmp::assert_approx_eq!(
            f64,
            analysis.gamma(SHELL.len() - 1).mean,
            expected,
            epsilon = 1e-8
        );
    }

    #[test]
    fn a_flexible_substrate_uses_the_current_cell() {
        use crate::cell::VolumeScalePolicy;
        use crate::context::WithSimulationCell as _;
        use crate::Context as _;

        let mut context = build_flexible_dimer(4.0, &BULK);
        let mut analysis = analysis(&context);
        analysis.sample(&context, 0).unwrap();

        let initial_volume = context.cell().volume().unwrap();
        context
            .scale_volume_and_positions(2.0 * initial_volume, VolumeScalePolicy::ScaleZ)
            .unwrap();
        analysis.sample(&context, 1).unwrap();

        assert!(!analysis.is_stopped());
        let domain = equal_sphere_union_volume(3.0 + 1.5 + SHELL.max, 4.0);
        let expected = -(BULK.len() as f64)
            * 0.5
            * (domain / (initial_volume - domain) + domain / (2.0 * initial_volume - domain));
        float_cmp::assert_approx_eq!(
            f64,
            analysis.gamma(SHELL.len() - 1).mean,
            expected,
            epsilon = 1e-8
        );
    }

    #[test]
    fn residue_output_averages_flexible_surface_area() {
        let mut context = build_flexible_dimer(4.0, &BULK);
        let mut analysis = analysis(&context);
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("residues.csv");
        analysis.residue_file = Some(output.clone());
        analysis.sample(&context, 0).unwrap();

        let separation = 12.0;
        set_flexible_dimer_separation(&mut context, separation);
        analysis.sample(&context, 1).unwrap();
        <PreferentialInteraction as Analyze<Backend>>::write_to_disk(&mut analysis).unwrap();

        let contents = std::fs::read_to_string(output).unwrap();
        let asa: f64 = contents
            .lines()
            .nth(1)
            .unwrap()
            .split(',')
            .nth(2)
            .unwrap()
            .parse()
            .unwrap();
        let radius = 3.0 + SOLVENT_PROBE;
        let exposed_at_four =
            2.0 * std::f64::consts::PI * radius.powi(2) + std::f64::consts::PI * radius * 4.0;
        let exposed_at_twelve = 4.0 * std::f64::consts::PI * radius.powi(2);
        let expected = 0.5 * (exposed_at_four + exposed_at_twelve);
        float_cmp::assert_approx_eq!(f64, asa, expected, epsilon = 1e-4);
    }

    /// Under periodic boundaries the domain must fit within half the shortest edge, or it wraps
    /// onto its own image and the minimum-image tessellation is wrong. In a 25 Å box the σ = 6 bead
    /// (R = 3) plus r_lig 1.5 plus δ_max 10 reaches 14.5 Å > 12.5 Å, so the build must refuse it.
    #[test]
    fn a_domain_wider_than_half_the_box_is_refused() {
        let context = build_system(25.0, 6.0, &[[0.0, 0.0, 0.0]], &[[5.0, 0.0, 0.0]]);
        let builder = PreferentialInteractionBuilder {
            substrate: Selection::parse("molecule substrate").unwrap(),
            ligand: Selection::parse("atomtype LIG").unwrap(),
            use_com: false,
            solvent: None,
            radius: Some(1.5),
            shell: SHELL,
            solvent_probe: SOLVENT_PROBE,
            profile: None,
            file: None,
            frequency: Frequency::Every(1),
        };
        assert!(builder.build(&context).is_err());
    }

    /// The b₁ water ladder reaches `solvent_probe + δ_max`, wider than the ligand ladder when
    /// `use_com` sets r_lig = 0. In a 27 Å box (half-edge 13.5) the ligand reach 3 + 0 + 10 = 13
    /// fits but the water reach 3 + 1.4 + 10 = 14.4 does not, so only a guard that covers the water
    /// ladder refuses it — the box is chosen to fail the fixed code and pass the old ligand-only one.
    #[test]
    fn the_guard_covers_the_water_ladder_not_only_the_ligand() {
        let dimer = |cell: f64| {
            build_system(
                cell,
                6.0,
                &[[0.0, 0.0, 0.0]],
                &[[7.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
            )
        };
        let builder = |context: &Backend| {
            PreferentialInteractionBuilder {
                substrate: Selection::parse("molecule substrate").unwrap(),
                ligand: Selection::parse("atomtype LIG").unwrap(),
                use_com: true, // r_lig defaults to 0; the water ladder is the wider one
                solvent: None,
                radius: None,
                shell: SHELL,
                solvent_probe: SOLVENT_PROBE,
                profile: None,
                file: None,
                frequency: Frequency::Every(1),
            }
            .build(context)
        };
        assert!(builder(&dimer(27.0)).is_err());
        assert!(builder(&dimer(40.0)).is_ok());
    }

    /// A rigid substrate straddling the periodic boundary is legitimate (the tessellation reconnects
    /// it by minimum image), so the self-overlap guard must measure its true extent, not the
    /// box-wide spread of its raw coordinates.
    #[test]
    fn a_substrate_straddling_the_boundary_is_not_falsely_rejected() {
        // Beads at ±19 in a 40 Å box: 38 Å apart by raw coordinate, 2 Å by minimum image.
        let context = build_system(40.0, 6.0, &[[19.0, 0.0, 0.0], [-19.0, 0.0, 0.0]], &BULK);
        let builder = PreferentialInteractionBuilder {
            substrate: Selection::parse("molecule substrate").unwrap(),
            ligand: Selection::parse("atomtype LIG").unwrap(),
            use_com: false,
            solvent: None,
            radius: Some(1.5),
            shell: Shell {
                max: 5.0,
                resolution: 1.0,
            },
            solvent_probe: SOLVENT_PROBE,
            profile: None,
            file: None,
            frequency: Frequency::Every(1),
        };
        assert!(builder.build(&context).is_ok());
    }

    /// A cell reshaped at constant volume (stretch z, shrink xy back) must stop sampling — a
    /// volume-only check would miss it, so the guard compares box dimensions.
    #[test]
    fn a_reshaped_cell_stops_sampling() {
        use crate::cell::VolumeScalePolicy;
        use crate::context::WithSimulationCell as _;
        use crate::Context as _;

        let mut context = one_bead(&BULK);
        let mut analysis = analysis(&context);
        analysis.sample(&context, 0).unwrap();

        let volume = context.cell().volume().unwrap();
        // z doubles (volume ×2), then xy shrinks to restore the original volume: same volume,
        // different shape.
        context
            .scale_volume_and_positions(2.0 * volume, VolumeScalePolicy::ScaleZ)
            .unwrap();
        context
            .scale_volume_and_positions(volume, VolumeScalePolicy::ScaleXY)
            .unwrap();
        assert!((context.cell().volume().unwrap() - volume).abs() < 1e-6);

        // The sample succeeds (the run is not aborted) but the analysis stops accumulating, so its
        // Γ is unchanged by the reshaped-cell frame.
        let gamma_before = analysis.gamma(0).mean;
        analysis.sample(&context, 1).unwrap();
        assert!(analysis.is_stopped());
        assert_eq!(analysis.gamma(0).mean, gamma_before);
    }

    #[test]
    fn a_negative_radius_or_probe_is_refused() {
        let context = one_bead(&BULK);
        let build = |radius: Option<f64>, solvent_probe: f64| {
            PreferentialInteractionBuilder {
                substrate: Selection::parse("molecule substrate").unwrap(),
                ligand: Selection::parse("atomtype LIG").unwrap(),
                use_com: false,
                solvent: None,
                radius,
                shell: SHELL,
                solvent_probe,
                profile: None,
                file: None,
                frequency: Frequency::Every(1),
            }
            .build(&context)
        };
        assert!(build(Some(-1.0), SOLVENT_PROBE).is_err());
        assert!(build(Some(f64::NAN), SOLVENT_PROBE).is_err());
        assert!(build(Some(1.5), -1.4).is_err());
    }

    /// A dense rigid chain that wraps across the boundary must be measured at its true extent. The
    /// far beads' periodic images sit near the start of the chain, so unwrapping against a single
    /// anchor would fold them and misreport the radius; the spanning tree follows the short hops.
    #[test]
    fn bounding_radius_unwraps_a_wrapped_dense_chain() {
        use crate::cell::{Cell, Cuboid};
        // True chain along x at 0,4,8,12,16,22,26; the last two wrap into [−20, 20) of a 40 Å box.
        let wrapped = [0.0, 4.0, 8.0, 12.0, 16.0, -18.0, -14.0];
        let true_x = [0.0, 4.0, 8.0, 12.0, 16.0, 22.0, 26.0];
        let cell = Cell::Cuboid(Cuboid::new(40.0, 40.0, 40.0));
        let radii = vec![1.0; wrapped.len()];
        let atoms: Vec<usize> = (0..wrapped.len()).collect();

        let bound = bounding_radius(&cell, &atoms, &radii, |i| Point::new(wrapped[i], 0.0, 0.0));

        // Expected radius about the true centroid, plus the atom radius.
        let centroid = true_x.iter().sum::<f64>() / true_x.len() as f64;
        let expected = true_x
            .iter()
            .map(|x| (x - centroid).abs() + 1.0)
            .fold(0.0, f64::max);
        assert!(
            (bound - expected).abs() < 1e-9,
            "bound = {bound}, expected {expected}"
        );
    }

    #[test]
    fn an_empty_ligand_selection_is_refused() {
        let context = one_bead(&BULK);
        let build = |use_com: bool, radius: Option<f64>| {
            PreferentialInteractionBuilder {
                substrate: Selection::parse("molecule substrate").unwrap(),
                ligand: Selection::parse("atomtype Xe").unwrap(), // matches nothing
                use_com,
                solvent: None,
                radius,
                shell: SHELL,
                solvent_probe: SOLVENT_PROBE,
                profile: None,
                file: None,
                frequency: Frequency::Every(1),
            }
            .build(&context)
        };
        // Every path must reject an empty ligand, not just the σ-derived one.
        assert!(build(false, None).is_err());
        assert!(build(false, Some(1.5)).is_err());
        assert!(build(true, None).is_err());
    }

    #[test]
    fn an_enormous_shell_ladder_is_refused() {
        let context = one_bead(&BULK);
        let builder = PreferentialInteractionBuilder {
            substrate: Selection::parse("molecule substrate").unwrap(),
            ligand: Selection::parse("atomtype LIG").unwrap(),
            use_com: false,
            solvent: None,
            radius: Some(1.5),
            shell: Shell {
                max: 1.0e6,
                resolution: 1.0e-3,
            }, // 10^9 rungs
            solvent_probe: SOLVENT_PROBE,
            profile: None,
            file: None,
            frequency: Frequency::Every(1),
        };
        assert!(builder.build(&context).is_err());
    }

    /// voronota cannot clip the domain to a hard spherical wall, so the stored volume is the
    /// free-space union. A substrate at the centre of a wide sphere is fine; one whose domain
    /// reaches the wall stops the analysis (without aborting the run) before it biases Γ.
    #[test]
    fn a_substrate_reaching_the_spherical_wall_stops_the_analysis() {
        let stopped_at = |bead_position: f64| {
            let input = format!(
                r#"
atoms:
  - {{name: SUB, mass: 1.0, charge: 0.0, sigma: 6.0}}
  - {{name: LIG, mass: 1.0, charge: 0.0, sigma: 3.0}}
molecules:
  - name: substrate
    degrees_of_freedom: Rigid
    from_structure:
      - SUB: [0.0, 0.0, 0.0]
  - name: ligand
    atoms: [LIG]
    atomic: true
system:
  cell: !Sphere {{radius: 30.0}}
  medium: {{permittivity: !Vacuum, temperature: 300.0}}
  energy: {{}}
  blocks:
    - molecule: substrate
      N: 1
      insert: !Manual [[{bead_position}, 0.0, 0.0]]
    - molecule: ligand
      N: 1
      insert: !Manual [[0.0, 0.0, 5.0]]
propagate: {{seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}}
"#
            );
            let context = Backend::from_yaml_str(&input, None, &mut rand::thread_rng()).unwrap();
            let mut analysis = PreferentialInteractionBuilder {
                substrate: Selection::parse("molecule substrate").unwrap(),
                ligand: Selection::parse("atomtype LIG").unwrap(),
                use_com: false,
                solvent: None,
                radius: Some(1.5),
                shell: SHELL,
                solvent_probe: SOLVENT_PROBE,
                profile: None,
                file: None,
                frequency: Frequency::Every(1),
            }
            .build(&context)
            .unwrap();
            analysis.sample(&context, 0).unwrap();
            analysis.is_stopped()
        };

        // Centre: domain reaches 3 + 1.5 + 10 = 14.5 Å < 30 Å radius. Fine, keeps sampling.
        assert!(!stopped_at(0.0));
        // Near the wall: 20 + 3 + 1.5 + 10 = 34.5 Å > 30 Å. Stops at sample time.
        assert!(stopped_at(20.0));
    }

    /// b₁ is not a material constant.
    ///
    /// The solute-partitioning model assumes the hydration volume is proportional to ASA with one
    /// universal b₁ — which is what lets it transfer a partition coefficient from a model compound
    /// to a protein. But for a sphere of solvent-accessible radius `a`, the shell of thickness δ
    /// carries
    ///
    /// ```text
    /// b₁ = δ(1 + δ/a + δ²/3a²) / v̄_w
    /// ```
    ///
    /// per unit area: the curvature terms vanish only as a → ∞, with `a` the water-accessible
    /// radius `substrate_radius + solvent_probe`. A tightly curved bead must therefore report a
    /// *larger* b₁ than a blunt one at the same δ. Both are checked against the closed form, so
    /// this pins the number and not merely the ordering. The ligand radius is set away from the
    /// water probe on purpose: b₁ is a substrate–water property and must not depend on it.
    #[test]
    fn hydration_per_unit_area_grows_with_curvature() {
        // For a sphere of water-accessible radius a, the shell of thickness δ carries this many
        // waters per Å² of surface. The curvature terms vanish only as a → ∞.
        let expected = |substrate_radius: f64, delta: f64| {
            let a = substrate_radius + SOLVENT_PROBE;
            delta * (1.0 + delta / a + delta.powi(2) / (3.0 * a.powi(2))) / WATER_VOLUME
        };

        let delta = 3.0;
        let k = (delta / SHELL.resolution) as usize;

        // A ligand radius (3.0) different from the water probe (1.4): b₁ must come out water-based
        // regardless, which is the whole point of measuring the shell at the water probe.
        let b1 = |substrate_radius: f64| {
            let context = lone_bead(substrate_radius);
            let mut analysis = analysis_with_radius(&context, Some(3.0));
            analysis.sample(&context, 0).unwrap();
            analysis.hydration_density(0, k).unwrap()
        };

        let (sharp, blunt) = (3.0, 20.0);
        let (curved, flat) = (b1(sharp), b1(blunt));

        assert!(
            (curved - expected(sharp, delta)).abs() < 1e-6,
            "curved bead: b₁ = {curved}, expected {}",
            expected(sharp, delta)
        );
        assert!(
            (flat - expected(blunt, delta)).abs() < 1e-6,
            "blunt bead: b₁ = {flat}, expected {}",
            expected(blunt, delta)
        );
        assert!(
            curved > flat,
            "curvature must raise b₁: curved {curved} vs blunt {flat}"
        );
        // Both sit above the flat-plate limit δ/v̄_w, which they approach from above as a → ∞.
        assert!(flat > delta / WATER_VOLUME);
    }

    /// A molecular ligand is located by its mass centre. Its radius defaults to zero, so δ is then
    /// measured from the substrate's own surface — an anisotropic ligand has no single exclusion
    /// radius, and inventing one would be a spherical approximation dressed up as geometry.
    #[test]
    fn a_molecular_ligand_is_placed_at_its_mass_centre() {
        // Two beads at 7 and 9 Å, so the mass centre sits at 8 Å: a surface distance of 8 − 3 = 5,
        // which with a zero ligand radius first enters the domain at δ = 5.
        const DIMER: &str = r#"
atoms:
  - {name: SUB, mass: 1.0, charge: 0.0, sigma: 6.0}
  - {name: LIG, mass: 1.0, charge: 0.0, sigma: 3.0}
molecules:
  - name: substrate
    degrees_of_freedom: Rigid
    from_structure:
      - SUB: [0.0, 0.0, 0.0]
  - name: dimer
    atoms: [LIG, LIG]
system:
  cell: !Cuboid [60.0, 60.0, 60.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: substrate
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0]]
    - molecule: dimer
      N: 2
      insert: !Manual [[7.0, 0.0, 0.0], [9.0, 0.0, 0.0],
                       [25.0, 0.0, 0.0], [27.0, 0.0, 0.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;
        let context = Backend::from_yaml_str(DIMER, None, &mut rand::thread_rng()).unwrap();
        let mut analysis = PreferentialInteractionBuilder {
            substrate: Selection::parse("atomtype SUB").unwrap(),
            ligand: Selection::parse("molecule dimer").unwrap(),
            use_com: true,
            solvent: None,
            radius: None,
            shell: SHELL,
            solvent_probe: SOLVENT_PROBE,
            profile: None,
            file: None,
            frequency: Frequency::Every(1),
        }
        .build(&context)
        .unwrap();
        analysis.sample(&context, 0).unwrap();

        // One dimer is bulk, so the excluded volume is referenced against it; the other appears at
        // δ = 5. The exclusion boundary is now the bare substrate surface, not an inflated one.
        let bulk_volume = 60.0f64.powi(3) - sphere(SUBSTRATE_RADIUS + SHELL.max);
        let concentration = 1.0 / bulk_volume;
        for k in 0..SHELL.len() {
            let counted = f64::from(u8::from(k >= 5));
            let expected = counted - concentration * sphere(SUBSTRATE_RADIUS + SHELL.delta(k));
            assert!(
                (analysis.gamma(k).mean - expected).abs() < 1e-9,
                "rung {k}: Γ = {}, expected {expected}",
                analysis.gamma(k).mean
            );
        }
    }

    /// One radius sets every reference volume, so a ligand selection spanning atom kinds of
    /// different size has no single answer. Averaging them would corrupt the geometry silently;
    /// refusing is the only honest option.
    #[test]
    fn a_ligand_selection_of_mixed_size_is_refused() {
        const TWO_SIZES: &str = r#"
atoms:
  - {name: SUB, mass: 1.0, charge: 0.0, sigma: 6.0}
  - {name: SMALL, mass: 1.0, charge: 0.0, sigma: 3.0}
  - {name: BIG, mass: 1.0, charge: 0.0, sigma: 5.0}
molecules:
  - name: substrate
    degrees_of_freedom: Rigid
    from_structure:
      - SUB: [0.0, 0.0, 0.0]
  - name: small
    atoms: [SMALL]
  - name: big
    atoms: [BIG]
system:
  cell: !Cuboid [60.0, 60.0, 60.0]
  medium: {permittivity: !Vacuum, temperature: 300.0}
  energy: {}
  blocks:
    - molecule: substrate
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0]]
    - molecule: small
      N: 1
      insert: !Manual [[25.0, 0.0, 0.0]]
    - molecule: big
      N: 1
      insert: !Manual [[-25.0, 0.0, 0.0]]
propagate: {seed: !Fixed 1, criterion: Metropolis, repeat: 0, collections: []}
"#;
        let context = Backend::from_yaml_str(TWO_SIZES, None, &mut rand::thread_rng()).unwrap();
        let builder = |ligand: &str, radius: Option<f64>| PreferentialInteractionBuilder {
            substrate: Selection::parse("atomtype SUB").unwrap(),
            ligand: Selection::parse(ligand).unwrap(),
            use_com: false,
            solvent: None,
            radius,
            shell: SHELL,
            solvent_probe: SOLVENT_PROBE,
            profile: None,
            file: None,
            frequency: Frequency::Every(1),
        };

        let mixed = "atomtype SMALL or atomtype BIG";
        assert!(builder(mixed, None).build(&context).is_err());
        // …unless the ambiguity is resolved explicitly.
        assert!(builder(mixed, Some(1.5)).build(&context).is_ok());
        // A single kind needs no help.
        assert!(builder("atomtype SMALL", None).build(&context).is_ok());
    }

    /// Γ is measured against bulk, so a ladder that swallows the cell leaves nothing to measure
    /// against. Failing at build is better than reporting a Γ referenced to an empty region.
    #[test]
    fn a_domain_that_fills_the_cell_is_refused() {
        let context = one_bead(&BULK);
        let builder = PreferentialInteractionBuilder {
            substrate: Selection::parse("atomtype SUB").unwrap(),
            ligand: Selection::parse("atomtype LIG").unwrap(),
            use_com: false,
            solvent: None,
            radius: None,
            shell: Shell {
                max: 100.0,
                resolution: 1.0,
            },
            solvent_probe: SOLVENT_PROBE,
            profile: None,
            file: None,
            frequency: Frequency::Every(1),
        };
        assert!(builder.build(&context).is_err());
    }

    #[test]
    fn an_empty_substrate_selection_is_refused() {
        let context = one_bead(&BULK);
        let builder = PreferentialInteractionBuilder {
            substrate: Selection::parse("atomtype Ca").unwrap(),
            ligand: Selection::parse("atomtype LIG").unwrap(),
            use_com: false,
            solvent: None,
            radius: Some(1.5),
            shell: SHELL,
            solvent_probe: SOLVENT_PROBE,
            profile: None,
            file: None,
            frequency: Frequency::Every(1),
        };
        assert!(builder.build(&context).is_err());
    }

    #[test]
    fn deserialize_via_analysis_builder() {
        let input = r#"
!PreferentialInteraction
  substrate: "atomtype SUB"
  ligand: "atomtype LIG"
  shell: {max: 10.0, resolution: 0.5}
  file: residues.csv
  profile: gamma.csv
  frequency: !Every 100
"#;
        let builder: crate::analysis::AnalysisBuilder = serde_yml::from_str(input).unwrap();
        let context = one_bead(&BULK);
        builder.build(&context, None).unwrap();
    }

    #[test]
    fn explicit_solvent_configuration_is_accepted() {
        let input = r#"
!PreferentialInteraction
  substrate: "atomtype SUB"
  ligand: "atomtype LIG"
  solvent: {selection: "molecule water", use_com: true}
  shell: {max: 10.0, resolution: 0.5}
  frequency: !Every 100
"#;
        let builder: crate::analysis::AnalysisBuilder = serde_yml::from_str(input).unwrap();
        assert!(matches!(
            builder,
            crate::analysis::AnalysisBuilder::PreferentialInteraction(_)
        ));
    }
}

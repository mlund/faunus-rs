//! Spatial distribution function analysis on a body-fixed grid.

mod grid;
mod normalize;
mod opendx;

use self::grid::Grid;
use self::normalize::{Normalization, OutputScale};
use super::multipole_distribution::{multipole_energy_scale, pair_descriptor, GroupMoments};
use super::{Analyze, Frequency, Sampling};
use crate::auxiliary::{ColumnWriter, MappingExt};
use crate::cell::{BoundaryConditions, Shape};
use crate::group::Group;
use crate::group::{AbsIndex, GroupIndex, MoleculeId};
use crate::selection::{first_unsupported_group, Atoms, CachedSelection, Groups, Selection};
use crate::topology::io::{self, StructureData};
use crate::ObserveContext;
use crate::Point;
use anyhow::Result;
use interatomic::coulomb::Medium;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

fn default_file() -> PathBuf {
    PathBuf::from("spatial.dx")
}

const fn default_resolution() -> f64 {
    1.0
}

const fn default_padding() -> f64 {
    8.0
}

const fn default_true() -> bool {
    true
}

/// YAML builder for [`SpatialDistribution`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialDistributionBuilder {
    /// Molecular group selection defining the reference frame.
    reference: Selection,
    /// Atom selection accumulated on the grid.
    selection: Selection,
    /// Output file path.
    #[serde(default = "default_file")]
    file: PathBuf,
    /// Optional structure file (xyz) for the reference molecule, written once in
    /// the body frame so the density grid can be visualized around it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reference_file: Option<PathBuf>,
    /// Cubic grid spacing in Å.
    #[serde(default = "default_resolution")]
    resolution: f64,
    /// Padding in Å added on every side of the reference molecule's bounding box.
    #[serde(default = "default_padding")]
    padding: f64,
    /// Normalize by instantaneous bulk density to produce dimensionless SDF.
    #[serde(default = "default_true")]
    bulk_normalize: bool,
    /// Skip target atoms belonging to the current reference group.
    #[serde(default = "default_true")]
    exclude_reference: bool,
    /// Sampling frequency.
    frequency: Frequency,
}

impl SpatialDistributionBuilder {
    pub fn apply_output_dir(&mut self, dir: &std::path::Path) -> Result<()> {
        crate::analysis::prefix_in_place(&mut self.file, dir)?;
        if let Some(reference_file) = self.reference_file.as_mut() {
            crate::analysis::prefix_in_place(reference_file, dir)?;
        }
        Ok(())
    }

    pub fn build(&self, context: &impl ObserveContext) -> Result<SpatialDistribution> {
        anyhow::ensure!(
            self.resolution > 0.0,
            "SpatialDistribution: resolution must be positive"
        );
        anyhow::ensure!(
            self.padding >= 0.0,
            "SpatialDistribution: padding must be non-negative"
        );
        if self.bulk_normalize {
            anyhow::ensure!(
                context.cell().volume().is_some(),
                "SpatialDistribution: bulk normalization requires a finite cell volume"
            );
        }

        if let Some(group) =
            first_unsupported_group(context, &self.reference, |kind| !kind.atomic())?
        {
            anyhow::bail!(
                "SpatialDistribution: reference selection '{}' matched atomic group {group}",
                self.reference.source()
            );
        }
        let reference_groups: Vec<GroupIndex> = context
            .resolve_groups(&self.reference)
            .into_iter()
            .map(GroupIndex::new)
            .collect();
        anyhow::ensure!(
            !reference_groups.is_empty(),
            "SpatialDistribution: reference selection '{}' matched no active groups",
            self.reference.source()
        );
        validate_reference_groups(context, &reference_groups, self.reference.source())?;

        let reference_points = reference_body_points(context, &reference_groups)?;
        let grid = Grid::from_points(&reference_points, self.resolution, self.padding)?;
        validate_grid_extent(context, &grid)?;

        let reference_structure = self
            .reference_file
            .as_ref()
            .map(|file| {
                capture_reference_structure(context, reference_groups[0].get(), file.clone())
            })
            .transpose()?;

        Ok(SpatialDistribution {
            reference: CachedSelection::groups(self.reference.clone()),
            selection: CachedSelection::atoms(self.selection.clone()),
            file: self.file.clone(),
            reference_structure,
            grid: grid.clone(),
            counts: vec![0.0; grid.num_voxels()],
            normalization: Normalization::default(),
            scale: OutputScale::from_bulk_normalize(self.bulk_normalize),
            exclude_reference: self.exclude_reference,
            sampling: Sampling::new(self.frequency),
        })
    }
}

/// Pair-level observable available to `PairSpatialDistribution` conditions.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum PairObservable {
    Ii,
    Id,
    Dd,
    Iq,
    Mucorr,
    P2,
    Long,
    Quadcorr,
    #[serde(rename = "quadcorr_norm")]
    QuadcorrNorm,
}

impl PairObservable {
    fn value(self, descriptor: &super::multipole_distribution::PairDescriptor) -> Option<f64> {
        match self {
            Self::Ii => Some(descriptor.ii),
            Self::Id => Some(descriptor.id),
            Self::Dd => Some(descriptor.dd),
            Self::Iq => Some(descriptor.iq),
            Self::Mucorr => descriptor.mucorr,
            Self::P2 => descriptor.p2,
            Self::Long => descriptor.long,
            Self::Quadcorr => Some(descriptor.quadcorr),
            Self::QuadcorrNorm => descriptor.quadcorr_norm,
        }
    }
}

/// One inclusive interval predicate on a pair observable.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PairPredicateBuilder {
    observable: PairObservable,
    min: Option<f64>,
    max: Option<f64>,
}

impl PairPredicateBuilder {
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.min.is_some() || self.max.is_some(),
            "PairSpatialDistribution: a condition needs min or max"
        );
        if let Some(min) = self.min {
            anyhow::ensure!(
                min.is_finite(),
                "PairSpatialDistribution: condition min must be finite"
            );
        }
        if let Some(max) = self.max {
            anyhow::ensure!(
                max.is_finite(),
                "PairSpatialDistribution: condition max must be finite"
            );
        }
        if let (Some(min), Some(max)) = (self.min, self.max) {
            anyhow::ensure!(
                min <= max,
                "PairSpatialDistribution: condition min must not exceed max"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct PairCondition {
    predicates: Vec<PairPredicateBuilder>,
    energy_scale: f64,
}

impl PairCondition {
    fn build(builders: &[PairPredicateBuilder], medium: Option<&Medium>) -> Result<Option<Self>> {
        if builders.is_empty() {
            return Ok(None);
        }
        let energy_scale = medium.map(multipole_energy_scale).ok_or_else(|| {
            anyhow::anyhow!("PairSpatialDistribution: a medium is required when condition is set")
        })?;
        for predicate in builders {
            predicate.validate()?;
        }
        Ok(Some(Self {
            predicates: builders.to_vec(),
            energy_scale,
        }))
    }

    fn matches(
        &self,
        first: &GroupMoments,
        second: &GroupMoments,
        cell: &impl BoundaryConditions,
    ) -> bool {
        let descriptor = pair_descriptor(first, second, cell, self.energy_scale);
        self.matches_descriptor(&descriptor)
    }

    fn matches_descriptor(
        &self,
        descriptor: &super::multipole_distribution::PairDescriptor,
    ) -> bool {
        self.predicates.iter().all(|predicate| {
            let Some(value) = predicate.observable.value(descriptor) else {
                return false;
            };
            predicate.min.is_none_or(|min| value >= min)
                && predicate.max.is_none_or(|max| value <= max)
        })
    }
}

/// Optional two-dimensional slab through the midpoint of a molecular pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairMidplaneBuilder {
    /// Slab thickness along the pair axis, in Å.
    thickness: f64,
    /// Radius of the circular output disk, in Å.
    radius: f64,
    /// CSV output path.
    file: PathBuf,
}

/// YAML builder for pair-conditioned spatial distribution analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairSpatialDistributionBuilder {
    /// Molecular group selection used to form unique pairs.
    reference: Selection,
    /// Atom selection accumulated around the pair midpoint.
    selection: Selection,
    /// Inclusive lower and upper pair-distance bounds, in Å.
    pair_range: [f64; 2],
    /// Three-dimensional DX output path.
    file: PathBuf,
    /// Optional pair reference structure written in the pair frame.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reference_file: Option<PathBuf>,
    /// Pair-frame grid spacing, in Å.
    #[serde(default = "default_resolution")]
    resolution: f64,
    /// Padding around the pair envelope, in Å.
    #[serde(default = "default_padding")]
    padding: f64,
    /// Exclude target atoms belonging to either reference molecule.
    #[serde(default = "default_true")]
    exclude_reference: bool,
    /// Optional per-pair multipole/orientation predicates, combined with AND.
    #[serde(default)]
    condition: Vec<PairPredicateBuilder>,
    /// Optional density disk in the pair midplane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    midplane: Option<PairMidplaneBuilder>,
    /// Sampling frequency.
    frequency: Frequency,
}

impl PairSpatialDistributionBuilder {
    pub fn apply_output_dir(&mut self, dir: &std::path::Path) -> Result<()> {
        crate::analysis::prefix_in_place(&mut self.file, dir)?;
        if let Some(reference_file) = self.reference_file.as_mut() {
            crate::analysis::prefix_in_place(reference_file, dir)?;
        }
        if let Some(midplane) = self.midplane.as_mut() {
            crate::analysis::prefix_in_place(&mut midplane.file, dir)?;
        }
        Ok(())
    }

    pub fn build(
        &self,
        context: &impl ObserveContext,
        medium: Option<&Medium>,
    ) -> Result<PairSpatialDistribution> {
        anyhow::ensure!(
            self.resolution > 0.0,
            "PairSpatialDistribution: resolution must be positive"
        );
        anyhow::ensure!(
            self.padding >= 0.0,
            "PairSpatialDistribution: padding must be non-negative"
        );
        anyhow::ensure!(
            self.pair_range[0] >= 0.0 && self.pair_range[1] >= self.pair_range[0],
            "PairSpatialDistribution: pair_range must be ordered and non-negative"
        );
        let condition = PairCondition::build(&self.condition, medium)?;
        anyhow::ensure!(
            context.cell().volume().is_some(),
            "PairSpatialDistribution: bulk normalization requires cell volume"
        );
        if let Some(midplane) = &self.midplane {
            anyhow::ensure!(
                midplane.thickness > 0.0 && midplane.radius > 0.0,
                "PairSpatialDistribution: midplane thickness and radius must be positive"
            );
        }

        let reference_groups: Vec<GroupIndex> = context
            .resolve_groups(&self.reference)
            .into_iter()
            .map(GroupIndex::new)
            .collect();
        anyhow::ensure!(
            reference_groups.len() >= 2,
            "PairSpatialDistribution: reference selection '{}' matched fewer than two groups",
            self.reference.source()
        );
        validate_reference_groups(context, &reference_groups, self.reference.source())?;

        let radius = reference_groups
            .iter()
            .filter_map(|&group_index| context.group(group_index).bounding_radius())
            .fold(0.0_f64, f64::max);
        let half_extent = 0.5 * self.pair_range[1] + radius + self.padding;
        let points = [Point::repeat(-half_extent), Point::repeat(half_extent)];
        let grid = Grid::from_points(&points, self.resolution, 0.0)?;
        validate_grid_extent(context, &grid)?;

        let midplane = self.midplane.as_ref().map(|builder| {
            MidplaneGrid::new(
                builder.radius,
                self.resolution,
                builder.thickness,
                builder.file.clone(),
            )
        });

        Ok(PairSpatialDistribution {
            reference: CachedSelection::groups(self.reference.clone()),
            selection: CachedSelection::atoms(self.selection.clone()),
            pair_range: self.pair_range,
            file: self.file.clone(),
            reference_file: self.reference_file.clone(),
            reference_structure: None,
            grid: grid.clone(),
            counts: vec![0.0; grid.num_voxels()],
            normalization: Normalization::default(),
            exclude_reference: self.exclude_reference,
            condition,
            midplane,
            sampling: Sampling::new(self.frequency),
        })
    }
}

/// Spatial distribution function analysis.
#[derive(Debug)]
pub struct SpatialDistribution {
    reference: CachedSelection<Groups>,
    selection: CachedSelection<Atoms>,
    file: PathBuf,
    /// Reference molecule snapshot in the body frame, written once for visualization.
    reference_structure: Option<ReferenceStructure>,
    grid: Grid,
    counts: Vec<f64>,
    normalization: Normalization,
    scale: OutputScale,
    exclude_reference: bool,
    /// Frequency and frame count, owned by the framework.
    sampling: Sampling,
}

#[derive(Debug)]
struct MidplaneGrid {
    origin: f64,
    file: PathBuf,
    radius: f64,
    dims: usize,
    spacing: f64,
    thickness: f64,
    counts: Vec<f64>,
}

impl MidplaneGrid {
    fn new(radius: f64, spacing: f64, thickness: f64, file: PathBuf) -> Self {
        let dims = 2 * (radius / spacing).ceil() as usize;
        Self {
            origin: -(dims as f64) * spacing / 2.0,
            file,
            radius,
            dims,
            spacing,
            thickness,
            counts: vec![0.0; dims * dims],
        }
    }

    fn index_of(&self, y: f64, z: f64) -> Option<usize> {
        if y * y + z * z > self.radius.powi(2) {
            return None;
        }
        let iy = ((y - self.origin) / self.spacing).floor() as isize;
        let iz = ((z - self.origin) / self.spacing).floor() as isize;
        if iy < 0 || iz < 0 || iy >= self.dims as isize || iz >= self.dims as isize {
            None
        } else {
            Some(iy as usize + self.dims * iz as usize)
        }
    }

    /// Accumulate one target atom given in pair-frame coordinates.
    ///
    /// The slab test lives here with the disk test, so a caller cannot apply one
    /// and forget the other.
    fn accumulate(&mut self, body: &Point, weight: f64) {
        if body.x.abs() >= 0.5 * self.thickness {
            return;
        }
        if let Some(voxel) = self.index_of(body.y, body.z) {
            self.counts[voxel] += weight;
        }
    }

    fn write(&self, normalization: &Normalization) -> Result<()> {
        let values = self.normalized(normalization);
        let mut writer = ColumnWriter::open(&self.file, &["y/Å", "z/Å", "relative_density"])?;
        for iz in 0..self.dims {
            for iy in 0..self.dims {
                let index = iy + self.dims * iz;
                let y = self.origin + (iy as f64 + 0.5) * self.spacing;
                let z = self.origin + (iz as f64 + 0.5) * self.spacing;
                writer.write_row(&[
                    &format!("{y:.6}"),
                    &format!("{z:.6}"),
                    &format!("{:.8}", values[index]),
                ])?;
            }
        }
        writer.flush()?;
        Ok(())
    }

    fn normalized(&self, normalization: &Normalization) -> Vec<f64> {
        let voxel_volume = self.spacing * self.spacing * self.thickness;
        normalization.normalize_counts(&self.counts, voxel_volume, OutputScale::RelativeBulk)
    }
}

#[derive(Debug)]
pub struct PairSpatialDistribution {
    reference: CachedSelection<Groups>,
    selection: CachedSelection<Atoms>,
    pair_range: [f64; 2],
    file: PathBuf,
    reference_structure: Option<ReferenceStructure>,
    reference_file: Option<PathBuf>,
    grid: Grid,
    counts: Vec<f64>,
    /// Sum of target number density over all accepted pair observations.
    normalization: Normalization,
    exclude_reference: bool,
    condition: Option<PairCondition>,
    midplane: Option<MidplaneGrid>,
    sampling: Sampling,
}

fn validate_reference_groups(
    context: &impl ObserveContext,
    reference_groups: &[GroupIndex],
    source: &str,
) -> Result<MoleculeId> {
    let topology = context.topology_ref();
    let groups = context.groups();
    let first_molecule = groups[reference_groups[0].get()].molecule();
    for &group_index in reference_groups {
        let group = &groups[group_index.get()];
        anyhow::ensure!(
            !group.is_empty(),
            "SpatialDistribution: reference selection '{source}' matched empty group {group_index}"
        );
        let molecule_id = group.molecule();
        let molecule = topology.moleculekind(molecule_id);
        anyhow::ensure!(
            molecule_id == first_molecule,
            "SpatialDistribution: reference selection '{source}' matched multiple molecule kinds"
        );
        anyhow::ensure!(
            !molecule.atomic(),
            "SpatialDistribution: reference selection '{source}' matched atomic group {group_index}"
        );
        anyhow::ensure!(
            molecule.degrees_of_freedom().is_rigid(),
            "SpatialDistribution: reference molecule '{}' is not rigid",
            molecule.name()
        );
        anyhow::ensure!(
            group.mass_center().is_some(),
            "SpatialDistribution: reference group {group_index} has no mass center"
        );
    }
    Ok(first_molecule)
}

/// A single reference molecule frozen in the body frame.
///
/// The rigid reference geometry is constant, so the snapshot is captured once at
/// build time and written verbatim, letting the density grid be drawn around it.
#[derive(Debug)]
struct ReferenceStructure {
    file: PathBuf,
    names: Vec<String>,
    positions: Vec<Point>,
    /// Single-molecule and pair overlays are in different frames, so each names
    /// its own.
    comment: String,
}

impl ReferenceStructure {
    fn write(&self) -> Result<()> {
        let data = StructureData {
            names: self.names.clone(),
            positions: self.positions.clone(),
            comment: Some(self.comment.clone()),
            ..Default::default()
        };
        io::write_structure_frame(&self.file, &data, false)
    }
}

fn capture_reference_structure(
    context: &impl ObserveContext,
    reference_group: usize,
    file: PathBuf,
) -> Result<ReferenceStructure> {
    let topology = context.topology_ref();
    let group = &context.groups()[reference_group];
    let molecule = topology.moleculekind(group.molecule());
    let center = group.mass_center().ok_or_else(|| {
        anyhow::anyhow!("SpatialDistribution: reference group {reference_group} has no mass center")
    })?;

    let mut names = Vec::new();
    let mut positions = Vec::new();
    for atom_index in group.iter_active() {
        let relative_index = atom_index - group.start();
        let topology_index = molecule.topology_index(relative_index);
        names.push(
            molecule
                .resolved_atom_name(topology_index, topology.atomkinds())
                .to_owned(),
        );
        let displacement = context
            .cell()
            .distance(&context.position(atom_index), center);
        positions.push(crate::geometry::to_body_frame(
            &displacement,
            group.quaternion(),
        ));
    }
    Ok(ReferenceStructure {
        file,
        names,
        positions,
        comment: "Faunus SDF reference molecule (body frame)".to_owned(),
    })
}

fn reference_body_points(
    context: &impl ObserveContext,
    reference_groups: &[GroupIndex],
) -> Result<Vec<Point>> {
    let mut points = Vec::new();
    for &group_index in reference_groups {
        let group = context.group(group_index);
        let center = group.mass_center().ok_or_else(|| {
            anyhow::anyhow!("SpatialDistribution: reference group {group_index} has no mass center")
        })?;
        for atom_index in group.iter_active() {
            let displacement = context
                .cell()
                .distance(&context.position(atom_index), center);
            points.push(crate::geometry::to_body_frame(
                &displacement,
                group.quaternion(),
            ));
        }
    }
    Ok(points)
}

fn validate_grid_extent(context: &impl ObserveContext, grid: &Grid) -> Result<()> {
    if let Some(box_lengths) = context.cell().bounding_box() {
        let extent = grid.extent();
        anyhow::ensure!(
            extent.x <= box_lengths.x && extent.y <= box_lengths.y && extent.z <= box_lengths.z,
            "SpatialDistribution: grid extent ({:.3}, {:.3}, {:.3}) exceeds cell bounding box \
             ({:.3}, {:.3}, {:.3})",
            extent.x,
            extent.y,
            extent.z,
            box_lengths.x,
            box_lengths.y,
            box_lengths.z
        );
    }
    Ok(())
}

fn atom_owners(groups: &[Group], num_particles: usize) -> Vec<Option<GroupIndex>> {
    let mut owners = vec![None; num_particles];
    for group in groups {
        for atom_index in group.iter_active() {
            owners[atom_index] = Some(GroupIndex::new(group.index()));
        }
    }
    owners
}

fn eligible_target_count(
    target_atoms: &[AbsIndex],
    owners: &[Option<GroupIndex>],
    reference_group: GroupIndex,
    exclude_reference: bool,
) -> usize {
    target_atoms
        .iter()
        .filter(|&&atom| !exclude_reference || owners[atom.get()] != Some(reference_group))
        .count()
}

#[derive(Clone, Copy)]
struct PairFrame {
    origin: Point,
    ex: Point,
    ey: Point,
    ez: Point,
    separation: f64,
}

impl PairFrame {
    #[cfg(test)]
    fn new(context: &impl ObserveContext, first: GroupIndex, second: GroupIndex) -> Option<Self> {
        let first_group = context.group(first);
        let second_group = context.group(second);
        let first_com = *first_group.mass_center()?;
        let second_com = *second_group.mass_center()?;
        let delta = context.cell().distance(&second_com, &first_com);
        let separation = delta.norm();
        Self::from_delta(context, first, first_com, delta, separation)
    }

    fn from_delta(
        context: &impl ObserveContext,
        first: GroupIndex,
        first_com: Point,
        delta: Point,
        separation: f64,
    ) -> Option<Self> {
        if separation <= f64::EPSILON {
            return None;
        }
        let first_group = context.group(first);
        let ex = delta / separation;
        let body_y = first_group
            .quaternion()
            .transform_vector(&Vector3::y_axis().into_inner());
        let mut ey = body_y - ex * body_y.dot(&ex);
        if ey.norm_squared() <= 1e-14 {
            let body_z = first_group
                .quaternion()
                .transform_vector(&Vector3::z_axis().into_inner());
            ey = body_z - ex * body_z.dot(&ex);
        }
        if ey.norm_squared() <= 1e-14 {
            return None;
        }
        let ey = ey.normalize();
        let ez = ex.cross(&ey).normalize();
        Some(Self {
            origin: first_com + delta * 0.5,
            ex,
            ey,
            ez,
            separation,
        })
    }

    fn coordinates(&self, displacement: &Point) -> Point {
        Point::new(
            displacement.dot(&self.ex),
            displacement.dot(&self.ey),
            displacement.dot(&self.ez),
        )
    }

    fn relative_position(&self, context: &impl ObserveContext, position: &Point) -> Point {
        let displacement = context.cell().distance(position, &self.origin);
        self.coordinates(&displacement)
    }

    fn molecule_atom_position(
        &self,
        context: &impl ObserveContext,
        group: GroupIndex,
        atom: usize,
    ) -> Option<Point> {
        let center = *context.group(group).mass_center()?;
        let local = context.cell().distance(&context.position(atom), &center);
        let center_offset = context.cell().distance(&center, &self.origin);
        Some(self.coordinates(&(center_offset + local)))
    }
}

fn unique_reference_pairs(
    context: &impl ObserveContext,
    references: &[GroupIndex],
    pair_range: [f64; 2],
) -> Vec<(GroupIndex, GroupIndex, PairFrame)> {
    let mut pairs = Vec::new();
    for (i, &first) in references.iter().enumerate() {
        for &second in &references[i + 1..] {
            let Some(first_com) = context.group(first).mass_center() else {
                continue;
            };
            let Some(second_com) = context.group(second).mass_center() else {
                continue;
            };
            let delta = context.cell().distance(second_com, first_com);
            let separation = delta.norm();
            if !(pair_range[0]..=pair_range[1]).contains(&separation) {
                continue;
            }
            let Some(frame) = PairFrame::from_delta(context, first, *first_com, delta, separation)
            else {
                continue;
            };
            pairs.push((first, second, frame));
        }
    }
    pairs
}

fn pair_reference_structure(
    context: &impl ObserveContext,
    pair: (GroupIndex, GroupIndex, PairFrame),
    target_separation: f64,
    file: PathBuf,
) -> Result<ReferenceStructure> {
    let (first, second, frame) = pair;
    let topology = context.topology_ref();
    let mut names = Vec::new();
    let mut positions = Vec::new();
    for group_index in [first, second] {
        let group = context.group(group_index);
        let molecule = topology.moleculekind(group.molecule());
        for atom in group.iter_active() {
            let relative = frame
                .molecule_atom_position(context, group_index, atom)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "PairSpatialDistribution: reference group {group_index} has no mass center"
                    )
                })?;
            let relative_index = atom - group.start();
            let topology_index = molecule.topology_index(relative_index);
            names.push(
                molecule
                    .resolved_atom_name(topology_index, topology.atomkinds())
                    .to_owned(),
            );
            // `relative` is already in the pair frame, whose x-axis is the pair
            // axis; `frame.ex` is that axis in lab coordinates and must not be
            // mixed in here.
            let shift = if group_index == first {
                (frame.separation - target_separation) * 0.5
            } else {
                (target_separation - frame.separation) * 0.5
            };
            positions.push(relative + Point::new(shift, 0.0, 0.0));
        }
    }
    Ok(ReferenceStructure {
        file,
        names,
        positions,
        comment: "Faunus pair-conditioned SDF reference (pair frame)".to_owned(),
    })
}

impl SpatialDistribution {
    fn normalized_values(&self) -> Vec<f64> {
        self.normalization
            .normalize_counts(&self.counts, self.grid.voxel_volume(), self.scale)
    }
}

impl_info!(SpatialDistribution, "sdf", "Spatial distribution function");

impl<T: ObserveContext> Analyze<T> for SpatialDistribution {
    impl_sampling_accessors!();

    fn perform_sample(&mut self, context: &T, _step: usize, weight: f64) -> Result<()> {
        let reference_groups = self.reference.resolve(context).to_vec();
        if !reference_groups.is_empty() {
            let source = self.reference.selection().source();
            validate_reference_groups(context, &reference_groups, source)?;
        }
        let target_atoms = self.selection.resolve(context).to_vec();

        let owners = atom_owners(context.groups(), context.num_particles());
        let volume = context.cell().volume();

        for reference_group in reference_groups {
            let group = context.group(reference_group);
            let center = group.mass_center().ok_or_else(|| {
                anyhow::anyhow!(
                    "SpatialDistribution: reference group {reference_group} has no mass center"
                )
            })?;
            let eligible_targets = eligible_target_count(
                &target_atoms,
                &owners,
                reference_group,
                self.exclude_reference,
            );
            self.normalization
                .observe_reference(weight, eligible_targets, volume, self.scale)?;

            for &atom_index in &target_atoms {
                if self.exclude_reference && owners[atom_index.get()] == Some(reference_group) {
                    continue;
                }
                let displacement = context
                    .cell()
                    .distance(&context.position(atom_index.get()), center);
                let body = crate::geometry::to_body_frame(&displacement, group.quaternion());
                if let Some(voxel) = self.grid.index_of(&body) {
                    self.counts[voxel] += weight;
                }
            }
        }

        Ok(())
    }

    fn write_to_disk(&mut self) -> Result<()> {
        if self.sampling.num_samples() == 0 {
            return Ok(());
        }
        let values = self.normalized_values();
        opendx::write(&self.file, &self.grid, &values, self.scale.unit_label())?;
        if let Some(reference_structure) = &self.reference_structure {
            reference_structure.write()?;
        }
        Ok(())
    }

    fn results(&self) -> Option<yaml_serde::Value> {
        if self.sampling.num_samples() == 0 {
            return None;
        }
        let mut map = yaml_serde::Mapping::new();
        map.try_insert("num_samples", self.sampling.num_samples())?;
        map.try_insert("grid", self.grid.dims())?;
        map.try_insert("resolution/Å", self.grid.spacing())?;
        map.try_insert("bulk_normalize", self.scale == OutputScale::RelativeBulk)?;
        map.try_insert(
            "reference_observations",
            self.normalization.reference_observations(),
        )?;
        map.try_insert("file", self.file.display().to_string())?;
        if let Some(reference_structure) = &self.reference_structure {
            map.try_insert(
                "reference_file",
                reference_structure.file.display().to_string(),
            )?;
        }
        Some(yaml_serde::Value::Mapping(map))
    }
}

impl_info!(
    PairSpatialDistribution,
    "pair_sdf",
    "Pair-conditioned spatial distribution function"
);

impl<T: ObserveContext> Analyze<T> for PairSpatialDistribution {
    impl_sampling_accessors!();

    fn perform_sample(&mut self, context: &T, _step: usize, weight: f64) -> Result<()> {
        let references = self.reference.resolve(context).to_vec();
        if references.len() < 2 {
            return Ok(());
        }
        validate_reference_groups(context, &references, self.reference.selection().source())?;
        let targets = self.selection.resolve(context).to_vec();
        let owners = atom_owners(context.groups(), context.num_particles());
        let volume = context.cell().volume().ok_or_else(|| {
            anyhow::anyhow!("PairSpatialDistribution: bulk normalization requires cell volume")
        })?;
        let pairs = unique_reference_pairs(context, &references, self.pair_range);

        for (first, second, frame) in pairs {
            if let Some(condition) = &self.condition {
                let Some(first_moments) = GroupMoments::from_group(first.get(), context) else {
                    continue;
                };
                let Some(second_moments) = GroupMoments::from_group(second.get(), context) else {
                    continue;
                };
                if !condition.matches(&first_moments, &second_moments, context.cell()) {
                    continue;
                }
            }
            if self.reference_structure.is_none() {
                if let Some(file) = &self.reference_file {
                    let midpoint_separation = 0.5 * (self.pair_range[0] + self.pair_range[1]);
                    self.reference_structure = Some(
                        pair_reference_structure(
                            context,
                            (first, second, frame),
                            midpoint_separation,
                            file.clone(),
                        )
                        .map_err(|error| {
                            anyhow::anyhow!(
                                "PairSpatialDistribution: failed to capture reference structure: {error}"
                            )
                        })?,
                    );
                }
            }
            let excluded = |atom: AbsIndex| {
                self.exclude_reference
                    && (owners[atom.get()] == Some(first) || owners[atom.get()] == Some(second))
            };
            let mut eligible = 0;
            for &atom in &targets {
                if excluded(atom) {
                    continue;
                }
                eligible += 1;
                let body = frame.relative_position(context, &context.position(atom.get()));
                if let Some(voxel) = self.grid.index_of(&body) {
                    self.counts[voxel] += weight;
                }
                if let Some(midplane) = self.midplane.as_mut() {
                    midplane.accumulate(&body, weight);
                }
            }
            // `Normalization` only accumulates, so counting during the pass above
            // and reporting afterwards is equivalent to a second filter pass.
            self.normalization.observe_reference(
                weight,
                eligible,
                Some(volume),
                OutputScale::RelativeBulk,
            )?;
        }
        Ok(())
    }

    fn write_to_disk(&mut self) -> Result<()> {
        if self.sampling.num_samples() == 0 {
            return Ok(());
        }
        if self.normalization.reference_observations() == 0.0 {
            log::warn!(
                "PairSpatialDistribution: no pairs matched pair_range and condition; writing zero density grids"
            );
        }
        let values = self.normalization.normalize_counts(
            &self.counts,
            self.grid.voxel_volume(),
            OutputScale::RelativeBulk,
        );
        opendx::write(
            &self.file,
            &self.grid,
            &values,
            OutputScale::RelativeBulk.unit_label(),
        )?;

        if let Some(midplane) = &self.midplane {
            midplane.write(&self.normalization)?;
        }

        if let Some(reference_structure) = &self.reference_structure {
            reference_structure.write()?;
        } else if self.reference_file.is_some() {
            log::error!(
                "PairSpatialDistribution: reference_file was requested, but no pair matched"
            );
        }
        Ok(())
    }

    fn results(&self) -> Option<yaml_serde::Value> {
        if self.sampling.num_samples() == 0 {
            return None;
        }
        let mut map = yaml_serde::Mapping::new();
        map.try_insert("num_samples", self.sampling.num_samples())?;
        map.try_insert(
            "pair_observations",
            self.normalization.reference_observations(),
        )?;
        map.try_insert("pair_range/Å", self.pair_range)?;
        map.try_insert("conditioned", self.condition.is_some())?;
        map.try_insert("grid", self.grid.dims())?;
        map.try_insert("resolution/Å", self.grid.spacing())?;
        map.try_insert("file", self.file.display().to_string())?;
        if let Some(reference_structure) = &self.reference_structure {
            map.try_insert(
                "reference_file",
                reference_structure.file.display().to_string(),
            )?;
        }
        if let Some(midplane) = &self.midplane {
            map.try_insert("midplane_file", midplane.file.display().to_string())?;
        }
        Some(yaml_serde::Value::Mapping(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisBuilder;
    use crate::backend::Backend;
    use crate::context::Context;
    use crate::context::WithSimulationCell;
    use crate::group::{GroupCollection, GroupCollectionMut};
    use crate::UnitQuaternion;
    use crate::WithTopology;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    fn test_context() -> Backend {
        let yaml = r#"
atoms:
  - {name: R, mass: 1.0, charge: 0.0}
  - {name: Na, mass: 1.0, charge: 1.0}
  - {name: Cl, mass: 1.0, charge: -1.0}
molecules:
  - name: REF
    degrees_of_freedom: Rigid
    from_structure:
      - R: [-1.0, 0.0, 0.0]
      - R: [1.0, 0.0, 0.0]
  - name: ION
    atoms: [Na]
    atomic: true
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium:
    permittivity: !Vacuum
    temperature: 298.15
  energy:
    - !Nonbonded
        default: []
  blocks:
    - molecule: REF
      N: 1
      insert: !Manual
        - [-1.0, 0.0, 0.0]
        - [1.0, 0.0, 0.0]
    - molecule: ION
      N: 2
      insert: !Manual
        - [2.0, 0.0, 0.0]
        - [-2.0, 0.0, 0.0]
"#;
        Backend::from_yaml_str(yaml, None, &mut rand::thread_rng()).unwrap()
    }

    fn pbc_context() -> Backend {
        let yaml = r#"
atoms:
  - {name: R, mass: 1.0, charge: 0.0}
  - {name: Na, mass: 1.0, charge: 1.0}
molecules:
  - name: REF
    degrees_of_freedom: Rigid
    from_structure:
      - R: [-1.0, 0.0, 0.0]
      - R: [1.0, 0.0, 0.0]
  - name: ION
    atoms: [Na]
    atomic: true
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium:
    permittivity: !Vacuum
    temperature: 298.15
  energy:
    - !Nonbonded
        default: []
  blocks:
    - molecule: REF
      N: 1
      insert: !Manual
        - [8.0, 0.0, 0.0]
        - [-10.0, 0.0, 0.0]
    - molecule: ION
      N: 1
      insert: !Manual
        - [-9.0, 0.0, 0.0]
"#;
        Backend::from_yaml_str(yaml, None, &mut rand::thread_rng()).unwrap()
    }

    fn cppm_context() -> Backend {
        let structure =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/cppm-p18.xyz");
        let yaml = format!(
            r#"
atoms:
  - {{name: PP, mass: 1.0, charge: 1.0}}
  - {{name: NP, mass: 1.0, charge: -1.0}}
  - {{name: MP, mass: 1.0, charge: 0.0}}
  - {{name: Na, mass: 1.0, charge: 1.0}}
molecules:
  - name: CPPM
    degrees_of_freedom: Rigid
    from_structure: "{}"
  - name: ION
    atoms: [Na]
    atomic: true
system:
  cell: !Cuboid [100.0, 100.0, 100.0]
  medium:
    permittivity: !Vacuum
    temperature: 298.15
  energy: []
  blocks:
    - molecule: CPPM
      N: 2
      insert: !GridCOM
    - molecule: ION
      N: 2
      insert: !Manual [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
"#,
            structure.display()
        );
        Backend::from_yaml_str(&yaml, None, &mut rand::thread_rng()).unwrap()
    }

    #[test]
    fn builder_defaults() {
        let yaml = r#"
reference: "molecule REF"
selection: "atomtype Na"
frequency: !Every 100
"#;
        let builder: SpatialDistributionBuilder = yaml_serde::from_str(yaml).unwrap();
        assert_eq!(builder.file, PathBuf::from("spatial.dx"));
        assert_relative_eq!(builder.resolution, 1.0);
        assert_relative_eq!(builder.padding, 8.0);
        assert!(builder.bulk_normalize);
        assert!(builder.exclude_reference);
    }

    #[test]
    fn deserialize_via_analysis_builder() {
        let yaml = r#"
- !SpatialDistribution
  reference: "molecule REF"
  selection: "atomtype Na"
  frequency: !Every 10
"#;
        let builders: Vec<AnalysisBuilder> = yaml_serde::from_str(yaml).unwrap();
        assert!(matches!(
            builders[0],
            AnalysisBuilder::SpatialDistribution(_)
        ));
    }

    #[test]
    fn pair_condition_predicates_are_finite_inclusive_and_conjunctive() {
        let builders: Vec<PairPredicateBuilder> = yaml_serde::from_str(
            r#"
- observable: dd
  max: 0.0
- observable: mucorr
  min: -0.5
  max: 0.5
- observable: quadcorr_norm
  min: 0.2
  max: 1.0
"#,
        )
        .unwrap();
        let condition = PairCondition::build(&builders, Some(&Medium::neat_water(298.15)))
            .unwrap()
            .unwrap();
        let descriptor = crate::analysis::multipole_distribution::PairDescriptor {
            ii: 0.0,
            id: 0.0,
            dd: -1.0,
            iq: 0.0,
            mucorr: Some(0.0),
            p2: Some(0.0),
            long: Some(0.0),
            quadcorr: 0.0,
            quadcorr_norm: Some(0.5),
        };
        assert!(condition.matches_descriptor(&descriptor));

        let mut repulsive = descriptor;
        repulsive.dd = 1.0;
        assert!(!condition.matches_descriptor(&repulsive));

        let mut misaligned = descriptor;
        misaligned.quadcorr_norm = Some(0.1);
        assert!(!condition.matches_descriptor(&misaligned));

        let mut apolar = descriptor;
        apolar.mucorr = None;
        assert!(!condition.matches_descriptor(&apolar));
    }

    #[test]
    fn pair_condition_requires_a_medium() {
        let context = cppm_context();
        let builder: PairSpatialDistributionBuilder = yaml_serde::from_str(
            r#"
reference: "molecule CPPM"
selection: "atomtype Na"
pair_range: [40.0, 50.0]
file: pair.dx
condition:
  - observable: dd
    max: 0.0
frequency: !Every 1
"#,
        )
        .unwrap();
        assert!(builder.build(&context, None).is_err());
    }

    #[test]
    fn build_rejects_flexible_reference() {
        let yaml = r#"
atoms:
  - {name: A, mass: 1.0}
molecules:
  - name: FLEX
    atoms: [A]
system:
  cell: !Cuboid [20.0, 20.0, 20.0]
  medium: {permittivity: !Vacuum, temperature: 298.15}
  energy: [!Nonbonded {default: []}]
  blocks:
    - molecule: FLEX
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0]]
"#;
        let context = Backend::from_yaml_str(yaml, None, &mut rand::thread_rng()).unwrap();
        let builder: SpatialDistributionBuilder = yaml_serde::from_str(
            r#"
reference: "molecule FLEX"
selection: "all"
frequency: !Every 1
"#,
        )
        .unwrap();
        assert!(builder.build(&context).is_err());
    }

    #[test]
    fn periodic_minimum_image_places_target_inside_grid() {
        let context = pbc_context();
        let tmp = tempfile::tempdir().unwrap();
        let mut builder: SpatialDistributionBuilder = yaml_serde::from_str(
            r#"
reference: "molecule REF"
selection: "atomtype Na"
file: spatial.dx
resolution: 1.0
padding: 3.0
bulk_normalize: false
frequency: !Every 1
"#,
        )
        .unwrap();
        builder.apply_output_dir(tmp.path()).unwrap();
        let mut sdf = builder.build(&context).unwrap();
        Analyze::<Backend>::sample_now(&mut sdf, &context, 0, 1.0).unwrap();

        let idx_direct = sdf.grid.index_of(&Point::new(2.0, 0.0, 0.0)).unwrap();
        assert_relative_eq!(sdf.counts[idx_direct], 1.0);
        assert_relative_eq!(sdf.normalization.reference_observations(), 1.0);
    }

    #[test]
    fn restart_reconstructs_com_and_quaternion_before_sdf_sampling() {
        let mut saved_context = pbc_context();
        let orientation =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_4);

        // The reference COM is at x = 9.5 Å. Rotate the molecule and target in
        // the lab frame, then wrap all positions; the second reference atom and
        // target therefore cross x = +10 Å.
        let com = Point::new(9.5, 0.0, 0.0);
        let body_reference = [Point::new(-1.0, 0.0, 0.0), Point::new(1.0, 0.0, 0.0)];
        let body_target = Point::new(2.5, 0.5, 0.5);
        let mut positions: Vec<Point> = body_reference
            .into_iter()
            .map(|body| {
                let mut position = com + orientation.transform_vector(&body);
                saved_context.cell().boundary(&mut position);
                position
            })
            .collect();
        let mut target = com + orientation.transform_vector(&body_target);
        saved_context.cell().boundary(&mut target);
        positions.push(target);
        saved_context.set_all_positions(&positions).unwrap();
        saved_context.set_group_orientation(0, orientation);
        let state = crate::state::State::save(&saved_context, 17);

        // Normal restarts construct analyses before restoring state.yaml.
        let mut restarted_context = pbc_context();
        let tmp = tempfile::tempdir().unwrap();
        let mut builder: SpatialDistributionBuilder = yaml_serde::from_str(
            r#"
reference: "molecule REF"
selection: "atomtype Na"
file: spatial.dx
resolution: 1.0
padding: 3.0
bulk_normalize: false
frequency: !Every 1
"#,
        )
        .unwrap();
        builder.apply_output_dir(tmp.path()).unwrap();
        let mut sdf = builder.build(&restarted_context).unwrap();

        state.load(&mut restarted_context).unwrap();
        Analyze::<Backend>::sample_now(&mut sdf, &restarted_context, 0, 1.0).unwrap();

        let expected_body = Point::new(2.5, 0.5, 0.5);
        let expected_voxel = sdf.grid.index_of(&expected_body).unwrap();
        assert_relative_eq!(sdf.counts[expected_voxel], 1.0);
        assert_relative_eq!(sdf.counts.iter().sum::<f64>(), 1.0);
    }

    #[test]
    fn multiple_patchy_references_average_in_their_body_frames_after_restart() {
        let mut saved_context = cppm_context();
        let rotations = [
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.37),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -0.61),
        ];
        for (group_index, rotation) in rotations.into_iter().enumerate() {
            saved_context.rotate_group(group_index, &rotation).unwrap();
        }

        // Put one ion at the same body-frame position around each CPPM.
        let body_target = Point::new(22.0, 0.5, 0.5);
        let ion_indices: Vec<_> = saved_context.groups()[2].iter_active().collect();
        let mut positions: Vec<_> = (0..saved_context.num_particles())
            .map(|index| saved_context.position(index))
            .collect();
        for (group_index, &ion_index) in ion_indices.iter().enumerate() {
            let group = &saved_context.groups()[group_index];
            let mut target =
                *group.mass_center().unwrap() + group.quaternion().transform_vector(&body_target);
            saved_context.cell().boundary(&mut target);
            positions[ion_index] = target;
        }
        saved_context.set_all_positions(&positions).unwrap();
        let state = crate::state::State::save(&saved_context, 23);

        // Build the analysis before loading state.yaml, as in a normal restart.
        let mut restarted_context = cppm_context();
        let tmp = tempfile::tempdir().unwrap();
        let mut builder: SpatialDistributionBuilder = yaml_serde::from_str(
            r#"
reference: "molecule CPPM"
selection: "atomtype Na"
file: spatial.dx
resolution: 1.0
padding: 3.0
bulk_normalize: false
frequency: !Every 1
"#,
        )
        .unwrap();
        builder.apply_output_dir(tmp.path()).unwrap();
        let mut sdf = builder.build(&restarted_context).unwrap();

        state.load(&mut restarted_context).unwrap();
        Analyze::<Backend>::sample_now(&mut sdf, &restarted_context, 0, 1.0).unwrap();

        let expected_voxel = sdf.grid.index_of(&body_target).unwrap();
        assert_relative_eq!(sdf.counts[expected_voxel], 2.0);
        assert_relative_eq!(sdf.counts.iter().sum::<f64>(), 2.0);
    }

    #[test]
    fn pair_density_aligns_midpoint_ions_with_pair_axis() {
        let mut context = cppm_context();
        let desired_centers = [Point::new(-22.0, 0.0, 0.0), Point::new(22.0, 0.0, 0.0)];
        for (group_index, desired_center) in desired_centers.into_iter().enumerate() {
            let current_center = *context.groups()[group_index].mass_center().unwrap();
            context
                .translate_group(group_index, &(desired_center - current_center))
                .unwrap();
        }
        let first = GroupIndex::new(0);
        let second = GroupIndex::new(1);
        let frame = PairFrame::new(&context, first, second).unwrap();
        let target = frame.origin + frame.ey;
        let ion_indices: Vec<_> = context.groups()[2].iter_active().collect();
        let mut positions: Vec<_> = (0..context.num_particles())
            .map(|index| context.position(index))
            .collect();
        for &ion in &ion_indices {
            positions[ion] = target;
        }
        context.set_all_positions(&positions).unwrap();

        let tmp = tempfile::tempdir().unwrap();
        let mut builder: PairSpatialDistributionBuilder = yaml_serde::from_str(
            r#"
reference: "molecule CPPM"
selection: "atomtype Na"
pair_range: [40.0, 50.0]
file: pair.dx
reference_file: pair.xyz
resolution: 1.0
padding: 2.0
midplane:
  thickness: 2.0
  radius: 5.0
  file: pair-midplane.csv
frequency: !Every 1
"#,
        )
        .unwrap();
        builder.apply_output_dir(tmp.path()).unwrap();
        let mut analysis = builder
            .build(&context, Some(&Medium::neat_water(298.15)))
            .unwrap();
        Analyze::<Backend>::sample_now(&mut analysis, &context, 0, 1.0).unwrap();

        let expected = analysis.grid.index_of(&Point::new(0.0, 1.0, 0.0)).unwrap();
        assert_relative_eq!(analysis.counts[expected], 2.0);
        assert_relative_eq!(analysis.counts.iter().sum::<f64>(), 2.0);
        assert_relative_eq!(
            analysis.normalization.normalize_count(
                analysis.counts[expected],
                analysis.grid.voxel_volume(),
                OutputScale::RelativeBulk,
            ),
            1_000_000.0
        );
        let midplane = analysis.midplane.as_ref().unwrap();
        let expected_midplane = midplane.index_of(1.0, 0.0).unwrap();
        assert_relative_eq!(midplane.counts[expected_midplane], 2.0);

        Analyze::<Backend>::write_to_disk(&mut analysis).unwrap();
        assert!(tmp.path().join("pair.dx").is_file());
        assert!(tmp.path().join("pair.xyz").is_file());
        assert!(tmp.path().join("pair-midplane.csv").is_file());
    }

    /// A pair skewed off the lab axes: the overlay is emitted in the pair frame,
    /// so a lab-frame separation adjustment tilts it out of the density grid.
    #[test]
    fn pair_reference_structure_lies_on_the_pair_axis() {
        let mut context = cppm_context();
        let desired_centers = [Point::new(-10.0, -10.0, 0.0), Point::new(10.0, 10.0, 0.0)];
        for (group_index, desired_center) in desired_centers.into_iter().enumerate() {
            let current_center = *context.groups()[group_index].mass_center().unwrap();
            context
                .translate_group(group_index, &(desired_center - current_center))
                .unwrap();
        }

        let tmp = tempfile::tempdir().unwrap();
        let mut builder: PairSpatialDistributionBuilder = yaml_serde::from_str(
            r#"
reference: "molecule CPPM"
selection: "atomtype Na"
pair_range: [25.0, 35.0]
file: pair.dx
reference_file: pair.xyz
resolution: 1.0
padding: 2.0
frequency: !Every 1
"#,
        )
        .unwrap();
        builder.apply_output_dir(tmp.path()).unwrap();
        let mut analysis = builder
            .build(&context, Some(&Medium::neat_water(298.15)))
            .unwrap();
        Analyze::<Backend>::sample_now(&mut analysis, &context, 0, 1.0).unwrap();
        Analyze::<Backend>::write_to_disk(&mut analysis).unwrap();

        let structure = io::read_structure(&tmp.path().join("pair.xyz")).unwrap();
        let half = structure.positions.len() / 2;
        // Every atom carries unit mass here, so the centroid is the mass center.
        let centroid = |atoms: &[Point]| {
            atoms.iter().fold(Point::zeros(), |sum, p| sum + p) / atoms.len() as f64
        };
        let separation =
            centroid(&structure.positions[half..]) - centroid(&structure.positions[..half]);

        // Rescaled to the midpoint of `pair_range`, and along +x by construction.
        assert_relative_eq!(separation.x, 30.0, epsilon = 1e-9);
        assert_relative_eq!(separation.y, 0.0, epsilon = 1e-9);
        assert_relative_eq!(separation.z, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn pair_condition_can_filter_pair_density() {
        let mut context = cppm_context();
        let desired_centers = [Point::new(-22.0, 0.0, 0.0), Point::new(22.0, 0.0, 0.0)];
        for (group_index, desired_center) in desired_centers.into_iter().enumerate() {
            let current_center = *context.groups()[group_index].mass_center().unwrap();
            context
                .translate_group(group_index, &(desired_center - current_center))
                .unwrap();
        }
        let tmp = tempfile::tempdir().unwrap();
        let mut builder: PairSpatialDistributionBuilder = yaml_serde::from_str(
            r#"
reference: "molecule CPPM"
selection: "atomtype Na"
pair_range: [40.0, 45.0]
file: pair-conditioned.dx
padding: 0.0
condition:
  - observable: mucorr
    min: -1.0
    max: 1.0
frequency: !Every 1
"#,
        )
        .unwrap();
        builder.apply_output_dir(tmp.path()).unwrap();
        let mut analysis = builder
            .build(&context, Some(&Medium::neat_water(298.15)))
            .unwrap();
        Analyze::<Backend>::sample_now(&mut analysis, &context, 0, 1.0).unwrap();

        assert_relative_eq!(analysis.normalization.reference_observations(), 1.0);
        assert_relative_eq!(analysis.counts.iter().sum::<f64>(), 2.0);

        let mut rejected: PairSpatialDistributionBuilder = yaml_serde::from_str(
            r#"
reference: "molecule CPPM"
selection: "atomtype Na"
pair_range: [40.0, 45.0]
file: pair-rejected.dx
reference_file: pair-rejected.xyz
padding: 0.0
condition:
  - observable: mucorr
    min: 2.0
    max: 3.0
frequency: !Every 1
"#,
        )
        .unwrap();
        rejected.apply_output_dir(tmp.path()).unwrap();
        let mut rejected_analysis = rejected
            .build(&context, Some(&Medium::neat_water(298.15)))
            .unwrap();
        Analyze::<Backend>::sample_now(&mut rejected_analysis, &context, 0, 1.0).unwrap();
        assert_relative_eq!(
            rejected_analysis.normalization.reference_observations(),
            0.0
        );
        assert_relative_eq!(rejected_analysis.counts.iter().sum::<f64>(), 0.0);
        // An unmatched `reference_file` is reported, not fatal: analyses sharing
        // the run must still reach disk.
        Analyze::<Backend>::write_to_disk(&mut rejected_analysis).unwrap();
        assert!(tmp.path().join("pair-rejected.dx").is_file());
        assert!(!tmp.path().join("pair-rejected.xyz").is_file());
    }

    #[test]
    fn pair_density_uses_minimum_image_across_periodic_boundary() {
        let mut context = cppm_context();
        let desired_centers = [Point::new(48.0, 0.0, 0.0), Point::new(-48.0, 0.0, 0.0)];
        for (group_index, desired_center) in desired_centers.into_iter().enumerate() {
            let current_center = *context.groups()[group_index].mass_center().unwrap();
            context
                .translate_group(group_index, &(desired_center - current_center))
                .unwrap();
        }

        let first = GroupIndex::new(0);
        let second = GroupIndex::new(1);
        let frame = PairFrame::new(&context, first, second).unwrap();
        assert_relative_eq!(frame.separation, 4.0, epsilon = 1e-10);

        // The physical midpoint is outside the primary image; wrapping it gives
        // the equivalent ion position used by the simulation.
        let mut target = frame.origin;
        context.cell().boundary(&mut target);
        let ion_indices: Vec<_> = context.groups()[2].iter_active().collect();
        let mut positions: Vec<_> = (0..context.num_particles())
            .map(|index| context.position(index))
            .collect();
        for &ion in &ion_indices {
            positions[ion] = target;
        }
        context.set_all_positions(&positions).unwrap();

        let tmp = tempfile::tempdir().unwrap();
        let mut builder: PairSpatialDistributionBuilder = yaml_serde::from_str(
            r#"
reference: "molecule CPPM"
selection: "atomtype Na"
pair_range: [0.0, 10.0]
file: pair-pbc.dx
resolution: 1.0
padding: 2.0
frequency: !Every 1
"#,
        )
        .unwrap();
        builder.apply_output_dir(tmp.path()).unwrap();
        let mut analysis = builder
            .build(&context, Some(&Medium::neat_water(298.15)))
            .unwrap();
        Analyze::<Backend>::sample_now(&mut analysis, &context, 0, 1.0).unwrap();

        let expected = analysis.grid.index_of(&Point::zeros()).unwrap();
        assert_relative_eq!(analysis.counts[expected], 2.0);
        assert_relative_eq!(analysis.counts.iter().sum::<f64>(), 2.0);
    }

    #[test]
    fn reference_quaternion_sets_body_frame() {
        let mut context = test_context();
        let axis = Vector3::z_axis();
        let orientation = UnitQuaternion::from_axis_angle(&axis, std::f64::consts::FRAC_PI_2);
        context.set_group_orientation(0, orientation);

        let builder: SpatialDistributionBuilder = yaml_serde::from_str(
            r#"
reference: "molecule REF"
selection: "atomtype Na"
resolution: 1.0
padding: 3.0
bulk_normalize: false
frequency: !Every 1
"#,
        )
        .unwrap();
        let mut sdf = builder.build(&context).unwrap();
        Analyze::<Backend>::sample_now(&mut sdf, &context, 0, 1.0).unwrap();

        let body = crate::geometry::to_body_frame(
            &Point::new(2.0, 0.0, 0.0),
            context.groups()[0].quaternion(),
        );
        assert_relative_eq!(body.x, 0.0, epsilon = 1e-12);
        assert_relative_eq!(body.y, -2.0, epsilon = 1e-12);
        let idx = sdf.grid.index_of(&body).unwrap();
        assert_relative_eq!(sdf.counts[idx], 1.0);
    }

    #[test]
    fn exclude_reference_skips_overlapping_target_atoms() {
        let context = test_context();
        let builder: SpatialDistributionBuilder = yaml_serde::from_str(
            r#"
reference: "molecule REF"
selection: "all"
resolution: 1.0
padding: 3.0
bulk_normalize: false
frequency: !Every 1
"#,
        )
        .unwrap();
        let mut sdf = builder.build(&context).unwrap();
        Analyze::<Backend>::sample_now(&mut sdf, &context, 0, 1.0).unwrap();
        assert_relative_eq!(sdf.counts.iter().sum::<f64>(), 2.0);
    }

    #[test]
    fn atomtype_selection_is_resolved_after_atom_kind_changes() {
        let mut context = test_context();
        let builder: SpatialDistributionBuilder = yaml_serde::from_str(
            r#"
reference: "molecule REF"
selection: "atomtype Na"
resolution: 1.0
padding: 3.0
bulk_normalize: false
frequency: !Every 1
"#,
        )
        .unwrap();
        let mut sdf = builder.build(&context).unwrap();
        Analyze::<Backend>::sample_now(&mut sdf, &context, 0, 1.0).unwrap();

        let cl_id = context
            .topology_ref()
            .atomkinds()
            .iter()
            .position(|kind| kind.name() == "Cl")
            .unwrap();
        context.set_atom_kind(2, crate::group::AtomKindId::new(cl_id));
        Analyze::<Backend>::sample_now(&mut sdf, &context, 1, 1.0).unwrap();

        assert_relative_eq!(sdf.counts.iter().sum::<f64>(), 3.0);
    }

    #[test]
    fn unknown_field_is_rejected() {
        let yaml = r#"
reference: "molecule REF"
selection: "atomtype Na"
frequency: !Every 100
typo_field: 1.0
"#;
        assert!(yaml_serde::from_str::<SpatialDistributionBuilder>(yaml).is_err());
    }

    #[test]
    fn reference_file_writes_body_frame_structure() {
        let context = test_context();
        let tmp = tempfile::tempdir().unwrap();
        let mut builder: SpatialDistributionBuilder = yaml_serde::from_str(
            r#"
reference: "molecule REF"
selection: "atomtype Na"
reference_file: reference.xyz
resolution: 1.0
padding: 3.0
bulk_normalize: false
frequency: !Every 1
"#,
        )
        .unwrap();
        builder.apply_output_dir(tmp.path()).unwrap();
        let mut sdf = builder.build(&context).unwrap();
        Analyze::<Backend>::sample_now(&mut sdf, &context, 0, 1.0).unwrap();
        Analyze::<Backend>::write_to_disk(&mut sdf).unwrap();

        let structure = io::read_structure(&tmp.path().join("reference.xyz")).unwrap();
        assert_eq!(structure.names, ["R", "R"]);
        // REF sits at the origin with identity orientation, so body-frame
        // positions equal the molecule's internal geometry.
        assert_relative_eq!(structure.positions[0].x, -1.0);
        assert_relative_eq!(structure.positions[1].x, 1.0);
    }

    #[test]
    fn bulk_normalized_uniform_counts_are_one() {
        let mut sdf = SpatialDistribution {
            reference: CachedSelection::groups(Selection::parse("molecule REF").unwrap()),
            selection: CachedSelection::atoms(Selection::parse("atomtype Na").unwrap()),
            file: PathBuf::from("spatial.dx"),
            reference_structure: None,
            grid: Grid::from_points(&[Point::new(0.25, 0.25, 0.25)], 1.0, 0.25).unwrap(),
            counts: vec![0.0; 1],
            normalization: Normalization::default(),
            scale: OutputScale::RelativeBulk,
            exclude_reference: true,
            sampling: Sampling::new(Frequency::Every(1)),
        };
        sdf.normalization
            .observe_reference(1.0, 8, Some(8.0), OutputScale::RelativeBulk)
            .unwrap();
        sdf.counts[0] = 1.0;
        assert_relative_eq!(sdf.normalized_values()[0], 1.0);
    }
}

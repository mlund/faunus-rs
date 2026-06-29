//! Spatial distribution function analysis on a body-fixed grid.

mod frame;
mod grid;
mod normalize;
mod opendx;

use self::grid::Grid;
use self::normalize::{Normalization, OutputScale};
use super::{Analyze, Frequency};
use crate::auxiliary::MappingExt;
use crate::cell::{BoundaryConditions, Shape};
use crate::group::Group;
use crate::selection::Selection;
use crate::{Context, Point};
use anyhow::Result;
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
pub struct SpatialDistributionBuilder {
    /// Molecular group selection defining the reference frame.
    reference: Selection,
    /// Atom selection accumulated on the grid.
    selection: Selection,
    /// Output file path.
    #[serde(default = "default_file")]
    file: PathBuf,
    /// Cubic grid spacing in Å.
    #[serde(default = "default_resolution")]
    resolution: f64,
    /// Extra grid extent around the reference molecule in Å.
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
        crate::analysis::prefix_in_place(&mut self.file, dir)
    }

    pub fn build(&self, context: &impl Context) -> Result<SpatialDistribution> {
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

        let reference_groups = context.resolve_groups_live(&self.reference);
        anyhow::ensure!(
            !reference_groups.is_empty(),
            "SpatialDistribution: reference selection '{}' matched no active groups",
            self.reference.source()
        );
        validate_reference_groups(context, &reference_groups, self.reference.source())?;

        let reference_points = reference_body_points(context, &reference_groups)?;
        let grid = Grid::from_points(&reference_points, self.resolution, self.padding)?;
        validate_grid_extent(context, &grid)?;

        Ok(SpatialDistribution {
            reference: self.reference.clone(),
            selection: self.selection.clone(),
            file: self.file.clone(),
            grid: grid.clone(),
            counts: vec![0.0; grid.num_voxels()],
            normalization: Normalization::default(),
            num_samples: 0,
            scale: OutputScale::from_bulk_normalize(self.bulk_normalize),
            exclude_reference: self.exclude_reference,
            frequency: self.frequency,
        })
    }
}

/// Spatial distribution function analysis.
#[derive(Debug)]
pub struct SpatialDistribution {
    reference: Selection,
    selection: Selection,
    file: PathBuf,
    grid: Grid,
    counts: Vec<f64>,
    normalization: Normalization,
    num_samples: usize,
    scale: OutputScale,
    exclude_reference: bool,
    frequency: Frequency,
}

fn validate_reference_groups(
    context: &impl Context,
    reference_groups: &[usize],
    source: &str,
) -> Result<usize> {
    let topology = context.topology_ref();
    let groups = context.groups();
    let first_molecule = groups[reference_groups[0]].molecule();
    for &group_index in reference_groups {
        let group = &groups[group_index];
        anyhow::ensure!(
            !group.is_empty(),
            "SpatialDistribution: reference selection '{source}' matched empty group {group_index}"
        );
        let molecule_id = group.molecule();
        let molecule = &topology.moleculekinds()[molecule_id];
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

fn reference_body_points(context: &impl Context, reference_groups: &[usize]) -> Result<Vec<Point>> {
    let mut points = Vec::new();
    for &group_index in reference_groups {
        let group = &context.groups()[group_index];
        let center = group.mass_center().ok_or_else(|| {
            anyhow::anyhow!("SpatialDistribution: reference group {group_index} has no mass center")
        })?;
        for atom_index in group.iter_active() {
            let displacement = context
                .cell()
                .distance(&context.position(atom_index), center);
            points.push(frame::to_body_frame(&displacement, group.quaternion()));
        }
    }
    Ok(points)
}

fn validate_grid_extent(context: &impl Context, grid: &Grid) -> Result<()> {
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

fn atom_owners(groups: &[Group], num_particles: usize) -> Vec<Option<usize>> {
    let mut owners = vec![None; num_particles];
    for group in groups {
        for atom_index in group.iter_active() {
            owners[atom_index] = Some(group.index());
        }
    }
    owners
}

fn eligible_target_count(
    target_atoms: &[usize],
    owners: &[Option<usize>],
    reference_group: usize,
    exclude_reference: bool,
) -> usize {
    target_atoms
        .iter()
        .filter(|&&atom_index| !exclude_reference || owners[atom_index] != Some(reference_group))
        .count()
}

impl SpatialDistribution {
    fn normalized_values(&self) -> Vec<f64> {
        let voxel_volume = self.grid.voxel_volume();
        self.counts
            .iter()
            .map(|&count| {
                self.normalization
                    .normalize_count(count, voxel_volume, self.scale)
            })
            .collect()
    }
}

impl crate::Info for SpatialDistribution {
    fn short_name(&self) -> Option<&'static str> {
        Some("sdf")
    }

    fn long_name(&self) -> Option<&'static str> {
        Some("Spatial distribution function")
    }
}

impl<T: Context> Analyze<T> for SpatialDistribution {
    fn frequency(&self) -> Frequency {
        self.frequency
    }

    fn set_frequency(&mut self, freq: Frequency) {
        self.frequency = freq;
    }

    fn perform_sample(&mut self, context: &T, _step: usize, weight: f64) -> Result<()> {
        let reference_groups = context.resolve_groups_live(&self.reference);
        if !reference_groups.is_empty() {
            validate_reference_groups(context, &reference_groups, self.reference.source())?;
        }
        let target_atoms = context.resolve_atoms_live(&self.selection);

        let owners = atom_owners(context.groups(), context.num_particles());
        let volume = context.cell().volume();

        for reference_group in reference_groups {
            let group = &context.groups()[reference_group];
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
                if self.exclude_reference && owners[atom_index] == Some(reference_group) {
                    continue;
                }
                let displacement = context
                    .cell()
                    .distance(&context.position(atom_index), center);
                let body = frame::to_body_frame(&displacement, group.quaternion());
                if let Some(voxel) = self.grid.index_of(&body) {
                    self.counts[voxel] += weight;
                }
            }
        }

        self.num_samples += 1;
        Ok(())
    }

    fn num_samples(&self) -> usize {
        self.num_samples
    }

    fn write_to_disk(&mut self) -> Result<()> {
        if self.num_samples == 0 {
            return Ok(());
        }
        let values = self.normalized_values();
        opendx::write(&self.file, &self.grid, &values, self.scale.unit_label())
    }

    fn to_yaml(&self) -> Option<serde_yml::Value> {
        if self.num_samples == 0 {
            return None;
        }
        let mut map = serde_yml::Mapping::new();
        map.try_insert("num_samples", self.num_samples)?;
        map.try_insert("grid", self.grid.dims())?;
        map.try_insert("resolution", self.grid.spacing())?;
        map.try_insert("bulk_normalize", self.scale == OutputScale::RelativeBulk)?;
        map.try_insert(
            "reference_observations",
            self.normalization.reference_observations(),
        )?;
        map.try_insert("file", self.file.display().to_string())?;
        Some(serde_yml::Value::Mapping(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::AnalysisBuilder;
    use crate::backend::Backend;
    use crate::group::GroupCollection;
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
    nonbonded:
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
    nonbonded:
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

    #[test]
    fn builder_defaults() {
        let yaml = r#"
reference: "molecule REF"
selection: "atomtype Na"
frequency: !Every 100
"#;
        let builder: SpatialDistributionBuilder = serde_yml::from_str(yaml).unwrap();
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
        let builders: Vec<AnalysisBuilder> = serde_yml::from_str(yaml).unwrap();
        assert!(matches!(
            builders[0],
            AnalysisBuilder::SpatialDistribution(_)
        ));
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
  energy: {nonbonded: {default: []}}
  blocks:
    - molecule: FLEX
      N: 1
      insert: !Manual [[0.0, 0.0, 0.0]]
"#;
        let context = Backend::from_yaml_str(yaml, None, &mut rand::thread_rng()).unwrap();
        let builder: SpatialDistributionBuilder = serde_yml::from_str(
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
        let mut builder: SpatialDistributionBuilder = serde_yml::from_str(
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
        sdf.perform_sample(&context, 0, 1.0).unwrap();

        let idx_direct = sdf.grid.index_of(&Point::new(2.0, 0.0, 0.0)).unwrap();
        assert_relative_eq!(sdf.counts[idx_direct], 1.0);
        assert_relative_eq!(sdf.normalization.reference_observations(), 1.0);
    }

    #[test]
    fn reference_quaternion_sets_body_frame() {
        let mut context = test_context();
        let axis = Vector3::z_axis();
        let orientation = UnitQuaternion::from_axis_angle(&axis, std::f64::consts::FRAC_PI_2);
        context.groups_mut()[0].set_quaternion(orientation);

        let builder: SpatialDistributionBuilder = serde_yml::from_str(
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
        sdf.perform_sample(&context, 0, 1.0).unwrap();

        let body =
            frame::to_body_frame(&Point::new(2.0, 0.0, 0.0), context.groups()[0].quaternion());
        assert_relative_eq!(body.x, 0.0, epsilon = 1e-12);
        assert_relative_eq!(body.y, -2.0, epsilon = 1e-12);
        let idx = sdf.grid.index_of(&body).unwrap();
        assert_relative_eq!(sdf.counts[idx], 1.0);
    }

    #[test]
    fn exclude_reference_skips_overlapping_target_atoms() {
        let context = test_context();
        let builder: SpatialDistributionBuilder = serde_yml::from_str(
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
        sdf.perform_sample(&context, 0, 1.0).unwrap();
        assert_relative_eq!(sdf.counts.iter().sum::<f64>(), 2.0);
    }

    #[test]
    fn atomtype_selection_is_resolved_after_atom_kind_changes() {
        let mut context = test_context();
        let builder: SpatialDistributionBuilder = serde_yml::from_str(
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
        sdf.perform_sample(&context, 0, 1.0).unwrap();

        let cl_id = context
            .topology_ref()
            .atomkinds()
            .iter()
            .position(|kind| kind.name() == "Cl")
            .unwrap();
        context.set_atom_kind(2, cl_id);
        sdf.perform_sample(&context, 1, 1.0).unwrap();

        assert_relative_eq!(sdf.counts.iter().sum::<f64>(), 3.0);
    }

    #[test]
    fn bulk_normalized_uniform_counts_are_one() {
        let mut sdf = SpatialDistribution {
            reference: Selection::parse("molecule REF").unwrap(),
            selection: Selection::parse("atomtype Na").unwrap(),
            file: PathBuf::from("spatial.dx"),
            grid: Grid::from_points(&[Point::new(0.25, 0.25, 0.25)], 1.0, 0.25).unwrap(),
            counts: vec![0.0; 1],
            normalization: Normalization::default(),
            num_samples: 1,
            scale: OutputScale::RelativeBulk,
            exclude_reference: true,
            frequency: Frequency::Every(1),
        };
        sdf.normalization
            .observe_reference(1.0, 8, Some(8.0), OutputScale::RelativeBulk)
            .unwrap();
        sdf.counts[0] = 1.0;
        assert_relative_eq!(sdf.normalized_values()[0], 1.0);
    }
}

// Copyright 2023 Mikael Lund
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

//! # Simulation cells with or without periodic boundary conditions
//!
//! This module contains the interface for the simulation cell, which describes the geometry of the simulation system.
//! The simulation cell is a geometric [`Shape`], e.g. a cube, sphere, etc., with defined [`BoundaryConditions`].
//! Some statistical thermodynamic ensembles require volume fluctuations, which is implemented by scaling the simulation cell
//! through the [`VolumeScale`] trait.
//!
//! ## Available cell types
//!
//! | Cell | PBC | Description |
//! |------|-----|-------------|
//! | [`Cuboid`] | XYZ | Orthorhombic box |
//! | [`HexagonalPrism`] | XYZ | Hexagonal cross-section with non-orthorhombic lattice |
//! | [`Slit`] | XY | Cuboidal box with hard walls in Z |
//! | [`Cylinder`] | Z | Cylindrical cell with hard walls in XY |
//! | [`Sphere`] | None | Spherical cell with hard walls |
//! | [`Endless`] | None | Infinite, open cell |

mod cuboid;
mod cylinder;
mod endless;
mod hexagonal_prism;
//pub(crate) mod lumol;
mod pbc_params;
mod slit;
mod sphere;

use std::path::Path;

use crate::Point;
pub use cuboid::Cuboid;
pub use cylinder::Cylinder;
pub use endless::Endless;
pub use hexagonal_prism::HexagonalPrism;
pub(crate) use pbc_params::PbcParams;
use rand::Rng;
use serde::{Deserialize, Serialize};
pub use slit::Slit;
pub use sphere::Sphere;

/// Final interface for a unit cell used to describe the geometry of a simulation system.
///
/// It is a combination of a [`Shape`], [`BoundaryConditions`] and [`VolumeScale`].
pub trait SimulationCell: Shape + BoundaryConditions + VolumeScale + std::fmt::Debug {}

/// Orthorhombic supercell expansion for I/O formats that require cuboid boxes.
pub struct OrthorhombicExpansion {
    /// Supercell dimensions
    pub box_lengths: Point,
    /// Translation vectors for replicating particles to fill the supercell.
    /// Each vector produces one additional copy of all particles.
    pub translations: Vec<Point>,
}

/// Geometric shape like a sphere, cube, etc.
pub trait Shape {
    /// Get volume
    fn volume(&self) -> Option<f64>;
    /// Position of the geometric center of the shape.
    ///
    /// For a cube, this is the center of the box;
    /// for a sphere, the center of the sphere etc.
    fn center(&self) -> Point {
        Point::zeros()
    }
    /// Determines if a point lies inside the boundaries of the shape
    fn is_inside(&self, point: &Point) -> bool;
    /// Determines if a point lies outside the boundaries of the shape
    #[inline(always)]
    fn is_outside(&self, point: &Point) -> bool {
        !self.is_inside(point)
    }
    /// Bounding box of the shape centered at `center()`
    fn bounding_box(&self) -> Option<Point>;
    /// Generate a uniformly random point inside the shape.
    ///
    /// Rejection sampling within the axis-aligned bounding box: draw uniformly in
    /// the box and keep the first point satisfying [`is_inside`](Shape::is_inside).
    /// This is uniform on the shape *iff* [`bounding_box`](Shape::bounding_box)
    /// encloses it — a too-small box would make part of the shape unreachable and
    /// bias the sample. Panics for cells without a bounding box (e.g. `Endless`).
    fn get_point_inside<R: Rng + ?Sized>(&self, rng: &mut R) -> Point {
        let half = self
            .bounding_box()
            .expect("finite cell required for point insertion")
            * 0.5;
        loop {
            let point = Point::new(
                rng.gen_range(-half.x..half.x),
                rng.gen_range(-half.y..half.y),
                rng.gen_range(-half.z..half.z),
            );
            if self.is_inside(&point) {
                return point;
            }
        }
    }
    /// Orthorhombic supercell expansion needed for I/O of non-orthorhombic cells.
    ///
    /// Returns `None` for cells whose bounding box is already orthorhombic.
    fn orthorhombic_expansion(&self) -> Option<OrthorhombicExpansion> {
        None
    }
}

/// Periodic boundary conditions in various directions
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PeriodicDirections {
    /// Periodic boundary conditions in Z direction
    PeriodicZ,
    /// 2d periodic boundary conditions in the XY plane, e.g. a slab
    PeriodicXY,
    /// 3d periodic boundary conditions in XYZ directions
    PeriodicXYZ,
    /// No periodic boundaries in any direction
    None,
}

impl PeriodicDirections {
    /// True if periodic in some direction
    pub fn is_some(&self) -> bool {
        *self != Self::None
    }
}

/// Interface for periodic boundary conditions and minimum image convention
pub trait BoundaryConditions {
    /// Report on periodic boundary conditions
    fn pbc(&self) -> PeriodicDirections;
    /// Wrap a point to fit within boundaries, if appropriate
    fn boundary(&self, point: &mut Point);
    /// Minimum image distance between two points inside a cell
    fn distance(&self, point1: &Point, point2: &Point) -> Point;
    /// Get the minimum squared distance between two points
    #[inline(always)]
    fn distance_squared(&self, point1: &Point, point2: &Point) -> f64 {
        self.distance(point1, point2).norm_squared()
    }
}

/// Policies for how to scale a volume
///
/// This is used to scale an old volume to a new volume.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum VolumeScalePolicy {
    /// Isotropic scaling (equal scaling in all directions)
    #[default]
    Isotropic,
    /// Isochoric scaling of z and the xy-plane (constant volume)
    IsochoricZ,
    /// Scale along z-axis only
    ScaleZ,
    /// Scale the XY plane
    ScaleXY,
}

/// Trait for scaling a position or the simulation cell according to a scaling policy.
pub trait VolumeScale {
    /// Scale a `position` inside a simulation cell according to a scaling policy.
    ///
    /// Errors if the scaling policy is unsupported.
    fn scale_position(
        &self,
        new_volume: f64,
        position: &mut Point,
        policy: VolumeScalePolicy,
    ) -> anyhow::Result<()>;

    /// Scale cell volume to a new volume according to a scaling policy.
    ///
    /// This should typically be followed by a call to `scale_position` for each particle or mass center.
    /// Errors if the scaling policy is unsupported.
    fn scale_volume(&mut self, new_volume: f64, policy: VolumeScalePolicy) -> anyhow::Result<()>;
}

/// Simulation cell enum used for reading information about cell from the input file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Cell {
    Cuboid(Cuboid),
    Cylinder(Cylinder),
    Endless(Endless),
    HexagonalPrism(HexagonalPrism),
    Slit(Slit),
    Sphere(Sphere),
}

impl Cell {
    /// Get simulation cell from a Faunus configuration file.
    pub(crate) fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let yaml = crate::auxiliary::read_yaml(&path)
            .map_err(|err| anyhow::anyhow!("Error reading file {:?}: {}", path.as_ref(), err))?;
        Self::from_str(&yaml)
    }

    /// Get simulation cell from a YAML string (the `system/cell` section).
    pub(crate) fn from_str(yaml: &str) -> anyhow::Result<Self> {
        let full: serde_yml::Value = serde_yml::from_str(yaml)?;

        let system = full
            .get("system")
            .ok_or_else(|| anyhow::Error::msg("Could not find `system` in the YAML file."))?;

        let Some(value) = system.get("cell") else {
            log::warn!("No cell defined for the system. Using Endless cell.");
            return Ok(Self::Endless(Endless));
        };
        let cell = crate::auxiliary::from_section_value("system/cell", value)?;
        Ok(cell)
    }

    /// The same cell, resized to the orthorhombic box recorded in a trajectory frame.
    ///
    /// The shape comes from the input file, so only its size has to be recovered: the box inverts
    /// [`Shape::bounding_box`]. `None` when there is nothing to do — the box is unchanged, the cell
    /// has none to invert, or the frame carries none.
    pub(crate) fn resized_to_box(&self, lengths: Point) -> anyhow::Result<Option<Self>> {
        // A cell with no box, or a frame with none — a trajectory of an endless cell records zeros.
        // Keeping the input's cell is what a rerun did before frames carried a box at all; adopting
        // a zero one would make every inverse box length infinite and every distance NaN.
        let Some(current) = self.bounding_box() else {
            return Ok(None);
        };
        if lengths.iter().any(|l| !l.is_finite() || *l <= 0.0) {
            return Ok(None);
        }
        if close(current, lengths) {
            return Ok(None);
        }
        if let Some(reason) = self.rerun_rejection() {
            anyhow::bail!(reason);
        }
        let mismatch = |expected: &str| {
            anyhow::anyhow!(
                "the trajectory's box changed ({current:?} → {lengths:?}), which is not {expected}"
            )
        };
        let square_xy = close(lengths, Point::new(lengths.y, lengths.x, lengths.z));

        match self {
            Self::Cuboid(_) => Ok(Some(Self::Cuboid(Cuboid::new(
                lengths.x, lengths.y, lengths.z,
            )))),
            Self::Slit(_) => Ok(Some(Self::Slit(Slit::new(lengths.x, lengths.y, lengths.z)))),
            Self::Sphere(_) => {
                if !square_xy || !close(lengths, Point::from_element(lengths.x)) {
                    return Err(mismatch("a cube, as a sphere's box must be"));
                }
                Ok(Some(Self::Sphere(Sphere::new(0.5 * lengths.x))))
            }
            Self::Cylinder(_) => {
                if !square_xy {
                    return Err(mismatch("square in xy, as a cylinder's box must be"));
                }
                Ok(Some(Self::Cylinder(Cylinder::new(
                    0.5 * lengths.x,
                    lengths.z,
                ))))
            }
            // Rejected above by `rerun_rejection`; `Endless` by the missing bounding box.
            Self::HexagonalPrism(_) | Self::Endless(_) => unreachable!(),
        }
    }

    /// Why a trajectory written in this cell cannot be replayed, if it cannot.
    ///
    /// A hexagonal prism is written as its orthorhombic supercell — twice the atoms — so a rerun
    /// would otherwise fail on atom count, sending the user after a topology error that is not there.
    pub(crate) const fn rerun_rejection(&self) -> Option<&'static str> {
        match self {
            Self::HexagonalPrism(_) => Some(
                "a hexagonal prism is written to a trajectory as an expanded orthorhombic \
                 supercell, so its trajectories cannot be rerun",
            ),
            _ => None,
        }
    }
}

/// Box lengths agree to the single precision a trajectory frame stores them in (~1e-7).
fn close(a: Point, b: Point) -> bool {
    const TOLERANCE: f64 = 1e-5;
    a.iter()
        .zip(b.iter())
        .all(|(a, b)| (a - b).abs() <= TOLERANCE * a.abs().max(b.abs()).max(1.0))
}

#[cfg(test)]
mod resized_to_box {
    use super::*;

    #[test]
    fn a_cuboid_takes_the_frames_box() {
        let cell = Cell::Cuboid(Cuboid::cubic(30.0));
        let resized = cell
            .resized_to_box(Point::new(10.0, 20.0, 40.0))
            .unwrap()
            .expect("a cuboid is determined by its box");
        assert_eq!(
            resized.bounding_box().unwrap(),
            Point::new(10.0, 20.0, 40.0)
        );
    }

    /// An unchanged box must be a no-op, or every frame of a constant-volume rerun would pay a cell
    /// list rebuild and swap the declared box for its single-precision round-trip.
    #[test]
    fn an_unchanged_box_leaves_the_cell_alone() {
        for cell in [
            Cell::Cuboid(Cuboid::cubic(30.0)),
            Cell::Sphere(Sphere::new(15.0)),
        ] {
            let unchanged = cell.bounding_box().unwrap();
            assert!(cell.resized_to_box(unchanged).unwrap().is_none());
        }
    }

    #[test]
    fn a_sphere_and_a_cylinder_follow_their_box() {
        let sphere = Cell::Sphere(Sphere::new(15.0))
            .resized_to_box(Point::new(20.0, 20.0, 20.0))
            .unwrap()
            .expect("a sphere is 2r on every side");
        assert_eq!(
            sphere.volume().unwrap(),
            Sphere::new(10.0).volume().unwrap()
        );

        let cylinder = Cell::Cylinder(Cylinder::new(5.0, 10.0))
            .resized_to_box(Point::new(20.0, 20.0, 30.0))
            .unwrap()
            .expect("a cylinder is (2r, 2r, h)");
        assert_eq!(
            cylinder.bounding_box().unwrap(),
            Point::new(20.0, 20.0, 30.0)
        );
    }

    /// A trajectory of an endless cell records a zero box. Adopting it would make every inverse box
    /// length infinite and every distance NaN, so the input's cell stands — as it did before frames
    /// carried a box at all.
    #[test]
    fn a_frame_without_a_box_leaves_the_cell_alone() {
        let cell = Cell::Cuboid(Cuboid::cubic(30.0));
        assert!(cell.resized_to_box(Point::zeros()).unwrap().is_none());
        assert!(cell
            .resized_to_box(Point::new(f64::NAN, 30.0, 30.0))
            .unwrap()
            .is_none());
    }

    /// An endless cell has no box, so a frame's box says nothing about it.
    #[test]
    fn an_endless_cell_is_left_alone() {
        let cell = Cell::Endless(Endless);
        assert!(cell.resized_to_box(Point::zeros()).unwrap().is_none());
        assert!(cell
            .resized_to_box(Point::new(10.0, 10.0, 10.0))
            .unwrap()
            .is_none());
    }

    #[test]
    fn a_hexagonal_prism_says_why_it_cannot_be_rerun() {
        let cell = Cell::HexagonalPrism(HexagonalPrism::new(10.0, 20.0));
        let err = cell
            .resized_to_box(Point::new(30.0, 30.0, 30.0))
            .unwrap_err();
        assert!(err.to_string().contains("supercell"), "{err}");
    }

    #[test]
    fn a_sphere_whose_box_is_not_a_cube_is_an_error() {
        let cell = Cell::Sphere(Sphere::new(15.0));
        let err = cell
            .resized_to_box(Point::new(20.0, 25.0, 30.0))
            .unwrap_err();
        assert!(err.to_string().contains("must be"), "{err}");
    }
}

impl TryFrom<Cell> for Cuboid {
    type Error = anyhow::Error;
    fn try_from(cell: Cell) -> Result<Self, Self::Error> {
        match cell {
            Cell::Cuboid(c) => Ok(c),
            _ => Err(anyhow::Error::msg("Cell is not a cuboid")),
        }
    }
}

impl TryFrom<Cell> for Cylinder {
    type Error = anyhow::Error;
    fn try_from(cell: Cell) -> Result<Self, Self::Error> {
        match cell {
            Cell::Cylinder(c) => Ok(c),
            _ => Err(anyhow::Error::msg("Cell is not a cylinder")),
        }
    }
}

impl TryFrom<Cell> for Sphere {
    type Error = anyhow::Error;
    fn try_from(cell: Cell) -> Result<Self, Self::Error> {
        match cell {
            Cell::Sphere(c) => Ok(c),
            _ => Err(anyhow::Error::msg("Cell is not a sphere")),
        }
    }
}

impl TryFrom<Cell> for Slit {
    type Error = anyhow::Error;
    fn try_from(cell: Cell) -> Result<Self, Self::Error> {
        match cell {
            Cell::Slit(c) => Ok(c),
            _ => Err(anyhow::Error::msg("Cell is not a slit")),
        }
    }
}

impl TryFrom<Cell> for Endless {
    type Error = anyhow::Error;
    fn try_from(cell: Cell) -> Result<Self, Self::Error> {
        match cell {
            Cell::Endless(c) => Ok(c),
            _ => Err(anyhow::Error::msg("Cell is not endless")),
        }
    }
}

impl TryFrom<Cell> for HexagonalPrism {
    type Error = anyhow::Error;
    fn try_from(cell: Cell) -> Result<Self, Self::Error> {
        match cell {
            Cell::HexagonalPrism(c) => Ok(c),
            _ => Err(anyhow::Error::msg("Cell is not a hexagonal prism")),
        }
    }
}

impl Shape for Cell {
    #[inline]
    fn volume(&self) -> Option<f64> {
        match self {
            Self::Cuboid(x) => x.volume(),
            Self::Cylinder(x) => x.volume(),
            Self::Endless(_) => None,
            Self::HexagonalPrism(x) => x.volume(),
            Self::Slit(x) => x.volume(),
            Self::Sphere(x) => x.volume(),
        }
    }

    #[inline]
    fn is_inside(&self, point: &Point) -> bool {
        match self {
            Self::Cuboid(x) => x.is_inside(point),
            Self::Cylinder(x) => x.is_inside(point),
            Self::Endless(_) => true,
            Self::HexagonalPrism(x) => x.is_inside(point),
            Self::Slit(x) => x.is_inside(point),
            Self::Sphere(x) => x.is_inside(point),
        }
    }

    #[inline]
    fn bounding_box(&self) -> Option<Point> {
        match self {
            Self::Cuboid(s) => s.bounding_box(),
            Self::Cylinder(s) => s.bounding_box(),
            Self::Endless(s) => s.bounding_box(),
            Self::HexagonalPrism(s) => s.bounding_box(),
            Self::Slit(s) => s.bounding_box(),
            Self::Sphere(s) => s.bounding_box(),
        }
    }

    fn orthorhombic_expansion(&self) -> Option<OrthorhombicExpansion> {
        match self {
            Self::HexagonalPrism(s) => s.orthorhombic_expansion(),
            _ => None,
        }
    }
}

impl VolumeScale for Cell {
    #[inline]
    fn scale_volume(&mut self, new_volume: f64, policy: VolumeScalePolicy) -> anyhow::Result<()> {
        match self {
            Self::Cuboid(x) => x.scale_volume(new_volume, policy),
            Self::Cylinder(x) => x.scale_volume(new_volume, policy),
            Self::Endless(x) => x.scale_volume(new_volume, policy),
            Self::HexagonalPrism(x) => x.scale_volume(new_volume, policy),
            Self::Slit(x) => x.scale_volume(new_volume, policy),
            Self::Sphere(x) => x.scale_volume(new_volume, policy),
        }
    }

    #[inline]
    fn scale_position(
        &self,
        new_volume: f64,
        point: &mut Point,
        policy: VolumeScalePolicy,
    ) -> anyhow::Result<()> {
        match self {
            Self::Cuboid(x) => x.scale_position(new_volume, point, policy),
            Self::Cylinder(x) => x.scale_position(new_volume, point, policy),
            Self::Endless(x) => x.scale_position(new_volume, point, policy),
            Self::HexagonalPrism(x) => x.scale_position(new_volume, point, policy),
            Self::Slit(x) => x.scale_position(new_volume, point, policy),
            Self::Sphere(x) => x.scale_position(new_volume, point, policy),
        }
    }
}

impl BoundaryConditions for Cell {
    #[inline]
    fn pbc(&self) -> PeriodicDirections {
        match self {
            Self::Cuboid(x) => x.pbc(),
            Self::Cylinder(x) => x.pbc(),
            Self::Endless(x) => x.pbc(),
            Self::HexagonalPrism(x) => x.pbc(),
            Self::Slit(x) => x.pbc(),
            Self::Sphere(x) => x.pbc(),
        }
    }

    // Force-inline so the compiler can hoist the variant check out of
    // tight per-particle loops and devirtualize the inner call.
    #[inline(always)]
    fn boundary(&self, point: &mut Point) {
        match self {
            Self::Cuboid(x) => x.boundary(point),
            Self::Cylinder(x) => x.boundary(point),
            Self::Endless(x) => x.boundary(point),
            Self::HexagonalPrism(x) => x.boundary(point),
            Self::Slit(x) => x.boundary(point),
            Self::Sphere(x) => x.boundary(point),
        }
    }

    // Force-inline so the compiler can hoist the variant check out of
    // nonbonded pair loops and devirtualize the inner distance call.
    #[inline(always)]
    fn distance(&self, point1: &Point, point2: &Point) -> Point {
        match self {
            Self::Cuboid(x) => x.distance(point1, point2),
            Self::Cylinder(x) => x.distance(point1, point2),
            Self::Endless(x) => x.distance(point1, point2),
            Self::HexagonalPrism(x) => x.distance(point1, point2),
            Self::Slit(x) => x.distance(point1, point2),
            Self::Sphere(x) => x.distance(point1, point2),
        }
    }
}

impl SimulationCell for Cell {}

#[cfg(test)]
mod tests {
    use super::Cell;
    use crate::{
        cell::{Cuboid, Endless, Shape, Sphere},
        Point,
    };

    #[test]
    fn test_read_from_file() {
        // cuboid
        let cell: Cuboid = Cell::from_file("tests/files/topology_pass.yaml")
            .unwrap()
            .try_into()
            .unwrap();
        let point1 = Point::new(-4.9, 2.4, 5.71);
        let point2 = Point::new(-5.1, 3.2, 4.6);
        assert!(cell.is_inside(&point1));
        assert!(!cell.is_inside(&point2));

        // sphere
        let cell: Sphere = Cell::from_file("tests/files/cell_sphere.yaml")
            .unwrap()
            .try_into()
            .unwrap();
        let point1 = Point::new(8.9, 5.2, 9.3);
        let point2 = Point::new(8.9, 7.2, 9.3);
        assert!(cell.is_inside(&point1));
        assert!(!cell.is_inside(&point2));

        // endless
        let cell: Endless = Cell::from_file("tests/files/cell_endless.yaml")
            .unwrap()
            .try_into()
            .unwrap();
        let point1 = Point::new(-203847.21, 947382.143, 2973212.14);
        assert!(cell.is_inside(&point1));

        // default. Note that we can use Cell directly for all shapes.
        let cell: Cell = Cell::from_file("tests/files/cell_none.yaml").unwrap();
        let point1 = Point::new(-203847.21, 947382.143, 2973212.14);
        assert!(cell.is_inside(&point1));
        assert!(TryInto::<Endless>::try_into(cell).is_ok());
    }

    /// `get_point_inside` samples uniformly in `bounding_box()/2` and rejects
    /// points failing `is_inside`, which is uniform on the shape *only if* the
    /// bounding box encloses it. A box too small on any axis would leave part of
    /// the shape unreachable and silently bias insertion — undetectable by a
    /// membership test, but a correctness bug for MC insertion moves. Here we
    /// probe a deliberately enlarged box and require every interior point to fall
    /// within the sampled half-extents. Tests our geometry, not the RNG, with a
    /// seeded generator so it is deterministic and cheap enough to run in CI.
    #[test]
    fn bounding_box_encloses_shape() {
        use crate::cell::{Cylinder, HexagonalPrism};
        use rand::{rngs::StdRng, Rng, SeedableRng};

        fn check(shape: &impl Shape) {
            let half = shape.bounding_box().expect("finite bounding box") * 0.5;
            let probe = half * 1.2; // deliberately larger than the sampling box
            let eps = 1e-9;
            let mut rng = StdRng::seed_from_u64(0x5EED);
            let mut hits = 0usize;
            for _ in 0..5_000 {
                let p = Point::new(
                    rng.gen_range(-probe.x..probe.x),
                    rng.gen_range(-probe.y..probe.y),
                    rng.gen_range(-probe.z..probe.z),
                );
                if shape.is_inside(&p) {
                    hits += 1;
                    assert!(
                        p.x.abs() <= half.x + eps
                            && p.y.abs() <= half.y + eps
                            && p.z.abs() <= half.z + eps,
                        "inside point {p:?} lies outside sampling half-extents {half:?}",
                    );
                }
            }
            assert!(hits > 0, "probe box never reached the shape interior");
        }

        check(&Cuboid::new(4.0, 6.0, 8.0));
        check(&Sphere::new(3.0));
        check(&Cylinder::new(3.0, 7.0));
        check(&HexagonalPrism::new(2.5, 5.0));
    }
}

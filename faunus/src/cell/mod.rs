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
mod slit;
mod sphere;

use std::path::Path;

use crate::Point;
pub use cuboid::Cuboid;
pub use cylinder::Cylinder;
use dyn_clone::DynClone;
pub use endless::Endless;
pub use hexagonal_prism::HexagonalPrism;
use rand::rngs::ThreadRng;
use serde::{Deserialize, Serialize};
pub use slit::Slit;
pub use sphere::Sphere;

/// Final interface for a unit cell used to describe the geometry of a simulation system.
///
/// It is a combination of a [`Shape`], [`BoundaryConditions`], [`VolumeScale`], [`CellToChemCell`].
/// Only used when `chemfiles` feature is active.
#[cfg(feature = "chemfiles")]
pub trait SimulationCell:
    Shape
    + BoundaryConditions
    + VolumeScale
    + DynClone
    + std::fmt::Debug
    + crate::topology::chemfiles_interface::CellToChemCell
{
}

/// Final interface for a unit cell used to describe the geometry of a simulation system.
///
/// It is a combination of a [`Shape`], [`BoundaryConditions`] and [`VolumeScale`].
#[cfg(not(feature = "chemfiles"))]
pub trait SimulationCell:
    Shape + BoundaryConditions + VolumeScale + DynClone + std::fmt::Debug
{
}

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
    /// Generate a random point positioned inside the boundaries of the shape
    fn get_point_inside(&self, rng: &mut ThreadRng) -> Point;
    /// Orthorhombic supercell expansion needed for I/O of non-orthorhombic cells.
    ///
    /// Returns `None` for cells whose bounding box is already orthorhombic.
    fn orthorhombic_expansion(&self) -> Option<OrthorhombicExpansion> {
        None
    }
}

dyn_clone::clone_trait_object!(SimulationCell);

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
        let yaml = std::fs::read_to_string(&path)
            .map_err(|err| anyhow::anyhow!("Error reading file {:?}: {}", path.as_ref(), err))?;
        let full: serde_yaml::Value = serde_yaml::from_str(&yaml)?;

        let system = full
            .get("system")
            .ok_or_else(|| anyhow::Error::msg("Could not find `system` in the YAML file."))?;

        let Some(value) = system.get("cell") else {
            log::warn!("No cell defined for the system. Using Endless cell.");
            return Ok(Self::Endless(Endless));
        };
        let cell = serde_yaml::from_value(value.clone()).map_err(anyhow::Error::msg)?;
        Ok(cell)
    }
}

impl From<Cell> for Box<dyn SimulationCell> {
    fn from(cell: Cell) -> Self {
        match cell {
            Cell::Cuboid(c) => Box::new(c),
            Cell::Cylinder(c) => Box::new(c),
            Cell::Endless(c) => Box::new(c),
            Cell::HexagonalPrism(c) => Box::new(c),
            Cell::Slit(c) => Box::new(c),
            Cell::Sphere(c) => Box::new(c),
        }
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
    fn get_point_inside(&self, rng: &mut ThreadRng) -> Point {
        match self {
            Self::Cuboid(s) => s.get_point_inside(rng),
            Self::Cylinder(s) => s.get_point_inside(rng),
            Self::Endless(s) => s.get_point_inside(rng),
            Self::HexagonalPrism(s) => s.get_point_inside(rng),
            Self::Slit(s) => s.get_point_inside(rng),
            Self::Sphere(s) => s.get_point_inside(rng),
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

    #[inline]
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

    #[inline]
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
}

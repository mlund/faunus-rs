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

//! Uniform slab grid along the z-axis of a prismatic cell.
//!
//! Profiles along z — density, charge, potential — all share the same layout: divide the box
//! into slabs of equal thickness, assign each position to a slab, and normalise by the slab
//! volume. That layout lives here so the profile analyses agree on where the bin edges fall and
//! on what "the cross-sectional area" means for each cell shape.

use crate::cell::{Cell, Shape};
use anyhow::Result;

/// Relative tolerance for detecting cell-volume drift: a committed volume move exceeds it, while a
/// virtual/rejected move restores the exact cell and stays within it.
const VOLUME_TOLERANCE: f64 = 1e-6;

/// A uniform grid of slabs spanning the cell along z, from `−half_length_z` to `+half_length_z`.
///
/// The grid is laid out once from the cell and does not follow later volume changes.
#[derive(Clone, Debug)]
pub(crate) struct ZGrid {
    half_length_z: f64,
    bin_width: f64,
    n_bins: usize,
    area: f64,
}

impl ZGrid {
    /// Lay out a grid of slabs approximately `resolution` thick across `cell`.
    ///
    /// The cell must be prismatic, i.e. have the same cross-section at every height, since a
    /// slab volume is otherwise ill-defined. Spheres and endless cells are rejected.
    pub(crate) fn from_cell(cell: &Cell, resolution: f64) -> Result<Self> {
        if !resolution.is_finite() || resolution <= 0.0 {
            anyhow::bail!("resolution must be a positive, finite number; got {resolution}");
        }
        match cell {
            Cell::Cuboid(_) | Cell::Slit(_) | Cell::Cylinder(_) | Cell::HexagonalPrism(_) => {}
            other => anyhow::bail!(
                "a z-profile requires a prismatic cell (cuboid, slit, cylinder, or \
                 hexagonal prism); got {other:?}"
            ),
        }
        let (Some(bbox), Some(volume)) = (cell.bounding_box(), cell.volume()) else {
            anyhow::bail!("a finite cell is required");
        };
        let length_z = bbox.z;
        // Cell side lengths are not validated on input, so a nonsensical box would otherwise
        // divide by zero here and silently poison every bin with NaN.
        if length_z <= 0.0 || volume <= 0.0 || !length_z.is_finite() || !volume.is_finite() {
            anyhow::bail!(
                "a cell with positive volume and z-length is required; \
                 got volume = {volume}, Lz = {length_z}"
            );
        }
        // The cross-section is the same at every height, so the volume fixes it exactly. Taking
        // it from the bounding box instead would overestimate a cylinder or a hexagonal prism.
        let area = volume / length_z;
        // Pick the bin count tiling the cell closest to `resolution`, but never zero bins.
        let n_bins = ((length_z / resolution).round() as usize).max(1);
        Ok(Self {
            half_length_z: 0.5 * length_z,
            bin_width: length_z / n_bins as f64,
            n_bins,
            area,
        })
    }

    pub(crate) fn n_bins(&self) -> usize {
        self.n_bins
    }

    pub(crate) fn bin_width(&self) -> f64 {
        self.bin_width
    }

    /// Cross-sectional area perpendicular to z (Å²).
    pub(crate) fn area(&self) -> f64 {
        self.area
    }

    /// Volume of a single slab (Å³).
    pub(crate) fn bin_volume(&self) -> f64 {
        self.area * self.bin_width
    }

    /// Volume of the cell the grid was laid out from (Å³).
    pub(crate) fn volume(&self) -> f64 {
        self.bin_volume() * self.n_bins as f64
    }

    /// The cell's current volume, but only if it has drifted from the grid's build-time volume
    /// beyond a small relative tolerance; otherwise `None`.
    ///
    /// The slabs keep their initial size, so any drift makes a z-profile meaningful only at
    /// constant volume. Consumers decide whether to warn or to error. Since every volume move in
    /// practice changes the total volume, this single scalar catches them all.
    pub(crate) fn volume_drift(&self, cell: &Cell) -> Option<f64> {
        let current = cell.volume()?;
        let initial = self.volume();
        ((current - initial).abs() > VOLUME_TOLERANCE * initial).then_some(current)
    }

    /// z at the centre of bin `index`.
    pub(crate) fn bin_center(&self, index: usize) -> f64 {
        -self.half_length_z + (index as f64 + 0.5) * self.bin_width
    }

    /// Bin holding axial position `z`, clamped to the grid so boundary positions fold in.
    pub(crate) fn bin_index(&self, z: f64) -> usize {
        let raw = ((z + self.half_length_z) / self.bin_width).floor();
        raw.clamp(0.0, (self.n_bins - 1) as f64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::{Cuboid, Cylinder, Endless, HexagonalPrism, Slit, Sphere};
    use std::f64::consts::PI;

    #[test]
    fn cuboid_area_is_the_base_and_needs_no_square_base() {
        // A rectangular base is fine for a density profile, unlike the slab electrostatics.
        let grid = ZGrid::from_cell(&Cell::Cuboid(Cuboid::new(10.0, 20.0, 30.0)), 1.0).unwrap();
        assert!((grid.area() - 200.0).abs() < 1e-9);
        assert_eq!(grid.n_bins(), 30);
        assert!((grid.bin_volume() - 200.0).abs() < 1e-9);
    }

    #[test]
    fn cylinder_area_is_the_disk_not_the_bounding_square() {
        let radius = 5.0;
        let grid = ZGrid::from_cell(&Cell::Cylinder(Cylinder::new(radius, 20.0)), 2.0).unwrap();
        assert!((grid.area() - PI * radius * radius).abs() < 1e-9);
        assert_eq!(grid.n_bins(), 10);
    }

    #[test]
    fn hexagonal_prism_area_is_the_hexagon_not_the_bounding_rectangle() {
        let (side, height) = (5.0, 12.0);
        let grid = ZGrid::from_cell(
            &Cell::HexagonalPrism(HexagonalPrism::new(side, height)),
            1.0,
        )
        .unwrap();
        let hexagon = 1.5 * 3.0_f64.sqrt() * side * side;
        assert!((grid.area() - hexagon).abs() < 1e-9);
        // The bounding rectangle √3a · 2a is strictly larger.
        assert!(hexagon < 3.0_f64.sqrt() * side * 2.0 * side);
    }

    #[test]
    fn slit_is_accepted() {
        assert!(ZGrid::from_cell(&Cell::Slit(Slit::new(10.0, 10.0, 10.0)), 1.0).is_ok());
    }

    #[test]
    fn non_prismatic_and_endless_cells_are_rejected() {
        assert!(ZGrid::from_cell(&Cell::Sphere(Sphere::new(10.0)), 1.0).is_err());
        assert!(ZGrid::from_cell(&Cell::Endless(Endless), 1.0).is_err());
    }

    /// Cell side lengths are not validated on input, so a degenerate box reaches this far.
    #[test]
    fn degenerate_cells_are_rejected() {
        for (a, b, c) in [(10.0, 10.0, 0.0), (10.0, 10.0, -5.0), (0.0, 10.0, 10.0)] {
            let cell = Cell::Cuboid(Cuboid::new(a, b, c));
            assert!(
                ZGrid::from_cell(&cell, 1.0).is_err(),
                "accepted a cuboid of {a}x{b}x{c}"
            );
        }
    }

    #[test]
    fn non_positive_resolution_is_rejected() {
        let cell = Cell::Cuboid(Cuboid::new(10.0, 10.0, 10.0));
        assert!(ZGrid::from_cell(&cell, 0.0).is_err());
        assert!(ZGrid::from_cell(&cell, -1.0).is_err());
    }

    #[test]
    fn bins_tile_the_cell_and_boundaries_fold_in() {
        let grid = ZGrid::from_cell(&Cell::Cuboid(Cuboid::new(4.0, 4.0, 10.0)), 1.0).unwrap();
        assert_eq!(grid.n_bins(), 10);
        assert!((grid.bin_center(0) - -4.5).abs() < 1e-9);
        assert!((grid.bin_center(9) - 4.5).abs() < 1e-9);
        assert_eq!(grid.bin_index(-5.0), 0);
        assert_eq!(grid.bin_index(0.0), 5);
        // Positions on or beyond either wall clamp into the end bins.
        assert_eq!(grid.bin_index(5.0), 9);
        assert_eq!(grid.bin_index(-99.0), 0);
        assert_eq!(grid.bin_index(99.0), 9);
    }

    #[test]
    fn a_resolution_coarser_than_the_cell_still_yields_one_bin() {
        let grid = ZGrid::from_cell(&Cell::Cuboid(Cuboid::new(4.0, 4.0, 3.0)), 100.0).unwrap();
        assert_eq!(grid.n_bins(), 1);
        assert!((grid.bin_center(0) - 0.0).abs() < 1e-9);
    }
}

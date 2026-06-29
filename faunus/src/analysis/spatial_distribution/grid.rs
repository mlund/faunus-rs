use crate::Point;
use anyhow::Result;

/// Regular Cartesian grid in the reference body frame.
#[derive(Clone, Debug)]
pub(super) struct Grid {
    origin: Point,
    dims: [usize; 3],
    spacing: f64,
}

impl Grid {
    pub(super) fn from_points(points: &[Point], spacing: f64, padding: f64) -> Result<Self> {
        anyhow::ensure!(
            spacing > 0.0,
            "SpatialDistribution: resolution must be positive"
        );
        anyhow::ensure!(
            padding >= 0.0,
            "SpatialDistribution: padding must be non-negative"
        );
        anyhow::ensure!(
            !points.is_empty(),
            "SpatialDistribution: reference selection has no active atoms"
        );

        let mut lower = points[0];
        let mut upper = points[0];
        for point in &points[1..] {
            lower.x = lower.x.min(point.x);
            lower.y = lower.y.min(point.y);
            lower.z = lower.z.min(point.z);
            upper.x = upper.x.max(point.x);
            upper.y = upper.y.max(point.y);
            upper.z = upper.z.max(point.z);
        }

        lower -= Point::repeat(padding);
        upper += Point::repeat(padding);

        let origin = Point::new(
            (lower.x / spacing).floor() * spacing,
            (lower.y / spacing).floor() * spacing,
            (lower.z / spacing).floor() * spacing,
        );
        let high = Point::new(
            (upper.x / spacing).ceil() * spacing,
            (upper.y / spacing).ceil() * spacing,
            (upper.z / spacing).ceil() * spacing,
        );

        let dims = [
            ((high.x - origin.x) / spacing).ceil() as usize,
            ((high.y - origin.y) / spacing).ceil() as usize,
            ((high.z - origin.z) / spacing).ceil() as usize,
        ];
        anyhow::ensure!(
            dims.iter().all(|&n| n > 0),
            "SpatialDistribution: grid has zero extent"
        );

        Ok(Self {
            origin,
            dims,
            spacing,
        })
    }

    pub(super) const fn origin(&self) -> Point {
        self.origin
    }

    pub(super) const fn dims(&self) -> [usize; 3] {
        self.dims
    }

    pub(super) const fn spacing(&self) -> f64 {
        self.spacing
    }

    pub(super) fn voxel_volume(&self) -> f64 {
        self.spacing.powi(3)
    }

    pub(super) fn num_voxels(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    pub(super) fn extent(&self) -> Point {
        Point::new(
            self.dims[0] as f64 * self.spacing,
            self.dims[1] as f64 * self.spacing,
            self.dims[2] as f64 * self.spacing,
        )
    }

    /// Return the linear x-fastest voxel index for a body-frame point.
    pub(super) fn index_of(&self, point: &Point) -> Option<usize> {
        let rel = (point - self.origin) / self.spacing;
        if rel.x < 0.0 || rel.y < 0.0 || rel.z < 0.0 {
            return None;
        }
        let ix = rel.x.floor() as usize;
        let iy = rel.y.floor() as usize;
        let iz = rel.z.floor() as usize;
        if ix < self.dims[0] && iy < self.dims[1] && iz < self.dims[2] {
            Some(self.linear_index(ix, iy, iz))
        } else {
            None
        }
    }

    pub(super) const fn linear_index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix + self.dims[0] * (iy + self.dims[1] * iz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bounds_are_rounded_and_padded() {
        let points = [Point::new(-0.2, 0.1, 1.2), Point::new(1.1, 2.4, 3.0)];
        let grid = Grid::from_points(&points, 1.0, 1.0).unwrap();
        assert_relative_eq!(grid.origin().x, -2.0);
        assert_relative_eq!(grid.origin().y, -1.0);
        assert_relative_eq!(grid.origin().z, 0.0);
        assert_eq!(grid.dims(), [5, 5, 4]);
        assert_relative_eq!(grid.voxel_volume(), 1.0);
    }

    #[test]
    fn indexing_is_x_fastest_and_half_open() {
        let grid = Grid::from_points(&[Point::zeros()], 1.0, 1.0).unwrap();
        assert_eq!(grid.dims(), [2, 2, 2]);
        assert_eq!(grid.index_of(&Point::new(-1.0, -1.0, -1.0)), Some(0));
        assert_eq!(grid.index_of(&Point::new(0.1, -1.0, -1.0)), Some(1));
        assert_eq!(grid.index_of(&Point::new(-1.0, 0.1, -1.0)), Some(2));
        assert_eq!(grid.index_of(&Point::new(-1.0, -1.0, 0.1)), Some(4));
        assert_eq!(grid.index_of(&Point::new(1.0, 0.0, 0.0)), None);
    }
}

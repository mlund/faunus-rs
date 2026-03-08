/// Precomputed parameters for branchless minimum image distance.
///
/// Unifies all orthorhombic cell types (Cuboid, Slit, Cylinder, Sphere, Endless)
/// into a single arithmetic path: `dx -= box * round(dx * inv_box)`.
/// Non-periodic directions use `(f64::MAX, 0.0)` so `round(dx * 0.0) = 0`
/// and the correction vanishes without branching.
#[derive(Clone, Copy, Debug)]
pub struct PbcParams {
    box_len: [f64; 3],
    inv_box_len: [f64; 3],
}

impl PbcParams {
    /// Build from a simulation cell. Returns `None` for HexagonalPrism
    /// which needs non-orthorhombic Wigner-Seitz nearest image reduction.
    pub(crate) fn try_from_cell(cell: &impl super::SimulationCell) -> Option<Self> {
        if cell.orthorhombic_expansion().is_some() {
            return None;
        }
        let pbc = cell.pbc();
        let periodic_xy = matches!(
            pbc,
            super::PeriodicDirections::PeriodicXYZ | super::PeriodicDirections::PeriodicXY
        );
        let periodic_z = matches!(
            pbc,
            super::PeriodicDirections::PeriodicXYZ | super::PeriodicDirections::PeriodicZ
        );

        let bb = cell.bounding_box();
        let make = |periodic: bool, len: f64| -> (f64, f64) {
            if periodic {
                (len, 1.0 / len)
            } else {
                (f64::MAX, 0.0)
            }
        };
        let (bx, by, bz) = bb.map_or((f64::MAX, f64::MAX, f64::MAX), |b| (b.x, b.y, b.z));
        let (lx, ix) = make(periodic_xy, bx);
        let (ly, iy) = make(periodic_xy, by);
        let (lz, iz) = make(periodic_z, bz);

        Some(Self {
            box_len: [lx, ly, lz],
            inv_box_len: [ix, iy, iz],
        })
    }

    /// Branchless minimum image displacement vector from point i to point j.
    #[inline(always)]
    pub(crate) fn distance_vector(
        &self,
        xi: f64,
        yi: f64,
        zi: f64,
        xj: f64,
        yj: f64,
        zj: f64,
    ) -> [f64; 3] {
        let mut dx = xj - xi;
        dx -= self.box_len[0] * (dx * self.inv_box_len[0]).round();
        let mut dy = yj - yi;
        dy -= self.box_len[1] * (dy * self.inv_box_len[1]).round();
        let mut dz = zj - zi;
        dz -= self.box_len[2] * (dz * self.inv_box_len[2]).round();
        [dx, dy, dz]
    }

    /// Branchless minimum image distance squared between two points.
    #[inline(always)]
    pub(crate) fn distance_squared(
        &self,
        xi: f64,
        yi: f64,
        zi: f64,
        xj: f64,
        yj: f64,
        zj: f64,
    ) -> f64 {
        let [dx, dy, dz] = self.distance_vector(xi, yi, zi, xj, yj, zj);
        dx.mul_add(dx, dy.mul_add(dy, dz * dz))
    }
}

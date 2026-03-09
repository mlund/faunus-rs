//! Cell list for spatial acceleration of short-range pair interactions.
//!
//! Partitions orthorhombic simulation boxes into a regular grid of cells
//! whose side length is at least `cutoff`. Neighbor queries for a particle
//! then only visit the 27 (3³) adjacent cells, reducing the inner loop
//! from O(N) to O(neighbors).

use crate::Point;

/// Flat index from 3D cell coordinates.
#[inline]
fn flat_index(ix: usize, iy: usize, iz: usize, ny: usize, nz: usize) -> usize {
    ix * ny * nz + iy * nz + iz
}

/// Spatial cell list for orthorhombic periodic boxes.
#[derive(Clone, Debug)]
pub struct CellList {
    /// Cell dimensions (≥ cutoff in each direction)
    cell_size: [f64; 3],
    /// Number of cells along each axis
    num_cells: [usize; 3],
    /// Box origin (minimum corner, typically -L/2 for centered boxes)
    origin: [f64; 3],
    /// cells[flat_idx] = list of particle indices in that cell
    cells: Vec<Vec<usize>>,
    /// Reverse map: particle index → flat cell index (for O(1) lookup/removal)
    particle_cell: Vec<usize>,
    /// Precomputed neighbor cell offsets per cell (including self), with PBC wrapping
    neighbor_offsets: Vec<Vec<usize>>,
    /// Interaction cutoff
    cutoff: f64,
}

impl CellList {
    /// Create a new cell list for a box with the given side lengths and cutoff.
    ///
    /// Each cell dimension is at least `cutoff`; the actual number of cells
    /// is `floor(box_len / cutoff)`, clamped to `[1, …]`.
    pub fn new(box_len: [f64; 3], cutoff: f64) -> Self {
        let num_cells = [
            (box_len[0] / cutoff).floor().max(1.0) as usize,
            (box_len[1] / cutoff).floor().max(1.0) as usize,
            (box_len[2] / cutoff).floor().max(1.0) as usize,
        ];
        let cell_size = [
            box_len[0] / num_cells[0] as f64,
            box_len[1] / num_cells[1] as f64,
            box_len[2] / num_cells[2] as f64,
        ];
        let origin = [-box_len[0] / 2.0, -box_len[1] / 2.0, -box_len[2] / 2.0];

        let total_cells = num_cells[0] * num_cells[1] * num_cells[2];
        let cells = vec![Vec::new(); total_cells];

        let neighbor_offsets = Self::build_neighbor_map(num_cells);

        CellList {
            cell_size,
            num_cells,
            origin,
            cells,
            particle_cell: Vec::new(),
            neighbor_offsets,
            cutoff,
        }
    }

    /// Interaction cutoff used to size the cells.
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Build from a set of positions. `num_particles` is the total capacity
    /// (including inactive particles); only indices yielded by `active_indices`
    /// are actually placed into cells.
    pub fn build(
        &mut self,
        positions: impl Fn(usize) -> Point,
        num_particles: usize,
        active_indices: impl Iterator<Item = usize>,
    ) {
        for cell in &mut self.cells {
            cell.clear();
        }
        self.particle_cell.clear();
        self.particle_cell.resize(num_particles, usize::MAX);

        for i in active_indices {
            let pos = positions(i);
            let ci = self.cell_index_for(&pos);
            self.cells[ci].push(i);
            self.particle_cell[i] = ci;
        }
    }

    /// Iterator over particle indices in neighbor cells of `particle`.
    /// Includes the particle's own cell (caller must skip self if needed).
    pub fn neighbors(&self, particle: usize) -> impl Iterator<Item = usize> + '_ {
        let ci = self.particle_cell[particle];
        self.neighbor_offsets[ci]
            .iter()
            .flat_map(move |&neighbor_cell| self.cells[neighbor_cell].iter().copied())
    }

    /// Update a particle's cell assignment after it moved.
    /// Silently skips particles not yet in the cell list (sentinel value).
    pub fn update_particle(&mut self, particle: usize, new_pos: &Point) {
        let old_cell = *self.particle_cell.get(particle).unwrap_or(&usize::MAX);
        if old_cell == usize::MAX {
            return;
        }
        let new_cell = self.cell_index_for(new_pos);
        if old_cell != new_cell {
            self.remove_from_cell(particle, old_cell);
            self.cells[new_cell].push(particle);
            self.particle_cell[particle] = new_cell;
        }
    }

    /// Add a newly activated particle to the cell list.
    pub fn add_particle(&mut self, particle: usize, pos: &Point) {
        if particle >= self.particle_cell.len() {
            self.particle_cell.resize(particle + 1, usize::MAX);
        }
        let ci = self.cell_index_for(pos);
        self.cells[ci].push(particle);
        self.particle_cell[particle] = ci;
    }

    /// Remove a deactivated particle from the cell list.
    pub fn remove_particle(&mut self, particle: usize) {
        let ci = self.particle_cell[particle];
        if ci != usize::MAX {
            self.remove_from_cell(particle, ci);
            self.particle_cell[particle] = usize::MAX;
        }
    }

    /// Full rebuild (e.g. after volume change). Recreates cell grid.
    pub fn rebuild(
        &mut self,
        box_len: [f64; 3],
        positions: impl Fn(usize) -> Point,
        num_particles: usize,
        active_indices: impl Iterator<Item = usize>,
    ) {
        *self = Self::new(box_len, self.cutoff);
        self.build(positions, num_particles, active_indices);
    }

    /// Compute the flat cell index for a position.
    #[inline]
    fn cell_index_for(&self, pos: &Point) -> usize {
        let ix = ((pos.x - self.origin[0]) / self.cell_size[0]).floor() as isize;
        let iy = ((pos.y - self.origin[1]) / self.cell_size[1]).floor() as isize;
        let iz = ((pos.z - self.origin[2]) / self.cell_size[2]).floor() as isize;
        // Wrap with PBC
        let nx = self.num_cells[0] as isize;
        let ny = self.num_cells[1] as isize;
        let nz = self.num_cells[2] as isize;
        let ix = ix.rem_euclid(nx) as usize;
        let iy = iy.rem_euclid(ny) as usize;
        let iz = iz.rem_euclid(nz) as usize;
        flat_index(ix, iy, iz, self.num_cells[1], self.num_cells[2])
    }

    /// Remove particle from a specific cell via swap_remove.
    fn remove_from_cell(&mut self, particle: usize, cell_idx: usize) {
        let cell = &mut self.cells[cell_idx];
        if let Some(pos) = cell.iter().position(|&p| p == particle) {
            cell.swap_remove(pos);
        }
    }

    /// Precompute the 27 neighbor cell indices (including self) for each cell,
    /// wrapping with PBC.
    fn build_neighbor_map(num_cells: [usize; 3]) -> Vec<Vec<usize>> {
        let [nx, ny, nz] = num_cells;
        let total = nx * ny * nz;
        let mut map = Vec::with_capacity(total);

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let mut neighbors = Vec::with_capacity(27);
                    for dx in [-1_isize, 0, 1] {
                        for dy in [-1_isize, 0, 1] {
                            for dz in [-1_isize, 0, 1] {
                                let jx = (ix as isize + dx).rem_euclid(nx as isize) as usize;
                                let jy = (iy as isize + dy).rem_euclid(ny as isize) as usize;
                                let jz = (iz as isize + dz).rem_euclid(nz as isize) as usize;
                                let fi = flat_index(jx, jy, jz, ny, nz);
                                if !neighbors.contains(&fi) {
                                    neighbors.push(fi);
                                }
                            }
                        }
                    }
                    map.push(neighbors);
                }
            }
        }
        map
    }
}

/// Lightweight backup for MC reject path.
///
/// Records the cell-level changes made during a move so they can be undone
/// without a full rebuild.
#[derive(Clone, Debug, Default)]
pub struct CellListBackup {
    /// (particle_idx, old_cell_flat_idx) for particles that moved cells
    moved: Vec<(usize, usize)>,
    /// particle indices that were added (for undo: remove them)
    added: Vec<usize>,
    /// (particle_idx, old_cell_flat_idx) for particles that were removed (for undo: re-add)
    removed: Vec<(usize, usize)>,
}

impl CellList {
    /// Begin tracking changes for potential undo.
    pub fn begin_changes(&self) -> CellListBackup {
        CellListBackup::default()
    }

    /// Update a particle's cell with change tracking.
    /// Silently skips particles not yet in the cell list (sentinel value).
    pub fn update_particle_tracked(
        &mut self,
        particle: usize,
        new_pos: &Point,
        backup: &mut CellListBackup,
    ) {
        let old_cell = *self.particle_cell.get(particle).unwrap_or(&usize::MAX);
        if old_cell == usize::MAX {
            return;
        }
        let new_cell = self.cell_index_for(new_pos);
        if old_cell != new_cell {
            backup.moved.push((particle, old_cell));
            self.remove_from_cell(particle, old_cell);
            self.cells[new_cell].push(particle);
            self.particle_cell[particle] = new_cell;
        }
    }

    /// Undo tracked changes (MC reject path).
    pub fn undo(&mut self, backup: CellListBackup) {
        // Reverse moves
        for (particle, old_cell) in backup.moved {
            let current_cell = self.particle_cell[particle];
            self.remove_from_cell(particle, current_cell);
            self.cells[old_cell].push(particle);
            self.particle_cell[particle] = old_cell;
        }
        // Remove added particles
        for particle in backup.added {
            self.remove_particle(particle);
        }
        // Re-add removed particles
        for (particle, old_cell) in backup.removed {
            self.cells[old_cell].push(particle);
            self.particle_cell[particle] = old_cell;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_cell_assignment() {
        let box_len = [10.0, 10.0, 10.0];
        let cutoff = 3.0;
        let mut cl = CellList::new(box_len, cutoff);

        // 3 cells per dimension (10/3 = 3.33 → 3)
        assert_eq!(cl.num_cells, [3, 3, 3]);
        assert_eq!(cl.cells.len(), 27);

        let positions = vec![
            Point::new(0.0, 0.0, 0.0),    // center cell
            Point::new(-4.0, -4.0, -4.0), // corner cell
            Point::new(4.0, 4.0, 4.0),    // opposite corner
        ];

        cl.build(|i| positions[i], 3, 0..3);

        // All three should be in cells
        assert_eq!(cl.particle_cell.len(), 3);
        for i in 0..3 {
            assert_ne!(cl.particle_cell[i], usize::MAX);
        }
    }

    #[test]
    fn neighbor_query_includes_nearby() {
        let box_len = [10.0, 10.0, 10.0];
        let cutoff = 3.0;
        let mut cl = CellList::new(box_len, cutoff);

        let positions = vec![
            Point::new(0.0, 0.0, 0.0),
            Point::new(1.0, 0.0, 0.0), // same or adjacent cell
            Point::new(4.5, 4.5, 4.5), // distant cell
        ];

        cl.build(|i| positions[i], 3, 0..3);

        let neighbors: Vec<usize> = cl.neighbors(0).collect();
        // Particle 1 should be a neighbor of 0
        assert!(neighbors.contains(&0)); // self
        assert!(neighbors.contains(&1)); // nearby
    }

    #[test]
    fn pbc_wrapping() {
        let box_len = [10.0, 10.0, 10.0];
        let cutoff = 3.0;
        let mut cl = CellList::new(box_len, cutoff);

        // Two particles near opposite edges — PBC neighbors
        let positions = vec![Point::new(-4.9, 0.0, 0.0), Point::new(4.9, 0.0, 0.0)];

        cl.build(|i| positions[i], 2, 0..2);

        let neighbors: Vec<usize> = cl.neighbors(0).collect();
        // Particle 1 should be in neighbor cell due to PBC wrapping
        assert!(
            neighbors.contains(&1),
            "PBC neighbors should be found: neighbors of 0 = {:?}",
            neighbors
        );
    }

    #[test]
    fn update_particle_moves_cell() {
        let box_len = [10.0, 10.0, 10.0];
        let cutoff = 3.0;
        let mut cl = CellList::new(box_len, cutoff);

        let positions = vec![Point::new(0.0, 0.0, 0.0)];
        cl.build(|i| positions[i], 1, 0..1);
        let old_cell = cl.particle_cell[0];

        // Move to a different region
        cl.update_particle(0, &Point::new(4.5, 4.5, 4.5));
        let new_cell = cl.particle_cell[0];

        assert_ne!(old_cell, new_cell);
        assert!(!cl.cells[old_cell].contains(&0));
        assert!(cl.cells[new_cell].contains(&0));
    }

    #[test]
    fn remove_particle() {
        let box_len = [10.0, 10.0, 10.0];
        let cutoff = 3.0;
        let mut cl = CellList::new(box_len, cutoff);

        let positions = vec![Point::new(0.0, 0.0, 0.0), Point::new(1.0, 0.0, 0.0)];
        cl.build(|i| positions[i], 2, 0..2);

        cl.remove_particle(0);
        assert_eq!(cl.particle_cell[0], usize::MAX);
        // Particle 0 should not appear in any neighbor list
        let neighbors: Vec<usize> = cl.neighbors(1).collect();
        assert!(!neighbors.contains(&0));
    }

    #[test]
    fn small_box_single_cell() {
        // Box smaller than cutoff → single cell
        let box_len = [2.0, 2.0, 2.0];
        let cutoff = 3.0;
        let mut cl = CellList::new(box_len, cutoff);

        assert_eq!(cl.num_cells, [1, 1, 1]);
        assert_eq!(cl.cells.len(), 1);
        // All neighbor cells are just cell 0
        assert_eq!(cl.neighbor_offsets[0], vec![0]);

        let positions = vec![Point::new(0.0, 0.0, 0.0), Point::new(0.5, 0.5, 0.5)];
        cl.build(|i| positions[i], 2, 0..2);

        let neighbors: Vec<usize> = cl.neighbors(0).collect();
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
    }
}

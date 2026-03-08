/// Pairwise inter-group nonbonded energy cache.
///
/// Stores E(i,j) for all group pairs so that `group_energies[m]` (the total
/// inter-group energy of group m) can be returned in O(1) instead of O(N_groups).
/// On accept, symmetric delta propagation keeps all entries consistent in O(N_groups).
#[derive(Debug, Clone, Default)]
pub(super) struct GroupEnergyCache {
    /// `pairwise[i * n + j]` = nonbonded energy between groups i and j
    pub(super) pairwise: Vec<f64>,
    /// `group_energies[i]` = Σ_j pairwise[i * n + j]
    pub(super) group_energies: Vec<f64>,
    pub(super) n_groups: usize,
    // Backup buffers live inline so save_backup() reuses capacity instead of
    // allocating new Vecs on every MC step.
    backup_row: Vec<f64>,
    backup_group_energies: Vec<f64>,
    backup_group_index: usize,
    has_backup: bool,
}

impl GroupEnergyCache {
    pub(super) fn new(pairwise: Vec<f64>, group_energies: Vec<f64>, n_groups: usize) -> Self {
        Self {
            pairwise,
            group_energies,
            n_groups,
            ..Default::default()
        }
    }

    pub(super) fn save_backup(&mut self, group_index: usize) {
        let n = self.n_groups;
        let row_start = group_index * n;
        self.backup_row.clear();
        self.backup_row
            .extend_from_slice(&self.pairwise[row_start..row_start + n]);
        self.backup_group_energies.clear();
        self.backup_group_energies
            .extend_from_slice(&self.group_energies);
        self.backup_group_index = group_index;
        self.has_backup = true;
    }

    /// Restore both row and column of the moved group to keep the matrix symmetric.
    pub(super) fn undo(&mut self) {
        if self.has_backup {
            let m = self.backup_group_index;
            let n = self.n_groups;
            for j in 0..n {
                self.pairwise[m * n + j] = self.backup_row[j];
                self.pairwise[j * n + m] = self.backup_row[j];
            }
            self.group_energies
                .copy_from_slice(&self.backup_group_energies);
            self.has_backup = false;
        }
    }

    pub(super) fn discard_backup(&mut self) {
        self.has_backup = false;
    }
}

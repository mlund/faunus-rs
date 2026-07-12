/// Pairwise inter-group nonbonded energy cache.
///
/// Stores E(i,j) for all group pairs so that `group_energies[m]` (the total
/// inter-group energy of group m) can be returned in O(1) instead of O(N_groups).
/// On accept, symmetric delta propagation keeps all entries consistent in O(N_groups).
///
/// Visible to `crate::energy` so that `TabulatedEnergy` can reuse the same cache logic.
#[derive(Debug, Clone, Default)]
pub(in crate::energy) struct GroupEnergyCache {
    /// `pairwise[i * n + j]` = nonbonded energy between groups i and j
    pub(in crate::energy) pairwise: Vec<f64>,
    /// `group_energies[i]` = Σ_j pairwise[i * n + j]
    pub(in crate::energy) group_energies: Vec<f64>,
    pub(in crate::energy) n_groups: usize,
    // Backup buffers live inline so save_backup() reuses capacity instead of
    // allocating new Vecs on every MC step. `backup_rows` holds the pre-move rows of
    // `backup_indices`, concatenated `n_groups` entries each.
    backup_indices: Vec<usize>,
    backup_rows: Vec<f64>,
    backup_group_energies: Vec<f64>,
    has_backup: bool,
}

impl GroupEnergyCache {
    pub(in crate::energy) fn new(
        pairwise: Vec<f64>,
        group_energies: Vec<f64>,
        n_groups: usize,
    ) -> Self {
        Self {
            pairwise,
            group_energies,
            n_groups,
            ..Default::default()
        }
    }

    /// Snapshot the rows (and columns, by symmetry) of the groups about to change, plus the whole
    /// `group_energies` vector, so a rejected move can be undone. Accepts one group (a rigid
    /// translate/rotate) or several at once — a cluster or speciation move recomputes several rows,
    /// and backing up only one would leave the matrix corrupted after a reject.
    pub(in crate::energy) fn save_backup(
        &mut self,
        group_indices: impl IntoIterator<Item = usize>,
    ) {
        let n = self.n_groups;
        self.backup_indices.clear();
        self.backup_rows.clear();
        for m in group_indices {
            self.backup_indices.push(m);
            let start = m * n;
            self.backup_rows
                .extend_from_slice(&self.pairwise[start..start + n]);
        }
        self.backup_group_energies.clear();
        self.backup_group_energies
            .extend_from_slice(&self.group_energies);
        self.has_backup = true;
    }

    /// Restore every backed-up row and column (kept symmetric) and the group energies.
    pub(in crate::energy) fn undo(&mut self) {
        if !self.has_backup {
            return;
        }
        let n = self.n_groups;
        for r in 0..self.backup_indices.len() {
            let m = self.backup_indices[r];
            for j in 0..n {
                let v = self.backup_rows[r * n + j];
                self.pairwise[m * n + j] = v;
                self.pairwise[j * n + m] = v;
            }
        }
        self.group_energies
            .copy_from_slice(&self.backup_group_energies);
        self.has_backup = false;
    }

    pub(in crate::energy) fn discard_backup(&mut self) {
        self.has_backup = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_group_backup_and_undo_restores_every_changed_row() {
        // Symmetric 3×3 pairwise matrix with matching row sums.
        let n = 3;
        #[rustfmt::skip]
        let pairwise = vec![
            0.0, 1.0, 2.0,
            1.0, 0.0, 3.0,
            2.0, 3.0, 0.0,
        ];
        let group_energies = vec![3.0, 4.0, 5.0];
        let mut cache = GroupEnergyCache::new(pairwise.clone(), group_energies.clone(), n);

        // A multi-group move backs up groups 0 and 2, then recomputes their rows/columns and the
        // group energies (as `update` does). Only one row backed up (the old behaviour) would leave
        // the other corrupted after undo — the regression in issue #62.
        cache.save_backup([0, 2]);
        for &m in &[0usize, 2] {
            for j in 0..n {
                cache.pairwise[m * n + j] = 100.0 + m as f64;
                cache.pairwise[j * n + m] = 100.0 + m as f64;
            }
        }
        cache.group_energies = vec![-1.0, -1.0, -1.0];

        cache.undo();
        assert_eq!(
            cache.pairwise, pairwise,
            "all changed rows and columns restored"
        );
        assert_eq!(cache.group_energies, group_energies);
    }

    #[test]
    fn full_matrix_backup_restores_after_a_rebuild() {
        // The volume/everything path backs up every row (`0..n`); a rejected volume move rebuilds
        // the whole matrix, so undo must restore all of it.
        let n = 3;
        #[rustfmt::skip]
        let pairwise = vec![
            0.0, 1.0, 2.0,
            1.0, 0.0, 3.0,
            2.0, 3.0, 0.0,
        ];
        let group_energies = vec![3.0, 4.0, 5.0];
        let mut cache = GroupEnergyCache::new(pairwise.clone(), group_energies.clone(), n);

        cache.save_backup(0..n);
        cache.pairwise.iter_mut().for_each(|x| *x = 42.0); // rebuild_all against the trial box
        cache.group_energies.iter_mut().for_each(|x| *x = 42.0);

        cache.undo();
        assert_eq!(cache.pairwise, pairwise);
        assert_eq!(cache.group_energies, group_energies);
    }

    #[test]
    fn discarded_backup_is_not_restored() {
        // On accept, discard_backup() must make a subsequent undo() a no-op.
        let mut cache = GroupEnergyCache::new(vec![0.0, 1.0, 1.0, 0.0], vec![1.0, 1.0], 2);
        cache.save_backup([0]);
        cache.pairwise[1] = 99.0;
        cache.discard_backup();
        cache.undo();
        assert_eq!(
            cache.pairwise[1], 99.0,
            "accepted move must not be rolled back"
        );
    }

    #[test]
    fn undo_without_a_backup_is_a_noop() {
        let mut cache = GroupEnergyCache::new(vec![7.0], vec![7.0], 1);
        cache.undo();
        assert_eq!(cache.pairwise, vec![7.0]);
    }
}

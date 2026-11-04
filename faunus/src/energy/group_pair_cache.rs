//! Cached inter-group energies, so a rigid-body trial reads its old energy in O(1).
//!
//! [`GroupPairCache`] owns lazy construction, staleness, backup/undo and what each [`Change`] means
//! for the cache. Energy terms supply only [`GroupPairs`] and forward their `StatefulEnergy` hooks.
//! [`PairMatrix`] is the storage underneath, for terms that need their own update policy.

use crate::{Change, GroupChange};
use std::sync::RwLock;

/// What varies between energy terms: the energy of one group pair at the current coordinates.
pub(in crate::energy) trait GroupPairs {
    fn n_groups(&self) -> usize;
    fn pair(&self, i: usize, j: usize) -> f64;

    /// Set `out[j]` to the energy of groups `m` and `j`, for `j ≥ first` and `j ≠ m`. All cache
    /// evaluations come through here, so a term that can sweep a whole row overrides this.
    fn row(&self, m: usize, first: usize, out: &mut [f64]) {
        row_from_pairs(self, m, first, out);
    }
}

/// [`GroupPairs::row`] one pair at a time, for terms that can only sometimes sweep.
pub(in crate::energy) fn row_from_pairs<T: GroupPairs + ?Sized>(
    pairs: &T,
    m: usize,
    first: usize,
    out: &mut [f64],
) {
    for j in (first..pairs.n_groups()).filter(|&j| j != m) {
        out[j] = pairs.pair(m, j);
    }
}

/// [`GroupPairs`] from a closure.
pub(in crate::energy) struct PairFn<F> {
    pub n_groups: usize,
    pub pair: F,
}

impl<F: Fn(usize, usize) -> f64> GroupPairs for PairFn<F> {
    fn n_groups(&self) -> usize {
        self.n_groups
    }
    fn pair(&self, i: usize, j: usize) -> f64 {
        (self.pair)(i, j)
    }
}

/// Symmetric group-pair energies with row sums, and undo for a set of rows.
#[derive(Debug, Clone, Default)]
pub(in crate::energy) struct PairMatrix {
    pairwise: Vec<f64>,
    row_sums: Vec<f64>,
    // Inline so a trial reuses capacity instead of allocating on every move.
    backup_indices: Vec<usize>,
    backup_rows: Vec<f64>,
    backup_row_sums: Vec<f64>,
    has_backup: bool,
}

impl PairMatrix {
    fn zeros(n_groups: usize) -> Self {
        Self {
            pairwise: vec![0.0; n_groups * n_groups],
            row_sums: vec![0.0; n_groups],
            ..Default::default()
        }
    }

    pub(in crate::energy) fn from_pairs(
        n_groups: usize,
        pair: impl FnMut(usize, usize) -> f64,
    ) -> Self {
        let mut matrix = Self::zeros(n_groups);
        matrix.rebuild(pair);
        matrix
    }

    /// `row(i, out)` sets `out[j]` for every `j > i`, so each pair is evaluated once.
    fn from_rows(n_groups: usize, row: impl FnMut(usize, &mut [f64])) -> Self {
        let mut matrix = Self::zeros(n_groups);
        matrix.rebuild_from_rows(row);
        matrix
    }

    /// Overwrite every pair, leaving any backup in place so a rebuild can be undone.
    pub(in crate::energy) fn rebuild(&mut self, mut pair: impl FnMut(usize, usize) -> f64) {
        self.rebuild_from_rows(|i, out| {
            for (j, energy) in out.iter_mut().enumerate().skip(i + 1) {
                *energy = pair(i, j);
            }
        });
    }

    fn rebuild_from_rows(&mut self, mut row: impl FnMut(usize, &mut [f64])) {
        let n = self.n_groups();
        let mut out = vec![0.0; n];
        self.row_sums.fill(0.0);
        for i in 0..n {
            row(i, &mut out);
            for (j, &energy) in out.iter().enumerate().skip(i + 1) {
                self.pairwise[i * n + j] = energy;
                self.pairwise[j * n + i] = energy;
                self.row_sums[i] += energy;
                self.row_sums[j] += energy;
            }
        }
    }

    pub(in crate::energy) fn n_groups(&self) -> usize {
        self.row_sums.len()
    }

    pub(in crate::energy) fn pair(&self, i: usize, j: usize) -> f64 {
        self.pairwise[i * self.n_groups() + j]
    }

    /// Energy of group `m` with every other group.
    pub(in crate::energy) fn row_sum(&self, m: usize) -> f64 {
        self.row_sums[m]
    }

    /// Energy of all group pairs. Row sums count each pair twice.
    pub(in crate::energy) fn total(&self) -> f64 {
        self.row_sums.iter().sum::<f64>() / 2.0
    }

    fn row_slice_sum(&self, m: usize) -> f64 {
        let n = self.n_groups();
        self.pairwise[m * n..(m + 1) * n].iter().sum()
    }

    /// Replace row and column `m`, keeping the other row sums consistent in O(n).
    pub(in crate::energy) fn write_row(&mut self, m: usize, mut pair: impl FnMut(usize) -> f64) {
        let n = self.n_groups();
        for j in 0..n {
            if j == m {
                continue;
            }
            let energy = pair(j);
            let delta = energy - self.pairwise[m * n + j];
            self.pairwise[m * n + j] = energy;
            self.pairwise[j * n + m] = energy;
            // An overlap leaving the row makes the delta non-finite, and ∞ - ∞ would turn the row
            // sum into NaN, so only a resum can recover it.
            if delta.is_finite() {
                self.row_sums[j] += delta;
            } else {
                self.row_sums[j] = self.row_slice_sum(j);
            }
        }
        // Resummed rather than updated by deltas, so the moved group's total cannot drift.
        self.row_sums[m] = self.row_slice_sum(m);
    }

    /// Snapshot rows (and, by symmetry, columns) about to change, and all row sums, since writing
    /// one row changes every sum.
    pub(in crate::energy) fn save_backup(&mut self, rows: impl IntoIterator<Item = usize>) {
        let n = self.n_groups();
        self.backup_indices.clear();
        self.backup_rows.clear();
        for m in rows {
            self.backup_indices.push(m);
            self.backup_rows
                .extend_from_slice(&self.pairwise[m * n..(m + 1) * n]);
        }
        self.backup_row_sums.clear();
        self.backup_row_sums.extend_from_slice(&self.row_sums);
        self.has_backup = true;
    }

    pub(in crate::energy) fn undo(&mut self) {
        if !self.has_backup {
            return;
        }
        let n = self.n_groups();
        for (r, &m) in self.backup_indices.iter().enumerate() {
            for j in 0..n {
                let energy = self.backup_rows[r * n + j];
                self.pairwise[m * n + j] = energy;
                self.pairwise[j * n + m] = energy;
            }
        }
        self.row_sums.copy_from_slice(&self.backup_row_sums);
        self.has_backup = false;
    }

    pub(in crate::energy) fn discard_backup(&mut self) {
        self.has_backup = false;
    }
}

#[derive(Debug, Clone)]
struct Inner {
    matrix: PairMatrix,
    /// Rows to recompute before the next read. A partial update costs a full row recompute however
    /// few atoms moved, so a burst of them on one group is folded into a single recompute.
    stale_rows: Vec<usize>,
    // Kept so a rigid trial doesn't allocate on every move.
    row: Vec<f64>,
}

impl Inner {
    fn build(pairs: &impl GroupPairs) -> Self {
        let n = pairs.n_groups();
        Self {
            matrix: PairMatrix::from_rows(n, |i, out| pairs.row(i, i + 1, out)),
            stale_rows: Vec::new(),
            row: vec![0.0; n],
        }
    }

    fn write_row(&mut self, m: usize, pairs: &impl GroupPairs) {
        pairs.row(m, 0, &mut self.row);
        let row = &self.row;
        self.matrix.write_row(m, |j| row[j]);
    }

    fn settle(&mut self, pairs: &impl GroupPairs) {
        while let Some(m) = self.stale_rows.pop() {
            self.write_row(m, pairs);
        }
    }
}

/// Settling k rows evaluates k(n-1) pairs and a rebuild n(n-1)/2, so past n/2 stale rows the
/// matrix isn't worth keeping.
fn mark_stale(cache: &mut Option<Inner>, rows: impl IntoIterator<Item = usize>) {
    let Some(inner) = cache.as_mut() else {
        return;
    };
    for m in rows {
        if !inner.stale_rows.contains(&m) {
            inner.stale_rows.push(m);
        }
    }
    if 2 * inner.stale_rows.len() > inner.matrix.n_groups() {
        *cache = None;
    }
}

/// Group energies that stay correct across any sequence of trial moves, given the
/// `StatefulEnergy` call order: `save_backup` pre-move, `refresh` post-move, then `undo` or
/// `discard_backup`.
#[derive(Debug, Default)]
pub(in crate::energy) struct GroupPairCache {
    // Not `RefCell`: downstream crates share `NonbondedMatrix` across threads, so it must stay `Sync`.
    inner: RwLock<Option<Inner>>,
}

impl Clone for GroupPairCache {
    fn clone(&self) -> Self {
        Self {
            inner: RwLock::new(self.inner.read().unwrap().clone()),
        }
    }
}

impl GroupPairCache {
    /// Energy of group `gi` with all other groups at the current coordinates.
    pub(in crate::energy) fn group_energy(&self, gi: usize, pairs: &impl GroupPairs) -> f64 {
        if let Some(inner) = self.inner.read().unwrap().as_ref() {
            debug_assert_eq!(inner.matrix.n_groups(), pairs.n_groups());
            if inner.stale_rows.is_empty() {
                return inner.matrix.row_sum(gi);
            }
        }
        let mut guard = self.inner.write().unwrap();
        let inner = guard.get_or_insert_with(|| Inner::build(pairs));
        inner.settle(pairs);
        inner.matrix.row_sum(gi)
    }

    /// Call pre-move. Only a rigid-body change is backed up: it alone recomputes eagerly, as its
    /// new energy is read straight back. The matrix must describe the old coordinates before the
    /// snapshot: a row built or settled later would disagree with what `undo` restores. A prior
    /// read can't be relied on, since the Hamiltonian stops at its first non-finite term.
    pub(in crate::energy) fn save_backup(&self, change: &Change, pairs: &impl GroupPairs) {
        if let Change::SingleGroup(gi, GroupChange::RigidBody) = change {
            let mut guard = self.inner.write().unwrap();
            let inner = guard.get_or_insert_with(|| Inner::build(pairs));
            inner.settle(pairs);
            inner.matrix.save_backup([*gi]);
        }
    }

    /// Call post-move. Marking a row stale needs no backup: it is recomputed from whichever
    /// coordinates are current at the next read, restored ones after a reject.
    pub(in crate::energy) fn refresh(&self, change: &Change, pairs: &impl GroupPairs) {
        let mut guard = self.inner.write().unwrap();
        match change {
            Change::None | Change::SingleGroup(_, GroupChange::None) => {}
            Change::SingleGroup(gi, GroupChange::RigidBody) => {
                if let Some(inner) = guard.as_mut() {
                    inner.write_row(*gi, pairs);
                }
            }
            Change::SingleGroup(gi, _) => mark_stale(&mut guard, [*gi]),
            Change::Groups(changes) => {
                let moved = changes
                    .iter()
                    .filter(|(_, group_change)| !matches!(group_change, GroupChange::None))
                    .map(|(gi, _)| *gi);
                mark_stale(&mut guard, moved);
            }
            Change::Everything | Change::Volume(..) => *guard = None,
        }
    }

    pub(in crate::energy) fn undo(&mut self) {
        if let Some(inner) = self.inner.get_mut().unwrap().as_mut() {
            inner.matrix.undo();
        }
    }

    pub(in crate::energy) fn discard_backup(&mut self) {
        if let Some(inner) = self.inner.get_mut().unwrap().as_mut() {
            inner.matrix.discard_backup();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::{GroupSize, RelIndex};
    use std::cell::{Cell, RefCell};

    /// Stand-in for an energy term: the true pair energies, which a test edits to act as a move.
    struct Truth {
        pairs: RefCell<[[f64; 3]; 3]>,
        evaluations: Cell<usize>,
    }

    impl Truth {
        /// Row sums are 3, 4 and 5.
        fn new() -> Self {
            Self {
                pairs: RefCell::new([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]),
                evaluations: Cell::new(0),
            }
        }

        fn set(&self, i: usize, j: usize, energy: f64) {
            let mut pairs = self.pairs.borrow_mut();
            pairs[i][j] = energy;
            pairs[j][i] = energy;
        }
    }

    impl GroupPairs for Truth {
        fn n_groups(&self) -> usize {
            3
        }
        fn pair(&self, i: usize, j: usize) -> f64 {
            self.evaluations.set(self.evaluations.get() + 1);
            self.pairs.borrow()[i][j]
        }
    }

    const fn rigid(group: usize) -> Change {
        Change::SingleGroup(group, GroupChange::RigidBody)
    }

    fn partial(group: usize) -> Change {
        Change::SingleGroup(group, GroupChange::PartialUpdate(vec![RelIndex::new(0)]))
    }

    fn energies(cache: &GroupPairCache, truth: &Truth) -> [f64; 3] {
        [0, 1, 2].map(|g| cache.group_energy(g, truth))
    }

    #[test]
    fn built_on_first_read_then_served_without_evaluating() {
        let truth = Truth::new();
        let cache = GroupPairCache::default();
        assert_eq!(energies(&cache, &truth), [3.0, 4.0, 5.0]);
        assert_eq!(truth.evaluations.get(), 3, "one evaluation per unique pair");
    }

    #[test]
    fn rejected_rigid_move_restores_and_accepted_one_keeps() {
        let truth = Truth::new();
        let mut cache = GroupPairCache::default();
        energies(&cache, &truth);

        cache.save_backup(&rigid(0), &truth);
        truth.set(0, 1, 10.0);
        cache.refresh(&rigid(0), &truth);
        assert_eq!(energies(&cache, &truth), [12.0, 13.0, 5.0]);

        truth.set(0, 1, 1.0);
        cache.undo();
        assert_eq!(energies(&cache, &truth), [3.0, 4.0, 5.0]);

        cache.save_backup(&rigid(0), &truth);
        truth.set(0, 1, 10.0);
        cache.refresh(&rigid(0), &truth);
        cache.discard_backup();
        cache.undo();
        assert_eq!(energies(&cache, &truth), [12.0, 13.0, 5.0]);
    }

    #[test]
    fn burst_of_partial_updates_costs_one_row_at_the_next_read() {
        let truth = Truth::new();
        let cache = GroupPairCache::default();
        energies(&cache, &truth);
        truth.evaluations.set(0);

        for energy in [10.0, 20.0, 30.0] {
            truth.set(0, 1, energy);
            cache.refresh(&partial(0), &truth);
        }
        assert_eq!(truth.evaluations.get(), 0);
        assert_eq!(energies(&cache, &truth), [32.0, 33.0, 5.0]);
        assert_eq!(
            truth.evaluations.get(),
            2,
            "row 0 has two off-diagonal pairs"
        );
    }

    /// A partial update leaves a row stale, then a rigid trial backs up with no read in between
    /// (a non-finite term earlier in the Hamiltonian ends the old-energy read before this one).
    #[test]
    fn rejecting_a_rigid_move_taken_over_a_stale_row_keeps_it_consistent() {
        let truth = Truth::new();
        let mut cache = GroupPairCache::default();
        energies(&cache, &truth);

        truth.set(0, 1, 10.0);
        truth.set(0, 2, 20.0);
        cache.refresh(&partial(0), &truth);
        cache.discard_backup();

        cache.save_backup(&rigid(1), &truth);
        truth.set(0, 1, 100.0);
        truth.set(1, 2, 300.0);
        cache.refresh(&rigid(1), &truth);
        assert_eq!(cache.group_energy(1, &truth), 400.0);

        truth.set(0, 1, 10.0);
        truth.set(1, 2, 3.0);
        cache.undo();
        assert_eq!(energies(&cache, &truth), [30.0, 13.0, 23.0]);
    }

    /// With no read before the trial (see above), the first read would otherwise build the matrix
    /// from trial coordinates, leaving nothing for `undo` to restore.
    #[test]
    fn rejecting_the_first_move_on_an_unbuilt_cache_keeps_the_old_coordinates() {
        let truth = Truth::new();
        let mut cache = GroupPairCache::default();

        cache.save_backup(&rigid(0), &truth);
        truth.set(0, 1, 10.0);
        cache.refresh(&rigid(0), &truth);
        assert_eq!(cache.group_energy(0, &truth), 12.0);

        truth.set(0, 1, 1.0);
        cache.undo();
        assert_eq!(energies(&cache, &truth), [3.0, 4.0, 5.0]);
    }

    #[test]
    fn leaving_an_overlap_recovers_every_row_sum() {
        let truth = Truth::new();
        truth.set(0, 1, f64::INFINITY);
        let cache = GroupPairCache::default();
        assert_eq!(cache.group_energy(1, &truth), f64::INFINITY);

        truth.set(0, 1, 1.0);
        cache.refresh(&rigid(0), &truth);
        assert_eq!(energies(&cache, &truth), [3.0, 4.0, 5.0]);
    }

    #[test]
    fn whole_system_change_is_never_served_from_the_old_cache() {
        let truth = Truth::new();
        let cache = GroupPairCache::default();
        energies(&cache, &truth);

        truth.set(0, 1, 10.0);
        cache.refresh(&Change::Everything, &truth);
        assert_eq!(energies(&cache, &truth), [12.0, 13.0, 5.0]);
    }

    fn insertion(group: usize) -> Change {
        Change::Groups(vec![(
            group,
            GroupChange::ResizePartial(GroupSize::Expand(1), vec![RelIndex::new(0)]),
        )])
    }

    /// Only the named row is evaluated: rebuilding here is what GCMC must not pay for.
    #[test]
    fn insertion_recomputes_one_row_at_the_next_read() {
        let truth = Truth::new();
        let cache = GroupPairCache::default();
        energies(&cache, &truth);
        truth.evaluations.set(0);

        truth.set(0, 1, 10.0);
        cache.refresh(&insertion(0), &truth);
        assert_eq!(truth.evaluations.get(), 0);
        assert_eq!(energies(&cache, &truth), [12.0, 13.0, 5.0]);
        assert_eq!(truth.evaluations.get(), 2);
    }

    #[test]
    fn rejected_insertion_reads_the_restored_coordinates() {
        let truth = Truth::new();
        let mut cache = GroupPairCache::default();
        energies(&cache, &truth);

        cache.save_backup(&insertion(0), &truth);
        truth.set(0, 1, 10.0);
        cache.refresh(&insertion(0), &truth);

        truth.set(0, 1, 1.0);
        cache.undo();
        assert_eq!(energies(&cache, &truth), [3.0, 4.0, 5.0]);
    }

    /// Settling two of three rows takes 4 evaluations, a rebuild only the 3 unique pairs.
    #[test]
    fn most_rows_stale_rebuilds_instead() {
        let truth = Truth::new();
        let cache = GroupPairCache::default();
        energies(&cache, &truth);
        truth.evaluations.set(0);

        truth.set(0, 1, 10.0);
        let cluster = [0, 1].map(|g| (g, GroupChange::RigidBody)).to_vec();
        cache.refresh(&Change::Groups(cluster), &truth);
        assert_eq!(energies(&cache, &truth), [12.0, 13.0, 5.0]);
        assert_eq!(truth.evaluations.get(), 3);
    }

    /// A term with a cheaper whole-row evaluation, where any single-pair call would defeat it.
    struct Sweeping(Truth);

    impl GroupPairs for Sweeping {
        fn n_groups(&self) -> usize {
            self.0.n_groups()
        }
        fn pair(&self, _: usize, _: usize) -> f64 {
            panic!("a sweeping term is only asked for rows");
        }
        fn row(&self, m: usize, first: usize, out: &mut [f64]) {
            self.0.row(m, first, out);
        }
    }

    /// Build, stale-row settling and rigid refresh are the only places pairs are evaluated.
    #[test]
    fn a_sweeping_term_is_only_asked_for_rows() {
        let sweeping = Sweeping(Truth::new());
        let mut cache = GroupPairCache::default();
        assert_eq!(
            [0, 1, 2].map(|g| cache.group_energy(g, &sweeping)),
            [3.0, 4.0, 5.0]
        );

        sweeping.0.set(0, 1, 10.0);
        cache.refresh(&insertion(0), &sweeping);
        assert_eq!(cache.group_energy(1, &sweeping), 13.0);

        cache.save_backup(&rigid(2), &sweeping);
        sweeping.0.set(1, 2, 20.0);
        cache.refresh(&rigid(2), &sweeping);
        cache.discard_backup();
        assert_eq!(
            [0, 1, 2].map(|g| cache.group_energy(g, &sweeping)),
            [12.0, 30.0, 22.0]
        );
    }

    #[test]
    fn no_change_keeps_the_cache() {
        let truth = Truth::new();
        let cache = GroupPairCache::default();
        energies(&cache, &truth);
        truth.evaluations.set(0);

        cache.save_backup(&Change::None, &truth);
        cache.refresh(&Change::None, &truth);
        assert_eq!(energies(&cache, &truth), [3.0, 4.0, 5.0]);
        assert_eq!(truth.evaluations.get(), 0);
    }

    fn symmetric_matrix() -> PairMatrix {
        let truth = Truth::new();
        PairMatrix::from_pairs(3, |i, j| truth.pair(i, j))
    }

    fn snapshot(matrix: &PairMatrix) -> ([f64; 3], [f64; 3]) {
        (
            [0, 1, 2].map(|m| matrix.row_sum(m)),
            [matrix.pair(0, 1), matrix.pair(0, 2), matrix.pair(1, 2)],
        )
    }

    /// A cluster or speciation move writes several rows; undo must restore all of them.
    #[test]
    fn undo_restores_every_backed_up_row() {
        let mut matrix = symmetric_matrix();
        matrix.save_backup([0, 2]);
        matrix.write_row(0, |_| 100.0);
        matrix.write_row(2, |_| 200.0);
        matrix.undo();
        assert_eq!(snapshot(&matrix), ([3.0, 4.0, 5.0], [1.0, 2.0, 3.0]));
    }

    /// A volume move backs up every row, then rebuilds against the trial box.
    #[test]
    fn undo_restores_a_rebuild() {
        let mut matrix = symmetric_matrix();
        matrix.save_backup(0..3);
        matrix.rebuild(|_, _| 42.0);
        assert_eq!(matrix.total(), 126.0);
        matrix.undo();
        assert_eq!(snapshot(&matrix), ([3.0, 4.0, 5.0], [1.0, 2.0, 3.0]));
    }
}

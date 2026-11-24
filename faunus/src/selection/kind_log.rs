// Copyright (c) 2026 Mikael Lund
// SPDX-License-Identifier: Apache-2.0

//! Atom-kind change counter that remembers which atoms its latest bumps touched.

/// Entries kept before the oldest half is dropped. A titration swap touches one or two atoms, so
/// a cache that falls further behind than this is cheaper to re-resolve in full anyway.
const CAPACITY: usize = 64;

/// Counts atom-kind changes and logs the index behind each recent one.
///
/// A selection cache that knows its last generation can then re-evaluate only the changed atoms
/// instead of the whole system. Bumping and logging happen in the same call, so the counter
/// cannot advance past a change the log does not hold.
#[derive(Clone, Debug, Default)]
pub(crate) struct AtomKindLog {
    /// Generation before `recent[0]`.
    base: u64,
    /// `recent[k]` is the atom whose change produced generation `base + k + 1`.
    recent: Vec<usize>,
}

impl AtomKindLog {
    /// Number of changes recorded so far.
    pub(crate) fn generation(&self) -> u64 {
        self.base + self.recent.len() as u64
    }

    /// Record that the kind of atom `index` changed.
    pub(crate) fn record(&mut self, index: usize) {
        if self.recent.len() == CAPACITY {
            // Halving, not popping one, keeps the shift amortised O(1) per record.
            let dropped = CAPACITY / 2;
            self.recent.drain(..dropped);
            self.base += dropped as u64;
        }
        self.recent.push(index);
    }

    /// Atoms changed after `generation`, oldest first; `None` if some are no longer logged.
    pub(crate) fn since(&self, generation: u64) -> Option<&[usize]> {
        let offset = usize::try_from(generation.checked_sub(self.base)?).ok()?;
        self.recent.get(offset..)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log_of(indices: impl IntoIterator<Item = usize>) -> AtomKindLog {
        let mut log = AtomKindLog::default();
        indices.into_iter().for_each(|i| log.record(i));
        log
    }

    #[test]
    fn since_returns_changes_after_the_given_generation() {
        let log = log_of([7, 3, 7]);
        assert_eq!(log.generation(), 3);
        assert_eq!(log.since(1), Some(&[3, 7][..]));
        assert_eq!(log.since(3), Some(&[][..]));
    }

    /// Dropping old entries must not shift the generation, or a cache could match a stale key.
    #[test]
    fn truncation_keeps_generation_and_refuses_dropped_ranges() {
        let log = log_of(0..CAPACITY + 1);
        assert_eq!(log.generation(), CAPACITY as u64 + 1);
        assert_eq!(log.since(0), None);
        assert_eq!(log.since(CAPACITY as u64), Some(&[CAPACITY][..]));
    }

    /// A generation from the future (another context, say) is not a range the log can vouch for.
    #[test]
    fn unknown_future_generation_is_refused() {
        assert_eq!(log_of([1]).since(2), None);
    }
}

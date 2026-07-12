// Copyright 2025 Mikael Lund
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

//! Complete-by-construction backup for move-mutable energy-term state.

/// Owns a term's move-mutable state `S` together with one reusable backup slot.
///
/// Because the whole `S` is captured by a single [`save`](Self::save), a field
/// added to `S` is automatically covered by [`undo`](Self::undo) — the
/// miss-a-field backup bug behind the Ewald reject-path regressions cannot recur.
/// Buffers are reused across trials: `save` copies via `clone_from` (reusing the
/// backup's capacity for `Vec`-typed fields) and `undo` swaps (no allocation).
///
/// `save` is deliberately explicit, not implicit on every trial: a term calls it
/// only when a move actually mutates `S`, so conditional-backup terms keep
/// skipping no-op trials at zero cost.
#[derive(Clone, Debug)]
pub(crate) struct Snapshot<S: Clone> {
    current: S,
    /// Kept allocated between trials so `save` reuses its capacity.
    backup: S,
    armed: bool,
}

impl<S: Clone> Snapshot<S> {
    pub(crate) fn new(state: S) -> Self {
        Self {
            backup: state.clone(),
            current: state,
            armed: false,
        }
    }

    /// Snapshot the current state so a later [`undo`](Self::undo) can restore it.
    ///
    /// A trial saves at most once (paired with a later `undo`/`discard`); a second `save` while
    /// still armed would silently narrow what `undo` restores, so it is a misuse the assertion
    /// catches in debug builds.
    pub(crate) fn save(&mut self) {
        debug_assert!(
            !self.armed,
            "Snapshot::save called twice without undo/discard"
        );
        self.backup.clone_from(&self.current);
        self.armed = true;
    }

    /// Restore the last saved state (MC reject). No-op when not armed.
    pub(crate) fn undo(&mut self) {
        if self.armed {
            std::mem::swap(&mut self.current, &mut self.backup);
            self.armed = false;
        }
    }

    /// Accept the current state, disarming the backup (MC accept).
    pub(crate) fn discard(&mut self) {
        self.armed = false;
    }
}

impl<S: Clone> std::ops::Deref for Snapshot<S> {
    type Target = S;
    fn deref(&self) -> &S {
        &self.current
    }
}

impl<S: Clone> std::ops::DerefMut for Snapshot<S> {
    fn deref_mut(&mut self) -> &mut S {
        &mut self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undo_restores_saved_state() {
        let mut snap = Snapshot::new(vec![1, 2, 3]);
        snap.save();
        snap.push(4);
        assert_eq!(*snap, vec![1, 2, 3, 4]);
        snap.undo();
        assert_eq!(*snap, vec![1, 2, 3]);
    }

    #[test]
    fn discard_keeps_current_state() {
        let mut snap = Snapshot::new(vec![1, 2, 3]);
        snap.save();
        snap.push(4);
        snap.discard();
        assert_eq!(*snap, vec![1, 2, 3, 4]);
        // A disarmed undo must not roll back an accepted move.
        snap.undo();
        assert_eq!(*snap, vec![1, 2, 3, 4]);
    }

    #[test]
    fn undo_without_save_is_a_noop() {
        let mut snap = Snapshot::new(vec![1, 2, 3]);
        snap.push(4);
        snap.undo();
        assert_eq!(*snap, vec![1, 2, 3, 4]);
    }
}

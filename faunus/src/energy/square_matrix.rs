// Copyright 2023-2024 Mikael Lund
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

//! Row-major square matrix indexed by particle or atom-kind pairs.

use std::ops::{Index, IndexMut};

/// Row-major square matrix indexed by a pair of atom kinds or particles.
///
/// Row-major (unlike nalgebra's column-major `DMatrix`) so that iterating row
/// `i` over columns `j` reads sequential memory, which is what the pair loops do.
///
/// Index it with a `(row, column)` tuple:
/// ```ignore
/// let potential = &potentials[(i, j)];
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SquareMatrix<T> {
    data: Vec<T>,
    order: usize,
}

impl<T> SquareMatrix<T> {
    /// Build an `order × order` matrix from a function of the row and column.
    pub(crate) fn from_fn(order: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut data = Vec::with_capacity(order * order);
        for i in 0..order {
            for j in 0..order {
                data.push(f(i, j));
            }
        }
        Self { data, order }
    }

    /// Number of rows, which equals the number of columns.
    pub const fn order(&self) -> usize {
        self.order
    }

    #[inline]
    pub fn get(&self, (i, j): (usize, usize)) -> Option<&T> {
        if i < self.order && j < self.order {
            self.data.get(i * self.order + j)
        } else {
            None
        }
    }

    /// Unchecked element access for inner loops.
    ///
    /// # Safety
    /// Both `i` and `j` must be less than [`order`](Self::order).
    #[inline]
    pub(crate) unsafe fn uget(&self, (i, j): (usize, usize)) -> &T {
        debug_assert!(i < self.order && j < self.order);
        self.data.get_unchecked(i * self.order + j)
    }

    /// Contiguous row slice, so an inner loop can use `get_unchecked(j)` on a
    /// single slice instead of recomputing `i * order + j` each iteration.
    #[inline]
    pub(crate) fn row(&self, i: usize) -> &[T] {
        debug_assert!(i < self.order);
        let start = i * self.order;
        &self.data[start..start + self.order]
    }

    /// Iterate all elements in row-major order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }
}

impl<T: Clone> SquareMatrix<T> {
    /// Build an `order × order` matrix with every element cloned from `value`.
    pub(crate) fn from_element(order: usize, value: T) -> Self {
        Self {
            data: vec![value; order * order],
            order,
        }
    }
}

impl<T> Index<(usize, usize)> for SquareMatrix<T> {
    type Output = T;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.data[i * self.order + j]
    }
}

impl<T> IndexMut<(usize, usize)> for SquareMatrix<T> {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        &mut self.data[i * self.order + j]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_fn_is_row_major() {
        let m = SquareMatrix::from_fn(3, |i, j| 10 * i + j);
        assert_eq!(m.order(), 3);
        // Row-major: row 1 is contiguous and holds (1,0), (1,1), (1,2).
        assert_eq!(m.row(1), &[10, 11, 12]);
        assert_eq!(m[(2, 0)], 20);
        assert_eq!(m[(0, 2)], 2);
    }

    #[test]
    fn row_matches_indexing() {
        let m = SquareMatrix::from_fn(4, |i, j| i * j);
        for i in 0..4 {
            let row = m.row(i);
            assert_eq!(row.len(), 4);
            for j in 0..4 {
                assert_eq!(row[j], m[(i, j)]);
            }
        }
    }

    #[test]
    fn get_bounds_checks_both_axes() {
        let m = SquareMatrix::from_fn(2, |i, j| (i, j));
        assert_eq!(m.get((1, 1)), Some(&(1, 1)));
        assert_eq!(m.get((2, 0)), None);
        // Column overflow must be rejected per-axis: the raw offset 0*2+2 would
        // otherwise land on (1, 0) and silently return the wrong element.
        assert_eq!(m.get((0, 2)), None);
    }

    #[test]
    fn uget_agrees_with_get() {
        let m = SquareMatrix::from_fn(3, |i, j| 10 * i + j);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(unsafe { m.uget((i, j)) }, m.get((i, j)).unwrap());
            }
        }
    }

    #[test]
    fn from_element_fills_every_slot() {
        let m = SquareMatrix::from_element(3, 7u8);
        assert_eq!(m.order(), 3);
        assert!(m.iter().all(|&v| v == 7));
        assert_eq!(m.iter().count(), 9);
    }

    #[test]
    fn index_mut_writes_one_element() {
        let mut m = SquareMatrix::from_element(2, 0u8);
        m[(0, 1)] = 5;
        assert_eq!(m[(0, 1)], 5);
        assert_eq!(m[(1, 0)], 0, "indexing must not be symmetric");
    }

    #[test]
    fn iter_is_row_major() {
        let m = SquareMatrix::from_fn(2, |i, j| 10 * i + j);
        assert_eq!(m.iter().copied().collect::<Vec<_>>(), vec![0, 1, 10, 11]);
    }

    #[test]
    fn zero_order_is_empty() {
        let m = SquareMatrix::<u8>::from_element(0, 1);
        assert_eq!(m.order(), 0);
        assert_eq!(m.iter().count(), 0);
        assert_eq!(m.get((0, 0)), None);
    }
}

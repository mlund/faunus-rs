1. Study new hermite spline in `interatomic`'s "hermite-spline" branch
2. Make a new `NonbondedMatrixSplined` struct that take an existing `NonbondedMatrix` and splines all stored pair-potentials.
3. Storage can either be on a (triagular) pair-matrix where current Spline data is stored. Or it may be better(??) to make a completely new spline storage for all NxN pair in contained vectors, something like:
```
pub struct PairPotentials {
    n: usize,
    y0: Vec<f32>,
    y1: Vec<f32>,
    m0: Vec<f32>,
    m1: Vec<f32>,
}

impl PairPotentials {
    pub fn new(n: usize) -> Self {
        let size = n * (n + 1) / 2;
        Self {
            n,
            y0: vec![0.0; size],
            y1: vec![0.0; size],
            m0: vec![0.0; size],
            m1: vec![0.0; size],
        }
    }

    #[inline(always)]
    fn idx(&self, i: usize, j: usize) -> usize {
        let lo = i.min(j);
        let hi = i.max(j);
        hi * (hi + 1) / 2 + lo
    }

    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> (f32, f32, f32, f32) {
        let k = self.idx(i, j);
        (
            self.y0[k],
            self.y1[k],
            self.m0[k],
            self.m1[k],
        )
    }
}
```
3. Implement `NonbondedTerm` for `NonbondedMatrixSplined`
4. Add unit test

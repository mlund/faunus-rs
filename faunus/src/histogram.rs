/// Reusable 1D histogram with uniform bin widths.
///
/// Bins span `[min, min + num_bins * bin_width)` where `num_bins = floor((max - min) / bin_width)`.
/// Partial trailing bins are excluded to keep all bins the same width.
#[derive(Debug, Clone)]
pub struct Histogram {
    min: f64,
    bin_width: f64,
    bins: Vec<f64>,
}

impl Histogram {
    /// Create a new histogram with the given range and bin width.
    ///
    /// The effective upper bound is `min + num_bins * bin_width`, which may be
    /// slightly less than `max` to avoid a partial trailing bin.
    pub fn new(min: f64, max: f64, bin_width: f64) -> Self {
        assert!(max > min, "max must be greater than min");
        assert!(bin_width > 0.0, "bin_width must be positive");
        let num_bins = ((max - min) / bin_width) as usize;
        assert!(num_bins > 0, "range too small for given bin_width");
        Self {
            min,
            bin_width,
            bins: vec![0.0; num_bins],
        }
    }

    /// Increment the bin corresponding to `value`. Out-of-range values are silently ignored.
    pub fn add(&mut self, value: f64) {
        if let Some(bin) = self.bin_index(value) {
            self.bins[bin] += 1.0;
        }
    }

    /// Number of bins.
    pub const fn num_bins(&self) -> usize {
        self.bins.len()
    }

    /// Center of the i-th bin.
    pub fn bin_center(&self, i: usize) -> f64 {
        (i as f64 + 0.5).mul_add(self.bin_width, self.min)
    }

    /// Bin width.
    pub const fn bin_width(&self) -> f64 {
        self.bin_width
    }

    /// Iterator over `(bin_center, count)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.bins
            .iter()
            .enumerate()
            .map(|(i, &count)| (self.bin_center(i), count))
    }

    /// Reset all bins to zero.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.bins.fill(0.0);
    }

    /// Raw bin index for a value, or `None` if out of range.
    fn bin_index(&self, value: f64) -> Option<usize> {
        if value < self.min {
            return None;
        }
        let i = ((value - self.min) / self.bin_width) as usize;
        if i < self.bins.len() {
            Some(i)
        } else {
            None
        }
    }

    /// Direct read access to the count in the i-th bin.
    #[allow(dead_code)]
    pub fn count(&self, i: usize) -> f64 {
        self.bins[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn basic_binning() {
        let mut h = Histogram::new(0.0, 10.0, 1.0);
        assert_eq!(h.num_bins(), 10);
        assert_relative_eq!(h.bin_center(0), 0.5);
        assert_relative_eq!(h.bin_center(9), 9.5);

        h.add(0.0);
        h.add(0.9);
        h.add(5.5);
        assert_relative_eq!(h.count(0), 2.0);
        assert_relative_eq!(h.count(5), 1.0);
    }

    #[test]
    fn out_of_range_ignored() {
        let mut h = Histogram::new(0.0, 10.0, 1.0);
        h.add(-0.1);
        h.add(10.0);
        h.add(100.0);
        let total: f64 = h.iter().map(|(_, c)| c).sum();
        assert_relative_eq!(total, 0.0);
    }

    #[test]
    fn partial_last_bin_excluded() {
        // 10.5 / 2.0 = 5.25 → 5 full bins, effective max = 10.0
        let h = Histogram::new(0.0, 10.5, 2.0);
        assert_eq!(h.num_bins(), 5);
        // Value at 10.0 is at bin boundary of the 6th (non-existent) bin
        let mut h2 = h.clone();
        h2.add(10.0);
        let total: f64 = h2.iter().map(|(_, c)| c).sum();
        assert_relative_eq!(total, 0.0);
    }

    #[test]
    fn clear_resets() {
        let mut h = Histogram::new(0.0, 10.0, 1.0);
        h.add(3.0);
        h.add(7.0);
        h.clear();
        let total: f64 = h.iter().map(|(_, c)| c).sum();
        assert_relative_eq!(total, 0.0);
    }

    #[test]
    fn boundary_values() {
        let mut h = Histogram::new(1.0, 3.0, 0.5);
        assert_eq!(h.num_bins(), 4);
        // Exactly at min → bin 0
        h.add(1.0);
        assert_relative_eq!(h.count(0), 1.0);
        // Just below effective max (3.0) → last bin
        h.add(2.99);
        assert_relative_eq!(h.count(3), 1.0);
        // Exactly at effective max → out of range
        h.add(3.0);
        assert_relative_eq!(h.count(3), 1.0);
    }
}

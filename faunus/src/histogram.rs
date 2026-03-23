/// Reusable 1D histogram with uniform bin widths.
///
/// Bins span `[min, min + num_bins * bin_width)` where `num_bins = floor((max - min) / bin_width)`.
/// Partial trailing bins are excluded to keep all bins the same width.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    pub fn new(min: f64, max: f64, bin_width: f64) -> anyhow::Result<Self> {
        anyhow::ensure!(max > min, "max must be greater than min");
        anyhow::ensure!(bin_width > 0.0, "bin_width must be positive");
        let num_bins = ((max - min) / bin_width) as usize;
        anyhow::ensure!(num_bins > 0, "range too small for given bin_width");
        Ok(Self {
            min,
            bin_width,
            bins: vec![0.0; num_bins],
        })
    }

    /// Increment the bin corresponding to `value`. Out-of-range values are silently ignored.
    pub fn add(&mut self, value: f64) {
        if let Some(bin) = self.bin_index(value) {
            self.bins[bin] += 1.0;
        }
    }

    /// Add a weighted count to the bin corresponding to `value`. Out-of-range values are silently ignored.
    pub fn add_weighted(&mut self, value: f64, weight: f64) {
        if let Some(bin) = self.bin_index(value) {
            self.bins[bin] += weight;
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

    /// Sum of all bin counts.
    pub fn total_count(&self) -> f64 {
        self.bins.iter().sum()
    }

    /// Fraction of counts in bins whose centers fall within `[lo, hi]`.
    pub fn fraction_in_range(&self, lo: f64, hi: f64) -> f64 {
        let total = self.total_count();
        if total == 0.0 {
            return 0.0;
        }
        let count: f64 = self
            .iter()
            .filter_map(|(center, c)| (center >= lo && center <= hi).then_some(c))
            .sum();
        count / total
    }

    /// Element-wise add bins from another histogram with the same shape.
    #[allow(dead_code)]
    pub fn merge(&mut self, other: &Histogram) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.bins.len() == other.bins.len()
                && (self.min - other.min).abs() < 1e-12
                && (self.bin_width - other.bin_width).abs() < 1e-12,
            "Cannot merge histograms with different shapes"
        );
        for (a, b) in self.bins.iter_mut().zip(&other.bins) {
            *a += b;
        }
        Ok(())
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
        let mut h = Histogram::new(0.0, 10.0, 1.0).unwrap();
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
        let mut h = Histogram::new(0.0, 10.0, 1.0).unwrap();
        h.add(-0.1);
        h.add(10.0);
        h.add(100.0);
        let total: f64 = h.iter().map(|(_, c)| c).sum();
        assert_relative_eq!(total, 0.0);
    }

    #[test]
    fn partial_last_bin_excluded() {
        // 10.5 / 2.0 = 5.25 → 5 full bins, effective max = 10.0
        let h = Histogram::new(0.0, 10.5, 2.0).unwrap();
        assert_eq!(h.num_bins(), 5);
        // Value at 10.0 is at bin boundary of the 6th (non-existent) bin
        let mut h2 = h.clone();
        h2.add(10.0);
        let total: f64 = h2.iter().map(|(_, c)| c).sum();
        assert_relative_eq!(total, 0.0);
    }

    #[test]
    fn clear_resets() {
        let mut h = Histogram::new(0.0, 10.0, 1.0).unwrap();
        h.add(3.0);
        h.add(7.0);
        h.clear();
        let total: f64 = h.iter().map(|(_, c)| c).sum();
        assert_relative_eq!(total, 0.0);
    }

    #[test]
    fn add_weighted() {
        let mut h = Histogram::new(0.0, 10.0, 1.0).unwrap();
        h.add_weighted(0.5, 2.5);
        h.add_weighted(0.9, 1.0);
        assert_relative_eq!(h.count(0), 3.5);

        // Out of range ignored
        h.add_weighted(-1.0, 10.0);
        h.add_weighted(10.0, 10.0);
        assert_relative_eq!(h.iter().map(|(_, c)| c).sum::<f64>(), 3.5);
    }

    #[test]
    fn boundary_values() {
        let mut h = Histogram::new(1.0, 3.0, 0.5).unwrap();
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

    #[test]
    fn total_count() {
        let mut h = Histogram::new(0.0, 10.0, 1.0).unwrap();
        assert_relative_eq!(h.total_count(), 0.0);
        h.add(1.0);
        h.add(5.0);
        h.add(9.0);
        assert_relative_eq!(h.total_count(), 3.0);
    }

    #[test]
    fn fraction_in_range_partial() {
        let mut h = Histogram::new(0.0, 10.0, 1.0).unwrap();
        // bin centers: 0.5, 1.5, ..., 9.5
        h.add(0.5); // center 0.5
        h.add(5.5); // center 5.5
        h.add(9.5); // center 9.5
                    // Range [0,2] covers bin center 0.5 and 1.5; only 0.5 has a count
        assert_relative_eq!(h.fraction_in_range(0.0, 2.0), 1.0 / 3.0);
        assert_relative_eq!(h.fraction_in_range(0.0, 10.0), 1.0);
    }

    #[test]
    fn fraction_in_range_empty_histogram() {
        let h = Histogram::new(0.0, 10.0, 1.0).unwrap();
        assert_relative_eq!(h.fraction_in_range(0.0, 10.0), 0.0);
    }

    #[test]
    fn merge_histograms() {
        let mut a = Histogram::new(0.0, 10.0, 1.0).unwrap();
        let mut b = Histogram::new(0.0, 10.0, 1.0).unwrap();
        a.add(1.0);
        a.add(5.0);
        b.add(1.0);
        b.add(9.0);
        a.merge(&b).unwrap();
        assert_relative_eq!(a.count(1), 2.0);
        assert_relative_eq!(a.count(5), 1.0);
        assert_relative_eq!(a.count(9), 1.0);
        assert_relative_eq!(a.total_count(), 4.0);
    }

    #[test]
    fn serde_roundtrip() {
        let mut h = Histogram::new(0.0, 10.0, 2.5).unwrap();
        h.add(1.0);
        h.add(5.0);
        let yaml = serde_yml::to_string(&h).unwrap();
        let h2: Histogram = serde_yml::from_str(&yaml).unwrap();
        assert_eq!(h2.num_bins(), h.num_bins());
        assert_relative_eq!(h2.total_count(), h.total_count());
        for i in 0..h.num_bins() {
            assert_relative_eq!(h2.count(i), h.count(i));
        }
    }
}

use super::grid::Grid;
use crate::Point;
use anyhow::Result;
use flate2::{write::GzEncoder, Compression};
use std::io::{BufWriter, Write};
use std::path::Path;

fn open_output(path: &Path) -> Result<Box<dyn Write + Send>> {
    let file = std::fs::File::create(path)
        .map_err(|err| anyhow::anyhow!("Error creating file {path:?}: {err}"))?;
    if path.extension().is_some_and(|ext| ext == "gz") {
        Ok(Box::new(GzEncoder::new(file, Compression::default())))
    } else {
        Ok(Box::new(file))
    }
}

pub(super) fn write(path: &Path, grid: &Grid, values: &[f64], label: &str) -> Result<()> {
    anyhow::ensure!(
        values.len() == grid.num_voxels(),
        "SpatialDistribution: grid has {} voxels but {} values",
        grid.num_voxels(),
        values.len()
    );

    let mut out = BufWriter::new(open_output(path)?);
    write_to(&mut out, grid, values, label)?;
    out.flush()?;
    Ok(())
}

fn write_point(out: &mut impl Write, keyword: &str, point: Point) -> Result<()> {
    writeln!(
        out,
        "{keyword} {:.8} {:.8} {:.8}",
        point.x, point.y, point.z
    )?;
    Ok(())
}

pub(super) fn write_to(
    out: &mut impl Write,
    grid: &Grid,
    values: &[f64],
    label: &str,
) -> Result<()> {
    let [nx, ny, nz] = grid.dims();
    writeln!(out, "# Faunus spatial distribution function ({label})")?;
    writeln!(out, "object 1 class gridpositions counts {nx} {ny} {nz}")?;
    write_point(out, "origin", grid.origin())?;
    write_point(out, "delta", Point::new(grid.spacing(), 0.0, 0.0))?;
    write_point(out, "delta", Point::new(0.0, grid.spacing(), 0.0))?;
    write_point(out, "delta", Point::new(0.0, 0.0, grid.spacing()))?;
    writeln!(out, "object 2 class gridconnections counts {nx} {ny} {nz}")?;
    writeln!(
        out,
        "object 3 class array type double rank 0 items {} data follows",
        values.len()
    )?;
    for chunk in values.chunks(3) {
        for value in chunk {
            write!(out, " {:.8e}", value)?;
        }
        writeln!(out)?;
    }
    writeln!(out, "attribute \"dep\" string \"positions\"")?;
    writeln!(out, "object \"spatial_distribution\" class field")?;
    writeln!(out, "component \"positions\" value 1")?;
    writeln!(out, "component \"connections\" value 2")?;
    writeln!(out, "component \"data\" value 3")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_header_and_x_fastest_data_order() {
        let grid = Grid::from_points(&[Point::new(0.25, 0.25, 0.25)], 0.5, 0.25).unwrap();
        let values: Vec<f64> = (0..grid.num_voxels()).map(|i| i as f64).collect();
        let mut bytes = Vec::new();
        write_to(&mut bytes, &grid, &values, "relative_density").unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("object 1 class gridpositions counts 1 1 1"));
        assert!(text.contains("origin 0.00000000 0.00000000 0.00000000"));
        assert!(text.contains("delta 0.50000000 0.00000000 0.00000000"));
        assert!(text.contains("object 3 class array type double rank 0 items 1 data follows"));
        assert!(text.contains(" 0.00000000e0\n"));
    }
}

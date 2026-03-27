//! Binary frame state file reader and writer for trajectory replay.
//!
//! XTC stores only atom positions. Rigid-body simulations also need
//! quaternions, COMs, and group sizes (for GC moves). Swap moves change
//! `atom_id`. The `.aux` file stores this per-frame microstate data,
//! written alongside each XTC frame.
//!
//! ## Format
//!
//! Pure binary, little-endian. No string keys.
//!
//! **Header** (written once):
//! ```text
//! n_groups:    u32
//! n_particles: u32
//! per group:   molecule_id: u32, capacity: u32
//! ```
//!
//! **Per frame** (appended for each XTC frame):
//! ```text
//! per group:    quaternion [f64; 4], com [f64; 3], size u32
//! per particle: atom_id u32
//! ```

use crate::{Point, UnitQuaternion};
use anyhow::{Context, Result};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Header of a frame state file.
#[derive(Debug)]
pub struct FrameStateHeader {
    pub n_groups: u32,
    pub n_particles: u32,
    /// (molecule_id, capacity) per group
    pub groups: Vec<(u32, u32)>,
}

/// Per-frame data from a frame state file. Reusable across frames via
/// [`FrameStateReader::read_frame_into`].
#[derive(Debug, Default)]
pub struct FrameStateFrame {
    pub quaternions: Vec<UnitQuaternion>,
    pub group_sizes: Vec<u32>,
    pub atom_ids: Vec<u32>,
}

/// Reads frame state files written alongside XTC trajectories.
pub struct FrameStateReader {
    reader: BufReader<std::fs::File>,
    header: FrameStateHeader,
}

impl FrameStateReader {
    /// Open a frame state file and read its header.
    #[cfg(feature = "cli")]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref()).with_context(|| {
            format!("Cannot open frame state file: {}", path.as_ref().display())
        })?;
        let mut reader = BufReader::new(file);
        let header = Self::read_header(&mut reader)?;
        Ok(Self { reader, header })
    }

    /// Reference to the parsed header.
    #[cfg(feature = "cli")]
    pub fn header(&self) -> &FrameStateHeader {
        &self.header
    }

    #[cfg(feature = "cli")]
    fn read_header(reader: &mut BufReader<std::fs::File>) -> Result<FrameStateHeader> {
        let n_groups = read_u32(reader)?;
        let n_particles = read_u32(reader)?;
        let mut groups = Vec::with_capacity(n_groups as usize);
        for _ in 0..n_groups {
            let molecule_id = read_u32(reader)?;
            let capacity = read_u32(reader)?;
            groups.push((molecule_id, capacity));
        }
        Ok(FrameStateHeader {
            n_groups,
            n_particles,
            groups,
        })
    }

    /// Read the next frame into a new allocation. Returns `None` at EOF.
    #[allow(dead_code)]
    pub fn read_frame(&mut self) -> Result<Option<FrameStateFrame>> {
        let mut frame = FrameStateFrame::default();
        if self.read_frame_into(&mut frame)? {
            Ok(Some(frame))
        } else {
            Ok(None)
        }
    }

    /// Read the next frame into `frame`, reusing its buffers. Returns `false` at EOF.
    pub fn read_frame_into(&mut self, frame: &mut FrameStateFrame) -> Result<bool> {
        let n_groups = self.header.n_groups as usize;
        let n_particles = self.header.n_particles as usize;

        frame.quaternions.clear();
        frame.group_sizes.clear();
        frame.atom_ids.clear();

        for i in 0..n_groups {
            // Only treat EOF as end-of-trajectory on the very first read of a frame;
            // EOF mid-frame means the file is truncated/corrupt.
            let w = match read_f64(&mut self.reader) {
                Ok(v) => v,
                Err(e) if i == 0 && is_eof(&e) => return Ok(false),
                Err(e) => return Err(e.context("Truncated frame state frame")),
            };
            let qi = read_f64(&mut self.reader)?;
            let qj = read_f64(&mut self.reader)?;
            let qk = read_f64(&mut self.reader)?;
            // Re-normalize to guard against floating-point drift in the stored quaternion
            let q = nalgebra::Quaternion::new(w, qi, qj, qk);
            frame.quaternions.push(UnitQuaternion::new_normalize(q));

            // Skip COM — recomputed from particle positions on load
            skip_bytes(&mut self.reader, 3 * 8)?;

            frame.group_sizes.push(read_u32(&mut self.reader)?);
        }

        for _ in 0..n_particles {
            frame.atom_ids.push(read_u32(&mut self.reader)?);
        }

        Ok(true)
    }
}

/// Writes frame state data alongside XTC trajectories.
///
/// Uses `BufWriter` internally; data is flushed on drop.
#[derive(Debug)]
pub struct FrameStateWriter {
    writer: BufWriter<std::fs::File>,
}

impl FrameStateWriter {
    /// Create a new writer. Writes the header immediately.
    pub fn create(
        path: impl AsRef<Path>,
        groups: &[(u32, u32)], // (molecule_id, capacity)
        n_particles: u32,
    ) -> Result<Self> {
        let file = std::fs::File::create(path.as_ref()).with_context(|| {
            format!(
                "Cannot create frame state file: {}",
                path.as_ref().display()
            )
        })?;
        let mut writer = BufWriter::new(file);
        write_u32(&mut writer, groups.len() as u32)?;
        write_u32(&mut writer, n_particles)?;
        for &(mol_id, capacity) in groups {
            write_u32(&mut writer, mol_id)?;
            write_u32(&mut writer, capacity)?;
        }
        // Flush header immediately so it's on disk even if the process crashes
        // before any frames are written
        writer.flush()?;
        Ok(Self { writer })
    }

    /// Append one frame of state data.
    pub fn write_frame(
        &mut self,
        quaternions: &[UnitQuaternion],
        mass_centers: &[Point],
        group_sizes: &[u32],
        atom_ids: &[u32],
    ) -> Result<()> {
        for ((q, com), &size) in quaternions.iter().zip(mass_centers).zip(group_sizes) {
            write_f64(&mut self.writer, q.w)?;
            write_f64(&mut self.writer, q.i)?;
            write_f64(&mut self.writer, q.j)?;
            write_f64(&mut self.writer, q.k)?;

            write_f64(&mut self.writer, com.x)?;
            write_f64(&mut self.writer, com.y)?;
            write_f64(&mut self.writer, com.z)?;

            write_u32(&mut self.writer, size)?;
        }
        for &id in atom_ids {
            write_u32(&mut self.writer, id)?;
        }
        Ok(())
    }
}

fn skip_bytes(reader: &mut impl Read, n: usize) -> Result<()> {
    let mut buf = [0u8; 24]; // max 3 * 8 bytes
    reader.read_exact(&mut buf[..n])?;
    Ok(())
}

fn is_eof(err: &anyhow::Error) -> bool {
    err.downcast_ref::<std::io::Error>()
        .is_some_and(|e| e.kind() == std::io::ErrorKind::UnexpectedEof)
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f64(reader: &mut impl Read) -> Result<f64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn write_u32(writer: &mut impl Write, v: u32) -> Result<()> {
    writer.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn write_f64(writer: &mut impl Write, v: f64) -> Result<()> {
    writer.write_all(&v.to_le_bytes())?;
    Ok(())
}

/// Derive the `.aux` path from a trajectory path by replacing the extension.
pub fn aux_path_from_traj(traj_path: &Path) -> PathBuf {
    traj_path.with_extension("aux")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_frame_state() {
        let dir = std::env::temp_dir().join("faunus_test_frame_state.aux");

        let groups = vec![(0u32, 3u32), (1, 5)];
        let n_particles = 8u32;

        let q1 = UnitQuaternion::identity();
        let q2 = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        let com1 = Point::new(1.0, 2.0, 3.0);
        let com2 = Point::new(4.0, 5.0, 6.0);
        let sizes = vec![3u32, 4];
        let atom_ids: Vec<u32> = (0..8).collect();

        // Write two frames
        {
            let mut writer = FrameStateWriter::create(&dir, &groups, n_particles).unwrap();
            writer
                .write_frame(&[q1, q2], &[com1, com2], &sizes, &atom_ids)
                .unwrap();
            writer
                .write_frame(&[q2, q1], &[com2, com1], &[2, 5], &atom_ids)
                .unwrap();
        }

        // Read back with buffer reuse
        let mut reader = FrameStateReader::open(&dir).unwrap();
        let header = reader.header();
        assert_eq!(header.n_groups, 2);
        assert_eq!(header.n_particles, 8);
        assert_eq!(header.groups, groups);

        let mut frame = FrameStateFrame::default();

        assert!(reader.read_frame_into(&mut frame).unwrap());
        assert_eq!(frame.group_sizes, vec![3, 4]);
        assert_eq!(frame.atom_ids, atom_ids);
        assert!((frame.quaternions[0].w - 1.0).abs() < 1e-15);

        assert!(reader.read_frame_into(&mut frame).unwrap());
        assert_eq!(frame.group_sizes, vec![2, 5]);

        assert!(!reader.read_frame_into(&mut frame).unwrap());

        std::fs::remove_file(&dir).ok();
    }
}

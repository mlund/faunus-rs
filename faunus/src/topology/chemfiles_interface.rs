// Copyright 2023 Mikael Lund
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

//! # Inteface to the [`chemfiles`] crate

use std::path::Path;

use crate::{topology, Point};
use chemfiles::Topology;
use itertools::Itertools;

/// Create a new chemfiles::Frame from an input file in a supported format.
pub(crate) fn frame_from_file(filename: &impl AsRef<Path>) -> anyhow::Result<chemfiles::Frame> {
    let mut trajectory = chemfiles::Trajectory::open(filename, 'r')?;
    let mut frame = chemfiles::Frame::new();
    trajectory.read(&mut frame)?;
    Ok(frame)
}

/// Get positions of particles from the chemfiles::Frame.
pub(crate) fn positions_from_frame(frame: &chemfiles::Frame) -> Vec<Point> {
    frame.positions().iter().map(|pos| (*pos).into()).collect()
}

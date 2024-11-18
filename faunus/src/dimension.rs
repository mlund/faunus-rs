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

use serde::{Deserialize, Serialize};

use crate::Point;

/// Represents dimensions in which operations/calculations are performed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Dimension {
    /// No dimension
    None,
    /// X axis
    X,
    /// Y axis
    Y,
    /// Z axis
    Z,
    /// XY plane
    XY,
    /// YZ plane
    YZ,
    /// XZ plane
    XZ,
    /// Volume of the cell
    #[default]
    XYZ,
}

impl From<[bool; 3]> for Dimension {
    /// Convert 3-member boolean array to Dimension enum.
    fn from(arr: [bool; 3]) -> Self {
        match arr {
            [false, false, false] => Dimension::None,
            [true, false, false] => Dimension::X,
            [false, true, false] => Dimension::Y,
            [false, false, true] => Dimension::Z,
            [true, true, false] => Dimension::XY,
            [true, false, true] => Dimension::XZ,
            [false, true, true] => Dimension::YZ,
            [true, true, true] => Dimension::XYZ,
        }
    }
}

impl From<Dimension> for [bool; 3] {
    /// Convert Dimension to a 3-member boolean array.
    fn from(dim: Dimension) -> Self {
        match dim {
            Dimension::None => [false, false, false],
            Dimension::X => [true, false, false],
            Dimension::Y => [false, true, false],
            Dimension::Z => [false, false, true],
            Dimension::XY => [true, true, false],
            Dimension::XZ => [true, false, true],
            Dimension::YZ => [false, true, true],
            Dimension::XYZ => [true, true, true],
        }
    }
}

impl Dimension {
    /// Return `true` if Dimension contains x-dimension.
    pub const fn is_x(&self) -> bool {
        matches!(
            self,
            Dimension::X | Dimension::XY | Dimension::XZ | Dimension::XYZ
        )
    }

    /// Return `true` if Dimension contains y-dimension.
    pub const fn is_y(&self) -> bool {
        matches!(
            self,
            Dimension::Y | Dimension::XY | Dimension::YZ | Dimension::XYZ
        )
    }

    /// Return `true` if Dimension contains z-dimension.
    pub const fn is_z(&self) -> bool {
        matches!(
            self,
            Dimension::Z | Dimension::XZ | Dimension::YZ | Dimension::XYZ
        )
    }

    /// Apply Dimension as a filter for Point. Creates a copy of the Point with
    /// dimensions that do not match the Dimension set to 0.
    pub fn filter(&self, point: Point) -> Point {
        let x = if self.is_x() { point[0] } else { 0.0 };
        let y = if self.is_y() { point[1] } else { 0.0 };
        let z = if self.is_z() { point[2] } else { 0.0 };

        Point::new(x, y, z)
    }
}

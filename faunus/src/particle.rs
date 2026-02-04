use crate::Point;
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
pub trait PointParticle {
    /// Type of the particle identifier
    type Idtype;
    /// Type of the particle position
    type Positiontype;
    /// Identifier for the particle type
    fn atom_id(&self) -> Self::Idtype;
    /// Get position
    fn pos(&self) -> &Self::Positiontype;
    /// Get mutable position
    fn pos_mut(&mut self) -> &mut Self::Positiontype;
    /// Index in main list of particle (immutable)
    fn index(&self) -> usize;
}

#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq)]
pub struct Particle {
    /// Type of the particle (index of the atom kind)
    pub(crate) atom_id: usize,
    /// Index in main list of particles
    pub(crate) index: usize,
    /// Position of the particle
    pub(crate) pos: Point,
}

impl Particle {
    /// Create a new particle.
    ///
    /// # Arguments
    /// * `atom_id` - Index of the atom kind in the topology
    /// * `index` - Index in the main particle list
    /// * `pos` - Position of the particle
    #[must_use]
    pub const fn new(atom_id: usize, index: usize, pos: Point) -> Self {
        Self {
            atom_id,
            index,
            pos,
        }
    }
}

impl PointParticle for Particle {
    type Idtype = usize;
    type Positiontype = Point;
    fn atom_id(&self) -> Self::Idtype {
        self.atom_id
    }
    fn pos(&self) -> &Self::Positiontype {
        &self.pos
    }
    fn pos_mut(&mut self) -> &mut Self::Positiontype {
        &mut self.pos
    }
    fn index(&self) -> usize {
        self.index
    }
}

use nalgebra::Vector3;

pub type Point = Vector3<f64>;
pub type PositionVec = Vec<Point>;
pub type ParticleVec = Vec<Particle>;

pub struct Particle {
    id: usize,
    pos: Point,
}

impl Particle {
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn pos(&self) -> &Point {
        &self.pos
    }
}

#[derive(Default)]
pub struct Group<'a> {
    /// Molecular id
    id: usize,
    /// Optional mass center
    mass_center: Option<Point>,
    /// Active number of particles
    num_active: usize,
    /// Slice of particles matching `range` (active and inactive)
    particles: &'a [Particle],
    /// Indices in main particle vector (active and inactive)
    range: std::ops::Range<usize>,
}

impl<'a> Iterator for Group<'a> {
    type Item = &'a Particle;
    fn next(&mut self) -> Option<Self::Item> {
        self.particles.iter().take(self.num_active).next()
    }
}

impl<'a> Group<'a> {
    /// Molecular id
    pub fn id(&self) -> usize {
        self.id
    }
    /// Maximum number of particles (active + inactive)
    pub fn capacity(&self) -> usize {
        self.particles.len()
    }
    /// Range of _indices_ in full particle vector (active particles, only)
    pub fn indices(&self) -> std::ops::Range<usize> {
        std::ops::Range {
            start: self.range.start,
            end: self.range.start + self.num_active,
        }
    }
    /// Iterator to all _inactive particles_
    pub fn inactive(&self) -> impl Iterator<Item = &'a Particle> {
        self.particles.iter().skip(self.num_active)
    }
    /// Center of mass of the group
    pub fn mass_center(&self) -> Option<&Point> {
        self.mass_center.as_ref()
    }
}

/// Collection of particles
///
/// # Examples
///
/// ~~~
/// use particles::ParticleCollection;
/// let mut state = ParticleCollection::default();
/// for (ids, pos) in std::iter::zip(state.ids(0..0), state.positions(0..0)) {
/// }
/// ~~~
#[derive(Default)]
pub struct ParticleCollection {
    positions: PositionVec,
    ids: Vec<usize>,
}

impl ParticleCollection {
    pub fn positions(&self, range: std::ops::Range<usize>) -> &[Point] {
        self.positions[range].as_ref()
    }

    pub fn positions_mut(&mut self, range: std::ops::Range<usize>) -> &mut [Point] {
        self.positions[range].as_mut()
    }

    pub fn ids(&self, range: std::ops::Range<usize>) -> &[usize] {
        self.ids[range].as_ref()
    }

    pub fn ids_mut(&mut self, range: std::ops::Range<usize>) -> &mut [usize] {
        self.ids[range].as_mut()
    }

    // pub fn particles(&self) -> &[Particle] {
    //     self.particles.as_ref()
    // }
    // pub fn group_particles(&self, group: &Group) -> Option<&[Particle]> {
    //     Some(&self.particles[group.begin..group.end])
    // }
}

/*
for (pos, q, id) in zip(state.positions, state.charges, state.id)[water] {}

state.particles[water]

 */

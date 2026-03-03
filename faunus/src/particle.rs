use crate::Point;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Particle {
    /// Type of the particle (index of the atom kind)
    pub(crate) atom_id: usize,
    /// Position of the particle
    pub(crate) pos: Point,
}

impl Particle {
    #[must_use]
    pub const fn new(atom_id: usize, pos: Point) -> Self {
        Self { atom_id, pos }
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
}

impl Serialize for Particle {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Particle", 2)?;
        s.serialize_field("atom_id", &self.atom_id)?;
        s.serialize_field("pos", &self.pos)?;
        s.end()
    }
}

/// Backward-compatible deserialization: ignores the legacy `index` field if present.
impl<'de> Deserialize<'de> for Particle {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Helper {
            atom_id: usize,
            #[serde(default)]
            #[allow(dead_code)]
            index: Option<usize>,
            pos: Point,
        }
        let h = Helper::deserialize(deserializer)?;
        Ok(Particle {
            atom_id: h.atom_id,
            pos: h.pos,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_legacy_state_with_index() {
        let yaml = "atom_id: 0\nindex: 42\npos:\n- 1.0\n- 2.0\n- 3.0\n";
        let p: Particle = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(p.atom_id, 0);
        assert_eq!(p.pos, Point::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn deserialize_new_format_without_index() {
        let yaml = "atom_id: 1\npos:\n- 4.0\n- 5.0\n- 6.0\n";
        let p: Particle = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(p.atom_id, 1);
        assert_eq!(p.pos, Point::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn roundtrip_serialization() {
        let p = Particle::new(3, Point::new(7.0, 8.0, 9.0));
        let yaml = serde_yaml::to_string(&p).unwrap();
        assert!(!yaml.contains("index"));
        let p2: Particle = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(p, p2);
    }
}

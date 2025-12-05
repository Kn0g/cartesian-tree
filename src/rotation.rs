use nalgebra::{Quaternion, UnitQuaternion, Vector3};

/// Unified representation for rotations, allowing different input formats.
#[derive(Clone, Copy, Debug)]
pub enum Rotation {
    /// Quaternion representation (x, y, z, w).
    Quaternion(UnitQuaternion<f64>),
    /// Roll-Pitch-Yaw (Euler angles in radians, ZYX convention).
    Rpy(Vector3<f64>),
}

impl Rotation {
    /// Creates a Rotation from a quaternion (x, y, z, w).
    #[must_use]
    pub fn from_quaternion(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self::Quaternion(UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z)))
    }

    /// Creates a Rotation from RPY angles in radians (roll, pitch, yaw).
    #[must_use]
    pub const fn from_rpy(roll: f64, pitch: f64, yaw: f64) -> Self {
        Self::Rpy(Vector3::new(roll, pitch, yaw))
    }

    /// Creates the identity rotation using the identity quaternion.
    #[must_use]
    pub fn identity() -> Self {
        Self::Quaternion(UnitQuaternion::identity())
    }

    /// Converts this rotation to a `UnitQuaternion`.
    #[must_use]
    pub fn as_quaternion(&self) -> UnitQuaternion<f64> {
        match self {
            Self::Quaternion(q) => *q,
            Self::Rpy(rpy) => UnitQuaternion::from_euler_angles(rpy.x, rpy.y, rpy.z),
        }
    }

    /// Converts to RPY (roll, pitch, yaw) in radians.
    #[must_use]
    pub fn as_rpy(&self) -> Vector3<f64> {
        match self {
            Self::Quaternion(q) => {
                let (roll, pitch, yaw) = UnitQuaternion::euler_angles(q);
                Vector3::new(roll, pitch, yaw)
            }
            Self::Rpy(rpy) => *rpy,
        }
    }
}

impl From<UnitQuaternion<f64>> for Rotation {
    fn from(q: UnitQuaternion<f64>) -> Self {
        Self::Quaternion(q)
    }
}

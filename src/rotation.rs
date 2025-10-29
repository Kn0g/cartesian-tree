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
    pub fn from_quat(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self::Quaternion(UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z)))
    }

    /// Creates a Rotation from RPY angles in radians (roll, pitch, yaw).
    #[must_use]
    pub const fn from_rpy(roll: f64, pitch: f64, yaw: f64) -> Self {
        Self::Rpy(Vector3::new(roll, pitch, yaw))
    }

    /// Converts this rotation to a `UnitQuaternion`.
    #[must_use]
    pub fn to_quat(&self) -> UnitQuaternion<f64> {
        match self {
            Self::Quaternion(q) => *q,
            Self::Rpy(rpy) => UnitQuaternion::from_euler_angles(rpy.x, rpy.y, rpy.z),
        }
    }

    /// Converts to RPY (roll, pitch, yaw) in radians.
    #[must_use]
    pub fn to_rpy(&self) -> Vector3<f64> {
        let rpy = self.to_quat().euler_angles();
        Vector3::new(rpy.0, rpy.1, rpy.2)
    }
}

impl From<UnitQuaternion<f64>> for Rotation {
    fn from(q: UnitQuaternion<f64>) -> Self {
        Self::Quaternion(q)
    }
}

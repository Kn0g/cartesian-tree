use crate::CartesianTreeError;
use nalgebra::{Quaternion, UnitQuaternion, Vector3};

/// Minimum norm below which a quaternion is considered degenerate.
pub(crate) const MIN_QUATERNION_NORM: f64 = 1.0e-9;

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
    ///
    /// The quaternion does not need to be normalized; it is normalized internally.
    ///
    /// # Errors
    /// Returns a [`CartesianTreeError`] if:
    /// - The quaternion's norm is too close to zero to normalize.
    pub fn from_quaternion(x: f64, y: f64, z: f64, w: f64) -> Result<Self, CartesianTreeError> {
        UnitQuaternion::try_new(Quaternion::new(w, x, y, z), MIN_QUATERNION_NORM)
            .map(Self::Quaternion)
            .ok_or(CartesianTreeError::InvalidQuaternion(x, y, z, w))
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn from_quaternion_normalizes_input() {
        let rotation = Rotation::from_quaternion(0.0, 0.0, 2.0, 0.0).unwrap();
        let q = rotation.as_quaternion();
        assert_relative_eq!(q.k, 1.0, epsilon = 1e-12);
        assert_relative_eq!(q.w, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn from_quaternion_rejects_zero_norm() {
        assert!(matches!(
            Rotation::from_quaternion(0.0, 0.0, 0.0, 0.0),
            Err(CartesianTreeError::InvalidQuaternion(..))
        ));
    }
}

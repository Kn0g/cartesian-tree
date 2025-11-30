use nalgebra::{Isometry3, Translation3};

#[derive(Clone)]
pub struct LazyTranslation {
    pub(crate) inner: Isometry3<f64>,
}

#[derive(Clone)]
pub struct LazyRotation {
    pub(crate) inner: Isometry3<f64>,
}

#[must_use]
pub fn x(value: f64) -> LazyTranslation {
    LazyTranslation {
        inner: Isometry3::from_parts(
            Translation3::new(value, 0.0, 0.0),
            nalgebra::UnitQuaternion::identity(),
        ),
    }
}

#[must_use]
pub fn y(value: f64) -> LazyTranslation {
    LazyTranslation {
        inner: Isometry3::from_parts(
            Translation3::new(0.0, value, 0.0),
            nalgebra::UnitQuaternion::identity(),
        ),
    }
}

#[must_use]
pub fn z(value: f64) -> LazyTranslation {
    LazyTranslation {
        inner: Isometry3::from_parts(
            Translation3::new(0.0, 0.0, value),
            nalgebra::UnitQuaternion::identity(),
        ),
    }
}

#[must_use]
pub fn rx(value: f64) -> LazyRotation {
    LazyRotation {
        inner: Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 0.0),
            nalgebra::UnitQuaternion::from_euler_angles(value, 0.0, 0.0),
        ),
    }
}

#[must_use]
pub fn ry(value: f64) -> LazyRotation {
    LazyRotation {
        inner: Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 0.0),
            nalgebra::UnitQuaternion::from_euler_angles(0.0, value, 0.0),
        ),
    }
}

#[must_use]
pub fn rz(value: f64) -> LazyRotation {
    LazyRotation {
        inner: Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 0.0),
            nalgebra::UnitQuaternion::from_euler_angles(0.0, 0.0, value),
        ),
    }
}

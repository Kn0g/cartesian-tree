use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::CartesianTreeError;
use crate::rotation::Rotation;

impl From<CartesianTreeError> for PyErr {
    fn from(err: CartesianTreeError) -> Self {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

#[pyclass(name = "Rotation", unsendable)]
#[derive(Clone, Copy, Debug)]
pub struct PyRotation {
    pub rust_rotation: Rotation,
}

#[pymethods]
impl PyRotation {
    #[classmethod]
    fn from_quaternion(_cls: &Bound<'_, PyType>, x: f64, y: f64, z: f64, w: f64) -> Self {
        Self {
            rust_rotation: Rotation::from_quaternion(x, y, z, w),
        }
    }

    #[classmethod]
    const fn from_rpy(_cls: &Bound<'_, PyType>, roll: f64, pitch: f64, yaw: f64) -> Self {
        Self {
            rust_rotation: Rotation::from_rpy(roll, pitch, yaw),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn as_quaternion(&self) -> (f64, f64, f64, f64) {
        let quat = self.rust_rotation.as_quaternion();
        (quat.coords.x, quat.coords.y, quat.coords.z, quat.coords.w)
    }

    #[allow(clippy::wrong_self_convention)]
    fn as_rpy(&self) -> (f64, f64, f64) {
        let rpy = self.rust_rotation.as_rpy();
        (rpy.x, rpy.y, rpy.z)
    }

    fn __str__(&self) -> String {
        match &self.rust_rotation {
            Rotation::Quaternion(q) => {
                format!(
                    "Quaternion(<{:.4}, {:.4}, {:.4}>, {:.4})",
                    q.i, q.j, q.k, q.w
                )
            }
            Rotation::Rpy(rpy) => format!("RPY({:.4}, {:.4}, {:.4})", rpy.x, rpy.y, rpy.z),
        }
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

#[pyclass(name = "Vector3", unsendable)]
#[derive(Clone, Copy, Debug)]
pub struct PyVector3 {
    pub inner: Vector3<f64>,
}

#[pymethods]
impl PyVector3 {
    #[new]
    const fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            inner: Vector3::new(x, y, z),
        }
    }

    #[getter]
    fn x(&self) -> f64 {
        self.inner.x
    }

    #[getter]
    fn y(&self) -> f64 {
        self.inner.y
    }

    #[getter]
    fn z(&self) -> f64 {
        self.inner.z
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_tuple(&self) -> (f64, f64, f64) {
        (self.inner.x, self.inner.y, self.inner.z)
    }

    fn __str__(&self) -> String {
        format!("({:.4}, {:.4}, {:.4})", self.x(), self.y(), self.z())
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

#[pyclass(name = "Isometry", unsendable)]
#[derive(Clone, Copy, Debug)]
pub struct PyIsometry {
    pub inner: Isometry3<f64>,
}

#[pymethods]
impl PyIsometry {
    #[classmethod]
    fn identity(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: Isometry3::identity(),
        }
    }

    #[classmethod]
    fn from_translation(_cls: &Bound<'_, PyType>, translation: PyVector3) -> Self {
        Self {
            inner: Isometry3::from_parts(
                Translation3::from(translation.inner),
                UnitQuaternion::identity(),
            ),
        }
    }

    #[classmethod]
    fn from_rotation(_cls: &Bound<'_, PyType>, rotation: PyRotation) -> Self {
        Self {
            inner: Isometry3::from_parts(
                Translation3::new(0.0, 0.0, 0.0),
                rotation.rust_rotation.as_quaternion(),
            ),
        }
    }

    #[classmethod]
    fn from_parts(_cls: &Bound<'_, PyType>, translation: PyVector3, rotation: PyRotation) -> Self {
        Self {
            inner: Isometry3::from_parts(
                Translation3::from(translation.inner),
                rotation.rust_rotation.as_quaternion(),
            ),
        }
    }

    const fn decompose(&self) -> (PyVector3, PyRotation) {
        (
            PyVector3 {
                inner: self.inner.translation.vector,
            },
            PyRotation {
                rust_rotation: Rotation::Quaternion(self.inner.rotation),
            },
        )
    }

    const fn translation(&self) -> PyVector3 {
        PyVector3 {
            inner: self.inner.translation.vector,
        }
    }

    const fn rotation(&self) -> PyRotation {
        PyRotation {
            rust_rotation: Rotation::Quaternion(self.inner.rotation),
        }
    }

    fn inverse(&self) -> Self {
        Self {
            inner: self.inner.inverse(),
        }
    }

    fn __mul__(&self, other: &Self) -> Self {
        Self {
            inner: self.inner * other.inner,
        }
    }

    fn __str__(&self) -> String {
        let (translation, rotation) = self.decompose();
        format!(
            "Isometry(translation: {}, rotation: {})",
            translation.__str__(),
            rotation.__str__()
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

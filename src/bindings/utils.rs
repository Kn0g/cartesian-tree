use nalgebra::Vector3;
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

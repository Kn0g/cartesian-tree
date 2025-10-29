use nalgebra::{UnitQuaternion, Vector3};
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::CartesianTreeError;
use crate::rotation::Rotation;

impl From<CartesianTreeError> for PyErr {
    fn from(err: CartesianTreeError) -> Self {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

#[pyclass(name = "RPY", unsendable)]
#[derive(Clone, Copy, Debug)]
pub struct PyRPY {
    pub rpy: Vector3<f64>,
}

#[pymethods]
impl PyRPY {
    #[new]
    const fn new(roll: f64, pitch: f64, yaw: f64) -> Self {
        Self {
            rpy: Vector3::new(roll, pitch, yaw),
        }
    }

    #[getter]
    fn roll(&self) -> f64 {
        self.rpy.x
    }

    #[getter]
    fn pitch(&self) -> f64 {
        self.rpy.y
    }

    #[getter]
    fn yaw(&self) -> f64 {
        self.rpy.z
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_rotation(&self) -> PyRotation {
        PyRotation {
            rust_rotation: Rotation::from_rpy(self.rpy.x, self.rpy.y, self.rpy.z),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_tuple(&self) -> (f64, f64, f64) {
        (self.rpy.x, self.rpy.y, self.rpy.z)
    }

    fn __str__(&self) -> String {
        format!(
            "({:.4}, {:.4}, {:.4})",
            self.roll(),
            self.pitch(),
            self.yaw(),
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

#[pyclass(name = "Quaternion", unsendable)]
#[derive(Clone, Copy, Debug)]
pub struct PyQuaternion {
    pub quat: UnitQuaternion<f64>,
}

#[pymethods]
impl PyQuaternion {
    #[new]
    fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self {
            quat: UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(w, x, y, z)),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_rotation(&self) -> PyRotation {
        PyRotation {
            rust_rotation: Rotation::from_quat(
                self.quat.coords.x,
                self.quat.coords.y,
                self.quat.coords.z,
                self.quat.coords.w,
            ),
        }
    }

    #[getter]
    fn x(&self) -> f64 {
        self.quat.coords.x
    }

    #[getter]
    fn y(&self) -> f64 {
        self.quat.coords.y
    }

    #[getter]
    fn z(&self) -> f64 {
        self.quat.coords.z
    }

    #[getter]
    fn w(&self) -> f64 {
        self.quat.coords.w
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_tuple(&self) -> (f64, f64, f64, f64) {
        (
            self.quat.coords.x,
            self.quat.coords.y,
            self.quat.coords.z,
            self.quat.coords.w,
        )
    }

    fn __str__(&self) -> String {
        format!(
            "(<{:.4}, {:.4}, {:.4}>, {:.4})",
            self.x(),
            self.y(),
            self.z(),
            self.w()
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
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
    fn from_quat(_cls: &Bound<'_, PyType>, x: f64, y: f64, z: f64, w: f64) -> Self {
        Self {
            rust_rotation: Rotation::from_quat(x, y, z, w),
        }
    }

    #[classmethod]
    const fn from_rpy(_cls: &Bound<'_, PyType>, roll: f64, pitch: f64, yaw: f64) -> Self {
        Self {
            rust_rotation: Rotation::from_rpy(roll, pitch, yaw),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_quat(&self) -> PyQuaternion {
        let quat = self.rust_rotation.to_quat();
        PyQuaternion { quat }
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_rpy(&self) -> PyRPY {
        let rpy = self.rust_rotation.to_rpy();
        PyRPY { rpy }
    }

    fn __str__(&self) -> String {
        match &self.rust_rotation {
            Rotation::Quaternion(q) => {
                format!("Quaternion({:.4}, {:.4}, {:.4}, {:.4})", q.i, q.j, q.k, q.w)
            }
            Rotation::Rpy(rpy) => format!("RPY({:.4}, {:.4}, {:.4})", rpy.x, rpy.y, rpy.z),
        }
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

#[pyclass(name = "Position", unsendable)]
#[derive(Clone, Copy, Debug)]
pub struct PyPosition {
    pub position: Vector3<f64>,
}

#[pymethods]
impl PyPosition {
    #[new]
    const fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            position: Vector3::new(x, y, z),
        }
    }

    #[getter]
    fn x(&self) -> f64 {
        self.position.x
    }

    #[getter]
    fn y(&self) -> f64 {
        self.position.y
    }

    #[getter]
    fn z(&self) -> f64 {
        self.position.z
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_tuple(&self) -> (f64, f64, f64) {
        (self.position.x, self.position.y, self.position.z)
    }

    fn __str__(&self) -> String {
        format!("({:.4}, {:.4}, {:.4})", self.x(), self.y(), self.z())
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

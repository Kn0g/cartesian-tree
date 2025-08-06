use nalgebra::{UnitQuaternion, Vector3};
use pyo3::prelude::*;

use crate::CartesianTreeError;

impl From<CartesianTreeError> for PyErr {
    fn from(err: CartesianTreeError) -> PyErr {
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
    fn new(roll: f64, pitch: f64, yaw: f64) -> Self {
        PyRPY {
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
    fn to_quaternion(&self) -> PyQuaternion {
        PyQuaternion {
            quat: UnitQuaternion::from_euler_angles(self.rpy.x, self.rpy.y, self.rpy.z),
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
        format!(
            "({:.4}, {:.4}, {:.4})",
            self.roll(),
            self.pitch(),
            self.yaw(),
        )
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
        PyQuaternion {
            quat: UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(w, x, y, z)),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_rpy(&self) -> PyRPY {
        let (roll, pitch, yaw) = self.quat.euler_angles();
        PyRPY::new(roll, pitch, yaw)
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
            "({:.4}, {:.4}, {:.4}, {:.4})",
            self.x(),
            self.y(),
            self.z(),
            self.w()
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "({:.4}, {:.4}, {:.4}, {:.4})",
            self.x(),
            self.y(),
            self.z(),
            self.w()
        )
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
    fn new(x: f64, y: f64, z: f64) -> Self {
        PyPosition {
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
        format!("({:.4}, {:.4}, {:.4})", self.x(), self.y(), self.z())
    }
}

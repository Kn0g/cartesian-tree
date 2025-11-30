use pyo3::prelude::*;

use crate::lazy_access::{LazyRotation, LazyTranslation};

use crate::lazy_access::rx as rust_rx;
use crate::lazy_access::ry as rust_ry;
use crate::lazy_access::rz as rust_rz;
use crate::lazy_access::x as rust_x;
use crate::lazy_access::y as rust_y;
use crate::lazy_access::z as rust_z;

#[pyclass(name = "LazyTranslation", unsendable)]
#[derive(Clone)]
pub struct PyLazyTranslation {
    pub(crate) inner: LazyTranslation,
}
#[pyclass(name = "LazyRotation", unsendable)]
#[derive(Clone)]
pub struct PyLazyRotation {
    pub(crate) inner: LazyRotation,
}

#[must_use]
#[pyfunction]
pub fn x(value: f64) -> PyLazyTranslation {
    PyLazyTranslation {
        inner: rust_x(value),
    }
}

#[must_use]
#[pyfunction]
pub fn y(value: f64) -> PyLazyTranslation {
    PyLazyTranslation {
        inner: rust_y(value),
    }
}

#[must_use]
#[pyfunction]
pub fn z(value: f64) -> PyLazyTranslation {
    PyLazyTranslation {
        inner: rust_z(value),
    }
}

#[must_use]
#[pyfunction]
pub fn rx(value: f64) -> PyLazyRotation {
    PyLazyRotation {
        inner: rust_rx(value),
    }
}

#[must_use]
#[pyfunction]
pub fn ry(value: f64) -> PyLazyRotation {
    PyLazyRotation {
        inner: rust_ry(value),
    }
}

#[must_use]
#[pyfunction]
pub fn rz(value: f64) -> PyLazyRotation {
    PyLazyRotation {
        inner: rust_rz(value),
    }
}

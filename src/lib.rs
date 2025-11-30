//! Cartesian Tree Library
//!
//! This crate provides a tree-based coordinate system where each frame has a position
//! and orientation relative to its parent. You can create hierarchical transformations
//! and convert poses between frames.

pub mod errors;
pub mod frame;
pub mod lazy_access;
pub mod pose;
pub mod rotation;

pub mod tree;
pub use errors::CartesianTreeError;
pub use frame::Frame;
pub use pose::Pose;

// The bindings module and the PyO3 initialization are only compiled when the
// "bindings" feature is enabled.
#[cfg(feature = "bindings")]
pub mod bindings;
#[cfg(feature = "bindings")]
use pyo3::prelude::*;

#[cfg(feature = "bindings")]
#[pymodule]
#[pyo3(name = "_cartesian_tree")]
fn cartesian_tree(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<bindings::frame::PyFrame>()?;
    m.add_class::<bindings::pose::PyPose>()?;
    m.add_class::<bindings::utils::PyVector3>()?;
    m.add_class::<bindings::utils::PyRotation>()?;
    m.add_class::<bindings::utils::PyIsometry>()?;
    m.add_class::<bindings::lazy_access::PyLazyTranslation>()?;
    m.add_class::<bindings::lazy_access::PyLazyRotation>()?;
    m.add_function(wrap_pyfunction!(bindings::lazy_access::x, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::lazy_access::y, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::lazy_access::z, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::lazy_access::rx, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::lazy_access::ry, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::lazy_access::rz, m)?)?;
    Ok(())
}

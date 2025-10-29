//! Cartesian Tree Library
//!
//! This crate provides a tree-based coordinate system where each frame has a position
//! and orientation relative to its parent. You can create hierarchical transformations
//! and convert poses between frames.

pub mod errors;
pub mod frame;
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
    Ok(())
}

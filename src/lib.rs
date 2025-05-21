//! Cartesian Tree Library
//!
//! This crate provides a tree-based coordinate system where each frame has a position
//! and orientation relative to its parent. You can create hierarchical transformations
//! and convert poses between frames.

mod frame;
mod orientation;

pub use frame::{Frame, FrameExt, FrameRef};

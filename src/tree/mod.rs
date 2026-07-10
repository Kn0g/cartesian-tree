//! Generic traits for walking tree-like structures (parent/child access, depth, root, LCA).

pub mod traits;

pub use traits::{HasChildren, HasParent, NodeEquality, Walking};

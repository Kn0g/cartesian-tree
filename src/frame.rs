use crate::orientation::IntoOrientation;
use nalgebra::{Isometry3, Translation3, Vector3};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// A shared, mutable reference to a [`Frame`].
pub type FrameRef = Rc<RefCell<Frame>>;

/// Represents a coordinate frame in a Cartesian tree structure.
///
/// Each frame can have one parent and multiple children. The frame stores its
/// transformation (position and orientation) relative to its parent.
///
/// Root frames (created via `Frame::new_origin`) have no parent and use the identity transform.
pub struct Frame {
    /// The name of the frame (must be unique among siblings).
    name: String,
    /// Reference to the parent frame.
    parent: Option<Weak<RefCell<Frame>>>,
    /// Transformation from this frame to its parent frame.
    transform_to_parent: Isometry3<f64>,
    /// Child frames directly connected to this frame.
    children: Vec<FrameRef>,
}

impl Frame {
    /// Creates a new root frame (origin) with the given name.
    ///
    /// The origin has no parent and uses the identity transform.
    /// # Arguments
    /// - `name`: The name of the root frame.
    ///
    /// # Example
    /// ```
    /// use cartesian_tree::Frame;
    ///
    /// let origin = Frame::new_origin("world");
    /// ```
    pub fn new_origin(name: impl Into<String>) -> FrameRef {
        Rc::new(RefCell::new(Frame {
            name: name.into(),
            parent: None,
            transform_to_parent: Isometry3::identity(),
            children: Vec::new(),
        }))
    }
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Frame")
            .field("name", &self.name)
            .field("transform_to_parent", &self.transform_to_parent)
            .field("children", &self.children.len())
            .finish()
    }
}

/// Extension trait for [`FrameRef`] to allow adding child frames.
///
/// This trait enables ergonomic methods on shared frame references,
/// such as `add_child(...)`, which adds a new frame as a child of the current one.
pub trait FrameExt {
    /// Adds a new child frame to the current frame.
    ///
    /// The child is positioned and oriented relative to this frame.
    ///
    /// Returns an error if a child with the same name already exists.
    ///
    /// # Arguments
    /// - `name`: The name of the new child frame.
    /// - `position`: A 3D vector representing the translational offset from the parent.
    /// - `orientation`: An orientation convertible into a unit quaternion.
    ///
    /// # Example
    /// ```
    /// use cartesian_tree::{Frame, FrameExt};
    /// use nalgebra::{Vector3, UnitQuaternion};
    ///
    /// let root = Frame::new_origin("base");
    /// let child = root
    ///     .add_child("camera", Vector3::new(0.0, 0.0, 1.0), UnitQuaternion::identity())
    ///     .unwrap();
    /// ```
    fn add_child<O>(
        &self,
        name: impl Into<String>,
        position: Vector3<f64>,
        orientation: O,
    ) -> Result<FrameRef, String>
    where
        O: IntoOrientation;
}

impl FrameExt for FrameRef {
    fn add_child<O>(
        &self,
        name: impl Into<String>,
        position: Vector3<f64>,
        orientation: O,
    ) -> Result<FrameRef, String>
    where
        O: IntoOrientation,
    {
        let child_name = name.into();
        {
            let frame = self.borrow();
            if frame
                .children
                .iter()
                .any(|child| child.borrow().name == child_name)
            {
                return Err(format!(
                    "A child with name '{}' already exists!",
                    child_name
                ));
            }
        }
        let quat = orientation.into_orientation();
        let transform = Isometry3::from_parts(Translation3::from(position), quat);

        let child = Rc::new(RefCell::new(Frame {
            name: child_name,
            parent: Some(Rc::downgrade(self)),
            transform_to_parent: transform,
            children: Vec::new(),
        }));

        self.borrow_mut().children.push(Rc::clone(&child));
        Ok(child)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    #[test]
    fn create_origin_frame() {
        let root = Frame::new_origin("world");
        let root_borrow = root.borrow();
        assert_eq!(root_borrow.name, "world");
        assert!(root_borrow.parent.is_none());
        assert_eq!(root_borrow.children.len(), 0);
    }

    #[test]
    fn add_child_frame_with_quaternion() {
        let root = Frame::new_origin("world");
        let child = root
            .add_child(
                "dummy",
                Vector3::new(1.0, 0.0, 0.0),
                UnitQuaternion::identity(),
            )
            .unwrap();

        let root_borrow = root.borrow();
        assert_eq!(root_borrow.children.len(), 1);

        let child_borrow = child.borrow();
        assert_eq!(child_borrow.name, "dummy");
        assert!(child_borrow.parent.is_some());

        let parent_name = child_borrow
            .parent
            .as_ref()
            .unwrap()
            .upgrade()
            .unwrap()
            .borrow()
            .name
            .clone();
        assert_eq!(parent_name, "world");
    }

    #[test]
    fn add_child_frame_with_rpy() {
        let root = Frame::new_origin("world");
        let child = root
            .add_child(
                "dummy",
                Vector3::new(0.0, 1.0, 0.0),
                (0.0, 0.0, std::f64::consts::FRAC_PI_2),
            )
            .unwrap();

        let child_borrow = child.borrow();
        assert_eq!(child_borrow.name, "dummy");

        let rotation = child_borrow.transform_to_parent.rotation;
        let expected = UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2);
        assert!((rotation.angle() - expected.angle()).abs() < 1e-10);
    }

    #[test]
    fn multiple_child_frames() {
        let root = Frame::new_origin("world");

        let a = root
            .add_child("a", Vector3::new(1.0, 0.0, 0.0), UnitQuaternion::identity())
            .unwrap();
        let b = root
            .add_child("b", Vector3::new(0.0, 1.0, 0.0), UnitQuaternion::identity())
            .unwrap();

        let root_borrow = root.borrow();
        assert_eq!(root_borrow.children.len(), 2);

        let a_borrow = a.borrow();
        let b_borrow = b.borrow();

        assert_eq!(
            a_borrow
                .parent
                .as_ref()
                .unwrap()
                .upgrade()
                .unwrap()
                .borrow()
                .name,
            "world"
        );
        assert_eq!(
            b_borrow
                .parent
                .as_ref()
                .unwrap()
                .upgrade()
                .unwrap()
                .borrow()
                .name,
            "world"
        );
    }

    #[test]
    fn reject_duplicate_child_name() {
        let root = Frame::new_origin("world");

        let _ = root
            .add_child(
                "duplicate",
                Vector3::new(1.0, 0.0, 0.0),
                UnitQuaternion::identity(),
            )
            .unwrap();

        let result = root.add_child(
            "duplicate",
            Vector3::new(2.0, 0.0, 0.0),
            UnitQuaternion::identity(),
        );
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "A child with name 'duplicate' already exists!"
        );
    }

    #[test]
    #[should_panic(expected = "already borrowed")]
    fn test_borrow_conflict() {
        let frame = Frame::new_origin("root");
        let _borrow = frame.borrow(); // Immutable borrow
        frame.borrow_mut(); // Should panic
    }
}

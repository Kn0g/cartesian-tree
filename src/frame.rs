use crate::CartesianTreeError;
use crate::Pose;
use crate::orientation::IntoOrientation;
use crate::tree::{HasChildren, HasParent, NodeEquality};

use nalgebra::{Isometry3, Translation3, Vector3};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// Represents a coordinate frame in a Cartesian tree structure.
///
/// Each frame can have one parent and multiple children. The frame stores its
/// transformation (position and orientation) relative to its parent.
///
/// Root frames (created via `Frame::new_origin`) have no parent and use the identity transform.
#[derive(Clone, Debug)]
pub struct Frame {
    pub(crate) data: Rc<RefCell<FrameData>>,
}

#[derive(Debug)]
pub(crate) struct FrameData {
    /// The name of the frame (must be unique among siblings).
    pub(crate) name: String,
    /// Reference to the parent frame.
    parent: Option<Weak<RefCell<FrameData>>>,
    /// Transformation from this frame to its parent frame.
    transform_to_parent: Isometry3<f64>,
    /// Child frames directly connected to this frame.
    children: Vec<Frame>,
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
    pub fn new_origin(name: impl Into<String>) -> Self {
        Frame {
            data: Rc::new(RefCell::new(FrameData {
                name: name.into(),
                parent: None,
                children: Vec::new(),
                transform_to_parent: Isometry3::identity(),
            })),
        }
    }

    pub(crate) fn borrow(&self) -> std::cell::Ref<'_, FrameData> {
        self.data.borrow()
    }

    fn borrow_mut(&self) -> std::cell::RefMut<'_, FrameData> {
        self.data.borrow_mut()
    }

    pub(crate) fn downgrade(&self) -> Weak<RefCell<FrameData>> {
        Rc::downgrade(&self.data)
    }

    pub(crate) fn walk_up_and_transform(
        &self,
        target: &Frame,
    ) -> Result<Isometry3<f64>, CartesianTreeError> {
        let mut transform = Isometry3::identity();
        let mut current = self.clone();

        while !current.is_same(target) {
            let transform_to_its_parent = {
                // Scope borrow
                let current_data = current.borrow();

                // If current frame is root and not target, then target is not an ancestor.
                if current_data.parent.is_none() {
                    return Err(CartesianTreeError::IsNoAncestor(target.name(), self.name()));
                }
                current_data.transform_to_parent
            };

            transform = transform_to_its_parent * transform;

            let parent_frame_opt = current.parent();
            current = parent_frame_opt
                .ok_or_else(|| CartesianTreeError::IsNoAncestor(target.name(), self.name()))?;
        }

        Ok(transform)
    }

    /// Returns the name of the frame.
    pub fn name(&self) -> String {
        self.borrow().name.clone()
    }

    /// Returns the transformation from this frame to its parent frame.
    ///
    /// # Returns
    /// - `Ok(Isometry3<f64>)` if the frame has a parent.
    /// - `Err(String)` if the frame has no parent.
    pub fn transform_to_parent(&self) -> Result<Isometry3<f64>, CartesianTreeError> {
        if self.parent().is_none() {
            return Err(CartesianTreeError::RootHasNoParent(self.name()));
        }
        Ok(self.borrow().transform_to_parent)
    }

    /// Updates the frame's transformation relative to its parent.
    ///
    /// This method modifies the frame's position and orientation relative to its parent frame.
    /// It fails if the frame is a root frame (i.e., has no parent).
    ///
    /// # Arguments
    /// - `position`: A 3D vector representing the new translational offset from the parent.
    /// - `orientation`: An orientation convertible into a unit quaternion for new orientational offset from the parent.
    ///
    /// # Returns
    /// - `Ok(())` if the transformation was updated successfully.
    /// - `Err(String)` if the frame has no parent.
    ///
    /// # Example
    /// ```
    /// use cartesian_tree::Frame;
    /// use nalgebra::{Vector3, UnitQuaternion};
    ///
    /// let root = Frame::new_origin("root");
    /// let child = root
    ///     .add_child("camera", Vector3::new(0.0, 0.0, 1.0), UnitQuaternion::identity())
    ///     .unwrap();
    /// child.update_transform(Vector3::new(1.0, 0.0, 0.0), UnitQuaternion::identity())
    ///     .unwrap();
    /// ```
    pub fn update_transform<O>(
        &self,
        position: Vector3<f64>,
        orientation: O,
    ) -> Result<(), CartesianTreeError>
    where
        O: IntoOrientation,
    {
        if self.parent().is_none() {
            return Err(CartesianTreeError::CannotUpdateRootTransform(self.name()));
        }
        self.borrow_mut().transform_to_parent =
            Isometry3::from_parts(Translation3::from(position), orientation.into_orientation());
        Ok(())
    }

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
    /// # Returns
    /// - `Ok(Rc<Frame>)` the newly added child frame.
    /// - `Err(String)` if a child with the same name already exists.
    ///
    /// # Example
    /// ```
    /// use cartesian_tree::Frame;
    /// use nalgebra::{Vector3, UnitQuaternion};
    ///
    /// let root = Frame::new_origin("base");
    /// let child = root
    ///     .add_child("camera", Vector3::new(0.0, 0.0, 1.0), UnitQuaternion::identity())
    ///     .unwrap();
    /// ```
    pub fn add_child<O>(
        &self,
        name: impl Into<String>,
        position: Vector3<f64>,
        orientation: O,
    ) -> Result<Frame, CartesianTreeError>
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
                return Err(CartesianTreeError::ChildNameConflict(
                    child_name,
                    self.name(),
                ));
            }
        }
        let quat = orientation.into_orientation();
        let transform = Isometry3::from_parts(Translation3::from(position), quat);

        let child = Frame {
            data: Rc::new(RefCell::new(FrameData {
                name: child_name,
                parent: Some(Rc::downgrade(&self.data)),
                children: Vec::new(),
                transform_to_parent: transform,
            })),
        };

        self.borrow_mut().children.push(child.clone());
        Ok(child)
    }

    /// Adds a pose to the current frame.
    ///
    /// # Arguments
    /// - `position`: The translational part of the pose.
    /// - `orientation`: The orientational part of the pose.
    ///
    /// # Returns
    /// - The newly added pose.
    ///
    /// # Example
    /// ```
    /// use cartesian_tree::Frame;
    /// use nalgebra::{Vector3, UnitQuaternion};
    ///
    /// let frame = Frame::new_origin("base");
    /// let pose = frame.add_pose(Vector3::new(0.5, 0.0, 0.0), UnitQuaternion::identity());
    /// ```
    pub fn add_pose<O>(&self, position: Vector3<f64>, orientation: O) -> Pose
    where
        O: IntoOrientation,
    {
        Pose::new(self.downgrade(), position, orientation)
    }
}

impl HasParent for Frame {
    type Node = Frame;

    fn parent(&self) -> Option<Self::Node> {
        self.borrow()
            .parent
            .clone()
            .and_then(|data_weak| data_weak.upgrade().map(|data_rc| Frame { data: data_rc }))
    }
}

impl NodeEquality for Frame {
    fn is_same(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }
}

impl HasChildren for Frame {
    type Node = Frame;
    fn children(&self) -> Vec<Frame> {
        self.borrow().children.clone()
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
    }

    #[test]
    #[should_panic(expected = "already borrowed")]
    fn test_borrow_conflict() {
        let frame = Frame::new_origin("root");
        let _borrow = frame.borrow(); // Immutable borrow
        frame.borrow_mut(); // Should panic
    }

    #[test]
    fn test_add_pose_to_frame() {
        let frame = Frame::new_origin("dummy");
        let pose = frame.add_pose(Vector3::new(1.0, 2.0, 3.0), UnitQuaternion::identity());

        assert_eq!(pose.frame().unwrap().name(), "dummy");
    }

    #[test]
    fn test_update_transform() {
        let root = Frame::new_origin("root");
        let child = root
            .add_child(
                "dummy",
                Vector3::new(0.0, 0.0, 1.0),
                UnitQuaternion::identity(),
            )
            .unwrap();
        child
            .update_transform(Vector3::new(1.0, 0.0, 0.0), UnitQuaternion::identity())
            .unwrap();
        assert_eq!(
            child.transform_to_parent().unwrap().translation.vector,
            Vector3::new(1.0, 0.0, 0.0)
        );

        // Test root frame error
        assert!(
            root.update_transform(Vector3::new(1.0, 0.0, 0.0), UnitQuaternion::identity())
                .is_err()
        );
    }

    #[test]
    fn test_pose_transformation_between_frames() {
        let root = Frame::new_origin("root");

        let f1 = root
            .add_child(
                "f1",
                Vector3::new(1.0, 0.0, 0.0),
                UnitQuaternion::identity(),
            )
            .unwrap();

        let f2 = f1
            .add_child(
                "f2",
                Vector3::new(0.0, 2.0, 0.0),
                UnitQuaternion::identity(),
            )
            .unwrap();

        let pose_in_f2 = f2.add_pose(Vector3::new(1.0, 1.0, 0.0), UnitQuaternion::identity());

        let pose_in_root = pose_in_f2.in_frame(&root).unwrap();
        let pos = pose_in_root.transformation().translation.vector;

        // Total offset should be: f2 (0,2,0) + pose (1,1,0) + f1 (1,0,0)
        assert!((pos - Vector3::new(2.0, 3.0, 0.0)).norm() < 1e-6);
    }
}

use crate::CartesianTreeError;
use crate::frame::{Frame, FrameData};
use crate::orientation::IntoOrientation;
use crate::tree::Walking;
use nalgebra::{Isometry3, Translation3, Vector3};
use std::cell::RefCell;
use std::rc::Weak;

/// Use [`Frame::add_pose`] to create a new pose.
#[derive(Clone, Debug)]
pub struct Pose {
    /// Reference to the parent frame.
    parent: Weak<RefCell<FrameData>>,
    /// Transformation from this frame to its parent frame.
    transform_to_parent: Isometry3<f64>,
}

impl Pose {
    /// Creates a new pose relative to a frame.
    ///
    /// This function is intended for internal use. To create a pose associated with a frame,
    /// use [`Frame::add_pose`], which handles the association safely.
    pub(crate) fn new<O>(
        frame: Weak<RefCell<FrameData>>,
        position: Vector3<f64>,
        orientation: O,
    ) -> Pose
    where
        O: IntoOrientation,
    {
        Pose {
            parent: frame,
            transform_to_parent: Isometry3::from_parts(
                Translation3::from(position),
                orientation.into_orientation(),
            ),
        }
    }

    /// Returns the name of the frame in which this pose is defined.
    ///
    /// # Returns
    ///
    /// `Some(String)` if the parent frame is still valid, or `None` if the frame
    /// has been dropped or no longer exists.
    ///
    /// # Example
    ///
    /// ```
    /// use cartesian_tree::Frame;
    /// use nalgebra::{Vector3, UnitQuaternion};
    ///
    /// let frame = Frame::new_origin("base");
    /// let pose = frame.add_pose(Vector3::new(0.0, 0.0, 0.0), UnitQuaternion::identity());
    /// assert_eq!(pose.frame_name().as_deref(), Some("base"));
    /// ```
    pub fn frame_name(&self) -> Option<String> {
        self.parent
            .upgrade()
            .map(|frame| frame.borrow().name.clone())
    }

    /// Returns the transformation from this pose to its parent frame.
    ///
    /// # Returns
    /// The transformation of the pose in its parent frame.
    pub fn transformation(&self) -> Isometry3<f64> {
        self.transform_to_parent
    }

    /// Updates the pose's transformation relative to its parent.
    ///
    /// # Arguments
    /// - `position`: A 3D vector representing the new translational offset from the parent.
    /// - `orientation`: An orientation convertible into a unit quaternion for new orientational offset from the parent.
    ///
    /// # Example
    /// ```
    /// use cartesian_tree::Frame;
    /// use nalgebra::{Vector3, UnitQuaternion};
    ///
    /// let root = Frame::new_origin("root");
    /// let mut pose = root.add_pose(Vector3::new(0.0, 0.0, 1.0), UnitQuaternion::identity());
    /// pose.update(Vector3::new(1.0, 0.0, 0.0), UnitQuaternion::identity());
    /// ```
    pub fn update<O>(&mut self, position: Vector3<f64>, orientation: O)
    where
        O: IntoOrientation,
    {
        self.transform_to_parent =
            Isometry3::from_parts(Translation3::from(position), orientation.into_orientation());
    }

    /// Transforms this pose into the coordinate system of the given target frame.
    ///
    /// # Arguments
    /// * `target` - The frame to express this pose in.
    ///
    /// # Returns
    /// - `Ok(Pose)` a new `Pose`, expressed in the `target` frame.
    /// - `Err(String)` if the frame hierarchy cannot be resolved (due to dropped frames or no common ancestor).
    ///
    /// # Example
    /// ```
    /// use cartesian_tree::Frame;
    /// use nalgebra::{Vector3, UnitQuaternion};
    ///
    /// let root = Frame::new_origin("root");
    /// let pose = root.add_pose(Vector3::new(0.0, 0.0, 1.0), UnitQuaternion::identity());
    /// let new_frame = root.add_child("child", Vector3::new(1.0, 0.0, 0.0), UnitQuaternion::identity()).unwrap();
    /// let pose_in_new_frame = pose.in_frame(&new_frame);
    /// ```
    pub fn in_frame(&self, target: &Frame) -> Result<Pose, CartesianTreeError> {
        let source_data = self
            .parent
            .upgrade()
            .ok_or(CartesianTreeError::WeakUpgradeFailed())?;
        let source = Frame { data: source_data };
        let ancestor = source
            .lca_with(target)
            .ok_or(CartesianTreeError::NoCommonAncestor(
                source.name(),
                target.name(),
            ))?;

        // Transformation from source frame up to ancestor
        let tf_up = source.walk_up_and_transform(&ancestor)? * self.transform_to_parent;

        // Transformation from target frame up to ancestor (to be inverted)
        let tf_down = target.walk_up_and_transform(&ancestor)?;

        Ok(Pose {
            parent: target.downgrade(),
            transform_to_parent: tf_down.inverse() * tf_up,
        })
    }
}

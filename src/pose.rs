use crate::CartesianTreeError;
use crate::frame::{Frame, FrameData};
use crate::rotation::Rotation;
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
    pub(crate) fn new(
        frame: Weak<RefCell<FrameData>>,
        position: Vector3<f64>,
        orientation: impl Into<Rotation>,
    ) -> Self {
        Self {
            parent: frame,
            transform_to_parent: Isometry3::from_parts(
                Translation3::from(position),
                orientation.into().as_quaternion(),
            ),
        }
    }

    /// Returns the parent frame of this pose.
    ///
    /// # Returns
    /// `Some(Frame)` if the parent frame is still valid, or `None` if the frame
    /// has been dropped or no longer exists.
    ///
    /// # Example
    /// ```
    /// use cartesian_tree::Frame;
    /// use nalgebra::{Vector3, UnitQuaternion};
    ///
    /// let frame = Frame::new_origin("base");
    /// let pose = frame.add_pose(Vector3::new(0.0, 0.0, 0.0), UnitQuaternion::identity());
    /// assert_eq!(pose.frame().unwrap().name(), "base");
    /// ```
    #[must_use]
    pub fn frame(&self) -> Option<Frame> {
        self.parent.upgrade().map(|data| Frame { data })
    }

    /// Returns the transformation from this pose to its parent frame.
    ///
    /// # Returns
    /// The transformation of the pose in its parent frame.
    #[must_use]
    pub const fn transformation(&self) -> Isometry3<f64> {
        self.transform_to_parent
    }

    /// Sets the pose's transformation relative to its parent.
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
    /// pose.set(Vector3::new(1.0, 0.0, 0.0), UnitQuaternion::identity());
    /// ```
    pub fn set(&mut self, position: Vector3<f64>, orientation: impl Into<Rotation>) {
        self.transform_to_parent = Isometry3::from_parts(
            Translation3::from(position),
            orientation.into().as_quaternion(),
        );
    }

    /// Applies the provided isometry interpreted in the parent frame to the pose.
    ///
    /// # Arguments
    /// - `isometry`: The isometry (describing a motion in the parent frame coordinates) to apply to the current transformation.
    ///
    /// # Example
    /// ```
    /// use cartesian_tree::Frame;
    /// use nalgebra::{Isometry3, Translation3, Vector3, UnitQuaternion};
    ///
    /// let root = Frame::new_origin("root");
    /// let mut pose = root.add_pose(Vector3::new(0.0, 0.0, 1.0), UnitQuaternion::identity());
    /// pose.apply_in_parent_frame(&Isometry3::from_parts(Translation3::new(1.0, 0.0, 0.0), UnitQuaternion::identity()));
    /// ```
    pub fn apply_in_parent_frame(&mut self, isometry: &Isometry3<f64>) {
        self.transform_to_parent = isometry * self.transform_to_parent;
    }

    /// Applies the provided isometry interpreted in the body frame to this pose.
    ///
    /// # Arguments
    /// - `isometry`: The isometry (describing a motion in the body frame coordinates) to apply to the current transformation.
    ///
    /// # Example
    /// ```
    /// use cartesian_tree::Frame;
    /// use nalgebra::{Isometry3, Translation3, Vector3, UnitQuaternion};
    ///
    /// let root = Frame::new_origin("root");
    /// let mut pose = root.add_pose(Vector3::new(0.0, 0.0, 1.0), UnitQuaternion::identity());
    /// pose.apply_in_local_frame(&Isometry3::from_parts(Translation3::new(1.0, 0.0, 0.0), UnitQuaternion::identity()));
    /// ```
    pub fn apply_in_local_frame(&mut self, isometry: &Isometry3<f64>) {
        self.transform_to_parent *= isometry;
    }

    /// Transforms this pose into the coordinate system of the given target frame.
    ///
    /// # Arguments
    /// * `target` - The frame to express this pose in.
    ///
    /// # Returns
    /// A new `Pose`, expressed in the `target` frame.
    ///
    /// # Errors
    /// Returns a [`CartesianTreeError`] if:
    /// - The frame hierarchy cannot be resolved (e.g., due to dropped frames).
    /// - There is no common ancestor between `self` and `target`.
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
    pub fn in_frame(&self, target: &Frame) -> Result<Self, CartesianTreeError> {
        let source_data = self
            .parent
            .upgrade()
            .ok_or(CartesianTreeError::WeakUpgradeFailed())?;
        let source = Frame { data: source_data };
        let ancestor = source
            .lca_with(target)
            .ok_or_else(|| CartesianTreeError::NoCommonAncestor(source.name(), target.name()))?;

        // Transformation from source frame up to ancestor
        let tf_up = source.walk_up_and_transform(&ancestor)? * self.transform_to_parent;

        // Transformation from target frame up to ancestor (to be inverted)
        let tf_down = target.walk_up_and_transform(&ancestor)?;

        Ok(Self {
            parent: target.downgrade(),
            transform_to_parent: tf_down.inverse() * tf_up,
        })
    }
}

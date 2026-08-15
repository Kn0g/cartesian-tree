use crate::CartesianTreeError;
use crate::frame::{Frame, FrameKind, SharedTree, read_tree};
use crate::lazy_access::{LazyRotation, LazyTranslation};
use crate::rotation::Rotation;
use nalgebra::{Isometry3, Translation3, Vector3};
use std::ops::{Add, Mul, Sub};
use std::sync::Arc;

/// Use [`Frame::add_pose`] to create a new pose.
///
/// A pose shares ownership of the tree of the frame it lives in, so holding a pose
/// keeps the tree alive. `Pose` is `Send + Sync`.
#[derive(Clone)]
pub struct Pose {
    /// The tree of the frame this pose lives in.
    pub(crate) tree: SharedTree,
    /// The frame this pose lives in.
    pub(crate) anchor: FrameKind,
    /// Transformation from this pose to its parent frame.
    transform_to_parent: Isometry3<f64>,
}

impl std::fmt::Debug for Pose {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Deliberately does not lock the tree, so Debug is safe in any context.
        f.debug_struct("Pose")
            .field("anchor", &self.anchor)
            .field("transform_to_parent", &self.transform_to_parent)
            .finish_non_exhaustive()
    }
}

impl Pose {
    /// Creates a new pose relative to a frame.
    ///
    /// This function is intended for internal use. To create a pose associated with a frame,
    /// use [`Frame::add_pose`], which handles the association safely.
    pub(crate) fn new(
        tree: SharedTree,
        anchor: FrameKind,
        position: Vector3<f64>,
        orientation: impl Into<Rotation>,
    ) -> Self {
        Self {
            tree,
            anchor,
            transform_to_parent: Isometry3::from_parts(
                Translation3::from(position),
                orientation.into().as_quaternion(),
            ),
        }
    }

    /// Returns the parent frame of this pose.
    ///
    /// # Returns
    /// `Some(Frame)` if the frame still exists in the tree, or `None` if it has been
    /// removed.
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
        let guard = read_tree(&self.tree);
        guard.contains(self.anchor.anchor()).then(|| Frame {
            tree: Arc::clone(&self.tree),
            kind: self.anchor.clone(),
        })
    }

    /// Returns the transformation from this pose to its parent frame.
    ///
    /// # Returns
    /// The transformation of the pose in its parent frame.
    #[must_use]
    pub const fn transformation(&self) -> Isometry3<f64> {
        self.transform_to_parent
    }

    /// Returns the position of this pose relative to its parent frame.
    /// # Returns
    /// The position of the pose in its parent frame.
    #[must_use]
    pub const fn position(&self) -> Vector3<f64> {
        self.transform_to_parent.translation.vector
    }

    /// Returns the orientation of this pose relative to its parent frame.
    /// # Returns
    /// The orientation of the pose in its parent frame.
    #[must_use]
    pub fn orientation(&self) -> Rotation {
        self.transform_to_parent.rotation.into()
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
    /// The computation runs under a single read lock, so it sees a consistent snapshot
    /// of the tree even while other threads are updating transforms.
    ///
    /// # Arguments
    /// * `target` - The frame to express this pose in.
    ///
    /// # Returns
    /// A new `Pose`, expressed in the `target` frame.
    ///
    /// # Errors
    /// Returns a [`CartesianTreeError`] if:
    /// - The pose's frame or the target frame has been removed from the tree.
    /// - The frames belong to different trees.
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
        if !Arc::ptr_eq(&self.tree, &target.tree) {
            let own_name = self
                .frame()
                .map_or_else(|| "<removed>".to_owned(), |frame| frame.name());
            return Err(CartesianTreeError::DifferentTrees(own_name, target.name()));
        }

        let guard = read_tree(&self.tree);
        let source_anchor = self.anchor.anchor();
        let target_anchor = target.kind.anchor();

        let ancestor = guard.lca(source_anchor, target_anchor)?.ok_or_else(|| {
            CartesianTreeError::NoCommonAncestor(
                guard.name_of(source_anchor),
                guard.name_of(target_anchor),
            )
        })?;

        // Transformation from the pose up to the ancestor.
        let tf_up = guard.transform_up(
            source_anchor,
            self.anchor.offset() * self.transform_to_parent,
            ancestor,
        )?;

        // Transformation from the target's anchor up to the ancestor (to be inverted).
        let tf_down = guard.transform_up(target_anchor, Isometry3::identity(), ancestor)?;

        Ok(Self {
            tree: Arc::clone(&self.tree),
            anchor: target.kind.clone(),
            transform_to_parent: target.kind.offset().inverse() * (tf_down.inverse() * tf_up),
        })
    }
}

impl Add<LazyTranslation> for &Pose {
    type Output = Pose;

    fn add(self, rhs: LazyTranslation) -> Self::Output {
        let mut new_pose = self.clone();
        new_pose.apply_in_parent_frame(&rhs.inner);
        new_pose
    }
}

impl Sub<LazyTranslation> for &Pose {
    type Output = Pose;

    fn sub(self, rhs: LazyTranslation) -> Self::Output {
        let mut new_pose = self.clone();
        new_pose.apply_in_parent_frame(&rhs.inner.inverse());
        new_pose
    }
}

impl Mul<LazyRotation> for &Pose {
    type Output = Pose;

    fn mul(self, rhs: LazyRotation) -> Self::Output {
        let mut new_pose = self.clone();
        new_pose.apply_in_local_frame(&rhs.inner);
        new_pose
    }
}

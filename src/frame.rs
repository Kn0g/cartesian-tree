use crate::CartesianTreeError;
use crate::Pose;
use crate::lazy_access::LazyRotation;
use crate::lazy_access::LazyTranslation;
use crate::rotation::{MIN_QUATERNION_NORM, Rotation};
use crate::tree::Walking;
use crate::tree::{HasChildren, HasParent, NodeEquality};

use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion, Vector3};
use std::cell::RefCell;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;
use std::rc::{Rc, Weak};

use serde::{Deserialize, Serialize};
use serde_json;
use uuid::Uuid;

/// Represents a coordinate frame in a Cartesian tree structure.
///
/// Each frame can have one parent and multiple children. The frame stores its
/// transformation (position and orientation) relative to its parent.
///
/// Root frames (created via `Frame::new_origin`) have no parent and use the identity transform.
///
/// # Ownership and lifetimes
///
/// A frame owns its children but holds only a weak reference to its parent.
/// Keeping a child alive therefore does **not** keep its ancestors alive: once the
/// last owning handle to an ancestor is dropped, the frame becomes detached and
/// operations that need the parent chain (e.g. [`Frame::transformation`],
/// [`Pose::in_frame`]) return [`CartesianTreeError::ParentDropped`].
/// Keep a handle to the root frame alive for as long as the tree is in use.
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

#[derive(Serialize, Deserialize, Debug, Clone)]
struct SerialFrame {
    name: String,
    position: Vector3<f64>,
    // Deserialized as a plain quaternion because nalgebra does not re-normalize
    // `UnitQuaternion`s on deserialization; validated in `apply_serial`.
    orientation: Quaternion<f64>,
    children: Vec<SerialFrame>,
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
        Self {
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

    /// Returns `Ok(true)` if the frame has a live parent and `Ok(false)` if it is a root frame.
    ///
    /// # Errors
    /// Returns [`CartesianTreeError::ParentDropped`] if the frame had a parent that has
    /// been dropped (the frame is detached from its tree).
    fn has_live_parent(&self) -> Result<bool, CartesianTreeError> {
        match &self.borrow().parent {
            None => Ok(false),
            Some(weak) if weak.upgrade().is_some() => Ok(true),
            Some(_) => Err(CartesianTreeError::ParentDropped(self.name())),
        }
    }

    pub(crate) fn walk_up_and_transform(
        &self,
        target: &Self,
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

            // The parent reference exists (checked above), so failing to upgrade it
            // means the parent frame has been dropped.
            let parent_frame_opt = current.parent();
            current = parent_frame_opt
                .ok_or_else(|| CartesianTreeError::ParentDropped(current.name()))?;
        }

        Ok(transform)
    }

    /// Returns the name of the frame.
    #[must_use]
    pub fn name(&self) -> String {
        self.borrow().name.clone()
    }

    /// Returns the transformation from this frame to its parent frame.
    ///
    /// # Returns
    /// - The isometry from this frame to its parent frame.
    ///
    /// # Errors
    /// Returns a [`CartesianTreeError`] if:
    /// - The frame has no parent.
    pub fn transformation(&self) -> Result<Isometry3<f64>, CartesianTreeError> {
        if !self.has_live_parent()? {
            return Err(CartesianTreeError::RootHasNoParent(self.name()));
        }
        Ok(self.borrow().transform_to_parent)
    }

    /// Returns the position of this frame relative to its parent frame.
    ///
    /// # Returns
    /// The position of the frame in its parent frame.
    #[must_use]
    pub fn position(&self) -> Vector3<f64> {
        self.borrow().transform_to_parent.translation.vector
    }

    /// Returns the orientation of this frame relative to its parent frame.
    ///
    /// # Returns
    /// The orientation of the frame in its parent frame.
    #[must_use]
    pub fn orientation(&self) -> Rotation {
        self.borrow().transform_to_parent.rotation.into()
    }

    /// Sets the frame's transformation relative to its parent.
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
    ///
    /// # Errors
    /// Returns a [`CartesianTreeError`] if:
    /// - The frame has no parent (i.e., the root frame).
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
    /// child.set(Vector3::new(1.0, 0.0, 0.0), UnitQuaternion::identity())
    ///     .unwrap();
    /// ```
    pub fn set(
        &self,
        position: Vector3<f64>,
        orientation: impl Into<Rotation>,
    ) -> Result<(), CartesianTreeError> {
        if !self.has_live_parent()? {
            return Err(CartesianTreeError::CannotUpdateRootTransform(self.name()));
        }
        self.borrow_mut().transform_to_parent = Isometry3::from_parts(
            Translation3::from(position),
            orientation.into().as_quaternion(),
        );
        Ok(())
    }

    /// Applies the provided isometry interpreted in the parent frame to this frame.
    ///
    /// This method modifies the frame's position and orientation relative to its current position and orientation.
    /// It fails if the frame is a root frame (i.e., has no parent).
    ///
    /// # Arguments
    /// - `isometry`: The isometry (describing a motion in the parent frame coordinates) to apply to the current transformation.
    ///
    /// # Returns
    /// - `Ok(())` if the transformation was updated successfully.
    ///
    /// # Errors
    /// Returns a [`CartesianTreeError`] if:
    /// - The frame has no parent (i.e., the root frame).
    ///
    /// # Example
    /// ```
    /// use cartesian_tree::Frame;
    /// use nalgebra::{Isometry3, Translation3, Vector3, UnitQuaternion};
    ///
    /// let root = Frame::new_origin("root");
    /// let child = root
    ///     .add_child("camera", Vector3::new(1.0, 0.0, 1.0), UnitQuaternion::identity())
    ///     .unwrap();
    /// child.apply_in_parent_frame(&Isometry3::from_parts(Translation3::new(1.0, 0.0, 0.0), UnitQuaternion::identity()))
    ///     .unwrap();
    ///
    /// ```
    pub fn apply_in_parent_frame(
        &self,
        isometry: &Isometry3<f64>,
    ) -> Result<(), CartesianTreeError> {
        if !self.has_live_parent()? {
            return Err(CartesianTreeError::CannotUpdateRootTransform(self.name()));
        }
        let mut borrow = self.borrow_mut();
        borrow.transform_to_parent = isometry * borrow.transform_to_parent;
        Ok(())
    }

    /// Applies the provided isometry interpreted in this frame to this frame.
    ///
    /// This method modifies the frame's position and orientation relative to its current position and orientation.
    /// It fails if the frame is a root frame (i.e., has no parent).
    ///
    /// # Arguments
    /// - `isometry`: The isometry (describing a motion in this frame) to apply to the current transformation.
    ///
    /// # Returns
    /// - `Ok(())` if the transformation was updated successfully.
    ///
    /// # Errors
    /// Returns a [`CartesianTreeError`] if:
    /// - The frame has no parent (i.e., the root frame).
    ///
    /// # Example
    /// ```
    /// use cartesian_tree::Frame;
    /// use nalgebra::{Isometry3, Translation3, Vector3, UnitQuaternion};
    ///
    /// let root = Frame::new_origin("root");
    /// let child = root
    ///     .add_child("camera", Vector3::zeros(), UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2))
    ///     .unwrap();
    /// child.apply_in_local_frame(&Isometry3::from_parts(Translation3::new(1.0, 0.0, 0.0), UnitQuaternion::identity()))
    ///     .unwrap();
    ///
    /// ```
    pub fn apply_in_local_frame(
        &self,
        isometry: &Isometry3<f64>,
    ) -> Result<(), CartesianTreeError> {
        if !self.has_live_parent()? {
            return Err(CartesianTreeError::CannotUpdateRootTransform(self.name()));
        }
        let mut borrow = self.borrow_mut();
        borrow.transform_to_parent *= isometry;
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
    /// The newly added child frame.
    ///
    /// # Errors
    /// Returns a [`CartesianTreeError`] if:
    /// - A child with the same name already exists.
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
    pub fn add_child(
        &self,
        name: impl Into<String>,
        position: Vector3<f64>,
        orientation: impl Into<Rotation>,
    ) -> Result<Self, CartesianTreeError> {
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
        let transform = Isometry3::from_parts(
            Translation3::from(position),
            orientation.into().as_quaternion(),
        );

        let child = Self {
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

    /// Adds a new child frame calibrated such that a reference pose, when expressed in the new frame,
    /// matches the desired position and orientation.
    ///
    /// # Arguments
    /// - `name`: The name of the new child frame.
    /// - `desired_position`: The desired position of the reference pose in the new frame.
    /// - `desired_orientation`: The desired orientation of the reference pose in the new frame.
    /// - `reference_pose`: The existing pose (in some frame A) used as the calibration reference.
    ///
    /// # Returns
    /// - The new child frame if successful.
    ///
    /// # Errors
    /// Returns a [`CartesianTreeError`] if:
    /// - The reference frame is invalid.
    /// - No common ancestor exists.
    /// - A child with the same name already exists.
    ///
    /// # Example
    /// ```
    /// use cartesian_tree::Frame;
    /// use nalgebra::{Vector3, UnitQuaternion};
    ///
    /// let root = Frame::new_origin("root");
    /// let reference_pose = root.add_pose(Vector3::new(1.0, 0.0, 0.0), UnitQuaternion::identity());
    /// let calibrated_child = root.calibrate_child(
    ///     "calibrated",
    ///     Vector3::zeros(),
    ///     UnitQuaternion::identity(),
    ///     &reference_pose,
    /// ).unwrap();
    /// ```
    pub fn calibrate_child(
        &self,
        name: impl Into<String>,
        desired_position: Vector3<f64>,
        desired_orientation: impl Into<Rotation>,
        reference_pose: &Pose,
    ) -> Result<Self, CartesianTreeError> {
        let reference_frame = reference_pose.frame().ok_or_else(|| {
            CartesianTreeError::FrameDropped("Reference pose frame has been dropped".to_string())
        })?;

        let ancestor = self.lca_with(&reference_frame).ok_or_else(|| {
            CartesianTreeError::NoCommonAncestor(self.name(), reference_frame.name())
        })?;

        let t_reference_to_ancestor = reference_frame.walk_up_and_transform(&ancestor)?;
        let t_pose_to_reference = reference_pose.transformation();
        let t_pose_to_ancestor = t_reference_to_ancestor * t_pose_to_reference;

        let t_parent_to_ancestor = self.walk_up_and_transform(&ancestor)?;
        let t_ancestor_to_parent = t_parent_to_ancestor.inverse();

        let desired_pose = Isometry3::from_parts(
            Translation3::from(desired_position),
            desired_orientation.into().as_quaternion(),
        );

        let t_calibrated_to_parent =
            t_ancestor_to_parent * t_pose_to_ancestor * desired_pose.inverse();

        self.add_child(
            name,
            t_calibrated_to_parent.translation.vector,
            t_calibrated_to_parent.rotation,
        )
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
    pub fn add_pose(&self, position: Vector3<f64>, orientation: impl Into<Rotation>) -> Pose {
        Pose::new(self.downgrade(), position, orientation)
    }

    /// Serializes the frame tree to a JSON string.
    ///
    /// This recursively serializes the hierarchy starting from this frame (ideally the root).
    /// Transforms for root frames are set to identity.
    ///
    /// # Returns
    /// The serialized tree as a JSON string.
    ///
    /// # Errors
    /// Returns a [`CartesianTreeError`] if:
    /// - On deserialization failure.
    pub fn to_json(&self) -> Result<String, CartesianTreeError> {
        let serial = self.to_serial();
        Ok(serde_json::to_string_pretty(&serial)?)
    }

    /// Helper function to convert the frame and its children recursively into a serializable structure.
    ///
    /// This is used internally for JSON serialization.
    fn to_serial(&self) -> SerialFrame {
        let (position, orientation) = if self.parent().is_some() {
            let iso = self
                .transformation()
                .unwrap_or_else(|_| Isometry3::identity());
            (iso.translation.vector, iso.rotation.into_inner())
        } else {
            (Vector3::zeros(), Quaternion::identity())
        };

        SerialFrame {
            name: self.name(),
            position,
            orientation,
            children: self.children().into_iter().map(|c| c.to_serial()).collect(),
        }
    }

    /// Applies a JSON config to this frame tree by updating matching transforms.
    ///
    /// Deserializes the JSON to a temporary structure, then recursively updates transforms
    /// where names match (partial apply; ignores unmatched frames in config).
    /// Skips updating root frames (identity assumed) - assumes this frame is the root.
    ///
    /// # Arguments
    /// - `json`: The JSON string to apply.
    ///
    /// # Returns
    /// `Ok(())` if applied successfully (even if partial).
    ///
    /// # Errors
    /// Returns a [`CartesianTreeError`] if:
    /// - On deserialization failure.
    /// - The frame names do not match at the root.
    /// - An orientation in the config has a norm too close to zero to normalize.
    ///
    pub fn apply_config(&self, json: &str) -> Result<(), CartesianTreeError> {
        let serial: SerialFrame = serde_json::from_str(json)?;
        self.apply_serial(&serial)
    }

    fn apply_serial(&self, serial: &SerialFrame) -> Result<(), CartesianTreeError> {
        if self.name() != serial.name {
            return Err(CartesianTreeError::Mismatch(format!(
                "Frame names do not match: {} vs {}",
                self.name(),
                serial.name
            )));
        }

        // only update if frame has parent
        if self.parent().is_some() {
            let orientation = UnitQuaternion::try_new(serial.orientation, MIN_QUATERNION_NORM)
                .ok_or_else(|| {
                    let q = &serial.orientation;
                    CartesianTreeError::InvalidQuaternion(q.i, q.j, q.k, q.w)
                })?;
            self.set(serial.position, orientation)?;
        }

        for potential_child in &serial.children {
            if let Some(child) = self
                .children()
                .into_iter()
                .find(|c| c.name() == potential_child.name)
            {
                child.apply_serial(potential_child)?;
            }
        }

        Ok(())
    }

    /// Creates an auto-named child frame that coincides with this frame moved by `isometry`,
    /// where `isometry` is interpreted in this frame's parent coordinates
    /// (like [`Frame::apply_in_parent_frame`]).
    ///
    /// For root frames, the parent coordinates are the root's own coordinates.
    fn derive_child_in_parent_frame(&self, isometry: &Isometry3<f64>) -> Self {
        // The child's transform is relative to this frame, so conjugate the parent-frame
        // motion by this frame's own transform (identity for roots).
        let transform_to_parent = self.borrow().transform_to_parent;
        let local = transform_to_parent.inverse() * isometry * transform_to_parent;
        self.add_child(
            Uuid::new_v4().to_string(),
            local.translation.vector,
            local.rotation,
        )
        .expect("UUID child names cannot conflict")
    }

    /// Creates an auto-named child frame that coincides with this frame moved by `isometry`,
    /// where `isometry` is interpreted in this frame's own coordinates
    /// (like [`Frame::apply_in_local_frame`]).
    fn derive_child_in_local_frame(&self, isometry: &Isometry3<f64>) -> Self {
        self.add_child(
            Uuid::new_v4().to_string(),
            isometry.translation.vector,
            isometry.rotation,
        )
        .expect("UUID child names cannot conflict")
    }
}

/// Creates a new auto-named child frame translated by `rhs`, interpreted in the
/// parent frame of `self` (matching the `Pose` operator semantics).
impl Add<LazyTranslation> for &Frame {
    type Output = Frame;

    fn add(self, rhs: LazyTranslation) -> Self::Output {
        self.derive_child_in_parent_frame(&rhs.inner)
    }
}

/// Creates a new auto-named child frame translated by the inverse of `rhs`, interpreted
/// in the parent frame of `self` (matching the `Pose` operator semantics).
impl Sub<LazyTranslation> for &Frame {
    type Output = Frame;

    fn sub(self, rhs: LazyTranslation) -> Self::Output {
        self.derive_child_in_parent_frame(&rhs.inner.inverse())
    }
}

/// Creates a new auto-named child frame rotated by `rhs` about the axes of `self`
/// (local frame, matching the `Pose` operator semantics).
impl Mul<LazyRotation> for &Frame {
    type Output = Frame;

    fn mul(self, rhs: LazyRotation) -> Self::Output {
        self.derive_child_in_local_frame(&rhs.inner)
    }
}

impl HasParent for Frame {
    type Node = Self;

    fn parent(&self) -> Option<Self::Node> {
        self.borrow()
            .parent
            .clone()
            .and_then(|data_weak| data_weak.upgrade().map(|data_rc| Self { data: data_rc }))
    }
}

impl NodeEquality for Frame {
    fn is_same(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }
}

impl HasChildren for Frame {
    type Node = Self;
    fn children(&self) -> Vec<Self> {
        self.borrow().children.clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::lazy_access::{rx, ry, rz, x, y, z};

    use super::*;
    use approx::assert_relative_eq;
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
                Rotation::from_rpy(0.0, 0.0, std::f64::consts::FRAC_PI_2),
            )
            .unwrap();

        let child_borrow = child.borrow();
        assert_eq!(child_borrow.name, "dummy");

        let rotation = child_borrow.transform_to_parent.rotation;
        let expected = UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2);
        assert!((rotation.angle() - expected.angle()).abs() < 1e-10);
    }

    #[test]
    fn test_child_frame_transform_to_parent() {
        let root = Frame::new_origin("world");
        let child = root
            .add_child(
                "dummy",
                Vector3::new(0.0, 0.0, 1.0),
                UnitQuaternion::identity(),
            )
            .unwrap();

        let transform = child.transformation().unwrap();
        assert_eq!(transform.translation.vector, Vector3::new(0.0, 0.0, 1.0));
        assert_eq!(transform.rotation, UnitQuaternion::identity());

        assert_eq!(child.position(), Vector3::new(0.0, 0.0, 1.0));
        assert_eq!(
            child.orientation().as_quaternion(),
            UnitQuaternion::identity()
        );
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
    fn test_detached_frame_errors() {
        let child = {
            let root = Frame::new_origin("root");
            root.add_child("child", Vector3::zeros(), UnitQuaternion::identity())
                .unwrap()
        }; // The root is dropped here, detaching the child.

        assert!(matches!(
            child.transformation(),
            Err(CartesianTreeError::ParentDropped(_))
        ));
        assert!(matches!(
            child.set(Vector3::zeros(), UnitQuaternion::identity()),
            Err(CartesianTreeError::ParentDropped(_))
        ));
        assert!(matches!(
            child.apply_in_parent_frame(&Isometry3::identity()),
            Err(CartesianTreeError::ParentDropped(_))
        ));
        assert!(matches!(
            child.apply_in_local_frame(&Isometry3::identity()),
            Err(CartesianTreeError::ParentDropped(_))
        ));

        // A true root still reports the root-specific errors.
        let root = Frame::new_origin("solo");
        assert!(matches!(
            root.transformation(),
            Err(CartesianTreeError::RootHasNoParent(_))
        ));
        assert!(matches!(
            root.set(Vector3::zeros(), UnitQuaternion::identity()),
            Err(CartesianTreeError::CannotUpdateRootTransform(_))
        ));
    }

    #[test]
    fn test_walk_up_over_dropped_parent_reports_detached() {
        let leaf = {
            let root = Frame::new_origin("root");
            let mid = root
                .add_child("mid", Vector3::zeros(), UnitQuaternion::identity())
                .unwrap();
            mid.add_child("leaf", Vector3::zeros(), UnitQuaternion::identity())
                .unwrap()
        }; // Dropping the root drops "mid" as well, detaching the leaf.
        let other_root = Frame::new_origin("other");

        assert!(matches!(
            leaf.walk_up_and_transform(&other_root),
            Err(CartesianTreeError::ParentDropped(_))
        ));
    }

    #[test]
    fn test_add_pose_to_frame() {
        let frame = Frame::new_origin("dummy");
        let pose = frame.add_pose(Vector3::new(1.0, 2.0, 3.0), UnitQuaternion::identity());

        assert_eq!(pose.frame().unwrap().name(), "dummy");
    }

    #[test]
    fn test_set_transform() {
        let root = Frame::new_origin("root");
        let child = root
            .add_child(
                "dummy",
                Vector3::new(0.0, 0.0, 1.0),
                UnitQuaternion::identity(),
            )
            .unwrap();
        child
            .set(Vector3::new(1.0, 0.0, 0.0), UnitQuaternion::identity())
            .unwrap();
        assert_eq!(
            child.transformation().unwrap().translation.vector,
            Vector3::new(1.0, 0.0, 0.0)
        );

        // Test root frame error
        assert!(
            root.set(Vector3::new(1.0, 0.0, 0.0), UnitQuaternion::identity())
                .is_err()
        );
    }

    #[test]
    fn test_apply_in_parent_frame() {
        let root = Frame::new_origin("root");
        let child = root
            .add_child(
                "dummy",
                Vector3::new(1.0, 0.0, 1.0),
                UnitQuaternion::identity(),
            )
            .unwrap();
        child
            .apply_in_parent_frame(&Isometry3::from_parts(
                Translation3::identity(),
                UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
            ))
            .unwrap();

        assert_relative_eq!(
            child.transformation().unwrap().translation.vector,
            Vector3::new(0.0, 1.0, 1.0),
            epsilon = 1e-10
        );

        child
            .apply_in_parent_frame(&Isometry3::from_parts(
                Translation3::new(1.0, 0.0, 1.0),
                UnitQuaternion::identity(),
            ))
            .unwrap();
        assert_relative_eq!(
            child.transformation().unwrap().translation.vector,
            Vector3::new(1.0, 1.0, 2.0),
            epsilon = 1e-10
        );

        // Test root frame error
        assert!(
            root.set(Vector3::new(1.0, 0.0, 0.0), UnitQuaternion::identity())
                .is_err()
        );
    }

    #[test]
    fn test_apply_in_local_frame() {
        let root = Frame::new_origin("root");
        let child = root
            .add_child(
                "dummy",
                Vector3::zeros(),
                UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
            )
            .unwrap();

        child
            .apply_in_local_frame(&Isometry3::from_parts(
                Translation3::new(1.0, 0.0, 0.0),
                UnitQuaternion::identity(),
            ))
            .unwrap();

        assert_relative_eq!(
            child.transformation().unwrap().translation.vector,
            Vector3::new(0.0, 1.0, 0.0),
            epsilon = 1e-10
        );

        child
            .apply_in_local_frame(&Isometry3::from_parts(
                Translation3::identity(),
                UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
            ))
            .unwrap();
        assert_relative_eq!(
            child.transformation().unwrap().translation.vector,
            Vector3::new(0.0, 1.0, 0.0),
            epsilon = 1e-10
        );

        let (roll, pitch, yaw) = child.transformation().unwrap().rotation.euler_angles();
        assert_relative_eq!(roll, 0.0, epsilon = 1e-10);
        assert_relative_eq!(pitch, 0.0, epsilon = 1e-10);
        assert_relative_eq!(yaw, std::f64::consts::PI, epsilon = 1e-10);

        // Test root frame error
        assert!(
            root.set(Vector3::new(1.0, 0.0, 0.0), UnitQuaternion::identity())
                .is_err()
        );
    }

    #[test]
    fn test_pose_apply_in_parent_frame() {
        let root = Frame::new_origin("root");
        let mut pose = root.add_pose(Vector3::new(1.0, 0.0, 1.0), UnitQuaternion::identity());

        pose.apply_in_parent_frame(&Isometry3::from_parts(
            Translation3::identity(),
            UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
        ));

        assert_relative_eq!(
            pose.transformation().translation.vector,
            Vector3::new(0.0, 1.0, 1.0),
            epsilon = 1e-10
        );

        pose.apply_in_parent_frame(&Isometry3::from_parts(
            Translation3::new(1.0, 0.0, 1.0),
            UnitQuaternion::identity(),
        ));
        assert_relative_eq!(
            pose.transformation().translation.vector,
            Vector3::new(1.0, 1.0, 2.0),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_pose_apply_in_local_frame() {
        let root = Frame::new_origin("root");
        let mut pose = root.add_pose(
            Vector3::zeros(),
            UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
        );

        pose.apply_in_local_frame(&Isometry3::from_parts(
            Translation3::new(1.0, 0.0, 0.0),
            UnitQuaternion::identity(),
        ));

        assert_relative_eq!(
            pose.transformation().translation.vector,
            Vector3::new(0.0, 1.0, 0.0),
            epsilon = 1e-10
        );

        pose.apply_in_local_frame(&Isometry3::from_parts(
            Translation3::identity(),
            UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
        ));
        assert_relative_eq!(
            pose.transformation().translation.vector,
            Vector3::new(0.0, 1.0, 0.0),
            epsilon = 1e-10
        );

        let (roll, pitch, yaw) = pose.transformation().rotation.euler_angles();
        assert_relative_eq!(roll, 0.0, epsilon = 1e-10);
        assert_relative_eq!(pitch, 0.0, epsilon = 1e-10);
        assert_relative_eq!(yaw, std::f64::consts::PI, epsilon = 1e-10);

        // Test root frame error
        assert!(
            root.set(Vector3::new(1.0, 0.0, 0.0), UnitQuaternion::identity())
                .is_err()
        );
    }

    #[test]
    fn test_pose_transform_to_parent() {
        let root = Frame::new_origin("root");
        let pose = root.add_pose(Vector3::new(1.0, 2.0, 3.0), UnitQuaternion::identity());

        let transformation = pose.transformation();
        assert_eq!(
            transformation.translation.vector,
            Vector3::new(1.0, 2.0, 3.0)
        );
        assert_eq!(transformation.rotation, UnitQuaternion::identity());

        assert_eq!(pose.position(), Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(
            pose.orientation().as_quaternion(),
            UnitQuaternion::identity()
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

    #[test]
    fn test_pose_round_trip_through_deep_tree() {
        // Two branches, each two levels deep, all with non-identity transforms:
        // expressing a pose in the other branch and back must be lossless.
        let root = Frame::new_origin("root");
        let a = root
            .add_child(
                "a",
                Vector3::new(0.3, -1.2, 2.5),
                UnitQuaternion::from_euler_angles(0.4, -0.3, 1.2),
            )
            .unwrap();
        let b = a
            .add_child(
                "b",
                Vector3::new(-2.0, 0.7, 0.1),
                UnitQuaternion::from_euler_angles(-1.0, 0.2, 0.5),
            )
            .unwrap();
        let c = root
            .add_child(
                "c",
                Vector3::new(1.5, 2.0, -0.4),
                UnitQuaternion::from_euler_angles(0.1, 1.1, -0.7),
            )
            .unwrap();
        let d = c
            .add_child(
                "d",
                Vector3::new(0.0, -0.5, 1.0),
                UnitQuaternion::from_euler_angles(0.9, -0.8, 0.3),
            )
            .unwrap();

        let pose = b.add_pose(
            Vector3::new(0.2, 0.4, -0.6),
            UnitQuaternion::from_euler_angles(0.5, 0.5, -0.5),
        );

        let round_tripped = pose.in_frame(&d).unwrap().in_frame(&b).unwrap();
        let original = pose.transformation();
        let result = round_tripped.transformation();
        assert_relative_eq!(
            result.translation.vector,
            original.translation.vector,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            result.rotation.angle_to(&original.rotation),
            0.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_in_frame_across_disjoint_trees_fails() {
        let tree_1 = Frame::new_origin("tree_1");
        let tree_2 = Frame::new_origin("tree_2");
        let pose = tree_1.add_pose(Vector3::zeros(), UnitQuaternion::identity());

        assert!(matches!(
            pose.in_frame(&tree_2),
            Err(CartesianTreeError::NoCommonAncestor(..))
        ));
    }

    #[test]
    fn test_apply_config_malformed_json_fails() {
        let root = Frame::new_origin("root");
        assert!(matches!(
            root.apply_config("not json"),
            Err(CartesianTreeError::SerdeError(_))
        ));
    }

    #[test]
    fn test_lazy_helpers_all_axes() {
        use nalgebra::UnitQuaternion;

        let root = Frame::new_origin("root");
        let pose = root.add_pose(Vector3::zeros(), UnitQuaternion::identity());

        let moved = &(&(&pose + x(1.0)) + y(2.0)) + z(3.0);
        assert_relative_eq!(
            moved.transformation().translation.vector,
            Vector3::new(1.0, 2.0, 3.0),
            epsilon = 1e-10
        );

        let rotated = &pose * rx(0.3);
        assert_relative_eq!(
            rotated
                .transformation()
                .rotation
                .angle_to(&UnitQuaternion::from_euler_angles(0.3, 0.0, 0.0)),
            0.0,
            epsilon = 1e-10
        );

        let rotated = &pose * ry(0.4);
        assert_relative_eq!(
            rotated
                .transformation()
                .rotation
                .angle_to(&UnitQuaternion::from_euler_angles(0.0, 0.4, 0.0)),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_calibrate_child() {
        let root = Frame::new_origin("root");

        let reference_pose = root.add_pose(
            Vector3::new(1.0, 2.0, 3.0),
            UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
        );

        // Calibrate a child where the reference pose should appear at (0,0,0) with identity orientation.
        let calibrated_frame = root
            .calibrate_child(
                "calibrated",
                Vector3::zeros(),
                UnitQuaternion::identity(),
                &reference_pose,
            )
            .unwrap();

        let pose_in_calibrated = reference_pose.in_frame(&calibrated_frame).unwrap();
        let transformation = pose_in_calibrated.transformation();

        assert!((transformation.translation.vector - Vector3::zeros()).norm() < 1e-6);
        assert!((transformation.rotation.angle() - 0.0).abs() < 1e-6);

        // Verify the child's transform matches the reference pose's original transform.
        let calibrated_transformation = calibrated_frame.transformation().unwrap();
        assert!(
            (calibrated_transformation.translation.vector - Vector3::new(1.0, 2.0, 3.0)).norm()
                < 1e-6
        );
        assert!(
            (calibrated_transformation.rotation.angle() - std::f64::consts::FRAC_PI_2).abs() < 1e-6
        );
    }

    #[test]
    fn test_calibrate_child_under_non_identity_parent() {
        let root = Frame::new_origin("root");

        // The parent of the calibrated frame is NOT the common ancestor and has a
        // non-identity transform
        let mount = root
            .add_child(
                "mount",
                Vector3::new(0.0, 0.0, 1.0),
                UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
            )
            .unwrap();

        let reference_pose = root.add_pose(
            Vector3::new(1.0, 2.0, 3.0),
            UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_4),
        );

        let desired_position = Vector3::new(0.5, 0.0, 0.0);
        let desired_orientation =
            UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2);

        let calibrated = mount
            .calibrate_child(
                "calibrated",
                desired_position,
                desired_orientation,
                &reference_pose,
            )
            .unwrap();

        // The reference pose expressed in the calibrated frame must match the desired transform.
        let pose_in_calibrated = reference_pose.in_frame(&calibrated).unwrap();
        let transformation = pose_in_calibrated.transformation();
        assert_relative_eq!(
            transformation.translation.vector,
            desired_position,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            transformation.rotation.angle_to(&desired_orientation),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_to_json_and_apply_config() {
        let root = Frame::new_origin("root");
        let _ = root
            .add_child(
                "child",
                Vector3::new(1.0, 2.0, 3.0),
                UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3),
            )
            .unwrap();

        let json = root.to_json().unwrap();
        // roughly verify JSON structure
        assert!(json.contains(r#""name": "root""#));
        assert!(json.contains(r#""name": "child""#));

        // Create a default tree with different transforms
        let default_root = Frame::new_origin("root");
        default_root
            .add_child(
                "child",
                Vector3::new(0.0, 0.0, 0.0),
                UnitQuaternion::identity(),
            )
            .unwrap();

        // Apply config
        default_root.apply_config(&json).unwrap();

        // Verify child transform updated
        let updated_child = default_root
            .children()
            .into_iter()
            .find(|c| c.name() == "child")
            .unwrap();
        let iso = updated_child.transformation().unwrap();
        assert_eq!(iso.translation.vector, Vector3::new(1.0, 2.0, 3.0));
        let (r, p, y) = iso.rotation.euler_angles();
        assert!((r - 0.1).abs() < 1e-6);
        assert!((p - 0.2).abs() < 1e-6);
        assert!((y - 0.3).abs() < 1e-6);

        // Test partial: If config has extra, ignore it
        let partial_json = r#"
        {
            "name": "root",
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "children": [
                {
                    "name": "child",
                    "position": [4.0, 5.0, 6.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "children": []
                },
                {
                    "name": "extra",
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "children": []
                }
            ]
        }
        "#;
        default_root.apply_config(partial_json).unwrap();
        let updated_child = default_root
            .children()
            .into_iter()
            .find(|c| c.name() == "child")
            .unwrap();
        assert_eq!(
            updated_child.transformation().unwrap().translation.vector,
            Vector3::new(4.0, 5.0, 6.0)
        );

        // Test mismatch
        let mismatch_json = r#"
        {
            "name": "wrong_root",
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "children": []
        }
        "#;
        assert!(default_root.apply_config(mismatch_json).is_err());
    }

    #[test]
    fn test_apply_config_validates_quaternions() {
        let root = Frame::new_origin("root");
        let child = root
            .add_child("child", Vector3::zeros(), UnitQuaternion::identity())
            .unwrap();

        // Non-unit quaternions are normalized on load.
        let scaled_json = r#"
        {
            "name": "root",
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "children": [
                {
                    "name": "child",
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 2.0, 0.0],
                    "children": []
                }
            ]
        }
        "#;
        root.apply_config(scaled_json).unwrap();
        let q = child.orientation().as_quaternion();
        assert_relative_eq!(q.k, 1.0, epsilon = 1e-12);
        assert_relative_eq!(q.w, 0.0, epsilon = 1e-12);

        // Zero-norm quaternions are rejected.
        let zero_json = scaled_json.replace("2.0", "0.0");
        assert!(matches!(
            root.apply_config(&zero_json),
            Err(CartesianTreeError::InvalidQuaternion(..))
        ));
    }

    #[test]
    fn test_lazy_translation_frame() {
        use nalgebra::UnitQuaternion;

        let root = Frame::new_origin("root");
        let child = root
            .add_child(
                "child",
                Vector3::new(0.0, 0.0, 0.0),
                UnitQuaternion::identity(),
            )
            .unwrap();

        let result = &child + z(5.0);
        assert_relative_eq!(
            result.transformation().unwrap().translation.vector,
            Vector3::new(0.0, 0.0, 5.0),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            child.transformation().unwrap().translation.vector,
            Vector3::new(0.0, 0.0, 0.0),
            epsilon = 1e-10
        );

        // Chained operations accumulate in world coordinates.
        let result = &result - y(3.0);
        let result_in_root = result
            .add_pose(Vector3::zeros(), UnitQuaternion::identity())
            .in_frame(&root)
            .unwrap()
            .transformation();
        assert_relative_eq!(
            result_in_root.translation.vector,
            Vector3::new(0.0, -3.0, 5.0),
            epsilon = 1e-10
        );

        let (roll, pitch, yaw) = result_in_root.rotation.euler_angles();
        assert_relative_eq!(
            Vector3::new(roll, pitch, yaw),
            Vector3::new(0.0, 0.0, 0.0),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_lazy_rotation_frame() {
        use nalgebra::UnitQuaternion;
        let root = Frame::new_origin("root");
        let child = root
            .add_child(
                "child",
                Vector3::new(0.0, 0.0, 0.0),
                UnitQuaternion::identity(),
            )
            .unwrap();
        let result = &child * rz(std::f64::consts::FRAC_PI_4);

        let (roll, pitch, yaw) = result.transformation().unwrap().rotation.euler_angles();
        assert_relative_eq!(
            Vector3::new(roll, pitch, yaw),
            Vector3::new(0.0, 0.0, std::f64::consts::FRAC_PI_4),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.transformation().unwrap().translation.vector,
            Vector3::new(0.0, 0.0, 0.0),
            epsilon = 1e-10
        );
        let (roll, pitch, yaw) = child.transformation().unwrap().rotation.euler_angles();
        assert_relative_eq!(
            Vector3::new(roll, pitch, yaw),
            Vector3::new(0.0, 0.0, 0.0),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_lazy_ops_on_non_identity_frame() {
        use nalgebra::UnitQuaternion;

        let root = Frame::new_origin("root");
        let yaw_90 = UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2);
        let child = root
            .add_child("child", Vector3::new(1.0, 0.0, 0.0), yaw_90)
            .unwrap();

        // Translation is interpreted in the parent frame: the derived frame must sit at
        // child + (0, 3, 0) in root coordinates, with unchanged orientation.
        let shifted = &child + y(3.0);
        let shifted_in_root = shifted
            .add_pose(Vector3::zeros(), UnitQuaternion::identity())
            .in_frame(&root)
            .unwrap()
            .transformation();
        assert_relative_eq!(
            shifted_in_root.translation.vector,
            Vector3::new(1.0, 3.0, 0.0),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            shifted_in_root.rotation.angle_to(&yaw_90),
            0.0,
            epsilon = 1e-10
        );

        // Rotation is interpreted in the local frame: position unchanged, yaw doubled.
        let rotated = &child * rz(std::f64::consts::FRAC_PI_2);
        let rotated_in_root = rotated
            .add_pose(Vector3::zeros(), UnitQuaternion::identity())
            .in_frame(&root)
            .unwrap()
            .transformation();
        assert_relative_eq!(
            rotated_in_root.translation.vector,
            Vector3::new(1.0, 0.0, 0.0),
            epsilon = 1e-10
        );
        let yaw_180 = UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::PI);
        assert_relative_eq!(
            rotated_in_root.rotation.angle_to(&yaw_180),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_lazy_translation_pose() {
        use nalgebra::UnitQuaternion;

        let root = Frame::new_origin("root");
        let pose = root.add_pose(Vector3::new(0.0, 0.0, 0.0), UnitQuaternion::identity());

        let result = &pose + z(5.0);
        assert_relative_eq!(
            result.transformation().translation.vector,
            Vector3::new(0.0, 0.0, 5.0),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            pose.transformation().translation.vector,
            Vector3::new(0.0, 0.0, 0.0),
            epsilon = 1e-10
        );

        let result = &result - y(3.0);
        assert_relative_eq!(
            result.transformation().translation.vector,
            Vector3::new(0.0, -3.0, 5.0),
            epsilon = 1e-10
        );

        let (roll, pitch, yaw) = result.transformation().rotation.euler_angles();
        assert_relative_eq!(
            Vector3::new(roll, pitch, yaw),
            Vector3::new(0.0, 0.0, 0.0),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_lazy_rotation_pose() {
        use nalgebra::UnitQuaternion;
        let root = Frame::new_origin("root");
        let pose = root.add_pose(Vector3::new(0.0, 0.0, 0.0), UnitQuaternion::identity());
        let result = &pose * rz(std::f64::consts::FRAC_PI_4);

        let (roll, pitch, yaw) = result.transformation().rotation.euler_angles();
        assert_relative_eq!(
            Vector3::new(roll, pitch, yaw),
            Vector3::new(0.0, 0.0, std::f64::consts::FRAC_PI_4),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.transformation().translation.vector,
            Vector3::new(0.0, 0.0, 0.0),
            epsilon = 1e-10
        );
        let (roll, pitch, yaw) = pose.transformation().rotation.euler_angles();
        assert_relative_eq!(
            Vector3::new(roll, pitch, yaw),
            Vector3::new(0.0, 0.0, 0.0),
            epsilon = 1e-10
        );
    }
}

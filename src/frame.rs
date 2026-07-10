use crate::CartesianTreeError;
use crate::Pose;
use crate::lazy_access::{LazyRotation, LazyTranslation};
use crate::rotation::{MIN_QUATERNION_NORM, Rotation};
use crate::tree::{HasChildren, HasParent, NodeEquality};

use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use slotmap::SlotMap;
use std::ops::{Add, Mul, Sub};
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
use uuid::Uuid;

slotmap::new_key_type! {
    /// Generational key identifying a frame node in a tree arena. Stale keys (of removed
    /// nodes) are detected by the generation counter and never alias a new node.
    pub(crate) struct NodeKey;
}

/// A single frame node stored in a tree arena.
#[derive(Debug)]
struct Node {
    /// The name of the frame (must be unique among siblings).
    name: String,
    /// The parent node, or `None` for the root.
    parent: Option<NodeKey>,
    /// Child nodes directly connected to this node.
    children: Vec<NodeKey>,
    /// Transformation from this frame to its parent frame.
    transform_to_parent: Isometry3<f64>,
}

/// The arena storage shared by all frames and poses of one tree.
#[derive(Debug)]
pub(crate) struct TreeInner {
    nodes: SlotMap<NodeKey, Node>,
}

pub(crate) type SharedTree = Arc<RwLock<TreeInner>>;

/// Acquires the tree read lock, recovering from poisoning (a panicked writer).
pub(crate) fn read_tree(tree: &SharedTree) -> RwLockReadGuard<'_, TreeInner> {
    tree.read().unwrap_or_else(PoisonError::into_inner)
}

/// Acquires the tree write lock, recovering from poisoning (a panicked writer).
pub(crate) fn write_tree(tree: &SharedTree) -> RwLockWriteGuard<'_, TreeInner> {
    tree.write().unwrap_or_else(PoisonError::into_inner)
}

impl TreeInner {
    fn node(&self, key: NodeKey) -> Result<&Node, CartesianTreeError> {
        self.nodes.get(key).ok_or(CartesianTreeError::FrameRemoved)
    }

    fn node_mut(&mut self, key: NodeKey) -> Result<&mut Node, CartesianTreeError> {
        self.nodes
            .get_mut(key)
            .ok_or(CartesianTreeError::FrameRemoved)
    }

    pub(crate) fn contains(&self, key: NodeKey) -> bool {
        self.nodes.contains_key(key)
    }

    pub(crate) fn name_of(&self, key: NodeKey) -> String {
        self.nodes
            .get(key)
            .map_or_else(|| "<removed>".to_owned(), |node| node.name.clone())
    }

    fn add_child_node(
        &mut self,
        parent: NodeKey,
        name: String,
        transform: Isometry3<f64>,
    ) -> Result<NodeKey, CartesianTreeError> {
        let children = self.node(parent)?.children.clone();
        if children
            .iter()
            .any(|&child| self.nodes.get(child).is_some_and(|node| node.name == name))
        {
            return Err(CartesianTreeError::ChildNameConflict(
                name,
                self.name_of(parent),
            ));
        }
        let key = self.nodes.insert(Node {
            name,
            parent: Some(parent),
            children: Vec::new(),
            transform_to_parent: transform,
        });
        self.nodes[parent].children.push(key);
        Ok(key)
    }

    fn remove_subtree(&mut self, key: NodeKey) {
        if let Some(node) = self.nodes.remove(key) {
            for child in node.children {
                self.remove_subtree(child);
            }
        }
    }

    fn depth_of(&self, key: NodeKey) -> Result<usize, CartesianTreeError> {
        let mut depth = 0;
        let mut current = key;
        while let Some(parent) = self.node(current)?.parent {
            depth += 1;
            current = parent;
        }
        Ok(depth)
    }

    /// Finds the lowest common ancestor of two nodes, or `None` if they are unconnected.
    pub(crate) fn lca(
        &self,
        a: NodeKey,
        b: NodeKey,
    ) -> Result<Option<NodeKey>, CartesianTreeError> {
        let mut own = a;
        let mut other = b;
        let mut own_depth = self.depth_of(own)?;
        let mut other_depth = self.depth_of(other)?;

        while own_depth > other_depth {
            own = self.node(own)?.parent.expect("depth guarantees a parent");
            own_depth -= 1;
        }
        while other_depth > own_depth {
            other = self.node(other)?.parent.expect("depth guarantees a parent");
            other_depth -= 1;
        }
        while own != other {
            match (self.node(own)?.parent, self.node(other)?.parent) {
                (Some(own_parent), Some(other_parent)) => {
                    own = own_parent;
                    other = other_parent;
                }
                _ => return Ok(None),
            }
        }
        Ok(Some(own))
    }

    /// Accumulates the transformation from `start` (pre-composed with `start_offset`)
    /// up to the ancestor `target`.
    pub(crate) fn transform_up(
        &self,
        start: NodeKey,
        start_offset: Isometry3<f64>,
        target: NodeKey,
    ) -> Result<Isometry3<f64>, CartesianTreeError> {
        let mut transform = start_offset;
        let mut current = start;
        while current != target {
            let node = self.node(current)?;
            let Some(parent) = node.parent else {
                return Err(CartesianTreeError::IsNoAncestor(
                    self.name_of(target),
                    self.name_of(start),
                ));
            };
            transform = node.transform_to_parent * transform;
            current = parent;
        }
        Ok(transform)
    }
}

/// Identifies which kind of frame a handle refers to.
#[derive(Clone, Debug)]
pub(crate) enum FrameKind {
    /// A regular frame stored in the tree arena.
    Node(NodeKey),
    /// A frame derived by the lazy operators: anchored to an arena node with a fixed
    /// offset, but not stored in the arena itself. Derived frames are read-only.
    Derived {
        anchor: NodeKey,
        offset: Isometry3<f64>,
        name: String,
    },
}

impl FrameKind {
    /// The arena node this frame resolves transforms through.
    pub(crate) const fn anchor(&self) -> NodeKey {
        match self {
            Self::Node(key) => *key,
            Self::Derived { anchor, .. } => *anchor,
        }
    }

    /// The fixed offset of this frame relative to its anchor node (identity for nodes).
    pub(crate) fn offset(&self) -> Isometry3<f64> {
        match self {
            Self::Node(_) => Isometry3::identity(),
            Self::Derived { offset, .. } => *offset,
        }
    }
}

/// Represents a coordinate frame in a Cartesian tree structure.
///
/// Each frame can have one parent and multiple children. The frame stores its
/// transformation (position and orientation) relative to its parent.
///
/// Root frames (created via `Frame::new_origin`) have no parent and use the identity transform.
///
/// # Ownership, lifetimes, and thread safety
///
/// All frames of a tree share ownership of a single tree arena: keeping any `Frame`
/// (or [`Pose`]) handle alive keeps the whole tree alive, so a leaf handle is always
/// enough to reach the root. Frames removed via [`Frame::remove_child`] become stale;
/// operations on stale handles return [`CartesianTreeError::FrameRemoved`].
///
/// `Frame` is `Send + Sync` and can be shared freely across threads. All operations
/// are synchronized through a tree-wide read/write lock, and every operation
/// (including multi-frame computations like [`Pose::in_frame`]) sees a consistent
/// snapshot of the tree.
#[derive(Clone)]
pub struct Frame {
    pub(crate) tree: SharedTree,
    pub(crate) kind: FrameKind,
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Deliberately does not lock the tree, so Debug is safe in any context.
        f.debug_struct("Frame")
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

impl Frame {
    /// Creates a new root frame (origin) with the given name.
    ///
    /// This allocates a new tree; all frames added below the root share it.
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
        let mut nodes = SlotMap::with_key();
        let root = nodes.insert(Node {
            name: name.into(),
            parent: None,
            children: Vec::new(),
            transform_to_parent: Isometry3::identity(),
        });
        Self {
            tree: Arc::new(RwLock::new(TreeInner { nodes })),
            kind: FrameKind::Node(root),
        }
    }

    fn read(&self) -> RwLockReadGuard<'_, TreeInner> {
        read_tree(&self.tree)
    }

    fn write(&self) -> RwLockWriteGuard<'_, TreeInner> {
        write_tree(&self.tree)
    }

    /// Returns the key of this frame if it is a regular (non-derived) frame,
    /// or a [`CartesianTreeError::DerivedFrameUnsupported`] error otherwise.
    fn node_key(&self) -> Result<NodeKey, CartesianTreeError> {
        match &self.kind {
            FrameKind::Node(key) => Ok(*key),
            FrameKind::Derived { name, .. } => {
                Err(CartesianTreeError::DerivedFrameUnsupported(name.clone()))
            }
        }
    }

    /// Returns the name of the frame.
    ///
    /// Returns `"<removed>"` for stale handles whose frame has been removed from the tree.
    #[must_use]
    pub fn name(&self) -> String {
        match &self.kind {
            FrameKind::Node(key) => self.read().name_of(*key),
            FrameKind::Derived { name, .. } => name.clone(),
        }
    }

    /// Returns the transformation from this frame to its parent frame.
    ///
    /// # Returns
    /// - The isometry from this frame to its parent frame.
    ///
    /// # Errors
    /// Returns a [`CartesianTreeError`] if:
    /// - The frame has no parent.
    /// - The frame has been removed from its tree.
    pub fn transformation(&self) -> Result<Isometry3<f64>, CartesianTreeError> {
        match &self.kind {
            FrameKind::Node(key) => {
                let guard = self.read();
                let node = guard.node(*key)?;
                if node.parent.is_none() {
                    return Err(CartesianTreeError::RootHasNoParent(node.name.clone()));
                }
                Ok(node.transform_to_parent)
            }
            FrameKind::Derived { offset, .. } => Ok(*offset),
        }
    }

    /// Returns the transformation of this frame relative to its parent, where root
    /// frames report identity (unlike [`Frame::transformation`], which errors).
    fn local_transform(&self) -> Result<Isometry3<f64>, CartesianTreeError> {
        match &self.kind {
            FrameKind::Node(key) => Ok(self.read().node(*key)?.transform_to_parent),
            FrameKind::Derived { offset, .. } => Ok(*offset),
        }
    }

    /// Returns the position of this frame relative to its parent frame.
    ///
    /// # Returns
    /// The position of the frame in its parent frame (zero for root frames).
    ///
    /// # Errors
    /// Returns a [`CartesianTreeError`] if:
    /// - The frame has been removed from its tree.
    pub fn position(&self) -> Result<Vector3<f64>, CartesianTreeError> {
        Ok(self.local_transform()?.translation.vector)
    }

    /// Returns the orientation of this frame relative to its parent frame.
    ///
    /// # Returns
    /// The orientation of the frame in its parent frame (identity for root frames).
    ///
    /// # Errors
    /// Returns a [`CartesianTreeError`] if:
    /// - The frame has been removed from its tree.
    pub fn orientation(&self) -> Result<Rotation, CartesianTreeError> {
        Ok(self.local_transform()?.rotation.into())
    }

    /// Sets the frame's transformation relative to its parent.
    ///
    /// This method modifies the frame's position and orientation relative to its parent frame.
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
    /// - The frame has been removed from its tree.
    /// - The frame is a derived frame (derived frames are read-only).
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
        let transform = Isometry3::from_parts(
            Translation3::from(position),
            orientation.into().as_quaternion(),
        );
        self.update_transform(|_| transform)
    }

    /// Applies the provided isometry interpreted in the parent frame to this frame.
    ///
    /// This method modifies the frame's position and orientation relative to its current position and orientation.
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
    /// - The frame has been removed from its tree.
    /// - The frame is a derived frame (derived frames are read-only).
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
        self.update_transform(|current| isometry * current)
    }

    /// Applies the provided isometry interpreted in this frame to this frame.
    ///
    /// This method modifies the frame's position and orientation relative to its current position and orientation.
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
    /// - The frame has been removed from its tree.
    /// - The frame is a derived frame (derived frames are read-only).
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
        self.update_transform(|current| current * isometry)
    }

    /// Applies `update` to this frame's transform-to-parent under the write lock.
    fn update_transform(
        &self,
        update: impl FnOnce(Isometry3<f64>) -> Isometry3<f64>,
    ) -> Result<(), CartesianTreeError> {
        let key = self.node_key()?;
        let mut guard = self.write();
        let node = guard.node_mut(key)?;
        if node.parent.is_none() {
            return Err(CartesianTreeError::CannotUpdateRootTransform(
                node.name.clone(),
            ));
        }
        node.transform_to_parent = update(node.transform_to_parent);
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
    /// - The frame has been removed from its tree.
    /// - The frame is a derived frame.
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
        let key = self.node_key()?;
        let transform = Isometry3::from_parts(
            Translation3::from(position),
            orientation.into().as_quaternion(),
        );
        let child_key = self.write().add_child_node(key, name.into(), transform)?;
        Ok(Self {
            tree: Arc::clone(&self.tree),
            kind: FrameKind::Node(child_key),
        })
    }

    /// Removes the child with the given name and its entire subtree from the tree.
    ///
    /// Existing handles to removed frames become stale and return
    /// [`CartesianTreeError::FrameRemoved`] when used.
    ///
    /// # Arguments
    /// - `name`: The name of the child frame to remove.
    ///
    /// # Errors
    /// Returns a [`CartesianTreeError`] if:
    /// - No child with the given name exists.
    /// - The frame has been removed from its tree.
    /// - The frame is a derived frame.
    ///
    /// # Example
    /// ```
    /// use cartesian_tree::Frame;
    /// use cartesian_tree::tree::HasChildren;
    /// use nalgebra::{Vector3, UnitQuaternion};
    ///
    /// let root = Frame::new_origin("root");
    /// let _ = root
    ///     .add_child("camera", Vector3::new(0.0, 0.0, 1.0), UnitQuaternion::identity())
    ///     .unwrap();
    /// root.remove_child("camera").unwrap();
    /// assert!(root.children().is_empty());
    /// ```
    pub fn remove_child(&self, name: &str) -> Result<(), CartesianTreeError> {
        let key = self.node_key()?;
        let mut guard = self.write();
        let children = guard.node(key)?.children.clone();
        let child_key = children
            .iter()
            .copied()
            .find(|&child| guard.nodes.get(child).is_some_and(|node| node.name == name))
            .ok_or_else(|| {
                CartesianTreeError::ChildNotFound(name.to_owned(), guard.name_of(key))
            })?;
        guard.node_mut(key)?.children.retain(|&c| c != child_key);
        guard.remove_subtree(child_key);
        Ok(())
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
    /// - The reference pose belongs to a different tree or its frame has been removed.
    /// - No common ancestor exists.
    /// - A child with the same name already exists.
    /// - The frame is a derived frame.
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
        let key = self.node_key()?;
        if !Arc::ptr_eq(&self.tree, &reference_pose.tree) {
            return Err(CartesianTreeError::DifferentTrees(
                self.name(),
                reference_pose
                    .frame()
                    .map_or_else(|| "<removed>".to_owned(), |frame| frame.name()),
            ));
        }

        let desired_pose = Isometry3::from_parts(
            Translation3::from(desired_position),
            desired_orientation.into().as_quaternion(),
        );

        let mut guard = self.write();
        let reference_anchor = reference_pose.anchor.anchor();
        let ancestor = guard.lca(key, reference_anchor)?.ok_or_else(|| {
            CartesianTreeError::NoCommonAncestor(
                guard.name_of(key),
                guard.name_of(reference_anchor),
            )
        })?;

        let t_pose_to_reference_anchor =
            reference_pose.anchor.offset() * reference_pose.transformation();
        let t_pose_to_ancestor =
            guard.transform_up(reference_anchor, t_pose_to_reference_anchor, ancestor)?;
        let t_parent_to_ancestor = guard.transform_up(key, Isometry3::identity(), ancestor)?;

        let t_calibrated_to_parent =
            t_parent_to_ancestor.inverse() * t_pose_to_ancestor * desired_pose.inverse();

        let child_key = guard.add_child_node(key, name.into(), t_calibrated_to_parent)?;
        Ok(Self {
            tree: Arc::clone(&self.tree),
            kind: FrameKind::Node(child_key),
        })
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
        Pose::new(
            Arc::clone(&self.tree),
            self.kind.clone(),
            position,
            orientation,
        )
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
    /// - On serialization failure.
    /// - The frame has been removed from its tree, or is a derived frame.
    pub fn to_json(&self) -> Result<String, CartesianTreeError> {
        let key = self.node_key()?;
        let guard = self.read();
        let serial = to_serial(&guard, key)?;
        Ok(serde_json::to_string_pretty(&serial)?)
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
    /// - The frame has been removed from its tree, or is a derived frame.
    ///
    pub fn apply_config(&self, json: &str) -> Result<(), CartesianTreeError> {
        let key = self.node_key()?;
        let serial: SerialFrame = serde_json::from_str(json)?;
        let mut guard = self.write();
        apply_serial(&mut guard, key, &serial)
    }

    /// Creates a derived frame that coincides with this frame moved by `isometry`,
    /// where `isometry` is interpreted in this frame's parent coordinates
    /// (like [`Frame::apply_in_parent_frame`]).
    ///
    /// For root frames, the parent coordinates are the root's own coordinates.
    fn derive_in_parent_frame(&self, isometry: &Isometry3<f64>) -> Self {
        let (anchor, offset) = match &self.kind {
            FrameKind::Node(key) => {
                // The derived frame is anchored to this node, so conjugate the
                // parent-frame motion by this node's own transform (identity for roots).
                let transform = self
                    .read()
                    .nodes
                    .get(*key)
                    .map_or_else(Isometry3::identity, |node| node.transform_to_parent);
                (*key, transform.inverse() * isometry * transform)
            }
            FrameKind::Derived { anchor, offset, .. } => (*anchor, isometry * offset),
        };
        Self {
            tree: Arc::clone(&self.tree),
            kind: FrameKind::Derived {
                anchor,
                offset,
                name: Uuid::new_v4().to_string(),
            },
        }
    }

    /// Creates a derived frame that coincides with this frame moved by `isometry`,
    /// where `isometry` is interpreted in this frame's own coordinates
    /// (like [`Frame::apply_in_local_frame`]).
    fn derive_in_local_frame(&self, isometry: &Isometry3<f64>) -> Self {
        let (anchor, offset) = match &self.kind {
            FrameKind::Node(key) => (*key, *isometry),
            FrameKind::Derived { anchor, offset, .. } => (*anchor, offset * isometry),
        };
        Self {
            tree: Arc::clone(&self.tree),
            kind: FrameKind::Derived {
                anchor,
                offset,
                name: Uuid::new_v4().to_string(),
            },
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct SerialFrame {
    name: String,
    position: Vector3<f64>,
    // Deserialized as a plain quaternion because nalgebra does not re-normalize
    // `UnitQuaternion`s on deserialization; validated in `apply_serial`.
    orientation: Quaternion<f64>,
    children: Vec<Self>,
}

fn to_serial(inner: &TreeInner, key: NodeKey) -> Result<SerialFrame, CartesianTreeError> {
    let node = inner.node(key)?;
    let (position, orientation) = if node.parent.is_some() {
        (
            node.transform_to_parent.translation.vector,
            node.transform_to_parent.rotation.into_inner(),
        )
    } else {
        (Vector3::zeros(), Quaternion::identity())
    };
    Ok(SerialFrame {
        name: node.name.clone(),
        position,
        orientation,
        children: node
            .children
            .iter()
            .map(|&child| to_serial(inner, child))
            .collect::<Result<_, _>>()?,
    })
}

fn apply_serial(
    inner: &mut TreeInner,
    key: NodeKey,
    serial: &SerialFrame,
) -> Result<(), CartesianTreeError> {
    let node = inner.node(key)?;
    if node.name != serial.name {
        return Err(CartesianTreeError::Mismatch(format!(
            "Frame names do not match: {} vs {}",
            node.name, serial.name
        )));
    }

    // only update if frame has parent
    if node.parent.is_some() {
        let orientation = UnitQuaternion::try_new(serial.orientation, MIN_QUATERNION_NORM)
            .ok_or_else(|| {
                let q = &serial.orientation;
                CartesianTreeError::InvalidQuaternion(q.i, q.j, q.k, q.w)
            })?;
        inner.node_mut(key)?.transform_to_parent =
            Isometry3::from_parts(Translation3::from(serial.position), orientation);
    }

    for potential_child in &serial.children {
        let children = inner.node(key)?.children.clone();
        let matching = children.iter().copied().find(|&child| {
            inner
                .nodes
                .get(child)
                .is_some_and(|node| node.name == potential_child.name)
        });
        if let Some(child_key) = matching {
            apply_serial(inner, child_key, potential_child)?;
        }
    }

    Ok(())
}

/// Creates a new derived frame translated by `rhs`, interpreted in the
/// parent frame of `self` (matching the `Pose` operator semantics).
///
/// The derived frame resolves transforms through `self` but is not stored in the tree:
/// it does not appear in `children()` or serialization, is read-only, and is freed when
/// dropped.
impl Add<LazyTranslation> for &Frame {
    type Output = Frame;

    fn add(self, rhs: LazyTranslation) -> Self::Output {
        self.derive_in_parent_frame(&rhs.inner)
    }
}

/// Creates a new derived frame translated by the inverse of `rhs`, interpreted
/// in the parent frame of `self` (matching the `Pose` operator semantics).
///
/// The derived frame resolves transforms through `self` but is not stored in the tree:
/// it does not appear in `children()` or serialization, is read-only, and is freed when
/// dropped.
impl Sub<LazyTranslation> for &Frame {
    type Output = Frame;

    fn sub(self, rhs: LazyTranslation) -> Self::Output {
        self.derive_in_parent_frame(&rhs.inner.inverse())
    }
}

/// Creates a new derived frame rotated by `rhs` about the axes of `self`
/// (local frame, matching the `Pose` operator semantics).
///
/// The derived frame resolves transforms through `self` but is not stored in the tree:
/// it does not appear in `children()` or serialization, is read-only, and is freed when
/// dropped.
impl Mul<LazyRotation> for &Frame {
    type Output = Frame;

    fn mul(self, rhs: LazyRotation) -> Self::Output {
        self.derive_in_local_frame(&rhs.inner)
    }
}

impl HasParent for Frame {
    type Node = Self;

    fn parent(&self) -> Option<Self::Node> {
        let guard = self.read();
        let parent_key = match &self.kind {
            FrameKind::Node(key) => guard.nodes.get(*key)?.parent?,
            FrameKind::Derived { anchor, .. } => {
                if !guard.contains(*anchor) {
                    return None;
                }
                *anchor
            }
        };
        drop(guard);
        Some(Self {
            tree: Arc::clone(&self.tree),
            kind: FrameKind::Node(parent_key),
        })
    }
}

impl NodeEquality for Frame {
    fn is_same(&self, other: &Self) -> bool {
        if !Arc::ptr_eq(&self.tree, &other.tree) {
            return false;
        }
        match (&self.kind, &other.kind) {
            (FrameKind::Node(own), FrameKind::Node(other)) => own == other,
            (FrameKind::Derived { name: own, .. }, FrameKind::Derived { name: other, .. }) => {
                own == other
            }
            _ => false,
        }
    }
}

impl HasChildren for Frame {
    type Node = Self;
    fn children(&self) -> Vec<Self> {
        let FrameKind::Node(key) = &self.kind else {
            return Vec::new();
        };
        let guard = self.read();
        let Some(node) = guard.nodes.get(*key) else {
            return Vec::new();
        };
        node.children
            .iter()
            .map(|&child| Self {
                tree: Arc::clone(&self.tree),
                kind: FrameKind::Node(child),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::lazy_access::{rx, ry, rz, x, y, z};

    use super::*;
    use crate::tree::Walking;
    use approx::assert_relative_eq;
    use nalgebra::{UnitQuaternion, Vector3};

    #[test]
    fn frame_and_pose_are_send_and_sync() {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Frame>();
        assert_send_sync::<Pose>();
    }

    #[test]
    fn create_origin_frame() {
        let root = Frame::new_origin("world");
        assert_eq!(root.name(), "world");
        assert!(root.parent().is_none());
        assert!(root.children().is_empty());
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

        assert_eq!(root.children().len(), 1);
        assert_eq!(child.name(), "dummy");
        assert_eq!(child.parent().unwrap().name(), "world");
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

        assert_eq!(child.name(), "dummy");

        let rotation = child.transformation().unwrap().rotation;
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

        assert_eq!(child.position().unwrap(), Vector3::new(0.0, 0.0, 1.0));
        assert_eq!(
            child.orientation().unwrap().as_quaternion(),
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

        assert_eq!(root.children().len(), 2);
        assert_eq!(a.parent().unwrap().name(), "world");
        assert_eq!(b.parent().unwrap().name(), "world");
    }

    #[test]
    fn test_remove_child() {
        let root = Frame::new_origin("root");
        let child = root
            .add_child(
                "child",
                Vector3::new(1.0, 0.0, 0.0),
                UnitQuaternion::identity(),
            )
            .unwrap();
        let grandchild = child
            .add_child("grandchild", Vector3::zeros(), UnitQuaternion::identity())
            .unwrap();

        root.remove_child("child").unwrap();
        assert!(root.children().is_empty());

        // Handles to removed frames (including the subtree) become stale.
        assert!(matches!(
            child.transformation(),
            Err(CartesianTreeError::FrameRemoved)
        ));
        assert!(matches!(
            grandchild.transformation(),
            Err(CartesianTreeError::FrameRemoved)
        ));
        assert_eq!(child.name(), "<removed>");
        assert!(child.parent().is_none());

        // The name becomes available again.
        assert!(
            root.add_child("child", Vector3::zeros(), UnitQuaternion::identity())
                .is_ok()
        );

        // Unknown names are rejected.
        assert!(matches!(
            root.remove_child("unknown"),
            Err(CartesianTreeError::ChildNotFound(..))
        ));
    }

    #[test]
    fn test_tree_stays_alive_through_any_handle() {
        let leaf = {
            let root = Frame::new_origin("root");
            let mid = root
                .add_child(
                    "mid",
                    Vector3::new(1.0, 0.0, 0.0),
                    UnitQuaternion::identity(),
                )
                .unwrap();
            mid.add_child(
                "leaf",
                Vector3::new(0.0, 2.0, 0.0),
                UnitQuaternion::identity(),
            )
            .unwrap()
        }; // All other handles are dropped here; the leaf keeps the tree alive.

        assert_eq!(leaf.root().name(), "root");
        assert_eq!(leaf.depth(), 2);
        assert!(leaf.transformation().is_ok());

        let leaf_in_root = leaf
            .add_pose(Vector3::zeros(), UnitQuaternion::identity())
            .in_frame(&leaf.root())
            .unwrap()
            .transformation();
        assert_relative_eq!(
            leaf_in_root.translation.vector,
            Vector3::new(1.0, 2.0, 0.0),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_threaded_access() {
        let root = Frame::new_origin("root");
        let child = root
            .add_child("child", Vector3::zeros(), UnitQuaternion::identity())
            .unwrap();

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let child = child.clone();
                let root = root.clone();
                std::thread::spawn(move || {
                    for j in 0..100 {
                        child
                            .set(
                                Vector3::new(f64::from(j), 0.0, f64::from(i)),
                                UnitQuaternion::identity(),
                            )
                            .unwrap();
                        let pose = root.add_pose(Vector3::zeros(), UnitQuaternion::identity());
                        pose.in_frame(&child).unwrap();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
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
    fn test_chained_lazy_frames_survive_intermediate_drop() {
        let root = Frame::new_origin("root");
        let derived = {
            let intermediate = &root + z(5.0);
            &intermediate - y(3.0)
        }; // The intermediate frame handle is dropped here.

        let derived_in_root = derived
            .add_pose(Vector3::zeros(), UnitQuaternion::identity())
            .in_frame(&root)
            .unwrap()
            .transformation();
        assert_relative_eq!(
            derived_in_root.translation.vector,
            Vector3::new(0.0, -3.0, 5.0),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_derived_frames_are_read_only() {
        let root = Frame::new_origin("root");
        let derived = &root + z(5.0);

        assert!(matches!(
            derived.set(Vector3::zeros(), UnitQuaternion::identity()),
            Err(CartesianTreeError::DerivedFrameUnsupported(_))
        ));
        assert!(matches!(
            derived.add_child("child", Vector3::zeros(), UnitQuaternion::identity()),
            Err(CartesianTreeError::DerivedFrameUnsupported(_))
        ));
        assert!(matches!(
            derived.to_json(),
            Err(CartesianTreeError::DerivedFrameUnsupported(_))
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
            Err(CartesianTreeError::DifferentTrees(..))
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
        // non-identity transform, so the composition order actually matters here.
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
        let q = child.orientation().unwrap().as_quaternion();
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
    fn test_lazy_ops_do_not_register_children() {
        let root = Frame::new_origin("root");
        let child = root
            .add_child("child", Vector3::zeros(), UnitQuaternion::identity())
            .unwrap();

        let derived = &child + z(5.0);
        let rotated = &child * rz(0.5);

        // Derived frames must not accumulate in the tree.
        assert!(child.children().is_empty());
        assert_eq!(root.children().len(), 1);

        // They still resolve transforms through the parent chain.
        let derived_in_root = derived
            .add_pose(Vector3::zeros(), UnitQuaternion::identity())
            .in_frame(&root)
            .unwrap()
            .transformation();
        assert_relative_eq!(
            derived_in_root.translation.vector,
            Vector3::new(0.0, 0.0, 5.0),
            epsilon = 1e-10
        );
        drop(rotated);
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

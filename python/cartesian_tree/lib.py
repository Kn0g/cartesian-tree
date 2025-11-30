"""Defines a the main lib classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base_types import Isometry, Rotation, Vector3
from cartesian_tree import _cartesian_tree as _core  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from .lazy_access import LazyRotation, LazyTranslation


class Frame:
    """Defines a coordinate frame in a Cartesian tree structure.

    Each frame can have one parent and multiple children. The frame stores its
    transformation (position and orientation) relative to its parent.
    """

    def __init__(self, name: str) -> None:
        """Initializes a new root frame (origin) with the given name.

        Args:
            name: The name of the root frame.
        """
        self._core_frame = _core.Frame(name)

    @property
    def name(self) -> str:
        """The name of the frame."""
        return self._core_frame.name

    @property
    def depth(self) -> int:
        """The depth from the frame to its root."""
        return self._core_frame.depth

    def add_child(self, name: str, position: Vector3, orientation: Rotation) -> Frame:
        """Adds a new child frame to the current frame.

        Args:
            name: The name of the new child frame.
            position: The translational offset from the parent.
            orientation: The orientational offset from the parent.

        Returns:
            The newly created child frame.

        Raises:
            ValueError: If a child with the same name already exists.
        """
        binding_frame = self._core_frame.add_child(name, position._binding_structure, orientation._binding_structure)
        return Frame._from_rust(binding_frame)

    def calibrate_child(
        self, name: str, desired_position: Vector3, desired_orientation: Rotation, reference_pose: Pose
    ) -> Frame:
        """Adds a child frame such that a reference pose, expressed in the new frame, matches the desired isometry.

        Args:
            name: The name of the new child frame.
            desired_position: The desired position of the reference pose in the new frame.
            desired_orientation: The desired orientation of the reference pose in the new frame.
            reference_pose: The reference pose for calibration.

        Returns:
            The newly created child frame.

        Raises:
            ValueError: If a child with the same name already exists.
        """
        binding_frame = self._core_frame.calibrate_child(
            name,
            desired_position._binding_structure,
            desired_orientation._binding_structure,
            reference_pose._binding_structure,
        )
        return Frame._from_rust(binding_frame)

    def add_pose(self, position: Vector3, orientation: Rotation) -> Pose:
        """Adds a pose to the current frame.

        Args:
            position: The translational part of the pose.
            orientation: The orientational part of the pose.

        Returns:
            The newly created pose.
        """
        binding_pose = self._core_frame.add_pose(position._binding_structure, orientation._binding_structure)
        return Pose._from_rust(binding_pose)

    def transformation_to_parent(self) -> tuple[Vector3, Rotation]:
        """Returns the transformation from this frame to its parent frame.

        Returns:
            The transformation from this frame to its parent frame (translation, rotation).

        Raises:
            ValueError: If the frame has no parent.
        """
        binding_position, binding_rotation = self._core_frame.transformation_to_parent()
        return (
            Vector3(*binding_position.to_tuple()),
            Rotation._from_rust(binding_rotation),
        )

    def set(self, position: Vector3, orientation: Rotation) -> None:
        """Sets the frames transformation relative to its parent.

        Args:
            position: The translational offset from the parent.
            orientation: The orientational offset from the parent.

        Raises:
            ValueError: If the frame has no parent.
        """
        self._core_frame.set(position._binding_structure, orientation._binding_structure)

    def apply_in_parent_frame(self, isometry: Isometry) -> None:
        """Applies the provided isometry interpreted in the parent frame to this frame.

        Args:
            isometry: The isometry (describing a motion in the parent frame coordinates) to
                apply to the current transformation.

        Raises:
            ValueError: If the frame has no parent.
        """
        self._core_frame.apply_in_parent_frame(isometry._binding_structure)

    def apply_in_local_frame(self, isometry: Isometry) -> None:
        """Applies the provided isometry interpreted in this frame to this frame.

        Args:
            isometry: The isometry (describing a motion in this frame) to apply to the current
                transformation.

        Raises:
            ValueError: If the frame has no parent.
        """
        self._core_frame.apply_in_local_frame(isometry._binding_structure)

    def to_json(self) -> str:
        """Serializes the frame tree to a JSON string.

        Returns:
            The JSON representation of the tree.

        Raises:
            ValueError: On serialization failure.
        """
        return self._core_frame.to_json()

    def apply_config(self, config_json: str) -> None:
        """Applies a JSON config to update matching transforms in the tree.

        Deserializes the JSON to a temporary structure, then recursively updates transforms to the
        parent frames where names match (partial apply; ignores unmatched frames in config).
        It is assumed this frame is the root.

        Args:
            config_json: The JSON string to apply.

        Raises:
            ValueError: On deserialization or mismatch errors (e.g. if this frame is not the root).
        """
        self._core_frame.apply_config(config_json)

    def parent(self) -> Frame | None:
        """Returns the parent of the frame.

        Returns:
            The parent of the frame.
        """
        binding_parent = self._core_frame.parent()
        if binding_parent is None:
            return None
        return Frame._from_rust(binding_parent)

    def root(self) -> Frame:
        """Returns the root frame of the tree.

        Returns:
            The root frame of the tree.
        """
        binding_root = self._core_frame.root()
        return Frame._from_rust(binding_root)

    def children(self) -> list[Frame]:
        """Returns the children of the frame.

        Returns:
            The children of the frame.
        """
        return [Frame._from_rust(binding_child) for binding_child in self._core_frame.children()]

    def __add__(self, lazy_access: LazyTranslation) -> Frame:
        return Frame._from_rust(self._core_frame + lazy_access.inner)

    def __sub__(self, lazy_access: LazyTranslation) -> Frame:
        return Frame._from_rust(self._core_frame - lazy_access.inner)

    def __mul__(self, lazy_access: LazyRotation) -> Frame:
        return Frame._from_rust(self._core_frame * lazy_access.inner)

    def __str__(self) -> str:
        return self._core_frame.__str__()

    def __repr__(self) -> str:
        return self._core_frame.__repr__()

    @property
    def _binding_structure(self) -> Any:
        return self._core_frame

    @classmethod
    def _from_rust(cls, rust_frame: _core.Frame) -> Frame:
        instance = cls.__new__(cls)
        instance._core_frame = rust_frame
        return instance


class Pose:
    """Defines a Cartesian pose."""

    _core_pose: _core.Pose

    def frame(self) -> Frame:
        """Returns the frame of the pose."""
        return Frame._from_rust(self._core_pose.frame())

    def transformation(self) -> tuple[Vector3, Rotation]:
        """Returns the transformation of the pose to its parent frame.

        Returns:
            The transformation from this frame to its parent frame (position, rotation).
        """
        binding_position, binding_rotation = self._core_pose.transformation()
        return (
            Vector3(*binding_position.to_tuple()),
            Rotation._from_rust(binding_rotation),
        )

    def set(self, position: Vector3, orientation: Rotation) -> None:
        """Sets the pose's transformation.

        Args:
            position: The translational part of the pose.
            orientation: The orientational part of the pose.
        """
        self._core_pose.set(position._binding_structure, orientation._binding_structure)

    def apply_in_parent_frame(self, isometry: Isometry) -> None:
        """Applies the provided isometry interpreted in the parent frame to this pose.

        Args:
            isometry: The isometry (describing a motion in the parent frame coordinates) to
                apply to the current transformation.
        """
        self._core_pose.apply_in_parent_frame(isometry._binding_structure)

    def apply_in_local_frame(self, isometry: Isometry) -> None:
        """Applies the provided isometry interpreted in the body frame to this pose.

        Args:
            isometry: The isometry (describing a motion in this frame) to apply to the current
                transformation.

        Raises:
            ValueError: If the frame has no parent.
        """
        self._core_pose.apply_in_local_frame(isometry._binding_structure)

    def in_frame(self, target_frame: Frame) -> Pose:
        """Transforms this pose into the coordinate system of the given target frame.

        Args:
            target_frame: The frame to express this pose in.

        Returns:
            This pose in the new frame.
        """
        binding_pose = self._core_pose.in_frame(target_frame._binding_structure)
        return Pose._from_rust(binding_pose)

    @property
    def _binding_structure(self) -> Any:
        return self._core_pose

    @classmethod
    def _from_rust(cls, rust_pose: _core.Pose) -> Pose:
        instance = cls.__new__(cls)
        instance._core_pose = rust_pose
        return instance

    def __add__(self, lazy_access: LazyTranslation) -> Pose:
        return Pose._from_rust(self._core_pose + lazy_access.inner)

    def __sub__(self, lazy_access: LazyTranslation) -> Pose:
        return Pose._from_rust(self._core_pose - lazy_access.inner)

    def __mul__(self, lazy_access: LazyRotation) -> Pose:
        return Pose._from_rust(self._core_pose * lazy_access.inner)

    def __str__(self) -> str:
        return self._core_pose.__str__()

    def __repr__(self) -> str:
        return self._core_pose.__repr__()

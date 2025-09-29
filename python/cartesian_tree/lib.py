"""Defines a the main lib classes."""

from __future__ import annotations

from typing import Any

from .helper import RPY, Position, Quaternion
from cartesian_tree import _cartesian_tree as _core  # type: ignore[attr-defined]


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

    def add_child(self, name: str, position: Position, orientation: RPY | Quaternion) -> Frame:
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
        if isinstance(orientation, RPY):
            orientation = orientation.to_quaternion()

        binding_frame = self._core_frame.add_child(name, position._binding_structure, orientation._binding_structure)
        return Frame._from_rust(binding_frame)

    def calibrate_child(
        self, name: str, desired_position: Position, desired_orientation: RPY | Quaternion, reference_pose: Pose
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
        if isinstance(desired_orientation, RPY):
            desired_orientation = desired_orientation.to_quaternion()

        binding_frame = self._core_frame.calibrate_child(
            name,
            desired_position._binding_structure,
            desired_orientation._binding_structure,
            reference_pose._binding_structure,
        )
        return Frame._from_rust(binding_frame)

    def add_pose(self, position: Position, orientation: RPY | Quaternion) -> Pose:
        """Adds a pose to the current frame.

        Args:
            position: The translational part of the pose.
            orientation: The orientational part of the pose.

        Returns:
            The newly created pose.
        """
        if isinstance(orientation, RPY):
            orientation = orientation.to_quaternion()

        binding_pose = self._core_frame.add_pose(position._binding_structure, orientation._binding_structure)
        return Pose._from_rust(binding_pose)

    def transformation_to_parent(self) -> tuple[Position, Quaternion]:
        """Returns the transformation from this frame to its parent frame.

        Returns:
            The transformation from this frame to its parent frame (Position, Quaternion(x, y, z, w)).

        Raises:
            ValueError: If the frame has no parent.
        """
        binding_position, binding_quat = self._core_frame.transformation_to_parent()
        return (
            Position(*binding_position.to_tuple()),
            Quaternion(*binding_quat.to_tuple()),
        )

    def update_transformation(self, position: Position, orientation: RPY | Quaternion) -> None:
        """Updates the frames transformation relative to its parent.

        Args:
            position: The translational offset from the parent.
            orientation: The orientational offset from the parent.

        Raises:
            ValueError: If the frame has no parent or invalid dimensions.
        """
        if isinstance(orientation, RPY):
            orientation = orientation.to_quaternion()
        self._core_frame.update_transformation(position._binding_structure, orientation._binding_structure)

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

    def transformation(self) -> tuple[Position, Quaternion]:
        """Returns the transformation of the pose to its parent frame.

        Returns:
            The transformation from this frame to its parent frame (position, quaternion(x, y, z, w)).
        """
        binding_position, binding_quat = self._core_pose.transformation()
        return (
            Position(*binding_position.to_tuple()),
            Quaternion(*binding_quat.to_tuple()),
        )

    def update(self, position: Position, orientation: RPY | Quaternion) -> None:
        """Updates the pose's transformation.

        Args:
            position: The translational part of the pose.
            orientation: The orientational part of the pose.
        """
        if isinstance(orientation, RPY):
            orientation = orientation.to_quaternion()
        self._core_pose.update(position._binding_structure, orientation._binding_structure)

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

    def __str__(self) -> str:
        return self._core_pose.__str__()

    def __repr__(self) -> str:
        return self._core_pose.__repr__()

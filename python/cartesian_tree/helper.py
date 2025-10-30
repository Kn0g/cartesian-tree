"""Defines helper classes for a more pythonic API."""

from __future__ import annotations

from typing import Any

from cartesian_tree import _cartesian_tree as _core  # type: ignore[attr-defined]


class RPY:
    """Defines a roll-pitch-yaw angle representation."""

    def __init__(self, roll: float, pitch: float, yaw: float) -> None:
        """Initializes the roll-pitch-yaw angles.

        Args:
            roll: The roll angle in radians.
            pitch: The pitch angle in radians.
            yaw: The yaw angle in radians
        """
        self._core_rpy = _core.RPY(roll, pitch, yaw)

    @property
    def roll(self) -> float:
        """The roll angle in radians."""
        return self._core_rpy.roll

    @property
    def pitch(self) -> float:
        """The pitch angle in radians."""
        return self._core_rpy.pitch

    @property
    def yaw(self) -> float:
        """The yaw angle in radians."""
        return self._core_rpy.yaw

    @property
    def _binding_structure(self) -> Any:
        return self._core_rpy

    def to_list(self) -> list[float]:
        """Returns the angles as list.

        Returns:
            The angle as list.
        """
        return [self.roll, self.pitch, self.yaw]

    def to_tuple(self) -> tuple[float, float, float]:
        """Returns the angles as tuple.

        Returns:
            The angles as tuple.
        """
        return (self.roll, self.pitch, self.yaw)

    def to_rotation(self) -> Rotation:
        """Converts the angles as unified rotation.

        Returns:
            The angles as unified rotation.
        """
        return Rotation.from_rpy(self.roll, self.pitch, self.yaw)

    def __str__(self) -> str:
        return self._core_rpy.__str__()

    def __repr__(self) -> str:
        return self._core_rpy.__repr__()


class Quaternion:
    """Defines a quaternion."""

    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        """Initializes the quaternion.

        Args:
            x: The x value.
            y: The y value.
            z: The z value.
            w: The w value.
        """
        self._core_quaternion = _core.Quaternion(x, y, z, w)

    @property
    def x(self) -> float:
        """The x value."""
        return self._core_quaternion.x

    @property
    def y(self) -> float:
        """The y value."""
        return self._core_quaternion.y

    @property
    def z(self) -> float:
        """The z value."""
        return self._core_quaternion.z

    @property
    def w(self) -> float:
        """The z value."""
        return self._core_quaternion.w

    @property
    def _binding_structure(self) -> Any:
        return self._core_quaternion

    def to_list(self) -> list[float]:
        """Returns the quaternion as list in the form x,y,z and w.

        Returns:
            The quaternion as list.
        """
        return [self.x, self.y, self.z, self.w]

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Returns the quaternion as tuple in the form x,y,z and w.

        Returns:
            The quaternion as tuple.
        """
        return (self.x, self.y, self.z, self.w)

    def to_rotation(self) -> Rotation:
        """Converts the quaternion to unified rotation.

        Returns:
            The angles as unified rotation.
        """
        return Rotation.from_quaternion(self.x, self.y, self.z, self.w)

    def __str__(self) -> str:
        return self._core_quaternion.__str__()

    def __repr__(self) -> str:
        return self._core_quaternion.__repr__()


class Rotation:
    """Defines a unified rotation representation."""

    _core_rotation: _core.Rotation

    @classmethod
    def from_quaternion(cls, x: float, y: float, z: float, w: float) -> Rotation:
        """Initializes the rotation from quaternion values.

        Args:
            x: The x value.
            y: The y value.
            z: The z value.
            w: The w value.

        Returns:
            The initialized instance.
        """
        instance = cls.__new__(cls)
        instance._core_rotation = _core.Rotation.from_quat(x, y, z, w)
        return instance

    @classmethod
    def from_rpy(cls, roll: float, pitch: float, yaw: float) -> Rotation:
        """Initializes the rotation from RPY values.

        Args:
            roll: The roll value.
            pitch: The pitch value.
            yaw: The yaw value.

        Returns:
            The initialized instance.
        """
        instance = cls.__new__(cls)
        instance._core_rotation = _core.Rotation.from_rpy(roll, pitch, yaw)
        return instance

    def to_quaternion(self) -> Quaternion:
        """Converts the rotation to quaternion.

        Returns:
            The quaternion representation of the rotation.
        """
        return Quaternion(*self._core_rotation.to_quat().to_tuple())

    def to_rpy(self) -> RPY:
        """Converts the rotation to RPY.

        Returns:
            The RPY representation of the rotation.
        """
        return RPY(*self._core_rotation.to_rpy().to_tuple())

    @property
    def _binding_structure(self) -> Any:
        return self._core_rotation

    @classmethod
    def _from_rust(cls, rust_rotation: _core.Rotation) -> Rotation:
        instance = cls.__new__(cls)
        instance._core_rotation = rust_rotation
        return instance

    def __str__(self) -> str:
        return self._core_rotation.__str__()

    def __repr__(self) -> str:
        return self._core_rotation.__repr__()


class Vector3:
    """Defines a vector in Cartesian space."""

    def __init__(self, x: float, y: float, z: float) -> None:
        """Initializes the vector.

        Args:
            x: The x value.
            y: The y value.
            z: The z value.
        """
        self._core_vector = _core.Vector3(x, y, z)

    @property
    def x(self) -> float:
        """The x value."""
        return self._core_vector.x

    @property
    def y(self) -> float:
        """The y value."""
        return self._core_vector.y

    @property
    def z(self) -> float:
        """The z value."""
        return self._core_vector.z

    @property
    def _binding_structure(self) -> Any:
        return self._core_vector

    def to_list(self) -> list[float]:
        """Returns the vector as list.

        Returns:
            The vector as list.
        """
        return [self.x, self.y, self.z]

    def to_tuple(self) -> tuple[float, float, float]:
        """Returns the vector as tuple.

        Returns:
            The vector as tuple.
        """
        return (self.x, self.y, self.z)

    def __str__(self) -> str:
        return self._core_vector.__str__()

    def __repr__(self) -> str:
        return self._core_vector.__repr__()

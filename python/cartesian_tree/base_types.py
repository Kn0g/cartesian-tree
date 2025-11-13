"""Defines helper classes for a more pythonic API."""

from __future__ import annotations

from typing import Any

from .angles import RPY
from .quaternion import Quaternion
from cartesian_tree import _cartesian_tree as _core  # type: ignore[attr-defined]


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
        instance._core_rotation = _core.Rotation.from_quaternion(x, y, z, w)
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

    def as_quaternion(self) -> Quaternion:
        """Converts the rotation to quaternion.

        Returns:
            The quaternion representation of the rotation.
        """
        return Quaternion._from_rust(self._core_rotation)

    def as_rpy(self) -> RPY:
        """Converts the rotation to RPY.

        Returns:
            The RPY representation of the rotation.
        """
        return RPY._from_rust(self._core_rotation)

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

    def as_list(self) -> list[float]:
        """Returns the vector as list.

        Returns:
            The vector as list.
        """
        return [self.x, self.y, self.z]

    def as_tuple(self) -> tuple[float, float, float]:
        """Returns the vector as tuple.

        Returns:
            The vector as tuple.
        """
        return (self.x, self.y, self.z)

    def __str__(self) -> str:
        return self._core_vector.__str__()

    def __repr__(self) -> str:
        return self._core_vector.__repr__()

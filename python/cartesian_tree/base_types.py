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

    @classmethod
    def identity(cls) -> Rotation:
        """Initializes the identity rotation."""
        instance = cls.__new__(cls)
        instance._core_rotation = _core.Rotation.identity()
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

    @classmethod
    def zeros(cls) -> Vector3:
        """Initializes the zero vector.

        Returns:
            The zero vector.
        """
        return cls(0.0, 0.0, 0.0)

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


class Isometry:
    """Rigid 3D transformation."""

    _core_isometry: _core.Isometry

    @classmethod
    def identity(cls) -> Isometry:
        """Initializes the identity isometry.

        This isometry applies the rotation R with its axis passing through the point P.
        This effectively lets P invariant.
        """
        instance = cls.__new__(cls)
        instance._core_isometry = _core.Isometry.identity()
        return instance

    @classmethod
    def from_translation(cls, translation: Vector3) -> Isometry:
        """Initializes the isometry from translation only.

        Note, the rotation will be identity.

        Args:
            translation: The translation part.

        Returns:
            The initialized isometry instance.
        """
        instance = cls.__new__(cls)
        instance._core_isometry = _core.Isometry.from_translation(translation._binding_structure)
        return instance

    @classmethod
    def from_rotation(cls, rotation: Rotation) -> Isometry:
        """Initializes the isometry from rotation only.

        Note, the translation will be identity.

        Args:
            rotation: The rotation part.

        Returns:
            The initialized isometry instance.
        """
        instance = cls.__new__(cls)
        instance._core_isometry = _core.Isometry.from_rotation(rotation._binding_structure)
        return instance

    @classmethod
    def from_parts(cls, translation: Vector3, rotation: Rotation) -> Isometry:
        """Initializes the isometry from translation and rotation.

        Args:
            translation: The translation part.
            rotation: The rotation part.

        Returns:
            The initialized isometry instance.
        """
        instance = cls.__new__(cls)
        instance._core_isometry = _core.Isometry.from_parts(translation._binding_structure, rotation._binding_structure)
        return instance

    def decompose(self) -> tuple[Vector3, Rotation]:
        """Decomposes the isometry into translation and rotation.

        Returns:
            The translation and rotation.
        """
        binding_pos, binding_rot = self._core_isometry.decompose()
        return Vector3(*binding_pos.to_tuple()), Rotation._from_rust(binding_rot)

    def translation(self) -> Vector3:
        """Returns the translation part of the isometry.

        Returns:
            The translation part.
        """
        binding_translation = self._core_isometry.translation()
        return Vector3(*binding_translation.to_tuple())

    def rotation(self) -> Rotation:
        """Returns the rotation part of the isometry.

        Returns:
            The rotation part.
        """
        binding_rotation = self._core_isometry.rotation()
        return Rotation._from_rust(binding_rotation)

    def inverse(self) -> Isometry:
        """Returns the inverse of the isometry.

        Returns:
            The inverse isometry.
        """
        return Isometry._from_rust(self._core_isometry.inverse())

    def __mul__(self, other: Isometry) -> Isometry:
        return Isometry._from_rust(self._core_isometry.__mul__(other._binding_structure))

    @property
    def _binding_structure(self) -> Any:
        return self._core_isometry

    @classmethod
    def _from_rust(cls, rust_isometry: _core.Isometry) -> Isometry:
        instance = cls.__new__(cls)
        instance._core_isometry = rust_isometry
        return instance

    def __str__(self) -> str:
        return self._core_isometry.__str__()

    def __repr__(self) -> str:
        return self._core_isometry.__repr__()

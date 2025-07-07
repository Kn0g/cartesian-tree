"""Defines helper classes for a more pythonic API."""

from __future__ import annotations
from cartesian_tree import cartesian_tree as _core


class RPY:
    """Defines a roll-pitch-yaw angle representation."""

    def __init__(self, roll: float, pitch: float, yaw: float):
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
    def _binding_structure(self):
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

    def to_quaternion(self) -> Quaternion:
        """Returns the angles as quaternion.

        Returns:
            The angles as quaternion.
        """
        return Quaternion(*self._core_rpy.to_quaternion().to_tuple())

    def __str__(self) -> str:
        return self._core_rpy.__str__()

    def __repr__(self) -> str:
        return self._core_rpy.__repr__()


class Quaternion:
    """Defines a quaternion."""

    def __init__(self, x: float, y: float, z: float, w: float):
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
    def _binding_structure(self):
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

    def to_rpy(self):
        """Returns the angles as quaternion.

        Returns:
            The angles as quaternion.
        """
        return RPY(*self._core_quaternion.to_rpy().to_tuple())

    def __str__(self) -> str:
        return self._core_quaternion.__str__()

    def __repr__(self) -> str:
        return self._core_quaternion.__repr__()


class Position:
    """Defines a position."""

    def __init__(self, x: float, y: float, z: float):
        """Initializes the position in meter.

        Args:
            x: The x value in meter.
            y: The y value in meter.
            z: The z value in meter.
        """
        self._core_position = _core.Position(x, y, z)

    @property
    def x(self) -> float:
        """The x value in meter."""
        return self._core_position.x

    @property
    def y(self) -> float:
        """The y value in meter."""
        return self._core_position.y

    @property
    def z(self) -> float:
        """The z value in meter."""
        return self._core_position.z

    @property
    def _binding_structure(self):
        return self._core_position

    def to_list(self) -> list[float]:
        """Returns the position as list.

        Returns:
            The position as list.
        """
        return [self.x, self.y, self.z]

    def to_tuple(self) -> tuple[float, float, float]:
        """Returns the position as tuple.

        Returns:
            The position as tuple.
        """
        return (self.x, self.y, self.z)

    def __str__(self) -> str:
        return self._core_position.__str__()

    def __repr__(self) -> str:
        return self._core_position.__repr__()

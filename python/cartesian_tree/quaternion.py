"""Contains a quaternion type."""

from __future__ import annotations

from cartesian_tree import _cartesian_tree as _core  # type: ignore[attr-defined]


class Quaternion:
    """Defines a quaternion."""

    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        """Initializes the quaternion.

        Note, the quaternion will be normalized.

        Args:
            x: The x value.
            y: The y value.
            z: The z value.
            w: The w value.
        """
        self._core_rotation = _core.Rotation.from_quaternion(x, y, z, w)

    @property
    def x(self) -> float:
        """The x value."""
        return self._core_rotation.as_quaternion()[0]

    @property
    def y(self) -> float:
        """The y value."""
        return self._core_rotation.as_quaternion()[1]

    @property
    def z(self) -> float:
        """The z value."""
        return self._core_rotation.as_quaternion()[2]

    @property
    def w(self) -> float:
        """The w value."""
        return self._core_rotation.as_quaternion()[3]

    def vector_part(self) -> tuple[float, float, float]:
        """Returns the vector part of the quaternion.

        Returns:
            The vector part of the quaternion.
        """
        return self._core_rotation.as_quaternion()[0:3]

    def as_list(self) -> list[float]:
        """Returns the quaternion as list in the form x,y,z and w.

        Returns:
            The quaternion as list.
        """
        return list(self._core_rotation.as_quaternion())

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Returns the quaternion as tuple in the form x,y,z and w.

        Returns:
            The quaternion as tuple.
        """
        return self._core_rotation.as_quaternion()

    @classmethod
    def _from_rust(cls, rust_rotation: _core.Rotation) -> Quaternion:
        instance = cls.__new__(cls)
        instance._core_rotation = rust_rotation
        return instance

    def __str__(self) -> str:
        x, y, z, w = self._core_rotation.as_quaternion()
        return f"(<{x}, {y}, {z}>, {w})"

    def __repr__(self) -> str:
        return self.__str__()

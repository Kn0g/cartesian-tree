"""Contains angle representations."""

from __future__ import annotations

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
        self._core_rotation = _core.Rotation.from_rpy(roll, pitch, yaw)

    @classmethod
    def identity(cls) -> RPY:
        """Initializes the identity RPY angles."""
        instance = cls.__new__(cls)
        instance._core_rotation = _core.Rotation.from_rpy(0.0, 0.0, 0.0)
        return instance

    @property
    def roll(self) -> float:
        """The roll angle in radians."""
        return self._core_rotation.as_rpy()[0]

    @property
    def pitch(self) -> float:
        """The pitch angle in radians."""
        return self._core_rotation.as_rpy()[1]

    @property
    def yaw(self) -> float:
        """The yaw angle in radians."""
        return self._core_rotation.as_rpy()[2]

    def as_list(self) -> list[float]:
        """Returns the angles as list.

        Returns:
            The angle as list.
        """
        return list(self._core_rotation.as_rpy())

    def as_tuple(self) -> tuple[float, float, float]:
        """Returns the angles as tuple.

        Returns:
            The angles as tuple.
        """
        return self._core_rotation.as_rpy()

    @classmethod
    def _from_rust(cls, rust_rotation: _core.Rotation) -> RPY:
        instance = cls.__new__(cls)
        instance._core_rotation = rust_rotation
        return instance

    def __str__(self) -> str:
        roll, pitch, yaw = self._core_rotation.as_rpy()
        return f"({roll}, {pitch}, {yaw})"

    def __repr__(self) -> str:
        return self.__str__()

"""Contains tests for the angles module."""

from cartesian_tree import RPY


def test_rpy_properties() -> None:
    rpy = RPY(1.0, 2.0, 3.0)

    assert rpy.roll == 1.0
    assert rpy.pitch == 2.0
    assert rpy.yaw == 3.0


def test_rpy_as_list() -> None:
    rpy = RPY(1.0, 2.0, 3.0)
    rpy_list = rpy.as_list()
    assert isinstance(rpy_list, list)


def test_rpy_as_tuple() -> None:
    rpy = RPY(1.0, 2.0, 3.0)
    rpy_tuple = rpy.as_tuple()
    assert isinstance(rpy_tuple, tuple)

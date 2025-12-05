"""Contains tests for the quaternion module."""

import pytest

from cartesian_tree import Quaternion


def test_quaternion_properties() -> None:
    q = Quaternion(0.0, 0.0, 0.707, 0.707)

    assert q.x == 0.0
    assert q.y == 0.0
    assert q.z == pytest.approx(0.707, abs=1e-3)
    assert q.w == pytest.approx(0.707, abs=1e-3)


def test_quaternion_identity() -> None:
    identity = Quaternion.identity()
    assert identity.x == 0.0
    assert identity.y == 0.0
    assert identity.z == 0.0
    assert identity.w == 1.0


def test_quaternion_vector_part() -> None:
    q = Quaternion(0.0, 0.0, 0.707, 0.707)
    vector = q.vector_part()
    assert vector == (0.0, 0.0, pytest.approx(0.707, abs=1e-3))


def test_quaternion_as_list() -> None:
    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    q_list = q.as_list()
    assert isinstance(q_list, list)


def test_quaternion_as_tuple() -> None:
    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    q_tuple = q.as_tuple()
    assert isinstance(q_tuple, tuple)

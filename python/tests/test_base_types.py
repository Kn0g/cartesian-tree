"""Contains tests for the base types."""

from math import radians

import pytest

from cartesian_tree import Isometry, Rotation, Vector3


def test_vector3_properties() -> None:
    v = Vector3(1.0, 2.0, 3.0)

    assert v.x == 1.0
    assert v.y == 2.0
    assert v.z == 3.0


def test_vector3_as_list() -> None:
    v = Vector3(1.0, 2.0, 3.0)
    v_list = v.as_list()
    assert isinstance(v_list, list)


def test_vector3_as_tuple() -> None:
    v = Vector3(1.0, 2.0, 3.0)
    v_tuple = v.as_tuple()
    assert isinstance(v_tuple, tuple)


def test_rotation_from_rpy() -> None:
    rpy = Rotation.from_rpy(1.0, 42.0, 3.0)
    assert rpy.as_rpy().as_tuple() == pytest.approx((1.0, 42.0, 3.0), abs=1e-5)

    rpy = Rotation.from_quaternion(0.0, 0.0, 0.7071, 0.7071)
    assert rpy.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, radians(90.0)), abs=1e-5)


def test_rotation_from_quaternion() -> None:
    quaternion = Rotation.from_quaternion(0.0, 0.0, 0.7071, 0.7071)
    assert quaternion.as_quaternion().as_tuple() == pytest.approx((0.0, 0.0, 0.7071, 0.7071), abs=1e-5)
    assert quaternion.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, radians(90.0)), abs=1e-5)


def test_build_isometry() -> None:
    isometry = Isometry.identity()
    assert isometry.translation().as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-5)
    assert isometry.rotation().as_rpy().as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-5)

    isometry = Isometry.from_translation(Vector3(1.0, 2.0, 3.0))
    assert isometry.translation().as_tuple() == pytest.approx((1.0, 2.0, 3.0), abs=1e-5)
    assert isometry.rotation().as_rpy().as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-5)

    rotation = Rotation.from_rpy(0.0, 0.0, radians(90.0))
    isometry = Isometry.from_rotation(rotation)
    assert isometry.translation().as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-5)
    assert isometry.rotation().as_rpy().as_tuple() == pytest.approx((0.0, 0.0, radians(90.0)), abs=1e-5)

    isometry = Isometry.from_parts(Vector3(1.0, 2.0, 3.0), rotation)
    assert isometry.translation().as_tuple() == pytest.approx((1.0, 2.0, 3.0), abs=1e-5)
    assert isometry.rotation().as_rpy().as_tuple() == pytest.approx((0.0, 0.0, radians(90.0)), abs=1e-5)


def test_isometry_decomposition() -> None:
    isometry = Isometry.from_parts(Vector3(1.0, 2.0, 3.0), Rotation.from_rpy(0.0, 0.0, radians(90.0)))
    translation, rotation = isometry.decompose()
    assert translation.as_tuple() == pytest.approx((1.0, 2.0, 3.0), abs=1e-5)
    assert rotation.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, radians(90.0)), abs=1e-5)


def test_isometry_inversion() -> None:
    isometry = Isometry.from_parts(Vector3(1.0, 2.0, 3.0), Rotation.from_rpy(0.0, 0.0, radians(90.0)))
    inv_isometry = isometry.inverse()
    translation, rotation = inv_isometry.decompose()
    assert translation.as_tuple() == pytest.approx((-2.0, 1.0, -3.0), abs=1e-5)
    assert rotation.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, radians(-90.0)), abs=1e-5)


def test_isometry_multiplication() -> None:
    isometry_1 = Isometry.from_parts(Vector3(1.0, 0.0, 0.0), Rotation.from_rpy(0.0, 0.0, radians(90.0)))
    isometry_2 = Isometry.from_parts(Vector3(0.0, 1.0, 0.0), Rotation.from_rpy(0.0, 0.0, radians(90.0)))
    result_isometry = isometry_1 * isometry_2
    translation, rotation = result_isometry.decompose()
    assert translation.as_tuple() == pytest.approx((0, 0, 0), abs=1e-5)
    assert rotation.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, radians(180.0)), abs=1e-5)

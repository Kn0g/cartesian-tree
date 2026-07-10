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
    assert v_list == [1.0, 2.0, 3.0]


def test_vector3_as_tuple() -> None:
    v = Vector3(1.0, 2.0, 3.0)
    v_tuple = v.as_tuple()
    assert isinstance(v_tuple, tuple)
    assert v_tuple == (1.0, 2.0, 3.0)


def test_rotation_from_rpy() -> None:
    rpy = Rotation.from_rpy(1.0, 42.0, 3.0)
    assert rpy.as_rpy().as_tuple() == pytest.approx((1.0, 42.0, 3.0), abs=1e-5)

    rpy = Rotation.from_quaternion(0.0, 0.0, 0.7071, 0.7071)
    assert rpy.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, radians(90.0)), abs=1e-5)


def test_rotation_from_quaternion() -> None:
    quaternion = Rotation.from_quaternion(0.0, 0.0, 0.7071, 0.7071)
    assert quaternion.as_quaternion().as_tuple() == pytest.approx((0.0, 0.0, 0.7071, 0.7071), abs=1e-5)
    assert quaternion.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, radians(90.0)), abs=1e-5)


def test_rotation_from_quaternion_normalizes() -> None:
    rotation = Rotation.from_quaternion(0.0, 0.0, 2.0, 0.0)
    assert rotation.as_quaternion().as_tuple() == pytest.approx((0.0, 0.0, 1.0, 0.0), abs=1e-12)


def test_rotation_from_quaternion_rejects_zero_norm() -> None:
    with pytest.raises(ValueError, match="norm is too close to zero"):
        Rotation.from_quaternion(0.0, 0.0, 0.0, 0.0)


def test_rotation_identity() -> None:
    identity = Rotation.identity()
    assert identity.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-5)
    assert identity.as_quaternion().as_tuple() == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-5)


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


def test_vector3_equality_and_hash() -> None:
    assert Vector3(1.0, 2.0, 3.0) == Vector3(1.0, 2.0, 3.0)
    assert Vector3(1.0, 2.0, 3.0) != Vector3(1.0, 2.0, 4.0)
    assert Vector3(1.0, 2.0, 3.0) != (1.0, 2.0, 3.0)
    assert len({Vector3(1.0, 2.0, 3.0), Vector3(1.0, 2.0, 3.0)}) == 1


def test_rotation_equality_across_representations() -> None:
    assert Rotation.from_rpy(0.0, 0.0, 0.0) == Rotation.identity()
    assert Rotation.from_rpy(0.1, 0.2, 0.3) == Rotation.from_rpy(0.1, 0.2, 0.3)
    assert Rotation.from_rpy(0.1, 0.2, 0.3) != Rotation.from_rpy(0.1, 0.2, 0.4)
    # q and -q represent the same rotation.
    assert Rotation.from_quaternion(0.0, 0.0, 0.0, 1.0) == Rotation.from_quaternion(0.0, 0.0, 0.0, -1.0)
    assert len({Rotation.identity(), Rotation.from_rpy(0.0, 0.0, 0.0)}) == 1


def test_isometry_equality_and_hash() -> None:
    isometry_1 = Isometry.from_parts(Vector3(1.0, 2.0, 3.0), Rotation.from_rpy(0.0, 0.0, 0.5))
    isometry_2 = Isometry.from_parts(Vector3(1.0, 2.0, 3.0), Rotation.from_rpy(0.0, 0.0, 0.5))
    isometry_3 = Isometry.from_parts(Vector3(1.0, 2.0, 3.0), Rotation.from_rpy(0.0, 0.0, 0.6))
    assert isometry_1 == isometry_2
    assert isometry_1 != isometry_3
    assert len({isometry_1, isometry_2}) == 1


def test_rotation_and_isometry_direct_construction_raises() -> None:
    with pytest.raises(TypeError, match="cannot be constructed directly"):
        Rotation()
    with pytest.raises(TypeError, match="cannot be constructed directly"):
        Isometry()


def test_isometry_multiplication() -> None:
    isometry_1 = Isometry.from_parts(Vector3(1.0, 0.0, 0.0), Rotation.from_rpy(0.0, 0.0, radians(90.0)))
    isometry_2 = Isometry.from_parts(Vector3(0.0, 1.0, 0.0), Rotation.from_rpy(0.0, 0.0, radians(90.0)))
    result_isometry = isometry_1 * isometry_2
    translation, rotation = result_isometry.decompose()
    assert translation.as_tuple() == pytest.approx((0, 0, 0), abs=1e-5)
    assert rotation.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, radians(180.0)), abs=1e-5)

"""Contains unit tests for the library."""

from math import radians

import pytest

from cartesian_tree import RPY, Frame, Pose, Position, Quaternion


def test_create_root_frame() -> None:
    frame = Frame("root")
    assert frame.name == "root"
    assert frame.parent() is None
    assert frame.depth == 0


def test_add_child_frame_with_quaternion() -> None:
    root = Frame("base")
    pos = Position(1.0, 2.0, 3.0)
    quat = Quaternion(0.0, 0.0, 0.0, 1.0)
    child = root.add_child("child", pos, quat)

    assert isinstance(child, Frame)
    assert child.name == "child"
    assert child.parent().name == "base"
    assert root.children()[0].name == "child"


def test_add_child_frame_with_rpy() -> None:
    root = Frame("world")
    pos = Position(0.0, 0.0, 0.0)
    rpy = RPY(0.0, 0.0, 0.0)
    child = root.add_child("child_rpy", pos, rpy)

    assert isinstance(child, Frame)
    assert child.name == "child_rpy"
    assert child.parent().name == "world"


def test_transformation_to_parent_and_update() -> None:
    root = Frame("root")
    pos = Position(1.0, 2.0, 3.0)
    quat = Quaternion(0.0, 0.0, 0.0, 1.0)
    child = root.add_child("child", pos, quat)

    orig_pos, orig_quat = child.transformation_to_parent()
    assert isinstance(orig_pos, Position)
    assert isinstance(orig_quat, Quaternion)

    # Update transformation
    new_pos = Position(5.0, 6.0, 7.0)
    new_quat = Quaternion(0.0, 0.7071, 0.0, 0.7071)
    child.update_transformation(new_pos, new_quat)

    upd_pos, upd_quat = child.transformation_to_parent()
    assert upd_pos.to_tuple() == pytest.approx((5.0, 6.0, 7.0), abs=1e-5)
    assert upd_quat.to_tuple() == pytest.approx((0.0, 0.7071, 0.0, 0.7071), abs=1e-5)


def test_add_pose_and_update() -> None:
    root = Frame("base")
    pos = Position(1.0, 2.0, 3.0)
    quat = Quaternion(0.0, 0.0, 0.0, 1.0)
    pose = root.add_pose(pos, quat)

    assert isinstance(pose, Pose)
    p_pos, p_quat = pose.transformation()
    assert p_pos.to_tuple() == pytest.approx((1.0, 2.0, 3.0), abs=1e-5)
    assert p_quat.to_tuple() == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-5)

    # Update the pose
    new_pos = Position(4.0, 5.0, 6.0)
    new_rpy = RPY(0.0, 0.0, 0.0)
    pose.update(new_pos, new_rpy)
    up_pos, _ = pose.transformation()
    assert up_pos.to_tuple() == pytest.approx((4.0, 5.0, 6.0), abs=1e-5)


def test_pose_in_frame() -> None:
    base = Frame("base")
    frame_1 = base.add_child("frame1", Position(1, 1, 1), Quaternion(0, 0, 0, 1))
    frame_2 = base.add_child("frame2", Position(-2, 0, 0), RPY(0, 0, radians(90)))

    pose_in_frame1 = frame_1.add_pose(Position(0, 0, 0), Quaternion(0, 0, 0, 1))
    transformed_pose = pose_in_frame1.in_frame(frame_2)

    pos, quat = transformed_pose.transformation()

    assert pos.to_tuple() == pytest.approx((1.0, -3.0, 1.0), abs=1e-5)
    assert quat.to_rpy().to_tuple() == pytest.approx((0.0, 0.0, -radians(90)), abs=1e-5)

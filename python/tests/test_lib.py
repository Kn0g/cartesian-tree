"""Contains unit tests for the library."""

from math import radians

import pytest

from cartesian_tree import Frame, Isometry, Pose, Rotation, Vector3


def test_create_root_frame() -> None:
    frame = Frame("root")
    assert frame.name == "root"
    assert frame.parent() is None
    assert frame.depth == 0


def test_tree_structure() -> None:
    frame = Frame("root")
    position = Vector3(1.0, 2.0, 3.0)
    orientation = Rotation.from_quaternion(0.0, 0.0, 0.0, 1.0)
    child = frame.add_child("child", position, orientation)
    grandchild = child.add_child("grandchild", position, orientation)
    assert grandchild.depth == 2
    parent = grandchild.parent()
    assert parent is not None
    assert parent.name == "child"
    assert grandchild.root().name == "root"


def test_add_child_frame_with_quaternion() -> None:
    root = Frame("base")
    position = Vector3(1.0, 2.0, 3.0)
    orientation = Rotation.from_quaternion(0.0, 0.0, 0.0, 1.0)
    child = root.add_child("child", position, orientation)

    assert isinstance(child, Frame)
    assert child.name == "child"
    parent = child.parent()
    assert parent is not None
    assert parent.name == "base"
    assert root.children()[0].name == "child"


def test_add_child_frame_with_rpy() -> None:
    root = Frame("world")
    position = Vector3(0.0, 0.0, 0.0)
    rpy = Rotation.from_rpy(0.0, 0.0, 0.0)
    child = root.add_child("child_rpy", position, rpy)

    assert isinstance(child, Frame)
    assert child.name == "child_rpy"

    parent = child.parent()
    assert parent is not None
    assert parent.name == "world"


def test_transformation_to_parent_and_update() -> None:
    root = Frame("root")
    position = Vector3(1.0, 2.0, 3.0)
    orientation = Rotation.from_quaternion(0.0, 0.0, 0.0, 1.0)
    child = root.add_child("child", position, orientation)

    orig_pos, orig_quat = child.transformation_to_parent()
    assert isinstance(orig_pos, Vector3)
    assert isinstance(orig_quat, Rotation)

    # Update transformation
    new_position = Vector3(5.0, 6.0, 7.0)
    new_orientation = Rotation.from_quaternion(0.0, 0.7071, 0.0, 0.7071)
    child.set(new_position, new_orientation)

    updated_pos, updated_quat = child.transformation_to_parent()
    assert updated_pos.as_tuple() == pytest.approx((5.0, 6.0, 7.0), abs=1e-5)
    assert updated_quat.as_quaternion().as_tuple() == pytest.approx((0.0, 0.7071, 0.0, 0.7071), abs=1e-5)


def test_apply_in_parent_frame() -> None:
    root = Frame("root")
    position = Vector3(1.0, 0.0, 1.0)
    orientation = Rotation.from_quaternion(0.0, 0.0, 0.0, 1.0)
    child = root.add_child("child", position, orientation)

    # Update transformation
    rotation_to_apply = Rotation.from_rpy(0.0, 0.0, radians(90))
    child.apply_in_parent_frame(Isometry.from_rotation(rotation_to_apply))

    updated_pos, _ = child.transformation_to_parent()
    assert updated_pos.as_tuple() == pytest.approx((0.0, 1.0, 1.0), abs=1e-5)

    # Update transformation
    translation_to_apply = Vector3(1.0, 0.0, 1.0)
    child.apply_in_parent_frame(Isometry.from_translation(translation_to_apply))

    updated_pos, _ = child.transformation_to_parent()
    assert updated_pos.as_tuple() == pytest.approx((1.0, 1.0, 2.0), abs=1e-5)


def test_apply_in_local_frame() -> None:
    root = Frame("root")
    position = Vector3(0.0, 0.0, 0.0)
    orientation = Rotation.from_rpy(0.0, 0.0, radians(90))
    child = root.add_child("child", position, orientation)

    # Update transformation
    translation_to_apply = Vector3(1.0, 0.0, 0.0)
    child.apply_in_local_frame(Isometry.from_translation(translation_to_apply))

    updated_pos, _ = child.transformation_to_parent()
    assert updated_pos.as_tuple() == pytest.approx((0.0, 1.0, 0.0), abs=1e-5)

    # Update transformation
    rotation_to_apply = Rotation.from_rpy(0.0, 0.0, radians(90))
    child.apply_in_local_frame(Isometry.from_rotation(rotation_to_apply))

    updated_pos, updated_rot = child.transformation_to_parent()
    assert updated_pos.as_tuple() == pytest.approx((0.0, 1.0, 0.0), abs=1e-5)
    assert updated_rot.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, radians(180)), abs=1e-5)


def test_pose_apply_in_parent_frame() -> None:
    root = Frame("root")
    position = Vector3(1.0, 0.0, 1.0)
    orientation = Rotation.from_quaternion(0.0, 0.0, 0.0, 1.0)
    pose = root.add_pose(position, orientation)

    # Update transformation
    rotation_to_apply = Rotation.from_rpy(0.0, 0.0, radians(90))
    pose.apply_in_parent_frame(Isometry.from_rotation(rotation_to_apply))

    updated_pos, _ = pose.transformation()
    assert updated_pos.as_tuple() == pytest.approx((0.0, 1.0, 1.0), abs=1e-5)

    # Update transformation
    translation_to_apply = Vector3(1.0, 0.0, 1.0)
    pose.apply_in_parent_frame(Isometry.from_translation(translation_to_apply))

    updated_pos, _ = pose.transformation()
    assert updated_pos.as_tuple() == pytest.approx((1.0, 1.0, 2.0), abs=1e-5)


def test_pose_apply_in_local_frame() -> None:
    root = Frame("root")
    position = Vector3(0.0, 0.0, 0.0)
    orientation = Rotation.from_rpy(0.0, 0.0, radians(90))
    pose = root.add_pose(position, orientation)
    # Update transformation
    translation_to_apply = Vector3(1.0, 0.0, 0.0)
    pose.apply_in_local_frame(Isometry.from_translation(translation_to_apply))

    updated_pos, _ = pose.transformation()
    assert updated_pos.as_tuple() == pytest.approx((0.0, 1.0, 0.0), abs=1e-5)

    # Update transformation
    rotation_to_apply = Rotation.from_rpy(0.0, 0.0, radians(90))
    pose.apply_in_local_frame(Isometry.from_rotation(rotation_to_apply))

    updated_pos, updated_rot = pose.transformation()
    assert updated_pos.as_tuple() == pytest.approx((0.0, 1.0, 0.0), abs=1e-5)
    assert updated_rot.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, radians(180)), abs=1e-5)


def test_add_pose_and_update() -> None:
    root = Frame("base")
    position = Vector3(1.0, 2.0, 3.0)
    orientation = Rotation.from_quaternion(0.0, 0.0, 0.0, 1.0)
    pose = root.add_pose(position, orientation)

    assert isinstance(pose, Pose)
    p_position, p_orientation = pose.transformation()
    assert p_position.as_tuple() == pytest.approx((1.0, 2.0, 3.0), abs=1e-5)
    assert p_orientation.as_quaternion().as_tuple() == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-5)

    # Update the pose
    new_position = Vector3(4.0, 5.0, 6.0)
    new_orientation = Rotation.from_rpy(0.0, 0.0, 0.0)
    pose.set(new_position, new_orientation)
    up_pos, _ = pose.transformation()
    assert up_pos.as_tuple() == pytest.approx((4.0, 5.0, 6.0), abs=1e-5)

    # Access frame
    frame_of_pose = pose.frame()
    assert frame_of_pose.name == "base"
    frame_of_pose.add_child("child_of_pose_frame", p_position, p_orientation)
    assert len(frame_of_pose.children()) == 1


def test_pose_in_frame() -> None:
    base = Frame("base")
    frame_1 = base.add_child("frame1", Vector3(1, 1, 1), Rotation.from_quaternion(0, 0, 0, 1))
    frame_2 = base.add_child("frame2", Vector3(-2, 0, 0), Rotation.from_rpy(0, 0, radians(90)))

    pose_in_frame1 = frame_1.add_pose(Vector3(0, 0, 0), Rotation.from_quaternion(0, 0, 0, 1))
    transformed_pose = pose_in_frame1.in_frame(frame_2)

    pos, quat = transformed_pose.transformation()

    assert pos.as_tuple() == pytest.approx((1.0, -3.0, 1.0), abs=1e-5)
    assert quat.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, -radians(90)), abs=1e-5)


def test_calibrate_frame() -> None:
    base = Frame("base")
    reference_frame = base.add_child("reference", Vector3(1, 1, 1), Rotation.from_quaternion(0, 0, 0, 1))
    reference_pose = reference_frame.add_pose(Vector3(1, 1, 1), Rotation.from_quaternion(0, 0, 0, 1))

    calibrated_frame = base.calibrate_child("calibrated", Vector3(0, 0, 0), Rotation.from_rpy(0, 0, 0), reference_pose)

    pos, quat = calibrated_frame.transformation_to_parent()

    assert pos.as_tuple() == pytest.approx((2.0, 2.0, 2.0), abs=1e-5)
    assert quat.as_quaternion().as_tuple() == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-5)


def test_serialization() -> None:
    root = Frame("root")
    child1 = root.add_child("child1", Vector3(1, 0, 0), Rotation.from_quaternion(0, 0, 0, 1))
    child2 = child1.add_child("child2", Vector3(0, 1, 0), Rotation.from_rpy(0, 0, radians(90)))
    child2.add_pose(Vector3(0, 0, 1), Rotation.from_quaternion(0, 0, 0, 1))

    json_str = root.to_json()

    default_root = Frame("root")
    default_child1 = default_root.add_child("child1", Vector3(2, 0, 0), Rotation.from_quaternion(0, 0, 0, 1))
    default_child2 = default_child1.add_child("child2", Vector3(0, 2, 0), Rotation.from_rpy(0, 0, radians(90)))

    default_root.apply_config(json_str)

    position, _ = default_child1.transformation_to_parent()
    assert position.as_tuple() == pytest.approx((1.0, 0.0, 0.0), abs=1e-5)  # Updated back to '1'
    position, _ = default_child2.transformation_to_parent()
    assert position.as_tuple() == pytest.approx((0.0, 1.0, 0.0), abs=1e-5)  # Updated back to '1'

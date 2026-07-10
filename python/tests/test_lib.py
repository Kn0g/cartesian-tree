"""Contains unit tests for the library."""

from math import pi, radians

import pytest

from cartesian_tree import Frame, Isometry, Pose, Rotation, Vector3, rx, ry, rz, x, y, z


def test_create_root_frame() -> None:
    frame = Frame("root")
    assert frame.name == "root"
    assert frame.parent() is None
    assert frame.depth == 0


def test_tree_structure() -> None:
    frame = Frame("root")
    position = Vector3(1.0, 2.0, 3.0)
    orientation = Rotation.identity()
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
    orientation = Rotation.identity()
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


def test_transformation_and_update() -> None:
    root = Frame("root")
    position = Vector3(1.0, 2.0, 3.0)
    orientation = Rotation.identity()
    child = root.add_child("child", position, orientation)

    orig_pos, orig_quat = child.transformation()
    assert isinstance(orig_pos, Vector3)
    assert isinstance(orig_quat, Rotation)

    assert child.position.as_tuple() == pytest.approx((1.0, 2.0, 3.0), abs=1e-5)
    assert child.orientation.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-5)

    # Update transformation
    new_position = Vector3(5.0, 6.0, 7.0)
    new_orientation = Rotation.from_quaternion(0.0, 0.7071, 0.0, 0.7071)
    child.set(new_position, new_orientation)

    updated_pos, updated_quat = child.transformation()
    assert updated_pos.as_tuple() == pytest.approx((5.0, 6.0, 7.0), abs=1e-5)
    assert updated_quat.as_quaternion().as_tuple() == pytest.approx((0.0, 0.7071, 0.0, 0.7071), abs=1e-5)


def test_apply_in_parent_frame() -> None:
    root = Frame("root")
    position = Vector3(1.0, 0.0, 1.0)
    orientation = Rotation.identity()
    child = root.add_child("child", position, orientation)

    # Update transformation
    rotation_to_apply = Rotation.from_rpy(0.0, 0.0, radians(90))
    child.apply_in_parent_frame(Isometry.from_rotation(rotation_to_apply))

    updated_pos, _ = child.transformation()
    assert updated_pos.as_tuple() == pytest.approx((0.0, 1.0, 1.0), abs=1e-5)

    # Update transformation
    translation_to_apply = Vector3(1.0, 0.0, 1.0)
    child.apply_in_parent_frame(Isometry.from_translation(translation_to_apply))

    updated_pos, _ = child.transformation()
    assert updated_pos.as_tuple() == pytest.approx((1.0, 1.0, 2.0), abs=1e-5)


def test_apply_in_local_frame() -> None:
    root = Frame("root")
    position = Vector3.zeros()
    orientation = Rotation.from_rpy(0.0, 0.0, radians(90))
    child = root.add_child("child", position, orientation)

    # Update transformation
    translation_to_apply = Vector3(1.0, 0.0, 0.0)
    child.apply_in_local_frame(Isometry.from_translation(translation_to_apply))

    updated_pos, _ = child.transformation()
    assert updated_pos.as_tuple() == pytest.approx((0.0, 1.0, 0.0), abs=1e-5)

    # Update transformation
    rotation_to_apply = Rotation.from_rpy(0.0, 0.0, radians(90))
    child.apply_in_local_frame(Isometry.from_rotation(rotation_to_apply))

    updated_pos, updated_rot = child.transformation()
    assert updated_pos.as_tuple() == pytest.approx((0.0, 1.0, 0.0), abs=1e-5)
    assert updated_rot.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, radians(180)), abs=1e-5)


def test_pose_apply_in_parent_frame() -> None:
    root = Frame("root")
    position = Vector3(1.0, 0.0, 1.0)
    orientation = Rotation.identity()
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
    position = Vector3.zeros()
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
    orientation = Rotation.identity()
    pose = root.add_pose(position, orientation)

    assert isinstance(pose, Pose)
    p_position, p_orientation = pose.transformation()
    assert p_position.as_tuple() == pytest.approx((1.0, 2.0, 3.0), abs=1e-5)
    assert p_orientation.as_quaternion().as_tuple() == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-5)

    assert pose.position.as_tuple() == pytest.approx((1.0, 2.0, 3.0), abs=1e-5)
    assert pose.orientation.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-5)

    # Update the pose
    new_position = Vector3(4.0, 5.0, 6.0)
    new_orientation = Rotation.from_rpy(0.0, 0.0, 0.0)
    pose.set(new_position, new_orientation)
    up_pos, _ = pose.transformation()
    assert up_pos.as_tuple() == pytest.approx((4.0, 5.0, 6.0), abs=1e-5)

    # Access frame
    frame_of_pose = pose.frame()
    assert frame_of_pose is not None
    assert frame_of_pose.name == "base"
    frame_of_pose.add_child("child_of_pose_frame", p_position, p_orientation)
    assert len(frame_of_pose.children()) == 1


def test_duplicate_child_name_raises() -> None:
    root = Frame("root")
    root.add_child("child", Vector3.zeros(), Rotation.identity())
    with pytest.raises(ValueError, match="already exists"):
        root.add_child("child", Vector3.zeros(), Rotation.identity())


def test_root_frame_mutation_raises() -> None:
    root = Frame("root")
    with pytest.raises(ValueError, match="no parent"):
        root.set(Vector3.zeros(), Rotation.identity())
    with pytest.raises(ValueError, match="no parent"):
        root.apply_in_parent_frame(Isometry.identity())
    with pytest.raises(ValueError, match="no parent"):
        root.apply_in_local_frame(Isometry.identity())
    with pytest.raises(ValueError, match="root frame"):
        root.transformation()


def test_in_frame_across_disjoint_trees_raises() -> None:
    tree_1 = Frame("tree_1")
    tree_2 = Frame("tree_2")
    pose = tree_1.add_pose(Vector3.zeros(), Rotation.identity())
    with pytest.raises(ValueError, match="common ancestor"):
        pose.in_frame(tree_2)


def test_apply_config_error_cases() -> None:
    root = Frame("root")
    mismatched = '{"name": "other", "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0], "children": []}'
    with pytest.raises(ValueError, match="do not match"):
        root.apply_config(mismatched)
    with pytest.raises(ValueError, match="error"):
        root.apply_config("not json")


def test_pose_round_trip_through_deep_tree() -> None:
    # Two branches, each two levels deep, all with non-identity transforms:
    # expressing a pose in the other branch and back must be lossless.
    root = Frame("root")
    a = root.add_child("a", Vector3(0.3, -1.2, 2.5), Rotation.from_rpy(0.4, -0.3, 1.2))
    b = a.add_child("b", Vector3(-2.0, 0.7, 0.1), Rotation.from_rpy(-1.0, 0.2, 0.5))
    c = root.add_child("c", Vector3(1.5, 2.0, -0.4), Rotation.from_rpy(0.1, 1.1, -0.7))
    d = c.add_child("d", Vector3(0.0, -0.5, 1.0), Rotation.from_rpy(0.9, -0.8, 0.3))

    pose = b.add_pose(Vector3(0.2, 0.4, -0.6), Rotation.from_rpy(0.5, 0.5, -0.5))
    round_tripped = pose.in_frame(d).in_frame(b)

    pos, rot = round_tripped.transformation()
    assert pos.as_tuple() == pytest.approx((0.2, 0.4, -0.6), abs=1e-9)
    assert rot.as_rpy().as_tuple() == pytest.approx((0.5, 0.5, -0.5), abs=1e-9)


def test_lazy_helpers_all_axes() -> None:
    root = Frame("root")
    pose = root.add_pose(Vector3.zeros(), Rotation.identity())

    moved = pose + x(1.0) + y(2.0) + z(3.0)
    pos, _ = moved.transformation()
    assert pos.as_tuple() == pytest.approx((1.0, 2.0, 3.0), abs=1e-10)

    _, rot = (pose * rx(0.3)).transformation()
    assert rot.as_rpy().as_tuple() == pytest.approx((0.3, 0.0, 0.0), abs=1e-10)

    _, rot = (pose * ry(0.4)).transformation()
    assert rot.as_rpy().as_tuple() == pytest.approx((0.0, 0.4, 0.0), abs=1e-10)


def test_detached_frame_raises_clear_error() -> None:
    def make_detached() -> Frame:
        root = Frame("root")
        return root.add_child("kid", Vector3.zeros(), Rotation.identity())

    kid = make_detached()  # The root is garbage-collected here.
    assert kid.parent() is None
    with pytest.raises(ValueError, match="detached"):
        kid.transformation()
    with pytest.raises(ValueError, match="detached"):
        kid.set(Vector3.zeros(), Rotation.identity())


def test_pose_frame_returns_none_when_frame_dropped() -> None:
    def make_orphan_pose() -> Pose:
        temp = Frame("temp")
        return temp.add_pose(Vector3.zeros(), Rotation.identity())

    orphan = make_orphan_pose()  # The frame is garbage-collected here.
    assert orphan.frame() is None


def test_pose_in_frame() -> None:
    base = Frame("base")
    frame_1 = base.add_child("frame1", Vector3(1, 1, 1), Rotation.identity())
    frame_2 = base.add_child("frame2", Vector3(-2, 0, 0), Rotation.from_rpy(0, 0, radians(90)))

    pose_in_frame1 = frame_1.add_pose(Vector3(0, 0, 0), Rotation.identity())
    transformed_pose = pose_in_frame1.in_frame(frame_2)

    pos, quat = transformed_pose.transformation()

    assert pos.as_tuple() == pytest.approx((1.0, -3.0, 1.0), abs=1e-5)
    assert quat.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, -radians(90)), abs=1e-5)


def test_calibrate_frame() -> None:
    base = Frame("base")
    reference_frame = base.add_child("reference", Vector3(1, 1, 1), Rotation.identity())
    reference_pose = reference_frame.add_pose(Vector3(1, 1, 1), Rotation.identity())

    calibrated_frame = base.calibrate_child("calibrated", Vector3(0, 0, 0), Rotation.from_rpy(0, 0, 0), reference_pose)

    pos, quat = calibrated_frame.transformation()

    assert pos.as_tuple() == pytest.approx((2.0, 2.0, 2.0), abs=1e-5)
    assert quat.as_quaternion().as_tuple() == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-5)


def test_calibrate_child_under_non_identity_parent() -> None:
    base = Frame("base")
    # The parent of the calibrated frame is not the common ancestor and is rotated
    mount = base.add_child("mount", Vector3(0.0, 0.0, 1.0), Rotation.from_rpy(0.0, 0.0, radians(90)))
    reference_pose = base.add_pose(Vector3(1.0, 0.0, 0.0), Rotation.identity())

    calibrated = mount.calibrate_child("calibrated", Vector3.zeros(), Rotation.identity(), reference_pose)

    pose_in_calibrated = reference_pose.in_frame(calibrated)
    pos, rot = pose_in_calibrated.transformation()
    assert pos.as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert rot.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_serialization() -> None:
    root = Frame("root")
    child1 = root.add_child("child1", Vector3(1, 0, 0), Rotation.identity())
    child2 = child1.add_child("child2", Vector3(0, 1, 0), Rotation.from_rpy(0, 0, radians(90)))
    child2.add_pose(Vector3(0, 0, 1), Rotation.identity())

    json_str = root.to_json()

    default_root = Frame("root")
    default_child1 = default_root.add_child("child1", Vector3(2, 0, 0), Rotation.identity())
    default_child2 = default_child1.add_child("child2", Vector3(0, 2, 0), Rotation.from_rpy(0, 0, radians(90)))

    default_root.apply_config(json_str)

    position, _ = default_child1.transformation()
    assert position.as_tuple() == pytest.approx((1.0, 0.0, 0.0), abs=1e-5)  # Updated back to '1'
    position, _ = default_child2.transformation()
    assert position.as_tuple() == pytest.approx((0.0, 1.0, 0.0), abs=1e-5)  # Updated back to '1'


def test_apply_config_rejects_zero_norm_quaternion() -> None:
    root = Frame("root")
    root.add_child("child", Vector3.zeros(), Rotation.identity())
    config = (
        '{"name": "root", "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0], "children": '
        '[{"name": "child", "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 0.0], "children": []}]}'
    )
    with pytest.raises(ValueError, match="norm is too close to zero"):
        root.apply_config(config)


def test_lazy_translation_frame() -> None:
    root = Frame("root")
    child = root.add_child("child", Vector3(0.0, 0.0, 0.0), Rotation.identity())

    result = child + z(5.0)
    pos, rot = result.transformation()
    assert pos.as_tuple() == pytest.approx((0.0, 0.0, 5.0), abs=1e-10)
    pos, rot = child.transformation()
    assert pos.as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-10)

    # Chained operations accumulate in world coordinates.
    result = result - y(3.0)
    pos, rot = result.add_pose(Vector3.zeros(), Rotation.identity()).in_frame(root).transformation()
    assert pos.as_tuple() == pytest.approx((0.0, -3.0, 5.0), abs=1e-10)

    roll, pitch, yaw = rot.as_rpy().as_tuple()
    assert Vector3(roll, pitch, yaw).as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-10)


def test_lazy_rotation_frame() -> None:
    root = Frame("root")
    child = root.add_child("child", Vector3(0.0, 0.0, 0.0), Rotation.identity())
    result = child * rz(pi / 4)

    pos, rot = result.transformation()
    roll, pitch, yaw = rot.as_rpy().as_tuple()
    assert Vector3(roll, pitch, yaw).as_tuple() == pytest.approx((0.0, 0.0, pi / 4), abs=1e-10)
    assert pos.as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-10)

    pos, rot = child.transformation()
    roll, pitch, yaw = rot.as_rpy().as_tuple()
    assert Vector3(roll, pitch, yaw).as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-10)


def test_lazy_ops_on_non_identity_frame() -> None:
    root = Frame("root")
    child = root.add_child("child", Vector3(1.0, 0.0, 0.0), Rotation.from_rpy(0.0, 0.0, radians(90)))

    # Translation is interpreted in the parent frame: result at child + (0, 3, 0) in root.
    shifted = child + y(3.0)
    pos, rot = shifted.add_pose(Vector3.zeros(), Rotation.identity()).in_frame(root).transformation()
    assert pos.as_tuple() == pytest.approx((1.0, 3.0, 0.0), abs=1e-9)
    assert rot.as_rpy().as_tuple() == pytest.approx((0.0, 0.0, radians(90)), abs=1e-9)

    # Rotation is interpreted in the local frame: position unchanged, yaw doubled.
    rotated = child * rz(radians(90))
    pos, rot = rotated.add_pose(Vector3.zeros(), Rotation.identity()).in_frame(root).transformation()
    assert pos.as_tuple() == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert abs(rot.as_rpy().as_tuple()[2]) == pytest.approx(radians(180), abs=1e-9)


def test_lazy_translation_pose() -> None:
    root = Frame("root")
    pose = root.add_pose(Vector3.zeros(), Rotation.identity())

    result = pose + z(5.0)
    pos, rot = result.transformation()
    assert pos.as_tuple() == pytest.approx((0.0, 0.0, 5.0), abs=1e-10)
    pos, rot = pose.transformation()
    assert pos.as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-10)

    result = result - y(3.0)
    pos, rot = result.transformation()
    assert pos.as_tuple() == pytest.approx((0.0, -3.0, 5.0), abs=1e-10)

    roll, pitch, yaw = rot.as_rpy().as_tuple()
    assert Vector3(roll, pitch, yaw).as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-10)


def test_lazy_rotation_pose() -> None:
    root = Frame("root")
    pose = root.add_pose(Vector3.zeros(), Rotation.identity())
    result = pose * rz(pi / 4)

    pos, rot = result.transformation()
    roll, pitch, yaw = rot.as_rpy().as_tuple()
    assert Vector3(roll, pitch, yaw).as_tuple() == pytest.approx((0.0, 0.0, pi / 4), abs=1e-10)
    assert pos.as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-10)

    pos, rot = pose.transformation()
    roll, pitch, yaw = rot.as_rpy().as_tuple()
    assert Vector3(roll, pitch, yaw).as_tuple() == pytest.approx((0.0, 0.0, 0.0), abs=1e-10)

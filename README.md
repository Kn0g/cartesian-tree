# cartesian-tree

cartesian-tree is a library for managing and transforming poses in a tree-structured hierarchy of 3D coordinate frames in rust as well as python.

## Key ideas
- Tree Structure: Frames form a tree where each frame defines its position and orientation relative to a parent frame.
- Poses and Transformations: A Pose represents a position + orientation in a frame. Easily transform poses between frames of the same tree.
- Unified Rotations: Supports quaternions and roll-pitch-yaw (RPY) interchangeably.
- Lazy Operations: Quick adjustments like translations (`+ x(1.0)`) or rotations (`* rz(PI/4)`).

## Installation
The project can be found on [crates.io](https://crates.io/crates/cartesiantree) and [pypi.org](https://pypi.org/project/cartesian-tree/#description)


## Usage
Rust:

```rust
use cartesian_tree::{Frame, Pose};
use nalgebra::{Vector3, UnitQuaternion};

fn main() {
    // Create root frame
    let world = Frame::new_origin("world");

    // Add a child frame
    let child = world.add_child(
        "child",
        Vector3::new(1.0, 0.0, 0.0),
        UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
    ).unwrap();

    // Create a pose in the world frame
    let pose = world.add_pose(Vector3::new(0.0, 1.0, 0.0), UnitQuaternion::identity());

    // Transform the pose to the child frame
    let pose_in_child = pose.in_frame(&child).unwrap();

    println!("Pose in child: {:?}", pose_in_base.transformation());
```
Python:

```python
from cartesian_tree import Frame, Rotation, Vector3

# Create root frame
world = Frame("world")

# Add a child frame
child = world.add_child(
    "child",
    Vector3(1.0, 0.0, 0.0),
    Rotation.identity()
)

# Create a pose in the world frame
pose = world.add_pose(
    Vector3(0.0, 1.0, 0.0), Rotation.identity()
)

# Transform the pose to the child frame
pose_in_child = pose.in_frame(child)

print(f"Pose in child: {pose_in_child}")
```
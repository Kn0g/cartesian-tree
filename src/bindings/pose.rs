use pyo3::prelude::*;

use crate::{
    Pose as RustPose,
    bindings::{
        PyFrame,
        utils::{PyQuaternion, PyVector3},
    },
};

#[pyclass(name = "Pose", unsendable)]
#[derive(Clone)]
pub struct PyPose {
    pub(crate) rust_pose: RustPose,
}

#[pymethods]
impl PyPose {
    fn frame(&self) -> Option<PyFrame> {
        self.rust_pose
            .frame()
            .map(|frame| PyFrame { rust_frame: frame })
    }

    const fn transformation(&self) -> (PyVector3, PyQuaternion) {
        let isometry = self.rust_pose.transformation();
        (
            PyVector3 {
                inner: isometry.translation.vector,
            },
            PyQuaternion {
                quat: isometry.rotation,
            },
        )
    }

    #[pyo3(signature = (position, quaternion))]
    fn update(&mut self, position: PyVector3, quaternion: PyQuaternion) {
        self.rust_pose.update(position.inner, quaternion.quat);
    }

    #[pyo3(signature = (target_frame))]
    fn in_frame(&self, target_frame: &PyFrame) -> PyResult<Self> {
        let new_rust_pose = self.rust_pose.in_frame(&target_frame.rust_frame)?;
        Ok(Self {
            rust_pose: new_rust_pose,
        })
    }

    fn __str__(&self) -> String {
        let isometry = self.rust_pose.transformation();
        let vector = isometry.translation.vector;
        let quaternion = isometry.rotation.coords;
        format!(
            "({:.2}, {:.2}, {:.2})({:.4}, {:.4}, {:.4}, {:.4})",
            vector.x, vector.y, vector.z, quaternion.x, quaternion.y, quaternion.z, quaternion.w,
        )
    }

    fn __repr__(&self) -> String {
        let isometry = self.rust_pose.transformation();
        let position = isometry.translation.vector;
        let quaternion = isometry.rotation.coords;
        format!(
            "'{}', ({:.2}, {:.2}, {:.2})({:.4}, {:.4}, {:.4}, {:.4}))",
            self.frame()
                .map_or_else(|| "Unknown".to_string(), |frame| frame.name()),
            position.x,
            position.y,
            position.z,
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        )
    }
}

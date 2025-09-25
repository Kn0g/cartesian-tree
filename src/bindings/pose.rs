use pyo3::prelude::*;

use crate::{
    Pose as RustPose,
    bindings::{
        PyFrame,
        utils::{PyPosition, PyQuaternion},
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

    fn transformation(&self) -> (PyPosition, PyQuaternion) {
        let isometry = self.rust_pose.transformation();
        (
            PyPosition {
                position: isometry.translation.vector,
            },
            PyQuaternion {
                quat: isometry.rotation,
            },
        )
    }

    #[pyo3(signature = (position, quaternion))]
    fn update(&mut self, position: PyPosition, quaternion: PyQuaternion) {
        self.rust_pose.update(position.position, quaternion.quat);
    }

    #[pyo3(signature = (target_frame))]
    fn in_frame(&self, target_frame: &PyFrame) -> PyResult<PyPose> {
        let new_rust_pose = self.rust_pose.in_frame(&target_frame.rust_frame)?;
        Ok(PyPose {
            rust_pose: new_rust_pose,
        })
    }

    fn __str__(&self) -> String {
        let isometry = self.rust_pose.transformation();
        let position = isometry.translation.vector;
        let quaternion = isometry.rotation.coords;
        format!(
            "({:.2}, {:.2}, {:.2})({:.4}, {:.4}, {:.4}, {:.4})",
            position.x,
            position.y,
            position.z,
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        )
    }

    fn __repr__(&self) -> String {
        let isometry = self.rust_pose.transformation();
        let position = isometry.translation.vector;
        let quaternion = isometry.rotation.coords;
        format!(
            "'{}', ({:.2}, {:.2}, {:.2})({:.4}, {:.4}, {:.4}, {:.4}))",
            self.frame()
                .map(|frame| frame.name())
                .unwrap_or_else(|| "Unknown".to_string()),
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

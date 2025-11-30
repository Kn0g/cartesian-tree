use pyo3::prelude::*;

use crate::{
    Pose as RustPose,
    bindings::{
        PyFrame,
        lazy_access::{PyLazyRotation, PyLazyTranslation},
        utils::{PyIsometry, PyRotation, PyVector3},
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

    fn transformation(&self) -> (PyVector3, PyRotation) {
        let isometry = self.rust_pose.transformation();
        (
            PyVector3 {
                inner: isometry.translation.vector,
            },
            PyRotation {
                rust_rotation: isometry.rotation.into(),
            },
        )
    }

    #[pyo3(signature = (position, orientation))]
    fn set(&mut self, position: PyVector3, orientation: PyRotation) {
        self.rust_pose
            .set(position.inner, orientation.rust_rotation);
    }

    #[pyo3(signature = (isometry))]
    fn apply_in_parent_frame(&mut self, isometry: PyIsometry) {
        self.rust_pose.apply_in_parent_frame(&isometry.inner);
    }

    #[pyo3(signature = (isometry))]
    fn apply_in_local_frame(&mut self, isometry: PyIsometry) {
        self.rust_pose.apply_in_local_frame(&isometry.inner);
    }

    #[pyo3(signature = (target_frame))]
    fn in_frame(&self, target_frame: &PyFrame) -> PyResult<Self> {
        let new_rust_pose = self.rust_pose.in_frame(&target_frame.rust_frame)?;
        Ok(Self {
            rust_pose: new_rust_pose,
        })
    }

    fn __add__(&self, other: PyLazyTranslation) -> Self {
        let new_rust_pose = &self.rust_pose + other.inner;
        Self {
            rust_pose: new_rust_pose,
        }
    }

    fn __sub__(&self, other: PyLazyTranslation) -> Self {
        let new_rust_pose = &self.rust_pose - other.inner;
        Self {
            rust_pose: new_rust_pose,
        }
    }

    fn __mul__(&self, other: PyLazyRotation) -> Self {
        let new_rust_pose = &self.rust_pose * other.inner;
        Self {
            rust_pose: new_rust_pose,
        }
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
        self.__str__()
    }
}

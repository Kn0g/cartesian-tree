use pyo3::prelude::*;

use crate::{
    Frame as RustFrame,
    bindings::{
        PyPose,
        utils::{PyPosition, PyQuaternion},
    },
    tree::{HasChildren, HasParent, Walking},
};

#[pyclass(name = "Frame", unsendable)]
#[derive(Clone)]
pub struct PyFrame {
    pub(crate) rust_frame: RustFrame,
}

#[pymethods]
impl PyFrame {
    #[new]
    #[pyo3(signature = (name))]
    fn new(name: String) -> Self {
        Self {
            rust_frame: RustFrame::new_origin(name),
        }
    }

    #[getter]
    pub(crate) fn name(&self) -> String {
        self.rust_frame.name()
    }

    #[pyo3(signature = (name, position, quaternion))]
    fn add_child(
        &self,
        name: String,
        position: PyPosition,
        quaternion: PyQuaternion,
    ) -> PyResult<Self> {
        let child_frame = self
            .rust_frame
            .add_child(name, position.position, quaternion.quat)?;
        Ok(Self {
            rust_frame: child_frame,
        })
    }

    #[pyo3(signature = (name, desired_position, desired_quaternion, reference_pose))]
    fn calibrate_child(
        &self,
        name: String,
        desired_position: PyPosition,
        desired_quaternion: PyQuaternion,
        reference_pose: &PyPose,
    ) -> PyResult<Self> {
        let new_rust_frame = self.rust_frame.calibrate_child(
            name,
            desired_position.position,
            desired_quaternion.quat,
            &reference_pose.rust_pose,
        )?;
        Ok(Self {
            rust_frame: new_rust_frame,
        })
    }

    #[pyo3(signature = (position, quaternion))]
    fn add_pose(&self, position: PyPosition, quaternion: PyQuaternion) -> PyPose {
        let rust_pose = self.rust_frame.add_pose(position.position, quaternion.quat);
        PyPose { rust_pose }
    }

    fn transformation_to_parent(&self) -> PyResult<(PyPosition, PyQuaternion)> {
        let isometry = self.rust_frame.transform_to_parent()?;
        Ok((
            PyPosition {
                position: isometry.translation.vector,
            },
            PyQuaternion {
                quat: isometry.rotation,
            },
        ))
    }

    #[pyo3(signature = (position, quaternion))]
    fn update_transformation(
        &self,
        position: PyPosition,
        quaternion: PyQuaternion,
    ) -> PyResult<()> {
        self.rust_frame
            .update_transform(position.position, quaternion.quat)?;
        Ok(())
    }

    fn to_json(&self) -> PyResult<String> {
        Ok(self.rust_frame.to_json()?)
    }

    #[pyo3(signature = (json))]
    fn apply_config(&self, json: &str) -> PyResult<()> {
        self.rust_frame.apply_config(json)?;
        Ok(())
    }

    #[getter]
    fn depth(&self) -> usize {
        self.rust_frame.depth()
    }

    fn root(&self) -> Self {
        Self {
            rust_frame: self.rust_frame.root(),
        }
    }

    fn parent(&self) -> Option<Self> {
        self.rust_frame.parent().map(|rf| Self { rust_frame: rf })
    }

    fn children(&self) -> Vec<Self> {
        self.rust_frame
            .children()
            .into_iter()
            .map(|rf| Self { rust_frame: rf })
            .collect()
    }

    fn __str__(&self) -> String {
        self.rust_frame.name()
    }

    fn __repr__(&self) -> String {
        self.rust_frame.name()
    }
}

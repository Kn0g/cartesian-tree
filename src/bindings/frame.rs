use pyo3::prelude::*;

use crate::{
    Frame as RustFrame,
    bindings::{
        PyPose,
        lazy_access::{PyLazyRotation, PyLazyTranslation},
        utils::{PyIsometry, PyRotation, PyVector3},
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

    #[pyo3(signature = (name, position, orientation))]
    fn add_child(
        &self,
        name: String,
        position: PyVector3,
        orientation: PyRotation,
    ) -> PyResult<Self> {
        let child_frame =
            self.rust_frame
                .add_child(name, position.inner, orientation.rust_rotation)?;
        Ok(Self {
            rust_frame: child_frame,
        })
    }

    #[pyo3(signature = (name, desired_position, desired_orientation, reference_pose))]
    fn calibrate_child(
        &self,
        name: String,
        desired_position: PyVector3,
        desired_orientation: PyRotation,
        reference_pose: &PyPose,
    ) -> PyResult<Self> {
        let new_rust_frame = self.rust_frame.calibrate_child(
            name,
            desired_position.inner,
            desired_orientation.rust_rotation,
            &reference_pose.rust_pose,
        )?;
        Ok(Self {
            rust_frame: new_rust_frame,
        })
    }

    #[pyo3(signature = (position, orientation))]
    fn add_pose(&self, position: PyVector3, orientation: PyRotation) -> PyPose {
        let rust_pose = self
            .rust_frame
            .add_pose(position.inner, orientation.rust_rotation);
        PyPose { rust_pose }
    }

    fn transformation_to_parent(&self) -> PyResult<(PyVector3, PyRotation)> {
        let isometry = self.rust_frame.transform_to_parent()?;
        Ok((
            PyVector3 {
                inner: isometry.translation.vector,
            },
            PyRotation {
                rust_rotation: isometry.rotation.into(),
            },
        ))
    }

    #[pyo3(signature = (position, orientation))]
    fn set(&self, position: PyVector3, orientation: PyRotation) -> PyResult<()> {
        self.rust_frame
            .set(position.inner, orientation.rust_rotation)?;
        Ok(())
    }

    #[pyo3(signature = (isometry))]
    fn apply_in_parent_frame(&self, isometry: PyIsometry) -> PyResult<()> {
        self.rust_frame.apply_in_parent_frame(&isometry.inner)?;
        Ok(())
    }

    #[pyo3(signature = (isometry))]
    fn apply_in_local_frame(&self, isometry: PyIsometry) -> PyResult<()> {
        self.rust_frame.apply_in_local_frame(&isometry.inner)?;
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

    fn __add__(&self, other: PyLazyTranslation) -> Self {
        let new_rust_frame = &self.rust_frame + other.inner;
        Self {
            rust_frame: new_rust_frame,
        }
    }

    fn __sub__(&self, other: PyLazyTranslation) -> Self {
        let new_rust_frame = &self.rust_frame - other.inner;
        Self {
            rust_frame: new_rust_frame,
        }
    }

    fn __mul__(&self, other: PyLazyRotation) -> Self {
        let new_rust_frame = &self.rust_frame * other.inner;
        Self {
            rust_frame: new_rust_frame,
        }
    }

    fn __str__(&self) -> String {
        self.rust_frame.name()
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

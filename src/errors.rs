#[derive(Debug, thiserror::Error)]
pub enum CartesianTreeError {
    #[error("Frame '{0}' is a root frame and has no parent")]
    RootHasNoParent(String),
    #[error("Frame handle is stale: the frame has been removed from its tree")]
    FrameRemoved,
    #[error("Cannot update transform for frame '{0}' as it has no parent")]
    CannotUpdateRootTransform(String),
    #[error("Operation not supported on derived frame '{0}'")]
    DerivedFrameUnsupported(String),
    #[error("A child frame with name '{0}' already exists for parent '{1}'")]
    ChildNameConflict(String, String),
    #[error("Frame '{1}' has no child named '{0}'")]
    ChildNotFound(String, String),
    #[error("Frames '{0}' and '{1}' belong to different trees")]
    DifferentTrees(String, String),
    #[error("Failed to find a common ancestor between frame '{0}' and '{1}'")]
    NoCommonAncestor(String, String),
    #[error("Frame '{0}' is not an ancestor of '{1}'")]
    IsNoAncestor(String, String),
    #[error("Invalid quaternion (x={0}, y={1}, z={2}, w={3}): norm is too close to zero")]
    InvalidQuaternion(f64, f64, f64, f64),
    #[error("Time {0} is not covered by the transform buffer (available range: [{1}, {2}])")]
    TimeNotCovered(f64, f64, f64),
    #[error("Invalid timestamp {0}: timestamps must be finite")]
    InvalidTimestamp(f64),
    #[error("Serialization/Deserialization error: {0}")]
    SerdeError(#[from] serde_json::Error),
    #[error("Tree structure mismatch during config apply: {0}")]
    Mismatch(String),
}

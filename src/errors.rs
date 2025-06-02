#[derive(Debug, thiserror::Error)]
pub enum CartesianTreeError {
    #[error("Frame '{0}' not found or has been dropped")]
    FrameDropped(String),
    #[error("Frame '{0}' is a root frame and has no parent")]
    RootHasNoParent(String),
    #[error("Cannot update transform for frame '{0}' as it has no parent")]
    CannotUpdateRootTransform(String),
    #[error("A child frame with name '{0}' already exists for parent '{1}'")]
    ChildNameConflict(String, String),
    #[error("Failed to find a common ancestor between frame '{0}' and '{1}'")]
    NoCommonAncestor(String, String),
    #[error("Frame '{0}' is not an ancestor of '{1}'")]
    IsNoAncestor(String, String),
    #[error("Internal error: Weak pointer upgrade failed")]
    WeakUpgradeFailed(),
}

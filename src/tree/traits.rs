/// Defines the parent trait.
pub trait HasParent {
    type Node: Clone;

    /// Returns the parent of this node, if it exists.
    fn parent(&self) -> Option<Self::Node>;
}

/// Defines the equality trait.
pub trait NodeEquality {
    /// Whether this node is the same as another.
    fn is_same(&self, other: &Self) -> bool;
}

/// Defines the child trait.
pub trait HasChildren {
    type Node: Clone;

    /// Returns all children.
    fn children(&self) -> Vec<Self::Node>;
}

///Defines the walking trait of tree-like structure.
pub trait Walking: HasParent<Node = Self> + NodeEquality + Clone {
    /// Gets the depth of this node in comparison to the root.
    ///
    /// # Returns
    ///
    /// The depth of this node in comparison to the root.
    fn depth(&self) -> usize {
        let mut depth = 0;
        let mut current = self.clone();
        while let Some(parent) = current.parent() {
            depth += 1;
            current = parent;
        }
        depth
    }

    /// Walks up the tree by a given number of steps.
    ///
    /// # Arguments
    ///
    /// * `steps` - The number of steps to traverse upward.
    ///
    /// # Returns
    ///
    /// The ancestor Node or `None` if it does not exist (root reached before).
    fn walk_up(&self, steps: usize) -> Option<Self> {
        let mut current = self.clone();
        for _ in 0..steps {
            current = current.parent()?;
        }
        Some(current)
    }

    /// Finds the lowest common ancestor with another Node.
    ///
    /// # Arguments.
    ///
    /// - `other`: The other Node to find the lca with.
    ///
    /// # Returns
    ///
    /// The lowest common ancestor Node or `None` if it does not exist.
    fn lca_with(&self, other: &Self) -> Option<Self> {
        let mut own = self.clone();
        let mut other = other.clone();

        let own_depth = self.depth();
        let other_depth = other.depth();

        // Equalize depths
        if own_depth > other_depth {
            own = own.walk_up(own_depth - other_depth)?;
        } else if other_depth > own_depth {
            other = other.walk_up(other_depth - own_depth)?;
        }

        // Walk up together
        while !own.is_same(&other) {
            own = own.parent()?;
            other = other.parent()?;
        }
        Some(own)
    }
}

impl<T> Walking for T where T: HasParent<Node = T> + NodeEquality + Clone {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use nalgebra::{UnitQuaternion, Vector3};

    #[test]
    fn test_depth() {
        let root = Frame::new_origin("root");
        assert_eq!(root.depth(), 0);

        let child = root
            .add_child("child", Vector3::zeros(), UnitQuaternion::identity())
            .unwrap();
        assert_eq!(child.depth(), 1);

        let grandchild = child
            .add_child("grandchild", Vector3::zeros(), UnitQuaternion::identity())
            .unwrap();
        assert_eq!(grandchild.depth(), 2);
    }

    #[test]
    fn test_walk_up() {
        let root = Frame::new_origin("root");
        let child = root
            .add_child("child", Vector3::zeros(), UnitQuaternion::identity())
            .unwrap();
        let grandchild = child
            .add_child("grandchild", Vector3::zeros(), UnitQuaternion::identity())
            .unwrap();

        assert!(grandchild.walk_up(0).unwrap().is_same(&grandchild));
        assert!(grandchild.walk_up(1).unwrap().is_same(&child));
        assert!(grandchild.walk_up(2).unwrap().is_same(&root));
        assert!(grandchild.walk_up(3).is_none());
    }

    #[test]
    fn test_lca_with() {
        let root = Frame::new_origin("root");
        let child1 = root
            .add_child("child1", Vector3::zeros(), UnitQuaternion::identity())
            .unwrap();
        let child2 = root
            .add_child("child2", Vector3::zeros(), UnitQuaternion::identity())
            .unwrap();
        let grandchild1 = child1
            .add_child("grandchild1", Vector3::zeros(), UnitQuaternion::identity())
            .unwrap();
        let grandchild2 = child2
            .add_child("grandchild2", Vector3::zeros(), UnitQuaternion::identity())
            .unwrap();

        // LCA of a node with itself is itself
        assert!(child1.lca_with(&child1).unwrap().is_same(&child1));
        // LCA of sibling nodes is their parent
        assert!(child1.lca_with(&child2).unwrap().is_same(&root));
        // LCA of grandchild and parent is the parent
        assert!(grandchild1.lca_with(&child1).unwrap().is_same(&child1));
        // LCA of two grandchildren with different parents is root
        assert!(grandchild1.lca_with(&grandchild2).unwrap().is_same(&root));
    }
}

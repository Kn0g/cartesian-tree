"""Contains lazy access functions."""

from __future__ import annotations

from cartesian_tree import _cartesian_tree as _core  # type: ignore[attr-defined]


class LazyTranslation:
    """Type wrapper for lazy translations."""

    inner: _core.LazyTranslation

    @classmethod
    def _from_rust(cls, rust_lazy_access: _core.LazyTranslation) -> LazyTranslation:
        instance = cls.__new__(cls)
        instance.inner = rust_lazy_access
        return instance


def x(value: float) -> LazyTranslation:
    """Creates a lazy translation along the X axis.

    Args:
        value: The translation distance.

    Returns:
        A lazy translation.
    """
    lazy_access_rust = _core.x(value)
    return LazyTranslation._from_rust(lazy_access_rust)


def y(value: float) -> LazyTranslation:
    """Creates a lazy translation along the Y axis.

    Args:
        value: The translation distance.

    Returns:
        A lazy translation.
    """
    lazy_access_rust = _core.y(value)
    return LazyTranslation._from_rust(lazy_access_rust)


def z(value: float) -> LazyTranslation:
    """Creates a lazy translation along the Z axis.

    Args:
        value: The translation distance.

    Returns:
        A lazy translation.
    """
    lazy_access_rust = _core.z(value)
    return LazyTranslation._from_rust(lazy_access_rust)


class LazyRotation:
    """Type wrapper for lazy rotations."""

    inner: _core.LazyRotation

    @classmethod
    def _from_rust(cls, rust_lazy_access: _core.LazyRotation) -> LazyRotation:
        instance = cls.__new__(cls)
        instance.inner = rust_lazy_access
        return instance


def rx(value: float) -> LazyRotation:
    """Creates a lazy rotation around the X axis.

    Args:
        value: The rotation angle in radians.

    Returns:
        A lazy rotation.
    """
    lazy_access_rust = _core.rx(value)
    return LazyRotation._from_rust(lazy_access_rust)


def ry(value: float) -> LazyRotation:
    """Creates a lazy rotation around the Y axis.

    Args:
        value: The rotation angle in radians.

    Returns:
        A lazy rotation.
    """
    lazy_access_rust = _core.ry(value)
    return LazyRotation._from_rust(lazy_access_rust)


def rz(value: float) -> LazyRotation:
    """Creates a lazy rotation around the Z axis.

    Args:
        value: The rotation angle in radians.

    Returns:
        A lazy rotation.
    """
    lazy_access_rust = _core.rz(value)
    return LazyRotation._from_rust(lazy_access_rust)

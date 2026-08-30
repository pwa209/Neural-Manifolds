"""Neural-manifold regime analysis package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("neural-manifolds")
except PackageNotFoundError:  # source checkout
    __version__ = "0.1.0"

__all__ = ["__version__"]

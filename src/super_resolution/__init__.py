"""The super-resolution Python project. Defines version using version defined in toml file."""
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError  # type: ignore
    from importlib_metadata import version


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

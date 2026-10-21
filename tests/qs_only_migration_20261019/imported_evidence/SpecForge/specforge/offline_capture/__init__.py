"""Dependency-lazy in-process SGLang target-capture utilities."""

from importlib import import_module

__all__ = [
    "OfflineCaptureBatch",
    "OfflineEagle3CaptureBatch",
    "OfflineEagle3SGLangCapture",
    "OfflineSGLangCapture",
    "LocalSGLangCapture",
    "load_local_capture",
    "load_offline_capture",
    "load_offline_eagle3_capture",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.sglang"), name)
    globals()[name] = value
    return value

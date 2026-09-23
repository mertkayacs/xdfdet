"""Frame-level deepfake detection with Grad-CAM region analysis (EfficientNet-B4, FaceForensics++)."""

from .config import CONFIGS

__version__ = "1.0.0"
__all__ = ["CONFIGS", "RELEASED", "build_model", "load_model"]


def __getattr__(name):
    # TensorFlow is imported on first use, so `xdfdet --help` stays fast.
    if name in ("RELEASED", "build_model", "load_model"):
        from . import model
        return getattr(model, name)
    raise AttributeError(name)

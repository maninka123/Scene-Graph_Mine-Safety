"""Paper-aligned underground mine perception-to-reasoning pipeline."""

from .config import load_config
from .pipeline import MineSafetyPipeline

__all__ = ["MineSafetyPipeline", "load_config"]
__version__ = "1.0.0"

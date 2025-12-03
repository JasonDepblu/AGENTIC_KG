"""
Patches for third-party library compatibility issues.
"""

from .adk_litellm_patch import apply_patches

__all__ = ["apply_patches"]

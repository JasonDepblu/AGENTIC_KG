"""
Patches for Google ADK LiteLLM integration.

Fixes compatibility issues between Google ADK and LiteLLM when handling
tool calls with empty arguments.
"""

import json
from typing import Optional

from google.genai import types


def apply_patches():
    """Apply all necessary patches for ADK LiteLLM compatibility."""
    _patch_function_call_args()


def _patch_function_call_args():
    """
    Patch google.genai.types.Part.from_function_call to handle string args.

    LiteLLM returns function arguments as strings, but the ADK expects dicts.
    This patch ensures proper conversion.
    """
    original_from_function_call = types.Part.from_function_call

    @classmethod
    def patched_from_function_call(cls, *, name: str, args: dict) -> 'types.Part':
        """Patched version that handles string args."""
        # If args is a string, parse it as JSON
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        # Ensure args is a dict
        if args is None:
            args = {}

        return original_from_function_call.__func__(cls, name=name, args=args)

    types.Part.from_function_call = patched_from_function_call

    # Also patch FunctionCall directly
    original_init = types.FunctionCall.__init__

    def patched_init(self, *args, **kwargs):
        """Patched FunctionCall init to handle string args."""
        if 'args' in kwargs and isinstance(kwargs['args'], str):
            try:
                kwargs['args'] = json.loads(kwargs['args'])
            except json.JSONDecodeError:
                kwargs['args'] = {}
        return original_init(self, *args, **kwargs)

    types.FunctionCall.__init__ = patched_init


# Auto-apply patches when module is imported
apply_patches()

"""
Patches for Google ADK LiteLLM integration.

Fixes compatibility issues between Google ADK and LiteLLM when handling
tool calls with empty arguments.
"""

import json
import re
from typing import Optional

from google.genai import types


def apply_patches():
    """Apply all necessary patches for ADK LiteLLM compatibility."""
    _patch_function_call_args()


def _fix_json_string(s: str) -> str:
    """
    Fix common JSON issues from LLM outputs.

    - Replace Chinese quotes with ASCII quotes
    - Fix unescaped special characters
    - Handle malformed JSON from LLM
    """
    replacements = {
        """: '"',
        """: '"',
        "'": "'",
        "'": "'",
        "，": ",",
        "：": ":",
        "【": "[",
        "】": "]",
        "｛": "{",
        "｝": "}",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    # Remove stray backslashes before quotes that commonly break JSON
    s = re.sub(r"\\(?=[\"'])", "", s)

    # Fix trailing commas before closing brackets (common LLM mistake)
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # Fix missing commas between array elements or object properties
    # e.g., "value1""value2" -> "value1","value2"
    s = re.sub(r'"\s*"(?=[^:,\]}])', '","', s)

    return s.strip()


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
            parsed = None
            last_error = None
            for candidate in (args, _fix_json_string(args)):
                try:
                    parsed = json.loads(candidate)
                    break
                except json.JSONDecodeError as e:
                    last_error = e
                    continue
            if parsed is None:
                # Show the full problematic JSON for debugging
                print(f"Warning: Failed to parse function args for {name}")
                print(f"  Error: {last_error}")
                print(f"  Original: {args}")
                print(f"  Fixed: {_fix_json_string(args)}")
                args = {}
            else:
                args = parsed

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
            original_args = kwargs['args']
            parsed = None
            last_error = None
            for candidate in (original_args, _fix_json_string(original_args)):
                try:
                    parsed = json.loads(candidate)
                    break
                except json.JSONDecodeError as e:
                    last_error = e
                    continue
            if parsed is None:
                # Show the full problematic JSON for debugging
                print(f"Warning: Failed to parse FunctionCall args")
                print(f"  Error: {last_error}")
                print(f"  Original: {original_args}")
                print(f"  Fixed: {_fix_json_string(original_args)}")
                kwargs['args'] = {}
            else:
                kwargs['args'] = parsed
        return original_init(self, *args, **kwargs)

    types.FunctionCall.__init__ = patched_init


# Auto-apply patches when module is imported
apply_patches()

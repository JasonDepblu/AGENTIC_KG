"""
Common tool utilities for Agentic KG.

Provides helper functions used across all tools.
"""

from typing import Any, Dict, Optional

from google.adk.tools import ToolContext


def tool_success(key: str, result: Any) -> Dict[str, Any]:
    """
    Create a success response for a tool.

    Args:
        key: The key name for the result data
        result: The result data to include

    Returns:
        Dictionary with 'status': 'success' and the result under the given key
    """
    return {
        'status': 'success',
        key: result
    }


def tool_error(message: str) -> Dict[str, Any]:
    """
    Create an error response for a tool.

    Args:
        message: The error message to include

    Returns:
        Dictionary with 'status': 'error' and the error message
    """
    return {
        'status': 'error',
        'error_message': message
    }


def get_state_value(
    tool_context: ToolContext,
    key: str,
    default: Any = None
) -> Any:
    """
    Get a value from the tool context state.

    Args:
        tool_context: The ADK ToolContext
        key: The state key to retrieve
        default: Default value if key not found

    Returns:
        The value from state or the default
    """
    return tool_context.state.get(key, default)


def set_state_value(
    tool_context: ToolContext,
    key: str,
    value: Any
) -> None:
    """
    Set a value in the tool context state.

    Args:
        tool_context: The ADK ToolContext
        key: The state key to set
        value: The value to store
    """
    tool_context.state[key] = value


def require_state_value(
    tool_context: ToolContext,
    key: str,
    error_message: Optional[str] = None
) -> Any:
    """
    Get a required value from state, raising an error if not found.

    Args:
        tool_context: The ADK ToolContext
        key: The state key to retrieve
        error_message: Custom error message if key not found

    Returns:
        The value from state

    Raises:
        KeyError: If the key is not found in state
    """
    if key not in tool_context.state:
        msg = error_message or f"Required state key '{key}' not found"
        raise KeyError(msg)
    return tool_context.state[key]


def validate_file_path(file_path: str) -> Dict[str, Any]:
    """
    Validate that a file path is relative (not absolute).

    Args:
        file_path: The file path to validate

    Returns:
        None if valid, or an error dict if invalid
    """
    from pathlib import Path

    if Path(file_path).is_absolute():
        return tool_error(
            "File path must be relative to the import directory. "
            "Make sure the file is from the list of available files."
        )
    return None

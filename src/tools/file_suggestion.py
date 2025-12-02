"""
File suggestion tools for Agentic KG.

Tools for listing, sampling, and managing files for knowledge graph import.
"""

from itertools import islice
from pathlib import Path
from typing import Any, Dict, List

from google.adk.tools import ToolContext

from .common import tool_success, tool_error, validate_file_path
from ..config import get_neo4j_import_dir

# State keys
ALL_AVAILABLE_FILES = "all_available_files"
SUGGESTED_FILES = "suggested_files"
APPROVED_FILES = "approved_files"
SEARCH_RESULTS = "search_results"


def list_available_files(tool_context: ToolContext) -> Dict[str, Any]:
    """
    List files available for knowledge graph construction.

    All files are relative to the Neo4j import directory.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and list of available file names,
        or error if import directory not configured
    """
    import_dir_path = get_neo4j_import_dir()

    if not import_dir_path:
        return tool_error(
            "NEO4J_IMPORT_DIR not configured. "
            "Set it in .env file or environment variables."
        )

    import_dir = Path(import_dir_path)

    if not import_dir.exists():
        return tool_error(f"Import directory does not exist: {import_dir_path}")

    # Get list of relative file names
    file_names = [
        str(x.relative_to(import_dir))
        for x in import_dir.rglob("*")
        if x.is_file()
    ]

    # Save to state for later inspection
    tool_context.state[ALL_AVAILABLE_FILES] = file_names

    return tool_success(ALL_AVAILABLE_FILES, file_names)


def sample_file(
    file_path: str,
    tool_context: ToolContext,
    max_lines: int = 100
) -> Dict[str, Any]:
    """
    Sample a file by reading its content as text.

    Treats any file as text and reads up to a maximum number of lines.

    Args:
        file_path: File to sample, relative to the import directory
        tool_context: ADK ToolContext for state management
        max_lines: Maximum number of lines to read (default: 100)

    Returns:
        Dictionary with status and file content,
        or error if file cannot be read
    """
    # Validate file path is relative
    validation_error = validate_file_path(file_path)
    if validation_error:
        return validation_error

    import_dir_path = get_neo4j_import_dir()
    if not import_dir_path:
        return tool_error("NEO4J_IMPORT_DIR not configured.")

    import_dir = Path(import_dir_path)
    full_path = import_dir / file_path

    if not full_path.exists():
        return tool_error(
            f"File does not exist in import directory. "
            f"Make sure {file_path} is from the list of available files."
        )

    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            lines = list(islice(file, max_lines))
            content = ''.join(lines)
            return tool_success("content", content)
    except Exception as e:
        return tool_error(f"Error reading file {file_path}: {e}")


def search_file(
    file_path: str,
    query: str,
    tool_context: ToolContext = None
) -> Dict[str, Any]:
    """
    Search a text file for lines containing the given query string.

    Simple grep-like functionality that works with any text file.
    Search is always case insensitive.

    Args:
        file_path: Path to the file, relative to the Neo4j import directory
        query: The string to search for
        tool_context: ADK ToolContext (optional)

    Returns:
        Dictionary with status and search results including matching lines
    """
    import_dir_path = get_neo4j_import_dir()
    if not import_dir_path:
        return tool_error("NEO4J_IMPORT_DIR not configured.")

    import_dir = Path(import_dir_path)
    full_path = import_dir / file_path

    if not full_path.exists():
        return tool_error(f"File does not exist: {file_path}")

    if not full_path.is_file():
        return tool_error(f"Path is not a file: {file_path}")

    # Handle empty query
    if not query:
        return tool_success(SEARCH_RESULTS, {
            "metadata": {
                "path": file_path,
                "query": query,
                "lines_found": 0
            },
            "matching_lines": []
        })

    matching_lines = []
    search_query = query.lower()

    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file, 1):
                if search_query in line.lower():
                    matching_lines.append({
                        "line_number": i,
                        "content": line.strip()
                    })
    except Exception as e:
        return tool_error(f"Error searching file {file_path}: {e}")

    result_data = {
        "metadata": {
            "path": file_path,
            "query": query,
            "lines_found": len(matching_lines)
        },
        "matching_lines": matching_lines
    }

    return tool_success(SEARCH_RESULTS, result_data)


def set_suggested_files(
    suggest_files: List[str],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Set the suggested files to be used for data import.

    Args:
        suggest_files: List of file paths to suggest
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and the suggested files list
    """
    tool_context.state[SUGGESTED_FILES] = suggest_files
    return tool_success(SUGGESTED_FILES, suggest_files)


def get_suggested_files(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get the files suggested for data import.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and the suggested files list
    """
    if SUGGESTED_FILES not in tool_context.state:
        return tool_error("suggested_files not set. Set suggested files first.")

    return tool_success(SUGGESTED_FILES, tool_context.state[SUGGESTED_FILES])


def approve_suggested_files(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Approve the suggested files for further processing.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and the approved files list,
        or error if suggested files not set
    """
    if SUGGESTED_FILES not in tool_context.state:
        return tool_error(
            "Current files have not been set. "
            "Take no action other than to inform user."
        )

    tool_context.state[APPROVED_FILES] = tool_context.state[SUGGESTED_FILES]
    return tool_success(APPROVED_FILES, tool_context.state[APPROVED_FILES])


def get_approved_files(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get the files that have been approved for import.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and the approved files list,
        or error if not approved yet
    """
    if APPROVED_FILES not in tool_context.state:
        return tool_error(
            "approved_files not set. "
            "Ask the user to approve the file suggestions."
        )

    return tool_success(APPROVED_FILES, tool_context.state[APPROVED_FILES])

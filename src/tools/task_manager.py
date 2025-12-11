"""
Task Manager for Agentic KG Pipeline.

Provides functionality for:
- Task ID generation and directory structure management
- Checkpoint persistence to disk for resume support
- Task listing and status tracking
"""

import json
import uuid
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from google.adk.tools import ToolContext

from .common import (
    get_state_value,
    set_state_value,
    tool_success,
    tool_error,
)
from ..config import get_neo4j_import_dir

logger = logging.getLogger(__name__)

# State keys for task management
TASK_ID_KEY = "task_id"
TASK_CONFIG_KEY = "task_config"

# Keys to persist in checkpoint
CHECKPOINT_STATE_KEYS = [
    "approved_user_goal",
    "approved_files",
    "approved_target_schema",
    "extraction_progress",
    "targeted_extraction_results",
    "targeted_entity_maps",
    "targeted_relationship_data",
    "generated_files",
    "current_phase",
    "schema_cleaned_file",
]


def generate_task_id() -> str:
    """
    Generate a unique task ID with timestamp.

    Format: task_YYYYMMDD_HHMMSS_<8-char-uuid>
    Example: task_20241210_150400_abc12345
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"task_{timestamp}_{short_uuid}"


def get_tasks_base_dir() -> Path:
    """Get the base directory for all tasks."""
    import_dir = get_neo4j_import_dir()
    if import_dir:
        return Path(import_dir) / "tasks"
    return Path("data") / "tasks"


def create_task_directory(task_id: str) -> Path:
    """
    Create directory structure for a new task.

    Creates:
        tasks/{task_id}/
            raw/        - Original input files
            cleaned/    - Schema-cleaned data
            extracted/  - Entity/relationship CSVs
    """
    tasks_dir = get_tasks_base_dir()
    task_dir = tasks_dir / task_id

    # Create subdirectories
    (task_dir / "raw").mkdir(parents=True, exist_ok=True)
    (task_dir / "cleaned").mkdir(parents=True, exist_ok=True)
    (task_dir / "extracted").mkdir(parents=True, exist_ok=True)

    logger.info(f"Created task directory: {task_dir}")
    return task_dir


def init_task(
    tool_context: ToolContext,
    copy_source_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Initialize a new task with directory structure.

    Args:
        tool_context: ADK ToolContext
        copy_source_files: Optional list of source file paths to copy to raw/

    Returns:
        Task configuration dictionary
    """
    task_id = generate_task_id()
    task_dir = create_task_directory(task_id)

    task_config = {
        "task_id": task_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "base_dir": str(task_dir),
        "raw_dir": str(task_dir / "raw"),
        "clean_dir": str(task_dir / "cleaned"),
        "extracted_dir": str(task_dir / "extracted"),
        "checkpoint_file": str(task_dir / "checkpoint.json"),
        "status": "initialized",
    }

    # Copy source files if provided
    if copy_source_files:
        copied_files = []
        for src_path in copy_source_files:
            src = Path(src_path)
            if src.exists():
                dst = task_dir / "raw" / src.name
                shutil.copy2(src, dst)
                copied_files.append(str(dst))
                logger.info(f"Copied source file: {src} -> {dst}")
        task_config["source_files"] = copied_files

    # Store in state
    set_state_value(tool_context, TASK_ID_KEY, task_id)
    set_state_value(tool_context, TASK_CONFIG_KEY, task_config)

    # Persist initial checkpoint
    save_checkpoint(tool_context)

    logger.info(f"Initialized new task: {task_id}")
    return tool_success("task_initialized", task_config)


def get_task_config(tool_context: ToolContext) -> Optional[Dict[str, Any]]:
    """Get current task configuration from state."""
    return get_state_value(tool_context, TASK_CONFIG_KEY)


def get_task_id(tool_context: ToolContext) -> Optional[str]:
    """Get current task ID from state."""
    return get_state_value(tool_context, TASK_ID_KEY)


def save_checkpoint(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Persist current state to checkpoint file.

    Should be called after:
    - Task initialization
    - Each entity extraction completion
    - Each relationship extraction completion
    - Schema approval
    - Phase transitions

    Returns:
        Success/error result
    """
    task_config = get_task_config(tool_context)
    if not task_config:
        # No active task, skip checkpoint
        logger.debug("No active task, skipping checkpoint save")
        return tool_success("checkpoint_skipped", {"reason": "No active task"})

    checkpoint_file = Path(task_config["checkpoint_file"])

    # Build checkpoint from current state
    checkpoint = {
        "task_id": task_config["task_id"],
        "version": "1.0",
        "created_at": task_config["created_at"],
        "updated_at": datetime.now().isoformat(),
        "status": task_config.get("status", "in_progress"),
        "task_config": task_config,
    }

    # Collect state values to persist
    for key in CHECKPOINT_STATE_KEYS:
        value = get_state_value(tool_context, key)
        if value is not None:
            checkpoint[key] = value

    # Write checkpoint file
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2, default=str)
        logger.debug(f"Checkpoint saved: {checkpoint_file}")
        return tool_success("checkpoint_saved", {"file": str(checkpoint_file)})
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        return tool_error(f"Failed to save checkpoint: {e}")


def load_checkpoint(
    task_id: str,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Load checkpoint and restore state for task resumption.

    Args:
        task_id: ID of the task to resume
        tool_context: ADK ToolContext

    Returns:
        Loaded checkpoint data or error
    """
    tasks_dir = get_tasks_base_dir()
    checkpoint_file = tasks_dir / task_id / "checkpoint.json"

    if not checkpoint_file.exists():
        return tool_error(f"Checkpoint not found for task: {task_id}")

    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
    except Exception as e:
        return tool_error(f"Failed to load checkpoint: {e}")

    # Restore task config
    task_config = checkpoint.get("task_config", {})
    if not task_config:
        # Reconstruct task config from checkpoint
        task_dir = tasks_dir / task_id
        task_config = {
            "task_id": task_id,
            "created_at": checkpoint.get("created_at"),
            "updated_at": checkpoint.get("updated_at"),
            "base_dir": str(task_dir),
            "raw_dir": str(task_dir / "raw"),
            "clean_dir": str(task_dir / "cleaned"),
            "extracted_dir": str(task_dir / "extracted"),
            "checkpoint_file": str(checkpoint_file),
            "status": checkpoint.get("status", "resumed"),
        }

    task_config["status"] = "resumed"
    task_config["resumed_at"] = datetime.now().isoformat()

    # Restore state values
    set_state_value(tool_context, TASK_ID_KEY, task_id)
    set_state_value(tool_context, TASK_CONFIG_KEY, task_config)

    for key in CHECKPOINT_STATE_KEYS:
        if key in checkpoint:
            set_state_value(tool_context, key, checkpoint[key])

    logger.info(f"Loaded checkpoint for task: {task_id}")
    return tool_success("checkpoint_loaded", {
        "task_id": task_id,
        "status": checkpoint.get("status"),
        "phase": checkpoint.get("current_phase"),
        "created_at": checkpoint.get("created_at"),
        "updated_at": checkpoint.get("updated_at"),
    })


def update_task_status(
    tool_context: ToolContext,
    status: str,
) -> Dict[str, Any]:
    """
    Update task status and save checkpoint.

    Args:
        tool_context: ADK ToolContext
        status: New status (e.g., "in_progress", "completed", "error", "rollback_pending")
    """
    task_config = get_task_config(tool_context)
    if not task_config:
        return tool_error("No active task")

    task_config["status"] = status
    task_config["updated_at"] = datetime.now().isoformat()
    set_state_value(tool_context, TASK_CONFIG_KEY, task_config)

    return save_checkpoint(tool_context)


def list_tasks() -> Dict[str, Any]:
    """
    List all tasks with their status.

    Returns:
        List of task summaries sorted by creation time (newest first)
    """
    tasks_dir = get_tasks_base_dir()

    if not tasks_dir.exists():
        return tool_success("tasks", [])

    tasks = []
    for task_dir in tasks_dir.iterdir():
        if task_dir.is_dir():
            checkpoint_file = task_dir / "checkpoint.json"
            if checkpoint_file.exists():
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        checkpoint = json.load(f)

                    # Count extracted files
                    extracted_dir = task_dir / "extracted"
                    extracted_count = len(list(extracted_dir.glob("*.csv"))) if extracted_dir.exists() else 0

                    tasks.append({
                        "task_id": checkpoint.get("task_id", task_dir.name),
                        "created_at": checkpoint.get("created_at"),
                        "updated_at": checkpoint.get("updated_at"),
                        "status": checkpoint.get("status"),
                        "phase": checkpoint.get("current_phase"),
                        "extracted_files": extracted_count,
                    })
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint for {task_dir.name}: {e}")
                    tasks.append({
                        "task_id": task_dir.name,
                        "status": "unknown",
                        "error": str(e),
                    })

    # Sort by creation time (newest first)
    tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return tool_success("tasks", tasks)


def get_task_checkpoint(task_id: str) -> Dict[str, Any]:
    """
    Get full checkpoint data for a specific task.

    Args:
        task_id: Task ID to query

    Returns:
        Full checkpoint data or error
    """
    tasks_dir = get_tasks_base_dir()
    checkpoint_file = tasks_dir / task_id / "checkpoint.json"

    if not checkpoint_file.exists():
        return tool_error(f"Task not found: {task_id}")

    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        return tool_success("checkpoint", checkpoint)
    except Exception as e:
        return tool_error(f"Failed to read checkpoint: {e}")


def delete_task(task_id: str) -> Dict[str, Any]:
    """
    Delete a task and all its data.

    Args:
        task_id: Task ID to delete

    Returns:
        Success/error result
    """
    tasks_dir = get_tasks_base_dir()
    task_dir = tasks_dir / task_id

    if not task_dir.exists():
        return tool_error(f"Task not found: {task_id}")

    try:
        shutil.rmtree(task_dir)
        logger.info(f"Deleted task: {task_id}")
        return tool_success("task_deleted", {"task_id": task_id})
    except Exception as e:
        return tool_error(f"Failed to delete task: {e}")


# Utility functions for getting task directories

def get_raw_dir(tool_context: ToolContext) -> Optional[str]:
    """Get the raw data directory for current task."""
    task_config = get_task_config(tool_context)
    return task_config.get("raw_dir") if task_config else None


def get_clean_dir(tool_context: ToolContext) -> Optional[str]:
    """Get the cleaned data directory for current task."""
    task_config = get_task_config(tool_context)
    return task_config.get("clean_dir") if task_config else None


def get_extracted_dir(tool_context: ToolContext) -> Optional[str]:
    """Get the extracted data directory for current task."""
    task_config = get_task_config(tool_context)
    return task_config.get("extracted_dir") if task_config else None

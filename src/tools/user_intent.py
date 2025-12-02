"""
User intent tools for Agentic KG.

Tools for capturing and managing user goals for knowledge graph construction.
"""

from typing import Any, Dict

from google.adk.tools import ToolContext

from .common import tool_success, tool_error

# State keys
PERCEIVED_USER_GOAL = "perceived_user_goal"
APPROVED_USER_GOAL = "approved_user_goal"


def set_perceived_user_goal(
    kind_of_graph: str,
    graph_description: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Set the perceived user's goal for knowledge graph construction.

    This tool encourages collaboration with the user by first setting
    a perceived goal that can be reviewed before approval.

    Args:
        kind_of_graph: 2-3 word definition of the kind of graph,
                      e.g., "recent US patents" or "social network"
        graph_description: A single paragraph description of the graph,
                          summarizing the user's intent
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and the perceived user goal data
    """
    user_goal_data = {
        "kind_of_graph": kind_of_graph,
        "graph_description": graph_description
    }
    tool_context.state[PERCEIVED_USER_GOAL] = user_goal_data
    return tool_success(PERCEIVED_USER_GOAL, user_goal_data)


def get_perceived_user_goal(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get the current perceived user goal.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and the perceived user goal,
        or error if not set
    """
    if PERCEIVED_USER_GOAL not in tool_context.state:
        return tool_error(
            "perceived_user_goal not set. "
            "Use set_perceived_user_goal first to capture the user's intent."
        )

    return tool_success(PERCEIVED_USER_GOAL, tool_context.state[PERCEIVED_USER_GOAL])


def approve_perceived_user_goal(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Approve the perceived user goal upon user confirmation.

    This tool records the perceived user goal as the approved user goal
    after the user has explicitly approved it.

    Only call this tool if the user has explicitly approved the perceived user goal.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and the approved user goal,
        or error if perceived goal not set
    """
    # Trust, but verify - require perceived goal was set before approving
    if PERCEIVED_USER_GOAL not in tool_context.state:
        return tool_error(
            "perceived_user_goal not set. "
            "Set perceived user goal first, or ask clarifying questions if you are unsure."
        )

    tool_context.state[APPROVED_USER_GOAL] = tool_context.state[PERCEIVED_USER_GOAL]
    return tool_success(APPROVED_USER_GOAL, tool_context.state[APPROVED_USER_GOAL])


def get_approved_user_goal(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get the approved user goal.

    This tool returns the user's approved goal, which contains
    the kind of graph and its description.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and the approved user goal,
        or error if not approved yet
    """
    if APPROVED_USER_GOAL not in tool_context.state:
        return tool_error(
            "approved_user_goal not set. "
            "Ask the user to clarify their goal (kind of graph and description)."
        )

    return tool_success(APPROVED_USER_GOAL, tool_context.state[APPROVED_USER_GOAL])


def clear_user_goal(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Clear both perceived and approved user goals.

    Useful for starting over with goal definition.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status indicating goals were cleared
    """
    tool_context.state.pop(PERCEIVED_USER_GOAL, None)
    tool_context.state.pop(APPROVED_USER_GOAL, None)
    return tool_success("message", "User goals have been cleared.")

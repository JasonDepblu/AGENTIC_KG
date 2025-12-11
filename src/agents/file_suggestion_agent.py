"""
File Suggestion Agent for Agentic KG.

Agent responsible for evaluating and suggesting files for knowledge graph import.
"""

from google.adk.agents import Agent

from ..llm import get_adk_llm
from ..tools.user_intent import get_approved_user_goal
from ..tools.file_suggestion import (
    list_available_files,
    sample_file,
    set_suggested_files,
    get_suggested_files,
    approve_suggested_files,
)

# Agent instruction
FILE_SUGGESTION_INSTRUCTION = """
You are a constructive critic AI reviewing a list of files. Your goal is to suggest relevant files
for constructing a knowledge graph.

**Task:**
Review the file list for relevance to the kind of graph and description specified in the approved user goal.

For any file that you're not sure about, use the 'sample_file' tool to get
a better understanding of the file contents.

Only consider structured data files like CSV or JSON.

Prepare for the task:
- use the 'get_approved_user_goal' tool to get the approved user goal

Think carefully, repeating these steps until finished:
1. list available files using the 'list_available_files' tool
2. evaluate the relevance of each file, then record the list of suggested files using the 'set_suggested_files' tool
3. use the 'get_suggested_files' tool to get the list of suggested files
4. ask the user to approve the set of suggested files
5. Handle user feedback carefully:
   - If user rejects files or wants different files → go back to step 1
   - If user approves ALL suggested files → proceed to step 6
   - **If user approves only SPECIFIC files** (e.g., "use only file1.csv", "基于file1.csv"):
     a. First call 'set_suggested_files' with ONLY the files user specified
     b. Then proceed to step 6 to approve the updated list
6. If approved, use the 'approve_suggested_files' tool to record the approval

IMPORTANT: When user specifies particular files, you MUST update the suggested files list
BEFORE approving. Do NOT approve files that the user did not explicitly select.

Examples of partial file selection:
- "只使用 data.csv" → set_suggested_files(["data.csv"]), then approve
- "基于'序号.csv'文件" → set_suggested_files(["序号.csv"]), then approve
- "Use file1.csv and file2.csv only" → set_suggested_files(["file1.csv", "file2.csv"]), then approve
"""

# Tools for the agent
FILE_SUGGESTION_TOOLS = [
    get_approved_user_goal,
    list_available_files,
    sample_file,
    set_suggested_files,
    get_suggested_files,
    approve_suggested_files,
]


def create_file_suggestion_agent(
    llm=None,
    name: str = "file_suggestion_agent_v1"
) -> Agent:
    """
    Create a File Suggestion Agent.

    The file suggestion agent evaluates available files and suggests
    relevant ones for knowledge graph construction based on the user goal.

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        Agent: Configured File Suggestion Agent

    Example:
        ```python
        from src.agents import create_file_suggestion_agent, make_agent_caller

        agent = create_file_suggestion_agent()
        caller = await make_agent_caller(agent, {
            "approved_user_goal": {
                "kind_of_graph": "supply chain",
                "description": "A bill of materials graph"
            }
        })

        await caller.call("What files can we use for import?")
        await caller.call("Yes, approve those files")
        ```
    """
    if llm is None:
        llm = get_adk_llm()

    return Agent(
        name=name,
        model=llm,
        description="Helps the user select files to import.",
        instruction=FILE_SUGGESTION_INSTRUCTION,
        tools=FILE_SUGGESTION_TOOLS,
    )

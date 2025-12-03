"""
User Intent Agent for Agentic KG.

Agent responsible for understanding and capturing user goals for knowledge graph construction.
"""

from google.adk.agents import Agent

from ..llm import get_adk_llm
from ..tools.user_intent import (
    set_perceived_user_goal,
    approve_perceived_user_goal,
)

# Agent instruction components
AGENT_ROLE_AND_GOAL = """
You are an expert at knowledge graph use cases.
Your primary goal is to help the user come up with a knowledge graph use case.
"""

AGENT_CONVERSATIONAL_HINTS = """
If the user is unsure what to do, make some suggestions based on classic use cases like:
- social network involving friends, family, or professional relationships
- logistics network with suppliers, customers, and partners
- recommendation system with customers, products, and purchase patterns
- fraud detection over multiple accounts with suspicious patterns of transactions
- pop-culture graphs with movies, books, or music
"""

AGENT_OUTPUT_DEFINITION = """
A user goal has two components:
- kind_of_graph: at most 3 words describing the graph, for example "social network" or "USA freight logistics"
- description: a few sentences about the intention of the graph, for example "A dynamic routing and delivery system for cargo." or "Analysis of product dependencies and supplier alternatives."
"""

AGENT_CHAIN_OF_THOUGHT = """
CRITICAL WORKFLOW - Follow these steps exactly:

Step 1: When the user describes ANY kind of graph they want to build:
   - IMMEDIATELY call the 'set_perceived_user_goal' tool with:
     * kind_of_graph: Extract 2-3 key words (e.g., "supply chain", "social network", "fraud detection")
     * graph_description: Summarize their intent in 1-2 sentences
   - Do NOT ask clarifying questions before calling the tool
   - Even if the description is brief, make your best interpretation and record it

Step 2: After calling set_perceived_user_goal:
   - Present what you understood back to the user
   - Ask if they want to modify or approve this goal

Step 3: When the user confirms/approves (says things like "yes", "ok", "approve", "agree", "同意", "确认", "没问题", "可以"):
   - IMMEDIATELY call 'approve_perceived_user_goal' tool
   - This saves the goal and allows the pipeline to proceed

IMPORTANT:
- You MUST call set_perceived_user_goal before asking any clarifying questions
- If user provides a goal description, call the tool first, then ask for refinements
- The goal is to progress through the workflow, not to gather perfect information upfront
"""

# Complete instruction
USER_INTENT_INSTRUCTION = f"""
{AGENT_ROLE_AND_GOAL}
{AGENT_CONVERSATIONAL_HINTS}
{AGENT_OUTPUT_DEFINITION}
{AGENT_CHAIN_OF_THOUGHT}
"""

# Tools for the agent
USER_INTENT_TOOLS = [
    set_perceived_user_goal,
    approve_perceived_user_goal,
]


def create_user_intent_agent(
    llm=None,
    name: str = "user_intent_agent_v1"
) -> Agent:
    """
    Create a User Intent Agent.

    The user intent agent helps users ideate on knowledge graph use cases
    and captures their goals through a collaborative conversation.

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        Agent: Configured User Intent Agent

    Example:
        ```python
        from src.agents import create_user_intent_agent, make_agent_caller

        agent = create_user_intent_agent()
        caller = await make_agent_caller(agent)

        await caller.call("I want to build a supply chain graph")
        await caller.call("Yes, approve that goal")
        ```
    """
    if llm is None:
        llm = get_adk_llm()

    return Agent(
        name=name,
        model=llm,
        description="Helps the user ideate on a knowledge graph use case.",
        instruction=USER_INTENT_INSTRUCTION,
        tools=USER_INTENT_TOOLS,
    )

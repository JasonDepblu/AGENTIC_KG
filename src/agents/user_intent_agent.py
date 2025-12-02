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
Think carefully and collaborate with the user:
1. Understand the user's goal, which is a kind_of_graph with description
2. Ask clarifying questions as needed
3. When you think you understand their goal, use the 'set_perceived_user_goal' tool to record your perception
4. Present the perceived user goal to the user for confirmation
5. If the user agrees, use the 'approve_perceived_user_goal' tool to approve the user goal. This will save the goal in state under the 'approved_user_goal' key.
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

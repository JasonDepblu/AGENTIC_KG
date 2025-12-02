"""
Knowledge Graph Builder Agent for Agentic KG.

Agent responsible for executing the construction plan to build the knowledge graph.
"""

from google.adk.agents import Agent

from ..llm import get_adk_llm
from ..tools.kg_construction import (
    get_proposed_construction_plan,
    get_approved_construction_plan,
    construct_domain_graph,
)

# Agent instruction
KG_BUILDER_INSTRUCTION = """
You are a knowledge graph construction specialist. Your role is to execute
the approved construction plan and build the knowledge graph in Neo4j.

**Task:**
Execute the approved construction plan to create nodes and relationships
in the Neo4j database.

Workflow:
1. Use 'get_approved_construction_plan' to retrieve the construction plan
2. Review the plan and explain what will be constructed
3. Execute 'construct_domain_graph' to build the graph
4. Report the results to the user

Important notes:
- Nodes are created first, then relationships (to ensure referenced nodes exist)
- Uniqueness constraints are automatically created for node labels
- The construction uses batch processing (1000 rows per transaction)
- Report any errors that occur during construction
"""


def create_kg_builder_agent(
    llm=None,
    name: str = "kg_builder_agent_v1"
) -> Agent:
    """
    Create a Knowledge Graph Builder Agent.

    The KG builder agent executes the approved construction plan
    to create nodes and relationships in Neo4j.

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        Agent: Configured KG Builder Agent

    Example:
        ```python
        from src.agents import create_kg_builder_agent, make_agent_caller

        agent = create_kg_builder_agent()
        caller = await make_agent_caller(agent, {
            "approved_construction_plan": { ... }
        })

        await caller.call("Build the knowledge graph")
        ```
    """
    if llm is None:
        llm = get_adk_llm()

    return Agent(
        name=name,
        model=llm,
        description="Builds the knowledge graph by executing the approved construction plan.",
        instruction=KG_BUILDER_INSTRUCTION,
        tools=[
            get_proposed_construction_plan,
            get_approved_construction_plan,
            construct_domain_graph,
        ],
    )


# Direct construction function (non-agent)
def build_domain_graph(construction_plan: dict, graphdb=None) -> dict:
    """
    Build the domain graph directly without using an agent.

    This is useful for programmatic construction when agent interaction
    is not needed.

    Args:
        construction_plan: Dictionary of construction rules
        graphdb: Optional Neo4j client instance

    Returns:
        Dictionary with construction results
    """
    return construct_domain_graph(construction_plan, graphdb)

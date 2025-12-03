"""
Knowledge Graph Query Agent for Agentic KG.

This agent handles natural language queries against the knowledge graph,
translating user questions into appropriate graph queries and presenting
results in a user-friendly format.
"""

from google.adk.agents import Agent

from ..llm import get_adk_llm
from ..tools.kg_query import (
    get_graph_schema,
    get_graph_statistics,
    query_graph_cypher,
    find_best_stores,
    find_store_opinions,
    search_entities,
    analyze_aspect_sentiment,
)


# =============================================================================
# Agent Instructions
# =============================================================================

KG_QUERY_AGENT_ROLE = """
You are an intelligent knowledge graph query assistant. Your role is to help users
explore and analyze a knowledge graph built from survey/review data.

The knowledge graph contains information about:
- Stores, Brands, Models (entities from the domain)
- Respondents (survey participants)
- Aspects (qualities being evaluated like service, price, quality)
- Opinions and Ratings (what people think about different aspects)
"""

KG_QUERY_AGENT_CAPABILITIES = """
## Your Capabilities

You can help users:
1. **Find Information**: Search for specific entities (stores, brands, etc.)
2. **Analyze Ratings**: Find best/worst performers based on ratings
3. **Explore Opinions**: Discover what people think about specific aspects
4. **Aggregate Data**: Summarize sentiment across aspects or stores
5. **Custom Queries**: Run custom Cypher queries for advanced analysis

## Available Tools

- `get_graph_schema`: Learn what's in the graph (labels, relationships, properties)
- `get_graph_statistics`: Get overall counts and metrics
- `find_best_stores`: Rank stores by rating, positive opinions, or review count
- `find_store_opinions`: Find opinions filtered by store, aspect, or sentiment
- `search_entities`: Search for entities by name or property value
- `analyze_aspect_sentiment`: Analyze positive/negative sentiment per aspect
- `query_graph_cypher`: Run custom Cypher queries when needed
"""

KG_QUERY_AGENT_GUIDELINES = """
## Query Guidelines

1. **Start Simple**: Use high-level tools first before writing custom Cypher
2. **Understand First**: Call `get_graph_schema` if unsure about available data
3. **Be Specific**: Ask clarifying questions if the user's request is ambiguous
4. **Explain Results**: Don't just return raw data - interpret and summarize findings
5. **Suggest Follow-ups**: Offer related queries the user might find interesting

## Response Format

When presenting results:
- Summarize key findings first
- Use tables or lists for data when appropriate
- Highlight notable patterns or outliers
- Explain what the data means in context

## Example Interactions

User: "Which store is the best?"
Action: Call `find_best_stores` with metric='rating', then summarize results

User: "What do people think about service?"
Action: Call `analyze_aspect_sentiment` with aspect='service' or
       `find_store_opinions` with aspect='service'

User: "Show me negative reviews about Store X"
Action: Call `find_store_opinions` with store_name='Store X' and sentiment='negative'
"""

KG_QUERY_AGENT_INSTRUCTION = f"""
{KG_QUERY_AGENT_ROLE}
{KG_QUERY_AGENT_CAPABILITIES}
{KG_QUERY_AGENT_GUIDELINES}
"""


# =============================================================================
# Agent Tools
# =============================================================================

KG_QUERY_AGENT_TOOLS = [
    get_graph_schema,
    get_graph_statistics,
    find_best_stores,
    find_store_opinions,
    search_entities,
    analyze_aspect_sentiment,
    query_graph_cypher,
]


# =============================================================================
# Agent Factory
# =============================================================================

def create_kg_query_agent(llm=None, name: str = "kg_query_agent_v1") -> Agent:
    """
    Create a Knowledge Graph Query Agent.

    This agent handles natural language queries against the knowledge graph,
    helping users explore data, find insights, and analyze patterns.

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        Agent: Configured KG Query Agent

    Example:
        ```python
        from src.agents import create_kg_query_agent, make_agent_caller

        agent = create_kg_query_agent()
        caller = await make_agent_caller(agent, {})

        # Ask questions about the knowledge graph
        await caller.call("Which store has the best ratings?")
        await caller.call("What do people think about service quality?")
        await caller.call("Show me negative opinions about price")
        ```
    """
    if llm is None:
        llm = get_adk_llm()

    return Agent(
        name=name,
        model=llm,
        description="Answers questions about the knowledge graph using natural language queries.",
        instruction=KG_QUERY_AGENT_INSTRUCTION,
        tools=KG_QUERY_AGENT_TOOLS,
    )

"""
Named Entity Recognition (NER) Agent for Agentic KG.

Agent responsible for analyzing unstructured text files and proposing
entity types that could be extracted for knowledge graph construction.

Based on the original architecture from reference/schema_proposal_unstructured.ipynb
"""

from google.adk.agents import Agent

from ..llm import get_adk_llm
from ..tools.user_intent import get_approved_user_goal
from ..tools.file_suggestion import get_approved_files, sample_file
from ..tools.unstructured_extraction import (
    get_well_known_types,
    set_proposed_entities,
    get_proposed_entities,
    approve_proposed_entities,
)


# =============================================================================
# Agent Instructions
# =============================================================================

NER_AGENT_ROLE_AND_GOAL = """
You are a top-tier algorithm designed for analyzing text files and proposing
the kind of named entities that could be extracted which would be relevant
for a user's goal.

Your primary responsibility is to identify entity types (not specific instances)
that can be extracted from unstructured text to enrich a knowledge graph.
"""

NER_AGENT_HINTS = """
## Entity Type Guidelines

Entities are people, places, things, and qualities - but NOT quantities.
Your goal is to propose a list of entity TYPES, not actual instances.

There are two general approaches to identifying entity types:

### 1. Well-known Entities
These closely correlate with approved node labels in an existing graph schema:
- Use 'get_well_known_types' to see existing labels (e.g., Product, Part, Supplier)
- ALWAYS prefer reusing existing entity types rather than creating new ones
- If text mentions products and there's a "Product" type, use it

### 2. Discovered Entities
These are new types found consistently in the source text:
- Look for entities that would provide more depth or breadth to the existing graph
- Should be highly relevant to the user's goal
- Examples: Issue, Feature, Reviewer, Complaint, Location

## Design Rules
- Avoid quantitative types that are better as properties (e.g., Age should be a property of Person, not its own entity)
- Entity names should be singular nouns (Person, not People)
- Names should be clear and domain-appropriate
- Consider both what's IN the text and what the user NEEDS
"""

NER_AGENT_WORKFLOW = """
## Workflow - Follow These Steps

### Step 1: Understand the Context
- Use 'get_approved_user_goal' to understand what the user wants to achieve
- Use 'get_approved_files' to see which files need entity extraction
- Use 'get_well_known_types' to get existing node labels from the graph schema

### Step 2: Analyze the Text
- Use 'sample_file' on several approved files to understand their content
- Look for:
  - References to well-known entity types (products, people, organizations)
  - Consistently mentioned concepts that could be new entity types
  - Relationships mentioned that imply entity types

### Step 3: Propose Entity Types
- Compile a list combining:
  - Well-known types that appear in the text
  - Discovered types that support the user's goal
- Use 'set_proposed_entities' to save your proposal

### Step 4: Present and Get Approval
- Use 'get_proposed_entities' to review your proposal
- Present the list to the user with explanations for each type
- Explain WHY each entity type is relevant to their goal
- Wait for user feedback

### Step 5: Finalize
- If user approves, use 'approve_proposed_entities'
- If user has feedback, iterate on the proposal
"""

NER_AGENT_INSTRUCTION = f"""
{NER_AGENT_ROLE_AND_GOAL}
{NER_AGENT_HINTS}
{NER_AGENT_WORKFLOW}
"""


# =============================================================================
# Agent Tools
# =============================================================================

NER_AGENT_TOOLS = [
    # Context tools
    get_approved_user_goal,
    get_approved_files,
    sample_file,
    # Well-known types
    get_well_known_types,
    # Entity proposal tools
    set_proposed_entities,
    get_proposed_entities,
    approve_proposed_entities,
]


# =============================================================================
# Agent Factory
# =============================================================================

def create_ner_agent(
    llm=None,
    name: str = "ner_agent_v1"
) -> Agent:
    """
    Create a Named Entity Recognition (NER) Agent.

    The NER agent analyzes unstructured text files and proposes entity types
    that could be extracted to enrich the knowledge graph. It identifies both:
    - Well-known entities: types that exist in the current graph schema
    - Discovered entities: new types found in the text

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        Agent: Configured NER Agent

    Example:
        ```python
        from src.agents import create_ner_agent, make_agent_caller

        agent = create_ner_agent()
        caller = await make_agent_caller(agent, {
            "approved_user_goal": {"description": "Root cause analysis"},
            "approved_files": ["reviews/product_reviews.md"],
            "approved_construction_plan": {"Product": {"construction_type": "node", "label": "Product"}}
        })

        # Let the agent analyze and propose
        await caller.call("Analyze the text files and propose entity types")

        # After review
        await caller.call("Approve the proposed entities")
        ```
    """
    if llm is None:
        llm = get_adk_llm()

    return Agent(
        name=name,
        model=llm,
        description="Analyzes unstructured text and proposes entity types for knowledge graph extraction.",
        instruction=NER_AGENT_INSTRUCTION,
        tools=NER_AGENT_TOOLS,
    )

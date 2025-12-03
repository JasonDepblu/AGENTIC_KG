"""
Fact Type Extraction Agent for Agentic KG.

Agent responsible for analyzing unstructured text files and proposing
fact types (subject-predicate-object triples) that could be extracted
for knowledge graph construction.

Based on the original architecture from reference/schema_proposal_unstructured.ipynb
"""

from google.adk.agents import Agent

from ..llm import get_adk_llm
from ..tools.user_intent import get_approved_user_goal
from ..tools.file_suggestion import get_approved_files, sample_file
from ..tools.unstructured_extraction import (
    get_approved_entities,
    add_proposed_fact,
    get_proposed_facts,
    remove_proposed_fact,
    approve_proposed_facts,
)


# =============================================================================
# Agent Instructions
# =============================================================================

FACT_AGENT_ROLE_AND_GOAL = """
You are a top-tier algorithm designed for analyzing text files and proposing
the type of facts that could be extracted from text that would be relevant
for a user's goal.

Your primary responsibility is to identify relationship types (fact types)
between approved entity types, forming subject-predicate-object triples.
"""

FACT_AGENT_HINTS = """
## Fact Type Guidelines

Do NOT propose specific individual facts, but instead propose the general TYPE
of facts that would be relevant for the user's goal.

### What is a Fact Type?
A fact type is a triplet of (Subject, Predicate, Object) where:
- **Subject**: An approved entity type (e.g., Product, Person)
- **Predicate**: A relationship label that describes how they connect
- **Object**: Another approved entity type

### Examples
- Good: `(Product, has_issue, Issue)` - describes a type of relationship
- Bad: `(Stockholm Chair, has_issue, Missing Screws)` - too specific

### Design Rules

1. **Only use approved entity types**
   - Subject and object MUST be from the approved entity types
   - Use 'get_approved_entities' to see the list
   - Do NOT propose new entity types here

2. **Predicate must appear in text**
   - The relationship described by the predicate should be found in the source text
   - Don't guess or infer relationships not present in the data

3. **Optimize for user's goal**
   - Focus on relationships that directly support what the user wants to achieve
   - For root cause analysis: focus on problem-related relationships
   - For social networks: focus on connection relationships

4. **Be concise**
   - Predicates should be short, clear verbs or verb phrases
   - Use snake_case: `has_issue`, `includes_feature`, `mentions`
   - Avoid redundant words: use `reviewed` not `reviewed_by_user`
"""

FACT_AGENT_WORKFLOW = """
## Workflow - Follow These Steps

### Step 1: Understand the Context
- Use 'get_approved_user_goal' to understand what facts would be valuable
- Use 'get_approved_files' to see which files to analyze
- Use 'get_approved_entities' to get the list of entity types you can use

### Step 2: Analyze for Relationships
- Use 'sample_file' on several approved files
- Look for how entities relate to each other:
  - What does Subject do to/with Object?
  - How are entities connected in sentences?
  - What verbs link entity mentions?

### Step 3: Propose Fact Types
- For each relationship pattern found, use 'add_proposed_fact' with:
  - approved_subject_label: The subject entity type
  - proposed_predicate_label: The relationship name
  - approved_object_label: The object entity type
- Add multiple fact types as you find them

### Step 4: Review and Present
- Use 'get_proposed_facts' to see all proposed fact types
- Present them to the user with:
  - The triplet: (Subject, Predicate, Object)
  - Why this relationship matters for their goal
  - Example from the text (if helpful)
- Wait for user feedback

### Step 5: Iterate and Finalize
- If user wants to remove a fact type, use 'remove_proposed_fact'
- If user wants more, analyze again and add more
- When user approves, use 'approve_proposed_facts'
"""

FACT_AGENT_INSTRUCTION = f"""
{FACT_AGENT_ROLE_AND_GOAL}
{FACT_AGENT_HINTS}
{FACT_AGENT_WORKFLOW}
"""


# =============================================================================
# Agent Tools
# =============================================================================

FACT_AGENT_TOOLS = [
    # Context tools
    get_approved_user_goal,
    get_approved_files,
    sample_file,
    # Entity constraint
    get_approved_entities,
    # Fact proposal tools
    add_proposed_fact,
    get_proposed_facts,
    remove_proposed_fact,
    approve_proposed_facts,
]


# =============================================================================
# Agent Factory
# =============================================================================

def create_fact_type_agent(
    llm=None,
    name: str = "fact_type_agent_v1"
) -> Agent:
    """
    Create a Fact Type Extraction Agent.

    The Fact Type agent analyzes unstructured text files and proposes fact types
    (subject-predicate-object triples) that represent relationships between
    approved entity types.

    Prerequisites:
    - NER Agent must have run first to produce approved_entity_types

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        Agent: Configured Fact Type Agent

    Example:
        ```python
        from src.agents import create_fact_type_agent, make_agent_caller

        agent = create_fact_type_agent()
        caller = await make_agent_caller(agent, {
            "approved_user_goal": {"description": "Root cause analysis"},
            "approved_files": ["reviews/product_reviews.md"],
            "approved_entity_types": ["Product", "Issue", "Feature"]
        })

        # Let the agent analyze and propose
        await caller.call("Analyze the text and propose fact types")

        # After review
        await caller.call("Approve the proposed fact types")
        ```
    """
    if llm is None:
        llm = get_adk_llm()

    return Agent(
        name=name,
        model=llm,
        description="Analyzes unstructured text and proposes fact types (relationship triples) for knowledge graph extraction.",
        instruction=FACT_AGENT_INSTRUCTION,
        tools=FACT_AGENT_TOOLS,
    )

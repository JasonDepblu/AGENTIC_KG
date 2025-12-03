"""
Unstructured Data Agent for Agentic KG.

Parent agent that orchestrates the complete workflow for extracting
knowledge from unstructured text files:
1. NER Agent - Proposes and approves entity types
2. Fact Type Agent - Proposes and approves relationship types
3. Knowledge Extraction - Extracts entities and relationships using LLM
4. Entity Resolution - Connects extracted entities to existing domain graph

Based on the original architecture from:
- reference/schema_proposal_unstructured.ipynb (Lesson 7)
- reference/kg_construction_2.ipynb (Lesson 8)
"""

from google.adk.agents import Agent

from ..llm import get_adk_llm
from ..tools.user_intent import get_approved_user_goal
from ..tools.file_suggestion import get_approved_files, sample_file
from ..tools.unstructured_extraction import (
    # NER tools
    get_well_known_types,
    set_proposed_entities,
    get_proposed_entities,
    approve_proposed_entities,
    get_approved_entities,
    # Fact Type tools
    add_proposed_fact,
    get_proposed_facts,
    remove_proposed_fact,
    approve_proposed_facts,
    get_approved_facts,
)
from ..tools.kg_extraction import (
    build_entity_schema,
    get_entity_schema,
    # Simplified mode (fallback)
    process_unstructured_file,
    get_extraction_results,
    correlate_entities,
    get_correlation_results,
    create_correspondence_relationships,
    # GraphRAG mode (full pipeline)
    process_unstructured_file_graphrag,
    process_unstructured_file_auto,
    correlate_entities_graphrag,
    auto_correlate_all_entities,
    is_graphrag_available,
)


# =============================================================================
# Agent Instructions
# =============================================================================

UNSTRUCTURED_AGENT_ROLE = """
You are an expert knowledge extraction agent responsible for building
knowledge graphs from unstructured text files (markdown, text, PDFs, etc.)

Your role is to guide users through the complete extraction workflow:
1. Identify what entity types exist in the text
2. Identify what relationships connect those entities
3. Extract actual entities and relationships
4. Connect extracted knowledge to the existing domain graph
"""

UNSTRUCTURED_AGENT_PHASES = """
## Workflow Phases

### Phase 1: Entity Type Discovery (NER)
Goal: Identify what types of entities can be extracted

Tools:
- `get_approved_user_goal` - Understand user's purpose
- `get_approved_files` - See which files to process
- `get_well_known_types` - Get existing graph node labels
- `sample_file` - Read file contents to find entities
- `set_proposed_entities` - Save your proposal
- `get_proposed_entities` - Review proposal
- `approve_proposed_entities` - Finalize (after user approval)

Steps:
1. Understand the user's goal and what entities would support it
2. Check existing graph labels (well-known types) to reuse
3. Sample files to discover additional entity types
4. Propose a combined list of entity types
5. Present to user and iterate until approved

### Phase 2: Relationship Type Discovery (Fact Types)
Goal: Identify what relationships exist between entity types

Prerequisite: Phase 1 complete (approved_entity_types exists)

Tools:
- `get_approved_entities` - Get the entity types to work with
- `sample_file` - Find relationships in text
- `add_proposed_fact` - Add a (subject, predicate, object) fact type
- `get_proposed_facts` - Review all proposed fact types
- `remove_proposed_fact` - Remove unwanted fact types
- `approve_proposed_facts` - Finalize (after user approval)

Steps:
1. Review approved entity types
2. Sample files to find relationships between entities
3. For each relationship found, add a fact type
4. Present fact types to user
5. Iterate until approved

### Phase 3: Knowledge Extraction
Goal: Extract actual entities and relationships from files

Prerequisite: Phase 2 complete (approved_fact_types exists)

Tools:
- `build_entity_schema` - Create extraction schema from approved types
- `get_entity_schema` - Review the schema
- `process_unstructured_file` - Extract from a file
- `get_extraction_results` - See all extraction results

Steps:
1. Build the entity schema from approved entities and facts
2. For each approved file, run extraction
3. Review extraction results
4. Report summary to user

### Phase 4: Entity Resolution (Optional)
Goal: Connect extracted entities to existing domain graph

Prerequisite: Phase 3 complete AND domain graph exists

Tools:
- `correlate_entities` - Find matching entities using string similarity
- `get_correlation_results` - Review correlations
- `create_correspondence_relationships` - Create Neo4j relationships

Steps:
1. For entity types that exist in both extracted and domain data
2. Run correlation to find matches
3. Review correlation quality
4. Create CORRESPONDS_TO relationships to link graphs
"""

UNSTRUCTURED_AGENT_GUIDANCE = """
## Important Guidelines

### Entity Type Guidelines
- Entities are people, places, things, qualities (NOT quantities)
- Prefer reusing well-known types from existing graph
- Discovered entities should support the user's goal
- Use singular nouns (Person not People)
- Avoid types that should be properties (Age is a property of Person)

### Fact Type Guidelines
- Facts are (Subject, Predicate, Object) triples
- Subject and object MUST be approved entity types
- Predicate should appear in source text
- Use snake_case: has_issue, includes_feature
- Focus on relationships relevant to user's goal

### User Interaction
- Always present proposals before approving
- Explain WHY each entity/fact type is relevant
- Accept user feedback and iterate
- Don't rush - quality over speed

### Current Phase Detection
Check state to understand where we are:
- No approved_entity_types → Start Phase 1 (NER)
- Has approved_entity_types but no approved_fact_types → Start Phase 2 (Facts)
- Has both → Start Phase 3 (Extraction)
- Has extraction_results → Can do Phase 4 (Resolution)
"""

UNSTRUCTURED_AGENT_INSTRUCTION = f"""
{UNSTRUCTURED_AGENT_ROLE}
{UNSTRUCTURED_AGENT_PHASES}
{UNSTRUCTURED_AGENT_GUIDANCE}
"""


# =============================================================================
# Agent Tools - All tools from all phases
# =============================================================================

UNSTRUCTURED_AGENT_TOOLS = [
    # Context tools
    get_approved_user_goal,
    get_approved_files,
    sample_file,

    # Phase 1: NER tools
    get_well_known_types,
    set_proposed_entities,
    get_proposed_entities,
    approve_proposed_entities,
    get_approved_entities,

    # Phase 2: Fact Type tools
    add_proposed_fact,
    get_proposed_facts,
    remove_proposed_fact,
    approve_proposed_facts,
    get_approved_facts,

    # Phase 3: Extraction tools
    build_entity_schema,
    get_entity_schema,
    process_unstructured_file,  # Simplified mode
    process_unstructured_file_auto,  # Auto-select mode (prefers GraphRAG)
    get_extraction_results,

    # Phase 4: Resolution tools
    correlate_entities,  # Simplified mode (Python-based)
    correlate_entities_graphrag,  # GraphRAG mode (Neo4j APOC)
    auto_correlate_all_entities,  # Auto-correlate all entity types
    get_correlation_results,
    create_correspondence_relationships,
]


# =============================================================================
# Agent Factory
# =============================================================================

def create_unstructured_data_agent(
    llm=None,
    name: str = "unstructured_data_agent_v1"
) -> Agent:
    """
    Create an Unstructured Data Agent.

    This is the parent agent that orchestrates the complete workflow for
    extracting knowledge from unstructured text files. It handles:

    1. Entity Type Discovery (NER)
    2. Relationship Type Discovery (Fact Types)
    3. Knowledge Extraction using LLM
    4. Entity Resolution to connect with domain graph

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        Agent: Configured Unstructured Data Agent

    Example:
        ```python
        from src.agents import create_unstructured_data_agent, make_agent_caller

        agent = create_unstructured_data_agent()
        caller = await make_agent_caller(agent, {
            "approved_user_goal": {"description": "Root cause analysis from reviews"},
            "approved_files": ["product_reviews/chair_reviews.md"],
            "approved_construction_plan": {"Product": {"construction_type": "node", "label": "Product"}}
        })

        # Phase 1: Entity discovery
        await caller.call("Analyze the text files and propose entity types")
        await caller.call("Approve the proposed entities")

        # Phase 2: Relationship discovery
        await caller.call("Now propose fact types based on the approved entities")
        await caller.call("Approve the fact types")

        # Phase 3: Extraction
        await caller.call("Extract entities and relationships from all files")

        # Phase 4: Resolution (optional)
        await caller.call("Connect Product entities to the domain graph")
        ```
    """
    if llm is None:
        llm = get_adk_llm()

    return Agent(
        name=name,
        model=llm,
        description="Extracts knowledge from unstructured text files using LLM-powered entity and relationship extraction.",
        instruction=UNSTRUCTURED_AGENT_INSTRUCTION,
        tools=UNSTRUCTURED_AGENT_TOOLS,
    )


# =============================================================================
# Specialized Agent Factories
# =============================================================================

# Re-export the specialized agents for use in sub-agent workflows
from .ner_agent import create_ner_agent
from .fact_type_agent import create_fact_type_agent

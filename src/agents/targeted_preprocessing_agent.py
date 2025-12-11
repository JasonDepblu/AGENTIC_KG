"""
Targeted Preprocessing Agent for Schema-First Pipeline.

This agent extracts entities and relationships from raw data based on
the approved target schema. It only extracts what the schema defines.
"""

from typing import AsyncGenerator

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event, EventActions

from ..llm import get_adk_llm, get_adk_llm_flash
from ..tools.file_suggestion import get_approved_files, sample_file
from ..tools.schema_design import get_approved_target_schema
from ..tools.targeted_preprocessing import (
    get_schema_extraction_plan,
    clean_columns_for_schema,  # NEW: Schema-driven data cleaning
    extract_entities_for_node,
    extract_relationship_data,
    # Text extraction tools
    extract_entities_from_text_column,
    extract_relationships_from_text_column,
    deduplicate_entities_with_llm,
    # Output and completion
    save_extracted_data,
    get_extraction_summary,
    complete_targeted_preprocessing,
    generate_construction_rules,
    get_preprocessing_feedback,
    request_schema_revision,
)
from ..tools.common import set_progress_total


# Targeted Preprocessing Agent Instructions
PREPROCESSING_ROLE_AND_GOAL = """
You are a data extraction specialist. Your task is to extract entities and relationships
from raw data files based on an approved target schema.

The schema defines exactly what to extract - you should follow it precisely.

NOTE: Schema approval is already verified by the coordinator before this agent runs.
You can assume the schema is approved and proceed with extraction.

WORKFLOW:
1. Use 'get_approved_target_schema' to get the schema definition
2. Use 'get_preprocessing_feedback' to check for any feedback from the critic agent
3. If there are issues, address them before completing preprocessing
4. Pay attention to the user's messages for any corrections or guidance
"""

PREPROCESSING_HINTS = """
## Extraction Guidelines

### Entity Extraction
- For each node type in the schema, extract entities from the specified column
- Use the extraction_hints in the schema to find the correct column
- Each entity should have a unique ID following the pattern: NodeLabel_index (e.g., Brand_0, Brand_1, Model_0)
- Preserve the original value as the "name" property

### Relationship Extraction
- For rating columns (e.g., "打多少分"), extract RATES relationships
- For foreign key columns, extract direct relationships
- Include all properties defined in the schema

### Data Quality
- Skip null or empty values
- Clean whitespace from string values
- Validate that extracted data matches expected types

### Text Extraction (source_type: "text_extraction")
For nodes/relationships with source_type="text_extraction":
1. Use 'extract_entities_from_text_column' to extract entities from text
   - This uses LLM to identify entity mentions in text feedback
   - Entities are extracted with their mention counts
2. Use 'deduplicate_entities_with_llm' to merge similar entities (optional)
   - Merges semantically similar entities (e.g., "车身线条流畅" → "线条流畅")
3. Use 'extract_relationships_from_text_column' for text-based relationships
   - Links Respondent to extracted entities with sentiment

### Schema Revision (Rollback)
If you encounter issues that indicate the schema design needs to be revised:
- Schema expects a column that doesn't exist in the data
- Extraction hints don't match actual data patterns (e.g., wrong column_pattern)
- Discovered entity types not represented in the schema
- Relationship structure doesn't match data relationships

In these cases, use 'request_schema_revision' to roll back to the schema design phase.
Provide a clear reason and list of suggested changes for the schema design agent.
"""

PREPROCESSING_CHAIN_OF_THOUGHT = """
## WORKFLOW - Follow these steps:

### Step 0: Set Progress Total (Do This First!)
After reviewing the schema, call 'set_progress_total' to set expected operations:
- Count nodes and relationships in schema
- Estimate: (nodes × 2) + (relationships × 2) + 5 (for save/cleanup)
- Example: 4 nodes + 6 relationships → set_progress_total(total=25, description="Extracting entities and relationships")
- This enables accurate progress bar display in the UI

### Step 0.5: Clean Schema Columns
Before extracting any data, clean the columns mentioned in the schema:
1. Call 'clean_columns_for_schema' to clean only relevant columns
2. This automatically:
   - Replaces placeholder values (跳过, N/A, null, etc.) with empty
   - Converts text in rating columns to numbers (e.g., "非常好" → 10)
   - Strips whitespace from values
3. The cleaned file will be used for extraction

### Step 1: Review Extraction Plan
- Use 'get_approved_target_schema' to see the schema definition
- Use 'get_schema_extraction_plan' to see what needs to be extracted
- Use 'get_approved_files' to get the list of data files

### Step 2: Extract Entities
For each node type in the schema:
- Check the source_type in extraction_hints
- If source_type is "entity_selection", "column_header", or "rating_column_header":
  - Use 'extract_entities_for_node' with the appropriate file and node label
- If source_type is "text_extraction":
  - Use 'extract_entities_from_text_column' with file, node label, and column name
  - Optionally use 'deduplicate_entities_with_llm' to merge similar entities
- Check the extraction results and sample data
- Report any issues (column not found, no data, etc.)

### Step 3: Extract Relationships
For each relationship type in the schema:
- Check the source_type in extraction_hints
- If source_type is "rating_column", "entity_reference", or "foreign_key":
  - Use 'extract_relationship_data' with the appropriate file and relationship type
- If source_type is "text_extraction":
  - Use 'extract_relationships_from_text_column' with file, relationship type, and column name
- Verify the relationship data looks correct

### Step 4: Save and Validate
- Use 'save_extracted_data' to save all extracted data to CSV files
- Use 'get_extraction_summary' to review what was extracted
- Report the summary to the user

### Step 5: Generate Construction Rules
- Use 'generate_construction_rules' to create the construction plan
- This plan will be used by the KG Builder

### Step 6: Complete Preprocessing
- If the user approves the extraction:
  Call 'complete_targeted_preprocessing' to mark as done
- If there are issues:
  Report them and ask for guidance

## Example Interaction

Agent: "Based on the target schema, I need to extract:
- Brand entities from the '品牌是?' column
- Model entities from the '车型是?' column
- RATES relationships from rating columns

Let me start extraction..."

[Extract entities and relationships]

Agent: "Extraction complete. Summary:
- Brands: 5 unique entities
- Models: 12 unique entities
- RATES relationships: 850 records

Shall I proceed with saving and completing the preprocessing?"

User: "Yes, looks good"

Agent: [Calls complete_targeted_preprocessing]
"""

PREPROCESSING_INSTRUCTION = f"""
{PREPROCESSING_ROLE_AND_GOAL}
{PREPROCESSING_HINTS}
{PREPROCESSING_CHAIN_OF_THOUGHT}
"""


# Critic Agent Instructions
CRITIC_ROLE_AND_GOAL = """
You are a data quality validator. Review the extracted data to ensure
it matches the target schema and contains valid values.
"""

CRITIC_HINTS = """
## Validation Rules

### Entity Validation
1. Each entity type should have reasonable number of entries
2. Entity values should be meaningful (not descriptions, not numbers for name fields)
3. All entities should have their unique property set

### Relationship Validation
1. Relationship endpoints should reference existing entity IDs
2. Relationship properties should be of correct types (scores should be numbers)
3. No orphaned relationships

### Schema Conformance
1. All node types in schema should have extracted entities
2. All relationship types should have extracted relationships
3. Required properties should be present
"""

CRITIC_CHAIN_OF_THOUGHT = """
## Workflow

1. Get the extraction summary using 'get_extraction_summary'
2. Get the target schema using 'get_approved_target_schema'
3. Compare extracted data against schema requirements
4. Check for data quality issues

## CRITICAL: Response Format

Think carefully, then respond with ONE of these two formats:

- If extraction passes validation, respond with exactly one word: 'valid'
- If issues can be fixed by re-running extraction, respond with 'retry' followed by bullet list

### IMPORTANT: When to return "valid" vs "retry"

**Return "valid"** when:
- Extraction is complete and data quality is good
- Issues exist but require SCHEMA CHANGES (preprocessing agent should call `request_schema_revision`)

**Return "retry"** ONLY when:
- Issues can be fixed by re-running extraction with current schema
- Example: extraction was incomplete, data wasn't saved

Example good response:
valid

Example bad response:
retry
- Extraction was incomplete - only 5 of 10 node types extracted

DO NOT add any text after 'valid'. The word must appear alone.
Schema issues = return 'valid' (allow rollback), not 'retry'!
"""

CRITIC_INSTRUCTION = f"""
{CRITIC_ROLE_AND_GOAL}
{CRITIC_HINTS}
{CRITIC_CHAIN_OF_THOUGHT}
"""


# Tool lists
PREPROCESSING_TOOLS = [
    # Progress tracking
    set_progress_total,
    # Context tools
    get_approved_files,
    sample_file,
    get_approved_target_schema,
    get_schema_extraction_plan,
    clean_columns_for_schema,  # Schema-driven data cleaning
    # Standard extraction
    extract_entities_for_node,
    extract_relationship_data,
    # Text extraction tools
    extract_entities_from_text_column,
    extract_relationships_from_text_column,
    deduplicate_entities_with_llm,
    # Output and completion
    save_extracted_data,
    get_extraction_summary,
    complete_targeted_preprocessing,
    generate_construction_rules,
    get_preprocessing_feedback,
    request_schema_revision,
]

CRITIC_TOOLS = [
    get_approved_target_schema,
    get_extraction_summary,
]


def log_agent(callback_context: CallbackContext) -> None:
    """Log agent entry for debugging."""
    print(f"\n### Entering Agent: {callback_context.agent_name}")


class CheckPreprocessingStatus(BaseAgent):
    """
    Agent that checks preprocessing feedback and decides whether to continue or stop.

    IMPORTANT: When running inside a coordinator (schema_preprocess_coordinator),
    this agent should NOT escalate, as escalate signals propagate to the root
    and would exit the entire coordinator. Instead, it sets a state flag.

    Also checks:
    - skip_preprocessing flag - if True, preprocessing was skipped because schema wasn't approved
    - needs_schema_revision flag - if True, schema revision was requested, exit loop for rollback
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check feedback and yield escalation event if valid."""
        from ..tools.targeted_preprocessing import TARGETED_PREPROCESSING_COMPLETE, NEEDS_SCHEMA_REVISION

        # Check if preprocessing was skipped (schema not approved)
        skip_preprocessing = ctx.session.state.get("skip_preprocessing", False)
        if skip_preprocessing:
            # Don't mark as complete, just exit the loop gracefully
            # The coordinator will continue and the schema design loop will run again
            print(f"\n### {self.name}: Preprocessing skipped (schema not approved), exiting loop without marking complete")
            # Clear the feedback so it doesn't interfere with next iteration
            ctx.session.state["preprocessing_feedback"] = "skipped"
            yield Event(
                author=self.name,
                actions=EventActions(escalate=False)
            )
            return

        # Check if schema revision was requested - if so, exit loop for rollback
        needs_schema_revision = ctx.session.state.get(NEEDS_SCHEMA_REVISION, False)
        if needs_schema_revision:
            print(f"\n### {self.name}: Schema revision requested, exiting preprocessing loop for rollback")
            # Set feedback to indicate rollback is needed
            ctx.session.state["preprocessing_feedback"] = "rollback_requested"
            # Exit the loop - coordinator will handle the rollback
            yield Event(
                author=self.name,
                actions=EventActions(escalate=True)
            )
            return

        feedback = ctx.session.state.get("preprocessing_feedback", "valid")
        feedback_str = str(feedback).strip().lower()
        should_stop = (feedback_str == "valid")  # Exact match, not startswith

        if should_stop:
            # Mark preprocessing phase as complete
            ctx.session.state[TARGETED_PREPROCESSING_COMPLETE] = True
            print(f"\n### {self.name}: Preprocessing feedback valid, marking phase complete")
        else:
            print(f"\n### {self.name}: Preprocessing feedback='{feedback[:50] if feedback else 'empty'}...', continuing")

        # Check if we're running inside a coordinator
        running_in_coordinator = ctx.session.state.get("running_in_coordinator", False)

        if running_in_coordinator:
            # Don't escalate - let the coordinator handle the flow
            print(f"\n### {self.name}: Running in coordinator mode, NOT escalating")
            yield Event(
                author=self.name,
                actions=EventActions(escalate=False)
            )
        else:
            # Standalone mode - escalate normally
            yield Event(
                author=self.name,
                actions=EventActions(escalate=should_stop)
            )


def create_targeted_preprocessing_agent(
    llm=None,
    name: str = "targeted_preprocessing_agent_v1"
) -> LlmAgent:
    """
    Create a Targeted Preprocessing Agent.

    This agent extracts entities and relationships from raw data
    based on the approved target schema.

    Uses qwen-flash model by default for faster and cheaper processing.

    Args:
        llm: Optional LLM instance. If None, uses qwen-flash from get_adk_llm_flash()
        name: Agent name for identification

    Returns:
        LlmAgent: Configured Targeted Preprocessing Agent
    """
    if llm is None:
        llm = get_adk_llm_flash()  # Use qwen-flash for preprocessing

    return LlmAgent(
        name=name,
        model=llm,
        description="Extracts entities and relationships based on target schema",
        instruction=PREPROCESSING_INSTRUCTION,
        tools=PREPROCESSING_TOOLS,
        before_agent_callback=log_agent,
    )


def create_preprocessing_critic_agent(
    llm=None,
    name: str = "preprocessing_critic_v1"
) -> LlmAgent:
    """
    Create a Preprocessing Critic Agent.

    This agent validates extracted data for quality and schema conformance.

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        LlmAgent: Configured Preprocessing Critic Agent
    """
    if llm is None:
        llm = get_adk_llm()  # Use qwen-plus-latest for critic (needs better reasoning)

    return LlmAgent(
        name=name,
        model=llm,
        description="Validates extracted data quality and schema conformance",
        instruction=CRITIC_INSTRUCTION,
        tools=CRITIC_TOOLS,
        output_key="preprocessing_feedback",
        before_agent_callback=log_agent,
    )


def create_targeted_preprocessing_loop(
    llm=None,
    max_iterations: int = 2,
    name: str = "targeted_preprocessing_loop"
) -> LoopAgent:
    """
    Create a Targeted Preprocessing Loop Agent.

    This loop agent coordinates extraction and validation in an
    iterative refinement process.

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        max_iterations: Maximum refinement iterations
        name: Agent name for identification

    Returns:
        LoopAgent: Configured Targeted Preprocessing Loop

    Example:
        ```python
        from src.agents import create_targeted_preprocessing_loop, make_agent_caller

        loop = create_targeted_preprocessing_loop()
        caller = await make_agent_caller(loop, {
            "preprocessing_feedback": "",
            "approved_target_schema": {...},
            "approved_files": ["survey.xlsx"]
        })

        # Run extraction
        await caller.call("Extract entities and relationships according to the schema")

        # User approves
        await caller.call("Looks good, complete the preprocessing")
        ```
    """
    preprocessing_agent = create_targeted_preprocessing_agent(llm)
    critic_agent = create_preprocessing_critic_agent(llm)
    stop_checker = CheckPreprocessingStatus(name="PreprocessingStopChecker")

    return LoopAgent(
        name=name,
        description="Extracts and validates data based on target schema",
        max_iterations=max_iterations,
        sub_agents=[preprocessing_agent, critic_agent, stop_checker],
        before_agent_callback=log_agent,
    )

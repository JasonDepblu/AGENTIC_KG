"""
Schema Design Agent for Agentic KG.

Agent responsible for designing the target knowledge graph schema based on raw data analysis.
This is used in the SCHEMA_DESIGN phase of the Schema-First pipeline.
"""

from typing import AsyncGenerator

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event, EventActions

from ..llm import get_adk_llm
from ..tools.user_intent import get_approved_user_goal
from ..tools.file_suggestion import get_approved_files, sample_file, search_file
from ..tools.schema_design import (
    sample_raw_file_structure,
    detect_potential_entities,
    propose_node_type,
    propose_relationship_type,
    remove_node_type,
    remove_relationship_type,
    get_target_schema,
    get_schema_design_feedback,
    approve_target_schema,
    update_node_extraction_hints,
    update_relationship_extraction_hints,
    get_schema_revision_reason,
    standardize_column_names,
    # Text feedback column tools
    identify_text_feedback_columns,
    sample_text_column,
    analyze_text_column_entities,
    analyze_text_column_relationships,
    add_text_entity_to_schema,
    add_text_relationship_to_schema,
    get_text_analysis_summary,
)
from ..tools.common import set_progress_total


# Schema Design Agent Instructions - SIMPLIFIED for stability
DESIGN_ROLE_AND_GOAL = """
You are an expert at knowledge graph modeling. Your task is to design a target schema
that defines what nodes and relationships should be extracted from the raw data.

## YOUR ONLY JOB: Design the schema, NOT extract data!

## MANDATORY: Progress Tracking (DO THIS OR UI WILL BREAK!)
After calling 'sample_raw_file_structure', look at the 'suggested_progress_total' field in the response.
You MUST call 'set_progress_total' IMMEDIATELY with that value:
```
set_progress_total(total=<suggested_progress_total>, description="Designing schema")
```
DO NOT SKIP THIS! The UI progress bar relies on this call.

## LLM-Powered Domain Detection (Automatic)
The system uses LLM to automatically detect entity types and text feedback columns
from any domain (medical, automotive, engineering, etc.). The tools `detect_potential_entities`
and `identify_text_feedback_columns` will analyze column names and data samples using LLM
to provide domain-agnostic suggestions. You don't need to rely on hardcoded patterns.

## CRITICAL: Check for Rollback FIRST!
Before doing anything else, you MUST call 'get_schema_revision_reason' to check if this
is a rollback from preprocessing. If is_rollback=true:
1. READ the 'reason' carefully - it explains what went wrong in preprocessing
2. FOLLOW the 'suggested_changes' to fix the schema
3. These issues caused preprocessing to FAIL - they MUST be addressed!
4. Use 'update_node_extraction_hints' and 'update_relationship_extraction_hints'
   to fix column pattern mismatches

If you need to check feedback from the critic, call 'get_schema_design_feedback' tool.
"""

DESIGN_HINTS = """
## Schema Design Guidelines

### SUPPORTED source_type VALUES (CRITICAL!)
You MUST ONLY use these supported source_types:

| source_type | Use For | Example |
|-------------|---------|---------|
| `entity_selection` | Extract unique values from a column | Brand, Model, Store, Respondent |
| `column_header` | Extract entity names from column headers | Aspect names from *_score columns |
| `rating_column` | Extract ratings/scores from columns | RATES relationship with score property |
| `entity_reference` | Link respondent to selected entity | EVALUATED_BRAND, VISITED_STORE |
| `foreign_key` | Link entities via foreign key column | BELONGS_TO (Model → Brand) |
| `text_extraction` | Extract entities from text using LLM | Feature, Issue from feedback columns |

**DO NOT USE** any unsupported source_type!

### Node Types
- Every node MUST have a unique identifier property
- Maximum 10 node types allowed
- Focus on CORE entities that can be extracted with supported source_types:
  - Entity columns (brand, model, store) → Entity nodes with `entity_selection`
  - Rating/score columns → Aspect nodes with `column_header`
  - Respondent/ID columns → Respondent nodes with `entity_selection`

### Text Feedback Columns - OPTIONAL ADVANCED PROCESSING
Text feedback columns (e.g., `*_positive`, `*_negative`, `*_insight`) contain unstructured text.
You have TWO options:

**Option A: Skip text columns (default, simpler)**
- Do NOT create nodes or relationships for text feedback columns
- Use only structured data columns

**Option B: Extract entities from text (advanced, LIMIT TO 2-3 COLUMNS)**
Use the text extraction tools to analyze text feedback:
1. `identify_text_feedback_columns(file_path)` - Find text columns
2. **SELECT ONLY 2-3 representative columns** (e.g., one positive, one negative)
3. `sample_text_column(file_path, column_name)` - Get text samples
4. `analyze_text_column_entities(column_name, samples, category)` - Extract entity types with LLM
5. `analyze_text_column_relationships(column_name, samples, entity_types, category)` - Extract relationships
6. `add_text_entity_to_schema(entity_type, description, source_columns)` - Add to schema

**IMPORTANT**: Do NOT analyze every text column! Select 2-3 representative columns to extract
entity patterns, then apply the same schema to similar columns during preprocessing.

Use `source_type: "text_extraction"` for text-extracted nodes/relationships

### Node Extraction Hints (CRITICAL!)

**For Aspect nodes (rating dimensions):**
Aspect nodes represent rating categories extracted from COLUMN HEADERS (not column values).
Use `source_type: "column_header"` to extract aspect names from column headers.

```json
{
  "node_type": "Aspect",
  "unique_property": "aspect_name",
  "extraction_hints": {
    "source_type": "column_header",
    "column_pattern": "_score$",
    "name_regex": "(.+)_score$"
  }
}
```

**For entity nodes (Brand, Model, Store, Respondent):**
Use `source_type: "entity_selection"` to extract unique values from a column.

```json
{
  "node_type": "Brand",
  "unique_property": "brand_name",
  "extraction_hints": {
    "source_type": "entity_selection",
    "column_pattern": "brand"
  }
}
```

IMPORTANT: Aspect nodes MUST use "column_header" source_type, NOT "entity_selection"!

### Relationship Types
- Use UNIQUE names for each relationship (e.g., EVALUATED_BRAND, EVALUATED_MODEL)
- Common relationships: RATES (with score), EVALUATED, VISITED, BELONGS_TO
- **CRITICAL: Maximum 15 relationship types allowed!**
- **CONSOLIDATE relationships**: Use ONE generic 'RATES' relationship with column_pattern to match
  ALL rating columns (e.g., "_score$" matches appearance_score, interior_score, etc.)
- Do NOT create separate relationship types for each column (e.g., DON'T create RATES_APPEARANCE,
  RATES_INTERIOR, RATES_COMFORT separately - use ONE 'RATES' relationship instead!)

### Extraction Hints for Relationships (CRITICAL!)

**Type 1: Rating Relationships (RATES)**
For relationships that extract scores from rating columns:
```json
{
  "relationship_type": "RATES",
  "from_node": "Respondent",
  "to_node": "Aspect",
  "properties": ["score"],
  "extraction_hints": {
    "source_type": "rating_column",
    "column_pattern": "_score$",
    "respondent_column": "respondent_id"
  }
}
```

**Type 2: Entity Reference Relationships (EVALUATED, VISITED)**
For relationships that link respondents to their selected entities:
```json
{
  "relationship_type": "EVALUATED_BRAND",
  "from_node": "Respondent",
  "to_node": "Brand",
  "extraction_hints": {
    "source_type": "entity_reference",
    "column_pattern": "brand",
    "respondent_column": "respondent_id"
  }
}
```

**Type 3: Foreign Key Relationships (BELONGS_TO)**
For relationships that link entities via foreign key:
```json
{
  "relationship_type": "BELONGS_TO",
  "from_node": "Model",
  "to_node": "Brand",
  "extraction_hints": {
    "source_type": "foreign_key",
    "from_column": "model",
    "to_column": "brand"
  }
}
```

IMPORTANT:
- EVERY relationship MUST have extraction_hints with a SUPPORTED source_type!
- Only use: entity_selection, column_header, rating_column, entity_reference, foreign_key
- For entity_reference, column_pattern should match the standardized column name
- If extraction_hints are missing or use unsupported source_type, extraction will FAIL!
"""

DESIGN_CHAIN_OF_THOUGHT = """
## WORKFLOW - Execute these steps in order:

### Step 0: Check for Rollback (ALWAYS DO THIS FIRST!)
Call 'get_schema_revision_reason' BEFORE anything else.
- If is_rollback=true: This is a revision from preprocessing!
  - Read 'reason' and 'suggested_changes' carefully
  - Skip to Step 1b to fix the issues
- If is_rollback=false: Continue with normal workflow

### Step 1: Gather Context
1. Call 'get_approved_user_goal' to understand the user's goal
2. Call 'get_approved_files' to see available data files
3. Call 'get_target_schema' to check existing schema (if any)

### Step 1b: Handle Rollback Issues (if is_rollback=true)
If you detected a rollback in Step 0:
- Focus on fixing the specific issues in 'reason'
- Apply ALL 'suggested_changes' from preprocessing
- Common issues to fix:
  - Extraction hints don't match actual column names
  - Column patterns are incorrect (e.g., "品牌是?\\??$" vs "8、您本次调研的品牌是?")
  - Use 'update_node_extraction_hints' to fix node column patterns
  - Use 'update_relationship_extraction_hints' to fix relationship column patterns
- After fixing, proceed to Step 5 to present the revised schema

### Step 2: Analyze Data Files & Standardize Column Names (CRITICAL!)
For each approved file:
1. Call 'sample_raw_file_structure' to see ALL columns and sample data
2. ⚠️ **IMMEDIATELY after step 1** - Look at 'suggested_progress_total' in the response!
   Call 'set_progress_total' with that value RIGHT NOW:
   ```
   set_progress_total(total=<suggested_progress_total from response>, description="Designing schema")
   ```
   ⚠️ DO NOT SKIP THIS OR PROCEED WITHOUT CALLING IT!
3. Call 'detect_potential_entities' for automatic suggestions
4. **IMMEDIATELY standardize column names** - Build rename_map and call 'standardize_column_names':
   ```python
   {
       '序号': 'respondent_id',
       '8、您本次调研的品牌是？': 'brand',
       '该车型"外观设计"方面您会打多少分呢？': 'appearance_design_score',
       '60、劣势点': 'space_negative_feedback',
       '61、针对乘坐舒适中的"舒适配置"方面您觉得该车型优劣点的体验是？—优秀点': 'comfort_config_positive_feedback',
       '61、劣势点': 'comfort_config_negative_feedback',
       # ... map ALL columns to clean names
   }
   ```
   ⚠️ This creates a standardized file (e.g., "序号_standardized.csv")

5. **USE STANDARDIZED FILE for all subsequent operations!**
   After standardization, categorize columns using the NEW standardized names:
   - Entity columns (brand, model, store) → Entity nodes
   - ID/sequence columns (respondent_id) → Respondent nodes
   - Score columns (*_score) → Aspect nodes + RATES relationships
   - Text feedback columns (*_positive_feedback, *_negative_feedback) → **ONLY 2-3 representative columns**

6. For text analysis, use the STANDARDIZED file and column names:
   - identify_text_feedback_columns(file_path="序号_standardized.csv")
   - sample_text_column(file_path="序号_standardized.csv", column_name="comfort_config_positive_feedback")

   ⚠️ NEVER construct or guess column names! Use EXACT names from the standardized file.

### Step 3: Propose Node Types
Based on your analysis, propose node types that best capture the data:
- Consider the user's goal when deciding what to extract
- For each entity type, call 'propose_node_type' with:
  - node_type: label (e.g., "Respondent", "Brand", "Feedback", "Insight")
  - identifier_property: unique identifier
  - properties: list of properties
  - extraction_hints: {"source_type": "...", "column_pattern": "..."}

### Step 4: Propose Relationship Types
For each relationship, call 'propose_relationship_type' with:
- relationship_type: name (e.g., "RATES", "EVALUATED_BRAND")
- from_node: source node label
- to_node: target node label
- properties: list (e.g., ["score"] for RATES)
- extraction_hints: extraction guidance

### Step 5: Present Schema
1. Call 'get_target_schema' to get the complete schema
2. Present the schema summary to the user
3. Mention that column names have been standardized for reliable extraction
4. Ask for user approval

### Handling Feedback
If the feedback (shown above) starts with "retry":
- Read the issues listed in the feedback
- Fix each issue by calling the appropriate tools
- Then proceed to Step 5

If feedback is "valid" or user approves:
- The system will automatically approve the schema
"""

DESIGN_INSTRUCTION = f"""
{DESIGN_ROLE_AND_GOAL}
{DESIGN_HINTS}
{DESIGN_CHAIN_OF_THOUGHT}
"""


# Schema Critic Agent Instructions - SIMPLIFIED
CRITIC_ROLE_AND_GOAL = """
You are a schema validation expert. Review the proposed schema and decide if it's usable.

Be pragmatic, not pedantic. Only fail for CRITICAL issues that would break the graph.
"""

CRITIC_HINTS = """
## Validation Checklist

CRITICAL Issues (return "retry"):
1. No Respondent node for survey data
2. No relationships defined
3. Relationship references non-existent node
4. Node missing unique_property
5. **UNSUPPORTED source_type used** - Only these are valid:
   - entity_selection
   - column_header
   - rating_column
   - entity_reference
   - foreign_key
   If you see "text_column" or any other source_type, return "retry"!

NOT Critical (return "valid" anyway):
- Missing optional properties
- Extraction hints could be better (as long as source_type is valid)
- Minor naming differences
- Text feedback columns not being extracted (this is expected!)
"""

CRITIC_CHAIN_OF_THOUGHT = """
## Workflow
1. Call 'get_target_schema' to see the current schema
2. Call 'get_approved_files' to see the data files
3. Optionally sample files to verify

## Response Format - CRITICAL!

Think carefully, then respond with ONE of these two formats:

- If the schema looks good, respond with exactly one word: 'valid'
- If the schema has problems, respond with 'retry' followed by a concise bullet list of problems

Example good response:
valid

Example bad response:
retry
- Missing RATES relationship
- Respondent node not defined

DO NOT add any text after 'valid'. The word must appear alone.
"""

CRITIC_INSTRUCTION = f"""
{CRITIC_ROLE_AND_GOAL}
{CRITIC_HINTS}
{CRITIC_CHAIN_OF_THOUGHT}
"""


# Tool lists
SCHEMA_DESIGN_TOOLS = [
    # Progress tracking
    set_progress_total,
    # Context tools
    get_approved_user_goal,
    get_approved_files,
    sample_file,
    search_file,
    sample_raw_file_structure,
    detect_potential_entities,
    propose_node_type,
    propose_relationship_type,
    remove_node_type,
    remove_relationship_type,
    get_target_schema,
    get_schema_design_feedback,
    approve_target_schema,
    update_node_extraction_hints,
    update_relationship_extraction_hints,
    get_schema_revision_reason,
    standardize_column_names,
    # Text feedback column tools
    identify_text_feedback_columns,
    sample_text_column,
    analyze_text_column_entities,
    analyze_text_column_relationships,
    add_text_entity_to_schema,
    add_text_relationship_to_schema,
    get_text_analysis_summary,
]

SCHEMA_CRITIC_TOOLS = [
    get_approved_files,
    sample_file,
    search_file,
    sample_raw_file_structure,
    get_target_schema,
]


def log_agent(callback_context: CallbackContext) -> None:
    """Log agent entry for debugging."""
    print(f"\n### Entering Agent: {callback_context.agent_name}")


class CheckSchemaDesignStatus(BaseAgent):
    """
    Simple status checker - stops loop when feedback is "valid".

    Like the reference example: just check state and escalate.

    The pipeline's event handler now properly ignores sub-agent escalates
    for nested coordinators, so we can safely escalate here.
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check if critic returned exactly 'valid' and stop the loop if so."""
        from ..models.target_schema import APPROVED_TARGET_SCHEMA_KEY

        # Check critic feedback - EXACT match like reference implementation
        feedback = ctx.session.state.get("schema_design_feedback", "")
        feedback_str = str(feedback).strip().lower()
        feedback_valid = (feedback_str == "valid")  # Exact match, not startswith

        # Check if already approved (by user via approve_target_schema tool)
        schema_approved = ctx.session.state.get(APPROVED_TARGET_SCHEMA_KEY) is not None

        should_stop = schema_approved or feedback_valid

        if should_stop:
            # Note: NO auto-approval. Design agent should call approve_target_schema explicitly.
            ctx.session.state["schema_design_phase_done"] = True
            print(f"\n### {self.name}: Stopping loop - critic said 'valid' or schema already approved")
        else:
            print(f"\n### {self.name}: Continuing loop - feedback: '{feedback[:30] if feedback else 'empty'}...'")

        yield Event(
            author=self.name,
            actions=EventActions(escalate=should_stop)
        )


def create_schema_design_agent(
    llm=None,
    name: str = "schema_design_agent_v1"
) -> LlmAgent:
    """
    Create a Schema Design Agent.

    The schema design agent analyzes raw data files and designs a target
    knowledge graph schema that guides the preprocessing phase.

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        LlmAgent: Configured Schema Design Agent
    """
    if llm is None:
        llm = get_adk_llm()

    return LlmAgent(
        name=name,
        model=llm,
        description="Designs target knowledge graph schema based on raw data analysis",
        instruction=DESIGN_INSTRUCTION,
        tools=SCHEMA_DESIGN_TOOLS,
        before_agent_callback=log_agent,
    )


def create_schema_design_critic_agent(
    llm=None,
    name: str = "schema_design_critic_v1"
) -> LlmAgent:
    """
    Create a Schema Design Critic Agent.

    The critic agent reviews and validates proposed schemas,
    providing feedback for refinement.

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        LlmAgent: Configured Schema Design Critic Agent
    """
    if llm is None:
        llm = get_adk_llm()

    return LlmAgent(
        name=name,
        model=llm,
        description="Validates proposed target schema for completeness and correctness",
        instruction=CRITIC_INSTRUCTION,
        tools=SCHEMA_CRITIC_TOOLS,
        output_key="schema_design_feedback",
        before_agent_callback=log_agent,
    )


def create_schema_design_loop(
    llm=None,
    max_iterations: int = 2,  # Reduced from 4 to match reference examples
    name: str = "schema_design_loop"
) -> LoopAgent:
    """
    Create a Schema Design Loop Agent.

    This loop agent coordinates the schema design and critic agents
    in an iterative refinement process.

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        max_iterations: Maximum refinement iterations
        name: Agent name for identification

    Returns:
        LoopAgent: Configured Schema Design Loop

    Example:
        ```python
        from src.agents import create_schema_design_loop, make_agent_caller

        loop = create_schema_design_loop()
        caller = await make_agent_caller(loop, {
            "schema_design_feedback": "",
            "approved_user_goal": {"kind_of_graph": "survey analysis", "description": "..."},
            "approved_files": ["survey_data.xlsx"]
        })

        # First call - analyze data and propose schema
        await caller.call("Design a schema for this survey data")

        # User provides feedback
        await caller.call("Add Store as a node type too")

        # User approves
        await caller.call("Approve the schema")
        ```
    """
    design_agent = create_schema_design_agent(llm)
    critic_agent = create_schema_design_critic_agent(llm)
    stop_checker = CheckSchemaDesignStatus(name="SchemaDesignStopChecker")

    return LoopAgent(
        name=name,
        description="Designs target schema through iterative refinement with critic feedback",
        max_iterations=max_iterations,
        sub_agents=[design_agent, critic_agent, stop_checker],
        before_agent_callback=log_agent,
    )

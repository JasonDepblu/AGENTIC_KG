"""
Data Cleaning Agent for Agentic KG.

Agent responsible for cleaning raw data files before schema proposal.
This is the first phase in the new pipeline: Files → Process → Schema

Architecture:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Data Cleaning Loop (LoopAgent)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐     ┌──────────────────────┐     ┌────────────────┐   │
│  │ Data Cleaning    │ ──▶ │ Data Cleaning Critic │ ──▶ │ Stop Checker   │   │
│  │ Agent            │     │                      │     │                │   │
│  └──────────────────┘     └──────────────────────┘     └───────┬────────┘   │
│                                                                 │            │
│                           ┌─────────────────────────────────────┤            │
│                           │                                     │            │
│                     feedback="valid"?                       escalate?        │
│                           │                                     │            │
│                           ▼                                     ▼            │
│                       Continue Loop                         Exit Loop        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
"""

from typing import AsyncGenerator

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event, EventActions

from ..llm import get_adk_llm, get_adk_llm_flash
from ..tools.file_suggestion import get_approved_files, sample_file
from ..tools.data_cleaning import (
    analyze_file_quality,
    clean_file,
    get_cleaned_files,
    approve_data_cleaning,
    detect_column_types,
    detect_anomalies,
    convert_column_values,
    clean_urls,
    analyze_column_names,
    propose_column_renames,
    apply_column_renames,
    DATA_CLEANING_COMPLETE_KEY,
)

# State key for feedback from critic
DATA_CLEANING_FEEDBACK_KEY = "data_cleaning_feedback"


# Data Cleaning Agent Instructions
CLEANING_ROLE_AND_GOAL = """
You are a data quality expert. Your task is to analyze and clean raw data files
before they are used for knowledge graph schema design.

Your goal is to:
1. Analyze each approved file for quality issues
2. Identify and remove meaningless columns (empty, constant values, unnamed)
3. Identify and remove invalid rows (mostly empty, duplicates)
4. Detect column types to help guide schema design
5. Produce clean files ready for schema proposal

IMPORTANT:
- Check the state for 'data_cleaning_feedback' to see any feedback from the critic
- Pay attention to user feedback about what columns/rows are important
"""

CLEANING_HINTS = """
## Data Quality Issues to Look For

### Column Name Issues (IMPORTANT - Check First!)
Use `analyze_column_names` to detect:
1. **Leading/trailing whitespace**: "列名 " vs "列名"
2. **Duplicate number prefixes**: "10、10 xxx" should be "10、xxx"
3. **Invisible characters**: Non-printable characters in column names
4. **Multiple spaces**: "列  名" should be "列 名"

If issues are found:
1. Call `propose_column_renames` with suggested fixes
2. Show user the proposed changes and ask for confirmation
3. After user confirms, call `apply_column_renames` to fix the column names

### Columns to Remove
1. **Empty columns**: More than 95% null values
2. **Constant columns**: Only one unique value across all rows
3. **Unnamed columns**: Auto-generated names like "Unnamed: 0", "Column 1"
4. **Meaningless metadata**: Internal survey codes, timestamp columns not needed

### Rows to Remove
1. **Mostly empty rows**: More than 80% missing values
2. **Duplicate rows**: Exact copies of other rows
3. **Header rows in data**: Sometimes headers are repeated in the middle of data

### Values to Clean (IMPORTANT!)
These placeholder values should be replaced with null/empty:
- "(跳过)", "(跳過)", "跳过" - Survey skip markers
- "N/A", "NA", "n/a", "na" - Not applicable
- "-", "--", "---" - Dash placeholders
- "无", "無", "(空)", "空" - Empty markers
- "null", "NULL", "None" - Null values as strings

When calling clean_file with auto_clean=True, these values will automatically
be replaced with null. You can also specify additional values using replace_with_null.

### Column Type Detection
After cleaning, detect what type each column is:
- **identifier**: Unique ID for each row (序号, ID)
- **selection**: Multiple choice questions (品牌是?, 车型是?)
- **rating**: Numeric scores (打多少分)
- **open_text**: Free-form text responses (启发项?, 意见建议)

### Anomaly Detection (IMPORTANT!)
Use `detect_anomalies` to find:
1. **Text in rating columns**: e.g., "非常好" in a column expecting 1-10 scores
2. **Mixed data types**: Columns with both numeric and text values
3. **Numeric outliers**: Values outside the expected range
4. **URLs in columns**: Detect http/https URLs and www links in data

When text values are found in rating columns, the tool will suggest mappings like:
- "非常好", "很好" → 10
- "好", "不错" → 8
- "一般" → 5
- "差" → 3

Use `convert_column_values` to apply these mappings or set anomalies to null.

### URL Cleaning (IMPORTANT!)
Use `detect_anomalies` with `detect_urls=True` to find URLs in columns:
- If a column is mostly URLs (is_url_column=True), remove the entire column
- If URLs are mixed with content, strip URLs from the values

Use `clean_urls` to:
- `remove_url_columns`: Remove columns that are primarily URLs
- `strip_urls_from_columns`: Strip URLs from values but keep other content
- `auto_clean=True`: Automatically detect and clean all URLs
"""

CLEANING_CHAIN_OF_THOUGHT = """
## WORKFLOW - Follow these steps in order:

### Step 0: CHECK FOR FEEDBACK
Always start by checking if there is feedback from the critic or user.
If feedback exists, address the issues before proceeding.

### Step 1: Get Approved Files
Call 'get_approved_files' to see which files need to be cleaned.

### Step 1.5: Check Column Names (IMPORTANT!)
For each approved file:
1. Call 'analyze_column_names' to detect column name issues
2. If issues found:
   - Review the suggested fixes (whitespace, duplicate prefixes, etc.)
   - Call 'propose_column_renames' with the fixes
   - Present the proposed changes to user: "发现以下列名问题，建议修复：..."
   - Ask user: "是否确认这些列名修改？"
   - After user confirms, call 'apply_column_renames' to apply the fixes
3. If no issues, proceed to next step

### Step 2: Analyze Each File
For each approved file:
1. Call 'analyze_file_quality' to detect quality issues
2. Review the suggested columns and rows to remove
3. Present findings to user and ask for confirmation

### Step 3: Clean Files
For each file with quality issues:
1. Call 'clean_file' with auto_clean=True to:
   - Remove suggested columns and rows
   - Replace placeholder values like "(跳过)" with null
2. Or specify specific columns/rows if user has preferences

### Step 4: Detect Anomalies and URLs
For each cleaned file:
1. Call 'detect_anomalies' to find:
   - Text values in rating columns (e.g., "非常好" in score column)
   - Mixed data types
   - Numeric outliers
   - URLs in columns (http/https links, www links)
2. If text anomalies found:
   - If tool suggests mappings (e.g., "非常好" → 10), ask user to confirm
   - Call 'convert_column_values' to apply mappings
   - Or set anomalous values to null if no mapping makes sense
3. If URLs found:
   - For URL-only columns (is_url_column=True), suggest removing the column
   - For URLs mixed with content, suggest stripping URLs from values
   - Call 'clean_urls' to apply URL cleaning

### Step 5: Detect Column Types
For each cleaned file:
1. Call 'detect_column_types' to classify columns
2. This information helps the schema proposal phase

### Step 6: Present Results
1. Call 'get_cleaned_files' to show the cleaning summary
2. Ask user: "Do you approve the cleaned files for schema design?"

### Step 7: Handle User Response
- If user approves (says "ok", "approve", "确认", etc.):
  Call 'approve_data_cleaning' to finalize and proceed
- If user provides corrections:
  Adjust cleaning as requested and re-clean

## CRITICAL RULES:
1. ALWAYS analyze files before cleaning
2. Show user what will be removed/converted before doing it
3. NEVER approve without user confirmation
4. If critic finds issues, fix them before proceeding
5. Handle anomalies BEFORE finalizing the cleaned files
"""

CLEANING_INSTRUCTION = f"""
{CLEANING_ROLE_AND_GOAL}
{CLEANING_HINTS}
{CLEANING_CHAIN_OF_THOUGHT}
"""


# Critic Agent Instructions
CRITIC_ROLE_AND_GOAL = """
You are a data quality validation expert. Your job is to review the cleaning
decisions made by the data cleaning agent and ensure data integrity.
"""

CRITIC_HINTS = """
## Validation Rules

### CRITICAL - Must Pass
1. Brand, Model, Store selection columns should NEVER be removed
2. Rating columns should NEVER be removed
3. Truly empty columns (>95% null) should be removed
4. Duplicate rows should be handled

### OPTIONAL - Nice to Have (Do NOT fail on these)
1. Open text columns preserved (but not critical)
2. Meaningless metadata removed (but not critical)

### Column Type Classification (INFORMATIONAL ONLY)
Column type detection is automatic and heuristic-based. The classification is just
a HINT for the schema design phase. Do NOT fail validation based on column type
classification accuracy - the schema design agent will analyze columns more carefully.

Examples of acceptable imprecision:
- "所用时间" classified as identifier instead of numeric - OK, schema will handle it
- Store column classified as text instead of selection - OK, schema will handle it
- Open text columns classified as unknown - OK, schema will handle it
"""

CRITIC_CHAIN_OF_THOUGHT = """
## Workflow

1. Get the list of cleaned files using 'get_cleaned_files'
2. For each cleaned file, check ONLY these CRITICAL issues:
   - Are Brand, Model, Store columns preserved? (MUST PASS)
   - Are Rating columns preserved? (MUST PASS)
   - Are truly empty columns removed? (SHOULD PASS)
3. Do NOT fail on column type classification accuracy - that's handled by schema design

## CRITICAL: Response Format

Think carefully, then respond with ONE of these two formats:

- If cleaning passes validation, respond with exactly one word: 'valid'
- If cleaning has critical issues, respond with 'retry' followed by a concise bullet list of problems

Example good response:
valid

Example bad response:
retry
- CRITICAL: Column '品牌是?' was removed but contains brand entities

DO NOT add any text after 'valid'. The word must appear alone.
"""

CRITIC_INSTRUCTION = f"""
{CRITIC_ROLE_AND_GOAL}
{CRITIC_HINTS}
{CRITIC_CHAIN_OF_THOUGHT}
"""


# Tool lists
DATA_CLEANING_TOOLS = [
    get_approved_files,
    sample_file,
    analyze_file_quality,
    analyze_column_names,
    propose_column_renames,
    apply_column_renames,
    clean_file,
    get_cleaned_files,
    detect_column_types,
    detect_anomalies,
    convert_column_values,
    clean_urls,
    approve_data_cleaning,
]

CRITIC_TOOLS = [
    get_approved_files,
    sample_file,
    get_cleaned_files,
    detect_column_types,
    detect_anomalies,
    clean_urls,
]


def log_agent(callback_context: CallbackContext) -> None:
    """Log agent entry for debugging."""
    print(f"\n### Entering Agent: {callback_context.agent_name}")


class CheckDataCleaningStatus(BaseAgent):
    """
    Agent that checks data cleaning status and decides whether to continue or stop.

    Escalates (stops the loop) when EITHER:
    1. Data cleaning has been approved (DATA_CLEANING_COMPLETE_KEY exists), OR
    2. The critic feedback is 'valid'

    When feedback is valid, automatically sets DATA_CLEANING_COMPLETE_KEY so the
    pipeline can detect that cleaning is done.
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check feedback and yield escalation event if cleaning is complete."""
        # Check if cleaning was explicitly approved
        cleaning_complete = ctx.session.state.get(DATA_CLEANING_COMPLETE_KEY, False)

        # Check critic feedback - EXACT match like reference implementation
        feedback = ctx.session.state.get(DATA_CLEANING_FEEDBACK_KEY, "")
        feedback_str = str(feedback).strip().lower()
        feedback_valid = (feedback_str == "valid")  # Exact match, not startswith

        # Stop the loop if cleaning is approved OR feedback is valid
        should_stop = cleaning_complete or feedback_valid

        if should_stop:
            # Auto-approve cleaning if critic said valid
            if feedback_valid and not cleaning_complete:
                ctx.session.state[DATA_CLEANING_COMPLETE_KEY] = True
                print(f"\n### {self.name}: AUTO-APPROVING cleaning (critic said valid)")
            print(f"\n### {self.name}: Cleaning complete={cleaning_complete}, feedback_valid={feedback_valid}, escalating")
        else:
            print(f"\n### {self.name}: Cleaning not complete, feedback='{feedback[:50] if feedback else 'empty'}...', continuing")

        yield Event(
            author=self.name,
            actions=EventActions(escalate=should_stop)
        )


def create_data_cleaning_agent(
    llm=None,
    name: str = "data_cleaning_agent_v1"
) -> LlmAgent:
    """
    Create a Data Cleaning Agent.

    Uses qwen-flash model by default for faster processing.

    Args:
        llm: Optional LLM instance. If None, uses qwen-flash from get_adk_llm_flash()
        name: Agent name for identification

    Returns:
        LlmAgent: Configured Data Cleaning Agent
    """
    if llm is None:
        llm = get_adk_llm_flash()  # Use qwen-flash for data cleaning

    return LlmAgent(
        name=name,
        model=llm,
        description="Analyzes and cleans raw data files before schema proposal",
        instruction=CLEANING_INSTRUCTION,
        tools=DATA_CLEANING_TOOLS,
        before_agent_callback=log_agent,
    )


def create_data_cleaning_critic(
    llm=None,
    name: str = "data_cleaning_critic_v1"
) -> LlmAgent:
    """
    Create a Data Cleaning Critic Agent.

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        LlmAgent: Configured Data Cleaning Critic
    """
    if llm is None:
        llm = get_adk_llm()

    return LlmAgent(
        name=name,
        model=llm,
        description="Validates data cleaning decisions for quality and completeness",
        instruction=CRITIC_INSTRUCTION,
        tools=CRITIC_TOOLS,
        output_key=DATA_CLEANING_FEEDBACK_KEY,
        before_agent_callback=log_agent,
    )


def create_data_cleaning_loop(
    llm=None,
    max_iterations: int = 3,
    name: str = "data_cleaning_loop"
) -> LoopAgent:
    """
    Create a Data Cleaning Loop Agent.

    This loop agent coordinates the data cleaning and critic agents
    in an iterative refinement process.

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        max_iterations: Maximum refinement iterations
        name: Agent name for identification

    Returns:
        LoopAgent: Configured Data Cleaning Loop

    Example:
        ```python
        from src.agents import create_data_cleaning_loop, make_agent_caller

        loop = create_data_cleaning_loop()
        caller = await make_agent_caller(loop, {
            "data_cleaning_feedback": "",
            "approved_files": ["survey_data.xlsx"]
        })

        # First call - analyze and clean files
        await caller.call("Clean the data files")

        # User provides feedback
        await caller.call("Keep the '备注' column, it has important notes")

        # User approves
        await caller.call("Approve the cleaning")
        ```
    """
    cleaning_agent = create_data_cleaning_agent(llm)
    critic_agent = create_data_cleaning_critic(llm)
    stop_checker = CheckDataCleaningStatus(name="DataCleaningStopChecker")

    return LoopAgent(
        name=name,
        description="Cleans raw data files through iterative refinement with critic feedback",
        max_iterations=max_iterations,
        sub_agents=[cleaning_agent, critic_agent, stop_checker],
        before_agent_callback=log_agent,
    )

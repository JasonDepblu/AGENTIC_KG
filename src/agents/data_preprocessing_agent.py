"""
Data Preprocessing Agent for Agentic KG.

Agent responsible for analyzing data formats and transforming files as needed.
Supports multiple data formats including wide tables, survey data, and more.
"""

from google.adk.agents import Agent

from ..llm import get_adk_llm
from ..tools.file_suggestion import get_approved_files, sample_file
from ..tools.data_preprocessing import (
    # Basic analysis
    analyze_data_format,
    # Wide format transformation
    transform_wide_to_long,
    # State management
    get_transformed_files,
    approve_preprocessing,
    # Survey format analysis (new)
    analyze_survey_format,
    classify_columns,
    # Data normalization (new)
    normalize_values,
    split_multi_value_column,
    # Entity and data extraction (new)
    extract_entities,
    extract_ratings,
    extract_opinion_pairs,
    # High-level orchestration (new)
    parse_survey_responses,
)

# Agent instruction
AGENT_ROLE_AND_GOAL = """
You are a data preprocessing expert. Your job is to analyze approved data files
and transform them into formats suitable for knowledge graph construction.

You can handle multiple data formats:

1. **Wide Format** (Pivot Table Style):
   - Rows represent one entity type (e.g., attributes, metrics)
   - Column headers represent another entity type (e.g., brands, products)
   - Cell values represent relationships/scores

2. **Survey Format**:
   - Rows represent individual responses/respondents
   - Multiple column categories: demographics, ratings, opinions, entities
   - May contain multi-value columns with delimiters like "┋"
   - Special null values like "(跳过)", "(空)"
   - Rating columns (打多少分)
   - Opinion pairs (优秀点/劣势点)

3. **Standard Row Format**:
   - Each row is a complete record
   - No transformation needed
"""

AGENT_WORKFLOW = """
WORKFLOW - Follow these steps:

Step 1: Get the approved files
   - Use 'get_approved_files' to see which files need preprocessing

Step 2: For each approved file, analyze its format
   - Use 'sample_file' to view the file content
   - Use 'analyze_data_format' to detect basic format (wide/row)
   - If the file appears to be survey data (many columns, mixed types),
     use 'analyze_survey_format' for detailed analysis
   - Use 'classify_columns' to understand column semantics

Step 3: Based on detected format, apply appropriate transformations

   FOR WIDE FORMAT (column headers = entity names):
   - Use 'transform_wide_to_long' with parameters:
     * id_column: Column containing row identifiers
     * value_column_name: Name for numeric values
     * entity_column_name: Name for entities from column headers
     * output_prefix: Short prefix for output files
     * skip_columns: Columns to exclude (e.g., averages)

   FOR SURVEY FORMAT:
   - First normalize: Use 'normalize_values' to handle "(跳过)", "(空)"
   - Extract entities: Use 'extract_entities' for brand, model, store columns
   - Extract ratings: Use 'extract_ratings' for score columns
   - Split multi-values: Use 'split_multi_value_column' for "┋" delimited columns
   - Extract opinions: Use 'extract_opinion_pairs' for positive/negative feedback

   OR for complete survey processing:
   - Use 'parse_survey_responses' which orchestrates all the above

Step 4: Present transformation results and ask for approval
   - Use 'get_transformed_files' to show what was created
   - Explain what each output file contains
   - Ask user if they want to proceed

Step 5: When user approves, finalize preprocessing
   - Use 'approve_preprocessing' to update the approved files list
   - This will replace original files with transformed ones

IMPORTANT PATTERNS:

For Survey Data:
- Multi-value columns use "┋" as delimiter (not comma or semicolon)
- Special values "(跳过)" and "(空)" should be treated as null/empty
- Opinion pairs come in adjacent columns: "优秀点" then "劣势点"
- Rating columns contain "打多少分" in their names
- Entity columns contain patterns like "品牌", "车型", "门店"
- Demographic columns contain "年龄", "性别", "家庭"

For Wide Format:
- Chinese text in column headers = likely entities (e.g., "上汽集团 纯电动")
- First column often contains attribute names
- Columns like "关注度均值" (averages) should be skipped
"""

DATA_PREPROCESSING_INSTRUCTION = f"""
{AGENT_ROLE_AND_GOAL}
{AGENT_WORKFLOW}
"""

# Tools for the agent - including all new ETL tools
DATA_PREPROCESSING_TOOLS = [
    # File access
    get_approved_files,
    sample_file,
    # Format analysis
    analyze_data_format,
    analyze_survey_format,
    classify_columns,
    # Wide format transformation
    transform_wide_to_long,
    # Survey format tools
    normalize_values,
    split_multi_value_column,
    extract_entities,
    extract_ratings,
    extract_opinion_pairs,
    parse_survey_responses,
    # State management
    get_transformed_files,
    approve_preprocessing,
]


def create_data_preprocessing_agent(
    llm=None,
    name: str = "data_preprocessing_agent_v2"
) -> Agent:
    """
    Create a Data Preprocessing Agent.

    The data preprocessing agent analyzes approved files, detects their format,
    and transforms data into formats suitable for knowledge graph construction.

    Supports:
    - Wide format (pivot tables) -> long format
    - Survey data -> normalized entity/rating/opinion files
    - Multi-value columns -> split into separate rows
    - Special null value handling

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        Agent: Configured Data Preprocessing Agent

    Example:
        ```python
        from src.agents import create_data_preprocessing_agent, make_agent_caller

        agent = create_data_preprocessing_agent()
        caller = await make_agent_caller(agent, {
            "approved_files": ["survey_data.csv"]
        })

        # For survey data
        await caller.call("Analyze the survey data format")
        await caller.call("Parse the survey responses into normalized files")
        await caller.call("Approve the transformation")
        ```
    """
    if llm is None:
        llm = get_adk_llm()

    return Agent(
        name=name,
        model=llm,
        description="Analyzes data files and transforms various formats (wide tables, surveys, multi-value) for knowledge graph import.",
        instruction=DATA_PREPROCESSING_INSTRUCTION,
        tools=DATA_PREPROCESSING_TOOLS,
    )

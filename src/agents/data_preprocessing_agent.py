"""
Data Preprocessing Agent for Agentic KG.

Agent responsible for analyzing data formats and transforming files as needed.
"""

from google.adk.agents import Agent

from ..llm import get_adk_llm
from ..tools.file_suggestion import get_approved_files, sample_file
from ..tools.data_preprocessing import (
    analyze_data_format,
    transform_wide_to_long,
    get_transformed_files,
    approve_preprocessing,
)

# Agent instruction
AGENT_ROLE_AND_GOAL = """
You are a data preprocessing expert. Your job is to analyze approved data files
and transform them into formats suitable for knowledge graph construction.

Many data files come in "wide format" (pivot table style) where:
- Rows represent one entity type (e.g., attributes, metrics)
- Column headers represent another entity type (e.g., brands, products)
- Cell values represent relationships or scores between them

These need to be transformed into "long format" for proper knowledge graph import:
- Each entity type in its own file with unique identifiers
- Relationships in a separate file linking entities
"""

AGENT_WORKFLOW = """
WORKFLOW - Follow these steps:

Step 1: Get the approved files
   - Use 'get_approved_files' to see which files need preprocessing

Step 2: For each approved file, analyze its format
   - Use 'sample_file' to view the file content
   - Use 'analyze_data_format' to detect if it's wide or row format
   - Pay attention to:
     * Whether column headers contain entity names (Chinese text = likely entities)
     * Whether the first column contains categorical identifiers
     * Whether most other columns contain numeric values

Step 3: If wide format is detected, transform the file
   - Use 'transform_wide_to_long' with appropriate parameters:
     * id_column: The column containing row identifiers (usually first column or first non-numeric column)
     * value_column_name: Name for the numeric values (e.g., "attention_score", "rating")
     * entity_column_name: Name for entities extracted from column headers (e.g., "brand_powertrain")
     * output_prefix: A short prefix for output files based on the data domain
     * skip_columns: Any columns that should not be transformed (e.g., average/summary columns)

Step 4: Present transformation results and ask for approval
   - Use 'get_transformed_files' to show what was created
   - Explain what each output file contains
   - Ask user if they want to proceed

Step 5: When user approves, finalize preprocessing
   - Use 'approve_preprocessing' to update the approved files list
   - This will replace original files with transformed ones

IMPORTANT:
- For Chinese data like "上汽集团 纯电动", these are brand-powertrain combinations that should be extracted as entities
- The first column often contains attribute names that should become node identifiers
- Columns like "关注度均值" (average attention) might be metadata to skip or include as properties
"""

DATA_PREPROCESSING_INSTRUCTION = f"""
{AGENT_ROLE_AND_GOAL}
{AGENT_WORKFLOW}
"""

# Tools for the agent
DATA_PREPROCESSING_TOOLS = [
    get_approved_files,
    sample_file,
    analyze_data_format,
    transform_wide_to_long,
    get_transformed_files,
    approve_preprocessing,
]


def create_data_preprocessing_agent(
    llm=None,
    name: str = "data_preprocessing_agent_v1"
) -> Agent:
    """
    Create a Data Preprocessing Agent.

    The data preprocessing agent analyzes approved files, detects their format,
    and transforms wide-format data into long format suitable for knowledge
    graph construction.

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
            "approved_files": ["data_process.csv"]
        })

        await caller.call("Analyze and preprocess the approved files")
        await caller.call("Approve the transformation")
        ```
    """
    if llm is None:
        llm = get_adk_llm()

    return Agent(
        name=name,
        model=llm,
        description="Analyzes data files and transforms wide-format data for knowledge graph import.",
        instruction=DATA_PREPROCESSING_INSTRUCTION,
        tools=DATA_PREPROCESSING_TOOLS,
    )

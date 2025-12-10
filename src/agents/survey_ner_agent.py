"""
Survey NER (Named Entity Recognition) Agent.

Agent responsible for extracting named entities (Brand, Model, Store) from survey data
based on pre-classified columns.
"""

from google.adk.agents import Agent

from ..llm import get_adk_llm
from ..tools.file_suggestion import get_approved_files, sample_file
from ..tools.survey_classification import (
    get_column_classification,
    get_columns_by_category,
    get_entity_columns,
)
from ..tools.survey_entity_extraction import (
    extract_entities_from_column,
    extract_all_entities,
    create_respondent_table,
    save_entities_to_csv,
    get_extracted_entities,
    get_entity_map,
)

# Agent instruction
SURVEY_NER_INSTRUCTION = """
You are a Named Entity Recognition (NER) specialist for survey data.
Your task is to extract structured entity data from survey responses based on
pre-classified column information.

## Prerequisites

Before running this agent, the Survey Column Classifier Agent must have:
1. Classified all columns in the survey file
2. Identified entity_selection columns and their entity_type (Brand, Model, Store)

## Entity Types

1. **Brand** (品牌) - Automotive brands being surveyed
   - Examples: 乐道, 小鹏, 理想, 蔚来, 华为, 吉利

2. **Model** (车型) - Specific vehicle models
   - Examples: L90_顶配, P7顶配, ES6, 问界M9

3. **Store** (门店) - Dealership locations
   - Examples: 广州市番禺大道店, 深圳福田展厅

4. **Media** (媒体) - Media channels (if applicable)
   - Examples: 抖音, 微博, 汽车之家

## Workflow

1. Use 'get_column_classification' to retrieve classification results
2. Use 'get_entity_columns' to find which columns contain each entity type
3. Use 'extract_all_entities' to extract unique entities from entity_selection columns
4. Use 'create_respondent_table' to create respondent records with entity foreign keys
5. Use 'save_entities_to_csv' to save the output files

## Output Files

The agent produces:

1. **survey_parsed_<entity_type>_entities.csv** - One file per entity type (e.g., survey_parsed_brand_entities.csv)
   - Columns: entity_id, entity_type, value, source_column
   - Example: Brand_0, Brand, 乐道, 8、您本次调研的品牌是?

2. **survey_parsed_respondents.csv** - Respondent table with entity FKs
   - Columns: respondent_id, demographics..., brand_id, model_id, store_id
   - Links each respondent to the entities they selected

## Important Considerations

- Only extract from columns classified as "entity_selection" with appropriate entity_type
- Do NOT extract from rating or open_text columns (those contain feedback, not entity names)
- Handle null/empty values appropriately
- Preserve the original value text while creating normalized IDs

## Output Report

After extraction, report:
- Number of unique entities per type (Brand: X, Model: Y, Store: Z)
- Number of respondent records created
- Sample of the entity mapping (value -> ID)
- Any issues or warnings encountered
"""

# Tools for the agent
SURVEY_NER_TOOLS = [
    # File access
    get_approved_files,
    sample_file,
    # Classification access (read-only)
    get_column_classification,
    get_columns_by_category,
    get_entity_columns,
    # Entity extraction tools
    extract_entities_from_column,
    extract_all_entities,
    create_respondent_table,
    save_entities_to_csv,
    get_extracted_entities,
    get_entity_map,
]


def create_survey_ner_agent(
    llm=None,
    name: str = "survey_ner_agent"
) -> Agent:
    """
    Create a Survey NER Agent.

    This agent extracts named entities (Brand, Model, Store, Media) from survey data
    based on column classification results from the Survey Column Classifier Agent.

    Outputs:
    - Entity dimension tables (one per entity type)
    - Respondent table with entity foreign keys

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        Agent: Configured Survey NER Agent
    """
    if llm is None:
        llm = get_adk_llm()

    return Agent(
        name=name,
        model=llm,
        description="Extracts named entities (Brand, Model, Store) from survey data based on column classification.",
        instruction=SURVEY_NER_INSTRUCTION,
        tools=SURVEY_NER_TOOLS,
    )

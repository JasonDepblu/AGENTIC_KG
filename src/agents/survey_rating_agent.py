"""
Survey Rating Agent.

Agent responsible for extracting rating data from survey responses based on
pre-classified columns.
"""

from google.adk.agents import Agent

from ..llm import get_adk_llm
from ..tools.file_suggestion import get_approved_files, sample_file
from ..tools.survey_classification import (
    get_column_classification,
    get_columns_by_category,
)
from ..tools.survey_rating_extraction import (
    extract_aspects,
    extract_ratings,
    save_ratings_to_csv,
    get_extracted_aspects,
    get_ratings_by_aspect,
    get_ratings_by_respondent,
)

# Agent instruction
SURVEY_RATING_INSTRUCTION = """
You are a survey rating extraction specialist.
Your task is to extract and structure rating data from survey responses based on
pre-classified column information.

## Prerequisites

Before running this agent, the Survey Column Classifier Agent must have:
1. Classified all columns in the survey file
2. Identified rating columns and their related_to entity type

## Understanding Ratings

Rating columns typically contain:
- Numeric scores (1-10 scale)
- Scores about specific aspects (外观设计, 内饰质感, 服务体验)
- Each rating is related to an entity (Brand, Model, or Store)

## Rating Column Patterns

Common patterns in Chinese survey data:
- "针对该品牌"外观设计"方面的评分" → Aspect: 外观设计, Related to: Brand
- "该门店"服务体验"方面您会打多少分呢？" → Aspect: 服务体验, Related to: Store
- "针对该车型"空间舒适度"的评分" → Aspect: 空间舒适度, Related to: Model

## Workflow

1. Use 'get_column_classification' to retrieve classification results
2. Use 'get_columns_by_category' with category="rating" to find all rating columns
3. Use 'extract_aspects' to create the aspect dimension table
4. Use 'extract_ratings' to create the rating fact table
5. Use 'save_ratings_to_csv' to save the output files

## Output Files

The agent produces:

1. **survey_parsed_aspects.csv** - Aspect dimension table
   - Columns: aspect_id, aspect_name, related_entity, source_column
   - Example: ASP_0, 外观设计, Brand, 针对该品牌"外观设计"方面的评分

2. **survey_parsed_ratings.csv** - Rating fact table
   - Columns: respondent_id, aspect_id, score, raw_value
   - Links each respondent's rating to an aspect

## Score Normalization

The agent handles:
- Numeric values: 8, 9.5, 10
- Text values: "非常好" → 10, "好" → 8, "一般" → 5
- Empty/null values: skipped

## Output Report

After extraction, report:
- Number of aspects extracted
- Number of rating records
- Statistics per aspect (mean, min, max, count)
- Any parsing issues encountered
"""

# Tools for the agent
SURVEY_RATING_TOOLS = [
    # File access
    get_approved_files,
    sample_file,
    # Classification access (read-only)
    get_column_classification,
    get_columns_by_category,
    # Rating extraction tools
    extract_aspects,
    extract_ratings,
    save_ratings_to_csv,
    get_extracted_aspects,
    get_ratings_by_aspect,
    get_ratings_by_respondent,
]


def create_survey_rating_agent(
    llm=None,
    name: str = "survey_rating_agent"
) -> Agent:
    """
    Create a Survey Rating Agent.

    This agent extracts rating data from survey responses based on column
    classification results from the Survey Column Classifier Agent.

    Outputs:
    - Aspect dimension table (aspect_id, aspect_name, related_entity)
    - Rating fact table (respondent_id, aspect_id, score)

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        Agent: Configured Survey Rating Agent
    """
    if llm is None:
        llm = get_adk_llm()

    return Agent(
        name=name,
        model=llm,
        description="Extracts rating data from survey responses based on column classification.",
        instruction=SURVEY_RATING_INSTRUCTION,
        tools=SURVEY_RATING_TOOLS,
    )

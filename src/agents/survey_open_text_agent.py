"""
Survey Open Text Agent.

Agent responsible for analyzing open-text responses and extracting insights,
features, issues, and sentiment from survey data using LLM.
"""

from google.adk.agents import Agent

from ..llm import get_adk_llm
from ..tools.file_suggestion import get_approved_files, sample_file
from ..tools.survey_classification import (
    get_column_classification,
    get_columns_by_category,
)
from ..tools.survey_text_analysis import (
    analyze_open_text,
    convert_rating_to_sentiment,
    batch_analyze_open_text,
    aggregate_entity_sentiment,
    save_text_analysis_to_csv,
    get_extracted_features,
    get_extracted_issues,
    get_text_insights,
)

# Agent instruction
SURVEY_OPEN_TEXT_INSTRUCTION = """
You are a survey text analysis and sentiment specialist.
Your task is to analyze open-text responses and extract structured insights using LLM.

## Prerequisites

Before running this agent, ensure:
1. Column classification has been completed (Survey Column Classifier Agent)
2. Entity extraction has been completed (Survey NER Agent)
3. Rating extraction has been completed (Survey Rating Agent)

## Open Text Analysis

Open-text columns contain free-form responses with valuable insights:
- Product features and innovations (场景创新、亮点设计)
- Cost control insights (成本控制方面的启发项)
- Suggestions and recommendations
- Problems and complaints

## LLM-Based Extraction

For each open-text response, the LLM extracts:

1. **Features** (特性)
   - Specific product features mentioned
   - Examples: 后排娱乐屏, 沉坑设计, 机械门把手

2. **Issues** (问题)
   - Problems, complaints, or areas for improvement
   - Examples: 塑料感强, 续航不足, 做工粗糙

3. **Insights** (见解)
   - Valuable observations or suggestions
   - Examples: 灵活的角度调节, 模块化设计降本

4. **Sentiment** (情感)
   - Overall sentiment: positive, neutral, negative
   - Sentiment score: -1.0 to 1.0

5. **Keywords** (关键词)
   - Key terms for search and categorization

## Sentiment Aggregation

The agent also aggregates sentiment from multiple sources:
- Rating scores → sentiment (7-10: positive, 5-6: neutral, 1-4: negative)
- Open text responses → LLM-extracted sentiment
- Per-entity sentiment aggregation

## Workflow

1. Use 'get_column_classification' to find open_text columns
2. Use 'batch_analyze_open_text' to process all open-text responses
3. Use 'aggregate_entity_sentiment' for entity-level sentiment (optional)
4. Use 'save_text_analysis_to_csv' to save the output files

## Output Files

The agent produces:

1. **survey_parsed_features.csv** - Feature dimension table
   - Columns: feature_id, feature_name, mentions, sentiment_association

2. **survey_parsed_issues.csv** - Issue dimension table
   - Columns: issue_id, issue_description, mentions, severity

3. **survey_parsed_insights.csv** - Insights fact table
   - Columns: respondent_id, source_column, feature_ids, issue_ids, sentiment,
     sentiment_score, keywords, raw_text

## Processing Considerations

- Open text analysis uses LLM calls which can be slow
- Use 'max_rows' parameter to limit processing during testing
- Empty/null responses ("无", "暂无", "-") are skipped
- Context (brand, model, store) is provided to LLM for better analysis

## Output Report

After analysis, report:
- Number of features extracted (with top 5 by mentions)
- Number of issues extracted (with top 5 by mentions)
- Number of insight records created
- Sentiment distribution (positive/neutral/negative counts)
- Any errors or warnings during LLM processing
"""

# Tools for the agent
SURVEY_OPEN_TEXT_TOOLS = [
    # File access
    get_approved_files,
    sample_file,
    # Classification access (read-only)
    get_column_classification,
    get_columns_by_category,
    # Text analysis tools
    analyze_open_text,
    convert_rating_to_sentiment,
    batch_analyze_open_text,
    aggregate_entity_sentiment,
    save_text_analysis_to_csv,
    get_extracted_features,
    get_extracted_issues,
    get_text_insights,
]


def create_survey_open_text_agent(
    llm=None,
    name: str = "survey_open_text_agent"
) -> Agent:
    """
    Create a Survey Open Text Agent.

    This agent analyzes open-text responses using LLM to extract:
    - Features (product features mentioned)
    - Issues (problems or complaints)
    - Insights (observations and suggestions)
    - Sentiment (positive/neutral/negative)

    Also aggregates sentiment from ratings and text for entity-level analysis.

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        Agent: Configured Survey Open Text Agent
    """
    if llm is None:
        llm = get_adk_llm()

    return Agent(
        name=name,
        model=llm,
        description="Analyzes open-text responses and extracts insights, features, issues, and sentiment using LLM.",
        instruction=SURVEY_OPEN_TEXT_INSTRUCTION,
        tools=SURVEY_OPEN_TEXT_TOOLS,
    )

"""
Survey Column Classifier Agent.

Agent responsible for classifying survey questionnaire columns into semantic categories
using LLM-based analysis.
"""

from google.adk.agents import Agent

from ..llm import get_adk_llm
from ..tools.file_suggestion import get_approved_files, sample_file
from ..tools.survey_classification import (
    classify_survey_column,
    classify_all_columns,
    get_column_classification,
    get_columns_by_category,
    get_entity_columns,
    save_classification_results,
)

# Agent instruction
COLUMN_CLASSIFIER_INSTRUCTION = """
You are an expert at analyzing survey questionnaire columns.
Your task is to classify each column into one of the following semantic categories:

## Column Categories

1. **entity_selection** - Column where respondent selects a specific entity
   - Examples: "您本次调研的品牌是?", "您本次到访的门店是", "您本次调研的车型及配置是?"
   - These contain the actual selected values of Brand, Model, Store, or Media entities
   - The column header asks "which X did you select/visit/research?"

2. **rating** - Column where respondent gives a numeric score
   - Examples: "该门店服务体验方面您会打多少分", "针对该品牌外观设计方面的评分"
   - Contains numeric ratings (1-10 scale typically)
   - Usually column name contains "打多少分", "评分", "评价"

3. **open_text** - Column with free-form text responses
   - Examples: "该车型是否有成本控制方面的启发项?", "有什么改进建议?"
   - Contains opinions, suggestions, detailed feedback
   - May contain innovation insights, cost-saving suggestions, etc.

4. **demographic** - Respondent characteristics
   - Examples: "您的年龄是?", "您的性别是?", "您的家庭情况是?", "您是否拥有车辆?"
   - Personal information about the respondent

5. **id** - Unique identifier for the response
   - Examples: "序号", "ID", "响应编号"

## Additional Classification Info

For **entity_selection** columns, also identify:
- entity_type: Brand(品牌), Model(车型), Store(门店), or Media(媒体)

For **rating** and **open_text** columns, also identify:
- related_to: Which entity this rating/opinion is about (Brand/Model/Store/null)

## Workflow

1. Use 'get_approved_files' to see available files
2. Use 'sample_file' to preview the file content
3. Use 'classify_all_columns' to classify all columns in the file using LLM
4. Review the classification summary
5. Use 'get_columns_by_category' or 'get_entity_columns' to explore specific categories
6. Use 'save_classification_results' to save the results for the next phase

## Important Notes

- The classification uses LLM to analyze column names AND sample values
- This is critical for distinguishing between:
  - Entity selection columns (actual entity values)
  - Rating/opinion columns (feedback ABOUT entities)
- Classification results are stored in state and used by subsequent agents

## Output

After classification, report:
- Total columns classified
- Count per category (entity_selection, rating, open_text, demographic, id)
- For entity columns, list which columns contain Brand, Model, Store
- Any columns classified as 'unknown' that need attention
"""

# Tools for the agent
COLUMN_CLASSIFIER_TOOLS = [
    # File access
    get_approved_files,
    sample_file,
    # Classification tools
    classify_survey_column,
    classify_all_columns,
    get_column_classification,
    get_columns_by_category,
    get_entity_columns,
    save_classification_results,
]


def create_survey_column_classifier_agent(
    llm=None,
    name: str = "survey_column_classifier_agent"
) -> Agent:
    """
    Create a Survey Column Classifier Agent.

    This agent uses LLM to classify survey columns into semantic categories:
    - entity_selection: Where respondent selects entities (Brand, Model, Store)
    - rating: Numeric scores/ratings
    - open_text: Free-form text responses
    - demographic: Respondent information
    - id: Unique identifiers

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        Agent: Configured Survey Column Classifier Agent
    """
    if llm is None:
        llm = get_adk_llm()

    return Agent(
        name=name,
        model=llm,
        description="Classifies survey questionnaire columns into semantic categories using LLM-based analysis.",
        instruction=COLUMN_CLASSIFIER_INSTRUCTION,
        tools=COLUMN_CLASSIFIER_TOOLS,
    )

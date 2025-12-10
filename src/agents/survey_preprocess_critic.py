"""
Survey Preprocess Critic Agent.

Agent responsible for validating the output of survey preprocessing stages.
Works as part of a LoopAgent to enable iterative refinement.
"""

from google.adk.agents import Agent

from ..llm import get_adk_llm
from ..tools.file_suggestion import get_approved_files, sample_file
from ..tools.survey_classification import (
    get_column_classification,
    get_columns_by_category,
)
from ..tools.survey_validation import (
    validate_column_classification,
    validate_entity_values,
    validate_extracted_entities,
    validate_all_entities,
    validate_extracted_aspects,
    validate_rating_scores,
    get_preprocess_feedback,
    set_preprocess_feedback,
)

# Agent instruction
SURVEY_PREPROCESS_CRITIC_INSTRUCTION = """
You are a survey data preprocessing validation specialist.
Your task is to critically evaluate the output of each preprocessing phase.

## Your Role

You review the work of preprocessing agents and identify data quality issues:
- Column classification errors (wrong category, wrong entity type)
- Entity extraction errors (wrong source column, invalid values)
- Rating extraction errors (wrong aspect names, invalid scores)

## Validation Rules by Phase

### Phase 1: Column Classification Validation

Check that columns are correctly classified:

1. **entity_selection columns**
   - "品牌是?" questions → should be Brand entity_selection
   - "车型及配置是?" questions → should be Model entity_selection
   - "门店是" questions → should be Store entity_selection
   - Values should be short names, NOT long descriptions

2. **rating columns**
   - "打多少分" questions → should be rating
   - Values should be numeric (1-10 scale)

3. **open_text columns**
   - "启发项?", "体验是?" questions → should be open_text
   - Values should be longer text responses

4. **Common Misclassification Errors**
   - Q90 (场景创新启发项) misclassified as entity_selection
   - Q92 (成本控制启发项) misclassified as entity_selection
   - Rating columns misclassified as entity_selection

### Phase 2: Entity Extraction Validation

Check that entities are extracted correctly:

1. **Brand entities**
   - Should be brand names (e.g., "乐道", "小鹏", "理想")
   - Should NOT be long descriptions from Q90
   - Source column should contain "品牌是?" NOT "启发项"

2. **Model entities**
   - Should be model names (e.g., "L90_顶配", "P7顶配")
   - Should NOT be descriptions from Q92
   - Source column should contain "车型及配置是?"

3. **Store entities**
   - Should be store names/addresses
   - Should NOT be numeric scores
   - Source column should contain "门店是"

### Phase 3: Rating Extraction Validation

Check that ratings are extracted correctly:

1. **Aspect names**
   - Should be semantic names (e.g., "外观设计", "内饰质感")
   - Should NOT be question codes (e.g., "Q11")
   - Should be extracted from quoted text in column name

2. **Scores**
   - Should be within 1-10 range
   - Should be numeric values

## Workflow

1. Use validation tools to check the current phase's output
2. Sample the actual data files to verify issues
3. Provide specific feedback about what needs to be fixed

## Feedback Format

After validation, set feedback to ONE of:
- "valid" - Output is correct, proceed to next phase
- "retry" - Issues found, current phase needs to re-run with fixes

When setting feedback, provide clear reasoning about:
- What specific issues were found
- What columns/values are problematic
- How the agent should fix the issues

## Example Feedback

### Example 1: Brand extraction from wrong column
```
Feedback: retry
Reason: Brand entities are being extracted from column "90、该车型/品牌是否有优秀的场景创新..."
which contains open-text descriptions, NOT brand names.
The correct source column should be "8、您本次调研的品牌是?" which contains
actual brand names like "乐道", "小鹏".
Fix: Re-classify column Q90 as open_text, and extract brands from Q8.
```

### Example 2: Store values are numbers
```
Feedback: retry
Reason: Store entities contain numeric values (8, 5, 9) which are rating scores,
not store names. Store should be extracted from "10、您本次到访的门店是"
which contains actual store addresses.
Fix: Re-classify the source column and extract from the correct column.
```

### Example 3: Valid extraction
```
Feedback: valid
Reason: All entity types (Brand, Model, Store) are correctly extracted.
- Brands are short names from Q8 column
- Models are name+config formats from Q9 column
- Stores are addresses/names from Q10 column
Proceed to next phase.
```
"""

# Tools for the critic agent
SURVEY_PREPROCESS_CRITIC_TOOLS = [
    # File access for sampling
    get_approved_files,
    sample_file,
    # Classification access (read-only)
    get_column_classification,
    get_columns_by_category,
    # Validation tools
    validate_column_classification,
    validate_entity_values,
    validate_extracted_entities,
    validate_all_entities,
    validate_extracted_aspects,
    validate_rating_scores,
    # Feedback management
    get_preprocess_feedback,
    set_preprocess_feedback,
]


def create_survey_preprocess_critic_agent(
    llm=None,
    name: str = "survey_preprocess_critic_agent"
) -> Agent:
    """
    Create a Survey Preprocess Critic Agent.

    This agent validates the output of each preprocessing phase and provides
    feedback for iterative refinement using a LoopAgent pattern.

    Validation includes:
    - Column classification correctness
    - Entity extraction source column validation
    - Entity value quality (not descriptions, not numbers)
    - Aspect name semantic validation
    - Score range validation

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification

    Returns:
        Agent: Configured Survey Preprocess Critic Agent
    """
    if llm is None:
        llm = get_adk_llm()

    return Agent(
        name=name,
        model=llm,
        description="Validates survey preprocessing output and provides feedback for iterative refinement.",
        instruction=SURVEY_PREPROCESS_CRITIC_INSTRUCTION,
        tools=SURVEY_PREPROCESS_CRITIC_TOOLS,
        output_key="preprocess_feedback",  # Store feedback in state
    )

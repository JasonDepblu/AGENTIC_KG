"""
Survey Preprocessing Coordinator Agent.

Coordinates the complete survey data preprocessing pipeline using a SequentialAgent
to orchestrate the individual survey processing agents.
"""

from google.adk.agents import SequentialAgent

from ..llm import get_adk_llm
from .survey_column_classifier_agent import create_survey_column_classifier_agent
from .survey_ner_agent import create_survey_ner_agent
from .survey_rating_agent import create_survey_rating_agent
from .survey_open_text_agent import create_survey_open_text_agent


def create_survey_preprocessing_coordinator(
    llm=None,
    name: str = "survey_preprocessing_coordinator",
    include_open_text: bool = True
) -> SequentialAgent:
    """
    Create a Survey Preprocessing Coordinator.

    This coordinator uses a SequentialAgent to orchestrate the complete survey
    data preprocessing pipeline in the following order:

    1. **Survey Column Classifier Agent** (Phase 1)
       - Uses LLM to classify each column semantically
       - Identifies entity_selection, rating, open_text, demographic, id columns
       - Output: column_classification.json

    2. **Survey NER Agent** (Phase 2)
       - Extracts Brand, Model, Store entities from entity_selection columns
       - Creates respondent table with entity foreign keys
       - Output: survey_parsed_*_entities.csv, survey_parsed_respondents.csv

    3. **Survey Rating Agent** (Phase 3)
       - Extracts aspect dimension table from rating column headers
       - Creates rating fact table
       - Output: survey_parsed_aspects.csv, survey_parsed_ratings.csv

    4. **Survey Open Text Agent** (Phase 4, Optional)
       - Uses LLM to analyze open-text responses
       - Extracts features, issues, insights, sentiment
       - Output: survey_parsed_features.csv, survey_parsed_issues.csv,
         survey_parsed_insights.csv

    ## Data Flow

    ```
    Raw Survey File
         │
         ▼
    ┌────────────────────────────┐
    │ Column Classifier Agent    │  → column_classification.json
    │ (LLM semantic analysis)    │
    └────────────────────────────┘
         │
         ▼
    ┌────────────────────────────┐
    │ NER Agent                  │  → *_entities.csv, respondents.csv
    │ (Entity extraction)        │
    └────────────────────────────┘
         │
         ▼
    ┌────────────────────────────┐
    │ Rating Agent               │  → aspects.csv, ratings.csv
    │ (Rating extraction)        │
    └────────────────────────────┘
         │
         ▼
    ┌────────────────────────────┐
    │ Open Text Agent            │  → features.csv, issues.csv, insights.csv
    │ (LLM text analysis)        │  [Optional]
    └────────────────────────────┘
    ```

    Args:
        llm: Optional LLM instance. If None, uses default from get_adk_llm()
        name: Agent name for identification
        include_open_text: Whether to include the Open Text Agent (Phase 4).
                          Set to False to skip LLM-heavy text analysis.

    Returns:
        SequentialAgent: Configured Survey Preprocessing Coordinator

    Example:
        ```python
        from src.agents import create_survey_preprocessing_coordinator, make_agent_caller

        # Full pipeline
        coordinator = create_survey_preprocessing_coordinator()
        caller = await make_agent_caller(coordinator, {
            "approved_files": ["survey_data.xlsx"]
        })

        await caller.call("Process the survey data through all phases")

        # Without open text analysis (faster)
        coordinator = create_survey_preprocessing_coordinator(include_open_text=False)
        ```
    """
    if llm is None:
        llm = get_adk_llm()

    # Create sub-agents
    column_classifier = create_survey_column_classifier_agent(llm=llm)
    ner_agent = create_survey_ner_agent(llm=llm)
    rating_agent = create_survey_rating_agent(llm=llm)

    # Build sub-agent list
    sub_agents = [
        column_classifier,
        ner_agent,
        rating_agent,
    ]

    if include_open_text:
        open_text_agent = create_survey_open_text_agent(llm=llm)
        sub_agents.append(open_text_agent)

    return SequentialAgent(
        name=name,
        description="Coordinates the complete survey data preprocessing pipeline: column classification, entity extraction, rating extraction, and optional text analysis.",
        sub_agents=sub_agents,
    )


# Convenience function for creating individual agents
def create_survey_agents(llm=None, include_open_text: bool = True):
    """
    Create all survey processing agents.

    Args:
        llm: Optional LLM instance
        include_open_text: Whether to include the Open Text Agent

    Returns:
        Dictionary of agent instances
    """
    if llm is None:
        llm = get_adk_llm()

    agents = {
        "column_classifier": create_survey_column_classifier_agent(llm=llm),
        "ner": create_survey_ner_agent(llm=llm),
        "rating": create_survey_rating_agent(llm=llm),
    }

    if include_open_text:
        agents["open_text"] = create_survey_open_text_agent(llm=llm)

    agents["coordinator"] = create_survey_preprocessing_coordinator(
        llm=llm,
        include_open_text=include_open_text
    )

    return agents

"""
Agents module for Agentic KG.

Contains agent definitions for the multi-agent knowledge graph construction system.
"""

from .base import AgentCaller, make_agent_caller, run_agent_conversation
from .user_intent_agent import create_user_intent_agent
from .file_suggestion_agent import create_file_suggestion_agent
from .data_preprocessing_agent import create_data_preprocessing_agent
from .schema_proposal_agent import (
    create_schema_proposal_agent,
    create_schema_critic_agent,
    create_schema_refinement_loop,
    CheckStatusAndEscalate,
)
from .kg_builder_agent import create_kg_builder_agent
from .ner_agent import create_ner_agent
from .fact_type_agent import create_fact_type_agent
from .unstructured_data_agent import create_unstructured_data_agent
from .kg_query_agent import create_kg_query_agent
from .cypher_generator_agent import create_cypher_generator_agent
from .cypher_validator_agent import create_cypher_validator_agent
from .cypher_loop_agent import (
    create_cypher_generation_loop,
    CheckCypherValidationStatus,
    generate_validated_cypher,
)
from .survey_column_classifier_agent import create_survey_column_classifier_agent
from .survey_ner_agent import create_survey_ner_agent
from .survey_rating_agent import create_survey_rating_agent
from .survey_open_text_agent import create_survey_open_text_agent
from .survey_preprocessing_coordinator import (
    create_survey_preprocessing_coordinator,
    create_survey_agents,
)
from .survey_preprocess_critic import create_survey_preprocess_critic_agent
from .preprocess_refinement_loop import (
    CheckPreprocessStatus,
    CheckPipelineStatus,
    create_column_classification_loop,
    create_ner_loop,
    create_rating_loop,
    create_preprocess_refinement_coordinator,
    create_complete_pipeline_coordinator,
)
from .schema_design_agent import (
    create_schema_design_agent,
    create_schema_design_critic_agent,
    create_schema_design_loop,
    CheckSchemaDesignStatus,
)
from .targeted_preprocessing_agent import (
    create_targeted_preprocessing_agent,
    create_preprocessing_critic_agent,
    create_targeted_preprocessing_loop,
    CheckPreprocessingStatus as CheckTargetedPreprocessingStatus,
)
from .schema_preprocess_coordinator import (
    create_schema_preprocess_coordinator,
    create_schema_preprocess_sequential,
    CheckCoordinatorStatus,
    CheckSchemaApproved,
)
from .data_cleaning_agent import (
    create_data_cleaning_agent,
    create_data_cleaning_critic,
    create_data_cleaning_loop,
    CheckDataCleaningStatus,
)
from .structured_data_coordinator import (
    create_structured_data_coordinator,
    create_schema_only_coordinator,
    create_data_cleaning_phase,
    create_schema_and_preprocessing_loop,
)
from .entity_type_agent import (
    create_entity_type_generator_agent,
    create_entity_type_critic_agent,
    create_entity_type_detection_loop,
    EntityTypeStopChecker,
    ENTITY_TYPE_DETECTION_COMPLETE,
    DETECTED_DOMAIN,
    PROPOSED_ENTITY_TYPES,
)

__all__ = [
    # Base
    "AgentCaller",
    "make_agent_caller",
    "run_agent_conversation",
    # User Intent
    "create_user_intent_agent",
    # File Suggestion
    "create_file_suggestion_agent",
    # Data Preprocessing
    "create_data_preprocessing_agent",
    # Schema Proposal
    "create_schema_proposal_agent",
    "create_schema_critic_agent",
    "create_schema_refinement_loop",
    "CheckStatusAndEscalate",
    # KG Builder
    "create_kg_builder_agent",
    # Unstructured Data Agents
    "create_ner_agent",
    "create_fact_type_agent",
    "create_unstructured_data_agent",
    # KG Query Agent
    "create_kg_query_agent",
    # Cypher Generation Loop
    "create_cypher_generator_agent",
    "create_cypher_validator_agent",
    "create_cypher_generation_loop",
    "CheckCypherValidationStatus",
    "generate_validated_cypher",
    # Survey Preprocessing Agents
    "create_survey_column_classifier_agent",
    "create_survey_ner_agent",
    "create_survey_rating_agent",
    "create_survey_open_text_agent",
    "create_survey_preprocessing_coordinator",
    "create_survey_agents",
    # Preprocess Critic and Refinement Loop
    "create_survey_preprocess_critic_agent",
    "CheckPreprocessStatus",
    "CheckPipelineStatus",
    "create_column_classification_loop",
    "create_ner_loop",
    "create_rating_loop",
    "create_preprocess_refinement_coordinator",
    "create_complete_pipeline_coordinator",
    # Schema Design Agent
    "create_schema_design_agent",
    "create_schema_design_critic_agent",
    "create_schema_design_loop",
    "CheckSchemaDesignStatus",
    # Targeted Preprocessing Agent
    "create_targeted_preprocessing_agent",
    "create_preprocessing_critic_agent",
    "create_targeted_preprocessing_loop",
    "CheckTargetedPreprocessingStatus",
    # Schema-Preprocess Coordinator (Super Coordinator)
    "create_schema_preprocess_coordinator",
    "create_schema_preprocess_sequential",
    "CheckCoordinatorStatus",
    "CheckSchemaApproved",
    # Data Cleaning Agent
    "create_data_cleaning_agent",
    "create_data_cleaning_critic",
    "create_data_cleaning_loop",
    "CheckDataCleaningStatus",
    # Structured Data Coordinator
    "create_structured_data_coordinator",
    "create_schema_only_coordinator",
    "create_data_cleaning_phase",
    "create_schema_and_preprocessing_loop",
    # Entity Type Detection Agent (LLM-driven)
    "create_entity_type_generator_agent",
    "create_entity_type_critic_agent",
    "create_entity_type_detection_loop",
    "EntityTypeStopChecker",
    "ENTITY_TYPE_DETECTION_COMPLETE",
    "DETECTED_DOMAIN",
    "PROPOSED_ENTITY_TYPES",
]

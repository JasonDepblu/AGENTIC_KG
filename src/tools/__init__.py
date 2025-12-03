"""
Tools module for Agentic KG.

Contains tool functions used by agents for various tasks.
"""

from .common import tool_success, tool_error, get_state_value, set_state_value
from .user_intent import (
    set_perceived_user_goal,
    approve_perceived_user_goal,
    get_approved_user_goal,
    PERCEIVED_USER_GOAL,
    APPROVED_USER_GOAL,
)
from .file_suggestion import (
    list_available_files,
    sample_file,
    search_file,
    set_suggested_files,
    get_suggested_files,
    approve_suggested_files,
    get_approved_files,
    ALL_AVAILABLE_FILES,
    SUGGESTED_FILES,
    APPROVED_FILES,
)
from .kg_construction import (
    create_uniqueness_constraint,
    load_nodes_from_csv,
    import_nodes,
    import_relationships,
    construct_domain_graph,
    propose_node_construction,
    propose_relationship_construction,
    remove_node_construction,
    remove_relationship_construction,
    get_proposed_construction_plan,
    approve_proposed_construction_plan,
    PROPOSED_CONSTRUCTION_PLAN,
    APPROVED_CONSTRUCTION_PLAN,
)
from .data_preprocessing import (
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
    # State keys
    DATA_FORMAT_ANALYSIS,
    TRANSFORMED_FILES,
    SURVEY_FORMAT_ANALYSIS,
    COLUMN_CLASSIFICATION,
    EXTRACTED_ENTITIES,
    EXTRACTED_RATINGS,
    EXTRACTED_OPINIONS,
    SPLIT_MULTI_VALUES,
    NORMALIZED_FILES,
)
from .unstructured_extraction import (
    # NER tools
    get_well_known_types,
    set_proposed_entities,
    get_proposed_entities,
    approve_proposed_entities,
    get_approved_entities,
    # Fact Type tools
    add_proposed_fact,
    get_proposed_facts,
    remove_proposed_fact,
    approve_proposed_facts,
    get_approved_facts,
    # State keys
    PROPOSED_ENTITIES,
    APPROVED_ENTITIES,
    PROPOSED_FACTS,
    APPROVED_FACTS,
)
from .kg_extraction import (
    # Schema tools
    build_entity_schema,
    get_entity_schema,
    # Extraction tools (simplified mode)
    extract_entities_from_text,
    process_unstructured_file,
    get_extraction_results,
    # Extraction tools (GraphRAG mode)
    process_unstructured_file_graphrag,
    process_unstructured_file_auto,
    is_graphrag_available,
    # GraphRAG components
    RegexTextSplitter,
    MarkdownDataLoader,
    make_kg_builder,
    get_graphrag_llm,
    get_graphrag_embedder,
    DashScopeEmbedderAdapter,
    create_contextualized_prompt,
    # Entity resolution tools (simplified mode)
    correlate_entities,
    get_correlation_results,
    create_correspondence_relationships,
    # Entity resolution tools (GraphRAG mode)
    correlate_entities_graphrag,
    auto_correlate_all_entities,
    # State keys
    ENTITY_SCHEMA,
    EXTRACTION_RESULTS,
    CORRELATION_RESULTS,
)

__all__ = [
    # Common
    "tool_success",
    "tool_error",
    "get_state_value",
    "set_state_value",
    # User Intent
    "set_perceived_user_goal",
    "approve_perceived_user_goal",
    "get_approved_user_goal",
    "PERCEIVED_USER_GOAL",
    "APPROVED_USER_GOAL",
    # File Suggestion
    "list_available_files",
    "sample_file",
    "search_file",
    "set_suggested_files",
    "get_suggested_files",
    "approve_suggested_files",
    "get_approved_files",
    "ALL_AVAILABLE_FILES",
    "SUGGESTED_FILES",
    "APPROVED_FILES",
    # KG Construction
    "create_uniqueness_constraint",
    "load_nodes_from_csv",
    "import_nodes",
    "import_relationships",
    "construct_domain_graph",
    "propose_node_construction",
    "propose_relationship_construction",
    "remove_node_construction",
    "remove_relationship_construction",
    "get_proposed_construction_plan",
    "approve_proposed_construction_plan",
    "PROPOSED_CONSTRUCTION_PLAN",
    "APPROVED_CONSTRUCTION_PLAN",
    # Data Preprocessing
    "analyze_data_format",
    "analyze_survey_format",
    "classify_columns",
    "transform_wide_to_long",
    "normalize_values",
    "split_multi_value_column",
    "extract_entities",
    "extract_ratings",
    "extract_opinion_pairs",
    "parse_survey_responses",
    "get_transformed_files",
    "approve_preprocessing",
    "DATA_FORMAT_ANALYSIS",
    "TRANSFORMED_FILES",
    "SURVEY_FORMAT_ANALYSIS",
    "COLUMN_CLASSIFICATION",
    "EXTRACTED_ENTITIES",
    "EXTRACTED_RATINGS",
    "EXTRACTED_OPINIONS",
    "SPLIT_MULTI_VALUES",
    "NORMALIZED_FILES",
    # Unstructured Extraction - NER
    "get_well_known_types",
    "set_proposed_entities",
    "get_proposed_entities",
    "approve_proposed_entities",
    "get_approved_entities",
    "PROPOSED_ENTITIES",
    "APPROVED_ENTITIES",
    # Unstructured Extraction - Fact Types
    "add_proposed_fact",
    "get_proposed_facts",
    "remove_proposed_fact",
    "approve_proposed_facts",
    "get_approved_facts",
    "PROPOSED_FACTS",
    "APPROVED_FACTS",
    # KG Extraction - Schema
    "build_entity_schema",
    "get_entity_schema",
    # KG Extraction - Simplified mode
    "extract_entities_from_text",
    "process_unstructured_file",
    "get_extraction_results",
    # KG Extraction - GraphRAG mode
    "process_unstructured_file_graphrag",
    "process_unstructured_file_auto",
    "is_graphrag_available",
    # KG Extraction - GraphRAG components
    "RegexTextSplitter",
    "MarkdownDataLoader",
    "make_kg_builder",
    "get_graphrag_llm",
    "get_graphrag_embedder",
    "create_contextualized_prompt",
    # KG Extraction - Entity resolution (simplified)
    "correlate_entities",
    "get_correlation_results",
    "create_correspondence_relationships",
    # KG Extraction - Entity resolution (GraphRAG)
    "correlate_entities_graphrag",
    "auto_correlate_all_entities",
    # KG Extraction - State keys
    "ENTITY_SCHEMA",
    "EXTRACTION_RESULTS",
    "CORRELATION_RESULTS",
]

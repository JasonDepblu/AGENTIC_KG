"""
Survey entity extraction tools.

This module provides tools for extracting named entities (Brand, Model, Store)
from survey data based on pre-classified columns.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from google.adk.tools import ToolContext

from src.config import get_config
from src.tools.common import tool_success, tool_error, set_state_value, get_state_value
from src.tools.survey_classification import COLUMN_CLASSIFICATION, CLASSIFICATION_SUMMARY

logger = logging.getLogger(__name__)

# State keys
EXTRACTED_ENTITIES = "extracted_entities"
ENTITY_MAPS = "entity_maps"
RESPONDENT_DATA = "respondent_data"


def extract_entities_from_column(
    df: pd.DataFrame,
    column_name: str,
    entity_type: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Extracts unique entities from a specific column.

    Args:
        df: DataFrame containing the survey data
        column_name: Column to extract entities from
        entity_type: "Brand", "Model", or "Store"
        tool_context: ADK ToolContext

    Returns:
        Dictionary with entities list and value_to_id_map
    """
    if column_name not in df.columns:
        return tool_error(f"Column '{column_name}' not found in DataFrame")

    valid_types = ["Brand", "Model", "Store", "Media"]
    if entity_type not in valid_types:
        return tool_error(f"Invalid entity_type. Must be one of: {valid_types}")

    # Get unique non-null values
    unique_values = df[column_name].dropna().unique()

    # Clean and deduplicate values
    cleaned_values = []
    for v in unique_values:
        v_str = str(v).strip()
        if v_str and v_str.lower() not in ['nan', 'none', '无', '-', '']:
            cleaned_values.append(v_str)

    # Remove duplicates while preserving order
    seen = set()
    unique_cleaned = []
    for v in cleaned_values:
        if v not in seen:
            seen.add(v)
            unique_cleaned.append(v)

    # Create entity list with IDs
    entities = []
    value_to_id_map = {}

    for i, value in enumerate(unique_cleaned):
        entity_id = f"{entity_type}_{i}"
        entities.append({
            "entity_id": entity_id,
            "entity_type": entity_type,
            "value": value,
            "source_column": column_name
        })
        value_to_id_map[value] = entity_id

    # Store in state
    extracted = get_state_value(tool_context, EXTRACTED_ENTITIES, {})
    extracted[entity_type] = entities
    set_state_value(tool_context, EXTRACTED_ENTITIES, extracted)

    entity_maps = get_state_value(tool_context, ENTITY_MAPS, {})
    entity_maps[entity_type] = value_to_id_map
    set_state_value(tool_context, ENTITY_MAPS, entity_maps)

    logger.info(f"Extracted {len(entities)} {entity_type} entities from column '{column_name}'")

    return tool_success("extraction_result", {
        "entity_type": entity_type,
        "entities": entities,
        "value_to_id_map": value_to_id_map,
        "count": len(entities)
    })


def extract_all_entities(
    file_path: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Extracts all entities from a survey file based on column classification.

    Args:
        file_path: Path to the survey file
        tool_context: ADK ToolContext

    Returns:
        Dictionary with all extracted entities by type
    """
    # Get column classification from state
    classifications = get_state_value(tool_context, COLUMN_CLASSIFICATION, {})

    if not classifications:
        return tool_error("No column classification found. Run classify_all_columns first.")

    # Load the data
    config = get_config()
    import_dir = config.neo4j.import_dir

    if import_dir:
        full_path = Path(import_dir) / file_path
    else:
        full_path = Path(file_path)

    if not full_path.exists():
        return tool_error(f"File not found: {file_path}")

    try:
        if full_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(full_path)
        elif full_path.suffix.lower() == '.csv':
            df = pd.read_csv(full_path)
        else:
            return tool_error(f"Unsupported file format: {full_path.suffix}")
    except Exception as e:
        return tool_error(f"Error loading file: {str(e)}")

    # Find entity_selection columns and extract entities
    results = {}

    for column_name, info in classifications.items():
        if info.get("category") == "entity_selection":
            entity_type = info.get("entity_type")
            if entity_type:
                result = extract_entities_from_column(df, column_name, entity_type, tool_context)
                if result.get("status") == "success":
                    results[entity_type] = result["extraction_result"]

    # Summary
    summary = {
        entity_type: len(data.get("entities", []))
        for entity_type, data in results.items()
    }

    logger.info(f"Entity extraction complete. Summary: {summary}")

    return tool_success("all_entities", {
        "results": results,
        "summary": summary
    })


def create_respondent_table(
    file_path: str,
    id_column: str,
    tool_context: ToolContext,
    demographic_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Creates respondent table with entity foreign keys.

    Args:
        file_path: Path to the survey file
        id_column: Column containing respondent IDs
        tool_context: ADK ToolContext
        demographic_columns: List of demographic column names (optional, auto-detected if not provided)

    Returns:
        Dictionary with respondent DataFrame info
    """
    # Get classifications and entity maps from state
    classifications = get_state_value(tool_context, COLUMN_CLASSIFICATION, {})
    entity_maps = get_state_value(tool_context, ENTITY_MAPS, {})

    if not classifications:
        return tool_error("No column classification found. Run classify_all_columns first.")

    if not entity_maps:
        return tool_error("No entity maps found. Run extract_all_entities first.")

    # Load the data
    config = get_config()
    import_dir = config.neo4j.import_dir

    if import_dir:
        full_path = Path(import_dir) / file_path
    else:
        full_path = Path(file_path)

    if not full_path.exists():
        return tool_error(f"File not found: {file_path}")

    try:
        if full_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(full_path)
        elif full_path.suffix.lower() == '.csv':
            df = pd.read_csv(full_path)
        else:
            return tool_error(f"Unsupported file format: {full_path.suffix}")
    except Exception as e:
        return tool_error(f"Error loading file: {str(e)}")

    if id_column not in df.columns:
        return tool_error(f"ID column '{id_column}' not found in DataFrame")

    # Auto-detect demographic columns if not provided
    if demographic_columns is None:
        demographic_columns = []
        for col, info in classifications.items():
            if info.get("category") == "demographic":
                demographic_columns.append(col)

    # Find entity selection columns
    entity_columns = {}
    for col, info in classifications.items():
        if info.get("category") == "entity_selection":
            entity_type = info.get("entity_type")
            if entity_type:
                entity_columns[entity_type] = col

    # Build respondent table
    respondent_data = []

    for _, row in df.iterrows():
        record = {
            "respondent_id": row[id_column]
        }

        # Add demographic columns
        for col in demographic_columns:
            if col in df.columns:
                # Clean column name for output
                clean_name = col
                # Remove question number prefix if present
                if '、' in clean_name:
                    clean_name = clean_name.split('、', 1)[1]
                record[clean_name] = row[col]

        # Add entity foreign keys
        for entity_type, col in entity_columns.items():
            value = row[col]
            if pd.notna(value):
                value_str = str(value).strip()
                entity_id = entity_maps.get(entity_type, {}).get(value_str)
                record[f"{entity_type.lower()}_id"] = entity_id
                record[f"{entity_type.lower()}_value"] = value_str
            else:
                record[f"{entity_type.lower()}_id"] = None
                record[f"{entity_type.lower()}_value"] = None

        respondent_data.append(record)

    # Create DataFrame
    respondent_df = pd.DataFrame(respondent_data)

    # Store in state
    set_state_value(tool_context, RESPONDENT_DATA, respondent_df.to_dict('records'))

    logger.info(f"Created respondent table with {len(respondent_df)} rows and {len(respondent_df.columns)} columns")

    return tool_success("respondent_table", {
        "row_count": len(respondent_df),
        "columns": list(respondent_df.columns),
        "sample": respondent_df.head(5).to_dict('records')
    })


def save_entities_to_csv(
    output_dir: str,
    tool_context: ToolContext,
    prefix: str = "survey_parsed"
) -> Dict[str, Any]:
    """
    Saves extracted entities and respondent table to CSV files.

    Args:
        output_dir: Directory to save the CSV files
        tool_context: ADK ToolContext
        prefix: Prefix for output file names

    Returns:
        Dictionary with saved file paths
    """
    config = get_config()
    import_dir = config.neo4j.import_dir

    if import_dir:
        full_output_dir = Path(import_dir) / output_dir
    else:
        full_output_dir = Path(output_dir)

    full_output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Save entity files
    extracted = get_state_value(tool_context, EXTRACTED_ENTITIES, {})

    for entity_type, entities in extracted.items():
        if entities:
            df = pd.DataFrame(entities)
            file_name = f"{prefix}_{entity_type.lower()}_entities.csv"
            file_path = full_output_dir / file_name
            df.to_csv(file_path, index=False, encoding='utf-8')
            saved_files.append(str(file_path))
            logger.info(f"Saved {len(entities)} {entity_type} entities to {file_path}")

    # Save respondent table
    respondent_data = get_state_value(tool_context, RESPONDENT_DATA, [])

    if respondent_data:
        df = pd.DataFrame(respondent_data)
        file_name = f"{prefix}_respondents.csv"
        file_path = full_output_dir / file_name
        df.to_csv(file_path, index=False, encoding='utf-8')
        saved_files.append(str(file_path))
        logger.info(f"Saved {len(df)} respondent records to {file_path}")

    return tool_success("saved_files", {
        "files": saved_files,
        "output_dir": str(full_output_dir)
    })


def get_extracted_entities(
    entity_type: Optional[str],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Get extracted entities from state.

    Args:
        entity_type: Type of entity to retrieve (Brand, Model, Store) or None for all
        tool_context: ADK ToolContext

    Returns:
        Extracted entities
    """
    extracted = get_state_value(tool_context, EXTRACTED_ENTITIES, {})

    if not extracted:
        return tool_error("No extracted entities found. Run extract_all_entities first.")

    if entity_type:
        if entity_type not in extracted:
            return tool_error(f"No entities found for type: {entity_type}")
        return tool_success("entities", extracted[entity_type])

    return tool_success("entities", extracted)


def get_entity_map(
    entity_type: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Get the value-to-ID mapping for an entity type.

    Args:
        entity_type: Type of entity (Brand, Model, Store)
        tool_context: ADK ToolContext

    Returns:
        Value to ID mapping dictionary
    """
    entity_maps = get_state_value(tool_context, ENTITY_MAPS, {})

    if not entity_maps:
        return tool_error("No entity maps found. Run extract_all_entities first.")

    if entity_type not in entity_maps:
        return tool_error(f"No mapping found for entity type: {entity_type}")

    return tool_success("entity_map", entity_maps[entity_type])


# Export tools list for agent registration
SURVEY_ENTITY_EXTRACTION_TOOLS = [
    extract_entities_from_column,
    extract_all_entities,
    create_respondent_table,
    save_entities_to_csv,
    get_extracted_entities,
    get_entity_map
]

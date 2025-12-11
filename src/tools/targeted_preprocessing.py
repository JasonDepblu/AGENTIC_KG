"""
Targeted Preprocessing Tools for Schema-First Pipeline.

This module provides tools for extracting entities and relationships
based on an approved target schema. The schema defines what to extract,
and these tools implement the extraction logic.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from google.adk.tools import ToolContext

from .common import tool_success, tool_error, set_state_value, get_state_value
from ..config import get_neo4j_import_dir
from ..models.target_schema import (
    TargetSchema,
    NodeDefinition,
    RelationshipDefinition,
    APPROVED_TARGET_SCHEMA_KEY,
)
# Import LLM detection for domain-agnostic rating conversion
from .llm_detection import convert_text_to_rating as llm_convert_ratings

# Configuration flag for LLM-based rating conversion
# Set to True to use LLM for text-to-rating conversion (domain-agnostic)
# Set to False to use hardcoded TEXT_TO_RATING_MAP (faster but domain-specific)
USE_LLM_RATING = True

logger = logging.getLogger(__name__)

# State keys for targeted preprocessing
TARGETED_EXTRACTION_RESULTS = "targeted_extraction_results"
TARGETED_ENTITY_MAPS = "targeted_entity_maps"
TARGETED_RELATIONSHIP_DATA = "targeted_relationship_data"
TARGETED_PREPROCESSING_COMPLETE = "targeted_preprocessing_complete"
GENERATED_FILES = "generated_files"
PREPROCESSING_FEEDBACK_KEY = "preprocessing_feedback"
NEEDS_SCHEMA_REVISION = "needs_schema_revision"
SCHEMA_REVISION_REASON = "schema_revision_reason"
# Progress tracking for checkpoint/resume support
EXTRACTION_PROGRESS = "extraction_progress"


def _get_import_dir() -> Optional[Path]:
    """Get the import directory as a Path object."""
    import_dir_path = get_neo4j_import_dir()
    if not import_dir_path:
        return None
    return Path(import_dir_path)


def _safe_regex_match(pattern: str, text: str, flags: int = re.IGNORECASE) -> bool:
    """
    Safely execute regex match, catching invalid pattern errors.

    Args:
        pattern: The regex pattern to match
        text: The text to search in
        flags: Regex flags (default: re.IGNORECASE)

    Returns:
        True if pattern matches, False otherwise (including for invalid patterns)
    """
    try:
        return bool(re.search(pattern, text, flags))
    except re.error as e:
        logger.warning(f"Invalid regex pattern '{pattern}': {e}. Falling back to substring match.")
        # Fallback to simple substring match
        try:
            return pattern.lower() in text.lower()
        except Exception:
            return False


def _get_approved_schema(tool_context: ToolContext) -> Optional[TargetSchema]:
    """Get the approved target schema from state."""
    schema_dict = get_state_value(tool_context, APPROVED_TARGET_SCHEMA_KEY)
    if not schema_dict:
        return None
    return TargetSchema.from_dict(schema_dict)


def _normalize_column_name(col: str) -> str:
    """Normalize a column name by stripping whitespace and fixing duplicate prefixes."""
    if not isinstance(col, str):
        col = str(col)
    # Strip whitespace
    col = col.strip()
    # Fix duplicate number prefixes like "10、10 xxx" -> "10、xxx"
    col = re.sub(r'^(\d+[、\.])\1+', r'\1', col)
    return col


def _load_data_file(file_path: str) -> Optional[pd.DataFrame]:
    """Load a data file (CSV or Excel) with normalized column names."""
    import_dir = _get_import_dir()
    if import_dir:
        full_path = import_dir / file_path
    else:
        full_path = Path(file_path)

    if not full_path.exists():
        return None

    suffix = full_path.suffix.lower()
    if suffix == '.csv':
        df = pd.read_csv(full_path)
    elif suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(full_path)
    else:
        return None

    # Normalize column names to avoid whitespace and duplicate prefix issues
    df.columns = [_normalize_column_name(col) for col in df.columns]
    return df


def _find_column_by_pattern(df: pd.DataFrame, pattern: str) -> Optional[str]:
    """Find a column that matches the given pattern."""
    for col in df.columns:
        # Normalize column name by stripping whitespace
        col_normalized = col.strip() if isinstance(col, str) else str(col)
        if _safe_regex_match(pattern, col_normalized):
            return col
    return None


def _find_columns_by_pattern(df: pd.DataFrame, pattern: str) -> List[str]:
    """Find all columns that match the given pattern."""
    matched = []
    for col in df.columns:
        # Normalize column name by stripping whitespace
        col_normalized = col.strip() if isinstance(col, str) else str(col)
        if _safe_regex_match(pattern, col_normalized):
            matched.append(col)
    return matched


def _extract_name_from_column_header(column_name: str, name_regex: Optional[str] = None) -> str:
    """
    Extract semantic name from a column header.

    For survey columns like '14、"外观设计"方面，您打多少分？',
    extracts the quoted part '外观设计'.

    Args:
        column_name: The full column name
        name_regex: Optional custom regex pattern with capture group

    Returns:
        Extracted name or the original column name if no match
    """
    if name_regex:
        try:
            match = re.search(name_regex, column_name)
            if match:
                # Try to get capture group 1, fall back to group 0 (full match)
                try:
                    return match.group(1).strip()
                except IndexError:
                    # Pattern matched but has no capture group
                    # Log warning and fall back to removing the matched suffix
                    logger.warning(
                        f"Regex pattern '{name_regex}' has no capture group. "
                        f"Falling back to removing matched part from '{column_name}'"
                    )
                    # Remove the matched part from the column name
                    matched_text = match.group(0)
                    result = column_name.replace(matched_text, '').strip('_').strip()
                    return result if result else column_name
        except re.error as e:
            logger.error(f"Invalid regex pattern '{name_regex}': {e}")

    # Default: try common quote patterns (Chinese and English)
    patterns = [
        r'["\'""''「」]([^"\'""''「」]+)["\'""''「」]',  # Quoted text
        r'["""]([^"""]+)["""]',  # Chinese quotes
        r"[「」]([^「」]+)[「」]",  # Japanese brackets
    ]

    for pattern in patterns:
        try:
            match = re.search(pattern, column_name)
            if match:
                return match.group(1).strip()
        except re.error as e:
            logger.warning(f"Regex pattern error in _extract_name_from_column_header: {e}")
            continue

    # No quotes found, return cleaned column name
    return column_name.strip()


def get_schema_extraction_plan(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get the extraction plan based on the approved target schema.

    Returns a plan showing what entities and relationships will be extracted
    and from which columns.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with the extraction plan
    """
    schema = _get_approved_schema(tool_context)
    if not schema:
        return tool_error("No approved target schema found. Complete schema design first.")

    plan = {
        "schema_name": schema.name,
        "nodes_to_extract": [],
        "relationships_to_extract": [],
    }

    # List nodes to extract
    for label, node in schema.nodes.items():
        node_plan = {
            "label": label,
            "unique_property": node.unique_property,
            "properties": node.properties,
            "extraction_hints": node.extraction_hints,
        }
        plan["nodes_to_extract"].append(node_plan)

    # List relationships to extract
    for rel_type, rel in schema.relationships.items():
        rel_plan = {
            "type": rel_type,
            "from": rel.from_node,
            "to": rel.to_node,
            "properties": rel.properties,
            "extraction_hints": rel.extraction_hints,
        }
        plan["relationships_to_extract"].append(rel_plan)

    return tool_success("extraction_plan", plan)


def _is_extracted_output_file(file_path: str) -> bool:
    """Check if a file path appears to be an already-extracted output file."""
    markers = ['_entities', '_relationships', 'extracted_data']
    file_lower = file_path.lower()
    return any(marker in file_lower for marker in markers)


def extract_entities_for_node(
    file_path: str,
    node_label: str,
    tool_context: ToolContext,
    column_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract entities for a specific node type from a data file.

    Uses the target schema's extraction hints to determine which column to extract from.
    Supports two extraction modes:
    1. Row-based: Extract unique values from a specific column (default)
    2. Column-header: Extract entity names from column headers (for Aspect nodes)

    Args:
        file_path: Path to the data file
        node_label: Label of the node type to extract (e.g., "Brand")
        tool_context: ADK ToolContext
        column_name: Optional explicit column name (overrides extraction hints)

    Returns:
        Dictionary with extracted entities
    """
    from .task_manager import save_checkpoint

    # Check if already completed (for checkpoint/resume support)
    file_name = Path(file_path).name if file_path else "unknown"
    progress_key = f"entity:{file_name}:{node_label}"
    progress = get_state_value(tool_context, EXTRACTION_PROGRESS, {})

    if progress.get(progress_key) == "completed":
        logger.info(f"Skipping already completed extraction: {progress_key}")
        existing_results = get_state_value(tool_context, TARGETED_EXTRACTION_RESULTS, {})
        existing_entities = existing_results.get(node_label, [])
        return tool_success("extraction_skipped", {
            "node_label": node_label,
            "count": len(existing_entities),
            "reason": "Already extracted (checkpoint)",
        })

    # Validate: prevent extraction from already-extracted output files
    if _is_extracted_output_file(file_path):
        logger.warning(
            f"File '{file_path}' appears to be an already-extracted output file. "
            f"Use the original source file from approved_files instead."
        )
        return tool_error(
            f"Cannot extract from '{file_path}' - this appears to be an already-extracted "
            f"entity/relationship file. Please use 'get_approved_files()' to get the original "
            f"source data file, then call extract_entities_for_node with that file path."
        )

    schema = _get_approved_schema(tool_context)
    if not schema:
        return tool_error("No approved target schema found.")

    node = schema.get_node(node_label)
    if not node:
        return tool_error(f"Node type '{node_label}' not found in schema.")

    # Load data
    df = _load_data_file(file_path)
    if df is None:
        return tool_error(f"Could not load file: {file_path}")

    hints = node.extraction_hints
    source_type = hints.get("source_type", "entity_selection")

    # Get existing entities to ensure globally unique IDs
    results = get_state_value(tool_context, TARGETED_EXTRACTION_RESULTS, {})
    existing_entities = results.get(node_label, [])
    existing_count = len(existing_entities)

    # Get existing entity maps to avoid duplicate names
    entity_maps = get_state_value(tool_context, TARGETED_ENTITY_MAPS, {})
    existing_map = entity_maps.get(node_label, {})

    entities = []
    value_to_id = dict(existing_map)  # Start with existing mappings

    # ============================================================
    # Mode 1: Extract from column headers (for Aspect nodes)
    # ============================================================
    if source_type in ["column_header", "rating_column_header"]:
        # Find columns matching the pattern
        pattern = hints.get("column_pattern", r"打多少分|评分")
        name_regex = hints.get("name_regex")

        matched_columns = _find_columns_by_pattern(df, pattern)

        if not matched_columns:
            return tool_error(
                f"No columns matching pattern '{pattern}' found for {node_label}. "
                f"Available columns: {list(df.columns[:10])}..."
            )

        new_entity_count = 0
        for col in matched_columns:
            # Extract semantic name from column header
            aspect_name = _extract_name_from_column_header(col, name_regex)

            # Skip if already seen
            if aspect_name in value_to_id:
                continue

            entity_id = f"{node_label}_{existing_count + new_entity_count}"
            new_entity_count += 1
            entity = {
                node.unique_property: entity_id,
                "name": aspect_name,
                "source_column": col,
                "source_file": file_path,
            }
            entities.append(entity)
            # Register multiple name variants for consistent lookup
            value_to_id[aspect_name] = entity_id
            value_to_id[col] = entity_id  # Also register full column name
            # Register simplified name (without _score suffix) for relationship lookup
            simplified = re.sub(r'_score$', '', aspect_name)
            if simplified != aspect_name:
                value_to_id[simplified] = entity_id

        logger.info(f"Extracted {len(entities)} {node_label} entities from column headers")

    # ============================================================
    # Mode 2: Row-based extraction (default)
    # ============================================================
    else:
        # Determine column to extract from
        if column_name:
            target_column = column_name
        elif "column_pattern" in hints:
            pattern = hints["column_pattern"]
            target_column = _find_column_by_pattern(df, pattern)
            if not target_column:
                # List available columns for debugging
                available = [c.strip() if isinstance(c, str) else str(c) for c in df.columns[:10]]
                return tool_error(
                    f"No column matching pattern '{pattern}' found for {node_label}. "
                    f"Available columns (first 10): {available}"
                )
        elif "column_name" in hints:
            target_column = hints["column_name"]
        else:
            return tool_error(
                f"No extraction hints for node '{node_label}'. "
                "Provide column_name or set extraction_hints in schema."
            )

        if target_column not in df.columns:
            # Try fuzzy match - maybe whitespace difference
            for col in df.columns:
                if col.strip() == target_column.strip():
                    target_column = col
                    break
            else:
                return tool_error(f"Column '{target_column}' not found in file.")

        # Extract unique values
        unique_values = df[target_column].dropna().unique()

        new_entity_count = 0
        for value in unique_values:
            value_str = str(value).strip()
            if not value_str or value_str.lower() in ['nan', 'none', '無', '-', '']:
                continue

            # Skip if already seen
            if value_str in value_to_id:
                continue

            entity_id = f"{node_label}_{existing_count + new_entity_count}"
            new_entity_count += 1
            entity = {
                node.unique_property: entity_id,
                "name": value_str,
                "source_column": target_column,
                "source_file": file_path,
            }

            # Add additional properties if they exist in the data
            for prop in node.properties:
                if prop in df.columns and prop != node.unique_property:
                    # Get first non-null value for this entity
                    mask = df[target_column] == value
                    prop_values = df.loc[mask, prop].dropna()
                    if len(prop_values) > 0:
                        entity[prop] = str(prop_values.iloc[0])

            entities.append(entity)
            value_to_id[value_str] = entity_id

        logger.info(f"Extracted {len(entities)} {node_label} entities from column '{target_column}'")

    # Store in state (append to existing, reusing results from above)
    existing_entities.extend(entities)
    results[node_label] = existing_entities
    set_state_value(tool_context, TARGETED_EXTRACTION_RESULTS, results)

    # Update entity maps (value_to_id already includes existing_map)
    entity_maps[node_label] = value_to_id
    set_state_value(tool_context, TARGETED_ENTITY_MAPS, entity_maps)

    # Mark as completed and save checkpoint
    progress[progress_key] = "completed"
    set_state_value(tool_context, EXTRACTION_PROGRESS, progress)
    save_checkpoint(tool_context)

    return tool_success("extracted_entities", {
        "node_label": node_label,
        "count": len(entities),
        "unique_property": node.unique_property,
        "source_type": source_type,
        "sample": entities[:5] if entities else [],
    })


def extract_relationship_data(
    file_path: str,
    relationship_type: str,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Extract relationship data based on schema definition.

    For survey data, this typically extracts rating relationships
    or entity associations from the data file.

    Supported source_types:
    - rating_column: Extract ratings from columns matching pattern (Respondent -> Aspect)
    - entity_reference: Extract relationships from entity selection columns (Respondent -> Brand/Model/Store)
    - foreign_key: Extract relationships via foreign key columns

    Args:
        file_path: Path to the data file
        relationship_type: Type of relationship to extract (e.g., "RATES")
        tool_context: ADK ToolContext

    Returns:
        Dictionary with extracted relationship data
    """
    from .task_manager import save_checkpoint

    # Check if already completed (for checkpoint/resume support)
    file_name = Path(file_path).name if file_path else "unknown"
    progress_key = f"rel:{file_name}:{relationship_type}"
    progress = get_state_value(tool_context, EXTRACTION_PROGRESS, {})

    if progress.get(progress_key) == "completed":
        logger.info(f"Skipping already completed extraction: {progress_key}")
        existing_rel_data = get_state_value(tool_context, TARGETED_RELATIONSHIP_DATA, {})
        existing_rels = existing_rel_data.get(relationship_type, [])
        return tool_success("extraction_skipped", {
            "relationship_type": relationship_type,
            "count": len(existing_rels),
            "reason": "Already extracted (checkpoint)",
        })

    # Validate: prevent extraction from already-extracted output files
    if _is_extracted_output_file(file_path):
        logger.warning(
            f"File '{file_path}' appears to be an already-extracted output file. "
            f"Use the original source file from approved_files instead."
        )
        return tool_error(
            f"Cannot extract from '{file_path}' - this appears to be an already-extracted "
            f"entity/relationship file. Please use 'get_approved_files()' to get the original "
            f"source data file, then call extract_relationship_data with that file path."
        )

    schema = _get_approved_schema(tool_context)
    if not schema:
        return tool_error("No approved target schema found.")

    rel = schema.get_relationship(relationship_type)
    if not rel:
        return tool_error(f"Relationship '{relationship_type}' not found in schema.")

    # Load data
    df = _load_data_file(file_path)
    if df is None:
        return tool_error(f"Could not load file: {file_path}")

    # Get entity maps for source and target nodes
    entity_maps = get_state_value(tool_context, TARGETED_ENTITY_MAPS, {})

    from_map = entity_maps.get(rel.from_node, {})
    to_map = entity_maps.get(rel.to_node, {})

    relationships = []
    hints = rel.extraction_hints
    source_type = hints.get("source_type", "")

    # Auto-infer source_type if not specified (fallback logic)
    if not source_type:
        rel_type_upper = relationship_type.upper()
        if "RATE" in rel_type_upper or "SCORE" in rel_type_upper:
            source_type = "rating_column"
            logger.info(f"Auto-inferred source_type='rating_column' for {relationship_type}")
        elif any(x in rel_type_upper for x in ["EVALUATED", "VISITED", "SELECTED", "CHOSE"]):
            source_type = "entity_reference"
            logger.info(f"Auto-inferred source_type='entity_reference' for {relationship_type}")
            # Auto-infer column_pattern from target node label if not set
            if not hints.get("column_pattern"):
                target_label = rel.to_node.lower()
                hints["column_pattern"] = target_label
                logger.info(f"Auto-inferred column_pattern='{target_label}' for {relationship_type}")

    # Helper function to find column by pattern with normalization
    def find_column(pattern: str) -> Optional[str]:
        for col in df.columns:
            col_normalized = col.strip() if isinstance(col, str) else str(col)
            if _safe_regex_match(pattern, col_normalized):
                return col
        return None

    # Helper to get respondent column
    def get_respondent_column():
        id_col = hints.get("respondent_column")
        if id_col:
            # Try exact match first
            if id_col in df.columns:
                return id_col
            # Try pattern match
            found = find_column(id_col)
            if found:
                return found
        # Default to first column (usually 序号)
        return df.columns[0]

    # ============================================================
    # Mode 1: Extract ratings from rating columns
    # ============================================================
    if source_type == "rating_column":
        column_pattern = hints.get("column_pattern", r"打多少分|评分")

        for col in df.columns:
            col_normalized = col.strip() if isinstance(col, str) else str(col)
            if _safe_regex_match(column_pattern, col_normalized):
                # Extract aspect name from column header
                aspect_name = _extract_name_from_column_header(col, hints.get("name_regex"))

                # Get target entity ID (e.g., aspect)
                # Try multiple name variants to find a match
                to_id = to_map.get(aspect_name)
                if not to_id:
                    # Try without _score suffix
                    simplified_name = re.sub(r'_score$', '', aspect_name)
                    to_id = to_map.get(simplified_name)
                if not to_id:
                    # Try the full column name
                    to_id = to_map.get(col)
                if not to_id:
                    # Still not found - create new ID and log warning
                    logger.warning(
                        f"Aspect '{aspect_name}' not found in entity map. "
                        f"Entity map keys: {list(to_map.keys())[:5]}... "
                        f"Creating new ID. Consider using source_type='column_header' for Aspect nodes."
                    )
                    to_id = f"Aspect_{len(to_map)}"
                    to_map[aspect_name] = to_id

                # Extract ratings for each row
                id_col = get_respondent_column()
                for _, row in df.iterrows():
                    respondent_val = row.get(id_col)
                    if pd.isna(respondent_val):
                        continue

                    # Use the extracted entity map so relationship endpoints align with extracted Respondent nodes
                    respondent_key = str(respondent_val).strip()
                    if not respondent_key:
                        continue
                    from_id = from_map.get(respondent_key)
                    if not from_id:
                        from_id = f"{rel.from_node}_{respondent_key}"
                        from_map[respondent_key] = from_id

                    score = row[col]
                    if pd.notna(score):
                        try:
                            score_val = float(score)
                        except (ValueError, TypeError):
                            continue  # Skip non-numeric values

                        rel_record = {
                            "from_id": from_id,
                            "to_id": to_id,
                            "to_name": aspect_name,
                            "relationship_type": relationship_type,
                        }
                        # Add properties
                        if "score" in rel.properties:
                            rel_record["score"] = score_val
                        rel_record["source_column"] = col
                        relationships.append(rel_record)

    # ============================================================
    # Mode 2: Extract entity references (EVALUATED, VISITED, etc.)
    # ============================================================
    elif source_type == "entity_reference":
        # Find the column containing entity selections
        column_pattern = hints.get("column_pattern", "")
        if not column_pattern:
            return tool_error(
                f"entity_reference source_type requires column_pattern in hints for {relationship_type}"
            )

        target_column = find_column(column_pattern)
        if not target_column:
            available = [c.strip() if isinstance(c, str) else str(c) for c in df.columns[:15]]
            return tool_error(
                f"No column matching pattern '{column_pattern}' found for {relationship_type}. "
                f"Available columns (first 15): {available}"
            )

        id_col = get_respondent_column()

        for _, row in df.iterrows():
            respondent_val = row.get(id_col)
            if pd.isna(respondent_val):
                continue
            respondent_key = str(respondent_val).strip()
            if not respondent_key:
                continue
            from_id = from_map.get(respondent_key)
            if not from_id:
                from_id = f"{rel.from_node}_{respondent_key}"
                from_map[respondent_key] = from_id

            entity_value = row[target_column]
            if pd.notna(entity_value):
                entity_str = str(entity_value).strip()
                if not entity_str or entity_str.lower() in ['nan', 'none', '无', '-', '']:
                    continue

                # Get target entity ID
                to_id = to_map.get(entity_str)
                if not to_id:
                    # Entity not found in map, log warning
                    logger.warning(f"Entity '{entity_str}' not found in {rel.to_node} map")
                    continue

                rel_record = {
                    "from_id": from_id,
                    "to_id": to_id,
                    "to_name": entity_str,
                    "relationship_type": relationship_type,
                    "source_column": target_column,
                }
                # Add any additional properties
                for prop in rel.properties:
                    if prop in df.columns:
                        prop_val = row[prop]
                        if pd.notna(prop_val):
                            rel_record[prop] = prop_val

                relationships.append(rel_record)

    # ============================================================
    # Mode 3: Foreign key relationships
    # ============================================================
    elif source_type == "foreign_key":
        from_column = hints.get("from_column")
        to_column = hints.get("to_column")

        # Try to find columns by pattern if not exact match
        if from_column and from_column not in df.columns:
            from_column = find_column(from_column)
        if to_column and to_column not in df.columns:
            to_column = find_column(to_column)

        if from_column and to_column:
            for _, row in df.iterrows():
                from_value = row.get(from_column)
                to_value = row.get(to_column)

                if pd.notna(from_value) and pd.notna(to_value):
                    from_id = from_map.get(str(from_value).strip())
                    to_id = to_map.get(str(to_value).strip())

                    if from_id and to_id:
                        rel_record = {
                            "from_id": from_id,
                            "to_id": to_id,
                            "relationship_type": relationship_type,
                        }
                        # Add properties
                        for prop in rel.properties:
                            if prop in df.columns:
                                rel_record[prop] = row[prop]
                        relationships.append(rel_record)
        else:
            return tool_error(
                f"foreign_key source_type requires from_column and to_column in hints. "
                f"Got from_column={from_column}, to_column={to_column}"
            )

    # ============================================================
    # Default: warn and try basic extraction
    # ============================================================
    else:
        logger.warning(
            f"Unknown source_type '{source_type}' for {relationship_type}. "
            "Supported types: rating_column, entity_reference, foreign_key"
        )

    # Store in state
    rel_data = get_state_value(tool_context, TARGETED_RELATIONSHIP_DATA, {})
    rel_data[relationship_type] = relationships
    set_state_value(tool_context, TARGETED_RELATIONSHIP_DATA, rel_data)

    # Update entity maps
    set_state_value(tool_context, TARGETED_ENTITY_MAPS, entity_maps)

    # Mark as completed and save checkpoint
    progress[progress_key] = "completed"
    set_state_value(tool_context, EXTRACTION_PROGRESS, progress)
    save_checkpoint(tool_context)

    logger.info(f"Extracted {len(relationships)} relationships of type '{relationship_type}'")

    return tool_success("extracted_relationships", {
        "relationship_type": relationship_type,
        "from_node": rel.from_node,
        "to_node": rel.to_node,
        "count": len(relationships),
        "sample": relationships[:5] if relationships else [],
    })


def save_extracted_data(
    tool_context: ToolContext,
    prefix: str = "targeted",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Save all extracted entity and relationship data to CSV files.

    Args:
        tool_context: ADK ToolContext
        prefix: Prefix for output file names
        output_dir: Directory to save files (optional, uses task dir if available)

    Returns:
        Dictionary with saved file paths
    """
    from .task_manager import get_task_config, save_checkpoint

    # Determine output directory
    task_config = get_task_config(tool_context)
    if task_config and not output_dir:
        # Use task's extracted directory
        full_output_dir = Path(task_config["extracted_dir"])
    elif output_dir:
        import_dir = _get_import_dir()
        if import_dir:
            full_output_dir = import_dir / output_dir
        else:
            full_output_dir = Path(output_dir)
    else:
        # Fallback for backward compatibility
        import_dir = _get_import_dir()
        if import_dir:
            full_output_dir = import_dir / "extracted_data"
        else:
            full_output_dir = Path("extracted_data")

    full_output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Save entity files
    results = get_state_value(tool_context, TARGETED_EXTRACTION_RESULTS, {})
    for node_label, entities in results.items():
        if entities:
            df = pd.DataFrame(entities)
            file_name = f"{prefix}_{node_label.lower()}_entities.csv"
            file_path = full_output_dir / file_name
            df.to_csv(file_path, index=False, encoding='utf-8')
            saved_files.append({
                "type": "node",
                "label": node_label,
                "path": str(file_path),
                "count": len(entities),
            })
            logger.info(f"Saved {len(entities)} {node_label} entities to {file_path}")

    # Save relationship files
    rel_data = get_state_value(tool_context, TARGETED_RELATIONSHIP_DATA, {})
    for rel_type, relationships in rel_data.items():
        if relationships:
            df = pd.DataFrame(relationships)
            file_name = f"{prefix}_{rel_type.lower()}_relationships.csv"
            file_path = full_output_dir / file_name
            df.to_csv(file_path, index=False, encoding='utf-8')
            saved_files.append({
                "type": "relationship",
                "relationship_type": rel_type,
                "path": str(file_path),
                "count": len(relationships),
            })
            logger.info(f"Saved {len(relationships)} {rel_type} relationships to {file_path}")

    # Store file list in state
    set_state_value(tool_context, GENERATED_FILES, saved_files)

    # Save checkpoint after data is saved
    save_checkpoint(tool_context)

    return tool_success("saved_files", {
        "output_dir": str(full_output_dir),
        "files": saved_files,
        "total_files": len(saved_files),
    })


def get_extraction_summary(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get a summary of all extracted data.

    Args:
        tool_context: ADK ToolContext

    Returns:
        Dictionary with extraction summary
    """
    results = get_state_value(tool_context, TARGETED_EXTRACTION_RESULTS, {})
    rel_data = get_state_value(tool_context, TARGETED_RELATIONSHIP_DATA, {})

    summary = {
        "entities": {
            label: len(entities)
            for label, entities in results.items()
        },
        "relationships": {
            rel_type: len(rels)
            for rel_type, rels in rel_data.items()
        },
        "total_entities": sum(len(e) for e in results.values()),
        "total_relationships": sum(len(r) for r in rel_data.values()),
    }

    return tool_success("extraction_summary", summary)


def complete_targeted_preprocessing(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Mark targeted preprocessing as complete.

    This signals that all entities and relationships have been extracted
    according to the target schema and the pipeline can proceed.

    Args:
        tool_context: ADK ToolContext

    Returns:
        Dictionary with completion status
    """
    results = get_state_value(tool_context, TARGETED_EXTRACTION_RESULTS, {})
    rel_data = get_state_value(tool_context, TARGETED_RELATIONSHIP_DATA, {})

    if not results:
        return tool_error("No entities extracted. Extract entities first.")

    # Mark as complete
    set_state_value(tool_context, TARGETED_PREPROCESSING_COMPLETE, True)

    summary = {
        "entities": {
            label: len(entities)
            for label, entities in results.items()
        },
        "relationships": {
            rel_type: len(rels)
            for rel_type, rels in rel_data.items()
        },
    }

    logger.info("Targeted preprocessing completed.")

    return tool_success("preprocessing_complete", {
        "complete": True,
        "summary": summary,
    })


def generate_construction_rules(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Generate construction rules from the target schema and extracted data.

    This creates the construction plan that will be used by the KG Builder
    to import the extracted data into Neo4j.

    Args:
        tool_context: ADK ToolContext

    Returns:
        Dictionary with generated construction rules
    """
    schema = _get_approved_schema(tool_context)
    if not schema:
        return tool_error("No approved target schema found.")

    generated_files = get_state_value(tool_context, GENERATED_FILES, [])
    if not generated_files:
        return tool_error("No generated files found. Save extracted data first.")

    # Build construction plan in a format the KG builder can import directly
    construction_plan = {}

    # Map node label -> unique property for later use in relationships
    node_unique_props = {}

    for file_info in generated_files:
        if file_info["type"] != "node":
            continue
        node = schema.get_node(file_info["label"])
        if not node:
            continue

        node_rule = {
            "construction_type": "node",
            "source_file": file_info["path"],
            "label": node.label,
            "unique_column_name": node.unique_property,
            "properties": node.properties,
        }
        construction_plan[f"node::{node.label}"] = node_rule
        node_unique_props[node.label] = node.unique_property

    # Relationships: use generic CSV column names produced by save_extracted_data (from_id, to_id)
    for file_info in generated_files:
        if file_info["type"] != "relationship":
            continue
        rel = schema.get_relationship(file_info["relationship_type"])
        if not rel:
            continue

        rel_rule = {
            "construction_type": "relationship",
            "source_file": file_info["path"],
            "relationship_type": rel.relationship_type,
            "from_node_label": rel.from_node,
            "to_node_label": rel.to_node,
            "from_node_column": "from_id",
            "to_node_column": "to_id",
            # Map CSV columns to actual node unique properties for MATCH
            "from_node_property": node_unique_props.get(rel.from_node, "from_id"),
            "to_node_property": node_unique_props.get(rel.to_node, "to_id"),
            "properties": rel.properties,
        }
        construction_plan[f"rel::{rel.relationship_type}"] = rel_rule

    # Store as proposed construction plan (for compatibility with existing pipeline)
    from .kg_construction import PROPOSED_CONSTRUCTION_PLAN
    set_state_value(tool_context, PROPOSED_CONSTRUCTION_PLAN, construction_plan)

    return tool_success("construction_rules", construction_plan)


def get_preprocessing_feedback(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get feedback from the preprocessing critic agent.

    This tool allows the preprocessing agent to check for any feedback
    or issues identified by the critic agent during the validation loop.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with feedback information
    """
    feedback = get_state_value(tool_context, PREPROCESSING_FEEDBACK_KEY, "")

    return tool_success(PREPROCESSING_FEEDBACK_KEY, {
        "feedback": feedback,
        "has_issues": bool(feedback and feedback != "valid"),
    })


# Default placeholder values to replace with null
DEFAULT_NULL_VALUES = [
    "(跳过)", "(跳過)", "跳过", "N/A", "n/a", "NA", "na",
    "-", "--", "---", "无", "無", "null", "NULL", "None",
    "(空)", "空", "(无)", "(none)", "(skip)", "skip",
]

# Common text-to-rating mappings for Chinese survey data (DEPRECATED - use LLM)
# NOTE: These mappings are kept for backward compatibility only.
# When USE_LLM_RATING = True, LLM-based conversion is used instead.
TEXT_TO_RATING_MAP = {
    "非常好": 10, "很好": 9, "好": 8, "不错": 7,
    "一般": 5, "较差": 4, "差": 3, "很差": 2, "非常差": 1,
    "非常满意": 10, "满意": 8, "一般": 5, "不满意": 3, "非常不满意": 1,
    "非常同意": 10, "同意": 8, "中立": 5, "不同意": 3, "非常不同意": 1,
}


def _convert_ratings_with_llm(
    df: pd.DataFrame,
    col: str,
    tool_context: ToolContext,
) -> int:
    """
    Convert text ratings to numeric values using LLM.

    This is the domain-agnostic alternative to TEXT_TO_RATING_MAP.

    Args:
        df: DataFrame with the data (modified in place)
        col: Column name to convert
        tool_context: ADK ToolContext

    Returns:
        Number of values converted
    """
    # Get unique non-numeric text values in the column
    text_values = []
    for val in df[col].dropna().unique():
        val_str = str(val).strip()
        # Skip if already numeric or empty
        if not val_str or val_str.replace('.', '').replace('-', '').isdigit():
            continue
        # Skip known placeholder values
        if val_str.lower() in ['nan', 'none', 'null', 'na', 'n/a', '-', '']:
            continue
        text_values.append(val_str)

    if not text_values:
        return 0

    logger.info(f"Converting {len(text_values)} unique text values in '{col}' using LLM")

    # Call LLM for conversion
    result = llm_convert_ratings(text_values, tool_context)

    if result.get("status") == "error":
        logger.warning(f"LLM rating conversion failed for '{col}': {result.get('error')}")
        return 0

    mappings = result.get("mappings", {})
    total_converted = 0

    # Apply mappings to dataframe
    for text_val, numeric_val in mappings.items():
        if numeric_val is not None:
            mask = df[col].astype(str).str.strip() == text_val
            count = mask.sum()
            if count > 0:
                df.loc[mask, col] = numeric_val
                total_converted += int(count)
                logger.debug(f"  Converted '{text_val}' -> {numeric_val} ({count} values)")

    return total_converted


def clean_columns_for_schema(
    tool_context: ToolContext,
    file_path: Optional[str] = None,
    replace_placeholders: bool = True,
    handle_rating_anomalies: bool = True,
    custom_null_values: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Clean columns mentioned in the approved schema before extraction.

    This is a schema-driven cleaning tool that only cleans columns that will
    be used for entity/relationship extraction based on the approved schema.

    Cleaning operations:
    1. Replace placeholder values (跳过, N/A, null, etc.) with empty
    2. Handle text anomalies in rating columns (e.g., "非常好" → numeric)
    3. Strip whitespace from string values

    Args:
        tool_context: ADK ToolContext for state management
        file_path: Optional file path. If None, uses first approved file.
        replace_placeholders: Replace common placeholder values with null
        handle_rating_anomalies: Convert text values in rating columns to numbers
        custom_null_values: Additional values to treat as null

    Returns:
        Dictionary with cleaning summary:
        - cleaned_file: Path to cleaned file
        - columns_cleaned: List of columns that were cleaned
        - changes_made: Details of changes per column
    """
    schema = _get_approved_schema(tool_context)
    if not schema:
        return tool_error("No approved target schema found. Complete schema design first.")

    # Get approved files
    approved_files = get_state_value(tool_context, "approved_files", [])
    if not approved_files:
        return tool_error("No approved files found.")

    # Use specified file or first approved file
    if file_path:
        target_file = file_path
    else:
        target_file = approved_files[0]

    # Load the file
    df = _load_data_file(target_file)
    if df is None:
        return tool_error(f"Could not load file: {target_file}")

    # Collect all column patterns from schema
    schema_columns = set()
    rating_column_patterns = []

    # Collect column patterns from node definitions
    for label, node in schema.nodes.items():
        hints = node.extraction_hints
        if "column_pattern" in hints:
            pattern = hints["column_pattern"]
            # Find matching columns
            matched = _find_columns_by_pattern(df, pattern)
            schema_columns.update(matched)
            # Check if this is a rating column type
            if hints.get("source_type") in ["rating_column", "rating_column_header"]:
                rating_column_patterns.append(pattern)

    # Collect column patterns from relationship definitions
    for rel_type, rel in schema.relationships.items():
        hints = rel.extraction_hints
        if "column_pattern" in hints:
            pattern = hints["column_pattern"]
            matched = _find_columns_by_pattern(df, pattern)
            schema_columns.update(matched)
            # Rating relationships
            if hints.get("source_type") == "rating_column":
                rating_column_patterns.append(pattern)

    if not schema_columns:
        # If no specific columns found, use all columns (fallback)
        schema_columns = set(df.columns)

    changes_made = {}
    total_placeholders_replaced = 0
    total_ratings_converted = 0

    # Build null values set
    null_values = set(DEFAULT_NULL_VALUES)
    if custom_null_values:
        null_values.update(custom_null_values)

    # Clean each schema column
    for col in schema_columns:
        if col not in df.columns:
            continue

        col_changes = {"placeholders_replaced": 0, "ratings_converted": 0}

        # 1. Replace placeholder values
        if replace_placeholders:
            for val in null_values:
                mask = df[col] == val
                count = mask.sum()
                if count > 0:
                    df.loc[mask, col] = pd.NA
                    col_changes["placeholders_replaced"] += int(count)
                    total_placeholders_replaced += int(count)

        # 2. Handle rating anomalies (text in rating columns)
        if handle_rating_anomalies:
            # Check if this column matches a rating pattern
            is_rating_col = False
            for pattern in rating_column_patterns:
                if _safe_regex_match(pattern, col):
                    is_rating_col = True
                    break

            if is_rating_col:
                # Convert text values to ratings
                # Use LLM when enabled for domain-agnostic conversion
                if USE_LLM_RATING:
                    converted = _convert_ratings_with_llm(df, col, tool_context)
                    col_changes["ratings_converted"] += converted
                    total_ratings_converted += converted
                else:
                    # Fallback to pattern-based conversion
                    for text_val, numeric_val in TEXT_TO_RATING_MAP.items():
                        mask = df[col].astype(str).str.strip() == text_val
                        count = mask.sum()
                        if count > 0:
                            df.loc[mask, col] = numeric_val
                            col_changes["ratings_converted"] += int(count)
                            total_ratings_converted += int(count)

        # 3. Strip whitespace from string values
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

        if col_changes["placeholders_replaced"] > 0 or col_changes["ratings_converted"] > 0:
            changes_made[col] = col_changes

    # Save cleaned file
    from .task_manager import get_task_config, save_checkpoint

    import_dir = _get_import_dir()
    if import_dir:
        original_path = import_dir / target_file
    else:
        original_path = Path(target_file)

    stem = original_path.stem
    suffix = original_path.suffix
    # Append _schema_cleaned to filename
    cleaned_filename = f"{stem}_schema_cleaned{suffix}"

    # Determine output directory (use task's clean_dir if available)
    task_config = get_task_config(tool_context)
    if task_config:
        cleaned_path = Path(task_config["clean_dir"]) / cleaned_filename
    elif import_dir:
        cleaned_path = import_dir / cleaned_filename
    else:
        cleaned_path = Path(cleaned_filename)

    # Save
    if suffix.lower() == '.csv':
        df.to_csv(cleaned_path, index=False, encoding='utf-8')
    else:
        df.to_excel(cleaned_path, index=False)

    # Store cleaned file path in state for extraction to use
    # If using task directory, store relative path within task
    if task_config:
        set_state_value(tool_context, "schema_cleaned_file", str(cleaned_path))
    else:
        set_state_value(tool_context, "schema_cleaned_file", cleaned_filename)

    # Save checkpoint after cleaning
    save_checkpoint(tool_context)

    logger.info(
        f"Schema-based cleaning complete: {total_placeholders_replaced} placeholders replaced, "
        f"{total_ratings_converted} ratings converted"
    )

    return tool_success("schema_cleaning_complete", {
        "original_file": target_file,
        "cleaned_file": str(cleaned_path),
        "columns_cleaned": list(schema_columns),
        "changes_made": changes_made,
        "summary": {
            "total_placeholders_replaced": total_placeholders_replaced,
            "total_ratings_converted": total_ratings_converted,
            "columns_affected": len(changes_made),
        }
    })


def request_schema_revision(
    reason: str,
    suggested_changes: List[str],
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Request to roll back to the schema design phase for revisions.

    Use this tool when data extraction encounters issues that indicate
    the schema design needs to be revised. Common scenarios include:
    - Schema expects a column that doesn't exist in the data
    - Extraction hints don't match actual data patterns
    - Discovered entity types not represented in the schema
    - Relationship structure doesn't match data relationships

    This will trigger a pipeline rollback to the SCHEMA_DESIGN phase,
    allowing the user and schema design agent to revise the schema.

    Args:
        reason: Brief explanation of why schema revision is needed
        suggested_changes: List of specific changes to consider
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary confirming the rollback request
    """
    # Set rollback flags in state
    set_state_value(tool_context, NEEDS_SCHEMA_REVISION, True)
    set_state_value(tool_context, SCHEMA_REVISION_REASON, {
        "reason": reason,
        "suggested_changes": suggested_changes,
        "requested_at": str(pd.Timestamp.now()),
    })

    # Clear approved schema to force re-entry into schema design
    # Note: The pipeline service will handle clearing APPROVED_TARGET_SCHEMA_KEY
    # when it detects NEEDS_SCHEMA_REVISION flag

    logger.info(f"Schema revision requested: {reason}")

    return tool_success("schema_revision_requested", {
        "rollback_to": "SCHEMA_DESIGN",
        "reason": reason,
        "suggested_changes": suggested_changes,
        "message": (
            "Schema revision has been requested. The pipeline will return "
            "to the schema design phase after this preprocessing iteration. "
            "Please inform the user about the issues found and suggested changes."
        ),
    })


# =============================================================================
# TEXT EXTRACTION TOOLS (for text feedback columns)
# =============================================================================

# LLM model for text extraction
TEXT_EXTRACTION_MODEL = "qwen-plus-latest"


def _get_text_llm_client():
    """Get OpenAI client configured for DashScope."""
    from openai import OpenAI
    from ..config import get_config
    config = get_config()
    return OpenAI(
        api_key=config.llm.api_key,
        base_url=config.llm.api_base
    )


def extract_entities_from_text_column(
    file_path: str,
    node_label: str,
    column_name: str,
    tool_context: ToolContext,
    batch_size: int = 10,
) -> Dict[str, Any]:
    """
    Extract entity instances from a text feedback column using LLM.

    For text columns marked with source_type="text_extraction", this function:
    1. Reads text content from the specified column
    2. Uses LLM to extract entity instances from the text
    3. Stores extracted entities for relationship extraction

    Args:
        file_path: Path to the data file
        node_label: Label of the entity type to extract (e.g., "Feature")
        column_name: Name of the text column to extract from
        tool_context: ADK ToolContext
        batch_size: Number of rows to process per LLM call

    Returns:
        Dictionary with extracted entities
    """
    import json

    # Check if already completed (for checkpoint/resume support)
    file_name = Path(file_path).name if file_path else "unknown"
    progress_key = f"entity:{file_name}:{column_name}:{node_label}"
    progress = get_state_value(tool_context, EXTRACTION_PROGRESS, {})

    if progress.get(progress_key) == "completed":
        logger.info(f"Skipping already completed extraction: {progress_key}")
        return tool_success("extraction_skipped", {
            "node_label": node_label,
            "column_name": column_name,
            "reason": "Already extracted in previous run",
            "progress_key": progress_key,
        })

    schema = _get_approved_schema(tool_context)
    if not schema:
        return tool_error("No approved target schema found.")

    node = schema.get_node(node_label)
    if not node:
        return tool_error(f"Node type '{node_label}' not found in schema.")

    # Load data
    df = _load_data_file(file_path)
    if df is None:
        return tool_error(f"Could not load file: {file_path}")

    if column_name not in df.columns:
        # Try pattern match
        matched = _find_column_by_pattern(df, column_name)
        if matched:
            column_name = matched
        else:
            return tool_error(f"Column '{column_name}' not found.")

    # Filter valid text rows
    placeholder_values = ['nan', 'none', '无', '-', '暂无', '未知', '不清楚', '']
    valid_rows = df[df[column_name].notna()].copy()
    valid_rows = valid_rows[
        ~valid_rows[column_name].astype(str).str.lower().isin(placeholder_values) &
        (valid_rows[column_name].astype(str).str.strip() != '')
    ]

    if len(valid_rows) == 0:
        return tool_success("extracted_entities", {
            "node_label": node_label,
            "column_name": column_name,
            "count": 0,
            "message": "No valid text content found in column",
        })

    # Get hints for entity type
    hints = node.extraction_hints
    entity_description = hints.get("entity_type", node_label)

    client = _get_text_llm_client()

    entity_to_row_map: Dict[str, List[str]] = {}  # Track which row each entity came from

    # Process in batches
    for batch_start in range(0, len(valid_rows), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_rows))
        batch_df = valid_rows.iloc[batch_start:batch_end]

        # Prepare texts for LLM
        texts_with_ids = []
        for idx, row in batch_df.iterrows():
            # Try to get respondent ID from first column or 'respondent_id'
            row_id = row.get('respondent_id', row.get(df.columns[0], idx))
            text = str(row[column_name])[:500]  # Truncate long texts
            texts_with_ids.append({"row_id": str(row_id), "text": text})

        prompt = f"""从以下文本中提取所有 {entity_description} 实体。

文本列表：
{json.dumps(texts_with_ids, ensure_ascii=False, indent=2)}

请提取每条文本中提到的 {entity_description} 实体，返回JSON格式：
{{
    "extractions": [
        {{
            "row_id": "1",
            "entities": ["实体1", "实体2"]
        }}
    ]
}}

提取要求：
1. 提取具体的、有意义的实体名称
2. 去除修饰词，保留核心名称
3. 每个实体应该是独立的概念
4. 如果文本中没有相关实体，entities为空数组"""

        try:
            response = client.chat.completions.create(
                model=TEXT_EXTRACTION_MODEL,
                messages=[
                    {"role": "system", "content": "你是知识图谱实体提取专家。请准确提取文本中的实体。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            result_text = (response.choices[0].message.content or "").strip()

            # Parse JSON response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)
            extractions = result.get("extractions", [])

            for extraction in extractions:
                row_id = extraction.get("row_id", "")
                entities = extraction.get("entities", [])
                for entity_name in entities:
                    entity_name = str(entity_name).strip()
                    if entity_name and entity_name not in entity_to_row_map:
                        entity_to_row_map[entity_name] = []
                    if entity_name:
                        entity_to_row_map[entity_name].append(row_id)

        except Exception as e:
            logger.warning(f"Error extracting entities from batch: {e}")
            continue

    # Build entity list with IDs
    entities = []
    value_to_id = {}

    # Get existing entity count to ensure globally unique IDs
    results = get_state_value(tool_context, TARGETED_EXTRACTION_RESULTS, {})
    existing_entities = results.get(node_label, [])
    existing_count = len(existing_entities)

    for i, (entity_name, row_ids) in enumerate(entity_to_row_map.items()):
        entity_id = f"{node_label}_{existing_count + i}"
        entity = {
            node.unique_property: entity_id,
            "name": entity_name,
            "source_column": column_name,
            "mention_count": len(row_ids),
        }
        entities.append(entity)
        value_to_id[entity_name] = entity_id

    # Store in state (reuse results from above)
    existing_entities.extend(entities)
    results[node_label] = existing_entities
    set_state_value(tool_context, TARGETED_EXTRACTION_RESULTS, results)

    entity_maps = get_state_value(tool_context, TARGETED_ENTITY_MAPS, {})
    existing_map = entity_maps.get(node_label, {})
    existing_map.update(value_to_id)
    entity_maps[node_label] = existing_map
    set_state_value(tool_context, TARGETED_ENTITY_MAPS, entity_maps)

    # Store row mapping for relationship extraction
    text_extraction_map_key = f"text_extraction_map_{node_label}_{column_name}"
    set_state_value(tool_context, text_extraction_map_key, entity_to_row_map)

    logger.info(f"Extracted {len(entities)} {node_label} entities from text column '{column_name}'")

    # Mark as completed for checkpoint/resume support
    progress[progress_key] = "completed"
    set_state_value(tool_context, EXTRACTION_PROGRESS, progress)

    return tool_success("extracted_entities", {
        "node_label": node_label,
        "column_name": column_name,
        "count": len(entities),
        "source_type": "text_extraction",
        "sample": entities[:5] if entities else [],
    })


def extract_relationships_from_text_column(
    file_path: str,
    relationship_type: str,
    column_name: str,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Extract relationship instances from a text feedback column.

    Creates relationships between Respondent and extracted entities
    based on the text extraction results.

    Args:
        file_path: Path to the data file
        relationship_type: Type of relationship to create
        column_name: Name of the text column
        tool_context: ADK ToolContext

    Returns:
        Dictionary with extracted relationships
    """
    # Check if already completed (for checkpoint/resume support)
    file_name = Path(file_path).name if file_path else "unknown"
    progress_key = f"rel:{file_name}:{column_name}:{relationship_type}"
    progress = get_state_value(tool_context, EXTRACTION_PROGRESS, {})

    if progress.get(progress_key) == "completed":
        logger.info(f"Skipping already completed extraction: {progress_key}")
        return tool_success("extraction_skipped", {
            "relationship_type": relationship_type,
            "column_name": column_name,
            "reason": "Already extracted in previous run",
            "progress_key": progress_key,
        })

    schema = _get_approved_schema(tool_context)
    if not schema:
        return tool_error("No approved target schema found.")

    rel = schema.get_relationship(relationship_type)
    if not rel:
        return tool_error(f"Relationship '{relationship_type}' not found in schema.")

    # Load data
    df = _load_data_file(file_path)
    if df is None:
        return tool_error(f"Could not load file: {file_path}")

    # Get entity maps
    entity_maps = get_state_value(tool_context, TARGETED_ENTITY_MAPS, {})
    from_map = entity_maps.get(rel.from_node, {})
    to_map = entity_maps.get(rel.to_node, {})

    if not to_map:
        return tool_error(
            f"No entities found for target node '{rel.to_node}'. "
            f"Run extract_entities_from_text_column first."
        )

    # Get text extraction map
    text_extraction_map_key = f"text_extraction_map_{rel.to_node}_{column_name}"
    entity_to_row_map = get_state_value(tool_context, text_extraction_map_key, {})

    if not entity_to_row_map:
        return tool_error(
            f"No text extraction map found for {rel.to_node} from {column_name}. "
            f"Run extract_entities_from_text_column first."
        )

    # Get hints
    hints = rel.extraction_hints
    sentiment = hints.get("sentiment", "neutral")

    # Build relationships
    relationships = []

    # Get respondent column
    respondent_col = hints.get("respondent_column", df.columns[0])
    if respondent_col not in df.columns:
        # Try pattern match
        matched = _find_column_by_pattern(df, respondent_col)
        if matched:
            respondent_col = matched
        else:
            respondent_col = df.columns[0]

    # For each entity, create relationships with the rows that mentioned it
    for entity_name, row_ids in entity_to_row_map.items():
        to_id = to_map.get(entity_name)
        if not to_id:
            continue

        for row_id in row_ids:
            # Find from_id (Respondent)
            from_id = from_map.get(str(row_id))
            if not from_id:
                from_id = f"{rel.from_node}_{row_id}"
                from_map[str(row_id)] = from_id

            rel_record = {
                "from_id": from_id,
                "to_id": to_id,
                "to_name": entity_name,
                "relationship_type": relationship_type,
                "sentiment": sentiment,
                "source_column": column_name,
            }

            # Add any additional properties
            for prop in rel.properties:
                if prop == "sentiment":
                    rel_record["sentiment"] = sentiment
                elif prop == "source_column":
                    rel_record["source_column"] = column_name

            relationships.append(rel_record)

    # Store in state
    rel_data = get_state_value(tool_context, TARGETED_RELATIONSHIP_DATA, {})
    existing = rel_data.get(relationship_type, [])
    existing.extend(relationships)
    rel_data[relationship_type] = existing
    set_state_value(tool_context, TARGETED_RELATIONSHIP_DATA, rel_data)

    # Update entity maps
    entity_maps[rel.from_node] = from_map
    set_state_value(tool_context, TARGETED_ENTITY_MAPS, entity_maps)

    logger.info(f"Extracted {len(relationships)} relationships of type '{relationship_type}' from text")

    # Mark as completed for checkpoint/resume support
    progress[progress_key] = "completed"
    set_state_value(tool_context, EXTRACTION_PROGRESS, progress)

    return tool_success("extracted_relationships", {
        "relationship_type": relationship_type,
        "from_node": rel.from_node,
        "to_node": rel.to_node,
        "count": len(relationships),
        "source_type": "text_extraction",
        "sentiment": sentiment,
        "sample": relationships[:5] if relationships else [],
    })


def deduplicate_entities_with_llm(
    node_label: str,
    tool_context: ToolContext,
    similarity_threshold: float = 0.8,  # Reserved for future use
) -> Dict[str, Any]:
    """
    Use LLM to deduplicate similar entities within a node type.

    Merges semantically similar entities (e.g., "车身线条流畅" and "线条流畅")
    into canonical forms.

    Args:
        node_label: Label of the node type to deduplicate
        tool_context: ADK ToolContext
        similarity_threshold: Reserved for future use (currently LLM decides)

    Returns:
        Dictionary with deduplication results
    """
    import json
    _ = similarity_threshold  # Reserved for future use

    results = get_state_value(tool_context, TARGETED_EXTRACTION_RESULTS, {})
    entities = results.get(node_label, [])

    if not entities or len(entities) < 2:
        return tool_success("deduplication", {
            "node_label": node_label,
            "original_count": len(entities),
            "deduplicated_count": len(entities),
            "merged_groups": 0,
            "message": "Not enough entities to deduplicate",
        })

    # Get entity names
    entity_names = [e.get("name", "") for e in entities if e.get("name")]

    if len(entity_names) > 100:
        # Too many entities, sample
        entity_names = entity_names[:100]

    client = _get_text_llm_client()

    prompt = f"""分析以下 {node_label} 实体列表，找出语义相似的实体并合并：

实体列表：
{json.dumps(entity_names, ensure_ascii=False)}

请识别相似的实体并返回合并方案，JSON格式：
{{
    "merged_groups": [
        {{
            "canonical": "规范名称",
            "aliases": ["别名1", "别名2"]
        }}
    ],
    "kept_as_is": ["保持不变的实体1", "保持不变的实体2"]
}}

合并规则：
1. 仅合并语义完全相同或高度相似的实体
2. 选择最简洁清晰的名称作为 canonical
3. 不确定时保持不变"""

    try:
        response = client.chat.completions.create(
            model=TEXT_EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": "你是实体去重专家。请准确识别相似实体。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )

        result_text = (response.choices[0].message.content or "").strip()

        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        result = json.loads(result_text)
        merged_groups = result.get("merged_groups", [])

        if not merged_groups:
            return tool_success("deduplication", {
                "node_label": node_label,
                "original_count": len(entities),
                "deduplicated_count": len(entities),
                "merged_groups": 0,
                "message": "No similar entities found to merge",
            })

        # Build alias to canonical mapping
        alias_to_canonical = {}
        for group in merged_groups:
            canonical = group.get("canonical", "")
            aliases = group.get("aliases", [])
            for alias in aliases:
                alias_to_canonical[alias] = canonical

        # Update entities
        new_entities = []
        seen_names = set()
        entity_maps = get_state_value(tool_context, TARGETED_ENTITY_MAPS, {})
        value_to_id = entity_maps.get(node_label, {})

        for entity in entities:
            name = entity.get("name", "")

            # Check if this name should be merged
            if name in alias_to_canonical:
                canonical = alias_to_canonical[name]
                # Update the entity ID mapping to point to canonical
                canonical_id = value_to_id.get(canonical)
                if canonical_id:
                    value_to_id[name] = canonical_id
                continue  # Skip this entity, it's an alias

            if name in seen_names:
                continue

            seen_names.add(name)
            new_entities.append(entity)

        # Update state
        results[node_label] = new_entities
        set_state_value(tool_context, TARGETED_EXTRACTION_RESULTS, results)

        entity_maps[node_label] = value_to_id
        set_state_value(tool_context, TARGETED_ENTITY_MAPS, entity_maps)

        merged_count = len(entities) - len(new_entities)
        logger.info(f"Deduplicated {node_label}: {len(entities)} -> {len(new_entities)} ({merged_count} merged)")

        return tool_success("deduplication", {
            "node_label": node_label,
            "original_count": len(entities),
            "deduplicated_count": len(new_entities),
            "merged_groups": len(merged_groups),
            "merged_count": merged_count,
            "sample_merges": merged_groups[:3],
        })

    except Exception as e:
        logger.error(f"Error in LLM deduplication: {e}")
        return tool_error(f"Error during deduplication: {e}")


# Export tools list for agent registration
TARGETED_PREPROCESSING_TOOLS = [
    get_schema_extraction_plan,
    clean_columns_for_schema,  # NEW: Schema-driven data cleaning
    extract_entities_for_node,
    extract_relationship_data,
    # Text extraction tools
    extract_entities_from_text_column,
    extract_relationships_from_text_column,
    deduplicate_entities_with_llm,
    # Output tools
    save_extracted_data,
    get_extraction_summary,
    complete_targeted_preprocessing,
    generate_construction_rules,
    get_preprocessing_feedback,
    request_schema_revision,
]

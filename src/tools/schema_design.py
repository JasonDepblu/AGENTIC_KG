"""
Schema Design Tools for Agentic KG.

Tools for designing target knowledge graph schemas based on raw data analysis.
These tools are used in the SCHEMA_DESIGN phase of the Schema-First pipeline.
"""

import re
import json
import logging
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from google.adk.tools import ToolContext

# Set up logging for schema design tools
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Ensure we have a handler
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from .common import tool_success, tool_error, validate_file_path
from ..config import get_neo4j_import_dir
from ..models.target_schema import (
    EntityType,
    NodeDefinition,
    RelationshipDefinition,
    TargetSchema,
    TARGET_SCHEMA_KEY,
    APPROVED_TARGET_SCHEMA_KEY,
    DETECTED_ENTITIES_KEY,
)
# Import LLM detection tools for domain-agnostic entity detection
from .llm_detection import (
    detect_entities_from_columns as llm_detect_entities,
    detect_text_feedback_columns as llm_detect_text_feedback,
)

# Configuration flag for detection method
# Set to True to use LLM-based detection (domain-agnostic)
# Set to False to use pattern-based detection (faster but domain-specific)
USE_LLM_DETECTION = True


def _robust_json_parse(text: str, default_key: str = "entity_types") -> Dict[str, Any]:
    """
    Robustly parse JSON from LLM response, handling common issues:
    - Extract JSON from markdown code blocks
    - Handle truncated JSON by attempting to repair
    - Return a default structure on failure

    Args:
        text: Raw text from LLM response
        default_key: Key expected in the result (used for fallback)

    Returns:
        Parsed JSON dict, or empty default structure on failure
    """
    if not text:
        return {default_key: [], "analysis_notes": "Empty response"}

    # Step 1: Extract JSON from markdown code blocks
    result_text = text.strip()
    if "```json" in result_text:
        result_text = result_text.split("```json")[1].split("```")[0].strip()
    elif "```" in result_text:
        parts = result_text.split("```")
        if len(parts) >= 2:
            result_text = parts[1].strip()

    # Step 2: Try direct parsing
    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        pass

    # Step 3: Try to find and extract JSON object
    # Look for { ... } pattern
    start_idx = result_text.find('{')
    if start_idx == -1:
        logger.warning("No JSON object found in response")
        return {default_key: [], "analysis_notes": "No valid JSON found"}

    # Find matching closing brace
    brace_count = 0
    end_idx = start_idx
    for i, char in enumerate(result_text[start_idx:], start_idx):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break

    if brace_count == 0:
        # Found complete JSON object
        try:
            return json.loads(result_text[start_idx:end_idx + 1])
        except json.JSONDecodeError:
            pass

    # Step 4: Try to repair truncated JSON
    # Common issue: JSON is cut off at max_tokens
    json_fragment = result_text[start_idx:]

    # Count unclosed brackets and braces
    open_braces = json_fragment.count('{') - json_fragment.count('}')
    open_brackets = json_fragment.count('[') - json_fragment.count(']')

    # Try to close them
    repaired = json_fragment
    if open_brackets > 0 or open_braces > 0:
        # Remove any trailing incomplete string/value
        # Look for last complete value
        for end_pattern in [',', ':', '"', ']', '}']:
            last_pos = repaired.rfind(end_pattern)
            if last_pos > len(repaired) // 2:  # Only truncate from latter half
                test_json = repaired[:last_pos + 1]
                # Add closing brackets/braces
                test_json += ']' * open_brackets + '}' * open_braces
                try:
                    result = json.loads(test_json)
                    logger.info(f"Successfully repaired truncated JSON")
                    return result
                except json.JSONDecodeError:
                    continue

    logger.warning(f"Failed to parse or repair JSON response")
    return {default_key: [], "analysis_notes": "JSON parsing failed"}


# Chinese patterns for entity detection (DEPRECATED - use LLM detection instead)
# NOTE: These patterns are kept for backward compatibility only.
# When USE_LLM_DETECTION = True, LLM-based detection is used instead.
# Patterns are matched against column names with trailing punctuation removed
ENTITY_PATTERNS = {
    EntityType.BRAND: [
        r"品牌是?[\?？]?$",      # Ends with "品牌" optionally followed by "是" and "?"
        r"brand",
        r"品牌名",
        r"的品牌",               # Contains "的品牌"
    ],
    EntityType.MODEL: [
        r"车型.*配置是?[\?？]?$",  # Ends with "配置" optionally followed by "是" and "?"
        r"model",
        r"车型名",
        r"型号",
        r"车型及配置",           # Contains "车型及配置"
    ],
    EntityType.STORE: [
        r"门店是?[\?？]?$",       # Ends with "门店" optionally followed by "是" and "?"
        r"store",
        r"店名",
        r"经销商",
        r"到访的门店",           # Contains "到访的门店"
    ],
    EntityType.RESPONDENT: [
        r"^序号$",               # Exact match for "序号"
        r"respondent",
        r"受访者",
        r"^id$",                 # Exact match for "id"
    ],
    EntityType.ASPECT: [
        r"方面.*打多少分",
        r"评分",
        r"rating",
        r"打分",
    ],
}


def _get_import_dir() -> Optional[Path]:
    """Get the import directory as a Path object."""
    import_dir_path = get_neo4j_import_dir()
    if not import_dir_path:
        return None
    return Path(import_dir_path)


def _get_or_create_schema(tool_context: ToolContext) -> TargetSchema:
    """Get existing schema from state or create a new one."""
    if TARGET_SCHEMA_KEY in tool_context.state:
        schema_data = tool_context.state[TARGET_SCHEMA_KEY]
        if isinstance(schema_data, dict):
            return TargetSchema.from_dict(schema_data)
        return schema_data

    # Create new schema
    schema = TargetSchema(
        name="Target Schema",
        description="Knowledge graph schema designed from raw data"
    )
    tool_context.state[TARGET_SCHEMA_KEY] = schema.to_dict()
    return schema


def _save_schema(tool_context: ToolContext, schema: TargetSchema) -> None:
    """Save schema to state."""
    tool_context.state[TARGET_SCHEMA_KEY] = schema.to_dict()


def _resolve_pattern(
    old_pattern: str,
    rename_map: Dict[str, str],
    new_columns: List[str]
) -> Tuple[str, bool, Optional[str]]:
    """
    Resolve and update column_pattern after column standardization.

    Handles three cases:
    1. Exact match: old_pattern is an original column name -> replace with new name
    2. Valid regex: old_pattern is a regex that still matches new columns -> keep as-is
    3. Partial match: old_pattern contains an original column name -> replace that part

    Args:
        old_pattern: The original column_pattern from extraction_hints
        rename_map: {original_column_name: new_column_name} mapping
        new_columns: List of new standardized column names

    Returns:
        Tuple of (new_pattern, was_updated, warning_message)
        - new_pattern: The resolved pattern (may be unchanged)
        - was_updated: True if pattern was modified
        - warning_message: Warning if pattern won't match any column (None otherwise)
    """
    # Case 1: Exact match - old_pattern is an original column name
    if old_pattern in rename_map:
        return rename_map[old_pattern], True, None

    # Case 2: Check if it's a valid regex that matches new column names
    try:
        regex = re.compile(old_pattern)
        matches = [col for col in new_columns if regex.search(col)]
        if matches:
            # Regex is still valid and matches new columns, keep it
            logger.debug(f"Pattern '{old_pattern}' matches new columns: {matches[:3]}...")
            return old_pattern, False, None
    except re.error:
        pass  # Not a valid regex, continue to other cases

    # Case 3: Partial match - original column name is contained in pattern
    for old_col, new_col in rename_map.items():
        if old_col in old_pattern:
            new_pattern = old_pattern.replace(old_col, new_col)
            logger.debug(f"Partial match: replaced '{old_col}' with '{new_col}' in pattern")
            return new_pattern, True, None

    # Pattern doesn't match anything - generate warning
    warning = f"Pattern '{old_pattern}' won't match any standardized column"
    logger.warning(f"{warning}. Available columns: {new_columns[:5]}...")
    return old_pattern, False, warning


def _sync_schema_column_patterns(
    tool_context: ToolContext,
    rename_map: Dict[str, str],
) -> Dict[str, Any]:
    """
    Sync Schema's extraction_hints to use standardized column names.

    After standardize_column_names() renames columns (e.g., '序号' -> 'respondent_id'),
    this function updates all extraction_hints.column_pattern values in the Schema
    to use the new column names.

    Handles three cases intelligently:
    1. Exact match: pattern is an original column name -> replace with new name
    2. Valid regex: pattern still matches new columns -> keep as-is
    3. Partial match: pattern contains original column name -> replace that part

    Args:
        tool_context: ADK ToolContext for state management
        rename_map: {original_column_name: new_column_name} mapping

    Returns:
        Dictionary with update details:
        - updated_nodes: list of updated node info
        - updated_relationships: list of updated relationship info
        - warnings: list of warning messages for patterns that won't match
    """
    schema = _get_or_create_schema(tool_context)
    if not schema:
        return {"updated_nodes": [], "updated_relationships": [], "warnings": []}

    new_column_names = list(rename_map.values())
    updated_nodes = []
    updated_relationships = []
    warnings = []

    # Update Node extraction_hints
    for label, node in schema.nodes.items():
        hints = node.extraction_hints
        if "column_pattern" in hints:
            old_pattern = hints["column_pattern"]
            new_pattern, was_updated, warning = _resolve_pattern(
                old_pattern, rename_map, new_column_names
            )

            if was_updated:
                hints["column_pattern"] = new_pattern
                updated_nodes.append({
                    "label": label,
                    "old_pattern": old_pattern,
                    "new_pattern": new_pattern,
                })
                logger.debug(f"Updated Node '{label}' column_pattern: '{old_pattern}' -> '{new_pattern}'")

            if warning:
                warnings.append(f"Node '{label}': {warning}")

    # Update Relationship extraction_hints
    for rel_type, rel in schema.relationships.items():
        hints = rel.extraction_hints
        if "column_pattern" in hints:
            old_pattern = hints["column_pattern"]
            new_pattern, was_updated, warning = _resolve_pattern(
                old_pattern, rename_map, new_column_names
            )

            if was_updated:
                hints["column_pattern"] = new_pattern
                updated_relationships.append({
                    "type": rel_type,
                    "old_pattern": old_pattern,
                    "new_pattern": new_pattern,
                })
                logger.debug(f"Updated Relationship '{rel_type}' column_pattern: '{old_pattern}' -> '{new_pattern}'")

            if warning:
                warnings.append(f"Relationship '{rel_type}': {warning}")

    # Save updated Schema to state
    if updated_nodes or updated_relationships:
        _save_schema(tool_context, schema)
        logger.info(f"Schema extraction_hints synced: {len(updated_nodes)} nodes, {len(updated_relationships)} relationships")

    if warnings:
        logger.warning(f"Schema sync warnings: {warnings}")

    return {
        "updated_nodes": updated_nodes,
        "updated_relationships": updated_relationships,
        "warnings": warnings,
    }


def sample_raw_file_structure(
    file_path: str,
    tool_context: ToolContext,
    sample_rows: int = 10
) -> Dict[str, Any]:
    """
    Analyze raw file structure to help design the target schema.

    Returns column names, inferred data types, and sample values for each column.
    This helps identify potential entity types and relationships.

    Args:
        file_path: File to analyze, relative to the import directory
        tool_context: ADK ToolContext for state management
        sample_rows: Number of sample rows to include (default: 10)

    Returns:
        Dictionary with file structure analysis:
        - columns: List of column info with name, dtype, sample_values, unique_count
        - row_count: Total number of rows
        - suggested_entities: Auto-detected entity type suggestions
    """
    # Validate file path
    validation_error = validate_file_path(file_path)
    if validation_error:
        return validation_error

    import_dir = _get_import_dir()
    if not import_dir:
        return tool_error("NEO4J_IMPORT_DIR not configured.")

    full_path = import_dir / file_path

    if not full_path.exists():
        return tool_error(f"File does not exist: {file_path}")

    try:
        # Read file based on extension
        suffix = full_path.suffix.lower()
        if suffix == '.csv':
            df = pd.read_csv(full_path)
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(full_path)
        else:
            return tool_error(f"Unsupported file type: {suffix}. Use .csv or .xlsx")

        # Analyze columns - normalize column names to avoid whitespace issues
        columns_info = []
        for col in df.columns:
            # Normalize column name: strip whitespace and fix duplicate number prefixes
            col_normalized = col.strip() if isinstance(col, str) else str(col)
            # Fix duplicate number prefixes like "10、10 xxx" -> "10、xxx"
            col_normalized = re.sub(r'^(\d+[、\.])\1+', r'\1', col_normalized)

            col_data = df[col]
            sample_values = col_data.dropna().head(sample_rows).tolist()

            # Convert to string for display
            sample_values = [str(v)[:100] for v in sample_values]  # Truncate long values

            columns_info.append({
                "name": col_normalized,  # Use normalized column name
                "dtype": str(col_data.dtype),
                "sample_values": sample_values,
                "unique_count": int(col_data.nunique()),  # Convert numpy.int64 to Python int
                "null_count": int(col_data.isnull().sum()),  # Convert numpy.int64 to Python int
            })

        # Auto-detect entity types based on normalized column names
        # Uses LLM detection when USE_LLM_DETECTION=True for domain-agnostic detection
        normalized_column_names = [info["name"] for info in columns_info]
        suggested_entities = detect_entities_smart(
            file_path=file_path,
            column_names=normalized_column_names,
            tool_context=tool_context,
        )

        # Calculate suggested progress total for the agent
        entity_cols = len([c for c in columns_info if c["unique_count"] < 50 and c["unique_count"] > 1])
        score_cols = len([c for c in columns_info if "score" in c["name"].lower() or "分" in c["name"]])
        text_cols = len([c for c in columns_info if c["dtype"] == "object" and c["unique_count"] > 50])
        suggested_total = entity_cols + score_cols + min(text_cols, 4) + 5  # Limit text cols to 2 pairs

        result = {
            "file_path": file_path,
            "row_count": int(len(df)),  # Convert to Python int
            "column_count": int(len(df.columns)),  # Convert to Python int
            "columns": columns_info,
            "suggested_entities": suggested_entities,
            # Progress tracking hint
            "NEXT_ACTION": f"IMMEDIATELY call set_progress_total(total={suggested_total}, description='Designing schema')",
            "suggested_progress_total": suggested_total,
        }

        return tool_success("file_structure", result)

    except Exception as e:
        return tool_error(f"Error analyzing file {file_path}: {e}")


def _detect_entity_types(column_names: List[str]) -> List[Dict[str, Any]]:
    """
    Detect potential entity types from column names using pattern matching.

    DEPRECATED: This function uses hardcoded patterns and is domain-specific.
    For domain-agnostic detection, use _detect_entity_types_with_llm() instead.
    """
    detected = []

    for col in column_names:
        # Normalize column name for robust matching
        col_normalized = col.strip() if isinstance(col, str) else str(col)
        col_lower = col_normalized.lower()
        for entity_type, patterns in ENTITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, col_lower, re.IGNORECASE):
                    detected.append({
                        "column": col_normalized,  # Use normalized column name
                        "entity_type": entity_type.value,
                        "pattern_matched": pattern,
                        "confidence": "high" if len(pattern) > 5 else "medium",
                    })
                    break
            else:
                continue
            break

    return detected


def _detect_entity_types_with_llm(
    file_path: str,
    tool_context: ToolContext
) -> List[Dict[str, Any]]:
    """
    Detect potential entity types using LLM analysis.

    This is the domain-agnostic alternative to _detect_entity_types().
    It uses LLM to analyze column names and data samples to infer entity types.

    Args:
        file_path: Path to the CSV file (relative to import dir)
        tool_context: ADK ToolContext for state management

    Returns:
        List of detected entity suggestions compatible with _detect_entity_types format
    """
    logger.info(f"Using LLM-based entity detection for: {file_path}")

    try:
        # Call LLM detection tool
        result = llm_detect_entities(file_path, tool_context)

        if result.get("status") == "error":
            logger.warning(f"LLM detection failed: {result.get('error')}")
            # Fallback to empty list
            return []

        # Convert LLM detection format to expected format
        detected = []
        details = result.get("details", [])

        for entity_info in details:
            # Map LLM column_type to entity_type format
            column_type = entity_info.get("column_type", "")
            suggested_entity = entity_info.get("suggested_entity_name", "")

            # Only include actual entity columns, not metadata
            if column_type in ["IDENTIFIER", "CATEGORICAL_ENTITY", "NUMERIC_RATING"]:
                detected.append({
                    "column": entity_info.get("column", ""),
                    "entity_type": suggested_entity or column_type,
                    "pattern_matched": "llm_detection",  # Mark as LLM detected
                    "confidence": "high",  # LLM provides semantic understanding
                    "column_type": column_type,
                    "reason": entity_info.get("reason", ""),
                    "is_key_entity": entity_info.get("is_key_entity", False),
                })

        logger.info(f"LLM detected {len(detected)} entities")
        return detected

    except Exception as e:
        logger.error(f"Error in LLM entity detection: {e}")
        return []


def detect_entities_smart(
    file_path: str,
    column_names: List[str],
    tool_context: ToolContext,
    use_llm: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """
    Smart entity detection that can use either LLM or pattern-based detection.

    Args:
        file_path: Path to the data file
        column_names: List of column names (for pattern-based fallback)
        tool_context: ADK ToolContext
        use_llm: Force LLM (True) or patterns (False). None uses global USE_LLM_DETECTION.

    Returns:
        List of detected entity suggestions
    """
    should_use_llm = use_llm if use_llm is not None else USE_LLM_DETECTION

    if should_use_llm:
        # Try LLM-based detection
        result = _detect_entity_types_with_llm(file_path, tool_context)
        if result:
            return result
        # Fallback to pattern-based if LLM fails
        logger.info("LLM detection returned empty, falling back to pattern-based")

    # Use pattern-based detection
    return _detect_entity_types(column_names)


def detect_potential_entities(
    file_path: str,
    tool_context: ToolContext,
    use_llm: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Auto-detect potential entity types in a raw data file.

    Uses column name patterns and value analysis to suggest entity types
    that could be included in the target schema.

    When USE_LLM_DETECTION=True (default), uses LLM for domain-agnostic detection.
    Otherwise, falls back to hardcoded patterns (domain-specific).

    Args:
        file_path: File to analyze, relative to the import directory
        tool_context: ADK ToolContext for state management
        use_llm: Override for LLM detection (None uses global USE_LLM_DETECTION)

    Returns:
        Dictionary with detected entities:
        - entities: List of detected entity suggestions
        - relationships: Suggested relationships based on column patterns
    """
    should_use_llm = use_llm if use_llm is not None else USE_LLM_DETECTION

    # Validate file path
    validation_error = validate_file_path(file_path)
    if validation_error:
        return validation_error

    import_dir = _get_import_dir()
    if not import_dir:
        return tool_error("NEO4J_IMPORT_DIR not configured.")

    full_path = import_dir / file_path

    if not full_path.exists():
        return tool_error(f"File does not exist: {file_path}")

    try:
        # Read file
        suffix = full_path.suffix.lower()
        if suffix == '.csv':
            df = pd.read_csv(full_path)
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(full_path)
        else:
            return tool_error(f"Unsupported file type: {suffix}")

        entities = []
        relationships = []

        # Try LLM-based detection first when enabled
        if should_use_llm:
            logger.info(f"Using LLM-based entity detection for: {file_path}")
            llm_result = llm_detect_entities(file_path, tool_context)

            if llm_result.get("status") != "error":
                # Use LLM detection results
                for entity_info in llm_result.get("details", []):
                    column_type = entity_info.get("column_type", "")
                    col = entity_info.get("column", "")

                    # Map column to entity suggestion
                    if column_type in ["IDENTIFIER", "CATEGORICAL_ENTITY"]:
                        # Get sample values
                        if col in df.columns:
                            sample_values = df[col].dropna().head(5).tolist()
                            sample_values = [str(v)[:50] for v in sample_values]
                            is_unique = df[col].nunique() == len(df[col].dropna())
                            unique_count = int(df[col].nunique())
                        else:
                            sample_values = []
                            is_unique = False
                            unique_count = 0

                        entities.append({
                            "column": col,
                            "suggested_label": entity_info.get("suggested_entity_name", col),
                            "entity_type": entity_info.get("suggested_entity_name", column_type),
                            "is_unique_identifier": column_type == "IDENTIFIER" or is_unique,
                            "unique_count": unique_count,
                            "sample_values": sample_values,
                            "extraction_hint": f"Extract from column: {col}",
                            "detection_method": "llm",
                            "reason": entity_info.get("reason", ""),
                        })

                # Use LLM-detected relationships
                for rel_info in llm_result.get("suggested_relationships", []):
                    relationships.append({
                        "column": ", ".join(rel_info.get("source_columns", [])),
                        "suggested_type": rel_info.get("type", "RELATES_TO"),
                        "from_node": rel_info.get("from_node", ""),
                        "to_node": rel_info.get("to_node", ""),
                        "properties": [],
                        "description": rel_info.get("description", ""),
                        "detection_method": "llm",
                    })

                # Store in state with LLM detection metadata
                detection_result = {
                    "entities": entities,
                    "relationships": relationships,
                    "domain": llm_result.get("domain", "unknown"),
                    "domain_description": llm_result.get("domain_description", ""),
                    "detection_method": "llm",
                }
                tool_context.state[DETECTED_ENTITIES_KEY] = detection_result

                return tool_success("detected_entities", {
                    **detection_result,
                    "summary": {
                        "entity_count": len(entities),
                        "relationship_count": len(relationships),
                    }
                })

        # Fallback to pattern-based detection
        logger.info(f"Using pattern-based entity detection for: {file_path}")

        for col in df.columns:
            col_lower = col.lower()

            # Check each entity type pattern
            for entity_type, patterns in ENTITY_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, col_lower, re.IGNORECASE):
                        # Get sample values
                        sample_values = df[col].dropna().head(5).tolist()
                        sample_values = [str(v)[:50] for v in sample_values]

                        # Determine if this is a unique identifier
                        is_unique = df[col].nunique() == len(df[col].dropna())

                        entities.append({
                            "column": col,
                            "suggested_label": entity_type.value,
                            "entity_type": entity_type.value,
                            "is_unique_identifier": bool(is_unique),
                            "unique_count": int(df[col].nunique()),
                            "sample_values": sample_values,
                            "extraction_hint": f"Extract from column: {col}",
                            "detection_method": "pattern",
                        })
                        break
                else:
                    continue
                break

            # Detect rating columns -> potential relationships
            if re.search(r"打多少分|评分|rating", col_lower, re.IGNORECASE):
                # Extract aspect name from column
                aspect_match = re.search(r"[\"'""]([^\"'""]+)[\"'""]", col)
                aspect_name = aspect_match.group(1) if aspect_match else col

                relationships.append({
                    "column": col,
                    "suggested_type": "RATES",
                    "from_node": "Respondent",
                    "to_node": "Aspect",
                    "properties": ["score"],
                    "aspect_name": aspect_name,
                    "detection_method": "pattern",
                })

        # Store in state
        tool_context.state[DETECTED_ENTITIES_KEY] = {
            "entities": entities,
            "relationships": relationships,
            "detection_method": "pattern",
        }

        return tool_success("detected_entities", {
            "entities": entities,
            "relationships": relationships,
            "summary": {
                "entity_count": len(entities),
                "relationship_count": len(relationships),
            }
        })

    except Exception as e:
        return tool_error(f"Error detecting entities in {file_path}: {e}")


# Maximum number of node types allowed to prevent runaway proposals
MAX_NODE_TYPES = 10


def propose_node_type(
    label: str,
    unique_property: str,
    tool_context: ToolContext,
    properties: Optional[List[str]] = None,
    entity_type: Optional[str] = None,
    extraction_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Propose a node type for the target schema.

    Adds a node definition that will guide entity extraction during preprocessing.

    Args:
        label: Node label in the graph (e.g., "Brand", "Model")
        unique_property: Property that uniquely identifies nodes (e.g., "brand_id")
        tool_context: ADK ToolContext for state management
        properties: Optional list of additional property names
        entity_type: Optional entity type category (Brand, Model, Store, etc.)
        extraction_hints: Optional hints for data extraction (column patterns, etc.)

    Returns:
        Dictionary with status and the proposed node definition
    """
    # DEBUG: Log when this tool is actually called
    logger.info(f"=== propose_node_type CALLED ===")
    logger.info(f"  label: {label}")
    logger.info(f"  unique_property: {unique_property}")
    logger.info(f"  properties: {properties}")
    logger.info(f"  entity_type: {entity_type}")
    logger.info(f"  extraction_hints: {extraction_hints}")
    print(f"\n>>> TOOL CALLED: propose_node_type(label={label}, unique_property={unique_property})")

    if not label:
        return tool_error("Node label is required")

    if not unique_property:
        return tool_error("Unique property is required")

    # Get or create schema (check limit before creating)
    schema = _get_or_create_schema(tool_context)

    # Check maximum node limit to prevent runaway proposals
    if len(schema.nodes) >= MAX_NODE_TYPES:
        existing_labels = list(schema.nodes.keys())
        return tool_error(
            f"⚠️ MAXIMUM NODE TYPE LIMIT REACHED ({MAX_NODE_TYPES} types)! "
            f"You cannot add more node types. Current nodes: {existing_labels}. "
            f"STOP proposing new node types. For survey data, you typically need only: "
            f"Respondent, Brand, Model, Store, Aspect (5 node types). "
            f"Call 'get_target_schema' to review and proceed to proposing relationships."
        )

    # Determine entity type
    if entity_type:
        try:
            etype = EntityType(entity_type)
        except ValueError:
            etype = EntityType.CUSTOM
    else:
        etype = EntityType.CUSTOM

    # Create node definition
    node = NodeDefinition(
        label=label,
        unique_property=unique_property,
        properties=properties or [],
        entity_type=etype,
        extraction_hints=extraction_hints or {},
    )

    # Add to schema
    schema.add_node(node)
    _save_schema(tool_context, schema)

    return tool_success("proposed_node", {
        "label": label,
        "definition": node.to_dict(),
        "current_node_count": len(schema.nodes),
    })


# Maximum number of relationship types allowed to prevent runaway proposals
MAX_RELATIONSHIP_TYPES = 15


def propose_relationship_type(
    relationship_type: str,
    from_node: str,
    to_node: str,
    tool_context: ToolContext,
    properties: Optional[List[str]] = None,
    extraction_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Propose a relationship type for the target schema.

    Adds a relationship definition that will guide relationship extraction during preprocessing.

    Args:
        relationship_type: Type name for the relationship (e.g., "RATES", "BELONGS_TO")
        from_node: Label of the source node
        to_node: Label of the target node
        tool_context: ADK ToolContext for state management
        properties: Optional list of property names for this relationship
        extraction_hints: Optional hints for data extraction

    Returns:
        Dictionary with status and the proposed relationship definition
    """
    # DEBUG: Log when this tool is actually called
    logger.info(f"=== propose_relationship_type CALLED ===")
    logger.info(f"  relationship_type: {relationship_type}")
    logger.info(f"  from_node: {from_node}")
    logger.info(f"  to_node: {to_node}")
    logger.info(f"  properties: {properties}")
    logger.info(f"  extraction_hints: {extraction_hints}")
    print(f"\n>>> TOOL CALLED: propose_relationship_type(type={relationship_type}, from={from_node}, to={to_node})")

    if not relationship_type:
        return tool_error("Relationship type is required")

    if not from_node:
        return tool_error("Source node (from_node) is required")

    if not to_node:
        return tool_error("Target node (to_node) is required")

    # Get schema
    schema = _get_or_create_schema(tool_context)

    # Check maximum relationship limit to prevent runaway proposals
    if len(schema.relationships) >= MAX_RELATIONSHIP_TYPES:
        existing_types = list(schema.relationships.keys())
        return tool_error(
            f"⚠️ MAXIMUM RELATIONSHIP LIMIT REACHED ({MAX_RELATIONSHIP_TYPES} types)! "
            f"You cannot add more relationship types. Current relationships: {existing_types}. "
            f"STOP proposing new relationship types. Instead, use generic relationships like "
            f"'RATES' (with column_pattern to match multiple rating columns) rather than "
            f"creating separate relationship types for each column. "
            f"Call 'get_target_schema' to review the current schema and proceed to the next step."
        )

    # Check if a relationship with this type already exists
    existing = schema.relationships.get(relationship_type)
    if existing:
        if existing.from_node == from_node and existing.to_node == to_node:
            # Exact same relationship already exists - inform agent, no action needed
            return tool_success("proposed_relationship", {
                "status": "already_exists",
                "relationship_type": relationship_type,
                "message": (
                    f"Relationship '{relationship_type}' ({from_node}→{to_node}) already exists in the schema. "
                    f"No action needed. Do NOT try to add it again."
                ),
                "definition": existing.to_dict(),
                "current_relationship_count": len(schema.relationships),
            })
        else:
            # Same type but different endpoints - error with guidance
            return tool_error(
                f"Relationship type '{relationship_type}' already exists with different endpoints "
                f"({existing.from_node}→{existing.to_node}). "
                f"Each relationship type name must be UNIQUE. "
                f"Please use a different name for this relationship, e.g., "
                f"'{relationship_type}_{to_node.upper()}' or 'HAS_{to_node.upper()}'."
            )

    # Validate that referenced nodes exist (or warn)
    warnings = []
    if from_node not in schema.nodes:
        warnings.append(f"Source node '{from_node}' not yet defined in schema")
    if to_node not in schema.nodes:
        warnings.append(f"Target node '{to_node}' not yet defined in schema")

    # Create relationship definition
    rel = RelationshipDefinition(
        relationship_type=relationship_type,
        from_node=from_node,
        to_node=to_node,
        properties=properties or [],
        extraction_hints=extraction_hints or {},
    )

    # Add to schema
    schema.add_relationship(rel)
    _save_schema(tool_context, schema)

    result = {
        "relationship_type": relationship_type,
        "definition": rel.to_dict(),
        "current_relationship_count": len(schema.relationships),
    }

    if warnings:
        result["warnings"] = warnings

    return tool_success("proposed_relationship", result)


def remove_node_type(
    label: str,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Remove a node type from the target schema.

    Also removes any relationships that reference this node.

    Args:
        label: Node label to remove
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and removal result
    """
    schema = _get_or_create_schema(tool_context)

    if label not in schema.nodes:
        return tool_error(f"Node '{label}' not found in schema")

    removed_node = schema.remove_node(label)
    _save_schema(tool_context, schema)

    return tool_success("removed_node", {
        "label": label,
        "removed_definition": removed_node.to_dict() if removed_node else None,
        "remaining_node_count": len(schema.nodes),
    })


def remove_relationship_type(
    relationship_type: str,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Remove a relationship type from the target schema.

    Args:
        relationship_type: Relationship type to remove
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and removal result
    """
    schema = _get_or_create_schema(tool_context)

    if relationship_type not in schema.relationships:
        return tool_error(f"Relationship '{relationship_type}' not found in schema")

    removed_rel = schema.remove_relationship(relationship_type)
    _save_schema(tool_context, schema)

    return tool_success("removed_relationship", {
        "relationship_type": relationship_type,
        "removed_definition": removed_rel.to_dict() if removed_rel else None,
        "remaining_relationship_count": len(schema.relationships),
    })


def get_target_schema(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get the current target schema definition.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and the current schema
    """
    # Track call count to detect loops
    call_count_key = "_get_target_schema_call_count"
    call_count = tool_context.state.get(call_count_key, 0) + 1
    tool_context.state[call_count_key] = call_count

    # DEBUG: Log when this tool is called
    logger.info(f"=== get_target_schema CALLED (call #{call_count}) ===")
    print(f"\n>>> TOOL CALLED: get_target_schema() - call #{call_count}")

    schema = _get_or_create_schema(tool_context)
    logger.info(f"  Current nodes: {list(schema.nodes.keys())}")
    logger.info(f"  Current relationships: {list(schema.relationships.keys())}")
    print(f">>> get_target_schema: {len(schema.nodes)} nodes, {len(schema.relationships)} relationships")

    # Validate schema
    connected, isolated = schema.validate_connectivity()
    valid_rels, rel_errors = schema.validate_relationships()

    # Check if schema is complete (has nodes and relationships and no errors)
    is_complete = (
        len(schema.nodes) > 0 and
        len(schema.relationships) > 0 and
        connected and
        valid_rels
    )

    # Build result with guidance for the agent
    result = {
        "schema": schema.to_dict(),
        "summary": schema.get_summary(),
        "validation": {
            "is_connected": connected,
            "isolated_nodes": isolated,
            "relationship_errors": rel_errors,
            "is_valid": connected and valid_rels,
        },
    }

    # Add explicit guidance based on schema state
    # Also add loop detection warning if called too many times
    loop_warning = ""
    if call_count > 3:
        loop_warning = (
            f"⚠️ WARNING: You have called get_target_schema {call_count} times! "
            "You appear to be in a loop. STOP calling get_target_schema and "
            "IMMEDIATELY call propose_node_type to add nodes. "
        )

    if is_complete:
        result["next_action"] = (
            "STOP! Schema is complete and valid. "
            "DO NOT call get_target_schema again. "
            "Present this schema to the user and ask for their approval. "
            "If user approves, call approve_target_schema."
        )
    elif len(schema.nodes) == 0:
        result["next_action"] = (
            f"{loop_warning}"
            "⚠️ CRITICAL: Schema is EMPTY! You MUST call 'propose_node_type' NOW to add your first node. "
            "DO NOT call get_target_schema again until you have added at least one node. "
            "Example: propose_node_type(node_type='Respondent', identifier_property='id', "
            "properties=[{'name': 'id', 'data_type': 'string'}], "
            "extraction_hints={'source_type': 'identifier', 'column_pattern': '序号'})"
        )
    elif len(schema.relationships) == 0:
        result["next_action"] = (
            f"{loop_warning}"
            "Schema has nodes but no relationships. "
            "Call propose_relationship_type to add relationships connecting the nodes."
        )
    elif not connected:
        result["next_action"] = f"{loop_warning}Schema has isolated nodes: {isolated}. Add relationships to connect them."
    elif rel_errors:
        result["next_action"] = f"{loop_warning}Fix relationship errors: {rel_errors}"

    return tool_success(TARGET_SCHEMA_KEY, result)


def _save_schema_to_file(tool_context: ToolContext, schema: TargetSchema) -> Dict[str, str]:
    """
    Save approved schema to files (JSON + Markdown).

    Args:
        tool_context: ADK ToolContext for state management
        schema: The approved TargetSchema

    Returns:
        Dictionary with paths to saved files
    """
    from .task_manager import get_task_config

    # Determine output directory
    task_config = get_task_config(tool_context)
    if task_config:
        output_dir = Path(task_config["base_dir"]) / "schema"
    else:
        import_dir = get_neo4j_import_dir()
        output_dir = Path(import_dir) / "schema" if import_dir else Path("data/schema")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%H%M")
    base_name = f"schema{timestamp}"

    saved_files = {}

    # Save JSON format (machine-readable)
    json_path = output_dir / f"{base_name}.json"
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(schema.to_json())
        saved_files["json"] = str(json_path)
        logger.info(f"Saved schema JSON to: {json_path}")
    except Exception as e:
        logger.warning(f"Failed to save schema JSON: {e}")

    # Save Markdown format (human-readable)
    md_path = output_dir / f"{base_name}.md"
    try:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(schema.to_markdown())
        saved_files["markdown"] = str(md_path)
        logger.info(f"Saved schema Markdown to: {md_path}")
    except Exception as e:
        logger.warning(f"Failed to save schema Markdown: {e}")

    return saved_files


def approve_target_schema(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Approve the current target schema for preprocessing.

    This signals that the schema design is complete and the pipeline
    should proceed to targeted preprocessing.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and the approved schema
    """
    if TARGET_SCHEMA_KEY not in tool_context.state:
        return tool_error(
            "No target schema has been designed. "
            "Use propose_node_type and propose_relationship_type to design the schema first."
        )

    schema = _get_or_create_schema(tool_context)

    # Validate before approval
    if not schema.nodes:
        return tool_error("Cannot approve empty schema. Add at least one node type.")

    connected, isolated = schema.validate_connectivity()
    valid_rels, rel_errors = schema.validate_relationships()

    if rel_errors:
        return tool_error(
            f"Cannot approve schema with relationship errors: {rel_errors}"
        )

    # Store as approved
    tool_context.state[APPROVED_TARGET_SCHEMA_KEY] = schema.to_dict()

    # Clear any pending rollback/revision reason since schema is now approved
    # This is the proper place to clear it (not in get_schema_revision_reason)
    tool_context.state[SCHEMA_REVISION_REASON_KEY] = None

    # Save schema to files (JSON + Markdown)
    saved_files = _save_schema_to_file(tool_context, schema)

    result = {
        "schema": schema.to_dict(),
        "summary": schema.get_summary(),
        "warnings": [],
        "saved_files": saved_files,
    }

    if isolated:
        result["warnings"].append(
            f"Warning: The following nodes are not connected: {isolated}"
        )

    return tool_success(APPROVED_TARGET_SCHEMA_KEY, result)


# State key for schema design feedback from critic
SCHEMA_DESIGN_FEEDBACK_KEY = "schema_design_feedback"


def get_schema_design_feedback(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get feedback from the schema design critic.

    Use this to check what issues the critic found with the schema design.
    If empty or "valid", the schema is ready for approval.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with the current feedback
    """
    feedback = tool_context.state.get(SCHEMA_DESIGN_FEEDBACK_KEY, "")
    return tool_success(SCHEMA_DESIGN_FEEDBACK_KEY, {
        "feedback": feedback,
        "has_issues": feedback and feedback != "valid",
    })


def get_approved_target_schema(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get the approved target schema.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and the approved schema,
        or error if not approved yet
    """
    if APPROVED_TARGET_SCHEMA_KEY not in tool_context.state:
        return tool_error(
            "Target schema has not been approved. "
            "Design the schema and call approve_target_schema first."
        )

    return tool_success(
        APPROVED_TARGET_SCHEMA_KEY,
        tool_context.state[APPROVED_TARGET_SCHEMA_KEY]
    )


def update_node_extraction_hints(
    label: str,
    extraction_hints: Dict[str, Any],
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Update extraction hints for a node type.

    Extraction hints guide the preprocessing agent on how to extract
    entities for this node type from the raw data.

    Args:
        label: Node label to update
        extraction_hints: New extraction hints (merged with existing)
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and updated node definition
    """
    schema = _get_or_create_schema(tool_context)

    if label not in schema.nodes:
        return tool_error(f"Node '{label}' not found in schema")

    node = schema.nodes[label]
    node.extraction_hints.update(extraction_hints)
    _save_schema(tool_context, schema)

    return tool_success("updated_node", {
        "label": label,
        "definition": node.to_dict(),
    })


def update_relationship_extraction_hints(
    relationship_type: str,
    extraction_hints: Dict[str, Any],
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Update extraction hints for a relationship type.

    Extraction hints guide the preprocessing agent on how to extract
    relationships of this type from the raw data.

    Args:
        relationship_type: Relationship type to update
        extraction_hints: New extraction hints (merged with existing)
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with status and updated relationship definition
    """
    schema = _get_or_create_schema(tool_context)

    if relationship_type not in schema.relationships:
        return tool_error(f"Relationship '{relationship_type}' not found in schema")

    rel = schema.relationships[relationship_type]
    rel.extraction_hints.update(extraction_hints)
    _save_schema(tool_context, schema)

    return tool_success("updated_relationship", {
        "relationship_type": relationship_type,
        "definition": rel.to_dict(),
    })


# State key for schema revision reason (from preprocessing rollback)
SCHEMA_REVISION_REASON_KEY = "schema_revision_reason"

# State key for column name standardization mapping
COLUMN_STANDARDIZATION_KEY = "column_standardization"
STANDARDIZED_FILE_KEY = "standardized_file"


# Standard names for entity types (used in auto-detection)
ENTITY_STANDARD_NAMES = {
    EntityType.RESPONDENT: "respondent_id",
    EntityType.BRAND: "brand",
    EntityType.MODEL: "model",
    EntityType.STORE: "store",
}


def _auto_detect_entity_columns(
    columns: List[str],
    schema: TargetSchema,
    existing_renames: Dict[str, str],
) -> Dict[str, str]:
    """
    Auto-detect entity columns based on schema node definitions.

    This function finds columns that match entity patterns but weren't included
    in the user-provided rename_map, and suggests standardized names for them.

    Args:
        columns: List of column names in the data file
        schema: The current target schema
        existing_renames: Already specified renames (won't be overwritten)

    Returns:
        Dictionary of additional column renames to apply
    """
    additional_renames = {}
    already_renamed_to = set(existing_renames.values())
    already_renamed_from = set(existing_renames.keys())

    # For each node type in the schema, try to find matching columns
    for node in schema.nodes:
        # Handle both NodeDefinition objects and strings
        if isinstance(node, str):
            node_type = node.lower()
        elif hasattr(node, 'node_type'):
            node_type = node.node_type.lower()
        else:
            continue

        # Try to determine the entity type
        entity_type = None
        for et in EntityType:
            if et.value.lower() == node_type:
                entity_type = et
                break

        if not entity_type:
            continue

        # Get the standard name for this entity type
        standard_name = ENTITY_STANDARD_NAMES.get(entity_type)
        if not standard_name:
            continue

        # Skip if this standard name is already in use
        if standard_name in already_renamed_to:
            continue

        # Get patterns for this entity type
        patterns = ENTITY_PATTERNS.get(entity_type, [])
        if not patterns:
            continue

        # Search for matching columns
        for col in columns:
            # Skip if already being renamed
            if col in already_renamed_from:
                continue

            col_normalized = col.strip() if isinstance(col, str) else str(col)
            # Remove duplicate number prefixes (e.g., "10、10" -> "10、")
            col_normalized = re.sub(r'^(\d+[、\.])\1+', r'\1', col_normalized)
            # Remove trailing punctuation and spaces
            col_cleaned = re.sub(r'[\s,，。.?？!！]+$', '', col_normalized)

            # Check if column matches any pattern
            for pattern in patterns:
                if re.search(pattern, col_cleaned, re.IGNORECASE):
                    additional_renames[col] = standard_name
                    already_renamed_to.add(standard_name)
                    already_renamed_from.add(col)
                    logger.info(f"Auto-detected entity column: '{col}' -> '{standard_name}' (pattern: {pattern})")
                    break

            # If we found a match for this entity type, move to next type
            if standard_name in already_renamed_to:
                break

    return additional_renames


def standardize_column_names(
    file_path: str,
    rename_map: Dict[str, str],
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Standardize column names in a data file based on the schema design.

    This tool renames complex column names to clean, standardized names that
    are easier for the preprocessing agent to work with. The standardized
    column names should align with the schema's extraction hints.

    Example rename_map:
    {
        '该车型"外观设计"方面您会打多少分呢？': 'appearance_design_score',
        '8、您本次调研的品牌是？': 'brand',
        '9、您本次调研的车型及配置是？': 'model',
        '10、您本次到访的门店是': 'store',
        '序号': 'respondent_id',
    }

    Args:
        file_path: File to process, relative to the import directory
        rename_map: Dictionary mapping original column names to new names
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with:
        - standardized_file: Path to the standardized file
        - columns_renamed: Number of columns renamed
        - rename_details: List of rename operations performed
        - unmatched_columns: Columns in rename_map that weren't found
    """
    logger.info(f"=== standardize_column_names CALLED ===")
    logger.info(f"  file_path: {file_path}")
    logger.info(f"  rename_map: {rename_map}")

    # Validate file path
    validation_error = validate_file_path(file_path)
    if validation_error:
        return validation_error

    import_dir = _get_import_dir()
    if not import_dir:
        return tool_error("NEO4J_IMPORT_DIR not configured.")

    full_path = import_dir / file_path

    if not full_path.exists():
        return tool_error(f"File does not exist: {file_path}")

    try:
        # Read file
        suffix = full_path.suffix.lower()
        if suffix == '.csv':
            df = pd.read_csv(full_path)
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(full_path)
        else:
            return tool_error(f"Unsupported file type: {suffix}")

        # Track renames
        rename_details = []
        unmatched_columns = []
        actual_renames = {}

        # Normalize current column names for matching
        # (handle whitespace, full-width vs half-width characters)
        col_to_normalized = {}
        for col in df.columns:
            normalized = col.strip() if isinstance(col, str) else str(col)
            # Fix duplicate number prefixes
            normalized = re.sub(r'^(\d+[、\.])\1+', r'\1', normalized)
            col_to_normalized[col] = normalized

        # AUTO-DETECT ENTITY COLUMNS based on schema
        # This ensures entity columns are renamed even if LLM forgets to include them
        schema = _get_or_create_schema(tool_context)
        auto_detected = _auto_detect_entity_columns(
            columns=list(df.columns),
            schema=schema,
            existing_renames=rename_map,
        )

        if auto_detected:
            logger.info(f"Auto-detected {len(auto_detected)} additional entity column(s)")
            # Merge auto-detected into rename_map (user-provided takes precedence)
            for col, new_name in auto_detected.items():
                if col not in rename_map and new_name not in rename_map.values():
                    rename_map[col] = new_name
                    logger.info(f"  Added auto-detected rename: '{col}' -> '{new_name}'")

        # Process rename_map
        for original, new_name in rename_map.items():
            # Try exact match first
            if original in df.columns:
                actual_renames[original] = new_name
                rename_details.append({
                    "original": original,
                    "new_name": new_name,
                    "match_type": "exact"
                })
                continue

            # Try normalized match
            original_normalized = original.strip() if isinstance(original, str) else str(original)
            original_normalized = re.sub(r'^(\d+[、\.])\1+', r'\1', original_normalized)

            matched = False
            for col, col_norm in col_to_normalized.items():
                # Normalize both for comparison
                if col_norm == original_normalized:
                    actual_renames[col] = new_name
                    rename_details.append({
                        "original": col,
                        "new_name": new_name,
                        "match_type": "normalized"
                    })
                    matched = True
                    break

                # Try partial match (contains check for long column names)
                # This helps with character encoding differences (full-width vs half-width)
                original_core = re.sub(r'[?？。.!！]', '', original_normalized)
                col_core = re.sub(r'[?？。.!！]', '', col_norm)
                if original_core and col_core and (original_core in col_core or col_core in original_core):
                    actual_renames[col] = new_name
                    rename_details.append({
                        "original": col,
                        "new_name": new_name,
                        "match_type": "partial",
                        "requested": original
                    })
                    matched = True
                    break

            if not matched:
                unmatched_columns.append(original)

        if not actual_renames:
            return tool_error(
                f"No columns could be matched for renaming. "
                f"Available columns: {list(df.columns)[:10]}... "
                f"Requested renames: {list(rename_map.keys())[:5]}..."
            )

        # Perform the rename
        df = df.rename(columns=actual_renames)

        # Generate output filename
        stem = full_path.stem
        output_filename = f"{stem}_standardized.csv"
        output_path = import_dir / output_filename

        # Save as CSV (standardized format)
        df.to_csv(output_path, index=False, encoding='utf-8')

        # Store standardization info in state
        tool_context.state[COLUMN_STANDARDIZATION_KEY] = {
            "original_file": file_path,
            "standardized_file": output_filename,
            "rename_map": actual_renames,
            "reverse_map": {v: k for k, v in actual_renames.items()},
        }
        tool_context.state[STANDARDIZED_FILE_KEY] = output_filename

        # Update approved_files to point to the standardized file
        approved_files = tool_context.state.get("approved_files", [])
        if file_path in approved_files:
            # Replace with standardized file
            approved_files = [output_filename if f == file_path else f for f in approved_files]
            tool_context.state["approved_files"] = approved_files

        logger.info(f"Standardized file saved: {output_filename}")
        logger.info(f"Columns renamed: {len(actual_renames)}")

        # Auto-sync Schema extraction_hints to use new column names
        sync_result = _sync_schema_column_patterns(tool_context, actual_renames)
        if sync_result["updated_nodes"] or sync_result["updated_relationships"]:
            logger.info("Auto-synced Schema extraction_hints:")
            for node_update in sync_result["updated_nodes"]:
                logger.info(f"  Node '{node_update['label']}': '{node_update['old_pattern']}' -> '{node_update['new_pattern']}'")
            for rel_update in sync_result["updated_relationships"]:
                logger.info(f"  Rel '{rel_update['type']}': '{rel_update['old_pattern']}' -> '{rel_update['new_pattern']}'")

        # Build informative message
        auto_detected_count = len(auto_detected) if auto_detected else 0
        message = f"Successfully standardized {len(actual_renames)} column names. "
        if auto_detected_count > 0:
            message += f"({auto_detected_count} auto-detected based on schema). "
        message += f"The standardized file '{output_filename}' will be used for preprocessing. "

        # Update message based on sync result
        sync_count = len(sync_result["updated_nodes"]) + len(sync_result["updated_relationships"])
        if sync_count > 0:
            message += f"Schema extraction_hints have been auto-updated ({sync_count} patterns synced)."
        else:
            message += "Schema extraction_hints are already up-to-date."

        # Add warnings about patterns that may not match
        sync_warnings = sync_result.get("warnings", [])
        if sync_warnings:
            message += f" WARNING: {len(sync_warnings)} pattern(s) may not match any column: "
            message += "; ".join(sync_warnings[:3])
            if len(sync_warnings) > 3:
                message += f"... and {len(sync_warnings) - 3} more"

        return tool_success("column_standardization", {
            "original_file": file_path,
            "standardized_file": output_filename,
            "columns_renamed": len(actual_renames),
            "auto_detected_count": auto_detected_count,
            "auto_detected_columns": auto_detected if auto_detected else {},
            "rename_details": rename_details,
            "unmatched_columns": unmatched_columns,
            "new_columns": list(df.columns),
            "schema_sync": sync_result,
            "schema_sync_warnings": sync_warnings,
            "message": message,
        })

    except Exception as e:
        logger.error(f"Error standardizing columns in {file_path}: {e}")
        return tool_error(f"Error standardizing columns: {e}")


def get_schema_revision_reason(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get the reason for schema revision if this is a rollback from preprocessing.

    When the preprocessing agent encounters issues with the schema (e.g., column
    patterns don't match, entity types are missing), it can request a rollback
    to the schema design phase. This tool retrieves the reason and suggested
    changes from that rollback request.

    Call this tool at the start of schema design to check if you need to
    address specific issues from a previous preprocessing attempt.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with revision reason and suggested changes, or empty
        if this is not a rollback scenario
    """
    revision_info = tool_context.state.get(SCHEMA_REVISION_REASON_KEY, None)

    if not revision_info:
        return tool_success("schema_revision_reason", {
            "is_rollback": False,
            "reason": None,
            "suggested_changes": [],
            "message": "No schema revision requested. This is a fresh schema design session.",
        })

    # NOTE: Do NOT clear the revision reason here!
    # The LoopAgent may run multiple iterations (designer → critic → checker),
    # and each iteration needs to see the rollback info.
    # The revision reason should only be cleared when schema is approved again.

    return tool_success("schema_revision_reason", {
        "is_rollback": True,
        "reason": revision_info.get("reason", "Unknown reason"),
        "suggested_changes": revision_info.get("suggested_changes", []),
        "requested_at": revision_info.get("requested_at", "Unknown"),
        "message": (
            "This is a rollback from preprocessing. Please address the issues "
            "mentioned in 'reason' and incorporate the 'suggested_changes' into "
            "your schema design."
        ),
        "next_action": (
            "IMPORTANT: This is a ROLLBACK from preprocessing! "
            "You MUST fix the issues listed above before the schema can work. "
            "Use 'update_node_extraction_hints' and 'update_relationship_extraction_hints' "
            "to correct the column patterns. Then proceed to Step 5 to present the revised schema."
        ),
    })


# =============================================================================
# TEXT FEEDBACK COLUMN TOOLS
# =============================================================================

# Patterns for identifying text feedback columns (DEPRECATED - use LLM detection)
# NOTE: These patterns are kept for backward compatibility only.
# When USE_LLM_DETECTION = True, LLM-based detection is used instead.
TEXT_FEEDBACK_PATTERNS = [
    r"_positive",       # *_positive columns
    r"_negative",       # *_negative columns
    r"优秀点",           # Chinese for "excellent points"
    r"劣势点",           # Chinese for "disadvantage points"
    r"_insight",        # *_insight columns
    r"启发项",           # Chinese for "inspiration items"
    r"_experience",     # *_experience columns
    r"体验是",           # Chinese for "experience is"
]

# State keys for text extraction
TEXT_FEEDBACK_COLUMNS_KEY = "text_feedback_columns"
TEXT_ENTITY_TYPES_KEY = "text_entity_types"
TEXT_RELATIONSHIP_TYPES_KEY = "text_relationship_types"

# LLM model for text analysis
TEXT_ANALYSIS_MODEL = "qwen-plus-latest"


def _get_text_llm_client():
    """Get OpenAI client configured for DashScope."""
    from openai import OpenAI
    from ..config import get_config
    config = get_config()
    return OpenAI(
        api_key=config.llm.api_key,
        base_url=config.llm.api_base,
        timeout=60.0  # 60 second timeout to prevent hanging
    )


def identify_text_feedback_columns(
    file_path: str,
    tool_context: ToolContext,
    use_llm: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Identify text feedback columns in a data file.

    When USE_LLM_DETECTION=True (default), uses LLM for domain-agnostic detection.
    Otherwise, scans column names for patterns that indicate text feedback content
    (e.g., *_positive, *_negative, *_insight, 优秀点, 劣势点).

    Args:
        file_path: File to analyze, relative to the import directory
        tool_context: ADK ToolContext for state management
        use_llm: Override for LLM detection (None uses global USE_LLM_DETECTION)

    Returns:
        Dictionary with identified text feedback columns:
        - text_columns: List of column info with name, category, sample_values
        - count: Number of text feedback columns found
    """
    should_use_llm = use_llm if use_llm is not None else USE_LLM_DETECTION

    validation_error = validate_file_path(file_path)
    if validation_error:
        return validation_error

    import_dir = _get_import_dir()
    if not import_dir:
        return tool_error("NEO4J_IMPORT_DIR not configured.")

    full_path = import_dir / file_path

    if not full_path.exists():
        return tool_error(f"File does not exist: {file_path}")

    try:
        # Read file
        suffix = full_path.suffix.lower()
        if suffix == '.csv':
            df = pd.read_csv(full_path)
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(full_path)
        else:
            return tool_error(f"Unsupported file type: {suffix}")

        text_columns = []

        # Try LLM-based detection first when enabled
        if should_use_llm:
            logger.info(f"Using LLM-based text feedback detection for: {file_path}")
            llm_result = llm_detect_text_feedback(file_path, tool_context)

            if llm_result.get("status") != "error" and llm_result.get("text_columns"):
                # Use LLM detection results
                for col_info in llm_result.get("text_columns", []):
                    col = col_info.get("column", "")

                    # Get sample values and stats if column exists in dataframe
                    if col in df.columns:
                        sample_values = df[col].dropna().head(5).tolist()
                        sample_values = [
                            str(v)[:200] for v in sample_values
                            if str(v).strip() and str(v).lower() not in ['nan', 'none', '无', '-', '暂无']
                        ]
                        non_empty_count = df[col].dropna().apply(
                            lambda x: str(x).strip() and str(x).lower() not in ['nan', 'none', '无', '-', '暂无']
                        ).sum()
                    else:
                        sample_values = []
                        non_empty_count = 0

                    # Map LLM feedback_type to category
                    feedback_type = col_info.get("feedback_type", "mixed")
                    if feedback_type == "positive":
                        category = "positive_feedback"
                    elif feedback_type == "negative":
                        category = "negative_feedback"
                    elif feedback_type == "neutral":
                        category = "insight"
                    else:
                        category = "text_feedback"

                    text_columns.append({
                        "column_name": col,
                        "category": category,
                        "sample_values": sample_values[:3],
                        "non_empty_count": int(non_empty_count),
                        "total_count": int(len(df)),
                        "fill_rate": round(non_empty_count / len(df) * 100, 1) if len(df) > 0 else 0,
                        "detection_method": "llm",
                        "content_category": col_info.get("content_category", ""),
                        "suggested_entities": col_info.get("suggested_entities", []),
                        "reason": col_info.get("reason", ""),
                        "priority": col_info.get("priority", 5),
                    })

                # Store in state
                tool_context.state[TEXT_FEEDBACK_COLUMNS_KEY] = text_columns

                return tool_success("text_feedback_columns", {
                    "file_path": file_path,
                    "text_columns": text_columns,
                    "count": len(text_columns),
                    "detection_method": "llm",
                    "message": f"Found {len(text_columns)} text feedback columns (LLM detection)",
                })

        # Fallback to pattern-based detection
        logger.info(f"Using pattern-based text feedback detection for: {file_path}")

        for col in df.columns:
            col_str = str(col).lower()

            # Check if column matches text feedback patterns
            for pattern in TEXT_FEEDBACK_PATTERNS:
                if re.search(pattern, col_str, re.IGNORECASE):
                    # Determine category based on pattern
                    if re.search(r"_positive|优秀点", col_str, re.IGNORECASE):
                        category = "positive_feedback"
                    elif re.search(r"_negative|劣势点", col_str, re.IGNORECASE):
                        category = "negative_feedback"
                    elif re.search(r"_insight|启发项", col_str, re.IGNORECASE):
                        category = "insight"
                    elif re.search(r"_experience|体验是", col_str, re.IGNORECASE):
                        category = "experience"
                    else:
                        category = "text_feedback"

                    # Get sample values (non-empty)
                    sample_values = df[col].dropna().head(5).tolist()
                    sample_values = [
                        str(v)[:200] for v in sample_values
                        if str(v).strip() and str(v).lower() not in ['nan', 'none', '无', '-', '暂无']
                    ]

                    # Count non-empty values
                    non_empty_count = df[col].dropna().apply(
                        lambda x: str(x).strip() and str(x).lower() not in ['nan', 'none', '无', '-', '暂无']
                    ).sum()

                    text_columns.append({
                        "column_name": col,
                        "category": category,
                        "sample_values": sample_values[:3],
                        "non_empty_count": int(non_empty_count),
                        "total_count": int(len(df)),
                        "fill_rate": round(non_empty_count / len(df) * 100, 1) if len(df) > 0 else 0,
                        "detection_method": "pattern",
                    })
                    break

        # Store in state
        tool_context.state[TEXT_FEEDBACK_COLUMNS_KEY] = text_columns

        return tool_success("text_feedback_columns", {
            "file_path": file_path,
            "text_columns": text_columns,
            "count": len(text_columns),
            "detection_method": "pattern",
            "message": f"Found {len(text_columns)} text feedback columns",
        })

    except Exception as e:
        logger.error(f"Error identifying text feedback columns in {file_path}: {e}")
        return tool_error(f"Error identifying text feedback columns: {e}")


def sample_text_column(
    file_path: str,
    column_name: str,
    tool_context: ToolContext,
    sample_size: int = 20
) -> Dict[str, Any]:
    """
    Sample text content from a specific column for entity type analysis.

    Returns diverse text samples that can be used to identify entity types
    and relationships present in the text feedback.

    Args:
        file_path: File to read, relative to the import directory
        column_name: Name of the column to sample
        tool_context: ADK ToolContext for state management
        sample_size: Number of samples to return (default: 20)

    Returns:
        Dictionary with:
        - samples: List of text samples
        - statistics: Column statistics (unique values, avg length, etc.)
    """
    validation_error = validate_file_path(file_path)
    if validation_error:
        return validation_error

    import_dir = _get_import_dir()
    if not import_dir:
        return tool_error("NEO4J_IMPORT_DIR not configured.")

    full_path = import_dir / file_path

    if not full_path.exists():
        return tool_error(f"File does not exist: {file_path}")

    try:
        # Read file
        suffix = full_path.suffix.lower()
        if suffix == '.csv':
            df = pd.read_csv(full_path)
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(full_path)
        else:
            return tool_error(f"Unsupported file type: {suffix}")

        # Normalize column names by stripping whitespace
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
        column_name_stripped = column_name.strip() if isinstance(column_name, str) else column_name

        # Try exact match first, then fuzzy match
        actual_column = None
        if column_name_stripped in df.columns:
            actual_column = column_name_stripped
        else:
            # Try to find a column that contains the search term
            for col in df.columns:
                if column_name_stripped in str(col) or str(col) in column_name_stripped:
                    actual_column = col
                    break

        if actual_column is None:
            return tool_error(f"Column not found: {column_name}. Available columns: {list(df.columns)[:10]}...")

        column_name = actual_column

        # Filter out empty/placeholder values
        placeholder_values = {'nan', 'none', '无', '-', '暂无', '未知', '不清楚', ''}
        str_series = df[column_name].dropna().astype(str)
        valid_mask = ~str_series.str.lower().isin(placeholder_values) & (str_series.str.strip() != '')
        valid_texts = str_series[valid_mask]

        # Sample diverse texts (try to get variety)
        if len(valid_texts) > sample_size:
            # Sample from different parts of the data for diversity
            samples = valid_texts.sample(n=sample_size, random_state=42).tolist()
        else:
            samples = valid_texts.tolist()

        # Truncate long texts
        samples = [s[:500] for s in samples]

        # Calculate statistics
        avg_length = valid_texts.str.len().mean() if len(valid_texts) > 0 else 0
        unique_count = valid_texts.nunique()

        return tool_success("text_samples", {
            "column_name": column_name,
            "samples": samples,
            "statistics": {
                "total_rows": int(len(df)),
                "valid_text_count": int(len(valid_texts)),
                "unique_count": int(unique_count),
                "avg_length": round(avg_length, 1),
                "fill_rate": round(len(valid_texts) / len(df) * 100, 1) if len(df) > 0 else 0,
            },
        })

    except Exception as e:
        logger.error(f"Error sampling text column {column_name}: {e}")
        return tool_error(f"Error sampling text column: {e}")


def analyze_text_column_entities(
    column_name: str,
    samples: List[str],
    column_category: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Use LLM to analyze text samples and identify entity types present.

    Examines text content to discover what types of entities are mentioned
    (e.g., Feature, Issue, Component, etc.) and provides examples.

    Args:
        column_name: Name of the column being analyzed
        samples: List of text samples from the column
        column_category: Category of the column (positive_feedback, negative_feedback, etc.)
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with discovered entity types:
        - entity_types: List of {type, examples, description}
        - recommended_nodes: Suggested node definitions for schema
    """
    if not samples:
        return tool_success("entity_analysis", {
            "entity_types": [],
            "recommended_nodes": [],
            "message": "No samples provided for analysis",
        })

    client = _get_text_llm_client()

    # Prepare sample text for analysis
    samples_text = "\n".join([f"- {s[:300]}" for s in samples[:15]])

    system_prompt = """你是知识图谱建模专家。分析调研问卷中的文本反馈，识别其中提到的实体类型。

实体类型要求：
1. 实体是"人、地点、事物、品质"——不是数量
2. 使用单数名词（Feature不是Features）
3. 选择领域相关、清晰的名称
4. 常见实体类型：Feature（特性/功能）、Issue（问题）、Component（部件）、Experience（体验）

请严格按JSON格式输出。"""

    user_prompt = f"""分析以下来自"{column_name}"列（类别: {column_category}）的文本样本：

{samples_text}

请识别文本中提到的实体类型，返回JSON格式：
{{
    "entity_types": [
        {{
            "type": "Feature",
            "description": "产品的具体特性或功能",
            "examples": ["后排娱乐屏", "沉坑设计", "贯穿式尾灯"]
        }}
    ],
    "analysis_notes": "简要说明识别逻辑"
}}"""

    try:
        response = client.chat.completions.create(
            model=TEXT_ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000  # Increased from 1000 to handle complex responses
        )

        result_text = (response.choices[0].message.content or "").strip()

        # Use robust JSON parser to handle truncated or malformed responses
        result = _robust_json_parse(result_text, default_key="entity_types")
        entity_types = result.get("entity_types", [])

        # Build recommended node definitions
        recommended_nodes = []
        for et in entity_types:
            node_def = {
                "node_type": et["type"],
                "unique_property": f"{et['type'].lower()}_id",
                "properties": ["name", "source_column"],
                "extraction_hints": {
                    "source_type": "text_extraction",
                    "column_pattern": column_name,
                    "entity_type": et["type"],
                },
                "description": et.get("description", ""),
                "examples": et.get("examples", []),
            }
            recommended_nodes.append(node_def)

        # Store in state
        existing_types = tool_context.state.get(TEXT_ENTITY_TYPES_KEY, [])
        for et in entity_types:
            et["source_column"] = column_name
            existing_types.append(et)
        tool_context.state[TEXT_ENTITY_TYPES_KEY] = existing_types

        return tool_success("entity_analysis", {
            "column_name": column_name,
            "column_category": column_category,
            "entity_types": entity_types,
            "recommended_nodes": recommended_nodes,
            "analysis_notes": result.get("analysis_notes", ""),
        })

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response for entity analysis: {e}")
        return tool_error(f"Failed to parse entity analysis response: {e}")
    except Exception as e:
        logger.error(f"Error in entity analysis for {column_name}: {e}")
        return tool_error(f"Error analyzing entities: {e}")


def analyze_text_column_relationships(
    column_name: str,
    samples: List[str],
    entity_types: List[str],
    column_category: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Use LLM to analyze text samples and identify relationship types.

    Based on discovered entity types, identifies what relationships exist
    between entities mentioned in the text feedback.

    Args:
        column_name: Name of the column being analyzed
        samples: List of text samples from the column
        entity_types: List of entity type names discovered earlier
        column_category: Category of the column
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with discovered relationship types:
        - relationship_types: List of {type, from_node, to_node, properties}
        - recommended_relationships: Suggested relationship definitions for schema
    """
    if not samples or not entity_types:
        return tool_success("relationship_analysis", {
            "relationship_types": [],
            "recommended_relationships": [],
            "message": "No samples or entity types provided",
        })

    client = _get_text_llm_client()

    samples_text = "\n".join([f"- {s[:300]}" for s in samples[:10]])
    entity_types_str = ", ".join(entity_types)

    # Determine sentiment based on column category
    sentiment = "positive" if "positive" in column_category else (
        "negative" if "negative" in column_category else "neutral"
    )

    system_prompt = """你是知识图谱建模专家。基于已识别的实体类型，分析文本中的关系类型。

关系命名规则：
1. 使用大写蛇形命名（如 MENTIONED_FEATURE）
2. 动词形式（如 HAS, CONTAINS, MENTIONS, RELATES_TO）
3. 考虑方向性（from → to）

请严格按JSON格式输出。"""

    user_prompt = f"""基于以下信息分析关系类型：

列名: {column_name}
类别: {column_category}（{sentiment}反馈）
已发现的实体类型: {entity_types_str}

文本样本：
{samples_text}

请识别文本中的关系类型，返回JSON格式：
{{
    "relationship_types": [
        {{
            "type": "MENTIONED_FEATURE",
            "from_node": "Respondent",
            "to_node": "Feature",
            "properties": ["sentiment", "source_aspect"],
            "description": "受访者提到的产品特性"
        }}
    ],
    "analysis_notes": "简要说明"
}}

注意：from_node 通常是 "Respondent"，表示受访者提到了某个实体。"""

    try:
        response = client.chat.completions.create(
            model=TEXT_ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000  # Increased from 1000 to handle complex responses
        )

        result_text = (response.choices[0].message.content or "").strip()

        # Use robust JSON parser to handle truncated or malformed responses
        result = _robust_json_parse(result_text, default_key="relationship_types")
        relationship_types = result.get("relationship_types", [])

        # Build recommended relationship definitions
        recommended_relationships = []
        for rt in relationship_types:
            rel_def = {
                "relationship_type": rt["type"],
                "from_node": rt["from_node"],
                "to_node": rt["to_node"],
                "properties": rt.get("properties", []),
                "extraction_hints": {
                    "source_type": "text_extraction",
                    "column_pattern": column_name,
                    "sentiment": sentiment,
                    "respondent_column": "respondent_id",
                },
                "description": rt.get("description", ""),
            }
            recommended_relationships.append(rel_def)

        # Store in state
        existing_types = tool_context.state.get(TEXT_RELATIONSHIP_TYPES_KEY, [])
        for rt in relationship_types:
            rt["source_column"] = column_name
            rt["sentiment"] = sentiment
            existing_types.append(rt)
        tool_context.state[TEXT_RELATIONSHIP_TYPES_KEY] = existing_types

        return tool_success("relationship_analysis", {
            "column_name": column_name,
            "entity_types_used": entity_types,
            "relationship_types": relationship_types,
            "recommended_relationships": recommended_relationships,
            "analysis_notes": result.get("analysis_notes", ""),
        })

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response for relationship analysis: {e}")
        return tool_error(f"Failed to parse relationship analysis response: {e}")
    except Exception as e:
        logger.error(f"Error in relationship analysis for {column_name}: {e}")
        return tool_error(f"Error analyzing relationships: {e}")


def add_text_entity_to_schema(
    entity_type: str,
    description: str,
    source_columns: List[str],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Add a text-extracted entity type to the target schema.

    Creates a new node definition for entities that will be extracted
    from text feedback columns using LLM.

    Args:
        entity_type: Name of the entity type (e.g., "Feature", "Issue")
        description: Description of what this entity represents
        source_columns: List of column names to extract from
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with the created node definition
    """
    schema = _get_or_create_schema(tool_context)

    # Check if entity type already exists
    if entity_type in schema.nodes:
        return tool_error(f"Entity type '{entity_type}' already exists in schema")

    # Create node definition
    node_def = NodeDefinition(
        label=entity_type,
        unique_property=f"{entity_type.lower()}_id",
        properties=["name", "category", "source_column"],
        extraction_hints={
            "source_type": "text_extraction",
            "column_pattern": "|".join(source_columns),
            "description": description,
        }
    )

    schema.add_node(node_def)
    _save_schema(tool_context, schema)

    logger.info(f"Added text entity type to schema: {entity_type}")

    return tool_success("text_entity_added", {
        "entity_type": entity_type,
        "unique_property": node_def.unique_property,
        "source_columns": source_columns,
        "extraction_hints": node_def.extraction_hints,
    })


def add_text_relationship_to_schema(
    relationship_type: str,
    from_node: str,
    to_node: str,
    properties: List[str],
    source_columns: List[str],
    sentiment: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Add a text-extracted relationship type to the target schema.

    Creates a new relationship definition for relationships that will be
    extracted from text feedback columns using LLM.

    Args:
        relationship_type: Name of the relationship (e.g., "MENTIONED_FEATURE")
        from_node: Source node type (usually "Respondent")
        to_node: Target node type (e.g., "Feature")
        properties: List of relationship properties
        source_columns: List of column names to extract from
        sentiment: Sentiment category (positive, negative, neutral)
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with the created relationship definition
    """
    schema = _get_or_create_schema(tool_context)

    # Check if relationship type already exists
    if relationship_type in schema.relationships:
        return tool_error(f"Relationship type '{relationship_type}' already exists")

    # Verify nodes exist
    if from_node not in schema.nodes:
        return tool_error(f"From node '{from_node}' not found in schema")
    if to_node not in schema.nodes:
        return tool_error(f"To node '{to_node}' not found in schema")

    # Create relationship definition
    rel_def = RelationshipDefinition(
        relationship_type=relationship_type,
        from_node=from_node,
        to_node=to_node,
        properties=properties,
        extraction_hints={
            "source_type": "text_extraction",
            "column_pattern": "|".join(source_columns),
            "sentiment": sentiment,
            "respondent_column": "respondent_id",
        }
    )

    schema.add_relationship(rel_def)
    _save_schema(tool_context, schema)

    logger.info(f"Added text relationship to schema: {relationship_type}")

    return tool_success("text_relationship_added", {
        "relationship_type": relationship_type,
        "from_node": from_node,
        "to_node": to_node,
        "properties": properties,
        "sentiment": sentiment,
        "source_columns": source_columns,
    })


def get_text_analysis_summary(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get a summary of all text feedback analysis results.

    Returns all discovered entity types and relationships from text
    feedback column analysis.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with:
        - text_columns: Identified text feedback columns
        - entity_types: Discovered entity types
        - relationship_types: Discovered relationship types
    """
    text_columns = tool_context.state.get(TEXT_FEEDBACK_COLUMNS_KEY, [])
    entity_types = tool_context.state.get(TEXT_ENTITY_TYPES_KEY, [])
    relationship_types = tool_context.state.get(TEXT_RELATIONSHIP_TYPES_KEY, [])

    return tool_success("text_analysis_summary", {
        "text_columns_count": len(text_columns),
        "text_columns": text_columns,
        "entity_types_count": len(entity_types),
        "entity_types": entity_types,
        "relationship_types_count": len(relationship_types),
        "relationship_types": relationship_types,
    })

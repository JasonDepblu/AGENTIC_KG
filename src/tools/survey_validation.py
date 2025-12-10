"""
Survey preprocessing validation tools.

This module provides tools for validating the output of survey preprocessing stages,
including column classification, entity extraction, and rating extraction.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from google.adk.tools import ToolContext

from src.config import get_config
from src.tools.common import tool_success, tool_error, get_state_value, set_state_value
from src.tools.survey_classification import COLUMN_CLASSIFICATION, CLASSIFICATION_SUMMARY

logger = logging.getLogger(__name__)

# State keys for validation results
PREPROCESS_FEEDBACK = "preprocess_feedback"
VALIDATION_ISSUES = "validation_issues"
CURRENT_PHASE = "current_phase"


# =============================================================================
# Column Classification Validation
# =============================================================================

def validate_column_classification(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Validates the column classification results.

    Checks:
    1. Entity selection columns contain appropriate values (short names, not long text)
    2. Rating columns contain numeric values
    3. Open text columns contain longer text responses
    4. Consistency in classification (similar questions should have same category)

    Returns:
        Validation result with issues list and feedback status
    """
    classifications = get_state_value(tool_context, COLUMN_CLASSIFICATION, {})
    summary = get_state_value(tool_context, CLASSIFICATION_SUMMARY, {})

    if not classifications:
        return tool_error("No column classification found. Run classify_all_columns first.")

    issues = []
    warnings = []

    # Validate entity_selection columns
    entity_selection_cols = summary.get("entity_selection", [])
    for col_name in entity_selection_cols:
        info = classifications.get(col_name, {})
        entity_type = info.get("entity_type")

        # Check if column name suggests entity selection
        if entity_type == "Brand":
            if "品牌" not in col_name and "brand" not in col_name.lower():
                warnings.append(f"Brand column '{col_name[:30]}...' doesn't contain '品牌' in name")
        elif entity_type == "Model":
            if "车型" not in col_name and "model" not in col_name.lower():
                warnings.append(f"Model column '{col_name[:30]}...' doesn't contain '车型' in name")
        elif entity_type == "Store":
            if "门店" not in col_name and "store" not in col_name.lower():
                warnings.append(f"Store column '{col_name[:30]}...' doesn't contain '门店' in name")

    # Validate rating columns
    rating_cols = summary.get("rating", [])
    for col_name in rating_cols:
        # Rating columns should contain "分" or "评" in the name
        if "分" not in col_name and "评" not in col_name and "score" not in col_name.lower():
            warnings.append(f"Rating column '{col_name[:30]}...' doesn't contain rating keywords")

    # Check for potentially misclassified columns
    open_text_cols = summary.get("open_text", [])
    for col_name in open_text_cols:
        # Open text columns asking for ratings might be misclassified
        if "打多少分" in col_name or "评分" in col_name:
            issues.append(f"Column '{col_name[:30]}...' classified as open_text but contains rating keywords")

    # Generate feedback
    if issues:
        feedback = "retry"
        set_state_value(tool_context, PREPROCESS_FEEDBACK, feedback)
        set_state_value(tool_context, VALIDATION_ISSUES, {
            "phase": "column_classification",
            "issues": issues,
            "warnings": warnings
        })
    else:
        feedback = "valid"
        set_state_value(tool_context, PREPROCESS_FEEDBACK, feedback)

    return tool_success("validation_result", {
        "status": feedback,
        "issues": issues,
        "warnings": warnings,
        "summary": {
            "entity_selection_count": len(entity_selection_cols),
            "rating_count": len(rating_cols),
            "open_text_count": len(open_text_cols),
            "demographic_count": len(summary.get("demographic", [])),
            "id_count": len(summary.get("id", []))
        }
    })


def validate_entity_values(
    file_path: str,
    column_name: str,
    entity_type: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Validates that entity values in a column are appropriate for the entity type.

    Checks:
    - Brand: short names (< 20 chars), not long descriptions
    - Model: model name + config format, not descriptions
    - Store: addresses or store names, not pure numbers

    Args:
        file_path: Path to the survey file
        column_name: Column to validate
        entity_type: "Brand", "Model", or "Store"

    Returns:
        Validation result with sample of invalid values
    """
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
        else:
            df = pd.read_csv(full_path)
    except Exception as e:
        return tool_error(f"Error reading file: {str(e)}")

    if column_name not in df.columns:
        return tool_error(f"Column '{column_name}' not found")

    values = df[column_name].dropna().unique()
    issues = []
    invalid_samples = []

    for value in values[:50]:  # Check first 50 unique values
        value_str = str(value).strip()

        if entity_type == "Brand":
            # Brand names should be short
            if len(value_str) > 20:
                issues.append(f"Brand value too long ({len(value_str)} chars): '{value_str[:30]}...'")
                invalid_samples.append(value_str[:50])
            # Should not contain description keywords
            if any(kw in value_str for kw in ["启发", "设计", "功能", "方面", "体验"]):
                issues.append(f"Brand value looks like description: '{value_str[:30]}...'")
                invalid_samples.append(value_str[:50])

        elif entity_type == "Model":
            # Model names should be short-medium length
            if len(value_str) > 50:
                issues.append(f"Model value too long ({len(value_str)} chars): '{value_str[:30]}...'")
                invalid_samples.append(value_str[:50])
            # Should not contain cost/innovation keywords
            if any(kw in value_str for kw in ["成本", "控制", "降本", "启发"]):
                issues.append(f"Model value looks like description: '{value_str[:30]}...'")
                invalid_samples.append(value_str[:50])

        elif entity_type == "Store":
            # Store values should not be pure numbers
            if value_str.replace(".", "").isdigit():
                issues.append(f"Store value is pure number: '{value_str}'")
                invalid_samples.append(value_str)
            # Should contain location keywords or be a proper name
            # (This is a soft check - not all store names have these)

    if issues:
        return tool_success("validation_result", {
            "status": "invalid",
            "entity_type": entity_type,
            "column_name": column_name,
            "issues": issues[:10],  # Limit to first 10 issues
            "invalid_samples": invalid_samples[:5],
            "total_values": len(values)
        })

    return tool_success("validation_result", {
        "status": "valid",
        "entity_type": entity_type,
        "column_name": column_name,
        "total_values": len(values)
    })


# =============================================================================
# Entity Extraction Validation
# =============================================================================

def validate_extracted_entities(
    entity_type: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Validates extracted entities for a specific type.

    Checks:
    1. Entity IDs are unique
    2. Entity values are appropriate (not descriptions, not numbers)
    3. Source column is correctly identified

    Args:
        entity_type: "Brand", "Model", or "Store"

    Returns:
        Validation result with issues
    """
    from src.tools.survey_entity_extraction import (
        EXTRACTED_ENTITIES, VALUE_TO_ID_MAP, ENTITY_COLUMNS
    )

    entities = get_state_value(tool_context, EXTRACTED_ENTITIES, {})
    entity_list = entities.get(entity_type, [])

    if not entity_list:
        return tool_error(f"No {entity_type} entities found. Run extraction first.")

    issues = []
    warnings = []

    # Check for duplicate IDs
    ids = [e.get("entity_id") for e in entity_list]
    if len(ids) != len(set(ids)):
        issues.append(f"Duplicate entity IDs found in {entity_type} entities")

    # Check entity values
    for entity in entity_list:
        value = str(entity.get("value", "")).strip()
        source = entity.get("source_column", "")

        # Common validation rules
        if not value or value in ["(空)", "无", "-", "暂无", ""]:
            warnings.append(f"Empty or null {entity_type} value: '{value}'")
            continue

        if entity_type == "Brand":
            if len(value) > 20:
                issues.append(f"Brand '{value[:20]}...' is too long - might be extracted from wrong column")
            if any(kw in value for kw in ["场景创新", "亮点设计", "启发项"]):
                issues.append(f"Brand '{value[:30]}...' looks like open-text answer")
            # Check source column
            if "90" in source or "启发" in source:
                issues.append(f"Brand extracted from open-text column: {source[:40]}...")

        elif entity_type == "Model":
            if len(value) > 50:
                issues.append(f"Model '{value[:30]}...' is too long - might be description")
            if any(kw in value for kw in ["成本控制", "降本", "启发"]):
                issues.append(f"Model '{value[:30]}...' looks like open-text answer")
            # Check source column
            if "92" in source or "成本" in source:
                issues.append(f"Model extracted from open-text column: {source[:40]}...")

        elif entity_type == "Store":
            if value.replace(".", "").isdigit():
                issues.append(f"Store '{value}' is a number - might be extracted from rating column")
            if len(value) < 2:
                issues.append(f"Store '{value}' is too short")
            # Check source column
            if "打多少分" in source or "评分" in source:
                issues.append(f"Store extracted from rating column: {source[:40]}...")

    # Determine feedback
    if issues:
        feedback = "retry"
        set_state_value(tool_context, PREPROCESS_FEEDBACK, feedback)
        set_state_value(tool_context, VALIDATION_ISSUES, {
            "phase": "entity_extraction",
            "entity_type": entity_type,
            "issues": issues,
            "warnings": warnings
        })
    else:
        feedback = "valid"

    return tool_success("validation_result", {
        "status": feedback,
        "entity_type": entity_type,
        "entity_count": len(entity_list),
        "issues": issues[:10],
        "warnings": warnings[:10]
    })


def validate_all_entities(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Validates all extracted entity types.

    Returns:
        Combined validation result for Brand, Model, Store
    """
    results = {}
    all_issues = []
    all_warnings = []

    for entity_type in ["Brand", "Model", "Store"]:
        result = validate_extracted_entities(entity_type, tool_context)
        if result.get("status") == "success":
            data = result.get("entity_extraction_validation", {})
            results[entity_type] = data
            all_issues.extend(data.get("issues", []))
            all_warnings.extend(data.get("warnings", []))

    feedback = "retry" if all_issues else "valid"
    set_state_value(tool_context, PREPROCESS_FEEDBACK, feedback)

    return tool_success("validation_result", {
        "status": feedback,
        "results": results,
        "total_issues": len(all_issues),
        "total_warnings": len(all_warnings),
        "issues_summary": all_issues[:10]
    })


# =============================================================================
# Rating Extraction Validation
# =============================================================================

def validate_extracted_aspects(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Validates extracted aspect data.

    Checks:
    1. Aspect names are semantic (e.g., "外观设计"), not codes (e.g., "Q11")
    2. Aspect IDs are unique
    3. Related entity is properly identified

    Returns:
        Validation result with issues
    """
    from src.tools.survey_rating_extraction import EXTRACTED_ASPECTS

    aspects = get_state_value(tool_context, EXTRACTED_ASPECTS, [])

    if not aspects:
        return tool_error("No aspects found. Run extract_aspects first.")

    issues = []
    warnings = []

    # Check for duplicate IDs
    ids = [a.get("aspect_id") for a in aspects]
    if len(ids) != len(set(ids)):
        issues.append("Duplicate aspect IDs found")

    for aspect in aspects:
        aspect_name = str(aspect.get("aspect_name", "")).strip()
        aspect_id = aspect.get("aspect_id", "")
        source = aspect.get("source_column", "")

        # Aspect name should be semantic, not a code
        if re.match(r'^Q\d+$', aspect_name):
            issues.append(f"Aspect '{aspect_name}' is a question code, not a semantic name")

        # Should be Chinese text
        if not any('\u4e00' <= c <= '\u9fff' for c in aspect_name):
            issues.append(f"Aspect '{aspect_name}' doesn't contain Chinese characters")

        # Should be reasonably short
        if len(aspect_name) > 30:
            warnings.append(f"Aspect name is long: '{aspect_name[:30]}...'")

        # Check if aspect was extracted from column name quotes
        if source and '"' in source:
            # Try to extract quoted text
            match = re.search(r'["""]([^"""]+)["""]', source)
            if match:
                expected_name = match.group(1).strip()
                if expected_name != aspect_name:
                    warnings.append(f"Aspect '{aspect_name}' differs from quoted text '{expected_name}'")

    feedback = "retry" if issues else "valid"
    set_state_value(tool_context, PREPROCESS_FEEDBACK, feedback)

    return tool_success("validation_result", {
        "status": feedback,
        "aspect_count": len(aspects),
        "issues": issues[:10],
        "warnings": warnings[:10]
    })


def validate_rating_scores(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Validates extracted rating scores.

    Checks:
    1. Scores are within expected range (1-10)
    2. No text values in score field
    3. Respondent IDs are valid

    Returns:
        Validation result with statistics
    """
    from src.tools.survey_rating_extraction import RATING_DATA

    ratings = get_state_value(tool_context, RATING_DATA, [])

    if not ratings:
        return tool_error("No rating data found. Run extract_ratings first.")

    issues = []
    warnings = []

    scores = []
    invalid_scores = []

    for rating in ratings:
        score = rating.get("score")
        if score is not None:
            try:
                score_val = float(score)
                scores.append(score_val)

                if score_val < 1 or score_val > 10:
                    invalid_scores.append(score_val)
            except (ValueError, TypeError):
                issues.append(f"Non-numeric score value: {score}")

    if invalid_scores:
        warnings.append(f"Found {len(invalid_scores)} scores outside 1-10 range")

    # Check for missing respondent or aspect IDs
    missing_respondent = sum(1 for r in ratings if not r.get("respondent_id"))
    missing_aspect = sum(1 for r in ratings if not r.get("aspect_id"))

    if missing_respondent > 0:
        issues.append(f"{missing_respondent} ratings missing respondent_id")
    if missing_aspect > 0:
        issues.append(f"{missing_aspect} ratings missing aspect_id")

    feedback = "retry" if issues else "valid"

    # Calculate statistics
    if scores:
        import statistics
        stats = {
            "count": len(scores),
            "mean": round(statistics.mean(scores), 2),
            "min": min(scores),
            "max": max(scores),
            "std": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0
        }
    else:
        stats = {"count": 0}

    return tool_success("validation_result", {
        "status": feedback,
        "statistics": stats,
        "issues": issues,
        "warnings": warnings
    })


# =============================================================================
# Overall Preprocess Validation
# =============================================================================

def get_preprocess_feedback(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Gets the current preprocess feedback status.

    Returns:
        Current feedback status and any validation issues
    """
    feedback = get_state_value(tool_context, PREPROCESS_FEEDBACK, "valid")
    issues = get_state_value(tool_context, VALIDATION_ISSUES, {})
    phase = get_state_value(tool_context, CURRENT_PHASE, 1)

    return tool_success("feedback", {
        "status": feedback,
        "current_phase": phase,
        "issues": issues
    })


def set_preprocess_feedback(
    feedback: str,
    reason: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Sets the preprocess feedback status.

    Args:
        feedback: "valid", "retry", or "reprocess:phase:reason"
        reason: Description of why feedback was set

    Returns:
        Confirmation of feedback set
    """
    set_state_value(tool_context, PREPROCESS_FEEDBACK, feedback)
    set_state_value(tool_context, VALIDATION_ISSUES, {
        "feedback": feedback,
        "reason": reason
    })

    return tool_success("feedback_set", {
        "feedback": feedback,
        "reason": reason
    })


def advance_preprocess_phase(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Advances to the next preprocess phase.

    Returns:
        New phase number
    """
    current_phase = get_state_value(tool_context, CURRENT_PHASE, 1)
    new_phase = current_phase + 1
    set_state_value(tool_context, CURRENT_PHASE, new_phase)

    # Reset feedback for new phase
    set_state_value(tool_context, PREPROCESS_FEEDBACK, "valid")

    return tool_success("phase_advanced", {
        "previous_phase": current_phase,
        "current_phase": new_phase
    })


# =============================================================================
# Export tools list
# =============================================================================

SURVEY_VALIDATION_TOOLS = [
    # Column classification validation
    validate_column_classification,
    validate_entity_values,
    # Entity extraction validation
    validate_extracted_entities,
    validate_all_entities,
    # Rating extraction validation
    validate_extracted_aspects,
    validate_rating_scores,
    # Feedback management
    get_preprocess_feedback,
    set_preprocess_feedback,
    advance_preprocess_phase,
]

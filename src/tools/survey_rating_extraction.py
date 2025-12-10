"""
Survey rating extraction tools.

This module provides tools for extracting rating data from survey responses
based on pre-classified columns.
"""

import re
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
EXTRACTED_ASPECTS = "extracted_aspects"
COLUMN_TO_ASPECT = "column_to_aspect"
RATING_DATA = "rating_data"


def _extract_aspect_name(column_name: str) -> str:
    """
    Extracts the aspect name from a rating column header.

    Examples:
    - "针对该品牌"外观设计"方面的评分" -> "外观设计"
    - "该门店"服务体验"方面您会打多少分呢？" -> "服务体验"
    - "针对该车型"空间舒适度"的评分" -> "空间舒适度"
    """
    # Try to find text in quotes (Chinese or English)
    patterns = [
        r'["""]([^"""]+)["""]',  # Text in Chinese/English quotes
        r'"([^"]+)"',             # Text in double quotes
        r"'([^']+)'",             # Text in single quotes
        r'「([^」]+)」',          # Text in Japanese brackets
    ]

    for pattern in patterns:
        match = re.search(pattern, column_name)
        if match:
            return match.group(1).strip()

    # Fallback: extract key descriptive words
    # Remove common prefixes and suffixes
    cleaned = column_name
    prefixes = ['针对该品牌', '针对该车型', '针对该门店', '该门店', '该品牌', '该车型', '请对', '您对']
    suffixes = ['方面的评分', '方面您会打多少分呢？', '方面您会打多少分', '的评分', '评分', '打多少分']

    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]

    for suffix in suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]

    return cleaned.strip() or column_name


def _determine_related_entity(column_name: str) -> Optional[str]:
    """
    Determines which entity type a rating column is related to.
    """
    column_lower = column_name.lower()

    if '品牌' in column_lower or 'brand' in column_lower:
        return 'Brand'
    elif '车型' in column_lower or 'model' in column_lower:
        return 'Model'
    elif '门店' in column_lower or 'store' in column_lower:
        return 'Store'

    return None


def extract_aspects(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Extracts aspect names from rating column headers using classification results.

    Args:
        tool_context: ADK ToolContext

    Returns:
        Dictionary with aspects list and column-to-aspect mapping
    """
    classifications = get_state_value(tool_context, COLUMN_CLASSIFICATION, {})

    if not classifications:
        return tool_error("No column classification found. Run classify_all_columns first.")

    aspects = []
    column_to_aspect = {}

    aspect_counter = 0

    for column_name, info in classifications.items():
        if info.get("category") == "rating":
            aspect_name = _extract_aspect_name(column_name)
            related_entity = info.get("related_to") or _determine_related_entity(column_name)

            aspect_id = f"ASP_{aspect_counter}"
            aspect_counter += 1

            aspects.append({
                "aspect_id": aspect_id,
                "aspect_name": aspect_name,
                "related_entity": related_entity,
                "source_column": column_name
            })

            column_to_aspect[column_name] = aspect_id

    # Store in state
    set_state_value(tool_context, EXTRACTED_ASPECTS, aspects)
    set_state_value(tool_context, COLUMN_TO_ASPECT, column_to_aspect)

    logger.info(f"Extracted {len(aspects)} aspects from rating columns")

    return tool_success("aspects_result", {
        "aspects": aspects,
        "column_to_aspect": column_to_aspect,
        "count": len(aspects)
    })


def extract_ratings(
    file_path: str,
    id_column: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Creates rating fact table from survey data.

    Args:
        file_path: Path to the survey file
        id_column: Column containing respondent IDs
        tool_context: ADK ToolContext

    Returns:
        Dictionary with rating data info
    """
    classifications = get_state_value(tool_context, COLUMN_CLASSIFICATION, {})
    column_to_aspect = get_state_value(tool_context, COLUMN_TO_ASPECT, {})

    if not classifications:
        return tool_error("No column classification found. Run classify_all_columns first.")

    if not column_to_aspect:
        return tool_error("No aspect mapping found. Run extract_aspects first.")

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

    # Find rating columns
    rating_columns = [col for col, info in classifications.items()
                     if info.get("category") == "rating"]

    # Build rating data
    rating_data = []

    for _, row in df.iterrows():
        respondent_id = row[id_column]

        for col in rating_columns:
            if col not in df.columns:
                continue

            value = row[col]
            if pd.notna(value):
                # Try to extract numeric score
                score = None
                try:
                    # Handle various formats
                    if isinstance(value, (int, float)):
                        score = float(value)
                    else:
                        # Try to extract number from text
                        value_str = str(value).strip()
                        # Handle "非常好" -> 10, "好" -> 8, etc.
                        text_to_score = {
                            '非常好': 10, '很好': 9, '好': 8, '较好': 7,
                            '一般': 6, '中等': 5, '较差': 4, '差': 3,
                            '很差': 2, '非常差': 1
                        }
                        if value_str in text_to_score:
                            score = text_to_score[value_str]
                        else:
                            # Try to parse as number
                            numbers = re.findall(r'[\d.]+', value_str)
                            if numbers:
                                score = float(numbers[0])
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse rating value: {value} for column {col}")
                    continue

                if score is not None:
                    aspect_id = column_to_aspect.get(col)
                    if aspect_id:
                        rating_data.append({
                            "respondent_id": respondent_id,
                            "aspect_id": aspect_id,
                            "score": score,
                            "raw_value": str(value)
                        })

    # Create DataFrame
    ratings_df = pd.DataFrame(rating_data)

    # Store in state
    set_state_value(tool_context, RATING_DATA, rating_data)

    logger.info(f"Extracted {len(rating_data)} rating records")

    return tool_success("ratings_result", {
        "row_count": len(rating_data),
        "columns": list(ratings_df.columns) if len(ratings_df) > 0 else [],
        "sample": rating_data[:5] if rating_data else [],
        "aspect_summary": _get_aspect_summary(rating_data)
    })


def _get_aspect_summary(rating_data: List[Dict]) -> Dict[str, Any]:
    """Get summary statistics per aspect."""
    if not rating_data:
        return {}

    df = pd.DataFrame(rating_data)
    summary = df.groupby('aspect_id').agg({
        'score': ['count', 'mean', 'min', 'max']
    }).round(2)

    result = {}
    for aspect_id in summary.index:
        result[aspect_id] = {
            'count': int(summary.loc[aspect_id, ('score', 'count')]),
            'mean': float(summary.loc[aspect_id, ('score', 'mean')]),
            'min': float(summary.loc[aspect_id, ('score', 'min')]),
            'max': float(summary.loc[aspect_id, ('score', 'max')])
        }

    return result


def save_ratings_to_csv(
    output_dir: str,
    tool_context: ToolContext,
    prefix: str = "survey_parsed"
) -> Dict[str, Any]:
    """
    Saves extracted aspects and ratings to CSV files.

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

    # Save aspects file
    aspects = get_state_value(tool_context, EXTRACTED_ASPECTS, [])

    if aspects:
        df = pd.DataFrame(aspects)
        file_name = f"{prefix}_aspects.csv"
        file_path = full_output_dir / file_name
        df.to_csv(file_path, index=False, encoding='utf-8')
        saved_files.append(str(file_path))
        logger.info(f"Saved {len(aspects)} aspects to {file_path}")

    # Save ratings file
    rating_data = get_state_value(tool_context, RATING_DATA, [])

    if rating_data:
        df = pd.DataFrame(rating_data)
        file_name = f"{prefix}_ratings.csv"
        file_path = full_output_dir / file_name
        df.to_csv(file_path, index=False, encoding='utf-8')
        saved_files.append(str(file_path))
        logger.info(f"Saved {len(rating_data)} rating records to {file_path}")

    return tool_success("saved_files", {
        "files": saved_files,
        "output_dir": str(full_output_dir)
    })


def get_extracted_aspects(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Get extracted aspects from state.

    Args:
        tool_context: ADK ToolContext

    Returns:
        Extracted aspects
    """
    aspects = get_state_value(tool_context, EXTRACTED_ASPECTS, [])

    if not aspects:
        return tool_error("No extracted aspects found. Run extract_aspects first.")

    return tool_success("aspects", aspects)


def get_ratings_by_aspect(
    aspect_id: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Get ratings for a specific aspect.

    Args:
        aspect_id: The aspect ID to filter by
        tool_context: ADK ToolContext

    Returns:
        Ratings for the specified aspect
    """
    rating_data = get_state_value(tool_context, RATING_DATA, [])

    if not rating_data:
        return tool_error("No rating data found. Run extract_ratings first.")

    filtered = [r for r in rating_data if r.get('aspect_id') == aspect_id]

    if not filtered:
        return tool_error(f"No ratings found for aspect: {aspect_id}")

    df = pd.DataFrame(filtered)
    stats = {
        'count': len(filtered),
        'mean': float(df['score'].mean()),
        'std': float(df['score'].std()),
        'min': float(df['score'].min()),
        'max': float(df['score'].max())
    }

    return tool_success("ratings", {
        "aspect_id": aspect_id,
        "ratings": filtered,
        "statistics": stats
    })


def get_ratings_by_respondent(
    respondent_id: Any,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Get all ratings for a specific respondent.

    Args:
        respondent_id: The respondent ID to filter by
        tool_context: ADK ToolContext

    Returns:
        Ratings for the specified respondent
    """
    rating_data = get_state_value(tool_context, RATING_DATA, [])

    if not rating_data:
        return tool_error("No rating data found. Run extract_ratings first.")

    # Handle type conversion for comparison
    filtered = [r for r in rating_data if str(r.get('respondent_id')) == str(respondent_id)]

    if not filtered:
        return tool_error(f"No ratings found for respondent: {respondent_id}")

    return tool_success("ratings", {
        "respondent_id": respondent_id,
        "ratings": filtered,
        "count": len(filtered)
    })


# Export tools list for agent registration
SURVEY_RATING_EXTRACTION_TOOLS = [
    extract_aspects,
    extract_ratings,
    save_ratings_to_csv,
    get_extracted_aspects,
    get_ratings_by_aspect,
    get_ratings_by_respondent
]

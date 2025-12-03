"""
Data preprocessing tools for Agentic KG.

Tools for analyzing and transforming data files for knowledge graph import.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext

from .common import tool_success, tool_error
from ..config import get_neo4j_import_dir


def _convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Any object that may contain numpy types

    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

# State keys
DATA_FORMAT_ANALYSIS = "data_format_analysis"
TRANSFORMED_FILES = "transformed_files"
PREPROCESSING_PLAN = "preprocessing_plan"

# New ETL state keys
SURVEY_FORMAT_ANALYSIS = "survey_format_analysis"
COLUMN_CLASSIFICATION = "column_classification"
EXTRACTED_ENTITIES = "extracted_entities"
EXTRACTED_RATINGS = "extracted_ratings"
EXTRACTED_OPINIONS = "extracted_opinions"
SPLIT_MULTI_VALUES = "split_multi_values"
NORMALIZED_FILES = "normalized_files"

# Common survey data patterns
SURVEY_NULL_VALUES = ["(跳过)", "(空)", "(无)", "跳过", "空"]
SURVEY_MULTI_VALUE_DELIMITER = "┋"
RATING_PATTERN = "打多少分"
POSITIVE_OPINION_PATTERN = "优秀点"
NEGATIVE_OPINION_PATTERN = "劣势点"


def analyze_data_format(
    file_path: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Analyze a CSV file to determine its data format.

    Detects whether the file is in:
    - Row format: Each row represents an entity (standard format)
    - Wide format: Column headers contain entity information (pivot table style)

    Args:
        file_path: Path to the CSV file relative to import directory
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with format analysis including:
        - format_type: "row" or "wide"
        - column_count: Number of columns
        - row_count: Number of rows
        - column_names: List of column names
        - sample_values: Sample values from first few rows
        - recommendation: Suggested preprocessing action
    """
    import_dir = get_neo4j_import_dir()
    if not import_dir:
        return tool_error("NEO4J_IMPORT_DIR not configured")

    full_path = Path(import_dir) / file_path
    if not full_path.exists():
        return tool_error(f"File not found: {file_path}")

    try:
        df = pd.read_csv(full_path)

        # Analyze structure
        column_count = len(df.columns)
        row_count = len(df)
        column_names = list(df.columns)

        # Check if it looks like a wide format (pivot table)
        # Indicators: many numeric columns with similar patterns, first column is categorical
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns

        # Heuristics for wide format detection
        is_wide_format = False
        wide_indicators = []

        # Check 1: Many columns with similar data types
        if len(numeric_cols) > 5 and len(numeric_cols) / column_count > 0.7:
            is_wide_format = True
            wide_indicators.append(f"Many numeric columns ({len(numeric_cols)}/{column_count})")

        # Check 2: First column looks like an identifier/category
        first_col = df.columns[0]
        first_col_unique = df[first_col].nunique()
        if first_col_unique == len(df) or first_col_unique / len(df) > 0.9:
            wide_indicators.append(f"First column '{first_col}' has unique values ({first_col_unique}/{len(df)})")

        # Check 3: Column names have similar patterns (e.g., all contain spaces, all Chinese)
        has_chinese = any('\u4e00' <= c <= '\u9fff' for col in column_names[1:] for c in col)
        if has_chinese and len(column_names) > 3:
            wide_indicators.append("Column names contain Chinese characters (likely entity names)")
            is_wide_format = True

        # Generate recommendation
        if is_wide_format:
            recommendation = "Transform from wide to long format: Extract column headers as entities"
        else:
            recommendation = "No transformation needed: Standard row format"

        analysis = {
            "file_path": file_path,
            "format_type": "wide" if is_wide_format else "row",
            "column_count": int(column_count),
            "row_count": int(row_count),
            "column_names": column_names[:20] if len(column_names) > 20 else column_names,
            "numeric_columns": list(numeric_cols[:10]),
            "non_numeric_columns": list(non_numeric_cols[:10]),
            "wide_format_indicators": wide_indicators,
            "recommendation": recommendation,
            "sample_first_column": df.iloc[:5, 0].tolist() if len(df) > 0 else [],
        }

        # Convert numpy types for JSON serialization
        analysis = _convert_numpy_types(analysis)

        # Save to state
        if DATA_FORMAT_ANALYSIS not in tool_context.state:
            tool_context.state[DATA_FORMAT_ANALYSIS] = {}
        tool_context.state[DATA_FORMAT_ANALYSIS][file_path] = analysis

        return tool_success(DATA_FORMAT_ANALYSIS, analysis)

    except Exception as e:
        return tool_error(f"Error analyzing file: {str(e)}")


def transform_wide_to_long(
    file_path: str,
    id_column: str,
    value_column_name: str,
    entity_column_name: str,
    output_prefix: str,
    tool_context: ToolContext,
    skip_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Transform a wide-format CSV to long format, creating multiple output files.

    This transforms pivot-table style data where:
    - Rows represent one entity type (e.g., attributes)
    - Column headers represent another entity type (e.g., brands)
    - Cell values represent relationships/scores

    Args:
        file_path: Path to the wide-format CSV file
        id_column: Column name containing row identifiers (e.g., attribute names)
        value_column_name: Name for the value column in output (e.g., "attention_score")
        entity_column_name: Name for the extracted entity column (e.g., "brand_powertrain")
        output_prefix: Prefix for output file names
        tool_context: ADK ToolContext for state management
        skip_columns: Optional list of columns to skip in transformation

    Returns:
        Dictionary with transformation results including paths to generated files:
        - entities_file: File with unique entities from column headers
        - ids_file: File with unique IDs from the id_column
        - relationships_file: File with long-format relationships

    Example:
        Input (wide format):
        | attribute | 品牌A | 品牌B |
        |-----------|-------|-------|
        | 性价比    | 7.5   | 8.0   |
        | 外观      | 7.8   | 7.2   |

        Output:
        1. {prefix}_attributes.csv: attribute (unique attributes)
        2. {prefix}_brands.csv: brand_name (unique brands from headers)
        3. {prefix}_scores.csv: attribute, brand_name, score (relationships)
    """
    import_dir = get_neo4j_import_dir()
    if not import_dir:
        return tool_error("NEO4J_IMPORT_DIR not configured")

    full_path = Path(import_dir) / file_path
    if not full_path.exists():
        return tool_error(f"File not found: {file_path}")

    try:
        df = pd.read_csv(full_path)

        # Identify columns to melt
        skip_cols = set(skip_columns or [])
        skip_cols.add(id_column)

        value_columns = [col for col in df.columns if col not in skip_cols]

        if not value_columns:
            return tool_error("No value columns found after excluding id_column and skip_columns")

        output_dir = Path(import_dir)

        # 1. Create IDs file (unique values from id_column)
        ids_df = pd.DataFrame({id_column: df[id_column].unique()})
        ids_file = f"{output_prefix}_ids.csv"
        ids_df.to_csv(output_dir / ids_file, index=False)

        # 2. Create entities file (from column headers)
        entities_data = []
        for col in value_columns:
            # Try to parse compound names (e.g., "品牌 动力类型")
            parts = col.split(" ", 1)
            if len(parts) == 2:
                entities_data.append({
                    entity_column_name: col,
                    "primary_name": parts[0],
                    "secondary_name": parts[1]
                })
            else:
                entities_data.append({
                    entity_column_name: col,
                    "primary_name": col,
                    "secondary_name": ""
                })

        entities_df = pd.DataFrame(entities_data)
        entities_file = f"{output_prefix}_entities.csv"
        entities_df.to_csv(output_dir / entities_file, index=False)

        # 3. Create long-format relationships file
        long_df = df.melt(
            id_vars=[id_column],
            value_vars=value_columns,
            var_name=entity_column_name,
            value_name=value_column_name
        )
        # Remove rows with null values
        long_df = long_df.dropna(subset=[value_column_name])

        relationships_file = f"{output_prefix}_relationships.csv"
        long_df.to_csv(output_dir / relationships_file, index=False)

        # Save transformation result to state
        result = {
            "source_file": file_path,
            "ids_file": ids_file,
            "entities_file": entities_file,
            "relationships_file": relationships_file,
            "ids_count": int(len(ids_df)),
            "entities_count": int(len(entities_df)),
            "relationships_count": int(len(long_df)),
            "id_column": id_column,
            "entity_column": entity_column_name,
            "value_column": value_column_name,
        }

        # Convert numpy types for JSON serialization
        result = _convert_numpy_types(result)

        if TRANSFORMED_FILES not in tool_context.state:
            tool_context.state[TRANSFORMED_FILES] = {}
        tool_context.state[TRANSFORMED_FILES][file_path] = result

        return tool_success(TRANSFORMED_FILES, result)

    except Exception as e:
        return tool_error(f"Error transforming file: {str(e)}")


def get_transformed_files(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get the list of transformed files from preprocessing.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with transformed files information
    """
    if TRANSFORMED_FILES not in tool_context.state:
        return tool_error("No files have been transformed yet")

    return tool_success(TRANSFORMED_FILES, tool_context.state[TRANSFORMED_FILES])


def approve_preprocessing(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Approve the preprocessing results and update approved files list.

    Collects output files from ALL ETL operations:
    - TRANSFORMED_FILES (from transform_wide_to_long)
    - EXTRACTED_ENTITIES (from extract_entities)
    - EXTRACTED_RATINGS (from extract_ratings)
    - EXTRACTED_OPINIONS (from extract_opinion_pairs)
    - NORMALIZED_FILES (from normalize_values)
    - SPLIT_MULTI_VALUES (from split_multi_value_column)

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with updated approved files list
    """
    # Collect all output files from various ETL operations
    all_output_files = set()

    # 1. From TRANSFORMED_FILES (transform_wide_to_long)
    if TRANSFORMED_FILES in tool_context.state:
        for result in tool_context.state[TRANSFORMED_FILES].values():
            if isinstance(result, dict):
                if "ids_file" in result:
                    all_output_files.add(result["ids_file"])
                if "entities_file" in result:
                    all_output_files.add(result["entities_file"])
                if "relationships_file" in result:
                    all_output_files.add(result["relationships_file"])

    # 2. From EXTRACTED_ENTITIES (extract_entities)
    if EXTRACTED_ENTITIES in tool_context.state:
        for result in tool_context.state[EXTRACTED_ENTITIES].values():
            if isinstance(result, dict) and "output_file" in result:
                all_output_files.add(result["output_file"])

    # 3. From EXTRACTED_RATINGS (extract_ratings)
    if EXTRACTED_RATINGS in tool_context.state:
        for result in tool_context.state[EXTRACTED_RATINGS].values():
            if isinstance(result, dict) and "output_file" in result:
                all_output_files.add(result["output_file"])

    # 4. From EXTRACTED_OPINIONS (extract_opinion_pairs)
    if EXTRACTED_OPINIONS in tool_context.state:
        for result in tool_context.state[EXTRACTED_OPINIONS].values():
            if isinstance(result, dict) and "output_file" in result:
                all_output_files.add(result["output_file"])

    # 5. From NORMALIZED_FILES (normalize_values)
    if NORMALIZED_FILES in tool_context.state:
        for result in tool_context.state[NORMALIZED_FILES].values():
            if isinstance(result, dict) and "output_file" in result:
                all_output_files.add(result["output_file"])

    # 6. From SPLIT_MULTI_VALUES (split_multi_value_column)
    if SPLIT_MULTI_VALUES in tool_context.state:
        for result in tool_context.state[SPLIT_MULTI_VALUES].values():
            if isinstance(result, dict) and "output_file" in result:
                all_output_files.add(result["output_file"])

    # Check if any output files were collected
    if not all_output_files:
        return tool_error(
            "No preprocessing output files found. "
            "Run preprocessing tools (extract_entities, extract_ratings, normalize_values, etc.) first."
        )

    # Update approved_files with all collected output files
    new_approved_files = sorted(list(all_output_files))
    tool_context.state["approved_files"] = new_approved_files
    tool_context.state["preprocessing_complete"] = True

    return tool_success("approved_files", {
        "files": new_approved_files,
        "count": len(new_approved_files),
        "message": f"Approved {len(new_approved_files)} preprocessed files"
    })


# ============================================================================
# New ETL Tools for Survey and Complex Data Formats
# ============================================================================

def analyze_survey_format(
    file_path: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Analyze a CSV file to detect survey-specific data patterns.

    Detects:
    - Column categories (ID, demographic, entity, rating, multi-value, opinion, media)
    - Multi-value delimiters (e.g., "┋")
    - Special null values (e.g., "(跳过)", "(空)")
    - Rating scale patterns
    - Opinion pair columns (优秀点/劣势点)

    Args:
        file_path: Path to the CSV file relative to import directory
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with detailed survey format analysis
    """
    import_dir = get_neo4j_import_dir()
    if not import_dir:
        return tool_error("NEO4J_IMPORT_DIR not configured")

    full_path = Path(import_dir) / file_path
    if not full_path.exists():
        return tool_error(f"File not found: {file_path}")

    try:
        df = pd.read_csv(full_path)
        column_names = list(df.columns)

        # Detect multi-value delimiter
        detected_delimiters = []
        multi_value_columns = []
        for col in column_names:
            if df[col].dtype == object:
                sample_values = df[col].dropna().head(20).astype(str)
                for val in sample_values:
                    if SURVEY_MULTI_VALUE_DELIMITER in val:
                        if col not in multi_value_columns:
                            multi_value_columns.append(col)
                        if SURVEY_MULTI_VALUE_DELIMITER not in detected_delimiters:
                            detected_delimiters.append(SURVEY_MULTI_VALUE_DELIMITER)
                        break

        # Detect null markers
        detected_null_markers = []
        for null_val in SURVEY_NULL_VALUES:
            for col in column_names:
                if df[col].dtype == object:
                    if df[col].astype(str).str.contains(null_val, na=False).any():
                        if null_val not in detected_null_markers:
                            detected_null_markers.append(null_val)
                        break

        # Detect rating columns (by pattern in column name)
        rating_columns = [col for col in column_names if RATING_PATTERN in col]

        # Detect opinion pair columns
        positive_opinion_columns = [col for col in column_names if POSITIVE_OPINION_PATTERN in col]
        negative_opinion_columns = [col for col in column_names if NEGATIVE_OPINION_PATTERN in col]

        # Detect entity columns (common patterns)
        entity_patterns = ["品牌", "车型", "门店", "配置"]
        entity_columns = []
        for col in column_names:
            for pattern in entity_patterns:
                if pattern in col:
                    entity_columns.append(col)
                    break

        # Detect demographic columns
        demographic_patterns = ["年龄", "性别", "家庭", "拥有"]
        demographic_columns = []
        for col in column_names:
            for pattern in demographic_patterns:
                if pattern in col:
                    demographic_columns.append(col)
                    break

        # Detect media/URL columns
        media_columns = []
        for col in column_names:
            if "图片" in col or "视频" in col or "资料" in col:
                media_columns.append(col)

        # Determine format type
        is_survey_format = (
            len(rating_columns) > 0 or
            len(positive_opinion_columns) > 0 or
            len(multi_value_columns) > 0 or
            len(entity_columns) > 0
        )

        # Find potential ID column
        id_column_candidates = []
        for col in column_names[:3]:  # Check first 3 columns
            if df[col].nunique() == len(df) or "序号" in col or "id" in col.lower():
                id_column_candidates.append(col)

        analysis = {
            "file_path": file_path,
            "format_type": "survey" if is_survey_format else "unknown",
            "column_count": int(len(column_names)),
            "row_count": int(len(df)),
            "id_column_candidates": id_column_candidates,
            "detected_delimiters": detected_delimiters,
            "detected_null_markers": detected_null_markers,
            "rating_columns": rating_columns[:10],  # Limit output
            "rating_columns_count": int(len(rating_columns)),
            "multi_value_columns": multi_value_columns[:10],
            "multi_value_columns_count": int(len(multi_value_columns)),
            "positive_opinion_columns": positive_opinion_columns[:5],
            "negative_opinion_columns": negative_opinion_columns[:5],
            "opinion_pairs_count": int(min(len(positive_opinion_columns), len(negative_opinion_columns))),
            "entity_columns": entity_columns,
            "demographic_columns": demographic_columns,
            "media_columns": media_columns[:5],
            "recommendation": _generate_survey_recommendation(
                is_survey_format, rating_columns, multi_value_columns,
                positive_opinion_columns, entity_columns
            ),
        }

        # Convert numpy types for JSON serialization
        analysis = _convert_numpy_types(analysis)

        # Save to state
        if SURVEY_FORMAT_ANALYSIS not in tool_context.state:
            tool_context.state[SURVEY_FORMAT_ANALYSIS] = {}
        tool_context.state[SURVEY_FORMAT_ANALYSIS][file_path] = analysis

        return tool_success(SURVEY_FORMAT_ANALYSIS, analysis)

    except Exception as e:
        return tool_error(f"Error analyzing survey format: {str(e)}")


def _generate_survey_recommendation(
    is_survey: bool,
    rating_cols: List[str],
    multi_value_cols: List[str],
    opinion_cols: List[str],
    entity_cols: List[str]
) -> str:
    """Generate preprocessing recommendation based on detected patterns."""
    if not is_survey:
        return "Not a survey format. Consider using analyze_data_format instead."

    recommendations = []
    if entity_cols:
        recommendations.append(f"Extract entities from: {', '.join(entity_cols[:3])}")
    if rating_cols:
        recommendations.append(f"Extract {len(rating_cols)} rating columns to long format")
    if multi_value_cols:
        recommendations.append(f"Split {len(multi_value_cols)} multi-value columns")
    if opinion_cols:
        recommendations.append(f"Extract {len(opinion_cols)} opinion pair columns")

    return "; ".join(recommendations) if recommendations else "Survey format detected but no specific transformations identified"


def classify_columns(
    file_path: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Classify columns in a survey file into semantic categories.

    Categories:
    - id: Unique record identifiers
    - demographic: Respondent attributes (age, gender, family)
    - entity: References to entities (brand, model, store)
    - rating: Numeric rating scores (1-9 scale)
    - multi_value: Columns with delimiter-separated values
    - opinion_positive: Positive opinion text columns
    - opinion_negative: Negative opinion text columns
    - media: URLs to uploaded media
    - metadata: Time, sequence, or system columns
    - other: Unclassified columns

    Args:
        file_path: Path to the CSV file
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary mapping column names to categories with detection confidence
    """
    import_dir = get_neo4j_import_dir()
    if not import_dir:
        return tool_error("NEO4J_IMPORT_DIR not configured")

    full_path = Path(import_dir) / file_path
    if not full_path.exists():
        return tool_error(f"File not found: {file_path}")

    try:
        df = pd.read_csv(full_path)
        column_names = list(df.columns)

        classifications = {}
        category_summary = {
            "id": [],
            "demographic": [],
            "entity": [],
            "rating": [],
            "multi_value": [],
            "opinion_positive": [],
            "opinion_negative": [],
            "media": [],
            "metadata": [],
            "other": []
        }

        for col in column_names:
            category = "other"
            confidence = 0.5

            # Check for ID column
            if col == "序号" or "id" in col.lower():
                if df[col].nunique() == len(df):
                    category = "id"
                    confidence = 1.0

            # Check for metadata
            elif "时间" in col or "所用时间" in col:
                category = "metadata"
                confidence = 0.9

            # Check for demographic
            elif any(p in col for p in ["年龄", "性别", "家庭", "拥有"]):
                category = "demographic"
                confidence = 0.9

            # Check for entity columns
            elif any(p in col for p in ["品牌", "车型", "门店", "配置"]):
                category = "entity"
                confidence = 0.9

            # Check for rating columns
            elif RATING_PATTERN in col:
                category = "rating"
                confidence = 0.95

            # Check for opinion columns
            elif POSITIVE_OPINION_PATTERN in col:
                category = "opinion_positive"
                confidence = 0.95
            elif NEGATIVE_OPINION_PATTERN in col:
                category = "opinion_negative"
                confidence = 0.95

            # Check for media columns
            elif any(p in col for p in ["图片", "视频", "资料", "上传"]):
                category = "media"
                confidence = 0.9

            # Check for multi-value columns (by content)
            elif df[col].dtype == object:
                sample_values = df[col].dropna().head(20).astype(str)
                has_delimiter = any(SURVEY_MULTI_VALUE_DELIMITER in str(v) for v in sample_values)
                if has_delimiter:
                    category = "multi_value"
                    confidence = 0.85

            classifications[col] = {
                "category": category,
                "confidence": confidence
            }
            category_summary[category].append(col)

        result = {
            "file_path": file_path,
            "total_columns": int(len(column_names)),
            "classifications": classifications,
            "category_summary": {k: v for k, v in category_summary.items() if v},
            "category_counts": {k: int(len(v)) for k, v in category_summary.items() if v},
        }

        # Convert numpy types for JSON serialization
        result = _convert_numpy_types(result)

        # Save to state
        if COLUMN_CLASSIFICATION not in tool_context.state:
            tool_context.state[COLUMN_CLASSIFICATION] = {}
        tool_context.state[COLUMN_CLASSIFICATION][file_path] = result

        return tool_success(COLUMN_CLASSIFICATION, result)

    except Exception as e:
        return tool_error(f"Error classifying columns: {str(e)}")


def normalize_values(
    file_path: str,
    output_prefix: str,
    tool_context: ToolContext,
    null_values: Optional[List[str]] = None,
    replacement: str = "",
    columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Normalize special values in a CSV file.

    Handles survey-specific null markers like "(跳过)", "(空)".

    Args:
        file_path: Source CSV file path
        output_prefix: Prefix for output file
        tool_context: ADK ToolContext
        null_values: List of values to treat as null (default: survey null values)
        replacement: Value to replace nulls with (default: empty string)
        columns: Specific columns to normalize (None = all columns)

    Returns:
        Dictionary with normalization results and output file path
    """
    import_dir = get_neo4j_import_dir()
    if not import_dir:
        return tool_error("NEO4J_IMPORT_DIR not configured")

    full_path = Path(import_dir) / file_path
    if not full_path.exists():
        return tool_error(f"File not found: {file_path}")

    try:
        df = pd.read_csv(full_path)
        null_vals = null_values or SURVEY_NULL_VALUES
        target_cols = columns or list(df.columns)

        # Count replacements
        replacement_counts = {}
        total_replacements = 0

        for col in target_cols:
            if col in df.columns and df[col].dtype == object:
                col_count = 0
                for null_val in null_vals:
                    mask = df[col].astype(str) == null_val
                    col_count += mask.sum()
                    df.loc[mask, col] = replacement

                if col_count > 0:
                    replacement_counts[col] = col_count
                    total_replacements += col_count

        # Save normalized file
        output_file = f"{output_prefix}_normalized.csv"
        output_path = Path(import_dir) / output_file
        df.to_csv(output_path, index=False)

        result = {
            "source_file": file_path,
            "output_file": output_file,
            "null_values_replaced": null_vals,
            "total_replacements": int(total_replacements),
            "replacement_counts_by_column": {k: int(v) for k, v in replacement_counts.items()},
            "rows": int(len(df)),
            "columns_processed": int(len(target_cols)),
        }

        # Convert numpy types for JSON serialization
        result = _convert_numpy_types(result)

        # Save to state
        if NORMALIZED_FILES not in tool_context.state:
            tool_context.state[NORMALIZED_FILES] = {}
        tool_context.state[NORMALIZED_FILES][file_path] = result

        return tool_success(NORMALIZED_FILES, result)

    except Exception as e:
        return tool_error(f"Error normalizing values: {str(e)}")


def split_multi_value_column(
    file_path: str,
    column_name: str,
    output_prefix: str,
    tool_context: ToolContext,
    delimiter: str = "┋",
    id_column: str = "序号",
    include_empty: bool = False
) -> Dict[str, Any]:
    """
    Split a multi-value column into separate rows.

    Transforms data like:
        序号 | 体验方面
        74   | 用料及工艺┋色彩与氛围

    Into:
        序号 | 体验方面
        74   | 用料及工艺
        74   | 色彩与氛围

    Args:
        file_path: Source CSV file path
        column_name: Column containing multi-values
        output_prefix: Prefix for output file
        tool_context: ADK ToolContext
        delimiter: Delimiter string (default: "┋")
        id_column: Name of the ID column to preserve
        include_empty: Whether to include empty values after split

    Returns:
        Dictionary with transformation results and output file path
    """
    import_dir = get_neo4j_import_dir()
    if not import_dir:
        return tool_error("NEO4J_IMPORT_DIR not configured")

    full_path = Path(import_dir) / file_path
    if not full_path.exists():
        return tool_error(f"File not found: {file_path}")

    try:
        df = pd.read_csv(full_path)

        if column_name not in df.columns:
            return tool_error(f"Column not found: {column_name}")
        if id_column not in df.columns:
            return tool_error(f"ID column not found: {id_column}")

        # Select only the columns we need
        result_df = df[[id_column, column_name]].copy()

        # Split the multi-value column
        result_df[column_name] = result_df[column_name].astype(str)
        result_df = result_df.assign(
            **{column_name: result_df[column_name].str.split(delimiter)}
        ).explode(column_name)

        # Clean up
        result_df[column_name] = result_df[column_name].str.strip()

        if not include_empty:
            result_df = result_df[result_df[column_name] != ""]
            result_df = result_df[result_df[column_name] != "nan"]
            result_df = result_df.dropna(subset=[column_name])

        # Remove duplicates
        result_df = result_df.drop_duplicates()

        # Save output file
        output_file = f"{output_prefix}_{column_name.replace(' ', '_')}_split.csv"
        output_path = Path(import_dir) / output_file
        result_df.to_csv(output_path, index=False)

        result = {
            "source_file": file_path,
            "output_file": output_file,
            "column_split": column_name,
            "delimiter": delimiter,
            "original_rows": int(len(df)),
            "result_rows": int(len(result_df)),
            "unique_values": int(result_df[column_name].nunique()),
        }

        # Convert numpy types for JSON serialization
        result = _convert_numpy_types(result)

        # Save to state
        if SPLIT_MULTI_VALUES not in tool_context.state:
            tool_context.state[SPLIT_MULTI_VALUES] = {}
        tool_context.state[SPLIT_MULTI_VALUES][column_name] = result

        return tool_success(SPLIT_MULTI_VALUES, result)

    except Exception as e:
        return tool_error(f"Error splitting multi-value column: {str(e)}")


def extract_entities(
    file_path: str,
    entity_columns: List[str],
    entity_type: str,
    output_prefix: str,
    tool_context: ToolContext,
    deduplicate: bool = True
) -> Dict[str, Any]:
    """
    Extract unique entities from specified columns.

    Creates a normalized entity file with unique values.

    Args:
        file_path: Source CSV file path
        entity_columns: List of columns containing entity values
        entity_type: Label for the entity type (e.g., "Brand", "Model", "Store")
        output_prefix: Prefix for output file
        tool_context: ADK ToolContext
        deduplicate: Whether to deduplicate entities (default: True)

    Returns:
        Dictionary with entities file path and count
    """
    import_dir = get_neo4j_import_dir()
    if not import_dir:
        return tool_error("NEO4J_IMPORT_DIR not configured")

    full_path = Path(import_dir) / file_path
    if not full_path.exists():
        return tool_error(f"File not found: {file_path}")

    try:
        df = pd.read_csv(full_path)

        # Validate columns exist
        missing_cols = [col for col in entity_columns if col not in df.columns]
        if missing_cols:
            return tool_error(f"Columns not found: {missing_cols}")

        # Extract unique entities from all specified columns
        all_entities = []
        for col in entity_columns:
            values = df[col].dropna().astype(str).unique()
            for val in values:
                if val and val != "nan":
                    all_entities.append({
                        "entity_id": f"{entity_type}_{len(all_entities)}",
                        "entity_type": entity_type,
                        "value": val,
                        "source_column": col
                    })

        entities_df = pd.DataFrame(all_entities)

        if deduplicate:
            # Keep first occurrence, deduplicate by value
            entities_df = entities_df.drop_duplicates(subset=["value"], keep="first")
            # Re-assign entity IDs after deduplication
            entities_df["entity_id"] = [f"{entity_type}_{i}" for i in range(len(entities_df))]

        # Save output file
        output_file = f"{output_prefix}_{entity_type.lower()}_entities.csv"
        output_path = Path(import_dir) / output_file
        entities_df.to_csv(output_path, index=False)

        result = {
            "source_file": file_path,
            "output_file": output_file,
            "entity_type": entity_type,
            "source_columns": entity_columns,
            "total_entities": int(len(entities_df)),
            "unique_values": int(entities_df["value"].nunique()),
            "sample_entities": entities_df["value"].head(10).tolist(),
        }

        # Convert numpy types for JSON serialization
        result = _convert_numpy_types(result)

        # Save to state
        if EXTRACTED_ENTITIES not in tool_context.state:
            tool_context.state[EXTRACTED_ENTITIES] = {}
        tool_context.state[EXTRACTED_ENTITIES][entity_type] = result

        return tool_success(EXTRACTED_ENTITIES, result)

    except Exception as e:
        return tool_error(f"Error extracting entities: {str(e)}")


def extract_ratings(
    file_path: str,
    id_column: str,
    output_prefix: str,
    tool_context: ToolContext,
    rating_columns: Optional[List[str]] = None,
    rating_pattern: str = "打多少分"
) -> Dict[str, Any]:
    """
    Extract and normalize rating columns from survey data.

    Detects rating columns by pattern (e.g., "打多少分") and extracts them
    into a normalized long-format file.

    Args:
        file_path: Source CSV file
        id_column: Respondent ID column
        output_prefix: Prefix for output file
        tool_context: ADK ToolContext
        rating_columns: Explicit list of rating columns (auto-detect if None)
        rating_pattern: Regex pattern to identify rating columns

    Returns:
        Dictionary with extracted ratings in long format
    """
    import_dir = get_neo4j_import_dir()
    if not import_dir:
        return tool_error("NEO4J_IMPORT_DIR not configured")

    full_path = Path(import_dir) / file_path
    if not full_path.exists():
        return tool_error(f"File not found: {file_path}")

    try:
        df = pd.read_csv(full_path)

        if id_column not in df.columns:
            return tool_error(f"ID column not found: {id_column}")

        # Auto-detect rating columns if not provided
        if rating_columns is None:
            rating_columns = [col for col in df.columns if rating_pattern in col]

        if not rating_columns:
            return tool_error(f"No rating columns found matching pattern: {rating_pattern}")

        # Extract aspect names from column names
        # Example: "11、该车型"外观设计"方面您会打多少分呢?" -> "外观设计"
        def extract_aspect(col_name: str) -> str:
            import re
            match = re.search(r'"([^"]+)"', col_name)
            if match:
                return match.group(1)
            # Try to extract question number
            match = re.search(r'^(\d+)、', col_name)
            if match:
                return f"Q{match.group(1)}"
            return col_name[:30]

        # Build long-format ratings
        ratings_data = []
        for _, row in df.iterrows():
            respondent_id = row[id_column]
            for col in rating_columns:
                value = row[col]
                if pd.notna(value):
                    try:
                        score = float(value)
                        ratings_data.append({
                            "respondent_id": respondent_id,
                            "question": col[:50],
                            "aspect": extract_aspect(col),
                            "score": score
                        })
                    except (ValueError, TypeError):
                        pass  # Skip non-numeric values

        ratings_df = pd.DataFrame(ratings_data)

        # Save output file
        output_file = f"{output_prefix}_ratings.csv"
        output_path = Path(import_dir) / output_file
        ratings_df.to_csv(output_path, index=False)

        result = {
            "source_file": file_path,
            "output_file": output_file,
            "id_column": id_column,
            "rating_columns_count": int(len(rating_columns)),
            "total_ratings": int(len(ratings_df)),
            "unique_respondents": int(ratings_df["respondent_id"].nunique()) if len(ratings_df) > 0 else 0,
            "unique_aspects": int(ratings_df["aspect"].nunique()) if len(ratings_df) > 0 else 0,
            "aspects": ratings_df["aspect"].unique().tolist()[:10] if len(ratings_df) > 0 else [],
            "score_stats": {
                "min": float(ratings_df["score"].min()) if len(ratings_df) > 0 else None,
                "max": float(ratings_df["score"].max()) if len(ratings_df) > 0 else None,
                "mean": float(ratings_df["score"].mean()) if len(ratings_df) > 0 else None,
            }
        }

        # Convert numpy types for JSON serialization
        result = _convert_numpy_types(result)

        # Save to state
        if EXTRACTED_RATINGS not in tool_context.state:
            tool_context.state[EXTRACTED_RATINGS] = {}
        tool_context.state[EXTRACTED_RATINGS][file_path] = result

        return tool_success(EXTRACTED_RATINGS, result)

    except Exception as e:
        return tool_error(f"Error extracting ratings: {str(e)}")


def extract_opinion_pairs(
    file_path: str,
    id_column: str,
    output_prefix: str,
    tool_context: ToolContext,
    positive_pattern: str = "优秀点",
    negative_pattern: str = "劣势点"
) -> Dict[str, Any]:
    """
    Extract opinion pairs (strengths/weaknesses) from survey data.

    Transforms paired columns like:
        - "优秀点" -> positive opinions
        - "劣势点" -> negative opinions

    Args:
        file_path: Source CSV file
        id_column: Respondent ID column
        output_prefix: Prefix for output file
        tool_context: ADK ToolContext
        positive_pattern: Pattern to identify positive opinion columns
        negative_pattern: Pattern to identify negative opinion columns

    Returns:
        Dictionary with extracted opinion pairs
    """
    import_dir = get_neo4j_import_dir()
    if not import_dir:
        return tool_error("NEO4J_IMPORT_DIR not configured")

    full_path = Path(import_dir) / file_path
    if not full_path.exists():
        return tool_error(f"File not found: {file_path}")

    try:
        df = pd.read_csv(full_path)

        if id_column not in df.columns:
            return tool_error(f"ID column not found: {id_column}")

        # Find opinion columns
        positive_columns = [col for col in df.columns if positive_pattern in col]
        negative_columns = [col for col in df.columns if negative_pattern in col]

        if not positive_columns and not negative_columns:
            return tool_error(f"No opinion columns found matching patterns: {positive_pattern}, {negative_pattern}")

        # Extract aspect from column name
        def extract_aspect(col_name: str) -> str:
            import re
            # Try to find text in quotes
            match = re.search(r'"([^"]+)"', col_name)
            if match:
                return match.group(1)
            # Try to find text after certain keywords
            for keyword in ["针对", "关于"]:
                if keyword in col_name:
                    parts = col_name.split(keyword)
                    if len(parts) > 1:
                        aspect_part = parts[1].split("方面")[0].strip()
                        if aspect_part:
                            return aspect_part[:20]
            return "general"

        # Build opinions data
        opinions_data = []
        for _, row in df.iterrows():
            respondent_id = row[id_column]

            # Process positive opinions
            for col in positive_columns:
                value = row[col]
                if pd.notna(value) and str(value) not in SURVEY_NULL_VALUES:
                    opinions_data.append({
                        "respondent_id": respondent_id,
                        "aspect": extract_aspect(col),
                        "polarity": "positive",
                        "text": str(value),
                        "source_column": col[:50]
                    })

            # Process negative opinions
            for col in negative_columns:
                value = row[col]
                if pd.notna(value) and str(value) not in SURVEY_NULL_VALUES:
                    opinions_data.append({
                        "respondent_id": respondent_id,
                        "aspect": extract_aspect(col),
                        "polarity": "negative",
                        "text": str(value),
                        "source_column": col[:50]
                    })

        opinions_df = pd.DataFrame(opinions_data)

        # Save output file
        output_file = f"{output_prefix}_opinions.csv"
        output_path = Path(import_dir) / output_file
        opinions_df.to_csv(output_path, index=False)

        result = {
            "source_file": file_path,
            "output_file": output_file,
            "id_column": id_column,
            "positive_columns_count": int(len(positive_columns)),
            "negative_columns_count": int(len(negative_columns)),
            "total_opinions": int(len(opinions_df)),
            "positive_opinions": int(len(opinions_df[opinions_df["polarity"] == "positive"])) if len(opinions_df) > 0 else 0,
            "negative_opinions": int(len(opinions_df[opinions_df["polarity"] == "negative"])) if len(opinions_df) > 0 else 0,
            "unique_respondents": int(opinions_df["respondent_id"].nunique()) if len(opinions_df) > 0 else 0,
            "unique_aspects": opinions_df["aspect"].unique().tolist()[:10] if len(opinions_df) > 0 else [],
        }

        # Convert numpy types for JSON serialization
        result = _convert_numpy_types(result)

        # Save to state
        if EXTRACTED_OPINIONS not in tool_context.state:
            tool_context.state[EXTRACTED_OPINIONS] = {}
        tool_context.state[EXTRACTED_OPINIONS][file_path] = result

        return tool_success(EXTRACTED_OPINIONS, result)

    except Exception as e:
        return tool_error(f"Error extracting opinion pairs: {str(e)}")


def parse_survey_responses(
    file_path: str,
    id_column: str,
    output_prefix: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Parse a complete survey file into multiple normalized files.

    This is a high-level orchestration tool that chains other tools to fully
    parse survey data.

    Creates:
    - {prefix}_respondents.csv: Demographic data per respondent
    - {prefix}_entities.csv: Unique entities (brands, models, stores)
    - {prefix}_ratings.csv: Normalized rating scores
    - {prefix}_opinions.csv: Opinion pairs (positive/negative)

    Args:
        file_path: Source survey CSV file
        id_column: Column with respondent ID
        output_prefix: Prefix for all output files
        tool_context: ADK ToolContext

    Returns:
        Dictionary with all output file paths and transformation summary
    """
    import_dir = get_neo4j_import_dir()
    if not import_dir:
        return tool_error("NEO4J_IMPORT_DIR not configured")

    full_path = Path(import_dir) / file_path
    if not full_path.exists():
        return tool_error(f"File not found: {file_path}")

    try:
        df = pd.read_csv(full_path)
        output_files = []
        processing_results = {}

        # Step 1: Classify columns
        classify_result = classify_columns(file_path, tool_context)
        if classify_result.get("status") == "error":
            return classify_result

        classification = tool_context.state.get(COLUMN_CLASSIFICATION, {}).get(file_path, {})
        category_summary = classification.get("category_summary", {})

        # Step 2: Extract respondent demographics
        demographic_cols = category_summary.get("demographic", [])
        if demographic_cols and id_column in df.columns:
            respondent_df = df[[id_column] + [col for col in demographic_cols if col in df.columns]].copy()
            respondent_file = f"{output_prefix}_respondents.csv"
            respondent_df.to_csv(Path(import_dir) / respondent_file, index=False)
            output_files.append(respondent_file)
            processing_results["respondents"] = {
                "file": respondent_file,
                "count": len(respondent_df),
                "columns": [id_column] + demographic_cols
            }

        # Step 3: Extract entities (brands, models, stores)
        entity_cols = category_summary.get("entity", [])
        for col in entity_cols:
            entity_type = "Entity"
            if "品牌" in col:
                entity_type = "Brand"
            elif "车型" in col:
                entity_type = "Model"
            elif "门店" in col:
                entity_type = "Store"

            entity_result = extract_entities(
                file_path, [col], entity_type, output_prefix, tool_context
            )
            if entity_result.get("status") == "success":
                entity_data = entity_result.get(EXTRACTED_ENTITIES, {})
                output_files.append(entity_data.get("output_file", ""))
                processing_results[f"entities_{entity_type}"] = entity_data

        # Step 4: Extract ratings
        rating_result = extract_ratings(file_path, id_column, output_prefix, tool_context)
        if rating_result.get("status") == "success":
            rating_data = rating_result.get(EXTRACTED_RATINGS, {})
            output_files.append(rating_data.get("output_file", ""))
            processing_results["ratings"] = rating_data

        # Step 5: Extract opinions
        opinion_result = extract_opinion_pairs(file_path, id_column, output_prefix, tool_context)
        if opinion_result.get("status") == "success":
            opinion_data = opinion_result.get(EXTRACTED_OPINIONS, {})
            output_files.append(opinion_data.get("output_file", ""))
            processing_results["opinions"] = opinion_data

        result = {
            "source_file": file_path,
            "output_prefix": output_prefix,
            "output_files": [f for f in output_files if f],
            "processing_results": processing_results,
            "total_files_created": int(len([f for f in output_files if f])),
        }

        # Convert numpy types for JSON serialization
        result = _convert_numpy_types(result)

        return tool_success("survey_parsing", result)

    except Exception as e:
        return tool_error(f"Error parsing survey responses: {str(e)}")

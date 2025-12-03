"""
Data preprocessing tools for Agentic KG.

Tools for analyzing and transforming data files for knowledge graph import.
"""

import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext

from .common import tool_success, tool_error
from ..config import get_neo4j_import_dir

# State keys
DATA_FORMAT_ANALYSIS = "data_format_analysis"
TRANSFORMED_FILES = "transformed_files"
PREPROCESSING_PLAN = "preprocessing_plan"


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
            "column_count": column_count,
            "row_count": row_count,
            "column_names": column_names[:20] if len(column_names) > 20 else column_names,
            "numeric_columns": list(numeric_cols[:10]),
            "non_numeric_columns": list(non_numeric_cols[:10]),
            "wide_format_indicators": wide_indicators,
            "recommendation": recommendation,
            "sample_first_column": df.iloc[:5, 0].tolist() if len(df) > 0 else [],
        }

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
            "ids_count": len(ids_df),
            "entities_count": len(entities_df),
            "relationships_count": len(long_df),
            "id_column": id_column,
            "entity_column": entity_column_name,
            "value_column": value_column_name,
        }

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

    This replaces the original files in approved_files with the transformed files.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with updated approved files list
    """
    if TRANSFORMED_FILES not in tool_context.state:
        return tool_error("No files have been transformed. Run transformations first.")

    if "approved_files" not in tool_context.state:
        return tool_error("No approved files found. Run file suggestion phase first.")

    # Get transformed files
    transformed = tool_context.state[TRANSFORMED_FILES]

    # Build new approved files list
    new_approved_files = []

    for original_file in tool_context.state["approved_files"]:
        if original_file in transformed:
            # Add transformed files instead of original
            transform_result = transformed[original_file]
            new_approved_files.append(transform_result["ids_file"])
            new_approved_files.append(transform_result["entities_file"])
            new_approved_files.append(transform_result["relationships_file"])
        else:
            # Keep original file
            new_approved_files.append(original_file)

    # Update approved files
    tool_context.state["approved_files"] = new_approved_files
    tool_context.state["preprocessing_complete"] = True

    return tool_success("approved_files", new_approved_files)

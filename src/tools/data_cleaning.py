"""
Data Cleaning Tools for Agentic KG.

Tools for analyzing and cleaning raw data files before schema proposal.
This phase performs:
- Format handling (encoding, delimiters)
- Removing meaningless columns and rows
- Cleaning content (empty values, duplicates, invalid data)
"""

import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from google.adk.tools import ToolContext

from .common import tool_success, tool_error, validate_file_path
from ..config import get_neo4j_import_dir

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# State keys for data cleaning
DATA_CLEANING_ANALYSIS_KEY = "data_cleaning_analysis"
CLEANED_FILES_KEY = "cleaned_files"
DATA_CLEANING_COMPLETE_KEY = "data_cleaning_complete"


def _get_import_dir() -> Optional[Path]:
    """Get the import directory as a Path object."""
    import_dir_path = get_neo4j_import_dir()
    if not import_dir_path:
        return None
    return Path(import_dir_path)


def _read_file(file_path: Path) -> pd.DataFrame:
    """Read a file into a DataFrame based on its extension."""
    suffix = file_path.suffix.lower()
    if suffix == '.csv':
        # Try different encodings
        for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-16']:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not read {file_path} with common encodings")
    elif suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def analyze_file_quality(
    file_path: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Analyze data quality of a raw data file.

    Examines a file to identify quality issues such as:
    - Empty or near-empty columns
    - Columns with all identical values
    - Rows with excessive missing data
    - Duplicate rows
    - Encoding issues

    Args:
        file_path: File to analyze, relative to the import directory
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with quality analysis:
        - columns_to_remove: Suggested columns to drop
        - rows_to_remove: Row indices to drop
        - quality_issues: List of detected issues
        - summary: Overall quality summary
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
        df = _read_file(full_path)

        quality_issues = []
        columns_to_remove = []
        rows_to_remove = []

        # 1. Analyze empty columns
        for col in df.columns:
            null_ratio = df[col].isnull().sum() / len(df)
            if null_ratio > 0.95:
                columns_to_remove.append({
                    "column": col,
                    "reason": "more_than_95_percent_empty",
                    "null_ratio": float(null_ratio)
                })
                quality_issues.append(f"Column '{col}' is {null_ratio*100:.1f}% empty")

        # 2. Analyze columns with single unique value
        for col in df.columns:
            if col not in [c["column"] for c in columns_to_remove]:
                unique_values = df[col].dropna().nunique()
                if unique_values == 1 and len(df) > 1:
                    single_value = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                    columns_to_remove.append({
                        "column": col,
                        "reason": "single_unique_value",
                        "value": str(single_value)[:50] if single_value else None
                    })
                    quality_issues.append(f"Column '{col}' has only one unique value")

        # 3. Analyze meaningless column names
        meaningless_patterns = [
            r"^unnamed:\s*\d+$",  # Unnamed columns from CSV
            r"^column\s*\d+$",
            r"^\d+$",  # Pure numeric column names
        ]
        for col in df.columns:
            col_lower = str(col).lower().strip()
            for pattern in meaningless_patterns:
                if re.match(pattern, col_lower):
                    if col not in [c["column"] for c in columns_to_remove]:
                        # Check if this column has meaningful data
                        if df[col].dropna().nunique() <= 3:
                            columns_to_remove.append({
                                "column": col,
                                "reason": "meaningless_column_name",
                                "pattern": pattern
                            })
                            quality_issues.append(f"Column '{col}' has meaningless name and little unique data")
                    break

        # 4. Analyze rows with excessive missing data (>80% missing)
        for idx in df.index:
            row = df.loc[idx]
            null_ratio = row.isnull().sum() / len(row)
            if null_ratio > 0.8:
                rows_to_remove.append({
                    "index": int(idx),
                    "reason": "more_than_80_percent_missing",
                    "null_ratio": float(null_ratio)
                })

        # Limit reported rows to avoid huge output
        if len(rows_to_remove) > 20:
            quality_issues.append(f"Found {len(rows_to_remove)} rows with >80% missing data (showing first 20)")
            rows_to_remove = rows_to_remove[:20]

        # 5. Check for duplicate rows
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            quality_issues.append(f"Found {duplicate_count} duplicate rows")
            duplicate_indices = df[df.duplicated()].index.tolist()[:20]
            for idx in duplicate_indices:
                if idx not in [r["index"] for r in rows_to_remove]:
                    rows_to_remove.append({
                        "index": int(idx),
                        "reason": "duplicate_row"
                    })

        # Summary
        summary = {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "columns_to_remove_count": len(columns_to_remove),
            "rows_to_remove_count": len(rows_to_remove),
            "quality_score": round(1.0 - (len(quality_issues) / max(len(df.columns) + 10, 1)), 2),
            "file_path": file_path,
        }

        result = {
            "file_path": file_path,
            "columns_to_remove": columns_to_remove,
            "rows_to_remove": rows_to_remove,
            "quality_issues": quality_issues,
            "summary": summary,
        }

        # Store analysis in state
        analyses = tool_context.state.get(DATA_CLEANING_ANALYSIS_KEY, {})
        analyses[file_path] = result
        tool_context.state[DATA_CLEANING_ANALYSIS_KEY] = analyses

        return tool_success("file_quality_analysis", result)

    except Exception as e:
        logger.error(f"Error analyzing file {file_path}: {e}")
        return tool_error(f"Error analyzing file {file_path}: {e}")


def clean_file(
    file_path: str,
    tool_context: ToolContext,
    remove_columns: Optional[List[str]] = None,
    remove_rows: Optional[List[int]] = None,
    remove_duplicates: bool = True,
    auto_clean: bool = False,
    replace_values: Optional[Dict[str, Any]] = None,
    replace_with_null: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Clean a raw data file by removing columns, rows, and replacing values.

    Applies cleaning operations to a file and saves the cleaned version.

    Args:
        file_path: File to clean, relative to the import directory
        tool_context: ADK ToolContext for state management
        remove_columns: List of column names to remove
        remove_rows: List of row indices to remove
        remove_duplicates: Whether to remove duplicate rows
        auto_clean: If True, automatically apply all suggested cleanings from analysis
        replace_values: Dict mapping old values to new values (e.g., {"N/A": "", "(跳过)": ""})
        replace_with_null: List of values to replace with null/empty (e.g., ["(跳过)", "N/A", "-"])

    Returns:
        Dictionary with cleaning result:
        - original_shape: Original (rows, columns)
        - cleaned_shape: Cleaned (rows, columns)
        - removed_columns: List of removed columns
        - removed_rows: Number of removed rows
        - values_replaced: Number of values replaced
        - output_path: Path to cleaned file
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
        df = _read_file(full_path)
        original_shape = df.shape

        columns_removed = []
        rows_removed_count = 0
        values_replaced_count = 0

        # Default values to replace with null (common placeholder/skip values)
        default_null_values = [
            "(跳过)", "(跳過)", "跳过", "N/A", "n/a", "NA", "na",
            "-", "--", "---", "无", "無", "null", "NULL", "None",
            "(空)", "空", "(无)", "(none)", "(skip)", "skip",
        ]

        # If auto_clean, get suggestions from previous analysis
        if auto_clean:
            analyses = tool_context.state.get(DATA_CLEANING_ANALYSIS_KEY, {})
            if file_path in analyses:
                analysis = analyses[file_path]
                if remove_columns is None:
                    remove_columns = [c["column"] for c in analysis.get("columns_to_remove", [])]
                if remove_rows is None:
                    remove_rows = [r["index"] for r in analysis.get("rows_to_remove", [])]
            else:
                return tool_error(
                    f"No analysis found for {file_path}. "
                    "Run analyze_file_quality first or provide explicit columns/rows to remove."
                )

        # Remove columns
        if remove_columns:
            existing_cols = [c for c in remove_columns if c in df.columns]
            if existing_cols:
                df = df.drop(columns=existing_cols)
                columns_removed = existing_cols

        # Remove rows
        if remove_rows:
            valid_indices = [i for i in remove_rows if i in df.index]
            if valid_indices:
                df = df.drop(index=valid_indices)
                rows_removed_count = len(valid_indices)

        # Remove duplicates
        if remove_duplicates:
            before_dedup = len(df)
            df = df.drop_duplicates()
            rows_removed_count += before_dedup - len(df)

        # Replace values with null/empty
        # Combine default null values with user-specified values
        null_values_to_replace = set(default_null_values)
        if replace_with_null:
            null_values_to_replace.update(replace_with_null)

        # Auto-clean always replaces default null values
        if auto_clean or replace_with_null is not None:
            for val in null_values_to_replace:
                # Count occurrences before replacement
                mask = df.isin([val])
                count = mask.sum().sum()
                if count > 0:
                    df = df.replace(val, pd.NA)
                    values_replaced_count += count

        # Apply custom value replacements
        if replace_values:
            for old_val, new_val in replace_values.items():
                mask = df.isin([old_val])
                count = mask.sum().sum()
                if count > 0:
                    df = df.replace(old_val, new_val if new_val != "" else pd.NA)
                    values_replaced_count += count

        cleaned_shape = df.shape

        # Generate output filename
        stem = full_path.stem
        suffix = full_path.suffix
        output_filename = f"{stem}_cleaned{suffix}"
        output_path = import_dir / output_filename

        # Save cleaned file
        if suffix == '.csv':
            df.to_csv(output_path, index=False, encoding='utf-8')
        else:
            df.to_excel(output_path, index=False)

        # Update cleaned files state
        cleaned_files = tool_context.state.get(CLEANED_FILES_KEY, {})
        cleaned_files[file_path] = {
            "original_path": file_path,
            "cleaned_path": output_filename,
            "original_shape": original_shape,
            "cleaned_shape": cleaned_shape,
        }
        tool_context.state[CLEANED_FILES_KEY] = cleaned_files

        result = {
            "file_path": file_path,
            "output_path": output_filename,
            "original_shape": {"rows": original_shape[0], "columns": original_shape[1]},
            "cleaned_shape": {"rows": cleaned_shape[0], "columns": cleaned_shape[1]},
            "removed_columns": columns_removed,
            "removed_rows_count": rows_removed_count,
            "values_replaced_count": int(values_replaced_count),
        }

        return tool_success("cleaned_file", result)

    except Exception as e:
        logger.error(f"Error cleaning file {file_path}: {e}")
        return tool_error(f"Error cleaning file {file_path}: {e}")


def get_cleaned_files(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get the list of cleaned files and their paths.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with cleaned files mapping
    """
    cleaned_files = tool_context.state.get(CLEANED_FILES_KEY, {})

    if not cleaned_files:
        return tool_success(CLEANED_FILES_KEY, {
            "message": "No files have been cleaned yet.",
            "files": {}
        })

    return tool_success(CLEANED_FILES_KEY, {
        "files": cleaned_files,
        "count": len(cleaned_files)
    })


def approve_data_cleaning(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Mark data cleaning as complete and approve cleaned files.

    This updates the approved_files state to use cleaned file paths
    instead of original file paths.

    Args:
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with approval status
    """
    cleaned_files = tool_context.state.get(CLEANED_FILES_KEY, {})

    if not cleaned_files:
        return tool_error(
            "No files have been cleaned. Clean at least one file before approving."
        )

    # Update approved_files to use cleaned paths
    approved_files = tool_context.state.get("approved_files", [])
    updated_files = []

    for file_path in approved_files:
        if file_path in cleaned_files:
            # Use cleaned version
            updated_files.append(cleaned_files[file_path]["cleaned_path"])
        else:
            # Keep original if not cleaned
            updated_files.append(file_path)

    # Update state
    tool_context.state["approved_files"] = updated_files
    tool_context.state[DATA_CLEANING_COMPLETE_KEY] = True

    return tool_success("data_cleaning_approved", {
        "original_files": approved_files,
        "updated_files": updated_files,
        "cleaned_count": len(cleaned_files),
        "message": "Data cleaning complete. Approved files updated to use cleaned versions."
    })


def detect_column_types(
    file_path: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Detect semantic types of columns in a file.

    Analyzes column content to categorize columns into:
    - identifier: Unique ID columns
    - selection: Multiple choice selection columns
    - rating: Numeric rating/score columns
    - open_text: Free-form text columns
    - demographic: Demographic information
    - metadata: File/survey metadata

    Args:
        file_path: File to analyze, relative to the import directory
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with column type classifications
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
        df = _read_file(full_path)

        column_types = {}

        for col in df.columns:
            col_data = df[col].dropna()
            col_str = str(col).lower()

            # Initialize classification
            classification = {
                "column": col,
                "dtype": str(df[col].dtype),
                "unique_count": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum()),
                "sample_values": [str(v)[:50] for v in col_data.head(3).tolist()],
            }

            # Detect identifier columns
            if df[col].nunique() == len(df[col].dropna()):
                if "id" in col_str or "序号" in col or "编号" in col_str:
                    classification["type"] = "identifier"
                    classification["confidence"] = "high"
                else:
                    classification["type"] = "identifier"
                    classification["confidence"] = "medium"
            # Detect rating columns
            elif "分" in col_str or "打多少分" in col or "rating" in col_str or "score" in col_str:
                classification["type"] = "rating"
                classification["confidence"] = "high"
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Check if values are in typical rating range
                if col_data.min() >= 0 and col_data.max() <= 10:
                    classification["type"] = "rating"
                    classification["confidence"] = "medium"
                else:
                    classification["type"] = "numeric"
                    classification["confidence"] = "medium"
            # Detect selection columns
            elif "是?" in col or "?" in col_str or "选择" in col_str:
                classification["type"] = "selection"
                classification["confidence"] = "high"
            elif df[col].nunique() < 50 and df[col].nunique() < len(df) * 0.1:
                classification["type"] = "selection"
                classification["confidence"] = "medium"
            # Detect open text columns
            elif pd.api.types.is_string_dtype(df[col]):
                avg_length = col_data.astype(str).str.len().mean()
                if avg_length > 50:
                    classification["type"] = "open_text"
                    classification["confidence"] = "high"
                elif avg_length > 20:
                    classification["type"] = "open_text"
                    classification["confidence"] = "medium"
                else:
                    classification["type"] = "text"
                    classification["confidence"] = "low"
            else:
                classification["type"] = "unknown"
                classification["confidence"] = "low"

            column_types[col] = classification

        # Group by type
        type_groups = {}
        for col, info in column_types.items():
            col_type = info["type"]
            if col_type not in type_groups:
                type_groups[col_type] = []
            type_groups[col_type].append(col)

        result = {
            "file_path": file_path,
            "column_types": column_types,
            "type_groups": type_groups,
            "summary": {
                "total_columns": len(column_types),
                "identifiers": len(type_groups.get("identifier", [])),
                "selections": len(type_groups.get("selection", [])),
                "ratings": len(type_groups.get("rating", [])),
                "open_text": len(type_groups.get("open_text", [])),
            }
        }

        return tool_success("column_types", result)

    except Exception as e:
        logger.error(f"Error detecting column types in {file_path}: {e}")
        return tool_error(f"Error detecting column types in {file_path}: {e}")


def detect_anomalies(
    file_path: str,
    tool_context: ToolContext,
    columns: Optional[List[str]] = None,
    detect_urls: bool = True,
) -> Dict[str, Any]:
    """
    Detect anomalous values in columns, especially mixed data types.

    Identifies columns where the expected data type doesn't match all values,
    such as text values in numeric rating columns, or URLs in content columns.

    Args:
        file_path: File to analyze, relative to the import directory
        tool_context: ADK ToolContext for state management
        columns: Optional list of columns to check. If None, checks all columns.
        detect_urls: Whether to detect URLs in text columns (default True)

    Returns:
        Dictionary with anomaly detection results:
        - anomalies: Dict mapping column names to anomaly details
        - summary: Overall anomaly summary
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
        df = _read_file(full_path)

        anomalies = {}
        columns_to_check = columns if columns else df.columns.tolist()

        for col in columns_to_check:
            if col not in df.columns:
                continue

            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            col_str = str(col).lower()
            col_anomalies = {
                "column": col,
                "total_values": len(col_data),
                "issues": [],
            }

            # Check if this looks like a rating column (contains "分" or "打多少分")
            is_rating_column = (
                "分" in col_str or
                "打多少分" in col or
                "rating" in col_str or
                "score" in col_str
            )

            if is_rating_column:
                # Try to convert to numeric
                numeric_values = pd.to_numeric(col_data, errors='coerce')
                non_numeric_mask = numeric_values.isna() & col_data.notna()
                non_numeric_values = col_data[non_numeric_mask]

                if len(non_numeric_values) > 0:
                    # Found text values in a rating column
                    value_counts = non_numeric_values.value_counts().head(10).to_dict()
                    col_anomalies["issues"].append({
                        "type": "text_in_rating_column",
                        "description": f"Found {len(non_numeric_values)} text values in rating column",
                        "non_numeric_count": int(len(non_numeric_values)),
                        "non_numeric_ratio": float(len(non_numeric_values) / len(col_data)),
                        "sample_values": {str(k): int(v) for k, v in value_counts.items()},
                        "suggestion": "Convert text to numeric or set to null",
                    })

                    # Check for common text-to-rating mappings
                    text_rating_map = {}
                    for val in non_numeric_values.unique():
                        val_str = str(val).strip()
                        # Common Chinese sentiment words
                        if val_str in ["非常好", "很好", "极好", "优秀", "满分"]:
                            text_rating_map[val_str] = 10
                        elif val_str in ["好", "不错", "良好"]:
                            text_rating_map[val_str] = 8
                        elif val_str in ["一般", "普通", "中等"]:
                            text_rating_map[val_str] = 5
                        elif val_str in ["差", "不好", "较差"]:
                            text_rating_map[val_str] = 3
                        elif val_str in ["很差", "非常差", "极差"]:
                            text_rating_map[val_str] = 1

                    if text_rating_map:
                        col_anomalies["suggested_mapping"] = text_rating_map

            # Check for mixed types in general
            else:
                # Check if column has mixed numeric and text values
                try:
                    numeric_count = int(pd.to_numeric(col_data, errors='coerce').notna().sum())
                    text_count = int(len(col_data) - numeric_count)

                    # If significant mix of both
                    if numeric_count > 0 and text_count > 0:
                        numeric_ratio = numeric_count / len(col_data)
                        if 0.1 < numeric_ratio < 0.9:  # Mixed column
                            col_anomalies["issues"].append({
                                "type": "mixed_types",
                                "description": f"Column has {numeric_count} numeric and {text_count} text values",
                                "numeric_count": int(numeric_count),
                                "text_count": int(text_count),
                                "numeric_ratio": float(numeric_ratio),
                            })
                except Exception:
                    pass

            # Check for URLs in text columns
            if detect_urls and pd.api.types.is_string_dtype(df[col]):
                # URL pattern for matching: non-capturing group to avoid pandas warning
                url_pattern_match = r'(?:https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+)'
                # URL pattern for extraction: capturing group for re.findall
                url_pattern_extract = r'(https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+)'
                url_matches = col_data.astype(str).str.contains(url_pattern_match, regex=True, na=False)
                url_count = int(url_matches.sum())

                if url_count > 0:
                    # Get sample URLs
                    url_values = col_data[url_matches]
                    sample_urls = []
                    for val in url_values.head(5):
                        urls = re.findall(url_pattern_extract, str(val))
                        sample_urls.extend(urls[:2])

                    # Calculate ratio of cells containing URLs
                    url_ratio = url_count / len(col_data)

                    # Check if URLs are the main content or mixed in
                    avg_length_with_urls = col_data[url_matches].astype(str).str.len().mean()
                    avg_url_length = sum(len(u) for u in sample_urls[:5]) / max(len(sample_urls[:5]), 1)

                    # If URL takes up most of the cell content, it's likely a URL column
                    is_url_column = avg_url_length > 0 and (avg_url_length / max(avg_length_with_urls, 1)) > 0.7

                    col_anomalies["issues"].append({
                        "type": "urls_in_column",
                        "description": f"Found {url_count} cells containing URLs ({url_ratio*100:.1f}%)",
                        "url_count": int(url_count),
                        "url_ratio": float(url_ratio),
                        "is_url_column": is_url_column,
                        "sample_urls": sample_urls[:5],
                        "suggestion": "remove_column" if is_url_column else "remove_urls_from_values",
                    })

            # Check for outliers in numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                if len(outliers) > 0:
                    col_anomalies["issues"].append({
                        "type": "numeric_outliers",
                        "description": f"Found {len(outliers)} numeric outliers",
                        "outlier_count": int(len(outliers)),
                        "outlier_ratio": float(len(outliers) / len(col_data)),
                        "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},
                        "sample_outliers": [float(v) for v in outliers.head(5).tolist()],
                    })

            if col_anomalies["issues"]:
                anomalies[col] = col_anomalies

        result = {
            "file_path": file_path,
            "anomalies": anomalies,
            "summary": {
                "total_columns_checked": len(columns_to_check),
                "columns_with_anomalies": len(anomalies),
                "anomaly_types": list(set(
                    issue["type"]
                    for col_info in anomalies.values()
                    for issue in col_info["issues"]
                )),
            }
        }

        return tool_success("anomaly_detection", result)

    except Exception as e:
        logger.error(f"Error detecting anomalies in {file_path}: {e}")
        return tool_error(f"Error detecting anomalies in {file_path}: {e}")


def convert_column_values(
    file_path: str,
    column: str,
    value_mapping: Dict[str, Any],
    tool_context: ToolContext,
    convert_non_mapped_to_null: bool = False,
) -> Dict[str, Any]:
    """
    Convert values in a specific column using a mapping.

    Useful for converting text values to numeric ratings or standardizing values.

    Args:
        file_path: File to modify, relative to the import directory
        column: Column name to apply conversions to
        value_mapping: Dict mapping old values to new values
                      e.g., {"非常好": 10, "好": 8, "一般": 5}
        tool_context: ADK ToolContext for state management
        convert_non_mapped_to_null: If True, values not in mapping become null

    Returns:
        Dictionary with conversion result:
        - conversions_made: Number of values converted
        - output_path: Path to modified file
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
        df = _read_file(full_path)

        if column not in df.columns:
            return tool_error(f"Column '{column}' not found in file")

        original_values = df[column].copy()
        conversions_made = 0
        conversion_details = {}

        # Apply mappings
        for old_val, new_val in value_mapping.items():
            mask = df[column] == old_val
            count = int(mask.sum())  # Convert numpy.int64 to int
            if count > 0:
                df.loc[mask, column] = new_val
                conversions_made += count
                conversion_details[str(old_val)] = {
                    "new_value": new_val if not hasattr(new_val, 'item') else new_val.item(),
                    "count": count
                }

        # Handle non-mapped values if requested
        non_mapped_count = 0
        if convert_non_mapped_to_null:
            # Find values that weren't in the mapping and aren't already null
            mapped_values = set(value_mapping.keys())
            for idx in df.index:
                val = original_values[idx]
                if pd.notna(val) and val not in mapped_values:
                    # Check if it's not already converted (could be numeric)
                    try:
                        float(val)  # If it's already numeric, keep it
                    except (ValueError, TypeError):
                        df.loc[idx, column] = pd.NA
                        non_mapped_count += 1

        # Generate output filename
        stem = full_path.stem
        suffix = full_path.suffix

        # Check if already a cleaned file
        if "_cleaned" in stem:
            output_filename = f"{stem}{suffix}"
        else:
            output_filename = f"{stem}_cleaned{suffix}"

        output_path = import_dir / output_filename

        # Save modified file
        if suffix == '.csv':
            df.to_csv(output_path, index=False, encoding='utf-8')
        else:
            df.to_excel(output_path, index=False)

        result = {
            "file_path": file_path,
            "output_path": output_filename,
            "column": column,
            "conversions_made": conversions_made,
            "non_mapped_nullified": non_mapped_count,
            "conversion_details": conversion_details,
        }

        return tool_success("column_values_converted", result)

    except Exception as e:
        logger.error(f"Error converting column values in {file_path}: {e}")
        return tool_error(f"Error converting column values in {file_path}: {e}")


def clean_urls(
    file_path: str,
    tool_context: ToolContext,
    columns: Optional[List[str]] = None,
    remove_url_columns: Optional[List[str]] = None,
    strip_urls_from_columns: Optional[List[str]] = None,
    auto_clean: bool = False,
) -> Dict[str, Any]:
    """
    Clean URLs from data file - either remove entire URL columns or strip URLs from values.

    Args:
        file_path: File to clean, relative to the import directory
        tool_context: ADK ToolContext for state management
        columns: Optional list of columns to check for URLs (if None, checks all)
        remove_url_columns: List of column names to completely remove (URL-only columns)
        strip_urls_from_columns: List of column names where URLs should be stripped from values
        auto_clean: If True, automatically detect and clean URLs based on anomaly analysis

    Returns:
        Dictionary with cleaning result:
        - columns_removed: List of removed URL columns
        - urls_stripped: Dict of columns with count of URLs stripped
        - output_path: Path to cleaned file
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
        df = _read_file(full_path)
        original_shape = df.shape

        columns_removed = []
        urls_stripped = {}

        # URL pattern for detection and stripping (non-capturing group to avoid pandas warning)
        url_pattern = r'(?:https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+)'

        if auto_clean:
            # Run anomaly detection to find URL columns
            from google.adk.tools import ToolContext as TC
            anomaly_result = detect_anomalies(file_path, tool_context, columns, detect_urls=True)

            if anomaly_result.get("status") == "success":
                anomalies = anomaly_result.get("result", {}).get("anomalies", {})

                for col, col_info in anomalies.items():
                    for issue in col_info.get("issues", []):
                        if issue.get("type") == "urls_in_column":
                            if issue.get("is_url_column"):
                                # This is a URL-only column, remove it
                                if remove_url_columns is None:
                                    remove_url_columns = []
                                if col not in remove_url_columns:
                                    remove_url_columns.append(col)
                            else:
                                # URLs mixed with content, strip them
                                if strip_urls_from_columns is None:
                                    strip_urls_from_columns = []
                                if col not in strip_urls_from_columns:
                                    strip_urls_from_columns.append(col)

        # Remove URL-only columns
        if remove_url_columns:
            existing_cols = [c for c in remove_url_columns if c in df.columns]
            if existing_cols:
                df = df.drop(columns=existing_cols)
                columns_removed = existing_cols

        # Strip URLs from values in specified columns
        if strip_urls_from_columns:
            for col in strip_urls_from_columns:
                if col not in df.columns:
                    continue

                # Count URLs before stripping
                original_urls = int(df[col].astype(str).str.count(url_pattern).sum())

                # Strip URLs from values
                df[col] = df[col].astype(str).str.replace(url_pattern, '', regex=True)

                # Clean up extra whitespace
                df[col] = df[col].str.strip()
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)

                # Replace empty strings with NA
                df[col] = df[col].replace('', pd.NA)
                df[col] = df[col].replace('nan', pd.NA)

                urls_stripped[col] = int(original_urls)

        cleaned_shape = df.shape

        # Generate output filename
        stem = full_path.stem
        suffix = full_path.suffix

        if "_cleaned" in stem:
            output_filename = f"{stem}{suffix}"
        else:
            output_filename = f"{stem}_cleaned{suffix}"

        output_path = import_dir / output_filename

        # Save cleaned file
        if suffix == '.csv':
            df.to_csv(output_path, index=False, encoding='utf-8')
        else:
            df.to_excel(output_path, index=False)

        # Update cleaned files state
        cleaned_files = tool_context.state.get(CLEANED_FILES_KEY, {})
        cleaned_files[file_path] = {
            "original_path": file_path,
            "cleaned_path": output_filename,
            "original_shape": original_shape,
            "cleaned_shape": cleaned_shape,
        }
        tool_context.state[CLEANED_FILES_KEY] = cleaned_files

        result = {
            "file_path": file_path,
            "output_path": output_filename,
            "original_shape": {"rows": original_shape[0], "columns": original_shape[1]},
            "cleaned_shape": {"rows": cleaned_shape[0], "columns": cleaned_shape[1]},
            "columns_removed": columns_removed,
            "urls_stripped": urls_stripped,
            "total_urls_removed": sum(urls_stripped.values()),
        }

        return tool_success("urls_cleaned", result)

    except Exception as e:
        logger.error(f"Error cleaning URLs in {file_path}: {e}")
        return tool_error(f"Error cleaning URLs in {file_path}: {e}")


# ============================================================================
# Column Name Normalization Tools
# ============================================================================

# State key for proposed column renames
COLUMN_RENAMES_KEY = "proposed_column_renames"


def _detect_column_name_issues(column_name: str) -> Dict[str, Any]:
    """
    Detect issues with a column name.

    Returns dictionary with:
    - has_issues: Boolean indicating if issues were found
    - issues: List of issue descriptions
    - suggested_name: Suggested normalized name
    """
    issues = []
    suggested_name = column_name

    # Issue 1: Leading/trailing whitespace
    if column_name != column_name.strip():
        issues.append("有前后空格")
        suggested_name = suggested_name.strip()

    # Issue 2: Duplicate number prefix (e.g., "10、10 xxx")
    import re
    dup_prefix_match = re.match(r'^(\d+[、\.])\1+', suggested_name)
    if dup_prefix_match:
        issues.append(f"重复的数字前缀: {dup_prefix_match.group(0)}")
        suggested_name = re.sub(r'^(\d+[、\.])\1+', r'\1', suggested_name)

    # Issue 3: Multiple consecutive spaces
    if '  ' in suggested_name:
        issues.append("包含多个连续空格")
        suggested_name = re.sub(r'\s+', ' ', suggested_name)

    # Issue 4: Special/invisible characters
    # Check for non-printable characters
    cleaned = ''.join(c for c in suggested_name if c.isprintable())
    if cleaned != suggested_name:
        issues.append("包含不可见字符")
        suggested_name = cleaned

    # Issue 5: Very long column names (truncation suggestion)
    if len(suggested_name) > 100:
        issues.append(f"列名过长 ({len(suggested_name)} 字符)")
        # Don't auto-truncate, just flag it

    return {
        "has_issues": len(issues) > 0,
        "issues": issues,
        "suggested_name": suggested_name,
    }


def analyze_column_names(
    file_path: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Analyze column names in a file and detect potential issues.

    Detects:
    - Leading/trailing whitespace
    - Duplicate number prefixes (e.g., "10、10 xxx")
    - Multiple consecutive spaces
    - Non-printable/invisible characters
    - Very long column names

    Args:
        file_path: File to analyze, relative to the import directory
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with:
        - total_columns: Number of columns
        - columns_with_issues: Number of columns with issues
        - issues: List of column issues with suggested fixes
        - message: Summary message
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
        df = _read_file(full_path)
        issues_list = []

        for col in df.columns:
            analysis = _detect_column_name_issues(col)
            if analysis["has_issues"]:
                issues_list.append({
                    "original_name": col,
                    "issues": analysis["issues"],
                    "suggested_name": analysis["suggested_name"],
                })

        result = {
            "file_path": file_path,
            "total_columns": len(df.columns),
            "columns_with_issues": len(issues_list),
            "issues": issues_list,
        }

        if issues_list:
            result["message"] = (
                f"发现 {len(issues_list)} 个列名有问题。"
                "请查看建议的修复并调用 'propose_column_renames' 来确认重命名方案。"
            )
            result["next_action"] = "propose_column_renames"
        else:
            result["message"] = "所有列名格式正常，无需修改。"

        return tool_success("column_name_analysis", result)

    except Exception as e:
        logger.error(f"Error analyzing column names in {file_path}: {e}")
        return tool_error(f"Error analyzing column names in {file_path}: {e}")


def propose_column_renames(
    file_path: str,
    renames: Dict[str, str],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Propose column renames for user confirmation.

    This saves the proposed renames to state for later application.
    User must confirm before the renames are applied.

    Args:
        file_path: File to rename columns in, relative to the import directory
        renames: Dictionary mapping old column names to new column names
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with proposed renames for user confirmation

    Example:
        propose_column_renames(
            "data.csv",
            {
                "10、10 您本次到访的门店是 ": "10、您本次到访的门店是",
                "9、您本次调研的车型及配置是? ": "9、您本次调研的车型及配置是?"
            }
        )
    """
    if not renames:
        return tool_error("No renames provided. Please provide a dictionary of renames.")

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
        df = _read_file(full_path)
        existing_columns = set(df.columns)

        # Validate renames
        valid_renames = {}
        invalid_renames = []

        for old_name, new_name in renames.items():
            if old_name not in existing_columns:
                invalid_renames.append({
                    "old_name": old_name,
                    "reason": "列名不存在于文件中"
                })
            elif new_name in existing_columns and old_name != new_name:
                invalid_renames.append({
                    "old_name": old_name,
                    "new_name": new_name,
                    "reason": "新列名与已存在的列名冲突"
                })
            else:
                valid_renames[old_name] = new_name

        if invalid_renames:
            return tool_error(
                f"部分重命名无效: {invalid_renames}. "
                "请修正后重试。"
            )

        # Store proposed renames in state
        proposed = tool_context.state.get(COLUMN_RENAMES_KEY, {})
        proposed[file_path] = {
            "renames": valid_renames,
            "status": "pending_confirmation"
        }
        tool_context.state[COLUMN_RENAMES_KEY] = proposed

        result = {
            "file_path": file_path,
            "proposed_renames": valid_renames,
            "rename_count": len(valid_renames),
            "status": "待用户确认",
            "message": (
                f"已保存 {len(valid_renames)} 个列名重命名建议。\n"
                "请向用户展示这些修改并询问是否确认。\n"
                "用户确认后，调用 'apply_column_renames' 来执行重命名。"
            ),
            "next_action": "等待用户确认后调用 apply_column_renames"
        }

        return tool_success("proposed_renames", result)

    except Exception as e:
        logger.error(f"Error proposing column renames for {file_path}: {e}")
        return tool_error(f"Error proposing column renames for {file_path}: {e}")


def apply_column_renames(
    file_path: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Apply the proposed column renames to a file.

    This reads the proposed renames from state (set by propose_column_renames)
    and applies them to the file, saving a new cleaned version.

    Args:
        file_path: File to apply renames to, relative to the import directory
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary with:
        - output_path: Path to the file with renamed columns
        - applied_renames: List of applied renames
        - message: Summary message
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

    # Get proposed renames from state
    proposed = tool_context.state.get(COLUMN_RENAMES_KEY, {})
    if file_path not in proposed:
        return tool_error(
            f"No proposed renames found for {file_path}. "
            "Call 'propose_column_renames' first to set up the renames."
        )

    renames = proposed[file_path].get("renames", {})
    if not renames:
        return tool_error("No renames to apply.")

    try:
        df = _read_file(full_path)

        # Apply renames
        df = df.rename(columns=renames)

        # Generate output filename
        stem = full_path.stem
        suffix = full_path.suffix

        # If file already has _cleaned suffix, don't add another
        if stem.endswith("_cleaned"):
            output_filename = f"{stem}{suffix}"
        else:
            output_filename = f"{stem}_cleaned{suffix}"

        output_path = import_dir / output_filename

        # Save file
        if suffix == '.csv':
            df.to_csv(output_path, index=False, encoding='utf-8')
        else:
            df.to_excel(output_path, index=False)

        # Update state
        proposed[file_path]["status"] = "applied"
        tool_context.state[COLUMN_RENAMES_KEY] = proposed

        # Update cleaned files tracking
        cleaned_files = tool_context.state.get(CLEANED_FILES_KEY, {})
        if file_path in cleaned_files:
            # Update existing cleaned file entry
            cleaned_files[file_path]["cleaned_path"] = output_filename
            cleaned_files[file_path]["column_renames_applied"] = True
        else:
            # Create new entry
            cleaned_files[file_path] = {
                "original_path": file_path,
                "cleaned_path": output_filename,
                "original_shape": df.shape,
                "cleaned_shape": df.shape,
                "column_renames_applied": True,
            }
        tool_context.state[CLEANED_FILES_KEY] = cleaned_files

        result = {
            "file_path": file_path,
            "output_path": output_filename,
            "applied_renames": [
                {"old": old, "new": new}
                for old, new in renames.items()
            ],
            "rename_count": len(renames),
            "message": f"成功重命名 {len(renames)} 个列名。文件已保存到 {output_filename}。"
        }

        return tool_success("column_renames_applied", result)

    except Exception as e:
        logger.error(f"Error applying column renames to {file_path}: {e}")
        return tool_error(f"Error applying column renames to {file_path}: {e}")

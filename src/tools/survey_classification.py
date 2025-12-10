"""
Survey column classification tools using LLM.

This module provides tools for classifying survey questionnaire columns
into semantic categories using LLM-based analysis.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from google.adk.tools import ToolContext
from openai import OpenAI

from src.config import get_config
from src.tools.common import tool_success, tool_error, set_state_value, get_state_value

logger = logging.getLogger(__name__)

# State keys
COLUMN_CLASSIFICATION = "column_classification"
CLASSIFICATION_SUMMARY = "classification_summary"

# Survey Processing Model
SURVEY_CLASSIFICATION_MODEL = "qwen3-next-80b-a3b-instruct"


def _get_llm_client() -> OpenAI:
    """Get OpenAI client configured for DashScope."""
    config = get_config()
    return OpenAI(
        api_key=config.llm.api_key,
        base_url=config.llm.api_base
    )


def _classify_column_with_llm(
    column_name: str,
    sample_values: List[Any],
    client: OpenAI
) -> Dict[str, Any]:
    """
    Use LLM to classify a single survey column.

    Args:
        column_name: The header text of the column
        sample_values: Sample values from the column
        client: OpenAI client

    Returns:
        Classification result dict
    """
    # Clean sample values for display
    clean_samples = []
    for v in sample_values[:5]:  # Limit to 5 samples
        if pd.notna(v):
            clean_samples.append(str(v)[:200])  # Truncate long values

    system_prompt = """你是调研问卷数据分析专家。你的任务是分析问卷列的类型。

列类型说明：
1. entity_selection - 受访者选择具体实体的列
   - 例如: "您本次调研的品牌是?", "您本次到访的门店是", "您本次调研的车型及配置是?"
   - 这些列包含品牌名称、门店名称、车型名称等实际选择值

2. rating - 受访者给出数值评分的列
   - 例如: "该门店服务体验方面您会打多少分", "针对该品牌外观设计方面的评分"
   - 包含1-10分的数值评分
   - 通常列名包含"打多少分"、"评分"、"评价"等词

3. open_text - 开放式文本回答的列
   - 例如: "该车型是否有成本控制方面的启发项?", "有什么改进建议?"
   - 包含意见、建议、详细反馈等自由文本

4. demographic - 受访者人口统计信息的列
   - 例如: "您的年龄是?", "您的性别是?", "您的家庭情况是?", "您是否拥有车辆?"
   - 关于受访者个人特征的信息

5. id - 唯一标识符列
   - 例如: "序号", "ID", "响应编号"

对于 entity_selection 类型，还需要识别:
- entity_type: Brand(品牌), Model(车型), Store(门店), Media(媒体), 或 null

对于 rating 和 open_text 类型，还需要识别:
- related_to: 这个评分/意见是关于哪个实体的 (Brand/Model/Store/null)

请严格按照JSON格式输出，不要包含任何其他文字。"""

    user_prompt = f"""分析这个调研问卷列：

列名: {column_name}
样本值: {clean_samples if clean_samples else ['(空)'] }

请返回JSON格式:
{{
    "category": "entity_selection|rating|open_text|demographic|id",
    "entity_type": "Brand|Model|Store|Media|null",
    "related_to": "Brand|Model|Store|null",
    "confidence": 0.0到1.0之间的数值,
    "reasoning": "简短解释分类原因"
}}"""

    try:
        response = client.chat.completions.create(
            model=SURVEY_CLASSIFICATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )

        result_text = response.choices[0].message.content.strip()

        # Try to extract JSON from response
        # Handle case where response might have markdown code blocks
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        result = json.loads(result_text)

        # Normalize null values
        if result.get("entity_type") in ["null", "None", None, ""]:
            result["entity_type"] = None
        if result.get("related_to") in ["null", "None", None, ""]:
            result["related_to"] = None

        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response for column '{column_name}': {e}")
        # Return a default classification
        return {
            "category": "unknown",
            "entity_type": None,
            "related_to": None,
            "confidence": 0.0,
            "reasoning": f"Failed to parse LLM response: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Error classifying column '{column_name}': {e}")
        return {
            "category": "unknown",
            "entity_type": None,
            "related_to": None,
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}"
        }


def classify_survey_column(
    column_name: str,
    sample_values: List[Any],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Uses LLM to classify a survey column based on its name and sample values.

    Args:
        column_name: The header text of the column
        sample_values: Sample values from the column (first 5 non-null values)
        tool_context: ADK ToolContext

    Returns:
        Classification result with category, entity_type, related_to, confidence
    """
    client = _get_llm_client()
    result = _classify_column_with_llm(column_name, sample_values, client)

    # Store individual classification in state
    classifications = get_state_value(tool_context, COLUMN_CLASSIFICATION, {})
    classifications[column_name] = result
    set_state_value(tool_context, COLUMN_CLASSIFICATION, classifications)

    return tool_success("classification", result)


def classify_all_columns(
    file_path: str,
    tool_context: ToolContext,
    batch_size: int = 10
) -> Dict[str, Any]:
    """
    Classifies all columns in a survey file using LLM.

    Args:
        file_path: Path to the survey file (Excel or CSV)
        tool_context: ADK ToolContext
        batch_size: Number of columns to process before logging progress

    Returns:
        Complete classification results and summary
    """
    # Get import directory from config
    config = get_config()
    import_dir = config.neo4j.import_dir

    if import_dir:
        full_path = Path(import_dir) / file_path
    else:
        full_path = Path(file_path)

    if not full_path.exists():
        return tool_error(f"File not found: {file_path}")

    # Load the data
    try:
        if full_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(full_path)
        elif full_path.suffix.lower() == '.csv':
            df = pd.read_csv(full_path)
        else:
            return tool_error(f"Unsupported file format: {full_path.suffix}")
    except Exception as e:
        return tool_error(f"Error loading file: {str(e)}")

    # Initialize LLM client
    client = _get_llm_client()

    # Classify each column
    classifications = {}
    summary = {
        "entity_selection": [],
        "rating": [],
        "open_text": [],
        "demographic": [],
        "id": [],
        "unknown": []
    }

    total_columns = len(df.columns)
    logger.info(f"Classifying {total_columns} columns from {file_path}")

    for i, column in enumerate(df.columns):
        # Get sample values (first 5 non-null)
        sample_values = df[column].dropna().head(5).tolist()

        # Classify
        result = _classify_column_with_llm(column, sample_values, client)
        classifications[column] = result

        # Add to summary
        category = result.get("category", "unknown")
        if category in summary:
            summary[category].append({
                "column": column,
                "entity_type": result.get("entity_type"),
                "related_to": result.get("related_to")
            })
        else:
            summary["unknown"].append({"column": column})

        # Log progress
        if (i + 1) % batch_size == 0:
            logger.info(f"Classified {i + 1}/{total_columns} columns")

    logger.info(f"Classification complete. Summary: "
                f"entity_selection={len(summary['entity_selection'])}, "
                f"rating={len(summary['rating'])}, "
                f"open_text={len(summary['open_text'])}, "
                f"demographic={len(summary['demographic'])}, "
                f"id={len(summary['id'])}")

    # Store in state
    set_state_value(tool_context, COLUMN_CLASSIFICATION, classifications)
    set_state_value(tool_context, CLASSIFICATION_SUMMARY, summary)

    return tool_success("classification_result", {
        "classifications": classifications,
        "summary": summary,
        "total_columns": total_columns
    })


def get_column_classification(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Get the current column classification results.

    Args:
        tool_context: ADK ToolContext

    Returns:
        Current classification results
    """
    classifications = get_state_value(tool_context, COLUMN_CLASSIFICATION, {})
    summary = get_state_value(tool_context, CLASSIFICATION_SUMMARY, {})

    if not classifications:
        return tool_error("No classification results found. Run classify_all_columns first.")

    return tool_success("classification_result", {
        "classifications": classifications,
        "summary": summary
    })


def get_columns_by_category(
    category: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Get all columns that match a specific category.

    Args:
        category: The category to filter by (entity_selection, rating, open_text, demographic, id)
        tool_context: ADK ToolContext

    Returns:
        List of columns matching the category
    """
    summary = get_state_value(tool_context, CLASSIFICATION_SUMMARY, {})

    if not summary:
        return tool_error("No classification results found. Run classify_all_columns first.")

    valid_categories = ["entity_selection", "rating", "open_text", "demographic", "id", "unknown"]
    if category not in valid_categories:
        return tool_error(f"Invalid category. Must be one of: {valid_categories}")

    columns = summary.get(category, [])

    return tool_success("columns", columns)


def get_entity_columns(
    entity_type: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Get columns that contain or relate to a specific entity type.

    Args:
        entity_type: The entity type (Brand, Model, Store, Media)
        tool_context: ADK ToolContext

    Returns:
        Dictionary with 'selection_columns' (where entity is selected)
        and 'related_columns' (ratings/opinions about the entity)
    """
    classifications = get_state_value(tool_context, COLUMN_CLASSIFICATION, {})

    if not classifications:
        return tool_error("No classification results found. Run classify_all_columns first.")

    selection_columns = []
    related_columns = []

    for column, info in classifications.items():
        # Entity selection columns
        if info.get("category") == "entity_selection" and info.get("entity_type") == entity_type:
            selection_columns.append(column)
        # Related columns (ratings/opinions about this entity)
        elif info.get("related_to") == entity_type:
            related_columns.append({
                "column": column,
                "category": info.get("category")
            })

    return tool_success("entity_columns", {
        "entity_type": entity_type,
        "selection_columns": selection_columns,
        "related_columns": related_columns
    })


def save_classification_results(
    output_path: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Save classification results to a JSON file.

    Args:
        output_path: Path to save the results (relative to import dir)
        tool_context: ADK ToolContext

    Returns:
        Success or error message
    """
    classifications = get_state_value(tool_context, COLUMN_CLASSIFICATION, {})
    summary = get_state_value(tool_context, CLASSIFICATION_SUMMARY, {})

    if not classifications:
        return tool_error("No classification results to save.")

    config = get_config()
    import_dir = config.neo4j.import_dir

    if import_dir:
        full_path = Path(import_dir) / output_path
    else:
        full_path = Path(output_path)

    try:
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump({
                "classifications": classifications,
                "summary": summary
            }, f, ensure_ascii=False, indent=2)

        return tool_success("saved", str(full_path))

    except Exception as e:
        return tool_error(f"Error saving classification: {str(e)}")


# Export tools list for agent registration
SURVEY_CLASSIFICATION_TOOLS = [
    classify_survey_column,
    classify_all_columns,
    get_column_classification,
    get_columns_by_category,
    get_entity_columns,
    save_classification_results
]

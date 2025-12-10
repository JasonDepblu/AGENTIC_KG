"""
Survey text analysis and sentiment tools.

This module provides tools for analyzing open-text responses and extracting
sentiment from survey data using LLM.
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
from src.tools.survey_classification import COLUMN_CLASSIFICATION
from src.tools.survey_rating_extraction import RATING_DATA, EXTRACTED_ASPECTS

logger = logging.getLogger(__name__)

# State keys
TEXT_INSIGHTS = "text_insights"
EXTRACTED_FEATURES = "extracted_features"
EXTRACTED_ISSUES = "extracted_issues"
ENTITY_SENTIMENTS = "entity_sentiments"

# Survey Text Analysis Model
SURVEY_TEXT_ANALYSIS_MODEL = "qwen3-next-80b-a3b-instruct"


def _get_llm_client() -> OpenAI:
    """Get OpenAI client configured for DashScope."""
    config = get_config()
    return OpenAI(
        api_key=config.llm.api_key,
        base_url=config.llm.api_base
    )


def analyze_open_text(
    text: str,
    context: Optional[Dict[str, str]],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Uses LLM to analyze an open-text response.

    Args:
        text: The open-text response to analyze
        context: Optional context (brand, model, store names)
        tool_context: ADK ToolContext

    Returns:
        Analysis result with features, issues, insights, and sentiment
    """
    if not text or not str(text).strip() or str(text).lower() in ['nan', 'none', '无', '-', '暂无', '未知', '不清楚']:
        return tool_success("analysis", {
            "features": [],
            "issues": [],
            "insights": [],
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "keywords": []
        })

    client = _get_llm_client()

    context_str = ""
    if context:
        context_parts = []
        if context.get("brand"):
            context_parts.append(f"品牌: {context['brand']}")
        if context.get("model"):
            context_parts.append(f"车型: {context['model']}")
        if context.get("store"):
            context_parts.append(f"门店: {context['store']}")
        if context_parts:
            context_str = f"\n背景信息: {', '.join(context_parts)}"

    system_prompt = """你是汽车行业调研分析专家。分析调研问卷的开放式文本回答。

请从文本中提取：
1. features - 提到的具体产品特性/功能（如"后排娱乐屏"、"沉坑设计"）
2. issues - 提到的问题或不足（如"塑料感强"、"续航不足"）
3. insights - 有价值的见解或建议
4. sentiment - 整体情感倾向 (positive/neutral/negative)
5. sentiment_score - 情感分数 (-1.0 到 1.0)
6. keywords - 关键词列表

请严格按照JSON格式输出，不要包含任何其他文字。"""

    user_prompt = f"""分析以下调研回答：{context_str}

回答内容: {text[:1000]}

请返回JSON格式:
{{
    "features": ["特性1", "特性2"],
    "issues": ["问题1"],
    "insights": ["见解1"],
    "sentiment": "positive|neutral|negative",
    "sentiment_score": 0.0到1.0之间的数值（正面为正，负面为负）,
    "keywords": ["关键词1", "关键词2"]
}}"""

    try:
        response = client.chat.completions.create(
            model=SURVEY_TEXT_ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )

        result_text = response.choices[0].message.content.strip()

        # Try to extract JSON from response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        result = json.loads(result_text)

        # Ensure all expected fields exist
        result.setdefault("features", [])
        result.setdefault("issues", [])
        result.setdefault("insights", [])
        result.setdefault("sentiment", "neutral")
        result.setdefault("sentiment_score", 0.0)
        result.setdefault("keywords", [])

        return tool_success("analysis", result)

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response for text analysis: {e}")
        return tool_success("analysis", {
            "features": [],
            "issues": [],
            "insights": [],
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "keywords": [],
            "error": f"Parse error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        return tool_error(f"Error analyzing text: {str(e)}")


def convert_rating_to_sentiment(
    score: float,
    scale_max: int = 10
) -> Dict[str, Any]:
    """
    Converts numeric rating to sentiment label.

    Args:
        score: The numeric rating score
        scale_max: Maximum value of the rating scale (default 10)

    Returns:
        Dictionary with sentiment_label and sentiment_score
    """
    # Normalize to 0-1 scale
    normalized = score / scale_max

    # Determine sentiment label
    if normalized >= 0.7:
        sentiment_label = "positive"
    elif normalized >= 0.5:
        sentiment_label = "neutral"
    else:
        sentiment_label = "negative"

    # Convert to -1 to 1 scale
    sentiment_score = (normalized * 2) - 1

    return {
        "sentiment_label": sentiment_label,
        "sentiment_score": round(sentiment_score, 2)
    }


def batch_analyze_open_text(
    file_path: str,
    id_column: str,
    tool_context: ToolContext,
    max_rows: Optional[int] = None
) -> Dict[str, Any]:
    """
    Batch processes all open-text responses in a survey file.

    Args:
        file_path: Path to the survey file
        id_column: Column containing respondent IDs
        tool_context: ADK ToolContext
        max_rows: Optional limit on number of rows to process

    Returns:
        Batch analysis results with features, issues, and insights tables
    """
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

    if id_column not in df.columns:
        return tool_error(f"ID column '{id_column}' not found in DataFrame")

    # Find open_text columns
    open_text_columns = [col for col, info in classifications.items()
                        if info.get("category") == "open_text"]

    if not open_text_columns:
        return tool_error("No open_text columns found in classification results.")

    # Find entity columns for context
    entity_columns = {}
    for col, info in classifications.items():
        if info.get("category") == "entity_selection":
            entity_type = info.get("entity_type")
            if entity_type:
                entity_columns[entity_type.lower()] = col

    # Process rows
    if max_rows:
        df = df.head(max_rows)

    all_features = {}  # feature_name -> {id, mentions, sentiment}
    all_issues = {}    # issue_description -> {id, mentions, severity}
    insights_table = []

    feature_counter = 0
    issue_counter = 0

    total_rows = len(df)
    logger.info(f"Processing {total_rows} rows with {len(open_text_columns)} open_text columns")

    for row_idx, (_, row) in enumerate(df.iterrows()):
        respondent_id = row[id_column]

        # Get context
        context = {}
        for entity_type, col in entity_columns.items():
            if col in df.columns:
                value = row[col]
                if pd.notna(value):
                    context[entity_type] = str(value)

        for col in open_text_columns:
            if col not in df.columns:
                continue

            text = row[col]
            if pd.isna(text):
                continue

            text_str = str(text).strip()
            if not text_str or text_str.lower() in ['nan', 'none', '无', '-', '暂无']:
                continue

            # Analyze text
            result = analyze_open_text(text_str, context, tool_context)

            if result.get("status") == "success":
                analysis = result.get("analysis", {})

                # Extract features
                feature_ids = []
                for feature in analysis.get("features", []):
                    if feature not in all_features:
                        all_features[feature] = {
                            "id": f"F_{feature_counter}",
                            "mentions": 0,
                            "positive_count": 0,
                            "negative_count": 0
                        }
                        feature_counter += 1

                    all_features[feature]["mentions"] += 1
                    sentiment = analysis.get("sentiment", "neutral")
                    if sentiment == "positive":
                        all_features[feature]["positive_count"] += 1
                    elif sentiment == "negative":
                        all_features[feature]["negative_count"] += 1

                    feature_ids.append(all_features[feature]["id"])

                # Extract issues
                issue_ids = []
                for issue in analysis.get("issues", []):
                    if issue not in all_issues:
                        all_issues[issue] = {
                            "id": f"I_{issue_counter}",
                            "mentions": 0,
                            "severity": "medium"  # Default
                        }
                        issue_counter += 1

                    all_issues[issue]["mentions"] += 1
                    issue_ids.append(all_issues[issue]["id"])

                # Add to insights table
                insights_table.append({
                    "respondent_id": respondent_id,
                    "source_column": col,
                    "feature_ids": ",".join(feature_ids) if feature_ids else None,
                    "issue_ids": ",".join(issue_ids) if issue_ids else None,
                    "sentiment": analysis.get("sentiment"),
                    "sentiment_score": analysis.get("sentiment_score"),
                    "keywords": ",".join(analysis.get("keywords", [])),
                    "raw_text": text_str[:500]  # Truncate for storage
                })

        # Log progress
        if (row_idx + 1) % 10 == 0:
            logger.info(f"Processed {row_idx + 1}/{total_rows} rows")

    # Build output structures
    features_list = [
        {
            "feature_id": info["id"],
            "feature_name": name,
            "mentions": info["mentions"],
            "sentiment_association": "positive" if info["positive_count"] > info["negative_count"]
                                    else "negative" if info["negative_count"] > info["positive_count"]
                                    else "neutral"
        }
        for name, info in all_features.items()
    ]

    issues_list = [
        {
            "issue_id": info["id"],
            "issue_description": desc,
            "mentions": info["mentions"],
            "severity": info["severity"]
        }
        for desc, info in all_issues.items()
    ]

    # Store in state
    set_state_value(tool_context, EXTRACTED_FEATURES, features_list)
    set_state_value(tool_context, EXTRACTED_ISSUES, issues_list)
    set_state_value(tool_context, TEXT_INSIGHTS, insights_table)

    logger.info(f"Text analysis complete. Features: {len(features_list)}, Issues: {len(issues_list)}, Insights: {len(insights_table)}")

    return tool_success("batch_result", {
        "features_count": len(features_list),
        "issues_count": len(issues_list),
        "insights_count": len(insights_table),
        "sample_features": features_list[:5],
        "sample_issues": issues_list[:5]
    })


def aggregate_entity_sentiment(
    entity_id: str,
    entity_type: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Aggregates sentiment from all sources for a single entity.

    Args:
        entity_id: The entity ID
        entity_type: The entity type (Brand, Model, Store)
        tool_context: ADK ToolContext

    Returns:
        Aggregated sentiment data for the entity
    """
    rating_data = get_state_value(tool_context, RATING_DATA, [])
    text_insights = get_state_value(tool_context, TEXT_INSIGHTS, [])
    aspects = get_state_value(tool_context, EXTRACTED_ASPECTS, [])

    # Initialize counters
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    total_rating = 0.0
    rating_count = 0

    # Process ratings
    # First, find aspects related to this entity type
    related_aspect_ids = [
        asp["aspect_id"] for asp in aspects
        if asp.get("related_entity") == entity_type
    ]

    for rating in rating_data:
        if rating.get("aspect_id") in related_aspect_ids:
            score = rating.get("score", 0)
            total_rating += score
            rating_count += 1

            sentiment = convert_rating_to_sentiment(score)
            label = sentiment["sentiment_label"]
            if label == "positive":
                positive_count += 1
            elif label == "negative":
                negative_count += 1
            else:
                neutral_count += 1

    # Process text insights sentiment
    for insight in text_insights:
        sentiment = insight.get("sentiment")
        if sentiment == "positive":
            positive_count += 1
        elif sentiment == "negative":
            negative_count += 1
        else:
            neutral_count += 1

    # Calculate aggregate sentiment
    total_count = positive_count + negative_count + neutral_count
    if total_count > 0:
        sentiment_score = (positive_count - negative_count) / total_count
    else:
        sentiment_score = 0.0

    avg_rating = total_rating / rating_count if rating_count > 0 else None

    # Determine sentiment label
    if sentiment_score >= 0.3:
        sentiment_label = "positive"
    elif sentiment_score <= -0.3:
        sentiment_label = "negative"
    else:
        sentiment_label = "mixed" if total_count > 0 else "unknown"

    result = {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "avg_rating": round(avg_rating, 2) if avg_rating else None,
        "sentiment_score": round(sentiment_score, 2),
        "sentiment_label": sentiment_label
    }

    return tool_success("entity_sentiment", result)


def save_text_analysis_to_csv(
    output_dir: str,
    tool_context: ToolContext,
    prefix: str = "survey_parsed"
) -> Dict[str, Any]:
    """
    Saves text analysis results to CSV files.

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

    # Save features file
    features = get_state_value(tool_context, EXTRACTED_FEATURES, [])
    if features:
        df = pd.DataFrame(features)
        file_name = f"{prefix}_features.csv"
        file_path = full_output_dir / file_name
        df.to_csv(file_path, index=False, encoding='utf-8')
        saved_files.append(str(file_path))
        logger.info(f"Saved {len(features)} features to {file_path}")

    # Save issues file
    issues = get_state_value(tool_context, EXTRACTED_ISSUES, [])
    if issues:
        df = pd.DataFrame(issues)
        file_name = f"{prefix}_issues.csv"
        file_path = full_output_dir / file_name
        df.to_csv(file_path, index=False, encoding='utf-8')
        saved_files.append(str(file_path))
        logger.info(f"Saved {len(issues)} issues to {file_path}")

    # Save insights file
    insights = get_state_value(tool_context, TEXT_INSIGHTS, [])
    if insights:
        df = pd.DataFrame(insights)
        file_name = f"{prefix}_insights.csv"
        file_path = full_output_dir / file_name
        df.to_csv(file_path, index=False, encoding='utf-8')
        saved_files.append(str(file_path))
        logger.info(f"Saved {len(insights)} insights to {file_path}")

    return tool_success("saved_files", {
        "files": saved_files,
        "output_dir": str(full_output_dir)
    })


def get_extracted_features(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Get extracted features from state.

    Args:
        tool_context: ADK ToolContext

    Returns:
        Extracted features
    """
    features = get_state_value(tool_context, EXTRACTED_FEATURES, [])

    if not features:
        return tool_error("No extracted features found. Run batch_analyze_open_text first.")

    return tool_success("features", features)


def get_extracted_issues(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Get extracted issues from state.

    Args:
        tool_context: ADK ToolContext

    Returns:
        Extracted issues
    """
    issues = get_state_value(tool_context, EXTRACTED_ISSUES, [])

    if not issues:
        return tool_error("No extracted issues found. Run batch_analyze_open_text first.")

    return tool_success("issues", issues)


def get_text_insights(
    respondent_id: Optional[Any],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Get text insights, optionally filtered by respondent.

    Args:
        respondent_id: Optional respondent ID to filter by
        tool_context: ADK ToolContext

    Returns:
        Text insights
    """
    insights = get_state_value(tool_context, TEXT_INSIGHTS, [])

    if not insights:
        return tool_error("No text insights found. Run batch_analyze_open_text first.")

    if respondent_id is not None:
        insights = [i for i in insights if str(i.get("respondent_id")) == str(respondent_id)]

    return tool_success("insights", insights)


# Export tools list for agent registration
SURVEY_TEXT_ANALYSIS_TOOLS = [
    analyze_open_text,
    convert_rating_to_sentiment,
    batch_analyze_open_text,
    aggregate_entity_sentiment,
    save_text_analysis_to_csv,
    get_extracted_features,
    get_extracted_issues,
    get_text_insights
]

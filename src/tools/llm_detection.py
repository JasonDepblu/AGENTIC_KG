"""
LLM-based detection tools for domain-agnostic entity and column type detection.

These tools use LLM to analyze column names and data samples to automatically
detect entity types, text feedback columns, and convert text ratings - replacing
hardcoded domain-specific patterns.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from google.adk.tools import ToolContext

from ..llm import get_litellm_client
from ..config import get_neo4j_import_dir

logger = logging.getLogger(__name__)


def _call_llm_for_detection(prompt: str, max_tokens: int = 2000) -> Dict[str, Any]:
    """
    Call LLM with a detection prompt and parse JSON response.

    Args:
        prompt: The detection prompt
        max_tokens: Maximum tokens in response

    Returns:
        Parsed JSON response or error dict
    """
    try:
        client = get_litellm_client()
        response = client.chat.completions.create(
            model="qwen-plus-latest",  # Use fast model for detection
            messages=[
                {"role": "system", "content": "你是一个数据分析专家。请分析数据并返回JSON格式的结果。只返回JSON，不要有其他文字。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.1,  # Low temperature for consistent results
        )

        result_text = response.choices[0].message.content.strip()

        # Try to extract JSON from response
        # Handle markdown code blocks
        if "```json" in result_text:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', result_text)
            if match:
                result_text = match.group(1)
        elif "```" in result_text:
            match = re.search(r'```\s*([\s\S]*?)\s*```', result_text)
            if match:
                result_text = match.group(1)

        return json.loads(result_text)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Raw response: {result_text[:500]}")
        return {"error": f"JSON parse error: {e}", "raw_response": result_text[:500]}
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {"error": str(e)}


def detect_entities_from_columns(
    file_path: str,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Use LLM to analyze column names and detect entity types.

    This tool reads the column names and sample data from a CSV file,
    then uses LLM to infer what type of entity each column represents.

    Args:
        file_path: Path to the CSV data file
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary containing:
        - entities: List of detected entity info for each column
        - domain: Inferred data domain (medical/automotive/engineering/other)
        - respondent_column: Detected unique identifier column
    """
    logger.info(f"=== detect_entities_from_columns CALLED ===")
    logger.info(f"  file_path: {file_path}")

    # Resolve file path
    import_dir = get_neo4j_import_dir()
    if not file_path.startswith("/"):
        full_path = f"{import_dir}/{file_path}"
    else:
        full_path = file_path

    try:
        # Read column names and sample data
        df = pd.read_csv(full_path, nrows=5, encoding='utf-8')
        columns = df.columns.tolist()

        # Get sample values for each column (first 3 non-null values)
        sample_data = {}
        for col in columns:
            non_null = df[col].dropna().head(3).tolist()
            sample_data[col] = [str(v)[:50] for v in non_null]  # Truncate long values

    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return {"status": "error", "error": f"Failed to read file: {e}"}

    # Build prompt for entity detection
    prompt = f"""
分析以下CSV文件的列名和样本数据，识别每列可能代表的实体或数据类型。

列名列表:
{json.dumps(columns, ensure_ascii=False, indent=2)}

各列样本数据:
{json.dumps(sample_data, ensure_ascii=False, indent=2)}

请对每列进行分类，识别以下类型:

1. **唯一标识符列** (IDENTIFIER): 用于唯一标识每行记录的列，如ID、序号、编号等
2. **分类实体列** (CATEGORICAL_ENTITY): 表示可枚举的实体类型，如品牌、医院、科室、设备型号等
3. **数值评分列** (NUMERIC_RATING): 包含评分、分数、测量值等数值型数据
4. **正面反馈文本列** (TEXT_POSITIVE): 包含正面评价、优点描述等文本
5. **负面反馈文本列** (TEXT_NEGATIVE): 包含负面评价、缺点描述等文本
6. **混合反馈文本列** (TEXT_MIXED): 包含正负面混合的反馈文本
7. **多选/标签列** (MULTI_SELECT): 包含多个选项或标签的列
8. **元数据列** (METADATA): 时间戳、来源、备注等辅助信息
9. **忽略列** (IGNORE): 无分析价值的列

返回JSON格式:
{{
    "domain": "推断的数据领域 (医疗/汽车/工程/调研/其他)",
    "domain_description": "对数据领域的简要描述",
    "respondent_column": "唯一标识符列的列名",
    "entities": [
        {{
            "column": "列名",
            "column_type": "类型代码 (IDENTIFIER/CATEGORICAL_ENTITY/NUMERIC_RATING/TEXT_POSITIVE/TEXT_NEGATIVE/TEXT_MIXED/MULTI_SELECT/METADATA/IGNORE)",
            "suggested_entity_name": "建议的实体名称 (英文，如Patient、Brand、Aspect等)",
            "suggested_property_name": "建议的属性名称 (英文，如patient_id、brand_name等)",
            "reason": "判断原因",
            "is_key_entity": true/false
        }}
    ],
    "suggested_node_types": [
        {{
            "label": "建议的节点标签 (如Respondent、Brand、Aspect等)",
            "source_columns": ["对应的列名列表"],
            "unique_property": "唯一属性名",
            "description": "节点类型描述"
        }}
    ],
    "suggested_relationships": [
        {{
            "type": "关系类型 (如RATES、EVALUATED、MENTIONED等)",
            "from_node": "起点节点类型",
            "to_node": "终点节点类型",
            "source_columns": ["相关列名"],
            "description": "关系描述"
        }}
    ]
}}
"""

    result = _call_llm_for_detection(prompt)

    if "error" in result:
        return {"status": "error", "error": result.get("error")}

    # Store detection result in session state
    tool_context.state["detected_entities"] = result

    logger.info(f"  Detected domain: {result.get('domain')}")
    logger.info(f"  Detected {len(result.get('entities', []))} column types")
    logger.info(f"  Suggested {len(result.get('suggested_node_types', []))} node types")

    return {
        "status": "success",
        "domain": result.get("domain"),
        "domain_description": result.get("domain_description"),
        "respondent_column": result.get("respondent_column"),
        "entity_count": len(result.get("entities", [])),
        "suggested_node_types": result.get("suggested_node_types", []),
        "suggested_relationships": result.get("suggested_relationships", []),
        "details": result.get("entities", []),
    }


def detect_text_feedback_columns(
    file_path: str,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Use LLM to identify which columns contain text feedback for NER extraction.

    This specifically identifies columns that contain open-ended text responses
    that should be processed with NER to extract entities like features, issues, etc.

    Args:
        file_path: Path to the CSV data file
        tool_context: ADK ToolContext for state management

    Returns:
        Dictionary containing list of text feedback columns with their types
    """
    logger.info(f"=== detect_text_feedback_columns CALLED ===")
    logger.info(f"  file_path: {file_path}")

    # Resolve file path
    import_dir = get_neo4j_import_dir()
    if not file_path.startswith("/"):
        full_path = f"{import_dir}/{file_path}"
    else:
        full_path = file_path

    try:
        df = pd.read_csv(full_path, nrows=10, encoding='utf-8')
        columns = df.columns.tolist()

        # Get longer sample values for text analysis
        sample_data = {}
        for col in columns:
            non_null = df[col].dropna().head(5).tolist()
            sample_data[col] = [str(v)[:200] for v in non_null]

    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return {"status": "error", "error": f"Failed to read file: {e}"}

    prompt = f"""
分析以下CSV文件，识别哪些列包含用户的开放性文本反馈，这些列需要进行实体抽取(NER)处理。

列名列表:
{json.dumps(columns, ensure_ascii=False, indent=2)}

各列样本数据:
{json.dumps(sample_data, ensure_ascii=False, indent=2)}

文本反馈列的特征:
- 包含用户的主观评价、意见或体验描述
- 非结构化的自由文本（不是选项、评分等）
- 可能包含对产品特性、服务体验、问题等的描述
- 文本长度通常较长（超过几个字符）

请识别所有适合进行NER抽取的文本列，并判断其情感倾向。

返回JSON格式:
{{
    "text_feedback_columns": [
        {{
            "column": "列名",
            "feedback_type": "positive/negative/neutral/mixed",
            "content_category": "内容分类 (如: 产品体验、服务反馈、问题描述、建议等)",
            "suggested_entities": ["可能抽取的实体类型，如Feature, Issue, Component等"],
            "reason": "判断原因",
            "priority": 1-5 (1最高优先级)
        }}
    ],
    "non_text_columns": ["不包含开放文本的列名列表"],
    "total_text_columns": 文本列数量
}}
"""

    result = _call_llm_for_detection(prompt)

    if "error" in result:
        return {"status": "error", "error": result.get("error")}

    # Store in session state
    tool_context.state["text_feedback_columns"] = result.get("text_feedback_columns", [])

    text_cols = result.get("text_feedback_columns", [])
    logger.info(f"  Detected {len(text_cols)} text feedback columns")

    return {
        "status": "success",
        "text_column_count": len(text_cols),
        "text_columns": text_cols,
        "non_text_columns": result.get("non_text_columns", []),
    }


def convert_text_to_rating(
    text_values: List[str],
    tool_context: ToolContext,
    rating_scale: str = "0-10",
) -> Dict[str, Any]:
    """
    Use LLM to convert text rating values to numeric scores.

    This replaces hardcoded TEXT_TO_RATING_MAP with LLM-based conversion
    that can understand any language and rating expression.

    Args:
        text_values: List of unique text values to convert
        tool_context: ADK ToolContext
        rating_scale: The target rating scale (default "0-10")

    Returns:
        Dictionary mapping text values to numeric ratings
    """
    logger.info(f"=== convert_text_to_rating CALLED ===")
    logger.info(f"  text_values count: {len(text_values)}")
    logger.info(f"  rating_scale: {rating_scale}")

    if not text_values:
        return {"status": "success", "mappings": {}, "unmapped": []}

    # Deduplicate and clean
    unique_values = list(set([str(v).strip() for v in text_values if v and str(v).strip()]))

    prompt = f"""
将以下文本评价值转换为数值评分。

文本值列表:
{json.dumps(unique_values, ensure_ascii=False, indent=2)}

目标评分范围: {rating_scale}

转换规则:
- 非常正面的评价 (如"非常好"、"excellent"、"很满意") -> 高分 (8-10)
- 正面评价 (如"好"、"good"、"满意") -> 中高分 (6-8)
- 中性评价 (如"一般"、"average"、"普通") -> 中分 (4-6)
- 负面评价 (如"差"、"bad"、"不满意") -> 中低分 (2-4)
- 非常负面的评价 (如"非常差"、"terrible"、"很不满意") -> 低分 (0-2)
- 数字字符串直接转换为数值
- 无法识别的值标记为null

返回JSON格式:
{{
    "mappings": {{
        "文本值1": 数值,
        "文本值2": 数值,
        ...
    }},
    "unmapped": ["无法映射的值列表"],
    "mapping_rules": "使用的映射逻辑说明"
}}
"""

    result = _call_llm_for_detection(prompt)

    if "error" in result:
        return {"status": "error", "error": result.get("error")}

    mappings = result.get("mappings", {})
    unmapped = result.get("unmapped", [])

    logger.info(f"  Mapped {len(mappings)} values")
    logger.info(f"  Unmapped {len(unmapped)} values")

    # Store in session state for reuse
    existing_mappings = tool_context.state.get("text_to_rating_mappings", {})
    existing_mappings.update(mappings)
    tool_context.state["text_to_rating_mappings"] = existing_mappings

    return {
        "status": "success",
        "mappings": mappings,
        "unmapped": unmapped,
        "mapping_rules": result.get("mapping_rules", ""),
    }


def get_detected_entities(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get the previously detected entity information from session state.

    Args:
        tool_context: ADK ToolContext

    Returns:
        Previously detected entity information or empty dict
    """
    detected = tool_context.state.get("detected_entities", {})
    text_cols = tool_context.state.get("text_feedback_columns", [])

    return {
        "status": "success",
        "detected_entities": detected,
        "text_feedback_columns": text_cols,
        "has_detection": bool(detected),
    }


def infer_schema_from_detection(
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Generate a schema proposal based on detected entities.

    This combines the entity detection and text feedback detection results
    to propose a complete knowledge graph schema.

    Args:
        tool_context: ADK ToolContext

    Returns:
        Proposed schema with nodes and relationships
    """
    logger.info("=== infer_schema_from_detection CALLED ===")

    detected = tool_context.state.get("detected_entities", {})
    text_cols = tool_context.state.get("text_feedback_columns", [])

    if not detected:
        return {
            "status": "error",
            "error": "No entity detection results found. Run detect_entities_from_columns first."
        }

    # Build schema proposal from detection results
    nodes = detected.get("suggested_node_types", [])
    relationships = detected.get("suggested_relationships", [])

    # Add text extraction nodes if text columns detected
    for text_col in text_cols:
        suggested_entities = text_col.get("suggested_entities", [])
        for entity in suggested_entities:
            # Check if node type already exists
            existing = [n for n in nodes if n.get("label") == entity]
            if not existing:
                nodes.append({
                    "label": entity,
                    "source_columns": [text_col.get("column")],
                    "unique_property": f"{entity.lower()}_id",
                    "description": f"Extracted from text column: {text_col.get('column')}",
                    "extraction_type": "text_extraction",
                })

    return {
        "status": "success",
        "domain": detected.get("domain"),
        "proposed_nodes": nodes,
        "proposed_relationships": relationships,
        "text_extraction_columns": text_cols,
        "respondent_column": detected.get("respondent_column"),
    }

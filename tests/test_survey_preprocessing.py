"""
Test script for the new survey preprocessing pipeline.
Tests the LLM-based column classification and entity extraction.
"""

import asyncio
import pandas as pd
from pathlib import Path

# Mock ToolContext for testing without full ADK setup
class MockToolContext:
    def __init__(self):
        self.state = {}

# Set up the mock before imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.survey_classification import (
    _classify_column_with_llm,
    _get_llm_client,
    SURVEY_CLASSIFICATION_MODEL,
)
from src.tools.survey_entity_extraction import (
    extract_entities_from_column,
)
from src.tools.survey_rating_extraction import (
    _extract_aspect_name,
)


def test_aspect_name_extraction():
    """Test extracting aspect names from column headers."""
    print("\n=== Testing Aspect Name Extraction ===")

    test_cases = [
        ('针对该品牌"外观设计"方面的评分', '外观设计'),
        ('该门店"服务体验"方面您会打多少分呢？', '服务体验'),
        ('针对该车型"空间舒适度"的评分', '空间舒适度'),
        ('该车型"外观设计"方面您会打多少分呢?', '外观设计'),
    ]

    for column_name, expected in test_cases:
        result = _extract_aspect_name(column_name)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{column_name[:30]}...' -> '{result}' (expected: '{expected}')")


def test_column_classification_with_llm():
    """Test LLM-based column classification with a few sample columns."""
    print("\n=== Testing LLM Column Classification ===")

    # Test columns with clear semantic meaning
    test_columns = [
        {
            "name": "8、您本次调研的品牌是?",
            "samples": ["乐道", "小鹏", "传祺", "理想", "蔚来"],
            "expected_category": "entity_selection",
            "expected_entity_type": "Brand"
        },
        {
            "name": "9、您本次调研的车型及配置是?",
            "samples": ["L90_顶配", "P7顶配", "E8 MAX", "X9-增程版"],
            "expected_category": "entity_selection",
            "expected_entity_type": "Model"
        },
        {
            "name": "10、您本次到访的门店是",
            "samples": ["广州市番禺大道店", "广州市番禺区天河城店", "番禺天河城店"],
            "expected_category": "entity_selection",
            "expected_entity_type": "Store"
        },
        {
            "name": '11、该车型"外观设计"方面您会打多少分呢?',
            "samples": ["9", "8", "7", "8", "7"],
            "expected_category": "rating",
            "expected_entity_type": None
        },
        {
            "name": "3、您的年龄是?",
            "samples": ["25-29岁", "40-44岁", "30-34岁", "35-39岁"],
            "expected_category": "demographic",
            "expected_entity_type": None
        },
        {
            "name": "90、该车型/品牌是否有优秀的场景创新、亮点设计等启发项?",
            "samples": [
                "后排娱乐屏的开合方式及角度调节很灵活",
                "换电是核心竞争力",
                "语音交互更接近真人互动"
            ],
            "expected_category": "open_text",
            "expected_entity_type": None
        },
        {
            "name": "序号",
            "samples": ["74", "75", "76", "77", "78"],
            "expected_category": "id",
            "expected_entity_type": None
        }
    ]

    try:
        client = _get_llm_client()
        print(f"  Using model: {SURVEY_CLASSIFICATION_MODEL}")

        for test in test_columns:
            print(f"\n  Testing: {test['name'][:50]}...")
            result = _classify_column_with_llm(test["name"], test["samples"], client)

            category_match = result.get("category") == test["expected_category"]
            entity_match = result.get("entity_type") == test["expected_entity_type"]

            cat_status = "✓" if category_match else "✗"
            ent_status = "✓" if entity_match else "✗"

            print(f"    {cat_status} Category: {result.get('category')} (expected: {test['expected_category']})")
            print(f"    {ent_status} Entity Type: {result.get('entity_type')} (expected: {test['expected_entity_type']})")
            print(f"    Confidence: {result.get('confidence', 'N/A')}")
            print(f"    Reasoning: {result.get('reasoning', 'N/A')[:80]}...")

    except Exception as e:
        print(f"  Error: {e}")
        print("  (This test requires valid DASHSCOPE_API_KEY)")


def test_entity_extraction():
    """Test entity extraction from a sample DataFrame."""
    print("\n=== Testing Entity Extraction ===")

    # Create sample data
    data = {
        "序号": [74, 75, 76, 77, 78],
        "8、您本次调研的品牌是?": ["乐道", "小鹏", "小鹏", "传祺", "理想"],
        "9、您本次调研的车型及配置是?": ["L90_顶配", "P7顶配", "X9-增程版", "E8 MAX", "L9 Pro"],
    }
    df = pd.DataFrame(data)

    # Mock context
    ctx = MockToolContext()

    # Test brand extraction
    result = extract_entities_from_column(df, "8、您本次调研的品牌是?", "Brand", ctx)

    if result.get("status") == "success":
        extraction = result.get("extraction_result", {})
        entities = extraction.get("entities", [])
        print(f"  ✓ Extracted {len(entities)} Brand entities:")
        for e in entities:
            print(f"    - {e['entity_id']}: {e['value']}")
    else:
        print(f"  ✗ Error: {result.get('error_message')}")

    # Test model extraction
    result = extract_entities_from_column(df, "9、您本次调研的车型及配置是?", "Model", ctx)

    if result.get("status") == "success":
        extraction = result.get("extraction_result", {})
        entities = extraction.get("entities", [])
        print(f"\n  ✓ Extracted {len(entities)} Model entities:")
        for e in entities:
            print(f"    - {e['entity_id']}: {e['value']}")
    else:
        print(f"  ✗ Error: {result.get('error_message')}")


def main():
    print("=" * 60)
    print("Survey Preprocessing Pipeline Tests")
    print("=" * 60)

    # Test 1: Aspect name extraction (no LLM needed)
    test_aspect_name_extraction()

    # Test 2: Entity extraction (no LLM needed)
    test_entity_extraction()

    # Test 3: LLM-based classification (requires API key)
    test_column_classification_with_llm()

    print("\n" + "=" * 60)
    print("Tests completed")
    print("=" * 60)


if __name__ == "__main__":
    main()

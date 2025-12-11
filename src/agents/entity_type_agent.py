"""
Entity Type Generator Agent for domain-agnostic schema detection.

This agent uses LLM to analyze data files and automatically generate
appropriate EntityType definitions for any domain, replacing hardcoded
automotive/survey-specific entity types.
"""

from typing import AsyncGenerator, Dict, Any, List

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event, EventActions

from ..llm import get_adk_llm, get_adk_llm_flash
from ..tools.file_suggestion import get_approved_files, sample_file
from ..tools.llm_detection import (
    detect_entities_from_columns,
    detect_text_feedback_columns,
    get_detected_entities,
    infer_schema_from_detection,
)


# ============================================================
# State Keys
# ============================================================
ENTITY_TYPE_DETECTION_COMPLETE = "entity_type_detection_complete"
DETECTED_DOMAIN = "detected_domain"
PROPOSED_ENTITY_TYPES = "proposed_entity_types"


# ============================================================
# Entity Type Generator Agent Instructions
# ============================================================
GENERATOR_INSTRUCTION = """
你是一个数据实体类型设计专家。你的任务是分析数据文件，自动识别并生成合适的实体类型定义。

## 工作流程

### Step 1: 获取数据文件
使用 'get_approved_files' 获取已批准的数据文件列表。

### Step 2: 分析列名和数据
使用 'detect_entities_from_columns' 分析文件的列名和样本数据，识别:
- 数据所属的领域（医疗、汽车、工程等）
- 每列代表的数据类型
- 建议的节点类型和关系类型

### Step 3: 检测文本反馈列
使用 'detect_text_feedback_columns' 识别需要进行实体抽取的文本列。

### Step 4: 生成实体类型提案
使用 'propose_entity_types' 基于检测结果生成实体类型定义。

### Step 5: 报告结果
使用 'get_detected_entities' 获取完整的检测结果，向用户报告:
- 检测到的数据领域
- 建议的节点类型列表
- 建议的关系类型列表
- 需要文本抽取的列

## 设计原则

1. **领域无关**: 不预设任何特定领域的实体类型
2. **语义清晰**: 每个实体类型应有明确的含义
3. **适度抽象**: 不过于泛化也不过于具体
4. **关系完整**: 考虑实体之间可能的关系
"""

GENERATOR_HINTS = """
## 实体类型命名规范

- 使用 PascalCase (如 Patient, BrandModel, TestResult)
- 名称应反映实体的语义含义
- 避免使用过于通用的名称 (如 Entity, Item, Thing)

## 常见实体类型模式

### 调研类数据
- Respondent: 受访者/填写者
- Rating/Aspect: 评分维度
- TextFeedback: 文本反馈

### 医疗类数据
- Patient: 患者
- Department: 科室
- Doctor: 医生
- Treatment: 治疗方案
- Symptom: 症状

### 工程类数据
- Component: 部件
- Parameter: 参数
- Measurement: 测量值
- TestScenario: 测试场景
- Anomaly: 异常

### 通用类型
- Feature: 特性/功能
- Issue: 问题/缺陷
- Quality: 质量属性
"""

GENERATOR_CHAIN_OF_THOUGHT = """
## 执行步骤

1. 首先调用 get_approved_files 获取文件列表
2. 对第一个文件调用 detect_entities_from_columns 进行列分析
3. 调用 detect_text_feedback_columns 检测文本列
4. 调用 infer_schema_from_detection 生成schema提案
5. 向用户展示检测结果，询问是否满意
6. 如果用户满意，调用 complete_entity_detection 完成检测

## 示例输出

"我已分析了数据文件，检测结果如下：

**数据领域**: 医疗调研

**建议的节点类型**:
- Patient (患者): 唯一标识每个患者记录
- Department (科室): 医院科室分类
- Doctor (医生): 接诊医生

**建议的关系类型**:
- VISITED: Patient -> Department
- TREATED_BY: Patient -> Doctor
- RATES: Patient -> Aspect (满意度评分)

**文本抽取列**:
- 服务体验 (正面): 提取 Feature 实体
- 改进建议 (负面): 提取 Issue 实体

是否同意这个实体类型设计？"
"""

GENERATOR_FULL_INSTRUCTION = f"""
{GENERATOR_INSTRUCTION}
{GENERATOR_HINTS}
{GENERATOR_CHAIN_OF_THOUGHT}
"""


# ============================================================
# Critic Agent Instructions
# ============================================================
CRITIC_INSTRUCTION = """
你是实体类型设计审核专家。你的任务是审核生成的实体类型定义是否完整和合理。

## 审核要点

### 1. 完整性检查
- 是否覆盖了所有关键数据列
- 是否有遗漏的重要实体类型
- 关系类型是否完整

### 2. 合理性检查
- 实体类型命名是否清晰
- 是否存在重复或冗余的类型
- 实体类型的粒度是否适当

### 3. 一致性检查
- 命名风格是否统一
- 关系方向是否正确
- 属性定义是否一致

## 响应格式

**如果检测结果合理完整**:
直接返回 'valid' (单词，无其他内容)

**如果需要修改**:
返回 'retry' 并列出需要改进的点:
- 问题1: ...
- 问题2: ...
- 建议: ...

## 重要提示
- 如果实体类型基本合理，即使不完美也返回 'valid'
- 只有在有明显遗漏或错误时才返回 'retry'
- 不要过于苛刻，允许后续在schema设计阶段细化
"""


# ============================================================
# Tool for completing detection
# ============================================================
def propose_entity_types(
    entity_types: List[Dict[str, Any]],
    tool_context,
) -> Dict[str, Any]:
    """
    Propose entity types based on detection results.

    Args:
        entity_types: List of entity type definitions
        tool_context: ADK ToolContext

    Returns:
        Status of the proposal
    """
    tool_context.state[PROPOSED_ENTITY_TYPES] = entity_types
    return {
        "status": "success",
        "message": f"Proposed {len(entity_types)} entity types",
        "entity_types": entity_types,
    }


def complete_entity_detection(tool_context) -> Dict[str, Any]:
    """
    Mark entity type detection as complete.

    Args:
        tool_context: ADK ToolContext

    Returns:
        Completion status
    """
    tool_context.state[ENTITY_TYPE_DETECTION_COMPLETE] = True

    detected = tool_context.state.get("detected_entities", {})
    domain = detected.get("domain", "unknown")
    tool_context.state[DETECTED_DOMAIN] = domain

    return {
        "status": "success",
        "message": "Entity type detection completed",
        "domain": domain,
    }


# ============================================================
# Tools List
# ============================================================
GENERATOR_TOOLS = [
    get_approved_files,
    sample_file,
    detect_entities_from_columns,
    detect_text_feedback_columns,
    get_detected_entities,
    infer_schema_from_detection,
    propose_entity_types,
    complete_entity_detection,
]

CRITIC_TOOLS = [
    get_detected_entities,
    infer_schema_from_detection,
]


# ============================================================
# Stop Checker Agent
# ============================================================
class EntityTypeStopChecker(BaseAgent):
    """
    Agent that checks if entity type detection is complete.
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check if detection is complete."""

        is_complete = ctx.session.state.get(ENTITY_TYPE_DETECTION_COMPLETE, False)
        feedback = ctx.session.state.get("entity_type_feedback", "")
        feedback_str = str(feedback).strip().lower()

        should_stop = is_complete or feedback_str == "valid"

        if should_stop:
            print(f"\n### {self.name}: Entity type detection complete, stopping loop")
        else:
            print(f"\n### {self.name}: Detection not complete, feedback='{feedback_str[:50]}'")

        yield Event(
            author=self.name,
            actions=EventActions(escalate=should_stop)
        )


# ============================================================
# Agent Callbacks
# ============================================================
def log_agent(callback_context: CallbackContext) -> None:
    """Log agent entry for debugging."""
    print(f"\n### Entering Agent: {callback_context.agent_name}")


# ============================================================
# Agent Factory Functions
# ============================================================
def create_entity_type_generator_agent(
    llm=None,
    name: str = "entity_type_generator_v1"
) -> LlmAgent:
    """
    Create an Entity Type Generator Agent.

    This agent analyzes data files and proposes entity types
    appropriate for the detected domain.

    Args:
        llm: Optional LLM instance
        name: Agent name

    Returns:
        LlmAgent: Configured generator agent
    """
    if llm is None:
        llm = get_adk_llm_flash()  # Use fast model

    return LlmAgent(
        name=name,
        model=llm,
        description="Analyzes data and proposes entity types",
        instruction=GENERATOR_FULL_INSTRUCTION,
        tools=GENERATOR_TOOLS,
        before_agent_callback=log_agent,
    )


def create_entity_type_critic_agent(
    llm=None,
    name: str = "entity_type_critic_v1"
) -> LlmAgent:
    """
    Create an Entity Type Critic Agent.

    This agent validates the proposed entity types.

    Args:
        llm: Optional LLM instance
        name: Agent name

    Returns:
        LlmAgent: Configured critic agent
    """
    if llm is None:
        llm = get_adk_llm()  # Use better model for critic

    return LlmAgent(
        name=name,
        model=llm,
        description="Validates proposed entity types",
        instruction=CRITIC_INSTRUCTION,
        tools=CRITIC_TOOLS,
        output_key="entity_type_feedback",
        before_agent_callback=log_agent,
    )


def create_entity_type_detection_loop(
    llm=None,
    max_iterations: int = 2,
    name: str = "entity_type_detection_loop"
) -> LoopAgent:
    """
    Create an Entity Type Detection Loop.

    This loop coordinates the generator and critic agents
    to produce validated entity type definitions.

    Args:
        llm: Optional LLM instance
        max_iterations: Maximum refinement iterations
        name: Agent name

    Returns:
        LoopAgent: Configured detection loop
    """
    generator = create_entity_type_generator_agent(llm)
    critic = create_entity_type_critic_agent(llm)
    stop_checker = EntityTypeStopChecker(name="EntityTypeStopChecker")

    return LoopAgent(
        name=name,
        description="Detects and validates entity types for any domain",
        max_iterations=max_iterations,
        sub_agents=[generator, critic, stop_checker],
        before_agent_callback=log_agent,
    )

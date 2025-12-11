"""
Pipeline service with streaming support.

Wraps the KGPipeline to provide real-time event streaming via WebSocket.
"""

import asyncio
import uuid
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from google.genai import types
from google.adk.agents import LoopAgent, RunConfig

from src.llm import get_adk_llm
from src.agents.base import make_agent_caller
from src.agents.user_intent_agent import create_user_intent_agent
from src.agents.file_suggestion_agent import create_file_suggestion_agent
from src.agents.data_preprocessing_agent import create_data_preprocessing_agent
from src.agents.survey_preprocessing_coordinator import create_survey_preprocessing_coordinator
from src.agents.schema_proposal_agent import create_schema_refinement_loop
from src.agents.kg_builder_agent import build_domain_graph
from src.agents.kg_query_agent import create_kg_query_agent
from src.agents.cypher_loop_agent import create_cypher_generation_loop
from src.agents.schema_design_agent import create_schema_design_loop
from src.agents.targeted_preprocessing_agent import create_targeted_preprocessing_loop
from src.agents.schema_preprocess_coordinator import create_schema_preprocess_coordinator
from src.agents.data_cleaning_agent import create_data_cleaning_loop
from src.agents.structured_data_coordinator import create_structured_data_coordinator
from src.tools.kg_construction import APPROVED_CONSTRUCTION_PLAN
from src.tools.data_cleaning import DATA_CLEANING_COMPLETE_KEY
from src.models.target_schema import APPROVED_TARGET_SCHEMA_KEY
from src.tools.targeted_preprocessing import (
    TARGETED_PREPROCESSING_COMPLETE,
    NEEDS_SCHEMA_REVISION,
    SCHEMA_REVISION_REASON,
)

from ..models.schemas import (
    PipelinePhase,
    SessionInfo,
    SessionState,
    WebSocketMessage,
    WebSocketMessageType,
)


class PipelineSession:
    """
    Represents a single pipeline session with state management.
    """

    def __init__(self, session_id: str):
        self.id = session_id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.phase = PipelinePhase.IDLE
        self.state: Dict[str, Any] = {}
        self.messages: list = []
        self.llm = get_adk_llm()
        self._current_caller = None
        self._cancel_requested = False
        # Cache agents and callers to maintain conversation history
        self._agents: Dict[str, Any] = {}
        self._callers: Dict[str, Any] = {}

    def to_info(self) -> SessionInfo:
        """Convert to SessionInfo model."""
        return SessionInfo(
            id=self.id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            phase=self.phase,
            state=SessionState(
                approved_user_goal=self.state.get("approved_user_goal"),
                approved_files=self.state.get("approved_files"),
                proposed_construction_plan=self.state.get("proposed_construction_plan"),
                approved_construction_plan=self.state.get("approved_construction_plan"),
                feedback=self.state.get("feedback"),
            ),
            message_count=len(self.messages),
        )

    def cancel(self):
        """Request cancellation of current operation."""
        self._cancel_requested = True

    async def run_phase_with_streaming(
        self,
        phase: PipelinePhase,
        user_message: str
    ) -> AsyncGenerator[WebSocketMessage, None]:
        """
        Run a pipeline phase with event streaming.

        Args:
            phase: The phase to run
            user_message: User's input message

        Yields:
            WebSocketMessage for each event
        """
        self._cancel_requested = False
        self.phase = phase
        self.updated_at = datetime.now()

        # Reset extraction progress counters for new phase
        self._extraction_counter = 0
        self._extraction_total = 0

        # Notify phase change
        yield WebSocketMessage(
            type=WebSocketMessageType.PHASE_CHANGE,
            phase=phase,
        )

        try:
            if phase == PipelinePhase.USER_INTENT:
                async for msg in self._run_user_intent(user_message):
                    yield msg
            elif phase == PipelinePhase.FILE_SUGGESTION:
                async for msg in self._run_file_suggestion(user_message):
                    yield msg
            # New Data Cleaning Phase
            elif phase == PipelinePhase.DATA_CLEANING:
                async for msg in self._run_data_cleaning(user_message):
                    yield msg
            # Schema-First Pipeline Phases (Super Coordinator mode)
            elif phase == PipelinePhase.SCHEMA_PREPROCESS_COORDINATOR:
                async for msg in self._run_schema_preprocess_coordinator(user_message):
                    yield msg
            # Schema-First Pipeline Phases (Separate phases)
            elif phase == PipelinePhase.SCHEMA_DESIGN:
                async for msg in self._run_schema_design(user_message):
                    yield msg
            elif phase == PipelinePhase.TARGETED_PREPROCESSING:
                async for msg in self._run_targeted_preprocessing(user_message):
                    yield msg
            elif phase == PipelinePhase.CONSTRUCTION_PLAN:
                async for msg in self._run_construction_plan(user_message):
                    yield msg
            # Legacy Pipeline Phases (still supported)
            elif phase == PipelinePhase.DATA_PREPROCESSING:
                async for msg in self._run_data_preprocessing(user_message):
                    yield msg
            elif phase == PipelinePhase.SCHEMA_PROPOSAL:
                async for msg in self._run_schema_proposal(user_message):
                    yield msg
            # Common Phases
            elif phase == PipelinePhase.CONSTRUCTION:
                async for msg in self._run_construction():
                    yield msg
            elif phase == PipelinePhase.QUERY:
                async for msg in self._run_query(user_message):
                    yield msg

        except Exception as e:
            self.phase = PipelinePhase.ERROR
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error=str(e),
            )

    async def _get_or_create_caller(self, agent, agent_key: str):
        """Get existing caller or create a new one for the agent."""
        if agent_key not in self._callers:
            caller = await make_agent_caller(agent, self.state)
            self._callers[agent_key] = caller
            self._agents[agent_key] = agent
        return self._callers[agent_key]

    async def _run_agent_with_streaming(
        self,
        agent,
        message: str,
        agent_name: str,
        max_retries: int = 2
    ) -> AsyncGenerator[WebSocketMessage, None]:
        """
        Run an agent and stream events with retry support for transient errors.

        Args:
            agent: The ADK agent to run
            message: User message
            agent_name: Name for display
            max_retries: Maximum number of retries for transient errors (default: 2)

        Yields:
            WebSocketMessage for each event
        """
        import json

        # Use agent name as key to reuse caller across messages
        caller = await self._get_or_create_caller(agent, agent.name)
        self._current_caller = caller

        # Prepare user message
        content = types.Content(
            role='user',
            parts=[types.Part(text=message)]
        )

        # Retry logic for transient errors (JSON parsing, etc.)
        retry_count = 0
        last_error = None
        success = False

        while retry_count <= max_retries and not success:
            try:
                # Configure RunConfig with higher LLM call limit
                run_config = RunConfig(max_llm_calls=2000)

                # Stream events
                async for event in caller.runner.run_async(
                    user_id=caller.user_id,
                    session_id=caller.session_id,
                    new_message=content,
                    run_config=run_config
                ):
                    if self._cancel_requested:
                        yield WebSocketMessage(
                            type=WebSocketMessageType.ERROR,
                            error="Operation cancelled",
                        )
                        return

                    # Process event content - check for tool calls and text
                    event_content = ""
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            # Check for function call (tool invocation)
                            if hasattr(part, 'function_call') and part.function_call:
                                tool_name = part.function_call.name
                                tool_args = part.function_call.args or {}

                                # Build tool display message with key arguments
                                tool_display = f"🔧 {tool_name}"

                                # Extract relevant arguments for display
                                display_args = []
                                # File-related arguments
                                if 'file_path' in tool_args:
                                    display_args.append(f"file: {tool_args['file_path']}")
                                elif 'file_name' in tool_args:
                                    display_args.append(f"file: {tool_args['file_name']}")
                                # Column-related arguments
                                if 'column_name' in tool_args:
                                    display_args.append(f"column: {tool_args['column_name']}")
                                # Node/Entity-related arguments
                                if 'node_label' in tool_args:
                                    display_args.append(f"node: {tool_args['node_label']}")
                                elif 'label' in tool_args:
                                    display_args.append(f"label: {tool_args['label']}")
                                # Relationship-related arguments
                                if 'relationship_type' in tool_args:
                                    display_args.append(f"rel: {tool_args['relationship_type']}")

                                if display_args:
                                    tool_display += f" ({', '.join(display_args)})"

                                # Send tool call progress message
                                yield WebSocketMessage(
                                    type=WebSocketMessageType.AGENT_EVENT,
                                    content=tool_display,
                                    author="Tool",
                                    is_final=False,
                                )

                                # Track progress for different phase tools
                                # Schema Design phase tools
                                schema_design_tools = [
                                    'sample_raw_file_structure',
                                    'detect_potential_entities',
                                    'propose_node_type',
                                    'propose_relationship_type',
                                    'identify_text_feedback_columns',
                                    'sample_text_column',
                                    'analyze_text_column_entities',
                                    'analyze_text_column_relationships',
                                    'add_text_entity_to_schema',
                                    'add_text_relationship_to_schema',
                                    'standardize_column_names',
                                ]
                                # Extraction/Preprocessing phase tools
                                extraction_tools = [
                                    'extract_entities_from_text_column',
                                    'extract_relationships_from_text_column',
                                    'extract_node_data',
                                    'extract_relationship_data',
                                ]

                                if tool_name in schema_design_tools or tool_name in extraction_tools:
                                    # Increment extraction counter for progress tracking
                                    if not hasattr(self, '_extraction_counter'):
                                        self._extraction_counter = 0
                                        self._extraction_total = 0
                                    self._extraction_counter += 1

                                    # Get current item description
                                    file_path = tool_args.get('file_path', '')
                                    file_name = file_path.split('/')[-1] if file_path else ''
                                    item_desc = (
                                        tool_args.get('column_name', '') or
                                        tool_args.get('node_label', '') or
                                        tool_args.get('label', '') or
                                        file_name or
                                        'Processing'
                                    )

                                    # Get total: Read from LIVE ADK session state (not cached self.state)
                                    # The Agent writes to tool_context.state which is ADK session state
                                    try:
                                        live_session = await caller.get_session()
                                        live_state = dict(live_session.state) if live_session else {}
                                    except Exception as e:
                                        logger.warning(f"Failed to get live session: {e}")
                                        live_state = self.state  # Fallback to cached state

                                    agent_total = live_state.get('progress_total', 0)
                                    logger.info(f"[Progress] Tool: {tool_name}, agent_total from live_state: {agent_total}, current _extraction_total: {self._extraction_total}")
                                    if agent_total > 0:
                                        # Agent has set a specific total - use it
                                        logger.info(f"[Progress] Using agent-set total: {agent_total}")
                                        self._extraction_total = agent_total
                                    elif self._extraction_total == 0:
                                        # No Agent-set value and no previous estimate - use fallback estimate
                                        if tool_name in schema_design_tools:
                                            # Schema design: estimate based on approved files
                                            approved_files = live_state.get('approved_files', [])
                                            self._extraction_total = max(len(approved_files) * 5, 10)
                                        elif live_state.get('approved_target_schema'):
                                            # Extraction: estimate based on schema
                                            schema = live_state.get('approved_target_schema', {})
                                            nodes = len(schema.get('nodes', {}))
                                            rels = len(schema.get('relationships', {}))
                                            self._extraction_total = max((nodes + rels) * 2, 10)

                                    # Use the total (Agent-set or estimated), default to 20 if still 0
                                    total = self._extraction_total if self._extraction_total > 0 else 20
                                    # Cap progress at 99% until completion (allow current > total without changing total)
                                    progress = min(int(self._extraction_counter / total * 100), 99)

                                    yield WebSocketMessage(
                                        type=WebSocketMessageType.STATE_UPDATE,
                                        progress=progress,
                                        progress_current=self._extraction_counter,
                                        progress_total=total,
                                        progress_item=item_desc[:50] if item_desc else tool_name,
                                    )
                            # Extract text content
                            elif hasattr(part, 'text') and part.text:
                                event_content = part.text

                    # Only send agent event if there's text content
                    if event_content:
                        yield WebSocketMessage(
                            type=WebSocketMessageType.AGENT_EVENT,
                            content=event_content,
                            author=event.author,
                            is_final=event.is_final_response(),
                        )

                    # Check for final response
                    # For LoopAgents: we should NOT break on sub-agent final responses
                    # Instead, wait for escalate=True event from the stop checker
                    # For regular LlmAgents: break on final response with content

                    # Check for escalate signal (used by LoopAgent's stop checkers)
                    if event.actions and event.actions.escalate:
                        # For nested LoopAgents (like coordinator), we need to distinguish
                        # between sub-agent escalate and coordinator-level escalate.
                        # Only break if:
                        # 1. This is NOT a nested coordinator (has nested LoopAgent sub-agents)
                        # 2. OR the escalate comes from the coordinator's own status checker
                        is_nested_coordinator = (
                            isinstance(agent, LoopAgent) and
                            hasattr(agent, 'sub_agents') and
                            any(isinstance(sub, LoopAgent) for sub in agent.sub_agents)
                        )

                        if is_nested_coordinator:
                            # For nested coordinators, only break on coordinator status checker's escalate
                            # The coordinator's status checker is named "coordinator_status_checker"
                            # (set in create_schema_preprocess_coordinator)
                            # Sub-agent status checkers are named like "SchemaDesignStopChecker"
                            author = event.author or ""
                            # Only break on the coordinator's own status checker
                            # It's explicitly named "coordinator_status_checker"
                            is_coordinator_escalate = "coordinator_status_checker" in author.lower()
                            if is_coordinator_escalate:
                                print(f"### Coordinator escalate from: {author}, breaking")
                                break
                            # Otherwise, ignore sub-agent escalates - they just mean the sub-loop finished
                            print(f"### Ignoring sub-agent escalate from: {author}")
                        else:
                            break

                    # For non-LoopAgents, break on final response with content
                    # LoopAgents have sub_agents attribute
                    is_loop_agent = isinstance(agent, LoopAgent)
                    if event.is_final_response() and event_content and not is_loop_agent:
                        break

                # If we get here, the run completed successfully
                success = True

            except (json.JSONDecodeError, ValueError) as e:
                # JSON parsing error - likely from malformed LLM output
                error_msg = str(e)
                last_error = e
                retry_count += 1

                if retry_count <= max_retries:
                    # Notify user about retry
                    yield WebSocketMessage(
                        type=WebSocketMessageType.AGENT_EVENT,
                        content=f"⚠️ JSON parsing error (attempt {retry_count}/{max_retries + 1}): {error_msg[:100]}... Retrying...",
                        author="System",
                        is_final=False,
                    )
                    # Small delay before retry
                    await asyncio.sleep(0.5)
                else:
                    # Max retries exceeded
                    yield WebSocketMessage(
                        type=WebSocketMessageType.ERROR,
                        error=f"JSON parsing error after {max_retries + 1} attempts: {error_msg}",
                    )
                    return

            except Exception as e:
                # Other errors - check if it's a JSON-related error in the message
                error_msg = str(e)
                if "Expecting" in error_msg and ("delimiter" in error_msg or "value" in error_msg):
                    # This looks like a JSON parsing error
                    last_error = e
                    retry_count += 1

                    if retry_count <= max_retries:
                        yield WebSocketMessage(
                            type=WebSocketMessageType.AGENT_EVENT,
                            content=f"⚠️ Format error (attempt {retry_count}/{max_retries + 1}): {error_msg[:100]}... Retrying...",
                            author="System",
                            is_final=False,
                        )
                        await asyncio.sleep(0.5)
                    else:
                        yield WebSocketMessage(
                            type=WebSocketMessageType.ERROR,
                            error=f"Format error after {max_retries + 1} attempts: {error_msg}",
                        )
                        return
                else:
                    # Non-retryable error - re-raise
                    raise

        # Update state from session
        session = await caller.get_session()
        self.state = dict(session.state)

        # Send state update
        yield WebSocketMessage(
            type=WebSocketMessageType.STATE_UPDATE,
            state=self.state,
        )

    def _get_or_create_agent(self, agent_key: str, creator_func):
        """Get existing agent or create a new one."""
        if agent_key not in self._agents:
            self._agents[agent_key] = creator_func(self.llm)
        return self._agents[agent_key]

    async def _run_user_intent(self, user_message: str) -> AsyncGenerator[WebSocketMessage, None]:
        """Run user intent phase."""
        # Reuse the same agent instance across messages
        agent = self._get_or_create_agent("user_intent", create_user_intent_agent)

        # Run agent with user's message
        async for msg in self._run_agent_with_streaming(agent, user_message, "User Intent Agent"):
            yield msg

        # Check if user goal was approved after this interaction
        if "approved_user_goal" in self.state:
            # Phase complete - goal was approved
            yield WebSocketMessage(
                type=WebSocketMessageType.PHASE_COMPLETE,
                phase=PipelinePhase.USER_INTENT,
                state=self.state,
            )
        # Otherwise, wait for user to continue the conversation (confirmation/refinement)

    async def _run_file_suggestion(self, user_message: str = "What files can we use for import?") -> AsyncGenerator[WebSocketMessage, None]:
        """Run file suggestion phase."""
        if "approved_user_goal" not in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error="User intent phase must be completed first",
            )
            return

        # Reuse the same agent instance across messages
        agent = self._get_or_create_agent("file_suggestion", create_file_suggestion_agent)

        # Run agent with user's message
        async for msg in self._run_agent_with_streaming(
            agent, user_message, "File Suggestion Agent"
        ):
            yield msg

        # Check if files were approved after this interaction
        if "approved_files" in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.PHASE_COMPLETE,
                phase=PipelinePhase.FILE_SUGGESTION,
                state=self.state,
            )
        # Otherwise, wait for user to continue the conversation

    async def _run_data_cleaning(self, user_message: str = "Analyze and clean the data files") -> AsyncGenerator[WebSocketMessage, None]:
        """Run data cleaning phase.

        This phase cleans raw data files before schema proposal:
        - Removes meaningless columns (empty, constant, unnamed)
        - Removes invalid rows (mostly empty, duplicates)
        - Detects column types for schema design
        """
        if "approved_files" not in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error="File suggestion phase must be completed first",
            )
            return

        if "data_cleaning_feedback" not in self.state:
            self.state["data_cleaning_feedback"] = ""

        # Use the data cleaning loop agent
        loop = self._get_or_create_agent("data_cleaning", create_data_cleaning_loop)

        async for msg in self._run_agent_with_streaming(
            loop, user_message, "Data Cleaning Agent"
        ):
            yield msg

        # Check if data cleaning was completed after this interaction
        if self.state.get(DATA_CLEANING_COMPLETE_KEY):
            yield WebSocketMessage(
                type=WebSocketMessageType.PHASE_COMPLETE,
                phase=PipelinePhase.DATA_CLEANING,
                state=self.state,
            )
        # Otherwise, wait for user to continue the conversation

    async def _run_data_preprocessing(self, user_message: str = "Analyze and preprocess the approved files") -> AsyncGenerator[WebSocketMessage, None]:
        """Run data preprocessing phase using the new Survey Preprocessing Coordinator."""
        if "approved_files" not in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error="File suggestion phase must be completed first",
            )
            return

        # Check if this is survey data (look for survey patterns in approved files)
        approved_files = self.state.get("approved_files", [])
        is_survey_data = any(
            "survey" in f.lower() or "调研" in f
            for f in approved_files
        )

        if is_survey_data:
            # Use the new Survey Preprocessing Coordinator with LLM-based classification
            agent = self._get_or_create_agent(
                "survey_preprocessing",
                create_survey_preprocessing_coordinator
            )
        else:
            # Fall back to the generic data preprocessing agent for non-survey data
            agent = self._get_or_create_agent(
                "data_preprocessing",
                create_data_preprocessing_agent
            )

        # Run agent with user's message
        async for msg in self._run_agent_with_streaming(
            agent, user_message, "Data Preprocessing Agent"
        ):
            yield msg

        # Check if preprocessing was completed after this interaction
        if self.state.get("preprocessing_complete"):
            yield WebSocketMessage(
                type=WebSocketMessageType.PHASE_COMPLETE,
                phase=PipelinePhase.DATA_PREPROCESSING,
                state=self.state,
            )
        # Otherwise, wait for user to continue the conversation (e.g., approve transformation)

    async def _run_schema_proposal(self, user_message: str = "How can these files be imported?") -> AsyncGenerator[WebSocketMessage, None]:
        """Run schema proposal phase."""
        if "approved_files" not in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error="File suggestion phase must be completed first",
            )
            return

        if "feedback" not in self.state:
            self.state["feedback"] = ""

        # Reuse the same loop agent instance
        loop = self._get_or_create_agent("schema_proposal", create_schema_refinement_loop)

        async for msg in self._run_agent_with_streaming(
            loop, user_message, "Schema Proposal Agent"
        ):
            yield msg

        # Check if schema was explicitly approved by the agent calling approve_proposed_construction_plan
        # Do NOT auto-approve - let the agent and user iterate on the schema
        if APPROVED_CONSTRUCTION_PLAN in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.PHASE_COMPLETE,
                phase=PipelinePhase.SCHEMA_PROPOSAL,
                state=self.state,
            )
        # Otherwise, continue the conversation - user may provide feedback to refine the schema

    # ========== Schema-First Pipeline Phase Handlers ==========

    async def _run_schema_design(self, user_message: str = "Design a target schema for this data") -> AsyncGenerator[WebSocketMessage, None]:
        """Run schema design phase (Schema-First Pipeline)."""
        if "approved_files" not in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error="File suggestion phase must be completed first",
            )
            return

        if "schema_design_feedback" not in self.state:
            self.state["schema_design_feedback"] = ""

        # Use the schema design loop agent
        loop = self._get_or_create_agent("schema_design", create_schema_design_loop)

        async for msg in self._run_agent_with_streaming(
            loop, user_message, "Schema Design Agent"
        ):
            yield msg

        # Check if target schema was approved
        if APPROVED_TARGET_SCHEMA_KEY in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.PHASE_COMPLETE,
                phase=PipelinePhase.SCHEMA_DESIGN,
                state=self.state,
            )
        # Otherwise, continue the conversation - user may provide feedback

    async def _run_targeted_preprocessing(self, user_message: str = "Extract entities and relationships according to the schema") -> AsyncGenerator[WebSocketMessage, None]:
        """Run targeted preprocessing phase (Schema-First Pipeline).

        Handles rollback to schema design with user confirmation:
        - If rollback_pending is set, check if user confirmed rollback
        - If user says "rollback"/"回滚", set rollback_confirmed and return
        - Otherwise, continue preprocessing
        """
        if APPROVED_TARGET_SCHEMA_KEY not in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error="Schema design phase must be completed first",
            )
            return

        # Check if rollback is pending and needs user confirmation
        if self.state.get("rollback_pending"):
            # Check if user is confirming rollback
            user_msg_lower = user_message.lower().strip()
            rollback_keywords = ["rollback", "回滚", "重新设计", "redesign", "yes", "确认", "是"]

            if any(keyword in user_msg_lower for keyword in rollback_keywords):
                # User confirmed rollback
                self.state["rollback_confirmed"] = True
                revision_reason = self.state.get(SCHEMA_REVISION_REASON, "User requested schema revision")
                yield WebSocketMessage(
                    type=WebSocketMessageType.AGENT_RESPONSE,
                    content=f"Rolling back to Schema Design phase.\n\nReason: {revision_reason}\n\nPlease redesign the schema and approve it to continue.",
                    author="Pipeline",
                    is_final=True,
                )
                # The next call to _determine_next_phase will handle the actual rollback
                return
            else:
                # User chose to continue without rollback
                self.state["rollback_pending"] = False
                self.state[NEEDS_SCHEMA_REVISION] = False
                yield WebSocketMessage(
                    type=WebSocketMessageType.AGENT_EVENT,
                    content="Continuing with current schema. You can manually fix extraction issues.",
                    author="Pipeline",
                    is_final=False,
                )

        if "preprocessing_feedback" not in self.state:
            self.state["preprocessing_feedback"] = ""

        # Use the targeted preprocessing loop agent
        loop = self._get_or_create_agent("targeted_preprocessing", create_targeted_preprocessing_loop)

        async for msg in self._run_agent_with_streaming(
            loop, user_message, "Targeted Preprocessing Agent"
        ):
            yield msg

        # Check for rollback request after preprocessing run
        if self.state.get(NEEDS_SCHEMA_REVISION) and not self.state.get("rollback_pending"):
            # Preprocessing agent requested schema revision - prompt user
            revision_reason = self.state.get(SCHEMA_REVISION_REASON, "Schema issues detected during extraction")
            self.state["rollback_pending"] = True
            yield WebSocketMessage(
                type=WebSocketMessageType.AGENT_RESPONSE,
                content=f"**Schema revision needed:**\n{revision_reason}\n\nType 'rollback' or '回滚' to return to Schema Design phase, or continue with manual fixes.",
                author="Preprocessing Agent",
                is_final=True,
            )
            return

        # Check if preprocessing was completed
        if self.state.get(TARGETED_PREPROCESSING_COMPLETE):
            yield WebSocketMessage(
                type=WebSocketMessageType.PHASE_COMPLETE,
                phase=PipelinePhase.TARGETED_PREPROCESSING,
                state=self.state,
            )
        # Otherwise, continue the conversation

    async def _run_schema_preprocess_coordinator(self, user_message: str = "Design schema and extract data") -> AsyncGenerator[WebSocketMessage, None]:
        """Run schema-preprocess coordinator phase (Super Coordinator mode).

        This combines Schema Design and Targeted Preprocessing into a single
        coordinated flow with bidirectional feedback support. The coordinator
        automatically handles rollback from preprocessing to schema design when
        data extraction issues are discovered.

        Flow:
        1. Schema Design Loop runs until schema is approved
        2. Schema approval is checked
        3. Targeted Preprocessing Loop extracts data based on schema
        4. If preprocessing finds issues → automatic rollback to schema design
        5. Repeat until preprocessing completes successfully
        """
        if "approved_files" not in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error="File suggestion phase must be completed first",
            )
            return

        # Initialize feedback state if not present
        if "schema_design_feedback" not in self.state:
            self.state["schema_design_feedback"] = ""
        if "preprocessing_feedback" not in self.state:
            self.state["preprocessing_feedback"] = ""

        # Use the schema-preprocess coordinator (super coordinator)
        coordinator = self._get_or_create_agent(
            "schema_preprocess_coordinator",
            create_schema_preprocess_coordinator
        )

        async for msg in self._run_agent_with_streaming(
            coordinator, user_message, "Schema-Preprocess Coordinator"
        ):
            yield msg

        # Check if both schema was approved AND preprocessing completed
        if APPROVED_TARGET_SCHEMA_KEY in self.state and self.state.get(TARGETED_PREPROCESSING_COMPLETE):
            yield WebSocketMessage(
                type=WebSocketMessageType.PHASE_COMPLETE,
                phase=PipelinePhase.SCHEMA_PREPROCESS_COORDINATOR,
                state=self.state,
            )
        # The coordinator handles rollback internally, so we don't need
        # to manage it here. Just continue the conversation if not complete.

    async def _run_construction_plan(self, user_message: str = "Generate construction rules from the schema") -> AsyncGenerator[WebSocketMessage, None]:
        """Run construction plan generation phase (Schema-First Pipeline).

        This phase auto-generates construction rules from the approved target schema
        and extracted data, then immediately proceeds to construction.
        """
        if not self.state.get(TARGETED_PREPROCESSING_COMPLETE):
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error="Targeted preprocessing phase must be completed first",
            )
            return

        yield WebSocketMessage(
            type=WebSocketMessageType.AGENT_EVENT,
            content="Generating construction rules from target schema...",
            author="Construction Plan Generator",
        )

        # The construction plan should already be generated by the preprocessing agent
        # via generate_construction_rules(). Just verify it exists and proceed.
        from src.tools.kg_construction import PROPOSED_CONSTRUCTION_PLAN

        if PROPOSED_CONSTRUCTION_PLAN not in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error="Construction plan not found. Run targeted preprocessing first.",
            )
            return

        # Auto-approve the construction plan for Schema-First pipeline
        self.state[APPROVED_CONSTRUCTION_PLAN] = self.state[PROPOSED_CONSTRUCTION_PLAN]

        yield WebSocketMessage(
            type=WebSocketMessageType.AGENT_RESPONSE,
            content=f"Construction plan generated from target schema. Ready to build the knowledge graph.",
            author="Construction Plan Generator",
            is_final=True,
        )

        yield WebSocketMessage(
            type=WebSocketMessageType.PHASE_COMPLETE,
            phase=PipelinePhase.CONSTRUCTION_PLAN,
            state=self.state,
        )

    # ========== End Schema-First Pipeline Phase Handlers ==========

    async def _run_construction(self) -> AsyncGenerator[WebSocketMessage, None]:
        """Run KG construction phase."""
        if APPROVED_CONSTRUCTION_PLAN not in self.state:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error="Schema proposal phase must be completed first",
            )
            return

        yield WebSocketMessage(
            type=WebSocketMessageType.AGENT_EVENT,
            content="Starting knowledge graph construction...",
            author="KG Builder",
        )

        try:
            construction_plan = self.state[APPROVED_CONSTRUCTION_PLAN]
            result = build_domain_graph(construction_plan)

            self.state["construction_result"] = result
            self.state["construction_complete"] = True  # Mark construction as done
            self.phase = PipelinePhase.QUERY  # Transition to query phase

            yield WebSocketMessage(
                type=WebSocketMessageType.AGENT_RESPONSE,
                content=f"Knowledge graph construction complete!\n\n{result}\n\nYou can now ask questions about the knowledge graph.",
                author="KG Builder",
                is_final=True,
            )

            yield WebSocketMessage(
                type=WebSocketMessageType.PHASE_COMPLETE,
                phase=PipelinePhase.CONSTRUCTION,
                state=self.state,
            )

        except Exception as e:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error=f"Construction failed: {str(e)}",
            )

    async def _run_query(self, user_message: str) -> AsyncGenerator[WebSocketMessage, None]:
        """Run knowledge graph query phase using Cypher Generation Loop."""
        if not self.state.get("construction_complete"):
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error="Knowledge graph must be constructed first before querying",
            )
            return

        # Use Cypher Generation Loop for validated query generation
        # This uses Silicon Cloud Qwen3-Coder for generation and validates against Neo4j
        loop = self._get_or_create_agent(
            "cypher_generation_loop",
            lambda llm: create_cypher_generation_loop()  # Uses its own LLMs
        )

        # Clear previous feedback before new query
        self.state["cypher_feedback"] = ""

        # Run the loop agent with user's query
        async for msg in self._run_agent_with_streaming(
            loop, user_message, "Cypher Generation Loop"
        ):
            yield msg

        # After loop completes, format the results for display
        validated_query = self.state.get("validated_cypher_query", "")
        query_result = self.state.get("query_result", [])

        if validated_query and query_result:
            # Send a formatted response with the query and results
            result_summary = f"**Generated Cypher Query:**\n```cypher\n{validated_query}\n```\n\n"
            result_summary += f"**Results ({len(query_result)} rows):**\n"
            # Format first few results
            for i, row in enumerate(query_result[:10]):
                result_summary += f"- {row}\n"
            if len(query_result) > 10:
                result_summary += f"... and {len(query_result) - 10} more rows"

            yield WebSocketMessage(
                type=WebSocketMessageType.AGENT_RESPONSE,
                content=result_summary,
                author="Cypher Query Engine",
                is_final=True,
            )

        # Query phase stays active - user can keep asking questions
        # No PHASE_COMPLETE is sent as this is an ongoing interaction phase


class SessionManager:
    """
    Manages pipeline sessions.
    """

    def __init__(self):
        self._sessions: Dict[str, PipelineSession] = {}

    def create_session(self) -> PipelineSession:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session = PipelineSession(session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[PipelineSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[SessionInfo]:
        """List all sessions."""
        return [s.to_info() for s in self._sessions.values()]


class PipelineService:
    """
    High-level service for pipeline operations.
    """

    def __init__(self):
        self.session_manager = SessionManager()

    def create_session(self) -> PipelineSession:
        """Create a new pipeline session."""
        return self.session_manager.create_session()

    def get_session(self, session_id: str) -> Optional[PipelineSession]:
        """Get a session by ID."""
        return self.session_manager.get_session(session_id)

    async def process_message(
        self,
        session_id: str,
        message: str,
        phase: Optional[PipelinePhase] = None
    ) -> AsyncGenerator[WebSocketMessage, None]:
        """
        Process a user message and stream responses.

        Args:
            session_id: Session ID
            message: User message
            phase: Optional phase override

        Yields:
            WebSocketMessage events
        """
        session = self.get_session(session_id)
        if not session:
            yield WebSocketMessage(
                type=WebSocketMessageType.ERROR,
                error=f"Session not found: {session_id}",
            )
            return

        # Determine phase
        if phase is None:
            phase = self._determine_next_phase(session)

        async for msg in session.run_phase_with_streaming(phase, message):
            yield msg

    def _determine_next_phase(self, session: PipelineSession) -> PipelinePhase:
        """Determine the next phase based on current state.

        Supports four pipeline modes:
        - Schema-First with Separate Phases (default, pipeline_mode="schema_first_separate"):
          USER_INTENT → FILE_SUGGESTION → DATA_CLEANING → SCHEMA_DESIGN → TARGETED_PREPROCESSING → CONSTRUCTION_PLAN → CONSTRUCTION → QUERY
          - Uses single-layer LoopAgents (simpler, more stable)
          - Rollback requires user confirmation
        - Schema-First with Super Coordinator (pipeline_mode="super_coordinator"):
          USER_INTENT → FILE_SUGGESTION → DATA_CLEANING → SCHEMA_PREPROCESS_COORDINATOR → CONSTRUCTION_PLAN → CONSTRUCTION → QUERY
          - Uses nested LoopAgents (more complex, handles rollback internally)
        - Schema-First with Super Coordinator (no cleaning) (pipeline_mode="super_coordinator_no_cleaning"):
          USER_INTENT → FILE_SUGGESTION → SCHEMA_PREPROCESS_COORDINATOR → CONSTRUCTION_PLAN → CONSTRUCTION → QUERY
        - Legacy (pipeline_mode="legacy"):
          USER_INTENT → FILE_SUGGESTION → DATA_PREPROCESSING → SCHEMA_PROPOSAL → CONSTRUCTION → QUERY

        For separate phases mode (default), rollback requires user to confirm
        by sending "rollback" or "回滚" message. The NEEDS_SCHEMA_REVISION flag
        only triggers a prompt, not automatic rollback.
        """
        # Check state to determine which phase to run
        # Order matters! Check from later phases to earlier phases

        pipeline_mode = session.state.get("pipeline_mode", "schema_first_separate")

        # ========== Rollback Detection (for separate phases mode only) ==========
        # The super coordinator handles rollback internally, but for separate phases
        # mode, rollback requires user confirmation.
        #
        # Flow:
        # 1. Preprocessing agent calls request_schema_revision() → sets NEEDS_SCHEMA_REVISION
        # 2. Pipeline detects flag → sets rollback_pending and prompts user
        # 3. User sends "rollback"/"回滚" → Pipeline clears state and returns to SCHEMA_DESIGN
        # 4. User sends anything else → Continue in TARGETED_PREPROCESSING (user can fix manually)
        if pipeline_mode == "schema_first_separate" and session.state.get(NEEDS_SCHEMA_REVISION):
            # Check if user has confirmed rollback
            user_confirmed_rollback = session.state.get("rollback_confirmed", False)

            if user_confirmed_rollback:
                # User confirmed - perform actual rollback
                session.state[NEEDS_SCHEMA_REVISION] = False
                session.state["rollback_confirmed"] = False
                session.state["rollback_pending"] = False
                # Keep the revision reason so schema design agent knows what to fix
                # Clear approved schema to allow redesign
                if APPROVED_TARGET_SCHEMA_KEY in session.state:
                    del session.state[APPROVED_TARGET_SCHEMA_KEY]
                # Clear preprocessing state
                session.state.pop(TARGETED_PREPROCESSING_COMPLETE, None)
                session.state.pop("targeted_extraction_results", None)
                session.state.pop("targeted_entity_maps", None)
                session.state.pop("targeted_relationship_data", None)
                session.state.pop("generated_files", None)
                # Reset schema design feedback to force the loop to re-run
                # If we don't reset this, CheckSchemaDesignStatus will see "valid" and auto-approve
                session.state["schema_design_feedback"] = "rollback"
                session.state.pop("schema_design_phase_done", None)
                # Clear any cached agents to force fresh start
                session._callers.pop("targeted_preprocessing", None)
                session._agents.pop("targeted_preprocessing", None)
                session._callers.pop("schema_design", None)
                session._agents.pop("schema_design", None)
                # Return to schema design phase
                return PipelinePhase.SCHEMA_DESIGN
            else:
                # Rollback not yet confirmed - set pending flag
                # The preprocessing handler will check for this
                session.state["rollback_pending"] = True

        # QUERY phase - common endpoint for all pipelines
        if session.state.get("construction_complete"):
            return PipelinePhase.QUERY

        # CONSTRUCTION phase - common for all pipelines
        if APPROVED_CONSTRUCTION_PLAN in session.state:
            return PipelinePhase.CONSTRUCTION

        # ========== Schema-First Pipeline Detection ==========
        # Check for Schema-First pipeline specific state keys

        # CONSTRUCTION_PLAN phase (Schema-First only, both modes)
        if session.state.get(TARGETED_PREPROCESSING_COMPLETE):
            return PipelinePhase.CONSTRUCTION_PLAN

        # ========== Separate Phases Mode (default) - Schema First ==========
        # In Schema First mode, we skip DATA_CLEANING and go directly to SCHEMA_DESIGN
        # The TARGETED_PREPROCESSING phase handles schema-based cleaning before extraction
        if pipeline_mode == "schema_first_separate":
            # TARGETED_PREPROCESSING phase
            if APPROVED_TARGET_SCHEMA_KEY in session.state:
                return PipelinePhase.TARGETED_PREPROCESSING

            # SCHEMA_DESIGN phase - directly after files are approved (no data cleaning needed)
            if "approved_files" in session.state:
                return PipelinePhase.SCHEMA_DESIGN

        # ========== Super Coordinator Mode (default, with data cleaning) ==========
        if pipeline_mode == "super_coordinator" or pipeline_mode == "schema_first":
            # SCHEMA_PREPROCESS_COORDINATOR phase - after data cleaning is complete
            if session.state.get(DATA_CLEANING_COMPLETE_KEY):
                return PipelinePhase.SCHEMA_PREPROCESS_COORDINATOR

            # DATA_CLEANING phase - after files are approved
            if "approved_files" in session.state:
                return PipelinePhase.DATA_CLEANING

        # ========== Super Coordinator Mode (no data cleaning) ==========
        if pipeline_mode == "super_coordinator_no_cleaning":
            # SCHEMA_PREPROCESS_COORDINATOR phase - directly after files approved
            if "approved_files" in session.state:
                return PipelinePhase.SCHEMA_PREPROCESS_COORDINATOR

        # ========== Legacy Pipeline Detection ==========
        if pipeline_mode == "legacy":
            # SCHEMA_PROPOSAL phase (Legacy only)
            if session.state.get("preprocessing_complete"):
                return PipelinePhase.SCHEMA_PROPOSAL

            # DATA_PREPROCESSING phase (Legacy only)
            if "approved_files" in session.state:
                return PipelinePhase.DATA_PREPROCESSING

        # ========== Default behavior for unrecognized modes ==========
        # If files are approved and mode is unknown, default to data cleaning first
        if "approved_files" in session.state:
            if session.state.get(DATA_CLEANING_COMPLETE_KEY):
                return PipelinePhase.SCHEMA_PREPROCESS_COORDINATOR
            return PipelinePhase.DATA_CLEANING

        # ========== Common Initial Phases ==========

        # FILE_SUGGESTION phase
        if "approved_user_goal" in session.state:
            return PipelinePhase.FILE_SUGGESTION

        # USER_INTENT phase - starting point
        return PipelinePhase.USER_INTENT

"""
Agents module for Agentic KG.

Contains agent definitions for the multi-agent knowledge graph construction system.
"""

from .base import AgentCaller, make_agent_caller, run_agent_conversation
from .user_intent_agent import create_user_intent_agent
from .file_suggestion_agent import create_file_suggestion_agent
from .data_preprocessing_agent import create_data_preprocessing_agent
from .schema_proposal_agent import (
    create_schema_proposal_agent,
    create_schema_critic_agent,
    create_schema_refinement_loop,
    CheckStatusAndEscalate,
)
from .kg_builder_agent import create_kg_builder_agent

__all__ = [
    # Base
    "AgentCaller",
    "make_agent_caller",
    "run_agent_conversation",
    # User Intent
    "create_user_intent_agent",
    # File Suggestion
    "create_file_suggestion_agent",
    # Data Preprocessing
    "create_data_preprocessing_agent",
    # Schema Proposal
    "create_schema_proposal_agent",
    "create_schema_critic_agent",
    "create_schema_refinement_loop",
    "CheckStatusAndEscalate",
    # KG Builder
    "create_kg_builder_agent",
]

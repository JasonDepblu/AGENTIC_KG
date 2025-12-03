"""
Pydantic models for API requests and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# File Models

class FileInfo(BaseModel):
    """Information about a file in the import directory."""
    name: str
    path: str
    is_directory: bool
    size: Optional[int] = None
    modified: Optional[datetime] = None
    extension: Optional[str] = None


class FileListResponse(BaseModel):
    """Response for listing files."""
    files: List[FileInfo]
    total: int
    base_path: str


class FileContentResponse(BaseModel):
    """Response for file content preview."""
    path: str
    content: str
    lines: int
    truncated: bool


class FileUploadResponse(BaseModel):
    """Response for file upload."""
    path: str
    size: int
    message: str


# Chat/Session Models

class PipelinePhase(str, Enum):
    """Current phase of the KG pipeline."""
    IDLE = "idle"
    USER_INTENT = "user_intent"
    FILE_SUGGESTION = "file_suggestion"
    DATA_PREPROCESSING = "data_preprocessing"
    SCHEMA_PROPOSAL = "schema_proposal"
    CONSTRUCTION = "construction"
    QUERY = "query"  # GraphRAG query phase - after KG construction
    COMPLETE = "complete"
    ERROR = "error"


class MessageRole(str, Enum):
    """Role of a chat message."""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """A single chat message."""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    agent_name: Optional[str] = None
    phase: Optional[PipelinePhase] = None
    is_streaming: bool = False


class SessionState(BaseModel):
    """Current state of a pipeline session."""
    approved_user_goal: Optional[Dict[str, Any]] = None
    approved_files: Optional[List[str]] = None
    preprocessing_complete: Optional[bool] = None
    proposed_construction_plan: Optional[Dict[str, Any]] = None
    approved_construction_plan: Optional[Dict[str, Any]] = None
    feedback: Optional[str] = None


class SessionInfo(BaseModel):
    """Information about a chat session."""
    id: str
    created_at: datetime
    updated_at: datetime
    phase: PipelinePhase
    state: SessionState
    message_count: int


class SessionCreateRequest(BaseModel):
    """Request to create a new session."""
    name: Optional[str] = None


class SessionCreateResponse(BaseModel):
    """Response for session creation."""
    id: str
    created_at: datetime


# WebSocket Models

class WebSocketMessageType(str, Enum):
    """Types of WebSocket messages."""
    # Client -> Server
    MESSAGE = "message"
    APPROVE = "approve"
    CANCEL = "cancel"

    # Server -> Client
    AGENT_EVENT = "agent_event"
    AGENT_RESPONSE = "agent_response"
    PHASE_CHANGE = "phase_change"
    PHASE_COMPLETE = "phase_complete"
    STATE_UPDATE = "state_update"
    ERROR = "error"
    CONNECTED = "connected"


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: WebSocketMessageType
    content: Optional[str] = None
    phase: Optional[PipelinePhase] = None
    author: Optional[str] = None
    state: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    is_final: bool = False


# API Response Models

class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None

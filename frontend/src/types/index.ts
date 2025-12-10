// File types
export interface FileInfo {
  name: string;
  path: string;
  is_directory: boolean;
  size?: number;
  modified?: string;
  extension?: string;
  children?: FileInfo[];
}

export interface FileTree {
  name: string;
  path: string;
  is_directory: boolean;
  size?: number;
  extension?: string;
  children?: FileTree[];
}

// Chat types
export type MessageRole = 'user' | 'agent' | 'system';

export type PipelinePhase =
  | 'idle'
  | 'user_intent'
  | 'file_suggestion'
  // Data Cleaning phase (new)
  | 'data_cleaning'
  // Schema-First Pipeline phases
  | 'schema_design'
  | 'targeted_preprocessing'
  | 'construction_plan'
  // Super Coordinator mode (combines schema + preprocessing)
  | 'schema_preprocess_coordinator'
  // Legacy Pipeline phases
  | 'data_preprocessing'
  | 'schema_proposal'
  // Common phases
  | 'construction'
  | 'query'
  | 'complete'
  | 'error';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  agentName?: string;
  phase?: PipelinePhase;
  isStreaming?: boolean;
  isToolProgress?: boolean;  // Flag for tool call progress messages
  isCriticFeedback?: boolean;  // Flag for critic agent validation messages
}

// Session types
export interface SessionState {
  approved_user_goal?: Record<string, unknown>;
  approved_files?: string[];
  preprocessing_complete?: boolean;
  proposed_construction_plan?: Record<string, unknown>;
  approved_construction_plan?: Record<string, unknown>;
  feedback?: string;
}

export interface Session {
  id: string;
  created_at: string;
  updated_at: string;
  phase: PipelinePhase;
  state: SessionState;
  message_count: number;
}

// WebSocket message types
export type WebSocketMessageType =
  | 'message'
  | 'approve'
  | 'cancel'
  | 'agent_event'
  | 'agent_response'
  | 'phase_change'
  | 'phase_complete'
  | 'state_update'
  | 'error'
  | 'connected';

export interface WebSocketMessage {
  type: WebSocketMessageType;
  content?: string;
  phase?: PipelinePhase;
  author?: string;
  state?: Record<string, unknown>;
  error?: string;
  is_final?: boolean;
}

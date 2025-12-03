import { create } from 'zustand';
import type { ChatMessage, PipelinePhase, SessionState } from '../types';

interface ChatStore {
  // Session
  sessionId: string | null;
  phase: PipelinePhase;
  sessionState: SessionState;

  // Messages
  messages: ChatMessage[];
  isConnected: boolean;
  isLoading: boolean;

  // Actions
  setSessionId: (id: string | null) => void;
  setPhase: (phase: PipelinePhase) => void;
  setSessionState: (state: SessionState) => void;
  setConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;

  addMessage: (message: ChatMessage) => void;
  updateMessage: (id: string, content: string) => void;
  clearMessages: () => void;
  reset: () => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  // Initial state
  sessionId: null,
  phase: 'idle',
  sessionState: {},
  messages: [],
  isConnected: false,
  isLoading: false,

  // Actions
  setSessionId: (id) => set({ sessionId: id }),
  setPhase: (phase) => set({ phase }),
  setSessionState: (state) => set({ sessionState: state }),
  setConnected: (connected) => set({ isConnected: connected }),
  setLoading: (loading) => set({ isLoading: loading }),

  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),

  updateMessage: (id, content) =>
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, content, isStreaming: false } : msg
      ),
    })),

  clearMessages: () => set({ messages: [] }),

  reset: () =>
    set({
      sessionId: null,
      phase: 'idle',
      sessionState: {},
      messages: [],
      isConnected: false,
      isLoading: false,
    }),
}));

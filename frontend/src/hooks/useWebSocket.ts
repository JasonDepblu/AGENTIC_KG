import { useCallback, useEffect, useRef, useState } from 'react';
import { useChatStore } from '../stores/chatStore';
import type { WebSocketMessage, PipelinePhase } from '../types';

// Global WebSocket instance to persist across React StrictMode remounts
let globalWs: WebSocket | null = null;

// Session persistence using localStorage
const SESSION_STORAGE_KEY = 'agentic_kg_session_id';
let globalSessionId: string | null = localStorage.getItem(SESSION_STORAGE_KEY);

export function useWebSocket() {
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  const reconnectTimeoutRef = useRef<number | null>(null);

  const {
    setSessionId,
    setPhase,
    setSessionState,
    setConnected,
    addMessage,
    updateMessage,
    setLoading,
    clearMessages,
    setExtractionProgress,
    resetExtractionProgress,
  } = useChatStore();

  const messageIdRef = useRef(0);
  const currentMessageIdRef = useRef<string | null>(null);
  const toolProgressIdRef = useRef<string | null>(null);  // Track tool progress message
  const authorMessageIdsRef = useRef<Map<string, string>>(new Map());  // Track message IDs per author

  const generateMessageId = () => {
    messageIdRef.current += 1;
    return `msg-${Date.now()}-${messageIdRef.current}`;
  };

  const formatPhase = (phase: PipelinePhase): string => {
    const labels: Partial<Record<PipelinePhase, string>> = {
      idle: 'Ready',
      user_intent: 'Understanding Your Goal',
      file_suggestion: 'Analyzing Files',
      data_cleaning: 'Cleaning Data',
      schema_design: 'Designing Schema',
      targeted_preprocessing: 'Extracting Data',
      construction_plan: 'Planning Construction',
      schema_preprocess_coordinator: 'Processing Schema',
      data_preprocessing: 'Preprocessing Data',
      schema_proposal: 'Designing Graph Schema',
      construction: 'Building Knowledge Graph',
      query: 'Querying Graph',
      complete: 'Complete',
      error: 'Error',
    };
    return labels[phase] || phase;
  };

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      console.log('Received message:', message.type, message);

      switch (message.type) {
        case 'connected':
          console.log('WebSocket connected, session:', message.content);
          globalSessionId = message.content || null;
          if (message.content) {
            setSessionId(message.content);
            // Persist session ID to localStorage
            localStorage.setItem(SESSION_STORAGE_KEY, message.content);
          }
          if (message.phase) {
            setPhase(message.phase);
          }
          if (message.state) {
            setSessionState(message.state as Record<string, unknown>);
          }
          break;

        case 'phase_change':
          if (message.phase) {
            setPhase(message.phase);
            setLoading(true);

            addMessage({
              id: generateMessageId(),
              role: 'system',
              content: `Phase: ${formatPhase(message.phase)}`,
              timestamp: new Date(),
              phase: message.phase,
            });
          }
          break;

        case 'agent_event':
          if (message.content) {
            // Check if this is a tool progress message (author="Tool")
            const isToolProgress = message.author === 'Tool';
            // Check if this is a critic agent message
            const isCritic = message.author?.includes('critic');
            const authorKey = message.author || 'unknown';

            if (isToolProgress) {
              // Tool progress messages: replace the existing tool progress message
              if (toolProgressIdRef.current) {
                updateMessage(toolProgressIdRef.current, message.content);
              } else {
                const id = generateMessageId();
                toolProgressIdRef.current = id;
                addMessage({
                  id,
                  role: 'agent',
                  content: message.content,
                  timestamp: new Date(),
                  agentName: message.author,
                  phase: message.phase,
                  isStreaming: true,
                  isToolProgress: true,
                });
              }
            } else if (isCritic) {
              // Critic messages: always create new message (don't consolidate)
              // This makes intermediate validation feedback visible
              addMessage({
                id: generateMessageId(),
                role: 'agent',
                content: message.content,
                timestamp: new Date(),
                agentName: message.author,
                phase: message.phase,
                isStreaming: false,
                isCriticFeedback: true,
              });
              // Clear the current message ref for this author to avoid mixing
              authorMessageIdsRef.current.delete(authorKey);
            } else if (message.is_final) {
              // Clear tool progress when we get a final message
              toolProgressIdRef.current = null;

              // Use author-based tracking
              const existingId = authorMessageIdsRef.current.get(authorKey);
              if (existingId) {
                updateMessage(existingId, message.content);
                authorMessageIdsRef.current.delete(authorKey);
              } else {
                addMessage({
                  id: generateMessageId(),
                  role: 'agent',
                  content: message.content,
                  timestamp: new Date(),
                  agentName: message.author,
                  phase: message.phase,
                  isStreaming: false,
                });
              }
              setLoading(false);
              // Clear currentMessageIdRef for backward compatibility
              currentMessageIdRef.current = null;
            } else {
              // Don't clear toolProgressIdRef here - let tool progress continue updating in place
              // It will be cleared on final messages (is_final=true) or phase completion

              // Use author-based tracking for streaming messages
              const existingId = authorMessageIdsRef.current.get(authorKey);
              if (existingId) {
                updateMessage(existingId, message.content);
              } else {
                const id = generateMessageId();
                authorMessageIdsRef.current.set(authorKey, id);
                addMessage({
                  id,
                  role: 'agent',
                  content: message.content,
                  timestamp: new Date(),
                  agentName: message.author,
                  phase: message.phase,
                  isStreaming: true,
                });
              }
              // Keep currentMessageIdRef updated for backward compatibility
              currentMessageIdRef.current = authorMessageIdsRef.current.get(authorKey) || null;
            }
          }
          break;

        case 'agent_response':
          if (message.content) {
            addMessage({
              id: generateMessageId(),
              role: 'agent',
              content: message.content,
              timestamp: new Date(),
              agentName: message.author,
              phase: message.phase,
              isStreaming: false,
            });
          }
          setLoading(false);
          break;

        case 'phase_complete':
          if (message.phase) {
            setPhase(message.phase);
          }
          if (message.state) {
            setSessionState(message.state as Record<string, unknown>);
          }
          setLoading(false);
          currentMessageIdRef.current = null;
          toolProgressIdRef.current = null;  // Clear tool progress on phase completion
          authorMessageIdsRef.current.clear();  // Clear author message tracking
          resetExtractionProgress();  // Clear extraction progress on phase completion
          break;

        case 'state_update':
          if (message.state) {
            setSessionState(message.state as Record<string, unknown>);
          }
          // Handle extraction progress updates
          if (message.progress !== undefined && message.progress_total !== undefined) {
            setExtractionProgress(
              message.progress,
              message.progress_current || 0,
              message.progress_total,
              message.progress_item || ''
            );
          }
          break;

        case 'error':
          addMessage({
            id: generateMessageId(),
            role: 'system',
            content: `Error: ${message.error || 'Unknown error'}`,
            timestamp: new Date(),
            phase: 'error',
          });
          setPhase('error');
          setLoading(false);
          currentMessageIdRef.current = null;
          toolProgressIdRef.current = null;  // Clear tool progress on error
          authorMessageIdsRef.current.clear();  // Clear author message tracking on error
          resetExtractionProgress();  // Clear extraction progress on error
          break;
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }, [addMessage, updateMessage, setPhase, setSessionId, setSessionState, setLoading, setExtractionProgress, resetExtractionProgress]);

  const connect = useCallback(() => {
    // If already connected, just update status
    if (globalWs && globalWs.readyState === WebSocket.OPEN) {
      console.log('Already connected, reusing existing connection');
      setConnectionStatus('connected');
      setConnected(true);
      return;
    }

    // If connecting, wait
    if (globalWs && globalWs.readyState === WebSocket.CONNECTING) {
      console.log('Connection in progress, waiting...');
      setConnectionStatus('connecting');
      return;
    }

    const sessionId = globalSessionId || 'new';
    console.log('Creating new WebSocket connection to session:', sessionId);

    setConnectionStatus('connecting');

    // Close any existing dead connection
    if (globalWs) {
      globalWs.onclose = null;
      globalWs.onerror = null;
      globalWs.onmessage = null;
      globalWs.onopen = null;
      if (globalWs.readyState !== WebSocket.CLOSED) {
        globalWs.close();
      }
    }

    const ws = new WebSocket(`ws://localhost:8000/api/chat/${sessionId}`);
    globalWs = ws;

    ws.onopen = () => {
      console.log('WebSocket opened successfully');
      setConnectionStatus('connected');
      setConnected(true);
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      setConnectionStatus('disconnected');
      setConnected(false);

      // Auto-reconnect after 3 seconds if not a clean close
      if (event.code !== 1000 && globalSessionId) {
        reconnectTimeoutRef.current = window.setTimeout(() => {
          console.log('Auto-reconnecting...');
          connect();
        }, 3000);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onmessage = handleMessage;
  }, [setConnected, handleMessage]);

  const disconnect = useCallback(() => {
    console.log('Disconnect called');
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    // Don't actually close the global WebSocket on component unmount
    // This allows it to persist across React StrictMode remounts
  }, []);

  const sendMessage = useCallback((content: string, phase?: PipelinePhase) => {
    console.log('sendMessage called, content:', content);
    console.log('WebSocket exists:', !!globalWs, 'readyState:', globalWs?.readyState);

    if (!globalWs || globalWs.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected, attempting to reconnect...');
      connect();
      // Queue the message to be sent after connection
      setTimeout(() => {
        if (globalWs && globalWs.readyState === WebSocket.OPEN) {
          sendMessage(content, phase);
        }
      }, 1000);
      return;
    }

    console.log('Adding user message to store');
    addMessage({
      id: generateMessageId(),
      role: 'user',
      content,
      timestamp: new Date(),
    });

    const message = {
      type: 'message',
      content,
      phase,
    };
    console.log('Sending to WebSocket:', message);
    globalWs.send(JSON.stringify(message));
    setLoading(true);
  }, [addMessage, setLoading, connect]);

  const approve = useCallback((phase: string) => {
    if (!globalWs || globalWs.readyState !== WebSocket.OPEN) {
      return;
    }

    const message = {
      type: 'approve',
      phase,
    };
    globalWs.send(JSON.stringify(message));
  }, []);

  const cancel = useCallback(() => {
    if (!globalWs || globalWs.readyState !== WebSocket.OPEN) {
      return;
    }

    // Send cancel message to backend
    globalWs.send(JSON.stringify({ type: 'cancel' }));

    // Immediately update UI state
    setLoading(false);
    resetExtractionProgress();

    // Add a system message to indicate cancellation
    addMessage({
      id: `cancel-${Date.now()}`,
      role: 'system',
      content: 'Task cancelled by user',
      timestamp: new Date(),
    });
  }, [setLoading, addMessage, resetExtractionProgress]);

  const startNewChat = useCallback(() => {
    console.log('Starting new chat session');

    // Clear localStorage
    localStorage.removeItem(SESSION_STORAGE_KEY);
    globalSessionId = null;

    // Close existing WebSocket connection
    if (globalWs) {
      globalWs.onclose = null;  // Prevent auto-reconnect
      globalWs.close();
      globalWs = null;
    }

    // Clear chat messages and reset state
    clearMessages();
    setPhase('idle');
    setSessionId('');
    setSessionState({});

    // Reconnect (will create new session)
    connect();
  }, [connect, clearMessages, setPhase, setSessionId, setSessionState]);

  // Connect on mount, update all handlers when they change
  useEffect(() => {
    connect();

    // Update ALL handlers on the existing connection to fix React StrictMode closure issues
    // Without this, the old handlers would reference stale state after remount
    if (globalWs) {
      globalWs.onmessage = handleMessage;
      globalWs.onopen = () => {
        console.log('WebSocket opened successfully (updated handler)');
        setConnectionStatus('connected');
        setConnected(true);
      };
      globalWs.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        setConnectionStatus('disconnected');
        setConnected(false);
      };
      globalWs.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      // If already connected, update status immediately
      if (globalWs.readyState === WebSocket.OPEN) {
        setConnectionStatus('connected');
        setConnected(true);
      } else if (globalWs.readyState === WebSocket.CONNECTING) {
        setConnectionStatus('connecting');
      }
    }

    return () => {
      disconnect();
    };
  }, [connect, disconnect, handleMessage, setConnected]);

  return {
    connectionStatus,
    connect,
    disconnect,
    sendMessage,
    approve,
    cancel,
    startNewChat,
  };
}

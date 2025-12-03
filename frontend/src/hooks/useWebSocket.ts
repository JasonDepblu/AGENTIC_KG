import { useCallback, useEffect, useRef, useState } from 'react';
import { useChatStore } from '../stores/chatStore';
import type { WebSocketMessage, PipelinePhase } from '../types';

// Global WebSocket instance to persist across React StrictMode remounts
let globalWs: WebSocket | null = null;
let globalSessionId: string | null = null;

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
  } = useChatStore();

  const messageIdRef = useRef(0);
  const currentMessageIdRef = useRef<string | null>(null);

  const generateMessageId = () => {
    messageIdRef.current += 1;
    return `msg-${Date.now()}-${messageIdRef.current}`;
  };

  const formatPhase = (phase: PipelinePhase): string => {
    const labels: Record<PipelinePhase, string> = {
      idle: 'Ready',
      user_intent: 'Understanding Your Goal',
      file_suggestion: 'Analyzing Files',
      schema_proposal: 'Designing Graph Schema',
      construction: 'Building Knowledge Graph',
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
            if (message.is_final) {
              if (currentMessageIdRef.current) {
                updateMessage(currentMessageIdRef.current, message.content);
                currentMessageIdRef.current = null;
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
            } else {
              if (!currentMessageIdRef.current) {
                const id = generateMessageId();
                currentMessageIdRef.current = id;
                addMessage({
                  id,
                  role: 'agent',
                  content: message.content,
                  timestamp: new Date(),
                  agentName: message.author,
                  phase: message.phase,
                  isStreaming: true,
                });
              } else {
                updateMessage(currentMessageIdRef.current, message.content);
              }
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
          break;

        case 'state_update':
          if (message.state) {
            setSessionState(message.state as Record<string, unknown>);
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
          break;
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }, [addMessage, updateMessage, setPhase, setSessionId, setSessionState, setLoading]);

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

    globalWs.send(JSON.stringify({ type: 'cancel' }));
  }, []);

  // Connect on mount, update message handler when it changes
  useEffect(() => {
    connect();

    // Update message handler on the existing connection
    if (globalWs) {
      globalWs.onmessage = handleMessage;
    }

    return () => {
      disconnect();
    };
  }, [connect, disconnect, handleMessage]);

  return {
    connectionStatus,
    connect,
    disconnect,
    sendMessage,
    approve,
    cancel,
  };
}

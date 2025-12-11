import { useEffect, useState } from 'react';
import { useChatStore } from '../../stores/chatStore';
import { useWebSocket } from '../../hooks/useWebSocket';
import { MessageList } from './MessageList';
import { InputBar } from './InputBar';
import { PhaseIndicator } from './PhaseIndicator';
import { GraphVisualization } from '../Graph/GraphVisualization';
import { TaskManager } from './TaskManager';
import { Network, Plus, Clock, StopCircle } from 'lucide-react';

export function ChatContainer() {
  const {
    phase,
    sessionId,
    isConnected,
    isLoading,
  } = useChatStore();
  const { connect, disconnect, sendMessage, connectionStatus, startNewChat, cancel } = useWebSocket();
  const [showGraph, setShowGraph] = useState(false);
  const [showTaskManager, setShowTaskManager] = useState(false);

  // Check if current task is running (can be cancelled)
  const isRunning = isLoading || !['idle', 'complete', 'error', 'query'].includes(phase);

  useEffect(() => {
    // Connect on mount
    connect();

    // Disconnect on unmount
    return () => {
      disconnect();
    };
  }, []); // Empty deps - only run on mount/unmount

  const handleSendMessage = (content: string) => {
    if (content.trim()) {
      sendMessage(content);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-border">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold text-text-primary">Agentic KG</h1>
          <PhaseIndicator phase={phase} />
        </div>
        <div className="flex items-center gap-1">
          {/* Cancel button - only show when task is running */}
          {isRunning && (
            <button
              onClick={cancel}
              className="p-2 rounded-full hover:bg-white/10 text-red-400 transition-colors"
              title="Stop"
            >
              <StopCircle className="w-5 h-5" />
            </button>
          )}
          <button
            onClick={startNewChat}
            className="p-2 rounded-full hover:bg-white/10 text-text-secondary hover:text-text-primary transition-colors"
            title="New Chat"
          >
            <Plus className="w-5 h-5" />
          </button>
          <button
            onClick={() => setShowTaskManager(true)}
            className="p-2 rounded-full hover:bg-white/10 text-text-secondary hover:text-text-primary transition-colors"
            title="History"
          >
            <Clock className="w-5 h-5" />
          </button>
          <button
            onClick={() => setShowGraph(true)}
            className="p-2 rounded-full hover:bg-white/10 text-text-secondary hover:text-text-primary transition-colors"
            title="View Graph"
          >
            <Network className="w-5 h-5" />
          </button>
          <div className="ml-2 flex items-center">
            <span
              className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connected'
                  ? 'bg-green-500'
                  : connectionStatus === 'connecting'
                  ? 'bg-yellow-500'
                  : 'bg-red-500'
              }`}
              title={connectionStatus}
            />
          </div>
        </div>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-hidden">
        <MessageList />
      </div>

      {/* Input */}
      <div className="border-t border-border">
        <InputBar
          onSend={handleSendMessage}
          disabled={!isConnected || isLoading}
          isLoading={isLoading}
        />
      </div>

      {/* Graph Visualization Modal */}
      <GraphVisualization isOpen={showGraph} onClose={() => setShowGraph(false)} />

      {/* Task Manager Modal */}
      <TaskManager
        isOpen={showTaskManager}
        onClose={() => setShowTaskManager(false)}
        currentSessionId={sessionId}
        onSwitchSession={(id) => {
          console.log('Switch to session:', id);
          setShowTaskManager(false);
        }}
        onDeleteSession={(id) => {
          console.log('Deleted session:', id);
        }}
        onCancelSession={(id) => {
          console.log('Cancelled session:', id);
        }}
      />
    </div>
  );
}

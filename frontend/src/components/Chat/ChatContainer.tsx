import { useEffect, useState } from 'react';
import { useChatStore } from '../../stores/chatStore';
import { useWebSocket } from '../../hooks/useWebSocket';
import { MessageList } from './MessageList';
import { InputBar } from './InputBar';
import { PhaseIndicator } from './PhaseIndicator';
import { GraphVisualization } from '../Graph/GraphVisualization';
import { Network } from 'lucide-react';

export function ChatContainer() {
  const { phase, isConnected, isLoading } = useChatStore();
  const { connect, disconnect, sendMessage, connectionStatus } = useWebSocket();
  const [showGraph, setShowGraph] = useState(false);

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
        <div className="flex items-center gap-4">
          <button
            onClick={() => setShowGraph(true)}
            className="flex items-center gap-2 px-3 py-1.5 bg-accent/10 hover:bg-accent/20 text-accent rounded-lg transition-colors"
            title="View Knowledge Graph"
          >
            <Network className="w-4 h-4" />
            <span className="text-sm font-medium">View Graph</span>
          </button>
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connected'
                  ? 'bg-green-500'
                  : connectionStatus === 'connecting'
                  ? 'bg-yellow-500'
                  : 'bg-red-500'
              }`}
            />
            <span className="text-sm text-text-secondary capitalize">{connectionStatus}</span>
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
    </div>
  );
}

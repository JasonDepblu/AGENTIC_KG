import { useState, useEffect } from 'react';
import { X, History, Trash2, Square, Clock, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import type { Session, PipelinePhase } from '../../types';

interface TaskManagerProps {
  isOpen: boolean;
  onClose: () => void;
  currentSessionId: string | null;
  onSwitchSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string) => void;
  onCancelSession: (sessionId: string) => void;
}

// Phase display names
const phaseDisplayNames: Record<PipelinePhase, string> = {
  idle: 'Idle',
  user_intent: 'Understanding Intent',
  file_suggestion: 'Analyzing Files',
  data_cleaning: 'Cleaning Data',
  schema_design: 'Designing Schema',
  targeted_preprocessing: 'Extracting Data',
  construction_plan: 'Planning Construction',
  schema_preprocess_coordinator: 'Schema & Preprocessing',
  data_preprocessing: 'Preprocessing Data',
  schema_proposal: 'Proposing Schema',
  construction: 'Building Graph',
  query: 'Query Mode',
  complete: 'Complete',
  error: 'Error',
};

// Phase status icons
function PhaseIcon({ phase }: { phase: PipelinePhase }) {
  switch (phase) {
    case 'idle':
      return <Clock className="w-4 h-4 text-gray-400" />;
    case 'complete':
      return <CheckCircle2 className="w-4 h-4 text-green-400" />;
    case 'error':
      return <AlertCircle className="w-4 h-4 text-red-400" />;
    case 'query':
      return <CheckCircle2 className="w-4 h-4 text-accent" />;
    default:
      return <Loader2 className="w-4 h-4 text-yellow-400 animate-spin" />;
  }
}

export function TaskManager({
  isOpen,
  onClose,
  currentSessionId,
  onSwitchSession,
  onDeleteSession,
  onCancelSession,
}: TaskManagerProps) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch sessions when opened
  useEffect(() => {
    if (isOpen) {
      fetchSessions();
    }
  }, [isOpen]);

  const fetchSessions = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/api/sessions');
      if (!response.ok) {
        throw new Error('Failed to fetch sessions');
      }
      const data = await response.json();
      if (data.success && data.data) {
        setSessions(data.data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load sessions');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (sessionId: string) => {
    if (!confirm('Are you sure you want to delete this session?')) {
      return;
    }
    try {
      const response = await fetch(`http://localhost:8000/api/sessions/${sessionId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        onDeleteSession(sessionId);
        fetchSessions(); // Refresh list
      }
    } catch (err) {
      console.error('Failed to delete session:', err);
    }
  };

  const handleCancel = async (sessionId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/sessions/${sessionId}/cancel`, {
        method: 'POST',
      });
      if (response.ok) {
        onCancelSession(sessionId);
        fetchSessions(); // Refresh list
      }
    } catch (err) {
      console.error('Failed to cancel session:', err);
    }
  };

  const formatTime = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const isRunning = (phase: PipelinePhase) => {
    return !['idle', 'complete', 'error', 'query'].includes(phase);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-bg-primary border border-border rounded-xl shadow-2xl w-full max-w-2xl max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="flex items-center gap-3">
            <History className="w-5 h-5 text-accent" />
            <h2 className="text-lg font-semibold text-text-primary">Task Manager</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-bg-secondary rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-text-secondary" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 text-accent animate-spin" />
              <span className="ml-2 text-text-secondary">Loading sessions...</span>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center py-8 text-red-400">
              <AlertCircle className="w-5 h-5 mr-2" />
              {error}
            </div>
          ) : sessions.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-text-secondary">
              <History className="w-12 h-12 mb-3 opacity-30" />
              <p>No sessions found</p>
            </div>
          ) : (
            <div className="space-y-2">
              {sessions.map((session) => (
                <div
                  key={session.id}
                  className={`
                    flex items-center justify-between p-4 rounded-lg border transition-colors
                    ${session.id === currentSessionId
                      ? 'bg-accent/10 border-accent/30'
                      : 'bg-bg-secondary/50 border-border hover:border-accent/30'
                    }
                  `}
                >
                  {/* Session Info */}
                  <div className="flex items-center gap-4 flex-1 min-w-0">
                    <PhaseIcon phase={session.phase} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-text-primary truncate">
                          {session.id.slice(0, 8)}...
                        </span>
                        {session.id === currentSessionId && (
                          <span className="px-2 py-0.5 text-xs bg-accent/20 text-accent rounded-full">
                            Current
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-3 mt-1 text-sm text-text-secondary">
                        <span>{phaseDisplayNames[session.phase]}</span>
                        <span>•</span>
                        <span>{formatTime(session.updated_at)}</span>
                        <span>•</span>
                        <span>{session.message_count} messages</span>
                      </div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2 ml-4">
                    {/* Cancel button - only show if running */}
                    {isRunning(session.phase) && (
                      <button
                        onClick={() => handleCancel(session.id)}
                        className="p-2 hover:bg-yellow-500/20 text-yellow-400 rounded-lg transition-colors"
                        title="Cancel task"
                      >
                        <Square className="w-4 h-4" />
                      </button>
                    )}

                    {/* Delete button */}
                    <button
                      onClick={() => handleDelete(session.id)}
                      className="p-2 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors"
                      title="Delete session"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-border">
          <div className="flex items-center justify-between">
            <span className="text-sm text-text-secondary">
              {sessions.length} session{sessions.length !== 1 ? 's' : ''}
            </span>
            <button
              onClick={fetchSessions}
              className="px-3 py-1.5 text-sm bg-bg-secondary hover:bg-bg-secondary/80 text-text-primary rounded-lg transition-colors"
            >
              Refresh
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

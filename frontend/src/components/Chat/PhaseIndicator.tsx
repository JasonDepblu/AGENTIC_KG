import { Circle, CheckCircle, Loader2 } from 'lucide-react';
import type { PipelinePhase } from '../../types';

interface PhaseIndicatorProps {
  phase: PipelinePhase;
}

const PHASES: { key: PipelinePhase; label: string }[] = [
  { key: 'user_intent', label: 'Intent' },
  { key: 'file_suggestion', label: 'Files' },
  { key: 'data_preprocessing', label: 'Preprocess' },
  { key: 'schema_proposal', label: 'Schema' },
  { key: 'construction', label: 'Build' },
];

export function PhaseIndicator({ phase }: PhaseIndicatorProps) {
  if (phase === 'idle') {
    return null;
  }

  const currentIndex = PHASES.findIndex((p) => p.key === phase);
  const isComplete = phase === 'complete';
  const isError = phase === 'error';

  return (
    <div className="flex items-center gap-1">
      {PHASES.map((p, index) => {
        const isActive = p.key === phase;
        const isPast = isComplete || (currentIndex > -1 && index < currentIndex);

        return (
          <div key={p.key} className="flex items-center">
            {/* Connector line */}
            {index > 0 && (
              <div
                className={`w-4 h-0.5 mx-0.5 ${
                  isPast ? 'bg-accent' : 'bg-border'
                }`}
              />
            )}

            {/* Phase indicator */}
            <div
              className={`
                flex items-center gap-1 px-2 py-1 rounded-full text-xs
                ${isActive
                  ? 'bg-accent/20 text-accent'
                  : isPast
                  ? 'text-accent'
                  : 'text-text-secondary'
                }
              `}
              title={p.label}
            >
              {isActive ? (
                <Loader2 size={12} className="animate-spin" />
              ) : isPast ? (
                <CheckCircle size={12} />
              ) : (
                <Circle size={12} />
              )}
              <span className="hidden sm:inline">{p.label}</span>
            </div>
          </div>
        );
      })}

      {/* Error indicator */}
      {isError && (
        <div className="flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-red-500/20 text-red-400 ml-2">
          <Circle size={12} />
          <span>Error</span>
        </div>
      )}
    </div>
  );
}

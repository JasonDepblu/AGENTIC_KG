import { Circle, CheckCircle, Loader2 } from 'lucide-react';
import type { PipelinePhase } from '../../types';

interface PhaseIndicatorProps {
  phase: PipelinePhase;
}

// Schema-First Pipeline phases with Data Cleaning (default)
const SCHEMA_FIRST_WITH_CLEANING_PHASES: { key: PipelinePhase; label: string }[] = [
  { key: 'user_intent', label: 'Intent' },
  { key: 'file_suggestion', label: 'Files' },
  { key: 'data_cleaning', label: 'Clean' },
  { key: 'schema_design', label: 'Schema' },
  { key: 'targeted_preprocessing', label: 'Extract' },
  { key: 'construction', label: 'Build' },
  { key: 'query', label: 'Query' },
];

// Schema-First Pipeline phases (without Data Cleaning)
const SCHEMA_FIRST_PHASES: { key: PipelinePhase; label: string }[] = [
  { key: 'user_intent', label: 'Intent' },
  { key: 'file_suggestion', label: 'Files' },
  { key: 'schema_design', label: 'Schema' },
  { key: 'targeted_preprocessing', label: 'Extract' },
  { key: 'construction', label: 'Build' },
  { key: 'query', label: 'Query' },
];

// Legacy Pipeline phases
const LEGACY_PHASES: { key: PipelinePhase; label: string }[] = [
  { key: 'user_intent', label: 'Intent' },
  { key: 'file_suggestion', label: 'Files' },
  { key: 'data_preprocessing', label: 'Preprocess' },
  { key: 'schema_proposal', label: 'Schema' },
  { key: 'construction', label: 'Build' },
  { key: 'query', label: 'Query' },
];

// Determine which pipeline we're using based on current phase
function getPhasesForCurrentPipeline(phase: PipelinePhase): { key: PipelinePhase; label: string }[] {
  // Data Cleaning phase indicates the new pipeline with cleaning
  if (phase === 'data_cleaning') {
    return SCHEMA_FIRST_WITH_CLEANING_PHASES;
  }
  // Super Coordinator phase also uses the cleaning pipeline
  if (phase === 'schema_preprocess_coordinator') {
    return SCHEMA_FIRST_WITH_CLEANING_PHASES;
  }
  // Schema-First phases (without cleaning) - this is the default backend mode
  const schemaFirstPhases = [
    'user_intent',
    'file_suggestion',
    'schema_design',
    'targeted_preprocessing',
    'construction_plan',
    'construction',
    'query',
    'complete'
  ];
  // Legacy phases
  const legacyPhases = ['data_preprocessing', 'schema_proposal'];

  if (schemaFirstPhases.includes(phase)) {
    return SCHEMA_FIRST_PHASES;
  }
  if (legacyPhases.includes(phase)) {
    return LEGACY_PHASES;
  }
  // Default to Schema-First (without Data Cleaning) - matches backend default
  return SCHEMA_FIRST_PHASES;
}

export function PhaseIndicator({ phase }: PhaseIndicatorProps) {
  if (phase === 'idle') {
    return null;
  }

  const PHASES = getPhasesForCurrentPipeline(phase);

  // Map special phases to display phases
  // - construction_plan → construction
  // - schema_preprocess_coordinator → schema_design (it handles both schema + preprocessing)
  let displayPhase: PipelinePhase = phase;
  if (phase === 'construction_plan') {
    displayPhase = 'construction';
  } else if (phase === 'schema_preprocess_coordinator') {
    // Show as schema_design since coordinator handles schema + preprocessing internally
    displayPhase = 'schema_design';
  }

  const currentIndex = PHASES.findIndex((p) => p.key === displayPhase);
  const isComplete = phase === 'complete';
  const isError = phase === 'error';

  return (
    <div className="flex items-center gap-1">
      {PHASES.map((p, index) => {
        const isActive = p.key === displayPhase;
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

interface ExtractionProgressProps {
  progress: number;
  current: number;
  total: number;
  item: string;
}

export function ExtractionProgress({ progress, current, total, item }: ExtractionProgressProps) {
  // Don't render if no progress data
  if (progress === 0 && total === 0) return null;

  // Ensure current doesn't exceed total for display
  const displayCurrent = Math.min(current, total);
  const displayProgress = Math.min(progress, 100);

  return (
    <div className="pt-1">
      {/* Apple-style thin progress bar */}
      <div className="h-1 bg-border/30 rounded-full overflow-hidden">
        <div
          className="h-full bg-accent transition-all duration-500 ease-out"
          style={{ width: `${displayProgress}%` }}
        />
      </div>
      {/* Progress text - compact */}
      <div className="mt-1 flex justify-between items-center text-[11px] text-text-secondary">
        <span className="truncate max-w-[280px] opacity-80">{item || 'Processing...'}</span>
        <span className="font-medium tabular-nums opacity-80">{displayCurrent}/{total}</span>
      </div>
    </div>
  );
}

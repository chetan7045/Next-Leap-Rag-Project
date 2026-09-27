type DisclaimerLineProps = {
  /** Shown under the composer on every screen size. */
  compact?: boolean;
};

/**
 * Persistent, understated product disclaimer. Replaces the former long
 * per-message block.
 */
export function DisclaimerLine({ compact = false }: DisclaimerLineProps) {
  return (
    <div className="mt-2.5 space-y-1 text-center">
      <p className="text-[12px] leading-relaxed text-muted-500">
        Facts-only. No investment advice. Please verify important information
        with the official source.
      </p>
      {compact ? null : (
        <p className="text-[12px] leading-relaxed text-muted-400">
          Please don&apos;t enter personal financial information.
        </p>
      )}
    </div>
  );
}

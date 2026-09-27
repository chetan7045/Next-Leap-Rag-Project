"use client";

import { ArrowRightIcon } from "./icons";

type PromptSuggestionProps = {
  label: string;
  onSelect: () => void;
  disabled?: boolean;
};

/**
 * Interactive example question. Sends the real question through the existing
 * chat endpoint — no canned or fabricated answers.
 */
export function PromptSuggestion({
  label,
  onSelect,
  disabled = false,
}: PromptSuggestionProps) {
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        disabled={disabled}
        className={[
          "group relative flex w-full items-center gap-3 rounded-chip border border-line-200 bg-canvas",
          "px-4 py-3.5 text-left text-[14px] leading-snug text-ink-800",
          "shadow-xs transition-all duration-200 ease-niva",
          "hover:-translate-y-0.5 hover:border-niva-300 hover:text-niva-700 hover:shadow-card",
          "focus-visible:border-niva-400 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-niva-500/10",
          "disabled:pointer-events-none disabled:opacity-55",
        ].join(" ")}
      >
        <span className="min-w-0 flex-1">{label}</span>
        <ArrowRightIcon
          size={15}
          className="hidden shrink-0 -translate-x-1 text-niva-500 opacity-0 transition-all duration-200 group-hover:translate-x-0 group-hover:opacity-100 sm:block"
        />
      </button>
    </li>
  );
}

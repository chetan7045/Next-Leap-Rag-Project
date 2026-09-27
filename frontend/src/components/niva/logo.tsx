"use client";

import { useId } from "react";

/**
 * Niva brand mark: a minimal flowing "N" drawn as one continuous intelligent
 * line, with a single accent point. Works beside the wordmark, inside the
 * circular avatar, and at every size from 20px to 64px.
 */

const MARK_PATH =
  "M5.6 17.6V7.35c0-.56.7-.81 1.07-.39l10.5 10.5c.37.42 1.07.17 1.07-.39V6.4";

type MarkProps = {
  size?: number;
  className?: string;
  /** Slow idle breath. Disabled for repeated instances inside a transcript. */
  animated?: boolean;
};

export function NivaMark({ size = 24, className = "", animated = false }: MarkProps) {
  const gradientId = useId();

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      focusable="false"
      className={className}
    >
      {animated ? (
        <defs>
          <linearGradient
            id={gradientId}
            x1="4"
            y1="20"
            x2="20"
            y2="4"
            gradientUnits="userSpaceOnUse"
          >
            <stop stopColor="var(--color-niva-500)" />
            <stop offset="1" stopColor="var(--color-niva-700)" />
          </linearGradient>
        </defs>
      ) : null}
      <path
        d={MARK_PATH}
        stroke={animated ? `url(#${gradientId})` : "var(--color-niva-600)"}
        strokeWidth="2.05"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={animated ? "animate-niva-mark" : undefined}
      />
      <circle cx="17.9" cy="4.6" r="1.45" fill="var(--color-niva-400)" />
    </svg>
  );
}

/** Circular avatar used in the transcript and header lockup. */
export function NivaAvatar({
  size = 30,
  className = "",
  animated = false,
}: MarkProps) {
  const gradientId = useId();

  return (
    <span
      className={`inline-flex shrink-0 items-center justify-center rounded-full shadow-[0_2px_8px_rgb(37_99_235/0.20)] ${className}`}
      style={{
        width: size,
        height: size,
        backgroundImage: `linear-gradient(145deg, var(--color-niva-400), var(--color-niva-600) 55%, var(--color-niva-700))`,
      }}
    >
      <svg
        width={Math.round(size * 0.62)}
        height={Math.round(size * 0.62)}
        viewBox="0 0 24 24"
        fill="none"
        aria-hidden="true"
        focusable="false"
      >
        {animated ? (
          <defs>
            <linearGradient
              id={gradientId}
              x1="0"
              y1="24"
              x2="24"
              y2="0"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="rgb(255 255 255 / 0.82)" />
              <stop offset="1" stopColor="#ffffff" />
            </linearGradient>
          </defs>
        ) : null}
        <path
          d={MARK_PATH}
          stroke={
            animated ? `url(#${gradientId})` : "rgb(255 255 255 / 0.95)"
          }
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className={animated ? "animate-niva-mark" : undefined}
        />
        <circle
          cx="17.9"
          cy="4.6"
          r="1.45"
          fill="rgb(255 255 255 / 0.7)"
        />
      </svg>
    </span>
  );
}

/** Header lockup: mark + wordmark. */
export function NivaWordmark({ size = 26 }: { size?: number }) {
  return (
    <span className="inline-flex items-center gap-2">
      <NivaMark size={size} animated />
      <span className="text-[1.0625rem] font-semibold tracking-[-0.018em] text-ink-900">
        Niva
      </span>
    </span>
  );
}

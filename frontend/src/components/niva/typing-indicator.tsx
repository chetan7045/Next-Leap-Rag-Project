"use client";

import { NivaAvatar } from "./logo";

/**
 * Conversational loading state. Deliberately says nothing about embeddings,
 * retrieval, top-k or model providers.
 */
export function TypingIndicator() {
  return (
    <li className="animate-niva-rise-soft flex gap-3">
      <NivaAvatar size={30} className="mt-0.5" />
      <div className="flex items-center gap-2.5 rounded-bubble rounded-tl-md border border-line-100 bg-canvas-soft px-4 py-3">
        <span className="text-[14.5px] leading-none text-muted-500">
          Niva is thinking
        </span>
        <span aria-hidden="true" className="flex items-center gap-1">
          {[0, 1, 2].map((index) => (
            <span
              key={index}
              className="size-1.5 animate-niva-dot rounded-full bg-niva-400"
              style={{ animationDelay: `${index * 0.16}s` }}
            />
          ))}
        </span>
      </div>
    </li>
  );
}

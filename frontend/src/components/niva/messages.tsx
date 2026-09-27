"use client";

import { formatSourceDate, optionLabel, toParagraphs } from "@/lib/format";
import type { ChatResponse } from "@/lib/types";
import { NivaAvatar } from "./logo";
import { SourceCitation } from "./source-citation";
import { PromptSuggestion } from "./prompt-suggestion";
import { RetryIcon } from "./icons";

/** Compact, right-aligned question. No timestamps, no metadata. */
export function UserMessage({ text }: { text: string }) {
  return (
    <li className="animate-niva-rise flex justify-end">
      <p className="max-w-[86%] rounded-bubble rounded-br-md bg-niva-100 px-4 py-2.5 text-[15px] leading-relaxed text-ink-900 sm:max-w-[80%]">
        {text}
      </p>
    </li>
  );
}

type AssistantMessageProps = {
  response: ChatResponse;
  onPick: (question: string) => void;
};

/**
 * The primary reading surface. Not a card: an avatar, a strong answer, then a
 * quiet source block.
 *
 * Refusals (ADVICE_REQUEST, PERFORMANCE_PROMISE, PII_DETECTED, OUT_OF_SCOPE) and
 * "no information found" (NO_CONTEXT / LOW_CONFIDENCE) render with the same
 * calm styling — they are normal conversational answers, not errors.
 */
export function AssistantMessage({ response, onPick }: AssistantMessageProps) {
  const paragraphs = toParagraphs(response.answer);
  const sources = response.sources ?? [];
  const options = response.clarification_options ?? [];
  const updated = formatSourceDate(
    response.last_updated ?? sources[0]?.last_updated,
  );

  return (
    <li className="animate-niva-rise-soft flex gap-3">
      <NivaAvatar size={30} className="mt-0.5" />

      <div className="min-w-0 flex-1 pt-0.5">
        <div className="space-y-3">
          {paragraphs.map((paragraph, index) => (
            <p
              key={index}
              className={
                index === 0
                  ? "text-[16.5px] leading-[1.62] tracking-[-0.008em] text-ink-900 sm:text-[17px]"
                  : "text-[15px] leading-[1.68] text-muted-600"
              }
            >
              {paragraph}
            </p>
          ))}
        </div>

        {options.length > 0 ? (
          <ul className="mt-3.5 flex flex-wrap gap-2">
            {options.map((option, index) => {
              const label = optionLabel(option);
              return (
                <li key={index} className="max-w-full">
                  <PromptSuggestion label={label} onSelect={() => onPick(label)} />
                </li>
              );
            })}
          </ul>
        ) : null}

        {sources.length > 0 ? (
          <div
            className="mt-3.5 animate-niva-fade space-y-2"
            style={{ animationDelay: "120ms" }}
          >
            {sources.map((source, index) => (
              <SourceCitation key={source.url || index} source={source} />
            ))}
          </div>
        ) : null}

        {updated ? (
          <p
            className="mt-2.5 animate-niva-fade text-[12.5px] leading-relaxed text-muted-500"
            style={{ animationDelay: "180ms" }}
          >
            Last updated from sources: {updated}
          </p>
        ) : null}
      </div>
    </li>
  );
}

type ErrorMessageProps = {
  detail: string;
  onRetry?: () => void;
};

/**
 * Genuine technical failure only. Never shows status codes, request ids,
 * stack traces or provider/infra names.
 */
export function ErrorMessage({ detail, onRetry }: ErrorMessageProps) {
  return (
    <li className="animate-niva-rise-soft flex gap-3">
      <NivaAvatar size={30} className="mt-0.5" />
      <div className="min-w-0 flex-1 pt-0.5">
        <p className="text-[15.5px] leading-relaxed text-ink-900">
          Something went wrong.
        </p>
        <p className="mt-1 text-[14px] leading-relaxed text-muted-500">
          {detail}
        </p>
        {onRetry ? (
          <button
            type="button"
            onClick={onRetry}
            className={[
              "mt-3 inline-flex items-center gap-1.5 rounded-full border border-line-200 bg-canvas",
              "px-3 py-1.5 text-[13px] font-medium text-niva-700",
              "transition-all duration-200 ease-niva",
              "hover:border-niva-300 hover:bg-niva-50 active:scale-[0.98]",
            ].join(" ")}
          >
            <RetryIcon size={14} />
            Try again
          </button>
        ) : null}
      </div>
    </li>
  );
}

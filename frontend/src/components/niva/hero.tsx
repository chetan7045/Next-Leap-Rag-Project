"use client";

import { NivaAvatar } from "./logo";
import { PromptSuggestion } from "./prompt-suggestion";

export const SUGGESTIONS = [
  "What is the expense ratio of HDFC Large Cap Fund?",
  "What is the exit load for HDFC ELSS Tax Saver Fund?",
  "What is the benchmark for HDFC Flexi Cap Fund?",
  "What is the lock-in period for ELSS?",
];

type WelcomeHeroProps = {
  leaving: boolean;
  onPick: (question: string) => void;
  disabled?: boolean;
};

/**
 * Empty state. Vertically centred, brand-forward, zero technical content.
 * Cross-fades out when the first question is sent.
 */
export function WelcomeHero({
  leaving,
  onPick,
  disabled = false,
}: WelcomeHeroProps) {
  return (
    <section
      className={[
        "relative flex flex-1 flex-col justify-center",
        leaving ? "animate-niva-hero-out pointer-events-none" : "animate-niva-hero-in",
      ].join(" ")}
    >
      {/* Barely-there blue glow behind the hero. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute left-1/2 top-0 h-[26rem] w-[min(52rem,135%)] -translate-x-1/2 animate-niva-halo"
        style={{
          background:
            "radial-gradient(ellipse 50% 50% at 50% 8%, rgb(37 99 235 / 0.07), transparent 70%)",
        }}
      />

      <div className="relative mx-auto flex w-full max-w-[40rem] flex-col items-center px-1 py-10 text-center sm:py-14">
        <NivaAvatar size={46} animated className="animate-niva-rise" />

        <h1 className="mt-5 animate-niva-rise text-[30px] font-semibold leading-tight tracking-[-0.032em] text-ink-900 sm:text-[38px]">
          Ask Niva.
        </h1>

        <p
          className="mt-3 max-w-[27rem] animate-niva-rise text-[15px] leading-relaxed text-muted-600 sm:text-[16px]"
          style={{ animationDelay: "40ms" }}
        >
          Clear, factual answers about HDFC mutual funds — with the source
          included.
        </p>

        <ul
          className="mt-8 grid w-full animate-niva-rise gap-2.5 sm:mt-10 sm:grid-cols-2"
          style={{ animationDelay: "80ms" }}
        >
          {SUGGESTIONS.map((question) => (
            <PromptSuggestion
              key={question}
              label={question}
              onSelect={() => onPick(question)}
              disabled={disabled}
            />
          ))}
        </ul>
      </div>
    </section>
  );
}

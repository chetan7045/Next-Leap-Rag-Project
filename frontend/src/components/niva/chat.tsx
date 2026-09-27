"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, sendChat } from "@/lib/api";
import type { ChatResponse } from "@/lib/types";
import { NivaHeader } from "./header";
import { WelcomeHero } from "./hero";
import { AssistantMessage, ErrorMessage, UserMessage } from "./messages";
import { TypingIndicator } from "./typing-indicator";
import { ChatComposer } from "./composer";
import { DisclaimerLine } from "./disclaimer";
import { DownIcon } from "./icons";

type Turn =
  | { id: string; role: "user"; text: string }
  | { id: string; role: "assistant"; response: ChatResponse }
  | { id: string; role: "assistant"; failure: { detail: string; retry: string } };

type Phase = "hero" | "leaving" | "chat";

/** Customer-safe copy. No status codes, request ids or provider names. */
function friendlyDetail(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 0) {
      return "Niva can't reach the service right now. Please check your connection and try again.";
    }
    if (error.status === 422) {
      return "That message looked empty. Ask a question about an HDFC mutual fund and try again.";
    }
    if (error.status >= 500) {
      return "Niva is having trouble right now. Please try again in a moment.";
    }
  }
  return "Please try again in a moment.";
}

export function Chat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [phase, setPhase] = useState<Phase>("hero");
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [failedLast, setFailedLast] = useState(false);
  const [showJump, setShowJump] = useState(false);

  const scrollerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const busyRef = useRef(false);
  const pinnedRef = useRef(true);
  const sequence = useRef(0);

  const nextId = () => `turn-${++sequence.current}`;

  const scrollToLatest = useCallback((behavior: ScrollBehavior = "smooth") => {
    const node = scrollerRef.current;
    if (!node) return;
    node.scrollTo({ top: node.scrollHeight, behavior });
  }, []);

  const ask = useCallback(
    async (raw: string) => {
      const question = raw.trim();
      // Ref guard prevents double submission before state has settled.
      if (!question || busyRef.current) return;

      busyRef.current = true;
      setBusy(true);
      setFailedLast(false);
      setShowJump(false);
      pinnedRef.current = true;

      setTurns((prev) => [...prev, { id: nextId(), role: "user", text: question }]);
      setDraft("");

      setPhase((current) => {
        if (current !== "hero") return current;
        window.setTimeout(() => setPhase("chat"), 200);
        return "leaving";
      });

      try {
        const response = await sendChat({ message: question });
        setTurns((prev) => [...prev, { id: nextId(), role: "assistant", response }]);
      } catch (error) {
        // Developer diagnostics only — never rendered.
        console.warn("[niva] chat request failed", error);
        setTurns((prev) => [
          ...prev,
          {
            id: nextId(),
            role: "assistant",
            failure: { detail: friendlyDetail(error), retry: question },
          },
        ]);
        setFailedLast(true);
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [],
  );

  const retry = useCallback(
    (id: string) => {
      const turn = turns.find((item) => item.id === id);
      if (!turn || turn.role !== "assistant" || !("failure" in turn)) return;
      setTurns((prev) => prev.filter((item) => item.id !== id));
      void ask(turn.failure.retry);
    },
    [turns, ask],
  );

  // Follow the conversation only while the user is already at the bottom.
  useEffect(() => {
    if (!pinnedRef.current) return;
    const frame = window.requestAnimationFrame(() => scrollToLatest("smooth"));
    return () => window.cancelAnimationFrame(frame);
  }, [turns, busy, phase, scrollToLatest]);

  const handleScroll = useCallback(() => {
    const node = scrollerRef.current;
    if (!node) return;
    const distance = node.scrollHeight - node.scrollTop - node.clientHeight;
    const atBottom = distance < 120;
    pinnedRef.current = atBottom;
    setShowJump(!atBottom && node.scrollHeight > node.clientHeight + 240);
  }, []);

  const awaitingReply =
    busy && turns.length > 0 && turns[turns.length - 1].role === "user";

  return (
    <div className="flex min-h-dvh flex-col bg-canvas">
      <NivaHeader separated={turns.length > 0} />

      <div
        ref={scrollerRef}
        onScroll={handleScroll}
        className="niva-scroll relative min-h-0 flex-1 overflow-y-auto"
      >
        <div className="mx-auto flex min-h-full w-full max-w-[768px] flex-col px-5 sm:px-6">
          {phase !== "chat" ? (
            <WelcomeHero
              leaving={phase === "leaving"}
              onPick={(question) => void ask(question)}
              disabled={busy}
            />
          ) : null}

          <ol
            role="log"
            aria-live="polite"
            aria-label="Conversation"
            aria-busy={busy}
            className="flex flex-col gap-6 pb-4 pt-1"
          >
            {turns.map((turn) => {
              if (turn.role === "user") {
                return <UserMessage key={turn.id} text={turn.text} />;
              }
              if ("response" in turn) {
                return (
                  <AssistantMessage
                    key={turn.id}
                    response={turn.response}
                    onPick={(question) => void ask(question)}
                  />
                );
              }
              return (
                <ErrorMessage
                  key={turn.id}
                  detail={turn.failure.detail}
                  onRetry={() => retry(turn.id)}
                />
              );
            })}

            {awaitingReply ? <TypingIndicator /> : null}
          </ol>
        </div>
      </div>

      {/* Composer dock: fixed to the bottom, so the last message is never hidden. */}
      <div className="relative shrink-0">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 -top-8 h-8 bg-gradient-to-t from-canvas to-transparent"
        />

        {showJump ? (
          <div className="pointer-events-none absolute inset-x-0 -top-11 flex justify-center">
            <button
              type="button"
              onClick={() => {
                pinnedRef.current = true;
                setShowJump(false);
                scrollToLatest("smooth");
              }}
              className={[
                "pointer-events-auto inline-flex items-center gap-1.5 rounded-full",
                "border border-line-200 bg-canvas px-3 py-1.5 text-[12.5px] font-medium text-muted-600",
                "shadow-card transition-all duration-200 ease-niva",
                "hover:border-niva-300 hover:text-niva-700 active:scale-[0.98]",
              ].join(" ")}
            >
              Jump to latest
              <DownIcon size={14} />
            </button>
          </div>
        ) : null}

        <div className="mx-auto w-full max-w-[768px] px-5 pt-1 pb-[max(0.875rem,env(safe-area-inset-bottom))] sm:px-6">
          <ChatComposer
            value={draft}
            onValueChange={setDraft}
            onSubmit={() => void ask(draft)}
            busy={busy}
            error={failedLast}
            textareaRef={textareaRef}
          />
          <DisclaimerLine />
        </div>
      </div>
    </div>
  );
}

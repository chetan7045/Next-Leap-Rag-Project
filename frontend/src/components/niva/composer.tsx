"use client";

import { useEffect, type RefObject } from "react";
import { SendIcon } from "./icons";

const MAX_TEXTAREA_HEIGHT = 152;

type ChatComposerProps = {
  value: string;
  onValueChange: (value: string) => void;
  onSubmit: () => void;
  busy: boolean;
  error?: boolean;
  textareaRef: RefObject<HTMLTextAreaElement | null>;
};

/**
 * The composer is the strongest element on screen: white, pill-ish, with a
 * soft shadow and a blue circular send button.
 *
 * Enter sends, Shift+Enter adds a newline. Duplicate submissions are blocked
 * while a request is in flight.
 */
export function ChatComposer({
  value,
  onValueChange,
  onSubmit,
  busy,
  error = false,
  textareaRef,
}: ChatComposerProps) {
  // Grow with the content up to a cap, then scroll internally.
  useEffect(() => {
    const node = textareaRef.current;
    if (!node) return;
    node.style.height = "0px";
    node.style.height = `${Math.min(node.scrollHeight, MAX_TEXTAREA_HEIGHT)}px`;
  }, [value, textareaRef]);

  const canSend = value.trim().length > 0 && !busy;

  return (
    <form
      noValidate
      onSubmit={(event) => {
        event.preventDefault();
        if (canSend) onSubmit();
      }}
    >
      <div
        className={[
          "flex items-end gap-2 rounded-composer border bg-canvas py-2 pl-4 pr-2",
          "transition-all duration-200 ease-niva",
          "shadow-composer",
          error
            ? "border-bad-200"
            : "border-line-200 focus-within:border-niva-300",
          "focus-within:shadow-[0_8px_34px_rgb(11_18_32/0.08),0_0_0_4px_rgb(37_99_235/0.08)]",
        ].join(" ")}
      >
        <label htmlFor="niva-composer" className="sr-only">
          Ask Niva about an HDFC mutual fund
        </label>
        <textarea
          id="niva-composer"
          ref={textareaRef}
          rows={1}
          value={value}
          placeholder="Ask Niva about an HDFC mutual fund..."
          onChange={(event) => onValueChange(event.target.value)}
          onKeyDown={(event) => {
            // Enter sends; Shift+Enter inserts a newline.
            // `isComposing` stops IME Enter from firing a send.
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              if (canSend) onSubmit();
            }
          }}
          aria-busy={busy}
          className={[
            "max-h-38 min-h-[1.5rem] flex-1 resize-none bg-transparent py-1.5",
            "text-[15px] leading-6 text-ink-900 outline-none",
            "placeholder:text-muted-400",
          ].join(" ")}
        />

        <button
          type="submit"
          disabled={!canSend}
          aria-label="Send"
          className={[
            "inline-flex size-11 shrink-0 items-center justify-center rounded-full sm:size-9",
            "transition-all duration-200 ease-niva",
            "bg-gradient-to-b from-niva-500 to-niva-600 text-white",
            "shadow-[0_2px_8px_rgb(37_99_235/0.26)]",
            "hover:from-niva-500 hover:to-niva-700 hover:shadow-[0_4px_12px_rgb(37_99_235/0.32)]",
            "active:scale-95",
            "disabled:bg-line-200 disabled:text-muted-300 disabled:shadow-none disabled:active:scale-100",
          ].join(" ")}
        >
          <SendIcon size={18} className={busy ? "opacity-45" : undefined} />
        </button>
      </div>
    </form>
  );
}

"use client";

import { useEffect, useId, useRef } from "react";
import { CloseIcon } from "./icons";

type AboutDialogProps = {
  open: boolean;
  onClose: () => void;
};

const POINTS = [
  {
    title: "Facts, not forecasts",
    body: "Niva shares what is published about HDFC Mutual Fund schemes. It does not predict returns, rank funds, or recommend where to invest.",
  },
  {
    title: "Every answer is sourced",
    body: "Factual answers link to the public source they were drawn from, with the date that source was last updated.",
  },
  {
    title: "Please keep it anonymous",
    body: "Do not enter PAN, Aadhaar, bank, folio or other personal financial information.",
  },
];

/**
 * Secondary "About Niva" surface. Uses a native <dialog> so focus trapping,
 * Escape-to-close and inert background come from the platform.
 */
export function AboutDialog({ open, onClose }: AboutDialogProps) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (open && !node.open) node.showModal();
    if (!open && node.open) node.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      aria-labelledby={titleId}
      onClose={onClose}
      onClick={(event) => {
        // Click on the backdrop (outside the panel) dismisses.
        if (event.target === ref.current) onClose();
      }}
      className="m-auto w-[min(26rem,calc(100vw-2rem))] rounded-2xl border border-line-200 bg-canvas p-0 text-left text-ink-900 shadow-float backdrop:bg-ink-900/20 backdrop:backdrop-blur-[3px]"
    >
      <div className="animate-niva-rise-soft p-6 sm:p-7">
        <div className="flex items-start justify-between gap-4">
          <h2
            id={titleId}
            className="text-[17px] font-semibold tracking-[-0.015em]"
          >
            About Niva
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="-mr-1 -mt-1 rounded-full p-1.5 text-muted-400 transition-colors duration-200 hover:bg-niva-50 hover:text-niva-700"
          >
            <CloseIcon size={18} />
          </button>
        </div>

        <p className="mt-3 text-[14px] leading-relaxed text-muted-600">
          Niva provides factual information about selected HDFC Mutual Fund
          schemes using indexed public sources. It does not provide investment
          recommendations or personalised financial advice. Please verify
          important information using the underlying source.
        </p>

        <ul className="mt-5 space-y-4">
          {POINTS.map((point) => (
            <li key={point.title} className="flex gap-3">
              <span
                aria-hidden="true"
                className="mt-[7px] size-1.5 shrink-0 rounded-full bg-niva-400"
              />
              <div>
                <p className="text-[13.5px] font-medium text-ink-800">
                  {point.title}
                </p>
                <p className="mt-0.5 text-[13px] leading-relaxed text-muted-500">
                  {point.body}
                </p>
              </div>
            </li>
          ))}
        </ul>

        <p className="mt-6 border-t border-line-100 pt-4 text-[12px] leading-relaxed text-muted-400">
          Facts-only · No investment advice · Please verify important
          information with the official source.
        </p>
      </div>
    </dialog>
  );
}

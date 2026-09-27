"use client";

import { useState } from "react";
import { LiveStatus } from "./live-status";
import { NivaWordmark } from "./logo";
import { AboutDialog } from "./about";
import { InfoIcon } from "./icons";

type NivaHeaderProps = {
  /** A hairline separator appears once a conversation begins. */
  separated: boolean;
};

/**
 * Minimal header: brand lockup on the left, the single status chip on the right.
 * No settings, no model, no index, no scheme menu.
 */
export function NivaHeader({ separated }: NivaHeaderProps) {
  const [aboutOpen, setAboutOpen] = useState(false);

  return (
    <>
      <header
        className={[
          "shrink-0 border-b bg-canvas/85 backdrop-blur-sm",
          "transition-colors duration-300",
          separated ? "border-line-100" : "border-transparent",
        ].join(" ")}
      >
        <div className="mx-auto flex w-full max-w-[768px] items-start justify-between gap-4 px-5 py-4 sm:px-6 sm:py-5">
          <div className="min-w-0">
            <div className="flex items-center gap-1">
              <NivaWordmark />
              <button
                type="button"
                onClick={() => setAboutOpen(true)}
                aria-label="About Niva"
                aria-haspopup="dialog"
                className="ml-0.5 rounded-full p-1.5 text-muted-400 transition-colors duration-200 hover:bg-niva-50 hover:text-niva-600"
              >
                <InfoIcon size={16} />
              </button>
            </div>
            <p className="mt-1 truncate text-[13px] leading-snug text-muted-500">
              Your factual guide to HDFC mutual funds.
            </p>
          </div>

          <div className="shrink-0 pt-0.5">
            <LiveStatus />
          </div>
        </div>
      </header>

      <AboutDialog open={aboutOpen} onClose={() => setAboutOpen(false)} />
    </>
  );
}

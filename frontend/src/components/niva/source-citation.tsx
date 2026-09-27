"use client";

import { authorityLabel, formatSourceDate, sourceDisplayName } from "@/lib/format";
import type { SourceCitation as SourceCitationData } from "@/lib/types";
import { DocumentIcon, ExternalIcon } from "./icons";

type SourceCitationProps = {
  source: SourceCitationData;
};

/**
 * Customer-facing citation.
 *
 * Shows only: authority label, source name, "View source" and the last-updated
 * date. Relevance scores, chunk ids, document types and internal enum values
 * are never rendered.
 */
export function SourceCitation({ source }: SourceCitationProps) {
  const name = sourceDisplayName(source);
  const authority = authorityLabel(source.source_type, source.source_type_label);
  const updated = formatSourceDate(source.last_updated);
  const href = typeof source.url === "string" ? source.url.trim() : "";

  // States: missing link degrades gracefully, never a broken card or fake URL.
  if (!href) {
    return (
      <div className="flex items-start gap-3 rounded-source border border-line-100 bg-canvas-soft px-3.5 py-3">
        <span
          aria-hidden="true"
          className="mt-0.5 inline-flex size-7 shrink-0 items-center justify-center rounded-lg bg-niva-50 text-niva-500"
        >
          <DocumentIcon size={15} />
        </span>
        <div className="min-w-0">
          <p className="text-[11.5px] font-semibold uppercase tracking-[0.08em] text-muted-500">
            Source
          </p>
          <p className="mt-1 text-[13.5px] font-medium text-ink-800">{name}</p>
          <p className="mt-0.5 text-[12.5px] text-muted-500">
            Source unavailable
          </p>
        </div>
      </div>
    );
  }

  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={[
        "group flex flex-wrap items-start gap-x-3 gap-y-2 rounded-source border px-3.5 py-3",
        "border-line-100 bg-canvas-soft",
        "transition-all duration-200 ease-niva",
        "hover:border-niva-200 hover:bg-niva-50/60",
        "focus-visible:border-niva-400 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-niva-500/10",
      ].join(" ")}
    >
      <span
        aria-hidden="true"
        className="mt-0.5 inline-flex size-7 shrink-0 items-center justify-center rounded-lg bg-niva-50 text-niva-600 transition-colors duration-200 group-hover:bg-niva-100"
      >
        <DocumentIcon size={15} />
      </span>

      <div className="min-w-0 flex-1 basis-40">
        <p className="text-[11.5px] font-semibold uppercase tracking-[0.08em] text-muted-500">
          Source
        </p>
        <p className="mt-1 truncate text-[13.5px] font-medium text-ink-800 transition-colors duration-200 group-hover:text-niva-700">
          {name}
        </p>
        {authority ? (
          <p className="mt-0.5 text-[12.5px] text-muted-500">{authority}</p>
        ) : null}
        {updated ? (
          <p className="mt-0.5 text-[12.5px] text-muted-500 sm:hidden">
            Updated {updated}
          </p>
        ) : null}
      </div>

      <span className="ml-10 inline-flex shrink-0 items-center gap-1 pt-1 text-[12.5px] font-medium text-niva-600 transition-colors duration-200 group-hover:text-niva-700 sm:ml-0 sm:items-center">
        View source
        <ExternalIcon
          size={13}
          className="transition-transform duration-200 group-hover:translate-x-0.5"
        />
        <span className="sr-only">(opens in a new tab)</span>
      </span>
    </a>
  );
}

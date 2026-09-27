import type { SourceCitation, SourceType } from "./types";

/**
 * Presentation helpers.
 *
 * These map the (frozen) backend contract onto customer-facing language.
 * Internal enum names, relevance scores and chunk metadata are never rendered.
 */

const MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

/** "2026-09-25" / "25 Sep 2026" -> "25 Sep 2026". Returns null if unparseable. */
export function formatSourceDate(value?: string | null): string | null {
  if (!value) return null;
  const trimmed = value.trim();
  if (!trimmed) return null;

  const human = /^(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{4})$/.exec(trimmed);
  if (human) return `${human[1]} ${human[2].slice(0, 3)} ${human[3]}`;

  const iso = /^(\d{4})-(\d{2})-(\d{2})/.exec(trimmed);
  if (iso) {
    const month = MONTHS[Number(iso[2]) - 1] ?? iso[2];
    return `${Number(iso[3])} ${month} ${iso[1]}`;
  }

  const parsed = new Date(trimmed);
  if (!Number.isNaN(parsed.getTime())) {
    return `${parsed.getDate()} ${MONTHS[parsed.getMonth()]} ${parsed.getFullYear()}`;
  }
  return null;
}

/** Friendly source-authority label, never the internal enum. */
const AUTHORITY: Record<SourceType, string> = {
  AMC_OFFICIAL: "HDFC Mutual Fund",
  AMFI: "AMFI",
  SEBI: "SEBI",
  REFERENCE: "Public source",
};

export function authorityLabel(
  type?: SourceType | null,
  fallback?: string | null,
): string | null {
  if (type && AUTHORITY[type]) return AUTHORITY[type];
  if (!fallback) return null;
  return /reference/i.test(fallback) ? "Public source" : fallback.trim();
}

/**
 * Prefer the clean scheme name. Fall back to the first segment of the long
 * page title (e.g. "HDFC Large Cap Fund Direct Growth - NAV, ..." ->
 * "HDFC Large Cap Fund Direct Growth").
 */
export function sourceDisplayName(source: SourceCitation): string {
  const scheme = source.scheme_name?.trim();
  if (scheme) return scheme;

  const title = source.title?.trim();
  if (!title) return "Source";

  const head = title.split(/\s+[|,\u00b7\u2013\u2014-]\s+/)[0];
  return (head || title).trim();
}

/** Pull a readable label out of the backend's clarification option objects. */
export function optionLabel(option: Record<string, string>): string {
  const preferred = [
    "label",
    "name",
    "text",
    "scheme_name",
    "title",
    "display_name",
    "id",
    "value",
  ];
  for (const key of preferred) {
    const found = option[key];
    if (typeof found === "string" && found.trim()) return found.trim();
  }
  for (const found of Object.values(option)) {
    if (typeof found === "string" && found.trim()) return found.trim();
  }
  return "Ask about this scheme";
}

/** Turn an answer into paragraphs without inventing content. */
export function toParagraphs(answer: string): string[] {
  return answer
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean);
}

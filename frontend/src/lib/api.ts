import type {
  ChatRequest,
  ChatResponse,
  HealthResponse,
  SchemeListResponse,
  SourceListResponse,
} from "./types";

/**
 * Browser-side API client.
 *
 * The backend is reached directly from the browser (the API is the only place
 * that holds the Gemini key). `NEXT_PUBLIC_API_BASE_URL` must therefore be
 * reachable from the user's machine; CORS on the API is restricted to the
 * frontend origin.
 *
 * There is deliberately no localhost fallback. A silently baked-in dev URL ships to
 * production and fails in a way that looks like an outage. A missing variable is a
 * configuration error and should fail loudly at startup.
 */
function resolveBaseUrl(): string {
  const configured = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (!configured) {
    throw new Error(
      "NEXT_PUBLIC_API_BASE_URL is not set. Copy frontend/.env.example to " +
        "frontend/.env.local for local work, or set it in your host's environment.",
    );
  }
  return configured.replace(/\/$/, "");
}

const BASE_URL = resolveBaseUrl();

export class ApiError extends Error {
  readonly status: number;
  readonly requestId?: string;

  constructor(message: string, status: number, requestId?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.requestId = requestId;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(
      `Cannot reach the API at ${BASE_URL}. Check that the backend is running.`,
      0,
    );
  }

  const requestId = response.headers.get("x-request-id") ?? undefined;
  const text = await response.text();
  const payload = text ? safeParse(text) : null;

  if (!response.ok) {
    throw new ApiError(
      extractErrorMessage(payload) ?? `Request failed (${response.status})`,
      response.status,
      requestId,
    );
  }
  return payload as T;
}

function safeParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

function extractErrorMessage(payload: unknown): string | null {
  if (typeof payload === "string") return payload;
  if (!payload || typeof payload !== "object") return null;
  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0] as { msg?: unknown };
    if (typeof first?.msg === "string") return first.msg;
  }
  return null;
}

export function apiBaseUrl(): string {
  return BASE_URL;
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function fetchSchemes(): Promise<SchemeListResponse> {
  return request<SchemeListResponse>("/api/v1/schemes");
}

export function fetchSources(): Promise<SourceListResponse> {
  return request<SourceListResponse>("/api/v1/sources");
}

export function sendChat(body: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>("/api/v1/chat", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

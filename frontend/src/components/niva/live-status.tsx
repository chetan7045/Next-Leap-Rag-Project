"use client";

import { useEffect, useRef, useState } from "react";
import { fetchHealth } from "@/lib/api";

type Reachability = "checking" | "live" | "offline";

const POLL_MS = 60_000;

/**
 * The single status chip in the product.
 *
 * Driven by `GET /health` behind the scenes. Only "Live" or "Offline" is ever
 * shown — no version, provider, model, index or chunk detail reaches the UI.
 */
export function LiveStatus() {
  const [state, setState] = useState<Reachability>("checking");
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;

    const check = async () => {
      try {
        const health = await fetchHealth();
        if (mounted.current) setState(health.status === "ok" ? "live" : "offline");
      } catch {
        if (mounted.current) setState("offline");
      }
    };

    void check();
    const timer = window.setInterval(check, POLL_MS);

    // A cold start on Render may have finished booting while the tab was hidden.
    const onVisible = () => {
      if (document.visibilityState === "visible") void check();
    };
    document.addEventListener("visibilitychange", onVisible);

    return () => {
      mounted.current = false;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  const live = state === "live";

  return (
    <span
      role="status"
      aria-live="polite"
      className={[
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1",
        "text-[12px] font-medium tracking-[-0.005em]",
        "transition-opacity duration-300",
        live
          ? "border-good-200 bg-good-50 text-good-700"
          : "border-bad-200 bg-bad-50 text-bad-600",
        state === "checking" ? "opacity-0" : "opacity-100",
      ].join(" ")}
    >
      <span className="relative flex size-1.5 shrink-0">
        <span
          className={[
            "absolute inline-flex size-full rounded-full",
            live ? "animate-ping bg-good-500/60" : "bg-bad-500/50",
          ].join(" ")}
        />
        <span
          className={[
            "relative inline-flex size-1.5 rounded-full",
            live ? "bg-good-500" : "bg-bad-600",
          ].join(" ")}
        />
      </span>
      {live ? "Live" : "Offline"}
    </span>
  );
}

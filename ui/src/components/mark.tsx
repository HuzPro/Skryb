import { useEffect, useRef } from "react";
import { onServerEvent } from "../lib/sse";
import type { AppInfo } from "../lib/types";

// The 256 grid of skryb/core/brand_mark.py: three rising bars feeding a text cursor.
const BARS = [
  { x: 56, height: 40 },
  { x: 86, height: 72 },
  { x: 116, height: 108 },
];
const BAR_WIDTH = 18;
const CARET = "152,58 200,58 200,72 185,72 185,184 200,184 200,198 152,198 152,184 167,184 167,72 152,72";
const QUIETEST = 0.3;

export type Motion = "still" | "voice" | "feed" | "sweep";

/** Recording follows the voice, transcribing streams the bars into the cursor, loading sweeps a glow across. */
export function motionFor(app: AppInfo | undefined): Motion {
  if (!app) return "still";
  if (app.state === "recording") return "voice";
  if (app.state === "transcribing") return "feed";
  if (app.state === "loading" || app.engine.loading) return "sweep";
  return "still";
}

/**
 * The app's mark, moving with what the app is doing. While recording, the
 * newest voice level lifts the bar beside the cursor and older ones pass
 * to the bars on its left, so the sound visibly flows into the cursor.
 */
export function Mark({ app, size = 20 }: { app: AppInfo | undefined; size?: number }) {
  const motion = motionFor(app);
  const bars = useRef<(SVGRectElement | null)[]>([]);

  useEffect(() => {
    if (motion !== "voice") return;
    const levels = BARS.map(() => 0);
    const show = () =>
      bars.current.forEach((bar, index) => {
        if (bar) bar.style.transform = `scaleY(${QUIETEST + (1 - QUIETEST) * (levels[index] ?? 0)})`;
      });
    show();
    const stop = onServerEvent((name, data) => {
      if (name !== "voice") return;
      levels.shift();
      levels.push(Math.min(Math.max((data as { level: number }).level, 0), 1));
      show();
    });
    return () => {
      stop();
      bars.current.forEach((bar) => bar?.style.removeProperty("transform"));
    };
  }, [motion]);

  return (
    <svg viewBox="0 0 256 256" width={size} height={size} aria-hidden className="mark shrink-0" data-motion={motion}>
      <defs>
        <linearGradient id="mark-tile" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#34D399" />
          <stop offset="1" stopColor="#047857" />
        </linearGradient>
      </defs>
      <rect width="256" height="256" rx="58" fill="url(#mark-tile)" />
      {BARS.map(({ x, height }, index) => (
        <rect
          key={x}
          ref={(bar) => {
            bars.current[index] = bar;
          }}
          className="mark-bar"
          style={{ animationDelay: `${index * (motion === "sweep" ? 0.15 : 0.43)}s` }}
          x={x}
          y={128 - height / 2}
          width={BAR_WIDTH}
          height={height}
          rx={BAR_WIDTH / 2}
          fill="#fff"
        />
      ))}
      <polygon className="mark-caret" points={CARET} fill="#fff" />
    </svg>
  );
}

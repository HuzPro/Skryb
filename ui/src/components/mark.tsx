// The 256 grid of skryb/core/brand_mark.py: three rising bars feeding a text cursor.
const BARS = [
  { x: 56, height: 40 },
  { x: 86, height: 72 },
  { x: 116, height: 108 },
];
const BAR_WIDTH = 18;
const CARET = "152,58 200,58 200,72 185,72 185,184 200,184 200,198 152,198 152,184 167,184 167,72 152,72";

/** The app's mark, standing still. */
export function Mark({ size = 20 }: { size?: number }) {
  return (
    <svg viewBox="0 0 256 256" width={size} height={size} aria-hidden className="shrink-0">
      <defs>
        <linearGradient id="mark-tile" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#34D399" />
          <stop offset="1" stopColor="#047857" />
        </linearGradient>
      </defs>
      <rect width="256" height="256" rx="58" fill="url(#mark-tile)" />
      {BARS.map(({ x, height }) => (
        <rect key={x} x={x} y={128 - height / 2} width={BAR_WIDTH} height={height} rx={BAR_WIDTH / 2} fill="#fff" />
      ))}
      <polygon points={CARET} fill="#fff" />
    </svg>
  );
}

import { ChevronRight, type LucideIcon } from "lucide-react";
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { keyCaps } from "../lib/keys";
import { cn } from "../lib/utils";
import { Input } from "./ui/input";

/** Every page starts at the same left edge and top; settings pages keep to a readable width. */
export const PAGE_EDGE = "px-8 pt-3";

export function PageBody({ children }: { children: ReactNode }) {
  return <div className={cn(PAGE_EDGE, "w-full max-w-[760px] pb-28")}>{children}</div>;
}

export function PageTitle({ title, children }: { title: ReactNode; children?: ReactNode }) {
  return (
    <div className="mb-1 flex min-h-10 items-center justify-between gap-4">
      <h1 className="page-title min-w-0 truncate">{title}</h1>
      {children}
    </div>
  );
}

export function Section({ title, note, children, action }: { title: string; note?: ReactNode; children: ReactNode; action?: ReactNode }) {
  return (
    <section className="mt-7">
      <div className="mb-2 flex items-end justify-between gap-4">
        <h2 className="text-[13px] font-semibold text-muted-foreground">{title}</h2>
        {action}
      </div>
      {note ? <p className="-mt-1 mb-2.5 text-[12.5px] text-muted-foreground">{note}</p> : null}
      {children}
    </section>
  );
}

export function Card({ className, children, title }: { className?: string; children?: ReactNode; title?: string }) {
  return (
    <div title={title} className={cn("surface rounded-xl border border-edge bg-card shadow-card", className)}>
      {children}
    </div>
  );
}

/** Rows inside one card, separated by hairlines. */
export function Rows({ children, className }: { children: ReactNode; className?: string }) {
  return <Card className={cn("divide-y", className)}>{children}</Card>;
}

export function Row({
  title,
  description,
  children,
  error,
  stacked = false,
}: {
  title: ReactNode;
  description?: ReactNode;
  children?: ReactNode;
  error?: string;
  stacked?: boolean;
}) {
  return (
    <div className="px-4 py-3">
      <div className={cn("flex gap-x-6 gap-y-2.5", stacked ? "flex-col" : "items-center justify-between")}>
        <div className="min-w-0 flex-1">
          <div className="text-[13.5px] font-medium">{title}</div>
          {description ? <div className="mt-0.5 max-w-[440px] text-[12.5px] text-muted-foreground">{description}</div> : null}
        </div>
        {children ? <div className={cn("flex shrink-0 items-center gap-2", stacked && "w-full")}>{children}</div> : null}
      </div>
      {error ? <InlineError>{error}</InlineError> : null}
    </div>
  );
}

export function InlineError({ children }: { children: ReactNode }) {
  return <p className="selectable mt-2 text-[12.5px] text-destructive">{children}</p>;
}

export interface Choice<T extends string> {
  value: T;
  title: ReactNode;
  description: string;
  disabled?: boolean;
}

export function RadioDot({ selected }: { selected: boolean }) {
  return (
    <span className={cn("grid size-3.5 shrink-0 place-items-center rounded-full border", selected ? "border-primary" : "border-input")}>
      {selected ? <span className="size-1.5 rounded-full bg-primary" /> : null}
    </span>
  );
}

/** Mutually exclusive options as cards with a radio dot: Mode, Output mode. */
export function ChoiceCards<T extends string>({ value, choices, onChange, label }: { value: T; choices: Choice<T>[]; onChange: (value: T) => void; label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="grid gap-2.5" style={{ gridTemplateColumns: `repeat(${choices.length}, minmax(0, 1fr))` }}>
      {choices.map((choice) => {
        const selected = choice.value === value;
        return (
          <button
            key={choice.value}
            role="radio"
            aria-checked={selected}
            disabled={choice.disabled}
            onClick={() => onChange(choice.value)}
            className={cn(
              "flex flex-col items-start surface rounded-xl border border-edge bg-card px-3.5 py-3 text-left shadow-card outline-none transition-colors hover:bg-accent/50 focus-visible:ring-2 focus-visible:ring-ring/40 disabled:pointer-events-none disabled:opacity-55",
              selected && "border-primary bg-selected hover:bg-selected",
            )}
          >
            <span className="flex items-center gap-2 text-[13.5px] font-medium">
              <RadioDot selected={selected} />
              {choice.title}
            </span>
            <span className="mt-1 pl-5.5 text-[12.5px] text-muted-foreground">{choice.description}</span>
          </button>
        );
      })}
    </div>
  );
}

export function KeyCaps({ combo, placeholder = "Not set" }: { combo: string; placeholder?: string }) {
  const caps = keyCaps(combo);
  if (!caps.length) return <span className="text-[12.5px] text-muted-foreground">{placeholder}</span>;
  return (
    <span className="flex flex-wrap items-center gap-1">
      {caps.map((cap, index) => (
        <kbd
          key={`${cap}-${index}`}
          className="inline-flex h-6 min-w-6 items-center justify-center rounded-md border border-b-2 border-input bg-card px-1.5 font-sans text-[12px] font-medium shadow-xs"
        >
          {cap}
        </kbd>
      ))}
    </span>
  );
}

export function ProgressBar({ value, className }: { value: number; className?: string }) {
  return (
    <div className={cn("h-1 overflow-hidden rounded-full bg-muted", className)}>
      <div className="h-full rounded-full bg-primary transition-[width] duration-150" style={{ width: `${Math.round(Math.min(Math.max(value, 0), 1) * 100)}%` }} />
    </div>
  );
}

/** A number box that holds what is typed and commits a clamped value when the user leaves it. */
export function NumberField({
  value,
  min,
  max,
  step,
  onChange,
  suffix,
  label,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  suffix: string;
  label: string;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const commit = () => {
    const parsed = Number.parseFloat(text);
    const next = Number.isFinite(parsed) ? Math.min(Math.max(parsed, min), max) : value;
    setText(String(next));
    if (next !== value) onChange(next);
  };
  return (
    <label className="flex items-center gap-1.5 text-[12.5px] text-muted-foreground">
      <Input
        aria-label={label}
        type="number"
        className="w-[72px] text-foreground"
        value={text}
        min={min}
        max={max}
        step={step}
        onChange={(event) => setText(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => event.key === "Enter" && commit()}
      />
      {suffix}
    </label>
  );
}

/** A native disclosure styled as a quiet link: "Advanced". */
export function Disclosure({ label, children, defaultOpen = false }: { label: string; children: ReactNode; defaultOpen?: boolean }) {
  return (
    <details className="group mt-3" open={defaultOpen}>
      <summary className="inline-flex cursor-default list-none items-center gap-1 text-[13px] font-medium text-primary [&::-webkit-details-marker]:hidden">
        {label}
        <ChevronRight className="size-3.5 transition-transform group-open:rotate-90" />
      </summary>
      <div className="mt-2">{children}</div>
    </details>
  );
}

export type Tone = "success" | "warning" | "destructive" | "muted" | "primary";

export function StatusDot({ tone }: { tone: Tone }) {
  const color = {
    success: "text-success",
    warning: "text-warning",
    destructive: "text-destructive",
    muted: "text-muted-foreground/60",
    primary: "text-primary",
  }[tone];
  return <span className={cn("inline-block size-1.5 shrink-0 rounded-[var(--r-dot)] bg-current shadow-[var(--dot-glow)]", color)} />;
}

const EXIT_MS = 160;

export interface MenuItem {
  label: string;
  icon: LucideIcon;
  onSelect: () => void;
  /** The key that does the same without the menu, shown on the right. */
  shortcut?: string;
  danger?: boolean;
}

/** A right-click menu at (x, y), kept inside the window. Escape, a click elsewhere, scrolling or leaving the window closes it. */
export function ContextMenu({ x, y, label, items, onClose: closed }: { x: number; y: number; label: string; items: MenuItem[]; onClose: () => void }) {
  const menu = useRef<HTMLDivElement>(null);
  const opener = useRef<Element | null>(null);
  const [place, setPlace] = useState({ left: x, top: y });
  const [leaving, setLeaving] = useState(false);
  // It fades out before it goes; reduced motion makes that fade a millisecond.
  const onClose = useCallback(() => setLeaving(true), []);
  useEffect(() => {
    if (!leaving) return;
    const timer = setTimeout(closed, EXIT_MS);
    return () => clearTimeout(timer);
  }, [leaving, closed]);
  useLayoutEffect(() => {
    opener.current = document.activeElement;
    const box = menu.current!.getBoundingClientRect();
    setPlace({ left: Math.max(4, Math.min(x, window.innerWidth - box.width - 4)), top: Math.max(4, Math.min(y, window.innerHeight - box.height - 4)) });
    menu.current!.querySelector("button")?.focus();
  }, [x, y]);
  useEffect(() => {
    const elsewhere = (event: Event) => {
      if (!menu.current?.contains(event.target as Node)) onClose();
    };
    window.addEventListener("pointerdown", elsewhere, true);
    window.addEventListener("scroll", elsewhere, true);
    window.addEventListener("blur", onClose);
    window.addEventListener("resize", onClose);
    return () => {
      window.removeEventListener("pointerdown", elsewhere, true);
      window.removeEventListener("scroll", elsewhere, true);
      window.removeEventListener("blur", onClose);
      window.removeEventListener("resize", onClose);
    };
  }, [onClose]);
  const close = () => {
    onClose();
    if (opener.current instanceof HTMLElement) opener.current.focus();
  };
  const keys = (event: KeyboardEvent<HTMLDivElement>) => {
    const buttons = [...menu.current!.querySelectorAll("button")];
    const at = buttons.indexOf(document.activeElement as HTMLButtonElement);
    const step = event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0;
    if (event.key === "Escape" || event.key === "Tab") {
      event.preventDefault();
      close();
    } else if (step) {
      event.preventDefault();
      buttons[(at + step + buttons.length) % buttons.length]?.focus();
    }
  };
  return (
    <div
      ref={menu}
      role="menu"
      aria-label={label}
      onKeyDown={keys}
      onContextMenu={(event) => event.preventDefault()}
      style={place}
      className={cn("surface fixed z-50 min-w-[176px] origin-top-left rounded-lg border border-edge bg-popover p-1 shadow-lg", leaving ? "pointer-events-none exit-pop" : "enter-pop")}
    >
      {items.map(({ label: name, icon: Icon, onSelect, shortcut, danger }) => (
        <button
          key={name}
          role="menuitem"
          onClick={() => {
            close();
            onSelect();
          }}
          className={cn(
            "flex h-8 w-full items-center gap-2.5 rounded-md px-2.5 text-[13px] outline-none hover:bg-accent focus:bg-accent [&_svg]:size-4",
            danger ? "text-destructive" : "text-foreground",
          )}
        >
          <Icon />
          <span className="flex-1 text-left">{name}</span>
          {shortcut ? <span className="text-[11.5px] text-muted-foreground">{shortcut}</span> : null}
        </button>
      ))}
    </div>
  );
}

/** One number on a card: what it counts, the number, and a line under it. */
export function Stat({ label, value, detail, className, title }: { label: string; value: string; detail?: ReactNode; className?: string; title?: string }) {
  return (
    <Card className={cn("px-5 py-4", className)} title={title}>
      <p className="text-[12.5px] text-muted-foreground">{label}</p>
      <p className="mt-1 text-[24px] leading-tight font-semibold tabular-nums">{value}</p>
      {detail ? <p className="mt-0.5 text-[12.5px] text-muted-foreground">{detail}</p> : null}
    </Card>
  );
}

/** One bar of a `Bars` chart: its value and what hovering it says it is. */
export interface Bar {
  value: number;
  label: string;
}

/**
 * Bars from a shared zero baseline, the last one drawn lighter while its period is still running. A dashed line marks
 * the usual value when there is one. Each bar's whole column takes the hover, so a short bar is as easy to point at as
 * a tall one; a table carries the numbers for screen readers.
 */
export function Bars({
  title,
  bars,
  format,
  axis,
  usual,
  partial = true,
}: {
  title: string;
  bars: Bar[];
  format: (value: number) => string;
  axis: [string, string];
  usual?: number | null;
  partial?: boolean;
}) {
  const [hovered, setHovered] = useState<number | null>(null);
  const top = Math.max(...bars.map((bar) => bar.value), usual ?? 0, 0);
  const shown = bars[hovered ?? bars.length - 1];
  const usualAt = usual && top > 0 ? (usual / top) * 100 : null;
  return (
    <Card className="flex flex-col px-5 py-4">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="text-[12.5px] text-muted-foreground">{title}</h3>
        {shown ? (
          <p className="text-[12.5px] tabular-nums" aria-hidden>
            <span className="text-muted-foreground">{shown.label} </span>
            <span className="font-medium">{format(shown.value)}</span>
          </p>
        ) : null}
      </div>
      <div aria-hidden className="relative mt-3 flex min-h-28 flex-1 items-end gap-[2px] border-b border-edge" onPointerLeave={() => setHovered(null)}>
        {bars.map((bar, index) => {
          const unfinished = partial && index === bars.length - 1;
          return (
            <div key={index} className="flex h-full flex-1 items-end" onPointerEnter={() => setHovered(index)}>
              <div
                className={cn(
                  "w-full rounded-t-[4px] transition-colors",
                  bar.value <= 0 ? "bg-muted" : unfinished ? "bg-primary/45" : "bg-primary",
                  hovered !== null && hovered !== index && bar.value > 0 && "opacity-50",
                )}
                style={{ height: bar.value > 0 && top > 0 ? `max(3px, ${(bar.value / top) * 100}%)` : "2px" }}
              />
            </div>
          );
        })}
        {usualAt !== null ? (
          <div className="pointer-events-none absolute inset-x-0 border-t border-dashed border-foreground/40" style={{ bottom: `${usualAt}%` }}>
            <span className={cn("absolute right-0 bg-card pl-1 text-[11px] text-muted-foreground tabular-nums", usualAt > 80 ? "top-[2px]" : "-top-[17px]")}>
              usual {format(usual ?? 0)}
            </span>
          </div>
        ) : null}
      </div>
      <div aria-hidden className="mt-1.5 flex justify-between text-[11.5px] text-muted-foreground">
        <span>{axis[0]}</span>
        <span>{axis[1]}</span>
      </div>
      <table className="sr-only">
        <caption>{title}</caption>
        <tbody>
          {bars.map((bar, index) => (
            <tr key={index}>
              <th scope="row">{bar.label}</th>
              <td>{format(bar.value)}</td>
            </tr>
          ))}
          {usual ? (
            <tr>
              <th scope="row">Usual</th>
              <td>{format(usual)}</td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </Card>
  );
}

const DAY_LABEL: Intl.DateTimeFormatOptions = { weekday: "short", day: "numeric", month: "short" };

/** Bars for the last `values.length` days, today last and labelled "Today so far". */
export function dayBars(values: number[], now: Date = new Date()): Bar[] {
  return values.map((value, index) => {
    const back = values.length - 1 - index;
    const day = new Date(now.getFullYear(), now.getMonth(), now.getDate() - back);
    return { value, label: back ? day.toLocaleDateString(undefined, DAY_LABEL) : "Today so far" };
  });
}

/** "4 Oct" for the first of `count` days ending today. */
export function firstDayLabel(count: number, now: Date = new Date()): string {
  return new Date(now.getFullYear(), now.getMonth(), now.getDate() - (count - 1)).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

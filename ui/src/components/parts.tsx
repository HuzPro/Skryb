import { ChevronRight, type LucideIcon } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { keyCaps } from "../lib/keys";
import { cn } from "../lib/utils";
import { Input } from "./ui/input";

export function PageBody({ children }: { children: ReactNode }) {
  return <div className="mx-auto w-full max-w-[720px] px-8 pt-3 pb-28">{children}</div>;
}

export function PageTitle({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="mb-1 flex min-h-8 items-center justify-between gap-4">
      <h1 className="text-[20px] font-semibold tracking-tight">{title}</h1>
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

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("rounded-xl border bg-card shadow-[0_1px_2px_rgb(0_0_0/0.03)]", className)}>{children}</div>;
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
  return <p className="mt-2 text-[12.5px] text-destructive">{children}</p>;
}

export interface Choice<T extends string> {
  value: T;
  title: string;
  description: string;
  disabled?: boolean;
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
              "flex flex-col items-start rounded-xl border bg-card px-3.5 py-3 text-left shadow-[0_1px_2px_rgb(0_0_0/0.03)] outline-none transition-colors duration-150 hover:bg-accent/50 focus-visible:ring-2 focus-visible:ring-ring/40 disabled:pointer-events-none disabled:opacity-55",
              selected && "border-primary bg-selected hover:bg-selected",
            )}
          >
            <span className="flex items-center gap-2 text-[13.5px] font-medium">
              <span className={cn("grid size-3.5 place-items-center rounded-full border", selected ? "border-primary" : "border-input")}>
                {selected ? <span className="size-1.5 rounded-full bg-primary" /> : null}
              </span>
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
        <ChevronRight className="size-3.5 transition-transform duration-150 group-open:rotate-90" />
      </summary>
      <div className="mt-2">{children}</div>
    </details>
  );
}

export function StatusDot({ tone }: { tone: "success" | "warning" | "destructive" | "muted" | "primary" }) {
  const color = {
    success: "bg-success",
    warning: "bg-warning",
    destructive: "bg-destructive",
    muted: "bg-muted-foreground/60",
    primary: "bg-primary",
  }[tone];
  return <span className={cn("inline-block size-1.5 shrink-0 rounded-full", color)} />;
}

export interface MenuItem {
  label: string;
  icon: LucideIcon;
  onSelect: () => void;
  /** The key that does the same without the menu, shown on the right. */
  shortcut?: string;
  danger?: boolean;
}

/** A right-click menu at (x, y), kept inside the window. Escape, a click elsewhere, scrolling or leaving the window closes it. */
export function ContextMenu({ x, y, label, items, onClose }: { x: number; y: number; label: string; items: MenuItem[]; onClose: () => void }) {
  const menu = useRef<HTMLDivElement>(null);
  const opener = useRef<Element | null>(null);
  const [place, setPlace] = useState({ left: x, top: y });
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
      className="fixed z-50 min-w-[176px] animate-[fade-in_100ms_ease-out] rounded-lg border bg-card p-1 shadow-lg"
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

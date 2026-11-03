import { useQuery } from "@tanstack/react-query";
import { CircleAlert, LoaderCircle, Mic, Minus, Settings, Square, Users, X, type LucideIcon } from "lucide-react";
import { createContext, useContext, useEffect, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";
import { Link } from "react-router";
import { isDirty } from "../lib/draft";
import { windowControls } from "../lib/platform";
import { queries } from "../lib/queries";
import { useStore } from "../lib/store";
import type { AppInfo } from "../lib/types";
import { cn } from "../lib/utils";
import { useInstall } from "./install";
import { Mark } from "./mark";
import { StatusCard, StatusDot, type Tone } from "./parts";
import { Button } from "./ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogTitle } from "./ui/dialog";

export type Mode = "dictation" | "meetings";

/** One sidebar item; both modes show the same items, each opening its own page per mode. */
export interface NavItem {
  label: string;
  icon: LucideIcon;
  dictation: string;
  meetings: string;
}

export function useAppInfo(): AppInfo | undefined {
  return useQuery(queries.app).data;
}

/** True once the core has been unreachable for five seconds. */
export const LostContext = createContext(false);

export function useEngineStatus(): { tone: Tone; label: string; app: AppInfo | undefined } {
  const app = useAppInfo();
  const install = useInstall();
  const updating = install !== null && install.progress.step !== "failed";
  return { ...engineStatus(app, useContext(LostContext), updating), app };
}

export function engineStatus(app: AppInfo | undefined, lost: boolean, updating = false): { tone: Tone; label: string } {
  if (updating) return { tone: "primary", label: "Updating" };
  if (lost) return { tone: "destructive", label: "Not running" };
  if (!app) return { tone: "muted", label: "Connecting" };
  if (app.state === "recording") return { tone: "destructive", label: "Recording" };
  if (app.state === "transcribing") return { tone: "primary", label: "Transcribing" };
  if (app.engine.error) return { tone: "destructive", label: "Failed" };
  if (app.engine.loading || app.state === "loading") return { tone: "warning", label: "Loading" };
  if (app.engine.ready && app.engine.device) {
    return { tone: app.engine.device_note ? "warning" : "success", label: `Ready on ${app.engine.device}` };
  }
  if (app.engine.ready) return { tone: "success", label: "Ready" };
  return { tone: "muted", label: "Not loaded" };
}

function WindowButton({ label, onClick, danger, children }: { label: string; onClick: () => void; danger?: boolean; children: ReactNode }) {
  return (
    <button
      aria-label={label}
      title={label}
      onClick={onClick}
      className={cn(
        "grid h-full w-11 place-items-center text-muted-foreground transition-colors hover:bg-accent hover:text-foreground [&_svg]:size-3.5",
        danger && "hover:bg-[#e81123] hover:text-white",
      )}
    >
      {children}
    </button>
  );
}

/** The draggable strip above the content; it shows the engine's status only where the sidebar doesn't, in setup. */
export function TopBar({ nativeTitlebar, onClose, showStatus, children }: { nativeTitlebar: boolean; onClose: () => void; showStatus: boolean; children?: ReactNode }) {
  const status = useEngineStatus();
  return (
    <div data-tauri-drag-region className="flex h-10 shrink-0 items-stretch justify-end">
      <div data-tauri-drag-region className="flex flex-1 items-center px-4">
        {children}
      </div>
      {showStatus ? (
        <div data-tauri-drag-region className="flex items-center pr-2">
          <span className="pointer-events-none flex items-center gap-1.5 rounded-[var(--r-pill)] border bg-card px-2.5 py-0.5 text-[11.5px] text-muted-foreground">
            <StatusDot tone={status.tone} />
            {status.label}
          </span>
        </div>
      ) : null}
      {nativeTitlebar ? null : (
        <div className="flex">
          <WindowButton label="Minimise" onClick={windowControls.minimize}>
            <Minus />
          </WindowButton>
          <WindowButton label="Maximise" onClick={windowControls.toggleMaximize}>
            <Square className="!size-3" />
          </WindowButton>
          <WindowButton label="Close" onClick={onClose} danger>
            <X />
          </WindowButton>
        </div>
      )}
    </div>
  );
}

const MODES: { value: Mode; label: string; icon: LucideIcon }[] = [
  { value: "dictation", label: "Dictation", icon: Mic },
  { value: "meetings", label: "Meetings", icon: Users },
];

/** A segmented switch: the thumb glides to the chosen mode without overshooting its track; arrow keys move it too. */
export function ModeToggle({ mode, onChange, recording }: { mode: Mode; onChange: (mode: Mode) => void; recording: boolean }) {
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const at = Math.max(0, MODES.findIndex((choice) => choice.value === mode));
  const keys = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 0;
    if (!step) return;
    event.preventDefault();
    const next = (at + step + MODES.length) % MODES.length;
    onChange(MODES[next]!.value);
    buttons.current[next]?.focus();
  };
  return (
    <div role="radiogroup" aria-label="Mode" onKeyDown={keys} className="relative grid h-8 grid-cols-2 rounded-button bg-muted p-[3px]">
      <span
        aria-hidden
        className="absolute inset-y-[3px] left-[3px] w-[calc(50%-3px)] transition-transform duration-[var(--page-dur)] rounded-[calc(var(--r-button)-3px)] border border-edge bg-card shadow-xs"
        style={{ transform: `translateX(${at * 100}%)` }}
      />
      {MODES.map(({ value, label, icon: Icon }, index) => {
        const selected = value === mode;
        const hint = value === "meetings" && recording ? `${label} (recording now)` : label;
        return (
          <button
            key={value}
            ref={(element) => {
              buttons.current[index] = element;
            }}
            role="radio"
            aria-checked={selected}
            aria-label={hint}
            title={hint}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(value)}
            className={cn(
              "relative flex items-center justify-center gap-1.5 rounded-[calc(var(--r-button)-3px)] text-[12.5px] font-medium text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/40 [&_svg]:size-3.5",
              selected && "text-foreground",
            )}
          >
            <Icon />
            {label}
            {value === "meetings" && recording ? <span className="size-1.5 rounded-full bg-destructive motion-safe:animate-pulse" /> : null}
          </button>
        );
      })}
    </div>
  );
}

export const SIDEBAR = { standard: 212, least: 180, most: 520 };
// The page beside the sidebar keeps at least this much of the window.
const PAGE_ROOM = 420;
const SIDEBAR_WIDTH = "sidebar-width";

type WidthMemory = Pick<Storage, "getItem" | "setItem">;

function browserStorage(): WidthMemory | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

/** A sidebar width that fits the window: never narrower than 180 px, never so wide the page has no room. */
export function fitSidebar(width: number, windowWidth: number): number {
  return Math.round(Math.max(SIDEBAR.least, Math.min(width, SIDEBAR.most, windowWidth - PAGE_ROOM)));
}

/** The width the sidebar was last dragged to; the standard width the first time or when storage is blocked. */
export function savedSidebarWidth(memory: WidthMemory | null = browserStorage()): number {
  try {
    const width = Number(memory?.getItem(SIDEBAR_WIDTH));
    return Number.isFinite(width) && width > 0 ? width : SIDEBAR.standard;
  } catch {
    return SIDEBAR.standard;
  }
}

export function saveSidebarWidth(width: number, memory: WidthMemory | null = browserStorage()): void {
  try {
    memory?.setItem(SIDEBAR_WIDTH, String(Math.round(width)));
  } catch {
    // Storage blocked: the sidebar opens at the standard width next time.
  }
}

function useSidebarWidth(): [number, (width: number) => void] {
  const [wanted, setWanted] = useState(savedSidebarWidth);
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  useEffect(() => {
    const resized = () => setWindowWidth(window.innerWidth);
    window.addEventListener("resize", resized);
    return () => window.removeEventListener("resize", resized);
  }, []);
  return [fitSidebar(wanted, windowWidth), setWanted];
}

/** The sidebar's right edge: drag it, or use the arrow keys, to widen it; double-click puts it back. Saved once it settles. */
function SidebarEdge({ width, onResize }: { width: number; onResize: (width: number) => void }) {
  const resize = (next: number) => {
    onResize(next);
    saveSidebarWidth(next);
  };
  const drag = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const edge = event.currentTarget;
    const from = event.clientX;
    let last = width;
    const move = (moved: globalThis.PointerEvent) => {
      last = fitSidebar(width + moved.clientX - from, window.innerWidth);
      onResize(last);
    };
    const stop = () => {
      edge.removeEventListener("pointermove", move);
      edge.removeEventListener("pointerup", stop);
      edge.removeEventListener("pointercancel", stop);
      saveSidebarWidth(last);
    };
    edge.setPointerCapture(event.pointerId);
    edge.addEventListener("pointermove", move);
    edge.addEventListener("pointerup", stop);
    edge.addEventListener("pointercancel", stop);
  };
  const nudge = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.key === "ArrowRight" ? 16 : event.key === "ArrowLeft" ? -16 : 0;
    if (!step) return;
    event.preventDefault();
    resize(fitSidebar(width + step, window.innerWidth));
  };
  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label="Sidebar width"
      aria-valuemin={SIDEBAR.least}
      aria-valuemax={SIDEBAR.most}
      aria-valuenow={width}
      tabIndex={0}
      title="Drag to resize. Double-click to reset."
      onPointerDown={drag}
      onDoubleClick={() => resize(SIDEBAR.standard)}
      onKeyDown={nudge}
      className="absolute inset-y-0 -right-[3px] z-10 w-[6px] cursor-col-resize outline-none transition-colors hover:bg-primary/40 focus-visible:bg-primary/60 active:bg-primary/60"
    />
  );
}

/** Dictation's footer card: whether the engine is ready, and with which model. */
export function DictationStatusCard() {
  const status = useEngineStatus();
  const app = status.app;
  const saved = useQuery(queries.settings).data;
  return (
    <StatusCard
      tone={status.tone}
      label={status.label}
      detail={app?.engine.model || saved?.model_size || "No model"}
      warning={app?.engine.device_note ?? undefined}
    />
  );
}

export function Sidebar({
  items,
  mode,
  onMode,
  pathname,
  isActive,
  status,
}: {
  items: NavItem[];
  mode: Mode;
  onMode: (mode: Mode) => void;
  pathname: string;
  isActive: (item: NavItem, pathname: string) => boolean;
  status: ReactNode;
}) {
  const [width, setWidth] = useSidebarWidth();
  const app = useAppInfo();
  const settingsOpen = pathname === "/updates";
  return (
    <aside style={{ width }} className="surface relative flex shrink-0 flex-col border-r border-[var(--sidebar-edge)] bg-sidebar">
      <SidebarEdge width={width} onResize={setWidth} />
      <div data-tauri-drag-region className="flex h-10 shrink-0 items-center gap-2 px-4">
        <span className="pointer-events-none flex">
          <Mark />
        </span>
        <span className="pointer-events-none text-[13px] font-semibold">Skryb</span>
      </div>
      <div className="px-3 pt-0.5">
        <ModeToggle mode={mode} onChange={onMode} recording={app?.meeting_recording ?? false} />
      </div>
      <nav aria-label="Pages" className="flex flex-col gap-0.5 px-2 pt-3">
        {items.map((item) => {
          const active = isActive(item, pathname);
          const Icon = item.icon;
          return (
            <Link
              key={item.label}
              to={item[mode]}
              draggable={false}
              aria-current={active ? "page" : undefined}
              className={cn(
                "flex h-8 items-center gap-2.5 rounded-nav px-2.5 text-[13px] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
                active && "bg-card font-medium text-foreground shadow-[0_0_0_1px_var(--edge)] hover:bg-card",
              )}
            >
              <Icon className="size-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="mt-auto grid grid-cols-[minmax(0,1fr)_auto] items-center gap-1.5 p-3">
        {status}
        <Link
          to="/updates"
          draggable={false}
          aria-label="Settings"
          title="Settings"
          aria-current={settingsOpen ? "page" : undefined}
          className={cn(
            "surface grid size-8 place-items-center rounded-lg border border-edge bg-card text-muted-foreground shadow-card transition-[color,background-color,scale] hover:bg-accent hover:text-foreground active:scale-[0.97]",
            settingsOpen && "text-foreground ring-2 ring-ring/40",
          )}
        >
          <Settings className="size-4" />
        </Link>
      </div>
    </aside>
  );
}

/** Floats over the bottom of the page while settings differ from what the core has, or are being applied. */
export function SaveBar() {
  const { settings, saveSettings, discardSettings, savePhase, saveNote, saveError, dismissSaveError } = useStore();
  const dirty = isDirty(settings);
  const open = dirty || savePhase !== "idle" || Boolean(saveError);
  const busy = savePhase !== "idle";
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-4 z-20 flex justify-center px-6">
      <div
        role="status"
        aria-hidden={!open}
        inert={!open}
        className={cn(
          "surface spring flex w-full max-w-[720px] items-center gap-3 rounded-xl border border-edge bg-popover py-2 pr-2 pl-4 shadow-lg",
          open ? "pointer-events-auto" : "translate-y-3 scale-[0.98] opacity-0",
        )}
      >
        {busy ? <LoaderCircle className="size-4 shrink-0 animate-spin text-primary" /> : saveError ? <CircleAlert className="size-4 shrink-0 text-destructive" /> : null}
        <span className={cn("min-w-0 flex-1 text-[13px]", saveError && !busy && "text-destructive")}>
          {savePhase === "saving" ? "Saving..." : savePhase === "applying" ? `Applying... ${saveNote}` : saveError || "You have unsaved changes"}
        </span>
        {dirty && !busy ? (
          <>
            <Button variant="ghost" onClick={discardSettings}>
              Discard
            </Button>
            <Button onClick={() => void saveSettings()}>Save</Button>
          </>
        ) : !busy ? (
          <Button variant="ghost" onClick={dismissSaveError}>
            Dismiss
          </Button>
        ) : null}
      </div>
    </div>
  );
}

export function ConnectionBanner({ onQuit }: { onQuit: () => void }) {
  return (
    <div role="alert" className="surface enter-fade absolute inset-x-0 top-10 z-30 flex items-center gap-3 border-b border-destructive/30 bg-popover px-4 py-2.5 shadow-md">
      <CircleAlert className="size-4 shrink-0 text-destructive" />
      <span className="flex-1 text-[13px] font-medium">Skryb is not running</span>
      <span className="text-[12.5px] text-muted-foreground">Start it from the Start menu, then this window reconnects.</span>
      <Button variant="outline" onClick={onQuit}>
        Quit
      </Button>
    </div>
  );
}

/** The open question, kept on screen while its dialog animates out after it is answered. */
export function Prompts() {
  const { prompt, closePrompt } = useStore();
  const last = useRef(prompt);
  if (prompt) last.current = prompt;
  const shown = last.current;
  if (!shown) return null;
  const open = prompt !== null;
  const answer = <T,>(resolve: (value: T) => void, value: T) => {
    if (!open) return;
    closePrompt();
    resolve(value);
  };
  if (shown.kind === "license") {
    return (
      <Dialog open={open} onOpenChange={() => answer(shown.resolve, false)}>
        <DialogContent className="w-[min(560px,calc(100vw-48px))]">
          <DialogTitle>{shown.license.title}</DialogTitle>
          <DialogDescription>Read and accept the licence to download and use this model.</DialogDescription>
          <pre className="selectable max-h-[46vh] overflow-y-auto rounded-lg border bg-muted/50 p-3 font-sans text-[12.5px] whitespace-pre-wrap">{shown.license.text}</pre>
          {shown.license.url ? <p className="selectable -mt-2 truncate text-[12px] text-muted-foreground">Source: {shown.license.url}</p> : null}
          <DialogFooter>
            <Button variant="ghost" onClick={() => answer(shown.resolve, false)}>
              Cancel
            </Button>
            <Button onClick={() => answer(shown.resolve, true)}>Accept</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }
  if (shown.kind === "confirm") {
    return (
      <Dialog open={open} onOpenChange={() => answer(shown.resolve, false)}>
        <DialogContent>
          <DialogTitle>{shown.title}</DialogTitle>
          <DialogDescription>{shown.message}</DialogDescription>
          <DialogFooter>
            <Button variant="ghost" onClick={() => answer(shown.resolve, false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={() => answer(shown.resolve, true)}>
              {shown.action}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }
  return (
    <Dialog open={open} onOpenChange={() => answer(shown.resolve, "cancel")}>
      <DialogContent>
        <DialogTitle>Save your changes?</DialogTitle>
        <DialogDescription>You have unsaved changes. Save them before closing, or discard them.</DialogDescription>
        <DialogFooter>
          <Button variant="ghost" onClick={() => answer(shown.resolve, "cancel")}>
            Cancel
          </Button>
          <Button variant="outline" onClick={() => answer(shown.resolve, "discard")}>
            Discard
          </Button>
          <Button onClick={() => answer(shown.resolve, "save")}>Save</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

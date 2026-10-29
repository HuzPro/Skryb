import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CircleAlert, FileDown, FileUp, FolderOpen, LoaderCircle, Mic, Minus, Square, Users, X, type LucideIcon } from "lucide-react";
import { createContext, useContext, useEffect, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";
import { NavLink } from "react-router";
import { api, errorText } from "../lib/api";
import { isDirty } from "../lib/draft";
import { chooseOpenPath, chooseSavePath, windowControls } from "../lib/platform";
import { useStore } from "../lib/store";
import type { AppInfo } from "../lib/types";
import { cn } from "../lib/utils";
import { useInstall } from "./install";
import { Mark } from "./mark";
import { StatusDot } from "./parts";
import { Button } from "./ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogTitle } from "./ui/dialog";

export interface NavItem {
  path: string;
  label: string;
  icon: LucideIcon;
}

type Tone = "success" | "warning" | "destructive" | "muted" | "primary";

export function useAppInfo(): AppInfo | undefined {
  return useQuery({ queryKey: ["app"], queryFn: () => api.get<AppInfo>("/api/app") }).data;
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
        "grid h-full w-11 place-items-center text-muted-foreground transition-colors duration-150 hover:bg-accent hover:text-foreground [&_svg]:size-3.5",
        danger && "hover:bg-[#e81123] hover:text-white",
      )}
    >
      {children}
    </button>
  );
}

/** The draggable strip above the content: engine status on the right, then the window buttons. */
export function TopBar({ nativeTitlebar, onClose, children }: { nativeTitlebar: boolean; onClose: () => void; children?: ReactNode }) {
  const status = useEngineStatus();
  return (
    <div data-tauri-drag-region className="flex h-10 shrink-0 items-stretch justify-end">
      <div data-tauri-drag-region className="flex flex-1 items-center px-4">
        {children}
      </div>
      <div data-tauri-drag-region className="flex items-center pr-2">
        <span className="pointer-events-none flex items-center gap-1.5 rounded-full border bg-card px-2.5 py-0.5 text-[11.5px] text-muted-foreground">
          <StatusDot tone={status.tone} />
          {status.label}
        </span>
      </div>
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

function FooterButton({ label, icon: Icon, onClick }: { label: string; icon: LucideIcon; onClick: () => void }) {
  return (
    <Button variant="ghost" size="icon" aria-label={label} title={label} onClick={onClick} className="text-muted-foreground hover:text-foreground">
      <Icon />
    </Button>
  );
}

export type Mode = "dictation" | "meetings";

const MODES: { value: Mode; label: string; icon: LucideIcon }[] = [
  { value: "dictation", label: "Dictation", icon: Mic },
  { value: "meetings", label: "Meetings", icon: Users },
];

/** Two icon buttons in one pill: the chosen mode sits raised on a lighter fill, the other lies flat. */
export function ModeToggle({ mode, onChange, recording }: { mode: Mode; onChange: (mode: Mode) => void; recording: boolean }) {
  return (
    <div role="radiogroup" aria-label="Mode" className="flex items-center gap-px rounded-[9px] border bg-muted p-[2px]">
      {MODES.map(({ value, label, icon: Icon }) => {
        const selected = value === mode;
        const hint = value === "meetings" && recording ? `${label} (recording now)` : label;
        return (
          <button
            key={value}
            role="radio"
            aria-checked={selected}
            aria-label={hint}
            title={hint}
            onClick={() => onChange(value)}
            className={cn(
              "relative grid h-[22px] w-7 place-items-center rounded-[7px] border border-transparent text-muted-foreground outline-none transition-colors duration-150 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/40 [&_svg]:size-3.5",
              selected && "border-border bg-card text-foreground shadow-[0_1px_2px_rgb(0_0_0/0.08)]",
            )}
          >
            <Icon />
            {value === "meetings" && recording ? (
              <span className="absolute top-[3px] right-[4px] size-[5px] animate-pulse rounded-full bg-destructive" />
            ) : null}
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
  const choose = (width: number) => {
    setWanted(width);
    saveSidebarWidth(width);
  };
  return [fitSidebar(wanted, windowWidth), choose];
}

/** The sidebar's right edge: drag it, or use the arrow keys, to widen it; double-click puts it back. */
function SidebarEdge({ width, onResize }: { width: number; onResize: (width: number) => void }) {
  const drag = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const edge = event.currentTarget;
    const from = event.clientX;
    const widthAt = (x: number) => fitSidebar(width + x - from, window.innerWidth);
    const move = (moved: globalThis.PointerEvent) => onResize(widthAt(moved.clientX));
    const stop = () => {
      edge.removeEventListener("pointermove", move);
      edge.removeEventListener("pointerup", stop);
      edge.removeEventListener("pointercancel", stop);
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
    onResize(fitSidebar(width + step, window.innerWidth));
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
      onDoubleClick={() => onResize(SIDEBAR.standard)}
      onKeyDown={nudge}
      className="absolute inset-y-0 -right-[3px] z-10 w-[6px] cursor-col-resize outline-none transition-colors duration-150 hover:bg-primary/40 focus-visible:bg-primary/60 active:bg-primary/60"
    />
  );
}

export function Sidebar({ items, mode, onMode, meetings }: { items: NavItem[]; mode: Mode; onMode: (mode: Mode) => void; meetings: ReactNode }) {
  const [width, setWidth] = useSidebarWidth();
  const status = useEngineStatus();
  const app = status.app;
  const { settings, discardSettings } = useStore();
  const client = useQueryClient();
  const [note, setNote] = useState("");

  const run = async (action: () => Promise<string>) => {
    try {
      setNote(await action());
    } catch (error) {
      setNote(errorText(error));
    }
  };
  const exportSettings = () =>
    run(async () => {
      const path = await chooseSavePath("Export settings", "skryb-settings.json", "Settings", "json");
      if (!path) return "";
      await api.post("/api/settings/export", { path });
      return "Settings exported.";
    });
  const importSettings = () =>
    run(async () => {
      const path = await chooseOpenPath("Import settings", "Settings", "json");
      if (!path) return "";
      await api.post("/api/settings/import", { path });
      discardSettings();
      await client.invalidateQueries({ queryKey: ["settings"] });
      return "Settings imported.";
    });

  return (
    <aside style={{ width }} className="relative flex shrink-0 flex-col border-r bg-sidebar">
      <SidebarEdge width={width} onResize={setWidth} />
      <div data-tauri-drag-region className="flex h-10 shrink-0 items-center gap-2 pr-2.5 pl-4">
        <span className="pointer-events-none flex">
          <Mark app={app} />
        </span>
        <span className="pointer-events-none text-[13px] font-semibold">Skryb</span>
        <span data-tauri-drag-region className="flex-1 self-stretch" />
        <ModeToggle mode={mode} onChange={onMode} recording={app?.meeting_recording ?? false} />
      </div>
      {mode === "meetings" ? meetings : <nav className="flex flex-col gap-0.5 px-2 pt-2">
        {items.map(({ path, label, icon: Icon }) => (
          <NavLink
            key={path}
            to={`/${path}`}
            draggable={false}
            className={({ isActive }) =>
              cn(
                "flex h-8 items-center gap-2.5 rounded-md px-2.5 text-[13px] text-muted-foreground transition-colors duration-150 hover:bg-accent hover:text-foreground",
                isActive && "bg-card font-medium text-foreground shadow-[0_0_0_1px_var(--border)] hover:bg-card",
              )
            }
          >
            <Icon className="size-4" />
            {label}
          </NavLink>
        ))}
      </nav>}
      <div className={cn("flex flex-col gap-2 p-3", mode === "dictation" && "mt-auto")}>
        {note ? <p className="px-1 text-[11.5px] text-muted-foreground">{note}</p> : null}
        <div className={cn("rounded-lg border bg-card px-3 py-2", mode === "meetings" && "hidden")}>
          <div className="flex items-center gap-1.5 text-[12px] font-medium">
            <StatusDot tone={status.tone} />
            {status.label}
          </div>
          <div className="mt-0.5 truncate text-[11.5px] text-muted-foreground" title={app?.engine.model}>
            {app?.engine.model || settings?.saved.model_size || "No model"}
          </div>
        </div>
        <div className="flex items-center justify-between pl-1">
          <span className="text-[11.5px] text-muted-foreground">{app ? `Version ${app.version}` : ""}</span>
          <span className="flex">
            <FooterButton label="Export settings" icon={FileDown} onClick={exportSettings} />
            <FooterButton label="Import settings" icon={FileUp} onClick={importSettings} />
            <FooterButton label="Open logs folder" icon={FolderOpen} onClick={() => run(async () => (await api.post("/api/logs/open"), ""))} />
          </span>
        </div>
      </div>
    </aside>
  );
}

/** Floats over the bottom of the page while settings differ from what the core has, or are being applied. */
export function SaveBar() {
  const { settings, saveSettings, discardSettings, savePhase, saveNote, saveError, dismissSaveError } = useStore();
  const dirty = isDirty(settings);
  if (!dirty && savePhase === "idle" && !saveError) return null;
  const busy = savePhase !== "idle";
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-4 z-20 flex justify-center px-6">
      <div role="status" className="pointer-events-auto flex w-full max-w-[720px] animate-[fade-in_150ms_ease-out] items-center gap-3 rounded-xl border bg-card py-2 pr-2 pl-4 shadow-lg">
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
    <div role="alert" className="absolute inset-x-0 top-10 z-30 flex items-center gap-3 border-b border-destructive/30 bg-card px-4 py-2.5 shadow-md">
      <CircleAlert className="size-4 shrink-0 text-destructive" />
      <span className="flex-1 text-[13px] font-medium">Skryb is not running</span>
      <span className="text-[12.5px] text-muted-foreground">Start it from the Start menu, then this window reconnects.</span>
      <Button variant="outline" onClick={onQuit}>
        Quit
      </Button>
    </div>
  );
}

export function Prompts() {
  const { prompt, closePrompt } = useStore();
  if (!prompt) return null;
  const answer = <T,>(resolve: (value: T) => void, value: T) => {
    closePrompt();
    resolve(value);
  };
  if (prompt.kind === "license") {
    return (
      <Dialog open onOpenChange={() => answer(prompt.resolve, false)}>
        <DialogContent className="w-[min(560px,calc(100vw-48px))]">
          <DialogTitle>{prompt.license.title}</DialogTitle>
          <DialogDescription>Read and accept the licence to download and use this model.</DialogDescription>
          <pre className="selectable max-h-[46vh] overflow-y-auto rounded-lg border bg-muted/50 p-3 font-sans text-[12.5px] whitespace-pre-wrap">{prompt.license.text}</pre>
          {prompt.license.url ? <p className="selectable -mt-2 truncate text-[12px] text-muted-foreground">Source: {prompt.license.url}</p> : null}
          <DialogFooter>
            <Button variant="ghost" onClick={() => answer(prompt.resolve, false)}>
              Cancel
            </Button>
            <Button onClick={() => answer(prompt.resolve, true)}>Accept</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }
  if (prompt.kind === "confirm") {
    return (
      <Dialog open onOpenChange={() => answer(prompt.resolve, false)}>
        <DialogContent>
          <DialogTitle>{prompt.title}</DialogTitle>
          <DialogDescription>{prompt.message}</DialogDescription>
          <DialogFooter>
            <Button variant="ghost" onClick={() => answer(prompt.resolve, false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={() => answer(prompt.resolve, true)}>
              {prompt.action}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }
  return (
    <Dialog open onOpenChange={() => answer(prompt.resolve, "cancel")}>
      <DialogContent>
        <DialogTitle>Save your changes?</DialogTitle>
        <DialogDescription>You have unsaved changes. Save them before closing, or discard them.</DialogDescription>
        <DialogFooter>
          <Button variant="ghost" onClick={() => answer(prompt.resolve, "cancel")}>
            Cancel
          </Button>
          <Button variant="outline" onClick={() => answer(prompt.resolve, "discard")}>
            Discard
          </Button>
          <Button onClick={() => answer(prompt.resolve, "save")}>Save</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

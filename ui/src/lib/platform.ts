import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { open, save } from "@tauri-apps/plugin-dialog";

export type Page = "settings" | "history" | "setup" | "updates" | "meetings";

export interface LaunchArgs {
  port: number;
  token: string;
  page: Page;
  native_titlebar: boolean;
  /** The page and scroll the last version's window showed when it handed over on updating. */
  restored?: { route: string; scroll: number } | null;
}

export const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const appWindow = inTauri ? getCurrentWindow() : null;

/** In a plain browser (the mock during development) the arguments come from the query string. */
export async function readLaunchArgs(): Promise<LaunchArgs> {
  if (inTauri) return invoke<LaunchArgs>("launch_args");
  const query = new URLSearchParams(location.search);
  return {
    port: Number(query.get("port") ?? 8765),
    token: query.get("token") ?? "dev",
    page: (query.get("page") ?? "settings") as Page,
    native_titlebar: true,
  };
}

/** Setup's progress file while an update installs, as written; null outside the app or with none. */
export async function readUpdateProgress(): Promise<string | null> {
  if (!inTauri) return null;
  return invoke<string | null>("update_progress");
}

/** Saves where this window is and what it shows for the new version, then closes it. */
export async function handOver(route: string, scroll: number): Promise<void> {
  if (inTauri) await invoke("hand_over", { route, scroll });
}

/** A second launch focuses this window and forwards its arguments. */
export function onRelaunch(handler: (args: LaunchArgs) => void): void {
  if (inTauri) void listen<LaunchArgs>("launch", (event) => handler(event.payload));
}

const beforeClose = new Set<() => Promise<unknown>>();

/** Work that must reach the core before the window goes: stopping the meter or a shortcut capture. */
export function onBeforeClose(task: () => Promise<unknown>): () => void {
  beforeClose.add(task);
  return () => beforeClose.delete(task);
}

export async function runBeforeClose(): Promise<void> {
  const timeout = new Promise((resolve) => setTimeout(resolve, 500));
  await Promise.race([Promise.allSettled([...beforeClose].map((task) => task())), timeout]);
}

export const windowControls = {
  minimize: () => void appWindow?.minimize(),
  toggleMaximize: () => void appWindow?.toggleMaximize(),
  close: () => (appWindow ? void appWindow.destroy() : window.close()),
};

/** Alt+F4 and the taskbar close ask first; `ask` returns true when the window may close. */
export function guardClose(ask: () => Promise<boolean>): void {
  void appWindow?.onCloseRequested(async (event) => {
    event.preventDefault();
    if (await ask()) {
      await runBeforeClose();
      windowControls.close();
    }
  });
}

export async function chooseSavePath(title: string, defaultPath: string, name: string, extension: string): Promise<string | null> {
  if (!inTauri) return window.prompt(title, defaultPath);
  return save({ title, defaultPath, filters: [{ name, extensions: [extension] }] });
}

export async function chooseOpenPath(title: string, name: string, extension: string): Promise<string | null> {
  if (!inTauri) return window.prompt(title, "");
  const chosen = await open({ title, multiple: false, directory: false, filters: [{ name, extensions: [extension] }] });
  return typeof chosen === "string" ? chosen : null;
}

export async function chooseFolder(title: string, defaultPath: string): Promise<string | null> {
  if (!inTauri) return window.prompt(title, defaultPath);
  const chosen = await open({ title, defaultPath, multiple: false, directory: true });
  return typeof chosen === "string" ? chosen : null;
}

/** A video or audio file picked in Windows' open dialog; outside the app, a typed path. */
export async function chooseMediaFile(extensions: string[]): Promise<string | null> {
  if (!inTauri) return window.prompt("Path of a video or audio file", "C:\\Videos\\meeting.mp4");
  const chosen = await open({
    title: "Choose a video or audio file",
    multiple: false,
    directory: false,
    filters: [{ name: "Video and audio", extensions }, { name: "All files", extensions: ["*"] }],
  });
  return typeof chosen === "string" ? chosen : null;
}

export type FileDrag = { type: "over" } | { type: "leave" } | { type: "drop"; paths: string[] };

/** Files dragged over and dropped on the window. The app hands over their paths; a browser has none to give. */
export function onFileDrag(handler: (drag: FileDrag) => void): () => void {
  if (!inTauri) return () => undefined;
  let stop: (() => void) | null = null;
  let stopped = false;
  void getCurrentWebview()
    .onDragDropEvent((event) => {
      const payload = event.payload;
      if (payload.type === "drop") handler({ type: "drop", paths: payload.paths });
      else if (payload.type === "leave") handler({ type: "leave" });
      else handler({ type: "over" });
    })
    .then((unlisten) => {
      if (stopped) unlisten();
      else stop = unlisten;
    });
  return () => {
    stopped = true;
    stop?.();
  };
}

export async function copyText(text: string): Promise<void> {
  await navigator.clipboard.writeText(text);
}

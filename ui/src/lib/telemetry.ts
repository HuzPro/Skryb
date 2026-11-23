import { api } from "./api";
import { onBeforeClose } from "./platform";
import { queries, queryClient } from "./queries";
import type { Settings } from "./types";

/**
 * Opt-in usage data: which pages were opened, which controls were used and which settings were saved.
 * Names come from the code (a route, an icon, a `data-track` attribute, a setting's key), never from
 * anything typed, said or shown; the core checks every event again before keeping it.
 */
export interface UsageEvent {
  kind: "page" | "click" | "setting";
  name: string;
  value?: boolean | number | string;
  ms?: number;
}

/** What keeping usage data means, said the same way wherever it is offered. */
export const USAGE_DATA_PROMISE =
  "Which pages, buttons and settings you use, so Skryb can put what you use most within reach. Never what you say, your words, names or folders.";

/** A batch goes to the core this long after its first event, or as soon as it holds FLUSH_AT. */
const FLUSH_MS = 5000;
const FLUSH_AT = 50;
/** Settings whose value is one of a fixed set of words, so it may be recorded; any other text is not. */
const CHOICES = new Set(["model_size", "language", "device", "engine", "output_mode", "transcription_mode", "theme", "colourway", "meetings.language", "recording.duck_other_audio"]);
const CONTROLS = "[data-track],button,a,[role=option],[role=radio],[role=switch],[role=tab],[role=menuitem]";

let enabled = false;
let queue: (UsageEvent & { at: number })[] = [];
let timer: ReturnType<typeof setTimeout> | undefined;
let page = "";
let pageSince = 0;

/** "/history" is "history"; a saved meeting is "meetings.saved" whichever meeting it is. */
export function pageName(pathname: string): string {
  const [first = "", ...rest] = pathname.split("/").filter(Boolean);
  return rest.length ? `${first}.saved` : first;
}

/** What a control is called: its `data-track`, else its icon, else its role or tag. */
export function controlName(tracked: string | null, iconClasses: string[], role: string | null, tag: string): string {
  if (tracked) return tracked;
  const icon = iconClasses.find((name) => name.startsWith("lucide-"));
  return icon ? icon.slice("lucide-".length) : (role ?? tag.toLowerCase());
}

function flat(settings: Settings): Record<string, unknown> {
  const { colourways, recording, meetings, ...rest } = settings;
  const nested = (prefix: string, group: object) => Object.entries(group).map(([key, value]) => [`${prefix}.${key}`, value] as const);
  return { ...rest, colourway: colourways[settings.theme], ...Object.fromEntries([...nested("recording", recording), ...nested("meetings", meetings)]) };
}

/** One event for each setting that differs between what was saved and what is being saved now. */
export function settingEvents(saved: Settings, next: Settings): UsageEvent[] {
  const before = flat(saved);
  return Object.entries(flat(next))
    .filter(([key, value]) => JSON.stringify(value) !== JSON.stringify(before[key]))
    .map(([name, value]) => {
      const plain = typeof value === "boolean" || typeof value === "number" || (typeof value === "string" && CHOICES.has(name));
      return plain ? { kind: "setting", name, value: value as boolean | number | string } : { kind: "setting", name };
    });
}

/**
 * *items* with the most used first, once together they have been used at least *least* times; until
 * then, and between equally used ones, the order they were given in.
 */
export function byUse<T>(items: T[], counts: Record<string, number>, key: (item: T) => string, least = 20): T[] {
  const used = (item: T) => counts[key(item)] ?? 0;
  if (items.reduce((total, item) => total + used(item), 0) < least) return items;
  return items
    .map((item, at) => ({ item, at }))
    .sort((a, b) => used(b.item) - used(a.item) || a.at - b.at)
    .map(({ item }) => item);
}

async function flush(): Promise<void> {
  clearTimeout(timer);
  timer = undefined;
  const sending = queue;
  queue = [];
  if (!sending.length) return;
  const now = Date.now();
  // Usage data is never worth an error message: a batch the core doesn't take is dropped.
  // Once it's kept, the count Settings shows is asked for again, if it's on screen.
  await api.post("/api/telemetry", { events: sending.map(({ at, ...event }) => ({ ...event, ago: now - at })) }).then(
    () => void queryClient.invalidateQueries({ queryKey: queries.telemetry.queryKey, exact: true }),
    () => undefined,
  );
}

/** Deletes everything kept; what Settings shows of it is asked for again. */
export async function deleteUsageData(): Promise<void> {
  await api.del("/api/telemetry");
  await queryClient.invalidateQueries({ queryKey: queries.telemetry.queryKey });
}

export function track(event: UsageEvent): void {
  if (!enabled) return;
  queue.push({ ...event, at: Date.now() });
  if (queue.length >= FLUSH_AT) void flush();
  else timer ??= setTimeout(() => void flush(), FLUSH_MS);
}

function leavePage(): void {
  if (page && pageSince) track({ kind: "page", name: page, ms: Date.now() - pageSince });
  pageSince = 0;
}

/** The page now showing; the one before it is recorded with the time spent on it. */
export function openedPage(pathname: string): void {
  leavePage();
  page = pageName(pathname);
  pageSince = Date.now();
}

export function setTelemetry(on: boolean): void {
  enabled = on;
  if (!on) queue = [];
}

/** Listens for clicks and for the window hiding or closing; returns how to stop. */
export function watchUsage(): () => void {
  const clicked = (event: MouseEvent) => {
    const control = event.target instanceof Element ? event.target.closest(CONTROLS) : null;
    if (!control || !page) return;
    const icon = control.querySelector("svg.lucide");
    track({ kind: "click", name: `${page}.${controlName(control.getAttribute("data-track"), icon ? [...icon.classList] : [], control.getAttribute("role"), control.tagName)}` });
  };
  // Time on a page stops counting while the window is hidden.
  const shown = () => {
    if (document.hidden) {
      leavePage();
      void flush();
    } else if (page) pageSince = Date.now();
  };
  const forget = onBeforeClose(async () => {
    leavePage();
    await flush();
  });
  document.addEventListener("click", clicked, true);
  document.addEventListener("visibilitychange", shown);
  return () => {
    forget();
    document.removeEventListener("click", clicked, true);
    document.removeEventListener("visibilitychange", shown);
  };
}

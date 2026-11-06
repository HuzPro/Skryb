import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Copy, Download, FileDown, Play, RotateCcw, Search, Square, Trash2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { Interlinear } from "../components/english";
import { Card, PAGE_EDGE, PageTitle } from "../components/parts";
import { useAppInfo } from "../components/shell";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Select } from "../components/ui/select";
import { Switch } from "../components/ui/switch";
import { api, errorText } from "../lib/api";
import { hasArabic } from "../lib/english";
import { chooseFolder, chooseSavePath, copyText } from "../lib/platform";
import { HISTORY_PAGE, queries } from "../lib/queries";
import { useServerEvent } from "../lib/sse";
import { useStore } from "../lib/store";
import type { EnglishLine, HistoryEntry } from "../lib/types";
import { cn } from "../lib/utils";

const RETENTION_DAYS = [0, 7, 30, 90];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const DAY_HEIGHT = 30;
const ROW_HEIGHT = 58;
// Each listed entry costs the core a disk check for its recording, so the
// list asks for this many more whenever the last LOAD_AHEAD rows come into view.
const PAGE_SIZE = HISTORY_PAGE;
const LOAD_AHEAD = 20;

type Item<T> = { kind: "day"; label: string } | { kind: "entry"; entry: T };
type Retry = { state: "running" } | { state: "done"; text: string } | { state: "failed"; error: string };

/** `at` is ISO local time; read its parts directly so the Windows date format never reorders them. */
function parts(at: string): { date: string; time: string; day: Date } {
  const [date = "", rest = ""] = at.split("T");
  const [year = 0, month = 1, dayOfMonth = 1] = date.split("-").map(Number);
  return { date, time: rest.slice(0, 5), day: new Date(year, month - 1, dayOfMonth) };
}

export function dayLabel(at: string, today: Date): string {
  const { day } = parts(at);
  const start = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  const daysAgo = Math.round((start.getTime() - day.getTime()) / 86_400_000);
  if (daysAgo === 0) return "Today";
  if (daysAgo === 1) return "Yesterday";
  return `${WEEKDAYS[day.getDay()]} ${day.getDate()} ${MONTHS[day.getMonth()]} ${day.getFullYear()}`;
}

export function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = String(total % 60).padStart(2, "0");
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${secs}` : `${minutes}:${secs}`;
}

/** Mirrors `HistoryEntry.shown_text`: what was typed when that was saved, else what the engine heard. */
export function finalText(entry: HistoryEntry): string {
  return entry.typed || entry.text;
}

/** Words as the Home numbers count them: split on whitespace. */
export function wordCount(text: string): number {
  return text.split(/\s+/).filter(Boolean).length;
}

/** Mirrors `history.entries.preview`: one line, and a hint when there is no text. */
export function preview(entry: HistoryEntry): string {
  const text = finalText(entry).split(/\s+/).filter(Boolean).join(" ");
  if (text) return text;
  if (entry.error) return `(${entry.error})`;
  return entry.status === "Cancelled" ? "(cancelled - not transcribed)" : "(no text)";
}

/** Newest first, as given, with a heading wherever the day changes; `at` reads each entry's ISO local time. */
export function groupByDay<T>(entries: T[], today: Date, at: (entry: T) => string): Item<T>[] {
  const items: Item<T>[] = [];
  let last = "";
  for (const entry of entries) {
    const label = dayLabel(at(entry), today);
    if (label !== last) items.push({ kind: "day", label });
    last = label;
    items.push({ kind: "entry", entry });
  }
  return items;
}

export function useDebounced(value: string, ms: number): string {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), ms);
    return () => clearTimeout(timer);
  }, [value, ms]);
  return debounced;
}

function statusVariant(status: HistoryEntry["status"]): "destructive" | "default" {
  return status === "Failed" ? "destructive" : "default";
}

/** A dictation with Arabic in it: each Arabic line word by word over its English, asked of the core once per text. */
function EnglishText({ text }: { text: string }) {
  const english = useQuery({
    queryKey: ["english", text],
    queryFn: () => api.post<{ lines: (EnglishLine | null)[] }>("/api/translation", { text }),
    retry: false,
  });
  return (
    <div className="selectable">
      {text.split("\n").map((line, index) => {
        const annotated = english.data?.lines[index];
        return annotated ? (
          <Interlinear key={index} line={annotated} className="mb-3 last:mb-0" />
        ) : (
          <p key={index} dir="auto" className="mb-1 whitespace-pre-wrap">
            {line || "\u00a0"}
          </p>
        );
      })}
      {english.isPending ? <p className="mt-2 text-[12.5px] text-muted-foreground">Translating…</p> : null}
      {english.isError ? <p className="mt-2 text-[12.5px] text-destructive">{errorText(english.error)}</p> : null}
    </div>
  );
}

function HistorySettings() {
  const { settings, editSettings, confirm } = useStore();
  const app = useAppInfo();
  const [note, setNote] = useState("");
  useServerEvent((name, data) => {
    if (name !== "history") return;
    const event = data as { event?: string; error?: string };
    if (event.event === "deleted_all") setNote("");
    if (event.event === "changed" && event.error) setNote(`Could not delete everything: ${event.error}`);
  });
  const draft = settings?.draft;
  if (!draft) return <div className="h-8" />;
  const days = draft.history_retention_days;
  const choices = [...new Set([...RETENTION_DAYS, Math.max(days, 0)])].sort((a, b) => a - b);
  const changeFolder = async (current: string) => {
    const path = await chooseFolder("Choose where logs and history are kept", current);
    if (!path) return;
    try {
      await api.put("/api/logs/directory", { path });
      setNote(`Moves to ${path} on the next start.`);
    } catch (error) {
      setNote(errorText(error));
    }
  };
  const deleteAll = async () => {
    if (!(await confirm("Delete all history", "Every dictation's text and recording goes. This can't be undone.", "Delete all"))) return;
    try {
      await api.del("/api/history");
      setNote("Deleting...");
    } catch (error) {
      setNote(errorText(error));
    }
  };
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-1 text-[12.5px]">
      <label className="flex items-center gap-2.5 font-medium">
        <Switch checked={draft.capture_content} onCheckedChange={(on) => editSettings((current) => ({ ...current, capture_content: on }))} />
        Save history
      </label>
      <label className="flex items-center gap-2.5 font-medium" title="Arabic dictations word by word, with the English under each word and the whole line in English.">
        <Switch checked={draft.show_english} onCheckedChange={(on) => editSettings((current) => ({ ...current, show_english: on }))} />
        Show English
      </label>
      <label className="flex items-center gap-2 text-muted-foreground">
        Keep for
        <Select
          label="Keep history for"
          className="w-28 text-foreground"
          value={String(days)}
          options={choices.map((choice) => ({ value: String(choice), label: choice ? `${choice} days` : "Forever" }))}
          onChange={(value) => editSettings((current) => ({ ...current, history_retention_days: Number(value) }))}
        />
      </label>
      {app?.diagnostics_directory ? (
        <span className="flex min-w-0 items-center gap-1 text-muted-foreground" title="Use a local folder that isn't synced or shared.">
          Saved in <span className="selectable min-w-0 truncate text-foreground">{app.diagnostics_directory}</span>
          <Button variant="link" className="text-[12.5px]" onClick={() => void changeFolder(app.diagnostics_directory)}>
            Change
          </Button>
          <Button variant="link" className="text-[12.5px]" onClick={() => void api.post("/api/logs/open").catch((error: unknown) => setNote(errorText(error)))}>
            Open
          </Button>
        </span>
      ) : null}
      <span className="ml-auto text-muted-foreground">{note}</span>
      <Button variant="ghost" size="sm" className="text-destructive hover:text-destructive" onClick={() => void deleteAll()}>
        <Trash2 />
        Delete all
      </Button>
    </div>
  );
}

export function HistoryPage() {
  const { settings, confirm } = useStore();
  const client = useQueryClient();
  const [query, setQuery] = useState("");
  const search = useDebounced(query.trim(), 150);
  const [shown, setShown] = useState({ search, limit: PAGE_SIZE });
  const limit = shown.search === search ? shown.limit : PAGE_SIZE;
  const history = useQuery({ ...queries.history(search, limit), placeholderData: keepPreviousData });
  const entries = useMemo(() => history.data ?? [], [history.data]);
  // Fewer than asked for means the end; previous data shown while more load is always fewer.
  const hasMore = entries.length === limit;
  const items = useMemo(() => groupByDay(entries, new Date(), (entry) => entry.at), [entries]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [retries, setRetries] = useState<Record<string, Retry>>({});
  const [playing, setPlaying] = useState<string | null>(null);
  const [note, setNote] = useState<{ text: string; error: boolean }>({ text: "", error: false });
  const [exportNote, setExportNote] = useState<{ text: string; error: boolean }>({ text: "", error: false });
  const listRef = useRef<HTMLDivElement>(null);
  const playTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => listRef.current,
    estimateSize: (index) => (items[index]?.kind === "day" ? DAY_HEIGHT : ROW_HEIGHT),
    overscan: 10,
  });
  const lastRendered = virtualizer.getVirtualItems().at(-1)?.index ?? 0;
  useEffect(() => {
    if (hasMore && lastRendered >= items.length - LOAD_AHEAD) setShown({ search, limit: limit + PAGE_SIZE });
  }, [hasMore, lastRendered, items.length, search, limit]);

  // The newest dictation shows until another is picked, and when the picked one is no longer listed.
  const selected = entries.find((entry) => entry.id === selectedId) ?? entries[0] ?? null;
  const retry = selected ? retries[selected.id] : undefined;

  useServerEvent((name, data) => {
    if (name !== "history") return;
    const event = data as { event: string; id?: string; text?: string; error?: string };
    if (event.event === "retry_done" && event.id) setRetries((all) => ({ ...all, [event.id as string]: { state: "done", text: event.text ?? "" } }));
    if (event.event === "retry_failed" && event.id) setRetries((all) => ({ ...all, [event.id as string]: { state: "failed", error: event.error ?? "" } }));
  });
  useEffect(() => () => clearTimeout(playTimer.current), []);

  const select = (id: string | null) => {
    setSelectedId(id);
    setNote({ text: "", error: false });
  };
  const act = async (work: () => Promise<string>) => {
    try {
      setNote({ text: await work(), error: false });
    } catch (error) {
      setNote({ text: errorText(error), error: true });
    }
  };

  const shownText = retry?.state === "done" ? retry.text : retry?.state === "running" ? "Transcribing…" : selected ? finalText(selected) : "";
  const statusNote =
    note.text ||
    (retry?.state === "done"
      ? "New transcription, not saved."
      : retry?.state === "failed"
        ? `Transcription failed: ${retry.error}`
        : selected?.error
          ? `Failed: ${selected.error}`
          : selected && !selected.has_audio && (selected.status === "Cancelled" || !selected.text)
            ? "No recording was saved."
            : "");
  const noteIsError = note.text ? note.error : retry?.state === "failed" || Boolean(selected?.error && !retry);

  const actions = selected && {
    copy: () => act(async () => (await copyText(shownText), "Copied.")),
    play: () =>
      act(async () => {
        clearTimeout(playTimer.current);
        if (playing === selected.id) {
          await api.post(`/api/history/${encodeURIComponent(selected.id)}/stop`);
          setPlaying(null);
          return "";
        }
        await api.post(`/api/history/${encodeURIComponent(selected.id)}/play`);
        setPlaying(selected.id);
        playTimer.current = setTimeout(() => setPlaying(null), selected.audio_seconds * 1000 + 250);
        return "";
      }),
    retry: () =>
      act(async () => {
        setRetries((all) => ({ ...all, [selected.id]: { state: "running" } }));
        try {
          await api.post(`/api/history/${encodeURIComponent(selected.id)}/retry`);
        } catch (error) {
          setRetries((all) => ({ ...all, [selected.id]: { state: "failed", error: errorText(error) } }));
        }
        return "";
      }),
    save: () =>
      act(async () => {
        const path = await chooseSavePath("Save audio as", `dictation-${parts(selected.at).date}-${parts(selected.at).time.replace(":", "")}.wav`, "WAV audio", "wav");
        if (!path) return "";
        await api.post(`/api/history/${encodeURIComponent(selected.id)}/export`, { path });
        return `Saved ${path.split(/[\\/]/).at(-1)}.`;
      }),
    remove: () =>
      act(async () => {
        if (!(await confirm("Delete dictation", "Delete this dictation's text and recording from history?", "Delete"))) return "";
        await api.del(`/api/history/${encodeURIComponent(selected.id)}`);
        const next = entries[entries.indexOf(selected) + 1] ?? entries[entries.indexOf(selected) - 1];
        select(next?.id ?? null);
        await client.invalidateQueries({ queryKey: ["history"] });
        return "";
      }),
  };

  const moveSelection = (event: KeyboardEvent) => {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    const at = selected ? entries.indexOf(selected) : -1;
    const next = entries[Math.min(Math.max(at + (event.key === "ArrowDown" ? 1 : -1), 0), entries.length - 1)];
    if (!next) return;
    select(next.id);
    const itemIndex = items.findIndex((item) => item.kind === "entry" && item.entry.id === next.id);
    virtualizer.scrollToIndex(itemIndex, { align: "auto" });
  };

  const exportAll = async () => {
    try {
      const path = await chooseSavePath("Export history", `skryb-history-${new Date().toISOString().slice(0, 10)}.csv`, "Spreadsheet (CSV)", "csv");
      if (!path) return;
      const { rows } = await api.post<{ rows: number }>(`/api/history/export?query=${encodeURIComponent(search)}`, { path });
      setExportNote({ text: `Exported ${rows} ${rows === 1 ? "dictation" : "dictations"}.`, error: false });
    } catch (error) {
      setExportNote({ text: errorText(error), error: true });
    }
  };

  const emptyText = history.isPending
    ? "Loading…"
    : history.isError
      ? "History could not be loaded."
      : search
        ? "No dictations match your search."
        : settings && !settings.saved.capture_content
          ? "History is off. Turn on Save history below."
          : "No dictations yet. Press your shortcut and speak.";

  return (
    <div className={cn(PAGE_EDGE, "flex h-full flex-col gap-3 pb-5")}>
      <PageTitle title="History">
        <div className="flex items-center gap-2">
          {exportNote.text ? <span className={cn("text-[12.5px]", exportNote.error ? "text-destructive" : "text-muted-foreground")}>{exportNote.text}</span> : null}
          <Button variant="outline" size="sm" disabled={!entries.length} onClick={() => void exportAll()}>
            <FileDown />
            {search ? "Export results" : "Export"}
          </Button>
          <div className="relative w-72">
            <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input aria-label="Search history" placeholder="Search dictations" className="pl-8" value={query} onChange={(event) => setQuery(event.target.value)} />
          </div>
        </div>
      </PageTitle>
      {/* Fixed proportions: what an entry holds must never move the split. */}
      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,3fr)_minmax(0,2fr)] grid-rows-[minmax(0,1fr)] gap-3">
        <Card className="relative min-w-0 overflow-hidden">
          <div ref={listRef} role="listbox" aria-label="Dictations" tabIndex={0} onKeyDown={moveSelection} className="h-full overflow-y-auto outline-none">
            <div className="relative w-full" style={{ height: virtualizer.getTotalSize() }}>
              {virtualizer.getVirtualItems().map((row) => {
                const item = items[row.index];
                if (!item) return null;
                const style = { transform: `translateY(${row.start}px)`, height: row.size };
                if (item.kind === "day") {
                  return (
                    <div key={row.key} style={style} className="absolute inset-x-0 top-0 flex items-end px-4 pb-1 text-[11.5px] font-semibold text-muted-foreground">
                      {item.label}
                    </div>
                  );
                }
                const { entry } = item;
                const active = entry.id === selected?.id;
                const blank = !finalText(entry).trim();
                return (
                  <div
                    key={row.key}
                    role="option"
                    aria-selected={active}
                    style={style}
                    onClick={() => select(entry.id)}
                    className={cn(
                      "absolute inset-x-1.5 top-0 flex flex-col justify-center rounded-lg px-3 transition-colors hover:bg-accent/60",
                      active && "bg-selected shadow-[inset_3px_0_0_var(--primary)] hover:bg-selected",
                    )}
                  >
                    <span dir="auto" className={cn("truncate text-[13px]", blank && "text-muted-foreground italic")}>{preview(entry)}</span>
                    <span className="mt-0.5 flex items-center gap-1.5 text-[11.5px] text-muted-foreground tabular-nums">
                      {parts(entry.at).time} · {formatDuration(entry.audio_seconds)}
                      {blank ? null : ` · ${wordCount(finalText(entry)).toLocaleString()} words`}
                      {entry.status ? <Badge variant={statusVariant(entry.status)}>{entry.status}</Badge> : null}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
          {!entries.length ? <p className="absolute inset-0 grid place-items-center p-6 text-center text-[12.5px] text-muted-foreground">{emptyText}</p> : null}
        </Card>
        <Card className="flex min-w-0 flex-col">
          {selected && actions ? (
            <>
              <div className="border-b px-5 pt-4 pb-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-[15px] font-semibold">
                      {dayLabel(selected.at, new Date())}, {parts(selected.at).time}
                    </div>
                    <div className="mt-0.5 flex items-center gap-2 text-[12px] text-muted-foreground tabular-nums">
                      {formatDuration(selected.audio_seconds)}
                      {shownText.trim() ? ` · ${wordCount(shownText).toLocaleString()} words` : null}
                      {selected.status ? <Badge variant={statusVariant(selected.status)}>{selected.status}</Badge> : null}
                    </div>
                  </div>
                  <Button variant="ghost" size="icon" aria-label="Delete" title="Delete" className="text-muted-foreground hover:text-destructive" onClick={() => void actions.remove()}>
                    <Trash2 />
                  </Button>
                </div>
                <div className="mt-3 flex flex-wrap gap-1.5">
                  <Button variant="outline" size="sm" disabled={!shownText.trim() || retry?.state === "running"} onClick={() => void actions.copy()}>
                    <Copy />
                    Copy
                  </Button>
                  <Button variant="outline" size="sm" disabled={!selected.has_audio} onClick={() => void actions.play()}>
                    {playing === selected.id ? <Square /> : <Play />}
                    {playing === selected.id ? "Stop" : "Play"}
                  </Button>
                  <Button variant="outline" size="sm" disabled={!selected.has_audio || retry?.state === "running"} onClick={() => void actions.retry()}>
                    <RotateCcw />
                    {selected.status === "Cancelled" ? "Transcribe" : "Retry"}
                  </Button>
                  <Button variant="outline" size="sm" disabled={!selected.has_audio} onClick={() => void actions.save()}>
                    <Download />
                    Save audio
                  </Button>
                </div>
              </div>
              {statusNote ? <p className={cn("px-5 pt-3 text-[12.5px]", noteIsError ? "selectable text-destructive" : "text-muted-foreground")}>{statusNote}</p> : null}
              <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4 text-[14px] leading-relaxed">
                {settings?.saved.show_english && hasArabic(shownText) ? (
                  <EnglishText text={shownText} />
                ) : (
                  <div dir="auto" className="selectable whitespace-pre-wrap">{shownText || <span className="text-muted-foreground italic">No text.</span>}</div>
                )}
              </div>
            </>
          ) : (
            <p className="m-auto px-6 text-center text-[12.5px] text-muted-foreground">Pick a dictation to read it.</p>
          )}
        </Card>
      </div>
      <HistorySettings />
    </div>
  );
}

import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Copy, Download, FileDown, Play, RotateCcw, Search, SlidersHorizontal, Square, Trash2, type LucideIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { Interlinear } from "../components/english";
import { Card, PAGE_TITLE, Row, TEXT_INSET } from "../components/parts";
import { ResizeEdge, saveWidth, savedWidth, useAppInfo } from "../components/shell";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogTitle } from "../components/ui/dialog";
import { Input } from "../components/ui/input";
import { Select } from "../components/ui/select";
import { Switch } from "../components/ui/switch";
import { api, errorText } from "../lib/api";
import { hasArabic } from "../lib/english";
import { chooseFolder, chooseSavePath, copyText } from "../lib/platform";
import { HISTORY_PAGE, queries } from "../lib/queries";
import { useServerEvent } from "../lib/sse";
import { useStore } from "../lib/store";
import { byUse } from "../lib/telemetry";
import type { EnglishLine, HistoryEntry } from "../lib/types";
import { cn } from "../lib/utils";

const RETENTION_DAYS = [0, 7, 30, 90];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const DAY_HEIGHT = 30;
const ROW_HEIGHT = 52;
// Each listed entry costs the core a disk check for its recording, so the list opens on PAGE_SIZE and asks
// for MORE whenever the last LOAD_AHEAD rows come into view. Show all asks for every one at once.
const PAGE_SIZE = HISTORY_PAGE;
const MORE = 100;
const LOAD_AHEAD = 8;
const ALL = 1_000_000;
/** The list beside the reader: as wide as it was last dragged, within these, and never leaving the reader less than its least. */
export const LIST = { standard: 340, least: 240, most: 900 };
const READER_LEAST = 320;
// Narrow, with a grip in it, so the two panes read as one thing split in two that the split can be dragged.
const PANE_GAP = 6;
const LIST_WIDTH = "history-list-width";

/** A list width that fits *room*, the width both panes share: the reader keeps its least before the list gets its wish. */
export function fitList(width: number, room: number): number {
  return Math.round(Math.max(LIST.least, Math.min(width, LIST.most, room - READER_LEAST - PANE_GAP)));
}

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

/** "1 word", "15 words", "1,912 words". */
export function words(text: string): string {
  const count = wordCount(text);
  return `${count.toLocaleString()} ${count === 1 ? "word" : "words"}`;
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

/** What History keeps, for how long and where, and deleting all of it; opened from the page's settings button. */
function HistorySettings({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { settings, editSettings, confirm } = useStore();
  const app = useAppInfo();
  const [note, setNote] = useState({ text: "", error: false });
  useServerEvent((name, data) => {
    if (name !== "history") return;
    const event = data as { event?: string; error?: string };
    if (event.event === "deleted_all") setNote({ text: "", error: false });
    if (event.event === "changed" && event.error) setNote({ text: `Could not delete everything: ${event.error}`, error: true });
  });
  const draft = settings?.draft;
  if (!draft) return null;
  const days = draft.history_retention_days;
  const choices = [...new Set([...RETENTION_DAYS, Math.max(days, 0)])].sort((a, b) => a - b);
  const failed = (error: unknown) => setNote({ text: errorText(error), error: true });
  const changeFolder = async (current: string) => {
    const path = await chooseFolder("Choose where logs and history are kept", current);
    if (!path) return;
    try {
      await api.put("/api/logs/directory", { path });
      setNote({ text: `Moves to ${path} on the next start.`, error: false });
    } catch (error) {
      failed(error);
    }
  };
  const deleteAll = async () => {
    if (!(await confirm("Delete all history", "Every dictation's text and recording goes. This can't be undone.", "Delete all"))) return;
    try {
      await api.del("/api/history");
      setNote({ text: "Deleting...", error: false });
    } catch (error) {
      failed(error);
    }
  };
  return (
    <Dialog open={open} onOpenChange={(next) => (next ? undefined : onClose())}>
      <DialogContent className="w-[min(520px,calc(100vw-48px))]">
        <DialogTitle>History settings</DialogTitle>
        <div className="-mx-4 -my-2 divide-y overflow-y-auto">
          <Row title="Save history" description="Keeps each dictation's text and recording on this computer.">
            <Switch aria-label="Save history" checked={draft.capture_content} onCheckedChange={(on) => editSettings((current) => ({ ...current, capture_content: on }))} />
          </Row>
          <Row
            title="Capture my edits"
            description="After you dictate into a window, keeps the keys you type there until you send, to learn what you corrected. Saved with that window's title, beside your history. Needs Save history, and starts the next time Skryb starts."
          >
            <Switch
              aria-label="Capture my edits"
              disabled={!draft.capture_content}
              checked={draft.capture_content && draft.capture_edits}
              onCheckedChange={(on) => editSettings((current) => ({ ...current, capture_edits: on }))}
            />
          </Row>
          <Row title="Show English" description="Arabic dictations word by word, with the English under each word.">
            <Switch aria-label="Show English" checked={draft.show_english} onCheckedChange={(on) => editSettings((current) => ({ ...current, show_english: on }))} />
          </Row>
          <Row title="Keep for">
            <Select
              label="Keep history for"
              className="w-28"
              value={String(days)}
              options={choices.map((choice) => ({ value: String(choice), label: choice ? `${choice} days` : "Forever" }))}
              onChange={(value) => editSettings((current) => ({ ...current, history_retention_days: Number(value) }))}
            />
          </Row>
          {app?.diagnostics_directory ? (
            <Row title="Saved in" description={<span className="selectable break-all" title="Use a local folder that isn't synced or shared.">{app.diagnostics_directory}</span>}>
              <Button variant="outline" data-track="change-folder" onClick={() => void changeFolder(app.diagnostics_directory)}>
                Change
              </Button>
              <Button variant="outline" data-track="open-folder" onClick={() => void api.post("/api/logs/open").catch(failed)}>
                Open
              </Button>
            </Row>
          ) : null}
          <Row title="Delete all history" description="Every dictation's text and recording." error={note.error ? note.text : undefined}>
            <Button variant="destructive" data-track="delete-all" onClick={() => void deleteAll()}>
              <Trash2 />
              Delete all
            </Button>
          </Row>
        </div>
        <DialogFooter className="items-center">
          <DialogDescription className="mr-auto text-[12.5px]">{note.error ? "" : note.text || "Changes take effect once you save them."}</DialogDescription>
          <Button data-track="settings-done" onClick={onClose}>
            Done
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
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
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [listWidth, setListWidth] = useState(() => savedWidth(LIST_WIDTH, LIST.standard));
  // Fetched once a session, so the actions keep their places while the window is open.
  const used = useQuery(queries.usageOrder).data?.counts ?? {};
  const panesRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const playTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => listRef.current,
    estimateSize: (index) => (items[index]?.kind === "day" ? DAY_HEIGHT : ROW_HEIGHT),
    overscan: 10,
  });
  // The last row in view, not the last one drawn: rows drawn ahead of the scroll must not fetch the next page on their own.
  const lastInView = virtualizer.range?.endIndex ?? 0;
  useEffect(() => {
    if (hasMore && lastInView >= items.length - LOAD_AHEAD) setShown({ search, limit: limit + MORE });
  }, [hasMore, lastInView, items.length, search, limit]);

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
          ? "History is off. Turn on Save history in History settings."
          : "No dictations yet. Press your shortcut and speak.";

  const isPlaying = playing === selected?.id;
  const retrying = retry?.state === "running";
  // The most used comes first and is the filled button; `name` is what usage data calls it.
  const buttons: { name: string; label: string; icon: LucideIcon; disabled: boolean; run: () => Promise<void> }[] =
    selected && actions
      ? byUse(
          [
            { name: "copy", label: "Copy", icon: Copy, disabled: !shownText.trim() || retrying, run: actions.copy },
            { name: "play", label: isPlaying ? "Stop" : "Play", icon: isPlaying ? Square : Play, disabled: !selected.has_audio, run: actions.play },
            { name: "retry", label: selected.status === "Cancelled" ? "Transcribe" : "Retry", icon: RotateCcw, disabled: !selected.has_audio || retrying, run: actions.retry },
            { name: "save-audio", label: "Save audio", icon: Download, disabled: !selected.has_audio, run: actions.save },
          ],
          used,
          (button) => `click:history.${button.name}`,
        )
      : [];

  return (
    <div className="flex h-full flex-col gap-3 px-3 pt-3 pb-3">
      {/* The title starts where the list's text does; the cards sit outside that line. */}
      <div className={cn(TEXT_INSET, "mb-1 flex min-h-10 items-center gap-2")}>
        <h1 className={cn(PAGE_TITLE, "flex-1")}>History</h1>
        {settings && !settings.saved.capture_content ? <Badge>Not saving</Badge> : null}
        {exportNote.text ? <span className={cn("text-[12.5px]", exportNote.error ? "text-destructive" : "text-muted-foreground")}>{exportNote.text}</span> : null}
        <div className="relative w-72 min-w-32 shrink">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input aria-label="Search history" placeholder="Search dictations" className="pl-8" value={query} onChange={(event) => setQuery(event.target.value)} />
        </div>
        <Button variant="outline" size="icon" className="size-8" data-track="export" aria-label={search ? "Export results" : "Export history"} title={search ? "Export results" : "Export history"} disabled={!entries.length} onClick={() => void exportAll()}>
          <FileDown />
        </Button>
        <Button variant="outline" size="icon" className="size-8" data-track="settings" aria-label="History settings" title="History settings" onClick={() => setSettingsOpen(true)}>
          <SlidersHorizontal />
        </Button>
      </div>
      {/* The list is as wide as it was dragged; CSS keeps it within the window, so what an entry holds never moves the split. */}
      <div
        ref={panesRef}
        className="grid min-h-0 flex-1 grid-rows-[minmax(0,1fr)] gap-1.5"
        style={{ gridTemplateColumns: `clamp(${LIST.least}px, ${listWidth}px, calc(100% - ${READER_LEAST + PANE_GAP}px)) minmax(0, 1fr)` }}
      >
        <div className="relative min-w-0">
          <ResizeEdge
            label="History list width"
            width={listWidth}
            standard={LIST.standard}
            least={LIST.least}
            most={LIST.most}
            fit={(wanted) => fitList(wanted, panesRef.current?.clientWidth ?? Infinity)}
            onResize={setListWidth}
            onSettle={(width) => saveWidth(LIST_WIDTH, width)}
            className="-right-[6px] rounded-full after:absolute after:top-1/2 after:left-1/2 after:h-7 after:w-0.5 after:-translate-x-1/2 after:-translate-y-1/2 after:rounded-full after:bg-input"
          />
          <Card className="relative flex h-full flex-col overflow-hidden">
            <div ref={listRef} role="listbox" aria-label="Dictations" tabIndex={0} onKeyDown={moveSelection} className="min-h-0 flex-1 overflow-y-auto outline-none">
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
                      data-track="row"
                      aria-selected={active}
                      style={style}
                      onClick={() => select(entry.id)}
                      className={cn(
                        "absolute inset-x-1 top-0 flex flex-col justify-center rounded-lg px-3 transition-colors hover:bg-accent/60",
                        active && "bg-selected hover:bg-selected",
                      )}
                    >
                      <span dir="auto" className={cn("truncate text-[13px]", blank && "text-muted-foreground italic")}>{preview(entry)}</span>
                      <span className="mt-0.5 flex items-center gap-1.5 text-[11.5px] text-muted-foreground tabular-nums">
                        {parts(entry.at).time} · {formatDuration(entry.audio_seconds)}
                        {blank ? null : ` · ${words(finalText(entry))}`}
                        {entry.status ? <Badge variant={statusVariant(entry.status)}>{entry.status}</Badge> : null}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
            {!entries.length ? <p className="absolute inset-0 grid place-items-center p-6 text-center text-[12.5px] text-muted-foreground">{emptyText}</p> : null}
            {/* Centred, under a rule that stops short of the edges: the card's corners are round, and anything set in them is squeezed. */}
            {entries.length ? (
              <div className="mx-4 flex h-9 shrink-0 items-center justify-center gap-2 border-t text-[11.5px] text-muted-foreground tabular-nums">
                {hasMore ? `Newest ${entries.length.toLocaleString()} shown` : `All ${entries.length.toLocaleString()} shown`}
                {hasMore ? (
                  <Button variant="link" className="text-[11.5px]" data-track="show-all" onClick={() => setShown({ search, limit: ALL })}>
                    Show all
                  </Button>
                ) : null}
              </div>
            ) : null}
          </Card>
        </div>
        <Card className="flex min-w-0 flex-col">
          {selected && actions ? (
            <>
              <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-0.5 px-4 pt-4">
                <span className="text-[15px] font-semibold">
                  {dayLabel(selected.at, new Date())}, {parts(selected.at).time}
                </span>
                <span className="flex items-center gap-2 text-[12.5px] text-muted-foreground tabular-nums">
                  {formatDuration(selected.audio_seconds)}
                  {shownText.trim() ? ` · ${words(shownText)}` : null}
                  {selected.status ? <Badge variant={statusVariant(selected.status)}>{selected.status}</Badge> : null}
                </span>
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3 text-[15px] leading-relaxed">
                {settings?.saved.show_english && hasArabic(shownText) ? (
                  <EnglishText text={shownText} />
                ) : (
                  <div dir="auto" className="selectable whitespace-pre-wrap">{shownText || <span className="text-muted-foreground italic">No text.</span>}</div>
                )}
              </div>
              {statusNote ? <p className={cn("px-4 pb-2 text-[12.5px]", noteIsError ? "selectable text-destructive" : "text-muted-foreground")}>{statusNote}</p> : null}
              <div className="flex flex-wrap items-center gap-2 border-t px-4 py-3">
                {buttons.map(({ name, label, icon: Icon, disabled, run }, at) => (
                  <Button key={name} variant={at === 0 ? "default" : "outline"} data-track={name} disabled={disabled} onClick={() => void run()}>
                    <Icon />
                    {label}
                  </Button>
                ))}
                <Button variant="ghost" size="icon" className="ml-auto size-8 text-muted-foreground hover:text-destructive" data-track="delete" aria-label="Delete" title="Delete" onClick={() => void actions.remove()}>
                  <Trash2 />
                </Button>
              </div>
            </>
          ) : (
            <p className="m-auto px-6 text-center text-[12.5px] text-muted-foreground">Pick a dictation to read it.</p>
          )}
        </Card>
      </div>
      <HistorySettings open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowDown, ArrowLeft, Circle, CircleAlert, Download, FileAudio, FolderOpen, LoaderCircle, PencilLine, Search, Square, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent, type ReactNode } from "react";
import { Link, useParams } from "react-router";
import { formatGigabytes } from "../components/models";
import { Bars, Card, ContextMenu, InlineError, PAGE_EDGE, PageBody, PageTitle, ProgressBar, Row, Rows, Section, Stat, type Bar } from "../components/parts";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Select } from "../components/ui/select";
import { Switch } from "../components/ui/switch";
import { api, errorText } from "../lib/api";
import { afterLeavingLivePage, applyMeetingEvent, colourOf, droppedMedia, labelOf, MEDIA_EXTENSIONS, meetingName, meetingStats, percent, talkShares, turnsOf } from "../lib/meetings";
import { chooseFolder, chooseMediaFile, onFileDrag } from "../lib/platform";
import { queries } from "../lib/queries";
import { useServerEvent } from "../lib/sse";
import { useStore } from "../lib/store";
import type { DownloadState, FileLanguage, LiveMeeting, MeetingEvent, MeetingLanguage, MeetingModel, MeetingOverview, MeetingSettings, MeetingSummary, MeetingTranscript } from "../lib/types";
import { cn, formatSaved, versusUsual } from "../lib/utils";
import { dayLabel, formatDuration, useDebounced } from "./history";


const NEAR_BOTTOM_PX = 120;
// Progress updates the cached list in place; a finished, failed or cancelled download fetches it again.
const IN_FLIGHT = new Set(["queued", "downloading", "paused"]);
const LANGUAGES: { value: MeetingLanguage; label: string }[] = [
  { value: "en", label: "English" },
  { value: "ur", label: "Urdu, written in Roman Urdu" },
  { value: "auto", label: "English and Urdu, each sentence as spoken" },
];
const FILE_LANGUAGES: { value: FileLanguage; label: string }[] = [
  { value: "en", label: "English" },
  { value: "ur", label: "Urdu, written in Roman Urdu" },
];

export function useMeetingOverview(): MeetingOverview | undefined {
  return useQuery(queries.meetings).data;
}

/** The meeting being recorded, or the last one recorded since Skryb started; `useMeetingEvents` keeps it current. */
export function useLiveMeeting(): LiveMeeting | null | undefined {
  return useQuery(queries.liveMeeting).data;
}

/** Folds `meeting` events into the live meeting and the saved meetings; mounted once, by the shell. */
export function useMeetingEvents(): void {
  const client = useQueryClient();
  useServerEvent((name, data) => {
    if (name === "download") {
      const download = data as { name: string; state: string; bytes_done: number; bytes_total: number; error: string };
      const overview = client.getQueryData<MeetingOverview>(["meetings"]);
      if (!overview?.models.some((model) => model.name === download.name)) return;
      if (!IN_FLIGHT.has(download.state)) {
        void client.invalidateQueries({ queryKey: ["meetings"] });
        return;
      }
      const progress = { ...download, state: download.state as DownloadState };
      client.setQueryData<MeetingOverview>(["meetings"], {
        ...overview,
        models: overview.models.map((model) => (model.name === download.name ? { ...model, download: progress } : model)),
      });
      return;
    }
    if (name !== "meeting") return;
    const event = data as MeetingEvent;
    const live = client.getQueryData<LiveMeeting | null>(["meeting-live"]);
    // A line or status from a meeting this window has not seen start yet: fetch it whole.
    if ((event.type === "status" || event.type === "entry") && live?.id !== event.id) void client.invalidateQueries({ queryKey: ["meeting-live"] });
    client.setQueryData<LiveMeeting | null>(["meeting-live"], (current) => applyMeetingEvent(current ?? null, event));
    if (event.type === "started" || event.type === "saved") void client.invalidateQueries({ queryKey: ["meetings"] });
    if (event.type === "names") client.setQueryData<MeetingTranscript>(["meeting", event.id], (saved) => (saved ? { ...saved, names: event.names } : saved));
  });
}

function useElapsed(started: string | undefined, running: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [running]);
  return started ? Math.max(0, (now - Date.parse(started)) / 1000) : 0;
}

function when(started: string): string {
  return `${dayLabel(started, new Date())}, ${started.slice(11, 16)}`;
}

// -- past meetings, under History ----------------------------------------------------

/** Past meetings whose name or date holds `search`, ignoring case; the meeting recording now is not past yet. */
export function pastMeetings(meetings: MeetingSummary[], recordingId: string | null, search: string): MeetingSummary[] {
  const needle = search.trim().toLocaleLowerCase();
  return meetings.filter((meeting) => meeting.id !== recordingId && (!needle || meeting.id.toLocaleLowerCase().includes(needle)));
}

export function MeetingHistoryPage() {
  const overview = useMeetingOverview();
  const live = useLiveMeeting();
  const recording = live?.recording ?? overview?.recording ?? false;
  const { confirm } = useStore();
  const client = useQueryClient();
  const [query, setQuery] = useState("");
  const search = useDebounced(query, 120);
  const [menu, setMenu] = useState<{ id: string; x: number; y: number } | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [problem, setProblem] = useState("");
  const closeMenu = useCallback(() => setMenu(null), []);
  const past = useMemo(() => pastMeetings(overview?.meetings ?? [], recording ? (live?.id ?? null) : null, search), [overview, recording, live?.id, search]);
  const today = new Date();

  const rename = async (id: string, name: string | null) => {
    setRenaming(null);
    if (name === null || name.trim() === meetingName(id)) return;
    setProblem("");
    try {
      const renamed = await api.post<{ id: string }>(`/api/meetings/${encodeURIComponent(id)}/name`, { name });
      // Shown at once, so the old name doesn't flash back while the list reloads.
      client.setQueryData<MeetingOverview>(["meetings"], (current) =>
        current && { ...current, meetings: current.meetings.map((meeting) => (meeting.id === id ? { ...meeting, id: renamed.id } : meeting)) },
      );
      await client.invalidateQueries({ queryKey: ["meetings"] });
    } catch (error) {
      setProblem(errorText(error));
    }
  };
  const remove = async (id: string) => {
    const sure = await confirm(
      "Delete this meeting?",
      `“${meetingName(id)}” goes to the Recycle Bin with its recording.`,
      "Delete",
    );
    if (!sure) return;
    setProblem("");
    try {
      await api.del(`/api/meetings/${encodeURIComponent(id)}`);
      client.removeQueries({ queryKey: ["meeting", id] });
      await client.invalidateQueries({ queryKey: ["meetings"] });
    } catch (error) {
      setProblem(errorText(error));
    }
  };
  const openMenu = (id: string, event: MouseEvent<HTMLElement>) => {
    event.preventDefault();
    const row = event.currentTarget.getBoundingClientRect();
    // The Menu key and Shift+F10 open it with no pointer: then it opens under the row.
    const byKeyboard = event.clientX === 0 && event.clientY === 0;
    setMenu({ id, x: byKeyboard ? row.left + 12 : event.clientX, y: byKeyboard ? row.bottom : event.clientY });
  };
  const keys = (id: string, event: KeyboardEvent<HTMLElement>) => {
    if (event.key === "F2") setRenaming(id);
    else if (event.key === "Delete") void remove(id);
    else return;
    event.preventDefault();
  };

  const empty = !overview ? "Loading…" : search ? "No meetings match." : "Meetings you record or transcribe land here.";

  return (
    <div className={cn(PAGE_EDGE, "flex h-full flex-col gap-3 pb-5")}>
      <PageTitle title="History">
        <div className="relative w-72">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input aria-label="Search meetings" placeholder="Search meetings" className="pl-8" value={query} onChange={(event) => setQuery(event.target.value)} />
        </div>
      </PageTitle>
      <div className="relative min-h-0 flex-1">
        <nav aria-label="Past meetings" className="h-full overflow-y-auto pb-2">
          <div className="grid grid-cols-[repeat(auto-fill,minmax(250px,1fr))] gap-3">
            {past.map((meeting) =>
              renaming === meeting.id ? (
                <RenameField key={meeting.id} meeting={meeting} onDone={(name) => void rename(meeting.id, name)} />
              ) : (
                <Link
                  key={meeting.id}
                  to={`/meetings/${encodeURIComponent(meeting.id)}`}
                  draggable={false}
                  title={meeting.id}
                  onContextMenu={(event) => openMenu(meeting.id, event)}
                  onKeyDown={(event) => keys(meeting.id, event)}
                  className={cn(
                    "surface group flex flex-col overflow-hidden rounded-xl border border-edge bg-card shadow-card outline-none transition-[border-color,box-shadow] hover:border-primary/50 focus-visible:ring-2 focus-visible:ring-ring/40",
                    menu?.id === meeting.id && "border-primary/50",
                  )}
                >
                  <MeetingCover meeting={meeting} />
                  <div className="px-3.5 py-2.5">
                    <p className="truncate text-[13.5px] font-medium">{meetingName(meeting.id)}</p>
                    <p className="mt-0.5 truncate text-[12px] text-muted-foreground tabular-nums">
                      {dayLabel(meeting.started, today)}, {meeting.started.slice(11, 16)} · {meeting.speakers} {meeting.speakers === 1 ? "voice" : "voices"}
                    </p>
                  </div>
                </Link>
              ),
            )}
          </div>
        </nav>
        {!past.length ? <p className="absolute inset-0 grid place-items-center p-6 text-center text-[12.5px] text-muted-foreground">{empty}</p> : null}
        {problem ? (
          <p role="alert" className="selectable absolute inset-x-0 bottom-0 rounded-lg border bg-card px-4 py-2 text-[12px] text-destructive">
            {problem}
          </p>
        ) : null}
      </div>
      {menu ? (
        <ContextMenu
          x={menu.x}
          y={menu.y}
          label={meetingName(menu.id)}
          onClose={closeMenu}
          items={[
            { label: "Rename", icon: PencilLine, shortcut: "F2", onSelect: () => setRenaming(menu.id) },
            { label: "Delete", icon: Trash2, shortcut: "Del", danger: true, onSelect: () => void remove(menu.id) },
          ]}
        />
      ) : null}
    </div>
  );
}

/** The meeting's name as a text box in its row: Enter or clicking away saves it, Escape keeps the old one. */
function RenameField({ meeting, onDone }: { meeting: MeetingSummary; onDone: (name: string | null) => void }) {
  const done = useRef(false);
  const finish = (name: string | null) => {
    if (done.current) return;
    done.current = true;
    onDone(name);
  };
  return (
    <div className="flex flex-col overflow-hidden rounded-xl border border-primary bg-card shadow-card">
      <MeetingCover meeting={meeting} />
      <div className="px-3.5 py-2.5">
        <input
          autoFocus
          defaultValue={meetingName(meeting.id)}
          maxLength={80}
          aria-label="Meeting name"
          onFocus={(event) => event.currentTarget.select()}
          onBlur={(event) => finish(event.currentTarget.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") finish(event.currentTarget.value);
            else if (event.key === "Escape") finish(null);
            else return;
            event.preventDefault();
          }}
          className="w-full rounded-sm bg-transparent px-1 text-[13.5px] font-medium text-foreground outline-none ring-1 ring-primary"
        />
        <p className="mt-0.5 px-1 text-[12px] text-muted-foreground">Enter saves, Esc cancels</p>
      </div>
    </div>
  );
}

/** The top of a meeting's card: the video's own thumbnail when it had one, else bars drawn from its name in the voices' colours. */
function MeetingCover({ meeting }: { meeting: MeetingSummary }) {
  const thumbnail = useQuery({
    queryKey: ["meeting-thumbnail", meeting.id],
    queryFn: () => api.image(`/api/meetings/${encodeURIComponent(meeting.id)}/thumbnail`),
    enabled: meeting.has_thumbnail,
    staleTime: Infinity,
    retry: false,
  }).data;
  const bars = useMemo(() => coverBars(meeting.id), [meeting.id]);
  const voices = Math.max(1, Math.min(meeting.speakers, 8));
  return (
    <div className="relative aspect-[2/1] overflow-hidden bg-muted">
      {thumbnail ? (
        <img src={thumbnail} alt="" draggable={false} className="size-full object-cover" />
      ) : (
        <div aria-hidden className="flex size-full items-center justify-center gap-[3px] bg-gradient-to-br from-primary/15 to-primary/[0.03] px-8">
          {bars.map((height, index) => (
            <span key={index} className="w-[3px] rounded-full opacity-75" style={{ height: `${height}%`, background: colourOf(`them:${index % voices}`) }} />
          ))}
        </div>
      )}
      <span className="absolute right-2 bottom-2 rounded-md bg-black/65 px-1.5 py-0.5 text-[11.5px] font-medium text-white tabular-nums">{formatDuration(meeting.seconds)}</span>
    </div>
  );
}

/** Bar heights, as percentages, the same for a meeting every time: a picture of a conversation, not of its sound. */
export function coverBars(id: string, count = 28): number[] {
  let seed = 2166136261;
  for (const char of id) seed = Math.imul(seed ^ char.charCodeAt(0), 16777619) >>> 0;
  return Array.from({ length: count }, () => {
    seed = Math.imul(seed ^ (seed >>> 15), 2246822507) >>> 0;
    seed = Math.imul(seed ^ (seed >>> 13), 3266489909) >>> 0;
    return 18 + (((seed ^ (seed >>> 16)) >>> 0) % 55);
  });
}

// -- shared pieces ---------------------------------------------------------------------

function StartStopButton({ live, overview }: { live: LiveMeeting | null | undefined; overview: MeetingOverview | undefined }) {
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState("");
  const recording = live?.recording ?? false;
  const finishing = live?.status === "Finishing";
  const file = recording && Boolean(live?.source);
  const run = async () => {
    setBusy(true);
    setProblem("");
    try {
      await api.post(recording ? "/api/meetings/stop" : "/api/meetings/start");
    } catch (error) {
      setProblem(errorText(error));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="flex flex-col items-end gap-1">
      <Button
        size="default"
        variant={file ? "outline" : recording ? "destructive" : "default"}
        disabled={busy || finishing || (!recording && overview?.models_ready === false)}
        onClick={() => void run()}
        className="h-10 min-w-[188px] px-5 text-[14px] [&_svg]:size-4"
      >
        {busy || finishing ? <LoaderCircle className="animate-spin" /> : file ? <X /> : recording ? <Square className="fill-current" /> : <Circle className="fill-current" />}
        {finishing ? "Finishing the transcript" : file ? "Cancel" : recording ? "Stop recording" : "Start meeting"}
      </Button>
      {problem ? <span className="max-w-[360px] text-right text-[12px] text-destructive">{problem}</span> : null}
    </div>
  );
}

function StatusBadge({ live }: { live: LiveMeeting }) {
  if (live.status === "Recording") {
    return (
      <Badge variant="destructive">
        <span className="size-1.5 rounded-full bg-destructive motion-safe:animate-pulse" />
        Recording
      </Badge>
    );
  }
  if (live.status === "Saved") return <Badge variant="success">Saved</Badge>;
  return (
    <Badge variant="primary">
      <LoaderCircle className="size-3 animate-spin" />
      {live.status}
    </Badge>
  );
}

/** The transcript, one heading per change of speaker, kept scrolled to the newest line unless the reader scrolled up. */
function TranscriptPane({ transcript, live, onNameClick, empty }: { transcript: MeetingTranscript; live: boolean; onNameClick: (key: string) => void; empty: ReactNode }) {
  const turns = useMemo(() => turnsOf(transcript.entries), [transcript.entries]);
  const box = useRef<HTMLDivElement>(null);
  const [atEnd, setAtEnd] = useState(true);
  const stick = useRef(true);
  const landed = useRef(false);
  const lastTop = useRef(0);
  const nearBottom = () => {
    const element = box.current;
    return !element || element.scrollHeight - element.scrollTop - element.clientHeight < NEAR_BOTTOM_PX;
  };
  // Opening the page lands on the newest line at once; after that each new line glides into view.
  useEffect(() => {
    const element = box.current;
    if (element && live && stick.current) {
      element.scrollTo({ top: element.scrollHeight, behavior: landed.current ? "auto" : "instant" });
      landed.current = transcript.entries.length > 0;
    }
    setAtEnd(stick.current);
  }, [transcript.entries, live]);
  // Only scrolling up lets go of the newest line, so a glide down to a tall new line never does.
  const onScroll = () => {
    const element = box.current;
    if (!element) return;
    const up = element.scrollTop < lastTop.current;
    lastTop.current = element.scrollTop;
    if (up || nearBottom()) stick.current = nearBottom();
    setAtEnd(stick.current);
  };
  const jump = () => {
    const element = box.current;
    stick.current = true;
    setAtEnd(true);
    if (element) element.scrollTop = element.scrollHeight;
  };
  return (
    <Card className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
      <div ref={box} onScroll={onScroll} aria-live={live ? "polite" : undefined} className="min-h-0 flex-1 overflow-y-auto px-5 py-3 motion-safe:scroll-smooth">
        {turns.length ? (
          turns.map((turn) => (
            <div
              key={`${turn.key}-${turn.lines[0]?.id ?? turn.start}`}
              style={{ ["--speaker" as string]: colourOf(turn.key) }}
              className="grid grid-cols-1 gap-x-5 gap-y-0.5 border-b py-2.5 last:border-b-0 min-[1200px]:grid-cols-[180px_minmax(0,95ch)]"
            >
              <div className="flex items-baseline gap-2 min-[1200px]:flex-col min-[1200px]:items-end min-[1200px]:gap-0">
                <button onClick={() => onNameClick(turn.key)} className="max-w-full truncate text-[13px] font-semibold text-[var(--speaker)] hover:underline">
                  {labelOf(turn.key, transcript.names)}
                </button>
                <time className="text-[11.5px] text-muted-foreground tabular-nums">{formatDuration(turn.start)}</time>
              </div>
              <div className="selectable text-[13.5px] leading-relaxed">
                {turn.lines.map((line) => (
                  <p key={line.id} className="mb-1 last:mb-0">
                    {line.text}
                  </p>
                ))}
              </div>
            </div>
          ))
        ) : (
          <div className="grid h-full place-items-center py-16 text-center text-[13px] text-muted-foreground">{empty}</div>
        )}
      </div>
      {!atEnd && live ? (
        <Button size="sm" onClick={jump} className="enter-pop absolute bottom-4 left-1/2 -translate-x-1/2 rounded-full shadow-lg">
          <ArrowDown />
          Jump to latest
        </Button>
      ) : null}
    </Card>
  );
}

/** Names and talk share. A name is saved when the box loses focus or on Enter; Esc puts the old one back. */
function SpeakersCard({ transcript, meetingId, focusKey }: { transcript: MeetingTranscript; meetingId: string; focusKey: { key: string; at: number } | null }) {
  const shares = useMemo(() => talkShares(transcript.entries, transcript.names), [transcript.entries, transcript.names]);
  const [problem, setProblem] = useState("");
  const inputs = useRef(new Map<string, HTMLInputElement>());
  useEffect(() => {
    if (focusKey) inputs.current.get(focusKey.key)?.select();
  }, [focusKey]);
  const save = async (key: string, name: string) => {
    if (name.trim() === labelOf(key, transcript.names)) return;
    setProblem("");
    try {
      await api.post(`/api/meetings/${encodeURIComponent(meetingId)}/rename`, { key, name });
    } catch (error) {
      setProblem(errorText(error));
    }
  };
  const keys = (event: KeyboardEvent<HTMLInputElement>, key: string) => {
    if (event.key === "Enter") event.currentTarget.blur();
    if (event.key === "Escape") {
      event.currentTarget.value = labelOf(key, transcript.names);
      event.currentTarget.blur();
    }
  };
  return (
    <Card className="px-4 py-3.5">
      <h2 className="mb-2.5 text-[13px] font-semibold text-muted-foreground">Speakers</h2>
      {shares.length ? (
        <div className="grid grid-cols-1 gap-x-5 gap-y-2.5 max-[1199px]:grid-cols-[repeat(auto-fill,minmax(220px,1fr))]">
          {shares.map(({ key, share }) => (
            <div key={key} style={{ ["--speaker" as string]: colourOf(key) }} className="grid grid-cols-[10px_minmax(0,1fr)_auto] items-center gap-x-2.5 gap-y-1">
              <span className="size-2.5 rounded-[3px] bg-[var(--speaker)]" />
              <Input
                key={`${key}:${labelOf(key, transcript.names)}`}
                ref={(element) => {
                  if (element) inputs.current.set(key, element);
                  else inputs.current.delete(key);
                }}
                aria-label={`Name for ${labelOf(key, transcript.names)}`}
                defaultValue={labelOf(key, transcript.names)}
                maxLength={60}
                onBlur={(event) => void save(key, event.currentTarget.value)}
                onKeyDown={(event) => keys(event, key)}
                className="h-7 border-transparent bg-transparent px-1.5 shadow-none hover:border-input focus-visible:bg-card"
              />
              <span className="w-9 text-right text-[12px] text-muted-foreground tabular-nums">{Math.round(share * 100)}%</span>
              <div className="col-start-2 col-end-4 h-1 overflow-hidden rounded-full bg-muted">
                <div className="h-full rounded-full bg-[var(--speaker)] transition-[width] duration-300" style={{ width: `${share * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-[12.5px] text-muted-foreground">Speakers appear as they talk.</p>
      )}
      <p className="mt-3 text-[12px] text-muted-foreground">Click a name to rename that speaker everywhere, saved files included.</p>
      {problem ? <InlineError>{problem}</InlineError> : null}
    </Card>
  );
}

export function Detail({ term, children }: { term: string; children: ReactNode }) {
  return (
    <>
      <dt className="text-muted-foreground">{term}</dt>
      <dd className="min-w-0 break-words">{children}</dd>
    </>
  );
}

function OpenFolderButton({ path }: { path: string }) {
  const [problem, setProblem] = useState("");
  return (
    <>
      <Button variant="link" className="text-[12.5px]" onClick={() => void api.post(path).catch((error: unknown) => setProblem(errorText(error)))}>
        <FolderOpen />
        Open folder
      </Button>
      {problem ? <InlineError>{problem}</InlineError> : null}
    </>
  );
}

/** Back to the list of meetings, beside the meeting's own title. */
function BackToHistory() {
  return (
    <Button asChild variant="ghost" size="icon" aria-label="Back to History" title="Back to History">
      <Link to="/meetings" draggable={false}>
        <ArrowLeft />
      </Link>
    </Button>
  );
}

/** Header, transcript and side cards; `fill` pins it to the window so the transcript scrolls, else the page scrolls as a whole. */
function MeetingLayout({ header, warning, transcript, aside, fill = true }: { header: ReactNode; warning?: string | null; transcript: ReactNode; aside: ReactNode; fill?: boolean }) {
  return (
    <div className={cn(PAGE_EDGE, "flex w-full flex-col gap-3 pb-5", fill ? "h-full" : "min-h-full")}>
      {header}
      {warning ? (
        <div role="alert" className="selectable flex items-center gap-2 rounded-lg border border-warning/30 bg-warning/10 px-3.5 py-2 text-[12.5px] text-warning">
          <CircleAlert className="size-4 shrink-0" />
          {warning}
        </div>
      ) : null}
      <div className="flex min-h-0 flex-1 flex-col gap-3 min-[1200px]:grid min-[1200px]:grid-cols-[minmax(0,1fr)_380px] min-[1200px]:grid-rows-[minmax(0,1fr)]">
        <div className="order-2 flex min-h-0 flex-1 flex-col min-[1200px]:order-1">{transcript}</div>
        <aside className="order-1 flex flex-col gap-3 overflow-y-auto min-[1200px]:order-2">{aside}</aside>
      </div>
    </div>
  );
}

// -- downloading the models -------------------------------------------------------------

function modelLine(model: MeetingModel): string {
  const download = model.download;
  if (model.installed) return "Installed";
  if (!download) return `${model.display_size} to download`;
  if (download.state === "queued") return "Queued";
  if (download.state === "failed") return `Download failed: ${download.error || "no reason was given"}`;
  const share = download.bytes_total ? download.bytes_done / download.bytes_total : 0;
  return `${download.state === "paused" ? "Paused" : "Downloading"} ${Math.round(share * 100)}% · ${formatGigabytes(download.bytes_done)} of ${formatGigabytes(download.bytes_total)}`;
}

export function MeetingModels({ overview, compact = false }: { overview: MeetingOverview; compact?: boolean }) {
  const [problem, setProblem] = useState("");
  const downloading = overview.models.some((model) => model.download && model.download.state !== "failed");
  const allInstalled = overview.models.every((model) => model.installed);
  const download = async () => {
    setProblem("");
    try {
      await api.post("/api/meetings/models/download");
    } catch (error) {
      setProblem(errorText(error));
    }
  };
  return (
    <Card className={cn("px-4 py-3.5", compact && "border-primary/30 bg-selected")}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-[13.5px] font-medium">{overview.models_ready ? "Meeting models" : "Meetings need these models"}</h2>
          <p className="mt-0.5 max-w-[560px] text-[12.5px] text-muted-foreground">
            One tells the voices apart, the others write down what they say. All run on this PC; nothing leaves it.
          </p>
        </div>
        {allInstalled ? null : (
          <Button size="sm" disabled={downloading} onClick={() => void download()}>
            <Download />
            {downloading ? "Downloading" : "Download meeting models"}
          </Button>
        )}
      </div>
      <div className="mt-3 grid gap-2.5 min-[900px]:grid-cols-2">
        {overview.models.map((model) => (
          <div key={model.name} className="rounded-lg border bg-card px-3 py-2">
            <div className="flex items-center gap-2">
              <span className="text-[13px] font-medium">{model.label}</span>
              <span className="text-[12px] text-muted-foreground">{model.display_size}</span>
              {model.installed ? <Badge variant="success">Installed</Badge> : null}
            </div>
            <div className="mt-0.5 text-[12px] text-muted-foreground">{model.description}</div>
            {model.download && model.download.state !== "failed" ? (
              <ProgressBar className="mt-2" value={model.download.bytes_total ? model.download.bytes_done / model.download.bytes_total : 0} />
            ) : null}
            {!model.installed ? <div className={cn("mt-1 text-[12px]", model.download?.state === "failed" ? "text-destructive" : "text-muted-foreground")}>{modelLine(model)}</div> : null}
          </div>
        ))}
      </div>
      {problem ? <InlineError>{problem}</InlineError> : null}
    </Card>
  );
}

/** Drop a video or audio file on the window, or pick one: it is transcribed as a meeting of its own. */
function FileDropCard({ ready, meetingLanguage, className }: { ready: boolean; meetingLanguage: MeetingLanguage | undefined; className?: string }) {
  const [language, setLanguage] = useState<FileLanguage>(meetingLanguage === "ur" ? "ur" : "en");
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState("");
  const transcribe = async (path: string | null) => {
    if (!path) return;
    setBusy(true);
    setProblem("");
    try {
      await api.post("/api/meetings/files", { path, language });
    } catch (error) {
      setProblem(errorText(error));
    } finally {
      setBusy(false);
    }
  };
  const latest = useRef(transcribe);
  latest.current = transcribe;
  useEffect(
    () =>
      onFileDrag((drag) => {
        if (drag.type !== "drop") return setOver(drag.type === "over");
        setOver(false);
        if (ready) void latest.current(droppedMedia(drag.paths));
        else setProblem("Download the meeting models first.");
      }),
    [ready],
  );
  return (
    <Card className={cn("border-dashed px-5 py-4 transition-colors", over && "border-primary bg-primary/5", className)}>
      <div className="flex items-start gap-3.5">
        <FileAudio className={cn("mt-0.5 size-5 shrink-0 text-muted-foreground", over && "text-primary")} aria-hidden />
        <div className="min-w-0 flex-1">
          <p className="text-[14px] font-medium text-foreground">{over ? "Drop it to transcribe it" : "Transcribe a recording"}</p>
          <p className="mt-0.5 text-[12.5px] text-muted-foreground">
            Drop a video or audio file on this window, or choose one.
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Select value={language} options={FILE_LANGUAGES} onChange={(value) => setLanguage(value as FileLanguage)} label="Language of the file" className="w-[220px]" />
            <Button variant="outline" disabled={busy || !ready} onClick={async () => void transcribe(await chooseMediaFile(MEDIA_EXTENSIONS))}>
              {busy ? <LoaderCircle className="animate-spin" /> : null}
              Choose file
            </Button>
          </div>
          {problem ? <InlineError>{problem}</InlineError> : null}
        </div>
      </div>
    </Card>
  );
}

// -- Home: the live meeting --------------------------------------------------------------

/** Meeting mode's Home: Start, a file to drop, and the meeting as it is written down. */
export function MeetingHomePage() {
  const live = useLiveMeeting();
  const overview = useMeetingOverview();
  const client = useQueryClient();
  // A meeting just saved stays on show until the page is left; back here, the page is ready for the next one.
  useEffect(
    () => () => {
      client.setQueryData<LiveMeeting | null>(["meeting-live"], (current) => afterLeavingLivePage(current ?? null));
    },
    [client],
  );
  const { settings } = useStore();
  const [focusKey, setFocusKey] = useState<{ key: string; at: number } | null>(null);
  const running = live?.recording ?? false;
  const file = Boolean(live?.source);
  const elapsed = useElapsed(live?.started, running && !file);
  const meetings = settings?.saved.meetings;
  const language = LANGUAGES.find((choice) => choice.value === meetings?.language)?.label ?? "English";

  const header = (
    <div className="flex min-h-10 items-center gap-4">
      <div className="flex min-w-0 flex-1 items-center gap-x-3">
        <h1 className="page-title shrink-0">{live ? (running ? (file ? "Transcribing a file" : "Live meeting") : meetingName(live.id)) : "Meetings"}</h1>
        {live ? <StatusBadge live={live} /> : null}
        {live && running && !file ? <span className="text-[13px] text-muted-foreground tabular-nums">{formatDuration(elapsed)}</span> : null}
        {live?.device && running ? <span className="min-w-0 truncate text-[12.5px] text-muted-foreground">Listening to {live.device}</span> : null}
        {file && running ? (
          <div className="flex min-w-0 flex-1 items-center gap-3">
            <span className="min-w-0 truncate text-[12.5px] text-muted-foreground" title={live?.source ?? ""}>{live?.source}</span>
            {live?.progress != null ? (
              <>
                <ProgressBar value={live.progress} className="h-1.5 min-w-[80px] max-w-[220px] flex-1" />
                <span className="text-[12.5px] text-muted-foreground tabular-nums">{percent(live.progress)}</span>
              </>
            ) : null}
          </div>
        ) : null}
      </div>
      <StartStopButton live={live} overview={overview} />
    </div>
  );

  const idle = (
    <div className="max-w-[460px] text-center">
      <p className="text-[15px] font-medium text-foreground">Ready when your meeting is</p>
      <p className="mt-1.5">
        Start here, from the tray or with your shortcut. Skryb records what your PC plays{meetings?.record_microphone === false ? "" : " and your microphone"} and writes it down live, in {language}.
      </p>
      <FileDropCard ready={overview?.models_ready !== false} meetingLanguage={meetings?.language} className="mt-6 text-left" />
    </div>
  );

  const transcript: MeetingTranscript = live ?? { folder: "", started: "", names: {}, entries: [] };
  return (
    <MeetingLayout
      header={header}
      warning={live?.warning}
      fill={Boolean(live)}
      transcript={
        <div className="flex min-h-0 flex-1 flex-col gap-3">
          {overview && !overview.models_ready ? <MeetingModels overview={overview} compact /> : null}
          {!live && overview?.meetings.length ? <MeetingNumbers meetings={overview.meetings} recordsMicrophone={meetings?.record_microphone !== false} /> : null}
          {live && !running ? <FileDropCard ready={overview?.models_ready !== false} meetingLanguage={meetings?.language} /> : null}
          <TranscriptPane
            transcript={transcript}
            live={Boolean(live) && live?.status !== "Saved"}
            onNameClick={(key) => setFocusKey({ key, at: Date.now() })}
            empty={live ? (running ? (file ? "Lines appear here as the file is transcribed." : "Listening. Lines appear a second or two after someone stops talking.") : "Nothing was said in this meeting.") : idle}
          />
        </div>
      }
      aside={
        <>
          {live ? <SpeakersCard transcript={transcript} meetingId={live.id} focusKey={focusKey} /> : null}
          {/* Narrow windows leave the transcript the room: the header already says what is recording. */}
          <Card className={cn("px-4 py-3.5", live && "max-[1199px]:hidden")}>
            <h2 className="mb-2.5 text-[13px] font-semibold text-muted-foreground">Session</h2>
            <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-[12.5px]">
              {live ? <Detail term="Started">{when(live.started)}</Detail> : null}
              {live ? <Detail term="Length">{formatDuration(running && !file ? elapsed : Math.max(0, ...live.entries.map((entry) => entry.end)))}</Detail> : null}
              {file ? (
                <Detail term="File">{live?.source}</Detail>
              ) : (
                <>
                  <Detail term="Listening to">{running ? live?.device || "Finding the speakers" : "What your PC plays"}</Detail>
                  <Detail term="Microphone">{meetings?.record_microphone === false ? "Not recorded" : "Recorded as Me"}</Detail>
                  <Detail term="Language">{language}</Detail>
                </>
              )}
              <Detail term="Saved to">
                <span className="selectable">{live?.folder || overview?.folder || "…"}</span>
              </Detail>
            </dl>
            <div className="mt-2 flex flex-wrap gap-x-4">
              <OpenFolderButton path={live ? `/api/meetings/${encodeURIComponent(live.id)}/open` : "/api/meetings/folder/open"} />
              <Link to="/dictionary" draggable={false} className="text-[12.5px] font-medium text-primary hover:underline">
                Edit names and terms
              </Link>
            </div>
          </Card>
        </>
      }
    />
  );
}

/** Home's numbers between meetings: this week against a usual week, every meeting saved, a typical one, and 12 weeks of meeting time. */
function MeetingNumbers({ meetings, recordsMicrophone }: { meetings: MeetingSummary[]; recordsMicrophone: boolean }) {
  const stats = useMemo(() => meetingStats(meetings), [meetings]);
  const plural = (value: number, one: string, many: string) => `${value.toLocaleString()} ${value === 1 ? one : many}`;
  const compare = versusUsual(stats.week.seconds, stats.usualWeek);
  const bars = weekBars(stats.weeks);
  const typicalVoices = stats.typicalSpeakers === null ? "" : `usually ${plural(Math.round(stats.typicalSpeakers), "voice", "voices")}`;
  return (
    <div className="flex flex-col gap-3">
      <div className={cn("grid grid-cols-2 gap-3", stats.talk && recordsMicrophone ? "min-[1500px]:grid-cols-4" : "min-[1500px]:grid-cols-3")}>
        <Stat label="This week so far" value={formatSaved(stats.week.seconds)} detail={[plural(stats.week.count, "meeting", "meetings"), compare ?? ""].filter(Boolean).join(" · ")} />
        <Stat label="Typical meeting" value={stats.typicalSeconds === null ? "–" : formatSaved(stats.typicalSeconds)} detail={typicalVoices || undefined} />
        {stats.talk && recordsMicrophone ? (
          <Stat
            label="Your share of talk"
            value={`${Math.round(stats.talk.share * 100)}%`}
            detail={`An even split is ${Math.round(stats.talk.even * 100)}% · ${plural(stats.talk.meetings, "meeting", "meetings")}, last 30 days`}
            title="Voices in the room your microphone picks up count as you."
          />
        ) : null}
        <Stat label="All saved" value={plural(stats.count, "meeting", "meetings")} detail={`${formatSaved(stats.seconds)} recorded`} />
      </div>
      <Bars
        title="Meeting time a week, last 12 weeks"
        bars={bars}
        format={(seconds) => formatSaved(seconds)}
        axis={[bars[0]?.label ?? "", "This week"]}
        usual={stats.usualFullWeek}
      />
    </div>
  );
}

/** Bars for the last weeks, labelled by their Monday, this one "This week so far". */
function weekBars(weeks: number[], now: Date = new Date()): Bar[] {
  const monday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - ((now.getDay() + 6) % 7));
  return weeks.map((value, index) => {
    const back = weeks.length - 1 - index;
    const start = new Date(monday.getFullYear(), monday.getMonth(), monday.getDate() - 7 * back);
    return { value, label: back ? `Week of ${start.toLocaleDateString(undefined, { day: "numeric", month: "short" })}` : "This week so far" };
  });
}

// -- a saved meeting --------------------------------------------------------------------

export function SavedMeetingPage() {
  const id = useParams().id ?? "";
  const [focusKey, setFocusKey] = useState<{ key: string; at: number } | null>(null);
  const meeting = useQuery({ queryKey: ["meeting", id], queryFn: () => api.get<MeetingTranscript>(`/api/meetings/${encodeURIComponent(id)}`) });
  const transcript = meeting.data;
  const seconds = transcript ? Math.max(0, ...transcript.entries.map((entry) => entry.end)) : 0;
  const words = transcript ? transcript.entries.reduce((sum, entry) => sum + entry.text.split(/\s+/).filter(Boolean).length, 0) : 0;
  const header = (
    <div className="flex min-h-10 flex-wrap items-center gap-x-3 gap-y-1">
      <BackToHistory />
      <h1 className="page-title truncate" title={id}>
        {meetingName(id)}
      </h1>
      {transcript ? <span className="text-[13px] text-muted-foreground">{when(transcript.started)}</span> : null}
    </div>
  );
  if (!transcript) {
    return (
      <MeetingLayout
        header={header}
        transcript={<Card className="grid flex-1 place-items-center p-6 text-[12.5px] text-muted-foreground">{meeting.isError ? errorText(meeting.error) : "Loading…"}</Card>}
        aside={null}
      />
    );
  }
  return (
    <MeetingLayout
      header={header}
      transcript={<TranscriptPane transcript={transcript} live={false} onNameClick={(key) => setFocusKey({ key, at: Date.now() })} empty="Nothing was said in this meeting." />}
      aside={
        <>
          <SpeakersCard transcript={transcript} meetingId={id} focusKey={focusKey} />
          <Card className="px-4 py-3.5 max-[1199px]:hidden">
            <h2 className="mb-2.5 text-[13px] font-semibold text-muted-foreground">Meeting</h2>
            <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-[12.5px]">
              <Detail term="Started">{when(transcript.started)}</Detail>
              <Detail term="Length">{formatDuration(seconds)}</Detail>
              <Detail term="Lines">{transcript.entries.length.toLocaleString()}</Detail>
              <Detail term="Words">{words.toLocaleString()}</Detail>
              <Detail term="Saved in">
                <span className="selectable">{transcript.folder}</span>
              </Detail>
            </dl>
            <div className="mt-2">
              <OpenFolderButton path={`/api/meetings/${encodeURIComponent(id)}/open`} />
            </div>
          </Card>
        </>
      }
    />
  );
}

// -- meeting settings, one per sidebar item ----------------------------------------------

function useMeetingDraft(): { draft: MeetingSettings | undefined; edit: (change: Partial<MeetingSettings>) => void } {
  const { settings, editSettings } = useStore();
  return {
    draft: settings?.draft.meetings,
    edit: (change) => editSettings((current) => ({ ...current, meetings: { ...current.meetings, ...change } })),
  };
}

export function MeetingMicrophonePage() {
  const { draft, edit } = useMeetingDraft();
  return (
    <PageBody>
      <PageTitle title="Microphone & sounds" />
      {draft ? (
        <Section title="Meetings" note="Records what your PC plays, and your voice from the default microphone.">
          <Rows>
            <Row title="Record my microphone too" description="Your voice is labelled Me. Off records only what your PC plays.">
              <Switch aria-label="Record my microphone too" checked={draft.record_microphone} onCheckedChange={(on) => edit({ record_microphone: on })} />
            </Row>
          </Rows>
        </Section>
      ) : null}
    </PageBody>
  );
}

export function MeetingOutputPage() {
  const { draft, edit } = useMeetingDraft();
  const [note, setNote] = useState("");
  const changeFolder = async (current: string) => {
    const path = await chooseFolder("Choose where meetings are saved", current);
    if (path) edit({ folder: path });
  };
  return (
    <PageBody>
      <PageTitle title="Output" />
      {draft ? (
        <Section title="Meetings folder">
          <Rows>
            <Row title="Saved to" description="Each meeting gets a folder here with its audio and transcript." stacked error={note}>
              <Input aria-label="Meetings folder" value={draft.folder} onChange={(event) => edit({ folder: event.target.value })} className="flex-1" />
              <Button variant="outline" onClick={() => void changeFolder(draft.folder)}>
                Change…
              </Button>
              <Button variant="outline" onClick={() => void api.post("/api/meetings/folder/open").catch((error: unknown) => setNote(errorText(error)))}>
                Open
              </Button>
            </Row>
          </Rows>
        </Section>
      ) : null}
    </PageBody>
  );
}

export function MeetingModelPage() {
  const { draft, edit } = useMeetingDraft();
  const overview = useMeetingOverview();
  return (
    <PageBody>
      <PageTitle title="Model" />
      {draft ? (
        <Section title="Speech">
          <Rows>
            <Row title="Language" description="Urdu is written in Roman Urdu.">
              <Select label="Meeting language" className="w-72" value={draft.language} options={LANGUAGES} onChange={(language) => edit({ language: language as MeetingLanguage })} />
            </Row>
          </Rows>
        </Section>
      ) : null}
      <Section title="Models">{overview ? <MeetingModels overview={overview} /> : <Card className="h-40" />}</Section>
    </PageBody>
  );
}

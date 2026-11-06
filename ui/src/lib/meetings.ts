import type { LiveMeeting, MeetingEntry, MeetingEvent, MeetingSummary } from "./types";

const COLOURS = 8;
// Once stopping starts no more audio is recorded, though the last lines are still being written.
const STOPPED = new Set(["Finishing", "Saved"]);

/** `them:N` takes one of eight speaker colours; the microphone has its own. */
export function colourOf(key: string): string {
  const [track, speaker] = key.split(":");
  return track === "me" ? "var(--speaker-me)" : `var(--speaker-${Number(speaker) % COLOURS})`;
}

export function defaultLabel(key: string): string {
  const [track, speaker] = key.split(":");
  return track === "me" ? "Me" : `Speaker ${Number(speaker) + 1}`;
}

export function labelOf(key: string, names: Record<string, string>): string {
  return names[key]?.trim() || defaultLabel(key);
}

/** Remote voices in the order they first spoke, then the microphone. */
export function speakerOrder(a: string, b: string): number {
  const [trackA = "", speakerA = "0"] = a.split(":");
  const [trackB = "", speakerB = "0"] = b.split(":");
  if (trackA !== trackB) return trackA === "me" ? 1 : -1;
  return Number(speakerA) - Number(speakerB);
}

export interface Turn {
  key: string;
  start: number;
  lines: MeetingEntry[];
}

/** Lines in time order, consecutive lines from one speaker under one heading. */
export function turnsOf(entries: MeetingEntry[]): Turn[] {
  const turns: Turn[] = [];
  for (const entry of [...entries].sort((a, b) => a.start - b.start || a.id - b.id)) {
    const last = turns.at(-1);
    if (last && last.key === entry.key) last.lines.push(entry);
    else turns.push({ key: entry.key, start: entry.start, lines: [entry] });
  }
  return turns;
}

export interface Share {
  key: string;
  seconds: number;
  share: number;
}

/** Each speaker's talk time and share of the whole, including named speakers who have not spoken yet. */
export function talkShares(entries: MeetingEntry[], names: Record<string, string>): Share[] {
  const seconds = new Map<string, number>(Object.keys(names).map((key) => [key, 0]));
  for (const entry of entries) seconds.set(entry.key, (seconds.get(entry.key) ?? 0) + Math.max(0, entry.end - entry.start));
  const total = [...seconds.values()].reduce((sum, value) => sum + value, 0);
  return [...seconds.entries()]
    .sort(([a], [b]) => speakerOrder(a, b))
    .map(([key, value]) => ({ key, seconds: value, share: total ? value / total : 0 }));
}

/** `2026-10-16 Ops Onboarding` reads "Ops Onboarding"; a folder named only by its time reads "Meeting". */
export function meetingName(id: string): string {
  const rest = id.replace(/^\d{4}-\d{2}-\d{2}(\s+\d{2}-\d{2}(?!\d))?\s*/, "").trim();
  return !rest || /^\(\d+\)$/.test(rest) ? "Meeting" : rest;
}

/** Video and audio files Windows reads; the picker offers these, and anything else is still tried. */
export const MEDIA_EXTENSIONS = ["mp4", "mov", "mkv", "webm", "m4a", "m4v", "wmv", "wma", "avi", "mp3", "aac", "flac", "wav", "3gp"];

/** The first dropped file worth trying: one with a media extension, else the first of them. */
export function droppedMedia(paths: string[]): string | null {
  const media = paths.find((path) => MEDIA_EXTENSIONS.includes(path.split(".").pop()?.toLowerCase() ?? ""));
  return media ?? paths[0] ?? null;
}

/** Progress as the page shows it: whole percents, never past 100. */
export function percent(progress: number | null | undefined): string {
  return `${Math.min(100, Math.max(0, Math.floor((progress ?? 0) * 100)))}%`;
}

/** What the live page shows when it is opened again: a meeting already saved is in the list, not here. */
export function afterLeavingLivePage(live: LiveMeeting | null): LiveMeeting | null {
  return live && !live.recording && live.status === "Saved" ? null : live;
}

/** The live meeting after one `meeting` event; events about other meetings leave it as it is. */
export function applyMeetingEvent(live: LiveMeeting | null, event: MeetingEvent): LiveMeeting | null {
  if (event.type === "started") return { ...event, recording: true };
  if (!live || event.id !== live.id) return live;
  switch (event.type) {
    case "status":
      return {
        ...live, status: event.status, folder: event.folder, device: event.device, warning: event.warning,
        source: event.source ?? live.source, progress: event.progress ?? null, recording: !STOPPED.has(event.status),
      };
    case "entry":
      return { ...live, entries: [...live.entries.filter((entry) => entry.id !== event.entry.id), event.entry] };
    case "remove":
      return { ...live, entries: live.entries.filter((entry) => entry.id !== event.entry_id) };
    case "names":
      return { ...live, names: event.names };
    case "saved":
      return { ...live, recording: false, status: "Saved" };
  }
}

/** Meeting Home's chart covers this many weeks, this one last. */
export const CHART_WEEKS = 12;
// A usual week is the median of up to eight earlier weeks, once there are two.
const USUAL_WEEKS = 8;
const USUAL_WEEK_MINIMUM = 2;
const DAY_MS = 86_400_000;

export interface MeetingStats {
  /** Every saved meeting. */
  count: number;
  seconds: number;
  /** Monday to now. */
  week: { count: number; seconds: number };
  /** Median seconds of earlier weeks from Monday to today's weekday; null with fewer than two such weeks. */
  usualWeek: number | null;
  /** Medians over meetings in which something was said; null with none. */
  typicalSeconds: number | null;
  typicalSpeakers: number | null;
  /** Seconds recorded each week, the last 12 weeks, this one last. */
  weeks: number[];
  /** Median seconds of the complete weeks on the chart since the first meeting; null with fewer than two. */
  usualFullWeek: number | null;
  /** Over the last 30 days' meetings with both you and others in them: your share of the talking, what an even split would be, and how many. */
  talk: { share: number; even: number; meetings: number } | null;
}

/** Talk share counts the last this-many days. */
const TALK_DAYS = 30;

/** The local calendar day of *date*, as a number of days that is the same all day. */
function dayNumber(date: Date): number {
  return Math.round(new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime() / DAY_MS);
}

export function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

/** Meeting Home's numbers, in local time; "usual" counts only weeks since the first meeting, so an empty past doesn't drag it down. */
export function meetingStats(meetings: MeetingSummary[], now: Date = new Date()): MeetingStats {
  const today = dayNumber(now);
  const weekday = (now.getDay() + 6) % 7;
  const monday = today - weekday;
  const dated = meetings.map((meeting) => ({ meeting, day: dayNumber(new Date(meeting.started)) })).filter(({ day }) => !Number.isNaN(day));
  const earliest = dated.reduce<Date | null>((first, { meeting }) => {
    const started = new Date(meeting.started);
    return !first || started < first ? started : first;
  }, null);
  const firstWeek = earliest ? dayNumber(earliest) - ((earliest.getDay() + 6) % 7) : today;
  const seconds = (from: number, to: number) => dated.filter(({ day }) => day >= from && day <= to).reduce((sum, { meeting }) => sum + meeting.seconds, 0);
  const thisWeek = dated.filter(({ day }) => day >= monday && day <= today);
  const soFar: number[] = [];
  for (let back = 1; back <= USUAL_WEEKS && monday - 7 * back >= firstWeek; back += 1) soFar.push(seconds(monday - 7 * back, monday - 7 * back + weekday));
  const weeks = Array.from({ length: CHART_WEEKS }, (_, index) => {
    const start = monday - 7 * (CHART_WEEKS - 1 - index);
    return seconds(start, start + 6);
  });
  const full = weeks.slice(0, -1).filter((_, index) => monday - 7 * (CHART_WEEKS - 1 - index) >= firstWeek);
  const spoken = dated.filter(({ meeting }) => meeting.lines > 0).map(({ meeting }) => meeting);
  const shared = dated.filter(({ day, meeting }) => day > today - TALK_DAYS && day <= today && meeting.me_seconds > 0 && meeting.talk_seconds > meeting.me_seconds).map(({ meeting }) => meeting);
  const talked = shared.reduce((sum, meeting) => sum + meeting.talk_seconds, 0);
  return {
    count: dated.length,
    seconds: dated.reduce((sum, { meeting }) => sum + meeting.seconds, 0),
    week: { count: thisWeek.length, seconds: thisWeek.reduce((sum, { meeting }) => sum + meeting.seconds, 0) },
    usualWeek: soFar.length >= USUAL_WEEK_MINIMUM ? median(soFar) : null,
    typicalSeconds: median(spoken.map((meeting) => meeting.seconds)),
    typicalSpeakers: median(spoken.map((meeting) => meeting.speakers)),
    weeks,
    usualFullWeek: full.length >= USUAL_WEEK_MINIMUM ? median(full) : null,
    talk: talked > 0
      ? {
          share: shared.reduce((sum, meeting) => sum + meeting.me_seconds, 0) / talked,
          even: shared.reduce((sum, meeting) => sum + meeting.talk_seconds / Math.max(meeting.speakers, 2), 0) / talked,
          meetings: shared.length,
        }
      : null,
  };
}

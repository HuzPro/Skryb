import type { LiveMeeting, MeetingEntry, MeetingEvent } from "./types";

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

/** The live meeting after one `meeting` event; events about other meetings leave it as it is. */
export function applyMeetingEvent(live: LiveMeeting | null, event: MeetingEvent): LiveMeeting | null {
  if (event.type === "started") return { ...event, recording: true };
  if (!live || event.id !== live.id) return live;
  switch (event.type) {
    case "status":
      return { ...live, status: event.status, folder: event.folder, device: event.device, warning: event.warning, recording: !STOPPED.has(event.status) };
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

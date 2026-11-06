import { describe, expect, it } from "vitest";
import { isNavActive, launchPath, modeOf, NAV, pageInMode } from "../app";
import { groupByDay } from "../pages/history";
import { coverBars, pastMeetings } from "../pages/meetings";
import { afterLeavingLivePage, applyMeetingEvent, CHART_WEEKS, colourOf, median, meetingStats, droppedMedia, percent, defaultLabel, labelOf, meetingName, speakerOrder, talkShares, turnsOf } from "./meetings";
import type { LiveMeeting, MeetingEntry, MeetingSummary } from "./types";

const line = (id: number, key: string, start: number, end: number, text: string): MeetingEntry => {
  const [track, speaker] = key.split(":");
  return { id, track: track as "me" | "them", speaker: Number(speaker), start, end, text, key };
};

const live = (entries: MeetingEntry[] = []): LiveMeeting => ({
  id: "2026-10-29 14-05",
  recording: true,
  status: "Recording",
  device: "Headset (Jabra Evolve2 65)",
  warning: null,
  folder: "D:\\Meetings\\2026-10-29 14-05",
  started: "2026-10-29T14:05:00",
  names: {},
  entries,
});

describe("speaker labels and colours", () => {
  it("numbers voices on the speakers from one and calls the microphone Me", () => {
    expect(defaultLabel("them:0")).toBe("Speaker 1");
    expect(defaultLabel("them:7")).toBe("Speaker 8");
    expect(defaultLabel("me:0")).toBe("Me");
  });

  it("shows a given name, Urdu script included, and falls back when the name is blank", () => {
    const names = { "them:0": "Leo Brandt", "them:1": "نادیہ رحمان", "them:2": "   " };
    expect(labelOf("them:0", names)).toBe("Leo Brandt");
    expect(labelOf("them:1", names)).toBe("نادیہ رحمان");
    expect(labelOf("them:2", names)).toBe("Speaker 3");
    expect(labelOf("them:5", names)).toBe("Speaker 6");
  });

  it("gives the microphone its own colour and repeats the eight speaker colours past the eighth voice", () => {
    expect(colourOf("me:0")).toBe("var(--speaker-me)");
    expect(colourOf("them:3")).toBe("var(--speaker-3)");
    expect(colourOf("them:11")).toBe("var(--speaker-3)");
  });

  it("lists remote voices by number and the microphone last", () => {
    expect(["me:0", "them:10", "them:2", "them:0"].sort(speakerOrder)).toEqual(["them:0", "them:2", "them:10", "me:0"]);
  });
});

describe("the transcript's turns", () => {
  it("groups consecutive lines from one speaker under one heading, in time order", () => {
    const turns = turnsOf([
      line(3, "them:1", 9, 10, "Sounds good."),
      line(1, "them:0", 0, 2, "All right, let's jump in."),
      line(2, "them:0", 3, 5, "Launch Board first."),
      line(4, "me:0", 11, 12, "Theek hai."),
    ]);
    expect(turns.map((turn) => [turn.key, turn.start, turn.lines.length])).toEqual([
      ["them:0", 0, 2],
      ["them:1", 9, 1],
      ["me:0", 11, 1],
    ]);
  });

  it("keeps two lines that start at the same moment apart, by the order they arrived", () => {
    const turns = turnsOf([line(8, "them:0", 4, 5, "second"), line(7, "them:0", 4, 6, "first")]);
    expect(turns[0]?.lines.map((entry) => entry.text)).toEqual(["first", "second"]);
  });

  it("turns an hour of two thousand lines into one heading per change of speaker", () => {
    const entries = Array.from({ length: 2000 }, (_, at) => line(at, `them:${Math.floor(at / 3) % 4}`, at * 1.8, at * 1.8 + 1.5, `Line ${at} ✓`));
    const turns = turnsOf(entries);
    expect(turns).toHaveLength(Math.ceil(2000 / 3));
    expect(turns.reduce((sum, turn) => sum + turn.lines.length, 0)).toBe(2000);
  });
});

describe("talk share", () => {
  it("adds up each speaker's time and divides by the whole", () => {
    const shares = talkShares([line(1, "them:0", 0, 30, "a"), line(2, "me:0", 30, 40, "b"), line(3, "them:0", 40, 50, "c")], {});
    expect(shares).toEqual([
      { key: "them:0", seconds: 40, share: 0.8 },
      { key: "me:0", seconds: 10, share: 0.2 },
    ]);
  });

  it("lists a named speaker who has not spoken yet at none, and an empty meeting as all zero", () => {
    expect(talkShares([], { "them:0": "Omar" })).toEqual([{ key: "them:0", seconds: 0, share: 0 }]);
    expect(talkShares([line(1, "them:0", 5, 5, "")], {})).toEqual([{ key: "them:0", seconds: 0, share: 0 }]);
  });
});

describe("meeting names", () => {
  it("reads the name after the date, and calls a folder named only by its time Meeting", () => {
    expect(meetingName("2026-10-16 Ops Onboarding")).toBe("Ops Onboarding");
    expect(meetingName("2026-10-28 22-45 Marta garden planning")).toBe("Marta garden planning");
    expect(meetingName("2026-10-20 22-53")).toBe("Meeting");
    expect(meetingName("2026-10-29 14-05")).toBe("Meeting");
    expect(meetingName("2026-10-29 14-05 (2)")).toBe("Meeting");
    expect(meetingName("Dana call")).toBe("Dana call");
    expect(meetingName("2026-10-29")).toBe("Meeting");
  });
});

describe("the live meeting's events", () => {
  it("starts from the started event and adds, replaces and removes lines by id", () => {
    const { recording: _recording, ...started } = live();
    let meeting = applyMeetingEvent(null, { type: "started", ...started });
    meeting = applyMeetingEvent(meeting, { type: "entry", id: started.id, entry: line(0, "me:0", 1, 2, "perks of Lisbon") });
    meeting = applyMeetingEvent(meeting, { type: "entry", id: started.id, entry: line(1, "them:0", 0.9, 2.1, "One of the perks of Lisbon.") });
    meeting = applyMeetingEvent(meeting, { type: "entry", id: started.id, entry: line(1, "them:0", 0.9, 2.1, "One of the perks of Lisbon.") });
    meeting = applyMeetingEvent(meeting, { type: "remove", id: started.id, entry_id: 0 });
    expect(meeting?.entries.map((entry) => entry.id)).toEqual([1]);
    expect(meeting?.recording).toBe(true);
  });

  it("ignores events about another meeting", () => {
    const meeting = live([line(0, "them:0", 0, 1, "Hi.")]);
    expect(applyMeetingEvent(meeting, { type: "names", id: "2026-10-16 Ops Onboarding", names: { "them:0": "Leo" } })).toBe(meeting);
    expect(applyMeetingEvent(null, { type: "saved", id: meeting.id })).toBeNull();
  });

  it("shows a rename, a new device and the saved state", () => {
    let meeting = applyMeetingEvent(live(), { type: "names", id: "2026-10-29 14-05", names: { "them:0": "Marta Lindgren" } });
    meeting = applyMeetingEvent(meeting, {
      type: "status", id: "2026-10-29 14-05", status: "Recording", folder: "D:\\Meetings\\2026-10-29 14-05",
      device: "Speakers (Realtek(R) Audio)", warning: "No microphone found: recording the speakers only.",
    });
    expect(meeting?.names).toEqual({ "them:0": "Marta Lindgren" });
    expect(meeting?.device).toBe("Speakers (Realtek(R) Audio)");
    meeting = applyMeetingEvent(meeting, {
      type: "status", id: "2026-10-29 14-05", status: "Finishing", folder: "D:\\Meetings\\2026-10-29 14-05", device: null, warning: null,
    });
    expect([meeting?.recording, meeting?.status]).toEqual([false, "Finishing"]);
    meeting = applyMeetingEvent(meeting, { type: "saved", id: "2026-10-29 14-05" });
    expect([meeting?.recording, meeting?.status]).toEqual([false, "Saved"]);
  });
});

describe("a dropped file", () => {
  it("picks the first video or audio file when several things are dropped at once", () => {
    expect(droppedMedia(["C:\Docs\notes.txt", "D:\Calls\سالانہ جائزہ.MKV", "D:\Calls\b.mp4"])).toBe("D:\Calls\سالانہ جائزہ.MKV");
  });

  it("still tries a file without a media extension when nothing else was dropped", () => {
    expect(droppedMedia(["C:\Downloads\recording"])).toBe("C:\Downloads\recording");
    expect(droppedMedia([])).toBeNull();
  });

  it("shows progress in whole percents between 0 and 100", () => {
    expect([percent(0), percent(0.4999), percent(1), percent(1.2), percent(-0.1), percent(null)]).toEqual(["0%", "49%", "100%", "100%", "0%", "0%"]);
  });

  it("carries the file's name and progress through its status events", () => {
    const { recording: _recording, ...started } = live();
    let meeting = applyMeetingEvent(null, { type: "started", ...started, source: "Weekly sync.mp4", progress: 0 });
    meeting = applyMeetingEvent(meeting, {
      type: "status", id: started.id, status: "Transcribing", folder: started.folder, device: null, warning: null, progress: 0.42,
    });
    expect([meeting?.source, meeting?.progress, meeting?.recording]).toEqual(["Weekly sync.mp4", 0.42, true]);
    meeting = applyMeetingEvent(meeting, { type: "saved", id: started.id });
    expect([meeting?.recording, meeting?.status]).toEqual([false, "Saved"]);
  });
});

describe("coming back to the live page", () => {
  it("is ready for the next meeting once the last one is saved", () => {
    const saved = applyMeetingEvent(live([line(0, "them:0", 0, 1, "Hi.")]), { type: "saved", id: "2026-10-29 14-05" });
    expect(afterLeavingLivePage(saved)).toBeNull();
  });

  it("still shows a meeting that is recording or finishing its transcript", () => {
    const recording = live();
    const finishing = { ...live(), recording: false, status: "Finishing" as const };
    expect(afterLeavingLivePage(recording)).toBe(recording);
    expect(afterLeavingLivePage(finishing)).toBe(finishing);
    expect(afterLeavingLivePage(null)).toBeNull();
  });
});

describe("the mode a page belongs to", () => {
  it("puts the meetings pages in meeting mode and the dictation pages in dictation", () => {
    for (const page of ["/meeting-home", "/meetings", "/meetings/2026-10-16 Ops Onboarding", "/meeting-microphone", "/meeting-output", "/meeting-model"]) {
      expect(modeOf(page)).toBe("meetings");
    }
    for (const page of ["/home", "/history", "/transcription", "/microphone", "/output", "/meetingsroom", "/meeting-settings"]) {
      expect(modeOf(page)).toBe("dictation");
    }
  });

  it("leaves the pages both modes share to whichever mode opened them", () => {
    for (const page of ["/dictionary", "/shortcuts", "/updates", "/setup"]) expect(modeOf(page)).toBeNull();
  });

  it("opens meeting mode's Home when the tray asks for meetings", () => {
    expect(launchPath("meetings")).toBe("/meeting-home");
  });
});

describe("switching mode", () => {
  it("keeps the same sidebar item and swaps the page behind it", () => {
    expect(pageInMode("/home", "meetings")).toBe("/meeting-home");
    expect(pageInMode("/meeting-home", "dictation")).toBe("/home");
    expect(pageInMode("/history", "meetings")).toBe("/meetings");
    expect(pageInMode("/transcription", "meetings")).toBe("/meeting-model");
    expect(pageInMode("/meeting-output", "dictation")).toBe("/output");
    expect(pageInMode("/meeting-microphone", "dictation")).toBe("/microphone");
  });

  it("leaves a saved meeting for dictation History", () => {
    expect(pageInMode("/meetings/2026-10-16 14-03 Café ☕ sync", "dictation")).toBe("/history");
  });

  it("stays on a page both modes share", () => {
    for (const page of ["/dictionary", "/shortcuts"]) {
      expect(pageInMode(page, "meetings")).toBe(page);
      expect(pageInMode(page, "dictation")).toBe(page);
    }
  });

  it("leaves Settings for the chosen mode's Home", () => {
    expect(pageInMode("/updates", "meetings")).toBe("/meeting-home");
    expect(pageInMode("/updates", "dictation")).toBe("/home");
  });

  it("lights Home up for the live meeting and History for every saved one", () => {
    const [home, history] = NAV;
    for (const page of ["/history", "/meetings", "/meetings/2026-10-16 Ops Onboarding"]) expect(isNavActive(history!, page)).toBe(true);
    for (const page of ["/home", "/meeting-home"]) expect(NAV.filter((item) => isNavActive(item, page))).toEqual([home]);
    expect(isNavActive(history!, "/meeting-model")).toBe(false);
    expect(NAV.filter((item) => isNavActive(item, "/dictionary"))).toHaveLength(1);
  });
});

function summary(id: string, started: string): MeetingSummary {
  return { id, started, seconds: 1800, speakers: 3, lines: 120, words: 2400, has_audio: true, has_thumbnail: false, me_seconds: 400, talk_seconds: 1700 };
}

describe("past meetings under History", () => {
  const meetings = [
    summary("2026-10-29 16-10 Ünïcode review 🎧", "2026-10-29T16:10:00"),
    summary("2026-10-29 09-00 Standup", "2026-10-29T09:00:00"),
    summary("2026-10-28 09-00 Standup", "2026-10-28T09:00:00"),
    summary("2026-10-28 09-00 Standup (2)", "2026-10-28T09:00:00"),
    summary("2026-10-27 11-30", "2026-10-27T11:30:00"),
  ];

  it("finds meetings by name or date, ignoring case and accents typed as written", () => {
    expect(pastMeetings(meetings, null, "standup").map((meeting) => meeting.id)).toEqual([
      "2026-10-29 09-00 Standup",
      "2026-10-28 09-00 Standup",
      "2026-10-28 09-00 Standup (2)",
    ]);
    expect(pastMeetings(meetings, null, "ÜNÏCODE")).toHaveLength(1);
    expect(pastMeetings(meetings, null, "2026-10-27")).toHaveLength(1);
    expect(pastMeetings(meetings, null, "   ")).toHaveLength(5);
    expect(pastMeetings(meetings, null, "nothing like this")).toEqual([]);
  });

  it("leaves out the meeting being recorded until it is saved", () => {
    expect(pastMeetings(meetings, "2026-10-29 16-10 Ünïcode review 🎧", "").map((meeting) => meeting.id)).not.toContain("2026-10-29 16-10 Ünïcode review 🎧");
  });

  it("groups thousands of meetings under one heading per day", () => {
    const many = Array.from({ length: 3000 }, (_, index) => {
      const day = String(28 - Math.floor(index / 1000)).padStart(2, "0");
      return summary(`2026-10-${day} ${String(index % 24).padStart(2, "0")}-00 Call ${index}`, `2026-10-${day}T10:00:00`);
    });
    const items = groupByDay(many, new Date(2026, 9, 28, 12), (meeting) => meeting.started);
    expect(items.filter((item) => item.kind === "day").map((item) => (item.kind === "day" ? item.label : ""))).toEqual(["Today", "Yesterday", "Mon 26 Oct 2026"]);
    expect(items).toHaveLength(3003);
  });
});


describe("meeting stats on Home", () => {
  // A Wednesday afternoon, local time.
  const now = new Date(2026, 10, 4, 15, 0);
  const at = (daysAgo: number, hour = 10) => new Date(2026, 10, 4 - daysAgo, hour, 30).toISOString();
  const meeting = (id: string, started: string, seconds: number, lines = 10, speakers = 2): MeetingSummary => ({
    id, started, seconds, speakers, lines, words: 100, has_audio: true, has_thumbnail: false, me_seconds: 0, talk_seconds: seconds,
  });

  it("adds up every meeting, and only Monday to now for this week so far", () => {
    const meetings = [
      meeting("2026-11-04 Café planning ☕", at(0), 1800),
      meeting("2026-11-02 Weekly sync", at(2), 3600),
      meeting("2026-11-01 Sunday call", at(3), 600),
      meeting("2026-11-01 Sunday call", at(3), 600),
    ];

    const stats = meetingStats(meetings, now);

    expect(stats.count).toBe(4);
    expect(stats.seconds).toBe(6600);
    expect(stats.week).toEqual({ count: 2, seconds: 5400 });
  });

  it("compares this week so far with the same days of earlier weeks, once there are two", () => {
    const meetings = [
      meeting("this week", at(1), 1200),
      meeting("last week, Tuesday", at(8), 600),
      meeting("last week, Friday, after this weekday", at(5), 9000),
      meeting("two weeks ago, Monday", at(16), 1800),
    ];

    expect(meetingStats(meetings, now).usualWeek).toBe(1200);
    expect(meetingStats([meeting("only this week", at(0), 60), meeting("last week", at(7), 60)], now).usualWeek).toBeNull();
  });

  it("calls a typical meeting the median of those where someone spoke, not the mean", () => {
    const meetings = [
      meeting("short", at(1), 600, 12, 2),
      meeting("medium", at(2), 1800, 40, 3),
      meeting("marathon", at(3), 4.5 * 3600, 900, 6),
      meeting("silent", at(4), 0, 0, 1),
    ];

    const stats = meetingStats(meetings, now);

    expect(stats.typicalSeconds).toBe(1800);
    expect(stats.typicalSpeakers).toBe(3);
    expect(meetingStats([meeting("silent", at(0), 0, 0)], now).typicalSeconds).toBeNull();
  });

  it("charts twelve weeks of meeting time, this week last, with a usual line from complete weeks only", () => {
    const meetings = [
      meeting("this week", at(0), 600),
      meeting("last week", at(7), 3600),
      meeting("two weeks ago", at(14), 1800),
      meeting("eleven weeks ago", at(77), 1200),
      meeting("too old for the chart", at(84), 9000),
      meeting("no date", "", 900),
    ];

    const stats = meetingStats(meetings, now);

    expect(stats.weeks).toHaveLength(CHART_WEEKS);
    expect(stats.weeks.at(-1)).toBe(600);
    expect(stats.weeks.at(-2)).toBe(3600);
    expect(stats.weeks[0]).toBe(1200);
    expect(stats.usualFullWeek).toBe(0);
    expect(stats.count).toBe(5);
  });

  it("is empty with no meetings, and handles thousands", () => {
    const stats = meetingStats([], now);
    expect(stats).toMatchObject({ count: 0, seconds: 0, week: { count: 0, seconds: 0 }, usualWeek: null, typicalSeconds: null, usualFullWeek: null });
    expect(stats.weeks).toEqual(new Array(CHART_WEEKS).fill(0));
    const many = Array.from({ length: 5000 }, (_, index) => meeting(`m${index}`, at(index % 90), 60));
    expect(meetingStats(many, now).count).toBe(5000);
  });

  it("gives your share of talk over the last 30 days' meetings with you and others in them, beside an even split", () => {
    const talk = (id: string, daysAgo: number, me: number, total: number, speakers: number): MeetingSummary => ({
      ...meeting(id, at(daysAgo), total, 20, speakers), me_seconds: me, talk_seconds: total,
    });
    const meetings = [
      talk("one to one", 1, 600, 1200, 2),
      talk("team of four", 3, 300, 2400, 4),
      talk("only me", 2, 500, 500, 1),
      talk("only them", 4, 0, 900, 3),
      talk("too old", 31, 900, 1000, 2),
    ];

    const stats = meetingStats(meetings, now);

    expect(stats.talk?.meetings).toBe(2);
    expect(stats.talk?.share).toBeCloseTo(900 / 3600);
    expect(stats.talk?.even).toBeCloseTo((600 + 600) / 3600);
    expect(meetingStats([talk("only them", 1, 0, 900, 3)], now).talk).toBeNull();
  });

  it("takes the median of an odd or even count, and nothing of none", () => {
    expect(median([5, 1, 3])).toBe(3);
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([])).toBeNull();
  });
});

describe("meeting History cards", () => {
  it("draws the same cover bars for a meeting every time, and different ones for another", () => {
    const first = coverBars("2026-11-04 Café planning ☕");

    expect(first).toEqual(coverBars("2026-11-04 Café planning ☕"));
    expect(first).not.toEqual(coverBars("2026-11-04 Café planning"));
    expect(first).toHaveLength(28);
    expect(Math.min(...first)).toBeGreaterThanOrEqual(18);
    expect(Math.max(...first)).toBeLessThan(73);
    expect(coverBars("")).toHaveLength(28);
  });
});

import { describe, expect, it } from "vitest";
import { launchPath, modeOf } from "../app";
import { afterLeavingLivePage, applyMeetingEvent, colourOf, droppedMedia, percent, defaultLabel, labelOf, meetingName, speakerOrder, talkShares, turnsOf } from "./meetings";
import type { LiveMeeting, MeetingEntry } from "./types";

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
    expect(meetingName("2026-10-28 22-45 Marta email tool setup")).toBe("Marta email tool setup");
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
  it("puts the meetings pages in meeting mode and everything else in dictation", () => {
    expect(modeOf("/meetings")).toBe("meetings");
    expect(modeOf("/meetings/2026-10-16 Ops Onboarding")).toBe("meetings");
    expect(modeOf("/meeting-settings")).toBe("meetings");
    expect(modeOf("/history")).toBe("dictation");
    expect(modeOf("/transcription")).toBe("dictation");
    expect(modeOf("/meetingsroom")).toBe("dictation");
  });

  it("opens the meetings page when the tray asks for it", () => {
    expect(launchPath("meetings", null)).toBe("/meetings");
  });
});

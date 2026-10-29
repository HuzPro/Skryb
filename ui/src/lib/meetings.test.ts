import { describe, expect, it } from "vitest";
import { launchPath, modeOf } from "../app";
import { applyMeetingEvent, colourOf, defaultLabel, labelOf, meetingName, speakerOrder, talkShares, turnsOf } from "./meetings";
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

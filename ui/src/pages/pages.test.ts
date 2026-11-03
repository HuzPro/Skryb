import { describe, expect, it } from "vitest";
import { launchPath, restoredPath } from "../app";
import type { HistoryEntry, Model, Settings, UpdateStatus } from "../lib/types";
import { duplicateAlias, joinAliases, newestFirst, splitAliases, termFrom, termProblem } from "./dictionary";
import { finalText, formatDuration, groupByDay, preview } from "./history";
import { formatSaved, timeSaved } from "./home";
import { nextCue } from "./microphone";
import { normalizedProcessName, withOverride } from "./output";
import { setOptionalKeys } from "./shortcuts";
import { byTier, withMode } from "./transcription";
import { describeUpdate } from "./updates";

const entry = (at: string, text: string, extra: Partial<HistoryEntry> = {}): HistoryEntry => ({
  id: at, at, text, audio_seconds: 3, status: "", error: null, has_audio: true, typed: "", ...extra,
});

describe("the volume slider's preview", () => {
  it("plays the start tone first, then alternates with the stop tone", () => {
    const played: ("start" | "stop")[] = [];
    let last: "start" | "stop" | null = null;
    for (let at = 0; at < 1000; at += 1) played.push((last = nextCue(last)));
    expect(played.slice(0, 4)).toEqual(["start", "stop", "start", "stop"]);
    expect(played.filter((cue) => cue === "start")).toHaveLength(500);
  });
});

describe("dictionary list order", () => {
  it("shows the word added last at the top and keeps each rule's saved position", () => {
    const rules = Array.from({ length: 1500 }, (_, at) => `word-${at}`);
    const shown = newestFirst(rules);
    expect(shown[0]).toEqual({ rule: "word-1499", index: 1499 });
    expect(shown.at(-1)).toEqual({ rule: "word-0", index: 0 });
    expect(shown.every(({ rule, index }) => rules[index] === rule)).toBe(true);
    expect(rules[0]).toBe("word-0");
  });

  it("keeps duplicates and blanks apart by position", () => {
    expect(newestFirst(["café", "", "café"])).toEqual([
      { rule: "café", index: 2 }, { rule: "", index: 1 }, { rule: "café", index: 0 },
    ]);
    expect(newestFirst([])).toEqual([]);
  });
});

describe("dictionary aliases", () => {
  it("splits on commas and drops blanks", () => {
    expect(splitAliases("jason, jayson")).toEqual(["jason", "jayson"]);
    expect(splitAliases("jason,, ,jayson")).toEqual(["jason", "jayson"]);
    expect(splitAliases("jason, JSON ")).toEqual(["jason ", "JSON "]);
    expect(splitAliases(" , ,")).toEqual([]);
  });

  it("keeps a comma at the start or end of the box as part of the alias", () => {
    expect(splitAliases(", underscore")).toEqual([", underscore"]);
    expect(splitAliases(",underscore")).toEqual([",underscore"]);
    expect(splitAliases("underscore,")).toEqual(["underscore,"]);
    expect(splitAliases(", underscore, under score")).toEqual([", underscore", "under score"]);
    expect(splitAliases(" , dash ")).toEqual([" , dash "]);
    expect(joinAliases(splitAliases(", underscore, under score,"))).toBe(", underscore, under score,");
  });

  it("does not call a leading-comma alias a duplicate of the same word without it", () => {
    const rules = [
      { kind: "literal" as const, spoken: [" underscore "], replacement: "_", case_sensitive: false, whole_words: true, phonetic: false, enabled: true },
    ];
    const candidate = { ...rules[0]!, spoken: splitAliases(", underscore") };
    expect(duplicateAlias(rules, candidate, -1)).toBe(false);
  });

  it("glues every alias to the neighbouring text when the box starts or ends with a space", () => {
    expect(splitAliases(" dot com, dotcom")).toEqual([" dot com", " dotcom"]);
    expect(joinAliases([" dot com", " dotcom"])).toBe(" dot com, dotcom");
  });

  it("round-trips unicode aliases", () => {
    const aliases = splitAliases("zürich, 東京, café");
    expect(joinAliases(aliases)).toBe("zürich, 東京, café");
  });

  it("spots an alias another word rule already uses, ignoring the rule being edited", () => {
    const rules = [
      { kind: "literal" as const, spoken: ["jason"], replacement: "JSON", case_sensitive: false, whole_words: true, phonetic: false, enabled: true },
      { kind: "regex" as const, pattern: "jason", replacement: "x", enabled: true },
    ];
    const candidate = { ...rules[0]!, spoken: [" jason"] };
    expect(duplicateAlias(rules, candidate, -1)).toBe(true);
    expect(duplicateAlias(rules, candidate, 0)).toBe(false);
    expect(duplicateAlias(rules, rules[1]!, -1)).toBe(false);
  });
});

describe("people and terms", () => {
  const people = [
    { written: "Skryb", note: "the app I'm building", heard_as: ["Skribe"], phonetic: false },
    { written: "Ngozi Adébáyọ̀", note: "", heard_as: [], phonetic: true },
  ];

  it("trims an entry and splits its heard-as spellings without repeats or its own spelling", () => {
    expect(termFrom({ written: "  Odile ", note: " the   designer ", heard: "Odeel, , odeel, ODILE, Audile ", phonetic: true })).toEqual({
      written: "Odile", note: "the designer", heard_as: ["Odeel", "Audile"], phonetic: true,
    });
  });

  it("refuses a blank entry, a name twice, and a spelling that would swap one entry for another", () => {
    const term = (written: string, heard_as: string[] = []) => ({ written, note: "", heard_as, phonetic: false });
    expect(termProblem(people, term(""), -1)).toBe("Write the name or term the way it should appear.");
    expect(termProblem(people, term("skryb"), -1)).toBe("“skryb” is already in the list.");
    expect(termProblem(people, term("Scribe", ["skryb"]), -1)).toContain("is an entry of its own");
    expect(termProblem(people, term("Scryb", ["SKRIBE"]), -1)).toBe("“SKRIBE” is already heard as “Skryb”.");
    expect(termProblem(people, term("Skribe"), -1)).toContain("is a heard-as spelling of “Skryb”");
  });

  it("lets an entry be saved over itself, and a thousand distinct entries pass", () => {
    expect(termProblem(people, people[0]!, 0)).toBe("");
    const many = Array.from({ length: 1000 }, (_, at) => ({ written: `Name ${at}`, note: "", heard_as: [`Nayme ${at}`], phonetic: false }));
    expect(termProblem(many, { written: "Name 1000", note: "", heard_as: ["Nayme 1000"], phonetic: false }, -1)).toBe("");
  });
});

describe("the model lineup", () => {
  it("lists Small, Medium and Large first, then models outside the lineup in the order sent", () => {
    const model = (name: string, tier?: string) => ({ name, tier } as Model);
    const sorted = byTier([model("live"), model("qwen", "Large"), model("phonon", "Small"), model("other"), model("parakeet", "Medium")]);
    expect(sorted.map((m) => m.name)).toEqual(["phonon", "parakeet", "qwen", "live", "other"]);
    expect(byTier([])).toEqual([]);
  });
});

describe("history versions", () => {
  it("lists what was typed, and falls back to what was heard for older entries", () => {
    const typed = entry("2026-09-28T18:48:37", "four five", { typed: "Testing 1, 2, 3, 4, 5." });
    const older = entry("2026-09-20T09:00:00", "An old dictation café");

    expect(finalText(typed)).toBe("Testing 1, 2, 3, 4, 5.");
    expect(preview(typed)).toBe("Testing 1, 2, 3, 4, 5.");
    expect(preview(older)).toBe("An old dictation café");
  });
});

describe("history list", () => {
  it("groups two and a half thousand entries under one header per day", () => {
    const start = new Date(2026, 8, 27, 23, 30);
    const entries = Array.from({ length: 2500 }, (_, i) => {
      const at = new Date(start.getTime() - i * 37 * 60_000);
      const iso = `${at.getFullYear()}-${String(at.getMonth() + 1).padStart(2, "0")}-${String(at.getDate()).padStart(2, "0")}T${String(at.getHours()).padStart(2, "0")}:${String(at.getMinutes()).padStart(2, "0")}:00`;
      return entry(iso, i % 7 ? `dictation ${i}` : "");
    });
    const items = groupByDay(entries, new Date(2026, 8, 27, 23, 59), (entry) => entry.at);
    const days = items.filter((item) => item.kind === "day");
    expect(items.length - days.length).toBe(2500);
    expect(days[0]).toEqual({ kind: "day", label: "Today" });
    expect(days[1]).toEqual({ kind: "day", label: "Yesterday" });
    expect(days[2]).toEqual({ kind: "day", label: "Fri 25 Sep 2026" });
    expect(new Set(days.map((day) => day.kind === "day" && day.label)).size).toBe(days.length);
  });

  it("previews one line of text, or says why there is none", () => {
    expect(preview(entry("2026-09-27T10:00:00", "  line one\n\nline   two "))).toBe("line one line two");
    expect(preview(entry("2026-09-27T10:00:00", "", { status: "Cancelled" }))).toBe("(cancelled - not transcribed)");
    expect(preview(entry("2026-09-27T10:00:00", " ", { status: "Failed", error: "CUDA error 2" }))).toBe("(CUDA error 2)");
    expect(preview(entry("2026-09-27T10:00:00", "", { status: "Empty" }))).toBe("(no text)");
  });

  it("formats lengths like the tray does", () => {
    expect([0, 0.9, 59.99, 61, 3599, 3600, 3661.5].map(formatDuration)).toEqual(["0:00", "0:00", "0:59", "1:01", "59:59", "1:00:00", "1:01:01"]);
  });
});

describe("settings helpers", () => {
  it("normalises app names the way the core does", () => {
    expect(normalizedProcessName(" Mintty ")).toBe("mintty.exe");
    expect(normalizedProcessName("NOTEPAD.EXE")).toBe("notepad.exe");
    expect(normalizedProcessName("   ")).toBe("");
  });

  it("keeps one paste method per app, sorted, when an app is added twice", () => {
    const once = withOverride([["zed.exe", "ctrl_v"]], "code.exe", "type");
    expect(withOverride(once, "code.exe", "shift_insert")).toEqual([["code.exe", "shift_insert"], ["zed.exe", "ctrl_v"]]);
  });

  it("adds, changes and removes an optional shortcut without touching dictation ones", () => {
    const base = [{ keys: "ctrl+alt+m", mode: "toggle" as const, action: "dictate" as const }];
    const added = setOptionalKeys(base, "paste_last", "ctrl+alt+v");
    expect(added[1]).toEqual({ keys: "ctrl+alt+v", mode: "press", action: "paste_last" });
    expect(setOptionalKeys(added, "paste_last", "f13")[1]?.keys).toBe("f13");
    expect(setOptionalKeys(added, "paste_last", "")).toEqual(base);
  });

  it("switching to Live picks an installed streaming model and back to Batch keeps a batch one", () => {
    const models = [
      { name: "parakeet", supports_batch: true, supports_streaming: false, installed: true },
      { name: "nemotron-a", supports_batch: false, supports_streaming: true, installed: false },
      { name: "nemotron-b", supports_batch: false, supports_streaming: true, installed: true },
    ] as Model[];
    const settings = { model_size: "parakeet", transcription_mode: "batch", engine: "nano_parakeet" } as Settings;
    const live = withMode(settings, "streaming", models);
    expect(live).toMatchObject({ transcription_mode: "streaming", model_size: "nemotron-b", engine: "auto" });
    expect(withMode(live, "batch", models)).toMatchObject({ transcription_mode: "batch", model_size: "parakeet" });
    expect(withMode(settings, "batch", models)).toMatchObject({ model_size: "parakeet", engine: "nano_parakeet" });
  });
});

describe("time saved on Home", () => {
  it("is typing time at 40 words a minute minus the time spent speaking", () => {
    expect(timeSaved({ words: 400, seconds: 120 })).toBe(480);
    expect(formatSaved(timeSaved({ words: 400, seconds: 120 }))).toBe("8 min");
  });

  it("never goes below nothing, and an empty history saved nothing", () => {
    expect(timeSaved({ words: 3, seconds: 600 })).toBe(0);
    expect(timeSaved({ words: 0, seconds: 0 })).toBe(0);
    expect(formatSaved(0)).toBe("0 min");
  });

  it("reads in minutes, then hours, for a year of dictation", () => {
    expect(formatSaved(20)).toBe("Under a minute");
    expect(formatSaved(3 * 3600 + 5 * 60)).toBe("3 h 5 min");
    const year = timeSaved({ words: 1_250_000, seconds: 180_000 });
    expect(formatSaved(year)).toBe(`${(470).toLocaleString()} h 50 min`);
  });
});

describe("the page a launch opens", () => {
  it("opens the window on Home, the tray's Meetings on meeting Home, and History, setup and updates on themselves", () => {
    expect(launchPath("settings")).toBe("/home");
    expect(launchPath("meetings")).toBe("/meeting-home");
    expect(launchPath("history")).toBe("/history");
    expect(launchPath("setup")).toBe("/setup");
    expect(launchPath("updates")).toBe("/updates");
  });

  it("reopens the page an update handed over, unless this version has no such page", () => {
    expect(restoredPath("/updates")).toBe("/updates");
    expect(restoredPath("/meeting-home")).toBe("/meeting-home");
    expect(restoredPath("/meetings/live")).toBeNull();
    expect(restoredPath("/../etc")).toBeNull();
    expect(restoredPath(null)).toBeNull();
  });
});

describe("updates page", () => {
  const status = (extra: Partial<UpdateStatus>): UpdateStatus => ({ state: "idle", version: "", bytes_done: 0, bytes_total: 0, error: "", ...extra });

  it("tells a git checkout that it updates through git", () => {
    expect(describeUpdate(null, "0.2.0").description).toMatch(/updates through git/);
  });

  it("shows whole percents while downloading, never past 100", () => {
    expect(describeUpdate(status({ state: "downloading", version: "0.3.0", bytes_done: 15_486_480, bytes_total: 30_972_960 }), "0.2.0")).toEqual({
      title: "Downloading Skryb 0.3.0",
      description: "50% downloaded. You choose when to install it.",
    });
    expect(describeUpdate(status({ state: "downloading", version: "0.3.0", bytes_done: 30_972_959, bytes_total: 30_972_960 }), "0.2.0").description).toMatch(/^99%/);
    expect(describeUpdate(status({ state: "downloading", version: "0.3.0" }), "0.2.0").description).toMatch(/^0%/);
  });

  it("names the waiting version once it is ready", () => {
    expect(describeUpdate(status({ state: "ready", version: "0.3.0" }), "0.2.0").title).toBe("Skryb 0.3.0 is ready");
  });

  it("keeps naming the running version while checking, up to date, off or failed", () => {
    for (const state of ["checking", "up_to_date", "off", "failed", "idle"] as const) {
      expect(describeUpdate(status({ state }), "0.2.0").title).toBe("Skryb 0.2.0");
    }
  });
});

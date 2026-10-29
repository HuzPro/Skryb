import { describe, expect, it } from "vitest";
import { launchPath, rememberPage } from "../app";
import type { HistoryEntry, Model, Settings, UpdateStatus } from "../lib/types";
import { duplicateAlias, joinAliases, newestFirst, splitAliases } from "./dictionary";
import { finalText, formatDuration, groupByDay, preview } from "./history";
import { nextCue } from "./microphone";
import { normalizedProcessName, withOverride } from "./output";
import { setOptionalKeys } from "./shortcuts";
import { withMode } from "./transcription";
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
      { kind: "literal" as const, spoken: [" underscore "], replacement: "_", case_sensitive: false, whole_words: true, phonetic: false, enabled: true, learned: false },
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
      { kind: "literal" as const, spoken: ["jason"], replacement: "JSON", case_sensitive: false, whole_words: true, phonetic: false, enabled: true, learned: false },
      { kind: "regex" as const, pattern: "jason", replacement: "x", enabled: true },
    ];
    const candidate = { ...rules[0]!, spoken: [" jason"] };
    expect(duplicateAlias(rules, candidate, -1)).toBe(true);
    expect(duplicateAlias(rules, candidate, 0)).toBe(false);
    expect(duplicateAlias(rules, rules[1]!, -1)).toBe(false);
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
    const items = groupByDay(entries, new Date(2026, 8, 27, 23, 59));
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

describe("the page a launch opens", () => {
  const memory = () => {
    const saved = new Map<string, string>();
    return { getItem: (key: string) => saved.get(key) ?? null, setItem: (key: string, value: string) => void saved.set(key, value) };
  };

  it("opens settings on the Model page the first time, and History and setup on themselves", () => {
    const fresh = memory();
    expect(launchPath("settings", fresh)).toBe("/transcription");
    expect(launchPath("history", fresh)).toBe("/history");
    expect(launchPath("setup", fresh)).toBe("/setup");
    expect(launchPath("settings", null)).toBe("/transcription");
  });

  it("reopens settings on the settings page shown last, never on History or setup", () => {
    const used = memory();
    for (const path of ["/dictionary", "/shortcuts", "/history", "/setup", "/nowhere", "/"]) rememberPage(path, used);
    expect(launchPath("settings", used)).toBe("/shortcuts");
    expect(launchPath("history", used)).toBe("/history");
  });

  it("falls back to the Model page when the stored page is unknown or storage is blocked", () => {
    const tampered = memory();
    tampered.setItem("last-settings-page", "../../etc");
    expect(launchPath("settings", tampered)).toBe("/transcription");
    const blocked = {
      getItem: () => {
        throw new Error("SecurityError");
      },
      setItem: () => {
        throw new Error("QuotaExceededError");
      },
    };
    expect(() => rememberPage("/output", blocked)).not.toThrow();
    expect(launchPath("settings", blocked)).toBe("/transcription");
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

import { describe, expect, it } from "vitest";
import { launchPath, rememberPage } from "../app";
import type { HistoryEntry, Model, Settings, UpdateStatus, WritingRecord } from "../lib/types";
import { duplicateAlias, joinAliases, splitAliases } from "./dictionary";
import { describeOutcome, finalText, formatDuration, groupByDay, hasVersions, preview } from "./history";
import { normalizedProcessName, withOverride } from "./output";
import { setOptionalKeys } from "./shortcuts";
import { withMode } from "./transcription";
import { describeUpdate } from "./updates";
import { describeTrackRecord, fileName } from "./writing";

const entry = (at: string, text: string, extra: Partial<HistoryEntry> = {}): HistoryEntry => ({
  id: at, at, text, audio_seconds: 3, status: "", error: null, has_audio: true, kind: "dictation", writing: null, ...extra,
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

const writing = (extra: Partial<WritingRecord> = {}): WritingRecord => ({
  outcome: "applied", reason: "", seconds: 0.42, answer: "", typed: "", context_before: "", window: "", ...extra,
});

describe("history versions", () => {
  it("lists what was typed, and falls back to what was heard for older entries", () => {
    const typed = entry("2026-09-28T18:48:37", "four five", { writing: writing({ typed: "Testing 1, 2, 3, 4, 5." }) });
    const older = entry("2026-09-20T09:00:00", "An old dictation café");

    expect(finalText(typed)).toBe("Testing 1, 2, 3, 4, 5.");
    expect(preview(typed)).toBe("Testing 1, 2, 3, 4, 5.");
    expect(preview(older)).toBe("An old dictation café");
  });

  it("only offers a comparison when there is more than one version to compare", () => {
    expect(hasVersions(entry("2026-09-28T10:00:00", "hello", { writing: writing({ answer: "Hello." }) }))).toBe(true);
    expect(hasVersions(entry("2026-09-28T10:00:00", "hello", { writing: writing({ typed: "Hello there." }) }))).toBe(true);
    expect(hasVersions(entry("2026-09-28T10:00:00", "hello ", { writing: writing({ typed: "hello" }) }))).toBe(false);
    expect(hasVersions(entry("2026-09-28T10:00:00", "hello", { writing: writing({ outcome: "skipped" }) }))).toBe(false);
    expect(hasVersions(entry("2026-09-28T10:00:00", "hello"))).toBe(false);
  });

  it("says what the writing model did in plain words", () => {
    expect(describeOutcome(writing())).toEqual({ text: "Writing model applied in 0.42 s.", bad: false });
    expect(describeOutcome(writing({ outcome: "rejected", reason: "a number changed" }))).toEqual({ text: "Writing model answer not used: a number changed.", bad: true });
    expect(describeOutcome(writing({ outcome: "too_slow", reason: "it took 4.50s, over the 4.0s limit" })).text).toBe("Writing model too slow: it took 4.50s, over the 4.0s limit.");
    expect(describeOutcome(writing({ outcome: "skipped", reason: "the writing model is busy" })).bad).toBe(true);
    expect(describeOutcome(writing({ outcome: "", seconds: null })).text).toBe("Writing model not asked (cleanup was off).");
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

  it("opens settings on Transcription the first time, and History and setup on themselves", () => {
    const fresh = memory();
    expect(launchPath("settings", fresh)).toBe("/transcription");
    expect(launchPath("history", fresh)).toBe("/history");
    expect(launchPath("setup", fresh)).toBe("/setup");
    expect(launchPath("settings", null)).toBe("/transcription");
  });

  it("reopens settings on the settings page shown last, never on History or setup", () => {
    const used = memory();
    for (const path of ["/writing", "/shortcuts", "/history", "/setup", "/nowhere", "/"]) rememberPage(path, used);
    expect(launchPath("settings", used)).toBe("/shortcuts");
    expect(launchPath("history", used)).toBe("/history");
  });

  it("falls back to Transcription when the stored page is unknown or storage is blocked", () => {
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

describe("writing model file", () => {
  it("names the file in use, whether it is the installed one or an override", () => {
    expect(fileName("C:\\Users\\Zoë\\AppData\\Local\\Skryb\\writing\\models\\Spark-X2.5-4B-Q4_K_M.gguf")).toBe("Spark-X2.5-4B-Q4_K_M.gguf");
    expect(fileName("D:/models/Qwen3.5-4B-Q4_K_M.gguf")).toBe("Qwen3.5-4B-Q4_K_M.gguf");
    expect(fileName("")).toBe("");
  });
});

describe("writing model track record", () => {
  it("counts the outcomes in plain words and lists why answers were not used", () => {
    const lines = describeTrackRecord({
      dictations: 50,
      outcomes: { applied: 38, rejected: 7, too_slow: 5 },
      rejections: [{ reason: "a number changed", count: 4 }, { reason: "a hedge was dropped", count: 3 }],
    });

    expect(lines).toEqual([
      "Last 50 dictations: 38 applied, 7 not used, 5 too slow.",
      "Not used because a number changed (4), a hedge was dropped (3).",
    ]);
  });

  it("reads right for one dictation and for none", () => {
    expect(describeTrackRecord({ dictations: 1, outcomes: { skipped: 1 }, rejections: [] })).toEqual(["Last 1 dictation: 1 skipped."]);
    expect(describeTrackRecord({ dictations: 0, outcomes: {}, rejections: [] })).toEqual(["No dictations have gone through the writing model yet."]);
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

import { describe, expect, it } from "vitest";
import type { HistoryEntry, Model, Settings } from "../lib/types";
import { duplicateAlias, joinAliases, splitAliases } from "./dictionary";
import { formatDuration, groupByDay, preview } from "./history";
import { normalizedProcessName, withOverride } from "./output";
import { setOptionalKeys } from "./shortcuts";
import { withMode } from "./transcription";

const entry = (at: string, text: string, extra: Partial<HistoryEntry> = {}): HistoryEntry => ({
  id: at, at, text, audio_seconds: 3, status: "", error: null, has_audio: true, kind: "dictation", ...extra,
});

describe("dictionary aliases", () => {
  it("splits on commas and drops blanks", () => {
    expect(splitAliases("jason, jayson")).toEqual(["jason", "jayson"]);
    expect(splitAliases("jason,, ,jayson,")).toEqual(["jason", "jayson"]);
    expect(splitAliases("jason, JSON ")).toEqual(["jason ", "JSON "]);
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

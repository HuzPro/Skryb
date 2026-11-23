import { QueryObserver } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import { connectTo } from "./api";
import { queries, queryClient } from "./queries";
import { byUse, controlName, deleteUsageData, openedPage, pageName, setTelemetry, settingEvents, track } from "./telemetry";
import type { Settings } from "./types";

const settings: Settings = {
  model_size: "parakeet-ultra",
  language: "en",
  device: "auto",
  engine: "auto",
  output_mode: "direct_paste",
  transcription_mode: "batch",
  question_marks: true,
  capture_content: true,
  capture_edits: false,
  input_device: "Microphone (Blue Yeti)",
  boost_quiet_speech: true,
  hotkey_bindings: [{ keys: "ctrl+alt+m", mode: "toggle", action: "dictate" }],
  double_tap_locks_hold: true,
  recording: {
    warn_when_silent: true,
    preferred_devices: ["Microphone (Blue Yeti)", "Laptop Array Microphone"],
    cues_enabled: true,
    cue_volume: 0.6,
    duck_other_audio: "off",
    duck_level: 0.2,
    max_duration_seconds: 600,
    warn_before_cap_seconds: 30,
  },
  history_retention_days: 90,
  paste_method_overrides: [["notepad.exe", "shift_insert"]],
  check_for_updates: true,
  meetings: { folder: "D:\\Meetings", record_microphone: true, language: "en" },
  show_english: false,
  telemetry: true,
  theme: "material",
  colourways: { default: "ink", glass: "ink", material: "green", paper: "green" },
};

describe("what a page is called in usage data", () => {
  it("is its route, and every saved meeting is the same page", () => {
    expect(pageName("/history")).toBe("history");
    expect(pageName("/meeting-home")).toBe("meeting-home");
    expect(pageName("/meetings")).toBe("meetings");
    expect(pageName("/meetings/2026-11-20-weekly-review")).toBe("meetings.saved");
    expect(pageName("/meetings/اجتماع الفريق")).toBe("meetings.saved");
    expect(pageName("/")).toBe("");
  });
});

describe("what a control is called in usage data", () => {
  it("is the name it was given, before its icon", () => {
    expect(controlName("delete-all", ["lucide", "lucide-trash-2"], null, "BUTTON")).toBe("delete-all");
  });

  it("is its icon when it has no name, never the words on it", () => {
    expect(controlName(null, ["lucide", "lucide-copy", "lucide-copy-icon"], null, "BUTTON")).toBe("copy");
    expect(controlName("", ["size-4", "lucide", "lucide-rotate-ccw"], "option", "DIV")).toBe("rotate-ccw");
  });

  it("is its role or its tag when it has neither a name nor an icon", () => {
    expect(controlName(null, [], "switch", "BUTTON")).toBe("switch");
    expect(controlName(null, [], null, "BUTTON")).toBe("button");
    expect(controlName(null, ["size-4"], null, "A")).toBe("a");
  });
});

describe("settings saved, as usage events", () => {
  it("records nothing when nothing changed", () => {
    expect(settingEvents(settings, structuredClone(settings))).toEqual([]);
  });

  it("records a switch, a number and a fixed choice with the new value", () => {
    const next: Settings = {
      ...settings,
      question_marks: false,
      history_retention_days: 0,
      model_size: "qwen3-asr-1.7b",
      recording: { ...settings.recording, cue_volume: 0.25, duck_other_audio: "lower" },
      meetings: { ...settings.meetings, language: "ar" },
    };

    expect(settingEvents(settings, next)).toEqual([
      { kind: "setting", name: "model_size", value: "qwen3-asr-1.7b" },
      { kind: "setting", name: "question_marks", value: false },
      { kind: "setting", name: "history_retention_days", value: 0 },
      { kind: "setting", name: "recording.cue_volume", value: 0.25 },
      { kind: "setting", name: "recording.duck_other_audio", value: "lower" },
      { kind: "setting", name: "meetings.language", value: "ar" },
    ]);
  });

  it("records that a folder, a microphone, a shortcut or an app list changed, and never what to", () => {
    const next: Settings = {
      ...settings,
      input_device: "Headset (Jabra Evolve2 65 Mono)",
      hotkey_bindings: [...settings.hotkey_bindings, { keys: "mouse4", mode: "hold", action: "dictate" }],
      paste_method_overrides: [],
      recording: { ...settings.recording, preferred_devices: [] },
      meetings: { ...settings.meetings, folder: "C:\\Users\\someone\\Private Meetings" },
    };

    const events = settingEvents(settings, next);

    expect(events).toEqual([
      { kind: "setting", name: "input_device" },
      { kind: "setting", name: "hotkey_bindings" },
      { kind: "setting", name: "paste_method_overrides" },
      { kind: "setting", name: "recording.preferred_devices" },
      { kind: "setting", name: "meetings.folder" },
    ]);
    expect(JSON.stringify(events)).not.toMatch(/Jabra|someone|mouse4|Private/);
  });

  it("records a new colour once, as the colour of the theme in use", () => {
    const next: Settings = { ...settings, colourways: { default: "plum", glass: "plum", material: "plum", paper: "plum" } };

    expect(settingEvents(settings, next)).toEqual([{ kind: "setting", name: "colourway", value: "plum" }]);
  });
});

describe("putting what is used most first", () => {
  const actions = ["copy", "play", "retry", "save-audio"];
  const key = (name: string) => `click:history.${name}`;

  it("keeps the given order until there is enough use to go on", () => {
    expect(byUse(actions, {}, key)).toEqual(actions);
    expect(byUse(actions, { "click:history.save-audio": 19 }, key)).toEqual(actions);
    expect(byUse(actions, { "click:history.save-audio": 2 }, key, 3)).toEqual(actions);
  });

  it("moves the most used to the front and leaves ties in the given order", () => {
    const counts = { "click:history.play": 340, "click:history.copy": 1200, "click:history.retry": 0, "click:home.play": 9000 };

    expect(byUse(actions, counts, key)).toEqual(["copy", "play", "retry", "save-audio"]);
    expect(byUse(actions, { ...counts, "click:history.save-audio": 1201 }, key)).toEqual(["save-audio", "copy", "play", "retry"]);
    expect(byUse(actions, { "click:history.play": 10, "click:history.retry": 10 }, key)).toEqual(["play", "retry", "copy", "save-audio"]);
  });

  it("orders thousands of items the same way every time", () => {
    const items = Array.from({ length: 5000 }, (_, at) => `item-${at}`);
    const counts = Object.fromEntries(items.map((item, at) => [item, at % 7]));

    const ordered = byUse(items, counts, (item) => item);

    expect(ordered).toHaveLength(5000);
    expect(new Set(ordered).size).toBe(5000);
    expect(ordered.slice(0, 3)).toEqual(["item-6", "item-13", "item-20"]);
    expect(ordered.at(-1)).toBe("item-4998");
    expect(byUse(items, counts, (item) => item)).toEqual(ordered);
  });
});

/** A core that keeps the events it is sent while usage data is on, and counts how often it was asked for the count. */
function fakeCore() {
  const core = { keeping: true, kept: 0, asked: 0, batches: 0 };
  vi.stubGlobal("fetch", async (url: string, init: { method: string; body?: string }) => {
    const path = new URL(url).pathname;
    if (path !== "/api/telemetry") return new Response(JSON.stringify({ error: "Not found." }), { status: 404 });
    if (init.method === "GET") {
      core.asked += 1;
      return new Response(JSON.stringify({ events: core.kept, counts: {} }));
    }
    if (init.method === "DELETE") core.kept = 0;
    else {
      core.batches += 1;
      if (core.keeping) core.kept += (JSON.parse(init.body ?? "{}") as { events: unknown[] }).events.length;
    }
    return new Response(null, { status: 204 });
  });
  connectTo(8765, "dev");
  return core;
}

/** The number Settings shows while its usage data row is on screen. */
function watchCount() {
  const observer = new QueryObserver(queryClient, { ...queries.telemetry, staleTime: 0 });
  const stop = observer.subscribe(() => undefined);
  return { shown: () => observer.getCurrentResult().data?.events, stop };
}

/** Enough clicks that the batch goes to the core at once, as it does after five seconds. */
function clicks(count: number): void {
  for (let n = 0; n < count; n++) track({ kind: "click", name: `updates.button-${n % 7}` });
}

describe("the usage data count on screen", () => {
  afterEach(() => {
    setTelemetry(false);
    queryClient.clear();
    vi.unstubAllGlobals();
  });

  it("follows new events as they are kept and goes to 0 after Delete", async () => {
    const core = fakeCore();
    core.kept = 88;
    setTelemetry(true);
    openedPage("/updates");
    const count = watchCount();
    await vi.waitFor(() => expect(count.shown()).toBe(88));

    clicks(50);
    await vi.waitFor(() => expect(count.shown()).toBe(138));
    for (let batch = 0; batch < 20; batch++) clicks(50);
    await vi.waitFor(() => expect(count.shown()).toBe(1138));

    await deleteUsageData();
    expect(count.shown()).toBe(0);
    count.stop();
  });

  it("stops changing once usage data is turned off", async () => {
    const core = fakeCore();
    setTelemetry(true);
    openedPage("/updates");
    const count = watchCount();
    await vi.waitFor(() => expect(count.shown()).toBe(0));
    clicks(50);
    await vi.waitFor(() => expect(count.shown()).toBe(50));

    setTelemetry(false);
    clicks(500);
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(core.batches).toBe(1);
    expect(count.shown()).toBe(50);
    count.stop();
  });

  it("isn't asked for while the row is off screen", async () => {
    const core = fakeCore();
    setTelemetry(true);
    openedPage("/history");
    const count = watchCount();
    await vi.waitFor(() => expect(count.shown()).toBe(0));
    count.stop();
    const asked = core.asked;

    for (let batch = 0; batch < 10; batch++) clicks(50);
    await vi.waitFor(() => expect(core.kept).toBe(500));
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(core.asked).toBe(asked);
  });
});

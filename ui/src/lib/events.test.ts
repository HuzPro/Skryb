import { describe, expect, it } from "vitest";
import { reduceEvent, type Snapshot } from "./events";
import type { AppInfo, Model, Settings } from "./types";

const model = (name: string, installed = false): Model => ({
  name,
  display_size: "2.5 GB",
  description: "",
  installed,
  supports_batch: true,
  supports_streaming: false,
  has_speech_styles: false,
  license_id: null,
  download: null,
});

const app: AppInfo = {
  version: "0.9.0",
  state: "idle",
  engine: { ready: true, loading: false, error: null, model: "parakeet-tdt-0.6b-v3" },
  first_run: false,
  diagnostics_directory: "D:\\logs",
  history_directory: "D:\\logs",
  pill_default_position: [0.5, 1],
  update: null,
};

const models = [model("parakeet-tdt-0.6b-v3", true), model("crisperwhisper-2.0-turbo"), model("crisperwhisper-2.0-large")];

describe("server events", () => {
  it("moves the progress of the downloading model and leaves the others alone", () => {
    const { snapshot, refetch } = reduceEvent({ models }, "download", {
      name: "crisperwhisper-2.0-turbo", state: "downloading", bytes_done: 500, bytes_total: 1630, error: "",
    });
    expect(snapshot.models?.[1]?.download).toEqual({ state: "downloading", bytes_done: 500, bytes_total: 1630, error: "" });
    expect(snapshot.models?.[0]).toBe(models[0]);
    expect(snapshot.models?.[2]).toBe(models[2]);
    expect(refetch).toEqual([]);
  });

  it("marks a finished download installed and asks for a fresh model list", () => {
    const { snapshot, refetch } = reduceEvent({ models }, "download", {
      name: "crisperwhisper-2.0-turbo", state: "completed", bytes_done: 1630, bytes_total: 1630, error: "",
    });
    expect(snapshot.models?.[1]).toMatchObject({ installed: true, download: null });
    expect(refetch).toEqual(["models"]);
  });

  it("clears the bar of a cancelled download and keeps a failed one with its reason", () => {
    const cancelled = reduceEvent({ models }, "download", { name: "crisperwhisper-2.0-large", state: "cancelled", bytes_done: 0, bytes_total: 0, error: "" });
    expect(cancelled.snapshot.models?.[2]?.download).toBeNull();
    const failed = reduceEvent({ models }, "download", { name: "crisperwhisper-2.0-large", state: "failed", bytes_done: 9, bytes_total: 3100, error: "Disk full" });
    expect(failed.snapshot.models?.[2]?.download).toMatchObject({ state: "failed", error: "Disk full" });
  });

  it("marks a deleted model not installed and asks for a fresh list", () => {
    const { snapshot, refetch } = reduceEvent({ models }, "download", { name: "parakeet-tdt-0.6b-v3", state: "deleted", bytes_done: 0, bytes_total: 0, error: "" });
    expect(snapshot.models?.[0]).toMatchObject({ installed: false, download: null });
    expect(refetch).toEqual(["models"]);
  });

  it("refetches the microphones when one is plugged in", () => {
    expect(reduceEvent({ models }, "microphones", { name: "USB Microphone (Røde NT-USB Mini)", will_be_used: true })).toEqual({ snapshot: { models }, refetch: ["microphones"] });
  });

  it("refetches the list when a download names a model it does not know", () => {
    const { snapshot, refetch } = reduceEvent({ models }, "download", { name: "new-model", state: "downloading", bytes_done: 1, bytes_total: 2, error: "" });
    expect(snapshot.models).toBe(models);
    expect(refetch).toEqual(["models"]);
    expect(reduceEvent({}, "download", { name: "x", state: "downloading", bytes_done: 1, bytes_total: 2, error: "" }).refetch).toEqual(["models"]);
  });

  it("ends on the last of a thousand progress events", () => {
    let snapshot: Snapshot = { models };
    for (let done = 0; done <= 1000; done++) {
      snapshot = reduceEvent(snapshot, "download", { name: "crisperwhisper-2.0-large", state: "downloading", bytes_done: done, bytes_total: 1000, error: "" }).snapshot;
    }
    expect(snapshot.models?.[2]?.download?.bytes_done).toBe(1000);
    expect(snapshot.models?.[0]).toBe(models[0]);
  });

  it("reads the core's engine event, where state and flags come together", () => {
    const failed = reduceEvent({ app }, "engine", { state: "failed", loading: false, ready: false, error: "Model file missing", model: "crisperwhisper-2.0-large" });
    expect(failed.snapshot.app?.engine).toEqual({ ready: false, loading: false, error: "Model file missing", model: "crisperwhisper-2.0-large" });
    const ready = reduceEvent({ app }, "engine", { state: "ready", loading: false, ready: true, error: null, model: "crisperwhisper-2.0-large" });
    expect(ready.snapshot.app?.engine).toEqual({ ready: true, loading: false, error: null, model: "crisperwhisper-2.0-large" });
  });

  it("follows the engine through loading, ready and failed in either event shape", () => {
    const loading = reduceEvent({ app }, "engine", { state: "loading", error: null, model: "crisperwhisper-2.0-turbo" }).snapshot.app?.engine;
    expect(loading).toEqual({ ready: false, loading: true, error: null, model: "crisperwhisper-2.0-turbo" });
    const ready = reduceEvent({ app }, "engine", { loading: false, ready: true, error: null, model: "m" }).snapshot.app?.engine;
    expect(ready).toMatchObject({ ready: true, loading: false });
    const failed = reduceEvent({ app }, "engine", { state: "failed", error: "CUDA out of memory" }).snapshot.app?.engine;
    expect(failed).toEqual({ ready: false, loading: false, error: "CUDA out of memory", model: "parakeet-tdt-0.6b-v3" });
    const failedWithoutReason = reduceEvent({ app }, "engine", { state: "failed", error: "" }).snapshot.app?.engine;
    expect(failedWithoutReason?.error).toBe("The model could not be loaded.");
  });

  it("asks for the app info when an engine event arrives before it", () => {
    expect(reduceEvent({}, "engine", { state: "ready" })).toEqual({ snapshot: {}, refetch: ["app"] });
  });

  it("replaces app and settings wholesale", () => {
    const next = { ...app, state: "recording" as const };
    expect(reduceEvent({ app }, "app", next).snapshot.app).toBe(next);
    const settings = { model_size: "parakeet-tdt-0.6b-v3", learn_words: false } as Settings;
    expect(reduceEvent({}, "settings", settings).snapshot.settings).toBe(settings);
  });

  it("refetches history after a new dictation or a wipe, not after a retry", () => {
    expect(reduceEvent({}, "history", { event: "changed" }).refetch).toEqual(["history"]);
    expect(reduceEvent({}, "history", { event: "deleted_all" }).refetch).toEqual(["history"]);
    expect(reduceEvent({}, "history", { event: "retry_done", id: "a", text: "b" }).refetch).toEqual([]);
  });

  it("refetches engine candidates when the hardware probe reports", () => {
    expect(reduceEvent({}, "hardware", { probe_pending: false }).refetch).toEqual(["engines"]);
  });

  it("changes nothing for heartbeats, meter levels and unknown events", () => {
    const snapshot: Snapshot = { app, models };
    for (const name of ["heartbeat", "meter", "capture", "something-new"]) {
      expect(reduceEvent(snapshot, name, { level: 0.4 })).toEqual({ snapshot, refetch: [] });
    }
  });
});

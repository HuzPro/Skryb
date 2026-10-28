import { describe, expect, it } from "vitest";
import type { AppInfo, AppState } from "../lib/types";
import { motionFor } from "./mark";

const app = (state: AppState, loading = false): AppInfo => ({
  version: "1.0.0",
  state,
  engine: { loading, ready: !loading, error: null, model: "parakeet" },
  first_run: false,
  diagnostics_directory: "",
  history_directory: "",
  pill_default_position: [0.5, 1],
  update: null,
});

describe("the logo's motion", () => {
  it("stands still while idle, failed, or before the core answers", () => {
    expect(motionFor(app("idle"))).toBe("still");
    expect(motionFor(app("error"))).toBe("still");
    expect(motionFor(undefined)).toBe("still");
  });

  it("follows the voice while recording", () => {
    expect(motionFor(app("recording"))).toBe("voice");
  });

  it("streams into the cursor while transcribing", () => {
    expect(motionFor(app("transcribing"))).toBe("feed");
  });

  it("sweeps while a model loads, even when the app itself reads idle", () => {
    expect(motionFor(app("loading"))).toBe("sweep");
    expect(motionFor(app("idle", true))).toBe("sweep");
  });
});

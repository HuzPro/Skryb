import { describe, expect, it } from "vitest";
import { restoredPath } from "../app";
import { installDescription, installTitle } from "../components/install";
import type { UpdateStatus } from "../lib/types";
import { describeUpdate } from "./updates";

const status = (overrides: Partial<UpdateStatus>): UpdateStatus => ({ state: "idle", version: "", bytes_done: 0, bytes_total: 0, error: "", ...overrides });

describe("the Updates page after an update", () => {
  it("says the new version was installed just now", () => {
    expect(describeUpdate(status({ state: "updated", version: "0.4.0" }), "0.4.0")).toEqual({
      title: "Skryb 0.4.0",
      description: "Updated just now. You have the latest version.",
    });
  });
});

describe("the update installing while Skryb is closed", () => {
  it("names the version, the step and the time left", () => {
    const view = { progress: { step: "copying" as const, percent: 50, version: "0.4.0", log: "" }, estimate: { fraction: 0.2, label: "Copying files", secondsLeft: 35 } };

    expect(installTitle(view)).toBe("Installing Skryb 0.4.0");
    expect(installDescription(view)).toBe("Copying files · About 35 seconds left");
  });

  it("says a failed update didn't finish, how to carry on, and where the log is", () => {
    const log = "C:\\Users\\Hüz\\AppData\\Local\\Skryb\\updates\\install-0.4.0.log";
    const view = { progress: { step: "failed" as const, percent: 0, version: "0.4.0", log }, estimate: { fraction: 0, label: "", secondsLeft: 0 } };

    expect(installTitle(view)).toBe("The update didn't finish");
    expect(installDescription(view)).toBe(
      `Skryb 0.4.0 couldn't be installed. Start Skryb from the Start menu to keep using this version. The installer's log is ${log}.`,
    );
  });
});

describe("reopening the window the last version handed over", () => {
  it("opens on the page the old window showed", () => {
    for (const route of ["/updates", "/history", "/microphone", "/dictionary"]) {
      expect(restoredPath(route)).toBe(route);
    }
  });

  it("falls back to the usual page for a page this version doesn't have, or none", () => {
    for (const route of ["/models-old", "", "/", "updates", "/updates/../history", null, undefined]) {
      expect(restoredPath(route)).toBeNull();
    }
  });
});

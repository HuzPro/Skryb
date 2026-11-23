import { describe, expect, it } from "vitest";
import { restoredPath } from "../app";
import { installDescription, installTitle } from "../components/install";
import type { UpdateStatus } from "../lib/types";
import { describeUpdate, everyTheme, keptTitle, updateControls } from "./updates";

const status = (overrides: Partial<UpdateStatus>): UpdateStatus => ({ state: "idle", version: "", bytes_done: 0, bytes_total: 0, error: "", ...overrides });

describe("the Updates page with a version ready", () => {
  it("offers to check for a newer one as well as to restart and update", () => {
    expect(updateControls(status({ state: "ready", version: "0.18.1" }))).toEqual(["check", "install"]);
  });

  it("says it installs anything newer, and keeps both buttons when the last check failed", () => {
    const failedCheck = status({ state: "ready", version: "0.18.1", error: "Couldn't check for updates. Timed out." });

    expect(describeUpdate(failedCheck, "0.18.0")).toEqual({
      title: "Skryb 0.18.1 is ready",
      description: "Skryb restarts to install it, or anything newer, once any dictation finishes.",
    });
    expect(updateControls(failedCheck)).toEqual(["check", "install"]);
  });

  it("offers only the check when nothing is ready, and nothing in a git checkout", () => {
    for (const state of ["idle", "off", "up_to_date", "updated", "failed", "checking", "installing"] as const) {
      expect(updateControls(status({ state }))).toEqual(["check"]);
    }
    expect(updateControls(null)).toEqual([]);
  });
});

describe("the Updates page while Restart and update fetches a newer version first", () => {
  it("shows the check, then the download of the newer version with its bar, then the install", () => {
    const checking = status({ state: "checking", version: "0.18.1" });
    const downloading = status({ state: "downloading", version: "0.19.1", bytes_done: 13_938_000, bytes_total: 30_972_960 });
    const installing = status({ state: "installing", version: "0.19.1" });

    expect(describeUpdate(checking, "0.18.0")).toEqual({ title: "Skryb 0.18.0", description: "Checking GitHub for a newer version..." });
    expect(describeUpdate(downloading, "0.18.0")).toEqual({ title: "Downloading Skryb 0.19.1", description: "45% downloaded." });
    expect(describeUpdate(installing, "0.18.0").title).toBe("Installing Skryb 0.19.1");
    expect(updateControls(checking)).toEqual(["check"]);
    expect(updateControls(downloading)).toEqual(["progress"]);
    expect(updateControls(installing)).toEqual(["check"]);
  });

  it("names the version it falls back to when the newer one couldn't be fetched", () => {
    const fallback = status({ state: "installing", version: "0.18.1", error: "Couldn't check for updates. Timed out. Installing Skryb 0.18.1, downloaded earlier, instead." });

    expect(describeUpdate(fallback, "0.18.0").title).toBe("Installing Skryb 0.18.1");
  });
});

describe("how much usage data is kept", () => {
  it("reads nothing, one event, or a count with its thousands separated", () => {
    expect(keptTitle(0)).toBe("Nothing kept yet");
    expect(keptTitle(1)).toBe("1 event kept");
    expect(keptTitle(2)).toBe("2 events kept");
    expect(keptTitle(1234)).toBe(`${(1234).toLocaleString()} events kept`);
    expect(keptTitle(100_000)).toBe(`${(100_000).toLocaleString()} events kept`);
  });
});

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

describe("the colour picker", () => {
  it("gives every theme the one colour picked", () => {
    expect(everyTheme("plum")).toEqual({ default: "plum", glass: "plum", material: "plum", paper: "plum" });
  });
});

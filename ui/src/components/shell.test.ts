import { describe, expect, it } from "vitest";
import type { AppInfo } from "../lib/types";
import { fitList, LIST } from "../pages/history";
import { engineStatus, fitSidebar, saveSidebarWidth, savedSidebarWidth, saveWidth, savedWidth, SIDEBAR } from "./shell";

function ready(device?: string | null, device_note?: string | null): AppInfo {
  return { state: "idle", engine: { ready: true, loading: false, error: null, model: "phonon-2.5-english", device, device_note } } as AppInfo;
}

describe("the engine status", () => {
  it("says the NPU is doing the work when it is", () => {
    expect(engineStatus(ready("NPU", null), false)).toEqual({ tone: "success", label: "Ready on NPU" });
  });

  it("warns when the CPU has taken over from the NPU", () => {
    expect(engineStatus(ready("CPU", "The NPU stopped working, so the CPU took over."), false)).toEqual({ tone: "warning", label: "Ready on CPU" });
  });

  it("stays a plain Ready for engines that don't say where they run", () => {
    expect(engineStatus(ready(), false)).toEqual({ tone: "success", label: "Ready" });
    expect(engineStatus(ready(null, null), false)).toEqual({ tone: "success", label: "Ready" });
  });
});

function memory(start: Record<string, string> = {}) {
  const saved = new Map(Object.entries(start));
  return { getItem: (key: string) => saved.get(key) ?? null, setItem: (key: string, value: string) => void saved.set(key, value), saved };
}

describe("the sidebar's width", () => {
  it("stays between 180 and 520 pixels on a wide window", () => {
    expect(fitSidebar(40, 2560)).toBe(SIDEBAR.least);
    expect(fitSidebar(300.6, 2560)).toBe(301);
    expect(fitSidebar(5000, 2560)).toBe(SIDEBAR.most);
  });

  it("leaves the page at least 420 pixels on a narrow window, but never goes under 180", () => {
    expect(fitSidebar(500, 800)).toBe(380);
    expect(fitSidebar(500, 300)).toBe(SIDEBAR.least);
  });

  it("starts at the standard width the first time, and when what was saved is junk", () => {
    expect(savedSidebarWidth(memory())).toBe(SIDEBAR.standard);
    for (const junk of ["", "wide", "-50", "NaN", "Infinity"]) expect(savedSidebarWidth(memory({ "sidebar-width": junk }))).toBe(SIDEBAR.standard);
  });

  it("comes back at the width it was dragged to", () => {
    const store = memory();
    saveSidebarWidth(347.4, store);
    expect(store.saved.get("sidebar-width")).toBe("347");
    expect(savedSidebarWidth(store)).toBe(347);
  });

  it("falls back to the standard width when storage is blocked", () => {
    const blocked = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
    };
    expect(() => saveSidebarWidth(300, blocked)).not.toThrow();
    expect(savedSidebarWidth(blocked)).toBe(SIDEBAR.standard);
    expect(savedSidebarWidth(null)).toBe(SIDEBAR.standard);
  });
});

describe("History's list width", () => {
  it("stays as dragged, between its least and its most", () => {
    expect(fitList(40, 2560)).toBe(LIST.least);
    expect(fitList(-300, 2560)).toBe(LIST.least);
    expect(fitList(412.6, 2560)).toBe(413);
    expect(fitList(LIST.least, 2560)).toBe(LIST.least);
    expect(fitList(LIST.most, 2560)).toBe(LIST.most);
    expect(fitList(5000, 2560)).toBe(LIST.most);
  });

  it("leaves the reader beside it at least 320 px, and keeps its own least when there is no room for both", () => {
    expect(fitList(700, 864)).toBe(538);
    expect(fitList(538, 864)).toBe(538);
    expect(fitList(539, 864)).toBe(538);
    expect(fitList(700, 500)).toBe(LIST.least);
    expect(fitList(700, 0)).toBe(LIST.least);
  });

  it("is remembered on its own, apart from the sidebar's", () => {
    const store = memory({ "sidebar-width": "300" });
    expect(savedWidth("history-list-width", LIST.standard, store)).toBe(LIST.standard);
    saveWidth("history-list-width", 451.5, store);
    expect(savedWidth("history-list-width", LIST.standard, store)).toBe(452);
    expect(savedSidebarWidth(store)).toBe(300);
    for (const junk of ["", "wide", "-50", "0", "NaN", "Infinity"]) expect(savedWidth("history-list-width", LIST.standard, memory({ "history-list-width": junk }))).toBe(LIST.standard);
  });
});

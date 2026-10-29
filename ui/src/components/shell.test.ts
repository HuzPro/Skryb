import { describe, expect, it } from "vitest";
import { fitSidebar, saveSidebarWidth, savedSidebarWidth, SIDEBAR } from "./shell";

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

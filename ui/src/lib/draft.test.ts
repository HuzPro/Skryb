import { describe, expect, it } from "vitest";
import { draftReducer, isDirty, sameValue, type Draft } from "./draft";

interface Doc {
  name: string;
  overrides: [string, string][];
  nested: { notes: string; devices: string[] };
}

const doc = (overrides = 3): Doc => ({
  name: "Zürich 日本語 🎙️",
  overrides: Array.from({ length: overrides }, (_, i) => [`app-${i}.exe`, i % 2 ? "ctrl_v" : "shift_insert"]),
  nested: { notes: "", devices: ["Microphone (Blue Yeti)", "Microphone (Blue Yeti)", ""] },
});

const loaded = (value: Doc): Draft<Doc> | null => draftReducer<Doc>(null, { type: "loaded", value });

describe("settings draft", () => {
  it("starts clean with what the core sent", () => {
    const state = loaded(doc());
    expect(isDirty(state)).toBe(false);
    expect(state?.draft).toEqual(doc());
  });

  it("is dirty after an edit and clean again when the edit is undone by hand", () => {
    let state = loaded(doc());
    state = draftReducer(state, { type: "edit", update: (d) => ({ ...d, nested: { ...d.nested, notes: "British spelling" } }) });
    expect(isDirty(state)).toBe(true);
    state = draftReducer(state, { type: "edit", update: (d) => ({ ...d, nested: { ...d.nested, notes: "" } }) });
    expect(isDirty(state)).toBe(false);
  });

  it("keeps the user's edits when the core sends a fresh copy mid-edit", () => {
    let state = loaded(doc());
    state = draftReducer(state, { type: "edit", update: (d) => ({ ...d, name: "edited" }) });
    const fromAnotherWindow = { ...doc(), overrides: [] as [string, string][] };
    state = draftReducer(state, { type: "loaded", value: fromAnotherWindow });
    expect(state?.draft.name).toBe("edited");
    expect(state?.saved).toEqual(fromAnotherWindow);
    expect(isDirty(state)).toBe(true);
  });

  it("takes a fresh copy from the core when nothing is being edited", () => {
    const fresh = { ...doc(), name: "renamed elsewhere" };
    const state = draftReducer(loaded(doc()), { type: "loaded", value: fresh });
    expect(state?.draft).toEqual(fresh);
    expect(isDirty(state)).toBe(false);
  });

  it("discard goes back to what the core has", () => {
    let state = loaded(doc());
    state = draftReducer(state, { type: "edit", update: (d) => ({ ...d, overrides: d.overrides.slice(1) }) });
    state = draftReducer(state, { type: "discard" });
    expect(state?.draft).toEqual(doc());
    expect(isDirty(state)).toBe(false);
  });

  it("a committed save makes the saved value the draft", () => {
    let state = loaded(doc());
    const saved = { ...doc(), name: "saved" };
    state = draftReducer(state, { type: "edit", update: () => saved });
    state = draftReducer(state, { type: "committed", value: saved });
    expect(isDirty(state)).toBe(false);
    expect(state?.saved.name).toBe("saved");
  });

  it("edits before anything has loaded are ignored", () => {
    expect(draftReducer<Doc>(null, { type: "edit", update: (d) => d })).toBeNull();
    expect(isDirty(null)).toBe(false);
  });

  it("notices one changed entry among two thousand paste overrides", () => {
    const big = doc(2000);
    let state = loaded(big);
    expect(isDirty(state)).toBe(false);
    state = draftReducer(state, {
      type: "edit",
      update: (d) => ({ ...d, overrides: d.overrides.map((o, i) => (i === 1999 ? [o[0], "type"] : o)) }),
    });
    expect(isDirty(state)).toBe(true);
  });

  it("compares by value, not by key order or identity", () => {
    expect(sameValue({ a: 1, b: [1, { c: "é" }] }, { b: [1, { c: "é" }], a: 1 })).toBe(true);
    expect(sameValue([1, 2], [1, 2, 3])).toBe(false);
    expect(sameValue({ a: undefined }, { b: undefined })).toBe(false);
    expect(sameValue([], {})).toBe(false);
    expect(sameValue(null, {})).toBe(false);
    expect(sameValue(0.5, 0.5)).toBe(true);
  });
});

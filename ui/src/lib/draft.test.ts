import { describe, expect, it } from "vitest";
import { draftReducer, inTurn, isDirty, sameValue, type Draft } from "./draft";

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

describe("the dictionary saving as it changes", () => {
  it("keeps edits made while a save was on its way, and stays unsaved until they are saved too", () => {
    let state = loaded(doc());
    state = draftReducer(state, { type: "edit", update: (d) => ({ ...d, name: "Ñoño, first edit" }) });
    const sent = state?.draft as Doc;
    state = draftReducer(state, { type: "edit", update: (d) => ({ ...d, name: "Ñoño, second edit" }) });

    state = draftReducer(state, { type: "saved", value: sent });

    expect(state?.saved.name).toBe("Ñoño, first edit");
    expect(state?.draft.name).toBe("Ñoño, second edit");
    expect(isDirty(state)).toBe(true);
  });

  it("is saved once the save carried the newest edit", () => {
    let state = loaded(doc(1500));
    state = draftReducer(state, { type: "edit", update: (d) => ({ ...d, overrides: d.overrides.slice(1) }) });

    state = draftReducer(state, { type: "saved", value: state?.draft as Doc });

    expect(isDirty(state)).toBe(false);
  });

  it("an edit undone while its save was on its way still gets saved back", () => {
    let state = loaded(doc());
    state = draftReducer(state, { type: "edit", update: (d) => ({ ...d, name: "" }) });
    const sent = state?.draft as Doc;
    state = draftReducer(state, { type: "edit", update: (d) => ({ ...d, name: doc().name }) });

    state = draftReducer(state, { type: "saved", value: sent });

    expect(isDirty(state)).toBe(true);
    expect(state?.draft).toEqual(doc());
  });

  it("runs saves one at a time, so of two quick edits the later one lands last", async () => {
    const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
    let newest = "first";
    const landed: string[] = [];
    let onTheWay = 0;
    let mostAtOnce = 0;
    const save = inTurn(async () => {
      const sent = newest;
      onTheWay += 1;
      mostAtOnce = Math.max(mostAtOnce, onTheWay);
      await pause(sent === "first" ? 30 : 0);
      landed.push(sent);
      onTheWay -= 1;
      return sent;
    });

    const first = save();
    await pause(5);
    newest = "second";
    const second = save();

    expect(await Promise.all([first, second])).toEqual(["first", "second"]);
    expect(landed).toEqual(["first", "second"]);
    expect(mostAtOnce).toBe(1);
  });

  it("a failed save does not stop the ones after it", async () => {
    let calls = 0;
    const save = inTurn(async () => {
      calls += 1;
      if (calls === 1) throw new Error("The core went away");
      return "saved";
    });

    const first = save().catch((error: Error) => error.message);
    const second = save();

    expect(await first).toBe("The core went away");
    expect(await second).toBe("saved");
  });
});

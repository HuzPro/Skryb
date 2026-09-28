import { describe, expect, it } from "vitest";
import { wordDiff, type DiffPart } from "./diff";

const joined = (parts: DiffPart[]) => parts.map((part) => part.text).join("");
const changed = (parts: DiffPart[]) => parts.filter((part) => part.change !== "same").map((part) => part.text.trim());

describe("word diff", () => {
  it("marks nothing when the model changed nothing", () => {
    const { before, after } = wordDiff("Testing 1, 2, 3.", "Testing 1, 2, 3.");

    expect(before).toEqual([{ text: "Testing 1, 2, 3.", change: "same" }]);
    expect(after).toEqual([{ text: "Testing 1, 2, 3.", change: "same" }]);
  });

  it("marks the words the model dropped on the left and the words it added on the right", () => {
    const { before, after } = wordDiff("Tuesday actually no Wednesday", "Tuesday, Wednesday");

    expect(changed(before)).toEqual(["Tuesday actually no"]);
    expect(changed(after)).toEqual(["Tuesday,"]);
  });

  it("keeps the words both share and only marks the rest", () => {
    const { before, after } = wordDiff("um so the meeting is on friday", "So the meeting is on Friday.");

    expect(changed(before)).toEqual(["um so", "friday"]);
    expect(changed(after)).toEqual(["So", "Friday."]);
  });

  it("gives back both texts exactly, line breaks, unicode and emoji included", () => {
    const heard = "  steps café in tokyo\nsay مرحبا 🎉  ";
    const typed = "Steps:\n1. Café in 東京\n2. Say مرحبا 🎉";
    const { before, after } = wordDiff(heard, typed);

    expect(joined(before)).toBe(heard);
    expect(joined(after)).toBe(typed);
  });

  it("handles blank sides", () => {
    expect(wordDiff("", "")).toEqual({ before: [], after: [] });
    expect(wordDiff("", "Hello there.").after).toEqual([{ text: "Hello there.", change: "added" }]);
    expect(wordDiff("   ", "").before).toEqual([{ text: "   ", change: "removed" }]);
  });

  it("compares a dictation of a thousand words quickly and exactly", () => {
    const heard = Array.from({ length: 1000 }, (_, index) => `word${index % 97}`).join(" ");
    const typed = heard.replace("word5 ", "").replace("word40", "Word40") + " done.";
    const started = performance.now();

    const { before, after } = wordDiff(heard, typed);

    expect(performance.now() - started).toBeLessThan(1000);
    expect(joined(before)).toBe(heard);
    expect(joined(after)).toBe(typed);
    expect(changed(before)).toEqual(["word5", "word40"]);
    expect(changed(after)).toEqual(["Word40", "done."]);
  });

  it("marks everything changed rather than freeze on a huge pair", () => {
    const huge = Array.from({ length: 3000 }, (_, index) => `w${index}`).join(" ");
    const { before, after } = wordDiff(huge, huge);

    expect(before).toEqual([{ text: huge, change: "removed" }]);
    expect(after).toEqual([{ text: huge, change: "added" }]);
  });
});

import { describe, expect, it } from "vitest";
import { englishStatusLine } from "../components/english";
import { awaitingEnglish, englishOf, hasArabic, withEntry } from "./english";
import type { MeetingEntry, MeetingTranscript, TranslationStatus } from "./types";

const line = (id: number, text: string, english?: string): MeetingEntry => ({
  id, track: "them", speaker: 0, start: id, end: id + 1, text, key: "them:0",
  ...(english ? { english, words: text.split(" ").map((word): [string, string] => [word, `g:${word}`]) } : {}),
});

const status = (over: Partial<TranslationStatus> = {}): TranslationStatus => ({
  name: "translation", label: "Jais-3 3.7B", description: "", display_size: "3.4 GB", installed: true,
  device: null, device_name: null, error: null, download: null, ...over,
});

describe("Show English in the shell", () => {
  it("counts a line as Arabic by its letters, not by Arabic punctuation or digits alone", () => {
    expect(hasArabic("بكرة نروح السوق")).toBe(true);
    expect(hasArabic("Meet at 5 ، ok؟ ١٢٣")).toBe(false);
    expect(hasArabic("")).toBe(false);
  });

  it("waits only on Arabic lines that have no English yet", () => {
    const unsent = line(0, "وين المفاتيح؟");
    const glossed = line(2, "هلا والله", "Hello indeed");
    expect(awaitingEnglish([unsent, line(1, "On the table."), glossed, line(3, "وين المفاتيح؟")])).toBe(2);
    expect(englishOf(glossed)?.words[0]).toEqual(["هلا", "g:هلا"]);
    expect(englishOf(unsent)).toBeNull();
  });

  it("replaces a re-sent line by its id and leaves the rest of a long meeting alone", () => {
    const entries = Array.from({ length: 3000 }, (_, id) => line(id, "زين"));
    const transcript: MeetingTranscript = { folder: "D:\\Meetings\\x", started: "2026-11-01T10:00:00", names: {}, entries };
    const updated = withEntry(transcript, line(1500, "زين", "Fine"));
    expect(updated.entries[1500]?.english).toBe("Fine");
    expect(updated.entries.filter((entry) => entry.english)).toHaveLength(1);
    expect(updated.entries[1499]).toBe(entries[1499]);
  });

  it("says what turning it on downloads, then how far the download has got, then where the model runs", () => {
    expect(englishStatusLine(status({ installed: false }), false)).toBe("Turning this on downloads Jais-3 3.7B, 3.4 GB.");
    expect(englishStatusLine(status({ installed: false, download: { state: "failed", bytes_done: 0, bytes_total: 1, error: "offline" } }), true)).toBe(
      "The download of Jais-3 3.7B failed: offline",
    );
    expect(englishStatusLine(status(), true)).toBe("Jais-3 3.7B is downloaded and loads the first time a line needs it.");
    expect(englishStatusLine(status({ device: "GPU", device_name: "AMD Radeon RX 9060 XT" }), true)).toBe("Jais-3 3.7B runs on the GPU (AMD Radeon RX 9060 XT).");
    expect(englishStatusLine(status({ device: "CPU" }), true)).toBe("Jais-3 3.7B runs on the processor.");
    expect(englishStatusLine(status({ error: "llama.cpp found no NVIDIA graphics card, so translation can't run on it." }), true)).toContain("couldn't start");
  });
});

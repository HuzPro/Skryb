import type { EnglishLine, MeetingEntry, MeetingTranscript } from "./types";

// Arabic letters, not its punctuation or digits: the core's `skryb.translation.gloss.is_arabic`.
const ARABIC_LETTER = /[\u0620-\u063f\u0641-\u064a\u066e\u066f\u0671-\u06d3\u06d5\u06ee\u06ef\u06fa-\u06fc\u06ff\u0750-\u077f\u08a0-\u08c9\ufb50-\ufdfb\ufe70-\ufefc]/;

/** Whether a line has Arabic in it, and so gets English and a gloss with Show English on. */
export function hasArabic(text: string): boolean {
  return ARABIC_LETTER.test(text);
}

/** A meeting line's English, once the core has sent it. */
export function englishOf(entry: MeetingEntry): EnglishLine | null {
  return entry.english && entry.words?.length ? { english: entry.english, words: entry.words } : null;
}

/** Arabic lines still waiting for their English. */
export function awaitingEnglish(entries: MeetingEntry[]): number {
  return entries.filter((entry) => hasArabic(entry.text) && !englishOf(entry)).length;
}

/** A saved meeting with one of its lines sent again, as when its English arrives; other lines stay as they are. */
export function withEntry(transcript: MeetingTranscript, entry: MeetingEntry): MeetingTranscript {
  return { ...transcript, entries: transcript.entries.map((line) => (line.id === entry.id ? entry : line)) };
}

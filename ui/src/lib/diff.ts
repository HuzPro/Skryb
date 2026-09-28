export interface DiffPart {
  text: string;
  change: "same" | "removed" | "added";
}

// Past this many word pairs the comparison table gets too big to build on every render.
const MAX_CELLS = 4_000_000;

/** Each word with the whitespace in front of it, so the parts put back together give the text again. */
function words(text: string): string[] {
  const found: string[] = [...(text.match(/\s*\S+/g) ?? [])];
  const tail = text.slice(found.join("").length);
  if (tail) found.push(tail);
  return found;
}

function merged(parts: DiffPart[]): DiffPart[] {
  const out: DiffPart[] = [];
  for (const part of parts) {
    const last = out.at(-1);
    if (last && last.change === part.change) last.text += part.text;
    else out.push({ ...part });
  }
  return out;
}

/**
 * `before` with the words `after` dropped marked removed, and `after` with the words it added marked added.
 * Words match ignoring the whitespace around them; the longest common run of words is kept.
 */
export function wordDiff(before: string, after: string): { before: DiffPart[]; after: DiffPart[] } {
  const a = words(before);
  const b = words(after);
  const n = a.length;
  const m = b.length;
  if (n * m > MAX_CELLS) {
    return { before: merged(a.map((text) => ({ text, change: "removed" }))), after: merged(b.map((text) => ({ text, change: "added" }))) };
  }
  const keyA = a.map((word) => word.trim());
  const keyB = b.map((word) => word.trim());
  // lengths[i * (m + 1) + j]: longest common run of a[i:] and b[j:].
  const lengths = new Uint32Array((n + 1) * (m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lengths[i * (m + 1) + j] =
        keyA[i] === keyB[j] ? (lengths[(i + 1) * (m + 1) + j + 1] ?? 0) + 1 : Math.max(lengths[(i + 1) * (m + 1) + j] ?? 0, lengths[i * (m + 1) + j + 1] ?? 0);
    }
  }
  const left: DiffPart[] = [];
  const right: DiffPart[] = [];
  let i = 0;
  let j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && keyA[i] === keyB[j]) {
      left.push({ text: a[i] ?? "", change: "same" });
      right.push({ text: b[j] ?? "", change: "same" });
      i++;
      j++;
    } else if (j < m && (i === n || (lengths[i * (m + 1) + j + 1] ?? 0) >= (lengths[(i + 1) * (m + 1) + j] ?? 0))) {
      right.push({ text: b[j] ?? "", change: "added" });
      j++;
    } else {
      left.push({ text: a[i] ?? "", change: "removed" });
      i++;
    }
  }
  return { before: merged(left), after: merged(right) };
}

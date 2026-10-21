import type { Binding } from "./types";

const MODIFIERS: Record<string, number> = {
  ctrl: 0xa2, ctrl_l: 0xa2, ctrl_r: 0xa3,
  alt: 0xa4, alt_l: 0xa4, alt_r: 0xa5,
  shift: 0xa0, shift_l: 0xa0, shift_r: 0xa1,
  cmd: 0x5b,
};
/** Mirrors `hotkeys.keys.PUNCTUATION_KEYS`: virtual-key code and the character it prints on a US layout. */
const PUNCTUATION: Record<string, [number, string]> = {
  semicolon: [0xba, ";"], equals: [0xbb, "="], comma: [0xbc, ","], minus: [0xbd, "-"],
  period: [0xbe, "."], slash: [0xbf, "/"], backtick: [0xc0, "`"], bracket_left: [0xdb, "["],
  backslash: [0xdc, "\\"], bracket_right: [0xdd, "]"], quote: [0xde, "'"],
};

const NAMED: Record<string, number> = {
  ...MODIFIERS,
  space: 0x20,
  mouse_middle: 0x04,
  mouse4: 0x05,
  mouse5: 0x06,
};
for (let n = 1; n <= 24; n++) NAMED[`f${n}`] = 0x70 + n - 1;
for (const [name, [vk]] of Object.entries(PUNCTUATION)) NAMED[name] = vk;

const MODIFIER_VKS = new Set([0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0x5b, 0x5c]);

const CAPS: Record<string, string> = {
  ctrl: "Ctrl", alt: "Alt", shift: "Shift", cmd: "Win", space: "Space", esc: "Esc",
  mouse_middle: "Middle mouse", mouse4: "Mouse 4", mouse5: "Mouse 5",
  ...Object.fromEntries(Object.entries(PUNCTUATION).map(([name, [, printed]]) => [name, printed])),
};

// The Copilot key reports itself as these three keys and is often remapped
// to act as Right Ctrl, so a recording that reaches them waits for one more.
const COPILOT_KEY = "cmd+shift+f23";

/** Mirrors `hotkeys.key_parser.parse_hotkey_combo`: virtual-key codes, or throws "Unknown key: x". */
export function parseCombo(combo: string): number[] {
  return combo.toLowerCase().split("+").map((raw) => {
    const part = raw.trim();
    const named = NAMED[part];
    if (named !== undefined) return named;
    if (/^[a-z0-9]$/.test(part)) return part.toUpperCase().charCodeAt(0);
    throw new Error(`Unknown key: ${part}`);
  });
}

/** One key as its cap reads: "ctrl" -> "Ctrl", "alt_r" -> "Right Alt", "mouse4" -> "Mouse 4". */
export function keyCap(part: string): string {
  const key = part.trim().toLowerCase();
  const sided = /^(ctrl|alt|shift)_([lr])$/.exec(key);
  if (sided) return `${sided[2] === "l" ? "Left" : "Right"} ${CAPS[sided[1] ?? ""] ?? ""}`;
  if (CAPS[key]) return CAPS[key];
  if (/^f\d+$/.test(key)) return key.toUpperCase();
  if (key.length === 1) return key.toUpperCase();
  return key.charAt(0).toUpperCase() + key.slice(1);
}

export function keyCaps(combo: string): string[] {
  return combo.split("+").filter((part) => part.trim()).map(keyCap);
}

/** The core reports a combination at every key press; it is complete once a key that is not a modifier joins it, the Copilot key aside. */
export function endsOnKey(combo: string): boolean {
  const parts = combo.split("+").map((part) => part.trim().toLowerCase()).filter(Boolean);
  const last = parts.at(-1);
  return last !== undefined && !(last in MODIFIERS) && parts.join("+") !== COPILOT_KEY;
}

/** "ctrl+shift+alt+m" -> "Ctrl + Shift + Alt + M". */
export function formatCombo(combo: string): string {
  return keyCaps(combo).join(" + ");
}

export interface BindingProblem {
  message: string;
  /** Index of the binding at fault, or -1 when the problem is the whole set. */
  index: number;
}

class ComboMatcher {
  private matched = 0;
  constructor(private readonly keys: number[]) {}

  press(key: number): boolean {
    if (this.matched === this.keys.length || this.keys.slice(0, this.matched).includes(key)) return false;
    if (key === this.keys[this.matched]) {
      this.matched += 1;
      return this.matched === this.keys.length;
    }
    this.matched = 0;
    return false;
  }
}

function keysThatWorkAlone(binding: Binding): number[] {
  let keys: number[];
  try {
    keys = parseCombo(binding.keys);
  } catch (error) {
    throw new Error(`Shortcut '${binding.keys}' is not valid: ${(error as Error).message}.`);
  }
  const first = keys[0];
  if (keys.length === 1 && first !== undefined && MODIFIER_VKS.has(first)) {
    throw new Error(
      `Shortcut ${binding.keys} needs another key: a modifier on its own fires with every shortcut that uses it.`,
    );
  }
  if (new Set(keys).size !== keys.length) throw new Error(`Shortcut ${binding.keys} repeats a key.`);
  return keys;
}

function sameKeys(a: number[], b: number[]): boolean {
  const left = new Set(a);
  const right = new Set(b);
  return left.size === right.size && [...left].every((key) => right.has(key));
}

/** Mirrors `hotkeys.bindings.validate_bindings`, same messages, plus which binding is at fault. */
export function validateBindings(bindings: Binding[]): BindingProblem | null {
  if (!bindings.some((binding) => binding.action === "dictate")) {
    return { message: "Add at least one dictation shortcut.", index: -1 };
  }
  const parsed: number[][] = [];
  for (const [index, binding] of bindings.entries()) {
    try {
      parsed.push(keysThatWorkAlone(binding));
    } catch (error) {
      return { message: (error as Error).message, index };
    }
  }
  for (const [i, first] of bindings.entries()) {
    for (const [j, second] of bindings.entries()) {
      if (i === j) continue;
      const firstKeys = parsed[i] ?? [];
      const secondKeys = parsed[j] ?? [];
      if (sameKeys(firstKeys, secondKeys)) {
        return { message: `Shortcut ${second.keys} is used twice. Give each shortcut its own keys.`, index: j };
      }
      const matcher = new ComboMatcher(firstKeys);
      if (secondKeys.some((key) => matcher.press(key))) {
        return { message: `Pressing ${second.keys} would also trigger ${first.keys}. Change one of them.`, index: j };
      }
    }
  }
  return null;
}

import { describe, expect, it } from "vitest";
import { endsOnKey, formatCombo, keyCaps, validateBindings } from "./keys";
import type { Binding, BindingAction, BindingMode } from "./types";

const bind = (keys: string, action: BindingAction = "dictate", mode: BindingMode = "toggle"): Binding => ({ keys, mode, action });

describe("key caps", () => {
  it("reads a stored combination the way the keys are labelled", () => {
    expect(formatCombo("ctrl+shift+alt+m")).toBe("Ctrl + Shift + Alt + M");
  });

  it("names mouse buttons in words", () => {
    expect(formatCombo("mouse4")).toBe("Mouse 4");
    expect(formatCombo("mouse5")).toBe("Mouse 5");
    expect(formatCombo("ctrl+mouse_middle")).toBe("Ctrl + Middle mouse");
  });

  it("shows function keys, digits, sides and the Windows key", () => {
    expect(keyCaps("alt_r+f13")).toEqual(["Right Alt", "F13"]);
    expect(keyCaps("ctrl_l+7")).toEqual(["Left Ctrl", "7"]);
    expect(keyCaps("cmd+space")).toEqual(["Win", "Space"]);
  });

  it("keeps recording while only modifiers are held and finishes on the first other key or button", () => {
    const pressed = ["ctrl", "ctrl+alt", "ctrl+alt+shift_r", "ctrl+alt+shift_r+m"];
    expect(pressed.map(endsOnKey)).toEqual([false, false, false, true]);
    expect(endsOnKey("cmd")).toBe(false);
    expect(endsOnKey("mouse4")).toBe(true);
    expect(endsOnKey("ctrl+f13")).toBe(true);
    expect(endsOnKey(" ")).toBe(false);
  });

  it("shows punctuation keys as the characters they print", () => {
    expect(formatCombo("cmd+shift+f23+quote")).toBe("Win + Shift + F23 + '");
    expect(keyCaps("semicolon+equals+comma+minus+period+slash+backtick+bracket_left+backslash+bracket_right+quote")).toEqual(
      [";", "=", ",", "-", ".", "/", "`", "[", "\\", "]", "'"],
    );
    expect(formatCombo("Ctrl + QUOTE")).toBe("Ctrl + '");
  });

  it("keeps recording after the Copilot key so another key can follow it", () => {
    const pressed = ["cmd", "cmd+shift", "cmd+shift+f23", "cmd+shift+f23+quote"];
    expect(pressed.map(endsOnKey)).toEqual([false, false, false, true]);
    expect(endsOnKey(" CMD + Shift + F23 ")).toBe(false);
    expect(endsOnKey("shift+f23")).toBe(true);
    expect(endsOnKey("ctrl+f23")).toBe(true);
    expect(endsOnKey("f23")).toBe(true);
  });

  it("ignores stray spaces and empty parts", () => {
    expect(formatCombo(" ctrl + a ")).toBe("Ctrl + A");
    expect(formatCombo("")).toBe("");
    expect(keyCaps("ctrl++m")).toEqual(["Ctrl", "M"]);
  });
});

describe("shortcut validation mirrors the core", () => {
  it("accepts the default shortcuts", () => {
    expect(validateBindings([bind("ctrl+alt+m"), bind("ctrl+alt+e", "edit_selection"), bind("ctrl+alt+v", "paste_last", "press")])).toBeNull();
  });

  it("needs at least one dictation shortcut", () => {
    expect(validateBindings([bind("ctrl+alt+v", "paste_last", "press")])).toEqual({ message: "Add at least one dictation shortcut.", index: -1 });
    expect(validateBindings([])).toEqual({ message: "Add at least one dictation shortcut.", index: -1 });
  });

  it("refuses a modifier on its own", () => {
    expect(validateBindings([bind("ctrl+alt+m"), bind("shift")])).toEqual({
      message: "Shortcut shift needs another key: a modifier on its own fires with every shortcut that uses it.",
      index: 1,
    });
  });

  it("refuses a key typed twice in one shortcut, even written two ways", () => {
    expect(validateBindings([bind("ctrl+ctrl_l+m")])?.message).toBe("Shortcut ctrl+ctrl_l+m repeats a key.");
  });

  it("names an unknown key the way the core does", () => {
    expect(validateBindings([bind("ctrl+ä")])).toEqual({ message: "Shortcut 'ctrl+ä' is not valid: Unknown key: ä.", index: 0 });
    expect(validateBindings([bind("ctrl+")])?.message).toBe("Shortcut 'ctrl+' is not valid: Unknown key: .");
  });

  it("accepts punctuation keys by name and refuses the character they print, like the core", () => {
    expect(validateBindings([bind("cmd+shift+f23+quote"), bind("ctrl+alt+bracket_left", "edit_selection")])).toBeNull();
    expect(validateBindings([bind("ctrl+'")])).toEqual({ message: "Shortcut 'ctrl+'' is not valid: Unknown key: '.", index: 0 });
  });

  it("refuses the same keys used twice in any order", () => {
    expect(validateBindings([bind("ctrl+alt+m"), bind("alt+ctrl+m", "edit_selection")])).toEqual({
      message: "Shortcut alt+ctrl+m is used twice. Give each shortcut its own keys.",
      index: 1,
    });
  });

  it("refuses a shortcut that would also fire a shorter one", () => {
    expect(validateBindings([bind("ctrl+m"), bind("ctrl+m+x", "copy_last", "press")])).toEqual({
      message: "Pressing ctrl+m+x would also trigger ctrl+m. Change one of them.",
      index: 1,
    });
  });

  it("lets a longer shortcut share a modifier when the shorter one is not inside it", () => {
    expect(validateBindings([bind("ctrl+a"), bind("ctrl+alt+a", "edit_selection")])).toBeNull();
  });

  it("refuses a longer shortcut that ends with a shorter one", () => {
    expect(validateBindings([bind("alt+a"), bind("ctrl+alt+a", "edit_selection")])?.message).toBe(
      "Pressing ctrl+alt+a would also trigger alt+a. Change one of them.",
    );
  });

  it("checks twelve hundred distinct shortcuts without a false alarm and finds the one clash", () => {
    const chars = [..."abcdefghijklmnopqrstuvwxyz0123456789"];
    const pairs = chars.flatMap((first, i) => chars.slice(i + 1).map((second) => `${first}+${second}`)).slice(0, 300);
    const bindings = ["ctrl", "alt", "shift", "cmd"].flatMap((modifier) => pairs.map((pair) => bind(`${modifier}+${pair}`)));
    expect(bindings).toHaveLength(1200);
    expect(validateBindings(bindings)).toBeNull();
    expect(validateBindings([...bindings, bind("b+ctrl+a", "copy_last", "press")])).toEqual({
      message: "Shortcut b+ctrl+a is used twice. Give each shortcut its own keys.",
      index: 1200,
    });
  });
});

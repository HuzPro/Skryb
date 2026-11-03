/** A value as the core has it (`saved`) and as the user is editing it (`draft`). */
export interface Draft<T> {
  saved: T;
  draft: T;
}

export type DraftAction<T> =
  | { type: "loaded"; value: T }
  | { type: "edit"; update: (draft: T) => T }
  | { type: "discard" }
  | { type: "committed"; value: T }
  | { type: "saved"; value: T };

export function sameValue(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const aKeys = Object.keys(a);
  const bKeys = Object.keys(b);
  if (aKeys.length !== bKeys.length) return false;
  return aKeys.every((key) =>
    Object.prototype.hasOwnProperty.call(b, key) &&
    sameValue((a as Record<string, unknown>)[key], (b as Record<string, unknown>)[key]),
  );
}

export function isDirty<T>(state: Draft<T> | null): boolean {
  return state !== null && !sameValue(state.saved, state.draft);
}

/**
 * A fresh copy from the core replaces the draft only while nothing is being
 * edited, so a sync from another page or window never throws away edits.
 * `saved` is a save that landed while editing went on: the core has that
 * value, and the edits made since stay.
 */
export function draftReducer<T>(state: Draft<T> | null, action: DraftAction<T>): Draft<T> | null {
  switch (action.type) {
    case "loaded":
      if (state === null || !isDirty(state)) return { saved: action.value, draft: action.value };
      return { saved: action.value, draft: state.draft };
    case "committed":
      return { saved: action.value, draft: action.value };
    case "saved":
      return state === null ? null : { saved: action.value, draft: state.draft };
    case "edit":
      return state === null ? null : { saved: state.saved, draft: action.update(state.draft) };
    case "discard":
      return state === null ? null : { saved: state.saved, draft: state.saved };
  }
}

/**
 * Runs *save* one call at a time, each after the one before has finished. A save that reads the newest
 * draft when its turn comes therefore always lands after any older one: the later edit wins.
 */
export function inTurn<T>(save: () => Promise<T>): () => Promise<T> {
  let last: Promise<unknown> = Promise.resolve();
  return () => {
    const next = last.then(save, save);
    last = next;
    return next;
  };
}

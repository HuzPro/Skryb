import { useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, useLayoutEffect, useMemo, useReducer, useRef, useState, type ReactNode, type RefObject } from "react";
import { ApiError, api, errorText } from "./api";
import { draftReducer, inTurn, isDirty, type Draft, type DraftAction } from "./draft";
import { engineState } from "./events";
import { validateBindings } from "./keys";
import { onBeforeClose } from "./platform";
import { queries } from "./queries";
import { onServerEvent } from "./sse";
import type { Dictionary, License, SaveResult, Settings } from "./types";

export type UnsavedChoice = "save" | "discard" | "cancel";

export type Prompt =
  | { kind: "license"; license: License; resolve: (accepted: boolean) => void }
  | { kind: "confirm"; title: string; message: string; action: string; resolve: (confirmed: boolean) => void }
  | { kind: "unsaved"; resolve: (choice: UnsavedChoice) => void };

export type SavePhase = "idle" | "saving" | "applying";

/** Where the dictionary's own saving is: it saves itself shortly after each change. */
export interface DictionarySave {
  state: "idle" | "saving" | "saved" | "failed";
  error: string;
  /** False when the core refused the change; the page then shows what the core kept. */
  retry: boolean;
}

/** How long the dictionary waits after a change before saving it, so a burst of changes is one save. */
const DICTIONARY_SAVE_DELAY_MS = 500;

type Reducer<T> = (state: Draft<T> | null, action: DraftAction<T>) => Draft<T> | null;

interface Store {
  settings: Draft<Settings> | null;
  editSettings: (update: (settings: Settings) => Settings) => void;
  discardSettings: () => void;
  saveSettings: () => Promise<boolean>;
  savePhase: SavePhase;
  saveNote: string;
  saveError: string;
  dismissSaveError: () => void;
  dictionary: Draft<Dictionary> | null;
  dispatchDictionary: (action: DraftAction<Dictionary>) => void;
  /** Saves the dictionary now if it has changed; "" when it is saved, else why not. */
  flushDictionary: () => Promise<string>;
  dictionarySave: DictionarySave;
  anyDirty: () => boolean;
  prompt: Prompt | null;
  closePrompt: () => void;
  confirm: (title: string, message: string, action: string) => Promise<boolean>;
  askUnsaved: () => Promise<UnsavedChoice>;
  withLicense: <T>(run: () => Promise<T>) => Promise<T | undefined>;
  saveAll: () => Promise<boolean>;
}

const StoreContext = createContext<Store | null>(null);
const StoreRefContext = createContext<RefObject<Store> | null>(null);

export function useStore(): Store {
  const store = useContext(StoreContext);
  if (!store) throw new Error("useStore outside StoreProvider");
  return store;
}

/** The store without re-rendering on every edit, for components that only act on it, like the shell. */
export function useStoreRef(): RefObject<Store> {
  const ref = useContext(StoreRefContext);
  if (!ref) throw new Error("useStoreRef outside StoreProvider");
  return ref;
}

/** Bindings without keys are rows the user added and left empty; the core never sees them. */
export function withoutEmptyBindings(settings: Settings): Settings {
  return { ...settings, hotkey_bindings: settings.hotkey_bindings.filter((binding) => binding.keys.trim()) };
}

export function StoreProvider({ children }: { children: ReactNode }) {
  const client = useQueryClient();
  // Settings fetched before the first render start the draft, so no page renders without them.
  const [settings, dispatchSettings] = useReducer(draftReducer as Reducer<Settings>, client.getQueryData<Settings>(["settings"]), (value) =>
    value ? { saved: value, draft: value } : null,
  );
  const [dictionary, dispatchDictionary] = useReducer(draftReducer as Reducer<Dictionary>, null);
  const [savePhase, setSavePhase] = useState<SavePhase>("idle");
  const [saveNote, setSaveNote] = useState("");
  const [saveError, setSaveError] = useState("");
  const [prompt, setPrompt] = useState<Prompt | null>(null);
  const [dictionarySave, setDictionarySave] = useState<DictionarySave>({ state: "idle", error: "", retry: false });

  const settingsQuery = useQuery(queries.settings);
  useEffect(() => {
    if (settingsQuery.data) dispatchSettings({ type: "loaded", value: settingsQuery.data });
  }, [settingsQuery.data]);

  const confirm = useCallback(
    (title: string, message: string, action: string) =>
      new Promise<boolean>((resolve) => setPrompt({ kind: "confirm", title, message, action, resolve })),
    [],
  );
  const askUnsaved = useCallback(() => new Promise<UnsavedChoice>((resolve) => setPrompt({ kind: "unsaved", resolve })), []);

  const withLicense = useCallback(async <T,>(run: () => Promise<T>): Promise<T | undefined> => {
    try {
      return await run();
    } catch (error) {
      if (!(error instanceof ApiError) || error.status !== 409 || !error.license) throw error;
      const license = error.license;
      const accepted = await new Promise<boolean>((resolve) => setPrompt({ kind: "license", license, resolve }));
      if (!accepted) return undefined;
      await api.post("/api/license/accept", { id: license.id });
      return await run();
    }
  }, []);

  const waitForEngine = useCallback((model: string) => {
    setSaveNote(`Loading ${model}...`);
    const stop = onServerEvent((name, data) => {
      if (name !== "engine") return;
      const event = data as { error?: string | null; model?: string };
      const state = engineState(data as object);
      if (state === "loading") {
        setSaveNote(`Loading ${event.model ?? model}...`);
        return;
      }
      stop();
      setSavePhase("idle");
      setSaveNote("");
      if (state === "failed") setSaveError(`Could not load ${event.model ?? model}: ${event.error ?? "no reason was given"}`);
    });
  }, []);

  const saveSettings = useCallback(async (): Promise<boolean> => {
    if (!settings) return true;
    const draft = withoutEmptyBindings(settings.draft);
    const problem = validateBindings(draft.hotkey_bindings);
    if (problem) {
      setSaveError(problem.message);
      return false;
    }
    setSaveError("");
    setSavePhase("saving");
    try {
      const result = await withLicense(() => api.put<SaveResult>("/api/settings", draft));
      if (!result) {
        setSavePhase("idle");
        return false;
      }
      dispatchSettings({ type: "committed", value: draft });
      client.setQueryData(["settings"], draft);
      if (result.engine_reload_started) {
        setSavePhase("applying");
        waitForEngine(draft.model_size);
      } else {
        setSavePhase("idle");
      }
      return true;
    } catch (error) {
      setSavePhase("idle");
      setSaveError(errorText(error));
      return false;
    }
  }, [settings, withLicense, waitForEngine, client]);

  // Saves read the newest draft when their turn comes. The ref takes every change at once, so a save
  // started right after an edit, or queued behind another save, sees it before React renders again.
  const dictionaryRef = useRef(dictionary);
  dictionaryRef.current = dictionary;
  const changeDictionary = useCallback((action: DraftAction<Dictionary>) => {
    dispatchDictionary(action);
    dictionaryRef.current = draftReducer(dictionaryRef.current, action);
  }, []);
  const flushDictionary = useMemo(
    () =>
      inTurn(async (): Promise<string> => {
        const current = dictionaryRef.current;
        if (!current || !isDirty(current)) return "";
        const sent = current.draft;
        setDictionarySave({ state: "saving", error: "", retry: false });
        try {
          await api.put("/api/dictionary", sent);
          changeDictionary({ type: "saved", value: sent });
          client.setQueryData(["dictionary"], sent);
          setDictionarySave({ state: "saved", error: "", retry: false });
          return "";
        } catch (error) {
          const refused = error instanceof ApiError && error.status === 400;
          if (refused) changeDictionary({ type: "discard" });
          const text = refused ? `Not saved. ${error.detail || error.message}` : `Not saved. ${errorText(error)}`;
          setDictionarySave({ state: "failed", error: text, retry: !refused });
          return text;
        }
      }),
    [client, changeDictionary],
  );

  useEffect(() => {
    if (!isDirty(dictionary)) return;
    const timer = setTimeout(() => void flushDictionary(), DICTIONARY_SAVE_DELAY_MS);
    return () => clearTimeout(timer);
  }, [dictionary, flushDictionary]);

  // A change still waiting out the delay goes to the core before the window closes.
  useEffect(() => {
    const forget = onBeforeClose(flushDictionary);
    return () => {
      forget();
      void flushDictionary();
    };
  }, [flushDictionary]);

  const saveAll = useCallback(async (): Promise<boolean> => {
    const settingsSaved = !isDirty(settings) || (await saveSettings());
    const problem = await flushDictionary();
    return settingsSaved && !problem;
  }, [settings, saveSettings, flushDictionary]);

  // The dictionary saves itself, so it only counts as unsaved once a save of it has failed.
  const latest = useRef({ settings, dictionary, dictionarySave });
  latest.current = { settings, dictionary, dictionarySave };
  const anyDirty = useCallback(
    () => isDirty(latest.current.settings) || (latest.current.dictionarySave.state === "failed" && isDirty(latest.current.dictionary)),
    [],
  );

  const store = useMemo<Store>(
    () => ({
      settings,
      editSettings: (update) => dispatchSettings({ type: "edit", update }),
      discardSettings: () => {
        dispatchSettings({ type: "discard" });
        setSaveError("");
      },
      saveSettings,
      savePhase,
      saveNote,
      saveError,
      dismissSaveError: () => setSaveError(""),
      dictionary,
      dispatchDictionary: changeDictionary,
      flushDictionary,
      dictionarySave,
      anyDirty,
      prompt,
      closePrompt: () => setPrompt(null),
      confirm,
      askUnsaved,
      withLicense,
      saveAll,
    }),
    [settings, saveSettings, savePhase, saveNote, saveError, dictionary, changeDictionary, flushDictionary, dictionarySave, anyDirty, prompt, confirm, askUnsaved, withLicense, saveAll],
  );

  const storeRef = useRef(store);
  storeRef.current = store;

  return (
    <StoreRefContext.Provider value={storeRef}>
      <StoreContext.Provider value={store}>{children}</StoreContext.Provider>
    </StoreRefContext.Provider>
  );
}

/** Loads a page-level document (the dictionary) into its draft once, and again after the core changes it. */
export function useSyncedDraft<T>(key: "dictionary", path: string, dispatch: (action: DraftAction<T>) => void): boolean {
  const query = useQuery({ queryKey: [key], queryFn: () => api.get<T>(path) });
  // Before paint, so a page opening with its data already fetched never shows empty first.
  useLayoutEffect(() => {
    if (query.data) dispatch({ type: "loaded", value: query.data });
  }, [query.data, dispatch]);
  return query.isError;
}

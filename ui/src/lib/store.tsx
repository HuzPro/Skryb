import { useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from "react";
import { ApiError, api, errorText } from "./api";
import { draftReducer, isDirty, type Draft, type DraftAction } from "./draft";
import { engineState } from "./events";
import { validateBindings } from "./keys";
import { onServerEvent } from "./sse";
import type { Dictionary, License, SaveResult, Settings, Transform } from "./types";

export type UnsavedChoice = "save" | "discard" | "cancel";

export type Prompt =
  | { kind: "license"; license: License; resolve: (accepted: boolean) => void }
  | { kind: "confirm"; title: string; message: string; action: string; resolve: (confirmed: boolean) => void }
  | { kind: "unsaved"; resolve: (choice: UnsavedChoice) => void };

export type SavePhase = "idle" | "saving" | "applying";

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
  saveDictionary: () => Promise<string>;
  transforms: Draft<Transform[]> | null;
  dispatchTransforms: (action: DraftAction<Transform[]>) => void;
  saveTransforms: () => Promise<string>;
  anyDirty: () => boolean;
  prompt: Prompt | null;
  closePrompt: () => void;
  confirm: (title: string, message: string, action: string) => Promise<boolean>;
  askUnsaved: () => Promise<UnsavedChoice>;
  withLicense: <T>(run: () => Promise<T>) => Promise<T | undefined>;
  saveAll: () => Promise<boolean>;
}

const StoreContext = createContext<Store | null>(null);

export function useStore(): Store {
  const store = useContext(StoreContext);
  if (!store) throw new Error("useStore outside StoreProvider");
  return store;
}

/** Bindings without keys are rows the user added and left empty; the core never sees them. */
export function withoutEmptyBindings(settings: Settings): Settings {
  return { ...settings, hotkey_bindings: settings.hotkey_bindings.filter((binding) => binding.keys.trim()) };
}

export function StoreProvider({ children }: { children: ReactNode }) {
  const client = useQueryClient();
  const [settings, dispatchSettings] = useReducer(draftReducer as Reducer<Settings>, null);
  const [dictionary, dispatchDictionary] = useReducer(draftReducer as Reducer<Dictionary>, null);
  const [transforms, dispatchTransforms] = useReducer(draftReducer as Reducer<Transform[]>, null);
  const [savePhase, setSavePhase] = useState<SavePhase>("idle");
  const [saveNote, setSaveNote] = useState("");
  const [saveError, setSaveError] = useState("");
  const [prompt, setPrompt] = useState<Prompt | null>(null);

  const settingsQuery = useQuery({ queryKey: ["settings"], queryFn: () => api.get<Settings>("/api/settings") });
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

  const saveDictionary = useCallback(async (): Promise<string> => {
    if (!dictionary || !isDirty(dictionary)) return "";
    try {
      await api.put("/api/dictionary", dictionary.draft);
      dispatchDictionary({ type: "committed", value: dictionary.draft });
      client.setQueryData(["dictionary"], dictionary.draft);
      return "";
    } catch (error) {
      return errorText(error);
    }
  }, [dictionary, client]);

  const saveTransforms = useCallback(async (): Promise<string> => {
    if (!transforms || !isDirty(transforms)) return "";
    try {
      await api.put("/api/transforms", transforms.draft);
      dispatchTransforms({ type: "committed", value: transforms.draft });
      client.setQueryData(["transforms"], transforms.draft);
      return "";
    } catch (error) {
      return errorText(error);
    }
  }, [transforms, client]);

  const saveAll = useCallback(async (): Promise<boolean> => {
    const settingsSaved = !isDirty(settings) || (await saveSettings());
    const problems = [await saveDictionary(), await saveTransforms()].filter(Boolean);
    if (problems.length) setSaveError(problems.join(" "));
    return settingsSaved && problems.length === 0;
  }, [settings, saveSettings, saveDictionary, saveTransforms]);

  const latest = useRef({ settings, dictionary, transforms });
  latest.current = { settings, dictionary, transforms };
  const anyDirty = useCallback(
    () => isDirty(latest.current.settings) || isDirty(latest.current.dictionary) || isDirty(latest.current.transforms),
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
      dispatchDictionary,
      saveDictionary,
      transforms,
      dispatchTransforms,
      saveTransforms,
      anyDirty,
      prompt,
      closePrompt: () => setPrompt(null),
      confirm,
      askUnsaved,
      withLicense,
      saveAll,
    }),
    [settings, saveSettings, savePhase, saveNote, saveError, dictionary, saveDictionary, transforms, saveTransforms, anyDirty, prompt, confirm, askUnsaved, withLicense, saveAll],
  );

  return <StoreContext.Provider value={store}>{children}</StoreContext.Provider>;
}

/** Loads a page-level document (dictionary, transforms) into its draft once, and again after the core changes it. */
export function useSyncedDraft<T>(key: "dictionary" | "transforms", path: string, dispatch: (action: DraftAction<T>) => void): boolean {
  const query = useQuery({ queryKey: [key], queryFn: () => api.get<T>(path) });
  useEffect(() => {
    if (query.data) dispatch({ type: "loaded", value: query.data });
  }, [query.data, dispatch]);
  return query.isError;
}

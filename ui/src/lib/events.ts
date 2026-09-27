import type { AppInfo, EngineInfo, Model, Settings, WritingStatus } from "./types";

/** The server data the event stream keeps current. */
export interface Snapshot {
  app?: AppInfo | undefined;
  models?: Model[] | undefined;
  writing?: WritingStatus | undefined;
  settings?: Settings | undefined;
}

export type QueryName = "app" | "models" | "writing" | "settings" | "engines" | "history" | "microphones";

export interface Reduction {
  snapshot: Snapshot;
  refetch: QueryName[];
}

export type EngineState = "loading" | "ready" | "failed";

interface EngineEvent {
  state?: string;
  status?: string;
  loading?: boolean;
  ready?: boolean;
  error?: string | null;
  model?: string;
}

interface DownloadEvent {
  name: string;
  state: "downloading" | "paused" | "completed" | "failed" | "cancelled" | "deleted";
  bytes_done: number;
  bytes_total: number;
  error: string;
}

/** The core sends `{state, loading, ready, error, model}`; the flags alone are read when `state` is missing. */
export function engineState(data: EngineEvent): EngineState {
  const named = data.state ?? data.status;
  if (named === "loading" || named === "ready" || named === "failed") return named;
  if (data.error) return "failed";
  if (data.loading) return "loading";
  return data.ready === false ? "loading" : "ready";
}

function engineInfo(data: EngineEvent, previous: EngineInfo | undefined): EngineInfo {
  const state = engineState(data);
  return {
    ready: state === "ready",
    loading: state === "loading",
    error: state === "failed" ? data.error || "The model could not be loaded." : null,
    model: data.model ?? previous?.model ?? "",
  };
}

function applyDownload(models: Model[], event: DownloadEvent): Model[] {
  return models.map((model) => {
    if (model.name !== event.name) return model;
    if (event.state === "completed") return { ...model, installed: true, download: null };
    if (event.state === "cancelled") return { ...model, download: null };
    if (event.state === "deleted") return { ...model, installed: false, download: null };
    return {
      ...model,
      download: { state: event.state, bytes_done: event.bytes_done, bytes_total: event.bytes_total, error: event.error ?? "" },
    };
  });
}

/** What one server event changes in the cached data, and which queries must be fetched again. */
export function reduceEvent(snapshot: Snapshot, name: string, data: unknown): Reduction {
  switch (name) {
    case "app":
      return { snapshot: { ...snapshot, app: data as AppInfo }, refetch: [] };
    case "engine": {
      if (!snapshot.app) return { snapshot, refetch: ["app"] };
      const engine = engineInfo(data as EngineEvent, snapshot.app.engine);
      return { snapshot: { ...snapshot, app: { ...snapshot.app, engine } }, refetch: [] };
    }
    case "download": {
      const event = data as DownloadEvent;
      const known = snapshot.models?.some((model) => model.name === event.name) ?? false;
      if (!snapshot.models || !known) return { snapshot, refetch: ["models"] };
      const models = applyDownload(snapshot.models, event);
      const finished = event.state === "completed" || event.state === "cancelled" || event.state === "deleted";
      return { snapshot: { ...snapshot, models }, refetch: finished ? ["models"] : [] };
    }
    case "writing":
      return { snapshot: { ...snapshot, writing: data as WritingStatus }, refetch: [] };
    case "settings":
      return { snapshot: { ...snapshot, settings: data as Settings }, refetch: [] };
    case "hardware":
      return { snapshot, refetch: ["engines"] };
    case "microphones":
      return { snapshot, refetch: ["microphones"] };
    case "history": {
      const kind = (data as { event?: string }).event;
      return { snapshot, refetch: kind === "changed" || kind === "deleted_all" ? ["history"] : [] };
    }
    default:
      return { snapshot, refetch: [] };
  }
}

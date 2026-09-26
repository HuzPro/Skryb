import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { eventsUrl } from "./api";
import { reduceEvent, type Snapshot } from "./events";

type Listener = (name: string, data: unknown) => void;

const listeners = new Set<Listener>();
const EVENT_NAMES = ["app", "engine", "download", "meter", "capture", "hardware", "writing", "history", "settings", "microphones", "heartbeat"];
const SNAPSHOT_KEYS = ["app", "models", "writing", "settings"] as const;
const LOST_AFTER_MS = 5000;
const MAX_RETRY_MS = 5000;

export function onServerEvent(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Calls `handler` for every server event while the component is mounted. */
export function useServerEvent(handler: Listener): void {
  const latest = useRef(handler);
  latest.current = handler;
  useEffect(() => onServerEvent((name, data) => latest.current(name, data)), []);
}

/**
 * Keeps one EventSource open, folds its events into the query cache, and
 * reports the core as lost once it has been unreachable for five seconds.
 */
export function useServerEvents(): boolean {
  const client = useQueryClient();
  const [lost, setLost] = useState(false);

  useEffect(() => {
    let source: EventSource | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    let lostTimer: ReturnType<typeof setTimeout> | undefined;
    let delay = 500;
    let wasDown = false;

    const handle = (name: string, data: unknown) => {
      const before: Snapshot = {};
      for (const key of SNAPSHOT_KEYS) before[key] = client.getQueryData([key]) as never;
      const { snapshot, refetch } = reduceEvent(before, name, data);
      for (const key of SNAPSHOT_KEYS) {
        if (snapshot[key] !== before[key]) client.setQueryData([key], snapshot[key]);
      }
      for (const key of refetch) void client.invalidateQueries({ queryKey: [key] });
      for (const listener of listeners) listener(name, data);
    };

    const connect = () => {
      source = new EventSource(eventsUrl());
      source.onopen = () => {
        clearTimeout(lostTimer);
        lostTimer = undefined;
        delay = 500;
        setLost(false);
        if (wasDown) void client.invalidateQueries();
        wasDown = false;
      };
      source.onerror = () => {
        source?.close();
        wasDown = true;
        lostTimer ??= setTimeout(() => setLost(true), LOST_AFTER_MS);
        retryTimer = setTimeout(connect, delay);
        delay = Math.min(delay * 2, MAX_RETRY_MS);
      };
      for (const name of EVENT_NAMES) {
        source.addEventListener(name, (event) => handle(name, JSON.parse((event as MessageEvent<string>).data || "{}")));
      }
    };

    connect();
    return () => {
      source?.close();
      clearTimeout(retryTimer);
      clearTimeout(lostTimer);
    };
  }, [client]);

  return lost;
}

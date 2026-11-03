import { QueryClient, queryOptions } from "@tanstack/react-query";
import { api } from "./api";
import type { AppInfo, Dictionary, HistoryEntry, LiveMeeting, MeetingOverview, Microphone, Model, Settings } from "./types";

export const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: Infinity, retry: 1, refetchOnWindowFocus: false } },
});

export const HISTORY_PAGE = 100;

/** What the pages read from the core; the shell fetches each once, the events keep them current. */
export const queries = {
  settings: queryOptions({ queryKey: ["settings"], queryFn: () => api.get<Settings>("/api/settings") }),
  app: queryOptions({ queryKey: ["app"], queryFn: () => api.get<AppInfo>("/api/app") }),
  models: queryOptions({ queryKey: ["models"], queryFn: () => api.get<Model[]>("/api/models") }),
  microphones: queryOptions({ queryKey: ["microphones"], queryFn: () => api.get<Microphone[]>("/api/microphones") }),
  dictionary: queryOptions({ queryKey: ["dictionary"], queryFn: () => api.get<Dictionary>("/api/dictionary") }),
  meetings: queryOptions({ queryKey: ["meetings"], queryFn: () => api.get<MeetingOverview>("/api/meetings") }),
  liveMeeting: queryOptions({
    queryKey: ["meeting-live"],
    queryFn: async () => (await api.get<{ meeting: LiveMeeting | null }>("/api/meetings/live")).meeting,
  }),
  history: (search: string, limit: number) =>
    queryOptions({
      queryKey: ["history", search, limit],
      queryFn: () => api.get<HistoryEntry[]>(`/api/history?query=${encodeURIComponent(search)}&limit=${limit}`),
    }),
};

/** Every page's data, fetched once the window is up so no page fills in after it opens. */
export function prefetchPages(client: QueryClient): void {
  void client.prefetchQuery(queries.models);
  void client.prefetchQuery(queries.microphones);
  void client.prefetchQuery(queries.dictionary);
  void client.prefetchQuery(queries.meetings);
  void client.prefetchQuery(queries.liveMeeting);
  void client.prefetchQuery(queries.history("", HISTORY_PAGE));
}

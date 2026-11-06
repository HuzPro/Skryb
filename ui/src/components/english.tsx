import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, errorText } from "../lib/api";
import { useServerEvent } from "../lib/sse";
import type { DownloadState, EnglishLine, TranslationStatus } from "../lib/types";
import { cn } from "../lib/utils";
import { formatGigabytes } from "./models";
import { InlineError, ProgressBar } from "./parts";
import { Button } from "./ui/button";

const IN_FLIGHT = new Set(["queued", "downloading", "paused"]);

/** An Arabic line word by word, right to left, each word over its English gloss, and the whole line in English below. */
export function Interlinear({ line, className }: { line: EnglishLine; className?: string }) {
  return (
    <div className={cn("py-0.5", className)}>
      <div dir="rtl" lang="ar" className="flex flex-wrap gap-1.5">
        {line.words.map(([word, gloss], index) => (
          <span key={index} className="inline-flex min-w-11 flex-col items-center rounded-md border bg-muted/40 px-2 pt-0.5 pb-1 text-center">
            <span className="text-[16px] leading-relaxed">{word}</span>
            <bdi lang="en" className="max-w-[18ch] text-[11.5px] leading-snug text-muted-foreground">
              {gloss}
            </bdi>
          </span>
        ))}
      </div>
      <p dir="ltr" lang="en" className="mt-1.5 text-[13.5px] leading-relaxed">
        {line.english}
      </p>
    </div>
  );
}

/** The model Show English uses on this computer, its download, and where it runs once it has started. */
export function useTranslationStatus(): TranslationStatus | undefined {
  const client = useQueryClient();
  useServerEvent((name, data) => {
    const download = data as { name?: string; state: string; bytes_done: number; bytes_total: number; error: string };
    if (name !== "download" || download.name !== "translation") return;
    if (!IN_FLIGHT.has(download.state)) {
      void client.invalidateQueries({ queryKey: ["translation"] });
      return;
    }
    client.setQueryData<TranslationStatus>(["translation"], (status) =>
      status ? { ...status, download: { state: download.state as DownloadState, bytes_done: download.bytes_done, bytes_total: download.bytes_total, error: download.error } } : status,
    );
  });
  return useQuery({ queryKey: ["translation"], queryFn: () => api.get<TranslationStatus>("/api/translation") }).data;
}

export function englishStatusLine(status: TranslationStatus, on: boolean): string {
  const download = status.download;
  if (status.error) return `${status.label} couldn't start: ${status.error}`;
  if (download?.state === "failed") return `The download of ${status.label} failed: ${download.error}`;
  if (download && IN_FLIGHT.has(download.state)) {
    return download.state === "queued" ? `${status.label} is waiting to download.` : `Downloading ${status.label}: ${formatGigabytes(download.bytes_done)} of ${formatGigabytes(download.bytes_total)}.`;
  }
  if (!status.installed) return on ? `${status.label} (${status.display_size}) isn't downloaded yet.` : `Turning this on downloads ${status.label}, ${status.display_size}.`;
  if (!status.device) return `${status.label} is downloaded and loads the first time a line needs it.`;
  const where = status.device === "CPU" ? "the processor" : `the ${status.device}${status.device_name ? ` (${status.device_name})` : ""}`;
  return `${status.label} runs on ${where}.`;
}

/** One line under the Show English switch: what downloads, how far it has got, and where the model runs. */
export function EnglishStatus({ on }: { on: boolean }) {
  const status = useTranslationStatus();
  const [problem, setProblem] = useState("");
  if (!status) return null;
  const download = status.download;
  const retry = on && !status.installed && (!download || download.state === "failed");
  return (
    <div>
      <div className={cn((status.error || download?.state === "failed") && "text-destructive")}>{englishStatusLine(status, on)}</div>
      {download && IN_FLIGHT.has(download.state) ? <ProgressBar className="mt-1.5 max-w-80" value={download.bytes_total ? download.bytes_done / download.bytes_total : 0} /> : null}
      {retry ? (
        <Button variant="link" className="text-[12.5px]" onClick={() => void api.post("/api/translation/download").then(() => setProblem(""), (error: unknown) => setProblem(errorText(error)))}>
          Download now
        </Button>
      ) : null}
      {problem ? <InlineError>{problem}</InlineError> : null}
    </div>
  );
}

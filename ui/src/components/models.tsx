import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Check } from "lucide-react";
import { useState, type MouseEvent } from "react";
import { api, errorText } from "../lib/api";
import { useStore } from "../lib/store";
import type { Model } from "../lib/types";
import { cn } from "../lib/utils";
import { InlineError, ProgressBar } from "./parts";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";

const GB = 1_000_000_000;

export function useModels(): Model[] | undefined {
  return useQuery({ queryKey: ["models"], queryFn: () => api.get<Model[]>("/api/models") }).data;
}

export function formatGigabytes(bytes: number): string {
  return `${(bytes / GB).toFixed(1)} GB`;
}

function downloadLine(done: number, total: number): string {
  const share = total ? done / total : 0;
  return `${Math.round(share * 100)}% · ${formatGigabytes(done)} of ${formatGigabytes(total)}`;
}

/** One speech model: select it when installed, or download, pause, resume, cancel or delete it. */
export function ModelCard({ model, selected, onSelect }: { model: Model; selected: boolean; onSelect: () => void }) {
  const { withLicense, confirm } = useStore();
  const client = useQueryClient();
  const [error, setError] = useState("");
  const path = `/api/models/${encodeURIComponent(model.name)}`;
  const download = model.download;
  const active = download?.state === "downloading" || download?.state === "paused";
  const queued = download?.state === "queued";

  const act = (verb: string) => async (event: MouseEvent) => {
    event.stopPropagation();
    setError("");
    try {
      if (verb === "download") {
        await withLicense(() => api.post(`${path}/download`));
      } else if (verb === "delete") {
        const sure = await confirm(`Delete ${model.name}?`, `This frees ${model.display_size}. You can download it again later.`, "Delete");
        if (!sure) return;
        await api.post(`${path}/delete`);
        await client.invalidateQueries({ queryKey: ["models"] });
      } else {
        await api.post(`${path}/${verb}`);
      }
    } catch (failure) {
      setError(errorText(failure));
    }
  };

  return (
    <div
      onClick={() => model.installed && onSelect()}
      className={cn(
        "rounded-xl border bg-card px-4 py-3 shadow-[0_1px_2px_rgb(0_0_0/0.03)] transition-colors duration-150 has-[[role=radio]:focus-visible]:ring-2 has-[[role=radio]:focus-visible]:ring-ring/40",
        model.installed && "hover:bg-accent/40",
        selected && "border-primary bg-selected hover:bg-selected",
      )}
    >
      <div className="flex items-center gap-4">
        <button
          role="radio"
          aria-checked={selected}
          disabled={!model.installed}
          aria-label={`${model.name}, ${model.display_size}${model.installed ? ", installed" : ""}`}
          className="min-w-0 flex-1 text-left outline-none"
        >
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="text-[13.5px] font-medium">{model.name}</span>
            <span className="text-[12px] text-muted-foreground">{model.display_size}</span>
            {model.installed ? (
              <Badge variant="success">
                <Check className="size-3" />
                Installed
              </Badge>
            ) : null}
          </div>
          <div className="mt-0.5 text-[12.5px] text-muted-foreground">{model.description}</div>
        </button>
        <div className="flex shrink-0 gap-2">
          {active ? (
            <>
              <Button variant="outline" size="sm" onClick={act(download?.state === "paused" ? "resume" : "pause")}>
                {download?.state === "paused" ? "Resume" : "Pause"}
              </Button>
              <Button variant="outline" size="sm" onClick={act("cancel")}>
                Cancel
              </Button>
            </>
          ) : queued ? (
            <Button variant="outline" size="sm" onClick={act("cancel")}>
              Cancel
            </Button>
          ) : model.installed ? (
            <Button variant="outline" size="sm" onClick={act("delete")}>
              Delete
            </Button>
          ) : (
            <Button size="sm" onClick={act("download")}>
              {download?.state === "failed" ? "Retry download" : "Download"}
            </Button>
          )}
        </div>
      </div>
      {active && download ? (
        <div className="mt-3">
          <ProgressBar value={download.bytes_total ? download.bytes_done / download.bytes_total : 0} />
          <div className="mt-1.5 text-[12px] text-muted-foreground">
            {download.state === "paused" ? "Paused · " : "Downloading "}
            {downloadLine(download.bytes_done, download.bytes_total)}
          </div>
        </div>
      ) : null}
      {queued ? (
        <div className="mt-3 text-[12px] text-muted-foreground">Queued: starts when another download finishes.</div>
      ) : null}
      {download?.state === "failed" ? <InlineError>Download failed: {download.error || "no reason was given"}</InlineError> : null}
      {error ? <InlineError>{error}</InlineError> : null}
    </div>
  );
}

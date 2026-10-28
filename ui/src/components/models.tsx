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

/** One card in a model list; `members` are the downloads behind it, more than one for a style family. */
export interface ModelEntry {
  model: Model;
  members: Model[];
}

/** A model that ships one download per speech style (Bark Plus) is one card: the family's name and size, its members downloaded together. */
export function modelEntries(models: Model[]): ModelEntry[] {
  const entries: ModelEntry[] = [];
  const families = new Map<string, Model[]>();
  for (const model of models) {
    const family = model.style_family;
    if (!family) {
      entries.push({ model, members: [model] });
      continue;
    }
    const members = families.get(family);
    if (members) {
      members.push(model);
      continue;
    }
    const created = [model];
    families.set(family, created);
    entries.push({ model, members: created });
  }
  return entries.map((entry) => (entry.model.style_family ? { model: familyModel(entry.members), members: entry.members } : entry));
}

function familyModel(members: Model[]): Model {
  const first = members[0] as Model;
  return {
    ...first,
    name: first.style_family ?? first.name,
    display_size: first.style_family_size ?? first.display_size,
    installed: members.every((member) => member.installed),
    speech_style: null,
    download: combinedDownload(members),
  };
}

const DOWNLOAD_STATES = ["downloading", "queued", "paused", "failed"] as const;

/** The members' downloads as one: the busiest state, their bytes added up. */
export function combinedDownload(members: Model[]): Model["download"] {
  const downloads = members.flatMap((member) => (member.download ? [member.download] : []));
  if (!downloads.length) return null;
  const states = new Set(downloads.map((download) => download.state));
  return {
    state: DOWNLOAD_STATES.find((state) => states.has(state)) ?? "failed",
    bytes_done: downloads.reduce((sum, download) => sum + download.bytes_done, 0),
    bytes_total: downloads.reduce((sum, download) => sum + download.bytes_total, 0),
    error: downloads.find((download) => download.error)?.error ?? "",
  };
}

const IN_FLIGHT = new Set(["queued", "downloading", "paused"]);
const ACTS_ON: Record<string, Set<string>> = {
  pause: new Set(["downloading"]),
  resume: new Set(["paused"]),
  cancel: IN_FLIGHT,
};

/** One speech model: select it when installed, or download, pause, resume, cancel or delete it. */
export function ModelCard({ model, members, selected, onSelect }: { model: Model; members?: Model[]; selected: boolean; onSelect: () => void }) {
  const { withLicense, confirm } = useStore();
  const client = useQueryClient();
  const [error, setError] = useState("");
  const parts = members ?? [model];
  const post = (target: Model, verb: string) => api.post(`/api/models/${encodeURIComponent(target.name)}/${verb}`);
  const download = model.download;
  const active = download?.state === "downloading" || download?.state === "paused";
  const queued = download?.state === "queued";

  const act = (verb: string) => async (event: MouseEvent) => {
    event.stopPropagation();
    setError("");
    try {
      if (verb === "download") {
        for (const part of parts) {
          if (!part.installed && !IN_FLIGHT.has(part.download?.state ?? "")) await withLicense(() => post(part, "download"));
        }
      } else if (verb === "delete") {
        if (selected && parts.length > 1) return setError("Switch to another model before deleting this one.");
        const sure = await confirm(`Delete ${model.name}?`, `This frees ${model.display_size}. You can download it again later.`, "Delete");
        if (!sure) return;
        for (const part of parts) if (part.installed) await post(part, "delete");
        await client.invalidateQueries({ queryKey: ["models"] });
      } else {
        for (const part of parts) if (ACTS_ON[verb]?.has(part.download?.state ?? "")) await post(part, verb);
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

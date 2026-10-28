import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { ModelCard, useModels } from "../components/models";
import { ChoiceCards, Disclosure, PageBody, PageTitle, Row, Rows, Section } from "../components/parts";
import { Select } from "../components/ui/select";
import { api } from "../lib/api";
import { useStore } from "../lib/store";
import type { Engines, Model, Settings, TranscriptionMode } from "../lib/types";

export const LANGUAGES = [
  { value: "en", label: "English" },
  { value: "es", label: "Spanish" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
  { value: "it", label: "Italian" },
  { value: "pt", label: "Portuguese" },
  { value: "nl", label: "Dutch" },
  { value: "ja", label: "Japanese" },
  { value: "ko", label: "Korean" },
  { value: "zh", label: "Chinese" },
];

const STYLES = [
  { value: "intended", label: "Intended (what I meant)" },
  { value: "verbatim", label: "Verbatim (what I said)" },
];

export function fitsMode(model: Model, mode: TranscriptionMode): boolean {
  return mode === "streaming" ? model.supports_streaming : model.supports_batch;
}

/** Switching mode keeps the model when it can run in the new mode, else takes the first installed one that can. */
export function withMode(settings: Settings, mode: TranscriptionMode, models: Model[]): Settings {
  const current = models.find((model) => model.name === settings.model_size);
  if (current && fitsMode(current, mode)) return { ...settings, transcription_mode: mode };
  const candidates = models.filter((model) => fitsMode(model, mode));
  const replacement = candidates.find((model) => model.installed) ?? candidates[0];
  return { ...settings, transcription_mode: mode, model_size: replacement?.name ?? settings.model_size, engine: "auto" };
}

type Style = Settings["crisperwhisper_mode"];

/** One card per style family: the selected member, else the member for the style in use. */
export function visibleModels(models: Model[], selected: string, style: Style): Model[] {
  const shown: Model[] = [];
  const families = new Set<string>();
  for (const model of models) {
    const family = model.style_family;
    if (!family) {
      shown.push(model);
      continue;
    }
    if (families.has(family)) continue;
    families.add(family);
    const members = models.filter((member) => member.style_family === family);
    shown.push(members.find((member) => member.name === selected) ?? members.find((member) => member.speech_style === style) ?? model);
  }
  return shown;
}

/** A family member is its own style; any other model reads the setting. */
export function styleInUse(settings: Settings, models: Model[]): Style {
  return models.find((model) => model.name === settings.model_size)?.speech_style ?? settings.crisperwhisper_mode;
}

/** A new style swaps a family member for its sibling; CrisperWhisper only changes the setting. */
export function withStyle(settings: Settings, style: Style, models: Model[]): Settings {
  const next = { ...settings, crisperwhisper_mode: style };
  const family = models.find((model) => model.name === settings.model_size)?.style_family;
  const sibling = family ? models.find((model) => model.style_family === family && model.speech_style === style) : undefined;
  return sibling ? { ...next, model_size: sibling.name } : next;
}

export function useEngines(model: string | undefined): Engines | undefined {
  const engines = useQuery({
    queryKey: ["engines", model],
    queryFn: () => api.get<Engines>(`/api/engines?model=${encodeURIComponent(model ?? "")}`),
    enabled: Boolean(model),
  }).data;
  const probed = useRef(false);
  useEffect(() => {
    if (engines?.hardware.probe_pending && !probed.current) {
      probed.current = true;
      void api.post("/api/hardware/probe");
    }
  }, [engines]);
  return engines;
}

export function TranscriptionPage() {
  const { settings, editSettings } = useStore();
  const models = useModels() ?? [];
  const draft = settings?.draft;
  const engines = useEngines(draft?.model_size);
  if (!draft) return <PageBody><PageTitle title="Model" /></PageBody>;

  const hardware = engines?.hardware;
  const liveReason = !hardware || hardware.probe_pending
    ? "Checking for a GPU..."
    : hardware.streaming_supported
      ? ""
      : "Needs a CUDA or ROCm GPU, and none was detected.";
  const shown = visibleModels(
    models.filter((model) => fitsMode(model, draft.transcription_mode)),
    draft.model_size,
    draft.crisperwhisper_mode,
  );
  const selectedModel = models.find((model) => model.name === draft.model_size);
  const hasStyles = selectedModel?.has_speech_styles ?? false;
  const candidates = engines?.candidates ?? [];
  const engineOptions = [
    { value: "auto", label: "Automatic" },
    ...candidates.map((candidate) => ({ value: candidate.id, label: candidate.recommended ? `${candidate.label} (recommended)` : candidate.label })),
  ];
  if (!engineOptions.some((option) => option.value === draft.engine)) engineOptions.push({ value: draft.engine, label: draft.engine });

  return (
    <PageBody>
      <PageTitle title="Model" />
      <Section title="Mode">
        <ChoiceCards
          label="Mode"
          value={draft.transcription_mode}
          onChange={(mode) => editSettings((current) => withMode(current, mode, models))}
          choices={[
            { value: "batch", title: "Batch", description: "Records your dictation, then transcribes it when you stop. Works on any computer." },
            {
              value: "streaming",
              title: "Live",
              description: liveReason || "Types live while you speak (GPU only). Lists only streaming models.",
              disabled: Boolean(liveReason) && draft.transcription_mode !== "streaming",
            },
          ]}
        />
      </Section>
      <Section title="Models">
        <div role="radiogroup" aria-label="Model" className="flex flex-col gap-2">
          {shown.map((model) => (
            <ModelCard
              key={model.name}
              model={model}
              selected={model.name === draft.model_size}
              onSelect={() => editSettings((current) => ({ ...current, model_size: model.name, engine: "auto" }))}
            />
          ))}
          {models.length && !shown.length ? <p className="text-[12.5px] text-muted-foreground">No models for this mode.</p> : null}
        </div>
      </Section>
      <Section title="Speech">
        <Rows>
          <Row title="Language" description="The language you speak in.">
            <Select
              label="Language"
              className="w-56"
              value={draft.language}
              options={LANGUAGES.some((language) => language.value === draft.language) ? LANGUAGES : [...LANGUAGES, { value: draft.language, label: draft.language }]}
              onChange={(language) => editSettings((current) => ({ ...current, language }))}
            />
          </Row>
          <Row
            title="Speech style"
            description={
              !hasStyles
                ? "Only CrisperWhisper and Underdog Bark Plus have speech styles."
                : selectedModel && !selectedModel.installed
                  ? `This style needs ${selectedModel.name}. Download it above, then save.`
                  : "Verbatim keeps fillers and skips the dictionary, spoken commands, and number formatting."
            }
          >
            <Select
              label="Speech style"
              className="w-56"
              value={styleInUse(draft, models)}
              options={STYLES}
              disabled={!hasStyles}
              onChange={(style) => editSettings((current) => withStyle(current, style as Style, models))}
            />
          </Row>
        </Rows>
        <Disclosure label="Advanced" defaultOpen={draft.engine !== "auto"}>
          <Rows>
            <Row title="Engine" description={candidates[0]?.reason ?? ""}>
              <Select
                label="Engine"
                className="w-56"
                value={draft.engine}
                options={engineOptions}
                onChange={(engine) => editSettings((current) => ({ ...current, engine }))}
              />
            </Row>
          </Rows>
        </Disclosure>
      </Section>
    </PageBody>
  );
}

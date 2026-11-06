import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { ModelCard, useModels } from "../components/models";
import { ChoiceCards, Disclosure, PageBody, PageTitle, Row, Rows, Section } from "../components/parts";
import { Select } from "../components/ui/select";
import { Switch } from "../components/ui/switch";
import { api } from "../lib/api";
import { queries } from "../lib/queries";
import { useStore } from "../lib/store";
import type { Engines, Model, Settings, TranscriptionMode } from "../lib/types";

// The languages the core holds Qwen3-ASR to (`LANGUAGE_NAMES`); any other dictates English.
export const LANGUAGES = [
  { value: "en", label: "English" },
  { value: "ar", label: "Arabic" },
];
// The models that hear Arabic, best first (the core's `ARABIC_MODELS`); Jais-ASR hears nothing else as well.
const ARABIC_MODELS = ["jais-asr", "qwen3-asr-1.7b"];
const ARABIC_ONLY = "jais-asr";

/** Picking Arabic keeps a downloaded model that hears it, else picks the best one downloaded; both run in batch mode. */
export function withLanguage(settings: Settings, language: string, models: Model[]): Settings {
  if (language !== "ar") return { ...settings, language };
  const installed = ARABIC_MODELS.filter((name) => models.some((model) => model.name === name && model.installed));
  if (installed.includes(settings.model_size)) return { ...settings, language, transcription_mode: "batch" };
  return { ...settings, language, transcription_mode: "batch", model_size: installed[0] ?? settings.model_size, engine: "auto" };
}

/** Jais-ASR means Arabic; any other model that can't hear Arabic means English. */
export function fitLanguageToModel(settings: Settings): Settings {
  if (settings.model_size === ARABIC_ONLY) return settings.language === "ar" ? settings : { ...settings, language: "ar" };
  return settings.language === "ar" && !ARABIC_MODELS.includes(settings.model_size) ? { ...settings, language: "en" } : settings;
}

const TIERS = ["Small", "Medium", "Large"];

/** Dictation's lineup smallest first, then any model outside it in the order the core sent. */
export function byTier(models: Model[]): Model[] {
  const rank = (model: Model) => (model.tier ? TIERS.indexOf(model.tier) : TIERS.length);
  return [...models].sort((a, b) => rank(a) - rank(b));
}

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

export function useEngines(model: string | undefined): Engines | undefined {
  const engines = useQuery({ ...queries.engines(model ?? ""), enabled: Boolean(model) }).data;
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
  const shown = byTier(models.filter((model) => fitsMode(model, draft.transcription_mode)));
  const arabicReady = models.some((model) => ARABIC_MODELS.includes(model.name) && model.installed);
  const languages = LANGUAGES.map((language) => (language.value === "ar" && !arabicReady ? { ...language, disabled: true } : language));
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
          onChange={(mode) => editSettings((current) => fitLanguageToModel(withMode(current, mode, models)))}
          choices={[
            { value: "batch", title: "Batch", description: "Transcribes when you stop. Works on any PC." },
            {
              value: "streaming",
              title: "Live",
              description: liveReason || "Types live while you speak (GPU only). Lists only streaming models.",
              disabled: Boolean(liveReason) && draft.transcription_mode !== "streaming",
            },
          ]}
        />
      </Section>
      <Section title="Models" note={draft.transcription_mode === "batch" ? "Small, Medium and Large: the larger the model, the fewer mistakes, and the more it needs." : undefined}>
        <div role="radiogroup" aria-label="Model" className="flex flex-col gap-2">
          {shown.map((model) => (
            <ModelCard
              key={model.name}
              model={model}
              selected={model.name === draft.model_size}
              onSelect={() => editSettings((current) => fitLanguageToModel({ ...current, model_size: model.name, engine: "auto" }))}
            />
          ))}
          {models.length && !shown.length ? <p className="text-[12.5px] text-muted-foreground">No models for this mode.</p> : null}
        </div>
      </Section>
      <Section title="Speech">
        <Rows>
          <Row
            title="Language"
            description={arabicReady
              ? "The language you speak in. Arabic runs on Jais-ASR when it is downloaded, else on the Large model."
              : "The language you speak in. Arabic needs Jais-ASR or the Large model: download one above."}
          >
            <Select
              label="Language"
              className="w-56"
              value={draft.language}
              options={languages.some((language) => language.value === draft.language) ? languages : [...languages, { value: draft.language, label: draft.language }]}
              onChange={(language) => editSettings((current) => withLanguage(current, language, models))}
            />
          </Row>
          <Row
            title="Hear questions"
            description="Adds a question mark when Qwen3-ASR almost chose one. Native Qwen3-ASR only."
          >
            <Switch aria-label="Hear questions" checked={draft.question_marks} onCheckedChange={(on) => editSettings((current) => ({ ...current, question_marks: on }))} />
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

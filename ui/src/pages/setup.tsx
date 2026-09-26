import { useEffect, useState } from "react";
import { ModelCard, useModels } from "../components/models";
import { InlineError, Section } from "../components/parts";
import { Button } from "../components/ui/button";
import { api, errorText } from "../lib/api";
import { runBeforeClose, windowControls } from "../lib/platform";
import { useStore } from "../lib/store";
import { MicrophonePicker } from "./microphone";
import { setBindingKeys, ShortcutField, useCapture } from "./shortcuts";

/** First run: pick a model, a microphone and a shortcut, then hand back to the core. */
export function SetupPage() {
  const { settings, editSettings, saveSettings, saveError } = useStore();
  const models = (useModels() ?? []).filter((model) => model.supports_batch);
  const capture = useCapture();
  const [finishing, setFinishing] = useState(false);
  const [error, setError] = useState("");
  const draft = settings?.draft;
  const chosen = models.find((model) => model.name === draft?.model_size);
  const firstInstalled = models.find((model) => model.installed);

  useEffect(() => {
    if (draft && !chosen?.installed && firstInstalled) {
      editSettings((current) => ({ ...current, model_size: firstInstalled.name, transcription_mode: "batch", engine: "auto" }));
    }
  }, [draft, chosen, firstInstalled, editSettings]);

  if (!draft) return null;
  const dictateIndex = draft.hotkey_bindings.findIndex((binding) => binding.action === "dictate");
  const dictate = draft.hotkey_bindings[dictateIndex];

  const finish = async () => {
    setFinishing(true);
    setError("");
    try {
      if (!(await saveSettings())) return;
      await api.post("/api/setup/done");
      await runBeforeClose();
      windowControls.close();
    } catch (failure) {
      setError(errorText(failure));
    } finally {
      setFinishing(false);
    }
  };

  return (
    <div className="mx-auto w-full max-w-[640px] px-8 pt-6 pb-10">
      <h1 className="text-[22px] font-semibold tracking-tight">Welcome to VoiceToText</h1>
      <p className="mt-1 text-[13px] text-muted-foreground">
        Three quick choices and you are ready to dictate. Everything runs on this computer; nothing is sent anywhere.
      </p>

      <Section title="1. Speech model" note="Download one to start. You can add or change models later in Transcription.">
        <div role="radiogroup" aria-label="Speech model" className="flex flex-col gap-2">
          {models.map((model) => (
            <ModelCard
              key={model.name}
              model={model}
              selected={model.name === draft.model_size && model.installed}
              onSelect={() => editSettings((current) => ({ ...current, model_size: model.name, transcription_mode: "batch", engine: "auto" }))}
            />
          ))}
        </div>
      </Section>

      <Section title="2. Microphone">
        <MicrophonePicker settings={draft} onChoose={(name) => editSettings((current) => ({ ...current, input_device: name }))} />
      </Section>

      <Section title="3. Dictation shortcut" note="Press it once to start dictating and again to stop.">
        <ShortcutField
          keys={dictate?.keys ?? ""}
          recording={capture.recordingRow === "setup"}
          onRecord={() =>
            capture.start("setup", (keys) =>
              editSettings((current) => ({
                ...current,
                hotkey_bindings:
                  dictateIndex >= 0 ? setBindingKeys(current.hotkey_bindings, dictateIndex, keys) : [...current.hotkey_bindings, { keys, mode: "toggle", action: "dictate" }],
              })),
            )
          }
          onStop={capture.stop}
        />
        {capture.error ? <InlineError>{capture.error.message}</InlineError> : null}
      </Section>

      <div className="mt-8 flex items-center justify-end gap-3 border-t pt-4">
        {error || saveError ? <span className="mr-auto text-[12.5px] text-destructive">{error || saveError}</span> : null}
        {!chosen?.installed ? <span className="text-[12.5px] text-muted-foreground">Download a model to continue.</span> : null}
        <Button disabled={!chosen?.installed || finishing} onClick={() => void finish()}>
          {finishing ? "Finishing..." : "Finish"}
        </Button>
      </div>
    </div>
  );
}

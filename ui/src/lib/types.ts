export type AppState = "loading" | "idle" | "recording" | "transcribing" | "error";
export type OutputMode = "direct_paste" | "paste_and_clipboard" | "clipboard_only";
export type TranscriptionMode = "batch" | "streaming";
export type Duck = "off" | "lower" | "mute";
export type PasteMethod = "ctrl_v" | "ctrl_shift_v" | "shift_insert" | "type";
export type BindingMode = "toggle" | "hold" | "press";
export type BindingAction = "dictate" | "paste_last" | "copy_last";

export interface Binding {
  keys: string;
  mode: BindingMode;
  action: BindingAction;
}

export interface RecordingSettings {
  warn_when_silent: boolean;
  preferred_devices: string[];
  cues_enabled: boolean;
  cue_volume: number;
  duck_other_audio: Duck;
  duck_level: number;
  max_duration_seconds: number;
  warn_before_cap_seconds: number;
}

export interface Settings {
  model_size: string;
  language: string;
  device: string;
  engine: string;
  output_mode: OutputMode;
  transcription_mode: TranscriptionMode;
  crisperwhisper_mode: "intended" | "verbatim";
  capture_content: boolean;
  input_device: string;
  boost_quiet_speech: boolean;
  learn_words: boolean;
  hotkey_bindings: Binding[];
  double_tap_locks_hold: boolean;
  recording: RecordingSettings;
  history_retention_days: number;
  paste_method_overrides: [string, PasteMethod][];
  check_for_updates: boolean;
}

export interface ChangeSet {
  engine_reload_required: boolean;
  hotkey_changed: boolean;
  output_changed: boolean;
  mode_changed: boolean;
  diagnostics_changed: boolean;
  learning_turned_on: boolean;
  history_retention_changed: boolean;
}

export type DownloadState = "queued" | "downloading" | "paused" | "failed";

export interface Model {
  name: string;
  display_size: string;
  description: string;
  installed: boolean;
  supports_batch: boolean;
  supports_streaming: boolean;
  has_speech_styles: boolean;
  license_id: string | null;
  download: null | { state: DownloadState; bytes_done: number; bytes_total: number; error: string };
}

export interface EngineCandidate {
  id: string;
  label: string;
  recommended: boolean;
  reason: string;
}

export interface Hardware {
  probe_pending: boolean;
  streaming_supported: boolean;
  kind: string;
  device_name: string;
}

export interface Microphone {
  name: string;
  label: string;
  connected: boolean;
  is_default: boolean;
}

export interface LiteralRule {
  kind: "literal";
  spoken: string[];
  replacement: string;
  case_sensitive: boolean;
  whole_words: boolean;
  phonetic: boolean;
  enabled: boolean;
  learned: boolean;
}

export interface RegexRule {
  kind: "regex";
  pattern: string;
  replacement: string;
  enabled: boolean;
}

export type DictionaryRule = LiteralRule | RegexRule;

export interface Dictionary {
  options: { remove_fillers: boolean; spoken_commands: boolean; numbers: boolean; punctuation_model: boolean };
  rules: DictionaryRule[];
}

export type HistoryStatus = "" | "Cancelled" | "Failed" | "Empty" | "Not saved";

export interface HistoryEntry {
  id: string;
  at: string;
  text: string;
  audio_seconds: number;
  status: HistoryStatus;
  error: string | null;
  has_audio: boolean;
  /** What was typed after the dictionary and formatting; empty with history off and in older entries. */
  typed: string;
}

export interface EngineInfo {
  ready: boolean;
  loading: boolean;
  error: string | null;
  model: string;
}

export interface AppInfo {
  version: string;
  state: AppState;
  engine: EngineInfo;
  first_run: boolean;
  diagnostics_directory: string;
  history_directory: string;
  pill_default_position: [number, number];
  update: UpdateStatus | null;
}

export interface UpdateStatus {
  state: "off" | "idle" | "checking" | "up_to_date" | "downloading" | "ready" | "installing" | "updated" | "failed";
  version: string;
  bytes_done: number;
  bytes_total: number;
  error: string;
}

export interface License {
  id: string;
  title: string;
  text: string;
  url?: string;
}

export interface Engines {
  candidates: EngineCandidate[];
  hardware: Hardware;
}

export interface SaveResult {
  changes: ChangeSet;
  engine_reload_started: boolean;
}

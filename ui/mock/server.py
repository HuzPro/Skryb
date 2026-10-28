"""A stdlib stand-in for the Skryb core API (ui_api_contract v1).

Run `python mock/server.py [--port 8765] [--token dev] [--first-run] [--gpu]`
and point the shell at it: `skryb-ui.exe --port 8765 --token dev`.
State lives in memory; downloads, installs, the meter and shortcut capture
are simulated on timers. Nothing is written to disk.
"""

import argparse
import json
import math
import queue
import random
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

GB = 1_000_000_000
TICK_SECONDS = 0.2
DOWNLOAD_SECONDS = 8.0
ENGINE_LOAD_SECONDS = 1.5
HISTORY_SIZE = 2500

LICENSE = {
    "id": "crisperwhisper-nc",
    "title": "CrisperWhisper licence (non-commercial research)",
    "url": "https://huggingface.co/nyralabs/CrisperWhisper2.0_turbo/blob/main/LICENSE.md",
    "text": (
        "CrisperWhisper 2.0 - Non-Commercial Research Licence\n\n"
        "1. You may use the model weights and their outputs for non-commercial research only.\n"
        "2. You may not use the weights or outputs in any product or service offered for money.\n"
        "3. Redistribution must keep this licence and the attribution to nyra labs.\n\n"
    ) + "\n".join(f"{n}. Placeholder clause {n}: the mock repeats this so the dialog scrolls." for n in range(4, 40)),
}

MODELS = [
    {"name": "parakeet-tdt-0.6b-v3", "display_size": "2.5 GB", "size": 2_510_000_000,
     "description": "Best accuracy, built-in punctuation (25 languages)", "installed": True,
     "supports_batch": True, "supports_streaming": False, "has_speech_styles": False, "license_id": None},
    {"name": "nemotron-speech-streaming-en-0.6b", "display_size": "2.5 GB", "size": 2_470_000_000,
     "description": "NVIDIA cache-aware streaming (English, low-latency, GPU)", "installed": False,
     "supports_batch": False, "supports_streaming": True, "has_speech_styles": False, "license_id": None},
    {"name": "crisperwhisper-2.0-turbo", "display_size": "1.6 GB", "size": 1_630_000_000,
     "description": "Fast 0.8B intended/verbatim Whisper; weights/outputs under non-commercial research license",
     "installed": False, "supports_batch": True, "supports_streaming": False, "has_speech_styles": True,
     "license_id": "crisperwhisper-nc"},
    {"name": "crisperwhisper-2.0-large", "display_size": "3.1 GB", "size": 3_100_000_000,
     "description": "Accurate 1.55B intended/verbatim Whisper; weights/outputs under non-commercial research license",
     "installed": False, "supports_batch": True, "supports_streaming": False, "has_speech_styles": True,
     "license_id": "crisperwhisper-nc"},
]

BROKEN_MICROPHONE = "Microphone Array (Realtek(R) A"
USB_MICROPHONE = {"name": "USB Microphone (Røde NT-USB Mi", "label": "USB Microphone (Røde NT-USB Mini)",
                  "connected": True, "is_default": False}
HOT_PLUG_AFTER_SECONDS = 45.0

ENGINES = {
    "parakeet-tdt-0.6b-v3": [("nano_parakeet", "Parakeet (ONNX Runtime)", "Runs on the CPU with ONNX Runtime; no GPU needed.")],
    "nemotron-speech-streaming-en-0.6b": [("nemotron_streaming", "Nemotron streaming (NeMo)", "Needs a CUDA GPU.")],
    "crisperwhisper-2.0-turbo": [
        ("crisperwhisper_ct2", "CrisperWhisper (CTranslate2)", "Fastest on this computer: CTranslate2 on the CPU, no torch needed."),
        ("crisperwhisper", "CrisperWhisper (Transformers)", "Uses torch; slower to start."),
    ],
    "crisperwhisper-2.0-large": [("crisperwhisper", "CrisperWhisper (Transformers)", "Uses torch; the only engine for this model.")],
}

BUILTIN_TRANSFORMS = [
    ("Concise", "Make it concise. Keep every fact, name and number."),
    ("Bullet points", 'Turn it into short bullet points, one idea per line, each starting with "- ".'),
    ("Professional", "Make the tone clear and professional, without adding greetings or new information."),
    ("Casual", "Make the tone relaxed and friendly, like a message to a colleague."),
]

SAMPLE_TEXTS = [
    "Email Sarah about the quarterly report and ask whether the numbers for March are final.",
    "Order: 12 widgets at $4.50 each, ship to the Leeds office by Friday.",
    "日本語のテキストも表示できるか確認してください。",
    "The quick brown fox jumps over the lazy dog. " * 6,
    "Duplicate line",
    "Duplicate line",
    "",
    "Remind me to call the dentist tomorrow at 3.",
    "Café au lait, naïve résumé, déjà vu — and an emoji 🎙️ for good measure.",
    "Meeting notes for Monday:\n- ship the build\n- review the pill animation\n- write the release notes",
    "Можно ли так же диктовать по-русски? Да, конечно.",
    "Fix the flaky test in test_history_management and push it before lunch.",
    "   ",
    "Refactor the tray menu so History opens the new window.",
]


def _default_settings():
    return {
        "model_size": "parakeet-tdt-0.6b-v3", "language": "en", "device": "auto", "engine": "auto",
        "output_mode": "direct_paste", "transcription_mode": "batch", "crisperwhisper_mode": "intended",
        "capture_content": True, "input_device": "", "boost_quiet_speech": True,
        "writing": {"cleanup": "light", "lists": True, "deadline_seconds": 4.0, "context": True,
                    "learn_words": True, "learn_style": True, "style_notes": "",
                    "after_dictation_transform": "", "runtime_path": "", "model_path": ""},
        "hotkey_bindings": [
            {"keys": "ctrl+alt+m", "mode": "toggle", "action": "dictate"},
            {"keys": "ctrl+shift+f9", "mode": "hold", "action": "dictate"},
            {"keys": "mouse4", "mode": "toggle", "action": "dictate"},
            {"keys": "ctrl+alt+e", "mode": "toggle", "action": "edit_selection"},
            {"keys": "ctrl+alt+v", "mode": "press", "action": "paste_last"},
        ],
        "double_tap_locks_hold": True,
        "recording": {"warn_when_silent": True, "preferred_devices": ["Microphone (Blue Yeti)", "Laptop Array Microphone"],
                      "cues_enabled": True, "cue_volume": 0.6, "duck_other_audio": "off", "duck_level": 0.2,
                      "max_duration_seconds": 600.0, "warn_before_cap_seconds": 30.0},
        "history_retention_days": 90,
        "paste_method_overrides": [["notepad.exe", "shift_insert"], ["windowsterminal.exe", "ctrl_shift_v"]],
        "check_for_updates": True,
    }


def _history():
    now = datetime.now().replace(second=0, microsecond=0)
    rng = random.Random(7)
    entries = []
    for index in range(HISTORY_SIZE):
        at = now - timedelta(minutes=17 + index * 37)
        text = SAMPLE_TEXTS[index % len(SAMPLE_TEXTS)]
        status, error, has_audio = "", None, True
        roll = rng.random()
        if not text.strip():
            status = "Empty"
        elif roll < 0.03:
            status, text = "Cancelled", ""
        elif roll < 0.05:
            status, error = "Failed", "The model ran out of memory (CUDA error 2)."
        elif roll < 0.07:
            status, has_audio = "Not saved", False
        text = f"#{HISTORY_SIZE - index} {text}" if text.strip() else text
        kind = "edit" if index % 23 == 5 else "dictation"
        heard, writing = text, None
        if text.strip() and not status and kind == "dictation":
            heard, writing = _writing_record(text, rng)
        entries.append({
            "id": f"{at:%Y%m%dT%H%M%SZ}-{index * 2654435761 % 16**8:08x}.{index % 40 + 3}", "at": at.isoformat(timespec="seconds"),
            "text": heard,
            "audio_seconds": round(1.5 + rng.random() * 40, 1), "status": status, "error": error,
            "has_audio": has_audio, "kind": kind, "writing": writing,
        })
    return entries


def _writing_record(clean: str, rng: random.Random):
    """What the engine heard for *clean*, and what the writing model made of it."""
    heard = "um so " + clean[:1].lower() + clean[1:].rstrip(".")
    roll = rng.random()
    record = {"outcome": "applied", "reason": "", "seconds": round(0.2 + rng.random() * 0.8, 2), "answer": clean,
              "typed": clean, "context_before": "Hi Sarah, " if roll < 0.3 else "", "window": "Outlook (outlook.exe)"}
    if roll < 0.12:
        record.update(outcome="rejected", reason="a number changed", answer=clean.replace("#", "No. ") + " Thanks!", typed=heard)
    elif roll < 0.17:
        record.update(outcome="too_slow", reason="it took 4.62s, over the 4.0s limit", seconds=4.62, typed=heard)
    elif roll < 0.22:
        record.update(outcome="skipped", reason="the writing model is busy", seconds=None, answer="", typed=heard)
    elif roll < 0.27:
        return clean, None  # from before versions were saved
    return heard, record


class Core:
    def __init__(self, first_run: bool, gpu: bool) -> None:
        self.lock = threading.RLock()
        self.clients: list[queue.Queue] = []
        self.settings = _default_settings()
        self.accepted: set[str] = set()
        self.models = [dict(model, download=None) for model in MODELS]
        if first_run:
            for model in self.models:
                model["installed"] = False
        self.app = {"version": "0.9.0", "state": "idle",
                    "engine": {"ready": not first_run, "loading": False, "error": None,
                               "model": "" if first_run else "parakeet-tdt-0.6b-v3"},
                    "first_run": first_run, "diagnostics_directory": "D:\\VoiceToText_logs",
                    "history_directory": "D:\\VoiceToText_logs", "pill_default_position": [0.5, 1.0],
                    "update": {"state": "up_to_date", "version": "", "bytes_done": 0, "bytes_total": 0, "error": ""}}
        self.gpu = gpu
        self.hardware = {"probe_pending": True, "streaming_supported": False, "kind": "", "device_name": ""}
        self.microphones = [
            {"name": "Microphone (Blue Yeti)", "label": "Microphone (Blue Yeti)", "connected": True, "is_default": True},
            {"name": "Headset Microphone (Jabra Evol", "label": "Headset Microphone (Jabra Evolve2 65)", "connected": True, "is_default": False},
            {"name": "Microphone Array (Realtek(R) A", "label": "Microphone Array (Realtek(R) Audio)", "connected": True, "is_default": False},
        ]
        self.dictionary = {
            "options": {"remove_fillers": True, "spoken_commands": True, "numbers": False, "punctuation_model": False},
            "rules": [
                {"kind": "literal", "spoken": ["jason", "jayson"], "replacement": "JSON", "case_sensitive": False,
                 "whole_words": True, "phonetic": False, "enabled": True, "learned": False},
                {"kind": "literal", "spoken": ["huzaifa", "huzeifa"], "replacement": "Huzaifa", "case_sensitive": False,
                 "whole_words": True, "phonetic": True, "enabled": True, "learned": True},
                {"kind": "literal", "spoken": [" dot com"], "replacement": ".com", "case_sensitive": False,
                 "whole_words": True, "phonetic": False, "enabled": True, "learned": False},
                {"kind": "literal", "spoken": ["zurich"], "replacement": "Zürich", "case_sensitive": False,
                 "whole_words": True, "phonetic": False, "enabled": False, "learned": True},
                {"kind": "regex", "pattern": r"\bteh\b", "replacement": "the", "enabled": True},
            ],
        }
        self.writing = {"state": "not_installed", "detail": "", "bytes_done": 0, "bytes_total": 0,
                        "install_bytes": 2_633_286_430,
                        "runtime_path": r"C:\Users\you\AppData\Local\Skryb\writing\llama-b11200\llama-server.exe",
                        "model_path": r"C:\Users\you\AppData\Local\Skryb\writing\models\Spark-X2.5-4B-Q4_K_M.gguf",
                        "last_cleanup_seconds": None,
                        "learned_style_summary": "short sentences, British spelling, no sign-off"}
        self.transforms = [{"name": n, "instruction": i, "builtin": True} for n, i in BUILTIN_TRANSFORMS]
        self.history = _history()
        self.meter_on = False
        self.capturing = False

    # -- events -----------------------------------------------------------
    def emit(self, name: str, data: object) -> None:
        payload = f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()
        with self.lock:
            for client in list(self.clients):
                client.put(payload)

    def later(self, seconds: float, action) -> None:
        timer = threading.Timer(seconds, action)
        timer.daemon = True
        timer.start()

    def model(self, name: str):
        return next((model for model in self.models if model["name"] == name), None)

    def public_models(self):
        return [{key: value for key, value in model.items() if key != "size"} for model in self.models]

    # -- engine -----------------------------------------------------------
    def load_engine(self, model: str) -> None:
        self.app["engine"] = {"ready": False, "loading": True, "error": None, "model": model}
        self.app["state"] = "loading"
        self.emit("engine", {"state": "loading", "loading": True, "ready": False, "error": None, "model": model})
        self.emit("app", self.app)

        def done() -> None:
            with self.lock:
                self.app["engine"] = {"ready": True, "loading": False, "error": None, "model": model}
                self.app["state"] = "idle"
            self.emit("engine", {"state": "ready", "loading": False, "ready": True, "error": None, "model": model})
            self.emit("app", self.app)

        self.later(ENGINE_LOAD_SECONDS, done)

    # -- timers -----------------------------------------------------------
    def tick(self) -> None:
        while True:
            time.sleep(TICK_SECONDS)
            with self.lock:
                self._advance_downloads()
                self._advance_writing()

    def _advance_downloads(self) -> None:
        for model in self.models:
            download = model["download"]
            if not download or download["state"] != "downloading":
                continue
            step = model["size"] * TICK_SECONDS / DOWNLOAD_SECONDS
            download["bytes_done"] = min(download["bytes_total"], download["bytes_done"] + int(step))
            if download["bytes_done"] >= download["bytes_total"]:
                model["installed"], model["download"] = True, None
                self.emit("download", {"name": model["name"], "state": "completed", "bytes_done": download["bytes_total"],
                                       "bytes_total": download["bytes_total"], "error": ""})
            else:
                self.emit("download", {"name": model["name"], **download})

    def _advance_writing(self) -> None:
        status = self.writing
        if status["state"] != "installing":
            return
        status["bytes_done"] = min(status["bytes_total"], status["bytes_done"] + status["bytes_total"] // 50)
        if status["bytes_done"] >= status["bytes_total"]:
            status["state"] = "starting"
            self.later(1.5, self._writing_ready)
        self.emit("writing", status)

    def _writing_ready(self) -> None:
        with self.lock:
            self.writing.update(state="ready", last_cleanup_seconds=0.84)
        self.emit("writing", self.writing)

    def meter(self) -> None:
        started = time.time()
        while True:
            time.sleep(0.05)
            if self.meter_on:
                t = time.time() - started
                level = max(0.0, 0.45 + 0.4 * math.sin(t * 2.3) * math.sin(t * 7.1))
                self.emit("meter", {"level": round(level, 3)})

    def cycle_states(self) -> None:
        """Dictate on a loop, so the logo can be watched recording, transcribing and loading."""
        steps = (("recording", 5.0), ("transcribing", 2.5), ("idle", 2.0), ("loading", 3.0), ("idle", 2.0))
        while True:
            for state, seconds in steps:
                with self.lock:
                    self.app["state"] = state
                self.emit("app", self.app)
                ends = time.time() + seconds
                while time.time() < ends:
                    time.sleep(0.05)
                    if state == "recording":
                        t = time.time()
                        level = max(0.0, 0.5 + 0.5 * math.sin(t * 3.1) * math.sin(t * 8.3))
                        self.emit("voice", {"level": round(level, 3)})

    def hot_plug(self) -> None:
        time.sleep(HOT_PLUG_AFTER_SECONDS)
        with self.lock:
            if not any(mic["name"].startswith("USB") for mic in self.microphones):
                self.microphones.append(dict(USB_MICROPHONE))
        self.emit("microphones", {"name": USB_MICROPHONE["label"], "will_be_used": False})

    def heartbeat(self) -> None:
        while True:
            time.sleep(15)
            self.emit("heartbeat", {})


class Handler(BaseHTTPRequestHandler):
    core: Core
    token: str

    def log_message(self, format: str, *args: object) -> None:
        print(f"{datetime.now():%H:%M:%S.%f}"[:-3], self.command, self.path.split("?token=")[0], flush=True)

    # -- plumbing ---------------------------------------------------------
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Max-Age", "600")

    def _send(self, status: int, body: object = None) -> None:
        data = b"" if body is None else json.dumps(body).encode()
        self.send_response(status)
        self._cors()
        if body is not None:
            self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, **extra: object) -> None:
        self._send(status, {"error": message, **extra})

    def _body(self) -> object:
        return self.body

    def _authorized(self, query: dict) -> bool:
        return self.headers.get("Authorization") == f"Bearer {self.token}" or query.get("token", [""])[0] == self.token

    def do_OPTIONS(self) -> None:
        self._send(204)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        self.body = json.loads(raw) if raw else None
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if not self._authorized(query):
            self._error(401, "Not allowed.")
            return
        parts = [unquote(part) for part in url.path.strip("/").split("/")]
        if parts[:1] != ["api"]:
            self._error(404, "Not found.")
            return
        area, rest = (parts[1] if len(parts) > 1 else ""), parts[2:]
        handler = getattr(self, f"_{area.replace('-', '_')}", None)
        if handler is None:
            self._error(404, "Not found.")
            return
        with_lock = area != "events"
        if with_lock:
            with self.core.lock:
                handler(method, rest, query)
        else:
            handler(method, rest, query)

    # -- endpoints --------------------------------------------------------
    def _events(self, method: str, rest: list, query: dict) -> None:
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        client: queue.Queue = queue.Queue()
        with self.core.lock:
            self.core.clients.append(client)
            client.put(f"event: app\ndata: {json.dumps(self.core.app)}\n\n".encode())
        try:
            while True:
                self.wfile.write(client.get())
                self.wfile.flush()
        except OSError:
            pass
        finally:
            with self.core.lock:
                self.core.clients.remove(client)
            self.close_connection = True

    def _app(self, method, rest, query) -> None:
        self._send(200, self.core.app)

    def _settings(self, method, rest, query) -> None:
        core = self.core
        if rest in (["export"], ["import"]):
            self._send(204)
            return
        if method == "GET":
            self._send(200, core.settings)
            return
        draft = self._body()
        if not any(binding["action"] == "dictate" for binding in draft["hotkey_bindings"]):
            self._error(400, "Add at least one dictation shortcut.")
            return
        model = core.model(draft["model_size"])
        if model is None or not model["installed"]:
            self._error(400, f"'{draft['model_size']}' is not installed. Download it first.")
            return
        if model["license_id"] and model["license_id"] not in core.accepted:
            self._error(409, "This model needs its licence accepted.", license=LICENSE)
            return
        if core.app["state"] in ("recording", "transcribing"):
            self._error(409, "Settings can be saved once the current dictation finishes.")
            return
        before = core.settings
        reload = any(before[key] != draft[key] for key in ("model_size", "engine", "device", "transcription_mode", "crisperwhisper_mode"))
        changes = {
            "engine_reload_required": reload, "hotkey_changed": before["hotkey_bindings"] != draft["hotkey_bindings"],
            "output_changed": before["output_mode"] != draft["output_mode"], "mode_changed": before["transcription_mode"] != draft["transcription_mode"],
            "diagnostics_changed": before["capture_content"] != draft["capture_content"],
            "writing_server_changed": before["writing"] != draft["writing"], "learning_turned_on": False,
            "history_retention_changed": before["history_retention_days"] != draft["history_retention_days"],
        }
        core.settings = draft
        self._send(200, {"changes": changes, "engine_reload_started": reload})
        core.emit("settings", draft)
        if reload:
            core.load_engine(draft["model_size"])

    def _license(self, method, rest, query) -> None:
        self.core.accepted.add(self._body()["id"])
        self._send(204)

    def _models(self, method, rest, query) -> None:
        core = self.core
        if not rest:
            self._send(200, core.public_models())
            return
        model, verb = core.model(rest[0]), rest[1] if len(rest) > 1 else ""
        if model is None:
            self._error(404, "No such model.")
            return
        download = model["download"]
        if verb == "download":
            if model["license_id"] and model["license_id"] not in core.accepted:
                self._error(409, "This model needs its licence accepted.", license=LICENSE)
                return
            model["download"] = {"state": "downloading", "bytes_done": 0, "bytes_total": model["size"], "error": ""}
        elif verb in ("pause", "resume") and download:
            download["state"] = "paused" if verb == "pause" else "downloading"
            core.emit("download", {"name": model["name"], **download})
        elif verb == "cancel" and download:
            model["download"] = None
            core.emit("download", {"name": model["name"], "state": "cancelled", "bytes_done": 0, "bytes_total": 0, "error": ""})
        elif verb == "delete":
            if model["name"] == core.settings["model_size"]:
                self._error(409, "This model is in use. Pick another model and save first.")
                return
            model["installed"] = False
            core.emit("download", {"name": model["name"], "state": "deleted", "bytes_done": 0, "bytes_total": 0, "error": ""})
        self._send(202)

    def _engines(self, method, rest, query) -> None:
        name = query.get("model", [""])[0]
        candidates = [
            {"id": engine_id, "label": label, "recommended": index == 0, "reason": reason}
            for index, (engine_id, label, reason) in enumerate(ENGINES.get(name, []))
        ]
        self._send(200, {"candidates": candidates, "hardware": self.core.hardware})

    def _hardware(self, method, rest, query) -> None:
        core = self.core

        def probed() -> None:
            with core.lock:
                core.hardware = {"probe_pending": False, "streaming_supported": core.gpu,
                                 "kind": "cuda" if core.gpu else "cpu", "device_name": "NVIDIA RTX 4070" if core.gpu else ""}
            core.emit("hardware", core.hardware)

        core.later(1.5, probed)
        self._send(202)

    def _microphones(self, method, rest, query) -> None:
        core = self.core
        if rest == ["meter"]:
            body = self._body()
            if body["on"] and body["device"] == BROKEN_MICROPHONE:
                self._error(400, "This microphone could not be opened.")
                return
            core.meter_on = bool(body["on"])
            self._send(204)
            return
        if query.get("rescan") == ["1"] and not any(mic["name"].startswith("USB") for mic in core.microphones):
            core.microphones.append(dict(USB_MICROPHONE))
        self._send(200, core.microphones)

    def _shortcuts(self, method, rest, query) -> None:
        core = self.core
        body = self._body()
        if rest == ["validate"]:
            ok = any(binding["action"] == "dictate" for binding in body["bindings"])
            self._send(200, {"ok": True}) if ok else self._error(400, "Add at least one dictation shortcut.")
            return
        core.capturing = bool(body["on"])
        if core.capturing:
            def captured() -> None:
                if core.capturing:
                    core.emit("capture", {"keys": "ctrl+shift+alt+m", "display": "Ctrl + Shift + Alt + M"})
            core.later(1.0, captured)
        self._send(204)

    def _dictionary(self, method, rest, query) -> None:
        core = self.core
        if rest == ["forget-learned"]:
            core.dictionary["rules"] = [r for r in core.dictionary["rules"] if not (r["kind"] == "literal" and r["learned"])]
            self._send(204)
        elif method == "DELETE" and rest[:1] == ["learned"] and len(rest) == 2:
            rules = core.dictionary["rules"]
            kept = [r for r in rules if not (r["kind"] == "literal" and r["learned"] and r["spoken"][0].strip() == rest[1].strip())]
            core.dictionary["rules"] = kept
            self._send(204) if len(kept) < len(rules) else self._error(404, "That learned word is no longer in the dictionary.")
        elif method == "PUT":
            core.dictionary = self._body()
            self._send(204)
        else:
            self._send(200, core.dictionary)

    def _writing(self, method, rest, query) -> None:
        core = self.core
        status = core.writing
        action = rest[0] if rest else ""
        if action == "install":
            status.update(state="installing", bytes_done=0, bytes_total=status["install_bytes"], detail="")
        elif action == "cancel":
            status.update(state="not_installed", bytes_done=0, bytes_total=0)
        elif action in ("retry", "restart"):
            status.update(state="starting")
            core.later(1.5, core._writing_ready)
        elif action == "forget-style":
            status["learned_style_summary"] = ""
        elif action == "reset-learning":
            status["learned_style_summary"] = ""
            core.dictionary["rules"] = [r for r in core.dictionary["rules"] if not (r["kind"] == "literal" and r["learned"])]
        elif action == "open-log":
            pass
        elif action == "track-record":
            asked = [entry["writing"] for entry in core.history if entry.get("writing") and entry["writing"]["outcome"]][:50]
            outcomes: dict[str, int] = {}
            reasons: dict[str, int] = {}
            for record in asked:
                outcomes[record["outcome"]] = outcomes.get(record["outcome"], 0) + 1
                if record["outcome"] == "rejected":
                    reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
            rejections = [{"reason": reason, "count": count} for reason, count in sorted(reasons.items(), key=lambda item: -item[1])]
            self._send(200, {"dictations": len(asked), "outcomes": outcomes, "rejections": rejections})
            return
        else:
            self._send(200, status)
            return
        self._send(202 if action in ("install", "cancel", "retry", "restart") else 204)
        core.emit("writing", status)

    def _transforms(self, method, rest, query) -> None:
        core = self.core
        if rest == ["restore-builtins"]:
            custom = [t for t in core.transforms if t["name"] not in dict(BUILTIN_TRANSFORMS)]
            core.transforms = [{"name": n, "instruction": i, "builtin": True} for n, i in BUILTIN_TRANSFORMS] + custom
            self._send(204)
        elif method == "PUT":
            core.transforms = self._body()
            self._send(204)
        else:
            self._send(200, core.transforms)

    def _history(self, method, rest, query) -> None:
        core = self.core
        needle = query.get("query", [""])[0].strip().casefold()
        found = [
            entry for entry in core.history
            if any(needle in text.casefold() for text in (entry["text"], *((entry["writing"] or {}).get(key, "") for key in ("typed", "answer"))))
        ]
        if method == "GET":
            limit = query.get("limit", [""])[0]
            self._send(200, found[:int(limit)] if limit.isdigit() else found)
            return
        if method == "POST" and rest == ["export"]:
            self._send(200, {"rows": len(found)})
            return
        if method == "DELETE" and not rest:
            def wipe() -> None:
                with core.lock:
                    core.history = []
                core.emit("history", {"event": "deleted_all"})
            core.later(1.0, wipe)
            self._send(202)
            return
        entry = next((entry for entry in core.history if entry["id"] == rest[0]), None)
        if entry is None:
            self._error(404, "That dictation is gone.")
            return
        verb = rest[1] if len(rest) > 1 else ""
        if method == "DELETE":
            core.history.remove(entry)
            self._send(204)
        elif verb == "retry":
            def retried() -> None:
                if "ran out of memory" in (entry["error"] or ""):
                    core.emit("history", {"event": "retry_failed", "id": entry["id"], "error": "The model is still out of memory."})
                else:
                    core.emit("history", {"event": "retry_done", "id": entry["id"], "text": (entry["text"] or "Recovered text") + " (retried)"})
            core.later(1.2, retried)
            self._send(202)
        else:
            self._send(204)

    def _logs(self, method, rest, query) -> None:
        self._send(204)

    def _pill(self, method, rest, query) -> None:
        self._send(204)

    def _setup(self, method, rest, query) -> None:
        self.core.app["first_run"] = False
        self._send(204)

    def _updates(self, method, rest, query) -> None:
        """A check finds 9.9.9, downloads its 31 MB in five steps, and offers it; install only says so."""
        core = self.core

        def show(**status) -> None:
            with core.lock:
                core.app["update"] = {"state": "idle", "version": "", "bytes_done": 0, "bytes_total": 0, "error": "", **status}
                core.emit("app", core.app)

        size = 30_972_960
        if rest == ["check"]:
            show(state="checking")
            for step in range(1, 6):
                core.later(0.6 * step, lambda step=step: show(state="downloading", version="9.9.9", bytes_done=size * (step - 1) // 5, bytes_total=size))
            core.later(3.6, lambda: show(state="ready", version="9.9.9"))
        elif rest == ["install"]:
            show(state="installing", version="9.9.9")
        self._send(202)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default="dev")
    parser.add_argument("--first-run", action="store_true", help="no model installed, as on a fresh install")
    parser.add_argument("--gpu", action="store_true", help="the hardware probe finds a GPU, so Live is available")
    parser.add_argument("--cycle-states", action="store_true", help="loop through recording, transcribing and loading")
    args = parser.parse_args()
    core = Core(first_run=args.first_run, gpu=args.gpu)
    for work in (core.tick, core.meter, core.heartbeat, core.hot_plug):
        threading.Thread(target=work, daemon=True).start()
    if args.cycle_states:
        threading.Thread(target=core.cycle_states, daemon=True).start()
    Handler.core, Handler.token = core, args.token
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"mock core on http://127.0.0.1:{args.port} token={args.token}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

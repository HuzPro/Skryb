"""Whisper encoder front end: 30 s windowing and log-mel features.

Every Whisper adapter shares this module, so it must stay free of
``torch`` and ``transformers`` imports - the CTranslate2 adapter exists
to avoid paying for those at startup.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np

from skryb.recording.audio_segment import AudioSegment


# From the model's preprocessor_config.json: 16 kHz mono, 400-sample FFT,
# 160-sample hop, 128 mel bins, 30 s chunks.
_SAMPLE_RATE = 16000
_N_FFT = 400
_HOP_LENGTH = 160
_MEL_BIN_COUNT = 128
# Audio past this never reaches the encoder, so longer recordings are
# split rather than silently losing their tail.
_ENCODER_WINDOW_SECONDS = 30.0
_ENCODER_WINDOW_SAMPLES = int(_ENCODER_WINDOW_SECONDS * _SAMPLE_RATE)
_ENCODER_WINDOW_FRAMES = _ENCODER_WINDOW_SAMPLES // _HOP_LENGTH

# Boundaries are nudged onto the quietest frame in this trailing slice of
# the window so a split lands between words instead of inside one.
_BOUNDARY_SEARCH_SECONDS = 2.0
_BOUNDARY_FRAME_SECONDS = 0.02

# Nyra's longform recipe, the one the model's continuation prompt was
# trained on: neighbouring windows share 4 s of audio, and each window is
# prompted with the last 12 words already transcribed. The last two words
# of a window are the ones its edge cut into and punctuated as if the
# speaker had stopped, so they are dropped for the next window to hear
# whole.
_WINDOW_OVERLAP_SECONDS = 4.0
_CONTEXT_WORDS = 12
_SEAM_WORDS = 2

# Seam words are only dropped when speech reaches this close to the
# window's end, late enough that the next window hears them again. Frames
# quieter than this share of the window's loudest one are not speech.
_SEAM_SPEECH_SECONDS = 2.5
_SPEECH_LEVEL = 0.1

# Nyra's hallucination-repair thresholds: this many back-to-back copies
# of a phrase of 1 to 5 words is a decoding loop, not speech.
_LOOP_COPIES = {1: 8, 2: 8, 3: 4, 4: 3, 5: 3}

# Slaney mel scale: linear below 1 kHz, logarithmic above.
_MEL_LINEAR_STEP_HZ = 200.0 / 3.0
_MEL_LOG_START_HZ = 1000.0
_MEL_LOG_START = _MEL_LOG_START_HZ / _MEL_LINEAR_STEP_HZ
_MEL_LOG_STEP = float(np.log(6.4) / 27.0)

# Whisper's log-mel normalisation: floor the power, clamp to eight
# decades below the peak, then map roughly onto [-1, 1].
_MEL_POWER_FLOOR = 1e-10
_LOG_DYNAMIC_RANGE = 8.0
_LOG_SHIFT = 4.0


def transcribe_in_windows(
    audio: AudioSegment,
    transcribe_window: Callable[[AudioSegment, str], str],
) -> str:
    """Transcribe *audio* one encoder window at a time, as one transcript.

    *transcribe_window* gets each window and the words confirmed so far,
    which it passes to the model as continuation context so a sentence
    that crosses a seam carries on instead of ending there.
    """
    windows = split_into_encoder_windows(audio)
    confirmed: list[str] = []
    seam: list[str] = []
    for index, window in enumerate(windows):
        context = " ".join(confirmed[-_CONTEXT_WORDS:])
        words = _heard_words(window, context, transcribe_window)
        if not words:
            # This window was to hear the seam words again and heard nothing.
            confirmed.extend(seam)
        seam = []
        if index < len(windows) - 1 and _ends_mid_speech(words, window):
            words, seam = words[:-_SEAM_WORDS], words[-_SEAM_WORDS:]
        confirmed.extend(words)
    return " ".join(confirmed)


def _heard_words(
    window: AudioSegment,
    context: str,
    transcribe_window: Callable[[AudioSegment, str], str],
) -> list[str]:
    """What *window* says after *context*, without any decoding loop.

    Context can prime the model to hear words in background audio, and it
    then tends to loop. A window that loops under context and holds no
    words without it is taken as silent.
    """
    words = transcribe_window(window, context).split()
    heard = _without_loop(words)
    if context and len(heard) < len(words) and not transcribe_window(window, "").strip():
        return []
    return heard


def _without_loop(words: list[str]) -> list[str]:
    """*words* cut after the first copy of a phrase the decoder got stuck on.

    Noise after speech can send Whisper round one phrase until it runs out
    of tokens.
    """
    # ponytail: cuts the loop and whatever follows it. Nyra rewinds, bans
    # the token that restarts the loop for one step and decodes on, which
    # needs token-by-token stepping their CTranslate2 build has and the
    # stock wheel lacks. Port that if speech after a loop goes missing.
    for start in range(len(words)):
        for size, copies in _LOOP_COPIES.items():
            phrase = words[start:start + size]
            if words[start:start + size * copies] == phrase * copies:
                return words[:start + size]
    return words


def split_into_encoder_windows(audio: AudioSegment) -> list[AudioSegment]:
    """Slice *audio* into overlapping windows the encoder will accept whole."""
    window_samples = int(_ENCODER_WINDOW_SECONDS * audio.sample_rate)
    overlap_samples = int(_WINDOW_OVERLAP_SECONDS * audio.sample_rate)
    windows: list[AudioSegment] = []
    start = 0
    while start < audio.sample_count:
        end = _window_end(audio, start, window_samples)
        windows.append(
            AudioSegment(
                data=audio.data[start:end],
                sample_rate=audio.sample_rate,
            ),
        )
        if end == audio.sample_count:
            break
        start = end - overlap_samples
    return windows


def _window_end(audio: AudioSegment, start: int, window_samples: int) -> int:
    limit = start + window_samples
    if limit >= audio.sample_count:
        return audio.sample_count
    return _quietest_split_point(audio, start, limit)


def _quietest_split_point(audio: AudioSegment, start: int, limit: int) -> int:
    """Index of the quietest frame boundary near *limit*, else *limit*.

    Cutting on the loudest part of a syllable mangles a word at every
    seam, so the split moves to the lowest-energy frame in the trailing
    search slice. The returned index always advances past *start*, so a
    caller looping on it terminates.
    """
    frame = int(_BOUNDARY_FRAME_SECONDS * audio.sample_rate)
    if frame < 1:
        return limit
    search_samples = int(_BOUNDARY_SEARCH_SECONDS * audio.sample_rate)
    search_start = max(start + frame, limit - search_samples)
    frame_count = (limit - search_start) // frame
    if frame_count < 2:
        return limit
    frames = audio.data[search_start:search_start + frame_count * frame]
    energy = np.abs(frames.reshape(frame_count, frame)).mean(axis=1)
    # The last of the quietest frames, not the first: on a silent gap
    # that keeps the whole gap with this window so the next one opens on
    # speech, and on flat audio it splits as late as possible.
    quietest = int(np.flatnonzero(energy == energy.min())[-1])
    return search_start + (quietest + 1) * frame


def _ends_mid_speech(words: list[str], window: AudioSegment) -> bool:
    # ponytail: a pause-then-one-word ending can still drop a seam word
    # spoken before the next window starts. Nyra times each word from
    # cross-attention instead; do that if seam words go missing.
    return len(words) > _SEAM_WORDS and _speech_in_tail(window)


def _speech_in_tail(window: AudioSegment) -> bool:
    frame = int(_BOUNDARY_FRAME_SECONDS * window.sample_rate)
    frame_count = window.sample_count // frame
    energy = np.abs(
        window.data[:frame_count * frame].reshape(frame_count, frame),
    ).mean(axis=1)
    tail_frames = int(_SEAM_SPEECH_SECONDS * window.sample_rate) // frame
    return bool(energy[-tail_frames:].max() > _SPEECH_LEVEL * energy.max())


def log_mel_spectrogram(audio: AudioSegment) -> np.ndarray:
    """Encoder input for one window: ``(128, 3000)`` float32 log-mel.

    Audio shorter than 30 s is zero-padded and anything past 30 s is
    dropped, matching what the encoder can attend to. Split long
    recordings with :func:`split_into_encoder_windows` first.
    """
    if audio.sample_rate != _SAMPLE_RATE:
        raise ValueError(
            f"Whisper features need {_SAMPLE_RATE} Hz audio, "
            f"got {audio.sample_rate}"
        )
    magnitudes = _power_spectrogram(_padded_to_window(audio.data))
    mel_power = _mel_filter_bank() @ magnitudes
    log_mel = np.log10(np.clip(mel_power, _MEL_POWER_FLOOR, None))
    log_mel = np.maximum(log_mel, log_mel.max() - _LOG_DYNAMIC_RANGE)
    return ((log_mel + _LOG_SHIFT) / _LOG_SHIFT).astype(np.float32)


def _padded_to_window(samples: np.ndarray) -> np.ndarray:
    truncated = samples[:_ENCODER_WINDOW_SAMPLES]
    return np.pad(truncated, (0, _ENCODER_WINDOW_SAMPLES - truncated.size))


def _power_spectrogram(samples: np.ndarray) -> np.ndarray:
    """Windowed power spectra, ``(201, 3000)``, one column per hop."""
    centred = np.pad(samples, _N_FFT // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(centred, _N_FFT)
    frames = frames[::_HOP_LENGTH][:_ENCODER_WINDOW_FRAMES]
    spectra = np.fft.rfft(frames * _hann_window(), axis=-1)
    return np.abs(spectra).T ** 2


@lru_cache(maxsize=1)
def _hann_window() -> np.ndarray:
    return np.hanning(_N_FFT + 1)[:-1]


@lru_cache(maxsize=1)
def _mel_filter_bank() -> np.ndarray:
    """Slaney-normalised triangular filters, ``(128, 201)``."""
    fft_frequencies = np.fft.rfftfreq(_N_FFT, 1.0 / _SAMPLE_RATE)
    band_edges = _mel_to_hertz(
        np.linspace(
            0.0,
            _hertz_to_mel(_SAMPLE_RATE / 2),
            _MEL_BIN_COUNT + 2,
        ),
    )
    edge_widths = np.diff(band_edges)
    ramps = band_edges[:, None] - fft_frequencies[None, :]
    rising = -ramps[:-2] / edge_widths[:-1, None]
    falling = ramps[2:] / edge_widths[1:, None]
    triangles = np.maximum(0.0, np.minimum(rising, falling))
    equal_energy = 2.0 / (band_edges[2:] - band_edges[:-2])
    return triangles * equal_energy[:, None]


def _hertz_to_mel(hertz: float) -> float:
    if hertz < _MEL_LOG_START_HZ:
        return hertz / _MEL_LINEAR_STEP_HZ
    return _MEL_LOG_START + float(
        np.log(hertz / _MEL_LOG_START_HZ) / _MEL_LOG_STEP
    )


def _mel_to_hertz(mel: np.ndarray) -> np.ndarray:
    hertz = mel * _MEL_LINEAR_STEP_HZ
    log_region = mel >= _MEL_LOG_START
    hertz[log_region] = _MEL_LOG_START_HZ * np.exp(
        _MEL_LOG_STEP * (mel[log_region] - _MEL_LOG_START),
    )
    return hertz

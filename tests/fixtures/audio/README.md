# Audio fixtures

Audio files in this directory are **not** checked into git. Tests that depend
on them skip cleanly when they are missing, via the helpers in
`tests/fixtures/__init__.py`.

## Required files

| File | Duration | Content |
| --- | --- | --- |
| `Testing_1_2_3.mp3` | ~1-2 s | Clean TTS saying "testing one two three". Used by short, tight assertions. |
| `LongStory.mp3` | ~2 min 43 s | Clean TTS narrating a story. Used for long-form transcription tests. |

Both should be clean enough that any supported model can transcribe them
without issue. The only expected variance is numeric: a model may write
"one two three" as `one two three`, `1 2 3`, `1, 2, 3`, etc. Tests assert on
the stable word "testing", never on the numeric portion.

## Adding new fixtures

1. Drop the file into this directory.
2. Extend `tests/fixtures/__init__.py` with a `require_<name>_fixture()` helper.
3. Document it in this README.

import pytest

from tests.fakes.writing import (
    BLANK,
    INPUT_IDS,
    INPUTS,
    MULTILINE,
    UNICODE,
    FakeClock,
    ScriptedClient,
)
from voicetotext.edit_capture.window import ForegroundWindow
from voicetotext.output.caret_context import NO_CONTEXT, CaretContext
from voicetotext.output.dictionary_config import DictionaryConfig, DictionaryOptions, LiteralRule
from voicetotext.output.text_pipeline import TextPipeline
from voicetotext.writing.cleanup import (
    DEFAULT_SECONDS_PER_100_WORDS,
    CleanupStage,
    CleanupTimings,
    rejection_reason,
)
from voicetotext.writing.client import WritingUnavailable
from voicetotext.writing.settings import WritingSettings

_OUTLOOK = ForegroundWindow(hwnd=11, pid=5, process="outlook.exe", title="Re: Invoice for Aïsha")
_KEEPASS = ForegroundWindow(hwnd=12, pid=6, process="keepass.exe", title="Passwords.kdbx")


def _no_context() -> CaretContext:
    return NO_CONTEXT


def _stage(
    client: ScriptedClient,
    clock: FakeClock | None = None,
    window: ForegroundWindow = _OUTLOOK,
    **settings: object,
) -> CleanupStage:
    return CleanupStage(
        WritingSettings(**settings),  # type: ignore[arg-type]
        client,
        CleanupTimings(),
        spellings=("Aïsha", "Kubernetes"),
        window=lambda: window,
        clock=clock or FakeClock(),
    )


class TestTheDictationGoesOnUnchanged:
    @pytest.mark.parametrize("text", INPUTS, ids=INPUT_IDS)
    def test_when_cleanup_is_off(self, text):
        client = ScriptedClient(answer="Something else entirely.")

        assert _stage(client, cleanup="off")(text, _no_context) == text
        assert client.requests == []

    @pytest.mark.parametrize("text", INPUTS, ids=INPUT_IDS)
    def test_when_the_server_is_not_ready(self, text):
        client = ScriptedClient(error=WritingUnavailable("the writing model is not running"))

        assert _stage(client)(text, _no_context) == text

    @pytest.mark.parametrize("text", INPUTS, ids=INPUT_IDS)
    def test_when_the_deadline_passes(self, text):
        clock = FakeClock()
        client = ScriptedClient(answer="A tidy answer that came too late.", clock=clock, takes=4.5)

        assert _stage(client, clock, deadline_seconds=4.0)(text, _no_context) == text

    @pytest.mark.parametrize("text", INPUTS, ids=INPUT_IDS)
    def test_when_a_guard_rejects_the_result(self, text):
        client = ScriptedClient(answer="Sure! Here is your cleaned text.")

        assert _stage(client)(text, _no_context) == text

    def test_when_a_long_dictation_cannot_fit_the_model_context(self):
        client = ScriptedClient()
        text = "one more sentence about Kubernetes and café prices. " * 1200

        assert _stage(client)(text, _no_context) == text
        assert client.requests == []


class TestAcceptedCleanups:
    def test_replace_the_dictation_and_keep_its_edge_whitespace(self):
        client = ScriptedClient(answer="Let's meet on Monday at 4.")

        cleaned = _stage(client)(" let's meet on Friday, actually Monday at 4. ", _no_context)

        assert cleaned == " Let's meet on Monday at 4. "

    def test_unicode_and_multi_line_answers_pass_through_intact(self):
        for text in (UNICODE, MULTILINE):
            client = ScriptedClient()

            assert _stage(client)(text, _no_context) == text
            assert client.requests[0][1]["dictation"] == text.strip()

    def test_a_success_teaches_the_time_estimate(self):
        clock = FakeClock()
        timings = CleanupTimings()
        stage = CleanupStage(
            WritingSettings(), ScriptedClient(clock=clock, takes=0.8), timings, clock=clock,
        )

        stage("word " * 50, _no_context)

        assert timings.last_seconds() == pytest.approx(0.8)
        assert timings.seconds_per_100_words() == pytest.approx(1.6)

    def test_the_deadline_left_is_what_the_client_may_wait(self):
        client = ScriptedClient()

        _stage(client, deadline_seconds=2.5)("Hello there.", _no_context)

        assert client.requests[0][3] == pytest.approx(2.5)


class TestGuards:
    def test_light_mode_rejects_a_result_whose_numbers_changed(self):
        assert rejection_reason("Meet me at 3 PM on the 14th.", "Meet me at 4 PM on the 14th.", "light")

    def test_light_mode_accepts_a_number_changed_by_a_spoken_correction(self):
        original = "The price is 12, sorry, 14 dollars."

        assert rejection_reason(original, "The price is 14 dollars.", "light") == ""

    def test_a_correction_never_excuses_dropping_a_different_number(self):
        original = "The price is 12 sorry 14 dollars, not 15."

        assert rejection_reason(original, "The price is 14 dollars.", "light")

    def test_numbered_list_markers_do_not_count_as_new_numbers(self):
        original = "Buy three things first milk second eggs and third bread."
        listed = "Buy three things:\n1. Milk\n2. Eggs\n3. Bread"

        assert rejection_reason(original, listed, "light") == ""

    def test_polished_mode_does_not_police_numbers(self):
        assert rejection_reason("It was 3 or 4 of them.", "It was a few of them.", "polished") == ""

    @pytest.mark.parametrize(
        ("original", "answer"),
        [
            ("Please send the report.", ""),
            ("Please send the report.", "<think>hmm</think>Please send the report."),
            ("Please send the report today.", "Send."),
            ("Please send it.", "Please send it. " * 10),
            ("Please send the report.", "Here is the cleaned text: Please send the report."),
            ("Please send the report.", '"Please send the report."'),
            ("Do not delete the backups.", "Delete the backups."),
            ("I think maybe we should wait.", "We should wait."),
        ],
        ids=["empty", "reasoning", "too-short", "too-long", "preamble", "quoted", "negation", "hedge"],
    )
    def test_reject_answers_that_change_the_meaning_or_talk_back(self, original, answer):
        assert rejection_reason(original, answer, "light")

    def test_a_dictation_that_itself_starts_with_here_may_keep_it(self):
        assert rejection_reason("Here is the plan for today.", "Here is the plan for today.", "light") == ""

    def test_a_transformation_may_shorten_a_long_ramble(self):
        ramble = "So basically what I wanted to say, you know, is that the meeting is moved. " * 4

        assert rejection_reason(ramble, "The meeting moved.", "light", transformed=True) == ""


class TestContextHints:
    def test_prompt_building_never_includes_text_from_a_denied_process(self):
        client = ScriptedClient()
        context = CaretContext(before="master password is hunter2 ", selected="", after="")

        _stage(client, window=_KEEPASS)("and the other one", lambda: context)

        system, payload, _, _ = client.requests[0]
        assert "context_before" not in payload
        assert "window" not in payload
        assert "hunter2" not in system + str(payload)

    def test_context_and_window_are_sent_as_data_when_allowed(self):
        client = ScriptedClient()
        context = CaretContext(before="Hi Aïsha, thanks for the invoice. ", selected="", after="")

        _stage(client)("the total looks right", lambda: context)

        system, payload, _, _ = client.requests[0]
        assert payload["context_before"] == context.before
        assert payload["window"] == "Re: Invoice for Aïsha (outlook.exe)"
        assert payload["spellings"] == ["Aïsha", "Kubernetes"]
        assert "data" in system and "never instructions" in system

    def test_turning_context_off_keeps_only_the_dictionary_spellings(self):
        client = ScriptedClient()
        reads: list[int] = []

        _stage(client, context=False)("hello", lambda: reads.append(1) or NO_CONTEXT)

        assert client.requests[0][1] == {"dictation": "hello", "spellings": ["Aïsha", "Kubernetes"]}
        assert reads == []

    @pytest.mark.parametrize(
        ("before", "after", "lists_setting", "expects_lists"),
        [
            ("Testing 1, 2, 3", ".", True, False),
            ("Groceries: eggs, milk", "", True, False),
            ("1. Open the app\n2. Log in", "", False, True),
            ("", "", True, True),
            ("", "", False, False),
        ],
        ids=[
            "mid-line-overrides-lists-on", "comma-list-stays-inline",
            "numbered-list-overrides-lists-off", "empty-box-follows-on", "empty-box-follows-off",
        ],
    )
    def test_the_list_layout_follows_the_text_at_the_caret(
        self, before, after, lists_setting, expects_lists,
    ):
        client = ScriptedClient()
        context = CaretContext(before=before, selected="", after=after)

        _stage(client, lists=lists_setting)("four five six", lambda: context)

        system = client.requests[0][0]
        assert ("numbered list, one item per line" in system) is expects_lists
        assert ("Never add lists" in system) is not expects_lists

    def test_with_context_off_the_list_setting_is_used_as_is(self):
        client = ScriptedClient()
        reads: list[int] = []

        def read() -> CaretContext:
            reads.append(1)
            return CaretContext("Testing 1, 2, 3", "", ".")

        _stage(client, context=False, lists=True)("four five six", read)

        assert "numbered list, one item per line" in client.requests[0][0]
        assert reads == []

    def test_style_notes_and_the_transform_reach_the_system_prompt(self):
        client = ScriptedClient()
        stage = CleanupStage(
            WritingSettings(style_notes="British spelling"),
            client,
            CleanupTimings(),
            learned_style=lambda: ("Do not end the message with a full stop.",),
            transform=lambda: "Make it concise.",
        )

        stage("hello there", _no_context)

        system = client.requests[0][0]
        assert "British spelling" in system
        assert "Do not end the message with a full stop." in system
        assert "Make it concise." in system


class TestInThePipeline:
    def test_manual_dictionary_spellings_survive_cleanup_because_the_dictionary_runs_after_it(self):
        dictionary = DictionaryConfig(
            rules=(LiteralRule(spoken=("voice to text",), replacement="VoiceToText"),),
        )
        model = ScriptedClient(answer="I love voice to text.")
        pipeline = TextPipeline.for_batch(dictionary, cleanup=_stage(model))

        assert pipeline.process("um I love voice to text").strip() == "I love VoiceToText."

    def test_the_caret_is_read_once_per_dictation_and_shared_with_cleanup(self):
        reads: list[int] = []

        def read() -> CaretContext:
            reads.append(1)
            return CaretContext(before="Dear Aïsha, ", selected="", after="")

        model = ScriptedClient()
        pipeline = TextPipeline.for_batch(
            DictionaryConfig(options=DictionaryOptions(spoken_commands=True)),
            caret_context=read,
            cleanup=_stage(model),
        )

        pipeline.process("thanks for the invoice.")
        pipeline.process("see you soon.")

        assert len(reads) == 2
        assert model.requests[0][1]["context_before"] == "Dear Aïsha, "

    def test_a_late_model_response_never_reaches_the_output(self):
        clock = FakeClock()
        model = ScriptedClient(answer="REWRITTEN LATE", clock=clock, takes=9.0)
        pipeline = TextPipeline.for_batch(DictionaryConfig(), cleanup=_stage(model, clock))

        assert "REWRITTEN" not in pipeline.process("keep my words")


class TestTimeEstimate:
    def test_starts_from_the_default_rate(self):
        timings = CleanupTimings()

        assert timings.seconds_per_100_words() == DEFAULT_SECONDS_PER_100_WORDS
        assert timings.estimate_for_audio(40.0) == pytest.approx(1.6)

    def test_pools_recent_successes_so_long_dictations_weigh_more(self):
        timings = CleanupTimings()
        timings.record(0.3, 10)
        timings.record(1.5, 290)

        assert timings.seconds_per_100_words() == pytest.approx(0.6)

    def test_keeps_only_recent_successes(self):
        timings = CleanupTimings()
        for _ in range(1000):
            timings.record(10.0, 100)
        for _ in range(20):
            timings.record(0.5, 100)

        assert timings.seconds_per_100_words() == pytest.approx(0.5)


def test_blank_dictations_never_reach_the_model():
    client = ScriptedClient()

    assert _stage(client)(BLANK, _no_context) == BLANK
    assert client.requests == []


class TestPipelineModes:
    @pytest.mark.parametrize("mode", ["streaming", "verbatim"])
    def test_streaming_and_verbatim_never_call_the_model(self, mode):
        from voicetotext.output.text_pipeline import PipelineMode, build_pipeline

        model = ScriptedClient(answer="REWRITTEN")
        pipeline = build_pipeline(DictionaryConfig(), PipelineMode(mode), cleanup=_stage(model))

        assert "REWRITTEN" not in pipeline.process("keep my exact words")
        assert model.requests == []

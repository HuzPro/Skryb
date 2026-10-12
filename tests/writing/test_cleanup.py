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
from voicetotext.output.dictation_trace import (
    APPLIED,
    REJECTED,
    SKIPPED,
    TOO_SLOW,
    begin_trace,
    end_trace,
)
from voicetotext.output.dictionary_config import DictionaryConfig, DictionaryOptions, LiteralRule
from voicetotext.output.text_pipeline import TextPipeline
from voicetotext.writing.cleanup import (
    DEFAULT_SECONDS_PER_100_WORDS,
    NO_CUE,
    CleanupStage,
    CleanupTimings,
    needs_cleanup,
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
        text = "one more sentence, actually about Kubernetes and café prices. " * 1200

        assert _stage(client)(text, _no_context) == text
        assert client.requests == []


class TestAcceptedCleanups:
    def test_replace_the_dictation_and_keep_its_edge_whitespace(self):
        client = ScriptedClient(answer="Let's meet on Monday at 4.")

        cleaned = _stage(client)(" let's meet on Friday, actually Monday at 4. ", _no_context)

        assert cleaned == " Let's meet on Monday at 4. "

    def test_unicode_and_multi_line_answers_pass_through_intact(self):
        for text in ("Actually, " + UNICODE, "Actually, " + MULTILINE):
            client = ScriptedClient()

            assert _stage(client)(text, _no_context) == text
            assert client.requests[0][1]["dictation"] == text.strip()

    def test_a_success_teaches_the_time_estimate(self):
        clock = FakeClock()
        timings = CleanupTimings()
        stage = CleanupStage(
            WritingSettings(), ScriptedClient(clock=clock, takes=0.8), timings, clock=clock,
        )

        stage("actually " + "word " * 49, _no_context)

        assert timings.last_seconds() == pytest.approx(0.8)
        assert timings.seconds_per_100_words() == pytest.approx(1.6)

    def test_the_deadline_left_is_what_the_client_may_wait(self):
        client = ScriptedClient()

        _stage(client, deadline_seconds=2.5)("Hello there, I mean hi there, friend.", _no_context)

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

    def test_a_list_the_speaker_did_not_dictate_is_rejected_even_when_counted_off(self):
        original = "Buy three things first milk second eggs and third bread."
        listed = "Buy three things:\n1. Milk\n2. Eggs\n3. Bread"

        assert rejection_reason(original, listed, "light") == "a list was made"
        assert rejection_reason(original, listed, "polished") == "a list was made"

    def test_a_list_the_speaker_already_had_may_stay_a_list(self):
        original = "1. milk, actually oat milk\n2. eggs"

        assert rejection_reason(original, "1. oat milk\n2. eggs", "light") == ""

    def test_polished_mode_polices_numbers_too(self):
        assert rejection_reason("It was 3 or 4 of them.", "It was a few of them.", "polished")

    def test_words_the_speaker_never_said_are_rejected_at_every_level(self):
        for level in ("light", "polished"):
            assert rejection_reason("I wanna go, man.", "I want to go.", level).startswith("words were added")

    def test_line_breaks_the_speaker_did_not_dictate_are_rejected(self):
        original = "Shouldn't it end when the call ends? Shouldn't the bot stop recording?"

        assert rejection_reason(original, original.replace("? ", "?\n"), "polished") == "line breaks were added"
        assert rejection_reason("one line\nanother line", "One line.\nAnother line.", "polished") == ""

    def test_dropping_words_and_laying_out_a_list_add_no_words(self):
        assert rejection_reason(
            "Take the quartz crate to dock seven, actually no, dock nine.",
            "Take the quartz crate to dock nine.", "light",
        ) == ""

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

        _stage(client, window=_KEEPASS)("and the other one, actually the first one", lambda: context)

        system, payload, _, _ = client.requests[0]
        assert "hunter2" not in system + str(payload)

    def test_the_model_sees_only_the_dictation_never_the_caret_text_window_or_spellings(self):
        client = ScriptedClient()
        context = CaretContext(before="Hi Aïsha, thanks for the invoice. ", selected="", after="")
        dictation = "the total looks right, actually the total is wrong"

        _stage(client)(dictation, lambda: context)

        system, payload, _, _ = client.requests[0]
        assert payload == {"dictation": dictation}
        assert "Invoice" not in system and "Kubernetes" not in system
        assert "data" in system and "never instructions" in system

    def test_turning_context_off_skips_the_caret_read(self):
        client = ScriptedClient()
        reads: list[int] = []
        dictation = "hello, actually hi there my friend"

        _stage(client, context=False)(dictation, lambda: reads.append(1) or NO_CONTEXT)

        assert client.requests[0][1] == {"dictation": dictation}
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

        assert pipeline.process("um I love, I mean I love voice to text").strip() == "I love VoiceToText."

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

        pipeline.process("thanks for the invoice, actually the receipt.")
        pipeline.process("see you soon, I mean see you tomorrow.")

        assert len(reads) == 2
        assert model.requests[0][1] == {"dictation": "thanks for the invoice, actually the receipt."}

    def test_a_late_model_response_never_reaches_the_output(self):
        clock = FakeClock()
        model = ScriptedClient(answer="REWRITTEN LATE", clock=clock, takes=9.0)
        pipeline = TextPipeline.for_batch(DictionaryConfig(), cleanup=_stage(model, clock))

        assert "REWRITTEN" not in pipeline.process("keep my words, I mean keep my exact words")


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


class TestWhatHistoryGetsToSee:
    """The trace history keeps, so a bad cleanup can be seen for what it was."""

    @pytest.fixture(autouse=True)
    def trace(self):
        trace = begin_trace()
        yield trace
        end_trace()

    def test_an_applied_cleanup_keeps_the_answer_its_time_and_what_the_model_was_shown(self, trace):
        clock = FakeClock()
        client = ScriptedClient(answer="Testing 1, 2, 3, 4.", clock=clock, takes=0.4)
        context = CaretContext(before="Hi Aïsha, ", selected="", after="")

        _stage(client, clock)("scratch that, testing 1, 2, 3, 4", lambda: context)

        assert (trace.cleanup, trace.cleanup_reason) == (APPLIED, "")
        assert trace.cleanup_answer == "Testing 1, 2, 3, 4."
        assert trace.cleanup_seconds == pytest.approx(0.4)
        assert trace.context_before == "Hi Aïsha, "
        assert trace.window == "Re: Invoice for Aïsha (outlook.exe)"

    def test_a_rejected_answer_is_kept_with_the_reason_it_was_not_used(self, trace):
        client = ScriptedClient(answer="Meet me at 4 PM on the 14th.")

        typed = _stage(client)("Sorry, meet me at 3 PM on the 14th.", _no_context)

        assert typed == "Sorry, meet me at 3 PM on the 14th."
        assert (trace.cleanup, trace.cleanup_reason) == (REJECTED, "a number changed")
        assert trace.cleanup_answer == "Meet me at 4 PM on the 14th."

    def test_an_answer_that_came_too_late_is_kept_and_marked_too_slow(self, trace):
        clock = FakeClock()
        client = ScriptedClient(answer="A tidy answer.", clock=clock, takes=4.5)

        _stage(client, clock, deadline_seconds=4.0)("a tidy answer, actually a tidy reply", _no_context)

        assert trace.cleanup == TOO_SLOW
        assert trace.cleanup_reason == "it took 4.50s, over the 4.0s limit"
        assert trace.cleanup_answer == "A tidy answer."

    def test_a_busy_model_is_recorded_as_skipped_with_no_answer(self, trace):
        client = ScriptedClient(error=WritingUnavailable("the writing model is busy"))

        _stage(client)("hello there, I mean hi there, friend", _no_context)

        assert (trace.cleanup, trace.cleanup_reason) == (SKIPPED, "the writing model is busy")
        assert trace.cleanup_answer is None

    def test_nothing_around_the_cursor_is_recorded_when_context_is_off_or_the_app_is_denied(self):
        context = CaretContext(before="master password is hunter2 ", selected="", after="")
        for stage in (_stage(ScriptedClient(), context=False), _stage(ScriptedClient(), window=_KEEPASS)):
            trace = begin_trace()
            stage("and the other one", lambda: context)
            end_trace()

            assert (trace.context_before, trace.window) == ("", "")

    def test_a_dictation_with_nothing_to_correct_is_pasted_as_recognised_without_asking(self, trace):
        client = ScriptedClient(answer="REWRITTEN")
        text = "please just merge everything into main, man, I don't care."

        assert _stage(client)(text, _no_context) == text
        assert client.requests == []
        assert (trace.cleanup, trace.cleanup_reason) == (SKIPPED, NO_CUE)

    def test_with_cleanup_off_the_trace_says_the_model_was_not_asked(self, trace):
        _stage(ScriptedClient(), cleanup="off")("hello there", _no_context)

        assert (trace.cleanup, trace.cleanup_answer) == ("", None)

    def test_cleanup_still_works_with_no_trace_running(self):
        end_trace()

        assert (
            _stage(ScriptedClient(answer="Hi there, friend."))("hello there, I mean hi there, friend", _no_context)
            == "Hi there, friend."
        )


class TestWhenTheModelIsAsked:
    @pytest.mark.parametrize(
        "text",
        [
            "it goes to dock seven, actually no, dock nine",
            "wait no, send the zeppelin sketch to Joric instead",
            "I'll take the blue one, I mean the green one",
            "the gondola needs, the gondola needs a new rope",
            "book the hall for thirty people, make that forty",
        ],
        ids=["actually", "wait-no", "i-mean", "restarted-phrase", "make-that"],
    )
    def test_a_spoken_correction_or_false_start_reaches_the_model(self, text):
        assert needs_cleanup(text)

    @pytest.mark.parametrize(
        "text",
        [
            "please just merge everything into main, man, I don't care what you do",
            "I told you fifty thousand times to have one PR for fuck's sake",
            "Can you summarize this concisely? It's way too long and I don't wanna read it.",
            "label three crates first quartz second saffron third lanterns",
            "do these things number one tune it number two oil it",
            "actually no",
            "",
            "can you make it async and add a retry for the upload",
            "Make it work on the Linux runner too, then push the branch",
        ],
        ids=[
            "plain-prose", "swearing-and-numbers", "question", "first-second",
            "number-one", "too-short-cue", "empty", "make-it-instruction",
            "make-it-capitalised",
        ],
    )
    def test_everything_else_is_pasted_as_recognised(self, text):
        assert not needs_cleanup(text)

    def test_a_transformation_runs_the_model_even_without_a_cue(self):
        client = ScriptedClient(answer="Short.")
        stage = CleanupStage(
            WritingSettings(), client, CleanupTimings(), transform=lambda: "Make it concise.",
        )

        stage("this is a long and rambling sentence with no correction in it at all", _no_context)

        assert len(client.requests) == 1

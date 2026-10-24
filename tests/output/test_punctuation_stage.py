from skryb.output.dictionary_config import DictionaryConfig, DictionaryOptions
from skryb.output.punctuation_stage import PunctuationStage
from skryb.output.text_pipeline import TextPipeline


class FakePunctuationBackend:
    def __init__(self, result: str) -> None:
        self._result = result
        self.inputs: list[str] = []

    def punctuate(self, text: str) -> str:
        self.inputs.append(text)
        return self._result


def test_should_delegate_unpunctuated_text_to_the_backend() -> None:
    backend = FakePunctuationBackend("Hello, world.")

    result = PunctuationStage(backend).apply("hello world")

    assert result == "Hello, world."
    assert backend.inputs == ["hello world"]


def test_should_skip_text_with_existing_terminal_punctuation() -> None:
    backend = FakePunctuationBackend("unused")
    stage = PunctuationStage(backend)

    assert stage.apply("Already. punctuated") == "Already. punctuated"
    assert stage.apply("Really?") == "Really?"
    assert stage.apply("Yes!") == "Yes!"
    assert backend.inputs == []


def test_should_degrade_to_pass_through_without_a_backend() -> None:
    assert PunctuationStage(None).apply("hello world") == "hello world"


def test_should_run_punctuation_before_all_other_batch_stages() -> None:
    backend = FakePunctuationBackend("twenty percent")
    config = DictionaryConfig(options=DictionaryOptions(
        numbers=True, punctuation_model=True,
    ))

    result = TextPipeline.for_batch(config, punctuation_backend=backend).process(
        "ignored input"
    )

    assert result == "20%"
    assert backend.inputs == ["ignored input"]

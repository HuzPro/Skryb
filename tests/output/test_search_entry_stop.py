import pytest

from skryb.output.dictionary_config import DictionaryConfig
from skryb.output.text_pipeline import PipelineMode, build_pipeline, drop_search_entry_stop


class TestDroppingTheFullStopOfAShortEntry:
    @pytest.mark.parametrize("process", ["zen.exe", "chrome.exe", "msedge.exe", "Firefox.EXE", "skryb-ui.exe"])
    def test_a_few_words_in_a_browser_or_skryb_lose_the_full_stop_and_space(self, process):
        assert drop_search_entry_stop("Open data. ", process) == "Open data"

    def test_a_one_word_entry_loses_it_too(self):
        assert drop_search_entry_stop("Reykjavik. ", "zen.exe") == "Reykjavik"

    def test_a_four_word_entry_is_still_an_entry(self):
        assert drop_search_entry_stop("GitHub developer pack pricing. ", "zen.exe") == "GitHub developer pack pricing"

    def test_an_address_loses_the_stop_after_it(self):
        assert drop_search_entry_stop("https://github.com/HuzPro. ", "zen.exe") == "https://github.com/HuzPro"

    def test_unicode_words_count_like_any_other(self):
        assert drop_search_entry_stop("Café résumé naïve. ", "zen.exe") == "Café résumé naïve"

    def test_keeps_the_space_skryb_put_in_front(self):
        assert drop_search_entry_stop(" bright data. ", "zen.exe") == " bright data"


class TestLeavingSentencesAlone:
    def test_five_words_are_a_sentence(self):
        assert drop_search_entry_stop("I will send it tomorrow. ", "zen.exe") == "I will send it tomorrow. "

    def test_two_sentences_stay_as_they_are(self):
        assert drop_search_entry_stop("Okay. Sounds good. ", "zen.exe") == "Okay. Sounds good. "

    @pytest.mark.parametrize("text", ["How long? ", "Wow! ", "Wait... "])
    def test_questions_exclamations_and_ellipses_keep_their_mark(self, text):
        assert drop_search_entry_stop(text, "zen.exe") == text

    @pytest.mark.parametrize("process", ["claude.exe", "discord.exe", "slack.exe", "whatsapp.root.exe", "", "Code.exe"])
    def test_chat_apps_and_editors_keep_the_full_stop(self, process):
        assert drop_search_entry_stop("Sounds good. ", process) == "Sounds good. "

    def test_a_lone_full_stop_is_not_emptied(self):
        assert drop_search_entry_stop(". ", "zen.exe") == ". "

    def test_blank_text_stays_blank(self):
        assert drop_search_entry_stop("", "zen.exe") == ""
        assert drop_search_entry_stop("   ", "zen.exe") == "   "

    def test_an_abbreviation_inside_counts_as_a_sentence_break(self):
        assert drop_search_entry_stop("Mr. Smith. ", "zen.exe") == "Mr. Smith. "


class TestInThePipeline:
    def test_the_batch_pipeline_drops_it_for_the_browser_it_pastes_into(self):
        pipeline = build_pipeline(DictionaryConfig(), PipelineMode.BATCH, target_app=lambda: "zen.exe")

        assert pipeline.process("Open data.") == "Open data"

    def test_the_batch_pipeline_keeps_it_for_a_chat_app(self):
        pipeline = build_pipeline(DictionaryConfig(), PipelineMode.BATCH, target_app=lambda: "claude.exe")

        assert pipeline.process("Open data.") == "Open data. "

    def test_a_thousand_dictations_in_a_row_each_ask_for_the_app_once(self):
        asked = []
        pipeline = build_pipeline(
            DictionaryConfig(), PipelineMode.BATCH, target_app=lambda: asked.append(1) or "zen.exe",
        )

        results = {pipeline.process(f"Entry {index}.") for index in range(1000)}

        assert len(asked) == 1000
        assert all(not result.endswith(".") for result in results)

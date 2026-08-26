import pytest

from voicetotext.settings.tk_pages.dictionary_page import TkDictionaryPage


@pytest.fixture
def page(tk_root):
    widget = TkDictionaryPage(tk_root)
    yield widget
    widget.destroy()


def _type_rule(page: TkDictionaryPage, spoken: str, replacement: str) -> None:
    page.spoken_entry().delete(0, "end")
    page.spoken_entry().insert(0, spoken)
    page.replacement_entry().delete(0, "end")
    page.replacement_entry().insert(0, replacement)


class TestInitialRules:
    def test_should_list_the_rules_it_was_given(self, page):
        page.set_rules({"god": "God", "python": "Python"})

        assert page.rule_labels() == ["god  ->  God", "python  ->  Python"]

    def test_should_start_clean(self, page):
        page.set_rules({"god": "God"})

        assert page.is_dirty() is False


class TestAdd:
    def test_should_stage_a_new_rule(self, page):
        _type_rule(page, "god", "God")

        page.add_button().invoke()

        assert page.rules() == {"god": "God"}
        assert page.rule_labels() == ["god  ->  God"]

    def test_should_mark_the_page_dirty(self, page):
        _type_rule(page, "god", "God")

        page.add_button().invoke()

        assert page.is_dirty() is True

    def test_should_clear_the_entry_fields(self, page):
        _type_rule(page, "god", "God")

        page.add_button().invoke()

        assert page.spoken_entry().get() == ""
        assert page.replacement_entry().get() == ""

    def test_should_overwrite_an_existing_spoken_form(self, page):
        page.set_rules({"god": "God"})
        _type_rule(page, "god", "Deity")

        page.add_button().invoke()

        assert page.rules() == {"god": "Deity"}


class TestAddValidation:
    def test_should_reject_a_multi_word_spoken_form(self, page):
        _type_rule(page, "oh god", "Oh God")

        page.add_button().invoke()

        assert page.rules() == {}
        assert "one word" in page.message()

    def test_should_reject_an_empty_replacement(self, page):
        _type_rule(page, "god", "")

        page.add_button().invoke()

        assert page.rules() == {}
        assert "required" in page.message()

    def test_should_reject_an_empty_spoken_form(self, page):
        _type_rule(page, "", "God")

        page.add_button().invoke()

        assert page.rules() == {}
        assert "required" in page.message()

    def test_should_leave_the_page_clean_after_invalid_input(self, page):
        _type_rule(page, "oh god", "Oh God")

        page.add_button().invoke()

        assert page.is_dirty() is False


class TestEdit:
    def test_should_replace_the_selected_rule_in_place(self, page):
        page.set_rules({"god": "God", "python": "Python"})
        page.select_rule(0)
        _type_rule(page, "god", "Deity")

        page.update_button().invoke()

        assert page.rules() == {"god": "Deity", "python": "Python"}

    def test_should_rename_the_spoken_form_keeping_its_position(self, page):
        page.set_rules({"god": "God", "python": "Python"})
        page.select_rule(0)
        _type_rule(page, "gosh", "Gosh")

        page.update_button().invoke()

        assert list(page.rules()) == ["gosh", "python"]

    def test_should_populate_the_entries_when_a_rule_is_selected(self, page):
        page.set_rules({"god": "God"})

        page.select_rule(0)

        assert page.spoken_entry().get() == "god"
        assert page.replacement_entry().get() == "God"

    def test_should_report_when_nothing_is_selected(self, page):
        page.set_rules({"god": "God"})
        _type_rule(page, "god", "Deity")

        page.update_button().invoke()

        assert page.rules() == {"god": "God"}
        assert "select a rule" in page.message().lower()

    def test_should_reject_a_multi_word_spoken_form(self, page):
        page.set_rules({"god": "God"})
        page.select_rule(0)
        _type_rule(page, "oh god", "Oh God")

        page.update_button().invoke()

        assert page.rules() == {"god": "God"}
        assert "one word" in page.message()

    def test_should_refuse_to_overwrite_another_rule_by_renaming(self, page):
        page.set_rules({"god": "God", "python": "Python"})
        page.select_rule(0)
        _type_rule(page, "python", "Snake")

        page.update_button().invoke()

        assert page.rules() == {"god": "God", "python": "Python"}
        assert "already a rule" in page.message()


class TestDelete:
    def test_should_remove_the_selected_rule(self, page):
        page.set_rules({"god": "God", "python": "Python"})
        page.select_rule(0)

        page.delete_button().invoke()

        assert page.rules() == {"python": "Python"}
        assert page.rule_labels() == ["python  ->  Python"]

    def test_should_mark_the_page_dirty(self, page):
        page.set_rules({"god": "God"})
        page.select_rule(0)

        page.delete_button().invoke()

        assert page.is_dirty() is True

    def test_should_report_when_nothing_is_selected(self, page):
        page.set_rules({"god": "God"})

        page.delete_button().invoke()

        assert page.rules() == {"god": "God"}
        assert "select a rule" in page.message().lower()

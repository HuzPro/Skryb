import json

from skryb.writing.transforms import (
    BUILTIN_TRANSFORMS,
    MAX_TRANSFORMS,
    TransformStore,
)


def test_a_fresh_store_offers_the_four_built_ins(tmp_path):
    store = TransformStore(tmp_path / "transforms.json")

    assert list(store.all()) == ["Concise", "Bullet points", "Professional", "Casual"]
    assert store.instruction("Concise") == BUILTIN_TRANSFORMS["Concise"]
    assert store.instruction("") == ""
    assert store.instruction("Missing") == ""


def test_saved_edits_and_removals_survive_a_restart(tmp_path):
    path = tmp_path / "transforms.json"
    TransformStore(path).save({"Casual": "Relaxed, keep facts.", "Résumé ✓": "Summarise in one line."})

    reopened = TransformStore(path)

    assert reopened.all() == {"Casual": "Relaxed, keep facts.", "Résumé ✓": "Summarise in one line."}


def test_blank_oversized_and_excess_entries_are_dropped(tmp_path):
    store = TransformStore(tmp_path / "transforms.json")
    entries = {f"Style {index}": f"Instruction {index}" for index in range(200)}
    entries.update({"": "no name", "No text": "   ", "Long": "x" * 5000})

    store.save(entries)

    saved = TransformStore(tmp_path / "transforms.json").all()
    assert len(saved) == MAX_TRANSFORMS
    assert "" not in saved and "No text" not in saved and "Long" not in saved


def test_a_corrupt_file_falls_back_to_the_built_ins(tmp_path):
    path = tmp_path / "transforms.json"
    path.write_text("{ not json", encoding="utf-8")
    assert TransformStore(path).all() == BUILTIN_TRANSFORMS

    path.write_text(json.dumps({"transforms": ["not", "a", "mapping"]}), encoding="utf-8")
    assert TransformStore(path).all() == BUILTIN_TRANSFORMS

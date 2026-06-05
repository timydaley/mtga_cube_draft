"""Tests for published cube-list name resolution (numpy/torch-free)."""

from __future__ import annotations

from cube_draft.cards.card import Card
from cube_draft.cards.vocab import CardVocab
from cube_draft.data import cube_list


def _vocab() -> CardVocab:
    cards = [
        Card(oracle_id="oid-ragavan", name="Ragavan, Nimble Pilferer", type_line="Creature"),
        Card(oracle_id="oid-fable", name="Fable of the Mirror-Breaker // Reflection of Kiki-Rik",
             type_line="Enchantment"),
        Card(oracle_id="oid-bolt", name="Lightning Bolt", type_line="Instant"),
        Card(oracle_id="oid-yawg", name="Yawgmoth's Will", type_line="Sorcery"),
        Card(oracle_id="oid-life", name="Life // Death", type_line="Sorcery"),
    ]
    return CardVocab(cards)


def test_normalize_name():
    assert cube_list.normalize_name("Ragavan, Nimble Pilferer") == "ragavan nimble pilferer"
    assert cube_list.normalize_name("Yawgmoth's Will") == "yawgmoths will"
    assert cube_list.normalize_name("Life // Death") == "life death"


def test_resolves_comma_and_apostrophe_variants():
    res = cube_list.resolve_names(_vocab(), ["Ragavan Nimble Pilferer", "Yawgmoths Will"])
    assert res.oracle_ids == ["oid-ragavan", "oid-yawg"]
    assert not res.unmatched


def test_resolves_dfc_front_face():
    # The published list gives only the front face of a DFC/adventure card.
    res = cube_list.resolve_names(_vocab(), ["Fable of the Mirror-Breaker"])
    assert res.oracle_ids == ["oid-fable"]
    assert not res.unmatched


def test_unmatched_reported_not_dropped_silently():
    res = cube_list.resolve_names(_vocab(), ["Lightning Bolt", "Totally Fake Card"])
    assert res.oracle_ids == ["oid-bolt"]
    assert res.unmatched == ["Totally Fake Card"]


def test_duplicates_skipped():
    res = cube_list.resolve_names(_vocab(), ["Lightning Bolt", "lightning bolt"])
    assert res.oracle_ids == ["oid-bolt"]
    assert res.duplicates == ["lightning bolt"]


def test_parse_name_file_ignores_comments_and_blanks():
    text = "# header\n\nLightning Bolt\n  Ragavan, Nimble Pilferer  \n# trailing\n"
    assert cube_list.parse_name_file(text) == ["Lightning Bolt", "Ragavan, Nimble Pilferer"]

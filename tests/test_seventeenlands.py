"""Tests for the 17lands teacher signal (join + value array + bot) — no network."""

from __future__ import annotations

import numpy as np

from cube_draft.bots.seventeenlands import SeventeenLandsBot
from cube_draft.cards.card import Card
from cube_draft.cards.vocab import CardVocab, CubeVocab
from cube_draft.data import seventeenlands as sl


def _vocab_and_cube():
    cards = [
        Card(oracle_id="oid-bolt", name="Lightning Bolt", type_line="Instant"),
        Card(oracle_id="oid-rag", name="Ragavan, Nimble Pilferer", type_line="Creature"),
        Card(oracle_id="oid-sol", name="Sol Ring", type_line="Artifact"),
    ]
    v = CardVocab(cards)
    return v, CubeVocab(v, [c.oracle_id for c in cards])


def test_ratings_by_oracle_joins_by_name():
    vocab, _ = _vocab_and_cube()
    rows = [
        {"name": "Lightning Bolt", "game_count": 1000, "drawn_improvement_win_rate": 0.05},
        {"name": "Ragavan Nimble Pilferer", "game_count": 500, "drawn_improvement_win_rate": 0.08},
        {"name": "Some Other Card", "game_count": 10, "drawn_improvement_win_rate": 0.0},
        {"name": "Zero Games", "game_count": 0, "drawn_improvement_win_rate": 0.9},
    ]
    by_oracle = sl.ratings_by_oracle(rows, vocab)
    assert by_oracle["oid-bolt"]["drawn_improvement_win_rate"] == 0.05
    assert by_oracle["oid-rag"]["drawn_improvement_win_rate"] == 0.08  # comma-variant resolved
    assert "Zero Games" not in {v["name"] for v in by_oracle.values()}  # game_count==0 dropped


def test_value_array_zscored_and_orders_by_metric():
    vocab, cube = _vocab_and_cube()
    by_oracle = {
        "oid-bolt": {"drawn_improvement_win_rate": 0.05},
        "oid-rag": {"drawn_improvement_win_rate": 0.08},
        "oid-sol": {"drawn_improvement_win_rate": 0.02},
    }
    vals = sl.value_array(cube, by_oracle, "drawn_improvement_win_rate")
    assert vals.shape == (3,)
    assert abs(float(vals.mean())) < 1e-5  # z-scored
    # ragavan (0.08) > bolt (0.05) > sol (0.02)
    assert vals[1] > vals[0] > vals[2]


def test_value_array_avg_pick_is_inverted():
    vocab, cube = _vocab_and_cube()
    by_oracle = {  # lower avg_pick = taken earlier = should rank higher
        "oid-bolt": {"avg_pick": 2.0},
        "oid-rag": {"avg_pick": 1.0},
        "oid-sol": {"avg_pick": 9.0},
    }
    vals = sl.value_array(cube, by_oracle, "avg_pick")
    assert vals[1] > vals[0] > vals[2]  # rag (ATA 1) best


def test_value_array_missing_falls_to_floor():
    vocab, cube = _vocab_and_cube()
    by_oracle = {"oid-bolt": {"drawn_improvement_win_rate": 0.1}}  # others missing
    vals = sl.value_array(cube, by_oracle, "drawn_improvement_win_rate")
    assert vals[0] == vals.max()
    assert vals[1] == vals[2] == vals.min()  # missing -> floor


def test_bot_picks_highest_value_in_pack():
    values = np.array([0.0, 5.0, 1.0, -3.0, 2.0], dtype=np.float32)
    bot = SeventeenLandsBot(values, seed=0)
    pack = np.zeros(5, dtype=np.int8)
    pack[[0, 2, 3]] = 1  # cards 0,2,3 in pack; 2 has the highest value among them
    obs = {"pack": pack}
    assert bot.pick(obs) == 2


def test_bot_empty_pack_raises():
    import pytest

    bot = SeventeenLandsBot(np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="empty pack"):
        bot.pick({"pack": np.zeros(4, dtype=np.int8)})

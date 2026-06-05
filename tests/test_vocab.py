from __future__ import annotations


def test_vocab_size(global_vocab):
    assert len(global_vocab) == 540


def test_vocab_roundtrip(global_vocab):
    card = global_vocab.card(42)
    assert global_vocab.index(card.oracle_id) == 42
    assert global_vocab.index_by_name(card.name) == 42


def test_arena_to_oracle(global_vocab):
    card = global_vocab.card(7)
    assert global_vocab.arena_to_oracle(card.arena_id) == card.oracle_id
    assert global_vocab.arena_to_oracle(99999999) is None


def test_cube_filter_discards_unknown(global_vocab):
    cube_oracles = [global_vocab.card(i).oracle_id for i in range(10)]
    cube_oracles.append("nonexistent-oracle-id")
    from cube_draft.cards.vocab import CubeVocab
    cube = CubeVocab(global_vocab, cube_oracles)
    assert cube.size == 10
    assert cube.discarded == ["nonexistent-oracle-id"]


def test_cube_local_to_global(cube, global_vocab):
    # Synthetic cube uses the entire global vocab in order.
    assert cube.to_global(0) == 0
    assert cube.to_global(100) == 100
    assert cube.to_local(100) == 100
    assert cube.to_local(99999) is None

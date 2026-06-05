"""CubeCobraBot mapping/fallback tests using a fake transport (no Node needed)."""

from __future__ import annotations

import numpy as np

from cube_draft.bots.cubecobra import CubeCobraBot


class FakeTransport:
    """Stand-in for the Node bridge. Canned recognized flags + pick behavior."""

    def __init__(self, pick_fn=None, recognized=None):
        self.recognized = recognized
        self.pick_fn = pick_fn
        self.calls: list[dict] = []

    def __call__(self, msg: dict) -> dict:
        self.calls.append(msg)
        if msg["op"] == "testRecognized":
            n = len(msg["cardOracleIds"])
            return {"recognized": self.recognized if self.recognized is not None else [True] * n}
        if msg["op"] == "pick":
            return self.pick_fn(msg) if self.pick_fn else {"pick": msg["cardsInPack"][0]}
        return {"error": "bad op"}


def make_obs(n, pack_cards, pool_cards=(), seen_cards=(), pack_idx=0, pick_idx=0):
    pack = np.zeros(n, dtype=np.int8)
    pack[list(pack_cards)] = 1
    pool = np.zeros(n, dtype=np.int8)
    for c in pool_cards:
        pool[c] += 1
    seen = np.zeros(n, dtype=np.int8)
    for c in seen_cards:
        seen[c] += 1
    return {
        "pack": pack, "pool": pool, "seen_unpicked": seen,
        "cube_mask": np.ones(n, dtype=np.int8),
        "pack_idx": pack_idx, "pick_idx": pick_idx, "seat_idx": 0,
    }


def test_card_oracle_ids_mapping(cube):
    basics = ["basic-A", "basic-B"]
    bot = CubeCobraBot(cube, transport=FakeTransport(), basics=basics, check_recognized=False)
    assert bot.card_oracle_ids[: cube.size] == [cube.oracle_id(i) for i in range(cube.size)]
    assert bot.card_oracle_ids[cube.size :] == basics
    assert bot.basics_idx == [cube.size, cube.size + 1]


def test_pick_returns_bot_choice(cube):
    transport = FakeTransport(pick_fn=lambda m: {"pick": m["cardsInPack"][2]})
    bot = CubeCobraBot(cube, transport=transport, check_recognized=False)
    obs = make_obs(cube.size, pack_cards=[5, 10, 17, 23], pool_cards=[1], seen_cards=[2, 3])
    assert bot.pick(obs) == 17  # third in-pack card

    msg = transport.calls[-1]
    assert msg["op"] == "pick"
    assert msg["cardsInPack"] == [5, 10, 17, 23]
    assert msg["picked"] == [1]
    assert set(msg["seen"]) == {1, 2, 3}  # seen_unpicked ∪ pool


def test_out_of_pack_pick_falls_back_in_pack(cube):
    # Bot returns an index not in the pack -> bot must fall back to an in-pack card.
    transport = FakeTransport(pick_fn=lambda m: {"pick": 999})
    bot = CubeCobraBot(cube, transport=transport, check_recognized=False)
    pack = [4, 8, 12]
    assert bot.pick(make_obs(cube.size, pack)) in pack


def test_error_response_falls_back(cube):
    transport = FakeTransport(pick_fn=lambda m: {"error": "boom"})
    bot = CubeCobraBot(cube, transport=transport, check_recognized=False)
    pack = [4, 8, 12]
    assert bot.pick(make_obs(cube.size, pack)) in pack


def test_unrecognized_cards_filtered_from_options(cube):
    flags = [True] * cube.size
    flags[8] = False  # card 8 is OOV for the bot
    transport = FakeTransport(recognized=flags, pick_fn=lambda m: {"pick": m["cardsInPack"][0]})
    bot = CubeCobraBot(cube, transport=transport)
    bot.pick(make_obs(cube.size, pack_cards=[8, 12, 20]))
    assert 8 not in transport.calls[-1]["cardsInPack"]
    assert transport.calls[-1]["cardsInPack"] == [12, 20]


def test_all_unrecognized_falls_back_without_pick_call(cube):
    flags = [True] * cube.size
    for c in (4, 8, 12):
        flags[c] = False
    transport = FakeTransport(recognized=flags)
    bot = CubeCobraBot(cube, transport=transport)
    n_calls_before = len(transport.calls)
    result = bot.pick(make_obs(cube.size, pack_cards=[4, 8, 12]))
    assert result in (4, 8, 12)
    # only the testRecognized call happened during init; no pick op was sent
    assert all(c["op"] != "pick" for c in transport.calls[n_calls_before:])

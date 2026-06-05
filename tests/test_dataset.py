"""Round-trip test for the distillation dataset on-disk format (numpy-only)."""

from __future__ import annotations

import numpy as np

from cube_draft.data import dataset as ds


def _fake_cube_data(n_cards: int, n_decisions: int, seed: int) -> ds.CubeData:
    rng = np.random.default_rng(seed)
    return ds.CubeData(
        oracle_ids=[f"oracle-{i:05d}" for i in range(n_cards)],
        train={
            "pack": rng.integers(0, 2, (n_decisions, n_cards)).astype(np.int8),
            "pool": rng.integers(0, 2, (n_decisions, n_cards)).astype(np.int8),
            "seen": rng.integers(0, 2, (n_decisions, n_cards)).astype(np.int8),
            "pack_idx": rng.integers(0, 3, n_decisions).astype(np.int64),
            "pick_idx": rng.integers(0, 15, n_decisions).astype(np.int64),
            "picked": rng.integers(0, n_cards, n_decisions).astype(np.int64),
        },
        eval={
            "pack": rng.integers(0, 2, (4, n_cards)).astype(np.int8),
            "pool": rng.integers(0, 2, (4, n_cards)).astype(np.int8),
            "seen": rng.integers(0, 2, (4, n_cards)).astype(np.int8),
            "pack_idx": rng.integers(0, 3, 4).astype(np.int64),
            "pick_idx": rng.integers(0, 15, 4).astype(np.int64),
            "picked": rng.integers(0, n_cards, 4).astype(np.int64),
        },
    )


def test_save_load_roundtrip(tmp_path):
    cubes = [_fake_cube_data(20, 10, seed=0), _fake_cube_data(20, 7, seed=1)]
    ds.save_dataset(tmp_path / "d", "cubecobra", cubes, config={"foo": 1})

    loaded = ds.load_dataset(str(tmp_path / "d"))
    assert loaded.teacher == "cubecobra"
    assert len(loaded) == 2
    for orig, got in zip(cubes, loaded.cubes, strict=True):
        assert got.oracle_ids == orig.oracle_ids
        for key in ("pack", "pool", "seen", "pack_idx", "pick_idx", "picked"):
            np.testing.assert_array_equal(got.train[key], orig.train[key])
            np.testing.assert_array_equal(got.eval[key], orig.eval[key])

"""Tests for structured card features (numpy-only; no torch needed)."""

from __future__ import annotations

import numpy as np

from cube_draft.cards import features


def test_feature_dim_matches_vector(synthetic_cards):
    vec = features.card_feature_vector(synthetic_cards[0])
    assert vec.shape == (features.FEATURE_DIM,)
    assert vec.dtype == np.float32


def test_feature_matrix_shape(synthetic_cards):
    mat = features.feature_matrix(synthetic_cards)
    assert mat.shape == (len(synthetic_cards), features.FEATURE_DIM)


def test_color_matrix_matches_color_vector(synthetic_cards):
    mat = features.color_matrix(synthetic_cards)
    assert mat.shape == (len(synthetic_cards), 5)
    for i, card in enumerate(synthetic_cards):
        assert tuple(mat[i].astype(int)) == card.color_vector()


def test_empty_inputs():
    assert features.feature_matrix([]).shape == (0, features.FEATURE_DIM)
    assert features.color_matrix([]).shape == (0, 5)


def test_features_are_finite(synthetic_cards):
    mat = features.feature_matrix(synthetic_cards)
    assert np.isfinite(mat).all()

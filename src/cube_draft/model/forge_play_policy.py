"""Learned Forge play policy over externally supplied legal actions.

This module deliberately does not depend on Forge internals.  A Forge adapter is
expected to send a JSON observation containing a serializable game state and the
current legal actions.  The model hashes the observation/action JSON into fixed
feature vectors and scores every legal action.

Expected training / serving record shape::

    {
      "state": {...},
      "legal_actions": [{"id": "...", "type": "...", "text": "..."}, ...],
      "chosen_action_id": "...",   # training only
      "reward": 1.0                 # optional, terminal or return-to-go
    }
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _tokens(prefix: str, obj: Any, depth: int = 0, max_depth: int = 5) -> list[str]:
    """Tokenize nested JSON into path/value features.

    This is intentionally generic so it can consume early Forge adapter payloads
    before we settle on a compact hand-written game-state schema.
    """
    if depth > max_depth:
        return [f"{prefix}=<deep>"]
    if obj is None or isinstance(obj, bool | int | float | str):
        return [f"{prefix}={obj}"]
    if isinstance(obj, list):
        out = [f"{prefix}.len={len(obj)}"]
        for item in obj:
            # Lists in Forge states are usually zones/cards/permanents where the
            # values matter more than exact order.
            out.extend(_tokens(f"{prefix}[]", item, depth + 1, max_depth))
        return out
    if isinstance(obj, dict):
        out: list[str] = []
        for key, value in sorted(obj.items()):
            if key == "legal_actions":
                continue
            out.extend(_tokens(f"{prefix}.{key}", value, depth + 1, max_depth))
        return out
    return [f"{prefix}={obj}"]


def _hash_token(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    raw = int.from_bytes(digest, "little", signed=False)
    idx = raw % dim
    sign = 1.0 if ((raw >> 63) & 1) == 0 else -1.0
    return idx, sign


def hashed_features(obj: Any, dim: int, prefix: str) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _tokens(prefix, obj):
        idx, sign = _hash_token(tok, dim)
        vec[idx] += sign
    # Add a whole-object fingerprint so very similar raw text actions can still
    # be separable.
    idx, sign = _hash_token(f"{prefix}.json={_stable_json(obj)}", dim)
    vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


@dataclass(frozen=True)
class PlayExample:
    state: dict[str, Any]
    legal_actions: list[dict[str, Any]]
    chosen_index: int
    reward: float = 1.0


def parse_play_example(record: dict[str, Any]) -> PlayExample | None:
    actions = list(record.get("legal_actions") or [])
    if not actions:
        return None
    chosen_id = record.get("chosen_action_id", record.get("action_id"))
    chosen_idx = record.get("chosen_action_index", record.get("action_index"))
    if chosen_idx is None:
        for i, action in enumerate(actions):
            if str(action.get("id", i)) == str(chosen_id):
                chosen_idx = i
                break
    if chosen_idx is None:
        return None
    chosen = int(chosen_idx)
    if chosen < 0 or chosen >= len(actions):
        return None
    return PlayExample(
        state=dict(record.get("state") or {}),
        legal_actions=[dict(a) for a in actions],
        chosen_index=chosen,
        reward=float(record.get("reward", record.get("weight", 1.0))),
    )


class ForgePlayPolicy(nn.Module):
    """Scores legal actions conditioned on a hashed game-state vector."""

    def __init__(self, state_dim: int = 4096, action_dim: int = 2048, hidden: int = 512) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.action_net = nn.Sequential(nn.Linear(action_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.joint = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Return masked action logits.

        state: (B, S), actions: (B, A, D), action_mask: (B, A) bool/0-1.
        """
        s = self.state_net(state.float())
        a = self.action_net(actions.float())
        s_exp = s[:, None, :].expand_as(a)
        logits = self.joint(torch.cat([s_exp, a, s_exp * a], dim=-1)).squeeze(-1)
        return logits.masked_fill(~action_mask.bool(), float("-inf"))


def featurize_observation(record: dict[str, Any], state_dim: int, action_dim: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    state = record.get("state") or {}
    actions = [dict(a) for a in (record.get("legal_actions") or [])]
    state_vec = hashed_features(state, state_dim, "state")
    action_vecs = np.stack([hashed_features(a, action_dim, "action") for a in actions]).astype(np.float32)
    action_ids = [str(a.get("id", i)) for i, a in enumerate(actions)]
    return state_vec, action_vecs, action_ids


@torch.no_grad()
def choose_action(model: ForgePlayPolicy, record: dict[str, Any], device: torch.device | str = "cpu") -> dict[str, Any]:
    model.eval()
    state_vec, action_vecs, action_ids = featurize_observation(record, model.state_dim, model.action_dim)
    if len(action_ids) == 0:
        raise ValueError("observation has no legal_actions")
    state_t = torch.as_tensor(state_vec[None, :], dtype=torch.float32, device=device)
    acts_t = torch.as_tensor(action_vecs[None, :, :], dtype=torch.float32, device=device)
    mask_t = torch.ones((1, len(action_ids)), dtype=torch.bool, device=device)
    logits = model(state_t, acts_t, mask_t)[0]
    idx = int(torch.argmax(logits).item())
    return {"action_id": action_ids[idx], "action_index": idx, "score": float(logits[idx].item())}

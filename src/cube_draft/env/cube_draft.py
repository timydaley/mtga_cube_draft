"""CubeDraftEnv — single-agent cube draft as a Gymnasium environment.

Plan §3.1. Observation tensors are over the cube vocab (size ~360-540),
not the global card vocab. The agent occupies one seat; other seats are
provided as `Drafter` instances and step internally.

Pack rotation: 3 packs of 15 cards, 8 seats, alternating direction.
Pack 1 goes left, pack 2 right, pack 3 left.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cube_draft.cards.vocab import CubeVocab
from cube_draft.env.pack import Pack


@dataclass(frozen=True)
class DraftConfig:
    num_seats: int = 8
    num_packs: int = 3
    pack_size: int = 15
    agent_seat: int = 0
    seed: int | None = None

    def total_picks(self) -> int:
        return self.num_packs * self.pack_size


@dataclass
class SeatState:
    pool: list[int] = field(default_factory=list)
    seen_unpicked: list[int] = field(default_factory=list)

    def pool_counts(self, vocab_size: int) -> np.ndarray:
        counts = np.zeros(vocab_size, dtype=np.int8)
        for c in self.pool:
            counts[c] += 1
        return counts

    def seen_counts(self, vocab_size: int) -> np.ndarray:
        counts = np.zeros(vocab_size, dtype=np.int8)
        for c in self.seen_unpicked:
            counts[c] += 1
        return counts


class CubeDraftEnv(gym.Env):
    """8-seat cube draft. The agent's actions are picks; opponents step
    internally between agent decisions.

    Observation is a dict with:
        pack          : Int8[|cube|]  current pack at agent's seat
        pool          : Int8[|cube|]  agent's accumulated picks
        seen_unpicked : Int8[|cube|]  cards seen at agent's seat, not picked
        cube_mask     : Int8[|cube|]  all-ones; reserved for partial cube knowledge
        pack_idx      : int (0..num_packs-1)
        pick_idx      : int (0..pack_size-1)
        seat_idx      : int (constant, = agent_seat)

    Reward is 0 at every step. Terminal reward is filled in by an
    external scoring function (heuristic or learned head). See plan §3.1.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cube_vocab: CubeVocab,
        opponents: list,  # list[Drafter] but circular-import-free
        config: DraftConfig | None = None,
    ) -> None:
        super().__init__()
        self.cube = cube_vocab
        self.cfg = config or DraftConfig()
        if len(opponents) != self.cfg.num_seats - 1:
            raise ValueError(
                f"need {self.cfg.num_seats - 1} opponents, got {len(opponents)}"
            )
        self.opponents = list(opponents)
        self._rng = np.random.default_rng(self.cfg.seed)

        n = self.cube.size
        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Dict(
            {
                "pack": spaces.Box(low=0, high=1, shape=(n,), dtype=np.int8),
                "pool": spaces.Box(low=0, high=self.cfg.total_picks(), shape=(n,), dtype=np.int8),
                "seen_unpicked": spaces.Box(low=0, high=127, shape=(n,), dtype=np.int8),
                "cube_mask": spaces.Box(low=0, high=1, shape=(n,), dtype=np.int8),
                "pack_idx": spaces.Discrete(self.cfg.num_packs),
                "pick_idx": spaces.Discrete(self.cfg.pack_size),
                "seat_idx": spaces.Discrete(self.cfg.num_seats),
            }
        )

        self._seats: list[SeatState] = []
        self._packs_in_flight: list[Pack] = []
        self._pack_idx = 0
        self._pick_idx = 0

    # ---------------- gym API ----------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._seats = [SeatState() for _ in range(self.cfg.num_seats)]
        self._pack_idx = 0
        self._pick_idx = 0
        self._open_new_packs()
        return self._observe_agent(), {}

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        agent_pack = self._packs_in_flight[self.cfg.agent_seat]
        if action not in agent_pack:
            raise ValueError(
                f"action {action} not in agent's pack {agent_pack.cards()}"
            )

        # Record agent's seen_unpicked: every other card in the pack at
        # this pick (plan §2.4). Pool is updated separately.
        seen_for_agent = [c for c in agent_pack if c != action]
        self._seats[self.cfg.agent_seat].seen_unpicked.extend(seen_for_agent)
        self._seats[self.cfg.agent_seat].pool.append(action)
        agent_pack.pick(action)

        # Opponents pick from their own current packs.
        for seat_idx, drafter in self._opponent_iter():
            opp_pack = self._packs_in_flight[seat_idx]
            opp_obs = self._observe_seat(seat_idx)
            choice = drafter.pick(opp_obs)
            if choice not in opp_pack:
                raise ValueError(
                    f"opponent at seat {seat_idx} chose {choice} not in pack {opp_pack.cards()}"
                )
            self._seats[seat_idx].pool.append(choice)
            opp_pack.pick(choice)

        # Rotate packs in the current direction.
        self._rotate_packs()
        self._pick_idx += 1

        terminated = False
        if self._pick_idx >= self.cfg.pack_size:
            # End of pack — open next pack or terminate.
            self._pick_idx = 0
            self._pack_idx += 1
            if self._pack_idx >= self.cfg.num_packs:
                terminated = True
            else:
                self._open_new_packs()

        reward = 0.0  # terminal reward is filled in externally
        return self._observe_agent(), reward, terminated, False, {}

    # ---------------- helpers ----------------

    def _opponent_iter(self):
        """Yield (seat_idx, drafter) pairs in seat order, skipping the agent."""
        opp = iter(self.opponents)
        for seat in range(self.cfg.num_seats):
            if seat == self.cfg.agent_seat:
                continue
            yield seat, next(opp)

    def _open_new_packs(self) -> None:
        """Sample fresh packs for the new pack-round.

        Singleton: each pack is `pack_size` cards drawn without replacement
        from the cube. Packs across seats are independent samples — this
        matches Cube Cobra's bot drafter, where the cube is large enough
        that pack content overlap between seats is not a concern.
        """
        n_cards = self.cfg.pack_size * self.cfg.num_seats
        if n_cards > self.cube.size:
            raise ValueError(
                f"cube has {self.cube.size} cards; need {n_cards} for one pack-round"
            )
        sampled = self._rng.choice(self.cube.size, size=n_cards, replace=False)
        self._packs_in_flight = [
            Pack(sampled[i * self.cfg.pack_size : (i + 1) * self.cfg.pack_size].tolist())
            for i in range(self.cfg.num_seats)
        ]

    def _rotate_packs(self) -> None:
        """Pass each seat's current pack to the next seat in the round's direction."""
        direction = self._direction_for_pack(self._pack_idx)
        # direction +1 means seat i passes to seat (i+1) % N
        # direction -1 means seat i passes to seat (i-1) % N
        n = self.cfg.num_seats
        new_packs = [Pack([])] * n
        for i, pack in enumerate(self._packs_in_flight):
            new_idx = (i + direction) % n
            new_packs[new_idx] = pack
        self._packs_in_flight = new_packs

    def _direction_for_pack(self, pack_idx: int) -> int:
        """Pack 1 (idx 0) goes left (+1), pack 2 right (-1), pack 3 left (+1)."""
        return 1 if pack_idx % 2 == 0 else -1

    def _observe_seat(self, seat_idx: int) -> dict[str, Any]:
        n = self.cube.size
        pack = self._packs_in_flight[seat_idx]
        pack_vec = np.zeros(n, dtype=np.int8)
        for c in pack:
            pack_vec[c] = 1
        seat = self._seats[seat_idx]
        return {
            "pack": pack_vec,
            "pool": seat.pool_counts(n),
            "seen_unpicked": seat.seen_counts(n),
            "cube_mask": np.ones(n, dtype=np.int8),
            "pack_idx": self._pack_idx,
            "pick_idx": self._pick_idx,
            "seat_idx": seat_idx,
        }

    def _observe_agent(self) -> dict[str, Any]:
        return self._observe_seat(self.cfg.agent_seat)

    # ---------------- introspection ----------------

    def pool_of(self, seat_idx: int) -> list[int]:
        """Return the current pool of a seat (cube indices)."""
        return list(self._seats[seat_idx].pool)

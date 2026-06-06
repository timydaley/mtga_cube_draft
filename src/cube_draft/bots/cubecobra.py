"""CubeCobraBot — a teacher Drafter backed by the real CubeCobra/CubeArtisan bot.

The bot is the AGPL `mtgdraftbots` package (C++ -> WASM, run under Node). We
drive it as a subprocess via scripts/mtgdraftbots_bridge.mjs and keep only its
pick decisions (data) — never vendoring its code. Per plan §4.7.4 this is used
ONLY offline to build a distillation dataset, never in the training inner loop.

The transport (message dict -> response dict) is injectable so the mapping
logic is unit-testable with a fake; the default spawns the Node bridge.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from cube_draft.bots.base import Drafter, DraftObs
from cube_draft.cards.vocab import CubeVocab

logger = logging.getLogger(__name__)

Transport = Callable[[dict], dict]

# Repo-root-relative paths to the Node bridge and the fetch shim.
_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
_BRIDGE = _SCRIPTS / "mtgdraftbots_bridge.mjs"
_FETCH_SHIM = _SCRIPTS / "node_fetch_shim.cjs"


class _ClosableTransport(Protocol):
    def __call__(self, msg: dict) -> dict: ...
    def close(self) -> None: ...


class NodeBridgeTransport:
    """Persistent `node mtgdraftbots_bridge.mjs` subprocess, one request per call."""

    def __init__(self, bridge: Path = _BRIDGE, node: str = "node") -> None:
        # Preload the fetch shim into every Node context (incl. the WASM worker
        # thread) so the package can load its .wasm under modern Node.
        env = dict(os.environ)
        require = f"--require {_FETCH_SHIM}"
        env["NODE_OPTIONS"] = f"{require} {env.get('NODE_OPTIONS', '')}".strip()
        try:
            self._proc = subprocess.Popen(
                [node, str(bridge)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"could not launch '{node}'. The CubeCobraBot teacher needs Node + "
                "the mtgdraftbots package (cloud_bootstrap.sh installs them; locally "
                "`brew install node && npm install`)."
            ) from e
        self._await_ready()

    def _await_ready(self) -> None:
        for line in self._proc.stdout:  # type: ignore[union-attr]
            if json.loads(line).get("ready"):
                return
        raise RuntimeError("mtgdraftbots bridge exited before signalling ready")

    def __call__(self, msg: dict) -> dict:
        assert self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("mtgdraftbots bridge died (no response)")
        return json.loads(line)

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()


class CubeCobraBot(Drafter):
    NUM_PACKS = 3
    PACK_SIZE = 15

    def __init__(
        self,
        cube: CubeVocab,
        transport: Transport | None = None,
        basics: list[str] | None = None,
        seed: int = 37,
        check_recognized: bool = True,
    ) -> None:
        self.cube = cube
        self.transport: Transport = transport or NodeBridgeTransport()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # cardOracleIds: cube cards first (so index == cube-local index), then basics.
        self.basics = basics or []
        self.card_oracle_ids = [cube.oracle_id(i) for i in range(cube.size)] + self.basics
        self.basics_idx = list(range(cube.size, cube.size + len(self.basics)))

        self.recognized: set[int] = set(range(cube.size))
        if check_recognized:
            resp = self.transport({"op": "testRecognized", "cardOracleIds": self.card_oracle_ids})
            flags = resp.get("recognized", [])
            self.recognized = {i for i in range(cube.size) if i < len(flags) and flags[i]}
            unknown = cube.size - len(self.recognized)
            if unknown:
                logger.warning("CubeCobraBot: %d/%d cube cards unrecognized by the bot", unknown, cube.size)

    def pick(self, obs: DraftObs) -> int:
        in_pack = np.flatnonzero(obs["pack"] > 0).tolist()
        if not in_pack:
            raise ValueError("empty pack passed to CubeCobraBot")

        candidates = [i for i in in_pack if i in self.recognized]
        if not candidates:
            # Whole pack is OOV for the bot — fall back so the draft can proceed.
            return int(self._rng.choice(in_pack))

        msg = {
            "op": "pick",
            "cardOracleIds": self.card_oracle_ids,
            "cardsInPack": candidates,
            "picked": np.flatnonzero(obs["pool"] > 0).tolist(),
            "seen": np.flatnonzero((obs["seen_unpicked"] > 0) | (obs["pool"] > 0)).tolist(),
            "basics": self.basics_idx,
            "packNum": int(obs["pack_idx"]),
            "numPacks": self.NUM_PACKS,
            "pickNum": int(obs["pick_idx"]),
            "numPicks": self.PACK_SIZE,
            "seed": self.seed,
        }
        resp = self.transport(msg)
        pick = resp.get("pick")
        if "error" in resp or pick is None or pick not in in_pack:
            logger.warning("CubeCobraBot fallback (resp=%s)", resp)
            return int(self._rng.choice(candidates))
        return int(pick)

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()

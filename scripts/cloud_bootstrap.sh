#!/usr/bin/env bash
# Bootstrap a fresh RunPod pod for training. Run once per pod.
#
# Usage:
#   curl -L https://raw.githubusercontent.com/<USER>/<REPO>/main/scripts/cloud_bootstrap.sh | bash
#   # or, if repo is already cloned:
#   bash scripts/cloud_bootstrap.sh
#
# Required pod environment variables (set in RunPod pod template "Environment Variables"):
#   GIT_REPO_URL              — e.g. https://github.com/<user>/mtga_cube_draft.git
#   R2_ACCOUNT_ID
#   R2_ACCESS_KEY_ID
#   R2_SECRET_ACCESS_KEY
#   R2_BUCKET                 (optional, defaults to cube-draft)
#   WANDB_API_KEY
#   WANDB_PROJECT             (optional, defaults to cube-draft)
#
# Optional tuning env vars:
#   TORCH_CUDA              PyTorch CUDA wheel tag (default cu128). Must be <= the
#                           CUDA version the pod's NVIDIA driver supports (see
#                           `nvidia-smi`). Set to "cpu" for a CPU-only torch.
#   INSTALL_MTGDRAFTBOTS    Set to 1 to install Node + the mtgdraftbots package
#                           for the CubeCobraBot distillation teacher. Off by
#                           default — only dataset-generation pods need it.
#
# This script is idempotent — safe to re-run after a spot interruption.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/mtga_cube_draft}"
PY_VER="${PY_VER:-3.12}"

echo "==> bootstrap starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1. System deps (idempotent on RunPod's PyTorch base image)
if ! command -v git >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq git curl build-essential
fi

# 2. uv (the Python package manager we use everywhere)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 3. Clone or refresh the repo
mkdir -p "$(dirname "$REPO_DIR")"
if [ ! -d "$REPO_DIR/.git" ]; then
  : "${GIT_REPO_URL:?GIT_REPO_URL must be set in pod env}"
  git clone "$GIT_REPO_URL" "$REPO_DIR"
else
  echo "==> repo already present, fetching latest"
  git -C "$REPO_DIR" fetch --quiet
  git -C "$REPO_DIR" pull --ff-only --quiet
fi
cd "$REPO_DIR"

# 4. Python env + ML deps
uv venv --python "$PY_VER" --quiet

# Install PyTorch built for the pod's CUDA driver BEFORE the rest of the deps.
# The default PyPI torch wheel targets the newest CUDA (currently cu130), which
# fails to initialize on an older driver and silently falls back to CPU. Pin the
# wheel to a CUDA build the driver supports (cu128 matches a CUDA 12.8 driver).
TORCH_CUDA="${TORCH_CUDA:-cu128}"
echo "==> installing torch ($TORCH_CUDA build)"
uv pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" --quiet

# Remaining deps. torch already satisfies torch>=2.4, so it is not replaced.
uv pip install -e ".[ml,dev]" --quiet

# 4b. Fail fast if torch can't see the GPU, rather than silently training on CPU.
if [ "$TORCH_CUDA" != "cpu" ]; then
  echo "==> verifying CUDA"
  .venv/bin/python -c "
import torch
print(f'torch {torch.__version__} | built for CUDA {torch.version.cuda}')
if not torch.cuda.is_available():
    raise SystemExit(
        'CUDA not available to torch. The installed wheel likely targets a newer '
        'CUDA than the driver supports — set TORCH_CUDA to a build <= the driver\'s '
        'CUDA version (see nvidia-smi) and re-run.'
    )
print(f'CUDA OK: {torch.cuda.get_device_name(0)}')
"
fi

# 4c. (optional) Node + mtgdraftbots for the CubeCobraBot distillation teacher.
# Only dataset-generation pods need this; off by default to keep training lean.
if [ "${INSTALL_MTGDRAFTBOTS:-0}" = "1" ]; then
  echo "==> installing Node + mtgdraftbots (CubeCobraBot teacher)"
  if ! command -v node >/dev/null 2>&1; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - >/dev/null 2>&1
    apt-get install -y -qq nodejs
  fi
  npm install --silent
  # Exercise the WASM worker through the fetch shim (NODE_OPTIONS reaches worker
  # threads). A bare `import()` resolves before the worker async-loads its .wasm,
  # so it must actually init + round-trip, or the check is meaningless.
  NODE_OPTIONS="--require $REPO_DIR/scripts/node_fetch_shim.cjs" node -e "
import('mtgdraftbots').then(async (m) => {
  await m.initializeDraftbots();
  await m.testRecognized(['9056eba4-612b-4e82-8689-fe098241b007']);
  await m.terminateDraftbots();
  console.log('mtgdraftbots OK');
}).catch((e) => { console.error(e); process.exit(1); });
"
fi

# 5. W&B login (no-op if WANDB_API_KEY already set in env)
if [ -n "${WANDB_API_KEY:-}" ]; then
  .venv/bin/wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi

# 6. Sanity-check R2 access
echo "==> R2 sanity check"
.venv/bin/python -c "
from cube_draft.utils import r2
client = r2.make_client()
objs = r2.list_prefix('')
print(f'R2 OK: {len(objs)} objects visible in r2://{r2.bucket_name()}/')
"

# 7. Pull working data (Scryfall is the only universal dep)
mkdir -p data
if .venv/bin/python -c "from cube_draft.utils import r2; assert any(o['Key']=='scryfall/oracle_cards.parquet' for o in r2.list_prefix('scryfall/'))" 2>/dev/null; then
  .venv/bin/python scripts/r2_sync.py pull scryfall/oracle_cards.parquet data/cards.parquet
else
  echo "==> Scryfall data not in R2 yet — fetching from source"
  .venv/bin/python scripts/download_scryfall.py
  echo "==> pushing to R2 for future pod sessions"
  .venv/bin/python scripts/r2_sync.py push data/cards.parquet scryfall/oracle_cards.parquet
fi

# 8. Run the smoke test to validate end-to-end cloud loop
echo "==> running smoke test"
.venv/bin/python scripts/train_smoketest.py --wandb-mode online --steps 5

echo "==> bootstrap complete. SSH back in and run your real training script."
echo "    repo:   $REPO_DIR"
echo "    venv:   $REPO_DIR/.venv/bin/python"

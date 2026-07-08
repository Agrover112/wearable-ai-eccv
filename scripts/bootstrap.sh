#!/usr/bin/env bash
# Rebuild the environment on ANY machine (Colab or local).
# Usage:  bash scripts/bootstrap.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- Detect Colab ---------------------------------------------------------
IN_COLAB=0
python -c "import google.colab" 2>/dev/null && IN_COLAB=1 || true

# --- Persistent cache location -------------------------------------------
# On Colab we point HF_HOME at Google Drive so datasets + model weights
# survive runtime restarts. Locally we keep a repo-local cache.
if [ "$IN_COLAB" = "1" ]; then
  echo ">> Colab detected. Mounting Google Drive..."
  python - <<'PY'
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs('/content/drive/MyDrive/wearable-ai-cache', exist_ok=True)
PY
  export HF_HOME=/content/drive/MyDrive/wearable-ai-cache/hf
else
  export HF_HOME="$REPO_ROOT/.cache/hf"
fi
mkdir -p "$HF_HOME"
echo ">> HF_HOME=$HF_HOME"
echo "export HF_HOME=$HF_HOME" > "$REPO_ROOT/.hf_env"   # source this in later cells

# --- Install deps ---------------------------------------------------------
echo ">> Installing requirements..."
pip install -q -r requirements.txt

# --- Resolve tokens (platform-agnostic) -----------------------------------
# load_env.py resolves HF_TOKEN/GH_TOKEN/WANDB_API_KEY from: env var -> .env ->
# Drive .env -> Colab Secrets, then writes .env (+ mirrors to Drive). Works on
# Colab AND local — no code path is Colab-only.
python scripts/load_env.py
set -a; [ -f .env ] && . ./.env; set +a

# --- HF auth (gated facebook/wearable-ai) ---------------------------------
if [ -n "${HF_TOKEN:-}" ]; then
  python -c "from huggingface_hub import login; login('${HF_TOKEN}')"
  echo ">> Logged into Hugging Face."
else
  echo "!! HF_TOKEN not found (Colab Secrets -> HF_TOKEN). Needed for the gated dataset."
fi

# --- GitHub auth: configure once so push never prompts --------------------
if [ -n "${GH_TOKEN:-}" ]; then
  git config --global credential.helper store
  printf 'https://x-access-token:%s@github.com\n' "$GH_TOKEN" > ~/.git-credentials
  chmod 600 ~/.git-credentials
  git config --global user.name  "Agrover112"
  git config --global user.email "agrover112@gmail.com"
  echo ">> GitHub credentials configured (push will not prompt)."
else
  echo "!! GH_TOKEN not found (Colab Secrets -> GH_TOKEN). Needed to push to GitHub."
fi

echo ">> Bootstrap complete. Next:  source .hf_env && python scripts/download_data.py"

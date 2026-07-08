#!/usr/bin/env python
"""Platform-agnostic token resolver.

Resolution order per key (HF_TOKEN, GH_TOKEN, WANDB_API_KEY):
  1. Already-set environment variable  (local machine / CI / other Claude env)
  2. .env file in repo root
  3. Colab Secrets (only when running inside a Colab notebook)

Side effects:
  - Writes/updates ./.env  (gitignored)
  - Mirrors ./.env to Google Drive when mounted, so it persists + is reused everywhere.

Usage:
  python scripts/load_env.py           # resolves + writes .env, prints status
  # then in shell:  set -a; source .env; set +a
"""
import os
from pathlib import Path

KEYS = ("HF_TOKEN", "GH_TOKEN", "WANDB_API_KEY")
ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
DRIVE_ENV = Path("/content/drive/MyDrive/wearable-ai-cache/.env")


def _read_env_file(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def _from_colab(key: str):
    try:
        from google.colab import userdata
        return userdata.get(key)
    except Exception:
        return None


def resolve() -> dict:
    file_vals = _read_env_file(ENV)
    if not any(file_vals.values()) and DRIVE_ENV.exists():
        file_vals = _read_env_file(DRIVE_ENV)  # fall back to Drive copy

    resolved, source = {}, {}
    for k in KEYS:
        if os.environ.get(k):
            resolved[k], source[k] = os.environ[k], "env"
        elif file_vals.get(k):
            resolved[k], source[k] = file_vals[k], ".env"
        else:
            v = _from_colab(k)
            if v:
                resolved[k], source[k] = v, "colab-secret"
            else:
                resolved[k], source[k] = "", "MISSING"
    return resolved, source


def write_env(vals: dict):
    lines = [f"{k}={vals.get(k,'')}" for k in KEYS]
    ENV.write_text("\n".join(lines) + "\n")
    ENV.chmod(0o600)
    # Mirror to Drive if mounted
    if DRIVE_ENV.parent.parent.exists():  # /content/drive/MyDrive exists
        DRIVE_ENV.parent.mkdir(parents=True, exist_ok=True)
        DRIVE_ENV.write_text(ENV.read_text())


def _main():
    # Keep token values LOCAL to this function so they never end up in module
    # globals (which runpy/Colab would print). Only masked status is shown.
    vals, src = resolve()
    write_env(vals)
    print("Token resolution:")
    for k in KEYS:
        state = "MISSING" if src[k] == "MISSING" else f"OK (from {src[k]})"
        print(f"  {k:14} {state}")
    print(f"\nWrote {ENV}")
    if DRIVE_ENV.exists():
        print(f"Mirrored to {DRIVE_ENV}")


if __name__ == "__main__":
    _main()

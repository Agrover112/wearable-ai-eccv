# 🔁 Restart & Continue (Colab)

How to reopen this project in Colab and pick up **exactly** where you left off —
after either kind of runtime restart.

## What survives a restart (nothing important is lost)

| Thing | Persists via |
|---|---|
| Code, scripts, **embeddings** (`features/*.npy`), outputs | **GitHub** |
| HF dataset + model weights (SigLIP 4.3 GB, etc.) | **`HF_HOME` on Google Drive** |
| Tokens (`HF_TOKEN`, `GH_TOKEN`) | **Colab 🔑 Secrets** (set once) |

The two Colab restart types are both handled:

| Colab action | Effect | `/content` files |
|---|---|---|
| **Runtime → Restart runtime** | clears Python state | ✅ kept |
| **Runtime → Disconnect and delete runtime** | fresh VM | ❌ wiped (re-cloned) |

---

## One-time prerequisite

In Colab, open the **🔑 Secrets** panel and add, each with *notebook access ON*:

- `HF_TOKEN` — a Hugging Face token with access to the gated `facebook/wearable-ai` dataset
- `GH_TOKEN` — a GitHub token that can read/push this **private** repo

Set once; every future restart just works.

---

## Opening the notebook — 2 ways

The repo is **private**, so a plain "Open from GitHub" needs authorization. Pick one:

### Way 1 — Open from GitHub (recommended for daily use)

1. Go to <https://colab.research.google.com>
2. **File → Open notebook → GitHub** tab
3. Sign in and **tick "Include private repositories"** → authorize Colab's GitHub app (one-time)
4. Search `Agrover112/wearable-ai-eccv` → open `notebooks/00_restart.ipynb`
5. Run the **Setup** cell.

Direct link (works once Colab↔GitHub is authorized):
<https://colab.research.google.com/github/Agrover112/wearable-ai-eccv/blob/main/notebooks/00_restart.ipynb>

### Way 2 — Zero-setup fallback (paste into ANY blank Colab)

No GitHub↔Colab auth needed. Open a **blank** Colab notebook and paste this one cell:

```python
import os, subprocess
from google.colab import userdata
GH = userdata.get("GH_TOKEN")                 # from Colab Secrets
url = f"https://x-access-token:{GH}@github.com/Agrover112/wearable-ai-eccv.git"
if not os.path.isdir("/content/wearable-ai-eccv"):
    subprocess.run(["git", "clone", url, "/content/wearable-ai-eccv"], check=True)
os.chdir("/content/wearable-ai-eccv")
subprocess.run(["bash", "scripts/bootstrap.sh"], check=True)   # Drive, HF_HOME, deps, auth
```

Then run any script, e.g. `!python scripts/eda_clustering.py`.

| | Way 1 (GitHub tab) | Way 2 (paste) |
|---|---|---|
| GitHub↔Colab auth needed | Yes (one-time) | **No** |
| Best for | daily use, edits saved back to GitHub | first time / quick continue |

---

## Why it runs *exactly* the same

- `scripts/bootstrap.sh` points **`HF_HOME` → Drive**, so the cached SigLIP weights and
  dataset are already present — **no re-download**.
- **Embeddings are committed to git** (`features/*.npy`), so clustering skips encoding and
  loads the caches directly.
- `scripts/load_env.py` resolves tokens from **Colab Secrets**, so there is no re-auth and
  no token typing.
- The notebook loads `.env` / `.hf_env` into `os.environ` — a bash `source` would **not**
  persist env vars to later Python cells; this does.

---

## TL;DR

1. First time only: add `HF_TOKEN` + `GH_TOKEN` in Colab **🔑 Secrets**.
2. Open `notebooks/00_restart.ipynb` (Way 1) — or paste the Way 2 cell into a blank notebook.
3. Run the **Setup** cell → continue with `!python scripts/…`.

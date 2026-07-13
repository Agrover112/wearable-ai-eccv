# HieraMamba Setup on Colab T4

These wheels were built for this exact stack:

- Python 3.12
- PyTorch 2.11.0 + CUDA 12.8
- NVIDIA T4 (`sm_75`)

Do not use them with a different Python, PyTorch, CUDA, or GPU architecture.

## Fast install on a matching runtime

```bash
cd /content/wearable-ai-eccv
python -c "import torch; print(torch.__version__, torch.version.cuda)"
pip install src/mamba/wheels/causal_conv1d-*.whl \
            src/mamba/wheels/mamba_ssm-*.whl

git clone --recursive https://github.com/jbistanbul/hieramamba.git /content/hieramamba
cd /content/hieramamba/libs/nms
python setup_nms.py build_ext --inplace
```

Download and unpack the official Ego4D checkpoint:

```bash
wget -c https://utexas.box.com/shared/static/69bfc16fhjz6md4fy2umajnxntfg33t2.zip \
  -O /content/hieramamba_ego4d_ckpt.zip
unzip -q /content/hieramamba_ego4d_ckpt.zip \
  -d /content/hieramamba/experiments
```

Verify imports:

```bash
cd /content/hieramamba
PYTHONPATH="$PWD:$PWD/libs/nms" python -c \
  "import torch, causal_conv1d, mamba_ssm, nms_1d_cpu_vg; print('imports OK', torch.cuda.get_device_name(0))"
```

If `nvidia-smi` cannot see the T4, do not start feature extraction. Confirm
that Colab is using a GPU runtime and reconnect first.

## Install EgoVLP

HieraMamba does not extract features from RGB video. Clone the official EgoVLP
repository and install the few missing packages used by its older code:

```bash
git clone --recursive https://github.com/showlab/EgoVLP.git /content/EgoVLP
pip install ffmpeg sacred tensorboardx dominate decord
mkdir -p /content/EgoVLP/pretrained
```

Download both official EgoVLP files into local `/content` storage:

```bash
gdown 1-cP3Gcg0NGDcMZalgJ_615BQdbFIbcj7 \
  -O /content/EgoVLP/pretrained/egovlp.pth
wget -c https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth \
  -O /content/EgoVLP/pretrained/jx_vit_base_p16_224-80ecf9dd.pth
```

The official 2022 EgoVLP checkpoint contains a serialized config object.
PyTorch 2.6+ therefore requires `torch.load(..., weights_only=False)` for this
trusted official file. `scripts/extract_egovlp_dev.py` applies that compatibility
setting only while loading the checkpoint.

## Required inputs

The Ego4D checkpoint does not consume RGB video directly. For each EgoLongQA
sample, prepare:

- 256-dimensional EgoVLP clip features;
- 768-dimensional token-level EgoVLP text features;
- a SnAG-style annotation record mapping feature indices to video time.

Then run HieraMamba inference, save its five post-NMS spans and scores, and
convert each span from feature indices to seconds before sampling frames for
Qwen.

Do not feed raw interrogative questions directly to HieraMamba. Convert each
question into declarative, visually observable events using
`HIERAMAMBA_QUERY_CONVERSION_PROMPT.md`. Never include MCQ options or answers in
the grounding query.

## EgoLongQA extraction

Use local `/content` paths unless persistent Drive storage was explicitly
requested:

```bash
cd /content/wearable-ai-eccv
HF_HOME=/content/hf-cache python -u scripts/extract_egovlp_dev.py \
  --manifest configs/egolongqa_dev20_seed20260709.json \
  --video-dir /content/dev20_videos \
  --output-dir /content/egovlp_dev20 \
  --egovlp-root /content/EgoVLP \
  --batch-size 32
```

The exact EgoVLP timeline uses a `32/30`-second window and an `8/30`-second
stride. A ten-minute video therefore produces roughly 2,250 overlapping clip
features. The extractor deduplicates overlapping frame indices within each
batch and caches one output per video.

`--target-clips N` creates a coarse, uniformly spaced timeline for pipeline
smoke tests only. Do not use that shortcut to judge localization quality or
report accuracy.

Prepare a one-video HieraMamba smoke-test experiment:

```bash
python scripts/prepare_hieramamba_smoke.py \
  --features /content/egovlp_smoke64 \
  --video-id VIDEO_ID \
  --hieramamba-root /content/hieramamba

cd /content/hieramamba
PYTHONPATH="$PWD:$PWD/libs/nms" python -u eval.py \
  --name egolongqa_smoke --ckpt last
```

The inherited evaluator requires a dummy timestamp target. Ignore its printed
IoU for EgoLongQA, which has no gold temporal annotations. Read the actual
top-five spans from:

```text
/content/hieramamba/experiments/egolongqa_smoke/predictions_last.json
```

Judge localization with span previews or a manually timestamped audit subset.
Only after localization is plausible should the selected frames be sent to
Qwen for MCQ answering.

## If the Colab stack changes

Rebuild both packages from source with build isolation disabled. Patch their
`setup.py` CUDA architecture lists to retain only the target GPU:

- T4: `compute_75,code=sm_75`
- A100: `compute_80,code=sm_80`
- H100: `compute_90,code=sm_90`

Build against the active Colab PyTorch installation:

```bash
pip install causal-conv1d==1.6.2.post1 mamba-ssm==2.3.1 \
  --no-build-isolation
```

Save the resulting wheels in this directory and record the exact Python,
PyTorch, CUDA, and GPU architecture above. Never commit access tokens.

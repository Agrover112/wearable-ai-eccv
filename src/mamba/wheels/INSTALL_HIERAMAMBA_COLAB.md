# HieraMamba on Colab T4

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

## Required inputs

The Ego4D checkpoint does not consume RGB video directly. For each EgoLongQA
sample, prepare:

- 256-dimensional EgoVLP clip features;
- 768-dimensional token-level EgoVLP text features;
- a SnAG-style annotation record mapping feature indices to video time.

Then run HieraMamba inference, save its five post-NMS spans and scores, and
convert each span from feature indices to seconds before sampling frames for
Qwen.

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

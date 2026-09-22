# Longa

Frame selection for long egocentric video question answering. A vision-language
model answers a multiple-choice question from at most 64 frames of a long video;
each script here chooses those frames differently. Built for the EgoLongQA track
of the [Wearable AI Workshop at ECCV 2026](https://wearable-ai-workshop.github.io/)
(`egolongqa` split of [`facebook/wearable-ai`](https://huggingface.co/datasets/facebook/wearable-ai)).

<img src="assets/dual_view_fusion.svg" width="100%" alt="Dual-view fusion of uniform and option-conditioned frames">

## Results

EgoLongQA validation accuracy.

| Method | Model | Questions | Accuracy | Uniform 64, same setup |
| --- | --- | --- | --- | --- |
| [Uniform 64](infer_uniform.py) | Qwen3.5-27B | 700 | 86.9% (608) | |
| [Dual-view fusion](infer_fusion.py) | Qwen3.5-27B | 700 | **88.3% (618)** | 86.9% (608) |
| [Option-conditioned temporal](infer_temporal.py) | Qwen3.5-27B | 700 | 84.4% (591) | 86.9% (608) |
| [Query-conditioned allocation](infer_qca.py) | Qwen3-VL-8B | dev140 | 74.3% (104) | 78.6% (110)\* |
| [Uncertainty-guided](infer_uncertainty.py) | Qwen3-VL-8B | dev20 | 80.0% (16) | 85.0% (17)\* |
| [Temporal chain of thought](infer_tcot.py) | Qwen3-VL-8B | dev140 | 67.9% (95) | 78.6% (110)\* |

The temporal view alone trails uniform sampling but answers 35 questions that
uniform misses, which fusion exploits. Numbers come from the experiment code
this repository was ported from.

\* These Qwen3-VL-8B baselines sampled 64 frames without the final frame.

## Setup

Python 3.10 and CUDA 12.8:

```bash
conda create -n longa python=3.10
conda activate longa
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Questions are JSONL with a video filename, the question, and the options:

```json
{"video_path": "clip.mp4", "question": "What did I pick up?", "mcq_options": "A. A cup\nB. A pen\nC. A book\nD. A plate"}
```

## Running

Start the VLM server on one 80 GB GPU, then run a script in a second shell
with the same environment.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/serve.sh
python infer_fusion.py --questions val.jsonl --videos videos/ --output outputs/fusion.jsonl
```

Each script writes `video_path`, `question`, and `mcq_answer` per line. Scripts
that use SigLIP2 take `--siglip-device` (default `cuda`).

To reproduce the Qwen3-VL-8B rows, serve that model and select the subset:

```bash
bash scripts/serve.sh Qwen/Qwen3-VL-8B-Instruct
grep -Ff splits/dev140.txt val.jsonl > dev140.jsonl   # splits/dev20.txt for uncertainty
python infer_qca.py --model Qwen/Qwen3-VL-8B-Instruct --questions dev140.jsonl --videos videos/ --output outputs/qca.jsonl
```

## Code layout

```
infer_*.py   one method per script, run directly
models/      vlm.py (prompt, vLLM requests), siglip.py (retrieval embeddings)
utils/       data.py (questions, frames), frame_selection.py (shared frame assembly)
scripts/     serve.sh (vLLM server)
splits/      video lists of the dev140 and dev20 subsets
assets/      README figure
```

## License

Adapted from the Wearable AI starter kit; CC BY-NC 4.0, see [LICENSE](LICENSE).
Model weights are under their own licenses. The figure shows frames and a
question from EgoLongQA validation video `e4947cad50181471` of
[`facebook/wearable-ai`](https://huggingface.co/datasets/facebook/wearable-ai)
(© 2026 Meta Platforms, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/));
frames are resized and the question is shortened.

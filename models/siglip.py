"""SigLIP2 image and text embeddings for retrieval (temporal selection, fusion, QCA)"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor, PreTrainedModel, ProcessorMixin


def load_siglip(model_id: str, device: str) -> tuple[PreTrainedModel, ProcessorMixin]:
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    return model, processor


@torch.inference_mode()
def image_features(
    model: PreTrainedModel, processor: ProcessorMixin, frames: list[Image.Image],
) -> np.ndarray:
    """Unit-length embedding of every frame, [N, D] float32"""
    batches = []
    for start in range(0, len(frames), 8):
        inputs = processor(images=frames[start:start + 8], return_tensors="pt").to(model.device)
        pooled = model.get_image_features(**inputs).pooler_output
        # [B, D] BF16, L2-normalized before the float32 cast
        batches.append(F.normalize(pooled, dim=-1).float().cpu().numpy())
    return np.concatenate(batches)  # [N, D] float32


@torch.inference_mode()
def relevance_scores(
    model: PreTrainedModel, processor: ProcessorMixin, features: np.ndarray, query: str,
) -> list[float]:
    """Cosine similarity of every frame to one text query"""
    # One query per call pads nothing. SigLIP2 was trained with 64-token max-length
    # padding; the reported results were produced without it, so keep it that way
    inputs = processor(text=[query], padding=True, truncation=True, return_tensors="pt").to(model.device)
    pooled = model.get_text_features(**inputs).pooler_output
    text_features = F.normalize(pooled, dim=-1).float().cpu().numpy()
    return (features @ text_features[0]).tolist()

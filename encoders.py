import torch
from transformers import AutoTokenizer
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

# text embedding (RoBERTa-large)
tokenizer = AutoTokenizer.from_pretrained(
    "roberta-large", use_fast=True, add_prefix_space=True
)

def encode_text(texts):
    """
    texts: list[str]
    returns: dict with 'input_ids' etc. on *CPU* (leave transfer to caller)
    """
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

# image embedding (OpenCLIP ViT-L/14)
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="laion2b_s32b_b82k"
)
clip_model = clip_model.to(device).eval()

@torch.inference_mode()
def encode_images(img_batch):
    """
    img_batch: torch.Tensor of shape (B,3,H,W) already on *device*
    returns: L2‑normalised CLIP embeddings (B, D)
    """
    with torch.autocast(device):
        feats = clip_model.encode_image(img_batch)
    return feats / feats.norm(dim=-1, keepdim=True)
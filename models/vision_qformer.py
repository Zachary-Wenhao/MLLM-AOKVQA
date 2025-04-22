import torch
from transformers import Blip2Processor, Blip2Model

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisionQFormer:
    """
    Encapsulates BLIP2 vision backbone + Q-Former + projection, with attention extraction.
    """
    def __init__(self, model_name: str = "Salesforce/blip2-flan-t5-xl"):
        # Initialize processor and full BLIP2 model
        self.processor = Blip2Processor.from_pretrained(model_name, use_safetensors=True)
        full_model = Blip2Model.from_pretrained(model_name)
        # Extract sub-modules
        self.vision_model = full_model.vision_model.to(device)
        self.qformer = full_model.qformer.to(device)
        self.visual_projection = full_model.visual_projection.to(device)

    def encode_with_attention(self, image):
        """
        Returns:
          feats: Tensor of shape [1, Q, D] (projected Q-Former features)
          attns: Tuple of cross-attention tensors (each [1, heads, Q, N])
        """
        # Preprocess image
        inputs = self.processor(images=[image], return_tensors="pt").to(device)
        # Vision encoding
        vision_out = self.vision_model(**inputs)
        # Q-Former cross attention
        q_out = self.qformer(
            inputs_embeds=None,
            encoder_hidden_states=vision_out.last_hidden_state,
            encoder_attention_mask=inputs.get("pixel_mask"),
            output_attentions=True
        )
        # Project to LLM embedding space
        feats = self.visual_projection(q_out.last_hidden_state)
        return feats, q_out.cross_attentions
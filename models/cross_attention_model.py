import torch
import torch.nn as nn
from transformers import Blip2Processor, Blip2Model, AutoTokenizer, BertConfig, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrossAttentionExtractor(nn.Module):
    def __init__(
        self,
        vision_model_name: str = "Salesforce/blip2-flan-t5-xl",
        text_model_name: str   = "bert-base-uncased"
    ):
        super().__init__()
        # 1) Vision Q-Former setup
        self.processor = Blip2Processor.from_pretrained(vision_model_name)
        full_blip2        = Blip2Model.from_pretrained(vision_model_name)
        # remove leading batch dim if present, store query tokens [query_len, dim]
        self.query_tokens = full_blip2.query_tokens.squeeze(0).to(device)
        self.vision_model = full_blip2.vision_model.to(device)
        self.qformer      = full_blip2.qformer.to(device)
        self.proj         = full_blip2.visual_projection.to(device) if hasattr(full_blip2, 'visual_projection') else nn.Identity()

        # determine visual feature dimension for cross-attention
        vision_dim = self.proj.out_features if isinstance(self.proj, nn.Linear) else self.vision_model.config.hidden_size
        cfg = BertConfig.from_pretrained(
            text_model_name,
            add_cross_attention=True,
            encoder_width=vision_dim,
            is_decoder=True
        )
        self.text_encoder = BertModel.from_pretrained(text_model_name, config=cfg).to(device)
        self.tokenizer    = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)

    def forward(self, images, questions, rationales):
        # --- Vision Q-Former encoding ---
        inputs = self.processor(images=images, return_tensors="pt").to(device)
        vision_out = self.vision_model(**inputs)
        batch_size = inputs["pixel_values"].shape[0]
        # expand query tokens to match batch size: [batch_size, query_len, dim]
        query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        q_out = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=vision_out.last_hidden_state,
            encoder_attention_mask=inputs.get("pixel_mask"),
            output_attentions=True
        )
        feats = self.proj(q_out.last_hidden_state)

        # --- Text cross-attention ---
        batch_texts = [f"Question: {q}\nRationale: {r}" for q, r in zip(questions, rationales)]
        txt_inputs  = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        text_out = self.text_encoder(
            input_ids=txt_inputs.input_ids,
            attention_mask=txt_inputs.attention_mask,
            encoder_hidden_states=feats,
            encoder_attention_mask=torch.ones(feats.size()[:-1], device=device),
            output_attentions=True,
        )
        # text_out.cross_attentions is a tuple of [batch, heads, seq_len, Q]
        return text_out.cross_attentions
from transformers import AutoImageProcessor, Blip2Model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vision_encoder(name="Salesforce/blip2-flan-t5-xl"):
    process = AutoImageProcessor.from_pretrained(name, use_safetensors=True)
    model = Blip2Model.from_pretrained(name).vision_model.to(device)
    model.eval()

    def encode(image):
        pixel_values = process(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            feats = model(pixel_values).last_hidden_state
        return feats
    
    return encode
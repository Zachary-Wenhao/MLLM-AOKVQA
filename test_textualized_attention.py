import base64
import os
import random
import io

import openai
from dotenv import load_dotenv
from PIL import Image
import torch

from aokvqa_dataset import AOKVQADataset
from utils import load_config
from models.cross_attention_model import CrossAttentionExtractor

load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_vanilla_gpt4o_mc(question, mc_choices, image_path):
    config = load_config("config.yaml")
    vision_model = config["vision_model"]
    text_model = config["text_model"]

    client = openai.OpenAI(
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url="https://cmu.litellm.ai",
    )

    extractor = CrossAttentionExtractor(
        vision_model_name=vision_model,
        text_model_name=text_model
    ).to(device)
    img = Image.open(image_path).convert("RGB")
    attns = extractor([img], [question], [rationales])
    # rebuild token list to map indices to actual tokens
    tokenizer = extractor.tokenizer
    txt_inputs = tokenizer(
        [f"Question: {question}\nRationale: {rationales}"],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    token_list = tokenizer.convert_ids_to_tokens(txt_inputs.input_ids[0])
    # use last layer cross-attn: shape [batch, heads, seq_len, q_len]
    attn_tensor = attns[-1][0]  # [heads, seq_len, q_len]
    scores = attn_tensor.mean(dim=2)  # average over q_len → [heads, seq_len]
    top_idxs = scores.argmax(dim=1).tolist()  # [heads]
    rationale_list = [
        f"Head {i} focused on token '{token_list[idx]}'"
        for i, idx in enumerate(top_idxs)
    ]
    rationale_text = "; ".join(rationale_list)
    # resize and compress image to reduce context size
    img_for_enc = Image.open(image_path).convert("RGB")
    img_for_enc.thumbnail((128, 128))
    buf = io.BytesIO()
    img_for_enc.save(buf, format="JPEG", quality=20)
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("ascii")

    full_prompt = (
        f"Question: {question}\n"
        f"Visual Rationale: {rationale_text}\n"
        f"A. {mc_choices[0]}\n"
        f"B. {mc_choices[1]}\n"
        f"C. {mc_choices[2]}\n"
        f"D. {mc_choices[3]}\n"
        f"Image:\n![image](data:image/jpeg;base64,{img_b64})\n\n"
        "Please respond with the single best answer text (not the letter). Only include the answer in your response."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a visual question answering assistant. Given an image and a multiple-choice question, select the correct answer and provide a brief one-sentence rationale."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.0,
    )
    answer = response.choices[0].message.content.strip()
    return answer, rationale_text


if __name__ == "__main__":
    val_dataset = AOKVQADataset(
        aokvqa_dir="datasets/aokvqa",
        coco_dir="datasets/coco",
        split="val"
    )

    for i in range(1):
        # sample = random.choice(val_dataset)
        sample = val_dataset[1]

        question = sample['question']
        image_path = sample['image_path']
        mc_choices = [
            sample['mc_choices_0'],
            sample['mc_choices_1'],
            sample['mc_choices_2'],
            sample['mc_choices_3']
        ]
        mc_answers = sample['mc_answers']
        mc_answers_index = sample['mc_answers_index']
        da_answers = sample['da_answers']
        rationales = sample['rationales']

        answer, rationale_text = test_vanilla_gpt4o_mc(question, mc_choices, image_path)
        
        print("Question:", question)
        print("Image Path:", image_path)
        print(f"Rationales: {rationales}")
        print(f"Choices: {mc_choices[0]}, {mc_choices[1]}, {mc_choices[2]}, {mc_choices[3]}")
        print(f"Correct MC Answer: {mc_answers} | GPT-4o Answer: {answer}")
        print(f"Visual Rationale: {rationale_text}")

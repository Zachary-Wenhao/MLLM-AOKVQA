
import base64
import io
import os
import random

import openai
from dotenv import load_dotenv
from PIL import Image
import torch

from aokvqa_dataset import AOKVQADataset
from utils import load_config

load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_vanilla_gpt4o_mc(question, mc_choices, image_path):
    config = load_config("config.yaml")
    client = openai.OpenAI(
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url="https://cmu.litellm.ai",
    )

    img = Image.open(image_path).convert('RGB')
    img.thumbnail((128, 128))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=30)
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('ascii')

    full_prompt = (
        f"Question: {question}\n"
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
    return answer


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

        answer = test_vanilla_gpt4o_mc(question, mc_choices, image_path)
        
        print("Question:", question)
        print("Image Path:", image_path)
        print(f"Rationales: {rationales}")
        print(f"Choices: {mc_choices[0]}, {mc_choices[1]}, {mc_choices[2]}, {mc_choices[3]}")
        print(f"Correct MC Answer: {mc_answers} | GPT-4o Answer: {answer}")

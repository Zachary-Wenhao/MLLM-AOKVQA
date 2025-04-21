import os
import json
from PIL import Image
from torch.utils.data import Dataset

class AOKVQADataset(Dataset):
    def __init__(
        self,
        aokvqa_dir: str,
        coco_dir: str,
        split: str,
        img_transform=None,
    ):
        assert split in {"train", "val", "test"}
        self.aokvqa_dir = aokvqa_dir
        self.coco_dir = coco_dir
        self.split = split
        self.img_transform = img_transform

        self.data = self._load_json()    

        self.image_ids = []
        self.question_ids = []
        self.questions = []
        self.mc_choices_0 = []
        self.mc_choices_1 = []
        self.mc_choices_2 = []
        self.mc_choices_3 = []
        self.mc_answers = []
        self.mc_answers_index = []
        self.da_answers = []
        self.rationales = []

        for item in self.data:
            self.image_ids.append(item["image_id"])
            self.question_ids.append(item["question_id"])
            self.questions.append(item["question"])
            self.mc_choices_0.append(item["choices"][0])
            self.mc_choices_1.append(item["choices"][1])
            self.mc_choices_2.append(item["choices"][2])
            self.mc_choices_3.append(item["choices"][3])
            self.mc_answers.append(item["choices"][item["correct_choice_idx"]])
            self.mc_answers_index.append(item["correct_choice_idx"])
            self.da_answers.append(item["direct_answers"])
            if "rationales" in item:
                self.rationales.append(item["rationales"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        pixel_value = Image.open(self._coco_path(self.image_ids[index]))
        pixel_value = pixel_value.convert("RGB")
        if self.img_transform:
            pixel_value = self.img_transform(pixel_value)
        
        return {
            "image_id": self.image_ids[index],
            "pixel_values": pixel_value,
            "question_id": self.question_ids[index],
            "question": self.questions[index],
            "mc_choices_0": self.mc_choices_0[index],
            "mc_choices_1": self.mc_choices_1[index],
            "mc_choices_2": self.mc_choices_2[index],
            "mc_choices_3": self.mc_choices_3[index],
            "mc_answers": self.mc_answers[index],
            "mc_answers_index": self.mc_answers_index[index],
            "da_answers": self.da_answers[index],
            "rationales": self.rationales[index] \
                if index < len(self.rationales) else None,
        }

    # ------------- helper functions -------------
    
    def _load_json(self):
        fp = self._aokvqa_path()
        with open(fp, "r") as f:
            return json.load(f)      
        
    def _aokvqa_path(self):
        return os.path.join(self.aokvqa_dir, f"aokvqa_v1p0_{self.split}.json")

    def _coco_path(self, image_id: int) -> str:
        return os.path.join(self.coco_dir, f"{self.split}2017", f"{image_id:012}.jpg")


if __name__ == "__main__":
    import random
    train_dataset = AOKVQADataset(
        aokvqa_dir="aokvqa",
        coco_dir="coco",
        split="train",
        img_transform=None,
    )
    val_dataset = AOKVQADataset(
        aokvqa_dir="aokvqa",
        coco_dir="coco",
        split="val",
        img_transform=None,
    )

    print("Loaded datasets")
    print(f"train_dataset: {len(train_dataset)} samples | val_dataset: {len(val_dataset)} samples")

    print("Sample data:")
    for i in range(5):
        item = random.choice(train_dataset)
        print(f"Sample {i+1}:")
        print(f"Question ID: {item['question_id']} | Image ID: {item['image_id']}")
        print(f"Question: {item['question']}")
        print(f"Choices: {item['mc_choices_0']}, {item['mc_choices_1']}, {item['mc_choices_2']}, {item['mc_choices_3']}")
        print(f"Correct MC Answer: {item['mc_answers']} | Index: {item['mc_answers_index']}")
        print(f"Direct Answers: {item['da_answers']}")
        print(f"Rationales: {item['rationales']}")
        print()

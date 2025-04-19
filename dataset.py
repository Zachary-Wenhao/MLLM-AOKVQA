import json
import os
import random
from typing import Dict
from PIL import Image
from torch.utils.data import Dataset

class AOKVQADataset(Dataset):
    def __init__(
        self,
        aokvqa_dir: str,
        coco_dir: str,
        split: str,
        version: str = "v1p0",
        img_transform=None,         
    ):
        assert split in {"train", "val", "test"}
        self.split = split
        self.aokvqa_dir = aokvqa_dir
        self.coco_dir = coco_dir
        self.version = version
        self.img_transform = img_transform

        self.data = self._load_json()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        img = Image.open(self._coco_path(item["image_id"])).convert("RGB")
        if self.img_transform:
            img = self.img_transform(img)

        return {
            "image": img,
            "question": item["question"],
            "rationales": item.get("rationales", ""),
            "choices": item["choices"],
            "correct_choice_idx": item["correct_choice_idx"],
            "direct_answers": item["direct_answers"],
        }

    # ------------- helper functions -------------
    
    def _load_json(self):
        fp = self._aokvqa_path()
        with open(fp, "r") as f:
            return json.load(f)      
        
    def _aokvqa_path(self):
        return os.path.join(
            self.aokvqa_dir, f"aokvqa_{self.version}_{self.split}.json"
        )

    def _coco_path(self, image_id: int) -> str:
        return os.path.join(
            self.coco_dir, f"{self.split}2017", f"{image_id:012}.jpg"
        )
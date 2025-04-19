import os
import torch
from torch.utils.data import DataLoader
from dataset import AOKVQADataset
import encoders                        

AOKVQA_DIR = "dataset/aokvqa/"
COCO_DIR   = "dataset/coco/"

def collate_fn(batch):
    texts = [
        f"Question: {ex['question']} Rationales: {ex['rationales']}"
        for ex in batch
    ]
    text_out = encoders.encode_text(texts)      

    imgs = torch.stack([ex["image"] for ex in batch]).to(encoders.device)
    img_emb = encoders.encode_images(imgs)       

    out = {
        **text_out,                                
        "img_emb": img_emb,       
        "choices": [ex["choices"] for ex in batch],
        "correct_choice_idx": torch.tensor(
            [ex["correct_choice_idx"] for ex in batch]
        ),
        "direct_answers": [ex["direct_answers"] for ex in batch],
    }
    return out

def get_loaders(batch_size=8, num_workers=4):
    train_dataset = AOKVQADataset(AOKVQA_DIR, COCO_DIR, split="train", img_transform=encoders.clip_preprocess)
    val_dataset   = AOKVQADataset(AOKVQA_DIR, COCO_DIR, split="val",   img_transform=encoders.clip_preprocess)
    test_dataset  = AOKVQADataset(AOKVQA_DIR, COCO_DIR, split="test",  img_transform=encoders.clip_preprocess)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,   num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size, shuffle=False,  num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size, shuffle=False,  num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    get_loaders()
import json
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_json(aokvqa_dir, split):
    with open(aokvqa_path(aokvqa_dir, split), "r") as f:
        data = json.load(f)
    return data

def aokvqa_path(aokvqa_dir, split):
    return f"{aokvqa_dir}/aokvqa_v1p0_{split}.json"

def coco_path(coco_dir, split, image_id):
    return f"{coco_dir}/{split}2017/{image_id:012}.jpg"
import os
import json
import string
import pandas as pd

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

def download_dataset(data_dir):
    image_dir = os.path.join(data_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    original_ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)")
    vision_ds = load_dataset("MMMU/MMMU_Pro", "vision")

    # ========== process original subset ==========
    original_list = []
    id_to_qa = {}

    for split in original_ds:
        for item in tqdm(original_ds[split], desc=f"Processing original {split}"):
            new_item = item.copy()
            
            # process images
            for i in range(1, 8):
                img_key = f"image_{i}"
                img = new_item[img_key]
                if img is not None:
                    filename = f"{new_item['id']}_{i}.png"
                    img_path = os.path.join(image_dir, filename)
                    
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    img.save(img_path)
                    
                    new_item[img_key] = filename
                else:
                    new_item[img_key] = None
            
            id_to_qa[new_item["id"]] = {
                "question": new_item["question"],
                "options": new_item["options"]
            }
            
            original_list.append(new_item)

    # save original.json
    with open(os.path.join(data_dir, "original.json"), "w") as f:
        json.dump(original_list, f, indent=2)

    # ========== process vision subset ==========
    vision_list = []

    for split in vision_ds:
        for item in tqdm(vision_ds[split], desc=f"Processing vision {split}"):
            new_item = item.copy()
            original_id = item["id"]
            
            # process images
            img = new_item["image"]
            filename = f"vision_{original_id}.png"
            img_path = os.path.join("MMMU_Pro/images", filename)
            
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(img_path)
            
            new_item["image"] = filename
            
            # add question and options
            if original_id in id_to_qa:
                new_item.update(id_to_qa[original_id])
            else:
                print(f"Warning: Missing original data for {original_id}")
                new_item["question"] = ""
                new_item["options"] = []
            
            vision_list.append(new_item)

    # save vision.json
    with open(os.path.join(data_dir, "vision.json"), "w") as f:
        json.dump(vision_list, f, indent=2)

    print("download MMMU_Pro success!")

def load_mmmupro_dataset(data_dir, subset='original'):
    """Load and preprocess MMMUPro dataset"""
    metadata_path = os.path.join(data_dir, f"{subset}.json")
    if not os.path.exists(metadata_path):
        download_dataset(data_dir)
    with open(metadata_path, 'r') as f:
        dataset = json.load(f)
    
    df = pd.DataFrame(dataset)
    
    image_dir = os.path.join(data_dir, "images")

    image_columns = [
        col for col in df.columns 
        if col.startswith('image')
    ]
    
    for col in image_columns:
        df[col] = df[col].apply(
            lambda x: os.path.join(image_dir, x) if pd.notnull(x) else None
        )
    
    def process_options(row):
        options = row['options']
        if isinstance(options, str):
            try:
                options = eval(options)
            except:
                pass
        choices = {}
        for idx, opt in enumerate(options):
            choices[string.ascii_uppercase[idx]] = str(opt)
        return choices
    
    df['choices'] = df.apply(process_options, axis=1)
    df['subset'] = subset
    
    annotated_data = []
    for _, row in df.iterrows():
        annotation = row.to_dict()
        images = [v for k, v in annotation.items() 
                if k.startswith('image_') and pd.notnull(v)]
        annotation['images'] = images
        annotated_data.append(annotation)
    
    return annotated_data

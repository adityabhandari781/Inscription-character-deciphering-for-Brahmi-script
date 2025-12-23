import json
import os
import cv2
import glob
from sklearn.model_selection import train_test_split
import shutil

# --- Configuration ---
RAW_DATA_DIR = "/mnt/windows/Users/adity/Downloads/output_clean"  # Folder containing your .jpg/.png and .json files
OUTPUT_DIR = "/mnt/windows/Users/adity/Downloads/output_yolo"
IMG_EXTENSIONS = {".png"}

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Converts [xmin, ymin, xmax, ymax] to [x_center, y_center, width, height]
    normalized by image dimensions.
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Calculate center, width, height
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    
    w = xmax - xmin
    h = ymax - ymin
    x_center = xmin + w / 2.0
    y_center = ymin + h / 2.0
    
    # Normalize
    x_center *= dw
    w *= dw
    y_center *= dh
    h *= dh
    
    return x_center, y_center, w, h

def main():
    # 1. Collect all unique characters to build a class map
    print("Building class vocabulary...")
    json_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.json"))
    unique_chars = set()
    
    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                unique_chars.add(item['char'])
    
    # Create a mapping: char -> id
    # Sort to ensure reproducibility
    char_to_id = {char: idx for idx, char in enumerate(sorted(list(unique_chars)))}
    id_to_char = {idx: char for char, idx in char_to_id.items()}
    
    print(f"Found {len(char_to_id)} unique characters.")
    
    # Save the mapping for later use (inference)
    with open("class_mapping.json", "w", encoding='utf-8') as f:
        json.dump(char_to_id, f, ensure_ascii=False, indent=2)

    # 2. Prepare Directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    # 3. Split Data
    # Get list of base filenames (without extension)
    files = [os.path.splitext(os.path.basename(x))[0] for x in json_files]
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)
    
    splits = {'train': train_files, 'val': val_files}

    # 4. Process Files
    print("Converting and moving files...")
    for split, file_list in splits.items():
        for base_name in file_list:
            # Find the image file (check different extensions)
            image_path = None
            for ext in IMG_EXTENSIONS:
                temp_path = os.path.join(RAW_DATA_DIR, base_name + ext)
                if os.path.exists(temp_path):
                    image_path = temp_path
                    break
            
            json_path = os.path.join(RAW_DATA_DIR, base_name + ".json")
            
            if not image_path:
                print(f"Warning: Image for {base_name} not found. Skipping.")
                continue

            # Read Image to get dims
            img = cv2.imread(image_path)
            if img is None: continue
            height, width = img.shape[:2]

            # Read JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)

            # Convert Annotations
            yolo_lines = []
            for ann in annotations:
                char = ann['char']
                if char not in char_to_id: continue # Should not happen
                
                cls_id = char_to_id[char]
                bbox = ann['bbox'] # [xmin, ymin, xmax, ymax]
                
                xc, yc, w, h = convert_bbox_to_yolo(bbox, width, height)
                yolo_lines.append(f"{cls_id} {xc} {yc} {w} {h}")

            # Write Label File
            label_out_path = os.path.join(OUTPUT_DIR, 'labels', split, base_name + ".txt")
            with open(label_out_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_lines))

            # Copy Image File
            shutil.copy(image_path, os.path.join(OUTPUT_DIR, 'images', split, os.path.basename(image_path)))

    # 5. Create YAML configuration for YOLO
    yaml_content = f"""
path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val

nc: {len(char_to_id)}
names: {list(id_to_char.values())}
    """
    
    with open("brahmi_config.yaml", "w", encoding='utf-8') as f:
        f.write(yaml_content)
        
    print("Data preparation complete. 'brahmi_config.yaml' created.")

if __name__ == "__main__":
    main()
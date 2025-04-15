import os
import shutil
from pathlib import Path

# Define paths
original_dataset = "dataset/ITA.yolo/train"
new_dataset = "dataset_2"  # Single folder (no subdirectories)

# Create the new directory if it doesn't exist
os.makedirs(new_dataset, exist_ok=True)

# Supported image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# Get all image files from the original dataset
image_files = []
for root, _, files in os.walk(os.path.join(original_dataset, "images")):
    for file in files:
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(root, file))

# Process each image file
for image_path in image_files:
    # Get the corresponding label file path
    rel_path = os.path.relpath(image_path, os.path.join(original_dataset, "images"))
    label_path = os.path.join(original_dataset, "labels", Path(rel_path).with_suffix('.txt'))
    
    # Check if label file is missing or empty
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        # Copy the image directly into dataset_2 (no subfolders)
        shutil.copy2(image_path, os.path.join(new_dataset, os.path.basename(image_path)))

print(f"Done! {len(os.listdir(new_dataset))} images copied to {new_dataset}.")
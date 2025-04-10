#print("Om Namaha Shivaya")
import os
import random
import cv2
import numpy as np
from glob import glob

# Define paths
base_dir = "extracted_landmine_patches"
background_dir = "diverse_backgrounds"
dataset_dir = "dataset_try_1"
images_dir = os.path.join(dataset_dir, "images_1")
labels_dir = os.path.join(dataset_dir, "labels_1")

# Create dataset directories
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Define landmine patch folders and corresponding labels
patch_folders = {
    "butterfly_patches_from_ITA_train": 0,
    "starfish_patches_from_ITA_train": 1
}

# Get background images
background_images = glob(os.path.join(background_dir, "*.jpg")) + glob(os.path.join(background_dir, "*.png"))

# Function to overlay patch onto a background
def overlay_patch(background, patch):
    bh, bw, _ = background.shape
    ph, pw, _ = patch.shape
    
    # Ensure patch fits within the background
    if ph > bh or pw > bw:
        scale = min(bh / ph, bw / pw) * 0.5  # Scale down if too big
        patch = cv2.resize(patch, (int(pw * scale), int(ph * scale)))
        ph, pw, _ = patch.shape
    
    # Random position for the patch
    x_offset = random.randint(0, bw - pw)
    y_offset = random.randint(0, bh - ph)
    
    # Place the patch onto the background
    background[y_offset:y_offset+ph, x_offset:x_offset+pw] = patch
    
    # YOLO format bounding box
    x_center = (x_offset + pw / 2) / bw
    y_center = (y_offset + ph / 2) / bh
    width = pw / bw
    height = ph / bh
    
    return background, (x_center, y_center, width, height)

# Process each patch type
for patch_folder, label in patch_folders.items():
    patch_files = glob(os.path.join(base_dir, patch_folder, "*.jpg")) + glob(os.path.join(base_dir, patch_folder, "*.png"))
    
    for patch_file in patch_files:
        patch = cv2.imread(patch_file)
        bg_file = random.choice(background_images)
        background = cv2.imread(bg_file)
        
        # Overlay the patch
        composite_image, bbox = overlay_patch(background, patch)
        
        # Save the new image
        image_name = os.path.basename(patch_file)
        image_path = os.path.join(images_dir, image_name)
        cv2.imwrite(image_path, composite_image)
        
        # Save the YOLO label
        label_path = os.path.join(labels_dir, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))
        with open(label_path, "w") as f:
            f.write(f"{label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

print("Dataset generation complete!")


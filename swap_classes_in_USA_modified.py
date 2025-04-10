

import os

def swap_class_ids_in_file(file_path):
    """
    Swaps class IDs 0 and 1 in a given annotation file.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()  # Read all lines in the file

    # Swap class IDs
    updated_lines = []
    for line in lines:
        parts = line.strip().split()  # Split the line into parts
        if parts:  # Ensure the line is not empty
            class_id = int(parts[0])  # Extract the class ID
            if class_id == 0:
                parts[0] = '1'  # Swap 0 to 1
            elif class_id == 1:
                parts[0] = '0'  # Swap 1 to 0
            updated_lines.append(' '.join(parts) + '\n')  # Reconstruct the line

    # Write the updated lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(updated_lines)

def swap_class_ids_in_directory(directory):
    """
    Recursively swaps class IDs 0 and 1 in all .txt files in the given directory and its subdirectories.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):  # Process only .txt files
                file_path = os.path.join(root, file)
                swap_class_ids_in_file(file_path)
                print(f"Processed: {file_path}")

# Path to the labels directory
labels_directory = "/home/sl3088/yolov8_suland/dataset/USA_modified.yolo/val/labels"

# Swap class IDs in all .txt files
swap_class_ids_in_directory(labels_directory)

print("Class IDs have been swapped successfully!")
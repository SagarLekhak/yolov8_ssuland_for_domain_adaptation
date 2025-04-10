

import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import os
from tkinter import Tk, filedialog

# Function to upload images
def upload_images():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    return file_paths

# Function to interactively segment and cut out regions
def segment_and_cutout(image_path, predictor):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set the image for SAM
    predictor.set_image(image)

    # Display the image for interactive segmentation
    plt.imshow(image)
    plt.title("Click to select points for segmentation. Press 'Enter' to confirm.")
    plt.axis("on")

    # Collect points from user input
    points = []
    labels = []  # 1 for foreground, 0 for background
    def onclick(event):
        if event.button == 1:  # Left click for foreground
            points.append([event.xdata, event.ydata])
            labels.append(1)
            plt.scatter(event.xdata, event.ydata, c="green", marker="o")
            plt.draw()
        elif event.button == 3:  # Right click for background
            points.append([event.xdata, event.ydata])
            labels.append(0)
            plt.scatter(event.xdata, event.ydata, c="red", marker="x")
            plt.draw()

    # Connect the click event
    cid = plt.gcf().canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    # Convert points and labels to numpy arrays
    input_points = np.array(points)
    input_labels = np.array(labels)

    # Perform segmentation
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,  # Return only the best mask
    )

    # Get the best mask
    mask = masks[0]

    # Cut out the segmented region
    segmented_image = np.zeros_like(image)
    segmented_image[mask] = image[mask]

    # Save the cutout
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{base_name}_segment.png"
    cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    print(f"Segmented image saved as {output_path}")

# Main function
def main():
    # Load the SAM model
    sam_checkpoint = "sam_vit_h_4b8939.pth"  # Path to the SAM checkpoint
    model_type = "vit_h"  # Model type
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)

    # Upload images
    image_paths = upload_images()
    if not image_paths:
        print("No images selected. Exiting.")
        return

    # Segment and save cutouts for each image
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        segment_and_cutout(image_path, predictor)

    print("All images processed.")

if __name__ == "__main__":
    main()
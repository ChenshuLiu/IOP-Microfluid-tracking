'''
cropping red microfluidic chamber from frame images
based on the data_acquisition.py results
using empirically determined hsv range for extraction
'''

import cv2
import numpy as np
import os

def process_image(image_path, output_path, effect = "crop"):
    """Extract red regions and grayscale the rest of the image."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Skipping {image_path}, unable to load image.")
        return
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV bounds for red (two ranges due to hue wrap-around)
    lower_red1 = np.array([0, 140, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 140, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red regions
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2  # Combine masks

    # Convert non-red areas to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels

    if effect == "crop":
        # Keeping only the red, others are NOT grayscale
        result = cv2.bitwise_and(frame, frame, mask = mask)
    elif effect == "enhance":
        # Apply mask: Keep red areas, convert others to grayscale
        result = np.where(mask[:, :, None] > 0, frame, gray)
        
    # Save the processed image
    cv2.imwrite(output_path, result)

source_dir = 'CNN_frame_data/Training_data_coarse_train_val'
output_dir = 'CNN_frame_data/Training_data_red_enhance_train_val'

os.makedirs(output_dir, exist_ok=True)

for train_val_folder in os.listdir(source_dir):
    # train, val two folders
    train_val_folder_dir = os.path.join(source_dir, train_val_folder)
    output_train_val_folder_dir = os.path.join(output_dir, train_val_folder)
    os.makedirs(output_train_val_folder_dir, exist_ok=True)
    if not os.path.isdir(train_val_folder_dir):
        continue

    for pressure_folder in sorted(os.listdir(train_val_folder_dir)):
        # different pressure folders, 16-59 pressure levels
        pressure_folder_dir = os.path.join(train_val_folder_dir, pressure_folder)
        output_pressure_folder_dir = os.path.join(output_train_val_folder_dir, pressure_folder)
        if not os.path.isdir(pressure_folder_dir):
            continue
        os.makedirs(output_pressure_folder_dir, exist_ok=True)
        for img in os.listdir(pressure_folder_dir):
            img_dir = os.path.join(pressure_folder_dir, img)
            output_img_dir = os.path.join(output_pressure_folder_dir, img)
            process_image(img_dir, output_img_dir, effect="enhance")


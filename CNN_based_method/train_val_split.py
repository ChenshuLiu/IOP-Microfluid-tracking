import os
import shutil
import random

# Define paths
original_dataset_path = "./CNN_frame_data/Training_data_coarse"  # Original dataset folder
split_dataset_path = "./CNN_frame_data/Training_data_coarse_train_val"  # New dataset folder

train_path = os.path.join(split_dataset_path, "train")
val_path = os.path.join(split_dataset_path, "val")

# Split ratio
val_split = 0.2  # 20% of data for validation

# Ensure new dataset folder exists
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Loop through each category folder
for label in os.listdir(original_dataset_path):
    label_path = os.path.join(original_dataset_path, label)
    
    if not os.path.isdir(label_path):  # Skip non-folder files
        continue
    
    # Create corresponding train/val subfolders
    os.makedirs(os.path.join(train_path, label), exist_ok=True)
    os.makedirs(os.path.join(val_path, label), exist_ok=True)
    
    # Get all image files in the current label folder
    images = [f for f in os.listdir(label_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle images
    random.shuffle(images)
    
    # Split into training and validation
    split_idx = int(len(images) * (1 - val_split))
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Copy files to corresponding directories
    for img in train_images:
        shutil.copy(os.path.join(label_path, img), os.path.join(train_path, label, img))
    
    for img in val_images:
        shutil.copy(os.path.join(label_path, img), os.path.join(val_path, label, img))

print("Dataset successfully split and saved in 'dataset_split'!")
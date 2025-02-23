import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
from transformers import ViTModel, ViTConfig

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Traverse the dataset folder and collect image paths & labels
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if not os.path.isdir(label_path):
                continue
            for img_file in os.listdir(label_path):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(label_path, img_file), int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # Load original size
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)
    
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))  # Ensures fixed size output
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)  # Regression task (predict single value)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))  # Pool to (8,8)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class DynamicPatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, patch_size=16):
        super(DynamicPatchEmbedding, self).__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, C, H, W) -> (B, D, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # Reshape to (B, Num_Patches, D)
        return x

class DynamicViT(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224", num_outputs=1):
        super(DynamicViT, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)

        # Replace patch embedding with dynamic one
        self.vit.embeddings.patch_embeddings = DynamicPatchEmbedding(
            in_channels=3, embed_dim=self.vit.config.hidden_size, patch_size=16
        )

        # Regression head for fine-grained changes
        self.regression_head = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)  # Predict continuous change
        )

    def forward(self, x):
        embeddings = self.vit.embeddings.patch_embeddings(x)  # Dynamically extracted patches
        outputs = self.vit.encoder(embeddings).last_hidden_state  # Pass through Transformer
        pooled_output = outputs.mean(dim=1)  # Global Average Pooling (GAP)
        return self.regression_head(pooled_output)
    
class RollingBuffer:
    '''
    for averaging the predicted output to make the reading more stable
    '''
    def __init__(self, size, dtype=np.float64):
        self.buffer = []
        self.size = size
        self.current_size = 0  # Keeps track of the current number of elements in the buffer

    def add(self, element):
        if self.current_size < self.size:
            # If the buffer is not full, add the element to the end and increment the size
            # self.buffer[self.current_size] = element
            self.buffer.append(element)
            self.current_size += 1
        else:
            # If the buffer is full, shift the array left and add the new element to the last index
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = element

    def get(self):
        return self.buffer[:self.current_size]
    
    def defined(self):
        define_status = 0 # not define by default
        if self.current_size == 3:
            define_status = 1
        return bool(define_status)
    
def red_region_crop(image, effect = "crop"):
    '''
    for the pressure prediction pipeline to extract/enhance the red regions
    args:
    effect: "crop" for only extracting the red region, "enhance" for grayscale the rest of regions
    '''
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels

    if effect == "crop":
        # Keeping only the red, others are NOT grayscale
        result = cv2.bitwise_and(image, image, mask = mask)
    elif effect == "enhance":
        # Apply mask: Keep red areas, convert others to grayscale
        result = np.where(mask[:, :, None] > 0, image, gray)

    return result # numpy array

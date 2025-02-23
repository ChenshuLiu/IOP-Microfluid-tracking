import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.optim as optim
from tqdm import tqdm
from helper import *
from torch.optim.lr_scheduler import ExponentialLR
import csv

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

checkpoint_save_dir = os.path.join(os.getcwd(), "model_checkpoints")
os.makedirs(checkpoint_save_dir, exist_ok=True)

# Define transformations (without resizing)
transform = transforms.Compose([
    transforms.ToTensor(),  # Keep original dimensions
])

# Load datasets
train_dataset = CustomDataset("./CNN_frame_data/Training_data_coarse_train_val/train", transform=transform)
val_dataset = CustomDataset("./CNN_frame_data/Training_data_coarse_train_val/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print("Dataloader done")
    
# model = CNN_Model().to(device)
model = CNN_Model()
# model.load_state_dict(torch.load('cnn_model.pth', map_location = device)) # for finetuning from existing pretrained model
model.to(device)
    
# Define loss function (MSE for regression) and optimizer
criterion = nn.MSELoss()  # Since we are predicting a numerical value
optimizer = optim.Adam(model.parameters(), lr=0.001)
gamma = 0.95
scheduler = ExponentialLR(optimizer, gamma=gamma)
print("CNN model defined")

csv_filename = "CNN_50ep_training_results.csv"
file_exists = os.path.isfile(csv_filename)

# Training loop
def train_model(model, train_loader, val_loader, num_epochs=10):
    global file_exists
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Wrap train_loader with tqdm for progress bar
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for images, labels in train_progress:
            images, labels = images.to(device), labels.float().to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            # optimizing for the integer value, less on the decimal places
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            train_progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")  # Update tqdm bar with loss
        scheduler.step()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

        with open(csv_filename, mode='a', newline = "") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Epoch', 'Training Loss', 'Validation loss', 'Learning rater'])
                file_exists = True
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, current_lr])

        # saving model checkpoints
        checkpoint_path = os.path.join(checkpoint_save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

# Train the model
train_model(model, train_loader, val_loader, num_epochs=50)

# Save trained model
torch.save(model.state_dict(), "cnn_model_50ep.pth")

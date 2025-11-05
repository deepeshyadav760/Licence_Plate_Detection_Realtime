import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# --- Constants ---
# Define image dimensions for the recognition model
IMG_WIDTH = 200
IMG_HEIGHT = 50
# All possible characters in the license plates (including a blank for CTC loss)
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZTN"


class LicensePlateDataset(Dataset):
    """
    Custom PyTorch Dataset for License Plate Recognition.
    - It reads the full image.
    - Uses bounding box data to crop the license plate.
    - Applies transformations to the cropped plate.
    - Pairs the processed image with its text label.
    """
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['img_id']
        label = self.df.iloc[idx]['text']
        img_path = os.path.normpath(os.path.join(self.img_dir, img_name))

        # --- NEW: Add a try-except block to handle missing files ---
        try:
            # Open the full image
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # If the file doesn't exist, print a warning and return None.
            # The DataLoader's collate_fn will handle this by skipping the sample.
            print(f"Warning: File not found, skipping: {img_path}")
            return None

        # Get bounding box coordinates to crop the plate
        xmin = self.df.iloc[idx]['xmin']
        ymin = self.df.iloc[idx]['ymin']
        xmax = self.df.iloc[idx]['xmax']
        ymax = self.df.iloc[idx]['ymax']

        # Crop the image to isolate the license plate
        plate_image = image.crop((xmin, ymin, xmax, ymax))

        # Convert to grayscale for the model
        plate_image = plate_image.convert("L")
        
        # Apply transformations (resize, convert to tensor)
        if self.transform:
            plate_image = self.transform(plate_image)

        return plate_image, label


def encode_text(text):
    """Encodes text label into a tensor of character indices."""
    encoded = [CHARS.find(char) for char in text]
    return torch.tensor(encoded, dtype=torch.long)

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) for text recognition.
    """
    def __init__(self, num_chars):
        super(CRNN, self).__init__()
        # Convolutional layers (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # Input: 64x50x200 -> Output: 64x25x100
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # Input: 128x25x100 -> Output: 128x12x50
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)), # Input: 256x12x50 -> Output: 256x6x50
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)), # Input: 512x6x50 -> Output: 512x3x50
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=(0, 1)), 
            nn.BatchNorm2d(512), nn.ReLU(True) # Output: 512x1x49
        )
        # Recurrent layers (RNN)
        self.rnn = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True),
        )
        # Fully connected layer for outputting character probabilities
        self.fc = nn.Linear(512, num_chars) # 256 * 2 (bidirectional) = 512

    def forward(self, x):
        # Pass input through the convolutional layers
        x = self.cnn(x)
        
        # Prepare the feature map for the RNN
        b, c, h, w = x.size()
        
        # This assertion will now pass
        assert h == 1, f"The height of the feature map must be 1, but got {h}"
        
        x = x.squeeze(2) # Remove the height dimension: [b, c, w]
        x = x.permute(0, 2, 1)  # Rearrange for RNN: [batch, width, channels]
        
        # Pass through the recurrent layers
        x, _ = self.rnn(x)
        
        # Pass through the final fully connected layer
        x = self.fc(x)
        
        # Prepare for CTC loss function
        x = x.permute(1, 0, 2) # [width, batch, num_chars]
        return x


def collate_fn(batch):
    """
    Custom collate function to filter out None values from the batch.
    This is used to handle images that were not found during dataset loading.
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def train_model():
    """
    Main function to orchestrate the training process.
    """
    train_image_dir = "data/train"
    
    # Load and merge the detection and recognition data
    detection_df = pd.read_csv("licplatesdetection_train.csv")
    recognition_df = pd.read_csv("licplatesrecognition_train.csv")
    df = pd.merge(detection_df, recognition_df, on="img_id")

    # --- 1. MODIFICATION: Add Data Augmentation ---
    # Replace your old 'transform' with this new one.
    # These augmentations will create new variations of your training data on-the-fly.
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        # Add random transformations
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize for better training stability
    ])

    dataset = LicensePlateDataset(df, train_image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Initialize model, loss, and optimizer
    model = CRNN(len(CHARS) + 1) # +1 for the blank character
    criterion = nn.CTCLoss(blank=len(CHARS), zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # A slightly lower learning rate can be more stable

    # --- 2. MODIFICATION: Increase Training Epochs ---
    # --- Change num_epochs from 20 to a higher value like 75 or 100 ---
    print("Starting model training with data augmentation for 75 epochs...")
    num_epochs = 75 
    
    for epoch in range(num_epochs):
        # Add a tqdm progress bar for the inner loop to see progress per epoch
        loop = tqdm(dataloader, leave=True)
        for i, (images, labels) in enumerate(loop):
            optimizer.zero_grad()
            
            outputs = model(images)
            outputs = outputs.log_softmax(2)
            
            input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long)
            encoded_labels = [encode_text(label) for label in labels]
            targets = torch.cat(encoded_labels)
            target_lengths = torch.tensor([len(label) for label in encoded_labels], dtype=torch.long)
            
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            
            loss.backward()
            # Gradient clipping can prevent unstable training
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            # Update the progress bar description
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

    # Save the NEWLY trained and much better model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/crnn_augmented.pth") # Save with a new name
    print("Model training complete and saved to models/crnn_augmented.pth")

if __name__ == "__main__":
    train_model()
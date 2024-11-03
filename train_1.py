import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
from tqdm import tqdm  # Import tqdm for progress visualization
from models.cnn_models import SimpleCNN
from dataset.custom_dataset_1 import CustomDataset  # Correct import for the custom dataset

# Define file paths as constants
CSV_FILE_PATH = r'C:\Users\Sandhra George\avalanche\data\dataset.csv'
ROOT_DIR_PATH = r'C:\Users\Sandhra George\avalanche'

# Load data into a DataFrame for easier processing
data = pd.read_csv(CSV_FILE_PATH)

# Limit dataset to the first 3084 images (excluding header)
data_limited = data.iloc[1:3085].reset_index(drop=True)

# Filter the dataset to only include images containing "print0"
data_filtered = data_limited[data_limited.iloc[:, 0].str.contains('print0', na=False)]

# Split the dataset into separate DataFrames for each class
class_datasets = {}
for class_id in data_filtered['hotend_class'].unique():
    class_datasets[class_id] = data_filtered[data_filtered['hotend_class'] == class_id].sample(frac=1, random_state=42)

# Print counts of each class dataset
for class_id, df in class_datasets.items():
    print(f'Class {class_id} dataset size: {len(df)}')

# Find the class with the minimum number of images
min_class_size = min(len(df) for df in class_datasets.values())
print(f'Minimum class size: {min_class_size}')

# Create balanced datasets by taking the minimum number of images from each class
balanced_data = []
for class_id, class_data in class_datasets.items():
    balanced_data.append(class_data.sample(n=min_class_size, random_state=42))

# Combine the balanced data
balanced_data = pd.concat(balanced_data).reset_index(drop=True)

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Create training, validation, and testing datasets
total_images = len(balanced_data)

# Calculate the sizes
train_size = int(0.8 * total_images)
val_size = int(0.1 * total_images)
test_size = total_images - train_size - val_size  # Remaining images for test

# Debug: Print the sizes of the datasets
print(f'Total images: {total_images}')
print(f'Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}')

# Split the data
train_data = balanced_data.iloc[:train_size]
val_data = balanced_data.iloc[train_size:train_size + val_size]
test_data = balanced_data.iloc[train_size + val_size:]

# Debug: Print the sizes of the datasets after the split
print(f'Training set size: {len(train_data)}')
print(f'Validation set size: {len(val_data)}')
print(f'Testing set size: {len(test_data)}')

# Initialize model, loss function, and optimizer
num_classes = 3  # Number of hot end rate classes
model = SimpleCNN(num_classes=num_classes)  # Assuming SimpleCNN is defined in cnn_models
criterion = nn.CrossEntropyLoss()  # Cross Entropy Loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer


# Balanced batch sampler class
class BalancedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size=15):
        self.data_source = data_source
        self.batch_size = batch_size

        # Group indices by class and limit samples to the minimum  class count
        self.class_indices = {class_id: np.array(indices) for class_id, indices in
                              data_source.groupby('hotend_class').groups.items()}
        self.min_class_samples = min(len(indices) for indices in self.class_indices.values())

        # Debug: Print class indices and their counts
        print(f'Class indices: {self.class_indices}')
        print(f'Minimum class samples: {self.min_class_samples}')

        # Determine number of samples per class per batch
        self.samples_per_class = self.batch_size // len(self.class_indices)

        # Total batches for one full epoch, based on minimum samples
        self.num_batches = self.min_class_samples // self.samples_per_class

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # Create balanced batches
        batch_indices = []
        batch_count = 0

        # Shuffle indices initially for randomness in batches
        for class_id, indices in self.class_indices.items():
            np.random.shuffle(indices)
            self.class_indices[class_id] = indices[:self.min_class_samples]  # Limit to min samples

        while batch_count < self.num_batches:
            batch = []
            for class_id, indices in self.class_indices.items():
                # Take only the needed samples for each batch per class
                batch.extend(indices[:self.samples_per_class])
                self.class_indices[class_id] = np.roll(indices, -self.samples_per_class)

            if len(batch) == self.batch_size:
                np.random.shuffle(batch)  # Shuffle within batch
                batch_indices.extend(batch)
                batch_count += 1

        # Debug: Print current batch indices
        print(f'Batch indices created: {batch_indices}')
        return iter(batch_indices)


# Function to create a DataLoader for the given dataset
def create_dataloader(data_subset, batch_size, transform):
    dataset = CustomDataset(
        dataframe=data_subset,  # Pass the subset DataFrame
        root_dir=ROOT_DIR_PATH,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=BalancedBatchSampler(data_subset, batch_size))

    # Debug: Print the DataLoader's dataset size
    print(f'Dataloader size: {len(dataloader)}')
    return dataloader


# Create DataLoaders for train and validation datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# **Separate DataLoaders for training and validation**
train_loader = create_dataloader(train_data, batch_size=15, transform=transform)
val_loader = create_dataloader(val_data, batch_size=15, transform=transform)

# Debug: Check if DataLoader has zero length
print(f'Train loader size: {len(train_loader)}')
print(f'Validation loader size: {len(val_loader)}')

# Training loop with validation
num_epochs = 5  # Number of epochs for training

for epoch in range(num_epochs):
    print(f'Training Epoch {epoch + 1}/{num_epochs}:')

    # Training phase
    model.train()  # Set the model to training mode
    epoch_loss = 0  # Initialize epoch loss
    for images, labels in tqdm(train_loader, desc='Training', leave=True, dynamic_ncols=True):
        optimizer.zero_grad()  # Reset gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        epoch_loss += loss.item()  # Accumulate loss

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f'Average Training Loss: {avg_epoch_loss:.4f}')

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation', leave=True, dynamic_ncols=True):
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            val_loss += loss.item()  # Accumulate validation loss

            _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
            correct_predictions += (predicted == labels).sum().item()  # Count correct predictions
            total_predictions += labels.size(0)  # Update total predictions

    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions
    print(f'Average Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')

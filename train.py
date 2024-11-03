import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from custom_dataset import CustomDataset  # Import CustomDataset class
from models.cnn_models import SimpleCNN  # Import SimpleCNN model
from tqdm import tqdm  # Import tqdm for progress visualization
import numpy as np

# Define the transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 128x128 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# Paths to the CSV file and the image directory
csv_file = 'data/dataset.csv'  # Path to CSV file
image_folder = 'caxton_dataset/print0'  # Only using images from the print0 directory

# Create dataset
dataset = CustomDataset(csv_file=csv_file, root_dir=image_folder, transform=transform)

# Limit dataset to the first 3084 images
limited_dataset = Subset(dataset, np.arange(3084))  # Create a subset with the first 3084 images

# Shuffle and split indices
indices = np.arange(len(limited_dataset))
np.random.shuffle(indices)  # Shuffle the indices

# Define the split sizes
train_size = int(0.8 * len(limited_dataset))
val_size = len(limited_dataset) - train_size

train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Create subsets
train_subset = Subset(limited_dataset, train_indices)
val_subset = Subset(limited_dataset, val_indices)

# Create DataLoaders for both sets
train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

# Debug: Print the sizes of the datasets
print(f'Training set size: {len(train_subset)}')
print(f'Validation set size: {len(val_subset)}')

# Initialize model, loss function, and optimizer
num_classes = 3  # Number of flow rate classes
model = SimpleCNN(num_classes=num_classes)  # Assuming SimpleCNN is defined in cnn_models
criterion = nn.CrossEntropyLoss()  # Cross Entropy Loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Training loop with validation
num_epochs = 500  # Number of epochs for training

for epoch in range(num_epochs):
    # Training phase
    model.train()  # Set the model to training mode
    epoch_loss = 0  # Initialize epoch loss
    print(f'Training Epoch {epoch + 1}/{num_epochs}:')

    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass

        if len(labels.shape) > 1:
            labels = labels.view(-1)  # Ensure labels are 1D for CrossEntropyLoss
            #print(labels)

        loss = criterion(outputs, labels)  # Compute training loss
        epoch_loss += loss.item()  # Accumulate training loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {epoch_loss / len(train_loader):.4f}')

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0  # Initialize validation loss
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for validation
        for val_images, val_labels in val_loader:
            val_outputs = model(val_images)  # Forward pass on validation data

            if len(val_labels.shape) > 1:
                val_labels = val_labels.view(-1)  # Ensure labels are 1D for CrossEntropyLoss

            val_loss += criterion(val_outputs, val_labels).item()  # Accumulate validation loss

            # Calculate validation accuracy (optional)
            _, predicted = torch.max(val_outputs, 1)  # Get the predicted class
            print("Predicted class:", predicted)
            total += val_labels.size(0)  # Total number of labels
            correct += (predicted == val_labels).sum().item()  # Number of correct predictions

    avg_val_loss = val_loss / len(val_loader)  # Average validation loss
    val_accuracy = correct / total  # Validation accuracy

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')
